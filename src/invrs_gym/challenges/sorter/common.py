"""Defines functions common across sorter challenges.

Copyright (c) 2023 The INVRS-IO authors.
"""

import dataclasses
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from fmmax import basis, fields, fmm, scattering  # type: ignore[import-untyped]
from jax import tree_util
from totypes import json_utils, types

from invrs_gym import utils
from invrs_gym.challenges import base

Params = Dict[str, types.BoundedArray | types.Density2DArray]
ThicknessInitializer = Callable[[jax.Array, types.BoundedArray], types.BoundedArray]


DENSITY_METASURFACE = "density_metasurface"
THICKNESS_CAP = "thickness_cap"
THICKNESS_METASURFACE = "thickness_metasurface"
THICKNESS_SPACER = "thickness_spacer"

EFIELD = "efield"
HFIELD = "hfield"
POYNTING_FLUX_Z = "poynting_flux_z"
COORDINATES = "coordinates"

DENSITY_LOWER_BOUND = 0.0
DENSITY_UPPER_BOUND = 1.0


@dataclasses.dataclass
class SorterSpec:
    """Defines the physical specification of a sorter.

    Attributes:
        permittivity_ambient: Permittivity of the ambient material.
        permittivity_cap: Permittivity of the cap layer.
        permittivity_metasurface_solid: Permittivity of solid metasurface regions.
        permittivity_metasurface_void: Permittivity of solid metasurface regions.
        permittivity_spacer: Permittivity of the spacer layer.
        permittivity_substrate: Permittivity of the substrate.
        thickness_cap: Thickness of the cap layer.
        thickness_metasurface: Thicknesses of the metasurface layers.
        thickness_spacer: Thicknesses of the spacer layers. The final spacer layer is
            between the substrate and the final metasurface; earlier spacers lie
            between adjacent metasurfaces.
        pitch: The size of the unit cell along the x and y directions.
        offset_monitor_substrate: Offset of the monitor plane from the interface
            between spacer and substrate.
    """

    permittivity_ambient: complex
    permittivity_cap: complex
    permittivity_metasurface_solid: complex
    permittivity_metasurface_void: complex
    permittivity_spacer: complex
    permittivity_substrate: complex

    thickness_cap: types.BoundedArray
    thickness_metasurface: Tuple[types.BoundedArray, ...]
    thickness_spacer: Tuple[types.BoundedArray, ...]

    pitch: float

    offset_monitor_substrate: float

    def __post_init__(self):
        if len(self.thickness_metasurface) != len(self.thickness_spacer):
            raise ValueError(
                f"Length of `thickness_metasurface` and `thickness_spacer` must match "
                f"but got {self.thickness_metasurface} and {self.thickness_spacer}."
            )


@dataclasses.dataclass
class SorterSimParams:
    """Parameters that configure the simulation of a sorter.

    Attributes:
        grid_spacing: The spacing of points on the real-space grid.
        wavelength: The wavelength of the excitation.
        polar_angle: The polar angle of the excitation.
        azimuthal_angle: The azimuthal angle of the excitation.
        formulation: The FMM formulation to be used.
        approximate_num_terms: Defines the number of terms in the Fourier expansion.
        truncation: Determines how the Fourier basis is truncated.
    """

    grid_spacing: float
    wavelength: float | jnp.ndarray
    polar_angle: float | jnp.ndarray
    azimuthal_angle: float | jnp.ndarray
    formulation: fmm.Formulation
    approximate_num_terms: int
    truncation: basis.Truncation


@dataclasses.dataclass
class SorterResponse:
    """Contains the response of the sorter.

    Attributes:
        wavelength: The wavelength for the sorter response.
        polar_angle: The polar angle for the sorter response.
        azimuthal_angle: The azimuthal angle for the sorter response.
        transmission: The transmission into the four quadrants for the four
            polarizations (i.e. x, x + y, x - y, y).
        reflection: The reflection back to the ambient for the four polarizations.
    """

    wavelength: jnp.ndarray
    polar_angle: jnp.ndarray
    azimuthal_angle: jnp.ndarray
    transmission: jnp.ndarray
    reflection: jnp.ndarray


json_utils.register_custom_type(SorterResponse)

tree_util.register_pytree_node(
    SorterResponse,
    lambda r: (
        (
            r.wavelength,
            r.polar_angle,
            r.azimuthal_angle,
            r.transmission,
            r.reflection,
        ),
        None,
    ),
    lambda _, children: SorterResponse(*children),
)


class SorterComponent(base.Component):
    """Defines a sorter component."""

    def __init__(
        self,
        spec: SorterSpec,
        sim_params: SorterSimParams,
        thickness_initializer: ThicknessInitializer,
        density_initializer: base.DensityInitializer,
        **seed_density_kwargs: Any,
    ) -> None:
        """Initializes the sorter component.

        Args:
            spec: Defines the physical specification of the sorter.
            sim_params: Defines simulation parameters for the sorter.
            thickness_initializer: Callable which returns the initial thickness for
                a layer from a random key and a bounded array with value equal the
                thickness from `spec`.
            density_initializer: Callable which generates the initial density from
                a random key and the seed density.
            **seed_density_kwargs: Keyword arguments which set the attributes of
                the seed density used to generate the inital parameters.
        """

        self.spec = spec
        self.sim_params = sim_params
        self.thickness_initializer = thickness_initializer
        self.density_initializer = density_initializer
        self.grid_shape = (_divide_and_round(spec.pitch, sim_params.grid_spacing),) * 2

        self.seed_density = seed_density(
            grid_shape=self.grid_shape, **seed_density_kwargs
        )
        self.expansion = basis.generate_expansion(
            primitive_lattice_vectors=basis.LatticeVectors(
                u=self.spec.pitch * basis.X,
                v=self.spec.pitch * basis.Y,
            ),
            approximate_num_terms=self.sim_params.approximate_num_terms,
            truncation=self.sim_params.truncation,
        )

    def init(self, key: jax.Array) -> Params:
        """Return the initial parameters for the sorter component."""
        keys = jax.random.split(key, 1 + 3 * len(self.spec.thickness_metasurface))
        keys_iter = iter(keys)
        params = {
            THICKNESS_CAP: self.thickness_initializer(
                next(keys_iter), self.spec.thickness_cap
            ),
            THICKNESS_METASURFACE: tuple(
                self.thickness_initializer(next(keys_iter), t)
                for t in self.spec.thickness_metasurface
            ),
            DENSITY_METASURFACE: tuple(
                self.density_initializer(next(keys_iter), self.seed_density)
                for _ in self.spec.thickness_metasurface
            ),
            THICKNESS_SPACER: tuple(
                self.thickness_initializer(next(keys_iter), t)
                for t in self.spec.thickness_spacer
            ),
        }
        # Ensure that there are no weak types in the initial parameters.
        return tree_util.tree_map(
            lambda x: jnp.asarray(x, jnp.asarray(x).dtype), params
        )

    def response(
        self,
        params: Params,
        *,
        wavelength: Optional[float | jnp.ndarray] = None,
        polar_angle: Optional[float | jnp.ndarray] = None,
        azimuthal_angle: Optional[float | jnp.ndarray] = None,
        expansion: Optional[basis.Expansion] = None,
    ) -> Tuple[SorterResponse, base.AuxDict]:
        """Computes the response of the sorter.

        Args:
            params: The parameters defining the sorter, with structure matching that
                of the parameters returned by the `init` method.
            wavelength: Optional wavelength to override the default in `sim_params`.
            polar_angle: Optional polar angle to override the default.
            azimuthal_angle: Optional azimuthal angle to override the default.
            expansion: Optional expansion to override the default `expansion`.

        Returns:
            The `(response, aux)` tuple.
        """
        if expansion is None:
            expansion = self.expansion
        if wavelength is None:
            wavelength = self.sim_params.wavelength
        if polar_angle is None:
            polar_angle = self.sim_params.polar_angle
        if azimuthal_angle is None:
            azimuthal_angle = self.sim_params.azimuthal_angle

        spec = dataclasses.replace(
            self.spec,
            thickness_cap=params[THICKNESS_CAP],  # type: ignore[arg-type]
            thickness_metasurface=(
                params[THICKNESS_METASURFACE]  # type: ignore[arg-type]
            ),
            thickness_spacer=params[THICKNESS_SPACER],  # type: ignore[arg-type]
        )
        return simulate_sorter(
            densities=params[DENSITY_METASURFACE],  # type: ignore[arg-type]
            spec=spec,
            wavelength=jnp.asarray(wavelength),
            polar_angle=jnp.asarray(polar_angle),
            azimuthal_angle=jnp.asarray(azimuthal_angle),
            expansion=expansion,
            formulation=self.sim_params.formulation,
        )


def _divide_and_round(a: float, b: float) -> int:
    """Checks that `a` is nearly evenly divisible by `b`, and returns `a / b`."""
    result = int(jnp.around(a / b))
    if not jnp.isclose(a / b, result):
        raise ValueError(
            f"`a` must be nearly evenly divisible by `b` spacing, but got `a` "
            f"{a} with `b` {b}."
        )
    return result


def seed_density(grid_shape: Tuple[int, int], **kwargs: Any) -> types.Density2DArray:
    """Return the seed density for a sorter component.

    Args:
        grid_shape: The shape of the grid on which the density is defined.
        kwargs: keyword arguments specifying additional properties of the seed
            density, e.g. symmetries.

    Returns:
        The seed density.
    """

    # Check kwargs that are required for a sorter component.
    invalid_kwargs = ("array", "lower_bound", "upper_bound", "periodic")
    if any(k in invalid_kwargs for k in kwargs):
        raise ValueError(
            f"Attributes were specified which confict with automatically-extracted "
            f"attributes. Got {kwargs.keys()} when {invalid_kwargs} are automatically "
            f"extracted."
        )

    mid_density_value = (DENSITY_LOWER_BOUND + DENSITY_UPPER_BOUND) / 2
    return types.Density2DArray(
        array=jnp.full(grid_shape, mid_density_value),
        lower_bound=DENSITY_LOWER_BOUND,
        upper_bound=DENSITY_UPPER_BOUND,
        periodic=(True, True),
        **kwargs,
    )


def simulate_sorter(
    densities: Tuple[types.Density2DArray, ...],
    spec: SorterSpec,
    wavelength: jnp.ndarray,
    polar_angle: jnp.ndarray,
    azimuthal_angle: jnp.ndarray,
    expansion: basis.Expansion,
    formulation: fmm.Formulation,
) -> Tuple[SorterResponse, base.AuxDict]:
    """Simulates a sorter component, e.g. a wavelength or polarization sorter.

    This code is adapted from the fmmax.examples.sorter script.

    The sorter consists of a metasurface layer situated above a quad of pixels,
    with each pixel being square in shape. Above the metasurface is a cap, and
    it is separated from the substrate by a spacer layer as illustrated below.

                               __________________________
                              /                         /|
              ambient        /                         //|
                            /                         ///|
                           /_________________________/// |
                  cap --> |_________________________|// /|
          metasurface --> |_________________________|/ / |
               spacer --> |                         | /| |     q0 | q1
                          |_________________________|/ |/     ---------
                          |            |            |  /       q2 | q3
            substrate --> |            |            | /
                          |____________|____________|/

    The sorter is illuminated by plane waves incident from the ambient, and its
    response consists of the power captured by substrate monitors within each of
    the quadrants, as well as the power reflected back toward the ambient.

    The sorter is always simulated with four incident polarizations, aligned with
    the x, x + y, x - y, and y directions, respectively.

    Args:
        densities: Defines the patterns of each metasurface layer.
        spec: Defines the physical specification of the sorter.
        wavelength: The wavelength of the excitation.
        polar_angle: The polar angle of the excitation.
        azimuthal_angle: The azimuthal angle of the excitation.
        expansion: Defines the Fourier expansion for the calculation.
        formulation: Defines the FMM formulation to be used.

    Returns:
        The `SorterResponse`, and an auxilliary dictionary containing the fields
        at the monitor plane.
    """
    assert len(spec.thickness_metasurface) == len(densities)
    density_arrays = [
        utils.transforms.rescaled_density_array(
            density,
            lower_bound=DENSITY_LOWER_BOUND,
            upper_bound=DENSITY_UPPER_BOUND,
        )
        for density in densities
    ]
    primitive_lattice_vectors = basis.LatticeVectors(
        u=spec.pitch * basis.X,
        v=spec.pitch * basis.Y,
    )
    in_plane_wavevector = basis.plane_wave_in_plane_wavevector(
        wavelength=wavelength,
        polar_angle=polar_angle,
        azimuthal_angle=azimuthal_angle,
        permittivity=jnp.asarray(spec.permittivity_ambient),
    )

    permittivities = (
        [
            jnp.full((1, 1), spec.permittivity_ambient),  # Ambient
            jnp.full((1, 1), spec.permittivity_cap),  # Cap
        ]
        + _alternate(  # Alternating metasurface and spacer layers.
            [
                utils.transforms.interpolate_permittivity(
                    permittivity_solid=jnp.asarray(spec.permittivity_metasurface_solid),
                    permittivity_void=jnp.asarray(spec.permittivity_metasurface_void),
                    density=density_array,
                )
                for density_array in density_arrays
            ],
            [jnp.full((1, 1), spec.permittivity_spacer) for _ in density_arrays],
        )
        + [jnp.full((1, 1), spec.permittivity_substrate)]  # Substrate.
    )

    layer_thicknesses = (
        [
            jnp.zeros(()),  # Ambient
            jnp.asarray(spec.thickness_cap.array),  # Cap
        ]
        + _alternate(  # Alternating metasurface and spacer layers
            [jnp.asarray(tm.array) for tm in spec.thickness_metasurface],
            [jnp.asarray(tm.array) for tm in spec.thickness_spacer],
        )
        + [jnp.asarray(spec.offset_monitor_substrate)]  # Substrate
    )

    assert len(permittivities) == len(layer_thicknesses)
    assert permittivities[0].shape == (1, 1)
    assert all([p.shape == (1, 1) for p in permittivities[1::2]])

    layer_solve_results = [
        fmm.eigensolve_isotropic_media(
            wavelength=wavelength,
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=primitive_lattice_vectors,
            permittivity=p,
            expansion=expansion,
            formulation=formulation,
        )
        for p in permittivities
    ]

    s_matrix = scattering.stack_s_matrix(layer_solve_results, layer_thicknesses)

    n = expansion.num_terms
    assert tuple(expansion.basis_coefficients[0, :]) == (0, 0)
    assert expansion.basis_coefficients.shape[0] == n

    # Generate wave amplitudes for forward-going waves at the start of the in
    # the ambient with the appropriate polarization for the four different quadrants.
    #
    # The quadrants are numbered as follows:      0  |  1
    #                                           -----------
    #                                             2  |  3
    #
    # with quadrant 0 targeting x polarization, quadrant 3 targeting y polarization,
    # quadrant 1 targeting (x + y) / sqrt(2), and finally quadrant 2 targeting
    # (x - y) / sqrt(2).
    fwd_ambient_start = jnp.zeros((2 * n, 4), dtype=complex)
    fwd_ambient_start = fwd_ambient_start.at[0, 0].set(1)
    fwd_ambient_start = fwd_ambient_start.at[0, 1].set(1 / jnp.sqrt(2))
    fwd_ambient_start = fwd_ambient_start.at[n, 1].set(1 / jnp.sqrt(2))
    fwd_ambient_start = fwd_ambient_start.at[0, 2].set(1 / jnp.sqrt(2))
    fwd_ambient_start = fwd_ambient_start.at[n, 2].set(-1 / jnp.sqrt(2))
    fwd_ambient_start = fwd_ambient_start.at[n, 3].set(1)

    # Compute the backward-going wave amplitudes at the start of the ambient. Since
    # the ambient has zero thickness, the fields at the start and end are colocated.
    bwd_ambient_end = s_matrix.s21 @ fwd_ambient_start
    sz_fwd_ambient, sz_bwd_ambient = fields.amplitude_poynting_flux(
        forward_amplitude=fwd_ambient_start,
        backward_amplitude=bwd_ambient_end,
        layer_solve_result=layer_solve_results[0],
    )
    sz_fwd_ambient_sum = jnp.sum(jnp.abs(sz_fwd_ambient), axis=-2)
    sz_bwd_ambient_sum = jnp.sum(jnp.abs(sz_bwd_ambient), axis=-2)
    reflection = jnp.abs(sz_bwd_ambient_sum) / jnp.abs(sz_fwd_ambient_sum)

    # Compute the forward-going and backward-going wave amplitudes in the substrate,
    # a distance `spec.offset_monitor_substrate` from the start of the substrate.
    fwd_substrate_start = s_matrix.s11 @ fwd_ambient_start
    fwd_substrate_offset, bwd_substrate_offset = fields.colocate_amplitudes(
        fwd_substrate_start,
        jnp.zeros_like(fwd_substrate_start),
        z_offset=layer_thicknesses[-1],
        layer_solve_result=layer_solve_results[-1],
        layer_thickness=layer_thicknesses[-1],
    )

    # Compute electric and magnetic fields at the monitor plane in their Fourier
    # representation, and then on the real-space grid.
    ef, hf = fields.fields_from_wave_amplitudes(
        forward_amplitude=fwd_substrate_offset,
        backward_amplitude=bwd_substrate_offset,
        layer_solve_result=layer_solve_results[-1],
    )
    grid_shape: Tuple[int, int]
    grid_shape = density_arrays[0].shape[-2:]  # type: ignore[assignment]
    (ex, ey, ez), (hx, hy, hz), (x, y) = fields.fields_on_grid(
        electric_field=ef,
        magnetic_field=hf,
        layer_solve_result=layer_solve_results[-1],
        shape=grid_shape,
        num_unit_cells=(1, 1),
    )
    batch_shape = layer_solve_results[0].batch_shape
    assert ex.shape == batch_shape + grid_shape + (4,)

    # Compute the Poynting flux on the real-space grid at the monitor plane.
    sz = _time_average_z_poynting_flux((ex, ey, ez), (hx, hy, hz))
    assert sz.shape == batch_shape + grid_shape + (4,)

    # Create masks selecting the four quadrants, and the circular target regions.
    quadrant_mask = _quadrant_mask(grid_shape)
    assert quadrant_mask.shape == grid_shape + (1, 4)

    # Use the mask to compute the time average Poynting flux into each quadrant. The
    # trailing two dimensions have shape `(4, 4)`; index `(i, j)` corresponds to
    # power for the `i` excitation (i.e. polarization) in the `j` quadrant.
    quadrant_sz = jnp.mean(quadrant_mask * sz[..., jnp.newaxis], axis=(-4, -3))
    quadrant_sz /= sz_fwd_ambient_sum[..., jnp.newaxis]

    assert quadrant_sz.shape == batch_shape + (4, 4)

    response = SorterResponse(
        wavelength=wavelength,
        polar_angle=polar_angle,
        azimuthal_angle=azimuthal_angle,
        transmission=quadrant_sz,
        reflection=reflection,
    )

    aux = {
        EFIELD: (ex, ey, ez),
        HFIELD: (hx, hy, hz),
        POYNTING_FLUX_Z: sz,
        COORDINATES: (x, y),
    }

    return response, aux


def _time_average_z_poynting_flux(
    electric_fields: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    magnetic_fields: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> jnp.ndarray:
    """Computes the time-average z-directed Poynting flux, given the physical fields."""
    ex, ey, _ = electric_fields
    hx, hy, _ = magnetic_fields
    return jnp.real(ex * jnp.conj(hy) - ey * jnp.conj(hx))


def _quadrant_mask(grid_shape: Tuple[int, int]) -> jnp.ndarray:
    """Return masks that select the four quadrants of a sorter.

    The quadrants are numbered as follows:     0  |  1
                                             -----------
                                               2  |  3

    Args:
        grid_shape: The shape of the grid for which to return the mask.

    Returns:
        The quadrant mask, with shape `grid_shape + (1, 4)`.
    """
    quadrant_mask = jnp.zeros(grid_shape + (1, 4))
    xdim = grid_shape[0] // 2
    ydim = grid_shape[1] // 2
    quadrant_mask = quadrant_mask.at[:xdim, :ydim, 0, 0].set(1)  # nw
    quadrant_mask = quadrant_mask.at[:xdim, ydim:, 0, 1].set(1)  # ne
    quadrant_mask = quadrant_mask.at[xdim:, :ydim, 0, 2].set(1)  # sw
    quadrant_mask = quadrant_mask.at[xdim:, ydim:, 0, 3].set(1)  # se
    return quadrant_mask


def _alternate(*iterables: Sequence[jnp.ndarray]) -> List[jnp.ndarray]:
    """Alterately takes values from each of the iterables."""
    return sum([list(group) for group in zip(*iterables, strict=True)], [])
