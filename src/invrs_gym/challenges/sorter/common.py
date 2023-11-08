"""Defines functions common across sorter challenges.

Copyright (c) 2023 The INVRS-IO authors.
"""

import dataclasses
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from fmmax import basis, fields, fmm, scattering, utils  # type: ignore[import-untyped]
from jax import tree_util
from totypes import types

from invrs_gym.challenges import base

Params = Dict[str, types.BoundedArray | types.Density2DArray]
ThicknessInitializer = Callable[[jax.Array, types.BoundedArray], types.BoundedArray]


DENSITY_METASURFACE = "density_metasurface"
THICKNESS_CAP = "thickness_cap"
THICKNESS_METASURFACE = "thickness_metasurface"
THICKNESS_SPACER = "thickness_spacer"

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
        thickness_metasurface: Thickness of the metasurface layer.
        thickness_spacer: Thickness of the spacer layer.
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

    thickness_cap: float | jnp.ndarray
    thickness_metasurface: float | jnp.ndarray
    thickness_spacer: float | jnp.ndarray

    pitch: float

    offset_monitor_substrate: float


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
            polarizations (i.e. x, y, x + y, x - y).
        reflection: The reflection back to the ambient for the four polarizations.
    """

    wavelength: jnp.ndarray
    polar_angle: jnp.ndarray
    azimuthal_angle: jnp.ndarray
    transmission: jnp.ndarray
    reflection: jnp.ndarray


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
    """Defines a photon extractor component."""

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
        self.grid_shape = (divide_and_round(spec.pitch, sim_params.grid_spacing),) * 2

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
        (
            key_thickness_cap,
            key_thickness_metasurface,
            key_density_metasurface,
            key_thickness_spacer,
        ) = jax.random.split(key, 4)
        params = {
            THICKNESS_CAP: self.thickness_initializer(
                key_thickness_cap,
                types.BoundedArray(
                    self.spec.thickness_cap,
                    lower_bound=0.0,
                    upper_bound=None,
                ),
            ),
            THICKNESS_METASURFACE: self.thickness_initializer(
                key_thickness_metasurface,
                types.BoundedArray(
                    self.spec.thickness_metasurface,
                    lower_bound=0.0,
                    upper_bound=None,
                ),
            ),
            DENSITY_METASURFACE: self.density_initializer(
                key_density_metasurface, self.seed_density
            ),
            THICKNESS_SPACER: self.thickness_initializer(
                key_thickness_spacer,
                types.BoundedArray(
                    self.spec.thickness_spacer,
                    lower_bound=0.0,
                    upper_bound=None,
                ),
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
            thickness_cap=params[THICKNESS_CAP].array,  # type: ignore[arg-type]
            thickness_metasurface=(
                params[THICKNESS_METASURFACE].array  # type: ignore[arg-type]
            ),
            thickness_spacer=params[THICKNESS_SPACER].array,  # type: ignore[arg-type]
        )
        return simulate_sorter(
            density_array=params[DENSITY_METASURFACE].array,  # type: ignore[arg-type]
            spec=spec,
            wavelength=jnp.asarray(wavelength),
            polar_angle=jnp.asarray(polar_angle),
            azimuthal_angle=jnp.asarray(azimuthal_angle),
            expansion=expansion,
            formulation=self.sim_params.formulation,
        )


def divide_and_round(a: float, b: float) -> int:
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
    density_array: jnp.ndarray,
    spec: SorterSpec,
    wavelength: jnp.ndarray,
    polar_angle: jnp.ndarray,
    azimuthal_angle: jnp.ndarray,
    expansion: basis.Expansion,
    formulation: fmm.Formulation,
) -> Tuple[SorterResponse, base.AuxDict]:
    """Simulates a sorter component, e.g. a wavelength or polarization sorter.

    This code is adapted from the fmmax.examples.sorter script.

    The sorter consists of a metasurface layer situated above a quad of pixels.
    Above the metasurface is a cap, and it is separated from the substrate by a
    spacer layer, as illustrated below.

                               __________________________
                              /                         /|
              ambient        /                         //|
                            /                         ///|
                           /_________________________/// |
                  cap --> |_________________________|// /|
          metasurface --> |_________________________|/ / |
               spacer --> |                         | /| |
                          |_________________________|/ |/
                          |            |            |  /
            substrate --> |     q1     |     q2     | /
                          |____________|____________|/

    The sorter is illuminated by plane waves incident from the ambient, and its
    response consists of the power captured by substrate monitors within each of
    the quadrants, as well as the power reflected back toward the ambient.

    The sorter is always simulated with four incident polarizations, aligned with
    the x, y, x + y, and x - y directions, respectively.

    Args:
        density_array: Defines the pattern of the metasurface layer.
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

    primitive_lattice_vectors = basis.LatticeVectors(
        u=spec.pitch * basis.X,
        v=spec.pitch * basis.Y,
    )
    in_plane_wavevector = basis.plane_wave_in_plane_wavevector(
        wavelength=wavelength,
        polar_angle=polar_angle,
        azimuthal_angle=azimuthal_angle,
        permittivity=spec.permittivity_ambient,
    )

    permittivities = [
        jnp.full((1, 1), spec.permittivity_ambient),
        jnp.full((1, 1), spec.permittivity_cap),
        utils.interpolate_permittivity(
            permittivity_solid=jnp.asarray(spec.permittivity_metasurface_solid),
            permittivity_void=jnp.asarray(spec.permittivity_metasurface_void),
            density=density_array,
        ),
        jnp.full((1, 1), spec.permittivity_spacer),
        jnp.full((1, 1), spec.permittivity_substrate),
    ]

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

    layer_thicknesses = [
        jnp.zeros(()),  # Ambient
        jnp.asarray(spec.thickness_cap),
        jnp.asarray(spec.thickness_metasurface),
        jnp.asarray(spec.thickness_spacer),
        jnp.asarray(spec.offset_monitor_substrate),  # Substrate
    ]

    s_matrix = scattering.stack_s_matrix(layer_solve_results, layer_thicknesses)

    n = expansion.num_terms
    assert tuple(expansion.basis_coefficients[0, :]) == (0, 0)
    assert expansion.basis_coefficients.shape[0] == n

    # Generate wave amplitudes for forward-going waves in the ambient with four
    # different polarizations: x, y, (x + y) / sqrt(2), and (x - y) / sqrt(2).
    fwd_amplitude_0_start = jnp.zeros((2 * n, 4), dtype=complex)
    fwd_amplitude_0_start = fwd_amplitude_0_start.at[0, 0].set(1)
    fwd_amplitude_0_start = fwd_amplitude_0_start.at[0, 1].set(1 / jnp.sqrt(2))
    fwd_amplitude_0_start = fwd_amplitude_0_start.at[n, 1].set(1 / jnp.sqrt(2))
    fwd_amplitude_0_start = fwd_amplitude_0_start.at[0, 2].set(1 / jnp.sqrt(2))
    fwd_amplitude_0_start = fwd_amplitude_0_start.at[n, 2].set(-1 / jnp.sqrt(2))
    fwd_amplitude_0_start = fwd_amplitude_0_start.at[n, 3].set(1)

    # Compute the backward-going wave amplitudes at the start of the ambient. Since
    # the ambient has zero thickness, the fields at the start and end are colocated.
    bwd_amplitude_0_end = s_matrix.s21 @ fwd_amplitude_0_start
    sz_fwd_0, sz_bwd_0 = fields.amplitude_poynting_flux(
        fwd_amplitude_0_start, bwd_amplitude_0_end, layer_solve_results[0]
    )
    sz_fwd_ambient_sum = jnp.sum(jnp.abs(sz_fwd_0), axis=-2)
    sz_bwd_ambient_sum = jnp.sum(jnp.abs(sz_bwd_0), axis=-2)
    reflection = jnp.abs(sz_bwd_ambient_sum) / jnp.abs(sz_fwd_ambient_sum)

    # Compute the forward-going and backward-going wave amplitudes in the substrate,
    # a distance `spec.offset_monitor_substrate` from the start of the substrate.
    fwd_amplitude_N_start = s_matrix.s11 @ fwd_amplitude_0_start
    fwd_amplitude_N_offset, bwd_amplitude_N_offset = fields.colocate_amplitudes(
        fwd_amplitude_N_start,
        jnp.zeros_like(fwd_amplitude_N_start),
        z_offset=layer_thicknesses[-1],
        layer_solve_result=layer_solve_results[-1],
        layer_thickness=layer_thicknesses[-1],
    )

    # Compute electric and magnetic fields at the monitor plane in their Fourier
    # representation, and then on the real-space grid.
    ef, hf = fields.fields_from_wave_amplitudes(
        fwd_amplitude_N_offset,
        bwd_amplitude_N_offset,
        layer_solve_result=layer_solve_results[-1],
    )
    grid_shape = density_array.shape[-2:]
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

    # Create masks selecting the four quadrants.
    mask = jnp.zeros(grid_shape + (1, 4))
    xdim = grid_shape[0] // 2
    ydim = grid_shape[1] // 2
    mask = mask.at[:xdim, :ydim, 0, 0].set(1)
    mask = mask.at[:xdim, ydim:, 0, 1].set(1)
    mask = mask.at[xdim:, :ydim, 0, 2].set(1)
    mask = mask.at[xdim:, ydim:, 0, 3].set(1)

    # Use the mask to compute the time average Poynting flux into each quadrant. The
    # trailing two dimensions have shape `(4, 4)`; index `(i, j)` corresponds to
    # power for the `i` excitation (i.e. polarization) in the `j` quadrant.
    quadrant_sz = jnp.mean(mask * sz[..., jnp.newaxis], axis=(-4, -3))
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
        "efield": (ex, ey, ez),
        "hfield": (hx, hy, hz),
        "coordinates": (x, y),
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
