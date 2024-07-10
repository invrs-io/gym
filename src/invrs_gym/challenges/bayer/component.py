"""Defines the bayer color sorter component.

Copyright (c) 2023 The INVRS-IO authors.
"""

import dataclasses
import functools
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as onp
from fmmax import basis, fields, fmm, scattering  # type: ignore[import-untyped]
from invrs_gym import utils
from invrs_gym.challenges import base
from jax import tree_util
from totypes import json_utils, types

from invrs_gym.utils import materials

NDArray = onp.ndarray[Any, Any]


Params = Dict[str, types.BoundedArray | types.Density2DArray]
ThicknessInitializer = Callable[[jax.Array, types.BoundedArray], types.BoundedArray]


THICKNESS_METASURFACE = "thickness_metasurface"
DENSITY_METASURFACE = "density_metasurface"
OFFSET_MONITOR_SUBSTRATE = "offset_monitor_substrate"

EFIELD_XY = "efield_xy"
HFIELD_XY = "hfield_xy"
POYNTING_FLUX_XY = "poynting_flux_xy"
COORDINATES_XY = "coordinates_xy"
TRANSMITTED_POWER = "transmitted_power"

EFIELD_XZ = "efield_xz"
HFIELD_XZ = "hfield_xz"
POYNTING_FLUX_XZ = "poynting_flux_xz"
COORDINATES_XZ = "coordinates_xz"

DENSITY_LOWER_BOUND = 0.0
DENSITY_UPPER_BOUND = 1.0


@dataclasses.dataclass
class BayerSpec:
    """Defines the physical specification of a bayer color sorter.

    Attributes:
        material_ambient: The ambient material.
        material_metasurface_solid: The material comprising metasurface solid regions.
        material_metasurface_void: The material comprising metasurface void regions.
        material_substrate: The substrate material.
        thickness_ambient: The thickness of the ambient.
        thickness_metasurface: Thicknesses of the metasurface layers.
        thickness_substrate: The thickness of the substrate.
        offset_monitor_substrate: Offset of the monitor plane from the interface
            between spacer and substrate.
        pixel_size: The size of a pixel in the bayer array.
        grid_spacing: The spacing of the grid on which grating permittivity is defined.
    """

    material_ambient: str
    material_metasurface_solid: str
    material_metasurface_void: str
    material_substrate: str

    thickness_ambient: float
    thickness_metasurface: types.BoundedArray
    thickness_substrate: float
    offset_monitor_substrate: types.BoundedArray

    pixel_size: float
    grid_spacing: float

    @property
    def period_x(self) -> float:
        """Returns the size of the unit cell in the `x` direction."""
        return 2 * self.pixel_size

    @property
    def period_y(self) -> float:
        """Returns the size of the unit cell in the `y` direction."""
        return 2 * self.pixel_size

    @property
    def grid_shape(self) -> Tuple[int, int]:
        """Return the shape of the grid implied by `grid_spacing`."""
        return (
            int(jnp.ceil(self.period_x / self.grid_spacing)),
            int(jnp.ceil(self.period_y / self.grid_spacing)),
        )


@dataclasses.dataclass
class BayerSimParams:
    """Parameters that configure the simulation of a bayer color sorter.

    Attributes:
        wavelength: The wavelength of the excitation.
        polar_angle: The polar angle of the excitation.
        azimuthal_angle: The azimuthal angle of the excitation.
        formulation: The FMM formulation to be used.
        approximate_num_terms: Defines the number of terms in the Fourier expansion.
        truncation: Determines how the Fourier basis is truncated.
    """

    wavelength: float | jnp.ndarray
    polar_angle: float | jnp.ndarray
    azimuthal_angle: float | jnp.ndarray
    formulation: fmm.Formulation
    approximate_num_terms: int
    truncation: basis.Truncation


@dataclasses.dataclass
class BayerResponse:
    """Contains the response of the bayer color sorter.

    Attributes:
        wavelength: The wavelength for the sorter response.
        polar_angle: The polar angle for the sorter response.
        azimuthal_angle: The azimuthal angle for the sorter response.
        transmission: The transmission into the four pixels for each polarization.
        reflection: The reflection back to the ambient for the each polarizations.
    """

    wavelength: jnp.ndarray
    polar_angle: jnp.ndarray
    azimuthal_angle: jnp.ndarray
    transmission: jnp.ndarray
    reflection: jnp.ndarray


json_utils.register_custom_type(BayerResponse)

tree_util.register_pytree_node(
    BayerResponse,
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
    lambda _, children: BayerResponse(*children),
)


class BayerComponent(base.Component):
    """Defines a bayer color sorter component."""

    def __init__(
        self,
        spec: BayerSpec,
        sim_params: BayerSimParams,
        thickness_initializer: ThicknessInitializer,
        density_initializer: base.DensityInitializer,
        **seed_density_kwargs: Any,
    ) -> None:
        """Initializes the bayer color sorter component.

        Args:
            spec: Defines the physical specification of the bayer color sorter.
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

        self.seed_density = _seed_density(
            grid_shape=self.spec.grid_shape, **seed_density_kwargs
        )
        self.expansion = basis.generate_expansion(
            primitive_lattice_vectors=basis.LatticeVectors(
                u=self.spec.period_x * basis.X,
                v=self.spec.period_y * basis.Y,
            ),
            approximate_num_terms=self.sim_params.approximate_num_terms,
            truncation=self.sim_params.truncation,
        )

    def init(self, key: jax.Array) -> Params:
        """Return the initial parameters for the bayer color sorter component."""
        key_thickness, key_density, key_offset = jax.random.split(key, num=3)
        params = {
            THICKNESS_METASURFACE: self.thickness_initializer(
                key_thickness, self.spec.thickness_metasurface
            ),
            DENSITY_METASURFACE: self.density_initializer(
                key_density, self.seed_density
            ),
            OFFSET_MONITOR_SUBSTRATE: self.thickness_initializer(
                key_offset, self.spec.offset_monitor_substrate
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
        compute_fields: bool = False,
    ) -> Tuple[BayerResponse, base.AuxDict]:
        """Computes the response of the sorter.

        Args:
            params: The parameters defining the sorter, with structure matching that
                of the parameters returned by the `init` method.
            wavelength: Optional wavelength to override the default in `sim_params`.
            polar_angle: Optional polar angle to override the default.
            azimuthal_angle: Optional azimuthal angle to override the default.
            expansion: Optional expansion to override the default `expansion`.
            compute_fields: If `True`, computes fields in an xz cross section.

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
            thickness_metasurface=(
                params[THICKNESS_METASURFACE]  # type: ignore[arg-type]
            ),
            offset_monitor_substrate=(
                params[OFFSET_MONITOR_SUBSTRATE]  # type: ignore[arg-type]
            ),
        )
        return simulate_color_sorter(
            density=params[DENSITY_METASURFACE],  # type: ignore[arg-type]
            spec=spec,
            wavelength=jnp.asarray(wavelength),
            polar_angle=jnp.asarray(polar_angle),
            azimuthal_angle=jnp.asarray(azimuthal_angle),
            expansion=expansion,
            formulation=self.sim_params.formulation,
            compute_fields=compute_fields,
        )


def _seed_density(grid_shape: Tuple[int, int], **kwargs: Any) -> types.Density2DArray:
    """Return the seed density for a bayer sorter component.

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


def simulate_color_sorter(
    density: types.Density2DArray,
    spec: BayerSpec,
    wavelength: jnp.ndarray,
    polar_angle: jnp.ndarray,
    azimuthal_angle: jnp.ndarray,
    expansion: basis.Expansion,
    formulation: fmm.Formulation,
    compute_fields: bool,
) -> Tuple[BayerResponse, base.AuxDict]:
    """Simulates a bayer color sorter.

    Args:
        density: Defines the pattern of the metasurface layer.
        spec: Defines the physical specification of the sorter.
        wavelength: The wavelength of the excitation.
        polar_angle: The polar angle of the excitation.
        azimuthal_angle: The azimuthal angle of the excitation.
        expansion: Defines the Fourier expansion for the calculation.
        formulation: Defines the FMM formulation to be used.
        compute_fields: If `True`, fields in an xz cross section are computed and
            and included in the `aux` return variable.

    Returns:
        The `BayerResponse`, and a dictionary containing the auxilliary quantities.
    """
    primitive_lattice_vectors = basis.LatticeVectors(
        u=spec.period_x * basis.X,
        v=spec.period_y * basis.Y,
    )
    in_plane_wavevector = basis.plane_wave_in_plane_wavevector(
        wavelength=wavelength,
        polar_angle=polar_angle,
        azimuthal_angle=azimuthal_angle,
        permittivity=materials.permittivity(spec.material_ambient, wavelength),
    )

    # -------------------------------------------------------------------------
    # Layer permittivities, thicknesses, and eigensolve results.
    # -------------------------------------------------------------------------

    def permittivity_fn(material_name):
        shape = (wavelength.size, 1, 1)
        return materials.permittivity(material_name, wavelength).reshape(shape)

    eigensolve_fn = functools.partial(
        fmm.eigensolve_isotropic_media,
        wavelength=wavelength,
        in_plane_wavevector=in_plane_wavevector,
        primitive_lattice_vectors=primitive_lattice_vectors,
        expansion=expansion,
        formulation=formulation,
    )

    # Eigensolve for the uniform ambient and substrate layers.
    solve_result_ambient = eigensolve_fn(
        permittivity=permittivity_fn(spec.material_ambient)
    )
    solve_result_substrate = eigensolve_fn(
        permittivity=permittivity_fn(spec.material_substrate)
    )

    # Eigensolve for the patterned metasurface layer.
    density_array = utils.transforms.rescaled_density_array(
        density=density,
        lower_bound=DENSITY_LOWER_BOUND,
        upper_bound=DENSITY_UPPER_BOUND,
    )
    permittivity_metasurface = utils.transforms.interpolate_permittivity(
        permittivity_solid=permittivity_fn(spec.material_metasurface_solid),
        permittivity_void=permittivity_fn(spec.material_metasurface_void),
        density=density_array,
    )
    solve_result_metasurface = eigensolve_fn(permittivity=permittivity_metasurface)

    # Assemble the list of all layer solve results.
    layer_solve_results = [
        solve_result_ambient,
        solve_result_metasurface,
        solve_result_substrate,
    ]
    layer_thicknesses = [
        jnp.asarray(spec.thickness_ambient),
        jnp.asarray(spec.thickness_metasurface.array),
        jnp.asarray(spec.thickness_substrate),
    ]

    # -------------------------------------------------------------------------
    # Scattering matrix assembly.
    # -------------------------------------------------------------------------

    if compute_fields:
        # If the field calculation is desired, compute the interior scattering
        # matrices. For each layer in the stack, the interior scattering matrices
        # consist of a pair of matrices, one for the substack below the layer, and
        # one for the substack above the layer.
        s_matrices_interior = scattering.stack_s_matrices_interior(
            layer_solve_results=layer_solve_results,
            layer_thicknesses=layer_thicknesses,
        )
        s_matrix = s_matrices_interior[-1][0]
    else:
        s_matrix = scattering.stack_s_matrix(layer_solve_results, layer_thicknesses)

    # -------------------------------------------------------------------------
    # Excitation and wave amplitude calculation.
    # -------------------------------------------------------------------------

    n = expansion.num_terms
    assert tuple(expansion.basis_coefficients[0, :]) == (0, 0)
    assert expansion.basis_coefficients.shape[0] == n

    # Generate wave amplitudes for forward-going waves with both polarizations.
    fwd_ambient_start = jnp.zeros((2 * n, 2), dtype=complex)
    fwd_ambient_start = fwd_ambient_start.at[0, 0].set(1)
    fwd_ambient_start = fwd_ambient_start.at[n, 1].set(1)

    # Compute the backward-goingpower at the start of the ambient.
    bwd_ambient_end = s_matrix.s21 @ fwd_ambient_start
    bwd_ambient_start = fields.propagate_amplitude(
        bwd_ambient_end,
        jnp.asarray(spec.thickness_ambient),
        layer_solve_result=layer_solve_results[0],
    )
    sz_fwd_ambient, sz_bwd_ambient = fields.amplitude_poynting_flux(
        forward_amplitude=fwd_ambient_start,
        backward_amplitude=bwd_ambient_start,
        layer_solve_result=layer_solve_results[0],
    )
    sz_fwd_ambient_sum = jnp.sum(jnp.abs(sz_fwd_ambient), axis=-2)
    sz_bwd_ambient_sum = jnp.sum(jnp.abs(sz_bwd_ambient), axis=-2)
    reflection = jnp.abs(sz_bwd_ambient_sum) / jnp.abs(sz_fwd_ambient_sum)

    # Compute the forward-going and backward-going wave amplitudes in the substrate,
    # a distance `spec.offset_monitor_substrate` from the start of the substrate.
    fwd_substrate_start = s_matrix.s11 @ fwd_ambient_start
    fwd_substrate_monitor, bwd_substrate_monitor = fields.colocate_amplitudes(
        fwd_substrate_start,
        jnp.zeros_like(fwd_substrate_start),
        z_offset=jnp.asarray(spec.offset_monitor_substrate.array),
        layer_solve_result=layer_solve_results[-1],
        layer_thickness=layer_thicknesses[-1],
    )

    # -------------------------------------------------------------------------
    # Compute fields at the monitor plane.
    # -------------------------------------------------------------------------

    # Compute transmitted power directly from the wave amplitudes.
    sz_fwd_monitor, _ = fields.amplitude_poynting_flux(
        forward_amplitude=fwd_substrate_monitor,
        backward_amplitude=bwd_substrate_monitor,
        layer_solve_result=layer_solve_results[-1],
    )
    sz_fwd_monitor_sum = jnp.sum(jnp.abs(sz_fwd_monitor), axis=-2)
    transmission = jnp.abs(sz_fwd_monitor_sum) / jnp.abs(sz_fwd_ambient_sum)

    # Compute electric and magnetic fields at the monitor plane in their Fourier
    # representation, and then on the real-space grid.
    ef, hf = fields.fields_from_wave_amplitudes(
        forward_amplitude=fwd_substrate_monitor,
        backward_amplitude=bwd_substrate_monitor,
        layer_solve_result=layer_solve_results[-1],
    )
    grid_shape: Tuple[int, int]
    grid_shape = density.shape[-2:]  # type: ignore[assignment]
    (ex, ey, ez), (hx, hy, hz), (x, y) = fields.fields_on_grid(
        electric_field=ef,
        magnetic_field=hf,
        layer_solve_result=layer_solve_results[-1],
        shape=grid_shape,
        num_unit_cells=(1, 1),
    )
    batch_shape = layer_solve_results[0].batch_shape
    assert ex.shape == batch_shape + grid_shape + (2,)

    # Compute the Poynting flux on the real-space grid at the monitor plane.
    sz = _time_average_z_poynting_flux((ex, ey, ez), (hx, hy, hz))
    assert sz.shape == batch_shape + grid_shape + (2,)

    # Create masks selecting the four pixelss.
    pixel_mask = _pixel_mask(grid_shape=grid_shape)
    assert pixel_mask.shape == grid_shape + (1, 4)

    # Use the mask to compute the time average Poynting flux into each quadrant. The
    # trailing two dimensions have shape `(2, 4)`; index `(i, j)` corresponds
    # to power for the `i` excitation (i.e. polarization) in the `j` pixel.
    pixel_transmission = jnp.mean(pixel_mask * sz[..., jnp.newaxis], axis=(-4, -3))
    pixel_transmission /= sz_fwd_ambient_sum[..., jnp.newaxis]

    assert pixel_transmission.shape == batch_shape + (2, 4)

    response = BayerResponse(
        wavelength=wavelength,
        polar_angle=polar_angle,
        azimuthal_angle=azimuthal_angle,
        transmission=pixel_transmission,
        reflection=reflection,
    )

    aux = {
        EFIELD_XY: (ex, ey, ez),
        HFIELD_XY: (hx, hy, hz),
        POYNTING_FLUX_XY: sz,
        COORDINATES_XY: (x, y),
        TRANSMITTED_POWER: transmission,
    }

    # -------------------------------------------------------------------------
    # Optionally compute fields in an xz cross section.
    # -------------------------------------------------------------------------

    if compute_fields:
        amplitudes_interior = fields.stack_amplitudes_interior(
            s_matrices_interior=s_matrices_interior,
            forward_amplitude_0_start=fwd_ambient_start,
            backward_amplitude_N_end=jnp.zeros_like(fwd_ambient_start),
        )
        layer_znum = tuple(
            [int(jnp.round(t / spec.grid_spacing) + 1) for t in layer_thicknesses]
        )
        x = jnp.stack(
            [jnp.arange(0, spec.grid_shape[0]) * spec.grid_spacing] * 2,
            axis=-1,
        )
        y = jnp.stack(
            [
                jnp.ones((spec.grid_shape[0],)) * spec.pixel_size * 0.5,
                jnp.ones((spec.grid_shape[0],)) * spec.pixel_size * 1.5,
            ],
            axis=-1,
        )
        assert x.shape == y.shape == (spec.grid_shape[0], 2)

        (
            (ex, ey, ez),
            (hx, hy, hz),
            (xf, yf, zf),
        ) = fields.stack_fields_3d_on_coordinates(
            amplitudes_interior=amplitudes_interior,
            layer_solve_results=layer_solve_results,
            layer_thicknesses=layer_thicknesses,
            layer_znum=layer_znum,
            x=x,
            y=y,
        )
        aux.update(
            {
                EFIELD_XZ: (ex, ey, ez),
                HFIELD_XZ: (hx, hy, hz),
                COORDINATES_XZ: (xf, yf, zf),
            }
        )

    return response, aux


# -----------------------------------------------------------------------------
# Functions related to the densities of various layers in the stack.
# -----------------------------------------------------------------------------


def _pixel_mask(grid_shape: Tuple[int, int]) -> jnp.ndarray:
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


def _time_average_z_poynting_flux(
    electric_fields: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    magnetic_fields: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> jnp.ndarray:
    """Computes the time-average z-directed Poynting flux, given the physical fields."""
    ex, ey, _ = electric_fields
    hx, hy, _ = magnetic_fields
    return jnp.real(ex * jnp.conj(hy) - ey * jnp.conj(hx))
