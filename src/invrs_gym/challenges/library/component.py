"""Defines the meta-atom library component and simulation routine.

Copyright (c) 2024 The INVRS-IO authors.
"""

import dataclasses
import functools
import pathlib
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as onp
from fmmax import basis, fields, fmm, scattering
from jax import tree_util
from totypes import json_utils, types

from invrs_gym.challenges import base
from invrs_gym.utils import materials, transforms

Params = Dict[str, types.BoundedArray | types.Density2DArray | jnp.ndarray]

DENSITY = "density"
THICKNESS = "thickness"

EFIELD_XZ = "efield_xz"
HFIELD_XZ = "hfield_xz"
COORDS_XZ = "coordinates_xz"

DENSITY_LOWER_BOUND = 0.0
DENSITY_UPPER_BOUND = 1.0

# Register a material with the refractive index data for TiO2 taken from Chen et al.
# https://www.nature.com/articles/s41467-023-38185-2
TIO2_CHEN = "tio2_chen"
TIO2_DATA_PATH = pathlib.Path(__file__).resolve().parent / "data/nk_tio2.csv"
materials.register_material(TIO2_CHEN, TIO2_DATA_PATH)


@dataclasses.dataclass
class LibrarySpec:
    """Defines the physical specification of the meta-atom library.

    Attributes:
        material_ambient: The ambient material.
        material_metasurface_solid: The material comprising metasurface solid regions.
        material_metasurface_void: The material comprising metasurface void regions.
        material_substrate: The substrate material.
        background_extinction_coefficient: Additional extinction coefficient to be
            included in the permittivity. Positive values correspond to optical loss
            and help stabilize the calculation.
        thickness_ambient: The thickness of the ambient layer.
        thickness_metasurface: The thickness of the metasurface layer.
        thickness_substrate: The thickness of the substrate layer.
        pitch: The pitch of the meta-atom, i.e. spacing in x and y directions.
        frame_width: The width of the frame surrounding each meta-atom. In the frame,
            all pixels are required to be void.
        grid_spacing: The spacing of the grid on which meta-atom density is defined.
    """

    material_ambient: materials.Material
    material_metasurface_solid: materials.Material
    material_metasurface_void: materials.Material
    material_substrate: materials.Material

    background_extinction_coeff: float

    thickness_ambient: float
    thickness_metasurface: float | jnp.ndarray | types.BoundedArray
    thickness_substrate: float

    pitch: float
    frame_width: float

    grid_spacing: float

    @property
    def grid_shape(self) -> Tuple[int, int]:
        """Return the shape of the grid implied by `grid_spacing`."""
        with jax.ensure_compile_time_eval():
            return (
                int(jnp.ceil(self.pitch / self.grid_spacing).astype(int)),
                int(jnp.ceil(self.pitch / self.grid_spacing).astype(int)),
            )

    @property
    def frame_pixels(self) -> int:
        """Return the width of the frame in pixels."""
        with jax.ensure_compile_time_eval():
            return int(jnp.ceil(self.frame_width / self.grid_spacing))


@dataclasses.dataclass
class LibrarySimParams:
    """Stores simulation parameters for the meta-atom library.

    Attributes:
        wavelength: The wavelength of the excitation.
        approximate_num_terms: Defines the number of terms in the Fourier expansion.
        formulation: The FMM formulation used for the calculation.
        truncation: The truncation of lattice vectors for the calculation.
    """

    wavelength: jnp.ndarray
    approximate_num_terms: int
    formulation: fmm.Formulation
    truncation: basis.Truncation


@dataclasses.dataclass
class LibraryResponse:
    """Stores the response of meta-atoms from the library.

    Attributes:
        wavelength: The wavelength of the excitation.
        transmission_rhcp: The complex amplitude transmission for excitation with
            right-hand circularly-polarized light. The coefficients are for right-
            hand and left-hand circular polarizations, i.e. the conserved and
            converted polarizations.
        transmission_lhcp: The complex amplitude transmission for excitation with
            left-hand circularly-polarized light.
        reflection_rhcp: The complex amplitude reflection for excitation with right-
            hand circularly-polarized light.
        reflection_lhcp: The complex amplitude reflection for excitation with left-
            hand circularly-polarized light.
    """

    wavelength: jnp.ndarray
    transmission_rhcp: jnp.ndarray
    transmission_lhcp: jnp.ndarray
    reflection_rhcp: jnp.ndarray
    reflection_lhcp: jnp.ndarray


tree_util.register_dataclass(
    nodetype=LibraryResponse,
    data_fields=[
        "wavelength",
        "transmission_rhcp",
        "transmission_lhcp",
        "reflection_rhcp",
        "reflection_lhcp",
    ],
    meta_fields=[],
)

json_utils.register_custom_type(LibraryResponse)


class LibraryComponent(base.Component):
    """Defines the meta-atom library component."""

    def __init__(
        self,
        spec: LibrarySpec,
        sim_params: LibrarySimParams,
        library_size: int,
        thickness_initializer: base.ThicknessInitializer,
        density_initializer: base.DensityInitializer,
        **seed_density_kwargs,
    ) -> None:
        """Initializes the meta-atom library component.

        Args:
            spec: The physical specification for the library of meta-atoms.
            sim_params: The default simulation parameters for the library of meta-atoms.
            library_size: The size of the library.
            thickness_initializer: Callable which generates the initial metasurface
                thickness from a random key and the seed thickness.
            density_initializer: Callable which generates the initial density from
                a random key and the seed density.
            **seed_density_kwargs: Keyword arguments which set the attributes of
                the seed density used to generate the inital parameters.
        """
        self.spec = spec
        self.sim_params = sim_params
        self.library_size = library_size
        self.thickness_initializer = thickness_initializer
        self.density_initializer = density_initializer

        self.seed_density = seed_density(
            library_size=library_size,
            grid_shape=spec.grid_shape,
            frame_pixels=spec.frame_pixels,
            **seed_density_kwargs,
        )
        self.expansion = basis.generate_expansion(
            primitive_lattice_vectors=basis.LatticeVectors(
                u=spec.pitch * basis.X, v=spec.pitch * basis.Y
            ),
            approximate_num_terms=self.sim_params.approximate_num_terms,
            truncation=self.sim_params.truncation,
        )

    def init(self, key: jax.Array) -> Params:
        """Return the initial parameters for a meta-atom library."""
        key_thickness, key_density = jax.random.split(key)
        params = {
            THICKNESS: self.thickness_initializer(
                key_thickness, self.spec.thickness_metasurface  # type: ignore[arg-type]
            ),
            DENSITY: self.density_initializer(key_density, self.seed_density),
        }
        # Ensure that there are no weak types in the initial parameters.
        return tree_util.tree_map(
            lambda x: jnp.asarray(x, jnp.asarray(x).dtype), params
        )

    def response(
        self,
        params: Params,
        wavelength: Optional[jnp.ndarray] = None,
        expansion: Optional[basis.Expansion] = None,
        compute_fields: bool = False,
    ) -> Tuple[LibraryResponse, base.AuxDict]:
        """Computes the response of the meta-atom library."""
        if wavelength is None:
            wavelength = self.sim_params.wavelength
        if expansion is None:
            expansion = self.expansion

        spec = dataclasses.replace(
            self.spec,
            thickness_metasurface=(
                params[THICKNESS].array,  # type: ignore[arg-type, union-attr]
            ),
        )

        return simulate_library(
            density=params[DENSITY],  # type: ignore[arg-type]
            spec=spec,
            wavelength=wavelength,
            expansion=expansion,
            formulation=self.sim_params.formulation,
            compute_fields=compute_fields,
        )


def seed_density(
    library_size: int,
    grid_shape: Tuple[int, int],
    frame_pixels: int,
    **kwargs: Any,
) -> types.Density2DArray:
    """Return the seed density for a meta-atom library.

    Args:
        library_size: The number of meta-atoms in the library. Determines the batch
            size of the density.
        grid_shape: The shape of the grid on which the density is defined.
        frame_pixels: The size of the fixed void frame surrounding each meta atom, in
            pixels.
        kwargs: keyword arguments specifying additional properties of the seed
            density, e.g. symmetries.

    Returns:
        The seed density.
    """

    # Check kwargs that are required for a sorter component.
    invalid_kwargs = ("array", "lower_bound", "upper_bound", "periodic", "fixed_void")
    if any(k in invalid_kwargs for k in kwargs):
        raise ValueError(
            f"Attributes were specified which confict with automatically-extracted "
            f"attributes. Got {kwargs.keys()} when {invalid_kwargs} are automatically "
            f"extracted."
        )

    mid_density_value = (DENSITY_LOWER_BOUND + DENSITY_UPPER_BOUND) / 2
    fixed_void = onp.zeros(grid_shape, dtype=bool)
    fixed_void[:frame_pixels, :] = True
    fixed_void[:, :frame_pixels] = True
    fixed_void[grid_shape[0] - frame_pixels :, :] = True
    fixed_void[:, grid_shape[1] - frame_pixels :] = True
    return types.Density2DArray(
        array=jnp.full((library_size,) + grid_shape, mid_density_value),
        lower_bound=DENSITY_LOWER_BOUND,
        upper_bound=DENSITY_UPPER_BOUND,
        periodic=(False, False),
        fixed_solid=None,
        fixed_void=fixed_void,
        **kwargs,
    )


def simulate_library(
    density: types.Density2DArray,
    spec: LibrarySpec,
    wavelength: jnp.ndarray,
    expansion: basis.Expansion,
    formulation: fmm.Formulation,
    compute_fields: bool,
) -> Tuple[LibraryResponse, base.AuxDict]:
    """Compute zeroth order complex reflection and transmission for each unit cell."""
    density_array = transforms.rescaled_density_array(
        density,
        lower_bound=DENSITY_LOWER_BOUND,
        upper_bound=DENSITY_UPPER_BOUND,
    )
    assert density_array.ndim in (2, 3)
    if density_array.ndim == 3:
        # Shift the batch axis of the density so it lies ahead of other batch axes.
        batch_dims = wavelength.ndim
        density_array = density_array.reshape(
            density_array.shape[0], *([1] * batch_dims), *density_array.shape[1:]
        )

    with jax.ensure_compile_time_eval():
        assert density_array.shape[-2] % spec.grid_shape[0] == 0
        assert density_array.shape[-1] % spec.grid_shape[1] == 0
    period_x = density_array.shape[-2] // spec.grid_shape[0] * spec.pitch
    period_y = density_array.shape[-1] // spec.grid_shape[1] * spec.pitch

    in_plane_wavevector = jnp.zeros((2,))
    eigensolve = functools.partial(
        fmm.eigensolve_isotropic_media,
        wavelength=jnp.asarray(wavelength),
        in_plane_wavevector=in_plane_wavevector,
        primitive_lattice_vectors=basis.LatticeVectors(
            u=period_x * basis.X,
            v=period_y * basis.Y,
        ),
        expansion=expansion,
        formulation=formulation,
    )

    def permittivity_fn(material: materials.Material) -> jnp.ndarray:
        dims_to_add = 1 + batch_dims - wavelength.ndim
        shape = (1,) * dims_to_add + wavelength.shape + (1, 1)
        return materials.permittivity(
            material=material,
            wavelength_um=wavelength,
            background_extinction_coeff=spec.background_extinction_coeff,
        ).reshape(shape)

    # Perform eigensolve for all the non-metasurface layers.
    with jax.ensure_compile_time_eval():
        solve_result_ambient = eigensolve(
            permittivity=permittivity_fn(spec.material_ambient)
        )
        solve_result_substrate = eigensolve(
            permittivity=permittivity_fn(spec.material_substrate)
        )

    permittivity_metasurface = transforms.interpolate_permittivity(
        permittivity_solid=permittivity_fn(spec.material_metasurface_solid),
        permittivity_void=permittivity_fn(spec.material_metasurface_void),
        density=density_array,
    )

    solve_result_metasurface = eigensolve(permittivity=permittivity_metasurface)
    layer_solve_results = (
        solve_result_ambient,
        solve_result_metasurface,
        solve_result_substrate,
    )
    layer_thicknesses = (
        jnp.asarray(spec.thickness_ambient),
        jnp.asarray(spec.thickness_metasurface),
        jnp.asarray(spec.thickness_substrate),
    )

    if compute_fields:
        s_matrices_interior = scattering.stack_s_matrices_interior(
            layer_solve_results=layer_solve_results,
            layer_thicknesses=layer_thicknesses,
        )
        s_matrix = s_matrices_interior[-1][0]
    else:
        s_matrix = scattering.stack_s_matrix(layer_solve_results, layer_thicknesses)

    n = expansion.num_terms
    assert tuple(expansion.basis_coefficients[0, :]) == (0, 0)
    assert expansion.basis_coefficients.shape[0] == n

    # Generate the incident wave amplitudes for two linear polarizations. The first
    # has electric field oriented in the x direction, the second has electric field
    # oriented in the y direction. Note that the power is incident from the substrate,
    # i.e. it is travelling "backward" in the fmmax convention.
    incident = jnp.zeros((2 * n, 2), dtype=complex)
    incident = incident.at[n, 0].set(1.0)
    incident = incident.at[0, 1].set(1.0)

    # Normalize the incident amplitudes, dividing by the square root of incident power.
    _, incident_power = fields.amplitude_poynting_flux(
        forward_amplitude=jnp.zeros_like(incident),
        backward_amplitude=incident,
        layer_solve_result=solve_result_substrate,
    )
    total_incident_power = jnp.sum(incident_power, axis=-2, keepdims=True)
    incident /= jnp.sqrt(jnp.abs(total_incident_power))

    # Calculate the reflected and transmitted mode amplitudes.
    reflected = s_matrix.s12 @ incident
    transmitted = s_matrix.s22 @ incident

    # Create the reflection matrix. It multiplies incident ampltudes in the linear
    # basis to give reflected zeroth-order fields in the linear basis.
    r_ex_from_ex = reflected[..., n, 0] / incident[..., n, 0]
    r_ey_from_ex = reflected[..., 0, 0] / incident[..., n, 0]
    r_ex_from_ey = reflected[..., n, 1] / incident[..., 0, 1]
    r_ey_from_ey = reflected[..., 0, 1] / incident[..., 0, 1]
    reflection_linear = jnp.stack(
        [
            jnp.stack([r_ex_from_ex, r_ex_from_ey], axis=-1),
            jnp.stack([r_ey_from_ex, r_ey_from_ey], axis=-1),
        ],
        axis=-2,
    )
    reflection_circular = circular_from_linear(reflection_linear)

    # Same operations for transmission coefficients.
    t_ex_from_ex = transmitted[..., n, 0] / incident[..., n, 0]
    t_ey_from_ex = transmitted[..., 0, 0] / incident[..., n, 0]
    t_ex_from_ey = transmitted[..., n, 1] / incident[..., 0, 1]
    t_ey_from_ey = transmitted[..., 0, 1] / incident[..., 0, 1]
    transmission_linear = jnp.stack(
        [
            jnp.stack([t_ex_from_ex, t_ex_from_ey], axis=-1),
            jnp.stack([t_ey_from_ex, t_ey_from_ey], axis=-1),
        ],
        axis=-2,
    )
    transmission_circular = circular_from_linear(transmission_linear)

    response = LibraryResponse(
        wavelength=wavelength,
        transmission_rhcp=transmission_circular[..., 0],
        transmission_lhcp=transmission_circular[..., 1],
        reflection_rhcp=reflection_circular[..., 0],
        reflection_lhcp=reflection_circular[..., 1],
    )

    aux = {}
    if compute_fields:
        amplitudes_interior = fields.stack_amplitudes_interior(
            s_matrices_interior=s_matrices_interior,
            forward_amplitude_0_start=jnp.zeros_like(incident),
            backward_amplitude_N_end=incident,
        )
        layer_znum = tuple(
            [int(jnp.round(t / spec.grid_spacing) + 1) for t in layer_thicknesses]
        )
        x = jnp.arange(density_array.shape[-2]) * spec.grid_spacing
        y = jnp.full_like(x, spec.pitch / 2)
        efield_xz, hfield_xz, coords_xz = fields.stack_fields_3d_on_coordinates(
            amplitudes_interior=amplitudes_interior,
            layer_solve_results=layer_solve_results,
            layer_thicknesses=layer_thicknesses,
            layer_znum=layer_znum,
            x=x,
            y=y,
        )
        aux.update({EFIELD_XZ: efield_xz, HFIELD_XZ: hfield_xz, COORDS_XZ: coords_xz})

    return response, aux


def circular_from_linear(x: jnp.ndarray) -> jnp.ndarray:
    """Return circular reflection or transmission coefficients from linear."""
    inv_conversion_matrix = jnp.asarray([[0.5, 0.5j], [0.5, -0.5j]])
    conversion_matrix = jnp.linalg.inv(inv_conversion_matrix)
    return inv_conversion_matrix @ x @ conversion_matrix
