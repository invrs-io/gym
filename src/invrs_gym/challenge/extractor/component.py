"""Defines the photon extractor component and simulation routine."""

import dataclasses
from typing import Any, Callable, Dict, Tuple, Union

import jax
import jax.numpy as jnp
from fmmax import (  # type: ignore[import]
    basis,
    fields,
    fmm,
    pml,
    scattering,
    sources,
    utils,
)
from jax import tree_util
from totypes import types  # type: ignore[import,attr-defined,unused-ignore]

AuxDict = Dict[str, Any]
Params = Any
DensityInitializer = Callable[[jax.Array, types.Density2DArray], types.Density2DArray]

DENSITY_LOWER_BOUND = 0.0
DENSITY_UPPER_BOUND = 1.0

EFIELD = "efield"
HFIELD = "hfield"
FIELD_COORDINATES = "field_coordinates"


def identity_initializer(key: jax.Array, seed_obj: Any) -> Any:
    """A basic identity initializer which simply returns the seed object."""
    del key
    return seed_obj


@dataclasses.dataclass
class ExtractorSpec:
    """Defines the physical specifcation of a photon extractor.

    Args:
        permittivity_ambient: Permittivity of the ambient material.
        permittivity_extractor: Permittivity of the extractor material.
        permittivity_substrate: Permittivity of the substrate.
        thickness_extractor: The thickness of the extractor layer.
        thickness_substrate_before_source: The distance between the substrate and
            the plane containing the source.
        thickness_substrate_after_source: The thickness of the substrate below the
            source plane.
        design_region_length: The length on one side of the square design region in
            the center of te unit cell.
        period_x: The size of the unit cell along the x direction.
        period_y: The size of the unit cell along the y direction.
        pml_thickness: Thickness of the perfectly matched layers at the borders
            of the simulation unit cell.
        source_monitor_offset: The distance along the z direction between the source
            and monitor planes above and below the source used to compute the total
            power emitted by the source.
        output_monitor_offset: The distance along the z direction between the top of
            the extractor structure and a monitor plane used to compute the total
            power extracted from the source.
    """

    permittivity_ambient: complex
    permittivity_extractor: complex
    permittivity_substrate: complex

    thickness_ambient: float
    thickness_extractor: float
    thickness_substrate_before_source: float
    thickness_substrate_after_source: float

    design_region_length: float
    period_x: float
    period_y: float
    pml_thickness: float

    source_monitor_offset: float
    output_monitor_offset: float


@dataclasses.dataclass
class ExtractorSimParams:
    """Parameters that configure the simulation of a photon extractor.

    Attributes:
        grid_shape: The shape of the grid on which the permittivity is defined.
        wavelength: The wavelength of the excitation.
        formulation: The FMM formulation to be used.
        approximate_num_terms: Defines the number of terms in the Fourier expansion.
        truncation: Determines how the Fourier basis is truncated.
    """

    grid_shape: Tuple[int, int]
    layer_znum: Tuple[int, int, int, int]
    wavelength: Union[float, jnp.ndarray]
    formulation: fmm.Formulation
    approximate_num_terms: int
    truncation: basis.Truncation


@dataclasses.dataclass
class ExtractorResponse:
    """Contains the response of the photon extractor.

    Attributes:
        wavelength: The wavelength for the efficiency calculation.
        emitted_power: The total power emitted by the source.
        extracted_power: The total power extracted from the source.
        expansion: Defines the Fourier expansion for the calculation.
    """

    wavelength: jnp.ndarray
    emitted_power: jnp.ndarray
    extracted_power: jnp.ndarray
    expansion: basis.Expansion


tree_util.register_pytree_node(
    ExtractorResponse,
    lambda r: (
        (
            r.wavelength,
            r.emitted_power,
            r.extracted_power,
            r.expansion,
        ),
        None,
    ),
    lambda _, children: ExtractorResponse(*children),
)


def seed_density(
    grid_shape: Tuple[int, int],
    spec: ExtractorSpec,
    **kwargs: Any,
) -> types.Density2DArray:
    """Return the seed density for a photon extractor component.

    Args:
        grid_shape: The shape of the grid on which the density is defined.
        spec: Defines the physical structure of the photon extractor.
        kwargs: keyword arguments specifying additional properties of the seed
            density, e.g. symmetries.

    Returns:
        The seed density.
    """

    # Check kwargs that are required for a photon extractor component.
    invalid_kwargs = (
        "array",
        "fixed_solid",
        "fixed_void",
        "lower_bound",
        "upper_bound",
        "periodic",
    )
    if any(k in invalid_kwargs for k in kwargs):
        raise ValueError(
            f"Attributes were specified which confict with automatically-extracted "
            f"attributes. Got {kwargs.keys()} when {invalid_kwargs} are automatically "
            f"extracted."
        )

    x, y = jnp.meshgrid(
        jnp.arange(0, spec.period_x, spec.period_x / grid_shape[0]),
        jnp.arange(0, spec.period_y, spec.period_y / grid_shape[1]),
        indexing="ij",
    )
    x_offset = (spec.period_x - spec.design_region_length) / 2
    y_offset = (spec.period_y - spec.design_region_length) / 2
    fixed_void = (
        (x < x_offset)
        | (y < y_offset)
        | (x > spec.period_x - x_offset)
        | (y > spec.period_y - y_offset)
    )

    mid_density_value = (DENSITY_LOWER_BOUND + DENSITY_UPPER_BOUND) / 2
    return types.Density2DArray(
        array=jnp.full(grid_shape, mid_density_value),
        lower_bound=DENSITY_LOWER_BOUND,
        upper_bound=DENSITY_UPPER_BOUND,
        fixed_solid=jnp.zeros_like(fixed_void),
        fixed_void=jnp.asarray(fixed_void),
        periodic=(True, True),
        **kwargs,
    )


def simulate_extractor(
    density_array: jnp.ndarray,
    spec: ExtractorSpec,
    layer_znum: Tuple[int, int, int, int],
    wavelength: jnp.ndarray,
    expansion: basis.Expansion,
    formulation: fmm.Formulation,
    compute_fields: bool,
) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
    """Simulates the photon extractor device.

    Args:
        density_array: Defines the pattern of the photon extractor layer.
        spec: Defines the physical specifcation of the photon extractor.
        wavelength: The wavelength of the excitation.
        expansion: Defines the Fourier expansion for the calculation.
        formulation: Defines the FMM formulation to be used.

    Returns:
        The per-order transmission and reflection efficiency, having shape
        `(num_wavelengths, nkx, nky)`.
    """
    in_plane_wavevector = jnp.zeros((2,))
    primitive_lattice_vectors = basis.LatticeVectors(
        u=spec.period_x * basis.X,
        v=spec.period_y * basis.Y,
    )

    grid_shape = density_array.shape

    def eigensolve_pml(permittivity: jnp.ndarray) -> fmm.LayerSolveResult:
        # Permittivities and permeabilities in order needed for eigensolve below.
        permittivities_pml, permeabilities_pml = pml.apply_uniaxial_pml(
            permittivity=permittivity,
            pml_params=pml.PMLParams(
                num_x=int(grid_shape[0] * spec.pml_thickness / spec.period_x),
                num_y=int(grid_shape[1] * spec.pml_thickness / spec.period_y),
            ),
        )
        return fmm.eigensolve_general_anisotropic_media(
            wavelength,
            in_plane_wavevector,
            primitive_lattice_vectors,
            *permittivities_pml,
            *permeabilities_pml,
            expansion=expansion,
            formulation=formulation,
            vector_field_source=density_array,
        )

    with jax.ensure_compile_time_eval():
        solve_result_ambient = eigensolve_pml(
            permittivity=jnp.full(grid_shape, spec.permittivity_ambient)
        )
        solve_result_extractor = eigensolve_pml(
            permittivity=utils.interpolate_permittivity(
                permittivity_solid=spec.permittivity_extractor,
                permittivity_void=spec.permittivity_ambient,
                density=density_array,
            ),
        )
        solve_result_substrate = eigensolve_pml(
            permittivity=jnp.full(grid_shape, spec.permittivity_substrate)
        )

    layer_solve_results = (
        solve_result_ambient,
        solve_result_extractor,
        solve_result_substrate,  # Before the source.
        solve_result_substrate,  # After the source.
    )
    layer_thicknesses = (
        jnp.asarray(spec.thickness_ambient),
        jnp.asarray(spec.thickness_extractor),
        jnp.asarray(spec.thickness_substrate_before_source),
        jnp.asarray(spec.thickness_substrate_after_source),
    )

    if compute_fields:
        # If the field calculation is desired, compute the interior scattering
        # matrices. For each layer in the stack, the interior scattering matrices
        # consist of a pair of matrices, one for the substack below the layer, and
        # one for the substack above the layer.
        s_matrices_interior_before_source = scattering.stack_s_matrices_interior(
            layer_solve_results=layer_solve_results[:-1],
            layer_thicknesses=layer_thicknesses[:-1],
        )
        s_matrices_interior_after_source = scattering.stack_s_matrices_interior(
            layer_solve_results=layer_solve_results[-1:],
            layer_thicknesses=layer_thicknesses[-1:],
        )
        s_matrix_before_source = s_matrices_interior_before_source[-1][0]
        s_matrix_after_source = s_matrices_interior_after_source[-1][0]

    else:
        s_matrix_before_source = scattering.stack_s_matrix(
            layer_solve_results=layer_solve_results[:-1],
            layer_thicknesses=layer_thicknesses[:-1],
        )
        s_matrix_after_source = scattering.stack_s_matrix(
            layer_solve_results=layer_solve_results[-1:],
            layer_thicknesses=layer_thicknesses[-1:],
        )

    # Generate the Fourier representation of x, y, and z-oriented point dipoles.
    dipole = sources.dirac_delta_source(
        location=jnp.asarray([[spec.period_x / 2, spec.period_y / 2]]),
        in_plane_wavevector=in_plane_wavevector,
        primitive_lattice_vectors=primitive_lattice_vectors,
        expansion=expansion,
    )
    zeros = jnp.zeros_like(dipole)
    jx = jnp.concatenate([dipole, zeros, zeros], axis=-1)
    jy = jnp.concatenate([zeros, dipole, zeros], axis=-1)
    jz = jnp.concatenate([zeros, zeros, dipole], axis=-1)

    # Solve for the eigenmode amplitudes that result from the dipole excitation.
    (
        bwd_amplitude_ambient_end,
        fwd_amplitude_before_start,
        bwd_amplitude_before_end,
        fwd_amplitude_after_start,
        bwd_amplitude_after_end,
        _,
    ) = sources.amplitudes_for_source(
        jx=jx,
        jy=jy,
        jz=jz,
        s_matrix_before_source=s_matrix_before_source,
        s_matrix_after_source=s_matrix_after_source,
    )

    # -------------------------------------------------------------------------
    # Compute fields in an xz cross section.
    # -------------------------------------------------------------------------

    aux = {}
    if compute_fields:
        amplitudes_interior = fields.stack_amplitudes_interior_with_source(
            s_matrices_interior_before_source=s_matrices_interior_before_source,
            s_matrices_interior_after_source=s_matrices_interior_after_source,
            backward_amplitude_before_end=bwd_amplitude_before_end,
            forward_amplitude_after_start=fwd_amplitude_after_start,
        )
        x = jnp.linspace(0, spec.period_x, grid_shape[0])
        y = jnp.ones_like(x) * spec.period_y / 2
        (ex, ey, ez), (hx, hy, hz), (x, y, z) = fields.stack_fields_3d_on_coordinates(
            amplitudes_interior=amplitudes_interior,
            layer_solve_results=layer_solve_results,
            layer_thicknesses=layer_thicknesses,
            layer_znum=layer_znum,
            x=x,
            y=y,
        )
        aux.update(
            {
                EFIELD: (ex, ey, ez),
                HFIELD: (hx, hy, hz),
                FIELD_COORDINATES: (x, y, z),
            }
        )

    # -------------------------------------------------------------------------
    # Compute the total emitted power and total extracted power.
    # -------------------------------------------------------------------------

    # Compute the Poynting flux in the layer before the source, at the monitor.
    fwd_amplitude_before_monitor = fields.propagate_amplitude(
        amplitude=fwd_amplitude_before_start,
        distance=spec.thickness_substrate_before_source - spec.source_monitor_offset,
        layer_solve_result=solve_result_substrate,
    )
    bwd_amplitude_before_monitor = fields.propagate_amplitude(
        amplitude=bwd_amplitude_before_end,
        distance=spec.source_monitor_offset,
        layer_solve_result=solve_result_substrate,
    )
    fwd_flux_before_monitor, bwd_flux_before_monitor = fields.directional_poynting_flux(
        forward_amplitude=fwd_amplitude_before_monitor,
        backward_amplitude=bwd_amplitude_before_monitor,
        layer_solve_result=solve_result_substrate,
    )

    # Compute the Poynting flux in the layer after the source, at the monitor.
    fwd_amplitude_after_monitor = fields.propagate_amplitude(
        amplitude=fwd_amplitude_after_start,
        distance=spec.source_monitor_offset,
        layer_solve_result=solve_result_substrate,
    )
    bwd_amplitude_after_monitor = fields.propagate_amplitude(
        amplitude=bwd_amplitude_after_end,
        distance=spec.thickness_substrate_after_source - spec.source_monitor_offset,
        layer_solve_result=solve_result_substrate,
    )
    fwd_flux_after_monitor, bwd_flux_after_monitor = fields.directional_poynting_flux(
        forward_amplitude=fwd_amplitude_after_monitor,
        backward_amplitude=bwd_amplitude_after_monitor,
        layer_solve_result=solve_result_substrate,
    )

    # Compute the Poynting flux at the monitor located above the extractor.
    bwd_amplitude_ambient_monitor = fields.propagate_amplitude(
        amplitude=bwd_amplitude_ambient_end,
        distance=spec.output_monitor_offset,
        layer_solve_result=solve_result_ambient,
    )
    _, bwd_flux_ambient_monitor = fields.directional_poynting_flux(
        forward_amplitude=jnp.zeros_like(bwd_amplitude_ambient_monitor),
        backward_amplitude=bwd_amplitude_ambient_monitor,
        layer_solve_result=s_matrix_before_source.start_layer_solve_result,
    )

    # Compute the total forward and backward flux resulting from the source. The
    # forward flux from the source is the difference between the forward flux just
    # after the source, and the forward flux just before the source. The backward
    # flux is defined analogously.
    fwd_flux_from_source = fwd_flux_after_monitor - fwd_flux_before_monitor
    bwd_flux_from_source = bwd_flux_before_monitor - bwd_flux_after_monitor

    # Sum the the flux over all Fourier orders.
    total_emitted = jnp.sum(fwd_flux_from_source, axis=-2) - jnp.sum(
        bwd_flux_from_source, axis=-2
    )
    total_extracted = -jnp.sum(bwd_flux_ambient_monitor, axis=-2)

    return total_extracted, total_emitted, aux
