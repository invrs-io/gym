"""Defines the photon extractor component and simulation routine.

Copyright (c) 2023 The INVRS-IO authors.
"""

import dataclasses
import functools
from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
from fmmax import basis, fields, fmm, pml, scattering, sources  # type: ignore[import]
from jax import tree_util
from totypes import types

from invrs_gym import utils
from invrs_gym.challenges import base

DENSITY_LOWER_BOUND = 0.0
DENSITY_UPPER_BOUND = 1.0

EFIELD = "efield"
HFIELD = "hfield"
FIELD_COORDINATES = "field_coordinates"


@dataclasses.dataclass
class ExtractorSpec:
    """Defines the physical specifcation of a photon extractor.

    Args:
        permittivity_ambient: Permittivity of the ambient material.
        permittivity_oxide: Permittivity of the oxide material.
        permittivity_extractor: Permittivity of the extractor material.
        permittivity_substrate: Permittivity of the substrate.
        thickness_ambient: The thickness of the ambient layer.
        thickness_oxide: The thickness of the oxide layer.
        thickness_extractor: The thickness of the extractor layer.
        thickness_substrate_before_source: The distance between the substrate and
            the plane containing the source.
        thickness_substrate_after_source: The thickness of the substrate below the
            source plane.
        width_design_region: The width of the square design region.
        width_padding: Width of the region between the design and the PML.
        width_pml: Width of the perfectly matched layers at the borders of the
            simulation unit cell.
        fwhm_source: The spatial full-width at half maximum for the Gaussian dipole.
        offset_monitor_source: The distance along the z direction between the source
            and monitor planes above and below the source used to compute the total
            power emitted by the source.
        offset_monitor_ambient: The distance along the z direction between the top of
            the extractor structure and a monitor plane used to compute the total
            power extracted from the source.
        width_monitor_ambient: The length on one side of the square flux monitor
            above the design region.
    """

    permittivity_ambient: complex
    permittivity_oxide: complex
    permittivity_extractor: complex
    permittivity_substrate: complex

    thickness_ambient: float
    thickness_oxide: float
    thickness_extractor: float
    thickness_substrate_before_source: float
    thickness_substrate_after_source: float

    width_design_region: float
    width_padding: float
    width_pml: float

    fwhm_source: float

    offset_monitor_source: float
    offset_monitor_ambient: float
    width_monitor_ambient: float

    @property
    def pitch(self) -> float:
        return self.width_design_region + 2 * (self.width_padding + self.width_pml)


@dataclasses.dataclass
class ExtractorSimParams:
    """Parameters that configure the simulation of a photon extractor.

    Attributes:
        grid_spacing: The spacing of points on the real-space grid.
        wavelength: The wavelength of the excitation.
        formulation: The FMM formulation to be used.
        approximate_num_terms: Defines the number of terms in the Fourier expansion.
        truncation: Determines how the Fourier basis is truncated.
    """

    grid_spacing: float
    wavelength: float | jnp.ndarray
    formulation: fmm.Formulation
    approximate_num_terms: int
    truncation: basis.Truncation


@dataclasses.dataclass
class ExtractorResponse:
    """Contains the response of the photon extractor.

    Attributes:
        wavelength: The wavelength for the efficiency calculation.
        emitted_power: The total power
        extracted_power: The total power extracted from the source, including power
            at large angles which is not included in `collected_power`.
        collected_power: The total power collected from the source, collected by the
            ambient monitor above the extractor. Since the monitor is smaller than
            the unit cell, not all emitted power is counted as collected.
    """

    wavelength: jnp.ndarray
    emitted_power: jnp.ndarray
    extracted_power: jnp.ndarray
    collected_power: jnp.ndarray


tree_util.register_pytree_node(
    ExtractorResponse,
    lambda r: (
        (
            r.wavelength,
            r.emitted_power,
            r.extracted_power,
            r.collected_power,
        ),
        None,
    ),
    lambda _, children: ExtractorResponse(*children),
)


class ExtractorComponent(base.Component):
    """Defines a photon extractor component."""

    def __init__(
        self,
        spec: ExtractorSpec,
        sim_params: ExtractorSimParams,
        density_initializer: base.DensityInitializer,
        **seed_density_kwargs: Any,
    ) -> None:
        """Initializes the photon extractor component.

        Args:
            spec: Defines the physical specification of the extractor.
            sim_params: Defines simulation parameters for the extractor.
            density_initializer: Callable which generates the initial density from
                a random key and the seed density.
            **seed_density_kwargs: Keyword arguments which set the attributes of
                the seed density used to generate the inital parameters.
        """

        self.spec = spec
        self.sim_params = sim_params
        self.density_initializer = density_initializer

        _num_gridpoints = functools.partial(
            divide_and_round,
            b=sim_params.grid_spacing,
        )
        self.grid_shape = (_num_gridpoints(spec.pitch),) * 2

        # When computing fields within each layer, a gridpoint is placed at the
        # very start and end of the layer, and so an additional gridpoint is needed
        # to ensure gridpoints are correctly spaced.
        self.layer_znum = (
            _num_gridpoints(spec.thickness_ambient) + 1,
            _num_gridpoints(spec.thickness_oxide) + 1,
            _num_gridpoints(spec.thickness_extractor) + 1,
            _num_gridpoints(spec.thickness_substrate_before_source) + 1,
            _num_gridpoints(spec.thickness_substrate_after_source) + 1,
        )

        self.seed_density = seed_density(
            grid_shape=self.grid_shape,
            spec=self.spec,
            **seed_density_kwargs,
        )
        self.expansion = basis.generate_expansion(
            primitive_lattice_vectors=basis.LatticeVectors(
                u=self.spec.pitch * basis.X,
                v=self.spec.pitch * basis.Y,
            ),
            approximate_num_terms=self.sim_params.approximate_num_terms,
            truncation=self.sim_params.truncation,
        )

    def init(self, key: jax.Array) -> types.Density2DArray:
        """Return the initial parameters for the photon extractor component."""
        params = self.density_initializer(key, self.seed_density)
        # Ensure that there are no weak types in the initial parameters.
        return tree_util.tree_map(
            lambda x: jnp.asarray(x, jnp.asarray(x).dtype), params
        )

    def response(
        self,
        params: types.Density2DArray,
        *,
        wavelength: Optional[float | jnp.ndarray] = None,
        expansion: Optional[basis.Expansion] = None,
        compute_fields: bool = False,
    ) -> Tuple[ExtractorResponse, base.AuxDict]:
        """Computes the response of the diffractive splitter.

        Args:
            params: The parameters defining the diffractive splitter, matching those
                returned by the `init` method.
            wavelength: Optional wavelength to override the default in `sim_params`.
            expansion: Optional expansion to override the default `expansion`.
            compute_fields: If `True`, computes and xz cross section for electric
                and magnetic fields, which makes the calculation more expensive.

        Returns:
            The `(response, aux)` tuple.
        """
        if expansion is None:
            expansion = self.expansion
        if wavelength is None:
            wavelength = self.sim_params.wavelength

        return simulate_extractor(
            density=params,  # type: ignore[arg-type]
            spec=self.spec,
            layer_znum=self.layer_znum,
            wavelength=jnp.asarray(wavelength),
            expansion=expansion,
            formulation=self.sim_params.formulation,
            compute_fields=compute_fields,
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

    design_mask = _mask(
        grid_shape,
        pitch=spec.pitch,
        width=spec.width_design_region,
    )
    fixed_void = ~design_mask

    mid_density_value = (DENSITY_LOWER_BOUND + DENSITY_UPPER_BOUND) / 2
    return types.Density2DArray(
        array=jnp.full(grid_shape, mid_density_value),
        lower_bound=DENSITY_LOWER_BOUND,
        upper_bound=DENSITY_UPPER_BOUND,
        fixed_solid=jnp.zeros_like(fixed_void),
        fixed_void=fixed_void,
        periodic=(False, False),
        **kwargs,
    )


def simulate_extractor(
    density: types.Density2DArray,
    spec: ExtractorSpec,
    layer_znum: Tuple[int, int, int, int, int],
    wavelength: jnp.ndarray,
    expansion: basis.Expansion,
    formulation: fmm.Formulation,
    compute_fields: bool,
) -> Tuple[ExtractorResponse, base.AuxDict]:
    """Simulates the photon extractor device.

    Args:
        density_array: Defines the pattern of the photon extractor layer.
        spec: Defines the physical specifcation of the photon extractor.
        layer_znum: The number of gridpoints in the z-direction used for fields.
        wavelength: The wavelength of the excitation.
        expansion: Defines the Fourier expansion for the calculation.
        formulation: Defines the FMM formulation to be used.
        compute_fields: If `True`, returns electric and magnetic fields in the
            `aux` dictionary.

    Returns:
        The `ExtractorResponse` and `aux` dictionary.
    """
    density_array = utils.transforms.rescaled_density_array(
        density,
        lower_bound=DENSITY_LOWER_BOUND,
        upper_bound=DENSITY_UPPER_BOUND,
    )
    in_plane_wavevector = jnp.zeros((2,))
    primitive_lattice_vectors = basis.LatticeVectors(
        u=spec.pitch * basis.X,
        v=spec.pitch * basis.Y,
    )

    grid_shape: Tuple[int, int] = density_array.shape  # type: ignore[assignment]

    def eigensolve_pml(permittivity: jnp.ndarray) -> fmm.LayerSolveResult:
        # Permittivities and permeabilities are returned in the order needed
        # for the anisotropic eigensolve below.
        permittivities_pml, permeabilities_pml = pml.apply_uniaxial_pml(
            permittivity=permittivity,
            pml_params=_pml_params(grid_shape, spec),
        )
        return fmm.eigensolve_general_anisotropic_media(
            wavelength,
            in_plane_wavevector,
            primitive_lattice_vectors,
            *permittivities_pml,
            *permeabilities_pml,
            expansion=expansion,
            formulation=formulation,
            vector_field_source=jnp.mean(jnp.asarray(permittivities_pml), axis=0),
        )

    with jax.ensure_compile_time_eval():
        solve_result_ambient = eigensolve_pml(
            permittivity=jnp.full(grid_shape, spec.permittivity_ambient)
        )
        solve_result_oxide = eigensolve_pml(
            permittivity=utils.transforms.interpolate_permittivity(
                permittivity_solid=jnp.asarray(spec.permittivity_oxide),
                permittivity_void=jnp.asarray(spec.permittivity_ambient),
                density=density_array,
            ),
        )
        solve_result_extractor = eigensolve_pml(
            permittivity=utils.transforms.interpolate_permittivity(
                permittivity_solid=jnp.asarray(spec.permittivity_extractor),
                permittivity_void=jnp.asarray(spec.permittivity_ambient),
                density=density_array,
            ),
        )
        solve_result_substrate = eigensolve_pml(
            permittivity=jnp.full(grid_shape, spec.permittivity_substrate)
        )

    layer_solve_results = (
        solve_result_ambient,
        solve_result_oxide,
        solve_result_extractor,
        solve_result_substrate,  # Before the source.
        solve_result_substrate,  # After the source.
    )
    layer_thicknesses = (
        jnp.asarray(spec.thickness_ambient),
        jnp.asarray(spec.thickness_oxide),
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
    dipole = sources.gaussian_source(
        fwhm=jnp.asarray(spec.fwhm_source),
        location=jnp.asarray([[spec.pitch / 2, spec.pitch / 2]]),
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
        x = jnp.linspace(0, spec.pitch, grid_shape[0])
        y = jnp.ones_like(x) * spec.pitch / 2
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
    # Total emitted power measured at monitors in the substrate.
    # -------------------------------------------------------------------------

    # Compute the Poynting flux in the layer before the source, at the monitor.
    fwd_amplitude_before_monitor = fields.propagate_amplitude(
        amplitude=fwd_amplitude_before_start,
        distance=jnp.asarray(
            spec.thickness_substrate_before_source - spec.offset_monitor_source
        ),
        layer_solve_result=solve_result_substrate,
    )
    bwd_amplitude_before_monitor = fields.propagate_amplitude(
        amplitude=bwd_amplitude_before_end,
        distance=jnp.asarray(spec.offset_monitor_source),
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
        distance=jnp.asarray(spec.offset_monitor_source),
        layer_solve_result=solve_result_substrate,
    )
    bwd_amplitude_after_monitor = fields.propagate_amplitude(
        amplitude=bwd_amplitude_after_end,
        distance=jnp.asarray(
            spec.thickness_substrate_after_source - spec.offset_monitor_source
        ),
        layer_solve_result=solve_result_substrate,
    )
    fwd_flux_after_monitor, bwd_flux_after_monitor = fields.directional_poynting_flux(
        forward_amplitude=fwd_amplitude_after_monitor,
        backward_amplitude=bwd_amplitude_after_monitor,
        layer_solve_result=solve_result_substrate,
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

    # -------------------------------------------------------------------------
    # Total extracted power measured at a monitor above the extractor.
    # -------------------------------------------------------------------------

    # Compute the eigenmode amplitudes at the ambient flux monitor.
    bwd_amplitude_ambient_monitor = fields.propagate_amplitude(
        amplitude=bwd_amplitude_ambient_end,
        distance=jnp.asarray(spec.offset_monitor_ambient),
        layer_solve_result=solve_result_ambient,
    )
    _, bwd_flux_ambient_monitor = fields.directional_poynting_flux(
        forward_amplitude=jnp.zeros_like(bwd_amplitude_ambient_monitor),
        backward_amplitude=bwd_amplitude_ambient_monitor,
        layer_solve_result=solve_result_ambient,
    )
    total_extracted = -jnp.sum(bwd_flux_ambient_monitor, axis=-2)

    # We also want to compute the power collected by a monitor that is located above
    # the extractor design, and does not extend to the edges of the unit cell. To find
    # the flux through this monitor, compute the flux on the real-space grid and sum
    # over the target region.
    #
    # First compute Fourier amplitudes of the electric and magnetic fields.
    ambient_monitor_ef, ambient_monitor_hf = fields.fields_from_wave_amplitudes(
        jnp.zeros_like(bwd_amplitude_ambient_monitor),
        bwd_amplitude_ambient_monitor,
        layer_solve_result=solve_result_ambient,
    )
    # Compute the real-space electric and magnetic fields at the monitor.
    ambient_monitor_ef, ambient_monitor_hf, (x, y) = fields.fields_on_grid(
        electric_field=ambient_monitor_ef,
        magnetic_field=ambient_monitor_hf,
        layer_solve_result=solve_result_ambient,
        shape=grid_shape,
        num_unit_cells=(1, 1),
    )
    assert ambient_monitor_ef[0].shape == wavelength.shape + grid_shape + (3,)
    # Compute the Poynting flux on the real-space grid at the monitor.
    bwd_flux_ambient_monitor = _time_average_z_poynting_flux(
        electric_field=ambient_monitor_ef,
        magnetic_field=ambient_monitor_hf,
    )
    # Compute the masked flux.
    monitor_mask = _mask(
        grid_shape=grid_shape,
        pitch=spec.pitch,
        width=spec.width_monitor_ambient,
    )
    masked_bwd_flux_ambient_monitor = jnp.where(
        monitor_mask[..., jnp.newaxis],
        bwd_flux_ambient_monitor,
        0.0,
    )
    total_collected = -jnp.mean(masked_bwd_flux_ambient_monitor, axis=(-3, -2))
    assert total_extracted.shape == total_emitted.shape == total_collected.shape

    response = ExtractorResponse(
        wavelength=wavelength,
        emitted_power=total_emitted,
        extracted_power=total_extracted,
        collected_power=total_collected,
    )
    return response, aux


def _pml_params(grid_shape: Tuple[int, int], spec: ExtractorSpec) -> pml.PMLParams:
    """Return PML parameters for the specified grid shape and extractor spec."""
    return pml.PMLParams(
        num_x=int(grid_shape[0] * spec.width_pml / spec.pitch),
        num_y=int(grid_shape[1] * spec.width_pml / spec.pitch),
    )


def _time_average_z_poynting_flux(
    electric_field: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    magnetic_field: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> jnp.ndarray:
    """Computes the time-average z-directed Poynting flux given the physical fields."""
    # https://github.com/facebookresearch/fmmax/blob/main/examples/sorter.py
    ex, ey, _ = electric_field
    hx, hy, _ = magnetic_field
    return jnp.real(ex * jnp.conj(hy) - ey * jnp.conj(hx))


def _mask(
    grid_shape: Tuple[int, int],
    pitch: float,
    width: float,
) -> jnp.ndarray:
    """Generate a mask that is `True` in a centered region having width `width`"""
    x, y = jnp.meshgrid(
        jnp.arange(0.5, grid_shape[0]) * pitch / grid_shape[0],
        jnp.arange(0.5, grid_shape[1]) * pitch / grid_shape[1],
        indexing="ij",
    )
    x_offset = (pitch - width) / 2
    y_offset = (pitch - width) / 2
    return (
        (x >= x_offset)
        & (y >= y_offset)
        & (x < pitch - x_offset)
        & (y < pitch - y_offset)
    )
