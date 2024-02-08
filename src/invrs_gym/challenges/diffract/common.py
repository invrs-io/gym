"""Defines functions common across diffract challenges.

Copyright (c) 2023 The INVRS-IO authors.
"""

import dataclasses
from typing import Any, Tuple

import jax.numpy as jnp
import numpy as onp
from fmmax import basis, fields, fmm, scattering  # type: ignore[import-untyped]
from jax import tree_util
from totypes import json_utils, types

from invrs_gym import utils

Params = Any

DENSITY_LOWER_BOUND = 0.0
DENSITY_UPPER_BOUND = 1.0


@dataclasses.dataclass
class GratingSpec:
    """Defines the physical specifcation of a grating.

    Attributes:
        permittivity_ambient: Permittivity of the ambient material.
        permittivity_grating: Permittivity of the grating teeth.
        permittivity_encapsulation: Permittivity of the material in gaps between
            grating teeth.
        permittivity_substrate: Permittivity of the substrate.
        thickness_grating: Thickness of the grating layer.
        period_x: The size of the unit cell along the x direction.
        period_y: The size of the unit cell along the y direction.
    """

    permittivity_ambient: complex
    permittivity_grating: complex
    permittivity_encapsulation: complex
    permittivity_substrate: complex

    thickness_grating: float | jnp.ndarray | types.BoundedArray

    period_x: float
    period_y: float


@dataclasses.dataclass
class GratingSimParams:
    """Parameters that configure the simulation of a grating.

    Attributes:
        grid_spacing: The spacing of the grid on which grating permittivity is defined.
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
class GratingResponse:
    """Contains the response of the grating.

    Attributes:
        wavelength: The wavelength for the efficiency calculation.
        polar_angle: The polar angle for the efficiency calculation.
        azimuthal_angle: The azimuthal angle for the efficiency calculation.
        transmission_efficiency: The per-order and per-wavelength coupling efficiency
            with which the excitation is transmitted.
        transmission_efficiency: The per-order and per-wavelength coupling efficiency
            with which the excitation is reflected.
        expansion: Defines the Fourier expansion for the calculation.
    """

    wavelength: jnp.ndarray
    polar_angle: jnp.ndarray
    azimuthal_angle: jnp.ndarray
    transmission_efficiency: jnp.ndarray
    reflection_efficiency: jnp.ndarray
    expansion: basis.Expansion


json_utils.register_custom_type(GratingResponse)

tree_util.register_pytree_node(
    GratingResponse,
    lambda r: (
        (
            r.wavelength,
            r.polar_angle,
            r.azimuthal_angle,
            r.transmission_efficiency,
            r.reflection_efficiency,
            r.expansion,
        ),
        None,
    ),
    lambda _, children: GratingResponse(*children),
)


def grid_shape(
    period_x: float, period_y: float, grid_spacing: float
) -> Tuple[int, int]:
    """Return the grid shape for the given unit cell parameters."""
    return (
        int(jnp.ceil(period_x / grid_spacing)),
        int(jnp.ceil(period_y / grid_spacing)),
    )


def seed_density(grid_shape: Tuple[int, int], **kwargs: Any) -> types.Density2DArray:
    """Return the seed density for a grating component.

    Args:
        grid_shape: The shape of the grid on which the density is defined.
        kwargs: keyword arguments specifying additional properties of the seed
            density, e.g. symmetries.

    Returns:
        The seed density.
    """

    # Check kwargs that are required for a grating component.
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


def index_for_order(
    order: Tuple[int, int],
    expansion: basis.Expansion,
) -> int:
    """Returns the index for the specified Fourier order and expansion."""
    ((order_idx,),) = onp.where(onp.all(expansion.basis_coefficients == order, axis=1))
    assert tuple(expansion.basis_coefficients[order_idx, :]) == order
    return int(order_idx)


def grating_efficiency(
    density: types.Density2DArray,
    spec: GratingSpec,
    wavelength: jnp.ndarray,
    polar_angle: jnp.ndarray,
    azimuthal_angle: jnp.ndarray,
    expansion: basis.Expansion,
    formulation: fmm.Formulation,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the per-order transmission and reflection efficiency for a grating.

    The excitation for the calculation are separate a TE- and TM-polarized plane waves
    at the specified wavelength(s), incident from the substrate with the specified
    polar and azimuthal angles.

    Args:
        density: Defines the pattern of the grating layer.
        spec: Defines the physical specifcation of the grating.
        wavelength: The wavelength of the excitation.
        polar_angle: The polar angle of the excitation.
        azimuthal_angle: The azimuthal angle of the excitation.
        expansion: Defines the Fourier expansion for the calculation.
        formulation: Defines the FMM formulation to be used.

    Returns:
        The per-order transmission and reflection efficiency, having shape
        `(num_wavelengths, num_fourier_terms, 2)`.
    """
    density_array = utils.transforms.rescaled_density_array(
        density,
        lower_bound=DENSITY_LOWER_BOUND,
        upper_bound=DENSITY_UPPER_BOUND,
    )
    permittivities = (
        jnp.full((1, 1), spec.permittivity_ambient),
        utils.transforms.interpolate_permittivity(
            permittivity_solid=jnp.asarray(spec.permittivity_grating),
            permittivity_void=jnp.asarray(spec.permittivity_encapsulation),
            density=density_array,
        ),
        jnp.full((1, 1), spec.permittivity_substrate),
    )

    in_plane_wavevector = basis.plane_wave_in_plane_wavevector(
        wavelength=wavelength,
        polar_angle=polar_angle,
        azimuthal_angle=azimuthal_angle,
        permittivity=jnp.asarray(spec.permittivity_ambient),
    )
    layer_solve_results = [
        fmm.eigensolve_isotropic_media(
            wavelength=jnp.asarray(wavelength),
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=basis.LatticeVectors(
                u=spec.period_x * basis.X,
                v=spec.period_y * basis.Y,
            ),
            permittivity=p,
            expansion=expansion,
            formulation=formulation,
        )
        for p in permittivities
    ]

    # Layer thicknesses for the ambient and substrate are set to zero; these do not
    # affect the result of the calculation.
    layer_thicknesses = (
        jnp.zeros(()),
        jnp.asarray(spec.thickness_grating),
        jnp.zeros(()),
    )

    s_matrix = scattering.stack_s_matrix(layer_solve_results, layer_thicknesses)

    n = expansion.num_terms
    assert tuple(expansion.basis_coefficients[0, :]) == (0, 0)
    assert expansion.basis_coefficients.shape[0] == n

    # Generate the wave amplitudes for backward-going TE or TM-polarized plane waves
    # at the end of substrate layer.
    bwd_amplitude_substrate_end = jnp.zeros((2 * n, 2), dtype=complex)
    bwd_amplitude_substrate_end = bwd_amplitude_substrate_end.at[0, 0].set(1.0)  # TE
    bwd_amplitude_substrate_end = bwd_amplitude_substrate_end.at[n, 1].set(1.0)  # TM

    # Calculate the incident power from the substrate. Since the substrate thickness
    # has been set to zero, the forward and backward amplitudes are already colocated.
    fwd_amplitude_substrate_start = s_matrix.s12 @ bwd_amplitude_substrate_end
    fwd_flux_substrate, bwd_flux_substrate = fields.amplitude_poynting_flux(
        forward_amplitude=fwd_amplitude_substrate_start,
        backward_amplitude=bwd_amplitude_substrate_end,
        layer_solve_result=layer_solve_results[-1],
    )

    # Sum over orders and polarizations to get the total incident flux.
    total_incident_flux = jnp.sum(bwd_flux_substrate, axis=-2, keepdims=True)

    # Calculate the transmitted power in the ambient.
    bwd_amplitude_ambient_end = s_matrix.s22 @ bwd_amplitude_substrate_end
    _, bwd_flux_ambient = fields.amplitude_poynting_flux(
        forward_amplitude=jnp.zeros_like(bwd_amplitude_ambient_end),
        backward_amplitude=bwd_amplitude_ambient_end,
        layer_solve_result=layer_solve_results[0],
    )

    # Sum the fluxes over the two polarizations for each order. Note that the sum is
    # done for each excitation condition (i.e. TE and TM), so as to capture the
    # effect of polarization conversion.
    bwd_flux_ambient = bwd_flux_ambient[..., :n, :] + bwd_flux_ambient[..., n:, :]
    fwd_flux_substrate = fwd_flux_substrate[..., :n, :] + fwd_flux_substrate[..., n:, :]

    transmission_efficiency = bwd_flux_ambient / total_incident_flux
    reflection_efficiency = fwd_flux_substrate / total_incident_flux

    return transmission_efficiency, reflection_efficiency
