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

TE = "te"
TM = "tm"

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
        grid_shape: The shape of the grid on which the permittivity is defined.
        wavelength: The wavelength of the excitation.
        polarization: The polarization of the excitation, TE or TM.
        formulation: The FMM formulation to be used.
        approximate_num_terms: Defines the number of terms in the Fourier expansion.
        truncation: Determines how the Fourier basis is truncated.
    """

    grid_shape: Tuple[int, int]
    wavelength: float | jnp.ndarray
    polarization: str
    formulation: fmm.Formulation
    approximate_num_terms: int
    truncation: basis.Truncation


@dataclasses.dataclass
class GratingResponse:
    """Contains the response of the grating.

    Attributes:
        wavelength: The wavelength for the efficiency calculation.
        transmission_efficiency: The per-order and per-wavelength coupling efficiency
            with which the excitation is transmitted.
        transmission_efficiency: The per-order and per-wavelength coupling efficiency
            with which the excitation is reflected.
        expansion: Defines the Fourier expansion for the calculation.
    """

    wavelength: jnp.ndarray
    transmission_efficiency: jnp.ndarray
    reflection_efficiency: jnp.ndarray
    expansion: basis.Expansion


json_utils.register_custom_type(GratingResponse)

tree_util.register_pytree_node(
    GratingResponse,
    lambda r: (
        (
            r.wavelength,
            r.transmission_efficiency,
            r.reflection_efficiency,
            r.expansion,
        ),
        None,
    ),
    lambda _, children: GratingResponse(*children),
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
    polarization: str,
    expansion: basis.Expansion,
    formulation: fmm.Formulation,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the per-order transmission and reflection efficiency for a grating.

    The excitation for the calculation is a TE- or TM-polarized plane wave at the
    specified wavelength(s), incident from the substrate.

    Args:
        density: Defines the pattern of the grating layer.
        spec: Defines the physical specifcation of the grating.
        wavelength: The wavelength of the excitation.
        polarization: The polarization of the excitation, TE or TM.
        expansion: Defines the Fourier expansion for the calculation.
        formulation: Defines the FMM formulation to be used.

    Returns:
        The per-order transmission and reflection efficiency, having shape
        `(num_wavelengths, nkx, nky)`.
    """
    if polarization not in (TE, TM):
        raise ValueError(
            f"`polarization` must be one of {(TE, TM)} but got {polarization}."
        )

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

    layer_solve_results = [
        fmm.eigensolve_isotropic_media(
            wavelength=jnp.asarray(wavelength),
            in_plane_wavevector=jnp.zeros((2,)),  # normal incidence
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
    bwd_amplitude_silica_end = jnp.zeros((2 * n, 1), dtype=complex)
    if polarization == TE:
        bwd_amplitude_silica_end = bwd_amplitude_silica_end.at[0, 0].set(1.0)
    else:
        bwd_amplitude_silica_end = bwd_amplitude_silica_end.at[n, 0].set(1.0)

    # Calculate the incident power in the silca. Since the substrate thickness has
    # been set to zero, the forward and backward amplitudes are already colocated.
    fwd_amplitude_silica_start = s_matrix.s12 @ bwd_amplitude_silica_end
    fwd_flux_silica, bwd_flux_silica = fields.amplitude_poynting_flux(
        forward_amplitude=fwd_amplitude_silica_start,
        backward_amplitude=bwd_amplitude_silica_end,
        layer_solve_result=layer_solve_results[-1],
    )

    # Sum over orders and polarizations to get the total incident flux.
    total_incident_flux = jnp.sum(bwd_flux_silica, axis=-2, keepdims=True)

    # Calculate the transmitted power in the ambient.
    bwd_amplitude_ambient_end = s_matrix.s22 @ bwd_amplitude_silica_end
    _, bwd_flux_ambient = fields.amplitude_poynting_flux(
        forward_amplitude=jnp.zeros_like(bwd_amplitude_ambient_end),
        backward_amplitude=bwd_amplitude_ambient_end,
        layer_solve_result=layer_solve_results[0],
    )

    # Sum the fluxes over the two polarizations for each order.
    bwd_flux_ambient = bwd_flux_ambient[..., :n, :] + bwd_flux_ambient[..., n:, :]
    fwd_flux_silica = fwd_flux_silica[..., :n, :] + fwd_flux_silica[..., n:, :]

    transmission_efficiency = bwd_flux_ambient / total_incident_flux
    reflection_efficiency = fwd_flux_silica / total_incident_flux

    return transmission_efficiency, reflection_efficiency
