"""Defines functions common across diffract challenges.

Copyright (c) 2025 invrs.io LLC
"""

import dataclasses
from typing import Any, Optional, Tuple, Union

import fmmax
import jax
import jax.numpy as jnp
import numpy as onp
from jax import tree_util
from totypes import json_utils, types

from invrs_gym import utils
from invrs_gym.challenges import base

Params = Any

DENSITY_LOWER_BOUND = 0.0
DENSITY_UPPER_BOUND = 1.0

THICKNESS_CAP = "thickness_cap"
THICKNESS_GRATING = "thickness_grating"
THICKNESS_SPACER = "thickness_spacer"
DENSITY = "density"

EFIELD = "efield"
HFIELD = "hfield"
FIELD_COORDINATES = "field_coordinates"


@dataclasses.dataclass
class GratingSpec:
    """Defines the physical specifcation of a grating.

    Attributes:
        permittivity_ambient: Permittivity of the ambient material.
        permittivity_cap: Permittivity of the material between grating and ambient.
        permittivity_grating: Permittivity of the grating teeth.
        permittivity_encapsulation: Permittivity of the material in gaps between
            grating teeth.
        permittivity_spacer: Permittivity of the spacer layer between grating and
            substrate.
        permittivity_substrate: Permittivity of the substrate.
        thickness_cap: Thickness of the cap layer between grating and ambient.
        thickness_grating: Thickness of the grating layer.
        thickness_spacer: Thickness of the spacer layer between grating and substrate.
        period_x: The size of the unit cell along the x direction.
        period_y: The size of the unit cell along the y direction.
        grid_spacing: The spacing of the grid on which grating permittivity is defined.
    """

    permittivity_ambient: complex
    permittivity_cap: complex
    permittivity_grating: complex
    permittivity_encapsulation: complex
    permittivity_spacer: complex
    permittivity_substrate: complex

    thickness_ambient: float
    thickness_cap: float | jnp.ndarray | types.BoundedArray
    thickness_grating: float | jnp.ndarray | types.BoundedArray
    thickness_spacer: float | jnp.ndarray | types.BoundedArray
    thickness_substrate: float

    period_x: float
    period_y: float

    grid_spacing: float

    @property
    def grid_shape(self) -> Tuple[int, int]:
        """Return the shape of the grid implied by `grid_spacing`."""
        return (
            int(jnp.ceil(self.period_x / self.grid_spacing)),
            int(jnp.ceil(self.period_y / self.grid_spacing)),
        )


@dataclasses.dataclass
class GratingSimParams:
    """Parameters that configure the simulation of a grating.

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
    formulation: fmmax.Formulation
    approximate_num_terms: int
    truncation: fmmax.Truncation


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
    expansion: fmmax.Expansion


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


# -----------------------------------------------------------------------------
# Define components used in diffract optimization challenges.
# -----------------------------------------------------------------------------


class SimpleGratingComponent(base.Component):
    """A simple grating component whose only optimizable parameter is density."""

    def __init__(
        self,
        spec: GratingSpec,
        sim_params: GratingSimParams,
        density_initializer: base.DensityInitializer,
        **seed_density_kwargs: Any,
    ) -> None:
        """Initializes the grating component.

        Args:
            spec: Defines the physical specification of the grating.
            sim_params: Defines simulation parameters for the grating.
            density_initializer: Callable which generates the initial density from
                a random key and the seed density.
            **seed_density_kwargs: Keyword arguments which set the attributes of
                the seed density used to generate the inital parameters.
        """

        self.spec = spec
        self.sim_params = sim_params
        self.seed_density = seed_density(
            grid_shape=self.spec.grid_shape,
            **seed_density_kwargs,
        )
        self.density_initializer = density_initializer

        self.expansion = fmmax.generate_expansion(
            primitive_lattice_vectors=fmmax.LatticeVectors(
                u=self.spec.period_x * fmmax.X,
                v=self.spec.period_y * fmmax.Y,
            ),
            approximate_num_terms=self.sim_params.approximate_num_terms,
            truncation=self.sim_params.truncation,
        )

    def init(self, key: jax.Array) -> types.Density2DArray:
        """Return the initial parameters for the grating component."""
        params = self.density_initializer(key, self.seed_density)
        # Ensure that there are no weak types in the initial parameters.
        return tree_util.tree_map(
            lambda x: jnp.asarray(x, jnp.asarray(x).dtype), params
        )

    def response(
        self,
        params: types.Density2DArray,
        *,
        wavelength: Optional[Union[float, jnp.ndarray]] = None,
        expansion: Optional[fmmax.Expansion] = None,
        compute_fields: bool = False,
    ) -> Tuple[GratingResponse, base.AuxDict]:
        """Computes the response of the grating.

        The response consists of the transmitted and reflected power into each order
        for both TE- and TM-polarized plane wave illumination.

        Args:
            params: The parameters defining the metagrating, matching those returned
                by the `init` method.
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
        (transmission_efficiency, reflection_efficiency), aux = grating_efficiency(
            density=params,
            spec=self.spec,
            wavelength=jnp.asarray(wavelength),
            polar_angle=jnp.asarray(self.sim_params.polar_angle),
            azimuthal_angle=jnp.asarray(self.sim_params.azimuthal_angle),
            expansion=expansion,
            formulation=self.sim_params.formulation,
            compute_fields=compute_fields,
        )
        response = GratingResponse(
            wavelength=jnp.asarray(wavelength),
            polar_angle=jnp.asarray(self.sim_params.polar_angle),
            azimuthal_angle=jnp.asarray(self.sim_params.azimuthal_angle),
            transmission_efficiency=transmission_efficiency,
            reflection_efficiency=reflection_efficiency,
            expansion=expansion,
        )
        return response, aux


class GratingWithOptimizableThicknessComponent(base.Component):
    """A grating whose optimizable parameters are density and layer thickness."""

    def __init__(
        self,
        spec: GratingSpec,
        sim_params: GratingSimParams,
        thickness_initializer: base.ThicknessInitializer,
        density_initializer: base.DensityInitializer,
        **seed_density_kwargs: Any,
    ) -> None:
        """Initializes the grating component splitter component.

        Args:
            spec: Defines the physical specification of the grating.
            sim_params: Defines simulation parameters for the grating.
            thickness_initializer: Callable which returns the initial thickness for
                the grating layer from a random key and a bounded array with value
                equal the thickness from `spec`.
            density_initializer: Callable which generates the initial density from
                a random key and the seed density.
            **seed_density_kwargs: Keyword arguments which set the attributes of
                the seed density used to generate the inital parameters.
        """

        self.spec = spec
        self.sim_params = sim_params
        self.thickness_initializer = thickness_initializer
        self.density_initializer = density_initializer

        self.seed_density = seed_density(
            grid_shape=self.spec.grid_shape,
            **seed_density_kwargs,
        )

        self.expansion = fmmax.generate_expansion(
            primitive_lattice_vectors=fmmax.LatticeVectors(
                u=self.spec.period_x * fmmax.X,
                v=self.spec.period_y * fmmax.Y,
            ),
            approximate_num_terms=self.sim_params.approximate_num_terms,
            truncation=self.sim_params.truncation,
        )

    def init(self, key: jax.Array) -> Params:
        """Return the initial parameters for the grating component."""
        keys = iter(jax.random.split(key, num=4))
        params = {
            THICKNESS_CAP: self.thickness_initializer(
                next(keys), self.spec.thickness_cap  # type: ignore[arg-type]
            ),
            THICKNESS_GRATING: self.thickness_initializer(
                next(keys), self.spec.thickness_grating  # type: ignore[arg-type]
            ),
            THICKNESS_SPACER: self.thickness_initializer(
                next(keys), self.spec.thickness_spacer  # type: ignore[arg-type]
            ),
            DENSITY: self.density_initializer(next(keys), self.seed_density),
        }
        # Ensure that there are no weak types in the initial parameters.
        return tree_util.tree_map(
            lambda x: jnp.asarray(x, jnp.asarray(x).dtype), params
        )

    def response(
        self,
        params: Params,
        *,
        wavelength: Optional[Union[float, jnp.ndarray]] = None,
        expansion: Optional[fmmax.Expansion] = None,
        compute_fields: bool = False,
    ) -> Tuple[GratingResponse, base.AuxDict]:
        """Computes the response of the grating component.

        The response consists of the transmitted and reflected power into each order
        for both TE- and TM-polarized plane wave illumination.

        Args:
            params: The parameters defining the grating, matching those returned by
                the `init` method.
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
        spec = dataclasses.replace(
            self.spec,
            thickness_cap=jnp.asarray(params[THICKNESS_CAP].array),
            thickness_grating=jnp.asarray(params[THICKNESS_GRATING].array),
            thickness_spacer=jnp.asarray(params[THICKNESS_SPACER].array),
        )
        (transmission_efficiency, reflection_efficiency), aux = grating_efficiency(
            density=params[DENSITY],  # type: ignore[arg-type]
            spec=spec,
            wavelength=jnp.asarray(wavelength),
            polar_angle=jnp.asarray(self.sim_params.polar_angle),
            azimuthal_angle=jnp.asarray(self.sim_params.azimuthal_angle),
            expansion=expansion,
            formulation=self.sim_params.formulation,
            compute_fields=compute_fields,
        )
        response = GratingResponse(
            wavelength=jnp.asarray(wavelength),
            polar_angle=jnp.asarray(self.sim_params.polar_angle),
            azimuthal_angle=jnp.asarray(self.sim_params.azimuthal_angle),
            transmission_efficiency=transmission_efficiency,
            reflection_efficiency=reflection_efficiency,
            expansion=expansion,
        )
        return response, aux


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
    expansion: fmmax.Expansion,
) -> int:
    """Returns the index for the specified Fourier order and expansion."""
    ((order_idx,),) = onp.where(onp.all(expansion.basis_coefficients == order, axis=1))
    assert tuple(expansion.basis_coefficients[order_idx, :]) == order
    return int(order_idx)


# -----------------------------------------------------------------------------
# Simulation method used by all gratings.
# -----------------------------------------------------------------------------


def grating_efficiency(
    density: types.Density2DArray,
    spec: GratingSpec,
    wavelength: jnp.ndarray,
    polar_angle: jnp.ndarray,
    azimuthal_angle: jnp.ndarray,
    expansion: fmmax.Expansion,
    formulation: fmmax.Formulation,
    compute_fields: bool,
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], base.AuxDict]:
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
        compute_fields: If `True`, computes fields in an xz cross section.

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
        jnp.full((1, 1), spec.permittivity_cap),
        utils.transforms.interpolate_permittivity(
            permittivity_solid=jnp.asarray(spec.permittivity_grating),
            permittivity_void=jnp.asarray(spec.permittivity_encapsulation),
            density=density_array,
        ),
        jnp.full((1, 1), spec.permittivity_spacer),
        jnp.full((1, 1), spec.permittivity_substrate),
    )

    in_plane_wavevector = fmmax.plane_wave_in_plane_wavevector(
        wavelength=wavelength,
        polar_angle=polar_angle,
        azimuthal_angle=azimuthal_angle,
        # Polar angle is defined in substrate, since light is incident from substrate.
        permittivity=jnp.asarray(spec.permittivity_substrate),
    )
    layer_solve_results = [
        fmmax.eigensolve_isotropic_media(
            wavelength=jnp.asarray(wavelength),
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=fmmax.LatticeVectors(
                u=spec.period_x * fmmax.X,
                v=spec.period_y * fmmax.Y,
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
        jnp.asarray(spec.thickness_ambient),
        jnp.asarray(spec.thickness_cap),
        jnp.asarray(spec.thickness_grating),
        jnp.asarray(spec.thickness_spacer),
        jnp.asarray(spec.thickness_substrate),
    )

    if compute_fields:
        # If fields wanted, compute the full set of interior scattering matrices.
        s_matrices_interior = fmmax.stack_s_matrices_interior(
            layer_solve_results=layer_solve_results,
            layer_thicknesses=layer_thicknesses,
        )
        s_matrix = s_matrices_interior[-1][0]
    else:
        s_matrix = fmmax.stack_s_matrix(
            layer_solve_results=layer_solve_results,
            layer_thicknesses=layer_thicknesses,
        )

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
    fwd_flux_substrate, bwd_flux_substrate = fmmax.amplitude_poynting_flux(
        forward_amplitude=fwd_amplitude_substrate_start,
        backward_amplitude=bwd_amplitude_substrate_end,
        layer_solve_result=layer_solve_results[-1],
    )

    # Sum over orders and polarizations to get the total incident flux.
    total_incident_flux = jnp.sum(bwd_flux_substrate, axis=-2, keepdims=True)

    # Calculate the transmitted power in the ambient.
    bwd_amplitude_ambient_end = s_matrix.s22 @ bwd_amplitude_substrate_end
    _, bwd_flux_ambient = fmmax.amplitude_poynting_flux(
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

    # -------------------------------------------------------------------------
    # Compute fields in an xz cross section.
    # -------------------------------------------------------------------------

    aux = {}
    if compute_fields:
        amplitudes_interior = fmmax.stack_amplitudes_interior(
            s_matrices_interior=s_matrices_interior,
            forward_amplitude_0_start=jnp.zeros_like(bwd_amplitude_substrate_end),
            backward_amplitude_N_end=bwd_amplitude_substrate_end,
        )
        x = jnp.linspace(0, spec.period_x, density.shape[0])
        y = jnp.ones_like(x) * spec.period_y / 2
        layer_znum = tuple(
            [int(jnp.round(t / spec.grid_spacing) + 1) for t in layer_thicknesses]
        )
        (ex, ey, ez), (hx, hy, hz), (x, y, z) = fmmax.stack_fields_3d_on_coordinates(
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

    return (transmission_efficiency, reflection_efficiency), aux
