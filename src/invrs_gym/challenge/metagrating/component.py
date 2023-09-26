"""Defines the metagrating challenges."""

import dataclasses
from typing import Any, Callable, Dict, Optional, Tuple, Union

import jax
from jax import tree_util
import jax.numpy as jnp
from fmmax import basis, fields, fmm, scattering, utils  # type: ignore[import]
from totypes import types  # type: ignore[import]

AuxDict = Dict[str, Any]
Params = Any
DensityInitializer = Callable[[jax.Array, types.Density2DArray], types.Density2DArray]

DENSITY_LOWER_BOUND = 0.0
DENSITY_UPPER_BOUND = 1.0


def identity_initializer(
    key: jax.Array, seed_density: types.Density2DArray
) -> types.Density2DArray:
    """A basic identity initializer which returns the seed density."""
    del key
    return seed_density


@dataclasses.dataclass
class MetagratingSpec:
    """Defines the physical specifcation of the metagrating.

    Args:
        permittivity_ambient: Permittivity of the ambient material.
        permittivity_silicon: Permittivity of the silicon metagrating.
        permittivity_silica: Permittivity of the silica substrate.
        thickness_silicon: Thickness of the silicon metagrating.
        period_x: The size of the unit cell along the x direction.
        period_y: The size of the unit cell along the y direction.
    """

    permittivity_ambient: complex = (1.0 + 0.0j) ** 2
    permittivity_silicon: complex = (3.45 + 0.0j) ** 2
    permittivity_silica: complex = (1.45 + 0.0j) ** 2

    thickness_silicon: float = 0.325

    period_x: float = float(1.050 / jnp.sin(jnp.deg2rad(50.0)))
    period_y: float = 0.525


@dataclasses.dataclass
class MetagratingSimParams:
    """Parameters that configure the simulation of the metagrating.

    Attributes:
        grid_shape: The shape of the grid that defines the permittivity distribution
            of the silicon metagrating.
        wavelength: The wavelength of the excitation.
        formulation: The FMM formulation to be used.
        approximate_num_terms: Defines the number of terms in the Fourier expansion.
        truncation: Determines how the Fourier basis is truncated.
    """

    grid_shape: Tuple[int, int] = (138, 55)
    wavelength: Union[float, jnp.ndarray] = 1.050
    formulation: fmm.Formulation = fmm.Formulation.JONES_DIRECT
    approximate_num_terms: int = 200
    truncation: basis.Truncation = basis.Truncation.CIRCULAR


@dataclasses.dataclass
class MetagratingResponse:
    """Contains the response of the metagrating.

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


tree_util.register_pytree_node(
    MetagratingResponse,
    lambda r: (
        (
            r.wavelength,
            r.transmission_efficiency,
            r.reflection_efficiency,
            r.expansion,
        ),
        None,
    ),
    lambda _, children: MetagratingResponse(*children),
)


class MetagratingComponent:
    """Defines a metagrating component."""

    def __init__(
        self,
        spec: MetagratingSpec,
        sim_params: MetagratingSimParams,
        density_initializer: DensityInitializer,
        **seed_density_kwargs: Any,
    ) -> None:
        """Initializes the metagrating component.

        Args:
            spec: Defines the physical specification of the metagrating.
            sim_params: Defines simulation parameters for the metagrating.
            density_initializer: Callable which generates the initial density from
                a random key and the seed density.
            **seed_density_kwargs: Keyword arguments which set the attributes of
                the seed density used to generate the inital parameters.
        """

        self.spec = spec
        self.sim_params = sim_params
        self.seed_density = _seed_density(
            self.sim_params.grid_shape, **seed_density_kwargs
        )
        self.density_initializer = density_initializer

        self.expansion = basis.generate_expansion(
            primitive_lattice_vectors=basis.LatticeVectors(
                u=self.spec.period_x * basis.X,
                v=self.spec.period_y * basis.Y,
            ),
            approximate_num_terms=self.sim_params.approximate_num_terms,
            truncation=self.sim_params.truncation,
        )

    def init(self, key: jax.Array) -> types.Density2DArray:
        """Return the initial parameters for the metagrating component."""
        return self.density_initializer(key, self.seed_density)

    def response(
        self,
        params: types.Density2DArray,
        wavelength: Optional[Union[float, jnp.ndarray]] = None,
        expansion: Optional[basis.Expansion] = None,
    ) -> Tuple[MetagratingResponse, AuxDict]:
        """Computes the response of the metagrating.

        Args:
            params: The parameters defining the metagrating, matching those returned
                by the `init` method.
            wavelength: Optional wavelength to override the default in `sim_params`.
            expansion: Optional expansion to override the default `expansion`.

        Returns:
            The `(response, aux)` tuple.
        """
        if expansion is None:
            expansion = self.expansion
        if wavelength is None:
            wavelength = self.sim_params.wavelength
        transmission_efficiency, reflection_efficiency = metagrating_efficiency(
            density_array=params.array,
            spec=self.spec,
            wavelength=jnp.asarray(wavelength),
            expansion=expansion,
            formulation=self.sim_params.formulation,
        )
        response = MetagratingResponse(
            wavelength=jnp.asarray(wavelength),
            transmission_efficiency=transmission_efficiency,
            reflection_efficiency=reflection_efficiency,
            expansion=expansion,
        )
        return response, {}


def _seed_density(grid_shape: Tuple[int, int], **kwargs: Any) -> types.Density2DArray:
    """Return the seed density for the ceviche component.

    The seed density has shape and fixed pixels as required by the `ceviche_model`,
    and with other properties determined by keyword arguments.

    Args:
        grid_shape: The shape of the grid on which the density is defined.
        kwargs: keyword arguments specifying additional properties of the seed
            density, e.g. symmetries.

    Returns:
        The seed density.
    """

    # Check kwargs that are required by the `CevicheComponent`,
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


def metagrating_efficiency(
    density_array: jnp.ndarray,
    spec: MetagratingSpec,
    wavelength: jnp.ndarray,
    expansion: basis.Expansion,
    formulation: fmm.Formulation,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the per-order transmission and reflection efficiency for a metagrating.

    The excitation for the calculation is a TM-polarized plane wave at the specified
    wavelength(s).

    Args:
        density_array: Defines the pattern of the silicon metagrating layer.
        spec: Defines the physical specifcation of the metagrating.
        wavelength: The wavelength of the excitation.
        expansion: Defines the Fourier expansion for the calculation.
        formulation: Defines the FMM formulation to be used.

    Returns:
        The per-order transmission and reflection efficiency, having shape
        `(num_wavelengths, nkx, nky)`.
    """

    permittivities = (
        jnp.full((1, 1), spec.permittivity_ambient),
        utils.interpolate_permittivity(
            permittivity_solid=jnp.asarray(spec.permittivity_silicon),
            permittivity_void=jnp.asarray(spec.permittivity_ambient),
            density=density_array,
        ),
        jnp.full((1, 1), spec.permittivity_silica),
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
        jnp.asarray(spec.thickness_silicon),
        jnp.zeros(()),
    )

    s_matrix = scattering.stack_s_matrix(layer_solve_results, layer_thicknesses)

    n = expansion.num_terms
    assert tuple(expansion.basis_coefficients[0, :]) == (0, 0)
    assert expansion.basis_coefficients.shape[0] == n

    # Generate the wave amplitudes for backward-going TM-polarized plane waves at the
    # end of silica layer.
    bwd_amplitude_silica_end = jnp.zeros((2 * n, 1), dtype=complex)
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
    total_incident_flux = jnp.sum(bwd_flux_silica, axis=-2)

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
