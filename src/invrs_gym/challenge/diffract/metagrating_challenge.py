"""Defines the metagrating challenge."""

import dataclasses
from typing import Any, Callable, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from fmmax import basis, fmm  # type: ignore[import]
from totypes import symmetry, types  # type: ignore[import,attr-defined,unused-ignore]

from invrs_gym.challenge.diffract import common

AuxDict = Dict[str, Any]
DensityInitializer = Callable[[jax.Array, types.Density2DArray], types.Density2DArray]


DISTANCE_TO_WINDOW = "distance_to_window"


class MetagratingComponent:
    """Defines a metagrating component."""

    def __init__(
        self,
        spec: common.GratingSpec,
        sim_params: common.GratingSimParams,
        density_initializer: DensityInitializer,
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
        self.seed_density = common.seed_density(
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
    ) -> Tuple[common.GratingResponse, AuxDict]:
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
        transmission_efficiency, reflection_efficiency = common.grating_efficiency(
            density_array=params.array,
            thickness=jnp.asarray(self.spec.thickness_grating),
            spec=self.spec,
            wavelength=jnp.asarray(wavelength),
            polarization=self.sim_params.polarization,
            expansion=expansion,
            formulation=self.sim_params.formulation,
        )
        response = common.GratingResponse(
            wavelength=jnp.asarray(wavelength),
            transmission_efficiency=transmission_efficiency,
            reflection_efficiency=reflection_efficiency,
            expansion=expansion,
        )
        return response, {}


@dataclasses.dataclass
class MetagratingChallenge:
    """Defines the metagrating challenge.

    The objective of the metagrating challenge is to design a density so that incident
    light is efficiently diffracted into the specified transmission order. The
    challenge is considered solved when the transmission is above the lower bound.

    Attributes:
        component: The component to be designed.
        transmission_order: The transmission diffraction order to be maximized.
        transmission_lower_bound: The lower bound for transmission into the specified
            order. When this value is exceeded, the challenge is considered solved.
    """

    component: MetagratingComponent
    transmission_order: Tuple[int, int]
    transmission_lower_bound: float

    def loss(self, response: common.GratingResponse) -> jnp.ndarray:
        """Compute a scalar loss from the component `response`."""
        efficiency = _value_for_order(
            response.transmission_efficiency,
            expansion=response.expansion,
            order=self.transmission_order,
        )
        return jnp.mean(1 - jnp.abs(jnp.sqrt(efficiency)))

    def metrics(
        self,
        response: common.GratingResponse,
        params: types.Density2DArray,
        aux: AuxDict,
    ) -> AuxDict:
        """Compute challenge metrics."""
        del params, aux
        efficiency = _value_for_order(
            response.transmission_efficiency,
            expansion=response.expansion,
            order=self.transmission_order,
        )
        elementwise_distance_to_window = jnp.maximum(
            0, self.transmission_lower_bound - efficiency
        )
        return {DISTANCE_TO_WINDOW: jnp.linalg.norm(elementwise_distance_to_window)}


def _value_for_order(
    array: jnp.ndarray,
    expansion: basis.Expansion,
    order: Tuple[int, int],
) -> jnp.ndarray:
    """Extracts the value from `array` for the specified Fourier order."""
    order_idx = common.index_for_order(order, expansion)
    return array[..., order_idx, :]


# -----------------------------------------------------------------------------
# Metagrating with 1.371 x 0.525 um design region.
# -----------------------------------------------------------------------------


METAGRATING_SPEC = common.GratingSpec(
    permittivity_ambient=(1.0 + 0.0j) ** 2,
    permittivity_grating=(3.45 + 0.00001j) ** 2,
    permittivity_encapsulation=(1.0 + 0.00001j) ** 2,
    permittivity_substrate=(1.45 + 0.0j) ** 2,
    thickness_grating=0.325,
    period_x=float(1.050 / jnp.sin(jnp.deg2rad(50.0))),
    period_y=0.525,
)

METAGRATING_SIM_PARAMS = common.GratingSimParams(
    grid_shape=(118, 45),
    wavelength=1.050,
    polarization=common.TM,
    formulation=fmm.Formulation.JONES_DIRECT,
    approximate_num_terms=200,
    truncation=basis.Truncation.CIRCULAR,
)

SYMMETRIES = (symmetry.REFLECTION_E_W,)

# Objective is to diffract light into the +1 transmitted order, with efficiency better
# than 95 percent.
TRANSMISSION_ORDER = (1, 0)
TRANSMISSION_LOWER_BOUND = 0.95


def metagrating(
    minimum_width: int = 7,
    minimum_spacing: int = 7,
    density_initializer: DensityInitializer = common.identity_initializer,
    transmission_order: Tuple[int, int] = TRANSMISSION_ORDER,
    transmission_lower_bound: float = TRANSMISSION_LOWER_BOUND,
    spec: common.GratingSpec = METAGRATING_SPEC,
    sim_params: common.GratingSimParams = METAGRATING_SIM_PARAMS,
) -> MetagratingChallenge:
    """Metagrating challenge with 1.371 x 0.525 um design region.

    The metagrating challenge is based on the metagrating example in "Validation and
    characterization of algorithms for photonics inverse design" by Chen et al.
    (in preparation).

    It involves maximizing diffraction of light transmitted from a silicon oxide
    substrate into the ambient using a patterned silicon metastructure. The excitation
    is TM-polarized plane wave with 1.05 micron wavelength.

    Args:
        minimum_width: The minimum width target for the challenge, in pixels. The
            physical minimum width is approximately 80 nm.
        minimum_spacing: The minimum spacing target for the challenge, in pixels.
        density_initializer: Callable which returns the initial density, given a
            key and seed density.
        transmission_order: The diffraction order to be maximized.
        transmission_lower_bound: The lower bound for transmission. When the lower
            bound is exceeded, the challenge is considered to be solved.
        spec: Defines the physical specification of the metagrating.
        sim_params: Defines the simulation settings of the metagrating.

    Returns:
        The `MetagratingChallenge`.
    """
    return MetagratingChallenge(
        component=MetagratingComponent(
            spec=spec,
            sim_params=sim_params,
            density_initializer=density_initializer,
            minimum_width=minimum_width,
            minimum_spacing=minimum_spacing,
            symmetries=SYMMETRIES,
        ),
        transmission_order=transmission_order,
        transmission_lower_bound=transmission_lower_bound,
    )


# -----------------------------------------------------------------------------
# Broadband metagrating with 1.807 x 0.874 um design region.
# -----------------------------------------------------------------------------


BROADBAND_METAGRATING_SPEC = common.GratingSpec(
    permittivity_ambient=(1.0 + 0.0j) ** 2,
    permittivity_grating=(3.5 + 0.0j) ** 2,
    permittivity_encapsulation=(1.0 + 0.0j) ** 2,
    permittivity_substrate=(1.45 + 0.0j) ** 2,
    thickness_grating=0.528,
    period_x=1.807,
    period_y=0.874,
)

BROADBAND_METAGRATING_SIM_PARAMS = common.GratingSimParams(
    grid_shape=(411, 199),
    wavelength=jnp.asarray((1.530, 1.538, 1.546, 1.554, 1.562, 1.570)),
    polarization=common.TM,
    formulation=fmm.Formulation.JONES_DIRECT,
    approximate_num_terms=400,
    truncation=basis.Truncation.CIRCULAR,
)


def broadband_metagrating(
    minimum_width: int = 41,
    minimum_spacing: int = 41,
    density_initializer: DensityInitializer = common.identity_initializer,
    transmission_order: Tuple[int, int] = TRANSMISSION_ORDER,
    transmission_lower_bound: float = TRANSMISSION_LOWER_BOUND,
    spec: common.GratingSpec = BROADBAND_METAGRATING_SPEC,
    sim_params: common.GratingSimParams = BROADBAND_METAGRATING_SIM_PARAMS,
) -> MetagratingChallenge:
    """Broadband metagrating challenge with 1.807 x 0.874 um design region.

    The broadband metagrating challenge is based on "General-purpose algorithm for
    two-material minimum feature size enforcement of nanophotonic devices" by Jenkins
    et al. (https://pubs.acs.org/doi/abs/10.1021/acsphotonics.2c01166).

    It involves maximizing diffraction of light transmitted from a silicon oxide
    substrate into the ambient using a patterned silicon metastructure. The excitation
    consists of TM-polarized plane waves with wavelengths near 1.550 microns.

    Args:
        minimum_width: The minimum width target for the challenge, in pixels. The
            physical minimum width is approximately 180 nm.
        minimum_spacing: The minimum spacing target for the challenge, in pixels.
        density_initializer: Callable which returns the initial density, given a
            key and seed density.
        transmission_order: The diffraction order to be maximized.
        transmission_lower_bound: The lower bound for transmission. When the lower
            bound is exceeded, the challenge is considered to be solved.
        spec: Defines the physical specification of the metagrating.
        sim_params: Defines the simulation settings of the metagrating.

    Returns:
        The `MetagratingChallenge`.
    """
    return MetagratingChallenge(
        component=MetagratingComponent(
            spec=spec,
            sim_params=sim_params,
            density_initializer=density_initializer,
            minimum_width=minimum_width,
            minimum_spacing=minimum_spacing,
            symmetries=SYMMETRIES,
        ),
        transmission_order=transmission_order,
        transmission_lower_bound=transmission_lower_bound,
    )
