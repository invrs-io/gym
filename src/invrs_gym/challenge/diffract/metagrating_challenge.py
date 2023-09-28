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
        transmission_efficiency = _value_for_order(
            response.transmission_efficiency,
            expansion=response.expansion,
            order=self.transmission_order,
        )
        return jnp.mean(jnp.sqrt(1 - transmission_efficiency))

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
    permittivity_grating=(3.45 + 0.0j) ** 2,
    permittivity_encapsulation=(1.0 + 0.0j) ** 2,
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

# Minimum width and spacing are approximately 80 nm for the default dimensions
# of 1.371 x 0.525 um and grid shape of (118, 45).
MINIMUM_WIDTH = 7
MINIMUM_SPACING = 7

# Objective is to diffract light into the +1 transmitted order, with efficiency better
# than 95 percent.
TRANSMISSION_ORDER = (1, 0)
TRANSMISSION_LOWER_BOUND = 0.95


def metagrating(
    minimum_width: int = MINIMUM_WIDTH,
    minimum_spacing: int = MINIMUM_SPACING,
    density_initializer: DensityInitializer = common.identity_initializer,
    transmission_order: Tuple[int, int] = TRANSMISSION_ORDER,
    transmission_lower_bound: float = TRANSMISSION_LOWER_BOUND,
) -> MetagratingChallenge:
    """Metagrating with 1.371 x 0.525 um design region."""
    return MetagratingChallenge(
        component=MetagratingComponent(
            spec=METAGRATING_SPEC,
            sim_params=METAGRATING_SIM_PARAMS,
            density_initializer=density_initializer,
            minimum_width=minimum_width,
            minimum_spacing=minimum_spacing,
            symmetries=(symmetry.REFLECTION_E_W,),
        ),
        transmission_order=transmission_order,
        transmission_lower_bound=transmission_lower_bound,
    )
