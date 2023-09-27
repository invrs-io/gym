import dataclasses
from typing import Any, Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as onp
from totypes import symmetry, types  # type: ignore[import,attr-defined,unused-ignore]

from fmmax import basis

from invrs_gym.challenge.metagrating import component as metagrating_component

AuxDict = Dict[str, Any]
DensityInitializer = Callable[[jax.Array, types.Density2DArray], types.Density2DArray]

# 80 nm with the default dimensions of 1.371 x 0.525 um and grid shape of (118, 45).
MINIMUM_WIDTH = 7
MINIMUM_SPACING = 7
TRANSMISSION_ORDER = (1, 0)
TRANSMISSION_LOWER_BOUND = 0.9

DISTANCE_TO_WINDOW = "distance_to_window"


def identity_initializer(
    key: jax.Array, seed_density: types.Density2DArray
) -> types.Density2DArray:
    """A basic identity initializer which returns the seed density."""
    del key
    return seed_density


@dataclasses.dataclass
class MetagratingChallenge:
    """Defines a general ceviche challenge.

    The objective of the ceviche challenge is to find a component that whose
    transmission into its various ports lies within the target window defined
    by the transmission lower and upper bounds.

    Attributes:
        component: The component to be designed.
        transmission_order: The transmission diffraction order to be maximized.
        transmission_lower_bound: The lower bound for transmission into the specified
            order. When this value is exceeded, the challenge is considered solved.
    """

    component: metagrating_component.MetagratingComponent
    transmission_order: Tuple[int, int]
    transmission_lower_bound: float

    def loss(self, response: metagrating_component.MetagratingResponse) -> jnp.ndarray:
        """Compute a scalar loss from the component `response`."""
        transmission_efficiency = _value_for_order(
            response.transmission_efficiency,
            expansion=response.expansion,
            order=self.transmission_order,
        )
        return jnp.mean(1 - transmission_efficiency)

    def metrics(
        self,
        response: metagrating_component.MetagratingResponse,
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
    array: metagrating_component.MetagratingResponse,
    expansion: basis.Expansion,
    order: Tuple[int, int],
) -> jnp.ndarray:
    """Extracts the value from `array` for the specified Fourier order."""
    assert array.shape[-2] == expansion.num_terms
    ((order_idx,),) = onp.where(onp.all(expansion.basis_coefficients == order, axis=1))
    assert tuple(expansion.basis_coefficients[order_idx, :]) == order
    return array[..., order_idx, :]


def metagrating(
    minimum_width: int = MINIMUM_WIDTH,
    minimum_spacing: int = MINIMUM_SPACING,
    density_initializer: DensityInitializer = identity_initializer,
    transmission_order: Tuple[int, int] = TRANSMISSION_ORDER,
    transmission_lower_bound: float = TRANSMISSION_LOWER_BOUND,
) -> MetagratingChallenge:
    """Metagrating with 1.371 x 0.525 um design region."""
    return MetagratingChallenge(
        component=metagrating_component.MetagratingComponent(
            spec=metagrating_component.MetagratingSpec(),
            sim_params=metagrating_component.MetagratingSimParams(),
            density_initializer=density_initializer,
            minimum_width=minimum_width,
            minimum_spacing=minimum_spacing,
            symmetries=(symmetry.REFLECTION_E_W,),
        ),
        transmission_order=transmission_order,
        transmission_lower_bound=transmission_lower_bound,
    )
