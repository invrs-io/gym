"""Defines loss functions that target transmission values into various output ports."""

from typing import Tuple

import jax
import jax.numpy as jnp

_MIN_PHYSICAL_TRANSMISSION = 0.0
_MAX_PHYSICAL_TRANSMISSION = 1.0


def orthotope_smooth_transmission_loss(
    transmission: jnp.ndarray,
    window_lower_bound: jnp.ndarray,
    window_upper_bound: jnp.ndarray,
    transmission_exponent: jnp.ndarray,
    scalar_exponent: jnp.ndarray,
) -> jnp.ndarray:
    """Compute a scalar loss from a array based on an orthotope transmission window.

    The loss is related to an orthotope window, i.e. a orthotope target region within
    the space that contains valid responses. With values for `transmission_exponent`
    and `scalar_exponent` of `0.5` and `2.0`, respectively, this loss function is
    equivalent to that of [2022 Schubert](https://arxiv.org/abs/2201.12965).

    Args:
        array: The array for which the loss is to be calculated.
        window_lower_bound: Array giving the lower bound of the target transmission
            window. Must be broadcast compatible with `transmission`.
        window_upper_bound: Defines the upper bound of the target window.
        exponent: Exponent applied to the final scalar loss.

    Returns:
        The scalar loss value.
    """
    # The signed distance to the target window is positive
    elementwise_signed_distance = elementwise_signed_distance_to_window(
        transmission=transmission**transmission_exponent,
        window_lower_bound=window_lower_bound**transmission_exponent,
        window_upper_bound=window_upper_bound**transmission_exponent,
    )

    window_size = jnp.minimum(
        window_upper_bound, _MAX_PHYSICAL_TRANSMISSION
    ) - jnp.maximum(window_lower_bound, _MIN_PHYSICAL_TRANSMISSION)

    transformed_elementwise_signed_distance = jax.nn.softplus(
        elementwise_signed_distance / jnp.amin(window_size)
    )

    return jnp.linalg.norm(transformed_elementwise_signed_distance) ** scalar_exponent


def distance_to_window(
    transmission: jnp.ndarray,
    window_lower_bound: jnp.ndarray,
    window_upper_bound: jnp.ndarray,
) -> jnp.ndarray:
    """Returns the distance to a transmission window.

    When the transmission lies inside the transmission window, the distance to
    the window is zero.

    Args:
        transmission: The transmission for which the signed distance is sought.
        window_lower_bound: Array defining the transmission window lower bound.
        window_upper_bound: Array defining the transmission window upper bound.

    Returns:
        The elementwise signed distance.
    """
    elementwise_signed_distance = elementwise_signed_distance_to_window(
        transmission=transmission,
        window_lower_bound=window_lower_bound,
        window_upper_bound=window_upper_bound,
    )
    elementwise_distance = jnp.maximum(elementwise_signed_distance, 0.0)
    return jnp.linalg.norm(elementwise_distance)


def elementwise_signed_distance_to_window(
    transmission: jnp.ndarray,
    window_lower_bound: jnp.ndarray,
    window_upper_bound: jnp.ndarray,
) -> jnp.ndarray:
    """Returns the elementwise signed distance to a transmission window.

    Args:
        transmission: The transmission for which the signed distance is sought.
        window_lower_bound: Array defining the transmission window lower bound.
        window_upper_bound: Array defining the transmission window upper bound.

    Returns:
        The elementwise signed distance.
    """
    # Signed distance to lower bound is positive when `transmission` is below the lower
    # bound and negative when it is above the lower bound. When the lower bound is less
    # or equal to the minimum physical transmission, disregard the distance entirely
    # and substitute a value equal to the signed distance to the upper bound when the
    # response value is equal to the lower bound.
    elementwise_signed_distance_to_lower_bound = window_lower_bound - transmission
    elementwise_signed_distance_to_lower_bound = jnp.where(
        window_lower_bound > _MIN_PHYSICAL_TRANSMISSION,
        elementwise_signed_distance_to_lower_bound,
        -(_MAX_PHYSICAL_TRANSMISSION - _MIN_PHYSICAL_TRANSMISSION),
    )

    # Signed distance to upper bound defined similarly to the lower bound.
    elementwise_signed_distance_to_upper_bound = transmission - window_upper_bound
    elementwise_signed_distance_to_upper_bound = jnp.where(
        window_upper_bound < _MAX_PHYSICAL_TRANSMISSION,
        elementwise_signed_distance_to_upper_bound,
        -(_MAX_PHYSICAL_TRANSMISSION - _MIN_PHYSICAL_TRANSMISSION),
    )

    return jnp.maximum(
        elementwise_signed_distance_to_lower_bound,
        elementwise_signed_distance_to_upper_bound,
    )
