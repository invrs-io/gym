"""Defines loss functions that target transmission values into various output ports.

Copyright (c) 2023 The INVRS-IO authors.
"""

from typing import Optional, Sequence

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
    axis: Optional[int | Sequence[int]] = None,
) -> jnp.ndarray:
    """Compute a scalar loss from a array based on an orthotope transmission window.

    The loss is related to an orthotope window in tansmission space, i.e. the space
    of the squared magnitude of scattering parameters. With values for
    `transmission_exponent` and `scalar_exponent` of `1.0` and `2.0`, respectively,
    this loss function is equivalent to that of [2022 Schubert]
    (https://arxiv.org/abs/2201.12965).

    Args:
        transmission: The transmission array for which the loss is to be calculated.
        window_lower_bound: Array giving the lower bound of the target transmission
            window. Must be broadcast compatible with `transmission`.
        window_upper_bound: Defines the upper bound of the target window.
        transmission_exponent: Exponent applied to the transmission and window
            bounds prior to scalarization.
        scalar_exponent: Exponent applied to the final scalar loss.
        axis: The axes for which scalarization is sought. Default is `None`, which
            means that a scalar is returned.

    Returns:
        The loss value.
    """
    # Compute the signed psuedodistance. This is equal to the signed distance to the
    # nearest bound, except when the bounds are the min and max physical transmission
    # values, in which case the distance is equal to the window size.
    elementwise_signed_psuedodistance = elementwise_signed_psuedodistance_to_window(
        transmission=transmission**transmission_exponent,
        window_lower_bound=window_lower_bound**transmission_exponent,
        window_upper_bound=window_upper_bound**transmission_exponent,
    )

    # Scale the signed distance by the maximum window size.
    lower_bound = jnp.maximum(window_lower_bound, _MIN_PHYSICAL_TRANSMISSION)
    upper_bound = jnp.minimum(window_upper_bound, _MAX_PHYSICAL_TRANSMISSION)
    window_size = upper_bound - lower_bound
    elementwise_signed_psuedodistance /= jnp.amin(window_size)

    transformed_elementwise_signed_distance = jax.nn.softplus(
        elementwise_signed_psuedodistance
    )

    loss: jnp.ndarray = (
        _l2_norm(transformed_elementwise_signed_distance, axis=axis) ** scalar_exponent
    )
    return loss


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
    elementwise_signed_distance = elementwise_signed_psuedodistance_to_window(
        transmission=transmission,
        window_lower_bound=window_lower_bound,
        window_upper_bound=window_upper_bound,
    )
    elementwise_distance = jnp.maximum(elementwise_signed_distance, 0.0)
    distance: jnp.ndarray = jnp.linalg.norm(elementwise_distance)
    return distance


def elementwise_signed_psuedodistance_to_window(
    transmission: jnp.ndarray,
    window_lower_bound: jnp.ndarray,
    window_upper_bound: jnp.ndarray,
) -> jnp.ndarray:
    """Returns the elementwise signed psuedodistance to a transmission window.

    The psuedodistance is given by,

        - the distance to the nearest bound, when both bounds are within the
          window (e.g. the lower bound is greater than the minimum physical
          transmission value).
        - the distance to the upper bound, when the lower bound is less than or
          equal to the minimum physical transmission value, and the upper bound
          is less than the maximum physical transmission value.
        - the distance to the lower bound, when the upper bound is greater than
          or equal to the maximum physicall transmission value, and the lower
          bound is greater than the minimum physical transmission value.
        - the difference between the minimum and  maximum physical transmission
          value, when both bounds equal or exceed their physical extremal values.
          In this case, the psuedodistance has no dependence on `transmission`.

    Args:
        transmission: The transmission for which the signed distance is sought.
        window_lower_bound: Array defining the transmission window lower bound.
        window_upper_bound: Array defining the transmission window upper bound.

    Returns:
        The elementwise signed psuedodistance.
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


def _l2_norm(x: jnp.ndarray, axis: Optional[int | Sequence[int]]) -> jnp.ndarray:
    """Compute the L2-norm for the specified axes."""
    if axis is None:
        axis = tuple(range(x.ndim))
    elif isinstance(axis, int):
        axis = (axis,)

    x = jnp.moveaxis(x, axis, tuple(range(len(axis))))
    x = x.reshape((-1,) + x.shape[len(axis) :])
    return jnp.linalg.norm(x, axis=0)
