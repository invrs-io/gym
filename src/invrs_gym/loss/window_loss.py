"""Defines loss functions that target windows in some output space."""

import jax
import jax.numpy as jnp


def orthotope_window_loss(
    response: jnp.ndarray,
    window_lower_bound: jnp.ndarray,
    window_upper_bound: jnp.ndarray,
    space_lower_bound: jnp.ndarray,
    space_upper_bound: jnp.ndarray,
    exponent: jnp.ndarray,
) -> jnp.ndarray:
    """Compute a scalar loss from a response array based on an orthotope window.

    The loss is related to an orthotope window, i.e. a orthotope target region within
    the space that contains valid responses.

    Args:
        response: The response array.
        window_lower_bound: Array giving the lower bound of the target window. Must be
            broadcast compatible with `response`.
        window_upper_bound: Defines the upper bound of the target window.
        space_lower_bound: Defines the lower bound of the response space, i.e. the
            vector space in which the response lies. Must be broadcast compatible with
            `response`.
        space_upper_bound: Defines the upper bound of the response space.
        loss_exponent: Exponent applied to the final scalar loss.

    Returns:
        The scalar loss value.
    """
    window_dim = jnp.minimum(window_upper_bound, space_upper_bound) - jnp.maximum(
        window_lower_bound, space_lower_bound
    )

    # Signed distance to lower bound is positive when `response` is below the lower
    # bound and negative when it is above the lower bound. When the space lower bound
    # is equal to the window lower bound, then we disregard the distance entirely
    # and substitute a value equal to the signed distance to the upper bound when the
    # response value is equal to the lower bound.
    elementwise_signed_distance_to_lower_bound = window_lower_bound - response
    elementwise_signed_distance_to_lower_bound = jnp.where(
        window_lower_bound > space_lower_bound,
        elementwise_signed_distance_to_lower_bound,
        -window_dim,
    )

    # Signed distance to upper bound defined similarly to the lower bound.
    elementwise_signed_distance_to_upper_bound = response - window_upper_bound
    elementwise_signed_distance_to_upper_bound = jnp.where(
        window_upper_bound < space_upper_bound,
        elementwise_signed_distance_to_upper_bound,
        -window_dim,
    )

    # The signed distance to the window is positive
    elementwise_signed_distance_to_nearest_bound = jnp.maximum(
        elementwise_signed_distance_to_lower_bound,
        elementwise_signed_distance_to_upper_bound,
    )

    transformed_elementwise_signed_distance = jax.nn.softplus(
        elementwise_signed_distance_to_nearest_bound / jnp.amin(window_dim)
    )

    return jnp.linalg.norm(transformed_elementwise_signed_distance) ** exponent
