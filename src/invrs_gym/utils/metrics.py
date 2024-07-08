"""Defines metrics used across challenges.

Copyright (c) 2023 The INVRS-IO authors.
"""

from typing import Any, Optional

import jax.numpy as jnp
from jax import lax
from jax import tree_util
from totypes import types

PyTree = Any


def binarization_degree(params: PyTree) -> Optional[jnp.ndarray]:
    """Compute binarization degree, the most sigificant deviation from binary.

    The binarization degree for each density in `params` is computed by doubling
    the most significant deviation from binary of any pixel, and then subtracting
    form one. Thus, for arrays having values within the declared density bounds,
    the minimum possible binarization degree is 0.0, and the maximum is 1.0
    (corresponding to a fully binary design).

    Note that intermediate-valued pixels at the interface between solid and void
    regions are ignored.

    Args:
        params: The params for which the binarization degree is sought.

    Returns:
        The binarization degree, or `None` if `params` contains no densities.
    """
    leaves = tree_util.tree_leaves(
        params, is_leaf=lambda x: isinstance(x, types.Density2DArray)
    )
    densities = [leaf for leaf in leaves if isinstance(leaf, types.Density2DArray)]

    if not densities:
        return None

    def _compute_degree(d: types.Density2DArray) -> jnp.ndarray:
        array = (d.array - d.lower_bound) / (d.upper_bound - d.lower_bound)
        return _array_binarization_degree(jnp.asarray(array))

    degrees = [_compute_degree(d) for d in densities]
    return jnp.amax(jnp.asarray(degrees))


def _array_binarization_degree(x: jnp.ndarray) -> jnp.ndarray:
    """Compute the binarization degree for an array.

    Args:
        x: The array for which the binarization degree is sought.

    Returns:
        The scalar binarization degree.
    """
    x = x.reshape((-1,) + x.shape[-2:])

    violation_lo = jnp.abs(x)
    violation_hi = jnp.abs(1 - x)
    violation = jnp.minimum(violation_lo, violation_hi)

    # Interface pixels must be surrounded by at least four solid or void pixels.
    solid_or_void_pixels = (violation == 0).astype(float)
    solid_or_void_pixels = jnp.pad(
        solid_or_void_pixels,
        ((0, 0), (1, 1), (1, 1)),
        constant_values=1,
    )
    neighbors = lax.conv_general_dilated(
        lhs=solid_or_void_pixels[:, jnp.newaxis, :, :],
        rhs=jnp.ones((1, 1, 3, 3)),
        padding="VALID",
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
        window_strides=(1, 1),
    )
    neighbors = jnp.squeeze(neighbors, axis=1)
    interface_pixels = neighbors >= 4

    violation = jnp.where(interface_pixels, 0.0, violation)
    degree = 1 - 2 * jnp.amax(violation)
    return degree
