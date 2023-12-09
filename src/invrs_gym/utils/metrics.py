"""Defines metrics used across challenges.

Copyright (c) 2023 The INVRS-IO authors.
"""

from typing import Any, Optional

import jax.numpy as jnp
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
        violation_lo = jnp.abs(array)
        violation_hi = jnp.abs(1 - array)
        violation = jnp.minimum(violation_lo, violation_hi)
        return 1 - 2 * jnp.amax(violation)

    degrees = [_compute_degree(d) for d in densities]
    return jnp.amax(jnp.asarray(degrees))
