"""Defines initializers for densities and other objects.

Copyright (c) 2023 The INVRS-IO authors.
"""

from typing import Any

import jax
import jax.numpy as jnp
from totypes import types


def identity_initializer(key: jax.Array, seed_obj: Any) -> Any:
    """Returns `seed_obj`."""
    del key
    return seed_obj


def noisy_density_initializer(
    key: jax.Array,
    seed_density: types.Density2DArray,
    relative_stddev: float,
) -> types.Density2DArray:
    """Return a noisy density."""
    density = add_noise(key, density=seed_density, relative_stddev=relative_stddev)
    density = types.symmetrize_density(density)
    return apply_fixed_pixels(density)


# -----------------------------------------------------------------------------
# Functions used in density initializers.
# -----------------------------------------------------------------------------


def add_noise(
    key: jax.Array,
    density: types.Density2DArray,
    relative_stddev: float,
    resize_method: jax.image.ResizeMethod = jax.image.ResizeMethod.CUBIC,
) -> types.Density2DArray:
    """Return a density with added random noise.

    The noise has a length scale determined by the minimum width and spacing of the
    `density`, and an amplitude determined by its lower and upper bounds.

    Args:
        key: Key used in the generation of random noise.
        density: The density to which noise is added.
        relative_stddev: The relative standard deviation of added noise.
        resize_method: The method used to resize a low-resolution array to the final
            array, ensuring the length scale of added noise.

    Returns:
        The `Density2DArray` with added noise.
    """
    length_scale = (density.minimum_spacing + density.minimum_width) / 2
    low_res_shape = density.shape[:-2] + tuple(
        int(jnp.ceil(s / length_scale)) for s in density.shape[-2:]
    )
    low_res_noise = jax.random.normal(key, low_res_shape)
    noise = jax.image.resize(low_res_noise, density.shape, method=resize_method)
    noise *= relative_stddev * (density.upper_bound - density.lower_bound)
    arr = density.array + noise
    arr = jnp.clip(arr, density.lower_bound, density.upper_bound)
    return _substitute_array(arr, density)


def apply_fixed_pixels(density: types.Density2DArray) -> types.Density2DArray:
    """Return a density with fixed void and solid pixels set to respective bounds."""
    arr = density.array
    if density.fixed_solid is not None:
        arr = jnp.where(density.fixed_solid, density.upper_bound, arr)
    if density.fixed_void is not None:
        arr = jnp.where(density.fixed_void, density.lower_bound, arr)
    return _substitute_array(jnp.asarray(arr), density)


def _substitute_array(
    arr: jnp.ndarray, density: types.Density2DArray
) -> types.Density2DArray:
    """Return a new density identical to `density` but with `arr` as the `array`."""
    assert arr.shape == density.shape
    return jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(density), [arr])
