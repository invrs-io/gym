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
    relative_mean: jnp.ndarray | float,
    relative_noise_amplitude: jnp.ndarray | float,
    resize_method: jax.image.ResizeMethod = jax.image.ResizeMethod.CUBIC,
) -> types.Density2DArray:
    """Return a density with specified mean and added random noise.

    Only metadata from the seed density (e.g. feature sizes) are retained; the array
    for the output density is randomly generated. It has the specified mean, and added
    uniform random noise with the specified amplitude and a length scale determined
    by the minimum width and spacing of the `seed_density`.

    Args:
        key: Key used in the generation of random noise.
        seed_density: The density used to provide metadata.
        relative_mean: The relative mean value of the output density. For a value of
            `0.5`, the mean is between the density upper and lower bounds.
        relative_noise_amplitude: The relative amplitude of noise added to the mean.
        resize_method: The method used to resize a low-resolution array to the final
            array, ensuring the length scale of added noise.

    Returns:
        The `Density2DArray` with added noise.
    """
    mean = (
        seed_density.lower_bound
        + (seed_density.upper_bound - seed_density.lower_bound) * relative_mean
    )
    array = jnp.full(seed_density.shape, mean)

    length_scale = (seed_density.minimum_spacing + seed_density.minimum_width) / 2
    low_res_shape = seed_density.shape[:-2] + tuple(
        int(jnp.ceil(s / length_scale)) for s in seed_density.shape[-2:]
    )
    low_res_noise = jax.random.uniform(key, low_res_shape) - 0.5
    noise = jax.image.resize(low_res_noise, seed_density.shape, method=resize_method)
    noise *= relative_noise_amplitude * (
        seed_density.upper_bound - seed_density.lower_bound
    )

    array += noise
    array = jnp.clip(array, seed_density.lower_bound, seed_density.upper_bound)

    density = substitute_array(array, seed_density)
    return types.symmetrize_density(density)


# -----------------------------------------------------------------------------
# Functions used in density initializers.
# -----------------------------------------------------------------------------


def apply_fixed_pixels(density: types.Density2DArray) -> types.Density2DArray:
    """Return a density with fixed void and solid pixels set to respective bounds."""
    arr = density.array
    if density.fixed_solid is not None:
        arr = jnp.where(density.fixed_solid, density.upper_bound, arr)
    if density.fixed_void is not None:
        arr = jnp.where(density.fixed_void, density.lower_bound, arr)
    return substitute_array(jnp.asarray(arr), density)


def substitute_array(
    arr: jnp.ndarray, density: types.Density2DArray
) -> types.Density2DArray:
    """Return a new density identical to `density` but with `arr` as the `array`."""
    assert arr.shape == density.shape
    return jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(density), [arr])
