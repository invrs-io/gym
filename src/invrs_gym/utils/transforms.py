"""Defines various transformation functions.

Copyright (c) 2023 The INVRS-IO authors.
"""

from typing import Tuple

import jax
from jax import numpy as jnp
from totypes import types


def rescaled_density_array(
    density: types.Density2DArray,
    lower_bound: float,
    upper_bound: float,
) -> jnp.ndarray:
    """Return a density array for specified lower and upper bounds."""
    array = density.array - density.lower_bound
    array /= density.upper_bound - density.lower_bound
    array *= upper_bound - lower_bound
    return jnp.asarray(array + lower_bound)


def interpolate_permittivity(
    permittivity_solid: jnp.ndarray,
    permittivity_void: jnp.ndarray,
    density: jnp.ndarray,
) -> jnp.ndarray:
    """Interpolates the permittivity with a scheme that avoids zero crossings.

    The interpolation uses the scheme introduced in [2019 Christiansen], which avoids
    zero crossings that can occur with metals or lossy materials having a negative
    real component of the permittivity. https://doi.org/10.1016/j.cma.2018.08.034

    Args:
        permittivity_solid: The permittivity of solid regions.
        permittivity_void: The permittivity of void regions.
        density: The density, specifying which locations are solid and which are void.

    Returns:
        The interpolated permittivity.
    """
    n_solid = jnp.real(jnp.sqrt(permittivity_solid))
    k_solid = jnp.imag(jnp.sqrt(permittivity_solid))
    n_void = jnp.real(jnp.sqrt(permittivity_void))
    k_void = jnp.imag(jnp.sqrt(permittivity_void))
    n = density * n_solid + (1 - density) * n_void
    k = density * k_solid + (1 - density) * k_void
    return (n + 1j * k) ** 2


def resample(
    x: jnp.ndarray,
    shape: Tuple[int, ...],
    method: jax.image.ResizeMethod = jax.image.ResizeMethod.LINEAR,
) -> jnp.ndarray:
    """Resamples `x` to have the specified `shape`.

    The algorithm first upsamples `x` so that the pixels in the output image are
    comprised of an integer number of pixels in the upsampled `x`, and then
    performs box downsampling.

    Args:
        x: The array to be resampled.
        shape: The shape of the output array.
        method: The method used to resize `x` prior to box downsampling.

    Returns:
        The resampled array.
    """
    if x.ndim != len(shape):
        raise ValueError(
            f"`shape` must have length matching number of dimensions in `x`, "
            f"but got {shape} when `x` had shape {x.shape}."
        )

    with jax.ensure_compile_time_eval():
        factor = [int(jnp.ceil(dx / d)) for dx, d in zip(x.shape, shape)]
        upsampled_shape = tuple([d * f for d, f in zip(shape, factor)])

    x_upsampled = jax.image.resize(
        image=x,
        shape=upsampled_shape,
        method=method,
    )

    return box_downsample(x_upsampled, shape)


def box_downsample(x: jnp.ndarray, shape: Tuple[int, ...]) -> jnp.ndarray:
    """Downsamples `x` to a coarser resolution array using box downsampling.

    Box downsampling forms nonoverlapping windows and simply averages the
    pixels within each window. For example, downsampling `(0, 1, 2, 3, 4, 5)`
    with a factor of `2` yields `(0.5, 2.5, 4.5)`.

    Args:
        x: The array to be downsampled.
        shape: The shape of the output array; each axis dimension must evenly
            divide the corresponding axis dimension in `x`.

    Returns:
        The output array with shape `shape`.
    """
    if x.ndim != len(shape) or any([(d % s) != 0 for d, s in zip(x.shape, shape)]):
        raise ValueError(
            f"Each axis of `shape` must evenly divide the corresponding axis "
            f"dimension in `x`, but got shape {shape} when `x` has shape "
            f"{x.shape}."
        )
    shape = sum([(s, d // s) for d, s in zip(x.shape, shape)], ())
    axes = list(range(1, 2 * x.ndim, 2))
    x = x.reshape(shape)
    return jnp.mean(x, axis=axes)
