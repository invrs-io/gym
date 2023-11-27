"""Defines various transformation functions.

Copyright (c) 2023 The INVRS-IO authors.
"""

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
