"""Defines functions that compute material properties.

Copyright (c) 2024 The INVRS-IO authors.
"""

import functools
import pathlib
import warnings
from packaging import version
from typing import Dict, Protocol, Union

import jax
import numpy as onp
import refractiveindex2 as ri
from jax import numpy as jnp

Material = str | ri.RefractiveIndexMaterial

# Define some common materials.
SI = ri.RefractiveIndexMaterial(shelf="main", book="Si", page="Green-1995")
AMORPHOUS_SI = ri.RefractiveIndexMaterial(shelf="main", book="Si", page="Pierce")
SIO2 = ri.RefractiveIndexMaterial(shelf="main", book="SiO2", page="Malitson")
SI3N4 = ri.RefractiveIndexMaterial(shelf="main", book="Si3N4", page="Luke")
TIO2 = ri.RefractiveIndexMaterial(shelf="main", book="TiO2", page="Jolivet-amorphous")
VACUUM = "vacuum"  # Permittivity is computed via dedicated function.


if version.Version(jax.__version__) > version.Version("0.4.31"):
    callback = functools.partial(jax.pure_callback, vmap_method="broadcast_all")
else:
    callback = functools.partial(jax.pure_callback, vectorized=True)


def permittivity(
    material: str | ri.RefractiveIndexMaterial,
    wavelength_um: float | jnp.ndarray,
    background_extinction_coeff: float = 0.0,
) -> jnp.ndarray:
    """Return the permittivity for the specified material.

    Args:
        material: The name of a registered material, or a `ri.RefractiveIndexMaterial`
            for which the permittivity is sought.
        wavelength_um: The wavelength for the permittivity, in units of microns.
        background_extinction_coeff: Additional extinction coefficient to be included
            in the permittivity. Positive values correspond to optical loss, and the
            default value is 0.

    Returns:
        The permittivity of the material at the specified wavelengths.
    """
    wavelength_um = jnp.asarray(wavelength_um)
    if isinstance(material, ri.RefractiveIndexMaterial):
        permittivity_fn: PermittivityFn = functools.partial(
            permittivity_from_database,
            material=material,
        )
    else:
        permittivity_fn = PERMITTIVITY_FNS[material]
    return permittivity_fn(
        wavelength_um=wavelength_um,
        background_extinction_coeff=background_extinction_coeff,
    )


def permittivity_from_database(
    material: ri.RefractiveIndexMaterial,
    wavelength_um: float | jnp.ndarray,
    background_extinction_coeff: float,
) -> jnp.ndarray:
    """Return the permittivity for the specified material from the database."""
    wavelength_um = jnp.asarray(wavelength_um)

    def _refractive_index_fn(wavelength_um: jnp.ndarray) -> onp.ndarray:
        numpy_wavelength_um = onp.asarray(wavelength_um)
        dtype = onp.promote_types(wavelength_um.dtype, onp.complex64)
        try:
            epsilon = material.get_epsilon(numpy_wavelength_um)
            refractive_index = onp.sqrt(epsilon)
        except ri.refractiveindex.NoExtinctionCoefficient:
            refractive_index = material.get_refractive_index(numpy_wavelength_um)
        return onp.asarray(refractive_index, dtype=dtype)

    dtype = jnp.promote_types(wavelength_um.dtype, jnp.complex64)
    result_shape_dtypes = jnp.zeros_like(wavelength_um, dtype=dtype)
    refractive_index = callback(
        _refractive_index_fn, result_shape_dtypes, wavelength_um
    )
    return (refractive_index + 1j * background_extinction_coeff) ** 2


def permittivity_vacuum(
    wavelength_um: float | jnp.ndarray,
    background_extinction_coeff: float = 0.0,
) -> jnp.ndarray:
    """Return the permittivity of vacuum, with optional background extinction coeff."""
    wavelength_um = jnp.asarray(wavelength_um)
    dtype = jnp.promote_types(wavelength_um.dtype, jnp.complex64)
    return jnp.full(
        wavelength_um.shape, 1.0 + 1j * background_extinction_coeff, dtype=dtype
    )


class PermittivityFn(Protocol):
    def __call__(
        self, wavelength_um: jnp.ndarray, background_extinction_coeff: float
    ) -> jnp.ndarray:
        ...


PERMITTIVITY_FNS: Dict[str, PermittivityFn] = {
    VACUUM: permittivity_vacuum,
}


def register_material(name: str, path: Union[str, pathlib.Path]) -> None:
    """Registers a material so it is accessible by `get_permittivity` function."""
    with jax.ensure_compile_time_eval():
        data_wvl_n_k = onp.genfromtxt(path, delimiter=",", comments="#")
        data_wavelength_um = jnp.asarray(data_wvl_n_k[:, 0])
        data_permittivity = (data_wvl_n_k[:, 1] + 1j * data_wvl_n_k[:, 2]) ** 2

        if name in PERMITTIVITY_FNS:
            warnings.warn(f"Material {name} already registered.")

        def _permittivity_fn(
            wavelength_um: float | jnp.ndarray,
            background_extinction_coeff: float,
        ) -> jnp.ndarray:
            wavelength_um = jnp.asarray(wavelength_um)
            dtype = jnp.promote_types(wavelength_um.dtype, jnp.complex64)
            refractive_index = jnp.sqrt(
                jnp.interp(wavelength_um, data_wavelength_um, data_permittivity)
            )
            return jnp.asarray(
                (refractive_index + 1j * background_extinction_coeff) ** 2, dtype=dtype
            )

        PERMITTIVITY_FNS[name] = _permittivity_fn
