"""Tests for `utils.materials`.

Copyright (c) 2024 The INVRS-IO authors.
"""

import pathlib
import unittest

import jax
import jax.numpy as jnp
from parameterized import parameterized

from invrs_gym.utils import materials

MATERIALS = (
    materials.VACUUM,
    materials.SI,
    materials.AMORPHOUS_SI,
    materials.SIO2,
    materials.SI3N4,
    materials.TIO2,
)


class GetPermittivityTest(unittest.TestCase):
    @parameterized.expand(MATERIALS)
    def test_get_permittivity(self, material):
        wavelength_um = jnp.linspace(0.4, 0.7)
        materials.permittivity(material=material, wavelength_um=wavelength_um)

    @parameterized.expand(MATERIALS)
    def test_get_permittivity_with_jit(self, material):
        @jax.jit
        def jit_fn(wavelength_um):
            return materials.permittivity(
                material=material, wavelength_um=wavelength_um
            )

        wavelength_um = jnp.linspace(0.4, 0.7)
        jit_fn(wavelength_um)

    def test_get_permittivity_from_file(self):
        path = pathlib.Path(__file__).resolve().parent / "data/nk_dummy.csv"
        materials.register_material(name="dummy_material", path=path)

        wavelength_um = jnp.linspace(0.4, 0.7)
        with self.subTest("without jit"):
            materials.permittivity(
                material="dummy_material", wavelength_um=wavelength_um
            )

        @jax.jit
        def jit_fn(wavelength_um):
            return materials.permittivity(
                material="dummy_material", wavelength_um=wavelength_um
            )

        with self.subTest("with jit"):
            jit_fn(wavelength_um)
