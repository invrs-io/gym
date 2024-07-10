"""Tests for `bayer.component`.

Copyright (c) 2023 The INVRS-IO authors.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as onp
from jax import tree_util
from fmmax import basis, fmm
from parameterized import parameterized
from totypes import types

from invrs_gym.challenges.bayer import component
from invrs_gym.utils import materials

EXAMPLE_SPEC = component.BayerSpec(
    material_ambient=materials.VACUUM,
    material_metasurface_solid=materials.SI3N4,
    material_metasurface_void=materials.VACUUM,
    material_substrate=materials.SIO2,
    thickness_ambient=1.0,
    thickness_metasurface=types.BoundedArray(0.6, 0.4, 0.8),
    thickness_substrate=1.0,
    pixel_size=1.0,
    grid_spacing=0.01,
    offset_monitor_substrate=types.BoundedArray(2.4, 2.0, 2.8),
)

EXAMPLE_SIM_PARAMS = component.BayerSimParams(
    wavelength=jnp.asarray([0.55]),
    polar_angle=0.0,
    azimuthal_angle=0.0,
    formulation=fmm.Formulation.FFT,
    approximate_num_terms=100,
    truncation=basis.Truncation.CIRCULAR,
)


class SorterResponseTest(unittest.TestCase):
    def test_flatten_unflatten(self):
        original = component.BayerResponse(
            wavelength=jnp.arange(3),
            polar_angle=jnp.asarray(4),
            azimuthal_angle=jnp.asarray(5),
            transmission=jnp.arange(4),
            reflection=jnp.arange(5),
        )
        leaves, treedef = tree_util.tree_flatten(original)
        restored = tree_util.tree_unflatten(treedef, leaves)
        onp.testing.assert_array_equal(restored.wavelength, original.wavelength)
        onp.testing.assert_array_equal(restored.polar_angle, original.polar_angle)
        onp.testing.assert_array_equal(
            restored.azimuthal_angle, original.azimuthal_angle
        )
        onp.testing.assert_array_equal(restored.transmission, original.transmission)
        onp.testing.assert_array_equal(restored.reflection, original.reflection)


class SorterComponentTest(unittest.TestCase):
    @parameterized.expand([[1, 1], [2, 3]])
    def test_density_has_expected_properties(self, minimum_width, minimum_spacing):
        sc = component.BayerComponent(
            spec=EXAMPLE_SPEC,
            sim_params=EXAMPLE_SIM_PARAMS,
            thickness_initializer=lambda _, t: t,
            density_initializer=lambda _, seed_density: seed_density,
            minimum_width=minimum_width,
            minimum_spacing=minimum_spacing,
        )
        params = sc.init(jax.random.PRNGKey(0))
        density = params["density_metasurface"]

        self.assertEqual(density.lower_bound, 0.0)
        self.assertEqual(density.upper_bound, 1.0)
        self.assertEqual(density.minimum_width, minimum_width)
        self.assertEqual(density.minimum_spacing, minimum_spacing)
        self.assertSequenceEqual(density.periodic, (True, True))

    def test_can_jit_response(self):
        sc = component.BayerComponent(
            spec=EXAMPLE_SPEC,
            sim_params=EXAMPLE_SIM_PARAMS,
            thickness_initializer=lambda _, t: t,
            density_initializer=lambda _, seed_density: seed_density,
        )
        params = sc.init(jax.random.PRNGKey(0))

        @jax.jit
        def jit_response_fn(params):
            return sc.response(params)

        jit_response_fn(params)

    def test_multiple_wavelengths(self):
        mc = component.BayerComponent(
            spec=EXAMPLE_SPEC,
            sim_params=EXAMPLE_SIM_PARAMS,
            thickness_initializer=lambda _, t: t,
            density_initializer=lambda _, seed_density: seed_density,
        )
        params = mc.init(jax.random.PRNGKey(0))
        response, aux = mc.response(params, wavelength=jnp.asarray([0.4, 0.5]))
        self.assertSequenceEqual(response.transmission.shape, (2, 2, 4))
        self.assertSequenceEqual(response.reflection.shape, (2, 2))
        self.assertSequenceEqual(aux[component.TRANSMITTED_POWER].shape, (2, 2))

    def test_compute_fields(self):
        mc = component.BayerComponent(
            spec=EXAMPLE_SPEC,
            sim_params=EXAMPLE_SIM_PARAMS,
            thickness_initializer=lambda _, t: t,
            density_initializer=lambda _, seed_density: seed_density,
        )
        params = mc.init(jax.random.PRNGKey(0))
        _, aux = mc.response(params, compute_fields=True)
        self.assertTrue(component.EFIELD_XZ in aux)
        self.assertTrue(component.HFIELD_XZ in aux)
        self.assertTrue(component.COORDINATES_XZ in aux)
