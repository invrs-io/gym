"""Tests for `sorter.common`.

Copyright (c) 2023 The INVRS-IO authors.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as onp
from fmmax import basis, fmm
from jax import tree_util
from parameterized import parameterized
from totypes import types

from invrs_gym.challenges.sorter import common

EXAMPLE_SPEC = common.SorterSpec(
    permittivity_ambient=(1.0 + 0.0j) ** 2,
    permittivity_cap=(1.5 + 0.0j) ** 2,
    permittivity_metasurface_solid=(4.0 + 0.0j) ** 2,
    permittivity_metasurface_void=(1.5 + 0.0j) ** 2,
    permittivity_spacer=(1.5 + 0.0j) ** 2,
    permittivity_substrate=(4.0730 + 0.028038j) ** 2,
    thickness_cap=types.BoundedArray(0.05, lower_bound=0.0, upper_bound=0.3),
    thickness_metasurface=(
        types.BoundedArray(0.15, lower_bound=0.05, upper_bound=0.3),
    ),
    thickness_spacer=(types.BoundedArray(1.0, lower_bound=0.5, upper_bound=1.5),),
    pitch=2.0,
    offset_monitor_substrate=0.05,
)

EXAMPLE_SIM_PARAMS = common.SorterSimParams(
    grid_spacing=0.01,
    wavelength=0.55,
    polar_angle=0.0,
    azimuthal_angle=0.0,
    formulation=fmm.Formulation.FFT,
    approximate_num_terms=100,
    truncation=basis.Truncation.CIRCULAR,
)


class SorterResponseTest(unittest.TestCase):
    def test_flatten_unflatten(self):
        original = common.SorterResponse(
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
        sc = common.SorterComponent(
            spec=EXAMPLE_SPEC,
            sim_params=EXAMPLE_SIM_PARAMS,
            thickness_initializer=lambda _, thickness: thickness,
            density_initializer=lambda _, seed_density: seed_density,
            minimum_width=minimum_width,
            minimum_spacing=minimum_spacing,
        )
        params = sc.init(jax.random.PRNGKey(0))
        self.assertEqual(
            set(params.keys()),
            {
                "density_metasurface",
                "thickness_metasurface",
                "thickness_cap",
                "thickness_spacer",
            },
        )

        self.assertEqual(params["density_metasurface"][0].lower_bound, 0.0)
        self.assertEqual(params["density_metasurface"][0].upper_bound, 1.0)
        self.assertEqual(params["density_metasurface"][0].minimum_width, minimum_width)
        self.assertEqual(
            params["density_metasurface"][0].minimum_spacing, minimum_spacing
        )
        self.assertSequenceEqual(
            params["density_metasurface"][0].periodic, (True, True)
        )

    def test_can_jit_response(self):
        sc = common.SorterComponent(
            spec=EXAMPLE_SPEC,
            sim_params=EXAMPLE_SIM_PARAMS,
            thickness_initializer=lambda _, thickness: thickness,
            density_initializer=lambda _, seed_density: seed_density,
        )
        params = sc.init(jax.random.PRNGKey(0))

        @jax.jit
        def jit_response_fn(params):
            return sc.response(params)

        jit_response_fn(params)

    def test_multiple_wavelengths(self):
        mc = common.SorterComponent(
            spec=EXAMPLE_SPEC,
            sim_params=EXAMPLE_SIM_PARAMS,
            thickness_initializer=lambda _, thickness: thickness,
            density_initializer=lambda _, seed_density: seed_density,
        )
        params = mc.init(jax.random.PRNGKey(0))
        response, aux = mc.response(params, wavelength=jnp.asarray([1.045, 1.055]))
        self.assertSequenceEqual(response.transmission.shape, (2, 4, 4))
        self.assertSequenceEqual(response.reflection.shape, (2, 4))
