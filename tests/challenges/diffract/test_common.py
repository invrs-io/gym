"""Tests for `diffract.common`.

Copyright (c) 2025 invrs.io LLC
"""

import unittest

import fmmax
import jax
import jax.numpy as jnp
import numpy as onp
from jax import tree_util
from totypes import types

from invrs_gym.challenges.diffract import common

SIMPLE_GRATING_SPEC = common.GratingSpec(
    permittivity_ambient=(1.0 + 0.0j) ** 2,
    permittivity_cap=(1.0 + 0.0j) ** 2,
    permittivity_grating=(3.45 + 0.00001j) ** 2,
    permittivity_encapsulation=(1.0 + 0.00001j) ** 2,
    permittivity_spacer=(1.45 + 0.0j) ** 2,
    permittivity_substrate=(1.45 + 0.0j) ** 2,
    thickness_ambient=0.0,
    thickness_cap=0.0,
    thickness_grating=0.325,
    thickness_spacer=0.0,
    thickness_substrate=0.0,
    period_x=float(1.050 / jnp.sin(jnp.deg2rad(50.0))),
    period_y=0.525,
    grid_spacing=0.0117,
)

GRATING_WITH_THICKNESS_SPEC = common.GratingSpec(
    permittivity_ambient=(1.0 + 0.0j) ** 2,
    permittivity_cap=(1.0 + 0.0j) ** 2,
    permittivity_grating=(3.45 + 0.00001j) ** 2,
    permittivity_encapsulation=(1.0 + 0.00001j) ** 2,
    permittivity_spacer=(1.45 + 0.0j) ** 2,
    permittivity_substrate=(1.45 + 0.0j) ** 2,
    thickness_ambient=0.0,
    thickness_cap=types.BoundedArray(array=0.0, lower_bound=0.0, upper_bound=0.1),
    thickness_grating=types.BoundedArray(array=0.6, lower_bound=0.5, upper_bound=1.5),
    thickness_spacer=types.BoundedArray(array=0.0, lower_bound=0.0, upper_bound=0.1),
    thickness_substrate=0.0,
    period_x=float(1.050 / jnp.sin(jnp.deg2rad(50.0))),
    period_y=0.525,
    grid_spacing=0.0117,
)


LIGHTWEIGHT_SIM_PARAMS = common.GratingSimParams(
    wavelength=1.050,
    polar_angle=0.0,
    azimuthal_angle=0.0,
    formulation=fmmax.Formulation.FFT,
    approximate_num_terms=100,
    truncation=fmmax.Truncation.CIRCULAR,
)


class GatingResponseTest(unittest.TestCase):
    def test_flatten_unflatten(self):
        original = common.GratingResponse(
            wavelength=jnp.arange(3),
            polar_angle=jnp.arange(4),
            azimuthal_angle=jnp.arange(5),
            transmission_efficiency=jnp.arange(4),
            reflection_efficiency=jnp.arange(5),
            expansion=fmmax.Expansion(
                basis_coefficients=onp.arange(10).reshape((5, 2))
            ),
        )
        leaves, treedef = tree_util.tree_flatten(original)
        restored = tree_util.tree_unflatten(treedef, leaves)
        onp.testing.assert_array_equal(restored.wavelength, original.wavelength)
        onp.testing.assert_array_equal(restored.polar_angle, original.polar_angle)
        onp.testing.assert_array_equal(
            restored.azimuthal_angle, original.azimuthal_angle
        )
        onp.testing.assert_array_equal(
            restored.transmission_efficiency, original.transmission_efficiency
        )
        onp.testing.assert_array_equal(
            restored.reflection_efficiency, original.reflection_efficiency
        )
        onp.testing.assert_array_equal(restored.expansion, original.expansion)


class SimpleGratingComponentTest(unittest.TestCase):
    def test_can_jit_response(self):
        mc = common.SimpleGratingComponent(
            spec=SIMPLE_GRATING_SPEC,
            sim_params=LIGHTWEIGHT_SIM_PARAMS,
            density_initializer=lambda _, seed_density: seed_density,
        )
        params = mc.init(jax.random.PRNGKey(0))

        @jax.jit
        def jit_response_fn(params):
            return mc.response(params)

        jit_response_fn(params)

    def test_multiple_wavelengths(self):
        mc = common.SimpleGratingComponent(
            spec=SIMPLE_GRATING_SPEC,
            sim_params=LIGHTWEIGHT_SIM_PARAMS,
            density_initializer=lambda _, seed_density: seed_density,
        )
        params = mc.init(jax.random.PRNGKey(0))
        response, aux = mc.response(params, wavelength=jnp.asarray([1.045, 1.055]))
        self.assertSequenceEqual(
            response.transmission_efficiency.shape,
            (2, mc.expansion.num_terms, 2),
        )

    def test_compute_fields(self):
        mc = common.SimpleGratingComponent(
            spec=SIMPLE_GRATING_SPEC,
            sim_params=LIGHTWEIGHT_SIM_PARAMS,
            density_initializer=lambda _, seed_density: seed_density,
        )
        params = mc.init(jax.random.PRNGKey(0))
        response, aux = mc.response(params, compute_fields=True)
        self.assertSequenceEqual(
            set(aux.keys()),
            {common.EFIELD, common.HFIELD, common.FIELD_COORDINATES},
        )


class GratingWithOptimizableThicknessComponentTest(unittest.TestCase):
    def test_can_jit_response(self):
        mc = common.GratingWithOptimizableThicknessComponent(
            spec=GRATING_WITH_THICKNESS_SPEC,
            sim_params=LIGHTWEIGHT_SIM_PARAMS,
            thickness_initializer=lambda _, thickness: thickness,
            density_initializer=lambda _, seed_density: seed_density,
        )
        params = mc.init(jax.random.PRNGKey(0))

        @jax.jit
        def jit_response_fn(params):
            return mc.response(params)

        jit_response_fn(params)

    def test_multiple_wavelengths(self):
        mc = common.GratingWithOptimizableThicknessComponent(
            spec=GRATING_WITH_THICKNESS_SPEC,
            sim_params=LIGHTWEIGHT_SIM_PARAMS,
            thickness_initializer=lambda _, thickness: thickness,
            density_initializer=lambda _, seed_density: seed_density,
        )
        params = mc.init(jax.random.PRNGKey(0))
        response, aux = mc.response(params, wavelength=jnp.asarray([1.045, 1.055]))
        self.assertSequenceEqual(
            response.transmission_efficiency.shape,
            (2, mc.expansion.num_terms, 2),
        )

    def test_compute_fields(self):
        mc = common.GratingWithOptimizableThicknessComponent(
            spec=GRATING_WITH_THICKNESS_SPEC,
            sim_params=LIGHTWEIGHT_SIM_PARAMS,
            thickness_initializer=lambda _, thickness: thickness,
            density_initializer=lambda _, seed_density: seed_density,
        )
        params = mc.init(jax.random.PRNGKey(0))
        response, aux = mc.response(params, compute_fields=True)
        self.assertSequenceEqual(
            set(aux.keys()),
            {common.EFIELD, common.HFIELD, common.FIELD_COORDINATES},
        )
