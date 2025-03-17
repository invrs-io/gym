"""Tests for the meta-atom library challenge.

Copyright (c) 2024 The INVRS-IO authors.
"""

import unittest

import fmmax
import jax.numpy as jnp
import numpy as onp
from totypes import types

from invrs_gym.challenges.library import component
from invrs_gym.utils import materials

LIBRARY_SPEC = component.LibrarySpec(
    material_ambient=materials.VACUUM,
    material_metasurface_solid=component.TIO2_CHEN,
    material_metasurface_void=materials.VACUUM,
    material_substrate=materials.SIO2,
    background_extinction_coeff=0.0001,
    thickness_ambient=1.2,
    thickness_metasurface=0.6,
    thickness_substrate=0.2,
    pitch=0.4,
    frame_width=0.03,
    grid_spacing=0.005,
)

LIBRARY_SIM_PARAMS = component.LibrarySimParams(
    wavelength=jnp.asarray([0.45, 0.55, 0.65]),
    approximate_num_terms=100,
    formulation=fmmax.Formulation.JONES_DIRECT_FOURIER,
    truncation=fmmax.Truncation.CIRCULAR,
)


class ComponentTest(unittest.TestCase):
    def test_transmission_with_no_metagrating(self):
        zeros_density = types.Density2DArray(
            array=jnp.zeros((8, 80, 80)),
            lower_bound=0.0,
            upper_bound=1.0,
        )
        expansion = fmmax.generate_expansion(
            primitive_lattice_vectors=fmmax.LatticeVectors(fmmax.X, fmmax.Y),
            approximate_num_terms=LIBRARY_SIM_PARAMS.approximate_num_terms,
            truncation=LIBRARY_SIM_PARAMS.truncation,
        )
        response, _ = component.simulate_library(
            density=zeros_density,
            spec=LIBRARY_SPEC,
            wavelength=jnp.asarray([0.45, 0.55, 0.65]),
            expansion=expansion,
            formulation=LIBRARY_SIM_PARAMS.formulation,
            compute_fields=False,
        )

        permittivity = materials.permittivity(
            materials.SIO2, wavelength_um=jnp.asarray([0.45, 0.55, 0.65])
        )
        refractive_index = jnp.sqrt(permittivity)

        power_rhcp = (
            jnp.abs(response.transmission_rhcp[0, :, 0]) ** 2 * refractive_index
            + jnp.abs(response.reflection_rhcp[0, :, 0]) ** 2
        )
        power_lhcp = (
            jnp.abs(response.transmission_lhcp[0, :, 1]) ** 2 * refractive_index
            + jnp.abs(response.reflection_lhcp[0, :, 1]) ** 2
        )
        onp.testing.assert_allclose(power_rhcp, 1.0, atol=0.002)
        onp.testing.assert_allclose(power_lhcp, 1.0, atol=0.002)
