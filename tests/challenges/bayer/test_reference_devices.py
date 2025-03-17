"""Simulates reference bayer sorter designs and compares to expected results.

Copyright (c) 2023 The INVRS-IO authors.
"""

import dataclasses
import unittest

import fmmax
import jax
import jax.numpy as jnp
import numpy as onp
import pytest
from totypes import types

from invrs_gym.challenges.bayer import component
from invrs_gym.utils import materials


class Li2022Test(unittest.TestCase):
    @pytest.mark.slow
    def test_transmission_matches_reference(self):
        # Simulates the structure from "Pixel-level Bayer-type colour router based on
        # metasurfaces", https://www.nature.com/articles/s41467-022-31019-7#Sec12

        # The design pattern is taken from supplementary figure 2.
        pattern = """0100100000000000
            0110001010000000
            0000100010000000
            0000000000000000
            1010000000000000
            1000100000000000
            0000011000000000
            1001000000000110
            0011110000000000
            1000000001000010
            0011000101000000
            0110100100100101
            0010010110000000
            0101110100010010
            0010100000000011
            0000001010110000"""

        design = [[float(i) for i in row.strip()] for row in pattern.split("\n")]
        design = onp.asarray(design)[::-1, :]  # Flip orientation compared to figure.
        design = onp.kron(design, onp.ones((10, 10)))

        # The design has Si3N4 pillars, unencapsulated on a SiO2 substrate.
        spec = component.BayerSpec(
            material_ambient=materials.VACUUM,
            material_metasurface_solid=materials.SI3N4,
            material_metasurface_void=materials.VACUUM,
            material_substrate=materials.SIO2,
            thickness_ambient=0.0,
            thickness_metasurface=types.BoundedArray(0.6, 0.4, 0.8),
            thickness_substrate=3.0,
            pixel_size=1.0,
            grid_spacing=0.01,
            offset_monitor_substrate=types.BoundedArray(2.4, 2.0, 2.5),
        )

        # Simulate for the three wavelengths depicted in supplementary figure 3.
        sim_params = component.BayerSimParams(
            wavelength=jnp.asarray([0.45, 0.54, 0.65]),
            polar_angle=jnp.zeros(()),
            azimuthal_angle=jnp.zeros(()),
            formulation=fmmax.Formulation.JONES_DIRECT_FOURIER,
            approximate_num_terms=600,
            truncation=fmmax.Truncation.CIRCULAR,
        )

        cc = component.BayerComponent(
            spec=spec,
            sim_params=sim_params,
            thickness_initializer=lambda _, t: t,
            density_initializer=lambda _, v: v,
        )

        params = cc.init(jax.random.PRNGKey(0))
        params["density_metasurface"] = dataclasses.replace(
            params["density_metasurface"], array=design
        )
        response, aux = cc.response(params, compute_fields=False)

        # Expected transmission and reflection, extracted from supplementary figure 7.
        # Use a per-value relative tolerance; these are generally around 25%, with only
        # one value being significantly larger at 44%. We verified that our calculation
        # is converged; increasing the number of terms does not significantly change
        # the result. This is likely due to some small differences in permittivities
        # we use versus those of the reference. Note also that the fields in the
        # reference have some asymmetry (despite a symmetric structure), and so it may
        # be that the reference is not converged.
        expected_transmission = onp.asarray(
            [
                # Blue, green, red pixels
                [0.436, 0.394, 0.133],  # 450 nm
                [0.160, 0.575, 0.237],  # 540 nm
                [0.117, 0.270, 0.589],  # 650 nm
            ]
        )
        transmission_rtol = onp.asarray(
            [
                # Blue, green, red pixels
                [0.27, 0.26, 0.27],  # 450 nm
                [0.25, 0.26, 0.44],  # 540 nm
                [0.11, 0.19, 0.19],  # 650 nm
            ]
        )
        expected_reflection = 1 - onp.sum(expected_transmission, axis=1)

        # Average the transmission over the two polarizations, and sum the values for
        # the two green pixels.
        transmission = onp.mean(response.transmission, axis=-2)
        transmission = onp.stack(
            [
                transmission[:, 3],  # Blue pixel
                transmission[:, 1] + transmission[:, 2],  # Green pixels
                transmission[:, 0],  # Red pixel
            ],
            axis=-1,
        )

        for t, et, rtol in zip(
            transmission.flatten(),
            expected_transmission.flatten(),
            transmission_rtol.flatten(),
        ):
            onp.testing.assert_allclose(t, et, rtol=rtol)

        # Average the reflection over the two polarizations.
        reflection = onp.mean(response.reflection, axis=1)
        onp.testing.assert_allclose(reflection, expected_reflection, atol=0.05)
