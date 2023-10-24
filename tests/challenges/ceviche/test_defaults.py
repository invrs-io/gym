"""Tests for `ceviche.defaults`.

Copyright (c) 2023 Martin F. Schubert
"""

import unittest

import numpy as onp
from ceviche_challenges import units as u
from parameterized import parameterized

from invrs_gym.challenges.ceviche import defaults


class CreateModelTest(unittest.TestCase):
    @parameterized.expand(
        [
            [defaults.BEAM_SPLITTER_MODEL, (320, 200)],
            [defaults.LIGHTWEIGHT_BEAM_SPLITTER_MODEL, (80, 50)],
            [defaults.MODE_CONVERTER_MODEL, (160, 160)],
            [defaults.LIGHTWEIGHT_MODE_CONVERTER_MODEL, (40, 40)],
            [defaults.WAVEGUIDE_BEND_MODEL, (160, 160)],
            [defaults.LIGHTWEIGHT_WAVEGUIDE_BEND_MODEL, (40, 40)],
            [defaults.WDM_MODEL, (640, 640)],
            [defaults.LIGHTWEIGHT_WDM_MODEL, (80, 80)],
        ]
    )
    def test_design_variable_shapes(
        self, ceviche_model, expected_design_variable_shape
    ):
        self.assertSequenceEqual(
            ceviche_model.design_variable_shape, expected_design_variable_shape
        )

    @parameterized.expand(
        [
            [defaults.BEAM_SPLITTER_MODEL],
            [defaults.LIGHTWEIGHT_BEAM_SPLITTER_MODEL],
            [defaults.MODE_CONVERTER_MODEL],
            [defaults.LIGHTWEIGHT_MODE_CONVERTER_MODEL],
            [defaults.WAVEGUIDE_BEND_MODEL],
            [defaults.LIGHTWEIGHT_WAVEGUIDE_BEND_MODEL],
            [defaults.WDM_MODEL],
            [defaults.LIGHTWEIGHT_WDM_MODEL],
        ]
    )
    def test_no_structure_within_320nm_of_pml_region(self, ceviche_model):
        density = ceviche_model.density(
            onp.full(ceviche_model.design_variable_shape, 1.0)
        )

        # Trim the PML region plus 320 nm from the density.
        margin = defaults.PML_WIDTH_GRIDPOINTS + u.resolve(
            320 * u.nm, ceviche_model.params.resolution
        )
        trimmed = density[margin:-margin, margin:-margin]

        # Pad with edge values. The resulting array should be identical to the
        # original density.
        padded = onp.pad(trimmed, ((margin, margin), (margin, margin)), mode="edge")
        onp.testing.assert_array_equal(padded, density)
