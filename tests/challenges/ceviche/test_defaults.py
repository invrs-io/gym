"""Tests for `ceviche.defaults`.

Copyright (c) 2023 The INVRS-IO authors.
"""

import unittest

import numpy as onp
from ceviche_challenges import beam_splitter, mode_converter, params
from ceviche_challenges import units as u
from ceviche_challenges import waveguide_bend, wdm
from parameterized import parameterized

from invrs_gym.challenges.ceviche import defaults


def beam_splitter_model(resolution):
    return beam_splitter.model.BeamSplitterModel(
        spec=defaults.BEAM_SPLITTER_SPEC,
        params=params.CevicheSimParams(
            resolution=resolution * u.nm,
            wavelengths=u.Array(defaults.WAVELENGTHS_NM, u.nm),
        ),
    )


def mode_converter_model(resolution):
    return mode_converter.model.ModeConverterModel(
        spec=defaults.MODE_CONVERTER_SPEC,
        params=params.CevicheSimParams(
            resolution=resolution * u.nm,
            wavelengths=u.Array(defaults.WAVELENGTHS_NM, u.nm),
        ),
    )


def waveguide_bend_model(resolution):
    return waveguide_bend.model.WaveguideBendModel(
        spec=defaults.WAVEGUIDE_BEND_SPEC,
        params=params.CevicheSimParams(
            resolution=resolution * u.nm,
            wavelengths=u.Array(defaults.WAVELENGTHS_NM, u.nm),
        ),
    )


def wdm_model(resolution):
    return wdm.model.WdmModel(
        spec=defaults.wdm_spec(
            design_extent_ij=u.Array([6400, 6400], u.nm),
            intended_sim_resolution=resolution * u.nm,
        ),
        params=params.CevicheSimParams(
            resolution=resolution * u.nm,
            wavelengths=u.Array(defaults.WAVELENGTHS_NM, u.nm),
        ),
    )


class CreateModelTest(unittest.TestCase):
    @parameterized.expand(
        [
            [beam_splitter_model, 10, (320, 200)],
            [beam_splitter_model, 20, (160, 100)],
            [beam_splitter_model, 40, (80, 50)],
            [mode_converter_model, 10, (160, 160)],
            [mode_converter_model, 20, (80, 80)],
            [mode_converter_model, 40, (40, 40)],
            [waveguide_bend_model, 10, (160, 160)],
            [waveguide_bend_model, 20, (80, 80)],
            [waveguide_bend_model, 40, (40, 40)],
            [wdm_model, 10, (640, 640)],
            [wdm_model, 20, (320, 320)],
            [wdm_model, 40, (160, 160)],
        ]
    )
    def test_beam_splitter_model(self, model_fn, resolution, expected_shape):
        model = model_fn(resolution)
        self.assertSequenceEqual(model.design_variable_shape, expected_shape)

    @parameterized.expand(
        [
            [beam_splitter_model, 10],
            [beam_splitter_model, 20],
            [beam_splitter_model, 40],
            [mode_converter_model, 10],
            [mode_converter_model, 20],
            [mode_converter_model, 40],
            [waveguide_bend_model, 10],
            [waveguide_bend_model, 20],
            [waveguide_bend_model, 40],
            [wdm_model, 10],
            [wdm_model, 20],
            [wdm_model, 40],
        ]
    )
    def test_no_structure_within_320nm_of_pml_region(self, model_fn, resolution):
        ceviche_model = model_fn(resolution)
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
