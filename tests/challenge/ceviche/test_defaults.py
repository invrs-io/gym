"""Tests for `challenge.ceviche.defaults`."""

import unittest

from ceviche_challenges import (
    beam_splitter,
    mode_converter,
    waveguide_bend,
    wdm,
)
from invrs_gym.challenge.ceviche import defaults


class CreateModelTest(unittest.TestCase):
    def test_create_beam_splitter(self):
        model = beam_splitter.model.BeamSplitterModel(
            defaults.SIM_PARAMS, defaults.BEAM_SPLITTER_SPEC
        )
        self.assertSequenceEqual(model.design_variable_shape, (320, 200))

    def test_create_lightweight_beam_splitter(self):
        model = beam_splitter.model.BeamSplitterModel(
            defaults.LIGHTWEIGHT_SIM_PARAMS, defaults.BEAM_SPLITTER_SPEC
        )
        self.assertSequenceEqual(model.design_variable_shape, (80, 50))

    def test_create_mode_converter(self):
        model = mode_converter.model.ModeConverterModel(
            defaults.SIM_PARAMS, defaults.MODE_CONVERTER_SPEC
        )
        self.assertSequenceEqual(model.design_variable_shape, (160, 160))

    def test_create_lightweight_mode_converter(self):
        model = mode_converter.model.ModeConverterModel(
            defaults.LIGHTWEIGHT_SIM_PARAMS, defaults.MODE_CONVERTER_SPEC
        )
        self.assertSequenceEqual(model.design_variable_shape, (40, 40))

    def test_create_waveguide_bend(self):
        model = waveguide_bend.model.WaveguideBendModel(
            defaults.SIM_PARAMS, defaults.WAVEGUIDE_BEND_SPEC
        )
        self.assertSequenceEqual(model.design_variable_shape, (160, 160))

    def test_create_lightweight_waveguide_bend(self):
        model = waveguide_bend.model.WaveguideBendModel(
            defaults.LIGHTWEIGHT_SIM_PARAMS, defaults.WAVEGUIDE_BEND_SPEC
        )
        self.assertSequenceEqual(model.design_variable_shape, (40, 40))

    def test_create_wdm(self):
        model = wdm.model.WdmModel(defaults.SIM_PARAMS, defaults.WDM_SPEC)
        self.assertSequenceEqual(model.design_variable_shape, (640, 640))

    def test_create_lightweight_wdm(self):
        model = wdm.model.WdmModel(
            defaults.LIGHTWEIGHT_SIM_PARAMS, defaults.LIGHTWEIGHT_WDM_SPEC
        )
        self.assertSequenceEqual(model.design_variable_shape, (80, 80))
