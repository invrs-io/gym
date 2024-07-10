from invrs_gym.challenges.bayer.challenge import bayer_sorter as bayer_sorter
from invrs_gym.challenges.ceviche.challenge import (
    beam_splitter as ceviche_beam_splitter,
)
from invrs_gym.challenges.ceviche.challenge import (
    lightweight_beam_splitter as ceviche_lightweight_beam_splitter,
)
from invrs_gym.challenges.ceviche.challenge import (
    lightweight_mode_converter as ceviche_lightweight_mode_converter,
)
from invrs_gym.challenges.ceviche.challenge import (
    lightweight_waveguide_bend as ceviche_lightweight_waveguide_bend,
)
from invrs_gym.challenges.ceviche.challenge import (
    lightweight_wdm as ceviche_lightweight_wdm,
)
from invrs_gym.challenges.ceviche.challenge import (
    mode_converter as ceviche_mode_converter,
)
from invrs_gym.challenges.ceviche.challenge import (
    waveguide_bend as ceviche_waveguide_bend,
)
from invrs_gym.challenges.ceviche.challenge import wdm as ceviche_wdm
from invrs_gym.challenges.diffract.metagrating_challenge import metagrating
from invrs_gym.challenges.diffract.splitter_challenge import diffractive_splitter
from invrs_gym.challenges.extractor.challenge import photon_extractor
from invrs_gym.challenges.library.challenge import meta_atom_library
from invrs_gym.challenges.metalens.challenge import metalens

BY_NAME = {
    "bayer_sorter": bayer_sorter,
    "ceviche_beam_splitter": ceviche_beam_splitter,
    "ceviche_mode_converter": ceviche_mode_converter,
    "ceviche_waveguide_bend": ceviche_waveguide_bend,
    "ceviche_wdm": ceviche_wdm,
    "ceviche_lightweight_beam_splitter": ceviche_lightweight_beam_splitter,
    "ceviche_lightweight_mode_converter": ceviche_lightweight_mode_converter,
    "ceviche_lightweight_waveguide_bend": ceviche_lightweight_waveguide_bend,
    "ceviche_lightweight_wdm": ceviche_lightweight_wdm,
    "metagrating": metagrating,
    "metalens": metalens,
    "meta_atom_library": meta_atom_library,
    "diffractive_splitter": diffractive_splitter,
    "photon_extractor": photon_extractor,
}
