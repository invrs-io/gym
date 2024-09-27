"""Tests involving simulations of reference nanostructure.

The reference nanostructures are from "Dispersion-engineered metasurfaces reaching
broadband 90% relative diffraction efficiency" by Chen et al. (2023).
https://www.nature.com/articles/s41467-023-38185-2

Copyright (c) 2024 The INVRS-IO authors.
"""

import pathlib
import unittest

import jax.numpy as jnp
import numpy as onp
from fmmax import basis, fmm
from invrs_gym.utils import materials
from totypes import json_utils

from invrs_gym.challenges.library import component

REPO_PATH = pathlib.Path(__file__).resolve().parent.parent.parent.parent
DESIGNS_DIR = REPO_PATH / "reference_designs/meta_atom_library"

# The nanostructure phases from Chen 2023, Figure 2a and 2b.
EXPECTED_RELATIVE_PHASE_CONSERVED = (
    #  450,   550,   650 nm
    (0.000, 0.000, 0.000),
    (0.451, 0.925, 0.501),
    (1.212, 1.898, 1.137),
    (1.985, 2.783, 1.910),
    (2.471, 3.195, 2.259),
    (3.843, 4.354, 3.693),
    (4.641, 5.140, 4.005),
    (5.489, 5.738, 4.641),
)
EXPECTED_RELATIVE_PHASE_CONVERTED = (
    (0.000, 0.000, 0.000),
    (0.391, 0.900, 0.502),
    (1.137, 2.331, 1.137),
    (1.821, 2.567, 1.945),
    (2.194, 3.052, 2.306),
    (3.749, 4.308, 3.600),
    (4.632, 4.433, 3.861),
    (5.465, 5.540, 4.843),
)


class NanostructurePhaseTest(unittest.TestCase):
    def test_nanostructure_phase_matches_expected(self):
        with open(DESIGNS_DIR / "library1.json") as f:
            serialized = f.read()

        params = json_utils.pytree_from_json(serialized)

        spec = component.LibrarySpec(
            material_ambient=materials.VACUUM,
            material_metasurface_solid=component.TIO2_CHEN,
            material_metasurface_void=materials.VACUUM,
            material_substrate=materials.SIO2,
            background_extinction_coeff=0.0001,
            thickness_ambient=0.0,
            thickness_metasurface=0.6,
            thickness_substrate=0.0,
            pitch=0.4,
            frame_width=0.03,
            grid_spacing=0.005,
        )
        expansion = basis.generate_expansion(
            primitive_lattice_vectors=basis.LatticeVectors(
                u=basis.X * spec.pitch, v=basis.Y * spec.pitch
            ),
            approximate_num_terms=200,
            truncation=basis.Truncation.CIRCULAR,
        )
        response, _ = component.simulate_library(
            density=params["density"],
            spec=spec,
            wavelength=jnp.asarray([0.45, 0.55, 0.65]),
            expansion=expansion,
            formulation=fmm.Formulation.JONES_DIRECT_FOURIER,
            compute_fields=False,
        )
        transmission = jnp.stack(
            [response.transmission_rhcp, response.transmission_lhcp],
            axis=-1,
        )

        def _relative_phase(t):
            t = onp.array(t)
            phase = onp.angle(t)
            phase = phase - phase[0, ...]
            phase = onp.where(phase < 0, 2 * onp.pi + phase, phase)
            phase[1:, :] = onp.where(
                phase[1:, :] - phase[:-1, :] < -onp.pi,
                2 * onp.pi + phase[1:, :],
                phase[1:, :],
            )
            return phase

        phase_conserved = _relative_phase(transmission[:, :, 0, 0])
        phase_converted = _relative_phase(transmission[:, :, 1, 0])

        for color_idx in [0, 1, 2]:
            expected_conserved = onp.asarray(EXPECTED_RELATIVE_PHASE_CONSERVED)[
                :, color_idx
            ]
            expected_converted = onp.asarray(EXPECTED_RELATIVE_PHASE_CONVERTED)[
                :, color_idx
            ]
            for i in range(8):
                with self.subTest(f"nanostructure_{i}_color_{color_idx}_conserved"):
                    onp.testing.assert_allclose(
                        phase_conserved[i, color_idx],
                        expected_conserved[i],
                        rtol=0.12,
                    )
                with self.subTest(f"nanostructure_{i}_color_{color_idx}_converted"):
                    onp.testing.assert_allclose(
                        phase_converted[i, color_idx],
                        expected_converted[i],
                        rtol=0.12,
                    )
