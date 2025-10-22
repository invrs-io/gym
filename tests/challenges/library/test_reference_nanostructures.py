"""Tests involving simulations of reference nanostructure.

The reference nanostructures are from "Dispersion-engineered metasurfaces reaching
broadband 90% relative diffraction efficiency" by Chen et al. (2023).
https://www.nature.com/articles/s41467-023-38185-2

Copyright (c) 2025 invrs.io LLC
"""

import unittest

import fmmax
import jax.numpy as jnp
import numpy as onp
from totypes import types

from invrs_gym.challenges.library import component
from invrs_gym.utils import materials


def _plus(x1, x2, y1, y2, dim):
    """Return density array for plus-shaped nanostructures."""
    x, y = onp.meshgrid(
        onp.arange(-dim // 2, dim // 2),
        onp.arange(-dim // 2, dim // 2),
        indexing="ij",
    )
    p = onp.ones(x.shape, dtype=bool)
    p &= (x < x1 / 2) & (x > -x1 / 2)
    p &= (y < y1 / 2) & (y > -y1 / 2)
    p &= (x < x2 / 2) & (x > -x2 / 2) | ((y < y2 / 2) & (y > -y2 / 2))
    return p.astype(float)


def _ibeam(x1, x2, y1, y2, dim, rotate):
    """Return density array for ibeam-shaped nanostructures."""
    x, y = onp.meshgrid(
        onp.arange(-dim // 2, dim // 2),
        onp.arange(-dim // 2, dim // 2),
        indexing="ij",
    )
    p = onp.ones(x.shape, dtype=bool)
    p &= (x < x1 / 2) & (x > -x1 / 2)
    p &= (y < y1 / 2) & (y > -y1 / 2)
    p &= (x > x2 / 2) | (x < -x2 / 2) | (y < (y1 / 2 - y2)) & (y > -(y1 / 2 - y2))
    if rotate:
        p = onp.rot90(p)
    return p.astype(float)


def get_nanostructure(design, **kwargs):
    """Return nanostructures of the specified design type."""
    if design == "plus":
        return _plus(**kwargs)
    elif design == "ibeam":
        return _ibeam(**kwargs)


def get_nanostructures(specs):
    """Return all nanostructures in a single density object with batch dimensions."""
    arrays = []
    for spec in specs:
        density = get_nanostructure(**spec)
        arrays.append(density)
    return types.Density2DArray(
        array=onp.stack(arrays, axis=0),
        lower_bound=0,
        upper_bound=1,
    )


NANOSTRUCTURE_SPEC = (
    dict(design="plus", x1=156, x2=86, y1=140, y2=86, dim=400),
    dict(design="plus", x1=230, x2=60, y1=220, y2=60, dim=400),
    dict(design="ibeam", y1=200, y2=60, x1=200, x2=80, dim=400, rotate=False),
    dict(design="ibeam", y1=320, y2=120, x1=180, x2=60, dim=400, rotate=True),
    dict(design="ibeam", y1=340, y2=120, x1=180, x2=60, dim=400, rotate=True),
    dict(design="ibeam", y1=340, y2=130, x1=280, x2=80, dim=400, rotate=True),
    dict(design="ibeam", y1=320, y2=110, x1=300, x2=60, dim=400, rotate=True),
    dict(design="ibeam", y1=340, y2=90, x1=300, x2=60, dim=400, rotate=True),
)

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
        density = get_nanostructures(specs=NANOSTRUCTURE_SPEC)

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
            grid_spacing=0.001,
        )
        expansion = fmmax.generate_expansion(
            primitive_lattice_vectors=fmmax.LatticeVectors(
                u=fmmax.X * spec.pitch, v=fmmax.Y * spec.pitch
            ),
            approximate_num_terms=200,
            truncation=fmmax.Truncation.CIRCULAR,
        )
        response, _ = component.simulate_library(
            density=density,
            spec=spec,
            wavelength=jnp.asarray([0.45, 0.55, 0.65]),
            expansion=expansion,
            formulation=fmmax.Formulation.JONES_DIRECT_FOURIER,
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
                        atol=0.44,
                    )
                with self.subTest(f"nanostructure_{i}_color_{color_idx}_converted"):
                    onp.testing.assert_allclose(
                        phase_converted[i, color_idx],
                        expected_converted[i],
                        atol=0.44,
                    )
