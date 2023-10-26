"""Tests that simulations of the reference extractor gives expected results.

Copyright (c) 2023 The INVRS-IO authors.
"""

import dataclasses
import pathlib
import unittest

import jax
import jax.numpy as jnp
import numpy as onp
import pytest
from fmmax import basis

from invrs_gym.challenges.extractor import challenge

REPO_PATH = pathlib.Path(__file__).resolve().parent.parent.parent.parent
DESIGNS_DIR = REPO_PATH / "reference_designs/photon_extractor"


class ReferenceExtractorTest(unittest.TestCase):
    @pytest.mark.slow
    def test_boost_matches_expected(self):
        # Larger number of terms lets us model dipoles with narrower spatial
        # distributions, so as better to match the reference results.
        spec = dataclasses.replace(challenge.EXTRACTOR_SPEC, fwhm_source=0.0)
        sim_params = dataclasses.replace(
            challenge.EXTRACTOR_SIM_PARAMS, approximate_num_terms=1200
        )

        # Compute the response of the reference design.
        pec = challenge.photon_extractor(spec=spec, sim_params=sim_params)
        params = pec.component.init(jax.random.PRNGKey(0))

        # Create parameters by loading the reference device. The region surrounding the
        # design is not included in the csv; pad to the correct shape.
        density_array = onp.genfromtxt(DESIGNS_DIR / "device1.csv", delimiter=",")
        assert density_array.shape == (150, 150)
        assert 1000 * spec.pitch / pec.component.grid_shape[0] == 10
        pad = (params.shape[0] - density_array.shape[0]) // 2
        density_array = onp.pad(density_array, ((pad, pad), (pad, pad)))
        assert density_array.shape == params.shape
        extractor_params = dataclasses.replace(params, array=jnp.asarray(density_array))

        # Jit-compile the response function to avoid duplicated eigensolves.
        response_fn = jax.jit(pec.component.response)

        # Compute the response.
        extractor_response, _ = response_fn(extractor_params)

        # Compute the response of a bare diamond substrate.
        bare_substrate_params = dataclasses.replace(
            extractor_params,
            array=jnp.zeros_like(extractor_params.array),
        )
        bare_response, _ = response_fn(bare_substrate_params)

        flux_boost_jx = (
            extractor_response.collected_power[0] / bare_response.collected_power[0]
        )
        flux_boost_jy = (
            extractor_response.collected_power[1] / bare_response.collected_power[1]
        )
        flux_boost_jz = (
            extractor_response.collected_power[2] / bare_response.collected_power[2]
        )

        dos_boost_jx = (
            extractor_response.emitted_power[0] / bare_response.emitted_power[0]
        )
        dos_boost_jy = (
            extractor_response.emitted_power[1] / bare_response.emitted_power[1]
        )
        dos_boost_jz = (
            extractor_response.emitted_power[2] / bare_response.emitted_power[2]
        )

        # Expected values were extracted from Fig. 1c of "Inverse-designed photon
        # extractors for optically addressable defect qubits". These are for point
        # dipoles, which are much smaller than the dipoles we can resolve with FMM.
        # Consequently, the enhancements calculated by FMM are strictly lower. The
        # supplementary material of the article shows strong dependence of
        # enhancement on dipole position, and so the tolerances here seem reasonable.
        # https://opg.optica.org/optica/fulltext.cfm?uri=optica-7-12-1805
        expected_flux_boost_jx = expected_flux_boost_jy = 10.8
        expected_flux_boost_jz = 357.2
        expected_dos_boost_jx = expected_dos_boost_jy = 1.42
        expected_dos_boost_jz = 1.35

        onp.testing.assert_allclose(flux_boost_jx, expected_flux_boost_jx, rtol=0.15)
        onp.testing.assert_allclose(flux_boost_jy, expected_flux_boost_jy, rtol=0.15)
        onp.testing.assert_allclose(flux_boost_jz, expected_flux_boost_jz, rtol=0.48)

        self.assertLess(flux_boost_jx, expected_flux_boost_jx)
        self.assertLess(flux_boost_jy, expected_flux_boost_jy)
        self.assertLess(flux_boost_jz, expected_flux_boost_jz)

        onp.testing.assert_allclose(dos_boost_jx, expected_dos_boost_jx, rtol=0.10)
        onp.testing.assert_allclose(dos_boost_jy, expected_dos_boost_jy, rtol=0.10)
        onp.testing.assert_allclose(dos_boost_jz, expected_dos_boost_jz, rtol=0.10)

        self.assertLess(dos_boost_jx, expected_dos_boost_jx)
        self.assertLess(dos_boost_jy, expected_dos_boost_jy)
        self.assertLess(dos_boost_jz, expected_dos_boost_jz)

    @pytest.mark.slow
    def test_convergence(self):
        # Test that the simulations of the reference design are converged.
        pec = challenge.photon_extractor()

        params = pec.component.init(jax.random.PRNGKey(0))

        # Create parameters by loading the reference device. The region surrounding the
        # design is not included in the csv; pad to the correct shape.
        density_array = onp.genfromtxt(DESIGNS_DIR / "device1.csv", delimiter=",")
        assert density_array.shape == (150, 150)
        pad = (params.shape[0] - density_array.shape[0]) // 2
        density_array = onp.pad(density_array, ((pad, pad), (pad, pad)))
        assert density_array.shape == params.shape
        extractor_params = dataclasses.replace(params, array=jnp.asarray(density_array))

        responses = []
        for approximate_num_terms in [1200, 1600]:
            expansion = basis.generate_expansion(
                primitive_lattice_vectors=basis.LatticeVectors(
                    u=pec.component.spec.pitch * basis.X,
                    v=pec.component.spec.pitch * basis.Y,
                ),
                approximate_num_terms=approximate_num_terms,
                truncation=pec.component.sim_params.truncation,
            )
            response, _ = pec.component.response(extractor_params, expansion=expansion)
            responses.append(response)

        response_1200, response_1600 = responses

        onp.testing.assert_allclose(
            response_1200.emitted_power,
            response_1600.emitted_power,
            rtol=0.05,
        )

        # Collected power for z-oriented dipoles converges a bit more slowly. Use a
        # larger tolerance for comparison.
        onp.testing.assert_allclose(
            response_1200.collected_power[:2],
            response_1600.collected_power[:2],
            rtol=0.08,
        )
        onp.testing.assert_allclose(
            response_1200.collected_power[2],
            response_1600.collected_power[2],
            rtol=0.18,
        )
