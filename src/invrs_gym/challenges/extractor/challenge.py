"""Defines the photon extractor challenge.

Copyright (c) 2023 The INVRS-IO authors.
"""

import dataclasses
import functools
from typing import Tuple

from fmmax import basis, fmm  # type: ignore[import-untyped]
from jax import numpy as jnp
from totypes import symmetry, types

from invrs_gym.challenges import base
from invrs_gym.challenges.extractor import component as extractor_component
from invrs_gym.utils import initializers

ENHANCEMENT_FLUX_PER_DIPOLE = "enhancement_flux_per_dipole"
ENHANCEMENT_FLUX_TOTAL = "enhancement_flux_total"
ENHANCEMENT_DOS_PER_DIPOLE = "enhancement_dos_per_dipole"
ENHANCEMENT_DOS_TOTAL = "enhancement_dos_total"


density_initializer = functools.partial(
    initializers.noisy_density_initializer,
    relative_mean=0.5,
    relative_noise_amplitude=0.1,
)


@dataclasses.dataclass
class PhotonExtractorChallenge(base.Challenge):
    """Defines the photon extractor challenge.

    The challenge is based on "Inverse-designed photon extractors for optically
    addressable defect qubits" by Chakravarthi et al. It involves optimizing a GaP
    patterned layer on diamond substrate above an implanted nitrogen vacancy defect.
    An oxide hard mask used to pattern the GaP is left in place after the etch.

    The goal of the optimization is to maximize extraction of 637 nm emission, i.e.
    to maximize the power coupled from the defect to the ambient above the extractor.

    https://opg.optica.org/optica/fulltext.cfm?uri=optica-7-12-1805

    Attributes:
        component: The component to be designed.
    """

    component: extractor_component.ExtractorComponent

    def loss(self, response: extractor_component.ExtractorResponse) -> jnp.ndarray:
        """Compute a scalar loss from the component `response`."""
        # The response should have a length-3 trailing axis, corresponding to x, y,
        # and z-oriented dipoles.
        assert response.collected_power.shape[-1] == 3
        return -jnp.mean(response.collected_power)

    def eval_metric(
        self,
        response: extractor_component.ExtractorResponse,
    ) -> jnp.ndarray:
        """Computes the eval metric from the component `response`.

        The evaluation metric is the enhancement in collected power, summing over all
        wavelengths and dipole polarizations.

        Args:
            response: The component response.

        Returns:
            The scalar eval metric.
        """
        assert response.collected_power.shape[-1] == 3
        enhancement_flux_total = jnp.sum(response.collected_power) / jnp.sum(
            response.bare_substrate_collected_power
        )
        return enhancement_flux_total

    def metrics(
        self,
        response: extractor_component.ExtractorResponse,
        params: types.Density2DArray,
        aux: base.AuxDict,
    ) -> base.AuxDict:
        """Compute challenge metrics.

        Args:
            response: The response of the extractor component.
            params: The parameters where the response was evaluated.
            aux: The auxilliary quantities returned by the component response method.

        Returns:
            The metrics dictionary, with the following quantities:
                - the per-dipole flux enhancement
                - the total flux enhancment
                - the per-dipole density of states enhancement
                - the total density of states enhancement
        """
        metrics = super().metrics(response, params, aux)
        enhancement_flux_per_dipole = (
            response.collected_power / response.bare_substrate_collected_power
        )
        enhancement_flux_total = jnp.sum(response.collected_power) / jnp.sum(
            response.bare_substrate_collected_power
        )
        enhancement_dos_per_dipole = (
            response.emitted_power / response.bare_substrate_emitted_power
        )
        enhancement_dos_total = jnp.sum(response.emitted_power) / jnp.sum(
            response.bare_substrate_emitted_power
        )
        metrics.update(
            {
                ENHANCEMENT_FLUX_PER_DIPOLE: enhancement_flux_per_dipole,
                ENHANCEMENT_FLUX_TOTAL: enhancement_flux_total,
                ENHANCEMENT_DOS_PER_DIPOLE: enhancement_dos_per_dipole,
                ENHANCEMENT_DOS_TOTAL: enhancement_dos_total,
            }
        )
        return metrics


EXTRACTOR_SPEC = extractor_component.ExtractorSpec(
    permittivity_ambient=(1.0 + 0.0j) ** 2,
    permittivity_oxide=(1.46 + 0.0j) ** 2,
    permittivity_extractor=(3.31 + 0.0j) ** 2,
    permittivity_substrate=(2.4102 + 0.0j) ** 2,
    thickness_ambient=1.0,
    thickness_oxide=0.13,
    thickness_extractor=0.25,
    thickness_substrate_before_source=0.1,
    thickness_substrate_after_source=0.9,
    width_design_region=1.5,
    width_padding=0.25,
    width_pml=0.4,
    fwhm_source=0.05,
    offset_monitor_source=0.025,
    offset_monitor_ambient=0.4,
    width_monitor_ambient=1.5,
    grid_spacing=0.005,
)

EXTRACTOR_SIM_PARAMS = extractor_component.ExtractorSimParams(
    wavelength=0.637,
    formulation=fmm.Formulation.JONES_DIRECT_FOURIER,
    approximate_num_terms=1200,
    truncation=basis.Truncation.CIRCULAR,
)

SYMMETRIES: Tuple[str, ...] = (
    symmetry.REFLECTION_N_S,
    symmetry.REFLECTION_E_W,
    symmetry.REFLECTION_NE_SW,
    symmetry.REFLECTION_NW_SE,
)

# Minimum width and spacing are 50 nm for the default dimensions.
MINIMUM_WIDTH = 10
MINIMUM_SPACING = 10


def photon_extractor(
    minimum_width: int = MINIMUM_WIDTH,
    minimum_spacing: int = MINIMUM_SPACING,
    density_initializer: base.DensityInitializer = density_initializer,
    spec: extractor_component.ExtractorSpec = EXTRACTOR_SPEC,
    sim_params: extractor_component.ExtractorSimParams = EXTRACTOR_SIM_PARAMS,
    symmetries: Tuple[str, ...] = SYMMETRIES,
) -> PhotonExtractorChallenge:
    """Photon extractor with 1.5 x 1.5 um design region.

    The challenge is based on "Inverse-designed photon extractors for optically
    addressable defect qubits" by Chakravarthi et al. It involves optimizing a GaP
    patterned layer on diamond substrate above an implanted nitrogen vacancy defect.
    An oxide hard mask used to pattern the GaP is left in place after the etch.

    The goal of the optimization is to maximize extraction of 637 nm emission, i.e.
    to maximize the power coupled from the defect to the ambient above the extractor.

    https://opg.optica.org/optica/fulltext.cfm?uri=optica-7-12-1805

    Args:
        minimum_width: The minimum width target for the challenge, in pixels.  The
            default value of 10 corresponds to a physical size of approximately 50 nm.
        minimum_spacing: The minimum spacing target for the challenge, in pixels.
        density_initializer: Callble which returns the initial density, given a
            key and seed density.
        spec: Defines the physical specification of the photon extractor.
        sim_params: Defines the simulation settings of the photon extractor.
        symmetries: Defines the symmetries of the photon extractor.

    Returns:
        The `PhotonExtractorChallenge`.
    """
    return PhotonExtractorChallenge(
        component=extractor_component.ExtractorComponent(
            spec=spec,
            sim_params=sim_params,
            density_initializer=density_initializer,
            minimum_width=minimum_width,
            minimum_spacing=minimum_spacing,
            symmetries=symmetries,
        ),
    )
