"""Defines the photon extractor challenge.

Copyright (c) 2023 The INVRS-IO authors.
"""

import dataclasses
import functools
from typing import Tuple

import jax
from fmmax import basis, fmm  # type: ignore[import-untyped]
from jax import numpy as jnp
from jax import tree_util
from totypes import symmetry, types

from invrs_gym.challenges import base
from invrs_gym.challenges.extractor import component as extractor_component
from invrs_gym.utils import initializers

ENHANCEMENT_FLUX = "enhancement_flux"
ENHANCEMENT_FLUX_MEAN = "enhancement_flux_mean"
ENHANCEMENT_DOS = "enhancement_dos"
ENHANCEMENT_DOS_MEAN = "enhancement_dos_mean"


density_initializer = functools.partial(
    initializers.noisy_density_initializer,
    relative_stddev=0.1,
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
        bare_substratee_emitted_power: The power emitted by a nitrogen vacancy defect
            in a bare diamond substrate, i.e. without the GaP extractor structure.
        bare_substrate_collected_power: The power collected from a nitrogen vacancy
            defect in a bare diamond structure.
        flux_enhancement_lower_bound: Scalar giving the minimum target for flux
            enhancement. When the flux enhancement exceeds the lower bound, the
            challenge is considered solved.
    """

    component: extractor_component.ExtractorComponent
    bare_substrate_emitted_power: jnp.ndarray
    bare_substrate_collected_power: jnp.ndarray
    flux_enhancement_lower_bound: float

    def loss(self, response: extractor_component.ExtractorResponse) -> jnp.ndarray:
        """Compute a scalar loss from the component `response`."""
        # The response should have a length-3 trailing axis, corresponding to x, y,
        # and z-oriented dipoles.
        assert response.collected_power.shape[-1] == 3
        return -jnp.mean(response.collected_power)

    def distance_to_target(
        self, response: extractor_component.ExtractorResponse
    ) -> jnp.ndarray:
        """Compute distance from the component `response` to the challenge target."""
        enhancement_flux = (
            response.collected_power / self.bare_substrate_collected_power
        )
        return jnp.maximum(
            self.flux_enhancement_lower_bound - jnp.mean(enhancement_flux), 0.0
        )

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
                - mean enhancement of collected flux
                - mean enhancement of dipole density of states
                - the distance to the target flux enhancement
        """
        del params, aux
        enhancement_flux = (
            response.collected_power / self.bare_substrate_collected_power
        )
        enhancement_dos = response.emitted_power / self.bare_substrate_emitted_power
        return {
            ENHANCEMENT_FLUX: enhancement_flux,
            ENHANCEMENT_FLUX_MEAN: jnp.mean(enhancement_flux),
            ENHANCEMENT_DOS: enhancement_dos,
            ENHANCEMENT_DOS_MEAN: jnp.mean(enhancement_dos),
        }


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
)

EXTRACTOR_SIM_PARAMS = extractor_component.ExtractorSimParams(
    grid_spacing=0.01,
    wavelength=0.637,
    formulation=fmm.Formulation.JONES_DIRECT,
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
MINIMUM_WIDTH = 5
MINIMUM_SPACING = 5

# Reference power values used to calculate the enhancement. These were computed
# by `compute_reference_response` with 1600 terms in the Fourier expansion.
BARE_SUBSTRATE_COLLECTED_POWER = jnp.asarray([2.469706, 2.469834, 0.13495])
BARE_SUBSTRATE_EMITTED_POWER = jnp.asarray([73.41745, 73.41583, 84.21051])

# Target is to achieve flux enhancement of 50 times or greater.
FLUX_ENHANCEMENT_LOWER_BOUND = 50.0


def photon_extractor(
    minimum_width: int = MINIMUM_WIDTH,
    minimum_spacing: int = MINIMUM_SPACING,
    density_initializer: base.DensityInitializer = density_initializer,
    bare_substrate_emitted_power: jnp.ndarray = BARE_SUBSTRATE_EMITTED_POWER,
    bare_substrate_collected_power: jnp.ndarray = BARE_SUBSTRATE_COLLECTED_POWER,
    flux_enhancement_lower_bound: float = FLUX_ENHANCEMENT_LOWER_BOUND,
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
        minimum_width: The minimum width target for the challenge, in pixels. The
            physical minimum width is approximately 180 nm.
        minimum_spacing: The minimum spacing target for the challenge, in pixels.
        density_initializer: Callble which returns the initial density, given a
            key and seed density.
        bare_substrate_emitted_power: The power emitted by a nitrogen vacancy defect
            in a bare diamond substrate, i.e. without the GaP extractor structure.
        bare_substrate_collected_power: The power collected from a nitrogen vacancy
            defect in a bare diamond structure.
        flux_enhancement_lower_bound: Scalar giving the minimum target for flux
            enhancement. When the flux enhancement exceeds the lower bound, the
            challenge is considered solved.
        spec: Defines the physical specification of the metagrating.
        sim_params: Defines the simulation settings of the metagrating.
        symmetries: Defines the symmetries of the metagrating.

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
        bare_substrate_emitted_power=bare_substrate_emitted_power,
        bare_substrate_collected_power=bare_substrate_collected_power,
        flux_enhancement_lower_bound=flux_enhancement_lower_bound,
    )


def bare_substrate_response(
    spec: extractor_component.ExtractorSpec = EXTRACTOR_SPEC,
    sim_params: extractor_component.ExtractorSimParams = EXTRACTOR_SIM_PARAMS,
) -> extractor_component.ExtractorResponse:
    """Computes the response of the nitrogen vacancy in a bare diamond substrate."""
    component = extractor_component.ExtractorComponent(
        spec=spec,
        sim_params=sim_params,
        density_initializer=lambda _, d: tree_util.tree_map(jnp.zeros_like, d),
    )
    params = component.init(jax.random.PRNGKey(0))
    response, _ = component.response(params)
    return response
