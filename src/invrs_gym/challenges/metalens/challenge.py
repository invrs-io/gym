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
from invrs_gym.challenges.metalens import component as metalens_component
from invrs_gym.utils import initializers

EX = "ex"
EY = "ey"


density_initializer = functools.partial(
    initializers.noisy_density_initializer,
    relative_mean=0.5,
    relative_noise_amplitude=0.1,
)


@dataclasses.dataclass
class MetalensChallenge(base.Challenge):
    """Defines the metalens challenge."""

    component: metalens_component.MetalensComponent
    bare_substrate_intensity_at_focus: jnp.ndarray
    intensity_enhancement_lower_bound: float
    incident_field: str

    def loss(self, response: metalens_component.MetalensResponse) -> jnp.ndarray:
        """Compute a scalar loss from the component `response`."""
        # The response should have a length-3 trailing axis, corresponding to x, y,
        # and z-oriented dipoles.
        intensity = (
            jnp.abs(response.ex) ** 2
            + jnp.abs(response.ey) ** 2
            + jnp.abs(response.ex) ** 2
        )
        assert intensity.shape[-1] == 2
        polarization_idx = 0 if self.incident_field == EX else 1
        intensity = intensity[..., polarization_idx]
        return -jnp.mean(intensity)

    def distance_to_target(
        self, response: metalens_component.MetalensResponse
    ) -> jnp.ndarray:
        """Compute distance from the component `response` to the challenge target."""
        return 1

    def metrics(
        self,
        response: metalens_component.MetalensResponse,
        params: types.Density2DArray,
        aux: base.AuxDict,
    ) -> base.AuxDict:
        """Compute challenge metrics.

        Args:
            response: The response of the metalens component.
            params: The parameters where the metalens was evaluated.
            aux: The auxilliary quantities returned by the component response method.

        Returns:
            The metrics dictionary.
        """
        metrics = super().metrics(response, params, aux)
        return metrics


METALENS_SPEC = metalens_component.MetalensSpec(
    permittivity_ambient=(1.0 + 0.00001j) ** 2,
    permittivity_metalens=(2.4 + 0.00001j) ** 2,
    permittivity_substrate=(2.4 + 0.00001j) ** 2,
    thickness_ambient=1.0,  # Beyond the focal point
    thickness_lens=1.0,
    thickness_substrate=1.0,
    width_lens=10.0,
    lens_offset=4.0,
    source_offset=3.0,
    source_smoothing_fwhm=1.0,
    focus_offset=2.4,
    grid_spacing=0.02,
)

METALENS_SIM_PARAMS = metalens_component.MetalensSimParams(
    wavelength=jnp.asarray([0.45]),  # , 0.55, 0.65]]]),
    approximate_num_terms=400,
    formulation=fmm.Formulation.JONES_DIRECT_FOURIER,  # Use a custom vector function here
    num_layers=8,
)

SYMMETRIES: Tuple[str, ...] = (symmetry.REFLECTION_N_S,)

# Minimum width and spacing are 50 nm for the default dimensions.
MINIMUM_WIDTH = 5
MINIMUM_SPACING = 5

INTENSITY_ENHANCEMENT_LOWER_BOUND = 20.0


def metalens(
    minimum_width: int = MINIMUM_WIDTH,
    minimum_spacing: int = MINIMUM_SPACING,
    density_initializer: base.DensityInitializer = density_initializer,
    bare_substrate_intensity_at_focus: jnp.ndarray = 1,
    intensity_enhancement_lower_bound: float = INTENSITY_ENHANCEMENT_LOWER_BOUND,
    spec: metalens_component.MetalensSpec = METALENS_SPEC,
    sim_params: metalens_component.MetalensSimParams = METALENS_SIM_PARAMS,
    incident_field=EX,
    symmetries: Tuple[str, ...] = SYMMETRIES,
) -> MetalensChallenge:
    """Metalens extractor with 10.0 x 1.0 um design region."""
    return MetalensChallenge(
        component=metalens_component.MetalensComponent(
            spec=spec,
            sim_params=sim_params,
            density_initializer=density_initializer,
            minimum_width=minimum_width,
            minimum_spacing=minimum_spacing,
            symmetries=symmetries,
        ),
        bare_substrate_intensity_at_focus=bare_substrate_intensity_at_focus,
        intensity_enhancement_lower_bound=intensity_enhancement_lower_bound,
        incident_field=incident_field,
    )
