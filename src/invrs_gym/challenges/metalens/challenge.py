"""Defines the metalens challenge.

Copyright (c) 2023 The INVRS-IO authors.
"""

import dataclasses
import functools
from typing import Tuple

from fmmax import fmm  # type: ignore[import-untyped]
from jax import numpy as jnp
from totypes import symmetry, types

from invrs_gym.challenges import base
from invrs_gym.challenges.metalens import component as metalens_component
from invrs_gym.utils import initializers


ENHANCEMENT_EX_MEAN = "enhancement_ex_mean"
ENHANCEMENT_EX_MIN = "enhancement_ex_min"
ENHANCEMENT_EY_MEAN = "enhancement_ey_mean"
ENHANCEMENT_EY_MIN = "enhancement_ey_min"

EX = "ex"
EY = "ey"


density_initializer = functools.partial(
    initializers.noisy_density_initializer,
    relative_mean=0.5,
    relative_noise_amplitude=0.1,
)


@dataclasses.dataclass
class MetalensChallenge(base.Challenge):
    """Defines the metalens challenge.

    The challenge is based on the RGB metalens problem from "Validation and
    characterization of algorithms and software for photonics inverse design" by
    Chen et al. It involves optimization of a metalens to focus 450, 550, and 650 nm
    light a distance 2.4 um away from a 1D metalens having a width of 10 um.

    https://opg.optica.org/josab/ViewMedia.cfm?uri=josab-41-2-A161
    https://github.com/NanoComp/photonics-opt-testbed/tree/main/RGB_metalens

    Attributes:
        component: The component to be designed.
        intensity_enhancement_lower_bound: The lower bound for intensity enhancement.
            When enhancement exceeds this value, the challenge is considered solved.
        incident_field: Either `EX` or `EY`, specifying whether the target of the
            challenge is to optimize for excitation with x-polarized or y-polarized
            electric fields.
    """

    component: metalens_component.MetalensComponent
    intensity_enhancement_lower_bound: float
    incident_field: str

    def loss(self, response: metalens_component.MetalensResponse) -> jnp.ndarray:
        """Compute a scalar loss from the component `response`."""
        assert self.incident_field in (EX, EY)
        if self.incident_field == EX:
            enhancement = response.enhancement_ex
        else:
            enhancement = response.enhancement_ey
        return -jnp.mean(enhancement)

    def distance_to_target(
        self, response: metalens_component.MetalensResponse
    ) -> jnp.ndarray:
        """Compute distance from the component `response` to the challenge target."""
        assert self.incident_field in (EX, EY)
        if self.incident_field == EX:
            enhancement = response.enhancement_ex
        else:
            enhancement = response.enhancement_ey
        return self.intensity_enhancement_lower_bound - jnp.amin(enhancement)

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
        metrics.update(
            {
                ENHANCEMENT_EX_MIN: jnp.amin(response.enhancement_ex),
                ENHANCEMENT_EX_MEAN: jnp.mean(response.enhancement_ex),
                ENHANCEMENT_EY_MIN: jnp.amin(response.enhancement_ey),
                ENHANCEMENT_EY_MEAN: jnp.mean(response.enhancement_ey),
            }
        )
        return metrics


METALENS_SPEC = metalens_component.MetalensSpec(
    permittivity_ambient=(1.0 + 0.0001j) ** 2,
    permittivity_metalens=(2.4 + 0.0001j) ** 2,
    permittivity_substrate=(2.4 + 0.0001j) ** 2,
    thickness_ambient=1.0,  # Above the focal point
    thickness_lens=1.0,
    thickness_substrate=1.0,
    width_lens=10.0,
    width_pml=0.0,
    pml_lens_offset=3.0,
    pml_source_offset=2.5,
    source_smoothing_fwhm=1.0,
    focus_offset=2.4,
    grid_spacing=0.02,
)

METALENS_SIM_PARAMS = metalens_component.MetalensSimParams(
    wavelength=jnp.asarray([0.45, 0.55, 0.65]),
    approximate_num_terms=280,
    formulation=fmm.Formulation.JONES_DIRECT_FOURIER,
    num_layers=25,
)

SYMMETRIES: Tuple[str, ...] = (symmetry.REFLECTION_N_S,)

# Minimum width and spacing are 100 nm for the default dimensions.
MINIMUM_WIDTH = 5
MINIMUM_SPACING = 5

INTENSITY_ENHANCEMENT_LOWER_BOUND = 15.0


def metalens(
    minimum_width: int = MINIMUM_WIDTH,
    minimum_spacing: int = MINIMUM_SPACING,
    density_initializer: base.DensityInitializer = density_initializer,
    intensity_enhancement_lower_bound: float = INTENSITY_ENHANCEMENT_LOWER_BOUND,
    incident_field: str = EX,
    spec: metalens_component.MetalensSpec = METALENS_SPEC,
    sim_params: metalens_component.MetalensSimParams = METALENS_SIM_PARAMS,
    symmetries: Tuple[str, ...] = SYMMETRIES,
) -> MetalensChallenge:
    """Metalens extractor with 10.0 x 1.0 um design region.

    The challenge is based on the RGB metalens problem from "Validation and
    characterization of algorithms and software for photonics inverse design" by
    Chen et al. It involves optimization of a metalens to focus 450, 550, and 650 nm
    light a distance 2.4 um away from a 1D metalens having a width of 10 um.

    https://opg.optica.org/josab/ViewMedia.cfm?uri=josab-41-2-A161
    https://github.com/NanoComp/photonics-opt-testbed/tree/main/RGB_metalens

    Args:
        minimum_width: The minimum width target for the challenge, in pixels.  The
            default value of 5 corresponds to a physical size of approximately 50 nm.
        minimum_spacing: The minimum spacing target for the challenge, in pixels.
        density_initializer: Callble which returns the initial density, given a
            key and seed density.
        intensity_enhancement_lower_bound: The lower bound for intensity enhancement.
            When enhancement exceeds this value, the challenge is considered solved.
        incident_field: Either `EX` or `EY`, specifying whether the target of the
            challenge is to optimize for excitation with x-polarized or y-polarized
            electric fields.
        spec: Defines the physical specification of the metalens.
        sim_params: Defines the simulation settings of the metalens.
        symmetries: Defines the symmetries of the metalens.

        symmetries: Tuple[str, ...] = SYMMETRIES,

    Returns:
        The `MetalensChallenge`.
    """
    return MetalensChallenge(
        component=metalens_component.MetalensComponent(
            spec=spec,
            sim_params=sim_params,
            density_initializer=density_initializer,
            minimum_width=minimum_width,
            minimum_spacing=minimum_spacing,
            symmetries=symmetries,
        ),
        intensity_enhancement_lower_bound=intensity_enhancement_lower_bound,
        incident_field=incident_field,
    )
