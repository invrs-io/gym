"""Defines the photon extractor challenge.

Copyright (c) 2023 The INVRS-IO authors.
"""

import dataclasses
import functools

from fmmax import basis, fmm  # type: ignore[import-untyped]
from jax import nn
from jax import numpy as jnp
from totypes import types

from invrs_gym.challenges import base
from invrs_gym.challenges.sorter import common
from invrs_gym.utils import initializers

POLARIZATION_RATIO_MIN = "polarization_ratio_min"
POLARIZATION_RATIO_MEAN = "polarization_ratio_mean"
EFFICIENCY_MIN = "efficiency_min"
EFFICIENCY_MEAN = "efficiency_mean"


density_initializer = functools.partial(
    initializers.noisy_density_initializer,
    relative_mean=0.5,
    relative_noise_amplitude=0.1,
)


@dataclasses.dataclass
class PolarizationSorterChallenge(base.Challenge):
    """Defines the polarization sorter challenge.

    The target of the polarization sorter challenge is to achieve coupling of incident
    plane waves into four individual pixels, depending upon the polarization of the
    incident wave.

    Attributes:
        component: The component to be optimized.
        efficiency_target: The target efficiency for the coupling of e.g. an
            x-polarized plane wave into its designated pixel. The theoretical maximum
            is 0.5.
        polarization_ratio_target: The target ratio of power coupled for e.g. an
            x-polarized plane wave into the "x-polarized pixel" and the power for
            the x-polarized plane wave into the "y-polarized pixel".
    """

    component: common.SorterComponent
    efficiency_target: float
    polarization_ratio_target: float

    def loss(self, response: common.SorterResponse) -> jnp.ndarray:
        """Compute a scalar loss from the component `response`."""
        # Include a loss term that penalizes unphysical results, which can help prevent
        # an optimizer from exploiting inaccuracies in the simulation when the number
        # of Fourier orders is insufficient.
        total_power = response.reflection + jnp.sum(response.transmission, axis=-1)
        excess_power = nn.relu(total_power - 1)
        excess_power_loss = 10 * jnp.sum(excess_power**2)

        ideal_transmission = jnp.asarray(
            [
                # Q1,  Q2,   Q3,   Q4
                [0.50, 0.25, 0.25, 0.00],  # x
                [0.25, 0.50, 0.00, 0.25],  # (x + y) / sqrt(2)
                [0.25, 0.00, 0.50, 0.25],  # (x - y) / sqrt(2)
                [0.00, 0.25, 0.25, 0.50],  # y
            ]
        )
        transmission_loss = jnp.sum((response.transmission - ideal_transmission) ** 2)
        return excess_power_loss + transmission_loss

    def distance_to_target(self, response: common.SorterResponse) -> jnp.ndarray:
        """Compute distance from the component `response` to the challenge target."""
        on_target_transmission = response.transmission[
            ..., tuple(range(4)), tuple(range(4))
        ]
        min_efficiency = jnp.amin(on_target_transmission)

        off_target_transmission = response.transmission[
            ..., tuple(range(4))[::-1], tuple(range(4))
        ]
        min_polarization_ratio = jnp.amin(
            on_target_transmission / off_target_transmission
        )
        return jnp.maximum(
            self.polarization_ratio_target - min_polarization_ratio, 0.0
        ) + jnp.maximum(self.efficiency_target - min_efficiency, 0.0)

    def metrics(
        self,
        response: common.SorterResponse,
        params: common.Params,
        aux: base.AuxDict,
    ) -> base.AuxDict:
        """Compute challenge metrics.

        Args:
            response: The response of the sorter component.
            params: The parameters where the response was evaluated.
            aux: The auxilliary quantities returned by the component response method.

        Returns:
            The metrics dictionary, with the following quantities:
                - minimum polarization ratio
                - mean polarization ratio
                - minimum efficiency
                - mean efficiency
        """
        del params, aux
        on_target_transmission = response.transmission[
            ..., tuple(range(4)), tuple(range(4))
        ]
        efficiency = on_target_transmission

        off_target_transmission = response.transmission[
            ..., tuple(range(4))[::-1], tuple(range(4))
        ]
        polarization_ratio = on_target_transmission / off_target_transmission
        return {
            EFFICIENCY_MEAN: jnp.mean(efficiency),
            EFFICIENCY_MIN: jnp.amin(efficiency),
            POLARIZATION_RATIO_MEAN: jnp.mean(polarization_ratio),
            POLARIZATION_RATIO_MIN: jnp.amin(polarization_ratio),
        }


POLARIZATION_SORTER_SPEC = common.SorterSpec(
    permittivity_ambient=(1.0 + 0.0j) ** 2,
    permittivity_cap=(1.5 + 0.00001j) ** 2,
    permittivity_metasurface_solid=(4.0 + 0.00001j) ** 2,
    permittivity_metasurface_void=(1.5 + 0.00001j) ** 2,
    permittivity_spacer=(1.5 + 0.00001j) ** 2,
    permittivity_substrate=(4.0730 + 0.028038j) ** 2,
    thickness_cap=types.BoundedArray(0.05, lower_bound=0.00, upper_bound=0.5),
    thickness_metasurface=types.BoundedArray(0.15, lower_bound=0.1, upper_bound=0.3),
    thickness_spacer=types.BoundedArray(1.0, lower_bound=0.8, upper_bound=1.2),
    pitch=2.0,
    offset_monitor_substrate=0.1,
)

POLARIZATION_SORTER_SIM_PARAMS = common.SorterSimParams(
    grid_spacing=0.01,
    wavelength=0.55,
    polar_angle=0.0,
    azimuthal_angle=0.0,
    formulation=fmm.Formulation.JONES_DIRECT_FOURIER,
    approximate_num_terms=1200,
    truncation=basis.Truncation.CIRCULAR,
)

# Minimum width and spacing are 80 nm for the default dimensions.
MINIMUM_WIDTH = 8
MINIMUM_SPACING = 8

# Target metrics for the sorter component.
EFFICIENCY_TARGET = 0.4
POLARIZATION_RATIO_TARGET = 10


def polarization_sorter(
    minimum_width: int = MINIMUM_WIDTH,
    minimum_spacing: int = MINIMUM_SPACING,
    thickness_initializer: common.ThicknessInitializer = (
        initializers.identity_initializer
    ),
    density_initializer: base.DensityInitializer = density_initializer,
    efficiency_target: float = EFFICIENCY_TARGET,
    polarization_ratio_target: float = POLARIZATION_RATIO_TARGET,
    spec: common.SorterSpec = POLARIZATION_SORTER_SPEC,
    sim_params: common.SorterSimParams = POLARIZATION_SORTER_SIM_PARAMS,
) -> PolarizationSorterChallenge:
    """Polarization sorter challenge.

    Args:
        minimum_width: The minimum width target for the challenge, in pixels. The
            physical minimum width is approximately 80 nm.
        minimum_spacing: The minimum spacing target for the challenge, in pixels.
        thickness_initializer: Callable which returns the initial thickness, given a
            key and seed thickness.
        density_initializer: Callable which returns the initial density, given a
            key and seed density.
        efficiency_target: The target efficiency for the coupling of e.g. an
            x-polarized plane wave into its designated pixel. The theoretical maximum
            is 0.5.
        polarization_ratio_target: The target ratio of power coupled for e.g. an
            x-polarized plane wave into the "x-polarized pixel" and the power for
            the x-polarized plane wave into the "y-polarized pixel".
        spec: Defines the physical specification of the polarization sorter.
        sim_params: Defines the simulation settings of the polarization sorter.

    Returns:
        The `PolarizationSorterChallenge`.
    """
    return PolarizationSorterChallenge(
        component=common.SorterComponent(
            spec=spec,
            sim_params=sim_params,
            thickness_initializer=thickness_initializer,
            density_initializer=density_initializer,
            minimum_width=minimum_width,
            minimum_spacing=minimum_spacing,
        ),
        efficiency_target=efficiency_target,
        polarization_ratio_target=polarization_ratio_target,
    )
