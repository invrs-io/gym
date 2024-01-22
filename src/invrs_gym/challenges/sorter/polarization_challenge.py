"""Defines the polarization sorter challenge.

Copyright (c) 2023 The INVRS-IO authors.
"""

import dataclasses
import functools

from fmmax import basis, fmm  # type: ignore[import-untyped]
from jax import numpy as jnp
from totypes import types

from invrs_gym import utils
from invrs_gym.challenges import base
from invrs_gym.challenges.sorter import common
from invrs_gym.loss import transmission_loss

POLARIZATION_RATIO_MIN = "polarization_ratio_min"
POLARIZATION_RATIO_MEAN = "polarization_ratio_mean"
EFFICIENCY_MIN = "efficiency_min"
EFFICIENCY_MEAN = "efficiency_mean"

TRANSMISSION_EXPONENT = 1.0
SCALAR_EXPONENT = 2.0


density_initializer = functools.partial(
    utils.initializers.noisy_density_initializer,
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
            x-polarized plane wave into its designated pixel. A polarization sorter
            with ideal absorpbing polarizers achieves a maximum efficiency of 0.25.
            The maximum for a non-lossy metasurface is 0.5.
        polarization_ratio_target: The target ratio of power coupled for e.g. an
            x-polarized plane wave into the "x-polarized pixel" and the power for
            the x-polarized plane wave into the "y-polarized pixel".
    """

    component: common.SorterComponent
    efficiency_target: float
    polarization_ratio_target: float

    def loss(self, response: common.SorterResponse) -> jnp.ndarray:
        """Compute a scalar loss from the component `response`."""
        # Lower bounds for transmission into each quadrant and total reflection.
        lb0 = self.efficiency_target  # Target quadrant.
        lb1 = self.efficiency_target / 2  # Partial target quadrant.
        rt_lower_bound = jnp.asarray(
            [
                # R,  Q1,  Q2,  Q3,  Q4
                [0.0, lb0, lb1, lb1, 0.0],  # x
                [0.0, lb1, lb0, 0.0, lb1],  # (x + y) / sqrt(2)
                [0.0, lb1, 0.0, lb0, lb1],  # (x - y) / sqrt(2)
                [0.0, 0.0, lb1, lb1, lb0],  # y
            ]
        )

        # Upper bounds for transmission into each quadrant.
        ub0 = 0.5  # Target quadrant.
        ub1 = 0.25  # Partial target quadrant.
        ub2 = (
            self.efficiency_target / self.polarization_ratio_target
        )  # Off-target quadrant.
        ubr = 1 - (lb0 + 2 * lb1)  # Upper bound for reflection.
        rt_upper_bound = jnp.asarray(
            [
                # R,  Q1,  Q2,  Q3,  Q4
                [ubr, ub0, ub1, ub1, ub2],  # x
                [ubr, ub1, ub0, ub2, ub1],  # (x + y) / sqrt(2)
                [ubr, ub1, ub2, ub0, ub1],  # (x - y) / sqrt(2)
                [ubr, ub2, ub1, ub1, ub0],  # y
            ]
        )
        rt = jnp.concatenate(
            [response.reflection[..., jnp.newaxis], response.transmission], axis=-1
        )
        assert rt.shape == rt_upper_bound.shape
        sorter_loss = transmission_loss.orthotope_smooth_transmission_loss(
            transmission=rt,
            window_lower_bound=rt_lower_bound,
            window_upper_bound=rt_upper_bound,
            transmission_exponent=jnp.asarray(TRANSMISSION_EXPONENT),
            scalar_exponent=jnp.asarray(SCALAR_EXPONENT),
            axis=(-2, -1),
        )
        return sorter_loss

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
        metrics = super().metrics(response, params, aux)
        on_target_transmission = response.transmission[
            ..., tuple(range(4)), tuple(range(4))
        ]
        efficiency = on_target_transmission

        off_target_transmission = response.transmission[
            ..., tuple(range(4))[::-1], tuple(range(4))
        ]
        polarization_ratio = on_target_transmission / off_target_transmission
        metrics.update(
            {
                EFFICIENCY_MEAN: jnp.mean(efficiency),
                EFFICIENCY_MIN: jnp.amin(efficiency),
                POLARIZATION_RATIO_MEAN: jnp.mean(polarization_ratio),
                POLARIZATION_RATIO_MIN: jnp.amin(polarization_ratio),
            }
        )
        return metrics


POLARIZATION_SORTER_SPEC = common.SorterSpec(
    permittivity_ambient=(1.0 + 0.0j) ** 2,
    permittivity_cap=(1.5 + 0.00001j) ** 2,
    permittivity_metasurface_solid=(4.0 + 0.00001j) ** 2,
    permittivity_metasurface_void=(1.5 + 0.00001j) ** 2,
    permittivity_spacer=(1.5 + 0.00001j) ** 2,
    permittivity_substrate=(4.0730 + 0.028038j) ** 2,
    thickness_cap=types.BoundedArray(0.05, lower_bound=0.00, upper_bound=0.5),
    thickness_metasurface=(
        # Default is a single metasurface layer. To model multiple metasurfaces,
        # simply provide multiple thicknesses.
        types.BoundedArray(0.1, lower_bound=0.025, upper_bound=0.3),
        types.BoundedArray(0.05, lower_bound=0.025, upper_bound=0.3),
    ),
    thickness_spacer=(
        # The first spacer is between the metasurfaces. The final spacer is
        # between the final spacer and the substrate.
        types.BoundedArray(0.1, lower_bound=0.025, upper_bound=0.3),
        types.BoundedArray(1.0, lower_bound=0.8, upper_bound=1.2),
    ),
    pitch=2.0,
    offset_monitor_substrate=0.1,
)

POLARIZATION_SORTER_SIM_PARAMS = common.SorterSimParams(
    grid_spacing=0.01,
    wavelength=0.55,
    polar_angle=0.0,
    azimuthal_angle=0.0,
    formulation=fmm.Formulation.JONES_DIRECT_FOURIER,
    approximate_num_terms=1600,
    truncation=basis.Truncation.CIRCULAR,
)

# Minimum width and spacing are 80 nm for the default dimensions.
MINIMUM_WIDTH = 8
MINIMUM_SPACING = 8

# Target metrics for the sorter component.
EFFICIENCY_TARGET = 0.3
POLARIZATION_RATIO_TARGET = 10


def polarization_sorter(
    minimum_width: int = MINIMUM_WIDTH,
    minimum_spacing: int = MINIMUM_SPACING,
    thickness_initializer: common.ThicknessInitializer = (
        utils.initializers.identity_initializer
    ),
    density_initializer: base.DensityInitializer = density_initializer,
    efficiency_target: float = EFFICIENCY_TARGET,
    polarization_ratio_target: float = POLARIZATION_RATIO_TARGET,
    spec: common.SorterSpec = POLARIZATION_SORTER_SPEC,
    sim_params: common.SorterSimParams = POLARIZATION_SORTER_SIM_PARAMS,
) -> PolarizationSorterChallenge:
    """Polarization sorter challenge.

    Args:
        minimum_width: The minimum width target for the challenge, in pixels.  The
            default value of 8 corresponds to a physical size of approximately 80 nm.
        minimum_spacing: The minimum spacing target for the challenge, in pixels.
        thickness_initializer: Callable which returns the initial thickness, given a
            key and seed thickness.
        density_initializer: Callable which returns the initial density, given a
            key and seed density.
        efficiency_target: The target efficiency for the coupling of e.g. an
            x-polarized plane wave into its designated pixel. A polarization sorter
            with ideal absorpbing polarizers achieves a maximum efficiency of 0.25.
            The maximum for a non-lossy metasurface is 0.5.
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
