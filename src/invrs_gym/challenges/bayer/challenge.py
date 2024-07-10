"""Defines the bayer color sorter challenge.

Copyright (c) 2023 The INVRS-IO authors.
"""

import dataclasses
import functools
from typing import Sequence, Tuple

from fmmax import basis, fmm  # type: ignore[import-untyped]
from invrs_gym import utils
from invrs_gym.challenges import base
from jax import nn
from jax import numpy as jnp
from totypes import symmetry, types

from invrs_gym.challenges.bayer import component as bayer_component
from invrs_gym.utils import materials

EFFICIENCY_BLUE_MEAN = "efficiency_blue_mean"
EFFICIENCY_GREEN_MEAN = "efficiency_green_mean"
EFFICIENCY_RED_MEAN = "efficiency_red_mean"
CROSSTALK_BLUE_MEAN = "crosstalk_blue_mean"
CROSSTALK_GREEN_MEAN = "crosstalk_green_mean"
CROSSTALK_RED_MEAN = "crosstalk_red_mean"

# Cutoff between UV/blue wavelengths, blue/green wavelengths, etc. in microns.
UV_BLUE_CUTOFF = 0.4
BLUE_GREEN_CUTOFF = 0.5
GREEN_RED_CUTOFF = 0.6
RED_IR_CUTOFF = 0.7


density_initializer = functools.partial(
    utils.initializers.noisy_density_initializer,
    relative_mean=0.5,
    relative_noise_amplitude=0.1,
)


@dataclasses.dataclass
class BayerChallenge(base.Challenge):
    """Defines the bayer color sorter challenge."""

    component: bayer_component.BayerComponent

    def loss(
        self,
        response: bayer_component.BayerResponse,
        efficiency_target: float = 0.55,
    ) -> jnp.ndarray:
        """Compute a scalar loss from the component `response`.

        Args:
            response: The response obtained from simulating the component.
            efficiency_target: Value used in a nonlinearity within the loss function.
                When the efficiency value is far from the target, the dependence of
                loss on efficiency is larger.

        Returns:
            The scalar loss value.
        """
        # Average over the two polarizations.
        transmission = jnp.mean(response.transmission, axis=-2)
        assert transmission.shape == (response.wavelength.size, 4)

        is_blue, is_green, is_red = _detect_color(response.wavelength)
        t_blue, t_green, t_red = _transmission_per_pixel(transmission)

        efficiency_blue = jnp.sum(t_blue * is_blue) / jnp.sum(is_blue)
        efficiency_green = jnp.sum(t_green * is_green) / jnp.sum(is_green)
        efficiency_red = jnp.sum(t_red * is_red) / jnp.sum(is_red)

        denom = 1 - efficiency_target
        loss_blue = nn.softplus((efficiency_target - efficiency_blue) / denom)
        loss_green = nn.softplus((efficiency_target - efficiency_green) / denom)
        loss_red = nn.softplus((efficiency_target - efficiency_red) / denom)
        return loss_blue + loss_green + loss_red

    def eval_metric(self, response: bayer_component.BayerResponse) -> jnp.ndarray:
        """Compute eval metric from the component `response`.

        The eval metric for the bayer sorter is the minimum of the red, green, and
        blue efficiencies, where the efficiency fo a given color is the average
        efficiency with which wavelengths associated with the color are coupled into
        the appropriate subpixel(s).

        Args:
            response: The component response.

        Returns:
            The scalar eval metric.
        """
        # Average over the two polarizations.
        transmission = jnp.mean(response.transmission, axis=-2)
        assert transmission.shape == (response.wavelength.size, 4)

        is_blue, is_green, is_red = _detect_color(response.wavelength)
        t_blue, t_green, t_red = _transmission_per_pixel(transmission)
        efficiency_blue = jnp.sum(t_blue * is_blue) / jnp.sum(is_blue)
        efficiency_green = jnp.sum(t_green * is_green) / jnp.sum(is_green)
        efficiency_red = jnp.sum(t_red * is_red) / jnp.sum(is_red)
        return jnp.amin(
            jnp.asarray([efficiency_blue, efficiency_green, efficiency_red])
        )

    def metrics(
        self,
        response: bayer_component.BayerResponse,
        params: types.Density2DArray,
        aux: base.AuxDict,
    ) -> base.AuxDict:
        """Compute challenge metrics."""
        metrics = super().metrics(response, params, aux)

        # Average over the two polarizations.
        transmission = jnp.mean(response.transmission, axis=-2)
        assert transmission.shape == (response.wavelength.size, 4)

        is_blue, is_green, is_red = _detect_color(response.wavelength)
        t_blue, t_green, t_red = _transmission_per_pixel(transmission)
        efficiency_blue = jnp.sum(t_blue * is_blue) / jnp.sum(is_blue)
        efficiency_green = jnp.sum(t_green * is_green) / jnp.sum(is_green)
        efficiency_red = jnp.sum(t_red * is_red) / jnp.sum(is_red)

        # Crosstalk for a given color is total unwanted power in that color's pixel.
        crosstalk_blue = jnp.sum(t_blue * ~is_blue) / jnp.sum(~is_blue)
        crosstalk_green = jnp.sum(t_green * ~is_green) / jnp.sum(~is_green)
        crosstalk_red = jnp.sum(t_red * ~is_red) / jnp.sum(~is_red)

        transmitted_power = aux[bayer_component.TRANSMITTED_POWER]
        assert response.reflection.shape == transmitted_power.shape

        transmitted_power = jnp.sum(response.transmission, axis=-1)
        power = transmitted_power + response.reflection
        assert power.shape == response.wavelength.shape + (2,)

        metrics.update(
            {
                EFFICIENCY_BLUE_MEAN: efficiency_blue,
                EFFICIENCY_GREEN_MEAN: efficiency_green,
                EFFICIENCY_RED_MEAN: efficiency_red,
                CROSSTALK_BLUE_MEAN: crosstalk_blue,
                CROSSTALK_GREEN_MEAN: crosstalk_green,
                CROSSTALK_RED_MEAN: crosstalk_red,
            }
        )
        return metrics


def _detect_color(
    wavelength: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Detects color (blue, green, red) from `wavelength`."""
    is_blue = (wavelength >= UV_BLUE_CUTOFF) & (wavelength < BLUE_GREEN_CUTOFF)
    is_green = (wavelength >= BLUE_GREEN_CUTOFF) & (wavelength < GREEN_RED_CUTOFF)
    is_red = (wavelength >= GREEN_RED_CUTOFF) & (wavelength < RED_IR_CUTOFF)
    return is_blue, is_green, is_red


def _transmission_per_pixel(
    transmission: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Computes transmission into the blue, green, and red pixels."""
    assert transmission.shape[-1] == 4
    blue_transmission = transmission[..., 0]
    green_transmission = transmission[..., 1] + transmission[..., 2]
    red_transmission = transmission[..., 3]
    return blue_transmission, green_transmission, red_transmission


SYMMETRIES = (symmetry.REFLECTION_NW_SE,)

# Minimum width and spacing are 80 nm for the default dimensions.
MINIMUM_WIDTH = 8
MINIMUM_SPACING = 8


BAYER_SPEC = bayer_component.BayerSpec(
    material_ambient=materials.VACUUM,
    material_metasurface_solid=materials.SI3N4,
    material_metasurface_void=materials.VACUUM,
    material_substrate=materials.SIO2,
    thickness_ambient=1.0,
    thickness_metasurface=types.BoundedArray(0.6, lower_bound=0.4, upper_bound=0.8),
    thickness_substrate=3.0,
    pixel_size=1.0,
    grid_spacing=0.01,
    offset_monitor_substrate=types.BoundedArray(2.4, lower_bound=2.0, upper_bound=2.5),
)

BAYER_SIM_PARAMS = bayer_component.BayerSimParams(
    wavelength=jnp.array([0.45, 0.55, 0.65]),
    polar_angle=0.0,
    azimuthal_angle=0.0,
    formulation=fmm.Formulation.JONES_DIRECT_FOURIER,
    approximate_num_terms=600,
    truncation=basis.Truncation.CIRCULAR,
)


def bayer_sorter(
    minimum_width: int = MINIMUM_WIDTH,
    minimum_spacing: int = MINIMUM_SPACING,
    thickness_initializer: bayer_component.ThicknessInitializer = (
        utils.initializers.identity_initializer
    ),
    density_initializer: base.DensityInitializer = density_initializer,
    spec: bayer_component.BayerSpec = BAYER_SPEC,
    sim_params: bayer_component.BayerSimParams = BAYER_SIM_PARAMS,
    symmetries: Sequence[str] = SYMMETRIES,
) -> BayerChallenge:
    """The bayer color sorter challenge.

    The bayer sorter challenge entails the design of a color-sorting metasurface that
    replaces the color filter array in a conventional image sensor. The thickness of
    the metasurface, and the distance between metasurface and focal plane are further
    degrees of freedom. The bayer challenge is based on "Pixel-level Bayer-type colour
    router based on metasurfaces" by Zou et al.

    https://www.nature.com/articles/s41467-022-31019-7

    Args:
        minimum_width: The minimum width target for the challenge, in pixels.  The
            default value of 8 corresponds to a physical size of approximately 80 nm.
        minimum_spacing: The minimum spacing target for the challenge, in pixels.
        thickness_initializer: Callable which returns the initial thickness, given a
            key and seed thickness.
        density_initializer: Callble which returns the initial density, given a
            key and seed density.
        spec: Defines the physical specification of the bayer sorter.
        sim_params: Defines the simulation settings of the bayer sorter.
        symmetries: Defines the symmetries of the bayer sorter.

    Returns:
        The `BayerChallenge`.
    """
    return BayerChallenge(
        component=bayer_component.BayerComponent(
            spec=spec,
            sim_params=sim_params,
            thickness_initializer=thickness_initializer,
            density_initializer=density_initializer,
            minimum_width=minimum_width,
            minimum_spacing=minimum_spacing,
            symmetries=symmetries,
        ),
    )
