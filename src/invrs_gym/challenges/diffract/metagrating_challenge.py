"""Defines the metagrating challenge.

Copyright (c) 2023 The INVRS-IO authors.
"""

import dataclasses
import functools
from typing import Sequence, Tuple

import jax.numpy as jnp
from fmmax import basis, fmm  # type: ignore[import-untyped]
from jax import nn
from totypes import symmetry, types

from invrs_gym.challenges import base
from invrs_gym.challenges.diffract import common
from invrs_gym.utils import initializers

AVERAGE_EFFICIENCY = "average_efficiency"
MIN_EFFICIENCY = "min_efficiency"

POLARIZATION = "TM"

density_initializer = functools.partial(
    initializers.noisy_density_initializer,
    relative_mean=0.5,
    relative_noise_amplitude=0.1,
)


@dataclasses.dataclass
class MetagratingChallenge(base.Challenge):
    """Defines the metagrating challenge.

    The objective of the metagrating challenge is to design a density so that incident
    light is efficiently diffracted into the specified transmission order.

    Attributes:
        component: The component to be designed.
        transmission_order: The transmission diffraction order to be maximized.
    """

    component: common.SimpleGratingComponent
    transmission_order: Tuple[int, int]

    def loss(
        self,
        response: common.GratingResponse,
        transmission_target: float = 0.95,
    ) -> jnp.ndarray:
        """Compute a scalar loss from the component `response`.

        Args:
            response: The response obtained from simulating the component.
            transmission_target: Value used in a nonlinearity within the loss function.
                When the transmission value is far from the target, the dependence of
                loss on tranmsission is larger.

        Returns:
            The scalar loss value.
        """
        # Compute efficiency, a per-wavelength scalar.
        efficiency = _value_for_order(
            response.transmission_efficiency,
            expansion=response.expansion,
            order=self.transmission_order,
            polarization=POLARIZATION,
        )
        batch_shape = jnp.broadcast_shapes(
            response.wavelength.shape,
            response.polar_angle.shape,
            response.azimuthal_angle.shape,
        )
        assert efficiency.shape == batch_shape
        window_size = 1 - transmission_target
        scaled_error = (transmission_target - efficiency) / window_size
        return jnp.mean(nn.softplus(scaled_error) ** 2)

    def eval_metric(self, response: common.GratingResponse) -> jnp.ndarray:
        """Compute eval metric from the component `response`.

        The eval metric is the minimum efficiency of diffraction into the target order
        across all wavelengths.

        Args:
            response: The component response.

        Returns:
            The scalar eval metric.
        """
        efficiency = _value_for_order(
            response.transmission_efficiency,
            expansion=response.expansion,
            order=self.transmission_order,
            polarization=POLARIZATION,
        )
        return jnp.amin(efficiency)

    def metrics(
        self,
        response: common.GratingResponse,
        params: types.Density2DArray,
        aux: base.AuxDict,
    ) -> base.AuxDict:
        """Compute challenge metrics."""
        metrics = super().metrics(response, params, aux)
        efficiency = _value_for_order(
            response.transmission_efficiency,
            expansion=response.expansion,
            order=self.transmission_order,
            polarization=POLARIZATION,
        )
        metrics.update(
            {
                AVERAGE_EFFICIENCY: jnp.mean(efficiency),
                MIN_EFFICIENCY: jnp.amin(efficiency),
            }
        )
        return metrics


def _value_for_order(
    array: jnp.ndarray,
    expansion: basis.Expansion,
    order: Tuple[int, int],
    polarization: str,
) -> jnp.ndarray:
    """Extracts the value from `array` for the specified Fourier order."""
    assert polarization in ("TE", "TM")
    polarization_idx = 0 if polarization == "TE" else 1
    order_idx = common.index_for_order(order, expansion)
    return array[..., order_idx, polarization_idx]


# -----------------------------------------------------------------------------
# Metagrating with 1.371 x 0.525 um design region.
# -----------------------------------------------------------------------------


METAGRATING_SPEC = common.GratingSpec(
    permittivity_ambient=(1.0 + 0.0j) ** 2,
    permittivity_cap=(1.0 + 0.0j) ** 2,
    permittivity_grating=(3.45 + 0.00001j) ** 2,
    permittivity_encapsulation=(1.0 + 0.00001j) ** 2,
    permittivity_spacer=(1.45 + 0.0j) ** 2,
    permittivity_substrate=(1.45 + 0.0j) ** 2,
    thickness_cap=0.0,
    thickness_grating=0.325,
    thickness_spacer=0.0,
    period_x=float(1.050 / jnp.sin(jnp.deg2rad(50.0))),
    period_y=0.525,
    grid_spacing=0.0117,  # Yields a grid shape of `(118, 45)`.
)

METAGRATING_SIM_PARAMS = common.GratingSimParams(
    wavelength=1.050,
    polar_angle=0.0,
    azimuthal_angle=0.0,
    formulation=fmm.Formulation.JONES_DIRECT_FOURIER,
    approximate_num_terms=300,
    truncation=basis.Truncation.CIRCULAR,
)

SYMMETRIES = (symmetry.REFLECTION_E_W,)

# Objective is to diffract light into the +1 transmitted order.
TRANSMISSION_ORDER = (1, 0)


def metagrating(
    minimum_width: int = 5,
    minimum_spacing: int = 5,
    density_initializer: base.DensityInitializer = density_initializer,
    transmission_order: Tuple[int, int] = TRANSMISSION_ORDER,
    spec: common.GratingSpec = METAGRATING_SPEC,
    sim_params: common.GratingSimParams = METAGRATING_SIM_PARAMS,
    symmetries: Sequence[str] = SYMMETRIES,
) -> MetagratingChallenge:
    """Metagrating challenge with 1.371 x 0.525 um design region.

    The metagrating challenge is based on the metagrating example in "Validation and
    characterization of algorithms for photonics inverse design" by Chen et al.
    (in preparation), in which designs with feature size from 35 to 66 nm are shown.

    It involves maximizing diffraction of light transmitted from a silicon oxide
    substrate into the ambient using a patterned silicon metastructure. The excitation
    is TM-polarized plane wave with 1.05 micron wavelength.

    Args:
        minimum_width: The minimum width target for the challenge, in pixels. The
            default value of 5 corresponds to a physical size of approximately 60 nm.
        minimum_spacing: The minimum spacing target for the challenge, in pixels.
        density_initializer: Callable which returns the initial density, given a
            key and seed density.
        transmission_order: The diffraction order to be maximized.
        spec: Defines the physical specification of the metagrating.
        sim_params: Defines the simulation settings of the metagrating.
        symmetries: Defines the symmetries of the metagrating.

    Returns:
        The `MetagratingChallenge`.
    """
    return MetagratingChallenge(
        component=common.SimpleGratingComponent(
            spec=spec,
            sim_params=sim_params,
            density_initializer=density_initializer,
            minimum_width=minimum_width,
            minimum_spacing=minimum_spacing,
            symmetries=symmetries,
        ),
        transmission_order=transmission_order,
    )
