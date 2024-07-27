"""Defines the diffractive beamsplitter challenge.

Copyright (c) 2023 The INVRS-IO authors.
"""

import dataclasses
import functools
import itertools
from typing import Dict, Sequence, Tuple

import jax.numpy as jnp
from fmmax import basis, fmm  # type: ignore[import-untyped]
from totypes import types

from invrs_gym import utils
from invrs_gym.challenges import base
from invrs_gym.challenges.diffract import common
from invrs_gym.loss import transmission_loss

Params = Dict[str, types.BoundedArray | types.Density2DArray]


DENSITY = "density"
THICKNESS = "thickness"

TOTAL_EFFICIENCY = "total_efficiency"
AVERAGE_EFFICIENCY = "average_efficiency"
MIN_EFFICIENCY = "min_efficiency"
ZEROTH_ORDER_EFFICIENCY = "zeroth_order_efficiency"
ZEROTH_ORDER_ERROR = "zeroth_order_error"
UNIFORMITY_ERROR = "uniformity_error"
UNIFORMITY_ERROR_WITHOUT_ZEROTH_ORDER = "uniformity_error_without_zeroth_order"

POLARIZATION = "TE"

TRANSMISSION_EXPONENT = 1.0
SCALAR_EXPONENT = 2.0

density_initializer = functools.partial(
    utils.initializers.noisy_density_initializer,
    relative_mean=0.5,
    relative_noise_amplitude=0.1,
)


@dataclasses.dataclass
class DiffractiveSplitterChallenge(base.Challenge):
    """Defines the diffractive beamsplitter challenge.

    The objective of the challenge is to evenly split light into an array of
    transmitted diffraction orders. The challenge is based on the LightTrans
    publication, "Design and rigorous analysis of a non-paraxial diffractive
    beamsplitter", retrieved from
    https://www.lighttrans.com/fileadmin/shared/UseCases/Application_UC_Rigorous%20Analysis%20of%20Non-paraxial%20Diffractive%20Beam%20Splitter.pdf

    Attributes:
        component: The component to be designed.
        splitting: Defines the target splitting for the beamsplitter.
    """

    component: common.GratingWithOptimizableThicknessComponent
    splitting: Tuple[int, int]

    def loss(
        self,
        response: common.GratingResponse,
        normalized_efficiency_lower_bound: float = 0.6,
        normalized_efficiency_upper_bound: float = 0.8,
    ) -> jnp.ndarray:
        """Compute a scalar loss from the component `response`.

        Args:
            response: The response obtained from simulating the component.
            normalized_efficiency_lower_bound: Value used in a nonlinearity within the
                loss function. Together with `normalized_efficiency_upper_bound`,
                defines a window within which the loss is less sensitive to efficiency,
                which promotes convergence to values within the window.
            normalized_efficiency_upper_bound: Defines the upper bound of the window.

        Returns:
            The scalar loss.
        """
        efficiency = extract_orders_for_splitting(
            response.transmission_efficiency,
            expansion=response.expansion,
            splitting=self.splitting,
            polarization=POLARIZATION,
        )
        batch_shape = jnp.broadcast_shapes(
            response.wavelength.shape,
            response.polar_angle.shape,
            response.azimuthal_angle.shape,
        )
        assert efficiency.shape == batch_shape + self.splitting + (1,)
        num_splits = self.splitting[0] * self.splitting[1]
        lower_bound = normalized_efficiency_lower_bound / num_splits
        upper_bound = normalized_efficiency_upper_bound / num_splits

        # Compute per-wavelength orthotope loss.
        loss = transmission_loss.orthotope_smooth_transmission_loss(
            transmission=efficiency,
            window_lower_bound=jnp.full(efficiency.shape, lower_bound),
            window_upper_bound=jnp.full(efficiency.shape, upper_bound),
            transmission_exponent=jnp.asarray(TRANSMISSION_EXPONENT),
            scalar_exponent=jnp.asarray(SCALAR_EXPONENT),
            axis=(-3, -2, -1),
        )
        return jnp.mean(loss)  # Mean reduction across wavelengths, if they exist.

    def eval_metric(self, response: common.GratingResponse) -> jnp.ndarray:
        """Compute eval metric from the component `response`.

        The eval metric rewards high total efficiency and minimum nonuniformity.
        It is computed by,

            eval_metric = total_efficiency * (1 - uniformity_error)

        In cases where multiple wavelengths or incident angles are considered, the
        eval metric is the minimum across all excitation conditions.

        Args:
            response: The component response.

        Returns:
            The scalar eval metric.
        """
        transmission = extract_orders_for_splitting(
            response.transmission_efficiency,
            expansion=response.expansion,
            splitting=self.splitting,
            polarization=POLARIZATION,
        )
        assert transmission.shape[-3:] == self.splitting + (1,)

        # Total efficiency, i.e. sum of power into all output orders. A perfect
        # solution has a value of `1`, and the lowest possible value is `0`.
        total_efficiency = jnp.sum(transmission, axis=(-3, -2, -1))

        # Uniformity error. The perfect solution has a uniformity error of `0`,
        # and the highest possible value is `1`.
        uniformity_error = (
            jnp.amax(transmission, axis=(-3, -2, -1))
            - jnp.amin(transmission, axis=(-3, -2, -1))
        ) / (
            jnp.amax(transmission, axis=(-3, -2, -1))
            + jnp.amin(transmission, axis=(-3, -2, -1))
        )

        return jnp.amin(total_efficiency * (1 - uniformity_error))

    def metrics(
        self,
        response: common.GratingResponse,
        params: types.Density2DArray,
        aux: base.AuxDict,
    ) -> base.AuxDict:
        """Compute challenge metrics.

        Args:
            response: The response of the diffractive splitter component.
            params: The parameters where the response was evaluated.
            aux: The auxilliary quantities returned by the component response method.

        Returns:
            The metrics dictionary, with the following quantities:
                - total efficiency
                - average efficiency
                - zeroth order efficiency
                - zeroth order error
                - uniformity error
                - uniformity error without zeroth order
            For details, see slide 14 of the LightTrans publication.
        """
        metrics = super().metrics(response, params, aux)
        transmission = extract_orders_for_splitting(
            response.transmission_efficiency,
            expansion=response.expansion,
            splitting=self.splitting,
            polarization=POLARIZATION,
        )
        # Metrics are averaged over the batch axes, if they exist.
        total_efficiency = jnp.mean(jnp.sum(transmission, axis=(-3, -2, -1)))
        average_efficiency = jnp.mean(transmission)
        min_efficiency = jnp.amin(transmission)

        i0 = self.splitting[0] // 2
        j0 = self.splitting[1] // 2
        zeroth_efficiency = jnp.mean(transmission[..., i0, j0, :])
        zeroth_error = (zeroth_efficiency - average_efficiency) / average_efficiency

        uniformity_error = (jnp.amax(transmission) - jnp.amin(transmission)) / (
            jnp.amax(transmission) + jnp.amin(transmission)
        )

        mask = jnp.ones(transmission.shape, dtype=bool)
        mask = mask.at[..., i0, j0, :].set(False)
        masked_max_transmission = jnp.amax(jnp.where(mask, transmission, 0.0))
        masked_min_transmission = jnp.amin(jnp.where(mask, transmission, 1.0))

        uniformity_error_without_zeroth = (
            masked_max_transmission - masked_min_transmission
        ) / (masked_max_transmission + masked_min_transmission)

        metrics.update(
            {
                TOTAL_EFFICIENCY: total_efficiency,
                AVERAGE_EFFICIENCY: average_efficiency,
                MIN_EFFICIENCY: min_efficiency,
                ZEROTH_ORDER_EFFICIENCY: zeroth_efficiency,
                ZEROTH_ORDER_ERROR: zeroth_error,
                UNIFORMITY_ERROR: uniformity_error,
                UNIFORMITY_ERROR_WITHOUT_ZEROTH_ORDER: uniformity_error_without_zeroth,
            }
        )
        return metrics


def indices_for_splitting(
    expansion: basis.Expansion,
    splitting: Tuple[int, int],
) -> Tuple[int, ...]:
    """Return the indices for the orders identified by the target `splitting`."""
    idxs = []
    orders_x = range(-splitting[0] // 2 + 1, splitting[0] // 2 + 1)
    orders_y = range(-splitting[1] // 2 + 1, splitting[1] // 2 + 1)
    for ox, oy in itertools.product(orders_x, orders_y):
        idxs.append(common.index_for_order((ox, oy), expansion))
    return tuple(idxs)


def extract_orders_for_splitting(
    array: jnp.ndarray,
    expansion: basis.Expansion,
    splitting: Tuple[int, int],
    polarization: str,
) -> jnp.ndarray:
    """Extract the values from `array` for the specified splitting."""
    assert polarization in ("TE", "TM")
    polarization_idx = 0 if polarization == "TE" else 1
    array = array[..., polarization_idx, jnp.newaxis]

    num_splits = splitting[0] * splitting[1]
    shape = array.shape[:-2] + splitting + (array.shape[-1],)
    flat_shape = array.shape[:-2] + (num_splits, array.shape[-1])

    idxs = indices_for_splitting(expansion, splitting)
    extracted = jnp.zeros(flat_shape, dtype=array.dtype)
    extracted = extracted.at[..., :, :].set(array[..., idxs, :])
    return extracted.reshape(shape)


# -----------------------------------------------------------------------------
# Diffractive splitter with 7.2 x 7.2 um design region.
# -----------------------------------------------------------------------------


DIFFRACTIVE_SPLITTER_SPEC = common.GratingSpec(
    permittivity_ambient=(1.46 + 0.0j) ** 2,
    permittivity_cap=(1.46 + 0.0j) ** 2,
    # Small imaginary part stabilizes the FMM calculation.
    permittivity_grating=(1.46 + 0.00001j) ** 2,
    permittivity_encapsulation=(1.0 + 0.00001j) ** 2,
    permittivity_spacer=(1.0 + 0.0j) ** 2,
    permittivity_substrate=(1.0 + 0.0j) ** 2,
    thickness_cap=types.BoundedArray(array=0.0, lower_bound=0.0, upper_bound=0.1),
    thickness_grating=types.BoundedArray(array=0.692, lower_bound=0.5, upper_bound=1.5),
    thickness_spacer=types.BoundedArray(array=0.0, lower_bound=0.0, upper_bound=0.1),
    period_x=7.2,
    period_y=7.2,
    grid_spacing=0.04,  # Yields a grid shape of `(180, 180)`.
)

DIFFRACTIVE_SPLITTER_SIM_PARAMS = common.GratingSimParams(
    wavelength=0.6328,
    polar_angle=0.0,
    azimuthal_angle=0.0,
    formulation=fmm.Formulation.JONES_DIRECT_FOURIER,
    approximate_num_terms=800,
    truncation=basis.Truncation.CIRCULAR,
)

# Objective is to split into a 7 x 7 array of beams.
SPLITTING = (7, 7)


def diffractive_splitter(
    minimum_width: int = 10,
    minimum_spacing: int = 10,
    thickness_initializer: base.ThicknessInitializer = (
        utils.initializers.identity_initializer
    ),
    density_initializer: base.DensityInitializer = density_initializer,
    splitting: Tuple[int, int] = SPLITTING,
    spec: common.GratingSpec = DIFFRACTIVE_SPLITTER_SPEC,
    sim_params: common.GratingSimParams = DIFFRACTIVE_SPLITTER_SIM_PARAMS,
    symmetries: Sequence[str] = (),
) -> DiffractiveSplitterChallenge:
    """Non-paraxial diffractive beamsplitter challenge.

    The diffractive splitter is based on "Design and rigorous analysis of a
    non-paraxial diffractive beamsplitter", an example of the LightTrans software
    (https://www.lighttrans.com/use-cases/application/design-and-rigorous-analysis-of-non-paraxial-diffractive-beam-splitter.html).

    It involves splitting a normally-incident TM-polarized plane wave into an
    array of 7x7 beams with maximal efficiency and uniformity.

    Args:
        minimum_width: The minimum width target for the challenge, in pixels. The
            default value of 10 corresponds to a physical size of approximately 400 nm.
        minimum_spacing: The minimum spacing target for the challenge, in pixels.
        thickness_initializer: Callable which returns the initial thickness, given a
            key and seed thickness.
        density_initializer: Callable which returns the initial density, given a
            key and seed density.
        splitting: Defines shape of the beam array to be created by the splitter.
        spec: Defines the physical specification of the diffractive splitter.
        sim_params: Defines the simulation settings of the diffractive splitter.
        symmetries: Defines the symmetries of the diffractive splitter.

    Returns:
        The `DiffractiveSplitterChallenge`.
    """
    return DiffractiveSplitterChallenge(
        component=common.GratingWithOptimizableThicknessComponent(
            spec=spec,
            sim_params=sim_params,
            thickness_initializer=thickness_initializer,
            density_initializer=density_initializer,
            minimum_width=minimum_width,
            minimum_spacing=minimum_spacing,
            symmetries=symmetries,
        ),
        splitting=splitting,
    )
