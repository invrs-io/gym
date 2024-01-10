"""Defines the diffractive beamsplitter challenge.

Copyright (c) 2023 The INVRS-IO authors.
"""

import dataclasses
import functools
import itertools
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
from fmmax import basis, fmm  # type: ignore[import-untyped]
from jax import tree_util
from totypes import types

from invrs_gym import utils
from invrs_gym.challenges import base
from invrs_gym.challenges.diffract import common
from invrs_gym.loss import transmission_loss

Params = Dict[str, types.BoundedArray | types.Density2DArray]
ThicknessInitializer = Callable[[jax.Array, types.BoundedArray], types.BoundedArray]


DENSITY = "density"
THICKNESS = "thickness"

TOTAL_EFFICIENCY = "total_efficiency"
AVERAGE_EFFICIENCY = "average_efficiency"
MIN_EFFICIENCY = "min_efficiency"
ZEROTH_ORDER_EFFICIENCY = "zeroth_order_efficiency"
ZEROTH_ORDER_ERROR = "zeroth_order_error"
UNIFORMITY_ERROR = "uniformity_error"
UNIFORMITY_ERROR_WITHOUT_ZEROTH_ORDER = "uniformity_error_without_zeroth_order"

TRANSMISSION_EXPONENT = 1.0
SCALAR_EXPONENT = 2.0

density_initializer = functools.partial(
    utils.initializers.noisy_density_initializer,
    relative_mean=0.5,
    relative_noise_amplitude=0.1,
)


class DiffractiveSplitterComponent(base.Component):
    """Defines a diffractive splitter component."""

    def __init__(
        self,
        spec: common.GratingSpec,
        sim_params: common.GratingSimParams,
        thickness_initializer: ThicknessInitializer,
        density_initializer: base.DensityInitializer,
        **seed_density_kwargs: Any,
    ) -> None:
        """Initializes the grating component.

        Args:
            spec: Defines the physical specification of the splitter.
            sim_params: Defines simulation parameters for the splitter.
            thickness_initializer: Callable which returns the initial thickness for
                the grating layer from a random key and a bounded array with value
                equal the thickness from `spec`.
            density_initializer: Callable which generates the initial density from
                a random key and the seed density.
            **seed_density_kwargs: Keyword arguments which set the attributes of
                the seed density used to generate the inital parameters.
        """

        self.spec = spec
        self.sim_params = sim_params
        self.thickness_initializer = thickness_initializer
        self.density_initializer = density_initializer

        self.seed_density = common.seed_density(
            self.sim_params.grid_shape, **seed_density_kwargs
        )

        self.expansion = basis.generate_expansion(
            primitive_lattice_vectors=basis.LatticeVectors(
                u=self.spec.period_x * basis.X,
                v=self.spec.period_y * basis.Y,
            ),
            approximate_num_terms=self.sim_params.approximate_num_terms,
            truncation=self.sim_params.truncation,
        )

    def init(self, key: jax.Array) -> Params:
        """Return the initial parameters for the diffractive splitter component."""
        key_thickness, key_density = jax.random.split(key)
        params = {
            THICKNESS: self.thickness_initializer(
                key_thickness, self.spec.thickness_grating  # type: ignore[arg-type]
            ),
            DENSITY: self.density_initializer(key_density, self.seed_density),
        }
        # Ensure that there are no weak types in the initial parameters.
        return tree_util.tree_map(
            lambda x: jnp.asarray(x, jnp.asarray(x).dtype), params
        )

    def response(
        self,
        params: Params,
        *,
        wavelength: Optional[Union[float, jnp.ndarray]] = None,
        expansion: Optional[basis.Expansion] = None,
    ) -> Tuple[common.GratingResponse, base.AuxDict]:
        """Computes the response of the diffractive splitter.

        Args:
            params: The parameters defining the diffractive splitter, matching those
                returned by the `init` method.
            wavelength: Optional wavelength to override the default in `sim_params`.
            expansion: Optional expansion to override the default `expansion`.

        Returns:
            The `(response, aux)` tuple.
        """
        if expansion is None:
            expansion = self.expansion
        if wavelength is None:
            wavelength = self.sim_params.wavelength
        spec = dataclasses.replace(
            self.spec,
            thickness_grating=jnp.asarray(params[THICKNESS].array),
        )
        transmission_efficiency, reflection_efficiency = common.grating_efficiency(
            density=params[DENSITY],  # type: ignore[arg-type]
            spec=spec,
            wavelength=jnp.asarray(wavelength),
            polarization=self.sim_params.polarization,
            expansion=expansion,
            formulation=self.sim_params.formulation,
        )
        response = common.GratingResponse(
            wavelength=jnp.asarray(wavelength),
            transmission_efficiency=transmission_efficiency,
            reflection_efficiency=reflection_efficiency,
            expansion=expansion,
        )
        return response, {}


@dataclasses.dataclass
class DiffractiveSplitterChallenge(base.Challenge):
    """Defines the diffractive beamsplitter challenge.

    The objective of the challenge is to evenly split light into an array of
    transmitted diffraction orders. The challenge is based on the LightTrans
    publication, "Design and rigorous analysis of a non-paraxial diffractive
    beamsplitter", retrieved from
    https://www.lighttrans.com/fileadmin/shared/UseCases/Application_UC_Rigorous%20Analysis%20of%20Non-paraxial%20Diffractive%20Beam%20Splitter.pdf

    The challenge is considered solved when the efficiency of each order is greater
    than `normalized_efficiency_lower_bound / num_splits` and less than
    `normalized_efficiency_upper_bound / num_splits`.

    Attributes:
        component: The component to be designed.
        splitting: Defines the target splitting for the beamsplitter.
        normalized_efficiency_lower_bound: The lower bound for normalized efficiency.
        normalized_efficiency_upper_bound: The upper bound for normalized efficiency.
    """

    component: DiffractiveSplitterComponent
    splitting: Tuple[int, int]
    normalized_efficiency_lower_bound: float
    normalized_efficiency_upper_bound: float

    def loss(self, response: common.GratingResponse) -> jnp.ndarray:
        """Compute a scalar loss from the component `response`."""
        assert response.transmission_efficiency.shape[-1] == 1

        efficiency = extract_orders_for_splitting(
            response.transmission_efficiency,
            expansion=response.expansion,
            splitting=self.splitting,
        )
        num_splits = self.splitting[0] * self.splitting[1]
        lower_bound = self.normalized_efficiency_lower_bound / num_splits
        upper_bound = self.normalized_efficiency_upper_bound / num_splits
        assert efficiency.shape[-3:] == self.splitting + (1,)

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

    def distance_to_target(self, response: common.GratingResponse) -> jnp.ndarray:
        """Compute distance from the component `response` to the challenge target."""
        efficiency = extract_orders_for_splitting(
            response.transmission_efficiency,
            expansion=response.expansion,
            splitting=self.splitting,
        )
        assert efficiency.shape[-3:] == self.splitting + (1,)
        num_splits = self.splitting[0] * self.splitting[1]
        lower_bound = self.normalized_efficiency_lower_bound / num_splits
        upper_bound = self.normalized_efficiency_upper_bound / num_splits
        lower_bound_error = jnp.maximum(0, lower_bound - efficiency)
        upper_bound_error = jnp.maximum(0, efficiency - upper_bound)
        error = jnp.maximum(upper_bound_error, lower_bound_error)
        return jnp.linalg.norm(error)

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
        )
        assert transmission.shape[-3:] == self.splitting + (1,)
        # Metrics are averaged over the wavelength axis, if one exists.
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
) -> jnp.ndarray:
    """Extract the values from `array` for the specified splitting."""

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
    # Small imaginary part stabilizes the FMM calculation.
    permittivity_grating=(1.46 + 0.00001j) ** 2,
    permittivity_encapsulation=(1.0 + 0.00001j) ** 2,
    permittivity_substrate=(1.0 + 0.0j) ** 2,
    thickness_grating=types.BoundedArray(array=0.692, lower_bound=0.5, upper_bound=1.5),
    period_x=7.2,
    period_y=7.2,
)

DIFFRACTIVE_SPLITTER_SIM_PARAMS = common.GratingSimParams(
    grid_shape=(180, 180),
    wavelength=0.6328,
    polarization=common.TM,
    formulation=fmm.Formulation.JONES_DIRECT_FOURIER,
    approximate_num_terms=800,
    truncation=basis.Truncation.CIRCULAR,
)

# Objective is to split into a 7 x 7 array of beams. The minimum efficiency of any
# beam should be `0.6 / (7 * 7)`, while the maximum should be `0.8 / (7 * 7)`.
SPLITTING = (7, 7)
NORMALIZED_EFFICIENCY_LOWER_BOUND = 0.6
NORMALIZED_EFFICIENCY_UPPER_BOUND = 0.8


def diffractive_splitter(
    minimum_width: int = 10,
    minimum_spacing: int = 10,
    thickness_initializer: ThicknessInitializer = (
        utils.initializers.identity_initializer
    ),
    density_initializer: base.DensityInitializer = density_initializer,
    splitting: Tuple[int, int] = SPLITTING,
    normalized_efficiency_lower_bound: float = NORMALIZED_EFFICIENCY_LOWER_BOUND,
    normalized_efficiency_upper_bound: float = NORMALIZED_EFFICIENCY_UPPER_BOUND,
    spec: common.GratingSpec = DIFFRACTIVE_SPLITTER_SPEC,
    sim_params: common.GratingSimParams = DIFFRACTIVE_SPLITTER_SIM_PARAMS,
    symmetries: Sequence[str] = (),
) -> DiffractiveSplitterChallenge:
    """Non-paraxial diffractive beamsplitter challenge.

    The diffractive splitter is based on "Design and rigorous analysis of a
    non-paraxial diffractive beamsplitter", an example of the LightTrans software
    (https://www.lighttrans.com/use-cases/application/design-and-rigorous-analysis-of-non-paraxial-diffractive-beam-splitter.html).

    It involves splitting a normally-incident TM-polarized plane wave into an
    array of 7x7 beams with maximal efficiency and uniformity. The challenge is
    considered solved when the efficiency of each order is greater than
    `normalized_efficiency_lower_bound / num_splits` and less than
    `normalized_efficiency_upper_bound / num_splits`.

    Args:
        minimum_width: The minimum width target for the challenge, in pixels. The
            physical minimum width is approximately 180 nm.
        minimum_spacing: The minimum spacing target for the challenge, in pixels.
        thickness_initializer: Callable which returns the initial thickness, given a
            key and seed thickness.
        density_initializer: Callable which returns the initial density, given a
            key and seed density.
        splitting: Defines shape of the beam array to be created by the splitter.
        normalized_efficiency_lower_bound: The lower bound for normalized efficiency.
        normalized_efficiency_upper_bound: The upper bound for normalized efficiency.
        spec: Defines the physical specification of the metagrating.
        sim_params: Defines the simulation settings of the metagrating.
        symmetries: Defines the symmetries of the metagrating.

    Returns:
        The `MetagratingChallenge`.
    """
    return DiffractiveSplitterChallenge(
        component=DiffractiveSplitterComponent(
            spec=spec,
            sim_params=sim_params,
            thickness_initializer=thickness_initializer,
            density_initializer=density_initializer,
            minimum_width=minimum_width,
            minimum_spacing=minimum_spacing,
            symmetries=symmetries,
        ),
        splitting=splitting,
        normalized_efficiency_lower_bound=normalized_efficiency_lower_bound,
        normalized_efficiency_upper_bound=normalized_efficiency_upper_bound,
    )
