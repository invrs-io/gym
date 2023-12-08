"""Defines the metagrating challenge.

Copyright (c) 2023 The INVRS-IO authors.
"""

import dataclasses
import functools
from typing import Any, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
from fmmax import basis, fmm  # type: ignore[import-untyped]
from jax import nn, tree_util
from totypes import symmetry, types

from invrs_gym.challenges import base
from invrs_gym.challenges.diffract import common
from invrs_gym.utils import initializers

AVERAGE_EFFICIENCY = "average_efficiency"
MIN_EFFICIENCY = "min_efficiency"

density_initializer = functools.partial(
    initializers.noisy_density_initializer,
    relative_mean=0.5,
    relative_noise_amplitude=0.1,
)


class MetagratingComponent(base.Component):
    """Defines a metagrating component."""

    def __init__(
        self,
        spec: common.GratingSpec,
        sim_params: common.GratingSimParams,
        density_initializer: base.DensityInitializer,
        **seed_density_kwargs: Any,
    ) -> None:
        """Initializes the grating component.

        Args:
            spec: Defines the physical specification of the grating.
            sim_params: Defines simulation parameters for the grating.
            density_initializer: Callable which generates the initial density from
                a random key and the seed density.
            **seed_density_kwargs: Keyword arguments which set the attributes of
                the seed density used to generate the inital parameters.
        """

        self.spec = spec
        self.sim_params = sim_params
        self.seed_density = common.seed_density(
            self.sim_params.grid_shape, **seed_density_kwargs
        )
        self.density_initializer = density_initializer

        self.expansion = basis.generate_expansion(
            primitive_lattice_vectors=basis.LatticeVectors(
                u=self.spec.period_x * basis.X,
                v=self.spec.period_y * basis.Y,
            ),
            approximate_num_terms=self.sim_params.approximate_num_terms,
            truncation=self.sim_params.truncation,
        )

    def init(self, key: jax.Array) -> types.Density2DArray:
        """Return the initial parameters for the metagrating component."""
        params = self.density_initializer(key, self.seed_density)
        # Ensure that there are no weak types in the initial parameters.
        return tree_util.tree_map(
            lambda x: jnp.asarray(x, jnp.asarray(x).dtype), params
        )

    def response(
        self,
        params: types.Density2DArray,
        *,
        wavelength: Optional[Union[float, jnp.ndarray]] = None,
        expansion: Optional[basis.Expansion] = None,
    ) -> Tuple[common.GratingResponse, base.AuxDict]:
        """Computes the response of the metagrating.

        Args:
            params: The parameters defining the metagrating, matching those returned
                by the `init` method.
            wavelength: Optional wavelength to override the default in `sim_params`.
            expansion: Optional expansion to override the default `expansion`.

        Returns:
            The `(response, aux)` tuple.
        """
        if expansion is None:
            expansion = self.expansion
        if wavelength is None:
            wavelength = self.sim_params.wavelength
        transmission_efficiency, reflection_efficiency = common.grating_efficiency(
            density=params,
            spec=self.spec,
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
class MetagratingChallenge(base.Challenge):
    """Defines the metagrating challenge.

    The objective of the metagrating challenge is to design a density so that incident
    light is efficiently diffracted into the specified transmission order. The
    challenge is considered solved when the transmission is above the lower bound.

    Attributes:
        component: The component to be designed.
        transmission_order: The transmission diffraction order to be maximized.
        transmission_lower_bound: The lower bound for transmission into the specified
            order. When this value is exceeded, the challenge is considered solved.
    """

    component: MetagratingComponent
    transmission_order: Tuple[int, int]
    transmission_lower_bound: float

    def loss(self, response: common.GratingResponse) -> jnp.ndarray:
        """Compute a scalar loss from the component `response`."""
        efficiency = _value_for_order(
            response.transmission_efficiency,
            expansion=response.expansion,
            order=self.transmission_order,
        )
        window_size = 1 - self.transmission_lower_bound
        scaled_error = (self.transmission_lower_bound - efficiency) / window_size
        return jnp.linalg.norm(nn.softplus(scaled_error))

    def distance_to_target(self, response: common.GratingResponse) -> jnp.ndarray:
        """Compute distance from the component `response` to the challenge target."""
        efficiency = _value_for_order(
            response.transmission_efficiency,
            expansion=response.expansion,
            order=self.transmission_order,
        )
        elementwise_distance_to_window = jnp.maximum(
            0, self.transmission_lower_bound - efficiency
        )
        return jnp.linalg.norm(elementwise_distance_to_window)

    def metrics(
        self,
        response: common.GratingResponse,
        params: types.Density2DArray,
        aux: base.AuxDict,
    ) -> base.AuxDict:
        """Compute challenge metrics."""
        del params, aux
        efficiency = _value_for_order(
            response.transmission_efficiency,
            expansion=response.expansion,
            order=self.transmission_order,
        )
        return {
            AVERAGE_EFFICIENCY: jnp.mean(efficiency),
            MIN_EFFICIENCY: jnp.amin(efficiency),
        }


def _value_for_order(
    array: jnp.ndarray,
    expansion: basis.Expansion,
    order: Tuple[int, int],
) -> jnp.ndarray:
    """Extracts the value from `array` for the specified Fourier order."""
    order_idx = common.index_for_order(order, expansion)
    return array[..., order_idx, :]


# -----------------------------------------------------------------------------
# Metagrating with 1.371 x 0.525 um design region.
# -----------------------------------------------------------------------------


METAGRATING_SPEC = common.GratingSpec(
    permittivity_ambient=(1.0 + 0.0j) ** 2,
    permittivity_grating=(3.45 + 0.00001j) ** 2,
    permittivity_encapsulation=(1.0 + 0.00001j) ** 2,
    permittivity_substrate=(1.45 + 0.0j) ** 2,
    thickness_grating=0.325,
    period_x=float(1.050 / jnp.sin(jnp.deg2rad(50.0))),
    period_y=0.525,
)

METAGRATING_SIM_PARAMS = common.GratingSimParams(
    grid_shape=(118, 45),
    wavelength=1.050,
    polarization=common.TM,
    formulation=fmm.Formulation.JONES_DIRECT,
    approximate_num_terms=200,
    truncation=basis.Truncation.CIRCULAR,
)

SYMMETRIES = (symmetry.REFLECTION_E_W,)

# Objective is to diffract light into the +1 transmitted order, with efficiency better
# than 95 percent.
TRANSMISSION_ORDER = (1, 0)
TRANSMISSION_LOWER_BOUND = 0.95


def metagrating(
    minimum_width: int = 5,
    minimum_spacing: int = 5,
    density_initializer: base.DensityInitializer = density_initializer,
    transmission_order: Tuple[int, int] = TRANSMISSION_ORDER,
    transmission_lower_bound: float = TRANSMISSION_LOWER_BOUND,
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
            default value of 5 corresponds to a physical size of approximately 80 nm.
        minimum_spacing: The minimum spacing target for the challenge, in pixels.
        density_initializer: Callable which returns the initial density, given a
            key and seed density.
        transmission_order: The diffraction order to be maximized.
        transmission_lower_bound: The lower bound for transmission. When the lower
            bound is exceeded, the challenge is considered to be solved.
        spec: Defines the physical specification of the metagrating.
        sim_params: Defines the simulation settings of the metagrating.
        symmetries: Defines the symmetries of the metagrating.

    Returns:
        The `MetagratingChallenge`.
    """
    return MetagratingChallenge(
        component=MetagratingComponent(
            spec=spec,
            sim_params=sim_params,
            density_initializer=density_initializer,
            minimum_width=minimum_width,
            minimum_spacing=minimum_spacing,
            symmetries=symmetries,
        ),
        transmission_order=transmission_order,
        transmission_lower_bound=transmission_lower_bound,
    )
