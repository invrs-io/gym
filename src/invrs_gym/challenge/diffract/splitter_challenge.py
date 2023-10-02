"""Defines the diffractive beamsplitter challenge."""

import dataclasses
import itertools
from typing import Any, Callable, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from fmmax import basis, fmm  # type: ignore[import]
from totypes import types  # type: ignore[import,attr-defined,unused-ignore]

from invrs_gym.challenge.diffract import common

PyTree = Any
AuxDict = Dict[str, Any]
ThicknessInitializer = Callable[[jax.Array, jnp.ndarray], jnp.ndarray]
DensityInitializer = Callable[[jax.Array, types.Density2DArray], types.Density2DArray]


DENSITY = "density"
THICKNESS = "thickness"

TOTAL_EFFICIENCY = "total_efficiency"
AVERAGE_EFFICIENCY = "average_efficiency"
ZEROTH_ORDER_EFFICIENCY = "zeroth_order_efficiency"
ZEROTH_ORDER_ERROR = "zeroth_order_error"
UNIFORMITY_ERROR = "uniformity_error"
UNIFORMITY_ERROR_WITHOUT_ZEROTH_ORDER = "uniformity_error_without_zeroth_order"


class DiffractiveSplitterComponent:
    """Defines a diffractive splitter component."""

    def __init__(
        self,
        spec: common.GratingSpec,
        sim_params: common.GratingSimParams,
        thickness_initializer: ThicknessInitializer,
        density_initializer: DensityInitializer,
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

        self.seed_thickness = types.BoundedArray(
            array=self.spec.thickness_grating,
            lower_bound=0.0,
            upper_bound=None,
        )
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

    def init(self, key: jax.Array) -> PyTree:
        """Return the initial parameters for the diffractive splitter component."""
        key_thickness, key_density = jax.random.split(key)
        return {
            THICKNESS: self.thickness_initializer(key_thickness, self.seed_thickness),
            DENSITY: self.density_initializer(key_density, self.seed_density),
        }

    def response(
        self,
        params: types.Density2DArray,
        wavelength: Optional[Union[float, jnp.ndarray]] = None,
        expansion: Optional[basis.Expansion] = None,
    ) -> Tuple[common.GratingResponse, AuxDict]:
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
        transmission_efficiency, reflection_efficiency = common.grating_efficiency(
            density_array=params[DENSITY].array,
            thickness=params[THICKNESS].array,
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
class DiffractiveSplitterChallenge:
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

    component: DiffractiveSplitterComponent
    splitting: Tuple[int, int]

    def loss(self, response: common.GratingResponse) -> jnp.ndarray:
        """Compute a scalar loss from the component `response`."""
        target = jnp.ones(self.splitting + (1,))
        target /= self.splitting[0] * self.splitting[1]

        transmission = extract_orders_for_splitting(
            response.transmission_efficiency,
            expansion=response.expansion,
            splitting=self.splitting,
        )
        assert transmission.shape[-3:] == target.shape[-3:]
        return jnp.linalg.norm(jnp.sqrt(target) - jnp.sqrt(transmission)) ** 2

    def metrics(
        self,
        response: common.GratingResponse,
        params: types.Density2DArray,
        aux: AuxDict,
    ) -> AuxDict:
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
        del params, aux
        transmission = extract_orders_for_splitting(
            response.transmission_efficiency,
            expansion=response.expansion,
            splitting=self.splitting,
        )
        assert transmission.shape[-3:] == self.splitting + (1,)
        # Metrics are averaged over the wavelength axis, if one exists.
        total_efficiency = jnp.mean(jnp.sum(transmission, axis=(-3, -2, -1)))
        average_efficiency = jnp.mean(transmission)

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

        return {
            TOTAL_EFFICIENCY: total_efficiency,
            AVERAGE_EFFICIENCY: average_efficiency,
            ZEROTH_ORDER_EFFICIENCY: zeroth_efficiency,
            ZEROTH_ORDER_ERROR: zeroth_error,
            UNIFORMITY_ERROR: uniformity_error,
            UNIFORMITY_ERROR_WITHOUT_ZEROTH_ORDER: uniformity_error_without_zeroth,
        }


def extract_orders_for_splitting(
    array: jnp.ndarray,
    expansion: basis.Expansion,
    splitting: Tuple[int, int],
) -> jnp.ndarray:
    """Extracts the values from `array` for the specified splitting."""

    num_splits = splitting[0] * splitting[1]
    shape = array.shape[:-2] + splitting + (array.shape[-1],)
    flat_shape = array.shape[:-2] + (num_splits, array.shape[-1])

    idxs = []
    orders_x = range(-splitting[0] // 2 + 1, splitting[0] // 2 + 1)
    orders_y = range(-splitting[1] // 2 + 1, splitting[1] // 2 + 1)
    for ox, oy in itertools.product(orders_x, orders_y):
        idxs.append(common.index_for_order((ox, oy), expansion))

    extracted = jnp.zeros(flat_shape, dtype=array.dtype)
    extracted = extracted.at[..., :, :].set(array[..., idxs, :])
    return extracted.reshape(shape)


# -----------------------------------------------------------------------------
# Diffractive splitter with 7.2 x 7.2 um design region.
# -----------------------------------------------------------------------------


DIFFRACTIVE_SPLITTER_SPEC = common.GratingSpec(
    permittivity_ambient=(1.46 + 0.0j) ** 2,
    permittivity_grating=(1.46 + 0.0j) ** 2,
    permittivity_encapsulation=(1.0 + 0.0j) ** 2,
    permittivity_substrate=(1.0 + 0.0j) ** 2,
    thickness_grating=0.692,
    period_x=7.2,
    period_y=7.2,
)

DIFFRACTIVE_SPLITTER_SIM_PARAMS = common.GratingSimParams(
    grid_shape=(360, 360),
    wavelength=0.6328,
    polarization=common.TM,
    formulation=fmm.Formulation.JONES_DIRECT,
    approximate_num_terms=800,
    truncation=basis.Truncation.CIRCULAR,
)

# Minimum width and spacing are 400 nm for the default dimensions.
MINIMUM_WIDTH = 20
MINIMUM_SPACING = 20

# Objective is to split into a 7 x 7 array of beams.
SPLITTING = (7, 7)


def diffractive_splitter(
    minimum_width: int = MINIMUM_WIDTH,
    minimum_spacing: int = MINIMUM_SPACING,
    thickness_initializer: ThicknessInitializer = common.identity_initializer,
    density_initializer: DensityInitializer = common.identity_initializer,
    splitting: Tuple[int, int] = SPLITTING,
) -> DiffractiveSplitterChallenge:
    """Diffractive splitter with 7.2 x 7.2 um design region."""
    return DiffractiveSplitterChallenge(
        component=DiffractiveSplitterComponent(
            spec=DIFFRACTIVE_SPLITTER_SPEC,
            sim_params=DIFFRACTIVE_SPLITTER_SIM_PARAMS,
            thickness_initializer=thickness_initializer,
            density_initializer=density_initializer,
            minimum_width=minimum_width,
            minimum_spacing=minimum_spacing,
            symmetries=(),
        ),
        splitting=splitting,
    )
