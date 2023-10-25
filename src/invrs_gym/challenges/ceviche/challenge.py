"""Defines the ceviche challenges.

Copyright (c) 2023 The INVRS-IO authors.
"""

import dataclasses
import functools
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import agjax  # type: ignore[import-untyped]
import jax
import jax.numpy as jnp
import numpy as onp
from totypes import types

from invrs_gym.challenges.ceviche import defaults
from invrs_gym.challenges.ceviche import transmission_loss

AuxDict = Dict[str, Any]
Params = Any
DensityInitializer = Callable[[jax.Array, types.Density2DArray], types.Density2DArray]

FIELDS = "fields"
SPARAMS = "sparams"
DISTANCE_TO_WINDOW = "distance_to_window"

TRANSMISSION_EXPONENT = 0.5
SCALAR_EXPONENT = 2.0

CEVICHE_DENSITY_LOWER_BOUND = 0.0
CEVICHE_DENSITY_UPPER_BOUND = 1.0


def identity_initializer(
    key: jax.Array, seed_density: types.Density2DArray
) -> types.Density2DArray:
    """A basic identity initializer which returns the seed density."""
    del key
    return seed_density


class CevicheComponent:
    """Defines a general ceviche component."""

    def __init__(
        self,
        ceviche_model: defaults.Model,
        density_initializer: DensityInitializer,
        **seed_density_kwargs: Any,
    ) -> None:
        """Initialize a `CevicheComponent`.

        Args:
            ceviche_model: The model for the component.
            density_initializer: Callable which generates the initial density from
                a random key and the seed density.
            **seed_density_kwargs: Keyword arguments which set the attributes of
                the seed density used to generate the inital parameters.
        """
        self.ceviche_model = ceviche_model
        self.seed_density = _seed_density(ceviche_model, **seed_density_kwargs)
        self.density_initializer = density_initializer

    def init(self, key: jax.Array) -> types.Density2DArray:
        """Return the initial parameters for the component."""
        return self.density_initializer(key, self.seed_density)

    def response(
        self,
        params: types.Density2DArray,
        excite_port_idxs: Sequence[int] = (0,),
        wavelengths_nm: Optional[jnp.ndarray] = None,
        max_parallelizm: Optional[int] = None,
    ) -> Tuple[jnp.ndarray, AuxDict]:
        """Compute the response of the component and auxilliary quantities."""

        # The ceviche simulation function is autograd-differentiable. Wrap it so that
        # it can be differentiated using jax.

        @functools.partial(
            agjax.wrap_for_jax,
            nondiff_argnums=(1, 2, 3),
            nondiff_outputnums=(1,),
        )
        def sim_fn(
            design_variable: jnp.ndarray,
            excite_port_idxs: Sequence[int],
            wavelengths_nm: Optional[jnp.ndarray],
            max_parallelizm: Optional[int],
        ) -> Tuple[jnp.ndarray, onp.ndarray[Any, Any]]:
            s_params, fields = self.ceviche_model.simulate(
                design_variable, excite_port_idxs, wavelengths_nm, max_parallelizm
            )
            return s_params, fields

        sparams, fields = sim_fn(
            params.array, excite_port_idxs, wavelengths_nm, max_parallelizm
        )
        return sparams, {SPARAMS: sparams, FIELDS: fields}


def _seed_density(ceviche_model: defaults.Model, **kwargs: Any) -> types.Density2DArray:
    """Return the seed density for the `ceviche_model`.

    The seed density has shape and fixed pixels as required by the `ceviche_model`,
    and with other properties determined by keyword arguments.

    Args:
        ceviche_model: The model from which the seed density determined.
        kwargs: keyword arguments specifying additional properties of the seed
            density, e.g. symmetries.

    Returns:
        The seed density.
    """

    # Check kwargs that are computed from `ceviche_model`, which must not be specified.
    invalid_kwargs = (
        "array",
        "fixed_solid",
        "fixed_void",
        "lower_bound",
        "upper_bound",
    )
    if any(k in invalid_kwargs for k in kwargs):
        raise ValueError(
            f"Attributes were specified which confict with automatically-extracted "
            f"attributes. Got {kwargs.keys()} when {invalid_kwargs} are automatically "
            f"extracted."
        )

    shape = ceviche_model.design_variable_shape
    fixed_solid, fixed_void = _fixed_pixels(ceviche_model)
    mid_density_value = (CEVICHE_DENSITY_LOWER_BOUND + CEVICHE_DENSITY_UPPER_BOUND) / 2
    seed_density = types.Density2DArray(
        array=jnp.full(shape, mid_density_value),
        fixed_solid=fixed_solid,
        fixed_void=fixed_void,
        # For the ceviche challenges, density must lie between 0 and 1.
        lower_bound=CEVICHE_DENSITY_LOWER_BOUND,
        upper_bound=CEVICHE_DENSITY_UPPER_BOUND,
        **kwargs,
    )
    return seed_density


def _fixed_pixels(ceviche_model: defaults.Model) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Determine the fixed pixels for the given ceviche model.

    The fixed pixels exist around the border of the design, and are solid where the
    adjacent pixels outside the design are solid. They are void where the adjacent
    pixels outside the design are void.

    Args:
        ceviche_model: The model from which the fixed pixels are determined.

    Returns:
        The fixed solid and void pixels.
    """
    design_shape = ceviche_model.design_variable_shape
    density_fn = ceviche_model.density

    assert not onp.any(density_fn(jnp.ones(design_shape)) == 0.5)
    density = density_fn(jnp.full(design_shape, 0.5))

    i, j = onp.where(density == 0.5)
    i_lo = i[0]
    i_hi = i[-1]
    j_lo = j[0]
    j_hi = j[-1]
    assert i_lo > 0 and j_lo > 0
    assert i_hi < density.shape[0] - 1 and j_hi < density.shape[1] - 1

    # Compute the fixed solid pixels.
    fixed_solid = onp.zeros(design_shape, dtype=bool)
    fixed_solid[0, :] = density[i_lo - 1, j_lo : j_hi + 1] == 1
    fixed_solid[-1, :] = density[i_hi + 1, j_lo : j_hi + 1] == 1
    fixed_solid[:, 0] = density[i_lo : i_hi + 1, j_lo - 1] == 1
    fixed_solid[:, -1] = density[i_lo : i_hi + 1, j_hi + 1] == 1

    # Compute the fixed void pixels.
    fixed_void = onp.zeros(design_shape, dtype=bool)
    fixed_void[0, :] = density[i_lo - 1, j_lo : j_hi + 1] == 0
    fixed_void[-1, :] = density[i_hi + 1, j_lo : j_hi + 1] == 0
    fixed_void[:, 0] = density[i_lo : i_hi + 1, j_lo - 1] == 0
    fixed_void[:, -1] = density[i_lo : i_hi + 1, j_hi + 1] == 0

    return jnp.asarray(fixed_solid), jnp.asarray(fixed_void)


# -----------------------------------------------------------------------------
# Base class for ceviche challenges.
# -----------------------------------------------------------------------------


@dataclasses.dataclass
class CevicheChallenge:
    """Defines a general ceviche challenge.

    The objective of the ceviche challenge is to find a component that whose
    transmission into its various ports lies within the target window defined
    by the transmission lower and upper bounds.

    Attributes:
        component: The component to be designed.
        transmission_lower_bound: The lower bound of the transmission window.
        transmission_upper_bound: The upper bound of the transmission window.
    """

    component: CevicheComponent
    transmission_lower_bound: jnp.ndarray
    transmission_upper_bound: jnp.ndarray

    def loss(self, response: jnp.ndarray) -> jnp.ndarray:
        """Compute a scalar loss from the component `response`."""
        # Power transmission is the squared magnitude of the scattering parameters.
        transmission = jnp.abs(response) ** 2
        # Repeat the bounds to match the transmission shape. Each value in the bounds
        # is then interpreted as the bounds for a wavelength band.
        lb = _wavelength_bound(self.transmission_lower_bound, transmission.shape)
        ub = _wavelength_bound(self.transmission_upper_bound, transmission.shape)
        return transmission_loss.orthotope_smooth_transmission_loss(
            transmission=transmission,
            window_lower_bound=lb,
            window_upper_bound=ub,
            transmission_exponent=jnp.asarray(TRANSMISSION_EXPONENT),
            scalar_exponent=jnp.asarray(SCALAR_EXPONENT),
        )

    def metrics(
        self,
        response: jnp.ndarray,
        params: types.Density2DArray,
        aux: AuxDict,
    ) -> AuxDict:
        """Compute challenge metrics."""
        del params, aux
        transmission = jnp.abs(response) ** 2
        lb = _wavelength_bound(self.transmission_lower_bound, transmission.shape)
        ub = _wavelength_bound(self.transmission_upper_bound, transmission.shape)
        distance_to_window = transmission_loss.distance_to_window(
            transmission=transmission,
            window_lower_bound=lb,
            window_upper_bound=ub,
        )
        return {DISTANCE_TO_WINDOW: distance_to_window}


def _wavelength_bound(
    band_bound: jnp.ndarray, transmission_shape: Tuple[int, ...]
) -> jnp.ndarray:
    """Obtain per-wavelength bound compatible with `transmission_shape`."""
    assert len(transmission_shape) == 3
    assert band_bound.ndim <= 3

    # Add leading batch dimensions as needed.
    dims_to_add = range(0, len(transmission_shape) - band_bound.ndim)
    band_bound = jnp.expand_dims(band_bound, axis=dims_to_add)

    if not (transmission_shape[0] % band_bound.shape[0]) == 0:
        raise ValueError(
            f"Could not repeat `band_bound` with shape {band_bound.shape} to match "
            f"`transmission_shape` of {transmission_shape}; leading dimension size "
            f"must evenly divide the transmission shape."
        )

    repeats = transmission_shape[0] // band_bound.shape[0]
    repeated: jnp.ndarray = jnp.repeat(band_bound, repeats, axis=0)
    return repeated


# -----------------------------------------------------------------------------
# Constructors for ceviche challenges, including lightweight versions.
# -----------------------------------------------------------------------------


def beam_splitter(
    minimum_width: int = defaults.MINIMUM_WIDTH,
    minimum_spacing: int = defaults.MINIMUM_SPACING,
    density_initializer: DensityInitializer = identity_initializer,
) -> CevicheChallenge:
    """Beamsplitter with 3.2 x 2.0 um design and standard simulation params."""
    return CevicheChallenge(
        component=CevicheComponent(
            ceviche_model=defaults.BEAM_SPLITTER_MODEL,
            symmetries=defaults.BEAM_SPLITTER_SYMMETRIES,
            minimum_width=minimum_width,
            minimum_spacing=minimum_spacing,
            density_initializer=density_initializer,
        ),
        transmission_lower_bound=defaults.BEAM_SPLITTER_TRANSMISSION_LOWER_BOUND,
        transmission_upper_bound=defaults.BEAM_SPLITTER_TRANSMISSION_UPPER_BOUND,
    )


def lightweight_beam_splitter(
    minimum_width: int = defaults.LIGHTWEIGHT_MINIMUM_WIDTH,
    minimum_spacing: int = defaults.LIGHTWEIGHT_MINIMUM_SPACING,
    density_initializer: DensityInitializer = identity_initializer,
) -> CevicheChallenge:
    """Beamsplitter with 3.2 x 2.0 um design and lightweight simulation params."""
    return CevicheChallenge(
        component=CevicheComponent(
            ceviche_model=defaults.LIGHTWEIGHT_BEAM_SPLITTER_MODEL,
            symmetries=defaults.BEAM_SPLITTER_SYMMETRIES,
            minimum_width=minimum_width,
            minimum_spacing=minimum_spacing,
            density_initializer=density_initializer,
        ),
        transmission_lower_bound=defaults.BEAM_SPLITTER_TRANSMISSION_LOWER_BOUND,
        transmission_upper_bound=defaults.BEAM_SPLITTER_TRANSMISSION_UPPER_BOUND,
    )


def mode_converter(
    minimum_width: int = defaults.MINIMUM_WIDTH,
    minimum_spacing: int = defaults.MINIMUM_SPACING,
    density_initializer: DensityInitializer = identity_initializer,
) -> CevicheChallenge:
    """Mode converter with 1.6 x 1.6 um design and standard simulation params."""
    return CevicheChallenge(
        component=CevicheComponent(
            ceviche_model=defaults.MODE_CONVERTER_MODEL,
            minimum_width=minimum_width,
            minimum_spacing=minimum_spacing,
            density_initializer=density_initializer,
        ),
        transmission_lower_bound=defaults.MODE_CONVERTER_TRANSMISSION_LOWER_BOUND,
        transmission_upper_bound=defaults.MODE_CONVERTER_TRANSMISSION_UPPER_BOUND,
    )


def lightweight_mode_converter(
    minimum_width: int = defaults.LIGHTWEIGHT_MINIMUM_WIDTH,
    minimum_spacing: int = defaults.LIGHTWEIGHT_MINIMUM_SPACING,
    density_initializer: DensityInitializer = identity_initializer,
) -> CevicheChallenge:
    """Mode converter with 1.6 x 1.6 um design and lightweight simulation params."""
    return CevicheChallenge(
        component=CevicheComponent(
            ceviche_model=defaults.LIGHTWEIGHT_MODE_CONVERTER_MODEL,
            minimum_width=minimum_width,
            minimum_spacing=minimum_spacing,
            density_initializer=density_initializer,
        ),
        transmission_lower_bound=defaults.MODE_CONVERTER_TRANSMISSION_LOWER_BOUND,
        transmission_upper_bound=defaults.MODE_CONVERTER_TRANSMISSION_UPPER_BOUND,
    )


def waveguide_bend(
    minimum_width: int = defaults.MINIMUM_WIDTH,
    minimum_spacing: int = defaults.MINIMUM_SPACING,
    density_initializer: DensityInitializer = identity_initializer,
) -> CevicheChallenge:
    """Waveguide bend with 1.6 x 1.6 um design and standard simulation params."""
    return CevicheChallenge(
        component=CevicheComponent(
            ceviche_model=defaults.WAVEGUIDE_BEND_MODEL,
            symmetries=defaults.WAVEGUIDE_BEND_SYMMETRIES,
            minimum_width=minimum_width,
            minimum_spacing=minimum_spacing,
            density_initializer=density_initializer,
        ),
        transmission_lower_bound=defaults.WAVEGUIDE_BEND_TRANSMISSION_LOWER_BOUND,
        transmission_upper_bound=defaults.WAVEGUIDE_BEND_TRANSMISSION_UPPER_BOUND,
    )


def lightweight_waveguide_bend(
    minimum_width: int = defaults.LIGHTWEIGHT_MINIMUM_WIDTH,
    minimum_spacing: int = defaults.LIGHTWEIGHT_MINIMUM_SPACING,
    density_initializer: DensityInitializer = identity_initializer,
) -> CevicheChallenge:
    """Waveguide bend with 1.6 x 1.6 um design and lightweight simulation params."""
    return CevicheChallenge(
        component=CevicheComponent(
            ceviche_model=defaults.LIGHTWEIGHT_WAVEGUIDE_BEND_MODEL,
            symmetries=defaults.WAVEGUIDE_BEND_SYMMETRIES,
            minimum_width=minimum_width,
            minimum_spacing=minimum_spacing,
            density_initializer=density_initializer,
        ),
        transmission_lower_bound=defaults.WAVEGUIDE_BEND_TRANSMISSION_LOWER_BOUND,
        transmission_upper_bound=defaults.WAVEGUIDE_BEND_TRANSMISSION_UPPER_BOUND,
    )


def wdm(
    minimum_width: int = defaults.MINIMUM_WIDTH,
    minimum_spacing: int = defaults.MINIMUM_SPACING,
    density_initializer: DensityInitializer = identity_initializer,
) -> CevicheChallenge:
    """Demultiplexer with 6.4 x 6.4 um design and standard simulation params."""
    return CevicheChallenge(
        component=CevicheComponent(
            ceviche_model=defaults.WDM_MODEL,
            minimum_width=minimum_width,
            minimum_spacing=minimum_spacing,
            density_initializer=density_initializer,
        ),
        transmission_lower_bound=defaults.WDM_TRANSMISSION_LOWER_BOUND,
        transmission_upper_bound=defaults.WDM_TRANSMISSION_UPPER_BOUND,
    )


def lightweight_wdm(
    minimum_width: int = defaults.LIGHTWEIGHT_MINIMUM_WIDTH,
    minimum_spacing: int = defaults.LIGHTWEIGHT_MINIMUM_SPACING,
    density_initializer: DensityInitializer = identity_initializer,
) -> CevicheChallenge:
    """Waveguide bend with 3.2 x 3.2 um design and lightweight simulation params."""
    return CevicheChallenge(
        component=CevicheComponent(
            ceviche_model=defaults.LIGHTWEIGHT_WDM_MODEL,
            minimum_width=minimum_width,
            minimum_spacing=minimum_spacing,
            density_initializer=density_initializer,
        ),
        transmission_lower_bound=defaults.WDM_TRANSMISSION_LOWER_BOUND,
        transmission_upper_bound=defaults.WDM_TRANSMISSION_UPPER_BOUND,
    )
