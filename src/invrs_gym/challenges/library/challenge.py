"""Defines the meta-atom library challenge.

Copyright (c) 2024 The INVRS-IO authors.
"""

import dataclasses
import functools
from typing import Sequence, Tuple

import jax
import jax.numpy as jnp
from fmmax import basis, fmm
from totypes import types

from invrs_gym.challenges import base
from invrs_gym.challenges.library import component as library_component
from invrs_gym.utils import initializers, materials

METAGRATING_EFFICIENCY = "metagrating_efficiency"
METAGRATING_RELATIVE_EFFICIENCY = "metagrating_relative_efficiency"
DISTANCE_TO_TARGET = "distance_to_target"


@dataclasses.dataclass
class LibraryChallenge(base.Challenge):
    """The meta-atom library design challenge."""

    component: library_component.LibraryComponent

    def loss(self, response: library_component.LibraryResponse) -> jnp.ndarray:
        """Compute a scalar loss from the component `response`."""
        (efficiency_rhcp, efficiency_lhcp), _ = _metagrating_efficiency(
            response, self.component.spec
        )
        loss_rhcp = jnp.sum(jnp.abs(1 - efficiency_rhcp)) ** 2
        loss_lhcp = jnp.sum(jnp.abs(1 - efficiency_lhcp)) ** 2
        return loss_rhcp + loss_lhcp

    def metrics(
        self,
        response: library_component.LibraryResponse,
        params: library_component.Params,
        aux: base.AuxDict,
    ) -> base.AuxDict:
        """Compute challenge metrics."""
        metrics = super().metrics(response, params, aux)
        (
            metagrating_efficiency,
            metagrating_relative_efficiency,
        ) = _metagrating_efficiency(response, self.component.spec)
        metrics.update(
            {
                METAGRATING_EFFICIENCY: metagrating_efficiency,
                METAGRATING_RELATIVE_EFFICIENCY: metagrating_relative_efficiency,
                DISTANCE_TO_TARGET: self.distance_to_target(response),
            }
        )
        return metrics

    def distance_to_target(
        self,
        response: library_component.LibraryResponse,
    ) -> jnp.ndarray:
        """Compute distance from the component `response` to the challenge target."""
        del response
        # TODO: implement a distance, or strip distance from the challenge object.
        return jnp.ones(())


def _metagrating_efficiency(
    response: library_component.LibraryResponse,
    spec: library_component.LibrarySpec,
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]:
    """Return efficiency of a metagrating assembled from the meta-atom libary."""

    # Scale the transmission coefficients to account for the difference in
    # material permittivity. With this rescaling, perfect transmission will
    # result in transmission coefficients with a norm of 1.
    permittivity_substrate = materials.permittivity(
        spec.material_substrate, response.wavelength
    )
    permittivity_ambient = materials.permittivity(
        spec.material_ambient, response.wavelength
    )
    scalar = jnp.sqrt(
        jnp.sqrt(permittivity_substrate).real / jnp.sqrt(permittivity_ambient).real
    )
    transmission_rhcp_conserved = response.transmission_rhcp[..., 0] * scalar
    transmission_rhcp_converted = response.transmission_rhcp[..., 1] * scalar
    transmission_lhcp_conserved = response.transmission_lhcp[..., 1] * scalar
    transmission_lhcp_converted = response.transmission_lhcp[..., 0] * scalar

    _fft = functools.partial(jnp.fft.fft, axis=0, norm="forward")
    efficiency_per_order_rhcp = (
        jnp.abs(_fft(transmission_rhcp_conserved)) ** 2
        + jnp.abs(_fft(transmission_rhcp_converted)) ** 2
    )
    efficiency_per_order_lhcp = (
        jnp.abs(_fft(transmission_lhcp_conserved)) ** 2
        + jnp.abs(_fft(transmission_lhcp_converted)) ** 2
    )

    efficiency_rhcp = efficiency_per_order_rhcp[1, ...]
    efficiency_lhcp = efficiency_per_order_lhcp[1, ...]

    total_efficiency_rhcp = jnp.sum(efficiency_per_order_rhcp, axis=0)
    total_efficiency_lhcp = jnp.sum(efficiency_per_order_lhcp, axis=0)

    relative_efficiency_rhcp = efficiency_rhcp / total_efficiency_rhcp
    relative_efficiency_lhcp = efficiency_lhcp / total_efficiency_lhcp

    return (
        (efficiency_rhcp, efficiency_lhcp),
        (relative_efficiency_rhcp, relative_efficiency_lhcp),
    )


def library_density_initializer(
    key: jax.Array,
    seed_density: types.Density2DArray,
    relative_mean_range: float = 0.8,
    relative_noise_amplitude=0.1,
    resize_method: jax.image.ResizeMethod = jax.image.ResizeMethod.CUBIC,
) -> types.Density2DArray:
    """Returns the random initial density."""
    assert seed_density.ndim == 3
    library_size = seed_density.shape[0]
    keys = jax.random.split(key, num=library_size)
    relative_mean_values = (
        jnp.arange(library_size) + 0.5
    ) / library_size * relative_mean_range + (1 - relative_mean_range) / 2
    return jax.vmap(
        initializers.noisy_density_initializer,
        in_axes=(0, 0, 0, None, None),
    )(keys, seed_density, relative_mean_values, relative_noise_amplitude, resize_method)


MINIMUM_WIDTH = 12
MINIMUM_SPACING = 12
LIBRARY_SIZE = 8

LIBRARY_SPEC = library_component.LibrarySpec(
    material_ambient=materials.VACUUM,
    material_metasurface_solid=library_component.TIO2_CHEN,
    material_metasurface_void=materials.VACUUM,
    material_substrate=materials.SIO2,
    background_extinction_coeff=0.0001,
    thickness_ambient=1.2,
    thickness_metasurface=types.BoundedArray(
        array=0.6, lower_bound=0.5, upper_bound=0.7
    ),
    thickness_substrate=0.2,
    pitch=0.4,
    frame_width=0.03,
    grid_spacing=0.005,
)

LIBRARY_SIM_PARAMS = library_component.LibrarySimParams(
    wavelength=jnp.asarray([0.45, 0.55, 0.65]),
    approximate_num_terms=300,
    formulation=fmm.Formulation.JONES_DIRECT_FOURIER,
    truncation=basis.Truncation.CIRCULAR,
)

SYMMETRIES = ("reflection_n_s", "reflection_e_w")


def meta_atom_library(
    minimum_width: int = MINIMUM_WIDTH,
    minimum_spacing: int = MINIMUM_SPACING,
    library_size: int = LIBRARY_SIZE,
    spec: library_component.LibrarySpec = LIBRARY_SPEC,
    sim_params: library_component.LibrarySimParams = LIBRARY_SIM_PARAMS,
    thickness_initializer: base.ThicknessInitializer = (
        initializers.identity_initializer
    ),
    density_initializer: base.DensityInitializer = library_density_initializer,
    symmetries: Sequence[str] = SYMMETRIES,
) -> LibraryChallenge:
    """Return the meta-atom library design challenge.

    The library design challenge is based on "Dispersion-engineered metasurfaces
    reaching broadband 90% relative diffraction efficiency" by Chen et al.
    https://www.nature.com/articles/s41467-023-38185-2

    Args:
        minimum_width: The minimum width target for the challenge, in pixels.
        minimum_spacing: The minimum spacing target for the challenge, in pixels.
        library_size: The number of meta-atoms in the library.
        spec: Defines the physical specification of the meta-atom library.
        sim_params: Defines the simulation parameters of the meta-atom library.
        thickness_initializer: Callable which returns the initial thickness, given a
            key and seed thickness.
        density_initializer: Callable which returns the initial density, given a
            key and seed density.
        symmetries: Defines the symmetries of the meta-atoms.

    Returns:
        The `LibraryChallenge`.
    """
    return LibraryChallenge(
        component=library_component.LibraryComponent(
            spec=spec,
            sim_params=sim_params,
            library_size=library_size,
            thickness_initializer=thickness_initializer,
            density_initializer=density_initializer,
            minimum_width=minimum_width,
            minimum_spacing=minimum_spacing,
            symmetries=symmetries,
        )
    )
