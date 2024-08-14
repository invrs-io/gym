"""Defines the meta-atom library challenge.

Copyright (c) 2024 The INVRS-IO authors.
"""

import dataclasses
import functools
from typing import Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as onp
from jax import tree_util
from fmmax import basis, fmm
from totypes import types

from invrs_gym.challenges import base
from invrs_gym.challenges.library import component as library_component
from invrs_gym.utils import initializers, materials

METAGRATING_EFFICIENCY_RHCP = "metagrating_efficiency_rhcp"
METAGRATING_EFFICIENCY_LHCP = "metagrating_efficiency_lhcp"
METAGRATING_RELATIVE_EFFICIENCY_RHCP = "metagrating_relative_efficiency_rhcp"
METAGRATING_RELATIVE_EFFICIENCY_LHCP = "metagrating_relative_efficiency_lhcp"
METAGRATING_AVERAGE_EFFICIENCY = "metagrating_average_efficiency"
METAGRATING_AVERAGE_RELATIVE_EFFICIENCY = "metagrating_average_relative_efficiency"


@dataclasses.dataclass
class LibraryChallenge(base.Challenge):
    """The meta-atom library design challenge."""

    component: library_component.LibraryComponent

    def loss(self, response: library_component.LibraryResponse) -> jnp.ndarray:
        """Compute a scalar loss from the component `response`."""
        response = response_with_optimal_rotation(response, self.component.spec)
        (efficiency_rhcp, efficiency_lhcp), _ = metagrating_efficiency(
            response, self.component.spec
        )
        loss_rhcp = jnp.sum(jnp.abs(1 - efficiency_rhcp)) ** 2
        loss_lhcp = jnp.sum(jnp.abs(1 - efficiency_lhcp)) ** 2
        return loss_rhcp + loss_lhcp

    def eval_metric(
        self,
        response: library_component.LibraryResponse,
    ) -> jnp.ndarray:
        """Computes the eval metric from the component `response`.

        The eval metric considers a metagrating assembled from the eight meta-atoms in
        the library. The relative efficiency of the grating (i.e. transmitted power
        into the target order divided by total transmitted power) is computed for each
        wavelength and the two incident polarization states. The eval metric is the
        minimum relative efficiency among all these cases.

        Args:
            response: The component response.

        Returns:
            The scalar eval metric.
        """
        response = response_with_optimal_rotation(response, self.component.spec)
        (
            _,
            (relative_efficiency_rhcp, relative_efficiency_lhcp),
        ) = metagrating_efficiency(response, self.component.spec)
        return jnp.minimum(
            jnp.amin(relative_efficiency_rhcp),
            jnp.amin(relative_efficiency_lhcp),
        )

    def metrics(
        self,
        response: library_component.LibraryResponse,
        params: library_component.Params,
        aux: base.AuxDict,
    ) -> base.AuxDict:
        """Compute challenge metrics.

        Several challenge metrics relate to the performance of a metagrating assembled
        from the meta-atom library.

        Args:
            response: The response of the meta-atom library.
            params: The parameters where the response was evaluated.
            aux: The auxilliary quantities returned by the component response method.

        Returns:
            The metrics dictionary, with the following quantities:
                - Per-wavelength, per-polarization metagrating efficiency for RHCP
                  excitation.
                - Per-wavelength, per-polarization metagrating efficiency for LHCP
                  excitation.
                - Per-wavelength, per-polarization metagrating relative efficiency for
                  RHCP excitation.
                - Per-wavelength, per-polarization metagrating relative efficiency for
                  LHCP excitation.
                - Average metagrating efficiency.
                - Average metagrating relative efficiency.
        """
        metrics = super().metrics(response, params, aux)
        response = response_with_optimal_rotation(response, self.component.spec)
        (
            (efficiency_rhcp, efficiency_lhcp),
            (relative_efficiency_rhcp, relative_efficiency_lhcp),
        ) = metagrating_efficiency(response, self.component.spec)
        metrics.update(
            {
                METAGRATING_EFFICIENCY_RHCP: efficiency_rhcp,
                METAGRATING_EFFICIENCY_LHCP: efficiency_lhcp,
                METAGRATING_RELATIVE_EFFICIENCY_RHCP: relative_efficiency_rhcp,
                METAGRATING_RELATIVE_EFFICIENCY_LHCP: relative_efficiency_lhcp,
                METAGRATING_AVERAGE_EFFICIENCY: jnp.mean(
                    jnp.asarray([efficiency_rhcp, efficiency_lhcp])
                ),
                METAGRATING_AVERAGE_RELATIVE_EFFICIENCY: jnp.mean(
                    jnp.asarray([relative_efficiency_rhcp, relative_efficiency_lhcp])
                ),
            }
        )
        return metrics


def metagrating_efficiency(
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


# -----------------------------------------------------------------------------
# Functions related to nanostructure rotations.
# -----------------------------------------------------------------------------


def response_with_optimal_rotation(
    response: library_component.LibraryResponse,
    spec: library_component.LibrarySpec,
) -> library_component.LibraryResponse:
    """Return a modified response with the optimal per-nanostructure rotation.

    The optimal rotation is the one which yields the highest relative efficiency
    averaged across all wavelengths for a diffraction grating built from the meta
    atom library.

    Args:
        response: The original response.
        spec: The physical specification of the meta-atom library.

    Returns:
        The response for the optimal rotation.
    """
    rotation_idx = optimal_rotation(response, spec)
    responses = _all_rotations_response(response)
    return tree_util.tree_map(lambda x: x[rotation_idx, ...], responses)


def optimal_rotation(
    response: library_component.LibraryResponse,
    spec: library_component.LibrarySpec,
) -> jnp.ndarray:
    """Return an integer representing the optimal per-nanostructure rotation.

    The optimal rotation is the one which yields the highest relative efficiency
    averaged across all wavelengths for a diffraction grating built from the meta
    atom library.

    Args:
        response: The original response.
        spec: The physical specification of the meta-atom library.

    Returns:
        The integer representing the optimal rotation. If n-th digit of the reversed
        binary representation of the optimal representation is 1, this indicates that
        the n-th nanostructure should be rotated.
    """
    responses = _all_rotations_response(response)
    (
        _,
        (relative_efficiency_rhcp, relative_efficiency_lhcp),
    ) = jax.vmap(
        metagrating_efficiency, in_axes=(0, None)
    )(responses, spec)

    # Compute average relative efficiency across wavelength and incident polarization.
    relative_efficiency = 0.5 * (relative_efficiency_rhcp + relative_efficiency_lhcp)
    relative_efficiency = jnp.mean(
        relative_efficiency,
        axis=tuple(range(1, relative_efficiency.ndim)),
    )
    return jnp.argmax(relative_efficiency)


def rotate_params(
    params: library_component.Params,
    rotation_idx: jnp.ndarray,
) -> library_component.Params:
    """Applies the specified rotation to nanostructures in `params`.

    Args:
        params: The params to be rotated.
        rotation_idx: The integer representing the rotation. If n-th digit of the
            reversed binary representation of the representation is 1, this indicates
            that the n-th nanostructure should be rotated.

    Returns:
        The parameters with rotated nanostructures.
    """
    rotation_idx = jnp.asarray(rotation_idx)
    assert rotation_idx.shape == ()
    density: types.Density2DArray
    density = params[library_component.DENSITY]  # type: ignore[assignment]
    num_nanostructures = density.shape[0]
    is_rotated = _rotation_for_idx(rotation_idx, num_nanostructures)

    array = jnp.stack(
        [
            jnp.rot90(x) if is_rotated_i else x
            for x, is_rotated_i in zip(density.array, is_rotated, strict=True)
        ],
        axis=0,
    )
    return {
        library_component.THICKNESS: params[library_component.THICKNESS],
        library_component.DENSITY: dataclasses.replace(density, array=array),
    }


def rotation_idx_from_is_rotated(is_rotated: jnp.ndarray) -> jnp.ndarray:
    """Return the rotation index for given `is_rotated`."""
    assert is_rotated.ndim == 1
    assert is_rotated.dtype == bool
    return jnp.sum(is_rotated * 2 ** jnp.arange(is_rotated.size)).astype(int)


def _all_rotations_response(
    response: library_component.LibraryResponse,
) -> library_component.LibraryResponse:
    """Return a batch of responses corresponding to all possible unique rotations."""
    num_nanostructures = response.transmission_rhcp.shape[0]
    is_rotated_nanostructure = _all_rotations(num_nanostructures)
    responses = jax.vmap(_rotate_response, in_axes=(None, 0))(
        response, is_rotated_nanostructure
    )
    return responses


def _rotate_response(
    response: library_component.LibraryResponse,
    is_rotated: jnp.ndarray,
) -> library_component.LibraryResponse:
    """Return response that results from optional 90 degree rotation.

    Args:
        response: The response to be modified.
        is_rotated: Array whose elements indicate whether specific meta-atoms are to
            be rotated. Must have length equal to size of the meta-atom library (i.e.
            the number of nanostructures in the library).

    Returns:
        The response for the specified rotation.
    """
    assert is_rotated.ndim == 1
    assert is_rotated.size == response.transmission_rhcp.shape[0]

    shape = is_rotated.shape + (1,) * (response.transmission_rhcp.ndim - 2)
    ones = jnp.ones(shape)
    shifted = jnp.where(is_rotated.reshape(shape), -ones, ones)

    rhcp_phase = jnp.stack([ones, shifted], axis=-1)
    transmission_rhcp = response.transmission_rhcp * rhcp_phase
    reflection_rhcp = response.reflection_rhcp * rhcp_phase

    lhcp_phase = jnp.stack([shifted, ones], axis=-1)
    transmission_lhcp = response.transmission_lhcp * lhcp_phase
    reflection_lhcp = response.reflection_lhcp * lhcp_phase

    return dataclasses.replace(
        response,
        transmission_rhcp=transmission_rhcp,
        transmission_lhcp=transmission_lhcp,
        reflection_rhcp=reflection_rhcp,
        reflection_lhcp=reflection_lhcp,
    )


def _all_rotations(num_nanostructures: int) -> jnp.ndarray:
    """Return all unique rotations, accounting for degeneracy."""
    num = 2 ** (num_nanostructures - 1)
    is_rotated_nanostructure = jnp.stack(
        [_rotation_for_idx(jnp.asarray(i), num_nanostructures) for i in range(num)],
        axis=0,
    )
    return is_rotated_nanostructure


def _rotation_for_idx(idx: jnp.ndarray, num_nanostructures: int) -> jnp.ndarray:
    """Return the array for rotation `i`."""

    def _fn(idx):
        is_rotated = [int(j) for j in onp.binary_repr(idx, width=num_nanostructures)]
        return jnp.asarray(is_rotated).astype(bool)[::-1]

    return jax.pure_callback(_fn, jnp.zeros((num_nanostructures,), dtype=bool), idx)


# -----------------------------------------------------------------------------
# Define the challenge.
# -----------------------------------------------------------------------------


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
    approximate_num_terms=200,
    formulation=fmm.Formulation.JONES_DIRECT_FOURIER,
    truncation=basis.Truncation.CIRCULAR,
)

SYMMETRIES = ("reflection_n_s",)


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

    In the library design challenge, all possible rotations of the individual meta-
    atoms are considered. The optimal rotation (i.e. that which yield the highest
    wavelength-averaged relative efficiency) is used to compute evaluation metrics
    and loss. The optimal rotations can be found and applied using the
    `optimal_rotation` and `rotate_params` functions in this module.

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
