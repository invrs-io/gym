"""Component based on the RGB metalens from the photonics opt testbed.

Copyright (c) 2023 The INVRS-IO authors.
"""

import dataclasses
import functools
from typing import Any, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as onp
from fmmax import basis, fields, fmm, scattering, sources
from jax import tree_util
from scipy import ndimage
from totypes import json_utils, types

from invrs_gym import utils
from invrs_gym.challenges import base


DENSITY_LOWER_BOUND = 0.0
DENSITY_UPPER_BOUND = 1.0

EFIELD = "efield"
HFIELD = "hfield"
FIELD_COORDINATES = "field_coordinates"


@dataclasses.dataclass
class MetalensSpec:
    """Defines the physical specification of the a metalens."""

    permittivity_ambient: complex
    permittivity_metalens: complex
    permittivity_substrate: complex
    thickness_ambient: float
    thickness_lens: float
    thickness_substrate: float

    width_lens: float

    focus_offset: float
    lens_offset: float
    source_offset: float
    source_smoothing_fwhm: float

    grid_spacing: float

    @property
    def width(self) -> float:
        return self.width_lens + 2 * self.lens_offset

    @property
    def grid_shape(self) -> Tuple[int, int]:
        with jax.ensure_compile_time_eval():
            return (int(jnp.round(self.width / self.grid_spacing)), 1)


@dataclasses.dataclass
class MetalensSimParams:
    """Parameters that configure the simulation of a metalens."""

    wavelength: jnp.ndarray
    approximate_num_terms: int
    formulation: fmm.Formulation
    num_layers: int


@dataclasses.dataclass
class MetalensResponse:
    """Contains the response of the metalens."""

    wavelength: jnp.ndarray
    ex: jnp.ndarray
    ey: jnp.ndarray
    ez: jnp.ndarray
    hx: jnp.ndarray
    hy: jnp.ndarray
    hz: jnp.ndarray


# json_utils.register_custom_type(MetalensResponse)

tree_util.register_pytree_node(
    MetalensResponse,
    lambda r: (
        (
            r.wavelength,
            r.ex,
            r.ey,
            r.ez,
            r.hx,
            r.hy,
            r.hz,
        ),
        None,
    ),
    lambda _, children: MetalensResponse(*children),
)

# -----------------------------------------------------------------------------
# Define the metalens component.
# -----------------------------------------------------------------------------


@dataclasses.dataclass
class MetalensComponent(base.Component):
    """A metalens component whose only optimizable parameter is density."""

    def __init__(
        self,
        spec: MetalensSpec,
        sim_params: MetalensSimParams,
        density_initializer: base.DensityInitializer,
        **seed_density_kwargs: Any,
    ) -> None:
        """Initializes the metalens components.

        Args:
            spec: Defines the physical specification of the metalens.
            sim_params: Defines simulation parameters for the metalens.
            density_initializer: Callable which generates the initial density from
                a random key and the seed density.
            **seed_density_kwargs: Keyword arguments which set the attributes of
                the seed density used to generate the inital parameters.
        """

        self.spec = spec
        self.sim_params = sim_params
        self.seed_density = seed_density(spec=self.spec, **seed_density_kwargs)
        self.density_initializer = density_initializer

        # Manually create the expansion.
        nmax = sim_params.approximate_num_terms // 2
        ix = onp.zeros((2 * nmax + 1,), dtype=int)
        ix[1::2] = -onp.arange(1, nmax + 1)
        ix[2::2] = onp.arange(1, nmax + 1)
        assert tuple(ix[:5].tolist()) == (0, -1, 1, -2, 2)
        ix = jnp.asarray(ix, dtype=int)
        self.expansion = basis.Expansion(
            basis_coefficients=jnp.stack([ix, jnp.zeros_like(ix)], axis=-1)
        )

    def init(self, key: jax.Array) -> types.Density2DArray:
        """Return the initial parameters for the metalens component."""
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
        compute_fields: bool = False,
    ) -> Tuple[MetalensResponse, base.AuxDict]:
        """Computes the response of the metalens.

        The response consists of the of the fields at the focal point for incident
        plane wave having x- and y-polarized electric fields.

        Args:
            params: The parameters defining the metalens, matching those returned
                by the `init` method.
            wavelength: Optional wavelength to override the default in `sim_params`.
            expansion: Optional expansion to override the default `expansion`.
            compute_fields: If `True`, the `aux` will include an x-z cross section
                of the electric and magnetic fields.

        Returns:
            The `(response, aux)` tuple.
        """
        if expansion is None:
            expansion = self.expansion
        if wavelength is None:
            wavelength = self.sim_params.wavelength
        ((ex, ey, ez), (hx, hy, hz)), aux = simulate_metalens(
            density=params,
            spec=self.spec,
            wavelength=jnp.asarray(wavelength),
            expansion=expansion,
            num_layers=self.sim_params.num_layers,
            formulation=self.sim_params.formulation,
            compute_fields=compute_fields,
        )
        response = MetalensResponse(
            wavelength=jnp.asarray(wavelength), ex=ex, ey=ey, ez=ez, hx=hx, hy=hy, hz=hz
        )
        return response, aux


def seed_density(spec: MetalensSpec, **kwargs: Any) -> types.Density2DArray:
    """Return the seed density for a grating component.

    Args:
        spec: The specification of the metalens component.
        kwargs: keyword arguments specifying additional properties of the seed
            density, e.g. symmetries.

    Returns:
        The seed density.
    """

    # Check kwargs that are required for a metalens component.
    invalid_kwargs = ("array", "lower_bound", "upper_bound", "periodic")
    if any(k in invalid_kwargs for k in kwargs):
        raise ValueError(
            f"Attributes were specified which confict with automatically-extracted "
            f"attributes. Got {kwargs.keys()} when {invalid_kwargs} are automatically "
            f"extracted."
        )

    mid_density_value = (DENSITY_LOWER_BOUND + DENSITY_UPPER_BOUND) / 2

    density_shape = (
        spec.grid_shape[0],
        int(jnp.round(spec.thickness_lens / spec.grid_spacing)),
    )
    fixed_solid = onp.zeros(density_shape, dtype=bool)
    fixed_void = onp.zeros(density_shape, dtype=bool)
    fixed_void[:, 0] = True  # Top, adjacent to ambient.
    delta = int(jnp.round((spec.lens_offset / spec.grid_spacing)))
    fixed_void[:delta, :-1] = True
    fixed_void[-delta:, :-1] = True
    fixed_solid[:, -1] = True  # Bottom, adjacent to substrate
    return types.Density2DArray(
        array=jnp.full(density_shape, mid_density_value),
        lower_bound=DENSITY_LOWER_BOUND,
        upper_bound=DENSITY_UPPER_BOUND,
        fixed_solid=fixed_solid,
        fixed_void=fixed_void,
        periodic=(False, False),
        **kwargs,
    )


# -----------------------------------------------------------------------------
# Simulation method for the metalens.
# -----------------------------------------------------------------------------


def simulate_metalens(
    density: types.Density2DArray,
    spec: MetalensSpec,
    wavelength: jnp.ndarray,
    expansion: basis.Expansion,
    num_layers: int,
    formulation: fmm.Formulation,
    compute_fields: bool,
) -> Tuple[
    Tuple[
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ],
    base.AuxDict,
]:
    """Simulates the metalens component."""
    density_array = utils.transforms.rescaled_density_array(
        density,
        lower_bound=DENSITY_LOWER_BOUND,
        upper_bound=DENSITY_UPPER_BOUND,
    )
    assert spec.grid_shape == (density_array.shape[0], 1)
    dim = spec.grid_shape[0]

    # Trim the top and bottom layer from the density, which should be entirely
    # void (ambient material) and entirely solid (substrate material), respectively.
    # Add the single-pixel thickness to the ambient and substrate so the total
    # stack height remains unchanged. This allows the discretization of the metalens
    # in the z-direction to be a bit more accurate.
    density_array = density_array[:, 1:-1]
    thickness_lens = spec.thickness_lens - 2 * spec.grid_spacing
    thickness_ambient = spec.thickness_ambient + spec.grid_spacing
    thickness_substrate = spec.thickness_substrate + spec.grid_spacing
    layer_thicknesses = (
        [jnp.asarray(spec.focus_offset + thickness_ambient)]
        + [jnp.asarray(thickness_lens / num_layers)] * num_layers
        + [jnp.asarray(thickness_substrate)]
    )

    # Split the metalens density into `num_layers` slices.
    density_array = utils.transforms.resample(
        density_array, shape=(density_array.shape[0], num_layers)
    )
    metalens_densities = jnp.split(density_array, list(range(1, num_layers)), axis=-1)
    assert len(metalens_densities) == num_layers
    assert all([d.shape == spec.grid_shape for d in metalens_densities])
    metalens_permittivities = [
        utils.transforms.interpolate_permittivity(
            permittivity_solid=jnp.asarray(spec.permittivity_metalens),
            permittivity_void=jnp.asarray(spec.permittivity_ambient),
            density=d,
        )
        for d in metalens_densities
    ]

    in_plane_wavevector = jnp.zeros((2,))
    primitive_lattice_vectors = basis.LatticeVectors(
        u=spec.width * basis.X,
        v=basis.Y,
    )

    eigensolve_fn = functools.partial(
        fmm.eigensolve_isotropic_media,
        wavelength=wavelength,
        in_plane_wavevector=in_plane_wavevector,
        primitive_lattice_vectors=primitive_lattice_vectors,
        expansion=expansion,
        formulation=formulation,
    )

    with jax.ensure_compile_time_eval():
        solve_result_ambient = eigensolve_fn(
            permittivity=jnp.full((1, 1), spec.permittivity_ambient)
        )
        solve_results_metalens = [
            eigensolve_fn(permittivity=p) for p in metalens_permittivities
        ]
        solve_result_substrate = eigensolve_fn(
            permittivity=jnp.full((1, 1), spec.permittivity_substrate)
        )

    layer_solve_results = (
        [solve_result_ambient] + solve_results_metalens + [solve_result_substrate]
    )

    if compute_fields:
        # If the field calculation is desired, compute the interior scattering
        # matrices. For each layer in the stack, the interior scattering matrices
        # consist of a pair of matrices, one for the substack below the layer, and
        # one for the substack above the layer.
        s_matrices_interior = scattering.stack_s_matrices_interior(
            layer_solve_results=layer_solve_results,
            layer_thicknesses=layer_thicknesses,
        )
        s_matrix = s_matrices_interior[-1][0]
    else:
        s_matrix = scattering.stack_s_matrix(
            layer_solve_results=layer_solve_results,
            layer_thicknesses=layer_thicknesses,
        )

    # Compute the source, consisting of a smoothed step function.
    # TODO: a jax implementation of the smoothing operation.
    with jax.ensure_compile_time_eval():
        x = onp.arange(dim) / dim * spec.width
        profile = (x > spec.source_offset) & (x < spec.width - spec.source_offset)
        profile = profile.astype(float)
        sigma = spec.source_smoothing_fwhm / (2 * jnp.sqrt(2 * jnp.log(2)))
        sigma_pixels = sigma / spec.grid_spacing
        profile = ndimage.gaussian_filter1d(profile, sigma=sigma_pixels)
        profile = jnp.asarray(profile)[:, jnp.newaxis]
        norm = 1 / jnp.sqrt(spec.permittivity_substrate)
        incident_ex = jnp.stack([-profile, jnp.zeros_like(profile)], axis=-1) * norm
        incident_ey = jnp.stack([jnp.zeros_like(profile), profile], axis=-1) * norm
        incident_hx = jnp.stack([jnp.zeros_like(profile), profile], axis=-1)
        incident_hy = jnp.stack([profile, jnp.zeros_like(profile)], axis=-1)

    _, bwd_amplitude_substrate_end = sources.amplitudes_for_fields(
        ex=incident_ex,
        ey=incident_ey,
        hx=incident_hx,
        hy=incident_hy,
        layer_solve_result=solve_result_substrate,
        brillouin_grid_axes=None,
    )

    # Compute the field at the focal point.
    bwd_amplitude_ambient_end = s_matrix.s22 @ bwd_amplitude_substrate_end
    bwd_amplitude_ambient_focus = fields.propagate_amplitude(
        bwd_amplitude_ambient_end,
        distance=spec.focus_offset,
        layer_solve_result=solve_result_ambient,
    )
    ef_focus, hf_focus = fields.fields_from_wave_amplitudes(
        forward_amplitude=jnp.zeros_like(bwd_amplitude_ambient_focus),
        backward_amplitude=bwd_amplitude_ambient_focus,
        layer_solve_result=solve_result_ambient,
    )
    (ex_focus, ey_focus, ez_focus), (hx_focus, hy_focus, hz_focus), _ = (
        fields.fields_on_coordinates(
            electric_field=ef_focus,
            magnetic_field=hf_focus,
            layer_solve_result=solve_result_ambient,
            x=jnp.asarray(spec.width / 2),
            y=jnp.asarray(0.0),
        )
    )

    # -------------------------------------------------------------------------
    # Compute fields in an xz cross section.
    # -------------------------------------------------------------------------

    aux = {}
    if compute_fields:
        amplitudes_interior = fields.stack_amplitudes_interior(
            s_matrices_interior=s_matrices_interior,
            forward_amplitude_0_start=jnp.zeros_like(bwd_amplitude_substrate_end),
            backward_amplitude_N_end=bwd_amplitude_substrate_end,
        )
        x = jnp.linspace(0, spec.width, dim)
        y = jnp.zeros_like(x)
        layer_znum = tuple(
            [int(jnp.round(t / spec.grid_spacing) + 1) for t in layer_thicknesses]
        )
        (ex, ey, ez), (hx, hy, hz), (x, y, z) = fields.stack_fields_3d_on_coordinates(
            amplitudes_interior=amplitudes_interior,
            layer_solve_results=layer_solve_results,
            layer_thicknesses=layer_thicknesses,
            layer_znum=layer_znum,
            x=x,
            y=y,
        )
        aux.update(
            {
                EFIELD: (ex, ey, ez),
                HFIELD: (hx, hy, hz),
                FIELD_COORDINATES: (x, y, z),
            }
        )

    return ((ex_focus, ey_focus, ez_focus), (hx_focus, hy_focus, hz_focus)), aux
