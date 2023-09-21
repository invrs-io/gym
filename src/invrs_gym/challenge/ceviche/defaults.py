"""Defines defaults for the ceviche challenges."""

from ceviche_challenges import units as u
from ceviche_challenges import (
    beam_splitter,
    mode_converter,
    params,
    waveguide_bend,
    wdm,
)


WG_WIDTH = 400 * u.nm
WG_MODE_PADDING = 560 * u.nm
WG_LENGTH = 720 * u.nm

PADDING = 400 * u.nm
PORT_PML_OFFSET = 40 * u.nm

CLADDING_PERMITTIVITY = 2.25
SLAB_PERMITTIVITY = 12.25
INPUT_MONITOR_OFFSET = 40 * u.nm

PML_WIDTH_GRIDPOINTS = 20


# -----------------------------------------------------------------------------
# Simulation parameter defaults.
# -----------------------------------------------------------------------------


SIM_PARAMS = params.CevicheSimParams(
    resolution=10 * u.nm,
    wavelengths=u.Array([1265.0, 1270.0, 1275.0, 1285.0, 1290.0, 1295.0], u.nm),
)

LIGHTWEIGHT_SIM_PARAMS = params.CevicheSimParams(
    resolution=40 * u.nm,
    wavelengths=u.Array([1270.0, 1290.0], u.nm),
)

# -----------------------------------------------------------------------------
# Defaults for the spec, which define the geometry of devices.
# -----------------------------------------------------------------------------

# Beamsplitter with a 2.0 x 3.2 um design region.
BEAM_SPLITTER_SPEC = beam_splitter.spec.BeamSplitterSpec(
    wg_width=WG_WIDTH,
    wg_length=WG_LENGTH,
    wg_separation=1120 * u.nm,
    wg_mode_padding=WG_MODE_PADDING,
    port_pml_offset=PORT_PML_OFFSET,
    variable_region_size=(3200 * u.nm, 2000 * u.nm),
    cladding_permittivity=CLADDING_PERMITTIVITY,
    slab_permittivity=SLAB_PERMITTIVITY,
    input_monitor_offset=INPUT_MONITOR_OFFSET,
    design_symmetry=None,
    pml_width=PML_WIDTH_GRIDPOINTS,
)

# Mode converter with a 1.6 x 1.6 um design region.
MODE_CONVERTER_SPEC = mode_converter.spec.ModeConverterSpec(
    left_wg_width=WG_WIDTH,
    left_wg_mode_padding=WG_MODE_PADDING,
    left_wg_mode_order=1,  # Fundamental mode.
    right_wg_width=WG_WIDTH,
    right_wg_mode_padding=WG_MODE_PADDING,
    right_wg_mode_order=2,  # Second mode.
    wg_length=WG_LENGTH,
    padding=PADDING,
    port_pml_offset=PORT_PML_OFFSET,
    variable_region_size=(1600 * u.nm, 1600 * u.nm),
    cladding_permittivity=CLADDING_PERMITTIVITY,
    slab_permittivity=SLAB_PERMITTIVITY,
    input_monitor_offset=INPUT_MONITOR_OFFSET,
    pml_width=PML_WIDTH_GRIDPOINTS,
)

# Waveguide bend with a 1.6 x 1.6 um design region.
WAVEGUIDE_BEND_SPEC = waveguide_bend.spec.WaveguideBendSpec(
    wg_width=WG_WIDTH,
    wg_length=WG_LENGTH,
    wg_mode_padding=WG_MODE_PADDING,
    padding=PADDING,
    port_pml_offset=PORT_PML_OFFSET,
    variable_region_size=(1600 * u.nm, 1600 * u.nm),
    cladding_permittivity=CLADDING_PERMITTIVITY,
    slab_permittivity=SLAB_PERMITTIVITY,
    input_monitor_offset=INPUT_MONITOR_OFFSET,
    pml_width=PML_WIDTH_GRIDPOINTS,
)


def make_wdm_spec(
    design_extent_ij: u.Array,
    intended_sim_resolution: u.Quantity,
) -> wdm.spec.WdmSpec:
    """Construct a `wdm.spec.WdmSpec` with the specified design region extent.

    Since the `wdm.spec.WdmSpec` does not automatically compensate the simulation
    extent to account for changes in the PML region thickness with changes in the
    simulation resolution, the intended simulation resolution must be specified.
    The actual simulation extent appropriate for that resolution (i.e. accounting
    for the thickness of PML layers) is then computed.

    Note that if the actual simulation resolution is coarser than the intended
    resolution specified here, the PML may overlap with the source and monitors.
    Care should be taken to ensure the correct resolution is specified here.

    Args:
        design_extent_ij: Specifies the size of the design region.
        intended_sim_resolution: Specifies the simulation resolution to be used.

    Returns:
        The `wdm.spec.WdmSpec`.
    """

    design_extent_i, design_extent_j = design_extent_ij
    design_extent_i_nm: int = u.resolve(design_extent_i, 1 * u.nm)
    design_extent_j_nm: int = u.resolve(design_extent_j, 1 * u.nm)

    wg_width_nm: int = u.resolve(WG_WIDTH, 1 * u.nm)

    pad_width_nm: int = u.resolve(PADDING, 1 * u.nm)
    pml_width_nm: int = u.resolve(
        PML_WIDTH_GRIDPOINTS * intended_sim_resolution, 1 * u.nm
    )

    extent_i_nm: int = design_extent_i_nm + 2 * pad_width_nm + 2 * pml_width_nm
    extent_j_nm: int = design_extent_j_nm + 2 * pad_width_nm + 2 * pml_width_nm

    return wdm.spec.WdmSpec(
        extent_ij=u.Array([extent_i_nm, extent_j_nm], u.nm),
        input_wg_j=extent_j_nm / 2 * u.nm,
        output_wgs_j=u.Array(
            (
                extent_j_nm / 2 - design_extent_j_nm / 2 + wg_width_nm / 2 + 320,
                extent_j_nm / 2 + design_extent_j_nm / 2 - wg_width_nm / 2 - 320,
            ),
            u.nm,
        ),
        wg_width=WG_WIDTH,
        wg_mode_padding=WG_MODE_PADDING,
        input_mode_i=pml_width_nm * u.nm + PORT_PML_OFFSET,
        output_mode_i=(extent_i_nm - pml_width_nm) * u.nm - PORT_PML_OFFSET,
        variable_region=(
            u.Array(
                (
                    extent_i_nm / 2 - design_extent_i_nm / 2,
                    extent_j_nm / 2 - design_extent_j_nm / 2,
                ),
                u.nm,
            ),
            u.Array(
                (
                    extent_i_nm / 2 + design_extent_i_nm / 2,
                    extent_j_nm / 2 + design_extent_j_nm / 2,
                ),
                u.nm,
            ),
        ),
        cladding_permittivity=CLADDING_PERMITTIVITY,
        slab_permittivity=SLAB_PERMITTIVITY,
        input_monitor_offset=INPUT_MONITOR_OFFSET,
        pml_width=PML_WIDTH_GRIDPOINTS,
    )


# Wavelength demultiplexer with 3.2 x 3.2 um design region.
LIGHTWEIGHT_WDM_SPEC = make_wdm_spec(
    design_extent_ij=u.Array([3200, 3200], u.nm),
    intended_sim_resolution=LIGHTWEIGHT_SIM_PARAMS.resolution,
)

# Wavelength demultiplexer with 6.4 x 6.4 um design region.
WDM_SPEC = make_wdm_spec(
    design_extent_ij=u.Array([6400, 6400], u.nm),
    intended_sim_resolution=SIM_PARAMS.resolution,
)
