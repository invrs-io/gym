"""Defines the photon extractor challenge."""

from fmmax import basis, fmm  # type: ignore[import]

from invrs_gym.challenge.extractor import component

# https://opg.optica.org/optica/fulltext.cfm?uri=optica-7-12-1805
EXTRACTOR_SPEC = component.ExtractorSpec(
    permittivity_ambient=(1.0 + 0.0j) ** 2,
    permittivity_extractor=(3.31 + 0.0j) ** 2,
    permittivity_substrate=(2.4102 + 0.0j) ** 2,
    thickness_ambient=1.0,
    thickness_extractor=0.25,
    thickness_substrate_before_source=0.1,
    thickness_substrate_after_source=0.9,
    design_region_length=1.5,
    period_x=3.5,
    period_y=3.5,
    pml_thickness=0.5,
    source_monitor_offset=0.025,
    output_monitor_offset=0.4,
)

EXTRACTOR_SIM_PARAMS = component.ExtractorSimParams(
    grid_shape=(350, 350),
    layer_znum=(100, 25, 10, 90),
    wavelength=0.637,
    formulation=fmm.Formulation.JONES_DIRECT,
    approximate_num_terms=800,
    truncation=basis.Truncation.CIRCULAR,
)
