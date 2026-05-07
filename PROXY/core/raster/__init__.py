"""Raster alignment, multiband writing, normalization and country-clip helpers.

Introduced in Phase 0 alongside the rest of the shared-core scaffolding and expanded in
Phase 3.2 when ``B_Industry`` / ``D_Fugitive`` pipelines were rewired from the external
``Waste.j_waste_weights`` tree to in-tree helpers.
"""
from __future__ import annotations

from .align import NoDataPolicy, ref_profile_to_kwargs, warp_raster_to_ref, warp_sum_to_ref, warp_to_ref
from .country_clip import (
    cntr_code_to_iso3,
    load_nuts_countries_union,
    rasterize_country_ids,
)
from .normalize import (
    normalize_indicator_quantile_minmax,
    normalize_stack_within_cells,
    normalize_within_bincount_cells,
    normalize_within_cams_cells,
    safe_divide,
    sum_to_one,
    validate_non_negative,
    validate_parent_weight_sums_strict,
    validate_weight_sums,
)
from .write import write_multiband_geotiff

__all__ = [
    "NoDataPolicy",
    "cntr_code_to_iso3",
    "load_nuts_countries_union",
    "normalize_indicator_quantile_minmax",
    "normalize_stack_within_cells",
    "normalize_within_bincount_cells",
    "normalize_within_cams_cells",
    "rasterize_country_ids",
    "ref_profile_to_kwargs",
    "safe_divide",
    "sum_to_one",
    "validate_non_negative",
    "validate_parent_weight_sums_strict",
    "validate_weight_sums",
    "warp_raster_to_ref",
    "warp_sum_to_ref",
    "warp_to_ref",
    "write_multiband_geotiff",
]
