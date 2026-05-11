"""Alpha primitives and CEIP-reported α stack.

CEIP profile indexing, merge helpers, and tensor builders live in this package
(:mod:`PROXY.core.alpha.ceip_index_loader`, :mod:`PROXY.core.alpha.reported_group_alpha`).
For offroad triple-share rasters see :mod:`PROXY.sectors.I_Offroad.share_broadcast`.
"""
from __future__ import annotations

from .aliases import norm_pollutant_key
from .ceip_index_loader import (
    DEFAULT_GNFR_GROUP_ORDER,
    clear_ceip_index_cache,
    default_ceip_profile_relpath,
    load_merged_ceip_profile_for_pipeline_paths,
    remap_legacy_ceip_relpath,
    shared_pollutant_aliases_relpath,
)
from .compute import AlphaRequest, compute_alpha
from .fallback import AlphaResolution, AlphaSource, format_provenance, resolve_alpha
from .matrix import finalize_alpha_matrix
from .reported import (
    normalize_inventory_sector,
    normalize_sector_token,
    parse_float_or_nan,
    resolve_iso3_reported,
    short_country,
    to_iso3,
)
from .reported_group_alpha import (
    load_ceip_and_alpha,
    load_ceip_and_alpha_solvents,
    load_group_mapping,
    load_subsector_mapping_from_yaml,
    read_ceip_long,
    read_reported_emissions_fugitive_long,
    read_reported_emissions_subsector_long,
)
from .workbook import (
    _is_semicolon_single_column_table,
    _read_alpha_workbook,
    _semicolon_rows_to_standard_columns,
    read_alpha_workbook,
)

__all__ = [
    "AlphaRequest",
    "AlphaResolution",
    "AlphaSource",
    "DEFAULT_GNFR_GROUP_ORDER",
    "clear_ceip_index_cache",
    "compute_alpha",
    "default_ceip_profile_relpath",
    "format_provenance",
    "finalize_alpha_matrix",
    "load_ceip_and_alpha",
    "load_ceip_and_alpha_solvents",
    "load_group_mapping",
    "load_merged_ceip_profile_for_pipeline_paths",
    "load_subsector_mapping_from_yaml",
    "norm_pollutant_key",
    "read_alpha_workbook",
    "read_ceip_long",
    "read_reported_emissions_fugitive_long",
    "read_reported_emissions_subsector_long",
    "remap_legacy_ceip_relpath",
    "resolve_alpha",
    "shared_pollutant_aliases_relpath",
    "normalize_inventory_sector",
    "normalize_sector_token",
    "parse_float_or_nan",
    "resolve_iso3_reported",
    "short_country",
    "to_iso3",
    "_read_alpha_workbook",
    "_is_semicolon_single_column_table",
    "_semicolon_rows_to_standard_columns",
]
