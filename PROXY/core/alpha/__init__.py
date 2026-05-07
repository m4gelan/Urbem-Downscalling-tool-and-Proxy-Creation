"""Alpha / CEIP / offroad-share package.

Re-exports the same public names that used to live in the monolithic
``PROXY.core.alpha`` module, so existing imports continue to work unchanged:

.. code-block:: python

    from PROXY.core.alpha import (
        AlphaRequest,
        compute_alpha,
        norm_pollutant_key,
        read_ceip_shares,
        build_share_arrays,
        lookup_offroad_triple_for_iso3,
        _read_alpha_workbook,  # intentionally preserved for existing sector code
    )

New fallback resolver (opt-in, additive):

.. code-block:: python

    from PROXY.core.alpha.fallback import AlphaSource, resolve_alpha
"""
from __future__ import annotations

from .aliases import norm_pollutant_key
from .ceip import read_ceip_shares
from .compute import AlphaRequest, compute_alpha
from .fallback import AlphaResolution, AlphaSource, format_provenance, resolve_alpha
from .matrix import finalize_alpha_matrix
from .offroad import (
    apply_offroad_yaml_overrides,
    build_share_arrays,
    load_offroad_mass_fractions_from_alpha_csv,
    lookup_offroad_triple_for_iso3,
    resolve_offroad_triple_with_yaml,
    resolve_subsector_emission_masses,
)
from .reported import (
    normalize_inventory_sector,
    normalize_sector_token,
    parse_float_or_nan,
    resolve_iso3_reported,
    short_country,
    to_iso3,
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
    "apply_offroad_yaml_overrides",
    "build_share_arrays",
    "compute_alpha",
    "format_provenance",
    "finalize_alpha_matrix",
    "load_offroad_mass_fractions_from_alpha_csv",
    "lookup_offroad_triple_for_iso3",
    "norm_pollutant_key",
    "read_alpha_workbook",
    "read_ceip_shares",
    "resolve_offroad_triple_with_yaml",
    "resolve_alpha",
    "normalize_inventory_sector",
    "normalize_sector_token",
    "parse_float_or_nan",
    "resolve_iso3_reported",
    "short_country",
    "to_iso3",
    "resolve_subsector_emission_masses",
    "_read_alpha_workbook",
    "_is_semicolon_single_column_table",
    "_semicolon_rows_to_standard_columns",
]
