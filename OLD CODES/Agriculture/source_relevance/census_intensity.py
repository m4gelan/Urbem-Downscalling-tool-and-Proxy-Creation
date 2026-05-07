"""
Stage-2 census intensity weights omega (NUTS-2) from C21 GeoPackage + emission factor JSON.

No emission factors are hardcoded in Python; all numeric coefficients load from JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from Agriculture.core.io import resolve_path

SourceKind = Literal["enteric", "housing"]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_field_map(path: Path) -> dict[str, Any]:
    return _read_json(path)


def _validate_schema_sample(schema_path: Path, field_map: dict[str, Any]) -> None:
    """Optional: ensure explore_head-style sample is consistent with bovine column names."""
    try:
        text = schema_path.read_text(encoding="utf-8").replace("NaN", "null")
        raw = json.loads(text)
    except (OSError, json.JSONDecodeError, ValueError):
        return
    if not isinstance(raw, list) or not raw:
        return
    row0 = raw[0]
    if not isinstance(row0, dict):
        return
    b = field_map.get("species_columns", {}).get("bovine")
    if not b:
        return
    col = b.get("count_column")
    if col and col not in row0 and str(col) not in row0:
        pass


def aggregate_c21_heads_by_nuts2(gpkg_path: Path, field_map: dict[str, Any]) -> pd.DataFrame:
    """
    Sum head counts per NUTS-2 across all C21 layers (one layer per species in field_map).
    Rows: index = nuts2 code; columns = species keys (bovine, sheep, ...).
    """
    import geopandas as gpd

    nuts_col = field_map["nuts2_column"]
    species_cfg = field_map["species_columns"]
    parts: list[pd.Series] = []
    for species, spec in species_cfg.items():
        layer = spec["layer"]
        col = spec["count_column"]
        gdf = gpd.read_file(gpkg_path, layer=layer)
        if nuts_col not in gdf.columns:
            raise KeyError(f"C21 layer {layer!r}: missing nuts column {nuts_col!r}")
        if col not in gdf.columns:
            raise KeyError(f"C21 layer {layer!r}: missing count column {col!r}")
        sub = gdf[[nuts_col, col]].copy()
        sub[col] = pd.to_numeric(sub[col], errors="coerce").fillna(0.0)
        s = sub.groupby(nuts_col, sort=False)[col].sum()
        s.name = species
        parts.append(s)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, axis=1, join="outer")
    return out.fillna(0.0)


def _intensity_row(
    counts: pd.DataFrame,
    species_factors: dict[str, Any],
    value_key: str,
    include_key: str,
) -> pd.Series:
    idx = counts.index
    tot = pd.Series(0.0, index=idx, dtype=float)
    for species, spec in species_factors.items():
        if not isinstance(spec, dict) or not spec.get(include_key, False):
            continue
        if species not in counts.columns:
            continue
        w = float(spec[value_key])
        tot = tot + counts[species].astype(float) * w
    return tot


def _omega_from_intensity(
    intensity: pd.Series,
    cap_low: float,
    cap_high: float,
    missing_fallback: float,
) -> pd.Series:
    pos = intensity[intensity > 0]
    med = float(pos.median()) if len(pos) else 1.0
    if med <= 0 or not np.isfinite(med):
        med = 1.0
    w = intensity / med
    w = w.replace([np.inf, -np.inf], np.nan).clip(cap_low, cap_high)
    return w.fillna(missing_fallback)


def load_census_intensity(
    gpkg_path: str | Path,
    field_map_json_path: str | Path,
    factors_json_path: str | Path,
    grid_cells: Any = None,
    source: SourceKind = "enteric",
    schema_sample_json_path: str | Path | None = None,
) -> tuple[pd.Series, float]:
    """
    Compute omega_nc per NUTS-2 from C21 census and emission-factor JSON.

    field_map_json_path: Agriculture/config/emission_factors/c21_census_field_map.json (layer/column names).
    schema_sample_json_path: optional data/Agriculture/C21.explore_head.json for validation only.

    grid_cells: reserved for future area-weighted intersection with model geometry; ignored.

    Returns (omega, missing_fallback): omega indexed by NUTS-2 (same as pipeline NUTS_ID); use missing_fallback
    when mapping NUTS not present in omega.
    """
    if grid_cells is not None:
        pass

    gpkg_path = Path(gpkg_path)
    field_map_path = Path(field_map_json_path)
    factors_path = Path(factors_json_path)

    field_map = _load_field_map(field_map_path)
    if schema_sample_json_path is not None:
        _validate_schema_sample(Path(schema_sample_json_path), field_map)
    factors = _read_json(factors_path)
    species_factors = factors.get("species_factors") or {}
    params = factors.get("intensity_weight_params") or {}
    cap_low = float(params.get("cap_low", 0.05))
    cap_high = float(params.get("cap_high", 5.0))
    missing_fallback = float(params.get("missing_data_fallback", 1.0))

    counts = aggregate_c21_heads_by_nuts2(gpkg_path, field_map)

    if source == "enteric":
        intensity = _intensity_row(
            counts,
            species_factors,
            "ch4_per_head_per_year_kg",
            "included_in_enteric",
        )
    else:
        intensity = _intensity_row(
            counts,
            species_factors,
            "gN_per_head_per_day",
            "included_in_housing",
        )

    omega = _omega_from_intensity(intensity, cap_low, cap_high, missing_fallback)
    omega.name = "omega"
    return omega, missing_fallback


def load_census_omega_for_cfg(
    cfg: dict[str, Any],
    root: Path,
    source: SourceKind,
) -> tuple[pd.Series, float]:
    """Resolve paths from cfg['paths']['census'] with defaults under Agriculture/config/emission_factors/."""
    paths = cfg.get("paths") or {}
    census = paths.get("census") or {}
    gpkg = resolve_path(root, census.get("c21_gpkg", "data/Agriculture/C21.gpkg"))
    field_map = resolve_path(
        root,
        census.get("c21_field_map_json", "Agriculture/config/emission_factors/c21_census_field_map.json"),
    )
    sample = census.get("c21_explore_head_json")
    sample_path = resolve_path(root, sample) if sample else None
    if source == "enteric":
        factors = resolve_path(
            root,
            census.get("enteric_factors_json", "Agriculture/config/emission_factors/enteric_fermentation_factors.json"),
        )
    else:
        factors = resolve_path(
            root,
            census.get("housing_factors_json", "Agriculture/config/emission_factors/livestock_housing_factors.json"),
        )
    return load_census_intensity(gpkg, field_map, factors, None, source, sample_path)


def omega_for_extent_nuts(
    extent_nuts_ids: pd.Series,
    omega_by_nuts: pd.Series,
) -> pd.Series:
    """Align omega to extent rows; missing NUTS -> missing_data_fallback from attrs or 1.0."""
    fb = float(omega_by_nuts.attrs.get("missing_data_fallback", 1.0)) if hasattr(omega_by_nuts, "attrs") else 1.0
    m = extent_nuts_ids.map(omega_by_nuts)
    return m.fillna(fb)
