from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

from PROXY_V2.core import log


def load_c21_headcounts(
    farmstock_path: Path,
    farmstock_cfg: dict[str, Any],
) -> pd.DataFrame:
    """
    Sum livestock head counts per NUTS2 across C21 layers.

    Returns a DataFrame indexed by NUTS2 code; columns = species keys from config.
    """
    nuts_col = str(farmstock_cfg.get("nuts2_column"))
    if not nuts_col:
        raise ValueError("Farmstock.nuts2_column required in sector config")
    species_cfg = farmstock_cfg.get("species_columns")
    if not isinstance(species_cfg, dict) or not species_cfg:
        raise ValueError("Farmstock.species_columns must be a non-empty mapping")

    parts: list[pd.Series] = []
    for species, spec in species_cfg.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Farmstock.species_columns.{species} must be a mapping")
        layer = str(spec.get("layer"))
        col = str(spec.get("count_column"))
        gdf = gpd.read_file(farmstock_path, layer=layer)
        if nuts_col not in gdf.columns:
            raise KeyError(f"C21 layer {layer!r}: missing column {nuts_col!r}")
        if col not in gdf.columns:
            raise KeyError(f"C21 layer {layer!r}: missing column {col!r}")
        sub = gdf[[nuts_col, col]].copy()
        sub[nuts_col] = sub[nuts_col].astype(str).str.strip().str.upper()
        sub[col] = pd.to_numeric(sub[col], errors="coerce").fillna(0.0)
        s = sub.groupby(nuts_col, sort=False)[col].sum()
        s.name = str(species).strip()
        parts.append(s)
        log.debug(f"C21 {species}: layer={layer} total_heads={float(s.sum()):.6g}")

    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, axis=1, join="outer").fillna(0.0)
    out.index = out.index.astype(str).str.strip().str.upper()
    log.info(f"C21 headcounts: {len(out)} NUTS2 rows, species={list(out.columns)}")
    return out
