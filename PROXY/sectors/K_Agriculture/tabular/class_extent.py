"""Class extent n_{n,c}: CORINE pixel counts per NUTS2 x CLC."""
from __future__ import annotations

from typing import Any

import geopandas as gpd
import pandas as pd

from .census_io import resolve_nuts_gpkg
from .run_countries import parse_run_country_codes
from .zonal import AG_CLC_CODES, zonal_histograms


def build_class_extent_long(
    nuts2: gpd.GeoDataFrame,
    corine_path: Any,
    nodata: float,
    ag_clc_codes: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    codes = ag_clc_codes if ag_clc_codes is not None else AG_CLC_CODES
    stats = zonal_histograms(nuts2, corine_path, nodata=nodata)
    rows: list[dict[str, Any]] = []
    for i, (_, row) in enumerate(nuts2.iterrows()):
        hist = stats[i] if stats[i] is not None else {}
        if not isinstance(hist, dict):
            hist = {}
        nid = str(row["NUTS_ID"]).strip()
        name_region = row.get("NAME_LATN", "")
        cntr = str(row["CNTR_CODE"]).strip()
        for c in codes:
            rows.append(
                {
                    "NUTS_ID": nid,
                    "NAME_REGION": name_region,
                    "COUNTRY": cntr,
                    "CLC_CODE": int(c),
                    "n_pixels": int(hist.get(c, 0) or 0),
                }
            )
    return pd.DataFrame(rows)


def load_nuts2_filtered(cfg: dict, root: Any) -> gpd.GeoDataFrame:
    run = cfg.get("run") or {}
    paths = cfg.get("paths") or {}
    geom = paths.get("geometry") or {}
    country_codes = parse_run_country_codes(run)

    gpkg = resolve_nuts_gpkg(root, geom.get("nuts_gpkg", "Data/geometry/NUTS_RG_20M_2021_3035.gpkg"))
    nuts = gpd.read_file(gpkg)
    nuts2 = nuts[nuts["LEVL_CODE"] == 2].copy()
    if nuts2.crs is None:
        raise ValueError("NUTS GeoPackage has no CRS.")
    if "CNTR_CODE" not in nuts2.columns:
        raise ValueError("GeoPackage missing CNTR_CODE.")
    if country_codes:
        cc = nuts2["CNTR_CODE"].astype(str).str.strip().str.upper()
        nuts2 = nuts2[cc.isin(country_codes)].copy()
        if nuts2.empty:
            raise ValueError(f"No NUTS-2 rows for CNTR_CODE in {sorted(country_codes)!r}.")
    return nuts2
