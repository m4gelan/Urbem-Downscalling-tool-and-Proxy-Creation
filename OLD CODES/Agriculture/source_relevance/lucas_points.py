"""
Shared LUCAS + CORINE point prep only: load survey rows, filter by run.country, sample CLC at each point.

Pathway-specific relevance lives in source_relevance/*.py modules. Result is cached on cfg for one read
+ one raster sample per pipeline run.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as warp_transform

from ..core.io import resolve_path
from ..core.run_countries import parse_run_country_codes

_CACHE_KEY = "_lucas_ag_points_prepared"


def sample_corine_clc(
    lons: np.ndarray,
    lats: np.ndarray,
    raster_path: Path,
    nodata: float,
) -> np.ndarray:
    """Sample CORINE at WGS84 lon/lat; nodata -> NaN."""
    with rasterio.open(raster_path) as ds:
        xs, ys = warp_transform("EPSG:4326", ds.crs, lons.tolist(), lats.tolist())
        coords = list(zip(xs, ys))
        vals = np.array([row[0] for row in ds.sample(coords)], dtype=float)
    vals = np.where(np.isclose(vals, nodata), np.nan, vals)
    return vals


def get_lucas_ag_points(cfg: dict[str, Any], root: Path) -> pd.DataFrame:
    """
    LUCAS rows with valid coords, NUTS_ID, COUNTRY, CLC_CODE from CORINE (ag classes 12–22 only).
    """
    hit = cfg.get(_CACHE_KEY)
    if isinstance(hit, pd.DataFrame):
        return hit

    lb = cfg.get("lucas_build") or {}
    paths = cfg.get("paths") or {}
    inputs = paths.get("inputs") or {}
    run = cfg.get("run") or {}
    nodata = float(run.get("nodata", -128.0))
    country_codes = parse_run_country_codes(run)

    lucas_rel = lb.get("lucas_data") or lb.get("lucas_xlsx")
    if not lucas_rel:
        lucas_rel = "data/Agriculture/EU_LUCAS_2022.csv"
    lucas_path = resolve_path(root, lucas_rel)

    corine_rel = lb.get("corine_raster") or inputs.get("corine_raster", "Input/CORINE/U2018_CLC2018_V2020_20u1.tif")
    corine_path = resolve_path(root, corine_rel)

    if not lucas_path.is_file():
        raise FileNotFoundError(f"LUCAS data not found: {lucas_path}")
    if not corine_path.is_file():
        raise FileNotFoundError(f"CORINE raster not found: {corine_path}")

    size = lucas_path.stat().st_size
    if size == 0:
        raise ValueError(
            f"LUCAS CSV is empty (0 bytes): {lucas_path}\n"
            "Replace the file or set lucas_build.lucas_data in agriculture.config.json "
            "to the path of a valid EU LUCAS export (CSV with a header row)."
        )

    usecols_base = [
        "POINT_NUTS0",
        "POINT_NUTS2",
        "POINT_LAT",
        "POINT_LONG",
        "SURVEY_LC1",
        "SURVEY_LU1",
        "SURVEY_GRAZING",
        "SURVEY_INSPIRE_ARABLE",
        "SURVEY_INSPIRE_ARTIF",
        "SURVEY_INSPIRE_ORGCON",
        "SURVEY_INSPIRE_PLCC7",
        "SURVEY_INSPIRE_PLCC4",
        "SURVEY_LM_CROP_RESID",
        "SURVEY_LM_CROP_RESID_PERC",
        "SURVEY_WM",
        "SURVEY_INSPIRE_WETCON",
    ]
    # Optional: biomass burning (NFR 4.F) residue weights — ignore if absent from CSV
    usecols_optional = [
        "SURVEY_LC1_PERC",
        "SURVEY_LC1_SPEC",
        "SURVEY_LC2",
        "SURVEY_LC2_PERC",
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        try:
            header = pd.read_csv(lucas_path, nrows=0).columns.tolist()
        except pd.errors.EmptyDataError as exc:
            raise ValueError(
                f"LUCAS CSV has no parseable header or is not a valid CSV: {lucas_path}\n"
                "If the file is empty, incomplete, or a placeholder, replace it with the full "
                "EU LUCAS survey export (or point lucas_build.lucas_data to the correct path)."
            ) from exc
        usecols = [c for c in usecols_base + usecols_optional if c in header]
        raw = pd.read_csv(lucas_path, usecols=usecols, low_memory=False)

    raw["NUTS_ID"] = raw["POINT_NUTS2"].astype(str).str.strip()
    raw["COUNTRY"] = raw["POINT_NUTS0"].astype(str).str.strip().str.upper()
    if country_codes:
        raw = raw[raw["COUNTRY"].isin(country_codes)].copy()
        if raw.empty:
            raise ValueError(f"No LUCAS rows for country in {sorted(country_codes)!r}.")

    lat = pd.to_numeric(raw["POINT_LAT"], errors="coerce")
    lon = pd.to_numeric(raw["POINT_LONG"], errors="coerce")
    ok = lat.notna() & lon.notna()
    raw = raw.loc[ok].copy()
    lat = lat.loc[ok].to_numpy(dtype=float)
    lon = lon.loc[ok].to_numpy(dtype=float)

    clc = sample_corine_clc(lon, lat, corine_path, nodata=nodata)
    raw["CLC_CODE"] = np.asarray(clc, dtype=float)
    raw = raw[np.isfinite(raw["CLC_CODE"])].copy()
    raw["CLC_CODE"] = raw["CLC_CODE"].astype(int)

    ag = tuple(range(12, 23))
    raw = raw[raw["CLC_CODE"].isin(ag)].copy()
    if raw.empty:
        raise ValueError("No LUCAS points on agricultural CLC (12–22) after CORINE sampling.")

    # TODO: implement ESDAC topsoil pH raster sampling (500 m) and populate ESDAC_PH_TOPSOIL.
    if "ESDAC_PH_TOPSOIL" not in raw.columns:
        raw["ESDAC_PH_TOPSOIL"] = np.nan

    cfg[_CACHE_KEY] = raw
    return raw
