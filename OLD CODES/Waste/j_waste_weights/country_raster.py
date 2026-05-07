"""
Rasterize NUTS country units onto the fine reference grid for CEIP weight lookup.

Polygons are drawn in **ascending area order** so that at administrative overlaps the
**larger** national polygon wins (common ``rasterize`` tie-break). This approximates
dominant coverage without per-pixel intersection tests (see README).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features

logger = logging.getLogger(__name__)

# NUTS CNTR_CODE is ISO 3166-1 alpha-2 (with EL for Greece). CAMS / CEIP use ISO3.
_CNTR2_TO_ISO3: dict[str, str] = {
    "EL": "GRC",
    "GR": "GRC",
    "UK": "GBR",
    "GB": "GBR",
    "AT": "AUT",
    "BE": "BEL",
    "BG": "BGR",
    "HR": "HRV",
    "CY": "CYP",
    "CZ": "CZE",
    "DK": "DNK",
    "EE": "EST",
    "FI": "FIN",
    "FR": "FRA",
    "DE": "DEU",
    "HU": "HUN",
    "IS": "ISL",
    "IE": "IRL",
    "IT": "ITA",
    "LV": "LVA",
    "LT": "LTU",
    "LU": "LUX",
    "MT": "MLT",
    "NL": "NLD",
    "NO": "NOR",
    "PL": "POL",
    "PT": "PRT",
    "RO": "ROU",
    "RS": "SRB",
    "SK": "SVK",
    "SI": "SVN",
    "ES": "ESP",
    "SE": "SWE",
    "CH": "CHE",
    "AL": "ALB",
    "BA": "BIH",
    "ME": "MNE",
    "MK": "MKD",
    "XK": "XKX",
    "UA": "UKR",
    "MD": "MDA",
    "TR": "TUR",
    "LI": "LIE",
}


def cntr_code_to_iso3(code: str) -> str | None:
    c = str(code).strip().upper()
    if not c:
        return None
    if len(c) == 3 and c.isalpha():
        return c
    return _CNTR2_TO_ISO3.get(c)


def load_nuts_countries_union(gpkg_path: Path) -> gpd.GeoDataFrame:
    """
    One multipolygon per country (CNTR_CODE), EPSG:3035 expected.
    Prefer ``LEVL_CODE == 0``; else dissolve NUTS-2 by ``CNTR_CODE``.
    """
    nuts = gpd.read_file(gpkg_path)
    if nuts.crs is None:
        raise ValueError(f"NUTS GeoPackage has no CRS: {gpkg_path}")
    if "CNTR_CODE" not in nuts.columns or "LEVL_CODE" not in nuts.columns:
        raise ValueError("NUTS GeoPackage must contain CNTR_CODE and LEVL_CODE.")
    n0 = nuts[nuts["LEVL_CODE"] == 0].copy()
    if n0.empty:
        n2 = nuts[nuts["LEVL_CODE"] == 2].copy()
        if n2.empty:
            raise ValueError("No NUTS level 0 or 2 features found.")
        n0 = n2.dissolve(by="CNTR_CODE", as_index=False)
    out = n0[["CNTR_CODE", "geometry"]].copy()
    out["iso3"] = out["CNTR_CODE"].map(cntr_code_to_iso3)
    miss = out["iso3"].isna()
    if miss.any():
        bad = out.loc[miss, "CNTR_CODE"].unique().tolist()
        logger.warning("Unknown CNTR_CODE values (dropped): %s", bad)
        out = out.loc[~miss].copy()
    out["area_m2"] = out.geometry.area
    out = out.sort_values("area_m2", ascending=True).reset_index(drop=True)
    return out


def rasterize_country_ids(
    nuts_gpkg: Path,
    ref: dict[str, Any],
) -> tuple[np.ndarray, list[str]]:
    """
    Return ``(country_id, iso3_for_index)``.

    - ``country_id``: uint16 (H,W), 0 = outside all countries / nodata
    - ``iso3_for_index[k]`` = ISO3 for pixel value ``k``; index 0 is ``""``
    """
    gdf = load_nuts_countries_union(nuts_gpkg)
    if gdf.crs is None:
        raise ValueError("Country geometries have no CRS.")
    crs = rasterio.crs.CRS.from_string(ref["crs"])
    g3035 = gdf.to_crs(crs)
    h, w = int(ref["height"]), int(ref["width"])
    transform = ref["transform"]
    iso3_for_index = [""]
    shapes: list[tuple[Any, int]] = []
    kid = 1
    for _, row in g3035.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        iso = str(row["iso3"]).strip().upper()
        iso3_for_index.append(iso)
        shapes.append((geom, kid))
        kid += 1
    if not shapes:
        raise ValueError("No country geometries to rasterize.")
    out = features.rasterize(
        shapes,
        out_shape=(h, w),
        transform=transform,
        fill=0,
        dtype=np.uint16,
        all_touched=False,
    )
    n_pix = int(np.count_nonzero(out > 0))
    logger.info(
        "Country raster: %d countries, %d pixels with ISO3 (%.1f%% of grid)",
        len(shapes),
        n_pix,
        100.0 * n_pix / max(h * w, 1),
    )
    return out, iso3_for_index
