from __future__ import annotations

from pathlib import Path

import geopandas as gpd


def read_vector(path: Path, layer: str | None = None) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
    if gdf.crs is None:
        raise ValueError(f"Vector dataset has no CRS: {path}")
    return gdf


def read_country_nuts2(nuts_gpkg: Path, country_code: str) -> gpd.GeoDataFrame:
    gdf = read_vector(nuts_gpkg)
    if "LEVL_CODE" not in gdf.columns or "CNTR_CODE" not in gdf.columns:
        raise ValueError("NUTS file missing LEVL_CODE and/or CNTR_CODE")
    nuts2 = gdf[gdf["LEVL_CODE"] == 2].copy()
    cc = nuts2["CNTR_CODE"].astype(str).str.upper().str.strip()
    nuts2 = nuts2[cc == country_code.strip().upper()].copy()
    if nuts2.empty:
        raise ValueError(f"No NUTS2 rows for country code {country_code}")
    return nuts2

