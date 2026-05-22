from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features as rio_features
from rasterio.enums import MergeAlg
from shapely.geometry import mapping

from proxy.core import log


def load_nuts2_polygons(
    nuts_path: Path,
    country_profile: dict[str, str],
) -> gpd.GeoDataFrame:
    """NUTS level-2 polygons for the run country (``country_profile['other']`` = NUTS CNTR_CODE)."""
    cntr = str(country_profile["other"]).strip().upper()
    nuts = gpd.read_file(nuts_path)
    if nuts.crs is None:
        raise ValueError(f"NUTS GeoPackage has no CRS: {nuts_path}")
    for col in ("LEVL_CODE", "CNTR_CODE", "NUTS_ID"):
        if col not in nuts.columns:
            raise ValueError(f"NUTS GeoPackage missing column {col!r}")
    n2 = nuts[nuts["LEVL_CODE"].astype(int) == 2].copy()
    cc = n2["CNTR_CODE"].astype(str).str.strip().str.upper()
    n2 = n2[cc == cntr].copy()
    if n2.empty:
        raise ValueError(f"No NUTS2 rows for CNTR_CODE={cntr!r} in {nuts_path}")
    log.info(f"NUTS2: {len(n2)} regions for CNTR_CODE={cntr}")
    log.debug(f"NUTS2 ids sample: {n2['NUTS_ID'].astype(str).head(5).tolist()}")
    return n2


def rasterize_nuts2_ids(
    nuts_gdf: gpd.GeoDataFrame,
    height: int,
    width: int,
    transform: Any,
    raster_crs: Any,
) -> tuple[np.ndarray, dict[str, int]]:
    """Rasterize NUTS2 polygons to 1-based indices; 0 = outside country NUTS2."""
    nuts_to_idx: dict[str, int] = {}
    shapes = []
    for k, (_, row) in enumerate(nuts_gdf.iterrows()):
        nid = str(row["NUTS_ID"]).strip()
        nuts_to_idx[nid] = k + 1
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        g = gpd.GeoDataFrame(geometry=[geom], crs=nuts_gdf.crs).to_crs(raster_crs)
        shapes.append((mapping(g.geometry.iloc[0]), k + 1))

    nuts_r = rio_features.rasterize(
        shapes,
        out_shape=(int(height), int(width)),
        transform=transform,
        fill=0,
        dtype=np.int32,
        merge_alg=MergeAlg.replace,
    )
    log.info(f"NUTS2 raster: {len(nuts_to_idx)} regions, {int((nuts_r > 0).sum())} pixels labelled")
    return nuts_r, nuts_to_idx
