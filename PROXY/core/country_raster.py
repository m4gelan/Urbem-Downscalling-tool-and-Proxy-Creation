"""NUTS-based country index raster (reusable for national share lookup)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from rasterio.enums import MergeAlg
from shapely.geometry import mapping


def rasterize_country_indices(
    nuts_gpkg: Path,
    ref: dict[str, Any],
    cntr_to_iso3: dict[str, str],
    fallback_iso3: str,
) -> tuple[np.ndarray, dict[int, str]]:
    """Integer raster 0..N; mapping idx -> ISO3 (0 = unknown → fallback)."""
    nuts = gpd.read_file(nuts_gpkg)
    n0 = nuts[nuts["LEVL_CODE"] == 0].copy()
    if n0.empty:
        raise ValueError("No NUTS LEVL_CODE==0 rows for country raster.")
    cc = n0["CNTR_CODE"].astype(str).str.strip().str.upper()
    uniq = sorted(cc.unique())
    cntr_to_i = {c: i + 1 for i, c in enumerate(uniq)}
    idx_to_iso: dict[int, str] = {0: fallback_iso3}
    for c in uniq:
        idx = cntr_to_i[c]
        idx_to_iso[idx] = str(cntr_to_iso3.get(c, fallback_iso3)).strip().upper()

    shapes = []
    for _, row in n0.iterrows():
        c = str(row["CNTR_CODE"]).strip().upper()
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        val = cntr_to_i.get(c, 0)
        if val:
            shapes.append((mapping(geom), val))

    h, w = int(ref["height"]), int(ref["width"])
    transform = ref["transform"]
    out = features.rasterize(
        shapes,
        out_shape=(h, w),
        transform=rasterio.transform.Affine(*transform[:6]),
        fill=0,
        dtype=np.int32,
        merge_alg=MergeAlg.replace,
    )
    return out, idx_to_iso
