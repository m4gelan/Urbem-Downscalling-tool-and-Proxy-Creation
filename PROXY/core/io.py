from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import rasterio
from rasterio.transform import Affine

if TYPE_CHECKING:
    import geopandas as gpd


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
    return path


def write_geotiff(
    *,
    path: Path,
    array: np.ndarray,
    crs: str,
    transform: Affine,
    nodata: float | int | None = None,
    tags: dict[str, str] | None = None,
    band_descriptions: list[str] | None = None,
    tiled: bool = False,
    predictor: int | None = None,
    bigtiff: str | None = None,
) -> Path:
    ensure_parent(path)
    if array.ndim == 2:
        data = array[np.newaxis, ...]
    elif array.ndim == 3:
        data = array
    else:
        raise ValueError("GeoTIFF write expects a 2D or 3D array")

    count, height, width = data.shape
    profile: dict[str, Any] = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": count,
        "dtype": str(data.dtype),
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
        "compress": "deflate",
    }
    if tiled:
        profile["tiled"] = True
    if predictor is not None:
        profile["predictor"] = int(predictor)
    if bigtiff:
        profile["BIGTIFF"] = str(bigtiff)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
        if band_descriptions:
            for i, desc in enumerate(band_descriptions, start=1):
                if i <= count:
                    dst.set_band_description(i, str(desc)[:256])
        if tags:
            dst.update_tags(**tags)
    return path


def rasterize_geodataframe_values(
    gdf: "gpd.GeoDataFrame",
    *,
    value_column: str,
    ref_profile: dict[str, Any],
    dtype: np.dtype = np.dtype(np.float32),
    merge_add: bool = True,
    show_progress: bool = False,
    progress_desc: str = "rasterize",
) -> np.ndarray:
    """Project geometries to the reference CRS and burn ``value_column`` into a raster."""
    from rasterio import features
    from rasterio.crs import CRS
    from rasterio.enums import MergeAlg
    from shapely.geometry import mapping

    h = int(ref_profile["height"])
    w = int(ref_profile["width"])
    transform = ref_profile["transform"]
    crs = CRS.from_string(str(ref_profile["crs"]))
    acc = np.zeros((h, w), dtype=dtype)
    if gdf.empty:
        return acc

    g_proj = gdf.to_crs(crs)
    shapes: list[tuple[object, float]] = []
    row_it = g_proj.iterrows()
    if show_progress:
        try:
            from tqdm import tqdm

            row_it = tqdm(
                row_it,
                total=int(len(g_proj)),
                desc=progress_desc,
                unit="geom",
                file=sys.stderr,
                disable=not sys.stderr.isatty(),
                mininterval=0.5,
            )
        except ImportError:
            pass

    for _, row in row_it:
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        val = float(row[value_column])
        if val <= 0:
            continue
        shapes.append((mapping(geom), val))

    if shapes:
        features.rasterize(
            shapes,
            out=acc,
            transform=transform,
            merge_alg=MergeAlg.add if merge_add else MergeAlg.replace,
        )
    return acc

