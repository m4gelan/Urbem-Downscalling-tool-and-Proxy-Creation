"""Rasterise OSM farmyard polygons onto the reference grid."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import yaml
from rasterio import features
from rasterio.enums import MergeAlg
from shapely.geometry import mapping

from PROXY.core.dataloaders import resolve_path


def _osm_yaml(root: Path) -> dict[str, Any]:
    p = root / "PROXY" / "config" / "agriculture" / "osm_agriculture_layers.yaml"
    if not p.is_file():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def rasterize_osm_farmyard(
    root: Path,
    ref: dict[str, Any],
    cfg: dict[str, Any],
) -> np.ndarray:
    """Binary (0/1) coverage from agricultural OSM GeoPackage."""
    h, w = int(ref["height"]), int(ref["width"])
    transform = ref["transform"]
    crs = ref["crs"]
    out = np.zeros((h, w), dtype=np.float32)
    inputs = (cfg.get("paths") or {}).get("inputs") or {}
    gpkg_rel = inputs.get("agriculture_osm_gpkg")
    if not gpkg_rel:
        return out
    gpkg = resolve_path(root, Path(str(gpkg_rel)))
    if not gpkg.is_file():
        return out
    ydoc = _osm_yaml(root)
    rast = (ydoc.get("rasterization") or {})
    burn = float(rast.get("burn_value", 1.0))
    fill = float(rast.get("fill", 0.0))
    all_touched = bool(rast.get("all_touched", False))
    merge_alg_name = str(rast.get("merge_alg", "replace")).lower()
    merge_alg = MergeAlg.add if merge_alg_name == "add" else MergeAlg.replace

    layers_cfg = (ydoc.get("layers") or {})
    primary = (layers_cfg.get("agricultural_farmyard") or {}).get("layer_name", "agricultural_farmyard")
    shapes: list[tuple[Any, float]] = []
    for layer in (primary, (layers_cfg.get("agricultural_points_buffered") or {}).get("layer_name")):
        if not layer:
            continue
        try:
            gdf = gpd.read_file(gpkg, layer=str(layer))
        except Exception:
            continue
        if gdf.empty:
            continue
        if gdf.crs is None:
            continue
        gdf = gdf.to_crs(crs)
        for geom in gdf.geometry:
            if geom is None or geom.is_empty:
                continue
            shapes.append((mapping(geom), burn))

    if not shapes:
        return out

    arr = features.rasterize(
        shapes,
        out_shape=(h, w),
        transform=transform,
        fill=fill,
        dtype=np.float32,
        merge_alg=merge_alg,
        all_touched=all_touched,
    )
    return np.asarray(arr, dtype=np.float32)
