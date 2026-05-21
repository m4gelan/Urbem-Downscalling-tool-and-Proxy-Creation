"""Family 7: GFED agricultural fire DM warped to reference grid (no LUCAS residue modulation)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject

from PROXY.core.dataloaders import resolve_path
from PROXY.sectors.K_Agriculture.combined_config import load_k_agriculture_rules


def _gfed_meta(cfg: dict[str, Any], root: Path) -> dict[str, Any]:
    doc = load_k_agriculture_rules(cfg, root)
    return doc.get("gfed") or {}


def build_family7(cfg: dict[str, Any], root: Path, ref: dict[str, Any]) -> np.ndarray:
    h, w = int(ref["height"]), int(ref["width"])
    lb = cfg.get("lucas_build") or {}
    dm_path = resolve_path(root, Path(str(lb.get("gfed41s_agri_dm_mean_npy", ""))))
    area_path = resolve_path(root, Path(str(lb.get("gfed41s_grid_area_npy", ""))))
    if not dm_path.is_file() or not area_path.is_file():
        return np.zeros((h, w), dtype=np.float32)

    dm_mean = np.load(dm_path).astype(np.float64)
    area_m2 = np.load(area_path).astype(np.float64)
    src = (dm_mean * area_m2).astype(np.float32)
    meta = _gfed_meta(cfg, root)
    lon0 = float(meta.get("lon_min", -180.0))
    lon1 = float(meta.get("lon_max", 180.0))
    la0 = float(meta.get("lat_min", -90.0))
    la1 = float(meta.get("lat_max", 90.0))
    sw = int(meta.get("width", src.shape[1] if src.ndim == 2 else 1440))
    sh = int(meta.get("height", src.shape[0] if src.ndim == 2 else 720))
    if src.ndim != 2:
        return np.zeros((h, w), dtype=np.float32)
    src_tf = from_bounds(lon0, la0, lon1, la1, sw, sh)
    dst = np.zeros((h, w), dtype=np.float32)
    reproject(
        source=src,
        destination=dst,
        src_transform=src_tf,
        src_crs="EPSG:4326",
        dst_transform=ref["transform"],
        dst_crs=ref["crs"],
        resampling=Resampling.sum,
    )
    dst = np.maximum(dst, 0.0)
    mx = float(dst.max()) + 1e-12
    return (dst / mx).astype(np.float32)
