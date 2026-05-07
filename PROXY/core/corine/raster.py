"""CORINE raster resolution, warping, window reads, and class masks.

This module owns CORINE raster IO on the shared reference grid. Inputs are a
configured CORINE path or GeoTIFF plus the standard reference profile; outputs
are nearest-neighbor CORINE code rasters or float32 masks for configured CLC
class groups. Pixel legend decoding lives in `PROXY.core.corine.encoding`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.errors import WindowError
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window, from_bounds

from PROXY.core.dataloaders import resolve_path
from PROXY.core.osm_corine_proxy import adapt_corine_classes_for_grid


def resolve_corine_tif(corine_path: Path) -> Path:
    """Resolve a configured CORINE file or directory to one GeoTIFF."""
    p = corine_path.expanduser().resolve()
    if p.is_file():
        return p
    if p.is_dir():
        tifs = sorted(p.glob("*.tif"))
        if not tifs:
            tifs = sorted(p.rglob("*.tif"))
        if not tifs:
            raise FileNotFoundError(f"No GeoTIFF under {p}")
        return tifs[0]
    raise FileNotFoundError(f"CORINE path not found: {corine_path}")


def _subwin_read_for_bounds(
    corine_tif: Path,
    dst_crs: rasterio.crs.CRS,
    bounds_3035: tuple[float, float, float, float],
) -> tuple[np.ndarray, object, object]:
    left, bottom, right, top = bounds_3035
    with rasterio.open(corine_tif) as src:
        if src.crs is None:
            raise ValueError(f"CORINE has no CRS: {corine_tif}")
        west, south, east, north = transform_bounds(
            dst_crs, src.crs, left, bottom, right, top, densify_pts=21
        )
        win = from_bounds(west, south, east, north, transform=src.transform)
        try:
            win = win.round_lengths().intersection(Window(0, 0, src.width, src.height))
        except WindowError as e:
            raise ValueError(f"CORINE window empty for requested bounds: {e}") from e
        if win.width < 1 or win.height < 1:
            raise ValueError("CORINE window has no pixels over fine-grid bounds.")
        arr = src.read(1, window=win).astype(np.float32)
        nodata = src.nodata
        if nodata is not None and np.isfinite(nodata):
            arr = np.where(arr == float(nodata), np.nan, arr)
        src_tr = src.window_transform(win)
        return arr, src_tr, src.crs


def warp_corine_codes_nearest(corine_tif: Path, ref: dict[str, Any]) -> np.ndarray:
    """Nearest-neighbour warp of CORINE class pixels to a reference grid, using -9999 nodata."""
    h, w = int(ref["height"]), int(ref["width"])
    dst_tr = ref["transform"]
    dst_crs = rasterio.crs.CRS.from_string(ref["crs"])
    left, bottom, right, top = rasterio.transform.array_bounds(h, w, dst_tr)
    arr, src_tr, src_crs = _subwin_read_for_bounds(corine_tif, dst_crs, (left, bottom, right, top))
    src_i = np.where(np.isfinite(arr), np.rint(arr).astype(np.float32), -9999.0).astype(np.float32)
    out = np.full((h, w), -9999.0, dtype=np.float32)
    reproject(
        source=src_i,
        destination=out,
        src_transform=src_tr,
        src_crs=src_crs,
        dst_transform=dst_tr,
        dst_crs=dst_crs,
        src_nodata=-9999.0,
        dst_nodata=-9999.0,
        resampling=Resampling.nearest,
    )
    return np.where(out < -9980.0, -9999, np.rint(out)).astype(np.int32)


def read_corine_window(
    root: Path,
    cfg: dict[str, Any],
    ref: dict[str, Any],
) -> np.ndarray:
    """Read integer CLC codes for the reference window."""
    corine_path = ref["corine_path"]
    if not Path(corine_path).is_absolute():
        corine_path = resolve_path(root, Path(corine_path))
    band = int((cfg.get("corine") or {}).get("band", 1))
    h, w = int(ref["height"]), int(ref["width"])
    left, bottom, right, top = (float(x) for x in ref["window_bounds_3035"])
    with rasterio.open(corine_path) as src:
        win = from_bounds(left, bottom, right, top, transform=src.transform)
        win = win.round_lengths().intersection(Window(0, 0, src.width, src.height))
        arr = src.read(band, window=win).astype(np.float64)
        nodata = src.nodata
    if arr.shape != (h, w):
        raise ValueError(f"CORINE window {arr.shape} != ref {(h, w)}")
    out = np.rint(arr).astype(np.int32)
    if nodata is not None:
        out = np.where(arr == float(nodata), -9999, out)
    return out


def corine_binary_mask(clc_nn: np.ndarray, codes: list[int]) -> np.ndarray:
    """Max indicator over mutually exclusive CLC codes."""
    ci = np.asarray(clc_nn, dtype=np.int32)
    acc = np.zeros(ci.shape, dtype=np.float32)
    for c in codes:
        acc = np.maximum(acc, (ci == int(c)).astype(np.float32))
    return acc


def clc_group_masks(
    clc: np.ndarray,
    code_groups: dict[str, list[int]],
) -> dict[str, np.ndarray]:
    """Build binary float32 masks for each named list of CLC classes."""
    return {name: corine_binary_mask(clc, codes) for name, codes in code_groups.items()}


def corine_binary_mask_adapted(
    clc_nn: np.ndarray,
    yaml_codes: list[int],
) -> tuple[np.ndarray, list[int]]:
    codes_adapted, _remapped = adapt_corine_classes_for_grid(clc_nn, [int(x) for x in yaml_codes])
    if not codes_adapted:
        codes_adapted = [int(x) for x in yaml_codes]
    acc = np.zeros(np.asarray(clc_nn).shape, dtype=np.float32)
    ci = np.asarray(clc_nn, dtype=np.int32)
    for c in codes_adapted:
        acc = np.maximum(acc, (ci == int(c)).astype(np.float32))
    return acc, codes_adapted


__all__ = [
    "clc_group_masks",
    "corine_binary_mask",
    "corine_binary_mask_adapted",
    "read_corine_window",
    "resolve_corine_tif",
    "warp_corine_codes_nearest",
]
