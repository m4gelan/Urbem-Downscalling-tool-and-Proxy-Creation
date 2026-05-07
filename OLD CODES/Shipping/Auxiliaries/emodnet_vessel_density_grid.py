#!/usr/bin/env python3
"""
EMODnet HA vessel density GeoTIFF for downscaling pipelines.

Uses native CRS from the file (typically EPSG:3035) and rasterio.reproject onto
your domain/reference grid. This is the correct path for integration with
proxy/downscaling code; Folium maps use separate display-only bounds tweaks.

Compatible with the ``ref`` dict from ``Waste.j_waste_weights.io_utils.load_ref_profile``
(height, width, transform, crs).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.errors import WindowError
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window, from_bounds


def read_vessel_density_metadata(tif_path: Path) -> dict[str, Any]:
    """CRS, shape, transform, nodata — for checks before warping."""
    with rasterio.open(tif_path) as src:
        epsg = None
        if src.crs:
            try:
                epsg = src.crs.to_epsg()
            except Exception:
                epsg = None
        return {
            "path": str(tif_path.resolve()),
            "crs_wkt": src.crs.to_wkt() if src.crs else None,
            "epsg": epsg,
            "width": int(src.width),
            "height": int(src.height),
            "transform": [float(x) for x in src.transform[:6]],
            "bounds": {
                "left": float(src.bounds.left),
                "bottom": float(src.bounds.bottom),
                "right": float(src.bounds.right),
                "top": float(src.bounds.top),
            },
            "nodata": float(src.nodata) if src.nodata is not None and np.isfinite(src.nodata) else None,
            "dtype": src.dtypes[0],
            "count": int(src.count),
        }


def _sanitize_band(arr: np.ndarray, nodata: float | None) -> np.ndarray:
    a = arr.astype(np.float32, copy=False)
    if nodata is not None and np.isfinite(nodata):
        a = np.where(a == np.float32(nodata), np.nan, a)
    a = np.where(np.abs(a) > 1e38, np.nan, a)
    return a


def ref_profile_to_kwargs(ref: dict[str, Any]) -> dict[str, Any]:
    return {
        "height": int(ref["height"]),
        "width": int(ref["width"]),
        "transform": ref["transform"],
        "crs": CRS.from_string(ref["crs"]) if isinstance(ref["crs"], str) else ref["crs"],
    }


def warp_vessel_density_to_ref(
    src_path: Path,
    ref: dict[str, Any],
    *,
    band: int = 1,
    resampling: Resampling = Resampling.bilinear,
    dst_nodata: float = np.nan,
) -> np.ndarray:
    """
    Reproject one band from ``src_path`` onto ``ref`` grid (same contract as
    ``Waste.j_waste_weights.io_utils.warp_raster_to_ref``).

    Parameters
    ----------
    ref : dict
        Must include ``height``, ``width``, ``transform``, ``crs`` (string or CRS).
    """
    kw = ref_profile_to_kwargs(ref)
    h, w = kw["height"], kw["width"]
    dst_transform = kw["transform"]
    dst_crs = kw["crs"]
    out = np.full((h, w), dst_nodata, dtype=np.float32)
    left, bottom, right, top = rasterio.transform.array_bounds(h, w, dst_transform)
    with rasterio.open(src_path) as src:
        nodata = src.nodata
        west, south, east, north = transform_bounds(
            dst_crs, src.crs, left, bottom, right, top, densify_pts=21
        )
        win = from_bounds(west, south, east, north, transform=src.transform)
        try:
            win = win.round_lengths().intersection(Window(0, 0, src.width, src.height))
        except WindowError:
            return out
        if win.width < 1 or win.height < 1:
            return out
        arr = src.read(band, window=win)
        arr = _sanitize_band(arr, nodata)
        src_transform = src.window_transform(win)
        reproject(
            source=arr,
            destination=out,
            src_transform=src_transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            src_nodata=np.nan,
            dst_nodata=np.nan,
            resampling=resampling,
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Print EMODnet vessel-density GeoTIFF metadata (downscaling sanity check)."
    )
    ap.add_argument(
        "tif",
        type=Path,
        nargs="?",
        default=None,
        help="Path to vesseldensity_*.tif (default: default EMODnet path under data/Shipping)",
    )
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    tif = args.tif
    if tif is None:
        tif = (
            root
            / "data"
            / "Shipping"
            / "EMODnet"
            / "EMODnet_HA_Vessel_Density_allAvg"
            / "vesseldensity_all_2019.tif"
        )
    tif = tif if tif.is_absolute() else root / tif
    if not tif.is_file():
        print(f"File not found: {tif}", file=sys.stderr)
        return 1
    meta = read_vessel_density_metadata(tif)
    print(json.dumps(meta, indent=2))
    if meta.get("epsg") != 3035:
        print(
            "\nWarning: expected EPSG:3035 for standard EMODnet HA vessel density; "
            f"got epsg={meta.get('epsg')}.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
