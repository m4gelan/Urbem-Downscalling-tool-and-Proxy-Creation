from __future__ import annotations

import math
from io import BytesIO

import numpy as np
from PIL import Image
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject, transform_bounds

from UrbEm_Visualizer.visualization.emission_style import EPS, colormap_for, threshold_for
from UrbEm_Visualizer.visualization.raster_grid import AreaRaster

TILE_SIZE = 256

_CMAP_CACHE: dict[str, np.ndarray] = {}


def _cmap_lut(name: str) -> np.ndarray:
    if name not in _CMAP_CACHE:
        import matplotlib.cm as cm

        cmap = cm.get_cmap(name)
        _CMAP_CACHE[name] = (cmap(np.linspace(0, 1, 256)) * 255).astype(np.uint8)
    return _CMAP_CACHE[name]


def _tile_bounds_wgs84(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    n = 2.0**z
    west = x / n * 360.0 - 180.0
    east = (x + 1) / n * 360.0 - 180.0

    def _lat(ty: float) -> float:
        t = math.pi * (1.0 - 2.0 * ty / n)
        return math.degrees(math.atan(math.sinh(t)))

    north = _lat(y)
    south = _lat(y + 1)
    return west, south, east, north


def _sample_tile_values(raster: AreaRaster, z: int, x: int, y: int) -> np.ndarray:
    west, south, east, north = _tile_bounds_wgs84(z, x, y)
    dst = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.float32)
    dst_crs = CRS.from_epsg(3857)
    src_crs = CRS.from_string(raster.crs)
    left, bottom, right, top = transform_bounds(
        CRS.from_epsg(4326),
        dst_crs,
        west,
        south,
        east,
        north,
    )
    dst_transform = from_bounds(left, bottom, right, top, TILE_SIZE, TILE_SIZE)
    reproject(
        source=raster.data,
        destination=dst,
        src_transform=raster.transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest,
        src_nodata=0,
        dst_nodata=0,
    )
    return dst


def render_emission_tile(
    raster: AreaRaster,
    pollutant: str,
    lower_bound: float,
    upper_bound: float,
    z: int,
    x: int,
    y: int,
    threshold: float | None = None,
) -> bytes:
    raw = _sample_tile_values(raster, z, x, y)
    thr = float(threshold) if threshold is not None else threshold_for(pollutant)
    cmap = _cmap_lut(colormap_for(pollutant))

    rgba = np.zeros((TILE_SIZE, TILE_SIZE, 4), dtype=np.uint8)
    positive = raw > 0
    if not np.any(positive):
        return _empty_png()

    logv = np.log10(np.maximum(raw, 0.0) + EPS)
    denom = max(upper_bound - lower_bound, 1e-9)
    norm = np.clip((logv - lower_bound) / denom, 0.0, 1.0)

    idx = (norm * 255).astype(np.int32)
    idx = np.clip(idx, 0, 255)
    colors = cmap[idx]

    alpha = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.float32)
    below_thr = raw < thr
    alpha[below_thr & positive] = 0.08
    visible = positive & ~below_thr
    # Per-sector layers are often skewed; avoid near-zero alpha for valid cells above threshold.
    alpha[visible] = np.clip(0.42 + norm[visible] * 0.48, 0.42, 0.9)

    rgba[..., :3] = colors[..., :3]
    rgba[..., 3] = (alpha * 255).astype(np.uint8)
    rgba[~positive, 3] = 0

    img = Image.fromarray(rgba, mode="RGBA")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _empty_png() -> bytes:
    img = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (0, 0, 0, 0))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
