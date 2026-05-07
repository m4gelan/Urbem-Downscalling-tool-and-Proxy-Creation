"""Shared GeoTIFF metadata and grayscale PNG preview (used by viz_app and main UI)."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any


def resolve_under_root(base: Path, rel: str) -> Path:
    base = Path(base).resolve()
    if not rel or str(rel).strip() in (".", "/"):
        return base
    cand = (base / rel).resolve()
    try:
        cand.relative_to(base)
    except ValueError as e:
        raise ValueError("path outside allowed root") from e
    return cand


def raster_meta(path: Path) -> dict[str, Any]:
    import rasterio

    with rasterio.open(path) as ds:
        return {
            "path": str(path),
            "driver": ds.driver,
            "width": ds.width,
            "height": ds.height,
            "count": ds.count,
            "dtype": str(ds.dtypes[0]),
            "crs": str(ds.crs) if ds.crs else None,
            "transform": list(ds.transform)[:6],
            "nodata": ds.nodata,
        }


def raster_preview_png_bytes(
    path: Path,
    *,
    max_side: int = 1024,
    band: int = 1,
) -> bytes:
    import numpy as np
    import rasterio
    from rasterio.enums import Resampling
    from PIL import Image

    max_side = max(64, min(int(max_side), 8192))
    with rasterio.open(path) as ds:
        if band < 1 or band > ds.count:
            raise ValueError("invalid band")
        h, w = ds.height, ds.width
        scale = max(h, w) / float(max_side)
        if scale < 1.0:
            scale = 1.0
        oh = max(1, int(round(h / scale)))
        ow = max(1, int(round(w / scale)))
        arr = ds.read(band, out_shape=(oh, ow), resampling=Resampling.nearest).astype(
            np.float64
        )
        nodata = ds.nodata
        valid = np.isfinite(arr)
        if nodata is not None and np.isfinite(nodata):
            valid &= arr != float(nodata)
        if valid.any():
            mn = float(np.nanmin(arr[valid]))
            mx = float(np.nanmax(arr[valid]))
        else:
            mn, mx = 0.0, 1.0
        if mx <= mn:
            mx = mn + 1.0
        u8 = np.zeros(arr.shape, dtype=np.uint8)
        u8[valid] = np.clip((arr[valid] - mn) / (mx - mn) * 255.0, 0, 255).astype(
            np.uint8
        )

    img = Image.fromarray(u8, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()
