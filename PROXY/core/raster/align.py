"""Single raster alignment primitive with an explicit nodata policy.

Today the codebase has at least five variants of "warp a raster to the reference grid":

- ``PROXY/core/dataloaders/raster.warp_band_to_ref`` -- nan -> 0 at the end.
- ``PROXY/core/dataloaders/raster.warp_raster_to_ref`` -- keeps NaN.
- ``PROXY/sectors/J_Waste/io_waste.warp_raster_to_ref`` -- subset-window + NaN.
- ``Shipping/shipping_areasource.warp_corine_codes_nearest`` -- -9999 sentinel.
- ``PROXY/core/proxy_layers.reproject_raster_to_ref`` -- clamps negative to 0.

This file is the single forward-compatible implementation. Callers pick one of three
explicit nodata policies via :class:`NoDataPolicy`, so behavior never changes silently.
Phase 0 adds the module; sectors migrate one by one later.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.errors import WindowError
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window, from_bounds

logger = logging.getLogger(__name__)


class NoDataPolicy(str, Enum):
    """What to do with nodata / invalid pixels after the reprojection."""

    #: Replace ``nan``/``dst_nodata`` with ``0.0`` (suitable for mass-like rasters).
    FILL_ZERO = "fill_zero"
    #: Preserve ``NaN`` in the output (caller handles downstream).
    KEEP_NAN = "keep_nan"
    #: Preserve a sentinel integer value (e.g. ``-9999`` for CLC nearest-neighbor warps).
    SENTINEL = "sentinel"


@dataclass(frozen=True)
class WarpResult:
    array: np.ndarray
    src_path: Path
    band: int
    resampling: str
    policy: str
    nodata: float | int | None


def warp_to_ref(
    src_path: Path,
    *,
    ref: dict[str, Any],
    band: int = 1,
    resampling: Resampling | str = Resampling.bilinear,
    policy: NoDataPolicy = NoDataPolicy.KEEP_NAN,
    sentinel: float | int | None = None,
    dtype: Any = np.float32,
) -> WarpResult:
    """Warp ``band`` of ``src_path`` onto the ``ref`` grid using ``resampling`` + ``policy``.

    ``ref`` is the dict returned by :func:`PROXY.core.grid.reference_window_profile`
    (keys: ``height``, ``width``, ``transform``, ``crs``).
    """
    if isinstance(resampling, str):
        resampling = Resampling[resampling]

    dst = np.full((int(ref["height"]), int(ref["width"])), np.nan, dtype=np.float32)
    dst_nodata: float | int | None
    if policy is NoDataPolicy.FILL_ZERO:
        dst_nodata = np.nan
    elif policy is NoDataPolicy.KEEP_NAN:
        dst_nodata = np.nan
    elif policy is NoDataPolicy.SENTINEL:
        if sentinel is None:
            raise ValueError("NoDataPolicy.SENTINEL requires a `sentinel` value.")
        dst_nodata = sentinel
        dst = np.full((int(ref["height"]), int(ref["width"])), sentinel, dtype=np.float32)
    else:  # pragma: no cover - exhaustive
        raise ValueError(f"Unknown NoDataPolicy: {policy!r}")

    with rasterio.open(src_path) as src:
        reproject(
            source=rasterio.band(src, band),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=ref["transform"],
            dst_crs=ref["crs"],
            dst_nodata=dst_nodata,
            resampling=resampling,
        )

    if policy is NoDataPolicy.FILL_ZERO:
        arr = np.nan_to_num(dst, nan=0.0, posinf=0.0, neginf=0.0)
    elif policy is NoDataPolicy.KEEP_NAN:
        arr = dst
    else:
        arr = dst

    return WarpResult(
        array=arr.astype(dtype, copy=False),
        src_path=Path(src_path),
        band=int(band),
        resampling=resampling.name if hasattr(resampling, "name") else str(resampling),
        policy=policy.value,
        nodata=dst_nodata,
    )


def warp_sum_to_ref(
    src_path: Path,
    ref: dict[str, Any],
    *,
    band: int = 1,
) -> np.ndarray:
    """Sum-resample a mass/count raster to ``ref`` and return non-negative ``float32``."""
    res = warp_to_ref(
        Path(src_path),
        ref=ref,
        band=band,
        resampling=Resampling.sum,
        policy=NoDataPolicy.FILL_ZERO,
        dtype=np.float32,
    )
    out = np.nan_to_num(res.array, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32,
        copy=False,
    )
    return np.maximum(out, 0.0)


def ref_profile_to_kwargs(ref: dict[str, Any]) -> dict[str, Any]:
    """Return rasterio profile kwargs extracted from a ``ref`` dict."""
    return {
        "height": int(ref["height"]),
        "width": int(ref["width"]),
        "transform": ref["transform"],
        "crs": rasterio.crs.CRS.from_string(ref["crs"]),
    }


def warp_raster_to_ref(
    src_path: Path,
    ref: dict[str, Any],
    *,
    band: int = 1,
    resampling: Resampling = Resampling.bilinear,
    src_nodata: float | None = None,
    dst_nodata: float = np.nan,
) -> np.ndarray:
    """Window-subset-aware warp used by ``B_Industry`` / ``D_Fugitive`` / ``J_Waste``.

    Ported verbatim from ``Waste/j_waste_weights/io_utils.warp_raster_to_ref`` so sector
    pipelines can drop their out-of-tree dependency without any behaviour change (same
    NaN-on-overlap-miss semantics, same subsetting strategy).
    """
    kw = ref_profile_to_kwargs(ref)
    h, w = kw["height"], kw["width"]
    dst_transform = kw["transform"]
    dst_crs = kw["crs"]
    out = np.full((h, w), dst_nodata, dtype=np.float32)
    left, bottom, right, top = rasterio.transform.array_bounds(h, w, dst_transform)
    with rasterio.open(src_path) as src:
        nodata = src_nodata if src_nodata is not None else src.nodata
        west, south, east, north = transform_bounds(
            dst_crs, src.crs, left, bottom, right, top, densify_pts=21
        )
        win = from_bounds(west, south, east, north, transform=src.transform)
        try:
            win = win.round_lengths().intersection(Window(0, 0, src.width, src.height))
        except WindowError:
            logger.warning(
                "No spatial overlap between source %s and reference grid; output is nodata.",
                src_path.name,
            )
            return out
        if win.width < 1 or win.height < 1:
            logger.warning(
                "Source window empty for %s vs ref bounds; output is nodata.", src_path.name
            )
            return out
        arr = src.read(band, window=win).astype(np.float32)
        if nodata is not None and np.isfinite(nodata):
            arr = np.where(arr == nodata, np.nan, arr)
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
