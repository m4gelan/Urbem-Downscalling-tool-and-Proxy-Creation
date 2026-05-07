"""Reference grid, raster warping, and path discovery for CORINE."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.errors import WindowError
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window, from_bounds

from .config_loader import project_root, resolve_path  # noqa: F401 — resolve_path for callers

logger = logging.getLogger(__name__)


def load_ref_profile(cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Build reference grid dict: height, width, transform, crs (str), corine_path.

    If ``paths.ref_tif`` is set and exists, use that GeoTIFF's profile.
    Otherwise use ``SourceProxies.grid.reference_window_profile`` with
    ``corine_window.nuts_cntr`` and ``paths.corine`` / ``paths.nuts_gpkg``.
    """
    root: Path = cfg["_project_root"]
    paths = cfg["paths"]
    from SourceProxies.grid import first_existing_corine

    ref_tif = paths.get("ref_tif")
    if ref_tif:
        p = resolve_path(root, Path(ref_tif))
        if p.is_file():
            with rasterio.open(p) as src:
                left, bottom, right, top = rasterio.transform.array_bounds(
                    src.height, src.width, src.transform
                )
                w, s, e, n = transform_bounds(
                    src.crs, "EPSG:4326", left, bottom, right, top, densify_pts=21
                )
                corine_path = first_existing_corine(root, paths.get("corine"))
                return {
                    "corine_path": corine_path,
                    "height": int(src.height),
                    "width": int(src.width),
                    "transform": src.transform,
                    "crs": src.crs.to_string(),
                    "window_bounds_3035": (left, bottom, right, top),
                    "domain_bbox_wgs84": (w, s, e, n),
                    "ref_tif": p,
                }
        logger.warning("ref_tif not found (%s); falling back to CORINE+NUTS window.", p)

    from SourceProxies.grid import reference_window_profile

    cw = cfg.get("corine_window") or {}
    nuts_cntr = str(cw.get("nuts_cntr", "EL"))
    pad_m = float(cw.get("pad_m", 5000.0))
    corine_path = first_existing_corine(root, paths.get("corine"))
    nuts_gpkg = resolve_path(root, Path(paths["nuts_gpkg"]))
    ref = reference_window_profile(
        corine_path=corine_path,
        nuts_gpkg=nuts_gpkg,
        nuts_cntr=nuts_cntr,
        pad_m=pad_m,
    )
    ref["corine_path"] = corine_path
    ref["ref_tif"] = None
    return ref


def ref_profile_to_kwargs(ref: dict[str, Any]) -> dict[str, Any]:
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
    """
    Read one band from ``src_path`` and reproject to ref grid shape/transform/CRS.

    Assumes ref CRS is EPSG:3035 (or any metric CRS); uses center-of-pixel sampling.
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
            logger.warning("Source window empty for %s vs ref bounds; output is nodata.", src_path.name)
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


def find_first_raster_in_dir(d: Path) -> Path | None:
    """Pick first .tif/.tiff under directory (shallow then rglob limited)."""
    if not d.exists():
        return None
    if d.is_file() and d.suffix.lower() in (".tif", ".tiff", ".jp2"):
        return d
    for pat in ("*.tif", "*.tiff", "*.jp2"):
        found = sorted(d.glob(pat))
        if found:
            return found[0]
    found = sorted(d.rglob("*.tif"))
    return found[0] if found else None


def read_raster_path_or_dir(path: Path, ref: dict[str, Any], **kwargs: Any) -> np.ndarray:
    """If path is directory, use first GeoTIFF inside."""
    p = path
    if p.is_dir():
        inner = find_first_raster_in_dir(p)
        if inner is None:
            raise FileNotFoundError(f"No raster found under directory: {p}")
        p = inner
        logger.info("Using raster from directory: %s", p)
    return warp_raster_to_ref(p, ref, **kwargs)
