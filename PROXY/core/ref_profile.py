"""Reference grid profile for area-source visualizations (replaces legacy Waste ``load_ref_profile`` for PROXY paths)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import rasterio
from rasterio.warp import transform_bounds

from PROXY.core.dataloaders import resolve_path
from PROXY.core.dataloaders.discovery import discover_corine
from PROXY.core.grid import reference_window_profile

logger = logging.getLogger(__name__)


def resolve_corine_path(root: Path, configured: str | Path | None) -> Path:
    """Resolve ``paths.corine`` to an existing GeoTIFF (direct path or ``discover_corine`` fallbacks)."""
    if configured is None:
        raise FileNotFoundError("paths.corine is missing in config")
    if isinstance(configured, str) and not configured.strip():
        raise FileNotFoundError("paths.corine is empty in config")
    p = resolve_path(root, configured)
    if p.is_file():
        return p
    return discover_corine(root, p)


def load_area_ref_profile(cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Build reference grid dict: ``height``, ``width``, ``transform``, ``crs`` (str), ``corine_path``,
    ``window_bounds_3035``, ``domain_bbox_wgs84``, optional ``ref_tif``.

    Same contract as ``Waste.j_waste_weights.io_utils.load_ref_profile`` but uses only PROXY + rasterio.
    Expects ``cfg["_project_root"]`` and ``cfg["paths"]``.
    """
    root = Path(cfg["_project_root"])
    paths = cfg["paths"]
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
                corine_path = resolve_corine_path(root, paths.get("corine"))
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

    cw = cfg.get("corine_window") or {}
    nuts_cntr = str(cw.get("nuts_cntr", "EL"))
    pad_m = float(cw.get("pad_m", 5000.0))
    corine_path = resolve_corine_path(root, paths.get("corine"))
    nuts_gpkg = resolve_path(root, Path(paths["nuts_gpkg"]))
    ref = reference_window_profile(
        corine_path=corine_path,
        nuts_gpkg=nuts_gpkg,
        nuts_country=nuts_cntr,
        pad_m=pad_m,
    )
    ref["corine_path"] = corine_path
    ref["ref_tif"] = None
    return ref
