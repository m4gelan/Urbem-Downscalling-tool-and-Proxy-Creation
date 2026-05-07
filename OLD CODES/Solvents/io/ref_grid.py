"""Fine reference grid: from existing GeoTIFF or from CORINE+NUTS (SourceProxies.grid)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import rasterio
from rasterio.warp import transform_bounds

from .paths import resolve_path as project_resolve


def load_ref_profile(root: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    paths = cfg["paths"]
    ref_tif = paths.get("ref_tif")
    if ref_tif:
        p = project_resolve(root, Path(ref_tif))
        if p.is_file():
            with rasterio.open(p) as src:
                left, bottom, right, top = rasterio.transform.array_bounds(
                    src.height, src.width, src.transform
                )
                w, s, e, n = transform_bounds(
                    src.crs, "EPSG:4326", left, bottom, right, top, densify_pts=21
                )
                return {
                    "corine_path": project_resolve(root, paths["corine"]),
                    "height": int(src.height),
                    "width": int(src.width),
                    "transform": src.transform,
                    "crs": src.crs.to_string(),
                    "window_bounds_3035": (left, bottom, right, top),
                    "domain_bbox_wgs84": (w, s, e, n),
                }

    nuts_gpkg = project_resolve(root, paths["nuts_gpkg"])
    pad_m = float((cfg.get("corine") or {}).get("pad_m", 5000.0))
    cntr = str(cfg["country"]["nuts_cntr"])
    try:
        from SourceProxies.grid import first_existing_corine, reference_window_profile as ref_sp

        corine_path = first_existing_corine(root, paths.get("corine"))
        ref = ref_sp(
            corine_path=corine_path,
            nuts_gpkg=nuts_gpkg,
            nuts_cntr=cntr,
            pad_m=pad_m,
        )
    except ImportError:
        from PROXY.core.dataloaders.discovery import discover_corine
        from PROXY.core.grid import reference_window_profile as ref_proxy

        corine_path = discover_corine(
            root, project_resolve(root, Path(paths.get("corine", "")))
        )
        ref = ref_proxy(
            corine_path=corine_path,
            nuts_gpkg=nuts_gpkg,
            nuts_country=cntr,
            pad_m=pad_m,
        )
    ref["corine_path"] = corine_path
    return ref
