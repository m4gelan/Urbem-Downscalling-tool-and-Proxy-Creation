"""E_Solvents proxy previews: population, per-CLC masks (corine_codes), OSM raw channels (load_osm_indicators)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from PROXY.visualization.fugitive_context import (
    intersect_ref_with_wgs84_bbox,
    ref_from_weight_geotiff,
    reproject_array_to_wgs84_grid,
)
from PROXY.visualization.overlay_utils import scalar_to_rgba

_CORINE_CMAPS: dict[str, str] = {
    "residential_share": "viridis",
    "urban_fabric": "Greens",
    "service_land": "YlGnBu",
    "industrial_clc": "OrRd",
}
_OSM_CMAPS: dict[str, str] = {
    "service_osm": "plasma",
    "industry_osm": "inferno",
    "industry_buildings": "magma",
    "road_length": "cividis",
    "weighted_road_length": "Wistia",
    "transport_area": "PuRd",
    "roof_area": "Greys",
    "_road_length_raw": "cividis",
    "_weighted_road_raw": "Wistia",
}
_OSM_ORDER: tuple[str, ...] = (
    "service_osm",
    "industry_osm",
    "industry_buildings",
    "road_length",
    "weighted_road_length",
    "transport_area",
    "roof_area",
)


def build_solvents_proxy_rgba_overlays(
    root: Path,
    solvents_cfg: dict[str, Any],
    weight_tif: Path,
    west: float,
    south: float,
    east: float,
    north: float,
    dst_transform,
    dst_shape: tuple[int, int],
    path_cfg: dict[str, Any],
    *,
    resampling: str = "bilinear",
    scalars_out: dict[str, np.ndarray] | None = None,
) -> list[tuple[str, str, np.ndarray]]:
    """
    Return ``(title, colormap_id, uint8 HxWx4)`` for Folium.

    Titles: population, then each key in ``corine_codes`` (binary CLC group masks), then
    each OSM-derived channel from ``load_osm_indicators`` (non-underscore internal keys, plus
    optional ``_road_length_raw`` when present). ``colormap_id`` is the matplotlib map name
    used for the overlay (for legend hints).
    """
    try:
        from rasterio.enums import Resampling
    except ImportError as exc:
        print(
            f"Warning: Solvents context layers require rasterio: {exc}",
            file=sys.stderr,
        )
        return []

    from PROXY.core.dataloaders import resolve_path
    from PROXY.core.dataloaders.raster import warp_raster_to_ref
    from PROXY.core.osm_corine_proxy import build_p_pop
    from PROXY.core.ref_profile import resolve_corine_path
    from PROXY.core.corine.raster import clc_group_masks
    from PROXY.sectors.E_Solvents.osm_indicators import load_osm_indicators

    try:
        ref0 = ref_from_weight_geotiff(weight_tif)
    except (OSError, ValueError) as exc:
        print(f"Warning: Solvents context ref: {exc}", file=sys.stderr)
        return []

    paths_aug = dict(solvents_cfg.get("paths") or {})
    if path_cfg:
        main = path_cfg.get("proxy_common") or {}
        if not paths_aug.get("corine") and main.get("corine_tif"):
            try:
                paths_aug["corine"] = str(resolve_path(root, Path(main["corine_tif"])).resolve())
            except (TypeError, OSError, ValueError):
                pass
        if (not paths_aug.get("population_tif")) and main.get("population_tif"):
            try:
                paths_aug["population_tif"] = str(
                    resolve_path(root, Path(main["population_tif"])).resolve()
                )
            except (TypeError, OSError, ValueError):
                pass
        osm_s = (path_cfg.get("osm") or {}).get("solvents")
        if (not paths_aug.get("osm_solvent_gpkg")) and osm_s:
            try:
                paths_aug["osm_solvent_gpkg"] = str(resolve_path(root, Path(osm_s)).resolve())
            except (TypeError, OSError, ValueError):
                pass
    solvents_cfg = {**solvents_cfg, "paths": paths_aug, "_project_root": root}

    try:
        mini = intersect_ref_with_wgs84_bbox(ref0, west, south, east, north)
    except Exception as exc:
        print(f"Warning: Solvents ref clip failed: {exc}", file=sys.stderr)
        return []
    if mini is None:
        print("Warning: map extent does not overlap the weight reference grid.", file=sys.stderr)
        return []

    mini = dict(mini)
    if "corine_path" not in mini:
        cp = (solvents_cfg.get("paths") or {}).get("corine")
        if cp:
            pth = Path(cp) if Path(cp).is_absolute() else (root / Path(cp))
            if pth.is_file():
                mini["corine_path"] = pth
            else:
                try:
                    mini["corine_path"] = str(resolve_corine_path(root, cp).resolve())
                except (OSError, ValueError, TypeError, FileNotFoundError):
                    pass

    crs_str = str(mini["crs"])
    transform = mini["transform"]
    gh, gw = int(dst_shape[0]), int(dst_shape[1])
    paths = dict(solvents_cfg.get("paths") or {})

    def _rp(p: str | None) -> Path | None:
        if not p:
            return None
        x = Path(p)
        return x if x.is_absolute() else (root / x)

    cor_p = _rp(paths.get("corine"))
    if cor_p is None or not cor_p.is_file():
        if paths.get("corine"):
            try:
                cor_p = resolve_corine_path(root, paths["corine"])
            except (FileNotFoundError, OSError, ValueError, TypeError):
                cor_p = None
        if (cor_p is None or not cor_p.is_file()) and path_cfg:
            alt = (path_cfg.get("proxy_common") or {}).get("corine_tif")
            if alt:
                try:
                    cor_p = resolve_corine_path(root, alt)
                except (FileNotFoundError, OSError, ValueError, TypeError):
                    cor_p = None

    pop_p = _rp(paths.get("population_tif"))
    if pop_p and not pop_p.is_file():
        if path_cfg and (path_cfg.get("proxy_common") or {}).get("population_tif"):
            pop_p = _rp(str((path_cfg.get("proxy_common") or {})["population_tif"]))

    if cor_p is None or not cor_p.is_file():
        print("Warning: E_Solvents context: CORINE path missing; skipping proxy layers.", file=sys.stderr)
        return []
    if pop_p is None or not pop_p.is_file():
        print("Warning: E_Solvents context: population_tif missing; skipping population/P_pop layers.", file=sys.stderr)

    layers_3035: list[tuple[str, str, np.ndarray]] = []

    try:
        clc = warp_raster_to_ref(
            cor_p,
            mini,
            band=int((solvents_cfg.get("corine") or {}).get("band", 1)),
            resampling=Resampling.nearest,
            src_nodata=None,
            dst_nodata=np.nan,
        )
        clc_nn = np.full(clc.shape, -9999, dtype=np.int32)
        m = np.isfinite(clc)
        clc_nn[m] = np.rint(clc[m]).astype(np.int32, copy=False)
        if pop_p is not None and pop_p.is_file():
            pop = warp_raster_to_ref(
                pop_p,
                mini,
                band=1,
                resampling=Resampling.bilinear,
                src_nodata=None,
                dst_nodata=np.nan,
            )
            p_pop = build_p_pop(np.asarray(pop, dtype=np.float32), mini)
            from PROXY.visualization._palettes import PROXY_PALETTE as _PP
            layers_3035.append(
                (
                    "E_Solvents · population (ref window)",
                    _PP["population"],
                    np.asarray(pop, dtype=np.float32),
                )
            )
            layers_3035.append(
                (
                    "E_Solvents · P_pop (z-score density)",
                    _PP["p_pop"],
                    np.asarray(p_pop, dtype=np.float32),
                )
            )
        code_groups: dict[str, list[int]] = dict(solvents_cfg.get("corine_codes") or {})
        masks = clc_group_masks(clc_nn, code_groups) if code_groups else {}
        for ck in sorted(masks.keys()):
            cmap = _CORINE_CMAPS.get(ck, "viridis")
            layers_3035.append(
                (
                    f"E_Solvents · CLC: {ck}",
                    cmap,
                    np.asarray(masks[ck], dtype=np.float32),
                )
            )
    except Exception as exc:
        print(f"Warning: E_Solvents CORINE/population layers failed: {exc}", file=sys.stderr)

    try:
        # Same cfg shape as the build (paths + corine + osm).
        osm_raw = load_osm_indicators(root, solvents_cfg, mini)
        for key in _OSM_ORDER:
            if key not in osm_raw:
                continue
            arr = osm_raw[key]
            if arr is None:
                continue
            a = np.asarray(arr, dtype=np.float32)
            if not np.any(np.isfinite(a)):
                continue
            cmap = _OSM_CMAPS.get(key, "plasma")
            layers_3035.append((f"E_Solvents · OSM: {key}", cmap, a))
        for ex in ("_road_length_raw", "_weighted_road_raw"):
            if ex in osm_raw and ex not in _OSM_ORDER:
                a = np.asarray(osm_raw[ex], dtype=np.float32)
                cmap = _OSM_CMAPS.get(ex, "cividis")
                layers_3035.append(
                    (f"E_Solvents · OSM: {ex} (from GPKG lines)", cmap, a)
                )
    except Exception as exc:
        print(f"Warning: E_Solvents OSM context layers failed: {exc}", file=sys.stderr)

    out: list[tuple[str, str, np.ndarray]] = []
    for title, cmap_name, grid in layers_3035:
        try:
            warped = reproject_array_to_wgs84_grid(
                grid,
                transform,
                crs_str,
                dst_transform,
                (gh, gw),
                resampling=resampling,
            )
        except Exception as exc:
            print(f"Warning: skip E_Solvents layer {title!r}: {exc}", file=sys.stderr)
            continue
        if scalars_out is not None:
            scalars_out[title] = np.asarray(warped, dtype=np.float32)
        rgba = scalar_to_rgba(
            warped,
            colour_mode="percentile",
            cmap_name=cmap_name,
            hide_zero=True,
            nodata_val=None,
        )
        if np.any(rgba[..., 3] > 0):
            out.append((title, cmap_name, rgba))

    return out
