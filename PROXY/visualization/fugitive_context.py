"""Fugitive proxy input layers (optional population for G3 context, VIIRS/GCMT/GOGET, per-group OSM/CLC/P_g) on the Folium WGS84 grid.

Matches the contract of ``SourceProxies.visualize_proxy_weights._build_fugitive_context_layers``,
using the **same** reference as the built weight GeoTIFF so layers align with ``read_weight_wgs84_only``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from PROXY.visualization._palettes import palette_for_title
from PROXY.visualization.overlay_utils import scalar_to_rgba


def is_fugitive_gem_auxiliary_layer(title: str) -> bool:
    """True for VIIRS / GEM / GOGET rasters that should treat zeros as transparent and use nearest neighbour warp."""
    return any(
        m in title
        for m in (
            "VIIRS",
            "GEM COAL MINE",
            "GEM_G2",
            "GEM_OG_G4",
            "VIIRS Nightfire",
            "GEM Coal Mine Tracker",
            "GOGET oil",
            "GOGET gas",
        )
    )


# Folium labels per CEIP group — must match ``proxy_mixture`` row order in merged D_Fugitive profile.
FUGITIVE_MIXTURE_LAYER_LABELS: dict[str, tuple[str, ...]] = {
    "G1": ("CLC 131", "GEM COAL MINE", "OSM_G1", "CLC121"),
    "G2": ("OSM_PIPE_G2", "GEM_G2", "OSM_PORT_G2", "CLC 123 (G2)", "CLC121 (G2)"),
    "G3": ("OSM_REFINERY_G3", "OSM_TANK_G3", "CLC 121 (G3)", "OSM_FUEL_G3", "POP"),
    "G4": ("VIIRS", "GEM_OG_G4", "OSM_FLARE_G4", "OSM_POWER_G4", "CLC 121 (G4)", "CLC 131 (G4)"),
}


def _fugitive_viz_debug_enabled(visualization_cfg: dict[str, Any] | None) -> bool:
    if os.environ.get("PROXY_DEBUG_FUGITIVE_VIZ", "").strip().lower() in ("1", "true", "yes"):
        return True
    v = visualization_cfg or {}
    return bool(v.get("debug_osm_gpkg") or v.get("debug_viz"))


def log_fugitive_viz_osm_gpkg_debug(osm_p: Path, loaded_gdf: Any) -> None:
    """Stderr: every layer name + feature count vs ``gpd.read_file(osm_p)`` (default/first layer)."""
    try:
        import fiona
        import geopandas as gpd

        layers = list(fiona.listlayers(str(osm_p)))
    except Exception as exc:
        print(f"[fugitive-viz-debug] osm_gpkg listlayers failed {osm_p}: {exc}", file=sys.stderr)
        return
    parts: list[str] = []
    for lay in layers:
        try:
            g = gpd.read_file(osm_p, layer=lay)
            parts.append(f"{lay}:{len(g)}")
        except Exception as exc:
            parts.append(f"{lay}:ERR({exc})")
    n_def = len(loaded_gdf) if loaded_gdf is not None else -1
    note = (
        "single layer — matches pipeline ``gpd.read_file``"
        if len(layers) <= 1
        else "multi-layer — viz & pipeline use DEFAULT (first) layer only (same as gnfr _load_default_osm)"
    )
    print(
        f"[fugitive-viz-debug] osm_fugitive_gpkg={osm_p.resolve()} "
        f"n_layers={len(layers)} layers={layers!r} "
        f"per_layer_rows=[{', '.join(parts)}] "
        f"read_file_default_rows={n_def} ({note})",
        file=sys.stderr,
    )


def _nearest_resampling_for_title(title: str) -> bool:
    """Nearest-neighbour warp for class masks / sparse flare grids (reduces bilinear ‘milky’ halos)."""
    short = title.replace("Fugitive · ", "", 1)
    if short.startswith("CLC"):
        return True
    if short.startswith("GEM") or short.startswith("VIIRS"):
        return True
    return False


def ref_from_weight_geotiff(tif: Path) -> dict[str, Any]:
    """Build a ``ref``-like dict from the weight raster (height, width, transform, crs, window bounds)."""
    import rasterio
    from rasterio.transform import array_bounds

    with rasterio.open(tif) as src:
        h, w = int(src.height), int(src.width)
        tr = src.transform
        crs = src.crs
        if crs is None:
            raise ValueError(f"Weight raster has no CRS: {tif}")
        l, b, r, t = array_bounds(h, w, tr)
    return {
        "height": h,
        "width": w,
        "transform": tr,
        "crs": crs.to_string(),
        "window_bounds_3035": (l, b, r, t),
    }


def _viz_auxiliary_mixture_kinds(paths: dict[str, Any]) -> set[str]:
    """
    Kinds to pass to ``prepare_fugitive_auxiliary_rasters`` so Folium previews always
    include VIIRS / GEM / GOGET when the dataset files exist, even if the CEIP groups
    YAML used for this run omits those ``proxy_mixture`` entries (misconfigured merge
    or stale profile).
    """
    extra: set[str] = set()
    v = paths.get("viirs_nightfire_csv")
    if v and Path(str(v)).is_file():
        extra.add("viirs_gaussian")
    g = paths.get("gcmt_xlsx")
    if g and Path(str(g)).is_file():
        extra.add("gcmt_coal")
    o = paths.get("goget_xlsx")
    if o and Path(str(o)).is_file():
        extra.update({"goget_g2", "goget_g4"})
    return extra


def intersect_ref_with_wgs84_bbox(
    ref: dict,
    west: float,
    south: float,
    east: float,
    north: float,
) -> dict | None:
    import rasterio
    from rasterio.transform import array_bounds
    from rasterio.windows import Window, from_bounds as win_from_bounds, transform as win_transform
    from rasterio.warp import transform_bounds

    crs = rasterio.crs.CRS.from_string(str(ref["crs"]))
    rw, rs, re, rn = (float(x) for x in ref["window_bounds_3035"])
    W, S, E, N = transform_bounds("EPSG:4326", crs, west, south, east, north, densify_pts=21)
    li = max(rw, min(W, E))
    ri = min(re, max(W, E))
    bi = max(rs, min(S, N))
    ti = min(rn, max(S, N))
    if li >= ri or bi >= ti:
        return None
    win = win_from_bounds(li, bi, ri, ti, transform=ref["transform"])
    win = win.round_lengths().intersection(Window(0, 0, int(ref["width"]), int(ref["height"])))
    if win.width < 2 or win.height < 2:
        return None
    mini_transform = win_transform(win, ref["transform"])
    ml, mb, mr, mt = array_bounds(int(win.height), int(win.width), mini_transform)
    return {
        **ref,
        "height": int(win.height),
        "width": int(win.width),
        "transform": mini_transform,
        "window_bounds_3035": (ml, mb, mr, mt),
        "domain_bbox_wgs84": (west, south, east, north),
    }


def reproject_array_to_wgs84_grid(
    arr: np.ndarray,
    src_transform,
    src_crs: str,
    dst_transform,
    dst_shape: tuple[int, int],
    *,
    resampling: str = "bilinear",
) -> np.ndarray:
    import rasterio
    from rasterio.warp import Resampling, reproject

    height, width = int(dst_shape[0]), int(dst_shape[1])
    dst = np.full((height, width), np.nan, dtype=np.float64)
    res = getattr(Resampling, str(resampling))
    reproject(
        source=np.asarray(arr, dtype=np.float64),
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs="EPSG:4326",
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=res,
    )
    return dst


def build_fugitive_proxy_rgba_overlays(
    root: Path,
    fugitive_cfg: dict[str, Any],
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
    group_pg_out: dict[str, np.ndarray] | None = None,
    scalars_out: dict[str, np.ndarray] | None = None,
    visualization_cfg: dict[str, Any] | None = None,
) -> list[tuple[str, np.ndarray]]:
    """
    Return (label, uint8 HxWx4) overlays for VIIRS/GCMT/GOGET (when configured), per-group OSM / CLC / P_g,
    and optional G3-only population context.

    Population raster overlays are omitted for groups where the methodology does not use population as an
    indicator (G1–G2–G4); **G3** still includes population inside ``P_g`` via ``proxy_mixture``. Standalone
    population maps are only included when relevant for interpreting **G3**.

    ``fugitive_cfg`` must be the same shape as for ``run_fugitive_pipeline`` (paths, proxy, ...).

    When ``scalars_out`` is set, each warped WGS84 float grid is stored under the Folium layer title
    (e.g. ``Fugitive · CLC 131``) for offline PNG exports.
    """
    try:
        import geopandas as gpd
        from rasterio.enums import Resampling
    except ImportError as exc:
        print(
            f"Warning: Fugitive context layers require geopandas, rasterio: {exc}",
            file=sys.stderr,
        )
        return []

    from PROXY.core.alpha import DEFAULT_GNFR_GROUP_ORDER
    from PROXY.core.alpha.ceip_index_loader import load_merged_ceip_profile_for_pipeline_paths
    from PROXY.core.dataloaders import resolve_path
    from PROXY.core.dataloaders.raster import warp_raster_to_ref
    from PROXY.core.osm_corine_proxy import build_p_pop
    from PROXY.sectors.D_Fugitive.fugitive_proxy import (
        build_fugitive_mixture_map_layers,
        build_fugitive_group_pg,
        mixture_kinds_for_groups,
        prepare_fugitive_auxiliary_rasters,
    )
    from PROXY.core.ref_profile import resolve_corine_path

    try:
        ref0 = ref_from_weight_geotiff(weight_tif)
    except (OSError, ValueError) as exc:
        print(f"Warning: Fugitive context ref: {exc}", file=sys.stderr)
        return []

    paths_aug = dict(fugitive_cfg.get("paths") or {})
    if path_cfg:
        main = path_cfg.get("proxy_common") or {}
        if not paths_aug.get("corine") and main.get("corine_tif"):
            try:
                paths_aug["corine"] = str(resolve_path(root, Path(main["corine_tif"])).resolve())
            except (TypeError, OSError, ValueError):
                pass
        if (not paths_aug.get("population_tif")) and main.get("population_tif"):
            try:
                paths_aug["population_tif"] = str(resolve_path(root, Path(main["population_tif"])).resolve())
            except (TypeError, OSError, ValueError):
                pass
        osm_f = (path_cfg.get("osm") or {}).get("fugitive")
        if (not paths_aug.get("osm_fugitive_gpkg")) and osm_f:
            try:
                paths_aug["osm_fugitive_gpkg"] = str(resolve_path(root, Path(osm_f)).resolve())
            except (TypeError, OSError, ValueError):
                pass
    fp_merge = fugitive_cfg.get("fugitive_paths") or {}
    if isinstance(fp_merge, dict):
        for k in ("viirs_nightfire_csv", "gcmt_xlsx", "goget_xlsx"):
            rel = fp_merge.get(k)
            if rel and not paths_aug.get(k):
                try:
                    paths_aug[k] = str(resolve_path(root, Path(rel)).resolve())
                except (TypeError, OSError, ValueError):
                    pass
    fugitive_cfg = {**fugitive_cfg, "paths": paths_aug, "_project_root": root}

    try:
        mini = intersect_ref_with_wgs84_bbox(ref0, west, south, east, north)
    except Exception as exc:
        print(f"Warning: Fugitive ref clip failed: {exc}", file=sys.stderr)
        return []
    if mini is None:
        print("Warning: map extent does not overlap the weight reference grid.", file=sys.stderr)
        return []

    crs_str = str(mini["crs"])
    transform = mini["transform"]
    gh, gw = int(dst_shape[0]), int(dst_shape[1])
    paths = dict(fugitive_cfg.get("paths") or {})

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
    osm_p = _rp(paths.get("osm_fugitive_gpkg"))
    gy = _rp(paths.get("ceip_groups_yaml"))

    if cor_p is None or not cor_p.is_file():
        print("Warning: Fugitive context: CORINE path missing; skipping proxy layers.", file=sys.stderr)
        return []
    if pop_p is None or not pop_p.is_file():
        print("Warning: Fugitive context: population_tif missing; skipping proxy layers.", file=sys.stderr)
        return []
    if osm_p is None or not osm_p.is_file():
        print("Warning: Fugitive context: osm_fugitive_gpkg missing; skipping proxy layers.", file=sys.stderr)
        return []
    if gy is None or not gy.is_file():
        print("Warning: Fugitive context: ceip_groups_yaml missing; skipping proxy layers.", file=sys.stderr)
        return []

    try:
        clc = warp_raster_to_ref(
            cor_p,
            mini,
            band=1,
            resampling=Resampling.nearest,
            src_nodata=None,
            dst_nodata=np.nan,
        )
        clc_nn = np.full(clc.shape, -9999, dtype=np.int32)
        m = np.isfinite(clc)
        clc_nn[m] = np.rint(clc[m]).astype(np.int32, copy=False)
        pop = warp_raster_to_ref(
            pop_p,
            mini,
            band=1,
            resampling=Resampling.bilinear,
            src_nodata=None,
            dst_nodata=np.nan,
        )
        p_pop = build_p_pop(pop, mini)
        group_specs_root = load_merged_ceip_profile_for_pipeline_paths(
            root, paths, profile_sector_id="D_Fugitive"
        )
        groups_raw: dict = dict(group_specs_root.get("groups") or {})
        osm_gdf = gpd.read_file(osm_p)
        if _fugitive_viz_debug_enabled(visualization_cfg):
            log_fugitive_viz_osm_gpkg_debug(osm_p, osm_gdf)
        pcfg = fugitive_cfg.get("proxy") or {}
        group_order = tuple(str(x).strip() for x in (fugitive_cfg.get("group_order") or ()))
        if not group_order:
            group_order = DEFAULT_GNFR_GROUP_ORDER
        fa = fugitive_cfg.get("fugitive_paths") or {}
        if isinstance(fa, dict):
            for k in ("viirs_nightfire_csv", "gcmt_xlsx", "goget_xlsx"):
                rel = fa.get(k)
                if rel and not paths_aug.get(k):
                    paths_aug[k] = str(resolve_path(root, Path(rel)).resolve())
            nuts_alt = path_cfg.get("proxy_common", {}).get("nuts_gpkg") if path_cfg else None
            if nuts_alt and not paths_aug.get("nuts_gpkg"):
                paths_aug["nuts_gpkg"] = str(resolve_path(root, Path(nuts_alt)).resolve())

        cfg_run = {
            **fugitive_cfg,
            "paths": paths_aug,
            "defaults": fugitive_cfg.get("defaults")
            or {"fallback_country_iso3": str(fugitive_cfg.get("cams_country_iso3", "GRC")).strip().upper()},
            "cntr_code_to_iso3": fugitive_cfg.get("cntr_code_to_iso3")
            or (fugitive_cfg.get("fugitive_paths") or {}).get("cntr_code_to_iso3")
            or {},
        }
        needed_kinds = set(mixture_kinds_for_groups({"groups": groups_raw}, group_order))
        needed_kinds |= _viz_auxiliary_mixture_kinds(paths)
        aux_rasters = prepare_fugitive_auxiliary_rasters(
            needed=needed_kinds,
            ref=mini,
            pcfg=pcfg,
            cfg=cfg_run,
            root=root,
            silent=True,
        )
        group_pg = build_fugitive_group_pg(
            clc_nn,
            osm_gdf,
            {"groups": groups_raw},
            mini,
            pcfg,
            p_pop,
            group_order=group_order,
            root=root,
            cfg=cfg_run,
            auxiliary_cache=aux_rasters,
        )

        layers_3035: list[tuple[str, np.ndarray]] = []

        mixture_flat = build_fugitive_mixture_map_layers(
            groups_raw=groups_raw,
            group_order=group_order,
            clc_nn=clc_nn,
            osm_gdf=osm_gdf,
            ref=mini,
            pcfg=pcfg,
            p_pop=p_pop,
            auxiliary_cache=aux_rasters,
            layer_labels=FUGITIVE_MIXTURE_LAYER_LABELS,
        )
        for lab, grid in mixture_flat:
            layers_3035.append((f"Fugitive · {lab}", np.asarray(grid, dtype=np.float32)))

        layers_3035.append(("Fugitive · P_pop (z-score)", np.asarray(p_pop, dtype=np.float32)))

        if "population" in needed_kinds:
            layers_3035.append(("Fugitive · population (raw, ref window)", np.asarray(pop, dtype=np.float32)))

        for gid in group_order:
            d = group_pg.get(gid) or {}
            pg = d.get("p_g")
            if pg is not None:
                layers_3035.append((f"Fugitive · {gid}_layer", np.asarray(pg, dtype=np.float32)))
    except Exception as exc:
        print(f"Warning: Fugitive proxy layer rebuild failed: {exc}", file=sys.stderr)
        return []

    out: list[tuple[str, np.ndarray]] = []
    for title, grid in layers_3035:
        try:
            rs = "nearest" if _nearest_resampling_for_title(title) else resampling
            warped = reproject_array_to_wgs84_grid(
                grid,
                transform,
                crs_str,
                dst_transform,
                (gh, gw),
                resampling=rs,
            )
        except Exception as exc:
            print(f"Warning: skip Fugitive layer {title!r}: {exc}", file=sys.stderr)
            continue
        if scalars_out is not None:
            scalars_out[title] = np.asarray(warped, dtype=np.float32).copy()
        if group_pg_out is not None:
            for _gid in group_order:
                if title == f"Fugitive · {_gid}_layer":
                    group_pg_out[_gid] = np.asarray(warped, dtype=np.float32)
                    break
        rgba = scalar_to_rgba(
            warped,
            colour_mode="percentile",
            cmap_name=palette_for_title(title),
            hide_zero=True,
            nodata_val=None,
        )
        out.append((title, rgba))

    return out
