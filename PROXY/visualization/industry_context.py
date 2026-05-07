"""B_Industry proxy layers on the Folium WGS84 grid.

Layer order matches the GNFR B preview contract: per-group CORINE class masks (YAML classes),
``OSM_<Gn>``, population + ``P_pop``, then per-group ``P_g`` (same naming pattern as fugitive previews).
"""

from __future__ import annotations

import sys
import yaml
from pathlib import Path
from typing import Any

import numpy as np

from PROXY.sectors.B_Industry.pipeline import _read_industry_gpkg_all_layers
from PROXY.visualization.fugitive_context import (
    intersect_ref_with_wgs84_bbox,
    ref_from_weight_geotiff,
    reproject_array_to_wgs84_grid,
)
from PROXY.visualization._palettes import palette_for_title
from PROXY.visualization.overlay_utils import scalar_to_rgba


def _nearest_resampling_for_industry_layer(title: str) -> bool:
    """Nearest-neighbour warp for discrete CORINE class masks (avoids bilinear halos)."""
    short = title.replace("Industry · ", "", 1)
    return short.startswith("CORINE")


def build_industry_proxy_rgba_overlays(
    root: Path,
    industry_cfg: dict[str, Any],
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
) -> list[tuple[str, np.ndarray]]:
    """
    Return (label, uint8 HxWx4) overlays: CORINE (per YAML classes × group), ``OSM_<Gn>``,
    population, ``P_pop``, ``G<n> (P_g)``.

    ``industry_cfg`` must match ``run_industry_pipeline`` (paths, proxy, ...).

    When ``scalars_out`` is provided, each warped WGS84 float grid is stored under its layer title
    (same keys as Folium labels) for offline PNG exports (e.g. bbox tools).
    """
    try:
        from rasterio.enums import Resampling
    except ImportError as exc:
        print(
            f"Warning: Industry context layers require rasterio: {exc}",
            file=sys.stderr,
        )
        return []

    from PROXY.core.dataloaders import resolve_path
    from PROXY.core.dataloaders.raster import warp_raster_to_ref
    from PROXY.core.ceip import DEFAULT_GNFR_GROUP_ORDER
    from PROXY.core.osm_corine_proxy import (
        build_all_group_pg,
        build_p_pop,
        clc_raw_group_score_robust,
    )
    from PROXY.core.ref_profile import resolve_corine_path

    try:
        ref0 = ref_from_weight_geotiff(weight_tif)
    except (OSError, ValueError) as exc:
        print(f"Warning: Industry context ref: {exc}", file=sys.stderr)
        return []

    paths_aug = dict(industry_cfg.get("paths") or {})
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
        osm_i = (path_cfg.get("osm") or {}).get("industry")
        if (not paths_aug.get("osm_industry_gpkg")) and osm_i:
            try:
                paths_aug["osm_industry_gpkg"] = str(resolve_path(root, Path(osm_i)).resolve())
            except (TypeError, OSError, ValueError):
                pass
    industry_cfg = {**industry_cfg, "paths": paths_aug, "_project_root": root}

    try:
        mini = intersect_ref_with_wgs84_bbox(ref0, west, south, east, north)
    except Exception as exc:
        print(f"Warning: Industry ref clip failed: {exc}", file=sys.stderr)
        return []
    if mini is None:
        print("Warning: map extent does not overlap the weight reference grid.", file=sys.stderr)
        return []

    crs_str = str(mini["crs"])
    transform = mini["transform"]
    gh, gw = int(dst_shape[0]), int(dst_shape[1])
    paths = dict(industry_cfg.get("paths") or {})

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
    osm_p = _rp(paths.get("osm_industry_gpkg"))
    gy = _rp(paths.get("ceip_groups_yaml"))

    if cor_p is None or not cor_p.is_file():
        print("Warning: Industry context: CORINE path missing; skipping proxy layers.", file=sys.stderr)
        return []
    if pop_p is None or not pop_p.is_file():
        print("Warning: Industry context: population_tif missing; skipping proxy layers.", file=sys.stderr)
        return []
    if osm_p is None or not osm_p.is_file():
        print("Warning: Industry context: osm_industry_gpkg missing; skipping proxy layers.", file=sys.stderr)
        return []
    if gy is None or not gy.is_file():
        print("Warning: Industry context: ceip_groups_yaml missing; skipping proxy layers.", file=sys.stderr)
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
        with gy.open(encoding="utf-8") as gf:
            group_specs_root = yaml.safe_load(gf) or {}
        groups_raw: dict = dict(group_specs_root.get("groups") or {})
        osm_gdf = _read_industry_gpkg_all_layers(osm_p)
        pcfg = industry_cfg.get("proxy") or {}
        group_order = tuple(str(x).strip() for x in (industry_cfg.get("group_order") or ()))
        if not group_order:
            group_order = DEFAULT_GNFR_GROUP_ORDER
        group_pg = build_all_group_pg(
            clc_nn,
            osm_gdf,
            {"groups": groups_raw},
            mini,
            pcfg,
            p_pop,
            group_order=group_order,
        )
    except Exception as exc:
        print(f"Warning: Industry proxy layer rebuild failed: {exc}", file=sys.stderr)
        return []

    layers_3035: list[tuple[str, np.ndarray]] = []
    for gid in group_order:
        spec = groups_raw.get(gid) or {}
        yaml_clc = [int(x) for x in (spec.get("corine_classes") or [])]
        if yaml_clc:
            cor_r = clc_raw_group_score_robust(clc_nn, yaml_clc)
            cls_str = ",".join(str(c) for c in yaml_clc)
            layers_3035.append((f"Industry · CORINE {cls_str} ({gid})", np.asarray(cor_r, dtype=np.float32)))

    for gid in group_order:
        d = group_pg.get(gid) or {}
        o = d.get("osm_raw")
        if o is not None:
            layers_3035.append((f"Industry · OSM_{gid}", np.asarray(o, dtype=np.float32)))

    layers_3035.extend(
        [
            ("Industry · Population", np.asarray(pop, dtype=np.float32)),
            ("Industry · P_pop (z-score)", np.asarray(p_pop, dtype=np.float32)),
        ]
    )

    for gid in group_order:
        d = group_pg.get(gid) or {}
        pg = d.get("p_g")
        if pg is not None:
            layers_3035.append((f"Industry · {gid} (P_g)", np.asarray(pg, dtype=np.float32)))

    out: list[tuple[str, np.ndarray]] = []
    for title, grid in layers_3035:
        try:
            rs = "nearest" if _nearest_resampling_for_industry_layer(title) else resampling
            warped = reproject_array_to_wgs84_grid(
                grid,
                transform,
                crs_str,
                dst_transform,
                (gh, gw),
                resampling=rs,
            )
        except Exception as exc:
            print(f"Warning: skip Industry layer {title!r}: {exc}", file=sys.stderr)
            continue
        if scalars_out is not None:
            scalars_out[title] = np.asarray(warped, dtype=np.float32).copy()
        if group_pg_out is not None:
            for _gid in group_order:
                if title == f"Industry · {_gid} (P_g)":
                    group_pg_out[_gid] = np.asarray(warped, dtype=np.float32)
                    break
        rgba = scalar_to_rgba(
            warped,
            colour_mode="percentile",
            cmap_name=palette_for_title(title),
            hide_zero=True,
            nodata_val=None,
        )
        if np.any(rgba[..., 3] > 0):
            out.append((title, rgba))

    return out
