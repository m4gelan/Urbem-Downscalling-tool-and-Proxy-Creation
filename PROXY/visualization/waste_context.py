"""J_Waste proxy previews: solid / wastewater / residual, CLC+OSM, UWWtD, GHSL components."""

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

from PROXY.visualization._palettes import palette_for_key as _palette_for_key


# Cross-sector palette for J_Waste context layers. Shared with every other sector
# via ``_palettes.py`` so that "population" is the same colour everywhere.
_WASTE_SPECIAL_CMAP: dict[str, str] = {
    "proxy_solid": "YlOrBr",       # solid-waste stack
    "proxy_wastewater": "BuPu",    # wastewater stack
    "proxy_residual": "cividis",   # residual / fallback stack
    "corine_clc_132": "Greens",    # dump sites
    "corine_clc_121": "YlGn",      # industrial/commercial adjacency
    "imperv_valid_mask": "Greys",  # binary mask rendered in Greys
    "residual_ghsl_rural_mask": "Greens",
    "residual_imperv_01": "bone",
}


def _cmap_for_key(name: str) -> str:
    if name in _WASTE_SPECIAL_CMAP:
        return _WASTE_SPECIAL_CMAP[name]
    return _palette_for_key(name, default="viridis")


def build_waste_proxy_rgba_overlays(
    root: Path,
    waste_cfg: dict[str, Any],
    weight_tif: Path,
    west: float,
    south: float,
    east: float,
    north: float,
    dst_transform,
    dst_shape: tuple[int, int],
    path_cfg: dict[str, Any] | None,
    *,
    resampling: str = "bilinear",
    scalars_out: dict[str, np.ndarray] | None = None,
) -> list[tuple[str, str, np.ndarray]]:
    """
    Return (title, matplotlib_cmap, rgba uint8) for three proxy families, CLC/OSM solid masks,
    wastewater intermediates, and residual GHSL / population / impervious inputs.
    """
    from PROXY.core.dataloaders import resolve_path
    from PROXY.core.ref_profile import resolve_corine_path
    from PROXY.sectors.J_Waste import proxy_waste

    try:
        ref0 = ref_from_weight_geotiff(weight_tif)
    except (OSError, ValueError) as exc:
        print(f"Warning: J_Waste context ref: {exc}", file=sys.stderr)
        return []

    paths_aug = dict(waste_cfg.get("paths") or {})
    if path_cfg:
        main = path_cfg.get("proxy_common") or {}
        if not paths_aug.get("corine") and main.get("corine_tif"):
            try:
                paths_aug["corine"] = str(
                    resolve_path(root, Path(main["corine_tif"])).resolve()
                )
            except (TypeError, OSError, ValueError):
                pass
        if (not paths_aug.get("population_tif")) and main.get("population_tif"):
            try:
                paths_aug["population_tif"] = str(
                    resolve_path(root, Path(main["population_tif"])).resolve()
                )
            except (TypeError, OSError, ValueError):
                pass
        osm_w = (path_cfg.get("osm") or {}).get("waste")
        if (not paths_aug.get("osm_waste_gpkg")) and osm_w:
            try:
                paths_aug["osm_waste_gpkg"] = str(
                    resolve_path(root, Path(osm_w)).resolve()
                )
            except (TypeError, OSError, ValueError):
                pass
        psw = (path_cfg.get("proxy_specific") or {}).get("waste") or {}
        for pkey, cfg_key in (
            ("uwwtd_agglomerations_gpkg", "agglomerations_gpkg"),
            ("uwwtd_plants_gpkg", "treatment_plants_gpkg"),
            ("imperviousness", "impervious_tif"),
            ("ghsl_smod_tif", "ghsl_smod_tif"),
        ):
            if not paths_aug.get(pkey) and psw.get(cfg_key):
                try:
                    paths_aug[pkey] = str(
                        resolve_path(root, Path(psw[cfg_key])).resolve()
                    )
                except (TypeError, OSError, ValueError):
                    pass
    waste_cfg = {**waste_cfg, "paths": paths_aug, "_project_root": root}

    try:
        mini = intersect_ref_with_wgs84_bbox(ref0, west, south, east, north)
    except Exception as exc:
        print(f"Warning: J_Waste ref clip failed: {exc}", file=sys.stderr)
        return []
    if mini is None:
        print("Warning: map extent does not overlap the weight reference grid.", file=sys.stderr)
        return []

    mini = dict(mini)
    if "corine_path" not in mini and paths_aug.get("corine"):
        cp = paths_aug["corine"]
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
    if not (mini.get("corine_path") and Path(str(mini["corine_path"])).is_file()):
        cor_p = paths_aug.get("corine")
        if cor_p:
            try:
                p = resolve_corine_path(root, cor_p)
                if p.is_file():
                    mini["corine_path"] = p
            except (FileNotFoundError, OSError, ValueError, TypeError):
                pass

    layers: list[tuple[str, str, np.ndarray]] = []
    try:
        built = proxy_waste.build_all_proxies(waste_cfg, mini)
        layers.append(
            ("J_Waste · proxy: solid (combined, normalized)", "plasma", built["proxy_solid"].astype(np.float32))
        )
        layers.append(
            ("J_Waste · proxy: wastewater (combined, normalized)", "inferno", built["proxy_wastewater"].astype(np.float32))
        )
        layers.append(
            ("J_Waste · proxy: residual (combined, normalized)", "cividis", built["proxy_residual"].astype(np.float32))
        )
    except Exception as exc:
        print(f"Warning: J_Waste combined proxy rebuild failed: {exc}", file=sys.stderr)

    try:
        wwg = proxy_waste.build_wastewater_context_grids(waste_cfg, mini)
        for k, v in wwg.items():
            if k == "imperv_valid_mask":
                arr = np.asarray(v, dtype=np.float32)
            else:
                arr = np.asarray(v, dtype=np.float32)
            if not np.any(np.isfinite(arr)) and k != "imperv_valid_mask":
                continue
            layers.append((f"J_Waste · WW stack: {k}", _cmap_for_key(k), arr))
    except Exception as exc:
        print(f"Warning: J_Waste wastewater context failed: {exc}", file=sys.stderr)

    try:
        resg = proxy_waste.build_residual_context_grids(waste_cfg, mini)
        for k, v in resg.items():
            arr = np.asarray(v, dtype=np.float32)
            if not np.any(np.isfinite(arr)):
                continue
            layers.append((f"J_Waste · residual: {k}", _cmap_for_key(k), arr))
    except Exception as exc:
        print(f"Warning: J_Waste residual context failed: {exc}", file=sys.stderr)

    try:
        solid_masks = proxy_waste.build_solid_waste_context_masks(waste_cfg, mini)
        for k, v in sorted(solid_masks.items()):
            layers.append(
                (f"J_Waste · solid: {k}", _cmap_for_key(k), np.asarray(v, dtype=np.float32))
            )
    except Exception as exc:
        print(f"Warning: J_Waste solid context masks failed: {exc}", file=sys.stderr)

    out: list[tuple[str, str, np.ndarray]] = []
    for title, cmap_name, grid in layers:
        try:
            if cmap_name == "binary":
                cmap_name = "Greys"
            g = np.asarray(grid, dtype=np.float32)
            warped = reproject_array_to_wgs84_grid(
                g,
                transform,
                crs_str,
                dst_transform,
                (gh, gw),
                resampling=resampling,
            )
        except Exception as exc:
            print(f"Warning: skip J_Waste layer {title!r}: {exc}", file=sys.stderr)
            continue
        if scalars_out is not None:
            scalars_out[title] = np.asarray(warped, dtype=np.float32)
        rgba = scalar_to_rgba(
            warped,
            colour_mode="percentile",
            cmap_name=cmap_name if cmap_name != "binary" else "Greys",
            hide_zero=True,
            nodata_val=None,
        )
        if np.any(rgba[..., 3] > 0):
            out.append((title, cmap_name, rgba))
    return out
