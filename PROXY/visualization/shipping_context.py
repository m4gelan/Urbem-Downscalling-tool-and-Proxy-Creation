"""G_Shipping: EMODnet, CORINE port, OSM — RGBA layers on the weight GeoTIFF WGS84 preview grid."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from PROXY.core.dataloaders import resolve_path
from PROXY.core.ref_profile import resolve_corine_path
from PROXY.sectors.G_Shipping import proxy_shipping
from PROXY.visualization.fugitive_context import (
    intersect_ref_with_wgs84_bbox,
    ref_from_weight_geotiff,
    reproject_array_to_wgs84_grid,
)
from PROXY.visualization.overlay_utils import scalar_to_rgba

_CMAP: dict[str, str] = {
    "combined": "plasma",
    "D_n_damped": "inferno",
    "emodnet_raw": "cividis",
    "osm_coverage": "magma",
    "clc_port_frac": "Greens",
    "z_osm": "viridis",
    "z_clc": "YlOrRd",
}


def build_shipping_proxy_rgba_overlays(
    root: Path,
    ship_cfg: dict[str, Any],
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
) -> list[tuple[str, str, np.ndarray]]:
    try:
        ref0 = ref_from_weight_geotiff(weight_tif)
    except (OSError, ValueError) as exc:
        print(f"Warning: G_Shipping ref: {exc}", file=sys.stderr)
        return []

    paths_aug: dict[str, Any] = dict(ship_cfg.get("paths") or {})
    if path_cfg:
        main = path_cfg.get("proxy_common") or {}
        if not paths_aug.get("corine") and main.get("corine_tif"):
            try:
                paths_aug["corine"] = str(
                    resolve_path(root, Path(main["corine_tif"])).resolve()
                )
            except (TypeError, OSError, ValueError):
                pass
        pss = (path_cfg.get("proxy_specific") or {}).get("shipping") or {}
        if not paths_aug.get("emodnet") and pss.get("vessel_density_tif"):
            try:
                paths_aug["emodnet"] = str(
                    resolve_path(root, Path(pss["vessel_density_tif"])).resolve()
                )
            except (TypeError, OSError, ValueError):
                pass
        osm_s = (path_cfg.get("osm") or {}).get("shipping")
        if not paths_aug.get("osm_gpkg") and osm_s:
            try:
                paths_aug["osm_gpkg"] = str(resolve_path(root, Path(osm_s)).resolve())
            except (TypeError, OSError, ValueError):
                pass
    scfg = {**ship_cfg, "paths": paths_aug, "_project_root": root}

    try:
        mini = intersect_ref_with_wgs84_bbox(ref0, west, south, east, north)
    except Exception as exc:
        print(f"Warning: G_Shipping ref clip failed: {exc}", file=sys.stderr)
        return []
    if mini is None:
        print("Warning: map extent does not overlap the weight reference grid.", file=sys.stderr)
        return []

    mini = dict(mini)
    if "corine_path" not in mini and paths_aug.get("corine"):
        try:
            mini["corine_path"] = str(resolve_corine_path(root, paths_aug["corine"]).resolve())
        except (OSError, FileNotFoundError, TypeError, ValueError):
            pass

    em = paths_aug.get("emodnet")
    osm = paths_aug.get("osm_gpkg")
    if not em or not osm or not Path(str(em)).is_file() or not Path(str(osm)).is_file():
        print("Warning: G_Shipping context needs emodnet + osm_gpkg paths.", file=sys.stderr)
        return []
    cor = mini.get("corine_path") or paths_aug.get("corine")
    if not cor or not Path(str(cor)).is_file():
        try:
            cor = str(resolve_corine_path(root, cor).resolve())
            mini["corine_path"] = cor
        except (OSError, FileNotFoundError, TypeError, ValueError) as exc:
            print(f"Warning: G_Shipping corine: {exc}", file=sys.stderr)
            return []

    pr = (scfg.get("proxy") or {}) or {}
    osm_sub = int(pr.get("osm_subdivide", 4))
    land_d = float(pr.get("land_damp", 0.12))
    w_e = float(pr.get("weight_emodnet", 0.25))
    w_o = float(pr.get("weight_osm", 0.5))
    w_p = float(pr.get("weight_port", 0.25))
    pl2 = int(pr.get("port_level2", 123))

    try:
        p_comb, diag = proxy_shipping.build_combined_proxy(
            mini,
            emodnet_path=Path(str(em)),
            corine_path=Path(str(cor)),
            osm_gpkg=Path(str(osm)),
            osm_subdivide=osm_sub,
            land_damp=land_d,
            w_emodnet=w_e,
            w_osm=w_o,
            w_port=w_p,
            port_level2=pl2,
        )
    except Exception as exc:
        print(f"Warning: G_Shipping build_combined_proxy: {exc}", file=sys.stderr)
        return []

    layers: list[tuple[str, str, np.ndarray]] = [
        ("G_Shipping · combined proxy (pre-CAMS norm)", _CMAP["combined"], p_comb.astype(np.float32)),
    ]
    for key, title_suffix in (
        ("D_n_damped", "EMODnet: damped D_n"),
        ("emodnet_raw", "EMODnet: raw density"),
        ("osm_coverage", "OSM: coverage"),
        ("clc_port_frac", "CORINE: port fraction"),
        ("z_osm", "z(OSM) minmax"),
        ("z_clc", "z(port) minmax"),
    ):
        if key not in diag:
            continue
        arr = np.asarray(diag[key], dtype=np.float32)
        if key == "clc_nn":
            continue
        layers.append((f"G_Shipping · {title_suffix}", _CMAP.get(key, "viridis"), arr))

    crs_str = str(mini["crs"])
    transform = mini["transform"]
    gh, gw = int(dst_shape[0]), int(dst_shape[1])
    out: list[tuple[str, str, np.ndarray]] = []
    for title, cmap_name, grid in layers:
        try:
            g = np.asarray(grid, dtype=np.float32)
            warped = reproject_array_to_wgs84_grid(
                g,
                transform,
                crs_str,
                dst_transform,
                (gh, gw),
                resampling=resampling,
            )
            rgba = scalar_to_rgba(
                warped,
                colour_mode="percentile",
                cmap_name=cmap_name,
                hide_zero=True,
                nodata_val=None,
            )
        except Exception as exc:
            print(f"Warning: skip G_Shipping layer {title!r}: {exc}", file=sys.stderr)
            continue
        if np.any(rgba[..., 3] > 0):
            out.append((title, cmap_name, rgba))
    return out
