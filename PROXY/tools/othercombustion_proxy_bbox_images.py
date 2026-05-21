#!/usr/bin/env python3
"""GNFR C (`C_OtherCombustion`) bbox PNG exports: stationary S×L bands, off-road CLC/OSM layers, PM10/NOx weights.

Mirrors the idea of ``industry_proxy_bbox_images`` / ``offroad_proxy_bbox_images``: pick a WGS84 view from the
output weight GeoTIFF, warp reference-grid rasters to that view, and save matplotlib PNGs.

**Stationary (appliance mode):** per requested class (default ``R_FIREPLACE`` and ``C_BOILER_AUT``), three maps
each — **stock** ``S_k``, **load** ``L_k``, and **X** band ``S_k L_k`` (same construction as ``appliance_proxy_stack``).

**Off-road ingredients:** one RGB composite on the reference grid then warped — forest CLC classes (G2),
CLC 112 × population (G3 context), CLC 121 + λ·OSM (G4 context). Not per-CAMS-cell normalized weights; for
cell-normalised behaviour see the weight PNGs.

**Weights:** per-CAMS-cell **2–98%** stretch on positive weights (below the cell’s 2nd percentile is
transparent); optional **√z01** alpha fade so pale tails do not wash out the basemap. Without CAMS,
the same 2–98% rule is applied globally.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

_root_boot = Path(__file__).resolve().parents[2]
if str(_root_boot) not in sys.path:
    sys.path.insert(0, str(_root_boot))

from PROXY.tools.waste_proxy_bbox_images import (
    _alpha_composite_rgb_under_rgba,
    _composite_rgba_over_osm,
    _save_png,
)

DEFAULT_OTHERC_BBOX_WGS84 = (23.45, 37.90, 23.85, 38.13)


def _ensure_otherc_display_colormaps() -> None:
    """Register sequential ramps used only by this bbox tool (matplotlib names)."""
    from matplotlib import colormaps
    from matplotlib.colors import LinearSegmentedColormap

    if "otherc_white_red" not in colormaps:
        colormaps.register(
            LinearSegmentedColormap.from_list(
                "otherc_white_red",
                [(1.0, 1.0, 1.0), (1.0, 0.45, 0.45), (0.78, 0.0, 0.0)],
                N=256,
            )
        )
    if "otherc_browns" not in colormaps:
        colormaps.register(
            LinearSegmentedColormap.from_list(
                "otherc_browns",
                [(0.98, 0.96, 0.92), (0.72, 0.52, 0.34), (0.40, 0.24, 0.12), (0.16, 0.09, 0.05)],
                N=256,
            )
        )


def _basemap_ctx_source(key: str) -> tuple[object, str]:
    """Return ``(contextily provider, short name)`` for non-satellite previews."""
    import contextily as ctx

    k = (key or "osm").strip().lower().replace("_", "-")
    if k in ("osm", "openstreetmap", "mapnik"):
        return ctx.providers.OpenStreetMap.Mapnik, "OpenStreetMap Mapnik"
    if k in ("carto-positron", "positron", "light"):
        return ctx.providers.CartoDB.Positron, "CartoDB Positron"
    if k in ("carto-voyager", "voyager"):
        return ctx.providers.CartoDB.Voyager, "CartoDB Voyager"
    if k in ("carto-dark", "dark-matter", "dark"):
        return ctx.providers.CartoDB.DarkMatter, "CartoDB Dark Matter"
    raise ValueError(
        f"Unknown --basemap {key!r}; try osm, carto-positron, carto-voyager, or carto-dark."
    )


def _per_cell_robust_stretch_01(
    z: np.ndarray,
    cell_id: np.ndarray,
    *,
    base_valid: np.ndarray,
    lo_pct: float = 2.0,
    hi_pct: float = 98.0,
    min_n: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Per CAMS cell: stretch ``z`` between per-cell ``p_lo`` and ``p_hi`` (percentiles of positives).

    Pixels with ``z <= p_lo`` stay **invalid** (transparent). ``z >= p_hi`` maps to 1. Avoids per-cell
    min–max always pinning one pixel to colormap 0 (opaque pale fill across the map).
    """
    z = np.asarray(z, dtype=np.float64)
    cid = np.asarray(cell_id)
    vin = np.asarray(base_valid, dtype=bool)
    if cid.shape != z.shape:
        if cid.size == z.size:
            cid = cid.reshape(z.shape)
        else:
            raise ValueError(f"cell_id shape {cid.shape} incompatible with z {z.shape}")
    z01 = np.full(z.shape, np.nan, dtype=np.float64)
    valid = np.zeros(z.shape, dtype=bool)
    for c in np.unique(cid):
        if int(c) < 0:
            continue
        m_cell = cid == int(c)
        sel = m_cell & vin
        if not np.any(sel):
            continue
        vals = z[sel]
        pos = np.isfinite(vals) & (vals > 0)
        if not np.any(pos):
            continue
        v = vals[pos]
        n = int(v.size)
        if n < int(min_n):
            p_lo = float(np.min(v))
            p_hi = float(np.max(v))
        else:
            p_lo = float(np.percentile(v, lo_pct))
            p_hi = float(np.percentile(v, hi_pct))
        if p_hi <= p_lo:
            p_hi = p_lo + 1e-12
        m_mid = m_cell & vin & np.isfinite(z) & (z > p_lo) & (z < p_hi)
        m_hi = m_cell & vin & np.isfinite(z) & (z >= p_hi)
        z01[m_mid] = (z[m_mid] - p_lo) / (p_hi - p_lo)
        z01[m_hi] = 1.0
        valid |= m_mid | m_hi
    return z01, valid


def _global_robust_stretch_01(
    z: np.ndarray,
    *,
    base_valid: np.ndarray,
    lo_pct: float = 2.0,
    hi_pct: float = 98.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Global 2–98% stretch on positives in ``base_valid``; below ``p_lo`` transparent."""
    z = np.asarray(z, dtype=np.float64)
    vin = np.asarray(base_valid, dtype=bool)
    z01 = np.full(z.shape, np.nan, dtype=np.float64)
    valid = np.zeros(z.shape, dtype=bool)
    if not np.any(vin):
        return z01, valid
    v = z[vin]
    pos = np.isfinite(v) & (v > 0)
    if not np.any(pos):
        return z01, valid
    vv = v[pos]
    p_lo = float(np.percentile(vv, lo_pct))
    p_hi = float(np.percentile(vv, hi_pct))
    if p_hi <= p_lo:
        p_hi = p_lo + 1e-12
    m_mid = vin & np.isfinite(z) & (z > p_lo) & (z < p_hi)
    m_hi = vin & np.isfinite(z) & (z >= p_hi)
    z01[m_mid] = (z[m_mid] - p_lo) / (p_hi - p_lo)
    z01[m_hi] = 1.0
    valid = m_mid | m_hi
    return z01, valid


def _apply_sqrt_alpha_fade_on_valid(rgba: np.ndarray, z01: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Reduce alpha where display ``z01`` is small so the basemap stays readable in the tail."""
    out = np.asarray(rgba, dtype=np.uint8).copy()
    vm = np.asarray(valid, dtype=bool)
    tt = np.clip(
        np.nan_to_num(np.asarray(z01, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0),
        0.0,
        1.0,
    )
    sf = np.zeros_like(tt, dtype=np.float64)
    sf[vm] = np.sqrt(tt[vm])
    out[..., 3] = np.clip(out[..., 3].astype(np.float64) * sf, 0, 255).astype(np.uint8)
    return out


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_import_path() -> Path:
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _merge_other_combustion_cfg(
    root: Path,
    path_cfg: dict[str, Any],
    sector_cfg: dict[str, Any],
    country_nuts: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from PROXY.core.alpha.ceip_index_loader import default_ceip_profile_relpath, remap_legacy_ceip_relpath
    from PROXY.core.dataloaders import resolve_path
    from PROXY.core.grid import reference_window_profile

    spec = (path_cfg.get("proxy_specific") or {}).get("other_combustion") or {}
    if not spec:
        raise ValueError("paths.yaml: missing proxy_specific.other_combustion")
    sector_config_dir = root / "PROXY" / "config" / "other_combustion"
    corine_path = Path(path_cfg["proxy_common"]["corine_tif"])
    nuts_gpkg = Path(path_cfg["proxy_common"]["nuts_gpkg"])
    corine_block = sector_cfg.get("corine") or {}
    pad_m = float(corine_block.get("pad_m", sector_cfg.get("pad_m", 5000.0)))
    ref = reference_window_profile(
        corine_path=corine_path,
        nuts_gpkg=nuts_gpkg,
        nuts_country=country_nuts.strip().upper(),
        pad_m=pad_m,
    )
    hm_dir = Path(spec["hotmaps_dir"])
    hm_names = sector_cfg.get("hotmaps") or {}
    proxy_common = path_cfg.get("proxy_common") or {}
    alpha_w = proxy_common.get("alpha_workbook")
    if not alpha_w:
        raise ValueError("paths.yaml proxy_common.alpha_workbook is required for GNFR C")
    path_osm = path_cfg.get("osm") or {}
    oc_rel = path_osm.get("other_combustion") or path_osm.get("industry")
    if not oc_rel:
        raise ValueError("paths.yaml: osm.other_combustion or osm.industry required")
    ceip_ov = sector_cfg.get("ceip") if isinstance(sector_cfg.get("ceip"), dict) else {}
    gy_rel = remap_legacy_ceip_relpath(
        str(ceip_ov.get("groups_yaml") or default_ceip_profile_relpath(root, "C_OtherCombustion", "groups_yaml"))
    )
    ry_rel = remap_legacy_ceip_relpath(
        str(ceip_ov.get("rules_yaml") or default_ceip_profile_relpath(root, "C_OtherCombustion", "rules_yaml"))
    )
    waste_spec = (path_cfg.get("proxy_specific") or {}).get("waste") or {}
    merged: dict[str, Any] = {
        "country": {
            "cams_iso3": str(sector_cfg.get("cams_country_iso3", "GRC")).strip().upper(),
            "nuts_cntr": country_nuts.strip().upper(),
        },
        "paths": {
            "cams_nc": path_cfg["emissions"]["cams_2019_nc"],
            "nuts_gpkg": nuts_gpkg,
            "gains_dir": spec["gains_dir"],
            "eurostat_xlsx": spec.get("eurostat_xlsx"),
            "population_tif": proxy_common.get("population_tif"),
            "ghsl_smod_tif": waste_spec.get("ghsl_smod_tif"),
            "ceip_workbook": resolve_path(root, Path(alpha_w)),
            "ceip_groups_yaml": resolve_path(root, Path(gy_rel)),
            "ceip_rules_yaml": resolve_path(root, Path(ry_rel)),
            "osm_other_combustion_gpkg": resolve_path(root, Path(oc_rel)),
            "emep_ef": sector_config_dir / "EMEP_emission_factors.yaml",
            "gains_mapping": sector_config_dir / "GAINS_mapping.yaml",
            "eurostat_end_use_json": sector_config_dir / "eurostat_end_use.yaml",
            "hotmaps": {
                "heat_res": hm_dir / str(hm_names.get("heat_res", "heat_res_curr_density.tif")),
                "heat_nonres": hm_dir / str(hm_names.get("heat_nonres", "heat_nonres_curr_density.tif")),
                "hdd_curr": hm_dir / str(hm_names.get("hdd_curr", "HDD_curr.tif")),
                "gfa_res": hm_dir / str(hm_names.get("gfa_res", "gfa_res_curr_density.tif")),
                "gfa_nonres": hm_dir / str(hm_names.get("gfa_nonres", "gfa_nonres_curr_density.tif")),
            },
        },
        "appliance_proxy": sector_cfg.get("appliance_proxy") or {},
        "corine": corine_block,
        "cams": sector_cfg.get("cams") or {},
        "base_proxy": sector_cfg["base_proxy"],
        "morphology": sector_cfg["morphology"],
        "gains": sector_cfg.get("gains") or {},
        "eurostat": sector_cfg.get("eurostat") or {},
        "co2": sector_cfg.get("co2") or {},
        "pollutants": sector_cfg.get("pollutants") or [],
        "run": sector_cfg.get("run") or {},
        "ceip": {
            "years": ceip_ov.get("years"),
            "cntr_code_to_iso3": ceip_ov.get("cntr_code_to_iso3") or {},
            "pollutant_aliases": ceip_ov.get("pollutant_aliases") or {},
        },
    }
    return merged, ref


def _warp_ref_to_wgs84(
    arr: np.ndarray,
    ref: dict[str, Any],
    *,
    west: float,
    south: float,
    east: float,
    north: float,
    width: int,
    height: int,
    resampling: str = "bilinear",
) -> np.ndarray:
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.warp import Resampling, reproject

    src = np.asarray(arr, dtype=np.float64)
    dst = np.full((height, width), np.nan, dtype=np.float64)
    dst_t = from_bounds(west, south, east, north, width, height)
    rs = getattr(Resampling, str(resampling))
    from rasterio.crs import CRS

    src_crs = CRS.from_string(str(ref["crs"]))
    reproject(
        source=src,
        destination=dst,
        src_transform=ref["transform"],
        src_crs=src_crs,
        dst_transform=dst_t,
        dst_crs="EPSG:4326",
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=rs,
    )
    return dst


def _warp_ref_rgb_to_wgs84(
    rgb: np.ndarray,
    ref: dict[str, Any],
    *,
    west: float,
    south: float,
    east: float,
    north: float,
    width: int,
    height: int,
) -> np.ndarray:
    out = np.zeros((height, width, 3), dtype=np.float64)
    for c in range(3):
        out[..., c] = _warp_ref_to_wgs84(rgb[..., c], ref, west=west, south=south, east=east, north=north, width=width, height=height)
    return np.clip(out, 0.0, 1.0)


def _offroad_rgb_ref(
    clc_l3: np.ndarray,
    pop: np.ndarray,
    osm_hit: np.ndarray,
    off_cfg: dict[str, Any],
) -> tuple[np.ndarray, list[tuple[str, tuple[int, int, int]]]]:
    """HxWx3 float01 composite on reference grid (not cell-normalised)."""
    clc = np.asarray(clc_l3, dtype=np.float64)
    pop_a = np.maximum(np.asarray(pop, dtype=np.float64), 0.0)
    o = np.clip(np.asarray(osm_hit, dtype=np.float64), 0.0, 1.0)
    h, w = clc.shape
    ci = np.where(np.isfinite(clc), np.rint(clc).astype(np.int32), -9999)

    forest_codes = [int(x) for x in off_cfg.get("forest_corine_classes", [311, 312, 313])]
    res_codes = frozenset(int(x) for x in off_cfg.get("residential_corine_codes", [112]))
    comm_clc = int(off_cfg.get("commercial_clc_code", 121))
    lam = float(off_cfg.get("lambda_osm", 1.0))

    forest_rgb = {
        311: (0.15, 0.45, 0.18),
        312: (0.25, 0.62, 0.22),
        313: (0.40, 0.78, 0.30),
    }
    rgb = np.zeros((h, w, 3), dtype=np.float64)
    for code in forest_codes:
        m = ci == int(code)
        col = forest_rgb.get(int(code), (0.2, 0.5, 0.2))
        rgb[m, 0] = col[0]
        rgb[m, 1] = col[1]
        rgb[m, 2] = col[2]

    m112 = np.isin(ci, np.array(list(res_codes), dtype=np.int32))
    p_hi = float(np.nanpercentile(pop_a[m112], 98.0)) if np.any(m112) else 1.0
    p_hi = max(p_hi, 1e-9)
    tpop = np.clip(pop_a / p_hi, 0.0, 1.0)
    rgb[m112, 0] = np.maximum(rgb[m112, 0], 0.95 * tpop[m112])
    rgb[m112, 1] = np.maximum(rgb[m112, 1], 0.75 * tpop[m112])
    rgb[m112, 2] = np.maximum(rgb[m112, 2], 0.15 * tpop[m112])

    comm = ci == int(comm_clc)
    sig = comm.astype(np.float64) + lam * o
    sig = np.clip(sig, 0.0, None)
    sm = float(np.nanpercentile(sig, 99.0)) if np.any(np.isfinite(sig) & (sig > 0)) else 1.0
    sm = max(sm, 1e-9)
    ts = np.clip(sig / sm, 0.0, 1.0)
    rgb[..., 0] = np.maximum(rgb[..., 0], 0.85 * ts)
    rgb[..., 1] = np.maximum(rgb[..., 1], 0.20 * ts)
    rgb[..., 2] = np.maximum(rgb[..., 2], 0.75 * ts)

    legend: list[tuple[str, tuple[int, int, int]]] = [
        ("Forest CLC (per-class greens)", (40, 120, 50)),
        ("Residential CLC112 + pop (warm)", (240, 190, 40)),
        ("Commercial CLC121 + OSM (magenta-ish)", (220, 50, 190)),
    ]
    return np.clip(rgb, 0.0, 1.0), legend


def main() -> int:
    root = _ensure_import_path()
    ap = argparse.ArgumentParser(
        description="C_OtherCombustion bbox PNGs: stationary S/L/X, off-road CLC composite, PM10/NOx weights."
    )
    ap.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        default=None,
        help=f"WGS84 bbox (default: {DEFAULT_OTHERC_BBOX_WGS84}).",
    )
    ap.add_argument("--root", type=Path, default=root)
    ap.add_argument("--paths-yaml", type=Path, default=None)
    ap.add_argument("--sector-yaml", type=Path, default=None)
    ap.add_argument(
        "--weight-tif",
        type=Path,
        default=None,
        help="C_OtherCombustion weights GeoTIFF (default: OUTPUT/Proxy_weights/C_OtherCombustion/othercombustion_areasource.tif).",
    )
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--country", default="EL", help="NUTS CNTR_CODE for reference window (e.g. EL).")
    ap.add_argument("--max-width", type=int, default=1400)
    ap.add_argument("--max-height", type=int, default=1200)
    ap.add_argument("--pad-deg", type=float, default=0.0)
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--cams-nc", type=Path, default=None)
    ap.add_argument("--skip-cams-grid", action="store_true")
    ap.add_argument("--no-basemap", action="store_true")
    ap.add_argument(
        "--basemap",
        type=str,
        default="osm",
        help="Non-satellite basemap: osm, carto-positron, carto-voyager, carto-dark (contextily).",
    )
    ap.add_argument("--basemap-zoom-adjust", type=int, default=None)
    ap.add_argument(
        "--classes",
        type=str,
        default="R_FIREPLACE,C_BOILER_AUT",
        help="Comma-separated MODEL_CLASSES keys for S/L/X exports.",
    )
    ap.add_argument(
        "--pollutants",
        type=str,
        default="pm10,nox",
        help="Comma-separated output keys for weight PNGs (must exist as bands in weight GeoTIFF).",
    )
    args = ap.parse_args()

    bbox = tuple(args.bbox) if args.bbox is not None else DEFAULT_OTHERC_BBOX_WGS84
    west, south, east, north = (float(x) for x in bbox)
    if west >= east or south >= north:
        print("ERROR: require west < east and south < north.", file=sys.stderr)
        return 1

    paths_yaml = args.paths_yaml or (args.root / "PROXY" / "config" / "paths.yaml")
    sector_yaml = args.sector_yaml or (args.root / "PROXY" / "config" / "sectors" / "othercombustion.yaml")
    wt_default = args.root / "OUTPUT" / "Proxy_weights" / "C_OtherCombustion" / "othercombustion_areasource.tif"
    weight_tif = args.weight_tif or wt_default
    wt = weight_tif if weight_tif.is_absolute() else args.root / weight_tif

    for label, pth in [("paths.yaml", paths_yaml), ("sector YAML", sector_yaml), ("weight GeoTIFF", wt)]:
        if not pth.is_file():
            print(f"ERROR: {label} not found: {pth}", file=sys.stderr)
            return 1

    import yaml

    from PROXY.core.dataloaders import load_path_config
    from PROXY.core.dataloaders.discovery import discover_cams_emissions, discover_corine
    from PROXY.core.dataloaders import resolve_path as resolve_p

    pc = load_path_config(Path(paths_yaml))
    path_cfg = pc.resolved
    path_cfg["proxy_common"]["corine_tif"] = str(discover_corine(args.root, Path(pc.require("proxy_common", "corine_tif"))))
    path_cfg["emissions"]["cams_2019_nc"] = str(
        discover_cams_emissions(args.root, Path(pc.require("emissions", "cams_2019_nc")))
    )

    with sector_yaml.open(encoding="utf-8") as f:
        sector_cfg = yaml.safe_load(f)
    if not isinstance(path_cfg, dict) or not isinstance(sector_cfg, dict):
        print("ERROR: YAML must parse to mappings.", file=sys.stderr)
        return 1

    out_dir = (args.out_dir.resolve() if args.out_dir else Path.cwd().resolve())
    out_dir.mkdir(parents=True, exist_ok=True)

    merged, ref = _merge_other_combustion_cfg(args.root, path_cfg, sector_cfg, str(args.country))

    from PROXY.sectors.C_OtherCombustion.constants import CLASS_TO_INDEX
    from PROXY.sectors.C_OtherCombustion.x_builder.appliance_proxy_config import load_appliance_proxy_from_rules_yaml
    from PROXY.sectors.C_OtherCombustion.x_builder.appliance_proxy_stack import (
        _compute_load,
        _compute_stock,
        _other_mask,
        _u_rural_mask,
    )
    from PROXY.sectors.C_OtherCombustion.x_builder.stack import load_and_build_fields
    from PROXY.sectors.C_OtherCombustion.alpha_beta import load_merged_c_other_profile, offroad_rules_dict
    from PROXY.sectors._shared.gnfr_groups import load_industry_osm_all_layers
    from PROXY.sectors.C_OtherCombustion.osm_commercial_mask import filter_osm_gdf_by_rules, rasterize_osm_binary_mask
    from PROXY.sectors.C_OtherCombustion.x_builder.corine_morphology import morphology_masks_from_clc
    from PROXY.core.dataloaders import warp_band_to_ref
    from rasterio.warp import Resampling

    rules_path = Path(merged["paths"]["ceip_rules_yaml"])
    appliance_doc = load_appliance_proxy_from_rules_yaml(rules_path)
    merged_prof = load_merged_c_other_profile(args.root)
    off_cfg = offroad_rules_dict(merged_prof)

    fields = load_and_build_fields(args.root, merged, ref)
    X = fields["X"]
    clc_l3 = fields["clc_l3"]
    H, W, K = X.shape
    morph = merged["morphology"]
    u111, u112, u121 = morphology_masks_from_clc(
        clc_l3,
        urban_111=int(morph["urban_111"]),
        urban_112=int(morph["urban_112"]),
        urban_121=int(morph["urban_121"]),
    )

    paths_m = merged["paths"]
    hm = paths_m["hotmaps"]
    p_res = resolve_p(args.root, hm["heat_res"])
    p_nres = resolve_p(args.root, hm["heat_nonres"])
    p_hdd = resolve_p(args.root, hm["hdd_curr"])
    p_pop = resolve_p(args.root, Path(str(paths_m["population_tif"])))
    p_ghs = resolve_p(args.root, Path(str(paths_m["ghsl_smod_tif"])))
    H_res = warp_band_to_ref(p_res, ref, resampling=Resampling.bilinear).astype(np.float32)
    H_nres = warp_band_to_ref(p_nres, ref, resampling=Resampling.bilinear).astype(np.float32)
    hdd = warp_band_to_ref(p_hdd, ref, resampling=Resampling.bilinear).astype(np.float32)
    pop = warp_band_to_ref(p_pop, ref, resampling=Resampling.bilinear, band=1).astype(np.float32)
    ghs_smod = warp_band_to_ref(p_ghs, ref, resampling=Resampling.nearest, band=1).astype(np.float32)
    u_rural = _u_rural_mask(
        ghs_smod,
        clc_l3,
        urban_111=int(morph["urban_111"]),
        urban_112=int(morph["urban_112"]),
        urban_121=int(morph["urban_121"]),
        ghs_rural_classes=frozenset(int(x) for x in (appliance_doc.get("ghs_rural_classes") or [11, 12, 13])),
    )
    other = _other_mask(u111, u112, u121)
    eps = float(appliance_doc.get("epsilon", 1.0e-12))

    osm_hit = np.zeros((H, W), dtype=np.float64)
    osp = paths_m.get("osm_other_combustion_gpkg")
    if osp:
        op = resolve_p(args.root, Path(str(osp)))
        if op.is_file():
            gdf = load_industry_osm_all_layers(op)
            om_rules = off_cfg.get("osm_commercial") or {}
            filt = filter_osm_gdf_by_rules(gdf, om_rules if isinstance(om_rules, dict) else {})
            crs_s = str(ref["crs"])
            osm_hit = rasterize_osm_binary_mask(
                filt,
                transform=ref["transform"],
                height=H,
                width=W,
                crs=crs_s,
            ).astype(np.float64)

    from PROXY.visualization._mapbuilder import (
        build_cams_area_grid_geojson_for_view,
        clip_rgba_to_cams_mask,
        compute_view_context,
        pick_band_by_pollutant,
        pick_first_positive_band,
        resolve_under_root,
        weight_rgba_percentile,
    )
    from PROXY.visualization.cams_grid import cams_cell_id_grid
    from PROXY.visualization.overlay_utils import read_weight_wgs84_only, scalar_to_rgba
    from PROXY.core.cams.mask import cams_gnfr_country_source_mask
    import xarray as xr

    wt_resolved = resolve_under_root(wt, args.root)
    view = compute_view_context(
        wt_resolved,
        pad_deg=float(args.pad_deg),
        max_width=int(args.max_width),
        max_height=int(args.max_height),
        override_bbox=(west, south, east, north),
    )
    gw, gh = int(view.gw), int(view.gh)
    print(
        f"View {gw}x{gh} | bbox [{view.west:.4f}, {view.south:.4f}, {view.east:.4f}, {view.north:.4f}]",
        flush=True,
    )

    grid_fc: dict | None = None
    cams_ds: xr.Dataset | None = None
    m_area = None
    cams_nc_file: Path | None = None
    iso3 = str(merged["country"]["cams_iso3"]).strip().upper()
    cams_block = merged.get("cams") or {}
    gnfr = str(cams_block.get("gnfr", "C"))
    domain_bbox = cams_block.get("domain_bbox_wgs84")
    domain_bbox_t = tuple(float(x) for x in domain_bbox) if domain_bbox else None
    stypes = tuple(cams_block.get("source_types") or ("area",))

    if not args.skip_cams_grid:
        nc_rel = (path_cfg.get("emissions") or {}).get("cams_2019_nc")
        nc_path = None
        if args.cams_nc is not None:
            nc_path = args.cams_nc if args.cams_nc.is_absolute() else args.root / args.cams_nc
        elif nc_rel:
            nc_path = discover_cams_emissions(args.root, resolve_p(args.root, Path(str(nc_rel))))
            nc_path = nc_path if nc_path.is_absolute() else args.root / nc_path
        if nc_path is not None and nc_path.is_file():
            cams_nc_file = nc_path.resolve()
            cams_ds = xr.open_dataset(nc_path, engine="netcdf4")
            try:
                m_area = cams_gnfr_country_source_mask(
                    cams_ds,
                    iso3,
                    gnfr=gnfr,
                    source_types=stypes,
                    domain_bbox_wgs84=domain_bbox_t,
                )
                grid_fc = build_cams_area_grid_geojson_for_view(cams_ds, m_area, view)
            except Exception as exc:
                print(f"WARNING: CAMS grid / mask failed ({exc}).", file=sys.stderr)
                cams_ds.close()
                cams_ds = None
                m_area = None
                grid_fc = None
                cams_nc_file = None

    try:
        import contextily  # noqa: F401
    except ImportError:
        contextily = None
    use_basemap = (not args.no_basemap) and (contextily is not None)
    if not args.no_basemap and contextily is None:
        print("WARNING: install contextily for OSM basemap.", file=sys.stderr)

    basemap_src: object | None = None
    basemap_attr = ""
    if use_basemap:
        try:
            basemap_src, basemap_attr = _basemap_ctx_source(str(args.basemap))
        except ValueError as exc:
            print(f"WARNING: {exc} Using osm.", file=sys.stderr)
            import contextily as ctx

            basemap_src = ctx.providers.OpenStreetMap.Mapnik
            basemap_attr = "OpenStreetMap Mapnik"

    from rasterio.transform import xy as transform_xy

    cell_id = None
    if cams_ds is not None and m_area is not None:
        rows, cols = np.indices((gh, gw))
        xs, ys = transform_xy(view.dst_t, rows + 0.5, cols + 0.5, offset="center")
        cell_id = np.asarray(
            cams_cell_id_grid(
                np.asarray(xs, dtype=np.float64),
                np.asarray(ys, dtype=np.float64),
                cams_ds,
                m_area,
            ),
            dtype=np.int64,
        ).reshape(gh, gw)

    _ensure_otherc_display_colormaps()

    def _scalar_png(
        z: np.ndarray,
        title: str,
        fname: str,
        *,
        field: str,
        cbar_label: str,
    ) -> None:
        zw = _warp_ref_to_wgs84(
            z, ref, west=view.west, south=view.south, east=view.east, north=view.north, width=gw, height=gh
        )
        f = field.strip().upper()
        if f == "S":
            cmap_name = "otherc_browns"
        elif f == "L":
            cmap_name = "otherc_white_red"
        elif f == "X":
            cmap_name = "YlOrRd"
        else:
            cmap_name = "viridis"

        if cell_id is not None:
            base_valid = np.isfinite(zw) & (zw > 0)
            z01, valid_pc = _per_cell_robust_stretch_01(
                zw, cell_id, base_valid=base_valid, lo_pct=2.0, hi_pct=98.0, min_n=4
            )
            rgba = scalar_to_rgba(
                zw,
                colour_mode="global",
                cmap_name=cmap_name,
                hide_zero=True,
                nodata_val=None,
                z_precomputed_01=z01,
                valid_precomputed=valid_pc,
            )
            rgba = _apply_sqrt_alpha_fade_on_valid(rgba, z01, valid_pc)
            cbar_d: dict[str, object] = {
                "vmin": 0.0,
                "vmax": 1.0,
                "cmap": cmap_name,
                "percent_ticks": True,
                "label": f"{cbar_label} (within CAMS cell; 2–98% stretch, below 2% transparent)",
            }
        else:
            finite = np.isfinite(zw) & (zw > 0)
            if not np.any(finite):
                print(f"WARNING: empty layer {fname}", file=sys.stderr)
                return
            z01, valid = _global_robust_stretch_01(zw, base_valid=finite, lo_pct=2.0, hi_pct=98.0)
            if not np.any(valid):
                print(f"WARNING: empty layer {fname}", file=sys.stderr)
                return
            rgba = scalar_to_rgba(
                zw,
                colour_mode="global",
                cmap_name=cmap_name,
                hide_zero=True,
                nodata_val=None,
                z_precomputed_01=z01,
                valid_precomputed=valid,
            )
            rgba = _apply_sqrt_alpha_fade_on_valid(rgba, z01, valid)
            cbar_d = {
                "vmin": 0.0,
                "vmax": 1.0,
                "cmap": cmap_name,
                "percent_ticks": True,
                "label": f"{cbar_label} (global 2–98% stretch; below 2% transparent)",
            }

        if not np.any(rgba[..., 3] > 0):
            print(f"WARNING: empty layer {fname}", file=sys.stderr)
            return

        if cams_ds is not None and m_area is not None and cams_nc_file is not None:
            rgba = clip_rgba_to_cams_mask(
                rgba,
                cams_nc_path=cams_nc_file,
                m_area=m_area,
                ds=cams_ds,
                view=view,
            )

        if use_basemap and basemap_src is not None:
            try:
                rgba = _composite_rgba_over_osm(
                    rgba,
                    view.dst_t,
                    (gh, gw),
                    view.west,
                    view.south,
                    view.east,
                    view.north,
                    zoom_adjust=args.basemap_zoom_adjust,
                    source=basemap_src,
                )
            except Exception as exc:
                print(f"WARNING: basemap failed ({exc})", file=sys.stderr)
        _save_png(
            rgba,
            title=title,
            west=view.west,
            south=view.south,
            east=view.east,
            north=view.north,
            out_path=out_dir / fname,
            dpi=int(args.dpi),
            grid_fc=grid_fc,
            legend_entries=None,
            colorbar_spec=cbar_d,
        )

    class_keys = [c.strip() for c in str(args.classes).split(",") if c.strip()]
    for cls in class_keys:
        if cls not in CLASS_TO_INDEX:
            print(f"WARNING: skip unknown class {cls!r}", file=sys.stderr)
            continue
        spec = appliance_doc.get(cls)
        if not isinstance(spec, dict):
            print(f"WARNING: no appliance_proxy spec for {cls}", file=sys.stderr)
            continue
        safe = cls.lower()
        S = _compute_stock(spec["stock"], pop, H_nres, u111, u112, u121, u_rural, other)
        L = _compute_load(spec["load"], H_res, H_nres, hdd, eps)
        Xk = np.asarray(X[:, :, CLASS_TO_INDEX[cls]], dtype=np.float32)
        _scalar_png(
            np.asarray(S, dtype=np.float64),
            f"GNFR C — {cls} stock S (warped)",
            f"otherc_bbox_{safe}_stock_S.png",
            field="S",
            cbar_label=f"{cls} S",
        )
        _scalar_png(
            np.asarray(L, dtype=np.float64),
            f"GNFR C — {cls} load L (warped)",
            f"otherc_bbox_{safe}_load_L.png",
            field="L",
            cbar_label=f"{cls} L",
        )
        _scalar_png(
            np.asarray(Xk, dtype=np.float64),
            f"GNFR C — {cls} X = S*L (warped)",
            f"otherc_bbox_{safe}_X_band.png",
            field="X",
            cbar_label=f"{cls} X",
        )

    rgb_ref, leg = _offroad_rgb_ref(clc_l3, pop, osm_hit, off_cfg)
    rgb_w = _warp_ref_rgb_to_wgs84(rgb_ref, ref, west=view.west, south=view.south, east=view.east, north=view.north, width=gw, height=gh)
    rgb_u8 = (np.clip(rgb_w, 0.0, 1.0) * 255.0).astype(np.uint8)
    sig = np.clip(np.max(rgb_w, axis=2), 0.0, 1.0)
    pos = sig > 1e-9
    if np.any(pos):
        p2 = float(np.percentile(sig[pos], 2.0))
        p98 = float(np.percentile(sig[pos], 98.0))
        span = max(p98 - p2, 1e-6)
        a = np.zeros((gh, gw), dtype=np.float64)
        m = sig > p2
        a[m] = np.clip((sig[m] - p2) / span, 0.0, 1.0) * 255.0
    else:
        a = np.zeros((gh, gw), dtype=np.float64)
    rgba_or = np.zeros((gh, gw, 4), dtype=np.uint8)
    rgba_or[..., 0] = rgb_u8[..., 0]
    rgba_or[..., 1] = rgb_u8[..., 1]
    rgba_or[..., 2] = rgb_u8[..., 2]
    rgba_or[..., 3] = a.astype(np.uint8)
    if cams_ds is not None and m_area is not None and cams_nc_file is not None:
        rgba_or = clip_rgba_to_cams_mask(
            rgba_or,
            cams_nc_path=cams_nc_file,
            m_area=m_area,
            ds=cams_ds,
            view=view,
        )
    if use_basemap and basemap_src is not None:
        try:
            rgb_out = _composite_rgba_over_osm(
                rgba_or,
                view.dst_t,
                (gh, gw),
                view.west,
                view.south,
                view.east,
                view.north,
                zoom_adjust=args.basemap_zoom_adjust,
                source=basemap_src,
            )
        except Exception as exc:
            print(f"WARNING: off-road basemap failed ({exc})", file=sys.stderr)
            gray_u8 = np.full((gh, gw, 3), int(0.933 * 255.0), dtype=np.uint8)
            rgb_out = _alpha_composite_rgb_under_rgba(gray_u8, rgba_or)
    else:
        gray_u8 = np.full((gh, gw, 3), int(0.933 * 255.0), dtype=np.uint8)
        rgb_out = _alpha_composite_rgb_under_rgba(gray_u8, rgba_or)

    _save_png(
        rgb_out,
        title="GNFR C — off-road proxy ingredients (forest CLC / CLC112+pop / CLC121+OSM, ref-grid logic)",
        west=view.west,
        south=view.south,
        east=view.east,
        north=view.north,
        out_path=out_dir / "otherc_bbox_offroad_corine_osm_composite.png",
        dpi=int(args.dpi),
        grid_fc=grid_fc,
        legend_entries=leg,
        colorbar_spec=None,
    )

    weights_per_cell = cams_ds is not None and m_area is not None
    area_proxy = merged.get("visualization") or sector_cfg.get("visualization") or {}

    def _weight_job(pol_key: str, display: str, fn: str) -> None:
        band = pick_band_by_pollutant(
            wt_resolved,
            {**area_proxy, "visualization_pollutant": pol_key, "visualization_weight_band": 999},
            strip_prefixes=(),
            sector_cfg=sector_cfg,
        )
        band, _ = pick_first_positive_band(
            wt_resolved,
            band,
            empty_message=f"No positive weights for {display}; using band anyway.",
        )
        stk = read_weight_wgs84_only(
            wt_resolved,
            west=view.west,
            south=view.south,
            east=view.east,
            north=view.north,
            display_width=gw,
            display_height=gh,
            weight_band=int(band),
        )
        w_arr = stk["weight_wgs84"]
        w_nd = stk["weight_nodata"]
        nd = float(w_nd) if w_nd is not None else None
        if weights_per_cell and cams_ds is not None and m_area is not None and cell_id is not None:
            finite = np.isfinite(w_arr)
            if w_nd is not None:
                finite &= w_arr != float(w_nd)
            base_valid = finite & (w_arr > 0)
            z01, valid_pc = _per_cell_robust_stretch_01(
                w_arr, cell_id, base_valid=base_valid, lo_pct=2.0, hi_pct=98.0, min_n=4
            )
            rgba = scalar_to_rgba(
                w_arr,
                colour_mode="global",
                cmap_name="YlOrRd",
                hide_zero=True,
                nodata_val=nd,
                z_precomputed_01=z01,
                valid_precomputed=valid_pc,
            )
            rgba = _apply_sqrt_alpha_fade_on_valid(rgba, z01, valid_pc)
            cbar_d: dict[str, object] = {
                "vmin": 0.0,
                "vmax": 1.0,
                "cmap": "YlOrRd",
                "percent_ticks": True,
                "label": f"{display} weight (within CAMS cell; 2–98% stretch, below 2% transparent)",
            }
        else:
            finite = np.isfinite(w_arr)
            if w_nd is not None:
                finite &= w_arr != float(w_nd)
            base_valid = finite & (w_arr > 0)
            z01, valid_pc = _global_robust_stretch_01(w_arr, base_valid=base_valid, lo_pct=2.0, hi_pct=98.0)
            if not np.any(valid_pc):
                rgba = weight_rgba_percentile(w_arr, w_nodata=w_nd, cmap="YlOrRd")
                cbar_d = {
                    "vmin": float(np.nanpercentile(w_arr[np.isfinite(w_arr) & (w_arr > 0)], 2))
                    if np.any(w_arr > 0)
                    else 0.0,
                    "vmax": float(np.nanpercentile(w_arr[np.isfinite(w_arr) & (w_arr > 0)], 98))
                    if np.any(w_arr > 0)
                    else 1.0,
                    "cmap": "YlOrRd",
                    "label": f"{display} weight (global percentile; add CAMS for 0–100%/cell)",
                }
            else:
                rgba = scalar_to_rgba(
                    w_arr,
                    colour_mode="global",
                    cmap_name="YlOrRd",
                    hide_zero=True,
                    nodata_val=nd,
                    z_precomputed_01=z01,
                    valid_precomputed=valid_pc,
                )
                rgba = _apply_sqrt_alpha_fade_on_valid(rgba, z01, valid_pc)
                cbar_d = {
                    "vmin": 0.0,
                    "vmax": 1.0,
                    "cmap": "YlOrRd",
                    "percent_ticks": True,
                    "label": f"{display} weight (global 2–98% stretch; below 2% transparent)",
                }
        if cams_ds is not None and m_area is not None and cams_nc_file is not None:
            rgba = clip_rgba_to_cams_mask(
                rgba,
                cams_nc_path=cams_nc_file,
                m_area=m_area,
                ds=cams_ds,
                view=view,
            )
        if use_basemap and basemap_src is not None:
            try:
                rgba = _composite_rgba_over_osm(
                    rgba,
                    view.dst_t,
                    (gh, gw),
                    view.west,
                    view.south,
                    view.east,
                    view.north,
                    zoom_adjust=args.basemap_zoom_adjust,
                    source=basemap_src,
                )
            except Exception:
                pass
        _save_png(
            rgba,
            title=f"GNFR C — weights ({display}, band {band})",
            west=view.west,
            south=view.south,
            east=view.east,
            north=view.north,
            out_path=out_dir / fn,
            dpi=int(args.dpi),
            grid_fc=grid_fc,
            legend_entries=None,
            colorbar_spec=cbar_d,
        )

    pols = [p.strip() for p in str(args.pollutants).split(",") if p.strip()]
    for pol in pols:
        safe = pol.lower().replace(".", "_")
        try:
            _weight_job(pol, pol.upper(), f"otherc_bbox_weights_{safe}.png")
        except Exception as exc:
            print(f"WARNING: weight export failed for {pol}: {exc}", file=sys.stderr)

    if cams_ds is not None:
        cams_ds.close()

    if use_basemap:
        print(
            f"\nBasemap: {basemap_attr} — see Contextily / provider terms (OSM: https://www.openstreetmap.org/copyright).",
            file=sys.stderr,
        )
    print(f"\nDone. Output: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
