#!/usr/bin/env python3
"""B_Industry proxy PNG exports for a WGS84 bbox.

Combined **G1..G4** maps: **CORINE** (dark green), **OSM** (red), **overlap** (dark blue),
**P_pop** (yellow, very low opacity on top). CLC/OSM strengths are span‑normalized separately,
then drawn with explicit hues (no channel mixing). Basemap is slightly dimmed for contrast.

Exports GNFR B area weights for **CO**, **NMVOC**, and **SOx**. When CAMS NetCDF loads, weights are
always drawn **per CAMS cell** (0–100% within cell); pixels below the **2nd percentile of
positive weights inside each CAMS cell** are hidden. Without CAMS, a global percentile
colormap is used instead.

Optional OSM basemap (EPSG:3857 tiles warped to the view grid) and CAMS GNFR B grid outlines.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_root_boot = Path(__file__).resolve().parents[2]
if str(_root_boot) not in sys.path:
    sys.path.insert(0, str(_root_boot))

from PROXY.tools.waste_proxy_bbox_images import (
    _alpha_composite_rgb_under_rgba,
    _composite_rgba_over_osm,
    _percentile_vmin_vmax,
    _save_png,
)

# Default view: slightly NW of the shared waste/Attica preview; east edge −0.1° vs waste default.
DEFAULT_INDUSTRY_BBOX_WGS84 = (23.45, 37.90, 23.85, 38.13)

# Display colours (RGB 0–255): CORINE-only / OSM-only / both / population tint.
_RGB_CLC_ONLY = (16, 104, 42)
_RGB_OSM_ONLY = (214, 42, 48)
_RGB_CLC_OSM_BOTH = (22, 52, 138)
_RGB_POP = (250, 216, 28)

_POP_TITLE = "Industry · P_pop (z-score)"


def _industry_corine_scalar_title(gid: str, groups_raw: dict[str, object]) -> str | None:
    spec = groups_raw.get(str(gid)) or {}
    yaml_clc = [int(x) for x in (spec.get("corine_classes") or [])]
    if not yaml_clc:
        return None
    cls_str = ",".join(str(c) for c in yaml_clc)
    return f"Industry · CORINE {cls_str} ({gid})"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_import_path() -> Path:
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _distinct_signal_rgba(
    z: object,
    rgb: tuple[int, int, int],
    *,
    lo_pct: float = 2.0,
    hi_pct: float = 98.0,
    max_alpha: float = 0.8,
    gamma: float = 1.0,
    alpha_power: float = 1.0,
    min_alpha_u8: int = 0,
    min_alpha_signal: float = 0.12,
    positive_only: bool = True,
) -> object:
    """HxWx4 uint8 RGBA: fixed hue, alpha from normalized signal.

    When ``positive_only`` is False (e.g. P_pop z-scores), uses all finite pixels for the
    percentile stretch so negatives are not dropped entirely; pairs well with a higher
    ``lo_pct`` on export so weak values stay transparent.
    """
    import numpy as np

    a = np.asarray(z, dtype=np.float64)
    h, w = a.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    if positive_only:
        m = np.isfinite(a) & (a > 0)
    else:
        m = np.isfinite(a)
    if not np.any(m):
        return rgba
    lo = float(np.percentile(a[m], lo_pct))
    hi = float(np.percentile(a[m], hi_pct))
    t = np.zeros((h, w), dtype=np.float64)
    # Binary / flat CORINE scores are often exactly constant on all positives (e.g. only 1.0).
    # Then lo == hi and (a-lo)/(hi-lo) becomes 0 if hi is nudged by epsilon — wipes the layer.
    if hi <= lo:
        t[m] = 1.0
    else:
        t[m] = np.clip((a[m] - lo) / (hi - lo), 0.0, 1.0)
    if float(gamma) != 1.0:
        g = float(gamma)
        if g <= 0.0:
            g = 1.0
        t[m] = np.power(t[m], g)
    t_a = np.clip(t, 0.0, 1.0)
    if float(alpha_power) != 1.0:
        ap = float(alpha_power)
        if ap <= 0.0:
            ap = 1.0
        t_a = np.power(t_a, ap)
    r_u8, g_u8, b_u8 = (np.uint8(int(c)) for c in rgb)
    alpha = np.clip(t_a * max_alpha * 255.0, 0.0, 255.0).astype(np.uint8)
    if int(min_alpha_u8) > 0:
        thr = float(min_alpha_signal)
        boost = t_a >= thr
        alpha = np.where(boost, np.maximum(alpha, np.uint8(int(min_alpha_u8))), alpha)
    rgba[..., 0] = r_u8
    rgba[..., 1] = g_u8
    rgba[..., 2] = b_u8
    rgba[..., 3] = alpha
    return rgba


def _signal_strength_t(
    z: object,
    *,
    lo_pct: float = 2.0,
    hi_pct: float = 98.0,
    gamma: float = 1.0,
    alpha_power: float = 1.0,
    t_floor: float = 0.0,
    t_floor_where_above: float = 0.0,
    positive_only: bool = True,
) -> object:
    """Return HxW float64 strengths in [0, 1] (same normalization core as ``_distinct_signal_rgba``)."""
    import numpy as np

    a = np.asarray(z, dtype=np.float64)
    h, w = a.shape
    out = np.zeros((h, w), dtype=np.float64)
    if positive_only:
        m = np.isfinite(a) & (a > 0)
    else:
        m = np.isfinite(a)
    if not np.any(m):
        return out
    lo = float(np.percentile(a[m], lo_pct))
    hi = float(np.percentile(a[m], hi_pct))
    if hi <= lo:
        out[m] = 1.0
    else:
        out[m] = np.clip((a[m] - lo) / (hi - lo), 0.0, 1.0)
    if float(gamma) != 1.0:
        g = float(gamma)
        if g <= 0.0:
            g = 1.0
        out[m] = np.power(out[m], g)
    t_a = np.clip(out, 0.0, 1.0)
    if float(alpha_power) != 1.0:
        ap = float(alpha_power)
        if ap <= 0.0:
            ap = 1.0
        t_a = np.power(t_a, ap)
    if float(t_floor) > 0.0 and float(t_floor_where_above) >= 0.0:
        thr = float(t_floor_where_above)
        fl = float(t_floor)
        bump = t_a >= thr
        t_a = np.where(bump & (t_a > 0), np.maximum(t_a, fl), t_a)
    return np.clip(t_a, 0.0, 1.0)


def _composite_rgb_industry_clc_osm(
    base_rgb_uint8: object,
    t_clc: object,
    t_osm: object,
    *,
    cover_scale: float = 0.95,
    rgb_clc: tuple[int, int, int] = _RGB_CLC_ONLY,
    rgb_osm: tuple[int, int, int] = _RGB_OSM_ONLY,
    rgb_both: tuple[int, int, int] = _RGB_CLC_OSM_BOTH,
    debug_stats: dict[str, float] | None = None,
) -> object:
    """Composite span‑normalized CLC/OSM strengths with fixed hues (clear separation).

    * CORINE only → dark green
    * OSM only → red
    * Both → dark blue

    Blend: ``out = base * (1 - a) + solid_rgb * a`` with ``a = cover_scale * strength``.
    """
    import numpy as np

    base_f = np.asarray(base_rgb_uint8, dtype=np.float32)
    if base_f.ndim != 3 or base_f.shape[2] != 3:
        raise ValueError("base must be HxWx3")
    if float(np.nanmax(base_f)) > 1.5:
        base_f = base_f / 255.0
    base_f = np.clip(base_f, 0.0, 1.0)

    tc = np.clip(np.asarray(t_clc, dtype=np.float64), 0.0, 1.0)
    to = np.clip(np.asarray(t_osm, dtype=np.float64), 0.0, 1.0)
    if tc.shape != to.shape:
        raise ValueError("CLC/OSM strength grids must match")

    def _span01(t: np.ndarray) -> np.ndarray:
        mx = float(np.nanmax(t))
        if mx <= 1e-15:
            return t
        return np.clip(t / mx, 0.0, 1.0)

    tc = _span01(tc)
    to = _span01(to)

    eps = 1e-6
    has_c = tc > eps
    has_o = to > eps
    only_c = has_c & ~has_o
    only_o = has_o & ~has_c
    both = has_c & has_o

    rc = np.array(rgb_clc, dtype=np.float64) / 255.0
    ro = np.array(rgb_osm, dtype=np.float64) / 255.0
    rb = np.array(rgb_both, dtype=np.float64) / 255.0

    sb = np.maximum(tc, to)
    a = np.where(only_c, tc, np.where(only_o, to, np.where(both, sb, 0.0)))
    k = float(np.clip(cover_scale, 0.0, 1.0))
    a = np.clip(a * k, 0.0, 1.0)

    solid = np.stack(
        [
            only_c * rc[0] + only_o * ro[0] + both * rb[0],
            only_c * rc[1] + only_o * ro[1] + both * rb[1],
            only_c * rc[2] + only_o * ro[2] + both * rb[2],
        ],
        axis=-1,
    )
    out = base_f * (1.0 - a[..., np.newaxis]) + solid * a[..., np.newaxis]

    if debug_stats is not None:
        debug_stats["n_px"] = float(tc.size)
        debug_stats["n_only_clc"] = float(np.count_nonzero(only_c))
        debug_stats["n_only_osm"] = float(np.count_nonzero(only_o))
        debug_stats["n_both"] = float(np.count_nonzero(both))
        debug_stats["n_neither"] = float(np.count_nonzero(~has_c & ~has_o))
        debug_stats["tc_max"] = float(np.nanmax(tc))
        debug_stats["to_max"] = float(np.nanmax(to))

    return (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)


def _dim_rgb_uint8(rgb: object, factor: float) -> object:
    """Darken basemap RGB so saturated overlays remain readable."""
    import numpy as np

    x = np.asarray(rgb, dtype=np.float32)
    if x.ndim != 3 or x.shape[2] != 3:
        return rgb
    f = float(factor)
    if f >= 0.999:
        return np.clip(x, 0.0, 255.0).astype(np.uint8)
    return np.clip(x * f, 0.0, 255.0).astype(np.uint8)


def _basemap_rgb_underlay(
    *,
    gh: int,
    gw: int,
    dst_transform,
    west: float,
    south: float,
    east: float,
    north: float,
    zoom_adjust: int | None,
    use_basemap: bool,
) -> object:
    """RGB background: OSM tiles if ``use_basemap`` else light gray."""
    import numpy as np

    if use_basemap:
        transparent = np.zeros((gh, gw, 4), dtype=np.uint8)
        return _composite_rgba_over_osm(
            transparent,
            dst_transform,
            (gh, gw),
            west,
            south,
            east,
            north,
            zoom_adjust=zoom_adjust,
        )
    gray = 238
    return np.full((gh, gw, 3), gray, dtype=np.uint8)


def _scalar_stats_lines(name: str, arr: object, *, pos_thr: float | None = 1e-12) -> list[str]:
    """Human-readable one-line stats for float rasters (stderr debug)."""
    import numpy as np

    a = np.asarray(arr, dtype=np.float64).ravel()
    fin = np.isfinite(a)
    n = int(a.size)
    if n == 0:
        return [f"  {name}: empty"]
    n_fin = int(np.count_nonzero(fin))
    if pos_thr is None:
        pos = fin
    else:
        pos = fin & (a > float(pos_thr))
    n_pos = int(np.count_nonzero(pos))
    mx = float(np.nanmax(a)) if n_fin else float("nan")
    mn = float(np.nanmin(a[fin])) if n_fin else float("nan")
    mean_pos = float(np.mean(a[pos])) if n_pos else float("nan")
    p95 = float(np.percentile(a[fin], 95)) if n_fin else float("nan")
    thr_note = "finite" if pos_thr is None else f"pos>{float(pos_thr):g}"
    return [
        f"  {name}: finite={n_fin}/{n}  {thr_note}={n_pos}  "
        f"min={mn:.6g} max={mx:.6g} mean_sel={mean_pos:.6g} p95_fin={p95:.6g}"
    ]


def _save_industry_group_combined(
    osm: object,
    clc: object,
    p_pop: object,
    *,
    west: float,
    south: float,
    east: float,
    north: float,
    out_path: Path,
    title: str,
    group_id: str,
    dpi: int,
    grid_fc: dict | None,
    dst_transform,
    gh: int,
    gw: int,
    use_basemap: bool,
    basemap_zoom_adjust: int | None,
    basemap_rgb_dim: float = 0.72,
    flat_basemap_rgb_dim: float = 0.92,
    debug_combined_layers: bool = False,
) -> None:
    import numpy as np

    base = _basemap_rgb_underlay(
        gh=gh,
        gw=gw,
        dst_transform=dst_transform,
        west=west,
        south=south,
        east=east,
        north=north,
        zoom_adjust=basemap_zoom_adjust,
        use_basemap=use_basemap,
    )
    dim = float(basemap_rgb_dim) if use_basemap else float(flat_basemap_rgb_dim)
    dim = float(np.clip(dim, 0.15, 1.0))
    base = _dim_rgb_uint8(base, dim)
    # Explicit hues: dark green / red / dark blue (see ``_composite_rgb_industry_clc_osm``).
    t_clc = _signal_strength_t(
        clc,
        lo_pct=0.5,
        hi_pct=99.5,
        gamma=0.52,
        alpha_power=1.05,
        t_floor=0.10,
        t_floor_where_above=0.05,
        positive_only=True,
    )
    t_osm = _signal_strength_t(
        osm,
        lo_pct=1.0,
        hi_pct=99.0,
        gamma=0.78,
        alpha_power=1.08,
        t_floor=0.09,
        t_floor_where_above=0.06,
        positive_only=True,
    )
    dbg: dict[str, float] = {}

    if debug_combined_layers:
        print(f"[debug combined] {group_id} → {out_path.name}", file=sys.stderr)
        for line in _scalar_stats_lines("CLC raw (warped scalar)", clc):
            print(line, file=sys.stderr)
        for line in _scalar_stats_lines("OSM raw (warped coverage)", osm):
            print(line, file=sys.stderr)
        for line in _scalar_stats_lines("P_pop raw", p_pop, pos_thr=None):
            print(line, file=sys.stderr)
        for line in _scalar_stats_lines("t_CLC after percentile/gamma", t_clc):
            print(line, file=sys.stderr)
        for line in _scalar_stats_lines("t_OSM after percentile/gamma", t_osm):
            print(line, file=sys.stderr)
        print(
            "  hues: CORINE=dark green  OSM=red  overlap=dark blue  POP=yellow (faint)",
            file=sys.stderr,
        )

    rgb = _composite_rgb_industry_clc_osm(
        base,
        t_clc,
        t_osm,
        cover_scale=0.95,
        rgb_clc=_RGB_CLC_ONLY,
        rgb_osm=_RGB_OSM_ONLY,
        rgb_both=_RGB_CLC_OSM_BOTH,
        debug_stats=dbg if debug_combined_layers else None,
    )
    if debug_combined_layers and dbg:
        print(
            "  CLC/OSM composite (after span-normalize): "
            f"only_CLC={int(dbg.get('n_only_clc', 0))} px  "
            f"only_OSM={int(dbg.get('n_only_osm', 0))} px  "
            f"both={int(dbg.get('n_both', 0))} px  "
            f"neither={int(dbg.get('n_neither', 0))} px  "
            f"tc_max={dbg.get('tc_max', float('nan')):.4f}  "
            f"to_max={dbg.get('to_max', float('nan')):.4f}",
            file=sys.stderr,
        )
    pop_r = _distinct_signal_rgba(
        p_pop,
        _RGB_POP,
        lo_pct=48.0,
        hi_pct=98.5,
        max_alpha=0.8,
        gamma=1.05,
        alpha_power=1.0,
        min_alpha_u8=0,
        min_alpha_signal=0.0,
        positive_only=False,
    )
    rgb = _alpha_composite_rgb_under_rgba(rgb, pop_r)

    legend = [
        (f"{group_id} CORINE (dark green)", _RGB_CLC_ONLY),
        (f"{group_id} OSM (red)", _RGB_OSM_ONLY),
        (f"{group_id} CORINE+OSM (dark blue)", _RGB_CLC_OSM_BOTH),
        (f"{group_id} P_pop (yellow, faint)", _RGB_POP),
    ]
    _save_png(
        rgb,
        title=title,
        west=west,
        south=south,
        east=east,
        north=north,
        out_path=out_path,
        dpi=dpi,
        grid_fc=grid_fc,
        legend_entries=legend,
        colorbar_spec=None,
    )


def _per_cell_drop_lowest_positive_pct(
    w_arr: object,
    cell_id: object,
    valid: object,
    w_nd: float | None,
    *,
    lo_pct: float = 2.0,
    min_pos: int = 4,
) -> object:
    """Turn off ``valid`` for pixels whose raw weight is below the ``lo_pct`` percentile of
    positives **within the same CAMS cell** (needs at least ``min_pos`` positive samples).
    """
    import numpy as np

    w = np.asarray(w_arr, dtype=np.float64)
    cid = np.asarray(cell_id)
    if cid.shape != w.shape:
        cid = cid.reshape(w.shape)
    out = np.asarray(valid, dtype=bool).copy()
    nd = float(w_nd) if w_nd is not None else None
    for c in np.unique(cid):
        if int(c) < 0:
            continue
        cm = (cid == int(c)) & out
        if not np.any(cm):
            continue
        vals = w[cm]
        pos = np.isfinite(vals) & (vals > 0)
        if nd is not None:
            pos &= vals != nd
        if int(np.count_nonzero(pos)) < int(min_pos):
            continue
        thr = float(np.percentile(vals[pos], lo_pct))
        drop_loc = np.zeros(vals.shape, dtype=bool)
        drop_loc[pos] = vals[pos] < thr
        drop_2d = np.zeros_like(out, dtype=bool)
        drop_2d[cm] = drop_loc
        out &= ~drop_2d
    return out


def main() -> int:
    root = _ensure_import_path()
    ap = argparse.ArgumentParser(
        description="Export B_Industry combined CLC/OSM/P_pop maps per group and CO/NMVOC/SOx weight maps."
    )
    ap.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        default=None,
        help=f"WGS84 bbox west south east north (default: {DEFAULT_INDUSTRY_BBOX_WGS84}).",
    )
    ap.add_argument("--root", type=Path, default=root)
    ap.add_argument("--paths-yaml", type=Path, default=None)
    ap.add_argument("--sector-yaml", type=Path, default=None)
    ap.add_argument(
        "--weight-tif",
        type=Path,
        default=None,
        help="B_Industry area weights GeoTIFF (default: OUTPUT/Proxy_weights/B_Industry/industry_areasource.tif).",
    )
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--country", default="EL")
    ap.add_argument("--max-width", type=int, default=1400)
    ap.add_argument("--max-height", type=int, default=1200)
    ap.add_argument("--pad-deg", type=float, default=0.0)
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument(
        "--cams-nc",
        type=Path,
        default=None,
        help="CAMS NetCDF (default: emissions.cams_2019_nc from paths.yaml).",
    )
    ap.add_argument("--skip-cams-grid", action="store_true")
    ap.add_argument("--no-basemap", action="store_true")
    ap.add_argument("--basemap-zoom-adjust", type=int, default=None)
    ap.add_argument(
        "--combined-basemap-dim",
        type=float,
        default=0.72,
        help="Multiply OSM basemap RGB (0.15–1) before group overlays; higher = brighter map / clearer context.",
    )
    ap.add_argument(
        "--debug-combined-layers",
        action="store_true",
        help="Stderr: per-group raw CLC/OSM/P_pop stats, normalized strengths, CLC/OSM pixel counts.",
    )
    args = ap.parse_args()

    bbox = tuple(args.bbox) if args.bbox is not None else DEFAULT_INDUSTRY_BBOX_WGS84
    west, south, east, north = (float(x) for x in bbox)
    if west >= east or south >= north:
        print("ERROR: require west < east and south < north.", file=sys.stderr)
        return 1
    cbd = float(args.combined_basemap_dim)
    if not (0.15 <= cbd <= 1.0):
        print("ERROR: --combined-basemap-dim must be between 0.15 and 1.0.", file=sys.stderr)
        return 1

    paths_yaml = args.paths_yaml or (args.root / "PROXY" / "config" / "paths.yaml")
    sector_yaml = args.sector_yaml or (args.root / "PROXY" / "config" / "sectors" / "industry.yaml")
    wt_default = args.root / "OUTPUT" / "Proxy_weights" / "B_Industry" / "industry_areasource.tif"
    weight_tif = args.weight_tif or wt_default
    wt = weight_tif if weight_tif.is_absolute() else args.root / weight_tif

    for label, pth in [
        ("paths.yaml", paths_yaml),
        ("sector YAML", sector_yaml),
        ("weight GeoTIFF", wt),
    ]:
        if not pth.is_file():
            print(f"ERROR: {label} not found: {pth}", file=sys.stderr)
            return 1

    import yaml

    with paths_yaml.open(encoding="utf-8") as f:
        path_cfg = yaml.safe_load(f)
    with sector_yaml.open(encoding="utf-8") as f:
        sector_cfg = yaml.safe_load(f)
    if not isinstance(path_cfg, dict) or not isinstance(sector_cfg, dict):
        print("ERROR: YAML must parse to mappings.", file=sys.stderr)
        return 1

    out_dir = (args.out_dir.resolve() if args.out_dir else Path.cwd().resolve())
    out_dir.mkdir(parents=True, exist_ok=True)

    from PROXY.core.ceip.loader import DEFAULT_GNFR_GROUP_ORDER
    from PROXY.core.dataloaders import resolve_path
    from PROXY.core.dataloaders.discovery import discover_cams_emissions
    from PROXY.core.cams.mask import cams_gnfr_country_source_mask
    from PROXY.sectors.B_Industry.builder import _merge_industry_pipeline_cfg
    from PROXY.visualization._mapbuilder import (
        build_cams_area_grid_geojson_for_view,
        compute_view_context,
        pick_band_by_pollutant,
        pick_first_positive_band,
        resolve_under_root,
        weight_rgba_percentile,
    )
    from PROXY.visualization.cams_grid import cams_cell_id_grid, normalize_weights_per_cams_cell
    from PROXY.visualization.industry_context import build_industry_proxy_rgba_overlays
    from PROXY.visualization.overlay_utils import read_weight_wgs84_only, scalar_to_rgba
    from rasterio.transform import xy as transform_xy
    import numpy as np
    import xarray as xr

    wt_resolved = resolve_under_root(wt, args.root)
    ind_merged = _merge_industry_pipeline_cfg(
        args.root,
        path_cfg,
        sector_cfg,
        country=str(args.country),
        output_path=wt_resolved.resolve(),
    )

    view = compute_view_context(
        wt_resolved,
        pad_deg=float(args.pad_deg),
        max_width=int(args.max_width),
        max_height=int(args.max_height),
        override_bbox=(west, south, east, north),
    )

    scalars: dict[str, object] = {}
    overlays = build_industry_proxy_rgba_overlays(
        args.root,
        ind_merged,
        wt_resolved,
        view.west,
        view.south,
        view.east,
        view.north,
        view.dst_t,
        (view.gh, view.gw),
        path_cfg,
        resampling="bilinear",
        group_pg_out=None,
        scalars_out=scalars,
    )
    _ = overlays

    groups_yaml_rel = (ind_merged.get("paths") or {}).get("ceip_groups_yaml") or (
        sector_cfg.get("industry_paths") or {}
    ).get("ceip_groups_yaml")
    groups_raw: dict[str, object] = {}
    if groups_yaml_rel:
        gy_p = Path(groups_yaml_rel)
        if not gy_p.is_absolute():
            gy_p = args.root / gy_p
        if gy_p.is_file():
            with gy_p.open(encoding="utf-8") as gf:
                groups_raw = dict((yaml.safe_load(gf) or {}).get("groups") or {})

    if _POP_TITLE not in scalars:
        print(
            "WARNING: P_pop layer missing from industry context; combined group maps need population.",
            file=sys.stderr,
        )

    grid_fc: dict | None = None
    cams_ds: xr.Dataset | None = None
    m_area: object | None = None
    nc_path_resolved: Path | None = None
    iso3 = str(sector_cfg.get("cams_country_iso3", "GRC")).strip().upper()
    cams_block = sector_cfg.get("cams") or {}
    gnfr = str(cams_block.get("gnfr", "B"))
    domain_bbox = cams_block.get("domain_bbox_wgs84")
    domain_bbox_t = tuple(float(x) for x in domain_bbox) if domain_bbox else None
    stypes = tuple(cams_block.get("source_types") or ("area",))

    if not args.skip_cams_grid:
        em = path_cfg.get("emissions") or {}
        nc_rel = em.get("cams_2019_nc")
        nc_path = None
        if args.cams_nc is not None:
            nc_path = args.cams_nc if args.cams_nc.is_absolute() else args.root / args.cams_nc
        elif nc_rel:
            nc_path = discover_cams_emissions(args.root, resolve_path(args.root, Path(str(nc_rel))))
            nc_path = nc_path if nc_path.is_absolute() else args.root / nc_path
        if nc_path is not None and nc_path.is_file():
            nc_path_resolved = nc_path
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
                nc_path_resolved = None

    try:
        import contextily  # noqa: F401
    except ImportError:
        contextily = None
    use_basemap = (not args.no_basemap) and (contextily is not None)
    if not args.no_basemap and contextily is None:
        print("WARNING: install contextily for OSM basemap.", file=sys.stderr)

    def _maybe_basemap(rgba: object) -> object:
        if not use_basemap:
            return rgba
        try:
            return _composite_rgba_over_osm(
                rgba,
                view.dst_t,
                (view.gh, view.gw),
                view.west,
                view.south,
                view.east,
                view.north,
                zoom_adjust=args.basemap_zoom_adjust,
            )
        except Exception as exc:
            print(f"  WARNING: basemap failed ({exc}).", file=sys.stderr)
            return rgba

    viz_cfg = sector_cfg.get("visualization") or {}
    weights_per_cell = cams_ds is not None and m_area is not None
    if not weights_per_cell and not args.skip_cams_grid:
        print(
            "NOTE: CAMS NetCDF/mask unavailable — weight maps use global percentile colormap "
            "(not per-cell 0–100%). Ensure paths.yaml emissions.cams_2019_nc resolves.",
            file=sys.stderr,
        )

    print(
        f"View {view.gw}x{view.gh} | bbox [{view.west:.4f}, {view.south:.4f}, {view.east:.4f}, {view.north:.4f}]"
    )

    p_pop_arr = scalars.get(_POP_TITLE)

    for gid in DEFAULT_GNFR_GROUP_ORDER:
        k_osm = f"Industry · OSM_{gid}"
        k_clc = _industry_corine_scalar_title(gid, groups_raw)
        if k_clc is None:
            print(f"WARNING: skip {gid}: no corine_classes in industry_groups.yaml.", file=sys.stderr)
            continue
        if k_osm not in scalars or k_clc not in scalars or p_pop_arr is None:
            print(
                f"WARNING: skip {gid}: missing OSM/CORINE/P_pop scalar layers "
                f"(have_osm={k_osm in scalars} have_clc={k_clc in scalars}).",
                file=sys.stderr,
            )
            continue
        clc_np = np.asarray(scalars[k_clc], dtype=np.float64)
        mx_clc = float(np.nanmax(clc_np)) if clc_np.size > 0 else 0.0
        npos_clc = int(np.count_nonzero(np.isfinite(clc_np) & (clc_np > 1e-12)))
        if mx_clc <= 1e-12:
            print(
                f"WARNING: CORINE layer for {gid} is all-zero in this view ({k_clc!r}; "
                f"{clc_np.size} px). Scalars come from the CORINE GeoTIFF × YAML "
                f"`corine_classes`; if this persists, inspect class codes in the raster.",
                file=sys.stderr,
            )
        elif npos_clc < 200:
            print(
                f"NOTE: CORINE for {gid}: only {npos_clc} positive pixels in bbox (max={mx_clc:.4g}).",
                file=sys.stderr,
            )
        fname = f"industry_bbox_group_{gid.lower()}_osm_clc_ppop.png"
        print(f"\nB_Industry — {gid} combined overlay")
        _save_industry_group_combined(
            scalars[k_osm],
            scalars[k_clc],
            p_pop_arr,
            west=view.west,
            south=view.south,
            east=view.east,
            north=view.north,
            out_path=out_dir / fname,
            title=(
                f"B_Industry — {gid}: CORINE green · OSM red · overlap blue · POP yellow (faint)"
            ),
            group_id=gid,
            dpi=int(args.dpi),
            grid_fc=grid_fc,
            dst_transform=view.dst_t,
            gh=int(view.gh),
            gw=int(view.gw),
            use_basemap=use_basemap,
            basemap_zoom_adjust=args.basemap_zoom_adjust,
            basemap_rgb_dim=cbd,
            debug_combined_layers=bool(args.debug_combined_layers),
        )

    def _weight_rgba_and_cbar(
        pol_key: str, *, display_label: str
    ) -> tuple[object, int, dict[str, object] | None] | None:
        band = pick_band_by_pollutant(
            wt_resolved,
            {**viz_cfg, "visualization_pollutant": pol_key},
            strip_prefixes=(),
            sector_cfg=sector_cfg,
        )
        band, _ = pick_first_positive_band(
            wt_resolved,
            band,
            empty_message=f"No positive weights for {display_label}; using band anyway.",
        )
        stk = read_weight_wgs84_only(
            wt_resolved,
            west=view.west,
            south=view.south,
            east=view.east,
            north=view.north,
            display_width=view.gw,
            display_height=view.gh,
            weight_band=int(band),
        )
        w_arr = stk["weight_wgs84"]
        w_nd = stk["weight_nodata"]
        nd = float(w_nd) if w_nd is not None else None

        if weights_per_cell and cams_ds is not None and m_area is not None:
            rows, cols = np.indices((view.gh, view.gw))
            xs, ys = transform_xy(view.dst_t, rows + 0.5, cols + 0.5, offset="center")
            lons = np.asarray(xs, dtype=np.float64)
            lats = np.asarray(ys, dtype=np.float64)
            cell_id = cams_cell_id_grid(lons, lats, cams_ds, m_area)
            finite = np.isfinite(w_arr)
            if w_nd is not None:
                finite &= w_arr != float(w_nd)
            base_valid = finite & (w_arr > 0)
            z01, valid_pc = normalize_weights_per_cams_cell(
                w_arr, cell_id, base_valid=base_valid
            )
            valid_pc = np.asarray(
                _per_cell_drop_lowest_positive_pct(w_arr, cell_id, valid_pc, w_nd),
                dtype=bool,
            )
            rgba = scalar_to_rgba(
                w_arr,
                colour_mode="global",
                cmap_name="plasma",
                hide_zero=True,
                nodata_val=nd,
                z_precomputed_01=z01,
                valid_precomputed=valid_pc,
            )
            cbar_d: dict[str, object] = {
                "vmin": 0.0,
                "vmax": 1.0,
                "cmap": "plasma",
                "percent_ticks": True,
                "label": (
                    f"{display_label} weight (within CAMS cell, 0–100%; "
                    "bottom 2% of positives per cell hidden)"
                ),
            }
        else:
            rgba = weight_rgba_percentile(w_arr, w_nodata=w_nd, cmap="plasma")
            pv = _percentile_vmin_vmax(w_arr)
            if pv:
                lo, hi = pv
                cbar_d = {
                    "vmin": lo,
                    "vmax": hi,
                    "cmap": "plasma",
                    "label": (
                        f"{display_label} weight (global 2–98% of positives; "
                        "add CAMS for per-cell 0–100%)"
                    ),
                }
            else:
                cbar_d = {
                    "vmin": 0.0,
                    "vmax": 1.0,
                    "cmap": "plasma",
                    "label": f"{display_label} weight",
                }
        if not np.any(rgba[..., 3] > 0):
            return None
        return rgba, int(band), cbar_d

    weight_jobs = (
        ("CO", "co", "industry_bbox_weights_co.png"),
        ("NMVOC", "nmvoc", "industry_bbox_weights_nmvoc.png"),
        ("SOx", "sox", "industry_bbox_weights_sox.png"),
    )
    for label, pol_key, fn in weight_jobs:
        got = _weight_rgba_and_cbar(pol_key, display_label=label)
        if got is None:
            print(f"WARNING: no weight raster for {label}; skip {fn}.", file=sys.stderr)
            continue
        rgba_w, bd, cbar_d = got
        title_w = f"B_Industry — weights ({label}, band {bd})"
        print(f"\n{title_w}")
        out_img = _maybe_basemap(rgba_w)
        _save_png(
            out_img,
            title=title_w,
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

    if cams_ds is not None:
        cams_ds.close()

    if use_basemap:
        print(
            "\nBasemap: OpenStreetMap – https://www.openstreetmap.org/copyright",
            file=sys.stderr,
        )
    print(f"\nDone. Output: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
