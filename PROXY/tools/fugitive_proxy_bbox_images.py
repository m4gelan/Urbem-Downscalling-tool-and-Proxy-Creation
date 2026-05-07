#!/usr/bin/env python3
"""D_Fugitive proxy PNG exports for a WGS84 bbox (same role as industry_proxy_bbox_images).

Per CEIP group G1..G4: mixture ingredients with fixed hues — CORINE green, GEM coal brown, OSM red,
VIIRS orange, GEM oil / GOGET black, two or more non-population types overlapping blue, population
yellow (faint, G3). Optional OSM basemap and CAMS GNFR D area grid. Weight rasters: default CO, PM10, and NOx.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

_root_boot = Path(__file__).resolve().parents[2]
if str(_root_boot) not in sys.path:
    sys.path.insert(0, str(_root_boot))

from PROXY.tools.industry_proxy_bbox_images import (
    _basemap_rgb_underlay,
    _dim_rgb_uint8,
    _distinct_signal_rgba,
    _per_cell_drop_lowest_positive_pct,
    _signal_strength_t,
)
from PROXY.tools.waste_proxy_bbox_images import (
    _alpha_composite_rgb_under_rgba,
    _composite_rgba_over_osm,
    _percentile_vmin_vmax,
    _save_png,
)

DEFAULT_FUGITIVE_BBOX_WGS84 = (22, 37.2, 22.3, 37.6)

_NONPOP_CATS = ("corine", "gem_coal", "osm", "viirs", "gem_oil")

_RGB_CORINE = (16, 104, 42)
_RGB_GEM_COAL = (139, 90, 43)
_RGB_OSM = (214, 42, 48)
_RGB_VIIRS = (255, 140, 0)
_RGB_GEM_OIL = (28, 28, 28)
_RGB_POP = (250, 216, 28)
_RGB_MULTI = (22, 52, 138)

# Same title as ``fugitive_context.build_fugitive_proxy_rgba_overlays`` (shared with Folium maps).
_P_POP_SCALAR_TITLE = "Fugitive · P_pop (z-score)"
# Fainter than industry combined maps — population is context only on these exports.
_POP_OVERLAY_MAX_ALPHA = 0.5

# Weight PNGs: raw shares in [0, 1] — drop pixels below 0.05% (transparent).
_MIN_VISIBLE_WEIGHT_FRAC = 0.05 / 100.0

_RGB_BY_CAT: dict[str, tuple[int, int, int]] = {
    "corine": _RGB_CORINE,
    "gem_coal": _RGB_GEM_COAL,
    "osm": _RGB_OSM,
    "viirs": _RGB_VIIRS,
    "gem_oil": _RGB_GEM_OIL,
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_import_path() -> Path:
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _zero_alpha_where_weight_below_threshold(
    rgba: object,
    w_arr: object,
    w_nd: object | None,
    *,
    min_weight_frac: float,
) -> object:
    """Set alpha to 0 where positive finite weight is below ``min_weight_frac`` (e.g. 0.05% → 5e-4)."""
    import numpy as np

    if float(min_weight_frac) <= 0.0:
        return rgba
    r = np.asarray(rgba, dtype=np.uint8).copy()
    w = np.asarray(w_arr, dtype=np.float64)
    m = np.isfinite(w) & (w > 0)
    if w_nd is not None:
        m &= w != float(w_nd)
    r[m & (w < float(min_weight_frac)), 3] = 0
    return r


def _category_for_fugitive_mixture_lab(lab: str) -> str | None:
    s = lab.strip()
    if s.upper() == "POP":
        return "population"
    u = s.upper()
    if "VIIRS" in u:
        return "viirs"
    if "COAL" in u:
        return "gem_coal"
    if s.startswith("OSM") or "OSM_" in s:
        return "osm"
    if s.startswith("CLC"):
        return "corine"
    if s.startswith("GEM") or "GOGET" in u:
        return "gem_oil"
    return None


def _composite_rgb_fugitive_mixture(
    base_rgb_uint8: object,
    t_by_cat: dict[str, object],
    *,
    cover_scale: float = 0.95,
    rgb_by_cat: dict[str, tuple[int, int, int]] = _RGB_BY_CAT,
    rgb_multi: tuple[int, int, int] = _RGB_MULTI,
) -> object:
    import numpy as np

    base_f = np.asarray(base_rgb_uint8, dtype=np.float32)
    if base_f.ndim != 3 or base_f.shape[2] != 3:
        raise ValueError("base must be HxWx3")
    if float(np.nanmax(base_f)) > 1.5:
        base_f = base_f / 255.0
    base_f = np.clip(base_f, 0.0, 1.0)

    h, w = int(base_f.shape[0]), int(base_f.shape[1])
    ts: dict[str, np.ndarray] = {}
    for c in _NONPOP_CATS:
        if c not in t_by_cat:
            continue
        t = np.clip(np.asarray(t_by_cat[c], dtype=np.float64), 0.0, 1.0)
        if t.shape != (h, w):
            raise ValueError(f"strength grid shape mismatch for {c}")
        mx = float(np.nanmax(t))
        if mx <= 1e-15:
            ts[c] = t
        else:
            ts[c] = np.clip(t / mx, 0.0, 1.0)

    cats_present = [c for c in _NONPOP_CATS if c in ts]
    if not cats_present:
        return (np.clip(base_f, 0.0, 1.0) * 255.0).astype(np.uint8)

    eps = 1e-6
    t_stack = np.stack([ts[c] for c in cats_present], axis=0)
    active = t_stack > eps
    n_active = np.sum(active, axis=0)
    am = np.max(np.where(active, t_stack, 0.0), axis=0)
    k = float(np.clip(cover_scale, 0.0, 1.0))
    alpha = np.clip(am * k, 0.0, 1.0)

    rb = np.array(rgb_multi, dtype=np.float64) / 255.0
    rcmap = {k2: np.array(v2, dtype=np.float64) / 255.0 for k2, v2 in rgb_by_cat.items()}
    solid = np.zeros((h, w, 3), dtype=np.float64)
    for ki, c in enumerate(cats_present):
        only = active[ki] & (n_active == 1)
        col = rcmap.get(c, rb)
        solid[only, 0] = col[0]
        solid[only, 1] = col[1]
        solid[only, 2] = col[2]
    solid[n_active >= 2, :] = rb

    out = base_f * (1.0 - alpha[..., np.newaxis]) + solid * alpha[..., np.newaxis]
    return (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)


def _gather_group_mixture_inputs(
    scalars: dict[str, object],
    gid: str,
    mixture_labels: dict[str, tuple[str, ...]],
) -> tuple[dict[str, list[object]], object | None]:
    labs = mixture_labels.get(str(gid)) or ()
    by_cat: dict[str, list[object]] = defaultdict(list)
    pop_arr: object | None = None
    for lab in labs:
        cat = _category_for_fugitive_mixture_lab(lab)
        key = f"Fugitive · {lab}"
        if key not in scalars:
            continue
        arr = scalars[key]
        if cat == "population":
            pop_arr = arr
        elif cat is not None:
            by_cat[cat].append(arr)
    return dict(by_cat), pop_arr


def _save_fugitive_group_combined(
    *,
    scalars: dict[str, object],
    gid: str,
    mixture_labels: dict[str, tuple[str, ...]],
    west: float,
    south: float,
    east: float,
    north: float,
    out_path: Path,
    dpi: int,
    grid_fc: dict | None,
    dst_transform,
    gh: int,
    gw: int,
    use_basemap: bool,
    basemap_zoom_adjust: int | None,
    basemap_rgb_dim: float = 0.72,
    flat_basemap_rgb_dim: float = 0.92,
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

    by_cat_raw, pop_mixture = _gather_group_mixture_inputs(scalars, gid, mixture_labels)
    pop_raw = scalars.get(_P_POP_SCALAR_TITLE)
    if pop_raw is None:
        pop_raw = pop_mixture
    t_by_cat: dict[str, object] = {}
    for cat, arrs in by_cat_raw.items():
        if not arrs:
            continue
        t_parts = [
            _signal_strength_t(
                a,
                lo_pct=0.5,
                hi_pct=99.5,
                gamma=0.55,
                alpha_power=1.05,
                t_floor=0.08,
                t_floor_where_above=0.05,
                positive_only=True,
            )
            for a in arrs
        ]
        t_by_cat[cat] = np.maximum.reduce(t_parts) if len(t_parts) > 1 else t_parts[0]

    rgb = _composite_rgb_fugitive_mixture(
        base,
        t_by_cat,
        cover_scale=0.95,
        rgb_by_cat=_RGB_BY_CAT,
        rgb_multi=_RGB_MULTI,
    )

    if pop_raw is not None:
        pop_r = _distinct_signal_rgba(
            pop_raw,
            _RGB_POP,
            lo_pct=48.0,
            hi_pct=98.5,
            max_alpha=float(_POP_OVERLAY_MAX_ALPHA),
            gamma=1.05,
            alpha_power=1.0,
            min_alpha_u8=0,
            min_alpha_signal=0.0,
            positive_only=False,
        )
        rgb = _alpha_composite_rgb_under_rgba(rgb, pop_r)

    legend = [
        (f"{gid} CORINE (green)", _RGB_CORINE),
        (f"{gid} GEM coal (brown)", _RGB_GEM_COAL),
        (f"{gid} OSM (red)", _RGB_OSM),
        (f"{gid} VIIRS (orange)", _RGB_VIIRS),
        (f"{gid} GEM oil / GOGET (black)", _RGB_GEM_OIL),
        (f"{gid} multi-type overlap (blue)", _RGB_MULTI),
        (f"{gid} population P_pop z-score (yellow, faint)", _RGB_POP),
    ]
    _save_png(
        rgb,
        title=f"D_Fugitive — {gid}: mixture hues (CORINE/OSM/VIIRS/GEM/POP)",
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


def main() -> int:
    root = _ensure_import_path()
    ap = argparse.ArgumentParser(
        description="Export D_Fugitive mixture hue maps per G1..G4 and weight PNGs for selected pollutants."
    )
    ap.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        default=None,
        help=f"WGS84 bbox (default: {DEFAULT_FUGITIVE_BBOX_WGS84}).",
    )
    ap.add_argument("--root", type=Path, default=root)
    ap.add_argument("--paths-yaml", type=Path, default=None)
    ap.add_argument("--sector-yaml", type=Path, default=None)
    ap.add_argument(
        "--weight-tif",
        type=Path,
        default=None,
        help="D_Fugitive area weights GeoTIFF (default: OUTPUT/Proxy_weights/D_Fugitive/fugitive_areasource.tif).",
    )
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--country", default="EL")
    ap.add_argument("--max-width", type=int, default=1400)
    ap.add_argument("--max-height", type=int, default=1200)
    ap.add_argument("--pad-deg", type=float, default=0.0)
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--cams-nc", type=Path, default=None)
    ap.add_argument("--skip-cams-grid", action="store_true")
    ap.add_argument("--no-basemap", action="store_true")
    ap.add_argument("--basemap-zoom-adjust", type=int, default=None)
    ap.add_argument(
        "--combined-basemap-dim",
        type=float,
        default=0.72,
        help="OSM basemap dim factor before overlays (0.15–1).",
    )
    ap.add_argument(
        "--pollutants",
        type=str,
        default="co,pm10,nox",
        help="Comma-separated pollutant tokens for weight PNGs (default: co,pm10,nox).",
    )
    args = ap.parse_args()

    bbox = tuple(args.bbox) if args.bbox is not None else DEFAULT_FUGITIVE_BBOX_WGS84
    west, south, east, north = (float(x) for x in bbox)
    if west >= east or south >= north:
        print("ERROR: require west < east and south < north.", file=sys.stderr)
        return 1
    cbd = float(args.combined_basemap_dim)
    if not (0.15 <= cbd <= 1.0):
        print("ERROR: --combined-basemap-dim must be between 0.15 and 1.0.", file=sys.stderr)
        return 1

    paths_yaml = args.paths_yaml or (args.root / "PROXY" / "config" / "paths.yaml")
    sector_yaml = args.sector_yaml or (args.root / "PROXY" / "config" / "sectors" / "fugitive.yaml")
    wt_default = args.root / "OUTPUT" / "Proxy_weights" / "D_Fugitive" / "fugitive_areasource.tif"
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
    from PROXY.sectors.D_Fugitive.builder import _merge_fugitive_pipeline_cfg
    from PROXY.visualization._mapbuilder import (
        build_cams_area_grid_geojson_for_view,
        compute_view_context,
        pick_band_by_pollutant,
        pick_first_positive_band,
        resolve_under_root,
        weight_rgba_percentile,
    )
    from PROXY.visualization.cams_grid import cams_cell_id_grid, normalize_weights_per_cams_cell
    from PROXY.visualization.fugitive_context import (
        FUGITIVE_MIXTURE_LAYER_LABELS,
        build_fugitive_proxy_rgba_overlays,
    )
    from PROXY.visualization.overlay_utils import read_weight_wgs84_only, scalar_to_rgba
    from rasterio.transform import xy as transform_xy
    import numpy as np
    import xarray as xr

    wt_resolved = resolve_under_root(wt, args.root)
    fug_merged = _merge_fugitive_pipeline_cfg(
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
    _ = build_fugitive_proxy_rgba_overlays(
        args.root,
        fug_merged,
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
        visualization_cfg=sector_cfg.get("visualization") or {},
    )

    grid_fc: dict | None = None
    cams_ds: xr.Dataset | None = None
    m_area: object | None = None
    iso3 = str(sector_cfg.get("cams_country_iso3", "GRC")).strip().upper()
    cams_block = sector_cfg.get("cams") or {}
    gnfr = str(cams_block.get("gnfr", "D"))
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
            "NOTE: CAMS NetCDF/mask unavailable — weight maps use global percentile colormap.",
            file=sys.stderr,
        )

    print(
        f"View {view.gw}x{view.gh} | bbox [{view.west:.4f}, {view.south:.4f}, {view.east:.4f}, {view.north:.4f}]"
    )

    for gid in DEFAULT_GNFR_GROUP_ORDER:
        by_cat, pop_mix = _gather_group_mixture_inputs(scalars, gid, FUGITIVE_MIXTURE_LAYER_LABELS)
        pop_any = scalars.get(_P_POP_SCALAR_TITLE) is not None or pop_mix is not None
        if not by_cat and not pop_any:
            print(f"WARNING: skip {gid}: no mixture scalars in bbox.", file=sys.stderr)
            continue
        fname = f"fugitive_bbox_group_{gid.lower()}_mixture_hues.png"
        print(f"\nD_Fugitive — {gid} mixture overlay")
        _save_fugitive_group_combined(
            scalars=scalars,
            gid=gid,
            mixture_labels=FUGITIVE_MIXTURE_LAYER_LABELS,
            west=view.west,
            south=view.south,
            east=view.east,
            north=view.north,
            out_path=out_dir / fname,
            dpi=int(args.dpi),
            grid_fc=grid_fc,
            dst_transform=view.dst_t,
            gh=int(view.gh),
            gw=int(view.gw),
            use_basemap=use_basemap,
            basemap_zoom_adjust=args.basemap_zoom_adjust,
            basemap_rgb_dim=cbd,
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
                    "bottom 2% of positives per cell hidden; "
                    f"<{_MIN_VISIBLE_WEIGHT_FRAC * 100:.3g}% transparent)"
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
                        "add CAMS for per-cell 0–100%; "
                        f"<{_MIN_VISIBLE_WEIGHT_FRAC * 100:.3g}% transparent)"
                    ),
                }
            else:
                cbar_d = {
                    "vmin": 0.0,
                    "vmax": 1.0,
                    "cmap": "plasma",
                    "label": (
                        f"{display_label} weight "
                        f"(<{_MIN_VISIBLE_WEIGHT_FRAC * 100:.3g}% transparent)"
                    ),
                }
        rgba = _zero_alpha_where_weight_below_threshold(
            rgba, w_arr, w_nd, min_weight_frac=_MIN_VISIBLE_WEIGHT_FRAC
        )
        if not np.any(rgba[..., 3] > 0):
            return None
        return rgba, int(band), cbar_d

    pol_tokens = [p.strip() for p in str(args.pollutants).split(",") if p.strip()]
    for pol in pol_tokens:
        label = pol.upper().replace("_", "")
        safe = pol.lower().replace(".", "").replace(" ", "_")
        fn = f"fugitive_bbox_weights_{safe}.png"
        got = _weight_rgba_and_cbar(pol, display_label=label)
        if got is None:
            print(f"WARNING: no weight raster for {pol}; skip {fn}.", file=sys.stderr)
            continue
        rgba_w, bd, cbar_d = got
        title_w = f"D_Fugitive — weights ({label}, band {bd})"
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
