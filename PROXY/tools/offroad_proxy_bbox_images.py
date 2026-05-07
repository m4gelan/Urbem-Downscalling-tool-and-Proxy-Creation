#!/usr/bin/env python3
"""I_Offroad proxy PNG exports for a WGS84 bbox (same idea as fugitive_proxy_bbox_images).

Default view: Thessaloniki region. Three group maps — G1 railways (OSM), G2 pipeline lines vs
infrastructure (buffered compressor points + facility footprints), G3 non-road CORINE classes
(four colours, red where dilated class footprints overlap). Weight bands: default NH3 and PM2.5;
weights below 0.05% (raw share) are transparent.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_root_boot = Path(__file__).resolve().parents[2]
if str(_root_boot) not in sys.path:
    sys.path.insert(0, str(_root_boot))

from PROXY.tools.industry_proxy_bbox_images import (
    _basemap_rgb_underlay,
    _dim_rgb_uint8,
    _per_cell_drop_lowest_positive_pct,
)
from PROXY.tools.waste_proxy_bbox_images import (
    _composite_rgba_over_osm,
    _percentile_vmin_vmax,
    _save_png,
)

# Thessaloniki metro / Thermaic gulf (WGS84, deg)
DEFAULT_OFFROAD_BBOX_WGS84 = (22.4, 40.40, 23.1, 40.9)

# G3: morphological dilation (px) so adjacent CLC class buffers can register as overlap
_CORINE_OVERLAP_DILATE_ITER = 2

_RGB_OVERLAP = (214, 42, 48)
# G1 bbox export: railways drawn green (Folium map still uses blue in offroad_area_map).
_RGB_RAILWAY = (34, 139, 34)

# Weight PNGs: hide pixels with raw share below 0.05% (same convention as fugitive bbox tool).
_MIN_VISIBLE_WEIGHT_FRAC = 0.05 / 100.0


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


def _utm_epsg_for_wgs84(lon: float, lat: float) -> int:
    zone = int((float(lon) + 180.0) / 6.0) + 1
    if float(lat) >= 0.0:
        return 32600 + zone
    return 32700 + zone


def _buffer_point_gdf_m(gdf_pts: object, buffer_m: float) -> object | None:
    """Return polygon GeoDataFrame (WGS84) of disk buffers around point / multipoint rows."""
    import geopandas as gpd

    if gdf_pts is None or getattr(gdf_pts, "empty", True):
        return None
    g4326 = gdf_pts.to_crs(4326)
    lon0 = float(g4326.geometry.x.mean())
    lat0 = float(g4326.geometry.y.mean())
    epsg = _utm_epsg_for_wgs84(lon0, lat0)
    gm = g4326.to_crs(epsg)
    buf = gm.geometry.buffer(float(buffer_m))
    out = gpd.GeoDataFrame(geometry=buf, crs=gm.crs).to_crs(4326)
    return out if not out.empty else None


def _rasterize_gdf_on_view(
    gdf: object,
    view: object,
    *,
    dilate_px: int = 0,
) -> object | None:
    """Rasterise any geometry types to a boolean mask on the view grid."""
    try:
        import rasterio.features as rfeat
    except ImportError:
        return None
    if gdf is None or getattr(gdf, "empty", True):
        return None
    try:
        g4326 = gdf.to_crs(4326)
        shapes = [(geom, 1) for geom in g4326.geometry if geom is not None and not geom.is_empty]
        if not shapes:
            return None
        mask = rfeat.rasterize(
            shapes=shapes,
            out_shape=(view.gh, view.gw),
            transform=view.dst_t,
            fill=0,
            all_touched=True,
            dtype="uint8",
        ).astype(bool)
    except Exception:
        return None
    if dilate_px > 0:
        try:
            from scipy.ndimage import binary_dilation

            mask = binary_dilation(mask, iterations=int(dilate_px))
        except ImportError:
            pass
    if not bool(mask.any()):
        return None
    return mask


def _mask_to_rgba(mask: object, rgb: tuple[int, int, int], *, alpha: int = 230) -> object:
    import numpy as np

    m = np.asarray(mask, dtype=bool)
    h, w = m.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[m, 0] = rgb[0]
    rgba[m, 1] = rgb[1]
    rgba[m, 2] = rgb[2]
    rgba[m, 3] = int(alpha)
    return rgba


def _alpha_blend_rgb_under_rgba(base_rgb_uint8: object, overlay_rgba: object) -> object:
    """RGB base (uint8) with RGBA overlay; returns uint8 RGB."""
    import numpy as np

    base = np.asarray(base_rgb_uint8, dtype=np.float32)
    if base.ndim == 3 and base.shape[2] == 3 and float(np.nanmax(base)) > 1.5:
        base = base / 255.0
    base = np.clip(base, 0.0, 1.0)
    ov = np.asarray(overlay_rgba, dtype=np.float32) / 255.0
    a = np.clip(ov[..., 3:4], 0.0, 1.0)
    rgb = ov[..., :3]
    out = rgb * a + base * (1.0 - a)
    return (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)


def _build_g3_corine_rgb(
    clc_i: object,
    nonroad_codes: tuple[int, ...],
    *,
    expand_fn,
    rgb_by_code: dict[int, tuple[int, int, int]],
    overlap_dilate_iter: int = _CORINE_OVERLAP_DILATE_ITER,
) -> object:
    """Per-class colours; red where dilated class masks overlap (>=2)."""
    import numpy as np

    try:
        from scipy.ndimage import binary_dilation
    except ImportError:
        binary_dilation = None

    clc = np.asarray(clc_i, dtype=np.int32)
    h, w = clc.shape
    base_rgb = np.full((h, w, 3), 240, dtype=np.uint8)
    if not nonroad_codes:
        return base_rgb

    masks: dict[int, np.ndarray] = {}
    dilated: list[np.ndarray] = []
    for code in sorted(set(int(c) for c in nonroad_codes)):
        exp = expand_fn(clc, (code,))
        m = np.isin(clc, np.asarray(exp, dtype=np.int32))
        masks[int(code)] = m
        if binary_dilation is not None and int(overlap_dilate_iter) > 0:
            dilated.append(binary_dilation(m, iterations=int(overlap_dilate_iter)))
        else:
            dilated.append(m.copy())

    stack = np.sum(np.stack(dilated, axis=0), axis=0).astype(np.int32)
    overlap = stack > 1

    for code in sorted(masks.keys()):
        m = masks[code] & (~overlap)
        col = rgb_by_code.get(int(code), (106, 27, 154))
        base_rgb[m, 0] = col[0]
        base_rgb[m, 1] = col[1]
        base_rgb[m, 2] = col[2]

    ro, go, bo = _RGB_OVERLAP
    base_rgb[overlap, 0] = ro
    base_rgb[overlap, 1] = go
    base_rgb[overlap, 2] = bo
    return base_rgb


def main() -> int:
    root = _ensure_import_path()
    ap = argparse.ArgumentParser(
        description="Export I_Offroad G1/G2/G3 context maps and weight PNGs (default NH3, PM2.5)."
    )
    ap.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        default=None,
        help=f"WGS84 bbox (default: {DEFAULT_OFFROAD_BBOX_WGS84}).",
    )
    ap.add_argument("--root", type=Path, default=root)
    ap.add_argument("--paths-yaml", type=Path, default=None)
    ap.add_argument("--sector-yaml", type=Path, default=None)
    ap.add_argument(
        "--weight-tif",
        type=Path,
        default=None,
        help="I_Offroad weights GeoTIFF (default: OUTPUT/Proxy_weights/I_Offroad/offroad_areasource.tif).",
    )
    ap.add_argument("--out-dir", type=Path, default=None)
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
        default="nh3,pm25",
        help="Comma-separated tokens for weight PNGs (default: nh3,pm25).",
    )
    ap.add_argument(
        "--facility-point-buffer-m",
        type=float,
        default=None,
        help="Buffer radius (m) for compressor points; default: area_proxy.pipeline_facility_sigma_m or 500.",
    )
    args = ap.parse_args()

    bbox = tuple(args.bbox) if args.bbox is not None else DEFAULT_OFFROAD_BBOX_WGS84
    west, south, east, north = (float(x) for x in bbox)
    if west >= east or south >= north:
        print("ERROR: require west < east and south < north.", file=sys.stderr)
        return 1
    cbd = float(args.combined_basemap_dim)
    if not (0.15 <= cbd <= 1.0):
        print("ERROR: --combined-basemap-dim must be between 0.15 and 1.0.", file=sys.stderr)
        return 1

    paths_yaml = args.paths_yaml or (args.root / "PROXY" / "config" / "paths.yaml")
    sector_yaml = args.sector_yaml or (args.root / "PROXY" / "config" / "sectors" / "offroad.yaml")
    wt_default = args.root / "OUTPUT" / "Proxy_weights" / "I_Offroad" / "offroad_areasource.tif"
    weight_tif = args.weight_tif or wt_default
    wt = weight_tif if weight_tif.is_absolute() else args.root / weight_tif

    for label, pth in [("paths.yaml", paths_yaml), ("sector YAML", sector_yaml), ("weight GeoTIFF", wt)]:
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

    area_proxy = sector_cfg.get("area_proxy") or {}
    pcfg = area_proxy.get("proxy") or {}
    buf_m = float(args.facility_point_buffer_m) if args.facility_point_buffer_m is not None else float(
        pcfg.get("pipeline_facility_sigma_m", 500.0)
    )

    out_dir = (args.out_dir.resolve() if args.out_dir else Path.cwd().resolve())
    out_dir.mkdir(parents=True, exist_ok=True)

    from PROXY.core.dataloaders import resolve_path
    from PROXY.core.dataloaders.discovery import discover_cams_emissions
    from PROXY.sectors.I_Offroad.cams_area_mask import offroad_union_area_mask
    from PROXY.sectors.I_Offroad.pipeline_osm import (
        load_pipeline_facilities,
        load_pipeline_lines,
        load_pipeline_union,
    )
    from PROXY.sectors.I_Offroad.proxy_rules import osm_railway_line_filter_sets
    from PROXY.sectors.I_Offroad.rail_osm import filter_rail_lines
    from PROXY.visualization._mapbuilder import (
        build_cams_area_grid_geojson_for_view,
        compute_view_context,
        pick_band_by_pollutant,
        pick_first_positive_band,
        resolve_under_root,
        weight_rgba_log,
    )
    from PROXY.visualization.cams_grid import cams_cell_id_grid, normalize_weights_per_cams_cell
    from PROXY.visualization.offroad_area_map import (
        _NONROAD_CLC_LAYER_RGB,
        _NONROAD_CLC_TITLES,
        _OFFROAD_COLOR_PIPELINE,
        _OFFROAD_COLOR_PIPELINE_FACILITY,
        _emission_indices_from_area_proxy,
        _expand_level2_highlights_for_clc_grid,
        _nonroad_corine_codes_from_proxy,
        _rasterize_lines_on_view,
        _split_facilities_points_other,
    )
    from PROXY.visualization.overlay_utils import read_corine_clc_wgs84_on_weight_grid, read_weight_wgs84_only, scalar_to_rgba
    from rasterio.transform import xy as transform_xy
    import geopandas as gpd
    import numpy as np
    import xarray as xr

    wt_resolved = resolve_under_root(wt, args.root)
    corine_rel = (path_cfg.get("proxy_common") or {}).get("corine_tif")
    if not corine_rel:
        print("ERROR: paths.yaml missing proxy_common.corine_tif.", file=sys.stderr)
        return 1
    corine_tif = resolve_path(args.root, Path(str(corine_rel)))

    if not corine_tif.is_file():
        print(f"ERROR: CORINE GeoTIFF not found: {corine_tif}", file=sys.stderr)
        return 1

    emission_indices = _emission_indices_from_area_proxy(area_proxy)
    source_type_index = int(area_proxy.get("source_type_index", 1))
    corine_band = int(area_proxy.get("corine_band", 1))
    nonroad_codes = _nonroad_corine_codes_from_proxy(area_proxy)

    view = compute_view_context(
        wt_resolved,
        pad_deg=float(args.pad_deg),
        max_width=int(args.max_width),
        max_height=int(args.max_height),
        override_bbox=(west, south, east, north),
    )

    stk_c = read_corine_clc_wgs84_on_weight_grid(
        wt_resolved,
        corine_tif,
        corine_band=corine_band,
        west=view.west,
        south=view.south,
        east=view.east,
        north=view.north,
        display_width=view.gw,
        display_height=view.gh,
    )
    clc_raw = stk_c["corine_wgs84"]
    clc_nd = stk_c["corine_nodata"]
    clc_i = np.full((view.gh, view.gw), -1, dtype=np.int32)
    ok = np.isfinite(clc_raw)
    if clc_nd is not None:
        ok = ok & (clc_raw != float(clc_nd))
    clc_i[ok] = np.rint(clc_raw[ok]).astype(np.int32)

    rules = osm_railway_line_filter_sets(path_cfg)
    osm_rel = (path_cfg.get("osm") or {}).get("offroad")
    pipeline_lines_gdf = None
    pipeline_facilities_gdf = None
    pipeline_gdf = None
    railway_gdf = None
    if osm_rel:
        osm_path = Path(str(osm_rel))
        if not osm_path.is_absolute():
            osm_path = args.root / osm_path
        if osm_path.is_file():
            try:
                pipeline_lines_gdf = load_pipeline_lines(osm_path)
                if getattr(pipeline_lines_gdf, "empty", True):
                    uni_try = load_pipeline_union(osm_path, args.root)
                    if uni_try is not None and not uni_try.empty:
                        mgeom = uni_try.geometry.geom_type.isin(["LineString", "MultiLineString"])
                        pipeline_lines_gdf = uni_try.loc[mgeom].copy()
            except Exception as exc:
                print(f"WARNING: pipeline lines load failed: {exc}", file=sys.stderr)
                pipeline_lines_gdf = None
            try:
                pipeline_facilities_gdf = load_pipeline_facilities(osm_path)
            except Exception as exc:
                print(f"WARNING: pipeline facilities load failed: {exc}", file=sys.stderr)
                pipeline_facilities_gdf = None
            try:
                pipeline_gdf = load_pipeline_union(osm_path, args.root)
            except Exception:
                pipeline_gdf = None
            try:
                rlines = gpd.read_file(osm_path, layer="osm_offroad_rail_lines")
                railway_gdf = filter_rail_lines(rlines, bad_line_types=rules[0], lifecycle_disallow=rules[1])
            except Exception:
                railway_gdf = None
        else:
            print(f"WARNING: OSM GPKG not found: {osm_path}", file=sys.stderr)

    pm_ln = _rasterize_lines_on_view(pipeline_lines_gdf, view, dilate_px=1)
    if pm_ln is None:
        pm_ln = _rasterize_lines_on_view(pipeline_gdf, view, dilate_px=1)
    fac_pts, fac_other = _split_facilities_points_other(pipeline_facilities_gdf)
    pm_fac_lines = _rasterize_lines_on_view(fac_other, view, dilate_px=3)
    buf_gdf = _buffer_point_gdf_m(fac_pts, buf_m) if fac_pts is not None else None
    pm_fac_buf = _rasterize_gdf_on_view(buf_gdf, view, dilate_px=0) if buf_gdf is not None else None

    parts_infra = [x for x in (pm_fac_lines, pm_fac_buf) if x is not None]
    if parts_infra:
        mask_infra = np.logical_or.reduce(parts_infra)
    else:
        mask_infra = None

    railway_mask = _rasterize_lines_on_view(railway_gdf, view, dilate_px=1)

    grid_fc: dict | None = None
    cams_ds: xr.Dataset | None = None
    m_area = None
    iso3 = str(sector_cfg.get("cams_country_iso3", "GRC")).strip().upper()

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
                m_area = offroad_union_area_mask(
                    cams_ds,
                    iso3,
                    emission_category_indices=emission_indices,
                    source_type_index=source_type_index,
                    domain_bbox_wgs84=None,
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

    def _basemap_rgb() -> np.ndarray:
        base = _basemap_rgb_underlay(
            gh=int(view.gh),
            gw=int(view.gw),
            dst_transform=view.dst_t,
            west=view.west,
            south=view.south,
            east=view.east,
            north=view.north,
            zoom_adjust=args.basemap_zoom_adjust,
            use_basemap=use_basemap,
        )
        dim = float(cbd) if use_basemap else 0.92
        return np.asarray(_dim_rgb_uint8(base, float(np.clip(dim, 0.15, 1.0))), dtype=np.uint8)

    def _maybe_basemap_rgba(rgba: object) -> object:
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

    weights_per_cell = cams_ds is not None and m_area is not None
    if not weights_per_cell and not args.skip_cams_grid:
        print("NOTE: CAMS NetCDF/mask unavailable — weight maps use global log stretch.", file=sys.stderr)

    print(
        f"View {view.gw}x{view.gh} | bbox [{view.west:.4f}, {view.south:.4f}, {view.east:.4f}, {view.north:.4f}]"
    )

    base_u8 = _basemap_rgb()

    # --- G1 railways
    if railway_mask is not None and bool(np.any(railway_mask)):
        r_rail = _mask_to_rgba(railway_mask, _RGB_RAILWAY, alpha=235)
        g1_rgb = _alpha_blend_rgb_under_rgba(base_u8, r_rail)
        legend_g1 = [
            ("G1 Railways (OSM)", _RGB_RAILWAY),
        ]
    else:
        g1_rgb = base_u8
        legend_g1 = [("G1 Railways — no geometry in bbox", (200, 200, 200))]
    _save_png(
        g1_rgb,
        title="I_Offroad — G1: railways (OSM)",
        west=view.west,
        south=view.south,
        east=view.east,
        north=view.north,
        out_path=out_dir / "offroad_bbox_group_g1_railways.png",
        dpi=int(args.dpi),
        grid_fc=grid_fc,
        legend_entries=legend_g1,
        colorbar_spec=None,
    )

    # --- G2 pipeline vs infrastructure
    g2_rgb = base_u8.copy()
    legend_g2: list[tuple[str, tuple[int, int, int]]] = []
    if pm_ln is not None and bool(np.any(pm_ln)):
        g2_rgb = _alpha_blend_rgb_under_rgba(g2_rgb, _mask_to_rgba(pm_ln, _OFFROAD_COLOR_PIPELINE, alpha=235))
        legend_g2.append(("G2 Pipeline lines (hydrocarbon)", _OFFROAD_COLOR_PIPELINE))
    if mask_infra is not None and bool(np.any(mask_infra)):
        g2_rgb = _alpha_blend_rgb_under_rgba(
            g2_rgb, _mask_to_rgba(mask_infra, _OFFROAD_COLOR_PIPELINE_FACILITY, alpha=235)
        )
        legend_g2.append(
            (f"G2 Infrastructure (facilities + {buf_m:g} m buffer on compressor points)", _OFFROAD_COLOR_PIPELINE_FACILITY)
        )
    if not legend_g2:
        legend_g2 = [("G2 — no pipeline / facility geometry in bbox", (200, 200, 200))]
    _save_png(
        g2_rgb,
        title="I_Offroad — G2: pipeline lines vs infrastructure (buffered points)",
        west=view.west,
        south=view.south,
        east=view.east,
        north=view.north,
        out_path=out_dir / "offroad_bbox_group_g2_pipeline_infrastructure.png",
        dpi=int(args.dpi),
        grid_fc=grid_fc,
        legend_entries=legend_g2,
        colorbar_spec=None,
    )

    # --- G3 CORINE classes + overlap (red where dilated class buffers intersect)
    g3_layer = _build_g3_corine_rgb(
        clc_i,
        tuple(int(x) for x in nonroad_codes),
        expand_fn=_expand_level2_highlights_for_clc_grid,
        rgb_by_code=_NONROAD_CLC_LAYER_RGB,
        overlap_dilate_iter=_CORINE_OVERLAP_DILATE_ITER,
    )
    try:
        from scipy.ndimage import binary_dilation as _bd
    except ImportError:
        _bd = None
    in_class = np.zeros((view.gh, view.gw), dtype=bool)
    for code in sorted(set(int(c) for c in nonroad_codes)):
        exp = _expand_level2_highlights_for_clc_grid(clc_i, (code,))
        in_class |= np.isin(clc_i, np.asarray(exp, dtype=np.int32))
    if _bd is not None and nonroad_codes:
        dilated = []
        for code in sorted(set(int(c) for c in nonroad_codes)):
            exp = _expand_level2_highlights_for_clc_grid(clc_i, (code,))
            m = np.isin(clc_i, np.asarray(exp, dtype=np.int32))
            dilated.append(_bd(m, iterations=_CORINE_OVERLAP_DILATE_ITER))
        overlap = np.sum(np.stack(dilated, axis=0), axis=0) > 1
    else:
        overlap = np.zeros((view.gh, view.gw), dtype=bool)
    in_draw = in_class | overlap
    rgba_g3 = np.zeros((view.gh, view.gw, 4), dtype=np.uint8)
    rgba_g3[in_draw, 0] = g3_layer[in_draw, 0]
    rgba_g3[in_draw, 1] = g3_layer[in_draw, 1]
    rgba_g3[in_draw, 2] = g3_layer[in_draw, 2]
    rgba_g3[in_draw, 3] = 220
    g3_final = _alpha_blend_rgb_under_rgba(base_u8, rgba_g3)

    legend_g3 = []
    for c in sorted(set(int(x) for x in nonroad_codes)):
        title = _NONROAD_CLC_TITLES.get(int(c), "").strip()
        lab = f"CLC {c}" + (f" — {title}" if title else "")
        legend_g3.append((lab, _NONROAD_CLC_LAYER_RGB.get(int(c), (128, 128, 128))))
    legend_g3.append(("Overlap (dilated class buffers)", _RGB_OVERLAP))
    _save_png(
        g3_final,
        title="I_Offroad — G3: non-road CORINE classes (overlap = red)",
        west=view.west,
        south=view.south,
        east=view.east,
        north=view.north,
        out_path=out_dir / "offroad_bbox_group_g3_corine_classes.png",
        dpi=int(args.dpi),
        grid_fc=grid_fc,
        legend_entries=legend_g3,
        colorbar_spec=None,
    )

    def _weight_rgba_and_cbar(
        pol_key: str, *, display_label: str
    ) -> tuple[object, int, dict[str, object] | None] | None:
        band = pick_band_by_pollutant(
            wt_resolved,
            {**area_proxy, "visualization_pollutant": pol_key},
            strip_prefixes=(),
            sector_cfg=area_proxy,
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
            z01, valid_pc = normalize_weights_per_cams_cell(w_arr, cell_id, base_valid=base_valid)
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
            rgba = weight_rgba_log(w_arr, w_nodata=w_nd, cmap="plasma")
            pv = _percentile_vmin_vmax(w_arr)
            if pv:
                lo, hi = pv
                cbar_d = {
                    "vmin": lo,
                    "vmax": hi,
                    "cmap": "plasma",
                    "label": (
                        f"{display_label} weight (log10; global 2–98% ref for legend; "
                        f"<{_MIN_VISIBLE_WEIGHT_FRAC * 100:.3g}% transparent)"
                    ),
                }
            else:
                cbar_d = {
                    "vmin": 0.0,
                    "vmax": 1.0,
                    "cmap": "plasma",
                    "label": (
                        f"{display_label} weight (log10; "
                        f"<{_MIN_VISIBLE_WEIGHT_FRAC * 100:.3g}% transparent)"
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
        fn = f"offroad_bbox_weights_{safe}.png"
        got = _weight_rgba_and_cbar(pol, display_label=label)
        if got is None:
            print(f"WARNING: no weight raster for {pol}; skip {fn}.", file=sys.stderr)
            continue
        rgba_w, bd, cbar_d = got
        title_w = f"I_Offroad — weights ({label}, band {bd})"
        print(f"\n{title_w}")
        out_img = _maybe_basemap_rgba(rgba_w)
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
