#!/usr/bin/env python3
"""Static PNG exports for A_PublicPower over a WGS84 bbox: eligible CORINE, population, weights, CAMS grid.

Optional OpenStreetMap basemap (EPSG:3857 tiles warped to the overlay EPSG:4326 grid), same approach as
``shipping_proxy_bbox_images.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_import_path() -> Path:
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _viz_cfg_from_sector(sector_cfg: dict) -> dict:
    """Band/pollutant keys expected by :func:`pick_band_by_pollutant`."""
    ap = dict(sector_cfg.get("area_proxy") or {})
    pol = sector_cfg.get("pollutant_name")
    if pol and str(pol).strip() and "visualization_pollutant" not in ap:
        ap["visualization_pollutant"] = str(pol).strip()
    vis = sector_cfg.get("visualization") or {}
    if isinstance(vis, dict):
        for k in ("visualization_weight_band", "visualization_pollutant", "visualization_pollutants"):
            if k in vis and vis[k] is not None:
                ap.setdefault(k, vis[k])
    return ap


def _reproject_basemap_to_overlay_grid(
    img_merc: object,
    extent_merc: tuple[float, float, float, float],
    dst_transform,
    dst_shape: tuple[int, int],
) -> object:
    """Warp OSM RGB (EPSG:3857) onto the same EPSG:4326 pixel grid as the proxy layers."""
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.warp import reproject, Resampling

    gh, gw = int(dst_shape[0]), int(dst_shape[1])
    img_merc = np.asarray(img_merc)
    left, right, bottom, top = extent_merc
    hm, wm = int(img_merc.shape[0]), int(img_merc.shape[1])

    src_transform = from_bounds(left, bottom, right, top, wm, hm)
    crs3857 = rasterio.crs.CRS.from_epsg(3857)
    crs4326 = rasterio.crs.CRS.from_epsg(4326)
    FALLBACK_GRAY = 0.933

    base_rgb = np.full((gh, gw, 3), FALLBACK_GRAY, dtype=np.float32)
    src_arr = np.asarray(img_merc[..., :3], dtype=np.float32) / 255.0

    for k in range(3):
        dst_band = np.full((gh, gw), np.nan, dtype=np.float64)
        reproject(
            source=src_arr[..., k].astype(np.float64),
            destination=dst_band,
            src_transform=src_transform,
            src_crs=crs3857,
            dst_transform=dst_transform,
            dst_crs=crs4326,
            resampling=Resampling.bilinear,
            dst_nodata=np.nan,
        )
        valid = np.isfinite(dst_band)
        base_rgb[..., k] = np.where(valid, dst_band.astype(np.float32), FALLBACK_GRAY)

    return base_rgb


def _alpha_composite_rgb_under_rgba(
    base_rgb: object,
    overlay_rgba: object,
) -> object:
    """Porter–Duff 'over': overlay on top of basemap RGB."""
    over = np.asarray(overlay_rgba, dtype=np.float32) / 255.0
    alpha = np.clip(over[..., 3:4], 0.0, 1.0)
    rgb_o = over[..., :3]
    base_f = np.asarray(base_rgb, dtype=np.float32)
    out = rgb_o * alpha + base_f * (1.0 - alpha)
    return (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)


def _composite_rgba_over_osm(
    rgba: object,
    dst_transform,
    dst_shape: tuple[int, int],
    west: float,
    south: float,
    east: float,
    north: float,
    *,
    zoom_adjust: int | None,
) -> object:
    import contextily as ctx
    gh, gw = int(dst_shape[0]), int(dst_shape[1])
    if np.asarray(rgba).shape[:2] != (gh, gw):
        raise ValueError(f"RGBA shape mismatch vs dst_shape ({gh}, {gw})")

    img_merc, extent = ctx.bounds2img(
        west,
        south,
        east,
        north,
        zoom="auto",
        ll=True,
        source=ctx.providers.OpenStreetMap.Mapnik,
        zoom_adjust=zoom_adjust,
    )
    base_rgb = _reproject_basemap_to_overlay_grid(img_merc, extent, dst_transform, (gh, gw))
    return _alpha_composite_rgb_under_rgba(base_rgb, rgba)


def _plot_cams_grid(
    ax,
    grid_fc: dict,
    *,
    color: str = "#0d47a1",
    lw: float = 1.35,
    alpha: float = 0.92,
) -> None:
    """Draw CAMS cell outlines from :func:`build_cams_area_grid_geojson_for_view`."""
    try:
        from shapely.geometry import shape
    except ImportError:
        print("WARNING: shapely not available; skipping CAMS grid.", file=sys.stderr)
        return

    for ft in grid_fc.get("features") or []:
        geom = ft.get("geometry")
        if not geom:
            continue
        g = shape(geom)
        if g.is_empty:
            continue
        if g.geom_type == "Polygon":
            x, y = g.exterior.xy
            ax.plot(x, y, color=color, lw=lw, alpha=alpha, zorder=10, solid_capstyle="round")
        elif g.geom_type == "MultiPolygon":
            for part in g.geoms:
                x, y = part.exterior.xy
                ax.plot(x, y, color=color, lw=lw, alpha=alpha, zorder=10, solid_capstyle="round")


def _weight_log_cbar_range(
    w_arr: object,
    w_nodata: float | None,
) -> tuple[float, float] | None:
    """Min/max of log10(weight) over positive finite pixels (matches ``scalar_to_rgba`` log mode)."""
    z = np.asarray(w_arr, dtype=np.float64)
    finite = np.isfinite(z)
    if w_nodata is not None:
        finite = finite & (z != float(w_nodata))
    valid = finite & (z > 0)
    if not np.any(valid):
        return None
    lv = np.log10(np.maximum(z[valid], 1e-18))
    lo, hi = float(np.min(lv)), float(np.max(lv))
    if hi <= lo:
        hi = lo + 1e-9
    return lo, hi


def _save_layer_png(
    img: object,
    *,
    title: str,
    west: float,
    south: float,
    east: float,
    north: float,
    out_path: Path,
    dpi: int,
    grid_fc: dict | None,
    weight_values_for_cbar: object | None = None,
    weight_nodata: float | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    rgba = np.asarray(img)
    gh, gw = int(rgba.shape[0]), int(rgba.shape[1])
    if rgba.ndim != 3 or rgba.shape[2] not in (3, 4):
        raise ValueError(f"Expected HxWx3 or HxWx4 image, got {rgba.shape}")
    lon_extent = east - west
    lat_extent = north - south
    lat_c = np.radians((south + north) / 2.0)
    geo_h_over_w = lat_extent / (lon_extent * np.cos(lat_c))

    fig_w = max(6.0, gw / dpi)
    fig_h = max(4.5, fig_w * geo_h_over_w + 0.85)
    if weight_values_for_cbar is not None:
        fig_w += 1.1

    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=dpi)

    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    ax.set_aspect(1.0 / np.cos(lat_c), adjustable="box")

    interp = "bilinear" if rgba.shape[2] == 3 else "nearest"
    ax.imshow(
        rgba,
        extent=[west, east, south, north],
        origin="upper",
        interpolation=interp,
        zorder=1,
    )

    if grid_fc:
        _plot_cams_grid(ax, grid_fc)

    ax.set_xlabel("Longitude (deg E)", fontsize=9)
    ax.set_ylabel("Latitude (deg N)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.grid(True, color="white", linestyle="--", linewidth=0.35, alpha=0.45, zorder=5)

    if weight_values_for_cbar is not None:
        cr = _weight_log_cbar_range(weight_values_for_cbar, weight_nodata)
        if cr is not None:
            lo, hi = cr
            sm = ScalarMappable(
                cmap="plasma",
                norm=Normalize(vmin=lo, vmax=hi),
            )
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
            cbar.set_label(r"$\log_{10}$(weight)", fontsize=8)
            cbar.ax.tick_params(labelsize=7)

    fig.tight_layout(pad=0.45)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main() -> int:
    root = _ensure_import_path()

    ap = argparse.ArgumentParser(
        description=(
            "Export three A_PublicPower maps (eligible CORINE land cover, population proxy, weights) "
            "with CAMS GNFR A area grid outlines."
        )
    )
    ap.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        required=True,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        help="WGS84 bounding box in degrees: west south east north.",
    )
    ap.add_argument("--root", type=Path, default=root)
    ap.add_argument("--paths-yaml", type=Path, default=None)
    ap.add_argument(
        "--sector-yaml",
        type=Path,
        default=None,
        help="Sector YAML (default: PROXY/config/sectors/publicpower.yaml).",
    )
    ap.add_argument(
        "--weight-tif",
        type=Path,
        default=None,
        help="Public power weight GeoTIFF (default: OUTPUT/Proxy_weights/A_PublicPower/publicpower_areasource.tif).",
    )
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--max-width", type=int, default=1400)
    ap.add_argument("--max-height", type=int, default=1200)
    ap.add_argument("--pad-deg", type=float, default=0.0)
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument(
        "--cams-nc",
        type=Path,
        default=None,
        help="CAMS emissions NetCDF (default: resolved from paths.yaml emissions.cams_2019_nc).",
    )
    ap.add_argument(
        "--skip-cams-grid",
        action="store_true",
        help="Do not draw CAMS cell outlines (skip NetCDF read).",
    )
    ap.add_argument(
        "--no-basemap",
        action="store_true",
        help="Do not fetch OSM tiles (transparent areas stay empty / white background only).",
    )
    ap.add_argument(
        "--basemap-zoom-adjust",
        type=int,
        default=None,
        help="Optional zoom tweak for contextily tile fetch (-1..1 recommended).",
    )
    args = ap.parse_args()

    paths_yaml = args.paths_yaml or (args.root / "PROXY" / "config" / "paths.yaml")
    sector_yaml = args.sector_yaml or (args.root / "PROXY" / "config" / "sectors" / "publicpower.yaml")
    wt_default = args.root / "OUTPUT" / "Proxy_weights" / "A_PublicPower" / "publicpower_areasource.tif"
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
        print("ERROR: YAML configs must be mappings.", file=sys.stderr)
        return 1

    west, south, east, north = (float(x) for x in args.bbox)
    if west >= east or south >= north:
        print("ERROR: require west < east and south < north.", file=sys.stderr)
        return 1

    out_dir = (args.out_dir.resolve() if args.out_dir else Path.cwd().resolve())
    out_dir.mkdir(parents=True, exist_ok=True)

    area_proxy = _viz_cfg_from_sector(sector_cfg)
    corine_band = int(area_proxy.get("corine_band", 1))
    codes_raw = area_proxy.get("corine_codes", [3, 121])
    industrial_codes = tuple(int(x) for x in codes_raw)

    from PROXY.core.dataloaders import resolve_path
    from PROXY.core.dataloaders.discovery import discover_cams_emissions, discover_corine
    from PROXY.sectors.A_PublicPower.cams_area_mask import public_power_area_mask
    from PROXY.visualization._mapbuilder import (
        compute_view_context,
        pick_band_by_pollutant,
        pick_first_positive_band,
        resolve_under_root,
        build_cams_area_grid_geojson_for_view,
    )
    from PROXY.visualization.corine_rgba import corine_clc_overlay_rgba
    from PROXY.visualization.overlay_utils import read_weight_corine_population_via_weight_grid_wgs84, scalar_to_rgba
    import xarray as xr

    wt_resolved = resolve_under_root(wt, args.root)
    corine_tif = discover_corine(
        args.root, resolve_path(args.root, Path(path_cfg["proxy_common"]["corine_tif"]))
    )
    pop_tif = resolve_path(args.root, Path(path_cfg["proxy_common"]["population_tif"]))

    if not corine_tif.is_file():
        print(f"ERROR: CORINE raster not found: {corine_tif}", file=sys.stderr)
        return 1
    if not pop_tif.is_file():
        print(f"ERROR: population raster not found: {pop_tif}", file=sys.stderr)
        return 1

    view = compute_view_context(
        wt_resolved,
        pad_deg=float(args.pad_deg),
        max_width=int(args.max_width),
        max_height=int(args.max_height),
        override_bbox=(west, south, east, north),
    )

    wb = pick_band_by_pollutant(wt_resolved, area_proxy)
    wb, _ = pick_first_positive_band(
        wt_resolved,
        wb,
        empty_message="All weight bands non-positive; using preferred band anyway.",
    )

    stacked = read_weight_corine_population_via_weight_grid_wgs84(
        wt_resolved,
        corine_path=corine_tif,
        corine_band=corine_band,
        population_path=pop_tif,
        population_band=1,
        west=view.west,
        south=view.south,
        east=view.east,
        north=view.north,
        display_width=view.gw,
        display_height=view.gh,
        weight_band=int(wb),
    )

    w_arr = stacked["weight_wgs84"]
    w_nd = stacked["weight_nodata"]
    clc_raw = stacked["corine_wgs84"]
    clc_nd = stacked["corine_nodata"]
    pop_arr = stacked["population_wgs84"]
    pop_nd = stacked["population_nodata"]

    clc_i = np.full((view.gh, view.gw), -1, dtype=np.int32)
    ok = np.isfinite(clc_raw)
    if clc_nd is not None:
        ok = ok & (clc_raw != float(clc_nd))
    clc_i[ok] = np.rint(clc_raw[ok]).astype(np.int32)
    rgba_elig = corine_clc_overlay_rgba(clc_i, highlight_codes=industrial_codes)

    rgba_pop = scalar_to_rgba(
        pop_arr,
        colour_mode="log",
        cmap_name="YlOrRd",
        hide_zero=True,
        nodata_val=float(pop_nd) if pop_nd is not None else None,
    )

    rgba_w = scalar_to_rgba(
        w_arr,
        colour_mode="log",
        cmap_name="plasma",
        hide_zero=True,
        nodata_val=float(w_nd) if w_nd is not None else None,
    )

    grid_fc: dict | None = None
    iso3 = str(sector_cfg.get("cams_country_iso3", "GRC")).strip().upper()

    if not args.skip_cams_grid:
        nc_path = args.cams_nc
        if nc_path is None:
            em = path_cfg.get("emissions") or {}
            nc_rel = em.get("cams_2019_nc")
            if not nc_rel:
                print("WARNING: paths.yaml has no emissions.cams_2019_nc; skipping CAMS grid.", file=sys.stderr)
            else:
                nc_path = discover_cams_emissions(
                    args.root, resolve_path(args.root, Path(str(nc_rel)))
                )
        else:
            nc_path = nc_path if nc_path.is_absolute() else args.root / nc_path

        if nc_path is not None and nc_path.is_file():
            ds = xr.open_dataset(nc_path, engine="netcdf4")
            try:
                m_area = public_power_area_mask(ds, iso3)
                grid_fc = build_cams_area_grid_geojson_for_view(ds, m_area, view)
            finally:
                ds.close()
        elif not args.skip_cams_grid:
            print(f"WARNING: CAMS NetCDF not found or unreadable; skipping grid. ({nc_path})", file=sys.stderr)

    exports: tuple[tuple[str, str, object], ...] = (
        ("Eligible CORINE (industrial / commercial codes)", "publicpower_bbox_eligible_landcover.png", rgba_elig),
        ("Population proxy (JRC, log scale)", "publicpower_bbox_population_proxy.png", rgba_pop),
        ("CAMS area weights (GNFR A, log scale)", "publicpower_bbox_weights.png", rgba_w),
    )

    print(
        f"View: {view.gw}x{view.gh} px | "
        f"[{view.west:.4f}, {view.south:.4f}, {view.east:.4f}, {view.north:.4f}] | "
        f"weight band {wb}"
    )

    try:
        import contextily  # noqa: F401
    except ImportError:
        contextily = None
    use_basemap = (not args.no_basemap) and (contextily is not None)
    if not args.no_basemap and contextily is None:
        print(
            "WARNING: contextily not installed; saving without OSM basemap. pip install contextily",
            file=sys.stderr,
        )

    for layer_title, fname, rgba in exports:
        print(f"\n{layer_title}")
        out_img: object = rgba
        if use_basemap:
            try:
                out_img = _composite_rgba_over_osm(
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
                print(f"  WARNING: basemap failed ({exc}); saving overlay only.", file=sys.stderr)
                out_img = rgba

        _save_layer_png(
            out_img,
            title=f"A_PublicPower — {layer_title}",
            west=view.west,
            south=view.south,
            east=view.east,
            north=view.north,
            out_path=out_dir / fname,
            dpi=int(args.dpi),
            grid_fc=grid_fc,
            weight_values_for_cbar=w_arr if fname == "publicpower_bbox_weights.png" else None,
            weight_nodata=w_nd if fname == "publicpower_bbox_weights.png" else None,
        )

    if use_basemap:
        print(
            "\nBasemap: OpenStreetMap – https://www.openstreetmap.org/copyright",
            file=sys.stderr,
        )

    if grid_fc is not None:
        n = len(grid_fc.get("features") or [])
        print(f"\nCAMS grid: {n} cell outlines (GNFR A area, {iso3}).")
    else:
        print("\nCAMS grid: not drawn.")

    print(f"\nOutput directory: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
