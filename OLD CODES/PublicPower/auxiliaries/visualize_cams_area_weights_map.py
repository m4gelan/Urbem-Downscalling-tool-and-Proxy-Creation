#!/usr/bin/env python3
"""
Folium map: CAMS area weight **polygons** (GeoJSON) plus the **same** CORINE + LandScan
rasters used to build weights (from ``*_manifest.json``).

Basemaps: CartoDB Positron + Esri World Imagery. Toggle raster layers to compare with
``weight_share`` fill on CORINE pixel footprints.

Usage (from project root)::

  python PublicPower/Auxiliaries/cams_area_downscale_corine_landscan.py
  python PublicPower/Auxiliaries/visualize_cams_area_weights_map.py
  python PublicPower/Auxiliaries/visualize_cams_area_weights_map.py --colour-mode per_cell

``--colour-mode``: ``global`` (default), ``log`` (log10 share), ``per_cell`` (0–1 within each CAMS cell).

Requires: folium, branca, geopandas, rasterio, matplotlib, numpy
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_context_map_helpers():
    p = Path(__file__).resolve().parent / "greece_public_power_context_map.py"
    spec = importlib.util.spec_from_file_location("_gpp_ctx_map", p)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {p}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    root = _project_root()
    ap = argparse.ArgumentParser(
        description="Map weight polygons + CORINE/LandScan overlays (manifest-driven).",
    )
    ap.add_argument(
        "--geojson",
        type=Path,
        default=root / "PublicPower" / "outputs" / "cams_area_weights_athens.geojson",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="JSON from export (default: <geojson stem>_manifest.json)",
    )
    ap.add_argument(
        "--out-html",
        type=Path,
        default=root / "PublicPower" / "outputs" / "cams_area_weights_athens_map.html",
    )
    ap.add_argument(
        "--pad-deg",
        type=float,
        default=0.015,
        help="Padding around manifest bbox for raster overlays",
    )
    ap.add_argument(
        "--colour-mode",
        choices=("global", "log", "per_cell"),
        default="global",
        help=(
            "Polygon fill: global=linear share across map; log=log10(share) for wide "
            "dynamic range; per_cell=0-1 stretch within each cams_source_index"
        ),
    )
    args = ap.parse_args()
    gj = args.geojson if args.geojson.is_absolute() else root / args.geojson
    out = args.out_html if args.out_html.is_absolute() else root / args.out_html

    if not gj.is_file():
        raise SystemExit(
            f"GeoJSON not found: {gj}\n"
            "Run: python PublicPower/Auxiliaries/cams_area_downscale_corine_landscan.py"
        )

    manifest_path = args.manifest
    if manifest_path is None:
        manifest_path = gj.with_name(gj.stem + "_manifest.json")
    elif not manifest_path.is_absolute():
        manifest_path = root / manifest_path

    try:
        import folium
        import geopandas as gpd
        from branca.colormap import LinearColormap
        from rasterio.transform import from_bounds
    except ImportError as exc:
        raise SystemExit("Need folium, branca, geopandas, rasterio.") from exc

    gdf = gpd.read_file(gj)
    if gdf.empty:
        raise SystemExit("GeoJSON is empty.")
    gdf = gdf.to_crs(4326)

    w = gdf["weight_share"].to_numpy(dtype=np.float64)
    wpos = w[w > 0]
    colours = ["#0d0887", "#6a00a8", "#b12a90", "#e16462", "#fca636", "#f0f921"]
    colour_mode = str(args.colour_mode)

    per_cell_bounds: dict[int, tuple[float, float]] = {}
    if colour_mode == "per_cell":
        for sid, grp in gdf.groupby("cams_source_index", sort=False):
            xs = grp["weight_share"].to_numpy(dtype=np.float64)
            per_cell_bounds[int(sid)] = (float(xs.min()), float(xs.max()))
        cmap = LinearColormap(
            colours,
            vmin=0.0,
            vmax=1.0,
            caption="Colour = rank within each CAMS cell (0=min share, 1=max)",
        )
    elif colour_mode == "log":
        eps = 1e-18
        if wpos.size:
            lw = np.log10(np.maximum(wpos, eps))
            log_lo, log_hi = float(np.min(lw)), float(np.max(lw))
            if log_lo >= log_hi:
                log_hi = log_lo + 1e-6
        else:
            log_lo, log_hi = -10.0, 0.0
        cmap = LinearColormap(
            colours,
            vmin=log_lo,
            vmax=log_hi,
            caption="log10(weight_share) — global scale",
        )
    else:
        if wpos.size:
            lo, hi = float(np.min(wpos)), float(np.max(wpos))
            if lo >= hi:
                hi = lo * 1.001 if lo > 0 else 1e-9
        else:
            lo, hi = 0.0, 1.0
        cmap = LinearColormap(
            colours,
            vmin=lo,
            vmax=hi,
            caption="weight_share — global linear scale",
        )

    b = gdf.total_bounds
    pad = float(args.pad_deg)
    south = float(b[1]) - pad
    west = float(b[0]) - pad
    north = float(b[3]) + pad
    east = float(b[2]) + pad

    fmap = folium.Map(
        location=[(south + north) / 2, (west + east) / 2],
        zoom_start=11,
        tiles=None,
    )
    folium.TileLayer(
        "CartoDB positron",
        name="Light (CartoDB Positron)",
        control=True,
    ).add_to(fmap)
    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        attr=(
            "Tiles &copy; Esri &mdash; "
            "Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community"
        ),
        name="Satellite (Esri World Imagery)",
        max_zoom=19,
        control=True,
    ).add_to(fmap)

    if manifest_path.is_file():
        with manifest_path.open(encoding="utf-8") as f:
            man = json.load(f)
        bw, bs, be, bn = [float(x) for x in man["domain_bbox_wgs84"]]
        south_m = min(south, bs - pad)
        west_m = min(west, bw - pad)
        north_m = max(north, bn + pad)
        east_m = max(east, be + pad)
        south, west, north, east = south_m, west_m, north_m, east_m

        mo = man.get("map_overlay") or {}
        gw = int(mo.get("grid_width", 950))
        gh = int(mo.get("grid_height", 820))
        corine_p = Path(man["corine_geotiff"])
        ls_p = Path(man["landscan_geotiff"])
        corine_band = int(man.get("corine_band", 1))
        codes = tuple(int(x) for x in man.get("corine_codes", [121, 3]))
        ls_res = str(man.get("landscan_resampling", "bilinear"))

        if corine_p.is_file() and ls_p.is_file():
            ctx = _load_context_map_helpers()
            dst_t = from_bounds(west, south, east, north, gw, gh)
            mask_arr = np.ones((gh, gw), dtype=np.uint8)
            pop = ctx._reproject_band_to_wgs84_grid(
                ls_p,
                dst_t,
                (gh, gw),
                resampling=ls_res,
            )
            pop_rgba = ctx.population_to_rgba(pop, mask_arr)
            fg_ls = folium.FeatureGroup(
                name=f"LandScan (weights use {ls_res} warp to CORINE)",
                show=False,
            )
            folium.raster_layers.ImageOverlay(
                image=pop_rgba,
                bounds=[[south, west], [north, east]],
                mercator_project=True,
                opacity=0.92,
                name="LandScan",
                interactive=False,
                cross_origin=False,
            ).add_to(fg_ls)
            fg_ls.add_to(fmap)

            clc = ctx._reproject_band_to_wgs84_grid(
                corine_p,
                dst_t,
                (gh, gw),
                resampling="nearest",
                band=corine_band,
            )
            u = ctx.industry_corine_mask_from_class(
                clc, mask_arr, class_codes=codes
            )
            cor_rgba = ctx.industry_mask_to_rgba(u)
            fg_cor = folium.FeatureGroup(
                name=f"CORINE classes {codes} mask (nearest; same band as weights)",
                show=False,
            )
            folium.raster_layers.ImageOverlay(
                image=cor_rgba,
                bounds=[[south, west], [north, east]],
                mercator_project=True,
                opacity=0.55,
                name="CORINE",
                interactive=False,
                cross_origin=False,
            ).add_to(fg_cor)
            fg_cor.add_to(fmap)
        else:
            print(
                f"Manifest rasters missing; overlays skipped:\n  {corine_p}\n  {ls_p}",
                file=sys.stderr,
            )
    else:
        print(
            f"No manifest at {manifest_path}; CORINE/LandScan overlays skipped. Re-export.",
            file=sys.stderr,
        )

    def _style(feat: dict) -> dict:
        props = feat.get("properties") or {}
        sh = float(props.get("weight_share", 0.0))
        if colour_mode == "per_cell":
            sid = int(float(props.get("cams_source_index", 0)))
            lo, hi = per_cell_bounds.get(sid, (sh, sh))
            if hi <= lo:
                t = 0.5
            else:
                t = (sh - lo) / (hi - lo)
            fill = cmap(float(np.clip(t, 0.0, 1.0)))
        elif colour_mode == "log":
            eps = 1e-18
            fill = cmap(float(np.log10(max(sh, eps))))
        else:
            fill = cmap(sh)
        return {
            "fillColor": fill,
            "color": "#1a1a1a",
            "weight": 0.35,
            "fillOpacity": 0.5,
        }

    layer_bits = {
        "global": "linear global",
        "log": "log10 global",
        "per_cell": "per-CAMS-cell 0-1",
    }.get(colour_mode, colour_mode)
    fg_w = folium.FeatureGroup(
        name=f"Weights ({layer_bits})",
        show=True,
    )
    folium.GeoJson(
        data=gdf.to_json(),
        style_function=_style,
        tooltip=folium.GeoJsonTooltip(
            fields=[
                "cams_source_index",
                "weight_share",
                "weight_raw",
                "landscan_pop",
                "corine_value",
                "weight_basis",
            ],
            aliases=[
                "CAMS src",
                "share",
                "raw w",
                "LandScan",
                "CORINE cls",
                "basis",
            ],
            localize=True,
        ),
    ).add_to(fg_w)
    fg_w.add_to(fmap)

    cmap.add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    out.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out))
    print(f"Wrote {out}")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
