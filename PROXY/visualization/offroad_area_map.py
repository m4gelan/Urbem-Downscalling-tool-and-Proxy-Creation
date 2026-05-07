"""Folium preview for I_Offroad: weights, OSM rail/pipeline, non-road CORINE proxy (aligned to weight grid)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from PROXY.sectors.I_Offroad.proxy_rules import osm_railway_line_filter_sets
from PROXY.sectors.I_Offroad.cams_area_mask import offroad_union_area_mask
from PROXY.sectors.I_Offroad.offroad_area_weights import parse_offroad_subsector_activity
from PROXY.sectors.I_Offroad.pipeline_osm import (
    load_pipeline_facilities,
    load_pipeline_lines,
    load_pipeline_union,
)
from PROXY.sectors.I_Offroad.rail_osm import filter_rail_lines
from PROXY.visualization._legend import (
    LegendSection,
    build_weight_legend_section,
    categorical_swatch_html,
    region_note,
    render_unified_legend,
)
from PROXY.visualization._mapbuilder import (
    VIZ_DEPS_MESSAGE,
    add_cams_grid_overlay,
    add_raster_overlay,
    build_cams_area_grid_geojson_for_view,
    compute_view_context,
    create_folium_map_with_tiles,
    pick_band_by_pollutant,
    pollutant_label_for_band,
    resolve_under_root,
    save_folium_map,
)
from PROXY.visualization._click_popup import enrich_cams_grid_with_popups
from PROXY.visualization._multipollutant import (
    add_multipollutant_weight_layers,
    visualization_pollutant_priority_from_cfg,
)
from PROXY.visualization.corine_rgba import corine_clc_overlay_rgba
from PROXY.visualization.overlay_utils import read_corine_clc_wgs84_on_weight_grid


def _expand_level2_highlights_for_clc_grid(
    clc_i: np.ndarray,
    level2_codes: tuple[int, ...],
) -> tuple[int, ...]:
    """Map YAML Level-2 codes to legend codes in ``clc_i`` (CLC 1–44 vs Level-2), same as proxy."""
    if not level2_codes:
        return ()
    from PROXY.core.osm_corine_proxy import adapt_corine_classes_for_grid

    # Engine uses -9999 nodata; viz grid uses -1 — align so max/classification match CORINE legend.
    clc_adapt = np.where(np.asarray(clc_i) >= 0, clc_i, -9999).astype(np.int32)
    expanded: list[int] = []
    for c in level2_codes:
        codes, _ = adapt_corine_classes_for_grid(clc_adapt, [int(c)])
        for x in codes:
            expanded.append(int(x))
    out = tuple(sorted(set(expanded)))
    if not out:
        return tuple(int(x) for x in level2_codes)
    return out

_OFFROAD_COLOR_NONROAD = (106, 27, 154)
_OFFROAD_COLOR_PIPELINE = (183, 28, 28)
_OFFROAD_COLOR_PIPELINE_FACILITY = (230, 81, 0)
_OFFROAD_COLOR_RAILWAY = (13, 71, 161)

# Non-road residual CLC L2 codes (1A3eii weights) — distinct RGB for per-layer previews.
_NONROAD_CLC_LAYER_RGB: dict[int, tuple[int, int, int]] = {
    121: (183, 28, 28),
    123: (230, 81, 0),
    124: (0, 151, 167),
    131: (121, 85, 72),
}
_NONROAD_CLC_TITLES: dict[int, str] = {
    121: "industrial / commercial",
    123: "port areas",
    124: "airports",
    131: "mineral extraction",
}


def _nonroad_clc_weight_label(area_proxy: dict[str, Any], code: int) -> str:
    p = area_proxy.get("proxy") or {}
    nw = p.get("nonroad_corine_weights") or {}
    for k, v in nw.items():
        try:
            if int(k) == int(code):
                return f"w={float(v):.2f}"
        except (TypeError, ValueError):
            continue
    return ""


def _split_facilities_points_other(gdf: Any) -> tuple[Any, Any]:
    """Compressors as points vs lines/polygons for separate map styling."""
    if gdf is None or getattr(gdf, "empty", True):
        return None, None
    gt = gdf.geometry.geom_type
    m_pt = gt.isin(["Point", "MultiPoint"])
    pts = gdf.loc[m_pt].copy()
    other = gdf.loc[~m_pt].copy()
    return (pts if not pts.empty else None, other if not other.empty else None)


def _iter_point_latlon(gdf: Any) -> Any:
    """Yield (lat, lon) for Point / MultiPoint rows for Folium markers."""
    if gdf is None or getattr(gdf, "empty", True):
        return
    for _, row in gdf.iterrows():
        g = row.geometry
        if g is None or g.is_empty:
            continue
        if g.geom_type == "Point":
            yield float(g.y), float(g.x)
        elif g.geom_type == "MultiPoint":
            for p in g.geoms:
                if p.is_empty:
                    continue
                yield float(p.y), float(p.x)


def _rasterize_lines_on_view(
    gdf: Any,
    view: Any,
    *,
    dilate_px: int = 1,
) -> np.ndarray | None:
    """Rasterise a WGS84 line ``GeoDataFrame`` to a boolean mask on ``view``'s grid.

    Lines are typically 1-pixel wide at display resolution which makes them
    invisible once we paint an RGBA on top. A small binary dilation thickens
    them to ``dilate_px * 2 + 1`` pixels so the combined layer reads like the
    individual coloured line overlays.
    """
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


def _build_offroad_combined_rgba(
    nonroad_mask: np.ndarray,
    pipeline_mask: np.ndarray | None,
    railway_mask: np.ndarray | None,
    *,
    alpha_nonroad: int = 170,
    alpha_lines: int = 230,
) -> np.ndarray | None:
    """Paint one RGBA where lines (pipelines/railways) override the CORINE base."""
    h, w = int(nonroad_mask.shape[0]), int(nonroad_mask.shape[1])
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    painted = False
    if nonroad_mask is not None and bool(nonroad_mask.any()):
        r, g, b = _OFFROAD_COLOR_NONROAD
        rgba[nonroad_mask, 0] = r
        rgba[nonroad_mask, 1] = g
        rgba[nonroad_mask, 2] = b
        rgba[nonroad_mask, 3] = int(alpha_nonroad)
        painted = True
    if pipeline_mask is not None and bool(pipeline_mask.any()):
        r, g, b = _OFFROAD_COLOR_PIPELINE
        rgba[pipeline_mask, 0] = r
        rgba[pipeline_mask, 1] = g
        rgba[pipeline_mask, 2] = b
        rgba[pipeline_mask, 3] = int(alpha_lines)
        painted = True
    if railway_mask is not None and bool(railway_mask.any()):
        r, g, b = _OFFROAD_COLOR_RAILWAY
        rgba[railway_mask, 0] = r
        rgba[railway_mask, 1] = g
        rgba[railway_mask, 2] = b
        rgba[railway_mask, 3] = int(alpha_lines)
        painted = True
    return rgba if painted else None


def _emission_indices_from_area_proxy(area_proxy: dict[str, Any]) -> tuple[int, ...]:
    raw = area_proxy.get("cams_emission_category_indices")
    if isinstance(raw, (list, tuple)) and len(raw) > 0:
        return tuple(sorted({int(x) for x in raw}))
    one = area_proxy.get("cams_emission_category_index")
    if one is not None:
        return (int(one),)
    act = area_proxy.get("subsector_activity")
    if isinstance(act, dict) and act:
        rows = parse_offroad_subsector_activity(area_proxy)
        return tuple(sorted({int(r["cams_emission_category_index"]) for r in rows}))
    return (12,)


def _nonroad_corine_codes_from_proxy(area_proxy: dict[str, Any]) -> tuple[int, ...]:
    p = area_proxy.get("proxy") or {}
    nw = p.get("nonroad_corine_weights") or {}
    if nw:
        try:
            return tuple(sorted({int(k) for k in nw.keys()}))
        except (TypeError, ValueError):
            pass
    codes: list[int] = []
    for k in ("corine_agri_codes", "corine_agri_optional", "corine_ind_codes", "corine_ind_optional"):
        for x in (p.get(k) or []):
            codes.append(int(x))
    return tuple(sorted(set(codes)))


def _osm_layer_feature_count(path: Path, layer: str) -> int | None:
    """Fast feature count from GeoPackage (pyogrio); ``None`` if unavailable."""
    try:
        import pyogrio as pg

        meta = pg.read_info(str(path), layer=layer)
        return int(meta.get("features", 0))
    except Exception:
        return None


def _full_osm_geojson_wgs84(gdf: Any, *, max_features: int = 120_000) -> dict[str, Any] | None:
    """Whole layer as GeoJSON in WGS84 (no map bbox clip)."""
    import geopandas as gpd

    if gdf is None or getattr(gdf, "empty", True):
        return None
    g4326 = gdf.to_crs(4326)
    if len(g4326) > max_features:
        g4326 = g4326.iloc[:max_features].copy()
    return json.loads(g4326.to_json())


def _clip_osm_lines_geojson(
    gdf: Any,
    bbox_wgs84: Any,
    *,
    max_features: int = 120_000,
) -> dict[str, Any] | None:
    import geopandas as gpd

    if gdf is None or getattr(gdf, "empty", True):
        return None
    g4326 = gdf.to_crs(4326)
    clipped = gpd.clip(g4326, bbox_wgs84)
    if clipped.empty:
        return None
    if len(clipped) > max_features:
        clipped = clipped.iloc[:max_features].copy()
    return json.loads(clipped.to_json())


def write_offroad_area_html(
    *,
    root: Path,
    weight_tif: Path,
    corine_tif: Path,
    population_tif: Path | None,
    out_html: Path,
    area_proxy: dict[str, Any] | None = None,
    path_cfg: dict[str, Any] | None = None,
    pad_deg: float = 0.02,
    max_width: int = 1400,
    max_height: int = 1200,
    weight_opacity: float = 0.9,
    context_opacity: float = 0.75,
    cams_nc_path: Path | None = None,
    cams_country_iso3: str = "GRC",
    weight_display_mode: str = "global_log",
    region: str | None = None,
    override_bbox: tuple[float, float, float, float] | None = None,
) -> Path:
    """
    Folium map: basemaps, non-road CORINE mask (agri+ind CLC from config), OSM pipelines and railways,
    area weights, optional CAMS grid. No population layer; no extra CORINE "context" stacks.
    """
    _ = population_tif

    try:
        import folium
        import geopandas as gpd
        from shapely.geometry import box
    except ImportError as exc:
        raise SystemExit(
            VIZ_DEPS_MESSAGE.replace(
                "folium, branca, rasterio, matplotlib",
                "folium, branca, rasterio, geopandas, shapely",
            )
        ) from exc

    area_proxy = area_proxy or {}
    emission_indices = _emission_indices_from_area_proxy(area_proxy)
    source_type_index = int(area_proxy.get("source_type_index", 1))
    corine_band = int(area_proxy.get("corine_band", 1))
    nonroad_codes = _nonroad_corine_codes_from_proxy(area_proxy)

    wt = resolve_under_root(weight_tif, root)
    ct = resolve_under_root(corine_tif, root)

    if not wt.is_file():
        raise FileNotFoundError(f"Weight GeoTIFF not found: {wt}")
    if not ct.is_file():
        raise FileNotFoundError(f"CORINE GeoTIFF not found: {ct}")

    weight_band = pick_band_by_pollutant(wt, area_proxy, sector_cfg=area_proxy)

    view = compute_view_context(
        wt, pad_deg=pad_deg, max_width=max_width, max_height=max_height,
        region=region, override_bbox=override_bbox,
    )
    bbox_wgs84 = box(view.west, view.south, view.east, view.north)

    stk_c = read_corine_clc_wgs84_on_weight_grid(
        wt,
        ct,
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

    has_cams = bool(cams_nc_path) and Path(cams_nc_path).is_file()
    cams_nc_display = Path(cams_nc_path).name if has_cams and cams_nc_path else ""
    use_per_cell = has_cams and weight_display_mode.strip().lower() == "per_cell"

    grid_fc: dict[str, Any] = {"type": "FeatureCollection", "features": []}
    m_area = None
    ds_handle = None
    if has_cams:
        ds_handle = xr.open_dataset(cams_nc_path, engine="netcdf4")
        try:
            m_area = offroad_union_area_mask(
                ds_handle,
                cams_country_iso3,
                emission_category_indices=emission_indices,
                source_type_index=source_type_index,
                domain_bbox_wgs84=None,
            )
            grid_fc = build_cams_area_grid_geojson_for_view(ds_handle, m_area, view)
        except Exception:
            ds_handle.close()
            raise

    clc_i = np.full((view.gh, view.gw), -1, dtype=np.int32)
    ok = np.isfinite(clc_raw)
    if clc_nd is not None:
        ok = ok & (clc_raw != float(clc_nd))
    clc_i[ok] = np.rint(clc_raw[ok]).astype(np.int32)

    highlight_union = _expand_level2_highlights_for_clc_grid(
        clc_i,
        tuple(int(x) for x in nonroad_codes),
    )

    fmap = create_folium_map_with_tiles(view, zoom_start=8)

    rgba_nr_union = corine_clc_overlay_rgba(
        clc_i,
        highlight_codes=highlight_union,
        rgb=(106, 27, 154),
    )
    add_raster_overlay(
        fmap,
        rgba_nr_union,
        view,
        name="Non-road — all residual CLC codes (union)",
        opacity=context_opacity,
        show=True,
    )

    for code in sorted(set(nonroad_codes)):
        icode = int(code)
        rgb_c = _NONROAD_CLC_LAYER_RGB.get(icode, (106, 27, 154))
        title = _NONROAD_CLC_TITLES.get(icode, "")
        wl = _nonroad_clc_weight_label(area_proxy, icode)
        suffix = f", {wl}" if wl else ""
        subt = f" ({title})" if title else ""
        exp_c = _expand_level2_highlights_for_clc_grid(clc_i, (icode,))
        rgba_c = corine_clc_overlay_rgba(
            clc_i,
            highlight_codes=exp_c,
            rgb=rgb_c,
        )
        add_raster_overlay(
            fmap,
            rgba_c,
            view,
            name=f"Non-road CLC {icode}{subt}{suffix}",
            opacity=context_opacity,
            show=False,
        )

    pipeline_lines_gdf: Any = None
    pipeline_facilities_gdf: Any = None
    pipeline_gdf: Any = None
    railway_gdf: Any = None
    rules = osm_railway_line_filter_sets(path_cfg or {})
    osm_rel = (path_cfg or {}).get("osm", {}).get("offroad") if path_cfg else None
    if not osm_rel:
        print("[I_Offroad viz] WARN: path_cfg has no osm.offroad — pipeline/rail OSM layers skipped", flush=True)
    if osm_rel:
        osm_path = Path(str(osm_rel))
        if not osm_path.is_absolute():
            osm_path = root / osm_path
        if not osm_path.is_file():
            print(f"[I_Offroad viz] WARN: OSM GPKG not found: {osm_path.resolve()}", flush=True)
        elif osm_path.is_file():
            n_gpkg_ln = _osm_layer_feature_count(osm_path, "osm_offroad_pipeline_lines")
            n_gpkg_fac = _osm_layer_feature_count(osm_path, "osm_offroad_pipeline_facilities")
            print(f"[I_Offroad viz] OSM GPKG: {osm_path.resolve()}", flush=True)
            print(
                f"[I_Offroad viz] GPKG counts — pipeline_lines={n_gpkg_ln}, "
                f"pipeline_facilities={n_gpkg_fac}",
                flush=True,
            )
            print(
                f"[I_Offroad viz] focus map bbox W,S,E,N (deg): "
                f"{view.west:.4f},{view.south:.4f},{view.east:.4f},{view.north:.4f}",
                flush=True,
            )

            try:
                pipeline_lines_gdf = load_pipeline_lines(osm_path)
                if getattr(pipeline_lines_gdf, "empty", True):
                    uni_try = load_pipeline_union(osm_path, root)
                    if uni_try is not None and not uni_try.empty:
                        mgeom = uni_try.geometry.geom_type.isin(
                            ["LineString", "MultiLineString"],
                        )
                        pipeline_lines_gdf = uni_try.loc[mgeom].copy()
                n_loaded = len(pipeline_lines_gdf) if pipeline_lines_gdf is not None else 0
                print(f"[I_Offroad viz] pipeline line geometries loaded for viz: {n_loaded}", flush=True)

                gj_ln = _clip_osm_lines_geojson(pipeline_lines_gdf, bbox_wgs84)
                n_clip = len(gj_ln.get("features", [])) if gj_ln else 0
                print(f"[I_Offroad viz] pipeline lines after bbox clip: {n_clip} features", flush=True)

                used_national_fallback = False
                if (not gj_ln or not gj_ln.get("features")) and n_loaded > 0:
                    tb = pipeline_lines_gdf.to_crs(4326).total_bounds
                    print(
                        "[I_Offroad viz] hint: 0 pipeline lines inside the preview bbox but "
                        f"{n_loaded} in GPKG — bounds WGS84 (minx,miny,maxx,maxy): "
                        f"{tuple(round(x, 4) for x in tb)}",
                        flush=True,
                    )
                    print(
                        "[I_Offroad viz] adding national pipeline GeoJSON (toggle layer; zoom out / pan). "
                        "Or rerun with  --region country  or  --bbox W,S,E,N",
                        flush=True,
                    )
                    gj_ln = _full_osm_geojson_wgs84(pipeline_lines_gdf)
                    used_national_fallback = bool(gj_ln and gj_ln.get("features"))

                if gj_ln and gj_ln.get("features"):
                    pl_name = (
                        "Pipeline lines — hydrocarbon (OSM, full extract — zoom out)"
                        if used_national_fallback
                        else "Pipeline lines — hydrocarbon (OSM)"
                    )
                    fg_ln = folium.FeatureGroup(name=pl_name, show=True)
                    folium.GeoJson(
                        gj_ln,
                        style_function=lambda _f: {
                            "color": "#b71c1c",
                            "weight": 2,
                            "opacity": 0.85,
                        },
                    ).add_to(fg_ln)
                    fg_ln.add_to(fmap)
            except Exception as exc:
                print(f"[I_Offroad viz] ERROR pipeline lines layer: {exc!r}", flush=True)
                pipeline_lines_gdf = None

            try:
                pipeline_facilities_gdf = load_pipeline_facilities(osm_path)
                nf = len(pipeline_facilities_gdf) if pipeline_facilities_gdf is not None else 0
                print(f"[I_Offroad viz] pipeline facilities geometries loaded: {nf}", flush=True)
                fac_pts, fac_other = _split_facilities_points_other(pipeline_facilities_gdf)
                if fac_other is not None:
                    gj_o = _clip_osm_lines_geojson(fac_other, bbox_wgs84)
                    nfo = len(gj_o.get("features", [])) if gj_o else 0
                    print(f"[I_Offroad viz] pipeline facility footprints/lines after clip: {nfo}", flush=True)
                    if gj_o and gj_o.get("features"):
                        fg_fo = folium.FeatureGroup(
                            name="Pipeline facilities — footprints / lines (OSM)",
                            show=True,
                        )
                        folium.GeoJson(
                            gj_o,
                            style_function=lambda _f: {
                                "color": "#e65100",
                                "weight": 2,
                                "opacity": 0.9,
                                "fillColor": "#ffe0b2",
                                "fillOpacity": 0.35,
                            },
                        ).add_to(fg_fo)
                        fg_fo.add_to(fmap)
                    elif nf > 0:
                        print(
                            "[I_Offroad viz] hint: facility polygons/lines exist but none inside preview bbox",
                            flush=True,
                        )
                if fac_pts is not None:
                    fac_wgs = fac_pts.to_crs(4326)
                    fac_clip = gpd.clip(fac_wgs, bbox_wgs84)
                    fg_fp = folium.FeatureGroup(
                        name="Pipeline facilities — compressor points (OSM)",
                        show=True,
                    )
                    n_mark = 0
                    for lat, lon in _iter_point_latlon(fac_clip):
                        folium.CircleMarker(
                            location=(lat, lon),
                            radius=6,
                            color="#e65100",
                            weight=2,
                            fill=True,
                            fill_color="#ffcc80",
                            fill_opacity=0.9,
                        ).add_to(fg_fp)
                        n_mark += 1
                    print(f"[I_Offroad viz] compressor points after bbox clip: {n_mark}", flush=True)
                    if n_mark > 0:
                        fg_fp.add_to(fmap)
            except Exception as exc:
                print(f"[I_Offroad viz] ERROR pipeline facilities: {exc!r}", flush=True)
                pipeline_facilities_gdf = None

            try:
                pipeline_gdf = load_pipeline_union(osm_path, root)
            except Exception as exc:
                print(f"[I_Offroad viz] WARN load_pipeline_union: {exc!r}", flush=True)
                pipeline_gdf = None

            try:
                rlines = gpd.read_file(osm_path, layer="osm_offroad_rail_lines")
                rlines = filter_rail_lines(
                    rlines,
                    bad_line_types=rules[0],
                    lifecycle_disallow=rules[1],
                )
                railway_gdf = rlines
                gj_r = _clip_osm_lines_geojson(rlines, bbox_wgs84)
                if gj_r and gj_r.get("features"):
                    fg_rl = folium.FeatureGroup(name="Railways (OSM)", show=True)
                    folium.GeoJson(
                        gj_r,
                        style_function=lambda _f: {
                            "color": "#0d47a1",
                            "weight": 2,
                            "opacity": 0.85,
                        },
                    ).add_to(fg_rl)
                    fg_rl.add_to(fmap)
            except Exception:
                railway_gdf = None

    nonroad_mask = np.zeros((view.gh, view.gw), dtype=bool)
    if highlight_union:
        nonroad_mask = np.isin(clc_i, np.asarray(highlight_union, dtype=np.int32))
    pm_ln = _rasterize_lines_on_view(pipeline_lines_gdf, view, dilate_px=1)
    pm_fac = _rasterize_lines_on_view(pipeline_facilities_gdf, view, dilate_px=3)
    if pm_ln is not None and pm_fac is not None:
        pipeline_mask = np.logical_or(pm_ln, pm_fac)
    elif pm_ln is not None:
        pipeline_mask = pm_ln
    elif pm_fac is not None:
        pipeline_mask = pm_fac
    else:
        pipeline_mask = _rasterize_lines_on_view(pipeline_gdf, view, dilate_px=1)
    railway_mask = _rasterize_lines_on_view(railway_gdf, view, dilate_px=1)

    combined_sources_present = False
    offroad_combined_rgba = _build_offroad_combined_rgba(
        nonroad_mask, pipeline_mask, railway_mask,
    )
    if offroad_combined_rgba is not None:
        combined_sources_present = True
        add_raster_overlay(
            fmap,
            offroad_combined_rgba,
            view,
            name="Off-road sources (combined: CORINE / pipelines / railways)",
            opacity=0.85,
            show=True,
        )

    pol_label = pollutant_label_for_band(wt, weight_band)
    _ppri, _pexc = visualization_pollutant_priority_from_cfg(area_proxy)
    mp = add_multipollutant_weight_layers(
        fmap,
        view,
        wt,
        preferred_band=weight_band,
        preferred_label=pol_label,
        sector_key="I_Offroad",
        display_mode=("per_cell" if use_per_cell else "global_log"),
        cmap="plasma",
        weight_opacity=weight_opacity,
        cams_nc_path=cams_nc_path if has_cams else None,
        m_area=m_area,
        cams_ds=ds_handle,
        clip_alpha_to_cams=has_cams,
        pollutant_priority=_ppri,
        exclusive_pollutant_panel=_pexc,
        max_bands=3,
    )
    if has_cams and ds_handle is not None:
        enrich_cams_grid_with_popups(
            grid_fc,
            view=view,
            m_area=m_area,
            ds=ds_handle,
            sector_title="I_Offroad",
        )
    if ds_handle is not None:
        ds_handle.close()
        ds_handle = None
    pol_label = mp.primary_label
    weight_band = mp.primary_band
    primary_w_arr = mp.bands_arrays[0] if mp.bands_arrays else None

    add_cams_grid_overlay(
        fmap,
        grid_fc,
        name="CAMS off-road area source grid (cell outlines)",
        show=False,
        colour="#0d47a1",
        weight=2.5,
        outline_opacity=0.95,
    )

    nr_label = ", ".join(str(c) for c in nonroad_codes) if nonroad_codes else "(none)"
    sections: list[LegendSection] = [
        LegendSection(
            title="I_Offroad - preview",
            html=(
                "<p class=\"pl-meta\">Weight raster: <code>"
                + wt.name
                + f"</code> (band {int(weight_band)} - <b>{pol_label}</b>)</p>"
                "<p class=\"pl-hint\">Toggle <b>CLC non-road</b> layers (per class), "
                "<b>pipeline lines</b>, <b>compressor points</b>, <b>railways</b>, and "
                "<b>weight</b> bands in the layer control. Pollutant weights follow "
                "<code>area_proxy.visualization_pollutant</code> when set (exclusive panel).</p>"
            ),
            open=True,
        ),
    ]
    sections.extend(mp.legend_sections)
    if primary_w_arr is not None:
        sections.append(
            build_weight_legend_section(
                primary_w_arr,
                display_mode=("per_cell" if use_per_cell else "global_log"),
                cmap="plasma",
                title=f"Weight scale (off-road - {pol_label})",
            )
        )
    if combined_sources_present:
        sections.append(
            LegendSection(
                title="Off-road sources (combined)",
                html=(
                    "<p class=\"pl-hint\">Raster overview: purple non-road CORINE mask; red pipeline geometry; "
                    "blue railways. Same palette as split layers above.</p>"
                    + categorical_swatch_html([
                        ("Non-road CLC (union)", "#6a1b9a"),
                        ("Pipeline lines", "#b71c1c"),
                        ("Compressor facilities", "#e65100"),
                        ("Railways", "#0d47a1"),
                    ])
                ),
                open=True,
            )
        )
    if has_cams:
        sections.append(
            LegendSection(
                title="CAMS grid",
                html=(
                    "<p class=\"pl-hint\">CAMS emission category indices for this sector: "
                    "<code>"
                    + ", ".join(str(i) for i in emission_indices)
                    + "</code></p>"
                    "<p class=\"pl-meta\">NetCDF: <code>" + cams_nc_display + "</code></p>"
                ),
                open=False,
            )
        )

    legend_html = render_unified_legend(
        "I_Offroad",
        sections,
        pollutant_badge=pol_label,
        region_note=region_note(region, override_bbox),
    )
    return save_folium_map(fmap, out_html, root, legend_html=legend_html)
