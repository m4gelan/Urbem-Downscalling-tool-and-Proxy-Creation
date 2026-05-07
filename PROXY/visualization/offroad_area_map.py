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
from PROXY.sectors.I_Offroad.pipeline_osm import load_pipeline_union
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

_OFFROAD_COLOR_NONROAD = (106, 27, 154)
_OFFROAD_COLOR_PIPELINE = (183, 28, 28)
_OFFROAD_COLOR_RAILWAY = (13, 71, 161)


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
    codes: list[int] = []
    for k in ("corine_agri_codes", "corine_agri_optional", "corine_ind_codes", "corine_ind_optional"):
        for x in (p.get(k) or []):
            codes.append(int(x))
    return tuple(sorted(set(codes)))


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

    rgba_nr = corine_clc_overlay_rgba(
        clc_i,
        highlight_codes=nonroad_codes,
        rgb=(106, 27, 154),
    )

    fmap = create_folium_map_with_tiles(view, zoom_start=8)

    add_raster_overlay(
        fmap,
        rgba_nr,
        view,
        name="Non-road proxy (CORINE agri + ind, CLC codes in legend)",
        opacity=context_opacity,
        show=False,
    )

    pipeline_gdf: Any = None
    railway_gdf: Any = None
    rules = osm_railway_line_filter_sets(path_cfg or {})
    osm_rel = (path_cfg or {}).get("osm", {}).get("offroad") if path_cfg else None
    if osm_rel:
        osm_path = Path(str(osm_rel))
        if not osm_path.is_absolute():
            osm_path = root / osm_path
        if osm_path.is_file():
            try:
                pipeline_gdf = load_pipeline_union(osm_path, root)
                gj_p = _clip_osm_lines_geojson(pipeline_gdf, bbox_wgs84)
                if gj_p and gj_p.get("features"):
                    fg_pl = folium.FeatureGroup(name="Pipelines (OSM)", show=False)
                    folium.GeoJson(
                        gj_p,
                        style_function=lambda _f: {
                            "color": "#b71c1c",
                            "weight": 2,
                            "opacity": 0.85,
                        },
                    ).add_to(fg_pl)
                    fg_pl.add_to(fmap)
            except Exception:
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
                    fg_rl = folium.FeatureGroup(name="Railways (OSM)", show=False)
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
    if nonroad_codes:
        nonroad_mask = np.isin(clc_i, np.asarray(nonroad_codes, dtype=np.int32))
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
                "<p class=\"pl-hint\">GNFR I / 12 off-road area proxies: CORINE agri + industrial "
                "classes (non-road base), OSM pipelines (red) and railways (blue), final weights on top. "
                "Good reference for lines; use <code>--region</code> for dense urban areas.</p>"
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
                    "<p class=\"pl-hint\">Single categorical layer; line sources are "
                    "drawn on top of the CORINE base (1-pixel dilation to keep them "
                    "visible at map scale). Non-road CORINE CLC codes: <code>"
                    + nr_label
                    + "</code>. Individual layers remain toggleable below.</p>"
                    + categorical_swatch_html([
                        ("Non-road CORINE (agri + ind)", "#6a1b9a"),
                        ("Pipelines (OSM)", "#b71c1c"),
                        ("Railways (OSM)", "#0d47a1"),
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
