"""Folium preview for E_Solvents: CORINE + OSM proxy groups, GNFR E weights, CAMS grid."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import xarray as xr

from PROXY.core.cams.gnfr import gnfr_code_to_index
from PROXY.core.cams.mask import cams_gnfr_country_source_mask
from PROXY.sectors.E_Solvents.pipeline import merge_solvents_pipeline_cfg
from PROXY.visualization._legend import (
    LegendSection,
    build_weight_legend_section,
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
    pick_first_positive_band,
    pollutant_label_for_band,
    shorten_w_gnfr_e_weight_band_label,
    resolve_under_root,
    save_folium_map,
)
from PROXY.visualization._dominance import (
    compute_dominance_rgba,
    dominance_legend_section,
)
from PROXY.visualization._color_debug import viz_color_debug_enabled, viz_color_log
from PROXY.visualization._family import family_argmax_rgba, family_legend_section
from PROXY.visualization._click_popup import enrich_cams_grid_with_popups
from PROXY.visualization._multipollutant import (
    add_multipollutant_weight_layers,
    visualization_pollutant_priority_from_cfg,
)
from PROXY.visualization.solvents_context import build_solvents_proxy_rgba_overlays

_SOLVENTS_FAMILIES: dict[str, dict[str, str]] = {
    "Residential": {
        "E_Solvents \u00b7 CLC: urban_fabric": "#16a34a",
        "E_Solvents \u00b7 CLC: residential_share": "#166534",
        "E_Solvents \u00b7 population (ref window)": "#86efac",
    },
    "Service": {
        "E_Solvents \u00b7 CLC: service_land": "#1d4ed8",
        "E_Solvents \u00b7 OSM: service_osm": "#38bdf8",
        "E_Solvents \u00b7 OSM: road_length": "#7c3aed",
        "E_Solvents \u00b7 OSM: weighted_road_length": "#a855f7",
        "E_Solvents \u00b7 OSM: transport_area": "#0891b2",
    },
    "Industrial": {
        "E_Solvents \u00b7 CLC: industrial_clc": "#b91c1c",
        "E_Solvents \u00b7 OSM: industry_osm": "#f97316",
        "E_Solvents \u00b7 OSM: industry_buildings": "#ea580c",
        "E_Solvents \u00b7 OSM: roof_area": "#eab308",
    },
}
_SOLVENTS_FAMILY_COLORS: dict[str, str] = {
    "Residential": "#16a34a",
    "Service": "#1d4ed8",
    "Industrial": "#b91c1c",
}


def _short(title: str) -> str:
    return title.replace("E_Solvents \u00b7 ", "", 1)


def write_solvents_area_html(
    *,
    root: Path,
    weight_tif: Path,
    corine_tif: Path,
    population_tif: Path | None,
    out_html: Path,
    sector_cfg: dict[str, Any],
    path_cfg: dict[str, Any],
    country: str = "EL",
    pad_deg: float = 0.02,
    max_width: int = 1400,
    max_height: int = 1200,
    weight_opacity: float = 0.92,
    cams_nc_path: Path | None = None,
    cams_country_iso3: str = "GRC",
    weight_display_mode: str = "global_log",
    region: str | None = None,
    override_bbox: tuple[float, float, float, float] | None = None,
) -> Path:
    """
    Folium map: basemaps, population / CLC code groups / OSM raw channels, GNFR E weight band, CAMS grid.
    """
    _ = (corine_tif, population_tif)

    try:
        import folium  # noqa: F401
    except ImportError as exc:
        raise SystemExit(VIZ_DEPS_MESSAGE) from exc

    viz_cfg = sector_cfg.get("visualization") or {}
    cams_block = sector_cfg.get("cams") or {}
    gnfr = str(cams_block.get("gnfr", "E"))
    bbox_cfg = cams_block.get("domain_bbox_wgs84")
    domain_bbox = tuple(float(x) for x in bbox_cfg) if bbox_cfg else None
    source_types = tuple(cams_block.get("source_types") or ("area",))

    wt = resolve_under_root(weight_tif, root)
    if not wt.is_file():
        raise FileNotFoundError(f"Weight GeoTIFF not found: {wt}")

    weight_band = pick_band_by_pollutant(wt, viz_cfg, sector_cfg=sector_cfg)
    weight_band, band_note = pick_first_positive_band(
        wt,
        weight_band,
        empty_message=(
            "All weight bands are non-positive in this file. Re-run the E_Solvents build "
            "or check CEIP. Proxy layers are independent of this raster."
        ),
    )
    pol_label = pollutant_label_for_band(wt, weight_band)
    pol_label = shorten_w_gnfr_e_weight_band_label(pol_label) or pol_label
    weight_opacity = float(viz_cfg.get("weight_opacity", weight_opacity))

    view = compute_view_context(
        wt, pad_deg=pad_deg, max_width=max_width, max_height=max_height,
        region=region, override_bbox=override_bbox,
    )

    has_cams = bool(cams_nc_path) and Path(cams_nc_path).is_file()
    cams_nc_display = Path(cams_nc_path).name if has_cams and cams_nc_path else ""
    _ = weight_display_mode

    grid_fc: dict[str, Any] = {"type": "FeatureCollection", "features": []}
    m_area = None
    ds_handle = None
    if has_cams:
        ds_handle = xr.open_dataset(cams_nc_path, engine="netcdf4")
        try:
            m_area = cams_gnfr_country_source_mask(
                ds_handle,
                cams_country_iso3,
                gnfr=gnfr,
                source_types=source_types,
                domain_bbox_wgs84=domain_bbox,
            )
            grid_fc = build_cams_area_grid_geojson_for_view(ds_handle, m_area, view)
        except Exception:
            ds_handle.close()
            raise

    fmap = create_folium_map_with_tiles(view, zoom_start=8)

    context_opacity = float(viz_cfg.get("context_layer_opacity", 0.82))
    try:
        sol_merged = merge_solvents_pipeline_cfg(
            root,
            path_cfg,
            sector_cfg,
            country=country,
            output_path=wt.resolve() if wt.is_file() else (root / wt).resolve(),
        )
    except (KeyError, OSError, TypeError, ValueError):
        sol_merged = None
    scalars_e: dict[str, Any] = {}
    if sol_merged is not None:
        for title, _cmap_name, rgba in build_solvents_proxy_rgba_overlays(
            root,
            sol_merged,
            wt,
            view.west,
            view.south,
            view.east,
            view.north,
            view.dst_t,
            (view.gh, view.gw),
            path_cfg,
            resampling="bilinear",
            scalars_out=scalars_e,
        ):
            add_raster_overlay(
                fmap, rgba, view, name=_short(title), opacity=context_opacity, show=False
            )

    family_sections: list = []
    family_scores: dict[str, Any] = {}
    for family, color_map in _SOLVENTS_FAMILIES.items():
        family_scalars = {
            title: scalars_e[title] for title in color_map.keys() if title in scalars_e
        }
        if viz_color_debug_enabled():
            viz_color_log(
                "solvents_family_config_alignment",
                family=family,
                color_map_keys_order=list(color_map.keys()),
                scalars_available=list(scalars_e.keys()),
                scalars_used_for_family=list(family_scalars.keys()),
                color_keys_missing_scalars=[k for k in color_map if k not in scalars_e],
            )
        if not family_scalars:
            continue
        res = family_argmax_rgba(family_scalars, color_map)
        if res is None:
            continue
        rgba_fam, used, score = res
        add_raster_overlay(
            fmap, rgba_fam, view,
            name=f"{family} datasets (combined, color = dominant)",
            opacity=1.0, show=False,
        )
        family_scores[family] = score
        family_sections.append(
            family_legend_section(
                family, used, color_map,
                key_labels={k: _short(k) for k in used},
                open_=False,
            )
        )

    cross_dom_used: list[str] = []
    cross_rgba: Any = None
    if len(family_scores) >= 2:
        if viz_color_debug_enabled():
            viz_color_log(
                "solvents_cross_family_dominance_inputs",
                family_scores_keys=list(family_scores.keys()),
                palette_keys=list(_SOLVENTS_FAMILY_COLORS.keys()),
                palette_hex=dict(_SOLVENTS_FAMILY_COLORS),
            )
        cross = compute_dominance_rgba(family_scores, colors=_SOLVENTS_FAMILY_COLORS)
        if cross is not None:
            cross_rgba, cross_dom_used = cross

    _ppri, _pexc = visualization_pollutant_priority_from_cfg(viz_cfg)
    mp = add_multipollutant_weight_layers(
        fmap,
        view,
        wt,
        preferred_band=weight_band,
        preferred_label=pol_label,
        sector_key="E_Solvents",
        display_mode="percentile",
        cmap="plasma",
        weight_opacity=weight_opacity,
        cams_nc_path=cams_nc_path if has_cams else None,
        m_area=m_area,
        cams_ds=ds_handle,
        clip_alpha_to_cams=has_cams,
        pollutant_priority=_ppri,
        exclusive_pollutant_panel=_pexc,
        shorten_w_gnfr_e_weight_labels=True,
    )
    if cross_rgba is not None:
        add_raster_overlay(
            fmap,
            cross_rgba,
            view,
            name="Dominant family (residential / service / industrial)",
            opacity=1.0,
            show=True,
        )
    if has_cams and ds_handle is not None:
        enrich_cams_grid_with_popups(
            grid_fc,
            view=view,
            m_area=m_area,
            ds=ds_handle,
            sector_title="E_Solvents",
            dominance_layers=family_scores if family_scores else None,
            dominant_heading="Dominant family",
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
        name=f"CAMS GNFR {gnfr} area source grid (cell outlines)",
        show=False,
    )

    sections: list[LegendSection] = [
        LegendSection(
            title="E_Solvents - preview",
            html=(
                "<p class=\"pl-meta\">Weight raster: <code>"
                + wt.name
                + f"</code> (band {int(weight_band)} - <b>{pol_label}</b>)</p>"
                "<p class=\"pl-hint\">GNFR E / solvents area weights with CORINE "
                "(residential, service, industrial) and OSM-derived (industry buildings, "
                "roads, transport) proxies. OSM road rasters are dense: use <code>--region "
                "attica</code> to see individual streets.</p>"
                + (
                    f"<p class=\"pl-note\">Note: {band_note}</p>" if band_note else ""
                )
            ),
            open=True,
        ),
    ]
    sections.extend(mp.legend_sections)
    if primary_w_arr is not None:
        sections.append(
            build_weight_legend_section(
                primary_w_arr, display_mode="percentile", cmap="plasma",
                title=f"Weight scale (GNFR {gnfr} - {pol_label})",
            )
        )
    sections.extend(family_sections)
    if cross_dom_used:
        dom_legend_gids = [g for g in _SOLVENTS_FAMILY_COLORS if g in cross_dom_used]
        sections.append(
            dominance_legend_section(
                dom_legend_gids,
                colors=_SOLVENTS_FAMILY_COLORS,
                sector_label="family",
                gid_labels={g: f"{g} dominant" for g in dom_legend_gids},
            )
        )
    if has_cams:
        sections.append(
            LegendSection(
                title="CAMS grid",
                html=(
                    "<p class=\"pl-hint\">Cell outlines follow CAMS-REG GNFR "
                    f"<b>{gnfr}</b> area sources for <code>country_id={cams_country_iso3}</code>, "
                    f"<code>emission_category_index={gnfr_code_to_index(gnfr)}</code>.</p>"
                    f"<p class=\"pl-meta\">NetCDF: <code>{cams_nc_display}</code></p>"
                ),
                open=False,
            )
        )

    legend_html = render_unified_legend(
        "E_Solvents",
        sections,
        pollutant_badge=pol_label,
        region_note=region_note(region, override_bbox),
    )
    return save_folium_map(fmap, out_html, root, legend_html=legend_html)
