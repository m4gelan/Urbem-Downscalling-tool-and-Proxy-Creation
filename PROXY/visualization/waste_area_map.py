"""Folium preview for J_Waste: proxy groups (solid / WW / residual), inputs, GNFR J weights, CAMS grid."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import xarray as xr

from PROXY.core.cams.gnfr import gnfr_code_to_index
from PROXY.core.cams.mask import cams_gnfr_country_source_mask
from PROXY.sectors.J_Waste.pipeline import merge_waste_pipeline_cfg
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
    resolve_under_root,
    save_folium_map,
)
from PROXY.visualization._dominance import (
    compute_dominance_rgba,
    dominance_legend_section,
)
from PROXY.visualization._color_debug import viz_color_debug_enabled, viz_color_log
from PROXY.visualization._family import (
    assign_auto_colors,
    family_argmax_rgba,
    family_legend_section,
)
from PROXY.visualization._click_popup import enrich_cams_grid_with_popups
from PROXY.visualization._multipollutant import (
    add_multipollutant_weight_layers,
    visualization_pollutant_priority_from_cfg,
)
from PROXY.visualization.waste_context import build_waste_proxy_rgba_overlays

_WASTE_FAMILY_PREFIXES: dict[str, str] = {
    "Solid waste": "J_Waste \u00b7 solid: ",
    "Wastewater": "J_Waste \u00b7 WW stack: ",
    "Residual": "J_Waste \u00b7 residual: ",
}
_WASTE_FAMILY_COLORS: dict[str, str] = {
    "Solid waste": "#d97706",
    "Wastewater": "#1d4ed8",
    "Residual": "#166534",
}

_WASTE_DATASET_COLOR_OVERRIDES: dict[str, str] = {
    # --- Wastewater family (deep blue -> purple; easy to tell apart) ---
    "J_Waste \u00b7 WW stack: uwwtd_agglomerations": "#1e3a8a",
    "J_Waste \u00b7 WW stack: uwwtd_treatment_plants": "#7c3aed",
    "J_Waste \u00b7 WW stack: population": "#38bdf8",
    "J_Waste \u00b7 WW stack: imperviousness": "#0891b2",
    "J_Waste \u00b7 WW stack: industrial_clc_mask": "#6b7280",
    "J_Waste \u00b7 WW stack: imperv_valid_mask": "#94a3b8",
    # --- Solid-waste family (warm browns / oranges / reds for OSM point buffers) ---
    "J_Waste \u00b7 solid: corine_clc_132": "#b45309",
    "J_Waste \u00b7 solid: corine_clc_121": "#ea580c",
    "J_Waste \u00b7 solid: osm_landfill": "#d97706",
    "J_Waste \u00b7 solid: osm_amenity_recycling": "#f59e0b",
    "J_Waste \u00b7 solid: osm_amenity_waste_disposal": "#b91c1c",
    "J_Waste \u00b7 solid: osm_wastewater_plant": "#dc2626",
    # --- Residual family (green -> brown; visually disjoint from solid + wastewater) ---
    "J_Waste \u00b7 residual: residual_pop": "#166534",
    "J_Waste \u00b7 residual: residual_ghsl_rural_mask": "#4d7c0f",
    "J_Waste \u00b7 residual: residual_imperv_01": "#a16207",
}


def _short_waste(title: str) -> str:
    return title.replace("J_Waste \u00b7 ", "", 1)


def write_waste_area_html(
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
    _ = (corine_tif, population_tif)

    try:
        import folium  # noqa: F401
    except ImportError as exc:
        raise SystemExit(VIZ_DEPS_MESSAGE) from exc

    viz_cfg = sector_cfg.get("visualization") or {}
    cams_block = sector_cfg.get("cams") or {}
    gnfr = str(cams_block.get("gnfr", "J"))
    bbox_cfg = cams_block.get("domain_bbox_wgs84")
    domain_bbox = tuple(float(x) for x in bbox_cfg) if bbox_cfg else None
    source_types = tuple(cams_block.get("source_types") or ("area",))

    wt = resolve_under_root(weight_tif, root)
    if not wt.is_file():
        raise FileNotFoundError(f"Weight GeoTIFF not found: {wt}")

    weight_band = pick_band_by_pollutant(
        wt, viz_cfg, strip_prefixes=("j_waste_weight_",), sector_cfg=sector_cfg
    )
    weight_band, band_note = pick_first_positive_band(
        wt,
        weight_band,
        empty_message="All weight bands are non-positive. Re-run J_Waste build or check inputs.",
    )
    pol_label = pollutant_label_for_band(wt, weight_band)
    weight_opacity = float(viz_cfg.get("weight_opacity", weight_opacity))

    view = compute_view_context(
        wt, pad_deg=pad_deg, max_width=max_width, max_height=max_height,
        region=region, override_bbox=override_bbox,
    )

    has_cams = bool(cams_nc_path) and Path(cams_nc_path).is_file()
    cams_nc_display = Path(cams_nc_path).name if has_cams and cams_nc_path else ""
    eff_mode = str(
        (viz_cfg.get("weight_display") or weight_display_mode) or "global_log"
    ).strip().lower()
    use_per_cell = has_cams and eff_mode == "per_cell"

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
        waste_merged = merge_waste_pipeline_cfg(
            root,
            path_cfg,
            sector_cfg,
            country=country,
            output_dir=wt.parent.resolve(),
        )
    except (KeyError, OSError, TypeError, ValueError):
        waste_merged = None
    scalars_j: dict[str, Any] = {}
    if waste_merged is not None:
        for title, _cmap_name, rgba in build_waste_proxy_rgba_overlays(
            root,
            waste_merged,
            wt,
            view.west,
            view.south,
            view.east,
            view.north,
            view.dst_t,
            (view.gh, view.gw),
            path_cfg,
            resampling="bilinear",
            scalars_out=scalars_j,
        ):
            add_raster_overlay(
                fmap, rgba, view, name=_short_waste(title), opacity=context_opacity, show=False
            )

    family_sections: list = []
    family_scores: dict[str, Any] = {}
    for family, prefix in _WASTE_FAMILY_PREFIXES.items():
        keys = [k for k in scalars_j if k.startswith(prefix)]
        if not keys:
            continue
        color_map = assign_auto_colors(keys, overrides=_WASTE_DATASET_COLOR_OVERRIDES)
        datasets = {k: scalars_j[k] for k in keys}
        if viz_color_debug_enabled():
            viz_color_log(
                "waste_family_config_alignment",
                family=family,
                prefix=prefix,
                dataset_keys_sorted=list(keys),
                color_map_hex=dict(color_map),
            )
        res = family_argmax_rgba(datasets, color_map)
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
                key_labels={k: _short_waste(k) for k in used},
                open_=False,
            )
        )

    cross_dom_used: list[str] = []
    cross_rgba: Any = None
    if len(family_scores) >= 2:
        if viz_color_debug_enabled():
            viz_color_log(
                "waste_cross_family_dominance_inputs",
                family_scores_keys=list(family_scores.keys()),
                palette_hex=dict(_WASTE_FAMILY_COLORS),
            )
        cross = compute_dominance_rgba(family_scores, colors=_WASTE_FAMILY_COLORS)
        if cross is not None:
            cross_rgba, cross_dom_used = cross

    _ppri, _pexc = visualization_pollutant_priority_from_cfg(viz_cfg)
    mp = add_multipollutant_weight_layers(
        fmap,
        view,
        wt,
        preferred_band=weight_band,
        preferred_label=pol_label,
        sector_key="J_Waste",
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
    if cross_rgba is not None:
        add_raster_overlay(
            fmap,
            cross_rgba,
            view,
            name="Dominant family (solid / wastewater / residual)",
            opacity=1.0,
            show=True,
        )
    if has_cams and ds_handle is not None:
        enrich_cams_grid_with_popups(
            grid_fc,
            view=view,
            m_area=m_area,
            ds=ds_handle,
            sector_title="J_Waste",
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

    legend_wmode = "per_cell" if use_per_cell else "global_log"
    sections: list[LegendSection] = [
        LegendSection(
            title="J_Waste - preview",
            html=(
                "<p class=\"pl-meta\">Weight raster: <code>"
                + wt.name
                + f"</code> (band {int(weight_band)} - <b>{pol_label}</b>)</p>"
                "<p class=\"pl-hint\">GNFR J / 13 waste weights = solid-waste / wastewater / "
                "residual stacks. Per-group input layers (sites, population, POIs) are in the "
                "layer control; all off by default to keep the map legible.</p>"
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
                primary_w_arr, display_mode=legend_wmode, cmap="plasma",
                title=f"Weight scale (GNFR {gnfr} - {pol_label})",
            )
        )
    sections.extend(family_sections)
    if cross_dom_used:
        dom_legend_gids = [g for g in _WASTE_FAMILY_COLORS if g in family_scores]
        sections.append(
            dominance_legend_section(
                dom_legend_gids,
                colors=_WASTE_FAMILY_COLORS,
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
        "J_Waste",
        sections,
        pollutant_badge=pol_label,
        region_note=region_note(region, override_bbox),
    )
    return save_folium_map(fmap, out_html, root, legend_html=legend_html)


write_j_waste_area_html = write_waste_area_html
