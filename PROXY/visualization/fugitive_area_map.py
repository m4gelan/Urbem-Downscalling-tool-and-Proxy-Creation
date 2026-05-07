"""Folium preview for D_Fugitive: GNFR D area weights and CAMS grid (same display contract as other PROXY area maps)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import xarray as xr

from PROXY.core.cams.gnfr import gnfr_code_to_index
from PROXY.core.cams.mask import cams_gnfr_country_source_mask
from PROXY.sectors.D_Fugitive import builder as fugitive_builder
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
from PROXY.visualization._click_popup import enrich_cams_grid_with_popups
from PROXY.visualization._multipollutant import (
    add_multipollutant_weight_layers,
    visualization_pollutant_priority_from_cfg,
)
from PROXY.visualization.fugitive_context import (
    build_fugitive_proxy_rgba_overlays,
    is_fugitive_gem_auxiliary_layer,
)


def _log_fugitive_weight_viz_debug(weight_path: Path, mp: Any) -> None:
    """Stderr: GeoTIFF band tags vs multipollutant panel (catch CO/NMVOC band mix-ups)."""
    try:
        import rasterio

        with rasterio.open(weight_path) as src:
            desc = list(src.descriptions) if src.descriptions else []
            n = int(src.count)
        panel = list(getattr(mp, "bands", []) or [])
        print(
            f"[fugitive-viz-debug] weight_tif={weight_path.resolve()} "
            f"n_bands={n} descriptions={desc!r} "
            f"mp.primary_band={getattr(mp, 'primary_band', None)} "
            f"mp.primary_label={getattr(mp, 'primary_label', None)!r} "
            f"multipollutant_panel={panel}",
            file=sys.stderr,
        )
    except Exception as exc:
        print(f"[fugitive-viz-debug] weight inspect failed: {exc}", file=sys.stderr)


def write_fugitive_area_html(
    *,
    root: Path,
    weight_tif: Path,
    corine_tif: Path,
    population_tif: Path | None,
    out_html: Path,
    sector_cfg: dict[str, Any],
    path_cfg: dict[str, Any],
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
    Folium map: basemaps, optional population + per-group proxy inputs (Viridis, same as legacy maps),
    GNFR D weight band (plasma), and optional CAMS grid. ``corine_tif`` / ``population_tif`` match the
    API of other sector map writers; data paths for context layers are taken from ``path_cfg`` and
    the merged fugitive pipeline config.
    """
    _ = (corine_tif, population_tif)

    try:
        import folium  # noqa: F401
    except ImportError as exc:
        raise SystemExit(VIZ_DEPS_MESSAGE) from exc

    viz_cfg = sector_cfg.get("visualization") or {}
    cams_block = sector_cfg.get("cams") or {}
    gnfr = str(cams_block.get("gnfr", "D"))
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
            "All weight bands are non-positive in this file. Re-run the D_Fugitive build "
            "or check CEIP/alphas. Proxy layers are independent of this raster."
        ),
    )
    pol_label = pollutant_label_for_band(wt, weight_band)
    weight_opacity = float(viz_cfg.get("weight_opacity", weight_opacity))

    view = compute_view_context(
        wt, pad_deg=pad_deg, max_width=max_width, max_height=max_height,
        region=region, override_bbox=override_bbox,
    )

    has_cams = bool(cams_nc_path) and Path(cams_nc_path).is_file()
    cams_nc_display = Path(cams_nc_path).name if has_cams and cams_nc_path else ""
    _ = weight_display_mode  # fugitive TIFs are pre-CAMS-cell normalized; caller override unused

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
        fug_merged = fugitive_builder._merge_fugitive_pipeline_cfg(
            root,
            path_cfg,
            sector_cfg,
            country="EL",
            output_path=wt.resolve() if wt.is_file() else (root / wt).resolve(),
        )
    except (KeyError, OSError, TypeError):
        fug_merged = None
    group_pg: dict[str, Any] = {}
    if fug_merged is not None:
        for title, rgba in build_fugitive_proxy_rgba_overlays(
            root,
            fug_merged,
            wt,
            view.west,
            view.south,
            view.east,
            view.north,
            view.dst_t,
            (view.gh, view.gw),
            path_cfg,
            resampling="bilinear",
            group_pg_out=group_pg,
            visualization_cfg=viz_cfg,
        ):
            show = title.startswith("Fugitive · population") or is_fugitive_gem_auxiliary_layer(title)
            add_raster_overlay(
                fmap,
                rgba,
                view,
                name=title.replace("Fugitive · ", "", 1),
                opacity=context_opacity,
                show=show,
            )

    dominance_used: list[str] = []
    dom_rgba: Any = None
    if group_pg:
        dom_res = compute_dominance_rgba(group_pg)
        if dom_res is not None:
            dom_rgba, dominance_used = dom_res

    _ppri, _pexc = visualization_pollutant_priority_from_cfg(viz_cfg)
    wnames = viz_cfg.get("weight_band_layer_names")
    weight_overlay_names = [str(x) for x in wnames] if isinstance(wnames, list) else None
    mp = add_multipollutant_weight_layers(
        fmap,
        view,
        wt,
        preferred_band=weight_band,
        preferred_label=pol_label,
        sector_key="D_Fugitive",
        display_mode="percentile",
        cmap="plasma",
        weight_opacity=weight_opacity,
        cams_nc_path=cams_nc_path if has_cams else None,
        m_area=m_area,
        cams_ds=ds_handle,
        clip_alpha_to_cams=has_cams,
        pollutant_priority=_ppri,
        exclusive_pollutant_panel=_pexc,
        weight_overlay_names=weight_overlay_names,
    )
    if dom_rgba is not None:
        add_raster_overlay(
            fmap,
            dom_rgba,
            view,
            name="Dominant CEIP group (G1..G4 argmax of P_g)",
            opacity=1.0,
            show=True,
        )
    if has_cams and ds_handle is not None:
        enrich_cams_grid_with_popups(
            grid_fc,
            view=view,
            m_area=m_area,
            ds=ds_handle,
            sector_title="D_Fugitive",
            dominance_layers=group_pg if group_pg else None,
            dominant_heading="Dominant CEIP group",
        )
    if ds_handle is not None:
        ds_handle.close()
        ds_handle = None
    pol_label = mp.primary_label
    weight_band = mp.primary_band
    primary_w_arr = mp.bands_arrays[0] if mp.bands_arrays else None

    if os.environ.get("PROXY_DEBUG_FUGITIVE_VIZ", "").strip().lower() in ("1", "true", "yes") or viz_cfg.get(
        "debug_viz"
    ):
        _log_fugitive_weight_viz_debug(wt, mp)

    add_cams_grid_overlay(
        fmap,
        grid_fc,
        name=f"CAMS GNFR {gnfr} area source grid (cell outlines)",
        show=False,
    )

    sections: list[LegendSection] = [
        LegendSection(
            title="D_Fugitive - preview",
            html=(
                "<p class=\"pl-meta\">Weight raster: <code>"
                + wt.name
                + f"</code> (band {int(weight_band)} - <b>{pol_label}</b>)</p>"
                "<p class=\"pl-hint\">GNFR D fugitive area weights, already normalized per "
                "CAMS cell in the build. Context layers follow each mixture ingredient (CLC / OSM / "
                "GEM / VIIRS / POP) plus one combined proxy per group (<code>G*_layer</code>). "
                "Toggle layers in the control (zeros are transparent).</p>"
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
    if dominance_used and group_pg:
        legend_gids = list(group_pg.keys())
        sections.append(
            dominance_legend_section(
                legend_gids,
                sector_label="CEIP group (P_g)",
                gid_labels={g: str(g) for g in legend_gids},
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
        "D_Fugitive",
        sections,
        pollutant_badge=pol_label,
        region_note=region_note(region, override_bbox),
    )
    return save_folium_map(fmap, out_html, root, legend_html=legend_html)
