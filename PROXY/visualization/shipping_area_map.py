"""Folium preview for G_Shipping: EMODnet, CORINE port, OSM, GNFR G weights, optional CAMS grid."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import xarray as xr

from PROXY.core.cams.gnfr import gnfr_code_to_index
from PROXY.core.cams.mask import cams_gnfr_country_source_mask
from PROXY.sectors.G_Shipping.pipeline import merge_shipping_pipeline_cfg
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
from PROXY.visualization._click_popup import enrich_cams_grid_with_popups
from PROXY.visualization._multipollutant import (
    add_multipollutant_weight_layers,
    visualization_pollutant_priority_from_cfg,
)
from PROXY.visualization.shipping_context import build_shipping_proxy_rgba_overlays


def write_shipping_area_html(
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
    gnfr = str(cams_block.get("gnfr", "G"))
    bbox_cfg = cams_block.get("domain_bbox_wgs84")
    domain_bbox = tuple(float(x) for x in bbox_cfg) if bbox_cfg else None
    source_types = tuple(cams_block.get("source_types") or ("area",))

    wt = resolve_under_root(weight_tif, root)
    if not wt.is_file():
        raise FileNotFoundError(f"Weight GeoTIFF not found: {wt}")

    weight_band = pick_band_by_pollutant(wt, viz_cfg, sector_cfg=sector_cfg)
    weight_band, _ship_band_note = pick_first_positive_band(
        wt,
        weight_band,
        empty_message="All weight bands are non-positive; showing the preferred band anyway.",
    )
    weight_opacity = float(viz_cfg.get("weight_opacity", weight_opacity))
    pol_label = pollutant_label_for_band(wt, weight_band)
    if pol_label == f"band {weight_band}":
        pol_label = "g_shipping_weight"

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
        ship_merged = merge_shipping_pipeline_cfg(
            root,
            path_cfg,
            sector_cfg,
            country=country,
            output_dir=wt.parent.resolve(),
        )
    except (KeyError, OSError, TypeError, ValueError):
        ship_merged = None
    if ship_merged is not None:
        for title, _cmap_name, rgba in build_shipping_proxy_rgba_overlays(
            root,
            ship_merged,
            wt,
            view.west,
            view.south,
            view.east,
            view.north,
            view.dst_t,
            (view.gh, view.gw),
            path_cfg,
            resampling="bilinear",
        ):
            show = "combined proxy" in title.lower()
            short = title
            if short.startswith("G_Shipping · "):
                short = short.replace("G_Shipping · ", "", 1)
            add_raster_overlay(
                fmap, rgba, view, name=short, opacity=context_opacity, show=show
            )

    _ppri, _pexc = visualization_pollutant_priority_from_cfg(viz_cfg)
    mp = add_multipollutant_weight_layers(
        fmap,
        view,
        wt,
        preferred_band=weight_band,
        preferred_label=pol_label,
        sector_key="G_Shipping",
        display_mode="percentile",
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
            sector_title="G_Shipping",
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
            title="G_Shipping - preview",
            html=(
                "<p class=\"pl-meta\">Weight raster: <code>"
                + wt.name
                + f"</code> (band {int(weight_band)} - <b>{pol_label}</b>)</p>"
                "<p class=\"pl-hint\">EMODnet shipping density + OSM port context + CORINE port "
                "polygons, combined as GNFR G area weights. Best read on a satellite basemap.</p>"
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
    if has_cams:
        sections.append(
            LegendSection(
                title="CAMS grid",
                html=(
                    "<p class=\"pl-hint\">Cell outlines: CAMS-REG GNFR <b>"
                    + gnfr
                    + "</b> area sources (<code>country_id="
                    + cams_country_iso3
                    + "</code>, <code>emission_category_index="
                    + str(gnfr_code_to_index(gnfr))
                    + "</code>).</p>"
                    f"<p class=\"pl-meta\">NetCDF: <code>{cams_nc_display}</code></p>"
                ),
                open=False,
            )
        )

    legend_html = render_unified_legend(
        "G_Shipping",
        sections,
        pollutant_badge=pol_label,
        region_note=region_note(region, override_bbox),
    )
    return save_folium_map(fmap, out_html, root, legend_html=legend_html)


write_g_shipping_area_html = write_shipping_area_html
