"""Folium preview: aligned weight/CORINE/population rasters, optional CAMS grid, legend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from PROXY.visualization._legend import (
    LegendSection,
    build_weight_legend_section,
    categorical_swatch_html,
    colormap_swatch_html,
    region_note,
    render_unified_legend,
    weight_log_percentile_stats,
)
from PROXY.visualization._mapbuilder import (
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
from PROXY.visualization.corine_rgba import corine_clc_overlay_rgba
from PROXY.visualization.overlay_utils import (
    read_weight_corine_population_via_weight_grid_wgs84,
    scalar_to_rgba,
)

# CLC Level-3 Industrial / Commercial labels used in the A_PublicPower preview. "3" is
# the rolled-up super-class value present in some CLC-derived rasters, kept as
# "Industrial / commercial (super-class)" for clarity.
_CORINE_CATEGORICAL_LABELS: dict[int, str] = {
    3: "Industrial / commercial / transport (super-class)",
    111: "Continuous urban fabric",
    112: "Discontinuous urban fabric",
    121: "Industrial or commercial units",
    122: "Road / rail networks",
    124: "Airports",
    131: "Mineral extraction sites",
    132: "Dump sites",
    133: "Construction sites",
}


def _corine_legend_section(highlight_codes: tuple[int, ...]) -> LegendSection:
    """Build a categorical CLC swatch section for the highlighted codes."""
    from PROXY.visualization.corine_rgba import industrial_commercial_hex

    colour = industrial_commercial_hex()
    rows: list[tuple[str, str]] = []
    for code in highlight_codes:
        label = _CORINE_CATEGORICAL_LABELS.get(int(code), f"CLC {int(code)}")
        rows.append((f"CLC {int(code)} - {label}", colour))
    body = (
        "<p class=\"pl-hint\">Industrial / commercial CORINE pixels rendered "
        "above the weight layer so you can spot which cells CORINE marked as "
        "industrial-commercial.</p>"
        + categorical_swatch_html(rows)
    )
    return LegendSection(title="CORINE industrial / commercial classes", html=body)


def write_public_power_area_html(
    *,
    root: Path,
    weight_tif: Path,
    corine_tif: Path,
    population_tif: Path,
    out_html: Path,
    area_proxy: dict[str, Any] | None = None,
    pad_deg: float = 0.02,
    max_width: int = 1400,
    max_height: int = 1200,
    weight_opacity: float = 0.82,
    context_opacity: float = 0.9,
    cams_nc_path: Path | None = None,
    cams_country_iso3: str = "GRC",
    weight_display_mode: str = "global_log",
    region: str | None = None,
    override_bbox: tuple[float, float, float, float] | None = None,
) -> Path:
    """
    Folium map: basemaps, JRC population, CORINE (CLC 3 / 121 only), weights, optional CAMS grid.

    CORINE and population are resampled onto the weight GeoTIFF grid, then all layers share one
    WGS84 warp. Default weight scale is **global log10**; use ``weight_display_mode='per_cell'``
    for CAMS-cell min-max (requires CAMS NetCDF). The CORINE layer is now drawn **above** the
    weight layer at slightly reduced opacity so industrial / commercial pixels are always visible.
    """
    try:
        import folium  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Visualization requires folium, branca, rasterio. "
            "Install with: pip install folium branca rasterio matplotlib"
        ) from exc

    area_proxy = area_proxy or {}
    corine_band = int(area_proxy.get("corine_band", 1))
    codes_raw = area_proxy.get("corine_codes", [3, 121])
    industrial_codes_tuple = tuple(int(x) for x in codes_raw)

    wt = resolve_under_root(weight_tif, root)
    ct = resolve_under_root(corine_tif, root)
    pt = resolve_under_root(population_tif, root)

    if not wt.is_file():
        raise FileNotFoundError(f"Weight GeoTIFF not found: {wt}")
    if not ct.is_file():
        raise FileNotFoundError(f"CORINE GeoTIFF not found: {ct}")
    if not pt.is_file():
        raise FileNotFoundError(f"Population GeoTIFF not found: {pt}")

    view = compute_view_context(
        wt, pad_deg=pad_deg, max_width=max_width, max_height=max_height,
        region=region, override_bbox=override_bbox,
    )
    stacked = read_weight_corine_population_via_weight_grid_wgs84(
        wt,
        corine_path=ct,
        corine_band=corine_band,
        population_path=pt,
        population_band=1,
        west=view.west,
        south=view.south,
        east=view.east,
        north=view.north,
        display_width=view.gw,
        display_height=view.gh,
    )
    w_arr = stacked["weight_wgs84"]
    w_nodata = stacked["weight_nodata"]
    clc_raw = stacked["corine_wgs84"]
    clc_nd = stacked["corine_nodata"]
    pop_arr = stacked["population_wgs84"]
    pop_nd = stacked["population_nodata"]

    has_cams = bool(cams_nc_path) and Path(cams_nc_path).is_file()
    cams_nc_display = Path(cams_nc_path).name if has_cams and cams_nc_path else ""
    use_per_cell = has_cams and weight_display_mode.strip().lower() == "per_cell"

    grid_fc: dict[str, Any] = {"type": "FeatureCollection", "features": []}
    m_area = None
    ds_handle = None
    if has_cams:
        from PROXY.sectors.A_PublicPower.cams_area_mask import public_power_area_mask

        ds_handle = xr.open_dataset(cams_nc_path, engine="netcdf4")
        try:
            m_area = public_power_area_mask(ds_handle, cams_country_iso3)
            grid_fc = build_cams_area_grid_geojson_for_view(ds_handle, m_area, view)
        except Exception:
            ds_handle.close()
            raise

    clc_i = np.full((view.gh, view.gw), -1, dtype=np.int32)
    ok = np.isfinite(clc_raw)
    if clc_nd is not None:
        ok = ok & (clc_raw != float(clc_nd))
    clc_i[ok] = np.rint(clc_raw[ok]).astype(np.int32)
    rgba_cor = corine_clc_overlay_rgba(clc_i, highlight_codes=industrial_codes_tuple)

    rgba_pop = scalar_to_rgba(
        pop_arr,
        colour_mode="log",
        cmap_name="YlOrRd",
        hide_zero=True,
        nodata_val=float(pop_nd) if pop_nd is not None else None,
    )

    fmap = create_folium_map_with_tiles(view, zoom_start=8)

    add_raster_overlay(
        fmap,
        rgba_pop,
        view,
        name="JRC population 2021 (log, YlOrRd)",
        opacity=context_opacity,
        show=False,
    )

    preferred_band = pick_band_by_pollutant(wt, area_proxy, sector_cfg=area_proxy)
    preferred_band, _band_note = pick_first_positive_band(
        wt, preferred_band,
        empty_message="All weight bands are non-positive; showing the preferred band anyway.",
    )
    pol_label = pollutant_label_for_band(wt, preferred_band)
    _ppri, _pexc = visualization_pollutant_priority_from_cfg(area_proxy)
    mp = add_multipollutant_weight_layers(
        fmap,
        view,
        wt,
        preferred_band=preferred_band,
        preferred_label=pol_label,
        sector_key="A_PublicPower",
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
            sector_title="A_PublicPower",
        )
    if ds_handle is not None:
        ds_handle.close()
        ds_handle = None
    pol_label = mp.primary_label
    primary_w_arr = mp.bands_arrays[0] if mp.bands_arrays else w_arr

    add_raster_overlay(
        fmap,
        rgba_cor,
        view,
        name="CORINE industrial / commercial (above weights)",
        opacity=0.55,
        show=True,
    )

    add_cams_grid_overlay(
        fmap,
        grid_fc,
        name="CAMS GNFR A grid (cell outlines)",
        show=True,
        colour="#0d47a1",
        weight=2.5,
        outline_opacity=0.95,
    )

    pop_lo, pop_hi = weight_log_percentile_stats(pop_arr)
    sections: list[LegendSection] = [
        LegendSection(
            title="A_PublicPower - preview",
            html=(
                "<p class=\"pl-meta\">Weight raster: <code>"
                + wt.name
                + f"</code> (band {int(mp.primary_band)} - <b>{pol_label}</b>)</p>"
                "<p class=\"pl-hint\">JRC population, CORINE industrial/commercial and the final "
                "GNFR A area weights on one WGS84 display grid. CORINE is drawn above the weights "
                "so you can see which pixels contributed.</p>"
            ),
            open=True,
        ),
    ]
    sections.extend(mp.legend_sections)
    sections.append(
        build_weight_legend_section(
            primary_w_arr,
            display_mode=("per_cell" if use_per_cell else "global_log"),
            cmap="plasma",
            title=f"Weight scale (GNFR A - {pol_label})",
        )
    )
    sections.extend([
        _corine_legend_section(industrial_codes_tuple),
        LegendSection(
            title="Population (optional context)",
            html=colormap_swatch_html(
                cmap="YlOrRd", vmin=pop_lo, vmax=pop_hi,
                caption="log10(population), 2-98%",
                label_min=f"1e{pop_lo:.1f}", label_max=f"1e{pop_hi:.1f}",
            )
            + "<p class=\"pl-hint\">Off by default; toggle in the layer control.</p>",
            open=False,
        ),
    ])
    if has_cams:
        sections.append(
            LegendSection(
                title="CAMS grid",
                html=(
                    "<p class=\"pl-hint\">Blue rectangles = GNFR A area source cells for "
                    "<code>" + cams_country_iso3 + "</code>. Weights outside these cells are "
                    "clipped to transparent.</p>"
                    "<p class=\"pl-meta\">NetCDF: <code>" + cams_nc_display + "</code></p>"
                ),
                open=False,
            )
        )
    legend_html = render_unified_legend(
        "A_PublicPower",
        sections,
        pollutant_badge=pol_label,
        region_note=region_note(region, override_bbox),
    )
    return save_folium_map(fmap, out_html, root, legend_html=legend_html)
