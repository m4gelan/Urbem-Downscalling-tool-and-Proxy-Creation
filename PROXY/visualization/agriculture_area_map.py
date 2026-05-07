"""Folium preview for K_Agriculture: NUTS2xCLC area weights (GNFR K+L), optional CAMS grid."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from PROXY.core.cams.mask import cams_gnfr_country_source_mask
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
from PROXY.visualization.overlay_utils import read_corine_clc_wgs84_on_weight_grid

# CLC agricultural classes used by K_Agriculture (see sectors/K_Agriculture/class_mapping.*).
_AGRI_CLC_SPECS: tuple[tuple[int, str, str], ...] = (
    (211, "Non-irrigated arable land", "#fdd835"),
    (212, "Permanently irrigated land", "#f57f17"),
    (213, "Rice fields", "#33691e"),
    (221, "Vineyards", "#880e4f"),
    (222, "Fruit trees / berry", "#6a1b9a"),
    (223, "Olive groves", "#558b2f"),
    (231, "Pastures", "#aed581"),
    (241, "Annual + permanent crops", "#ffca28"),
    (242, "Complex cultivation patterns", "#ef6c00"),
    (243, "Land principally occupied by agriculture", "#8d6e63"),
    (244, "Agro-forestry areas", "#4e342e"),
)


def _cams_kl_combined_mask(
    ds: xr.Dataset,
    iso3: str,
    *,
    domain_bbox_wgs84: tuple[float, float, float, float] | None,
    source_types: tuple[str, ...],
) -> np.ndarray:
    """True for CAMS area sources in GNFR K or L (emission 14/15) for the country domain."""
    m_k = cams_gnfr_country_source_mask(
        ds,
        iso3,
        gnfr="K",
        source_types=source_types,
        domain_bbox_wgs84=domain_bbox_wgs84,
    )
    m_l = cams_gnfr_country_source_mask(
        ds,
        iso3,
        gnfr="L",
        source_types=source_types,
        domain_bbox_wgs84=domain_bbox_wgs84,
    )
    return m_k | m_l


def _agri_corine_rgba(clc_i: np.ndarray) -> np.ndarray:
    """Categorical RGBA for the main CORINE agricultural classes listed in :data:`_AGRI_CLC_SPECS`."""
    import matplotlib.colors as mcolors

    h, w = clc_i.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for code, _label, hexcol in _AGRI_CLC_SPECS:
        r, g, b = mcolors.to_rgb(hexcol)
        mask = clc_i == int(code)
        if np.any(mask):
            rgba[mask, 0] = int(r * 255)
            rgba[mask, 1] = int(g * 255)
            rgba[mask, 2] = int(b * 255)
            rgba[mask, 3] = 215
    return rgba


def write_k_agriculture_area_html(
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
    _ = (population_tif, country)

    try:
        import folium  # noqa: F401
    except ImportError as exc:
        raise SystemExit(VIZ_DEPS_MESSAGE) from exc

    viz_cfg = sector_cfg.get("visualization") or {}
    cams_block = sector_cfg.get("cams") or {}
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
        empty_message="All weight bands are non-positive. Re-run K_Agriculture build or check inputs.",
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
        (viz_cfg.get("weight_display") or weight_display_mode) or "per_cell"
    ).strip().lower()
    use_per_cell = has_cams and eff_mode == "per_cell"

    grid_fc: dict[str, Any] = {"type": "FeatureCollection", "features": []}
    m_area = None
    ds_handle = None
    if has_cams:
        ds_handle = xr.open_dataset(cams_nc_path, engine="netcdf4")
        try:
            m_area = _cams_kl_combined_mask(
                ds_handle, cams_country_iso3, domain_bbox_wgs84=domain_bbox, source_types=source_types
            )
            grid_fc = build_cams_area_grid_geojson_for_view(ds_handle, m_area, view)
        except Exception:
            ds_handle.close()
            raise

    corine_tif = resolve_under_root(corine_tif, root)
    clc_i: np.ndarray | None = None
    if corine_tif.is_file():
        try:
            stk_c = read_corine_clc_wgs84_on_weight_grid(
                wt,
                corine_tif,
                corine_band=int((sector_cfg.get("corine") or {}).get("band", 1)),
                west=view.west,
                south=view.south,
                east=view.east,
                north=view.north,
                display_width=view.gw,
                display_height=view.gh,
            )
            clc_raw = stk_c["corine_wgs84"]
            clc_nd = stk_c["corine_nodata"]
            ok = np.isfinite(clc_raw)
            if clc_nd is not None:
                ok = ok & (clc_raw != float(clc_nd))
            clc_i = np.full((view.gh, view.gw), -1, dtype=np.int32)
            clc_i[ok] = np.rint(clc_raw[ok]).astype(np.int32)
        except Exception:
            clc_i = None

    fmap = create_folium_map_with_tiles(view, zoom_start=8)

    if clc_i is not None:
        rgba_agri = _agri_corine_rgba(clc_i)
        add_raster_overlay(
            fmap,
            rgba_agri,
            view,
            name="CORINE agricultural classes (211-244)",
            opacity=0.65,
            show=True,
        )

    _ppri, _pexc = visualization_pollutant_priority_from_cfg(viz_cfg)
    mp = add_multipollutant_weight_layers(
        fmap,
        view,
        wt,
        preferred_band=weight_band,
        preferred_label=pol_label,
        sector_key="K_Agriculture",
        display_mode=("per_cell" if use_per_cell else "global_log"),
        cmap="plasma",
        weight_opacity=weight_opacity,
        cams_nc_path=cams_nc_path if has_cams else None,
        m_area=m_area,
        cams_ds=ds_handle,
        clip_alpha_to_cams=has_cams,
        strip_prefixes=("weight_share_agri_",),
        pollutant_priority=_ppri,
        exclusive_pollutant_panel=_pexc,
    )
    if has_cams and ds_handle is not None:
        enrich_cams_grid_with_popups(
            grid_fc,
            view=view,
            m_area=m_area,
            ds=ds_handle,
            sector_title="K_Agriculture",
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
        name="CAMS GNFR K+L area source grid (cell outlines)",
        show=False,
        colour="#2e7d32",
    )

    legend_wmode = "per_cell" if use_per_cell else "global_log"
    sections: list[LegendSection] = [
        LegendSection(
            title="K_Agriculture - preview",
            html=(
                "<p class=\"pl-meta\">Weight raster: <code>"
                + wt.name
                + f"</code> (band {int(weight_band)} - <b>{pol_label}</b>)</p>"
                "<p class=\"pl-hint\">GNFR K+L agriculture weights allocated by NUTS2 "
                "and CORINE agricultural class. The CORINE layer below shows which pixels "
                "actually receive agricultural emissions.</p>"
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
                title=f"Weight scale (GNFR K+L - {pol_label})",
            )
        )
    sections.append(
        LegendSection(
            title="CORINE agricultural classes",
            html=categorical_swatch_html([
                (f"CLC {code} - {label}", hex_)
                for code, label, hex_ in _AGRI_CLC_SPECS
            ], cols=1),
            open=False,
        )
    )
    if has_cams:
        sections.append(
            LegendSection(
                title="CAMS grid (GNFR K + L)",
                html=(
                    "<p class=\"pl-hint\">Cell outlines: CAMS-REG GNFR K (emission 14) + "
                    f"L (15) area sources for <code>country_id={cams_country_iso3}</code>.</p>"
                    f"<p class=\"pl-meta\">NetCDF: <code>{cams_nc_display}</code></p>"
                ),
                open=False,
            )
        )

    legend_html = render_unified_legend(
        "K_Agriculture",
        sections,
        pollutant_badge=pol_label,
        region_note=region_note(region, override_bbox),
    )
    return save_folium_map(fmap, out_html, root, legend_html=legend_html)
