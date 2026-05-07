"""Folium preview for C_OtherCombustion: CORINE urban morphology (CLC used in weights), Hotmaps, weights."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from PROXY.core.cams.gnfr import gnfr_code_to_index
from PROXY.core.cams.mask import other_combustion_area_mask
from PROXY.core.corine.encoding import decode_corine_to_l3_pixels, normalized_corine_pixel_encoding
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
from PROXY.visualization.overlay_utils import (
    read_corine_clc_wgs84_on_weight_grid,
    read_raster_on_weight_grid_wgs84,
    scalar_to_rgba,
)


def _hotmaps_paths(
    root: Path,
    path_cfg: dict[str, Any],
    sector_cfg: dict[str, Any],
) -> dict[str, Path]:
    spec = (path_cfg.get("proxy_specific") or {}).get("other_combustion") or {}
    if not spec:
        raise ValueError("paths.yaml must define proxy_specific.other_combustion for Hotmaps paths.")
    hm_dir = Path(spec["hotmaps_dir"])
    if not hm_dir.is_absolute():
        hm_dir = root / hm_dir
    hm = sector_cfg.get("hotmaps") or {}
    return {
        "heat_res": hm_dir / str(hm.get("heat_res", "heat_res_curr_density.tif")),
        "heat_nonres": hm_dir / str(hm.get("heat_nonres", "heat_nonres_curr_density.tif")),
        "gfa_res": hm_dir / str(hm.get("gfa_res", "gfa_res_curr_density.tif")),
        "gfa_nonres": hm_dir / str(hm.get("gfa_nonres", "gfa_nonres_curr_density.tif")),
    }


def _clc_index_for_morphology(
    clc_raw: np.ndarray,
    ok: np.ndarray,
    *,
    urban_111: int,
    urban_112: int,
    urban_121: int,
    pixel_encoding: str | None = "l3_code",
    repo_root: Path | None = None,
    pixel_value_map: str | Path | None = None,
) -> np.ndarray:
    """Integer Level-3 CLC code per display pixel, aligned with ``preprocess.load_and_build_fields``."""
    gh, gw = clc_raw.shape
    z = np.full((gh, gw), -1, dtype=np.int32)
    enc = normalized_corine_pixel_encoding(pixel_encoding)
    raw_masked = np.where(ok, clc_raw, np.nan).astype(np.float32)
    clc_dec = decode_corine_to_l3_pixels(
        raw_masked,
        enc,
        repo_root=repo_root,
        pixel_value_map=pixel_value_map,
    )
    finite = ok & np.isfinite(clc_dec)
    z[finite] = np.rint(np.clip(clc_dec[finite], -1e9, 1e9)).astype(np.int32)
    targets = (int(urban_111), int(urban_112), int(urban_121))
    if any(int(np.sum(z == c)) > 0 for c in targets):
        return z

    if enc == "l3_code":
        z_raw = np.full((gh, gw), -1, dtype=np.int32)
        z_raw[ok] = np.rint(np.clip(clc_raw[ok], -1e9, 1e9)).astype(np.int32)
        z2 = z_raw.copy()
        m = ok & (z_raw >= 100)
        z2[m] = (z_raw[m].astype(np.int64) % 1000).astype(np.int32)
        if any(int(np.sum(z2 == c)) > 0 for c in targets):
            print(
                "[visualize] CORINE: no exact CLC match for morphology; using (code % 1000) "
                f"for pixels with code >= 100 (classes {targets}).",
                file=sys.stderr,
            )
            return z2

    finite_raw = np.isfinite(clc_raw) & ok
    if np.any(finite_raw):
        sr = np.rint(np.clip(clc_raw[finite_raw], -1e9, 1e9)).astype(np.int64)
        u = np.unique(sr)
        ustr = ", ".join(str(int(x)) for x in u[: min(25, u.size)])
        print(
            "[visualize] CORINE morphology: zero pixels for CLC "
            f"{urban_111}/{urban_112}/{urban_121}. Sample raster values: {ustr}.",
            file=sys.stderr,
        )
    return z


def _morphology_weighting_corine_rgba(
    clc_i: np.ndarray,
    *,
    urban_111: int,
    urban_112: int,
    urban_121: int,
) -> np.ndarray:
    """RGB mask for CLC classes that enter ``preprocess.build_X_stack`` morphology (same as build)."""
    h, w = clc_i.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    specs: list[tuple[int, tuple[int, int, int]]] = [
        (int(urban_111), (230, 81, 0)),
        (int(urban_112), (253, 216, 53)),
        (int(urban_121), (142, 68, 173)),
    ]
    for code, rgb in specs:
        m = clc_i == code
        if np.any(m):
            rgba[m, 0] = rgb[0]
            rgba[m, 1] = rgb[1]
            rgba[m, 2] = rgb[2]
            rgba[m, 3] = 255
    return np.ascontiguousarray(rgba)


def _residential_nonresidential_diff_rgba(
    z_res: np.ndarray,
    z_nonres: np.ndarray,
) -> np.ndarray:
    """Diverging RdBu overlay: (res - nonres) / (res + nonres) per display pixel.

    Red = residential dominates, blue = non-residential dominates, near-white = similar.
    This is the "do the two morphology stacks actually differ?" diagnostic the user asked for.
    """
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import colormaps

    a = np.where(np.isfinite(z_res), z_res, 0.0)
    b = np.where(np.isfinite(z_nonres), z_nonres, 0.0)
    denom = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(denom > 0, (a - b) / denom, np.nan)
    h, w = frac.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    valid = np.isfinite(frac)
    if not np.any(valid):
        return rgba
    t = np.clip((frac[valid] + 1.0) * 0.5, 0.0, 1.0)
    cmap = colormaps["RdBu_r"]
    c = cmap(t)
    rgba[valid, 0] = (np.clip(c[:, 0], 0.0, 1.0) * 255).astype(np.uint8)
    rgba[valid, 1] = (np.clip(c[:, 1], 0.0, 1.0) * 255).astype(np.uint8)
    rgba[valid, 2] = (np.clip(c[:, 2], 0.0, 1.0) * 255).astype(np.uint8)
    rgba[valid, 3] = 210
    return rgba


def write_other_combustion_area_html(
    *,
    root: Path,
    weight_tif: Path,
    corine_tif: Path,
    out_html: Path,
    sector_cfg: dict[str, Any],
    path_cfg: dict[str, Any],
    pad_deg: float = 0.02,
    max_width: int = 1400,
    max_height: int = 1200,
    weight_opacity: float = 0.92,
    corine_opacity: float = 0.75,
    hotmaps_opacity: float = 0.62,
    cams_nc_path: Path | None = None,
    cams_country_iso3: str = "GRC",
    weight_display_mode: str = "global_log",
    region: str | None = None,
    override_bbox: tuple[float, float, float, float] | None = None,
) -> Path:
    """
    Folium map: basemaps, CORINE CLC pixels used in weight morphology (111/112/121 on the weight grid),
    four Hotmaps rasters (log), multiband GNFR C weights, optional CAMS grid. A new
    "residential vs non-residential" diagnostic RdBu layer shows where the two stacks actually differ.
    """
    try:
        import folium  # noqa: F401
    except ImportError as exc:
        raise SystemExit(VIZ_DEPS_MESSAGE) from exc

    viz_cfg = sector_cfg.get("visualization") or {}
    corine_band = int(
        viz_cfg.get("corine_band") or (sector_cfg.get("corine") or {}).get("band", 1)
    )
    cams_block = sector_cfg.get("cams") or {}
    bbox_cfg = cams_block.get("domain_bbox_wgs84")
    domain_bbox = tuple(float(x) for x in bbox_cfg) if bbox_cfg else None
    source_types = tuple(cams_block.get("source_types") or ("area",))
    gnfr = str(cams_block.get("gnfr", "C"))

    wt = resolve_under_root(weight_tif, root)
    ct = resolve_under_root(corine_tif, root)
    if not wt.is_file():
        raise FileNotFoundError(f"Weight GeoTIFF not found: {wt}")
    if not ct.is_file():
        raise FileNotFoundError(f"CORINE GeoTIFF not found: {ct}")

    weight_band = pick_band_by_pollutant(wt, viz_cfg, sector_cfg=sector_cfg)
    pol_label = pollutant_label_for_band(wt, weight_band)
    corine_opacity = float(viz_cfg.get("corine_opacity", corine_opacity))
    hotmaps_opacity = float(viz_cfg.get("hotmaps_opacity", hotmaps_opacity))
    weight_opacity = float(viz_cfg.get("weight_opacity", weight_opacity))

    view = compute_view_context(
        wt, pad_deg=pad_deg, max_width=max_width, max_height=max_height,
        region=region, override_bbox=override_bbox,
    )

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
    ok = np.isfinite(clc_raw)
    if clc_nd is not None:
        ok = ok & (clc_raw != float(clc_nd))
    morph = sector_cfg.get("morphology") or {}
    u111 = int(morph.get("urban_111", 111))
    u112 = int(morph.get("urban_112", 112))
    u121 = int(morph.get("urban_121", 121))
    cor_block = sector_cfg.get("corine") or {}
    px_enc = cor_block.get("pixel_encoding")
    pmap = cor_block.get("pixel_value_map")
    clc_i = _clc_index_for_morphology(
        clc_raw,
        ok,
        urban_111=u111,
        urban_112=u112,
        urban_121=u121,
        pixel_encoding=px_enc,
        repo_root=root,
        pixel_value_map=pmap,
    )
    rgba_corine = _morphology_weighting_corine_rgba(
        clc_i, urban_111=u111, urban_112=u112, urban_121=u121
    )

    hm_paths = _hotmaps_paths(root, path_cfg, sector_cfg)
    for label, p in hm_paths.items():
        if not p.is_file():
            raise FileNotFoundError(
                f"Hotmaps raster missing ({label}): {p}. Check proxy_specific.other_combustion and sector hotmaps filenames."
            )

    has_cams = bool(cams_nc_path) and Path(cams_nc_path).is_file()
    cams_nc_display = Path(cams_nc_path).name if has_cams and cams_nc_path else ""
    use_per_cell = has_cams and weight_display_mode.strip().lower() == "per_cell"

    grid_fc: dict[str, Any] = {"type": "FeatureCollection", "features": []}
    m_area = None
    ds_handle = None
    if has_cams:
        ds_handle = xr.open_dataset(cams_nc_path, engine="netcdf4")
        try:
            m_area = other_combustion_area_mask(
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

    hotmap_layers: list[tuple[str, str, np.ndarray, np.ndarray | None]] = []
    hm_specs = (
        ("Hotmaps: residential heat density", "viridis", "heat_res"),
        ("Hotmaps: non-residential heat density", "cividis", "heat_nonres"),
        ("Hotmaps: residential GFA density", "plasma", "gfa_res"),
        ("Hotmaps: non-residential GFA density", "inferno", "gfa_nonres"),
    )
    heat_res_z: np.ndarray | None = None
    heat_nonres_z: np.ndarray | None = None
    for title, cmap_name, key in hm_specs:
        stk = read_raster_on_weight_grid_wgs84(
            wt,
            hm_paths[key],
            aux_band=1,
            west=view.west,
            south=view.south,
            east=view.east,
            north=view.north,
            display_width=view.gw,
            display_height=view.gh,
            resampling="bilinear",
        )
        z = stk["values_wgs84"]
        nd = stk["nodata"]
        rgba_h = scalar_to_rgba(
            z,
            colour_mode="log",
            cmap_name=cmap_name,
            hide_zero=True,
            nodata_val=float(nd) if nd is not None else None,
        )
        hotmap_layers.append((title, cmap_name, rgba_h, z))
        if key == "heat_res":
            heat_res_z = z
        elif key == "heat_nonres":
            heat_nonres_z = z

    fmap = create_folium_map_with_tiles(view, zoom_start=8)

    for title, _cmap_name, rgba_h, _z in hotmap_layers:
        add_raster_overlay(
            fmap, rgba_h, view, name=title, opacity=hotmaps_opacity, show=False
        )

    if heat_res_z is not None and heat_nonres_z is not None:
        rgba_diff = _residential_nonresidential_diff_rgba(heat_res_z, heat_nonres_z)
        add_raster_overlay(
            fmap,
            rgba_diff,
            view,
            name="Residential vs non-residential diff (RdBu: red=res, blue=non-res)",
            opacity=0.85,
            show=True,
        )

    add_raster_overlay(
        fmap,
        rgba_corine,
        view,
        name=(
            f"CORINE: weighting morphology (CLC {u111} / {u112} / {u121} only, "
            "same grid as weights)"
        ),
        opacity=corine_opacity,
        show=False,
    )

    _ppri, _pexc = visualization_pollutant_priority_from_cfg(viz_cfg)
    mp = add_multipollutant_weight_layers(
        fmap,
        view,
        wt,
        preferred_band=weight_band,
        preferred_label=pol_label,
        sector_key="C_OtherCombustion",
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
            sector_title="C_OtherCombustion",
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
        name="CAMS GNFR C area source grid (cell outlines)",
        show=False,
    )

    sections: list[LegendSection] = [
        LegendSection(
            title="C_OtherCombustion - preview",
            html=(
                "<p class=\"pl-meta\">Weight raster: <code>"
                + wt.name
                + f"</code> (band {int(weight_band)} - <b>{pol_label}</b>)</p>"
                "<p class=\"pl-hint\">GNFR C / other combustion weights built from Hotmaps "
                "residential + non-residential heat &amp; GFA and CORINE urban morphology. "
                "The RdBu layer flags pixels where the two stacks differ.</p>"
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
                title=f"Weight scale (GNFR {gnfr} - {pol_label})",
            )
        )
    sections.extend([
        LegendSection(
            title="Residential vs non-residential (diff)",
            html=(
                "<p class=\"pl-hint\">"
                "<b>Red</b>: residential Hotmaps dominates; "
                "<b>blue</b>: non-residential dominates; "
                "<b>white</b>: both tracks are similar in magnitude.</p>"
                + categorical_swatch_html([
                    ("Residential dominant", "#b2182b"),
                    ("Similar (near-white)", "#f7f7f7"),
                    ("Non-residential dominant", "#2166ac"),
                ])
            ),
            open=True,
        ),
        LegendSection(
            title="CORINE morphology classes",
            html=categorical_swatch_html([
                (f"CLC {u111} - continuous urban fabric", "#e65100"),
                (f"CLC {u112} - discontinuous urban fabric", "#fdd835"),
                (f"CLC {u121} - industrial / commercial", "#8e44ad"),
            ]),
            open=False,
        ),
    ])
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
        "C_OtherCombustion",
        sections,
        pollutant_badge=pol_label,
        region_note=region_note(region, override_bbox),
    )
    return save_folium_map(fmap, out_html, root, legend_html=legend_html)
