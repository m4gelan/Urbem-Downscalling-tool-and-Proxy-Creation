from __future__ import annotations

import io
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from matplotlib import colormaps
from matplotlib import cm as mpl_cm
from rasterio.enums import Resampling
from rasterio.transform import array_bounds, from_bounds as affine_from_bounds
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window, from_bounds as window_from_bounds, transform as window_transform

# =============================================================================
# Configuration — edit layer colours / titles per sector (used for maps + JPG names)
# =============================================================================
#
# Matplotlib / Folium colormap names you can use (kind="float"):
#   viridis, plasma, inferno, magma, cividis, turbo, YlOrBr, PuBu, Blues, Greens,
#   Oranges, Purples, YlGn, terrain, copper, bone, gist_heat
#
# kind options:
#   float       — continuous raster + colour scale
#   corine_gray — binary mask (gray)
#   mask_rgb    — binary mask with fixed RGB (set rgb=(r,g,b), alpha=0-255)
#   osm_blue    — OSM-style blue points/lines
#
# scale_per_cams: W layers normalised to max within each CAMS cell; low tail masked (see _CAMS_DISPLAY_FLOOR_FRAC)
# show: default visibility on interactive map (LayerControl still toggles all)

@dataclass(frozen=True)
class LayerStyle:
    title: str
    cmap: str = "viridis"
    kind: str = "float"
    opacity: float = 0.82
    show: bool = False
    scale_per_cams: bool = False
    nearest: bool = False
    vmin: float | None = None
    vmax: float | None = None
    rgb: tuple[int, int, int] = (65, 65, 65)
    alpha: int = 220


# Per-sector layer styles (alphabetical). Keys must match raster dict keys passed to _emit_sector_viz.
# Dynamic layers: use *_default entries; optional overrides e.g. w_poll_nh3, osm_refineries_petroleum.
# Sectors without area-weight debug maps: F_Roads, H_Aviation (point matching only).
SECTOR_LAYER_STYLES: dict[str, dict[str, LayerStyle]] = {
    "A_PublicPower": {
        "corine": LayerStyle("CORINE mask", kind="corine_gray", show=True, opacity=0.88),
        "popz": LayerStyle("Population z-score", cmap="viridis", show=False),
        "W": LayerStyle("Weight W", cmap="inferno", scale_per_cams=True, show=True, opacity=0.85),
    },
    "B_Industry": {
        "c121": LayerStyle("CORINE L3 121 industrial", kind="mask_rgb", rgb=(139, 90, 43), show=True),
        "c131": LayerStyle("CORINE L3 131 mineral", kind="mask_rgb", rgb=(34, 139, 34), show=False),
        "osm_refineries_petroleum": LayerStyle("OSM refineries", kind="mask_rgb", rgb=(30, 110, 255), show=False),
        "osm_manufacturing_combustion_residual": LayerStyle("OSM manufacturing", kind="mask_rgb", rgb=(40, 170, 65), show=False),
        "osm_mineral": LayerStyle("OSM mineral", kind="mask_rgb", rgb=(165, 95, 40), show=False),
        "osm_chemical_metal": LayerStyle("OSM chemical metal", kind="mask_rgb", rgb=(140, 45, 200), show=False),
        "W_default": LayerStyle("W inventory group", cmap="inferno", scale_per_cams=True, show=False, opacity=0.84),
        "wp_default": LayerStyle("W fused pollutant", cmap="plasma", scale_per_cams=True, show=True, opacity=0.86),
        "wp_pm10": LayerStyle("W fused PM10", cmap="plasma", scale_per_cams=True, show=True),
        "wp_co": LayerStyle("W fused CO", cmap="plasma", scale_per_cams=True, show=False),
    },
    "C_Othercombustion": {
        "popz": LayerStyle("Population z-score", cmap="viridis", show=True),
        "hres": LayerStyle("Hotmaps H_res z-score", cmap="viridis", show=False),
        "hnres": LayerStyle("Hotmaps H_nres z-score", cmap="viridis", show=False),
        "hdd": LayerStyle("HDD z-score", cmap="viridis", show=False),
        "u111": LayerStyle("CORINE u111 continuous urban", kind="mask_rgb", rgb=(205, 92, 92), show=True),
        "u112": LayerStyle("CORINE u112 discontinuous urban", kind="mask_rgb", rgb=(210, 105, 30), show=False),
        "u121": LayerStyle("CORINE u121 industrial", kind="mask_rgb", rgb=(30, 144, 200), show=False),
        "S_default": LayerStyle("Stock S", cmap="YlOrBr", show=False, opacity=0.84),
        "L_default": LayerStyle("Load L", cmap="PuBu", show=False, opacity=0.84),
        "ws_default": LayerStyle("W stationary", cmap="inferno", scale_per_cams=True, show=False, opacity=0.86),
        "wo_default": LayerStyle("W offroad", cmap="cividis", scale_per_cams=True, show=False, opacity=0.86),
        "wc_default": LayerStyle("W combined", cmap="plasma", scale_per_cams=True, show=True, opacity=0.86),
    },
    "D_Fugitive": {
        "c121": LayerStyle("CORINE L3 121", kind="mask_rgb", rgb=(139, 90, 43), show=True),
        "c123": LayerStyle("CORINE L3 123 ports", kind="mask_rgb", rgb=(30, 144, 200), show=False),
        "c131": LayerStyle("CORINE L3 131 mineral", kind="mask_rgb", rgb=(34, 139, 34), show=False),
        "popz": LayerStyle("Population z-score", cmap="viridis", show=False),
        "gemc": LayerStyle("GEM coal mines", kind="mask_rgb", rgb=(101, 67, 33), show=False),
        "gemo": LayerStyle("GEM oil/gas", kind="mask_rgb", rgb=(28, 28, 28), show=False),
        "vnf": LayerStyle("VIIRS Nightfire VNF", kind="mask_rgb", rgb=(255, 140, 0), show=False),
        "osm_slot_default": LayerStyle("OSM slot", kind="mask_rgb", rgb=(30, 110, 255), show=False, nearest=True),
        "W_default": LayerStyle("W subgroup", cmap="inferno", scale_per_cams=True, show=False, opacity=0.84),
        "wp_default": LayerStyle("W fused pollutant", cmap="plasma", scale_per_cams=True, show=True, opacity=0.86),
        "wp_nmvoc": LayerStyle("W fused NMVOC", cmap="plasma", scale_per_cams=True, show=True),
        "wp_nox": LayerStyle("W fused NOx", cmap="plasma", scale_per_cams=True, show=False),
    },
    "E_Solvents": {
        "household": LayerStyle("CORINE household", kind="mask_rgb", rgb=(180, 60, 60), show=True),
        "service": LayerStyle("CORINE service", kind="mask_rgb", rgb=(30, 144, 200), show=False),
        "industrial": LayerStyle("CORINE industrial", kind="mask_rgb", rgb=(139, 90, 43), show=False),
        "transport": LayerStyle("CORINE transport", kind="mask_rgb", rgb=(255, 165, 0), show=False),
        "osm_slot_default": LayerStyle("OSM slot", kind="mask_rgb", rgb=(30, 110, 255), show=False, nearest=True),
        "W_default": LayerStyle("W subsector", cmap="inferno", scale_per_cams=True, show=False, opacity=0.84),
        "wp_default": LayerStyle("W fused pollutant", cmap="plasma", scale_per_cams=True, show=True, opacity=0.86),
        "wp_nmvoc": LayerStyle("W fused NMVOC", cmap="plasma", scale_per_cams=True, show=True),
        "wp_nox": LayerStyle("W fused NOx", cmap="plasma", scale_per_cams=True, show=False),
    },
    "G_Shipping": {
        "corine": LayerStyle("CORINE mask", kind="corine_gray", show=True),
        "osm": LayerStyle("OSM shipping", kind="osm_blue", show=True),
        "emod": LayerStyle("EMODNET z-score", cmap="viridis", show=False),
        "W": LayerStyle("Weight W", cmap="inferno", scale_per_cams=True, show=True, opacity=0.85),
    },
    "I_Offroad": {
        "c121": LayerStyle("CORINE L3 121", kind="mask_rgb", rgb=(139, 90, 43), show=True),
        "c123": LayerStyle("CORINE L3 123 ports", kind="mask_rgb", rgb=(30, 144, 200), show=False),
        "c124": LayerStyle("CORINE L3 124 transport", kind="mask_rgb", rgb=(70, 130, 180), show=False),
        "c131": LayerStyle("CORINE L3 131 mineral", kind="mask_rgb", rgb=(34, 139, 34), show=False),
        "osm_slot_default": LayerStyle("OSM slot", kind="mask_rgb", rgb=(30, 110, 255), show=False, nearest=True),
        "W_default": LayerStyle("W subgroup", cmap="inferno", scale_per_cams=True, show=False, opacity=0.84),
        "wp_default": LayerStyle("W fused pollutant", cmap="plasma", scale_per_cams=True, show=True, opacity=0.86),
        "wp_nmvoc": LayerStyle("W fused NMVOC", cmap="plasma", scale_per_cams=True, show=True),
        "wp_nox": LayerStyle("W fused NOx", cmap="plasma", scale_per_cams=True, show=False),
    },
    "J_Waste": {
        "osm": LayerStyle("OSM waste layers", kind="osm_blue", show=True),
        "c132": LayerStyle("CORINE CLC 132 dump sites", kind="mask_rgb", rgb=(65, 65, 65), show=False),
        "c121": LayerStyle("CORINE CLC 121 industrial", kind="mask_rgb", rgb=(139, 90, 43), show=False),
        "impz": LayerStyle("Imperviousness z-score", cmap="cividis", show=False),
        "upl": LayerStyle("UWWTD treatment plants", cmap="Oranges", vmin=0.0, vmax=1.0, show=False),
        "uag": LayerStyle("UWWTD agglomerations", cmap="Purples", vmin=0.0, vmax=1.0, show=False),
        "rur": LayerStyle("GHSL rural mask", cmap="Greens", vmin=0.0, vmax=1.0, show=False),
        "popz": LayerStyle("Population z-score", cmap="viridis", show=False),
        "Ws": LayerStyle("W solid waste", cmap="inferno", scale_per_cams=True, show=False),
        "Ww": LayerStyle("W wastewater", cmap="inferno", scale_per_cams=True, show=False),
        "Wr": LayerStyle("W residual", cmap="inferno", scale_per_cams=True, show=False),
        "wp_default": LayerStyle("W fused pollutant", cmap="plasma", scale_per_cams=True, show=True, opacity=0.85),
        "wp_nmvoc": LayerStyle("W fused NMVOC", cmap="plasma", scale_per_cams=True, show=True),
        "wp_sox": LayerStyle("W fused SOx", cmap="plasma", scale_per_cams=True, show=False),
    },
    "K_Agriculture": {
        "w_group_default": LayerStyle("W group", cmap="inferno", scale_per_cams=True, show=False),
        "w_poll_default": LayerStyle("W fused pollutant", cmap="plasma", scale_per_cams=True, show=False),
        "w_group_livestock": LayerStyle("W livestock", cmap="inferno", scale_per_cams=True, show=True),
        "w_group_manure": LayerStyle("W manure", cmap="inferno", scale_per_cams=True, show=False),
        "w_group_grazed_pastures": LayerStyle("W grazed pastures", cmap="inferno", scale_per_cams=True, show=False),
        "w_group_farm_buildings": LayerStyle("W farm buildings", cmap="inferno", scale_per_cams=True, show=False),
        "w_group_inorganic_n": LayerStyle("W inorganic N", cmap="inferno", scale_per_cams=True, show=False),
        "w_group_crop_nmvoc": LayerStyle("W crop NMVOC", cmap="inferno", scale_per_cams=True, show=False),
        "w_group_broad_agricultural": LayerStyle("W broad agricultural", cmap="inferno", scale_per_cams=True, show=False),
        "w_group_biomass_burning": LayerStyle("W biomass burning", cmap="inferno", scale_per_cams=True, show=False),
        "w_poll_nh3": LayerStyle("W fused NH3", cmap="plasma", scale_per_cams=True, show=True),
        "w_poll_nmvoc": LayerStyle("W fused NMVOC", cmap="plasma", scale_per_cams=True, show=False),
    },
}

_CORINE_L3_RGB: dict[str, tuple[int, int, int, int]] = {
    "c121": (139, 90, 43, 220),
    "c123": (30, 144, 200, 218),
    "c124": (70, 130, 180, 218),
    "c131": (34, 139, 34, 215),
    "c132": (65, 65, 65, 235),
    "u111": (205, 92, 92, 220),
    "u112": (210, 105, 30, 220),
    "u121": (30, 144, 200, 218),
}

_CORINE_SOLVENTS_RGB: dict[str, tuple[int, int, int, int]] = {
    "household": (180, 60, 60, 220),
    "service": (30, 144, 200, 218),
    "industrial": (139, 90, 43, 220),
    "transport": (255, 165, 0, 215),
}

_INDUSTRY_OSM_RGB: dict[str, tuple[int, int, int]] = {
    "refineries_petroleum": (30, 110, 255),
    "manufacturing_combustion_residual": (40, 170, 65),
    "mineral": (165, 95, 40),
    "chemical_metal": (140, 45, 200),
}

_OSM_SLOT_RGB: list[tuple[int, int, int]] = [
    (30, 110, 255),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
]

_OSM_TILES = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
_SATELLITE_TILES = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
)
_MAX_DISPLAY_PX = 2400
_EXPORT_DPI = 200
_TITLE_BAND_IN = 0.55
_FOOTER_BAND_IN = 0.32
_HPAD_IN = 0.12
# Per CAMS cell: hide values below this fraction of the cell max (transparent on map, not colormap floor).
_CAMS_DISPLAY_FLOOR_FRAC = 0.05


# =============================================================================
# Shared helpers
# =============================================================================


def map_type() -> str:
    """INTERACTIVE or FIXED_IMAGE — from proxy.entry.MAP_TYPE."""
    try:
        from proxy.entry import MAP_TYPE

        return str(MAP_TYPE).strip().upper()
    except Exception:
        return "INTERACTIVE"


def _slug(title: str) -> str:
    s = re.sub(r"[^\w\-]+", "_", title.strip().lower())
    return re.sub(r"_+", "_", s).strip("_")[:80] or "layer"


def _intersect_bbox_wgs84_with_raster_extent(
    bbox_wgs84: tuple[float, float, float, float],
    transform: rasterio.Affine,
    raster_crs: Any,
    height: int,
    width: int,
) -> tuple[float, float, float, float]:
    west, south, east, north = bbox_wgs84
    rw, rs, re, rn = transform_bounds(
        raster_crs, "EPSG:4326", *array_bounds(height, width, transform), densify_pts=21
    )
    iw, ie = max(float(west), float(rw)), min(float(east), float(re))
    is_, in_ = max(float(south), float(rs)), min(float(north), float(rn))
    if iw >= ie or is_ >= in_:
        raise ValueError(
            "DEBUG map bbox (WGS84) does not overlap this raster window. "
            f"bbox=({west:g},{south:g},{east:g},{north:g}) "
            f"raster_extent_WGS84≈({rw:.4f},{rs:.4f},{re:.4f},{rn:.4f})."
        )
    return (iw, is_, ie, in_)


def _bbox_crop_window(
    height: int,
    width: int,
    transform: rasterio.Affine,
    raster_crs: Any,
    bbox_wgs84: tuple[float, float, float, float],
) -> tuple[int, int, int, int, rasterio.Affine, tuple[float, float, float, float]]:
    bbox_use = _intersect_bbox_wgs84_with_raster_extent(
        bbox_wgs84, transform, raster_crs, height, width
    )
    west, south, east, north = bbox_use
    left, bottom, right, top = transform_bounds(
        "EPSG:4326", raster_crs, west, south, east, north, densify_pts=11
    )
    win = window_from_bounds(left, bottom, right, top, transform).intersection(
        Window(0, 0, width, height)
    )
    if win.width <= 0 or win.height <= 0:
        raise ValueError("bbox/raster intersection is empty after projection.")
    win = win.round_offsets(op="floor").round_lengths(op="ceil")
    row_off, col_off = int(win.row_off), int(win.col_off)
    hh, ww = int(win.height), int(win.width)
    if hh <= 0 or ww <= 0:
        raise ValueError("bbox does not intersect the raster window")
    return row_off, col_off, hh, ww, window_transform(win, transform), bbox_use


def _crop_array_window(arr: np.ndarray, row_off: int, col_off: int, hh: int, ww: int) -> np.ndarray:
    return arr[row_off : row_off + hh, col_off : col_off + ww]


def _crop_to_wgs84_bbox(
    arrs: dict[str, np.ndarray],
    transform: rasterio.Affine,
    raster_crs: Any,
    bbox_wgs84: tuple[float, float, float, float],
) -> tuple[dict[str, np.ndarray], rasterio.Affine, tuple[float, float, float, float]]:
    h0, w0 = next(iter(arrs.values())).shape
    row_off, col_off, hh, ww, sub_tr, bbox_use = _bbox_crop_window(
        h0, w0, transform, raster_crs, bbox_wgs84
    )
    sub = {k: _crop_array_window(v, row_off, col_off, hh, ww) for k, v in arrs.items()}
    return sub, sub_tr, bbox_use


def _warp_to_wgs84_grid(
    src: np.ndarray,
    src_transform: rasterio.Affine,
    src_crs: Any,
    west: float,
    south: float,
    east: float,
    north: float,
    dst_h: int,
    dst_w: int,
    resampling: Resampling,
    *,
    dst_dtype: np.dtype,
    fill_value: float | int | None,
) -> np.ndarray:
    dst_tr = affine_from_bounds(west, south, east, north, dst_w, dst_h)
    if np.issubdtype(dst_dtype, np.integer):
        dst = np.full((dst_h, dst_w), int(fill_value or 0), dtype=dst_dtype)
        extra: dict[str, Any] = {}
    elif fill_value is not None and isinstance(fill_value, float) and np.isnan(fill_value):
        dst = np.full((dst_h, dst_w), np.nan, dtype=dst_dtype)
        extra = {"dst_nodata": np.nan}
    else:
        dst = np.full((dst_h, dst_w), fill_value or 0, dtype=dst_dtype)
        extra = {}
    reproject(
        source=np.asarray(src),
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_tr,
        dst_crs="EPSG:4326",
        resampling=resampling,
        **extra,
    )
    return dst


def _W_scaled_per_cell(W: np.ndarray, cell_id: np.ndarray, cams_cells: dict[int, dict[str, Any]]) -> np.ndarray:
    out = np.zeros_like(W, dtype=np.float32)
    for cid in cams_cells:
        m = cell_id == int(cid)
        if not np.any(m):
            continue
        mx = float(np.max(W[m]))
        if mx > 0.0:
            out[m] = (W[m] / mx).astype(np.float32)
    return out


def _mask_low_within_cams_cells(
    z: np.ndarray,
    cell_id: np.ndarray,
    cams_cells: dict[int, dict[str, Any]],
    *,
    floor_frac: float = _CAMS_DISPLAY_FLOOR_FRAC,
) -> np.ndarray:
    """Display-only: NaN pixels below floor_frac * cell max (drops low in-cell tail from colormap)."""
    out = np.asarray(z, dtype=np.float32).copy()
    thr_frac = float(floor_frac)
    for cid in cams_cells:
        m = cell_id == int(cid)
        if not np.any(m):
            continue
        band = out[m]
        mx = float(np.max(band))
        if mx <= 0.0:
            out[m] = np.nan
            continue
        band = band.copy()
        band[band < thr_frac * mx] = np.nan
        out[m] = band
    return out


def _get_cmap(name: str):
    try:
        return colormaps[name]
    except (KeyError, AttributeError, TypeError):
        return mpl_cm.get_cmap(name)


def _float_rgba(
    z: np.ndarray,
    cmap_name: str,
    valid: np.ndarray,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
) -> tuple[np.ndarray, float, float]:
    cmap = _get_cmap(cmap_name)
    zf = np.asarray(z, dtype=np.float64)
    lo, hi = 0.0, 1.0
    if np.any(valid):
        lo = float(np.nanmin(zf[valid])) if vmin is None else float(vmin)
        hi = float(np.nanmax(zf[valid])) if vmax is None else float(vmax)
        rng = max(hi - lo, 1e-12)
        t = np.clip((zf - lo) / rng, 0.0, 1.0)
    else:
        t = np.zeros_like(zf)
    rgba = (cmap(t) * 255.0).astype(np.uint8)
    rgba[~valid, 3] = 0
    return rgba, lo, hi


def _mask_rgba(mask: np.ndarray, r: int, g: int, b: int, alpha: int = 215) -> np.ndarray:
    return _mask_rgba_outlined(mask, r, g, b, alpha=alpha)


def _dilate_once(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask).astype(bool)
    out = m.copy()
    out[1:, :] |= m[:-1, :]
    out[:-1, :] |= m[1:, :]
    out[:, 1:] |= m[:, :-1]
    out[:, :-1] |= m[:, 1:]
    return out


def _binary_edge(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask).astype(bool)
    inner = m.copy()
    inner[1:, :] &= m[:-1, :]
    inner[:-1, :] &= m[1:, :]
    inner[:, 1:] &= m[:, :-1]
    inner[:, :-1] &= m[:, 1:]
    return m & ~inner


def _mask_rgba_outlined(
    mask: np.ndarray,
    r: int,
    g: int,
    b: int,
    *,
    alpha: int = 115,
    outline_rgb: tuple[int, int, int] = (17, 17, 17),
    outline_alpha: int = 255,
    dilate: bool = True,
) -> np.ndarray:
    m = _dilate_once(np.asarray(mask).astype(bool)) if dilate else np.asarray(mask).astype(bool)
    edge = _binary_edge(m)
    rgba = np.zeros((*m.shape, 4), dtype=np.uint8)
    rgba[edge, 0], rgba[edge, 1], rgba[edge, 2], rgba[edge, 3] = outline_rgb[0], outline_rgb[1], outline_rgb[2], outline_alpha
    rgba[m, 0], rgba[m, 1], rgba[m, 2], rgba[m, 3] = r, g, b, alpha
    return rgba


def _corine_gray_rgba(mask: np.ndarray) -> np.ndarray:
    return _mask_rgba(mask, 65, 65, 65, 235)


def _osm_blue_rgba(osm: np.ndarray) -> np.ndarray:
    return _mask_rgba(np.asarray(osm) > 0, 30, 110, 255, 220)


def _style_for_key(sector: str, key: str, overrides: dict[str, LayerStyle] | None) -> LayerStyle:
    if overrides and key in overrides:
        return overrides[key]
    sec = SECTOR_LAYER_STYLES.get(sector, {})
    if key in sec:
        return sec[key]
    if key.startswith("wg_"):
        g = key[3:]
        sk = f"w_group_{g}"
        base = sec.get(sk, sec.get("w_group_default", LayerStyle(f"W {g.replace('_', ' ')}", cmap="inferno", scale_per_cams=True)))
        return replace(base, title=f"W {g.replace('_', ' ')}")
    if key.startswith("wp_"):
        pk = key[3:]
        sk = f"wp_{pk}" if f"wp_{pk}" in sec else f"w_poll_{pk}"
        base = sec.get(sk, sec.get("wp_default", sec.get("w_poll_default", LayerStyle(f"W fused {pk.upper()}", cmap="plasma", scale_per_cams=True))))
        return replace(base, title=f"W fused {pk.upper()}")
    if key.startswith("W_"):
        g = key[2:]
        base = sec.get(f"W_{g}", sec.get("W_default", LayerStyle(f"W {g}", cmap="inferno", scale_per_cams=True, show=False, opacity=0.84)))
        return replace(base, title=f"W {g.replace('_', ' ')}")
    if key.startswith("osm_") and key != "osm":
        if key in sec:
            return sec[key]
        g = key[4:]
        rgb = _INDUSTRY_OSM_RGB.get(g, (80, 80, 80))
        return LayerStyle(f"OSM {g.replace('_', ' ')}", kind="mask_rgb", rgb=rgb, show=False, nearest=True)
    if key.startswith("osm__"):
        parts = key.split("__", 2)
        label = f"OSM {parts[1]} / {parts[2]}" if len(parts) > 2 else key
        base = sec.get("osm_slot_default", LayerStyle(label[:80], kind="mask_rgb", rgb=(30, 110, 255), nearest=True))
        return replace(base, title=label[:80])
    if key in _CORINE_L3_RGB:
        r, g, b, a = _CORINE_L3_RGB[key]
        labels = {
            "c121": "CORINE L3 121",
            "c123": "CORINE L3 123",
            "c124": "CORINE L3 124",
            "c131": "CORINE L3 131",
            "c132": "CORINE L3 132",
            "u111": "CORINE u111",
            "u112": "CORINE u112",
            "u121": "CORINE u121",
        }
        return LayerStyle(labels.get(key, key), kind="mask_rgb", rgb=(r, g, b), alpha=a, nearest=True, show=key in ("c121", "u111"))
    if key in _CORINE_SOLVENTS_RGB:
        r, g, b, a = _CORINE_SOLVENTS_RGB[key]
        return LayerStyle(f"CORINE {key}", kind="mask_rgb", rgb=(r, g, b), alpha=a, nearest=True, show=key == "household")
    if key == "gemc":
        return LayerStyle("GEM coal mines", kind="mask_rgb", rgb=(101, 67, 33), show=False, nearest=True)
    if key == "gemo":
        return LayerStyle("GEM oil/gas", kind="mask_rgb", rgb=(28, 28, 28), show=False, nearest=True)
    if key == "vnf":
        return LayerStyle("VIIRS Nightfire VNF", kind="mask_rgb", rgb=(255, 140, 0), show=False, nearest=True)
    if key.startswith("ws_") or key.startswith("wo_") or key.startswith("wc_"):
        prefix = key[:2] + "_"
        branch = {"ws": "stationary", "wo": "offroad", "wc": "combined"}[key[:2]]
        pk = key[3:]
        base = sec.get(key, sec.get(f"{prefix}default", LayerStyle(f"W {branch} {pk}", cmap="inferno", scale_per_cams=True, show=key.startswith("wc_"), opacity=0.86)))
        return replace(base, title=f"W {branch} {pk}")
    if key.startswith("S_") or key.startswith("L_"):
        kind, cls = key[0], key[2:]
        dk = f"{kind}_default"
        base = sec.get(dk, LayerStyle(f"{kind} {cls}", cmap="YlOrBr" if kind == "S" else "PuBu", show=False, opacity=0.84))
        title = f"{kind} stock {cls}" if kind == "S" else f"{kind} load {cls}"
        return replace(base, title=title)
    return LayerStyle(key, cmap="viridis", show=False)


@dataclass
class RenderedLayer:
    key: str
    title: str
    rgba: np.ndarray
    cmap: str
    vmin: float
    vmax: float
    opacity: float
    show: bool
    kind: str


def _prepare_layers(
    sector: str,
    arrays: dict[str, np.ndarray],
    *,
    bbox_wgs84: tuple[float, float, float, float],
    transform: rasterio.Affine,
    raster_crs: Any,
    cams_cells: dict[int, dict[str, Any]],
    cell_id: np.ndarray,
    style_overrides: dict[str, LayerStyle] | None = None,
) -> tuple[list[RenderedLayer], tuple[float, float, float, float], tuple[int, int]]:
    keys = {k: v for k, v in arrays.items() if k != "cid"}
    keys["cid"] = cell_id
    cropped, sub_tr, bbox_use = _crop_to_wgs84_bbox(keys, transform, raster_crs, bbox_wgs84)
    west, south, east, north = bbox_use
    cid = cropped["cid"].astype(np.int64)
    hh, ww = cid.shape
    sc = min(1.0, _MAX_DISPLAY_PX / max(hh, ww))
    dst_h = max(2, int(round(hh * sc)))
    dst_w = max(2, int(round(ww * sc)))

    def warp(src: np.ndarray, nearest: bool, dtype: np.dtype, fill: Any) -> np.ndarray:
        res = Resampling.nearest if nearest else Resampling.bilinear
        return _warp_to_wgs84_grid(
            src, sub_tr, raster_crs, west, south, east, north, dst_h, dst_w, res,
            dst_dtype=dtype, fill_value=fill,
        )

    cid_w = warp(cid, True, np.dtype(np.int64), -1)
    valid_cam = cid_w >= 0
    rendered: list[RenderedLayer] = []
    slot_i = 0

    for key in sorted(k for k in cropped if k != "cid"):
        st = _style_for_key(sector, key, style_overrides)
        raw = cropped[key]
        if st.scale_per_cams:
            raw = _W_scaled_per_cell(raw.astype(np.float32), cid, cams_cells)
            raw = _mask_low_within_cams_cells(raw, cid, cams_cells)

        nearest = st.nearest or st.kind in ("corine_gray", "mask_rgb", "osm_blue")
        if st.kind == "float":
            w = warp(raw.astype(np.float32), nearest, np.dtype(np.float32), np.nan)
            valid = valid_cam & np.isfinite(w)
            rgba, vmin, vmax = _float_rgba(w, st.cmap, valid, vmin=st.vmin, vmax=st.vmax)
        elif st.kind == "corine_gray":
            w = warp(raw, True, np.dtype(np.uint8), 0)
            rgba = _corine_gray_rgba(w > 0)
            vmin, vmax = 0.0, 1.0
        elif st.kind == "osm_blue":
            w = warp(raw.astype(np.float32), True, np.dtype(np.float32), 0.0)
            rgba = _osm_blue_rgba(w)
            vmin, vmax = 0.0, 1.0
        elif st.kind == "mask_rgb":
            w = warp(raw, True, np.dtype(np.uint8), 0)
            if key.startswith("osm__"):
                rgb = _OSM_SLOT_RGB[slot_i % len(_OSM_SLOT_RGB)]
                slot_i += 1
            else:
                rgb = st.rgb
            fill_alpha = min(int(st.alpha), 120)
            rgba = _mask_rgba(
                w > 0 if w.dtype != np.uint8 else w.astype(bool),
                rgb[0], rgb[1], rgb[2], fill_alpha,
            )
            vmin, vmax = 0.0, 1.0
        else:
            continue

        rendered.append(
            RenderedLayer(key, st.title, rgba, st.cmap, vmin, vmax, st.opacity, st.show, st.kind)
        )

    return rendered, bbox_use, (dst_w, dst_h)


def _cmap_gradient_css(cmap_name: str, n: int = 9) -> str:
    cmap = _get_cmap(cmap_name)
    stops = [f"rgb({int(r*255)},{int(g*255)},{int(b*255)})" for r, g, b, _ in [cmap(i / (n - 1)) for i in range(n)]]
    return f"linear-gradient(to right, {', '.join(stops)})"


def _tile_xy(lon: float, lat: float, z: int) -> tuple[int, int]:
    n = 2**z
    x = int((lon + 180.0) / 360.0 * n)
    lat_r = math.radians(max(min(lat, 85.05112878), -85.05112878))
    y = int((1.0 - math.asinh(math.tan(lat_r)) / math.pi) / 2.0 * n)
    return x, y


def _mercator_pixel(lon: float, lat: float, z: int) -> tuple[float, float]:
    n = 2**z
    x = (lon + 180.0) / 360.0 * n * 256.0
    lat_r = math.radians(max(min(lat, 85.05112878), -85.05112878))
    y = (1.0 - math.asinh(math.tan(lat_r)) / math.pi) / 2.0 * n * 256.0
    return x, y


def _zoom_for_bbox(
    west: float, south: float, east: float, north: float, dst_w: int, max_tiles: int = 128,
) -> int:
    for z in range(19, 8, -1):
        corners = ((west, north), (east, north), (west, south), (east, south))
        xs = [_tile_xy(lon, lat, z)[0] for lon, lat in corners]
        ys = [_tile_xy(lon, lat, z)[1] for lon, lat in corners]
        nx, ny = max(xs) - min(xs) + 1, max(ys) - min(ys) + 1
        mosaic_w = nx * 256
        if nx * ny <= max_tiles and mosaic_w >= max(dst_w, 512):
            return z
    return 12


def _fetch_basemap_tile(url_template: str, z: int, x: int, y: int) -> np.ndarray:
    from PIL import Image
    from urllib.request import Request, urlopen

    url = url_template.format(z=z, y=y, x=x)
    req = Request(url, headers={"User-Agent": "PDM-proxy-debug-map/1.0 (local research)"})
    with urlopen(req, timeout=25) as resp:
        return np.asarray(Image.open(io.BytesIO(resp.read())).convert("RGB"))


def _sample_mosaic_bilinear(mosaic: np.ndarray, fx: np.ndarray, fy: np.ndarray) -> np.ndarray:
    h, w = mosaic.shape[:2]
    x0 = np.floor(fx).astype(np.int32)
    y0 = np.floor(fy).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x0 = np.clip(x0, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)
    wx = (fx - x0)[..., np.newaxis]
    wy = (fy - y0)[..., np.newaxis]
    c00 = mosaic[y0, x0].astype(np.float32)
    c01 = mosaic[y0, x1].astype(np.float32)
    c10 = mosaic[y1, x0].astype(np.float32)
    c11 = mosaic[y1, x1].astype(np.float32)
    top = c00 * (1.0 - wx) + c01 * wx
    bot = c10 * (1.0 - wx) + c11 * wx
    return np.clip(top * (1.0 - wy) + bot * wy, 0.0, 255.0).astype(np.uint8)


def _basemap_rgb_for_bbox(
    bbox_wgs84: tuple[float, float, float, float],
    dst_wh: tuple[int, int],
) -> np.ndarray:
    """OpenStreetMap standard tiles, bilinear-resampled to the overlay grid."""
    west, south, east, north = bbox_wgs84
    dst_w, dst_h = dst_wh
    z = _zoom_for_bbox(west, south, east, north, dst_w)
    corners = ((west, north), (east, north), (west, south), (east, south))
    xs = [_tile_xy(lon, lat, z)[0] for lon, lat in corners]
    ys = [_tile_xy(lon, lat, z)[1] for lon, lat in corners]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    mosaic_w = (x1 - x0 + 1) * 256
    mosaic_h = (y1 - y0 + 1) * 256
    mosaic = np.zeros((mosaic_h, mosaic_w, 3), dtype=np.uint8)
    for ty in range(y0, y1 + 1):
        for tx in range(x0, x1 + 1):
            tile = _fetch_basemap_tile(_OSM_TILES, z, tx, ty)
            row = (ty - y0) * 256
            col = (tx - x0) * 256
            mosaic[row : row + 256, col : col + 256] = tile

    n = 2**z
    ox = x0 * 256.0
    oy = y0 * 256.0
    lon = west + (np.arange(dst_w, dtype=np.float64) + 0.5) / dst_w * (east - west)
    lat = north - (np.arange(dst_h, dtype=np.float64) + 0.5) / dst_h * (north - south)
    lon_g, lat_g = np.meshgrid(lon, lat)
    lat_c = np.clip(lat_g, -85.05112878, 85.05112878)
    lat_r = np.radians(lat_c)
    mx = (lon_g + 180.0) / 360.0 * n * 256.0 - ox
    my = (1.0 - np.arcsinh(np.tan(lat_r)) / math.pi) / 2.0 * n * 256.0 - oy
    return _sample_mosaic_bilinear(mosaic, mx, my)


def _blend_rgba_on_rgb(base: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    a = overlay[..., 3:4].astype(np.float32) / 255.0
    fg = overlay[..., :3].astype(np.float32)
    bg = base.astype(np.float32)
    return np.clip(bg * (1.0 - a) + fg * a, 0, 255).astype(np.uint8)


def _draw_cams_outlines_on_ax(
    ax: Any,
    cams_cells: dict[int, dict[str, Any]],
    bbox_use: tuple[float, float, float, float],
    dst_wh: tuple[int, int],
) -> None:
    west, south, east, north = bbox_use
    dst_w, dst_h = dst_wh
    lon_span = max(east - west, 1e-12)
    lat_span = max(north - south, 1e-12)

    def to_px(lon: float, lat: float) -> tuple[float, float]:
        x = (lon - west) / lon_span * (dst_w - 1)
        y = (north - lat) / lat_span * (dst_h - 1)
        return x, y

    for row in cams_cells.values():
        b = row["cell_bounds_wgs84"]
        cw, ce, cs, cn = b["west"], b["east"], b["south"], b["north"]
        if ce < west or cw > east or cn < south or cs > north:
            continue
        ring = (to_px(cw, cs), to_px(ce, cs), to_px(ce, cn), to_px(cw, cn), to_px(cw, cs))
        xs, ys = zip(*ring)
        ax.plot(xs, ys, color="#888888", linewidth=0.6, alpha=0.55, linestyle=(0, (4, 3)))


def _bbox_footer(bbox_use: tuple[float, float, float, float], dst_wh: tuple[int, int]) -> str:
    west, south, east, north = bbox_use
    dst_w, dst_h = dst_wh
    return (
        f"WGS84 {west:.4f}, {south:.4f} → {east:.4f}, {north:.4f}  |  "
        f"{dst_w}×{dst_h} px @ {_EXPORT_DPI} dpi"
    )


def _write_fixed_images(
    out_path: Path,
    sector: str,
    layers: list[RenderedLayer],
    *,
    bbox_use: tuple[float, float, float, float],
    dst_wh: tuple[int, int],
    cams_cells: dict[int, dict[str, Any]],
) -> Path:
    import matplotlib.pyplot as plt

    from proxy.core import log

    out_dir = out_path.with_suffix("")
    out_dir.mkdir(parents=True, exist_ok=True)
    sector_label = sector.replace("_", " ")
    footer = _bbox_footer(bbox_use, dst_wh)
    dst_w, dst_h = dst_wh
    map_w_in = dst_w / _EXPORT_DPI
    map_h_in = dst_h / _EXPORT_DPI
    fig_w = map_w_in + 2 * _HPAD_IN
    fig_h = map_h_in + _TITLE_BAND_IN + _FOOTER_BAND_IN + _HPAD_IN
    ax_left = _HPAD_IN / fig_w
    ax_width = map_w_in / fig_w
    ax_bottom = _FOOTER_BAND_IN / fig_h
    ax_height = map_h_in / fig_h

    try:
        basemap = _basemap_rgb_for_bbox(bbox_use, dst_wh)
    except Exception as exc:
        log.warning(f"FIXED_IMAGE basemap fetch failed ({exc}); saving overlays only")
        basemap = None

    for layer in layers:
        if basemap is not None and basemap.shape[:2] == layer.rgba.shape[:2]:
            img = _blend_rgba_on_rgb(basemap, layer.rgba)
        else:
            img = layer.rgba[:, :, :3]

        fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
        ax = fig.add_axes([ax_left, ax_bottom, ax_width, ax_height])
        ax.imshow(img, origin="upper", aspect="equal", interpolation="nearest")
        ax.set_xlim(-0.5, dst_w - 0.5)
        ax.set_ylim(dst_h - 0.5, -0.5)
        ax.axis("off")
        _draw_cams_outlines_on_ax(ax, cams_cells, bbox_use, dst_wh)

        title_y = 1.0 - _HPAD_IN / fig_h / 2
        fig.text(
            0.5,
            title_y,
            f"{sector_label}  ·  {layer.title}",
            ha="center",
            va="top",
            fontsize=13,
            fontweight="semibold",
            color="#1a1a1a",
        )
        fig.text(0.5, _FOOTER_BAND_IN / fig_h / 2, footer, ha="center", va="center", fontsize=8, color="#555555")

        if layer.kind == "float":
            cax = fig.add_axes([ax_left + ax_width + 0.01, ax_bottom, 0.015, ax_height])
            sm = plt.cm.ScalarMappable(cmap=_get_cmap(layer.cmap), norm=plt.Normalize(layer.vmin, layer.vmax))
            cb = fig.colorbar(sm, cax=cax)
            cb.ax.tick_params(labelsize=7)
            cb.set_label(f"{layer.vmin:.4g} – {layer.vmax:.4g}", fontsize=8)

        out_file = out_dir / f"{_slug(sector)}__{_slug(layer.title)}.png"
        fig.savefig(out_file, dpi=_EXPORT_DPI, pad_inches=0)
        plt.close(fig)
    return out_dir


def _write_interactive_map(
    out_path: Path,
    sector: str,
    layers: list[RenderedLayer],
    *,
    bbox_use: tuple[float, float, float, float],
    cams_cells: dict[int, dict[str, Any]],
    dst_wh: tuple[int, int],
    legend_html: str = "",
) -> Path:
    try:
        import folium
        from folium import Element
        from folium.raster_layers import ImageOverlay
    except ImportError as exc:
        raise ImportError("Area-weights map needs folium: pip install folium") from exc

    west, south, east, north = bbox_use
    dst_w, dst_h = dst_wh
    image_bounds: list[list[float]] = [[south, west], [north, east]]
    fmap = folium.Map(
        location=[0.5 * (south + north), 0.5 * (west + east)],
        zoom_start=10,
        tiles=None,
        control_scale=True,
    )
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap", show=True).add_to(fmap)
    folium.TileLayer(tiles=_SATELLITE_TILES, attr="Esri", name="Satellite", overlay=False, show=False).add_to(fmap)

    fg_cam = folium.FeatureGroup(name="CAMS cells (outline)", show=True)
    for _cid, row in cams_cells.items():
        b = row["cell_bounds_wgs84"]
        cw, ce, cs, cn = b["west"], b["east"], b["south"], b["north"]
        if ce < west or cw > east or cn < south or cs > north:
            continue
        folium.Polygon(
            locations=[[cs, cw], [cs, ce], [cn, ce], [cn, cw], [cs, cw]],
            color="#888888", weight=1, dash_array="4 3", fill=False, opacity=0.55,
        ).add_to(fg_cam)
    fg_cam.add_to(fmap)

    scale_rows: list[str] = []
    for layer in layers:
        fg = folium.FeatureGroup(name=layer.title, show=layer.show)
        ImageOverlay(image=layer.rgba, bounds=image_bounds, opacity=layer.opacity, name=layer.key).add_to(fg)
        fg.add_to(fmap)
        if layer.kind == "float":
            grad = _cmap_gradient_css(layer.cmap)
            scale_rows.append(
                f'<div class="scale-row" style="margin:6px 0 8px;">'
                f'<div style="font-size:10px;font-weight:600;">{layer.title}</div>'
                f'<div style="height:10px;width:180px;background:{grad};border:1px solid #666;margin-top:3px;"></div>'
                f'<span style="font-size:9px;color:#333;">{layer.vmin:.4g} – {layer.vmax:.4g} ({layer.cmap})</span></div>'
            )

    folium.LayerControl(collapsed=False).add_to(fmap)
    scales_block = "".join(scale_rows)
    cmap_toggle = ""
    if scales_block:
        cmap_toggle = (
            '<button type="button" id="cmap-scales-btn" '
            'onclick="var p=document.getElementById(\'cmap-scales-panel\');'
            "var b=document.getElementById('cmap-scales-btn');"
            "var open=p.style.display==='none';"
            "p.style.display=open?'block':'none';"
            "b.textContent=open?'Hide colormap scales':'Show colormap scales';"
            '" style="margin-top:6px;padding:4px 10px;font-size:11px;cursor:pointer;">'
            "Show colormap scales</button>"
            '<div id="cmap-scales-panel" style="display:none;margin-top:6px;max-height:28vh;overflow-y:auto;">'
            f"{scales_block}</div>"
        )
    extra = f"<br>{legend_html}" if legend_html else ""
    legend = f"""
    <div style="position:fixed;bottom:20px;left:12px;z-index:9999;background:white;padding:10px 12px;
      border:1px solid #888;font-size:11px;max-width:420px;max-height:45vh;overflow-y:auto;">
      <b>{sector} area weights (DEBUG)</b><br>
      Base: <b>Satellite</b> (toggle layers in control, top-right)<br>
      Bbox WGS84: {west:.3f},{south:.3f} … {east:.3f},{north:.3f}<br>
      Display ~ {dst_w}×{dst_h} px{extra}
      <hr style="margin:6px 0;">
      {cmap_toggle}
    </div>"""
    fmap.get_root().html.add_child(Element(legend))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out_path))
    return out_path


def _emit_sector_viz(
    out_path: Path,
    sector: str,
    arrays: dict[str, np.ndarray],
    *,
    bbox_wgs84: tuple[float, float, float, float],
    transform: rasterio.Affine,
    raster_crs: Any,
    cams_cells: dict[int, dict[str, Any]],
    cell_id: np.ndarray,
    legend_html: str = "",
    style_overrides: dict[str, LayerStyle] | None = None,
) -> Path:
    layers, bbox_use, dst_wh = _prepare_layers(
        sector, arrays, bbox_wgs84=bbox_wgs84, transform=transform, raster_crs=raster_crs,
        cams_cells=cams_cells, cell_id=cell_id, style_overrides=style_overrides,
    )
    if map_type() == "FIXED_IMAGE":
        return _write_fixed_images(
            out_path, sector, layers, bbox_use=bbox_use, dst_wh=dst_wh, cams_cells=cams_cells,
        )
    return _write_interactive_map(
        out_path, sector, layers, bbox_use=bbox_use, cams_cells=cams_cells,
        dst_wh=dst_wh, legend_html=legend_html,
    )


# =============================================================================
# Per-sector writers (alphabetical)
# =============================================================================


def write_area_weights_map(
    out_html: Path,
    *,
    bbox_wgs84: tuple[float, float, float, float],
    corine_map: np.ndarray,
    population_z: np.ndarray,
    W: np.ndarray,
    cell_id: np.ndarray,
    transform: rasterio.Affine,
    raster_crs: Any,
    cams_cells: dict[int, dict[str, Any]],
    pollutant_label: str,
) -> Path:
    """A_PublicPower — CORINE, population z, W."""
    overrides = {
        "W": replace(
            SECTOR_LAYER_STYLES["A_PublicPower"]["W"],
            title=f"Weight W — {pollutant_label}",
        )
    }
    return _emit_sector_viz(
        out_html,
        "A_PublicPower",
        {"corine": corine_map, "popz": population_z, "W": W},
        bbox_wgs84=bbox_wgs84,
        transform=transform,
        raster_crs=raster_crs,
        cams_cells=cams_cells,
        cell_id=cell_id,
        style_overrides=overrides,
    )


def write_b_industry_area_weights_debug_map(
    out_html: Path,
    *,
    bbox_wgs84: tuple[float, float, float, float],
    transform: rasterio.Affine,
    raster_crs: Any,
    cams_cells: dict[int, dict[str, Any]],
    cell_id: np.ndarray,
    corine_l3_121: np.ndarray,
    corine_l3_131: np.ndarray,
    osm_by_group: dict[str, np.ndarray],
    W_by_group: dict[str, np.ndarray],
    W_pollutant: dict[str, np.ndarray],
) -> Path:
    arrays: dict[str, np.ndarray] = {"c121": corine_l3_121.astype(np.uint8), "c131": corine_l3_131.astype(np.uint8)}
    for g, a in osm_by_group.items():
        arrays[f"osm_{g}"] = np.asarray(a, dtype=np.float32)
    for g, a in W_by_group.items():
        arrays[f"W_{g}"] = np.asarray(a, dtype=np.float32)
    for pk, a in W_pollutant.items():
        arrays[f"wp_{pk}"] = np.asarray(a, dtype=np.float32)
    return _emit_sector_viz(
        out_html, "B_Industry", arrays,
        bbox_wgs84=bbox_wgs84, transform=transform, raster_crs=raster_crs,
        cams_cells=cams_cells, cell_id=cell_id,
    )


def parse_c_othercombustion_debug_viz(
    sector_cfg: dict,
    pollutant_outputs: list[str],
    *,
    model_classes: tuple[str, ...],
) -> tuple[list[str], list[str]]:
    dbg = sector_cfg.get("area_weights_debug") or {}
    classes = [str(c).strip() for c in (dbg.get("viz_classes") or []) if str(c).strip()]
    pollutants = [str(p).strip() for p in (dbg.get("viz_pollutants") or []) if str(p).strip()]
    for cls in classes:
        if cls not in model_classes:
            raise ValueError(f"area_weights_debug.viz_classes: unknown class {cls!r}")
    for p in pollutants:
        if p not in pollutant_outputs:
            raise ValueError(f"area_weights_debug.viz_pollutants: {p!r} not in sector pollutants")
    if len(classes) != 2:
        raise ValueError("area_weights_debug.viz_classes: set exactly 2 MODEL_CLASSES names")
    if len(pollutants) != 2:
        raise ValueError("area_weights_debug.viz_pollutants: set exactly 2 pollutant names")
    return classes, pollutants


def write_c_othercombustion_area_weights_debug_map(
    out_html: Path,
    *,
    sector_cfg: dict,
    pollutant_outputs: list[str],
    model_classes: tuple[str, ...],
    bbox_wgs84: tuple[float, float, float, float],
    x_build: Any,
    W_stationary_poll: dict[str, np.ndarray],
    W_offroad_poll: dict[str, np.ndarray],
    W_combined_poll: dict[str, np.ndarray],
) -> Path:
    viz_classes, viz_pollutants = parse_c_othercombustion_debug_viz(
        sector_cfg, pollutant_outputs, model_classes=model_classes,
    )
    arrays: dict[str, np.ndarray] = {
        "popz": x_build.pop_z.astype(np.float32),
        "hres": x_build.H_res_z.astype(np.float32),
        "hnres": x_build.H_nres_z.astype(np.float32),
        "hdd": x_build.Hdd_z.astype(np.float32),
        "u111": x_build.u111.astype(np.uint8),
        "u112": x_build.u112.astype(np.uint8),
        "u121": x_build.u121.astype(np.uint8),
    }
    for cls in viz_classes:
        arrays[f"S_{cls}"] = np.asarray(x_build.stock_by_class[cls], dtype=np.float32)
        arrays[f"L_{cls}"] = np.asarray(x_build.load_by_class[cls], dtype=np.float32)
    for pk in viz_pollutants:
        arrays[f"ws_{pk}"] = np.asarray(W_stationary_poll[pk], dtype=np.float32)
        arrays[f"wo_{pk}"] = np.asarray(W_offroad_poll[pk], dtype=np.float32)
        arrays[f"wc_{pk}"] = np.asarray(W_combined_poll[pk], dtype=np.float32)
    pol_txt = ", ".join(sorted(viz_pollutants))
    legend = f"Classes: {', '.join(viz_classes)} · W: stat/offroad/combined — {pol_txt}"
    return _emit_sector_viz(
        out_html, "C_Othercombustion", arrays,
        bbox_wgs84=bbox_wgs84, transform=x_build.transform, raster_crs=x_build.crs,
        cams_cells=x_build.cams_cells, cell_id=x_build.cell_id, legend_html=legend,
    )


def write_d_fugitive_area_weights_debug_map(
    out_html: Path,
    *,
    bbox_wgs84: tuple[float, float, float, float],
    transform: rasterio.Affine,
    raster_crs: Any,
    cams_cells: dict[int, dict[str, Any]],
    cell_id: np.ndarray,
    corine_l3_121: np.ndarray,
    corine_l3_123: np.ndarray,
    corine_l3_131: np.ndarray,
    population_z: np.ndarray,
    gem_coal: np.ndarray,
    gem_oil: np.ndarray,
    vnf: np.ndarray,
    osm_by_subgroup: dict[str, dict[str, np.ndarray]],
    W_by_subgroup: dict[str, np.ndarray],
    W_pollutant: dict[str, np.ndarray],
    legend_alpha_html: str,
) -> Path:
    arrays: dict[str, np.ndarray] = {
        "c121": corine_l3_121.astype(np.uint8),
        "c123": corine_l3_123.astype(np.uint8),
        "c131": corine_l3_131.astype(np.uint8),
        "popz": np.asarray(population_z, dtype=np.float32),
        "gemc": np.asarray(gem_coal, dtype=np.float32),
        "gemo": np.asarray(gem_oil, dtype=np.float32),
        "vnf": np.asarray(vnf, dtype=np.float32),
    }
    for sg in sorted(osm_by_subgroup.keys()):
        for sid in sorted((osm_by_subgroup[sg] or {}).keys()):
            arrays[f"osm__{sg}__{sid}"] = np.asarray(osm_by_subgroup[sg][sid], dtype=np.float32)
    for g, a in W_by_subgroup.items():
        arrays[f"W_{g}"] = np.asarray(a, dtype=np.float32)
    for pk, a in W_pollutant.items():
        arrays[f"wp_{pk}"] = np.asarray(a, dtype=np.float32)
    return _emit_sector_viz(
        out_html, "D_Fugitive", arrays,
        bbox_wgs84=bbox_wgs84, transform=transform, raster_crs=raster_crs,
        cams_cells=cams_cells, cell_id=cell_id, legend_html=legend_alpha_html,
    )


def write_e_solvents_area_weights_debug_map(
    out_html: Path,
    *,
    bbox_wgs84: tuple[float, float, float, float],
    transform: rasterio.Affine,
    raster_crs: Any,
    cams_cells: dict[int, dict[str, Any]],
    cell_id: np.ndarray,
    corine_household: np.ndarray,
    corine_service: np.ndarray,
    corine_industrial: np.ndarray,
    corine_transport: np.ndarray,
    osm_by_subgroup: dict[str, dict[str, np.ndarray]],
    W_by_subsector: dict[str, np.ndarray],
    W_pollutant: dict[str, np.ndarray],
    legend_alpha_html: str,
) -> Path:
    h0, w0 = cell_id.shape
    row_off, col_off, hh, ww, sub_tr, bbox_use = _bbox_crop_window(h0, w0, transform, raster_crs, bbox_wgs84)

    def crop(a: np.ndarray) -> np.ndarray:
        return _crop_array_window(a, row_off, col_off, hh, ww)

    arrays: dict[str, np.ndarray] = {
        "household": crop(corine_household).astype(np.uint8),
        "service": crop(corine_service).astype(np.uint8),
        "industrial": crop(corine_industrial).astype(np.uint8),
        "transport": crop(corine_transport).astype(np.uint8),
    }
    for sg in sorted(osm_by_subgroup.keys()):
        for sid in sorted((osm_by_subgroup[sg] or {}).keys()):
            arrays[f"osm__{sg}__{sid}"] = crop(osm_by_subgroup[sg][sid]).astype(np.float32)
    for g, a in W_by_subsector.items():
        arrays[f"W_{g}"] = crop(a).astype(np.float32)
    for pk, a in W_pollutant.items():
        arrays[f"wp_{pk}"] = crop(a).astype(np.float32)
    # Re-use emit with pre-cropped window: fake full grid via sub_tr on cropped arrays
    return _emit_sector_viz(
        out_html, "E_Solvents", arrays,
        bbox_wgs84=bbox_use, transform=sub_tr, raster_crs=raster_crs,
        cams_cells=cams_cells, cell_id=crop(cell_id), legend_html=legend_alpha_html,
    )


def write_shipping_area_weights_map(
    out_html: Path,
    *,
    bbox_wgs84: tuple[float, float, float, float],
    corine_map: np.ndarray,
    osm_raster: np.ndarray,
    emodnet_z: np.ndarray,
    W: np.ndarray,
    cell_id: np.ndarray,
    transform: rasterio.Affine,
    raster_crs: Any,
    cams_cells: dict[int, dict[str, Any]],
    pollutant_label: str,
) -> Path:
    overrides = {
        "W": replace(SECTOR_LAYER_STYLES["G_Shipping"]["W"], title=f"Weight W — {pollutant_label}"),
    }
    return _emit_sector_viz(
        out_html, "G_Shipping",
        {"corine": corine_map, "osm": osm_raster, "emod": emodnet_z, "W": W},
        bbox_wgs84=bbox_wgs84, transform=transform, raster_crs=raster_crs,
        cams_cells=cams_cells, cell_id=cell_id, style_overrides=overrides,
    )


def write_i_offroad_area_weights_debug_map(
    out_html: Path,
    *,
    bbox_wgs84: tuple[float, float, float, float],
    transform: rasterio.Affine,
    raster_crs: Any,
    cams_cells: dict[int, dict[str, Any]],
    cell_id: np.ndarray,
    corine_l3_121: np.ndarray,
    corine_l3_123: np.ndarray,
    corine_l3_124: np.ndarray,
    corine_l3_131: np.ndarray,
    osm_by_subgroup: dict[str, dict[str, np.ndarray]],
    W_by_subgroup: dict[str, np.ndarray],
    W_pollutant: dict[str, np.ndarray],
    legend_alpha_html: str,
) -> Path:
    arrays: dict[str, np.ndarray] = {
        "c121": corine_l3_121.astype(np.uint8),
        "c123": corine_l3_123.astype(np.uint8),
        "c124": corine_l3_124.astype(np.uint8),
        "c131": corine_l3_131.astype(np.uint8),
    }
    for sg in sorted(osm_by_subgroup.keys()):
        for sid in sorted((osm_by_subgroup[sg] or {}).keys()):
            arrays[f"osm__{sg}__{sid}"] = np.asarray(osm_by_subgroup[sg][sid], dtype=np.float32)
    for g, a in W_by_subgroup.items():
        arrays[f"W_{g}"] = np.asarray(a, dtype=np.float32)
    for pk, a in W_pollutant.items():
        arrays[f"wp_{pk}"] = np.asarray(a, dtype=np.float32)
    return _emit_sector_viz(
        out_html, "I_Offroad", arrays,
        bbox_wgs84=bbox_wgs84, transform=transform, raster_crs=raster_crs,
        cams_cells=cams_cells, cell_id=cell_id, legend_html=legend_alpha_html,
    )


def write_j_waste_area_weights_debug_map(
    out_html: Path,
    *,
    bbox_wgs84: tuple[float, float, float, float],
    transform: rasterio.Affine,
    raster_crs: Any,
    cams_cells: dict[int, dict[str, Any]],
    cell_id: np.ndarray,
    osm_raster: np.ndarray,
    corine_l3_132: np.ndarray,
    corine_l3_121: np.ndarray,
    imperviousness_z: np.ndarray,
    uwwtd_plants_raster: np.ndarray,
    uwwtd_agg_raster: np.ndarray,
    rural_mask: np.ndarray,
    population_z: np.ndarray,
    W_solid: np.ndarray,
    W_wastewater: np.ndarray,
    W_residual: np.ndarray,
    W_pollutant: dict[str, np.ndarray],
) -> Path:
    arrays: dict[str, np.ndarray] = {
        "osm": osm_raster,
        "c132": corine_l3_132,
        "c121": corine_l3_121,
        "impz": imperviousness_z,
        "upl": uwwtd_plants_raster,
        "uag": uwwtd_agg_raster,
        "rur": rural_mask,
        "popz": population_z.astype(np.float32),
        "Ws": W_solid,
        "Ww": W_wastewater,
        "Wr": W_residual,
    }
    for pk, a in W_pollutant.items():
        arrays[f"wp_{pk}"] = np.asarray(a, dtype=np.float32)
    return _emit_sector_viz(
        out_html, "J_Waste", arrays,
        bbox_wgs84=bbox_wgs84, transform=transform, raster_crs=raster_crs,
        cams_cells=cams_cells, cell_id=cell_id,
    )


def write_k_agriculture_area_weights_debug_map(
    out_html: Path,
    *,
    bbox_wgs84: tuple[float, float, float, float],
    transform: rasterio.Affine,
    raster_crs: Any,
    cams_cells: dict[int, dict[str, Any]],
    cell_id: np.ndarray,
    W_by_group: dict[str, np.ndarray],
    W_pollutant: dict[str, np.ndarray],
    legend_alpha_html: str = "",
) -> Path:
    arrays: dict[str, np.ndarray] = {}
    for g, a in W_by_group.items():
        arrays[f"wg_{g}"] = np.asarray(a, dtype=np.float32)
    for pk, a in W_pollutant.items():
        arrays[f"wp_{pk}"] = np.asarray(a, dtype=np.float32)
    return _emit_sector_viz(
        out_html, "K_Agriculture", arrays,
        bbox_wgs84=bbox_wgs84, transform=transform, raster_crs=raster_crs,
        cams_cells=cams_cells, cell_id=cell_id, legend_html=legend_alpha_html,
    )
