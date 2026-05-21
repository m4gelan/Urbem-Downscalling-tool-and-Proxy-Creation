from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from matplotlib import colormaps
from matplotlib import cm as mpl_cm
from rasterio.transform import array_bounds, from_bounds as affine_from_bounds
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.windows import Window, from_bounds as window_from_bounds, transform as window_transform


def _intersect_bbox_wgs84_with_raster_extent(
    bbox_wgs84: tuple[float, float, float, float],
    transform: rasterio.Affine,
    raster_crs: Any,
    height: int,
    width: int,
) -> tuple[float, float, float, float]:
    """
    Shrink ``(west, south, east, north)`` to the intersection with the raster footprint in WGS84.

    If there is no overlap, raises with a message (e.g. debug bbox in another country than the run).
    """
    west, south, east, north = bbox_wgs84
    rw, rs, re, rn = transform_bounds(
        raster_crs, "EPSG:4326", *array_bounds(height, width, transform), densify_pts=21
    )
    iw = max(float(west), float(rw))
    ie = min(float(east), float(re))
    is_ = max(float(south), float(rs))
    in_ = min(float(north), float(rn))
    if iw >= ie or is_ >= in_:
        raise ValueError(
            "DEBUG map bbox (WGS84) does not overlap this raster window. "
            f"bbox=({west:g},{south:g},{east:g},{north:g}) "
            f"raster_extent_WGS84≈({rw:.4f},{rs:.4f},{re:.4f},{rn:.4f}). "
            "Use a bbox over the same country/region as the sector run, or unset BOUNDING_BOX."
        )
    return (iw, is_, ie, in_)


def _bbox_crop_window(
    height: int,
    width: int,
    transform: rasterio.Affine,
    raster_crs: Any,
    bbox_wgs84: tuple[float, float, float, float],
) -> tuple[int, int, int, int, rasterio.Affine, tuple[float, float, float, float]]:
    """Return ``row_off, col_off, hh, ww, sub_transform, (west,south,east,north)`` for a WGS84 bbox."""
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
        raise ValueError(
            "bbox/raster intersection is empty after projection (try a denser bbox or check CRS)."
        )
    win = win.round_offsets(op="floor").round_lengths(op="ceil")
    row_off, col_off = int(win.row_off), int(win.col_off)
    hh, ww = int(win.height), int(win.width)
    if hh <= 0 or ww <= 0:
        raise ValueError("bbox does not intersect the CORINE raster window")
    sub_tr = window_transform(win, transform)
    return row_off, col_off, hh, ww, sub_tr, bbox_use


def _crop_array_window(
    arr: np.ndarray,
    row_off: int,
    col_off: int,
    hh: int,
    ww: int,
) -> np.ndarray:
    return arr[row_off : row_off + hh, col_off : col_off + ww]


def _crop_to_wgs84_bbox(
    arrs: dict[str, np.ndarray],
    transform: rasterio.Affine,
    raster_crs: Any,
    bbox_wgs84: tuple[float, float, float, float],
) -> tuple[dict[str, np.ndarray], rasterio.Affine, tuple[float, float, float, float]]:
    """Crop 2-D arrays to the window covering ``bbox_wgs84`` (west, south, east, north) in WGS84."""
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
    """
    Warp a 2-D array onto a regular lon/lat grid (EPSG:4326) covering the bbox.

    Folium ``ImageOverlay`` assumes pixels align with a WGS84 rectangle; warping here
    removes shear when the source CRS is projected (e.g. EPSG:3035).
    """
    dst_tr = affine_from_bounds(west, south, east, north, dst_w, dst_h)
    fv = fill_value
    if np.issubdtype(dst_dtype, np.integer):
        init = int(fv) if fv is not None else 0
        dst = np.full((dst_h, dst_w), init, dtype=dst_dtype)
        extra: dict[str, Any] = {}
    elif fv is not None and isinstance(fv, float) and np.isnan(fv):
        dst = np.full((dst_h, dst_w), np.nan, dtype=dst_dtype)
        extra = {"dst_nodata": np.nan}
    elif fv is not None:
        dst = np.full((dst_h, dst_w), fv, dtype=dst_dtype)
        extra = {}
    else:
        dst = np.zeros((dst_h, dst_w), dtype=dst_dtype)
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
    """Per-pixel W / max(W in same CAMS cell) for map intensity (0..1 within each cell)."""
    out_dtype = np.float32 if W.dtype == np.float32 else np.float64
    out = np.zeros_like(W, dtype=out_dtype)
    for cid in cams_cells:
        m = cell_id == int(cid)
        if not np.any(m):
            continue
        mx = float(np.max(W[m]))
        if mx > 0.0:
            out[m] = W[m] / mx
    return out


def _float_rgba(
    z: np.ndarray,
    cmap_name: str,
    valid: np.ndarray,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
) -> np.ndarray:
    try:
        cmap = colormaps[cmap_name]
    except (KeyError, AttributeError, TypeError):
        cmap = mpl_cm.get_cmap(cmap_name)
    zf = np.asarray(z, dtype=np.float64)
    t = np.zeros_like(zf, dtype=np.float64)
    if np.any(valid):
        lo = float(np.nanmin(zf[valid])) if vmin is None else float(vmin)
        hi = float(np.nanmax(zf[valid])) if vmax is None else float(vmax)
        rng = max(hi - lo, 1e-12)
        t = np.clip((zf - lo) / rng, 0.0, 1.0)
    rgba = (cmap(t) * 255.0).astype(np.uint8)
    rgba[~valid, 3] = 0
    rgba[~valid, :3] = 0
    return rgba


def _corine_gray_rgba(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(bool)
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[m, 0] = 65
    rgba[m, 1] = 65
    rgba[m, 2] = 65
    rgba[m, 3] = 235
    return rgba


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
    """
    Interactive Folium map (DEBUG): CAMS cell outlines, CORINE mask, pop z-score (viridis),
    area weight W for one pollutant (inferno, scaled per CAMS cell for display).
    """
    try:
        import folium
        from folium import Element
        from folium.raster_layers import ImageOverlay
    except ImportError as exc:
        raise ImportError("Area-weights map needs folium: pip install folium") from exc

    west, south, east, north = bbox_wgs84
    cropped, sub_tr, _ = _crop_to_wgs84_bbox(
        {
            "corine": corine_map,
            "popz": population_z,
            "W": W,
            "cid": cell_id,
        },
        transform,
        raster_crs,
        bbox_wgs84,
    )
    cor = cropped["corine"]
    pz = cropped["popz"]
    wv = cropped["W"]
    cid = cropped["cid"]

    W_disp = _W_scaled_per_cell(wv, cid.astype(np.int64), cams_cells)

    hh0, ww0 = cor.shape
    max_dim = 900
    sc = min(1.0, max_dim / max(hh0, ww0))
    dst_h = max(2, int(round(hh0 * sc)))
    dst_w = max(2, int(round(ww0 * sc)))

    cor_w = _warp_to_wgs84_grid(
        cor,
        sub_tr,
        raster_crs,
        west,
        south,
        east,
        north,
        dst_h,
        dst_w,
        Resampling.nearest,
        dst_dtype=np.dtype(np.uint8),
        fill_value=0,
    )
    pz_w = _warp_to_wgs84_grid(
        pz.astype(np.float32),
        sub_tr,
        raster_crs,
        west,
        south,
        east,
        north,
        dst_h,
        dst_w,
        Resampling.bilinear,
        dst_dtype=np.dtype(np.float32),
        fill_value=np.nan,
    )
    Ww = _warp_to_wgs84_grid(
        W_disp.astype(np.float64),
        sub_tr,
        raster_crs,
        west,
        south,
        east,
        north,
        dst_h,
        dst_w,
        Resampling.bilinear,
        dst_dtype=np.dtype(np.float64),
        fill_value=np.nan,
    )
    cid_w = _warp_to_wgs84_grid(
        cid.astype(np.int64),
        sub_tr,
        raster_crs,
        west,
        south,
        east,
        north,
        dst_h,
        dst_w,
        Resampling.nearest,
        dst_dtype=np.dtype(np.int64),
        fill_value=-1,
    )

    valid_pop = np.isfinite(pz_w)
    valid_w = cid_w >= 0

    rgba_cor = _corine_gray_rgba(cor_w)
    rgba_pz = _float_rgba(pz_w, "viridis", valid_pop)
    rgba_w = _float_rgba(Ww, "inferno", valid_w)

    image_bounds: list[list[float]] = [[south, west], [north, east]]

    center_lat = 0.5 * (south + north)
    center_lon = 0.5 * (west + east)
    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    fg_cam = folium.FeatureGroup(name="CAMS cells (outline)", show=True)
    for _cid, row in cams_cells.items():
        b = row["cell_bounds_wgs84"]
        cw, ce, cs, cn = b["west"], b["east"], b["south"], b["north"]
        if ce < west or cw > east or cn < south or cs > north:
            continue
        folium.Polygon(
            locations=[
                [cs, cw],
                [cs, ce],
                [cn, ce],
                [cn, cw],
                [cs, cw],
            ],
            color="#0066cc",
            weight=2,
            fill=False,
            opacity=0.9,
        ).add_to(fg_cam)
    fg_cam.add_to(fmap)

    # Row 0 = north (rasterio from_bounds + reproject to EPSG:4326). Do not flipud —
    # flipud inverts N–S and misaligns coast vs Folium ImageOverlay bounds.
    fg_cor = folium.FeatureGroup(name="CORINE mask (gray)", show=True)
    ImageOverlay(
        image=rgba_cor,
        bounds=image_bounds,
        opacity=0.88,
        name="corine",
    ).add_to(fg_cor)
    fg_cor.add_to(fmap)

    fg_pz = folium.FeatureGroup(name="Population z-score (viridis)", show=False)
    ImageOverlay(
        image=rgba_pz,
        bounds=image_bounds,
        opacity=0.78,
        name="popz",
    ).add_to(fg_pz)
    fg_pz.add_to(fmap)

    fg_w = folium.FeatureGroup(
        name=f"Weight W — {pollutant_label} (inferno, per-cell scale)", show=True
    )
    ImageOverlay(
        image=rgba_w,
        bounds=image_bounds,
        opacity=0.85,
        name="W",
    ).add_to(fg_w)
    fg_w.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)

    legend = f"""
    <div style="position: fixed; bottom: 20px; left: 12px; z-index: 9999;
      background: white; padding: 10px 12px; border: 1px solid #888; font-size: 11px; max-width: 320px;">
      <b>Area weights (DEBUG)</b><br>
      Pollutant: {pollutant_label}<br>
      Bbox WGS84 (warped display grid): {west:.3f},{south:.3f} … {east:.3f},{north:.3f}<br>
      Display ~ {dst_w}×{dst_h} px (EPSG:4326)<br>
      <span style="color:#06c">CAMS</span> cell edges &middot;
      <b>Gray</b> CORINE &middot;
      <b>Viridis</b> pop z &middot;
      <b>Inferno</b> W (max-normalised per CAMS cell)
    </div>
    """
    fmap.get_root().html.add_child(Element(legend))

    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out_html))
    return out_html


def _osm_blue_rgba(osm: np.ndarray) -> np.ndarray:
    m = np.asarray(osm) > 0
    rgba = np.zeros((*osm.shape, 4), dtype=np.uint8)
    rgba[m, 0] = 30
    rgba[m, 1] = 110
    rgba[m, 2] = 255
    rgba[m, 3] = 220
    return rgba


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
    """
    DEBUG Folium map (G_Shipping): CAMS outlines, OSM (blue), CORINE (gray), EMODNET z (viridis),
    W (inferno, per-CAMS-cell max scale). Same WGS84 warp + bbox as ``write_area_weights_map``.
    """
    try:
        import folium
        from folium import Element
        from folium.raster_layers import ImageOverlay
    except ImportError as exc:
        raise ImportError("Shipping area-weights map needs folium: pip install folium") from exc

    west, south, east, north = bbox_wgs84
    cropped, sub_tr, _ = _crop_to_wgs84_bbox(
        {
            "corine": corine_map,
            "osm": osm_raster,
            "emod": emodnet_z,
            "W": W,
            "cid": cell_id,
        },
        transform,
        raster_crs,
        bbox_wgs84,
    )
    cor = cropped["corine"]
    osm = cropped["osm"]
    em = cropped["emod"]
    wv = cropped["W"]
    cid = cropped["cid"]

    W_disp = _W_scaled_per_cell(wv, cid.astype(np.int64), cams_cells)

    hh0, ww0 = cor.shape
    max_dim = 900
    sc = min(1.0, max_dim / max(hh0, ww0))
    dst_h = max(2, int(round(hh0 * sc)))
    dst_w = max(2, int(round(ww0 * sc)))

    def _warp(src: np.ndarray, res: Resampling, dtype: np.dtype, fill: Any) -> np.ndarray:
        return _warp_to_wgs84_grid(
            src,
            sub_tr,
            raster_crs,
            west,
            south,
            east,
            north,
            dst_h,
            dst_w,
            res,
            dst_dtype=dtype,
            fill_value=fill,
        )

    cor_w = _warp(cor, Resampling.nearest, np.dtype(np.uint8), 0)
    osm_w = _warp(osm.astype(np.float32), Resampling.nearest, np.dtype(np.float32), 0.0)
    em_w = _warp(em.astype(np.float32), Resampling.bilinear, np.dtype(np.float32), np.nan)
    Ww = _warp(W_disp.astype(np.float64), Resampling.bilinear, np.dtype(np.float64), np.nan)
    cid_w = _warp(cid.astype(np.int64), Resampling.nearest, np.dtype(np.int64), -1)

    valid_em = np.isfinite(em_w)
    valid_w = cid_w >= 0

    rgba_osm = _osm_blue_rgba(osm_w)
    rgba_cor = _corine_gray_rgba(cor_w)
    rgba_em = _float_rgba(em_w, "viridis", valid_em)
    rgba_w = _float_rgba(Ww, "inferno", valid_w)

    image_bounds: list[list[float]] = [[south, west], [north, east]]
    center_lat = 0.5 * (south + north)
    center_lon = 0.5 * (west + east)
    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    fg_cam = folium.FeatureGroup(name="CAMS cells (outline)", show=True)
    for _cid, row in cams_cells.items():
        b = row["cell_bounds_wgs84"]
        cw, ce, cs, cn = b["west"], b["east"], b["south"], b["north"]
        if ce < west or cw > east or cn < south or cs > north:
            continue
        folium.Polygon(
            locations=[
                [cs, cw],
                [cs, ce],
                [cn, ce],
                [cn, cw],
                [cs, cw],
            ],
            color="#0066cc",
            weight=2,
            fill=False,
            opacity=0.9,
        ).add_to(fg_cam)
    fg_cam.add_to(fmap)

    fg_cor = folium.FeatureGroup(name="CORINE mask (gray)", show=True)
    ImageOverlay(
        image=rgba_cor,
        bounds=image_bounds,
        opacity=0.88,
        name="corine",
    ).add_to(fg_cor)
    fg_cor.add_to(fmap)

    fg_em = folium.FeatureGroup(name="EMODNET z-score (viridis)", show=False)
    ImageOverlay(
        image=rgba_em,
        bounds=image_bounds,
        opacity=0.78,
        name="emodnet_z",
    ).add_to(fg_em)
    fg_em.add_to(fmap)

    fg_osm = folium.FeatureGroup(name="OSM (blue)", show=True)
    ImageOverlay(
        image=rgba_osm,
        bounds=image_bounds,
        opacity=0.88,
        name="osm",
    ).add_to(fg_osm)
    fg_osm.add_to(fmap)

    fg_w = folium.FeatureGroup(
        name=f"Weight W — {pollutant_label} (inferno, per-cell scale)", show=True
    )
    ImageOverlay(
        image=rgba_w,
        bounds=image_bounds,
        opacity=0.85,
        name="W",
    ).add_to(fg_w)
    fg_w.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)

    legend = f"""
    <div style="position: fixed; bottom: 20px; left: 12px; z-index: 9999;
      background: white; padding: 10px 12px; border: 1px solid #888; font-size: 11px; max-width: 340px;">
      <b>G_Shipping area weights (DEBUG)</b><br>
      Pollutant: {pollutant_label}<br>
      Bbox WGS84: {west:.3f},{south:.3f} … {east:.3f},{north:.3f}<br>
      Display ~ {dst_w}×{dst_h} px (EPSG:4326)<br>
      <span style="color:#06c">CAMS</span> &middot; <b style="color:#1e6eff">OSM</b> &middot;
      <b>Gray</b> CORINE &middot; <b>Viridis</b> EMODNET z &middot; <b>Inferno</b> W
    </div>
    """
    fmap.get_root().html.add_child(Element(legend))

    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out_html))
    return out_html


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
    """
    DEBUG Folium map: OSM, CORINE masks (CLC 132 / 121), drivers, base W per sub-score,
    and alpha-fused W for selected pollutants (e.g. NMVOC, SOx).
    """
    try:
        import folium
        from folium import Element
        from folium.raster_layers import ImageOverlay
    except ImportError as exc:
        raise ImportError("J_Waste area-weights debug map needs folium: pip install folium") from exc

    west, south, east, north = bbox_wgs84
    keys: dict[str, np.ndarray] = {
        "osm": osm_raster,
        "c132": corine_l3_132,
        "c121": corine_l3_121,
        "impz": imperviousness_z,
        "upl": uwwtd_plants_raster,
        "uag": uwwtd_agg_raster,
        "rur": rural_mask,
        "popz": population_z.astype(np.float64),
        "Ws": W_solid,
        "Ww": W_wastewater,
        "Wr": W_residual,
        "cid": cell_id,
    }
    for k, v in W_pollutant.items():
        keys[f"wp_{k}"] = v
    cropped, sub_tr, _ = _crop_to_wgs84_bbox(keys, transform, raster_crs, bbox_wgs84)
    cid = cropped["cid"].astype(np.int64)

    hh0, ww0 = cid.shape
    max_dim = 900
    sc = min(1.0, max_dim / max(hh0, ww0))
    dst_h = max(2, int(round(hh0 * sc)))
    dst_w = max(2, int(round(ww0 * sc)))

    def _warp(src: np.ndarray, res: Resampling, dtype: np.dtype, fill: Any) -> np.ndarray:
        return _warp_to_wgs84_grid(
            src,
            sub_tr,
            raster_crs,
            west,
            south,
            east,
            north,
            dst_h,
            dst_w,
            res,
            dst_dtype=dtype,
            fill_value=fill,
        )

    osm_w = _warp(cropped["osm"].astype(np.float32), Resampling.nearest, np.dtype(np.float32), 0.0)
    c132_w = _warp(cropped["c132"].astype(np.uint8), Resampling.nearest, np.dtype(np.uint8), 0)
    c121_w = _warp(cropped["c121"].astype(np.uint8), Resampling.nearest, np.dtype(np.uint8), 0)
    imp_w = _warp(cropped["impz"].astype(np.float64), Resampling.bilinear, np.dtype(np.float64), np.nan)
    upl_w = _warp(cropped["upl"].astype(np.float32), Resampling.nearest, np.dtype(np.float32), 0.0)
    uag_w = _warp(cropped["uag"].astype(np.float32), Resampling.nearest, np.dtype(np.float32), 0.0)
    rur_w = _warp(cropped["rur"].astype(np.float32), Resampling.nearest, np.dtype(np.float32), 0.0)
    pop_w = _warp(cropped["popz"].astype(np.float64), Resampling.bilinear, np.dtype(np.float64), np.nan)
    cid_w = _warp(cid, Resampling.nearest, np.dtype(np.int64), -1)

    Ws_disp = _W_scaled_per_cell(cropped["Ws"], cid, cams_cells)
    Ww_disp = _W_scaled_per_cell(cropped["Ww"], cid, cams_cells)
    Wr_disp = _W_scaled_per_cell(cropped["Wr"], cid, cams_cells)
    Ws_w = _warp(Ws_disp.astype(np.float64), Resampling.bilinear, np.dtype(np.float64), np.nan)
    Ww_w = _warp(Ww_disp.astype(np.float64), Resampling.bilinear, np.dtype(np.float64), np.nan)
    Wr_w = _warp(Wr_disp.astype(np.float64), Resampling.bilinear, np.dtype(np.float64), np.nan)

    valid_cam = cid_w >= 0
    image_bounds: list[list[float]] = [[south, west], [north, east]]
    center_lat = 0.5 * (south + north)
    center_lon = 0.5 * (west + east)
    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    fg_cam = folium.FeatureGroup(name="CAMS cells (outline)", show=True)
    for _cid, row in cams_cells.items():
        b = row["cell_bounds_wgs84"]
        cw, ce, cs, cn = b["west"], b["east"], b["south"], b["north"]
        if ce < west or cw > east or cn < south or cs > north:
            continue
        folium.Polygon(
            locations=[[cs, cw], [cs, ce], [cn, ce], [cn, cw], [cs, cw]],
            color="#0066cc",
            weight=2,
            fill=False,
            opacity=0.9,
        ).add_to(fg_cam)
    fg_cam.add_to(fmap)

    rgba_osm = _osm_blue_rgba(osm_w)
    fg = folium.FeatureGroup(name="OSM (waste layers, blue)", show=True)
    ImageOverlay(image=rgba_osm, bounds=image_bounds, opacity=0.85, name="osm").add_to(fg)
    fg.add_to(fmap)

    rgba_c132 = _corine_gray_rgba(c132_w)
    fg = folium.FeatureGroup(name="CORINE CLC 132 (dump sites)", show=False)
    ImageOverlay(image=rgba_c132, bounds=image_bounds, opacity=0.82, name="c132").add_to(fg)
    fg.add_to(fmap)

    m121 = c121_w.astype(bool)
    rgba121 = np.zeros((*c121_w.shape, 4), dtype=np.uint8)
    rgba121[m121, 0] = 139
    rgba121[m121, 1] = 90
    rgba121[m121, 2] = 43
    rgba121[m121, 3] = 220
    fg = folium.FeatureGroup(name="CORINE CLC 121 (industrial / commercial)", show=False)
    ImageOverlay(image=rgba121, bounds=image_bounds, opacity=0.82, name="c121").add_to(fg)
    fg.add_to(fmap)

    fg = folium.FeatureGroup(name="Imperviousness z-score (0–1)", show=False)
    ImageOverlay(
        image=_float_rgba(imp_w, "cividis", np.isfinite(imp_w) & valid_cam),
        bounds=image_bounds,
        opacity=0.78,
        name="impz",
    ).add_to(fg)
    fg.add_to(fmap)

    fg = folium.FeatureGroup(name="UWWTD treatment plants (500 m buffer)", show=False)
    ImageOverlay(
        image=_float_rgba(upl_w, "Oranges", (upl_w > 0.01) & valid_cam, vmin=0.0, vmax=1.0),
        bounds=image_bounds,
        opacity=0.75,
        name="upl",
    ).add_to(fg)
    fg.add_to(fmap)

    fg = folium.FeatureGroup(name="UWWTD agglomerations (50 m buffer)", show=False)
    ImageOverlay(
        image=_float_rgba(uag_w, "Purples", (uag_w > 0.01) & valid_cam, vmin=0.0, vmax=1.0),
        bounds=image_bounds,
        opacity=0.75,
        name="uag",
    ).add_to(fg)
    fg.add_to(fmap)

    fg = folium.FeatureGroup(name="GHSL rural mask", show=False)
    ImageOverlay(
        image=_float_rgba(rur_w, "Greens", (rur_w > 0.01) & valid_cam, vmin=0.0, vmax=1.0),
        bounds=image_bounds,
        opacity=0.72,
        name="rur",
    ).add_to(fg)
    fg.add_to(fmap)

    fg = folium.FeatureGroup(name="Population z-score (0–1)", show=False)
    ImageOverlay(
        image=_float_rgba(pop_w, "viridis", np.isfinite(pop_w) & valid_cam),
        bounds=image_bounds,
        opacity=0.78,
        name="popz",
    ).add_to(fg)
    fg.add_to(fmap)

    for title, arr_w, cmap in (
        ("W solid waste (inferno, per-CAMS-cell scale)", Ws_w, "inferno"),
        ("W wastewater (inferno, per-CAMS-cell scale)", Ww_w, "inferno"),
        ("W residual (inferno, per-CAMS-cell scale)", Wr_w, "inferno"),
    ):
        fg = folium.FeatureGroup(name=title, show=False)
        ImageOverlay(
            image=_float_rgba(arr_w, cmap, valid_cam & np.isfinite(arr_w)),
            bounds=image_bounds,
            opacity=0.82,
            name=title[:16],
        ).add_to(fg)
        fg.add_to(fmap)

    for pk, _ in W_pollutant.items():
        Wd = _W_scaled_per_cell(cropped[f"wp_{pk}"], cid, cams_cells)
        Wp_w = _warp(Wd.astype(np.float64), Resampling.bilinear, np.dtype(np.float64), np.nan)
        fg = folium.FeatureGroup(
            name=f"W fused alpha — {pk.upper()} (plasma, per-CAMS-cell scale)",
            show=False,
        )
        ImageOverlay(
            image=_float_rgba(Wp_w, "plasma", valid_cam & np.isfinite(Wp_w)),
            bounds=image_bounds,
            opacity=0.85,
            name=f"Wp_{pk}",
        ).add_to(fg)
        fg.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)

    extra = ", ".join(sorted(W_pollutant.keys())) if W_pollutant else "(none)"
    legend = f"""
    <div style="position: fixed; bottom: 20px; left: 12px; z-index: 9999;
      background: white; padding: 10px 12px; border: 1px solid #888; font-size: 11px; max-width: 380px;">
      <b>J_Waste area weights (DEBUG)</b><br>
      Bbox WGS84: {west:.3f},{south:.3f} … {east:.3f},{north:.3f}<br>
      Display ~ {dst_w}×{dst_h} px (EPSG:4326)<br>
      Alpha-fused layers: {extra}
    </div>
    """
    fmap.get_root().html.add_child(Element(legend))

    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out_html))
    return out_html


def _mask_rgba_bool(mask: np.ndarray, r: int, g: int, b: int, alpha: int = 215) -> np.ndarray:
    m = np.asarray(mask).astype(bool)
    rgba = np.zeros((*m.shape, 4), dtype=np.uint8)
    rgba[m, 0] = int(r)
    rgba[m, 1] = int(g)
    rgba[m, 2] = int(b)
    rgba[m, 3] = int(alpha)
    return rgba


_INDUSTRY_OSM_GROUP_RGB: dict[str, tuple[int, int, int]] = {
    "refineries_petroleum": (30, 110, 255),
    "manufacturing_combustion_residual": (40, 170, 65),
    "mineral": (165, 95, 40),
    "chemical_metal": (140, 45, 200),
}


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
    """
    DEBUG Folium map (B_Industry): CAMS outlines, CORINE L3 121 / 131 masks, OSM evidence per inventory group,
    normalized W per group (inferno, per-CAMS-cell scale), alpha-fused W for selected pollutants (e.g. PM10, CO).
    """
    try:
        import folium
        from folium import Element
        from folium.raster_layers import ImageOverlay
    except ImportError as exc:
        raise ImportError("B_Industry area-weights debug map needs folium: pip install folium") from exc

    west, south, east, north = bbox_wgs84
    keys: dict[str, np.ndarray] = {
        "c121": corine_l3_121.astype(np.uint8),
        "c131": corine_l3_131.astype(np.uint8),
        "cid": cell_id.astype(np.int64),
    }
    for g, arr in osm_by_group.items():
        keys[f"osm_{g}"] = np.asarray(arr, dtype=np.float32)
    for g, arr in W_by_group.items():
        keys[f"W_{g}"] = np.asarray(arr, dtype=np.float64)
    for pk, arr in W_pollutant.items():
        keys[f"wp_{pk}"] = np.asarray(arr, dtype=np.float32)

    cropped, sub_tr, _ = _crop_to_wgs84_bbox(keys, transform, raster_crs, bbox_wgs84)
    cid = cropped["cid"].astype(np.int64)

    hh0, ww0 = cid.shape
    max_dim = 900
    sc = min(1.0, max_dim / max(hh0, ww0))
    dst_h = max(2, int(round(hh0 * sc)))
    dst_w = max(2, int(round(ww0 * sc)))

    def _warp(src: np.ndarray, res: Resampling, dtype: np.dtype, fill: Any) -> np.ndarray:
        return _warp_to_wgs84_grid(
            src,
            sub_tr,
            raster_crs,
            west,
            south,
            east,
            north,
            dst_h,
            dst_w,
            res,
            dst_dtype=dtype,
            fill_value=fill,
        )

    c121_w = _warp(cropped["c121"], Resampling.nearest, np.dtype(np.uint8), 0)
    c131_w = _warp(cropped["c131"], Resampling.nearest, np.dtype(np.uint8), 0)
    cid_w = _warp(cid, Resampling.nearest, np.dtype(np.int64), -1)
    valid_cam = cid_w >= 0
    image_bounds: list[list[float]] = [[south, west], [north, east]]
    center_lat = 0.5 * (south + north)
    center_lon = 0.5 * (west + east)
    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    fg_cam = folium.FeatureGroup(name="CAMS cells (outline)", show=True)
    for _cid, row in cams_cells.items():
        b = row["cell_bounds_wgs84"]
        cw, ce, cs, cn = b["west"], b["east"], b["south"], b["north"]
        if ce < west or cw > east or cn < south or cs > north:
            continue
        folium.Polygon(
            locations=[[cs, cw], [cs, ce], [cn, ce], [cn, cw], [cs, cw]],
            color="#0066cc",
            weight=2,
            fill=False,
            opacity=0.9,
        ).add_to(fg_cam)
    fg_cam.add_to(fmap)

    rgba121 = _mask_rgba_bool(c121_w, 139, 90, 43, 220)
    fg = folium.FeatureGroup(name="CORINE L3 121 (industrial / commercial)", show=True)
    ImageOverlay(image=rgba121, bounds=image_bounds, opacity=0.82, name="c121").add_to(fg)
    fg.add_to(fmap)

    rgba131 = _mask_rgba_bool(c131_w, 34, 139, 34, 215)
    fg = folium.FeatureGroup(name="CORINE L3 131 (mineral extraction)", show=False)
    ImageOverlay(image=rgba131, bounds=image_bounds, opacity=0.82, name="c131").add_to(fg)
    fg.add_to(fmap)

    for g in sorted(osm_by_group.keys()):
        raw = cropped[f"osm_{g}"].astype(np.float32)
        ow = _warp(raw, Resampling.nearest, np.dtype(np.float32), 0.0)
        rgb = _INDUSTRY_OSM_GROUP_RGB.get(g, (80, 80, 80))
        rgba = _mask_rgba_bool(ow > 0, rgb[0], rgb[1], rgb[2], 210)
        short = g[:22] + "..." if len(g) > 24 else g
        fg = folium.FeatureGroup(name=f"OSM — {short} (buffered)", show=False)
        ImageOverlay(image=rgba, bounds=image_bounds, opacity=0.84, name=f"osm_{g}").add_to(fg)
        fg.add_to(fmap)

    for g in sorted(W_by_group.keys()):
        Wd = _W_scaled_per_cell(cropped[f"W_{g}"], cid, cams_cells)
        Ww = _warp(Wd.astype(np.float64), Resampling.bilinear, np.dtype(np.float64), np.nan)
        short = g[:20] + "..." if len(g) > 22 else g
        fg = folium.FeatureGroup(
            name=f"W — {short} (inferno, per-CAMS-cell scale)",
            show=False,
        )
        ImageOverlay(
            image=_float_rgba(Ww, "inferno", valid_cam & np.isfinite(Ww)),
            bounds=image_bounds,
            opacity=0.84,
            name=f"W_{g}",
        ).add_to(fg)
        fg.add_to(fmap)

    for pk in sorted(W_pollutant.keys()):
        Wd = _W_scaled_per_cell(cropped[f"wp_{pk}"], cid, cams_cells)
        Wp_w = _warp(Wd.astype(np.float64), Resampling.bilinear, np.dtype(np.float64), np.nan)
        fg = folium.FeatureGroup(
            name=f"W alpha-fused - {pk.upper()} (plasma, per-CAMS-cell scale)",
            show=True,
        )
        ImageOverlay(
            image=_float_rgba(Wp_w, "plasma", valid_cam & np.isfinite(Wp_w)),
            bounds=image_bounds,
            opacity=0.86,
            name=f"wp_{pk}",
        ).add_to(fg)
        fg.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)

    extra = ", ".join(sorted(W_pollutant.keys())) if W_pollutant else "(none)"
    legend = f"""
    <div style="position: fixed; bottom: 20px; left: 12px; z-index: 9999;
      background: white; padding: 10px 12px; border: 1px solid #888; font-size: 11px; max-width: 400px;">
      CORINE 121 / 131 &middot; OSM per group &middot; W per group &middot; Alpha-fused: {extra}
    </div>
    """
    fmap.get_root().html.add_child(Element(legend))

    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out_html))
    return out_html


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
    """
    DEBUG Folium map (D_Fugitive): CORINE 121/123/131, every OSM slot, population z, GEM coal/oil, VNF,
    W per subgroup, alpha-fused W for selected pollutants (e.g. NMVOC, NOx).
    """
    try:
        import folium
        from folium import Element
        from folium.raster_layers import ImageOverlay
    except ImportError as exc:
        raise ImportError("D_Fugitive area-weights debug map needs folium: pip install folium") from exc

    west, south, east, north = bbox_wgs84
    keys: dict[str, np.ndarray] = {
        "c121": corine_l3_121.astype(np.uint8),
        "c123": corine_l3_123.astype(np.uint8),
        "c131": corine_l3_131.astype(np.uint8),
        "cid": cell_id.astype(np.int64),
        "popz": np.asarray(population_z, dtype=np.float64),
        "gemc": np.asarray(gem_coal, dtype=np.float32),
        "gemo": np.asarray(gem_oil, dtype=np.float32),
        "vnf": np.asarray(vnf, dtype=np.float32),
    }
    for sg in sorted(osm_by_subgroup.keys()):
        for sid in sorted((osm_by_subgroup[sg] or {}).keys()):
            k = f"osm__{sg}__{sid}"
            keys[k] = np.asarray(osm_by_subgroup[sg][sid], dtype=np.float32)
    for g, arr in W_by_subgroup.items():
        keys[f"W_{g}"] = np.asarray(arr, dtype=np.float64)
    for pk, arr in W_pollutant.items():
        keys[f"wp_{pk}"] = np.asarray(arr, dtype=np.float32)

    cropped, sub_tr, _ = _crop_to_wgs84_bbox(keys, transform, raster_crs, bbox_wgs84)
    cid = cropped["cid"].astype(np.int64)

    hh0, ww0 = cid.shape
    max_dim = 900
    sc = min(1.0, max_dim / max(hh0, ww0))
    dst_h = max(2, int(round(hh0 * sc)))
    dst_w = max(2, int(round(ww0 * sc)))

    def _warp(src: np.ndarray, res: Resampling, dtype: np.dtype, fill: Any) -> np.ndarray:
        return _warp_to_wgs84_grid(
            src,
            sub_tr,
            raster_crs,
            west,
            south,
            east,
            north,
            dst_h,
            dst_w,
            res,
            dst_dtype=dtype,
            fill_value=fill,
        )

    c121_w = _warp(cropped["c121"], Resampling.nearest, np.dtype(np.uint8), 0)
    c123_w = _warp(cropped["c123"], Resampling.nearest, np.dtype(np.uint8), 0)
    c131_w = _warp(cropped["c131"], Resampling.nearest, np.dtype(np.uint8), 0)
    cid_w = _warp(cid, Resampling.nearest, np.dtype(np.int64), -1)
    valid_cam = cid_w >= 0
    image_bounds: list[list[float]] = [[south, west], [north, east]]
    center_lat = 0.5 * (south + north)
    center_lon = 0.5 * (west + east)
    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    fg_cam = folium.FeatureGroup(name="CAMS cells (outline)", show=True)
    for _cid, row in cams_cells.items():
        b = row["cell_bounds_wgs84"]
        cw, ce, cs, cn = b["west"], b["east"], b["south"], b["north"]
        if ce < west or cw > east or cn < south or cs > north:
            continue
        folium.Polygon(
            locations=[[cs, cw], [cs, ce], [cn, ce], [cn, cw], [cs, cw]],
            color="#0066cc",
            weight=2,
            fill=False,
            opacity=0.9,
        ).add_to(fg_cam)
    fg_cam.add_to(fmap)

    rgba121 = _mask_rgba_bool(c121_w, 139, 90, 43, 220)
    fg = folium.FeatureGroup(name="CORINE L3 121 (industrial / commercial)", show=True)
    ImageOverlay(image=rgba121, bounds=image_bounds, opacity=0.82, name="c121").add_to(fg)
    fg.add_to(fmap)

    rgba123 = _mask_rgba_bool(c123_w, 30, 144, 200, 218)
    fg = folium.FeatureGroup(name="CORINE L3 123 (port areas)", show=False)
    ImageOverlay(image=rgba123, bounds=image_bounds, opacity=0.82, name="c123").add_to(fg)
    fg.add_to(fmap)

    rgba131 = _mask_rgba_bool(c131_w, 34, 139, 34, 215)
    fg = folium.FeatureGroup(name="CORINE L3 131 (mineral extraction)", show=False)
    ImageOverlay(image=rgba131, bounds=image_bounds, opacity=0.82, name="c131").add_to(fg)
    fg.add_to(fmap)

    pop_w = _warp(cropped["popz"], Resampling.bilinear, np.dtype(np.float64), np.nan)
    fg = folium.FeatureGroup(name="Population z (0-1, viridis)", show=False)
    ImageOverlay(
        image=_float_rgba(pop_w, "viridis", valid_cam & np.isfinite(pop_w)),
        bounds=image_bounds,
        opacity=0.82,
        name="popz",
    ).add_to(fg)
    fg.add_to(fmap)

    gc_w = _warp(cropped["gemc"], Resampling.nearest, np.dtype(np.float32), 0.0)
    rgba_gc = _mask_rgba_bool(gc_w > 0, 101, 67, 33, 220)
    fg = folium.FeatureGroup(name="GEM coal mines (buffered)", show=False)
    ImageOverlay(image=rgba_gc, bounds=image_bounds, opacity=0.84, name="gem_coal").add_to(fg)
    fg.add_to(fmap)

    go_w = _warp(cropped["gemo"], Resampling.nearest, np.dtype(np.float32), 0.0)
    rgba_go = _mask_rgba_bool(go_w > 0, 28, 28, 28, 215)
    fg = folium.FeatureGroup(name="GEM oil/gas extractors (buffered)", show=False)
    ImageOverlay(image=rgba_go, bounds=image_bounds, opacity=0.84, name="gem_oil").add_to(fg)
    fg.add_to(fmap)

    v_w = _warp(cropped["vnf"], Resampling.nearest, np.dtype(np.float32), 0.0)
    rgba_v = _mask_rgba_bool(v_w > 0, 255, 140, 0, 215)
    fg = folium.FeatureGroup(name="VIIRS Nightfire (VNF, buffered)", show=False)
    ImageOverlay(image=rgba_v, bounds=image_bounds, opacity=0.84, name="vnf").add_to(fg)
    fg.add_to(fmap)

    slot_i = 0
    for sg in sorted(osm_by_subgroup.keys()):
        for sid in sorted((osm_by_subgroup[sg] or {}).keys()):
            ck = f"osm__{sg}__{sid}"
            raw = cropped[ck].astype(np.float32)
            ow = _warp(raw, Resampling.nearest, np.dtype(np.float32), 0.0)
            rgb = _OSM_DF_SLOT_RGB[slot_i % len(_OSM_DF_SLOT_RGB)]
            slot_i += 1
            rgba = _mask_rgba_bool(ow > 0, rgb[0], rgb[1], rgb[2], 210)
            label = f"OSM - {sg} / {sid}"
            fg = folium.FeatureGroup(name=label[:80], show=False)
            ImageOverlay(image=rgba, bounds=image_bounds, opacity=0.84, name=ck).add_to(fg)
            fg.add_to(fmap)

    for g in sorted(W_by_subgroup.keys()):
        Wd = _W_scaled_per_cell(cropped[f"W_{g}"], cid, cams_cells)
        Ww = _warp(Wd.astype(np.float64), Resampling.bilinear, np.dtype(np.float64), np.nan)
        short = g[:22] + "..." if len(g) > 24 else g
        fg = folium.FeatureGroup(
            name=f"W - {short} (inferno, per-CAMS-cell scale)",
            show=False,
        )
        ImageOverlay(
            image=_float_rgba(Ww, "inferno", valid_cam & np.isfinite(Ww)),
            bounds=image_bounds,
            opacity=0.84,
            name=f"W_{g}",
        ).add_to(fg)
        fg.add_to(fmap)

    for pk in sorted(W_pollutant.keys()):
        Wd = _W_scaled_per_cell(cropped[f"wp_{pk}"], cid, cams_cells)
        Wp_w = _warp(Wd.astype(np.float64), Resampling.bilinear, np.dtype(np.float64), np.nan)
        fg = folium.FeatureGroup(
            name=f"W alpha-fused - {pk.upper()} (plasma, per-CAMS-cell scale)",
            show=True,
        )
        ImageOverlay(
            image=_float_rgba(Wp_w, "plasma", valid_cam & np.isfinite(Wp_w)),
            bounds=image_bounds,
            opacity=0.86,
            name=f"wp_{pk}",
        ).add_to(fg)
        fg.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)

    extra = ", ".join(sorted(W_pollutant.keys())) if W_pollutant else "(none)"
    alpha_block = legend_alpha_html if legend_alpha_html else "(no NMVOC/NOx alpha lines; check pollutant list)"
    legend = f"""
    <div style="position: fixed; bottom: 20px; left: 12px; z-index: 9999;
      background: white; padding: 10px 12px; border: 1px solid #888; font-size: 11px; max-width: 440px;">
      <b>D_Fugitive area weights (DEBUG)</b><br>
      Bbox WGS84: {west:.3f},{south:.3f} … {east:.3f},{north:.3f}<br>
      Display ~ {dst_w}×{dst_h} px (EPSG:4326)<br>
      CORINE 121/123/131 &middot; OSM slots &middot; Pop z &middot; GEM &middot; VNF &middot; W per subgroup &middot; Alpha-fused: {extra}<br>
      {alpha_block}
    </div>
    """
    fmap.get_root().html.add_child(Element(legend))

    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out_html))
    return out_html


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
    """
    DEBUG Folium map (I_Offroad): CORINE L3 121/123/124/131, every OSM slot, W per subgroup,
    alpha-fused W for selected pollutants (e.g. NMVOC, NOx).
    """
    try:
        import folium
        from folium import Element
        from folium.raster_layers import ImageOverlay
    except ImportError as exc:
        raise ImportError("I_Offroad area-weights debug map needs folium: pip install folium") from exc

    west, south, east, north = bbox_wgs84
    keys: dict[str, np.ndarray] = {
        "c121": corine_l3_121.astype(np.uint8),
        "c123": corine_l3_123.astype(np.uint8),
        "c124": corine_l3_124.astype(np.uint8),
        "c131": corine_l3_131.astype(np.uint8),
        "cid": cell_id.astype(np.int64),
    }
    for sg in sorted(osm_by_subgroup.keys()):
        for sid in sorted((osm_by_subgroup[sg] or {}).keys()):
            k = f"osm__{sg}__{sid}"
            keys[k] = np.asarray(osm_by_subgroup[sg][sid], dtype=np.float32)
    for g, arr in W_by_subgroup.items():
        keys[f"W_{g}"] = np.asarray(arr, dtype=np.float64)
    for pk, arr in W_pollutant.items():
        keys[f"wp_{pk}"] = np.asarray(arr, dtype=np.float32)

    cropped, sub_tr, _ = _crop_to_wgs84_bbox(keys, transform, raster_crs, bbox_wgs84)
    cid = cropped["cid"].astype(np.int64)

    hh0, ww0 = cid.shape
    max_dim = 900
    sc = min(1.0, max_dim / max(hh0, ww0))
    dst_h = max(2, int(round(hh0 * sc)))
    dst_w = max(2, int(round(ww0 * sc)))

    def _warp(src: np.ndarray, res: Resampling, dtype: np.dtype, fill: Any) -> np.ndarray:
        return _warp_to_wgs84_grid(
            src,
            sub_tr,
            raster_crs,
            west,
            south,
            east,
            north,
            dst_h,
            dst_w,
            res,
            dst_dtype=dtype,
            fill_value=fill,
        )

    c121_w = _warp(cropped["c121"], Resampling.nearest, np.dtype(np.uint8), 0)
    c123_w = _warp(cropped["c123"], Resampling.nearest, np.dtype(np.uint8), 0)
    c124_w = _warp(cropped["c124"], Resampling.nearest, np.dtype(np.uint8), 0)
    c131_w = _warp(cropped["c131"], Resampling.nearest, np.dtype(np.uint8), 0)
    cid_w = _warp(cid, Resampling.nearest, np.dtype(np.int64), -1)
    valid_cam = cid_w >= 0
    image_bounds: list[list[float]] = [[south, west], [north, east]]
    center_lat = 0.5 * (south + north)
    center_lon = 0.5 * (west + east)
    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    fg_cam = folium.FeatureGroup(name="CAMS cells (outline)", show=True)
    for _cid, row in cams_cells.items():
        b = row["cell_bounds_wgs84"]
        cw, ce, cs, cn = b["west"], b["east"], b["south"], b["north"]
        if ce < west or cw > east or cn < south or cs > north:
            continue
        folium.Polygon(
            locations=[[cs, cw], [cs, ce], [cn, ce], [cn, cw], [cs, cw]],
            color="#0066cc",
            weight=2,
            fill=False,
            opacity=0.9,
        ).add_to(fg_cam)
    fg_cam.add_to(fmap)

    rgba121 = _mask_rgba_bool(c121_w, 139, 90, 43, 220)
    fg = folium.FeatureGroup(name="CORINE L3 121 (industrial / commercial)", show=True)
    ImageOverlay(image=rgba121, bounds=image_bounds, opacity=0.82, name="c121").add_to(fg)
    fg.add_to(fmap)

    rgba123 = _mask_rgba_bool(c123_w, 30, 144, 200, 218)
    fg = folium.FeatureGroup(name="CORINE L3 123 (port areas)", show=False)
    ImageOverlay(image=rgba123, bounds=image_bounds, opacity=0.82, name="c123").add_to(fg)
    fg.add_to(fmap)

    rgba124 = _mask_rgba_bool(c124_w, 138, 43, 226, 218)
    fg = folium.FeatureGroup(name="CORINE L3 124 (airports)", show=False)
    ImageOverlay(image=rgba124, bounds=image_bounds, opacity=0.82, name="c124").add_to(fg)
    fg.add_to(fmap)

    rgba131 = _mask_rgba_bool(c131_w, 34, 139, 34, 215)
    fg = folium.FeatureGroup(name="CORINE L3 131 (mineral extraction)", show=False)
    ImageOverlay(image=rgba131, bounds=image_bounds, opacity=0.82, name="c131").add_to(fg)
    fg.add_to(fmap)

    slot_i = 0
    for sg in sorted(osm_by_subgroup.keys()):
        for sid in sorted((osm_by_subgroup[sg] or {}).keys()):
            ck = f"osm__{sg}__{sid}"
            raw = cropped[ck].astype(np.float32)
            ow = _warp(raw, Resampling.nearest, np.dtype(np.float32), 0.0)
            rgb = _OSM_DF_SLOT_RGB[slot_i % len(_OSM_DF_SLOT_RGB)]
            slot_i += 1
            rgba = _mask_rgba_bool(ow > 0, rgb[0], rgb[1], rgb[2], 210)
            label = f"OSM - {sg} / {sid}"
            fg = folium.FeatureGroup(name=label[:80], show=False)
            ImageOverlay(image=rgba, bounds=image_bounds, opacity=0.84, name=ck).add_to(fg)
            fg.add_to(fmap)

    for g in sorted(W_by_subgroup.keys()):
        Wd = _W_scaled_per_cell(cropped[f"W_{g}"], cid, cams_cells)
        Ww = _warp(Wd.astype(np.float64), Resampling.bilinear, np.dtype(np.float64), np.nan)
        short = g[:22] + "..." if len(g) > 24 else g
        fg = folium.FeatureGroup(
            name=f"W - {short} (inferno, per-CAMS-cell scale)",
            show=False,
        )
        ImageOverlay(
            image=_float_rgba(Ww, "inferno", valid_cam & np.isfinite(Ww)),
            bounds=image_bounds,
            opacity=0.84,
            name=f"W_{g}",
        ).add_to(fg)
        fg.add_to(fmap)

    for pk in sorted(W_pollutant.keys()):
        Wd = _W_scaled_per_cell(cropped[f"wp_{pk}"], cid, cams_cells)
        Wp_w = _warp(Wd.astype(np.float64), Resampling.bilinear, np.dtype(np.float64), np.nan)
        fg = folium.FeatureGroup(
            name=f"W alpha-fused - {pk.upper()} (plasma, per-CAMS-cell scale)",
            show=True,
        )
        ImageOverlay(
            image=_float_rgba(Wp_w, "plasma", valid_cam & np.isfinite(Wp_w)),
            bounds=image_bounds,
            opacity=0.86,
            name=f"wp_{pk}",
        ).add_to(fg)
        fg.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)

    extra = ", ".join(sorted(W_pollutant.keys())) if W_pollutant else "(none)"
    alpha_block = legend_alpha_html if legend_alpha_html else "(no NMVOC/NOx alpha lines; check pollutant list)"
    legend = f"""
    <div style="position: fixed; bottom: 20px; left: 12px; z-index: 9999;
      background: white; padding: 10px 12px; border: 1px solid #888; font-size: 11px; max-width: 440px;">
      <b>I_Offroad area weights (DEBUG)</b><br>
      Bbox WGS84: {west:.3f},{south:.3f} … {east:.3f},{north:.3f}<br>
      Display ~ {dst_w}×{dst_h} px (EPSG:4326)<br>
      CORINE 121/123/124/131 &middot; OSM slots &middot; W per subgroup &middot; Alpha-fused: {extra}<br>
      {alpha_block}
    </div>
    """
    fmap.get_root().html.add_child(Element(legend))

    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out_html))
    return out_html


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
    """
    DEBUG Folium map (E_Solvents): CORINE masks per activity archetype, OSM slots,
    W per subsector, alpha-fused W for selected pollutants (e.g. NMVOC, NOx).
    """
    try:
        import folium
        from folium import Element
        from folium.raster_layers import ImageOverlay
    except ImportError as exc:
        raise ImportError("E_Solvents area-weights debug map needs folium: pip install folium") from exc

    west, south, east, north = bbox_wgs84
    h0, w0 = cell_id.shape
    row_off, col_off, hh0, ww0, sub_tr, bbox_use = _bbox_crop_window(
        h0, w0, transform, raster_crs, bbox_wgs84
    )
    west, south, east, north = bbox_use

    cid = _crop_array_window(cell_id, row_off, col_off, hh0, ww0).astype(np.int32, copy=False)

    def _crop_layer(arr: np.ndarray) -> np.ndarray:
        return _crop_array_window(arr, row_off, col_off, hh0, ww0)
    max_dim = 900
    sc = min(1.0, max_dim / max(hh0, ww0))
    dst_h = max(2, int(round(hh0 * sc)))
    dst_w = max(2, int(round(ww0 * sc)))

    def _warp(src: np.ndarray, res: Resampling, dtype: np.dtype, fill: Any) -> np.ndarray:
        return _warp_to_wgs84_grid(
            src,
            sub_tr,
            raster_crs,
            west,
            south,
            east,
            north,
            dst_h,
            dst_w,
            res,
            dst_dtype=dtype,
            fill_value=fill,
        )

    cid_w = _warp(cid, Resampling.nearest, np.dtype(np.int32), -1)
    valid_cam = cid_w >= 0
    image_bounds: list[list[float]] = [[south, west], [north, east]]
    center_lat = 0.5 * (south + north)
    center_lon = 0.5 * (west + east)
    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    fg_cam = folium.FeatureGroup(name="CAMS cells (outline)", show=True)
    for _cid, row in cams_cells.items():
        b = row["cell_bounds_wgs84"]
        cw, ce, cs, cn = b["west"], b["east"], b["south"], b["north"]
        if ce < west or cw > east or cn < south or cs > north:
            continue
        folium.Polygon(
            locations=[[cs, cw], [cs, ce], [cn, ce], [cn, cw], [cs, cw]],
            color="#0066cc",
            weight=2,
            fill=False,
            opacity=0.9,
        ).add_to(fg_cam)
    fg_cam.add_to(fmap)

    for arr, style_key, label in (
        (corine_household, "household", "CORINE household (111, 112)"),
        (corine_service, "service", "CORINE service (121)"),
        (corine_industrial, "industrial", "CORINE industrial (121, 133)"),
        (corine_transport, "transport", "CORINE transport (122)"),
    ):
        cwarp = _warp(
            _crop_layer(arr).astype(np.uint8, copy=False),
            Resampling.nearest,
            np.dtype(np.uint8),
            0,
        )
        r, g, b, a = _CORINE_SOLVENTS_RGB[style_key]
        rgba = _mask_rgba_bool(cwarp, r, g, b, a)
        fg = folium.FeatureGroup(name=label, show=(style_key == "household"))
        ImageOverlay(image=rgba, bounds=image_bounds, opacity=0.82, name=style_key).add_to(fg)
        fg.add_to(fmap)

    slot_i = 0
    for sg in sorted(osm_by_subgroup.keys()):
        for sid in sorted((osm_by_subgroup[sg] or {}).keys()):
            ck = f"osm__{sg}__{sid}"
            raw = _crop_layer(osm_by_subgroup[sg][sid]).astype(np.float32, copy=False)
            ow = _warp(raw, Resampling.nearest, np.dtype(np.float32), 0.0)
            rgb = _OSM_DF_SLOT_RGB[slot_i % len(_OSM_DF_SLOT_RGB)]
            slot_i += 1
            rgba = _mask_rgba_bool(ow > 0, rgb[0], rgb[1], rgb[2], 210)
            label = f"OSM - {sg} / {sid}"
            fg = folium.FeatureGroup(name=label[:80], show=False)
            ImageOverlay(image=rgba, bounds=image_bounds, opacity=0.84, name=ck).add_to(fg)
            fg.add_to(fmap)

    for g in sorted(W_by_subsector.keys()):
        Wd = _W_scaled_per_cell(_crop_layer(W_by_subsector[g]), cid, cams_cells)
        Ww = _warp(Wd, Resampling.bilinear, np.dtype(np.float32), np.nan)
        short = g[:22] + "..." if len(g) > 24 else g
        fg = folium.FeatureGroup(
            name=f"W - {short} (inferno, per-CAMS-cell scale)",
            show=False,
        )
        ImageOverlay(
            image=_float_rgba(Ww, "inferno", valid_cam & np.isfinite(Ww)),
            bounds=image_bounds,
            opacity=0.84,
            name=f"W_{g}",
        ).add_to(fg)
        fg.add_to(fmap)

    for pk in sorted(W_pollutant.keys()):
        Wd = _W_scaled_per_cell(_crop_layer(W_pollutant[pk]), cid, cams_cells)
        Wp_w = _warp(Wd, Resampling.bilinear, np.dtype(np.float32), np.nan)
        fg = folium.FeatureGroup(
            name=f"W alpha-fused - {pk.upper()} (plasma, per-CAMS-cell scale)",
            show=True,
        )
        ImageOverlay(
            image=_float_rgba(Wp_w, "plasma", valid_cam & np.isfinite(Wp_w)),
            bounds=image_bounds,
            opacity=0.86,
            name=f"wp_{pk}",
        ).add_to(fg)
        fg.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)

    extra = ", ".join(sorted(W_pollutant.keys())) if W_pollutant else "(none)"
    alpha_block = legend_alpha_html if legend_alpha_html else "(no NMVOC/NOx alpha lines; check pollutant list)"
    legend = f"""
    <div style="position: fixed; bottom: 20px; left: 12px; z-index: 9999;
      background: white; padding: 10px 12px; border: 1px solid #888; font-size: 11px; max-width: 440px;">
      <b>E_Solvents area weights (DEBUG)</b><br>
      Bbox WGS84: {west:.3f},{south:.3f} … {east:.3f},{north:.3f}<br>
      Display ~ {dst_w}×{dst_h} px (EPSG:4326)<br>
      CORINE household/service/industrial/transport &middot; OSM &middot; W per subsector &middot; Alpha-fused: {extra}<br>
      {alpha_block}
    </div>
    """
    fmap.get_root().html.add_child(Element(legend))

    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out_html))
    return out_html


_CORINE_OTHERCOMB_RGB: dict[str, tuple[int, int, int, int]] = {
    "u111": (205, 92, 92, 220),
    "u112": (210, 105, 30, 220),
    "u121": (30, 144, 200, 218),
}


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
    """Folium DEBUG map: inputs, S/L for 2 classes, W stationary/offroad/combined per pollutant."""
    transform = x_build.transform
    raster_crs = x_build.crs
    cams_cells = x_build.cams_cells
    cell_id = x_build.cell_id
    pop_z = x_build.pop_z
    H_res_z = x_build.H_res_z
    H_nres_z = x_build.H_nres_z
    Hdd_z = x_build.Hdd_z
    u111 = x_build.u111
    u112 = x_build.u112
    u121 = x_build.u121
    stock_by_class = x_build.stock_by_class
    load_by_class = x_build.load_by_class

    viz_classes, viz_pollutants = parse_c_othercombustion_debug_viz(
        sector_cfg, pollutant_outputs, model_classes=model_classes
    )

    try:
        import folium
        from folium import Element
        from folium.raster_layers import ImageOverlay
    except ImportError as exc:
        raise ImportError("C_OtherCombustion area-weights debug map needs folium: pip install folium") from exc

    keys: dict[str, np.ndarray] = {
        "popz": pop_z.astype(np.float32),
        "hres": H_res_z.astype(np.float32),
        "hnres": H_nres_z.astype(np.float32),
        "hdd": Hdd_z.astype(np.float32),
        "u111": u111.astype(np.uint8),
        "u112": u112.astype(np.uint8),
        "u121": u121.astype(np.uint8),
        "cid": cell_id.astype(np.int64),
    }
    for cls in viz_classes:
        keys[f"S_{cls}"] = np.asarray(stock_by_class[cls], dtype=np.float32)
        keys[f"L_{cls}"] = np.asarray(load_by_class[cls], dtype=np.float32)
    for pk in viz_pollutants:
        keys[f"ws_{pk}"] = np.asarray(W_stationary_poll[pk], dtype=np.float32)
        keys[f"wo_{pk}"] = np.asarray(W_offroad_poll[pk], dtype=np.float32)
        keys[f"wc_{pk}"] = np.asarray(W_combined_poll[pk], dtype=np.float32)

    cropped, sub_tr, bbox_use = _crop_to_wgs84_bbox(keys, transform, raster_crs, bbox_wgs84)
    west, south, east, north = bbox_use
    cid = cropped["cid"].astype(np.int64)

    hh0, ww0 = cid.shape
    sc = min(1.0, 900 / max(hh0, ww0))
    dst_h = max(2, int(round(hh0 * sc)))
    dst_w = max(2, int(round(ww0 * sc)))

    def _warp(src: np.ndarray, res: Resampling, dtype: np.dtype, fill: Any) -> np.ndarray:
        return _warp_to_wgs84_grid(
            src, sub_tr, raster_crs, west, south, east, north,
            dst_h, dst_w, res, dst_dtype=dtype, fill_value=fill,
        )

    cid_w = _warp(cid, Resampling.nearest, np.dtype(np.int64), -1)
    valid_cam = cid_w >= 0
    image_bounds: list[list[float]] = [[south, west], [north, east]]
    fmap = folium.Map(
        location=[0.5 * (south + north), 0.5 * (west + east)],
        zoom_start=10,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    fg_cam = folium.FeatureGroup(name="CAMS cells (outline)", show=True)
    for _cid, row in cams_cells.items():
        b = row["cell_bounds_wgs84"]
        cw, ce, cs, cn = b["west"], b["east"], b["south"], b["north"]
        if ce < west or cw > east or cn < south or cs > north:
            continue
        folium.Polygon(
            locations=[[cs, cw], [cs, ce], [cn, ce], [cn, cw], [cs, cw]],
            color="#0066cc",
            weight=2,
            fill=False,
            opacity=0.9,
        ).add_to(fg_cam)
    fg_cam.add_to(fmap)

    for ck, cmap, label, show in (
        ("popz", "viridis", "Population z-score", True),
        ("hres", "viridis", "Hotmaps H_res z-score", False),
        ("hnres", "viridis", "Hotmaps H_nres z-score", False),
        ("hdd", "viridis", "HDD z-score", False),
    ):
        zw = _warp(cropped[ck], Resampling.bilinear, np.dtype(np.float32), np.nan)
        fg = folium.FeatureGroup(name=label, show=show)
        ImageOverlay(
            image=_float_rgba(zw, cmap, valid_cam & np.isfinite(zw)),
            bounds=image_bounds,
            opacity=0.82,
            name=ck,
        ).add_to(fg)
        fg.add_to(fmap)

    for ck, sk, label in (
        ("u111", "u111", "CORINE u111 (continuous urban)"),
        ("u112", "u112", "CORINE u112 (discontinuous urban)"),
        ("u121", "u121", "CORINE u121 (industrial / commercial)"),
    ):
        cwarp = _warp(cropped[ck], Resampling.nearest, np.dtype(np.uint8), 0)
        r, g, b, a = _CORINE_OTHERCOMB_RGB[sk]
        fg = folium.FeatureGroup(name=label, show=(sk == "u111"))
        ImageOverlay(
            image=_mask_rgba_bool(cwarp > 0, r, g, b, a),
            bounds=image_bounds,
            opacity=0.82,
            name=ck,
        ).add_to(fg)
        fg.add_to(fmap)

    for cls in viz_classes:
        for kind, cmap in (("S", "YlOrBr"), ("L", "PuBu")):
            k = f"{kind}_{cls}"
            lw = _warp(cropped[k].astype(np.float32), Resampling.bilinear, np.dtype(np.float32), np.nan)
            name = f"{kind} stock — {cls}" if kind == "S" else f"{kind} load — {cls}"
            fg = folium.FeatureGroup(name=name, show=False)
            ImageOverlay(
                image=_float_rgba(lw, cmap, valid_cam & np.isfinite(lw)),
                bounds=image_bounds,
                opacity=0.84,
                name=k,
            ).add_to(fg)
            fg.add_to(fmap)

    for pk in sorted(viz_pollutants):
        for prefix, branch, cmap, show in (
            ("ws", "stationary", "inferno", False),
            ("wo", "offroad", "cividis", False),
            ("wc", "combined", "plasma", True),
        ):
            Wd = _W_scaled_per_cell(cropped[f"{prefix}_{pk}"], cid, cams_cells)
            Wp_w = _warp(Wd.astype(np.float64), Resampling.bilinear, np.dtype(np.float64), np.nan)
            fg = folium.FeatureGroup(name=f"W {branch} — {pk} (per-CAMS-cell)", show=show)
            ImageOverlay(
                image=_float_rgba(Wp_w, cmap, valid_cam & np.isfinite(Wp_w)),
                bounds=image_bounds,
                opacity=0.86,
                name=f"{prefix}_{pk}",
            ).add_to(fg)
            fg.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    pol_txt = ", ".join(sorted(viz_pollutants))
    legend = f"""
    <div style="position: fixed; bottom: 20px; left: 12px; z-index: 9999;
      background: white; padding: 10px 12px; border: 1px solid #888; font-size: 11px; max-width: 440px;">
      <b>C_OtherCombustion area weights (DEBUG)</b><br>
      Bbox: {west:.3f},{south:.3f} … {east:.3f},{north:.3f} · ~{dst_w}×{dst_h} px<br>
      Z: pop, H_res, H_nres, HDD · CORINE u111/u112/u121<br>
      Classes: {", ".join(viz_classes)} (S/L) · W per poll: stat / offroad / combined — {pol_txt}
    </div>
    """
    fmap.get_root().html.add_child(Element(legend))

    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out_html))
    return out_html


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
    """DEBUG Folium map: per-group CAMS-normalized W layers + alpha-fused W (e.g. NH3, NMVOC)."""
    try:
        import folium
        from folium import Element
        from folium.raster_layers import ImageOverlay
    except ImportError as exc:
        raise ImportError("K_Agriculture area-weights debug map needs folium: pip install folium") from exc

    west, south, east, north = bbox_wgs84
    keys: dict[str, np.ndarray] = {"cid": cell_id}
    for gname, arr in W_by_group.items():
        keys[f"wg_{gname}"] = arr
    for pk, arr in W_pollutant.items():
        keys[f"wp_{pk}"] = arr
    cropped, sub_tr, _ = _crop_to_wgs84_bbox(keys, transform, raster_crs, bbox_wgs84)
    cid = cropped["cid"].astype(np.int64)

    hh0, ww0 = cid.shape
    max_dim = 900
    sc = min(1.0, max_dim / max(hh0, ww0))
    dst_h = max(2, int(round(hh0 * sc)))
    dst_w = max(2, int(round(ww0 * sc)))

    def _warp(src: np.ndarray, res: Resampling, dtype: np.dtype, fill: Any) -> np.ndarray:
        return _warp_to_wgs84_grid(
            src,
            sub_tr,
            raster_crs,
            west,
            south,
            east,
            north,
            dst_h,
            dst_w,
            res,
            dst_dtype=dtype,
            fill_value=fill,
        )

    cid_w = _warp(cid, Resampling.nearest, np.dtype(np.int64), -1)
    valid_cam = cid_w >= 0
    image_bounds: list[list[float]] = [[south, west], [north, east]]
    center_lat = 0.5 * (south + north)
    center_lon = 0.5 * (west + east)
    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    fg_cam = folium.FeatureGroup(name="CAMS cells (outline)", show=True)
    for _cid, row in cams_cells.items():
        b = row["cell_bounds_wgs84"]
        cw, ce, cs, cn = b["west"], b["east"], b["south"], b["north"]
        if ce < west or cw > east or cn < south or cs > north:
            continue
        folium.Polygon(
            locations=[[cs, cw], [cs, ce], [cn, ce], [cn, cw], [cs, cw]],
            color="#0066cc",
            weight=2,
            fill=False,
            opacity=0.9,
        ).add_to(fg_cam)
    fg_cam.add_to(fmap)

    for gname in W_by_group:
        Wd = _W_scaled_per_cell(cropped[f"wg_{gname}"], cid, cams_cells)
        Ww = _warp(Wd.astype(np.float32), Resampling.bilinear, np.dtype(np.float32), np.nan)
        fg = folium.FeatureGroup(
            name=f"W {gname.replace('_', ' ')} (inferno, per-CAMS-cell)",
            show=gname == "livestock",
        )
        ImageOverlay(
            image=_float_rgba(Ww, "inferno", valid_cam & np.isfinite(Ww)),
            bounds=image_bounds,
            opacity=0.82,
            name=f"wg_{gname}",
        ).add_to(fg)
        fg.add_to(fmap)

    for pk in W_pollutant:
        Wd = _W_scaled_per_cell(cropped[f"wp_{pk}"], cid, cams_cells)
        Ww = _warp(Wd.astype(np.float32), Resampling.bilinear, np.dtype(np.float32), np.nan)
        fg = folium.FeatureGroup(
            name=f"W fused α — {pk.upper()} (plasma, per-CAMS-cell)",
            show=pk == "nh3",
        )
        ImageOverlay(
            image=_float_rgba(Ww, "plasma", valid_cam & np.isfinite(Ww)),
            bounds=image_bounds,
            opacity=0.85,
            name=f"wp_{pk}",
        ).add_to(fg)
        fg.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)

    alpha_line = f"<br>{legend_alpha_html}" if legend_alpha_html else ""
    legend = f"""
    <div style="position: fixed; bottom: 20px; left: 12px; z-index: 9999;
      background: white; padding: 10px 12px; border: 1px solid #888; font-size: 11px; max-width: 420px;">
      <b>K_Agriculture area weights (DEBUG)</b><br>
      Bbox WGS84: {west:.3f},{south:.3f} … {east:.3f},{north:.3f}<br>
      Display ~ {dst_w}×{dst_h} px (EPSG:4326){alpha_line}
    </div>
    """
    fmap.get_root().html.add_child(Element(legend))

    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out_html))
    return out_html

