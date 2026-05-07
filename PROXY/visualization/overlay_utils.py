"""Raster warp to a WGS84 display grid and scalar-to-RGBA (same role as legacy visualize_proxy_weights)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _mask_nodata_unless_zero_float(
    arr: np.ndarray, nodata: float | None
) -> tuple[np.ndarray, float | None]:
    """
    Apply nodata -> NaN for display/reproject, except when nodata is 0.0 on a float
    array: 0 is a valid weight, but many GeoTIFFs incorrectly tag nodata=0.0.
    """
    if nodata is None:
        return arr, None
    ndv = float(nodata)
    a = np.asarray(arr, dtype=np.float64)
    if np.issubdtype(a.dtype, np.floating) and abs(ndv) < 1e-30:
        return a, None
    return np.where(a == ndv, np.nan, a), nodata


def sample_cmap_hex(cmap_name: str, n: int = 11) -> list[str]:
    """Sample ``n`` colours from a matplotlib colormap as #rrggbb."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import colormaps

    cmap = colormaps[cmap_name]
    if n < 2:
        n = 2
    out: list[str] = []
    for i in range(n):
        t = i / (n - 1)
        r, g, b, *_ = cmap(t)
        out.append(
            f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
        )
    return out


def scalar_to_rgba(
    z: np.ndarray,
    *,
    colour_mode: str,
    cmap_name: str,
    hide_zero: bool,
    nodata_val: float | None,
    z_precomputed_01: np.ndarray | None = None,
    valid_precomputed: np.ndarray | None = None,
) -> np.ndarray:
    """Return uint8 (H, W, 4) RGBA for Folium ImageOverlay."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import colormaps

    z = np.asarray(z, dtype=np.float64)
    h, w = z.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    if z_precomputed_01 is not None:
        t_full = np.asarray(z_precomputed_01, dtype=np.float64)
        valid = np.asarray(valid_precomputed, dtype=bool)
        if hide_zero:
            valid = valid & (z > 0)
        if nodata_val is not None:
            valid = valid & np.isfinite(z) & (z != float(nodata_val))
    else:
        finite = np.isfinite(z)
        if nodata_val is not None:
            finite = finite & (z != float(nodata_val))
        if hide_zero:
            valid = finite & (z > 0)
        else:
            valid = finite
        if not np.any(valid):
            return rgba
        vals = z[valid]

        if colour_mode == "log":
            vals = np.log10(np.maximum(vals, 1e-18))
        elif colour_mode == "percentile":
            lo = float(np.percentile(vals, 2.0))
            hi = float(np.percentile(vals, 98.0))
            if lo >= hi:
                hi = lo + 1e-9
            vals = np.clip((vals - lo) / (hi - lo), 0.0, 1.0)

        lo, hi = float(np.min(vals)), float(np.max(vals))
        if lo >= hi:
            hi = lo + 1e-6
        t = np.clip((vals - lo) / (hi - lo), 0.0, 1.0)
        cmap = colormaps[cmap_name]
        c = cmap(t)
        rgba[valid, 0] = (np.clip(c[:, 0], 0.0, 1.0) * 255).astype(np.uint8)
        rgba[valid, 1] = (np.clip(c[:, 1], 0.0, 1.0) * 255).astype(np.uint8)
        rgba[valid, 2] = (np.clip(c[:, 2], 0.0, 1.0) * 255).astype(np.uint8)
        rgba[valid, 3] = 255
        return rgba

    if not np.any(valid):
        return rgba

    cmap = colormaps[cmap_name]
    t = np.clip(t_full[valid], 0.0, 1.0)
    c = cmap(t)
    rgba[valid, 0] = (np.clip(c[:, 0], 0.0, 1.0) * 255).astype(np.uint8)
    rgba[valid, 1] = (np.clip(c[:, 1], 0.0, 1.0) * 255).astype(np.uint8)
    rgba[valid, 2] = (np.clip(c[:, 2], 0.0, 1.0) * 255).astype(np.uint8)
    rgba[valid, 3] = 255
    return rgba


def read_band_warped_to_wgs84_grid(
    tif_path: Path,
    *,
    band: int,
    west: float,
    south: float,
    east: float,
    north: float,
    width: int,
    height: int,
    resampling: str = "bilinear",
) -> tuple[np.ndarray, float | None]:
    """Sample one band onto a WGS84 pixel grid (west..east, south..north, width x height)."""
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.warp import Resampling, reproject, transform_bounds
    from rasterio.windows import Window, from_bounds as win_from_bounds

    dst_t = from_bounds(west, south, east, north, width, height)
    dst = np.full((height, width), np.nan, dtype=np.float64)
    rs = getattr(Resampling, str(resampling))

    with rasterio.open(tif_path) as src:
        nodata = src.nodata
        if int(band) < 1 or int(band) > int(src.count):
            raise ValueError(f"Band {band} invalid for {tif_path} (count={src.count})")
        if src.crs is None:
            raise ValueError(f"Raster has no CRS: {tif_path}")
        lb, bb, rb, tb = transform_bounds(
            "EPSG:4326", src.crs, west, south, east, north, densify_pts=21
        )
        win = win_from_bounds(lb, bb, rb, tb, transform=src.transform)
        win = win.round_lengths().intersection(Window(0, 0, src.width, src.height))
        if win.width < 1 or win.height < 1:
            return dst, nodata
        arr = src.read(int(band), window=win).astype(np.float64)
        wt = src.window_transform(win)
        if nodata is not None:
            arr = np.where(arr == float(nodata), np.nan, arr)
        reproject(
            source=arr,
            destination=dst,
            src_transform=wt,
            src_crs=src.crs,
            dst_transform=dst_t,
            dst_crs="EPSG:4326",
            src_nodata=np.nan,
            dst_nodata=np.nan,
            resampling=rs,
        )
    return dst, nodata


def weight_display_bounds_from_raster(
    tif_path: Path, *, pad_deg: float
) -> tuple[float, float, float, float]:
    import rasterio
    from rasterio.warp import transform_bounds

    with rasterio.open(tif_path) as src:
        if src.crs is None:
            raise ValueError(f"Raster has no CRS: {tif_path}")
        w, s, e, n = transform_bounds(
            src.crs, "EPSG:4326", *src.bounds, densify_pts=21
        )
    return (
        w - pad_deg,
        s - pad_deg,
        e + pad_deg,
        n + pad_deg,
    )


def read_weight_corine_population_via_weight_grid_wgs84(
    weight_path: Path,
    *,
    corine_path: Path,
    corine_band: int,
    population_path: Path,
    population_band: int,
    west: float,
    south: float,
    east: float,
    north: float,
    display_width: int,
    display_height: int,
    weight_band: int = 1,
) -> dict[str, Any]:
    """
    Resample CORINE and population onto the **weight GeoTIFF grid** (same CRS and transform),
    then warp all three to the same WGS84 display grid.

    This keeps CORINE / population / weights **pixel-aligned** in native space before the
    final geographic warp (avoids independent window+reproject drift between rasters).
    """
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.warp import Resampling, reproject

    dst_t = from_bounds(west, south, east, north, display_width, display_height)
    gh, gw = int(display_height), int(display_width)

    with rasterio.open(weight_path) as wsrc:
        w_tr = wsrc.transform
        w_crs = wsrc.crs
        if w_crs is None:
            raise ValueError(f"Weight raster has no CRS: {weight_path}")
        h_ref, w_ref = int(wsrc.height), int(wsrc.width)
        wb = int(weight_band)
        if wb < 1 or wb > int(wsrc.count):
            raise ValueError(f"Weight band {wb} invalid for {weight_path} (count={wsrc.count})")
        w_native = wsrc.read(wb).astype(np.float64)
        w_nodata = wsrc.nodata
        w_native, w_nodata = _mask_nodata_unless_zero_float(w_native, w_nodata)

    cor_native = np.full((h_ref, w_ref), np.nan, dtype=np.float64)
    with rasterio.open(corine_path) as csrc:
        if corine_band < 1 or corine_band > int(csrc.count):
            raise ValueError(f"CORINE band {corine_band} invalid for {corine_path}")
        if csrc.crs is None:
            raise ValueError(f"CORINE raster has no CRS: {corine_path}")
        c_nd = csrc.nodata
        reproject(
            source=rasterio.band(csrc, int(corine_band)),
            destination=cor_native,
            src_transform=csrc.transform,
            src_crs=csrc.crs,
            dst_transform=w_tr,
            dst_crs=w_crs,
            resampling=Resampling.nearest,
            src_nodata=np.nan if c_nd is None else float(c_nd),
            dst_nodata=np.nan,
        )

    pop_native = np.full((h_ref, w_ref), np.nan, dtype=np.float64)
    with rasterio.open(population_path) as psrc:
        if psrc.crs is None:
            raise ValueError(f"Population raster has no CRS: {population_path}")
        p_nd = psrc.nodata
        reproject(
            source=rasterio.band(psrc, int(population_band)),
            destination=pop_native,
            src_transform=psrc.transform,
            src_crs=psrc.crs,
            dst_transform=w_tr,
            dst_crs=w_crs,
            resampling=Resampling.bilinear,
            src_nodata=np.nan if p_nd is None else float(p_nd),
            dst_nodata=np.nan,
        )

    def _to_wgs84(src_arr: np.ndarray, resampling: str) -> np.ndarray:
        out = np.full((gh, gw), np.nan, dtype=np.float64)
        rs = getattr(Resampling, str(resampling))
        reproject(
            source=src_arr,
            destination=out,
            src_transform=w_tr,
            src_crs=w_crs,
            dst_transform=dst_t,
            dst_crs="EPSG:4326",
            resampling=rs,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )
        return out

    return {
        "weight_wgs84": _to_wgs84(w_native, "bilinear"),
        "corine_wgs84": _to_wgs84(cor_native, "nearest"),
        "population_wgs84": _to_wgs84(pop_native, "bilinear"),
        "weight_nodata": w_nodata,
        "corine_nodata": c_nd,
        "population_nodata": p_nd,
        "weight_crs": w_crs.to_string() if w_crs is not None else "",
        "weight_shape_native": (h_ref, w_ref),
    }


def read_weight_wgs84_only(
    weight_path: Path,
    *,
    west: float,
    south: float,
    east: float,
    north: float,
    display_width: int,
    display_height: int,
    weight_band: int = 1,
) -> dict[str, Any]:
    """Warp one weight band to WGS84 display bounds (no CORINE / population reads)."""
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.warp import Resampling, reproject

    dst_t = from_bounds(west, south, east, north, display_width, display_height)
    gh, gw = int(display_height), int(display_width)

    with rasterio.open(weight_path) as wsrc:
        w_tr = wsrc.transform
        w_crs = wsrc.crs
        if w_crs is None:
            raise ValueError(f"Weight raster has no CRS: {weight_path}")
        h_ref, w_ref = int(wsrc.height), int(wsrc.width)
        wb = int(weight_band)
        if wb < 1 or wb > int(wsrc.count):
            raise ValueError(f"Weight band {wb} invalid for {weight_path} (count={wsrc.count})")
        w_native = wsrc.read(wb).astype(np.float64)
        w_nodata = wsrc.nodata
        w_native, w_nodata = _mask_nodata_unless_zero_float(w_native, w_nodata)

    def _to_wgs84(src_arr: np.ndarray, resampling: str) -> np.ndarray:
        out = np.full((gh, gw), np.nan, dtype=np.float64)
        rs = getattr(Resampling, str(resampling))
        reproject(
            source=src_arr,
            destination=out,
            src_transform=w_tr,
            src_crs=w_crs,
            dst_transform=dst_t,
            dst_crs="EPSG:4326",
            resampling=rs,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )
        return out

    return {
        "weight_wgs84": _to_wgs84(w_native, "bilinear"),
        "weight_nodata": w_nodata,
    }


def read_corine_clc_wgs84_on_weight_grid(
    weight_path: Path,
    corine_path: Path,
    *,
    corine_band: int,
    west: float,
    south: float,
    east: float,
    north: float,
    display_width: int,
    display_height: int,
) -> dict[str, Any]:
    """
    Reproject CORINE onto the weight raster's native grid, then warp to the same WGS84 display
    grid as ``read_weight_wgs84_only`` (pixel alignment with weights).
    """
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.warp import Resampling, reproject

    dst_t = from_bounds(west, south, east, north, display_width, display_height)
    gh, gw = int(display_height), int(display_width)

    with rasterio.open(weight_path) as wsrc:
        w_tr = wsrc.transform
        w_crs = wsrc.crs
        if w_crs is None:
            raise ValueError(f"Weight raster has no CRS: {weight_path}")
        h_ref, w_ref = int(wsrc.height), int(wsrc.width)

    cor_native = np.full((h_ref, w_ref), np.nan, dtype=np.float64)
    c_nd: float | None = None
    with rasterio.open(corine_path) as csrc:
        if corine_band < 1 or corine_band > int(csrc.count):
            raise ValueError(f"CORINE band {corine_band} invalid for {corine_path}")
        if csrc.crs is None:
            raise ValueError(f"CORINE raster has no CRS: {corine_path}")
        c_nd = csrc.nodata
        reproject(
            source=rasterio.band(csrc, int(corine_band)),
            destination=cor_native,
            src_transform=csrc.transform,
            src_crs=csrc.crs,
            dst_transform=w_tr,
            dst_crs=w_crs,
            resampling=Resampling.nearest,
            src_nodata=np.nan if c_nd is None else float(c_nd),
            dst_nodata=np.nan,
        )

    def _to_wgs84_nearest(src_arr: np.ndarray) -> np.ndarray:
        out = np.full((gh, gw), np.nan, dtype=np.float64)
        reproject(
            source=src_arr,
            destination=out,
            src_transform=w_tr,
            src_crs=w_crs,
            dst_transform=dst_t,
            dst_crs="EPSG:4326",
            resampling=Resampling.nearest,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )
        return out

    return {
        "corine_wgs84": _to_wgs84_nearest(cor_native),
        "corine_nodata": c_nd,
    }


def read_raster_on_weight_grid_wgs84(
    weight_path: Path,
    aux_path: Path,
    *,
    aux_band: int = 1,
    west: float,
    south: float,
    east: float,
    north: float,
    display_width: int,
    display_height: int,
    resampling: str = "bilinear",
) -> dict[str, Any]:
    """
    Reproject one band of ``aux_path`` onto the weight raster's native grid, then warp to the
    same WGS84 display grid as ``read_weight_wgs84_only`` (pixel alignment with weights).
    """
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.warp import Resampling, reproject

    dst_t = from_bounds(west, south, east, north, display_width, display_height)
    gh, gw = int(display_height), int(display_width)

    with rasterio.open(weight_path) as wsrc:
        w_tr = wsrc.transform
        w_crs = wsrc.crs
        if w_crs is None:
            raise ValueError(f"Weight raster has no CRS: {weight_path}")
        h_ref, w_ref = int(wsrc.height), int(wsrc.width)

    aux_native = np.full((h_ref, w_ref), np.nan, dtype=np.float64)
    aux_nd: float | None = None
    rs = getattr(Resampling, str(resampling))
    with rasterio.open(aux_path) as asrc:
        if int(aux_band) < 1 or int(aux_band) > int(asrc.count):
            raise ValueError(f"Band {aux_band} invalid for {aux_path} (count={asrc.count})")
        if asrc.crs is None:
            raise ValueError(f"Auxiliary raster has no CRS: {aux_path}")
        aux_nd = asrc.nodata
        reproject(
            source=rasterio.band(asrc, int(aux_band)),
            destination=aux_native,
            src_transform=asrc.transform,
            src_crs=asrc.crs,
            dst_transform=w_tr,
            dst_crs=w_crs,
            resampling=rs,
            src_nodata=np.nan if aux_nd is None else float(aux_nd),
            dst_nodata=np.nan,
        )

    def _to_wgs84(src_arr: np.ndarray, wgs_rs: Resampling) -> np.ndarray:
        out = np.full((gh, gw), np.nan, dtype=np.float64)
        reproject(
            source=src_arr,
            destination=out,
            src_transform=w_tr,
            src_crs=w_crs,
            dst_transform=dst_t,
            dst_crs="EPSG:4326",
            resampling=wgs_rs,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )
        return out

    wgs_rs = Resampling.nearest if str(resampling).lower() == "nearest" else Resampling.bilinear
    return {
        "values_wgs84": _to_wgs84(aux_native, wgs_rs),
        "nodata": aux_nd,
    }
