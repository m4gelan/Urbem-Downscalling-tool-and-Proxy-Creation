from __future__ import annotations

from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio import features as rio_features
from rasterio.crs import CRS
from rasterio.warp import reproject, Resampling
from shapely.geometry import Point, mapping


def pixel_centre_axes(
    transform: rasterio.Affine, height: int, width: int
) -> tuple[np.ndarray, np.ndarray]:
    """1-D pixel-centre coordinates in the raster CRS (x along columns, y along rows)."""
    x = transform.c + (np.arange(width, dtype=np.float64) + 0.5) * transform.a
    y = transform.f + (np.arange(height, dtype=np.float64) + 0.5) * transform.e
    return x, y


def _cams_cell_id_flat(
    flat_lon: np.ndarray,
    flat_lat: np.ndarray,
    lon_bounds: np.ndarray,
    lat_bounds: np.ndarray,
    nlon: int,
    nlat: int,
) -> np.ndarray:
    lo = np.searchsorted(lon_bounds[:, 0], flat_lon, side="right") - 1
    lo = np.clip(lo, 0, nlon - 1)
    ok_lon = (flat_lon >= lon_bounds[lo, 0]) & (flat_lon < lon_bounds[lo, 1])
    la = np.searchsorted(lat_bounds[:, 0], flat_lat, side="right") - 1
    la = np.clip(la, 0, nlat - 1)
    ok_lat = (flat_lat >= lat_bounds[la, 0]) & (flat_lat < lat_bounds[la, 1])
    cid = la.astype(np.int32) * int(nlon) + lo.astype(np.int32)
    cid[~(ok_lon & ok_lat)] = -1
    return cid


def cams_cell_id_for_raster(
    transform: rasterio.Affine,
    raster_crs: Any,
    height: int,
    width: int,
    lon_bounds: np.ndarray,
    lat_bounds: np.ndarray,
    nlon: int,
    nlat: int,
    *,
    row_chunk: int = 256,
) -> np.ndarray:
    """CAMS cell id per pixel without allocating full-grid lon/lat arrays."""
    from pyproj import Transformer

    h, w = int(height), int(width)
    px, py = pixel_centre_axes(transform, h, w)
    tr = Transformer.from_crs(raster_crs, "EPSG:4326", always_xy=True)
    out = np.full((h, w), -1, dtype=np.int32)
    step = max(1, int(row_chunk))
    for i0 in range(0, h, step):
        i1 = min(h, i0 + step)
        yy, xx = np.meshgrid(py[i0:i1], px, indexing="ij")
        lon, lat = tr.transform(xx, yy)
        cid = _cams_cell_id_flat(
            np.asarray(lon, dtype=np.float64).ravel(),
            np.asarray(lat, dtype=np.float64).ravel(),
            lon_bounds,
            lat_bounds,
            nlon,
            nlat,
        )
        out[i0:i1, :] = cid.reshape(i1 - i0, w)
    return out


def restrict_cell_ids_to_country(
    cell_id: np.ndarray,
    cams_cells: dict[int, dict[str, Any]],
) -> np.ndarray:
    """``-1`` where the flat index is not a key of ``cams_cells``."""
    if not cams_cells:
        return np.full_like(cell_id, -1)
    keys = np.fromiter(cams_cells.keys(), dtype=np.int32)
    out = np.asarray(cell_id, dtype=np.int32).copy()
    out[~np.isin(out, keys)] = -1
    return out


def buffer_binary_mask(mask: np.ndarray, transform: rasterio.Affine, buffer_m: float) -> np.ndarray:
    """Dilate a 0/1 mask by ``buffer_m`` (square structuring element in map units)."""
    from scipy.ndimage import binary_dilation

    res = float(min(abs(transform.a), abs(transform.e)))
    if res <= 0:
        raise ValueError("invalid raster resolution for buffer")
    radius_px = max(1, int(round(float(buffer_m) / res)))
    y, x = np.ogrid[-radius_px : radius_px + 1, -radius_px : radius_px + 1]
    structure = (x * x + y * y) <= radius_px * radius_px
    return binary_dilation(mask > 0, structure=structure).astype(np.float32)


def points_in_mask(
    lon: np.ndarray,
    lat: np.ndarray,
    mask: np.ndarray,
    transform: rasterio.Affine,
    raster_crs: Any,
) -> np.ndarray:
    """True where each WGS84 point falls on a mask pixel > 0."""
    from rasterio.transform import rowcol

    tr = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
    xs, ys = tr.transform(lon, lat)
    rows, cols = rowcol(transform, xs, ys)
    h, w = mask.shape
    ok = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    hit = np.zeros(len(lon), dtype=bool)
    hit[ok] = mask[rows[ok], cols[ok]] > 0
    return hit


def rasterize_points_max(
    lon: np.ndarray,
    lat: np.ndarray,
    values: np.ndarray,
    *,
    height: int,
    width: int,
    transform: rasterio.Affine,
    raster_crs: Any,
) -> np.ndarray:
    """Per-pixel maximum ``values`` at point locations (WGS84 lon/lat)."""
    from rasterio.transform import rowcol

    h, w = int(height), int(width)
    out = np.zeros((h, w), dtype=np.float32)
    if lon.size == 0:
        return out

    vals = np.asarray(values, dtype=np.float32)
    tr = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
    xs, ys = tr.transform(lon, lat)
    rows, cols = rowcol(transform, xs, ys)
    ok = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    if ok.any():
        np.maximum.at(out, (rows[ok], cols[ok]), vals[ok])
    return out


def _as_crs(c: Any) -> CRS:
    """Build ``rasterio.crs.CRS`` without ``from_user`` (not in all rasterio versions)."""
    if isinstance(c, CRS):
        return c
    if isinstance(c, (int, np.integer)):
        return CRS.from_epsg(int(c))
    if isinstance(c, str):
        return CRS.from_string(c)
    return CRS(c)


def rasters_differ_on_grid(
    shape_a: tuple[int, ...],
    transform_a: rasterio.Affine,
    crs_a: Any,
    shape_b: tuple[int, ...],
    transform_b: rasterio.Affine,
    crs_b: Any,
) -> bool:
    """True if shape, CRS, or affine differ."""
    if shape_a != shape_b:
        return True
    ca, cb = _as_crs(crs_a), _as_crs(crs_b)
    if hasattr(ca, "equals"):
        if not ca.equals(cb):
            return True
    elif ca != cb:
        return True
    return not np.allclose(
        np.array(transform_a)[:6], np.array(transform_b)[:6], rtol=0.0, atol=1e-6
    )


def reproject_to_reference(
    data: np.ndarray,
    src_transform: rasterio.Affine,
    src_crs: Any,
    dst_height: int,
    dst_width: int,
    dst_transform: rasterio.Affine,
    dst_crs: Any,
    *,
    resampling: Resampling,
    src_nodata: float | int | None = None,
    dst_dtype: np.dtype,
    dest_init_nan: bool = False,
) -> np.ndarray:
    """
    Sample ``data`` onto another raster's grid (CORINE reference, population warp, etc.).

    If ``dest_init_nan`` is True, the destination starts as NaN and ``dst_nodata`` is NaN
    so pixels with no source coverage stay NaN (population on a wider CORINE grid).
    """
    src = np.asarray(data)
    if dest_init_nan:
        if not np.issubdtype(dst_dtype, np.floating):
            raise ValueError("dest_init_nan requires a float dst_dtype")
        out = np.full((dst_height, dst_width), np.nan, dtype=dst_dtype)
        dst_nodata = np.nan
    else:
        out = np.zeros((dst_height, dst_width), dtype=dst_dtype)
        dst_nodata = None

    reproject(
        source=src,
        destination=out,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=resampling,
        src_nodata=src_nodata,
        dst_nodata=dst_nodata,
    )
    return out


def warp_raster_to_grid(
    data: np.ndarray,
    src_transform: rasterio.Affine,
    src_crs: Any,
    ref_height: int,
    ref_width: int,
    ref_transform: rasterio.Affine,
    ref_crs: Any,
    *,
    src_nodata: float | int | None = None,
    resampling: Resampling = Resampling.bilinear,
    dest_init_nan: bool = True,
    nan_fill: float | None = None,
) -> np.ndarray:
    """Reproject *data* onto a reference grid if shape/CRS/affine differ; else return float32 copy."""
    ref_shape = (int(ref_height), int(ref_width))
    if not rasters_differ_on_grid(
        data.shape, src_transform, src_crs, ref_shape, ref_transform, ref_crs
    ):
        return np.asarray(data, dtype=np.float32)
    out = reproject_to_reference(
        data,
        src_transform,
        src_crs,
        ref_height,
        ref_width,
        ref_transform,
        ref_crs,
        resampling=resampling,
        src_nodata=src_nodata,
        dst_dtype=np.dtype(np.float32),
        dest_init_nan=dest_init_nan,
    )
    if nan_fill is not None:
        fill = float(nan_fill)
        out = np.nan_to_num(out, nan=fill, posinf=fill, neginf=fill)
    return out


def rasterize_buffered_points(
    lons: np.ndarray,
    lats: np.ndarray,
    *,
    buffer_m: float,
    metric_crs: str,
    height: int,
    width: int,
    transform: rasterio.Affine,
    raster_crs: Any,
    burn_value: float = 1.0,
    fill: float = 0.0,
    dtype: Any = np.float32,
    all_touched: bool = True,
) -> np.ndarray:
    """Burn circular buffers (metres) around WGS84 lon/lat points onto the reference grid."""
    out_dtype = np.dtype(dtype)
    acc = np.full((int(height), int(width)), float(fill), dtype=np.float32)
    if lons.size == 0:
        return acc.astype(out_dtype, copy=False)

    gdf = gpd.GeoDataFrame(
        geometry=[Point(xy) for xy in zip(lons.tolist(), lats.tolist())],
        crs="EPSG:4326",
    )
    gdf = gdf.to_crs(metric_crs)
    buf = float(buffer_m)
    if buf > 0:
        gdf = gdf.copy()
        gdf["geometry"] = gdf.geometry.buffer(buf)
    gdf = gdf.to_crs(raster_crs)
    shapes = (
        (geom, float(burn_value))
        for geom in gdf.geometry
        if geom is not None and not geom.is_empty
    )
    rio_features.rasterize(
        shapes,
        out_shape=(int(height), int(width)),
        transform=transform,
        fill=float(fill),
        out=acc,
        dtype=np.float32,
        all_touched=all_touched,
        merge_alg=rasterio.enums.MergeAlg.replace,
    )
    return acc.astype(out_dtype, copy=False)
