from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.warp import reproject

from UrbEm_Visualizer.dataset_loaders.cams_emissions import cell_id_from_lonlat
from UrbEm_Visualizer.downscaling.spatial import FineGrid, NativeGridMeta
from UrbEm_Visualizer.pollutants import band_index_for_pollutant


def pixel_centre_axes(transform: rasterio.Affine, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    x = transform.c + (np.arange(width, dtype=np.float64) + 0.5) * transform.a
    y = transform.f + (np.arange(height, dtype=np.float64) + 0.5) * transform.e
    return x, y


def cell_id_on_raster(
    transform: rasterio.Affine,
    crs: str,
    height: int,
    width: int,
    cams_grid: dict,
    valid_cell_ids: set[int] | frozenset[int],
    *,
    row_chunk: int = 256,
) -> np.ndarray:
    lon_bounds = np.asarray(cams_grid["lon_bounds"], dtype=np.float64)
    lat_bounds = np.asarray(cams_grid["lat_bounds"], dtype=np.float64)
    nlon = int(cams_grid["n_longitude"])
    nlat = int(cams_grid["n_latitude"])
    h, w = int(height), int(width)
    px, py = pixel_centre_axes(transform, h, w)
    tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    out = np.full((h, w), -1, dtype=np.int32)
    keys = frozenset(valid_cell_ids) if valid_cell_ids else frozenset()
    step = max(1, int(row_chunk))
    for i0 in range(0, h, step):
        i1 = min(h, i0 + step)
        yy, xx = np.meshgrid(py[i0:i1], px, indexing="ij")
        lon, lat = tr.transform(xx, yy)
        cid = cell_id_from_lonlat(
            np.asarray(lon, dtype=np.float64).ravel(),
            np.asarray(lat, dtype=np.float64).ravel(),
            lon_bounds,
            lat_bounds,
            nlon,
            nlat,
        )
        if keys:
            cid[~np.isin(cid, list(keys))] = -1
        out[i0:i1, :] = cid.reshape(i1 - i0, w)
    return out


def pixels_in_domain_bbox(
    transform: rasterio.Affine,
    crs: str,
    height: int,
    width: int,
    domain: dict,
) -> np.ndarray:
    """True where pixel centre lies inside domain xmin/ymin/xmax/ymax (domain crs)."""
    px, py = pixel_centre_axes(transform, height, width)
    dcrs = str(domain["crs"])
    if crs != dcrs:
        tr = Transformer.from_crs(crs, dcrs, always_xy=True)
        px, py = tr.transform(px, py)
    xmin = float(domain["xmin"])
    ymin = float(domain["ymin"])
    xmax = float(domain["xmax"])
    ymax = float(domain["ymax"])
    yy, xx = np.meshgrid(py, px, indexing="ij")
    return (xx >= xmin) & (xx <= xmax) & (yy >= ymin) & (yy <= ymax)


def read_weight_stack_native(path: Path) -> tuple[list[str], np.ndarray, rasterio.Affine, str]:
    with rasterio.open(path) as src:
        names = []
        for i in range(1, src.count + 1):
            d = src.descriptions[i - 1] if src.descriptions else None
            names.append(str(d) if d else f"band_{i}")
        stack = src.read().astype(np.float32)
        nodata = src.nodata
        transform = src.transform
        crs = str(src.crs)
    if nodata is not None:
        stack[stack == float(nodata)] = 0.0
    np.nan_to_num(stack, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return names, stack, transform, crs


def aggregate_plane_to_grid(
    plane: np.ndarray,
    nat_tr: rasterio.Affine,
    nat_crs: str,
    grid: FineGrid,
) -> np.ndarray:
    h, w = plane.shape
    px, py = pixel_centre_axes(nat_tr, h, w)
    yy, xx = np.meshgrid(py, px, indexing="ij")
    if nat_crs != grid.crs:
        tr = Transformer.from_crs(nat_crs, grid.crs, always_xy=True)
        xx, yy = tr.transform(xx, yy)
    cols = np.floor((xx - grid.transform.c) / grid.transform.a).astype(np.int64)
    rows = np.floor((yy - grid.transform.f) / grid.transform.e).astype(np.int64)
    agg = np.zeros((grid.height, grid.width), dtype=np.float64)
    flat_w = plane.ravel().astype(np.float64)
    flat_r = rows.ravel()
    flat_c = cols.ravel()
    valid = (flat_r >= 0) & (flat_r < grid.height) & (flat_c >= 0) & (flat_c < grid.width)
    np.add.at(agg, (flat_r[valid], flat_c[valid]), flat_w[valid])
    return agg.astype(np.float32)


def weight_planes_on_grid(
    area_path: Path,
    grid: FineGrid,
    output_resolution_m: int,
    native_meta: NativeGridMeta,
    pollutants: list[str],
) -> dict[str, np.ndarray]:
    labels, native_stack, nat_tr, nat_crs = read_weight_stack_native(area_path)
    native_res = int(native_meta.res_x)
    out: dict[str, np.ndarray] = {}
    for pol in pollutants:
        bi = band_index_for_pollutant(labels, pol)
        if output_resolution_m > native_res:
            out[pol] = aggregate_plane_to_grid(native_stack[bi], nat_tr, nat_crs, grid)
        else:
            out[pol] = reproject_plane_to_grid(native_stack[bi], nat_tr, nat_crs, grid)
    return out


def reproject_plane_to_grid(
    plane: np.ndarray,
    src_transform: rasterio.Affine,
    src_crs: str,
    grid: FineGrid,
) -> np.ndarray:
    out = np.zeros((grid.height, grid.width), dtype=np.float32)
    reproject(
        source=plane,
        destination=out,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=grid.transform,
        dst_crs=grid.crs,
        resampling=Resampling.nearest,
    )
    np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def cell_id_plane(
    grid: FineGrid,
    cams_grid: dict,
    valid_cell_ids: set[int] | frozenset[int],
    *,
    row_chunk: int = 256,
) -> np.ndarray:
    return cell_id_on_raster(
        grid.transform, grid.crs, grid.height, grid.width, cams_grid, valid_cell_ids, row_chunk=row_chunk,
    )


def load_raster_to_grid(path: Path, grid: FineGrid, band: int = 1) -> np.ndarray:
    out = np.zeros((grid.height, grid.width), dtype=np.float32)
    with rasterio.open(path) as src:
        src_data = np.zeros((grid.height, grid.width), dtype=np.float32)
        reproject(
            source=rasterio.band(src, band),
            destination=src_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=grid.transform,
            dst_crs=grid.crs,
            resampling=Resampling.nearest,
        )
        nodata = src.nodata
    if nodata is not None:
        src_data[src_data == float(nodata)] = 0.0
    np.nan_to_num(src_data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    out[:] = src_data
    return out


def load_multiband_to_grid(path: Path, grid: FineGrid) -> tuple[list[str], np.ndarray]:
    with rasterio.open(path) as src:
        names = []
        for i in range(1, src.count + 1):
            d = src.descriptions[i - 1] if src.descriptions else None
            names.append(str(d) if d else f"band_{i}")
        stack = np.zeros((src.count, grid.height, grid.width), dtype=np.float32)
        for b in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, b),
                destination=stack[b - 1],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=grid.transform,
                dst_crs=grid.crs,
                resampling=Resampling.nearest,
            )
        nodata = src.nodata
    if nodata is not None:
        stack[stack == float(nodata)] = 0.0
    np.nan_to_num(stack, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return names, stack


def lonlat_to_rowcol(grid: FineGrid, lon: float, lat: float) -> tuple[int, int]:
    tr = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True)
    x, y = tr.transform(lon, lat)
    inv = ~grid.transform
    col, row = inv * (x, y)
    return int(row), int(col)


def deposit_point(
    grid: FineGrid,
    plane: np.ndarray,
    lon: float,
    lat: float,
    mass: float,
) -> None:
    r, c = lonlat_to_rowcol(grid, lon, lat)
    if 0 <= r < grid.height and 0 <= c < grid.width:
        plane[r, c] += np.float32(mass)
