"""
Grid warping functions replicating R's projectRaster, crop(snap='out'), resample.

Used for CAMS-to-domain projection and proxy raster warping.
"""

import logging
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.warp import (
    Resampling,
    calculate_default_transform,
    reproject,
)

from urbem_interface.logging_config import get_logger

logger = get_logger(__name__)


_RESAMPLING_MAP = {
    "nearest":  Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "cubic":    Resampling.cubic,
}


def project_raster_like_R(
    src_array: np.ndarray,
    src_transform: rasterio.Affine,
    src_crs: rasterio.CRS,
    dst_crs: rasterio.CRS,
    src_nodata: float | None = None,
) -> tuple[np.ndarray, rasterio.Affine, int, int]:
    """
    Reproject a raster to dst_crs using a fixed 7000x5550 m resolution,
    matching R's projectRaster behaviour.
    """
    logger.debug(f"--------------------------------")
    logger.debug(f"WARPING RASTER")
    h_src, w_src = src_array.shape
    logger.debug(
        f"project_raster_like_R | input shape=({h_src}, {w_src}) "
        f"src_crs={src_crs.to_epsg()} dst_crs={dst_crs.to_epsg()} "
        f"src_nodata={src_nodata}"
    )

    src_f64 = src_array if src_array.dtype == np.float64 else src_array.astype(np.float64)
    res_x, res_y = 7000.0, 5550.0

    left   = src_transform.c
    top    = src_transform.f
    right  = left + src_transform.a * w_src
    bottom = top  + src_transform.e * h_src

    logger.debug(
        f"project_raster_like_R | src bbox "
        f"left={left:.2f} right={right:.2f} bottom={bottom:.2f} top={top:.2f}"
    )

    try:
        dst_transform, w_out, h_out = calculate_default_transform(
            src_crs, dst_crs, w_src, h_src,
            left=left, bottom=bottom, right=right, top=top,
            resolution=(res_x, res_y),
        )
    except Exception as exc:
        logger.error(f"project_raster_like_R | calculate_default_transform failed: {exc}")
        raise

    logger.debug(
        f"project_raster_like_R | output shape=({h_out}, {w_out}) "
        f"res=({res_x}, {res_y}) dst_transform={dst_transform}"
    )

    FILL = -9999.0
    dst_array = np.full((h_out, w_out), FILL, dtype=np.float64)

    try:
        reproject(
            source=src_f64,
            destination=dst_array,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
            src_nodata=src_nodata,
            dst_nodata=FILL,
        )
    except Exception as exc:
        logger.error(f"project_raster_like_R | reproject failed: {exc}")
        raise

    fill_mask = dst_array == FILL
    dst_array[fill_mask] = np.nan

    n_total = h_out * w_out
    n_fill  = int(fill_mask.sum())
    n_valid = n_total - n_fill

    logger.debug(
        f"project_raster_like_R | fill pixels={n_fill}/{n_total} "
        f"({100 * n_fill / n_total:.1f}% outside projection)"
    )

    if n_valid > 0:
        valid = dst_array[~fill_mask]
        logger.debug(
            f"project_raster_like_R | valid stats "
            f"min={valid.min():.4g} max={valid.max():.4g} "
            f"mean={valid.mean():.4g} n={n_valid}"
        )
    else:
        logger.warning(
            "project_raster_like_R | output contains no valid pixels — "
            "check CRS compatibility or source extent"
        )

    logger.info(
        f"project_raster_like_R | done "
        f"({h_src}x{w_src}) -> ({h_out}x{w_out}) "
        f"src_epsg={src_crs.to_epsg()} dst_epsg={dst_crs.to_epsg()}"
    )
    logger.debug(f"--------------------------------")
    return dst_array, dst_transform, w_out, h_out


def _compute_proxy_coarse_grid(
    src_shape: tuple[int, int],
    src_transform: rasterio.Affine,
    src_crs: rasterio.CRS,
    dst_crs: rasterio.CRS,
    domain_bounds: tuple[float, float, float, float],
    res_x: float = 7000.0,
    res_y: float = 5550.0,
) -> tuple[np.ndarray, rasterio.Affine, int, int, tuple[float, float, float, float]]:
    """
    Build the coarse CAMS grid used for proxy normalization.
    Returns cams_origin_arr, cams_origin_transform, ncol, nrow, bounds.
    """
    logger.debug(f"--------------------------------")
    logger.debug(f"COMPUTING PROXY COARSE GRID")
    h_src, w_src = src_shape
    xmin_d, ymin_d, xmax_d, ymax_d = domain_bounds

    logger.debug(
        f"_compute_proxy_coarse_grid | src shape=({h_src}, {w_src}) "
        f"res=({res_x}, {res_y}) "
        f"src_epsg={src_crs.to_epsg()} dst_epsg={dst_crs.to_epsg()}"
    )

    left_src   = src_transform.c
    top_src    = src_transform.f
    right_src  = left_src + src_transform.a * w_src
    bottom_src = top_src  + src_transform.e * h_src

    try:
        dst_transform, w_out, h_out = calculate_default_transform(
            src_crs, dst_crs, w_src, h_src,
            left=left_src, bottom=bottom_src, right=right_src, top=top_src,
            resolution=(res_x, res_y),
        )
    except Exception as exc:
        logger.error(f"_compute_proxy_coarse_grid | calculate_default_transform failed: {exc}")
        raise

    snapped_left   = dst_transform.c
    snapped_top    = dst_transform.f
    snapped_right  = snapped_left + w_out * res_x
    snapped_bottom = snapped_top  - h_out * res_y

    crop_xmin = snapped_left + np.floor((xmin_d - snapped_left) / res_x) * res_x
    crop_xmax = snapped_left + np.ceil ((xmax_d - snapped_left) / res_x) * res_x
    crop_ymin = snapped_bottom + np.floor((ymin_d - snapped_bottom) / res_y) * res_y
    crop_ymax = snapped_bottom + np.ceil ((ymax_d - snapped_bottom) / res_y) * res_y

    crop_xmin = max(crop_xmin, snapped_left)
    crop_xmax = min(crop_xmax, snapped_right)
    crop_ymin = max(crop_ymin, snapped_bottom)
    crop_ymax = min(crop_ymax, snapped_top)

    if crop_xmin >= crop_xmax or crop_ymin >= crop_ymax:
        logger.warning(
            "_compute_proxy_coarse_grid | snap-crop produced empty extent, "
            "falling back to raw domain bounds"
        )
        crop_xmin, crop_ymin, crop_xmax, crop_ymax = xmin_d, ymin_d, xmax_d, ymax_d

    col_off = int(round((crop_xmin - snapped_left) / res_x))
    col_end = int(round((crop_xmax - snapped_left) / res_x))
    row_off = int(round((snapped_top  - crop_ymax) / res_y))
    row_end = int(round((snapped_top  - crop_ymin) / res_y))

    col_off = max(0,          min(col_off, w_out - 1))
    col_end = max(col_off + 1, min(col_end, w_out))
    row_off = max(0,          min(row_off, h_out - 1))
    row_end = max(row_off + 1, min(row_end, h_out))

    ncol = col_end - col_off
    nrow = row_end - row_off

    left   = snapped_left + col_off * res_x
    top    = snapped_top  - row_off * res_y
    bottom = top  - nrow * res_y
    right  = left + ncol * res_x

    cams_origin_transform = Affine(res_x, 0.0, left, 0.0, -res_y, top)

    cams_origin_arr = np.arange(1, nrow * ncol + 1, dtype=np.float64).reshape(
        (nrow, ncol), order="F"
    )

    logger.debug(
        f"_compute_proxy_coarse_grid | coarse grid "
        f"nrow={nrow} ncol={ncol} n_cells={nrow * ncol} "
        f"left={left:.2f} top={top:.2f} right={right:.2f} bottom={bottom:.2f}"
    )
    logger.info(
        f"_compute_proxy_coarse_grid | done  "
        f"coarse grid ({nrow}x{ncol})  "
        f"dst_epsg={dst_crs.to_epsg()}"
    )
    logger.debug(f"--------------------------------")
    return cams_origin_arr, cams_origin_transform, ncol, nrow, (left, bottom, right, top)


def _warp_to_domain(
    src_arr: np.ndarray,
    src_transform: rasterio.Affine,
    src_crs: rasterio.CRS,
    domain_transform: rasterio.Affine,
    domain_shape: tuple[int, int],
    dst_crs: rasterio.CRS,
    domain_bounds: tuple[float, float, float, float],
    cams_utm_grid: dict | None = None,
    resampling: str = "nearest",
    label: str | None = None,
) -> np.ndarray:
    """Replicate R's 3-step warp: projectRaster -> crop(snap='out') -> resample."""
    resamp = _RESAMPLING_MAP.get(resampling, Resampling.nearest)
    tag    = f" [{label}]" if label else ""

    h_src, w_src = src_arr.shape
    xmin_d, ymin_d, xmax_d, ymax_d = domain_bounds

    logger.debug(
        f"warp{tag} | start  src_shape=({h_src}, {w_src}) "
        f"domain_shape={domain_shape} resampling={resampling}"
    )

    intermediate, transform_utm, w_utm, h_utm = project_raster_like_R(
        src_arr, src_transform, src_crs, dst_crs, src_nodata=None
    )

    res_x          = transform_utm.a
    res_y          = abs(transform_utm.e)
    snapped_left   = transform_utm.c
    snapped_top    = transform_utm.f
    snapped_right  = snapped_left + w_utm * res_x
    snapped_bottom = snapped_top  - h_utm * res_y

    if logger.isEnabledFor(logging.DEBUG):
        fin1 = np.isfinite(intermediate)
        if fin1.any():
            v1 = intermediate[fin1]
            logger.debug(
                f"warp{tag} | step1_project  shape={intermediate.shape} "
                f"valid_frac={fin1.mean():.3f} sum={v1.sum():.1f} "
                f"utm_bounds=({snapped_left:.1f},{snapped_bottom:.1f},"
                f"{snapped_right:.1f},{snapped_top:.1f})"
            )
        else:
            logger.warning(f"warp{tag} | step1_project produced no finite values")

    crop_xmin = snapped_left   + np.floor((xmin_d - snapped_left)   / res_x) * res_x
    crop_xmax = snapped_left   + np.ceil ((xmax_d - snapped_left)   / res_x) * res_x
    crop_ymin = snapped_bottom + np.floor((ymin_d - snapped_bottom) / res_y) * res_y
    crop_ymax = snapped_bottom + np.ceil ((ymax_d - snapped_bottom) / res_y) * res_y

    logger.debug(
        f"warp{tag} | step2_snap  "
        f"crop=({crop_xmin:.1f},{crop_ymin:.1f},{crop_xmax:.1f},{crop_ymax:.1f})"
    )

    crop_xmin = max(crop_xmin, snapped_left)
    crop_xmax = min(crop_xmax, snapped_right)
    crop_ymin = max(crop_ymin, snapped_bottom)
    crop_ymax = min(crop_ymax, snapped_top)

    if crop_xmin >= crop_xmax or crop_ymin >= crop_ymax:
        logger.warning(
            f"warp{tag} | step2_crop  domain outside projected extent — returning NaN array"
        )
        return np.full(domain_shape, np.nan, dtype=np.float64)

    col_off = int(round((crop_xmin - snapped_left) / res_x))
    col_end = int(round((crop_xmax - snapped_left) / res_x))
    row_off = int(round((snapped_top - crop_ymax)  / res_y))
    row_end = int(round((snapped_top - crop_ymin)  / res_y))

    col_off = max(0,           min(col_off, w_utm - 1))
    col_end = max(col_off + 1, min(col_end, w_utm))
    row_off = max(0,           min(row_off, h_utm - 1))
    row_end = max(row_off + 1, min(row_end, h_utm))

    crop_h = row_end - row_off
    crop_w = col_end - col_off

    cropped        = intermediate[row_off:row_end, col_off:col_end]
    crop_left      = snapped_left + col_off * res_x
    crop_top       = snapped_top  - row_off * res_y
    crop_transform = Affine(res_x, 0.0, crop_left, 0.0, -res_y, crop_top)

    if logger.isEnabledFor(logging.DEBUG):
        fin2 = np.isfinite(cropped)
        if fin2.any():
            v2 = cropped[fin2]
            logger.debug(
                f"warp{tag} | step2_crop  "
                f"rows={row_off}:{row_end} cols={col_off}:{col_end} "
                f"size=({crop_h}x{crop_w}) "
                f"valid_frac={fin2.mean():.3f} sum={v2.sum():.1f}"
            )

    dst = np.full(domain_shape, np.nan, dtype=np.float64)
    reproject(
        source=cropped,
        destination=dst,
        src_transform=crop_transform,
        src_crs=dst_crs,
        dst_transform=domain_transform,
        dst_crs=dst_crs,
        resampling=resamp,
    )

    if logger.isEnabledFor(logging.DEBUG):
        fin3 = np.isfinite(dst)
        if fin3.any():
            v3 = dst[fin3]
            logger.debug(
                f"warp{tag} | step3_resample  "
                f"valid_frac={fin3.mean():.3f} sum={v3.sum():.1f} "
                f"min={v3.min():.3g} max={v3.max():.3g}"
            )
        else:
            logger.warning(f"warp{tag} | step3_resample produced no finite values")

    logger.info(f"warp{tag} | done  domain_shape={domain_shape}")
    return dst


def _load_and_warp_proxy(
    proxy_path: Path,
    domain_transform: rasterio.Affine,
    domain_shape: tuple[int, int],
    domain_crs: rasterio.CRS,
    resampling: Resampling = Resampling.bilinear,
) -> np.ndarray:
    """Load proxy raster and warp to domain grid."""
    with rasterio.open(proxy_path) as src:
        src_crs       = src.crs
        src_transform = src.transform
        src_arr       = src.read(1)

    logger.debug(
        f"_load_and_warp_proxy | loaded {proxy_path.name}  "
        f"shape={src_arr.shape} crs={src_crs.to_epsg()} "
        f"dtype={src_arr.dtype}"
    )

    dst = np.full(domain_shape, np.nan, dtype=np.float64)
    reproject(
        source=src_arr,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=domain_transform,
        dst_crs=domain_crs,
        resampling=resampling,
    )

    np.nan_to_num(dst, nan=0.0, copy=False)

    if logger.isEnabledFor(logging.DEBUG):
        nonzero = np.count_nonzero(dst)
        logger.debug(
            f"_load_and_warp_proxy | done  "
            f"nonzero={nonzero}/{dst.size} "
            f"({100 * nonzero / dst.size:.1f}%) "
            f"sum={dst.sum():.3g}"
        )

    logger.info(
        f"_load_and_warp_proxy | done  {proxy_path.name} -> shape={domain_shape}"
    )
    return dst
