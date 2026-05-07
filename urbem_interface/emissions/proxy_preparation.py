"""
Replicate proxy_preparation.R: proxy_cwd and proxy_distribution.
"""

from typing import Literal

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling as WarpResampling

from urbem_interface.logging_config import get_logger

logger = get_logger(__name__)


def proxy_cwd(
    cams_origin: np.ndarray,
    cams_transform: rasterio.Affine,
    cams_crs: rasterio.CRS,
    domain_arr: np.ndarray,
    domain_transform: rasterio.Affine,
    domain_crs: rasterio.CRS,
    proxy_arr: np.ndarray,
    proxy_transform: rasterio.Affine,
    proxy_crs: rasterio.CRS,
    sparse_threshold: float = 0.05,
    max_weight_per_cell: float = 0.5,
) -> np.ndarray:
    """Cell-wise distribution of proxy weights. Normalize proxy within each coarse CAMS cell."""
    logger.debug(f"PROXY PREPARATION")
    proxy = np.where(np.isnan(proxy_arr), 0.0, proxy_arr).astype(np.float64)
    cams_ids = np.asarray(cams_origin, dtype=np.float64)

    domain_h, domain_w = domain_arr.shape
    logger.debug(f"Domain shape for proxy (height, width): {domain_h} x {domain_w}")

    cams_domain_ids = np.full((domain_h, domain_w), np.nan, dtype=np.float64)
    reproject(
        source=cams_ids,
        destination=cams_domain_ids,
        src_transform=cams_transform,
        src_crs=cams_crs,
        dst_transform=domain_transform,
        dst_crs=domain_crs,
        resampling=WarpResampling.nearest,
    )

    flat_ids = cams_domain_ids.ravel()
    flat_proxy = proxy.ravel()

    valid_mask = np.isfinite(flat_ids) & (flat_ids > 0)
    valid_pixels = np.where(valid_mask)[0]

    if valid_pixels.size == 0:
        logger.debug("No valid pixels found after masking. Returning zeros.")
        return np.zeros((domain_h, domain_w), dtype=np.float64)

    ids_valid = flat_ids[valid_pixels]
    unique_ids, labels = np.unique(ids_valid, return_inverse=True)
    n_coarse = len(unique_ids)

    proxy_valid = flat_proxy[valid_pixels]
    is_nonzero = proxy_valid > 0.0

    res_c = (abs(cams_transform[0]), abs(cams_transform[4]))
    res_d = (abs(domain_transform[0]), abs(domain_transform[4]))
    max_cell = (res_c[0] / res_d[0]) * (res_c[1] / res_d[1])

    n_cells    = np.bincount(labels, minlength=n_coarse).astype(np.float64)
    proxy_sum  = np.bincount(labels, weights=proxy_valid, minlength=n_coarse)
    n_nonzero  = np.bincount(labels, weights=is_nonzero.astype(np.float64), minlength=n_coarse)

    coverage_fraction = np.where(n_cells > 0, n_nonzero / n_cells, 0.0)
    coverage_ratio    = np.minimum(n_cells / max_cell, 1.0)
    apply_coverage    = coverage_ratio < 0.8

    uniform_per_pixel = 1.0 / max_cell

    cell_proxy_sum  = proxy_sum[labels]
    cell_n_nonzero  = n_nonzero[labels]
    cell_n_cells    = n_cells[labels]
    cell_cov_frac   = coverage_fraction[labels]

    has_signal = cell_proxy_sum > 0.0
    proxy_norm = np.where(
        has_signal,
        proxy_valid / np.where(has_signal, cell_proxy_sum, 1.0),
        uniform_per_pixel,
    )

    proxy_sq_sum = np.bincount(labels, weights=proxy_valid ** 2, minlength=n_coarse)
    mean_sq  = np.where(n_cells > 0, proxy_sq_sum / n_cells, 0.0)
    mean_val = np.where(n_cells > 0, proxy_sum   / n_cells, 0.0)
    is_uniform_cell = (mean_sq - mean_val ** 2) < 1e-30

    is_uniform_pixel = is_uniform_cell[labels]
    proxy_norm = np.where(is_uniform_pixel & has_signal, uniform_per_pixel, proxy_norm)

    blend_few   = (~is_uniform_cell) & (n_nonzero > 0) & (n_nonzero <= 3)
    blend_sparse = (
        (~is_uniform_cell)
        & ~blend_few
        & (coverage_fraction > 0)
        & (coverage_fraction < sparse_threshold)
    )

    proxy_weight_few = np.clip(n_nonzero / 10.0, 0.0, 0.3)
    uniform_weight_sparse = np.clip(
        1.0 - (coverage_fraction / sparse_threshold), 0.0, 0.8
    )

    blend_few_px    = blend_few[labels]
    blend_sparse_px = blend_sparse[labels]
    pw_few          = proxy_weight_few[labels]
    uw_sparse       = uniform_weight_sparse[labels]

    proxy_norm = np.where(
        blend_few_px & has_signal,
        (1.0 - pw_few) * uniform_per_pixel + pw_few * proxy_norm,
        proxy_norm,
    )
    proxy_norm = np.where(
        blend_sparse_px & has_signal,
        (1.0 - uw_sparse) * proxy_norm + uw_sparse * uniform_per_pixel,
        proxy_norm,
    )

    needs_renorm = blend_few_px | blend_sparse_px
    if needs_renorm.any():
        blended_sum = np.bincount(
            labels[needs_renorm],
            weights=proxy_norm[needs_renorm],
            minlength=n_coarse,
        )
        cell_blended_sum = blended_sum[labels]
        safe_sum = np.where(cell_blended_sum > 0, cell_blended_sum, 1.0)
        proxy_norm = np.where(
            needs_renorm,
            proxy_norm / safe_sum,
            proxy_norm,
        )

    dynamic_max = np.where(
        blend_few,
        np.maximum(0.1, max_weight_per_cell * (n_nonzero / 3.0)),
        np.where(
            blend_sparse,
            max_weight_per_cell * (1.0 - (sparse_threshold - coverage_fraction)),
            max_weight_per_cell,
        ),
    )

    cell_dynamic_max = dynamic_max[labels]

    exceeds = proxy_norm > cell_dynamic_max
    if exceeds.any():
        excess_per_pixel = np.where(exceeds, proxy_norm - cell_dynamic_max, 0.0)
        excess_per_cell  = np.bincount(labels, weights=excess_per_pixel, minlength=n_coarse)

        under_count = np.bincount(
            labels, weights=(~exceeds).astype(np.float64), minlength=n_coarse
        )
        redist = np.where(under_count > 0, excess_per_cell / under_count, 0.0)
        cell_redist = redist[labels]

        proxy_norm = np.where(exceeds, cell_dynamic_max, proxy_norm + cell_redist)

        final_sum = np.bincount(labels, weights=proxy_norm, minlength=n_coarse)
        cell_final_sum = final_sum[labels]
        proxy_norm = np.where(
            cell_final_sum > 0, proxy_norm / cell_final_sum, proxy_norm
        )

    cell_coverage_ratio = coverage_ratio[labels]
    apply_cov_px        = apply_coverage[labels]
    proxy_norm = np.where(apply_cov_px, proxy_norm * cell_coverage_ratio, proxy_norm)

    out = np.zeros(domain_h * domain_w, dtype=np.float64)
    out[valid_pixels] = proxy_norm

    return out.reshape(domain_h, domain_w)


def proxy_distribution(
    emissions: dict[str, np.ndarray],
    proxy: np.ndarray,
    proxy_method: Literal["top_down_proxy", "coarse_cells_proxy"] = "coarse_cells_proxy",
) -> dict[str, np.ndarray]:
    """Apply proxy weights to emission layers."""
    result = {}
    for poll, arr in emissions.items():
        if proxy_method == "top_down_proxy":
            total = float(np.nansum(arr))
            result[poll] = total * proxy
        else:
            result[poll] = arr * proxy
    return result
