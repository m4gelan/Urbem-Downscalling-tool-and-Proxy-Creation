"""
Within-CAMS-cell normalization: for each native CAMS grid cell, fine-grid weights sum to 1.

Pixels with ``cam_cell_id < 0`` are excluded from grouping. Fallback when proxy sum is zero:
uniform weight over domain-valid pixels in that CAMS cell (``valid_mask``).
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Process ravelled arrays in slices so we never allocate e.g. ``flat[cid >= 0]`` for the full grid.
_DEFAULT_CHUNK = 2_000_000


def normalize_within_cams_cells(
    P: np.ndarray,
    cam_cell_id: np.ndarray,
    valid_mask: np.ndarray | None = None,
    *,
    chunk_elems: int = _DEFAULT_CHUNK,
    return_fallback_mask: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Normalize fine-grid weights so each CAMS cell sums to 1 over pixels with ``cam_cell_id >= 0``.

    Returns ``(weights, fallback_uniform_mask)``. Weights match ``P`` shape and dtype float32.

    **Memory:** Result is written into the float32 C-contiguous buffer backing ``P`` when possible
    (``P`` float32 and C-contiguous); otherwise a new float32 array is returned as the first element.
    A separate full-grid ``weights`` allocation is never created.

    When ``return_fallback_mask`` is False, the second return value is ``None`` and uniform-fallback
    pixels are only counted for logging (no bool raster), which avoids a large mask allocation.

    ``valid_mask`` is reserved for future use (pass None).

    Large grids are processed in chunks along the flattened index so peak RAM stays on the order of
    ``chunk_elems`` rather than the full raster size (aside from optional ``fallback_uniform_mask``).
    """
    _ = valid_mask
    work = np.asarray(P, dtype=np.float32, order="C")
    flat = work.ravel()
    cid = np.asarray(cam_cell_id, dtype=np.int64, order="C").ravel()
    n = int(flat.size)
    if int(cid.size) != n:
        raise ValueError("P and cam_cell_id must have the same number of elements.")
    ce = max(10_000, int(chunk_elems))
    out_shape = P.shape

    any_pos = False
    for s in range(0, n, ce):
        e = min(n, s + ce)
        if np.any(cid[s:e] >= 0):
            any_pos = True
            break
    if not any_pos:
        work.fill(0.0)
        fb_ret: np.ndarray | None
        if return_fallback_mask:
            fb_ret = np.zeros_like(work, dtype=bool)
        else:
            fb_ret = None
        return work.reshape(out_shape), fb_ret

    max_c_id = -1
    for s in range(0, n, ce):
        e = min(n, s + ce)
        cc = cid[s:e]
        m = cc >= 0
        if np.any(m):
            max_c_id = max(max_c_id, int(cc[m].max()))
    max_c = int(max_c_id) + 1
    if max_c <= 0:
        work.fill(0.0)
        if return_fallback_mask:
            return work.reshape(out_shape), np.zeros_like(work, dtype=bool)
        return work.reshape(out_shape), None

    sum_c = np.zeros(max_c, dtype=np.float64)
    cnt_c = np.zeros(max_c, dtype=np.int64)
    for s in range(0, n, ce):
        e = min(n, s + ce)
        cc = cid[s:e]
        ff = flat[s:e].astype(np.float64, copy=False)
        m = cc >= 0
        if np.any(m):
            sum_c += np.bincount(cc[m], weights=ff[m], minlength=max_c)
            cnt_c += np.bincount(cc[m], minlength=max_c)

    fb: np.ndarray | None = np.zeros(n, dtype=bool) if return_fallback_mask else None
    inv = np.zeros(max_c, dtype=np.float64)
    mcnt = cnt_c > 0
    inv[mcnt] = 1.0 / cnt_c[mcnt]

    n_fb_total = 0
    for s in range(0, n, ce):
        e = min(n, s + ce)
        cc = cid[s:e]
        ff = flat[s:e]
        flat_s = flat[s:e]
        pos = cc >= 0
        sc = np.zeros(e - s, dtype=np.float64)
        co = np.zeros(e - s, dtype=np.int64)
        sc[pos] = sum_c[cc[pos]]
        co[pos] = cnt_c[cc[pos]]
        ok = pos & (sc > 0.0)
        need_u = pos & (sc <= 0.0) & (co > 0)
        flat_s[ok] = (ff[ok] / sc[ok]).astype(np.float32, copy=False)
        flat_s[need_u] = inv[cc[need_u]].astype(np.float32, copy=False)
        nu = int(np.count_nonzero(need_u))
        if nu:
            n_fb_total += nu
        if return_fallback_mask and fb is not None:
            fb_s = fb[s:e]
            fb_s[need_u] = True

    if n_fb_total:
        logger.warning("Uniform fallback applied on %d fine pixels (zero proxy sum in CAMS cell).", n_fb_total)

    fb_out: np.ndarray | None = fb.reshape(out_shape) if return_fallback_mask and fb is not None else None
    return work.reshape(out_shape), fb_out


def validate_weight_sums(
    weights: np.ndarray,
    cam_cell_id: np.ndarray,
    valid_mask: np.ndarray | None = None,
    tol: float = 1e-4,
    *,
    chunk_elems: int = _DEFAULT_CHUNK,
) -> list[str]:
    """Return list of error strings if any CAMS cell sum deviates from 1."""
    _ = valid_mask
    errs: list[str] = []
    flat = np.asarray(weights, dtype=np.float32, order="C").ravel()
    cid = np.asarray(cam_cell_id, dtype=np.int64, order="C").ravel()
    n = int(flat.size)
    ce = max(10_000, int(chunk_elems))

    any_pos = False
    for s in range(0, n, ce):
        e = min(n, s + ce)
        if np.any(cid[s:e] >= 0):
            any_pos = True
            break
    if not any_pos:
        return errs

    max_c_id = -1
    for s in range(0, n, ce):
        e = min(n, s + ce)
        cc = cid[s:e]
        m = cc >= 0
        if np.any(m):
            max_c_id = max(max_c_id, int(cc[m].max()))
    max_c = int(max_c_id) + 1
    if max_c <= 0:
        return errs

    sums = np.zeros(max_c, dtype=np.float64)
    for s in range(0, n, ce):
        e = min(n, s + ce)
        cc = cid[s:e]
        ff = flat[s:e].astype(np.float64, copy=False)
        m = cc >= 0
        if np.any(m):
            sums += np.bincount(cc[m], weights=ff[m], minlength=max_c)

    seen = np.zeros(max_c, dtype=bool)
    for s in range(0, n, ce):
        e = min(n, s + ce)
        cc = cid[s:e]
        m = cc >= 0
        if not np.any(m):
            continue
        u = np.unique(cc[m])
        for c in u.astype(np.int64, copy=False):
            ci = int(c)
            if ci < 0 or seen[ci]:
                continue
            seen[ci] = True
            ssum = float(sums[ci])
            if abs(ssum - 1.0) > tol and ssum > 0:
                errs.append(f"cam_cell_id={ci} sum={ssum}")
    return errs
