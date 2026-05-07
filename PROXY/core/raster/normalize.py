"""Guarded normalization helpers (sum-to-one, per-cell, safe division).

Today the codebase has near-duplicate implementations in:

- ``PROXY/sectors/J_Waste/normalization_waste.py`` (within-CAMS-cell normalization)
- ``Waste/j_waste_weights/normalization`` (same algorithm, parallel copy)
- ``PROXY/core/area_allocation.normalize_subsectors_within_cells``
- ``PROXY/sectors/E_Solvents/within_cams.py`` (parent-cell normalization)

This module consolidates the generic primitives. Sectors migrate to these helpers during
Phase 3/4.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_CHUNK = 2_000_000


def safe_divide(numerator: np.ndarray, denominator: np.ndarray, *, fill: float = 0.0) -> np.ndarray:
    """Element-wise division that returns ``fill`` where ``denominator`` is 0 or non-finite."""
    out = np.full_like(numerator, fill, dtype=np.float64)
    np.divide(
        numerator,
        denominator,
        out=out,
        where=(denominator != 0) & np.isfinite(denominator),
    )
    out[~np.isfinite(out)] = fill
    return out


def sum_to_one(values: np.ndarray, *, axis: int | None = None) -> np.ndarray:
    """Normalize ``values`` so they sum to 1 along ``axis``; zeros-array stays zeros."""
    total = np.sum(values, axis=axis, keepdims=True if axis is not None else False)
    if axis is None:
        if not np.isfinite(total) or total <= 0:
            return np.asarray(values, dtype=np.float64)
        return np.asarray(values, dtype=np.float64) / float(total)
    return safe_divide(np.asarray(values, dtype=np.float64), np.asarray(total, dtype=np.float64), fill=0.0)


def normalize_indicator_quantile_minmax(
    arr: np.ndarray,
    *,
    method: str = "quantile_minmax",
    q_low: float = 0.01,
    q_high: float = 0.99,
) -> np.ndarray:
    """Map finite positive indicator values to ``[0, 1]`` using quantile min-max clipping."""
    x = np.asarray(arr, dtype=np.float32)
    if method != "quantile_minmax":
        raise ValueError(f"Unknown normalization method: {method!r}")
    m = np.isfinite(x) & (x > 0)
    if not np.any(m):
        return np.zeros_like(x, dtype=np.float32)
    vals = x[m]
    lo = float(np.quantile(vals.astype(np.float64), q_low))
    hi = float(np.quantile(vals.astype(np.float64), q_high))
    if hi <= lo:
        hi = lo + 1e-30
    y = np.clip(x, np.float32(lo), np.float32(hi), out=np.empty_like(x))
    den = float(hi) - float(lo)
    if den <= 0.0:
        den = 1e-30
    y = ((y.astype(np.float64) - float(lo)) / den).astype(np.float32)
    np.nan_to_num(y, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.maximum(y, 0.0, out=y)
    return y


def normalize_within_bincount_cells(
    values: np.ndarray,
    cell_ids: np.ndarray,
    *,
    invalid_id: int = -1,
) -> np.ndarray:
    """Sum-to-one of ``values`` within each unique ``cell_ids`` group.

    Arrays must share shape. Pixels whose ``cell_ids`` equal ``invalid_id`` are zeroed.
    Cells whose sum is non-positive are left at zero (callers pick their own fallback).
    """
    if values.shape != cell_ids.shape:
        raise ValueError(
            f"values and cell_ids shape mismatch: {values.shape} vs {cell_ids.shape}"
        )
    v = np.asarray(values, dtype=np.float64).copy()
    c = np.asarray(cell_ids, dtype=np.int64)
    invalid = c == int(invalid_id)
    v[invalid] = 0.0
    c = c.copy()
    c[invalid] = 0
    max_id = int(c.max()) if c.size else 0
    sums = np.bincount(c.ravel(), weights=v.ravel(), minlength=max_id + 1)
    denom = sums[c]
    out = np.zeros_like(v)
    np.divide(v, denom, out=out, where=denom > 0)
    out[invalid] = 0.0
    return out


def _pixel_segments_by_cell(flat_c: np.ndarray, valid: np.ndarray) -> list[tuple[int, np.ndarray]]:
    px = np.flatnonzero(valid)
    if px.size == 0:
        return []
    keys = flat_c.ravel()[px]
    order = np.argsort(keys, kind="mergesort")
    sk = keys[order]
    breaks = np.flatnonzero(np.diff(sk)) + 1
    starts = np.r_[0, breaks]
    ends = np.r_[breaks, len(sk)]
    out: list[tuple[int, np.ndarray]] = []
    for a, b in zip(starts, ends, strict=False):
        out.append((int(sk[a]), px[order[a:b]]))
    return out


def normalize_stack_within_cells(
    cell_ids: np.ndarray,
    value_stack: np.ndarray,
    generic_fallback: np.ndarray,
    *,
    tol: float = 1e-30,
) -> tuple[np.ndarray, list[dict]]:
    """
    Normalize each stack channel to sum to one inside each parent cell.

    If a channel has zero mass in a parent cell, use ``generic_fallback`` in that cell;
    if that also sums to zero, use a uniform distribution over the cell's pixels.
    """
    h, w, n_channels = value_stack.shape
    flat_c = np.asarray(cell_ids, dtype=np.int64).ravel()
    valid = flat_c >= 0
    n_src = int(flat_c[valid].max()) + 1 if valid.any() else 0

    rho_flat = np.asarray(value_stack, dtype=np.float32).reshape(-1, n_channels)
    gen_flat = np.asarray(generic_fallback, dtype=np.float32).ravel()
    out_flat = np.zeros_like(rho_flat, dtype=np.float32)
    logs: list[dict] = []
    idx = flat_c[valid].astype(np.int64, copy=False)

    for s in range(n_channels):
        vals = rho_flat[:, s].astype(np.float64)
        den = np.zeros_like(vals)
        sums = np.bincount(idx, weights=vals[valid], minlength=n_src)
        den[valid] = sums[idx]
        ok = valid & (den > tol)
        out_flat[:, s] = np.where(ok, vals / np.maximum(den, tol), 0.0).astype(np.float32)

    segments = _pixel_segments_by_cell(flat_c, valid)
    for s in range(n_channels):
        vals = out_flat[:, s]
        for cid, inds in segments:
            blk = vals[inds].astype(np.float64)
            if float(blk.sum()) > 1e-6:
                continue
            gvals = gen_flat[inds].astype(np.float64)
            gsum = float(gvals.sum())
            if gsum > tol:
                vals[inds] = (gvals / gsum).astype(np.float32)
                logs.append({"cams_source": cid, "subsector_index": s, "fallback": "generic"})
            else:
                npx = int(inds.size)
                if npx > 0:
                    vals[inds] = np.float32(1.0 / npx)
                    logs.append({"cams_source": cid, "subsector_index": s, "fallback": "uniform"})

    return out_flat.reshape(h, w, n_channels), logs


def normalize_within_cams_cells(
    P: np.ndarray,
    cam_cell_id: np.ndarray,
    valid_mask: np.ndarray | None = None,
    *,
    chunk_elems: int = _DEFAULT_CHUNK,
    return_fallback_mask: bool = True,
    context: str = "",
    uniform_fallback_summary: list[tuple[str, int]] | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Sum-to-one fine-grid weights per CAMS cell with uniform fallback on zero-sum cells.

    Ported verbatim from ``Waste/j_waste_weights/normalization.normalize_within_cams_cells``
    (Phase 3.2 in-tree copy). Pixels with ``cam_cell_id < 0`` are excluded; when a cell's
    proxy sum is zero the result is a uniform weight over that cell's valid pixels.

    ``context`` is an optional human-readable label (e.g. ``"A_PublicPower NOx band=1"``)
    appended to the uniform-fallback warning so multi-sector build logs show which pollutant
    or band hit the fallback instead of a row of anonymous identical warnings.

    Returns ``(weights, fallback_uniform_mask)``. Weights match ``P`` shape and dtype float32.
    If ``return_fallback_mask`` is False, the second return value is ``None`` and uniform-
    fallback pixels are only counted for logging, avoiding a bool raster allocation.
    ``valid_mask`` is reserved for future use; pass ``None``.

    If ``uniform_fallback_summary`` is a list, each uniform-fallback pass appends
    ``(context, n_pixels)`` so callers can emit one INFO summary instead of many
    WARNING lines. Per-call detail is logged at DEBUG only.
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
        suffix = f" [{context}]" if context else ""
        logger.debug(
            "Uniform fallback applied on %d fine pixels (zero proxy sum in CAMS cell).%s",
            n_fb_total,
            suffix,
        )
        if uniform_fallback_summary is not None:
            uniform_fallback_summary.append((context or "normalize_within_cams_cells", int(n_fb_total)))

    fb_out: np.ndarray | None = (
        fb.reshape(out_shape) if return_fallback_mask and fb is not None else None
    )
    return work.reshape(out_shape), fb_out


def validate_parent_weight_sums_strict(
    weights: np.ndarray,
    cell_ids: np.ndarray,
    *,
    tol: float = 1e-4,
) -> list[str]:
    """Require every populated parent cell to sum to 1 for each weight band."""
    h, w, p = weights.shape
    flat_c = np.asarray(cell_ids, dtype=np.int64).ravel()
    valid = flat_c >= 0
    if not valid.any():
        return ["no CAMS parent cells on grid (cell_of all -1)"]
    n_src = int(flat_c[valid].max()) + 1
    idx = flat_c[valid].astype(np.int64, copy=False)
    counts = np.bincount(idx, minlength=n_src)
    errs: list[str] = []
    for pi in range(p):
        wp = np.asarray(weights[:, :, pi], dtype=np.float64).ravel()
        sums = np.bincount(idx, weights=wp[valid], minlength=n_src)
        for i in range(n_src):
            if int(counts[i]) == 0:
                continue
            s = float(sums[i])
            if abs(s - 1.0) > tol:
                errs.append(f"band {pi}: CAMS source {i} pixel mass sum={s} (expected 1)")
    return errs


def validate_non_negative(weights: np.ndarray, tol: float = -1e-6) -> list[str]:
    if float(np.min(weights)) < tol:
        return [f"negative weights min={float(np.min(weights))}"]
    return []


def validate_weight_sums(
    weights: np.ndarray,
    cam_cell_id: np.ndarray,
    valid_mask: np.ndarray | None = None,
    tol: float = 1e-4,
    *,
    chunk_elems: int = _DEFAULT_CHUNK,
) -> list[str]:
    """Return list of error strings if any CAMS cell's weight sum deviates from 1 by more than ``tol``.

    Ported verbatim from ``Waste/j_waste_weights/normalization.validate_weight_sums``.
    """
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
