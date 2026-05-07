"""Normalize subsector (and generic) proxies within each CAMS parent cell."""

from __future__ import annotations

import numpy as np


def _pixel_segments_by_cell(flat_c: np.ndarray, valid: np.ndarray) -> list[tuple[int, np.ndarray]]:
    """Linear indices grouped by CAMS cell id (only cells with >=1 pixel)."""
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
        cid = int(sk[a])
        inds = px[order[a:b]]
        out.append((cid, inds))
    return out


def normalize_within_cams_parents(
    cell_of: np.ndarray,
    rho_stack: np.ndarray,
    generic_rho: np.ndarray,
    *,
    tol: float = 1e-30,
) -> tuple[np.ndarray, list[dict]]:
    """
    rho_stack: (H,W,S). generic_rho: (H,W).

    Returns rho_norm (H,W,S) where for each CAMS source index i and s,
    sum_{c in i} rho_norm[c,s] = 1 when denominator positive; else generic then uniform.
    """
    h, w, S = rho_stack.shape
    flat_c = cell_of.ravel()
    valid = flat_c >= 0
    n_src = int(flat_c[valid].max()) + 1 if valid.any() else 0

    rho_flat = rho_stack.reshape(-1, S)
    gen_flat = generic_rho.ravel()
    out_flat = np.zeros_like(rho_flat, dtype=np.float32)
    logs: list[dict] = []

    idx = flat_c[valid].astype(np.int64, copy=False)

    for s in range(S):
        vals = rho_flat[:, s].astype(np.float64)
        den = np.zeros_like(vals)
        sums = np.bincount(idx, weights=vals[valid], minlength=n_src)
        den[valid] = sums[idx]
        ok = valid & (den > tol)
        out_flat[:, s] = np.where(ok, vals / np.maximum(den, tol), 0.0).astype(np.float32)

    segments = _pixel_segments_by_cell(flat_c, valid)

    for s in range(S):
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

    return out_flat.reshape(h, w, S), logs
