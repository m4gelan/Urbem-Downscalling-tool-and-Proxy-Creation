"""Generic alpha allocation helpers.

This module combines normalized subsector proxy stacks with pollutant alpha
matrices. Alpha matrix normalization lives in `PROXY.core.alpha.matrix`; CAMS
cell-id construction lives in `PROXY.core.cams.cell_id`.
"""

from __future__ import annotations

import numpy as np


def allocate_weights_from_normalized_stack(rho_norm: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """
    Combine normalized spatial shares ``(H, W, S)`` with alpha ``(P, S)`` into ``(H, W, P)``.
    """
    h, w, _s = rho_norm.shape
    p = int(alpha.shape[0])
    out = np.zeros((h, w, p), dtype=np.float32)
    a = np.asarray(alpha, dtype=np.float32)
    for pi in range(p):
        out[:, :, pi] = np.sum(rho_norm * a[pi], axis=2, dtype=np.float32)
    return out


def normalize_subsectors_within_cells(
    *,
    subsector_arrays: dict[str, np.ndarray],
    cams_cell_id: np.ndarray,
    subsector_shares: dict[str, float],
) -> dict[str, np.ndarray]:
    keys = list(subsector_arrays.keys())
    if not keys:
        return {}
    shape = cams_cell_id.shape
    for k in keys:
        if subsector_arrays[k].shape != shape:
            raise ValueError(f"Subsector array shape mismatch for {k}")

    stack = np.stack([np.maximum(subsector_arrays[k], 0.0) for k in keys], axis=0).astype(np.float64)
    weights = np.array([float(subsector_shares.get(k, 1.0)) for k in keys], dtype=np.float64)
    weighted = stack * weights[:, None, None]
    out = np.zeros_like(weighted, dtype=np.float32)

    valid_cells = np.unique(cams_cell_id[cams_cell_id >= 0])
    for cid in valid_cells.tolist():
        m = cams_cell_id == int(cid)
        vals = weighted[:, m]
        denom = vals.sum(axis=0)
        ok = denom > 0.0
        if np.any(ok):
            out[:, m][:, ok] = (vals[:, ok] / denom[ok]).astype(np.float32)
    return {k: out[i] for i, k in enumerate(keys)}

