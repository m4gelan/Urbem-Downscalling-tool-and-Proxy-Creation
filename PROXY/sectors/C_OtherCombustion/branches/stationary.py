"""Stationary branch: same normalisation as ``U = X @ M.T`` per pollutant."""

from __future__ import annotations

import numpy as np


def compute_W_stat(U: np.ndarray, pollutant_outputs: list[str]) -> tuple[dict[str, np.ndarray], bool]:
    """
    Per-pollutant shares over pixels (columns of ``U``), same rules as allocator.

    Returns
    -------
    weights
        Map output name -> length ``n_pix`` weights summing to 1 (or uniform fallback).
    used_fallback
        True if any column used uniform fallback.
    """
    n_pix = int(U.shape[0])
    used_fb = False
    out: dict[str, np.ndarray] = {}
    for pi, name in enumerate(pollutant_outputs):
        u_col = U[:, pi]
        ssum = float(np.sum(u_col))
        if ssum <= 0.0 or not np.isfinite(ssum):
            w_col = np.full(n_pix, 1.0 / max(n_pix, 1), dtype=np.float64)
            used_fb = True
        else:
            w_col = (u_col / ssum).astype(np.float64, copy=False)
        out[str(name)] = w_col
    return out, used_fb
