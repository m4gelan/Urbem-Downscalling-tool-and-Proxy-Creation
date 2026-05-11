"""Forestry off-road: weight ∝ 1{CLC in forest classes}."""

from __future__ import annotations

import numpy as np


def compute_W_F(clc_l3: np.ndarray, forest_codes: frozenset[int]) -> tuple[np.ndarray, bool]:
    """
    Flattened window weights summing to 1. Uniform fallback if no forest pixel.
    """
    flat = np.asarray(clc_l3, dtype=np.float64).ravel()
    r = np.zeros_like(flat, dtype=np.int32)
    finite = np.isfinite(flat)
    r[finite] = np.rint(flat[finite]).astype(np.int32, copy=False)
    m = finite & np.isin(r, list(forest_codes))
    w = m.astype(np.float64)
    s = float(np.sum(w))
    n = int(flat.size)
    if s <= 0.0 or n == 0:
        return np.full(n, 1.0 / max(n, 1), dtype=np.float64), True
    return (w / s).astype(np.float64, copy=False), False
