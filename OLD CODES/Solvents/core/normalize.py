"""Normalize fine-grid indicator arrays to [0, 1] for archetype mixing."""

from __future__ import annotations

import numpy as np


def normalize_indicator(
    arr: np.ndarray,
    *,
    method: str = "quantile_minmax",
    q_low: float = 0.01,
    q_high: float = 0.99,
) -> np.ndarray:
    """
    Map non-negative (or any) finite values to [0,1].

    quantile_minmax: clip to [q_low, q_high] quantiles of finite positives,
    then min-max to [0,1]; NaN and non-finite -> 0.
    """
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
    lo32 = np.float32(lo)
    hi32 = np.float32(hi)
    y = np.clip(x, lo32, hi32, out=np.empty_like(x))
    y = (y - lo32) / (hi32 - lo32)
    np.nan_to_num(y, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.maximum(y, 0.0, out=y)
    return y
