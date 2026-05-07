"""
Rural bias multiplier from population or GHSL settlement layer.

``rural_bias = clip(1 - smooth_normalize(density), rural_min, 1)`` boosts fireplace /
heating-stove proxies in lower-density pixels when enabled.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def _uniform_smooth(x: np.ndarray, win: int = 5) -> np.ndarray:
    """Mean filter with edge padding (odd window)."""
    if win < 3 or win % 2 == 0:
        win = 5
    pad = win // 2
    xp = np.pad(x, pad, mode="edge")
    sw = sliding_window_view(xp, (win, win))
    return sw.mean(axis=(-2, -1)).astype(np.float32)


def smooth_normalize(density: np.ndarray) -> np.ndarray:
    """Log1p + min–max to [0,1] + light smoothing on valid positive cells."""
    d = np.maximum(np.asarray(density, dtype=np.float64), 0.0)
    z = np.log1p(d)
    finite = np.isfinite(z) & (z > 0)
    if not np.any(finite):
        return np.zeros_like(density, dtype=np.float32)
    lo, hi = float(np.min(z[finite])), float(np.max(z[finite]))
    if hi <= lo:
        u = np.where(finite, 0.5, 0.0).astype(np.float32)
    else:
        u = np.clip((z - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
        u[~finite] = 0.0
    return _uniform_smooth(u, 5)


def rural_bias_from_density(density: np.ndarray, *, rural_min: float) -> np.ndarray:
    s = smooth_normalize(density)
    rb = 1.0 - s
    return np.clip(rb, float(rural_min), 1.0).astype(np.float32)
