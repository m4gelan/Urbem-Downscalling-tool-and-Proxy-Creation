"""Commercial / institutional off-road: CLC 121 + λ × OSM mask."""

from __future__ import annotations

import numpy as np


def compute_W_B(
    clc_l3: np.ndarray,
    osm_hit: np.ndarray,
    *,
    commercial_clc: int,
    lambda_osm: float,
) -> tuple[np.ndarray, bool]:
    flat_c = np.asarray(clc_l3, dtype=np.float64).ravel()
    o = np.asarray(osm_hit, dtype=np.float64).ravel()
    if o.size != flat_c.size:
        raise ValueError("osm_hit must match clc_l3 size")
    r = np.zeros_like(flat_c, dtype=np.int32)
    finite = np.isfinite(flat_c)
    r[finite] = np.rint(flat_c[finite]).astype(np.int32, copy=False)
    clc_hit = finite & (r == int(commercial_clc))
    w = clc_hit.astype(np.float64) + float(lambda_osm) * np.clip(o, 0.0, 1.0)
    s = float(np.sum(w))
    n = int(w.size)
    if s <= 0.0:
        return np.full(n, 1.0 / max(n, 1), dtype=np.float64), True
    return (w / s).astype(np.float64, copy=False), False
