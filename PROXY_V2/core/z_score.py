from __future__ import annotations

import numpy as np

from PROXY_V2.core import log


def z_score_inside(
    arr: np.ndarray,
    inside: np.ndarray,
    *,
    upper_quantile: float = 0.99,
    rescale_to_01: bool = True,
) -> np.ndarray:
    """
    Quantile-clipped z-score on pixels where *inside* is True, optional map to [0, 1].

    1. Cap values at the *upper_quantile* of in-mask pixels (default 99th %).
    2. Z-score on masked pixels: (clipped - mean) / std.
    3. If *rescale_to_01*, linearly rescale z on masked pixels to [0, 1]
       (0 = min z inside, 1 = z at the quantile cap).
    Pixels outside the effective mask (``inside`` and finite ``arr``) are set to 0.
    Non-finite values (e.g. NaN population after warping) are excluded from statistics.
    """
    if arr.shape != inside.shape:
        raise ValueError("arr and inside must have the same shape")

    inside = np.asarray(inside, dtype=bool)
    x = arr.astype(np.float64)
    valid = inside & np.isfinite(x)
    out = np.zeros(arr.shape, dtype=np.float32)

    n_inside = int(valid.sum())
    if n_inside == 0:
        log.warning("z_score_inside: no pixels inside mask")
        return out

    vals = x[valid]
    q = float(upper_quantile)
    p_q = float(np.quantile(vals, q))
    clipped = np.minimum(x, p_q)

    mu = float(clipped[valid].mean())
    sigma = float(clipped[valid].std())
    if sigma == 0.0:
        out[valid] = 1.0 if rescale_to_01 else 0.0
        log.info(
            f"z_score_inside: flat field (sigma=0), mu={mu:.4g}, p{q:.0%}={p_q:.4g}"
        )
        return out

    z = (clipped - mu) / sigma

    if not rescale_to_01:
        out[valid] = z[valid].astype(np.float32)
        log.info(
            f"z_score_inside: q={q}, p_q={p_q:.4g}, mu={mu:.4g}, sigma={sigma:.4g}"
        )
        return out

    z_min = float(z[valid].min())
    z_max = float((p_q - mu) / sigma)
    denom = z_max - z_min
    if denom <= 0.0:
        out[valid] = 1.0
    else:
        out[valid] = ((z[valid] - z_min) / denom).astype(np.float32)

    log.info(
        f"z_score_inside: q={q}, p_q={p_q:.4g}, mu={mu:.4g}, sigma={sigma:.4g}, "
        f"rescaled to [0,1] over {n_inside} pixels"
    )
    return out
