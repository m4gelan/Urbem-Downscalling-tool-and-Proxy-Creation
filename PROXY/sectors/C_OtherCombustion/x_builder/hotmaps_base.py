"""
Residential vs non-residential **Hotmaps** intensity fields on the reference grid.

**Outputs**: ``R_base`` (res heat × res GFA proxy) and ``C_base`` (non-res heat × non-res GFA),
using either multiplicative exponents or weighted additive blend from sector YAML.
"""

from __future__ import annotations

import numpy as np


def combine_base(
    heat: np.ndarray,
    gfa: np.ndarray,
    *,
    heat_exp: float,
    gfa_exp: float,
    use_additive: bool,
    add_w_heat: float,
    add_w_gfa: float,
    epsilon: float,
) -> np.ndarray:
    H = np.maximum(heat.astype(np.float64), 0.0)
    G = np.maximum(gfa.astype(np.float64), 0.0)
    if use_additive:
        return (add_w_heat * H + add_w_gfa * G).astype(np.float32)
    e = max(float(epsilon), 1e-30)
    R = np.power(H + e, float(heat_exp)) * np.power(G + e, float(gfa_exp))
    if not np.all(np.isfinite(R)):
        R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
    return R.astype(np.float32)
