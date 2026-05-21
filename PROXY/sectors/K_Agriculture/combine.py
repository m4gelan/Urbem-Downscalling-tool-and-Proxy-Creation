"""Linear combine of family rasters with CEIP-derived alpha (one vector per pollutant)."""

from __future__ import annotations

import numpy as np


def combine_family_rasters(
    *,
    pollutant: str,
    alpha_vec: np.ndarray,
    group_order: tuple[str, ...],
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    p4: np.ndarray,
    p5_by_pol: dict[str, np.ndarray] | None,
    p6: np.ndarray,
    p7: np.ndarray,
) -> np.ndarray:
    """S_p,j = sum_g alpha_g * P_g,j (non-negative). ``alpha_vec`` aligns with ``group_order``."""
    h, w = p1.shape
    v = np.asarray(alpha_vec, dtype=np.float64).ravel()
    if v.size != len(group_order):
        raise ValueError(
            f"alpha_vec length {v.size} != len(group_order)={len(group_order)} "
            f"for pollutant={pollutant!r}"
        )
    weights = {str(group_order[i]).strip(): float(v[i]) for i in range(len(group_order))}
    acc = np.zeros((h, w), dtype=np.float64)
    fam_arrays = {
        "family_1": p1,
        "family_2": p2,
        "family_3": p3,
        "family_4": p4,
        "family_6": p6,
        "family_7": p7,
    }
    for fk, arr in fam_arrays.items():
        a = float(weights.get(fk, 0.0))
        if a == 0.0:
            continue
        acc += a * np.maximum(np.asarray(arr, dtype=np.float64), 0.0)

    p5bp = p5_by_pol or {}
    pol_u = str(pollutant).strip().upper()
    a5 = float(weights.get("family_5", 0.0))
    if a5 != 0.0:
        if pol_u == "NMVOC" and "NMVOC" in p5bp:
            acc += a5 * np.maximum(np.asarray(p5bp["NMVOC"], dtype=np.float64), 0.0)
        elif pol_u == "PM10" and "PM10" in p5bp:
            acc += a5 * np.maximum(np.asarray(p5bp["PM10"], dtype=np.float64), 0.0)
        elif pol_u in ("PM2.5", "PM2_5") and "PM2.5" in p5bp:
            acc += a5 * np.maximum(np.asarray(p5bp["PM2.5"], dtype=np.float64), 0.0)

    return acc.astype(np.float32)
