"""Alpha matrix normalization helpers.

Inputs are pollutant-by-subsector alpha arrays from CEIP/workbook loaders.
Outputs are finite, non-negative rows normalized to sum to one, with a uniform
fallback only for rows that have no positive mass.
"""

from __future__ import annotations

import numpy as np


def finalize_alpha_matrix(
    alpha: np.ndarray,
    labels: list[str],
    *,
    tol: float = 1e-12,
) -> np.ndarray:
    """Normalize each pollutant row of an ``(P, S)`` alpha matrix."""
    a = np.asarray(alpha, dtype=np.float64).copy()
    p, s = a.shape
    if p == 0 or s == 0:
        return a
    if len(labels) != s:
        raise ValueError(f"labels length {len(labels)} != alpha columns {s}")
    uniform = np.full(s, 1.0 / float(s), dtype=np.float64)
    for pi in range(p):
        row = np.nan_to_num(a[pi], nan=0.0, posinf=0.0, neginf=0.0)
        row = np.maximum(row, 0.0)
        total = float(row.sum())
        if total <= tol:
            a[pi, :] = uniform
        else:
            a[pi, :] = row / total
    return a


__all__ = ["finalize_alpha_matrix"]
