"""Pollutant weights W_{c,p} = sum_s alpha_{p,s} * rho_norm_{c,s}."""

from __future__ import annotations

import numpy as np
import pandas as pd


def alpha_matrix(
    agg: pd.DataFrame,
    pollutants: list[str],
    subsectors: list[str],
) -> np.ndarray:
    """Shape (P, S) float64; missing entries are 0."""
    P, S = len(pollutants), len(subsectors)
    mat = np.zeros((P, S), dtype=np.float64)
    pol_row = {str(p).upper(): i for i, p in enumerate(pollutants)}
    sub_ix = {s: j for j, s in enumerate(subsectors)}
    for _, row in agg.iterrows():
        p_key = str(row["pollutant"]).strip().upper()
        ri = pol_row.get(p_key)
        if ri is None:
            continue
        s = str(row["subsector"])
        if s not in sub_ix:
            continue
        v = float(row["alpha"])
        if np.isfinite(v):
            mat[ri, sub_ix[s]] = v
    return mat


def finalize_alpha_matrix(
    alpha: np.ndarray,
    subsectors: list[str],
    *,
    tol: float = 1e-12,
) -> np.ndarray:
    """
    Per pollutant (row):

    - If there is **no** reported mass (row sum ~ 0): use **uniform** ``1/S`` over all
      ``S = len(subsectors)`` CEIP subsectors (e.g. no CH4 in the sheet at all).

    - If there **is** mass on a **subset** of subsectors only: keep **zeros** on the
      rest and **renormalize** the row so the positive entries sum to 1 (proportional
      to CEIP emissions on those sectors only).

    This replaces copying another pollutant's mix (e.g. NMVOC) onto missing species.
    """
    a = np.asarray(alpha, dtype=np.float64).copy()
    P, S = a.shape
    if P == 0 or S == 0:
        return a
    if len(subsectors) != S:
        raise ValueError(f"subsectors length {len(subsectors)} != alpha columns {S}")
    uniform = np.full(S, 1.0 / float(S), dtype=np.float64)
    for pi in range(P):
        row = np.nan_to_num(a[pi], nan=0.0, posinf=0.0, neginf=0.0)
        row = np.maximum(row, 0.0)
        s = float(row.sum())
        if s <= tol:
            a[pi, :] = uniform
        else:
            a[pi, :] = row / s
    return a


def allocate_weights(rho_norm: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """
    rho_norm: (H, W, S), alpha: (P, S).
    Returns W: (H, W, P) float32.
    """
    h, w, _s = rho_norm.shape
    p = int(alpha.shape[0])
    out = np.zeros((h, w, p), dtype=np.float32)
    a = np.asarray(alpha, dtype=np.float32)
    for pi in range(p):
        out[:, :, pi] = np.sum(rho_norm * a[pi], axis=2, dtype=np.float32)
    return out
