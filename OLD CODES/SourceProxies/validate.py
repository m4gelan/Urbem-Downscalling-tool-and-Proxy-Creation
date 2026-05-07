"""Sanity checks for area proxy weights (CAMS-cell sums ~ 1)."""

from __future__ import annotations

import sys
from typing import Any

import numpy as np


def check_public_power_area_gdf(gdf: Any, *, tol: float = 1e-4) -> list[tuple[int, float]]:
    """Return (cams_source_index, sum_weight_share) for cells that deviate from 1."""
    if gdf is None or getattr(gdf, "empty", True):
        return []
    bad: list[tuple[int, float]] = []
    for sid, grp in gdf.groupby("cams_source_index", sort=False):
        s = float(grp["weight_share"].sum())
        if abs(s - 1.0) > tol:
            bad.append((int(sid), s))
    return bad


def check_agriculture_raster(
    out: np.ndarray,
    cell_of: np.ndarray,
    m_kl: np.ndarray,
    *,
    tol: float = 2e-3,
) -> list[tuple[int, float]]:
    """Per CAMS K/L area source, sum of raster values over assigned pixels should be ~1."""
    n_src = int(m_kl.size)
    flat = np.asarray(out, dtype=np.float64).ravel()
    fc = np.asarray(cell_of, dtype=np.int64).ravel()
    m = fc >= 0
    if not np.any(m):
        return []
    ii = fc[m].astype(np.int64, copy=False)
    sums = np.bincount(ii, weights=flat[m], minlength=n_src)
    counts = np.bincount(ii, minlength=n_src)
    bad: list[tuple[int, float]] = []
    for i in np.flatnonzero(m_kl):
        idx = int(i)
        if counts[idx] == 0:
            continue
        s = float(sums[idx])
        if abs(s - 1.0) > tol:
            bad.append((idx, s))
    return bad


def report_validation(name: str, bad: list[tuple[int, float]], *, max_show: int = 12) -> None:
    if not bad:
        print(f"validate: {name} OK (all checked cells sum ~ 1).", file=sys.stderr)
        return
    show = bad[:max_show]
    print(
        f"validate: {name} — {len(bad)} cell(s) off tolerance (showing {len(show)}):",
        file=sys.stderr,
    )
    for sid, s in show:
        print(f"  cams_source_index={sid} sum={s:.6f}", file=sys.stderr)
