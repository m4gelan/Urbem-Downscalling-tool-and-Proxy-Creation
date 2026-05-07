"""Mass conservation and sanity checks for solvent area weights."""

from __future__ import annotations

import numpy as np


def check_within_cell_mass(
    W: np.ndarray,
    cell_of: np.ndarray,
    *,
    tol: float = 1e-4,
) -> list[str]:
    """
    W: (H, W, P). For each pollutant band, sums over fine pixels in each CAMS
    parent index should be 1 (sources that appear on the grid).
    """
    h, w, p = W.shape
    flat_c = cell_of.ravel()
    valid = flat_c >= 0
    if not valid.any():
        return ["no CAMS parent cells on grid (cell_of all -1)"]
    n_src = int(flat_c[valid].max()) + 1
    idx = flat_c[valid].astype(np.int64, copy=False)
    counts = np.bincount(idx, minlength=n_src)
    errs: list[str] = []
    for pi in range(p):
        wp = W[:, :, pi].ravel().astype(np.float64)
        sums = np.bincount(idx, weights=wp[valid], minlength=n_src)
        for i in range(n_src):
            if int(counts[i]) == 0:
                continue
            s = float(sums[i])
            if abs(s - 1.0) > tol:
                errs.append(
                    f"band {pi}: CAMS source {i} pixel mass sum={s} (expected 1)"
                )
    return errs


def check_non_negative(W: np.ndarray, tol: float = -1e-6) -> list[str]:
    if float(W.min()) < tol:
        return [f"negative weights min={float(W.min())}"]
    return []
