from __future__ import annotations

import numpy as np

from PROXY_V2.core.area_weights import normalize_W_per_cams_cell
from PROXY_V2.core import log


def assemble_U(X: np.ndarray, M: np.ndarray) -> np.ndarray:
    """U[h,w,p] = sum_k X[h,w,k] * M[p,k]."""
    Xf = np.asarray(X, dtype=np.float32)
    Mf = np.asarray(M, dtype=np.float32)
    return np.einsum("hwk,pk->hwp", Xf, Mf, dtype=np.float32)


def normalize_U_per_cams_cell(
    U: np.ndarray,
    cell_id: np.ndarray,
    cams_cells: dict,
) -> np.ndarray:
    """Per CAMS cell and pollutant: W = U / sum(U) inside the cell (in-place on *U*)."""
    for pi in range(U.shape[2]):
        U[:, :, pi] = normalize_W_per_cams_cell(U[:, :, pi], cell_id, cams_cells)
    return U


def log_U_summary(U: np.ndarray, pollutant_outputs: list[str]) -> None:
    pols = [str(p).strip() for p in pollutant_outputs if str(p).strip()]
    log.info("--- U = X @ M.T (before CAMS normalization) ---")
    for pi, pol in enumerate(pols):
        band = U[:, :, pi]
        m = np.isfinite(band) & (band > 0)
        if np.any(m):
            log.info(
                f"  {pol}: min={float(band[m].min()):.6g} max={float(band[m].max()):.6g} "
                f"sum={float(band[m].sum()):.6g}"
            )
        else:
            log.info(f"  {pol}: (all zero)")
