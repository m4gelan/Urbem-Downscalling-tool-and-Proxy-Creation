from __future__ import annotations

import numpy as np

from proxy.core.area_weights import normalize_W_per_cams_cell


def stationary_weight_band(
    X: np.ndarray,
    M_row: np.ndarray,
    cell_id: np.ndarray,
    cams_cells: dict,
) -> np.ndarray:
    """One pollutant: U = X @ M_row then CAMS-normalise (single h×w plane, no 3-D stack)."""
    U_p = np.einsum("hwk,k->hw", np.asarray(X, dtype=np.float32), np.asarray(M_row, dtype=np.float32), dtype=np.float32)
    return normalize_W_per_cams_cell(U_p, cell_id, cams_cells)
