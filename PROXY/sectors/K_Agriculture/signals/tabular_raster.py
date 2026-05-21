"""Rasterise NUTS2×CLC tabular columns onto the agriculture reference grid."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from PROXY.sectors.K_Agriculture.corine_weight_codes import corine_grid_to_weight_codes


def tabular_column_to_raster(
    df: pd.DataFrame,
    *,
    value_col: str,
    nuts_r: np.ndarray,
    corine_arr: np.ndarray,
    nuts_to_idx: dict[str, int],
    nodata: float | None = None,
) -> np.ndarray:
    """
    Map (NUTS_ID, CLC_CODE) rows to pixels: raw[j] = lookup[nuts_r[j], wclc[j]].

    ``df`` must contain NUTS_ID, CLC_CODE, and ``value_col``.
    ``corine_arr`` is raw CORINE class values (level 3 or 12–22).
    """
    h, w = nuts_r.shape
    rint = np.zeros_like(corine_arr, dtype=np.int32)
    ok = np.isfinite(corine_arr)
    if nodata is not None:
        ok = ok & (corine_arr != float(nodata))
    rint[ok] = np.rint(corine_arr[ok]).astype(np.int32)
    wclc = corine_grid_to_weight_codes(rint)

    n_nuts = len(nuts_to_idx) + 1
    lookup_m = np.zeros((n_nuts, 23), dtype=np.float64)
    for _, r in df.iterrows():
        nid = str(r["NUTS_ID"]).strip()
        idx = nuts_to_idx.get(nid)
        if idx is None:
            continue
        clc = int(r["CLC_CODE"])
        if 0 <= clc <= 22:
            lookup_m[idx, clc] = float(r[value_col])

    raw = np.zeros((h, w), dtype=np.float64)
    valid_nuts = (nuts_r >= 1) & (nuts_r < n_nuts)
    valid = valid_nuts & (wclc >= 0)
    raw[valid] = lookup_m[nuts_r[valid], wclc[valid]]
    return raw.astype(np.float32)
