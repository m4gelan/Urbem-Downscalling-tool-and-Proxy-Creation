"""CORINE raster values vs CLC_CODE in agriculture weights_long (12–22)."""

from __future__ import annotations

import numpy as np

# Official CLC2018 level-3 classes aggregated to the 12–22 scheme used in weights_long.csv
CLC_LEVEL3_TO_WEIGHT_CODE: dict[int, int] = {
    211: 12,
    212: 13,
    213: 14,
    221: 15,
    222: 16,
    223: 17,
    231: 18,
    241: 19,
    242: 20,
    243: 21,
    244: 22,
}


def corine_grid_to_weight_codes(clc: np.ndarray) -> np.ndarray:
    """
    Map raw CORINE class integers to weights CLC_CODE (12–22), or -1 if not an ag class.

    Pass-through: pixels already in 12–22 (e.g. pre-aggregated rasters) are unchanged.
    """
    x = np.asarray(clc, dtype=np.int64)
    out = np.full(x.shape, -1, dtype=np.int32)
    m1222 = (x >= 12) & (x <= 22)
    out[m1222] = x[m1222].astype(np.int32)
    for gv, wc in CLC_LEVEL3_TO_WEIGHT_CODE.items():
        out[x == gv] = np.int32(wc)
    return out
