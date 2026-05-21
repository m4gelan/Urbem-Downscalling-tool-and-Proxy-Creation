"""Family 4: synthetic N — tabular mu rasterised."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from PROXY.sectors.K_Agriculture.signals.tabular_raster import tabular_column_to_raster
from PROXY.sectors.K_Agriculture.source_relevance import fertilized_land


def build_family4(
    extent_df: pd.DataFrame,
    cfg: dict[str, Any],
    *,
    nuts_r: np.ndarray,
    corine_arr: np.ndarray,
    nuts_to_idx: dict[str, int],
    corine_nodata: float | None,
) -> np.ndarray:
    rho = fertilized_land.compute_rho_df(extent_df, cfg)
    return tabular_column_to_raster(
        rho,
        value_col="mu",
        nuts_r=nuts_r,
        corine_arr=corine_arr,
        nuts_to_idx=nuts_to_idx,
        nodata=corine_nodata,
    )
