"""Family 5: crop operations — NMVOC from LUCAS table; PM from EMEP totals × Köppen regime."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from PROXY.sectors.K_Agriculture.combined_config import load_k_agriculture_rules
from PROXY.sectors.K_Agriculture.signals.tabular_raster import tabular_column_to_raster
from PROXY.sectors.K_Agriculture.source_relevance import agricultural_soils


def _arable_mask(corine_arr: np.ndarray, corine_nodata: float | None, arable_codes: list[int]) -> np.ndarray:
    x = np.asarray(corine_arr, dtype=np.float64)
    ok = np.isfinite(x)
    if corine_nodata is not None:
        ok = ok & (x != float(corine_nodata))
    ri = np.rint(x).astype(np.int32)
    m = np.zeros(x.shape, dtype=bool)
    for c in arable_codes:
        m |= ok & (ri == int(c))
    return m


def build_family5(
    extent_df: pd.DataFrame,
    cfg: dict[str, Any],
    root: Path,
    *,
    nuts_r: np.ndarray,
    corine_arr: np.ndarray,
    nuts_to_idx: dict[str, int],
    corine_nodata: float | None,
    gamma_series: pd.Series,
) -> dict[str, np.ndarray]:
    rules = load_k_agriculture_rules(cfg, root)
    crop = rules.get("crop_operations") or {}
    arable_codes = [int(x) for x in (crop.get("arable_raw_clc_codes") or [211, 212, 213, 221, 222, 223])]
    emep = (crop.get("emep_totals") or {})
    row_key = str((crop.get("group_emep_row") or {}).get("g1_nonfixing_cereals", "wheat"))

    rho = agricultural_soils.compute_rho_df(extent_df, cfg)
    nmvoc = tabular_column_to_raster(
        rho,
        value_col="mu",
        nuts_r=nuts_r,
        corine_arr=corine_arr,
        nuts_to_idx=nuts_to_idx,
        nodata=corine_nodata,
    )

    h, w = nmvoc.shape
    n_nuts = len(nuts_to_idx) + 1
    idx_to_nuts = {int(v): str(k).strip() for k, v in nuts_to_idx.items()}
    emep = (crop.get("emep_totals") or {})
    row_key = str((crop.get("group_emep_row") or {}).get("g1_nonfixing_cereals", "wheat"))
    ef10_nuts = np.zeros(n_nuts, dtype=np.float32)
    ef25_nuts = np.zeros(n_nuts, dtype=np.float32)
    for idx in range(1, n_nuts):
        nid = idx_to_nuts.get(idx)
        if nid is None:
            continue
        regime = str(gamma_series.get(str(nid).strip().upper(), "wet")).lower()
        block = (emep.get(regime) or emep.get("wet") or {})
        row = (block.get(row_key) or block.get("wheat") or {})
        ef10_nuts[idx] = float(row.get("PM10", 3.7))
        ef25_nuts[idx] = float(row.get("PM2.5", 0.2))
    ni = np.clip(np.asarray(nuts_r, dtype=np.int32), 0, n_nuts - 1)
    ef10 = ef10_nuts[ni]
    ef25 = ef25_nuts[ni]
    mask = _arable_mask(corine_arr, corine_nodata, arable_codes).astype(np.float32)
    pm10 = mask * ef10
    pm25 = mask * ef25
    mx = float(nmvoc.max()) + 1e-12
    nmvoc_n = (nmvoc / mx).astype(np.float32)
    return {
        "NMVOC": nmvoc_n,
        "PM10": pm10.astype(np.float32),
        "PM2.5": pm25.astype(np.float32),
    }
