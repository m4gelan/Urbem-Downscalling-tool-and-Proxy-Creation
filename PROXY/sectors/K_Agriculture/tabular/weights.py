"""Normalize agriculture scores into NUTS-local CLC weights."""
from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_weights(df: pd.DataFrame, score_col: str, weight_col: str) -> pd.DataFrame:
    out = df.copy()

    def _norm(s: pd.Series) -> pd.Series:
        tot = float(s.sum(skipna=True))
        if tot > 0 and np.isfinite(tot):
            return s / tot
        return s * np.nan

    out[weight_col] = out.groupby("NUTS_ID")[score_col].transform(_norm)
    return out
