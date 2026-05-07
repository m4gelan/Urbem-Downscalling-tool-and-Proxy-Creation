"""Internal helpers shared across the alpha subpackage (parsing, YAML, value coercion)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from PROXY.core.alpha.reported import normalize_inventory_sector
from PROXY.core.dataloaders.config import load_yaml as _load_yaml


def _norm_token(value: Any) -> str:
    return normalize_inventory_sector(value)


def _coerce_total(value: Any) -> float | None:
    if value is None:
        return None
    txt = str(value).strip()
    if not txt:
        return None
    try:
        return float(txt)
    except ValueError:
        return None


def _pick_col(df: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    cols = {str(c).lower().strip(): c for c in df.columns}
    for n in names:
        if n in cols:
            return cols[n]
    return None


def _parse_emission_value(v: Any) -> float:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return float("nan")
    s = str(v).strip().upper()
    if s in ("NA", "N/A", "C", "-", "", "NAN", "NO", "NE", "IE"):
        return float("nan")
    try:
        return float(str(v).strip().replace(",", "."))
    except ValueError:
        return float("nan")


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))
