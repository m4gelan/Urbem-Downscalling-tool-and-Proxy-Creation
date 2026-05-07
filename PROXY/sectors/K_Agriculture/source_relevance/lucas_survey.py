"""Shared LUCAS survey string normalization and INSPIRE percentage parsing."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def norm_lucas_str(s: Any) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    t = str(s).strip().strip('"').strip("'").upper()
    t = " ".join(t.split())
    if t in ("NAN", "NONE", "NA", "#N/A", "NAT"):
        return ""
    return t


def inspire_pct(x: Any) -> float:
    """INSPIRE cover share 0..100; missing or invalid -> NaN."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return float("nan")
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(v):
        return float("nan")
    return max(0.0, min(100.0, v))


def inspire_pct_or(x: Any, default: float) -> float:
    """Use default when missing (for fallbacks only)."""
    v = inspire_pct(x)
    if np.isnan(v):
        return default
    return v
