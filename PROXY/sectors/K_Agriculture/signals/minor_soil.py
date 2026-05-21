"""Family 6: uniform minor-soil proxy over configured CORINE classes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from PROXY.sectors.K_Agriculture.combined_config import load_k_agriculture_rules


def _raw_minor(cfg: dict[str, Any], root: Path) -> list[int]:
    doc = load_k_agriculture_rules(cfg, root)
    codes = (doc.get("minor_soil") or {}).get("raw_clc_codes") or []
    if codes:
        return [int(x) for x in codes]
    return [211, 212, 213, 221, 222, 223]


def build_family6(
    cfg: dict[str, Any],
    root: Path,
    *,
    corine_arr: np.ndarray,
    corine_nodata: float | None,
) -> np.ndarray:
    raw_list = _raw_minor(cfg, root)
    x = np.asarray(corine_arr, dtype=np.float64)
    ok = np.isfinite(x)
    if corine_nodata is not None:
        ok = ok & (x != float(corine_nodata))
    ri = np.rint(x).astype(np.int32)
    out = np.zeros_like(x, dtype=np.float32)
    for gv in raw_list:
        m = ok & (ri == int(gv))
        out[m] = 1.0
    return out
