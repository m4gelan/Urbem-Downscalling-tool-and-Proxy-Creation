from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds

from ..io.paths import resolve_path as project_resolve


def read_corine_window(
    root: Path,
    cfg: dict[str, Any],
    ref: dict,
) -> np.ndarray:
    """Integer CLC codes on ref window (H,W)."""
    corine_path = ref["corine_path"]
    if not Path(corine_path).is_absolute():
        corine_path = project_resolve(root, Path(corine_path))
    band = int((cfg.get("corine") or {}).get("band", 1))
    transform = ref["transform"]
    h, w = int(ref["height"]), int(ref["width"])
    left, bottom, right, top = (float(x) for x in ref["window_bounds_3035"])
    with rasterio.open(corine_path) as src:
        win = from_bounds(left, bottom, right, top, transform=src.transform)
        win = win.round_lengths().intersection(Window(0, 0, src.width, src.height))
        arr = src.read(band, window=win).astype(np.float64)
        cn = src.nodata
    if arr.shape != (h, w):
        raise ValueError(f"CORINE window {arr.shape} != ref {(h, w)}")
    z = np.rint(arr).astype(np.int32)
    if cn is not None:
        z = np.where(arr == float(cn), -9999, z)
    return z


def clc_group_masks(
    clc: np.ndarray,
    code_groups: dict[str, list[int]],
) -> dict[str, np.ndarray]:
    """Per-key binary float32 mask (1 where CLC in list)."""
    out: dict[str, np.ndarray] = {}
    for name, codes in code_groups.items():
        m = np.zeros_like(clc, dtype=np.float32)
        for c in codes:
            m = np.maximum(m, (clc == int(c)).astype(np.float32))
        out[name] = m
    return out
