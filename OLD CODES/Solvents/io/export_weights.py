"""Write multi-band GeoTIFF of pollutant allocation weights."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio


def write_solvents_area_weights(
    out_path: Path,
    W: np.ndarray,
    ref: dict[str, Any],
    pollutants: list[str],
    *,
    meta: dict[str, Any] | None = None,
) -> Path:
    """
    W: (H, W, P) float32. Writes GTiff LZW, band descriptions per pollutant.
    Optional JSON sidecar with same stem + _meta.json.
    """
    h, w, p = W.shape
    if p != len(pollutants):
        raise ValueError(f"W bands {p} != len(pollutants) {len(pollutants)}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": p,
        "dtype": "float32",
        "crs": ref["crs"],
        "transform": ref["transform"],
        "compress": "lzw",
    }
    stack = np.transpose(W, (2, 0, 1))
    with rasterio.open(out_path, "w", **profile) as dst:
        for b in range(p):
            dst.write(stack[b], b + 1)
            dst.set_band_description(
                b + 1,
                f"W_{pollutants[b]}_GNFR_E_area",
            )
    if meta is not None:
        side = out_path.with_name(out_path.stem + "_meta.json")
        with side.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    return out_path
