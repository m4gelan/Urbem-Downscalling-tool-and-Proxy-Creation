from __future__ import annotations

from typing import Any

import numpy as np

from proxy.core import log
from proxy.dataset_loaders.load_eurostat_f_road import RoadFuelSplitResult


def model_axes(cfg: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    m = cfg["model"]
    return [str(x) for x in m["road_types"]], [str(x) for x in m["vehicle_classes"]], [str(x) for x in m["fuels"]]


def build_s_tensor(
    pi: dict[str, dict[str, float]],
    fuel_split: RoadFuelSplitResult,
    road_types: list[str],
    classes: list[str],
    fuels: list[str],
) -> np.ndarray:
    """S[r,c,f] = Pi[r,c] * F[c,f]."""
    s = np.zeros((len(road_types), len(classes), len(fuels)), dtype=np.float32)
    for ri, r in enumerate(road_types):
        for ci, c in enumerate(classes):
            f_row = fuel_split.split_by_class[c]
            for fi, f in enumerate(fuels):
                s[ri, ci, fi] = float(pi[r][c]) * float(f_row[f])
    log.info(f"F_Roads S tensor shape={s.shape}")
    return s


def build_x_from_s(
    s: np.ndarray,
    aadt_rasters: dict[str, np.ndarray],
    road_types: list[str],
    classes: list[str],
    fuels: list[str],
) -> np.ndarray:
    """X[:,:,k] class-major then fuel; X[c,f,j] = sum_r S[r,c,f] * L_r(j)."""
    ref = next(iter(aadt_rasters.values()))
    h, w = ref.shape
    n_bands = len(classes) * len(fuels)
    x = np.zeros((h, w, n_bands), dtype=np.float32)
    for ci, c in enumerate(classes):
        for fi, f in enumerate(fuels):
            band = ci * len(fuels) + fi
            acc = np.zeros((h, w), dtype=np.float32)
            for ri, r in enumerate(road_types):
                acc += s[ri, ci, fi] * aadt_rasters[r]
            x[:, :, band] = acc
    labels = [f"{c}_{f}" for c in classes for f in fuels]
    log.info(f"F_Roads X stack shape=({h}, {w}, {n_bands}) bands={labels}")
    return x
