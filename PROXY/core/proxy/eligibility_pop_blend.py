"""
Min–max population (within finite CORINE pixels) + eligibility blend for area weights.

Used by A_PublicPower (CORINE clip per CAMS cell) and GNFR C residential off-road
(reference-grid window). Formula:

    pop_01 = min–max(pop) on ok_data mask
    score = a * eligibility^(1 + pop_01) + b * pop_01
"""

from __future__ import annotations

from typing import Literal

import numpy as np


def pop_01_within_cell(pop_work: np.ndarray, ok_data: np.ndarray) -> np.ndarray:
    """Min–max population to [0, 1] over pixels where ``ok_data`` is true."""
    pop_01 = np.zeros_like(pop_work, dtype=np.float64)
    if not np.any(ok_data):
        return pop_01
    p = pop_work[ok_data]
    pmin = float(np.min(p))
    pmax = float(np.max(p))
    if pmax > pmin:
        pop_01[ok_data] = (pop_work[ok_data] - pmin) / (pmax - pmin)
    else:
        pop_01[ok_data] = 1.0 if pmax > 0.0 else 0.0
    return np.clip(pop_01, 0.0, 1.0)


def share_tensor_eligibility_pop_blend(
    corine_arr: np.ndarray,
    pop_dst: np.ndarray,
    code_set: frozenset[int],
    floor: float,
    fallback_if_no_corine: Literal["pop_in_cell", "skip"],
    *,
    blend_eligibility_coef: float,
    blend_population_coef: float,
) -> tuple[np.ndarray, np.ndarray, str] | None:
    """
    Per-pixel scores on ``corine_arr`` grid, normalized to sum 1.

    ``corine_arr`` may be L3 class codes or raw CORINE values; eligibility uses
    ``rint(corine) in code_set`` for finite pixels.
    """
    ok_data = np.isfinite(corine_arr)
    if not np.any(ok_data):
        return None

    rint = np.zeros_like(corine_arr, dtype=np.int32)
    rint[ok_data] = np.rint(corine_arr[ok_data]).astype(np.int32)
    corine_hit = ok_data & np.isin(rint, list(code_set))

    pop_work = np.maximum(np.nan_to_num(pop_dst, nan=0.0), floor)
    pop_01 = pop_01_within_cell(pop_work, ok_data)

    elig = corine_hit.astype(np.float64)
    exp_elig = np.zeros_like(corine_arr, dtype=np.float64)
    exp_elig[ok_data] = np.power(elig[ok_data], 1.0 + pop_01[ok_data])

    w_pix = np.zeros_like(corine_arr, dtype=np.float64)
    w_pix[ok_data] = (
        float(blend_eligibility_coef) * exp_elig[ok_data]
        + float(blend_population_coef) * pop_01[ok_data]
    )
    s = float(np.sum(w_pix))
    basis = "eligibility_pop_blend"

    if s <= 0:
        if fallback_if_no_corine == "skip":
            return None
        w_pix = np.where(ok_data, 1.0, 0.0)
        s = float(np.sum(w_pix))
        basis = "uniform_cell"
        if s <= 0:
            return None

    share = (w_pix / s).astype(np.float64, copy=False)
    return share, w_pix.astype(np.float64, copy=False), basis


__all__ = [
    "pop_01_within_cell",
    "share_tensor_eligibility_pop_blend",
]
