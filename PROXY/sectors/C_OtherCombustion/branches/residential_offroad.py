"""Residential off-road: PublicPower-style eligibility × population blend on CLC 112."""

from __future__ import annotations

import numpy as np

from PROXY.core.proxy.eligibility_pop_blend import share_tensor_eligibility_pop_blend


def compute_W_R(
    clc_l3: np.ndarray,
    pop: np.ndarray,
    *,
    code_set: frozenset[int],
    floor: float,
    blend_eligibility_coef: float,
    blend_population_coef: float,
) -> tuple[np.ndarray, bool]:
    """
    Flattened weights; uniform fallback if the blend returns empty scores.
    """
    n = int(np.asarray(clc_l3).size)
    cor = np.asarray(clc_l3, dtype=np.float64)
    pop_a = np.asarray(pop, dtype=np.float64)
    if cor.shape != pop_a.shape:
        raise ValueError("clc_l3 and pop must have the same shape for residential off-road")
    packed = share_tensor_eligibility_pop_blend(
        cor,
        pop_a,
        code_set,
        float(floor),
        "pop_in_cell",
        blend_eligibility_coef=float(blend_eligibility_coef),
        blend_population_coef=float(blend_population_coef),
    )
    if packed is None:
        return np.full(n, 1.0 / max(n, 1), dtype=np.float64), True
    share, _raw, basis = packed
    flat = share.ravel()
    if basis == "uniform_cell" or float(np.sum(flat)) <= 0.0:
        return np.full(n, 1.0 / max(n, 1), dtype=np.float64), True
    return flat.astype(np.float64, copy=False), False
