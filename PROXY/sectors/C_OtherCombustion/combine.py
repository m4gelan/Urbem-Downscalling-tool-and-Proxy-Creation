"""Mass-conserving blend of stationary and off-road subgroups."""

from __future__ import annotations

import numpy as np

from PROXY.sectors.C_OtherCombustion.alpha_beta import AlphaBetaRow


def combine_branches(
    W_stat: dict[str, np.ndarray],
    w_F: np.ndarray,
    w_R: np.ndarray,
    w_B: np.ndarray,
    rows: dict[str, AlphaBetaRow],
    pollutant_outputs: list[str],
) -> dict[str, np.ndarray]:
    """
    W_C(p) = alpha_stat*W_stat(p) + alpha_off*(beta_F*w_F + beta_R*w_R + beta_B*w_B).
    """
    out: dict[str, np.ndarray] = {}
    for p in pollutant_outputs:
        row = rows[str(p)]
        w_off = (
            float(row.beta_F) * w_F
            + float(row.beta_R) * w_R
            + float(row.beta_B) * w_B
        )
        ws = W_stat[str(p)]
        wc = float(row.alpha_stat) * ws + float(row.alpha_off) * w_off
        out[str(p)] = wc.astype(np.float64, copy=False)
    return out


def assert_weights_sum_to_one(
    weights_by_pol: dict[str, np.ndarray],
    *,
    tol: float = 1e-6,
) -> None:
    for p, w in weights_by_pol.items():
        s = float(np.sum(w))
        if not np.isfinite(s) or abs(s - 1.0) > tol:
            raise AssertionError(f"pollutant {p!r}: weight sum {s} != 1 (tol {tol})")
