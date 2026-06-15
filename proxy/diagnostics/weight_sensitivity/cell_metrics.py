from __future__ import annotations

from typing import Any

import numpy as np

from proxy.core import log


def _cell_aggregate_setup(
    cell_id: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    flat = cell_id.ravel()
    valid = flat >= 0
    global_fid = flat[valid].astype(np.int64)
    unique, dense = np.unique(global_fid, return_inverse=True)
    return valid, dense.astype(np.int64), unique, len(unique)


def _flat_valid(cell_id: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    flat = cell_id.ravel()
    valid = flat >= 0
    fid = flat[valid].astype(np.int64)
    nb = int(fid.max()) + 1 if fid.size else 1
    return valid, fid, nb


def _group_mass_and_norm_sq(
    W_by_group: dict[str, np.ndarray],
    valid: np.ndarray,
    dense_fid: np.ndarray,
    ndense: int,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    names = list(W_by_group.keys())
    g = len(names)
    mass = np.zeros((g, ndense), dtype=np.float64)
    norm_sq = np.zeros((g, ndense), dtype=np.float64)
    for i, plane in enumerate(W_by_group.values()):
        w = plane.ravel()[valid].astype(np.float64)
        mass[i] = np.bincount(dense_fid, weights=w, minlength=ndense)
        norm_sq[i] = np.bincount(dense_fid, weights=w * w, minlength=ndense)
    return names, mass, norm_sq


def tv_per_cell_all(
    W: np.ndarray,
    Wp: np.ndarray,
    cell_id: np.ndarray,
) -> np.ndarray:
    """``TV(c) = 0.5 * sum |W'-W|`` per cell; returned array indexed by global CAMS cell id."""
    valid, dense, unique, ndense = _cell_aggregate_setup(cell_id)
    d = np.abs(
        Wp.ravel()[valid].astype(np.float64) - W.ravel()[valid].astype(np.float64)
    )
    tv_dense = 0.5 * np.bincount(dense, weights=d, minlength=ndense)
    out = np.zeros(int(unique.max()) + 1, dtype=np.float64)
    out[unique] = tv_dense
    return out


def compute_prong_a(
    W_by_group: dict[str, np.ndarray],
    cell_id: np.ndarray,
    nox_by_cell: dict[int, float],
    *,
    active_eps: float,
    similarity_threshold: float,
) -> dict[str, Any]:
    valid, dense, unique, ndense = _cell_aggregate_setup(cell_id)
    g2d = {int(g): i for i, g in enumerate(unique)}

    cids = [c for c in nox_by_cell if nox_by_cell[c] > 0]
    if not cids:
        _, mass, _ = _group_mass_and_norm_sq(W_by_group, valid, dense, ndense)
        cell_tot = mass.sum(axis=0)
        cids = [int(unique[i]) for i in range(ndense) if cell_tot[i] > active_eps]
        nox_by_cell = {c: 1.0 for c in cids}
        log.warning(
            "no positive CAMS mass in bundle; using equal unit mass over cells with group footprint"
        )
    if not cids:
        raise ValueError("no CAMS cells with positive mass or footprint")

    names, mass, norm_sq = _group_mass_and_norm_sq(W_by_group, valid, dense, ndense)
    g = len(names)

    pair_dots: dict[tuple[int, int], np.ndarray] = {}
    planes = [W_by_group[n].ravel()[valid].astype(np.float64) for n in names]
    for i in range(g):
        for j in range(i + 1, g):
            pair_dots[(i, j)] = np.bincount(
                dense, weights=planes[i] * planes[j], minlength=ndense
            )

    total_nox = float(sum(nox_by_cell[c] for c in cids))
    a1_bins: dict[int, float] = {}
    sims_weighted: list[tuple[float, float]] = []
    sensitive_mass = 0.0
    sensitive_cell_ids: list[int] = []

    for cid in cids:
        di = g2d.get(int(cid))
        if di is None:
            continue
        m_c = nox_by_cell[cid]
        active_ix = [i for i in range(g) if mass[i, di] > active_eps]
        n_act = len(active_ix)
        a1_bins[n_act] = a1_bins.get(n_act, 0.0) + m_c

        if n_act < 2:
            continue

        sims: list[float] = []
        for a in range(len(active_ix)):
            for b in range(a + 1, len(active_ix)):
                i, j = active_ix[a], active_ix[b]
                key = (i, j) if i < j else (j, i)
                ng = float(np.sqrt(norm_sq[i, di]))
                nh = float(np.sqrt(norm_sq[j, di]))
                if ng <= 0 or nh <= 0:
                    continue
                sims.append(float(pair_dots[key][di]) / (ng * nh))

        if not sims:
            continue
        sim = float(np.mean(sims))
        sims_weighted.append((sim, m_c))
        if sim < similarity_threshold:
            sensitive_mass += m_c
            sensitive_cell_ids.append(int(cid))

    a1_share = {int(k): v / total_nox for k, v in sorted(a1_bins.items())}
    a3_frac = sensitive_mass / total_nox

    if sims_weighted:
        sim_vals = np.array([s for s, _ in sims_weighted], dtype=np.float64)
        wts = np.array([w for _, w in sims_weighted], dtype=np.float64)
        w_sum = float(wts.sum())
        mean_sim = float(np.average(sim_vals, weights=wts)) if w_sum > 0 else 0.0
    else:
        mean_sim = None

    return {
        "total_nox": total_nox,
        "n_cells_with_nox": len(cids),
        "a1_mass_share_by_active_groups": a1_share,
        "a2_mean_similarity_multi_group": mean_sim,
        "a2_similarity_samples": [s for s, _ in sims_weighted],
        "a2_mass_weights": [w for _, w in sims_weighted],
        "a3_alpha_exposure_fraction": a3_frac,
        "sensitive_cell_ids": sensitive_cell_ids,
        "similarity_threshold": similarity_threshold,
        "active_eps": active_eps,
        "n_groups": g,
    }
