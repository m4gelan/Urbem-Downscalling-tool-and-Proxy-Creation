from __future__ import annotations

from typing import Any

import numpy as np

from proxy.core import log


def _flat_valid(cell_id: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    flat = cell_id.ravel()
    valid = flat >= 0
    fid = flat[valid].astype(np.int64)
    nb = int(fid.max()) + 1 if fid.size else 1
    return valid, fid, nb


def _group_mass_and_norm_sq(
    W_by_group: dict[str, np.ndarray],
    valid: np.ndarray,
    fid: np.ndarray,
    nb: int,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    names = list(W_by_group.keys())
    g = len(names)
    mass = np.zeros((g, nb), dtype=np.float64)
    norm_sq = np.zeros((g, nb), dtype=np.float64)
    for i, plane in enumerate(W_by_group.values()):
        w = plane.ravel()[valid].astype(np.float64)
        mass[i] = np.bincount(fid, weights=w, minlength=nb)
        norm_sq[i] = np.bincount(fid, weights=w * w, minlength=nb)
    return names, mass, norm_sq


def _pairwise_dot_matrices(
    W_by_group: dict[str, np.ndarray],
    valid: np.ndarray,
    fid: np.ndarray,
    nb: int,
) -> list[np.ndarray]:
    planes = [p.ravel()[valid].astype(np.float64) for p in W_by_group.values()]
    g = len(planes)
    dots: list[np.ndarray] = []
    for i in range(g):
        row: list[np.ndarray] = []
        for j in range(g):
            if i == j:
                row.append(norm_sq_row := np.bincount(fid, weights=planes[i] * planes[i], minlength=nb))
            else:
                row.append(np.bincount(fid, weights=planes[i] * planes[j], minlength=nb))
        dots.append(row)
    return dots


def tv_per_cell_all(
    W: np.ndarray,
    Wp: np.ndarray,
    cell_id: np.ndarray,
) -> np.ndarray:
    """``TV(c) = 0.5 * sum |W'-W|`` per cell index; index = CAMS cell id."""
    valid, fid, nb = _flat_valid(cell_id)
    d = np.abs(
        Wp.ravel()[valid].astype(np.float64) - W.ravel()[valid].astype(np.float64)
    )
    return 0.5 * np.bincount(fid, weights=d, minlength=nb)


def compute_prong_a(
    W_by_group: dict[str, np.ndarray],
    cell_id: np.ndarray,
    nox_by_cell: dict[int, float],
    *,
    active_eps: float,
    similarity_threshold: float,
) -> dict[str, Any]:
    cids = [c for c in nox_by_cell if nox_by_cell[c] > 0]
    if not cids:
        valid, fid, nb = _flat_valid(cell_id)
        _, mass, _ = _group_mass_and_norm_sq(W_by_group, valid, fid, nb)
        cell_tot = mass.sum(axis=0)
        cids = [i for i in range(nb) if cell_tot[i] > active_eps]
        nox_by_cell = {c: 1.0 for c in cids}
        log.warning(
            "no positive NOx in bundle; using equal unit mass over cells with group footprint"
        )
    if not cids:
        raise ValueError("no CAMS cells with positive mass or footprint")

    valid, fid, nb = _flat_valid(cell_id)
    names, mass, norm_sq = _group_mass_and_norm_sq(W_by_group, valid, fid, nb)
    g = len(names)

    # Upper-triangle pairwise dot products (one pass per pair)
    pair_dots: dict[tuple[int, int], np.ndarray] = {}
    planes = [W_by_group[n].ravel()[valid].astype(np.float64) for n in names]
    for i in range(g):
        for j in range(i + 1, g):
            pair_dots[(i, j)] = np.bincount(fid, weights=planes[i] * planes[j], minlength=nb)

    total_nox = float(sum(nox_by_cell[c] for c in cids))
    a1_bins: dict[int, float] = {}
    sims_weighted: list[tuple[float, float]] = []
    sensitive_mass = 0.0
    sensitive_cell_ids: list[int] = []

    for cid in cids:
        if cid >= nb:
            continue
        m_c = nox_by_cell[cid]
        active_ix = [i for i in range(g) if mass[i, cid] > active_eps]
        n_act = len(active_ix)
        a1_bins[n_act] = a1_bins.get(n_act, 0.0) + m_c

        if n_act < 2:
            continue

        sims: list[float] = []
        for a in range(len(active_ix)):
            for b in range(a + 1, len(active_ix)):
                i, j = active_ix[a], active_ix[b]
                key = (i, j) if i < j else (j, i)
                ng = float(np.sqrt(norm_sq[i, cid]))
                nh = float(np.sqrt(norm_sq[j, cid]))
                if ng <= 0 or nh <= 0:
                    continue
                sims.append(float(pair_dots[key][cid]) / (ng * nh))

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
