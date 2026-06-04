from __future__ import annotations

from typing import Any

import numpy as np

from proxy.core import log
from proxy.diagnostics.weight_sensitivity.cell_metrics import _flat_valid


def compute_prong_a_mix(
    mix_by_group: dict[str, dict[str, Any]],
    cell_id: np.ndarray,
    nox_by_cell: dict[int, float],
    *,
    active_eps: float,
    similarity_threshold: float,
) -> dict[str, Any]:
    """Prong A on expert mixture terms inside each group (w story, not alpha)."""
    valid, fid, nb = _flat_valid(cell_id)
    cids = [c for c in nox_by_cell if nox_by_cell[c] > 0]
    if not cids:
        cids = [i for i in range(nb) if (cell_id == i).any()]
        nox_by_cell = {c: 1.0 for c in cids}
        log.warning("no positive NOx in bundle; equal unit mass over cells with footprint")

    group_specs: list[tuple[str, list[str], list[np.ndarray], np.ndarray, np.ndarray]] = []
    for gname, spec in mix_by_group.items():
        terms = spec["terms"]
        names = list(terms.keys())
        planes = [terms[k].ravel()[valid].astype(np.float64) for k in names]
        g = len(names)
        mass = np.zeros((g, nb), dtype=np.float64)
        norm_sq = np.zeros((g, nb), dtype=np.float64)
        pair_dots: dict[tuple[int, int], np.ndarray] = {}
        for i in range(g):
            mass[i] = np.bincount(fid, weights=planes[i], minlength=nb)
            norm_sq[i] = np.bincount(fid, weights=planes[i] * planes[i], minlength=nb)
        for i in range(g):
            for j in range(i + 1, g):
                pair_dots[(i, j)] = np.bincount(fid, weights=planes[i] * planes[j], minlength=nb)
        group_specs.append((gname, names, planes, mass, norm_sq, pair_dots))

    total_nox = float(sum(nox_by_cell[c] for c in cids))
    a1_bins: dict[int, float] = {}
    sims_weighted: list[tuple[float, float]] = []
    sensitive_mass = 0.0
    sensitive_cell_ids: list[int] = []

    for cid in cids:
        if cid >= nb:
            continue
        m_c = nox_by_cell[cid]
        max_active = 0
        cell_min_sim = 1.0
        best_mean_sim: float | None = None
        best_n_act = 0

        for _gname, _names, _planes, mass, norm_sq, pair_dots in group_specs:
            active_ix = [i for i in range(len(_names)) if mass[i, cid] > active_eps]
            n_act = len(active_ix)
            max_active = max(max_active, n_act)
            if n_act < 2:
                continue

            sims: list[float] = []
            for a in range(len(active_ix)):
                for b in range(a + 1, len(active_ix)):
                    i, j = active_ix[a], active_ix[b]
                    key = (i, j) if i < j else (j, i)
                    ni = float(np.sqrt(norm_sq[i, cid]))
                    nj = float(np.sqrt(norm_sq[j, cid]))
                    if ni <= 0 or nj <= 0:
                        continue
                    sims.append(float(pair_dots[key][cid]) / (ni * nj))

            if not sims:
                continue
            g_min = float(min(sims))
            g_mean = float(np.mean(sims))
            cell_min_sim = min(cell_min_sim, g_min)
            if n_act > best_n_act:
                best_n_act = n_act
                best_mean_sim = g_mean

        a1_bins[max_active] = a1_bins.get(max_active, 0.0) + m_c

        if best_n_act >= 2 and best_mean_sim is not None:
            sims_weighted.append((best_mean_sim, m_c))

        if max_active >= 2 and cell_min_sim < similarity_threshold:
            sensitive_mass += m_c
            sensitive_cell_ids.append(int(cid))

    a1_share = {int(k): v / total_nox for k, v in sorted(a1_bins.items())}
    a3_frac = sensitive_mass / total_nox if total_nox > 0 else 0.0

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
        "a1_mass_share_by_active_terms": a1_share,
        "a2_mean_similarity_multi_term": mean_sim,
        "a2_similarity_samples": [s for s, _ in sims_weighted],
        "a2_mass_weights": [w for _, w in sims_weighted],
        "a3_w_exposure_fraction": a3_frac,
        "a3_mix_exposure_nox_frac": a3_frac,
        "sensitive_cell_ids": sensitive_cell_ids,
        "similarity_threshold": similarity_threshold,
        "active_eps": active_eps,
        "n_mix_groups": len(mix_by_group),
    }
