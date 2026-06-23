from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from proxy.core import log
from proxy.diagnostics.weight_sensitivity.cell_metrics import _cell_aggregate_setup

MixPrecompute = tuple[np.ndarray, dict[int, int], list[tuple[str, list[str], np.ndarray, np.ndarray, dict[tuple[int, int], np.ndarray]]]]


def precompute_mix_group_specs(
    mix_by_group: dict[str, dict[str, Any]],
    cell_id: np.ndarray,
    *,
    bundle_dir: Path | None = None,
) -> MixPrecompute:
    from proxy.diagnostics.weight_sensitivity.prong_w_core import _clear_terms, _load_terms

    valid, dense, unique, ndense = _cell_aggregate_setup(cell_id)
    g2d = {int(g): i for i, g in enumerate(unique)}
    group_specs: list[
        tuple[str, list[str], np.ndarray, np.ndarray, dict[tuple[int, int], np.ndarray]]
    ] = []
    for gname, spec in mix_by_group.items():
        if bundle_dir is not None:
            _load_terms(spec, bundle_dir)
        terms = spec["terms"]
        names = list(terms.keys())
        planes = [terms[k].ravel()[valid].astype(np.float64) for k in names]
        g = len(names)
        mass = np.zeros((g, ndense), dtype=np.float64)
        norm_sq = np.zeros((g, ndense), dtype=np.float64)
        pair_dots: dict[tuple[int, int], np.ndarray] = {}
        for i in range(g):
            mass[i] = np.bincount(dense, weights=planes[i], minlength=ndense)
            norm_sq[i] = np.bincount(dense, weights=planes[i] * planes[i], minlength=ndense)
        for i in range(g):
            for j in range(i + 1, g):
                pair_dots[(i, j)] = np.bincount(dense, weights=planes[i] * planes[j], minlength=ndense)
        group_specs.append((gname, names, mass, norm_sq, pair_dots))
        if bundle_dir is not None:
            _clear_terms(spec)
    return unique, g2d, group_specs


def compute_prong_a_mix(
    mix_by_group: dict[str, dict[str, Any]],
    cell_id: np.ndarray,
    nox_by_cell: dict[int, float],
    *,
    active_eps: float,
    similarity_threshold: float,
    precomputed: MixPrecompute | None = None,
    bundle_dir: Path | None = None,
) -> dict[str, Any]:
    """Prong A on expert mixture terms inside each group (w story, not alpha)."""
    if precomputed is None:
        unique, g2d, group_specs = precompute_mix_group_specs(
            mix_by_group, cell_id, bundle_dir=bundle_dir
        )
    else:
        unique, g2d, group_specs = precomputed

    cids = [c for c in nox_by_cell if nox_by_cell[c] > 0]
    if not cids:
        cids = [int(g) for g in unique]
        nox_by_cell = {c: 1.0 for c in cids}
        log.warning("no positive CAMS mass in bundle; equal unit mass over cells with footprint")

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
        max_active = 0
        cell_min_sim = 1.0
        best_mean_sim: float | None = None
        best_n_act = 0

        for _gname, _names, mass, norm_sq, pair_dots in group_specs:
            active_ix = [i for i in range(len(_names)) if mass[i, di] > active_eps]
            n_act = len(active_ix)
            max_active = max(max_active, n_act)
            if n_act < 2:
                continue

            sims: list[float] = []
            for a in range(len(active_ix)):
                for b in range(a + 1, len(active_ix)):
                    i, j = active_ix[a], active_ix[b]
                    key = (i, j) if i < j else (j, i)
                    ni = float(np.sqrt(norm_sq[i, di]))
                    nj = float(np.sqrt(norm_sq[j, di]))
                    if ni <= 0 or nj <= 0:
                        continue
                    sims.append(float(pair_dots[key][di]) / (ni * nj))

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
