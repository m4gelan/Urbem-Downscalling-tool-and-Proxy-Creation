from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from proxy.core import log
from proxy.diagnostics.weight_sensitivity.cell_metrics import tv_per_cell_all
from proxy.diagnostics.weight_sensitivity.load_exports import (
    bundle_path,
    load_bundle,
    mass_for_pollutant,
)
from proxy.diagnostics.weight_sensitivity.plots import (
    apply_thesis_style,
    plot_a1_cross_sector,
    plot_a2_overlay,
    plot_a3_sensitivity_scatter,
    plot_b2_violin_by_weight,
    plot_b3,
    plot_b4,
    plot_headline_w_vs_alpha_sector,
    plot_stability_summary,
)
from proxy.diagnostics.weight_sensitivity.pollutants_config import reference_pollutants_from_cfg
from proxy.diagnostics.weight_sensitivity.prong_b_core import (
    fuse_alpha_stack,
    perturb_alpha_local,
)
from proxy.diagnostics.weight_sensitivity.prong_w_core import (
    crop_cell_patch,
    perturb_keys_for_spec,
    perturb_weight,
    prepare_w2_state,
    stack_with_perturbed_group,
)


def _stats(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"mean": 0.0, "p90": 0.0}
    a = np.asarray(vals, dtype=np.float64)
    return {"mean": float(a.mean()), "p90": float(np.percentile(a, 90))}


def sectors_with_mix(export_root: Path, country_tag: str, year: int) -> list[str]:
    out: list[str] = []
    if not export_root.is_dir():
        return out
    for sector_dir in sorted(export_root.iterdir()):
        if not sector_dir.is_dir():
            continue
        manifest_path = sector_dir / f"{country_tag}_{int(year)}" / "groups_manifest.yaml"
        if not manifest_path.is_file():
            continue
        with manifest_path.open(encoding="utf-8") as f:
            doc = yaml.safe_load(f)
        if isinstance(doc, dict) and doc.get("mix"):
            out.append(sector_dir.name)
    return out


def has_multi_alpha(alpha_doc: dict[str, Any]) -> bool:
    mat = np.asarray(alpha_doc.get("alpha") or [], dtype=np.float64)
    return mat.ndim == 2 and mat.shape[0] > 0 and mat.shape[1] > 1


def alpha_in_matrix(alpha_doc: dict[str, Any], pollutant: str) -> bool:
    return pollutant in (alpha_doc.get("pollutant_labels") or [])


def alpha_row_for_pollutant(alpha_doc: dict[str, Any], pollutant: str, n_groups: int) -> np.ndarray:
    labels = list(alpha_doc.get("pollutant_labels") or [])
    mat = np.asarray(alpha_doc.get("alpha") or [], dtype=np.float32)
    if pollutant in labels and mat.ndim == 2 and mat.shape[0]:
        return mat[labels.index(pollutant)]
    return np.ones(n_groups, dtype=np.float32) / float(max(1, n_groups))


def state_for_pollutant(state: dict[str, Any], mass: dict[int, float]) -> dict[str, Any]:
    sp = dict(state)
    sp["nox"] = mass
    sp["cids"] = [c for c in mass if mass[c] > 0]
    return sp


def _tv_values_w(
    state: dict[str, Any],
    alpha_row: np.ndarray,
    pct: float,
) -> list[float]:
    cell_id = state["cell_id"]
    valid, fid, nb = state["valid"], state["fid"], state["nb"]
    cids = state["cids"]
    mix_by_group = state["mix_by_group"]
    group_names = state["group_names"]
    gi_map = {g: i for i, g in enumerate(group_names)}
    base_stack = state["stack"]
    bundle_dir = state.get("bundle_dir")
    W0 = fuse_alpha_stack(base_stack, alpha_row, cell_id, valid, fid, nb)
    tv_all: list[float] = []
    for gname in group_names:
        spec = mix_by_group[gname]
        mixer = str(spec["mixer"])
        wkeys = spec.get("weight_keys")
        for wkey in perturb_keys_for_spec(spec):
            for sign in (-1, 1):
                w_new = perturb_weight(
                    spec["weights"], wkey, pct, sign, mixer, weight_keys=wkeys
                )
                stack_p = stack_with_perturbed_group(
                    base_stack,
                    gi_map[gname],
                    spec,
                    w_new,
                    cell_id,
                    valid,
                    fid,
                    nb,
                    bundle_dir=bundle_dir,
                )
                Wp = fuse_alpha_stack(stack_p, alpha_row, cell_id, valid, fid, nb)
                tv_arr = tv_per_cell_all(
                    np.nan_to_num(W0, nan=0.0), np.nan_to_num(Wp, nan=0.0), cell_id
                )
                for cid in cids:
                    if cid < tv_arr.size:
                        tv_all.append(float(tv_arr[cid]))
    return tv_all


def _tv_values_alpha(
    state: dict[str, Any],
    alpha_row: np.ndarray,
    pct: float,
) -> list[float]:
    if not has_multi_alpha(state["alpha_doc"]):
        return []
    cell_id = state["cell_id"]
    valid, fid, nb = state["valid"], state["fid"], state["nb"]
    cids = state["cids"]
    stack = state["stack"]
    g = len(state["group_names"])
    W0 = fuse_alpha_stack(stack, alpha_row, cell_id, valid, fid, nb)
    tv_all: list[float] = []
    for gi in range(g):
        for sign in (-1, 1):
            ap = perturb_alpha_local(alpha_row, gi, pct, sign)
            Wp = fuse_alpha_stack(stack, ap, cell_id, valid, fid, nb)
            tv_arr = tv_per_cell_all(
                np.nan_to_num(W0, nan=0.0), np.nan_to_num(Wp, nan=0.0), cell_id
            )
            for cid in cids:
                if cid < tv_arr.size:
                    tv_all.append(float(tv_arr[cid]))
    return tv_all


def collect_tv_by_weight(
    state: dict[str, Any],
    alpha_row: np.ndarray,
    pct: float,
) -> list[tuple[str, list[float]]]:
    cell_id = state["cell_id"]
    valid, fid, nb = state["valid"], state["fid"], state["nb"]
    cids = state["cids"]
    mix_by_group = state["mix_by_group"]
    group_names = state["group_names"]
    gi_map = {g: i for i, g in enumerate(group_names)}
    base_stack = state["stack"]
    bundle_dir = state.get("bundle_dir")
    W0 = fuse_alpha_stack(base_stack, alpha_row, cell_id, valid, fid, nb)
    out: list[tuple[str, list[float]]] = []
    for gname in group_names:
        spec = mix_by_group[gname]
        mixer = str(spec["mixer"])
        wkeys = spec.get("weight_keys")
        for wkey in perturb_keys_for_spec(spec):
            tvs: list[float] = []
            for sign in (-1, 1):
                w_new = perturb_weight(
                    spec["weights"], wkey, pct, sign, mixer, weight_keys=wkeys
                )
                stack_p = stack_with_perturbed_group(
                    base_stack,
                    gi_map[gname],
                    spec,
                    w_new,
                    cell_id,
                    valid,
                    fid,
                    nb,
                    bundle_dir=bundle_dir,
                )
                Wp = fuse_alpha_stack(stack_p, alpha_row, cell_id, valid, fid, nb)
                tv_arr = tv_per_cell_all(
                    np.nan_to_num(W0, nan=0.0), np.nan_to_num(Wp, nan=0.0), cell_id
                )
                for cid in cids:
                    if cid < tv_arr.size:
                        tvs.append(float(tv_arr[cid]))
            out.append((f"{gname} · {wkey}", tvs))
    return out


def collect_b4_rows(
    state: dict[str, Any],
    alpha_row: np.ndarray,
    sector: str,
    pollutant: str,
    pct: float,
) -> list[tuple[str, str, str, str, float]]:
    cell_id = state["cell_id"]
    valid, fid, nb = state["valid"], state["fid"], state["nb"]
    cids = state["cids"]
    mix_by_group = state["mix_by_group"]
    group_names = state["group_names"]
    gi_map = {g: i for i, g in enumerate(group_names)}
    base_stack = state["stack"]
    bundle_dir = state.get("bundle_dir")
    W0 = fuse_alpha_stack(base_stack, alpha_row, cell_id, valid, fid, nb)
    rows: list[tuple[str, str, str, str, float]] = []
    for gname in group_names:
        spec = mix_by_group[gname]
        mixer = str(spec["mixer"])
        wkeys = spec.get("weight_keys")
        for wkey in perturb_keys_for_spec(spec):
            w_new = perturb_weight(spec["weights"], wkey, pct, 1, mixer, weight_keys=wkeys)
            stack_p = stack_with_perturbed_group(
                base_stack,
                gi_map[gname],
                spec,
                w_new,
                cell_id,
                valid,
                fid,
                nb,
                bundle_dir=bundle_dir,
            )
            Wp = fuse_alpha_stack(stack_p, alpha_row, cell_id, valid, fid, nb)
            tv_arr = tv_per_cell_all(
                np.nan_to_num(W0, nan=0.0), np.nan_to_num(Wp, nan=0.0), cell_id
            )
            tv = float(np.mean([tv_arr[c] for c in cids if c < tv_arr.size]))
            rows.append((pollutant, sector, gname, wkey, tv / pct))
    return rows


def pick_b3_case(
    state: dict[str, Any],
    alpha_row: np.ndarray,
    sector: str,
    pollutant: str,
    pct: float = 0.4,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int], int, str, float]:
    rows = collect_b4_rows(state, alpha_row, sector, pollutant, 0.2)
    if not rows:
        raise ValueError("no b4 rows for b3 case")
    _p, _s, gname, wkey, _ = max(rows, key=lambda r: r[-1])
    spec = state["mix_by_group"][gname]
    mixer = str(spec["mixer"])
    wkeys = spec.get("weight_keys")
    sens = state["sensitive"]
    nox = state["nox"]
    if not sens:
        raise ValueError("no mix-sensitive cells for b3")
    cid = max(sens, key=lambda c: float(nox.get(c, 0.0)))
    cell_id = state["cell_id"]
    valid, fid, nb = state["valid"], state["fid"], state["nb"]
    gi_map = {g: i for i, g in enumerate(state["group_names"])}
    base_stack = state["stack"]
    bundle_dir = state.get("bundle_dir")
    W0_full = fuse_alpha_stack(base_stack, alpha_row, cell_id, valid, fid, nb)
    w_new = perturb_weight(spec["weights"], wkey, pct, 1, mixer, weight_keys=wkeys)
    stack_p = stack_with_perturbed_group(
        base_stack,
        gi_map[gname],
        spec,
        w_new,
        cell_id,
        valid,
        fid,
        nb,
        bundle_dir=bundle_dir,
    )
    W1_full = fuse_alpha_stack(stack_p, alpha_row, cell_id, valid, fid, nb)
    patch0 = crop_cell_patch(W0_full, cell_id, cid)
    patch1 = crop_cell_patch(W1_full, cell_id, cid)
    tv_arr = tv_per_cell_all(W0_full, W1_full, cell_id)
    tv_c = float(tv_arr[cid]) if cid < tv_arr.size else 0.0
    title = (
        f"{sector} · {pollutant} · cell {cid} · {gname} · {wkey} "
        f"+{int(pct * 100)}% (TV={tv_c:.4f})"
    )
    return patch0, patch1, patch0.shape, cid, title, tv_c


def load_prong_a_w_summaries(
    repo_root: Path,
    sectors: list[str],
    country_tag: str,
    year: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for sk in sectors:
        for base in ("Output", "OUTPUT"):
            p = (
                repo_root
                / base
                / "Proxy_diagnostics"
                / "figures"
                / "prong_a_w"
                / sk
                / f"{country_tag}_{year}"
                / "summary.json"
            )
            if p.is_file():
                out.append(json.loads(p.read_text(encoding="utf-8")))
                break
    return out


def sensitive_for_pollutant(
    prong_a_w: list[dict[str, Any]],
    sector: str,
    pollutant: str,
) -> set[int] | None:
    for r in prong_a_w:
        if r.get("sector_key") != sector:
            continue
        bp = (r.get("by_pollutant") or {}).get(pollutant)
        if bp and bp.get("sensitive_cell_ids"):
            return {int(x) for x in bp["sensitive_cell_ids"]}
    return None


def _a3_w_for_sector(prong_a_w: list[dict[str, Any]], sector: str, pollutant: str) -> float:
    for r in prong_a_w:
        if r.get("sector_key") != sector:
            continue
        bp = (r.get("by_pollutant") or {}).get(pollutant)
        if bp:
            return float(bp.get("a3_w_exposure_fraction") or 0.0)
        if pollutant == (r.get("reference_pollutants") or ["NOx"])[0]:
            return float(r.get("a3_w_exposure_fraction") or 0.0)
    return 0.0


def headline_w_vs_alpha_stats(
    state: dict[str, Any],
    pollutant: str,
    pct: float,
) -> dict[str, float]:
    n_g = len(state["group_names"])
    alpha_row = alpha_row_for_pollutant(state["alpha_doc"], pollutant, n_g)
    sw = _stats(_tv_values_w(state, alpha_row, pct))
    if has_multi_alpha(state["alpha_doc"]) and alpha_in_matrix(state["alpha_doc"], pollutant):
        sa = _stats(_tv_values_alpha(state, alpha_row, pct))
    else:
        sa = {"mean": 0.0, "p90": 0.0}
    return {
        "tv_w_mean": sw["mean"],
        "tv_w_p90": sw["p90"],
        "tv_a_mean": sa["mean"],
        "tv_a_p90": sa["p90"],
    }


def _headline_sector_key(cfg: dict[str, Any]) -> str:
    rep = cfg["representative_sectors"]
    if "high" in rep:
        return rep["high"]["sector_key"]
    if "industry" in rep:
        return rep["industry"]["sector_key"]
    return next(iter(rep.values()))["sector_key"]


def make_report_figures(
    export_root: Path,
    repo_root: Path,
    cfg: dict[str, Any],
    country_tag: str,
    year: int,
    fig_dir: Path,
) -> dict[str, Any]:
    apply_thesis_style()
    fig_dir.mkdir(parents=True, exist_ok=True)
    pct_headline = 0.2
    pollutants = reference_pollutants_from_cfg(cfg)
    sectors = sectors_with_mix(export_root, country_tag, year)
    log.info(f"report figures: {len(sectors)} sectors, pollutants={pollutants}")

    prong_a_w = load_prong_a_w_summaries(repo_root, sectors, country_tag, year)
    high_sk = _headline_sector_key(cfg)
    high_state_by_pol: dict[str, dict[str, Any]] = {}
    high_alpha_by_pol: dict[str, np.ndarray] = {}

    stability_by_pol: dict[str, list[dict[str, Any]]] = {p: [] for p in pollutants}
    a3_points_by_pol: dict[str, list[dict[str, Any]]] = {p: [] for p in pollutants}
    all_b4: list[tuple[str, str, str, str, float]] = []

    for sk in sectors:
        data = load_bundle(bundle_path(export_root, sk, country_tag, year))
        state = prepare_w2_state(
            data,
            repo_root=repo_root,
            active_eps=float(cfg["active_eps"]),
            similarity_threshold=float(cfg["similarity_threshold"]),
        )
        n_g = len(state["group_names"])

        for pol in pollutants:
            mass = mass_for_pollutant(data["mass_by_pollutant"], pol)
            sp = state_for_pollutant(state, mass)
            alpha_row = alpha_row_for_pollutant(sp["alpha_doc"], pol, n_g)
            tv_w = _tv_values_w(sp, alpha_row, pct_headline)
            sw = _stats(tv_w)
            if has_multi_alpha(sp["alpha_doc"]) and alpha_in_matrix(sp["alpha_doc"], pol):
                sa = _stats(_tv_values_alpha(sp, alpha_row, pct_headline))
            else:
                sa = {"mean": 0.0, "p90": 0.0}

            stability_by_pol[pol].append({
                "sector": sk,
                "tv_w_mean": sw["mean"],
                "tv_w_p90": sw["p90"],
                "tv_a_mean": sa["mean"],
                "tv_a_p90": sa["p90"],
            })

            if sk == high_sk:
                high_state_by_pol[pol] = sp
                high_alpha_by_pol[pol] = alpha_row

            if sk == "A_PublicPower":
                continue
            if not has_multi_alpha(sp["alpha_doc"]) or not alpha_in_matrix(sp["alpha_doc"], pol):
                continue

            a3_base = _a3_w_for_sector(prong_a_w, sk, pol)
            tv_w = _stats(_tv_values_w(sp, alpha_row, pct_headline))
            a3_points_by_pol[pol].append({
                "sector": sk,
                "pollutant_role": pol,
                "pollutant_label": pol,
                "legend_label": pol,
                "a3_w": a3_base,
                "tv_w_mean": tv_w["mean"],
            })
            all_b4.extend(collect_b4_rows(sp, alpha_row, sk, pol, pct_headline))

    for pol in pollutants:
        plot_stability_summary(
            stability_by_pol[pol],
            fig_dir / f"headline_w_vs_alpha__{pol}.png",
            pollutant=pol,
        )

        sector_a1 = {}
        for r in prong_a_w:
            bp = (r.get("by_pollutant") or {}).get(pol)
            if bp:
                sector_a1[r["sector_key"]] = {int(k): float(v) for k, v in bp["a1_mass_share_by_active_terms"].items()}
        if sector_a1:
            plot_a1_cross_sector(sector_a1, fig_dir / f"a1_cross_sector__{pol}.png", pollutant=pol)

        a2_slice = []
        for r in prong_a_w:
            bp = (r.get("by_pollutant") or {}).get(pol)
            if not bp:
                continue
            a2_slice.append({
                "sector_key": r["sector_key"],
                "a2_similarity_samples": bp.get("a2_similarity_samples") or [],
                "a2_mass_weights": bp.get("a2_mass_weights") or [],
            })
        if a2_slice:
            plot_a2_overlay(
                a2_slice,
                fig_dir / f"a2_overlay__{pol}.png",
                threshold=float(cfg["similarity_threshold"]),
                pollutant=pol,
            )

        if a3_points_by_pol[pol]:
            plot_a3_sensitivity_scatter(
                a3_points_by_pol[pol],
                fig_dir / f"a3_sensitivity_scatter__{pol}.png",
                pollutant=pol,
            )

        if pol in high_state_by_pol:
            plot_b2_violin_by_weight(
                collect_tv_by_weight(high_state_by_pol[pol], high_alpha_by_pol[pol], pct_headline),
                fig_dir / f"b2_violin_by_weight__{pol}.png",
                sector=high_sk,
                pollutant=pol,
            )
            b3_state = dict(high_state_by_pol[pol])
            sens = sensitive_for_pollutant(prong_a_w, high_sk, pol)
            if sens:
                b3_state["sensitive"] = sens
            try:
                p0, p1, shape, _cid, title, _tv = pick_b3_case(
                    b3_state,
                    high_alpha_by_pol[pol],
                    high_sk,
                    pol,
                    pct=0.4,
                )
                plot_b3(p0, p1, shape, fig_dir / f"b3_case_study__{pol}.png", title=title)
            except ValueError as exc:
                log.warning(f"B3 skip {high_sk} {pol}: {exc}")

    if all_b4:
        plot_b4(
            [(r[1], r[2], r[3], r[4]) for r in all_b4],
            fig_dir / "b4_elasticity.png",
        )
        plot_b4(
            [(f"{r[0]}·{r[1]}", r[2], r[3], r[4]) for r in all_b4],
            fig_dir / "b4_elasticity_by_pollutant.png",
        )

    summary = {
        "reference_pollutants": pollutants,
        "sectors": sectors,
        "stability_by_pollutant": stability_by_pol,
        "a3_points_by_pollutant": a3_points_by_pol,
        "b4_rows": all_b4,
    }
    (fig_dir / "report_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
