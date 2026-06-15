from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from proxy.core import log
from proxy.core.area_weights import combined_S_industry_group, combined_S_publicpower
from proxy.diagnostics.weight_sensitivity.cell_metrics import tv_per_cell_all
from proxy.diagnostics.weight_sensitivity.load_exports import (
    load_bundle,
    mass_for_pollutant,
)
from proxy.diagnostics.weight_sensitivity.mix_metrics import compute_prong_a_mix
from proxy.diagnostics.weight_sensitivity.prong_b_core import (
    _normalize_S,
    crop_cell_patch,
    fuse_alpha_stack,
)


def load_sensitive_from_prong_a_w(
    repo_root: Path,
    sector_key: str,
    country_tag: str,
    year: int,
) -> set[int] | None:
    for base in ("Output", "OUTPUT"):
        p = (
            repo_root
            / base
            / "Proxy_diagnostics"
            / "figures"
            / "prong_a_w"
            / sector_key
            / f"{country_tag}_{int(year)}"
            / "summary.json"
        )
        if p.is_file():
            doc = json.loads(p.read_text(encoding="utf-8"))
            ids = doc.get("sensitive_cell_ids") or []
            return {int(x) for x in ids}
    return None


def load_sensitive_for_pollutant(
    repo_root: Path,
    sector_key: str,
    country_tag: str,
    year: int,
    pollutant: str,
) -> set[int] | None:
    for base in ("Output", "OUTPUT"):
        p = (
            repo_root
            / base
            / "Proxy_diagnostics"
            / "figures"
            / "prong_a_w"
            / sector_key
            / f"{country_tag}_{int(year)}"
            / "summary.json"
        )
        if not p.is_file():
            continue
        doc = json.loads(p.read_text(encoding="utf-8"))
        bp = (doc.get("by_pollutant") or {}).get(pollutant)
        if bp and bp.get("sensitive_cell_ids"):
            return {int(x) for x in bp["sensitive_cell_ids"]}
    return None


def _state_for_pollutant(state: dict[str, Any], mass: dict[int, float]) -> dict[str, Any]:
    sp = dict(state)
    sp["nox"] = mass
    sp["cids"] = [c for c in mass if mass[c] > 0]
    return sp


def _alpha_row_for_pollutant(
    alpha_doc: dict[str, Any],
    pollutant: str,
    n_groups: int,
) -> np.ndarray:
    labels = list(alpha_doc.get("pollutant_labels") or [])
    mat = np.asarray(alpha_doc.get("alpha") or [], dtype=np.float32)
    if pollutant in labels and mat.ndim == 2 and mat.shape[0]:
        return mat[labels.index(pollutant)]
    return np.ones(n_groups, dtype=np.float32) / float(max(1, n_groups))


def _flat_valid(cell_id: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    flat = cell_id.ravel()
    valid = flat >= 0
    fid = flat[valid].astype(np.int64)
    nb = int(fid.max()) + 1 if fid.size else 1
    return valid, fid, nb


def score_from_mix(spec: dict[str, Any]) -> np.ndarray:
    mixer = str(spec["mixer"])
    terms = spec["terms"]
    w = spec["weights"]
    if mixer == "industry_group":
        return combined_S_industry_group(
            terms["osm"],
            terms["corine"],
            terms["pop_z"],
            w_osm=float(w["w_osm"]),
            w_clc=float(w["w_clc"]),
            w_pop=float(w["w_pop"]),
        )
    if mixer == "linear3":
        s = 0.0
        for i in range(1, 4):
            s = s + float(w[f"w{i}"]) * terms[f"t{i}"]
        return np.asarray(s, dtype=np.float32)
    if mixer == "linear4":
        s = 0.0
        for i in range(1, 5):
            s = s + float(w[f"w{i}"]) * terms[f"t{i}"]
        return np.asarray(s, dtype=np.float32)
    if mixer == "publicpower":
        return combined_S_publicpower(
            terms["pop_z"],
            terms["corine"],
            w1=float(w["w1"]),
            w2=float(w["w2"]),
        )
    if mixer == "linear":
        s = 0.0
        for wk, tk in zip(spec["weight_keys"], spec["term_keys"]):
            s = s + float(w[wk]) * terms[tk].astype(np.float64)
        return np.asarray(s, dtype=np.float32)
    raise ValueError(f"unknown mixer {mixer!r}")


def _pack_linear_terms(raw: dict[str, np.ndarray], keys: list[str]) -> dict[str, np.ndarray]:
    return {f"t{i + 1}": raw[k] for i, k in enumerate(keys)}


def load_mix_from_manifest(manifest: dict[str, Any], bundle_dir: Path) -> dict[str, dict[str, Any]]:
    mix_doc = manifest.get("mix")
    if not mix_doc:
        raise ValueError(
            f"bundle has no mix terms (re-run build_and_export with mix export): {bundle_dir}"
        )
    out: dict[str, dict[str, Any]] = {}
    for ent in mix_doc:
        gname = str(ent["name"])
        terms: dict[str, np.ndarray] = {}
        term_keys: list[str] = []
        for te in ent["terms"]:
            tkey = str(te["key"])
            term_keys.append(tkey)
            with rasterio.open(bundle_dir / str(te["file"])) as src:
                terms[tkey] = src.read(1).astype(np.float32)
        mixer = str(ent["mixer"])
        weights = {str(k): float(v) for k, v in ent["weights"].items()}
        if mixer == "linear":
            wkeys = [str(k) for k in ent.get("weight_keys") or []]
            if len(wkeys) != len(term_keys):
                raise ValueError(f"mix[{gname}]: weight_keys len != terms len")
            out[gname] = {
                "mixer": "linear",
                "weights": weights,
                "terms": terms,
                "term_keys": term_keys,
                "weight_keys": wkeys,
            }
            continue
        if mixer in ("linear3", "linear4"):
            n = 3 if mixer == "linear3" else 4
            packed = _pack_linear_terms(terms, term_keys[:n])
            spec = {"mixer": mixer, "weights": weights, "terms": packed, "term_keys": term_keys[:n]}
        else:
            spec = {"mixer": mixer, "weights": weights, "terms": terms, "term_keys": term_keys}
        out[gname] = spec
    return out


def perturb_weight(
    weights: dict[str, float],
    key: str,
    pct: float,
    sign: int,
    mixer: str,
    *,
    weight_keys: list[str] | None = None,
) -> dict[str, float]:
    w = deepcopy(weights)
    factor = 1.0 + sign * float(pct)
    w[key] = max(0.0, float(w[key]) * factor)
    if mixer == "industry_group":
        if key == "w_pop":
            w["w_pop"] = min(1.0, max(0.0, float(w["w_pop"])))
            return w
        if key == "w_osm":
            pair = float(w["w_osm"]) + float(w["w_clc"])
            if pair <= 0:
                raise ValueError("w_osm + w_clc is zero")
            w["w_osm"] = float(w["w_osm"]) / pair
            w["w_clc"] = float(w["w_clc"]) / pair
            return w
    if mixer == "publicpower" and key == "w1":
        s = float(w["w1"]) + float(w["w2"])
        if s <= 0:
            raise ValueError("w1 + w2 is zero")
        w["w1"] /= s
        w["w2"] /= s
        return w
    keys = list(weight_keys) if weight_keys else [k for k in w if str(k).startswith("w")]
    s = sum(float(w[k]) for k in keys)
    if s <= 0:
        raise ValueError("weight simplex sums to zero")
    for k in keys:
        w[k] = float(w[k]) / s
    return w


def perturb_keys_for_spec(spec: dict[str, Any]) -> list[str]:
    mixer = str(spec["mixer"])
    if mixer == "linear":
        return [str(k) for k in spec["weight_keys"]]
    if mixer == "industry_group":
        return ["w_osm", "w_pop"]
    if mixer == "publicpower":
        return ["w1"]
    if mixer == "linear3":
        return ["w1", "w2", "w3"]
    if mixer == "linear4":
        return ["w1", "w2", "w3", "w4"]
    raise ValueError(f"unknown mixer {mixer!r}")


def W_g_from_spec(
    spec: dict[str, Any],
    cell_id: np.ndarray,
    valid: np.ndarray,
    fid: np.ndarray,
    nb: int,
) -> np.ndarray:
    S = score_from_mix(spec)
    return _normalize_S(S, cell_id, valid, fid, nb)


def stack_from_mix(
    mix_by_group: dict[str, dict[str, Any]],
    group_names: list[str],
    cell_id: np.ndarray,
    valid: np.ndarray,
    fid: np.ndarray,
    nb: int,
) -> np.ndarray:
    planes = [W_g_from_spec(mix_by_group[g], cell_id, valid, fid, nb) for g in group_names]
    return np.stack(planes, axis=0)


def stack_with_perturbed_group(
    base_stack: np.ndarray,
    group_ix: int,
    spec: dict[str, Any],
    weights: dict[str, float],
    cell_id: np.ndarray,
    valid: np.ndarray,
    fid: np.ndarray,
    nb: int,
) -> np.ndarray:
    """Reuse frozen group planes; recompute only the perturbed group."""
    out = np.asarray(base_stack, dtype=np.float32, order="C").copy()
    perturbed = dict(spec)
    perturbed["weights"] = weights
    out[group_ix] = W_g_from_spec(perturbed, cell_id, valid, fid, nb)
    return out


def prepare_w2_state(
    data: dict[str, Any],
    *,
    repo_root: Path | None = None,
    active_eps: float = 1e-9,
    similarity_threshold: float = 0.7,
) -> dict[str, Any]:
    manifest = data["manifest"]
    bundle_dir = data["bundle_dir"]
    mix_by_group = load_mix_from_manifest(manifest, bundle_dir)
    group_names = list(data["group_names"])
    cell_id = data["cell_id"]
    valid, fid, nb = _flat_valid(cell_id)
    nox = data["nox_by_cell"]

    sensitive: set[int] | None = None
    sk = str(data["manifest"].get("sector_key", ""))
    ct = str(data["manifest"].get("country_tag", ""))
    yr = int(data["manifest"].get("year", 2019))
    if repo_root is not None and sk and ct:
        sensitive = load_sensitive_from_prong_a_w(repo_root, sk, ct, yr)

    pa: dict[str, Any]
    if sensitive is None:
        pa = compute_prong_a_mix(
            mix_by_group,
            cell_id,
            nox,
            active_eps=active_eps,
            similarity_threshold=similarity_threshold,
        )
        sensitive = set(pa.get("sensitive_cell_ids") or [])
        log.info(f"Prong B (w): computed {len(sensitive)} mix-sensitive cells")
    else:
        pa = {"a3_w_exposure_fraction": None}
        log.info(f"Prong B (w): loaded {len(sensitive)} mix-sensitive cells from prong_a_w")

    stack = stack_from_mix(mix_by_group, group_names, cell_id, valid, fid, nb)
    return {
        "group_names": group_names,
        "mix_by_group": mix_by_group,
        "stack": stack,
        "cell_id": cell_id,
        "valid": valid,
        "fid": fid,
        "nb": nb,
        "nox": nox,
        "cids": [c for c in nox if nox[c] > 0],
        "sensitive": sensitive,
        "alpha_doc": data["alpha"],
        "mix_a3": pa.get("a3_w_exposure_fraction"),
    }


def run_w2_on_state(
    state: dict[str, Any],
    alpha_row: np.ndarray,
    perturb_pct: list[float],
    *,
    pollutant: str = "",
    label: str = "",
) -> dict[str, Any]:
    g = len(state["group_names"])
    alpha_row = np.asarray(alpha_row, dtype=np.float32)
    if alpha_row.size != g:
        raise ValueError(f"alpha_row length {alpha_row.size} != {g} groups")

    cell_id = state["cell_id"]
    valid, fid, nb = state["valid"], state["fid"], state["nb"]
    nox = state["nox"]
    cids = state["cids"]
    sensitive = state["sensitive"]
    mix_by_group = state["mix_by_group"]
    group_names = state["group_names"]

    tag = f" {label}" if label else ""
    log.info(f"Prong B (w){tag}: fuse baseline W0 ({g} groups, alpha fixed)")
    W0 = fuse_alpha_stack(state["stack"], alpha_row, cell_id, valid, fid, nb)

    tv_all: list[float] = []
    tv_sens: list[float] = []
    n_pert = 0
    jobs: list[tuple[str, str]] = []
    for gname in group_names:
        for wkey in perturb_keys_for_spec(mix_by_group[gname]):
            jobs.append((gname, wkey))
    n_pert = len(perturb_pct) * len(jobs) * 2

    gi_map = {g: i for i, g in enumerate(group_names)}
    base_stack = state["stack"]
    k = 0
    for pct in perturb_pct:
        for gname, wkey in jobs:
            spec = mix_by_group[gname]
            mixer = str(spec["mixer"])
            wkeys = spec.get("weight_keys")
            for sign in (-1, 1):
                k += 1
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
                )
                Wp = fuse_alpha_stack(stack_p, alpha_row, cell_id, valid, fid, nb)
                tv_arr = tv_per_cell_all(
                    np.nan_to_num(W0, nan=0.0),
                    np.nan_to_num(Wp, nan=0.0),
                    cell_id,
                )
                for cid in cids:
                    if cid >= tv_arr.size:
                        continue
                    tv = float(tv_arr[cid])
                    if not np.isfinite(tv):
                        continue
                    tv_all.append(tv)
                    if cid in sensitive:
                        tv_sens.append(tv)
                if k % 4 == 0 or k == n_pert:
                    log.info(f"Prong B (w){tag}: perturb {k}/{n_pert}")

    def _stats(vals: list[float]) -> dict[str, float]:
        if not vals:
            log.warning(f"Prong B (w){tag}: no TV samples — check pollutant CAMS mass")
            return {"mean": 0.0, "p90": 0.0}
        a = np.asarray(vals, dtype=np.float64)
        return {"mean": float(a.mean()), "p90": float(np.percentile(a, 90))}

    return {
        "analysis": "w_mixture",
        "tv_all": _stats(tv_all),
        "tv_sensitive": _stats(tv_sens),
        "n_sensitive_cells": len(sensitive),
        "mix_a3_nox_frac": state.get("mix_a3"),
        "pollutant": pollutant,
        "group_names": group_names,
    }


def run_w2_all_representatives(
    export_root: Path,
    repo_root: Path,
    cfg: dict[str, Any],
    country_tag: str,
    year: int,
) -> dict[str, Any]:
    pct = [float(x) for x in cfg["perturb_pct"]]
    report_pol = cfg["report_pollutant"]
    summaries: dict[str, Any] = {}
    for role, rep in cfg["representative_sectors"].items():
        sk = rep["sector_key"]
        bdir = export_root / sk / f"{country_tag}_{int(year)}"
        log.info(f"Prong B (w) role={role} sector={sk} pollutant={report_pol}")
        data = load_bundle(bdir)
        if not data["manifest"].get("mix"):
            raise ValueError(f"{sk}: no mix export in bundle — re-run build_and_export")
        state = prepare_w2_state(
            data,
            repo_root=repo_root,
            active_eps=float(cfg["active_eps"]),
            similarity_threshold=float(cfg["similarity_threshold"]),
        )
        n_g = len(state["group_names"])
        mass = mass_for_pollutant(data["mass_by_pollutant"], report_pol)
        sp = _state_for_pollutant(state, mass)
        sens = load_sensitive_for_pollutant(repo_root, sk, country_tag, year, report_pol)
        if sens:
            sp["sensitive"] = sens
        alpha_row = _alpha_row_for_pollutant(state["alpha_doc"], report_pol, n_g)
        summaries[role] = {
            "sector": sk,
            "pollutant": report_pol,
            "b2": run_w2_on_state(
                sp, alpha_row, pct, pollutant=report_pol, label=f"{sk} {report_pol}",
            ),
        }
    return summaries


def collect_w_perturb_tvs(
    state: dict[str, Any],
    alpha_row: np.ndarray,
    perturb_pct: float,
    *,
    label: str = "",
) -> tuple[list[float], list[float]]:
    """TV samples for violin plots; one perturbed group per job (no term deepcopy)."""
    cell_id = state["cell_id"]
    valid, fid, nb = state["valid"], state["fid"], state["nb"]
    cids, sensitive = state["cids"], state["sensitive"]
    mix_by_group = state["mix_by_group"]
    group_names = state["group_names"]
    gi_map = {g: i for i, g in enumerate(group_names)}
    base_stack = state["stack"]
    pct = float(perturb_pct)

    tag = f" {label}" if label else ""
    log.info(f"Prong B (w) violin{tag}: fuse baseline W0")
    W0 = fuse_alpha_stack(base_stack, alpha_row, cell_id, valid, fid, nb)

    jobs: list[tuple[str, str]] = []
    for gname in group_names:
        for wkey in perturb_keys_for_spec(mix_by_group[gname]):
            jobs.append((gname, wkey))
    tv_all: list[float] = []
    tv_sens: list[float] = []

    for k, (gname, wkey) in enumerate(jobs, start=1):
        spec = mix_by_group[gname]
        mixer = str(spec["mixer"])
        wkeys = spec.get("weight_keys")
        for sign in (-1, 1):
            w_new = perturb_weight(
                spec["weights"], wkey, pct, sign, mixer, weight_keys=wkeys
            )
            stack_p = stack_with_perturbed_group(
                base_stack, gi_map[gname], spec, w_new, cell_id, valid, fid, nb
            )
            Wp = fuse_alpha_stack(stack_p, alpha_row, cell_id, valid, fid, nb)
            tv_arr = tv_per_cell_all(np.nan_to_num(W0, nan=0.0), np.nan_to_num(Wp, nan=0.0), cell_id)
            for cid in cids:
                if cid >= tv_arr.size:
                    continue
                tv = float(tv_arr[cid])
                if not np.isfinite(tv):
                    continue
                tv_all.append(tv)
                if cid in sensitive:
                    tv_sens.append(tv)
        if k % 2 == 0 or k == len(jobs):
            log.info(f"Prong B (w) violin{tag}: {k}/{len(jobs)} groups")

    return tv_all, tv_sens


def collect_w_perturb_tvs_multi(
    state: dict[str, Any],
    by_pollutant: dict[str, dict[str, Any]],
    perturb_pct: float,
    *,
    label: str = "",
) -> dict[str, tuple[list[float], list[float]]]:
    """One w-perturbation pass per sector; fuse/TV per pollutant (alpha_row differs)."""
    if not by_pollutant:
        return {}
    cell_id = state["cell_id"]
    valid, fid, nb = state["valid"], state["fid"], state["nb"]
    mix_by_group = state["mix_by_group"]
    group_names = state["group_names"]
    gi_map = {g: i for i, g in enumerate(group_names)}
    base_stack = state["stack"]
    pct = float(perturb_pct)

    w0: dict[str, np.ndarray] = {}
    out: dict[str, tuple[list[float], list[float]]] = {}
    for pol, spec in by_pollutant.items():
        alpha_row = spec["alpha_row"]
        w0[pol] = fuse_alpha_stack(base_stack, alpha_row, cell_id, valid, fid, nb)
        out[pol] = ([], [])

    jobs: list[tuple[str, str]] = []
    for gname in group_names:
        for wkey in perturb_keys_for_spec(mix_by_group[gname]):
            jobs.append((gname, wkey))

    tag = f" {label}" if label else ""
    log.info(f"Prong B (w) violin multi{tag}: {len(jobs)} weight jobs, {len(by_pollutant)} pollutants")
    for k, (gname, wkey) in enumerate(jobs, start=1):
        spec = mix_by_group[gname]
        mixer = str(spec["mixer"])
        wkeys = spec.get("weight_keys")
        for sign in (-1, 1):
            w_new = perturb_weight(
                spec["weights"], wkey, pct, sign, mixer, weight_keys=wkeys
            )
            stack_p = stack_with_perturbed_group(
                base_stack, gi_map[gname], spec, w_new, cell_id, valid, fid, nb
            )
            for pol, pspec in by_pollutant.items():
                cids = pspec["cids"]
                sensitive = pspec["sensitive"]
                Wp = fuse_alpha_stack(stack_p, pspec["alpha_row"], cell_id, valid, fid, nb)
                tv_arr = tv_per_cell_all(
                    np.nan_to_num(w0[pol], nan=0.0),
                    np.nan_to_num(Wp, nan=0.0),
                    cell_id,
                )
                tv_all, tv_sens = out[pol]
                for cid in cids:
                    if cid >= tv_arr.size:
                        continue
                    tv = float(tv_arr[cid])
                    if not np.isfinite(tv):
                        continue
                    tv_all.append(tv)
                    if cid in sensitive:
                        tv_sens.append(tv)
                out[pol] = (tv_all, tv_sens)
        if k % 4 == 0 or k == len(jobs):
            log.info(f"Prong B (w) violin multi{tag}: {k}/{len(jobs)} jobs")
    return out
