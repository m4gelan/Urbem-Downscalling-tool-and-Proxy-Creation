from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from proxy.core import log
from proxy.diagnostics.weight_sensitivity.cell_metrics import (
    _flat_valid,
    compute_prong_a,
    tv_per_cell_all,
)
from proxy.diagnostics.weight_sensitivity.load_exports import load_bundle


def alpha_entropy(alpha_row: np.ndarray) -> float:
    a = np.asarray(alpha_row, dtype=np.float64)
    a = a[a > 0]
    if a.size == 0:
        return 0.0
    return float(-np.sum(a * np.log(a)))


def pick_pollutant_index(
    alpha_doc: dict[str, Any],
    mode: str,
) -> int:
    labels = alpha_doc.get("pollutant_labels") or []
    mat = np.asarray(alpha_doc.get("alpha") or [], dtype=np.float64)
    if mat.size == 0 or not labels:
        return 0
    if mode == "auto_entropy":
        ent = [alpha_entropy(mat[j]) for j in range(mat.shape[0])]
        return int(np.argmax(ent))
    if mode == "auto_dominant":
        dom = [float(np.max(mat[j])) for j in range(mat.shape[0])]
        return int(np.argmax(dom))
    if mode in labels:
        return labels.index(mode)
    raise ValueError(f"unknown pollutant pick mode {mode!r}")


def pick_reference_pollutant(
    alpha_doc: dict[str, Any],
    mode: str,
    reference_pollutants: list[str],
) -> str:
    """Like pick_pollutant_index but auto_* modes only consider reference_pollutants."""
    labels = alpha_doc.get("pollutant_labels") or reference_pollutants
    mat = np.asarray(alpha_doc.get("alpha") or [], dtype=np.float64)
    allowed = [i for i, l in enumerate(labels) if l in reference_pollutants]
    if not allowed or mat.size == 0:
        return reference_pollutants[0]
    if mode in reference_pollutants:
        return mode
    if mode == "auto_entropy":
        ix = max(allowed, key=lambda i: alpha_entropy(mat[i]))
        return labels[ix]
    if mode == "auto_dominant":
        ix = max(allowed, key=lambda i: float(np.max(mat[i])))
        return labels[ix]
    if mode in labels and mode in reference_pollutants:
        return mode
    return labels[allowed[0]]


def renormalize_alpha(alpha: np.ndarray) -> np.ndarray:
    a = np.asarray(alpha, dtype=np.float64).copy()
    s = a.sum()
    if s <= 0:
        raise ValueError("alpha row sums to zero")
    return (a / s).astype(np.float32)


def perturb_alpha_local(alpha: np.ndarray, g: int, pct: float, sign: int) -> np.ndarray:
    a = renormalize_alpha(alpha).astype(np.float64)
    factor = 1.0 + sign * float(pct)
    a[g] *= factor
    a = np.maximum(a, 0.0)
    return renormalize_alpha(a)


def _normalize_S(S: np.ndarray, cell_id: np.ndarray, valid: np.ndarray, fid: np.ndarray, nb: int) -> np.ndarray:
    flat_s = np.nan_to_num(S.ravel(), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    flat_id = cell_id.ravel()
    sums = np.bincount(fid, weights=flat_s[valid], minlength=nb)
    counts = np.bincount(fid, minlength=nb)
    out = np.zeros(flat_s.size, dtype=np.float32)
    idx = np.maximum(flat_id, 0)
    d = sums[idx]
    c = counts[idx]
    v = flat_id >= 0
    pos = v & (d > 0.0)
    np.divide(flat_s, d, out=out, where=pos)
    uniform = v & (d <= 0.0) & (c > 0)
    out[uniform] = (1.0 / c[uniform]).astype(np.float32)
    return out.reshape(S.shape)


def fuse_alpha_stack(
    stack: np.ndarray,
    alpha: np.ndarray,
    cell_id: np.ndarray,
    valid: np.ndarray,
    fid: np.ndarray,
    nb: int,
) -> np.ndarray:
    """``stack`` shape (G, H, W); fuse with α then per-cell normalize."""
    a = np.asarray(alpha, dtype=np.float32).ravel()
    if a.shape[0] != stack.shape[0]:
        raise ValueError(f"alpha len {a.shape[0]} != stack groups {stack.shape[0]}")
    S = np.tensordot(a, stack.astype(np.float32), axes=(0, 0))
    return _normalize_S(S, cell_id, valid, fid, nb)


def load_sensitive_from_prong_a(
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
            / "prong_a"
            / sector_key
            / f"{country_tag}_{int(year)}"
            / "summary.json"
        )
        if p.is_file():
            doc = json.loads(p.read_text(encoding="utf-8"))
            ids = doc.get("sensitive_cell_ids") or []
            return {int(x) for x in ids}
    return None


def prepare_b2_state(
    data: dict[str, Any],
    *,
    repo_root: Path | None = None,
    sector_key: str | None = None,
    country_tag: str | None = None,
    year: int | None = None,
    active_eps: float = 1e-9,
    similarity_threshold: float = 0.7,
) -> dict[str, Any]:
    group_names = list(data["group_names"])
    stack = np.stack([data["W_by_group"][g] for g in group_names], axis=0)
    cell_id = data["cell_id"]
    valid, fid, nb = _flat_valid(cell_id)
    nox = data["nox_by_cell"]
    cids = [c for c in nox if nox[c] > 0]

    sensitive: set[int] | None = None
    if repo_root and sector_key and country_tag and year is not None:
        sensitive = load_sensitive_from_prong_a(repo_root, sector_key, country_tag, year)
    if sensitive is None:
        log.info("Prong B: computing sensitive cells (no prong_a summary.json)")
        pa = compute_prong_a(
            data["W_by_group"],
            cell_id,
            nox,
            active_eps=active_eps,
            similarity_threshold=similarity_threshold,
        )
        sensitive = set(pa.get("sensitive_cell_ids") or [])
    else:
        log.info(f"Prong B: loaded {len(sensitive)} sensitive cells from prong_a summary")

    return {
        "group_names": group_names,
        "stack": stack,
        "cell_id": cell_id,
        "valid": valid,
        "fid": fid,
        "nb": nb,
        "nox": nox,
        "cids": cids,
        "sensitive": sensitive,
        "alpha_doc": data["alpha"],
    }


def run_b2_on_state(
    state: dict[str, Any],
    pollutant_ix: int,
    perturb_pct: list[float],
    *,
    label: str = "",
) -> dict[str, Any]:
    alpha_doc = state["alpha_doc"]
    mat = np.asarray(alpha_doc.get("alpha") or [], dtype=np.float32)
    g = len(state["group_names"])
    if mat.size == 0:
        alpha_row = np.ones(g, dtype=np.float32) / float(g)
    else:
        alpha_row = mat[pollutant_ix].astype(np.float32)

    stack = state["stack"]
    cell_id = state["cell_id"]
    valid, fid, nb = state["valid"], state["fid"], state["nb"]
    nox = state["nox"]
    cids = state["cids"]
    sensitive = state["sensitive"]

    tag = f" {label}" if label else ""
    log.info(f"Prong B{tag}: fuse baseline W0 ({g} groups)")
    W0 = fuse_alpha_stack(stack, alpha_row, cell_id, valid, fid, nb)

    tv_all: list[float] = []
    tv_sens: list[float] = []
    n_pert = len(perturb_pct) * g * 2
    k = 0
    for pct in perturb_pct:
        for gi in range(g):
            for sign in (-1, 1):
                k += 1
                ap = perturb_alpha_local(alpha_row, gi, pct, sign)
                Wp = fuse_alpha_stack(stack, ap, cell_id, valid, fid, nb)
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
                    log.info(f"Prong B{tag}: perturb {k}/{n_pert}")

    def _stats(vals: list[float]) -> dict[str, float]:
        if not vals:
            return {"mean": 0.0, "p90": 0.0}
        a = np.asarray(vals, dtype=np.float64)
        return {"mean": float(a.mean()), "p90": float(np.percentile(a, 90))}

    return {
        "tv_all": _stats(tv_all),
        "tv_sensitive": _stats(tv_sens),
        "n_sensitive_cells": len(sensitive),
        "pollutant_ix": pollutant_ix,
        "group_names": state["group_names"],
    }


def run_b2_local(
    bundle_dir: str | Path,
    pollutant_ix: int,
    perturb_pct: list[float],
    *,
    repo_root: Path | None = None,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if data is None:
        log.info(f"Prong B: loading bundle {bundle_dir}")
        data = load_bundle(Path(bundle_dir))
    sk = data["manifest"].get("sector_key", "")
    ct = data["manifest"].get("country_tag", "")
    yr = int(data["manifest"].get("year", 2019))
    state = prepare_b2_state(
        data,
        repo_root=repo_root,
        sector_key=str(sk),
        country_tag=str(ct),
        year=yr,
    )
    return run_b2_on_state(state, pollutant_ix, perturb_pct, label=str(sk))


def crop_cell_patch(W: np.ndarray, cell_id: np.ndarray, cid: int) -> np.ndarray:
    """Minimal bounding box of pixels in CAMS cell ``cid`` (preserves grid layout)."""
    m = cell_id == int(cid)
    if not m.any():
        raise ValueError(f"no pixels for cell_id {cid}")
    rows = np.where(m.any(axis=1))[0]
    cols = np.where(m.any(axis=0))[0]
    r0, r1 = int(rows[0]), int(rows[-1]) + 1
    c0, c1 = int(cols[0]), int(cols[-1]) + 1
    return np.asarray(W[r0:r1, c0:c1], dtype=np.float32)


def run_b2_all_representatives(
    export_root: Path,
    repo_root: Path,
    cfg: dict[str, Any],
    country_tag: str,
    year: int,
) -> dict[str, Any]:
    """One disk load per representative sector; primary + optional contrast."""
    pct = [float(x) for x in cfg["perturb_pct"]]
    summaries: dict[str, Any] = {}
    for role, rep in cfg["representative_sectors"].items():
        sk = rep["sector_key"]
        bdir = export_root / sk / f"{country_tag}_{int(year)}"
        log.info(f"Prong B role={role} sector={sk}")
        data = load_bundle(bdir)
        state = prepare_b2_state(data, repo_root=repo_root, sector_key=sk, country_tag=country_tag, year=year)
        pix = pick_pollutant_index(state["alpha_doc"], rep["primary_pollutant"])
        summaries[role] = {
            "sector": sk,
            "primary_ix": pix,
            "b2": run_b2_on_state(state, pix, pct, label=f"{sk} primary"),
        }
        if rep.get("contrast_pollutant"):
            cix = pick_pollutant_index(state["alpha_doc"], rep["contrast_pollutant"])
            summaries[role]["contrast_ix"] = cix
            summaries[role]["contrast_b2"] = run_b2_on_state(
                state, cix, pct, label=f"{sk} contrast",
            )
    return summaries
