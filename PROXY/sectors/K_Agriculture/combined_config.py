"""Load K_Agriculture CEIP profile (groups + rules) and derive NUTS2 scalars (lambda, climate)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from PROXY.core.alpha.ceip_profile_merge import load_merged_ceip_profile
from PROXY.core.dataloaders import resolve_path
from PROXY.sectors.K_Agriculture.source_relevance.census_intensity import aggregate_c21_heads_by_nuts2
from PROXY.sectors.K_Agriculture.tabular import emission_factors as ag_ef


def load_koppen_wet_dry_config(root: Path) -> dict[str, Any]:
    p = root / "PROXY" / "config" / "agriculture" / "koppen_wet_dry.yaml"
    if not p.is_file():
        return {"default": "wet", "dry_class_ids": []}
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def load_k_agriculture_rules(cfg: dict[str, Any], root: Path) -> dict[str, Any]:
    """Merged CEIP rules (numeric proxy parameters), excluding ``groups``."""
    paths = cfg.get("paths") or {}
    gy = paths.get("ceip_groups_yaml")
    ry = paths.get("ceip_rules_yaml")
    if not gy or not ry:
        return {}
    gp = resolve_path(root, Path(str(gy)))
    rp = resolve_path(root, Path(str(ry)))
    if not gp.is_file() or not rp.is_file():
        return {}
    merged = load_merged_ceip_profile(gp, rp)
    return {k: v for k, v in merged.items() if k != "groups"}


def _dry_id_set(cfg_k: dict[str, Any]) -> set[int]:
    raw = cfg_k.get("dry_class_ids") or []
    return {int(x) for x in raw}


def compute_lambda_series(cfg: dict[str, Any], root: Path) -> pd.Series:
    """
    lambda_n = sum_s LSU_s * H_{n,s} * f_s^farm / sum_s LSU_s * H_{n,s}
    Returns Series indexed by NUTS_ID (uppercase strings).
    """
    rules = load_k_agriculture_rules(cfg, root)
    liv = rules.get("livestock") or {}
    lsu = liv.get("lsu_per_head") or {}
    ff = liv.get("farm_fraction") or {}
    census = (cfg.get("paths") or {}).get("census") or {}
    gpkg = census.get("c21_gpkg")
    if not gpkg:
        return pd.Series(dtype=np.float64)
    gpkg_path = resolve_path(root, Path(str(gpkg)))
    field_map = ag_ef.load_c21_census_field_map()
    counts = aggregate_c21_heads_by_nuts2(gpkg_path, field_map)
    if counts.empty:
        return pd.Series(dtype=np.float64)

    idx = counts.index.astype(str).str.strip().str.upper()
    num = np.zeros(len(counts), dtype=np.float64)
    den = np.zeros(len(counts), dtype=np.float64)
    for species, col in counts.items():
        skey = str(species).strip().lower()
        lsu_s = float(lsu.get(skey, lsu.get(species, 0.0)) or 0.0)
        f_s = float(ff.get(skey, ff.get(species, 0.5)) or 0.0)
        h = pd.to_numeric(col, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        w = lsu_s * h
        den += w
        num += w * f_s
    lam = np.where(den > 1e-12, num / den, 0.6)
    return pd.Series(lam, index=idx, name="lambda")


def compute_gamma_series(
    cfg: dict[str, Any],
    root: Path,
    *,
    koppen_on_ref: np.ndarray | None,
    nuts_r: np.ndarray,
    nuts_idx_to_id: dict[int, str],
) -> pd.Series:
    """
    Per NUTS2 label: 'wet' or 'dry' from Köppen majority over pixels in that NUTS2.
    If ``koppen_on_ref`` is None, all 'wet'.
    """
    cfg_k = load_koppen_wet_dry_config(root)
    default = str(cfg_k.get("default", "wet")).lower()
    dry_ids = _dry_id_set(cfg_k)
    wet_ids = {int(x) for x in (cfg_k.get("wet_class_ids") or [])}

    out: dict[str, str] = {}
    if koppen_on_ref is None or koppen_on_ref.size == 0:
        for nid in nuts_idx_to_id.values():
            out[str(nid).strip().upper()] = default if default in ("wet", "dry") else "wet"
        return pd.Series(out, dtype=object)

    kop = np.asarray(koppen_on_ref)
    nuts = np.asarray(nuts_r, dtype=np.int32)
    for k, nid in nuts_idx_to_id.items():
        if int(k) <= 0:
            continue
        m = nuts == int(k)
        if not np.any(m):
            out[str(nid).strip().upper()] = default if default in ("wet", "dry") else "wet"
            continue
        vals = kop[m]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            out[str(nid).strip().upper()] = default if default in ("wet", "dry") else "wet"
            continue
        dry_votes = 0
        wet_votes = 0
        for v in vals.ravel():
            vi = int(round(float(v)))
            if vi in dry_ids:
                dry_votes += 1
            elif wet_ids and vi in wet_ids:
                wet_votes += 1
            else:
                wet_votes += 1
        nid_u = str(nid).strip().upper()
        if dry_ids and dry_votes > wet_votes:
            out[nid_u] = "dry"
        else:
            out[nid_u] = "wet"
    return pd.Series(out, dtype=object)
