"""
CEIP α tensor for GNFR C (G1 stationary, G2–G4 off-road) plus optional YAML overrides.

Reads merged profile for ``offroad`` / ``alpha_beta_override`` from
``C_OtherCombustion_rules.yaml`` (merged over groups YAML).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from PROXY.core.alpha.aliases import norm_pollutant_key
from PROXY.core.alpha.ceip_index_loader import (
    default_ceip_profile_relpath,
    remap_legacy_ceip_relpath,
    shared_pollutant_aliases_relpath,
)
from PROXY.core.alpha.ceip_profile_merge import load_merged_ceip_profile
from PROXY.core.alpha.reported_group_alpha import load_ceip_and_alpha
from PROXY.core.dataloaders import resolve_path

logger = logging.getLogger(__name__)

GROUP_ORDER_C: tuple[str, ...] = ("G1", "G2", "G3", "G4")
_EPS_OFF = 1e-15


@dataclass(frozen=True)
class AlphaBetaRow:
    alpha_stat: float
    alpha_off: float
    beta_F: float
    beta_R: float
    beta_B: float


def load_merged_c_other_profile(repo_root: Path) -> dict[str, Any]:
    """Merge ``C_OtherCombustion_groups.yaml`` + ``C_OtherCombustion_rules.yaml``."""
    gy_rel = remap_legacy_ceip_relpath(default_ceip_profile_relpath(repo_root, "C_OtherCombustion", "groups_yaml"))
    ry_rel = remap_legacy_ceip_relpath(default_ceip_profile_relpath(repo_root, "C_OtherCombustion", "rules_yaml"))
    gy = resolve_path(repo_root, Path(gy_rel))
    ry = resolve_path(repo_root, Path(ry_rel))
    return load_merged_ceip_profile(gy, ry if ry.is_file() else None)


def load_shared_pollutant_aliases(repo_root: Path) -> dict[str, str]:
    rel = shared_pollutant_aliases_relpath(repo_root)
    if not rel:
        return {}
    p = resolve_path(repo_root, Path(rel))
    if not p.is_file():
        return {}
    import yaml

    with p.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return {str(k): str(v) for k, v in raw.items()} if isinstance(raw, dict) else {}


def _validate_override_block(override: dict[str, Any], pollutant_outputs: list[str]) -> None:
    pols = override.get("pollutants")
    if not isinstance(pols, dict):
        raise ValueError("alpha_beta_override.pollutants must be a dict when override is set")
    for key in pollutant_outputs:
        pk = norm_pollutant_key(key)
        block = None
        for k, v in pols.items():
            if norm_pollutant_key(str(k)) == pk:
                block = v
                break
        if not isinstance(block, dict):
            raise ValueError(f"alpha_beta_override missing or invalid block for pollutant {key!r}")
        ast = float(block["alpha_stat"])
        aoff = float(block["alpha_off"])
        bF = float(block["beta_F"])
        bR = float(block["beta_R"])
        bB = float(block["beta_B"])
        if abs(ast + aoff - 1.0) > 1e-5:
            raise ValueError(f"override {key}: alpha_stat + alpha_off must be 1, got {ast + aoff}")
        if abs(bF + bR + bB - 1.0) > 1e-5:
            raise ValueError(f"override {key}: beta sum must be 1, got {bF + bR + bB}")


def parse_override(
    merged_profile: dict[str, Any],
    pollutant_outputs: list[str],
) -> dict[str, AlphaBetaRow] | None:
    raw = merged_profile.get("alpha_beta_override")
    if raw is None or raw is False:
        return None
    if not isinstance(raw, dict):
        return None
    pols = raw.get("pollutants")
    if not isinstance(pols, dict) or not pols:
        return None
    _validate_override_block(raw, pollutant_outputs)
    out: dict[str, AlphaBetaRow] = {}
    for key in pollutant_outputs:
        pk = norm_pollutant_key(key)
        block = None
        for k, v in pols.items():
            if norm_pollutant_key(str(k)) == pk:
                block = v
                break
        if not isinstance(block, dict):
            continue
        out[key] = AlphaBetaRow(
            alpha_stat=float(block["alpha_stat"]),
            alpha_off=float(block["alpha_off"]),
            beta_F=float(block["beta_F"]),
            beta_R=float(block["beta_R"]),
            beta_B=float(block["beta_B"]),
        )
    return out if len(out) == len(pollutant_outputs) else None


def ceip_rows_from_tensor(
    *,
    alpha: np.ndarray,
    iso3_list: list[str],
    country_iso3: str,
    pollutant_outputs: list[str],
    fallback_iso3: str,
) -> dict[str, AlphaBetaRow]:
    """Map CEIP α[G1..G4] to alpha_stat, alpha_off, beta_F/R/B per pollutant output name."""
    iso = str(country_iso3).strip().upper()
    fb = str(fallback_iso3).strip().upper()
    if iso in iso3_list:
        ri = iso3_list.index(iso)
    elif fb in iso3_list:
        ri = iso3_list.index(fb)
        logger.warning(
            "[other_combustion] CEIP alpha: country %s not in iso3_list, using fallback %s",
            iso,
            fb,
        )
    else:
        ri = 0
        logger.warning(
            "[other_combustion] CEIP alpha: country %s and fallback %s not in iso3_list, using row 0",
            iso,
            fb,
        )

    out: dict[str, AlphaBetaRow] = {}
    for j, p in enumerate(pollutant_outputs):
        a_g1 = float(alpha[ri, 0, j])
        a_g2 = float(alpha[ri, 1, j])
        a_g3 = float(alpha[ri, 2, j])
        a_g4 = float(alpha[ri, 3, j])
        alpha_stat = a_g1
        alpha_off = a_g2 + a_g3 + a_g4
        if alpha_off <= _EPS_OFF:
            beta_F = beta_R = beta_B = 1.0 / 3.0
            alpha_off = 0.0
            alpha_stat = 1.0
        else:
            beta_F = a_g2 / alpha_off
            beta_R = a_g3 / alpha_off
            beta_B = a_g4 / alpha_off
        out[p] = AlphaBetaRow(
            alpha_stat=alpha_stat,
            alpha_off=alpha_off,
            beta_F=beta_F,
            beta_R=beta_R,
            beta_B=beta_B,
        )
    return out


def merge_rows_with_override(
    ceip_rows: dict[str, AlphaBetaRow],
    override: dict[str, AlphaBetaRow] | None,
) -> dict[str, AlphaBetaRow]:
    if not override:
        return ceip_rows
    merged = dict(ceip_rows)
    merged.update(override)
    return merged


def build_ceip_alpha_context(
    *,
    repo_root: Path,
    ceip_workbook: Path,
    ceip_groups_yaml: Path,
    ceip_rules_yaml: Path | None,
    pollutant_outputs: list[str],
    iso3_list: list[str],
    cntr_code_to_iso3: dict[str, str] | None,
    ceip_years: list[int] | None,
    ceip_pollutant_aliases: dict[str, str] | None = None,
) -> tuple[np.ndarray, list[str], dict[str, AlphaBetaRow] | None]:
    """
    Load CEIP α tensor and optional YAML override rows (per pollutant).

    Returns
    -------
    alpha
        Shape ``(n_iso, 4, n_pol)``.
    iso3_list
        Country axis for ``alpha`` (same order as ``load_ceip_and_alpha``).
    override_rows
        Parsed override dict or ``None``.
    """
    paths_dict: dict[str, Any] = {
        "ceip_workbook": ceip_workbook,
        "ceip_groups_yaml": ceip_groups_yaml,
    }
    if ceip_rules_yaml is not None and ceip_rules_yaml.is_file():
        paths_dict["ceip_rules_yaml"] = ceip_rules_yaml
    if ceip_years:
        paths_dict["ceip_years"] = ceip_years

    merged = load_merged_ceip_profile(
        ceip_groups_yaml,
        ceip_rules_yaml if ceip_rules_yaml is not None and ceip_rules_yaml.is_file() else None,
    )
    override_full = parse_override(merged, pollutant_outputs)

    aliases = load_shared_pollutant_aliases(repo_root)
    if ceip_pollutant_aliases:
        aliases = {**aliases, **ceip_pollutant_aliases}

    cfg: dict[str, Any] = {
        "_project_root": repo_root,
        "pollutants": list(pollutant_outputs),
        "paths": paths_dict,
        "group_order": list(GROUP_ORDER_C),
        "cntr_code_to_iso3": dict(cntr_code_to_iso3 or {}),
        "ceip_pollutant_aliases": aliases,
    }

    alpha, _fb, _wide = load_ceip_and_alpha(
        cfg,
        iso3_list,
        sector_key="C_OtherCombustion",
    )
    return alpha, iso3_list, override_full


def resolve_rows_for_cell(
    *,
    alpha: np.ndarray,
    iso3_list: list[str],
    country_iso3: str,
    fallback_iso3: str,
    pollutant_outputs: list[str],
    override_rows: dict[str, AlphaBetaRow] | None,
) -> dict[str, AlphaBetaRow]:
    ceip_rows = ceip_rows_from_tensor(
        alpha=alpha,
        iso3_list=iso3_list,
        country_iso3=country_iso3,
        pollutant_outputs=pollutant_outputs,
        fallback_iso3=fallback_iso3,
    )
    return merge_rows_with_override(ceip_rows, override_rows)


def offroad_rules_dict(merged_profile: dict[str, Any]) -> dict[str, Any]:
    off = merged_profile.get("offroad")
    return off if isinstance(off, dict) else {}


__all__ = [
    "GROUP_ORDER_C",
    "AlphaBetaRow",
    "build_ceip_alpha_context",
    "ceip_rows_from_tensor",
    "load_merged_c_other_profile",
    "load_shared_pollutant_aliases",
    "merge_rows_with_override",
    "offroad_rules_dict",
    "parse_override",
    "resolve_rows_for_cell",
]
