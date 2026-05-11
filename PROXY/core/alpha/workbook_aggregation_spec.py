"""Alpha-workbook GNFR aggregation: derive mapping + code groups from CEIP profile YAMLs.

The former ``ceip/alpha/mapping_gnfr_to_nfr2.yaml`` and ``grouped_subsectors.yaml`` duplicated
information that already lives under ``PROXY/config/ceip/profiles/``. This module rebuilds the
same structures for :func:`compute_alpha` so those standalone files are not needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ._common import _load_yaml

# GNFR-wide workbook sector filters (NFR-style tokens; ``*`` = prefix match in compute_alpha).
_GNFR_SECTOR_TOTAL_CODES: dict[str, list[str]] = {
    "A_PublicPower": ["1A1A*"],
    "B_Industry": ["1A2*"],
    "C_OtherCombustion": ["1A4*"],
    "D_Fugitive": ["1B*"],
    "E_Solvents": ["2D3*"],
    "G_Shipping": ["1A3D*"],
    "I_Offroad": ["1A3*"],
    "J_Waste": ["5*"],
    "K_Agriculture": ["3*"],
}

# I_Offroad triple-leg codes (same semantics as legacy ``grouped_subsectors.yaml``).
_OFFROAD_GROUPS: dict[str, list[str]] = {
    "offroad_railways": ["1A3C"],
    "offroad_pipelines": ["1A3EI"],
    "offroad_non_road": ["1A3EII"],
}

# Explicit union for solvents ``d3_all`` (avoid relying on a bare ``2D3`` token in workbooks).
_SOLVENTS_D3_ALL_CODES: list[str] = [
    "2D3A",
    "2D3B",
    "2D3C",
    "2D3D",
    "2D3E",
    "2D3F",
    "2D3G",
    "2D3H",
    "2D3I",
]


def load_workbook_aggregation_spec(repo_root: Path) -> tuple[dict[str, Any], dict[str, list[str]]]:
    """
    Return ``(mapping, grouped)`` compatible with :mod:`PROXY.core.alpha.compute`.

    ``mapping`` is the inner ``mapping:`` dict (GNFR keys -> sector_total_codes + subsectors).
    ``grouped`` maps group name -> list of sector tokens for ``group:`` expansion.
    """
    cfg_dir = repo_root / "PROXY" / "config" / "ceip" / "profiles"
    solvents = _load_yaml(cfg_dir / "E_Solvents_groups.yaml")
    waste = _load_yaml(cfg_dir / "J_Waste_groups.yaml")
    c_other = _load_yaml(cfg_dir / "C_OtherCombustion_groups.yaml")

    grouped: dict[str, list[str]] = {}
    grouped["solvents_d3_all"] = list(_SOLVENTS_D3_ALL_CODES)

    _waste_profile_keys = (
        ("G1", "waste_solid"),
        ("G2", "waste_wastewater"),
        ("G3", "waste_residual"),
    )
    _waste_legacy_keys = (
        ("solid", "waste_solid"),
        ("ww", "waste_wastewater"),
        ("res", "waste_residual"),
    )
    groups_w = waste.get("groups") if isinstance(waste.get("groups"), dict) else {}
    families = waste.get("families") or {}
    if isinstance(groups_w, dict) and groups_w:
        use_g = any(k in groups_w for k in ("G1", "G2", "G3"))
        keys_iter = _waste_profile_keys if use_g else _waste_legacy_keys
        for fam_key, gname in keys_iter:
            block = groups_w.get(fam_key)
            if not isinstance(block, dict):
                continue
            codes = block.get("ceip_sectors") or block.get("nfr") or []
            if isinstance(codes, list):
                grouped[gname] = [str(x).strip() for x in codes if str(x).strip()]
    elif isinstance(families, dict):
        for fam_key, gname in _waste_legacy_keys:
            codes = families.get(fam_key)
            if isinstance(codes, list):
                grouped[gname] = [str(x).strip() for x in codes if str(x).strip()]

    grouped.update(_OFFROAD_GROUPS)

    # C_OtherCombustion: G1–G4 from CEIP profile (stationary + three off-road legs).
    groups_c = c_other.get("groups") if isinstance(c_other.get("groups"), dict) else {}
    for gid in ("G1", "G2", "G3", "G4"):
        block = groups_c.get(gid)
        if not isinstance(block, dict):
            continue
        raw = block.get("ceip_sectors") or []
        if isinstance(raw, list):
            gname = f"gnfr_c_{gid}"
            grouped[gname] = [str(x).strip() for x in raw if str(x).strip()]

    mapping: dict[str, Any] = {}
    for gnfr_key, sector_total in _GNFR_SECTOR_TOTAL_CODES.items():
        mapping[gnfr_key] = {"sector_total_codes": list(sector_total), "subsectors": {}}

    # E_Solvents from profile ``subsectors`` blocks.
    sol_subs = solvents.get("subsectors") or {}
    e_sub: dict[str, Any] = {}
    if isinstance(sol_subs, dict):
        for sub_name, block in sol_subs.items():
            if not isinstance(block, dict):
                continue
            raw = block.get("ceip_sectors") or []
            codes = [str(x).strip() for x in raw if str(x).strip()]
            if codes:
                e_sub[str(sub_name)] = {"codes": codes}
        e_sub["d3_all"] = {"groups": ["solvents_d3_all"]}
    mapping["E_Solvents"]["subsectors"] = e_sub

    # J_Waste: subsector keys match CEIP ``groups`` ids (G1–G3) or legacy solid/ww/res.
    j_sub: dict[str, Any] = {}
    j_pairs = (
        (("G1", "waste_solid"), ("G2", "waste_wastewater"), ("G3", "waste_residual"))
        if isinstance(waste.get("groups"), dict) and any(k in (waste.get("groups") or {}) for k in ("G1", "G2", "G3"))
        else (("solid", "waste_solid"), ("ww", "waste_wastewater"), ("res", "waste_residual"))
    )
    for fam_key, gname in j_pairs:
        if gname in grouped:
            j_sub[fam_key] = {"groups": [gname]}
    mapping["J_Waste"]["subsectors"] = j_sub

    # I_Offroad triple legs.
    mapping["I_Offroad"]["subsectors"] = {
        "offroad_railways": {"groups": ["offroad_railways"]},
        "offroad_pipelines": {"groups": ["offroad_pipelines"]},
        "offroad_non_road": {"groups": ["offroad_non_road"]},
    }

    # C_OtherCombustion: subsector keys align with CEIP group ids G1–G4.
    c_sub: dict[str, Any] = {}
    for gid in ("G1", "G2", "G3", "G4"):
        gname = f"gnfr_c_{gid}"
        if gname in grouped:
            c_sub[gid] = {"groups": [gname]}
    if c_sub:
        mapping["C_OtherCombustion"]["subsectors"] = c_sub

    return mapping, grouped
