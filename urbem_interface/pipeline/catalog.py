"""
Proxy catalog: merge downscaling roles (proxies + gnfr_to_proxy) with auxiliary
factory outputs and optional semantic labels. Backward compatible with legacy JSON.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

# Default factory outputs not referenced by gnfr_to_proxy in config/proxies.json
DEFAULT_AUXILIARY_PROXIES: list[dict[str, Any]] = [
    {
        "id": "eprtr_snap34",
        "file": "Proxy_EPRTR_SNAP34.tif",
        "snap": 34,
        "used_in_downscaling": False,
        "description": "E-PRTR + CORINE fallback for SNAP34-class activities; factory output for snap_proxy_map.",
    },
    {
        "id": "eprtr_snap1",
        "file": "Proxy_EPRTR_SNAP1.tif",
        "snap": 1,
        "used_in_downscaling": False,
        "description": "E-PRTR + CORINE fallback for SNAP1-class activities; factory output for snap_proxy_map.",
    },
]

# Optional semantic names for gnfr_to_proxy *values* (role keys in "proxies")
DEFAULT_SEMANTIC_PROXY_ROLES: dict[str, str] = {
    "industry": "corine_industry_commercial_energy",
    "waste_wastewater": "eprtr_waste_wastewater_corine",
    "agriculture": "corine_agriculture_combined",
    "shipping": "shipping_ports_merged",
    "aviation": "airports_corine",
    "offroad": "non_road_mobility_combined",
    "population": "ghs_population_density",
}

# Reserved for pollutant-specific or class-specific agriculture (Topic 4)
DEFAULT_FUTURE_KEYS: dict[str, Any] = {
    "_comment_agri": (
        "Planned: agri_subproxies {role: filename}, pollutant_proxy_weights for "
        "GNFR x pollutant -> proxy overrides without breaking mass conservation in proxy_cwd."
    ),
    "agri_subproxies": {},
    "pollutant_proxy_weights": {},
}


@dataclass
class ProxyCatalogEntry:
    id: str
    file: str
    used_in_downscaling: bool
    description: str = ""
    snap: int | None = None


@dataclass
class ProxyPipelineBundle:
    """Normalized view of proxies.json for tooling and future orchestration."""

    raw: dict[str, Any]
    proxies_folder_key: str | None = None
    role_to_file: dict[str, str] = field(default_factory=dict)
    gnfr_to_proxy: dict[str, str] = field(default_factory=dict)
    downscaling_roles: set[str] = field(default_factory=set)
    auxiliary: list[ProxyCatalogEntry] = field(default_factory=list)
    semantic_proxy_roles: dict[str, str] = field(default_factory=dict)
    future: dict[str, Any] = field(default_factory=dict)


def _aux_from_raw(raw: dict[str, Any]) -> list[ProxyCatalogEntry]:
    items = raw.get("auxiliary_proxies")
    if not items:
        items = DEFAULT_AUXILIARY_PROXIES
    out: list[ProxyCatalogEntry] = []
    for x in items:
        if not isinstance(x, dict):
            continue
        out.append(
            ProxyCatalogEntry(
                id=str(x.get("id", "")),
                file=str(x.get("file", "")),
                used_in_downscaling=bool(x.get("used_in_downscaling", False)),
                description=str(x.get("description", "")),
                snap=int(x["snap"]) if x.get("snap") is not None else None,
            )
        )
    return out


def merge_proxy_catalog(raw: dict[str, Any]) -> dict[str, Any]:
    """
    Return a deep copy of raw with defaults filled for optional sections.
    Does not remove keys; safe to pass to existing area_sources / line_sources.
    """
    cfg = copy.deepcopy(raw)
    if "auxiliary_proxies" not in cfg:
        cfg["auxiliary_proxies"] = copy.deepcopy(DEFAULT_AUXILIARY_PROXIES)
    sem = cfg.get("semantic_proxy_roles")
    if not isinstance(sem, dict):
        cfg["semantic_proxy_roles"] = copy.deepcopy(DEFAULT_SEMANTIC_PROXY_ROLES)
    else:
        merged = copy.deepcopy(DEFAULT_SEMANTIC_PROXY_ROLES)
        merged.update({str(k): str(v) for k, v in sem.items()})
        cfg["semantic_proxy_roles"] = merged
    fut = cfg.get("future_disaggregation")
    if not isinstance(fut, dict):
        cfg["future_disaggregation"] = copy.deepcopy(DEFAULT_FUTURE_KEYS)
    else:
        base = copy.deepcopy(DEFAULT_FUTURE_KEYS)
        base.update(fut)
        cfg["future_disaggregation"] = base
    return cfg


def build_proxy_pipeline_bundle(raw: dict[str, Any]) -> ProxyPipelineBundle:
    merged = merge_proxy_catalog(raw)
    proxies = merged.get("proxies") or {}
    gmap = merged.get("gnfr_to_proxy") or {}
    role_to_file = {str(k): str(v) for k, v in proxies.items() if isinstance(v, str)}
    downscaling_roles = {str(v) for v in gmap.values()}
    aux = _aux_from_raw(merged)
    sem = merged.get("semantic_proxy_roles") or {}
    fut = merged.get("future_disaggregation") or {}
    return ProxyPipelineBundle(
        raw=merged,
        role_to_file=role_to_file,
        gnfr_to_proxy={str(k): str(v) for k, v in gmap.items()},
        downscaling_roles=downscaling_roles,
        auxiliary=aux,
        semantic_proxy_roles={str(k): str(v) for k, v in sem.items()},
        future=dict(fut),
    )


def validate_proxy_files(proxies_folder, bundle: ProxyPipelineBundle) -> list[str]:
    """Return list of human-readable issues (missing files)."""
    from pathlib import Path

    root = Path(proxies_folder)
    issues: list[str] = []
    seen: set[str] = set()
    for role, fname in bundle.role_to_file.items():
        p = root / fname
        if not p.is_file():
            issues.append(f"downscaling role {role!r} -> missing {p}")
        seen.add(fname)
    for entry in bundle.auxiliary:
        if not entry.file:
            continue
        p = root / entry.file
        if not p.is_file():
            issues.append(f"auxiliary {entry.id!r} -> missing {p}")
    ghsl = bundle.raw.get("ghsl_urbancentre")
    if ghsl:
        p = root / str(ghsl)
        if not p.is_file():
            issues.append(f"GHSL urban centre -> missing {p}")
    return issues
