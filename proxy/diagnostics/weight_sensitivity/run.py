from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from proxy.core import log
from proxy.core.alias import resolve_country_profile
from proxy.diagnostics.weight_sensitivity.load_exports import MULTI_GROUP_SECTORS
from proxy.diagnostics.weight_sensitivity.pollutants_config import reference_pollutants_from_cfg
from proxy.diagnostics.weight_sensitivity.prong_a import run_all_prong_a
from proxy.diagnostics.weight_sensitivity.prong_a_w import run_all_prong_a_w


def _country_tag(country: str) -> str:
    return resolve_country_profile(country)["full_name"].replace(" ", "_")


def run_prong_a(
    repo_root: Path,
    *,
    country: str,
    year: int,
    sector_keys: tuple[str, ...] | list[str] | str,
    export_root: Path | None = None,
    active_eps: float,
    similarity_threshold: float,
) -> list[dict[str, Any]]:
    root = Path(repo_root)
    if export_root is not None:
        exp = Path(export_root)
    else:
        exp = root / "OUTPUT" / "Proxy_diagnostics" / "W_groups"
        if not exp.is_dir() and (root / "Output" / "Proxy_diagnostics" / "W_groups").is_dir():
            exp = root / "Output" / "Proxy_diagnostics" / "W_groups"
    if sector_keys == "multi_group":
        keys = MULTI_GROUP_SECTORS
    else:
        keys = tuple(sector_keys)
    tag = _country_tag(country)
    cfg = load_prong_a_settings(root)
    pollutants = reference_pollutants_from_cfg(cfg)
    return run_all_prong_a(
        exp,
        tag,
        year,
        keys,
        active_eps=active_eps,
        similarity_threshold=similarity_threshold,
        reference_pollutants=pollutants,
    )


def run_prong_a_w(
    repo_root: Path,
    *,
    country: str,
    year: int,
    sector_keys: tuple[str, ...] | list[str] | str,
    export_root: Path | None = None,
    active_eps: float,
    similarity_threshold: float,
) -> list[dict[str, Any]]:
    root = Path(repo_root)
    if export_root is not None:
        exp = Path(export_root)
    else:
        exp = root / "OUTPUT" / "Proxy_diagnostics" / "W_groups"
        if not exp.is_dir() and (root / "Output" / "Proxy_diagnostics" / "W_groups").is_dir():
            exp = root / "Output" / "Proxy_diagnostics" / "W_groups"
    if sector_keys == "mix_export":
        keys = _sectors_with_mix(exp, _country_tag(country), year)
    else:
        keys = tuple(sector_keys)
    tag = _country_tag(country)
    cfg = load_prong_a_w_settings(root)
    pollutants = reference_pollutants_from_cfg(cfg)
    return run_all_prong_a_w(
        exp,
        tag,
        year,
        keys,
        active_eps=active_eps,
        similarity_threshold=similarity_threshold,
        reference_pollutants=pollutants,
    )


def _sectors_with_mix(export_root: Path, country_tag: str, year: int) -> tuple[str, ...]:
    import yaml

    out: list[str] = []
    if not export_root.is_dir():
        return ()
    for sector_dir in sorted(export_root.iterdir()):
        if not sector_dir.is_dir():
            continue
        bundle = sector_dir / f"{country_tag}_{int(year)}"
        manifest_path = bundle / "groups_manifest.yaml"
        if not manifest_path.is_file():
            continue
        with manifest_path.open(encoding="utf-8") as f:
            doc = yaml.safe_load(f)
        if isinstance(doc, dict) and doc.get("mix"):
            out.append(sector_dir.name)
    return tuple(out)


def load_prong_a_w_settings(repo_root: Path) -> dict[str, Any]:
    path = repo_root / "proxy" / "config" / "weight_sensitivity_prong_a_w.yaml"
    if not path.is_file():
        return load_prong_a_settings(repo_root)
    with path.open(encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    return doc if isinstance(doc, dict) else {}


def load_prong_a_settings(repo_root: Path) -> dict[str, Any]:
    path = repo_root / "proxy" / "config" / "weight_sensitivity_prong_a.yaml"
    if not path.is_file():
        return {
            "reference_pollutants": ["PM10", "NOx", "SOx", "NMVOC"],
            "similarity_threshold": 0.7,
            "active_eps": 1e-9,
            "year": 2019,
        }
    with path.open(encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    return doc if isinstance(doc, dict) else {}
