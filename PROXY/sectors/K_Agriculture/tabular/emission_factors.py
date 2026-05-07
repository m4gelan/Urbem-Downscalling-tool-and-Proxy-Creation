"""YAML-backed emission-factor and stage-1-score constants for K_Agriculture."""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_EF_YAML = Path("PROXY") / "config" / "agriculture" / "emission_factors.yaml"
# Legacy per-file JSON (removed from repo after migration).
_LEGACY_EF_DIR = Path("PROXY") / "config" / "agriculture" / "emission_factors"


def _repo_root_candidates() -> list[Path]:
    here = Path(__file__).resolve()
    return [here.parents[4], Path.cwd()]


def _ef_yaml_path() -> Path:
    for ancestor in _repo_root_candidates():
        candidate = ancestor / _EF_YAML
        if candidate.is_file():
            return candidate
    return _repo_root_candidates()[0] / _EF_YAML


@lru_cache(maxsize=1)
def load_emission_factors_document() -> dict[str, Any]:
    """Full merged agriculture science YAML (single source of truth)."""
    path = _ef_yaml_path()
    if not path.is_file():
        logger.warning("emission_factors: missing %s; callers use module defaults.", path)
        return {}
    try:
        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("emission_factors: failed to read %s (%s); using empty doc.", path, exc)
        return {}


def clear_emission_factors_cache() -> None:
    load_emission_factors_document.cache_clear()


def _load_json_legacy(name: str) -> dict[str, Any] | None:
    for ancestor in _repo_root_candidates():
        candidate = ancestor / _LEGACY_EF_DIR / name
        if candidate.is_file():
            try:
                with candidate.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as exc:  # noqa: BLE001
                logger.warning("emission_factors: failed legacy read %s (%s)", candidate, exc)
                return None
    return None


def _section(key: str) -> dict[str, Any]:
    doc = load_emission_factors_document()
    block = doc.get(key)
    return block if isinstance(block, dict) else {}


@lru_cache(maxsize=1)
def load_livestock_housing_nh3() -> dict[str, Any]:
    data = _section("livestock_housing_nh3") or _load_json_legacy("livestock_housing_nh3.json") or {}
    tiers = data.get("tiers") or {}
    return {
        "u111_building": float(tiers.get("u111_building", {}).get("score", 1.00)),
        "u111_grassland": float(tiers.get("u111_grassland", {}).get("score", 0.55)),
        "u111_cropland_generic": float(tiers.get("u111_cropland_generic", {}).get("score", 0.20)),
        "u111_ley": float(tiers.get("u111_ley", {}).get("score", 0.30)),
        "u112_fallow": float(tiers.get("u112_fallow", {}).get("score", 0.20)),
        "missing_grazing_damping": float(data.get("missing_grazing_damping", 0.40)),
        "grazing_observed_score": float(data.get("grazing_observed_score", 0.00)),
        "excluded_lu1": list(data.get("excluded_lu1", ["U113"])),
    }


@lru_cache(maxsize=1)
def load_manure_land_application() -> dict[str, float]:
    data = _section("manure_land_application") or _load_json_legacy("manure_land_application.json") or {}
    groups = data.get("crop_groups") or {}

    def _w(name: str, default: float) -> float:
        entry = groups.get(name) or {}
        try:
            return float(entry.get("weight", default))
        except (TypeError, ValueError):
            return default

    return {
        "high_manure_exact": _w("high_manure_exact", 0.80),
        "high_manure_prefix": _w("high_manure_prefix", 0.80),
        "intermediate_manure": _w("intermediate_manure", 0.60),
        "residual_b_prefix": _w("residual_b_prefix", 0.10),
        "managed_grassland": _w("managed_grassland", 0.80),
        "fallow": _w("fallow", 0.70),
        "u111_other": _w("u111_other", 0.10),
        "u113_other": _w("u113_other", 0.10),
    }


@lru_cache(maxsize=1)
def load_nmvoc_crop_ef() -> dict[str, float]:
    data = _section("nmvoc_crop_ef") or _load_json_legacy("nmvoc_crop_ef.json") or {}
    raw = data.get("lc1_ef") or {}
    out: dict[str, float] = {}
    for k, v in raw.items():
        try:
            out[str(k).upper()] = float(v)
        except (TypeError, ValueError):
            continue
    if not out:
        out = {
            "B11": 0.32, "B12": 0.32, "B13": 1.03, "B14": 0.32, "B15": 0.32,
            "B16": 0.32, "B17": 0.32, "B18": 0.32, "B19": 0.32, "B21": 0.32,
            "B22": 0.32, "B23": 0.32, "B32": 1.34, "B41": 0.32, "B51": 0.41,
            "B52": 0.41, "B53": 0.41, "B54": 0.41, "B55": 0.41, "E10": 0.41,
            "E20": 0.41,
        }
    return out


@lru_cache(maxsize=1)
def load_residue_ratios() -> dict[str, float]:
    data = _section("residue_ratios") or _load_json_legacy("residue_ratios.json") or {}
    raw = data.get("lc1_ratio") or {}
    out: dict[str, float] = {}
    for k, v in raw.items():
        try:
            out[str(k).upper()] = float(v)
        except (TypeError, ValueError):
            continue
    if not out:
        out = {
            "B11": 1.3, "B12": 1.3, "B13": 1.2, "B14": 1.6, "B15": 1.3,
            "B16": 1.0, "B17": 1.4, "B18": 1.2, "B19": 1.2, "B33": 2.1,
            "B41": 1.7,
        }
    return out


@lru_cache(maxsize=1)
def load_ipcc_defaults() -> dict[str, dict[str, float]]:
    data = _section("ipcc_defaults") or _load_json_legacy("ipcc_defaults.json") or {}
    ch4 = data.get("ch4_enteric_ef_kg_per_head_per_year") or {}
    man = data.get("manure_n_kg_per_head_per_year") or {}

    def _f(d: dict[str, Any], key: str, default: float) -> float:
        try:
            return float(d.get(key, default))
        except (TypeError, ValueError):
            return default

    return {
        "ch4_ef": {
            "dairy_cow": _f(ch4, "dairy_cow", 138.0),
            "other_cattle": _f(ch4, "other_cattle", 52.0),
            "pig_light": _f(ch4, "pig_light", 1.0),
            "pig_heavy": _f(ch4, "pig_heavy", 1.5),
            "sheep_blend": _f(ch4, "sheep_blend", (9.0 * 65.0 + 5.0 * 45.0) / (65.0 + 45.0)),
            "goat_blend": _f(ch4, "goat_blend", (9.0 * 50.0 + 5.0 * 28.0) / (50.0 + 28.0)),
        },
        "manure_n": {
            "dairy_cow": _f(man, "dairy_cow", 0.48),
            "other_cattle": _f(man, "other_cattle", 0.33),
            "swine": _f(man, "swine", 0.5),
            "sheep": _f(man, "sheep", 0.85),
            "goat": _f(man, "goat", 1.28),
        },
    }


def load_c21_census_field_map() -> dict[str, Any]:
    """C21 layer/column mapping for census intensity (from merged EF YAML)."""
    data = _section("c21_census_field_map") or _load_json_legacy("c21_census_field_map.json") or {}
    return dict(data) if data else {}


def load_enteric_fermentation_factors() -> dict[str, Any]:
    data = _section("enteric_fermentation_factors") or {}
    if not data:
        legacy = _load_json_legacy("enteric_fermentation_factors.json")
        data = legacy or {}
    return dict(data) if data else {}


def load_livestock_housing_factors() -> dict[str, Any]:
    data = _section("livestock_housing_factors") or {}
    if not data:
        legacy = _load_json_legacy("livestock_housing_factors.json")
        data = legacy or {}
    return dict(data) if data else {}
