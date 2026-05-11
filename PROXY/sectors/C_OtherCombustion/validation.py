"""
Startup validation for GNFR C other-combustion (fail-fast).

Checks mapping coverage, morphology weights, CORINE L3 codes in the LUT, Eurostat
cache/API reachability, and that each configured pollutant can resolve at least one EF.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from PROXY.core.corine.encoding import default_corine_index_map_path, load_ordered_l3_codes
from PROXY.core.dataloaders import resolve_path

from .constants import MODEL_CLASSES
from .eurostat_api import _cache_path, eurostat_geo_for_iso3
from .exceptions import ConfigurationError
from .m_builder.emep_ef import ef_kg_per_tj, load_emep
from .m_builder.mapping_io import load_gains_mapping
from .m_builder.sidecar_io import load_sidecar_dict
from ._log import LOG
from .x_builder.appliance_proxy_config import load_appliance_proxy_from_rules_yaml


def validate_pipeline_config(
    *,
    repo_root: Path,
    cfg: dict[str, Any],
    mapping_path: Path,
    emep_path: Path,
    sidecar_path: Path,
    pollutant_outputs: list[str],
) -> None:
    rules, _ = load_gains_mapping(mapping_path)
    classes_in_rules = {str(r["class"]) for r in rules if isinstance(r, dict) and "class" in r}
    missing = [c for c in MODEL_CLASSES if c not in classes_in_rules]
    if missing:
        raise ConfigurationError(
            f"GAINS mapping sidecar must reference every MODEL_CLASSES entry. Missing classes: {missing}"
        )

    side = load_sidecar_dict(sidecar_path) if sidecar_path.is_file() else {}
    ctm = side.get("class_to_metric") or {}
    for cls in MODEL_CLASSES:
        if cls.startswith("C_"):
            continue
        if cls not in ctm:
            raise ConfigurationError(
                f"eurostat end-use sidecar class_to_metric must cover residential class {cls!r} "
                "(or disable eurostat.enabled)."
            )

    morph = cfg.get("morphology") or {}
    for block_key in ("residential_fireplace_heating_stove", "commercial_boilers"):
        block = morph.get(block_key) or {}
        wsum = 0.0
        for wk in ("w111", "w112", "w121", "w_other"):
            if wk in block:
                v = float(block[wk])
                if (not math.isfinite(v)) or v < 0.0:
                    raise ConfigurationError(
                        f"morphology.{block_key}.{wk} must be finite and >= 0, got {v!r}"
                    )
                wsum += v
        LOG.info("[other_combustion] morphology %s coefficient sum (informative)=%.4g", block_key, wsum)

    lut_path = morph.get("pixel_value_map")
    if lut_path:
        p = Path(lut_path)
        if not p.is_absolute():
            p = resolve_path(repo_root, p)
    else:
        p = default_corine_index_map_path()
    l3 = load_ordered_l3_codes(p)
    l3set = set(int(x) for x in l3.tolist())
    for key in ("urban_111", "urban_112", "urban_121"):
        code = int(morph.get(key, 0))
        if code not in l3set:
            LOG.warning(
                "[other_combustion] morphology %s=%s not found in CORINE L3 LUT %s — verify encoding",
                key,
                code,
                p,
            )

    euro = cfg.get("eurostat") or {}
    if bool(euro.get("enabled", False)):
        iso3 = str(cfg["country"]["cams_iso3"]).strip().upper()
        geo = eurostat_geo_for_iso3(iso3, side.get("iso3_to_geo_labels") or {})
        if not geo:
            raise ConfigurationError(
                f"No 2-letter Eurostat geo label for cams_iso3={iso3} in eurostat end-use sidecar"
            )
        year = int(side.get("year", euro.get("year", 2021)))
        offline = bool((euro.get("api") or {}).get("offline", False))
        c1 = _cache_path(repo_root, "nrg_bal_s", geo, year)
        c2 = _cache_path(repo_root, "nrg_d_hhq", geo, year)
        if offline and not (c1.is_file() and c2.is_file()):
            raise ConfigurationError(
                f"eurostat.api.offline=true but cache files missing: {c1.name} / {c2.name} under PROXY/cache/eurostat/"
            )
        if not offline and not (c1.is_file() or c2.is_file()):
            LOG.warning(
                "[other_combustion] No Eurostat cache yet for %s %s — pipeline will try API (configure offline+prefetch if needed)",
                geo,
                year,
            )

    emep = load_emep(emep_path)
    paths = cfg.get("paths") or {}
    hm = paths.get("hotmaps") or {}
    ap_run = cfg.get("appliance_proxy") or {}
    if bool(ap_run.get("enabled", True)):
        hdd = hm.get("hdd_curr")
        if not hdd:
            raise ConfigurationError(
                "appliance_proxy.enabled requires paths.hotmaps.hdd_curr (merged from sector hotmaps.hdd_curr)."
            )
        p_hdd = resolve_path(repo_root, Path(hdd))
        if not p_hdd.is_file():
            raise ConfigurationError(f"appliance_proxy: HDD raster not found: {p_hdd}")

        pop_rel = paths.get("population_tif")
        ghs_rel = paths.get("ghsl_smod_tif")
        if not pop_rel:
            raise ConfigurationError("appliance_proxy.enabled requires paths.population_tif (paths.yaml proxy_common)")
        if not ghs_rel:
            raise ConfigurationError(
                "appliance_proxy.enabled requires paths.ghsl_smod_tif (paths.yaml proxy_specific.waste or equivalent)"
            )
        p_pop = resolve_path(repo_root, Path(pop_rel))
        p_ghs = resolve_path(repo_root, Path(ghs_rel))
        if not p_pop.is_file():
            raise ConfigurationError(f"appliance_proxy: population raster not found: {p_pop}")
        if not p_ghs.is_file():
            raise ConfigurationError(f"appliance_proxy: GHSL SMOD raster not found: {p_ghs}")

        ry = paths.get("ceip_rules_yaml")
        if not ry:
            raise ConfigurationError("appliance_proxy: ceip_rules_yaml missing from merged paths")
        rules_path = resolve_path(repo_root, Path(ry))
        if not rules_path.is_file():
            raise ConfigurationError(f"appliance_proxy: CEIP rules YAML not found: {rules_path}")
        load_appliance_proxy_from_rules_yaml(rules_path)

    probes = (
        ("Natural gas", "Small (single household scale, capacity <=50 kWth) boilers"),
        ("Solid fuel (not biomass)", "Fireplaces, saunas and outdoor heaters"),
    )
    for pol in pollutant_outputs:
        if str(pol).lower() in {"co2_total", "co2"}:
            continue
        best = 0.0
        for fuel, app in probes:
            best = max(best, ef_kg_per_tj(pol, fuel, app, emep, emep_fuel_hints=None))
        if best <= 0.0:
            raise ConfigurationError(
                f"Pollutant {pol!r}: no positive EMEP EF on standard probe rows. "
                "Check EMEP_emission_factors sidecar tables."
            )
