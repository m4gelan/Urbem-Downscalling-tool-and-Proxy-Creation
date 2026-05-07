"""
Eurostat-derived **end-use bucket weights** and GAINS-derived **appliance splits** for ``M``.

**What**: ``f_enduse[bucket]`` scales national household/commercial energy budgets;
``f_appliance[class]`` normalises GAINS fuel shares *within* each bucket so
``f_enduse × f_appliance`` replaces the former single ``f_enduse[class]`` factor.

**Why**: separates statistical end-use structure (Eurostat) from inventory appliance
split (GAINS) without double counting when both are active.

**Inputs**: sector merged ``cfg``, ``repo_root``, focus ``iso3``, optional GAINS path
for appliance normalisation. **Outputs**: :class:`EndUseFactors` + provenance dicts for logging.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PROXY.core.dataloaders import resolve_path

from ..constants import END_USE_COMMERCIAL, MODEL_CLASSES
from ..eurostat_api import (
    NRG_D_HHQ_BAL_TO_METRIC,
    eurostat_geo_for_iso3,
    fetch_json_dataset,
    parse_nrg_bal_s_commercial_alpha,
    parse_nrg_d_hhq_metric_tj,
)
from .._log import LOG
from .gains_activity import _norm_pct_cell, load_gains_rows, map_gains_row_to_class
from .sidecar_io import load_sidecar_dict


def _sidecar(repo_root: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    paths_obj = cfg.get("paths") or {}
    rel = paths_obj.get("eurostat_end_use_json")
    if not rel:
        return {}
    p = resolve_path(repo_root, rel)
    if not p.is_file():
        return {}
    return load_sidecar_dict(p)


def activity_share_by_class(factors: "EndUseFactors") -> dict[str, float]:
    """
    Combined per-class multiplier ``f_enduse[bucket(k)] × f_appliance[k]`` (diagnostic).

    .. deprecated::
        Prefer inspecting ``f_enduse_by_bucket`` and ``f_appliance_by_class`` separately.
    """
    warnings.warn(
        "activity_share_by_class() is a diagnostic aggregate; use EndUseFactors row_multiplier()",
        DeprecationWarning,
        stacklevel=2,
    )
    return {k: factors.row_multiplier(k) for k in MODEL_CLASSES}


def _bucket_for_class(cls: str, class_to_metric: dict[str, Any]) -> str:
    if cls.startswith("C_"):
        return END_USE_COMMERCIAL
    raw = class_to_metric.get(cls, "default")
    if raw == "default" or raw is None:
        return "__residential_default__"
    if isinstance(raw, list):
        keys = [str(x) for x in raw]
        return "|".join(sorted(keys))
    return str(raw)


def _aggregate_gains_by_class(
    gains_path: Path | None,
    year_col: str,
    rules: list[dict[str, Any]],
    bucket_for_class: dict[str, str],
) -> dict[str, float]:
    agg: dict[str, float] = {k: 0.0 for k in MODEL_CLASSES}
    if gains_path is None or not gains_path.is_file():
        return agg
    for fuel, app, ycell in load_gains_rows(gains_path, year_col):
        cls = map_gains_row_to_class(fuel, app, rules)
        if cls is None:
            continue
        sh = _norm_pct_cell(ycell)
        if sh <= 0:
            continue
        agg[cls] += sh
    return agg


def _f_appliance_from_gains(
    bucket_for_class: dict[str, str],
    agg: dict[str, float],
) -> dict[str, float]:
    by_bucket: dict[str, float] = {}
    for cls, v in agg.items():
        b = bucket_for_class[cls]
        by_bucket[b] = by_bucket.get(b, 0.0) + v
    out: dict[str, float] = {}
    for cls in MODEL_CLASSES:
        b = bucket_for_class[cls]
        den = by_bucket.get(b, 0.0)
        if den <= 0.0:
            out[cls] = 1.0
        else:
            out[cls] = float(agg.get(cls, 0.0) / den)
    return out


@dataclass
class EndUseFactors:
    """End-use bucket scalars × per-class GAINS appliance fractions."""

    f_enduse_by_bucket: dict[str, float] = field(default_factory=dict)
    f_appliance_by_class: dict[str, float] = field(default_factory=dict)
    bucket_for_class: dict[str, str] = field(default_factory=dict)
    legacy_class_scalars: dict[str, float] | None = None
    provenance_enduse: dict[str, str] = field(default_factory=dict)

    def row_multiplier(self, cls: str) -> float:
        if self.legacy_class_scalars is not None:
            return float(self.legacy_class_scalars.get(cls, 1.0))
        b = self.bucket_for_class.get(cls, cls)
        fe = float(self.f_enduse_by_bucket.get(b, 1.0))
        fa = float(self.f_appliance_by_class.get(cls, 1.0))
        return fe * fa

    @staticmethod
    def disabled_uniform() -> "EndUseFactors":
        """Eurostat off: per-class legacy scalars all 1.0 (matches old ``f_enduse``)."""
        return EndUseFactors(
            legacy_class_scalars={k: 1.0 for k in MODEL_CLASSES},
            bucket_for_class={k: f"__class_{k}" for k in MODEL_CLASSES},
        )

    @staticmethod
    def from_legacy_class_scalars(f_enduse: dict[str, float]) -> "EndUseFactors":
        """Backward-compat: one scalar per class (product fe×fa collapsed)."""
        return EndUseFactors(legacy_class_scalars=dict(f_enduse))


def compute_end_use_factors(
    *,
    repo_root: Path,
    cfg: dict[str, Any],
    iso3: str,
    gains_path: Path | None,
    year_col: str,
    rules: list[dict[str, Any]],
) -> EndUseFactors:
    """Build :class:`EndUseFactors` from config + optional Eurostat API + GAINS rows."""
    euro = cfg.get("eurostat") or {}
    side = _sidecar(repo_root, cfg)
    class_to_metric = side.get("class_to_metric") or {}
    def_res = float(side.get("default_residential_if_missing", 1.0))
    year = int(side.get("year", euro.get("year", 2021)))
    iso3u = iso3.strip().upper()
    bucket_for_class = {cls: _bucket_for_class(cls, class_to_metric) for cls in MODEL_CLASSES}

    if not bool(euro.get("enabled", False)):
        LOG.info("Eurostat disabled — using uniform end-use + appliance factors (1.0)")
        return EndUseFactors.disabled_uniform()

    geo = eurostat_geo_for_iso3(iso3u, side.get("iso3_to_geo_labels") or {})
    if not geo:
        LOG.warning("No 2-letter Eurostat geo label for iso3=%s — fallback f_enduse=1.0", iso3u)
        return EndUseFactors.disabled_uniform()

    api_cfg = euro.get("api") or {}
    timeout_s = float(api_cfg.get("timeout_s", 30.0))
    offline = bool(api_cfg.get("offline", False))
    retries = int(api_cfg.get("retries", 2))
    retry_sleep_s = float(api_cfg.get("retry_sleep_s", 4.0))

    prov: dict[str, str] = {}

    root_nrg_s, src_s = fetch_json_dataset(
        "nrg_bal_s",
        repo_root=repo_root,
        geo=geo,
        year=year,
        filters={"nrg_bal": ["FC_OTH_CP_E", "FC_OTH_HH_E"], "unit": "TJ"},
        timeout_s=timeout_s,
        offline=offline,
        retries=retries,
        retry_sleep_s=retry_sleep_s,
    )
    alpha, src_alpha = parse_nrg_bal_s_commercial_alpha(root_nrg_s, geo=geo, year=year)
    if alpha is None:
        alpha = 0.0
        src_alpha = "fallback"
        LOG.warning("Eurostat nrg_bal_s commercial alpha unavailable for %s %s — using 0.0", geo, year)
    prov["commercial_alpha"] = src_alpha

    tj_by_metric: dict[str, float] = {}
    for nrg_bal, mkey in NRG_D_HHQ_BAL_TO_METRIC:
        root_hhq, src_h = fetch_json_dataset(
            "nrg_d_hhq",
            repo_root=repo_root,
            geo=geo,
            year=year,
            filters={"siec": "TOTAL", "unit": "TJ", "nrg_bal": nrg_bal},
            cache_tag=nrg_bal,
            timeout_s=timeout_s,
            offline=offline,
            retries=retries,
            retry_sleep_s=retry_sleep_s,
        )
        slice_m, _ = parse_nrg_d_hhq_metric_tj(root_hhq, geo=geo, year=year)
        if mkey in slice_m:
            tj_by_metric[mkey] = slice_m[mkey]
            prov[f"hhq_{mkey}"] = src_h

    f_res_total = max(0.0, min(1.0, 1.0 - float(alpha)))
    total_tj = sum(max(0.0, v) for v in tj_by_metric.values())
    f_enduse: dict[str, float] = {END_USE_COMMERCIAL: float(alpha)}
    if total_tj <= 0.0:
        LOG.warning("Eurostat nrg_d_hhq: no positive TJ for geo=%s year=%s — residential buckets use default", geo, year)
        n_m = max(len(NRG_D_HHQ_BAL_TO_METRIC), 1)
        flat = def_res * f_res_total / n_m
        for _bal, mkey in NRG_D_HHQ_BAL_TO_METRIC:
            f_enduse.setdefault(mkey, flat)
    else:
        for mkey, tj in tj_by_metric.items():
            f_enduse[mkey] = f_res_total * (float(tj) / total_tj)
    for cls in MODEL_CLASSES:
        if cls.startswith("C_"):
            continue
        b = bucket_for_class[cls]
        if b == "__residential_default__":
            f_enduse[b] = def_res * f_res_total
        elif "|" in b:
            parts = b.split("|")
            f_enduse[b] = float(sum(f_enduse.get(p, def_res * f_res_total) for p in parts) / max(len(parts), 1))

    agg = _aggregate_gains_by_class(gains_path, year_col, rules, bucket_for_class)
    f_appliance = _f_appliance_from_gains(bucket_for_class, agg)

    return EndUseFactors(
        f_enduse_by_bucket=f_enduse,
        f_appliance_by_class=f_appliance,
        bucket_for_class=bucket_for_class,
        provenance_enduse=prov,
    )


def log_enduse_tables(iso3: str, factors: EndUseFactors) -> None:
    """INFO: ``f_enduse`` buckets and ``f_appliance`` by class (one log record per line for readable consoles)."""
    if factors.legacy_class_scalars is not None:
        LOG.info(
            "[other_combustion] %s legacy per-class scalars (Eurostat off or compat):\n%s",
            iso3,
            "\n".join(f"  {k}: {factors.legacy_class_scalars.get(k, 1.0):.8g}" for k in MODEL_CLASSES),
        )
        return
    LOG.info(
        "[other_combustion] %s Eurostat f_enduse: commercial vs household + residential HH end-use buckets",
        iso3,
    )
    for k in sorted(factors.f_enduse_by_bucket.keys()):
        src = factors.provenance_enduse.get(k, factors.provenance_enduse.get(f"hhq_{k}", "—"))
        scope = "commercial CP share" if k == END_USE_COMMERCIAL else "residential HH Eurostat"
        LOG.info(
            "[other_combustion] %s   %-42s %12.8g  source=%s  (%s)",
            iso3,
            k,
            factors.f_enduse_by_bucket[k],
            src,
            scope,
        )
    LOG.info("[other_combustion] %s GAINS f_appliance (within-bucket); residential R_* then commercial C_*", iso3)
    for cls in MODEL_CLASSES:
        scope = "residential" if cls.startswith("R_") else "commercial"
        LOG.info(
            "[other_combustion] %s   %-20s %12.8g  (%s)",
            iso3,
            cls,
            factors.f_appliance_by_class.get(cls, 1.0),
            scope,
        )
