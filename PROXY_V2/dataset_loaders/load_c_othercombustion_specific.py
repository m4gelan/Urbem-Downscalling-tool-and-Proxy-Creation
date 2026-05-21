from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from PROXY_V2.core import log
from PROXY_V2.dataset_loaders.load_eurostat_c_othercombustion import (
    fetch_json_dataset,
    parse_nrg_bal_s_commercial_alpha,
    parse_nrg_bal_s_tj,
    parse_nrg_d_hhq_metric_tj,
)


@dataclass(frozen=True)
class GainsMappedRow:
    fuel: str
    appliance: str
    share: float
    class_name: str
    bucket: str
    s_r: float = 0.0


@dataclass(frozen=True)
class GainsActivityResult:
    year: int
    country_tag: str
    rows: tuple[GainsMappedRow, ...]
    f_gains_by_class: dict[str, float]
    w_sum_by_class: dict[str, float]
    w_sum_by_bucket: dict[str, float]
    unmapped_rows: tuple[tuple[str, str, float], ...]


def _norm_pct_cell(raw: str) -> float:
    s = str(raw).strip().strip('"').replace(",", ".")
    if not s or s.lower() in {"n.a", "na", "..", "-", ""}:
        return 0.0
    try:
        return float(s) / 100.0
    except ValueError:
        return 0.0


def _load_gains_mapping(mapping_path: Path) -> tuple[dict[str, str], list[dict[str, Any]]]:
    with mapping_path.open(encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    if not isinstance(doc, dict):
        raise ValueError(f"{mapping_path}: mapping must be a YAML mapping")
    buckets = doc.get("class_buckets")
    if not isinstance(buckets, dict) or not buckets:
        raise ValueError(f"{mapping_path}: class_buckets missing or empty")
    class_buckets = {str(k).strip(): str(v).strip() for k, v in buckets.items()}
    rules = doc.get("rules")
    if not isinstance(rules, list) or not rules:
        raise ValueError(f"{mapping_path}: rules must be a non-empty list")
    return class_buckets, rules


def _match_rule(fuel: str, app: str, rule: dict[str, Any]) -> bool:
    fl = fuel.lower()
    al = app.lower()
    fc = rule.get("fuel_contains")
    if fc and str(fc).lower() not in fl:
        return False
    fca = rule.get("fuel_contains_all")
    if fca:
        for frag in fca:
            if str(frag).lower() not in fl:
                return False
    fany = rule.get("fuel_contains_any")
    if fany and not any(str(frag).lower() in fl for frag in fany):
        return False
    ac = rule.get("appliance_contains")
    if ac and str(ac).lower() not in al:
        return False
    aca = rule.get("appliance_contains_all")
    if aca:
        for frag in aca:
            if str(frag).lower() not in al:
                return False
    aany = rule.get("appliance_contains_any")
    if aany and not any(str(frag).lower() in al for frag in aany):
        return False
    anc = rule.get("appliance_not_contains")
    if anc:
        for frag in anc:
            if str(frag).lower() in al:
                return False
    return True


def map_gains_row_to_class(
    fuel: str,
    appliance: str,
    rules: list[dict[str, Any]],
    class_buckets: dict[str, str],
) -> tuple[str, str] | None:
    for rule in rules:
        if not isinstance(rule, dict) or "class" not in rule:
            continue
        if not _match_rule(fuel, appliance, rule):
            continue
        cls = str(rule["class"]).strip()
        bucket = class_buckets.get(cls)
        if bucket is None:
            raise KeyError(f"class {cls!r} from mapping rule has no class_buckets entry")
        return cls, bucket
    return None


def _load_gains_table_rows(path: Path, year: int) -> list[tuple[str, str, float]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if '"[% of fuel input]"' in line or "[% of fuel input]" in line:
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"No GAINS header row in {path}")
    reader = csv.reader(lines[header_idx:])
    header = next(reader)
    clean = [c.strip().strip('"') for c in header]
    year_s = str(year)
    try:
        yi = clean.index(year_s)
    except ValueError as exc:
        raise ValueError(f"Year {year_s!r} not in GAINS header of {path}") from exc
    out: list[tuple[str, str, float]] = []
    for row in reader:
        if len(row) < 2:
            continue
        fuel = row[0].strip().strip('"')
        app = row[1].strip().strip('"') if len(row) > 1 else ""
        if not fuel:
            continue
        if len(row) <= yi:
            continue
        share = _norm_pct_cell(row[yi])
        if share <= 0.0:
            continue
        out.append((fuel, app, share))
    return out


def gains_dom_share_path(gains_folder: Path, country_profile: dict[str, str]) -> Path:
    tag = country_profile["full_name"]
    path = gains_folder / f"dom_share_ENE_{tag}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"GAINS domestic share file not found: {path}")
    return path


def compute_gains_activity(
    gains_filepath: Path,
    year_gains: int,
    gains_mapping_filepath: Path,
    *,
    country_profile: dict[str, str] | None = None,
) -> GainsActivityResult:
    """
    Map GAINS (fuel, appliance) rows to classes, then compute raw weights w_r, within-class s_r,
    and f_GAINS(k) = sum_{r in k} w_r / sum_{r in bucket(k)} w_r.
    """
    class_buckets, rules = _load_gains_mapping(gains_mapping_filepath)
    raw_rows = _load_gains_table_rows(gains_filepath, int(year_gains))

    mapped: list[tuple[str, str, float, str, str]] = []
    unmapped: list[tuple[str, str, float]] = []
    for fuel, app, share in raw_rows:
        hit = map_gains_row_to_class(fuel, app, rules, class_buckets)
        if hit is None:
            unmapped.append((fuel, app, share))
            continue
        cls, bucket = hit
        mapped.append((fuel, app, share, cls, bucket))

    w_by_class: dict[str, float] = {c: 0.0 for c in class_buckets}
    w_by_bucket: dict[str, float] = {}
    for _, _, share, cls, bucket in mapped:
        w_by_class[cls] += share
        w_by_bucket[bucket] = w_by_bucket.get(bucket, 0.0) + share

    f_gains: dict[str, float] = {}
    for cls in class_buckets:
        bucket = class_buckets[cls]
        den = w_by_bucket.get(bucket, 0.0)
        f_gains[cls] = float(w_by_class[cls] / den) if den > 0.0 else 0.0

    out_rows: list[GainsMappedRow] = []
    for fuel, app, share, cls, bucket in mapped:
        den_c = w_by_class[cls]
        s_r = float(share / den_c) if den_c > 0.0 else 0.0
        out_rows.append(
            GainsMappedRow(
                fuel=fuel,
                appliance=app,
                share=share,
                class_name=cls,
                bucket=bucket,
                s_r=s_r,
            )
        )

    country_tag = ""
    if country_profile:
        country_tag = str(country_profile.get("full_name", ""))

    return GainsActivityResult(
        year=int(year_gains),
        country_tag=country_tag,
        rows=tuple(out_rows),
        f_gains_by_class=f_gains,
        w_sum_by_class=dict(w_by_class),
        w_sum_by_bucket=dict(w_by_bucket),
        unmapped_rows=tuple(unmapped),
    )


def log_gains_activity_debug(result: GainsActivityResult) -> None:
    log.info(
        f"C_OtherCombustion GAINS activity year={result.year} country={result.country_tag!r} "
        f"mapped_rows={len(result.rows)} unmapped_rows={len(result.unmapped_rows)}"
    )

    bucket_for_class: dict[str, str] = {}
    for r in result.rows:
        bucket_for_class[r.class_name] = r.bucket

    log.info("--- f_GAINS by class (GAINS share within bucket) ---")
    for cls in sorted(result.f_gains_by_class.keys()):
        b = bucket_for_class.get(cls, "?")
        log.info(
            f"  {cls} bucket={b} w_class={result.w_sum_by_class.get(cls, 0.0):.4f} "
            f"f_GAINS={result.f_gains_by_class[cls]:.6g}"
        )

    log.debug("--- bucket totals (denominator for f_GAINS) ---")
    for b in sorted(result.w_sum_by_bucket.keys()):
        log.debug(f"  bucket={b} w_sum={result.w_sum_by_bucket[b]:.6g}")

    by_class: dict[str, list[GainsMappedRow]] = {}
    for r in result.rows:
        by_class.setdefault(r.class_name, []).append(r)

    log.debug("--- s_r by class (fuel | appliance | w_r | s_r) ---")
    for cls in sorted(by_class.keys()):
        log.debug(f"  [{cls}] bucket={by_class[cls][0].bucket}")
        for r in sorted(by_class[cls], key=lambda x: (-x.s_r, x.fuel, x.appliance)):
            log.debug(
                f"    s_r={r.s_r:.6g} w_r={r.share:.6g} | {r.fuel} | {r.appliance}"
            )

    if result.unmapped_rows:
        log.debug("--- unmapped GAINS rows (first 20) ---")
        for fuel, app, share in result.unmapped_rows[:20]:
            log.debug(f"    w_r={share:.6g} | {fuel} | {app}")
        if len(result.unmapped_rows) > 20:
            log.debug(f"    ... and {len(result.unmapped_rows) - 20} more")


@dataclass(frozen=True)
class EurostatEndUseResult:
    year: int
    geo: str | None
    iso3: str
    f_enduse_by_bucket: dict[str, float]
    f_enduse_by_class: dict[str, float]
    bucket_for_class: dict[str, str]
    tj_by_metric: dict[str, float]
    commercial_tj: dict[str, float]
    f_by_metric: dict[str, float]
    residential_share: float
    commercial_alpha: float | None
    api_sources: dict[str, str]


def _load_eurostat_end_use_config(config_path: Path) -> dict[str, Any]:
    with config_path.open(encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    if not isinstance(doc, dict):
        raise ValueError(f"{config_path}: must be a YAML mapping")
    return doc


def _empty_eurostat_result(
    *,
    year: int,
    iso3: str,
    geo: str | None,
    uniform: dict[str, float],
    bucket_for_class: dict[str, str],
) -> EurostatEndUseResult:
    return EurostatEndUseResult(
        year=year,
        geo=geo,
        iso3=iso3,
        f_enduse_by_bucket={},
        f_enduse_by_class=uniform,
        bucket_for_class=bucket_for_class,
        tj_by_metric={},
        commercial_tj={},
        f_by_metric={},
        residential_share=0.0,
        commercial_alpha=None,
        api_sources={},
    )


def compute_eurostat_f_enduse(
    repo_root: Path,
    country_profile: dict[str, str],
    eurostat_config_path: Path,
    *,
    enabled: bool = True,
) -> EurostatEndUseResult:
    cfg = _load_eurostat_end_use_config(eurostat_config_path)
    year = int(cfg["year"])
    class_to_bucket = {str(k): str(v) for k, v in cfg["class_to_bucket"].items()}
    bucket_for_class = dict(class_to_bucket)
    commercial_bucket = str(cfg["commercial_bucket"])
    def_res = float(cfg["default_residential_if_missing"])
    api = cfg.get("api") or {}
    timeout_s = float(api["timeout_s"])
    offline = bool(api.get("offline", False))
    retries = int(api.get("retries", 2))
    retry_sleep_s = float(api.get("retry_sleep_s", 4.0))
    cache_dirs = [str(x) for x in api.get("cache_dirs", ["PROXY_V2/cache/eurostat"])]
    write_cache_dir = str(api.get("write_cache_dir", "PROXY_V2/cache/eurostat"))

    iso3 = str(country_profile.get("ISO3", "")).strip().upper()
    geo = str(country_profile.get("other", "")).strip().upper()
    uniform = {cls: 1.0 for cls in class_to_bucket}
    if not enabled:
        log.info("C_OtherCombustion Eurostat disabled — f_enduse=1.0 for all classes")
        return _empty_eurostat_result(
            year=year, iso3=iso3, geo=None, uniform=uniform, bucket_for_class=bucket_for_class
        )

    if len(geo) != 2:
        log.warning(
            f"C_OtherCombustion: country_profile['other']={geo!r} is not a 2-letter Eurostat geo — f_enduse=1.0"
        )
        return _empty_eurostat_result(
            year=year, iso3=iso3, geo=None, uniform=uniform, bucket_for_class=bucket_for_class
        )

    api_sources: dict[str, str] = {}
    bal_to_metric = {str(k): str(v) for k, v in cfg["nrg_d_hhq_balances"].items()}
    root_bal, src_bal = fetch_json_dataset(
        "nrg_bal_s",
        repo_root=repo_root,
        geo=geo,
        year=year,
        filters={"nrg_bal": ["FC_OTH_CP_E", "FC_OTH_HH_E"], "unit": "TJ"},
        cache_dirs=cache_dirs,
        write_cache_dir=write_cache_dir,
        timeout_s=timeout_s,
        offline=offline,
        retries=retries,
        retry_sleep_s=retry_sleep_s,
    )
    api_sources["nrg_bal_s"] = src_bal
    commercial_tj, _ = parse_nrg_bal_s_tj(root_bal, geo=geo, year=year)
    alpha, _ = parse_nrg_bal_s_commercial_alpha(root_bal, geo=geo, year=year)
    if alpha is None:
        alpha = 0.0
        log.warning(f"C_OtherCombustion: nrg_bal_s missing for {geo} {year} — commercial_alpha=0")

    tj_by_metric: dict[str, float] = {}
    for nrg_bal, mkey in bal_to_metric.items():
        root_hhq, src_h = fetch_json_dataset(
            "nrg_d_hhq",
            repo_root=repo_root,
            geo=geo,
            year=year,
            filters={"siec": "TOTAL", "unit": "TJ", "nrg_bal": nrg_bal},
            cache_dirs=cache_dirs,
            write_cache_dir=write_cache_dir,
            timeout_s=timeout_s,
            offline=offline,
            cache_tag=nrg_bal,
            retries=retries,
            retry_sleep_s=retry_sleep_s,
        )
        api_sources[f"nrg_d_hhq/{nrg_bal}"] = src_h
        slice_m, _ = parse_nrg_d_hhq_metric_tj(
            root_hhq, geo=geo, year=year, balance_to_metric={nrg_bal: mkey}
        )
        if mkey in slice_m:
            tj_by_metric[mkey] = slice_m[mkey]

    f_res_total = max(0.0, min(1.0, 1.0 - float(alpha)))
    total_tj = sum(max(0.0, v) for v in tj_by_metric.values())
    f_by_metric: dict[str, float] = {}
    if total_tj <= 0.0:
        log.warning(f"C_OtherCombustion: nrg_d_hhq empty for {geo} {year} — flat residential split")
        n_m = max(len(bal_to_metric), 1)
        flat = def_res * f_res_total / n_m
        for mkey in bal_to_metric.values():
            f_by_metric[mkey] = flat
    else:
        for mkey, tj in tj_by_metric.items():
            f_by_metric[mkey] = f_res_total * (float(tj) / total_tj)

    f_enduse_bucket: dict[str, float] = {commercial_bucket: float(alpha)}
    bucket_rules = cfg.get("bucket_from_metrics") or {}
    for bucket, rule in bucket_rules.items():
        b = str(bucket)
        if not isinstance(rule, dict):
            continue
        if rule.get("source") == "nrg_bal_s_alpha":
            f_enduse_bucket[b] = float(alpha)
            continue
        metrics = rule.get("metrics")
        if not isinstance(metrics, list) or not metrics:
            continue
        vals = [f_by_metric.get(str(m), def_res * f_res_total) for m in metrics]
        combine = str(rule.get("combine", "")).lower()
        if combine == "sum":
            f_enduse_bucket[b] = float(sum(vals))
        elif combine == "mean":
            f_enduse_bucket[b] = float(sum(vals) / len(vals))
        else:
            f_enduse_bucket[b] = float(vals[0])

    f_enduse_class = {
        cls: float(f_enduse_bucket.get(bucket_for_class[cls], 1.0))
        for cls in class_to_bucket
    }
    return EurostatEndUseResult(
        year=year,
        geo=geo,
        iso3=iso3,
        f_enduse_by_bucket=f_enduse_bucket,
        f_enduse_by_class=f_enduse_class,
        bucket_for_class=bucket_for_class,
        tj_by_metric=dict(tj_by_metric),
        commercial_tj=dict(commercial_tj),
        f_by_metric=dict(f_by_metric),
        residential_share=float(f_res_total),
        commercial_alpha=float(alpha),
        api_sources=dict(api_sources),
    )


def log_eurostat_end_use_debug(result: EurostatEndUseResult, country_profile: dict[str, str]) -> None:
    name = country_profile.get("full_name", "?")
    log.info(
        f"C_OtherCombustion Eurostat {name} ISO3={result.iso3} geo={result.geo!r} year={result.year}"
    )
    if result.commercial_tj:
        cp = result.commercial_tj.get("FC_OTH_CP_E", 0.0)
        hh = result.commercial_tj.get("FC_OTH_HH_E", 0.0)
        tot = cp + hh
        log.info("--- nrg_bal_s final energy, other sectors (unit: TJ) ---")
        log.info(
            "  FC_OTH_CP_E commercial products & services: "
            f"{cp:.3f} TJ"
        )
        log.info(
            "  FC_OTH_HH_E households: "
            f"{hh:.3f} TJ"
        )
        log.info(f"  total CP+HH: {tot:.3f} TJ")
        log.info(
            "  f_enduse(commercial) = CP/(CP+HH) = "
            f"{result.commercial_alpha:.6g} (dimensionless)"
        )
        log.info(
            "  f_enduse(residential total) = HH/(CP+HH) = "
            f"{result.residential_share:.6g} (dimensionless)"
        )
    if result.tj_by_metric:
        tj_sum = sum(result.tj_by_metric.values())
        log.info("--- nrg_d_hhq household end-use (TJ) ---")
        for m in sorted(result.tj_by_metric.keys()):
            tj = result.tj_by_metric[m]
            pct = 100.0 * tj / tj_sum if tj_sum > 0 else 0.0
            log.info(f"  {m}: TJ={tj:.3f}  ({pct:.1f}% of HH total)")
    if result.f_by_metric:
        log.info("--- f_enduse share within residential (before bucket rules) ---")
        for m in sorted(result.f_by_metric.keys()):
            log.info(f"  {m}: {result.f_by_metric[m]:.6g}")
    log.info("--- f_enduse by bucket (national energy budget share, dimensionless) ---")
    for b in sorted(result.f_enduse_by_bucket.keys()):
        log.info(f"  {b}: {result.f_enduse_by_bucket[b]:.6g}")
    log.info("--- f_enduse by class ---")
    for cls in sorted(result.f_enduse_by_class.keys()):
        b = result.bucket_for_class.get(cls, "?")
        log.info(f"  {cls} ({b}): {result.f_enduse_by_class[cls]:.6g}")

