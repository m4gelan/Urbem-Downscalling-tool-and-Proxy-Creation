from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from proxy.core import log
from proxy.dataset_loaders.load_eurostat_c_othercombustion import (
    SourceKind,
    _cache_read_path,
    _cache_write_path,
    _fname,
    _observation,
    _parse_dataset_values,
    fetch_json_dataset,
)

ROAD_FUELS = ("gasoline", "diesel", "lpg", "cng")
BASE = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"


@dataclass(frozen=True)
class RoadFuelSplitResult:
    year: int
    geo: str | None
    iso3: str
    split_by_class: dict[str, dict[str, float]]
    counts_by_class: dict[str, dict[str, float]]
    fallback_by_class: dict[str, str]
    api_sources: dict[str, str]


def _load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open(encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    if not isinstance(doc, dict):
        raise ValueError(f"{config_path}: must be a YAML mapping")
    return doc


def _coord_defaults(dataset: str) -> dict[str, str]:
    return {"freq": "A", "unit": "NR"}


def _mot_nrg_codes(root: dict[str, Any]) -> list[str]:
    if not root:
        return []
    try:
        id_order, _, index_maps, _ = _parse_dataset_values(root)
    except (KeyError, TypeError, ValueError):
        return []
    if "mot_nrg" not in id_order:
        return []
    return list((index_maps.get("mot_nrg") or {}).keys())


def _read_count(
    root: dict[str, Any],
    *,
    geo: str,
    year: int,
    coord: dict[str, str],
    dataset: str,
) -> float | None:
    if not root:
        return None
    try:
        id_order, sizes, index_maps, values = _parse_dataset_values(root)
    except (KeyError, TypeError, ValueError):
        return None
    merged = dict(_coord_defaults(dataset))
    merged.update(coord)
    merged["geo"] = geo
    merged["time"] = str(year)
    v = _observation(id_order, sizes, index_maps, values, merged)
    if v is None or v < 0:
        return None
    return float(v)


def _counts_from_mot_nrg(
    root: dict[str, Any],
    *,
    geo: str,
    year: int,
    fixed_coord: dict[str, str],
    mot_nrg_to_fuel: dict[str, str],
    dataset: str,
) -> dict[str, float]:
    counts: dict[str, float] = {}
    for code in _mot_nrg_codes(root):
        if code == "TOTAL":
            continue
        n = _read_count(
            root,
            geo=geo,
            year=year,
            coord={**fixed_coord, "mot_nrg": code},
            dataset=dataset,
        )
        if n is None:
            continue
        fuel = mot_nrg_to_fuel.get(code)
        if fuel is None:
            continue
        counts[fuel] = counts.get(fuel, 0.0) + n
    return counts


def _counts_eu_aggregate(
    root: dict[str, Any],
    *,
    year: int,
    fixed_coord: dict[str, str],
    mot_nrg_to_fuel: dict[str, str],
    dataset: str,
) -> dict[str, float]:
    if not root:
        return {}
    try:
        id_order, sizes, index_maps, values = _parse_dataset_values(root)
    except (KeyError, TypeError, ValueError):
        return {}
    geos = list((index_maps.get("geo") or {}).keys())
    totals: dict[str, float] = {}
    for geo in geos:
        row = _counts_from_mot_nrg(
            root,
            geo=geo,
            year=year,
            fixed_coord=fixed_coord,
            mot_nrg_to_fuel=mot_nrg_to_fuel,
            dataset=dataset,
        )
        for fuel, n in row.items():
            totals[fuel] = totals.get(fuel, 0.0) + n
    return totals


def _split_from_counts(counts: dict[str, float]) -> dict[str, float]:
    total = sum(counts.values())
    if total <= 0:
        return {f: 0.0 for f in ROAD_FUELS}
    return {f: counts.get(f, 0.0) / total for f in ROAD_FUELS}


def _http_get_json(url: str, *, timeout: float, retries: int, retry_sleep_s: float) -> dict[str, Any]:
    last_exc: BaseException | None = None
    attempts = max(1, int(retries) + 1)
    for attempt in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
                raw = resp.read()
            return json.loads(raw.decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            last_exc = exc
            if attempt + 1 < attempts:
                log.warning(f"Eurostat GET retry {attempt + 1}/{attempts}: {exc}")
                time.sleep(retry_sleep_s)
    assert last_exc is not None
    raise last_exc


def _fetch_root(
    dataset: str,
    *,
    repo_root: Path,
    year: int,
    filters: dict[str, str | list[str]],
    api: dict[str, Any],
    cache_key: str,
    geo: str | None = None,
) -> tuple[dict[str, Any], SourceKind]:
    geo_tag = geo if geo else "_EUagg"
    fname = _fname(dataset, geo_tag, year, cache_key)
    cache_dirs = [str(x) for x in api["cache_dirs"]]
    write_cache_dir = str(api["write_cache_dir"])
    hit = _cache_read_path(repo_root, cache_dirs, fname)
    if hit is not None:
        log.debug(f"Eurostat cache hit {hit.name}")
        return json.loads(hit.read_text(encoding="utf-8")), "cache"

    if bool(api.get("offline", False)):
        log.warning(f"Eurostat offline, no cache for {dataset} {geo_tag} {year}")
        return {}, "fallback"

    q: list[tuple[str, str]] = [("format", "JSON"), ("lang", "EN"), ("time", str(year))]
    if geo:
        q.append(("geo", geo))
    for dim, val in filters.items():
        if isinstance(val, list):
            for v in val:
                q.append((dim, str(v)))
        else:
            q.append((dim, str(val)))
    url = f"{BASE}/{dataset}?{urllib.parse.urlencode(q)}"
    try:
        root = _http_get_json(
            url,
            timeout=float(api["timeout_s"]),
            retries=int(api.get("retries", 2)),
            retry_sleep_s=float(api.get("retry_sleep_s", 4.0)),
        )
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        log.warning(f"Eurostat API failed {dataset}: {exc}")
        return {}, "fallback"

    out_path = _cache_write_path(repo_root, write_cache_dir, fname)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(root, indent=2), encoding="utf-8")
    log.debug(f"Eurostat cached {out_path.name}")
    return root, "api"


def _fetch_country(
    dataset: str,
    *,
    repo_root: Path,
    geo: str,
    year: int,
    filters: dict[str, str | list[str]],
    api: dict[str, Any],
    cache_tag: str,
) -> tuple[dict[str, Any], SourceKind]:
    return fetch_json_dataset(
        dataset,
        repo_root=repo_root,
        geo=geo,
        year=year,
        filters=filters,
        cache_dirs=[str(x) for x in api["cache_dirs"]],
        write_cache_dir=str(api["write_cache_dir"]),
        timeout_s=float(api["timeout_s"]),
        offline=bool(api.get("offline", False)),
        retries=int(api.get("retries", 2)),
        retry_sleep_s=float(api.get("retry_sleep_s", 4.0)),
        cache_tag=cache_tag,
    )


def load_road_fuel_split(
    repo_root: Path,
    country_profile: dict[str, str],
    config_path: Path,
    *,
    enabled: bool = True,
) -> RoadFuelSplitResult:
    cfg = _load_config(config_path)
    year = int(cfg["year"])
    api = cfg["api"]
    fallback = str(cfg["fallback"])
    mot_map = {str(k): str(v) for k, v in cfg["mot_nrg_to_fuel"].items()}
    vclasses = cfg["vehicle_classes"]
    iso3 = str(country_profile.get("ISO3", "")).strip().upper()
    geo = str(country_profile.get("other", "")).strip().upper()
    empty = {f: 0.0 for f in ROAD_FUELS}

    if not enabled:
        log.info("F_Roads Eurostat disabled — uniform gasoline split for all classes")
        uniform = dict(empty)
        uniform["gasoline"] = 1.0
        return RoadFuelSplitResult(
            year=year,
            geo=None,
            iso3=iso3,
            split_by_class={str(k): dict(uniform) for k in vclasses},
            counts_by_class={},
            fallback_by_class={},
            api_sources={},
        )

    if len(geo) != 2:
        log.warning(f"F_Roads: country_profile['other']={geo!r} not 2-letter Eurostat geo")
        uniform = dict(empty)
        uniform["gasoline"] = 1.0
        return RoadFuelSplitResult(
            year=year,
            geo=None,
            iso3=iso3,
            split_by_class={str(k): dict(uniform) for k in vclasses},
            counts_by_class={},
            fallback_by_class={},
            api_sources={},
        )

    split_by_class: dict[str, dict[str, float]] = {}
    counts_by_class: dict[str, dict[str, float]] = {}
    fallback_by_class: dict[str, str] = {}
    api_sources: dict[str, str] = {}
    eu_cache: dict[tuple[str, str], tuple[dict[str, Any], SourceKind]] = {}

    for vkey, vcfg in vclasses.items():
        if not isinstance(vcfg, dict):
            raise ValueError(f"vehicle_classes.{vkey} must be a mapping")

        if "default_split" in vcfg:
            sp = {str(f): float(vcfg["default_split"][f]) for f in ROAD_FUELS if f in vcfg["default_split"]}
            for f in ROAD_FUELS:
                sp.setdefault(f, 0.0)
            s = sum(sp.values())
            if s > 0:
                sp = {f: sp[f] / s for f in ROAD_FUELS}
            split_by_class[vkey] = sp
            counts_by_class[vkey] = {}
            fallback_by_class[vkey] = "default_split"
            continue

        dataset = str(vcfg["dataset"])
        filters = {str(k): str(v) for k, v in (vcfg.get("filters") or {}).items()}
        root, src = _fetch_country(
            dataset,
            repo_root=repo_root,
            geo=geo,
            year=year,
            filters=filters,
            api=api,
            cache_tag=vkey,
        )
        api_sources[f"{vkey}:{dataset}:{geo}"] = src
        counts = _counts_from_mot_nrg(
            root,
            geo=geo,
            year=year,
            fixed_coord=filters,
            mot_nrg_to_fuel=mot_map,
            dataset=dataset,
        )
        source = "country"

        if sum(counts.values()) <= 0 and fallback == "eu_aggregate":
            cache_key = (dataset, json.dumps(filters, sort_keys=True))
            if cache_key not in eu_cache:
                eu_root, eu_src = _fetch_root(
                    dataset,
                    repo_root=repo_root,
                    year=year,
                    filters=filters,
                    api=api,
                    cache_key=f"{vkey}_euagg",
                    geo=None,
                )
                eu_cache[cache_key] = (eu_root, eu_src)
            else:
                eu_root, eu_src = eu_cache[cache_key]
            api_sources[f"{vkey}:{dataset}:EUagg"] = eu_src
            counts = _counts_eu_aggregate(
                eu_root,
                year=year,
                fixed_coord=filters,
                mot_nrg_to_fuel=mot_map,
                dataset=dataset,
            )
            source = "eu_aggregate"
            if sum(counts.values()) <= 0:
                log.warning(
                    f"F_Roads Eurostat: no fuel breakdown for {vkey} {geo} {year} "
                    f"and EU aggregate empty — gasoline=1"
                )
                split_by_class[vkey] = dict(empty)
                split_by_class[vkey]["gasoline"] = 1.0
                counts_by_class[vkey] = {}
                fallback_by_class[vkey] = "gasoline_default"
                continue
            log.warning(
                f"F_Roads Eurostat: {geo} has no {vkey} fuel breakdown in {dataset} "
                f"— using EU aggregate shares"
            )

        split_by_class[vkey] = _split_from_counts(counts)
        counts_by_class[vkey] = counts
        fallback_by_class[vkey] = source

    return RoadFuelSplitResult(
        year=year,
        geo=geo,
        iso3=iso3,
        split_by_class=split_by_class,
        counts_by_class=counts_by_class,
        fallback_by_class=fallback_by_class,
        api_sources=api_sources,
    )


def log_road_fuel_split(result: RoadFuelSplitResult, country_profile: dict[str, str]) -> None:
    name = country_profile.get("full_name", result.iso3)
    log.info(f"F_Roads Eurostat fuel split — {name} {result.geo} {result.year}")
    for vkey, split in sorted(result.split_by_class.items()):
        src = result.fallback_by_class.get(vkey, "?")
        parts = ", ".join(f"{f}={split.get(f, 0.0):.3f}" for f in ROAD_FUELS)
        log.info(f"  {vkey} ({src}): {parts}")
    for ds, src in sorted(result.api_sources.items()):
        log.debug(f"  source {ds}: {src}")
