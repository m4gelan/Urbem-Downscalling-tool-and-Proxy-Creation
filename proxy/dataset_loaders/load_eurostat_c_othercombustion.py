from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Literal

from proxy.core import log

SourceKind = Literal["api", "cache", "fallback"]
BASE = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"


def _cache_write_path(repo_root: Path, cache_dir_rel: str, fname: str) -> Path:
    return (repo_root / cache_dir_rel.replace("\\", "/") / fname).resolve()


def _cache_read_path(repo_root: Path, cache_dirs: list[str], fname: str) -> Path | None:
    for rel in cache_dirs:
        p = _cache_write_path(repo_root, rel, fname)
        if p.is_file():
            return p
    return None


def _fname(dataset: str, geo: str, year: int, tag: str | None) -> str:
    base = f"{dataset}_{geo}_{year}".replace("/", "_")
    if tag:
        base = f"{base}_{tag.replace('/', '_')}"
    return f"{base}.json"


def _product(sizes: list[int]) -> int:
    n = 1
    for s in sizes:
        n *= int(s)
    return max(n, 1)


def _normalize_value_block(val: Any, total_n: int) -> list[float | None]:
    if isinstance(val, list):
        out: list[float | None] = []
        for x in val:
            try:
                out.append(float(x) if x is not None else None)
            except (TypeError, ValueError):
                out.append(None)
        while len(out) < total_n:
            out.append(None)
        return out[:total_n]
    if isinstance(val, dict):
        out = [None] * total_n
        for k, v in val.items():
            try:
                ki = int(str(k))
            except (TypeError, ValueError):
                continue
            if 0 <= ki < total_n and v is not None:
                try:
                    out[ki] = float(v)
                except (TypeError, ValueError):
                    pass
        return out
    raise ValueError("dataset.value must be list or dict")


def _linear_index(id_order: list[str], sizes: list[int], coord: dict[str, int]) -> int:
    idx = 0
    for j, dim in enumerate(id_order):
        idx = idx * int(sizes[j]) + int(coord[dim])
    return idx


def _parse_dataset_values(j: dict[str, Any]) -> tuple[list[str], list[int], dict[str, dict[str, int]], list[float | None]]:
    if isinstance(j.get("dataset"), dict):
        d = j["dataset"]
        val = d.get("value")
        dim = d.get("dimension") or {}
        did = d.get("id")
        size = d.get("size")
    else:
        val = j.get("value")
        dim = j.get("dimension") or {}
        did = j.get("id")
        size = j.get("size")
    if not isinstance(did, list) or not isinstance(size, list) or val is None:
        raise ValueError("Unrecognised Eurostat JSON shape")
    sizes_i = [int(x) for x in size]
    total_n = _product(sizes_i)
    values = _normalize_value_block(val, total_n)
    index_maps: dict[str, dict[str, int]] = {}
    for dname in did:
        block = dim.get(str(dname)) or {}
        cat = block.get("category") or {}
        idx = cat.get("index")
        if isinstance(idx, dict):
            index_maps[str(dname)] = {str(k): int(v) for k, v in idx.items()}
    return [str(x) for x in did], sizes_i, index_maps, values


def _energy_coord_defaults() -> dict[str, str]:
    return {"freq": "A", "siec": "TOTAL"}


def _observation(
    id_order: list[str],
    sizes: list[int],
    index_maps: dict[str, dict[str, int]],
    values: list[float | None],
    coord: dict[str, str],
    *,
    defaults: dict[str, str] | None = None,
) -> float | None:
    merged: dict[str, str] = dict(defaults or {})
    merged.update(coord)
    ci: dict[str, int] = {}
    for dim in id_order:
        v = merged.get(dim)
        if v is None:
            return None
        m = index_maps.get(dim) or {}
        vs = str(v)
        if vs not in m:
            return None
        ci[dim] = m[vs]
    li = _linear_index(id_order, sizes, ci)
    if li < 0 or li >= len(values):
        return None
    return values[li]


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


def fetch_json_dataset(
    dataset: str,
    *,
    repo_root: Path,
    geo: str,
    year: int,
    filters: dict[str, str | list[str]],
    cache_dirs: list[str],
    write_cache_dir: str,
    timeout_s: float,
    offline: bool,
    cache_tag: str | None = None,
    retries: int = 2,
    retry_sleep_s: float = 4.0,
) -> tuple[dict[str, Any], SourceKind]:
    fname = _fname(dataset, geo, year, cache_tag)
    hit = _cache_read_path(repo_root, cache_dirs, fname)
    if hit is not None:
        log.debug(f"Eurostat cache hit {hit.name}")
        return json.loads(hit.read_text(encoding="utf-8")), "cache"

    if offline:
        log.warning(f"Eurostat offline, no cache for {dataset} {geo} {year}")
        return {}, "fallback"

    q: list[tuple[str, str]] = [("format", "JSON"), ("lang", "EN"), ("geo", geo), ("time", str(year))]
    for dim, val in filters.items():
        if isinstance(val, list):
            for v in val:
                q.append((dim, str(v)))
        else:
            q.append((dim, str(val)))
    url = f"{BASE}/{dataset}?{urllib.parse.urlencode(q)}"
    try:
        root = _http_get_json(url, timeout=timeout_s, retries=retries, retry_sleep_s=retry_sleep_s)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        log.warning(f"Eurostat API failed {dataset}: {exc}")
        return {}, "fallback"

    out_path = _cache_write_path(repo_root, write_cache_dir, fname)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(root, indent=2), encoding="utf-8")
    log.debug(f"Eurostat cached {out_path.name}")
    return root, "api"


def parse_nrg_bal_s_tj(
    root: dict[str, Any],
    *,
    geo: str,
    year: int,
) -> tuple[dict[str, float], SourceKind]:
    """FC_OTH_CP_E and FC_OTH_HH_E in TJ from nrg_bal_s."""
    if not root:
        return {}, "fallback"
    try:
        id_order, sizes, index_maps, values = _parse_dataset_values(root)
    except (KeyError, TypeError, ValueError):
        return {}, "fallback"
    unit = "TJ"
    defs = _energy_coord_defaults()
    out: dict[str, float] = {}
    for code in ("FC_OTH_CP_E", "FC_OTH_HH_E"):
        v = _observation(
            id_order, sizes, index_maps, values,
            {"nrg_bal": code, "unit": unit, "geo": geo, "time": str(year)},
            defaults=defs,
        )
        if v is not None and v >= 0:
            out[code] = float(v)
    if not out:
        return {}, "fallback"
    return out, "api"


def parse_nrg_bal_s_commercial_alpha(
    root: dict[str, Any],
    *,
    geo: str,
    year: int,
) -> tuple[float | None, SourceKind]:
    tj, src = parse_nrg_bal_s_tj(root, geo=geo, year=year)
    cp = tj.get("FC_OTH_CP_E")
    hh = tj.get("FC_OTH_HH_E")
    if cp is None or hh is None or cp + hh <= 0:
        return None, "fallback" if not tj else src
    return float(cp / (cp + hh)), src


def parse_nrg_d_hhq_metric_tj(
    root: dict[str, Any],
    *,
    geo: str,
    year: int,
    balance_to_metric: dict[str, str],
) -> tuple[dict[str, float], SourceKind]:
    out: dict[str, float] = {}
    if not root:
        return out, "fallback"
    try:
        id_order, sizes, index_maps, values = _parse_dataset_values(root)
    except (KeyError, TypeError, ValueError):
        return out, "fallback"
    unit = "TJ"
    defs = _energy_coord_defaults()
    for nrg_bal, mkey in balance_to_metric.items():
        v = _observation(
            id_order, sizes, index_maps, values,
            {"nrg_bal": nrg_bal, "unit": unit, "geo": geo, "time": str(year)},
            defaults=defs,
        )
        if v is not None and v >= 0:
            out[mkey] = float(v)
    if not out:
        return out, "fallback"
    return out, "api"
