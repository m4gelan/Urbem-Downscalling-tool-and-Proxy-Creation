"""
Eurostat dissemination API client (JSON-stat style datasets) with on-disk cache.

**Role**: fetch `nrg_d_hhq` household end-use TJ and `nrg_bal_s` commercial vs household
split for GNFR C scaling. **Inputs**: geo (Eurostat 2-letter where applicable), year,
repo root for cache path. **Outputs**: floats + provenance (api | cache | fallback).

No hardcoded repo paths except default cache under ``PROXY/cache/eurostat/``.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Literal

from ._log import LOG

SourceKind = Literal["api", "cache", "fallback"]

BASE = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"

# nrg_bal codes for household disaggregation (Table-style end uses)
NRG_D_HHQ_BAL_TO_METRIC: tuple[tuple[str, str], ...] = (
    ("FC_OTH_HH_E_SH", "space_heating"),
    ("FC_OTH_HH_E_WH", "water_heating"),
    ("FC_OTH_HH_E_CK", "cooking"),
    ("FC_OTH_HH_E_CL", "space_cooling"),
    ("FC_OTH_HH_E_LE", "lighting_appliances"),
    ("FC_OTH_HH_E_OE", "other_end_use"),
)


def _cache_path(repo_root: Path, dataset: str, geo: str, year: int, tag: str | None = None) -> Path:
    base = f"{dataset}_{geo}_{year}".replace("/", "_")
    if tag:
        base = f"{base}_{tag.replace('/', '_')}"
    return Path(repo_root) / "PROXY" / "cache" / "eurostat" / f"{base}.json"


def _product(sizes: list[int]) -> int:
    n = 1
    for s in sizes:
        n *= int(s)
    return max(n, 1)


def _normalize_value_block(val: Any, total_n: int) -> list[float | None]:
    """Eurostat JSON-stat 2.0 uses a sparse ``value`` object; older payloads use a list."""
    if isinstance(val, list):
        out: list[float | None] = []
        for x in val:
            if x is None:
                out.append(None)
            else:
                try:
                    out.append(float(x))
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
    """Unravel indices in ``id`` order (first dimension slowest, Eurostat convention)."""
    idx = 0
    for j, dim in enumerate(id_order):
        idx = idx * int(sizes[j]) + int(coord[dim])
    return idx


def _parse_dataset_values(j: dict[str, Any]) -> tuple[list[str], list[int], dict[str, dict[str, int]], list[float | None]]:
    """Return (id_order, sizes, index_maps, dense values) from a JSON-stat 2.0 dataset object."""
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
        raise ValueError("Unrecognised Eurostat JSON shape (expected id/size/value)")
    sizes_i = [int(x) for x in size]
    total_n = _product(sizes_i)
    values = _normalize_value_block(val, total_n)
    index_maps: dict[str, dict[str, int]] = {}
    for dname in did:
        block = dim.get(str(dname)) or {}
        cat = block.get("category") or {}
        idx = cat.get("index")
        if not isinstance(idx, dict):
            continue
        index_maps[str(dname)] = {str(k): int(v) for k, v in idx.items()}
    return [str(x) for x in did], sizes_i, index_maps, values


def _energy_coord_defaults() -> dict[str, str]:
    """Eurostat energy datasets use annual frequency and SIEC total unless filtered."""
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
                LOG.warning(
                    "Eurostat GET attempt %d/%d failed (%s), retry in %.1fs",
                    attempt + 1,
                    attempts,
                    exc,
                    retry_sleep_s,
                )
                time.sleep(retry_sleep_s)
    assert last_exc is not None
    raise last_exc


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def fetch_json_dataset(
    dataset: str,
    *,
    repo_root: Path,
    geo: str,
    year: int,
    filters: dict[str, str | list[str]],
    timeout_s: float,
    offline: bool,
    cache_tag: str | None = None,
    retries: int = 2,
    retry_sleep_s: float = 4.0,
) -> tuple[dict[str, Any], SourceKind]:
    """
    GET ``{BASE}/{dataset}`` with query filters. Returns (parsed JSON root, source).

    ``filters`` maps dimension name to one value or list (repeated query keys).
    ``cache_tag`` distinguishes cache files for different slices of the same dataset.
    """
    cache = _cache_path(repo_root, dataset, geo, year, tag=cache_tag)
    if cache.is_file():
        LOG.info("Eurostat cache hit dataset=%s geo=%s year=%s path=%s", dataset, geo, year, cache)
        return _load_json(cache), "cache"

    if offline:
        LOG.warning(
            "Eurostat offline mode: no cache for dataset=%s geo=%s year=%s — caller should fallback",
            dataset,
            geo,
            year,
        )
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
        LOG.warning("Eurostat API GET failed dataset=%s: %s", dataset, exc)
        return {}, "fallback"

    _save_json(cache, root)
    LOG.info("Eurostat API stored cache dataset=%s geo=%s year=%s path=%s", dataset, geo, year, cache)
    return root, "api"


def parse_nrg_bal_s_commercial_alpha(
    root: dict[str, Any],
    *,
    geo: str,
    year: int,
) -> tuple[float | None, SourceKind]:
    """Return alpha = FC_OTH_CP_E / (FC_OTH_CP_E + FC_OTH_HH_E) in TJ, or (None, fallback)."""
    if not root:
        return None, "fallback"
    try:
        id_order, sizes, index_maps, values = _parse_dataset_values(root)
    except (KeyError, TypeError, ValueError):
        return None, "fallback"
    unit = "TJ"
    defs = _energy_coord_defaults()
    cp = _observation(
        id_order,
        sizes,
        index_maps,
        values,
        {"nrg_bal": "FC_OTH_CP_E", "unit": unit, "geo": geo, "time": str(year)},
        defaults=defs,
    )
    hh = _observation(
        id_order,
        sizes,
        index_maps,
        values,
        {"nrg_bal": "FC_OTH_HH_E", "unit": unit, "geo": geo, "time": str(year)},
        defaults=defs,
    )
    if cp is None or hh is None or not (cp >= 0 and hh >= 0) or cp + hh <= 0:
        return None, "fallback"
    return float(cp / (cp + hh)), "api"


def parse_nrg_d_hhq_metric_tj(
    root: dict[str, Any],
    *,
    geo: str,
    year: int,
) -> tuple[dict[str, float], SourceKind]:
    """Return metric_key -> TJ for six household end-use balances."""
    out: dict[str, float] = {}
    if not root:
        return out, "fallback"
    try:
        id_order, sizes, index_maps, values = _parse_dataset_values(root)
    except (KeyError, TypeError, ValueError):
        return out, "fallback"
    unit = "TJ"
    defs = _energy_coord_defaults()
    src: SourceKind = "api"
    for nrg_bal, mkey in NRG_D_HHQ_BAL_TO_METRIC:
        v = _observation(
            id_order,
            sizes,
            index_maps,
            values,
            {"nrg_bal": nrg_bal, "unit": unit, "geo": geo, "time": str(year)},
            defaults=defs,
        )
        if v is None:
            continue
        if v >= 0:
            out[mkey] = float(v)
    if not out:
        return out, "fallback"
    return out, src


def eurostat_geo_for_iso3(iso3: str, iso3_to_geo_labels: dict[str, list[str]]) -> str | None:
    """Pick a 2-letter Eurostat geo code from sidecar labels (e.g. EL for Greece)."""
    labels = list(iso3_to_geo_labels.get(iso3.strip().upper(), []))
    for lab in labels:
        s = str(lab).strip()
        if len(s) == 2 and s.isalpha():
            return s.upper()
    return None
