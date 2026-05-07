"""
EMEP emission-factor lookup for GNFR C.

**What**: resolve kg pollutant / TJ fuel energy from packaged JSON tables.
**Why**: centralises EF scoring so ``assemble`` stays a thin sum over GAINS rows.
**Inputs**: ``EMEP_emission_factors.yaml`` (or legacy ``.json``) dict, gains fuel/appliance strings, optional fuel hints.
**Outputs**: scalar EF per (pollutant, fuel, appliance) or 0.0 if unresolved.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .sidecar_io import load_sidecar_dict


def load_emep(path: Path) -> dict[str, Any]:
    return load_sidecar_dict(path)


def _inv_key_for_pollutant(p: str, emep: dict[str, Any]) -> str | None:
    aliases = emep.get("pollutant_aliases_for_inventory") or {}
    key = str(aliases.get(p, p))
    inv = emep.get("inventory_ef_kg_per_tj") or {}
    if key in inv:
        return key
    if p in inv:
        return p
    return None


def _ef_pollutant_keys(pollutant_output: str) -> list[str]:
    p = pollutant_output.lower().strip()
    aliases = {"sox": "so2", "co2_total": "co2", "pm25": "pm2_5"}
    keys = [p]
    if p in aliases:
        keys.append(aliases[p])
    out: list[str] = []
    for k in keys:
        if k not in out:
            out.append(k)
    return out


def _ef_value_from_row_dict(ef_dict: Any, pollutant_output: str) -> float | None:
    if not isinstance(ef_dict, dict):
        return None
    for k in _ef_pollutant_keys(pollutant_output):
        v = ef_dict.get(k)
        if v == "missing" or v is None:
            continue
        try:
            return float(v)
        except (TypeError, ValueError):
            continue
    return None


def _token_overlap(a: str, b: str) -> int:
    sa = set(re.findall(r"[a-z0-9]+", a.lower()))
    sb = set(re.findall(r"[a-z0-9]+", b.lower()))
    if not sa or not sb:
        return 0
    return len(sa & sb)


def _emep_fuel_hint_bonus(
    gains_fuel: str,
    emep_row_fuel: str,
    hints: list[dict[str, Any]] | None,
) -> int:
    if not hints:
        return 0
    g = gains_fuel.lower()
    e = emep_row_fuel.lower()
    bonus = 0
    for h in hints:
        if not isinstance(h, dict):
            continue
        gc = str(h.get("gains_fuel_contains", "")).lower().strip()
        if not gc or gc not in g:
            continue
        em = str(h.get("emep_fuel_contains", h.get("emep_fuel", ""))).lower().strip()
        if em and em in e:
            try:
                bonus += int(h.get("score_bonus", 25))
            except (TypeError, ValueError):
                bonus += 25
    return bonus


def ef_kg_per_tj(
    pollutant_output: str,
    fuel: str,
    appliance: str,
    emep: dict[str, Any],
    *,
    emep_fuel_hints: list[dict[str, Any]] | None = None,
) -> float:
    tables = emep.get("tables")
    if isinstance(tables, list) and tables:
        best_score = -1
        best_val: float | None = None
        for t in tables:
            if not isinstance(t, dict):
                continue
            ef_d = t.get("ef")
            emep_f = str(t.get("fuel", ""))
            sc = _token_overlap(emep_f, fuel) + _token_overlap(str(t.get("appliance", "")), appliance)
            sc += _emep_fuel_hint_bonus(fuel, emep_f, emep_fuel_hints)
            if sc <= 0:
                continue
            val = _ef_value_from_row_dict(ef_d, pollutant_output)
            if val is None:
                continue
            if sc > best_score:
                best_score = sc
                best_val = val
        if best_val is not None:
            return float(best_val)

    rows = emep.get("per_row_ef_kg_per_tj") or []
    fuel_l, app_l = fuel.lower(), appliance.lower()
    for row in rows:
        if str(row.get("pollutant", "")).lower() != pollutant_output.lower():
            continue
        fpat = row.get("fuel_contains")
        apat = row.get("appliance_contains")
        if fpat and str(fpat).lower() not in fuel_l:
            continue
        if apat and str(apat).lower() not in app_l:
            continue
        val = row.get("value")
        if val is None:
            continue
        try:
            return float(val)
        except (TypeError, ValueError):
            continue
    ik = _inv_key_for_pollutant(pollutant_output, emep)
    inv = emep.get("inventory_ef_kg_per_tj") or {}
    if ik and inv.get(ik) is not None:
        try:
            return float(inv[ik])
        except (TypeError, ValueError):
            pass
    return 0.0
