from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from PROXY_V2.core.alias import cams_pollutant_var


@dataclass(frozen=True)
class EmepTableRow:
    emep_fuel: str
    emep_appliance: str
    gains_classes: tuple[str, ...]
    gains_fuels: tuple[str, ...]
    ef: dict[str, Any]


def load_emep(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: EMEP config must be a YAML mapping")
    return doc


def _as_str_list(val: Any) -> tuple[str, ...]:
    if val is None:
        return ()
    if isinstance(val, str):
        s = val.strip()
        return (s,) if s else ()
    if isinstance(val, list):
        out = tuple(str(x).strip() for x in val if str(x).strip())
        return out
    return ()


def parse_emep_tables(emep: dict[str, Any]) -> list[EmepTableRow]:
    tables = emep.get("tables")
    if not isinstance(tables, list):
        return []
    out: list[EmepTableRow] = []
    for t in tables:
        if not isinstance(t, dict):
            continue
        classes = _as_str_list(t.get("gains_class"))
        fuels = _as_str_list(t.get("gains_fuel"))
        if not classes or not fuels:
            continue
        ef = t.get("ef")
        if not isinstance(ef, dict):
            continue
        out.append(
            EmepTableRow(
                emep_fuel=str(t.get("fuel", "")),
                emep_appliance=str(t.get("appliance", "")),
                gains_classes=classes,
                gains_fuels=fuels,
                ef=ef,
            )
        )
    return out


def _ef_scalar(ef_dict: dict[str, Any], pollutant_label: str) -> float | None:
    key = cams_pollutant_var(pollutant_label)
    v = ef_dict.get(key)
    if v == "missing" or v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _token_overlap(a: str, b: str) -> int:
    sa = set(re.findall(r"[a-z0-9]+", a.lower()))
    sb = set(re.findall(r"[a-z0-9]+", b.lower()))
    if not sa or not sb:
        return 0
    return len(sa & sb)


def ef_kg_per_tj(
    pollutant_label: str,
    gains_class: str,
    gains_fuel: str,
    gains_appliance: str,
    emep_rows: list[EmepTableRow],
) -> float:
    """
    EF (g/GJ) for a mapped GAINS row via explicit gains_class + gains_fuel on each EMEP table row.
    If several rows match, pick the one whose EMEP appliance label best overlaps the GAINS appliance text.
    """
    hits = [
        r
        for r in emep_rows
        if gains_class in r.gains_classes and gains_fuel in r.gains_fuels
    ]
    if not hits:
        return 0.0
    if len(hits) == 1:
        row = hits[0]
    else:
        row = max(hits, key=lambda r: _token_overlap(gains_appliance, r.emep_appliance))
    val = _ef_scalar(row.ef, pollutant_label)
    return float(val) if val is not None else 0.0
