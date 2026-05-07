from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

from ..constants import CLASS_TO_INDEX


def _parse_region_name(path: Path, max_lines: int = 30) -> str | None:
    with path.open(encoding="utf-8", errors="replace", newline="") as f:
        for _ in range(max_lines):
            line = f.readline()
            if not line:
                break
            raw = line.strip().strip('"')
            if raw.lower().startswith("region:"):
                return raw.split(":", 1)[1].strip()
    return None


_SPECIAL_REGION_TO_ISO3: dict[str, str] = {
    "albania": "ALB",
    "armenia": "ARM",
    "austria": "AUT",
    "belarus": "BLR",
    "belgium": "BEL",
    "bosnia and herzegovina": "BIH",
    "bulgaria": "BGR",
    "croatia": "HRV",
    "cyprus": "CYP",
    "czech republic": "CZE",
    "denmark": "DNK",
    "estonia": "EST",
    "finland": "FIN",
    "france": "FRA",
    "georgia": "GEO",
    "germany": "DEU",
    "greece": "GRC",
    "hungary": "HUN",
    "iceland": "ISL",
    "ireland": "IRL",
    "italy": "ITA",
    "kosovo": "XKX",
    "latvia": "LVA",
    "lithuania": "LTU",
    "luxembourg": "LUX",
    "malta": "MLT",
    "montenegro": "MNE",
    "netherlands": "NLD",
    "north macedonia": "MKD",
    "norway": "NOR",
    "poland": "POL",
    "portugal": "PRT",
    "republic of moldova": "MDA",
    "romania": "ROU",
    "serbia": "SRB",
    "slovak republic": "SVK",
    "slovenia": "SVN",
    "spain": "ESP",
    "sweden": "SWE",
    "switzerland": "CHE",
    "united kingdom": "GBR",
}


def region_name_to_iso3(name: str) -> str | None:
    key = name.strip().lower()
    if key in _SPECIAL_REGION_TO_ISO3:
        return _SPECIAL_REGION_TO_ISO3[key]
    try:
        import pycountry  # type: ignore
    except ImportError:
        return None
    try:
        c = pycountry.countries.lookup(name.strip())
        return str(c.alpha_3)
    except Exception:
        pass
    try:
        matches = pycountry.countries.search_fuzzy(name.strip(), limit=1)
        if matches:
            return str(matches[0].alpha_3)
    except Exception:
        pass
    return None


def index_gains_files(
    gains_dir: Path,
    overrides: dict[str, str],
    root: Path,
) -> dict[str, Path]:
    out: dict[str, Path] = {}
    if not gains_dir.is_dir():
        return out
    for path in sorted(gains_dir.glob("dom_share_ENE_*.csv")):
        region = _parse_region_name(path)
        iso3: str | None = None
        if region:
            iso3 = region_name_to_iso3(region)
        if iso3 is None:
            stem = path.stem
            if stem.startswith("dom_share_ENE_"):
                guess = stem[len("dom_share_ENE_") :].replace("_", " ")
                iso3 = region_name_to_iso3(guess)
        if iso3:
            out[iso3.upper()] = path
    for iso, rel in (overrides or {}).items():
        p = Path(rel)
        if not p.is_absolute():
            p = root / p
        if p.is_file():
            out[iso.strip().upper()] = p.resolve()
    return out


def _norm_pct_cell(raw: str) -> float:
    s = str(raw).strip().strip('"').replace(",", ".")
    if not s or s.lower() in {"n.a", "na", "..", "-", ""}:
        return 0.0
    try:
        return float(s) / 100.0
    except ValueError:
        return 0.0


def _load_gains_table(path: Path, year_col: str) -> tuple[list[str], list[list[str]]]:
    rows: list[list[str]] = []
    header_idx = None
    with path.open(encoding="utf-8", errors="replace", newline="") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if '"[% of fuel input]"' in line or "[% of fuel input]" in line:
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"No GAINS header row in {path}")
    block = "".join(lines[header_idx:])
    reader = csv.reader(block.splitlines())
    header = next(reader)
    clean_header = [c.strip().strip('"') for c in header]
    try:
        yi = clean_header.index(year_col)
    except ValueError as exc:
        raise ValueError(f"Year column {year_col!r} not in header of {path}") from exc
    for row in reader:
        if len(row) < 2:
            continue
        fuel = row[0].strip().strip('"')
        app = row[1].strip().strip('"') if len(row) > 1 else ""
        if not fuel:
            continue
        if len(row) <= yi:
            continue
        rows.append([fuel, app, row[yi]])
    return clean_header, rows


def _match_rule(fuel: str, app: str, rule: dict[str, Any]) -> bool:
    fc = rule.get("fuel_contains")
    if fc and str(fc).lower() not in fuel.lower():
        return False
    fca = rule.get("fuel_contains_all")
    if fca:
        fl = fuel.lower()
        for frag in fca:
            if str(frag).lower() not in fl:
                return False
    ac = rule.get("appliance_contains")
    if ac and str(ac).lower() not in app.lower():
        return False
    aca = rule.get("appliance_contains_all")
    if aca:
        for frag in aca:
            if str(frag).lower() not in app.lower():
                return False
    return True


def map_gains_row_to_class(
    fuel: str,
    app: str,
    rules: list[dict[str, Any]],
) -> str | None:
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        if "class" not in rule:
            continue
        if _match_rule(fuel, app, rule):
            cls = str(rule["class"])
            if cls in CLASS_TO_INDEX:
                return cls
    return None


def load_gains_rows(path: Path, year_col: str) -> list[tuple[str, str, str]]:
    """Return GAINS rows as ``(fuel, appliance, year_cell)`` tuples."""
    _, rows = _load_gains_table(path, year_col)
    return [(str(r[0]), str(r[1]), str(r[2])) for r in rows]
