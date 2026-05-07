from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from .constants import CLASS_TO_INDEX, MODEL_CLASSES

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


def load_emep(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
            sc = _token_overlap(emep_f, fuel) + _token_overlap(
                str(t.get("appliance", "")), appliance
            )
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


def build_M_for_country(
    iso3: str,
    gains_path: Path | None,
    year_col: str,
    mapping_rules: list[dict[str, Any]],
    f_enduse: dict[str, float],
    emep: dict[str, Any],
    pollutant_outputs: list[str],
    *,
    emep_fuel_hints: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    """
    Shape (n_pollutants, n_classes): M[p, k] = sum_rows A_corr(row) * EF(p,row).
    """
    n_p = len(pollutant_outputs)
    n_k = len(MODEL_CLASSES)
    M = np.zeros((n_p, n_k), dtype=np.float64)
    if gains_path is None or not gains_path.is_file():
        return M
    _, rows = _load_gains_table(gains_path, year_col)
    for fuel, app, ycell in rows:
        cls = map_gains_row_to_class(fuel, app, mapping_rules)
        if cls is None:
            continue
        share = _norm_pct_cell(ycell)
        if share <= 0:
            continue
        f_use = float(f_enduse.get(cls, 1.0))
        a_corr = share * f_use
        ki = CLASS_TO_INDEX[cls]
        for pi, pol in enumerate(pollutant_outputs):
            ef = ef_kg_per_tj(pol, fuel, app, emep, emep_fuel_hints=emep_fuel_hints)
            M[pi, ki] += a_corr * ef
    return M


def load_gains_mapping(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rules = data.get("rules")
    if not isinstance(rules, list):
        rules = []
    clean_rules = [r for r in rules if isinstance(r, dict) and "class" in r]
    hints = data.get("emep_fuel_hints") or []
    if not isinstance(hints, list):
        hints = []
    clean_hints = [h for h in hints if isinstance(h, dict)]
    return clean_rules, clean_hints
