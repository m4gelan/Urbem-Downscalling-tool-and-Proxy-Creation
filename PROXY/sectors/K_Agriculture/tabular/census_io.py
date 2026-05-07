"""Agriculture-specific spreadsheet parsing and census proxy helpers."""
from __future__ import annotations

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd

from PROXY.core.dataloaders import resolve_path as core_resolve_path
from PROXY.sectors.K_Agriculture.tabular.emission_factors import load_ipcc_defaults

SHEET_BOVINE = "Sheet 1"
SHEET_DAIRY = "Sheet 2"
SHEET_PIGS_LIGHT = "Sheet 3"
SHEET_PIGS_HEAVY = "Sheet 4"
SHEET_SHEEP = "Sheet 5"
SHEET_GOATS = "Sheet 6"

_IPCC_DEFAULTS = load_ipcc_defaults()
_CH4_EF = _IPCC_DEFAULTS["ch4_ef"]
_MAN_N = _IPCC_DEFAULTS["manure_n"]

EF_DAIRY_COW = _CH4_EF["dairy_cow"]
EF_OTHER_CATTLE = _CH4_EF["other_cattle"]
EF_PIG_LIGHT = _CH4_EF["pig_light"]
EF_PIG_HEAVY = _CH4_EF["pig_heavy"]
EF_SHEEP_BLEND = _CH4_EF["sheep_blend"]
EF_GOAT_BLEND = _CH4_EF["goat_blend"]

N_DAIRY = _MAN_N["dairy_cow"]
N_OTHER_CATTLE = _MAN_N["other_cattle"]
N_SWINE = _MAN_N["swine"]
N_SHEEP = _MAN_N["sheep"]
N_GOAT = _MAN_N["goat"]


def resolve_path(root: Path, p: str | Path) -> Path:
    return core_resolve_path(root, p).resolve()


def resolve_nuts_gpkg(root: Path, rel: str) -> Path:
    primary = resolve_path(root, rel)
    if primary.is_file():
        return primary
    for d in (root / "Data" / "geometry", root / "Auxiliaries"):
        if not d.is_dir():
            continue
        found = sorted(d.glob("**/NUTS*.gpkg"))
        if found:
            return found[-1]
    return primary


def _parse_count_cell(v) -> float | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    s = str(v).strip().replace("\xa0", " ").replace(" ", "")
    if not s or s in {":", "c", "e", "p", "b", "d"}:
        return None
    s = re.sub(r"[a-z]$", "", s, flags=re.IGNORECASE).replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def _nuts_code_from_cell(v, nuts2_ids: set[str]) -> str | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v).strip().upper()
    if not s or s == "NAN":
        return None
    if s in nuts2_ids:
        return s
    return None


def _read_excel_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Workbook contains no default style",
            category=UserWarning,
        )
        return pd.read_excel(path, sheet_name=sheet_name, header=None)


def _find_year_column(time_row: list, year: int) -> int | None:
    for idx, cell in enumerate(time_row):
        if cell is None or (isinstance(cell, float) and np.isnan(cell)):
            continue
        try:
            y = int(float(cell))
        except (TypeError, ValueError):
            continue
        if y == year:
            return idx
    return None


def parse_animal_sheet(
    excel_path: Path,
    sheet_name: str,
    year: int,
    nuts2_ids: set[str],
) -> dict[str, float]:
    raw = _read_excel_sheet(excel_path, sheet_name)
    time_row_idx = None
    for ri in range(min(15, len(raw))):
        row = raw.iloc[ri]
        if str(row.iloc[0]).strip().upper() == "TIME":
            time_row_idx = ri
            break
    if time_row_idx is None:
        raise ValueError(f"{sheet_name}: could not find TIME row.")

    time_row = raw.iloc[time_row_idx].tolist()
    col_year = _find_year_column(time_row, year)
    if col_year is None:
        years = []
        for c in time_row:
            try:
                years.append(int(float(c)))
            except (TypeError, ValueError):
                pass
        raise ValueError(f"{sheet_name}: year {year} not in columns. Found: {sorted(set(years))}")

    out: dict[str, float] = {}
    for r in range(time_row_idx + 2, len(raw)):
        code = _nuts_code_from_cell(raw.iloc[r, 0], nuts2_ids)
        if code is None:
            continue
        val = _parse_count_cell(raw.iloc[r, col_year])
        if val is not None:
            out[code] = val
    return out


def parse_fertilizer_sheet(
    excel_path: Path,
    sheet_name: str,
    year: int,
    nuts2_ids: set[str],
) -> dict[str, float]:
    raw = _read_excel_sheet(excel_path, sheet_name)
    time_row_idx = None
    for ri in range(min(20, len(raw))):
        row = raw.iloc[ri]
        if str(row.iloc[0]).strip().upper() == "TIME":
            time_row_idx = ri
            break
    if time_row_idx is None:
        raise ValueError(f"{sheet_name}: could not find TIME row.")

    time_row = raw.iloc[time_row_idx].tolist()
    col_year = _find_year_column(time_row, year)
    if col_year is None:
        years = []
        for c in time_row:
            try:
                years.append(int(float(c)))
            except (TypeError, ValueError):
                pass
        raise ValueError(f"{sheet_name}: year {year} not in columns. Found: {sorted(set(years))}")

    out: dict[str, float] = {}
    for r in range(time_row_idx + 2, len(raw)):
        code = _nuts_code_from_cell(raw.iloc[r, 0], nuts2_ids)
        if code is None:
            continue
        val = _parse_count_cell(raw.iloc[r, col_year])
        if val is not None:
            out[code] = float(val)
    return out


def _ch4_from_thousands_heads(ths: float, ef: float) -> float:
    return ths * 1000.0 * ef


def build_total_ch4_kg_per_nuts(
    excel_path: Path,
    year: int,
    nuts2_ids: set[str],
) -> dict[str, float]:
    bovine = parse_animal_sheet(excel_path, SHEET_BOVINE, year, nuts2_ids)
    dairy = parse_animal_sheet(excel_path, SHEET_DAIRY, year, nuts2_ids)
    pigs_lt50 = parse_animal_sheet(excel_path, SHEET_PIGS_LIGHT, year, nuts2_ids)
    pigs_ge50 = parse_animal_sheet(excel_path, SHEET_PIGS_HEAVY, year, nuts2_ids)
    sheep = parse_animal_sheet(excel_path, SHEET_SHEEP, year, nuts2_ids)
    goats = parse_animal_sheet(excel_path, SHEET_GOATS, year, nuts2_ids)
    total: dict[str, float] = {nid: 0.0 for nid in nuts2_ids}
    for nid in nuts2_ids:
        b = bovine.get(nid)
        d = dairy.get(nid)
        if b is not None and d is not None:
            total[nid] += _ch4_from_thousands_heads(d, EF_DAIRY_COW)
            total[nid] += _ch4_from_thousands_heads(max(0.0, b - d), EF_OTHER_CATTLE)
        elif b is not None:
            total[nid] += _ch4_from_thousands_heads(b, EF_OTHER_CATTLE)
        elif d is not None:
            total[nid] += _ch4_from_thousands_heads(d, EF_DAIRY_COW)
        if nid in pigs_lt50:
            total[nid] += _ch4_from_thousands_heads(pigs_lt50[nid], EF_PIG_LIGHT)
        if nid in pigs_ge50:
            total[nid] += _ch4_from_thousands_heads(pigs_ge50[nid], EF_PIG_HEAVY)
        if nid in sheep:
            total[nid] += _ch4_from_thousands_heads(sheep[nid], EF_SHEEP_BLEND)
        if nid in goats:
            total[nid] += _ch4_from_thousands_heads(goats[nid], EF_GOAT_BLEND)
    return total


def build_manure_n_proxy_per_nuts(
    excel_path: Path,
    year: int,
    nuts2_ids: set[str],
) -> dict[str, float]:
    bovine = parse_animal_sheet(excel_path, SHEET_BOVINE, year, nuts2_ids)
    dairy = parse_animal_sheet(excel_path, SHEET_DAIRY, year, nuts2_ids)
    pigs_lt50 = parse_animal_sheet(excel_path, SHEET_PIGS_LIGHT, year, nuts2_ids)
    pigs_ge50 = parse_animal_sheet(excel_path, SHEET_PIGS_HEAVY, year, nuts2_ids)
    sheep = parse_animal_sheet(excel_path, SHEET_SHEEP, year, nuts2_ids)
    goats = parse_animal_sheet(excel_path, SHEET_GOATS, year, nuts2_ids)
    total: dict[str, float] = {nid: 0.0 for nid in nuts2_ids}
    for nid in nuts2_ids:
        b = bovine.get(nid)
        d = dairy.get(nid)
        if b is not None and d is not None:
            total[nid] += d * N_DAIRY
            total[nid] += max(0.0, b - d) * N_OTHER_CATTLE
        elif b is not None:
            total[nid] += b * N_OTHER_CATTLE
        elif d is not None:
            total[nid] += d * N_DAIRY
        total[nid] += (pigs_lt50.get(nid, 0.0) + pigs_ge50.get(nid, 0.0)) * N_SWINE
        if nid in sheep:
            total[nid] += sheep[nid] * N_SHEEP
        if nid in goats:
            total[nid] += goats[nid] * N_GOAT
    return total


def resolve_animal_xlsx(path: Path, project_root: Path | None = None) -> Path:
    if path.is_file():
        return path
    alt = sorted(path.parent.glob("agr_r_animal*.xlsx"))
    if alt:
        return alt[-1]
    alt2 = sorted(path.parent.glob("Livestock*.xlsx"))
    if alt2:
        return alt2[-1]
    if project_root is not None:
        for d in (
            project_root / "Auxiliaries" / "Data",
            project_root / "Auxiliaries",
            project_root / "Data" / "Agriculture",
        ):
            if not d.is_dir():
                continue
            for pat in ("agr_r_animal*.xlsx", "Livestock*.xlsx"):
                found = sorted(d.glob(pat))
                if found:
                    return found[-1]
    raise FileNotFoundError(f"Animal spreadsheet not found: {path}")
