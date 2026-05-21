from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from PROXY_V2.core.alias import normalize_workbook_pollutant_cell


def _norm_sector_token(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value).strip().upper())


def _coerce_total(value: Any) -> float | None:
    if value is None:
        return None
    txt = str(value).strip()
    if not txt:
        return None
    try:
        return float(txt)
    except ValueError:
        return None


def _parse_sheet(df: pd.DataFrame) -> pd.DataFrame:
    expected = {"COUNTRY", "YEAR", "SECTOR", "POLLUTANT", "UNIT", "TOTAL"}
    cols = [str(c).strip().upper() for c in df.columns]

    if len(cols) == 1 and ";" in cols[0]:
        raw_col = df.columns[0]
        parsed = (
            df[raw_col]
            .astype(str)
            .str.split(";", n=5, expand=True)
            .rename(
                columns={
                    0: "COUNTRY",
                    1: "YEAR",
                    2: "SECTOR",
                    3: "POLLUTANT",
                    4: "UNIT",
                    5: "TOTAL",
                }
            )
        )
    else:
        normalized = {str(c): str(c).strip().upper() for c in df.columns}
        parsed = df.rename(columns=normalized).copy()
        if "TOTAL" not in parsed.columns and "VALUE" in parsed.columns:
            parsed = parsed.rename(columns={"VALUE": "TOTAL"})
        if "SECTOR" not in parsed.columns and "NFR" in parsed.columns:
            parsed = parsed.rename(columns={"NFR": "SECTOR"})

    missing = expected.difference(set(parsed.columns))
    if missing:
        raise ValueError(f"Missing expected columns in alpha sheet: {sorted(missing)}")

    out = parsed.loc[:, ["COUNTRY", "YEAR", "SECTOR", "POLLUTANT", "UNIT", "TOTAL"]].copy()
    out["COUNTRY"] = out["COUNTRY"].astype(str).str.strip().str.upper()
    out["YEAR"] = pd.to_numeric(out["YEAR"], errors="coerce").astype("Int64")
    out["SECTOR"] = out["SECTOR"].astype(str).str.strip()
    out["SECTOR_NORM"] = out["SECTOR"].map(_norm_sector_token)
    out["POLLUTANT_RAW"] = out["POLLUTANT"].astype(str).str.strip()
    out["POLLUTANT_ALPHA"] = out["POLLUTANT_RAW"].map(
        lambda x: normalize_workbook_pollutant_cell(x) if str(x).strip() else ""
    )
    out["UNIT"] = out["UNIT"].astype(str).str.strip()
    out["TOTAL_VALUE"] = out["TOTAL"].map(_coerce_total)
    out = out.dropna(subset=["YEAR", "TOTAL_VALUE"])
    return out


def read_alpha_workbook(workbook_path: Path) -> pd.DataFrame:
    """Read every sheet in the reported-emissions workbook and concatenate parsed rows."""
    xls = pd.ExcelFile(workbook_path)
    chunks: list[pd.DataFrame] = []
    for sheet in xls.sheet_names:
        df = pd.read_excel(workbook_path, sheet_name=sheet)
        parsed = _parse_sheet(df)
        chunks.append(parsed)
    if not chunks:
        raise ValueError(f"No data sheets found in workbook: {workbook_path}")
    data = pd.concat(chunks, ignore_index=True)
    data["YEAR"] = data["YEAR"].astype(int)
    return data
