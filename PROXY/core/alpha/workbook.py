"""Reported-emissions workbook reader used by ``compute_alpha`` and sector CEIP loaders.

Supports two layouts:

- Standard multi-column sheets with (at least) ``COUNTRY, YEAR, SECTOR, POLLUTANT, UNIT, TOTAL``.
- Single-column semicolon-delimited export (header cell lists the field names).

The public entry point is :func:`read_alpha_workbook`. The private underscore-prefixed aliases
``_read_alpha_workbook`` / ``_parse_sheet`` / ``_is_semicolon_single_column_table`` /
``_semicolon_rows_to_standard_columns`` are preserved for back-compat with sector code that
imports from :mod:`PROXY.core.alpha`.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ._common import _coerce_total, _norm_token


def _parse_sheet(df: pd.DataFrame) -> pd.DataFrame:
    expected = {"COUNTRY", "YEAR", "SECTOR", "POLLUTANT", "UNIT", "TOTAL"}
    cols = [str(c).strip().upper() for c in df.columns]

    # Workbook variant where every row is semicolon-delimited in a single column.
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
    out["SECTOR_NORM"] = out["SECTOR"].map(_norm_token)
    out["POLLUTANT"] = out["POLLUTANT"].astype(str).str.strip().str.upper()
    out["UNIT"] = out["UNIT"].astype(str).str.strip()
    out["TOTAL_VALUE"] = out["TOTAL"].map(_coerce_total)
    out = out.dropna(subset=["YEAR", "TOTAL_VALUE"])
    return out


def read_alpha_workbook(workbook_path: Path) -> pd.DataFrame:
    """Read every sheet in ``workbook_path`` and return a concatenated long dataframe."""
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


def _is_semicolon_single_column_table(df: pd.DataFrame) -> bool:
    if df.shape[1] != 1:
        return False
    name = str(df.columns[0])
    return ";" in name


def _semicolon_rows_to_standard_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Expand one-column CEIP exports where each cell is ``COUNTRY;YEAR;SECTOR;POLLUTANT;UNIT;TOTAL``.

    Header cells often carry only five semicolons (SECTOR omitted from the label) while data
    rows carry six fields; this mirrors ``_parse_sheet``'s semicolon branch.
    """
    raw_col = df.columns[0]
    parsed = (
        df[raw_col]
        .astype(str)
        .str.strip()
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
    hdr = parsed["COUNTRY"].astype(str).str.strip().str.upper() == "COUNTRY"
    parsed = parsed.loc[~hdr].copy()
    return parsed


_read_alpha_workbook = read_alpha_workbook
