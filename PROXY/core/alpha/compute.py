"""``compute_alpha`` CLI implementation and its mapping/grouping helpers.

This is the code that derives GNFR subsector alpha fractions from the reported-emissions
workbook (CEIP / EU27 long format) for a given country. GNFR totals and subsector code
groups are derived from ``PROXY/config/ceip/profiles/`` (see ``workbook_aggregation_spec``).
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from PROXY.core.io import write_json

from ._common import _norm_token, _safe_mean
from .aliases import normalize_country_alpha
from .workbook import read_alpha_workbook
from .workbook_aggregation_spec import load_workbook_aggregation_spec

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AlphaRequest:
    country: str
    pollutant: str | None = None


def _expand_grouped_codes(
    entries: list[Any], grouped: dict[str, list[str]]
) -> list[tuple[str, str]]:
    expanded: list[tuple[str, str]] = []
    for item in entries:
        if isinstance(item, str):
            token = item.strip()
            if token.startswith("group:"):
                group_name = token.split(":", 1)[1].strip()
                for code in grouped.get(group_name, []):
                    expanded.append((str(code), "exact"))
            elif token.endswith("*"):
                expanded.append((token[:-1], "prefix"))
            else:
                expanded.append((token, "exact"))
        elif isinstance(item, dict):
            code = str(item.get("code", "")).strip()
            match = str(item.get("match", "exact")).strip().lower()
            if code:
                expanded.append((code, "prefix" if match == "prefix" else "exact"))
    return expanded


def _match_mask(series: pd.Series, codes_with_mode: list[tuple[str, str]]) -> pd.Series:
    if not codes_with_mode:
        return pd.Series(False, index=series.index)
    mask = pd.Series(False, index=series.index)
    for raw_code, mode in codes_with_mode:
        code = _norm_token(raw_code)
        if not code:
            continue
        if mode == "prefix":
            mask = mask | series.str.startswith(code)
        else:
            mask = mask | (series == code)
    return mask


def compute_alpha(
    *,
    workbook_path: Path,
    output_dir: Path | None,
    request: AlphaRequest,
    repo_root: Path | None = None,
) -> dict:
    data = read_alpha_workbook(workbook_path)
    root = repo_root or Path(__file__).resolve().parents[3]
    mapping, grouped_raw = load_workbook_aggregation_spec(root)
    grouped = {str(k): [str(x) for x in (v or [])] for k, v in grouped_raw.items()}

    country = normalize_country_alpha(request.country)
    filt = data["COUNTRY"] == country
    if request.pollutant:
        filt = filt & (data["POLLUTANT"] == request.pollutant.strip().upper())
    data = data.loc[filt].copy()

    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []

    for gnfr_key, conf in mapping.items():
        if not isinstance(conf, dict):
            continue
        sector_total = conf.get("sector_total_codes", []) or []
        subsectors = conf.get("subsectors", {}) or {}
        sector_mask = _match_mask(
            data["SECTOR_NORM"], _expand_grouped_codes(sector_total, grouped)
        )
        sector_df = data.loc[sector_mask].copy()
        if sector_df.empty:
            diagnostics.append(
                {"sector": gnfr_key, "status": "no_sector_total_rows", "country": country}
            )
            continue

        for sub_name, sub_conf in subsectors.items():
            if isinstance(sub_conf, list):
                raw_codes = sub_conf
            elif isinstance(sub_conf, dict):
                raw_codes = sub_conf.get("codes", []) or []
                group_refs = sub_conf.get("groups", []) or []
                raw_codes = list(raw_codes) + [f"group:{g}" for g in group_refs]
            else:
                raw_codes = []

            sub_mask = _match_mask(
                data["SECTOR_NORM"], _expand_grouped_codes(raw_codes, grouped)
            )
            sub_df = data.loc[sub_mask].copy()
            if sub_df.empty:
                diagnostics.append(
                    {
                        "sector": gnfr_key,
                        "subsector": sub_name,
                        "status": "no_subsector_rows",
                        "country": country,
                    }
                )
                continue

            sec_year_poll = (
                sector_df.groupby(["YEAR", "POLLUTANT"], as_index=False)["TOTAL_VALUE"]
                .sum()
                .rename(columns={"TOTAL_VALUE": "SECTOR_TOTAL"})
            )
            sub_year_poll = (
                sub_df.groupby(["YEAR", "POLLUTANT"], as_index=False)["TOTAL_VALUE"]
                .sum()
                .rename(columns={"TOTAL_VALUE": "SUBSECTOR_TOTAL"})
            )

            merged = sec_year_poll.merge(sub_year_poll, on=["YEAR", "POLLUTANT"], how="inner")
            merged = merged[merged["SECTOR_TOTAL"] > 0.0].copy()
            if merged.empty:
                diagnostics.append(
                    {
                        "sector": gnfr_key,
                        "subsector": sub_name,
                        "status": "no_valid_years",
                        "country": country,
                    }
                )
                continue
            merged["ALPHA_YEAR"] = merged["SUBSECTOR_TOTAL"] / merged["SECTOR_TOTAL"]

            for pollutant, g in merged.groupby("POLLUTANT"):
                values = g["ALPHA_YEAR"].astype(float).tolist()
                alpha_mean = _safe_mean(values)
                rows.append(
                    {
                        "country": country,
                        "gnfr_sector": gnfr_key,
                        "subsector": sub_name,
                        "pollutant": str(pollutant),
                        "alpha": alpha_mean,
                        "n_years": int(len(values)),
                        "years": sorted(g["YEAR"].astype(int).unique().tolist()),
                    }
                )

    result = {
        "status": "ok",
        "country": country,
        "pollutant_filter": None if request.pollutant is None else request.pollutant.upper(),
        "workbook_path": str(workbook_path),
        "aggregation_spec": "builtin:ceip/profiles/solvents_subsectors.yaml+waste_families.yaml+constants",
        "records": rows,
        "diagnostics": diagnostics,
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "alpha_values.json", result)
        if rows:
            pd.DataFrame(rows).to_csv(output_dir / "alpha_values.csv", index=False)
        if diagnostics:
            pd.DataFrame(diagnostics).to_csv(
                output_dir / "alpha_diagnostics.csv", index=False
            )

    return {
        "status": "ok",
        "country": result["country"],
        "records": len(rows),
        "diagnostics": len(diagnostics),
        "output_dir": None if output_dir is None else str(output_dir),
    }
