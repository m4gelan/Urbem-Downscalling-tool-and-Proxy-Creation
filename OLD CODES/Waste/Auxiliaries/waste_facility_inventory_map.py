#!/usr/bin/env python3
"""
Unified waste-relevant facility inventory from EEA E-PRTR / IED CSV extracts,
with optional CAMS GNFR J (Waste) point overlay and Folium map.

Broad recovery: no default NMVOC-only, sector-5-only, or Annex-5-only filters.
Classification uses keyword groups (editable below) plus source-file flags.

Outputs (under --output-dir):
  - waste_facilities_master.csv
  - waste_facilities_raw_union.csv
  - waste_facilities_duplicate_candidates.csv
  - waste_facilities_summary_by_category.csv
  - waste_cams_nearest_facility.csv (if CAMS + scipy)
  - waste_facility_nearest_cams.csv (if CAMS + scipy)
  - waste_facilities_map.html (unless disabled)

Requires: pandas, numpy, folium, branca; xarray + netCDF4 for CAMS; scipy optional (nearest-neighbor).
"""

from __future__ import annotations

import argparse
import html
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    from rapidfuzz import fuzz as _rf_fuzz

    def _name_similarity(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        return float(_rf_fuzz.token_set_ratio(a, b)) / 100.0

except ImportError:

    def _name_similarity(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        a, b = a.lower().strip(), b.lower().strip()
        if a == b:
            return 1.0
        return 0.5 if (a in b or b in a) else 0.0


# ---------------------------------------------------------------------------
# Keyword groups (edit here). Matching is case-insensitive on concatenated text.
# ---------------------------------------------------------------------------

WASTE_GENERIC = (
    "waste",
    "wastewater",
    "sewage",
    "sludge",
    "landfill",
    "disposal",
    "recovery",
    "treatment",
    "recycling",
    "compost",
    "anaerobic",
    "digestion",
    "biogas",
    "transfer",
)

INCINERATION = (
    "incineration",
    "incinerator",
    "thermal treatment",
    "combustion",
    "waste-to-energy",
    "waste to energy",
    "energy recovery",
)

CO_INCINERATION = (
    "co-incineration",
    "coincineration",
    "co firing",
    "co-firing",
    "alternative fuel",
    "rdf",
    "refuse derived fuel",
    "srf",
    "solid recovered fuel",
)

WASTEWATER = (
    "urban wastewater",
    "wastewater treatment",
    "sewage treatment",
    "wwtp",
    "uwwtp",
    "effluent",
)

LANDFILL = ("landfill", "dump", "disposal site")

WTE_EXTRA = ("wte", "waste derived fuel", "waste-derived", "municipal solid waste", "mswi")

# Merged facilities: keep only if ``reporting_year_max`` (max ``reportingYear`` across E-PRTR rows) exceeds this.
DEFAULT_EPRTR_REPORTING_YEAR_CUTOFF = 2018

# Primary category display order (highest precedence wins for primary_category)
CATEGORY_PRIORITY: tuple[str, ...] = (
    "waste_incineration",
    "co_incineration",
    "waste_to_energy",
    "landfill",
    "wastewater",
    "waste_transfer",
    "ied_waste_installation",
    "waste_air_release",
    "other_potential_waste",
    "unclear_but_relevant",
)

# Folium colors (hex) by primary category + CAMS
CATEGORY_COLOR: dict[str, str] = {
    "waste_air_release": "#e41a1c",
    "waste_transfer": "#984ea3",
    "ied_waste_installation": "#377eb8",
    "waste_incineration": "#ff7f00",
    "co_incineration": "#a65628",
    "wastewater": "#00ced1",
    "landfill": "#006400",
    "waste_to_energy": "#000000",
    "other_potential_waste": "#999999",
    "unclear_but_relevant": "#cccccc",
    "cams_gnfr_j": "#ff00ff",
}

# EEA UWWTD treatment plant points (GeoPackage layer), distinct from PRTR ``wastewater`` colour.
UWWTD_GPKG_POINT_COLOR = "#009acd"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _first_col(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        k = cand.lower()
        if k in cols:
            return cols[k]
    return None


def _norm_id(x: Any) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    return s


def _norm_name(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _norm_country(s: Any) -> str:
    if pd.isna(s):
        return ""
    return str(s).strip()


def _round_coord(v: float, nd: int = 4) -> float:
    if not math.isfinite(v):
        return float("nan")
    return round(float(v), nd)


def _concat_text(parts: Iterable[Any]) -> str:
    out: list[str] = []
    for p in parts:
        if p is None or (isinstance(p, float) and not math.isfinite(p)):
            continue
        if isinstance(p, str) and not p.strip():
            continue
        out.append(str(p))
    return " | ".join(out)


def _matches_any(text: str, keywords: tuple[str, ...]) -> bool:
    t = text.lower()
    return any(k in t for k in keywords)


def _read_table(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        with path.open("rb") as fh:
            head = fh.read(4096).decode("utf-8", errors="replace")
        sep = ";" if head.count(";") > head.count(",") else ","
        return pd.read_csv(path, sep=sep, low_memory=False)
    except Exception as exc:
        print(f"Warning: failed to read {path}: {exc}")
        return None


def _iso3_to_eprtr_country() -> dict[str, str]:
    return {
        "AUT": "Austria",
        "BEL": "Belgium",
        "BGR": "Bulgaria",
        "HRV": "Croatia",
        "CYP": "Cyprus",
        "CZE": "Czechia",
        "DNK": "Denmark",
        "EST": "Estonia",
        "FIN": "Finland",
        "FRA": "France",
        "DEU": "Germany",
        "GRC": "Greece",
        "HUN": "Hungary",
        "ISL": "Iceland",
        "IRL": "Ireland",
        "ITA": "Italy",
        "LVA": "Latvia",
        "LTU": "Lithuania",
        "LUX": "Luxembourg",
        "MLT": "Malta",
        "NLD": "Netherlands",
        "NOR": "Norway",
        "POL": "Poland",
        "PRT": "Portugal",
        "ROU": "Romania",
        "SRB": "Serbia",
        "SVK": "Slovakia",
        "SVN": "Slovenia",
        "ESP": "Spain",
        "SWE": "Sweden",
        "CHE": "Switzerland",
        "GBR": "United Kingdom",
    }


def _country_filter_value(iso_or_name: str, all_countries: bool) -> str | None:
    if all_countries:
        return None
    u = iso_or_name.strip()
    if len(u) == 3:
        m = _iso3_to_eprtr_country()
        return m.get(u.upper(), u)
    return u


# --- loaders (return standardized long-form facility rows) -----------------

STANDARD_COLS = [
    "source_file",
    "FacilityInspireId",
    "InstallationInspireId",
    "ParentFacilityInspireId",
    "facility_name",
    "country",
    "longitude",
    "latitude",
    "reporting_year_min",
    "reporting_year_max",
    "air_rows",
    "air_releases_sum",
    "pollutants_concat",
    "air_pollutants_json",
    "sector_codes_concat",
    "sector_names_concat",
    "annex_concat",
    "waste_transfers_sum",
    "ied_activity_concat",
    "lcp_mw",
    "text_blob",
    "from_f1_4",
    "from_f4_2",
    "from_f5_2",
    "from_f6_1",
    "from_f7_1",
]


def _empty_standard() -> pd.DataFrame:
    return pd.DataFrame(columns=STANDARD_COLS)


def _parse_air_pollutants_json(s: object) -> dict[str, float]:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return {}
    if isinstance(s, dict):
        raw = s
    else:
        try:
            raw = json.loads(str(s))
        except (json.JSONDecodeError, TypeError):
            return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for k, v in raw.items():
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(fv) and fv > 0:
            out[str(k)] = fv
    return out


def _dump_air_pollutants_json(d: dict[str, float]) -> str:
    d2 = {k: float(v) for k, v in d.items() if math.isfinite(float(v)) and float(v) > 0}
    return json.dumps(d2, ensure_ascii=False, sort_keys=True)


def _fmt_emission_cell(x: float) -> str:
    if not math.isfinite(x) or x == 0.0:
        return f"{x:.6g}"
    ax = abs(x)
    if ax >= 1e9:
        return f"{x/1e9:.4f}e9"
    if ax >= 1e6:
        return f"{x/1e6:.4f}e6"
    if ax >= 1e3:
        return f"{x/1e3:.4f}e3"
    return f"{x:.6g}"


def _air_pollutants_table_html(json_str: object) -> str:
    d = _parse_air_pollutants_json(json_str)
    if not d:
        return (
            "<p style='color:#666;margin:6px 0 0 0'>No positive finite F1_4 AIR totals "
            "(by pollutant) in merged inventory.</p>"
        )
    pairs = sorted(d.items(), key=lambda t: (-t[1], t[0].lower()))
    lines = [
        "<div style='margin-top:6px'><b>F1_4 AIR emissions</b> "
        "<span style='color:#555'>(kg/year, &gt; 0)</span></div>",
        "<table style='border-collapse:collapse;width:100%;font-size:11px;margin-top:4px'>",
        "<tr><th align='left'>Pollutant</th><th align='right'>kg/year</th></tr>",
    ]
    for name, val in pairs:
        lines.append(
            f"<tr><td>{html.escape(name)}</td><td align='right'>{html.escape(_fmt_emission_cell(val))}</td></tr>"
        )
    lines.append("</table>")
    return "".join(lines)


_cams_j_map_module_cache: Any = None


def _get_cams_j_map_module() -> Any:
    global _cams_j_map_module_cache
    if _cams_j_map_module_cache is not None:
        return _cams_j_map_module_cache
    path = Path(__file__).resolve().parent / "cams_J_greece_map.py"
    spec = importlib.util.spec_from_file_location("_cams_J_map_dyn", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load cams_J_greece_map.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _cams_j_map_module_cache = mod
    return mod


def _apply_bbox(df: pd.DataFrame, lon_col: str, lat_col: str, bbox: tuple[float, float, float, float] | None) -> pd.DataFrame:
    if bbox is None:
        return df
    lon0, lat0, lon1, lat1 = bbox
    return df[
        (df[lon_col].astype(float) >= lon0)
        & (df[lon_col].astype(float) <= lon1)
        & (df[lat_col].astype(float) >= lat0)
        & (df[lat_col].astype(float) <= lat1)
    ]


def load_f1_4_facilities(
    path: Path,
    *,
    country_filter: str | None,
    bbox: tuple[float, float, float, float] | None,
) -> pd.DataFrame:
    df = _read_table(path)
    if df is None or df.empty:
        return _empty_standard()
    c_fid = _first_col(df, ["FacilityInspireId", "facilityInspireId"])
    c_lon = _first_col(df, ["Longitude", "longitude"])
    c_lat = _first_col(df, ["Latitude", "latitude"])
    c_ctry = _first_col(df, ["countryName", "CountryName"])
    c_year = _first_col(df, ["reportingYear", "ReportingYear"])
    c_rel = _first_col(df, ["Releases", "releases"])
    c_poll = _first_col(df, ["Pollutant", "pollutant"])
    c_sec = _first_col(df, ["EPRTR_SectorCode", "eprtr_sectorcode"])
    c_snm = _first_col(df, ["EPRTR_SectorName", "eprtr_sectorname"])
    c_ann = _first_col(df, ["EPRTRAnnexIMainActivity", "eprtranneximainactivity"])
    c_name = _first_col(df, ["facilityName", "FacilityName"])
    c_city = _first_col(df, ["city", "City"])
    req = [c_fid, c_lon, c_lat, c_ctry]
    if any(x is None for x in req):
        print("F1_4: missing one of required columns; skipping.")
        return _empty_standard()
    work = df.copy()
    work[c_fid] = work[c_fid].map(_norm_id)
    work = work[work[c_fid].astype(str).str.len() > 0]
    work[c_lon] = pd.to_numeric(work[c_lon], errors="coerce")
    work[c_lat] = pd.to_numeric(work[c_lat], errors="coerce")
    work = work[work[c_lon].notna() & work[c_lat].notna()]
    if country_filter:
        work = work[work[c_ctry].astype(str).str.strip() == country_filter]
    work = _apply_bbox(work, c_lon, c_lat, bbox)
    print(f"F1_4: {len(work)} rows after id/coord/country/bbox filter")
    if work.empty:
        return _empty_standard()

    # Avoid groupby().apply(): newer pandas may omit grouping columns from `g`.
    rows_out: list[dict[str, Any]] = []
    for fid, g in work.groupby(c_fid, sort=False):
        fid_s = _norm_id(fid)
        y = pd.to_numeric(g[c_year], errors="coerce") if c_year else pd.Series(dtype=float)
        rel = pd.to_numeric(g[c_rel], errors="coerce") if c_rel else pd.Series(dtype=float)
        pols = sorted({str(x) for x in g[c_poll].dropna().unique()}) if c_poll else []
        pol_d: dict[str, float] = {}
        if c_poll and c_rel:
            tmp = g[[c_poll, c_rel]].copy()
            tmp["_v"] = pd.to_numeric(tmp[c_rel], errors="coerce")
            gsum = tmp.dropna(subset=[c_poll]).groupby(c_poll, dropna=True)["_v"].sum()
            for pol_name, v in gsum.items():
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                ps = str(pol_name).strip()
                if not ps:
                    continue
                if math.isfinite(fv) and fv > 0:
                    pol_d[ps] = fv
        air_j = _dump_air_pollutants_json(pol_d)
        secs = sorted({str(x) for x in g[c_sec].dropna().unique()}) if c_sec else []
        snms = sorted({str(x) for x in g[c_snm].dropna().unique()}) if c_snm else []
        anns = sorted({str(x) for x in g[c_ann].dropna().unique()}) if c_ann else []
        blob = _concat_text(
            [
                g[c_name].iloc[0] if c_name else "",
                g[c_city].iloc[0] if c_city else "",
                " ".join(snms),
                " ".join(anns),
                " ".join(secs),
            ]
        )
        rows_out.append(
            {
                "source_file": "F1_4",
                "FacilityInspireId": fid_s,
                "InstallationInspireId": "",
                "ParentFacilityInspireId": "",
                "facility_name": str(g[c_name].iloc[0]) if c_name else "",
                "country": str(g[c_ctry].iloc[0]).strip(),
                "longitude": float(g[c_lon].iloc[0]),
                "latitude": float(g[c_lat].iloc[0]),
                "reporting_year_min": int(y.min()) if y.notna().any() else "",
                "reporting_year_max": int(y.max()) if y.notna().any() else "",
                "air_rows": int(len(g)),
                "air_releases_sum": float(rel.sum()) if rel.notna().any() else 0.0,
                "pollutants_concat": "|".join(pols[:400]),
                "air_pollutants_json": air_j,
                "sector_codes_concat": "|".join(secs[:120]),
                "sector_names_concat": "|".join(snms[:120]),
                "annex_concat": "|".join(anns[:200]),
                "waste_transfers_sum": 0.0,
                "ied_activity_concat": "",
                "lcp_mw": float("nan"),
                "text_blob": blob,
                "from_f1_4": True,
                "from_f4_2": False,
                "from_f5_2": False,
                "from_f6_1": False,
                "from_f7_1": False,
            }
        )
    out = pd.DataFrame(rows_out)
    print(f"F1_4: {len(out)} facilities after aggregation")
    return out


def load_f4_2_facilities(
    path: Path,
    *,
    country_filter: str | None,
    bbox: tuple[float, float, float, float] | None,
) -> pd.DataFrame:
    df = _read_table(path)
    if df is None or df.empty:
        return _empty_standard()
    c_fid = _first_col(df, ["FacilityInspireId"])
    c_lon = _first_col(df, ["Longitude"])
    c_lat = _first_col(df, ["Latitude"])
    c_ctry = _first_col(df, ["countryName", "CountryName"])
    c_year = _first_col(df, ["reportingYear"])
    c_wt = _first_col(df, ["wasteTransfers", "WasteTransfers"])
    c_sec = _first_col(df, ["EPRTR_SectorCode"])
    c_snm = _first_col(df, ["EPRTR_SectorName"])
    c_ann = _first_col(df, ["EPRTRAnnexIMainActivity"])
    c_name = _first_col(df, ["facilityName"])
    if not all([c_fid, c_lon, c_lat, c_ctry]):
        print("F4_2: missing columns; skipping.")
        return _empty_standard()
    work = df.copy()
    work[c_fid] = work[c_fid].map(_norm_id)
    work = work[work[c_fid].astype(str).str.len() > 0]
    work[c_lon] = pd.to_numeric(work[c_lon], errors="coerce")
    work[c_lat] = pd.to_numeric(work[c_lat], errors="coerce")
    work = work[work[c_lon].notna() & work[c_lat].notna()]
    if country_filter:
        work = work[work[c_ctry].astype(str).str.strip() == country_filter]
    work = _apply_bbox(work, c_lon, c_lat, bbox)
    print(f"F4_2: {len(work)} rows after filter")
    if work.empty:
        return _empty_standard()

    rows_f4: list[dict[str, Any]] = []
    for fid, g in work.groupby(c_fid, sort=False):
        fid_s = _norm_id(fid)
        y = pd.to_numeric(g[c_year], errors="coerce")
        wt = pd.to_numeric(g[c_wt], errors="coerce") if c_wt else pd.Series(dtype=float)
        secs = sorted({str(x) for x in g[c_sec].dropna().unique()}) if c_sec else []
        snms = sorted({str(x) for x in g[c_snm].dropna().unique()}) if c_snm else []
        anns = sorted({str(x) for x in g[c_ann].dropna().unique()}) if c_ann else []
        blob = _concat_text([g[c_name].iloc[0] if c_name else "", " ".join(snms), " ".join(anns)])
        rows_f4.append(
            {
                "source_file": "F4_2",
                "FacilityInspireId": fid_s,
                "InstallationInspireId": "",
                "ParentFacilityInspireId": "",
                "facility_name": str(g[c_name].iloc[0]) if c_name else "",
                "country": str(g[c_ctry].iloc[0]).strip(),
                "longitude": float(g[c_lon].iloc[0]),
                "latitude": float(g[c_lat].iloc[0]),
                "reporting_year_min": int(y.min()) if y.notna().any() else "",
                "reporting_year_max": int(y.max()) if y.notna().any() else "",
                "air_rows": 0,
                "air_releases_sum": 0.0,
                "pollutants_concat": "",
                "air_pollutants_json": "{}",
                "sector_codes_concat": "|".join(secs[:120]),
                "sector_names_concat": "|".join(snms[:120]),
                "annex_concat": "|".join(anns[:200]),
                "waste_transfers_sum": float(wt.sum()) if wt.notna().any() else 0.0,
                "ied_activity_concat": "",
                "lcp_mw": float("nan"),
                "text_blob": blob,
                "from_f1_4": False,
                "from_f4_2": True,
                "from_f5_2": False,
                "from_f6_1": False,
                "from_f7_1": False,
            }
        )
    out = pd.DataFrame(rows_f4)
    print(f"F4_2: {len(out)} facilities aggregated")
    return out


def load_f6_1_installations(
    path: Path,
    *,
    country_filter: str | None,
    bbox: tuple[float, float, float, float] | None,
) -> pd.DataFrame:
    df = _read_table(path)
    if df is None or df.empty:
        return _empty_standard()
    c_inst = _first_col(df, ["InstallationInspireId", "installationInspireId"])
    c_parent = _first_col(df, ["parent_facilityInspireId", "ParentFacilityInspireId"])
    c_lon = _first_col(df, ["Longitude"])
    c_lat = _first_col(df, ["Latitude"])
    c_ctry = _first_col(df, ["CountryName", "countryName"])
    c_year = _first_col(df, ["reportingYear"])
    c_name = _first_col(df, ["installationName", "InstallationName"])
    c_city = _first_col(df, ["City_of_Facility", "city_of_facility"])
    c_ied = _first_col(df, ["IEDMainActivityName", "iedmainactivityname"])
    c_epr = _first_col(df, ["EPRTRAnnexIMainActivity", "eprtranneximainactivity"])
    c_ied_ann = _first_col(df, ["IEDAnnexIMainActivity", "iedanneximainactivity"])
    if not all([c_inst, c_lon, c_lat, c_ctry]):
        print("F6_1: missing columns; skipping.")
        return _empty_standard()
    work = df.copy()
    work[c_lon] = pd.to_numeric(work[c_lon], errors="coerce")
    work[c_lat] = pd.to_numeric(work[c_lat], errors="coerce")
    work = work[work[c_lon].notna() & work[c_lat].notna()]
    if country_filter:
        work = work[work[c_ctry].astype(str).str.strip() == country_filter]
    work = _apply_bbox(work, c_lon, c_lat, bbox)
    text_cols = [c for c in [c_name, c_city, c_ied, c_epr, c_ied_ann] if c]
    work["_t"] = work.apply(
        lambda r: _concat_text([r.get(c, "") for c in text_cols]), axis=1
    )
    mask = work["_t"].str.lower().apply(
        lambda t: _matches_any(t, WASTE_GENERIC)
        or _matches_any(t, WASTEWATER)
        or _matches_any(t, LANDFILL)
        or _matches_any(t, INCINERATION)
        or _matches_any(t, CO_INCINERATION)
        or _matches_any(t, WTE_EXTRA)
    )
    work = work[mask]
    print(f"F6_1: {len(work)} installations after coord/country/waste-text filter")
    if work.empty:
        return _empty_standard()

    rows = []
    for _, r in work.iterrows():
        pid = _norm_id(r.get(c_parent, "")) if c_parent else ""
        iid = _norm_id(r.get(c_inst, ""))
        fid = pid if pid else iid
        y = pd.to_numeric(r.get(c_year, float("nan")), errors="coerce")
        blob = _concat_text(
            [r.get(c_name, ""), r.get(c_city, ""), r.get(c_ied, ""), r.get(c_epr, "")]
        )
        rows.append(
            {
                "source_file": "F6_1",
                "FacilityInspireId": fid,
                "InstallationInspireId": iid,
                "ParentFacilityInspireId": pid,
                "facility_name": str(r.get(c_name, "") or ""),
                "country": str(r.get(c_ctry, "")).strip(),
                "longitude": float(r[c_lon]),
                "latitude": float(r[c_lat]),
                "reporting_year_min": int(y) if pd.notna(y) else "",
                "reporting_year_max": int(y) if pd.notna(y) else "",
                "air_rows": 0,
                "air_releases_sum": 0.0,
                "pollutants_concat": "",
                "air_pollutants_json": "{}",
                "sector_codes_concat": "",
                "sector_names_concat": "",
                "annex_concat": str(r.get(c_epr, "") or ""),
                "waste_transfers_sum": 0.0,
                "ied_activity_concat": str(r.get(c_ied, "") or "")[:2000],
                "lcp_mw": float("nan"),
                "text_blob": blob,
                "from_f1_4": False,
                "from_f4_2": False,
                "from_f5_2": False,
                "from_f6_1": True,
                "from_f7_1": False,
            }
        )
    out = pd.DataFrame(rows)
    print(f"F6_1: {len(out)} installation rows kept")
    return out


def load_f7_1_wi_cowi(
    path: Path,
    *,
    country_filter: str | None,
    bbox: tuple[float, float, float, float] | None,
) -> pd.DataFrame:
    df = _read_table(path)
    if df is None or df.empty:
        return _empty_standard()
    c_pf = _first_col(df, ["Parent_FacilityInspireId", "parent_facilityinspireid"])
    c_lon = _first_col(df, ["Longitude"])
    c_lat = _first_col(df, ["Latitude"])
    c_ctry = _first_col(df, ["countryName", "CountryName"])
    c_year = _first_col(df, ["reportingYear"])
    c_part = _first_col(df, ["installationPartName", "InstallationPartName"])
    c_city = _first_col(df, ["City_Of_Facility", "City_of_Facility"])
    c_epr = _first_col(df, ["EPRTRAnnexIMainActivity"])
    c_ied = _first_col(df, ["IEDAnnexIMainActivity"])
    c_cap = _first_col(df, ["totalNominalCapacityAnyWaste"])
    if not all([c_pf, c_lon, c_lat, c_ctry]):
        print("F7_1: missing columns; skipping.")
        return _empty_standard()
    work = df.copy()
    work[c_pf] = work[c_pf].map(_norm_id)
    work = work[work[c_pf].astype(str).str.len() > 0]
    work[c_lon] = pd.to_numeric(work[c_lon], errors="coerce")
    work[c_lat] = pd.to_numeric(work[c_lat], errors="coerce")
    work = work[work[c_lon].notna() & work[c_lat].notna()]
    if country_filter:
        work = work[work[c_ctry].astype(str).str.strip() == country_filter]
    work = _apply_bbox(work, c_lon, c_lat, bbox)
    print(f"F7_1: {len(work)} rows after filter")
    if work.empty:
        return _empty_standard()

    rows_f7: list[dict[str, Any]] = []
    for pf, g in work.groupby(c_pf, sort=False):
        pf_s = _norm_id(pf)
        y = pd.to_numeric(g[c_year], errors="coerce") if c_year else pd.Series(dtype=float)
        cap = pd.to_numeric(g[c_cap], errors="coerce") if c_cap else pd.Series(dtype=float)
        blob = _concat_text(
            [
                g[c_part].iloc[0] if c_part else "",
                g[c_city].iloc[0] if c_city else "",
                g[c_epr].iloc[0] if c_epr else "",
                g[c_ied].iloc[0] if c_ied else "",
            ]
        )
        rows_f7.append(
            {
                "source_file": "F7_1",
                "FacilityInspireId": pf_s,
                "InstallationInspireId": "",
                "ParentFacilityInspireId": pf_s,
                "facility_name": str(g[c_part].iloc[0]) if c_part else "",
                "country": str(g[c_ctry].iloc[0]).strip(),
                "longitude": float(g[c_lon].median()),
                "latitude": float(g[c_lat].median()),
                "reporting_year_min": int(y.min()) if y.notna().any() else "",
                "reporting_year_max": int(y.max()) if y.notna().any() else "",
                "air_rows": 0,
                "air_releases_sum": 0.0,
                "pollutants_concat": "",
                "air_pollutants_json": "{}",
                "sector_codes_concat": "",
                "sector_names_concat": "waste incineration / co-incineration (F7_1)",
                "annex_concat": "|".join(
                    sorted({str(x) for x in g[c_epr].dropna().unique()})[:80]
                )
                if c_epr
                else "",
                "waste_transfers_sum": float(cap.sum()) if cap.notna().any() else 0.0,
                "ied_activity_concat": "|".join(
                    sorted({str(x) for x in g[c_ied].dropna().unique()})[:80]
                )
                if c_ied
                else "",
                "lcp_mw": float("nan"),
                "text_blob": blob,
                "from_f1_4": False,
                "from_f4_2": False,
                "from_f5_2": False,
                "from_f6_1": False,
                "from_f7_1": True,
            }
        )
    out = pd.DataFrame(rows_f7)
    print(f"F7_1: {len(out)} facilities aggregated")
    return out


def load_f5_2_lcp(
    path: Path,
    *,
    country_filter: str | None,
    bbox: tuple[float, float, float, float] | None,
) -> pd.DataFrame:
    df = _read_table(path)
    if df is None or df.empty:
        return _empty_standard()
    c_ft = _first_col(df, ["featureType", "featuretype"])
    c_unit = _first_col(df, ["unit", "Unit"])
    c_lcp = _first_col(df, ["LCPInspireId", "lcpinspireid"])
    c_lon = _first_col(df, ["Longitude"])
    c_lat = _first_col(df, ["Latitude"])
    c_ctry = _first_col(df, ["countryName", "CountryName"])
    c_year = _first_col(df, ["reportingYear"])
    c_name = _first_col(df, ["installationPartName"])
    c_city = _first_col(df, ["City_Of_Facility"])
    c_val = _first_col(df, ["featureValue", "featurevalue"])
    if not all([c_ft, c_unit, c_lcp, c_lon, c_lat, c_ctry]):
        print("F5_2: missing columns; skipping.")
        return _empty_standard()
    work = df[(df[c_ft].astype(str) == "LCPCharacteristics") & (df[c_unit].astype(str) == "MW")].copy()
    work[c_lon] = pd.to_numeric(work[c_lon], errors="coerce")
    work[c_lat] = pd.to_numeric(work[c_lat], errors="coerce")
    work = work[work[c_lon].notna() & work[c_lat].notna()]
    if country_filter:
        work = work[work[c_ctry].astype(str).str.strip() == country_filter]
    work = _apply_bbox(work, c_lon, c_lat, bbox)
    work["_t"] = work.apply(
        lambda r: _concat_text([r.get(c_name, ""), r.get(c_city, "")]), axis=1
    )
    mask = work["_t"].str.lower().apply(
        lambda t: _matches_any(t, WASTE_GENERIC)
        or _matches_any(t, INCINERATION)
        or _matches_any(t, CO_INCINERATION)
        or _matches_any(t, WTE_EXTRA)
    )
    work = work[mask]
    print(f"F5_2: {len(work)} LCPCharacteristics MW rows after waste-like name filter")
    if work.empty:
        return _empty_standard()

    rows_f5: list[dict[str, Any]] = []
    for lid, g in work.groupby(c_lcp, sort=False):
        lid_s = str(lid)
        y = pd.to_numeric(g[c_year], errors="coerce") if c_year else pd.Series(dtype=float)
        mw = pd.to_numeric(g[c_val], errors="coerce") if c_val else pd.Series(dtype=float)
        blob = _concat_text([g[c_name].iloc[0] if c_name else "", g[c_city].iloc[0] if c_city else ""])
        rows_f5.append(
            {
                "source_file": "F5_2",
                "FacilityInspireId": lid_s,
                "InstallationInspireId": "",
                "ParentFacilityInspireId": "",
                "facility_name": str(g[c_name].iloc[0]) if c_name else lid_s,
                "country": str(g[c_ctry].iloc[0]).strip(),
                "longitude": float(g[c_lon].iloc[0]),
                "latitude": float(g[c_lat].iloc[0]),
                "reporting_year_min": int(y.min()) if y.notna().any() else "",
                "reporting_year_max": int(y.max()) if y.notna().any() else "",
                "air_rows": 0,
                "air_releases_sum": 0.0,
                "pollutants_concat": "",
                "air_pollutants_json": "{}",
                "sector_codes_concat": "",
                "sector_names_concat": "LCP (waste-like filter)",
                "annex_concat": "",
                "waste_transfers_sum": 0.0,
                "ied_activity_concat": "",
                "lcp_mw": float(mw.max()) if mw.notna().any() else float("nan"),
                "text_blob": blob,
                "from_f1_4": False,
                "from_f4_2": False,
                "from_f5_2": True,
                "from_f6_1": False,
                "from_f7_1": False,
            }
        )
    out = pd.DataFrame(rows_f5)
    print(f"F5_2: {len(out)} LCP sites aggregated")
    return out


def classify_facility_row(row: pd.Series) -> tuple[str, dict[str, bool]]:
    """Return (primary_category, extra tag bools: has_* / is_* only; from_* read from row)."""
    t = (
        str(row.get("text_blob", ""))
        + " "
        + str(row.get("sector_names_concat", ""))
        + " "
        + str(row.get("annex_concat", ""))
        + " "
        + str(row.get("ied_activity_concat", ""))
    ).lower()
    f1 = bool(row.get("from_f1_4"))
    f4 = bool(row.get("from_f4_2"))
    f5 = bool(row.get("from_f5_2"))
    f6 = bool(row.get("from_f6_1"))
    f7 = bool(row.get("from_f7_1"))
    tags: dict[str, bool] = {
        "has_air_release": float(row.get("air_releases_sum", 0) or 0) > 0
        or int(row.get("air_rows", 0) or 0) > 0,
        "has_waste_transfer": float(row.get("waste_transfers_sum", 0) or 0) > 0,
        "is_ied_installation": f6,
        "is_incinerator": _matches_any(t, INCINERATION) and not _matches_any(t, CO_INCINERATION),
        "is_co_incinerator": _matches_any(t, CO_INCINERATION)
        or ("co-incineration" in t)
        or ("coincineration" in t),
        "is_wastewater": _matches_any(t, WASTEWATER),
        "is_landfill": _matches_any(t, LANDFILL),
        "is_wte": _matches_any(t, WTE_EXTRA)
        or _matches_any(t, ("waste to energy", "waste-to-energy"))
        or (f5 and _matches_any(t, INCINERATION + WTE_EXTRA)),
    }
    scodes = str(row.get("sector_codes_concat", "")).lower()
    s5 = "5" in scodes.split("|") or "5.0" in scodes

    active: set[str] = set()
    if f7:
        active.add("waste_incineration")
        if tags["is_co_incinerator"]:
            active.add("co_incineration")
    if f4:
        active.add("waste_transfer")
    if tags["is_landfill"]:
        active.add("landfill")
    if tags["is_wastewater"]:
        active.add("wastewater")
    if tags["is_wte"] or (f5 and (_matches_any(t, WTE_EXTRA) or _matches_any(t, INCINERATION))):
        active.add("waste_to_energy")
    if tags["is_co_incinerator"]:
        active.add("co_incineration")
    if tags["is_incinerator"] and "waste_incineration" not in active:
        active.add("waste_incineration")
    if f6 and (
        _matches_any(t, WASTE_GENERIC + WASTEWATER + LANDFILL + INCINERATION + CO_INCINERATION)
        or s5
    ):
        active.add("ied_waste_installation")
    if f1 and (
        tags["has_air_release"]
        and (
            s5
            or _matches_any(t, WASTE_GENERIC + WASTEWATER + LANDFILL + INCINERATION)
            or _matches_any(str(row.get("sector_names_concat", "")).lower(), WASTE_GENERIC)
        )
    ):
        active.add("waste_air_release")
    elif f1:
        active.add("other_potential_waste")

    if not active:
        if f4 or f7:
            active.add("waste_transfer" if f4 else "waste_incineration")
        elif _matches_any(t, WASTE_GENERIC):
            active.add("other_potential_waste")
        else:
            active.add("unclear_but_relevant")

    primary = "unclear_but_relevant"
    for cat in CATEGORY_PRIORITY:
        if cat in active:
            primary = cat
            break
    return primary, tags


def filter_merged_facilities_active_after_reporting_year(
    df: pd.DataFrame,
    *,
    cutoff_year: int | None = DEFAULT_EPRTR_REPORTING_YEAR_CUTOFF,
) -> pd.DataFrame:
    """
    Drop merged facilities whose ``reporting_year_max`` is missing or at or below
    ``cutoff_year`` (no reporting after that year). Keeps rows with ``reporting_year_max``
    strictly greater than ``cutoff_year``.
    """
    if df.empty or cutoff_year is None:
        return df
    if "reporting_year_max" not in df.columns:
        return df
    ym = pd.to_numeric(df["reporting_year_max"], errors="coerce")
    keep = ym > float(cutoff_year)
    n_before = len(df)
    n_after = int(keep.sum())
    if n_after < n_before:
        print(
            f"Reporting-year filter (reporting_year_max > {cutoff_year}): "
            f"kept {n_after} of {n_before} merged facilities "
            f"(dropped {n_before - n_after} inactive after {cutoff_year})."
        )
    return df.loc[keep].reset_index(drop=True)


def merge_facility_sources(
    parts: list[pd.DataFrame],
    *,
    name_match_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge facility rows on FacilityInspireId (non-empty), then propose duplicate candidates
    for same (country, rounded lon/lat) with different IDs or similar names.
    """
    raw = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    if raw.empty:
        return pd.DataFrame(), pd.DataFrame()

    raw["merge_key"] = raw["FacilityInspireId"].map(
        lambda x: _norm_id(x) if _norm_id(x) else ""
    )
    raw["merge_key"] = raw["merge_key"].where(raw["merge_key"].str.len() > 0, pd.NA)
    raw["rlon"] = raw["longitude"].map(lambda v: _round_coord(float(v), 4))
    raw["rlat"] = raw["latitude"].map(lambda v: _round_coord(float(v), 4))
    raw["norm_facility_name"] = raw["facility_name"].map(_norm_name)

    dup_rows: list[dict[str, Any]] = []
    master_rows: list[dict[str, Any]] = []

    def _merge_block(sub: pd.DataFrame, key_label: str) -> dict[str, Any]:
        base = sub.iloc[0].to_dict()
        base["source_file"] = "|".join(sorted({str(x) for x in sub["source_file"].unique()}))
        for _, r in sub.iloc[1:].iterrows():
            for c in STANDARD_COLS:
                if c == "source_file":
                    continue
                if c.startswith("from_"):
                    base[c] = bool(base.get(c)) or bool(r.get(c))
                elif c in ("air_rows",):
                    base[c] = int(base.get(c) or 0) + int(r.get(c) or 0)
                elif c in ("air_releases_sum", "waste_transfers_sum"):
                    base[c] = float(base.get(c) or 0) + float(r.get(c) or 0)
                elif c in ("lcp_mw",):
                    v0, v1 = base.get(c), r.get(c)
                    vals = [x for x in (v0, v1) if pd.notna(x) and math.isfinite(float(x))]
                    base[c] = max(vals) if vals else float("nan")
                elif c.endswith("_concat") or c in ("text_blob", "ied_activity_concat"):
                    parts_m = {str(base.get(c, "")), str(r.get(c, ""))}
                    parts_m.discard("")
                    base[c] = " || ".join(sorted(parts_m))
                elif c == "air_pollutants_json":
                    d0 = _parse_air_pollutants_json(base.get(c))
                    d1 = _parse_air_pollutants_json(r.get(c))
                    merged = dict(d0)
                    for k, v in d1.items():
                        merged[k] = merged.get(k, 0.0) + float(v)
                    base[c] = _dump_air_pollutants_json(merged)
                elif c in ("reporting_year_min", "reporting_year_max"):
                    ys = [base.get(c), r.get(c)]
                    nums = []
                    for y in ys:
                        try:
                            nums.append(int(y))
                        except (TypeError, ValueError):
                            pass
                    if nums:
                        base[c] = min(nums) if "min" in c else max(nums)
        base["_merge_key_used"] = key_label
        return base

    # Group 1: non-empty FacilityInspireId
    sub_with = raw[raw["merge_key"].notna()].copy()
    for key, sub in sub_with.groupby("merge_key", sort=False):
        if len(sub) > 1:
            dup_rows.append(
                {
                    "reason": "same_FacilityInspireId_multi_sources",
                    "FacilityInspireId": key,
                    "n_rows": len(sub),
                    "sources": "|".join(sorted(sub["source_file"].unique())),
                }
            )
        master_rows.append(_merge_block(sub, "FacilityInspireId"))

    # Group 2: rows without FacilityInspireId — use rounded lat/lon + country
    sub_wo = raw[raw["merge_key"].isna()].copy()
    if not sub_wo.empty:
        sub_wo["geo_key"] = (
            sub_wo["country"].astype(str) + "|" + sub_wo["rlon"].astype(str) + "|" + sub_wo["rlat"].astype(str)
        )
        for gkey, sub in sub_wo.groupby("geo_key", sort=False):
            if len(sub) > 1:
                dup_rows.append(
                    {
                        "reason": "same_country_rounded_latlon",
                        "geo_key": gkey,
                        "n_rows": len(sub),
                        "sources": "|".join(sorted(sub["source_file"].unique())),
                    }
                )
            master_rows.append(_merge_block(sub, "geo_rounded"))

    master = pd.DataFrame(master_rows)
    if master.empty:
        return master, pd.DataFrame(dup_rows)

    # Cross-key fuzzy duplicate candidates (different FacilityInspireId, close coords, similar name)
    m = master.copy()
    m["norm_facility_name"] = m["facility_name"].map(_norm_name)
    for i in range(len(m)):
        for j in range(i + 1, len(m)):
            a, b = m.iloc[i], m.iloc[j]
            if a.get("FacilityInspireId") == b.get("FacilityInspireId"):
                continue
            if str(a.get("country", "")).strip() != str(b.get("country", "")).strip():
                continue
            dlon = abs(float(a["longitude"]) - float(b["longitude"]))
            dlat = abs(float(a["latitude"]) - float(b["latitude"]))
            if dlon > 0.02 or dlat > 0.02:
                continue
            sim = _name_similarity(
                str(a.get("norm_facility_name", "")),
                str(b.get("norm_facility_name", "")),
            )
            if sim >= name_match_threshold:
                dup_rows.append(
                    {
                        "reason": "fuzzy_name_close_coords",
                        "facility_a": a.get("FacilityInspireId"),
                        "facility_b": b.get("FacilityInspireId"),
                        "name_similarity": round(sim, 4),
                        "country": a.get("country"),
                    }
                )

    return master, pd.DataFrame(dup_rows)


def build_master_table(master: pd.DataFrame) -> pd.DataFrame:
    if master.empty:
        return master
    primaries: list[str] = []
    tag_dicts: list[dict[str, bool]] = []
    for _, row in master.iterrows():
        p, tags = classify_facility_row(row)
        primaries.append(p)
        tag_dicts.append(tags)
    tag_df = pd.DataFrame(tag_dicts)
    out = pd.concat([master.reset_index(drop=True), tag_df], axis=1)
    out.insert(0, "master_id", [f"M{i+1:06d}" for i in range(len(out))])
    out["primary_category"] = primaries
    out["category_tags"] = out["primary_category"]
    out["normalized_name"] = out["facility_name"].map(_norm_name)
    cols = [
        "master_id",
        "FacilityInspireId",
        "longitude",
        "latitude",
        "country",
        "facility_name",
        "normalized_name",
        "primary_category",
        "category_tags",
    ]
    for c in cols:
        if c not in out.columns:
            out[c] = ""
    return out


def export_summary_by_category(master: pd.DataFrame, path: Path) -> None:
    if master.empty:
        pd.DataFrame({"primary_category": [], "count": []}).to_csv(path, index=False)
        return
    g = master.groupby("primary_category", as_index=False).size().rename(columns={"size": "count"})
    src_cols = [c for c in master.columns if c.startswith("from_")]
    for c in src_cols:
        g[c + "_facilities"] = [
            int(master.loc[master["primary_category"] == cat, c].fillna(False).astype(bool).sum())
            for cat in g["primary_category"]
        ]
    g.to_csv(path, index=False)


def load_cams_j_points(
    nc_path: Path,
    *,
    country_iso3: str | None,
    all_countries: bool,
    bbox: tuple[float, float, float, float] | None,
    pollutant: str | None = None,
) -> pd.DataFrame:
    import xarray as xr

    cj = _get_cams_j_map_module()
    ds = xr.open_dataset(nc_path)
    try:
        emis = np.asarray(ds["emission_category_index"].values).ravel().astype(np.int64)
        st = np.asarray(ds["source_type_index"].values).ravel().astype(np.int64)
        ci = np.asarray(ds["country_index"].values).ravel().astype(np.int64)
        lon = np.asarray(ds["longitude_source"].values).ravel().astype(np.float64)
        lat = np.asarray(ds["latitude_source"].values).ravel().astype(np.float64)
        m = (emis == 13) & (st == 2)
        if not all_countries and country_iso3:
            raw = ds["country_id"].values
            codes: list[str] = []
            for x in raw:
                if isinstance(x, bytes):
                    codes.append(x.decode("utf-8", "replace").strip())
                else:
                    codes.append(str(x).strip())
            try:
                c1 = codes.index(country_iso3.strip().upper()) + 1
            except ValueError:
                raise SystemExit(f"Country {country_iso3!r} not in CAMS country_id.") from None
            m = m & (ci == c1)
        if bbox is not None:
            lon0, lat0, lon1, lat1 = bbox
            m = m & (lon >= lon0) & (lon <= lon1) & (lat >= lat0) & (lat <= lat1)
        idx = np.flatnonzero(m)
        pollutants = cj._pollutant_vars(ds)
        units: dict[str, str] = {}
        for name in pollutants:
            v = ds[name]
            u = v.attrs.get("units", "kg/year")
            lu = v.attrs.get("long_units", "")
            units[name] = f"{u}" + (f" ({lu})" if lu else "")
        emis_ar = np.asarray(ds["emission_category_index"].values).ravel().astype(np.int64)
        map_pol = ""
        pol_var: str | None = None
        if pollutant:
            want = str(pollutant).strip().lower()
            pmap = {str(v).lower(): str(v) for v in ds.data_vars}
            if want not in pmap:
                raise SystemExit(
                    f"CAMS NetCDF has no pollutant variable matching {pollutant!r}. "
                    f"Examples: {', '.join(list(pmap.keys())[:16])}"
                )
            pol_var = pmap[want]
        pol_arr = (
            np.asarray(ds[pol_var].values).ravel().astype(np.float64) if pol_var else None
        )
        rows: list[dict[str, Any]] = []
        for i in idx:
            ii = int(i)
            pop = cj._point_popup_html(
                ds,
                ii,
                pollutants,
                map_pol,
                units=units,
                country_idx=ci,
                emis_idx=emis_ar,
                st_idx=st,
            )
            row: dict[str, Any] = {
                "cams_lon": float(lon[ii]),
                "cams_lat": float(lat[ii]),
                "emission_category_index": int(emis[ii]),
                "source_type_index": int(st[ii]),
                "country_index": int(ci[ii]),
                "nc_source_index": ii,
                "cams_source_id": f"CAMS_J_{ii:07d}",
                "popup_html": pop,
            }
            if pol_arr is not None:
                row["cams_pollutant_kg"] = float(pol_arr[ii]) if ii < pol_arr.size else float("nan")
            rows.append(row)
        return pd.DataFrame(rows)
    finally:
        ds.close()


def _nearest_pairs(
    facilities: pd.DataFrame,
    cams: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        print("Warning: scipy not installed; skipping nearest-neighbor CSV exports.")
        return pd.DataFrame(), pd.DataFrame()
    if facilities.empty or cams.empty:
        return pd.DataFrame(), pd.DataFrame()
    xy_f = np.asarray(
        list(zip(facilities["longitude"].astype(float), facilities["latitude"].astype(float)))
    )
    xy_c = np.asarray(list(zip(cams["cams_lon"].astype(float), cams["cams_lat"].astype(float))))
    tree_c = cKDTree(xy_c)
    dist_fc, idx_fc = tree_c.query(xy_f, k=1)
    df_fc = facilities[["master_id", "FacilityInspireId", "facility_name", "longitude", "latitude"]].copy()
    df_fc["nearest_cams_idx"] = idx_fc
    df_fc["dist_deg"] = dist_fc
    df_fc["nearest_cams_lon"] = cams["cams_lon"].values[idx_fc]
    df_fc["nearest_cams_lat"] = cams["cams_lat"].values[idx_fc]

    tree_f = cKDTree(xy_f)
    dist_cf, idx_cf = tree_f.query(xy_c, k=1)
    df_cf = cams.copy()
    df_cf["nearest_master_id"] = facilities["master_id"].values[idx_cf]
    df_cf["dist_deg"] = dist_cf
    return df_fc, df_cf


def _uwwtd_country_match_tokens(iso3_or_country: str, display_country: str | None) -> set[str]:
    """Lowercase tokens for filtering UWWTD rows by country-like columns."""
    out: set[str] = set()
    u = iso3_or_country.strip()
    if display_country:
        out.add(display_country.strip().casefold())
    out.add(u.casefold())
    if len(u) == 3:
        out.add(u.upper().casefold())
        if u.upper() == "GRC":
            out.update({"el", "gr", "greece", "hellas"})
    return out


def load_uwwtd_treatment_plants_gpkg(
    path: Path,
    *,
    country_filter: str | None,
    country_iso3: str,
    bbox: tuple[float, float, float, float] | None,
) -> pd.DataFrame:
    """
    Load point coordinates from ``UWWTD_TreatmentPlants.gpkg`` (or compatible path).
    Requires ``geopandas``. Returns columns ``uwwtd_lat``, ``uwwtd_lon``, ``uwwtd_popup_html``.
    """
    if not path.is_file():
        return pd.DataFrame()
    try:
        import geopandas as gpd
    except ImportError:
        print(
            "Warning: geopandas not installed; cannot load UWWTD gpkg. "
            "Install with: pip install geopandas pyogrio"
        )
        return pd.DataFrame()
    try:
        gdf = gpd.read_file(path)
    except Exception as exc:
        print(f"Warning: could not read UWWTD GeoPackage ({path}): {exc}")
        return pd.DataFrame()
    if gdf.empty or gdf.geometry.isna().all():
        return pd.DataFrame()
    if gdf.crs is not None:
        try:
            epsg = gdf.crs.to_epsg()
            if epsg is not None and epsg != 4326:
                gdf = gdf.to_crs(4326)
        except Exception:
            try:
                gdf = gdf.to_crs("EPSG:4326")
            except Exception:
                pass

    geom = gdf.geometry
    try:
        centroids = geom.centroid
        gdf = gdf.assign(uwwtd_lon=centroids.x, uwwtd_lat=centroids.y)
    except Exception:
        return pd.DataFrame()

    gdf = gdf[np.isfinite(gdf["uwwtd_lon"]) & np.isfinite(gdf["uwwtd_lat"])]
    if gdf.empty:
        return pd.DataFrame()

    if country_filter:
        tokens = _uwwtd_country_match_tokens(country_iso3, country_filter)
        preferred = (
            "MS",
            "memberState",
            "MemberState",
            "member_state",
            "MS_CD",
            "MSCode",
            "isoCountryCode",
            "CountryName",
            "countryName",
            "COUNTRY",
            "Country",
        )
        country_cols = [c for c in preferred if c in gdf.columns]
        if not country_cols:
            country_cols = [
                c
                for c in gdf.columns
                if c != "geometry"
                and any(
                    k in c.casefold()
                    for k in ("country", "member", "ms", "state", "nation", "cca", "eu")
                )
            ][:12]
        if country_cols:
            mask = np.zeros(len(gdf), dtype=bool)
            for c in country_cols:
                s = gdf[c].astype(str).str.strip().str.casefold()
                for t in tokens:
                    mask |= s == t
            gdf = gdf[mask]
            print(f"UWWTD gpkg: {len(gdf)} features after country filter ({country_filter}).")
        else:
            print(
                "UWWTD gpkg: no obvious country column found; skipping country filter "
                "(use --bbox or inspect the gpkg field names)."
            )

    if bbox is not None:
        lon0, lat0, lon1, lat1 = bbox
        gdf = gdf[
            (gdf["uwwtd_lon"] >= lon0)
            & (gdf["uwwtd_lon"] <= lon1)
            & (gdf["uwwtd_lat"] >= lat0)
            & (gdf["uwwtd_lat"] <= lat1)
        ]
        print(f"UWWTD gpkg: {len(gdf)} features after bbox filter.")

    rows: list[str] = []
    for _, row in gdf.iterrows():
        parts = ["<div style='min-width:220px;max-width:420px;font-size:12px'>"]
        parts.append("<b>UWWTD treatment plant</b> (EEA gpkg)<br/>")
        for c in gdf.columns:
            if c in ("geometry", "uwwtd_lon", "uwwtd_lat"):
                continue
            v = row.get(c)
            if pd.isna(v) or str(v).strip() == "":
                continue
            parts.append(f"<b>{html.escape(str(c))}</b> {html.escape(str(v)[:500])}<br/>")
        parts.append(
            f"<b>Coords</b> {float(row['uwwtd_lon']):.5f}, {float(row['uwwtd_lat']):.5f}<br/></div>"
        )
        rows.append("".join(parts))
    gdf = gdf.reset_index(drop=True)
    gdf["uwwtd_popup_html"] = rows
    return gdf


def make_folium_map(
    master: pd.DataFrame,
    cams: pd.DataFrame | None,
    *,
    out_html: Path,
    map_center: tuple[float, float] | None,
    zoom_start: int,
    proxy_bundle: dict[str, Any] | None = None,
    uwwtd: pd.DataFrame | None = None,
) -> None:
    import folium
    from folium import plugins

    import waste_cams_proxy_match as wpm_proxy

    has_uwwtd = uwwtd is not None and not uwwtd.empty
    if master.empty and (cams is None or cams.empty) and not has_uwwtd:
        print("No data to map; skipping HTML.")
        return

    lats, lons = [], []
    if not master.empty:
        lats.extend(master["latitude"].astype(float).tolist())
        lons.extend(master["longitude"].astype(float).tolist())
    if cams is not None and not cams.empty:
        lats.extend(cams["cams_lat"].astype(float).tolist())
        lons.extend(cams["cams_lon"].astype(float).tolist())
    if has_uwwtd:
        lats.extend(uwwtd["uwwtd_lat"].astype(float).tolist())
        lons.extend(uwwtd["uwwtd_lon"].astype(float).tolist())
    if map_center is not None:
        clat, clon = map_center
    else:
        clat = float(np.mean(lats)) if lats else 50.0
        clon = float(np.mean(lons)) if lons else 10.0

    fmap = folium.Map(
        location=[clat, clon],
        zoom_start=zoom_start,
        tiles="CartoDB positron",
        attr=(
            '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> '
            'contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
        ),
    )
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png",
        attr=(
            '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> '
            'contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
        ),
        name="Street (Carto Voyager)",
        subdomains="abcd",
        max_zoom=20,
        control=True,
    ).add_to(fmap)
    folium.TileLayer("CartoDB dark_matter", name="Map (dark)", control=True).add_to(fmap)
    folium.TileLayer(
        "OpenStreetMap",
        name="OpenStreetMap",
        control=True,
        max_zoom=19,
    ).add_to(fmap)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr=(
            "Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics, "
            "and the GIS User Community"
        ),
        name="Satellite (Esri)",
        overlay=False,
        control=True,
    ).add_to(fmap)

    legend_html = (
        "<div style='position: fixed; bottom: 28px; left: 28px; width: 320px; height: auto; "
        "z-index:9999; font-size:12px; background: white; border:2px solid grey; padding: 8px;'>"
        "<b>Legend</b><br/>"
        + "<br/>".join(
            f"<span style='color:{c};'>&#9679;</span> {k.replace('_', ' ')}"
            for k, c in CATEGORY_COLOR.items()
            if k != "cams_gnfr_j"
        )
        + f"<br/><span style='color:{CATEGORY_COLOR['cams_gnfr_j']};'>&#9679;</span> CAMS GNFR J points"
        + (
            f"<br/><span style='color:{UWWTD_GPKG_POINT_COLOR};'>&#9679;</span> "
            "UWWTD treatment plants (EEA gpkg)"
            if has_uwwtd
            else ""
        )
        + "</div>"
    )
    fmap.get_root().html.add_child(folium.Element(legend_html))

    use_proxy = proxy_bundle is not None
    fcounts: dict[str, int] = {}
    match_by_sid: pd.DataFrame | None = None
    if use_proxy:
        fcounts = proxy_bundle.get("facility_match_counts") or {}
        mt = proxy_bundle.get("match_table")
        if mt is not None and not mt.empty and "cams_source_id" in mt.columns:
            match_by_sid = mt.drop_duplicates(subset=["cams_source_id"], keep="first").set_index(
                "cams_source_id", drop=True
            )

    def _facility_marker(
        r: pd.Series,
        *,
        fg: Any,
        color: str,
        radius: float,
        fill_opacity: float,
        edge: str = "#222222",
    ) -> None:
        cat = str(r.get("primary_category", "unclear_but_relevant"))
        srcs = [
            x
            for x, c in [
                ("F1_4", r.get("from_f1_4")),
                ("F4_2", r.get("from_f4_2")),
                ("F5_2", r.get("from_f5_2")),
                ("F6_1", r.get("from_f6_1")),
                ("F7_1", r.get("from_f7_1")),
            ]
            if c
        ]
        pop = (
            "<div style='min-width:240px;max-width:480px;font-size:12px'>"
            f"<b>{html.escape(str(r.get('facility_name','')))}</b><br/>"
            f"<b>Category</b> {html.escape(cat)}<br/>"
            f"<b>Sources</b> {html.escape(', '.join(srcs))}<br/>"
            f"<b>FacilityInspireId</b><br/><span style='word-break:break-all'>"
            f"{html.escape(str(r.get('FacilityInspireId','')))}</span><br/>"
            f"<b>master_id</b> {html.escape(str(r.get('master_id','')))}<br/>"
            f"<b>Coords</b> {float(r['longitude']):.5f}, {float(r['latitude']):.5f}<br/>"
            f"<b>Years</b> {html.escape(str(r.get('reporting_year_min','')))}–"
            f"{html.escape(str(r.get('reporting_year_max','')))}<br/>"
            f"<b>Air releases (sum)</b> {html.escape(str(r.get('air_releases_sum','')))}<br/>"
            f"<b>Waste transfers (sum)</b> {html.escape(str(r.get('waste_transfers_sum','')))}<br/>"
            f"<b>Sectors</b> {html.escape(str(r.get('sector_names_concat',''))[:400])}<br/>"
            f"<b>Annex / IED</b> {html.escape(str(r.get('annex_concat',''))[:300])}"
            f"{_air_pollutants_table_html(r.get('air_pollutants_json') or '{}')}"
            "</div>"
        )
        mid = str(r.get("master_id", ""))
        if use_proxy and mid:
            pop = wpm_proxy.facility_popup_append_match_count(pop, mid, fcounts)
        folium.CircleMarker(
            location=[float(r["latitude"]), float(r["longitude"])],
            radius=radius,
            color=edge,
            weight=1,
            fill=True,
            fill_color=color,
            fill_opacity=fill_opacity,
            popup=folium.Popup(pop, max_width=500),
        ).add_to(fg)

    if use_proxy:
        elig_set = set(wpm_proxy.ELIGIBLE_CATEGORIES)
        groups_elig: dict[str, Any] = {}
        for cat in wpm_proxy.ELIGIBLE_CATEGORIES:
            groups_elig[cat] = folium.FeatureGroup(
                name=f"Facilities (proxy candidates): {cat.replace('_', ' ')}",
                show=True,
            )
        fg_muted = folium.FeatureGroup(
            name="Facilities (not used for proxy matching)",
            show=True,
        )
        for _, r in master.iterrows():
            cat = str(r.get("primary_category", "unclear_but_relevant"))
            color = CATEGORY_COLOR.get(cat, "#999999")
            radius = 6.0
            if r.get("is_incinerator") or r.get("from_f7_1"):
                radius = 8.5
            if r.get("is_wte") or r.get("from_f5_2"):
                radius = 9.0
            if cat in elig_set:
                _facility_marker(
                    r,
                    fg=groups_elig[cat],
                    color=color,
                    radius=radius,
                    fill_opacity=0.88,
                )
            else:
                _facility_marker(
                    r,
                    fg=fg_muted,
                    color=color,
                    radius=max(3.5, radius * 0.72),
                    fill_opacity=0.38,
                    edge="#999999",
                )
        for fg in list(groups_elig.values()) + [fg_muted]:
            fg.add_to(fmap)
    else:
        groups: dict[str, Any] = {}
        for cat in CATEGORY_PRIORITY:
            groups[cat] = folium.FeatureGroup(name=f"Facilities: {cat}", show=True)
        groups["other"] = folium.FeatureGroup(name="Facilities: other / unclear", show=True)
        for _, r in master.iterrows():
            cat = str(r.get("primary_category", "unclear_but_relevant"))
            fg = groups.get(cat) or groups["other"]
            color = CATEGORY_COLOR.get(cat, "#999999")
            radius = 6.0
            if r.get("is_incinerator") or r.get("from_f7_1"):
                radius = 8.5
            if r.get("is_wte") or r.get("from_f5_2"):
                radius = 9.0
            _facility_marker(r, fg=fg, color=color, radius=radius, fill_opacity=0.85)
        for fg in groups.values():
            fg.add_to(fmap)

    if has_uwwtd:
        fg_uw = folium.FeatureGroup(name="UWWTD treatment plants (EEA)", show=True)
        for _, ur in uwwtd.iterrows():
            pop_u = str(ur.get("uwwtd_popup_html", "")).strip()
            if not pop_u:
                pop_u = "<div style='font-size:12px'>UWWTD treatment plant</div>"
            folium.CircleMarker(
                location=[float(ur["uwwtd_lat"]), float(ur["uwwtd_lon"])],
                radius=5,
                color="#005a80",
                weight=1,
                fill=True,
                fill_color=UWWTD_GPKG_POINT_COLOR,
                fill_opacity=0.88,
                popup=folium.Popup(pop_u, max_width=480),
            ).add_to(fg_uw)
        fg_uw.add_to(fmap)

    if cams is not None and not cams.empty:
        fg_c = folium.FeatureGroup(name="CAMS GNFR J (points)", show=False)
        for _, r in cams.iterrows():
            base = str(r.get("popup_html", "")).strip()
            if not base:
                base = "<div style='min-width:200px;font-size:12px'>CAMS GNFR J waste point</div>"
            sid = str(r.get("cams_source_id", ""))
            mrow = None
            if match_by_sid is not None and sid and sid in match_by_sid.index:
                mrow = match_by_sid.loc[sid]
                if isinstance(mrow, pd.DataFrame):
                    mrow = mrow.iloc[0]
            pop_c = (
                wpm_proxy.cams_popup_html_with_match(base, mrow)
                if use_proxy
                else base
            )
            folium.CircleMarker(
                location=[float(r["cams_lat"]), float(r["cams_lon"])],
                radius=4,
                color="#333333",
                weight=1,
                fill=True,
                fill_color=CATEGORY_COLOR["cams_gnfr_j"],
                fill_opacity=0.9,
                popup=folium.Popup(pop_c, max_width=520),
            ).add_to(fg_c)
        fg_c.add_to(fmap)

        if use_proxy and proxy_bundle.get("proxy_df") is not None:
            pdf = proxy_bundle["proxy_df"]
            if not pdf.empty:
                fg_px = folium.FeatureGroup(
                    name="Proxy point locations (CAMS relocated to facility)",
                    show=False,
                )
                for _, pr in pdf.iterrows():
                    if str(pr.get("match_status")) != "matched":
                        continue
                    folium.CircleMarker(
                        location=[float(pr["proxy_lat"]), float(pr["proxy_lon"])],
                        radius=5,
                        color="#3f007d",
                        weight=2,
                        fill=True,
                        fill_color="#b794f6",
                        fill_opacity=0.95,
                        popup=folium.Popup(
                            "<div style='font-size:12px'><b>Relocated proxy</b><br/>"
                            f"cams_source_id {html.escape(str(pr.get('cams_source_id','')))}<br/>"
                            f"emission kg/yr {html.escape(str(pr.get('emission_kg','')))}</div>",
                            max_width=280,
                        ),
                    ).add_to(fg_px)
                fg_px.add_to(fmap)

        if use_proxy and proxy_bundle.get("match_table") is not None:
            pool = proxy_bundle.get("facility_candidate_pool")
            if pool is None or pool.empty:
                pool = master
            wpm_proxy.add_match_lines_to_map(fmap, proxy_bundle["match_table"], pool)

    plugins.Fullscreen().add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out_html))
    print(f"Wrote map: {out_html}")


def export_outputs(
    master: pd.DataFrame,
    raw_union: pd.DataFrame,
    dup_cand: pd.DataFrame,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    master.to_csv(out_dir / "waste_facilities_master.csv", index=False)
    raw_union.to_csv(out_dir / "waste_facilities_raw_union.csv", index=False)
    dup_cand.to_csv(out_dir / "waste_facilities_duplicate_candidates.csv", index=False)
    export_summary_by_category(master, out_dir / "waste_facilities_summary_by_category.csv")


def main() -> None:
    root = _project_root()
    ap = argparse.ArgumentParser(
        description="Unified waste-relevant facility inventory from EEA CSVs + optional CAMS GNFR J map."
    )
    ap.add_argument("--cams-nc", type=Path, default=None, help="CAMS NetCDF (optional)")
    ap.add_argument("--country", default="GRC", help="ISO3 country filter (default GRC)")
    ap.add_argument("--all-countries", action="store_true", help="Do not filter PRTR/CAMS by country")
    ap.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("MINLON", "MINLAT", "MAXLON", "MAXLAT"),
    )
    ap.add_argument(
        "--f1-4-csv",
        type=Path,
        default=root / "data" / "E_PRTR" / "eptr_csv" / "F1_4_Air_Releases_Facilities.csv",
    )
    ap.add_argument(
        "--f4-2-csv",
        type=Path,
        default=root / "data" / "E_PRTR" / "eptr_csv" / "F4_2_WasteTransfers_Facilities.csv",
    )
    ap.add_argument(
        "--f5-2-csv",
        type=Path,
        default=root / "data" / "E_PRTR" / "eptr_csv" / "F5_2_LCP_Energy_Emissions.csv",
    )
    ap.add_argument(
        "--f6-1-csv",
        type=Path,
        default=root / "data" / "E_PRTR" / "eptr_csv" / "F6_1_IED_Installations.csv",
    )
    ap.add_argument(
        "--f7-1-csv",
        type=Path,
        default=root / "data" / "E_PRTR" / "eptr_csv" / "F7_1_IED_WI_coWI.csv",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=root / "Waste" / "outputs" / "waste_facility_inventory",
    )
    ap.add_argument(
        "--output-map",
        type=Path,
        default=None,
        help="HTML map path (default: <output-dir>/waste_facilities_map.html)",
    )
    ap.add_argument("--include-cams", action="store_true", help="Overlay CAMS GNFR J points (needs --cams-nc)")
    ap.add_argument("--no-cams", action="store_true", help="Do not load or plot CAMS")
    ap.add_argument(
        "--cams-facility-proxy",
        action="store_true",
        help=(
            "Run CAMS↔facility empirical proxy matching, export CSVs + Waste_pointsource.tif "
            "(needs --include-cams and rasterio). Extends the map with match lines and proxy points."
        ),
    )
    ap.add_argument(
        "--proxy-pollutant",
        default="nmvoc",
        help="NetCDF variable name (case-insensitive) for emissions in proxy GeoTIFF (default nmvoc).",
    )
    ap.add_argument(
        "--proxy-pixel-deg",
        type=float,
        default=0.01,
        help="GeoTIFF pixel size in degrees for Waste_pointsource.tif (default 0.01).",
    )
    ap.add_argument(
        "--proxy-distance-scale",
        type=float,
        default=1.0,
        help=(
            "Multiply all per-class max matching distances (km) in waste_cams_proxy_match.MAX_DISTANCE_KM; "
            "e.g. 1.5 widens search radii by 50 percent. Scoring distance_score uses the same scaled cap (default 1.0)."
        ),
    )
    ap.add_argument(
        "--proxy-tif-name",
        default="Waste_pointsource.tif",
        help="Output GeoTIFF filename inside --output-dir (default Waste_pointsource.tif).",
    )
    ap.add_argument("--no-map", action="store_true", help="Skip Folium HTML")
    ap.add_argument(
        "--uwwtd-gpkg",
        type=Path,
        default=None,
        help=(
            "GeoPackage of EEA UWWTD treatment plant locations for the map "
            "(default: <project>/data/Waste/UWWTD_TreatmentPlants.gpkg). Requires geopandas."
        ),
    )
    ap.add_argument(
        "--no-uwwtd-layer",
        action="store_true",
        help="Do not load or plot the UWWTD treatment plant GeoPackage layer.",
    )
    ap.add_argument("--map-center", type=float, nargs=2, metavar=("LAT", "LON"))
    ap.add_argument("--map-zoom", type=int, default=7)
    ap.add_argument(
        "--name-match-threshold",
        type=float,
        default=0.82,
        help="Fuzzy name similarity [0-1] for duplicate candidate logging",
    )
    ap.add_argument(
        "--eprtr-reporting-cutoff-year",
        type=int,
        default=DEFAULT_EPRTR_REPORTING_YEAR_CUTOFF,
        metavar="Y",
        help=(
            "After merge, drop facilities whose reporting_year_max is at or below Y "
            "(no E-PRTR-style reporting after Y). Default "
            f"{DEFAULT_EPRTR_REPORTING_YEAR_CUTOFF}."
        ),
    )
    ap.add_argument(
        "--no-eprtr-reporting-year-filter",
        action="store_true",
        help="Keep all merged facilities regardless of reporting_year_max.",
    )
    args = ap.parse_args()

    out_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    bbox = tuple(args.bbox) if args.bbox else None
    c_filter = _country_filter_value(args.country, args.all_countries)

    paths = {
        "F1_4": args.f1_4_csv if args.f1_4_csv.is_absolute() else root / args.f1_4_csv,
        "F4_2": args.f4_2_csv if args.f4_2_csv.is_absolute() else root / args.f4_2_csv,
        "F5_2": args.f5_2_csv if args.f5_2_csv.is_absolute() else root / args.f5_2_csv,
        "F6_1": args.f6_1_csv if args.f6_1_csv.is_absolute() else root / args.f6_1_csv,
        "F7_1": args.f7_1_csv if args.f7_1_csv.is_absolute() else root / args.f7_1_csv,
    }
    for label, p in paths.items():
        print(f"Input {label}: {p}  ({'ok' if p.is_file() else 'MISSING'})")

    parts = [
        load_f1_4_facilities(paths["F1_4"], country_filter=c_filter, bbox=bbox),
        load_f4_2_facilities(paths["F4_2"], country_filter=c_filter, bbox=bbox),
        load_f6_1_installations(paths["F6_1"], country_filter=c_filter, bbox=bbox),
        load_f7_1_wi_cowi(paths["F7_1"], country_filter=c_filter, bbox=bbox),
        load_f5_2_lcp(paths["F5_2"], country_filter=c_filter, bbox=bbox),
    ]
    raw_union = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    print(f"Raw union rows (pre-merge facility rows): {len(raw_union)}")

    merged_df, dup_cand = merge_facility_sources(
        parts, name_match_threshold=args.name_match_threshold
    )
    ycut = None if args.no_eprtr_reporting_year_filter else int(args.eprtr_reporting_cutoff_year)
    merged_df = filter_merged_facilities_active_after_reporting_year(
        merged_df, cutoff_year=ycut
    )
    master = build_master_table(merged_df)
    print(f"Master facilities: {len(master)}")
    export_outputs(master, raw_union, dup_cand, out_dir)

    uw_path = (
        args.uwwtd_gpkg
        if args.uwwtd_gpkg is not None
        else root / "data" / "Waste" / "UWWTD_TreatmentPlants.gpkg"
    )
    uw_path = uw_path if uw_path.is_absolute() else root / uw_path
    uwwtd_df: pd.DataFrame | None = None
    if uw_path.is_file() and (not args.no_uwwtd_layer or args.cams_facility_proxy):
        uwwtd_df = load_uwwtd_treatment_plants_gpkg(
            uw_path,
            country_filter=c_filter,
            country_iso3=str(args.country),
            bbox=bbox,
        )
        if uwwtd_df.empty:
            print(f"UWWTD: no points from {uw_path} (after country/bbox filters).")
        else:
            print(f"UWWTD treatment plants loaded: {len(uwwtd_df)} ({uw_path.name})")
    elif not args.no_uwwtd_layer or args.cams_facility_proxy:
        print(f"UWWTD: GeoPackage not found ({uw_path}); map layer / proxy candidates omitted.")

    cams_df: pd.DataFrame | None = None
    use_cams = bool(args.include_cams) and not args.no_cams
    proxy_bundle: dict[str, Any] | None = None
    if use_cams:
        nc = args.cams_nc
        if nc is None:
            nc = (
                root
                / "data"
                / "given_CAMS"
                / "CAMS-REG-ANT_v8.1_TNO_ftp"
                / "netcdf"
                / "CAMS-REG-v8_1_emissions_year2019.nc"
            )
        else:
            nc = nc if nc.is_absolute() else root / nc
        if not nc.is_file():
            print(f"Warning: CAMS NetCDF not found ({nc}); skipping CAMS overlay and NN exports.")
        else:
            pol_kw: str | None = None
            if args.cams_facility_proxy:
                pol_kw = str(args.proxy_pollutant).strip().lower()
            cams_df = load_cams_j_points(
                nc,
                country_iso3=args.country,
                all_countries=args.all_countries,
                bbox=bbox,
                pollutant=pol_kw,
            )
            print(f"CAMS GNFR J points: {len(cams_df)}")
            df_fc, df_cf = _nearest_pairs(master, cams_df)
            if not df_fc.empty:
                df_fc.to_csv(out_dir / "waste_facility_nearest_cams.csv", index=False)
            if not df_cf.empty:
                df_cf.to_csv(out_dir / "waste_cams_nearest_facility.csv", index=False)

            if args.cams_facility_proxy:
                import waste_cams_proxy_match as wpm

                if cams_df.empty or "cams_pollutant_kg" not in cams_df.columns:
                    print("Warning: --cams-facility-proxy needs CAMS points with cams_pollutant_kg; skipping proxy.")
                else:
                    proxy_bundle = wpm.run_proxy_workflow(
                        master,
                        cams_df,
                        out_dir=out_dir,
                        pixel_size_deg=float(args.proxy_pixel_deg),
                        tif_filename=str(args.proxy_tif_name),
                        distance_scale=float(args.proxy_distance_scale),
                        uwwtd_points=uwwtd_df if uwwtd_df is not None and not uwwtd_df.empty else None,
                    )

    if not args.no_map:
        out_map = args.output_map
        if out_map is None:
            out_map = out_dir / "waste_facilities_map.html"
        else:
            out_map = out_map if out_map.is_absolute() else root / out_map
        mc = None
        if args.map_center is not None:
            mc = (float(args.map_center[0]), float(args.map_center[1]))
        make_folium_map(
            master,
            cams_df if use_cams else None,
            out_html=out_map,
            map_center=mc,
            zoom_start=int(args.map_zoom),
            proxy_bundle=proxy_bundle,
            uwwtd=uwwtd_df
            if (not args.no_uwwtd_layer and uwwtd_df is not None and not uwwtd_df.empty)
            else None,
        )

    print(f"Done. Outputs in: {out_dir}")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
