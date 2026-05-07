#!/usr/bin/env python3
"""
Interactive Folium map of CAMS-REG-ANT GNFR J (Waste) for Greece.

Area sources (source_type_index == 1) are drawn as grid-cell rectangles; GNFR **J point**
sources (``source_type_index == 2``) are circle markers (CAMS-REG). When the default
E-PRTR CSV exists, ``--all-countries`` adds an **E-PRTR F1_4** facility layer on the **same**
map (toggle layers in the control). E-PRTR rows default to **EPRTR_SectorCode 5** (Waste and
wastewater management) and **EPRTRAnnexIMainActivity** matching **Annex I activity 5** (a
standalone ``5`` in the activity text, not ``15`` / ``51``). ``--eprtr-all-sectors`` keeps
every sector code (Annex 5 filter still on unless ``--eprtr-no-annex-5-filter``).
Use ``--all-eprtr-sectors`` (alias ``--all-erptr-sectors``) for **exploration**: all sector
codes, no Annex-5 row filter, and **sum AIR releases across all pollutants** per facility so
more sites appear than with a single ``Pollutant`` filter. Use ``--cams-map`` for **CAMS
only** (no E-PRTR). For one country, add ``--eprtr-map`` for the E-PRTR overlay. CAMS point
popups use
NetCDF metadata; E-PRTR popups use facility name, sector, annex, inspire id, summed
releases, years. **F4_2** waste-transfer facilities (``F4_2_WasteTransfers_Facilities.csv``)
are drawn as a second E-PRTR layer (purple markers) when the CSV is present; use
``--no-waste-transfers-layer`` to skip. ``--list-f4-2-sectors`` prints sector codes/names
from F4_2; ``--f4-2-sector-codes`` restricts the F4_2 layer. Default basemap is CartoDB
Positron (avoids OSM.org tile blocks when opening HTML as a local file).

With ``--all-countries`` (no ``--cams-map``), the CAMS side is **point-only** unless
``--map-include-area``. Use ``--map-points-only`` for point-only CAMS on one country.

Basemaps: OpenStreetMap (default), light (CartoDB positron), Esri World Imagery. Use the
layer control to switch.

emission_category_index: J = 13 (1-based), same ordering as GNFR codes in this project
(A..E, then F1..F4, then G..L).

Usage (from project root):
  python Waste/Auxiliaries/cams_J_greece_map.py
  python Waste/Auxiliaries/cams_J_greece_map.py --pollutant nmvoc
  python Waste/Auxiliaries/cams_J_greece_map.py --out-map Waste/outputs/cams_J_GRC_nmvoc.html
  python Waste/Auxiliaries/cams_J_greece_map.py --whole-grid
  python Waste/Auxiliaries/cams_J_greece_map.py --all-countries --map-zoom 4
  python Waste/Auxiliaries/cams_J_greece_map.py --all-countries --cams-map --map-zoom 4
  python Waste/Auxiliaries/cams_J_greece_map.py --country GRC --eprtr-map
  python Waste/Auxiliaries/cams_J_greece_map.py --country GRC --eprtr-map --all-eprtr-sectors
  python Waste/Auxiliaries/cams_J_greece_map.py --country GRC --list-f4-2-sectors
  python Waste/Auxiliaries/cams_J_greece_map.py --country GRC --map-points-only

Requires: xarray, netCDF4, numpy, pandas; folium, branca
"""

from __future__ import annotations

import argparse
import html
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# GNFR J = index 13 (1-based): A..E = 1..5, F1..F4 = 6..9, G=10, H=11, I=12, J=13, ...
IDX_J_WASTE = 13

# E-PRTR CSV ``reportingYear`` aggregates to ``year_max`` / ``year_min`` per facility.
# Facilities with ``year_max`` at or below this year are dropped (not active after that year).
DEFAULT_EPRTR_REPORTING_YEAR_CUTOFF = 2018


def _eprtr_drop_inactive_by_year_max(
    g: pd.DataFrame,
    last_stale_year: int | None,
    *,
    label: str,
) -> pd.DataFrame:
    """
    Keep rows with finite ``year_max`` strictly greater than ``last_stale_year``.
    If ``last_stale_year`` is None, return ``g`` unchanged.
    """
    if g.empty or last_stale_year is None or "year_max" not in g.columns:
        return g
    ym = pd.to_numeric(g["year_max"], errors="coerce")
    keep = ym > float(last_stale_year)
    n_drop = int((~keep).sum())
    if n_drop:
        print(
            f"{label}: drop {n_drop} site(s) with reporting year_max <= {last_stale_year} "
            f"(inactive after {last_stale_year}); {int(keep.sum())} kept."
        )
    return g.loc[keep].reset_index(drop=True)


DEFAULT_NC = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "given_CAMS"
    / "CAMS-REG-ANT_v8.1_TNO_ftp"
    / "netcdf"
    / "CAMS-REG-v8_1_emissions_year2019.nc"
)

DEFAULT_EPRTR_FACILITIES_CSV = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "E_PRTR"
    / "eptr_csv"
    / "F1_4_Air_Releases_Facilities.csv"
)

DEFAULT_F4_2_WASTE_TRANSFERS_CSV = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "E_PRTR"
    / "eptr_csv"
    / "F4_2_WasteTransfers_Facilities.csv"
)

DEFAULT_EPRTR_POLLUTANT = "Non-methane volatile organic compounds (NMVOC)"

# E-PRTR sector 5 — "Waste and wastewater management" (this dataset / GNFR J overlay)
DEFAULT_EPRTR_SECTOR_CODE = 5.0

_ISO3_TO_EPRTR_COUNTRY: dict[str, str] = {
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

SKIP_VARS = frozenset(
    {
        "longitude_source",
        "latitude_source",
        "longitude_index",
        "latitude_index",
        "country_index",
        "emission_category_index",
        "source_type_index",
    }
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _pollutant_vars(ds) -> list[str]:
    out: list[str] = []
    for name, v in ds.data_vars.items():
        if name in SKIP_VARS:
            continue
        dims = tuple(v.dims)
        if len(dims) != 1 or dims[0] != "source":
            continue
        if not np.issubdtype(v.dtype, np.number):
            continue
        out.append(name)
    return sorted(out)


def _decode_country_ids(ds) -> list[str]:
    raw = ds["country_id"].values
    out: list[str] = []
    for x in raw:
        if isinstance(x, bytes):
            out.append(x.decode("utf-8", "replace").strip())
        else:
            out.append(str(x).strip())
    return out


def _decode_nc_text(x) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8", "replace").strip()
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, np.generic):
        if x.dtype.kind == "S":
            return x.item().decode("utf-8", "replace").strip()
        if x.dtype.kind == "U":
            return str(x.item()).strip()
    return str(x).strip()


def _country_index_1based(ds, iso3: str) -> int:
    codes = _decode_country_ids(ds)
    u = iso3.strip().upper()
    try:
        return codes.index(u) + 1
    except ValueError as exc:
        raise SystemExit(
            f"Country {iso3!r} not in NetCDF country_id (have {len(codes)} countries)."
        ) from exc


def _build_domain_mask(
    lon: np.ndarray,
    lat: np.ndarray,
    country_idx: np.ndarray,
    country_1based: int | None,
    bbox: tuple[float, float, float, float] | None,
) -> np.ndarray:
    if country_1based is None:
        m = np.ones(country_idx.shape[0], dtype=bool)
    else:
        m = country_idx == country_1based
    if bbox is not None:
        lon0, lat0, lon1, lat1 = bbox
        m = m & (lon >= lon0) & (lon <= lon1) & (lat >= lat0) & (lat <= lat1)
    return m


def _country_iso_and_name(ds, country_index_1based: int) -> tuple[str, str]:
    codes = _decode_country_ids(ds)
    names_raw = np.asarray(ds["country_name"].values)
    i = int(country_index_1based) - 1
    if i < 0 or i >= len(codes):
        return "?", "?"
    iso = codes[i]
    name = _decode_nc_text(names_raw[i]) if i < len(names_raw) else ""
    return iso, name


def _emis_cat_labels(ds, emission_category_index_1based: int) -> tuple[str, str]:
    codes = np.asarray(ds["emis_cat_code"].values)
    names = np.asarray(ds["emis_cat_name"].values)
    i = int(emission_category_index_1based) - 1
    if i < 0 or i >= len(codes):
        return "?", "?"
    return _decode_nc_text(codes[i]), _decode_nc_text(names[i])


def _source_type_labels(ds, source_type_index_1based: int) -> tuple[str, str]:
    codes = np.asarray(ds["source_type_code"].values)
    names = np.asarray(ds["source_type_name"].values)
    i = int(source_type_index_1based) - 1
    if i < 0 or i >= len(codes):
        return "?", "?"
    return _decode_nc_text(codes[i]), _decode_nc_text(names[i])


def _point_popup_html(
    ds,
    i: int,
    pollutants: list[str],
    map_pollutant: str,
    *,
    units: dict[str, str],
    country_idx: np.ndarray,
    emis_idx: np.ndarray,
    st_idx: np.ndarray,
) -> str:
    ci = int(country_idx[i])
    ei = int(emis_idx[i])
    si = int(st_idx[i])
    iso, cname = _country_iso_and_name(ds, ci)
    ecode, ename = _emis_cat_labels(ds, ei)
    stc, stname = _source_type_labels(ds, si)
    lon = float(np.asarray(ds["longitude_source"].values).ravel()[i])
    lat = float(np.asarray(ds["latitude_source"].values).ravel()[i])
    lines = [
        "<div style='min-width:220px;max-width:420px;font-size:13px;line-height:1.35'>",
        "<b>GNFR J (waste) — point</b> <span style='color:#555'>(CAMS-REG)</span><br/>",
        f"<b>Country</b> {html.escape(iso)} — {html.escape(cname)}<br/>",
        f"<b>Sector</b> {html.escape(ecode)} — {html.escape(ename)}<br/>",
        f"<b>Source type</b> {html.escape(stc)} — {html.escape(stname)}<br/>",
        f"<b>Location</b> lon {lon:.5f}, lat {lat:.5f}<br/>",
        "<hr style='margin:6px 0;border:none;border-top:1px solid #ccc'/>",
        "<b>Emissions (kg/year)</b> "
        "<span style='color:#555'>(species with value &gt; 0, finite)</span>",
    ]
    mp = map_pollutant.strip().lower()
    rows_data: list[tuple[str, float]] = []
    for name in pollutants:
        arr = np.asarray(ds[name].values).ravel().astype(np.float64)
        v = float(arr[i]) if i < arr.size else float("nan")
        if math.isfinite(v) and v > 0:
            rows_data.append((name, v))
    rows_data.sort(key=lambda t: (-t[1], t[0].lower()))
    if not rows_data:
        lines.append(
            "<p style='color:#666;margin:4px 0 0 0'>No positive finite emissions at this point.</p></div>"
        )
        return "".join(lines)
    lines.append(
        "<table style='border-collapse:collapse;width:100%;font-size:12px;margin-top:4px'>"
        "<tr><th align='left'>Species</th><th align='right'>kg/yr</th><th align='left'>units</th></tr>"
    )
    for name, v in rows_data:
        row_style = "background:#f0f7ff;" if name.lower() == mp else ""
        u = html.escape(units.get(name, ""))
        bold = "font-weight:bold;" if name.lower() == mp else ""
        lines.append(
            f"<tr style='{row_style}'><td style='{bold}'>{html.escape(name)}</td>"
            f"<td align='right' style='{bold}'>{_fmt(v)}</td>"
            f"<td style='color:#666'>{u}</td></tr>"
        )
    lines.append("</table></div>")
    return "".join(lines)


def _eprtr_country_for_filter(iso_or_name: str) -> str:
    u = iso_or_name.strip()
    if len(u) == 3 and u.upper() in _ISO3_TO_EPRTR_COUNTRY:
        return _ISO3_TO_EPRTR_COUNTRY[u.upper()]
    return u


def _eprtr_annex_join(s: pd.Series) -> str:
    raw = sorted({str(x) for x in s.dropna().unique()})
    t = "; ".join(raw[:18])
    if len(raw) > 18:
        t += "; …"
    return t


def _eprtr_csv_join_unique_short(s: pd.Series, *, max_items: int = 14) -> str:
    raw = sorted({str(x).strip() for x in s.dropna().unique() if str(x).strip()})
    t = "; ".join(raw[:max_items])
    if len(raw) > max_items:
        t += "; …"
    return t


def _eprtr_map_layer_name(
    sector_code: float | None,
    *,
    annex_activity_5_only: bool,
) -> str:
    if sector_code is not None:
        if abs(float(sector_code) - 5.0) < 1e-6:
            sc = "sector 5 (waste / wastewater)"
        else:
            sc = f"sector code {float(sector_code):g}"
    else:
        sc = "all sector codes"
    ann = "Annex I activity 5" if annex_activity_5_only else "all Annex I rows"
    return f"E-PRTR — AIR ({sc}, {ann})"


def _eprtr_annex_i_main_activity_includes_point_5(value: object) -> bool:
    """
    True if EPRTRAnnexIMainActivity denotes Annex I **activity 5** (digit 5 as its own
    code), e.g. leading ``5``, ``5.``, ``5(``, not ``15`` or ``51``.
    """
    if pd.isna(value):
        return False
    s = str(value).strip()
    if not s:
        return False
    return re.search(r"(?:^|[^\d])5(?:[^\d]|$)", s) is not None


def _eprtr_air_pollutant_pairs_by_fid(df: pd.DataFrame) -> dict[str, list[tuple[str, float]]]:
    """Per facility: list of (Pollutant, summed Releases) with finite value > 0, sorted by mass desc."""
    if df.empty or "Pollutant" not in df.columns:
        return {}
    tmp = (
        df.groupby(["FacilityInspireId", "Pollutant"], as_index=False)
        .agg(
            Releases_sum=(
                "Releases",
                lambda s: float(pd.to_numeric(s, errors="coerce").sum()),
            )
        )
    )
    rs = tmp["Releases_sum"].astype(np.float64)
    tmp = tmp[np.isfinite(rs) & (rs > 0)]
    out: dict[str, list[tuple[str, float]]] = {}
    for fid, sub in tmp.groupby("FacilityInspireId", sort=False):
        pairs = [(str(r["Pollutant"]), float(r["Releases_sum"])) for _, r in sub.iterrows()]
        pairs.sort(key=lambda t: (-t[1], t[0].lower()))
        out[str(fid)] = pairs
    return out


def load_eprtr_facilities_aggregated(
    csv_path: Path,
    *,
    all_countries: bool,
    country_arg: str,
    pollutant: str | None,
    sector_code: float | None,
    bbox: tuple[float, float, float, float] | None,
    annex_activity_5_only: bool = True,
    reporting_year_cutoff: int | None = DEFAULT_EPRTR_REPORTING_YEAR_CUTOFF,
) -> pd.DataFrame:
    """
    sector_code: e.g. 5.0 for EPRTR waste/wastewater sector; None = all sectors.
    pollutant: exact ``Pollutant`` column match, or None to keep all pollutants (summed per facility).
    """
    cols = [
        "PublicationDate",
        "countryName",
        "reportingYear",
        "EPRTR_SectorCode",
        "EPRTR_SectorName",
        "EPRTRAnnexIMainActivity",
        "FacilityInspireId",
        "facilityName",
        "city",
        "Longitude",
        "Latitude",
        "TargetRelease",
        "Pollutant",
        "Releases",
    ]
    df = pd.read_csv(csv_path, usecols=cols, low_memory=False)
    df = df[df["TargetRelease"].astype(str).str.strip().str.upper() == "AIR"]
    if pollutant is not None:
        df = df[df["Pollutant"] == pollutant]
    if sector_code is not None:
        sc = pd.to_numeric(df["EPRTR_SectorCode"], errors="coerce")
        df = df[np.isclose(sc, float(sector_code), rtol=0.0, atol=1e-9)]
    if annex_activity_5_only:
        act = df["EPRTRAnnexIMainActivity"]
        df = df[act.map(_eprtr_annex_i_main_activity_includes_point_5)]
    df = df[df["FacilityInspireId"].notna()]
    df = df[df["Longitude"].notna() & df["Latitude"].notna()]
    if not all_countries:
        cname = _eprtr_country_for_filter(country_arg)
        df = df[df["countryName"] == cname]
    if bbox is not None:
        lon0, lat0, lon1, lat1 = bbox
        df = df[
            (df["Longitude"] >= lon0)
            & (df["Longitude"] <= lon1)
            & (df["Latitude"] >= lat0)
            & (df["Latitude"] <= lat1)
        ]
    if df.empty:
        return df
    pair_map = _eprtr_air_pollutant_pairs_by_fid(df)
    annex = df.groupby("FacilityInspireId")["EPRTRAnnexIMainActivity"].apply(_eprtr_annex_join)
    g = df.groupby("FacilityInspireId", as_index=False).agg(
        Longitude=("Longitude", "first"),
        Latitude=("Latitude", "first"),
        facilityName=("facilityName", "first"),
        city=("city", "first"),
        countryName=("countryName", "first"),
        EPRTR_SectorCode=("EPRTR_SectorCode", "first"),
        EPRTR_SectorName=("EPRTR_SectorName", "first"),
        Releases_sum=("Releases", lambda s: float(pd.to_numeric(s, errors="coerce").sum())),
        year_min=("reportingYear", "min"),
        year_max=("reportingYear", "max"),
        pub_max=("PublicationDate", "max"),
    )
    g = g.merge(annex.reset_index(name="annex_activities"), on="FacilityInspireId")
    g["air_pollutant_pairs"] = g["FacilityInspireId"].astype(str).map(
        lambda x: pair_map.get(str(x), [])
    )
    g = _eprtr_drop_inactive_by_year_max(
        g, reporting_year_cutoff, label="E-PRTR F1_4 (AIR facilities)"
    )
    return g


def _f4_2_filter_chunk_for_map(
    ch: pd.DataFrame,
    *,
    cname: str | None,
    bbox: tuple[float, float, float, float] | None,
) -> pd.DataFrame:
    if cname is not None:
        ch = ch[ch["countryName"] == cname]
    if ch.empty:
        return ch
    ch = ch[ch["FacilityInspireId"].notna()]
    ch = ch[ch["Longitude"].notna() & ch["Latitude"].notna()]
    if bbox is not None:
        lon0, lat0, lon1, lat1 = bbox
        ch = ch[
            (ch["Longitude"] >= lon0)
            & (ch["Longitude"] <= lon1)
            & (ch["Latitude"] >= lat0)
            & (ch["Latitude"] <= lat1)
        ]
    return ch


def print_f4_2_sector_summary(
    csv_path: Path,
    *,
    all_countries: bool,
    country_arg: str,
    bbox: tuple[float, float, float, float] | None,
) -> None:
    """
    Print unique EPRTR_SectorCode / EPRTR_SectorName from F4_2 with row and facility counts
    (same country / bbox / coordinate filters as the map loader; no sector filter).
    """
    cols = [
        "countryName",
        "EPRTR_SectorCode",
        "EPRTR_SectorName",
        "FacilityInspireId",
        "Longitude",
        "Latitude",
    ]
    cname: str | None = None
    if not all_countries:
        cname = _eprtr_country_for_filter(country_arg)

    parts: list[pd.DataFrame] = []
    reader = pd.read_csv(
        csv_path,
        usecols=cols,
        low_memory=False,
        chunksize=250_000,
    )
    for ch in reader:
        ch = _f4_2_filter_chunk_for_map(ch, cname=cname, bbox=bbox)
        if ch.empty:
            continue
        parts.append(ch[["EPRTR_SectorCode", "EPRTR_SectorName", "FacilityInspireId"]])

    if not parts:
        print("No F4_2 rows after country / bbox / coordinate filters.")
        return

    df = pd.concat(parts, ignore_index=True)
    g = (
        df.groupby(["EPRTR_SectorCode", "EPRTR_SectorName"], dropna=False)
        .agg(
            n_rows=("FacilityInspireId", "size"),
            n_facilities=("FacilityInspireId", "nunique"),
        )
        .reset_index()
    )
    g = g.sort_values("n_rows", ascending=False)
    dom = "all countries" if all_countries else country_arg.strip().upper()
    print(f"F4_2 sector summary ({dom}) from {csv_path.name}")
    print(f"{'code':>8}  {'n_rows':>10}  {'n_facilities':>14}  sector name")
    print("-" * 100)
    for _, r in g.iterrows():
        code = r["EPRTR_SectorCode"]
        name = str(r["EPRTR_SectorName"] if pd.notna(r["EPRTR_SectorName"]) else "")
        print(
            f"{str(code):>8}  {int(r['n_rows']):>10}  {int(r['n_facilities']):>14}  {name}"
        )
    print("-" * 100)
    print(f"Total distinct (code, name) pairs: {len(g)}")
    print(
        "Use --f4-2-sector-codes CODE [CODE ...] to restrict the F4_2 map layer to chosen codes."
    )


def load_f4_2_waste_transfers_aggregated(
    csv_path: Path,
    *,
    all_countries: bool,
    country_arg: str,
    bbox: tuple[float, float, float, float] | None,
    sector_codes: frozenset[float] | None = None,
    reporting_year_cutoff: int | None = DEFAULT_EPRTR_REPORTING_YEAR_CUTOFF,
) -> pd.DataFrame:
    """
    E-PRTR form **F4_2** (waste transfers, facility-level): one map row per
    ``FacilityInspireId`` with ``wasteTransfers`` summed over CSV rows.
    ``sector_codes``: if set, keep only rows whose ``EPRTR_SectorCode`` is in the set.
    """
    cols = [
        "PublicationDate",
        "countryName",
        "reportingYear",
        "EPRTR_SectorCode",
        "EPRTR_SectorName",
        "EPRTRAnnexIMainActivity",
        "FacilityInspireId",
        "facilityName",
        "city",
        "Longitude",
        "Latitude",
        "wasteTreatment",
        "wasteClassification",
        "wasteTransfers",
    ]
    cname: str | None = None
    if not all_countries:
        cname = _eprtr_country_for_filter(country_arg)

    chunks: list[pd.DataFrame] = []
    chunk_size = 200_000
    reader = pd.read_csv(
        csv_path,
        usecols=cols,
        low_memory=False,
        chunksize=chunk_size,
    )
    for ch in reader:
        ch = _f4_2_filter_chunk_for_map(ch, cname=cname, bbox=bbox)
        if not ch.empty:
            chunks.append(ch)

    if not chunks:
        return pd.DataFrame()

    df = pd.concat(chunks, ignore_index=True)
    if sector_codes:
        sc = pd.to_numeric(df["EPRTR_SectorCode"], errors="coerce")
        mask = np.zeros(len(df), dtype=bool)
        for code in sector_codes:
            mask |= np.isclose(sc.astype(np.float64), float(code), rtol=0.0, atol=1e-9)
        df = df[mask]
    if df.empty:
        return pd.DataFrame()
    annex = df.groupby("FacilityInspireId")["EPRTRAnnexIMainActivity"].apply(_eprtr_annex_join)
    g = df.groupby("FacilityInspireId", as_index=False).agg(
        Longitude=("Longitude", "first"),
        Latitude=("Latitude", "first"),
        facilityName=("facilityName", "first"),
        city=("city", "first"),
        countryName=("countryName", "first"),
        EPRTR_SectorCode=("EPRTR_SectorCode", "first"),
        EPRTR_SectorName=("EPRTR_SectorName", "first"),
        transfers_sum=(
            "wasteTransfers",
            lambda s: float(pd.to_numeric(s, errors="coerce").sum()),
        ),
        waste_treatment=(
            "wasteTreatment",
            lambda s: _eprtr_csv_join_unique_short(s),
        ),
        waste_class=(
            "wasteClassification",
            lambda s: _eprtr_csv_join_unique_short(s),
        ),
        year_min=("reportingYear", "min"),
        year_max=("reportingYear", "max"),
        pub_max=("PublicationDate", "max"),
    )
    g = g.merge(annex.reset_index(name="annex_activities"), on="FacilityInspireId")
    g = _eprtr_drop_inactive_by_year_max(
        g, reporting_year_cutoff, label="E-PRTR F4_2 (waste transfers)"
    )
    return g


def _eprtr_facility_popup(row: pd.Series, pollutant_label: str) -> str:
    fid = str(row["FacilityInspireId"])
    name = str(row.get("facilityName", "") or "")
    city = str(row.get("city", "") or "")
    ctry = str(row.get("countryName", "") or "")
    scode = row.get("EPRTR_SectorCode", "")
    sname = str(row.get("EPRTR_SectorName", "") or "")
    annex = str(row.get("annex_activities", "") or "")
    rsum = float(row.get("Releases_sum", float("nan")))
    y0, y1 = row.get("year_min"), row.get("year_max")
    try:
        ys = f"{int(y0)}–{int(y1)}"
    except (TypeError, ValueError):
        ys = f"{y0}–{y1}"
    pub = str(row.get("pub_max", "") or "")
    lon, lat = float(row["Longitude"]), float(row["Latitude"])
    pairs_raw = row.get("air_pollutant_pairs")
    pairs: list[tuple[str, float]] = []
    if isinstance(pairs_raw, list):
        for item in pairs_raw:
            if not item or len(item) < 2:
                continue
            pname, val = str(item[0]), float(item[1])
            if math.isfinite(val) and val > 0:
                pairs.append((pname, val))
    if not pairs and math.isfinite(rsum) and rsum > 0:
        pairs = [(pollutant_label, rsum)]
    hl = pollutant_label.strip().lower()
    table_lines = [
        "<hr style='margin:6px 0;border:none;border-top:1px solid #ccc'/>",
        "<b>Pollutant emissions</b> <span style='color:#555'>(AIR, kg/year, &gt; 0)</span><br/>",
    ]
    if not pairs:
        table_lines.append(
            "<p style='color:#666;margin:0'>No positive finite AIR releases for this facility.</p>"
        )
    else:
        table_lines.append(
            "<table style='border-collapse:collapse;width:100%;font-size:12px;margin-top:4px'>"
            "<tr><th align='left'>Pollutant</th><th align='right'>kg/year</th></tr>"
        )
        for pname, val in pairs:
            row_style = "background:#f0fff4;font-weight:bold;" if pname.strip().lower() == hl else ""
            table_lines.append(
                f"<tr style='{row_style}'><td>{html.escape(pname)}</td>"
                f"<td align='right'>{_fmt(val)}</td></tr>"
            )
        table_lines.append("</table>")
    table_html = "".join(table_lines)
    return (
        "<div style='min-width:220px;max-width:460px;font-size:13px;line-height:1.35'>"
        "<b>E-PRTR</b> <span style='color:#555'>(F1_4 air releases, facility)</span><br/>"
        f"<b>Facility</b> {html.escape(name)}<br/>"
        f"<b>City · country</b> {html.escape(city)} — {html.escape(ctry)}<br/>"
        f"<b>EPRTR sector</b> {html.escape(str(scode))} — {html.escape(sname)}<br/>"
        f"<b>Annex I (subset)</b> {html.escape(annex)}<br/>"
        f"{table_html}"
        f"<b>Reporting years</b> {html.escape(ys)}"
        f"&nbsp;· <b>Publication (max)</b> {html.escape(pub)}<br/>"
        f"<b>Location</b> lon {lon:.5f}, lat {lat:.5f}<br/>"
        f"<b>FacilityInspireId</b><br/><span style='word-break:break-all'>{html.escape(fid)}</span>"
        "</div>"
    )


def add_eprtr_facility_markers(
    fmap,
    agg: pd.DataFrame,
    pollutant_label: str,
    *,
    skip_if_empty: bool = False,
    layer_name: str = "E-PRTR — AIR (facilities)",
) -> None:
    import folium
    from branca.colormap import LinearColormap

    if agg.empty:
        if skip_if_empty:
            print(
                "E-PRTR overlay: no facilities after filters (map shows CAMS layers only)."
            )
            return
        raise SystemExit(
            "No E-PRTR facility rows left after filters (check pollutant / country / bbox)."
        )
    vals = agg["Releases_sum"].astype(np.float64).values
    pos = vals[np.isfinite(vals) & (vals > 0)]
    if pos.size:
        vmin = float(np.min(pos))
        vmax = float(np.max(pos))
        if vmin >= vmax:
            vmax = vmin * 1.001 if vmin > 0 else 1.0
    else:
        vmin, vmax = 0.0, 1.0
    colors = ["#00441b", "#1a9850", "#66bd63", "#a6d96a", "#d9ef8b", "#ffffbf"]
    cmap = LinearColormap(
        colors, vmin=vmin, vmax=vmax, caption=f"E-PRTR: {pollutant_label} (kg/yr, summed)"
    )

    def radius_for(v: float) -> float:
        if not math.isfinite(v) or v <= 0:
            return 4.0
        t = math.sqrt(v / max(vmax, 1e-30))
        return min(22.0, 5.0 + 15.0 * t)

    fg = folium.FeatureGroup(name=layer_name, show=True)
    n = 0
    for _, row in agg.iterrows():
        lon, lat = float(row["Longitude"]), float(row["Latitude"])
        v = float(row["Releases_sum"])
        color = cmap(v) if math.isfinite(v) and v > 0 else "#888888"
        folium.CircleMarker(
            location=[lat, lon],
            radius=radius_for(v),
            color="#b35806",
            weight=2,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            popup=folium.Popup(
                _eprtr_facility_popup(row, pollutant_label),
                max_width=480,
            ),
        ).add_to(fg)
        n += 1
    fg.add_to(fmap)
    cmap.add_to(fmap)
    print(
        f"Map: {n} E-PRTR facilities ({layer_name}; "
        "aggregated AIR releases, green/orange markers)."
    )


def _f42_waste_transfer_popup(row: pd.Series) -> str:
    fid = str(row["FacilityInspireId"])
    name = str(row.get("facilityName", "") or "")
    city = str(row.get("city", "") or "")
    ctry = str(row.get("countryName", "") or "")
    scode = row.get("EPRTR_SectorCode", "")
    sname = str(row.get("EPRTR_SectorName", "") or "")
    annex = str(row.get("annex_activities", "") or "")
    wtreat = str(row.get("waste_treatment", "") or "")
    wclass = str(row.get("waste_class", "") or "")
    tsum = float(row.get("transfers_sum", float("nan")))
    y0, y1 = row.get("year_min"), row.get("year_max")
    try:
        ys = f"{int(y0)}–{int(y1)}"
    except (TypeError, ValueError):
        ys = f"{y0}–{y1}"
    pub = str(row.get("pub_max", "") or "")
    lon, lat = float(row["Longitude"]), float(row["Latitude"])
    return (
        "<div style='min-width:220px;max-width:460px;font-size:13px;line-height:1.35'>"
        "<b>E-PRTR F4_2</b> <span style='color:#555'>(waste transfers, facility)</span><br/>"
        f"<b>Facility</b> {html.escape(name)}<br/>"
        f"<b>City · country</b> {html.escape(city)} — {html.escape(ctry)}<br/>"
        f"<b>EPRTR sector</b> {html.escape(str(scode))} — {html.escape(sname)}<br/>"
        f"<b>Annex I (subset)</b> {html.escape(annex)}<br/>"
        f"<b>Waste treatment</b> {html.escape(wtreat)}<br/>"
        f"<b>Waste classification</b> {html.escape(wclass)}<br/>"
        "<hr style='margin:6px 0;border:none;border-top:1px solid #ccc'/>"
        "<b>Waste transfers</b> (F4_2 <code>wasteTransfers</code>, summed for this facility): "
        f"<b>{_fmt(tsum)}</b><br/>"
        "<span style='color:#666;font-size:11px'>See E-PRTR documentation for the unit "
        "used in your extract (often tonnes).</span><br/>"
        f"<b>Reporting years</b> {html.escape(ys)}"
        f"&nbsp;· <b>Publication (max)</b> {html.escape(pub)}<br/>"
        f"<b>Location</b> lon {lon:.5f}, lat {lat:.5f}<br/>"
        f"<b>FacilityInspireId</b><br/><span style='word-break:break-all'>{html.escape(fid)}</span>"
        "</div>"
    )


def add_f4_2_waste_transfer_markers(
    fmap,
    agg: pd.DataFrame,
    *,
    skip_if_empty: bool = True,
    layer_name: str = "E-PRTR F4_2 — waste transfers (facilities)",
) -> None:
    import folium
    from branca.colormap import LinearColormap

    if agg.empty:
        if skip_if_empty:
            print("F4_2 waste transfers: no facilities after filters (layer omitted).")
        return
    vals = agg["transfers_sum"].astype(np.float64).values
    pos = vals[np.isfinite(vals) & (vals > 0)]
    if pos.size:
        vmin = float(np.min(pos))
        vmax = float(np.max(pos))
        if vmin >= vmax:
            vmax = vmin * 1.001 if vmin > 0 else 1.0
    else:
        vmin, vmax = 0.0, 1.0
    colors = ["#f7f7f7", "#d9d9ea", "#b3b3d9", "#8c8cc8", "#6a51a3", "#3f007d"]
    cmap = LinearColormap(
        colors,
        vmin=vmin,
        vmax=vmax,
        caption="E-PRTR F4_2: wasteTransfers (summed; see popup for units)",
    )

    def radius_for(v: float) -> float:
        if not math.isfinite(v) or v <= 0:
            return 4.0
        t = math.sqrt(v / max(vmax, 1e-30))
        return min(20.0, 5.0 + 13.0 * t)

    fg = folium.FeatureGroup(name=layer_name, show=True)
    n = 0
    for _, row in agg.iterrows():
        lon, lat = float(row["Longitude"]), float(row["Latitude"])
        v = float(row["transfers_sum"])
        color = cmap(v) if math.isfinite(v) and v > 0 else "#aaaaaa"
        folium.CircleMarker(
            location=[lat, lon],
            radius=radius_for(v),
            color="#3f007d",
            weight=2,
            fill=True,
            fill_color=color,
            fill_opacity=0.82,
            popup=folium.Popup(_f42_waste_transfer_popup(row), max_width=480),
        ).add_to(fg)
        n += 1
    fg.add_to(fmap)
    cmap.add_to(fmap)
    print(
        f"Map: {n} F4_2 waste-transfer facilities ({layer_name}; purple markers)."
    )


def _summarize_pollutants(
    ds,
    base_mask: np.ndarray,
    pollutants: list[str],
) -> tuple[dict[str, tuple[float, float, float]], dict[str, str]]:
    st = np.asarray(ds["source_type_index"].values).ravel().astype(np.int64)
    area_m = base_mask & (st == 1)
    point_m = base_mask & (st == 2)
    units: dict[str, str] = {}
    rows: dict[str, tuple[float, float, float]] = {}
    for name in pollutants:
        arr = np.asarray(ds[name].values).ravel().astype(np.float64)
        if arr.size != base_mask.size:
            continue
        ta = float(np.nansum(arr[area_m]))
        tp = float(np.nansum(arr[point_m]))
        rows[name] = (ta, tp, ta + tp)
        u = ds[name].attrs.get("units", "kg/year")
        lu = ds[name].attrs.get("long_units", "")
        units[name] = f"{u}" + (f" ({lu})" if lu else "")
    return rows, units


def _fmt(x: float) -> str:
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


def _print_table(
    totals: dict[str, tuple[float, float, float]],
    units: dict[str, str],
) -> None:
    print("Totals by pollutant: area vs point (GNFR J waste, same domain)")
    print(
        f"{'pollutant':<12} {'area':>14} {'point':>14} {'total':>14}  "
        f"point_share%  units"
    )
    print("-" * 88)
    for name in sorted(totals.keys()):
        ta, tp, tt = totals[name]
        share = (100.0 * tp / tt) if tt > 0 else float("nan")
        u = units.get(name, "")
        print(
            f"{name:<12} {_fmt(ta):>14} {_fmt(tp):>14} {_fmt(tt):>14}  "
            f"{share:10.2f}  {u}"
        )


def _add_extra_basemaps(fmap) -> None:
    import folium

    folium.TileLayer(
        "OpenStreetMap",
        name="OpenStreetMap",
        control=True,
        max_zoom=19,
    ).add_to(fmap)
    _carto_attr = (
        '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> '
        'contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
    )
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png",
        attr=_carto_attr,
        name="Street (Carto Voyager)",
        subdomains="abcd",
        max_zoom=20,
        control=True,
    ).add_to(fmap)
    folium.TileLayer(
        "CartoDB dark_matter",
        name="Map (dark)",
        control=True,
    ).add_to(fmap)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community",
        name="Satellite (Esri)",
        overlay=False,
        control=True,
    ).add_to(fmap)


def add_cams_j_layers(
    fmap,
    ds,
    base_mask: np.ndarray,
    pollutant: str,
    pollutants: list[str],
    units: dict[str, str],
    *,
    map_max_area_cells: int = 0,
    map_sample_seed: int = 42,
    map_points_only: bool = False,
) -> None:
    import folium
    from branca.colormap import LinearColormap

    lon = np.asarray(ds["longitude_source"].values).ravel()
    lat = np.asarray(ds["latitude_source"].values).ravel()
    st = np.asarray(ds["source_type_index"].values).ravel().astype(np.int64)
    vals = np.asarray(ds[pollutant].values).ravel().astype(np.float64)
    m_area = base_mask & (st == 1)
    m_point = base_mask & (st == 2)
    country_idx = np.asarray(ds["country_index"].values).ravel().astype(np.int64)
    emis_ar = np.asarray(ds["emission_category_index"].values).ravel().astype(np.int64)

    pos = vals > 0
    vmask = m_point if map_points_only else (m_area | m_point)
    vpos = vals[pos & vmask]
    if vpos.size:
        vmin = float(np.min(vpos))
        vmax = float(np.max(vpos))
        if vmin >= vmax:
            vmax = vmin * 1.001 if vmin > 0 else 1.0
    else:
        vmin, vmax = 0.0, 1.0

    colors = ["#0d0887", "#6a00a8", "#b12a90", "#e16462", "#fca636", "#f0f921"]
    cmap = LinearColormap(colors, vmin=vmin, vmax=vmax, caption=f"{pollutant} (kg/year)")

    def radius_for(v: float) -> float:
        if not math.isfinite(v) or v <= 0:
            return 3.0
        t = math.sqrt(v / max(vmax, 1e-30))
        return min(28.0, 4.0 + 18.0 * t)

    fg_point = folium.FeatureGroup(
        name="GNFR J (Waste) — point",
        show=True,
    )

    if not map_points_only:
        fg_area = folium.FeatureGroup(
            name="J Waste — area (grid cell)",
            show=True,
        )

        lon_b = np.asarray(ds["longitude_bounds"].values, dtype=np.float64)
        lat_b = np.asarray(ds["latitude_bounds"].values, dtype=np.float64)
        nlon = int(lon_b.shape[0])
        nlat = int(lat_b.shape[0])
        lon_idx_raw = np.asarray(ds["longitude_index"].values).ravel().astype(np.int64)
        lat_idx_raw = np.asarray(ds["latitude_index"].values).ravel().astype(np.int64)
        if lon_idx_raw.max() >= nlon or lat_idx_raw.max() >= nlat:
            lon_idx_raw = np.maximum(0, lon_idx_raw - 1)
            lat_idx_raw = np.maximum(0, lat_idx_raw - 1)
        lon_ii = np.clip(lon_idx_raw, 0, nlon - 1)
        lat_ii = np.clip(lat_idx_raw, 0, nlat - 1)

        idx_area = np.flatnonzero(m_area)
        n_area_all = int(idx_area.size)
        if map_max_area_cells > 0 and n_area_all > map_max_area_cells:
            rng = np.random.default_rng(map_sample_seed)
            idx_area = np.sort(
                rng.choice(idx_area, size=map_max_area_cells, replace=False)
            )
            print(
                f"Map: drawing {map_max_area_cells} of {n_area_all} area grid cells "
                f"(totals in table use full domain)."
            )
        elif map_max_area_cells == 0 and n_area_all > 0:
            print(f"Map: drawing all {n_area_all} area grid cells (no subsampling).")

        for i in idx_area:
            v = float(vals[i])
            if not math.isfinite(v):
                continue
            color = cmap(v) if v > 0 else "#cccccc"
            li, ji = int(lon_ii[i]), int(lat_ii[i])
            west, east = float(lon_b[li, 0]), float(lon_b[li, 1])
            south, north = float(lat_b[ji, 0]), float(lat_b[ji, 1])
            if south > north:
                south, north = north, south
            if west > east:
                west, east = east, west
            pop = (
                f"<b>{html.escape(pollutant)}</b><br/>{v:.6g} kg/yr<br/>area (grid cell)<br/>"
                f"cell lon [{west:.4f}, {east:.4f}]<br/>"
                f"cell lat [{south:.4f}, {north:.4f}]<br/>"
                f"centre lon {lon[i]:.4f}, lat {lat[i]:.4f}"
            )
            folium.Rectangle(
                bounds=[[south, west], [north, east]],
                color="#333333",
                weight=1,
                fill=True,
                fill_color=color,
                fill_opacity=0.45,
                popup=folium.Popup(pop, max_width=280),
            ).add_to(fg_area)

    n_drawn_points = 0
    for i in np.flatnonzero(m_point):
        v = float(vals[i])
        if not math.isfinite(v):
            continue
        color = cmap(v) if v > 0 else "#cccccc"
        r = radius_for(v)
        pop_html = _point_popup_html(
            ds,
            int(i),
            pollutants,
            pollutant,
            units=units,
            country_idx=country_idx,
            emis_idx=emis_ar,
            st_idx=st,
        )
        folium.CircleMarker(
            location=[float(lat[i]), float(lon[i])],
            radius=r,
            color="#333333",
            weight=2,
            fill=True,
            fill_color=color,
            fill_opacity=0.75,
            popup=folium.Popup(pop_html, max_width=460),
        ).add_to(fg_point)
        n_drawn_points += 1

    if map_points_only:
        print(f"Map: {n_drawn_points} GNFR J point sources (area layer omitted).")

    if not map_points_only:
        fg_area.add_to(fmap)
    fg_point.add_to(fmap)
    cmap.add_to(fmap)


def build_folium_map(
    ds,
    base_mask: np.ndarray,
    pollutant: str,
    pollutants: list[str],
    units: dict[str, str],
    *,
    out_html: Path,
    map_center_latlon: tuple[float, float] | None,
    zoom_start: int,
    map_max_area_cells: int,
    map_sample_seed: int,
    map_points_only: bool,
    eprtr_agg: pd.DataFrame | None = None,
    eprtr_pollutant_label: str | None = None,
    eprtr_layer_name: str = "E-PRTR — AIR (facilities)",
    waste_transfers_agg: pd.DataFrame | None = None,
    waste_transfers_layer_name: str = "E-PRTR F4_2 — waste transfers (facilities)",
) -> None:
    import folium

    lon = np.asarray(ds["longitude_source"].values).ravel()
    lat = np.asarray(ds["latitude_source"].values).ravel()
    st = np.asarray(ds["source_type_index"].values).ravel().astype(np.int64)
    m_pt = base_mask & (st == 2)
    has_cam_pts = int(np.count_nonzero(m_pt)) > 0
    has_eprtr = (
        eprtr_agg is not None
        and eprtr_pollutant_label is not None
        and not eprtr_agg.empty
    )
    has_wt = waste_transfers_agg is not None and not waste_transfers_agg.empty

    if map_center_latlon is not None:
        center_lat, center_lon = map_center_latlon
    else:
        means: list[tuple[float, float]] = []
        if has_cam_pts:
            means.append(
                (float(np.mean(lat[m_pt])), float(np.mean(lon[m_pt])))
            )
        if has_eprtr:
            means.append(
                (
                    float(eprtr_agg["Latitude"].mean()),
                    float(eprtr_agg["Longitude"].mean()),
                )
            )
        if has_wt:
            means.append(
                (
                    float(waste_transfers_agg["Latitude"].mean()),
                    float(waste_transfers_agg["Longitude"].mean()),
                )
            )
        if means:
            center_lat = sum(m[0] for m in means) / len(means)
            center_lon = sum(m[1] for m in means) / len(means)
        else:
            center_lat = float(np.mean(lat[base_mask]))
            center_lon = float(np.mean(lon[base_mask]))

    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=int(zoom_start),
        tiles="CartoDB positron",
        attr=(
            '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> '
            'contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
        ),
    )
    _add_extra_basemaps(fmap)
    add_cams_j_layers(
        fmap,
        ds,
        base_mask,
        pollutant,
        pollutants,
        units,
        map_max_area_cells=map_max_area_cells,
        map_sample_seed=map_sample_seed,
        map_points_only=map_points_only,
    )
    if eprtr_agg is not None and eprtr_pollutant_label is not None:
        add_eprtr_facility_markers(
            fmap,
            eprtr_agg,
            eprtr_pollutant_label,
            skip_if_empty=True,
            layer_name=eprtr_layer_name,
        )
    if waste_transfers_agg is not None:
        add_f4_2_waste_transfer_markers(
            fmap,
            waste_transfers_agg,
            skip_if_empty=True,
            layer_name=waste_transfers_layer_name,
        )
    folium.LayerControl(collapsed=False).add_to(fmap)

    out_html.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out_html))


def main() -> None:
    root = _project_root()
    ap = argparse.ArgumentParser(
        description="CAMS GNFR J (Waste): Greece map + area/point source counts and totals."
    )
    ap.add_argument("--nc", type=Path, default=DEFAULT_NC, help="CAMS NetCDF path")
    ap.add_argument("--country", default="GRC", help="ISO3 (default GRC)")
    ap.add_argument(
        "--all-countries",
        action="store_true",
        help=(
            "Ignore --country; use full CAMS domain (still GNFR J only). "
            "If the default E-PRTR CSV exists, the map shows CAMS GNFR J points and "
            "E-PRTR facilities on the same map. Use --cams-map for CAMS only. "
            "See --map-include-area for CAMS area cells."
        ),
    )
    ap.add_argument(
        "--cams-map",
        action="store_true",
        help="Do not add the E-PRTR facility layer (CAMS only on the map).",
    )
    ap.add_argument(
        "--eprtr-map",
        action="store_true",
        help="Include E-PRTR F1_4 facility markers on the map (with CAMS layers).",
    )
    ap.add_argument(
        "--eprtr-csv",
        type=Path,
        default=None,
        help="Path to F1_4_Air_Releases_Facilities.csv (default: data/E_PRTR/eptr_csv/...).",
    )
    ap.add_argument(
        "--waste-transfers-csv",
        type=Path,
        default=None,
        help=(
            "Path to F4_2_WasteTransfers_Facilities.csv "
            "(default: data/E_PRTR/eptr_csv/F4_2_WasteTransfers_Facilities.csv)."
        ),
    )
    ap.add_argument(
        "--no-waste-transfers-layer",
        action="store_true",
        help="Do not load or draw the E-PRTR F4_2 waste-transfers facility layer.",
    )
    ap.add_argument(
        "--eprtr-reporting-cutoff-year",
        type=int,
        default=DEFAULT_EPRTR_REPORTING_YEAR_CUTOFF,
        metavar="Y",
        help=(
            "E-PRTR F1_4 / F4_2 map layers: drop facilities whose aggregated reportingYear "
            "maximum is at or below Y (inactive after Y). Default "
            f"{DEFAULT_EPRTR_REPORTING_YEAR_CUTOFF}."
        ),
    )
    ap.add_argument(
        "--no-eprtr-reporting-year-filter",
        action="store_true",
        help="Keep all E-PRTR F1_4 / F4_2 facilities regardless of reporting year_max.",
    )
    ap.add_argument(
        "--list-f4-2-sectors",
        action="store_true",
        help=(
            "Print EPRTR_SectorCode / EPRTR_SectorName with row and facility counts from "
            "the F4_2 CSV (respects --country / --all-countries and --bbox), then exit."
        ),
    )
    ap.add_argument(
        "--f4-2-sector-codes",
        type=float,
        nargs="+",
        default=None,
        metavar="CODE",
        help=(
            "Restrict the F4_2 map layer to these EPRTR_SectorCode values "
            "(e.g. --f4-2-sector-codes 5 6). Use --list-f4-2-sectors to list codes in your extract."
        ),
    )
    ap.add_argument(
        "--eprtr-pollutant",
        default=DEFAULT_EPRTR_POLLUTANT,
        metavar="TEXT",
        help="Exact Pollutant column value to filter (default: NMVOC long name).",
    )
    ap.add_argument(
        "--eprtr-sector-code",
        type=float,
        default=DEFAULT_EPRTR_SECTOR_CODE,
        metavar="N",
        help=(
            "EPRTR_SectorCode for E-PRTR facilities (default: 5 = Waste and wastewater). "
            "Ignored if --eprtr-all-sectors."
        ),
    )
    ap.add_argument(
        "--eprtr-all-sectors",
        action="store_true",
        help="Include all EPRTR_SectorCode values (do not restrict to the default sector).",
    )
    ap.add_argument(
        "--eprtr-no-annex-5-filter",
        action="store_true",
        help=(
            "Do not require EPRTRAnnexIMainActivity to denote Annex I activity 5 "
            "(digit 5 as its own code)."
        ),
    )
    ap.add_argument(
        "--all-eprtr-sectors",
        "--all-erptr-sectors",
        dest="all_eprtr_sectors",
        action="store_true",
        help=(
            "E-PRTR exploration overlay: turns on the E-PRTR layer (unless --cams-map); "
            "all EPRTR_SectorCode values; no Annex-5 row filter; sum AIR releases across "
            "all Pollutant values per facility (still AIR-only; same country/bbox as the run). "
            "Much fuller facility set than the default waste+Annex-5+single-pollutant filter. "
            "Ignores --eprtr-pollutant for row selection."
        ),
    )
    ap.add_argument(
        "--map-points-only",
        action="store_true",
        help="Map: draw GNFR J point sources only (no area rectangles).",
    )
    ap.add_argument(
        "--map-include-area",
        action="store_true",
        help="With --all-countries, also draw the area layer on the map (heavy).",
    )
    ap.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
        help="Optional WGS84 bbox",
    )
    ap.add_argument("--pollutant", default="nmvoc", help="Variable for map colouring")
    ap.add_argument("--no-map", action="store_true", help="Skip HTML output")
    ap.add_argument(
        "--no-table",
        action="store_true",
        help="Do not print area/point totals table",
    )
    ap.add_argument(
        "--out-map",
        type=Path,
        default=None,
        help="Output HTML (default: Waste/outputs/cams_J_<ISO>_pollutant.html)",
    )
    ap.add_argument(
        "--whole-grid",
        action="store_true",
        help=(
            "Draw every area grid cell on the map (no random subsampling). "
            "Same as --map-max-area-cells 0. Combine with --all-countries for the full "
            "CAMS domain (large HTML)."
        ),
    )
    ap.add_argument(
        "--map-max-area-cells",
        type=int,
        default=25_000,
        help="Max area rectangles on map (0 = all; overridden by --whole-grid)",
    )
    ap.add_argument("--map-sample-seed", type=int, default=42)
    ap.add_argument("--map-zoom", type=int, default=None)
    ap.add_argument("--map-center-lat", type=float, default=None)
    ap.add_argument("--map-center-lon", type=float, default=None)
    ap.add_argument("--out-csv", type=Path, default=None)
    args = ap.parse_args()
    all_eprtr_explore = bool(args.all_eprtr_sectors)
    if all_eprtr_explore:
        eprtr_sector_for_load: float | None = None
    else:
        eprtr_sector_for_load = (
            None if args.eprtr_all_sectors else float(args.eprtr_sector_code)
        )
    eprtr_annex_5_only = not bool(args.eprtr_no_annex_5_filter) and not all_eprtr_explore
    eprtr_pollutant_for_load: str | None = (
        None if all_eprtr_explore else args.eprtr_pollutant
    )
    map_points_only = bool(
        args.map_points_only or (args.all_countries and not args.map_include_area)
    )

    if args.eprtr_csv is not None:
        eprtr_csv_path = (
            args.eprtr_csv if args.eprtr_csv.is_absolute() else root / args.eprtr_csv
        )
    else:
        eprtr_csv_path = DEFAULT_EPRTR_FACILITIES_CSV

    if args.waste_transfers_csv is not None:
        wt_csv_path = (
            args.waste_transfers_csv
            if args.waste_transfers_csv.is_absolute()
            else root / args.waste_transfers_csv
        )
    else:
        wt_csv_path = DEFAULT_F4_2_WASTE_TRANSFERS_CSV

    bbox_f4: tuple[float, float, float, float] | None = (
        tuple(args.bbox) if args.bbox else None
    )
    if args.list_f4_2_sectors:
        if not wt_csv_path.is_file():
            raise SystemExit(f"F4_2 CSV not found: {wt_csv_path}")
        print_f4_2_sector_summary(
            wt_csv_path,
            all_countries=args.all_countries,
            country_arg=args.country,
            bbox=bbox_f4,
        )
        return

    f4_2_sector_codes: frozenset[float] | None = (
        frozenset(float(x) for x in args.f4_2_sector_codes)
        if args.f4_2_sector_codes
        else None
    )

    eprtr_reporting_cutoff: int | None = (
        None
        if args.no_eprtr_reporting_year_filter
        else int(args.eprtr_reporting_cutoff_year)
    )

    auto_eprtr_map = (
        args.all_countries and not args.cams_map and eprtr_csv_path.is_file()
    )
    use_eprtr_map = (
        not args.cams_map
        and (
            bool(args.eprtr_map)
            or auto_eprtr_map
            or bool(args.all_eprtr_sectors)
        )
    )
    if (args.eprtr_map or args.all_eprtr_sectors) and not eprtr_csv_path.is_file():
        raise SystemExit(f"E-PRTR CSV not found: {eprtr_csv_path}")

    nc_path = args.nc if args.nc.is_absolute() else root / args.nc
    if not nc_path.is_file():
        raise SystemExit(f"NetCDF not found: {nc_path}")

    import xarray as xr

    ds = xr.open_dataset(nc_path)
    try:
        pollutants = _pollutant_vars(ds)
        if not pollutants:
            raise SystemExit("No pollutant variables with dimension 'source' found.")

        country_1b: int | None = None if args.all_countries else _country_index_1based(
            ds, args.country
        )

        emis = np.asarray(ds["emission_category_index"].values).ravel().astype(np.int64)
        ci = np.asarray(ds["country_index"].values).ravel().astype(np.int64)
        lon = np.asarray(ds["longitude_source"].values).ravel()
        lat = np.asarray(ds["latitude_source"].values).ravel()
        bbox_t = bbox_f4
        sector_j = emis == IDX_J_WASTE
        domain = _build_domain_mask(lon, lat, ci, country_1b, bbox_t)
        base = sector_j & domain
        if int(np.count_nonzero(base)) == 0:
            raise SystemExit("No GNFR J (waste) sources match country/bbox filter.")

        print(f"NetCDF: {nc_path.name}")
        print(
            f"Filter: GNFR J waste (emission_category_index={IDX_J_WASTE}), "
            f"{'all countries' if args.all_countries else args.country.upper()}"
        )
        if args.bbox:
            print(
                f"  bbox lon[{args.bbox[0]},{args.bbox[2]}] "
                f"lat[{args.bbox[1]},{args.bbox[3]}]"
            )
        st = np.asarray(ds["source_type_index"].values).ravel().astype(np.int64)
        n_area = int(np.count_nonzero(base & (st == 1)))
        n_point = int(np.count_nonzero(base & (st == 2)))
        print(
            f"Sources in domain: {int(np.count_nonzero(base))} "
            f"(area type={n_area}, point type={n_point})"
        )
        print()

        totals, units = _summarize_pollutants(ds, base, pollutants)
        if not args.no_table:
            _print_table(totals, units)
            print()

        pol = args.pollutant.strip().lower()
        if pol not in totals:
            raise SystemExit(
                f"Unknown pollutant {pol!r}. Examples: {', '.join(pollutants[:8])}"
            )

        if not args.no_map:
            agg_eprtr: pd.DataFrame | None = None
            agg_wt: pd.DataFrame | None = None
            if not args.no_waste_transfers_layer:
                if wt_csv_path.is_file():
                    try:
                        agg_wt = load_f4_2_waste_transfers_aggregated(
                            wt_csv_path,
                            all_countries=args.all_countries,
                            country_arg=args.country,
                            bbox=bbox_t,
                            sector_codes=f4_2_sector_codes,
                            reporting_year_cutoff=eprtr_reporting_cutoff,
                        )
                        print(
                            f"E-PRTR F4_2: {len(agg_wt)} facilities "
                            f"(waste transfers summed) from {wt_csv_path}"
                        )
                        if f4_2_sector_codes:
                            scs = ", ".join(str(c) for c in sorted(f4_2_sector_codes))
                            print(f"  F4_2 sector filter: EPRTR_SectorCode in {{{scs}}}")
                        print()
                    except Exception as exc:
                        agg_wt = None
                        print(
                            f"Warning: could not load F4_2 waste transfers "
                            f"({wt_csv_path}): {exc}"
                        )
                        print()
                else:
                    print(
                        f"Note: F4_2 waste transfers CSV not found ({wt_csv_path}); "
                        "layer skipped."
                    )
                    print()

            if use_eprtr_map:
                agg_eprtr = load_eprtr_facilities_aggregated(
                    eprtr_csv_path,
                    all_countries=args.all_countries,
                    country_arg=args.country,
                    pollutant=eprtr_pollutant_for_load,
                    sector_code=eprtr_sector_for_load,
                    bbox=bbox_t,
                    annex_activity_5_only=eprtr_annex_5_only,
                    reporting_year_cutoff=eprtr_reporting_cutoff,
                )
                print(
                    f"E-PRTR: {len(agg_eprtr)} facilities (after filter) "
                    f"from {eprtr_csv_path}"
                )
                if all_eprtr_explore:
                    print(
                        "  E-PRTR exploration (--all-eprtr-sectors): all sector codes, "
                        "no Annex-5 row filter, all pollutants summed per facility (AIR only)."
                    )
                elif eprtr_sector_for_load is not None:
                    label = (
                        " (Waste and wastewater management)"
                        if abs(eprtr_sector_for_load - 5.0) < 1e-6
                        else ""
                    )
                    print(
                        f"  E-PRTR sector filter: EPRTR_SectorCode == "
                        f"{eprtr_sector_for_load:g}{label}"
                    )
                else:
                    print("  E-PRTR sector filter: (all sectors, --eprtr-all-sectors)")
                if not all_eprtr_explore:
                    if args.eprtr_no_annex_5_filter:
                        print(
                            "  E-PRTR Annex I filter: off (--eprtr-no-annex-5-filter)"
                        )
                    else:
                        print(
                            "  E-PRTR Annex I filter: EPRTRAnnexIMainActivity contains "
                            "activity 5 (not 15/51/…)"
                        )
                if auto_eprtr_map and not args.eprtr_map:
                    print(
                        "  (same Folium map as CAMS GNFR J points; "
                        "--cams-map to omit E-PRTR.)"
                    )

            out_map = args.out_map
            tag = "ALL" if args.all_countries else args.country.upper()
            if out_map is None:
                if use_eprtr_map:
                    suffix = f"{pol}_eprtr_explore" if args.all_eprtr_sectors else f"{pol}_eprtr"
                else:
                    suffix = pol
                out_map = root / "Waste" / "outputs" / f"cams_J_{tag}_{suffix}.html"
            else:
                out_map = out_map if out_map.is_absolute() else root / out_map

            mclat, mclon = args.map_center_lat, args.map_center_lon
            if (mclat is None) ^ (mclon is None):
                raise SystemExit("Set both --map-center-lat and --map-center-lon, or neither.")
            map_center: tuple[float, float] | None
            if mclat is not None:
                map_center = (float(mclat), float(mclon))
            elif args.all_countries:
                map_center = (51.0, 14.0)
            else:
                map_center = None

            zoom_use = args.map_zoom if args.map_zoom is not None else (
                4 if args.all_countries else 7
            )

            map_max_area_cells = (
                0 if args.whole_grid else max(0, int(args.map_max_area_cells))
            )
            if (
                not map_points_only
                and map_max_area_cells == 0
                and n_area > 80_000
            ):
                print(
                    f"Warning: {n_area} area cells will be drawn; the HTML can be very "
                    "large and slow in the browser."
                )
            if (
                args.all_countries
                and not args.cams_map
                and not eprtr_csv_path.is_file()
            ):
                print(
                    f"Note: default E-PRTR CSV missing ({eprtr_csv_path}); "
                    "map shows CAMS only."
                )

            if args.all_eprtr_sectors:
                eprtr_layer_name = (
                    "E-PRTR — AIR (explore: all sectors, all pollutants, no Annex-5 filter)"
                )
            else:
                eprtr_layer_name = _eprtr_map_layer_name(
                    eprtr_sector_for_load,
                    annex_activity_5_only=eprtr_annex_5_only,
                )
            eprtr_lbl = (
                "All pollutants (AIR, summed per facility)"
                if (use_eprtr_map and args.all_eprtr_sectors)
                else (args.eprtr_pollutant if use_eprtr_map else None)
            )
            wt_layer_name = "E-PRTR F4_2 — waste transfers (facilities)"
            if f4_2_sector_codes:
                wt_layer_name += (
                    " [sectors "
                    + ", ".join(str(c) for c in sorted(f4_2_sector_codes))
                    + "]"
                )
            build_folium_map(
                ds,
                base,
                pol,
                pollutants,
                units,
                out_html=out_map,
                map_center_latlon=map_center,
                zoom_start=int(zoom_use),
                map_max_area_cells=map_max_area_cells,
                map_sample_seed=int(args.map_sample_seed),
                map_points_only=map_points_only,
                eprtr_agg=agg_eprtr if use_eprtr_map else None,
                eprtr_pollutant_label=eprtr_lbl,
                eprtr_layer_name=eprtr_layer_name,
                waste_transfers_agg=agg_wt,
                waste_transfers_layer_name=wt_layer_name,
            )
            n_ep = 0 if agg_eprtr is None else len(agg_eprtr)
            n_wt = 0 if agg_wt is None else len(agg_wt)
            if use_eprtr_map and n_ep > 0 and n_wt > 0:
                print(
                    f"Interactive map (CAMS GNFR J + E-PRTR F1_4 + F4_2 waste transfers): {out_map}"
                )
            elif use_eprtr_map and n_ep > 0:
                print(f"Interactive map (CAMS GNFR J + E-PRTR): {out_map}")
            elif use_eprtr_map and n_wt > 0:
                print(
                    f"Interactive map (CAMS GNFR J + F4_2 waste transfers; "
                    f"F1_4 overlay empty): {out_map}"
                )
            elif use_eprtr_map:
                print(f"Interactive map (CAMS GNFR J; E-PRTR F1_4 overlay empty): {out_map}")
            elif n_wt > 0:
                print(f"Interactive map (CAMS GNFR J + E-PRTR F4_2 waste transfers): {out_map}")
            else:
                print(f"Interactive map ({pol}): {out_map}")

        if args.out_csv:
            csv_path = args.out_csv if args.out_csv.is_absolute() else root / args.out_csv
            rows = []
            for name in sorted(totals.keys()):
                ta, tp, tt = totals[name]
                rows.append(
                    {
                        "pollutant": name,
                        "area_kg_yr": ta,
                        "point_kg_yr": tp,
                        "total_kg_yr": tt,
                        "point_fraction": (tp / tt) if tt > 0 else float("nan"),
                        "units": units.get(name, ""),
                    }
                )
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            print(f"CSV: {csv_path}")
    finally:
        ds.close()


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
