#!/usr/bin/env python3
"""
Interactive Folium map of CAMS-REG-ANT GNFR E (Solvents / product use) for Greece.

Area sources (source_type_index == 1) are drawn as grid-cell rectangles; GNFR **E point**
sources (``source_type_index == 2``) are circle markers (CAMS-REG). When the default
E-PRTR CSV exists, ``--all-countries`` adds an **E-PRTR F1_4** facility layer on the **same**
map (toggle layers in the control). E-PRTR rows default to **EPRTR_SectorCode 9** (Other
activities); ``--eprtr-all-sectors`` keeps every sector. Use ``--cams-map`` for **CAMS
only** (no E-PRTR). For one country, add ``--eprtr-map`` for the E-PRTR overlay. CAMS point
popups use
NetCDF metadata; E-PRTR popups use facility name, sector, annex, inspire id, summed
releases, years.

With ``--all-countries`` (no ``--cams-map``), the CAMS side is **point-only** unless
``--map-include-area``. Use ``--map-points-only`` for point-only CAMS on one country.

Basemaps: OpenStreetMap (default), light (CartoDB positron), Esri World Imagery. Use the
layer control to switch.

emission_category_index: E = 5 (1-based), same ordering as GNFR A..L in this project.

Usage (from project root):
  python Solvents/Auxiliaries/cams_E_greece_map.py
  python Solvents/Auxiliaries/cams_E_greece_map.py --pollutant nmvoc
  python Solvents/Auxiliaries/cams_E_greece_map.py --out-map Solvents/outputs/cams_E_GRC_nmvoc.html
  python Solvents/Auxiliaries/cams_E_greece_map.py --whole-grid
  python Solvents/Auxiliaries/cams_E_greece_map.py --all-countries --map-zoom 4
  python Solvents/Auxiliaries/cams_E_greece_map.py --all-countries --cams-map --map-zoom 4
  python Solvents/Auxiliaries/cams_E_greece_map.py --country GRC --eprtr-map
  python Solvents/Auxiliaries/cams_E_greece_map.py --country GRC --map-points-only

Requires: xarray, netCDF4, numpy, pandas; folium, branca
"""

from __future__ import annotations

import argparse
import html
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# GNFR E = index 5 (1-based): A=1, B=2, C=3, D=4, E=5, ...
IDX_E_SOLVENTS = 5

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

DEFAULT_EPRTR_POLLUTANT = "Non-methane volatile organic compounds (NMVOC)"

# E-PRTR sector 9 — "Other activities" (product use / solvent-type reporting, etc.)
DEFAULT_EPRTR_SECTOR_CODE = 9.0

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
        "<b>GNFR E — point</b> <span style='color:#555'>(CAMS-REG)</span><br/>",
        f"<b>Country</b> {html.escape(iso)} — {html.escape(cname)}<br/>",
        f"<b>Sector</b> {html.escape(ecode)} — {html.escape(ename)}<br/>",
        f"<b>Source type</b> {html.escape(stc)} — {html.escape(stname)}<br/>",
        f"<b>Location</b> lon {lon:.5f}, lat {lat:.5f}<br/>",
        "<hr style='margin:6px 0;border:none;border-top:1px solid #ccc'/>",
        "<b>Emissions (kg/year)</b>",
        "<table style='border-collapse:collapse;width:100%;font-size:12px'>",
        "<tr><th align='left'>Species</th><th align='right'>kg/yr</th><th align='left'>units</th></tr>",
    ]
    mp = map_pollutant.strip().lower()
    for name in sorted(pollutants):
        arr = np.asarray(ds[name].values).ravel().astype(np.float64)
        v = float(arr[i]) if i < arr.size else float("nan")
        row_style = "background:#f0f7ff;" if name == mp else ""
        u = html.escape(units.get(name, ""))
        bold = "font-weight:bold;" if name == mp else ""
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


def load_eprtr_facilities_aggregated(
    csv_path: Path,
    *,
    all_countries: bool,
    country_arg: str,
    pollutant: str,
    sector_code: float | None,
    bbox: tuple[float, float, float, float] | None,
) -> pd.DataFrame:
    """sector_code: e.g. 9.0 for EPRTR \"Other activities\"; None = all sectors."""
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
    df = df[df["Pollutant"] == pollutant]
    if sector_code is not None:
        sc = pd.to_numeric(df["EPRTR_SectorCode"], errors="coerce")
        df = df[np.isclose(sc, float(sector_code), rtol=0.0, atol=1e-9)]
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
    return (
        "<div style='min-width:220px;max-width:460px;font-size:13px;line-height:1.35'>"
        "<b>E-PRTR</b> <span style='color:#555'>(F1_4 air releases, facility)</span><br/>"
        f"<b>Facility</b> {html.escape(name)}<br/>"
        f"<b>City · country</b> {html.escape(city)} — {html.escape(ctry)}<br/>"
        f"<b>EPRTR sector</b> {html.escape(str(scode))} — {html.escape(sname)}<br/>"
        f"<b>Annex I (subset)</b> {html.escape(annex)}<br/>"
        "<hr style='margin:6px 0;border:none;border-top:1px solid #ccc'/>"
        f"<b>{html.escape(pollutant_label)}</b> (reported AIR releases, summed over rows): "
        f"<b>{_fmt(rsum)}</b> kg/year<br/>"
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

    fg = folium.FeatureGroup(name="E-PRTR — AIR (facilities)", show=True)
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
    print(f"Map: {n} E-PRTR facilities (aggregated AIR releases, green/orange markers).")


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
    print("Totals by pollutant: area vs point (GNFR E, same domain)")
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

    folium.TileLayer("CartoDB positron", name="Map (light)", control=True).add_to(fmap)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community",
        name="Satellite (Esri)",
        overlay=False,
        control=True,
    ).add_to(fmap)


def add_cams_e_layers(
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
        name="GNFR E (Solvents) — point",
        show=True,
    )

    if not map_points_only:
        fg_area = folium.FeatureGroup(
            name="E Solvents - area (grid cell)",
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
        print(f"Map: {n_drawn_points} GNFR E point sources (area layer omitted).")

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
    if map_center_latlon is not None:
        center_lat, center_lon = map_center_latlon
    elif has_cam_pts and has_eprtr:
        center_lat = (
            float(np.mean(lat[m_pt])) + float(eprtr_agg["Latitude"].mean())
        ) / 2.0
        center_lon = (
            float(np.mean(lon[m_pt])) + float(eprtr_agg["Longitude"].mean())
        ) / 2.0
    elif has_cam_pts:
        center_lat = float(np.mean(lat[m_pt]))
        center_lon = float(np.mean(lon[m_pt]))
    elif has_eprtr:
        center_lat = float(eprtr_agg["Latitude"].mean())
        center_lon = float(eprtr_agg["Longitude"].mean())
    else:
        center_lat = float(np.mean(lat[base_mask]))
        center_lon = float(np.mean(lon[base_mask]))

    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=int(zoom_start),
        tiles="OpenStreetMap",
    )
    _add_extra_basemaps(fmap)
    add_cams_e_layers(
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
        )
    folium.LayerControl(collapsed=False).add_to(fmap)

    out_html.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out_html))


def main() -> None:
    root = _project_root()
    ap = argparse.ArgumentParser(
        description="CAMS GNFR E (Solvents): Greece map + area/point source counts and totals."
    )
    ap.add_argument("--nc", type=Path, default=DEFAULT_NC, help="CAMS NetCDF path")
    ap.add_argument("--country", default="GRC", help="ISO3 (default GRC)")
    ap.add_argument(
        "--all-countries",
        action="store_true",
        help=(
            "Ignore --country; use full CAMS domain (still GNFR E only). "
            "If the default E-PRTR CSV exists, the map shows CAMS GNFR E points and "
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
            "EPRTR_SectorCode for E-PRTR facilities (default: 9 = Other activities). "
            "Ignored if --eprtr-all-sectors."
        ),
    )
    ap.add_argument(
        "--eprtr-all-sectors",
        action="store_true",
        help="Include all EPRTR_SectorCode values (do not restrict to sector 9).",
    )
    ap.add_argument(
        "--map-points-only",
        action="store_true",
        help="Map: draw GNFR E point sources only (no area rectangles).",
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
        help="Output HTML (default: Solvents/outputs/cams_E_<ISO>_pollutant.html)",
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
    eprtr_sector_for_load: float | None = (
        None if args.eprtr_all_sectors else float(args.eprtr_sector_code)
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

    auto_eprtr_map = (
        args.all_countries and not args.cams_map and eprtr_csv_path.is_file()
    )
    use_eprtr_map = bool(args.eprtr_map) or auto_eprtr_map
    if args.eprtr_map and not eprtr_csv_path.is_file():
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
        bbox_t = tuple(args.bbox) if args.bbox else None
        sector_e = emis == IDX_E_SOLVENTS
        domain = _build_domain_mask(lon, lat, ci, country_1b, bbox_t)
        base = sector_e & domain
        if int(np.count_nonzero(base)) == 0:
            raise SystemExit("No GNFR E sources match country/bbox filter.")

        print(f"NetCDF: {nc_path.name}")
        print(
            f"Filter: GNFR E (emission_category_index={IDX_E_SOLVENTS}), "
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
            if use_eprtr_map:
                agg_eprtr = load_eprtr_facilities_aggregated(
                    eprtr_csv_path,
                    all_countries=args.all_countries,
                    country_arg=args.country,
                    pollutant=args.eprtr_pollutant,
                    sector_code=eprtr_sector_for_load,
                    bbox=bbox_t,
                )
                print(
                    f"E-PRTR: {len(agg_eprtr)} facilities (after filter) "
                    f"from {eprtr_csv_path}"
                )
                if eprtr_sector_for_load is not None:
                    label = (
                        " (Other activities)"
                        if abs(eprtr_sector_for_load - 9.0) < 1e-6
                        else ""
                    )
                    print(
                        f"  E-PRTR sector filter: EPRTR_SectorCode == "
                        f"{eprtr_sector_for_load:g}{label}"
                    )
                else:
                    print("  E-PRTR sector filter: (all sectors, --eprtr-all-sectors)")
                if auto_eprtr_map and not args.eprtr_map:
                    print(
                        "  (same Folium map as CAMS GNFR E points; "
                        "--cams-map to omit E-PRTR.)"
                    )

            out_map = args.out_map
            tag = "ALL" if args.all_countries else args.country.upper()
            if out_map is None:
                suffix = f"{pol}_eprtr" if use_eprtr_map else pol
                out_map = root / "Solvents" / "outputs" / f"cams_E_{tag}_{suffix}.html"
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
                eprtr_pollutant_label=args.eprtr_pollutant if use_eprtr_map else None,
            )
            if use_eprtr_map and agg_eprtr is not None and len(agg_eprtr) > 0:
                print(f"Interactive map (CAMS + E-PRTR): {out_map}")
            elif use_eprtr_map:
                print(f"Interactive map (CAMS; E-PRTR overlay empty): {out_map}")
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
