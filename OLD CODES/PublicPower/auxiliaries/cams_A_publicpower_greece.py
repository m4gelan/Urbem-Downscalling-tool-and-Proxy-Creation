#!/usr/bin/env python3
"""
Explore CAMS-REG-ANT v8.1 emissions for GNFR A (Public Power): compare area vs point
sources. Default domain is one ISO3 country (GRC). Use ``--all-countries`` for the
full NetCDF domain (still GNFR A only). Overlays JRC OPEN UNITS
(``data/PublicPower/JRC/JRC_OPEN_UNITS.csv``) for the matching country filter / bbox.
By default JRC rows with ``type_g`` starting with ``Hydro`` are dropped; use
``--jrc-include-hydro`` to keep them.

CAMS **area** sources are drawn as full grid-cell rectangles; **point** sources as
circle markers. With many area cells, ``--map-max-area-cells`` subsamples rectangles
for the HTML only (totals are always the full domain).

The inventory stores one row per ``source`` with:
  - emission_category_index 1..15  =>  A..L (A = Public Power)
  - source_type_index 1 = area, 2 = point
  - country_index 1-based into ``country_id`` / ``country_name``

Usage (from project root):
  python PublicPower/Auxiliaries/cams_A_publicpower_greece.py
  python PublicPower/Auxiliaries/cams_A_publicpower_greece.py --all-countries --no-map
  python PublicPower/Auxiliaries/cams_A_publicpower_greece.py --pollutant nox --out-map PublicPower/outputs/cams_A_GRC_nox.html
  python PublicPower/Auxiliaries/cams_A_publicpower_greece.py --country GRC --bbox 19.2 34.6 30.2 41.8

Requires: xarray, netCDF4, pandas; for maps: folium, branca (usually with folium)
"""

from __future__ import annotations

import argparse
import html
import math
import sys
from pathlib import Path

import numpy as np

import pandas as pd

# GNFR A = index 1 (1-based), matches urbem_interface.emissions.prepare_cams.GNFR_CODES
IDX_A_PUBLIC_POWER = 1

DEFAULT_NC = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "given_CAMS"
    / "CAMS-REG-ANT_v8.1_TNO_ftp"
    / "netcdf"
    / "CAMS-REG-v8_1_emissions_year2019.nc"
)

DEFAULT_JRC_UNITS_CSV = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "PublicPower"
    / "JRC"
    / "JRC_OPEN_UNITS.csv"
)

# JRC ``country`` column uses full English names, not ISO3.
ISO3_TO_JRC_COUNTRY_NAME: dict[str, str] = {
    "GRC": "Greece",
    "ALB": "Albania",
    "BEL": "Belgium",
    "AUT": "Austria",
    "BGR": "Bulgaria",
    "HRV": "Croatia",
    "CYP": "Cyprus",
    "CZE": "Czechia",
    "DEU": "Germany",
    "DNK": "Denmark",
    "ESP": "Spain",
    "EST": "Estonia",
    "FIN": "Finland",
    "FRA": "France",
    "HUN": "Hungary",
    "IRL": "Ireland",
    "ITA": "Italy",
    "LTU": "Lithuania",
    "LUX": "Luxembourg",
    "LVA": "Latvia",
    "MLT": "Malta",
    "NLD": "Netherlands",
    "POL": "Poland",
    "PRT": "Portugal",
    "ROU": "Romania",
    "SVK": "Slovakia",
    "SVN": "Slovenia",
    "SWE": "Sweden",
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


def _country_index_1based(ds, iso3: str) -> int:
    codes = _decode_country_ids(ds)
    u = iso3.strip().upper()
    try:
        return codes.index(u) + 1
    except ValueError as exc:
        raise SystemExit(f"Country {iso3!r} not in NetCDF country_id (have {len(codes)} countries).") from exc


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


def _jrc_country_name_for_cams_iso3(iso3: str, override: str | None) -> str | None:
    if override and override.strip():
        return override.strip()
    return ISO3_TO_JRC_COUNTRY_NAME.get(iso3.strip().upper())


def load_jrc_open_units(
    csv_path: Path,
    *,
    country_name: str | None,
    bbox: tuple[float, float, float, float] | None,
    exclude_hydro_types: bool = True,
) -> pd.DataFrame:
    """Rows from JRC_OPEN_UNITS; optional ``country`` filter and WGS84 bbox."""
    df = pd.read_csv(csv_path, dtype=str, low_memory=False)
    if "country" not in df.columns or "lat" not in df.columns or "lon" not in df.columns:
        raise SystemExit("JRC_OPEN_UNITS.csv missing expected columns (country, lat, lon).")
    if country_name is not None:
        df = df[df["country"].str.strip() == country_name].copy()
    else:
        df = df.copy()
    df["lat_f"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon_f"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat_f", "lon_f"])
    if bbox is not None:
        lon0, lat0, lon1, lat1 = bbox
        df = df[
            (df["lon_f"] >= lon0)
            & (df["lon_f"] <= lon1)
            & (df["lat_f"] >= lat0)
            & (df["lat_f"] <= lat1)
        ]
    df = df.reset_index(drop=True)
    if exclude_hydro_types and "type_g" in df.columns and len(df):
        tg = df["type_g"].fillna("").astype(str).str.strip()
        df = df[~tg.str.startswith("Hydro")].reset_index(drop=True)
    return df


def print_jrc_summary(
    df: pd.DataFrame,
    *,
    label: str,
    hydro_excluded: bool,
) -> None:
    print()
    print(f"JRC OPEN UNITS - {label}")
    if hydro_excluded:
        print(
            "  Filter: excluded type_g starting with 'Hydro' "
            "(reservoir, pumped storage, run-of-river, etc.); "
            "use --jrc-include-hydro to keep them."
        )
    print(f"  Rows (generating units): {len(df)}")
    if df.empty:
        return
    if "type_g" in df.columns:
        print("  By type_g:")
        for t, c in df["type_g"].value_counts().items():
            print(f"    {t}: {int(c)}")
    cap = pd.to_numeric(df.get("capacity_g", pd.Series(dtype=float)), errors="coerce")
    if cap.notna().any():
        print(f"  Sum capacity_g (MW, where numeric): {float(cap.sum(skipna=True)):.2f}")
    plant = df.get("name_p", pd.Series(dtype=str)).fillna("").astype(str).str.strip()
    n_plant = plant.replace("", pd.NA).dropna().nunique()
    if n_plant:
        print(f"  Distinct name_p (plants): {int(n_plant)}")


def print_comparison_table(
    totals: dict[str, tuple[float, float, float]],
    units: dict[str, str],
    *,
    title: str,
) -> None:
    print(title)
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


def add_cams_a_publicpower_layers_to_map(
    fmap,
    ds,
    base_mask: np.ndarray,
    pollutant: str,
    *,
    jrc_df: pd.DataFrame | None = None,
    jrc_layer_suffix: str = "",
    map_max_area_cells: int = 0,
    map_sample_seed: int = 42,
) -> None:
    """Append CAMS GNFR A and optional JRC vector layers to an existing Folium map.

    Coordinates are **not** reprojected for display: CAMS uses ``longitude_source`` /
    ``latitude_source`` from the NetCDF as WGS84 lon/lat; JRC uses ``lon`` / ``lat``
    from the CSV the same way. Small offsets between the two are expected (different
    inventories and location definitions, e.g. grid association vs plant geocode).
    """
    try:
        import folium
        from branca.colormap import LinearColormap
    except ImportError as exc:
        raise SystemExit(
            "folium (and branca) are required for maps. Install with: pip install folium"
        ) from exc

    lon = np.asarray(ds["longitude_source"].values).ravel()
    lat = np.asarray(ds["latitude_source"].values).ravel()
    st = np.asarray(ds["source_type_index"].values).ravel().astype(np.int64)
    vals = np.asarray(ds[pollutant].values).ravel().astype(np.float64)
    m_area = base_mask & (st == 1)
    m_point = base_mask & (st == 2)

    pos = vals > 0
    vpos = vals[pos & (m_area | m_point)]
    if vpos.size:
        vmin = float(np.min(vpos))
        vmax = float(np.max(vpos))
        if vmin >= vmax:
            vmax = vmin * 1.001 if vmin > 0 else 1.0
    else:
        vmin, vmax = 0.0, 1.0

    colors = ["#0d0887", "#6a00a8", "#b12a90", "#e16462", "#fca636", "#f0f921"]
    cmap = LinearColormap(colors, vmin=vmin, vmax=vmax, caption=f"{pollutant} (kg/year)")

    def radius_for(v: float, *, point: bool) -> float:
        if not math.isfinite(v) or v <= 0:
            r = 3.0
        else:
            t = math.sqrt(v / max(vmax, 1e-30))
            r = 4.0 + 18.0 * t
        return min(28.0, r) if point else min(22.0, r * 0.92)

    fg_area = folium.FeatureGroup(
        name="A PublicPower - area sources (grid cell)",
        show=True,
    )
    fg_point = folium.FeatureGroup(name="A PublicPower - point sources", show=True)

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
            f"(--map-max-area-cells); totals still use all cells."
        )

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

    for i in np.flatnonzero(m_point):
        v = float(vals[i])
        if not math.isfinite(v):
            continue
        color = cmap(v) if v > 0 else "#cccccc"
        r = radius_for(v, point=True)
        folium.CircleMarker(
            location=[float(lat[i]), float(lon[i])],
            radius=r,
            color="#333333",
            weight=2,
            fill=True,
            fill_color=color,
            fill_opacity=0.75,
            popup=folium.Popup(
                f"<b>{html.escape(pollutant)}</b><br/>{v:.6g} kg/yr<br/>point<br/>"
                f"lon {lon[i]:.4f}, lat {lat[i]:.4f}",
                max_width=260,
            ),
        ).add_to(fg_point)

    fg_area.add_to(fmap)
    fg_point.add_to(fmap)

    if jrc_df is not None and not jrc_df.empty:
        jrc_name = "JRC OPEN UNITS (generators)" + jrc_layer_suffix
        fg_jrc = folium.FeatureGroup(name=jrc_name, show=True)
        for _, row in jrc_df.iterrows():
            la = float(row["lat_f"])
            lo = float(row["lon_f"])
            name_p = html.escape(str(row.get("name_p", "") or ""))
            name_g = html.escape(str(row.get("name_g", "") or ""))
            type_g = html.escape(str(row.get("type_g", "") or ""))
            cap_g = html.escape(str(row.get("capacity_g", "") or ""))
            st_g = html.escape(str(row.get("status_g", "") or ""))
            eic_g = html.escape(str(row.get("eic_g", "") or ""))
            nuts = html.escape(str(row.get("NUTS2", "") or ""))
            pop_html = (
                f"<b>JRC OPEN UNITS</b><br/>"
                f"<b>Plant</b> {name_p}<br/>"
                f"<b>Unit</b> {name_g}<br/>"
                f"<b>type_g</b> {type_g}<br/>"
                f"<b>capacity_g</b> {cap_g} MW<br/>"
                f"<b>status_g</b> {st_g}<br/>"
                f"<b>NUTS2</b> {nuts}<br/>"
                f"<b>eic_g</b> {eic_g}<br/>"
                f"lon {lo:.4f}, lat {la:.4f}"
            )
            folium.CircleMarker(
                location=[la, lo],
                radius=6,
                color="#1a5f1a",
                weight=2,
                fill=True,
                fill_color="#8fd98f",
                fill_opacity=0.85,
                popup=folium.Popup(pop_html, max_width=320),
            ).add_to(fg_jrc)
        fg_jrc.add_to(fmap)

    cmap.add_to(fmap)


def build_folium_map(
    ds,
    base_mask: np.ndarray,
    pollutant: str,
    *,
    out_html: Path,
    jrc_df: pd.DataFrame | None = None,
    jrc_layer_suffix: str = "",
    map_center_latlon: tuple[float, float] | None = None,
    zoom_start: int = 7,
    map_max_area_cells: int = 0,
    map_sample_seed: int = 42,
) -> None:
    try:
        import folium
    except ImportError as exc:
        raise SystemExit(
            "folium (and branca) are required for --out-map. "
            "Install with: pip install folium"
        ) from exc

    lon = np.asarray(ds["longitude_source"].values).ravel()
    lat = np.asarray(ds["latitude_source"].values).ravel()
    if map_center_latlon is not None:
        center_lat, center_lon = map_center_latlon
    else:
        center_lat = float(np.mean(lat[base_mask]))
        center_lon = float(np.mean(lon[base_mask]))
    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=int(zoom_start),
        tiles="CartoDB positron",
    )
    add_cams_a_publicpower_layers_to_map(
        fmap,
        ds,
        base_mask,
        pollutant,
        jrc_df=jrc_df,
        jrc_layer_suffix=jrc_layer_suffix,
        map_max_area_cells=map_max_area_cells,
        map_sample_seed=map_sample_seed,
    )
    folium.LayerControl(collapsed=False).add_to(fmap)

    out_html.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out_html))


def main() -> None:
    root = _project_root()
    p = argparse.ArgumentParser(
        description="CAMS-REG v8.1: A_PublicPower emissions - area vs point totals and map."
    )
    p.add_argument(
        "--nc",
        type=Path,
        default=DEFAULT_NC,
        help="Path to CAMS-REG emissions NetCDF",
    )
    p.add_argument(
        "--all-countries",
        action="store_true",
        help="Use all countries in the NetCDF (GNFR A only); optional --bbox still applies",
    )
    p.add_argument(
        "--country",
        default="GRC",
        help="ISO3 country code when not using --all-countries (default GRC)",
    )
    p.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
        help="Optional WGS84 bounding box (lon/lat min then max)",
    )
    p.add_argument(
        "--pollutant",
        default="nox",
        help="Pollutant variable for the interactive map (default nox)",
    )
    p.add_argument(
        "--no-map",
        action="store_true",
        help="Skip Folium HTML (only print table / optional CSV)",
    )
    p.add_argument(
        "--out-map",
        type=Path,
        default=None,
        help="Write Folium HTML map (default: PublicPower/outputs/cams_A_<ISO|ALL>_<pollutant>.html)",
    )
    p.add_argument(
        "--map-max-area-cells",
        type=int,
        default=25_000,
        help="Max area rectangles drawn on the map (0 = all). Totals always use full domain.",
    )
    p.add_argument(
        "--map-sample-seed",
        type=int,
        default=42,
        help="RNG seed when subsampling area cells for the map",
    )
    p.add_argument(
        "--map-zoom",
        type=int,
        default=None,
        help="Initial map zoom (default: 4 if --all-countries else 7)",
    )
    p.add_argument(
        "--map-center-lat",
        type=float,
        default=None,
        help="Map centre latitude (use with --map-center-lon)",
    )
    p.add_argument(
        "--map-center-lon",
        type=float,
        default=None,
        help="Map centre longitude (use with --map-center-lat)",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Optional CSV path for area/point/total columns per pollutant",
    )
    p.add_argument(
        "--jrc-csv",
        type=Path,
        default=DEFAULT_JRC_UNITS_CSV,
        help="Path to JRC_OPEN_UNITS.csv",
    )
    p.add_argument(
        "--no-jrc",
        action="store_true",
        help="Do not load JRC OPEN UNITS or add them to the map",
    )
    p.add_argument(
        "--jrc-country",
        default=None,
        metavar="NAME",
        help="JRC country column: with --all-countries, optional filter (default: all rows). "
        "Single-country mode: overrides ISO3 inference from --country",
    )
    p.add_argument(
        "--jrc-include-hydro",
        action="store_true",
        help="Keep JRC units whose type_g starts with 'Hydro' (default: exclude them)",
    )
    args = p.parse_args()
    nc_path = args.nc if args.nc.is_absolute() else root / args.nc
    if not nc_path.is_file():
        raise SystemExit(f"NetCDF not found: {nc_path}")

    import xarray as xr

    ds = xr.open_dataset(nc_path)
    try:
        pollutants = _pollutant_vars(ds)
        if not pollutants:
            raise SystemExit("No pollutant variables with dimension 'source' found.")

        country_1b: int | None
        if args.all_countries:
            country_1b = None
        else:
            country_1b = _country_index_1based(ds, args.country)

        emis = np.asarray(ds["emission_category_index"].values).ravel().astype(np.int64)
        ci = np.asarray(ds["country_index"].values).ravel().astype(np.int64)
        lon = np.asarray(ds["longitude_source"].values).ravel()
        lat = np.asarray(ds["latitude_source"].values).ravel()

        bbox_t = tuple(args.bbox) if args.bbox else None
        sector_a = emis == IDX_A_PUBLIC_POWER
        domain = _build_domain_mask(lon, lat, ci, country_1b, bbox_t)
        base = sector_a & domain
        n = int(np.count_nonzero(base))
        if n == 0:
            raise SystemExit("No A_PublicPower sources match country/bbox filter.")

        n_area = int(np.count_nonzero(base & (np.asarray(ds["source_type_index"].values).ravel() == 1)))
        n_point = int(np.count_nonzero(base & (np.asarray(ds["source_type_index"].values).ravel() == 2)))

        hist = str(ds.attrs.get("history", nc_path.name))
        print(f"NetCDF: {nc_path.name}")
        print(f"History: {hist[:120]}{'...' if len(hist) > 120 else ''}")
        if args.all_countries:
            print(
                f"Filter: GNFR A (emission_category_index={IDX_A_PUBLIC_POWER}), "
                "all countries in file"
            )
        else:
            print(
                f"Filter: GNFR A (emission_category_index={IDX_A_PUBLIC_POWER}), "
                f"country {args.country.upper()} (country_index={country_1b})"
            )
        if args.bbox:
            print(f"  + bbox lon[{args.bbox[0]},{args.bbox[2]}] lat[{args.bbox[1]},{args.bbox[3]}]")
        print(f"Sources in domain: {n} (area {n_area}, point {n_point})")
        print()

        jrc_df: pd.DataFrame | None = None
        jrc_label = ""
        if not args.no_jrc:
            jrc_path = args.jrc_csv if args.jrc_csv.is_absolute() else root / args.jrc_csv
            if not jrc_path.is_file():
                raise SystemExit(f"JRC CSV not found: {jrc_path}")
            jrc_filter_country: str | None
            if args.all_countries:
                jc = (args.jrc_country or "").strip()
                jrc_filter_country = jc if jc else None
                jrc_label = (jc if jc else "all countries") + (" (bbox clip)" if bbox_t else "")
            else:
                jrc_filter_country = _jrc_country_name_for_cams_iso3(args.country, args.jrc_country)
                if jrc_filter_country is None:
                    print(
                        f"Skipping JRC: no country name for ISO3 {args.country!r}. "
                        f"Pass --jrc-country \"...\" (JRC ``country`` spelling) or add a mapping."
                    )
                else:
                    jrc_label = jrc_filter_country + (" (bbox clip)" if bbox_t else "")

            if args.all_countries or jrc_filter_country is not None:
                jrc_df = load_jrc_open_units(
                    jrc_path,
                    country_name=jrc_filter_country,
                    bbox=bbox_t,
                    exclude_hydro_types=not args.jrc_include_hydro,
                )
                print_jrc_summary(
                    jrc_df,
                    label=jrc_label,
                    hydro_excluded=not args.jrc_include_hydro,
                )

        totals, units = _summarize_pollutants(ds, base, pollutants)
        print_comparison_table(
            totals,
            units,
            title="Totals by pollutant: area vs point (same domain)",
        )

        pol = args.pollutant.strip().lower()
        if pol not in totals:
            raise SystemExit(
                f"Unknown pollutant {pol!r}. Examples: {', '.join(pollutants[:6])} ..."
            )

        if not args.no_map:
            out_map = args.out_map
            out_tag = "ALL" if args.all_countries else args.country.upper()
            if out_map is None:
                out_map = (
                    root
                    / "PublicPower"
                    / "outputs"
                    / f"cams_A_{out_tag}_{pol}.html"
                )
            else:
                out_map = out_map if out_map.is_absolute() else root / out_map

            mclat, mclon = args.map_center_lat, args.map_center_lon
            if (mclat is None) ^ (mclon is None):
                raise SystemExit("Set both --map-center-lat and --map-center-lon, or neither.")
            if mclat is not None:
                map_center: tuple[float, float] | None = (float(mclat), float(mclon))
            elif args.all_countries:
                map_center = (51.0, 14.0)
            else:
                map_center = None

            zoom_use = args.map_zoom if args.map_zoom is not None else (4 if args.all_countries else 7)
            max_area = max(0, int(args.map_max_area_cells))

            jrc_layer_suffix = (
                " - excl. Hydro"
                if jrc_df is not None
                and not jrc_df.empty
                and not args.jrc_include_hydro
                else ""
            )
            build_folium_map(
                ds,
                base,
                pol,
                out_html=out_map,
                jrc_df=jrc_df,
                jrc_layer_suffix=jrc_layer_suffix,
                map_center_latlon=map_center,
                zoom_start=int(zoom_use),
                map_max_area_cells=max_area,
                map_sample_seed=int(args.map_sample_seed),
            )
            print()
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
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            print(f"CSV: {csv_path}")
    finally:
        ds.close()


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
