#!/usr/bin/env python3
"""
Analyze CAMS-REG-ANT (e.g. v8.1 TNO) for **commercial / residential-type** stationary
combustion. In this inventory’s GNFR split (see ``urbem_interface.emissions.prepare_cams``),
that is typically **GNFR C — Other stationary combustion** (not public power A, not
industry B). Road, shipping, etc. are other letters.

For each pollutant (1-D ``source`` variable in the NetCDF), the script reports:

  - **Area** emissions (``source_type_index == 1``): kg/year summed over selected domain
  - **Point** emissions (``source_type_index == 2``)
  - **Total**, and fractions area vs point

Use this to decide whether your downscaling pipeline should implement **area**, **point**,
or **both** for the chosen GNFR code.

Usage (from project root)::

  python PublicPower/auxiliaries/Residential/analyze_cams_residential_heating.py
  python PublicPower/auxiliaries/Residential/analyze_cams_residential_heating.py --gnfr C --country GRC
  python PublicPower/auxiliaries/Residential/analyze_cams_residential_heating.py --list-gnfr
  python PublicPower/auxiliaries/Residential/analyze_cams_residential_heating.py --csv-out Residential/cams_C_GRC_summary.csv

Requires: xarray, netCDF4, numpy, pandas
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# Same order as urbem_interface.emissions.prepare_cams.GNFR_CODES (index 1-based in NetCDF)
GNFR_CODES: tuple[str, ...] = (
    "A",
    "B",
    "C",
    "D",
    "E",
    "F1",
    "F2",
    "F3",
    "F4",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
)

GNFR_FULL_NAMES: tuple[str, ...] = (
    "A_PublicPower",
    "B_Industry",
    "C_OtherStationaryComb",
    "D_Fugitives",
    "E_Solvents",
    "F1_RoadTransport_Exhaust_Gasoline",
    "F2_RoadTransport_Exhaust_Diesel",
    "F3_RoadTransport_Exhaust_LPG_gas",
    "F4_RoadTransport_NonExhaust",
    "G_Shipping",
    "H_Aviation",
    "I_OffRoad",
    "J_Waste",
    "K_AgriLivestock",
    "L_AgriOther",
)

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
    return Path(__file__).resolve().parents[3]


def default_netcdf(root: Path) -> Path:
    return (
        root
        / "data"
        / "given_CAMS"
        / "CAMS-REG-ANT_v8.1_TNO_ftp"
        / "netcdf"
        / "CAMS-REG-v8_1_emissions_year2019.nc"
    )


def gnfr_to_emission_category_index(code: str) -> int:
    c = code.strip().upper()
    if c == "F":
        raise SystemExit("Use F1, F2, F3, or F4 (not F alone).")
    try:
        return GNFR_CODES.index(c) + 1
    except ValueError as exc:
        raise SystemExit(
            f"Unknown GNFR code {code!r}. Valid: {', '.join(GNFR_CODES)}"
        ) from exc


def _pollutant_vars(ds: xr.Dataset) -> list[str]:
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


def _decode_country_ids(ds: xr.Dataset) -> list[str]:
    raw = ds["country_id"].values
    out: list[str] = []
    for x in raw:
        if isinstance(x, bytes):
            out.append(x.decode("utf-8", "replace").strip())
        else:
            out.append(str(x).strip())
    return out


def _country_index_1based(ds: xr.Dataset, iso3: str) -> int:
    codes = _decode_country_ids(ds)
    u = iso3.strip().upper()
    try:
        return codes.index(u) + 1
    except ValueError as exc:
        raise SystemExit(
            f"Country {iso3!r} not in NetCDF country_id ({len(codes)} countries)."
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


def _fmt_kg(x: float) -> str:
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


def analyze(
    ds: xr.Dataset,
    *,
    emission_category_1based: int,
    domain_mask: np.ndarray,
    pollutant_filter: set[str] | None,
) -> tuple[pd.DataFrame, dict[str, str], dict[str, int]]:
    emis = np.asarray(ds["emission_category_index"].values).ravel().astype(np.int64)
    st = np.asarray(ds["source_type_index"].values).ravel().astype(np.int64)
    ci = np.asarray(ds["country_index"].values).ravel().astype(np.int64)

    sector_m = domain_mask & (emis == int(emission_category_1based))
    area_m = sector_m & (st == 1)
    point_m = sector_m & (st == 2)

    n_area = int(np.count_nonzero(area_m))
    n_point = int(np.count_nonzero(point_m))
    n_sector = int(np.count_nonzero(sector_m))

    pollutants = _pollutant_vars(ds)
    if pollutant_filter is not None:
        pollutants = [p for p in pollutants if p.lower() in pollutant_filter]

    units: dict[str, str] = {}
    rows: list[dict[str, object]] = []
    for name in pollutants:
        arr = np.asarray(ds[name].values).ravel().astype(np.float64)
        if arr.size != domain_mask.size:
            continue
        ta = float(np.nansum(arr[area_m]))
        tp = float(np.nansum(arr[point_m]))
        tt = ta + tp
        fa = (ta / tt) if tt > 0 else float("nan")
        fp = (tp / tt) if tt > 0 else float("nan")
        u = ds[name].attrs.get("units", "kg/year")
        lu = ds[name].attrs.get("long_units", "")
        units[name] = f"{u}" + (f" ({lu})" if lu else "")
        rows.append(
            {
                "pollutant": name,
                "area_kg_yr": ta,
                "point_kg_yr": tp,
                "total_kg_yr": tt,
                "frac_area": fa,
                "frac_point": fp,
            }
        )

    meta = {
        "n_sources_sector": n_sector,
        "n_sources_area": n_area,
        "n_sources_point": n_point,
    }
    df = pd.DataFrame(rows)
    return df, units, meta


def main() -> None:
    root = _project_root()
    ap = argparse.ArgumentParser(
        description="CAMS area vs point emissions for a GNFR sector (default C = other stationary combustion).",
    )
    ap.add_argument(
        "--nc",
        type=Path,
        default=None,
        help="CAMS NetCDF path (default: data/given_CAMS/.../CAMS-REG-v8_1_emissions_year2019.nc under project root)",
    )
    ap.add_argument(
        "--gnfr",
        type=str,
        default="C",
        help=f"GNFR code (default C = residential/commercial-type stationary). One of: {','.join(GNFR_CODES)}",
    )
    ap.add_argument(
        "--country",
        type=str,
        default="GRC",
        help="ISO3 country filter (default GRC). Use ALL with --all-countries.",
    )
    ap.add_argument(
        "--all-countries",
        action="store_true",
        help="Ignore --country; use full NetCDF domain (still filter by GNFR).",
    )
    ap.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("LON0", "LAT0", "LON1", "LAT1"),
        default=None,
        help="Optional WGS84 bounding box (min_lon min_lat max_lon max_lat)",
    )
    ap.add_argument(
        "--pollutant",
        action="append",
        default=None,
        metavar="NAME",
        help="Restrict to pollutant variable(s) in the file (repeatable), e.g. --pollutant nox --pollutant pm2_5",
    )
    ap.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Write summary table CSV (relative paths from project root)",
    )
    ap.add_argument(
        "--list-gnfr",
        action="store_true",
        help="Print GNFR code ↔ emission_category_index ↔ name and exit",
    )
    args = ap.parse_args()

    if args.list_gnfr:
        print("emission_category_index (1-based) | GNFR | full_name")
        for i, (code, name) in enumerate(zip(GNFR_CODES, GNFR_FULL_NAMES, strict=True)):
            print(f"  {i + 1:2d}  | {code:3s} | {name}")
        print(
            "\nNote: For commercial/residential **heating** in this CAMS GNFR split, "
            "the usual bucket is **C (OtherStationaryComb)** - confirm for your policy "
            "if you also need B (Industry) or splits inside the NetCDF metadata."
        )
        return

    nc_path = args.nc
    if nc_path is None:
        nc_path = default_netcdf(root)
    elif not nc_path.is_absolute():
        nc_path = root / nc_path
    if not nc_path.is_file():
        raise SystemExit(f"NetCDF not found: {nc_path}")

    emis_cat = gnfr_to_emission_category_index(args.gnfr)
    gnfr_name = GNFR_FULL_NAMES[emis_cat - 1]

    pol_filter: set[str] | None = None
    if args.pollutant:
        pol_filter = {str(p).strip().lower() for p in args.pollutant}

    ds = xr.open_dataset(nc_path)
    try:
        lon = np.asarray(ds["longitude_source"].values).ravel().astype(np.float64)
        lat = np.asarray(ds["latitude_source"].values).ravel().astype(np.float64)
        country_idx = np.asarray(ds["country_index"].values).ravel().astype(np.int64)
        c1: int | None = None
        if not args.all_countries:
            c1 = _country_index_1based(ds, str(args.country))
        bbox = tuple(args.bbox) if args.bbox is not None else None
        domain_mask = _build_domain_mask(lon, lat, country_idx, c1, bbox)

        df, units, meta = analyze(
            ds,
            emission_category_1based=emis_cat,
            domain_mask=domain_mask,
            pollutant_filter=pol_filter,
        )
    finally:
        ds.close()

    scope = "all countries" if args.all_countries else str(args.country).strip().upper()
    if bbox is not None:
        scope += f" bbox={bbox}"

    print(f"CAMS file: {nc_path.name}")
    print(f"GNFR {args.gnfr.strip().upper()} (emission_category_index={emis_cat}) - {gnfr_name}")
    print(f"Domain: {scope}")
    print(
        f"Sources in sector (after domain filter): {meta['n_sources_sector']} "
        f"(area={meta['n_sources_area']}, point={meta['n_sources_point']})"
    )
    print()

    if df.empty:
        print("No pollutant rows (check --pollutant names or NetCDF contents).")
        return

    display = df.copy()
    display["area_kg_yr"] = display["area_kg_yr"].map(_fmt_kg)
    display["point_kg_yr"] = display["point_kg_yr"].map(_fmt_kg)
    display["total_kg_yr"] = display["total_kg_yr"].map(_fmt_kg)
    display["frac_area"] = display["frac_area"].map(lambda x: f"{100.0 * x:.2f}%" if math.isfinite(x) else "")
    display["frac_point"] = display["frac_point"].map(lambda x: f"{100.0 * x:.2f}%" if math.isfinite(x) else "")
    print(display.to_string(index=False))
    print()
    for name in df["pollutant"]:
        if name in units:
            print(f"  {name}: units = {units[name]}")

    if args.csv_out is not None:
        out_p = args.csv_out if args.csv_out.is_absolute() else root / args.csv_out
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_p, index=False)
        print(f"Wrote {out_p}")

    print(
        "\nInterpretation: if **frac_area** dominates, plan an **area** downscaling proxy "
        "(e.g. population/buildings raster); if **frac_point** is material, add or prioritize "
        "**point** sources (gridded points or explicit locations)."
    )


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
