"""Nearest E-PRTR facility (sector 9, selected Annex I, post-2019) per CAMS GNFR E point, on CORINE grid."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from rasterio.transform import rowcol

from ..grid import resolve_path
from ..manifest import write_manifest
from ..progress_util import note

# Same country names as E-PRTR F1_4 `countryName` (align with Solvents aux map).
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

IDX_E_SOLVENTS = 5
IDX_POINT = 2


def _eprtr_country_name(iso3: str) -> str:
    u = iso3.strip().upper()
    if u not in _ISO3_TO_EPRTR_COUNTRY:
        raise ValueError(
            f"No E-PRTR countryName mapping for CAMS ISO3 {iso3!r}. "
            f"Extend _ISO3_TO_EPRTR_COUNTRY in solvents_eprtr_point_link.py."
        )
    return _ISO3_TO_EPRTR_COUNTRY[u]


def _country_index_1based(ds: xr.Dataset, iso3: str) -> int:
    raw = ds["country_id"].values
    codes: list[str] = []
    for x in raw:
        if isinstance(x, bytes):
            codes.append(x.decode("utf-8", "replace").strip())
        else:
            codes.append(str(x).strip())
    u = iso3.strip().upper()
    try:
        return codes.index(u) + 1
    except ValueError as exc:
        raise ValueError(
            f"Country {iso3!r} not in CAMS country_id (have {len(codes)} countries)."
        ) from exc


def _load_filtered_eprtr(
    csv_path: Path,
    *,
    country_name: str,
    pollutant: str,
    sector_code: float,
    annex_codes: frozenset[str],
    min_reporting_year_exclusive: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return deduped facilities (one row per FacilityInspireId) and filter-stage counts."""
    cols = [
        "countryName",
        "reportingYear",
        "EPRTR_SectorCode",
        "EPRTRAnnexIMainActivity",
        "FacilityInspireId",
        "Longitude",
        "Latitude",
        "TargetRelease",
        "Pollutant",
    ]
    df = pd.read_csv(csv_path, usecols=cols, low_memory=False)
    counts: dict[str, int] = {"raw_rows": int(len(df))}

    df = df[df["TargetRelease"].astype(str).str.strip().str.upper() == "AIR"]
    counts["after_air"] = int(len(df))
    df = df[df["Pollutant"] == pollutant]
    counts["after_pollutant"] = int(len(df))
    df = df[df["countryName"] == country_name]
    counts["after_country"] = int(len(df))

    sc = pd.to_numeric(df["EPRTR_SectorCode"], errors="coerce")
    df = df[np.isclose(sc, float(sector_code), rtol=0.0, atol=1e-9)]
    counts["after_sector"] = int(len(df))

    ann = df["EPRTRAnnexIMainActivity"].astype(str).str.strip()
    df = df[ann.isin(annex_codes)]
    counts["after_annex"] = int(len(df))

    df = df[df["reportingYear"] > int(min_reporting_year_exclusive)]
    counts["after_reporting_year"] = int(len(df))

    df = df[df["FacilityInspireId"].notna()]
    df = df[df["Longitude"].notna() & df["Latitude"].notna()]
    counts["after_coords"] = int(len(df))

    if df.empty:
        return df, counts

    df = df.sort_values(["FacilityInspireId", "reportingYear"], ascending=[True, False])
    dedup = df.groupby("FacilityInspireId", as_index=False).first()
    counts["facilities_deduped"] = int(len(dedup))
    return dedup, counts


def _load_cams_e_points(
    nc_path: Path,
    *,
    country_1based: int,
) -> tuple[np.ndarray, np.ndarray]:
    ds = xr.open_dataset(nc_path)
    try:
        emis = np.asarray(ds["emission_category_index"].values).ravel().astype(np.int64)
        st = np.asarray(ds["source_type_index"].values).ravel().astype(np.int64)
        ci = np.asarray(ds["country_index"].values).ravel().astype(np.int64)
        lon = np.asarray(ds["longitude_source"].values).ravel().astype(np.float64)
        lat = np.asarray(ds["latitude_source"].values).ravel().astype(np.float64)
        m = (
            (emis == IDX_E_SOLVENTS)
            & (st == IDX_POINT)
            & (ci == int(country_1based))
        )
        return lon[m], lat[m]
    finally:
        ds.close()


def build_solvents_eprtr_point_link(
    root: Path,
    cfg: dict[str, Any],
    ref: dict[str, Any],
    *,
    sector_entry: dict[str, Any],
) -> Path:
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise ImportError(
            "solvents_eprtr_point_link requires scipy (e.g. pip install scipy)."
        ) from exc

    paths = cfg["paths"]
    country = cfg["country"]
    show_progress = bool(cfg.get("show_progress", True))
    iso3 = str(country["cams_iso3"]).strip().upper()

    nc_path = resolve_path(root, paths["cams_nc"])
    if not nc_path.is_file():
        raise FileNotFoundError(f"CAMS NetCDF not found: {nc_path}")

    eprtr_csv = sector_entry.get("eprtr_csv", "data/E_PRTR/eptr_csv/F1_4_Air_Releases_Facilities.csv")
    eprtr_path = resolve_path(root, Path(eprtr_csv))
    if not eprtr_path.is_file():
        raise FileNotFoundError(f"E-PRTR CSV not found: {eprtr_path}")

    pollutant = str(
        sector_entry.get(
            "eprtr_pollutant",
            "Non-methane volatile organic compounds (NMVOC)",
        )
    )
    sector_code = float(sector_entry.get("eprtr_sector_code", 9))
    annex_list = sector_entry.get("eprtr_annex_codes", ["9(c)", "9(e)"])
    annex_codes = frozenset(str(x).strip() for x in annex_list)
    min_year_excl = int(sector_entry.get("eprtr_min_reporting_year_exclusive", 2019))

    country_name = _eprtr_country_name(iso3)
    fac, counts = _load_filtered_eprtr(
        eprtr_path,
        country_name=country_name,
        pollutant=pollutant,
        sector_code=sector_code,
        annex_codes=annex_codes,
        min_reporting_year_exclusive=min_year_excl,
    )

    if fac.empty:
        raise ValueError(
            "No E-PRTR facilities after filters. Stage counts: "
            + json.dumps(counts, indent=2)
        )

    ds = xr.open_dataset(nc_path)
    try:
        c1 = _country_index_1based(ds, iso3)
    finally:
        ds.close()

    lon_c, lat_c = _load_cams_e_points(nc_path, country_1based=c1)
    n_cams = int(lon_c.size)
    if n_cams == 0:
        raise ValueError(
            f"No CAMS GNFR E point sources for country_index={c1} ({iso3}) in {nc_path.name}."
        )

    crs = rasterio.crs.CRS.from_string(ref["crs"])
    h, w = int(ref["height"]), int(ref["width"])
    transform = ref["transform"]

    g_ep = gpd.GeoDataFrame(
        fac,
        geometry=gpd.points_from_xy(
            fac["Longitude"].astype(float),
            fac["Latitude"].astype(float),
            crs="EPSG:4326",
        ),
    ).to_crs(crs)
    ex = g_ep.geometry.x.to_numpy(dtype=np.float64)
    ey = g_ep.geometry.y.to_numpy(dtype=np.float64)
    tree = cKDTree(np.column_stack([ex, ey]))

    inspire_ids = fac["FacilityInspireId"].astype(str).tolist()
    n_fac = len(inspire_ids)
    id_map = {str(i + 1): inspire_ids[i] for i in range(n_fac)}

    g_cam = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(lon_c, lat_c, crs="EPSG:4326"),
    ).to_crs(crs)
    cx = g_cam.geometry.x.to_numpy(dtype=np.float64)
    cy = g_cam.geometry.y.to_numpy(dtype=np.float64)

    dist_m, idx = tree.query(np.column_stack([cx, cy]), k=1)
    fac_ids = (idx.astype(np.int64) + 1).astype(np.float32)
    dist_f = dist_m.astype(np.float32)

    band_ids = np.zeros((h, w), dtype=np.float32)
    band_dist = np.zeros((h, w), dtype=np.float32)
    collisions = 0

    for k in range(n_cams):
        x, y = float(cx[k]), float(cy[k])
        try:
            r, c = rowcol(transform, x, y)
        except Exception:
            continue
        if not (0 <= r < h and 0 <= c < w):
            continue
        if band_ids[r, c] != 0.0:
            collisions += 1
        band_ids[r, c] = float(fac_ids[k])
        band_dist[r, c] = float(dist_f[k])

    if collisions and show_progress:
        note(
            f"solvents_eprtr_point_link: {collisions} pixel collision(s); last CAMS point kept.",
            file=sys.stderr,
        )

    out_dir = root / "SourceProxies" / "Solvents_pointsourcetif"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = str(sector_entry.get("filename", "Solvents_E_cams_to_eprtr.tif"))
    if "/" in stem or "\\" in stem:
        raise ValueError("sector filename must be a single file name, not a path.")
    out_tif = out_dir / stem
    manifest_path = out_tif.with_suffix(".json")

    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 2,
        "dtype": "float32",
        "crs": ref["crs"],
        "transform": transform,
        "compress": "lzw",
    }
    if show_progress:
        note(f"solvents_eprtr_point_link: writing {out_tif.name} ({n_cams} CAMS → {n_fac} E-PRTR)…")
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(band_ids, 1)
        dst.write(band_dist, 2)
        dst.set_band_description(1, "eprtr_facility_id_float")
        dst.set_band_description(2, "distance_m_epsg3035")

    rel_out = out_tif
    try:
        rel_out = out_tif.relative_to(root)
    except ValueError:
        pass

    write_manifest(
        manifest_path,
        {
            "builder": "solvents_eprtr_point_link",
            "output_geotiff": str(rel_out),
            "band_1": "E-PRTR facility id (1..N) stored as float32; see facility_id_to_inspire_id",
            "band_2": "Distance in metres (EPSG:3035 planar) from CAMS point to matched facility",
            "crs": ref["crs"],
            "width": w,
            "height": h,
            "cams_nc": str(nc_path),
            "cams_iso3": iso3,
            "cams_country_index_1based": int(c1),
            "n_cams_e_points": n_cams,
            "eprtr_csv": str(eprtr_path),
            "eprtr_country_name": country_name,
            "eprtr_pollutant": pollutant,
            "eprtr_sector_code": sector_code,
            "eprtr_annex_codes": sorted(annex_codes),
            "eprtr_min_reporting_year_exclusive": min_year_excl,
            "eprtr_filter_stage_counts": counts,
            "n_eprtr_facilities": n_fac,
            "facility_id_to_inspire_id": id_map,
            "pixel_collisions_last_wins": int(collisions),
            "domain_bbox_wgs84": list(ref["domain_bbox_wgs84"]),
        },
    )
    return out_tif
