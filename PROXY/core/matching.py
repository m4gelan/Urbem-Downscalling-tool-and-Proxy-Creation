"""CAMS point ↔ facility matching.

Facility **candidate pools** (``sector_cfg["point_matching"]["facility_pool"]``):

- **``eprtr``** (default) — ``paths.yaml`` → ``eprtr.facilities`` (EU EPRTR air releases).
- **``jrc_public_power``** — ``paths.yaml`` → ``proxy_specific.public_power.units_csv``
  (JRC Open Power Plants), combustion units only; country from ``jrc_country_name`` or
  ``cams_country_iso3`` mapping.
- **``uw_wtd_treatment_plants``** — UWWTD treatment plant GPKG (``proxy_specific.waste``),
  NUTS-filtered with ``nuts_cntr``.
- **``aviation``** — ``paths.yaml`` → ``osm.aviation`` (aerodrome polygons + optional nodes);
  built by :func:`PROXY.sectors.H_Aviation.aviation_matching.build_aviation_facility_candidates`.

**Distance cap:** ``max_match_distance_km`` in ``PROXY/config/eprtr_scoring.yaml`` under
``scoring`` (global) or ``sector_profiles.<sector>`` / ``point_matching.max_match_distance_km``
(sector YAML overrides scoring profile, which overrides global).

**EPRTR activity-sector filter:** ``preferred_eprtr_sector_codes`` (E-PRTR main activity sector
integers 1–9) — set ``point_matching.preferred_eprtr_sector_codes`` in the sector YAML to
override ``sector_profiles.<sector>`` in ``eprtr_scoring.yaml``. See
:func:`resolve_eprtr_sector_preferences`.

CAMS emissions: ``paths.yaml`` → ``emissions.cams_2019_nc``; sector YAML selects GNFR rows.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from PROXY.core.dataloaders.config import load_yaml
from PROXY.core.io import write_json

# JRC Open Power: combustion ``type_g`` values (excludes hydro / non-thermal for public-power point match).
_JRC_COMBUSTION_TYPE_G: frozenset[str] = frozenset(
    {
        "Fossil Hard coal",
        "Fossil Brown coal/Lignite",
        "Fossil Peat",
        "Fossil Oil",
        "Fossil Oil shale",
        "Fossil Gas",
        "Fossil Coal-derived gas",
        "Biomass",
        "Waste",
    }
)

_CAMS_ISO3_TO_JRC_COUNTRY: dict[str, str] = {
    "GRC": "Greece",
    "ALB": "Albania",
    "ITA": "Italy",
    "ESP": "Spain",
    "DEU": "Germany",
    "FRA": "France",
}


def _source_dataset_label(registry: str) -> str:
    r = str(registry or "EPRTR").strip().upper()
    if r == "UWWTD":
        return "UWWTD_UWWTP"
    if r == "JRC":
        return "JRC_OPEN_UNITS"
    if r.startswith("OSM_AVIATION"):
        return "OSM_AVIATION"
    return "EPRTR"


@dataclass(frozen=True)
class MatchRequest:
    sector: str
    year: int = 2019
    pollutant: str | None = None
    max_points: int | None = None
    cams_iso3: str | None = None


def _haversine_km(
    lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray
) -> np.ndarray:
    r = 6371.0
    lon1r = np.radians(lon1)
    lat1r = np.radians(lat1)
    lon2r = np.radians(lon2)
    lat2r = np.radians(lat2)
    dlon = lon2r - lon1r
    dlat = lat2r - lat1r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    return 2.0 * r * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def _extract_cams_points(
    cams_nc: Path,
    year: int,
    *,
    pollutant_var: str | list[str] | None,
    emission_category_indices: list[int] | None,
    source_type_indices: list[int] | None,
    max_points: int | None,
    cams_iso3: str | None,
) -> pd.DataFrame:
    with xr.open_dataset(cams_nc, engine="netcdf4") as ds:
        required = ["longitude_source", "latitude_source"]
        for var in required:
            if var not in ds:
                raise ValueError(f"Missing CAMS variable {var} in {cams_nc}")
        out = pd.DataFrame(
            {
                "cams_point_id": np.arange(ds["longitude_source"].values.size, dtype=np.int64),
                "longitude": np.asarray(ds["longitude_source"].values).ravel().astype(float),
                "latitude": np.asarray(ds["latitude_source"].values).ravel().astype(float),
            }
        )
        if "source_type_index" in ds:
            out["source_type_index"] = (
                np.asarray(ds["source_type_index"].values).ravel().astype(np.int64)
            )
        if "emission_category_index" in ds:
            out["emission_category_index"] = (
                np.asarray(ds["emission_category_index"].values).ravel().astype(np.int64)
            )
        if "country_index" in ds:
            out["country_index"] = np.asarray(ds["country_index"].values).ravel().astype(np.int64)
        cams_country_index: int | None = None
        if cams_iso3 and "country_id" in ds:
            ids = [str(x.decode("utf-8") if isinstance(x, bytes) else x).strip().upper() for x in ds["country_id"].values]
            try:
                cams_country_index = ids.index(cams_iso3.strip().upper()) + 1
            except ValueError:
                cams_country_index = None
        n = int(ds["longitude_source"].values.size)
        if isinstance(pollutant_var, (list, tuple)):
            acc = np.zeros(n, dtype=np.float64)
            for v in pollutant_var:
                vs = str(v).strip()
                if vs in ds.data_vars:
                    acc += np.asarray(ds[vs].values).ravel().astype(np.float64)
            out["cams_pollutant_value"] = acc
        elif pollutant_var and str(pollutant_var) == "co2_total":
            acc = np.zeros(n, dtype=np.float64)
            for vx in ("co2_ff", "co2_bf"):
                if vx in ds.data_vars:
                    acc += np.asarray(ds[vx].values).ravel().astype(np.float64)
            out["cams_pollutant_value"] = acc
        elif pollutant_var and str(pollutant_var) in ds.data_vars:
            out["cams_pollutant_value"] = (
                np.asarray(ds[str(pollutant_var)].values).ravel().astype(np.float64)
            )
        else:
            out["cams_pollutant_value"] = np.nan
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["longitude", "latitude"])
    if emission_category_indices and "emission_category_index" in out.columns:
        out = out[out["emission_category_index"].isin([int(x) for x in emission_category_indices])]
    if source_type_indices and "source_type_index" in out.columns:
        out = out[out["source_type_index"].isin([int(x) for x in source_type_indices])]
    if cams_iso3 and "country_index" in out.columns and cams_country_index is not None:
        out = out[out["country_index"] == int(cams_country_index)].copy()
    out["cams_pollutant_value"] = pd.to_numeric(out["cams_pollutant_value"], errors="coerce").fillna(0.0)
    pv_any = pollutant_var is not None and (
        not isinstance(pollutant_var, (list, tuple)) or len(pollutant_var) > 0
    )
    if pv_any:
        out = out[out["cams_pollutant_value"] > 0.0].copy()
    if max_points is not None and len(out) > int(max_points):
        out = out.nlargest(int(max_points), columns=["cams_pollutant_value"]).copy()
    out["year"] = year
    return out


def _prepare_eprtr(eprtr_csv: Path, year: int) -> pd.DataFrame:
    e = pd.read_csv(eprtr_csv)
    cols = {c.lower(): c for c in e.columns}
    lon_col = cols.get("longitude", "Longitude")
    lat_col = cols.get("latitude", "Latitude")
    year_col = cols.get("reportingyear", "reportingYear")
    rel_col = cols.get("targetrelease", "TargetRelease")
    pol_col = cols.get("pollutant", "Pollutant")
    sec_col = cols.get("eprtr_sectorcode", "EPRTR_SectorCode")
    fac_col = cols.get("facilityinspireid", "FacilityInspireId")
    name_col = cols.get("facilityname", "facilityName")

    e[lon_col] = pd.to_numeric(e[lon_col], errors="coerce")
    e[lat_col] = pd.to_numeric(e[lat_col], errors="coerce")
    e[year_col] = pd.to_numeric(e[year_col], errors="coerce")
    e = e.dropna(subset=[lon_col, lat_col, year_col])
    e = e[e[rel_col].astype(str).str.upper().str.strip() == "AIR"].copy()
    e = e[e[year_col] <= year].copy()
    if e.empty:
        return pd.DataFrame(
            columns=[
                "facility_id",
                "facility_name",
                "eprtr_sector_code",
                "pollutant",
                "longitude",
                "latitude",
                "reporting_year",
            ]
        )
    e = (
        e.sort_values(year_col)
        .groupby([fac_col, pol_col], as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    e_out = pd.DataFrame(
        {
            "facility_id": e[fac_col].astype(str),
            "facility_name": e[name_col].astype(str),
            "eprtr_sector_code": pd.to_numeric(e[sec_col], errors="coerce"),
            "pollutant": e[pol_col].astype(str),
            "longitude": e[lon_col].astype(float),
            "latitude": e[lat_col].astype(float),
            "reporting_year": e[year_col].astype(int),
        }
    )
    e_out = e_out.dropna(subset=["eprtr_sector_code"])
    e_out = e_out.copy()
    e_out["_registry"] = "EPRTR"
    return e_out


def _prepare_uwwtd_treatment_plants(
    gpkg_path: Path,
    *,
    year: int,
    pollutant_label: str,
    nuts_prefix: str | None,
) -> pd.DataFrame:
    """Normalize UWWTD treatment plant points to the same columns as :func:`_prepare_eprtr`."""
    empty_cols = [
        "facility_id",
        "facility_name",
        "eprtr_sector_code",
        "pollutant",
        "longitude",
        "latitude",
        "reporting_year",
    ]
    if not gpkg_path.is_file():
        return pd.DataFrame(columns=empty_cols)
    gdf = gpd.read_file(gpkg_path)
    if gdf.empty:
        return pd.DataFrame(columns=empty_cols)
    prefix = str(nuts_prefix or "").strip().upper()
    if prefix and "uwwNUTS" in gdf.columns:
        nuts = gdf["uwwNUTS"].astype(str).str.upper()
        gdf = gdf[nuts.str.startswith(prefix)].copy()
    if gdf.empty:
        return pd.DataFrame(columns=empty_cols)
    if "uwwLatitude" in gdf.columns and "uwwLongitude" in gdf.columns:
        lat = pd.to_numeric(gdf["uwwLatitude"], errors="coerce")
        lon = pd.to_numeric(gdf["uwwLongitude"], errors="coerce")
    else:
        cen = gdf.geometry.centroid
        lat = pd.to_numeric(cen.y, errors="coerce")
        lon = pd.to_numeric(cen.x, errors="coerce")
    id_col = "uwwCode" if "uwwCode" in gdf.columns else "OBJECTID"
    name_col = "uwwName" if "uwwName" in gdf.columns else id_col
    pol = str(pollutant_label or "CH4").strip() or "CH4"
    e_out = pd.DataFrame(
        {
            "facility_id": gdf[id_col].astype(str),
            "facility_name": gdf[name_col].astype(str),
            "eprtr_sector_code": np.full(len(gdf), 5, dtype=np.float64),
            "pollutant": pol,
            "longitude": lon.astype(float),
            "latitude": lat.astype(float),
            "reporting_year": np.full(len(gdf), int(year), dtype=np.int64),
            "_registry": np.full(len(gdf), "UWWTD", dtype=object),
        }
    )
    return e_out.dropna(subset=["longitude", "latitude"])


def _normalize_jrc_type_g(raw: object) -> str:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return ""
    s = str(raw).strip()
    if not s:
        return ""
    for c in _JRC_COMBUSTION_TYPE_G:
        if s.lower() == c.lower():
            return c
    return s


def _prepare_jrc_public_power_units(
    csv_path: Path,
    *,
    year: int,
    pollutant_label: str,
    jrc_country: str | None,
    exclude_hydro: bool,
) -> pd.DataFrame:
    """JRC Open Power units as facility rows (combustion-only, optional country filter)."""
    empty_cols = [
        "facility_id",
        "facility_name",
        "eprtr_sector_code",
        "pollutant",
        "longitude",
        "latitude",
        "reporting_year",
        "_registry",
    ]
    if not csv_path.is_file():
        return pd.DataFrame(columns=empty_cols)
    df = pd.read_csv(csv_path, low_memory=False)
    if df.empty or "lat" not in df.columns or "lon" not in df.columns:
        return pd.DataFrame(columns=empty_cols)
    country_f = str(jrc_country or "").strip()
    if country_f and "country" in df.columns:
        df = df[df["country"].astype(str).str.strip().eq(country_f)].copy()
    if exclude_hydro and "type_g" in df.columns:
        tg = df["type_g"].map(_normalize_jrc_type_g)
        df = df[tg.isin(_JRC_COMBUSTION_TYPE_G)].copy()
    lat = pd.to_numeric(df["lat"], errors="coerce")
    lon = pd.to_numeric(df["lon"], errors="coerce")
    id_col = "eic_g" if "eic_g" in df.columns else "eic_p"
    name_col = "name_g" if "name_g" in df.columns else "name_p"
    pol = str(pollutant_label or "NOx").strip() or "NOx"
    e_out = pd.DataFrame(
        {
            "facility_id": df[id_col].astype(str),
            "facility_name": df[name_col].astype(str),
            "eprtr_sector_code": np.full(len(df), 1, dtype=np.float64),
            "pollutant": pol,
            "longitude": lon.astype(float),
            "latitude": lat.astype(float),
            "reporting_year": np.full(len(df), int(year), dtype=np.int64),
            "_registry": np.full(len(df), "JRC", dtype=object),
        }
    )
    return e_out.dropna(subset=["longitude", "latitude", "facility_id"])


def _prepare_waste_eprtr_plus_uwwtd(
    *,
    eprtr_csv: Path,
    gpkg_path: Path,
    year: int,
    pollutant_label: str,
    nuts_prefix: str | None,
) -> pd.DataFrame:
    """EPRTR air facilities plus UWWTD plants (disjoint facility_id namespaces)."""
    a = _prepare_eprtr(eprtr_csv, year)
    b = _prepare_uwwtd_treatment_plants(
        gpkg_path, year=year, pollutant_label=pollutant_label, nuts_prefix=nuts_prefix
    )
    parts: list[pd.DataFrame] = []
    if not a.empty:
        a2 = a.copy()
        a2["facility_id"] = "EPRTR:" + a2["facility_id"].astype(str)
        a2["_registry"] = "EPRTR"
        parts.append(a2)
    if not b.empty:
        b2 = b.copy()
        b2["facility_id"] = "UWWTD:" + b2["facility_id"].astype(str)
        b2["_registry"] = "UWWTD"
        parts.append(b2)
    if not parts:
        return pd.DataFrame(
            columns=[
                "facility_id",
                "facility_name",
                "eprtr_sector_code",
                "pollutant",
                "longitude",
                "latitude",
                "reporting_year",
                "_registry",
            ]
        )
    out = pd.concat(parts, ignore_index=True)
    if "_registry" not in out.columns:
        out["_registry"] = "EPRTR"
    return out


def _resolve_max_match_distance_km(
    sector_key: str,
    scoring_cfg: dict[str, Any],
    sector_cfg: dict[str, Any],
) -> float:
    """Maximum great-circle distance (km) for a facility to be a match candidate."""
    pm = sector_cfg.get("point_matching") if isinstance(sector_cfg.get("point_matching"), dict) else {}
    if "max_match_distance_km" in pm:
        return max(0.5, float(pm["max_match_distance_km"]))
    profiles = scoring_cfg.get("sector_profiles", {}) or {}
    prof = profiles.get(sector_key, {}) or {}
    if "max_match_distance_km" in prof:
        return max(0.5, float(prof["max_match_distance_km"]))
    scoring = scoring_cfg.get("scoring", {}) or {}
    return max(0.5, float(scoring.get("max_match_distance_km", 30.0)))


def load_facility_candidates_for_sector(
    *,
    repo_root: Path,
    paths_resolved: dict[str, Any],
    sector_cfg: dict[str, Any],
    eprtr_csv_path: Path,
    year: int,
) -> tuple[pd.DataFrame, str, str]:
    """
    Return ``(facilities_df, facility_pool_id, resolved_source_path)`` for ``run_matching``.

    ``facility_pool_id`` identifies the pool (see module docstring).
    """
    pm = sector_cfg.get("point_matching") if isinstance(sector_cfg.get("point_matching"), dict) else {}
    pool = str(pm.get("facility_pool", "eprtr")).strip().lower()
    if pool in ("eprtr", "", "default", "air"):
        df = _prepare_eprtr(eprtr_csv_path, year)
        return df, "eprtr", str(eprtr_csv_path.resolve())
    if pool in ("uw_wtd_treatment_plants", "uwwtd", "wwtp", "uw_wtd"):
        psw = (paths_resolved.get("proxy_specific") or {}).get("waste") or {}
        override = pm.get("plants_gpkg") or pm.get("treatment_plants_gpkg")
        rel = override or psw.get("treatment_plants_gpkg")
        if not rel:
            raise ValueError(
                "point_matching.facility_pool=uw_wtd_treatment_plants but no "
                "treatment_plants_gpkg in paths.yaml proxy_specific.waste and no sector override."
            )
        gpkg = Path(rel) if Path(rel).is_absolute() else repo_root / rel
        nuts = sector_cfg.get("nuts_cntr") or sector_cfg.get("nuts_country") or "EL"
        pol = str(sector_cfg.get("pollutant_name") or "CH4")
        df = _prepare_uwwtd_treatment_plants(
            gpkg, year=year, pollutant_label=pol, nuts_prefix=str(nuts)
        )
        return df, "uw_wtd_treatment_plants", str(gpkg.resolve())
    if pool in ("jrc_public_power", "jrc", "jrc_open_units"):
        psp = (paths_resolved.get("proxy_specific") or {}).get("public_power") or {}
        rel = pm.get("jrc_units_csv") or psp.get("units_csv")
        if not rel:
            raise ValueError(
                "point_matching.facility_pool=jrc_public_power but no units_csv in "
                "paths.yaml proxy_specific.public_power and no sector override."
            )
        csv_path = Path(rel) if Path(rel).is_absolute() else repo_root / rel
        iso3 = str(sector_cfg.get("cams_country_iso3", "")).strip().upper()
        jrc_country = str(pm.get("jrc_country_name") or "").strip() or _CAMS_ISO3_TO_JRC_COUNTRY.get(
            iso3, ""
        )
        if not jrc_country:
            raise ValueError(
                "point_matching.facility_pool=jrc_public_power requires jrc_country_name in "
                "sector YAML or a known cams_country_iso3 entry in matching._CAMS_ISO3_TO_JRC_COUNTRY."
            )
        exclude_hydro = bool(pm.get("exclude_hydro", True))
        pol = str(sector_cfg.get("pollutant_name") or sector_cfg.get("pollutant_var") or "NOx")
        df = _prepare_jrc_public_power_units(
            csv_path,
            year=year,
            pollutant_label=pol,
            jrc_country=jrc_country,
            exclude_hydro=exclude_hydro,
        )
        return df, "jrc_public_power", str(csv_path.resolve())
    if pool in ("waste_eprtr_plus_uwwtd", "eprtr_uwwtd", "eprtr_plus_uwwtd"):
        psw = (paths_resolved.get("proxy_specific") or {}).get("waste") or {}
        override = pm.get("plants_gpkg") or pm.get("treatment_plants_gpkg")
        grel = override or psw.get("treatment_plants_gpkg")
        if not grel:
            raise ValueError(
                "point_matching.facility_pool=waste_eprtr_plus_uwwtd requires "
                "treatment_plants_gpkg (paths.yaml proxy_specific.waste or sector override)."
            )
        gpkg = Path(grel) if Path(grel).is_absolute() else repo_root / grel
        nuts = sector_cfg.get("nuts_cntr") or sector_cfg.get("nuts_country") or "EL"
        pol = str(sector_cfg.get("pollutant_name") or "CH4")
        df = _prepare_waste_eprtr_plus_uwwtd(
            eprtr_csv=eprtr_csv_path,
            gpkg_path=gpkg,
            year=year,
            pollutant_label=pol,
            nuts_prefix=str(nuts),
        )
        src = f"{Path(eprtr_csv_path).resolve()}|{gpkg.resolve()}"
        return df, "waste_eprtr_plus_uwwtd", src
    if pool in ("aviation", "osm_aviation"):
        from PROXY.sectors.H_Aviation.aviation_matching import build_aviation_facility_candidates

        df, src = build_aviation_facility_candidates(
            repo_root=repo_root,
            paths_resolved=paths_resolved,
            sector_cfg=sector_cfg,
        )
        return df, "aviation", src
    raise ValueError(
        f"Unknown point_matching.facility_pool={pool!r} "
        f"(eprtr, jrc_public_power, uw_wtd_treatment_plants, waste_eprtr_plus_uwwtd, aviation)."
    )


def _eprtr_sector_preferences_from_scoring(
    sector_key: str, scoring_cfg: dict[str, Any]
) -> set[int]:
    profiles = scoring_cfg.get("sector_profiles", {}) or {}
    prof = profiles.get(sector_key, {}) or {}
    prefs = prof.get("preferred_eprtr_sector_codes", []) or []
    return {int(x) for x in prefs}


def resolve_eprtr_sector_preferences(
    sector_key: str,
    scoring_cfg: dict[str, Any],
    sector_cfg: dict[str, Any] | None = None,
) -> set[int]:
    """E-PRTR main-activity sector codes that receive full activity score in point matching.

    Resolution (first match wins):

    1. Non-empty ``sector_cfg["point_matching"]["preferred_eprtr_sector_codes"]``.
    2. ``scoring_cfg["sector_profiles"][sector_key]["preferred_eprtr_sector_codes"]``.
    3. Empty set: every facility gets neutral activity weight (0.5), same as a missing profile.
    """
    sector_cfg = sector_cfg or {}
    pm = (
        sector_cfg.get("point_matching")
        if isinstance(sector_cfg.get("point_matching"), dict)
        else {}
    )
    override = pm.get("preferred_eprtr_sector_codes")
    if override is not None:
        raw = list(override) if isinstance(override, (list, tuple, set)) else []
        parsed: list[int] = []
        for x in raw:
            try:
                parsed.append(int(x))
            except (TypeError, ValueError):
                continue
        if parsed:
            return set(parsed)
    return _eprtr_sector_preferences_from_scoring(sector_key, scoring_cfg)


def _pollutant_match_score(
    *,
    desired_pollutant: str | None,
    eprtr_pollutant: str,
) -> float:
    if not desired_pollutant:
        return 0.5
    d = desired_pollutant.strip().upper()
    p = str(eprtr_pollutant).strip().upper()
    if d in p or p in d:
        return 1.0
    aliases = {
        "NOX": ["NITROGEN OXIDES"],
        "SOX": ["SULPHUR OXIDES", "SULFUR OXIDES"],
        "NMVOC": ["NON-METHANE VOLATILE ORGANIC COMPOUNDS"],
        "PM10": ["PARTICULATE MATTER"],
        "PM2_5": ["PARTICULATE MATTER"],
        "CH4": ["METHANE"],
    }
    for token in aliases.get(d, []):
        if token in p:
            return 0.9
    return 0.0


def _compute_match_qa(cams_count: int, out_df: pd.DataFrame) -> dict[str, Any]:
    matches = int(len(out_df))
    unmatched = int(max(cams_count - matches, 0))
    coverage = float(matches / cams_count) if cams_count > 0 else 0.0
    if out_df.empty:
        return {
            "cams_points_considered": cams_count,
            "matches": matches,
            "unmatched": unmatched,
            "coverage_rate": coverage,
            "distance_km_median": None,
            "distance_km_p90": None,
            "score_median": None,
            "score_p10": None,
            "score_p90": None,
        }
    dist = pd.to_numeric(out_df["distance_km"], errors="coerce").dropna()
    score = pd.to_numeric(out_df["score"], errors="coerce").dropna()
    return {
        "cams_points_considered": cams_count,
        "matches": matches,
        "unmatched": unmatched,
        "coverage_rate": coverage,
        "distance_km_median": None if dist.empty else float(dist.median()),
        "distance_km_p90": None if dist.empty else float(dist.quantile(0.90)),
        "score_median": None if score.empty else float(score.median()),
        "score_p10": None if score.empty else float(score.quantile(0.10)),
        "score_p90": None if score.empty else float(score.quantile(0.90)),
    }


def run_matching(
    *,
    request: MatchRequest,
    cams_nc_path: Path,
    eprtr_csv_path: Path,
    scoring_cfg_path: Path,
    output_dir: Path,
    sector_cfg: dict[str, Any] | None = None,
    link_ref_weights_tif: Path | None = None,
    facilities_df: pd.DataFrame | None = None,
    facility_pool_id: str = "eprtr",
    facility_source_path: str = "",
) -> dict:
    scoring_cfg = load_yaml(scoring_cfg_path)
    scoring = scoring_cfg.get("scoring", {}) or {}
    w_dist = float(scoring.get("distance_weight", 0.5))
    w_pol = float(scoring.get("pollutant_weight", 0.3))
    w_act = float(scoring.get("activity_weight", 0.2))
    min_score = float(scoring.get("min_score", 0.4))
    nearest_candidates = int(scoring.get("nearest_candidates", 25))
    fallback_min_score = float(scoring.get("fallback_min_score", 0.25))
    fallback_nearest_candidates = int(scoring.get("fallback_nearest_candidates", 60))
    fallback_w_dist = float(scoring.get("fallback_distance_weight", 0.75))
    fallback_w_pol = float(scoring.get("fallback_pollutant_weight", 0.15))
    fallback_w_act = float(scoring.get("fallback_activity_weight", 0.10))
    min_coverage_rate = float(scoring.get("min_coverage_rate", 0.20))

    sector_cfg = sector_cfg or {}
    pm = sector_cfg.get("point_matching") if isinstance(sector_cfg.get("point_matching"), dict) else {}
    pool_l = str(facility_pool_id or "").strip().lower()
    distance_only = bool(pm.get("distance_only")) or pool_l == "aviation"
    single_stage = bool(pm.get("single_match_stage")) or distance_only
    if distance_only:
        w_dist, w_pol, w_act = 1.0, 0.0, 0.0
        fallback_w_dist, fallback_w_pol, fallback_w_act = 1.0, 0.0, 0.0
        fallback_min_score = min_score
    nearest_candidates = int(pm.get("nearest_candidates", nearest_candidates))
    fallback_nearest_candidates = int(pm.get("fallback_nearest_candidates", fallback_nearest_candidates))
    if distance_only:
        fallback_nearest_candidates = nearest_candidates
    max_km = _resolve_max_match_distance_km(request.sector, scoring_cfg, sector_cfg)
    cams = _extract_cams_points(
        cams_nc_path,
        request.year,
        pollutant_var=sector_cfg.get("pollutant_var", request.pollutant),
        emission_category_indices=sector_cfg.get("cams_emission_category_indices"),
        source_type_indices=sector_cfg.get("cams_source_type_indices", [2]),
        max_points=request.max_points,
        cams_iso3=request.cams_iso3,
    )
    if facilities_df is None:
        eprtr = _prepare_eprtr(eprtr_csv_path, request.year)
        pool_id = "eprtr"
        src_path = str(Path(eprtr_csv_path).resolve())
    else:
        eprtr = facilities_df
        pool_id = str(facility_pool_id or "custom")
        src_path = str(facility_source_path or "")
    if cams.empty or eprtr.empty:
        output_dir.mkdir(parents=True, exist_ok=True)
        empty = pd.DataFrame(
            columns=[
                "match_id",
                "sector",
                "cams_point_id",
                "facility_id",
                "facility_name",
                "distance_km",
                "score",
                "score_distance",
                "score_pollutant",
                "score_activity",
            ]
        )
        empty.to_csv(output_dir / f"{request.sector}_point_matches_{request.year}.csv", index=False)
        write_json(
            output_dir / f"{request.sector}_point_matches_{request.year}.json",
            {
                "status": "ok",
                "sector": request.sector,
                "year": request.year,
                "matches": 0,
                "reason": "empty_cams_or_facilities",
                "facility_pool": pool_id,
                "facility_source": src_path,
                "max_match_distance_km": max_km,
                "preferred_eprtr_sector_codes": sorted(
                    resolve_eprtr_sector_preferences(
                        request.sector, scoring_cfg, sector_cfg
                    )
                ),
            },
        )
        return {
            "status": "ok",
            "matches": 0,
            "facility_pool": pool_id,
            "facility_source": src_path,
            "max_match_distance_km": max_km,
        }

    prefs = resolve_eprtr_sector_preferences(
        request.sector, scoring_cfg, sector_cfg
    )
    fac_lon = eprtr["longitude"].to_numpy(dtype=float)
    fac_lat = eprtr["latitude"].to_numpy(dtype=float)
    if "eprtr_sector_code" in eprtr.columns:
        fac_sector = pd.to_numeric(eprtr["eprtr_sector_code"], errors="coerce").fillna(0).astype(np.int64).to_numpy()
    else:
        fac_sector = np.zeros(len(eprtr), dtype=np.int64)

    # One physical facility (EPRTR facility_id) may match at most one CAMS point.
    # CAMS points are processed in descending pollutant load so higher emitters
    # keep the best available facility; others take next-best unused candidates.
    cams_sorted = cams.sort_values(
        ["cams_pollutant_value", "cams_point_id"],
        ascending=[False, True],
    ).reset_index(drop=True)
    assigned_facility_ids: set[str] = set()

    records: list[dict[str, Any]] = []
    fallback_used = 0
    stage_default_lbl = str(pm.get("match_stage_label", "aviation_single"))
    for row in cams_sorted.itertuples(index=False):
        d = _haversine_km(
            np.full(fac_lon.shape, float(row.longitude)),
            np.full(fac_lat.shape, float(row.latitude)),
            fac_lon,
            fac_lat,
        )
        # shortlist nearest candidates for speed
        cand_ix = np.argsort(d)[:fallback_nearest_candidates]
        ranked: list[dict[str, Any]] = []
        for ix in cand_ix:
            dist_km = float(d[ix])
            if dist_km > max_km:
                continue
            s_dist = 1.0 / (1.0 + dist_km / 10.0)
            s_pol = _pollutant_match_score(
                desired_pollutant=sector_cfg.get("pollutant_name", request.pollutant),
                eprtr_pollutant=str(eprtr.iloc[ix]["pollutant"]),
            )
            if int(fac_sector[ix]) in prefs:
                s_act = 1.0
            elif not prefs:
                s_act = 0.5
            else:
                s_act = 0.0
            reg = (
                str(eprtr.iloc[ix]["_registry"])
                if "_registry" in eprtr.columns
                else "EPRTR"
            )
            candidate = {
                "facility_id": str(eprtr.iloc[ix]["facility_id"]),
                "facility_name": str(eprtr.iloc[ix]["facility_name"]),
                "facility_longitude": float(eprtr.iloc[ix]["longitude"]),
                "facility_latitude": float(eprtr.iloc[ix]["latitude"]),
                "distance_km": dist_km,
                "score": float(w_dist * s_dist + w_pol * s_pol + w_act * s_act),
                "fallback_score": float(
                    fallback_w_dist * s_dist + fallback_w_pol * s_pol + fallback_w_act * s_act
                ),
                "score_distance": float(s_dist),
                "score_pollutant": float(s_pol),
                "score_activity": float(s_act),
                "eprtr_sector_code": int(fac_sector[ix]),
                "reporting_year": int(eprtr.iloc[ix]["reporting_year"]),
                "eprtr_pollutant": str(eprtr.iloc[ix]["pollutant"]),
                "registry": reg,
            }
            for _col in (
                "icao",
                "iata",
                "osm_source",
                "area_km2",
                "polygon_centroid_lon",
                "polygon_centroid_lat",
                "osm_element_type",
                "osm_numeric_id",
            ):
                if _col in eprtr.columns:
                    v = eprtr.iloc[ix][_col]
                    candidate[_col] = "" if pd.isna(v) else v
            ranked.append(candidate)

        best: dict[str, Any] | None = None
        strict_pool = sorted(
            [
                c
                for c in ranked[:nearest_candidates]
                if c["score"] >= min_score
                and c["facility_id"] not in assigned_facility_ids
            ],
            key=lambda c: (-float(c["score"]), float(c["distance_km"])),
        )
        if single_stage:
            best = strict_pool[0] if strict_pool else None
            if best is not None:
                best["match_stage"] = stage_default_lbl
        elif strict_pool:
            best = strict_pool[0]
            best["match_stage"] = "strict"
        else:
            fallback_pool = sorted(
                [
                    c
                    for c in ranked
                    if c["fallback_score"] >= fallback_min_score
                    and c["facility_id"] not in assigned_facility_ids
                ],
                key=lambda c: (-float(c["fallback_score"]), float(c["distance_km"])),
            )
            if fallback_pool:
                best = fallback_pool[0]
                best["score"] = float(best["fallback_score"])
                best["match_stage"] = "fallback"
                fallback_used += 1
        if best is None:
            continue
        assigned_facility_ids.add(str(best["facility_id"]))
        match_id = f"{request.sector}_{request.year}_{int(row.cams_point_id)}"
        rec: dict[str, Any] = {
            "match_id": match_id,
            "sector": request.sector,
            "year": request.year,
            "cams_point_id": int(row.cams_point_id),
            "cams_longitude": float(row.longitude),
            "cams_latitude": float(row.latitude),
            "cams_pollutant_value": float(getattr(row, "cams_pollutant_value", 0.0)),
        }
        rec.update(best)
        rec["source_dataset"] = _source_dataset_label(str(best.get("registry", "EPRTR")))
        rec["match_method"] = (
            "distance_only_facility_unique" if distance_only else "distance+activity_weighted_facility_unique"
        )
        if distance_only:
            rec["stage"] = str(rec.get("match_stage", stage_default_lbl))
            rec["match_stage"] = rec["stage"]
            rec["distance_m"] = float(best["distance_km"]) * 1000.0
            rec["pollutant_value"] = float(rec["cams_pollutant_value"])
            rec["cams_lon"] = rec["cams_longitude"]
            rec["cams_lat"] = rec["cams_latitude"]
            rec["match_lon"] = float(best["facility_longitude"])
            rec["match_lat"] = float(best["facility_latitude"])
            rec["name"] = str(best.get("facility_name", ""))
            rec["osm_id"] = str(best.get("facility_id", ""))
        records.append(rec)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / f"{request.sector}_point_matches_{request.year}.csv"
    out_json = output_dir / f"{request.sector}_point_matches_{request.year}.json"
    out_df = pd.DataFrame.from_records(records)
    out_df.to_csv(out_csv, index=False)
    unmatched_csv = output_dir / f"{request.sector}_point_matches_unmatched_{request.year}.csv"
    matched_ids: set[int] = set()
    if not out_df.empty and "cams_point_id" in out_df.columns:
        matched_ids = {int(x) for x in out_df["cams_point_id"].tolist()}
    um = cams_sorted[~cams_sorted["cams_point_id"].isin(matched_ids)].copy()
    um.to_csv(unmatched_csv, index=False)
    gpq_path = output_dir / f"{request.sector}_point_matches_{request.year}.parquet"
    try:
        if not out_df.empty:
            gdf = gpd.GeoDataFrame(
                out_df,
                geometry=gpd.points_from_xy(out_df["cams_longitude"], out_df["cams_latitude"]),
                crs="EPSG:4326",
            )
            gdf.to_parquet(gpq_path, index=False)
    except Exception:
        gpq_path = None
    write_json(
        out_json,
        {
            "status": "ok",
            "sector": request.sector,
            "year": request.year,
            "matches": int(len(out_df)),
            "fallback_matches": int(fallback_used),
            "facility_pool": pool_id,
            "facility_source": src_path,
            "max_match_distance_km": max_km,
            "preferred_eprtr_sector_codes": sorted(prefs),
            "columns": list(out_df.columns),
            "csv": str(out_csv),
            "unmatched_csv": str(unmatched_csv),
            "parquet": None if gpq_path is None else str(gpq_path),
        },
    )
    qa = _compute_match_qa(len(cams), out_df)
    qa["max_match_distance_km"] = float(max_km)
    qa["preferred_eprtr_sector_codes"] = sorted(prefs)
    qa["sector"] = request.sector
    qa["year"] = request.year
    qa["fallback_matches"] = int(fallback_used)
    qa["fallback_match_rate"] = (
        float(fallback_used / len(out_df)) if len(out_df) > 0 else 0.0
    )
    qa["meets_min_coverage"] = bool(qa["coverage_rate"] >= min_coverage_rate)
    qa["unmatched_csv"] = str(unmatched_csv)
    if distance_only and not out_df.empty and "distance_km" in out_df.columns:
        distv = pd.to_numeric(out_df["distance_km"], errors="coerce").dropna()
        if not distv.empty:
            qa["distance_km_min"] = float(distv.min())
            qa["distance_km_mean"] = float(distv.mean())
            qa["distance_km_max"] = float(distv.max())
            qa["distance_km_p95"] = float(distv.quantile(0.95))
        if "osm_source" in out_df.columns:
            qa["node_buffer_match_share"] = float(
                (out_df["osm_source"].astype(str) == "node_buffer").mean()
            )
        else:
            qa["node_buffer_match_share"] = 0.0
    qa_json = output_dir / f"{request.sector}_point_matches_qa_{request.year}.json"
    qa_csv = output_dir / f"{request.sector}_point_matches_qa_{request.year}.csv"
    write_json(qa_json, qa)
    pd.DataFrame([qa]).to_csv(qa_csv, index=False)
    qa_log = output_dir / f"{request.sector}_point_matches_{request.year}_qa.log"
    try:
        lines = [
            f"sector={request.sector} year={request.year}",
            f"facility_pool={pool_id}",
            f"cams_points_considered={qa.get('cams_points_considered')}",
            f"matches={qa.get('matches')} unmatched={qa.get('unmatched')} coverage={qa.get('coverage_rate')}",
            f"max_match_distance_km={max_km}",
        ]
        if distance_only:
            lines.extend(
                [
                    f"distance_km min/mean/median/max/p95="
                    f"{qa.get('distance_km_min')}/{qa.get('distance_km_mean')}/"
                    f"{qa.get('distance_km_median')}/{qa.get('distance_km_max')}/"
                    f"{qa.get('distance_km_p95')}",
                    f"node_buffer_match_share={qa.get('node_buffer_match_share')}",
                ]
            )
        lines.append(f"unmatched_csv={unmatched_csv}")
        qa_log.write_text("\n".join(str(x) for x in lines) + "\n", encoding="utf-8")
    except OSError:
        pass

    link_geotiff: str | None = None
    if (
        link_ref_weights_tif is not None
        and link_ref_weights_tif.is_file()
        and not out_df.empty
        and {"facility_longitude", "facility_latitude"}.issubset(out_df.columns)
    ):
        from PROXY.core.point_link_geotiff import write_cams_facility_link_geotiff

        link_fn = pm.get("link_geotiff_filename")
        link_path = (
            output_dir / str(link_fn)
            if link_fn
            else output_dir / f"{request.sector}_cams_facility_link_{request.year}.tif"
        )
        try:
            write_cams_facility_link_geotiff(
                matches_df=out_df,
                ref_weights_tif=link_ref_weights_tif,
                out_tif=link_path,
            )
            link_geotiff = str(link_path)
        except Exception:
            link_geotiff = None

    out: dict[str, Any] = {
        "status": "ok",
        "matches": int(len(out_df)),
        "coverage_rate": float(qa["coverage_rate"]),
        "fallback_matches": int(fallback_used),
        "facility_pool": pool_id,
        "facility_source": src_path,
        "max_match_distance_km": float(max_km),
        "preferred_eprtr_sector_codes": sorted(prefs),
        "csv": str(out_csv),
        "json": str(out_json),
        "qa_json": str(qa_json),
        "qa_csv": str(qa_csv),
        "parquet": None if gpq_path is None else str(gpq_path),
        "unmatched_csv": str(unmatched_csv),
    }
    if link_geotiff is not None:
        out["link_geotiff"] = link_geotiff
    if pool_l == "aviation":
        try:
            from PROXY.sectors.H_Aviation.aviation_matching import write_aviation_match_diagnostic_png

            png_p = write_aviation_match_diagnostic_png(
                matches_csv=out_csv,
                out_png=output_dir / f"{request.sector}_point_matches_{request.year}_map.png",
                title=f"{request.sector} CAMS↔OSM aerodrome matches ({request.year})",
            )
            if png_p is not None:
                out["diagnostic_png"] = str(png_p)
        except Exception:
            pass
    return out

