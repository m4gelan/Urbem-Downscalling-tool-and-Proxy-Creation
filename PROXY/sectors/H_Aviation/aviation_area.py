"""GNFR H aviation **area-source** downscaling: binary aerodrome proxy on a fine grid.

Uses CAMS-REG area rows (``source_type_index == 1``) for GNFR H, OSM aerodrome polygons
from :func:`PROXY.sectors.H_Aviation.aviation_matching.build_aviation_aerodrome_pool_gdf`,
and per-CAMS-cell normalisation with a small floor when no airport pixels fall inside a cell.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from rasterio import features
from rasterio.enums import MergeAlg
from shapely.geometry import mapping

from PROXY.core.cams.domain import country_index_1based
from PROXY.core.cams.grid import build_cam_cell_id
from PROXY.core.cams.gnfr import gnfr_code_to_index
from PROXY.core.dataloaders import resolve_path
from PROXY.core.grid import nuts2_for_country, resolve_nuts_cntr_code
from PROXY.core.io import write_geotiff, write_json
from PROXY.core.raster.country_clip import cams_iso3_from_cli_country
from PROXY.sectors.H_Aviation.aviation_matching import build_aviation_aerodrome_pool_gdf

logger = logging.getLogger(__name__)


def _rel_to_repo(path: Path, repo_root: Path) -> str:
    rp = path.resolve()
    rr = repo_root.resolve()
    try:
        return str(rp.relative_to(rr))
    except ValueError:
        return str(rp)


_CAMS_ST_AREA = 1
_CAMS_ST_POINT = 2


def _lon_lat_indices_clipped(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray, int, int]:
    lon_b = np.asarray(ds["longitude_bounds"].values, dtype=np.float64)
    lat_b = np.asarray(ds["latitude_bounds"].values, dtype=np.float64)
    nlon = int(lon_b.shape[0])
    nlat = int(lat_b.shape[0])
    lon_idx = np.asarray(ds["longitude_index"].values).ravel().astype(np.int64)
    lat_idx = np.asarray(ds["latitude_index"].values).ravel().astype(np.int64)
    if lon_idx.max() >= nlon or lat_idx.max() >= nlat:
        lon_idx = np.maximum(0, lon_idx - 1)
        lat_idx = np.maximum(0, lat_idx - 1)
    lon_idx = np.clip(lon_idx, 0, nlon - 1)
    lat_idx = np.clip(lat_idx, 0, nlat - 1)
    return lon_idx, lat_idx, nlon, nlat


def _pollutant_vector(ds: xr.Dataset, cams_var: str) -> np.ndarray:
    """1-D CAMS source vector for one pollutant (or ``co2_total`` = ff + bf)."""
    n = int(np.asarray(ds["longitude_source"].values).ravel().size)
    key = str(cams_var).strip()
    if key.lower() == "co2_total":
        acc = np.zeros(n, dtype=np.float64)
        for vx in ("co2_ff", "co2_bf"):
            if vx in ds.variables:
                acc += np.asarray(ds[vx].values).ravel().astype(np.float64)
        return acc
    if key not in ds.variables:
        raise KeyError(key)
    return np.asarray(ds[key].values).ravel().astype(np.float64)


def _pollutant_units(ds: xr.Dataset, cams_var: str) -> str:
    key = str(cams_var).strip()
    if key.lower() == "co2_total":
        for vx in ("co2_ff", "co2_bf"):
            if vx in ds.variables:
                u = ds[vx].attrs.get("units")
                if u is not None:
                    return str(u)
        return "unknown"
    if key not in ds.variables:
        return "unknown"
    u = ds[key].attrs.get("units")
    return str(u) if u is not None else "unknown"


def _parse_pollutant_entries(raw: object) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in raw or []:
        if isinstance(item, dict):
            v = item.get("cams_var") or item.get("var") or item.get("variable")
            slug = item.get("output_slug") or item.get("slug") or v
        else:
            v = str(item).strip()
            slug = v
        if not v:
            continue
        slug_s = str(slug).strip().replace(" ", "_").replace("/", "_")
        rows.append({"cams_var": str(v).strip(), "slug": slug_s})
    return rows


def _area_point_totals(
    ds: xr.Dataset,
    *,
    gnfr_index: int,
    country_idx_1based: int,
    pollutant_vec: np.ndarray,
) -> tuple[float, float]:
    emis = np.asarray(ds["emission_category_index"].values).ravel().astype(np.int64)
    st = np.asarray(ds["source_type_index"].values).ravel().astype(np.int64)
    ci = np.asarray(ds["country_index"].values).ravel().astype(np.int64)
    m_area = (emis == gnfr_index) & (st == _CAMS_ST_AREA) & (ci == country_idx_1based)
    m_point = (emis == gnfr_index) & (st == _CAMS_ST_POINT) & (ci == country_idx_1based)
    v = np.asarray(pollutant_vec, dtype=np.float64)
    return float(v[m_area].sum()), float(v[m_point].sum())


def _mass_by_geo_cell_area(
    ds: xr.Dataset,
    *,
    gnfr_index: int,
    country_idx_1based: int,
    pollutant_vec: np.ndarray,
) -> dict[int, float]:
    emis = np.asarray(ds["emission_category_index"].values).ravel().astype(np.int64)
    st = np.asarray(ds["source_type_index"].values).ravel().astype(np.int64)
    ci = np.asarray(ds["country_index"].values).ravel().astype(np.int64)
    lon_idx, lat_idx, nlon, _nlat = _lon_lat_indices_clipped(ds)
    mask = (emis == gnfr_index) & (st == _CAMS_ST_AREA) & (ci == country_idx_1based)
    out: dict[int, float] = {}
    v = np.asarray(pollutant_vec, dtype=np.float64)
    n = int(mask.size)
    for i in range(n):
        if not mask[i]:
            continue
        val = float(v[i])
        if not math.isfinite(val) or val <= 0.0:
            continue
        cid = int(lat_idx[i] * nlon + lon_idx[i])
        out[cid] = out.get(cid, 0.0) + val
    return out


def _reference_profile_100m_nuts(
    nuts_gpkg: Path,
    nuts_country: str,
    *,
    pad_m: float,
    resolution_m: float,
) -> dict[str, Any]:
    """EPSG:3035 reference grid aligned to ``resolution_m`` cells covering NUTS2 union + pad."""
    from rasterio.transform import from_bounds

    res = float(resolution_m)
    n2 = nuts2_for_country(nuts_gpkg, nuts_country)
    union = n2.dissolve().geometry.iloc[0]
    if union is None or union.is_empty:
        raise ValueError("NUTS union is empty")
    g = gpd.GeoDataFrame(geometry=[union], crs=n2.crs).to_crs("EPSG:3035")
    minx, miny, maxx, maxy = g.geometry.iloc[0].bounds
    minx -= float(pad_m)
    miny -= float(pad_m)
    maxx += float(pad_m)
    maxy += float(pad_m)
    west = math.floor(minx / res) * res
    south = math.floor(miny / res) * res
    east = math.ceil(maxx / res) * res
    north = math.ceil(maxy / res) * res
    width = max(1, int(round((east - west) / res)))
    height = max(1, int(round((north - south) / res)))
    east = west + width * res
    south = north - height * res
    transform = from_bounds(west, south, east, north, width, height)
    return {
        "transform": transform,
        "height": height,
        "width": width,
        "crs": "EPSG:3035",
        "bounds_3035": (west, south, east, north),
    }


def _rasterize_binary_proxy(
    pool: gpd.GeoDataFrame,
    ref: dict[str, Any],
) -> np.ndarray:
    """Binary {0,1} aerodrome presence (merge-add overlaps, then threshold)."""
    h, w = int(ref["height"]), int(ref["width"])
    transform = ref["transform"]
    crs = rasterio.crs.CRS.from_string(str(ref["crs"]))
    acc = np.zeros((h, w), dtype=np.float32)
    if pool.empty:
        return acc
    g = pool.to_crs(crs)
    shapes = []
    for _, row in g.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        shapes.append((mapping(geom), 1.0))
    if shapes:
        features.rasterize(
            shapes,
            out=acc,
            transform=transform,
            merge_alg=MergeAlg.add,
            dtype=np.float32,
        )
    return np.minimum(acc, 1.0).astype(np.float32)


def _cell_center_lon_lat(ds: xr.Dataset, cid: int, *, nlon: int, nlat: int) -> tuple[float, float]:
    lon_b = np.asarray(ds["longitude_bounds"].values, dtype=np.float64)
    lat_b = np.asarray(ds["latitude_bounds"].values, dtype=np.float64)
    li = int(cid % nlon)
    ji = int(cid // nlon)
    lon_c = 0.5 * (float(lon_b[li, 0]) + float(lon_b[li, 1]))
    lat_c = 0.5 * (float(lat_b[ji, 0]) + float(lat_b[ji, 1]))
    return lon_c, lat_c


def run_aviation_area_proxy_pipeline(
    *,
    repo_root: Path,
    path_cfg: dict[str, Any],
    sector_cfg: dict[str, Any],
    nuts_country: str,
    year: int,
    cams_iso3: str | None,
    log_file: Path | None = None,
) -> dict[str, Any]:
    """End-to-end aviation area proxy + downscaling for configured pollutants."""
    area_cfg = sector_cfg.get("area_source") if isinstance(sector_cfg.get("area_source"), dict) else {}
    if not bool(area_cfg.get("enabled", False)):
        logger.warning("area_source.enabled is false; nothing to do.")
        return {"status": "disabled"}

    _cams_block = sector_cfg.get("cams")
    _gnfr_yaml = area_cfg.get("gnfr") or (
        _cams_block.get("gnfr") if isinstance(_cams_block, dict) else None
    )
    gnfr_code = str(_gnfr_yaml or "H").strip().upper()
    gnfr_index = gnfr_code_to_index(gnfr_code)

    resolved_pm = sector_cfg.get("point_matching") if isinstance(sector_cfg.get("point_matching"), dict) else {}
    iso3 = (cams_iso3 or resolved_pm.get("cams_country_iso3") or "").strip().upper()
    if not iso3:
        iso3_guess = cams_iso3_from_cli_country(nuts_country)
        if iso3_guess:
            iso3 = iso3_guess
    if not iso3:
        raise ValueError(
            "CAMS ISO3 is required (pass cams_iso3= or set point_matching.cams_country_iso3 in sector YAML)."
        )

    pollutants = _parse_pollutant_entries(area_cfg.get("pollutants"))
    if not pollutants:
        raise ValueError("area_source.pollutants is empty.")

    res_m = float(area_cfg.get("resolution_m", 100.0))
    pad_m = float(area_cfg.get("pad_m", sector_cfg.get("pad_m", 5000.0)))
    floor = float(area_cfg.get("proxy_weight_floor", 1e-6))

    out_rel = Path(str(sector_cfg.get("output_dir", "OUTPUT/Proxy_weights/H_Aviation")))
    out_dir = out_rel if out_rel.is_absolute() else repo_root / out_rel
    out_dir.mkdir(parents=True, exist_ok=True)

    nuts_path = Path(str(resolve_path(repo_root, Path(path_cfg["proxy_common"]["nuts_gpkg"]))))
    cams_nc = Path(str(resolve_path(repo_root, Path(path_cfg["emissions"]["cams_2019_nc"]))))

    country_tag = iso3
    meta_country = resolve_nuts_cntr_code(nuts_country)

    file_handler: logging.Handler | None = None
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logging.getLogger().addHandler(file_handler)

    summary_rows: list[dict[str, Any]] = []
    try:
        with xr.open_dataset(cams_nc, engine="netcdf4") as ds:
            cidx = country_index_1based(ds, iso3)
            lon_b = np.asarray(ds["longitude_bounds"].values, dtype=np.float64)
            lat_b = np.asarray(ds["latitude_bounds"].values, dtype=np.float64)
            nlon = int(lon_b.shape[0])
            nlat = int(lat_b.shape[0])

            for spec in pollutants:
                var = spec["cams_var"]
                slug = spec["slug"]
                try:
                    vec = _pollutant_vector(ds, var)
                    s_area, s_point = _area_point_totals(ds, gnfr_index=gnfr_index, country_idx_1based=cidx, pollutant_vec=vec)
                    units = _pollutant_units(ds, var)
                except KeyError as exc:
                    logger.warning(
                        "Skipping pollutant %s (%s): CAMS variable missing (%s).",
                        slug,
                        var,
                        exc,
                    )
                    summary_rows.append(
                        {
                            "pollutant_slug": slug,
                            "cams_var": var,
                            "sum_area": math.nan,
                            "sum_point": math.nan,
                            "units": "unknown",
                            "action": "skip",
                            "reason": f"missing_variable:{exc}",
                        }
                    )
                    continue

                if s_area <= 0.0 or not math.isfinite(s_area):
                    logger.info(
                        "Skipping aviation area downscaling for %s (%s): all GNFR %s mass is on point rows "
                        "(area sum=%.6g, point sum=%.6g).",
                        slug,
                        var,
                        gnfr_code,
                        s_area,
                        s_point,
                    )
                    summary_rows.append(
                        {
                            "pollutant_slug": slug,
                            "cams_var": var,
                            "sum_area": s_area,
                            "sum_point": s_point,
                            "units": units,
                            "action": "skip",
                            "reason": "point_only_no_area_mass",
                        }
                    )
                else:
                    logger.info(
                        "Processing aviation area downscaling for %s (%s): area=%.6g point=%.6g (%s).",
                        slug,
                        var,
                        s_area,
                        s_point,
                        units,
                    )
                    summary_rows.append(
                        {
                            "pollutant_slug": slug,
                            "cams_var": var,
                            "sum_area": s_area,
                            "sum_point": s_point,
                            "units": units,
                            "action": "process",
                            "reason": "",
                        }
                    )

            summary_df = pd.DataFrame(summary_rows)
            summary_path = out_dir / f"aviation_area_summary_{country_tag}_{year}.csv"
            summary_df.to_csv(summary_path, index=False)
            logger.info("Wrote summary table: %s", summary_path)

            to_process = [r for r in summary_rows if r.get("action") == "process"]
            if not to_process:
                logger.info(
                    "No pollutants with non-zero GNFR %s area mass for %s — proxy raster not built.",
                    gnfr_code,
                    iso3,
                )
                write_json(
                    out_dir / f"aviation_area_meta_{country_tag}_{year}.json",
                    {
                        "year": year,
                        "nuts_country": meta_country,
                        "cams_iso3": iso3,
                        "gnfr": gnfr_code,
                        "resolution_m": res_m,
                        "proxy_weight_floor": floor,
                        "skipped_all": True,
                        "summary_csv": str(summary_path.relative_to(repo_root)),
                    },
                )
                return {"status": "skipped_all", "summary_csv": str(summary_path)}

            ref = _reference_profile_100m_nuts(nuts_path, nuts_country, pad_m=pad_m, resolution_m=res_m)
            cam_cell_id = build_cam_cell_id(cams_nc, ref)

            pool_gdf, _src_gpkg = build_aviation_aerodrome_pool_gdf(
                repo_root=repo_root,
                paths_resolved=path_cfg,
                sector_cfg=sector_cfg,
            )
            nuts_union = nuts2_for_country(nuts_path, nuts_country).dissolve()
            nu_geom = nuts_union.geometry.iloc[0]
            if pool_gdf.empty:
                logger.warning("Aerodrome pool is empty before country clip.")
                pool_clip = pool_gdf
            else:
                pool_clip = gpd.clip(pool_gdf, gpd.GeoDataFrame(geometry=[nu_geom], crs=nuts_union.crs).to_crs(
                    pool_gdf.crs
                ))

            pool_path = out_dir / f"aviation_area_aerodrome_pool_{country_tag}_{year}.gpkg"
            pool_clip.to_file(pool_path, driver="GPKG")
            logger.info("Wrote aerodrome pool (%d features): %s", len(pool_clip), pool_path)

            proxy_raw = _rasterize_binary_proxy(pool_clip, ref)
            proxy_path = out_dir / f"aviation_area_proxy_{country_tag}_{year}.tif"
            write_geotiff(
                path=proxy_path,
                array=proxy_raw,
                crs=str(ref["crs"]),
                transform=ref["transform"],
                nodata=0.0,
                tags={
                    "DESCRIPTION": "Binary aerodrome presence (0/1) before CAMS-cell normalisation.",
                    "SOURCE": "OSM aviation aerodrome polygons/nodes",
                    "UNITS": "dimensionless",
                    "GNFR": gnfr_code,
                    "YEAR": str(year),
                    "COUNTRY_ISO3": iso3,
                },
            )
            logger.info("Wrote proxy GeoTIFF: %s", proxy_path)

            qa_records: list[dict[str, Any]] = []
            meta_pollutants: dict[str, Any] = {}

            for row in to_process:
                var = row["cams_var"]
                slug = row["pollutant_slug"]
                vec = _pollutant_vector(ds, var)
                units = _pollutant_units(ds, var)
                mass_by_cid = _mass_by_geo_cell_area(
                    ds,
                    gnfr_index=gnfr_index,
                    country_idx_1based=cidx,
                    pollutant_vec=vec,
                )
                mass_in = float(sum(mass_by_cid.values()))

                emission = np.zeros((ref["height"], ref["width"]), dtype=np.float64)
                n_cells = 0
                n_with_airport = 0
                n_floor = 0
                mass_assigned = 0.0
                mass_lost_nopixels = 0.0

                for cid, mass in mass_by_cid.items():
                    if mass <= 0.0:
                        continue
                    sel = cam_cell_id == cid
                    npx = int(np.count_nonzero(sel))
                    if npx == 0:
                        lon_c, lat_c = _cell_center_lon_lat(ds, cid, nlon=nlon, nlat=nlat)
                        logger.warning(
                            "CAMS cell id=%s center=(%.4f,%.4f) has area mass=%.6g but no fine-grid pixels "
                            "on the reference window — mass omitted.",
                            cid,
                            lon_c,
                            lat_c,
                            mass,
                        )
                        mass_lost_nopixels += mass
                        qa_records.append(
                            {
                                "pollutant_slug": slug,
                                "cams_var": var,
                                "cam_cell_id": cid,
                                "cell_lon": lon_c,
                                "cell_lat": lat_c,
                                "cell_mass": mass,
                                "n_pixels": 0,
                                "proxy_sum_pre_norm": math.nan,
                                "used_background_floor": False,
                                "mass_allocated": 0.0,
                                "note": "no_pixels_on_ref_grid",
                            }
                        )
                        continue

                    n_cells += 1
                    p = proxy_raw[sel].astype(np.float64)
                    s = float(p.sum())
                    used_floor = False
                    if s <= 0.0:
                        wts = np.full(npx, floor, dtype=np.float64)
                        used_floor = True
                        n_floor += 1
                        n_with_airport += 0
                        lon_c, lat_c = _cell_center_lon_lat(ds, cid, nlon=nlon, nlat=nlat)
                        logger.warning(
                            "Background floor applied: CAMS cell id=%s center=(%.4f,%.4f) mass=%.6g "
                            "has no aerodrome proxy pixels (floor=%.3g).",
                            cid,
                            lon_c,
                            lat_c,
                            mass,
                            floor,
                        )
                    else:
                        wts = p.copy()
                        n_with_airport += 1

                    wts = wts / float(wts.sum())
                    alloc = wts * mass
                    emission[sel] += alloc
                    mass_assigned += float(alloc.sum())

                    lon_c, lat_c = _cell_center_lon_lat(ds, cid, nlon=nlon, nlat=nlat)
                    qa_records.append(
                        {
                            "pollutant_slug": slug,
                            "cams_var": var,
                            "cam_cell_id": cid,
                            "cell_lon": lon_c,
                            "cell_lat": lat_c,
                            "cell_mass": mass,
                            "n_pixels": npx,
                            "proxy_sum_pre_norm": s if s > 0.0 else float(npx * floor),
                            "used_background_floor": used_floor,
                            "mass_allocated": float(alloc.sum()),
                            "note": "",
                        }
                    )

                out_tif = out_dir / f"aviation_area_{slug}_{country_tag}_{year}.tif"
                write_geotiff(
                    path=out_tif,
                    array=emission.astype(np.float32),
                    crs=str(ref["crs"]),
                    transform=ref["transform"],
                    nodata=None,
                    tags={
                        "DESCRIPTION": f"GNFR {gnfr_code} area-source emissions downscaled from CAMS ({var}).",
                        "CAMS_VARIABLE": var,
                        "UNITS": units,
                        "GNFR": gnfr_code,
                        "YEAR": str(year),
                        "COUNTRY_ISO3": iso3,
                    },
                )
                logger.info(
                    "Wrote emission GeoTIFF %s (CAMS variable %s, reported units=%s).",
                    out_tif.name,
                    var,
                    units,
                )

                bal = mass_in - mass_assigned - mass_lost_nopixels
                meta_pollutants[slug] = {
                    "cams_var": var,
                    "units": units,
                    "mass_in_cams_area_rows": mass_in,
                    "mass_out_raster": mass_assigned,
                    "mass_lost_no_pixels": mass_lost_nopixels,
                    "balance_residual": bal,
                    "n_area_cells_nonzero": len([k for k, v in mass_by_cid.items() if v > 0]),
                    "n_cells_processed_with_pixels": n_cells,
                    "n_cells_with_airport_proxy": n_with_airport,
                    "n_cells_background_floor": n_floor,
                }
                logger.info(
                    "[%s] mass balance: in=%.12g out=%.12g lost_no_pixels=%.12g residual=%.3g",
                    slug,
                    mass_in,
                    mass_assigned,
                    mass_lost_nopixels,
                    bal,
                )

            qa_path = out_dir / f"aviation_area_qa_{country_tag}_{year}.csv"
            pd.DataFrame(qa_records).to_csv(qa_path, index=False)
            logger.info("Wrote QA table: %s", qa_path)

            meta = {
                "year": year,
                "nuts_country": meta_country,
                "cams_iso3": iso3,
                "gnfr": gnfr_code,
                "cams_netcdf": str(cams_nc.resolve()),
                "resolution_m": res_m,
                "proxy_weight_floor": floor,
                "summary_csv": _rel_to_repo(summary_path, repo_root),
                "proxy_tif": _rel_to_repo(proxy_path, repo_root),
                "qa_csv": _rel_to_repo(qa_path, repo_root),
                "pool_gpkg": _rel_to_repo(pool_path, repo_root),
                "pollutants": meta_pollutants,
                "combinations_skipped": [r for r in summary_rows if r.get("action") != "process"],
            }
            write_json(out_dir / f"aviation_area_meta_{country_tag}_{year}.json", meta)

            return {
                "status": "ok",
                "output_dir": str(out_dir),
                "summary_csv": str(summary_path),
                "proxy_tif": str(proxy_path),
                "qa_csv": str(qa_path),
                "meta_json": str(out_dir / f"aviation_area_meta_{country_tag}_{year}.json"),
            }
    finally:
        if file_handler is not None:
            logging.getLogger().removeHandler(file_handler)
            file_handler.close()
