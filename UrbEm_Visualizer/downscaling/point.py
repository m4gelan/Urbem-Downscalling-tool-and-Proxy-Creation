from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from pyproj import Transformer

from UrbEm_Visualizer.dataset_loaders.cams_alias import country_iso3
from UrbEm_Visualizer.dataset_loaders.cams_emissions import load_cams_points
from UrbEm_Visualizer.dataset_loaders.tif_grid import deposit_point
from UrbEm_Visualizer.downscaling.point_meta import (
    appointed_meta,
    flatten_meta_to_row,
    facility_links,
    load_match_sidecar,
    meta_for_cams,
    sidecar_has_emissions,
    sidecar_pollutant_mass,
)
from UrbEm_Visualizer.downscaling.sector_meta import load_sector_yaml
from UrbEm_Visualizer.downscaling.spatial import FineGrid
from UrbEm_Visualizer.dataset_loaders.cams_emissions import cell_id_from_lonlat


def domain_wgs84(domain: dict) -> tuple[float, float, float, float]:
    tr = Transformer.from_crs(str(domain["crs"]), "EPSG:4326", always_xy=True)
    xmin, ymin, xmax, ymax = (
        float(domain["xmin"]),
        float(domain["ymin"]),
        float(domain["xmax"]),
        float(domain["ymax"]),
    )
    xs = [xmin, xmax, xmin, xmax]
    ys = [ymin, ymin, ymax, ymax]
    lons, lats = tr.transform(xs, ys)
    return float(min(lons)), float(min(lats)), float(max(lons)), float(max(lats))


def _in_wgs84_bbox(lon: float, lat: float, west: float, south: float, east: float, north: float) -> bool:
    return west <= lon <= east and south <= lat <= north


def _pixel_lonlat(transform: rasterio.Affine, crs: Any, row: int, col: int) -> tuple[float, float]:
    x, y = rasterio.transform.xy(transform, row, col, offset="center")
    tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon, lat = tr.transform(x, y)
    return float(lon), float(lat)


def _build_facility_index(b2: np.ndarray) -> dict[float, list[tuple[int, int]]]:
    idx: dict[float, list[tuple[int, int]]] = {}
    rows, cols = np.nonzero(b2 > 0)
    for r, c in zip(rows, cols):
        v = float(b2[r, c])
        idx.setdefault(v, []).append((int(r), int(c)))
    return idx


def _facility_from_tif(
    b1: np.ndarray,
    fac_idx: dict[float, list[tuple[int, int]]],
    transform: rasterio.Affine,
    crs: Any,
    cams_row: int,
    cams_col: int,
    tol_frac: float = 1e-5,
) -> tuple[float, float] | None:
    v = float(b1[cams_row, cams_col])
    if v <= 0:
        return None
    tol = max(tol_frac * v, 1e-9)

    def _pick(candidates: list[tuple[int, int]]) -> tuple[float, float] | None:
        for r, c in candidates:
            if int(r) == cams_row and int(c) == cams_col:
                continue
            return _pixel_lonlat(transform, crs, int(r), int(c))
        return None

    if v in fac_idx:
        hit = _pick(fac_idx[v])
        if hit:
            return hit
    for fv, locs in fac_idx.items():
        if abs(fv - v) <= tol:
            hit = _pick(locs)
            if hit:
                return hit
    return None


def _split_by_domain(
    base: dict[str, Any],
    fac: tuple[float, float] | None,
    meta: dict[str, Any],
    west: float,
    south: float,
    east: float,
    north: float,
    appointed: list[dict],
    not_appointed: list[dict],
    unmatched: list[dict],
) -> None:
    cams_in = _in_wgs84_bbox(base["cams_lon"], base["cams_lat"], west, south, east, north)
    if fac is None:
        if cams_in:
            unmatched.append({**base, "match_status": "unmatched"})
        return
    flon, flat = fac
    base["facility_lon"] = flon
    base["facility_lat"] = flat
    fac_in = _in_wgs84_bbox(flon, flat, west, south, east, north)
    if cams_in and fac_in:
        row = {**base, "match_status": "matched_appointed"}
        row.update(flatten_meta_to_row(meta))
        appointed.append(row)
    elif cams_in or fac_in:
        not_appointed.append({**base, "match_status": "matched_not_appointed"})


def _emis_from_sidecar(rec: dict[str, Any], link: dict[str, Any] | None, pollutants: list[str]) -> dict[str, float]:
    attributed = (link or {}).get("attributed_pollutants") or {}
    cams_pols = rec.get("pollutants") or {}
    out: dict[str, float] = {}
    for pol in pollutants:
        if attributed:
            out[f"emis_{pol}"] = sidecar_pollutant_mass(attributed, pol)
        else:
            out[f"emis_{pol}"] = sidecar_pollutant_mass(cams_pols, pol)
    return out


def _records_from_sidecar(
    sidecar: dict[str, dict[str, Any]],
    domain: dict,
    pollutants: list[str],
) -> tuple[list[dict], list[dict], list[dict]]:
    west, south, east, north = domain_wgs84(domain)
    appointed, not_appointed, unmatched = [], [], []
    for pid_str, rec in sidecar.items():
        pols = rec.get("pollutants") or {}
        if not isinstance(pols, dict):
            continue
        base_cams = {
            "cams_point_id": int(pid_str),
            "cams_lon": float(rec["cams_lon"]),
            "cams_lat": float(rec["cams_lat"]),
        }
        links = facility_links(rec)
        if rec.get("matched") == "yes" and links:
            for lk in links:
                flon, flat = lk.get("facility_lon"), lk.get("facility_lat")
                if flon is None or flat is None:
                    continue
                base = {**base_cams, **_emis_from_sidecar(rec, lk, pollutants)}
                meta = appointed_meta(rec, lk)
                _split_by_domain(
                    base,
                    (float(flon), float(flat)),
                    meta,
                    west,
                    south,
                    east,
                    north,
                    appointed,
                    not_appointed,
                    unmatched,
                )
            continue
        base = {**base_cams, **_emis_from_sidecar(rec, None, pollutants)}
        fac = None
        if rec.get("matched") == "yes":
            lon, lat = rec.get("facility_lon"), rec.get("facility_lat")
            if lon is not None and lat is not None:
                fac = (float(lon), float(lat))
        meta = appointed_meta(rec) if fac else rec
        _split_by_domain(base, fac, meta, west, south, east, north, appointed, not_appointed, unmatched)
    return appointed, not_appointed, unmatched


def _records_from_cams_and_tif(
    cams_points: dict[int, dict[str, Any]],
    link_path: Path,
    domain: dict,
    pollutants: list[str],
    sidecar: dict[str, dict[str, Any]],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Legacy path when sidecar has metadata only — CAMS for emissions, GeoTIFF for facility coords."""
    west, south, east, north = domain_wgs84(domain)
    appointed, not_appointed, unmatched = [], [], []

    with rasterio.open(link_path) as src:
        b1 = src.read(1).astype(np.float32)
        b2 = src.read(2).astype(np.float32)
        transform = src.transform
        crs = src.crs
        tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    fac_idx = _build_facility_index(b2)

    for pid, row in cams_points.items():
        lon = float(row["longitude"])
        lat = float(row["latitude"])
        x, y = tr.transform(lon, lat)
        r, c = rasterio.transform.rowcol(transform, x, y)
        if not (0 <= r < b1.shape[0] and 0 <= c < b1.shape[1]):
            continue
        pols = {k: float(v) for k, v in (row.get("pollutants") or {}).items()}
        meta = meta_for_cams(sidecar, int(pid))
        fac = None
        flon, flat = meta.get("facility_lon"), meta.get("facility_lat")
        if flon is not None and flat is not None:
            fac = (float(flon), float(flat))
        if fac is None:
            fac = _facility_from_tif(b1, fac_idx, transform, crs, int(r), int(c))
        base = {
            "cams_point_id": int(pid),
            "cams_lon": lon,
            "cams_lat": lat,
            **{f"emis_{pol}": float(pols.get(pol, 0) or 0) for pol in pollutants},
        }
        _split_by_domain(base, fac, meta, west, south, east, north, appointed, not_appointed, unmatched)

    return appointed, not_appointed, unmatched


def load_point_records(
    link_path: Path,
    domain: dict,
    pollutants: list[str],
    *,
    cams_points: dict[int, dict[str, Any]] | None = None,
    sidecar: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    sidecar = sidecar if sidecar is not None else load_match_sidecar(link_path)
    if sidecar_has_emissions(sidecar):
        return _records_from_sidecar(sidecar, domain, pollutants)
    if cams_points is None:
        raise ValueError(
            f"{link_path.name}: missing {link_path.stem}_matches.json with pollutant data — "
            "re-run proxy point matching for this sector"
        )
    return _records_from_cams_and_tif(cams_points, link_path, domain, pollutants, sidecar)


def _records_to_gdf(records: list[dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def add_unmatched_to_cams_cells(
    cams_cells: dict[int, dict[str, Any]],
    unmatched: list[dict],
    cams_grid: dict[str, Any],
    pollutants: list[str],
) -> dict[int, dict[str, Any]]:
    if not unmatched or not cams_grid:
        return cams_cells
    lon_bounds = cams_grid["lon_bounds"]
    lat_bounds = cams_grid["lat_bounds"]
    nlon = int(cams_grid["n_longitude"])
    nlat = int(cams_grid["n_latitude"])
    out = {cid: dict(row) for cid, row in cams_cells.items()}
    for rec in unmatched:
        cid = int(cell_id_from_lonlat(
            np.array([rec["cams_lon"]], dtype=np.float32),
            np.array([rec["cams_lat"]], dtype=np.float32),
            lon_bounds,
            lat_bounds,
            nlon,
            nlat,
        )[0])
        if cid < 0:
            continue
        if cid not in out:
            out[cid] = {"pollutants_within_cell": {p: 0.0 for p in pollutants}}
        pols = out[cid].setdefault("pollutants_within_cell", {p: 0.0 for p in pollutants})
        for pol in pollutants:
            mass = float(rec.get(f"emis_{pol}", 0.0) or 0.0)
            if mass > 0:
                pols[pol] = float(pols.get(pol, 0.0)) + mass
    return out


def _facility_count_for_record(rec: dict[str, Any], link_path: Path, cams_point_id: int) -> int:
    if rec.get("matched") != "yes":
        return 0
    links = facility_links(rec)
    if links:
        return len(links)
    flon, flat = rec.get("facility_lon"), rec.get("facility_lat")
    if flon is not None and flat is not None:
        return 1
    with rasterio.open(link_path) as src:
        b1 = src.read(1).astype(np.float32)
        b2 = src.read(2).astype(np.float32)
        fac_idx = _build_facility_index(b2)
        v = float(cams_point_id)
        tol = max(1e-5 * v, 1e-9)
        locs: set[tuple[int, int]] = set()
        if v in fac_idx:
            locs.update(fac_idx[v])
        for fv, cand in fac_idx.items():
            if abs(fv - v) <= tol:
                locs.update(cand)
        return len(locs)


def point_match_stats(
    link_path: Path,
    domain: dict,
    pollutants: list[str],
    *,
    sidecar: dict[str, dict[str, Any]] | None = None,
) -> dict[str, int]:
    sidecar = sidecar if sidecar is not None else load_match_sidecar(link_path)
    west, south, east, north = domain_wgs84(domain)
    total = n0 = n1 = n2p = 0
    for pid_str, rec in sidecar.items():
        if not isinstance(rec.get("pollutants"), dict):
            continue
        lon, lat = float(rec["cams_lon"]), float(rec["cams_lat"])
        if not _in_wgs84_bbox(lon, lat, west, south, east, north):
            continue
        if not any(sidecar_pollutant_mass(rec["pollutants"], p) > 0 for p in pollutants):
            continue
        total += 1
        nf = _facility_count_for_record(rec, link_path, int(pid_str))
        if nf == 0:
            n0 += 1
        elif nf == 1:
            n1 += 1
        else:
            n2p += 1
    _, not_appointed, _ = load_point_records(
        link_path, domain, pollutants, sidecar=sidecar,
    )
    fac_out, cams_out = partial_outside_counts(not_appointed, domain)
    return {
        "total": total,
        "facilities_0": n0,
        "facilities_1": n1,
        "facilities_2plus": n2p,
        "partial_match": len(not_appointed),
        "facility_outside_domain": fac_out,
        "cams_outside_domain": cams_out,
    }


def partial_outside_counts(
    not_appointed: list[dict],
    domain: dict,
) -> tuple[int, int]:
    west, south, east, north = domain_wgs84(domain)
    fac_out = cams_out = 0
    for rec in not_appointed:
        cams_in = _in_wgs84_bbox(
            float(rec["cams_lon"]), float(rec["cams_lat"]), west, south, east, north,
        )
        fac_in = _in_wgs84_bbox(
            float(rec["facility_lon"]), float(rec["facility_lat"]), west, south, east, north,
        )
        if cams_in and not fac_in:
            fac_out += 1
        elif fac_in and not cams_in:
            cams_out += 1
    return fac_out, cams_out


def _facility_in_domain(rec: dict, domain: dict) -> bool:
    west, south, east, north = domain_wgs84(domain)
    return _in_wgs84_bbox(
        float(rec["facility_lon"]), float(rec["facility_lat"]), west, south, east, north,
    )


def allocate_points(
    *,
    grid: FineGrid,
    appointed: list[dict],
    not_appointed: list[dict],
    unmatched: list[dict],
    pollutants: list[str],
    burn_unmatched_to_area: bool,
    merged_mode: bool = False,
    partial_match_handling: str | None = None,
    domain: dict | None = None,
) -> np.ndarray:
    n_pol = len(pollutants)
    stack = np.zeros((n_pol, grid.height, grid.width), dtype=np.float32)

    def _deposit_row(rec: dict, use_facility: bool) -> None:
        lon = rec["facility_lon"] if use_facility else rec["cams_lon"]
        lat = rec["facility_lat"] if use_facility else rec["cams_lat"]
        for pi, pol in enumerate(pollutants):
            mass = np.float32(rec.get(f"emis_{pol}", 0.0))
            if mass > 0:
                deposit_point(grid, stack[pi], lon, lat, float(mass))

    for rec in appointed:
        _deposit_row(rec, use_facility=True)

    facility_or_drop = partial_match_handling == "facility_or_drop" and domain is not None
    for rec in not_appointed:
        if facility_or_drop:
            if _facility_in_domain(rec, domain):
                _deposit_row(rec, use_facility=True)
            continue
        if not merged_mode:
            _deposit_row(rec, use_facility=False)

    for rec in unmatched:
        if burn_unmatched_to_area:
            continue
        _deposit_row(rec, use_facility=False)

    return stack


def run_point_sector(
    *,
    grid: FineGrid,
    link_path: Path,
    cams_nc: Path,
    sector_id: str,
    country: str,
    year: int,
    pollutants: list[str],
    domain: dict,
    procedure: str,
    burn_unmatched_to_area: bool,
    partial_match_handling: str | None = None,
    on_progress: Callable[[str, float], None] | None = None,
) -> tuple[xr.DataArray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sidecar = load_match_sidecar(link_path)
    cams_points = None
    if not sidecar_has_emissions(sidecar):
        sec_yaml = load_sector_yaml(sector_id)
        cps = sec_yaml.get("cams_point_sources")
        if not cps:
            raise ValueError(f"{sector_id}: no cams_point_sources in sector config")
        if on_progress:
            on_progress("Loading CAMS emissions (legacy link)", 0.15)
        cams_points = load_cams_points(
            cams_nc,
            year=int(cps["year"]),
            country_iso3=country_iso3(country),
            emission_category_indices=list(cps["emission_category_indices"]),
            source_type_indices=list(cps["source_type_indices"]),
            pollutants=pollutants,
        )
    elif on_progress:
        on_progress("Reading point link sidecar", 0.2)

    if on_progress:
        on_progress("Classifying points in domain", 0.35)
    appointed, not_appointed, unmatched = load_point_records(
        link_path,
        domain,
        pollutants,
        cams_points=cams_points,
        sidecar=sidecar,
    )

    if on_progress:
        on_progress("Allocating to grid", 0.7)
    stack = allocate_points(
        grid=grid,
        appointed=appointed,
        not_appointed=not_appointed,
        unmatched=unmatched,
        pollutants=pollutants,
        burn_unmatched_to_area=burn_unmatched_to_area,
        merged_mode=procedure == "merged",
        partial_match_handling=partial_match_handling,
        domain=domain,
    )
    da = xr.DataArray(
        stack,
        dims=("pollutant", "y", "x"),
        coords={"pollutant": pollutants, "y": np.arange(grid.height), "x": np.arange(grid.width)},
    )
    return (
        da,
        _records_to_gdf(appointed),
        _records_to_gdf(not_appointed),
        _records_to_gdf(unmatched),
    )
