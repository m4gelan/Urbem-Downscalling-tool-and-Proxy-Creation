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
from UrbEm_Visualizer.dataset_loaders.tif_grid import deposit_point, load_multiband_to_grid, lonlat_to_rowcol
from UrbEm_Visualizer.downscaling.point_meta import (
    flatten_meta_to_row,
    load_match_sidecar,
    meta_for_cams,
    sidecar_has_emissions,
)
from UrbEm_Visualizer.downscaling.sector_meta import load_sector_yaml, sector_mode
from UrbEm_Visualizer.downscaling.spatial import FineGrid
from UrbEm_Visualizer.pollutants import band_index_for_pollutant


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
        base = {
            "cams_point_id": int(pid_str),
            "cams_lon": float(rec["cams_lon"]),
            "cams_lat": float(rec["cams_lat"]),
            **{f"emis_{pol}": float(pols.get(pol, 0) or 0) for pol in pollutants},
        }
        fac = None
        if rec.get("matched") == "yes":
            lon, lat = rec.get("facility_lon"), rec.get("facility_lat")
            if lon is not None and lat is not None:
                fac = (float(lon), float(lat))
        _split_by_domain(base, fac, rec, west, south, east, north, appointed, not_appointed, unmatched)
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


def allocate_points(
    *,
    grid: FineGrid,
    appointed: list[dict],
    not_appointed: list[dict],
    unmatched: list[dict],
    pollutants: list[str],
    layer_mode: str,
    cell_id: np.ndarray,
    weight_planes: dict[str, np.ndarray] | None,
    point_only: bool,
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

    for rec in not_appointed:
        _deposit_row(rec, use_facility=False)

    for rec in unmatched:
        if layer_mode == "separate":
            _deposit_row(rec, use_facility=False)
            continue
        r0, c0 = lonlat_to_rowcol(grid, rec["cams_lon"], rec["cams_lat"])
        cid = int(cell_id[r0, c0]) if 0 <= r0 < grid.height and 0 <= c0 < grid.width else -1
        if cid < 0:
            _deposit_row(rec, use_facility=False)
            continue
        mask = cell_id == cid
        n_pix = int(np.count_nonzero(mask))
        if n_pix <= 0:
            continue
        for pi, pol in enumerate(pollutants):
            mass = np.float32(rec.get(f"emis_{pol}", 0.0))
            if mass <= 0:
                continue
            if weight_planes and pol in weight_planes and not point_only:
                w = weight_planes[pol]
                w_cell = w[mask].astype(np.float32)
                s = float(w_cell.sum())
                if s > 0:
                    stack[pi][mask] += w_cell * (mass / np.float32(s))
            else:
                share = mass / np.float32(n_pix)
                stack[pi][mask] += share

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
    layer_mode: str,
    cell_id: np.ndarray,
    area_weight_path: Path | None,
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

    weight_planes = None
    if area_weight_path and area_weight_path.is_file():
        names, wstack = load_multiband_to_grid(area_weight_path, grid)
        weight_planes = {}
        for pol in pollutants:
            bi = band_index_for_pollutant(names, pol)
            weight_planes[pol] = wstack[bi]
    if on_progress:
        on_progress("Allocating to grid", 0.7)
    ponly = sector_mode(sector_id) == "point_only"
    stack = allocate_points(
        grid=grid,
        appointed=appointed,
        not_appointed=not_appointed,
        unmatched=unmatched,
        pollutants=pollutants,
        layer_mode=layer_mode,
        cell_id=cell_id,
        weight_planes=weight_planes,
        point_only=ponly,
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
