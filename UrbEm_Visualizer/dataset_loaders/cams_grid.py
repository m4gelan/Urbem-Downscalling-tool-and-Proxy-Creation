from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import Any, Callable

import numpy as np
import xarray as xr
import yaml
from pyproj import Transformer
from rasterio.warp import transform_bounds

from UrbEm_Visualizer.dataset_loaders.countries import country_iso3
from UrbEm_Visualizer.paths import project_root

ProgressFn = Callable[[int, int, str], None]


def _collect_cams_filters(root: Path) -> list[dict[str, Any]]:
    seen: set[tuple[str, tuple[int, ...], tuple[int, ...]]] = set()
    filters: list[dict[str, Any]] = []
    sector_dir = root / "proxy" / "config" / "sector"
    if not sector_dir.is_dir():
        return [{
            "sector_id": "default",
            "role": "area",
            "emission_category_indices": [1],
            "source_type_indices": [1, 2],
        }]

    for d in sorted(sector_dir.iterdir()):
        if not d.is_dir():
            continue
        cfg_path = d / f"{d.name}_sector_config.yaml"
        if not cfg_path.is_file():
            continue
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        for block, role in (
            ("cams_sector_cells", "cells"),
            ("cams_area_emissions", "area"),
            ("cams_point_sources", "point"),
        ):
            block_cfg = cfg.get(block)
            if not isinstance(block_cfg, dict):
                continue
            ec = tuple(int(x) for x in block_cfg["emission_category_indices"])
            st = tuple(int(x) for x in block_cfg["source_type_indices"])
            key = (d.name, ec, st)
            if key in seen:
                continue
            seen.add(key)
            filters.append({
                "sector_id": d.name,
                "role": role,
                "emission_category_indices": list(ec),
                "source_type_indices": list(st),
            })
    if not filters:
        filters.append({
            "sector_id": "default",
            "role": "area",
            "emission_category_indices": [1],
            "source_type_indices": [1, 2],
        })
    return filters


def _aggregate_cells(
    sel: np.ndarray,
    lon_idx: np.ndarray,
    lat_idx: np.ndarray,
    nlon: int,
    lon_bounds: np.ndarray,
    lat_bounds: np.ndarray,
) -> dict[int, dict[str, Any]]:
    if sel.size == 0:
        return {}
    cell_ids = lat_idx[sel] * nlon + lon_idx[sel]
    unique_cells = np.unique(cell_ids)
    out: dict[int, dict[str, Any]] = {}
    for cell_id in unique_cells.tolist():
        lo = int(cell_id % nlon)
        la = int(cell_id // nlon)
        west, east = float(lon_bounds[lo, 0]), float(lon_bounds[lo, 1])
        south, north = float(lat_bounds[la, 0]), float(lat_bounds[la, 1])
        out[int(cell_id)] = {
            "cell_bounds_wgs84": {"west": west, "south": south, "east": east, "north": north},
        }
    return out


def _coverage_warnings(sector_cells: dict[str, set[int]]) -> tuple[str | None, list[str]]:
    if not sector_cells:
        return None, []
    ref_sid = max(sector_cells, key=lambda s: len(sector_cells[s]))
    ref = sector_cells[ref_sid]
    msgs: list[str] = []
    for sid, cells in sector_cells.items():
        if sid == ref_sid:
            continue
        extra = cells - ref
        if extra:
            msgs.append(
                f"{sid}: {len(extra)} CAMS cell(s) outside {ref_sid} "
                f"(largest footprint, {len(ref)} cells)"
            )
    return ref_sid, msgs


def _proxy_grid_warnings(config: dict | None) -> list[str]:
    if not config:
        return []
    root = project_root()
    tiles: list[dict[str, Any]] = []
    for sid, sec in (config.get("sectors") or {}).items():
        aw = sec.get("area_weights") or {}
        if aw.get("absent") or not aw.get("path"):
            continue
        p = Path(str(aw["path"]))
        if not p.is_absolute():
            p = root / p
        if not p.is_file():
            continue
        import rasterio

        with rasterio.open(p) as src:
            if src.crs is None:
                continue
            w, s, e, n = transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=4)
            tiles.append({
                "sector_id": sid,
                "west": w,
                "south": s,
                "east": e,
                "north": n,
                "area": (e - w) * (n - s),
            })
    if len(tiles) < 2:
        return []
    ref = max(tiles, key=lambda t: t["area"])
    msgs: list[str] = []
    eps = 1e-6
    for t in tiles:
        if t["sector_id"] == ref["sector_id"]:
            continue
        outside = (
            t["west"] < ref["west"] - eps
            or t["east"] > ref["east"] + eps
            or t["south"] < ref["south"] - eps
            or t["north"] > ref["north"] + eps
        )
        if outside:
            msgs.append(
                f"Weight grid {t['sector_id']} extends outside {ref['sector_id']} "
                "(largest proxy extent)"
            )
    return msgs


def _load_cells_union(
    cams_nc: Path,
    *,
    iso3: str,
    year: int,
    filters: list[dict[str, Any]],
    pollutants: list[str],
    on_progress: ProgressFn | None = None,
) -> tuple[dict[int, dict[str, Any]], list[str]]:
    _ = year
    root = project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from proxy.core.alias import cams_country_index_from_iso3, cams_pollutant_var

    labels = [p.strip() for p in pollutants if p.strip()]
    if not labels:
        raise ValueError("pollutants must be non-empty")

    if on_progress:
        on_progress(0, 1, "Reading CAMS file…")

    with xr.open_dataset(cams_nc, engine="netcdf4") as ds:
        country_idx = cams_country_index_from_iso3(ds, iso3)
        lon_src = np.asarray(ds["longitude_source"].values, dtype=np.float32).ravel()
        lat_src = np.asarray(ds["latitude_source"].values, dtype=np.float32).ravel()
        src_type = np.asarray(ds["source_type_index"].values, dtype=np.int64).ravel()
        emis_cat = np.asarray(ds["emission_category_index"].values, dtype=np.int64).ravel()
        country_index = np.asarray(ds["country_index"].values, dtype=np.int64).ravel()
        lon_idx = np.asarray(ds["longitude_index"].values, dtype=np.int64).ravel()
        lat_idx = np.asarray(ds["latitude_index"].values, dtype=np.int64).ravel()
        nlon = int(ds.sizes["longitude"])
        nlat = int(ds.sizes["latitude"])
        lon_bounds = np.asarray(ds["longitude_bounds"].values, dtype=np.float64)
        lat_bounds = np.asarray(ds["latitude_bounds"].values, dtype=np.float64)
        if lon_idx.size and (lon_idx.max() >= nlon or lat_idx.max() >= nlat):
            lon_idx = lon_idx - 1
            lat_idx = lat_idx - 1
        np.clip(lon_idx, 0, nlon - 1, out=lon_idx)
        np.clip(lat_idx, 0, nlat - 1, out=lat_idx)
        pol_mat = np.column_stack([
            np.nan_to_num(
                np.asarray(ds[cams_pollutant_var(lab)].values, dtype=np.float32).ravel(),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            for lab in labels
        ]).astype(np.float32)

    base = (
        np.isfinite(lon_src)
        & np.isfinite(lat_src)
        & (country_index == int(country_idx))
        & (pol_mat.max(axis=1) > 0.0)
    )

    merged: dict[int, dict[str, Any]] = {}
    by_sector: dict[str, set[int]] = {}
    total = len(filters)
    for i, flt in enumerate(filters):
        sid = flt["sector_id"]
        if on_progress:
            on_progress(i, total, f"CAMS cells — {sid} ({flt['role']})")
        ec = np.asarray(flt["emission_category_indices"], dtype=np.int64)
        st = np.asarray(flt["source_type_indices"], dtype=np.int64)
        mask = base & np.isin(emis_cat, ec) & np.isin(src_type, st)
        sel = np.flatnonzero(mask)
        cells = _aggregate_cells(sel, lon_idx, lat_idx, nlon, lon_bounds, lat_bounds)
        by_sector.setdefault(sid, set()).update(cells.keys())
        for cid, row in cells.items():
            if cid not in merged:
                merged[cid] = row

    if on_progress:
        on_progress(total, total, "CAMS grid complete")

    _, cov_msgs = _coverage_warnings(by_sector)
    return merged, cov_msgs


def cells_to_geojson(cells: dict[int, dict[str, Any]]) -> dict[str, Any]:
    features = []
    for cid, row in cells.items():
        b = row["cell_bounds_wgs84"]
        ring = [
            [b["west"], b["south"]],
            [b["east"], b["south"]],
            [b["east"], b["north"]],
            [b["west"], b["north"]],
            [b["west"], b["south"]],
        ]
        features.append({
            "type": "Feature",
            "id": cid,
            "properties": {"cell_id": cid},
            "geometry": {"type": "Polygon", "coordinates": [ring]},
        })
    return {"type": "FeatureCollection", "features": features}


def _cell_in_domain_envelope(cell: dict[str, Any], west: float, south: float, east: float, north: float) -> bool:
    b = cell["cell_bounds_wgs84"]
    return not (
        b["east"] < west or b["west"] > east or b["north"] < south or b["south"] > north
    )


def load_domain_cams_geojson(
    cams_path: Path,
    country: str,
    year: int,
    pollutants: list[str],
    domain: dict,
) -> dict[str, Any]:
    iso3 = country_iso3(country)
    filters = _collect_cams_filters(project_root())
    cells, _ = _load_cells_union(
        cams_path,
        iso3=iso3,
        year=year,
        filters=filters,
        pollutants=pollutants,
    )
    if not cells:
        raise ValueError(f"No CAMS grid cells for {country} ({iso3})")
    w, s, e, n = domain_wgs84_from_domain(domain)
    picked = {cid: row for cid, row in cells.items() if _cell_in_domain_envelope(row, w, s, e, n)}
    gj = cells_to_geojson(picked)
    gj["cell_count"] = len(picked)
    gj["crs"] = str(domain.get("crs", ""))
    return gj


def domain_wgs84_from_domain(domain: dict) -> tuple[float, float, float, float]:
    from UrbEm_Visualizer.visualization.load_run import domain_wgs84

    return domain_wgs84(domain)


def load_country_cams_geojson(
    cams_path: Path,
    country: str,
    year: int,
    pollutants: list[str],
    on_progress: ProgressFn | None = None,
    config: dict | None = None,
) -> dict[str, Any]:
    iso3 = country_iso3(country)
    filters = _collect_cams_filters(project_root())
    cells, cov_msgs = _load_cells_union(
        cams_path,
        iso3=iso3,
        year=year,
        filters=filters,
        pollutants=pollutants,
        on_progress=on_progress,
    )
    if not cells:
        raise ValueError(f"No CAMS grid cells for {country} ({iso3})")
    gj = cells_to_geojson(cells)
    west = min(c["cell_bounds_wgs84"]["west"] for c in cells.values())
    east = max(c["cell_bounds_wgs84"]["east"] for c in cells.values())
    south = min(c["cell_bounds_wgs84"]["south"] for c in cells.values())
    north = max(c["cell_bounds_wgs84"]["north"] for c in cells.values())
    gj["bounds"] = {"west": west, "south": south, "east": east, "north": north}
    gj["cell_count"] = len(cells)
    gj["country_iso3"] = iso3
    warnings = list(cov_msgs) + _proxy_grid_warnings(config)
    if warnings:
        gj["coverage_warnings"] = warnings
    return gj


def transform_bbox(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    from_crs: str,
    to_crs: str,
) -> dict[str, float | list[list[float]]]:
    tr = Transformer.from_crs(from_crs, to_crs, always_xy=True)
    xs = [xmin, xmax, xmax, xmin, xmin]
    ys = [ymin, ymin, ymax, ymax, ymin]
    xo, yo = tr.transform(xs, ys)
    ring = [[float(xo[i]), float(yo[i])] for i in range(len(xs))]
    return {
        "xmin": float(min(xo)),
        "ymin": float(min(yo)),
        "xmax": float(max(xo)),
        "ymax": float(max(yo)),
        "ring": ring,
    }


_CAMS_JOBS: dict[str, dict[str, Any]] = {}


def start_cams_grid_job(
    cams_path: Path,
    country: str,
    year: int,
    pollutants: list[str],
    config: dict | None = None,
) -> str:
    import threading

    job_id = uuid.uuid4().hex
    _CAMS_JOBS[job_id] = {
        "done": False,
        "error": None,
        "progress": 0.0,
        "message": "Starting…",
        "sector": None,
        "geojson": None,
    }

    def run():
        state = _CAMS_JOBS[job_id]

        def on_progress(i: int, total: int, msg: str) -> None:
            state["progress"] = (i / total) if total else 0.0
            state["message"] = msg
            if " — " in msg:
                state["sector"] = msg.split("—", 1)[1].strip().split("(")[0].strip()

        try:
            gj = load_country_cams_geojson(
                cams_path,
                country,
                year,
                pollutants,
                on_progress=on_progress,
                config=config,
            )
            state["geojson"] = gj
            state["progress"] = 1.0
            state["message"] = "CAMS grid loaded"
            state["done"] = True
        except Exception as e:
            state["error"] = str(e)
            state["done"] = True

    threading.Thread(target=run, daemon=True).start()
    return job_id


def cams_job_status(job_id: str) -> dict[str, Any] | None:
    return _CAMS_JOBS.get(job_id)
