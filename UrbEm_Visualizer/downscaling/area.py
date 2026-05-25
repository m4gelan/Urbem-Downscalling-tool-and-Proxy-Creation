from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import xarray as xr

from UrbEm_Visualizer.dataset_loaders.cams_alias import country_iso3
from UrbEm_Visualizer.dataset_loaders.cams_emissions import (
    cams_mass_lookup,
    load_cams_area_cells,
    load_cams_grid_meta,
    load_cams_points,
    point_cell_ids,
)
from UrbEm_Visualizer.dataset_loaders.tif_grid import (
    cell_id_on_raster,
    pixels_in_domain_bbox,
    read_weight_stack_native,
    reproject_plane_to_grid,
)
from UrbEm_Visualizer.downscaling.sector_meta import load_sector_yaml
from UrbEm_Visualizer.downscaling.spatial import FineGrid
from UrbEm_Visualizer.pollutants import band_index_for_pollutant

WEIGHT_TOL = 1e-6


def cell_weight_sums(weights: np.ndarray, cell_id: np.ndarray) -> np.ndarray:
    flat_c = cell_id.ravel()
    flat_w = weights.ravel()
    valid = flat_c >= 0
    if not np.any(valid):
        return np.zeros(1, dtype=np.float64)
    c = flat_c[valid].astype(np.int64)
    max_c = int(c.max())
    return np.bincount(c, weights=flat_w[valid].astype(np.float64), minlength=max_c + 1)


def check_weights_native(
    weights: np.ndarray,
    cell_id: np.ndarray,
    pollutant: str,
    cells_to_check: np.ndarray,
    sector_id: str,
) -> list[dict[str, Any]]:
    """Sum weights on full native cell (all pixels with cell_id); only cells touching domain."""
    sums = cell_weight_sums(weights, cell_id)
    fails: list[dict[str, Any]] = []
    for cid in cells_to_check:
        ic = int(cid)
        if ic < 0 or ic >= sums.size:
            continue
        s = float(sums[ic])
        if abs(s - 1.0) > WEIGHT_TOL:
            fails.append({"cell_id": ic, "pollutant": pollutant, "sum": s})
    return fails


def downscale_area(
    *,
    grid: FineGrid,
    area_path: Path,
    sector_id: str,
    domain: dict,
    pollutants: list[str],
    cams_cells: dict[int, dict[str, Any]],
    cams_grid: dict[str, Any],
    on_pollutant_done: Callable[[str], None] | None = None,
) -> tuple[xr.DataArray, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    labels, native_stack, nat_tr, nat_crs = read_weight_stack_native(area_path)
    h, w = int(native_stack.shape[1]), int(native_stack.shape[2])
    valid_keys = frozenset(cams_cells.keys()) if cams_cells else frozenset()
    cell_id = cell_id_on_raster(nat_tr, nat_crs, h, w, cams_grid, valid_keys)
    in_domain = pixels_in_domain_bbox(nat_tr, nat_crs, h, w, domain)
    domain_cells = np.unique(cell_id[in_domain & (cell_id >= 0)])

    mass_lut = cams_mass_lookup(cams_cells, pollutants)
    n_pol = len(pollutants)
    out_stack = np.zeros((n_pol, grid.height, grid.width), dtype=np.float32)
    clip_log: list[dict[str, Any]] = []
    weight_log: dict[str, Any] = {"sector": sector_id, "pollutants": {}, "passed": True}

    ok = cell_id >= 0
    cid_ok = cell_id[ok]

    for pi, pol in enumerate(pollutants):
        bi = band_index_for_pollutant(labels, pol)
        w_plane = native_stack[bi]
        fails = check_weights_native(w_plane, cell_id, pol, domain_cells, sector_id)
        weight_log["pollutants"][pol] = {"failures": fails, "cells_checked": int(domain_cells.size)}
        if fails:
            weight_log["passed"] = False
            return (
                xr.DataArray(
                    out_stack,
                    dims=("pollutant", "y", "x"),
                    coords={"pollutant": pollutants, "y": np.arange(grid.height), "x": np.arange(grid.width)},
                ),
                weight_log,
                fails,
                clip_log,
            )

        lut = mass_lut[pol]
        e_native = np.zeros((h, w), dtype=np.float32)
        e_native[ok] = lut[cid_ok] * w_plane[ok]
        out_stack[pi] = reproject_plane_to_grid(e_native, nat_tr, nat_crs, grid)

        for cid in domain_cells:
            m = cell_id == int(cid)
            w_sum = float(w_plane[m].sum())
            w_dom = float(w_plane[m & in_domain].sum())
            if w_sum > 0.0 and w_dom < 1.0 - WEIGHT_TOL:
                clip_log.append({
                    "sector": sector_id,
                    "cell_id": int(cid),
                    "pollutant": pol,
                    "weight_sum_full_cell": w_sum,
                    "weight_sum_in_domain": w_dom,
                    "clipped_mass_fraction": w_dom / w_sum,
                })

    da = xr.DataArray(
        out_stack,
        dims=("pollutant", "y", "x"),
        coords={"pollutant": pollutants, "y": np.arange(grid.height), "x": np.arange(grid.width)},
    )
    return da, weight_log, [], clip_log


def prepare_sector_cams(
    cams_nc: Path,
    sector_id: str,
    country: str,
    year: int,
    pollutants: list[str],
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    sec_yaml = load_sector_yaml(sector_id)
    cps = sec_yaml.get("cams_area_sources")
    if cps:
        return load_cams_area_cells(
            cams_nc,
            year=int(cps["year"]),
            country_iso3=country_iso3(country),
            emission_category_indices=list(cps["emission_category_indices"]),
            source_type_indices=list(cps["source_type_indices"]),
            pollutants=pollutants,
        )
    cpt = sec_yaml["cams_point_sources"]
    grid = load_cams_grid_meta(cams_nc)
    pts = load_cams_points(
        cams_nc,
        year=int(cpt["year"]),
        country_iso3=country_iso3(country),
        emission_category_indices=list(cpt["emission_category_indices"]),
        source_type_indices=list(cpt["source_type_indices"]),
        pollutants=pollutants,
    )
    valid = point_cell_ids(pts, grid)
    cells = {cid: {"pollutants_within_cell": {p: 0.0 for p in pollutants}} for cid in valid}
    return cells, grid
