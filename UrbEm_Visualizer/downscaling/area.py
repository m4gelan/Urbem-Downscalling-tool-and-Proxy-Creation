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
    aggregate_plane_to_grid,
    cell_id_on_raster,
    cell_id_plane,
    pixels_in_domain_bbox,
    read_weight_stack_native,
    reproject_plane_to_grid,
)
from UrbEm_Visualizer.downscaling.sector_meta import load_sector_yaml
from UrbEm_Visualizer.downscaling.spatial import FineGrid, NativeGridMeta
from UrbEm_Visualizer.pollutants import band_index_for_pollutant

# Proxy normalizes each CAMS cell to sum=1 in float32 before writing the GeoTIFF.
# Re-summing stored float32 pixels can drift (~2% on large cells) — repaired on read, not a bad proxy.
WEIGHT_TOL = 1.5e-2  # domain-clip warning threshold only
WEIGHT_FAIL_TOL = 5e-2  # hard fail: likely broken weights, not float32 round-trip


def cell_weight_sums(weights: np.ndarray, cell_id: np.ndarray) -> np.ndarray:
    flat_c = cell_id.ravel()
    flat_w = weights.ravel()
    valid = flat_c >= 0
    if not np.any(valid):
        return np.zeros(1, dtype=np.float64)
    c = flat_c[valid].astype(np.int64)
    max_c = int(c.max())
    # float32 pixels, float64 sum — only the per-cell totals, not the full raster
    return np.bincount(c, weights=flat_w[valid].astype(np.float64), minlength=max_c + 1)


def renormalize_weights_per_cams_cell(weights: np.ndarray, cell_id: np.ndarray) -> np.ndarray:
    """Scale weights so each CAMS cell sums to 1 (fixes float32 GeoTIFF round-trip drift)."""
    sums = cell_weight_sums(weights, cell_id)
    out = weights.astype(np.float32, copy=True)
    ok = cell_id >= 0
    cid = cell_id[ok].astype(np.int64)
    s = sums[cid]
    good = s > 0.0
    flat = np.zeros(int(ok.sum()), dtype=np.float32)
    if np.any(good):
        flat[good] = (weights[ok][good].astype(np.float64) / s[good]).astype(np.float32)
    out[ok] = flat
    return out


def check_weights_native(
    weights: np.ndarray,
    cell_id: np.ndarray,
    pollutant: str,
    cells_to_check: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Check raw stored sums; fail only on large drift. Return (failures, repairs)."""
    sums = cell_weight_sums(weights, cell_id)
    fails: list[dict[str, Any]] = []
    repairs: list[dict[str, Any]] = []
    for cid in cells_to_check:
        ic = int(cid)
        if ic < 0 or ic >= sums.size:
            continue
        s = float(sums[ic])
        drift = abs(s - 1.0)
        if drift > WEIGHT_FAIL_TOL:
            fails.append({"cell_id": ic, "pollutant": pollutant, "sum": s})
        elif drift > 1e-6:
            repairs.append({"cell_id": ic, "pollutant": pollutant, "sum_before": s})
    return fails, repairs


def prepare_weight_plane(
    weights: np.ndarray,
    cell_id: np.ndarray,
    pollutant: str,
    cells_to_check: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    fails, repairs = check_weights_native(weights, cell_id, pollutant, cells_to_check)
    if fails:
        return weights, fails, repairs
    if repairs:
        weights = renormalize_weights_per_cams_cell(weights, cell_id)
    return weights, fails, repairs


def downscale_area(
    *,
    grid: FineGrid,
    area_path: Path,
    sector_id: str,
    domain: dict,
    pollutants: list[str],
    cams_cells: dict[int, dict[str, Any]],
    cams_grid: dict[str, Any],
    output_resolution_m: int,
    native_meta: NativeGridMeta,
    on_pollutant_done: Callable[[str], None] | None = None,
) -> tuple[xr.DataArray, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    labels, native_stack, nat_tr, nat_crs = read_weight_stack_native(area_path)
    h, w = int(native_stack.shape[1]), int(native_stack.shape[2])
    valid_keys = frozenset(cams_cells.keys()) if cams_cells else frozenset()
    cell_id = cell_id_on_raster(nat_tr, nat_crs, h, w, cams_grid, valid_keys)
    in_domain = pixels_in_domain_bbox(nat_tr, nat_crs, h, w, domain)
    domain_cells = np.unique(cell_id[in_domain & (cell_id >= 0)])
    use_aggregate = output_resolution_m > int(native_meta.res_x)

    mass_lut = cams_mass_lookup(cams_cells, pollutants)
    n_pol = len(pollutants)
    out_stack = np.zeros((n_pol, grid.height, grid.width), dtype=np.float32)
    clip_log: list[dict[str, Any]] = []
    weight_log: dict[str, Any] = {"sector": sector_id, "pollutants": {}, "passed": True}

    cell_id_out = None
    if use_aggregate:
        cell_id_out = cell_id_plane(grid, cams_grid, valid_keys)

    for pi, pol in enumerate(pollutants):
        bi = band_index_for_pollutant(labels, pol)
        w_native, fails, repairs = prepare_weight_plane(
            native_stack[bi], cell_id, pol, domain_cells,
        )
        weight_log["pollutants"][pol] = {
            "failures": fails,
            "repairs": repairs,
            "cells_checked": int(domain_cells.size),
        }
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
        if use_aggregate:
            w_plane = aggregate_plane_to_grid(w_native, nat_tr, nat_crs, grid)
            ok = (cell_id_out >= 0) & grid.domain_mask
            out_stack[pi][ok] = lut[cell_id_out[ok]] * w_plane[ok]
        else:
            ok = cell_id >= 0
            e_native = np.zeros((h, w), dtype=np.float32)
            e_native[ok] = lut[cell_id[ok]] * w_native[ok]
            out_stack[pi] = reproject_plane_to_grid(e_native, nat_tr, nat_crs, grid)

        if on_pollutant_done:
            on_pollutant_done(pol)

        for cid in domain_cells:
            m = cell_id == int(cid)
            w_sum = float(w_native[m].sum())
            w_dom = float(w_native[m & in_domain].sum())
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
    cps = sec_yaml.get("cams_area_emissions")
    if sector_id == "G_Shipping":
        from proxy.core.alias import resolve_country_profile
        from proxy.core.cams_sector_config import load_shipping_sector_cells_mask
        from UrbEm_Visualizer.paths import project_root

        prof = resolve_country_profile(country)
        fp = sec_yaml.get("filepaths") or {}
        nuts_rel = (fp.get("NUTS REGIONS") or {}).get("path")
        if not nuts_rel:
            raise ValueError("G_Shipping sector config: filepaths.NUTS REGIONS.path required")
        return load_shipping_sector_cells_mask(
            cams_nc,
            sec_yaml,
            country_profile=prof,
            country_iso3=prof["ISO3"],
            pollutants=pollutants,
            nuts_path=project_root() / str(nuts_rel).replace("\\", "/"),
            crs="EPSG:3035",
            resolution_m=100.0,
            pad_m=10.0,
        )
    if cps:
        return load_cams_area_cells(
            cams_nc,
            year=int(year),
            country_iso3=country_iso3(country),
            emission_category_indices=list(cps["emission_category_indices"]),
            source_type_indices=list(cps["source_type_indices"]),
            pollutants=pollutants,
        )
    cpt = sec_yaml["cams_point_sources"]
    grid = load_cams_grid_meta(cams_nc)
    pts = load_cams_points(
        cams_nc,
        year=int(year),
        country_iso3=country_iso3(country),
        emission_category_indices=list(cpt["emission_category_indices"]),
        source_type_indices=list(cpt["source_type_indices"]),
        pollutants=pollutants,
    )
    valid = point_cell_ids(pts, grid)
    cells = {cid: {"pollutants_within_cell": {p: 0.0 for p in pollutants}} for cid in valid}
    return cells, grid
