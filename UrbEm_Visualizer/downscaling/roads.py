from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import xarray as xr

from UrbEm_Visualizer.dataset_loaders.cams_alias import country_iso3
from UrbEm_Visualizer.dataset_loaders.cams_emissions import load_cams_area_cells, load_cams_grid_meta
from UrbEm_Visualizer.downscaling.area import downscale_area
from UrbEm_Visualizer.downscaling.merge import merge_grids
from UrbEm_Visualizer.downscaling.sector_meta import load_sector_yaml, roads_category_names
from UrbEm_Visualizer.downscaling.spatial import FineGrid, NativeGridMeta


def roads_proxy_paths(
    proxy_root: Path,
    country_tag: str,
    year: int,
    categories: list[str] | None = None,
) -> dict[str, Path]:
    folder = proxy_root / "F_Roads"
    cats = categories or roads_category_names()
    out: dict[str, Path] = {}
    for cat in cats:
        p = folder / f"F_Roads_{country_tag}_{cat}_{year}.tif"
        if p.is_file():
            out[cat] = p
    return out


def downscale_roads_sector(
    *,
    grid: FineGrid,
    category_paths: dict[str, str | Path],
    country: str,
    domain: dict,
    pollutants: list[str],
    cams_nc: Path,
    output_resolution_m: int,
    native_meta: NativeGridMeta,
    on_progress: Callable[[str, float], None] | None = None,
) -> tuple[xr.DataArray, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    sec_yaml = load_sector_yaml("F_Roads")
    f_cats = sec_yaml.get("cams_f_categories") or {}
    cps = sec_yaml["cams_area_sources"]
    cams_year = int(cps["year"])
    st = list(cps["source_type_indices"])
    iso3 = country_iso3(country)

    combined: xr.DataArray | None = None
    weight_log: dict[str, Any] = {"sector": "F_Roads", "categories": {}, "passed": True}
    clip_log: list[dict[str, Any]] = []
    order = [c for c in roads_category_names() if c in category_paths]
    n = len(order) or 1

    for i, cat in enumerate(order):
        if cat not in f_cats:
            raise ValueError(f"F_Roads: unknown category {cat!r} in config")
        if on_progress:
            on_progress(f"Area — {cat}", i / n)

        ec = [int(f_cats[cat]["emission_category_index"])]
        cams_cells, cams_grid = load_cams_area_cells(
            cams_nc,
            year=cams_year,
            country_iso3=iso3,
            emission_category_indices=ec,
            source_type_indices=st,
            pollutants=pollutants,
        )
        if not cams_grid:
            cams_grid = load_cams_grid_meta(cams_nc)

        tif = Path(category_paths[cat])
        if not tif.is_file():
            raise FileNotFoundError(f"F_Roads proxy missing: {tif}")
        da, wlog, fails, clips = downscale_area(
            grid=grid,
            area_path=tif,
            sector_id=f"F_Roads/{cat}",
            domain=domain,
            pollutants=pollutants,
            cams_cells=cams_cells,
            cams_grid=cams_grid,
            output_resolution_m=output_resolution_m,
            native_meta=native_meta,
        )
        weight_log["categories"][cat] = wlog
        if fails:
            weight_log["passed"] = False
            empty = da * 0.0
            return empty, weight_log, fails, clip_log
        clip_log.extend(clips)
        combined = merge_grids(combined, da, pollutants, (grid.height, grid.width))
        if on_progress:
            on_progress(f"Area — {cat}", (i + 1) / n)

    if combined is None:
        raise ValueError("F_Roads: no category proxy paths")
    return combined, weight_log, [], clip_log
