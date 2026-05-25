from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np

from UrbEm_Visualizer.downscaling.area import downscale_area, prepare_sector_cams
from UrbEm_Visualizer.dataset_loaders.tif_grid import cell_id_plane
from UrbEm_Visualizer.downscaling.merge import merge_grids
from UrbEm_Visualizer.downscaling.point import run_point_sector
from UrbEm_Visualizer.writer.downscale_export import export_run
from UrbEm_Visualizer.downscaling.sector_meta import sector_label, sector_mode, sector_order
from UrbEm_Visualizer.downscaling.spatial import (
    fine_grid_from_reference,
    find_reference_tif,
    resolve_path,
)
from UrbEm_Visualizer.paths import project_root
from UrbEm_Visualizer.writer.create_configuration import load_yaml


def output_dir_for_run(config_path: Path, config: dict) -> Path:
    out = config.get("output") or {}
    run_name = out.get("run_name")
    if not run_name:
        run_name = config_path.stem
    return project_root() / "Output" / "UrbEm" / str(run_name)


def run_downscaling(
    config_path: Path,
    *,
    on_progress: Callable[[dict], None] | None = None,
    cancel_flag: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    root = project_root()
    for key in ("country", "year", "domain", "pollutants", "output", "paths"):
        if key not in config:
            raise KeyError(f"run config missing {key!r}")

    domain = config["domain"]
    pollutants = list(config["pollutants"])
    layer_mode = config["output"]["layer_mode"]
    fmt = config["output"]["format"]
    order = sector_order(config)
    ref = find_reference_tif(config, order)
    if ref is None:
        raise ValueError("no area_weights or point_source path in configuration")

    grid = fine_grid_from_reference(ref, domain)
    cams_nc = resolve_path(config["paths"]["cams"], root)
    out_dir = output_dir_for_run(config_path, config)

    sectors_state = [
        {
            "id": sid,
            "label": sector_label(sid),
            "status": "waiting",
            "pollutants": pollutants,
            "pollutants_done": [],
        }
        for sid in order
    ]
    state = {
        "status": "running",
        "sectors": sectors_state,
        "error": None,
        "output_dir": str(out_dir),
    }

    def push() -> None:
        if on_progress:
            on_progress(dict(state))

    push()
    sector_results: dict[str, dict[str, Any]] = {}
    weight_check_log: dict[str, Any] = {}
    clip_log: list[dict] = []
    merged_acc = None
    cell_id_cache: dict[str, np.ndarray] = {}

    try:
        for i, sid in enumerate(order):
            if cancel_flag and cancel_flag():
                state["status"] = "cancelled"
                push()
                return state

            sectors_state[i]["status"] = "running"
            sectors_state[i]["pollutants_done"] = []
            push()
            sec = config["sectors"][sid]
            mode = sector_mode(sid)
            res: dict[str, Any] = {}
            area_da = None
            point_da = None

            cams_cells, cams_grid_meta = prepare_sector_cams(
                cams_nc, sid, config["country"], int(config["year"]), pollutants,
            )

            if cams_grid_meta:
                cache_key = sid
                if cache_key not in cell_id_cache:
                    valid = frozenset(cams_cells.keys()) if cams_cells else frozenset()
                    cell_id_cache[cache_key] = cell_id_plane(grid, cams_grid_meta, valid)
                cell_id = cell_id_cache[cache_key]
            else:
                cell_id = np.full((grid.height, grid.width), -1, dtype=np.int32)

            aw = sec.get("area_weights") or {}
            if aw.get("path") and mode in ("both", "area_only"):
                aw_path = resolve_path(aw["path"], root)
                def _pol_done(pol: str) -> None:
                    done = sectors_state[i]["pollutants_done"]
                    if pol not in done:
                        done.append(pol)
                    push()

                area_da, wlog, fails, sector_clip = downscale_area(
                    grid=grid,
                    area_path=aw_path,
                    sector_id=sid,
                    domain=domain,
                    pollutants=pollutants,
                    cams_cells=cams_cells or {},
                    cams_grid=cams_grid_meta,
                    on_pollutant_done=_pol_done,
                )
                weight_check_log[sid] = wlog
                if fails:
                    f0 = fails[0]
                    msg = (
                        f"Weight check failed — sector {sid}, pollutant {f0['pollutant']}, "
                        f"cell {f0['cell_id']}, sum={f0['sum']:.6f}"
                    )
                    sectors_state[i]["status"] = "error"
                    state["status"] = "error"
                    state["error"] = msg
                    push()
                    return state
                res["area_emission"] = area_da
                clip_log.extend(sector_clip)

            ps = sec.get("point_source") or {}
            if ps.get("path") and mode in ("both", "point_only"):
                ps_path = resolve_path(ps["path"], root)
                aw_path = None
                if aw.get("path"):
                    aw_path = resolve_path(aw["path"], root)
                point_da, g1, g2, g3 = run_point_sector(
                    grid=grid,
                    link_path=ps_path,
                    cams_nc=cams_nc,
                    sector_id=sid,
                    country=config["country"],
                    year=int(config["year"]),
                    pollutants=pollutants,
                    domain=domain,
                    layer_mode=layer_mode,
                    cell_id=cell_id,
                    area_weight_path=aw_path,
                )
                res["point_emission"] = point_da
                res["point_appointed"] = g1
                res["point_not_appointed"] = g2
                res["point_unmatched"] = g3
                if not aw.get("path") or mode not in ("both", "area_only"):
                    sectors_state[i]["pollutants_done"] = list(pollutants)
                    push()

            sector_results[sid] = res
            sectors_state[i]["pollutants_done"] = list(pollutants)
            sectors_state[i]["status"] = "done"
            push()

        merged_acc = None
        if layer_mode == "merged":
            for sid in order:
                r = sector_results.get(sid, {})
                merged_acc = merge_grids(
                    merged_acc,
                    merge_grids(
                        r.get("area_emission"),
                        r.get("point_emission"),
                        pollutants,
                        (grid.height, grid.width),
                    ),
                    pollutants,
                    (grid.height, grid.width),
                )

        export_run(
            out_dir,
            config=config,
            fmt=fmt,
            layer_mode=layer_mode,
            sector_results=sector_results,
            merged=merged_acc if layer_mode == "merged" else None,
            weight_check_log=weight_check_log,
            clip_log=clip_log,
            grid_transform=grid.transform,
            crs=grid.crs,
        )
        state["status"] = "done"
        push()
        return state
    except Exception as exc:
        state["status"] = "error"
        state["error"] = f"{exc}\n{traceback.format_exc()}"
        push()
        return state
