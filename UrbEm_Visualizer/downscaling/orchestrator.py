from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np

from UrbEm_Visualizer.downscaling.area import downscale_area, prepare_sector_cams
from UrbEm_Visualizer.downscaling.roads import downscale_roads_sector
from UrbEm_Visualizer.dataset_loaders.tif_grid import cell_id_plane
from UrbEm_Visualizer.downscaling.merge import merge_grids
from UrbEm_Visualizer.downscaling.output_config import parse_point_matching
from UrbEm_Visualizer.downscaling.point import (
    add_unmatched_to_cams_cells,
    load_point_records,
    point_match_stats,
    run_point_sector,
)
from UrbEm_Visualizer.writer.downscale_export import export_run
from UrbEm_Visualizer.downscaling.sector_meta import sector_label, sector_mode, sector_order
from UrbEm_Visualizer.downscaling.spatial import (
    build_output_grid,
    find_reference_tif,
    native_grid_metadata,
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


def collect_point_match_stats(config: dict, root: Path) -> dict[str, Any]:
    domain = config["domain"]
    pollutants = list(config["pollutants"])
    order = sector_order(config)
    by_sector: dict[str, dict[str, int]] = {}
    labels: dict[str, str] = {}
    partial_total = 0
    for sid in order:
        labels[sid] = sector_label(sid)
        sec = config["sectors"][sid]
        mode = sector_mode(sid)
        ps = sec.get("point_source") or {}
        if not ps.get("path") or mode not in ("both", "point_only"):
            continue
        ps_path = resolve_path(ps["path"], root)
        st = point_match_stats(ps_path, domain, pollutants)
        by_sector[sid] = st
        partial_total += int(st.get("partial_match", 0))
    fac_out = sum(int(st.get("facility_outside_domain", 0)) for st in by_sector.values())
    cams_out = sum(int(st.get("cams_outside_domain", 0)) for st in by_sector.values())
    return {
        "by_sector": by_sector,
        "sector_labels": labels,
        "partial_match_total": partial_total,
        "facility_outside_domain": fac_out,
        "cams_outside_domain": cams_out,
    }


def _sector_step_plan(sec: dict, mode: str) -> list[tuple[str, float]]:
    aw = sec.get("area_weights") or {}
    ps = sec.get("point_source") or {}
    has_area = mode in ("both", "area_only") and (bool(aw.get("path")) or bool(aw.get("categories")))
    has_point = bool(ps.get("path")) and mode in ("both", "point_only")
    steps: list[tuple[str, float]] = [("Loading CAMS", 0.12)]
    if has_area and has_point:
        steps.append(("Area downscaling", 0.48))
        steps.append(("Point sources", 0.35))
    elif has_area:
        steps.append(("Area downscaling", 0.83))
    elif has_point:
        steps.append(("Point sources", 0.83))
    total = sum(w for _, w in steps)
    return [(label, w / total) for label, w in steps]


def run_downscaling(
    config_path: Path,
    *,
    on_progress: Callable[[dict], None] | None = None,
    cancel_flag: Callable[[], bool] | None = None,
    partial_match_handling: str | None = None,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    if partial_match_handling:
        config = dict(config)
        config["output"] = dict(config["output"])
        config["output"]["partial_match_handling"] = partial_match_handling
    root = project_root()
    for key in ("country", "year", "domain", "pollutants", "output", "paths"):
        if key not in config:
            raise KeyError(f"run config missing {key!r}")

    domain = config["domain"]
    pollutants = list(config["pollutants"])
    output_cfg = config["output"]
    if "grid_resolution_m" not in output_cfg:
        raise KeyError("run config missing output.grid_resolution_m")
    pm_cfg = parse_point_matching(output_cfg)
    procedure = pm_cfg["procedure"]
    burn_unmatched_to_area = procedure == "merged" or (
        procedure == "separate" and pm_cfg["unmatched"] == "burn_to_area"
    )
    pm_handling = partial_match_handling or output_cfg.get("partial_match_handling")
    skip_not_app_for_area = pm_handling == "facility_or_drop"
    fmt = output_cfg["format"]
    grid_resolution_m = int(output_cfg["grid_resolution_m"])
    order = sector_order(config)
    ref = find_reference_tif(config, order)
    if ref is None:
        raise ValueError("no area_weights or point_source path in configuration")

    native_meta = native_grid_metadata(ref)
    grid = build_output_grid(domain, grid_resolution_m, native_meta)
    cams_nc = resolve_path(config["paths"]["cams"], root)
    out_dir = output_dir_for_run(config_path, config)

    sectors_state = [
        {
            "id": sid,
            "label": sector_label(sid),
            "status": "waiting",
            "progress": 0,
            "step": "",
        }
        for sid in order
    ]
    state = {
        "status": "running",
        "sectors": sectors_state,
        "error": None,
        "output_dir": str(out_dir),
        "point_matching": pm_cfg,
        "point_match_stats": {},
    }

    def push() -> None:
        if on_progress:
            on_progress(dict(state))

    for sid in order:
        sec = config["sectors"][sid]
        mode = sector_mode(sid)
        ps = sec.get("point_source") or {}
        if ps.get("path") and mode in ("both", "point_only"):
            ps_path = resolve_path(ps["path"], root)
            state["point_match_stats"][sid] = point_match_stats(
                ps_path, domain, pollutants,
            )
    state["partial_match_total"] = sum(
        int(st.get("partial_match", 0)) for st in state["point_match_stats"].values()
    )

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

            sec = config["sectors"][sid]
            mode = sector_mode(sid)
            res: dict[str, Any] = {}
            plan = _sector_step_plan(sec, mode)

            sectors_state[i]["status"] = "running"
            sectors_state[i]["progress"] = 0
            sectors_state[i]["step"] = plan[0][0]
            push()

            def _set_progress(step: int, sub: float, label: str | None = None) -> None:
                base = sum(w for _, w in plan[:step])
                _, w = plan[step]
                pct = int((base + w * max(0.0, min(1.0, sub))) * 100)
                sectors_state[i]["progress"] = min(pct, 99)
                sectors_state[i]["step"] = label or plan[step][0]
                push()

            step = 0
            _set_progress(step, 0.0)
            cams_cells, cams_grid_meta = prepare_sector_cams(
                cams_nc, sid, config["country"], int(config["year"]), pollutants,
            )
            _set_progress(step, 1.0)
            step += 1

            ps = sec.get("point_source") or {}
            point_records = None
            if ps.get("path") and mode in ("both", "point_only"):
                ps_path = resolve_path(ps["path"], root)
                if on_progress:
                    sectors_state[i]["step"] = "Classifying point sources"
                    push()
                point_records = load_point_records(ps_path, domain, pollutants)
                if burn_unmatched_to_area and mode == "both":
                    _, not_app, unmatched_burn = point_records
                    extra = list(unmatched_burn)
                    if procedure == "merged" and not skip_not_app_for_area:
                        extra.extend(not_app)
                    if extra:
                        if not cams_grid_meta:
                            raise ValueError(
                                f"{sid}: unmatched points need CAMS grid metadata for area downscaling"
                            )
                        cams_cells = add_unmatched_to_cams_cells(
                            cams_cells, extra, cams_grid_meta, pollutants,
                        )
                elif burn_unmatched_to_area and mode == "point_only":
                    _, _, unmatched_burn = point_records
                    if unmatched_burn:
                        raise ValueError(
                            f"{sid}: merged / burn_to_area needs area weights (sector is point_only)"
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
            if mode in ("both", "area_only") and (aw.get("path") or aw.get("categories")):
                _set_progress(step, 0.0)
                if sid == "F_Roads" and aw.get("categories"):
                    cat_paths = {
                        cat: resolve_path(p, root) for cat, p in aw["categories"].items()
                    }

                    def _roads_prog(label: str, sub: float) -> None:
                        _set_progress(step, sub, label)

                    area_da, wlog, fails, sector_clip = downscale_roads_sector(
                        grid=grid,
                        category_paths=cat_paths,
                        country=config["country"],
                        domain=domain,
                        pollutants=pollutants,
                        cams_nc=cams_nc,
                        output_resolution_m=grid_resolution_m,
                        native_meta=native_meta,
                        on_progress=_roads_prog,
                    )
                else:
                    aw_path = resolve_path(aw["path"], root)
                    n_pol = len(pollutants) or 1

                    def _pol_done(pol: str) -> None:
                        pi = pollutants.index(pol) if pol in pollutants else n_pol - 1
                        _set_progress(step, (pi + 1) / n_pol, f"Area — {pol}")

                    area_da, wlog, fails, sector_clip = downscale_area(
                        grid=grid,
                        area_path=aw_path,
                        sector_id=sid,
                        domain=domain,
                        pollutants=pollutants,
                        cams_cells=cams_cells or {},
                        cams_grid=cams_grid_meta,
                        output_resolution_m=grid_resolution_m,
                        native_meta=native_meta,
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
                _set_progress(step, 1.0)
                step += 1

            if ps.get("path") and mode in ("both", "point_only"):
                ps_path = resolve_path(ps["path"], root)
                _set_progress(step, 0.0)

                def _point_prog(label: str, sub: float) -> None:
                    _set_progress(step, sub, label)

                point_da, g1, g2, g3 = run_point_sector(
                    grid=grid,
                    link_path=ps_path,
                    cams_nc=cams_nc,
                    sector_id=sid,
                    country=config["country"],
                    year=int(config["year"]),
                    pollutants=pollutants,
                    domain=domain,
                    procedure=procedure,
                    burn_unmatched_to_area=burn_unmatched_to_area,
                    partial_match_handling=pm_handling,
                    on_progress=_point_prog,
                )
                res["point_emission"] = point_da
                res["point_appointed"] = g1
                res["point_not_appointed"] = g2
                res["point_unmatched"] = g3
                _set_progress(step, 1.0)

            if procedure == "merged":
                res["sector_merged"] = merge_grids(
                    res.get("area_emission"),
                    res.get("point_emission"),
                    pollutants,
                    (grid.height, grid.width),
                )

            sector_results[sid] = res
            sectors_state[i]["progress"] = 100
            sectors_state[i]["step"] = "Complete"
            sectors_state[i]["status"] = "done"
            push()

        merged_acc = None
        if procedure == "merged":
            for sid in order:
                sm = sector_results.get(sid, {}).get("sector_merged")
                if sm is not None:
                    merged_acc = merge_grids(
                        merged_acc, sm, pollutants, (grid.height, grid.width),
                    )

        export_run(
            out_dir,
            config=config,
            fmt=fmt,
            procedure=procedure,
            sector_results=sector_results,
            merged=merged_acc if procedure == "merged" else None,
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
