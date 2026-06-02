from __future__ import annotations

import re
import sys
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

_PKG = Path(__file__).resolve().parent.parent
_ROOT = _PKG.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from UrbEm_Visualizer.dataset_loaders.cams_grid import (
    cams_job_status,
    start_cams_grid_job,
    transform_bbox,
)
from UrbEm_Visualizer.dataset_loaders.check import check_input, list_countries, load_expected
from UrbEm_Visualizer.paths import config_dir, project_root, runs_dir
from UrbEm_Visualizer.pollutants import AVAILABLE_POLLUTANTS
from UrbEm_Visualizer.ui import dialogs
from UrbEm_Visualizer.writer.create_configuration import (
    apply_manual_paths,
    config_from_check,
    load_yaml,
    merge_check_paths,
    new_writer_config,
    save_run_config,
)
from UrbEm_Visualizer.writer.inputs_table import build_inputs_table

app = Flask(__name__, static_folder="static", static_url_path="")
_SESSION: dict = {
    "config": None,
    "config_path": None,
    "absent_sources": [],
    "country": None,
}


def _session_absent() -> list[dict]:
    return list(_SESSION.get("absent_sources") or [])


def _config_response() -> dict:
    cfg = _SESSION.get("config")
    out = {"config": cfg, "config_path": _SESSION.get("config_path")}
    if cfg:
        out["inputs_table"] = build_inputs_table(cfg)
        out["runs_dir"] = str(runs_dir().resolve())
    return out


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/<path:asset>")
def static_asset(asset):
    return send_from_directory(app.static_folder, asset)


@app.route("/api/pollutants", methods=["GET"])
def api_pollutants():
    return jsonify({"pollutants": AVAILABLE_POLLUTANTS})


@app.route("/api/countries", methods=["GET"])
def api_countries():
    try:
        return jsonify({"countries": list_countries()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/expected-sectors", methods=["GET"])
def api_expected_sectors():
    try:
        spec = load_expected()
        required = [
            {"id": sid, "mode": sec["mode"], "optional": False}
            for sid, sec in spec["sectors"].items()
        ]
        optional = [
            {"id": sid, "mode": sec["mode"], "optional": True}
            for sid, sec in (spec.get("optional_sectors") or {}).items()
        ]
        return jsonify({
            "sectors": required + optional,
            "required": required,
            "optional": optional,
            "year": spec["year"],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/check-input", methods=["POST"])
def api_check_input():
    data = request.get_json() or {}
    country = data.get("country")
    if not country:
        return jsonify({"error": "country required"}), 400
    try:
        absent = data.get("absent_sources")
        if absent is None:
            absent = _session_absent()
        result = check_input(country, absent_sources=absent)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/waiver/mark", methods=["POST"])
def api_waiver_mark():
    data = request.get_json() or {}
    sector = data.get("sector")
    role = data.get("role")
    if not sector or not role:
        return jsonify({"error": "sector and role required"}), 400
    absent = _session_absent()
    key = (sector, role)
    if key not in {(a["sector"], a["role"]) for a in absent}:
        absent.append({"sector": sector, "role": role})
    _SESSION["absent_sources"] = absent
    country = data.get("country") or _SESSION.get("country")
    check = None
    if country:
        check = check_input(country, absent_sources=absent)
    return jsonify({"absent_sources": absent, "check": check})


@app.route("/api/waiver/clear", methods=["POST"])
def api_waiver_clear():
    _SESSION["absent_sources"] = []
    return jsonify({"absent_sources": []})


@app.route("/api/dialog/open-yaml", methods=["POST"])
def api_dialog_open_yaml():
    path = dialogs.pick_yaml_open()
    if not path:
        return jsonify({"cancelled": True})
    try:
        cfg = load_yaml(Path(path))
        check = check_input(cfg.get("country", ""), absent_sources=list(cfg.get("absent_sources") or []))
        cfg = merge_check_paths(cfg, check)
        _SESSION["config"] = cfg
        _SESSION["config_path"] = path
        _SESSION["absent_sources"] = list(cfg.get("absent_sources") or [])
        _SESSION["country"] = cfg.get("country")
        out = {"cancelled": False, "path": path, "config": cfg}
        out["inputs_table"] = build_inputs_table(cfg)
        return jsonify(out)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/dialog/pick-file", methods=["POST"])
def api_dialog_pick_file():
    data = request.get_json() or {}
    title = data.get("title") or "Select file"
    path = dialogs.pick_file(title)
    if not path:
        return jsonify({"cancelled": True})
    return jsonify({"cancelled": False, "path": path})


@app.route("/api/writer/new", methods=["POST"])
def api_writer_new():
    data = request.get_json() or {}
    country = data.get("country")
    if not country:
        return jsonify({"error": "country required"}), 400
    try:
        _SESSION["country"] = country
        _SESSION["absent_sources"] = []
        cfg = new_writer_config(country)
        _SESSION["config"] = cfg
        _SESSION["config_path"] = None
        return jsonify({"config": cfg})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/writer/from-check", methods=["POST"])
def api_writer_from_check():
    data = request.get_json() or {}
    country = data.get("country")
    pollutants = data.get("pollutants")
    if not country:
        return jsonify({"error": "country required"}), 400
    if not pollutants or not isinstance(pollutants, list):
        return jsonify({"error": "pollutants required (non-empty list)"}), 400
    _SESSION["country"] = country
    absent = data.get("absent_sources")
    if absent is None:
        absent = _session_absent()
    else:
        _SESSION["absent_sources"] = list(absent)
    try:
        result = check_input(country, absent_sources=absent)
        if not result["ok"]:
            return jsonify({"ok": False, "check": result, "absent_sources": absent}), 400
        cfg = config_from_check(country, absent_sources=absent)
        cfg["pollutants"] = list(pollutants)
        _SESSION["config"] = cfg
        _SESSION["config_path"] = None
        return jsonify({
            "ok": True,
            "config": cfg,
            "check": result,
            "inputs_table": build_inputs_table(cfg),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/writer/apply-manual", methods=["POST"])
def api_writer_apply_manual():
    data = request.get_json() or {}
    manual = data.get("manual")
    pollutants = data.get("pollutants")
    if not isinstance(manual, dict):
        return jsonify({"error": "manual mapping required"}), 400
    if not pollutants or not isinstance(pollutants, list):
        return jsonify({"error": "pollutants required (non-empty list)"}), 400
    if _SESSION["config"] is None:
        return jsonify({"error": "no active configuration; start writer or load a file"}), 400
    try:
        cfg = apply_manual_paths(_SESSION["config"], manual)
        cfg["pollutants"] = list(pollutants)
        _SESSION["config"] = cfg
        _SESSION["absent_sources"] = list(cfg.get("absent_sources") or [])
        return jsonify({"config": cfg, "inputs_table": build_inputs_table(cfg)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/config/pollutants", methods=["POST"])
def api_config_pollutants():
    data = request.get_json() or {}
    pollutants = data.get("pollutants")
    if not pollutants or not isinstance(pollutants, list):
        return jsonify({"error": "pollutants required"}), 400
    if _SESSION.get("config") is None:
        return jsonify({"error": "no active configuration"}), 400
    _SESSION["config"]["pollutants"] = list(pollutants)
    return _config_response()


@app.route("/api/cams/grid/start", methods=["POST"])
def api_cams_grid_start():
    cfg = _SESSION.get("config")
    if not cfg:
        return jsonify({"error": "no active configuration"}), 400
    pollutants = cfg.get("pollutants")
    if not pollutants:
        return jsonify({"error": "select pollutants before loading CAMS grid"}), 400
    country = cfg.get("country") or _SESSION.get("country")
    cams_rel = (cfg.get("paths") or {}).get("cams")
    if not cams_rel:
        return jsonify({"error": "CAMS path missing"}), 400
    cams_path = Path(cams_rel)
    if not cams_path.is_absolute():
        cams_path = project_root() / cams_rel
    if not cams_path.is_file():
        return jsonify({"error": f"CAMS file not found: {cams_path}"}), 404
    try:
        job_id = start_cams_grid_job(
            cams_path, country, int(cfg["year"]), list(pollutants), config=cfg
        )
        return jsonify({"job_id": job_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/cams/grid/status/<job_id>", methods=["GET"])
def api_cams_grid_status(job_id: str):
    st = cams_job_status(job_id)
    if st is None:
        return jsonify({"error": "unknown job"}), 404
    out = {
        "done": st["done"],
        "error": st.get("error"),
        "progress": st.get("progress", 0),
        "message": st.get("message", ""),
        "sector": st.get("sector"),
    }
    if st.get("done") and st.get("geojson") and not st.get("error"):
        out["geojson"] = st["geojson"]
    return jsonify(out)


@app.route("/api/domain/transform", methods=["POST"])
def api_domain_transform():
    data = request.get_json() or {}
    for k in ("xmin", "ymin", "xmax", "ymax", "from_crs", "to_crs"):
        if k not in data:
            return jsonify({"error": f"{k} required"}), 400
    try:
        out = transform_bbox(
            float(data["xmin"]),
            float(data["ymin"]),
            float(data["xmax"]),
            float(data["ymax"]),
            str(data["from_crs"]),
            str(data["to_crs"]),
        )
        return jsonify(out)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/config/domain", methods=["POST"])
def api_config_domain():
    data = request.get_json() or {}
    domain = data.get("domain")
    if not isinstance(domain, dict):
        return jsonify({"error": "domain object required"}), 400
    for k in ("crs", "xmin", "ymin", "xmax", "ymax"):
        if k not in domain:
            return jsonify({"error": f"domain.{k} required"}), 400
    if _SESSION.get("config") is None:
        return jsonify({"error": "no active configuration"}), 400
    _SESSION["config"]["domain"] = domain
    resp = _config_response()
    wgs = transform_bbox(
        float(domain["xmin"]),
        float(domain["ymin"]),
        float(domain["xmax"]),
        float(domain["ymax"]),
        str(domain["crs"]),
        "EPSG:4326",
    )
    resp["domain_wgs84"] = wgs
    return jsonify(resp)


@app.route("/api/config/output", methods=["POST"])
def api_config_output():
    data = request.get_json() or {}
    output = data.get("output")
    if not isinstance(output, dict):
        return jsonify({"error": "output object required"}), 400
    for k in ("format", "layer_mode"):
        if k not in output:
            return jsonify({"error": f"output.{k} required"}), 400
    if output["format"] not in ("csv", "netcdf4"):
        return jsonify({"error": "output.format must be csv or netcdf4"}), 400
    if output["layer_mode"] not in ("separate", "merged"):
        return jsonify({"error": "output.layer_mode must be separate or merged"}), 400
    if _SESSION.get("config") is None:
        return jsonify({"error": "no active configuration"}), 400
    _SESSION["config"]["output"] = {
        "format": output["format"],
        "layer_mode": output["layer_mode"],
    }
    return _config_response()


@app.route("/api/config/save", methods=["POST"])
def api_config_save():
    data = request.get_json() or {}
    name = data.get("name")
    if not name:
        return jsonify({"error": "name required"}), 400
    cfg = _SESSION.get("config")
    if cfg is None:
        return jsonify({"error": "no active configuration"}), 400
    if not cfg.get("pollutants"):
        return jsonify({"error": "pollutants not set"}), 400
    if not cfg.get("domain"):
        return jsonify({"error": "domain not set"}), 400
    if not cfg.get("output"):
        return jsonify({"error": "output options not set"}), 400
    try:
        path = save_run_config(cfg, name)
        _SESSION["config_path"] = str(path)
        return jsonify({"ok": True, "path": str(path), "config": cfg})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/downscale/start", methods=["POST"])
def api_downscale_start():
    data = request.get_json() or {}
    config_path = data.get("config_path") or _SESSION.get("config_path")
    if not config_path:
        return jsonify({"error": "save a configuration file first"}), 400
    path = Path(config_path)
    if not path.is_file():
        return jsonify({"error": f"configuration not found: {path}"}), 404
    try:
        cfg = load_yaml(path)
        for key in ("domain", "pollutants", "output", "sectors", "paths"):
            if key not in cfg:
                return jsonify({"error": f"run config missing {key!r}"}), 400
        from UrbEm_Visualizer.downscaling.job import start_downscale_job

        job_id = start_downscale_job(path.resolve())
        return jsonify({"ok": True, "job_id": job_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/downscale/status/<job_id>", methods=["GET"])
def api_downscale_status(job_id: str):
    from UrbEm_Visualizer.downscaling.job import downscale_job_status

    st = downscale_job_status(job_id)
    if st is None:
        return jsonify({"error": "unknown job"}), 404
    return jsonify(st)


@app.route("/api/downscale/cancel/<job_id>", methods=["POST"])
def api_downscale_cancel(job_id: str):
    from UrbEm_Visualizer.downscaling.job import cancel_downscale_job

    if not cancel_downscale_job(job_id):
        return jsonify({"error": "unknown job"}), 404
    return jsonify({"ok": True})


@app.route("/api/dialog/pick-folder", methods=["POST"])
def api_dialog_pick_folder():
    data = request.get_json() or {}
    title = data.get("title") or "Select output folder"
    path = dialogs.pick_folder(title)
    if not path:
        return jsonify({"cancelled": True})
    return jsonify({"path": path})


@app.route("/api/viz/validate", methods=["POST"])
def api_viz_validate():
    data = request.get_json() or {}
    folder = data.get("output_dir")
    if not folder:
        return jsonify({"error": "output_dir required"}), 400
    from UrbEm_Visualizer.visualization.validate import validate_output_folder

    return jsonify(validate_output_folder(Path(folder)))


@app.route("/api/viz/open", methods=["POST"])
def api_viz_open():
    data = request.get_json() or {}
    folder = data.get("output_dir")
    if not folder:
        return jsonify({"error": "output_dir required"}), 400
    from UrbEm_Visualizer.visualization.validate import validate_output_folder
    from UrbEm_Visualizer.visualization.session import build_meta, open_output

    out = Path(folder)
    check = validate_output_folder(out)
    if not check["ok"]:
        return jsonify({"error": "validation failed", "errors": check["errors"]}), 400
    ctx = open_output(out)
    return jsonify({"ok": True, "meta": build_meta(ctx)})


@app.route("/api/viz/meta", methods=["GET"])
def api_viz_meta():
    from UrbEm_Visualizer.visualization.session import build_meta, get_context

    ctx = get_context()
    if ctx is None:
        return jsonify({"error": "no visualization session"}), 400
    return jsonify(build_meta(ctx))


@app.route("/api/viz/domain", methods=["GET"])
def api_viz_domain():
    from UrbEm_Visualizer.visualization.geojson_layers import domain_bbox_geojson
    from UrbEm_Visualizer.visualization.session import get_context

    ctx = get_context()
    if ctx is None:
        return jsonify({"error": "no visualization session"}), 400
    return jsonify(domain_bbox_geojson(ctx))


@app.route("/api/viz/cams-grid", methods=["GET"])
def api_viz_cams_grid():
    from UrbEm_Visualizer.dataset_loaders.cams_grid import load_domain_cams_geojson
    from UrbEm_Visualizer.visualization.session import get_context

    ctx = get_context()
    if ctx is None:
        return jsonify({"error": "no visualization session"}), 400
    cfg = ctx.config
    cams_rel = (cfg.get("paths") or {}).get("cams")
    if not cams_rel:
        return jsonify({"error": "manifest paths.cams missing"}), 400
    if not cfg.get("country") or not cfg.get("year"):
        return jsonify({"error": "manifest country/year missing"}), 400
    cams_path = Path(cams_rel)
    if not cams_path.is_absolute():
        cams_path = project_root() / cams_rel
    if not cams_path.is_file():
        return jsonify({"error": f"CAMS file not found: {cams_path}"}), 404
    try:
        gj = load_domain_cams_geojson(
            cams_path,
            str(cfg["country"]),
            int(cfg["year"]),
            list(cfg["pollutants"]),
            ctx.domain,
        )
        return jsonify(gj)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/viz/tiles/<sector_id>/<int:z>/<int:x>/<int:y>.png", methods=["GET"])
def api_viz_tiles(sector_id: str, z: int, x: int, y: int):
    from flask import Response

    from UrbEm_Visualizer.visualization.session import get_context
    from UrbEm_Visualizer.visualization.tiles import render_emission_tile

    ctx = get_context()
    if ctx is None:
        return jsonify({"error": "no visualization session"}), 400
    pollutant = request.args.get("pollutant")
    if not pollutant:
        return jsonify({"error": "pollutant required"}), 400
    sectors = request.args.get("sectors", "")
    active = [s.strip() for s in sectors.split(",") if s.strip() and s.strip() != "TOTAL"] or None
    scale = ctx.scale_for(sector_id, pollutant)
    if not scale:
        from UrbEm_Visualizer.visualization.tiles import _empty_png

        return Response(_empty_png(), mimetype="image/png")
    raster = ctx.area_raster(sector_id, pollutant, active)
    thr = request.args.get("threshold", type=float)
    if thr is None:
        thr = ctx.get_threshold(pollutant)
    png = render_emission_tile(
        raster,
        pollutant,
        float(scale["lower_bound"]),
        float(scale["upper_bound"]),
        z,
        x,
        y,
        threshold=thr,
    )
    return Response(png, mimetype="image/png")


@app.route("/api/viz/legend", methods=["GET"])
def api_viz_legend():
    from UrbEm_Visualizer.visualization.session import get_context

    ctx = get_context()
    if ctx is None:
        return jsonify({"error": "no visualization session"}), 400
    pollutant = request.args.get("pollutant")
    sector_id = request.args.get("sector", "TOTAL")
    if not pollutant:
        return jsonify({"error": "pollutant required"}), 400
    return jsonify(ctx.scale_for(sector_id, pollutant))


@app.route("/api/viz/points", methods=["GET"])
def api_viz_points():
    from UrbEm_Visualizer.visualization.geojson_layers import points_geojson
    from UrbEm_Visualizer.visualization.session import get_context

    ctx = get_context()
    if ctx is None:
        return jsonify({"error": "no visualization session"}), 400
    pollutant = request.args.get("pollutant")
    if not pollutant:
        return jsonify({"error": "pollutant required"}), 400
    sectors = request.args.get("sectors", "")
    active = [s.strip() for s in sectors.split(",") if s.strip() and s.strip() != "TOTAL"]
    if not active:
        active = ctx.sector_ids()
    return jsonify(points_geojson(ctx, active, pollutant))


@app.route("/api/viz/facility", methods=["GET"])
def api_viz_facility():
    from UrbEm_Visualizer.visualization.geojson_layers import facility_detail
    from UrbEm_Visualizer.visualization.session import get_context

    ctx = get_context()
    if ctx is None:
        return jsonify({"error": "no visualization session"}), 400
    pollutant = request.args.get("pollutant")
    lon = request.args.get("lon")
    lat = request.args.get("lat")
    sectors = request.args.get("sectors", "")
    if not pollutant or lon is None or lat is None:
        return jsonify({"error": "pollutant, lon, lat required"}), 400
    active = [s.strip() for s in sectors.split(",") if s.strip() and s.strip() != "TOTAL"]
    if not active:
        active = ctx.sector_ids()
    from UrbEm_Visualizer.visualization.analytics import facility_comparison

    return jsonify(
        facility_comparison(ctx, float(lon), float(lat), pollutant, active)
    )


@app.route("/api/viz/analytics", methods=["GET"])
def api_viz_analytics():
    from UrbEm_Visualizer.visualization.analytics import compute_analytics
    from UrbEm_Visualizer.visualization.session import get_context

    ctx = get_context()
    if ctx is None:
        return jsonify({"error": "no visualization session"}), 400
    return jsonify(compute_analytics(ctx))


@app.route("/api/viz/viewport", methods=["POST"])
def api_viz_viewport():
    from UrbEm_Visualizer.visualization.analytics import viewport_stats
    from UrbEm_Visualizer.visualization.session import get_context

    ctx = get_context()
    if ctx is None:
        return jsonify({"error": "no visualization session"}), 400
    data = request.get_json() or {}
    pollutant = data.get("pollutant")
    bbox = data.get("bbox") or {}
    for k in ("west", "south", "east", "north"):
        if k not in bbox:
            return jsonify({"error": f"bbox.{k} required"}), 400
    if not pollutant:
        return jsonify({"error": "pollutant required"}), 400
    return jsonify(
        viewport_stats(
            ctx,
            pollutant,
            float(bbox["west"]),
            float(bbox["south"]),
            float(bbox["east"]),
            float(bbox["north"]),
        )
    )


@app.route("/api/viz/threshold", methods=["POST"])
def api_viz_threshold():
    from UrbEm_Visualizer.visualization.session import get_context

    ctx = get_context()
    if ctx is None:
        return jsonify({"error": "no visualization session"}), 400
    data = request.get_json() or {}
    pollutant = data.get("pollutant")
    value = data.get("threshold")
    if not pollutant or value is None:
        return jsonify({"error": "pollutant and threshold required"}), 400
    ctx.user_threshold[str(pollutant)] = float(value)
    return jsonify({"ok": True, "pollutant": pollutant, "threshold": float(value)})


@app.route("/api/session", methods=["GET"])
def api_session():
    root = project_root()
    out = {
        "config": _SESSION.get("config"),
        "config_path": _SESSION.get("config_path"),
        "absent_sources": _session_absent(),
        "country": _SESSION.get("country"),
        "runs_dir": str(runs_dir().resolve()),
        "outputs_dir": str((root / "Output" / "UrbEm").resolve()),
        "pollutants_available": AVAILABLE_POLLUTANTS,
    }
    if _SESSION.get("config"):
        out["inputs_table"] = build_inputs_table(_SESSION["config"])
    return jsonify(out)
