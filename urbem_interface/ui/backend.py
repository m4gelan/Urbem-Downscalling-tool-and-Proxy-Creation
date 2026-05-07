"""
Flask backend for UrbEm Interface - config, validation, pipeline run.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory

logging.getLogger("werkzeug").setLevel(logging.WARNING)

app = Flask(__name__, static_folder="static", static_url_path="")

# Ensure project root is on path
_root = Path(__file__).resolve().parent.parent.parent
import sys
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# Global state for pipeline progress
_pipeline_status = {
    "stage": None,
    "output_path": None,
    "output_folder": None,
    "source_type": None,
    "error": None,
    "completed": False,
}
_pipeline_lock = threading.Lock()

_proxies_build_status: dict = {
    "running": False,
    "completed": False,
    "error": None,
    "message": "",
    "exit_code": None,
    "publish_log": [],
}
_proxies_build_lock = threading.Lock()


def _get_config_dir() -> Path:
    """Config dir is relative to project root."""
    return _root / "urbem_interface" / "config"


_RUN_CONFIG_JSON_SKIP = frozenset(
    {
        "proxies.json",
        "proxy_pipeline.json",
        "loaded_config.json",
        "snap_mapping.json",
        "pointsources.json",
        "linesources.json",
    }
)


@app.route("/api/config/run-files", methods=["GET"])
def config_run_files():
    """JSON files in config/ that look like emission run configs (have a domain block)."""
    cfgd = _get_config_dir()
    out: list[dict[str, str]] = []
    for p in sorted(cfgd.glob("*.json"), key=lambda x: x.name.lower()):
        if p.name in _RUN_CONFIG_JSON_SKIP or p.name.startswith("."):
            continue
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict) or not isinstance(data.get("domain"), dict):
            continue
        out.append({"name": p.name, "path": str(p.resolve())})
    return jsonify({"files": out, "config_dir": str(cfgd.resolve())})


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(app.static_folder, path)


@app.route("/api/config/load", methods=["POST"])
def load_config():
    """Load run config from file path or from file content."""
    data = request.get_json() or {}
    path_str = data.get("path")
    content = data.get("content")

    if content is not None:
        try:
            cfg = json.loads(content) if isinstance(content, str) else content
            config_dir = _get_config_dir()
            save_path = config_dir / "loaded_config.json"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
            return jsonify({
                "config": cfg,
                "config_dir": str(config_dir),
                "config_path": str(save_path),
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    if not path_str:
        return jsonify({"error": "path or content required"}), 400
    path = Path(path_str)
    if not path.exists():
        return jsonify({"error": f"File not found: {path}"}), 404
    try:
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
        return jsonify({
            "config": cfg,
            "config_dir": str(path.parent),
            "config_path": str(path),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/config/save", methods=["POST"])
def save_config():
    """Save run config to file."""
    data = request.get_json() or {}
    config = data.get("config")
    path_str = data.get("path")
    if not config or not path_str:
        return jsonify({"error": "config and path required"}), 400
    path = Path(path_str)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        return jsonify({"ok": True, "path": str(path)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/config/default", methods=["GET"])
def default_config():
    """Return default config path and sample config."""
    config_dir = _get_config_dir()
    default_path = config_dir / "ioannina_2019.json"
    if default_path.exists():
        with open(default_path, encoding="utf-8") as f:
            cfg = json.load(f)
        return jsonify({
            "config": cfg,
            "config_dir": str(config_dir),
            "config_path": str(default_path),
        })
    return jsonify({
        "config": {
            "region": "Ioannina",
            "year": 2019,
            "source_type": "area",
            "domain": {
                "nrow": 30,
                "ncol": 30,
                "xmin": 468812,
                "ymin": 4375636,
                "xmax": 498812,
                "ymax": 4405636,
                "crs": "EPSG:32634",
            },
            "paths": {
                "input_root": "Input",
                "output_root": "Output",
                "proxy_country": "default",
                "emission_region": "Ioannina",
                "cams_folder": "given_CAMS/CAMS-REG-ANT_v8.1_TNO_ftp/netcdf",
            },
        },
        "config_dir": str(config_dir),
    })


@app.route("/api/proxies/validate", methods=["POST"])
def validate_proxies():
    """Validate proxies folder against config."""
    data = request.get_json() or {}
    config_dir = Path(data.get("config_dir", _get_config_dir()))
    run_config = data.get("run_config")
    if not run_config:
        return jsonify({"error": "run_config required"}), 400

    from urbem_interface.utils import resolve_paths, validate_proxies_folder, load_proxies_config

    paths = resolve_paths(run_config, config_dir)
    proxies_folder = paths["proxies_folder"]
    proxies_cfg_path = config_dir / "proxies.json"
    if not proxies_cfg_path.exists():
        return jsonify({"error": "proxies.json not found"}), 404
    proxies_config = load_proxies_config(
        proxies_cfg_path, proxies_folder=proxies_folder
    )
    result = validate_proxies_folder(proxies_config, proxies_folder)
    return jsonify(result)


def _resolved_proxies_folder(run_config: dict, config_dir: Path) -> Path:
    from urbem_interface.utils import resolve_paths

    paths = resolve_paths(run_config, Path(config_dir))
    return Path(paths["proxies_folder"])


@app.route("/api/proxies/tifs", methods=["POST"])
def proxies_list_tifs():
    """List GeoTIFFs in the run config's proxies folder (for step-1 visualization)."""
    data = request.get_json() or {}
    config_dir = Path(data.get("config_dir", _get_config_dir()))
    run_config = data.get("run_config")
    if not run_config:
        return jsonify({"error": "run_config required"}), 400
    try:
        folder = _resolved_proxies_folder(run_config, config_dir)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    if not folder.is_dir():
        return jsonify({"folder": str(folder), "tifs": []})
    tifs = []
    for p in sorted(folder.glob("*.tif"), key=lambda x: x.name.lower()):
        if p.name.startswith("."):
            continue
        tifs.append({"name": p.name, "rel": p.name})
    return jsonify({"folder": str(folder), "tifs": tifs})


@app.route("/api/proxies/preview.png", methods=["POST"])
def proxies_preview_png():
    """Downsampled grayscale PNG for one proxy raster under the resolved proxies folder."""
    data = request.get_json() or {}
    config_dir = Path(data.get("config_dir", _get_config_dir()))
    run_config = data.get("run_config")
    filename = data.get("filename")
    if not run_config or not filename:
        return jsonify({"error": "run_config and filename required"}), 400
    fn = str(filename).replace("\\", "/").split("/")[-1]
    if not fn or fn.startswith(".") or ".." in fn:
        return jsonify({"error": "invalid filename"}), 400
    try:
        from urbem_interface.proxies.factory.raster_preview import (
            raster_preview_png_bytes,
            resolve_under_root,
        )

        folder = _resolved_proxies_folder(run_config, config_dir)
        path = resolve_under_root(folder, fn)
        if not path.is_file():
            return jsonify({"error": "file not found"}), 404
        body = raster_preview_png_bytes(
            path,
            max_side=int(data.get("max_side", 1024)),
            band=int(data.get("band", 1)),
        )
        return Response(body, mimetype="image/png")
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/proxies/vector-subsets", methods=["GET"])
def proxies_vector_subsets():
    from urbem_interface.proxies.factory.regions import VECTOR_SUBSET_BBOX_3035

    return jsonify({"subsets": sorted(VECTOR_SUBSET_BBOX_3035.keys())})


@app.route("/api/proxies/validate-factory", methods=["POST"])
def proxies_validate_factory():
    """Check proxy_factory.json inputs exist before starting a long build."""
    data = request.get_json() or {}
    fc = data.get("factory_config")
    cfg_path = Path(fc).resolve() if fc else (_get_config_dir() / "proxy_factory.json").resolve()
    if not cfg_path.is_file():
        return jsonify(
            {"ok": False, "error": f"Factory config not found: {cfg_path}", "missing": []}
        ), 404

    from urbem_interface.proxies.factory.validate_factory import validate_proxy_factory_config

    result = validate_proxy_factory_config(cfg_path)
    return jsonify(result)


@app.route("/api/proxies/build", methods=["POST"])
def proxies_build():
    """Run CORINE/E-PRTR proxy factory in a background thread."""
    global _proxies_build_status
    data = request.get_json() or {}
    fc = data.get("factory_config")
    cfg_path = Path(fc).resolve() if fc else (_get_config_dir() / "proxy_factory.json").resolve()
    if not cfg_path.is_file():
        return jsonify({"error": f"Factory config not found: {cfg_path}"}), 404
    raw_vs = data.get("vector_subset")
    vector_subset = (
        None
        if raw_vs is None or str(raw_vs).strip() == ""
        else str(raw_vs).strip().lower()
    )

    with _proxies_build_lock:
        if _proxies_build_status["running"]:
            return jsonify({"error": "A proxy build is already running."}), 409
        _proxies_build_status["running"] = True
        _proxies_build_status["completed"] = False
        _proxies_build_status["error"] = None
        _proxies_build_status["message"] = "Building proxies..."
        _proxies_build_status["exit_code"] = None
        _proxies_build_status["publish_log"] = []

    cfg_str = str(cfg_path)

    def run_build():
        global _proxies_build_status
        from urbem_interface.proxies.factory import build_proxies as bp
        from urbem_interface.proxies.factory.build_proxies import main as build_main

        def on_published(internal_name: str, final_name: str) -> None:
            stem = Path(internal_name).stem
            line = stem + " -> " + final_name
            with _proxies_build_lock:
                _proxies_build_status.setdefault("publish_log", []).append(line)

        argv = ["--config", cfg_str]
        if vector_subset:
            argv.extend(["--vector-subset", vector_subset])
        bp.set_publish_progress_callback(on_published)
        try:
            code = build_main(argv)
            with _proxies_build_lock:
                _proxies_build_status["exit_code"] = code
                _proxies_build_status["completed"] = True
                _proxies_build_status["running"] = False
                if code != 0:
                    _proxies_build_status["error"] = f"build_proxies exited with code {code}"
                else:
                    _proxies_build_status["message"] = "Proxy build finished OK."
        except Exception as e:
            with _proxies_build_lock:
                _proxies_build_status["error"] = str(e)
                _proxies_build_status["completed"] = True
                _proxies_build_status["running"] = False
        finally:
            bp.set_publish_progress_callback(None)

    threading.Thread(target=run_build, daemon=True).start()
    return jsonify({"ok": True, "message": "Proxy build started"})


@app.route("/api/proxies/build-status", methods=["GET"])
def proxies_build_status():
    with _proxies_build_lock:
        d = dict(_proxies_build_status)
    d.setdefault("publish_log", [])
    return jsonify(d)


@app.route("/api/run", methods=["POST"])
def run_pipeline():
    """Start pipeline run in background."""
    global _pipeline_status
    data = request.get_json() or {}
    config_path = data.get("config_path")
    if not config_path:
        return jsonify({"error": "config_path required"}), 400

    path = Path(config_path)
    if not path.exists():
        return jsonify({"error": f"Config not found: {path}"}), 404

    run_config = None
    try:
        from urbem_interface.utils.config_loader import load_run_config
        run_config = load_run_config(path)
    except Exception:
        pass

    with _pipeline_lock:
        _pipeline_status["stage"] = None
        _pipeline_status["output_path"] = None
        _pipeline_status["source_type"] = run_config.get("source_type", "area") if run_config else "area"
        _pipeline_status["error"] = None
        _pipeline_status["completed"] = False

    def run():
        global _pipeline_status
        try:
            from urbem_interface.routines import run_pipeline as do_run

            def progress(stage_id: str, stage_data: dict):
                with _pipeline_lock:
                    _pipeline_status["stage"] = stage_id
                    out_p = stage_data.get("output_path")
                    _pipeline_status["output_path"] = out_p
                    if out_p:
                        _pipeline_status["output_folder"] = str(Path(out_p).parent)
                    _pipeline_status["completed"] = stage_id == "export"

            do_run(path, path.parent, progress_callback=progress)
        except Exception as e:
            with _pipeline_lock:
                _pipeline_status["error"] = str(e)
                _pipeline_status["completed"] = True

    thread = threading.Thread(target=run)
    thread.start()
    return jsonify({"ok": True, "message": "Pipeline started"})


@app.route("/api/status", methods=["GET"])
def pipeline_status():
    """Get current pipeline status."""
    with _pipeline_lock:
        return jsonify(dict(_pipeline_status))


@app.route("/api/domain/geojson", methods=["POST"])
def domain_geojson():
    """Get domain as GeoJSON for map."""
    data = request.get_json() or {}
    domain_cfg = data.get("domain")
    if not domain_cfg:
        return jsonify({"error": "domain required"}), 400
    from urbem_interface.visualization import domain_to_geojson
    return jsonify(domain_to_geojson(domain_cfg))


@app.route("/api/intermediates/list", methods=["POST"])
def intermediates_list():
    """List available intermediate layers for visualization."""
    data = request.get_json() or {}
    output_folder = data.get("output_folder")
    if not output_folder:
        return jsonify({"error": "output_folder required"}), 400

    base = Path(output_folder)
    intermediates = base / "intermediates"
    if not intermediates.exists():
        return jsonify({
            "layers": ["domain", "results"],
            "cams_grid": False,
            "proxies": [],
            "cams_sectors": [],
            "snap_ids": [],
            "pollutants": ["NOx", "NMVOC", "CO", "SO2", "NH3", "PM2.5", "PM10"],
        })

    proxies_dir = intermediates / "step2_proxies"
    proxies = []
    if proxies_dir.exists():
        proxies = [p.stem for p in proxies_dir.glob("*.csv")]

    cams_dir = intermediates / "step1_cams_warped"
    cams_sectors = []
    if cams_dir.exists():
        cams_sectors = [p.stem.replace("-", "_") for p in cams_dir.glob("*.csv")]

    snap_dir = intermediates / "step4_snap"
    snap_ids = []
    if snap_dir.exists():
        for p in snap_dir.glob("snap*.csv"):
            try:
                snap_ids.append(int(p.stem.replace("snap", "")))
            except ValueError:
                pass
        snap_ids.sort()

    cams_grid = (intermediates / "step2_coarse_grid" / "cams_origin_metadata.csv").exists()
    line_cams = (intermediates / "line_cams" / "cams_stacked.csv").exists()
    line_downscaled = (intermediates / "line_downscaled" / "cams_stacked.csv").exists()

    layers = ["domain"]
    if cams_grid:
        layers.append("cams_grid")
    if proxies:
        layers.extend(["proxies", "cams_emissions", "downscaled"])
    elif cams_sectors:
        layers.extend(["cams_emissions", "downscaled"])
    if line_cams:
        layers.append("line_cams")
    if line_downscaled:
        layers.append("line_downscaled")
    layers.append("results")

    return jsonify({
        "layers": layers,
        "cams_grid": cams_grid,
        "proxies": proxies,
        "cams_sectors": cams_sectors,
        "snap_ids": snap_ids,
        "line_cams": line_cams,
        "line_downscaled": line_downscaled,
        "pollutants": ["NOx", "NMVOC", "CO", "SO2", "NH3", "PM2.5", "PM10"],
    })


@app.route("/api/intermediates/geojson", methods=["POST"])
def intermediates_geojson():
    """Get intermediate layer as GeoJSON for map display."""
    data = request.get_json() or {}
    output_folder = data.get("output_folder")
    layer_type = data.get("layer_type")
    domain_cfg = data.get("domain")
    if not output_folder or not layer_type or not domain_cfg:
        return jsonify({"error": "output_folder, layer_type, domain required"}), 400

    base = Path(output_folder)
    intermediates = base / "intermediates"

    try:
        from urbem_interface.visualization import (
            domain_to_geojson,
            output_to_geojson,
            raster_csv_to_geojson,
            raster_sum_to_geojson,
            cams_grid_to_geojson,
        )
    except ImportError:
        return jsonify({"error": "visualization module not available"}), 500

    if layer_type == "domain":
        return jsonify(domain_to_geojson(domain_cfg))

    if layer_type == "results":
        output_path = data.get("output_path")
        source_type = data.get("source_type", "area")
        if not output_path:
            return jsonify({"error": "output_path required for results layer"}), 400
        pollutant = data.get("pollutant")
        mode = data.get("mode", "total")
        snap_id = data.get("snap_id")
        return jsonify(output_to_geojson(
            output_path, source_type, domain_cfg,
            pollutant=pollutant, mode=mode, snap_id=snap_id,
        ))

    if layer_type == "cams_grid":
        geojson = cams_grid_to_geojson(intermediates, domain_cfg)
        return jsonify(geojson)

    if layer_type == "line_cams":
        pollutant = data.get("pollutant", "NOx")
        csv_path = intermediates / "line_cams" / "cams_stacked.csv"
        geojson = raster_csv_to_geojson(csv_path, domain_cfg, value_column=pollutant)
        return jsonify(geojson)

    if layer_type == "line_downscaled":
        pollutant = data.get("pollutant", "NOx")
        csv_path = intermediates / "line_downscaled" / "cams_stacked.csv"
        geojson = raster_csv_to_geojson(csv_path, domain_cfg, value_column=pollutant)
        return jsonify(geojson)

    if layer_type == "proxies":
        proxy_name = data.get("proxy_name")
        if not proxy_name:
            return jsonify({"error": "proxy_name required"}), 400
        csv_path = intermediates / "step2_proxies" / f"{proxy_name}.csv"
        geojson = raster_csv_to_geojson(csv_path, domain_cfg, value_column="weight")
        return jsonify(geojson)

    if layer_type == "cams_emissions":
        mode = data.get("mode", "sector")
        pollutant = data.get("pollutant", "NOx")
        cams_dir = intermediates / "step1_cams_warped"
        if mode == "total":
            paths = list(cams_dir.glob("*.csv")) if cams_dir.exists() else []
            geojson = raster_sum_to_geojson(paths, domain_cfg, value_column=pollutant)
        else:
            sector = data.get("sector")
            if not sector:
                return jsonify({"error": "sector required for by-sector view"}), 400
            sector_file = sector.replace("_", "-")
            csv_path = cams_dir / f"{sector_file}.csv"
            geojson = raster_csv_to_geojson(csv_path, domain_cfg, value_column=pollutant)
        return jsonify(geojson)

    if layer_type == "downscaled":
        mode = data.get("mode", "sector")
        pollutant = data.get("pollutant", "NOx")
        if mode == "total":
            snap_dir = intermediates / "step4_snap"
            paths = list(snap_dir.glob("snap*.csv")) if snap_dir.exists() else []
            geojson = raster_sum_to_geojson(paths, domain_cfg, value_column=pollutant)
        elif mode == "sector":
            sector = data.get("sector")
            if not sector:
                return jsonify({"error": "sector required"}), 400
            sector_file = sector.replace("_", "-")
            csv_path = intermediates / "step3_after_proxy" / f"{sector_file}.csv"
            geojson = raster_csv_to_geojson(csv_path, domain_cfg, value_column=pollutant)
        else:
            snap_id = data.get("snap_id")
            if snap_id is None:
                return jsonify({"error": "snap_id required"}), 400
            csv_path = intermediates / "step4_snap" / f"snap{snap_id}.csv"
            geojson = raster_csv_to_geojson(csv_path, domain_cfg, value_column=pollutant)
        return jsonify(geojson)

    return jsonify({"error": f"Unknown layer_type: {layer_type}"}), 400


@app.route("/api/output/snaps-and-pollutants", methods=["POST"])
def output_snaps_and_pollutants():
    """Return available pollutants and SNAPs per pollutant from output CSV (area sources)."""
    data = request.get_json() or {}
    output_path = data.get("output_path")
    if not output_path:
        return jsonify({"error": "output_path required"}), 400
    try:
        from urbem_interface.visualization import output_snaps_and_pollutants as get_snaps
        return jsonify(get_snaps(output_path))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/output/geojson", methods=["POST"])
def output_geojson():
    """Convert output CSV to GeoJSON for map display."""
    data = request.get_json() or {}
    output_path = data.get("output_path")
    source_type = data.get("source_type", "area")
    domain_cfg = data.get("domain")
    if not output_path:
        return jsonify({"error": "output_path required"}), 400
    if not domain_cfg:
        return jsonify({"error": "domain required"}), 400
    try:
        from urbem_interface.visualization import output_to_geojson
        geojson = output_to_geojson(output_path, source_type, domain_cfg)
        return jsonify(geojson)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/output/statistics", methods=["POST"])
def output_statistics():
    """Compute statistics for output CSV (KPIs, CAMS comparison, charts data)."""
    data = request.get_json() or {}
    output_path = data.get("output_path")
    output_folder = data.get("output_folder")
    source_type = data.get("source_type", "area")
    config_dir = data.get("config_dir")
    if not output_path:
        return jsonify({"error": "output_path required"}), 400
    output_folder = output_folder or str(Path(output_path).parent)
    try:
        from urbem_interface.visualization.statistics import compute_output_statistics
        result = compute_output_statistics(output_path, output_folder, source_type, config_dir=config_dir)
        if "error" in result:
            return jsonify(result), 400
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
