"""
Flask app to browse a folder of GeoTIFFs and preview downsampled grayscale PNGs.

  python -m urbem_interface.proxies.factory.viz_app --root path/to/created_proxies --port 8765
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory, Response

from urbem_interface.proxies.factory.raster_preview import (
    raster_meta as _raster_meta,
    raster_preview_png_bytes,
    resolve_under_root,
)

logging.getLogger("werkzeug").setLevel(logging.WARNING)

_pkg = Path(__file__).resolve().parent
_root = _pkg.parents[3]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

app = Flask(__name__, static_folder=str(_pkg / "static"), static_url_path="")


def _get_allowed_root() -> Path:
    r = getattr(app, "proxy_viz_root", None)
    if r is None:
        return (_root / "Output" / "proxy" / "default").resolve()
    return Path(r).resolve()


def _resolve_under_root(rel: str) -> Path:
    return resolve_under_root(_get_allowed_root(), rel)


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/root", methods=["GET"])
def api_root():
    return jsonify({"root": str(_get_allowed_root())})


@app.route("/api/list", methods=["GET"])
def api_list():
    sub = request.args.get("path", "") or ""
    try:
        folder = _resolve_under_root(sub)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    if not folder.is_dir():
        return jsonify({"error": "not a directory"}), 404
    entries = []
    for p in sorted(folder.iterdir(), key=lambda x: x.name.lower()):
        if p.name.startswith("."):
            continue
        entries.append(
            {
                "name": p.name,
                "is_dir": p.is_dir(),
                "rel": str(p.relative_to(_get_allowed_root())).replace("\\", "/"),
            }
        )
    return jsonify({"path": str(folder), "entries": entries})


@app.route("/api/meta", methods=["GET"])
def api_meta():
    rel = request.args.get("path", "")
    try:
        path = _resolve_under_root(rel)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    if not path.is_file():
        return jsonify({"error": "not a file"}), 404
    try:
        return jsonify(_raster_meta(path))
    except ImportError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/preview.png", methods=["GET"])
def api_preview():
    rel = request.args.get("path", "")
    max_side = int(request.args.get("max", 1024))
    band = int(request.args.get("band", 1))
    try:
        path = _resolve_under_root(rel)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    if not path.is_file():
        return jsonify({"error": "not a file"}), 404
    try:
        data = raster_preview_png_bytes(path, max_side=max_side, band=band)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except ImportError as e:
        return jsonify({"error": str(e)}), 500
    return Response(data, mimetype="image/png")


@app.route("/api/pipeline", methods=["GET"])
def api_pipeline():
    """
    Declarative proxy_pipeline.json (inputs, proxy_definitions summary, downscaling).
    Start viz_app with --pipeline-json path to enable.
    """
    p = getattr(app, "proxy_pipeline_json_path", None)
    if not p:
        return jsonify(
            {
                "error": "not configured",
                "hint": "python -m urbem_interface.proxies.factory.viz_app --root ... --pipeline-json path/to/proxy_pipeline.json",
            }
        ), 404
    path = Path(str(p)).resolve()
    if not path.is_file():
        return jsonify({"error": f"not found: {path}"}), 404
    try:
        from urbem_interface.pipeline.job_config import load_raw_pipeline
    except ImportError as e:
        return jsonify({"error": str(e)}), 500
    pl = load_raw_pipeline(path)
    defs = pl.get("proxy_definitions") or {}
    slim_defs: dict = {}
    for k, v in defs.items():
        if not isinstance(v, dict):
            continue
        slim_defs[k] = {
            "output_file": v.get("output_file"),
            "proxy_config": v.get("proxy_config"),
            "snap_config": v.get("snap_config"),
        }
    down = pl.get("downscaling") or {}
    return jsonify(
        {
            "pipeline_path": str(path),
            "inputs": pl.get("inputs"),
            "factory_reference": pl.get("factory_reference"),
            "proxy_definitions": slim_defs,
            "gnfr_to_proxy": down.get("gnfr_to_proxy"),
            "snap_proxy_map": down.get("snap_proxy_map"),
            "auxiliary_proxies": pl.get("auxiliary_proxies"),
        }
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Proxy GeoTIFF preview server")
    ap.add_argument("--root", type=Path, required=True, help="Allowed folder to browse (e.g. created_proxies)")
    ap.add_argument(
        "--pipeline-json",
        type=Path,
        default=None,
        help="Optional proxy_pipeline.json for GET /api/pipeline (same folder as urbem_interface/config/).",
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args(argv)
    app.proxy_viz_root = str(Path(args.root).resolve())
    app.proxy_pipeline_json_path = (
        str(Path(args.pipeline_json).resolve()) if args.pipeline_json else None
    )
    logging.getLogger(__name__).info("Serving %s on http://%s:%s", app.proxy_viz_root, args.host, args.port)
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
