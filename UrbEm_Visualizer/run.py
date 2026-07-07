"""
Launch the UrbEm downscaling desktop UI (PyWebView + Flask), or run headless from CLI.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

_PKG = Path(__file__).resolve().parent
_ROOT = _PKG.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from UrbEm_Visualizer.paths import project_root, runs_dir

HOST = "127.0.0.1"
PORT = 5010


def _resolve_config_path(raw: str) -> Path:
    p = Path(raw)
    if p.is_file():
        return p.resolve()
    for base in (runs_dir(), project_root(), Path.cwd()):
        cand = (base / raw).resolve()
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"run config not found: {raw}")


def _run_headless(config_path: Path, sector: str | None) -> int:
    from UrbEm_Visualizer.downscaling.orchestrator import run_downscaling
    from UrbEm_Visualizer.writer.create_configuration import load_yaml

    cfg = load_yaml(config_path)
    sectors = [sector] if sector else None
    if sector:
        print(f"Sector: {sector}")
    print(f"Config: {config_path}")
    print(f"Country: {cfg['country']} | year: {cfg.get('emissions_year') or cfg['year']}")

    last_line = ""

    def on_progress(state: dict) -> None:
        nonlocal last_line
        running = next((s for s in state.get("sectors") or [] if s.get("status") == "running"), None)
        if not running:
            return
        line = f"{running['label']}: {running.get('step') or ''} ({running.get('progress', 0)}%)"
        if line != last_line:
            print(line)
            last_line = line

    state = run_downscaling(config_path, on_progress=on_progress, sectors=sectors)
    status = state.get("status")
    out_dir = state.get("output_dir")
    if status == "done":
        print(f"Done. Output: {out_dir}")
        return 0
    print(f"Failed ({status}): {state.get('error')}")
    return 1


def _run_ui() -> None:
    import webview
    from UrbEm_Visualizer.ui.backend import app

    def start_server():
        app.run(host=HOST, port=PORT, debug=False, use_reloader=False)

    threading.Thread(target=start_server, daemon=True).start()
    time.sleep(0.5)

    webview.create_window(
        "UrbEm Downscaling Tool",
        f"http://{HOST}:{PORT}",
        width=1200,
        height=820,
        resizable=True,
    )
    webview.start()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="UrbEm downscaling tool")
    p.add_argument("--no-ui", action="store_true", help="Run downscaling without the desktop UI")
    p.add_argument("--config", metavar="PATH", help="Run config YAML (required with --no-ui)")
    p.add_argument(
        "--sector",
        metavar="ID",
        help="Single sector only, e.g. C_OtherCombustion (optional; default: all sectors in config)",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if args.no_ui:
        if not args.config:
            print("error: --config is required with --no-ui", file=sys.stderr)
            return 2
        try:
            config_path = _resolve_config_path(args.config)
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        return _run_headless(config_path, args.sector)
    if args.config or args.sector:
        print("error: --config and --sector require --no-ui", file=sys.stderr)
        return 2
    _run_ui()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
