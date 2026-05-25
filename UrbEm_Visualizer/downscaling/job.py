from __future__ import annotations

import threading
import uuid
from pathlib import Path
from typing import Any

from UrbEm_Visualizer.downscaling.orchestrator import run_downscaling

_JOBS: dict[str, dict[str, Any]] = {}
_LOCK = threading.Lock()


def start_downscale_job(config_path: Path) -> str:
    job_id = uuid.uuid4().hex
    with _LOCK:
        _JOBS[job_id] = {
            "done": False,
            "cancel": False,
            "state": {
                "status": "running",
                "sectors": [],
                "error": None,
                "output_dir": None,
            },
        }

    def _cancel() -> bool:
        with _LOCK:
            return bool(_JOBS[job_id].get("cancel"))

    def _on_progress(state: dict) -> None:
        with _LOCK:
            _JOBS[job_id]["state"] = state

    def _worker() -> None:
        try:
            run_downscaling(config_path, on_progress=_on_progress, cancel_flag=_cancel)
        except Exception as exc:
            with _LOCK:
                st = _JOBS[job_id]["state"]
                st["status"] = "error"
                st["error"] = str(exc)
        finally:
            with _LOCK:
                _JOBS[job_id]["done"] = True

    threading.Thread(target=_worker, daemon=True).start()
    return job_id


def downscale_job_status(job_id: str) -> dict[str, Any] | None:
    with _LOCK:
        job = _JOBS.get(job_id)
        if not job:
            return None
        return {
            "done": job["done"],
            "state": dict(job["state"]),
        }


def cancel_downscale_job(job_id: str) -> bool:
    with _LOCK:
        job = _JOBS.get(job_id)
        if not job:
            return False
        job["cancel"] = True
        return True
