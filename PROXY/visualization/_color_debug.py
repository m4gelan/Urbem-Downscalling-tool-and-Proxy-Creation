"""Opt-in tracing for categorical overlay vs legend colour alignment.

Enable with environment variable::

    PROXY_VIZ_COLOR_DEBUG=1

Logs JSON lines to stderr prefixed with ``[PROXY_VIZ_COLOR_DEBUG]`` so you can
grep a run of ``python -m PROXY.main visualize ...`` and compare:

  * hex inputs and Matplotlib ``to_rgb`` fractions
  * uint8 RGBA bytes written into numpy overlays
  * legend swatch CSS (opaque vs dark-basemap simulation)
  * key order used for argmax / dominance (order affects colour assignment)
"""
from __future__ import annotations

import json
import os
from typing import Any


def viz_color_debug_enabled() -> bool:
    return os.environ.get("PROXY_VIZ_COLOR_DEBUG", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def viz_color_log(event: str, **fields: Any) -> None:
    if not viz_color_debug_enabled():
        return
    payload: dict[str, Any] = {"event": event, **fields}
    line = json.dumps(payload, default=str, sort_keys=True)
    print(f"[PROXY_VIZ_COLOR_DEBUG] {line}", flush=True)
