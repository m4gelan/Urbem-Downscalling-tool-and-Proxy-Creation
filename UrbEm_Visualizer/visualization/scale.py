from __future__ import annotations

from typing import Any

import numpy as np

from UrbEm_Visualizer.visualization.emission_style import EPS, threshold_for


def _fmt_sci(v: float) -> str:
    if v == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(v))))
    mant = v / (10**exp)
    return f"{mant:.1f}\u00d710^{exp}"


def scale_from_values(values: np.ndarray, pollutant: str) -> dict[str, Any]:
    thr = threshold_for(pollutant)
    positive = values[np.isfinite(values) & (values > 0)]
    empty = {
        "pollutant": pollutant,
        "lower_bound": 0.0,
        "upper_bound": 1.0,
        "threshold": thr,
        "colormap": None,
        "legend_ticks": [
            {"label": _fmt_sci(thr), "value": thr, "norm": 0.0},
        ],
    }
    if positive.size == 0:
        return empty

    logv = np.log10(positive + EPS)
    lower = float(np.percentile(logv, 5))
    upper = float(np.percentile(logv, 95))
    if upper <= lower:
        upper = lower + 1.0

    p25, p50, p75, p98 = [float(np.percentile(positive, q)) for q in (25, 50, 75, 98)]

    def norm_raw(raw: float) -> float:
        lv = np.log10(max(raw, 0.0) + EPS)
        return float(np.clip((lv - lower) / (upper - lower), 0.0, 1.0))

    ticks = [
        {"label": _fmt_sci(p50), "value": p50, "norm": norm_raw(p50), "role": "mid"},
        {"label": _fmt_sci(p98), "value": p98, "norm": norm_raw(p98), "role": "max"},
    ]
    return {
        "pollutant": pollutant,
        "lower_bound": lower,
        "upper_bound": upper,
        "threshold": thr,
        "legend_ticks": ticks,
        "unit": "kg/yr/cell",
    }
