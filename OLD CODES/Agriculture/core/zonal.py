"""Zonal CORINE histograms."""

from __future__ import annotations

from typing import Any

from rasterstats import zonal_stats

AG_CLC_CODES = tuple(range(12, 23))


def zonal_histograms(
    nuts2: Any,
    raster_path: Any,
    nodata: float,
) -> list[dict]:
    return zonal_stats(
        nuts2.geometry,
        str(raster_path),
        categorical=True,
        nodata=nodata,
    )


def apply_synthetic_grassland_leakage(
    score_synth: dict[int, float],
    hist: dict,
    leakage: float,
) -> None:
    if leakage <= 0 or leakage >= 1:
        return
    cropland = tuple(range(12, 18))
    total_crops = sum(float(score_synth.get(c, 0.0)) for c in cropland)
    if total_crops <= 0:
        return
    n18 = int(hist.get(18, 0) or 0)
    n21 = int(hist.get(21, 0) or 0)
    denom = n18 + n21
    for c in cropland:
        score_synth[c] = float(score_synth.get(c, 0.0)) * (1.0 - leakage)
    moved = leakage * total_crops
    if denom > 0:
        score_synth[18] = float(score_synth.get(18, 0.0)) + moved * (n18 / denom)
        score_synth[21] = float(score_synth.get(21, 0.0)) + moved * (n21 / denom)
    else:
        score_synth[18] = float(score_synth.get(18, 0.0)) + moved * 0.5
        score_synth[21] = float(score_synth.get(21, 0.0)) + moved * 0.5
