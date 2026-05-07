"""Named EPSG:3035 (ETRS89-LAEA) boxes for clipping vector work (CORINE GDB / E-PRTR)."""

from __future__ import annotations

# left, bottom, right, top style as xmin, ymin, xmax, ymax in projected metres (EPSG:3035)
VECTOR_SUBSET_BBOX_3035: dict[str, dict[str, float]] = {
    "france": {
        "xmin": 3_480_000.0,
        "ymin": 2_150_000.0,
        "xmax": 4_150_000.0,
        "ymax": 3_180_000.0,
    },
    "germany": {
        "xmin": 4_020_000.0,
        "ymin": 2_620_000.0,
        "xmax": 4_720_000.0,
        "ymax": 3_520_000.0,
    },
    "iberia": {
        "xmin": 2_500_000.0,
        "ymin": 1_550_000.0,
        "xmax": 3_650_000.0,
        "ymax": 2_650_000.0,
    },
    "italy": {
        "xmin": 3_850_000.0,
        "ymin": 1_520_000.0,
        "xmax": 4_750_000.0,
        "ymax": 2_450_000.0,
    },
    "poland": {
        "xmin": 4_350_000.0,
        "ymin": 3_050_000.0,
        "xmax": 5_100_000.0,
        "ymax": 3_650_000.0,
    },
    "greece": {
        "xmin": 5_160_000.0,
        "ymin": 1_410_000.0,
        "xmax": 6_060_000.0,
        "ymax": 2_210_000.0,
    },
}


def intersect_projected_bboxes(
    grid: dict[str, float], subset: dict[str, float]
) -> dict[str, float]:
    """Intersect two axis-aligned boxes in the same projected CRS."""
    out = {
        "xmin": max(float(grid["xmin"]), float(subset["xmin"])),
        "ymin": max(float(grid["ymin"]), float(subset["ymin"])),
        "xmax": min(float(grid["xmax"]), float(subset["xmax"])),
        "ymax": min(float(grid["ymax"]), float(subset["ymax"])),
    }
    if out["xmin"] >= out["xmax"] or out["ymin"] >= out["ymax"]:
        raise ValueError(
            "Vector subset does not intersect the template grid "
            f"(grid={grid}, subset={subset})"
        )
    return out
