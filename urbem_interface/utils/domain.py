"""
Consolidated domain utilities - parse_domain and domain_bounds_wgs84.
"""

from __future__ import annotations

from typing import Any

from rasterio.crs import CRS
from rasterio.transform import from_bounds


def domain_bounds_wgs84(domain_cfg: dict) -> tuple[float, float, float, float]:
    """
    Project domain bounds to WGS84 for filtering.
    Returns (xmin, ymin, xmax, ymax) as (lon_min, lat_min, lon_max, lat_max).
    All four corners are projected to handle non-rectangular projections correctly.
    """
    from pyproj import Transformer

    xmin, ymin, xmax, ymax = (
        domain_cfg["xmin"],
        domain_cfg["ymin"],
        domain_cfg["xmax"],
        domain_cfg["ymax"],
    )
    domain_crs = CRS.from_string(domain_cfg["crs"])
    to_wgs = Transformer.from_crs(domain_crs, CRS.from_epsg(4326), always_xy=True)

    xs = [xmin, xmin, xmax, xmax]
    ys = [ymin, ymax, ymin, ymax]
    lons, lats = to_wgs.transform(xs, ys)

    return (float(min(lons)), float(min(lats)), float(max(lons)), float(max(lats)))


def parse_domain(run_config: dict) -> dict[str, Any]:
    """
    Parse domain from run config into a structured dict with transform, shape, crs, bounds.
    """
    domain_cfg = run_config["domain"]
    nrow = int(domain_cfg["nrow"])
    ncol = int(domain_cfg["ncol"])
    xmin = float(domain_cfg["xmin"])
    ymin = float(domain_cfg["ymin"])
    xmax = float(domain_cfg["xmax"])
    ymax = float(domain_cfg["ymax"])
    crs_str = domain_cfg["crs"]

    domain_crs = CRS.from_string(crs_str)
    domain_transform = from_bounds(xmin, ymin, xmax, ymax, ncol, nrow)
    domain_shape = (nrow, ncol)
    domain_bounds = (xmin, ymin, xmax, ymax)

    return {
        "cfg": domain_cfg,
        "crs": domain_crs,
        "crs_str": crs_str,
        "transform": domain_transform,
        "shape": domain_shape,
        "bounds": domain_bounds,
        "nrow": nrow,
        "ncol": ncol,
        "xmin": xmin,
        "ymin": ymin,
        "xmax": xmax,
        "ymax": ymax,
    }
