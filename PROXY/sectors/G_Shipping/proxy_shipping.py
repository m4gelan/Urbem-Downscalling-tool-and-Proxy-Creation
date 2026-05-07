"""
GNFR G shipping: EMODnet vessel density, CORINE port areas, OSM shipping layers.

Single spatial proxy for all pollutants (no per-pollutant alpha). Weights sum to 1 per CAMS cell.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from PROXY.core.osm_corine_proxy import adapt_corine_classes_for_grid, osm_coverage_fraction
from PROXY.core.cams.grid import build_cam_cell_id
from PROXY.core.corine.raster import (
    _subwin_read_for_bounds,
    resolve_corine_tif,
    warp_corine_codes_nearest,
)
from PROXY.core.raster import normalize_within_cams_cells, validate_weight_sums
from PROXY.sectors.G_Shipping.emodnet_vessel_density import warp_vessel_density_to_ref

logger = logging.getLogger(__name__)

_CLC_FULL_EMODNET: frozenset[int] = frozenset({35, 36, 37, 38, 39, 40, 41, 42, 43, 44})
_CLC_PORT_CODE_44 = 5
_OUT_NODATA = -9999.0


def load_ref_from_fine_grid_tif(fine_grid_path: Path) -> dict[str, Any]:
    with rasterio.open(fine_grid_path) as src:
        h, w = int(src.height), int(src.width)
        tr = src.transform
        crs = src.crs
        if crs is None:
            raise ValueError(f"Fine grid has no CRS: {fine_grid_path}")
        left, bottom, right, top = rasterio.transform.array_bounds(h, w, tr)
        return {
            "height": h,
            "width": w,
            "transform": tr,
            "crs": crs.to_string(),
            "window_bounds_3035": (float(left), float(bottom), float(right), float(top)),
            "fine_grid_tif": fine_grid_path.resolve(),
        }


def ensure_ref_window_bounds(ref: dict[str, Any]) -> dict[str, Any]:
    ref = dict(ref)
    if "window_bounds_3035" not in ref:
        h, w = int(ref["height"]), int(ref["width"])
        left, bottom, right, top = rasterio.transform.array_bounds(h, w, ref["transform"])
        ref["window_bounds_3035"] = (float(left), float(bottom), float(right), float(top))
    return ref


def corine_port_fraction_and_codes(
    corine_tif: Path,
    ref: dict[str, Any],
    *,
    port_level2: int = 123,
) -> tuple[np.ndarray, np.ndarray]:
    clc_nn = warp_corine_codes_nearest(corine_tif, ref)
    classes_for_score, _remapped = adapt_corine_classes_for_grid(clc_nn, [int(port_level2)])
    if not classes_for_score:
        classes_for_score = [int(port_level2)]
    mx_valid = int(np.max(clc_nn[clc_nn != -9999])) if np.any(clc_nn != -9999) else 0
    if mx_valid <= 99 and _remapped:
        port_codes = classes_for_score
    else:
        port_codes = [int(port_level2)]
    h, w = int(ref["height"]), int(ref["width"])
    left, bottom, right, top = ref["window_bounds_3035"]
    dst_crs = rasterio.crs.CRS.from_string(ref["crs"])
    arr, src_tr, src_crs = _subwin_read_for_bounds(
        corine_tif, dst_crs, tuple(float(x) for x in (left, bottom, right, top))
    )
    arr_i = np.full(arr.shape, -9999, dtype=np.int32)
    m = np.isfinite(arr)
    arr_i[m] = np.rint(arr[m]).astype(np.int32, copy=False)
    src_bin = np.zeros_like(arr_i, dtype=np.float32)
    for c in port_codes:
        src_bin = np.maximum(src_bin, (arr_i == int(c)).astype(np.float32))
    port_out = np.zeros((h, w), dtype=np.float32)
    reproject(
        source=src_bin,
        destination=port_out,
        src_transform=src_tr,
        src_crs=src_crs,
        dst_transform=ref["transform"],
        dst_crs=dst_crs,
        src_nodata=0.0,
        dst_nodata=0.0,
        resampling=Resampling.average,
    )
    port_out = np.clip(port_out.astype(np.float32), 0.0, 1.0)
    return port_out, clc_nn


def minmax01(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    m = np.isfinite(a)
    if not np.any(m):
        return np.zeros_like(x, dtype=np.float32)
    lo = float(np.min(a[m]))
    hi = float(np.max(a[m]))
    if hi <= lo:
        return np.zeros(x.shape, dtype=np.float32)
    y = (a - lo) / (hi - lo)
    y = np.where(m, y, 0.0)
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def emodnet_normalized(
    dense: np.ndarray, clc_nn: np.ndarray, *, land_damp: float = 0.12
) -> np.ndarray:
    d = np.asarray(dense, dtype=np.float32)
    valid = np.isfinite(d) & (d >= 0)
    dmax = float(np.max(d[valid])) if np.any(valid) else 0.0
    if dmax <= 0:
        dn = np.zeros_like(d, dtype=np.float32)
    else:
        dn = np.divide(d, np.float32(dmax), out=np.zeros_like(d, dtype=np.float32), where=valid)
        dn = np.where(valid, dn, 0.0)
    ci = np.asarray(clc_nn, dtype=np.int32)
    unknown_clc = ci == -9999
    keep_full_emodnet = (
        unknown_clc | np.isin(ci, list(_CLC_FULL_EMODNET)) | (ci == _CLC_PORT_CODE_44)
    )
    damp = np.where(keep_full_emodnet, 1.0, float(land_damp)).astype(np.float32)
    return (dn * damp).astype(np.float32)


_INDUSTRIAL_SHIPPING_FAMILIES = frozenset({"industrial_shipyard", "landuse_industrial"})


def load_osm_shipping_union(osm_gpkg: Path) -> gpd.GeoDataFrame:
    frames: list[gpd.GeoDataFrame] = []
    for layer in ("osm_shipping_high", "osm_shipping_medium"):
        try:
            g = gpd.read_file(osm_gpkg, layer=layer)
            if g.empty:
                continue
            if layer == "osm_shipping_medium" and "shipping_family" in g.columns:
                fam = g["shipping_family"].astype(str)
                g = g.loc[~fam.isin(_INDUSTRIAL_SHIPPING_FAMILIES)].copy()
            if not g.empty:
                frames.append(g)
        except Exception as exc:
            logger.warning("Could not read layer %s: %s", layer, exc)
    if not frames:
        return gpd.GeoDataFrame(geometry=[], crs=None)
    out = pd.concat(frames, ignore_index=True)
    if out.crs is None:
        raise ValueError(f"OSM layers have no CRS: {osm_gpkg}")
    return gpd.GeoDataFrame(out, geometry=out.geometry, crs=out.crs)


def build_combined_proxy(
    ref: dict[str, Any],
    *,
    emodnet_path: Path,
    corine_path: Path,
    osm_gpkg: Path,
    osm_subdivide: int = 4,
    land_damp: float = 0.12,
    w_emodnet: float = 0.25,
    w_osm: float = 0.5,
    w_port: float = 0.25,
    port_level2: int = 123,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    corine_tif = resolve_corine_tif(corine_path)
    port_frac, clc_nn = corine_port_fraction_and_codes(
        corine_tif, ref, port_level2=int(port_level2)
    )
    dense = warp_vessel_density_to_ref(Path(emodnet_path), ref, resampling=Resampling.bilinear)
    dn = emodnet_normalized(dense, clc_nn, land_damp=land_damp)
    gdf = load_osm_shipping_union(osm_gpkg)
    osm_cov = osm_coverage_fraction(gdf, ref, subdivide_factor=osm_subdivide)
    z_o = minmax01(osm_cov)
    z_c = minmax01(port_frac)
    wsum = float(w_emodnet) + float(w_osm) + float(w_port)
    if wsum <= 0:
        wsum = 1.0
    a0, a1, a2 = w_emodnet / wsum, w_osm / wsum, w_port / wsum
    p = a0 * dn.astype(np.float32) + a1 * z_o + a2 * z_c
    p = np.where(np.isfinite(p), p, 0.0).astype(np.float32)
    diag = {
        "D_n_damped": dn,
        "emodnet_raw": dense.astype(np.float32),
        "osm_coverage": osm_cov.astype(np.float32),
        "clc_port_frac": port_frac.astype(np.float32),
        "z_osm": z_o,
        "z_clc": z_c,
        "clc_nn": clc_nn,
    }
    return p, diag


def run_shipping_areasource(
    *,
    cams_nc: Path,
    osm_gpkg: Path,
    emodnet_path: Path,
    output_folder: Path,
    corine_path: Path | None = None,
    ref: dict[str, Any] | None = None,
    fine_grid_tif: Path | None = None,
    osm_subdivide: int = 4,
    output_filename: str = "shipping_areasource.tif",
    write_diagnostics: bool = False,
    land_damp: float = 0.12,
    w_emodnet: float = 0.25,
    w_osm: float = 0.5,
    w_port: float = 0.25,
    port_level2: int = 123,
) -> dict[str, Any]:
    output_folder = Path(output_folder).expanduser().resolve()
    output_folder.mkdir(parents=True, exist_ok=True)
    out_tif = output_folder / str(output_filename)

    if ref is not None:
        ref_use = ensure_ref_window_bounds(ref)
    elif fine_grid_tif is not None:
        ref_use = load_ref_from_fine_grid_tif(Path(fine_grid_tif))
    else:
        raise ValueError("run_shipping_areasource: pass ref=... or fine_grid_tif=...")

    if corine_path is not None:
        cp = Path(corine_path)
    elif "corine_path" in ref_use:
        cp = Path(ref_use["corine_path"])
    else:
        raise ValueError(
            "corine_path is required when the reference grid has no corine_path "
            "(e.g. fine grid without parent CORINE)."
        )
    p, diag = build_combined_proxy(
        ref_use,
        emodnet_path=Path(emodnet_path),
        corine_path=cp,
        osm_gpkg=Path(osm_gpkg),
        osm_subdivide=osm_subdivide,
        land_damp=land_damp,
        w_emodnet=w_emodnet,
        w_osm=w_osm,
        w_port=w_port,
        port_level2=port_level2,
    )
    cam = build_cam_cell_id(Path(cams_nc), ref_use).astype(np.int64)
    work = np.asarray(p, dtype=np.float32, order="C").copy()
    work[cam < 0] = 0.0
    fb_summary: list[tuple[str, int]] = []
    wgt, _fb = normalize_within_cams_cells(
        work,
        cam.astype(np.int32),
        valid_mask=None,
        return_fallback_mask=True,
        context="G_Shipping stage=single_band",
        uniform_fallback_summary=fb_summary,
    )
    if fb_summary:
        tot = sum(c for _, c in fb_summary)
        logger.info(
            "G_Shipping: CAMS-cell uniform proxy fallback %d fine pixels (DEBUG per stage).",
            tot,
        )
    wgt = np.asarray(wgt, dtype=np.float32)
    wgt[cam < 0] = np.float32(_OUT_NODATA)
    errs = validate_weight_sums(wgt, cam, None, tol=1e-3)
    profile = {
        "driver": "GTiff",
        "width": ref_use["width"],
        "height": ref_use["height"],
        "count": 1,
        "dtype": "float32",
        "crs": ref_use["crs"],
        "transform": ref_use["transform"],
        "nodata": float(_OUT_NODATA),
        "compress": "deflate",
        "predictor": 3,
        "tiled": True,
        "BIGTIFF": "IF_SAFER",
    }
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(wgt, 1)
        dst.set_band_description(1, "g_shipping_weight")
        dst.update_tags(
            SOFTWARE="PDM PROXY G_Shipping",
            PROXY=f"{w_emodnet}*D_n + {w_osm}*z(OSM) + {w_port}*z(CLC port)",
            CAMS_GRID="PROXY J_Waste normalization (within-CAMS-cell sum=1)",
        )

    n_valid = int(np.count_nonzero(cam >= 0))
    out: dict[str, Any] = {
        "output_tif": str(out_tif),
        "ref": ref_use,
        "pixels_valid_cams": n_valid,
        "validate_errors": errs,
        "diagnostics": diag,
    }
    if write_diagnostics:
        write_diagnostic_rasters(output_folder, ref_use, diag)
    return out


def write_diagnostic_rasters(
    output_folder: Path,
    ref: dict[str, Any],
    diag: dict[str, np.ndarray],
    *,
    prefix: str = "G_Shipping_diag_",
) -> list[Path]:
    output_folder = Path(output_folder).resolve()
    output_folder.mkdir(parents=True, exist_ok=True)
    base_profile: dict[str, Any] = {
        "driver": "GTiff",
        "width": ref["width"],
        "height": ref["height"],
        "count": 1,
        "crs": ref["crs"],
        "transform": ref["transform"],
        "dtype": "float32",
        "nodata": np.nan,
        "compress": "deflate",
        "predictor": 3,
        "tiled": True,
    }
    out_paths: list[Path] = []
    keys = ("emodnet_raw", "D_n_damped", "osm_coverage", "clc_port_frac", "z_osm", "z_clc")
    for key in keys:
        if key not in diag:
            continue
        arr = np.asarray(diag[key], dtype=np.float32)
        pth = output_folder / f"{prefix}{key}.tif"
        with rasterio.open(pth, "w", **base_profile) as dst:
            dst.write(arr, 1)
        out_paths.append(pth)
    return out_paths
