#!/usr/bin/env python3
"""Regression checks for PROXY sector output rasters."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from PROXY.core.cams.grid import build_cam_cell_id
from PROXY.core.raster.normalize import validate_weight_sums


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass
class RasterSummary:
    path: str
    bands: int
    height: int
    width: int
    crs: str | None
    transform: tuple[float, ...]
    nodata: float | None
    finite_count: int
    positive_count: int
    min_value: float
    max_value: float
    total_mass: float
    sha256: str


def _sha256_of_array(arr: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def _read_multiband(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32, copy=False)
        meta = {
            "count": src.count,
            "height": src.height,
            "width": src.width,
            "crs": None if src.crs is None else src.crs.to_string(),
            "transform": tuple(float(v) for v in src.transform[:6]),
            "nodata": src.nodata,
        }
    return arr, meta


def summarize_raster(path: Path) -> RasterSummary:
    arr, meta = _read_multiband(path)
    nodata = meta["nodata"]
    data = arr
    if nodata is not None and np.isfinite(nodata):
        data = np.where(data == np.float32(nodata), np.nan, data)
    finite = np.isfinite(data)
    pos = finite & (data > 0)
    return RasterSummary(
        path=str(path.resolve()),
        bands=int(meta["count"]),
        height=int(meta["height"]),
        width=int(meta["width"]),
        crs=meta["crs"],
        transform=meta["transform"],
        nodata=None if nodata is None else float(nodata),
        finite_count=int(np.count_nonzero(finite)),
        positive_count=int(np.count_nonzero(pos)),
        min_value=float(np.nanmin(data)) if np.any(finite) else float("nan"),
        max_value=float(np.nanmax(data)) if np.any(finite) else float("nan"),
        total_mass=float(np.nansum(data)),
        sha256=_sha256_of_array(np.nan_to_num(data, nan=-9999.0).astype(np.float32, copy=False)),
    )


def _assert_same_grid(a: dict[str, Any], b: dict[str, Any], *, atol: float = 1e-6) -> None:
    same = (
        int(a["height"]) == int(b["height"])
        and int(a["width"]) == int(b["width"])
        and a["crs"] == b["crs"]
        and np.allclose(np.array(a["transform"]), np.array(b["transform"]), atol=atol, rtol=0.0)
    )
    if not same:
        raise ValueError("Grid mismatch between baseline and candidate.")


def compare_rasters(
    baseline: Path,
    candidate: Path,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    base_arr, base_meta = _read_multiband(baseline)
    cand_arr, cand_meta = _read_multiband(candidate)
    _assert_same_grid(base_meta, cand_meta)
    if int(base_meta["count"]) != int(cand_meta["count"]):
        raise ValueError("Band-count mismatch between baseline and candidate.")

    base_nd = base_meta["nodata"]
    cand_nd = cand_meta["nodata"]
    if base_nd is not None and np.isfinite(base_nd):
        base_arr = np.where(base_arr == np.float32(base_nd), np.nan, base_arr)
    if cand_nd is not None and np.isfinite(cand_nd):
        cand_arr = np.where(cand_arr == np.float32(cand_nd), np.nan, cand_arr)

    diff = cand_arr - base_arr
    abs_diff = np.abs(diff)
    finite = np.isfinite(base_arr) & np.isfinite(cand_arr)
    if not np.any(finite):
        raise ValueError("No overlapping finite pixels to compare.")
    max_abs = float(np.nanmax(abs_diff[finite]))
    mae = float(np.nanmean(abs_diff[finite]))
    rmse = float(np.sqrt(np.nanmean((diff[finite]) ** 2)))

    close = np.isclose(cand_arr, base_arr, atol=atol, rtol=rtol, equal_nan=True)
    mismatch_count = int(np.count_nonzero(~close))

    return {
        "baseline": str(baseline.resolve()),
        "candidate": str(candidate.resolve()),
        "atol": float(atol),
        "rtol": float(rtol),
        "max_abs_diff": max_abs,
        "mae": mae,
        "rmse": rmse,
        "mismatch_count": mismatch_count,
        "total_pixels": int(close.size),
        "pass": mismatch_count == 0,
    }


def per_cell_sum_check(
    raster_path: Path,
    cams_nc: Path,
    *,
    tol: float,
) -> dict[str, Any]:
    with rasterio.open(raster_path) as src:
        ref = {
            "height": src.height,
            "width": src.width,
            "transform": src.transform,
            "crs": src.crs.to_string() if src.crs else "EPSG:3035",
        }
        cam_cell_id = build_cam_cell_id(cams_nc, ref)
        checks: list[dict[str, Any]] = []
        for b in range(1, src.count + 1):
            arr = src.read(b).astype(np.float32)
            nd = src.nodata
            if nd is not None and np.isfinite(nd):
                arr = np.where(arr == np.float32(nd), np.nan, arr)
            errs = validate_weight_sums(arr, cam_cell_id, None, tol=tol)
            checks.append(
                {
                    "band": b,
                    "error_count": len(errs),
                    "sample_errors": errs[:5],
                    "pass": len(errs) == 0,
                }
            )
    all_pass = all(c["pass"] for c in checks)
    return {
        "raster": str(raster_path.resolve()),
        "cams_nc": str(cams_nc.resolve()),
        "tol": float(tol),
        "bands": checks,
        "pass": all_pass,
    }


def _print_json(data: dict[str, Any]) -> None:
    print(json.dumps(data, indent=2, sort_keys=False))


def _cmd_summary(args: argparse.Namespace) -> int:
    p = Path(args.raster)
    if not p.is_absolute():
        p = _root() / p
    if not p.is_file():
        print(f"Missing raster: {p}", file=sys.stderr)
        return 1
    _print_json(asdict(summarize_raster(p)))
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    b = Path(args.baseline)
    c = Path(args.candidate)
    root = _root()
    if not b.is_absolute():
        b = root / b
    if not c.is_absolute():
        c = root / c
    if not b.is_file() or not c.is_file():
        print(f"Missing input raster(s): baseline={b} candidate={c}", file=sys.stderr)
        return 1
    out = compare_rasters(b, c, atol=float(args.atol), rtol=float(args.rtol))
    _print_json(out)
    return 0 if out["pass"] else 2


def _cmd_per_cell(args: argparse.Namespace) -> int:
    r = Path(args.raster)
    nc = Path(args.cams_nc)
    root = _root()
    if not r.is_absolute():
        r = root / r
    if not nc.is_absolute():
        nc = root / nc
    if not r.is_file() or not nc.is_file():
        print(f"Missing input(s): raster={r} cams_nc={nc}", file=sys.stderr)
        return 1
    out = per_cell_sum_check(r, nc, tol=float(args.tol))
    _print_json(out)
    return 0 if out["pass"] else 2


def build_parser() -> argparse.ArgumentParser:
    root = _root()
    ap = argparse.ArgumentParser(description="Regression checks for PROXY outputs.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_sum = sub.add_parser("summary", help="Print summary statistics for one raster.")
    p_sum.add_argument("--raster", required=True, help="Path to GeoTIFF.")
    p_sum.set_defaults(func=_cmd_summary)

    p_cmp = sub.add_parser("compare", help="Compare candidate raster against baseline.")
    p_cmp.add_argument("--baseline", required=True, help="Baseline GeoTIFF path.")
    p_cmp.add_argument("--candidate", required=True, help="Candidate GeoTIFF path.")
    p_cmp.add_argument("--atol", type=float, default=1e-6)
    p_cmp.add_argument("--rtol", type=float, default=1e-6)
    p_cmp.set_defaults(func=_cmd_compare)

    p_cell = sub.add_parser("per-cell", help="Validate per-CAMS-cell normalization sums.")
    p_cell.add_argument("--raster", required=True, help="GeoTIFF path.")
    p_cell.add_argument(
        "--cams-nc",
        default=str(root / "INPUT" / "Emissions" / "CAMS_REG_ANT_EU_2019.nc"),
        help="CAMS NetCDF path.",
    )
    p_cell.add_argument("--tol", type=float, default=1e-3)
    p_cell.set_defaults(func=_cmd_per_cell)
    return ap


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
