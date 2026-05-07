#!/usr/bin/env python3
"""
Describe dimensions, coordinates, data variables, and attributes of a CAMS NetCDF
file (e.g. CAMS-REG-ANT v8.1 from TNO FTP) or scan a folder of .nc files.

Usage:
  python describe_cams_dataset.py path/to/file.nc
  python describe_cams_dataset.py path/to/cams_folder --summary
  python describe_cams_dataset.py path/to/file.nc --json

Requires: xarray, netCDF4 (or h5netcdf for some files)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Hints aligned with urbem_interface/emissions/prepare_cams.py and
# code/UrbEm/.../prepare_cams_v8.1_TNOftp_emissions.R
KNOWN_CAMS_REG_ANT_V81: dict[str, str] = {
    "longitude": "Target grid axis: cell-centre longitudes (deg) for raster output.",
    "latitude": "Target grid axis: cell-centre latitudes (deg) for raster output.",
    "longitude_source": "Per-source longitude (deg WGS84); one entry per emission record.",
    "latitude_source": "Per-source latitude (deg WGS84); one entry per emission record.",
    "longitude_index": "Optional 1-based index into longitude axis for each source.",
    "latitude_index": "Optional 1-based index into latitude axis for each source.",
    "emission_category_index": (
        "GNFR sector index per source (typically 1..15: A,B,C,D,E,F1-F4,G,H,I,J,K,L)."
    ),
    "source_type_index": "Per source: 1 = area, 2 = point (CAMS-REG-ANT v8.1 convention).",
    "emis_cat_name": "Sector labels (often char array) matching emission categories.",
    "ch4": "CH4 emission per source record (same length as source index).",
    "co": "CO emission per source record.",
    "nh3": "NH3 emission per source record.",
    "nmvoc": "NMVOC emission per source record.",
    "nox": "NOx emission per source record.",
    "pm10": "PM10 emission per source record.",
    "pm2_5": "PM2.5 emission per source record.",
    "sox": "SOx (as SO2 proxy) emission per source record.",
}


def _fmt_attrs(attrs: dict) -> str:
    if not attrs:
        return ""
    parts = []
    for k, v in sorted(attrs.items()):
        s = str(v)
        if len(s) > 120:
            s = s[:117] + "..."
        parts.append(f"{k}={s!r}")
    return " | ".join(parts)


def _describe_array_stats(da) -> str:
    try:
        import numpy as np

        arr = da.values
        if arr.size == 0:
            return "empty"
        flat = np.asarray(arr).ravel()
        if np.issubdtype(arr.dtype, np.number):
            fnum = flat.astype(float, copy=False)
            finite = fnum[np.isfinite(fnum)]
            if finite.size == 0:
                return "all non-finite"
            return f"min={finite.min():.6g} max={finite.max():.6g} mean={finite.mean():.6g}"
        return f"sample={str(flat[0])[:40]}"
    except Exception as ex:  # noqa: BLE001
        return f"(stats unavailable: {ex})"


def describe_dataset(ds, path: Path, *, with_hints: bool = True) -> list[str]:
    lines: list[str] = []
    lines.append(f"File: {path.resolve()}")
    lines.append("")

    lines.append("=== Global attributes ===")
    if ds.attrs:
        for k, v in sorted(ds.attrs.items()):
            s = str(v)
            if len(s) > 200:
                s = s[:197] + "..."
            lines.append(f"  {k}: {s}")
    else:
        lines.append("  (none)")
    lines.append("")

    lines.append("=== Dimensions ===")
    for name, size in ds.sizes.items():
        lines.append(f"  {name}: {size}")
    lines.append("")

    lines.append("=== Coordinates ===")
    for name, coord in ds.coords.items():
        hint = f"  [hint] {KNOWN_CAMS_REG_ANT_V81[name]}" if with_hints and name in KNOWN_CAMS_REG_ANT_V81 else ""
        extra = _fmt_attrs(dict(coord.attrs))
        lines.append(f"  {name}: dims={coord.dims} shape={tuple(coord.shape)} dtype={coord.dtype}")
        if extra:
            lines.append(f"    attrs: {extra}")
        if hint:
            lines.append(hint)
        if coord.size > 0 and coord.size <= 10:
            lines.append(f"    values: {coord.values.ravel()}")
    lines.append("")

    lines.append("=== Data variables ===")
    for name, da in ds.data_vars.items():
        hint = f"  [hint] {KNOWN_CAMS_REG_ANT_V81[name]}" if with_hints and name in KNOWN_CAMS_REG_ANT_V81 else ""
        extra = _fmt_attrs(dict(da.attrs))
        lines.append(f"  {name}: dims={da.dims} shape={tuple(da.shape)} dtype={da.dtype}")
        if extra:
            lines.append(f"    attrs: {extra}")
        if hint:
            lines.append(hint)
        stats = _describe_array_stats(da)
        lines.append(f"    values: {stats}")
    lines.append("")

    lines.append("=== Encoding (first variable sample) ===")
    if ds.data_vars:
        first = next(iter(ds.data_vars.values()))
        enc = getattr(first, "encoding", None) or {}
        if enc:
            lines.append(f"  {json.dumps(enc, default=str)}")
        else:
            lines.append("  (no encoding on first data var)")
    return lines


def dataset_to_json_dict(ds, path: Path) -> dict:
    out: dict = {
        "path": str(path.resolve()),
        "global_attributes": {k: str(v) for k, v in ds.attrs.items()},
        "dimensions": {k: int(v) for k, v in ds.sizes.items()},
        "coordinates": {},
        "data_variables": {},
    }
    for name, coord in ds.coords.items():
        out["coordinates"][name] = {
            "dims": list(coord.dims),
            "shape": list(coord.shape),
            "dtype": str(coord.dtype),
            "attributes": {k: str(v) for k, v in coord.attrs.items()},
            "hint": KNOWN_CAMS_REG_ANT_V81.get(name),
        }
    for name, da in ds.data_vars.items():
        row = {
            "dims": list(da.dims),
            "shape": list(da.shape),
            "dtype": str(da.dtype),
            "attributes": {k: str(v) for k, v in da.attrs.items()},
            "hint": KNOWN_CAMS_REG_ANT_V81.get(name),
        }
        try:
            row["value_stats"] = _describe_array_stats(da)
        except Exception:  # noqa: BLE001
            pass
        out["data_variables"][name] = row
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Describe CAMS / NetCDF fields (CAMS-REG-ANT v8.1 aware).")
    p.add_argument(
        "path",
        type=Path,
        help="Path to a .nc file or a directory containing NetCDF files",
    )
    p.add_argument("--json", action="store_true", help="Print machine-readable JSON to stdout")
    p.add_argument(
        "--no-hints",
        action="store_true",
        help="Do not print CAMS-REG-ANT v8.1 interpretive hints",
    )
    p.add_argument(
        "--summary",
        action="store_true",
        help="If path is a directory, only list files and dimension names (faster)",
    )
    args = p.parse_args(argv)

    try:
        import xarray as xr
    except ImportError:
        print("Install xarray: pip install xarray netCDF4", file=sys.stderr)
        return 1

    path: Path = args.path
    if not path.exists():
        print(f"Not found: {path}", file=sys.stderr)
        return 1

    nc_files: list[Path]
    if path.is_file():
        nc_files = [path]
    else:
        nc_files = sorted(path.glob("*.nc"))
        if not nc_files:
            print(f"No .nc files under {path}", file=sys.stderr)
            return 1

    if path.is_dir() and args.summary:
        print(f"Found {len(nc_files)} NetCDF file(s) in {path.resolve()}\n")
        for f in nc_files:
            with xr.open_dataset(f) as ds:
                print(f"{f.name}: dims={dict(ds.sizes)}")
        return 0

    first = True
    for f in nc_files:
        if not first:
            print("\n" + "=" * 72 + "\n")
        first = False
        with xr.open_dataset(f) as ds:
            if args.json:
                print(json.dumps(dataset_to_json_dict(ds, f), indent=2))
            else:
                for line in describe_dataset(ds, f, with_hints=not args.no_hints):
                    print(line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
