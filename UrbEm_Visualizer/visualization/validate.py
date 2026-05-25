from __future__ import annotations

from pathlib import Path

import pandas as pd
import xarray as xr
import yaml

from UrbEm_Visualizer.downscaling.sector_meta import sector_mode
from UrbEm_Visualizer.pollutants import AVAILABLE_POLLUTANTS

_GRID_NAMES = ("area_emission_grid", "point_emission_grid")
_CSV_COLS = {"pollutant", "x", "y", "emission"}


def _pollutants_in_csv(path: Path) -> set[str]:
    if path.stat().st_size < 10:
        return set()
    try:
        df = pd.read_csv(path, nrows=50000)
    except pd.errors.EmptyDataError:
        return set()
    if not _CSV_COLS.issubset(df.columns):
        raise ValueError(f"missing columns {sorted(_CSV_COLS - set(df.columns))}")
    return set(df["pollutant"].dropna().astype(str).unique())


def _pollutants_in_nc(path: Path) -> set[str]:
    da = xr.open_dataarray(path)
    try:
        if "pollutant" not in da.dims:
            raise ValueError("missing pollutant dimension")
        return set(str(p) for p in da.coords["pollutant"].values)
    finally:
        da.close()


def validate_output_folder(output_dir: Path) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    output_dir = output_dir.resolve()

    if not output_dir.is_dir():
        return {"ok": False, "errors": [f"not a directory: {output_dir}"], "warnings": []}

    manifest_path = output_dir / "manifest.yaml"
    if not manifest_path.is_file():
        errors.append("manifest.yaml missing")
        return {"ok": False, "errors": errors, "warnings": warnings}

    with open(manifest_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    config = (raw or {}).get("config") if isinstance(raw, dict) else None
    if not isinstance(config, dict):
        errors.append("manifest.yaml: config block missing")
        return {"ok": False, "errors": errors, "warnings": warnings}

    domain = config.get("domain")
    if not isinstance(domain, dict):
        errors.append("manifest: domain missing")
    else:
        for k in ("crs", "xmin", "ymin", "xmax", "ymax"):
            if k not in domain:
                errors.append(f"manifest: domain.{k} missing")

    pollutants = config.get("pollutants")
    if not pollutants:
        errors.append("manifest: pollutants list empty")
    else:
        for p in pollutants:
            if p not in AVAILABLE_POLLUTANTS:
                warnings.append(f"pollutant {p!r} not in standard list")

    out_fmt = (config.get("output") or {}).get("format", "csv")
    if out_fmt not in ("csv", "netcdf4"):
        errors.append(f"unsupported output.format: {out_fmt!r}")

    sectors_cfg = config.get("sectors") or {}
    found_sectors: list[str] = []
    pollutants_found: set[str] = set()
    for sid in sectors_cfg:
        sub = output_dir / sid
        if not sub.is_dir():
            errors.append(f"sector folder missing: {sid}/")
            continue
        mode = sector_mode(sid)
        has_area = False
        has_point = False
        for stem in _GRID_NAMES:
            csv_p = sub / f"{stem}.csv"
            nc_p = sub / f"{stem}.nc"
            path = csv_p if csv_p.is_file() else (nc_p if nc_p.is_file() else None)
            if path is None:
                continue
            try:
                pols = _pollutants_in_csv(path) if path.suffix == ".csv" else _pollutants_in_nc(path)
            except Exception as exc:
                if path.suffix == ".csv" and path.stat().st_size < 10:
                    pols = set()
                else:
                    errors.append(f"{sid}/{path.name}: {exc}")
                    continue
            if stem == "area_emission_grid":
                has_area = True
            else:
                has_point = True
            pollutants_found |= pols
        if mode in ("both", "area_only") and not has_area:
            errors.append(f"{sid}: area_emission_grid missing (mode={mode})")
        if mode in ("both", "point_only") and not has_point:
            pt_path = sub / f"point_emission_grid.{'csv' if out_fmt == 'csv' else 'nc'}"
            if not pt_path.is_file():
                errors.append(f"{sid}: point_emission_grid missing (mode={mode})")
            elif out_fmt == "csv" and pt_path.stat().st_size == 0:
                warnings.append(f"{sid}: point_emission_grid.csv is empty")
        if has_area or has_point:
            found_sectors.append(sid)

    if pollutants:
        for p in pollutants:
            if p not in pollutants_found:
                errors.append(f"pollutant {p!r} not found in any sector grid")

    merged_csv = output_dir / "merged_emission_grid.csv"
    merged_nc = output_dir / "merged_emission_grid.nc"
    if (config.get("output") or {}).get("layer_mode") == "merged":
        mp = merged_csv if merged_csv.is_file() else (merged_nc if merged_nc.is_file() else None)
        if mp is None:
            warnings.append("layer_mode merged but merged_emission_grid file missing")

    if not found_sectors:
        errors.append("no sector emission grids found")

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "output_dir": str(output_dir),
        "config": config,
        "sectors": found_sectors,
    }
