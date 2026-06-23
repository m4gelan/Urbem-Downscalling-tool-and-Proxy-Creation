from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
import yaml


def _write_netcdf(path: Path, da: xr.DataArray, sector: str, kind: str) -> None:
    out = da.astype(np.float32)
    out.name = kind
    out.attrs["sector"] = sector
    out.to_netcdf(path, engine="netcdf4")


def _write_csv_grid(path: Path, da: xr.DataArray, grid_transform) -> None:
    rows = []
    pols = list(da.coords["pollutant"].values)
    data = da.values.astype(np.float32)
    for pi, pol in enumerate(pols):
        plane = data[pi]
        nz = np.argwhere(plane > 0)
        for r, c in nz:
            x, y = rasterio.transform.xy(grid_transform, int(r), int(c), offset="center")
            rows.append({"pollutant": pol, "x": x, "y": y, "emission": float(plane[r, c])})
    pd.DataFrame(rows).to_csv(path, index=False)


def export_run(
    output_dir: Path,
    *,
    config: dict,
    fmt: str,
    procedure: str,
    sector_results: dict[str, dict[str, Any]],
    merged: xr.DataArray | None,
    weight_check_log: dict[str, Any],
    clip_log: list[dict],
    grid_transform,
    crs: str,
) -> None:
    _ = crs
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "manifest.yaml", "w", encoding="utf-8") as f:
        yaml.dump(
            {"config": config, "crs": crs, "domain": config.get("domain")},
            f,
            default_flow_style=False,
            sort_keys=False,
        )

    with open(output_dir / "weight_check_log.json", "w", encoding="utf-8") as f:
        json.dump(weight_check_log, f, indent=2)

    with open(output_dir / "clip_log.json", "w", encoding="utf-8") as f:
        json.dump(clip_log, f, indent=2)

    for sid, res in sector_results.items():
        sub = output_dir / sid
        sub.mkdir(exist_ok=True)
        for name, key in (
            ("point_matched_appointed", "point_appointed"),
            ("point_matched_not_appointed", "point_not_appointed"),
            ("point_unmatched", "point_unmatched"),
        ):
            df = res.get(key)
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.to_csv(sub / f"{name}.csv", index=False)
        if procedure == "merged":
            sm = res.get("sector_merged")
            if sm is not None:
                if fmt == "netcdf4":
                    _write_netcdf(sub / "merged_emission_grid.nc", sm, sid, "merged")
                else:
                    _write_csv_grid(sub / "merged_emission_grid.csv", sm, grid_transform)
        elif fmt == "netcdf4":
            if res.get("area_emission") is not None:
                _write_netcdf(sub / "area_emission_grid.nc", res["area_emission"], sid, "area")
            if res.get("point_emission") is not None:
                _write_netcdf(sub / "point_emission_grid.nc", res["point_emission"], sid, "point")
        else:
            if res.get("area_emission") is not None:
                _write_csv_grid(sub / "area_emission_grid.csv", res["area_emission"], grid_transform)
            if res.get("point_emission") is not None:
                _write_csv_grid(sub / "point_emission_grid.csv", res["point_emission"], grid_transform)

    if procedure == "merged" and merged is not None:
        if fmt == "netcdf4":
            _write_netcdf(output_dir / "merged_emission_grid.nc", merged, "all", "merged")
        else:
            _write_csv_grid(output_dir / "merged_emission_grid.csv", merged, grid_transform)
