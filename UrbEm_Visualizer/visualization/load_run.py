from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from pyproj import Transformer

from UrbEm_Visualizer.downscaling.sector_meta import sector_mode
from UrbEm_Visualizer.visualization.emission_style import (
    colormap_for,
    default_threshold,
    pollutant_key,
    threshold_for,
)
from UrbEm_Visualizer.visualization.map_config import load_map_config
from UrbEm_Visualizer.visualization.raster_grid import (
    AreaRaster,
    GridTemplate,
    build_template,
    df_to_raster,
    sum_rasters,
)
from UrbEm_Visualizer.visualization.scale import scale_from_values


def _grid_path(sector_dir: Path, stem: str, fmt: str) -> Path | None:
    ext = "csv" if fmt == "csv" else "nc"
    p = sector_dir / f"{stem}.{ext}"
    return p if p.is_file() else None


def load_manifest(output_dir: Path) -> dict:
    with open(output_dir / "manifest.yaml", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    config = raw.get("config") if isinstance(raw, dict) else None
    if not isinstance(config, dict):
        raise ValueError("manifest.yaml: invalid config")
    return config


def domain_wgs84(domain: dict) -> tuple[float, float, float, float]:
    tr = Transformer.from_crs(str(domain["crs"]), "EPSG:4326", always_xy=True)
    xmin, ymin, xmax, ymax = (
        float(domain["xmin"]),
        float(domain["ymin"]),
        float(domain["xmax"]),
        float(domain["ymax"]),
    )
    xs = [xmin, xmax, xmin, xmax]
    ys = [ymin, ymin, ymax, ymax]
    lons, lats = tr.transform(xs, ys)
    return float(min(lons)), float(min(lats)), float(max(lons)), float(max(lats))


class RunContext:
    def __init__(self, output_dir: Path, config: dict):
        self.output_dir = output_dir.resolve()
        self.config = config
        self.domain = config["domain"]
        self.pollutants = list(config["pollutants"])
        self.fmt = (config.get("output") or {}).get("format", "csv")
        self.layer_mode = (config.get("output") or {}).get("layer_mode", "separate")
        self.unit = load_map_config().get("emission_unit", "kg/yr")
        self._to_wgs = Transformer.from_crs(str(self.domain["crs"]), "EPSG:4326", always_xy=True)
        self._cache: dict[tuple, pd.DataFrame] = {}
        self._raster_cache: dict[tuple, AreaRaster] = {}
        self._template: GridTemplate | None = None
        self.sector_scale: dict[str, dict] = {}
        self.per_sector_scale: dict[str, dict[str, dict]] = {}
        self.total_scale: dict[str, dict] = {}
        self.user_threshold: dict[str, float] = {}
        self._init_grid_and_scales()

    def get_threshold(self, pollutant: str) -> float:
        if pollutant in self.user_threshold:
            return float(self.user_threshold[pollutant])
        pk = pollutant_key(pollutant)
        if pk in self.user_threshold:
            return float(self.user_threshold[pk])
        return default_threshold(pollutant)

    def _init_grid_and_scales(self) -> None:
        frames = []
        for sid in self.sector_ids():
            if not self.sector_layers(sid)["area"]:
                continue
            for pol in self.pollutants:
                df = self.grid_df(sid, "area", pol)
                if not df.empty:
                    frames.append(df)
        self._template = build_template(self.domain, frames)

        for pol in self.pollutants:
            combined = []
            for sid in self.sector_ids():
                if not self.sector_layers(sid)["area"]:
                    continue
                df = self.grid_df(sid, "area", pol)
                if not df.empty:
                    combined.append(df["emission"].values)
                    bundle = scale_from_values(df["emission"].values, pol)
                    bundle["colormap"] = colormap_for(pol)
                    self.per_sector_scale.setdefault(sid, {})[pol] = bundle
            if combined:
                all_vals = np.concatenate(combined)
                bundle = scale_from_values(all_vals, pol)
                bundle["colormap"] = colormap_for(pol)
                self.sector_scale[pol] = bundle

            total_r = self.area_raster("TOTAL", pol, None)
            bundle_t = scale_from_values(total_r.data.ravel(), pol)
            bundle_t["colormap"] = colormap_for(pol)
            self.total_scale[pol] = bundle_t

    def scale_for(self, sector_id: str, pollutant: str) -> dict:
        if sector_id == "TOTAL":
            store = self.total_scale
        else:
            per = self.per_sector_scale.get(sector_id, {})
            if pollutant in per:
                return per[pollutant]
            pk = pollutant_key(pollutant)
            if pk in per:
                return per[pk]
            store = self.sector_scale
        if pollutant in store:
            return store[pollutant]
        return store.get(pollutant_key(pollutant), {})

    def area_raster(
        self,
        sector_id: str,
        pollutant: str,
        active_sectors: list[str] | None,
    ) -> AreaRaster:
        key = (sector_id, pollutant, tuple(active_sectors or ()))
        if key in self._raster_cache:
            return self._raster_cache[key]
        template = self._template
        if template is None:
            template = build_template(self.domain, [])

        if sector_id == "TOTAL":
            ids = active_sectors or self.sector_ids()
            rasters = [
                df_to_raster(self.grid_df(s, "area", pollutant), template)
                for s in ids
                if s != "TOTAL" and self.sector_layers(s)["area"]
            ]
            out = sum_rasters(rasters, template) if rasters else df_to_raster(
                pd.DataFrame(columns=["x", "y", "emission"]), template
            )
        else:
            out = df_to_raster(self.grid_df(sector_id, "area", pollutant), template)

        self._raster_cache[key] = out
        return out

    def sector_ids(self) -> list[str]:
        order = []
        for sid in (self.config.get("sectors") or {}):
            sub = self.output_dir / sid
            if sub.is_dir():
                order.append(sid)
        return order

    def _match_csv_paths(self, sector_id: str) -> list[Path]:
        sub = self.output_dir / sector_id
        names = (
            "point_matched_appointed",
            "point_matched_not_appointed",
            "point_unmatched",
        )
        out = []
        for name in names:
            p = sub / f"{name}.csv"
            if p.is_file() and p.stat().st_size > 10:
                out.append(p)
        return out

    def has_facility_points(self, sector_id: str) -> bool:
        return bool(self._match_csv_paths(sector_id))

    def sector_layers(self, sector_id: str) -> dict[str, bool]:
        if sector_id == "TOTAL":
            return {"area": True, "point": False}
        mode = sector_mode(sector_id)
        sub = self.output_dir / sector_id
        area = _grid_path(sub, "area_emission_grid", self.fmt) is not None
        has_match = self.has_facility_points(sector_id)
        has_grid = _grid_path(sub, "point_emission_grid", self.fmt) is not None
        if mode not in ("both", "point_only"):
            point = False
        elif self.layer_mode == "merged":
            point = has_match
        else:
            point = has_match or has_grid
        return {
            "area": area and mode in ("both", "area_only"),
            "point": point,
        }

    def use_facility_points(self, sector_id: str) -> bool:
        return self.has_facility_points(sector_id)

    def facility_points_df(self, sector_id: str, pollutant: str) -> pd.DataFrame:
        key = ("facility", sector_id, pollutant)
        if key in self._cache:
            return self._cache[key]
        emis_col = f"emis_{pollutant}"
        rows: list[dict] = []
        for path in self._match_csv_paths(sector_id):
            df = pd.read_csv(path)
            if emis_col not in df.columns:
                continue
            use_facility = "facility_lon" in df.columns and path.name == "point_matched_appointed.csv"
            for rec in df.itertuples(index=False):
                mass = float(getattr(rec, emis_col, 0) or 0)
                if mass <= 0:
                    continue
                if use_facility:
                    lon = float(rec.facility_lon)
                    lat = float(rec.facility_lat)
                else:
                    lon = float(rec.cams_lon)
                    lat = float(rec.cams_lat)
                rows.append({
                    "lon": lon,
                    "lat": lat,
                    "emission": mass,
                    "match_status": str(getattr(rec, "match_status", path.stem)),
                    "cams_point_id": int(getattr(rec, "cams_point_id", -1)),
                })
        out = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["lon", "lat", "emission"])
        if not out.empty:
            out = out.groupby(["lon", "lat"], as_index=False).agg({
                "emission": "sum",
                "match_status": "first",
                "cams_point_id": "first",
            })
        self._cache[key] = out
        return out

    def _read_csv_grid(self, path: Path, pollutant: str) -> pd.DataFrame:
        if path.stat().st_size < 10:
            return pd.DataFrame(columns=["x", "y", "emission"])
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame(columns=["x", "y", "emission"])
        need = {"pollutant", "x", "y", "emission"}
        if not need.issubset(df.columns):
            raise ValueError(f"{path.name}: expected columns {sorted(need)}")
        out = df[df["pollutant"].astype(str) == pollutant].copy()
        out["x"] = pd.to_numeric(out["x"], errors="coerce")
        out["y"] = pd.to_numeric(out["y"], errors="coerce")
        out["emission"] = pd.to_numeric(out["emission"], errors="coerce").fillna(0)
        return out[out["emission"] > 0]

    def _read_nc_grid(self, path: Path, pollutant: str) -> pd.DataFrame:
        da = xr.open_dataarray(path)
        try:
            if pollutant not in [str(p) for p in da.coords["pollutant"].values]:
                raise ValueError(f"pollutant {pollutant!r} not in {path.name}")
            plane = da.sel(pollutant=pollutant).values.astype(np.float64)
            h, w = plane.shape
            xmin = float(self.domain["xmin"])
            ymin = float(self.domain["ymin"])
            xmax = float(self.domain["xmax"])
            ymax = float(self.domain["ymax"])
            dx = (xmax - xmin) / w
            dy = (ymax - ymin) / h
            rows = []
            nz = np.argwhere(plane > 0)
            for r, c in nz:
                x = xmin + (c + 0.5) * dx
                y = ymax - (r + 0.5) * dy
                rows.append({"x": x, "y": y, "emission": float(plane[r, c])})
            return pd.DataFrame(rows)
        finally:
            da.close()

    def grid_df(self, sector_id: str, layer: str, pollutant: str) -> pd.DataFrame:
        key = (sector_id, layer, pollutant)
        if key in self._cache:
            return self._cache[key]
        stem = "area_emission_grid" if layer == "area" else "point_emission_grid"
        path = _grid_path(self.output_dir / sector_id, stem, self.fmt)
        if path is None:
            empty = pd.DataFrame(columns=["x", "y", "emission"])
            self._cache[key] = empty
            return empty
        df = self._read_csv_grid(path, pollutant) if path.suffix == ".csv" else self._read_nc_grid(path, pollutant)
        self._cache[key] = df
        return df

    def sum_sectors_df(self, sector_ids: list[str], layer: str, pollutant: str) -> pd.DataFrame:
        parts = []
        for sid in sector_ids:
            df = self.grid_df(sid, layer, pollutant)
            if df.empty:
                continue
            d = df[["x", "y", "emission"]].copy()
            d["sector"] = sid
            parts.append(d)
        if not parts:
            return pd.DataFrame(columns=["x", "y", "emission"])
        comb = pd.concat(parts, ignore_index=True)
        return comb.groupby(["x", "y"], as_index=False)["emission"].sum()
