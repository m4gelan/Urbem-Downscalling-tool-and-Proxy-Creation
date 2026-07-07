from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from pyproj import Transformer

from UrbEm_Visualizer.downscaling.output_config import parse_point_matching, procedure_label
from UrbEm_Visualizer.downscaling.sector_meta import sector_mode
from UrbEm_Visualizer.paths import project_root
from UrbEm_Visualizer.visualization.scale import _fmt_sci
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


def _coord_key(lon: float, lat: float) -> tuple[float, float]:
    return round(lon, 5), round(lat, 5)


def load_manifest(output_dir: Path) -> dict:
    with open(output_dir / "manifest.yaml", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    config = raw.get("config") if isinstance(raw, dict) else None
    if not isinstance(config, dict):
        raise ValueError("manifest.yaml: invalid config")
    return config


def domain_corners_wgs84(domain: dict) -> list[list[float]]:
    """Closed ring [lon, lat] for domain rectangle in its CRS (true footprint on map)."""
    tr = Transformer.from_crs(str(domain["crs"]), "EPSG:4326", always_xy=True)
    xmin, ymin, xmax, ymax = (
        float(domain["xmin"]),
        float(domain["ymin"]),
        float(domain["xmax"]),
        float(domain["ymax"]),
    )
    xs = [xmin, xmax, xmax, xmin, xmin]
    ys = [ymin, ymin, ymax, ymax, ymin]
    lons, lats = tr.transform(xs, ys)
    return [[float(lons[i]), float(lats[i])] for i in range(len(xs))]


def domain_wgs84(domain: dict) -> tuple[float, float, float, float]:
    ring = domain_corners_wgs84(domain)
    lons = [p[0] for p in ring]
    lats = [p[1] for p in ring]
    return float(min(lons)), float(min(lats)), float(max(lons)), float(max(lats))


def _in_wgs84_bbox(lon: float, lat: float, west: float, south: float, east: float, north: float) -> bool:
    return west <= lon <= east and south <= lat <= north


class RunContext:
    def __init__(self, output_dir: Path, config: dict):
        self.output_dir = output_dir.resolve()
        self.config = config
        self.domain = config["domain"]
        self.pollutants = list(config["pollutants"])
        self.fmt = (config.get("output") or {}).get("format", "csv")
        self.point_matching = parse_point_matching(config.get("output") or {})
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
        self._template = build_template(self.domain, frames, config=self.config, output_dir=self.output_dir)

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
            template = build_template(self.domain, [], config=self.config, output_dir=self.output_dir)

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
        has_match = self.has_facility_points(sector_id)
        if self.point_matching["procedure"] == "merged":
            has_merged = _grid_path(sub, "merged_emission_grid", self.fmt) is not None
            return {
                "area": has_merged and mode in ("both", "area_only"),
                "point": has_match and mode in ("both", "point_only"),
            }
        area = _grid_path(sub, "area_emission_grid", self.fmt) is not None
        has_grid = _grid_path(sub, "point_emission_grid", self.fmt) is not None
        if mode not in ("both", "point_only"):
            point = False
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
                    "point_shape": (
                        "sphere" if path.name == "point_matched_appointed.csv"
                        else "diamond" if path.name == "point_matched_not_appointed.csv"
                        else "box"
                    ),
                })
        out = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["lon", "lat", "emission"])
        if not out.empty:
            out = out.groupby(["lon", "lat"], as_index=False).agg({
                "emission": "sum",
                "match_status": "first",
                "cams_point_id": "first",
                "point_shape": "first",
            })
        self._cache[key] = out
        return out

    def _sector_link_path(self, sector_id: str) -> Path | None:
        ps = (self.config.get("sectors") or {}).get(sector_id, {}).get("point_source") or {}
        rel = ps.get("path")
        if not rel:
            return None
        p = Path(str(rel))
        if not p.is_absolute():
            p = project_root() / p
        return p if p.is_file() else None

    def appointed_facility_at(self, sector_id: str, lon: float, lat: float) -> dict | None:
        key = _coord_key(lon, lat)
        cache_key = ("appointed", sector_id, key)
        if cache_key in self._cache:
            return self._cache[cache_key]

        path = self.output_dir / sector_id / "point_matched_appointed.csv"
        if not path.is_file() or path.stat().st_size < 10:
            self._cache[cache_key] = None
            return None

        df = pd.read_csv(path)
        lon_k = round(lon, 5)
        lat_k = round(lat, 5)
        match = df[
            (df["facility_lon"].astype(float).round(5) == lon_k)
            & (df["facility_lat"].astype(float).round(5) == lat_k)
        ]
        if match.empty:
            self._cache[cache_key] = None
            return None

        rec = match.iloc[0]

        def _cell(name: str):
            if name not in rec.index:
                return None
            v = rec[name]
            return None if pd.isna(v) else v

        pollutants = []
        for pol in self.pollutants:
            col = f"emis_{pol}"
            if col not in df.columns:
                continue
            val = float(rec[col] or 0)
            pollutants.append({
                "pollutant": pol,
                "emission": val,
                "emission_label": _fmt_sci(val) if val > 0 else "—",
            })
        details = []
        for col in df.columns:
            if col.startswith("meta_") and pd.notna(rec.get(col)):
                label = col[5:].replace("_", " ").strip().title()
                details.append({"label": label, "value": str(rec[col])})

        out = {
            "sector_id": sector_id,
            "match_status": str(_cell("match_status") or "matched_appointed"),
            "mass_outcome": self.mass_outcome_line(str(_cell("match_status") or "matched_appointed")),
            "dataset": _cell("match_dataset"),
            "facility_name": _cell("facility_name"),
            "facility_id": _cell("facility_id"),
            "cams_point_id": int(_cell("cams_point_id") or -1),
            "cams_lon": float(_cell("cams_lon")) if _cell("cams_lon") is not None else None,
            "cams_lat": float(_cell("cams_lat")) if _cell("cams_lat") is not None else None,
            "pollutants": pollutants,
            "details": details,
        }
        if not out["dataset"]:
            link = self._sector_link_path(sector_id)
            if link:
                from UrbEm_Visualizer.downscaling.point_meta import (
                    appointed_meta,
                    facility_links,
                    flatten_meta_to_row,
                    load_match_sidecar,
                    meta_for_cams,
                    sidecar_pollutant_mass,
                )

                raw = meta_for_cams(load_match_sidecar(link), out["cams_point_id"])
                link_hit = None
                for lk in facility_links(raw):
                    if lk.get("facility_lon") is None or lk.get("facility_lat") is None:
                        continue
                    if (
                        round(float(lk["facility_lon"]), 5) == lon_k
                        and round(float(lk["facility_lat"]), 5) == lat_k
                    ):
                        link_hit = lk
                        break
                meta = appointed_meta(raw, link_hit)
                flat = flatten_meta_to_row(meta)
                out["dataset"] = meta.get("dataset") or flat.get("match_dataset")
                out["facility_name"] = out["facility_name"] or meta.get("facility_name") or flat.get("facility_name")
                out["facility_id"] = out["facility_id"] or meta.get("facility_id") or flat.get("facility_id")
                if meta.get("match_distance_km") is not None:
                    out["match_distance_km"] = float(meta["match_distance_km"])
                if meta.get("n_facility_links"):
                    details.append({
                        "label": "Facility links",
                        "value": str(meta["n_facility_links"]),
                    })
                for d in meta.get("details") or []:
                    if isinstance(d, dict) and d.get("label"):
                        details.append({
                            "label": str(d["label"]),
                            "value": str(d.get("value", "")),
                        })
                if link_hit and isinstance(link_hit.get("attributed_pollutants"), dict):
                    for row in out["pollutants"]:
                        val = sidecar_pollutant_mass(link_hit["attributed_pollutants"], row["pollutant"])
                        row["emission"] = val
                        row["emission_label"] = _fmt_sci(val) if val > 0 else "—"
                for k, v in flat.items():
                    if k.startswith("meta_") and v:
                        details.append({
                            "label": k[5:].replace("_", " ").title(),
                            "value": str(v),
                        })
                out["details"] = details
        if _cell("match_distance_km") is not None:
            try:
                out["match_distance_km"] = float(rec["match_distance_km"])
            except (TypeError, ValueError):
                pass
        self._cache[cache_key] = out
        return out

    def mass_outcome_line(self, match_status: str) -> str:
        pm = self.point_matching
        proc = pm["procedure"]
        status = str(match_status or "")
        handling = (self.config.get("output") or {}).get("partial_match_handling")
        if status == "matched_not_appointed" and handling == "facility_or_drop":
            return (
                f"{procedure_label(pm)}: partial match — mass attributed to the facility when "
                "it falls inside the domain; otherwise disregarded."
            )
        if proc == "merged":
            if status == "matched_appointed":
                return (
                    f"{procedure_label(pm)}: matched emission burned onto the facility pixel "
                    "in the merged emission grid (single output layer)."
                )
            if status == "unmatched":
                return (
                    f"{procedure_label(pm)}: unmatched mass added to the CAMS cell area budget, "
                    "downscaled with area weights, and included in the merged emission grid."
                )
            if status == "matched_not_appointed":
                return (
                    f"{procedure_label(pm)}: partial match outside domain — mass downscaled with "
                    "area weights in the merged emission grid."
                )
            return f"{procedure_label(pm)}: emission in the merged emission grid."
        if pm["unmatched"] == "burn_to_area":
            if status == "matched_appointed":
                return (
                    f"{procedure_label(pm)}: matched emission in point_emission_grid; "
                    "area sources in area_emission_grid."
                )
            if status == "unmatched":
                return (
                    f"{procedure_label(pm)}: unmatched mass added to the area grid and "
                    "downscaled with area weights (not in point_emission_grid)."
                )
            return f"{procedure_label(pm)}: see area_emission_grid and point_emission_grid."
        if status == "matched_appointed":
            return (
                f"{procedure_label(pm)}: matched emission at facility location in point_emission_grid."
            )
        if status == "unmatched":
            return (
                f"{procedure_label(pm)}: unmatched emission kept at CAMS location in point_emission_grid."
            )
        return f"{procedure_label(pm)}: emission in point_emission_grid."

    def facility_in_domain(self, lon: float, lat: float) -> bool:
        west, south, east, north = domain_wgs84(self.domain)
        return _in_wgs84_bbox(lon, lat, west, south, east, north)

    def partial_match_notice(
        self,
        match_status: str,
        *,
        facility_in_domain: bool | None = None,
    ) -> str | None:
        if str(match_status or "") != "matched_not_appointed":
            return None
        handling = (self.config.get("output") or {}).get("partial_match_handling")
        intro = (
            "This point is matched but either the CAMS point or the facility location "
            "does not fall inside your domain."
        )
        if handling == "facility_or_drop":
            if facility_in_domain is False:
                return (
                    intro + " Mass is disregarded because the facility falls outside your domain."
                )
            if facility_in_domain is True:
                return intro + " Mass is attributed to the facility pixel inside your domain."
            return (
                intro + " Mass is attributed to the facility when inside the domain; "
                "otherwise it is disregarded."
            )
        pm = self.point_matching
        if pm["procedure"] == "merged":
            mass = (
                " With merged layers, its mass is downscaled with area weights on the "
                "CAMS grid rather than at the facility pixel."
            )
        else:
            mass = " Therefore mass is kept at the CAMS original point in the point output."
        return intro + mass + " We strongly advise adjusting your bounding box."

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
        out["emission"] = pd.to_numeric(out["emission"], errors="coerce").fillna(0).astype(np.float32)
        return out[out["emission"] > 0]

    def _read_nc_grid(self, path: Path, pollutant: str) -> pd.DataFrame:
        da = xr.open_dataarray(path)
        try:
            if pollutant not in [str(p) for p in da.coords["pollutant"].values]:
                raise ValueError(f"pollutant {pollutant!r} not in {path.name}")
            plane = da.sel(pollutant=pollutant).values.astype(np.float32)
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
        if layer == "point" and self.point_matching["procedure"] == "merged":
            empty = pd.DataFrame(columns=["x", "y", "emission"])
            self._cache[key] = empty
            return empty
        if self.point_matching["procedure"] == "merged":
            if sector_id == "TOTAL":
                path = _grid_path(self.output_dir, "merged_emission_grid", self.fmt)
            else:
                path = _grid_path(self.output_dir / sector_id, "merged_emission_grid", self.fmt)
        else:
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
