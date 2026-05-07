"""Shared OSM I/O, bbox extract, clip, GPKG write (EPSG:3035)."""

from __future__ import annotations

import json
import shutil
import subprocess
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import geopandas as gpd
from shapely import from_wkb
from shapely.geometry import Point as ShpPoint

try:
    import osmium
except ImportError as e:  # pragma: no cover
    raise SystemExit("Install: pip install osmium geopandas shapely pyproj") from e

TARGET_CRS = "EPSG:3035"
MAX_PBF_BYTES_WITHOUT_OSMIUM_TOOL = 250 * 1024 * 1024


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def wkb_to_bytes(wkb: bytes | str | memoryview) -> bytes:
    if isinstance(wkb, bytes):
        return wkb
    if isinstance(wkb, memoryview):
        return bytes(wkb)
    if isinstance(wkb, str):
        return wkb.encode("latin-1")
    return bytes(wkb)


def tags_to_dict(taglist: Any) -> dict[str, str]:
    return {t.k: t.v for t in taglist}


def resolve_osmium_exe(cli: str | None) -> str | None:
    if cli:
        p = Path(cli).expanduser().resolve()
        if not p.is_file():
            raise SystemExit(f"--osmium does not exist: {p}")
        return str(p)
    exe = shutil.which("osmium")
    return str(Path(exe).resolve()) if exe else None


def run_osmium(osmium_exe: str, *tail: str) -> None:
    r = subprocess.run([osmium_exe, *tail], check=False)
    if r.returncode != 0:
        raise RuntimeError(
            f"osmium failed ({r.returncode}): {' '.join(tail[:8])}\n"
            "See osmium error output printed above."
        )


def extract_bbox(osmium_exe: str, bbox: str, source: Path, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    last: RuntimeError | None = None
    for want_progress in (True, False):
        try:
            if want_progress:
                run_osmium(
                    osmium_exe,
                    "extract",
                    "--progress",
                    "--overwrite",
                    "-b",
                    bbox,
                    "-o",
                    str(out),
                    str(source),
                )
            else:
                run_osmium(
                    osmium_exe,
                    "extract",
                    "--no-progress",
                    "--overwrite",
                    "-b",
                    bbox,
                    "-o",
                    str(out),
                    str(source),
                )
            if not want_progress:
                warnings.warn(
                    "osmium extract succeeded without progress bar (first attempt failed).",
                    stacklevel=2,
                )
            return
        except RuntimeError as e:
            last = e
            if want_progress:
                warnings.warn(
                    "osmium extract failed with --progress; retrying with --no-progress.",
                    stacklevel=2,
                )
    raise RuntimeError(
        "osmium extract failed twice (with and without progress bar)."
    ) from last


def tags_filter(osmium_exe: str, source: Path, out: Path, filters: Iterable[str]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    filt = tuple(filters)
    last: RuntimeError | None = None
    for want_progress in (True, False):
        try:
            if want_progress:
                run_osmium(
                    osmium_exe,
                    "tags-filter",
                    "--progress",
                    "--overwrite",
                    "-o",
                    str(out),
                    str(source),
                    *filt,
                )
            else:
                run_osmium(
                    osmium_exe,
                    "tags-filter",
                    "--no-progress",
                    "--overwrite",
                    "-o",
                    str(out),
                    str(source),
                    *filt,
                )
            if not want_progress:
                warnings.warn(
                    "osmium tags-filter succeeded without progress bar (first attempt failed).",
                    stacklevel=2,
                )
            return
        except RuntimeError as e:
            last = e
            if want_progress:
                warnings.warn(
                    "osmium tags-filter failed with --progress; retrying with --no-progress.",
                    stacklevel=2,
                )
    raise RuntimeError("osmium tags-filter failed twice.") from last


def load_boundary(nuts_path: Path, cntr_code: str | None) -> tuple[gpd.GeoDataFrame, int]:
    gdf = gpd.read_file(nuts_path)
    if cntr_code is not None:
        gdf = gdf[gdf["CNTR_CODE"] == cntr_code.upper()].copy()
        if gdf.empty:
            raise SystemExit(f"No features for CNTR_CODE={cntr_code!r} in {nuts_path}")
    n_features = len(gdf)
    return gdf.dissolve(), n_features


def bbox_str_wgs84(boundary_3035: gpd.GeoDataFrame) -> str:
    b = boundary_3035.to_crs(4326).total_bounds
    west, south, east, north = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    return f"{west},{south},{east},{north}"


def empty_gdf_wgs84() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"geometry": gpd.GeoSeries([], crs="EPSG:4326")})


def dedupe_osm_id(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    k = gdf["osm_element_type"].astype(str) + ":" + gdf["osm_element_id"].astype(str)
    return gdf.loc[~k.duplicated(keep="first")].copy()


def clip_mixed_to_3035(gdf: gpd.GeoDataFrame, boundary_wgs: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    gt = gdf.geometry.geom_type
    poly_m = gt.isin(["Polygon", "MultiPolygon"])
    if poly_m.any():
        gdf.loc[poly_m, "geometry"] = gdf.loc[poly_m, "geometry"].buffer(0)
        gdf = gdf[~gdf.geometry.is_empty]
    gdf = gpd.clip(gdf, boundary_wgs)
    if gdf.empty:
        return gdf
    return gdf.to_crs(TARGET_CRS)


def clip_lines_to_3035(gdf: gpd.GeoDataFrame, boundary_wgs: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    gdf = gpd.clip(gdf, boundary_wgs)
    if gdf.empty:
        return gdf
    return gdf.to_crs(TARGET_CRS)


def clip_points_to_3035(gdf: gpd.GeoDataFrame, boundary_wgs: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    gdf = gpd.clip(gdf, boundary_wgs)
    if gdf.empty:
        return gdf
    return gdf.to_crs(TARGET_CRS)


def filter_polygon_min_area(gdf: gpd.GeoDataFrame, min_m2: float) -> gpd.GeoDataFrame:
    if gdf.empty or min_m2 <= 0:
        return gdf
    poly = gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    if not poly.any():
        return gdf
    area = gdf.geometry.area
    keep = ~poly | (area >= float(min_m2))
    return gdf.loc[keep].copy()


def write_gpkg(out: Path, layers: Iterable[tuple[str, gpd.GeoDataFrame]]) -> None:
    nonempty = [(n, g) for n, g in layers if not g.empty]
    if not nonempty:
        raise SystemExit("All layers empty after processing; check PBF, NUTS mask, and tag rules.")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        try:
            out.unlink()
        except OSError:
            pass
    name0, gdf0 = nonempty[0]
    gdf0.to_file(out, layer=name0, driver="GPKG", mode="w")
    for name, gdf in nonempty[1:]:
        gdf.to_file(out, layer=name, driver="GPKG", mode="a")
    print(f"Wrote {out} ({len(nonempty)} layers, crs={TARGET_CRS})")
    for name, gdf in nonempty:
        print(f"  {name}: {len(gdf)} features", flush=True)


def write_gpkg_allow_empty(out: Path, layers: Iterable[tuple[str, gpd.GeoDataFrame]]) -> None:
    """Write GPKG with at least one layer (first nonempty, or first layer empty)."""
    layers = list(layers)
    nonempty = [(n, g) for n, g in layers if not g.empty]
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        try:
            out.unlink()
        except OSError:
            pass
    if nonempty:
        name0, gdf0 = nonempty[0]
        gdf0.to_file(out, layer=name0, driver="GPKG", mode="w")
        for name, gdf in nonempty[1:]:
            gdf.to_file(out, layer=name, driver="GPKG", mode="a")
    else:
        name0, gdf0 = layers[0]
        gdf0.to_file(out, layer=name0, driver="GPKG", mode="w")
        for name, gdf in layers[1:]:
            gdf.to_file(out, layer=name, driver="GPKG", mode="a")
    print(f"Wrote {out} (crs={TARGET_CRS})")
