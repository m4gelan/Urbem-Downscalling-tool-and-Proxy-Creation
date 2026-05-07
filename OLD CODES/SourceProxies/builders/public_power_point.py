"""JRC OPEN UNITS (combustion) rasterized on the national CORINE reference grid."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import rowcol

from .._load_aux import load_cams_a_publicpower, load_module_at
from ..grid import resolve_path
from ..manifest import write_manifest
from ..progress_util import note, tqdm_if_installed


def build_public_power_sourcepoint(
    root: Path,
    cfg: dict[str, Any],
    ref: dict[str, Any],
    *,
    sector_entry: dict[str, Any],
) -> Path:
    ca = load_cams_a_publicpower(root)
    jpp = load_module_at(root / "PublicPower" / "jrc_public_power.py", "jrc_public_power")

    paths = cfg["paths"]
    country = cfg["country"]
    show_progress = bool(cfg.get("show_progress", True))
    iso3 = str(country["cams_iso3"]).strip().upper()
    jrc_name = sector_entry.get("jrc_country_override")
    if jrc_name:
        jrc_name = str(jrc_name).strip()
    else:
        jrc_name = ca._jrc_country_name_for_cams_iso3(iso3, None)
    if not jrc_name:
        raise ValueError(
            f"No JRC country name mapping for CAMS ISO3 {iso3!r}. "
            "Set sector jrc_country_override or extend ISO3_TO_JRC_COUNTRY_NAME."
        )

    csv_path = resolve_path(root, paths["jrc_csv"])
    if not csv_path.is_file():
        raise FileNotFoundError(f"JRC CSV not found: {csv_path}")

    exclude_hydro = bool(sector_entry.get("exclude_hydro", True))
    combustion_only = bool(sector_entry.get("combustion_only", True))
    mode = str(sector_entry.get("point_mode", "binary")).lower()
    if mode not in ("binary", "count"):
        mode = "binary"

    df = ca.load_jrc_open_units(
        csv_path,
        country_name=jrc_name,
        bbox=None,
        exclude_hydro_types=exclude_hydro,
    )
    if df.empty:
        raise ValueError(f"No JRC rows after filter for country={jrc_name!r}")

    if combustion_only and "type_g" in df.columns:
        canonical = jpp.load_datapackage_type_g_enum()
        tg = df["type_g"].map(lambda x: jpp.normalize_type_g(x, canonical))
        df = df[tg.isin(jpp.COMBUSTION_TYPES)].copy()
        if df.empty:
            raise ValueError(
                f"No JRC combustion units for country={jrc_name!r} after type_g filter."
            )

    h, w = int(ref["height"]), int(ref["width"])
    transform = ref["transform"]
    crs = rasterio.crs.CRS.from_string(ref["crs"])

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon_f"], df["lat_f"], crs="EPSG:4326"),
    )
    g3035 = gdf.to_crs(crs)

    acc = np.zeros((h, w), dtype=np.float32)
    n_pt = int(len(g3035))
    if show_progress:
        note(f"Public power point: burning {n_pt} JRC units onto {w}×{h} grid…")
    geom_iter = tqdm_if_installed(
        g3035.geometry,
        desc="JRC points → pixels",
        unit="pt",
        total=n_pt,
    )
    for geom in geom_iter:
        if geom is None or geom.is_empty:
            continue
        x, y = float(geom.x), float(geom.y)
        try:
            r, c = rowcol(transform, x, y)
        except Exception:
            continue
        if 0 <= r < h and 0 <= c < w:
            if mode == "count":
                acc[r, c] += 1.0
            else:
                acc[r, c] = 1.0

    out_dir = Path(cfg["output_dir"])
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_tif = out_dir / str(sector_entry["filename"])
    manifest_path = out_tif.with_suffix(".json")

    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 1,
        "dtype": "float32",
        "crs": ref["crs"],
        "transform": transform,
        "compress": "lzw",
    }
    if show_progress:
        note(f"Public power point: writing GeoTIFF {out_tif.name}…")
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(acc, 1)
        desc = "jrc_point_binary" if mode == "binary" else "jrc_point_count"
        dst.set_band_description(1, desc)

    rel_out = out_tif
    try:
        rel_out = out_tif.relative_to(root)
    except ValueError:
        pass

    write_manifest(
        manifest_path,
        {
            "builder": "public_power_point",
            "output_geotiff": str(rel_out),
            "crs": ref["crs"],
            "width": w,
            "height": h,
            "jrc_csv": str(csv_path),
            "jrc_country": jrc_name,
            "point_mode": mode,
            "n_points": int(len(g3035)),
            "domain_bbox_wgs84": list(ref["domain_bbox_wgs84"]),
        },
    )
    return out_tif
