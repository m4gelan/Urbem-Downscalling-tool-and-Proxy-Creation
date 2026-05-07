"""Public power CAMS area: CORINE+LandScan weight_share raster on national CORINE window."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import xarray as xr
from rasterio import features
from rasterio.enums import MergeAlg
from shapely.geometry import mapping

from .._load_aux import load_cams_a_publicpower, load_cams_area_downscale
from ..manifest import write_manifest
from ..progress_util import note, tqdm_if_installed


def _public_power_area_mask(ds: xr.Dataset, iso3: str, root: Path) -> np.ndarray:
    ca = load_cams_a_publicpower(root)
    ix = ca._country_index_1based(ds, iso3)
    emis = np.asarray(ds["emission_category_index"].values).ravel().astype(np.int64)
    ci = np.asarray(ds["country_index"].values).ravel().astype(np.int64)
    st = np.asarray(ds["source_type_index"].values).ravel().astype(np.int64)
    lon = np.asarray(ds["longitude_source"].values).ravel()
    lat = np.asarray(ds["latitude_source"].values).ravel()
    base = (emis == ca.IDX_A_PUBLIC_POWER) & ca._build_domain_mask(
        lon, lat, ci, ix, None
    )
    return base & (st == 1)


def build_public_power_sourcearea(
    root: Path,
    cfg: dict[str, Any],
    ref: dict[str, Any],
    *,
    sector_entry: dict[str, Any],
    run_validate: bool = False,
) -> Path:
    paths = cfg["paths"]
    country = cfg["country"]
    corine_cfg = cfg.get("corine") or {}
    ls_cfg = cfg.get("landscan") or {}

    nc = Path(paths["cams_nc"])
    if not nc.is_absolute():
        nc = root / nc
    if not nc.is_file():
        raise FileNotFoundError(f"CAMS NetCDF not found: {nc}")

    corine_path = Path(ref["corine_path"])
    landscan = Path(paths["landscan"])
    if not landscan.is_absolute():
        landscan = root / landscan
    if not landscan.is_file():
        raise FileNotFoundError(f"LandScan not found: {landscan}")

    out_dir = Path(cfg["output_dir"])
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_tif = out_dir / str(sector_entry["filename"])
    manifest_path = out_tif.with_suffix(".json")

    down = load_cams_area_downscale(root)
    codes = tuple(int(x) for x in corine_cfg.get("public_power_corine_codes", [121, 3]))
    corine_band = int(corine_cfg.get("band", 1))
    resampling = str(ls_cfg.get("resampling", "bilinear"))
    if resampling not in ("bilinear", "nearest"):
        resampling = "bilinear"

    show_progress = bool(cfg.get("show_progress", True))

    ds = xr.open_dataset(nc)
    try:
        mask = _public_power_area_mask(ds, str(country["cams_iso3"]), root)
        if show_progress:
            note("Public power area: building CORINE/LandScan weights per CAMS cell…")
        gdf = down.build_cams_area_corine_landscan_weights(
            ds,
            mask,
            corine_path=corine_path,
            landscan_path=landscan,
            corine_codes=codes,
            corine_band=corine_band,
            pop_exponent=float(ls_cfg.get("pop_exponent", 1.0)),
            pop_floor=float(ls_cfg.get("pop_floor", 0.0)),
            landscan_resampling=resampling,  # type: ignore[arg-type]
            domain_bbox_wgs84=tuple(float(x) for x in ref["domain_bbox_wgs84"]),
            show_progress=show_progress,
        )
    finally:
        ds.close()

    if run_validate:
        from ..validate import check_public_power_area_gdf, report_validation

        report_validation(
            "public_power_area (per-CAMS-cell weight_share in GeoDataFrame)",
            check_public_power_area_gdf(gdf),
        )

    h, w = int(ref["height"]), int(ref["width"])
    transform = ref["transform"]
    acc = np.zeros((h, w), dtype=np.float32)
    if not gdf.empty:
        crs = rasterio.crs.CRS.from_string(ref["crs"])
        if show_progress:
            note("Public power area: reprojecting footprints to CORINE CRS…")
        g3035 = gdf.to_crs(crs)
        shapes: list[tuple[object, float]] = []
        n_rows = int(len(g3035))
        row_it = tqdm_if_installed(
            g3035.iterrows(),
            desc="PP area: footprints → raster shapes",
            unit="poly",
            total=n_rows,
        )
        for _, row in row_it:
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            sh = float(row["weight_share"])
            if sh <= 0:
                continue
            shapes.append((mapping(geom), sh))
        if shapes:
            if show_progress:
                note(
                    f"Public power area: rasterizing {len(shapes):,} polygons "
                    f"onto {w}×{h} grid (single pass)…"
                )
            features.rasterize(
                shapes,
                out=acc,
                transform=transform,
                merge_alg=MergeAlg.add,
            )

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
        note(f"Public power area: writing GeoTIFF {out_tif.name}…")
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(acc, 1)
        dst.set_band_description(1, "weight_share_public_power_area")

    rel_out = out_tif
    try:
        rel_out = out_tif.relative_to(root)
    except ValueError:
        pass

    write_manifest(
        manifest_path,
        {
            "builder": "public_power_area",
            "output_geotiff": str(rel_out),
            "crs": ref["crs"],
            "width": w,
            "height": h,
            "domain_bbox_wgs84": list(ref["domain_bbox_wgs84"]),
            "cams_nc": str(nc),
            "corine_path": str(corine_path),
            "landscan_path": str(landscan),
            "corine_codes": list(codes),
            "n_weight_features": int(len(gdf)),
        },
    )
    return out_tif
