"""Agriculture K+L CAMS area: NUTS2 x CLC w_p on CORINE grid, renormalized per CAMS cell."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from rasterio import features
from rasterio.enums import MergeAlg
from rasterio.windows import Window, from_bounds
from shapely.geometry import mapping

from .._load_aux import load_cams_a_publicpower
from ..corine_clc import corine_grid_to_weight_codes
from ..grid import nuts2_for_country, resolve_path
from ..manifest import write_manifest
from ..progress_util import note, tqdm_if_installed


def _kl_area_mask(ds: xr.Dataset, iso3: str, root: Path) -> np.ndarray:
    ca = load_cams_a_publicpower(root)
    ix = ca._country_index_1based(ds, iso3)
    emis = np.asarray(ds["emission_category_index"].values).ravel().astype(np.int64)
    ci = np.asarray(ds["country_index"].values).ravel().astype(np.int64)
    st = np.asarray(ds["source_type_index"].values).ravel().astype(np.int64)
    lon = np.asarray(ds["longitude_source"].values).ravel()
    lat = np.asarray(ds["latitude_source"].values).ravel()
    kl = np.isin(emis, np.array([14, 15], dtype=np.int64))
    base = kl & ca._build_domain_mask(lon, lat, ci, ix, None)
    return base & (st == 1)


def _cams_indices(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lon_b = np.asarray(ds["longitude_bounds"].values, dtype=np.float64)
    lat_b = np.asarray(ds["latitude_bounds"].values, dtype=np.float64)
    nlon = int(lon_b.shape[0])
    nlat = int(lat_b.shape[0])
    lon_idx_raw = np.asarray(ds["longitude_index"].values).ravel().astype(np.int64)
    lat_idx_raw = np.asarray(ds["latitude_index"].values).ravel().astype(np.int64)
    if lon_idx_raw.max() >= nlon or lat_idx_raw.max() >= nlat:
        lon_idx_raw = np.maximum(0, lon_idx_raw - 1)
        lat_idx_raw = np.maximum(0, lat_idx_raw - 1)
    lon_ii = np.clip(lon_idx_raw, 0, nlon - 1)
    lat_ii = np.clip(lat_idx_raw, 0, nlat - 1)
    return lon_ii, lat_ii, lon_b, lat_b


def build_agriculture_sourcearea(
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

    nc = Path(paths["cams_nc"])
    if not nc.is_absolute():
        nc = root / nc
    if not nc.is_file():
        raise FileNotFoundError(f"CAMS NetCDF not found: {nc}")

    weights_csv = Path(paths["weights_long"])
    if not weights_csv.is_absolute():
        weights_csv = root / weights_csv
    if not weights_csv.is_file():
        raise FileNotFoundError(f"weights_long not found: {weights_csv}")

    nuts_gpkg = resolve_path(root, paths["nuts_gpkg"])
    if not nuts_gpkg.is_file():
        raise FileNotFoundError(f"NUTS gpkg not found: {nuts_gpkg}")

    corine_path = Path(ref["corine_path"])
    corine_band = int(corine_cfg.get("band", 1))
    ag_codes = tuple(
        int(x) for x in (sector_entry.get("ag_clc_codes") or range(12, 23))
    )
    pollutants = sector_entry.get("pollutants") or ["NH3"]
    if isinstance(pollutants, str):
        pollutants = [pollutants]

    show_progress = bool(cfg.get("show_progress", True))

    h, w = int(ref["height"]), int(ref["width"])
    transform = ref["transform"]
    left, bottom, right, top = (float(x) for x in ref["window_bounds_3035"])

    n2 = nuts2_for_country(nuts_gpkg, str(country["nuts_cntr"]))
    if show_progress:
        note("Agriculture area: rasterizing NUTS2 ids on reference grid…")
    nuts_to_idx: dict[str, int] = {}
    shapes = []
    for k, (_, row) in enumerate(n2.iterrows()):
        nid = str(row["NUTS_ID"]).strip()
        nuts_to_idx[nid] = k + 1
        shapes.append((mapping(row.geometry), k + 1))

    nuts_r = features.rasterize(
        shapes,
        out_shape=(h, w),
        transform=transform,
        fill=0,
        dtype=np.int32,
        merge_alg=MergeAlg.replace,
    )

    if show_progress:
        note(f"Agriculture area: reading CORINE window ({w}×{h})…")
    with rasterio.open(corine_path) as src:
        if corine_band < 1 or corine_band > int(src.count):
            raise ValueError(f"corine band {corine_band} invalid")
        win = from_bounds(left, bottom, right, top, transform=src.transform)
        win = win.round_lengths().intersection(Window(0, 0, src.width, src.height))
        corine_arr = src.read(corine_band, window=win).astype(np.float64)
        cn = src.nodata

    if corine_arr.shape != (h, w):
        raise ValueError(
            f"CORINE window shape {corine_arr.shape} != ref grid {(h, w)}; check NUTS/CORINE alignment."
        )

    ok_cor = np.isfinite(corine_arr)
    if cn is not None:
        ok_cor = ok_cor & (corine_arr != float(cn))
    rint = np.zeros_like(corine_arr, dtype=np.int32)
    rint[ok_cor] = np.rint(corine_arr[ok_cor]).astype(np.int32)
    wclc = corine_grid_to_weight_codes(rint)
    in_ag = ok_cor & (wclc >= 0) & np.isin(wclc, np.array(ag_codes, dtype=np.int32))

    from rasterio.transform import xy as transform_xy
    from rasterio.warp import transform as rio_transform_pts

    if show_progress:
        note("Agriculture area: assigning CAMS K/L cells on grid…")
    ds = xr.open_dataset(nc)
    try:
        lon_ii, lat_ii, lon_b, lat_b = _cams_indices(ds)
        m_kl = _kl_area_mask(ds, str(country["cams_iso3"]), root)

        crs_obj = rasterio.crs.CRS.from_string(ref["crs"])
        rows, cols = np.indices((h, w))
        xs, ys = transform_xy(transform, rows + 0.5, cols + 0.5, offset="center")
        lons, lats = rio_transform_pts(crs_obj, "EPSG:4326", xs.flatten(), ys.flatten())
        lons = np.asarray(lons, dtype=np.float64).reshape(h, w)
        lats = np.asarray(lats, dtype=np.float64).reshape(h, w)

        nlon = int(lon_b.shape[0])
        nlat = int(lat_b.shape[0])
        li = np.searchsorted(lon_b[:, 0], lons, side="right") - 1
        li = np.clip(li, 0, nlon - 1)
        ji = np.searchsorted(lat_b[:, 0], lats, side="right") - 1
        ji = np.clip(ji, 0, nlat - 1)
        valid_lon = (lons >= lon_b[li, 0]) & (lons <= lon_b[li, 1])
        valid_lat = (lats >= lat_b[ji, 0]) & (lats <= lat_b[ji, 1])
        in_bounds = valid_lon & valid_lat

        lookup = np.full(nlon * nlat, -1, dtype=np.int64)
        for i in sorted(int(x) for x in np.flatnonzero(m_kl)):
            li_i, ji_i = int(lon_ii[i]), int(lat_ii[i])
            k = li_i * nlat + ji_i
            if lookup[k] < 0:
                lookup[k] = int(i)
        fk = li * nlat + ji
        cid = lookup[fk]
        cell_of = np.where(in_bounds & (cid >= 0), cid, -1).astype(np.int64)
    finally:
        ds.close()

    n_nuts = len(nuts_to_idx) + 1
    bands: list[np.ndarray] = []
    df_all = pd.read_csv(weights_csv)

    pol_iter = tqdm_if_installed(
        pollutants,
        desc="Agriculture area: pollutants",
        unit="band",
        total=len(pollutants),
    )
    for pol in pol_iter:
        pol_u = str(pol).strip().upper()
        sub = df_all[df_all["pollutant"].astype(str).str.upper() == pol_u]
        if sub.empty:
            raise ValueError(f"No rows in weights_long for pollutant={pol!r}")

        lookup_m = np.zeros((n_nuts, 23), dtype=np.float64)
        for _, r in sub.iterrows():
            nid = str(r["NUTS_ID"]).strip()
            idx = nuts_to_idx.get(nid)
            if idx is None:
                continue
            clc = int(r["CLC_CODE"])
            if 0 <= clc <= 22:
                lookup_m[idx, clc] = float(r["w_p"])

        raw = np.zeros((h, w), dtype=np.float64)
        valid_nuts = (nuts_r >= 1) & (nuts_r < n_nuts)
        for cl in ag_codes:
            m = in_ag & (wclc == cl) & valid_nuts
            if not m.any():
                continue
            ii = nuts_r[m]
            raw[m] = lookup_m[ii, cl]

        n_src = int(m_kl.size)
        flat_cell = cell_of.ravel()
        flat_raw = raw.ravel()
        assigned = flat_cell >= 0
        idx_pos = flat_cell[assigned].astype(np.int64, copy=False)
        sums = np.bincount(idx_pos, weights=flat_raw[assigned], minlength=n_src)

        den = np.zeros(flat_cell.shape[0], dtype=np.float64)
        den[assigned] = sums[idx_pos]
        out_flat = np.zeros_like(flat_raw, dtype=np.float64)
        ok = assigned & (den > 1e-30)
        np.divide(flat_raw, den, out=out_flat, where=ok)
        out = out_flat.reshape(h, w).astype(np.float32)

        for i in np.flatnonzero(m_kl):
            ii = int(i)
            if sums[ii] > 0:
                continue
            mask = cell_of == ii
            if not mask.any():
                continue
            agpix = mask & in_ag
            n_ag = int(np.count_nonzero(agpix))
            if n_ag > 0:
                out[agpix] = np.float32(1.0 / n_ag)

        bands.append(out)

    if run_validate:
        from ..validate import check_agriculture_raster, report_validation

        for b, pol in enumerate(pollutants):
            report_validation(
                f"agriculture_area raster band {pol!r}",
                check_agriculture_raster(bands[b], cell_of, m_kl),
            )

    out_dir = Path(cfg["output_dir"])
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_tif = out_dir / str(sector_entry["filename"])
    manifest_path = out_tif.with_suffix(".json")

    stack = np.stack(bands, axis=0) if len(bands) > 1 else bands[0][np.newaxis, ...]
    count = stack.shape[0]

    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": count,
        "dtype": "float32",
        "crs": ref["crs"],
        "transform": transform,
        "compress": "lzw",
    }
    if show_progress:
        note(f"Agriculture area: writing GeoTIFF {out_tif.name} ({count} band(s))…")
    with rasterio.open(out_tif, "w", **profile) as dst:
        for b in range(count):
            dst.write(stack[b], b + 1)
            dst.set_band_description(
                b + 1,
                f"weight_share_agri_{str(pollutants[b]).upper()}",
            )

    rel_out = out_tif
    try:
        rel_out = out_tif.relative_to(root)
    except ValueError:
        pass

    write_manifest(
        manifest_path,
        {
            "builder": "agriculture_area",
            "output_geotiff": str(rel_out),
            "crs": ref["crs"],
            "width": w,
            "height": h,
            "pollutants": [str(p) for p in pollutants],
            "domain_bbox_wgs84": list(ref["domain_bbox_wgs84"]),
            "weights_long": str(weights_csv),
            "ag_clc_codes": list(ag_codes),
        },
    )
    return out_tif
