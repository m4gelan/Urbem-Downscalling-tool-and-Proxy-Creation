"""
Multi-band agriculture area weights: NUTS2 x CLC w_p on CORINE ref grid,
renormalized per CAMS K/L area cell.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from rasterio.enums import MergeAlg
from rasterio.windows import Window, from_bounds
from shapely.geometry import mapping

from PROXY.core.cams import build_cams_source_index_grid_any_gnfr
from PROXY.core.grid import nuts2_for_country
from PROXY.core.io import write_geotiff, write_json

from .corine_weight_codes import corine_grid_to_weight_codes

logger = logging.getLogger(__name__)

def _note(msg: str) -> None:
    logger.info("%s", msg)


def build_kl_sourcearea_tif(
    root: Path,
    *,
    ref: dict[str, Any],
    cams_nc: Path,
    weights_long: Path,
    nuts_gpkg: Path,
    out_tif: Path,
    cams_iso3: str,
    nuts_cntr: str,
    pollutants: list[str],
    ag_clc_codes: tuple[int, ...] | None = None,
    corine_band: int = 1,
    cams_emission_category_indices: Sequence[int] = (14, 15),
    run_validate: bool = False,
) -> Path:
    ag_codes = tuple(int(x) for x in (ag_clc_codes or range(12, 23)))
    pols = [str(p).strip() for p in pollutants]
    if not pols:
        raise ValueError("pollutants list is empty")

    if not cams_nc.is_file():
        raise FileNotFoundError(f"CAMS NetCDF not found: {cams_nc}")
    if not weights_long.is_file():
        raise FileNotFoundError(f"weights_long not found: {weights_long}")
    if not nuts_gpkg.is_file():
        raise FileNotFoundError(f"NUTS gpkg not found: {nuts_gpkg}")

    h, w = int(ref["height"]), int(ref["width"])
    transform = ref["transform"]
    left, bottom, right, top = (float(x) for x in ref["window_bounds_3035"])
    corine_path = Path(str(ref["corine_path"]))

    n2 = nuts2_for_country(nuts_gpkg, nuts_cntr)
    _note("Agriculture K/L: rasterizing NUTS2 ids on reference grid…")
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

    _note(f"Agriculture K/L: reading CORINE window ({w}×{h})…")
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

    _note("Agriculture K/L: assigning CAMS cells on grid…")
    cell_of, m_kl = build_cams_source_index_grid_any_gnfr(
        cams_nc,
        ref,
        gnfr_indices=tuple(int(x) for x in cams_emission_category_indices),
        source_type_index=1,
        country_iso3=cams_iso3,
    )
    logger.info(
        "Agriculture K/L: CAMS source rows=%d valid fine pixels=%d",
        int(np.count_nonzero(m_kl)),
        int(np.count_nonzero(cell_of >= 0)),
    )

    n_nuts = len(nuts_to_idx) + 1
    df_all = pd.read_csv(weights_long)
    df_all["_pollutant_upper"] = df_all["pollutant"].astype(str).str.upper()
    bands: list[np.ndarray] = []
    n_src = int(m_kl.size)
    valid_nuts = (nuts_r >= 1) & (nuts_r < n_nuts)
    valid_pix = in_ag & valid_nuts

    for pol in pols:
        pol_u = str(pol).strip().upper()
        sub = df_all[df_all["_pollutant_upper"] == pol_u]
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
        raw[valid_pix] = lookup_m[nuts_r[valid_pix], wclc[valid_pix]]
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
        logger.info(
            "Agriculture K/L: pollutant=%s raw_sum=%.6g final_sum=%.6g active_pixels=%d",
            pol_u,
            float(np.sum(raw)),
            float(np.sum(out)),
            int(np.count_nonzero(out > 0)),
        )
        bands.append(out)

    if run_validate:
        try:
            from SourceProxies.validate import check_agriculture_raster, report_validation

            for b, pol in enumerate(pols):
                report_validation(
                    f"K_Agriculture raster band {pol!r}",
                    check_agriculture_raster(bands[b], cell_of, m_kl),
                )
        except Exception as ex:
            logger.warning("Agriculture validation skipped: %s", ex)

    stack = np.stack(bands, axis=0) if len(bands) > 1 else bands[0][np.newaxis, ...]
    count = stack.shape[0]
    _note(f"Agriculture K/L: writing GeoTIFF {out_tif.name} ({count} band(s))…")
    write_geotiff(
        path=out_tif,
        array=stack.astype(np.float32, copy=False),
        crs=str(ref["crs"]),
        transform=transform,
        band_descriptions=[f"weight_share_agri_{str(pol).upper()}" for pol in pols],
    )
    man = out_tif.with_suffix(".json")
    try:
        rel_out = str(out_tif.relative_to(root))
    except ValueError:
        rel_out = str(out_tif)
    write_json(
        man,
        {
            "builder": "K_Agriculture_kl",
            "output_geotiff": rel_out,
            "crs": str(ref["crs"]),
            "width": w,
            "height": h,
            "pollutants": [str(p) for p in pols],
            "domain_bbox_wgs84": list(ref.get("domain_bbox_wgs84", ())),
            "weights_long": str(weights_long),
            "ag_clc_codes": list(ag_codes),
        },
    )
    return out_tif
