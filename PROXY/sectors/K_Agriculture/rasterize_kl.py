"""
Multi-band agriculture area weights: seven spatial proxy families on the CORINE ref grid,
linearly combined with CEIP-derived alpha (method 1 / EU27 pool), then renormalized per CAMS K/L cell.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio import features
from rasterio.enums import MergeAlg, Resampling
from rasterio.windows import Window, from_bounds
from shapely.geometry import mapping

from PROXY.core.cams import build_cams_source_index_grid_any_gnfr
from PROXY.core.dataloaders.raster import warp_band_to_ref
from PROXY.core.grid import nuts2_for_country
from PROXY.core.io import write_geotiff, write_json

from .combine import combine_family_rasters
from .combined_config import compute_gamma_series
from .corine_weight_codes import corine_grid_to_weight_codes
from .signals.crop_operations import build_family5
from .signals.field_burning import build_family7
from .signals.grazing_pasture import build_family3
from .signals.housing_pasture import build_family1
from .signals.manure_application import build_family2
from .signals.minor_soil import build_family6
from .signals.synthetic_n import build_family4
from .tabular.class_extent import build_class_extent_long, load_nuts2_filtered

logger = logging.getLogger(__name__)


def _note(msg: str) -> None:
    logger.info("%s", msg)


def _cams_normalize_raw_to_weights(
    raw: np.ndarray,
    cell_of: np.ndarray,
    m_kl: np.ndarray,
    in_ag: np.ndarray,
) -> np.ndarray:
    """Divide raw scores by CAMS-cell sums; uniform fallback on ag pixels when cell sum is zero."""
    h, w = raw.shape
    flat_cell = cell_of.ravel()
    flat_raw = raw.ravel().astype(np.float64, copy=False)
    assigned = flat_cell >= 0
    idx_pos = flat_cell[assigned].astype(np.int64, copy=False)
    n_src = int(m_kl.size)
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
    return out


def build_kl_sourcearea_tif(
    root: Path,
    *,
    ref: dict[str, Any],
    cams_nc: Path,
    nuts_gpkg: Path,
    out_tif: Path,
    cams_iso3: str,
    nuts_cntr: str,
    pipeline_cfg: dict[str, Any],
    ag_clc_codes: tuple[int, ...] | None = None,
    corine_band: int = 1,
    cams_emission_category_indices: Sequence[int] = (14, 15),
    run_validate: bool = False,
) -> Path:
    cfg = pipeline_cfg
    ag_codes = tuple(int(x) for x in (ag_clc_codes or range(12, 23)))
    pol_keys = [str(p).strip() for p in (cfg.get("pollutants") or [])]
    if not pol_keys:
        raise ValueError("cfg['pollutants'] is empty (CEIP keys, e.g. nh3, nox)")

    if not cams_nc.is_file():
        raise FileNotFoundError(f"CAMS NetCDF not found: {cams_nc}")
    if not nuts_gpkg.is_file():
        raise FileNotFoundError(f"NUTS gpkg not found: {nuts_gpkg}")

    alpha_np = np.asarray(cfg.get("_ceip_alpha"), dtype=np.float64)
    if alpha_np.ndim != 3:
        raise ValueError("pipeline_cfg['_ceip_alpha'] must be a 3D array (1, n_groups, n_pollutants)")
    group_order = tuple(str(x).strip() for x in (cfg.get("group_order") or ()))
    if not group_order:
        raise ValueError("cfg['group_order'] is empty")
    if alpha_np.shape[1] != len(group_order):
        raise ValueError(
            f"alpha axis 1 ({alpha_np.shape[1]}) != len(group_order) ({len(group_order)})"
        )
    if alpha_np.shape[2] != len(pol_keys):
        raise ValueError(
            f"alpha axis 2 ({alpha_np.shape[2]}) != len(cfg['pollutants']) ({len(pol_keys)})"
        )

    band_labels = list(cfg.get("_raster_pollutant_labels") or [])
    while len(band_labels) < len(pol_keys):
        pk = pol_keys[len(band_labels)]
        pl = str(pk).strip().lower().replace(".", "_")
        band_labels.append("PM2.5" if pl == "pm2_5" else str(pk).strip().upper())

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

    bands: list[np.ndarray] = []

    _note("Agriculture K/L: seven-family proxy stack + CEIP alpha…")
    nodata = float((cfg.get("run") or {}).get("nodata", -128.0))
    nuts2_df = load_nuts2_filtered(cfg, root)
    extent_df = build_class_extent_long(nuts2_df, corine_path, nodata=nodata, ag_clc_codes=ag_codes)

    kop_path_str = (cfg.get("paths") or {}).get("inputs", {}).get("koppen_raster_tif")
    kop_arr: np.ndarray | None = None
    if kop_path_str:
        kop_p = Path(str(kop_path_str))
        if kop_p.is_file():
            kop_arr = warp_band_to_ref(kop_p, ref, resampling=Resampling.nearest, band=1).astype(np.float32)

    nuts_idx_to_id = {int(v): str(k) for k, v in nuts_to_idx.items()}
    gamma = compute_gamma_series(cfg, root, koppen_on_ref=kop_arr, nuts_r=nuts_r, nuts_idx_to_id=nuts_idx_to_id)

    p1 = build_family1(root, ref, cfg, corine_arr=corine_arr, corine_nodata=cn, nuts_r=nuts_r, nuts_to_idx=nuts_to_idx)
    p2 = build_family2(extent_df, cfg, nuts_r=nuts_r, corine_arr=corine_arr, nuts_to_idx=nuts_to_idx, corine_nodata=cn)
    p3 = build_family3(root, ref, cfg, corine_arr=corine_arr, corine_nodata=cn)
    p4 = build_family4(extent_df, cfg, nuts_r=nuts_r, corine_arr=corine_arr, nuts_to_idx=nuts_to_idx, corine_nodata=cn)
    p5 = build_family5(extent_df, cfg, root, nuts_r=nuts_r, corine_arr=corine_arr, nuts_to_idx=nuts_to_idx, corine_nodata=cn, gamma_series=gamma)
    p6 = build_family6(cfg, root, corine_arr=corine_arr, corine_nodata=cn)
    p7 = build_family7(cfg, root, ref)

    for j, pol_key in enumerate(pol_keys):
        pol_display = band_labels[j]
        raw = combine_family_rasters(
            pollutant=pol_display,
            alpha_vec=alpha_np[0, :, j],
            group_order=group_order,
            p1=p1,
            p2=p2,
            p3=p3,
            p4=p4,
            p5_by_pol=p5,
            p6=p6,
            p7=p7,
        )
        out = _cams_normalize_raw_to_weights(raw, cell_of, m_kl, in_ag)
        logger.info(
            "Agriculture K/L: pollutant=%s raw_sum=%.6g final_sum=%.6g active_pixels=%d",
            str(pol_display).upper(),
            float(np.sum(raw)),
            float(np.sum(out)),
            int(np.count_nonzero(out > 0)),
        )
        bands.append(out)

    if run_validate:
        try:
            from SourceProxies.validate import check_agriculture_raster, report_validation

            for b, pol in enumerate(band_labels):
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
        band_descriptions=[f"weight_share_agri_{str(lab).upper()}" for lab in band_labels[: len(bands)]],
    )
    man = out_tif.with_suffix(".json")
    try:
        rel_out = str(out_tif.relative_to(root))
    except ValueError:
        rel_out = str(out_tif)
    meta: dict[str, Any] = {
        "builder": "K_Agriculture_kl_ceip",
        "output_geotiff": rel_out,
        "crs": str(ref["crs"]),
        "width": w,
        "height": h,
        "pollutants": [str(x) for x in band_labels[: len(bands)]],
        "domain_bbox_wgs84": list(ref.get("domain_bbox_wgs84", ())),
        "ag_clc_codes": list(ag_codes),
        "ceip_group_order": list(group_order),
    }
    write_json(man, meta)
    return out_tif
