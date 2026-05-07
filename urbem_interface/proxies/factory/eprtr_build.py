from __future__ import annotations

import gc
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from urbem_interface.proxies.factory.eprtr_core import run_eprtr_vector_pipeline
from urbem_interface.proxies.factory.gdal_rasterize import rasterize_polygons_ones
from urbem_interface.proxies.factory.progress_util import tqdm_iter, tqdm_available

logger = logging.getLogger(__name__)

_RASTER_SNAPS = [
    ("Snap34", "snap34", "lu_snap34.tif"),
    ("Snap1", "snap1", "lu_snap1.tif"),
    ("Snap9", "snap9", "lu_waste.tif"),
]


def normalize_waste_tif_name(name: str | None) -> str:
    """Return basename like ``lu_waste.tif`` or ``lu_waste_france.tif``."""
    if name is None or not str(name).strip():
        return "lu_waste.tif"
    s = str(name).strip()
    return s if s.lower().endswith(".tif") else f"{s}.tif"


def _raster_snaps_for_waste(waste_tif: str) -> list[tuple[str, str, str]]:
    return [
        ("Snap34", "snap34", "lu_snap34.tif"),
        ("Snap1", "snap1", "lu_snap1.tif"),
        ("Snap9", "snap9", waste_tif),
    ]

_WASTE_CORINE_FALLBACK_ONLY = "lu_waste_corine_fallback_only.tif"
_WASTE_EPRTR_ONLY = "lu_waste_eprtr_only.tif"
_WASTE_EPRTR_PLUS_CORINE_NO_AG = "lu_waste_eprtr_plus_corine_no_ag.tif"

# CORINE vector Code_18 (Level 3) used in corine_fallback.snap9 -> cell values in
# corine_utm.tif (CLC 44-class nomenclature, same encoding as corine_classes.json).
_SNAP9_CODE18_TO_CLC44 = {
    "211": 12,
    "212": 13,
    "213": 14,
    "221": 15,
    "222": 16,
    "223": 17,
    "231": 18,
    "241": 19,
    "242": 20,
    "243": 21,
    "244": 22,
}


def _snap9_fallback_clc44_values(eprtr_snap_assignment: dict) -> frozenset[int]:
    fb = eprtr_snap_assignment.get("corine_fallback") or {}
    raw = fb.get("snap9")
    if not isinstance(raw, list):
        return frozenset()
    vals: set[int] = set()
    for c in raw:
        s = str(c).strip()
        if s in _SNAP9_CODE18_TO_CLC44:
            vals.add(_SNAP9_CODE18_TO_CLC44[s])
        else:
            logger.warning(
                "snap9 CORINE Code_18 %r has no CLC44 mapping; extend _SNAP9_CODE18_TO_CLC44 in eprtr_build.py",
                s,
            )
    return frozenset(vals)


def write_lu_waste_corine_fallback_only_raster(
    proxies_folder: Path,
    eprtr_snap_assignment: dict,
) -> Path:
    """
    Build lu_waste_corine_fallback_only.tif from corine_utm.tif only: pixels where
    CLC class (1-44) matches corine_fallback.snap9 Code_18 list. No FileGDB, no GeoPackage.
    """
    proxies_folder = Path(proxies_folder)
    corine_utm = proxies_folder / "corine_utm.tif"
    ref_raster = proxies_folder / "lu_industry.tif"
    out_path = proxies_folder / _WASTE_CORINE_FALLBACK_ONLY

    if not corine_utm.is_file():
        raise FileNotFoundError(
            f"{corine_utm.name} not found; run the CORINE raster phase first."
        )
    if not ref_raster.is_file():
        raise FileNotFoundError(
            f"{ref_raster.name} not found; run the CORINE raster phase first."
        )

    clc_vals = _snap9_fallback_clc44_values(eprtr_snap_assignment)
    if not clc_vals:
        raise ValueError("eprtr_snap_assignment['corine_fallback']['snap9'] missing or empty")

    clc_list = np.array(sorted(clc_vals), dtype=np.float64)

    with rasterio.open(ref_raster) as rref:
        prof = rref.profile.copy()
        prof.update(
            dtype="float64",
            nodata=None,
            compress="lzw",
            tiled=True,
            blockxsize=256,
            blockysize=256,
            BIGTIFF="IF_SAFER",
        )
        h, w = rref.height, rref.width

    with rasterio.open(corine_utm) as src:
        if (src.height, src.width) != (h, w):
            raise ValueError(
                f"{corine_utm.name} shape {(src.height, src.width)} != "
                f"{ref_raster.name} {(h, w)}"
            )

    with rasterio.open(corine_utm) as src, rasterio.open(out_path, "w", **prof) as dst:
        for _, window in src.block_windows(1):
            raw = src.read(1, window=window).astype(np.float64, copy=False)
            out = np.zeros_like(raw, dtype=np.float64)
            ri = np.rint(raw)
            out[np.isin(ri, clc_list)] = 1.0
            dst.write(out, 1, window=window)

    logger.info(
        "Wrote %s from %s (CLC classes %s); no GDB, no E-PRTR GeoPackage",
        out_path.name,
        corine_utm.name,
        sorted(clc_vals),
    )
    return out_path


def eprtr_phase_complete(
    proxies_folder: Path,
    *,
    waste_tif: str = "lu_waste.tif",
    snap9_only: bool = False,
) -> bool:
    p = Path(proxies_folder)
    waste_tif = normalize_waste_tif_name(waste_tif)
    if snap9_only:
        return (p / waste_tif).is_file()
    snaps = _raster_snaps_for_waste(waste_tif)
    return all((p / fn).is_file() for _, _, fn in snaps)


def build_eprtr_proxies_on_grid(
    *,
    job_paths: dict[str, Path],
    bbox: dict[str, float],
    crs_projected: str,
    crs_wgs: str,
    crs_corine: str,
    eprtr_snap_assignment: dict,
    keep_intermediate_shp: bool,
    proxies_folder: Path,
    skip_existing: bool = False,
    show_progress: bool = True,
    corine_fallback_only_waste: bool = False,
    eprtr_facilities_only_waste: bool = False,
    waste_eprtr_plus_corine_no_ag: bool = False,
    waste_output_tif: str | None = None,
    eprtr_snap9_only: bool = False,
) -> dict[str, Path]:
    """
    E-PRTR proxy rasters on the same grid as lu_industry.tif (built from CORINE).

    If ``corine_fallback_only_waste`` is True, write only ``lu_waste_corine_fallback_only.tif``
    from ``corine_utm.tif`` + ``corine_fallback.snap9`` (raster-only; no FileGDB read, no E-PRTR).

    If ``eprtr_facilities_only_waste`` is True, run the vector pipeline with E-PRTR joins but
    **without** ``corine_fallback``, and rasterize only ``snap9`` to ``lu_waste_eprtr_only.tif``
    (CORINE polygons that contain a Snap9-class E-PRTR point — sector 5 waste/wastewater in config).

    If ``waste_eprtr_plus_corine_no_ag`` is True, run the full ``corine_fallback`` for snap34/snap1
    but **omit** the agricultural ``snap9`` list; use only ``snap9_supplement`` from JSON if non-empty.
    Rasterize snap9 to ``lu_waste_eprtr_plus_corine_no_ag.tif`` (does not overwrite ``lu_waste.tif``).

    ``waste_output_tif`` sets the Snap9 raster basename (e.g. ``lu_waste_france.tif``); default
    ``lu_waste.tif``.

    If ``eprtr_snap9_only`` is True (normal E-PRTR build only), run the full vector pipeline but
    rasterize **only** the waste (Snap9) layer — skips ``lu_snap34.tif`` and ``lu_snap1.tif``.
    """
    proxies_folder = Path(proxies_folder)
    waste_tif = normalize_waste_tif_name(waste_output_tif)
    raster_snaps = _raster_snaps_for_waste(waste_tif)
    proxies_folder.mkdir(parents=True, exist_ok=True)

    ref_raster = proxies_folder / "lu_industry.tif"
    if not ref_raster.exists():
        raise FileNotFoundError(
            f"Reference raster not found: {ref_raster}. Run CORINE proxy build first."
        )

    if corine_fallback_only_waste:
        out_fb = proxies_folder / _WASTE_CORINE_FALLBACK_ONLY
        if skip_existing and out_fb.is_file():
            logger.info(
                "CORINE-fallback-only waste skipped (--skip-existing): %s",
                out_fb.name,
            )
            return {"waste_corine_fallback_only": out_fb}
        logger.info(
            "=== lu_waste CORINE fallback only: raster from %s (no GDB, no E-PRTR) ===",
            "corine_utm.tif",
        )
        out_path = write_lu_waste_corine_fallback_only_raster(
            proxies_folder, eprtr_snap_assignment
        )
        gc.collect()
        return {"waste_corine_fallback_only": out_path}

    if eprtr_facilities_only_waste:
        out_eo = proxies_folder / _WASTE_EPRTR_ONLY
        if skip_existing and out_eo.is_file():
            logger.info(
                "E-PRTR-facilities-only waste skipped (--skip-existing): %s",
                out_eo.name,
            )
            return {"waste_eprtr_only": out_eo}
        logger.info(
            "=== lu_waste from E-PRTR Snap9 hits only (no CORINE Code_18 fallback) ==="
        )
        if show_progress and not tqdm_available():
            logger.warning(
                "Install tqdm (pip install tqdm) for progress bars during long E-PRTR steps."
            )
        proxy_utm = run_eprtr_vector_pipeline(
            corine_gdb=job_paths["corine_gdb"],
            eprtr_gpkg=job_paths["eprtr_gpkg"],
            bbox=bbox,
            crs_projected=crs_projected,
            crs_wgs=crs_wgs,
            crs_corine=crs_corine,
            eprtr_snap_assignment=eprtr_snap_assignment,
            keep_intermediate_shp=keep_intermediate_shp,
            proxies_folder=proxies_folder,
            show_progress=show_progress,
            use_corine_fallback=False,
        )
        marked = proxy_utm.loc[proxy_utm["snap9"] == 1, ["geometry"]].copy()
        rasterize_polygons_ones(marked, ref_raster, out_eo)
        gc.collect()
        logger.info("Wrote %s (%d polygons)", out_eo.name, len(marked))
        return {"waste_eprtr_only": out_eo}

    if waste_eprtr_plus_corine_no_ag:
        out_pc = proxies_folder / _WASTE_EPRTR_PLUS_CORINE_NO_AG
        if skip_existing and out_pc.is_file():
            logger.info(
                "E-PRTR + CORINE (no ag snap9 fallback) waste skipped (--skip-existing): %s",
                out_pc.name,
            )
            return {"waste_eprtr_plus_corine_no_ag": out_pc}
        logger.info(
            "=== lu_waste: E-PRTR Snap9 + CORINE fallback without agricultural snap9 ==="
        )
        if show_progress and not tqdm_available():
            logger.warning(
                "Install tqdm (pip install tqdm) for progress bars during long E-PRTR steps."
            )
        proxy_utm = run_eprtr_vector_pipeline(
            corine_gdb=job_paths["corine_gdb"],
            eprtr_gpkg=job_paths["eprtr_gpkg"],
            bbox=bbox,
            crs_projected=crs_projected,
            crs_wgs=crs_wgs,
            crs_corine=crs_corine,
            eprtr_snap_assignment=eprtr_snap_assignment,
            keep_intermediate_shp=keep_intermediate_shp,
            proxies_folder=proxies_folder,
            show_progress=show_progress,
            use_corine_fallback=True,
            force_omit_snap9_agriculture_fallback=True,
        )
        marked = proxy_utm.loc[proxy_utm["snap9"] == 1, ["geometry"]].copy()
        rasterize_polygons_ones(marked, ref_raster, out_pc)
        gc.collect()
        logger.info("Wrote %s (%d polygons)", out_pc.name, len(marked))
        return {"waste_eprtr_plus_corine_no_ag": out_pc}

    if skip_existing and eprtr_phase_complete(
        proxies_folder, waste_tif=waste_tif, snap9_only=eprtr_snap9_only
    ):
        logger.info(
            "E-PRTR phase skipped (--skip-existing): %s",
            ", ".join(f for _, _, f in (raster_snaps if not eprtr_snap9_only else [raster_snaps[-1]])),
        )
        if eprtr_snap9_only:
            wname = waste_tif.replace("lu_", "").replace(".tif", "")
            return {wname: proxies_folder / waste_tif}
        return {
            fn.replace("lu_", "").replace(".tif", ""): proxies_folder / fn
            for _, _, fn in raster_snaps
        }

    logger.info("=== Building E-PRTR proxies ===")
    if eprtr_snap9_only:
        logger.info(
            "Raster outputs: %s only (skipping lu_snap34.tif, lu_snap1.tif)",
            waste_tif,
        )

    if show_progress and not tqdm_available():
        logger.warning("Install tqdm (pip install tqdm) for progress bars during long E-PRTR steps.")

    proxy_utm = run_eprtr_vector_pipeline(
        corine_gdb=job_paths["corine_gdb"],
        eprtr_gpkg=job_paths["eprtr_gpkg"],
        bbox=bbox,
        crs_projected=crs_projected,
        crs_wgs=crs_wgs,
        crs_corine=crs_corine,
        eprtr_snap_assignment=eprtr_snap_assignment,
        keep_intermediate_shp=keep_intermediate_shp,
        proxies_folder=proxies_folder,
        show_progress=show_progress,
    )

    written: dict[str, Path] = {}
    jobs: list[tuple[str, Path, Any]] = []
    snap_rows = [raster_snaps[-1]] if eprtr_snap9_only else raster_snaps
    for _snap_flag_col, snap_out_col, filename in snap_rows:
        proxy_name = filename.replace("lu_", "").replace(".tif", "")
        out_path = proxies_folder / filename
        marked = proxy_utm.loc[proxy_utm[snap_out_col] == 1, ["geometry"]].copy()
        jobs.append((proxy_name, out_path, marked))

    n_threads = int(os.environ.get("URBLEM_EPRTR_RASTER_THREADS", "3"))
    n_threads = max(1, min(n_threads, len(jobs)))
    use_pbar = show_progress and tqdm_available()
    desc = "gdal_rasterize (E-PRTR masks)"

    if n_threads <= 1:
        seq = tqdm_iter(jobs, desc=desc, total=len(jobs), unit="tif", enabled=use_pbar)
        for proxy_name, out_path, marked in seq:
            rasterize_polygons_ones(marked, ref_raster, out_path)
            written[proxy_name] = out_path
    else:
        futures: dict = {}
        with ThreadPoolExecutor(max_workers=n_threads) as ex:
            for proxy_name, out_path, marked in jobs:
                fut = ex.submit(rasterize_polygons_ones, marked, ref_raster, out_path)
                futures[fut] = (proxy_name, out_path)
            done = as_completed(futures)
            if use_pbar:
                from tqdm.auto import tqdm

                done = tqdm(done, total=len(futures), desc=desc, unit="tif")
            for fut in done:
                fut.result()
                proxy_name, out_path = futures[fut]
                written[proxy_name] = out_path

    gc.collect()
    logger.info("=== E-PRTR proxies complete: %d rasters ===", len(written))
    return written
