"""
E-PRTR + CORINE vector pipeline (vendored; no dependency on urbem_v2).

Rasterization is handled separately (gdal_rasterize) in gdal_rasterize.py.
"""

from __future__ import annotations

import gc
import logging
import multiprocessing as mp
import os
import re
import shutil
import tempfile
import time
import traceback
import warnings
from pathlib import Path

import fiona
import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from urbem_interface.proxies.factory.progress_util import (
    run_with_pulse_progress,
    tqdm_available,
    tqdm_iter,
)

logger = logging.getLogger(__name__)

def _ogr_organize_polygons_skip() -> None:
    """Avoid slow organizePolygons() on large multiparts (must run before opening the dataset)."""
    os.environ["OGR_ORGANIZE_POLYGONS_METHOD"] = "SKIP"
    try:
        from osgeo import gdal

        gdal.SetConfigOption("OGR_ORGANIZE_POLYGONS_METHOD", "SKIP")
    except Exception:
        pass


def _read_corine_gdf_bulk(corine_gdb: Path, read_kw: dict) -> gpd.GeoDataFrame:
    """
    Bulk read (fast). Per-feature Fiona iteration is orders of magnitude slower on multi-million layers.
    Prefer pyogrio when installed.
    """
    path = str(corine_gdb)
    try:
        import pyogrio

        pyogrio.set_gdal_config_options({"OGR_ORGANIZE_POLYGONS_METHOD": "SKIP"})
    except Exception:
        pass
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*organizePolygons\(\) received a polygon.*",
            category=RuntimeWarning,
        )
        for engine in ("pyogrio", "fiona"):
            try:
                return gpd.read_file(path, engine=engine, **read_kw)
            except Exception as exc:
                logger.debug("CORINE read engine %r failed: %s", engine, exc)
        logger.info("CORINE GDB: pyogrio/fiona bulk read failed; using GeoPandas default engine.")
        return gpd.read_file(path, **read_kw)


def _ogr_bbox_feature_count(path: Path, layer: str, bbox: tuple) -> int | None:
    """
    Fast feature count for layer + bbox (GetFeatureCount(force=False)).
    No ETA for the actual read, but gives scale (e.g. ~2.4M polys in filter).
    """
    try:
        from osgeo import gdal, ogr

        gdal.UseExceptions()
        gdal.SetConfigOption("OGR_ORGANIZE_POLYGONS_METHOD", "SKIP")
        ds = ogr.Open(str(path), 0)
        if ds is None:
            return None
        lyr = ds.GetLayerByName(layer) if layer else ds.GetLayer(0)
        if lyr is None:
            return None
        minx, miny, maxx, maxy = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        lyr.SetSpatialFilterRect(minx, miny, maxx, maxy)
        n = lyr.GetFeatureCount(False)
        del lyr, ds
        if n is None or n < 0:
            return None
        return int(n)
    except Exception:
        logger.debug("OGR bbox feature count failed", exc_info=True)
        return None


def _mp_corine_read_worker(result_q, path_str: str, read_kw: dict) -> None:
    """
    Runs in a child process so the parent GIL stays free for tqdm / logging.

    Sends result via a temp GeoParquet file when possible. Queue-pickling a
    multi-million-row GeoDataFrame can take 30+ minutes and huge RAM; Parquet
    avoids that path.
    """
    import tempfile

    tmp_parquet: str | None = None
    try:
        gdf = _read_corine_gdf_bulk(Path(path_str), read_kw)
        try:
            fd, tmp_parquet = tempfile.mkstemp(suffix=".parquet", prefix="urbem_corine_read_")
            os.close(fd)
            gdf.to_parquet(tmp_parquet, index=False)
            path_sent = tmp_parquet
            tmp_parquet = None
            del gdf
            gc.collect()
            result_q.put(("parquet", path_sent))
        except Exception:
            if tmp_parquet:
                try:
                    os.unlink(tmp_parquet)
                except OSError:
                    pass
                tmp_parquet = None
            result_q.put(("ok", gdf))
    except Exception:
        if tmp_parquet:
            try:
                os.unlink(tmp_parquet)
            except OSError:
                pass
        result_q.put(("err", traceback.format_exc()))


def _read_corine_gdf_bulk_subprocess_progress(
    corine_gdb: Path,
    read_kw: dict,
    *,
    desc: str,
) -> gpd.GeoDataFrame:
    """
    Run bulk GDB read in a spawn child process. Parent shows tqdm + INFO every 10s wall time
    (reliable on Windows where pyogrio can hold the GIL in the worker thread for minutes).
    """
    import sys

    from tqdm.auto import tqdm

    layer = read_kw.get("layer") or ""
    bbox = read_kw.get("bbox")
    n_hint: int | None = None
    hint_scope = ""
    if isinstance(bbox, (tuple, list)) and len(bbox) == 4:
        n_hint = _ogr_bbox_feature_count(corine_gdb, str(layer), tuple(bbox))
        if n_hint is not None:
            hint_scope = "bbox"
    if n_hint is None and layer:
        try:
            import pyogrio

            info = pyogrio.read_info(str(corine_gdb), layer=str(layer))
            nf = info.get("features")
            if nf is not None and int(nf) >= 0:
                n_hint = int(nf)
                hint_scope = "full layer"
        except Exception:
            pass
    if n_hint is not None and hint_scope == "bbox":
        logger.info(
            "CORINE GDB: OGR counts ~%s features in bbox (no per-row ETA; bar is wall time only)",
            f"{n_hint:,}",
        )
    elif n_hint is not None:
        logger.info(
            "CORINE GDB: ~%s features in layer (%s; bbox may be smaller; no per-row ETA)",
            f"{n_hint:,}",
            hint_scope,
        )
    else:
        logger.info(
            "CORINE GDB: feature count unavailable; bar shows seconds waited (no ETA)"
        )

    ctx = mp.get_context("spawn")
    q = ctx.Queue(maxsize=1)
    path_str = str(corine_gdb.resolve())
    proc = ctx.Process(target=_mp_corine_read_worker, args=(q, path_str, read_kw))
    t0 = time.perf_counter()
    last_log = t0
    proc.start()
    scale = (
        f"~{n_hint:,} feats ({hint_scope})"
        if n_hint is not None and hint_scope
        else (f"~{n_hint:,} feats" if n_hint is not None else "feat count n/a")
    )
    bar_fmt = "{desc} | {n_fmt}s waited | " + scale + " | ETA unknown | {elapsed}"
    try:
        with tqdm(
            total=None,
            desc=desc,
            unit="s",
            mininterval=0,
            file=sys.stderr,
            dynamic_ncols=True,
            bar_format=bar_fmt,
        ) as pbar:
            while proc.is_alive():
                proc.join(timeout=1.0)
                if proc.is_alive():
                    pbar.update(1)
                    now = time.perf_counter()
                    if now - last_log >= 10.0:
                        print(file=sys.stderr, flush=True)
                        logger.info(
                            "CORINE GDB bulk read (subprocess) still running: %.0f s wall (ETA unknown)",
                            now - t0,
                        )
                        last_log = now
                    pbar.refresh()
                    sys.stderr.flush()
        proc.join(timeout=30)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=15)
            raise RuntimeError("CORINE read subprocess did not finish in time")
        ec = proc.exitcode
        if ec is not None and ec != 0:
            err_tail = ""
            try:
                kind, payload = q.get_nowait()
                if kind == "err":
                    err_tail = "\n" + str(payload)
            except Exception:
                pass
            raise RuntimeError(f"CORINE read subprocess failed (exit {ec}){err_tail}")
        logger.info("CORINE GDB: subprocess read finished; loading result...")
        kind, payload = q.get(timeout=7200)
        if kind == "err":
            raise RuntimeError(f"CORINE read in subprocess failed:\n{payload}")
        if kind == "parquet":
            pth = str(payload)
            try:
                logger.info(
                    "CORINE GDB: reading temp Parquet (faster than queue pickle for large layers): %s",
                    pth,
                )
                return gpd.read_parquet(pth)
            finally:
                try:
                    os.unlink(pth)
                except OSError:
                    pass
        if kind == "ok":
            logger.warning(
                "CORINE GDB: received in-memory GeoDataFrame via queue (slow for huge layers); "
                "install pyarrow if Parquet handoff failed."
            )
            return payload
        raise RuntimeError(f"CORINE subprocess: unexpected queue payload kind {kind!r}")
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=15)


_CORINE_POST_CHUNK = 80_000


def _corine_to_crs_chunk_size() -> int:
    """
    Rows per batch for to_crs on CORINE. Complex polygons have many vertices; 80k rows
    can exceed available RAM during coordinate transform. Override with URBLEM_CORINE_TO_CRS_CHUNK.
    """
    raw = os.environ.get("URBLEM_CORINE_TO_CRS_CHUNK", "").strip()
    default = 5_000
    if not raw:
        return default
    try:
        n = int(raw)
    except ValueError:
        logger.warning(
            "URBLEM_CORINE_TO_CRS_CHUNK=%r invalid; using default %s",
            raw,
            default,
        )
        return default
    if n < 1:
        return default
    return n


def _geoparquet_spill_available() -> bool:
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        return False
    return True


def _use_disk_spill_for_tocrs() -> bool:
    v = os.environ.get("URBLEM_TO_CRS_DISK_SPILL", "1").strip().lower()
    return v not in ("0", "false", "no")


def _spill_gdf_row_chunks_to_parquet(
    gdf: gpd.GeoDataFrame,
    out_dir: Path,
    chunk_size: int,
    *,
    desc: str,
    show_progress: bool,
) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(gdf)
    nchunk = (n + chunk_size - 1) // chunk_size
    use_bar = show_progress and tqdm_available()
    paths: list[Path] = []
    for i, start in enumerate(
        tqdm_iter(
            range(0, n, chunk_size),
            desc=desc,
            total=nchunk,
            unit="chunk",
            enabled=use_bar,
        )
    ):
        sub = gdf.iloc[start : start + chunk_size]
        p = out_dir / f"in_{i:06d}.parquet"
        sub.to_parquet(p, index=False)
        paths.append(p)
        del sub
        gc.collect()
    return paths


def _reproject_parquet_row_chunks(
    paths: list[Path],
    target_crs: str,
    *,
    desc: str,
    show_progress: bool,
) -> gpd.GeoDataFrame:
    merged: gpd.GeoDataFrame | None = None
    use_bar = show_progress and tqdm_available()
    seq = paths
    if use_bar:
        from tqdm.auto import tqdm

        seq = tqdm(paths, desc=desc, total=len(paths), unit="chunk", dynamic_ncols=True)
    for pin in seq:
        pin = Path(pin)
        sub = gpd.read_parquet(pin)
        try:
            pin.unlink()
        except OSError:
            pass
        part = sub.to_crs(target_crs)
        del sub
        gc.collect()
        if merged is None:
            merged = part
        else:
            merged = pd.concat([merged, part], ignore_index=True)
            del part
        gc.collect()
    if merged is None:
        return gpd.GeoDataFrame(geometry=[], crs=target_crs)
    return merged


def _batched_clip_corine(
    gdf: gpd.GeoDataFrame,
    domain: gpd.GeoDataFrame,
    *,
    show_progress: bool,
    chunk_size: int = _CORINE_POST_CHUNK,
) -> gpd.GeoDataFrame:
    n = len(gdf)
    if n == 0:
        return gdf
    geom_col = gdf.geometry.name
    if not show_progress or not tqdm_available() or n <= chunk_size:
        return gpd.clip(gdf, domain)
    parts: list[gpd.GeoDataFrame] = []
    nchunk = (n + chunk_size - 1) // chunk_size
    for start in tqdm_iter(
        range(0, n, chunk_size),
        desc="CORINE: clip to domain",
        total=nchunk,
        unit="chunk",
        enabled=True,
    ):
        sub = gdf.iloc[start : start + chunk_size]
        parts.append(gpd.clip(sub, domain))
    merged = pd.concat(parts, ignore_index=True)
    return gpd.GeoDataFrame(merged, geometry=geom_col, crs=gdf.crs)


def _batched_to_crs_gdf(
    gdf: gpd.GeoDataFrame,
    target_crs: str,
    *,
    show_progress: bool,
    desc: str,
    chunk_size: int = _CORINE_POST_CHUNK,
) -> gpd.GeoDataFrame:
    n = len(gdf)
    if n == 0:
        return gdf
    geom_col = gdf.geometry.name
    if n <= chunk_size:
        return gdf.to_crs(target_crs)
    parts: list[gpd.GeoDataFrame] = []
    nchunk = (n + chunk_size - 1) // chunk_size
    use_bar = show_progress and tqdm_available()
    for start in tqdm_iter(
        range(0, n, chunk_size),
        desc=desc,
        total=nchunk,
        unit="chunk",
        enabled=use_bar,
    ):
        sub = gdf.iloc[start : start + chunk_size]
        parts.append(sub.to_crs(target_crs))
        gc.collect()
    merged = pd.concat(parts, ignore_index=True)
    return gpd.GeoDataFrame(merged, geometry=geom_col, crs=target_crs)


def parse_annex_i_activity(s) -> str:
    if pd.isna(s) or not str(s).strip():
        return "0 -z"
    s = str(s).strip()
    if "," in s:
        s = s.split(",")[0].strip()
    sector = "0"
    subsector = "z"
    m = re.search(r"(\d+)\s*[.\-\s(]*(\d+)", s)
    if m:
        sector = m.group(1)[0]
        num = int(m.group(2))
        subsector = chr(ord("a") + num - 1) if 1 <= num <= 26 else "z"
    else:
        digits = re.findall(r"\d", s)
        if digits:
            sector = digits[0]
            if len(digits) >= 2 and 1 <= int(digits[1]) <= 9:
                subsector = chr(ord("a") + int(digits[1]) - 1)
        if subsector == "z" and any(c.isalpha() for c in s):
            for c in s:
                if c.isalpha():
                    subsector = c.lower()
                    break
    return sector + " -" + subsector


def build_corine_polygons(
    corine_gdb: Path,
    crs_corine: str,
    crs_wgs: str,
    domain_corine: gpd.GeoDataFrame,
    *,
    show_progress: bool = False,
) -> gpd.GeoDataFrame:
    corine_gdb = Path(corine_gdb)
    _ogr_organize_polygons_skip()

    layers = fiona.listlayers(str(corine_gdb))
    logger.info("Reading CORINE GDB layer '%s' (bulk read)...", layers[0])

    bbox_corine = domain_corine.total_bounds
    bbox_tuple = (
        bbox_corine[0] - 100,
        bbox_corine[1] - 100,
        bbox_corine[2] + 100,
        bbox_corine[3] + 100,
    )

    read_kw = {"layer": layers[0], "bbox": bbox_tuple}
    t0 = time.perf_counter()
    use_progress = show_progress and tqdm_available()
    no_mp = os.environ.get("URBLEM_CORINE_READ_NO_MP", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if use_progress and not no_mp:
        try:
            corine_polygons = _read_corine_gdf_bulk_subprocess_progress(
                corine_gdb,
                read_kw,
                desc="CORINE GDB (bulk read, subprocess)",
            )
        except Exception as exc:
            logger.warning(
                "CORINE subprocess read failed (%s); falling back to in-process read.",
                exc,
            )
            corine_polygons = run_with_pulse_progress(
                lambda: _read_corine_gdf_bulk(corine_gdb, read_kw),
                desc="CORINE GDB (bulk read)",
                enabled=True,
            )
    elif use_progress:
        corine_polygons = run_with_pulse_progress(
            lambda: _read_corine_gdf_bulk(corine_gdb, read_kw),
            desc="CORINE GDB (bulk read)",
            enabled=True,
        )
    else:
        corine_polygons = _read_corine_gdf_bulk(corine_gdb, read_kw)
    logger.info(
        "CORINE bulk read: %d raw features in %.1f s",
        len(corine_polygons),
        time.perf_counter() - t0,
    )

    en = show_progress and tqdm_available()
    nfeat = len(corine_polygons)
    if en and nfeat > 0 and nfeat <= _CORINE_POST_CHUNK:
        from tqdm.auto import tqdm

        with tqdm(total=3, desc="CORINE after read", unit="step", dynamic_ncols=True) as pbar:
            pbar.set_postfix_str("clip to domain")
            corine_poly_croped = gpd.clip(corine_polygons, domain_corine)
            pbar.update(1)
            pbar.set_postfix_str("set CRS")
            corine_poly_croped = corine_poly_croped.set_crs(crs_corine, allow_override=True)
            pbar.update(1)
            pbar.set_postfix_str("to WGS84")
            corine_poly_wgs = corine_poly_croped.to_crs(crs_wgs)
            pbar.update(1)
    else:
        corine_poly_croped = _batched_clip_corine(
            corine_polygons, domain_corine, show_progress=en
        )
        del corine_polygons
        gc.collect()
        corine_poly_croped = corine_poly_croped.set_crs(crs_corine, allow_override=True)
        chunk_sz = _corine_to_crs_chunk_size()
        nclip = len(corine_poly_croped)
        if (
            _use_disk_spill_for_tocrs()
            and _geoparquet_spill_available()
            and nclip > chunk_sz
        ):
            tmpdir = tempfile.mkdtemp(prefix="urbem_corine_tocrs_")
            try:
                paths = _spill_gdf_row_chunks_to_parquet(
                    corine_poly_croped,
                    Path(tmpdir),
                    chunk_sz,
                    desc="CORINE: reproject to WGS84 (write)",
                    show_progress=en,
                )
                del corine_poly_croped
                gc.collect()
                corine_poly_wgs = _reproject_parquet_row_chunks(
                    paths,
                    crs_wgs,
                    desc="CORINE: reproject to WGS84",
                    show_progress=en,
                )
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)
        else:
            if nclip > chunk_sz and not _geoparquet_spill_available():
                logger.warning(
                    "Install pyarrow for disk-spill reproject (URBLEM_TO_CRS_DISK_SPILL=1); "
                    "otherwise large CORINE reprojects keep the full frame and all chunks in RAM."
                )
            corine_poly_wgs = _batched_to_crs_gdf(
                corine_poly_croped,
                crs_wgs,
                show_progress=en,
                desc="CORINE: reproject to WGS84",
                chunk_size=chunk_sz,
            )

    logger.info(
        "CORINE polygons loaded: %d features, CRS=%s",
        len(corine_poly_wgs),
        corine_poly_wgs.crs,
    )
    return corine_poly_wgs


def load_eprtr(eprtr_gpkg: Path, *, show_progress: bool = False) -> gpd.GeoDataFrame:
    eprtr_gpkg = Path(eprtr_gpkg)
    logger.info("Reading E-PRTR GeoPackage: %s", eprtr_gpkg)

    path = str(eprtr_gpkg)
    eprtr_gdf = None
    for engine in ("pyogrio", "fiona"):
        try:
            eprtr_gdf = gpd.read_file(path, engine=engine)
            break
        except Exception as exc:
            logger.debug("E-PRTR read engine %r failed: %s", engine, exc)
    if eprtr_gdf is None:
        eprtr_gdf = gpd.read_file(path)
    eprtr_gdf["geometry_layer"] = eprtr_gdf["eprtr_AnnexIActivity"].map(parse_annex_i_activity)
    eprtr_gdf["geometry_layer"] = eprtr_gdf["geometry_layer"].astype(str)
    eprtr_gdf["Sector"] = eprtr_gdf["geometry_layer"].str.get(0)
    eprtr_gdf["SubSector"] = eprtr_gdf["geometry_layer"].str.get(3)
    eprtr_gdf["Sector"] = eprtr_gdf["Sector"].astype(str)
    eprtr_gdf["SubSector"] = eprtr_gdf["SubSector"].astype(str)

    for col in ("Snap1", "Snap6", "Snap9", "Snap34", "Snap10"):
        eprtr_gdf[col] = 0

    logger.info("E-PRTR loaded: %d sites, CRS=%s", len(eprtr_gdf), eprtr_gdf.crs)
    return eprtr_gdf


def assign_snap_flags(eprtr_gdf: gpd.GeoDataFrame, flags_cfg: dict) -> gpd.GeoDataFrame:
    for snap_col, rules in flags_cfg.items():
        if snap_col.startswith("_") or not isinstance(rules, list):
            continue
        for rule in rules:
            sector = str(rule["sector"])
            subsector = str(rule["subsector"])
            mask = (eprtr_gdf["Sector"] == sector) & (eprtr_gdf["SubSector"] == subsector)
            eprtr_gdf.loc[mask, snap_col] = 1

    logger.info(
        "SNAP flag counts — Snap1: %d  Snap34: %d  Snap9: %d  Snap6: %d  Snap10: %d",
        (eprtr_gdf["Snap1"] == 1).sum(),
        (eprtr_gdf["Snap34"] == 1).sum(),
        (eprtr_gdf["Snap9"] == 1).sum(),
        (eprtr_gdf["Snap6"] == 1).sum(),
        (eprtr_gdf["Snap10"] == 1).sum(),
    )
    return eprtr_gdf


def assign_snap_to_corine(
    corine_poly: gpd.GeoDataFrame,
    eprtr_gdf: gpd.GeoDataFrame,
    snap_flag_col: str,
    snap_out_col: str,
    domain_wgs: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    flagged = eprtr_gdf.loc[eprtr_gdf[snap_flag_col] == 1].copy()
    flagged = flagged.dropna(subset=["geometry"])
    point_gdf = gpd.GeoDataFrame(flagged, geometry="geometry")
    point_gdf = point_gdf.set_crs(eprtr_gdf.crs, allow_override=True).to_crs(domain_wgs.crs)

    corine_poly[snap_out_col] = 0

    if len(point_gdf) == 0:
        logger.info("  %s: no E-PRTR points in dataset — all zeros", snap_flag_col)
        return corine_poly

    try:
        clipped = gpd.clip(point_gdf, domain_wgs)
        if len(clipped) == 0:
            logger.info("  %s: no E-PRTR points in domain — all zeros", snap_flag_col)
            return corine_poly

        # Points as left + predicate "within" uses the point index and polygon sindex;
        # often faster than millions of polygons as left with "contains".
        joined = gpd.sjoin(clipped, corine_poly, how="inner", predicate="within")
        if len(joined) == 0:
            logger.info("  %s: spatial join returned no hits — all zeros", snap_flag_col)
            return corine_poly
        matched = joined["index_right"].unique()
        corine_poly.loc[matched, snap_out_col] = 1
        logger.info(
            "  %s: %d E-PRTR points → %d CORINE polygons marked",
            snap_flag_col,
            len(clipped),
            len(matched),
        )
    except Exception as exc:
        logger.warning("  %s: spatial join failed (%s) — using all-zero proxy", snap_flag_col, exc)

    return corine_poly


def apply_corine_fallback(proxy_gdf: gpd.GeoDataFrame, fallback_cfg: dict) -> gpd.GeoDataFrame:
    for snap_col, clc_codes in fallback_cfg.items():
        if snap_col.startswith("_") or not isinstance(clc_codes, list):
            continue
        if snap_col not in proxy_gdf.columns:
            proxy_gdf[snap_col] = 0

        code_col = "Code_18"
        if code_col not in proxy_gdf.columns:
            logger.warning("CORINE fallback: '%s' column not found — skipping", code_col)
            continue

        mask = proxy_gdf[code_col].astype(str).isin(clc_codes)
        proxy_gdf.loc[mask, snap_col] = 1
        logger.info(
            "  CORINE fallback %s: %d polygons filled from Code_18 codes %s",
            snap_col,
            int(mask.sum()),
            clc_codes,
        )

    return proxy_gdf


def domain_frames_from_bbox(
    bbox: dict[str, float],
    crs_projected: str,
    crs_wgs: str,
    crs_corine: str,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    utm_box = box(bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"])
    domain_utm = gpd.GeoDataFrame(geometry=[utm_box], crs=crs_projected)
    domain_wgs = domain_utm.to_crs(crs_wgs)
    domain_corine = domain_utm.to_crs(crs_corine)
    return domain_utm, domain_wgs, domain_corine


def run_eprtr_vector_pipeline(
    *,
    corine_gdb: Path,
    eprtr_gpkg: Path,
    bbox: dict[str, float],
    crs_projected: str,
    crs_wgs: str,
    crs_corine: str,
    eprtr_snap_assignment: dict,
    keep_intermediate_shp: bool,
    proxies_folder: Path,
    show_progress: bool = False,
    use_corine_fallback: bool = True,
    force_omit_snap9_agriculture_fallback: bool = False,
) -> gpd.GeoDataFrame:
    """
    Returns CORINE polygons in projected CRS with snap34, snap1, snap9 columns set.

    If ``use_corine_fallback`` is False, skip ``corine_fallback`` Code_18 fills — snap
    columns reflect only E-PRTR point→polygon joins (whole CORINE polygons that contain
    a flagged site).

    If ``corine_fallback.omit_snap9_agriculture_fallback`` is true (or
    ``force_omit_snap9_agriculture_fallback``), do not apply the usual ``snap9`` list
    (agriculture Code_18); optionally apply only ``snap9_supplement`` from JSON.
    """
    domain_utm, domain_wgs, domain_corine = domain_frames_from_bbox(
        bbox, crs_projected, crs_wgs, crs_corine
    )

    corine_poly = build_corine_polygons(
        corine_gdb, crs_corine, crs_wgs, domain_corine, show_progress=show_progress
    )
    corine_poly.index = range(len(corine_poly))

    eprtr_gdf = load_eprtr(eprtr_gpkg, show_progress=show_progress)
    flags_cfg = {
        k: v
        for k, v in eprtr_snap_assignment.items()
        if not k.startswith("_") and k != "corine_fallback"
    }
    eprtr_gdf = assign_snap_flags(eprtr_gdf, flags_cfg)

    raster_snaps = [
        ("Snap34", "snap34"),
        ("Snap1", "snap1"),
        ("Snap9", "snap9"),
    ]
    for snap_flag_col, snap_out_col in tqdm_iter(
        raster_snaps,
        desc="E-PRTR spatial joins",
        total=len(raster_snaps),
        unit="step",
        enabled=show_progress and tqdm_available(),
    ):
        corine_poly = assign_snap_to_corine(
            corine_poly, eprtr_gdf, snap_flag_col, snap_out_col, domain_wgs
        )
        gc.collect()

    if use_corine_fallback:
        raw_fb = eprtr_snap_assignment.get("corine_fallback", {})
        omit_snap9_ag = bool(raw_fb.get("omit_snap9_agriculture_fallback", False)) or bool(
            force_omit_snap9_agriculture_fallback
        )
        fallback_cfg: dict = {}
        for k, v in raw_fb.items():
            if k.startswith("_") or k in ("omit_snap9_agriculture_fallback", "snap9_supplement"):
                continue
            if isinstance(v, list):
                fallback_cfg[k] = v
        if omit_snap9_ag:
            fallback_cfg.pop("snap9", None)
            sup = raw_fb.get("snap9_supplement")
            if isinstance(sup, list) and sup:
                fallback_cfg["snap9"] = [str(x).strip() for x in sup]
            logger.info(
                "CORINE fallback: omit_snap9_agriculture_fallback=true "
                "(snap9 from E-PRTR only%s)",
                f" + snap9_supplement {fallback_cfg.get('snap9', [])}" if fallback_cfg.get("snap9") else "",
            )
        corine_poly = apply_corine_fallback(corine_poly, fallback_cfg)
    else:
        logger.info("CORINE Code_18 fallback skipped (E-PRTR facility hits only)")

    proxies_folder = Path(proxies_folder)
    if keep_intermediate_shp:
        corine_poly.to_file(str(proxies_folder / "corine_poly_final.shp"), driver="ESRI Shapefile")
        logger.info("Saved: corine_poly_final.shp")
        eprtr_snap9 = gpd.clip(
            eprtr_gdf.loc[eprtr_gdf["Snap9"] == 1]
            .dropna(subset=["geometry"])
            .set_crs(eprtr_gdf.crs, allow_override=True)
            .to_crs(crs_wgs),
            domain_wgs,
        )
        eprtr_snap9.to_file(str(proxies_folder / "eprtr_croped_snap9.shp"), driver="ESRI Shapefile")

    chunk_sz = _corine_to_crs_chunk_size()
    sp_bar = show_progress and tqdm_available()
    if (
        _use_disk_spill_for_tocrs()
        and _geoparquet_spill_available()
        and len(corine_poly) > chunk_sz
    ):
        tmpdir = tempfile.mkdtemp(prefix="urbem_eprtr_tocrs_")
        try:
            paths = _spill_gdf_row_chunks_to_parquet(
                corine_poly,
                Path(tmpdir),
                chunk_sz,
                desc="E-PRTR: CORINE to grid (write)",
                show_progress=sp_bar,
            )
            del corine_poly
            gc.collect()
            proxy_utm = _reproject_parquet_row_chunks(
                paths,
                crs_projected,
                desc="E-PRTR: CORINE to projected CRS",
                show_progress=sp_bar,
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
    else:
        if len(corine_poly) > chunk_sz and not _geoparquet_spill_available():
            logger.warning(
                "Install pyarrow for disk-spill reproject to the output grid; "
                "reduces peak RAM before rasterize."
            )
        proxy_utm = _batched_to_crs_gdf(
            corine_poly,
            crs_projected,
            show_progress=sp_bar,
            desc="E-PRTR: CORINE to projected CRS",
            chunk_size=chunk_sz,
        )
    if keep_intermediate_shp:
        proxy_utm.to_file(str(proxies_folder / "eprtr_df_proxy.shp"), driver="ESRI Shapefile")
        logger.info("Saved: eprtr_df_proxy.shp")

    gc.collect()
    return proxy_utm
