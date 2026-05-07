"""
Statistics for emission output - KPIs, CAMS comparison, spatial stats, charts data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

POLLUTANTS = ["NOx", "NMVOC", "CO", "SO2", "NH3", "PM2.5", "PM10"]
NA_VAL = -999.0


def _clean_df(df: pd.DataFrame, source_type: str) -> pd.DataFrame:
    """Drop invalid rows and normalize pollutant columns."""
    df = df.copy()
    if source_type == "area":
        for c in ["xcor_sw", "ycor_sw", "xcor_ne", "ycor_ne"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["xcor_sw", "ycor_sw", "xcor_ne", "ycor_ne"])
        df = df[(df["xcor_sw"] > NA_VAL) & (df["ycor_sw"] > NA_VAL)]
    elif source_type == "line":
        for c in ["xcor_start", "ycor_start", "xcor_end", "ycor_end"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["xcor_start", "ycor_start", "xcor_end", "ycor_end"])
        df = df[(df["xcor_start"] > NA_VAL) & (df["ycor_start"] > NA_VAL)]
    for p in POLLUTANTS:
        if p in df.columns:
            df[p] = pd.to_numeric(df[p], errors="coerce").fillna(0)
            df.loc[df[p] < 0, p] = 0
    return df


def _compute_interior_data(
    intermediates: Path,
    domain_nrow: int,
    domain_ncol: int,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
) -> tuple[np.ndarray, np.ndarray, dict[int, int]] | None:
    """
    Compute interior mask, cams_domain_ids, and n_cells_per_coarse.
    Returns (interior, cams_domain_ids, n_cells_per_coarse) or None.
    """
    try:
        from rasterio.crs import CRS
        from rasterio.warp import reproject, Resampling as WarpResampling
        from affine import Affine

        meta_path = intermediates / "step2_coarse_grid" / "cams_origin_metadata.csv"
        if not meta_path.exists():
            return None
        meta = pd.read_csv(meta_path).iloc[0]
        nrow_c = int(meta["nrow"])
        ncol_c = int(meta["ncol"])
        res_x = float(meta["res_x"])
        res_y = float(meta["res_y"])
        left = float(meta["xmin"])
        top = float(meta["ymax"])

        cams_transform = Affine(res_x, 0.0, left, 0.0, -res_y, top)
        cams_origin = np.arange(1, nrow_c * ncol_c + 1, dtype=np.float64).reshape(
            (nrow_c, ncol_c), order="F"
        )

        cell_w = (xmax - xmin) / domain_ncol
        cell_h = (ymax - ymin) / domain_nrow
        domain_transform = Affine(cell_w, 0.0, xmin, 0.0, -cell_h, ymax)
        domain_shape = (domain_nrow, domain_ncol)

        crs = CRS.from_string(str(meta.get("crs", "EPSG:32634")))
        cams_domain_ids = np.full(domain_shape, np.nan, dtype=np.float64)
        reproject(
            source=cams_origin,
            destination=cams_domain_ids,
            src_transform=cams_transform,
            src_crs=crs,
            dst_transform=domain_transform,
            dst_crs=crs,
            resampling=WarpResampling.nearest,
        )

        flat_ids = cams_domain_ids.ravel()
        valid = np.isfinite(flat_ids) & (flat_ids > 0)
        if not np.any(valid):
            return None

        ids_valid = flat_ids[valid]
        unique_ids, labels = np.unique(ids_valid.astype(int), return_inverse=True)
        max_cell = (res_x / cell_w) * (res_y / cell_h)
        n_cells = np.bincount(labels, minlength=len(unique_ids)).astype(np.float64)
        coverage_ratio = np.minimum(n_cells / max_cell, 1.0)
        cell_id_to_ratio = {int(uid): float(ratio) for uid, ratio in zip(unique_ids, coverage_ratio)}
        n_cells_per_coarse = {int(uid): int(n_cells[i]) for i, uid in enumerate(unique_ids)}

        interior = np.zeros(domain_shape, dtype=bool)
        for r in range(domain_nrow):
            for c in range(domain_ncol):
                cid = int(cams_domain_ids[r, c])
                if cid > 0 and cid in cell_id_to_ratio and cell_id_to_ratio[cid] >= 0.8:
                    interior[r, c] = True

        return interior, cams_domain_ids, n_cells_per_coarse
    except Exception:
        return None


def _compute_interior_mask(
    intermediates: Path,
    domain_nrow: int,
    domain_ncol: int,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
) -> np.ndarray | None:
    """Interior mask only. See _compute_interior_data for full data."""
    data = _compute_interior_data(intermediates, domain_nrow, domain_ncol, xmin, ymin, xmax, ymax)
    return data[0] if data else None


def _compute_cams_interior_totals(
    step1_dir: Path,
    gnfr_bases: set[str],
    interior: np.ndarray,
    cams_domain_ids: np.ndarray,
    n_cells_per_coarse: dict[int, int],
    poll_cols: list[str],
) -> dict[str, float]:
    """
    True CAMS totals for interior coarse cells (coverage >= 0.8).
    Corrects step1 duplication: true total per coarse cell = sum(step1 in cell) / n_cells.
    """
    nrow, ncol = interior.shape
    ref = {p: 0.0 for p in poll_cols}
    for f in step1_dir.glob("*.csv"):
        sector = f.stem.replace("-", "_")
        if sector not in gnfr_bases:
            continue
        try:
            cf = pd.read_csv(f)
            for _, r in cf.iterrows():
                rr, cc = int(r["row"]), int(r["col"])
                if 0 <= rr < nrow and 0 <= cc < ncol and interior[rr, cc]:
                    cid = int(cams_domain_ids[rr, cc])
                    n = n_cells_per_coarse.get(cid, 1)
                    if n <= 0:
                        continue
                    for p in poll_cols:
                        if p in cf.columns:
                            v = r[p]
                            if not (pd.isna(v) or v == NA_VAL):
                                ref[p] += float(v) / n
        except (ValueError, Exception):
            pass
    return ref


def _segment_length_m(df: pd.DataFrame) -> np.ndarray:
    """Euclidean length in meters for line segments (domain CRS assumed metric)."""
    dx = df["xcor_end"].values - df["xcor_start"].values
    dy = df["ycor_end"].values - df["ycor_start"].values
    return np.sqrt(dx * dx + dy * dy)


def _morans_i_knn(values: np.ndarray, x: np.ndarray, y: np.ndarray, k: int = 8) -> float | None:
    """
    Moran's I for line segments using k-nearest neighbors by centroid distance.
    """
    n = len(values)
    if n < 2 or k >= n:
        return None
    v = np.array(values, dtype=float)
    v_mean = v.mean()
    var = np.sum((v - v_mean) ** 2)
    if var < 1e-20:
        return None
    from scipy.spatial import cKDTree
    pts = np.column_stack([x, y])
    tree = cKDTree(pts)
    w_sum = 0
    cross = 0
    for i in range(n):
        dists, idxs = tree.query(pts[i], k=k + 1)
        dists = dists[1:]
        idxs = idxs[1:]
        for d, j in zip(dists, idxs):
            if d > 1e-6:
                w = 1.0 / d
                w_sum += w
                cross += w * (v[i] - v_mean) * (v[j] - v_mean)
    if w_sum < 1e-20:
        return None
    return (n / w_sum) * (cross / var)


def compute_output_statistics(
    output_path: str | Path,
    output_folder: str | Path,
    source_type: str,
    config_dir: str | Path | None = None,
) -> dict[str, Any]:
    """
    Compute full statistics for output CSV.
    Returns dict with: summary, cams_comparison, stacked_bar, radar, cdf, moran, line_scatter, line_box.
    """
    path = Path(output_path)
    base = Path(output_folder)
    if not path.exists():
        return {"error": "Output file not found"}

    df = pd.read_csv(path)
    df = _clean_df(df, source_type)

    poll_cols = [c for c in POLLUTANTS if c in df.columns]
    if not poll_cols:
        return {"error": "No pollutant columns found"}

    result: dict[str, Any] = {
        "source_type": source_type,
        "units": "g/s" if source_type == "line" else "kg/year",
        "pollutants": poll_cols,
        "summary": {},
        "cams_comparison": None,
        "cams_comparisons": [],
        "stacked_bar": None,
        "radar": None,
        "cdf": {},
        "moran": None,
        "line_scatter": None,
        "line_box": None,
    }

    for p in poll_cols:
        vals = df[p].values
        vals = vals[vals > 0]
        if len(vals) == 0:
            result["summary"][p] = {
                "total": 0, "mean": 0, "median": 0, "std": 0,
                "min": 0, "max": 0, "count": 0, "sparsity": 100,
                "p25": 0, "p50": 0, "p75": 0, "p90": 0,
                "top10_share": 0,
            }
            continue
        total = float(np.sum(vals))
        n_total = len(df)
        n_nz = len(vals)
        sorted_vals = np.sort(vals)[::-1]
        cumsum = np.cumsum(sorted_vals)
        top10_n = max(1, int(0.1 * n_nz))
        top10_share = float(cumsum[top10_n - 1] / total) if total > 0 else 0

        result["summary"][p] = {
            "total": total,
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "std": float(np.std(vals)) if len(vals) > 1 else 0,
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "count": int(n_nz),
            "sparsity": 100 * (1 - n_nz / n_total) if n_total > 0 else 0,
            "p25": float(np.percentile(vals, 25)),
            "p50": float(np.percentile(vals, 50)),
            "p75": float(np.percentile(vals, 75)),
            "p90": float(np.percentile(vals, 90)),
            "top10_share": top10_share,
        }

    cdf_data = {}
    for p in poll_cols:
        vals = df[p].values
        vals = vals[vals > 0]
        if len(vals) == 0:
            cdf_data[p] = {"pct_cells": [0, 100], "pct_emission": [0, 100]}
            continue
        total = np.sum(vals)
        sorted_vals = np.sort(vals)[::-1]
        cumsum = np.cumsum(sorted_vals)
        n = len(vals)
        pct_cells = 100 * np.arange(1, n + 1) / n
        pct_emission = 100 * cumsum / total if total > 0 else np.zeros_like(cumsum)
        cdf_data[p] = {
            "pct_cells": [0] + [float(x) for x in pct_cells],
            "pct_emission": [0] + [float(x) for x in pct_emission],
        }
    result["cdf"] = cdf_data

    if source_type == "area":
        intermediates = base / "intermediates"
        snap_export = None
        gnfr_to_snap = None
        for cfg_dir in [Path(config_dir) if config_dir else None, Path(__file__).parent.parent / "config"]:
            if cfg_dir is None:
                continue
            snap_cfg = cfg_dir / "snap_mapping.json"
            if snap_cfg.exists():
                try:
                    import json
                    with open(snap_cfg, encoding="utf-8") as f:
                        snap_cfg_data = json.load(f)
                    snap_export = set(snap_cfg_data.get("snap_sectors_export", []))
                    gnfr_to_snap = snap_cfg_data.get("gnfr_to_snap", {})
                    break
                except Exception:
                    pass

        gnfr_bases_to_include = set()
        if gnfr_to_snap and snap_export:
            for key, mapping in gnfr_to_snap.items():
                if mapping.get("snap") in snap_export:
                    base = key.replace("_snap4", "").replace("_snap3", "")
                    gnfr_bases_to_include.add(base)

        step4_dir = intermediates / "step4_snap"
        if step4_dir.exists() and snap_export:
            ref_totals = {}
            for snap_id in snap_export:
                snap_path = step4_dir / f"snap{snap_id}.csv"
                if not snap_path.exists():
                    continue
                try:
                    cf = pd.read_csv(snap_path)
                    for p in poll_cols:
                        if p in cf.columns:
                            ref_totals[p] = ref_totals.get(p, 0) + cf[p].replace(NA_VAL, 0).sum()
                except (ValueError, Exception):
                    pass
            if ref_totals:
                out_totals = {p: result["summary"][p]["total"] for p in poll_cols}
                result["cams_comparison"] = {
                    "label": "Output vs step4 (after proxy, GNFR->SNAP). Ratio 1 = mass conserved.",
                    "pollutant": poll_cols,
                    "output": [out_totals.get(p, 0) for p in poll_cols],
                    "reference": [ref_totals.get(p, 0) for p in poll_cols],
                    "ratio": [
                        out_totals.get(p, 0) / ref_totals[p] if ref_totals.get(p, 0) > 0 else 0
                        for p in poll_cols
                    ],
                }
                result["cams_comparisons"].append(result["cams_comparison"])

                grid_meta_path = intermediates / "grid_metadata.csv"
                step1_dir = intermediates / "step1_cams_warped"
                if (
                    grid_meta_path.exists()
                    and step1_dir.exists()
                    and gnfr_bases_to_include
                ):
                    grid_meta = pd.read_csv(grid_meta_path).iloc[0]
                    nrow_d = int(grid_meta["nrow"])
                    ncol_d = int(grid_meta["ncol"])
                    xmin_d = float(grid_meta["xmin"])
                    xmax_d = float(grid_meta["xmax"])
                    ymin_d = float(grid_meta["ymin"])
                    ymax_d = float(grid_meta["ymax"])
                    interior_data = _compute_interior_data(
                        intermediates, nrow_d, ncol_d, xmin_d, ymin_d, xmax_d, ymax_d
                    )
                    if interior_data is not None:
                        interior, cams_domain_ids, n_cells_per_coarse = interior_data
                        ref_interior = _compute_cams_interior_totals(
                            step1_dir,
                            gnfr_bases_to_include,
                            interior,
                            cams_domain_ids,
                            n_cells_per_coarse,
                            poll_cols,
                        )
                        if any(ref_interior.get(p, 0) > 0 for p in poll_cols):
                            cell_w = (xmax_d - xmin_d) / ncol_d
                            cell_h = (ymax_d - ymin_d) / nrow_d
                            out_interior = {p: 0.0 for p in poll_cols}
                            for _, r in df.iterrows():
                                xsw = float(r["xcor_sw"])
                                ysw = float(r["ycor_sw"])
                                cc = int((xsw - xmin_d) / cell_w)
                                rr = int((ymax_d - ysw) / cell_h) - 1
                                if 0 <= rr < nrow_d and 0 <= cc < ncol_d and interior[rr, cc]:
                                    for p in poll_cols:
                                        if p in df.columns:
                                            v = r[p]
                                            if v != NA_VAL and pd.notna(v) and float(v) >= 0:
                                                out_interior[p] += float(v)
                            result["cams_comparisons"].append({
                                "label": "Output vs CAMS (interior cells only, pre-warped totals)",
                                "pollutant": poll_cols,
                                "output": [out_interior.get(p, 0) for p in poll_cols],
                                "reference": [ref_interior.get(p, 0) for p in poll_cols],
                                "ratio": [
                                    out_interior.get(p, 0) / ref_interior[p]
                                    if ref_interior.get(p, 0) > 0 else 0
                                    for p in poll_cols
                                ],
                            })

        if "snap" in df.columns:
            snaps = sorted(df["snap"].dropna().astype(int).unique().tolist())
            stacked = []
            for p in poll_cols:
                for snap in snaps:
                    sub = df[df["snap"] == snap]
                    tot = sub[p].sum()
                    if tot > 0:
                        stacked.append({"pollutant": p, "snap": int(snap), "total": float(tot)})
            result["stacked_bar"] = stacked

            radar = []
            for snap in snaps:
                sub = df[df["snap"] == snap]
                row = {"snap": int(snap)}
                for p in poll_cols:
                    row[p] = float(sub[p].sum())
                radar.append(row)
            result["radar"] = radar

        moran_data = {}
        try:
            from scipy.spatial import cKDTree
            for p in poll_cols:
                for snap in df["snap"].dropna().astype(int).unique():
                    sub = df[df["snap"] == snap]
                    if len(sub) < 4:
                        continue
                    cx = (sub["xcor_sw"] + sub["xcor_ne"]) / 2
                    cy = (sub["ycor_sw"] + sub["ycor_ne"]) / 2
                    vals = sub[p].values
                    mi = _morans_i_knn(vals, cx.values, cy.values, k=min(4, len(sub) - 1))
                    key = f"{p}_snap{snap}"
                    moran_data[key] = float(mi) if mi is not None else None
            if moran_data:
                result["moran"] = moran_data
        except ImportError:
            pass

    elif source_type == "line":
        line_cams = base / "intermediates" / "line_cams" / "cams_stacked.csv"
        if line_cams.exists():
            try:
                cams_df = pd.read_csv(line_cams)
                ref_totals = {}
                kgyr_to_gs = 1e3 / (365 * 24 * 3600)
                for p in poll_cols:
                    if p in cams_df.columns:
                        ref_totals[p] = cams_df[p].replace(NA_VAL, 0).sum() * kgyr_to_gs
                if ref_totals:
                    out_totals = {p: result["summary"][p]["total"] for p in poll_cols}
                    result["cams_comparison"] = {
                        "label": "Output vs line CAMS raster",
                        "pollutant": poll_cols,
                        "output": [out_totals.get(p, 0) for p in poll_cols],
                        "reference": [ref_totals.get(p, 0) for p in poll_cols],
                        "ratio": [
                            out_totals.get(p, 0) / ref_totals[p] if ref_totals.get(p, 0) > 0 else 0
                            for p in poll_cols
                        ],
                    }
            except Exception:
                pass

        length_m = _segment_length_m(df)
        scatter_data = []
        for p in poll_cols:
            for i in range(len(df)):
                if length_m[i] > 0 and df[p].iloc[i] > 0:
                    scatter_data.append({
                        "pollutant": p,
                        "length_m": float(length_m[i]),
                        "emission": float(df[p].iloc[i]),
                    })
        if scatter_data:
            result["line_scatter"] = scatter_data[:2000]

        if "roadtype" in df.columns:
            box_data = []
            for p in poll_cols:
                for rt in df["roadtype"].dropna().unique():
                    sub = df[df["roadtype"] == rt]
                    length_m = _segment_length_m(sub)
                    em = sub[p].values
                    mask = length_m > 0.1
                    em_per_m = em[mask] / length_m[mask] if np.any(mask) else np.array([])
                    if len(em_per_m) > 0:
                        box_data.append({
                            "pollutant": p,
                            "roadtype": str(rt),
                            "values": [float(x) for x in em_per_m],
                            "median": float(np.median(em_per_m)),
                            "q25": float(np.percentile(em_per_m, 25)),
                            "q75": float(np.percentile(em_per_m, 75)),
                        })
            if box_data:
                result["line_box"] = box_data

        try:
            from scipy.spatial import cKDTree
            cx = (df["xcor_start"] + df["xcor_end"]) / 2
            cy = (df["ycor_start"] + df["ycor_end"]) / 2
            moran_data = {}
            for p in poll_cols:
                vals = df[p].values
                if np.var(vals) < 1e-20:
                    continue
                mi = _morans_i_knn(vals, cx.values, cy.values, k=min(8, len(df) - 1))
                moran_data[p] = float(mi) if mi is not None else None
            if moran_data:
                result["moran"] = moran_data
        except ImportError:
            pass

    return result
