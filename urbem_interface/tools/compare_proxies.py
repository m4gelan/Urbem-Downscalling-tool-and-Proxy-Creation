"""
Compare new proxy rasters vs reference outputs (table + optional diff PNGs).

Usage:
    python -m urbem_interface.tools.compare_proxies \\
        --ref  path/to/reference_rasters \\
        --new  path/to/Output/proxy/default \\
        --out  path/to/comparison_output
"""

import argparse
import json
import logging
import math
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

REGION_BOUNDS_3035: dict[str, tuple[float, float, float, float]] = {
    "france": (3_480_000.0, 2_150_000.0, 4_150_000.0, 3_180_000.0),
    "germany": (4_020_000.0, 2_620_000.0, 4_720_000.0, 3_520_000.0),
    "iberia": (2_500_000.0, 1_550_000.0, 3_650_000.0, 2_650_000.0),
    "italy": (3_850_000.0, 1_520_000.0, 4_750_000.0, 2_450_000.0),
    "poland": (4_350_000.0, 3_050_000.0, 5_100_000.0, 3_650_000.0),
}


def _compare_block_size() -> int:
    raw = os.environ.get("URBLEM_COMPARE_BLOCK", "").strip()
    default = 2048
    if not raw:
        return default
    try:
        n = int(raw)
    except ValueError:
        return default
    return max(256, min(n, 16384))


def _diff_figure_max_side() -> int:
    raw = os.environ.get("URBLEM_COMPARE_FIG_MAX_SIDE", "").strip()
    default = 2048
    if not raw:
        return default
    try:
        n = int(raw)
    except ValueError:
        return default
    return max(256, min(n, 8192))


def _sanitize_json_value(obj: Any) -> Any:
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_json_value(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_json_value(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        x = float(obj)
        return None if math.isnan(x) or math.isinf(x) else x
    return obj


def _compare_window(
    transform,
    width: int,
    height: int,
    bounds: tuple[float, float, float, float] | None,
):
    from rasterio.windows import Window, from_bounds

    if bounds is None:
        return Window(0, 0, width, height)
    left, bottom, right, top = bounds
    win = from_bounds(left, bottom, right, top, transform)
    win = win.round_offsets(op="floor").round_lengths(op="ceil")
    inner = Window(0, 0, width, height)
    return win.intersection(inner)


PROXY_PAIRS: dict[str, str] = {
    "Proxy_Industry.tif": "LU_Industry.tif",
    "Proxy_Agriculture.tif": "LU_Agriculture.tif",
    "Proxy_Aviation.tif": "LU_Airports.tif",
    "Proxy_OffRoad_Mobility.tif": "Non_Road_Mob_Sources.tif",
    "Proxy_Waste_Wastewater.tif": "LU_Waste.tif",
    "Proxy_Shipping.tif": "LU_Shipping.tif",
    "Proxy_EPRTR_SNAP34.tif": "LU_Snap34.tif",
    "Proxy_EPRTR_SNAP1.tif": "LU_Snap1.tif",
    "lu_snap34.tif": "LU_Snap34.tif",
}


def _read_raster(path: Path) -> tuple[np.ndarray, dict]:
    try:
        import rasterio
    except ImportError:
        raise ImportError("rasterio is required for proxy comparison.")
    with rasterio.open(path) as src:
        array = src.read(1).astype(np.float64)
        profile = src.profile
    return array, profile


def _finite_and_not_nodata(arr: np.ndarray, nodata) -> np.ndarray:
    m = np.isfinite(arr)
    if nodata is None:
        return m
    if isinstance(nodata, float) and np.isnan(nodata):
        return m
    return m & (arr != nodata)


def compare_rasters(
    ref_path: Path,
    new_path: Path,
    name: str,
    *,
    bounds: tuple[float, float, float, float] | None = None,
) -> dict:
    if not ref_path.exists():
        return {"name": name, "status": "REF_MISSING", "ref_path": str(ref_path)}
    if not new_path.exists():
        return {"name": name, "status": "NEW_MISSING", "new_path": str(new_path)}

    try:
        import rasterio
        from rasterio.windows import Window
    except ImportError:
        raise ImportError("rasterio is required for proxy comparison.")

    with rasterio.open(ref_path) as rsrc, rasterio.open(new_path) as nsrc:
        if rsrc.shape != nsrc.shape:
            return {
                "name": name,
                "status": "SHAPE_MISMATCH",
                "ref_shape": rsrc.shape,
                "new_shape": nsrc.shape,
            }
        height, width = int(rsrc.height), int(rsrc.width)
        win = _compare_window(rsrc.transform, width, height, bounds)
        row0 = int(win.row_off)
        col0 = int(win.col_off)
        h = int(win.height)
        w = int(win.width)
        n_total = h * w

        nd_r = rsrc.nodata
        nd_n = nsrc.nodata

        bh = _compare_block_size()
        bw = _compare_block_size()
        n_valid = 0
        n_different = 0
        max_abs_diff = 0.0
        sum_abs_diff = 0.0
        sum_sq_diff = 0.0
        sum_r = 0.0
        sum_n = 0.0
        sum_r2 = 0.0
        sum_n2 = 0.0
        sum_rn = 0.0
        n_ref_only_valid = 0
        n_new_only_valid = 0
        n_both_nodata = 0
        tolerance = 1e-9

        for row in range(row0, row0 + h, bh):
            rh = min(bh, row0 + h - row)
            for col in range(col0, col0 + w, bw):
                cw = min(bw, col0 + w - col)
                win_blk = Window(col, row, cw, rh)
                r = rsrc.read(1, window=win_blk).astype(np.float64, copy=False)
                n = nsrc.read(1, window=win_blk).astype(np.float64, copy=False)

                ref_ok = _finite_and_not_nodata(r, nd_r)
                new_ok = _finite_and_not_nodata(n, nd_n)
                valid = ref_ok & new_ok

                n_ref_only_valid += int(np.count_nonzero(ref_ok & ~new_ok))
                n_new_only_valid += int(np.count_nonzero(new_ok & ~ref_ok))
                n_both_nodata += int(np.count_nonzero(~ref_ok & ~new_ok))

                nv = int(np.count_nonzero(valid))
                if nv == 0:
                    continue

                rv = r[valid]
                nv_arr = n[valid]
                diff = np.abs(rv - nv_arr)
                n_valid += nv
                n_different += int(np.count_nonzero(diff > tolerance))
                max_abs_diff = max(max_abs_diff, float(diff.max()))
                sum_abs_diff += float(diff.sum())
                d = rv - nv_arr
                sum_sq_diff += float((d * d).sum())
                sum_r += float(rv.sum())
                sum_n += float(nv_arr.sum())
                sum_r2 += float((rv * rv).sum())
                sum_n2 += float((nv_arr * nv_arr).sum())
                sum_rn += float((rv * nv_arr).sum())

        n_nodata = n_total - n_valid

    out: dict = {
        "name": name,
        "ref_path": str(ref_path),
        "new_path": str(new_path),
        "n_total": n_total,
        "n_ref_only_valid": n_ref_only_valid,
        "n_new_only_valid": n_new_only_valid,
        "n_both_nodata": n_both_nodata,
    }
    if bounds is not None:
        out["compare_bounds"] = [bounds[0], bounds[1], bounds[2], bounds[3]]

    if n_valid == 0:
        out.update(
            {
                "status": "ALL_NODATA",
                "n_nodata": n_nodata,
                "n_valid": 0,
            }
        )
        return out

    match_pct = 100.0 * (n_valid - n_different) / n_valid
    mean_abs_diff = sum_abs_diff / n_valid
    rmse = float(np.sqrt(sum_sq_diff / n_valid))

    vn = float(n_valid)
    cov = sum_rn / vn - (sum_r / vn) * (sum_n / vn)
    var_r = sum_r2 / vn - (sum_r / vn) ** 2
    var_n = sum_n2 / vn - (sum_n / vn) ** 2
    if var_r > 0 and var_n > 0:
        correlation = float(cov / (np.sqrt(var_r) * np.sqrt(var_n)))
    else:
        correlation = 1.0 if n_different == 0 else float("nan")

    status = "PASS" if match_pct >= 99.9 else "FAIL"

    out.update(
        {
            "status": status,
            "n_nodata": n_nodata,
            "n_valid": n_valid,
            "n_different": n_different,
            "match_pct": round(match_pct, 4),
            "max_abs_diff": round(max_abs_diff, 10),
            "mean_abs_diff": round(mean_abs_diff, 10),
            "rmse": round(rmse, 10),
            "correlation": round(correlation, 6),
        }
    )
    return out


def generate_diff_figure(
    ref_path: Path,
    new_path: Path,
    output_png: Path,
    name: str,
    report: dict | None = None,
    *,
    bounds: tuple[float, float, float, float] | None = None,
) -> None:
    try:
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping diff figure for %s", name)
        return

    if not ref_path.exists() or not new_path.exists():
        logger.warning("Skipping diff figure for %s (file missing)", name)
        return

    try:
        import rasterio
        from rasterio.enums import Resampling
    except ImportError:
        logger.warning("rasterio not available — skipping diff figure for %s", name)
        return

    max_side = _diff_figure_max_side()
    with rasterio.open(ref_path) as src:
        win = _compare_window(src.transform, src.width, src.height, bounds)
        h_win, w_win = int(win.height), int(win.width)
        if h_win < 1 or w_win < 1:
            logger.warning("Empty window for diff figure %s", name)
            return
        scale = max(h_win, w_win) / max_side
        if scale <= 1:
            out_h, out_w = h_win, w_win
        else:
            out_h = max(1, int(round(h_win / scale)))
            out_w = max(1, int(round(w_win / scale)))
        ref = src.read(
            1,
            window=win,
            out_shape=(out_h, out_w),
            resampling=Resampling.nearest,
        ).astype(np.float64)
        nd_r = src.nodata

    with rasterio.open(new_path) as src:
        win_n = _compare_window(src.transform, src.width, src.height, bounds)
        new = src.read(
            1,
            window=win_n,
            out_shape=(ref.shape[0], ref.shape[1]),
            resampling=Resampling.nearest,
        ).astype(np.float64)
        nd_n = src.nodata

    ref_valid = _finite_and_not_nodata(ref, nd_r)
    new_valid = _finite_and_not_nodata(new, nd_n)

    tol = 1e-9
    both_nodata = ~ref_valid & ~new_valid
    match = ref_valid & new_valid & (np.abs(ref - new) <= tol)
    ref_only_mask = ref_valid & ~new_valid
    new_only_mask = ~ref_valid & new_valid
    both_diff = ref_valid & new_valid & ~match
    yellow_mismatch = both_diff & (ref > new + tol)
    purple_mismatch = both_diff & (new > ref + tol)
    yellow = ref_only_mask | yellow_mismatch
    purple = new_only_mask | purple_mismatch

    n_green = int(np.count_nonzero(match))
    n_yellow = int(np.count_nonzero(yellow))
    n_purple = int(np.count_nonzero(purple))
    n_grey = int(np.count_nonzero(both_nodata))

    grey_rgb = (0.73, 0.73, 0.73)
    green_rgb = (0.15, 0.72, 0.28)
    yellow_rgb = (0.92, 0.82, 0.18)
    purple_rgb = (0.55, 0.22, 0.78)
    rgb = np.zeros((ref.shape[0], ref.shape[1], 3), dtype=np.float32)
    rgb[both_nodata] = grey_rgb
    rgb[match] = green_rgb
    rgb[yellow] = yellow_rgb
    rgb[purple] = purple_rgb

    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    ax.imshow(rgb, interpolation="nearest", origin="upper")
    ax.set_title("Agreement map (green=match, yellow=ref not new, purple=new not ref)")
    ax.axis("off")

    leg = [
        mpatches.Patch(facecolor=green_rgb, edgecolor="0.3", label=f"Match ({n_green:,})"),
        mpatches.Patch(facecolor=yellow_rgb, edgecolor="0.3", label=f"Ref only / ref > new ({n_yellow:,})"),
        mpatches.Patch(facecolor=purple_rgb, edgecolor="0.3", label=f"New only / new > ref ({n_purple:,})"),
        mpatches.Patch(facecolor=grey_rgb, edgecolor="0.3", label=f"Both nodata ({n_grey:,})"),
    ]
    fig.legend(handles=leg, loc="lower center", ncol=2, fontsize=9, frameon=True)

    lines = [name, f"yellow={n_yellow:,}  purple={n_purple:,}  green={n_green:,}"]
    if report:
        corr = report.get("correlation")
        corr_s = "nan" if corr is None or (isinstance(corr, float) and math.isnan(corr)) else str(corr)
        lines.append(
            f"match={report.get('match_pct','?')}%  max_diff={report.get('max_abs_diff','?')}  corr={corr_s}"
        )
        ro = report.get("n_ref_only_valid")
        no = report.get("n_new_only_valid")
        bv = report.get("n_valid")
        if ro is not None and no is not None and bv is not None:
            lines.append(
                f"mask diag: ref_only_valid={ro:,}  new_only_valid={no:,}  both_valid={bv:,}"
            )
    if bounds is not None:
        lines.append(f"window: {bounds[0]:.0f},{bounds[1]:.0f} … {bounds[2]:.0f},{bounds[3]:.0f} (raster CRS)")
    fig.suptitle("\n".join(lines), fontsize=9)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.tight_layout(rect=(0, 0.06, 1, 0.98))
    fig.savefig(str(output_png), dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Diff figure saved: %s", output_png)


def compare_all(
    ref_folder: Path,
    new_folder: Path,
    output_folder: Path | None = None,
    proxy_pairs: dict[str, str] | None = None,
    *,
    save_diff_plots: bool = True,
    bounds: tuple[float, float, float, float] | None = None,
    plot_suffix: str = "",
) -> list[dict]:
    if proxy_pairs is None:
        proxy_pairs = PROXY_PAIRS

    ref_folder = Path(ref_folder)
    new_folder = Path(new_folder)

    results = []
    header = f"{'Proxy':<30} {'Status':<12} {'Match%':>8} {'MaxDiff':>14} {'Corr':>10} {'NDiff':>8}"
    sep = "-" * len(header)

    print()
    print("=" * len(header))
    print(f"  Reference : {ref_folder}")
    print(f"  New       : {new_folder}")
    if bounds is not None:
        print(
            f"  Window    : L={bounds[0]:.0f} B={bounds[1]:.0f} "
            f"R={bounds[2]:.0f} T={bounds[3]:.0f} (raster CRS)"
        )
    print("=" * len(header))
    print(header)
    print(sep)

    for new_fname, ref_fname in proxy_pairs.items():
        ref_path = ref_folder / ref_fname
        new_path = new_folder / new_fname
        name = new_fname.replace(".tif", "")

        report = compare_rasters(ref_path, new_path, name, bounds=bounds)
        results.append(report)

        status = report.get("status", "?")
        if status == "PASS":
            row = (
                f"{name:<30} {status:<12} "
                f"{report.get('match_pct',0):>7.3f}% "
                f"{report.get('max_abs_diff',0):>14.2e} "
                f"{report.get('correlation',0):>10.6f} "
                f"{report.get('n_different',0):>8}"
            )
            print(row)
        elif status == "FAIL":
            corr = report.get("correlation")
            if isinstance(corr, float) and math.isnan(corr):
                corr_s = "nan"
            else:
                corr_s = f"{corr:>10.6f}"
            print(
                f"{name:<30} {status:<12} "
                f"{report.get('match_pct',0):>7.3f}% "
                f"{report.get('max_abs_diff',0):>14.2e} "
                f"{corr_s:>10} "
                f"{report.get('n_different',0):>8}"
            )
            print(
                f"{'':30}   mask: ref_only={report.get('n_ref_only_valid', 0)} "
                f"new_only={report.get('n_new_only_valid', 0)} "
                f"both_valid={report.get('n_valid', 0)}"
            )
        else:
            print(f"{name:<30} {status:<12}")

    print(sep)
    n_pass = sum(1 for r in results if r.get("status") == "PASS")
    n_fail = sum(1 for r in results if r.get("status") == "FAIL")
    n_missing = sum(1 for r in results if "MISSING" in r.get("status", ""))
    print(f"  PASS: {n_pass}   FAIL: {n_fail}   MISSING: {n_missing}")
    print("=" * len(header))
    print()

    if output_folder:
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        summary_name = f"comparison_summary{plot_suffix}.json" if plot_suffix else "comparison_summary.json"
        summary_path = output_folder / summary_name
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(_sanitize_json_value(results), fh, indent=2)
        print(f"Summary JSON saved: {summary_path}")

        if save_diff_plots:
            for report in results:
                if report.get("status") not in ("PASS", "FAIL"):
                    continue
                new_path = Path(report["new_path"])
                ref_path = Path(report["ref_path"])
                png_out = output_folder / f"diff_{report['name']}{plot_suffix}.png"
                generate_diff_figure(
                    ref_path, new_path, png_out, report["name"], report, bounds=bounds
                )

    return results


def compare_shapefiles(
    ref_path: Path,
    new_path: Path,
    name: str,
    type_col: str = "typeOfRoad",
    output_folder: Path | None = None,
) -> dict:
    try:
        import geopandas as gpd
    except ImportError:
        raise ImportError("geopandas is required for shapefile comparison.")

    if not Path(ref_path).exists():
        return {"name": name, "status": "REF_MISSING", "ref_path": str(ref_path)}
    if not Path(new_path).exists():
        return {"name": name, "status": "NEW_MISSING", "new_path": str(new_path)}

    ref_gdf = gpd.read_file(str(ref_path))
    new_gdf = gpd.read_file(str(new_path))

    ref_count = len(ref_gdf)
    new_count = len(new_gdf)
    count_diff_pct = abs(new_count - ref_count) / max(ref_count, 1) * 100

    crs_match = str(ref_gdf.crs) == str(new_gdf.crs)

    ref_types: list[str] = sorted(ref_gdf[type_col].dropna().unique().tolist()) if type_col in ref_gdf.columns else []
    new_types: list[str] = sorted(new_gdf[type_col].dropna().unique().tolist()) if type_col in new_gdf.columns else []
    missing_types = sorted(set(ref_types) - set(new_types))
    extra_types = sorted(set(new_types) - set(ref_types))

    def _per_type_stats(gdf, col: str) -> dict:
        stats = {}
        if col not in gdf.columns:
            return stats
        for rt, grp in gdf.groupby(col):
            total_len_km = float(grp.geometry.length.sum()) / 1000.0
            stats[str(rt)] = {"count": len(grp), "length_km": round(total_len_km, 3)}
        return stats

    per_type_ref = _per_type_stats(ref_gdf, type_col)
    per_type_new = _per_type_stats(new_gdf, type_col)

    if not crs_match:
        status = "FAIL"
    elif missing_types:
        status = "FAIL"
    elif count_diff_pct > 20:
        status = "FAIL"
    elif count_diff_pct > 5:
        status = "WARN"
    else:
        status = "PASS"

    result = {
        "name": name,
        "status": status,
        "ref_count": ref_count,
        "new_count": new_count,
        "count_diff_pct": round(count_diff_pct, 2),
        "crs_match": crs_match,
        "ref_crs": str(ref_gdf.crs),
        "new_crs": str(new_gdf.crs),
        "ref_types": ref_types,
        "new_types": new_types,
        "missing_types": missing_types,
        "extra_types": extra_types,
        "per_type_ref": per_type_ref,
        "per_type_new": per_type_new,
        "ref_path": str(ref_path),
        "new_path": str(new_path),
    }

    if output_folder is not None:
        _generate_shapefile_figure(ref_gdf, new_gdf, name, type_col, result, Path(output_folder))

    return result


def _generate_shapefile_figure(
    ref_gdf,
    new_gdf,
    name: str,
    type_col: str,
    report: dict,
    output_folder: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import matplotlib.cm as cm
    except ImportError:
        logger.warning("matplotlib not available — skipping shapefile figure for %s", name)
        return

    output_folder.mkdir(parents=True, exist_ok=True)
    out_png = output_folder / f"diff_{name}.png"

    all_types = sorted(set(report.get("ref_types", [])) | set(report.get("new_types", [])))
    cmap = cm.get_cmap("tab10", max(len(all_types), 1))
    type_color = {t: cmap(i) for i, t in enumerate(all_types)}

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, gdf, title in [
        (axes[0], ref_gdf, "Reference (original script)"),
        (axes[1], new_gdf, "New (module output)"),
    ]:
        if type_col in gdf.columns:
            for rt, grp in gdf.groupby(type_col):
                grp.plot(ax=ax, color=type_color.get(str(rt), "grey"), linewidth=0.6)
        else:
            gdf.plot(ax=ax, color="steelblue", linewidth=0.6)
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal")
        ax.axis("off")

    legend_patches = [
        mpatches.Patch(color=type_color[t], label=t) for t in all_types
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=min(len(all_types), 4),
        fontsize=8,
        framealpha=0.8,
        bbox_to_anchor=(0.5, 0.0),
    )

    subtitle = (
        f"ref={report['ref_count']} features  "
        f"new={report['new_count']} features  "
        f"diff={report['count_diff_pct']:.1f}%  "
        f"status={report['status']}"
    )
    fig.suptitle(f"{name}\n{subtitle}", fontsize=11)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(str(out_png), dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Shapefile figure saved: %s", out_png)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare new proxy rasters against reference (original script) outputs."
    )
    parser.add_argument(
        "--ref",
        required=True,
        help="Folder with reference rasters (original script outputs).",
    )
    parser.add_argument(
        "--new",
        required=True,
        help="Folder with new module rasters (e.g. Output/proxy/default).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional: folder to save PNG diff maps and JSON summary.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="With --out, write comparison_summary.json only (skip PNG figures).",
    )
    parser.add_argument(
        "--bounds",
        nargs=4,
        type=float,
        metavar=("LEFT", "BOTTOM", "RIGHT", "TOP"),
        default=None,
        help="Compare only this window (left bottom right top in the rasters' CRS, e.g. EPSG:3035).",
    )
    parser.add_argument(
        "--region",
        choices=sorted(REGION_BOUNDS_3035.keys()),
        default=None,
        help="Shorthand bounds in EPSG:3035 (ETRS89-LAEA). Ignored if --bounds is set.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.bounds is not None:
        bounds = (args.bounds[0], args.bounds[1], args.bounds[2], args.bounds[3])
        plot_suffix = "_sub"
    elif args.region is not None:
        bounds = REGION_BOUNDS_3035[args.region]
        plot_suffix = f"_{args.region}"
    else:
        bounds = None
        plot_suffix = ""

    results = compare_all(
        ref_folder=Path(args.ref),
        new_folder=Path(args.new),
        output_folder=Path(args.out) if args.out else None,
        save_diff_plots=not args.no_plots,
        bounds=bounds,
        plot_suffix=plot_suffix,
    )
    n_fail = sum(1 for r in results if r.get("status") == "FAIL")
    return 1 if n_fail > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
