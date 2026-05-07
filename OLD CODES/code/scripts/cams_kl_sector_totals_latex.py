#!/usr/bin/env python3
"""
Sum CAMS-REG-ANT v8.1 NetCDF emissions for GNFR sectors K (AgriLivestock) and L (AgriOther),
for all pollutant variables (per-source arrays), and write a LaTeX table.

Sector indices follow emission_category_index 1..15 = A..L, so K=14, L=15 (1-based).

**Pollutants:** Every numeric data variable with dimension ``source`` is included (typical set:
CH4, CO, NH3, NMVOC, NOx, PM10, PM2.5, SOx, CO2 fossil, CO2 biofuel). New variables in a
future file are picked up automatically.

**Source geometry:** This inventory uses ``source_type_index`` 1 = area, 2 = point only.
There is **no separate "line" source row** in the NetCDF; road line emissions are GNFR F1--F4,
not K/L. Use ``--source area`` to exclude point sources from totals if needed.

Usage:
  python code/scripts/cams_kl_sector_totals_latex.py --nc path/to/emissions_year2019.nc -o out.tex
  python code/scripts/cams_kl_sector_totals_latex.py --dir data/given_CAMS/.../netcdf -o out.tex --source area

If multiple .nc files are in --dir, totals are summed across files (same inventory split across files).

Requires: xarray, netCDF4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# GNFR order in CAMS-REG-ANT v8.1 (emis_cat index 1-based)
GNFR_CODES = (
    "A",
    "B",
    "C",
    "D",
    "E",
    "F1",
    "F2",
    "F3",
    "F4",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
)
GNFR_NAMES = {
    "K": "AgriLivestock",
    "L": "AgriOther",
}

IDX_K = GNFR_CODES.index("K") + 1  # 14
IDX_L = GNFR_CODES.index("L") + 1  # 15

# Preferred row order in the LaTeX table (any extra vars from the file are appended alphabetically)
POLLUTANT_ORDER = (
    "ch4",
    "co",
    "nh3",
    "nmvoc",
    "nox",
    "pm10",
    "pm2_5",
    "sox",
    "co2_ff",
    "co2_bf",
)


def _collect_nc_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() == ".nc":
            return [path]
        raise SystemExit(f"Not a NetCDF file: {path}")
    if path.is_dir():
        files = sorted(path.glob("*.nc"))
        if not files:
            raise SystemExit(f"No .nc files in {path}")
        return files
    raise SystemExit(f"Path not found: {path}")


def _pollutant_vars(ds) -> list[str]:
    """Numeric emission-like variables with dimension ``source`` (auto-discovered, complete)."""
    import numpy as np

    skip = {
        "longitude_source",
        "latitude_source",
        "longitude_index",
        "latitude_index",
        "country_index",
        "emission_category_index",
        "source_type_index",
    }
    out: list[str] = []
    for name, v in ds.data_vars.items():
        if name in skip:
            continue
        dims = tuple(v.dims)
        if dims != ("source",) and not (len(dims) == 1 and dims[0] == "source"):
            continue
        if not np.issubdtype(v.dtype, np.number):
            continue
        out.append(name)
    return out


def _sort_pollutant_keys(keys: list[str]) -> list[str]:
    known = [k for k in POLLUTANT_ORDER if k in keys]
    rest = sorted(k for k in keys if k not in POLLUTANT_ORDER)
    return known + rest


def _pretty_pollutant(name: str) -> str:
    mapping = {
        "ch4": "CH$_4$",
        "co": "CO",
        "nh3": "NH$_3$",
        "nmvoc": "NMVOC",
        "nox": "NO$_x$",
        "pm10": "PM$_{10}$",
        "pm2_5": "PM$_{2.5}$",
        "sox": "SO$_x$",
        "co2_ff": "CO$_{2}$ (fossil)",
        "co2_bf": "CO$_{2}$ (biofuel)",
    }
    return mapping.get(name, name.replace("_", "\\_"))


def _source_type_mask(ds, mode: str) -> "np.ndarray | None":
    """Optional mask for ``source`` rows: area only (1), point only (2), or None = all."""
    import numpy as np

    if mode == "all":
        return None
    st = np.asarray(ds["source_type_index"].values).ravel().astype(np.int64)
    if mode == "area":
        return st == 1
    if mode == "point":
        return st == 2
    raise ValueError(f"Unknown --source mode: {mode!r}")


def _sum_sector(
    ds,
    emis_idx: "np.ndarray",
    sector_code: int,
    var: str,
    extra_mask: "np.ndarray | None" = None,
) -> float:
    import numpy as np

    mask = emis_idx == int(sector_code)
    if extra_mask is not None:
        mask = mask & extra_mask
    arr = np.asarray(ds[var].values).ravel()
    if arr.size != emis_idx.size:
        return float("nan")
    return float(np.nansum(arr[mask]))


def compute_totals(
    nc_paths: list[Path],
    *,
    source_mode: str = "all",
) -> tuple[dict[str, tuple[float, float]], dict[str, str], str, list[str]]:
    """Returns (pollutant -> (total_K, total_L)), units per var, history string, sorted pollutant keys."""
    import numpy as np
    import xarray as xr

    totals_k: dict[str, float] = {}
    totals_l: dict[str, float] = {}
    units: dict[str, str] = {}
    title_bits: list[str] = []
    all_keys: set[str] = set()

    for p in nc_paths:
        ds = xr.open_dataset(p)
        title_bits.append(str(ds.attrs.get("history", p.name)))
        emis = np.asarray(ds["emission_category_index"].values).ravel().astype(np.int64)
        st_mask = _source_type_mask(ds, source_mode)

        for v in _pollutant_vars(ds):
            all_keys.add(v)
            if v not in totals_k:
                totals_k[v] = 0.0
                totals_l[v] = 0.0
            tk = _sum_sector(ds, emis, IDX_K, v, st_mask)
            tl = _sum_sector(ds, emis, IDX_L, v, st_mask)
            if np.isfinite(tk):
                totals_k[v] += tk
            if np.isfinite(tl):
                totals_l[v] += tl
            if v not in units:
                u = ds[v].attrs.get("units", "kg/year")
                lu = ds[v].attrs.get("long_units", "")
                units[v] = f"{u}" + (f" ({lu})" if lu else "")

        ds.close()

    order = _sort_pollutant_keys(sorted(all_keys))
    out = {v: (totals_k[v], totals_l[v]) for v in order}
    title = " ".join(title_bits[:1]) if title_bits else ""
    return out, units, title, order


def _latex_escape(s: str) -> str:
    return s.replace("%", "\\%").replace("_", "\\_")


def write_latex(
    totals: dict[str, tuple[float, float]],
    units: dict[str, str],
    *,
    out_path: Path,
    caption: str,
    label: str,
    kt: bool,
    source_mode: str,
    pollutant_order: list[str],
) -> None:
    """Write booktabs-style table; if kt, divide kg by 1e6 for Mt/kt display note."""
    lines: list[str] = []
    lines.append("% Generated by cams_kl_sector_totals_latex.py")
    lines.append("% GNFR K = AgriLivestock, L = AgriOther (CAMS-REG-ANT v8.1 emission_category_index 14, 15)")
    lines.append(
        "% Pollutants: all numeric NetCDF variables with dimension 'source' "
        "(see list in comment below)."
    )
    lines.append(
        "% Note: CAMS-REG-ANT v8.1 has source_type_index 1=area, 2=point only; "
        "no separate 'line' source. Road line emissions are GNFR F1--F4, not K/L."
    )
    lines.append(f"% Variables included ({len(pollutant_order)}): {', '.join(pollutant_order)}")
    lines.append("")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    scale = 1e6 if kt else 1.0
    unit_note = "kt yr$^{-1}$" if kt else "kg yr$^{-1}$"
    if source_mode == "all":
        src_txt = "area and point sources"
    elif source_mode == "area":
        src_txt = "area sources only (point sources excluded)"
    else:
        src_txt = "point sources only (area sources excluded)"
    full_cap = (
        f"CAMS-REG-ANT v8.1: total emissions by pollutant for GNFR K (AgriLivestock) "
        f"and L (AgriOther), summing {src_txt}. "
        f"NetCDF: {caption}. Units: {unit_note}"
        + (" (converted from kg yr$^{-1}$)." if kt else ".")
    )
    if len(full_cap) > 500:
        full_cap = full_cap[:497] + "..."
    lines.append(f"\\caption{{{_latex_escape(full_cap)}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\begin{tabular}{lrr}")
    lines.append("\\hline")
    lines.append(f"Pollutant & K ({_latex_escape(GNFR_NAMES['K'])}) & L ({_latex_escape(GNFR_NAMES['L'])}) \\\\")
    lines.append("\\hline")
    for v in pollutant_order:
        if v not in totals:
            continue
        tk, tl = totals[v]
        tk_s = (tk / scale) if kt else tk
        tl_s = (tl / scale) if kt else tl
        if kt:
            lines.append(
                f"{_pretty_pollutant(v)} & {_fmt_num(tk_s)} & {_fmt_num(tl_s)} \\\\"
            )
        else:
            lines.append(
                f"{_pretty_pollutant(v)} & {_fmt_latex_sci(tk)} & {_fmt_latex_sci(tl)} \\\\"
            )
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")
    lines.append("% Column units: " + ("1 kt = 10^6 kg" if kt else "kg/year as in NetCDF"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _fmt_sci(x: float) -> str:
    """Consistent LaTeX-safe numeric format (scientific for large |x|)."""
    if x != x:  # nan
        return "---"
    if x == 0.0:
        return "0"
    ax = abs(x)
    if ax >= 1e4 or ax < 1e-2:
        return f"{x:.3e}"
    return f"{x:.3f}"


def _fmt_latex_sci(x: float) -> str:
    """Format as $m \\times 10^{e}$ for LaTeX math mode."""
    if x != x:
        return "---"
    if x == 0.0:
        return "$0$"
    import math

    ax = abs(x)
    if ax >= 1e4 or ax < 1e-2:
        exp = int(math.floor(math.log10(ax)))
        mant = x / (10**exp)
        return f"${mant:.3f} \\times 10^{{{exp}}}$"
    return f"${x:.3f}$"


def _fmt_num(x: float) -> str:
    if x != x:
        return "---"
    return f"{x:,.4f}"


def _resolve_output_tex(path: Path, default_name: str = "cams_kl_emissions_totals.tex") -> Path:
    """
    If ``path`` ends with ``.tex``, use it as the output file (parent dirs created).
    Otherwise treat ``path`` as a directory and write ``default_name`` inside it.
    """
    if path.suffix.lower() == ".tex":
        path.parent.mkdir(parents=True, exist_ok=True)
        return path.resolve()
    path.mkdir(parents=True, exist_ok=True)
    return (path / default_name).resolve()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="CAMS K/L sector totals to LaTeX table.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--nc", type=Path, help="Single CAMS NetCDF file")
    g.add_argument("--dir", type=Path, help="Folder with one or more .nc files")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output .tex file path, or a directory (writes cams_kl_emissions_totals.tex there)",
    )
    p.add_argument(
        "--kt",
        action="store_true",
        help="Express masses in kilotonnes per year (10^6 kg)",
    )
    p.add_argument(
        "--label",
        default="tab:cams_kl_totals",
        help="LaTeX label for the table",
    )
    p.add_argument(
        "--source",
        choices=("all", "area", "point"),
        default="all",
        help="Restrict to area (1) or point (2) source_type_index rows; default includes both",
    )
    args = p.parse_args(argv)

    try:
        import xarray  # noqa: F401
    except ImportError:
        print("Install xarray and netCDF4: pip install xarray netCDF4", file=sys.stderr)
        return 1

    path = args.nc if args.nc is not None else args.dir
    nc_files = _collect_nc_files(path)

    totals, units, hist, order = compute_totals(nc_files, source_mode=args.source)
    if not totals:
        print("No pollutant variables found with dimension 'source'.", file=sys.stderr)
        return 1

    cap = ", ".join(f.name for f in nc_files)
    if hist:
        cap += f" ({hist[:120]})"

    out_tex = _resolve_output_tex(args.output)
    write_latex(
        totals,
        units,
        out_path=out_tex,
        caption=cap,
        label=args.label,
        kt=args.kt,
        source_mode=args.source,
        pollutant_order=order,
    )
    print(f"Wrote {out_tex}")
    print(f"Pollutants included ({len(order)}): {', '.join(order)}")
    print(f"Source filter: {args.source} (CAMS has no separate 'line' type; roads are F1/F4)")
    for v in order:
        tk, tl = totals[v]
        print(f"  {v}: K={tk:.6g} L={tl:.6g} kg/yr")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
