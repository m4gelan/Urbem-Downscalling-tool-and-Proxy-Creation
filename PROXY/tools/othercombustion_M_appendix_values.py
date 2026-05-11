#!/usr/bin/env python3
"""
Emit GNFR C country-matrix ingredients for one ISO3 (default Greece ``GRC``) for report appendices.

Reuses the same merged config as ``C_OtherCombustion`` builder: Eurostat (when enabled),
GAINS file for the country, ``eurostat_end_use.yaml`` bucket mapping, and ``GAINS_mapping.yaml``.

Examples::

  python -m PROXY.tools.othercombustion_M_appendix_values --iso3 GRC
  python -m PROXY.tools.othercombustion_M_appendix_values --iso3 GRC --format latex
  python -m PROXY.tools.othercombustion_M_appendix_values --iso3 GRC --out OUTPUT/M_appendix_grc

Text output goes to stdout; with ``--out``, CSV tables are written under that directory.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

from PROXY.core.alpha.ceip_index_loader import default_ceip_profile_relpath, remap_legacy_ceip_relpath
from PROXY.core.dataloaders import load_path_config, load_yaml, project_root, resolve_path
from PROXY.sectors.C_OtherCombustion.constants import END_USE_COMMERCIAL, MODEL_CLASSES
from PROXY.sectors.C_OtherCombustion.m_builder.assemble import build_m_matrix
from PROXY.sectors.C_OtherCombustion.m_builder.emep_ef import load_emep
from PROXY.sectors.C_OtherCombustion.m_builder.enduse_factors import compute_end_use_factors
from PROXY.sectors.C_OtherCombustion.m_builder.gains_activity import (
    index_gains_files,
    load_gains_rows,
    map_gains_row_to_class,
    _norm_pct_cell,
)
from PROXY.sectors.C_OtherCombustion.m_builder.mapping_io import load_gains_mapping


def _merge_sector_cfg(
    root: Path,
    path_cfg: dict[str, Any],
    sector_cfg: dict[str, Any],
) -> dict[str, Any]:
    spec = (path_cfg.get("proxy_specific") or {}).get("other_combustion") or {}
    if not spec:
        raise ValueError("paths.yaml: missing proxy_specific.other_combustion")
    sector_config_dir = root / "PROXY" / "config" / "other_combustion"
    proxy_common = path_cfg.get("proxy_common") or {}
    waste_spec = (path_cfg.get("proxy_specific") or {}).get("waste") or {}
    ceip_ov = sector_cfg.get("ceip") if isinstance(sector_cfg.get("ceip"), dict) else {}
    gy_rel = remap_legacy_ceip_relpath(
        str(ceip_ov.get("groups_yaml") or default_ceip_profile_relpath(root, "C_OtherCombustion", "groups_yaml"))
    )
    ry_rel = remap_legacy_ceip_relpath(
        str(ceip_ov.get("rules_yaml") or default_ceip_profile_relpath(root, "C_OtherCombustion", "rules_yaml"))
    )
    path_osm = path_cfg.get("osm") or {}
    oc_rel = path_osm.get("other_combustion") or path_osm.get("industry")
    if not oc_rel:
        raise ValueError("paths.yaml: need osm.other_combustion or osm.industry")
    hm_dir = Path(spec["hotmaps_dir"])
    hm_names = sector_cfg.get("hotmaps") or {}
    return {
        "country": {
            "cams_iso3": str(sector_cfg.get("cams_country_iso3", "GRC")).strip().upper(),
            "nuts_cntr": "EL",
        },
        "paths": {
            "gains_dir": spec["gains_dir"],
            "population_tif": proxy_common.get("population_tif"),
            "ghsl_smod_tif": waste_spec.get("ghsl_smod_tif"),
            "ceip_groups_yaml": resolve_path(root, Path(gy_rel)),
            "ceip_rules_yaml": resolve_path(root, Path(ry_rel)),
            "osm_other_combustion_gpkg": resolve_path(root, Path(oc_rel)),
            "emep_ef": sector_config_dir / "EMEP_emission_factors.yaml",
            "gains_mapping": sector_config_dir / "GAINS_mapping.yaml",
            "eurostat_end_use_json": sector_config_dir / "eurostat_end_use.yaml",
            "hotmaps": {
                "heat_res": hm_dir / str(hm_names.get("heat_res", "heat_res_curr_density.tif")),
                "heat_nonres": hm_dir / str(hm_names.get("heat_nonres", "heat_nonres_curr_density.tif")),
                "hdd_curr": hm_dir / str(hm_names.get("hdd_curr", "HDD_curr.tif")),
                "gfa_res": hm_dir / str(hm_names.get("gfa_res", "gfa_res_curr_density.tif")),
                "gfa_nonres": hm_dir / str(hm_names.get("gfa_nonres", "gfa_nonres_curr_density.tif")),
            },
        },
        "gains": sector_cfg.get("gains") or {},
        "eurostat": sector_cfg.get("eurostat") or {},
        "pollutants": sector_cfg.get("pollutants") or [],
    }


def _latex_tt(s: str) -> str:
    return "\\texttt{" + s.replace("_", r"\_") + "}"


def _emit_text(
    iso3: str,
    gains_path: Path | None,
    year_col: str,
    factors: Any,
    pollutant_outputs: list[str],
    M: Any,
    rows_by_class: dict[str, list[tuple[str, str, float]]],
    class_sums: dict[str, float],
) -> None:
    print(f"=== GNFR C — M-matrix appendix values — ISO3={iso3} ===\n")
    print(f"GAINS file: {gains_path}")
    print(f"GAINS year column: {year_col}\n")

    if factors.legacy_class_scalars is not None:
        print("Eurostat: disabled or uniform — row multiplier μ_k = legacy scalar (here all 1.0).\n")
    else:
        print("Eurostat f_enduse (bucket b) — national weights for this country/year:")
        for k in sorted(factors.f_enduse_by_bucket.keys()):
            print(f"  {k!r:50s}  {factors.f_enduse_by_bucket[k]:.12g}")
        prov = getattr(factors, "provenance_enduse", {}) or {}
        if prov:
            print("\nProvenance (where available):")
            for k, v in sorted(prov.items()):
                print(f"  {k}: {v}")
        print()

    print("Bucket b(k) per MODEL_CLASS (code mapping):")
    for cls in MODEL_CLASSES:
        b = factors.bucket_for_class.get(cls, "?")
        print(f"  {cls:22s}  b = {b!r}")
    print()

    print("GAINS within-bucket appliance fraction f_GAINS(k):")
    for cls in MODEL_CLASSES:
        fa = float(factors.f_appliance_by_class.get(cls, 1.0))
        print(f"  {cls:22s}  {fa:.12g}")
    print()

    print("Row multiplier μ_k = f_enduse(b(k)) × f_GAINS(k):")
    for cls in MODEL_CLASSES:
        mu = factors.row_multiplier(cls)
        print(f"  {cls:22s}  {mu:.12g}")
    print()

    print("GAINS activity share s_r (cell/100) summed by class Σ_{r∈k} s_r (not necessarily 1):")
    for cls in MODEL_CLASSES:
        print(f"  {cls:22s}  {class_sums.get(cls, 0.0):.12g}")
    print()

    print("Per-row detail (fuel, appliance, s_r); s_r_norm = s_r / Σ_{q∈k} s_q when sum > 0:")
    for cls in MODEL_CLASSES:
        lst = rows_by_class.get(cls, [])
        if not lst:
            continue
        den = class_sums.get(cls, 0.0)
        print(f"\n--- {cls} (rows={len(lst)}, sum_s={den:.6g}) ---")
        for fuel, app, sr in sorted(lst, key=lambda t: (-t[2], t[0], t[1])):
            nr = (sr / den) if den > 0 else float("nan")
            print(f"  s_r={sr:.8g}  s_r_norm={nr:.8g}  |  {fuel[:56]!r}  |  {app[:56]!r}")

    print("\n=== Assembled M^{(c)} matrix [pollutant × class] (same as assemble.py) ===")
    hdr = "".join(f"{p:>12s}" for p in pollutant_outputs)
    print(f"{'class':<22s}{hdr}")
    for ki, cls in enumerate(MODEL_CLASSES):
        row = "".join(f"{M[pi, ki]:12.6g}" for pi in range(len(pollutant_outputs)))
        print(f"{cls:<22s}{row}")
    print()


def _emit_latex(
    iso3: str,
    gains_path: Path | None,
    year_col: str,
    factors: Any,
    pollutant_outputs: list[str],
    M: Any,
    class_sums: dict[str, float],
) -> None:
    print(f"% Auto-generated appendix snippets (ISO3={iso3}, GAINS={gains_path}, year_col={year_col})\n")

    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Eurostat end-use bucket weights $f_{\mathrm{enduse}}^{(c)}(b)$ for " + iso3 + r".}")
    print(r"\begin{tabular}{lr}")
    print(r"\hline")
    print(r"Bucket $b$ & $f_{\mathrm{enduse}}^{(c)}(b)$ \\")
    print(r"\hline")
    if factors.legacy_class_scalars is not None:
        print(r"\multicolumn{2}{l}{\small Eurostat disabled --- uniform factors.} \\")
    else:
        for k in sorted(factors.f_enduse_by_bucket.keys()):
            kb = str(k).replace("_", r"\_")
            v = factors.f_enduse_by_bucket[k]
            print(f"{kb} & {v:.8g} \\\\")
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print()

    print(r"\begin{table}[t]")
    print(r"\centering")
    print(
        r"\caption{GAINS appliance fractions $f_{\mathrm{GAINS}}^{(c)}(k)$ within bucket and multiplier "
        r"$\mu_k^{(c)}=f_{\mathrm{enduse}}^{(c)}(b(k))\,f_{\mathrm{GAINS}}^{(c)}(k)$ for " + iso3 + r".}"
    )
    print(r"\begin{tabular}{llrr}")
    print(r"\hline")
    print(r"Class $k$ & $b(k)$ & $f_{\mathrm{GAINS}}^{(c)}(k)$ & $\mu_k^{(c)}$ \\")
    print(r"\hline")
    for cls in MODEL_CLASSES:
        b = str(factors.bucket_for_class.get(cls, "")).replace("_", r"\_")
        fk = _latex_tt(cls)
        fa = float(factors.f_appliance_by_class.get(cls, 1.0))
        mu = factors.row_multiplier(cls)
        print(f"{fk} & {b} & {fa:.8g} & {mu:.8g} \\\\")
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print()

    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{GAINS class activity totals $\sum_{r\in k} s_r$ (shares as in code, before $\mu$ and EF).}")
    print(r"\begin{tabular}{lr}")
    print(r"\hline")
    print(r"Class $k$ & $\sum_{r\in k} s_r$ \\")
    print(r"\hline")
    for cls in MODEL_CLASSES:
        print(f"{_latex_tt(cls)} & {class_sums.get(cls, 0.0):.8g} \\\\")
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print()

    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Matrix $M^{(c)}_{p,k}$ assembled by \texttt{build\_m\_matrix} (pollutant $\times$ class).}")
    print(r"\begin{tabular}{l" + "r" * len(pollutant_outputs) + "}")
    print(r"\hline")
    print("Class $k$ & " + " & ".join(str(p).replace("_", r"\_") for p in pollutant_outputs) + r" \\")
    print(r"\hline")
    for ki, cls in enumerate(MODEL_CLASSES):
        cells = " & ".join(f"{M[pi, ki]:.6g}" for pi in range(len(pollutant_outputs)))
        print(f"{_latex_tt(cls)} & {cells} \\\\")
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")


def _write_csv_out(
    out_dir: Path,
    iso3: str,
    factors: Any,
    pollutant_outputs: list[str],
    M: Any,
    rows_by_class: dict[str, list[tuple[str, str, float]]],
    class_sums: dict[str, float],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = out_dir / f"M_appendix_meta_{iso3}.csv"
    with meta.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["iso3", iso3])
        w.writerow(["note", "f_enduse and f_GAINS from compute_end_use_factors; M from build_m_matrix"])

    fe = out_dir / f"M_appendix_f_enduse_{iso3}.csv"
    with fe.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["bucket_b", "f_enduse"])
        if factors.legacy_class_scalars is None:
            for k in sorted(factors.f_enduse_by_bucket.keys()):
                w.writerow([k, factors.f_enduse_by_bucket[k]])
        else:
            w.writerow(["Eurostat_off_or_uniform", "1.0"])

    fc = out_dir / f"M_appendix_class_factors_{iso3}.csv"
    with fc.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_k", "bucket_b_k", "f_GAINS_k", "mu_k", "sum_s_r_in_class"])
        for cls in MODEL_CLASSES:
            w.writerow(
                [
                    cls,
                    factors.bucket_for_class.get(cls, ""),
                    factors.f_appliance_by_class.get(cls, 1.0),
                    factors.row_multiplier(cls),
                    class_sums.get(cls, 0.0),
                ]
            )

    fg = out_dir / f"M_appendix_gains_rows_{iso3}.csv"
    with fg.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_k", "fuel", "appliance", "s_r", "s_r_norm_within_class"])
        for cls in MODEL_CLASSES:
            den = class_sums.get(cls, 0.0)
            for fuel, app, sr in rows_by_class.get(cls, []):
                nr = (sr / den) if den > 0 else ""
                w.writerow([cls, fuel, app, sr, nr])

    fm = out_dir / f"M_matrix_{iso3}.csv"
    with fm.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pollutant_index", "pollutant"] + list(MODEL_CLASSES))
        for pi, pol in enumerate(pollutant_outputs):
            w.writerow([pi, pol] + [M[pi, ki] for ki in range(len(MODEL_CLASSES))])


def main(argv: list[str] | None = None) -> int:
    root = project_root()
    p = argparse.ArgumentParser(description="Print GNFR C M-matrix ingredients for one country (appendix tables).")
    p.add_argument("--config", type=Path, default=root / "PROXY" / "config" / "paths.yaml", help="paths.yaml")
    p.add_argument(
        "--sector-yaml",
        type=Path,
        default=root / "PROXY" / "config" / "sectors" / "othercombustion.yaml",
        help="Sector science YAML (Eurostat on/off, pollutants, gains year column).",
    )
    p.add_argument("--iso3", type=str, default="", help="Focus ISO3 (default: cams_country_iso3 from sector YAML).")
    p.add_argument(
        "--format",
        choices=("text", "latex", "csv"),
        default="text",
        help="text = human-readable; latex = table snippets; csv = only with --out",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="If set, write CSV tables here (always written for --format csv).",
    )
    args = p.parse_args(argv)

    path_cfg = load_path_config(Path(args.config)).resolved
    sector_cfg = load_yaml(Path(args.sector_yaml))
    iso3 = str(args.iso3).strip().upper() if args.iso3 else str(sector_cfg.get("cams_country_iso3", "GRC")).strip().upper()

    merged = _merge_sector_cfg(root, path_cfg, sector_cfg)
    merged["country"]["cams_iso3"] = iso3

    mapping_path = Path(merged["paths"]["gains_mapping"])
    rules, emep_hints = load_gains_mapping(mapping_path)
    spec_oc = (path_cfg.get("proxy_specific") or {}).get("other_combustion") or {}
    gains_dir = resolve_path(root, Path(spec_oc["gains_dir"]))
    overrides = (merged.get("gains") or {}).get("iso3_file_overrides") or {}
    gains_index = index_gains_files(gains_dir, overrides, root)
    gains_path = gains_index.get(iso3)
    year_col = str((merged.get("gains") or {}).get("year_column", "2020"))

    factors = compute_end_use_factors(
        repo_root=root,
        cfg=merged,
        iso3=iso3,
        gains_path=gains_path,
        year_col=year_col,
        rules=rules,
    )

    pollutant_outputs = [str(x["output"]) for x in (merged.get("pollutants") or []) if isinstance(x, dict)]
    if not pollutant_outputs:
        print("No pollutants in sector YAML — using nox pm2_5 nh3 as placeholders", file=sys.stderr)
        pollutant_outputs = ["nox", "pm2_5", "nh3"]

    emep_path = Path(merged["paths"]["emep_ef"])
    emep = load_emep(emep_path)
    M = build_m_matrix(
        gains_path,
        year_col,
        rules,
        factors,
        emep,
        pollutant_outputs,
        emep_fuel_hints=emep_hints if emep_hints else None,
    )

    rows_by_class: dict[str, list[tuple[str, str, float]]] = {k: [] for k in MODEL_CLASSES}
    class_sums: dict[str, float] = {k: 0.0 for k in MODEL_CLASSES}
    if gains_path is not None and gains_path.is_file():
        for fuel, app, ycell in load_gains_rows(gains_path, year_col):
            cls = map_gains_row_to_class(fuel, app, rules)
            if cls is None:
                continue
            sr = _norm_pct_cell(ycell)
            if sr <= 0:
                continue
            rows_by_class.setdefault(cls, []).append((fuel, app, sr))
            class_sums[cls] = class_sums.get(cls, 0.0) + sr

    out_dir = args.out
    if args.format == "csv" and out_dir is None:
        print("--format csv requires --out <directory>", file=sys.stderr)
        return 2

    if out_dir is not None:
        _write_csv_out(Path(out_dir), iso3, factors, pollutant_outputs, M, rows_by_class, class_sums)
        print(f"Wrote CSV tables under {out_dir.resolve()}", file=sys.stderr)

    if args.format == "text":
        _emit_text(iso3, gains_path, year_col, factors, pollutant_outputs, M, rows_by_class, class_sums)
    elif args.format == "latex":
        _emit_latex(iso3, gains_path, year_col, factors, pollutant_outputs, M, class_sums)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
