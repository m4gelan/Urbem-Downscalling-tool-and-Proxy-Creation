#!/usr/bin/env python3
"""
Print J_Waste CEIP / reported family shares (w_solid, w_ww, w_res) and audit rows
by country and pollutant, using the same merge + :func:`PROXY.core.alpha.load_ceip_and_alpha`
as the pipeline.

Example::

  python -m PROXY.tools.diagnose_waste_ceip --iso3 GRC
  python -m PROXY.tools.diagnose_waste_ceip --iso3 ALL --csv OUTPUT/j_waste_ceip_wide.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> int:
    root = _root()
    ap = argparse.ArgumentParser(
        description="J_Waste: show reported-emissions -> family weight table (CEIP / alphas)."
    )
    ap.add_argument(
        "--config",
        type=Path,
        default=root / "PROXY" / "config" / "paths.yaml",
        help="PROXY paths.yaml",
    )
    ap.add_argument(
        "--country",
        default="EL",
        help="NUTS merge country for merge_waste_pipeline_cfg (e.g. EL for Greece).",
    )
    ap.add_argument(
        "--iso3",
        default="GRC",
        help="Show only this country_iso3, or ALL for all countries (default: GRC).",
    )
    ap.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Write the displayed table to this CSV path.",
    )
    args = ap.parse_args()

    cfg_p = args.config if args.config.is_absolute() else root / args.config
    if not cfg_p.is_file():
        print(f"Config not found: {cfg_p}", file=sys.stderr)
        return 1

    import pandas as pd

    from PROXY.core.alpha import load_ceip_and_alpha
    from PROXY.core.dataloaders import load_path_config, load_yaml
    from PROXY.core.dataloaders.discovery import discover_cams_emissions, discover_corine
    from PROXY.sectors.J_Waste.pipeline import merge_waste_pipeline_cfg

    pcfg = load_path_config(cfg_p)
    pcfg.resolved["proxy_common"]["corine_tif"] = str(
        discover_corine(
            root,
            Path(pcfg.resolved["proxy_common"]["corine_tif"]),
        )
    )
    pcfg.resolved["emissions"]["cams_2019_nc"] = str(
        discover_cams_emissions(
            root,
            Path(pcfg.resolved["emissions"]["cams_2019_nc"]),
        )
    )

    sectors_data = load_yaml(root / "PROXY" / "config" / "sectors.yaml")
    entry = next(
        (e for e in (sectors_data.get("sectors") or []) if str(e.get("key")) == "J_Waste"),
        None,
    )
    if entry is None:
        print("J_Waste not listed in PROXY/config/sectors.yaml", file=sys.stderr)
        return 1
    sector_cfg = load_yaml(root / str(entry["config"]))
    out_dir = (root / str(sector_cfg["output_dir"])).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    merged = merge_waste_pipeline_cfg(
        root,
        pcfg.resolved,
        sector_cfg,
        country=str(args.country).strip() or "EL",
        output_dir=out_dir,
    )
    merged["_project_root"] = root
    _alpha, _fb, wide = load_ceip_and_alpha(
        merged,
        None,
        sector_key="J_Waste",
        focus_country_iso3=str(args.iso3).strip().upper()
        if str(args.iso3).strip().upper() != "ALL"
        else None,
    )
    audit_path = Path(merged["paths"].get("alpha_method_audit_dir") or out_dir) / "J_Waste_alpha_method_audit.csv"
    audit = None
    if audit_path.is_file():
        audit = pd.read_csv(audit_path)

    wide = wide.rename(
        columns={
            "alpha_G1": "w_solid",
            "alpha_G2": "w_ww",
            "alpha_G3": "w_res",
        }
    )

    want = [
        "country_iso3",
        "pollutant",
        "w_solid",
        "w_ww",
        "w_res",
        "method",
    ]
    cols = [c for c in want if c in wide.columns]
    view = wide[cols].copy()
    iso = str(args.iso3).strip().upper()
    if iso != "ALL" and "country_iso3" in view.columns:
        view = view[view["country_iso3"].astype(str).str.strip().str.upper() == iso]
    sort_keys = [k for k in ("country_iso3", "pollutant") if k in view.columns]
    if sort_keys:
        view = view.sort_values(sort_keys, kind="mergesort")

    pd.set_option("display.max_rows", 500)
    pd.set_option("display.width", 220)
    print("=== J_Waste: reported emissions -> w_solid, w_ww, w_res (same as pipeline) ===\n")
    if iso != "ALL":
        print(f"Filter country_iso3 = {iso!r}  (merge --country {str(args.country)!r})\n")
    if view.empty:
        print("No rows (check --iso3; workbook may use a different code or missing emissions).")
    else:
        print(view.to_string(index=False))
    pols = [str(p) for p in (merged.get("cams") or {}).get("pollutants_nc") or []]
    if pols and iso != "ALL" and "pollutant" in wide.columns:
        def _np(x: str) -> str:
            return str(x).upper().replace(".", "_").strip()

        have = {
            _np(x)
            for x in wide[wide["country_iso3"].astype(str).str.upper() == iso]["pollutant"]
        }
        miss = [p for p in pols if _np(p) not in have]
        if miss:
            print(
                f"\nNote: no reported row for {iso!r} and pollutants {miss}; "
                "pipeline uses default w=(1/3,1/3,1/3) for those."
            )
    print()
    if audit is not None and len(audit) > 0:
        print(f"Long-form alpha method audit rows: {len(audit)}")
    if args.csv is not None:
        outp = args.csv if args.csv.is_absolute() else root / args.csv
        outp.parent.mkdir(parents=True, exist_ok=True)
        view.to_csv(outp, index=False)
        print(f"Wrote {outp.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
