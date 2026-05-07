"""
Print distinct SURVEY_LC1 / SURVEY_LU1 (and optional LC2/LU2) from a LUCAS CSV,
plus LC1 codes that have no entry in aux_lucas_lc1_mapping.

Usage (from project root):
  python Agriculture/Auxiliaries/print_lucas_survey_distinct.py
  python Agriculture/Auxiliaries/print_lucas_survey_distinct.py --csv data/Agriculture/EU_LUCAS_2022.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from Agriculture.config import default_config_path, load_agriculture_config
from Agriculture.core import resolve_path
from Agriculture.aux_lucas_lc1_mapping import crop_category_from_lc1


def _norm(s: pd.Series) -> pd.Series:
    t = s.astype(str).str.strip().str.upper()
    return t.replace({"NAN": "", "NONE": "", "NAT": ""})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=None, help="LUCAS CSV path")
    p.add_argument("--top", type=int, default=100, help="Max rows in value_counts to print")
    args = p.parse_args()

    cfg = load_agriculture_config(default_config_path())
    root = Path(cfg.get("project_root") or _ROOT)
    lb = cfg.get("lucas_build") or {}
    csv_path = args.csv
    if csv_path is None:
        rel = lb.get("lucas_data") or lb.get("lucas_xlsx")
        if not rel:
            rel = "data/Agriculture/EU_LUCAS_2022.csv"
        csv_path = resolve_path(root, rel)

    if not csv_path.is_file():
        raise SystemExit(f"File not found: {csv_path}")

    use = ["SURVEY_LC1", "SURVEY_LU1", "SURVEY_LC2", "SURVEY_LU2", "SURVEY_GRAZING"]
    df = pd.read_csv(csv_path, usecols=lambda c: c in use, low_memory=False)

    for col in use:
        if col not in df.columns:
            continue
        s = _norm(df[col])
        vc = s[s.str.len() > 0].value_counts()
        print("=" * 72)
        print(f"{col}: {len(vc)} distinct codes, {int((s.str.len() > 0).sum())} non-empty rows")
        print(vc.head(args.top).to_string())
        if len(vc) > args.top:
            print(f"... ({len(vc) - args.top} more codes; increase --top to list them)")

    if "SURVEY_LC1" in df.columns:
        lc1 = _norm(df["SURVEY_LC1"])
        uniq = sorted(lc1[lc1.str.len() > 0].unique())
        unmapped = [c for c in uniq if crop_category_from_lc1(c) is None]
        mapped = [c for c in uniq if crop_category_from_lc1(c) is not None]
        print("=" * 72)
        print(f"SURVEY_LC1: {len(uniq)} distinct codes total")
        print(f"  Mapped to crop_category (synthetic N bucket): {len(mapped)}")
        print(f"  Unmapped (fall back to CLC-only proxies in pipeline): {len(unmapped)}")
        print("All distinct LC1 codes (sorted):")
        print(",".join(uniq))
        print()
        print("Unmapped LC1 codes with row counts:")
        for c in unmapped:
            n = int((lc1 == c).sum())
            print(f"  {c!r}\t{n}")
    if 'SURVEY_GRAZING' in df.columns:
        grazing = _norm(df["SURVEY_GRAZING"])
        grazing_vc = grazing.value_counts()
        print("=" * 72)
        print(f"SURVEY_GRAZING: {len(grazing_vc)} distinct codes total")
        print(grazing_vc.head(args.top).to_string())
        if len(grazing_vc) > args.top:
            print(f"... ({len(grazing_vc) - args.top} more codes; increase --top to list them)")
    crosstab = pd.crosstab(
    df["SURVEY_GRAZING"],
    df["SURVEY_LC1"].str[:1],
    dropna=False
    )
    print(crosstab)
    print(df["SURVEY_LC2"].value_counts(dropna=False).head(30)
)
    print(df["SURVEY_LU1"].value_counts().head(30))
    # Cross-tab: LU1 vs GRAZING presence
    ct = pd.crosstab(df["SURVEY_LU1"], df["SURVEY_GRAZING"].fillna("no_data"))
    print(ct.head(20))
    # LC1 vs grazing
    ct2 = pd.crosstab(df["SURVEY_LC1"], df["SURVEY_GRAZING"].fillna("no_data"))
    print(ct2)
    # What LC1 codes appear where grazing=1?
    grazing_pts = df[df["SURVEY_GRAZING"] == 1.0]
    print("LC1 where grazing=1:")
    print(grazing_pts["SURVEY_LC1"].value_counts().head(20))
    print("\nLU1 where grazing=1:")
    print(grazing_pts["SURVEY_LU1"].value_counts().head(20))
    # What LU1 codes appear where grazing=1?
    # What is U420?
    print(df[df["SURVEY_LU1"] == "U420"]["SURVEY_LC1"].value_counts().head(10))

    # Grazing rates by LC1 (top categories)
    for lc in ["E10", "E20", "E30", "B55", "C10", "D20"]:
        sub = df[df["SURVEY_LC1"] == lc]
        g = (sub["SURVEY_GRAZING"] == 1.0).sum()
        total = len(sub)
        surveyed = sub["SURVEY_GRAZING"].notna().sum()
        print(f"{lc}: {g}/{surveyed} surveyed = {100*g/surveyed:.1f}% grazed" if surveyed else f"{lc}: no data")




if __name__ == "__main__":
    main()
