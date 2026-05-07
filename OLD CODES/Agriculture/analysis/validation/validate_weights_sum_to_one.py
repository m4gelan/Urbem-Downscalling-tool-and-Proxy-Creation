"""Check that pollutant weights sum to 1 within each NUTS-2."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import re


def _is_long_format(df: pd.DataFrame) -> bool:
    return {"pollutant", "NUTS_ID", "w_p"}.issubset(df.columns)


def _weight_cols_wide(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if re.fullmatch(r"W_[A-Za-z0-9_]+", c)]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("weights_csv", type=Path, help="weights_long.csv or weights_wide.csv")
    p.add_argument("--atol", type=float, default=1e-5)
    args = p.parse_args()
    if not args.weights_csv.is_file():
        print(f"Missing file: {args.weights_csv}", file=sys.stderr)
        return 1
    df = pd.read_csv(args.weights_csv)
    bad = []
    if _is_long_format(df):
        for pollutant, sub in df.groupby("pollutant", sort=False):
            s = sub.groupby("NUTS_ID")["w_p"].sum()
            for nid, tot in s.items():
                if pd.isna(tot) or abs(float(tot) - 1.0) > args.atol:
                    bad.append((str(pollutant), nid, tot))
    else:
        for col in _weight_cols_wide(df):
            s = df.groupby("NUTS_ID")[col].sum()
            for nid, tot in s.items():
                if pd.isna(tot) or abs(float(tot) - 1.0) > args.atol:
                    bad.append((col, nid, tot))
    if bad:
        print("FAILED:", bad[:20], "..." if len(bad) > 20 else "")
        return 1
    print("OK: weights sum to 1 per NUTS (checked detected pollutant columns).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
