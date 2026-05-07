"""Check rho diagnostics in [0,1] for wide or long pipeline outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import re


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("weights_csv", type=Path)
    args = p.parse_args()
    if not args.weights_csv.is_file():
        print(f"Missing: {args.weights_csv}", file=sys.stderr)
        return 1
    df = pd.read_csv(args.weights_csv)
    cols = []
    if "rho_weighted" in df.columns:
        cols.append("rho_weighted")
    cols.extend([c for c in df.columns if re.fullmatch(r"RHO_[A-Za-z0-9_]+", c)])
    if not cols:
        print("No rho columns in CSV; skip.")
        return 0
    for c in cols:
        lo, hi = df[c].min(), df[c].max()
        if lo < -1e-9 or hi > 1.0 + 1e-9:
            print(f"FAILED: {c} out of [0,1]: min={lo} max={hi}")
            return 1
    print("OK: rho columns in [0,1].")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
