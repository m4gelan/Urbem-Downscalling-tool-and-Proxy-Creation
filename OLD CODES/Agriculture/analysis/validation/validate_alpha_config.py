"""Check alpha.config.json: process rows and rough sum-to-one per pollutant."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("alpha_json", type=Path, nargs="?", default=None)
    args = p.parse_args()
    path = args.alpha_json or Path(__file__).resolve().parents[2] / "config" / "alpha.config.json"
    if not path.is_file():
        print(f"Missing: {path}", file=sys.stderr)
        return 1
    data = json.loads(path.read_text(encoding="utf-8"))
    pol = data.get("pollutants") or {}
    atol = 0.02
    for name, rows in pol.items():
        if not isinstance(rows, list):
            continue
        s = sum(float(r.get("alpha", 0)) for r in rows if isinstance(r, dict))
        if abs(s - 1.0) > atol:
            print(f"WARNING: {name} alpha sum = {s:.6f} (expected ~1, tol={atol})")
    print("OK: alpha file readable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
