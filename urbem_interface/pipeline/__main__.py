"""python -m urbem_interface.pipeline -- print design options and/or proxy report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from urbem_interface.pipeline import DESIGN_OPTIONS_SUMMARY
from urbem_interface.pipeline.report import report_from_proxies_json


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="UrbEm proxy pipeline tooling.")
    p.add_argument(
        "--design-options",
        action="store_true",
        help="Print Topic 1-3 design options (and Topic 4 pointer) for you to choose.",
    )
    p.add_argument(
        "--proxies-json",
        type=Path,
        default=None,
        help="Path to proxies.json (e.g. urbem_interface/config/proxies.json).",
    )
    p.add_argument(
        "--proxies-folder",
        type=Path,
        default=None,
        help="If set, check that proxy files exist under this folder.",
    )
    args = p.parse_args(argv)

    if args.design_options or not args.proxies_json:
        print(DESIGN_OPTIONS_SUMMARY)
        print()

    if args.proxies_json:
        path = args.proxies_json.resolve()
        if not path.is_file():
            print(f"Not found: {path}", file=sys.stderr)
            return 1
        text = report_from_proxies_json(path, proxies_folder=args.proxies_folder)
        print(text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
