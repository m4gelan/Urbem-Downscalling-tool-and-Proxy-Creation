"""CLI: python -m Solvents [--config path/to.json]"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description="GNFR E solvent area downscaling weights")
    ap.add_argument(
        "--config",
        type=Path,
        default=root / "Solvents" / "config" / "solvents.defaults.json",
        help="JSON config (default: Solvents/config/solvents.defaults.json)",
    )
    ns = ap.parse_args(argv)
    cfg_path: Path = ns.config
    from .core.pipeline import load_json_config, run_solvents_area_pipeline

    cfg = load_json_config(root, cfg_path)
    out = run_solvents_area_pipeline(root, cfg, config_path=cfg_path)
    print(out["output_tif"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
