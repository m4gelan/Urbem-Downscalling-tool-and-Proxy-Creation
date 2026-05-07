"""CLI entry when run as ``python Agriculture/main.py`` from project root."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Agriculture.core.pipeline import run_pipeline


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()
