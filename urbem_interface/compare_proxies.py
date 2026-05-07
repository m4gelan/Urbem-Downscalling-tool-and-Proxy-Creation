"""Shim: use ``python -m urbem_interface.tools.compare_proxies``."""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from urbem_interface.tools.compare_proxies import main

if __name__ == "__main__":
    raise SystemExit(main())
