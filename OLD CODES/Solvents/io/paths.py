from __future__ import annotations

from pathlib import Path


def resolve_path(root: Path, p: str | Path) -> Path:
    x = Path(p)
    return x if x.is_absolute() else (root / x)
