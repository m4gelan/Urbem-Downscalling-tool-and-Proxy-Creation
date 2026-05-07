from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)


def read_excel(path: Path, sheet_name: str | int | None = 0, **kwargs) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=sheet_name, **kwargs)

