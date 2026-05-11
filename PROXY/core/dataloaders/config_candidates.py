"""Load YAML/JSON sector base configs from ordered path candidates (canonical first)."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import yaml


def load_first_existing_yaml_or_json(
    candidates: list[Path],
    *,
    context: str,
) -> dict[str, Any]:
    """Return the first existing file in ``candidates`` as a dict.

    If index ``i > 0`` is used, emits :exc:`DeprecationWarning` naming the canonical
    ``candidates[0]`` path.

    Raises
    ------
    FileNotFoundError
        If no candidate path exists.
    """
    chosen_idx: int | None = None
    chosen_path: Path | None = None
    for i, p in enumerate(candidates):
        if p.is_file():
            chosen_idx, chosen_path = i, p
            break
    if chosen_idx is None or chosen_path is None:
        joined = ", ".join(str(p) for p in candidates)
        raise FileNotFoundError(f"{context}: none of the candidate files exist: {joined}")
    if chosen_idx > 0:
        warnings.warn(
            f"{context}: loaded legacy config from {chosen_path}; "
            f"prefer canonical file {candidates[0]}",
            DeprecationWarning,
            stacklevel=2,
        )
    p = chosen_path
    if p.suffix.lower() in (".yaml", ".yml"):
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    with p.open(encoding="utf-8") as f:
        return json.load(f)
