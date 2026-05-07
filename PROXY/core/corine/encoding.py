"""CORINE pixel encoding and Level-3 class lookup helpers.

Inputs are CORINE pixel arrays plus a declared encoding (`l3_code` or
`eea44_index`) and the configured class-map YAML. Outputs are Level-3 CLC
pixels and small indicator masks used by sector proxy builders. This module is
pure array/legend logic; raster IO and warping live in `PROXY.core.corine.raster`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from PROXY.core.dataloaders.config import load_yaml, resolve_path


def default_corine_index_map_path() -> Path:
    """Built-in EEA-style 1..N class index -> Level-3 CLC map."""
    return Path(__file__).resolve().parents[2] / "config" / "corine" / "eea44_index_to_l3.yaml"


def load_ordered_l3_codes(path: Path) -> np.ndarray:
    """Load ``l3_codes`` from YAML: list position ``n`` is the L3 code for raster value ``n + 1``."""
    data = load_yaml(path)
    codes = data.get("l3_codes")
    if not isinstance(codes, list) or not codes:
        raise ValueError(f"{path}: expected non-empty list 'l3_codes'")
    int_codes = [int(x) for x in codes]
    classes = data.get("classes")
    if isinstance(classes, list) and classes:
        if len(classes) != len(int_codes):
            raise ValueError(
                f"{path}: classes length {len(classes)} != l3_codes length {len(int_codes)}"
            )
        for i, row in enumerate(classes):
            if not isinstance(row, dict):
                raise ValueError(f"{path}: classes[{i}] must be a mapping")
            idx = int(row.get("index", -1))
            l3 = int(row.get("l3", -1))
            if idx != i + 1:
                raise ValueError(f"{path}: classes[{i}].index expected {i + 1}, got {idx}")
            if l3 != int_codes[i]:
                raise ValueError(f"{path}: classes[{i}].l3 expected {int_codes[i]}, got {l3}")
    out = np.asarray(int_codes, dtype=np.float32)
    return out


@lru_cache(maxsize=16)
def _ordered_l3_lut_cached(resolved_path: str) -> np.ndarray:
    return load_ordered_l3_codes(Path(resolved_path))


def resolve_corine_l3_lut(
    *,
    repo_root: Path | None,
    pixel_value_map: str | Path | None,
) -> np.ndarray:
    """Resolve which YAML defines raster class index -> Level-3 CLC order."""
    if pixel_value_map is not None and str(pixel_value_map).strip():
        p = Path(pixel_value_map)
        if not p.is_absolute():
            if repo_root is None:
                raise ValueError("repo_root is required when pixel_value_map is a relative path")
            p = resolve_path(repo_root, pixel_value_map)
    else:
        p = default_corine_index_map_path()
    if not p.is_file():
        raise FileNotFoundError(f"CORINE class map not found: {p}")
    return _ordered_l3_lut_cached(str(p.resolve()))


def normalized_corine_pixel_encoding(encoding: str | None) -> str:
    s = (encoding or "l3_code").strip().lower()
    if s in ("l3", "l3_code", "physical", "clc_code"):
        return "l3_code"
    if s in ("eea44", "eea44_index", "index_44", "corine_44", "8bit", "indexed"):
        return "eea44_index"
    raise ValueError(
        f"Unknown corine pixel_encoding {encoding!r}; use 'l3_code' or 'eea44_index'."
    )


def decode_corine_to_l3_pixels(
    clc: np.ndarray,
    encoding: str | None,
    *,
    repo_root: Path | None = None,
    pixel_value_map: str | Path | None = None,
) -> np.ndarray:
    """
    Return float32 Level-3 CLC per cell. For ``eea44_index``, invalid indices are NaN.

    ``pixel_value_map`` overrides the default YAML (path relative to ``repo_root`` if not absolute).
    """
    mode = normalized_corine_pixel_encoding(encoding)
    a = np.asarray(clc, dtype=np.float64)
    if mode == "l3_code":
        return a.astype(np.float32, copy=False)
    lut = resolve_corine_l3_lut(repo_root=repo_root, pixel_value_map=pixel_value_map)
    nmax = int(lut.size)
    out = np.full(a.shape, np.nan, dtype=np.float32)
    finite = np.isfinite(a)
    z = np.zeros(a.shape, dtype=np.int64)
    z[finite] = np.rint(np.clip(a[finite], -1e9, 1e9)).astype(np.int64, copy=False)
    m = finite & (z >= 1) & (z <= nmax)
    out[m] = lut[z[m] - 1]
    return out


def build_clc_indicators(
    clc: np.ndarray, c111: int, c112: int, c121: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Binary masks for three Level-3 CLC codes (typical urban morphology)."""
    z = np.rint(clc).astype(np.int32)
    u111 = (z == int(c111)).astype(np.float32)
    u112 = (z == int(c112)).astype(np.float32)
    u121 = (z == int(c121)).astype(np.float32)
    return u111, u112, u121


__all__ = [
    "build_clc_indicators",
    "decode_corine_to_l3_pixels",
    "default_corine_index_map_path",
    "load_ordered_l3_codes",
    "normalized_corine_pixel_encoding",
    "resolve_corine_l3_lut",
]
