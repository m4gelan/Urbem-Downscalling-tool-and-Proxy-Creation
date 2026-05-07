"""
CORINE Level-3 **urban morphology** multipliers ``mr`` (residential) and ``mc`` (commercial).

**Inputs**: binary masks ``u111,u112,u121`` for configured L3 codes plus YAML weights.
**Outputs**: per-pixel weights in [0, ~max weight] applied to ``R_base`` / ``C_base`` in ``stack``.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def morph_residential(u111: np.ndarray, u112: np.ndarray, u121: np.ndarray, cfg: dict[str, float]) -> np.ndarray:
    one = np.ones_like(u111, dtype=np.float32)
    other = one - u111 - u112 - u121
    other = np.clip(other, 0.0, 1.0)
    return (
        float(cfg["w111"]) * u111
        + float(cfg["w112"]) * u112
        + float(cfg["w_other"]) * other
    ).astype(np.float32)


def morph_commercial(u111: np.ndarray, u112: np.ndarray, u121: np.ndarray, cfg: dict[str, float]) -> np.ndarray:
    one = np.ones_like(u111, dtype=np.float32)
    other = one - u111 - u112 - u121
    other = np.clip(other, 0.0, 1.0)
    return (
        float(cfg["w111"]) * u111
        + float(cfg["w121"]) * u121
        + float(cfg["w_other"]) * other
    ).astype(np.float32)


def morphology_masks_from_clc(
    clc_l3: np.ndarray,
    *,
    urban_111: int,
    urban_112: int,
    urban_121: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from PROXY.core.corine.encoding import build_clc_indicators

    clc_for_ind = np.where(np.isfinite(clc_l3), clc_l3, -1.0).astype(np.float32)
    return build_clc_indicators(clc_for_ind, int(urban_111), int(urban_112), int(urban_121))
