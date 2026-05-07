"""Categorical dominant-group overlay for multi-proxy sectors.

The user feedback on B_Industry / D_Fugitive (and friends) was:

    > "weitghs everywhere [...] think of a better way to represent group proxy,
    > maybe 1 layer, different color for dataset"

Stacking four semi-transparent P_g overlays (G1..G4) on top of each other only
tells you where each group is, but it does not tell you *which* group dominates
a given pixel - which is really the question the modeller asks. This module
turns a dict of per-group scalar arrays into a single categorical raster where
each pixel is painted with the colour of its dominant group, plus a tie /
"no data" class for pixels where no group has positive contribution.

Public API:
  * :func:`compute_dominance_rgba` - argmax over group arrays, produces RGBA.
  * :func:`dominance_legend_section` - build a :class:`LegendSection` for the unified legend.
  * :data:`DEFAULT_GROUP_COLORS` - four-color palette reused across sectors.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

from PROXY.visualization._color_debug import viz_color_log
from PROXY.visualization._legend import (
    OVERLAY_RGBA_ALPHA,
    LegendSection,
    categorical_swatch_html,
)

DEFAULT_GROUP_COLORS: dict[str, str] = {
    "G1": "#1f77b4",
    "G2": "#d62728",
    "G3": "#2ca02c",
    "G4": "#ff7f0e",
}


def compute_dominance_rgba(
    groups: Mapping[str, np.ndarray],
    *,
    colors: Mapping[str, str] | None = None,
    min_positive: float = 0.0,
    alpha: int = OVERLAY_RGBA_ALPHA,
) -> tuple[np.ndarray, list[str]] | None:
    """Return an RGBA dominance map plus the list of group ids that were used.

    Parameters
    ----------
    groups:
        Mapping from group id (e.g. ``"G1"``) to a 2D scalar array in view space.
        Arrays must share the same shape. Non-finite / non-positive entries are
        treated as "absent".
    colors:
        Hex colour per group id; falls back to :data:`DEFAULT_GROUP_COLORS`.
    min_positive:
        A pixel is considered "dominated" by a group only if that group's value
        is strictly greater than ``min_positive``.
    alpha:
        Alpha byte (0..255) for pixels that have a dominant group.

    Returns
    -------
    ``None`` when no group provided any positive pixel; otherwise
    ``(rgba, gids_in_draw_order)``.
    """
    import matplotlib.colors as mcolors

    palette = dict(colors or DEFAULT_GROUP_COLORS)
    gids = [g for g in groups if groups[g] is not None]
    if not gids:
        return None
    shapes = {groups[g].shape for g in gids}
    if len(shapes) != 1:
        return None
    h, w = next(iter(shapes))

    viz_color_log(
        "dominance_rgba_start",
        module="PROXY.visualization._dominance",
        function="compute_dominance_rgba",
        alpha=int(alpha),
        min_positive=float(min_positive),
        group_ids_order=list(gids),
        palette_hex={g: palette.get(g, "#888888") for g in gids},
    )

    stack = np.full((len(gids), h, w), -np.inf, dtype=np.float32)
    for i, g in enumerate(gids):
        arr = np.asarray(groups[g], dtype=np.float32)
        ok = np.isfinite(arr) & (arr > float(min_positive))
        stack[i][ok] = arr[ok]

    any_pos = np.any(np.isfinite(stack) & (stack > -np.inf), axis=0)
    if not np.any(any_pos):
        return None

    arg = np.argmax(stack, axis=0)
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    used: list[str] = []
    for i, g in enumerate(gids):
        hex_ = palette.get(g, "#888888")
        r, gb, bb = mcolors.to_rgb(hex_)
        m = any_pos & (arg == i)
        if not np.any(m):
            continue
        used.append(g)
        rgba[m, 0] = int(r * 255)
        rgba[m, 1] = int(gb * 255)
        rgba[m, 2] = int(bb * 255)
        rgba[m, 3] = int(alpha)
        viz_color_log(
            "dominance_class_rgba",
            group_id=g,
            stack_index=i,
            hex=hex_,
            matplotlib_rgb_float=(float(r), float(gb), float(bb)),
            rgba_uint8=(int(r * 255), int(gb * 255), int(bb * 255), int(alpha)),
            pixel_count=int(np.count_nonzero(m)),
        )
    if not used:
        return None
    viz_color_log("dominance_rgba_done", used_group_ids_order=list(used), shape=[int(h), int(w)])
    return rgba, used


def dominance_legend_section(
    used_gids: list[str],
    *,
    colors: Mapping[str, str] | None = None,
    sector_label: str = "group",
    gid_labels: Mapping[str, str] | None = None,
    swatch_dark_overlay_alpha: float | None = OVERLAY_RGBA_ALPHA / 255.0,
    swatch_dark_base: str = "#2a2a2a",
) -> LegendSection:
    """Build a categorical legend entry for a dominance overlay."""
    palette = dict(colors or DEFAULT_GROUP_COLORS)
    labels = dict(gid_labels or {})
    rows: list[tuple[str, str]] = []
    for g in used_gids:
        rows.append((labels.get(g, g), palette.get(g, "#888888")))
    viz_color_log(
        "dominance_legend_section_built",
        sector_label=sector_label,
        used_gids=list(used_gids),
        legend_rows=[{"gid": g, "label": labels.get(g, g), "hex": palette.get(g, "#888888")} for g in used_gids],
        swatch_dark_overlay_alpha=swatch_dark_overlay_alpha,
        swatch_dark_base=swatch_dark_base,
    )
    rule = (
        "largest normalized family score at each pixel"
        if str(sector_label).lower() == "family"
        else "largest P_g value at each pixel"
    )
    layer_hint = (
        "Turn off the per-dataset layers in the layer control to read this overlay clearly."
        if str(sector_label).lower() == "family"
        else "Turn off the individual P_g layers in the layer control to read this one clearly."
    )
    body = (
        '<p class="pl-hint">Each pixel is painted with the '
        f"<b>dominant {sector_label}</b> ({rule}). "
        'Gray / transparent pixels have no positive contribution from any group. '
        f"{layer_hint} "
        "Swatches approximate overlay colour on a dark basemap.</p>"
        + categorical_swatch_html(
            rows,
            cols=1,
            dark_overlay_alpha=swatch_dark_overlay_alpha,
            dark_overlay_base=swatch_dark_base,
        )
    )
    return LegendSection(title=f"Dominant {sector_label}", html=body, open=True)
