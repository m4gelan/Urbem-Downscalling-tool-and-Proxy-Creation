"""Per-family categorical overlays for multi-dataset sectors (E / I / J).

Context:
    Sectors like E_Solvents and J_Waste combine several heterogeneous
    datasets ("treatment plants", "agglomerations", "imperviousness", ...)
    into a single emission proxy. Showing one overlay per dataset drowns
    the map, but showing only the final proxy hides *which* dataset
    contributed where. The user asked for one overlay per **family** of
    datasets (e.g. wastewater = {treatment_plants, agglomerations,
    imperviousness, ...}), with a distinct colour per dataset inside that
    family and a matching legend.

This module implements the common pieces:

  * :func:`family_argmax_rgba` - given ``{dataset: scalar_array}`` and
    ``{dataset: hex_color}``, paint each pixel with the colour of the
    dataset that dominates locally (argmax on percentile-normalised values).
  * :func:`family_legend_section` - build a :class:`LegendSection` listing
    the colour swatches for the datasets that actually have pixels.
  * :data:`FAMILY_COLORS` - 10-colour tab10 palette used as fallback.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from PROXY.visualization._color_debug import viz_color_log
from PROXY.visualization._legend import (
    OVERLAY_RGBA_ALPHA,
    LegendSection,
    categorical_swatch_html,
)

FAMILY_COLORS: tuple[str, ...] = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
)


def _percentile_normalise(arr: np.ndarray) -> np.ndarray:
    """Map positive values to ``[0, 1]`` using 2/98 percentiles; non-positives -> ``-inf``.

    Using a robust 2/98 clip means that a noisy tail in one dataset does not
    make a family-mate's mild signal invisible in the argmax.
    """
    a = np.asarray(arr, dtype=np.float32)
    out = np.full_like(a, -np.inf, dtype=np.float32)
    ok = np.isfinite(a) & (a > 0.0)
    if not np.any(ok):
        return out
    pos = a[ok]
    lo = float(np.percentile(pos, 2.0))
    hi = float(np.percentile(pos, 98.0))
    if hi <= lo:
        hi = lo + 1e-9
    v = (pos - lo) / (hi - lo)
    out[ok] = np.clip(v, 0.0, 1.0)
    return out


def family_argmax_rgba(
    datasets: Mapping[str, np.ndarray],
    colors: Mapping[str, str],
    *,
    alpha: int = OVERLAY_RGBA_ALPHA,
) -> tuple[np.ndarray, list[str], np.ndarray] | None:
    """Return ``(rgba, used_dataset_keys, family_score_array)``.

    ``family_score_array`` is the per-pixel max of the normalised datasets
    (same shape as any input, ``nan`` where no dataset is positive). The
    caller can feed this to :func:`PROXY.visualization._dominance.compute_dominance_rgba`
    to get a cross-family dominance overlay without re-computing percentiles.
    """
    import matplotlib.colors as mcolors

    keys = [k for k, v in datasets.items() if v is not None]
    if not keys:
        return None
    shapes = {datasets[k].shape for k in keys}
    if len(shapes) != 1:
        return None
    h, w = next(iter(shapes))

    viz_color_log(
        "family_argmax_start",
        module="PROXY.visualization._family",
        function="family_argmax_rgba",
        alpha=int(alpha),
        dataset_keys_order=list(keys),
        color_map_hex={k: colors.get(k, FAMILY_COLORS[i % len(FAMILY_COLORS)]) for i, k in enumerate(keys)},
        missing_color_keys=[k for k in keys if k not in colors],
    )

    stack = np.full((len(keys), h, w), -np.inf, dtype=np.float32)
    for i, k in enumerate(keys):
        stack[i] = _percentile_normalise(datasets[k])

    any_pos = np.any(np.isfinite(stack) & (stack > -np.inf), axis=0)
    if not np.any(any_pos):
        return None
    arg = np.argmax(stack, axis=0)
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    used: list[str] = []
    for i, k in enumerate(keys):
        hex_ = colors.get(k, FAMILY_COLORS[i % len(FAMILY_COLORS)])
        r, g, b = mcolors.to_rgb(hex_)
        m = any_pos & (arg == i)
        if not np.any(m):
            continue
        used.append(k)
        rgba[m, 0] = int(r * 255)
        rgba[m, 1] = int(g * 255)
        rgba[m, 2] = int(b * 255)
        rgba[m, 3] = int(alpha)
        n_px = int(np.count_nonzero(m))
        viz_color_log(
            "family_argmax_class_rgba",
            dataset_key=k,
            stack_index=i,
            hex=hex_,
            matplotlib_rgb_float=(float(r), float(g), float(b)),
            rgba_uint8=(int(r * 255), int(g * 255), int(b * 255), int(alpha)),
            pixel_count=n_px,
        )

    score = np.where(any_pos, np.max(stack, axis=0), np.nan)
    viz_color_log(
        "family_argmax_done",
        used_dataset_keys_order=list(used),
        shape=[int(h), int(w)],
    )
    return rgba, used, score


def family_legend_section(
    family_label: str,
    used_keys: Sequence[str],
    colors: Mapping[str, str],
    *,
    key_labels: Mapping[str, str] | None = None,
    open_: bool = True,
    extra_html: str = "",
    swatch_dark_overlay_alpha: float | None = OVERLAY_RGBA_ALPHA / 255.0,
    swatch_dark_base: str = "#2a2a2a",
) -> LegendSection:
    """Build a categorical legend entry for a single family overlay."""
    labels = dict(key_labels or {})
    rows: list[tuple[str, str]] = []
    for i, k in enumerate(used_keys):
        hex_ = colors.get(k, FAMILY_COLORS[i % len(FAMILY_COLORS)])
        rows.append((labels.get(k, k), hex_))
    viz_color_log(
        "family_legend_section_built",
        family_label=family_label,
        used_keys=list(used_keys),
        legend_rows=[{"key": k, "label": labels.get(k, k), "hex": colors.get(k, FAMILY_COLORS[j % len(FAMILY_COLORS)])} for j, k in enumerate(used_keys)],
        swatch_dark_overlay_alpha=swatch_dark_overlay_alpha,
        swatch_dark_base=swatch_dark_base,
    )
    body = (
        f'<p class="pl-hint">Each pixel is painted with the colour of the '
        f"<b>dominant dataset</b> within the <i>{family_label}</i> family. "
        "Individual per-dataset rasters stay available in the layer control "
        "(all off by default). Swatches approximate overlay colour on a dark "
        "basemap (semi-transparent pixels).</p>"
        + categorical_swatch_html(
            rows,
            cols=1,
            dark_overlay_alpha=swatch_dark_overlay_alpha,
            dark_overlay_base=swatch_dark_base,
        )
        + (extra_html or "")
    )
    return LegendSection(title=f"{family_label} datasets", html=body, open=bool(open_))


def assign_auto_colors(
    keys: Sequence[str],
    overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return a stable ``{key: hex_color}`` mapping using :data:`FAMILY_COLORS` as fallback."""
    out: dict[str, str] = {}
    overrides = dict(overrides or {})
    fallback_ix = 0
    for k in keys:
        if k in overrides:
            out[k] = overrides[k]
        else:
            out[k] = FAMILY_COLORS[fallback_ix % len(FAMILY_COLORS)]
            fallback_ix += 1
    return out
