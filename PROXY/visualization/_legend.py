"""Unified right-side legend panel for ``PROXY.visualization`` sector previews.

Before this module every sector emitted:
 - a bottom-left HTML block from ``*_legend.py`` (metadata, CORINE codes, hints),
 - one or two floating top-right Branca ``LinearColormap`` strips (weight / pop).

Those two legends drifted (different colours, different captions, some anchored
top-right, some bottom-left) and the floating strips could not be resized or
hidden without editing Folium internals. This module centralises both into a
single collapsible panel pinned to the right edge of the map.

Each caller assembles a list of :class:`LegendSection` items - prose, colormap
swatches, categorical swatches, data tables - and :func:`render_unified_legend`
emits one self-contained HTML fragment with a shared stylesheet. No JS is
required; sections collapse via the native ``<details>`` element so the panel
shrinks when the user does not need it.

Public entry points:
  * :class:`LegendSection` - a titled block of body HTML.
  * :func:`colormap_swatch_html` - continuous-colormap swatch with min/max ticks.
  * :func:`categorical_swatch_html` - discrete swatches with labels (CORINE, group palette).
  * :func:`render_unified_legend` - assemble and return the full panel HTML.
"""
from __future__ import annotations

import html
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from PROXY.visualization._color_debug import viz_color_log
from PROXY.visualization.overlay_utils import sample_cmap_hex

# Alpha byte baked into categorical RGBA overlays (family / dominance layers).
# Legend swatches use this to simulate semi-transparent colour over a dark base.
OVERLAY_RGBA_ALPHA: int = 240


@dataclass(frozen=True)
class LegendSection:
    """A titled block inside the unified legend panel."""

    title: str
    html: str
    open: bool = True


def colormap_swatch_html(
    *,
    cmap: str,
    vmin: float,
    vmax: float,
    caption: str = "",
    n_stops: int = 32,
    label_min: str | None = None,
    label_max: str | None = None,
) -> str:
    """Return an inline continuous-colormap bar with min/max tick labels.

    Used in place of ``branca.colormap.LinearColormap.add_to(fmap)`` which
    otherwise free-floats in the top-right corner of every map.
    """
    stops = sample_cmap_hex(cmap, max(4, int(n_stops)))
    gradient = ", ".join(stops)
    lo = label_min if label_min is not None else _fmt_num(vmin)
    hi = label_max if label_max is not None else _fmt_num(vmax)
    safe_cap = html.escape(caption)
    return f"""
    <div class="pl-cmap">
      <div class="pl-cmap-cap">{safe_cap}</div>
      <div class="pl-cmap-bar" style="background: linear-gradient(to right, {gradient});"></div>
      <div class="pl-cmap-ticks"><span>{html.escape(lo)}</span><span>{html.escape(hi)}</span></div>
    </div>
    """


def _hex_to_rgb(colour: str) -> tuple[int, int, int]:
    c = str(colour).strip().lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    if len(c) != 6:
        return 136, 136, 136
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


def categorical_swatch_html(
    entries: Iterable[tuple[str, str]],
    *,
    cols: int = 1,
    dark_overlay_alpha: float | None = None,
    dark_overlay_base: str = "#2a2a2a",
) -> str:
    """Return a discrete swatch grid: ``entries`` is ``[(label, '#rrggbb'), ...]``.

    When ``dark_overlay_alpha`` is set (0..1), each swatch stacks the colour at
    that opacity on top of ``dark_overlay_base``, approximating how the same RGBA
    pixels read on a dark satellite basemap (Folium image overlay opacity should
    be 1.0 so this matches a single alpha pass).
    """
    items = []
    base_esc = html.escape(dark_overlay_base)
    for label, colour in entries:
        if dark_overlay_alpha is not None:
            r, g, b = _hex_to_rgb(colour)
            a = float(dark_overlay_alpha)
            a = max(0.0, min(1.0, a))
            bg = (
                f"linear-gradient(rgba({r},{g},{b},{a:.5f}), rgba({r},{g},{b},{a:.5f})), "
                f"{base_esc}"
            )
            sw = (
                f'<span class="pl-swatch pl-swatch-blend" style="background:{bg};"></span>'
            )
            viz_color_log(
                "legend_swatch_blended",
                module="PROXY.visualization._legend",
                function="categorical_swatch_html",
                label=str(label),
                hex_input=str(colour),
                hex_parsed_rgb=(r, g, b),
                dark_overlay_alpha=a,
                dark_overlay_base=str(dark_overlay_base),
                css_background=bg[:220],
            )
        else:
            sw = (
                f'<span class="pl-swatch" style="background:{html.escape(colour)};"></span>'
            )
            viz_color_log(
                "legend_swatch_opaque",
                label=str(label),
                hex_input=str(colour),
            )
        items.append(
            f"<li>{sw}"
            f'<span class="pl-swatch-label">{html.escape(label)}</span></li>'
        )
    grid_class = "pl-swatch-grid-2" if int(cols) >= 2 else "pl-swatch-grid"
    return f'<ul class="{grid_class}">{"".join(items)}</ul>'


def render_unified_legend(
    sector_title: str,
    sections: Iterable[LegendSection],
    *,
    pollutant_badge: str | None = None,
    region_note: str | None = None,
) -> str:
    """Assemble a single right-side legend panel from labelled sections.

    ``pollutant_badge`` renders a high-visibility tag at the top of the panel
    ("Active pollutant: NOx"). ``region_note`` shows the focus region (Attica /
    full country / custom bbox) when the caller wants it visible.
    """
    safe_title = html.escape(str(sector_title))
    header_bits: list[str] = [f'<span class="pl-title">{safe_title}</span>']
    if pollutant_badge:
        header_bits.append(
            f'<span class="pl-badge" title="Currently displayed pollutant">'
            f'{html.escape(pollutant_badge)}</span>'
        )
    header_html = "".join(header_bits)

    body_parts: list[str] = []
    if region_note:
        body_parts.append(
            f'<div class="pl-region">{html.escape(region_note)}</div>'
        )
    for sec in sections:
        attr = " open" if sec.open else ""
        body_parts.append(
            f"<details class=\"pl-details\"{attr}>"
            f"<summary>{html.escape(sec.title)}</summary>"
            f"<div class=\"pl-body\">{sec.html}</div>"
            f"</details>"
        )
    body_html = "".join(body_parts)

    return f"""
<details class="proxy-map-legend" open>
  <summary class="pl-root-summary">
    <span class="pl-chev" aria-hidden="true"></span>
    <span class="pl-header">{header_html}</span>
  </summary>
  <div class="pl-root-body">{body_html}</div>
</details>
<style>
.proxy-map-legend {{
  position: fixed !important;
  top: 12px !important;
  right: 12px !important;
  width: min(300px, 32vw) !important;
  max-height: calc(100vh - 30px) !important;
  overflow-x: hidden !important;
  z-index: 650 !important;
  background: rgba(255,255,255,0.95) !important;
  border: 1px solid #888 !important;
  border-radius: 6px !important;
  box-shadow: 0 1px 8px rgba(0,0,0,0.25) !important;
  font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif !important;
  font-size: 11px !important;
  color: #111 !important;
}}
/* When the master <details> is collapsed, shrink to a pill: header only, auto width */
.proxy-map-legend:not([open]) {{
  width: auto !important;
  max-width: 220px !important;
  background: rgba(255,255,255,0.92) !important;
}}
.proxy-map-legend:not([open]) .pl-root-summary {{
  padding: 4px 10px !important;
  border-radius: 6px !important;
}}
.proxy-map-legend .pl-root-summary {{
  list-style: none !important;
  cursor: pointer !important;
  padding: 6px 10px !important;
  display: flex !important;
  align-items: center !important;
  gap: 6px !important;
  user-select: none !important;
}}
.proxy-map-legend .pl-root-summary::-webkit-details-marker {{ display: none !important; }}
.proxy-map-legend .pl-chev {{
  display: inline-block;
  width: 0;
  height: 0;
  border-left: 5px solid transparent;
  border-right: 5px solid transparent;
  border-top: 6px solid #444;
  transition: transform 0.15s ease;
  transform: rotate(-90deg);
}}
.proxy-map-legend[open] .pl-chev {{ transform: rotate(0deg); }}
.proxy-map-legend .pl-root-body {{
  padding: 4px 10px 8px 10px !important;
  overflow-y: auto !important;
  max-height: calc(100vh - 70px) !important;
  border-top: 1px solid #e5e5e5 !important;
}}
.proxy-map-legend .pl-header {{
  font-weight: 700;
  font-size: 12.5px;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  flex: 1;
  min-width: 0;
}}
.proxy-map-legend .pl-title {{
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}}
.proxy-map-legend .pl-badge {{
  background: #0d47a1;
  color: #fff;
  border-radius: 4px;
  padding: 1px 7px;
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.2px;
  white-space: nowrap;
}}
.proxy-map-legend .pl-region {{
  font-size: 10px;
  color: #444;
  margin: 0 0 6px 0;
  font-style: italic;
}}
.proxy-map-legend .pl-details {{
  border-top: 1px solid #ddd;
  margin: 0;
  padding: 4px 0 2px 0;
}}
.proxy-map-legend .pl-details:first-of-type {{ border-top: none; padding-top: 0; }}
.proxy-map-legend summary {{
  cursor: pointer;
  font-weight: 600;
  font-size: 11.5px;
  list-style: none;
  padding: 2px 0;
}}
.proxy-map-legend summary::-webkit-details-marker {{ display: none; }}
.proxy-map-legend .pl-body {{
  padding: 4px 2px 2px 2px;
  color: #222;
  line-height: 1.35;
}}
.proxy-map-legend code {{
  background: rgba(0,0,0,0.06);
  padding: 0 3px;
  border-radius: 3px;
  font-size: 10.5px;
}}
.proxy-map-legend .pl-cmap {{ margin: 4px 0 6px 0; }}
.proxy-map-legend .pl-cmap-cap {{
  font-size: 10.5px;
  color: #333;
  margin-bottom: 2px;
}}
.proxy-map-legend .pl-cmap-bar {{
  height: 10px;
  border-radius: 2px;
  border: 1px solid rgba(0,0,0,0.2);
}}
.proxy-map-legend .pl-cmap-ticks {{
  display: flex;
  justify-content: space-between;
  font-size: 10px;
  color: #333;
  margin-top: 2px;
}}
.proxy-map-legend ul.pl-swatch-grid,
.proxy-map-legend ul.pl-swatch-grid-2 {{
  list-style: none;
  margin: 4px 0 2px 0;
  padding: 0;
  display: grid;
  gap: 2px 10px;
}}
.proxy-map-legend ul.pl-swatch-grid {{ grid-template-columns: 1fr; }}
.proxy-map-legend ul.pl-swatch-grid-2 {{ grid-template-columns: 1fr 1fr; }}
.proxy-map-legend .pl-swatch {{
  display: inline-block;
  width: 12px;
  height: 12px;
  border-radius: 2px;
  border: 1px solid rgba(0,0,0,0.25);
  vertical-align: middle;
  margin-right: 6px;
}}
.proxy-map-legend .pl-swatch.pl-swatch-blend {{
  background-origin: border-box;
}}
.proxy-map-legend .pl-swatch-label {{
  vertical-align: middle;
  font-size: 10.5px;
}}
.proxy-map-legend .pl-hint {{
  color: #555;
  font-size: 10px;
  margin: 4px 0 0 0;
}}
.proxy-map-legend .pl-meta {{
  color: #333;
  font-size: 10px;
  margin: 2px 0 0 0;
}}
.proxy-map-legend .pl-note {{
  color: #b00020;
  font-size: 10.5px;
  margin: 4px 0 0 0;
}}
.proxy-map-legend ul.pl-plain {{
  list-style: disc;
  padding-left: 16px;
  margin: 4px 0 2px 0;
}}
.proxy-map-legend ul.pl-plain li {{
  margin: 1px 0;
}}
.proxy-map-legend .pl-muted {{
  color: #666;
  font-weight: 400;
}}
</style>
"""


def _fmt_num(v: float) -> str:
    """Compact numeric formatter for colormap tick labels."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not np.isfinite(x):
        return "n/a"
    a = abs(x)
    if a == 0.0:
        return "0"
    if a >= 1000.0 or a < 1e-3:
        return f"{x:.2e}"
    if a >= 100.0:
        return f"{x:.0f}"
    if a >= 10.0:
        return f"{x:.1f}"
    if a >= 1.0:
        return f"{x:.2f}"
    return f"{x:.3f}"


def weight_percentile_stats(w_arr: np.ndarray) -> tuple[float, float]:
    """Return ``(p2, p98)`` over strictly-positive finite values (linear)."""
    pos = w_arr[np.isfinite(w_arr) & (w_arr > 0)]
    if pos.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(pos, 2.0))
    hi = float(np.percentile(pos, 98.0))
    if lo >= hi:
        hi = lo + 1e-9
    return lo, hi


def weight_log_percentile_stats(w_arr: np.ndarray) -> tuple[float, float]:
    """Return ``(p2, p98)`` of ``log10`` over strictly-positive finite values."""
    pos = w_arr[np.isfinite(w_arr) & (w_arr > 0)]
    if pos.size == 0:
        return 0.0, 1.0
    lp = np.log10(np.maximum(pos, 1e-30))
    lo = float(np.percentile(lp, 2.0))
    hi = float(np.percentile(lp, 98.0))
    if lo >= hi:
        hi = lo + 1e-6
    return lo, hi


def region_note(
    region: str | None,
    override_bbox: tuple[float, float, float, float] | None,
) -> str | None:
    """Human-readable label for the focus area, displayed atop the legend panel."""
    if override_bbox is not None:
        w, s, e, n = override_bbox
        return f"Focus bbox: {w:.2f} / {s:.2f} / {e:.2f} / {n:.2f}"
    if region and str(region).lower() not in ("country", "full"):
        return f"Focus region: {region}"
    return "Focus: full country"


def build_weight_legend_section(
    w_arr: np.ndarray,
    *,
    display_mode: str,
    cmap: str = "plasma",
    title: str = "Weight scale",
) -> LegendSection:
    """Build the standard "Weight scale" legend section.

    ``display_mode`` is one of ``"per_cell"`` (fixed 0-1), ``"global_log"``
    (log10 of positives, 2-98%), or ``"percentile"`` (linear 2-98% of positives).
    Every sector used to emit a floating Branca colormap with one of these three
    behaviours; this centralises the min/max choice and swatch rendering.
    """
    mode = str(display_mode).strip().lower()
    if mode == "per_cell":
        caption = "Per-CAMS-cell 0-1"
        vmin, vmax = 0.0, 1.0
        swatch = colormap_swatch_html(
            cmap=cmap, vmin=vmin, vmax=vmax, caption=caption,
            label_min="0", label_max="1",
        )
    elif mode == "global_log":
        lo, hi = weight_log_percentile_stats(w_arr)
        swatch = colormap_swatch_html(
            cmap=cmap, vmin=lo, vmax=hi,
            caption="log10(weight), 2-98% of positives",
            label_min=f"1e{lo:.1f}", label_max=f"1e{hi:.1f}",
        )
    else:
        lo, hi = weight_percentile_stats(w_arr)
        swatch = colormap_swatch_html(
            cmap=cmap, vmin=lo, vmax=hi,
            caption="Weight, 2-98% of positives (linear)",
        )
    return LegendSection(title=title, html=swatch, open=True)
