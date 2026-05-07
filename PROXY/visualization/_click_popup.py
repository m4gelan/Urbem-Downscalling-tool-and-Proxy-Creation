"""Per-CAMS-cell click-to-inspect popup with dominance breakdown.

Uses the same rule as :func:`PROXY.visualization._dominance.compute_dominance_rgba`:
per-pixel argmax among layers with strictly positive values. The popup reports
the **modal** winner in the cell and a **percentage breakdown** of how often
each group wins among display pixels inside the CAMS cell.

For B_Industry / D_Fugitive pass ``group_pg``; for E_Solvents / J_Waste pass
``family_scores`` (the dict fed to ``compute_dominance_rgba``).

Folium wiring: :func:`PROXY.visualization._mapbuilder.add_cams_grid_overlay`.
"""
from __future__ import annotations

import html as _html
from typing import Any, Mapping

import numpy as np


def _dominance_stats_in_cell(
    mask: np.ndarray,
    groups: Mapping[str, np.ndarray],
    *,
    min_positive: float = 0.0,
) -> tuple[str | None, list[tuple[str, float]]]:
    """Return ``(modal_winner, [(group_id, share), ...])`` sorted by share desc.

    ``share`` is the fraction of pixels in the cell (among those with at least
    one positive group score) where that group wins the argmax.
    """
    keys = [k for k, v in groups.items() if v is not None]
    if not keys:
        return None, []
    arrs = [np.asarray(groups[k], dtype=np.float64) for k in keys]
    shape = arrs[0].shape
    if any(a.shape != shape for a in arrs) or mask.shape != shape:
        return None, []
    stack = np.stack(arrs, axis=0)
    valid = np.isfinite(stack) & (stack > float(min_positive))
    scored = np.where(valid, stack, -np.inf)
    any_g = np.any(valid, axis=0)
    winners = np.argmax(scored, axis=0)
    in_cell = mask & any_g
    if not np.any(in_cell):
        return None, []
    w = winners[in_cell].astype(np.int64, copy=False)
    bc = np.bincount(w, minlength=len(keys))
    total = float(w.size)
    if total <= 0:
        return None, []
    modal = keys[int(np.argmax(bc))]
    shares = [(keys[i], bc[i] / total) for i in range(len(keys))]
    shares.sort(key=lambda t: (-t[1], str(t[0])))
    return modal, shares


def _format_pct(x: float) -> str:
    return f"{100.0 * float(x):.1f}%"


def _build_popup_html(
    *,
    sector_title: str,
    cams_source_index: int,
    lon_c: float,
    lat_c: float,
    dominant_label: str | None,
    dominant_heading: str = "Dominant group",
    sector_has_dominance: bool,
    share_rows: list[tuple[str, float]] | None,
) -> str:
    dom_block = ""
    if sector_has_dominance:
        if dominant_label:
            dom_disp = _html.escape(str(dominant_label))
        else:
            dom_disp = (
                '<span class="cp-none">No pixel in this cell with a positive group score.</span>'
            )
        dom_block = (
            f'<div class="cp-dominant"><b>{_html.escape(dominant_heading)} (modal):</b> '
            f"{dom_disp}</div>"
        )
    shares_block = ""
    if share_rows:
        rows_html = "".join(
            "<tr>"
            f"<td>{_html.escape(str(lbl))}</td>"
            f"<td class=\"cp-pct\">{_format_pct(frac)}</td>"
            "</tr>"
            for lbl, frac in share_rows
        )
        shares_block = (
            '<div class="cp-shares-hint">Share of pixels in this cell where each group '
            "wins (same rule as the dominance overlay).</div>"
            '<table class="cp-table">'
            "<tr><th>Group</th><th>Share</th></tr>"
            f"{rows_html}</table>"
        )
    return (
        '<div class="cams-popup">'
        f'<div class="cp-title"><b>{_html.escape(sector_title)}</b> '
        f"&mdash; CAMS cell #{int(cams_source_index)}</div>"
        '<div class="cp-meta">'
        f'Centre: <code>{lon_c:.4f}&deg;E, {lat_c:.4f}&deg;N</code>'
        "</div>"
        f"{dom_block}"
        f"{shares_block}"
        "</div>"
    )


def enrich_cams_grid_with_popups(
    grid_fc: dict[str, Any],
    *,
    view: Any,
    m_area: np.ndarray | None,
    ds: Any,
    sector_title: str,
    dominance_layers: Mapping[str, np.ndarray] | None = None,
    dominant_heading: str = "Dominant group",
) -> dict[str, Any]:
    """Add ``popup_html`` to every CAMS cell feature in ``grid_fc`` (in place)."""
    feats = grid_fc.get("features") or []
    if not feats or m_area is None or ds is None:
        return grid_fc
    try:
        cell_id = _compute_cell_id_on_view(view, ds, m_area)
    except Exception:
        return grid_fc

    layers: dict[str, np.ndarray] | None = None
    if dominance_layers:
        layers = {
            str(k): np.asarray(v, dtype=np.float64)
            for k, v in dominance_layers.items()
            if v is not None
        }
        layers = {k: v for k, v in layers.items() if v.shape == (view.gh, view.gw)}
        if not layers:
            layers = None

    for feat in feats:
        props = feat.setdefault("properties", {})
        cid = int(props.get("cams_source_index", -1))
        lon_c = float(props.get("lon_c", float("nan")))
        lat_c = float(props.get("lat_c", float("nan")))
        mask = cell_id == cid
        if not np.any(mask):
            props["popup_html"] = (
                f'<div class="cams-popup"><b>{_html.escape(sector_title)}</b> '
                f"&mdash; CAMS cell #{cid}<br>"
                "<i>No map pixels in this cell for the current view.</i></div>"
            )
            continue
        dom: str | None = None
        share_rows: list[tuple[str, float]] | None = None
        if layers:
            dom, share_rows = _dominance_stats_in_cell(mask, layers, min_positive=0.0)
            if not share_rows:
                share_rows = None
        props["popup_html"] = _build_popup_html(
            sector_title=sector_title,
            cams_source_index=cid,
            lon_c=lon_c,
            lat_c=lat_c,
            dominant_label=dom,
            dominant_heading=dominant_heading,
            sector_has_dominance=bool(layers),
            share_rows=share_rows,
        )

    return grid_fc


def _compute_cell_id_on_view(
    view: Any,
    ds: Any,
    m_area: np.ndarray,
) -> np.ndarray:
    from rasterio.transform import xy as transform_xy

    from PROXY.visualization.cams_grid import cams_cell_id_grid

    rows, cols = np.indices((view.gh, view.gw))
    xs, ys = transform_xy(view.dst_t, rows + 0.5, cols + 0.5, offset="center")
    lons = np.asarray(xs, dtype=np.float64).reshape(view.gh, view.gw)
    lats = np.asarray(ys, dtype=np.float64).reshape(view.gh, view.gw)
    return np.asarray(cams_cell_id_grid(lons, lats, ds, m_area)).reshape(view.gh, view.gw)


CAMS_POPUP_CSS: str = """
<style>
.cams-popup {
  font: 12px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  color: #1f2937;
  min-width: 220px;
  max-width: 340px;
}
.cams-popup .cp-title {
  font-size: 13px;
  color: #111827;
  margin-bottom: 4px;
  padding-bottom: 3px;
  border-bottom: 1px solid #e5e7eb;
}
.cams-popup .cp-meta {
  color: #4b5563;
  margin-bottom: 8px;
  font-size: 11px;
}
.cams-popup .cp-dominant {
  font-size: 13px;
  color: #111827;
  line-height: 1.35;
  margin-bottom: 6px;
}
.cams-popup .cp-shares-hint {
  color: #6b7280;
  font-size: 10.5px;
  margin-bottom: 4px;
}
.cams-popup .cp-none {
  color: #6b7280;
  font-style: italic;
  font-size: 12px;
}
.cams-popup .cp-table {
  width: 100%;
  border-collapse: collapse;
  margin: 0 0 4px 0;
}
.cams-popup .cp-table th,
.cams-popup .cp-table td {
  border-bottom: 1px solid #f1f5f9;
  padding: 3px 4px;
  text-align: left;
  font-size: 11.5px;
}
.cams-popup .cp-table th {
  color: #6b7280;
  font-weight: 600;
}
.cams-popup .cp-table td.cp-pct {
  text-align: right;
  font-variant-numeric: tabular-nums;
}
.cams-popup code {
  background: #f3f4f6;
  padding: 1px 3px;
  border-radius: 2px;
  font-size: 11px;
}
</style>
""".strip()
