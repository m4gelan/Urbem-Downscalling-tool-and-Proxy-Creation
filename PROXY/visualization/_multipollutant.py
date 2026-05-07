"""Multi-pollutant weight overlays + CoV variance layer for sector previews.

Before this module every sector map baked exactly **one** weight band chosen
via ``pick_band_by_pollutant`` and told the user "here is NOx". That hid the
fact that each weight GeoTIFF carries up to nine pollutant bands whose spatial
distribution can differ substantially (e.g. NOx is concentrated along roads
while PM2.5 tracks fuel-use intensity).

:func:`add_multipollutant_weight_layers` picks up to three pollutants - the
preferred one plus the next two in a stable priority order - bakes an RGBA
per pollutant, and adds them to the map as radio-grouped toggles so the user
can flip between pollutants. It also computes a coefficient-of-variation
layer (``CoV = std / mean``) across those bands and adds it as a separate
toggle, highlighting where pollutants disagree spatially.

Public API:
  * :class:`MultipollutantResult` - summary returned to the caller for the legend.
  * :func:`add_multipollutant_weight_layers` - one-call integration for sector writers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

import rasterio

from PROXY.visualization._legend import (
    LegendSection,
    colormap_swatch_html,
    weight_log_percentile_stats,
    weight_percentile_stats,
)
from PROXY.visualization.overlay_utils import scalar_to_rgba

# Display priority used when the sector does not express a preference via
# ``viz_cfg['visualization_pollutant']``. Kept small and ordered so the first
# bands shown are always the air-quality workhorses (NOx / PM2.5 / CO2).
DEFAULT_POLLUTANT_PRIORITY: tuple[str, ...] = (
    "NOx", "PM2.5", "CO2", "NH3", "SO2", "CO", "NMVOC", "CH4", "PM10",
)

# Normalized priority token -> alternate normalized key for band description lookup.
_POLLUTANT_SYNONYMS: dict[str, str] = {
    "sox": "so2",
}


def visualization_pollutant_priority_from_cfg(
    viz_cfg: dict[str, Any] | None,
) -> tuple[tuple[str, ...] | None, bool]:
    """Parse sector ``visualization`` (or ``area_proxy``) block for panel order.

    Returns ``(priority_tuple, exclusive)``. When ``exclusive`` is ``True``, the
    multipollutant panel lists **only** pollutants from that tuple (plus any
    needed to reach ``max_bands`` that appear in the tuple), never the global
    :data:`DEFAULT_POLLUTANT_PRIORITY` fillers.

    Supported YAML:

    * ``visualization_pollutants: [CO, NMVOC, SO2]``
    * ``visualization_pollutant: [CO, NMVOC, SO2]`` (same order; some sectors use this key only)
    * ``visualization_pollutant: "CO, NMVOC, SO2"`` (comma-separated string)
    """
    if not viz_cfg:
        return None, False
    raw = viz_cfg.get("visualization_pollutants")
    if isinstance(raw, (list, tuple)):
        seq = tuple(str(x).strip() for x in raw if str(x).strip())
        return (seq if seq else None), bool(seq)
    pol = viz_cfg.get("visualization_pollutant")
    if isinstance(pol, (list, tuple)):
        seq = tuple(str(x).strip() for x in pol if str(x).strip())
        return (seq if seq else None), bool(seq)
    if isinstance(pol, str) and "," in pol:
        seq = tuple(p.strip() for p in pol.split(",") if p.strip())
        return (seq if seq else None), bool(seq)
    return None, False


@dataclass
class MultipollutantResult:
    """What the sector writer needs to know after the helper ran."""

    primary_band: int
    primary_label: str
    bands: list[tuple[int, str]] = field(default_factory=list)
    bands_arrays: list[np.ndarray] = field(default_factory=list)
    cov_added: bool = False
    # How many distinct pollutants contributed to the CoV layer.
    cov_n_bands: int = 0
    legend_sections: list[LegendSection] = field(default_factory=list)


def _norm_label(label: str) -> str:
    s = str(label or "").strip().lower()
    for ch in ("_", "-", " ", "."):
        s = s.replace(ch, "")
    return s


def _positive_mask(arr: np.ndarray, nodata: float | None) -> np.ndarray:
    ok = np.isfinite(arr)
    if nodata is not None:
        ok &= arr != float(nodata)
    ok &= arr > 0.0
    return ok


def _list_pollutant_bands(
    weight_path: Path,
    *,
    strip_prefixes: Iterable[str],
    shorten_w_gnfr_e_weight_labels: bool = False,
) -> list[tuple[int, str]]:
    """Return ``[(band_idx, label)]`` for every band with a parseable description."""
    out: list[tuple[int, str]] = []
    prefixes = tuple(strip_prefixes)
    with rasterio.open(weight_path) as src:
        n = int(src.count)
        descriptions = list(src.descriptions or [])
    for i in range(1, n + 1):
        d = descriptions[i - 1] if i - 1 < len(descriptions) else None
        if not d:
            out.append((i, f"band {i}"))
            continue
        lab = str(d).strip()
        if shorten_w_gnfr_e_weight_labels:
            from PROXY.visualization._mapbuilder import shorten_w_gnfr_e_weight_band_label

            short = shorten_w_gnfr_e_weight_band_label(lab)
            if short is not None:
                lab = short
            else:
                for pfx in prefixes:
                    if lab.lower().startswith(pfx.lower()):
                        lab = lab[len(pfx) :]
                        break
        else:
            for pfx in prefixes:
                if lab.lower().startswith(pfx.lower()):
                    lab = lab[len(pfx) :]
                    break
        out.append((i, lab or f"band {i}"))
    return out


def _pick_panel_bands(
    bands: list[tuple[int, str]],
    preferred: int,
    *,
    preferred_label: str,
    priority: tuple[str, ...],
    max_bands: int,
    weight_path: Path,
    exclusive: bool = False,
) -> list[tuple[int, str]]:
    """Choose up to ``max_bands`` ``(band, label)`` pairs, preferred first.

    Bands with no positive values are skipped unless every sibling is empty
    (in which case we still show the preferred one so the user sees the context
    "all bands empty").

    When ``exclusive`` is ``True``, only the preferred band and matches from
    ``priority`` are considered (no scan of every raster band to fill slots).
    """
    picked: list[tuple[int, str]] = []
    seen_bands: set[int] = set()
    seen_labels: set[str] = set()

    def _has_positive(band_idx: int) -> bool:
        try:
            with rasterio.open(weight_path) as src:
                if band_idx < 1 or band_idx > int(src.count):
                    return False
                a = src.read(band_idx).astype(np.float64)
            return bool(np.any(np.isfinite(a) & (a > 0.0)))
        except Exception:
            return False

    def _add(b: int, lab: str) -> None:
        if b in seen_bands:
            return
        n = _norm_label(lab)
        if n in seen_labels:
            return
        picked.append((b, lab))
        seen_bands.add(b)
        seen_labels.add(n)

    _add(int(preferred), preferred_label)

    priority_norm = [_norm_label(p) for p in priority]
    label_to_band: dict[str, int] = {}
    for b, lab in bands:
        key = _norm_label(lab)
        if key and key not in label_to_band:
            label_to_band[key] = b

    for pn in priority_norm:
        if len(picked) >= max_bands:
            break
        cand_keys = [pn]
        if pn in _POLLUTANT_SYNONYMS:
            cand_keys.append(_norm_label(_POLLUTANT_SYNONYMS[pn]))
        for key in cand_keys:
            if key not in label_to_band:
                continue
            b = label_to_band[key]
            lab = next((lab for bb, lab in bands if bb == b), f"band {b}")
            if _has_positive(b):
                _add(b, lab)
            break

    if not exclusive:
        for b, lab in bands:
            if len(picked) >= max_bands:
                break
            if _has_positive(b):
                _add(b, lab)

    return picked[:max_bands]


def _bake_weight_rgba(
    w_arr: np.ndarray,
    *,
    display_mode: str,
    cmap: str,
    w_nodata: float | None,
    cams_nc_path: Path | None,
    m_area: np.ndarray | None,
    cams_ds: Any | None,
    view: Any,
) -> np.ndarray:
    """Return an RGBA for ``w_arr`` using the requested ``display_mode``."""
    mode = str(display_mode).strip().lower()
    if mode == "per_cell" and cams_nc_path and m_area is not None and cams_ds is not None:
        from PROXY.visualization._mapbuilder import weight_rgba_per_cell

        return weight_rgba_per_cell(
            w_arr,
            w_nodata=w_nodata,
            cams_nc_path=cams_nc_path,
            m_area=m_area,
            ds=cams_ds,
            view=view,
            cmap=cmap,
        )
    if mode == "global_log":
        return scalar_to_rgba(
            w_arr, colour_mode="log", cmap_name=cmap, hide_zero=True,
            nodata_val=float(w_nodata) if w_nodata is not None else None,
        )
    return scalar_to_rgba(
        w_arr, colour_mode="percentile", cmap_name=cmap, hide_zero=True,
        nodata_val=float(w_nodata) if w_nodata is not None else None,
    )


def _cov_rgba(
    band_arrays: list[np.ndarray],
    band_nodatas: list[float | None],
    *,
    cmap: str = "YlGnBu",
) -> tuple[np.ndarray, float, float] | None:
    """Return ``(rgba, vmin, vmax)`` for the pixel-wise coefficient of variation.

    Pixels where fewer than two bands are positive are transparent. ``vmin``
    and ``vmax`` are the 2/98 percentiles of the CoV distribution used for
    the legend tick labels.
    """
    if len(band_arrays) < 2:
        return None
    h = int(band_arrays[0].shape[0])
    w = int(band_arrays[0].shape[1])
    stack = np.full((len(band_arrays), h, w), np.nan, dtype=np.float64)
    for i, (arr, nd) in enumerate(zip(band_arrays, band_nodatas)):
        if arr.shape != (h, w):
            return None
        pos = _positive_mask(arr, nd)
        stack[i][pos] = arr[pos]
    count = np.sum(np.isfinite(stack), axis=0)
    valid = count >= 2
    import warnings as _warnings
    with _warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        _warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(stack, axis=0)
        std = np.nanstd(stack, axis=0, ddof=0)
        cov = np.where(valid & (mean > 0.0), std / np.maximum(mean, 1e-30), np.nan)
    pos = cov[np.isfinite(cov) & (cov > 0.0)]
    if pos.size == 0:
        return None
    vmin = float(np.percentile(pos, 2.0))
    vmax = float(np.percentile(pos, 98.0))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    rgba = scalar_to_rgba(
        cov.astype(np.float32),
        colour_mode="percentile",
        cmap_name=cmap,
        hide_zero=True,
        nodata_val=None,
    )
    return rgba, vmin, vmax


def add_multipollutant_weight_layers(
    fmap: Any,
    view: Any,
    weight_path: Path,
    *,
    preferred_band: int,
    preferred_label: str,
    sector_key: str,
    display_mode: str,
    cmap: str = "plasma",
    weight_opacity: float = 0.92,
    cams_nc_path: Path | None = None,
    m_area: np.ndarray | None = None,
    cams_ds: Any | None = None,
    strip_prefixes: Iterable[str] = (),
    max_bands: int = 3,
    pollutant_priority: tuple[str, ...] | None = None,
    exclusive_pollutant_panel: bool = False,
    cov_cmap: str = "YlGnBu",
    clip_alpha_to_cams: bool = True,
    shorten_w_gnfr_e_weight_labels: bool = False,
    weight_overlay_names: Sequence[str] | None = None,
    cov_overlay_name: str | None = None,
) -> MultipollutantResult:
    """Add up to ``max_bands`` pollutant weight layers + a CoV layer to ``fmap``.

    The preferred band is shown by default; the remaining are available in
    the layer control as radio toggles via a ``GroupedLayerControl`` group
    titled ``"Pollutant (<sector_key>)"`` when more than one pollutant was
    selected. A CoV (std/mean) overlay is added only when at least two
    pollutants contributed positive pixels.

    Returns a :class:`MultipollutantResult` with the chosen bands, whether a
    CoV layer was added, and a legend section ready to append to the unified
    panel.
    """
    import folium

    from PROXY.visualization._mapbuilder import (
        add_raster_overlay,
        clip_rgba_to_cams_mask,
        pollutant_label_for_band,
        read_weight_on_view,
    )

    available = _list_pollutant_bands(
        weight_path,
        strip_prefixes=strip_prefixes,
        shorten_w_gnfr_e_weight_labels=shorten_w_gnfr_e_weight_labels,
    )
    eff_priority = (
        pollutant_priority
        if pollutant_priority is not None
        else DEFAULT_POLLUTANT_PRIORITY
    )
    exclusive = bool(exclusive_pollutant_panel and pollutant_priority is not None)
    if exclusive and pollutant_priority is not None:
        avail_keys = {_norm_label(lab) for _, lab in available}
        filt = tuple(p for p in pollutant_priority if _norm_label(p) in avail_keys)
        if not filt:
            exclusive = False
            eff_priority = DEFAULT_POLLUTANT_PRIORITY
        elif len(filt) < len(pollutant_priority):
            eff_priority = filt
    panel_bands = _pick_panel_bands(
        available,
        int(preferred_band),
        preferred_label=preferred_label,
        priority=eff_priority,
        max_bands=int(max_bands),
        weight_path=weight_path,
        exclusive=exclusive,
    )

    band_arrays: list[np.ndarray] = []
    band_nodatas: list[float | None] = []
    feature_groups: list[tuple[Any, int, str]] = []

    primary_band = int(preferred_band)
    primary_label = preferred_label

    for idx, (band_idx, label) in enumerate(panel_bands):
        stk = read_weight_on_view(weight_path, view, weight_band=int(band_idx))
        w_arr = stk["weight_wgs84"]
        w_nodata = stk.get("weight_nodata")
        band_arrays.append(np.asarray(w_arr, dtype=np.float32))
        band_nodatas.append(float(w_nodata) if w_nodata is not None else None)

        rgba = _bake_weight_rgba(
            w_arr, display_mode=display_mode, cmap=cmap, w_nodata=w_nodata,
            cams_nc_path=cams_nc_path, m_area=m_area, cams_ds=cams_ds, view=view,
        )
        if clip_alpha_to_cams and cams_nc_path is not None and m_area is not None and cams_ds is not None:
            try:
                rgba = clip_rgba_to_cams_mask(
                    rgba, cams_nc_path=cams_nc_path, m_area=m_area, ds=cams_ds, view=view,
                )
            except Exception:
                pass

        if weight_overlay_names is not None and idx < len(weight_overlay_names):
            overlay_label = weight_overlay_names[idx]
        else:
            overlay_label = pollutant_label_for_band(weight_path, int(band_idx))

        fg = folium.FeatureGroup(
            name=overlay_label,
            overlay=True,
            show=(idx == 0),
        )
        feature_groups.append((fg, int(band_idx), label))
        add_raster_overlay(
            fg, rgba, view,
            name=overlay_label,
            opacity=weight_opacity,
            show=True,
        )
        fg.add_to(fmap)

        if idx == 0:
            primary_band = int(band_idx)
            primary_label = pollutant_label_for_band(weight_path, int(band_idx))

    legend_sections: list[LegendSection] = []
    cov_added = False
    cov_n = 0

    pollutant_list_html = "".join(
        (
            f'<li><b>{label}</b> <span class="pl-muted">(band {band_idx})</span></li>'
            if band_idx != primary_band
            else f'<li><b>{label}</b> <span class="pl-muted">(band {band_idx}, shown by default)</span></li>'
        )
        for band_idx, label in panel_bands
    )
    pollutant_section_html = (
        '<p class="pl-hint">Toggle pollutants from the layer control '
        '(top-left). Only one pollutant layer should be on at a time to keep '
        'the weight colours meaningful.</p>'
        f'<ul class="pl-plain">{pollutant_list_html}</ul>'
    )
    if panel_bands:
        legend_sections.append(
            LegendSection(
                title=f"Pollutants on view ({len(panel_bands)})",
                html=pollutant_section_html,
                open=True,
            )
        )

    if len(band_arrays) >= 2:
        cov_res = _cov_rgba(band_arrays, band_nodatas, cmap=cov_cmap)
        if cov_res is not None:
            cov_rgba, cov_lo, cov_hi = cov_res
            if clip_alpha_to_cams and cams_nc_path is not None and m_area is not None and cams_ds is not None:
                try:
                    cov_rgba = clip_rgba_to_cams_mask(
                        cov_rgba, cams_nc_path=cams_nc_path, m_area=m_area, ds=cams_ds, view=view,
                    )
                except Exception:
                    pass
            cov_name = (
                str(cov_overlay_name).strip()
                if cov_overlay_name is not None and str(cov_overlay_name).strip()
                else f"Pollutant variance (CoV, {len(band_arrays)} bands)"
            )
            add_raster_overlay(
                fmap, cov_rgba, view,
                name=cov_name,
                opacity=0.82,
                show=False,
            )
            cov_added = True
            cov_n = len(band_arrays)
            legend_sections.append(
                LegendSection(
                    title="Pollutant variance (CoV)",
                    html=(
                        '<p class="pl-hint">Per-pixel coefficient of variation '
                        f'(<code>std/mean</code>) across <b>{cov_n}</b> pollutant bands. '
                        'Cooler pixels = pollutants agree on the distribution; '
                        'hotter pixels = one pollutant dominates.</p>'
                        + colormap_swatch_html(
                            cmap=cov_cmap,
                            vmin=cov_lo, vmax=cov_hi,
                            caption="CoV, 2-98% of positive pixels",
                        )
                    ),
                    open=False,
                )
            )

    _ = sector_key
    _ = (weight_percentile_stats, weight_log_percentile_stats)

    return MultipollutantResult(
        primary_band=primary_band,
        primary_label=primary_label,
        bands=list(panel_bands),
        bands_arrays=band_arrays,
        cov_added=cov_added,
        cov_n_bands=cov_n,
        legend_sections=legend_sections,
    )
