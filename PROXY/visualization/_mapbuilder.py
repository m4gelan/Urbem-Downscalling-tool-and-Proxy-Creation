"""Shared folium/branca scaffolding for ``PROXY.visualization`` sector writers.

Every ``*_area_map.py`` writer was cloning the same boilerplate:
 - folium / branca / rasterio imports with a consistent error message,
 - a folium.Map + CartoDB Positron + Esri World Imagery base layers,
 - picking a band by pollutant name and/or the first positive band,
 - the standard CAMS-cell outlines GeoJson layer,
 - an image overlay wrapped in a FeatureGroup,
 - log / per-cell / percentile legend colormaps,
 - a final save-to-html.

This module centralises those pieces so each sector writer only has to define
its own data reads, RGBA generation, and legend HTML. The public surface is
intentionally small and side-effect free; each helper accepts an explicit
Folium map and returns it or writes onto it.

None of these helpers alter the scientific content of a map. They reproduce
the exact same Folium calls that each sector's previous copy used (layer
names, colours, opacities, tooltips). The only behavioural change is that
sector files no longer drift against each other when a fix is applied
centrally.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from PROXY.core.alpha import norm_pollutant_key
from PROXY.visualization.overlay_utils import (
    read_weight_wgs84_only,
    sample_cmap_hex,
    scalar_to_rgba,
    weight_display_bounds_from_raster,
)

VIZ_DEPS_MESSAGE = (
    "Visualization requires folium, branca, rasterio, matplotlib. "
    "Install with: pip install folium branca rasterio matplotlib"
)

_ESRI_ATTR = (
    "Tiles &copy; Esri &mdash; "
    "Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community"
)
_ESRI_TILES = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)

_VOYAGER_TILES = (
    "https://{s}.basemaps.cartocdn.com/rastertiles/voyager/"
    "{z}/{x}/{y}.png"
)
_VOYAGER_ATTR = (
    "&copy; <a href=\"https://www.openstreetmap.org/copyright\">OpenStreetMap</a> "
    "contributors &copy; <a href=\"https://carto.com/attributions\">CARTO</a>"
)

# Named focus regions for ``--region`` / ``visualize`` CLI. WGS84 (W, S, E, N).
# Keeps dense OSM / CORINE previews readable on small areas instead of whole-country.
REGION_BBOXES: dict[str, tuple[float, float, float, float] | None] = {
    "attica": (23.30, 37.70, 24.25, 38.45),
    "thessaloniki": (22.60, 40.40, 23.25, 40.85),
    "patras": (21.60, 37.95, 22.00, 38.35),
    "heraklion": (24.95, 35.20, 25.40, 35.45),
    "crete": (23.45, 34.75, 26.45, 35.80),
    "athens_extended": (22.90, 37.40, 24.60, 38.60),
    "country": None,
    "full": None,
}


def resolve_view_bbox(
    raster_path: Path,
    *,
    pad_deg: float,
    override_bbox: tuple[float, float, float, float] | None = None,
    region: str | None = None,
) -> tuple[float, float, float, float]:
    """Decide the display bbox.

    Priority:
    1. ``override_bbox`` (explicit --bbox) wins.
    2. ``region`` (named entry in :data:`REGION_BBOXES`).
    3. Full raster bbox (backwards-compatible default).

    Pad is applied on every branch that returns a bbox.
    """
    if override_bbox is not None:
        w, s, e, n = (float(x) for x in override_bbox)
        return (w - pad_deg, s - pad_deg, e + pad_deg, n + pad_deg)
    if region:
        key = str(region).strip().lower()
        if key in REGION_BBOXES and REGION_BBOXES[key] is not None:
            w, s, e, n = REGION_BBOXES[key]  # type: ignore[misc]
            return (w - pad_deg, s - pad_deg, e + pad_deg, n + pad_deg)
    return weight_display_bounds_from_raster(raster_path, pad_deg=pad_deg)


def require_folium_imports() -> dict[str, Any]:
    """Return the common folium/branca/rasterio bundle used by every writer.

    Raises ``SystemExit`` with the shared install message if any dependency is
    missing, matching the legacy behaviour of the per-sector files.
    """
    try:
        import folium
        from branca.colormap import LinearColormap
        from folium.plugins import Fullscreen
        from rasterio.transform import from_bounds
        from rasterio.transform import xy as transform_xy
    except ImportError as exc:
        raise SystemExit(VIZ_DEPS_MESSAGE) from exc
    return {
        "folium": folium,
        "LinearColormap": LinearColormap,
        "Fullscreen": Fullscreen,
        "from_bounds": from_bounds,
        "transform_xy": transform_xy,
    }


def resolve_under_root(path: Path, root: Path) -> Path:
    """Return ``path`` unchanged if absolute, else ``root / path``."""
    return path if path.is_absolute() else root / path


@dataclass(frozen=True)
class ViewContext:
    """Geographic display window + matching transform for a single map."""

    west: float
    south: float
    east: float
    north: float
    gw: int
    gh: int
    dst_t: Any  # rasterio Affine

    @property
    def bounds(self) -> list[list[float]]:
        return [[self.south, self.west], [self.north, self.east]]

    @property
    def centre(self) -> tuple[float, float]:
        return ((self.south + self.north) / 2.0, (self.west + self.east) / 2.0)

    @property
    def bbox_tuple(self) -> tuple[float, float, float, float]:
        return (self.west, self.south, self.east, self.north)


def compute_view_context(
    weight_path: Path,
    *,
    pad_deg: float,
    max_width: int,
    max_height: int,
    override_bbox: tuple[float, float, float, float] | None = None,
    region: str | None = None,
) -> ViewContext:
    """Standard display window derived from the weight GeoTIFF.

    ``override_bbox`` and ``region`` let the caller focus the preview on a sub-area
    (e.g. Attica) which is essential for reading dense OSM / CORINE overlays.
    """
    from rasterio.transform import from_bounds

    west, south, east, north = resolve_view_bbox(
        weight_path, pad_deg=pad_deg, override_bbox=override_bbox, region=region
    )
    gw = max(64, int(max_width))
    gh = max(64, int(max_height))
    dst_t = from_bounds(west, south, east, north, gw, gh)
    return ViewContext(
        west=west, south=south, east=east, north=north, gw=gw, gh=gh, dst_t=dst_t
    )


def read_weight_on_view(weight_path: Path, view: ViewContext, *, weight_band: int = 1) -> dict[str, Any]:
    """Thin wrapper over :func:`read_weight_wgs84_only` with the view geometry."""
    return read_weight_wgs84_only(
        weight_path,
        west=view.west,
        south=view.south,
        east=view.east,
        north=view.north,
        display_width=view.gw,
        display_height=view.gh,
        weight_band=weight_band,
    )


def _band_desc_matches_target(norm_desc: str, tgt: str) -> bool:
    """True if raster band description ``norm_desc`` matches requested pollutant ``tgt``."""
    if norm_desc == tgt:
        return True
    pair = {norm_desc, tgt}
    if pair == {"sox", "so2"}:
        return True
    return False


def shorten_w_gnfr_e_weight_band_label(description: str | None) -> str | None:
    """If ``description`` matches ``W_<pollutant>_GNFR_E_area``, return the pollutant token.

    E_Solvents (and similar) export band names like ``W_NMVOC_GNFR_E_area``. That
    string does not normalize to the same key as ``NMVOC`` alone, so band lookup
    must peel the fixed wrapper before comparing to config pollutants.
    """
    if not description:
        return None
    import re

    m = re.match(r"^W_(.+)_GNFR_E_area$", str(description).strip(), re.IGNORECASE)
    if not m:
        return None
    inner = m.group(1).strip()
    return inner if inner else None


def effective_visualization_pollutant(viz_cfg: dict[str, Any]) -> Any:
    """Return the primary pollutant token for band lookup.

    If ``visualization_pollutants`` is a non-empty list, uses its first entry.
    If ``visualization_pollutant`` is a non-empty list/tuple, uses its first entry.
    If ``visualization_pollutant`` is a comma-separated string, uses the first
    segment. Otherwise returns ``visualization_pollutant`` as-is.
    """
    raw = viz_cfg.get("visualization_pollutants")
    if isinstance(raw, (list, tuple)) and len(raw) > 0:
        t = str(raw[0]).strip()
        return t if t else None
    pol = viz_cfg.get("visualization_pollutant")
    if isinstance(pol, (list, tuple)) and len(pol) > 0:
        t = str(pol[0]).strip()
        return t if t else None
    if isinstance(pol, str) and "," in pol:
        t = pol.split(",")[0].strip()
        return t if t else None
    return pol


def pick_band_by_pollutant(
    weight_path: Path,
    viz_cfg: dict[str, Any],
    *,
    sector_cfg: dict[str, Any] | None = None,
    strip_prefixes: Iterable[str] = (),
    pollutant_key: str = "visualization_pollutant",
    band_key: str = "visualization_weight_band",
) -> int:
    """Return the 1-based band index whose description matches the configured pollutant.

    Falls back to ``viz_cfg[band_key]`` (default 1) if the pollutant is missing
    or unmatched. ``strip_prefixes`` lets callers match descriptions such as
    ``weight_share_agri_NH3`` or ``j_waste_weight_nox`` that carry a fixed
    sector prefix in front of the pollutant token.

    When several pollutants are configured (comma-separated ``visualization_pollutant``
    or ``visualization_pollutants`` list), the **first** name selects the preferred band;
    use :func:`visualization_pollutant_priority_from_cfg` for the full panel order.

    If GeoTIFF band descriptions do not match (missing tags, different spelling),
    and ``sector_cfg["pollutants"]`` is provided (or injected via ``sector_cfg=``),
    the band index follows that list order — same order as :func:`write_multiband_geotiff`
    stacks bands — so ``visualization_weight_band`` no longer masks CO/NMVOC incorrectly.
    """
    lookup = dict(viz_cfg or {})
    if sector_cfg is not None:
        pol_order = sector_cfg.get("pollutants")
        if pol_order is not None:
            lookup.setdefault("pollutants", pol_order)

    pol = effective_visualization_pollutant(lookup)
    default_band = int(lookup.get(band_key, 1))
    if pol is None or (isinstance(pol, str) and pol.strip().lower() in ("", "null", "none")):
        return default_band
    tgt = norm_pollutant_key(str(pol))
    try:
        import rasterio

        prefixes = tuple(strip_prefixes)
        with rasterio.open(weight_path) as src:
            for i in range(1, int(src.count) + 1):
                d = src.descriptions[i - 1] if src.descriptions else None
                if d is None:
                    continue
                candidate = str(d).strip()
                nd = norm_pollutant_key(candidate)
                if _band_desc_matches_target(nd, tgt):
                    return i
                short = shorten_w_gnfr_e_weight_band_label(candidate)
                if short is not None and norm_pollutant_key(short) == tgt:
                    return i
                for px in prefixes:
                    if candidate.startswith(px) and _band_desc_matches_target(
                        norm_pollutant_key(candidate[len(px) :]), tgt
                    ):
                        return i
    except Exception:
        pass
    po = lookup.get("pollutants")
    if isinstance(po, (list, tuple)):
        for i, name in enumerate(po):
            if norm_pollutant_key(name) == tgt:
                return i + 1
    return default_band


def pollutant_label_for_band(weight_path: Path, band: int) -> str:
    """Return the GeoTIFF band description or ``"band <n>"``."""
    try:
        import rasterio

        with rasterio.open(weight_path) as src:
            if 1 <= int(band) <= int(src.count):
                d = src.descriptions[int(band) - 1] if src.descriptions else None
                if d:
                    return str(d).strip()
    except Exception:
        pass
    return f"band {int(band)}"


def max_positive_band_value(weight_path: Path, band: int) -> float:
    """Return ``nanmax`` of the raw band, or 0.0 if non-finite / out of range."""
    import rasterio

    with rasterio.open(weight_path) as src:
        if band < 1 or band > int(src.count):
            return 0.0
        a = src.read(band).astype(np.float64)
    return float(np.nanmax(a)) if np.any(np.isfinite(a)) else 0.0


def pick_first_positive_band(
    weight_path: Path,
    preferred: int,
    *,
    empty_message: str,
) -> tuple[int, str | None]:
    """Keep ``preferred`` if it has any positive weight, else fall back to the
    first sibling band with positive values. Returns ``(band, note)`` where
    ``note`` is a human-readable string describing the substitution (``None``
    if no substitution was needed) or ``empty_message`` if every band is empty.
    """
    if max_positive_band_value(weight_path, int(preferred)) > 0.0:
        return int(preferred), None
    import rasterio

    with rasterio.open(weight_path) as src:
        n = int(src.count)
    for b in range(1, n + 1):
        if b == int(preferred):
            continue
        if max_positive_band_value(weight_path, b) > 0.0:
            with rasterio.open(weight_path) as src:
                d = src.descriptions[b - 1] if src.descriptions else None
            return b, (
                f"Band {int(preferred)} has no positive weights; showing band {b} "
                f"({d or 'unlabeled'})."
            )
    return int(preferred), empty_message


def create_folium_map_with_tiles(
    view: ViewContext,
    *,
    zoom_start: int = 8,
    default_basemap: str = "satellite",
) -> Any:
    """Return a :class:`folium.Map` with three basemap options.

    Basemaps:
      * ``"positron"`` - CartoDB Positron (light, clean for weights)
      * ``"satellite"`` - Esri World Imagery (terrain / coast context)
      * ``"osm"`` - CartoDB Voyager (OSM-based, labeled roads / place names)

    ``default_basemap`` controls which tile layer is visible when the HTML
    opens; the other two are off but selectable via the layer control.
    """
    import folium

    wanted = str(default_basemap).strip().lower()
    if wanted not in {"positron", "satellite", "osm"}:
        wanted = "satellite"

    fmap = folium.Map(
        location=list(view.centre),
        zoom_start=zoom_start,
        tiles=None,
        control_scale=True,
    )
    folium.TileLayer(
        "CartoDB positron",
        name="Light (CartoDB Positron)",
        control=True,
        show=(wanted == "positron"),
    ).add_to(fmap)
    folium.TileLayer(
        tiles=_ESRI_TILES,
        attr=_ESRI_ATTR,
        name="Satellite (Esri World Imagery)",
        max_zoom=19,
        control=True,
        show=(wanted == "satellite"),
    ).add_to(fmap)
    folium.TileLayer(
        tiles=_VOYAGER_TILES,
        attr=_VOYAGER_ATTR,
        name="OSM / roads (CartoDB Voyager)",
        max_zoom=19,
        subdomains="abcd",
        control=True,
        show=(wanted == "osm"),
    ).add_to(fmap)
    return fmap


def add_raster_overlay(
    fmap: Any,
    rgba: np.ndarray,
    view: ViewContext,
    *,
    name: str,
    opacity: float,
    show: bool = True,
) -> None:
    """Add an RGBA image overlay wrapped in a :class:`folium.FeatureGroup`."""
    import folium

    from PROXY.visualization._color_debug import viz_color_debug_enabled, viz_color_log

    if viz_color_debug_enabled():
        arr = np.asarray(rgba)
        u_alphas: list[int] = []
        rgb_preview: list[tuple[int, int, int]] = []
        if arr.ndim == 3 and arr.shape[-1] >= 4:
            alpha_ch = arr[..., 3].astype(np.int64).ravel()
            u_alphas = sorted({int(x) for x in np.unique(alpha_ch).tolist()})[:12]
            m = alpha_ch > 0
            if np.any(m):
                rgb_flat = arr[..., :3].reshape(-1, 3)
                sub = rgb_flat[m]
                uniq = np.unique(sub, axis=0)
                for row in uniq[:16]:
                    rgb_preview.append((int(row[0]), int(row[1]), int(row[2])))
        viz_color_log(
            "folium_raster_overlay_queued",
            layer_name=name,
            folium_image_overlay_opacity=float(opacity),
            feature_group_show_default=bool(show),
            rgba_shape=list(arr.shape),
            distinct_alpha_values_in_raster=u_alphas,
            sample_distinct_rgb_where_alpha_positive=rgb_preview,
        )

    fg = folium.FeatureGroup(name=name, show=show)
    folium.raster_layers.ImageOverlay(
        image=rgba,
        bounds=view.bounds,
        mercator_project=True,
        opacity=float(opacity),
        interactive=False,
        cross_origin=False,
    ).add_to(fg)
    fg.add_to(fmap)


def add_cams_grid_overlay(
    fmap: Any,
    grid_fc: dict[str, Any],
    *,
    name: str,
    show: bool = False,
    colour: str = "#1565c0",
    weight: float = 2.0,
    outline_opacity: float = 0.9,
    popup: bool = True,
) -> None:
    """Add the CAMS-cell outline FeatureGroup (no-op if the GeoJSON is empty).

    If ``popup`` is ``True`` and the features carry a ``popup_html`` property
    (populated by
    :func:`PROXY.visualization._click_popup.enrich_cams_grid_with_popups`),
    clicking a cell opens a short HTML card (dominant group when configured)
    in addition to the hover tooltip.
    """
    import folium

    feats = grid_fc.get("features") or []
    has_popup_html = bool(feats) and any(
        isinstance(f.get("properties"), dict) and f["properties"].get("popup_html")
        for f in feats
    )
    use_popup = bool(popup and has_popup_html)

    fg_grid = folium.FeatureGroup(name=name, show=show)
    if not feats:
        fg_grid.add_to(fmap)
        return

    def _style(_f):
        return {
            "fillColor": "#00000000",
            "fillOpacity": 0.0,
            "color": colour,
            "weight": float(weight),
            "opacity": float(outline_opacity),
        }

    gj = folium.GeoJson(
        grid_fc,
        style_function=_style,
        tooltip=folium.GeoJsonTooltip(
            fields=["cams_source_index", "lon_c", "lat_c"],
            aliases=["CAMS source index", "Centre lon", "Centre lat"],
            sticky=True,
        ),
    )
    gj.add_to(fg_grid)

    if use_popup:
        from PROXY.visualization._click_popup import CAMS_POPUP_CSS

        fmap.get_root().header.add_child(folium.Element(CAMS_POPUP_CSS))
        bind_js = (
            "<script>"
            "(function(){"
            "var __bind=function(){"
            f"var layer=window.{gj.get_name()};"
            "if(!layer){setTimeout(__bind,50);return;}"
            "layer.eachLayer(function(sub){"
            "var p=sub.feature&&sub.feature.properties;"
            "if(p&&p.popup_html){"
            "sub.bindPopup(p.popup_html,{maxWidth:380,minWidth:240});"
            "}"
            "});"
            "};"
            "setTimeout(__bind,0);"
            "})();"
            "</script>"
        )
        fmap.get_root().html.add_child(folium.Element(bind_js))

    fg_grid.add_to(fmap)


def clip_rgba_to_cams_mask(
    rgba: np.ndarray,
    *,
    cams_nc_path: Path,
    m_area: np.ndarray,
    ds: Any,
    view: ViewContext,
) -> np.ndarray:
    """Zero the alpha of every RGBA pixel whose centre is outside the CAMS area mask.

    This is the visual fix for "weights bleeding outside the CAMS cells" when the
    raster warp to the display grid smears small values into no-source areas. It
    does not touch the weight raster, only the RGBA payload passed to Folium.

    Input ``rgba`` is modified in place and also returned for chaining.
    """
    from rasterio.transform import xy as transform_xy

    from PROXY.visualization.cams_grid import cams_cell_id_grid

    rows, cols = np.indices((view.gh, view.gw))
    xs, ys = transform_xy(view.dst_t, rows + 0.5, cols + 0.5, offset="center")
    lons = np.asarray(xs, dtype=np.float64).reshape(view.gh, view.gw)
    lats = np.asarray(ys, dtype=np.float64).reshape(view.gh, view.gw)
    cell_id = np.asarray(cams_cell_id_grid(lons, lats, ds, m_area)).reshape(view.gh, view.gw)
    outside = cell_id < 0
    if np.any(outside):
        rgba[outside, 3] = 0
    return rgba


def weight_rgba_log(w_arr: np.ndarray, *, w_nodata: float | None, cmap: str = "plasma") -> np.ndarray:
    """RGBA with ``colour_mode='log'`` (the legacy "global log10" colouring)."""
    return scalar_to_rgba(
        w_arr,
        colour_mode="log",
        cmap_name=cmap,
        hide_zero=True,
        nodata_val=float(w_nodata) if w_nodata is not None else None,
    )


def weight_rgba_percentile(
    w_arr: np.ndarray, *, w_nodata: float | None, cmap: str = "plasma"
) -> np.ndarray:
    """RGBA with ``colour_mode='percentile'`` (2-98% of positives)."""
    return scalar_to_rgba(
        w_arr,
        colour_mode="percentile",
        cmap_name=cmap,
        hide_zero=True,
        nodata_val=float(w_nodata) if w_nodata is not None else None,
    )


def weight_rgba_per_cell(
    w_arr: np.ndarray,
    *,
    w_nodata: float | None,
    cams_nc_path: Path,
    m_area: np.ndarray,
    ds: Any,
    view: ViewContext,
    cmap: str = "plasma",
) -> np.ndarray:
    """Per-CAMS-cell renormalisation to [0, 1] using the sector's CAMS mask.

    Implements the common path used by every sector that supports
    ``weight_display_mode='per_cell'``. The caller is responsible for opening
    the CAMS dataset, building ``m_area`` with its own mask function, and
    closing the dataset afterwards.
    """
    from rasterio.transform import xy as transform_xy

    from PROXY.visualization.cams_grid import (
        cams_cell_id_grid,
        normalize_weights_per_cams_cell,
    )

    rows, cols = np.indices((view.gh, view.gw))
    xs, ys = transform_xy(view.dst_t, rows + 0.5, cols + 0.5, offset="center")
    lons = np.asarray(xs, dtype=np.float64)
    lats = np.asarray(ys, dtype=np.float64)
    cell_id = cams_cell_id_grid(lons, lats, ds, m_area)
    finite = np.isfinite(w_arr)
    if w_nodata is not None:
        finite = finite & (w_arr != float(w_nodata))
    base_valid = finite & (w_arr > 0)
    z01, valid_pc = normalize_weights_per_cams_cell(
        w_arr, cell_id, base_valid=base_valid
    )
    return scalar_to_rgba(
        w_arr,
        colour_mode="global",
        cmap_name=cmap,
        hide_zero=True,
        nodata_val=float(w_nodata) if w_nodata is not None else None,
        z_precomputed_01=z01,
        valid_precomputed=valid_pc,
    )


def build_cams_area_grid_geojson_for_view(
    ds: Any, m_area: np.ndarray, view: ViewContext
) -> dict[str, Any]:
    """Wrapper around :func:`build_cams_area_grid_geojson` using the view bbox."""
    from PROXY.visualization.cams_grid import build_cams_area_grid_geojson

    lon_src = np.asarray(ds["longitude_source"].values).ravel()
    lat_src = np.asarray(ds["latitude_source"].values).ravel()
    return build_cams_area_grid_geojson(ds, m_area, view.bbox_tuple, lon_src, lat_src)


def add_log_percentile_colormap(
    fmap: Any,
    w_arr: np.ndarray,
    *,
    cmap: str = "plasma",
    caption: str = "log10(weight) global 2-98% (positives)",
) -> None:
    """2-98% percentile colormap of log10 of positives (the legacy "global log" legend)."""
    from branca.colormap import LinearColormap

    pos = w_arr[np.isfinite(w_arr) & (w_arr > 0)]
    if pos.size:
        lp = np.log10(np.maximum(pos, 1e-30))
        lo = float(np.percentile(lp, 2.0))
        hi = float(np.percentile(lp, 98.0))
        if lo >= hi:
            hi = lo + 1e-6
    else:
        lo, hi = 0.0, 1.0
    LinearColormap(sample_cmap_hex(cmap, 11), vmin=lo, vmax=hi, caption=caption).add_to(fmap)


def add_linear_percentile_colormap(
    fmap: Any,
    w_arr: np.ndarray,
    *,
    cmap: str = "plasma",
    caption: str = "weight 2-98% (positives, display)",
) -> None:
    """2-98% percentile colormap of positives (linear, used by industry/fugitive/etc.)."""
    from branca.colormap import LinearColormap

    posw = w_arr[np.isfinite(w_arr) & (w_arr > 0)]
    if posw.size:
        wlo = float(np.percentile(posw, 2.0))
        whi = float(np.percentile(posw, 98.0))
        if wlo >= whi:
            whi = wlo + 1e-9
    else:
        wlo, whi = 0.0, 1.0
    LinearColormap(sample_cmap_hex(cmap, 11), vmin=wlo, vmax=whi, caption=caption).add_to(fmap)


def add_per_cell_colormap(
    fmap: Any,
    *,
    cmap: str = "plasma",
    caption: str = "Weight (per-CAMS-cell 0-1)",
) -> None:
    """Fixed 0-1 colormap used when the per-cell restretch is active."""
    from branca.colormap import LinearColormap

    LinearColormap(sample_cmap_hex(cmap, 11), vmin=0.0, vmax=1.0, caption=caption).add_to(fmap)


def save_folium_map(
    fmap: Any,
    out_html: Path,
    root: Path,
    *,
    legend_html: str | None = None,
    add_fullscreen: bool = True,
    add_layer_control: bool = True,
    layer_control_position: str = "topleft",
    layer_control_collapsed: bool = True,
) -> Path:
    """Attach legend + controls and save the map, creating ``out_html.parent`` as needed.

    ``layer_control_position`` defaults to ``"topleft"`` because the unified legend
    panel now pins itself to the top-right; a collapsed-by-default layer control on
    the left keeps the map uncluttered and matches the new layout.
    """
    import folium
    from folium.plugins import Fullscreen

    if legend_html is not None:
        fmap.get_root().html.add_child(folium.Element(legend_html))
    if add_fullscreen:
        Fullscreen(position="bottomright").add_to(fmap)
    if add_layer_control:
        folium.LayerControl(
            collapsed=bool(layer_control_collapsed),
            position=str(layer_control_position),
        ).add_to(fmap)
    out = resolve_under_root(out_html, root)
    out.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out))
    return out
