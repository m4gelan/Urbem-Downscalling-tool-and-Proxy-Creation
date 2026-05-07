#!/usr/bin/env python3
"""Export three PNG previews for G_Shipping over a WGS84 bbox.

Improvements:
- Fixed CRS alignment between OSM basemap (EPSG:3857) and overlay (EPSG:4326)
- Proper aspect ratio correction for geographic projections
- Enhanced visualization with titles and annotations (no colorbar)
- Better error handling and progress feedback
- Anti-aliasing and proper image compositing

Writes:
  vessel density (EMODnet raw),
  OSM port coverage,
  CORINE port-area fraction.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_import_path() -> Path:
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


# Layer titles must match PROXY.visualization.shipping_context.build_shipping_proxy_rgba_overlays
_EXPORT: tuple[tuple[str, str], ...] = (
    (
        "G_Shipping · EMODnet: raw density",
        "shipping_bbox_vessel_density.png",
    ),
    (
        "G_Shipping · OSM: coverage",
        "shipping_bbox_osm_port_area.png",
    ),
    (
        "G_Shipping · CORINE: port fraction",
        "shipping_bbox_corine_port_area.png",
    ),
)


# ─────────────────────────────────────────────────────────────────────────────
# Core compositing logic (fixed alignment)
# ─────────────────────────────────────────────────────────────────────────────

class CompositeResult(NamedTuple):
    rgb: np.ndarray          # (H, W, 3) uint8
    alpha_mask: np.ndarray   # (H, W) bool — True where overlay is present


def _reproject_basemap_to_overlay_grid(
    img_merc: np.ndarray,
    extent_merc: tuple[float, float, float, float],
    dst_transform,
    dst_shape: tuple[int, int],
) -> np.ndarray:
    """
    Reproject OSM Web Mercator basemap onto the overlay's EPSG:4326 pixel grid.

    Parameters
    ----------
    img_merc     : (H, W, C) uint8 array in EPSG:3857
    extent_merc  : (left, right, bottom, top) in EPSG:3857 metres
    dst_transform: rasterio Affine for the destination EPSG:4326 grid
    dst_shape    : (height, width) of the destination grid

    Returns
    -------
    base_rgb : (gh, gw, 3) float32 in [0, 1]
    """
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.warp import reproject, Resampling

    gh, gw = dst_shape
    left, right, bottom, top = extent_merc
    hm, wm = img_merc.shape[:2]

    src_transform = from_bounds(left, bottom, right, top, wm, hm)
    crs3857 = rasterio.crs.CRS.from_epsg(3857)
    crs4326 = rasterio.crs.CRS.from_epsg(4326)

    # Fallback fill: light-gray background for out-of-bounds pixels
    FALLBACK_GRAY = 0.933  # ≈ #EEEEEE

    base_rgb = np.full((gh, gw, 3), FALLBACK_GRAY, dtype=np.float32)
    src_arr = np.asarray(img_merc[..., :3], dtype=np.float32) / 255.0

    for k in range(3):
        dst_band = np.full((gh, gw), np.nan, dtype=np.float64)
        reproject(
            source=src_arr[..., k].astype(np.float64),
            destination=dst_band,
            src_transform=src_transform,
            src_crs=crs3857,
            dst_transform=dst_transform,
            dst_crs=crs4326,
            resampling=Resampling.bilinear,
            dst_nodata=np.nan,
        )
        valid = np.isfinite(dst_band)
        base_rgb[..., k] = np.where(valid, dst_band.astype(np.float32), FALLBACK_GRAY)

    return base_rgb


def _alpha_composite(
    base_rgb: np.ndarray,   # (H, W, 3) float32 [0,1]
    overlay_rgba: np.ndarray,  # (H, W, 4) uint8
) -> tuple[np.ndarray, np.ndarray]:
    """
    Standard Porter-Duff 'over' compositing.

    Returns
    -------
    out_rgb    : (H, W, 3) uint8
    alpha_mask : (H, W) bool
    """
    over = overlay_rgba.astype(np.float32) / 255.0
    alpha = np.clip(over[..., 3:4], 0.0, 1.0)          # (H, W, 1)
    rgb_o = over[..., :3]                               # (H, W, 3)

    # Porter-Duff OVER
    out = rgb_o * alpha + base_rgb * (1.0 - alpha)
    out_uint8 = (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)

    alpha_mask = (overlay_rgba[..., 3] > 0)
    return out_uint8, alpha_mask


def composite_rgba_over_osm(
    rgba: np.ndarray,
    dst_transform,
    dst_shape: tuple[int, int],
    west: float,
    south: float,
    east: float,
    north: float,
    *,
    zoom_adjust: int | None = None,
) -> CompositeResult:
    """
    Fetch OSM tiles, reproject to overlay grid, alpha-blend overlay on top.

    The overlay is in EPSG:4326 (regular lon/lat grid).
    Tiles are fetched in Web Mercator (EPSG:3857) then reprojected —
    this avoids the 'twisted' appearance caused by naively stretching
    Mercator tiles to fit a geographic bounding box.
    """
    import contextily as ctx

    gh, gw = int(dst_shape[0]), int(dst_shape[1])
    if rgba.shape[:2] != (gh, gw):
        raise ValueError(
            f"RGBA shape {rgba.shape[:2]} does not match dst_shape ({gh}, {gw})"
        )

    # ── 1. Fetch OSM tiles in Web Mercator ──────────────────────────────────
    img_merc, extent = ctx.bounds2img(
        west, south, east, north,
        zoom="auto",
        ll=True,                                # input coords are lon/lat
        source=ctx.providers.OpenStreetMap.Mapnik,
        zoom_adjust=zoom_adjust,
    )
    # extent = (left, right, bottom, top) in EPSG:3857

    # ── 2. Reproject basemap onto overlay pixel grid ─────────────────────────
    base_rgb = _reproject_basemap_to_overlay_grid(
        img_merc, extent, dst_transform, (gh, gw)
    )

    # ── 3. Porter-Duff composite ─────────────────────────────────────────────
    rgb, alpha_mask = _alpha_composite(base_rgb, rgba)
    return CompositeResult(rgb=rgb, alpha_mask=alpha_mask)


# ─────────────────────────────────────────────────────────────────────────────
# Enhanced figure rendering
# ─────────────────────────────────────────────────────────────────────────────

def _add_map_decorations(
    ax,
    title: str,
    west: float,
    south: float,
    east: float,
    north: float,
    rgb: np.ndarray,
) -> None:
    """
    Add title, axis labels, gridlines, and north arrow to a matplotlib Axes.

    Uses geographic aspect on the axes (Plate Carrée with cosine latitude scaling)
    so distances look consistent. Do not pass ``aspect='auto'`` to ``imshow``: that
    stretches pixels and overrides ``set_aspect``, which caused the twisted map.
    """

    # ── Axes limits ──────────────────────────────────────────────────────────
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)

    # One degree of longitude is shorter in km than one degree of latitude by cos(lat).
    lat_centre = (south + north) / 2.0
    cos_lat = np.cos(np.radians(lat_centre))
    ax.set_aspect(1.0 / cos_lat, adjustable="box")

    # ── Display the composited image (extent = left, right, bottom, top) ──────
    ax.imshow(
        rgb,
        extent=[west, east, south, north],
        origin="upper",
        interpolation="lanczos",
        zorder=0,
    )

    # ── Gridlines ────────────────────────────────────────────────────────────
    ax.grid(
        True,
        color="white",
        linewidth=0.4,
        linestyle="--",
        alpha=0.6,
        zorder=1,
    )

    # ── Axis labels ──────────────────────────────────────────────────────────
    ax.set_xlabel("Longitude (°E)", fontsize=9, labelpad=4)
    ax.set_ylabel("Latitude (°N)", fontsize=9, labelpad=4)
    ax.tick_params(labelsize=8)

    # ── Title ────────────────────────────────────────────────────────────────
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)

    # ── North arrow ──────────────────────────────────────────────────────────
    # Placed in the upper-right corner of the axes in axes-fraction coordinates
    ax.annotate(
        "N",
        xy=(0.965, 0.92),
        xycoords="axes fraction",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color="black",
        zorder=5,
    )
    ax.annotate(
        "",
        xy=(0.965, 0.92),
        xytext=(0.965, 0.80),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(
            arrowstyle="-|>",
            color="black",
            lw=1.5,
        ),
        zorder=5,
    )

    # ── Scale bar ────────────────────────────────────────────────────────────
    _add_scale_bar(ax, west, south, east, north, lat_centre)

    # ── Attribution ──────────────────────────────────────────────────────────
    ax.text(
        0.01, 0.01,
        "© OpenStreetMap contributors",
        transform=ax.transAxes,
        fontsize=6,
        color="gray",
        va="bottom",
        ha="left",
        zorder=6,
    )


def _add_scale_bar(
    ax,
    west: float,
    south: float,
    east: float,
    north: float,
    lat_centre: float,
    target_fraction: float = 0.20,
) -> None:
    """
    Draw a simple scale bar in the lower-right corner.

    The bar represents a round-number distance (km) that fills roughly
    ``target_fraction`` of the map width.
    """

    # Approximate km per degree of longitude at the map centre
    km_per_deg_lon = 111.32 * np.cos(np.radians(lat_centre))
    map_width_km = (east - west) * km_per_deg_lon
    target_km = map_width_km * target_fraction

    # Snap to a nice round number
    magnitude = 10 ** int(np.floor(np.log10(target_km)))
    for step in (1, 2, 5, 10):
        candidate = step * magnitude
        if candidate >= target_km * 0.5:
            bar_km = candidate
            break
    else:
        bar_km = magnitude

    bar_deg = bar_km / km_per_deg_lon

    # Position: bottom-right, 3 % inset
    x1 = east  - 0.03 * (east - west) - bar_deg
    x2 = x1 + bar_deg
    y  = south + 0.04 * (north - south)

    ax.plot([x1, x2], [y, y], color="black", linewidth=2, zorder=5, solid_capstyle="butt")
    ax.plot([x1, x1], [y - 0.003 * (north - south), y + 0.003 * (north - south)],
            color="black", linewidth=2, zorder=5)
    ax.plot([x2, x2], [y - 0.003 * (north - south), y + 0.003 * (north - south)],
            color="black", linewidth=2, zorder=5)
    ax.text(
        (x1 + x2) / 2,
        y + 0.008 * (north - south),
        f"{bar_km:.0f} km",
        ha="center",
        va="bottom",
        fontsize=7,
        color="black",
        zorder=5,
    )


def save_enhanced_png(
    rgb: np.ndarray,
    out_path: Path,
    title: str,
    west: float,
    south: float,
    east: float,
    north: float,
    dpi: int = 150,
) -> None:
    """
    Render the composited RGB array with map decorations and save to PNG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gh, gw = rgb.shape[:2]
    lon_extent = east - west
    lat_extent = north - south
    lat_c = np.radians((south + north) / 2.0)
    # Figure aspect (height/width) matching geographic proportions so axes + tight_layout do not squash.
    geo_height_over_width = lat_extent / (lon_extent * np.cos(lat_c))

    fig_w = max(6.0, gw / dpi)
    fig_h = max(4.5, fig_w * geo_height_over_width + 0.9)

    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=dpi)

    _add_map_decorations(ax, title, west, south, east, north, rgb)

    fig.tight_layout(pad=0.5)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    root = _ensure_import_path()

    ap = argparse.ArgumentParser(
        description=(
            "Export G_Shipping proxy context images "
            "(vessel density, OSM ports, CORINE ports) for a WGS84 bbox."
        )
    )
    ap.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        required=True,
        help="Bounding box in WGS84 degrees: west south east north.",
    )
    ap.add_argument("--root",        type=Path, default=root)
    ap.add_argument("--paths-yaml",  type=Path, default=None)
    ap.add_argument("--sector-yaml", type=Path, default=None)
    ap.add_argument("--weight-tif",  type=Path, default=None)
    ap.add_argument("--out-dir",     type=Path, default=None)
    ap.add_argument("--country",     default="EL")
    ap.add_argument("--max-width",   type=int,  default=1400)
    ap.add_argument("--max-height",  type=int,  default=1200)
    ap.add_argument("--pad-deg",     type=float, default=0.0)
    ap.add_argument("--dpi",         type=int,  default=150,
                    help="Output PNG resolution (default 150 dpi).")
    ap.add_argument(
        "--no-basemap",
        action="store_true",
        help="Skip OSM basemap; save raw RGBA overlay only.",
    )
    ap.add_argument(
        "--basemap-zoom-adjust",
        type=int,
        default=None,
        help="Optional zoom level adjustment for contextily tile fetch.",
    )
    args = ap.parse_args()

    # ── Resolve paths ─────────────────────────────────────────────────────────
    paths_yaml  = args.paths_yaml  or (args.root / "PROXY" / "config" / "paths.yaml")
    sector_yaml = args.sector_yaml or (args.root / "PROXY" / "config" / "sectors" / "shipping.yaml")
    wt_default  = args.root / "OUTPUT" / "Proxy_weights" / "G_Shipping" / "shipping_areasource.tif"
    weight_tif  = args.weight_tif or wt_default
    wt = weight_tif if weight_tif.is_absolute() else args.root / weight_tif

    for label, path in [
        ("paths.yaml",  paths_yaml),
        ("sector YAML", sector_yaml),
        ("weight GeoTIFF", wt),
    ]:
        if not path.is_file():
            print(f"ERROR – {label} not found: {path}", file=sys.stderr)
            return 1

    # ── Load configs ──────────────────────────────────────────────────────────
    import yaml

    with paths_yaml.open(encoding="utf-8") as f:
        path_cfg = yaml.safe_load(f)
    with sector_yaml.open(encoding="utf-8") as f:
        sector_cfg = yaml.safe_load(f)

    for label, cfg in [("paths.yaml", path_cfg), ("sector YAML", sector_cfg)]:
        if not isinstance(cfg, dict):
            print(f"ERROR – {label} must parse to a mapping.", file=sys.stderr)
            return 1

    # ── Validate bbox ─────────────────────────────────────────────────────────
    west, south, east, north = (float(x) for x in args.bbox)
    if west >= east or south >= north:
        print("ERROR – Invalid bbox: require west < east and south < north.", file=sys.stderr)
        return 1

    out_dir = (args.out_dir.resolve() if args.out_dir else Path.cwd().resolve())
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Optional contextily ───────────────────────────────────────────────────
    try:
        import contextily  # noqa: F401
        have_contextily = True
    except ImportError:
        have_contextily = False
        if not args.no_basemap:
            print(
                "WARNING – contextily not installed; saving overlays without basemap.\n"
                "          Install with: pip install contextily",
                file=sys.stderr,
            )

    use_basemap = (not args.no_basemap) and have_contextily

    # ── Build overlays ────────────────────────────────────────────────────────
    from PROXY.sectors.G_Shipping.pipeline import merge_shipping_pipeline_cfg
    from PROXY.visualization._mapbuilder import compute_view_context, resolve_under_root
    from PROXY.visualization.shipping_context import build_shipping_proxy_rgba_overlays

    wt_resolved = resolve_under_root(wt, args.root)
    if not wt_resolved.is_file():
        print(f"ERROR – Weight GeoTIFF not found: {wt_resolved}", file=sys.stderr)
        return 1

    ship_merged = merge_shipping_pipeline_cfg(
        args.root, path_cfg, sector_cfg,
        country=str(args.country),
        output_dir=wt_resolved.parent.resolve(),
    )

    view = compute_view_context(
        wt_resolved,
        pad_deg=float(args.pad_deg),
        max_width=int(args.max_width),
        max_height=int(args.max_height),
        override_bbox=(west, south, east, north),
    )

    print(
        f"Grid: {view.gw} × {view.gh} px  |  "
        f"bbox: [{view.west:.4f}, {view.south:.4f}, {view.east:.4f}, {view.north:.4f}]"
    )

    overlays = build_shipping_proxy_rgba_overlays(
        args.root, ship_merged, wt_resolved,
        view.west, view.south, view.east, view.north,
        view.dst_t, (view.gh, view.gw),
        path_cfg,
        resampling="bilinear",
    )

    by_title = {t: rgba for t, _, rgba in overlays}
    missing: list[str] = []
    written: list[Path] = []

    # ── Export each layer ─────────────────────────────────────────────────────
    for title, fname in _EXPORT:
        rgba = by_title.get(title)
        if rgba is None:
            missing.append(title)
            continue

        outp = out_dir / fname
        print(f"\nRendering: {title}")

        if use_basemap:
            try:
                result = composite_rgba_over_osm(
                    rgba,
                    view.dst_t,
                    (view.gh, view.gw),
                    view.west, view.south, view.east, view.north,
                    zoom_adjust=args.basemap_zoom_adjust,
                )
                rgb = result.rgb
            except Exception as exc:
                print(
                    f"  WARNING – Basemap failed ({exc}); saving overlay only.",
                    file=sys.stderr,
                )
                # Fallback: white background
                base_white = np.full((*rgba.shape[:2], 3), 255, dtype=np.uint8)
                base_f = base_white.astype(np.float32) / 255.0
                rgb, _ = _alpha_composite(base_f, rgba)
        else:
            # White background fallback
            base_white = np.full((*rgba.shape[:2], 3), 255, dtype=np.uint8)
            base_f = base_white.astype(np.float32) / 255.0
            rgb, _ = _alpha_composite(base_f, rgba)

        save_enhanced_png(
            rgb,
            outp,
            title=title,
            west=view.west,
            south=view.south,
            east=view.east,
            north=view.north,
            dpi=args.dpi,
        )
        written.append(outp)

    # ── Attribution ───────────────────────────────────────────────────────────
    if use_basemap and written:
        print(
            "\nBasemap: OpenStreetMap – https://www.openstreetmap.org/copyright",
            file=sys.stderr,
        )

    # ── Report missing layers ─────────────────────────────────────────────────
    if missing:
        print(
            "\nERROR – Could not build some layers "
            "(check EMODnet, OSM GPKG, CORINE paths in paths.yaml "
            "and that the bbox overlaps the weight raster).\nMissing:",
            file=sys.stderr,
        )
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        return 1

    print(f"\nDone — {len(written)} file(s) written to: {out_dir}")
    for p in written:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
