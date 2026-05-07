#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]

def _ensure_import_path() -> Path:
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root

# Default bounding box covering the Mediterranean + Adriatic around Greece.
# (W, S, E, N) in WGS84 degrees.
DEFAULT_BBOX_WGS84 = (-5.0, 20.0, 35.0, 55.0)
#DEFAULT_BBOX_WGS84 = (21, 39.0, 26.0, 42.0)

def _linspace(a: float, b: float, n: int) -> list[float]:
    if n <= 1:
        return [a]
    step = (b - a) / float(n - 1)
    return [a + i * step for i in range(n)]

def _bbox_perimeter_points_wgs84(
    w: float, s: float, e: float, n: float, *, samples: int = 25
) -> Iterable[tuple[float, float]]:
    xs = _linspace(w, e, samples)
    ys = _linspace(s, n, samples)
    for x in xs:
        yield (x, s)
        yield (x, n)
    for y in ys:
        yield (w, y)
        yield (e, y)

def _bbox_wgs84_to_bounds_3035(
    w: float, s: float, e: float, n: float, *, samples: int = 25
) -> tuple[float, float, float, float]:
    from pyproj import Transformer

    tr = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    xs: list[float] = []
    ys: list[float] = []
    for lon, lat in _bbox_perimeter_points_wgs84(w, s, e, n, samples=samples):
        x, y = tr.transform(lon, lat)
        xs.append(float(x))
        ys.append(float(y))
    return min(xs), min(ys), max(xs), max(ys)

def _auto_resolution_m(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    res_m: float,
    *,
    max_dim: int = 3500,
) -> float:
    import math

    dx = max(float(x1) - float(x0), 1.0)
    dy = max(float(y1) - float(y0), 1.0)
    r = max(float(res_m), 1.0)
    w = int(math.ceil(dx / r))
    h = int(math.ceil(dy / r))
    if max(w, h) <= int(max_dim):
        return r
    scale = float(max(w, h)) / float(max_dim)
    return float(r * scale)

def _build_ref_3035_from_bbox(
    bbox_wgs84: tuple[float, float, float, float],
    *,
    resolution_m: float,
) -> dict[str, object]:
    import math

    from rasterio.crs import CRS
    from rasterio.transform import from_origin

    w, s, e, n = (float(x) for x in bbox_wgs84)
    x0, y0, x1, y1 = _bbox_wgs84_to_bounds_3035(w, s, e, n, samples=25)
    res = _auto_resolution_m(x0, y0, x1, y1, float(resolution_m), max_dim=3500)
    width = int(math.ceil((x1 - x0) / res))
    height = int(math.ceil((y1 - y0) / res))
    tr = from_origin(x0, y1, res, res)
    return {
        "crs": CRS.from_epsg(3035),
        "transform": tr,
        "width": width,
        "height": height,
    }

def _read_viirs_csv_bbox(
    csv_path: Path,
    bbox_wgs84: tuple[float, float, float, float],
) -> "object":
    import numpy as np
    import pandas as pd
    import geopandas as gpd

    w, s, e, n = (float(x) for x in bbox_wgs84)
    usecols = ["Lat_GMTCO", "Lon_GMTCO", "RHI"]
    df = pd.read_csv(csv_path, usecols=lambda c: c in set(usecols), low_memory=False)
    for c in usecols:
        if c not in df.columns:
            raise ValueError(f"VIIRS CSV missing column {c!r}: {csv_path}")

    df = df.replace([1.0e6, 999999], np.nan)
    df = df[np.isfinite(df["Lat_GMTCO"]) & np.isfinite(df["Lon_GMTCO"])]
    df = df[(df["RHI"] < 9.0e5) & (df["RHI"] > 0) & np.isfinite(df["RHI"])]
    df = df[
        (df["Lon_GMTCO"] >= w)
        & (df["Lon_GMTCO"] <= e)
        & (df["Lat_GMTCO"] >= s)
        & (df["Lat_GMTCO"] <= n)
    ].copy()

    if df.empty:
        return gpd.GeoDataFrame(df, geometry=[], crs="EPSG:4326")
    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["Lon_GMTCO"], df["Lat_GMTCO"], crs="EPSG:4326"),
        crs="EPSG:4326",
    )

def _heat_rgba(
    intensity: "object",
    *,
    cmap_name: str = "magma",
    lo_pct: float = 1.0,
    hi_pct: float = 99.8,
    gamma: float = 0.75,
    max_alpha: float = 0.95,
) -> tuple["object", dict[str, float]]:
    import numpy as np
    import matplotlib

    a = np.asarray(intensity, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"Expected 2D intensity array, got shape={a.shape}")

    disp = np.log1p(np.maximum(a, 0.0))
    m = np.isfinite(disp) & (disp > 0)
    if not np.any(m):
        rgba = np.zeros((a.shape[0], a.shape[1], 4), dtype=np.uint8)
        return rgba, {"vmin": 0.0, "vmax": 1.0}

    vmin = float(np.percentile(disp[m], float(lo_pct)))
    vmax = float(np.percentile(disp[m], float(hi_pct)))
    if not np.isfinite(vmin):
        vmin = 0.0
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0

    t = np.zeros_like(disp, dtype=np.float64)
    t[m] = np.clip((disp[m] - vmin) / (vmax - vmin), 0.0, 1.0)
    g = float(gamma)
    if g <= 0:
        g = 1.0
    t[m] = np.power(t[m], g)

    try:
        cmap = matplotlib.colormaps.get_cmap(str(cmap_name))
    except Exception:
        cmap = matplotlib.colormaps["magma"]
    rgb = (cmap(t)[..., :3] * 255.0).astype(np.uint8)
    alpha = np.zeros_like(t, dtype=np.float64)
    alpha[m] = np.clip(t[m] * float(max_alpha), 0.0, 1.0)
    a_u8 = (alpha * 255.0).astype(np.uint8)
    rgba = np.dstack([rgb, a_u8]).astype(np.uint8)
    return rgba, {"vmin": vmin, "vmax": vmax}

def _load_natural_earth_borders(
    bbox_wgs84: tuple[float, float, float, float],
    *,
    resolution: str = "10m",
) -> "object | None":
    """
    Load country borders from Natural Earth data.
    Tries multiple sources in order of preference.
    Returns a GeoDataFrame in EPSG:4326 clipped to bbox, or None on failure.
    """
    import geopandas as gpd
    from shapely.geometry import box

    w, s, e, n = (float(x) for x in bbox_wgs84)
    clip_box = box(w, s, e, n)

    # --- attempt 1: cartopy shapefiles (most reliable, high-res) ---
    try:
        import cartopy.io.shapereader as shpreader
        shpfile = shpreader.natural_earth(
            resolution=resolution,
            category="cultural",
            name="admin_0_countries",
        )
        gdf = gpd.read_file(shpfile)
        gdf = gdf.to_crs("EPSG:4326")
        gdf = gdf[gdf.intersects(clip_box)]
        if not gdf.empty:
            return gdf
    except Exception:
        pass

    # --- attempt 2: geopandas bundled dataset ---
    try:
        world_path = gpd.datasets.get_path("naturalearth_lowres")
        gdf = gpd.read_file(world_path).to_crs("EPSG:4326")
        gdf = gdf[gdf.intersects(clip_box)]
        if not gdf.empty:
            return gdf
    except Exception:
        pass

    # --- attempt 3: geodatasets / geopandas new API ---
    try:
        import geodatasets
        gdf = gpd.read_file(geodatasets.get_path("naturalearth.land"))
        gdf = gdf.to_crs("EPSG:4326")
        gdf = gdf[gdf.intersects(clip_box)]
        if not gdf.empty:
            return gdf
    except Exception:
        pass

    return None

def _render_png(
    out_png: Path,
    *,
    bbox_wgs84: tuple[float, float, float, float],
    overlay_rgba: "object",
    ref: dict[str, object],
    flares_wgs84: "object",
    title: str,
    use_basemap: bool,
    zoom_adjust: int | None,
    draw_boundaries: bool,
    bg: str,
    basemap_brightness: float,
) -> None:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from rasterio.transform import array_bounds
    from rasterio.warp import transform_bounds

    h = int(ref["height"])
    w = int(ref["width"])
    tr = ref["transform"]
    l, b_m, r, t_m = array_bounds(h, w, tr)
    W, S, E, N = transform_bounds(str(ref["crs"]), "EPSG:4326", l, b_m, r, t_m)

    w0, s0, e0, n0 = (float(x) for x in bbox_wgs84)

    # ------------------------------------------------------------------
    # Build the figure early so we can use axes-level drawing throughout.
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(1, 1, figsize=(11, 8), dpi=180)

    # ------------------------------------------------------------------
    # Background fill
    # ------------------------------------------------------------------
    bg_l = str(bg or "").strip().lower()
    if bg_l in ("white", "w"):
        bg_color = "#ffffff"
    elif bg_l in ("gray", "grey", "g"):
        bg_color = "#181818"   # very dark grey — not pure black, not washed out
    else:
        bg_color = "#000000"

    ax.set_facecolor(bg_color)
    fig.patch.set_facecolor(bg_color)

    # ------------------------------------------------------------------
    # Optional OSM basemap (only when explicitly requested)
    # ------------------------------------------------------------------
    if use_basemap:
        try:
            from PROXY.tools.waste_proxy_bbox_images import _composite_rgba_over_osm
            out_img = _composite_rgba_over_osm(
                overlay_rgba, tr, (h, w), W, S, E, N,
                zoom_adjust=zoom_adjust,
            )
            bb = max(float(basemap_brightness), 0.1)
            if bb != 1.0:
                arr = np.asarray(out_img, dtype=np.float32) * bb
                out_img = np.clip(arr, 0.0, 255.0).astype(np.uint8)
            ax.imshow(
                np.asarray(out_img),
                extent=[W, E, S, N],
                origin="upper",
                interpolation="bilinear",
                zorder=1,
            )
        except Exception as exc:
            print(f"  [warn] basemap failed ({exc}), falling back to plain background.")
            use_basemap = False   # fall through to heat overlay only

    # ------------------------------------------------------------------
    # Heat overlay (always drawn)
    # ------------------------------------------------------------------
    # Convert RGBA uint8 → float for imshow (handles transparency correctly).
    ov = np.asarray(overlay_rgba, dtype=np.float32) / 255.0   # shape (H, W, 4)
    ax.imshow(
        ov,
        extent=[W, E, S, N],
        origin="upper",
        interpolation="bilinear",
        zorder=2,
    )

    # ------------------------------------------------------------------
    # Country / coastline borders  ← the important part
    # ------------------------------------------------------------------
    if draw_boundaries:
        borders = _load_natural_earth_borders(bbox_wgs84)
        if borders is not None and not borders.empty:
            # Fill land with a subtle dark tone so sea vs land is legible.
            borders.plot(
                ax=ax,
                facecolor="#1e1e2e",   # very dark blue-grey land fill
                edgecolor="#c8c8d0",   # crisp light-grey border
                linewidth=0.9,
                alpha=1.0,
                zorder=3,              # below heat overlay
            )
            # Re-draw the heat on top (imshow is already at zorder=2, but
            # we need borders at zorder=3 so the heat at zorder=4 wins).
            # Easiest fix: re-raise the heat overlay zorder here.
            # Instead let's just re-order: land fill z=2, heat z=3, border lines z=4.
            # We already drew heat at z=2 above; borders at z=3 would hide heat.
            # So draw ONLY border *lines* at z=4 (separate from fill).
            borders.boundary.plot(
                ax=ax,
                edgecolor="#d0d0dc",
                linewidth=0.9,
                alpha=0.95,
                zorder=5,
            )
        else:
            print("  [warn] Could not load Natural Earth borders.")

    # Re-draw heat overlay at the top zorder so it sits above the land fill.
    # (The earlier imshow at z=2 is now obscured by land fill at z=3.)
    ax.imshow(
        ov,
        extent=[W, E, S, N],
        origin="upper",
        interpolation="bilinear",
        zorder=4,
    )

    # ------------------------------------------------------------------
    # Faint sample-location dots (optional context)
    # ------------------------------------------------------------------
    try:
        g = flares_wgs84
        if hasattr(g, "empty") and not g.empty:
            pts = g[["Lon_GMTCO", "Lat_GMTCO"]].to_numpy(dtype=np.float64)
            pts = pts[np.isfinite(pts[:, 0]) & np.isfinite(pts[:, 1])]
            if len(pts) > 0:
                ax.scatter(
                    pts[:, 0], pts[:, 1],
                    s=2.5,
                    c="#00eeff",
                    alpha=0.55,
                    linewidths=0,
                    zorder=6,
                )
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Axes cosmetics
    # ------------------------------------------------------------------
    ax.set_xlim(w0, e0)
    ax.set_ylim(s0, n0)
    ax.set_xlabel("Longitude (°E)", fontsize=9, color="#cccccc")
    ax.set_ylabel("Latitude (°N)", fontsize=9, color="#cccccc")
    ax.tick_params(labelsize=8, colors="#aaaaaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#555555")
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6, color="#eeeeee")
    ax.grid(True, color="#444444", linestyle="--", linewidth=0.35, alpha=0.5, zorder=7)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.4)
    fig.savefig(out_png, dpi=180, bbox_inches="tight", facecolor=bg_color)
    plt.close(fig)
    print(f"  Saved: {out_png}")


def main(argv: list[str] | None = None) -> int:
    _ensure_import_path()

    p = argparse.ArgumentParser(
        description="Map VIIRS Nightfire flares with Gaussian intensity over a WGS84 bbox."
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=Path("INPUT/Proxy/ProxySpecific/Fugitive/VNF_npp_d20210101_noaa_v30.csv"),
        help="Path to VIIRS Nightfire CSV",
    )
    p.add_argument(
        "--bbox",
        nargs=4, type=float, metavar=("W", "S", "E", "N"),
        default=DEFAULT_BBOX_WGS84,
    )
    p.add_argument(
        "--sigma-m",
        type=float,
        default=3000.0,          # ← changed from 1500 to 3000
        help="Gaussian sigma in metres (default: 3000)",
    )
    p.add_argument(
        "--resolution-m",
        type=float,
        default=1200.0,
        help="Raster resolution in metres (auto-increases if image would be too large)",
    )
    p.add_argument(
        "--basemap",
        action="store_true",
        default=False,           # ← basemap OFF by default
        help="Enable OSM basemap (disabled by default; clean border rendering is the default)",
    )
    p.add_argument(
        "--no-boundaries",
        action="store_true",
        help="Disable country boundary overlay",
    )
    p.add_argument(
        "--bg",
        type=str,
        default="gray",
        help="Background colour when basemap is off: black | gray | white  (default: gray)",
    )
    p.add_argument(
        "--basemap-brightness",
        type=float,
        default=0.55,
        help="Multiply basemap RGB by this factor when --basemap is set (default: 0.55)",
    )
    p.add_argument(
        "--zoom-adjust",
        type=int,
        default=-1,
        help="OSM basemap zoom adjustment (contextily). -1 is a good default for large bboxes.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("Output/images/viirs_flares_gaussian_med_adriatic.png"),
        help="Output PNG path",
    )
    args = p.parse_args(argv)

    from PROXY.sectors.D_Fugitive.fugitive_proxy import accumulate_viirs_gaussian_raster

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    bbox = tuple(float(x) for x in args.bbox)
    flares_wgs84 = _read_viirs_csv_bbox(csv_path, bbox)
    ref = _build_ref_3035_from_bbox(bbox, resolution_m=float(args.resolution_m))
    flares_3035 = flares_wgs84.to_crs(ref["crs"]) if hasattr(flares_wgs84, "to_crs") else flares_wgs84

    intensity = accumulate_viirs_gaussian_raster(
        flares_3035,
        "RHI",
        ref,
        sigma_m=float(args.sigma_m),
    )

    w, s, e, n = bbox
    overlay_rgba, _ = _heat_rgba(
        intensity,
        cmap_name="magma",
        lo_pct=1.0,
        hi_pct=99.8,
        gamma=0.75,
        max_alpha=0.95,
    )
    title = (
        f"VIIRS Nightfire flares · Gaussian intensity (σ = {float(args.sigma_m):.0f} m) · log₁p scale\n"
        f"BBOX  W={w:.2f}°  S={s:.2f}°  E={e:.2f}°  N={n:.2f}°"
    )

    _render_png(
        Path(args.out),
        bbox_wgs84=bbox,
        overlay_rgba=overlay_rgba,
        ref=ref,
        flares_wgs84=flares_wgs84,
        title=title,
        use_basemap=bool(args.basemap),
        zoom_adjust=int(args.zoom_adjust) if args.zoom_adjust is not None else None,
        draw_boundaries=not bool(args.no_boundaries),
        bg=str(args.bg),
        basemap_brightness=float(args.basemap_brightness),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
