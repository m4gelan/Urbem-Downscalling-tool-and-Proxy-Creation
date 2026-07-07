from __future__ import annotations

import math
import urllib.request
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject

from UrbEm_Visualizer.visualization.emission_style import EPS, colormap_for
from UrbEm_Visualizer.visualization.geojson_layers import domain_bbox_geojson, points_geojson
from UrbEm_Visualizer.visualization.load_run import RunContext
from UrbEm_Visualizer.visualization.scale import _fmt_sci
from UrbEm_Visualizer.visualization.tiles import _cmap_lut

# Web Mercator constants
_R = 6378137.0
_ORIGIN = math.pi * _R  # 20037508.342789244
_TILE = 256

_MAX_PX = 4096
_BG = (15, 17, 23, 255)
_PANEL = (24, 28, 37, 255)
_CAMS_OUTLINE = (160, 168, 190, 70)
_DOMAIN_OUTLINE = (79, 124, 255, 255)
_TEXT = (232, 234, 240)
_TEXT_DIM = (139, 145, 168)
_ACCENT = (79, 124, 255)
_TILE_SUBDOMAINS = ("a", "b", "c")

_TILE_CACHE: dict[str, Image.Image | None] = {}


def _hex_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _lonlat_to_global_px(lon: float, lat: float, z: int) -> tuple[float, float]:
    world = _TILE * (2 ** z)
    x = (lon + 180.0) / 360.0 * world
    siny = math.sin(math.radians(lat))
    siny = min(max(siny, -0.9999), 0.9999)
    y = (0.5 - math.log((1 + siny) / (1 - siny)) / (4 * math.pi)) * world
    return x, y


def _choose_zoom(west: float, east: float, target_w: int) -> int:
    frac = (east - west) / 360.0
    if frac <= 0:
        return 12
    z = math.log2(target_w / (frac * _TILE))
    return int(max(1, min(18, math.floor(z))))


def _fetch_tile(url: str) -> Image.Image | None:
    if url in _TILE_CACHE:
        return _TILE_CACHE[url]
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "UrbEm-Visualizer/1.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            tile = Image.open(BytesIO(resp.read())).convert("RGBA")
    except Exception:
        tile = None
    _TILE_CACHE[url] = tile
    return tile


def _paste_basemap(base: Image.Image, url: str, z: int, left: int, top: int, w: int, h: int) -> None:
    n = 2 ** z
    tx0 = left // _TILE
    tx1 = (left + w) // _TILE
    ty0 = top // _TILE
    ty1 = (top + h) // _TILE
    for tx in range(tx0, tx1 + 1):
        for ty in range(ty0, ty1 + 1):
            if ty < 0 or ty >= n:
                continue
            txx = tx % n
            sub = _TILE_SUBDOMAINS[(tx + ty) % len(_TILE_SUBDOMAINS)]
            tile_url = (
                url.replace("{s}", sub)
                .replace("{z}", str(z))
                .replace("{x}", str(txx))
                .replace("{y}", str(ty))
                .replace("{r}", "")
            )
            tile = _fetch_tile(tile_url)
            if tile is None:
                continue
            base.paste(tile, (tx * _TILE - left, ty * _TILE - top))


def _emission_overlay(
    raster,
    lower: float,
    upper: float,
    threshold: float,
    cmap: np.ndarray,
    z: int,
    left: int,
    top: int,
    w: int,
    h: int,
) -> Image.Image:
    mpp = (2 * _ORIGIN) / (_TILE * (2 ** z))
    left_m = -_ORIGIN + left * mpp
    right_m = -_ORIGIN + (left + w) * mpp
    top_m = _ORIGIN - top * mpp
    bottom_m = _ORIGIN - (top + h) * mpp
    dst_transform = from_bounds(left_m, bottom_m, right_m, top_m, w, h)

    dst = np.zeros((h, w), dtype=np.float32)
    reproject(
        source=raster.data,
        destination=dst,
        src_transform=raster.transform,
        src_crs=CRS.from_string(raster.crs),
        dst_transform=dst_transform,
        dst_crs=CRS.from_epsg(3857),
        resampling=Resampling.nearest,
        src_nodata=0,
        dst_nodata=0,
    )

    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    positive = dst > 0
    if np.any(positive):
        logv = np.log10(np.maximum(dst, 0.0) + EPS)
        denom = max(upper - lower, 1e-9)
        norm = np.clip((logv - lower) / denom, 0.0, 1.0)
        idx = np.clip((norm * 255).astype(np.int32), 0, 255)
        rgba[..., :3] = cmap[idx][..., :3]

        alpha = np.zeros((h, w), dtype=np.float32)
        below = dst < threshold
        alpha[below & positive] = 0.08
        visible = positive & ~below
        alpha[visible] = np.clip(0.42 + norm[visible] * 0.48, 0.42, 0.9)
        rgba[..., 3] = (alpha * 255).astype(np.uint8)
        rgba[~positive, 3] = 0
    return Image.fromarray(rgba, mode="RGBA")


def _draw_colorbar(
    draw: ImageDraw.ImageDraw,
    cmap: np.ndarray,
    lower: float,
    upper: float,
    threshold: float,
    x: int,
    y: int,
    w: int,
    h: int,
) -> None:
    # gradient bar
    for i in range(w):
        c = cmap[int(i / max(w - 1, 1) * 255)]
        draw.line([(x + i, y), (x + i, y + h)], fill=(int(c[0]), int(c[1]), int(c[2]), 255))
    draw.rectangle([x, y, x + w, y + h], outline=(255, 255, 255, 60), width=1)

    f_lbl = _font(15)
    f_small = _font(13)

    vmin = 10 ** lower
    vmax = 10 ** upper
    draw.text((x, y + h + 8), _fmt_sci(vmin), font=f_lbl, fill=_TEXT)
    rt = _fmt_sci(vmax)
    rtw = draw.textlength(rt, font=f_lbl)
    draw.text((x + w - rtw, y + h + 8), rt, font=f_lbl, fill=_TEXT)

    # threshold marker
    denom = max(upper - lower, 1e-9)
    lv = math.log10(max(threshold, 0.0) + EPS)
    norm = max(0.0, min(1.0, (lv - lower) / denom))
    mx = x + int(norm * w)
    draw.line([(mx, y - 7), (mx, y + h + 7)], fill=_ACCENT + (255,), width=2)
    lbl = f"hide < {_fmt_sci(threshold)}"
    lw = draw.textlength(lbl, font=f_small)
    tx = min(max(mx - lw / 2, x), x + w - lw)
    draw.text((tx, y - 24), lbl, font=f_small, fill=_ACCENT + (255,))


def _load_cams_geojson(ctx: RunContext) -> dict[str, Any]:
    from pathlib import Path

    from UrbEm_Visualizer.dataset_loaders.cams_grid import load_domain_cams_geojson
    from UrbEm_Visualizer.paths import project_root

    cfg = ctx.config
    cams_rel = (cfg.get("paths") or {}).get("cams")
    if not cams_rel or not cfg.get("country") or not cfg.get("year"):
        raise ValueError("CAMS path or country/year missing in session config")
    cams_path = Path(cams_rel)
    if not cams_path.is_absolute():
        cams_path = project_root() / cams_rel
    if not cams_path.is_file():
        raise ValueError(f"CAMS file not found: {cams_path}")
    return load_domain_cams_geojson(
        cams_path,
        str(cfg["country"]),
        int(cfg.get("emissions_year") or cfg["year"]),
        list(cfg["pollutants"]),
        ctx.domain,
    )


def _primary_area_sector(area_sectors: list[str]) -> str:
    for sid in area_sectors:
        if sid != "TOTAL":
            return sid
    return "TOTAL"


def render_map_view_png(
    ctx: RunContext,
    west: float,
    south: float,
    east: float,
    north: float,
    width: int,
    height: int,
    pollutant: str,
    threshold: float,
    area_sectors: list[str],
    point_sectors: list[str],
    basemap_url: str | None = None,
    **_kwargs: Any,
) -> bytes:
    if east <= west or north <= south:
        raise ValueError("invalid map bounds")

    sector_id = _primary_area_sector(area_sectors)
    scale = ctx.scale_for(sector_id, pollutant)
    if not scale:
        raise ValueError(f"no emission scale for {sector_id} / {pollutant}")
    lower = float(scale["lower_bound"])
    upper = float(scale["upper_bound"])
    cmap = _cmap_lut(colormap_for(pollutant))

    active = [s for s in area_sectors if s != "TOTAL"] or None
    raster = ctx.area_raster(sector_id, pollutant, active)

    # Web Mercator pixel box covering the requested bounds, with a small pad.
    target_w = int(max(700, min(int(width) if width else 1500, 2400)))
    z = _choose_zoom(west, east, target_w)
    x_tl, y_tl = _lonlat_to_global_px(west, north, z)
    x_br, y_br = _lonlat_to_global_px(east, south, z)
    pad = 26
    left = int(math.floor(min(x_tl, x_br))) - pad
    top = int(math.floor(min(y_tl, y_br))) - pad
    right = int(math.ceil(max(x_tl, x_br))) + pad
    bottom = int(math.ceil(max(y_tl, y_br))) + pad
    w = min(_MAX_PX, right - left)
    h = min(_MAX_PX, bottom - top)

    base = Image.new("RGBA", (w, h), _BG)
    if basemap_url:
        _paste_basemap(base, basemap_url, z, left, top, w, h)

    overlay = _emission_overlay(raster, lower, upper, float(threshold), cmap, z, left, top, w, h)
    base.alpha_composite(overlay)

    def to_px(lon: float, lat: float) -> tuple[float, float]:
        gx, gy = _lonlat_to_global_px(lon, lat, z)
        return gx - left, gy - top

    draw = ImageDraw.Draw(base, "RGBA")

    cams = _load_cams_geojson(ctx)
    for feat in cams.get("features") or []:
        geom = feat.get("geometry") or {}
        if geom.get("type") != "Polygon":
            continue
        ring = geom["coordinates"][0]
        lons = [p[0] for p in ring]
        lats = [p[1] for p in ring]
        if max(lons) < west or min(lons) > east or max(lats) < south or min(lats) > north:
            continue
        draw.polygon([to_px(lon, lat) for lon, lat in ring], outline=_CAMS_OUTLINE)

    dom = domain_bbox_geojson(ctx)
    dom_pts = [to_px(lon, lat) for lon, lat in dom["geometry"]["coordinates"][0]]
    draw.line(dom_pts + [dom_pts[0]], fill=_DOMAIN_OUTLINE, width=2)

    if point_sectors:
        gj = points_geojson(ctx, point_sectors, pollutant)
        for feat in gj.get("features") or []:
            lon, lat = feat["geometry"]["coordinates"]
            if lon < west or lon > east or lat < south or lat > north:
                continue
            px, py = to_px(lon, lat)
            rgb = _hex_rgb((feat["properties"].get("accents") or ["#4f7cff"])[0])
            r = 7
            draw.ellipse(
                [px - r, py - r, px + r, py + r],
                fill=rgb + (255,),
                outline=(255, 255, 255, 200),
                width=2,
            )

    # Compose map + colorbar panel below.
    cb_h = 96
    out = Image.new("RGBA", (w, h + cb_h), _PANEL)
    out.paste(base, (0, 0))
    cb_draw = ImageDraw.Draw(out, "RGBA")
    margin = 90
    bar_w = w - 2 * margin
    bar_h = 22
    bar_y = h + 36
    _draw_colorbar(cb_draw, cmap, lower, upper, float(threshold), margin, bar_y, bar_w, bar_h)

    buf = BytesIO()
    out.convert("RGB").save(buf, format="PNG", optimize=True)
    return buf.getvalue()
