"""
GNFR D fugitive area proxies: weighted mixture of CORINE, OSM subsets, population,
VIIRS Nightfire (Gaussian kernel), GEM Coal Mine Tracker disks, and GOGET disks.

Loaded from ``proxy_mixture`` under each group in ``fugitive_groups.yaml``
(``PROXY/config/ceip/profiles/fugitive_groups.yaml``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio import features
from rasterio.enums import MergeAlg
from shapely.geometry import Point, mapping

from PROXY.core.ceip import DEFAULT_GNFR_GROUP_ORDER
from PROXY.core.dataloaders import resolve_path
from PROXY.core.osm_corine_proxy import (
    adapt_corine_classes_for_grid,
    clc_raw_group_score,
    clc_weighted_class_score,
    filter_osm_by_rules,
    mask_no_sector_z_signal_in_cams_cells,
    osm_coverage_fraction,
    z_score,
)

logger = logging.getLogger(__name__)

GOGET_OK_STATUS = frozenset({"operating", "in-development", "discovered"})


def _gcmt_status_weight(raw: Any) -> float:
    """
    GEM Coal Mine Tracker ``Status`` → raster contribution multiplier.
    Operating = 1; Mothballed / Cancelled = 0.3; Proposed / Shelved = 0; unknown = 0.
    """
    s = str(raw or "").strip().lower()
    if s == "operating":
        return 1.0
    if s in ("mothballed", "cancelled"):
        return 0.3
    if s in ("proposed", "shelved"):
        return 0.0
    return 0.0

_VIIRS_RHI_MAX_VALID = 1.0e6


def _iso3_to_cntr_code(cfg: dict[str, Any], iso3: str) -> str | None:
    iso3 = str(iso3 or "").strip().upper()
    m = (cfg.get("cntr_code_to_iso3") or {}) if isinstance(cfg, dict) else {}
    for cntr, iso in m.items():
        if str(iso).strip().upper() == iso3:
            return str(cntr).strip().upper()
    if iso3 == "GRC":
        return "EL"
    return None


def _country_name_match(series: pd.Series, iso3: str) -> pd.Series:
    """Case-insensitive match of spreadsheet country names to ISO-3 (Greece / Elláda, etc.)."""
    iso3 = str(iso3 or "").strip().upper()
    aliases: set[str] = set()
    if iso3 == "GRC":
        aliases.update(
            {
                "greece",
                "hellas",
                "ellada",
                "elláda",
                "ελλάδα",
            }
        )
    try:
        import pycountry  # type: ignore

        c = pycountry.countries.get(alpha_3=iso3)
        if c is not None:
            aliases.add(c.name.lower())
            if hasattr(c, "common_name") and c.common_name:
                aliases.add(str(c.common_name).lower())
    except Exception:
        pass
    s = series.astype(str).str.strip().str.lower()
    out = pd.Series(False, index=series.index, dtype=bool)
    for a in aliases:
        if a:
            out = out | s.str.contains(a, case=False, na=False, regex=False)
    if iso3 == "GRC":
        out = out | s.isin(["el", "gr", "grc", "el "])
    return out


def _load_nuts_country_polygon(nuts_path: Path, cntr_code: str) -> Any:
    import geopandas as gpd
    from shapely.ops import unary_union

    nuts = gpd.read_file(nuts_path)
    sub = nuts[(nuts["LEVL_CODE"] == 0) & (nuts["CNTR_CODE"] == str(cntr_code).strip().upper())]
    if sub.empty:
        raise ValueError(f"No NUTS level-0 row for CNTR_CODE={cntr_code!r} in {nuts_path}")
    geom = unary_union(sub.geometry.values)
    return gpd.GeoSeries([geom], crs=sub.crs)


def _viirs_dedupe_max_rhi(df: pd.DataFrame, lat_col: str, lon_col: str, rhi_col: str) -> pd.DataFrame:
    df = df.copy()
    df["_lat_r"] = df[lat_col].round(5)
    df["_lon_r"] = df[lon_col].round(5)
    df = df.sort_values(rhi_col, ascending=False).drop_duplicates(subset=["_lat_r", "_lon_r"], keep="first")
    return df.drop(columns=["_lat_r", "_lon_r"])


def _read_gcmt_closed_columns(df: pd.DataFrame) -> tuple[str, str, str | None]:
    cols = {str(c).replace("\n", " ").strip(): c for c in df.columns}
    lat_c = next((cols[k] for k in cols if k.lower() == "latitude"), None)
    lon_c = next((cols[k] for k in cols if k.lower() == "longitude"), None)
    size_c = None
    for k in cols:
        kl = k.lower().replace(" ", "")
        if "minesize" in kl or ("mine" in kl and "km2" in kl):
            size_c = cols[k]
            break
    if lat_c is None or lon_c is None:
        raise ValueError(f"GCMT sheet missing Latitude/Longitude. Columns: {list(df.columns)[:15]}…")
    return str(lat_c), str(lon_c), str(size_c) if size_c else None


def _ref_window_polygon_wgs84(ref: dict[str, Any]) -> gpd.GeoDataFrame:
    from rasterio.transform import array_bounds
    from rasterio.warp import transform_bounds

    h, w = int(ref["height"]), int(ref["width"])
    tr = ref["transform"]
    l, b, r, t = array_bounds(h, w, tr)
    crs_s = str(ref["crs"])
    W, S, E, N = transform_bounds(crs_s, "EPSG:4326", l, b, r, t)
    from shapely.geometry import box

    return gpd.GeoDataFrame(geometry=[box(W, S, E, N)], crs="EPSG:4326")


def load_viirs_flares_wgs84(
    csv_path: Path,
    country_poly_wgs84: gpd.GeoDataFrame,
    *,
    country_iso3: str | None = None,
) -> gpd.GeoDataFrame:
    label = f" (ISO3={country_iso3})" if country_iso3 else ""
    usecols = ["Lat_GMTCO", "Lon_GMTCO", "RHI"]
    df = pd.read_csv(csv_path, usecols=lambda c: c in set(usecols), low_memory=False)
    for c in usecols:
        if c not in df.columns:
            raise ValueError(f"VIIRS CSV missing column {c!r}: {csv_path}")
    n_csv = int(len(df))
    df = df.replace([_VIIRS_RHI_MAX_VALID, 999999], np.nan)
    df = df[np.isfinite(df["Lat_GMTCO"]) & np.isfinite(df["Lon_GMTCO"])]
    n_xy = int(len(df))
    df = df[(df["RHI"] < 9.0e5) & (df["RHI"] > 0) & np.isfinite(df["RHI"])]
    n_rhi = int(len(df))
    if df.empty:
        logger.info(
            "[fugitive datasets] VIIRS Nightfire%s: CSV rows=%d; finite lat/lon=%d; valid RHI=%d; "
            "inside country=0 (empty before spatial clip)",
            label,
            n_csv,
            n_xy,
            n_rhi,
        )
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    g = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["Lon_GMTCO"], df["Lat_GMTCO"], crs="EPSG:4326"),
        crs="EPSG:4326",
    )
    g = g.to_crs(country_poly_wgs84.crs)
    poly = country_poly_wgs84.union_all() if hasattr(country_poly_wgs84, "union_all") else country_poly_wgs84.unary_union
    g = g[g.intersects(poly)]
    n_clip = int(len(g))
    if g.empty:
        logger.info(
            "[fugitive datasets] VIIRS Nightfire%s: CSV rows=%d; finite lat/lon=%d; valid RHI=%d; "
            "inside country=%d; after dedupe=0",
            label,
            n_csv,
            n_xy,
            n_rhi,
            n_clip,
        )
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    g = _viirs_dedupe_max_rhi(g, "Lat_GMTCO", "Lon_GMTCO", "RHI")
    n_fin = int(len(g))
    logger.info(
        "[fugitive datasets] VIIRS Nightfire%s: CSV rows=%d; finite lat/lon=%d; valid RHI=%d; "
        "inside country=%d; after dedupe=%d (Gaussian sigma uses proxy.viirs_sigma_m)",
        label,
        n_csv,
        n_xy,
        n_rhi,
        n_clip,
        n_fin,
    )
    return g


def accumulate_viirs_gaussian_raster(
    flares_3035: gpd.GeoDataFrame,
    value_col: str,
    ref: dict[str, Any],
    *,
    sigma_m: float,
) -> np.ndarray:
    """Sum_i v_i * exp(-d_i(x)^2 / (2 sigma^2)) on pixel centers (EPSG:3035)."""
    from rasterio.transform import rowcol

    h, w = int(ref["height"]), int(ref["width"])
    tr = ref["transform"]
    out = np.zeros((h, w), dtype=np.float64)
    if flares_3035.empty:
        return out.astype(np.float32)
    sig2 = float(sigma_m) ** 2 * 2.0
    if sig2 <= 0:
        return out.astype(np.float32)
    g = flares_3035.to_crs(ref["crs"])
    pixel_w = abs(float(tr[0]))
    pixel_h = abs(float(tr[4]))
    px_m = max(pixel_w, pixel_h, 1e-6)
    rad_px = int(np.ceil(4.0 * float(sigma_m) / px_m)) + 3

    for _, row in g.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        v = float(row.get(value_col, 0.0) or 0.0)
        if not np.isfinite(v) or v <= 0:
            continue
        cx, cy = float(geom.x), float(geom.y)
        rc = rowcol(tr, cx, cy)
        ir, ic = int(rc[0]), int(rc[1])

        r0 = max(0, ir - rad_px)
        r1 = min(h, ir + rad_px + 1)
        c0 = max(0, ic - rad_px)
        c1 = min(w, ic + rad_px + 1)
        if r0 >= r1 or c0 >= c1:
            continue
        rows_s = np.arange(r0, r1, dtype=np.float64) + 0.5
        cols_s = np.arange(c0, c1, dtype=np.float64) + 0.5
        yy, xx = np.meshgrid(rows_s, cols_s, indexing="ij")
        xs = tr.c + xx * tr.a + yy * tr.b
        ys = tr.f + xx * tr.d + yy * tr.e
        d2 = (xs - cx) ** 2 + (ys - cy) ** 2
        out[r0:r1, c0:c1] += v * np.exp(-d2 / sig2)

    return out.astype(np.float32)


def _gcmt_radius_m(row: pd.Series, size_col: str | None, *, closed: bool) -> float:
    if closed:
        return 500.0
    if not size_col or size_col not in row.index:
        return 500.0
    s = row[size_col]
    try:
        km2 = float(s)
    except (TypeError, ValueError):
        return 500.0
    if not np.isfinite(km2) or km2 <= 0:
        return 500.0
    area_m2 = km2 * 1_000_000.0
    return float(np.sqrt(area_m2 / np.pi))


def build_gcmt_weighted_raster(
    xlsx_path: Path,
    iso3: str,
    ref: dict[str, Any],
) -> np.ndarray:
    crs_tgt = ref["crs"]
    h, w = int(ref["height"]), int(ref["width"])
    tr = ref["transform"]
    out = np.zeros((h, w), dtype=np.float32)

    non = pd.read_excel(xlsx_path, sheet_name="GCMT Non-closed Mines")
    closed = pd.read_excel(xlsx_path, sheet_name="GCMT Closed Mines")
    non.columns = [str(c).replace("\n", " ").strip() for c in non.columns]
    closed.columns = [str(c).replace("\n", " ").strip() for c in closed.columns]

    lat_n, lon_n, size_n = _read_gcmt_closed_columns(non)
    lat_c, lon_c, _ = _read_gcmt_closed_columns(closed)
    if "Mine Size (Km2)" in non.columns:
        size_n = "Mine Size (Km2)"

    parts: list[gpd.GeoDataFrame] = []
    n_nc = 0
    n_cl = 0
    for (df, is_closed), lat_col, lon_col in (
        ((non, False), lat_n, lon_n),
        ((closed, True), lat_c, lon_c),
    ):
        if df.empty:
            continue
        m = _country_name_match(df["Country / Area"], iso3)
        df = df.loc[m].copy()
        if df.empty:
            continue
        if "Status" in df.columns:
            df = df.assign(_gcmt_w=df["Status"].map(_gcmt_status_weight))
            df = df.loc[df["_gcmt_w"] > 0].copy()
        else:
            df = df.assign(_gcmt_w=1.0)
        if df.empty:
            continue
        if is_closed:
            n_cl += len(df)
        else:
            n_nc += len(df)
        g = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df[lon_col].astype(float), df[lat_col].astype(float), crs="EPSG:4326"),
            crs="EPSG:4326",
        )
        g = g.to_crs(crs_tgt)
        radii = [_gcmt_radius_m(g.iloc[i], size_n or "", closed=is_closed) for i in range(len(g))]
        g["geometry"] = [g.geometry.iloc[i].buffer(radii[i]) for i in range(len(g))]
        g["w"] = df["_gcmt_w"].astype(np.float64).to_numpy()
        parts.append(g[["geometry", "w"]])

    if not parts:
        logger.info(
            "[fugitive datasets] GEM Coal Mine Tracker (ISO3=%s): non-closed mines=%d; closed mines=%d; "
            "total disks rasterized=0",
            iso3,
            n_nc,
            n_cl,
        )
        return out

    allg = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=crs_tgt)
    shapes = [
        (mapping(geom), float(w))
        for geom, w in zip(allg.geometry, allg["w"])
        if geom is not None and not geom.is_empty and float(w) > 0.0
    ]
    if not shapes:
        logger.info(
            "[fugitive datasets] GEM Coal Mine Tracker (ISO3=%s): non-closed mines=%d; closed mines=%d; "
            "total disks rasterized=0",
            iso3,
            n_nc,
            n_cl,
        )
        return out
    acc = features.rasterize(
        shapes,
        out_shape=(h, w),
        transform=tr,
        fill=0.0,
        dtype=np.float32,
        merge_alg=MergeAlg.add,
    )
    logger.info(
        "[fugitive datasets] GEM Coal Mine Tracker (ISO3=%s): non-closed mines=%d; closed mines=%d; "
        "total disks rasterized=%d (status weights applied)",
        iso3,
        n_nc,
        n_cl,
        len(shapes),
    )
    return np.maximum(acc.astype(np.float32), 0.0)


def _goget_fuel_weight_and_buffers(
    fuel: str,
    onshore: str,
) -> tuple[float, float, float]:
    """Returns (w_g2, w_g4, buffer_m)."""
    f = str(fuel or "").strip().lower()
    on = str(onshore or "").strip().lower()
    buf_on = 1500.0
    buf_off = 5000.0
    buf = buf_on if on in ("onshore", "unknown", "") else buf_off if on == "offshore" else buf_on

    if f == "oil":
        return 1.0, 0.0, buf
    if f in ("gas", "gas and condensate"):
        return 0.0, 1.0, buf
    if f == "oil and gas":
        return 0.5, 0.5, buf
    return 0.0, 0.0, buf


def build_goget_weighted_rasters(
    xlsx_path: Path,
    iso3: str,
    ref: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Heavy-tailed production weights via log1p(``Quantity (original)``); separate G2 / G4 contributions."""
    h, w = int(ref["height"]), int(ref["width"])
    tr = ref["transform"]
    crs_tgt = ref["crs"]
    out2 = np.zeros((h, w), dtype=np.float64)
    out4 = np.zeros((h, w), dtype=np.float64)

    main = pd.read_excel(xlsx_path, sheet_name="Field-level main data")
    prod = pd.read_excel(xlsx_path, sheet_name="Field-level production data")
    main.columns = [str(c).strip() for c in main.columns]
    prod.columns = [str(c).strip() for c in prod.columns]

    if "Country/Area" not in main.columns:
        raise ValueError("GOGET main sheet missing Country/Area")
    main = main.loc[_country_name_match(main["Country/Area"], iso3)].copy()
    n_country = int(len(main))
    if main.empty:
        logger.info(
            "[fugitive datasets] GOGET oil/gas fields (ISO3=%s): rows matching country in main sheet=0",
            iso3,
        )
        return out2.astype(np.float32), out4.astype(np.float32)

    main["Status_lt"] = main["Status"].astype(str).str.strip().str.lower()
    main = main[main["Status_lt"].isin(GOGET_OK_STATUS)]
    n_after_ceip_status = int(len(main))

    qty_col = "Quantity (original)"
    if qty_col not in prod.columns:
        raise ValueError(f"GOGET production sheet missing {qty_col!r}")
    if "Unit ID" not in prod.columns:
        raise ValueError("GOGET production sheet missing Unit ID")
    year_cols = [c for c in prod.columns if "year" in str(c).lower()]
    if year_cols:
        prod = prod.sort_values(year_cols[0], na_position="last")
    prod_last = prod.groupby("Unit ID", as_index=False).last()

    merged = main.merge(prod_last[["Unit ID", qty_col]], on="Unit ID", how="left")
    merged[qty_col] = pd.to_numeric(merged[qty_col], errors="coerce").fillna(0.0)
    merged["logq"] = np.log1p(np.maximum(merged[qty_col].astype(np.float64), 0.0))

    crsx = ref["crs"]
    n_stamp_g2 = 0
    n_stamp_g4 = 0
    for _, row in merged.iterrows():
        w2, w4, buf_m = _goget_fuel_weight_and_buffers(
            str(row.get("Fuel type", "")),
            str(row.get("Onshore/Offshore", "")),
        )
        if w2 <= 0 and w4 <= 0:
            continue
        lat, lon = row.get("Latitude"), row.get("Longitude")
        try:
            lat_f = float(lat)
            lon_f = float(lon)
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(lat_f) and np.isfinite(lon_f)):
            continue
        if w2 > 0:
            n_stamp_g2 += 1
        if w4 > 0:
            n_stamp_g4 += 1
        pt = gpd.GeoDataFrame(geometry=[Point(lon_f, lat_f)], crs="EPSG:4326").to_crs(crsx)
        geom = pt.geometry.iloc[0].buffer(float(buf_m))
        base = float(row["logq"])
        if w2 > 0:
            shapes2 = [(mapping(geom), base * w2)]
            acc2 = features.rasterize(
                shapes2,
                out_shape=(h, w),
                transform=tr,
                fill=0.0,
                dtype=np.float64,
                merge_alg=MergeAlg.add,
            )
            out2 += acc2
        if w4 > 0:
            shapes4 = [(mapping(geom), base * w4)]
            acc4 = features.rasterize(
                shapes4,
                out_shape=(h, w),
                transform=tr,
                fill=0.0,
                dtype=np.float64,
                merge_alg=MergeAlg.add,
            )
            out4 += acc4

    logger.info(
        "[fugitive datasets] GOGET oil/gas fields (ISO3=%s): country rows=%d; after allowed status "
        "(%s)=%d; merged with production=%d; units stamped to G2 raster=%d; to G4 raster=%d "
        "(buffers onshore/offshore; weight=log1p Quantity original))",
        iso3,
        n_country,
        ", ".join(sorted(GOGET_OK_STATUS)),
        n_after_ceip_status,
        len(merged),
        n_stamp_g2,
        n_stamp_g4,
    )

    return out2.astype(np.float32), out4.astype(np.float32)


def _corine_score_for_mixture(clc: np.ndarray, entry: dict[str, Any]) -> np.ndarray:
    if "corine_class_weights" in entry:
        cw = {int(k): float(v) for k, v in dict(entry["corine_class_weights"]).items()}
        return clc_weighted_class_score(clc, cw)
    yaml_clc = [int(x) for x in (entry.get("classes") or [])]
    clc_use, _ = adapt_corine_classes_for_grid(clc, yaml_clc)
    return clc_raw_group_score(clc, clc_use)


def _raw_score_for_mixture_entry(
    entry: dict[str, Any],
    *,
    clc: np.ndarray,
    osm_gdf: gpd.GeoDataFrame,
    ref: dict[str, Any],
    pcfg: dict[str, Any],
    p_pop: np.ndarray,
    auxiliary: dict[str, np.ndarray | None],
    subf: int,
    tile_px: int,
) -> np.ndarray:
    """Raw (pre–z-score) score for one ``proxy_mixture`` entry."""
    kind = str(entry.get("kind", "")).strip().lower()
    if kind == "corine":
        return _corine_score_for_mixture(clc, entry)
    if kind == "osm":
        rules = entry.get("osm_rules") or {}
        gsub = filter_osm_by_rules(osm_gdf, rules)
        return osm_coverage_fraction(gsub, ref, subdivide_factor=subf, tile_pixels=tile_px)
    if kind == "population":
        return np.asarray(p_pop, dtype=np.float32)
    if kind in ("viirs_gaussian", "viirs"):
        viirs_r = auxiliary.get("viirs")
        return np.asarray(viirs_r, dtype=np.float32) if viirs_r is not None else np.zeros_like(clc, dtype=np.float32)
    if kind == "gcmt_coal":
        gcmt_r = auxiliary.get("gcmt_coal")
        return np.asarray(gcmt_r, dtype=np.float32) if gcmt_r is not None else np.zeros_like(clc, dtype=np.float32)
    if kind == "goget_g2":
        go2_r = auxiliary.get("goget_g2")
        return np.asarray(go2_r, dtype=np.float32) if go2_r is not None else np.zeros_like(clc, dtype=np.float32)
    if kind == "goget_g4":
        go4_r = auxiliary.get("goget_g4")
        return np.asarray(go4_r, dtype=np.float32) if go4_r is not None else np.zeros_like(clc, dtype=np.float32)
    raise ValueError(f"Unknown fugitive proxy_mixture kind {kind!r}")


def mixture_kinds_for_groups(
    group_specs: dict[str, Any],
    group_order: tuple[str, ...] | None,
) -> set[str]:
    """Union of all ``proxy_mixture`` ``kind`` values in order."""
    from PROXY.core.ceip import DEFAULT_GNFR_GROUP_ORDER

    groups = {str(k): v for k, v in dict(group_specs.get("groups") or {}).items()}
    order = group_order if group_order is not None else tuple(groups.keys()) or DEFAULT_GNFR_GROUP_ORDER
    needed: set[str] = set()
    for gid in order:
        spec = groups.get(str(gid)) or {}
        for entry in spec.get("proxy_mixture") or []:
            needed.add(str(entry.get("kind", "")).strip().lower())
    return needed


def prepare_fugitive_auxiliary_rasters(
    *,
    needed: set[str],
    ref: dict[str, Any],
    pcfg: dict[str, Any],
    cfg: dict[str, Any],
    root: Path,
    silent: bool = False,
) -> dict[str, np.ndarray]:
    """
    Load VIIRS / GCMT / GOGET rasters on ``ref`` for mixture kinds in ``needed``.

    Returns a dict with any of keys ``viirs``, ``gcmt_coal``, ``goget_g2``, ``goget_g4``;
    values are float32 HxW arrays (zeros if a path is missing or kind not needed).
    """
    h, w = int(ref["height"]), int(ref["width"])
    z = np.zeros((h, w), dtype=np.float32)

    def _info(msg: str, *args: object) -> None:
        if not silent:
            logger.info(msg, *args)

    def _warn(msg: str, *args: object) -> None:
        if not silent:
            logger.warning(msg, *args)

    out: dict[str, np.ndarray] = {}

    paths = cfg.get("paths") or {}
    fb_iso = str((cfg.get("defaults") or {}).get("fallback_country_iso3", "GRC")).strip().upper()
    cntr = _iso3_to_cntr_code(cfg, fb_iso)
    nuts_path = resolve_path(root, paths["nuts_gpkg"])

    viirs_p = paths.get("viirs_nightfire_csv")
    gcmt_p = paths.get("gcmt_xlsx")
    goget_p = paths.get("goget_xlsx")

    sigma_m = float(pcfg.get("viirs_sigma_m", 750.0))

    aux_needed = {"viirs_gaussian", "viirs", "gcmt_coal", "goget_g2", "goget_g4"} & needed
    if aux_needed:
        _info(
            "[fugitive datasets] Auxiliary rasters enabled (mixture kinds): %s | CEIP country_iso3=%s | "
            "NUTS CNTR_CODE=%s",
            sorted(aux_needed),
            fb_iso,
            cntr if cntr else "None (VIIRS uses reference-window bbox)",
        )

    if "viirs_gaussian" in needed or "viirs" in needed:
        if cntr:
            poly3035 = _load_nuts_country_polygon(nuts_path, cntr)
            country_poly_wgs84 = poly3035.to_crs("EPSG:4326")
        else:
            _warn(
                "[fugitive datasets] Could not map ISO3 %s to NUTS CNTR_CODE; clipping VIIRS points to "
                "reference window instead.",
                fb_iso,
            )
            country_poly_wgs84 = _ref_window_polygon_wgs84(ref)
        if viirs_p and Path(viirs_p).is_file():
            flw = load_viirs_flares_wgs84(Path(viirs_p), country_poly_wgs84, country_iso3=fb_iso)
            fl3035 = flw.to_crs(ref["crs"]) if not flw.empty else flw
            viirs_r = accumulate_viirs_gaussian_raster(fl3035, "RHI", ref, sigma_m=sigma_m)
            _info("[fugitive datasets] VIIRS Nightfire: applied Gaussian sigma_m=%.1f m", sigma_m)
            out["viirs"] = viirs_r
        else:
            _warn(
                "[fugitive datasets] VIIRS CSV path missing or not a file (%s); using zeros for viirs_gaussian.",
                viirs_p,
            )
            out["viirs"] = np.zeros((h, w), dtype=np.float32)

    if "gcmt_coal" in needed:
        if gcmt_p and Path(gcmt_p).is_file():
            out["gcmt_coal"] = build_gcmt_weighted_raster(Path(gcmt_p), fb_iso, ref)
        else:
            _warn(
                "[fugitive datasets] GCMT xlsx missing or not a file (%s); using zeros for gcmt_coal.",
                gcmt_p,
            )
            out["gcmt_coal"] = np.zeros((h, w), dtype=np.float32)

    if "goget_g2" in needed or "goget_g4" in needed:
        if goget_p and Path(goget_p).is_file():
            go2_r, go4_r = build_goget_weighted_rasters(Path(goget_p), fb_iso, ref)
            out["goget_g2"] = go2_r
            out["goget_g4"] = go4_r
        else:
            _warn(
                "[fugitive datasets] GOGET xlsx missing or not a file (%s); using zeros for goget_g2/goget_g4.",
                goget_p,
            )
            out["goget_g2"] = z.copy()
            out["goget_g4"] = z.copy()

    return out


def build_fugitive_group_pg(
    clc: np.ndarray,
    osm_gdf: gpd.GeoDataFrame,
    group_specs: dict[str, Any],
    ref: dict[str, Any],
    pcfg: dict[str, Any],
    p_pop: np.ndarray,
    *,
    group_order: tuple[str, ...] | None,
    root: Path,
    cfg: dict[str, Any],
    auxiliary_cache: dict[str, np.ndarray] | None = None,
    cam_cell_id: np.ndarray | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Per-group ``proxy_mixture`` from YAML: z-score each indicator, then weighted sum.

    ``auxiliary_cache`` optional dict from :func:`prepare_fugitive_auxiliary_rasters` (same ``ref`` as this
    call) so callers can rasterize VIIRS/GCMT/GOGET once for visualization without reloading files.

    When ``cam_cell_id`` is set (production GNFR pipeline), pixels in CAMS cells whose peak blended
    sector z-score is ~0 may use ``p_pop`` if ``proxy.cam_cell_zero_z_pop_fallback`` is true (default).
    Set it to false to keep ``acc_score`` (zeros) in those cells so normalization decides redistribution
    instead of the population proxy. Pixel-level fallback ``acc_score < 1e-10`` for mixture rows that
    include ``population`` is unchanged.
    """
    from PROXY.core.ceip import DEFAULT_GNFR_GROUP_ORDER

    groups = {str(k): v for k, v in dict(group_specs.get("groups") or {}).items()}
    order = group_order if group_order is not None else tuple(groups.keys()) or DEFAULT_GNFR_GROUP_ORDER

    if auxiliary_cache is not None:
        viirs_r = auxiliary_cache.get("viirs")
        gcmt_r = auxiliary_cache.get("gcmt_coal")
        go2_r = auxiliary_cache.get("goget_g2")
        go4_r = auxiliary_cache.get("goget_g4")
    else:
        needed = mixture_kinds_for_groups(group_specs, group_order)
        aux = prepare_fugitive_auxiliary_rasters(
            needed=needed,
            ref=ref,
            pcfg=pcfg,
            cfg=cfg,
            root=root,
            silent=False,
        )
        viirs_r = aux.get("viirs")
        gcmt_r = aux.get("gcmt_coal")
        go2_r = aux.get("goget_g2")
        go4_r = aux.get("goget_g4")

    subf = int(pcfg.get("osm_subdivide_factor", 4))
    tile_px = int(pcfg.get("osm_tile_pixels", 256))

    auxiliary_by_kind: dict[str, np.ndarray | None] = {
        "viirs": viirs_r,
        "gcmt_coal": gcmt_r,
        "goget_g2": go2_r,
        "goget_g4": go4_r,
    }

    out: dict[str, dict[str, np.ndarray]] = {}
    for gid in order:
        spec = groups.get(str(gid))
        if spec is None:
            raise KeyError(f"fugitive proxy: group {gid!r} missing")
        mix = spec.get("proxy_mixture")
        if not mix:
            raise ValueError(
                f"Group {gid!r} has no proxy_mixture — update fugitive_groups.yaml for the new fugitive model."
            )

        acc_score = np.zeros(clc.shape, dtype=np.float32)
        wsum = 0.0
        osm_raw_dbg = np.zeros(clc.shape, dtype=np.float32)
        clc_raw_dbg = np.zeros(clc.shape, dtype=np.float32)

        for entry in mix:
            kind = str(entry.get("kind", "")).strip().lower()
            wt = float(entry.get("weight", 0.0))
            if wt <= 0:
                continue
            wsum += wt

            raw = _raw_score_for_mixture_entry(
                entry,
                clc=clc,
                osm_gdf=osm_gdf,
                ref=ref,
                pcfg=pcfg,
                p_pop=p_pop,
                auxiliary=auxiliary_by_kind,
                subf=subf,
                tile_px=tile_px,
            )

            if kind == "corine":
                clc_raw_dbg = np.maximum(clc_raw_dbg, raw)
            elif kind == "osm":
                osm_raw_dbg = np.maximum(osm_raw_dbg, raw)

            zr = z_score(raw)
            acc_score += np.float32(wt) * zr

        if wsum > 0 and abs(wsum - 1.0) > 1e-3:
            logger.warning("Group %s proxy_mixture weights sum to %s (expected ~1).", gid, wsum)

        mix_has_population = any(
            str(e.get("kind", "")).strip().lower() == "population"
            for e in mix
        )
        cell_no_z = (
            mask_no_sector_z_signal_in_cams_cells(
                acc_score,
                cam_cell_id,
                max_abs_floor=float(pcfg.get("cell_no_sector_z_floor", 1e-10)),
            )
            if cam_cell_id is not None
            else np.zeros(clc.shape, dtype=bool)
        )
        # When false: CAMS cells with no blended sector z-score do **not** switch to p_pop;
        # downstream CAMS-cell normalization spreads mass uniformly (or via other rules).
        use_pop_for_cell_no_z = bool(pcfg.get("cam_cell_zero_z_pop_fallback", True))
        cell_no_z_fb = cell_no_z if use_pop_for_cell_no_z else np.zeros(clc.shape, dtype=bool)
        if mix_has_population:
            use_pop = (acc_score < 1e-10) | cell_no_z_fb
        else:
            use_pop = cell_no_z_fb
        p_g = np.where(use_pop, p_pop, acc_score).astype(np.float32)
        used_fb = use_pop.astype(np.uint8)

        out[str(gid)] = {
            "osm_raw": osm_raw_dbg.astype(np.float32),
            "clc_raw": clc_raw_dbg.astype(np.float32),
            "p_sector": acc_score.astype(np.float32),
            "p_g": p_g,
            "used_pop_fallback": used_fb,
        }

    return out


def build_fugitive_mixture_map_layers(
    *,
    groups_raw: dict[str, Any],
    group_order: tuple[str, ...],
    clc_nn: np.ndarray,
    osm_gdf: gpd.GeoDataFrame,
    ref: dict[str, Any],
    pcfg: dict[str, Any],
    p_pop: np.ndarray,
    auxiliary_cache: dict[str, np.ndarray],
    layer_labels: dict[str, tuple[str, ...]],
) -> list[tuple[str, np.ndarray]]:
    """One raw score raster per ``proxy_mixture`` row for Folium (labels must align with YAML order)."""
    groups = {str(k): v for k, v in dict(groups_raw).items()}
    subf = int(pcfg.get("osm_subdivide_factor", 4))
    tile_px = int(pcfg.get("osm_tile_pixels", 256))
    auxiliary_by_kind: dict[str, np.ndarray | None] = {
        "viirs": auxiliary_cache.get("viirs"),
        "gcmt_coal": auxiliary_cache.get("gcmt_coal"),
        "goget_g2": auxiliary_cache.get("goget_g2"),
        "goget_g4": auxiliary_cache.get("goget_g4"),
    }
    out: list[tuple[str, np.ndarray]] = []
    for gid in group_order:
        spec = groups.get(str(gid))
        if spec is None:
            continue
        mix = spec.get("proxy_mixture") or []
        labels = layer_labels.get(str(gid))
        if not labels:
            continue
        if len(labels) != len(mix):
            logger.warning(
                "fugitive map viz: group %s — %d mixture rows vs %d labels (expected equal); skipping group.",
                gid,
                len(mix),
                len(labels),
            )
            continue
        for entry, lab in zip(mix, labels):
            raw = _raw_score_for_mixture_entry(
                entry,
                clc=clc_nn,
                osm_gdf=osm_gdf,
                ref=ref,
                pcfg=pcfg,
                p_pop=p_pop,
                auxiliary=auxiliary_by_kind,
                subf=subf,
                tile_px=tile_px,
            )
            out.append((str(lab), np.asarray(raw, dtype=np.float32)))
    return out
