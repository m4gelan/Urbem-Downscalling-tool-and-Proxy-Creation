#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from PROXY.core.alpha import AlphaRequest, compute_alpha
from PROXY.core.dataloaders import load_path_config, load_yaml
from PROXY.core.raster.country_clip import resolve_cams_country_iso3
from PROXY.core.dataloaders.discovery import discover_cams_emissions, discover_corine
from PROXY.core.matching import (
    MatchRequest,
    _resolve_max_match_distance_km,
    load_facility_candidates_for_sector,
    run_matching,
)
from PROXY.visualization.point_link_sectors import POINT_LINK_SECTOR_KEYS


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_gtiff_srs_source_epsg() -> None:
    """Prefer EPSG registry CRS over GeoTIFF geokeys (reduces GDAL EPSG:3035 mismatch warnings)."""
    os.environ["GTIFF_SRS_SOURCE"] = "EPSG"


def _configure_logging(level_name: str) -> None:
    """Install a root logging handler so every sector's ``logger.info(...)`` is visible.

    The J_Waste pipeline calls ``logging.basicConfig`` itself when invoked, which is why
    its messages show up under the legacy single-sector run; other sectors rely on the root
    logger already being configured. This helper installs the same format used by J_Waste so
    downstream output stays consistent whether or not J_Waste is in the selected set. Called
    once per CLI invocation; if handlers already exist we only promote the level.
    """
    try:
        level = getattr(logging, level_name.upper())
    except AttributeError:
        level = logging.INFO
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root.addHandler(handler)
    root.setLevel(level)


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    m, s = divmod(int(round(seconds)), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def _print_banner(msg: str) -> None:
    """Print a bold-ish banner that survives interleaved sector logging."""
    bar = "=" * max(60, len(msg) + 6)
    print(f"\n{bar}\n= {msg}\n{bar}", flush=True)


def _build_cmd(args: argparse.Namespace) -> int:
    _configure_logging(getattr(args, "log_level", "INFO"))
    _ensure_gtiff_srs_source_epsg()

    cfg = load_path_config(Path(args.config))
    root = Path(__file__).resolve().parents[1]
    cfg.resolved["proxy_common"]["corine_tif"] = discover_corine(
        root, Path(cfg.require("proxy_common", "corine_tif"))
    )
    cfg.resolved["emissions"]["cams_2019_nc"] = discover_cams_emissions(
        root, Path(cfg.require("emissions", "cams_2019_nc"))
    )
    sectors_file = root / "PROXY" / "config" / "sectors.yaml"
    sectors_data = load_yaml(sectors_file)
    entries = sectors_data.get("sectors", []) or []
    target = args.sector.strip() if args.sector else None
    selected = [
        e for e in entries if bool(e.get("enabled", True)) and (target is None or e.get("key") == target)
    ]
    if not selected:
        raise SystemExit(f"No enabled sectors matched target={target!r}")

    total = len(selected)
    keys_listed = ", ".join(str(e["key"]) for e in selected)
    _print_banner(
        f"[build] country={args.country} sectors={total} ({keys_listed}) "
        f"started={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    built = 0
    skipped = 0
    timings: list[tuple[str, str, float, str]] = []  # (key, status, seconds, detail)
    t_overall = time.perf_counter()

    for idx, entry in enumerate(selected, start=1):
        key = str(entry["key"])
        header = f"[build {idx}/{total}] {key}"
        _print_banner(f"{header} starting (country={args.country})")
        t_start = time.perf_counter()

        try:
            mod = importlib.import_module(f"PROXY.sectors.{key}.builder")
        except ModuleNotFoundError as exc:
            dt = time.perf_counter() - t_start
            print(f"{header} skip: builder module not found ({exc})", flush=True)
            timings.append((key, "skip", dt, "builder not found"))
            skipped += 1
            continue

        sector_cfg_path = root / str(entry["config"])
        sector_cfg = load_yaml(sector_cfg_path)
        sector_cfg["build_part"] = str(getattr(args, "part", "both")).strip().lower()
        out_dir = root / str(sector_cfg["output_dir"])
        out_name = str(sector_cfg.get("output_filename", f"{key}_areasource.tif"))
        sector_cfg["output_path"] = out_dir / out_name
        sector_cfg["output_dir"] = out_dir
        print(f"{header} config={sector_cfg_path.relative_to(root)} output={out_dir / out_name}", flush=True)

        try:
            result = mod.build(path_cfg=cfg.resolved, sector_cfg=sector_cfg, country=args.country)
            dt = time.perf_counter() - t_start
            built += 1
            out_path = result.get("output_path", out_dir / out_name)
            print(f"{header} OK -> {out_path} (in {_format_duration(dt)})", flush=True)
            timings.append((key, "OK", dt, str(out_path)))
        except Exception as exc:
            dt = time.perf_counter() - t_start
            skipped += 1
            print(f"{header} skip: {exc} (after {_format_duration(dt)})", flush=True)
            if getattr(args, "log_level", "INFO").upper() == "DEBUG":
                traceback.print_exc()
            timings.append((key, "skip", dt, str(exc)))

    total_dt = time.perf_counter() - t_overall
    _print_banner(
        f"[build] completed built={built} skipped={skipped} elapsed={_format_duration(total_dt)}"
    )
    if timings:
        width_key = max(len(k) for k, _, _, _ in timings)
        width_st = max(len(s) for _, s, _, _ in timings)
        print("[build] per-sector timings:", flush=True)
        for key, status, dt, detail in timings:
            print(
                f"  - {key:<{width_key}}  {status:<{width_st}}  {_format_duration(dt):>7}  "
                f"{detail}",
                flush=True,
            )
    return 0


def _alpha_cmd(args: argparse.Namespace) -> int:
    cfg = load_path_config(Path(args.config))
    alpha_workbook = cfg.require("proxy_common", "alpha_workbook")
    root = Path(__file__).resolve().parents[1]
    result = compute_alpha(
        workbook_path=Path(alpha_workbook),
        output_dir=root / "OUTPUT" / "Proxy_weights" / "_alpha",
        request=AlphaRequest(country=args.country, pollutant=args.pollutant),
        repo_root=root,
    )
    print(
        f"[alpha] country={args.country} pollutant={args.pollutant or 'all'} records={result['records']} diagnostics={result['diagnostics']}"
    )
    return 0


def _sector_area_weight_tif_for_link(root: Path, sector_cfg: dict) -> Path | None:
    """Area proxy GeoTIFF used as the reference grid for CAMS↔facility link rasters."""
    if not sector_cfg:
        return None
    od = Path(str(sector_cfg.get("output_dir", "")))
    if not od.is_absolute():
        od = root / od
    name = str(sector_cfg.get("output_filename", "areasource.tif"))
    p = od / name
    return p if p.is_file() else None


def _match_points_cmd(args: argparse.Namespace) -> int:
    cfg = load_path_config(Path(args.config))
    root = Path(__file__).resolve().parents[1]
    cams_nc = discover_cams_emissions(root, Path(cfg.require("emissions", "cams_2019_nc")))
    eprtr_csv = Path(cfg.require("eprtr", "facilities"))
    sectors_data = load_yaml(root / "PROXY" / "config" / "sectors.yaml")
    entries = sectors_data.get("sectors", []) or []
    sector_entry = next((e for e in entries if str(e.get("key")) == args.sector), None)
    sector_cfg = {}
    if sector_entry is not None:
        sector_cfg_path = root / str(sector_entry["config"])
        sector_cfg = load_yaml(sector_cfg_path)
    link_ref: Path | None = None
    if not getattr(args, "no_link_geotiff", False):
        link_ref = _sector_area_weight_tif_for_link(root, sector_cfg)
        if link_ref is None:
            print(
                "[match-points] link GeoTIFF skipped: area weight raster not found "
                f"(build the sector area proxy first; expected under output_dir + output_filename).",
                flush=True,
            )
    fac_df, pool_id, fac_src = load_facility_candidates_for_sector(
        repo_root=root,
        paths_resolved=cfg.resolved,
        sector_cfg=sector_cfg,
        eprtr_csv_path=eprtr_csv,
        year=args.year,
    )
    scoring_yaml = load_yaml(root / "PROXY" / "config" / "eprtr_scoring.yaml")
    max_km_hint = _resolve_max_match_distance_km(args.sector, scoring_yaml, sector_cfg)
    print(
        f"[match-points] facility_pool={pool_id} source={fac_src} max_distance_km={max_km_hint}",
        flush=True,
    )
    result = run_matching(
        request=MatchRequest(
            sector=args.sector,
            year=args.year,
            pollutant=args.pollutant,
            max_points=args.max_points,
            cams_iso3=args.cams_iso3,
        ),
        cams_nc_path=cams_nc,
        eprtr_csv_path=eprtr_csv,
        scoring_cfg_path=root / "PROXY" / "config" / "eprtr_scoring.yaml",
        output_dir=root / "OUTPUT" / "Proxy_weights" / args.sector,
        sector_cfg=sector_cfg,
        link_ref_weights_tif=link_ref,
        facilities_df=fac_df,
        facility_pool_id=pool_id,
        facility_source_path=fac_src,
    )
    print(
        f"[match-points] sector={args.sector} year={args.year} matches={result.get('matches', 0)} coverage={result.get('coverage_rate', 0.0):.3f} fallback={result.get('fallback_matches', 0)}",
        flush=True,
    )
    lg = result.get("link_geotiff")
    if lg:
        print(f"[match-points] link GeoTIFF (2 bands): {lg}", flush=True)
    return 0


def _resolve_under_root(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


_AREA_PREVIEW_SECTORS = frozenset(
    {
        "A_PublicPower",
        "I_Offroad",
        "C_OtherCombustion",
        "D_Fugitive",
        "B_Industry",
        "E_Solvents",
        "J_Waste",
        "G_Shipping",
        "K_Agriculture",
    }
)

_AREA_DEFAULT_WEIGHT_TIFS = {
    "A_PublicPower": "publicpower_areasource.tif",
    "I_Offroad": "offroad_areasource.tif",
    "C_OtherCombustion": "othercombustion_areasource.tif",
    "D_Fugitive": "fugitive_areasource.tif",
    "B_Industry": "industry_areasource.tif",
    "E_Solvents": "solvents_areasource.tif",
    "J_Waste": "waste_areasource.tif",
    "G_Shipping": "shipping_areasource.tif",
    "K_Agriculture": "agriculture_areasource.tif",
    "H_Aviation": "aviation_areasource.tif",
}

# Sectors with CAMS↔facility match CSV + optional 2-band link GeoTIFF + point-context HTML.
_POINT_LINK_SECTORS = POINT_LINK_SECTOR_KEYS


def _point_link_artifact_paths(root: Path, sector_key: str, year: int) -> tuple[Path, Path] | None:
    base = root / "OUTPUT" / "Proxy_weights" / sector_key
    csv_p = base / f"{sector_key}_point_matches_{year}.csv"
    if not csv_p.is_file():
        return None
    tif_candidates = [base / f"{sector_key}_cams_facility_link_{year}.tif"]
    try:
        sectors_doc = load_yaml(root / "PROXY" / "config" / "sectors.yaml")
        entries = sectors_doc.get("sectors", []) or []
        entry = next((e for e in entries if str(e.get("key")) == sector_key), None)
        if entry:
            sc = load_yaml(root / str(entry["config"]))
            pm = sc.get("point_matching") or {}
            alt = pm.get("link_geotiff_filename")
            if alt:
                tif_candidates.insert(0, base / str(alt))
    except Exception:
        pass
    for tif_p in tif_candidates:
        if tif_p.is_file():
            return tif_p, csv_p
    return None


def _parse_bbox_cli(raw: str | None) -> tuple[float, float, float, float] | None:
    """Parse a ``--bbox W,S,E,N`` CLI argument into a tuple (or ``None``).

    Accepts comma- or space-separated lists of four numbers in WGS84 degrees.
    """
    if raw is None:
        return None
    txt = str(raw).strip()
    if not txt:
        return None
    for sep in (",", ";", " "):
        if sep in txt:
            parts = [p for p in (s.strip() for s in txt.split(sep)) if p]
            break
    else:
        parts = [txt]
    if len(parts) != 4:
        raise SystemExit(
            f"[visualize] --bbox expects 'W,S,E,N' (4 numbers), got: {raw!r}"
        )
    try:
        w, s, e, n = (float(p) for p in parts)
    except ValueError as exc:
        raise SystemExit(f"[visualize] --bbox parse error: {exc} (value={raw!r})") from exc
    if e <= w or n <= s:
        raise SystemExit(
            f"[visualize] --bbox must satisfy W<E and S<N, got W={w} S={s} E={e} N={n}"
        )
    return (w, s, e, n)


def _write_area_preview_html(
    *,
    root: Path,
    path_cfg: dict,
    sector_key: str,
    sector_cfg: dict,
    weight_tif: Path,
    out_html: Path,
    corine_tif: Path,
    population_tif: Path,
    cams_nc: Path | None,
    country_code: str,
    pad_deg: float,
    max_width: int,
    max_height: int,
    weight_display: str,
    region: str | None = None,
    override_bbox: tuple[float, float, float, float] | None = None,
) -> Path:
    """Dispatch to the sector-specific Folium writer. ``sector_key`` must be in ``_AREA_PREVIEW_SECTORS``."""
    cams_iso3, _src = resolve_cams_country_iso3(
        cli_country=country_code,
        explicit_iso3=sector_cfg.get("cams_country_iso3"),
    )
    _common: dict = {
        "root": root,
        "weight_tif": weight_tif,
        "corine_tif": corine_tif,
        "population_tif": population_tif,
        "out_html": out_html,
        "pad_deg": pad_deg,
        "max_width": max_width,
        "max_height": max_height,
        "cams_nc_path": cams_nc,
        "cams_country_iso3": cams_iso3,
        "weight_display_mode": weight_display,
        "region": region,
        "override_bbox": override_bbox,
    }
    if sector_key == "A_PublicPower":
        from PROXY.visualization.public_power_area_map import write_public_power_area_html

        return write_public_power_area_html(
            **_common,
            area_proxy=sector_cfg.get("area_proxy") or {},
        )
    if sector_key == "I_Offroad":
        from PROXY.visualization.offroad_area_map import write_offroad_area_html

        _kw = {**_common, "population_tif": None}
        return write_offroad_area_html(
            **_kw,
            area_proxy=sector_cfg.get("area_proxy") or {},
            path_cfg=path_cfg,
        )
    if sector_key == "C_OtherCombustion":
        from PROXY.visualization.other_combustion_area_map import write_other_combustion_area_html

        _kw = {k: v for k, v in _common.items() if k != "population_tif"}
        return write_other_combustion_area_html(
            **_kw,
            sector_cfg=sector_cfg,
            path_cfg=path_cfg,
        )
    if sector_key == "D_Fugitive":
        from PROXY.visualization.fugitive_area_map import write_fugitive_area_html

        return write_fugitive_area_html(
            **_common,
            sector_cfg=sector_cfg,
            path_cfg=path_cfg,
        )
    if sector_key == "B_Industry":
        from PROXY.visualization.industry_area_map import write_industry_area_html

        return write_industry_area_html(
            **_common,
            sector_cfg=sector_cfg,
            path_cfg=path_cfg,
        )
    if sector_key == "E_Solvents":
        from PROXY.visualization.solvents_area_map import write_solvents_area_html

        return write_solvents_area_html(
            **_common,
            sector_cfg=sector_cfg,
            path_cfg=path_cfg,
            country=country_code,
        )
    if sector_key == "J_Waste":
        from PROXY.visualization.waste_area_map import write_waste_area_html

        return write_waste_area_html(
            **_common,
            sector_cfg=sector_cfg,
            path_cfg=path_cfg,
            country=country_code,
        )
    if sector_key == "G_Shipping":
        from PROXY.visualization.shipping_area_map import write_shipping_area_html

        return write_shipping_area_html(
            **_common,
            sector_cfg=sector_cfg,
            path_cfg=path_cfg,
            country=country_code,
        )
    if sector_key == "K_Agriculture":
        from PROXY.visualization.agriculture_area_map import write_k_agriculture_area_html

        return write_k_agriculture_area_html(
            **_common,
            sector_cfg=sector_cfg,
            path_cfg=path_cfg,
            country=country_code,
        )
    raise ValueError(
        f"no area HTML preview for sector={sector_key!r} "
        f"(supported: {sorted(_AREA_PREVIEW_SECTORS)})"
    )


def _visualize_cmd(args: argparse.Namespace) -> int:
    _ensure_gtiff_srs_source_epsg()
    root = Path(__file__).resolve().parents[1]
    cfg = load_path_config(Path(args.config))
    cfg.resolved["proxy_common"]["corine_tif"] = discover_corine(
        root, Path(cfg.require("proxy_common", "corine_tif"))
    )
    cfg.resolved["emissions"]["cams_2019_nc"] = discover_cams_emissions(
        root, Path(cfg.require("emissions", "cams_2019_nc"))
    )

    sectors_file = root / "PROXY" / "config" / "sectors.yaml"
    entries = load_yaml(sectors_file).get("sectors") or []

    raw = getattr(args, "sector", None)
    sector_token = str(raw).strip() if raw is not None else ""
    batch = (not sector_token) or sector_token.lower() == "all"
    point_link = bool(getattr(args, "point_link", False))
    point_link_only = bool(getattr(args, "point_link_only", False))
    link_year = int(getattr(args, "link_year", 2019))
    if point_link_only:
        point_link = True
    if batch and (args.weight_tif or args.out_html):
        raise SystemExit(
            "[visualize] batch mode (omit --sector or use --sector all) does not support "
            "--weight-tif or --out-html; each sector writes <output>/<sector>_area_context_map.html."
        )
    if batch and point_link_only and (args.weight_tif or args.out_html):
        raise SystemExit(
            "[visualize] batch point-link-only mode does not support --weight-tif or --out-html."
        )

    corine_tif = _resolve_under_root(Path(cfg.resolved["proxy_common"]["corine_tif"]), root)
    population_tif = _resolve_under_root(
        Path(cfg.require("proxy_common", "population_tif")), root
    )

    out_dir = _resolve_under_root(Path(args.output), root)
    out_dir.mkdir(parents=True, exist_ok=True)

    cams_nc: Path | None = None
    if not getattr(args, "no_cams", False):
        cams_nc = _resolve_under_root(
            Path(cfg.resolved["emissions"]["cams_2019_nc"]), root
        )

    country_code = str(getattr(args, "country", "EL"))
    pad_deg = float(args.pad_deg)
    max_width = int(args.max_width)
    max_height = int(args.max_height)
    weight_display = str(args.weight_display)
    region_raw = getattr(args, "region", None)
    region = str(region_raw).strip().lower() if region_raw else None
    override_bbox = _parse_bbox_cli(getattr(args, "bbox", None))
    if override_bbox and region and region not in ("country", "full"):
        print(
            f"[visualize] --bbox provided; ignoring --region={region!r}.",
            flush=True,
        )
        region = None
    if override_bbox:
        w, s, e, n = override_bbox
        print(
            f"[visualize] focus bbox: W={w:.3f} S={s:.3f} E={e:.3f} N={n:.3f} (custom)",
            flush=True,
        )
    elif region and region not in ("country", "full"):
        print(f"[visualize] focus region: {region}", flush=True)
    else:
        print("[visualize] focus region: full country (from weight raster bbox)", flush=True)

    if point_link_only:
        if batch:
            sector_keys = [
                str(e["key"])
                for e in entries
                if bool(e.get("enabled", True)) and str(e["key"]) in _POINT_LINK_SECTORS
            ]
            if not sector_keys:
                raise SystemExit(
                    "[visualize] batch point-link: no enabled sectors in sectors.yaml "
                    f"with point-link previews (keys: {sorted(_POINT_LINK_SECTORS)})."
                )
            print(
                f"[visualize] batch point-link: {len(sector_keys)} sector(s) -> {out_dir} "
                f"({', '.join(sector_keys)})",
                flush=True,
            )
        else:
            if sector_token not in _POINT_LINK_SECTORS:
                raise SystemExit(
                    f"[visualize] --point-link-only requires a point-link sector "
                    f"({', '.join(sorted(_POINT_LINK_SECTORS))}), got {sector_token!r}."
                )
            sector_keys = [sector_token]
    elif batch:
        sector_keys = [
            str(e["key"])
            for e in entries
            if bool(e.get("enabled", True)) and str(e["key"]) in _AREA_PREVIEW_SECTORS
        ]
        if not sector_keys:
            raise SystemExit(
                "[visualize] batch: no enabled sectors with area HTML previews in sectors.yaml "
                f"(preview-capable keys: {sorted(_AREA_PREVIEW_SECTORS)})."
            )
        print(
            f"[visualize] batch: {len(sector_keys)} sector(s) -> {out_dir} ({', '.join(sector_keys)})",
            flush=True,
        )
    else:
        sector_keys = [sector_token]

    failed = 0
    for sector_key in sector_keys:
        sector_entry = next((e for e in entries if str(e.get("key")) == sector_key), None)
        if sector_entry is None:
            if batch:
                print(f"[visualize] skip {sector_key}: not listed in sectors.yaml", flush=True)
                failed += 1
                continue
            raise SystemExit(f"Unknown sector key: {sector_key!r}")

        if not point_link_only and sector_key not in _AREA_PREVIEW_SECTORS:
            if batch:
                print(
                    f"[visualize] skip {sector_key}: no area HTML preview in this CLI "
                    f"(supported: {sorted(_AREA_PREVIEW_SECTORS)})",
                    flush=True,
                )
                failed += 1
                continue
            print(
                f"[visualize] no HTML preview implemented for sector={sector_key!r} "
                f"(supported: {', '.join(sorted(_AREA_PREVIEW_SECTORS))})."
            )
            return 1

        sector_cfg = load_yaml(root / str(sector_entry["config"]))

        if not point_link_only:
            if args.weight_tif and not batch:
                weight_tif = _resolve_under_root(Path(args.weight_tif), root)
            else:
                default_name = str(
                    sector_cfg.get(
                        "output_filename",
                        _AREA_DEFAULT_WEIGHT_TIFS.get(sector_key, "areasource.tif"),
                    )
                )
                weight_tif = _resolve_under_root(
                    Path(sector_cfg["output_dir"]) / default_name,
                    root,
                )

            if batch or not args.out_html:
                out_html = out_dir / f"{sector_key}_area_context_map.html"
            else:
                out_html = _resolve_under_root(Path(args.out_html), root)
                out_html.parent.mkdir(parents=True, exist_ok=True)

            try:
                path = _write_area_preview_html(
                    root=root,
                    path_cfg=cfg.resolved,
                    sector_key=sector_key,
                    sector_cfg=sector_cfg,
                    weight_tif=weight_tif,
                    out_html=out_html,
                    corine_tif=corine_tif,
                    population_tif=population_tif,
                    cams_nc=cams_nc,
                    country_code=country_code,
                    pad_deg=pad_deg,
                    max_width=max_width,
                    max_height=max_height,
                    weight_display=weight_display,
                    region=region,
                    override_bbox=override_bbox,
                )
            except Exception as exc:
                print(f"[visualize] skip {sector_key}: {exc}", flush=True)
                failed += 1
                continue

            print(f"[visualize] sector={sector_key} wrote {path}", flush=True)

        if point_link and sector_key in _POINT_LINK_SECTORS:
            art = _point_link_artifact_paths(root, sector_key, link_year)
            if art is None:
                print(
                    f"[visualize] skip point-link {sector_key}: missing "
                    f"{sector_key}_cams_facility_link_{link_year}.tif or matches CSV "
                    f"(run: python -m PROXY.main match-points --sector {sector_key} --year {link_year}).",
                    flush=True,
                )
                if point_link_only:
                    failed += 1
            else:
                link_tif, matches_csv = art
                out_pt = out_dir / f"{sector_key}_point_context_map.html"
                try:
                    from PROXY.visualization.point_link_context_map import (
                        write_point_link_context_html,
                    )

                    ptp = write_point_link_context_html(
                        root=root,
                        sector_key=sector_key,
                        link_tif=link_tif,
                        matches_csv=matches_csv,
                        out_html=out_pt,
                        pad_deg=pad_deg,
                        max_width=max_width,
                        max_height=max_height,
                        region=region,
                        override_bbox=override_bbox,
                    )
                    print(f"[visualize] sector={sector_key} wrote {ptp}", flush=True)
                except Exception as exc:
                    print(f"[visualize] skip point-link {sector_key}: {exc}", flush=True)
                    failed += 1

    if failed:
        print(f"[visualize] finished with {failed} failure(s) out of {len(sector_keys)}", flush=True)
        return 1
    if batch:
        print(f"[visualize] batch OK: {len(sector_keys)} file(s) under {out_dir}", flush=True)
    return 0


def _validate_cmd(args: argparse.Namespace) -> int:
    _ensure_gtiff_srs_source_epsg()
    root = Path(__file__).resolve().parents[1]
    cfg = load_path_config(Path(args.config))
    cfg.resolved["proxy_common"]["corine_tif"] = discover_corine(
        root, Path(cfg.require("proxy_common", "corine_tif"))
    )
    cfg.resolved["emissions"]["cams_2019_nc"] = discover_cams_emissions(
        root, Path(cfg.require("emissions", "cams_2019_nc"))
    )

    import numpy as np
    import rasterio

    from PROXY.core.cams.grid import build_cam_cell_id
    from PROXY.core.raster.normalize import validate_weight_sums

    warnings: list[str] = []
    errors: list[str] = []

    # Global input checks.
    for label, raw in (
        ("corine_tif", cfg.require("proxy_common", "corine_tif")),
        ("nuts_gpkg", cfg.require("proxy_common", "nuts_gpkg")),
        ("population_tif", cfg.require("proxy_common", "population_tif")),
        ("cams_2019_nc", cfg.require("emissions", "cams_2019_nc")),
    ):
        p = Path(raw)
        if not p.is_file():
            errors.append(f"missing required input {label}: {p}")

    sectors_data = load_yaml(root / "PROXY" / "config" / "sectors.yaml")
    entries = sectors_data.get("sectors", []) or []
    target = str(args.sector).strip() if args.sector else ""
    selected = [
        e for e in entries if bool(e.get("enabled", True)) and (not target or str(e.get("key")) == target)
    ]
    if not selected:
        errors.append(f"no enabled sectors matched target={target!r}")

    # Validate generated outputs and per-cell normalization for area-source rasters.
    cams_nc = Path(cfg.require("emissions", "cams_2019_nc"))
    for entry in selected:
        key = str(entry["key"])
        sector_cfg = load_yaml(root / str(entry["config"]))
        out_dir = root / str(sector_cfg["output_dir"])
        area_name = str(sector_cfg.get("output_filename") or f"{key}_areasource.tif")
        area_path = out_dir / area_name
        if not area_path.is_file():
            warnings.append(f"[{key}] missing output raster: {area_path}")
            continue
        try:
            with rasterio.open(area_path) as src:
                if src.count < 1:
                    errors.append(f"[{key}] output has no bands: {area_path}")
                    continue
                if src.crs is None:
                    errors.append(f"[{key}] output has no CRS: {area_path}")
                    continue
                ref = {
                    "height": src.height,
                    "width": src.width,
                    "transform": src.transform,
                    "crs": src.crs.to_string(),
                }
                cam_cell_id = build_cam_cell_id(cams_nc, ref)
                for b in range(1, src.count + 1):
                    arr = src.read(b).astype(np.float32)
                    if src.nodata is not None and np.isfinite(src.nodata):
                        arr = np.where(arr == np.float32(src.nodata), np.nan, arr)
                    verrs = validate_weight_sums(arr, cam_cell_id, None, tol=1e-3)
                    if verrs:
                        warnings.append(
                            f"[{key}] band={b} per-cell sums off in {len(verrs)} CAMS cells (sample: {verrs[:2]})"
                        )
        except Exception as exc:
            errors.append(f"[{key}] failed to inspect output {area_path}: {exc}")

    print(f"[validate] sector={args.sector or 'all'} strict={bool(args.strict)}")
    if errors:
        print(f"[validate] errors: {len(errors)}")
        for msg in errors:
            print(f"  - ERROR: {msg}")
    if warnings:
        print(f"[validate] warnings: {len(warnings)}")
        for msg in warnings:
            print(f"  - WARN: {msg}")

    if errors:
        return 2
    if warnings and bool(args.strict):
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    root = _root()
    parser = argparse.ArgumentParser(
        description="Unified proxy-weight pipeline entrypoint."
    )
    parser.add_argument(
        "--config",
        default=str(root / "PROXY" / "config" / "paths.yaml"),
        help="Path to root path-config yaml.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="Build area/point proxy outputs.")
    p_build.add_argument("--sector", default=None, help="Sector key (e.g. I_Offroad).")
    p_build.add_argument("--country", default="EL", help="Country code for NUTS domain clip.")
    p_build.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Root logging level. INFO shows per-sector progress (default); DEBUG also prints tracebacks.",
    )
    p_build.add_argument(
        "--part",
        default="both",
        choices=("both", "area", "point"),
        help=(
            "J_Waste only: write both area+point GeoTIFFs (both), only waste_areasource.tif (area), "
            "or only waste_pointsource.tif (point). Other sectors ignore this (always full area build)."
        ),
    )
    p_build.set_defaults(func=_build_cmd)

    p_alpha = sub.add_parser("alpha", help="Compute alpha factors.")
    p_alpha.add_argument("--country", default="EL", help="Country code (NUTS/CAMS domain).")
    p_alpha.add_argument("--pollutant", default=None, help="Optional pollutant key.")
    p_alpha.set_defaults(func=_alpha_cmd)

    p_match = sub.add_parser("match-points", help="Run CAMS-to-facility matching.")
    p_match.add_argument("--sector", required=True, help="Point-source sector key.")
    p_match.add_argument("--year", type=int, default=2019, help="Reference CAMS year.")
    p_match.add_argument(
        "--cams-iso3",
        default="GRC",
        help="CAMS country_id ISO3 filter (e.g., GRC).",
    )
    p_match.add_argument("--pollutant", default=None, help="Optional pollutant token (e.g., NOX).")
    p_match.add_argument(
        "--max-points",
        type=int,
        default=50000,
        help="Optional cap on CAMS points matched (highest pollutant load first when pollutant_var is set).",
    )
    p_match.add_argument(
        "--no-link-geotiff",
        action="store_true",
        help="Do not write the 2-band CAMS↔facility link GeoTIFF next to the match CSV.",
    )
    p_match.set_defaults(func=_match_points_cmd)

    p_vis = sub.add_parser("visualize", help="Render proxy output previews.")
    p_vis.add_argument(
        "--sector",
        default=None,
        help=(
            "Sector key to visualize, or omit / pass 'all' to render every enabled sector "
            "that has an area HTML preview (writes <output>/<sector>_area_context_map.html)."
        ),
    )
    p_vis.add_argument(
        "--country",
        default="EL",
        help="NUTS / merge country (e.g. EL for Greece; used for E_Solvents / J_Waste pipeline cfg merge in previews).",
    )
    p_vis.add_argument(
        "--output",
        default="OUTPUT/Proxy_visualization",
        help="Output folder (default HTML name if --out-html omitted).",
    )
    p_vis.add_argument(
        "--weight-tif",
        default=None,
        help="Override path to area weight GeoTIFF (default: sector output_filename).",
    )
    p_vis.add_argument(
        "--out-html",
        default=None,
        type=Path,
        help="Output HTML path (default: <output>/<sector>_area_context_map.html).",
    )
    p_vis.add_argument("--pad-deg", type=float, default=0.02, help="Pad map extent in degrees.")
    p_vis.add_argument("--max-width", type=int, default=1400, help="Overlay raster width in px.")
    p_vis.add_argument("--max-height", type=int, default=1200, help="Overlay raster height in px.")
    p_vis.add_argument(
        "--no-cams",
        action="store_true",
        help="Do not load CAMS NetCDF (skip CAMS grid and per-cell weight mode).",
    )
    p_vis.add_argument(
        "--weight-display",
        choices=("auto", "global_log", "per_cell"),
        default="auto",
        help=(
            "Weight colours: auto (sector YAML / default per-cell when CAMS NetCDF loads), "
            "global log10, or forced per-CAMS-cell 0–1."
        ),
    )
    p_vis.add_argument(
        "--region",
        default="attica",
        choices=(
            "attica",
            "thessaloniki",
            "patras",
            "heraklion",
            "crete",
            "athens_extended",
            "country",
            "full",
        ),
        help=(
            "Focus preview on a named region (default: attica). Use 'country' or 'full' "
            "to render the entire weight raster bbox. Overridden by --bbox."
        ),
    )
    p_vis.add_argument(
        "--bbox",
        default=None,
        help=(
            "Override focus window with an explicit WGS84 bbox as 'W,S,E,N' "
            "(e.g. '23.4,37.8,24.2,38.4'). Takes precedence over --region."
        ),
    )
    p_vis.add_argument(
        "--point-link",
        action="store_true",
        help=(
            "Also write <output>/<sector>_point_context_map.html for sectors with point matching "
            f"({', '.join(sorted(POINT_LINK_SECTOR_KEYS))}) when the link GeoTIFF and match CSV exist."
        ),
    )
    p_vis.add_argument(
        "--point-link-only",
        action="store_true",
        help="Only render point-link HTML maps (no area previews). Implies --point-link.",
    )
    p_vis.add_argument(
        "--link-year",
        type=int,
        default=2019,
        help="Match / link GeoTIFF year suffix used for *_point_matches_<year>.csv and *_cams_facility_link_<year>.tif.",
    )
    p_vis.set_defaults(func=_visualize_cmd)

    p_val = sub.add_parser("validate", help="Run quality checks for outputs.")
    p_val.add_argument("--sector", default=None, help="Optional single sector key.")
    p_val.add_argument("--strict", action="store_true", help="Fail on warnings.")
    p_val.set_defaults(func=_validate_cmd)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
