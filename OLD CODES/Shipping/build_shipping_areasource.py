#!/usr/bin/env python3
"""CLI: build ``Shipping_areasource.tif`` (CORINE+NUTS reference grid; paths from YAML)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    root = _root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    ap = argparse.ArgumentParser(
        description=(
            "Build shipping proxy weights on the CORINE+NUTS fine grid. "
            "Defaults match SourceProxies/build_proxies (see Shipping/config/shipping_area.yaml). "
            "Alternatively pass --fine-grid for a custom reference GeoTIFF."
        )
    )
    ap.add_argument(
        "--config",
        type=Path,
        default=root / "Shipping" / "config" / "shipping_area.yaml",
        help="YAML with paths, corine_window, proxy options (default: Shipping/config/shipping_area.yaml)",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output folder (default: output.filename next to yaml or set in YAML later — use with --output-dir)",
    )
    ap.add_argument(
        "--fine-grid",
        type=Path,
        default=None,
        help="Optional reference GeoTIFF; if set, skips CORINE+NUTS window and uses this grid's profile",
    )
    ap.add_argument("--cams-nc", type=Path, default=None, help="Override paths.cams_nc from YAML")
    ap.add_argument("--corine", type=Path, default=None, help="Override paths.corine from YAML")
    ap.add_argument("--nuts-gpkg", type=Path, default=None, help="Override paths.nuts_gpkg from YAML")
    ap.add_argument("--emodnet", type=Path, default=None, help="Override paths.emodnet from YAML")
    ap.add_argument("--osm-gpkg", type=Path, default=None, help="Override paths.osm_gpkg from YAML")
    ap.add_argument(
        "--write-diagnostics",
        action="store_true",
        help="Write intermediate rasters (also proxy.write_diagnostics in YAML)",
    )
    ap.add_argument("--osm-subdivide", type=int, default=None)
    ap.add_argument("--quiet", action="store_true")

    args = ap.parse_args()
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    import yaml

    from Shipping.shipping_areasource import (
        build_ref_corine_nuts,
        load_ref_from_fine_grid_tif,
        run_shipping_areasource,
    )

    cfg_path = args.config if args.config.is_absolute() else root / args.config
    if not cfg_path.is_file():
        raise SystemExit(f"Config not found: {cfg_path}")
    with cfg_path.open(encoding="utf-8") as f:
        yml: dict = yaml.safe_load(f) or {}

    paths = dict(yml.get("paths") or {})
    if args.cams_nc:
        paths["cams_nc"] = str(args.cams_nc)
    if args.corine:
        paths["corine"] = str(args.corine)
    if args.nuts_gpkg:
        paths["nuts_gpkg"] = str(args.nuts_gpkg)
    if args.emodnet:
        paths["emodnet"] = str(args.emodnet)
    if args.osm_gpkg:
        paths["osm_gpkg"] = str(args.osm_gpkg)

    country = yml.get("country") or {}
    cw = yml.get("corine_window") or {}
    proxy_blk = yml.get("proxy") or {}
    out_conf = yml.get("output") or {}

    from SourceProxies.grid import resolve_path

    cams_nc = resolve_path(root, Path(paths["cams_nc"]))
    emodnet = resolve_path(root, Path(paths["emodnet"]))
    osm_gpkg = resolve_path(root, Path(paths["osm_gpkg"]))
    corine_path_arg: Path | None = None
    fine_grid = Path(args.fine_grid).expanduser().resolve() if args.fine_grid else None

    if fine_grid is not None and fine_grid.is_file():
        ref = load_ref_from_fine_grid_tif(fine_grid)
        corine_path_arg = resolve_path(root, Path(paths["corine"]))
    else:
        nuts_cntr = str(country.get("nuts_cntr", "EL"))
        pad_m = float(cw.get("pad_m", 5000.0))
        ref = build_ref_corine_nuts(
            root,
            corine=paths.get("corine"),
            nuts_gpkg=paths["nuts_gpkg"],
            nuts_cntr=nuts_cntr,
            pad_m=pad_m,
        )

    out_dir = args.output_dir
    if out_dir is None:
        od = out_conf.get("dir")
        if not od:
            raise SystemExit(
                "Set output.dir in YAML, pass --output-dir, or use: "
                "python -m SourceProxies.build_proxies --only shipping_area"
            )
        out_dir = Path(od)
    out_dir = out_dir.expanduser().resolve()
    if not out_dir.is_absolute():
        out_dir = (root / out_dir).resolve()
    out_name = str(out_conf.get("filename", "Shipping_areasource.tif"))
    osm_sub = int(args.osm_subdivide if args.osm_subdivide is not None else proxy_blk.get("osm_subdivide", 4))
    write_diag = bool(args.write_diagnostics or proxy_blk.get("write_diagnostics", False))

    res = run_shipping_areasource(
        cams_nc=cams_nc,
        corine_path=corine_path_arg,
        osm_gpkg=osm_gpkg,
        emodnet_path=emodnet,
        output_folder=out_dir,
        ref=ref,
        fine_grid_tif=None,
        osm_subdivide=osm_sub,
        output_filename=out_name,
        write_diagnostics=write_diag,
    )
    print(res["output_tif"])
    err_list = res.get("validate_errors") or []
    if err_list:
        logging.warning(
            "validate_weight_sums reported %d issues (first 5): %s",
            len(err_list),
            err_list[:5],
        )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        sys.exit(0)
