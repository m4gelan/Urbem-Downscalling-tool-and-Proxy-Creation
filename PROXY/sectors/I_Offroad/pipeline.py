"""GNFR I offroad pipeline: three-leg proxy construction and CEIP share allocation.

This module replaces the former ``multiband_builder.py`` catch-all.  The builder now
handles entrypoint concerns, while this file owns config merging and the end-to-end
offroad workflow: rail, pipeline, and nonroad proxies mixed with CEIP triple shares,
then normalized within CAMS cells per pollutant band.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from PROXY.core.alpha import (
    apply_offroad_yaml_overrides,
    build_share_arrays,
    lookup_offroad_triple_for_iso3,
    norm_pollutant_key,
    read_ceip_shares,
    resolve_offroad_triple_with_yaml,
)
from PROXY.core.cams.grid import build_cam_cell_id
from PROXY.core.corine.raster import resolve_corine_tif, warp_corine_codes_nearest
from PROXY.core.country_raster import rasterize_country_indices
from PROXY.core.dataloaders import resolve_path
from PROXY.core.grid import reference_window_profile
from PROXY.core.io import write_geotiff, write_json
from PROXY.core.raster import normalize_within_cams_cells
from PROXY.sectors.I_Offroad.proxy_rules import osm_railway_line_filter_sets
from PROXY.sectors.I_Offroad.nonroad_corine_only import build_nonroad_corine_proxy
from PROXY.sectors.I_Offroad.pipeline_osm import build_pipeline_z_final
from PROXY.sectors.I_Offroad.rail_osm import build_rail_coverage_and_z

logger = logging.getLogger(__name__)


def _none_if_blank(value: Any) -> Any:
    """Treat YAML spellings like ``null`` / ``none`` / empty string as missing."""
    if isinstance(value, str) and value.strip().lower() in ("null", "none", ""):
        return None
    return value


def _default_triple(defaults: dict[str, Any]) -> tuple[float, float, float]:
    """Return the rail/pipeline/nonroad fallback shares used when CEIP has no row."""
    return (
        float(defaults.get("default_shares_rail", 1.0 / 3)),
        float(defaults.get("default_shares_pipe", 1.0 / 3)),
        float(defaults.get("default_shares_nonroad", 1.0 / 3)),
    )


def _normalized_pollutants(area_cfg: dict[str, Any]) -> list[str]:
    """Normalize pollutant labels once so CEIP lookup and output band names agree."""
    return [norm_pollutant_key(str(x)) for x in (area_cfg.get("pollutants") or ["nox", "pm2_5"])]


def _pollutant_aliases(area_cfg: dict[str, Any]) -> dict[str, str]:
    """Normalize configured CEIP pollutant aliases to the same keys used by core.alpha."""
    aliases = dict(area_cfg.get("ceip_pollutant_aliases") or {})
    for k, v in list(aliases.items()):
        aliases[str(k).upper()] = norm_pollutant_key(str(v))
    return aliases


def _country_token_to_iso3_upper(token: str) -> str:
    """Resolve YAML country tokens (ISO3 or common ISO2 / CEIP quirks) for CEIP dict keys."""
    t = str(token).strip().upper()
    if len(t) == 3 and t.isalpha():
        return t
    iso2 = {
        "FR": "FRA",
        "DE": "DEU",
        "GE": "DEU",
        "AT": "AUT",
        "AU": "AUT",
        "NL": "NLD",
        "UK": "GBR",
        "GB": "GBR",
        "EL": "GRC",
        "GR": "GRC",
    }
    return iso2.get(t, t)


def apply_offroad_ceip_reference_pool(
    share_dict: dict[str, dict[str, tuple[float, float, float]]],
    *,
    area_proxy: dict[str, Any],
    pollutants: list[str],
    fallback_iso: str,
    logger: logging.Logger | None = None,
) -> None:
    """When enabled, replace target-country triples with the mean of reference ISO3 triples from CEIP."""
    fb = area_proxy.get("alpha_fallback") or {}
    if not bool(fb.get("enabled")):
        return
    targets = {_country_token_to_iso3_upper(str(x)) for x in (fb.get("target_countries_iso3") or [])}
    refs_raw = list(
        fb.get("reference_countries_iso3")
        or fb.get("reference_countries")
        or [],
    )
    refs = [_country_token_to_iso3_upper(str(x)) for x in refs_raw]
    refs = [r for r in refs if len(r) == 3 and r.isalpha()]
    iso_target = _country_token_to_iso3_upper(fallback_iso)
    if iso_target not in targets:
        return
    pol_spec = fb.get("pollutants", "all")
    if pol_spec not in (None, "all") and isinstance(pol_spec, (list, tuple)):
        wanted = {norm_pollutant_key(str(x)) for x in pol_spec}
    else:
        wanted = {norm_pollutant_key(str(x)) for x in pollutants}

    for pol in pollutants:
        pk = norm_pollutant_key(str(pol))
        if pk not in wanted:
            continue
        table = share_dict.setdefault(pk, {})
        trips: list[tuple[float, float, float]] = []
        for r_iso in refs:
            t = table.get(r_iso)
            if t is not None and len(t) == 3:
                trips.append(tuple(float(x) for x in t))
        if not trips:
            if logger:
                logger.warning(
                    "I_Offroad alpha_fallback: no CEIP triple for pollutant %s among refs %s",
                    pk,
                    refs,
                )
            continue
        m = tuple(sum(x[i] for x in trips) / len(trips) for i in range(3))
        sm = sum(m)
        if sm > 0:
            m = tuple(x / sm for x in m)
        table[iso_target] = m
        if logger:
            logger.info(
                "I_Offroad alpha_fallback: %s %s triple <- mean(ref %s) = (%.4f, %.4f, %.4f)",
                iso_target,
                pk,
                refs,
                m[0],
                m[1],
                m[2],
            )


def merge_offroad_pipeline_cfg(
    root: Path,
    path_cfg: dict[str, Any],
    sector_cfg: dict[str, Any],
    *,
    country: str,
    output_path: Path,
) -> dict[str, Any]:
    """Resolve paths and shape ``offroad.yaml`` into the structure used by the runner."""
    area_cfg = dict(sector_cfg.get("area_proxy") or {})
    defaults = dict(area_cfg.get("defaults") or {})
    proxy_cfg = dict(area_cfg.get("proxy") or {})

    wb = area_cfg.get("ceip_workbook") or (path_cfg.get("proxy_common") or {}).get("alpha_workbook")
    if not wb:
        raise ValueError("I_Offroad: set area_proxy.ceip_workbook or paths.proxy_common.alpha_workbook.")

    fac_tif = _none_if_blank(area_cfg.get("facilities_tif"))
    paths: dict[str, Any] = {
        "corine": str(resolve_path(root, Path(path_cfg["proxy_common"]["corine_tif"])).resolve()),
        "nuts_gpkg": str(resolve_path(root, Path(path_cfg["proxy_common"]["nuts_gpkg"])).resolve()),
        "osm_gpkg": str(resolve_path(root, Path(path_cfg["osm"]["offroad"])).resolve()),
        "cams_nc": str(resolve_path(root, Path(path_cfg["emissions"]["cams_2019_nc"])).resolve()),
        "ceip_workbook": str(resolve_path(root, Path(str(wb))).resolve()),
        "facilities_tif": str(resolve_path(root, Path(str(fac_tif))).resolve()) if fac_tif else None,
    }

    ceip_year_raw = defaults.get("ceip_year")
    ceip_year: int | None
    if ceip_year_raw is None or str(ceip_year_raw).strip() == "":
        ceip_year = None
    else:
        ceip_year = int(ceip_year_raw)

    fallback_iso = str(
        sector_cfg.get("cams_country_iso3") or defaults.get("fallback_country_iso3", "GRC")
    ).strip().upper()

    return {
        "_project_root": root,
        "paths": paths,
        "country": {
            "nuts_country": str(country).strip().upper(),
            "fallback_iso3": fallback_iso,
            "cntr_code_to_iso3": dict(area_cfg.get("cntr_code_to_iso3") or {}),
        },
        "corine_window": {"pad_m": float(sector_cfg.get("pad_m", 5000.0))},
        "ceip": {
            "sheet": _none_if_blank(area_cfg.get("ceip_sheet")),
            "year": ceip_year,
            "pollutant_aliases": _pollutant_aliases(area_cfg),
            "default_triple": _default_triple(defaults),
        },
        "pollutants": _normalized_pollutants(area_cfg),
        "proxy": proxy_cfg,
        "area_proxy": {k: v for k, v in area_cfg.items() if k != "defaults"},
        "defaults": defaults,
        "output": {"path": str(output_path.resolve())},
        "path_cfg": path_cfg,
    }


def _log_raster_stats(name: str, arr: np.ndarray) -> None:
    """Emit compact stats for key intermediate rasters in the offroad proxy chain."""
    a = np.asarray(arr)
    finite = np.isfinite(a)
    if not np.any(finite):
        logger.info("I_Offroad %s: shape=%s dtype=%s no finite values", name, a.shape, a.dtype)
        return
    vals = a[finite]
    logger.info(
        "I_Offroad %s: shape=%s dtype=%s min=%.6g mean=%.6g max=%.6g",
        name,
        a.shape,
        a.dtype,
        float(np.min(vals)),
        float(np.mean(vals)),
        float(np.max(vals)),
    )


def _build_reference_grid(cfg: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    """Build the CORINE/NUTS reference grid used by every offroad proxy layer."""
    paths = cfg["paths"]
    corine_path = resolve_corine_tif(Path(paths["corine"]))
    ref = reference_window_profile(
        corine_path=corine_path,
        nuts_gpkg=Path(paths["nuts_gpkg"]),
        nuts_country=str(cfg["country"]["nuts_country"]),
        pad_m=float(cfg["corine_window"].get("pad_m", 5000.0)),
    )
    return corine_path, ref


def run_offroad_pipeline(root: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    """Build multiband I_Offroad weights from rail, pipeline, nonroad and CEIP shares."""
    paths = cfg["paths"]
    proxy_cfg = cfg["proxy"]
    country_cfg = cfg["country"]
    ceip_cfg = cfg["ceip"]
    pollutants = [str(p) for p in cfg["pollutants"]]
    fallback_iso = str(country_cfg["fallback_iso3"]).strip().upper()
    default_triple = tuple(float(x) for x in ceip_cfg["default_triple"])

    corine_path, ref = _build_reference_grid(cfg)
    logger.info(
        "I_Offroad reference grid: crs=%r width=%s height=%s",
        ref.get("crs"),
        ref.get("width"),
        ref.get("height"),
    )
    logger.info("I_Offroad paths: CAMS=%s", paths["cams_nc"])
    logger.info("I_Offroad paths: CORINE=%s", corine_path)
    logger.info("I_Offroad paths: OSM=%s", paths["osm_gpkg"])
    logger.info("I_Offroad paths: NUTS=%s", paths["nuts_gpkg"])
    logger.info("I_Offroad paths: CEIP=%s", paths["ceip_workbook"])

    clc_nn = warp_corine_codes_nearest(corine_path, ref)
    _log_raster_stats("clc_nn", clc_nn)

    bad_rail, life_rail = osm_railway_line_filter_sets(cfg.get("path_cfg") or {})
    _, z_rail = build_rail_coverage_and_z(
        Path(paths["osm_gpkg"]),
        ref,
        rail_buffer_m=float(proxy_cfg.get("rail_buffer_m", 75)),
        osm_subdivide=int(proxy_cfg.get("osm_subdivide", 4)),
        bad_line_types=bad_rail,
        lifecycle_disallow=life_rail,
        proxy_cfg=proxy_cfg,
    )
    _log_raster_stats("z_rail", z_rail)

    z_pipe, _, _ = build_pipeline_z_final(
        Path(paths["osm_gpkg"]),
        root,
        ref,
        {"facilities_tif": paths.get("facilities_tif")},
        proxy_cfg,
    )
    _log_raster_stats("z_pipeline", z_pipe)

    nr = build_nonroad_corine_proxy(clc_nn=clc_nn, proxy_cfg=proxy_cfg)
    p_nr = nr["p_nr"]
    _log_raster_stats("p_nonroad", p_nr)

    share_dict = read_ceip_shares(
        Path(paths["ceip_workbook"]),
        sheet=ceip_cfg.get("sheet") if isinstance(ceip_cfg.get("sheet"), str) else None,
        pollutant_aliases=dict(ceip_cfg.get("pollutant_aliases") or {}),
        pollutants_wanted=pollutants,
        cntr_code_to_iso3=dict(country_cfg.get("cntr_code_to_iso3") or {}),
        default_triple=default_triple,
        ceip_year=ceip_cfg.get("year"),
    )

    apply_offroad_ceip_reference_pool(
        share_dict,
        area_proxy=dict(cfg.get("area_proxy") or {}),
        pollutants=pollutants,
        fallback_iso=fallback_iso,
        logger=logger,
    )

    logger.info(
        "[I_Offroad] downscale triple-leg shares (1A3c rail / 1A3ei pipeline / 1A3eii non-road) "
        "for CAMS ISO3=%r, NUTS window=%r:",
        fallback_iso,
        country_cfg["nuts_country"],
    )
    for pol in pollutants:
        triple = lookup_offroad_triple_for_iso3(
            share_dict, pol, fallback_iso, default_triple=default_triple
        )
        sr0, sp0, sn0 = resolve_offroad_triple_with_yaml(
            triple,
            default_triple,
            pollutant=pol,
            iso3=fallback_iso,
            logger=logger,
        )
        logger.info(
            "  %s: rail=%.4f  pipeline=%.4f  non_road=%.4f  (sum=%.4f)",
            pol,
            sr0,
            sp0,
            sn0,
            sr0 + sp0 + sn0,
        )

    country_idx, idx_to_iso = rasterize_country_indices(
        Path(paths["nuts_gpkg"]),
        ref,
        dict(country_cfg.get("cntr_code_to_iso3") or {}),
        fallback_iso,
    )
    logger.info(
        "I_Offroad country raster: country_ids=%s pixels_with_country=%d/%d",
        sorted(idx_to_iso.items()),
        int(np.count_nonzero(country_idx > 0)),
        int(country_idx.size),
    )

    iso_codes_in_grid = sorted(
        {str(v).strip().upper() for v in idx_to_iso.values() if v}
        | {fallback_iso.strip().upper()}
    )
    apply_offroad_yaml_overrides(
        share_dict,
        pollutants=pollutants,
        isos=iso_codes_in_grid,
        default_triple=default_triple,
        logger=logger,
    )

    cam_id = build_cam_cell_id(Path(paths["cams_nc"]), ref)
    logger.info(
        "I_Offroad CAMS cells: valid_pixels=%d/%d unique_cells=%d",
        int(np.count_nonzero(cam_id >= 0)),
        int(cam_id.size),
        int(np.unique(cam_id[cam_id >= 0]).size) if np.any(cam_id >= 0) else 0,
    )

    bands: list[np.ndarray] = []
    fb_summary: list[tuple[str, int]] = []
    for pol in pollutants:
        sr, sp, sn = build_share_arrays(
            country_idx,
            idx_to_iso,
            share_dict,
            pol,
            fallback_iso,
            default_triple=default_triple,
        )
        p_comb = (sr * z_rail + sp * z_pipe + sn * p_nr).astype(np.float32)
        _log_raster_stats(f"combined_proxy[{pol}]", p_comb)
        w_arr, _fb = normalize_within_cams_cells(
            p_comb,
            cam_id,
            None,
            return_fallback_mask=False,
            context=f"I_Offroad pollutant={pol}",
            uniform_fallback_summary=fb_summary,
        )
        bands.append(np.asarray(w_arr, dtype=np.float32))

    if fb_summary:
        tot = sum(c for _, c in fb_summary)
        logger.info(
            "I_Offroad: CAMS-cell uniform proxy fallback %d pollutant pass(es), %d fine pixels total.",
            len(fb_summary),
            tot,
        )

    stacked = np.stack(bands, axis=0) if bands else np.zeros((0, 0, 0), dtype=np.float32)
    out_path = Path(cfg["output"]["path"]).resolve()
    write_geotiff(
        path=out_path,
        array=stacked,
        crs=ref["crs"],
        transform=ref["transform"],
        nodata=0.0,
        band_descriptions=pollutants,
        tiled=True,
        predictor=3,
        bigtiff="IF_SAFER",
        tags={
            "sector": "I_Offroad",
            "allocation": "osm_rail_osm_pipeline_corine_nonroad_ceip_shares_cams_norm",
            "n_bands": str(len(pollutants)),
        },
    )

    manifest = {
        "sector": "I_Offroad",
        "layer": "area_source_weights_multiband",
        "output_geotiff": str(out_path),
        "pollutants": pollutants,
        "band_order": pollutants,
        "cams_nc": str(paths["cams_nc"]),
        "cams_country_iso3": fallback_iso,
        "ceip_workbook": str(paths["ceip_workbook"]),
        "ceip_sheet": ceip_cfg.get("sheet"),
        "ceip_year": ceip_cfg.get("year"),
        "corine_path": str(corine_path),
        "osm_gpkg": str(paths["osm_gpkg"]),
        "nuts_gpkg": str(paths["nuts_gpkg"]),
        "reference": {
            "crs": ref["crs"],
            "width": int(ref["width"]),
            "height": int(ref["height"]),
        },
        "area_proxy": cfg.get("area_proxy") or {},
        "defaults": cfg.get("defaults") or {},
        "proxy": proxy_cfg,
    }
    write_json(out_path.with_suffix(".json"), manifest)
    logger.info("I_Offroad pipeline complete: wrote %s (%d bands)", out_path, len(pollutants))
    return {"output_path": str(out_path), "manifest_path": str(out_path.with_suffix(".json"))}
