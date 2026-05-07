"""
A_PublicPower area-weight builder (GNFR A, area sources only).

Steps: (1) define the output reference grid from CORINE + NUTS + ``--country``;
(2) select CAMS area sources for that country; (3) for each source, clip CORINE and
population inside the CAMS cell, form a normalized share tensor, add to the ref
grid; (4) write GeoTIFF + JSON manifest.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from PROXY.core.grid import reference_window_profile
from PROXY.core.io import write_geotiff, write_json
from PROXY.core.raster.country_clip import resolve_cams_country_iso3
from PROXY.sectors.A_PublicPower.cams_area_mask import public_power_area_mask
from PROXY.sectors.A_PublicPower.corine_population_weights import (
    burn_corine_population_weights_to_ref,
)

logger = logging.getLogger(__name__)


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def build(*, path_cfg: dict[str, Any], sector_cfg: dict[str, Any], country: str) -> dict[str, Any]:
    """
    Build the sector area proxy (weights on the reference window).

    Parameters
    ----------
    path_cfg
        Resolved path bundle (``proxy_common`` CORINE, population, NUTS; ``emissions`` CAMS NetCDF).
    sector_cfg
        Merged ``publicpower.yaml`` plus ``output_path``/``output_dir`` injected by ``PROXY.main``.
    country
        CLI ``--country`` (NUTS country / region string). Used for the **reference** window;
        also drives CAMS **ISO3** when mappable (see :func:`resolve_cams_country_iso3`).

    Returns
    -------
    ``{"output_path", "manifest_path"}`` for the GeoTIFF and its sidecar JSON.
    """
    repo_root = Path(__file__).resolve().parents[3]
    logger.info("A_PublicPower build: repo_root=%s, cli country=%r", repo_root, country)
    corine_path = _resolve_path(Path(path_cfg["proxy_common"]["corine_tif"]), repo_root)
    logger.info("CORINE (reference) raster: %s", corine_path)
    ref = reference_window_profile(
        corine_path=corine_path,
        nuts_gpkg=_resolve_path(Path(path_cfg["proxy_common"]["nuts_gpkg"]), repo_root),
        nuts_country=country,
        pad_m=float(sector_cfg.get("pad_m", 5000.0)),
    )
    _log_reference_grid(ref)

    area_cfg = sector_cfg.get("area_proxy") or {}
    pop_override = area_cfg.get("population_tif")
    if pop_override:
        population_path = _resolve_path(Path(str(pop_override)), repo_root)
    else:
        population_path = _resolve_path(
            Path(path_cfg["proxy_common"]["population_tif"]), repo_root
        )

    codes_raw = area_cfg.get("corine_codes", [121, 3])
    corine_codes = tuple(int(x) for x in codes_raw)
    corine_band = int(area_cfg.get("corine_band", 1))
    pop_exponent = float(area_cfg.get("pop_exponent", 1.0))
    pop_floor = float(area_cfg.get("pop_floor", 0.0))
    resampling = str(area_cfg.get("population_resampling", "bilinear"))
    if resampling not in ("bilinear", "nearest"):
        resampling = "bilinear"
    fallback = str(area_cfg.get("fallback_if_no_corine", "pop_in_cell"))
    if fallback not in ("pop_in_cell", "skip"):
        fallback = "pop_in_cell"
    show_progress = bool(area_cfg.get("show_progress", True))
    weight_model = str(area_cfg.get("weight_model", "eligibility_pop_blend")).strip()
    blend_a = float(area_cfg.get("blend_eligibility_coef", 0.7))
    blend_b = float(area_cfg.get("blend_population_coef", 0.3))

    cams_nc = _resolve_path(Path(path_cfg["emissions"]["cams_2019_nc"]), repo_root)
    iso3, iso3_source = resolve_cams_country_iso3(
        cli_country=country,
        explicit_iso3=sector_cfg.get("cams_country_iso3"),
    )
    logger.info(
        "CAMS country ISO3 for masks: %s (resolved from %s; optional YAML cams_country_iso3 was %r)",
        iso3,
        iso3_source,
        sector_cfg.get("cams_country_iso3"),
    )
    bbox = ref.get("domain_bbox_wgs84")
    bbox_t = tuple(float(x) for x in bbox) if bbox else None
    if bbox_t is not None:
        logger.info(
            "Reference domain WGS84 bbox (W,S,E,N): (%.4f, %.4f, %.4f, %.4f)",
            *bbox_t,
        )

    logger.info("Opening CAMS NetCDF: %s", cams_nc)
    ds = xr.open_dataset(cams_nc)
    try:
        _log_cams_grid(ds, cams_nc)
        logger.info("Building public-power area mask (GNFR A + area sources + %s)", iso3)
        mask = public_power_area_mask(ds, iso3)
        n_mask = int(np.count_nonzero(mask))
        n_all = int(mask.size)
        logger.info(
            "CAMS public-power area mask: %d of %d sources selected (%.2f%%)",
            n_mask,
            n_all,
            100.0 * n_mask / max(n_all, 1),
        )
        logger.info(
            "Population raster (for per-cell downscaling): %s; area_proxy weight_model=%r; "
            "corine_codes=%s band=%d resampling=%s",
            population_path,
            weight_model,
            corine_codes,
            corine_band,
            resampling,
        )
        logger.info("Burning CORINE x population shares onto reference grid…")
        # Using the formula weight = weight_eligibility * eligibility ^ (1+pop_exponent) + weight_population * population
        combined, n_weight_pixels = burn_corine_population_weights_to_ref(
            ds,
            mask,
            ref,
            corine_path=corine_path,
            population_path=population_path,
            corine_codes=corine_codes,
            corine_band=corine_band,
            pop_exponent=pop_exponent,
            pop_floor=pop_floor,
            population_resampling=resampling,  # type: ignore[arg-type]
            fallback_if_no_corine=fallback,  # type: ignore[arg-type]
            domain_bbox_wgs84=bbox_t,
            show_progress=show_progress,
            weight_model=weight_model,
            blend_eligibility_coef=blend_a,
            blend_population_coef=blend_b,
        )
        logger.info(
            "Burn complete: %d ref-grid pixels received weight (count of positive shares).",
            n_weight_pixels,
        )
    finally:
        ds.close()

    out_path = Path(sector_cfg["output_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    write_geotiff(
        path=out_path,
        array=combined,
        crs=ref["crs"],
        transform=ref["transform"],
        nodata=0.0,
        tags={
            "sector": "A_PublicPower",
            "allocation": (
                "cams_area_eligibility_pop_blend"
                if weight_model.lower().replace("-", "_")
                in ("eligibility_pop_blend", "blend")
                else "cams_area_corine_population_per_cell"
            ),
            "accumulation": "direct_ref_burn",
            "weight_model": weight_model,
        },
    )

    manifest = {
        "sector": "A_PublicPower",
        "layer": "area_source_weights",
        "output_geotiff": str(out_path),
        "cams_nc": str(cams_nc),
        "nuts_cli_country": str(country),
        "cams_country_iso3": iso3,
        "cams_country_iso3_source": iso3_source,
        "corine_path": str(corine_path),
        "population_raster": str(population_path),
        "domain_bbox_wgs84": list(bbox_t) if bbox_t else None,
        "area_proxy": {
            "corine_codes": list(corine_codes),
            "corine_band": corine_band,
            "weight_model": weight_model,
            "blend_eligibility_coef": blend_a,
            "blend_population_coef": blend_b,
            "pop_exponent": pop_exponent,
            "pop_floor": pop_floor,
            "population_resampling": resampling,
            "fallback_if_no_corine": fallback,
        },
        "n_weight_features": int(n_weight_pixels),
        "reference": {
            "crs": ref["crs"],
            "width": int(ref["width"]),
            "height": int(ref["height"]),
        },
    }
    write_json(out_path.with_suffix(".json"), manifest)
    logger.info("Wrote weights GeoTIFF: %s", out_path)
    return {"output_path": str(out_path), "manifest_path": str(out_path.with_suffix(".json"))}


def _log_reference_grid(ref: dict[str, Any]) -> None:
    """Log CRS and dimensions of the CORINE-NUTS reference window."""
    crs = ref.get("crs", "?")
    w = int(ref.get("width", 0) or 0)
    h = int(ref.get("height", 0) or 0)
    logger.info("Reference window: crs=%r width=%d height=%d (CORINE + NUTS + pad_m)", crs, w, h)


def _log_cams_grid(ds: xr.Dataset, cams_path: Path) -> None:
    """Log CAMS 1D source list size and a few key coordinates (if present)."""
    try:
        for name, label in (
            ("longitude_source", "CAMS longitudes"),
            ("latitude_source", "CAMS latitudes"),
        ):
            if name in ds:
                a = np.asarray(ds[name].values).ravel()
                logger.info(
                    "%s: %d points (file %s)",
                    label,
                    int(a.size),
                    cams_path.name,
                )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not log CAMS grid summary: %s", exc)
