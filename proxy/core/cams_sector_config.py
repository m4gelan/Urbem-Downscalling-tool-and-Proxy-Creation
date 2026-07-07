from __future__ import annotations

from pathlib import Path

from typing import Any


def cams_block(cfg: dict, key: str) -> dict[str, Any]:
    block = cfg.get(key)
    if not isinstance(block, dict):
        raise KeyError(f"sector config missing {key!r}")
    for req in ("year", "emission_category_indices", "source_type_indices"):
        if req not in block:
            raise KeyError(f"{key} missing {req!r}")
    return block


def cams_sector_cells(cfg: dict) -> dict[str, Any]:
    return cams_block(cfg, "cams_sector_cells")


def cams_area_emissions(cfg: dict) -> dict[str, Any]:
    return cams_block(cfg, "cams_area_emissions")


def load_shipping_sector_cells_mask(
    cams_nc,
    cfg: dict,
    *,
    country_profile: dict[str, str],
    country_iso3: str,
    pollutants: list[str],
    nuts_path: Path,
    crs: str,
    resolution_m: float,
    pad_m: float,
):
    from proxy.dataset_loaders.load_cams_cells_mask import load_cams_shipping_cells_mask

    cells = cams_sector_cells(cfg)
    md = cfg.get("maritime_domain")
    if not isinstance(md, dict):
        raise ValueError("G_Shipping sector config: set 'maritime_domain' with buffer_m and metric_crs")
    return load_cams_shipping_cells_mask(
        cams_nc,
        year=int(cells["year"]),
        country_iso3=country_iso3,
        country_profile=country_profile,
        nuts_path=nuts_path,
        maritime_buffer_m=float(md["buffer_m"]),
        maritime_metric_crs=str(md["metric_crs"]),
        emission_category_indices=list(cells["emission_category_indices"]),
        source_type_indices=list(cells["source_type_indices"]),
        pollutants=pollutants,
        crs=crs,
        resolution_m=resolution_m,
        pad_m=pad_m,
    )


def load_sector_cells_mask(
    cams_nc,
    cfg: dict,
    *,
    country_iso3: str,
    pollutants: list[str],
    crs: str,
    resolution_m: float,
    pad_m: float,
):
    from proxy.dataset_loaders.load_cams_cells_mask import load_cams_cells_mask

    cells = cams_sector_cells(cfg)
    mass = cams_area_emissions(cfg)
    return load_cams_cells_mask(
        cams_nc,
        year=int(cells["year"]),
        country_iso3=country_iso3,
        emission_category_indices=list(cells["emission_category_indices"]),
        source_type_indices=list(cells["source_type_indices"]),
        mass_source_type_indices=list(mass["source_type_indices"]),
        pollutants=pollutants,
        crs=crs,
        resolution_m=resolution_m,
        pad_m=pad_m,
    )
