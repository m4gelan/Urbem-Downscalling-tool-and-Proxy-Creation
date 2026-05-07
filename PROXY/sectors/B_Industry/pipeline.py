"""
End-to-end **GNFR B** industry area weights (thin layer over the shared group pipeline).

``run_industry_pipeline`` is the sector-specific name for
:func:`PROXY.sectors._shared.gnfr_groups.run_gnfr_group_pipeline` with
``sector_key="B_Industry"`` and a multi-layer **industry** OSM GeoPackage reader.

Why a separate module: keeps `builder.py` free of the large shared implementation,
and gives a single import target for tests or ad-hoc runs that already have
``industry_cfg`` and ``ref`` built.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PROXY.sectors._shared.gnfr_groups import (
    load_industry_osm_all_layers,
    run_gnfr_group_pipeline,
)

logger = logging.getLogger(__name__)


def _read_industry_gpkg_all_layers(path: Path) -> Any:
    """Public alias to :func:`load_industry_osm_all_layers` for monkeypatching in tests."""
    return load_industry_osm_all_layers(path)


def run_industry_pipeline(
    root: Path,
    industry_cfg: dict[str, Any],
    ref: dict[str, Any],
    *,
    country_iso3_fallback: str | None = None,
    show_progress: bool = True,
) -> Path:
    """
    Run the B_Industry spatial + CEIP α pipeline and return the path to the weights TIF.

    Parameters
    ----------
    industry_cfg
        Output of :func:`PROXY.sectors._shared.gnfr_groups.merge_ceip_group_sector_cfg`
        for this sector (with ``osm_industry_gpkg`` set by the builder).
    ref
        Reference window profile from :func:`PROXY.core.grid.reference_window_profile`.
    country_iso3_fallback
        ISO-3 for CEIP / NUTS **pixel** α rows when a grid cell does not map to a
        country id (default ``GRC`` if not provided; often mirrored from
        ``cams_country_iso3`` in ``industry.yaml``).
    show_progress
        Log major stages (CAMS ids, warps, OSM load) when true.

    See also
    --------
    :func:`PROXY.sectors._shared.gnfr_groups.run_gnfr_group_pipeline` — full stage list.
    """
    return run_gnfr_group_pipeline(
        root=root,
        cfg=industry_cfg,
        ref=ref,
        sector_key="B_Industry",
        output_prefix="industry",
        country_iso3_fallback=str(country_iso3_fallback or "GRC").strip().upper(),
        show_progress=show_progress,
        logger=logger,
        diag_tag="industry-debug",
        osm_loader=_read_industry_gpkg_all_layers,
    )
