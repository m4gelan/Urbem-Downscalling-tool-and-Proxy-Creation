"""
End-to-end **GNFR D** fugitive area weights (thin layer over the shared group pipeline).

``run_fugitive_pipeline`` is the sector-specific name for
:func:`PROXY.sectors._shared.gnfr_groups.run_gnfr_group_pipeline` with
``sector_key="D_Fugitive"`` and the **default** OSM GeoPackage reader from
``gnfr_groups`` (single-layer read; industry uses a multi-layer loader).

Why a separate module: same pattern as ``B_Industry.pipeline`` — keeps ``builder.py``
focused on path merge + reference grid, and gives a single import target for tests
or ad-hoc runs that already have ``fugitive_cfg`` and ``ref`` built.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PROXY.sectors._shared.gnfr_groups import run_gnfr_group_pipeline

logger = logging.getLogger(__name__)


def run_fugitive_pipeline(
    root: Path,
    fugitive_cfg: dict[str, Any],
    ref: dict[str, Any],
    *,
    country_iso3_fallback: str | None = None,
    show_progress: bool = True,
) -> Path:
    """
    Run the D_Fugitive spatial + CEIP α pipeline and return the path to the weights TIF.

    Parameters
    ----------
    root
        Project root (``project_root()`` from the builder).
    fugitive_cfg
        Output of :func:`PROXY.sectors._shared.gnfr_groups.merge_ceip_group_sector_cfg`
        for this sector (with ``osm_fugitive_gpkg`` set by the builder).
    ref
        Reference window profile from :func:`PROXY.core.grid.reference_window_profile`.
    country_iso3_fallback
        ISO-3 for CEIP / NUTS **pixel** α rows when a grid cell does not map to a
        country id (default ``GRC`` if not provided; often mirrored from
        ``cams_country_iso3`` in ``fugitive.yaml``).
    show_progress
        Log major stages when true (same contract as ``B_Industry``).

    Returns
    -------
    Path to the written multiband GeoTIFF (same as ``cfg`` output path).

    See also
    --------
    :func:`PROXY.sectors._shared.gnfr_groups.run_gnfr_group_pipeline` — full stage list.
    """
    return run_gnfr_group_pipeline(
        root=root,
        cfg=fugitive_cfg,
        ref=ref,
        sector_key="D_Fugitive",
        output_prefix="fugitive",
        country_iso3_fallback=str(country_iso3_fallback or "GRC").strip().upper(),
        show_progress=show_progress,
        logger=logger,
        diag_tag="fugitive-debug",
    )
