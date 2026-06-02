from __future__ import annotations

from pathlib import Path

from proxy.dataset_loaders.load_eurostat_f_road import (
    RoadFuelSplitResult,
    load_road_fuel_split,
    log_road_fuel_split,
)


def compute_f_road_fuel_split(
    repo_root: Path,
    country_profile: dict[str, str],
    eurostat_config_path: Path,
    *,
    enabled: bool = True,
) -> RoadFuelSplitResult:
    return load_road_fuel_split(
        repo_root,
        country_profile,
        eurostat_config_path,
        enabled=enabled,
    )
