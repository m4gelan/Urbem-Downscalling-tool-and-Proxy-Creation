"""Dataset loaders used across sector pipelines."""

from .config import PathConfig, load_path_config, load_yaml, project_root, resolve_path
from .config_candidates import load_first_existing_yaml_or_json
from .emissions import dataset_summary, open_cams_dataset
from .raster import (
    assert_same_grid,
    raster_metadata,
    read_band,
    ref_profile_to_kwargs,
    warp_band_to_ref,
    warp_raster_to_ref,
)
from .tabular import read_csv, read_excel
from .vector import read_country_nuts2, read_vector

__all__ = [
    "PathConfig",
    "assert_same_grid",
    "dataset_summary",
    "load_path_config",
    "load_yaml",
    "load_first_existing_yaml_or_json",
    "open_cams_dataset",
    "project_root",
    "raster_metadata",
    "read_band",
    "ref_profile_to_kwargs",
    "read_country_nuts2",
    "read_csv",
    "read_excel",
    "read_vector",
    "resolve_path",
    "warp_band_to_ref",
    "warp_raster_to_ref",
]

