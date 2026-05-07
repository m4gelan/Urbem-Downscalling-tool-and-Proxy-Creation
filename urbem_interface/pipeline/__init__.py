"""Proxy pipeline helpers: catalog, merge defaults, text report."""

from urbem_interface.pipeline.catalog import (
    DEFAULT_AUXILIARY_PROXIES,
    ProxyCatalogEntry,
    ProxyPipelineBundle,
    build_proxy_pipeline_bundle,
    merge_proxy_catalog,
    validate_proxy_files,
)
DESIGN_OPTIONS_SUMMARY = (
    "Declarative proxy configuration: urbem_interface/config/proxy_pipeline.json (pipeline_schema 1). "
    "Run: python -m urbem_interface.pipeline --proxies-json path/to/proxies.json [--proxies-folder DIR]"
)
from urbem_interface.pipeline.job_config import (
    get_factory_inputs,
    get_proxy_definitions,
    load_raw_pipeline,
    materialize_downscaling_config,
    materialize_proxies_config_file,
)
from urbem_interface.pipeline.report import (
    render_proxy_pipeline_report,
    report_from_proxies_json,
)

__all__ = [
    "DEFAULT_AUXILIARY_PROXIES",
    "DESIGN_OPTIONS_SUMMARY",
    "ProxyCatalogEntry",
    "ProxyPipelineBundle",
    "build_proxy_pipeline_bundle",
    "merge_proxy_catalog",
    "validate_proxy_files",
    "render_proxy_pipeline_report",
    "report_from_proxies_json",
    "get_factory_inputs",
    "get_proxy_definitions",
    "load_raw_pipeline",
    "materialize_downscaling_config",
    "materialize_proxies_config_file",
]
