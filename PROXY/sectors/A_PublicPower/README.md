# A_PublicPower (area proxy)

GNFR **A** (public power) **area** emissions are distributed using a per-CAMS-cell
proxy: **CORINE** eligibility (industrial / commercial type classes) combined with
**JRC population** (see `area_proxy.weight_model` in `publicpower.yaml`). Shares are
**burned** onto the same reference grid as the rest of the PROXY stack (aligned with
`proxy_common.corine_tif` and NUTS window).

## Entry point

The CLI loads `PROXY.sectors.A_PublicPower.builder` and calls `build(path_cfg=..., sector_cfg=..., country=...)`:

```text
python -m PROXY.main build --sector A_PublicPower --country EL
```

* **Config file**: `PROXY/config/sectors/publicpower.yaml` (path listed in `PROXY/config/sectors.yaml`).

* **Output**: path from `output_dir` + `output_filename` in that YAML: a single-band
  GeoTIFF of non-negative weights (nodata 0) plus a **manifest** JSON with the
  same basename.

## Country vs CAMS country (ISO3)

* **NUTS / reference grid** uses `country` (CLI `--country`, e.g. `EL` or a NUTS2
  code like `EL30`) in `reference_window_profile` so the output raster aligns with
  the same window as other sectors for that run.

* **CAMS country** for raster masks uses **ISO-3166-1 alpha-3** (e.g. `GRC`).
  The builder resolves it with `PROXY.core.raster.country_clip.resolve_cams_country_iso3`:

  1. Prefer **derivation** from `--country` (e.g. `EL*`, `EL` -> `GRC`; `DE` -> `DEU`).
  2. If that fails, use optional `cams_country_iso3` in `publicpower.yaml`.
  3. If still unknown, default `GRC`.

  The manifest records `cams_country_iso3` and `cams_country_iso3_source` for traceability.

## What is *not* in `build()`

`cams_emission_category_indices` / `cams_source_type_indices` in the sector YAML
target the **point-matching** path (`PROXY.core.matching`), not this area
`build()`. The area mask is implemented in `cams_area_mask.py` (GNFR A, domain,
**area** sources: `source_type_index == 1`).

## Module map

| Module | Role |
|--------|------|
| `builder.py` | Resolves paths, reference grid, CAMS mask + CORINE/pop burn, writes GeoTIFF + manifest. |
| `cams_area_mask.py` | Boolean mask: CAMS sources that are GNFR A + country + **area** type. |
| `corine_population_weights.py` | Per selected CAMS cell: clip CORINE, resample population, per-pixel weight model, accumulate shares on ref grid. |

## Further reading

Project methodology: `PROXY/docs/sector_methodology.md` (section A_PublicPower).
