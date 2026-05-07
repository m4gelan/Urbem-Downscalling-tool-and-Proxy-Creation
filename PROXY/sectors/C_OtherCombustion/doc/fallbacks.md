# Fallbacks and warnings (GNFR C)

| Situation | Effect | Log |
|-----------|--------|-----|
| `paths.eurostat_xlsx` set | **Fail** (`ConfigurationError`) | — |
| `eurostat.enabled` false | `EndUseFactors.disabled_uniform()` (all multipliers 1.0) | INFO |
| Eurostat API / parse miss for `nrg_bal_s` | `alpha = 0`, residential totals unchanged vs “no split” path | WARNING |
| Eurostat `nrg_d_hhq` no positive TJ | Flat metric shares `def_res × f_res_total / 6` | WARNING |
| No 2-letter geo label for `cams_iso3` | Eurostat path skipped → uniform factors | WARNING |
| `eurostat.api.offline` true and cache missing | `ConfigurationError` at validation | — |
| GAINS row unmapped | Row skipped in **M** | (none per row) |
| EMEP EF missing for a row | `ef_kg_per_tj` returns 0 for that pollutant row | (none) |
| CORINE nodata / invalid L3 | `u*` masks zero; morphology reduces to `w_other·other` | (none) |
| Allocator `sum(U[:,p]) ≤ 0` or non-finite | Uniform shares over overlapping pixels for that pollutant window | Summary INFO line: `N/M … uniform fallback` |
| CAMS ISO3 without GAINS file | **M** zeros for that country | WARNING (unchanged) |
| `rural_bias` enabled but population/GHSL path missing | Rural bias skipped | WARNING |

Startup **validation** (`C_OtherCombustion.validation.validate_pipeline_config`) can **raise** `ConfigurationError` when GAINS mapping does not reference every `MODEL_CLASSES` value, `class_to_metric` omits any `R_*` class, or standard EMEP probes return zero for a configured pollutant (except `co2_total`).
