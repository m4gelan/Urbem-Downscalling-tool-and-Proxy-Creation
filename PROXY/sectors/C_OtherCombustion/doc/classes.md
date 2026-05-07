# Model classes (GNFR C)

| Class | Description (inventory sense) | Proxy in **X** | End-use bucket **e(k)** | GAINS appliance axis | CORINE morphology |
|-------|----------------------------------|----------------|-------------------------|----------------------|---------------------|
| R_FIREPLACE | Residential fireplaces | `R_base × mr × rural_bias?` | `space_heating` (Eurostat API) | Fireplaces (households / agr…) | `mr` (111/112/other) |
| R_HEATING_STOVE | Residential heating stoves | `R_base × mr × rural_bias?` | `space_heating` | Heating stoves | `mr` |
| R_COOKING_STOVE | Residential cooking | `R_base` | `cooking` | Cooking stoves | none (`mr` not applied) |
| R_BOILER_MAN | Residential manual boilers | `R_base` | mean(`space_heating`,`water_heating`) | Small household boilers | none |
| R_BOILER_AUT | Residential automatic boilers | `R_base` | mean(`space_heating`,`water_heating`) | Small household boilers | none |
| C_BOILER_MAN | Commercial / institutional manual boilers | `C_base × mc` | `commercial` (share from `nrg_bal_s`) | Non-residential boiler rules | `mc` (111/121/other) |
| C_BOILER_AUT | Commercial / institutional automatic boilers | `C_base × mc` | `commercial` | Non-residential boiler rules | `mc` |

`rural_bias` (optional, YAML `rural_bias.enabled`) applies only to **R_FIREPLACE** and **R_HEATING_STOVE** by default.

## Why `R_COOKING_STOVE` / `R_BOILER_*` do **not** use `mr`

Cooking and residential boilers are allocated using **residential heat + GFA** (`R_base`) only. The extra morphology factor `mr` concentrates **fireplaces and heating stoves** toward continuous / discontinuous urban fabric (CLC 111 / 112), which matches the spatial narrative for those appliances. Cooking and central heating are driven primarily by building stock and heat demand density from Hotmaps rather than the same urban-fabric contrast, so they intentionally skip `mr` to avoid over-constraining those bands.

## Naming note

Eurostat **household** shares and GAINS **appliance splits** are exposed separately in logs as `f_enduse_by_bucket` and `f_appliance_by_class`. The diagnostic product per class is available as `activity_share_by_class(factors)` (deprecated alias for the combined multipliers used in older docs).
