# PROXY changelog

## 2026-05-08

- **GNFR C (`C_OtherCombustion`)**: Stationary **X** can use the seven-band **S×L appliance proxy** (POP × morphology + non-res heat × commercial morphology, times heat/HDD load shapes) driven by `appliance_proxy` in `PROXY/config/ceip/profiles/C_OtherCombustion_rules.yaml`. Sector flag `appliance_proxy.enabled` in `othercombustion.yaml` (default true) toggles vs legacy Hotmaps×GFA×morphology; requires Hotmaps `HDD_curr`, `paths.population_tif`, and `paths.ghsl_smod_tif`.

- **GNFR C (`C_OtherCombustion`)**: Optional two-branch downscaling — CEIP α splits stationary (1A4ai/bi/ci) vs off-road (1A4aii/bii/cii), combined with forestry / residential / commercial spatial proxies from `C_OtherCombustion_rules.yaml`. CLI: `--enable-offroad` / `--no-enable-offroad` (default on). Legacy behaviour: `--no-enable-offroad`. Shared eligibility–population blend extracted to `PROXY/core/proxy/eligibility_pop_blend.py` (PublicPower unchanged numerically).
