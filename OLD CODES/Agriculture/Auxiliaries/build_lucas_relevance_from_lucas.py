"""Export NUTS-2 x CLC mu/rho for grazing, manure, and synthetic N (each pathway in source_relevance/*)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Agriculture.config import default_config_path, load_agriculture_config, project_root
from Agriculture.core.io import resolve_path
from Agriculture.source_relevance.common import aggregate_nuts_clc_mu
from Agriculture.source_relevance.enteric_grazing import point_grazing_metric
from Agriculture.source_relevance.fertilized_land import load_synthetic_n_rates_json, point_synth_n_rate
from Agriculture.source_relevance.lucas_points import get_lucas_ag_points
from Agriculture.source_relevance.manure import point_manure_metric


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=None, help="agriculture.config.json path")
    args = p.parse_args()

    cfg_path = args.config or default_config_path()
    cfg = load_agriculture_config(cfg_path)
    root = project_root(cfg)
    lb = cfg.get("lucas_build") or {}
    out_rel = lb.get("output_csv") or "Agriculture/results/lucas_nuts2_clc_relevance.csv"
    out = resolve_path(root, out_rel)

    pts = get_lucas_ag_points(cfg, root).copy()
    pts["mu_graze"] = [
        point_grazing_metric(g, lc) for g, lc in zip(pts["SURVEY_GRAZING"], pts["SURVEY_LC1"])
    ]
    pts["mu_manure"] = [
        point_manure_metric(g, lc, lu)
        for g, lc, lu in zip(pts["SURVEY_GRAZING"], pts["SURVEY_LC1"], pts["SURVEY_LU1"])
    ]
    rate_rel = lb.get("synthetic_n_rate_json") or "Agriculture/config/synthetic_N_rate.json"
    rates = load_synthetic_n_rates_json(resolve_path(root, rate_rel))
    pts["mu_synth_n"] = [
        point_synth_n_rate(lc, lu, rates) for lc, lu in zip(pts["SURVEY_LC1"], pts["SURVEY_LU1"])
    ]

    g1 = aggregate_nuts_clc_mu(pts, "mu_graze").rename(
        columns={"mu": "mu_graze", "rho": "relevance_ch4_graze"}
    )
    g2 = aggregate_nuts_clc_mu(pts, "mu_manure").rename(columns={"mu": "mu_manure", "rho": "relevance_manure"})
    g3 = aggregate_nuts_clc_mu(pts, "mu_synth_n").rename(
        columns={"mu": "mu_synth_n", "rho": "relevance_synth_n"}
    )
    key = ["NUTS_ID", "CLC_CODE", "COUNTRY"]
    merged = g1.merge(g2[key + ["mu_manure", "relevance_manure"]], on=key)
    merged = merged.merge(g3[key + ["mu_synth_n", "relevance_synth_n"]], on=key)

    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False, float_format="%.8f", na_rep="")
    print(f"Wrote {out} ({len(merged)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
