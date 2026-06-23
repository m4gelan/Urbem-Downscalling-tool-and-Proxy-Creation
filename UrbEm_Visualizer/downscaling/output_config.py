from __future__ import annotations


def parse_point_matching(output_cfg: dict) -> dict[str, str | None]:
    pm = output_cfg.get("point_matching")
    if isinstance(pm, dict):
        procedure = pm["procedure"]
        if procedure not in ("separate", "merged"):
            raise ValueError("output.point_matching.procedure must be separate or merged")
        if procedure == "separate":
            unmatched = pm["unmatched"]
            if unmatched not in ("keep_location", "burn_to_area"):
                raise ValueError(
                    "output.point_matching.unmatched must be keep_location or burn_to_area"
                )
            return {"procedure": procedure, "unmatched": unmatched}
        return {"procedure": procedure, "unmatched": None}
    if "layer_mode" in output_cfg:
        if output_cfg["layer_mode"] == "merged":
            return {"procedure": "merged", "unmatched": None}
        return {"procedure": "separate", "unmatched": "keep_location"}
    raise KeyError("run config missing output.point_matching")


def procedure_label(pm: dict[str, str | None]) -> str:
    if pm["procedure"] == "merged":
        return "Merged layers"
    if pm["unmatched"] == "burn_to_area":
        return "Separate layers (unmatched burned to area grid)"
    return "Separate layers (unmatched kept as point sources)"
