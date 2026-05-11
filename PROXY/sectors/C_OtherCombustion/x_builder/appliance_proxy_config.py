"""
Load and validate ``appliance_proxy`` from CEIP rules YAML (GNFR C stationary X stack).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..constants import MODEL_CLASSES
from ..exceptions import ConfigurationError


def _default_appliance_proxy_doc() -> dict[str, Any]:
    """Defaults matching PROXY/config/ceip/profiles/C_OtherCombustion_rules.yaml."""
    return {
        "ghs_rural_classes": [11, 12, 13],
        "epsilon": 1.0e-12,
        "R_FIREPLACE": {
            "stock": {
                "carrier": "POP",
                "corine_weights": {"111": 0.2, "112": 1.0, "rural_res": 0.6},
            },
            "load": {
                "type": "product",
                "terms": [{"name": "H_RES", "exponent": 0.6}, {"name": "HDD", "exponent": 0.4}],
            },
        },
        "R_HEATING_STOVE": {
            "stock": {
                "carrier": "POP",
                "corine_weights": {"111": 0.2, "112": 1.0, "rural_res": 0.6},
            },
            "load": {
                "type": "product",
                "terms": [{"name": "H_RES", "exponent": 0.6}, {"name": "HDD", "exponent": 0.4}],
            },
        },
        "R_COOKING_STOVE": {
            "stock": {
                "carrier": "POP",
                "corine_weights": {"111": 1.0, "112": 1.0, "rural_res": 0.8},
            },
            "load": {"type": "constant", "value": 1.0},
        },
        "R_BOILER_MAN": {
            "stock": {
                "carrier": "POP",
                "corine_weights": {"111": 0.3, "112": 1.0, "rural_res": 0.9},
            },
            "load": {"type": "variable", "name": "H_RES", "exponent": 1.0},
        },
        "R_BOILER_AUT": {
            "stock": {
                "carrier": "POP",
                "corine_weights": {"111": 1.0, "112": 1.1, "rural_res": 0.4},
            },
            "load": {"type": "variable", "name": "H_RES", "exponent": 1.0},
        },
        "C_BOILER_MAN": {
            "stock": {
                "carrier": "H_NRES",
                "corine_weights": {"111": 1.0, "112": 0.7, "121": 1.5},
                "commercial_other_weight": 0.3,
            },
            "load": {
                "type": "product",
                "terms": [{"name": "H_NRES", "exponent": 0.7}, {"name": "HDD", "exponent": 0.3}],
            },
        },
        "C_BOILER_AUT": {
            "stock": {
                "carrier": "H_NRES",
                "corine_weights": {"111": 1.0, "112": 0.7, "121": 1.5},
                "commercial_other_weight": 0.3,
            },
            "load": {"type": "variable", "name": "H_NRES", "exponent": 1.0},
        },
    }


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_appliance_proxy_from_rules_yaml(rules_path: Path) -> dict[str, Any]:
    if not rules_path.is_file():
        raise FileNotFoundError(f"CEIP rules YAML not found: {rules_path}")
    doc = yaml.safe_load(rules_path.read_text(encoding="utf-8")) or {}
    block = doc.get("appliance_proxy")
    if not isinstance(block, dict):
        return _default_appliance_proxy_doc()
    merged = _deep_merge(_default_appliance_proxy_doc(), block)
    validate_appliance_proxy_doc(merged)
    return merged


def validate_appliance_proxy_doc(block: dict[str, Any]) -> None:
    eps = float(block.get("epsilon", 1.0e-12))
    if not (eps > 0.0 and eps < 1.0):
        raise ConfigurationError(f"appliance_proxy.epsilon must be in (0,1), got {eps!r}")

    ghs = block.get("ghs_rural_classes")
    if not isinstance(ghs, list) or not ghs:
        raise ConfigurationError("appliance_proxy.ghs_rural_classes must be a non-empty list of ints")
    for x in ghs:
        if not isinstance(x, int):
            raise ConfigurationError(f"appliance_proxy.ghs_rural_classes must be ints, got {x!r}")

    for cls in MODEL_CLASSES:
        if cls not in block:
            raise ConfigurationError(f"appliance_proxy missing MODEL_CLASSES entry: {cls!r}")
        spec = block[cls]
        if not isinstance(spec, dict):
            raise ConfigurationError(f"appliance_proxy.{cls} must be a mapping")
        _validate_stock(cls, spec.get("stock") or {})
        _validate_load(cls, spec.get("load") or {})


def _validate_stock(cls: str, stock: dict[str, Any]) -> None:
    carrier = str(stock.get("carrier", "")).upper()
    if carrier not in ("POP", "H_NRES"):
        raise ConfigurationError(f"{cls}.stock.carrier must be POP or H_NRES, got {carrier!r}")
    if cls.startswith("C_") and carrier != "H_NRES":
        raise ConfigurationError(f"{cls} commercial stock must use carrier H_NRES, got {carrier!r}")
    if cls.startswith("R_") and carrier != "POP":
        raise ConfigurationError(f"{cls} residential stock must use carrier POP, got {carrier!r}")

    cw = stock.get("corine_weights")
    if not isinstance(cw, dict) or not cw:
        raise ConfigurationError(f"{cls}.stock.corine_weights must be a non-empty map")
    for k, v in cw.items():
        key = str(k)
        if key not in ("111", "112", "121", "rural_res"):
            raise ConfigurationError(f"{cls}.stock.corine_weights invalid key {key!r}")
        fv = float(v)
        if fv < 0.0:
            raise ConfigurationError(f"{cls}.stock.corine_weights[{key!r}] must be >= 0")

    cow = stock.get("commercial_other_weight")
    if cow is not None:
        fv = float(cow)
        if fv < 0.0:
            raise ConfigurationError(f"{cls}.stock.commercial_other_weight must be >= 0")


def _validate_load(cls: str, load: dict[str, Any]) -> None:
    t = str(load.get("type", "")).lower()
    if t not in ("constant", "variable", "product"):
        raise ConfigurationError(f"{cls}.load.type must be constant|variable|product, got {t!r}")
    if t == "constant":
        if "value" not in load:
            raise ConfigurationError(f"{cls}.load constant requires value")
        float(load["value"])
    elif t == "variable":
        name = str(load.get("name", "")).upper()
        if name not in ("H_RES", "H_NRES", "HDD"):
            raise ConfigurationError(f"{cls}.load.variable name must be H_RES|H_NRES|HDD, got {name!r}")
        if "exponent" in load:
            float(load["exponent"])
    else:
        terms = load.get("terms")
        if not isinstance(terms, list) or not terms:
            raise ConfigurationError(f"{cls}.load.product requires non-empty terms list")
        for term in terms:
            if not isinstance(term, dict):
                raise ConfigurationError(f"{cls}.load.product terms must be mappings")
            name = str(term.get("name", "")).upper()
            if name not in ("H_RES", "H_NRES", "HDD"):
                raise ConfigurationError(f"{cls}.load.product term name invalid: {name!r}")
            float(term["exponent"])
