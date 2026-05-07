"""YAML ``match`` → tag predicate (all_of / any_of / tag)."""

from __future__ import annotations

from typing import Any


def match_tags(tags: dict[str, str], spec: dict[str, Any] | None) -> bool:
    if spec is None or spec is False:
        return True
    if not isinstance(spec, dict):
        raise TypeError(f"match spec must be a dict, got {type(spec)}")

    if "all_of" in spec:
        return all(match_tags(tags, s) for s in spec["all_of"])
    if "any_of" in spec:
        return any(match_tags(tags, s) for s in spec["any_of"])
    if "not" in spec:
        return not match_tags(tags, spec["not"])
    if "tag" in spec:
        t = spec["tag"]
        if not isinstance(t, dict):
            raise TypeError("tag: must be a dict")
        key = str(t["key"])
        val = tags.get(key)
        if "equals" in t:
            return val == t["equals"]
        if "in" in t:
            return val in t["in"]
        if t.get("exists") is True:
            return val is not None and str(val) != ""
        if t.get("exists") is False:
            return val is None or str(val) == ""
        raise ValueError(f"tag: needs equals, in, or exists: {t}")

    raise ValueError(f"Unknown match spec keys: {list(spec)}")
