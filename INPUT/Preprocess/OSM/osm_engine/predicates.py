from __future__ import annotations

import re
from typing import Any


def match_tags(tags: dict[str, str], spec: dict[str, Any] | None) -> bool:
    """Evaluate a YAML match spec (all_of / any_of / not / tag / name_regex) against OSM tags."""
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
    if "name_regex" in spec:
        pat = str(spec["name_regex"])
        return bool(re.search(pat, tags.get("name") or "", re.I))
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


def match_classify_rule(
    tags: dict[str, str],
    elem_kind: str,
    rule: dict[str, Any],
) -> bool:
    """Return True if a classify rule matches tags and element kind."""
    kinds = rule.get("elem_kind_in")
    if kinds is not None and elem_kind not in kinds:
        return False
    return match_tags(tags, rule.get("match"))
