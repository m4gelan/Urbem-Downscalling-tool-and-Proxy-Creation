from __future__ import annotations

import re
from typing import Any

# Cache compiled regex patterns to avoid recompiling on every match
_REGEX_CACHE: dict[str, re.Pattern] = {}


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
        # Use cached compiled pattern
        if pat not in _REGEX_CACHE:
            _REGEX_CACHE[pat] = re.compile(pat, re.I)
        return bool(_REGEX_CACHE[pat].search(tags.get("name") or ""))
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


def extract_tag_keys_from_match(match_spec: dict[str, Any] | None) -> set[str]:
    """Collect tag keys referenced in a match spec (for rule indexing)."""
    if not match_spec or not isinstance(match_spec, dict):
        return set()
    keys: set[str] = set()
    if "tag" in match_spec:
        t = match_spec["tag"]
        if isinstance(t, dict) and "key" in t:
            keys.add(str(t["key"]))
    if "all_of" in match_spec:
        for sub in match_spec["all_of"]:
            keys.update(extract_tag_keys_from_match(sub))
    if "any_of" in match_spec:
        for sub in match_spec["any_of"]:
            keys.update(extract_tag_keys_from_match(sub))
    if "not" in match_spec:
        keys.update(extract_tag_keys_from_match(match_spec["not"]))
    return keys


def build_rule_index(rules: list[dict[str, Any]]) -> dict[str, set[int]]:
    """Map tag keys to rule indices that reference them."""
    index: dict[str, set[int]] = {}
    for i, rule in enumerate(rules):
        for key in extract_tag_keys_from_match(rule.get("match")):
            index.setdefault(key, set()).add(i)
    return index


def first_matching_rule(
    tags: dict[str, str],
    rules: list[dict[str, Any]],
    rule_index: dict[str, set[int]] | None = None,
) -> dict[str, Any] | None:
    """Return the first rule whose match spec fits tags."""
    if rule_index:
        candidates: set[int] = set()
        for key in tags:
            candidates.update(rule_index.get(key, ()))
        if not candidates:
            return None
        for i, rule in enumerate(rules):
            if i in candidates and match_tags(tags, rule.get("match")):
                return rule
        return None
    for rule in rules:
        if match_tags(tags, rule.get("match")):
            return rule
    return None


def matching_rules(
    tags: dict[str, str],
    rules: list[dict[str, Any]],
    rule_index: dict[str, set[int]] | None = None,
) -> list[dict[str, Any]]:
    """Return all rules whose match spec fits tags."""
    if rule_index:
        candidates: set[int] = set()
        for key in tags:
            candidates.update(rule_index.get(key, ()))
        if not candidates:
            return []
        return [rules[i] for i in sorted(candidates) if match_tags(tags, rules[i].get("match"))]
    return [r for r in rules if match_tags(tags, r.get("match"))]
