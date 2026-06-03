from __future__ import annotations

from typing import Any

from .predicates import match_tags


def _offroad_sets_frozen(sector: dict[str, Any]) -> dict[str, frozenset[str]]:
    """Build frozenset lookup tables from sector offroad_sets YAML."""
    raw = sector.get("offroad_sets") or {}
    return {k: frozenset(str(x) for x in v) for k, v in raw.items()}


def _match_with_ctx(
    tags: dict[str, str],
    spec: dict[str, Any] | None,
    ctx: dict[str, Any],
) -> bool:
    """Evaluate a match spec, including in_set references against ctx offroad_sets."""
    if spec is None:
        return True
    if not isinstance(spec, dict):
        return False
    if "in_set" in spec:
        key = str(spec["in_set"])
        sets = ctx.get("offroad_sets") or {}
        field = str(spec.get("tag_key", ""))
        val = tags.get(field)
        return val in sets.get(key, frozenset())
    if "tag" in spec and isinstance(spec["tag"], dict):
        t = dict(spec["tag"])
        if "in_set" in t:
            sets = ctx.get("offroad_sets") or {}
            val = tags.get(str(t["key"]))
            return val in sets.get(str(t["in_set"]), frozenset())
    return match_tags(tags, spec)


def _format_value(template: str, tags: dict[str, str], row: dict[str, Any]) -> str:
    """Expand {tag_key} placeholders from tags and row."""
    merged = {**tags, **{k: v for k, v in row.items() if isinstance(v, str)}}
    try:
        return template.format(**merged)
    except KeyError:
        return template


def _first_match_value(
    tags: dict[str, str],
    rules: list[dict[str, Any]],
    ctx: dict[str, Any],
    row: dict[str, Any],
    *,
    default: str,
    default_template: str | None = None,
) -> str:
    """Return value from first matching rule in a first_match list."""
    for r in rules:
        if _match_with_ctx(tags, r.get("match"), ctx):
            if "value_template" in r:
                return _format_value(str(r["value_template"]), tags, row)
            return str(r["value"])
    if default_template:
        return _format_value(default_template, tags, row)
    return default


def _multi_match_values(
    tags: dict[str, str],
    rules: list[dict[str, Any]],
    ctx: dict[str, Any],
) -> str:
    """Return comma-separated sorted values for all matching multi_match rules."""
    fam: set[str] = set()
    for r in rules:
        if _match_with_ctx(tags, r.get("match"), ctx):
            fam.add(str(r["value"]))
    return ",".join(sorted(fam))


def apply_row_augment(
    sector: dict[str, Any],
    tags: dict[str, str],
    row: dict[str, Any],
    ctx: dict[str, Any],
) -> None:
    """Apply sector augment.columns definitions to a feature row in place."""
    aug = sector.get("augment")
    if not isinstance(aug, dict):
        return
    columns = aug.get("columns")
    if isinstance(columns, dict):
        offroad_sets = _offroad_sets_frozen(sector)
        actx = dict(ctx)
        if offroad_sets:
            actx["offroad_sets"] = offroad_sets

        for col, spec in columns.items():
            if not isinstance(spec, dict):
                continue
            if "first_match" in spec:
                default = str(spec.get("default", "other"))
                dt = spec.get("default_template")
                row[col] = _first_match_value(
                    tags, spec["first_match"], actx, row, default=default, default_template=dt
                )
            elif "multi_match" in spec:
                row[col] = _multi_match_values(tags, spec["multi_match"], actx)

    copy_tags = aug.get("copy_tags")
    if isinstance(copy_tags, list):
        for k in copy_tags:
            if k not in row:
                row[k] = tags.get(k)
