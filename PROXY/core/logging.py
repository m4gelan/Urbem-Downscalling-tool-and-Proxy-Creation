"""Structured logging helper with sector / country / pollutant context.

Usage::

    from PROXY.core.logging import get_logger

    log = get_logger("K_Agriculture", country="EL")
    log.info("loading CLC raster", extra={"path": str(p)})

The returned ``logging.LoggerAdapter`` prefixes every message with the context so grep
over build logs is cheap and greppable. This module does not configure the root logger
(callers remain in charge of handlers / levels).
"""
from __future__ import annotations

import logging
from typing import Any


class _ContextAdapter(logging.LoggerAdapter):
    def process(self, msg: Any, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        ctx = self.extra or {}
        prefix_parts: list[str] = []
        for key in ("sector", "country", "pollutant"):
            v = ctx.get(key)
            if v is not None:
                prefix_parts.append(f"{key}={v}")
        prefix = " ".join(prefix_parts)
        if prefix:
            msg = f"[{prefix}] {msg}"
        return msg, kwargs


def get_logger(
    sector: str,
    *,
    country: str | None = None,
    pollutant: str | None = None,
    name: str | None = None,
) -> logging.LoggerAdapter:
    """Return a :class:`logging.LoggerAdapter` that prefixes messages with context."""
    logger_name = name or f"PROXY.{sector}"
    base = logging.getLogger(logger_name)
    extra: dict[str, Any] = {"sector": sector}
    if country is not None:
        extra["country"] = country
    if pollutant is not None:
        extra["pollutant"] = pollutant
    return _ContextAdapter(base, extra)
