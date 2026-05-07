from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .constants import MODEL_CLASSES


def _resolve(root: Path, p: str | Path) -> Path:
    x = Path(p)
    return x if x.is_absolute() else (root / x)


def _cell_str(v: Any) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    return str(v).strip()


def _parse_share_cell(v: Any) -> float | None:
    s = _cell_str(v)
    if not s or s.lower() in {"..", ":", "-", "nan"}:
        return None
    s = s.replace(",", ".").replace("\xa0", " ")
    m = re.search(r"(-?\d+(?:\.\d+)?)", s)
    if not m:
        return None
    try:
        x = float(m.group(1))
    except ValueError:
        return None
    if x > 1.5:
        return x / 100.0
    return x


def _norm_key(s: str) -> str:
    s = (
        str(s)
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\xa0", " ")
    )
    return re.sub(r"\s+", " ", s.strip().lower())


def _geo_row_match(cell_norm: str, geo_labels: list[str]) -> bool:
    if not cell_norm:
        return False
    gln = {_norm_key(g) for g in geo_labels}
    if cell_norm in gln:
        return True
    if cell_norm == "el" and any(_norm_key(g) == "el" for g in geo_labels):
        return True
    for g in geo_labels:
        gn = _norm_key(g)
        if len(gn) >= 3 and (gn in cell_norm or cell_norm in gn):
            return True
    return False


def _text_matches_synonym(label: str, synonym: str) -> bool:
    """Row/product label vs config synonym (handles long Eurostat strings)."""
    a = _norm_key(label)
    b = _norm_key(synonym)
    if not a or not b:
        return False
    return b in a or a in b


def _load_eurostat_sidecar(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _geo_token_matches_cell(cell_norm: str, geo_norm: set[str]) -> bool:
    """
    Match a spreadsheet cell to configured country labels without false positives
    for short Eurostat codes (e.g. EL must not match inside 'label').
    """
    if not cell_norm:
        return False
    if cell_norm in geo_norm:
        return True
    for g in geo_norm:
        if not g:
            continue
        if len(g) <= 3:
            if cell_norm == g:
                return True
            for sep in (" ", ",", ";", "/", "-", ":", "("):
                if cell_norm.startswith(g + sep) or cell_norm.endswith(sep + g):
                    return True
                if sep + g + sep in cell_norm:
                    return True
            continue
        if g in cell_norm or cell_norm in g:
            return True
    return False


def _metric_shares_country_column_a(
    grid: np.ndarray,
    *,
    geo_labels: list[str],
    synonyms: dict[str, list[str]],
    header_row_0based: int,
    country_col: int,
    first_metric_col: int,
) -> dict[str, float]:
    """
    Eurostat custom Table 3 layout: title block in rows 1..header_row_0based,
    header row with end-use names in columns first_metric_col.. (e.g. Excel E–O),
    country names in column A (0-based country_col), one row per country.
    """
    nr, nc = grid.shape
    if header_row_0based < 0 or header_row_0based >= nr:
        return {}
    if country_col < 0 or country_col >= nc or first_metric_col < 0 or first_metric_col >= nc:
        return {}

    data_row = -1
    for r in range(header_row_0based + 1, nr):
        cell = _norm_key(_cell_str(grid[r, country_col]))
        if _geo_row_match(cell, geo_labels):
            data_row = r
            break
    if data_row < 0:
        return {}

    out: dict[str, float] = {}
    for c in range(first_metric_col, nc):
        hdr = _cell_str(grid[header_row_0based, c])
        if not hdr:
            continue
        metric_key: str | None = None
        for metric, names in synonyms.items():
            for n in names:
                if _text_matches_synonym(hdr, n):
                    metric_key = metric
                    break
            if metric_key:
                break
        if metric_key is None:
            continue
        v = _parse_share_cell(grid[data_row, c])
        if v is None:
            continue
        if metric_key not in out:
            out[metric_key] = v
    return out


def _metric_shares_from_grid(
    grid: np.ndarray,
    *,
    geo_labels: list[str],
    year: int,
    synonyms: dict[str, list[str]],
) -> dict[str, float]:
    """
    Wide layout: a header row contains a GEO label; optional column also contains
    the target year. First column holds product / end-use labels; values sit in the
    GEO (or year) column.
    """
    nr, nc = grid.shape
    geo_norm = {_norm_key(g) for g in geo_labels}

    hr = -1
    hc_geo = -1
    for r in range(min(40, nr)):
        for c in range(nc):
            t = _norm_key(_cell_str(grid[r, c]))
            if _geo_token_matches_cell(t, geo_norm):
                hr, hc_geo = r, c
                break
        if hc_geo >= 0:
            break

    out: dict[str, float] = {}
    if hc_geo < 0:
        return out

    value_col = hc_geo
    ystr = str(year)
    for c in range(nc):
        cell = _cell_str(grid[hr, c])
        if ystr in cell:
            value_col = c
            break

    syn_flat: list[tuple[str, str]] = []
    for metric, names in synonyms.items():
        for n in names:
            syn_flat.append((metric, _norm_key(n)))

    for r in range(hr + 1, nr):
        parts: list[str] = []
        for lc in range(min(4, nc)):
            p = _norm_key(_cell_str(grid[r, lc]))
            if p:
                parts.append(p)
        label = " ".join(parts)
        if not label:
            continue
        for metric, nk in syn_flat:
            if _text_matches_synonym(label, nk):
                v = _parse_share_cell(grid[r, value_col])
                if v is not None:
                    out[metric] = v
                break
    return out


def _metric_shares_long_df(df, geo_labels: list[str], year: int, synonyms: dict[str, list[str]]) -> dict[str, float]:
    try:
        import pandas as pd
    except ImportError:
        return {}

    cols = {str(c).strip().lower(): c for c in df.columns}
    geo_c = None
    for k in ("geo", "geography", "geopolitical entity (reporting)", "country"):
        if k in cols:
            geo_c = cols[k]
            break
    time_c = None
    for k in ("time", "year", "period"):
        if k in cols:
            time_c = cols[k]
            break
    prod_c = None
    for k in ("product", "nrg_bal", "end use", "end-use", "label", "item"):
        if k in cols:
            prod_c = cols[k]
            break
    val_c = None
    for k in ("value", "obs_value", "data"):
        if k in cols:
            val_c = cols[k]
            break
    if geo_c is None or prod_c is None or val_c is None:
        return {}

    gl = [str(g).strip() for g in geo_labels]
    sub = df[df[geo_c].astype(str).str.strip().isin(gl)]
    if sub.empty:
        mask = False
        for g in gl:
            mask = mask | df[geo_c].astype(str).str.contains(re.escape(g), case=False, na=False)
        sub = df[mask]
    if sub.empty:
        return {}
    if time_c is not None:
        yt = sub[time_c].astype(str).str.strip()
        sub = sub[(yt == str(year)) | yt.str.startswith(str(year))]

    out: dict[str, float] = {}
    prod_series = sub[prod_c].astype(str)
    for metric, names in synonyms.items():
        for n in names:
            hit = sub[prod_series.map(lambda p, syn=n: _text_matches_synonym(str(p), syn))]
            if hit.empty:
                continue
            v = _parse_share_cell(hit.iloc[0][val_c])
            if v is not None:
                out[metric] = v
            break
    return out


def load_f_enduse_for_country(
    root: Path,
    iso3: str,
    main_cfg: dict[str, Any],
) -> dict[str, float]:
    """
    Build F_enduse(class) from Eurostat Table_3 (household end-use shares) plus
    Residential/config/eurostat_end_use.json.

    Commercial classes use default_commercial_factor. Residential classes map to
    a metric (space_heating, cooking, ...); the metric value is the national share
    (0--1) used as multiplicative correction.
    """
    euro = main_cfg.get("eurostat") or {}
    default_all = 1.0
    out: dict[str, float] = {k: default_all for k in MODEL_CLASSES}

    if not bool(euro.get("enabled", False)):
        return out

    xlsx_rel = (main_cfg.get("paths") or {}).get("eurostat_xlsx")
    if not xlsx_rel:
        return out
    xlsx_path = _resolve(root, xlsx_rel)
    if not xlsx_path.is_file():
        print(
            f"[eurostat] enabled but file missing: {xlsx_path} — using F_enduse=1.0",
            file=sys.stderr,
        )
        return out

    side_path_rel = euro.get("end_use_config", "Residential/config/eurostat_end_use.json")
    side = _load_eurostat_sidecar(_resolve(root, side_path_rel))
    year = int(side.get("year", euro.get("year", 2021)))
    iso3u = iso3.strip().upper()
    geo_map = side.get("iso3_to_geo_labels") or {}
    geo_labels = list(geo_map.get(iso3u, [iso3u]))
    synonyms = side.get("metric_row_synonyms") or {}
    class_to_metric = side.get("class_to_metric") or {}
    def_res = float(side.get("default_residential_if_missing", 1.0))
    def_com = float(side.get("default_commercial_factor", 1.0))

    sheet_requested = str(euro.get("end_use_sheet", "Table_3_2023"))
    try:
        import pandas as pd
    except ImportError:
        print("[eurostat] pandas not installed — F_enduse=1.0", file=sys.stderr)
        return out

    try:
        xl_names = pd.ExcelFile(xlsx_path, engine="openpyxl").sheet_names
    except Exception as exc:
        print(f"[eurostat] cannot open workbook: {exc}", file=sys.stderr)
        return out

    sheet = sheet_requested
    if sheet_requested not in xl_names:
        wn = sheet_requested.lower().replace(" ", "_")
        alt = None
        for n in xl_names:
            nn = n.lower().replace(" ", "_")
            if nn == wn or wn in nn or nn in wn:
                alt = n
                break
        if alt is None:
            for n in xl_names:
                nl = n.lower().replace(" ", "_")
                if "table_3" in nl or "table3" in nl or "table 3" in n.lower():
                    alt = n
                    break
        if alt is not None:
            print(
                f"[eurostat] sheet {sheet_requested!r} not found; using {alt!r}",
                file=sys.stderr,
            )
            sheet = alt
        else:
            sn = ", ".join(repr(s) for s in xl_names[:20])
            print(
                f"[eurostat] sheet {sheet_requested!r} missing; available: {sn} — F_enduse=1.0",
                file=sys.stderr,
            )
            return out

    try:
        raw = pd.read_excel(xlsx_path, sheet_name=sheet, header=None, engine="openpyxl")
    except Exception as exc:
        print(f"[eurostat] cannot read sheet {sheet!r}: {exc}", file=sys.stderr)
        return out

    grid = raw.to_numpy(dtype=object)
    t3 = side.get("table_3_country_rows") or {}
    metrics = _metric_shares_country_column_a(
        grid,
        geo_labels=geo_labels,
        synonyms=synonyms,
        header_row_0based=int(t3.get("header_row_0based", 11)),
        country_col=int(t3.get("country_column_0based", 0)),
        first_metric_col=int(t3.get("first_metric_column_0based", 4)),
    )
    if not metrics:
        metrics = _metric_shares_from_grid(
            grid, geo_labels=geo_labels, year=year, synonyms=synonyms
        )
    if not metrics:
        for header_row in range(min(12, len(raw))):
            try:
                df2 = pd.read_excel(
                    xlsx_path,
                    sheet_name=sheet,
                    header=int(header_row),
                    engine="openpyxl",
                )
            except Exception:
                continue
            metrics = _metric_shares_long_df(df2, geo_labels, year, synonyms)
            if metrics:
                break

    if not metrics:
        try:
            xl = pd.ExcelFile(xlsx_path, engine="openpyxl")
            sn = ", ".join(repr(s) for s in xl.sheet_names[:12])
            extra = f" Workbook sheets (first 12): {sn}" if sn else ""
        except Exception:
            extra = ""
        print(
            f"[eurostat] could not parse shares for {iso3u} on sheet {sheet!r} — "
            f"F_enduse=1.0.{extra}",
            file=sys.stderr,
        )
        return out

    def _metric_value(mkey: str | list | None) -> float:
        if mkey is None or mkey == "default":
            return def_res
        if isinstance(mkey, list):
            vals: list[float] = []
            for k in mkey:
                if k in metrics and metrics[k] is not None:
                    try:
                        vals.append(float(metrics[k]))
                    except (TypeError, ValueError):
                        pass
            if not vals:
                return def_res
            return float(sum(vals) / len(vals))
        try:
            return float(metrics.get(str(mkey), def_res))
        except (TypeError, ValueError):
            return def_res

    for cls in MODEL_CLASSES:
        if cls.startswith("C_"):
            out[cls] = def_com
            continue
        mkey = class_to_metric.get(cls, "default")
        out[cls] = _metric_value(mkey)

    return out


def load_end_use_by_class(path: Path) -> dict[str, float]:
    """Deprecated: kept for tests; prefer load_f_enduse_for_country."""
    data = json.loads(path.read_text(encoding="utf-8"))
    default = float(data.get("default", 1.0))
    by_class = data.get("by_class") or {}
    out: dict[str, float] = {}
    for k in MODEL_CLASSES:
        v = by_class.get(k, default)
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            out[k] = default
    return out

