from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import yaml

from proxy.alpha.Compute_alpha_matrix import AlphaMatrixResult
from proxy.core import log
from proxy.diagnostics.weight_sensitivity.pollutants_config import REFERENCE_POLLUTANTS


def w_groups_bundle_dir(export_root: Path, sector_key: str, country_tag: str, year: int) -> Path:
    return export_root / sector_key / f"{country_tag}_{int(year)}"


def _safe_group_filename(name: str) -> str:
    s = re.sub(r"[^\w\-]+", "_", str(name).strip())
    return s or "group"


def _cell_id_meta(h: int, w: int, transform: Any, crs: Any) -> dict[str, Any]:
    return {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 1,
        "dtype": "int32",
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
        "nodata": -1,
    }


def _mass_by_cell(cams_cells: dict[int, dict[str, Any]], pollutant: str) -> dict[str, float]:
    ref = pollutant.strip()
    out: dict[str, float] = {}
    for cid, row in cams_cells.items():
        pols = row.get("pollutants_within_cell") or {}
        v = pols.get(ref)
        if v is None:
            for k, val in pols.items():
                if str(k).strip().upper() == ref.upper():
                    v = val
                    break
        if v is None or float(v) <= 0.0:
            v = 0.0
        out[str(int(cid))] = float(v)
    return out


def _mass_by_pollutant(
    cams_cells: dict[int, dict[str, Any]],
    pollutants: list[str],
) -> dict[str, dict[str, float]]:
    return {p: _mass_by_cell(cams_cells, p) for p in pollutants}


def _write_float_raster(path: Path, plane: np.ndarray, transform: Any, crs: Any) -> None:
    h, w = plane.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        compress="deflate",
        tiled=True,
    ) as dst:
        dst.write(np.asarray(plane, dtype=np.float32), 1)


def _write_mix_terms(
    bundle_dir: Path,
    mix_by_group: dict[str, dict[str, Any]],
    transform: Any,
    crs: Any,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for gname, spec in mix_by_group.items():
        mixer = str(spec["mixer"])
        weights = {str(k): float(v) for k, v in spec["weights"].items()}
        terms_in = spec["terms"]
        if not isinstance(terms_in, dict):
            raise TypeError(f"mix[{gname!r}].terms must be a mapping")
        term_entries: list[dict[str, str]] = []
        for tkey, plane in terms_in.items():
            fname = f"mix__{_safe_group_filename(gname)}__{_safe_group_filename(str(tkey))}.tif"
            _write_float_raster(bundle_dir / fname, plane, transform, crs)
            term_entries.append({"key": str(tkey), "file": fname})
        ent: dict[str, Any] = {
            "name": str(gname),
            "mixer": mixer,
            "weights": weights,
            "terms": term_entries,
        }
        if "weight_keys" in spec:
            ent["weight_keys"] = [str(k) for k in spec["weight_keys"]]
        entries.append(ent)
    return entries


def write_w_groups_bundle(
    bundle_dir: Path,
    *,
    sector_key: str,
    country_tag: str,
    year: int,
    W_by_group: dict[str, np.ndarray],
    cell_id: np.ndarray,
    transform: Any,
    crs: Any,
    alpha_result: AlphaMatrixResult | None,
    cams_cells: dict[int, dict[str, Any]],
    reference_pollutants: list[str] | None = None,
    mix_by_group: dict[str, dict[str, Any]] | None = None,
) -> Path:
    pollutants = list(reference_pollutants or REFERENCE_POLLUTANTS)
    if not W_by_group:
        raise ValueError("W_by_group must be non-empty")
    bundle_dir.mkdir(parents=True, exist_ok=True)
    h, w = next(iter(W_by_group.values())).shape
    cid = np.asarray(cell_id, dtype=np.int32)
    if cid.shape != (h, w):
        raise ValueError(f"cell_id shape {cid.shape} != W shape {(h, w)}")

    cell_path = bundle_dir / "cell_id.tif"
    with rasterio.open(cell_path, "w", **_cell_id_meta(h, w, transform, crs)) as dst:
        dst.write(cid, 1)

    group_entries: list[dict[str, str]] = []
    for gname, plane in W_by_group.items():
        if plane.shape != (h, w):
            raise ValueError(f"W[{gname!r}] shape {plane.shape} != {(h, w)}")
        fname = f"W_g__{_safe_group_filename(gname)}.tif"
        gpath = bundle_dir / fname
        with rasterio.open(
            gpath,
            "w",
            driver="GTiff",
            height=h,
            width=w,
            count=1,
            dtype="float32",
            crs=crs,
            transform=transform,
            compress="deflate",
            tiled=True,
        ) as dst:
            dst.write(np.asarray(plane, dtype=np.float32), 1)
        group_entries.append({"name": str(gname), "file": fname})

    alpha_doc: dict[str, Any] = {"pollutant_labels": [], "group_names": [], "alpha": []}
    if alpha_result is not None:
        alpha_doc = {
            "pollutant_labels": list(alpha_result.pollutant_labels),
            "group_names": list(alpha_result.group_names),
            "alpha": np.asarray(alpha_result.alpha, dtype=np.float64).tolist(),
        }

    mass_doc = _mass_by_pollutant(cams_cells, pollutants)
    mass_path = bundle_dir / "cams_mass_by_pollutant.json"
    mass_path.write_text(json.dumps(mass_doc, indent=2), encoding="utf-8")
    legacy_pol = "NOx" if "NOx" in pollutants else pollutants[0]
    nox_path = bundle_dir / "cams_nox_by_cell.json"
    nox_path.write_text(json.dumps(mass_doc.get(legacy_pol, {}), indent=2), encoding="utf-8")
    (bundle_dir / "alpha_matrix.json").write_text(json.dumps(alpha_doc, indent=2), encoding="utf-8")

    manifest: dict[str, Any] = {
        "sector_key": sector_key,
        "country_tag": country_tag,
        "year": int(year),
        "reference_pollutants": pollutants,
        "height": h,
        "width": w,
        "groups": group_entries,
        "cell_id_file": "cell_id.tif",
        "alpha_matrix_file": "alpha_matrix.json",
        "cams_mass_file": "cams_mass_by_pollutant.json",
        "cams_nox_file": "cams_nox_by_cell.json",
    }
    if mix_by_group:
        manifest["mix"] = _write_mix_terms(bundle_dir, mix_by_group, transform, crs)
    with (bundle_dir / "groups_manifest.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)

    n_mix = len(mix_by_group) if mix_by_group else 0
    log.info(f"W_groups bundle: {bundle_dir} ({len(group_entries)} groups, {n_mix} mix)")
    return bundle_dir


def maybe_export_w_groups(
    enabled: bool,
    export_root: Path | None,
    *,
    sector_key: str,
    country_tag: str,
    year: int,
    W_by_group: dict[str, np.ndarray],
    cell_id: np.ndarray,
    transform: Any,
    crs: Any,
    alpha_result: AlphaMatrixResult | None,
    cams_cells: dict[int, dict[str, Any]],
    reference_pollutants: list[str] | None = None,
    mix_by_group: dict[str, dict[str, Any]] | None = None,
) -> Path | None:
    if not enabled:
        return None
    if export_root is None:
        raise ValueError("export_w_groups=True requires w_groups_export_root")
    bundle = w_groups_bundle_dir(export_root, sector_key, country_tag, year)
    return write_w_groups_bundle(
        bundle,
        sector_key=sector_key,
        country_tag=country_tag,
        year=year,
        W_by_group=W_by_group,
        cell_id=cell_id,
        transform=transform,
        crs=crs,
        alpha_result=alpha_result,
        cams_cells=cams_cells,
        reference_pollutants=reference_pollutants,
        mix_by_group=mix_by_group,
    )
