"""
Explore Eurostat / EU livestock GeoPackage C21.gpkg (livestock numbers by region).

Run from repo root:
  python Agriculture/Auxiliaries/explore_c21_gpkg.py
  python Agriculture/Auxiliaries/explore_c21_gpkg.py --path data/Agriculture/C21.gpkg --layer 0

Requires: geopandas (and pyogrio or fiona for .gpkg I/O).

Use this output to design NUTS-level joins to LUCAS (replacing or augmenting grazing-based
proxies in methodology §2.4).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _list_layers(path: Path) -> list[str]:
    """GeoPackage stores layer names in gpkg_contents (SQLite)."""
    import sqlite3

    con = sqlite3.connect(path)
    try:
        rows = con.execute(
            "SELECT table_name FROM gpkg_contents WHERE data_type = 'features' ORDER BY table_name"
        ).fetchall()
        if rows:
            return [r[0] for r in rows]
    except Exception:
        pass
    finally:
        con.close()
    try:
        import fiona

        return list(fiona.listlayers(path))
    except Exception:
        return []


def _read_layer(path: Path, layer: str | int | None):
    import geopandas as gpd

    if layer is None:
        layers = _list_layers(path)
        if not layers:
            raise ValueError("No feature layers found in GeoPackage.")
        layer = layers[0]
    if isinstance(layer, int):
        layers = _list_layers(path)
        layer = layers[layer]
    return gpd.read_file(path, layer=layer)


def explore(path: Path, layer: str | int | None, max_rows: int, sample_frac: float | None) -> int:
    if not path.is_file():
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        print(f"  Expected livestock GeoPackage at this path (e.g. Eurostat C21 / Agric census).", file=sys.stderr)
        return 1

    try:
        import geopandas as gpd
    except ImportError:
        print("ERROR: geopandas is required. Install with: pip install geopandas pyogrio", file=sys.stderr)
        return 1

    layers = _list_layers(path)
    print(f"File: {path.resolve()}")
    print(f"Feature layers ({len(layers)}): {layers}\n")

    gdf = _read_layer(path, layer)
    print(f"Active layer: {getattr(gdf, 'name', layer)}")
    print(f"CRS: {gdf.crs}")
    print(f"Rows: {len(gdf):,}  |  Geometry types: {gdf.geometry.geom_type.value_counts().to_dict() if len(gdf) else {}}")
    print(f"Bounds: {gdf.total_bounds if len(gdf) else 'n/a'}\n")

    print("Columns (dtypes):")
    for c in gdf.columns:
        if c == "geometry":
            continue
        print(f"  {c!r}: {gdf[c].dtype}")
    print()

    # Non-geometry column stats
    num_cols = gdf.select_dtypes(include=["number"]).columns.tolist()
    if num_cols:
        print("Numeric summary (non-null count, min, max, mean):")
        for c in num_cols[:40]:
            s = gdf[c].dropna()
            if len(s) == 0:
                print(f"  {c}: all NaN")
            else:
                print(f"  {c}: n={len(s):,} min={s.min():.6g} max={s.max():.6g} mean={s.mean():.6g}")
        if len(num_cols) > 40:
            print(f"  ... ({len(num_cols) - 40} more numeric columns omitted)")
        print()

    obj_cols = [c for c in gdf.columns if c != "geometry" and gdf[c].dtype == "object"]
    for c in obj_cols[:15]:
        nu = gdf[c].nunique(dropna=True)
        print(f"Column {c!r}: {nu} distinct values (sample)")
        print(f"  head: {gdf[c].dropna().head(5).tolist()}")
    if len(obj_cols) > 15:
        print(f"... ({len(obj_cols) - 15} more object columns not detailed)")

    if sample_frac is not None and 0 < sample_frac < 1 and len(gdf) > 100:
        gdf = gdf.sample(frac=sample_frac, random_state=42)
        print(f"\nSampled {len(gdf)} rows ({sample_frac:.0%}) for preview.")
    elif max_rows and len(gdf) > max_rows:
        gdf = gdf.head(max_rows)
        print(f"\nPreview first {max_rows} rows.")

    out_json = path.with_suffix(".explore_head.json")
    try:
        # Drop geometry for JSON dump
        preview = gdf.drop(columns=["geometry"], errors="ignore")
        rec = preview.head(50).to_dict(orient="records")
        out_json.write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
        print(f"\nWrote first 50 attribute rows (no geometry) to: {out_json}")
    except Exception as e:
        print(f"\nCould not write JSON preview: {e}")

    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--path",
        type=Path,
        default=_ROOT / "data" / "Agriculture" / "C21.gpkg",
        help="Path to C21.gpkg (or other livestock GeoPackage)",
    )
    p.add_argument(
        "--layer",
        default=None,
        help="Layer name, or integer index (default: first layer)",
    )
    p.add_argument("--max-rows", type=int, default=5000, help="Max rows to load for stats (default 5000)")
    p.add_argument(
        "--sample-frac",
        type=float,
        default=None,
        help="If set (e.g. 0.1), random sample for large files",
    )
    args = p.parse_args()
    layer_arg: str | int | None = args.layer
    if layer_arg is not None:
        try:
            layer_arg = int(layer_arg)
        except ValueError:
            pass
    return explore(args.path.resolve(), layer_arg, args.max_rows, args.sample_frac)


if __name__ == "__main__":
    raise SystemExit(main())
