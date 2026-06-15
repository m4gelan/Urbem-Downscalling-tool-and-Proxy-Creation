from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml
from pyproj import Transformer

from lines_osm import POLLUTANTS, distribute_roads_to_lines

ROADS_SNAP = 7
AREA_COLS = ["snap", "xcor_sw", "ycor_sw", "zcor_sw", "xcor_ne", "ycor_ne", "zcor_ne", *POLLUTANTS]
POINT_COLS = ["snap", "xcor", "ycor", "Hi", "Vi", "Ti", "radi", *POLLUTANTS]


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in (
        "Input_folder", "Output_folder", "City", "EPSG", "domain", "Grid_step_m",
        "Roads_lines", "SNAP_TO_GNFR", "ZCOR_SW", "ZCOR_NE",
    ):
        if key not in cfg:
            raise KeyError(f"config missing {key!r}")
    for key in ("xmin", "ymin", "xmax", "ymax"):
        if key not in cfg["domain"]:
            raise KeyError(f"domain missing {key!r}")
    roads = cfg["Roads_lines"]
    for key in ("path", "layer", "highway_column"):
        if key not in roads:
            raise KeyError(f"Roads_lines missing {key!r}")
    return cfg


def domain_box(cfg: dict) -> tuple[float, float, float, float]:
    d = cfg["domain"]
    return float(d["xmin"]), float(d["ymin"]), float(d["xmax"]), float(d["ymax"])


def clip_point_rows(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    if df.empty:
        return df
    xmin, ymin, xmax, ymax = domain_box(cfg)
    m = (
        (df["xcor"].astype(float) >= xmin)
        & (df["ycor"].astype(float) >= ymin)
        & (df["xcor"].astype(float) <= xmax)
        & (df["ycor"].astype(float) <= ymax)
    )
    return df.loc[m].reset_index(drop=True)


def clip_area_rows(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    if df.empty:
        return df
    xmin, ymin, xmax, ymax = domain_box(cfg)
    m = (
        (df["xcor_sw"].astype(float) >= xmin)
        & (df["ycor_sw"].astype(float) >= ymin)
        & (df["xcor_ne"].astype(float) <= xmax)
        & (df["ycor_ne"].astype(float) <= ymax)
    )
    return df.loc[m].reset_index(drop=True)


def resolve_path(raw: str, root: Path) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else root / p


def iter_sector_snaps(raw) -> list[tuple[int, str]]:
    if isinstance(raw, dict):
        pairs = raw.items()
    elif isinstance(raw, list):
        pairs = []
        for entry in raw:
            if not isinstance(entry, dict) or len(entry) != 1:
                raise ValueError(f"bad SNAP entry: {entry!r}")
            pairs.append(next(iter(entry.items())))
    else:
        raise TypeError("SNAP map must be dict or list")
    out: list[tuple[int, str]] = []
    for key, val in pairs:
        snap = int(str(key).replace("SNAP_", ""))
        folders = val if isinstance(val, list) else [val]
        for folder in folders:
            out.append((snap, str(folder)))
    return sorted(out, key=lambda x: (x[0], x[1]))


def parse_snap_map(raw) -> dict[int, str]:
    out: dict[int, str] = {}
    for snap, folder in iter_sector_snaps(raw):
        out.setdefault(snap, folder)
    return out


def sector_snap(raw, folder: str) -> int:
    for snap, name in iter_sector_snaps(raw):
        if name == folder:
            return snap
    raise KeyError(f"SNAP_TO_GNFR missing sector {folder!r}")


def parse_snap_int_map(raw) -> dict[int, int]:
    if isinstance(raw, dict):
        pairs = raw.items()
    else:
        raise TypeError("SNAP int map must be dict")
    out: dict[int, int] = {}
    for key, val in pairs:
        snap = int(str(key).replace("SNAP_", ""))
        out[snap] = int(val)
    return out


def source_crs(input_dir: Path) -> str:
    manifest = input_dir / "manifest.yaml"
    if not manifest.is_file():
        raise FileNotFoundError(f"missing {manifest}")
    with open(manifest, encoding="utf-8") as f:
        meta = yaml.safe_load(f)
    crs = meta.get("crs")
    if not crs:
        crs = (meta.get("config") or {}).get("domain", {}).get("crs")
    if not crs:
        raise KeyError("manifest missing crs")
    return str(crs)


def read_grid_csv(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size <= len("pollutant,x,y,emission\n"):
        return pd.DataFrame(columns=["pollutant", "x", "y", "emission"])
    df = pd.read_csv(path)
    for col in ("pollutant", "x", "y", "emission"):
        if col not in df.columns:
            raise ValueError(f"{path}: missing column {col!r}")
    return df


def reproject_xy(df: pd.DataFrame, src_crs: str, dst_epsg: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    tr = Transformer.from_crs(src_crs, f"EPSG:{dst_epsg}", always_xy=True)
    xs, ys = tr.transform(df["x"].astype(float).values, df["y"].astype(float).values)
    out = df.copy()
    out["x"] = xs
    out["y"] = ys
    return out


def km_sw(v: float, step: int) -> int:
    return int(step * (float(v) // step))


def snap_tags(cfg: dict, internal_snap: int) -> tuple[int, int, int]:
    zcor_sw = parse_snap_int_map(cfg["ZCOR_SW"])
    if internal_snap not in zcor_sw:
        raise KeyError(f"ZCOR_SW missing SNAP_{internal_snap}")
    return internal_snap, zcor_sw[internal_snap], int(cfg["ZCOR_NE"])


def to_area_rows(df: pd.DataFrame, cfg: dict, internal_snap: int, step: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=AREA_COLS)
    snap, z_sw, z_ne = snap_tags(cfg, internal_snap)
    tmp = df.copy()
    tmp["xcor_sw"] = tmp["x"].map(lambda v: km_sw(v, step))
    tmp["ycor_sw"] = tmp["y"].map(lambda v: km_sw(v, step))
    grouped = tmp.groupby(["xcor_sw", "ycor_sw", "pollutant"], as_index=False)["emission"].sum()
    wide = grouped.pivot(index=["xcor_sw", "ycor_sw"], columns="pollutant", values="emission").reset_index()
    for pol in POLLUTANTS:
        if pol not in wide.columns:
            wide[pol] = 0.0
    wide["snap"] = snap
    wide["xcor_ne"] = wide["xcor_sw"] + step
    wide["ycor_ne"] = wide["ycor_sw"] + step
    wide["zcor_sw"] = z_sw
    wide["zcor_ne"] = z_ne
    wide = wide[AREA_COLS]
    return wide[wide[POLLUTANTS].sum(axis=1) > 0]


def merge_area_rows(parts: list[pd.DataFrame]) -> pd.DataFrame:
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame(columns=AREA_COLS)
    all_rows = pd.concat(parts, ignore_index=True)
    keys = ["snap", "xcor_sw", "ycor_sw", "zcor_sw", "xcor_ne", "ycor_ne", "zcor_ne"]
    return all_rows.groupby(keys, as_index=False)[POLLUTANTS].sum()


def to_point_rows(df: pd.DataFrame, cfg: dict, internal_snap: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=POINT_COLS)
    snap, _, _ = snap_tags(cfg, internal_snap)
    grouped = df.groupby(["x", "y", "pollutant"], as_index=False)["emission"].sum()
    wide = grouped.pivot(index=["x", "y"], columns="pollutant", values="emission").reset_index()
    for pol in POLLUTANTS:
        if pol not in wide.columns:
            wide[pol] = 0.0
    wide = wide.rename(columns={"x": "xcor", "y": "ycor"})
    wide["snap"] = snap
    wide["Hi"] = 0
    wide["Vi"] = 0
    wide["Ti"] = 0
    wide["radi"] = 0
    wide["xcor"] = wide["xcor"].round(0).astype(int)
    wide["ycor"] = wide["ycor"].round(0).astype(int)
    wide = wide[POINT_COLS]
    return wide[wide[POLLUTANTS].sum(axis=1) > 0]


def merge_point_rows(parts: list[pd.DataFrame]) -> pd.DataFrame:
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame(columns=POINT_COLS)
    all_rows = pd.concat(parts, ignore_index=True)
    keys = ["snap", "xcor", "ycor", "Hi", "Vi", "Ti", "radi"]
    return all_rows.groupby(keys, as_index=False)[POLLUTANTS].sum()


def run(cfg_path: Path | None = None) -> None:
    root = _repo_root()
    cfg = load_config(cfg_path or Path(__file__).resolve().parent / "config.yaml")
    input_dir = resolve_path(cfg["Input_folder"], root)
    output_dir = resolve_path(cfg["Output_folder"], root)
    sector_snaps = iter_sector_snaps(cfg["SNAP_TO_GNFR"])
    roads_cfg = cfg["Roads_lines"]
    roads_path = resolve_path(roads_cfg["path"], root)
    if not roads_path.is_file():
        raise FileNotFoundError(f"Roads_lines.path not found: {roads_path}")

    step = int(cfg["Grid_step_m"])
    dst_epsg = int(cfg["EPSG"])
    city = str(cfg["City"])
    src_crs = source_crs(input_dir)

    area_parts: list[pd.DataFrame] = []
    road_cell_parts: list[pd.DataFrame] = []
    point_parts: list[pd.DataFrame] = []

    for internal_snap, sector in sector_snaps:
        sector_dir = input_dir / sector
        if not sector_dir.is_dir():
            print(f"skip missing sector folder: {sector_dir}")
            continue

        area_df = reproject_xy(read_grid_csv(sector_dir / "area_emission_grid.csv"), src_crs, dst_epsg)
        point_df = reproject_xy(read_grid_csv(sector_dir / "point_emission_grid.csv"), src_crs, dst_epsg)

        if internal_snap == ROADS_SNAP:
            road_cell_parts.append(to_area_rows(area_df, cfg, internal_snap, step))
        else:
            area_parts.append(to_area_rows(area_df, cfg, internal_snap, step))
        point_parts.append(to_point_rows(point_df, cfg, internal_snap))

    areas = clip_area_rows(merge_area_rows(area_parts), cfg)
    road_cells = clip_area_rows(merge_area_rows(road_cell_parts), cfg)
    points = clip_point_rows(merge_point_rows(point_parts), cfg)
    lines = distribute_roads_to_lines(
        road_cells,
        roads_path,
        layer=str(roads_cfg["layer"]),
        highway_column=str(roads_cfg["highway_column"]),
        dst_epsg=dst_epsg,
        domain=domain_box(cfg),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    areas_path = output_dir / f"area_source_{city}.csv"
    lines_path = output_dir / f"line_source_{city}.csv"
    points_path = output_dir / f"point_source_{city}.csv"

    areas.to_csv(areas_path, index=False)
    lines.to_csv(lines_path, index=False)
    points.to_csv(points_path, index=False)

    print(f"areas:  {len(areas)} rows -> {areas_path}")
    print(f"lines:  {len(lines)} rows -> {lines_path}")
    print(f"points: {len(points)} rows -> {points_path}")


if __name__ == "__main__":
    run()
