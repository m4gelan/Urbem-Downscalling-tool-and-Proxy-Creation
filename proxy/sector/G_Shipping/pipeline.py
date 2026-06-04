from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from rasterio.warp import Resampling

from proxy.core import log
from proxy.core.alias import resolve_osm_filepath
from proxy.core.area_weights import combined_S_shipping, normalize_W_per_cams_cell
from proxy.core.raster_helpers import warp_raster_to_grid
from proxy.core.z_score import z_score_inside
from proxy.dataset_loaders import require_filepaths_exist
from proxy.dataset_loaders.load_cams_cells_mask import load_cams_cells_mask, pixels_inside_cams_cells
from proxy.dataset_loaders.load_corine import load_corine
from proxy.dataset_loaders.load_emodnet import load_emodnet
from proxy.dataset_loaders.load_osm import load_osm, rasterize_osm
from proxy.visualizers.area_weights_map import write_shipping_area_weights_map
from proxy.writers.area_weight_stack import area_weights_tif_path, write_area_weight_equal_multiband
from proxy.writers.w_groups_export import maybe_export_w_groups


def build(
    output_dir: Path,
    sector_config_path: Path,
    *,
    area_weights: bool = True,
    point_matching: bool = False,
    country_profile: dict[str, str] | None = None,
    crs: str,
    resolution_m: float,
    pad_m: float,
    area_weights_viz_bbox_wgs84: tuple[float, float, float, float] | None = None,
    export_w_groups: bool = False,
    w_groups_export_root: Path | None = None,
) -> None:

    # 1. FIRST STEP: CHECKING EVERYTHING IS OK WITH THE FILEPATHS
    repo_root = Path(__file__).resolve().parents[3]

    with sector_config_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("sector config must be a YAML mapping")

    filepaths = cfg.get("filepaths")
    pols = cfg.get("pollutants")
    if not isinstance(pols, list) or not pols:
        raise ValueError("sector config pollutants list missing or empty")

    require_filepaths_exist(repo_root, filepaths, sector_config_path, country_profile=country_profile)

    cams_filepath = filepaths.get("CAMS", {}).get("path")
    emodnet_filepath = filepaths.get("EMODNET", {}).get("path")
    corine_filepath = filepaths.get("CORINE", {}).get("path")

    if point_matching:
        log.info("--------------------------------")
        log.info(" NO POINT MATCHING IMPLEMENTED FOR G_Shipping")
        log.info("--------------------------------")

    if area_weights:

        log.info("--------------------------------")
        log.info("AREA WEIGHTS")
        log.info("--------------------------------")

        if not country_profile:
            log.error("area_weights needs country_profile from entry")
            raise ValueError("area_weights needs country_profile from entry")

        cps_area = cfg.get("cams_area_sources") or {}
        year = int(cps_area.get("year", 2019))
        ec = list(cps_area.get("emission_category_indices") or [])
        st = list(cps_area.get("source_type_indices") or [])

        corine_cfg = cfg.get("corine") or {}
        corine_l3_codes = [int(x) for x in (corine_cfg.get("l3_codes") or [])]
        corine_band = int(corine_cfg.get("band", 1))
        if not corine_l3_codes:
            raise ValueError("sector config: under 'corine', set non-empty 'l3_codes' (CLC L3 integers)")

        # 1 CAMS area cells + global lon/lat grid metadata (single bundle for downstream)
        cams_cells, cams_grid = load_cams_cells_mask(
            repo_root / cams_filepath.replace("\\", "/"),
            year=year,
            country_iso3=country_profile["ISO3"],
            emission_category_indices=ec,
            source_type_indices=st,
            pollutants=[str(x).strip() for x in pols if str(x).strip()],
            crs=crs,
            resolution_m=resolution_m,
            pad_m=pad_m,
        )
        if not cams_cells:
            log.warning("G_Shipping area_weights: no CAMS area cells for this filter; skipping.")
            return

        # 2 Load CORINE, EMODNET and OSM Raster Maps
        corine_map, cor_tr, cor_crs, cell_id = load_corine(
            repo_root / corine_filepath.replace("\\", "/"),
            corine_l3_codes,
            corine_band,
            cams_cells,
            cams_grid,
        )

        emod_cfg = cfg.get("emodnet")

        if not isinstance(emod_cfg, dict) or "band" not in emod_cfg:
            raise ValueError("sector config: set 'emodnet.band' (GeoTIFF band index, 1-based)")

        emodnet_map, emodnet_tr, emodnet_crs, _ = load_emodnet(
            repo_root / emodnet_filepath.replace("\\", "/"),
            cams_cells,
            cams_grid,
            band=int(emod_cfg["band"]),
        )

        ch, cw = corine_map.shape
        em_src_nd = float(emod_cfg["nodata"]) if "nodata" in emod_cfg else None
        emodnet_map = warp_raster_to_grid(
            emodnet_map, emodnet_tr, emodnet_crs, ch, cw, cor_tr, cor_crs,
            src_nodata=em_src_nd, dest_init_nan=False,
        )

        osm_cfg = cfg["osm"]

        osm_polygons = load_osm(
            repo_root / resolve_osm_filepath(filepaths.get("OSM", {}).get("path"), country_profile),
            cams_cells, osm_cfg,
        )
        osm_raster = rasterize_osm(
            osm_polygons,
            ch,
            cw,
            cor_tr,
            cor_crs,
            osm_cfg["rasterize"],
            cams_cells,
        )

        log.info(
            f"G_Shipping: CORINE + EMODNET + OSM raster aligned "
            f"(CORINE {corine_map.shape}, EMODNET {emodnet_map.shape}, "
            f"osm_1px={int((osm_raster > 0).sum())})"
        )

        inside_z = pixels_inside_cams_cells(ch, cw, cor_tr, cor_crs, cams_cells) & np.isfinite(
            emodnet_map
        )
        emodnet_z = z_score_inside(
            emodnet_map,
            inside_z,
            upper_quantile=0.99,
            rescale_to_01=True,
        )
        # 4 Compute the area weights
        aw = cfg.get("area_weights") or {}
        w1 = float(aw.get("w1"))
        w2 = float(aw.get("w2"))
        w3 = float(aw.get("w3"))

        corine_01 = corine_map.astype(np.float64)
        log.info("Raster ready, computing area weights...")
        S = combined_S_shipping(corine_01, emodnet_z, osm_raster, w1=w1, w2=w2, w3=w3)
        W = normalize_W_per_cams_cell(S, cell_id, cams_cells)

        country_tag = country_profile["full_name"].replace(" ", "_")
        maybe_export_w_groups(
            export_w_groups,
            w_groups_export_root,
            sector_key="G_Shipping",
            country_tag=country_tag,
            year=year,
            W_by_group={"shipping": W},
            cell_id=cell_id,
            transform=cor_tr,
            crs=cor_crs,
            alpha_result=None,
            cams_cells=cams_cells,
            mix_by_group={
                "shipping": {
                    "mixer": "linear3",
                    "weights": {"w1": w1, "w2": w2, "w3": w3},
                    "terms": {
                        "emodnet": np.asarray(emodnet_z, dtype=np.float32),
                        "osm": np.asarray(osm_raster, dtype=np.float32),
                        "corine": np.asarray(corine_01, dtype=np.float32),
                    },
                }
            },
        )

        # 5 Multi-band GeoTIFF on CORINE reference grid
        band_vals = W
        out_tif = area_weights_tif_path(output_dir, "G_Shipping", country_tag, year)
        write_area_weight_equal_multiband(
            out_tif,
            band_vals,
            [str(x).strip() for x in pols if str(x).strip()],
            cor_tr,
            cor_crs,
        )
        log.info(f"PIPELINE FINISHED: Area-weight stack written: {out_tif}")

        if log.debug_enabled() and area_weights_viz_bbox_wgs84 is not None:
            map_html = output_dir / f"G_Shipping_{country_tag}_area_weights_map_{year}.html"
            try:
                write_shipping_area_weights_map(
                    map_html,
                    bbox_wgs84=area_weights_viz_bbox_wgs84,
                    corine_map=corine_map,
                    osm_raster=osm_raster,
                    emodnet_z=emodnet_z,
                    transform=cor_tr,
                    raster_crs=cor_crs,
                    cell_id=cell_id,
                )
                log.info(f"G_Shipping area-weights debug map: {map_html}")
            except Exception as exc:
                log.error(f"G_Shipping area-weights debug map failed: {exc}")

        return

    log.info("G_Shipping: non-area-weights path not implemented")
    return
