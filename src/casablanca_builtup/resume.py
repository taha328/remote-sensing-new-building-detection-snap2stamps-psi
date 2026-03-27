from __future__ import annotations

from pathlib import Path
import logging

import geopandas as gpd
import xarray as xr

from casablanca_builtup.config import ExportConfig, Sentinel2Config
from casablanca_builtup.fusion import FusionArtifacts
from casablanca_builtup.io import (
    is_valid_raster_artifact,
    is_valid_vector_artifact,
    read_raster,
    read_vector,
    write_raster,
    write_vector,
)
from casablanca_builtup.runtime import log_timing
from casablanca_builtup.run_context import RunContext
from casablanca_builtup.s1.detection import S1DetectionArtifacts
from casablanca_builtup.s2.refinement import S2SupportArtifacts

LOGGER = logging.getLogger(__name__)

S1_BANDS = ("vv", "vh")
S2_BANDS_BASE = (
    "blue",
    "green",
    "red",
    "nir",
    "swir1",
    "ndvi",
    "ndbi",
    "mndwi",
    "clear_count",
    "valid_count",
    "clear_fraction",
    "valid_fraction",
)
S2_BANDS_WITH_BSI = S2_BANDS_BASE + ("bsi",)


def _all_exist(paths: dict[str, Path]) -> bool:
    return all(is_valid_raster_artifact(path, cleanup=True) for path in paths.values())


def _dataset_from_band_paths(paths: dict[str, Path]) -> xr.Dataset | None:
    if not _all_exist(paths):
        return None
    data = {}
    for name, path in paths.items():
        data[name] = read_raster(path).rename(name)
    return xr.Dataset(data)


def s1_composite_paths(context: RunContext, period_id: str, phase: str) -> dict[str, Path]:
    return {
        band: context.rasters_dir / f"{period_id}_s1_{phase}_{band}.tif"
        for band in S1_BANDS
    }


def load_s1_composite(context: RunContext, period_id: str, phase: str) -> xr.Dataset | None:
    return _dataset_from_band_paths(s1_composite_paths(context, period_id, phase))


def save_s1_composite(
    dataset: xr.Dataset,
    context: RunContext,
    period_id: str,
    phase: str,
    crs: str,
    export_config: ExportConfig,
) -> None:
    for band, path in s1_composite_paths(context, period_id, phase).items():
        write_raster(dataset[band], path, crs, export_config)


def s2_composite_paths(context: RunContext, period_id: str, phase: str, use_bsi: bool) -> dict[str, Path]:
    bands = S2_BANDS_WITH_BSI if use_bsi else S2_BANDS_BASE
    return {
        band: context.rasters_dir / f"{period_id}_s2_{phase}_{band}.tif"
        for band in bands
    }


def load_s2_composite(
    context: RunContext,
    period_id: str,
    phase: str,
    use_bsi: bool,
) -> xr.Dataset | None:
    return _dataset_from_band_paths(s2_composite_paths(context, period_id, phase, use_bsi))


def save_s2_composite(
    dataset: xr.Dataset,
    context: RunContext,
    period_id: str,
    phase: str,
    crs: str,
    export_config: ExportConfig,
    use_bsi: bool,
) -> None:
    for band, path in s2_composite_paths(context, period_id, phase, use_bsi).items():
        if band in dataset:
            with log_timing(
                LOGGER,
                "Save S2 composite band",
                period_id=period_id,
                phase=phase,
                band=band,
                path=path,
            ):
                write_raster(dataset[band], path, crs, export_config)
            LOGGER.info(
                "Saved S2 composite band artifact | period_id=%s phase=%s band=%s bytes=%s",
                period_id,
                phase,
                band,
                path.stat().st_size,
            )


def s1_detection_paths(context: RunContext, period_id: str) -> dict[str, Path]:
    return {
        "pvalue_vv": context.rasters_dir / f"{period_id}_s1_pvalue_vv.tif",
        "ratio_vv": context.rasters_dir / f"{period_id}_s1_ratio_vv.tif",
        "ratio_vh": context.rasters_dir / f"{period_id}_s1_ratio_vh.tif",
        "candidate": context.rasters_dir / f"{period_id}_s1_candidate.tif",
    }


def load_s1_detection(context: RunContext, period_id: str) -> S1DetectionArtifacts | None:
    paths = s1_detection_paths(context, period_id)
    if not _all_exist(paths):
        return None
    return S1DetectionArtifacts(
        pvalue_vv=read_raster(paths["pvalue_vv"]).rename("pvalue_vv"),
        ratio_vv=read_raster(paths["ratio_vv"]).rename("ratio_vv"),
        ratio_vh=read_raster(paths["ratio_vh"]).rename("ratio_vh"),
        candidate=read_raster(paths["candidate"]).rename("construction").astype(bool),
    )


def save_s1_detection(
    artifacts: S1DetectionArtifacts,
    context: RunContext,
    period_id: str,
    crs: str,
    export_config: ExportConfig,
) -> None:
    paths = s1_detection_paths(context, period_id)
    write_raster(artifacts.pvalue_vv, paths["pvalue_vv"], crs, export_config)
    write_raster(artifacts.ratio_vv, paths["ratio_vv"], crs, export_config)
    write_raster(artifacts.ratio_vh, paths["ratio_vh"], crs, export_config)
    write_raster(artifacts.candidate.astype("uint8"), paths["candidate"], crs, export_config)


def refinement_paths(context: RunContext, period_id: str) -> dict[str, Path]:
    return {
        "score": context.rasters_dir / f"{period_id}_s2_score.tif",
        "reliable": context.rasters_dir / f"{period_id}_s2_reliable.tif",
        "decision": context.rasters_dir / f"{period_id}_fusion_decision.tif",
        "refined": context.rasters_dir / f"{period_id}_refined_mask.tif",
    }


def load_refinement(
    context: RunContext,
    period_id: str,
    s2_config: Sentinel2Config,
) -> tuple[S2SupportArtifacts, FusionArtifacts] | None:
    paths = refinement_paths(context, period_id)
    if not _all_exist(paths):
        return None
    score = read_raster(paths["score"]).rename("s2_score").astype("uint8")
    reliable = read_raster(paths["reliable"]).rename("s2_reliable").astype(bool)
    support = (score >= s2_config.support_score_min).rename("s2_support")
    decision = read_raster(paths["decision"]).rename("fusion_decision").astype("uint8")
    refined = read_raster(paths["refined"]).rename("construction").astype(bool)
    return (
        S2SupportArtifacts(score=score, support=support, reliable=reliable),
        FusionArtifacts(refined=refined, decision=decision),
    )


def save_refinement(
    support: S2SupportArtifacts,
    fusion: FusionArtifacts,
    context: RunContext,
    period_id: str,
    crs: str,
    export_config: ExportConfig,
) -> None:
    paths = refinement_paths(context, period_id)
    write_raster(support.score, paths["score"], crs, export_config)
    write_raster(support.reliable.astype("uint8"), paths["reliable"], crs, export_config)
    write_raster(fusion.decision, paths["decision"], crs, export_config)
    write_raster(fusion.refined.astype("uint8"), paths["refined"], crs, export_config)


def postprocess_paths(context: RunContext) -> dict[str, Path]:
    return {
        "cumulative_raw": context.rasters_dir / "cumulative_raw.tif",
        "cumulative_refined": context.rasters_dir / "cumulative_refined.tif",
        "density": context.rasters_dir / "zone_density.tif",
        "zone": context.rasters_dir / "zone_mask.tif",
    }


def load_postprocess_rasters(context: RunContext) -> dict[str, xr.DataArray] | None:
    paths = postprocess_paths(context)
    if not _all_exist(paths):
        return None
    return {
        "cumulative_raw": read_raster(paths["cumulative_raw"]).rename("cumulative_raw"),
        "cumulative_refined": read_raster(paths["cumulative_refined"]).rename("cumulative_refined"),
        "density": read_raster(paths["density"]).rename("density"),
        "zone": read_raster(paths["zone"]).rename("zone"),
    }


def save_postprocess_rasters(
    context: RunContext,
    cumulative_raw: xr.DataArray,
    cumulative_refined: xr.DataArray,
    density: xr.DataArray,
    zone: xr.DataArray,
    crs: str,
    export_config: ExportConfig,
) -> None:
    paths = postprocess_paths(context)
    write_raster(cumulative_raw, paths["cumulative_raw"], crs, export_config)
    write_raster(cumulative_refined, paths["cumulative_refined"], crs, export_config)
    write_raster(density, paths["density"], crs, export_config)
    write_raster(zone, paths["zone"], crs, export_config)


def polygon_stage_path(context: RunContext) -> Path:
    return context.vectors_dir / "zone_polygons.parquet"


def load_polygon_artifact(context: RunContext) -> gpd.GeoDataFrame | None:
    path = polygon_stage_path(context)
    if not is_valid_vector_artifact(path, cleanup=True):
        return None
    return read_vector(path)


def save_polygon_artifact(frame: gpd.GeoDataFrame, context: RunContext) -> None:
    write_vector(frame, polygon_stage_path(context))


def export_vector_path(context: RunContext, extension: str) -> Path:
    suffix = extension if extension.startswith(".") else f".{extension}"
    return context.vectors_dir / f"builtup_zones{suffix}"
