from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import dask
import xarray as xr

from aoi_builtup.acquisition.stac import build_period_manifests, sign_manifest_items
from aoi_builtup.config import PipelineConfig, load_config
from aoi_builtup.fusion import apply_soft_refinement
from aoi_builtup.grid import build_grid, load_aoi_frame
from aoi_builtup.io import write_json, write_vector, write_yaml
from aoi_builtup.logging_utils import configure_logging
from aoi_builtup.postprocess.vectorize import (
    build_cumulative_first_change,
    build_density_zone_mask,
    polygonize_mask,
)
from aoi_builtup.qa import (
    base_run_report,
    binary_area_ha,
    binary_pixel_count,
    decision_histogram,
    finalize_run_report,
    mark_stage,
    mark_running_stages,
    mask_fraction,
    mean_over_mask,
    stage_status,
)
from aoi_builtup.resume import (
    export_vector_path,
    load_polygon_artifact,
    load_postprocess_rasters,
    load_refinement,
    load_s1_composite,
    load_s1_detection,
    load_s2_composite,
    save_polygon_artifact,
    save_postprocess_rasters,
    save_refinement,
    save_s1_composite,
    save_s1_detection,
    save_s2_composite,
)
from aoi_builtup.run_context import RunContext
from aoi_builtup.runtime import PipelineInterruptedError, interruption_guard, log_timing
from aoi_builtup.s1.composite import build_s1_composite
from aoi_builtup.s1.detection import detect_s1_change
from aoi_builtup.s2.composite import build_s2_composite
from aoi_builtup.s2.refinement import build_s2_support, build_unavailable_s2_support_like

LOGGER = logging.getLogger(__name__)
STAGE_ORDER = {
    "acquire": 1,
    "build_composites": 2,
    "detect_s1": 3,
    "refine_s2": 4,
    "polygonize": 5,
    "export": 6,
}


def _project_root_from_config_path(config_path: Path) -> Path:
    if config_path.parent.name == "configs":
        return config_path.parent.parent
    return config_path.parent

def _period_report(period_id: str, manifest: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {
        "period_id": period_id,
        "s1_before_items": len(manifest["s1_before"]),
        "s1_after_items": len(manifest["s1_after"]),
        "s2_before_items": len(manifest.get("s2_before", [])),
        "s2_after_items": len(manifest.get("s2_after", [])),
    }


def _persist_report(report: dict[str, Any], context: RunContext) -> None:
    write_json(report, context.reports_dir / "run_report.json")


def run_pipeline(
    config_path: str | Path,
    stop_after: str | None = None,
    run_dir: str | Path | None = None,
    resume_latest: bool = False,
) -> RunContext:
    config_file = Path(config_path).resolve()
    config = load_config(config_file)
    project_root = _project_root_from_config_path(config_file)
    context = RunContext.create(
        config,
        project_root,
        run_dir=Path(run_dir).resolve() if run_dir is not None else None,
        resume_latest=resume_latest,
    )
    return execute_pipeline(context, config, config_file, stop_after=stop_after)


def execute_pipeline(
    context: RunContext,
    config: PipelineConfig,
    config_file: Path,
    stop_after: str | None = None,
) -> RunContext:
    context.ensure_directories()
    configure_logging(
        config.run.log_level,
        context.logs_dir / "pipeline.log" if config.run.write_log_file else None,
    )
    write_yaml(config.model_dump(mode="json", exclude_none=True), context.reports_dir / "resolved_config.yaml")
    report: dict[str, Any] = {}
    overwrite = config.cache.overwrite

    try:
        with interruption_guard():
            if config.dask.enabled:
                dask.config.set(scheduler=config.dask.scheduler)

            aoi = load_aoi_frame(config.aoi)
            grid = build_grid(aoi, config.grid)
            geometry = aoi.to_crs("EPSG:4326").geometry.iloc[0]
            grid_summary = {
                "crs": grid.crs,
                "resolution_m": grid.resolution_m,
                "bounds": list(grid.bounds),
                "width": grid.width,
                "height": grid.height,
            }
            report = base_run_report(context.run_id, str(config_file), grid_summary)
            report["run_group_id"] = context.group_id
            report["attempt_id"] = context.attempt_id
            report["run_root"] = str(context.root)
            _persist_report(report, context)

            write_json(
                {
                    "run_id": context.run_id,
                    "grid_crs": grid.crs,
                    "resolution_m": grid.resolution_m,
                    "bounds": list(grid.bounds),
                    "width": grid.width,
                    "height": grid.height,
                },
                context.reports_dir / "grid.json",
            )

            mark_stage(report, "acquire", "running")
            _persist_report(report, context)
            include_s2 = stop_after != "detect_s1"
            manifests = build_period_manifests(config, context.manifests_dir, geometry, include_s2=include_s2)
            for period in config.periods:
                report["periods"].append(_period_report(period.id, manifests[period.id]))
            mark_stage(report, "acquire", "completed")
            _persist_report(report, context)

            if stop_after == "acquire":
                finalize_run_report(report, "completed")
                _persist_report(report, context)
                return context

            raw_masks: list[tuple[str, xr.DataArray]] = []
            refined_masks: list[tuple[str, xr.DataArray]] = []

            mark_stage(report, "build_composites", "running")
            _persist_report(report, context)
            for period in config.periods:
                period_manifest = manifests[period.id]
                period_report = next(item for item in report["periods"] if item["period_id"] == period.id)
                s1_before = None if overwrite else load_s1_composite(context, period.id, "before")
                s1_after = None if overwrite else load_s1_composite(context, period.id, "after")
                period_report["s1_before_source"] = "artifact" if s1_before is not None else "pending"
                period_report["s1_after_source"] = "artifact" if s1_after is not None else "pending"
                if s1_before is None:
                    s1_before = build_s1_composite(
                        sign_manifest_items(period_manifest["s1_before"]),
                        grid.geobox,
                        config.sentinel1,
                        config.dask,
                    )
                    save_s1_composite(s1_before, context, period.id, "before", grid.crs, config.export)
                    period_report["s1_before_source"] = "computed"
                if s1_after is None:
                    s1_after = build_s1_composite(
                        sign_manifest_items(period_manifest["s1_after"]),
                        grid.geobox,
                        config.sentinel1,
                        config.dask,
                    )
                    save_s1_composite(s1_after, context, period.id, "after", grid.crs, config.export)
                    period_report["s1_after_source"] = "computed"
                s2_before = None
                s2_after = None
                s2_items_available = bool(period_manifest.get("s2_before")) and bool(period_manifest.get("s2_after"))
                period_report["s2_items_available"] = s2_items_available
                if stop_after == "build_composites" or stop_after is None or STAGE_ORDER[stop_after] >= STAGE_ORDER["refine_s2"]:
                    s2_before = None if overwrite else load_s2_composite(context, period.id, "before", config.sentinel2.use_bsi)
                    s2_after = None if overwrite else load_s2_composite(context, period.id, "after", config.sentinel2.use_bsi)
                    period_report["s2_before_source"] = (
                        "artifact" if s2_before is not None else ("pending" if s2_items_available else "unavailable")
                    )
                    period_report["s2_after_source"] = (
                        "artifact" if s2_after is not None else ("pending" if s2_items_available else "unavailable")
                    )
                    if s2_before is None and s2_items_available:
                        with log_timing(
                            LOGGER,
                            "Build and save S2 composite",
                            period_id=period.id,
                            phase="before",
                            item_count=len(period_manifest["s2_before"]),
                        ):
                            s2_before = build_s2_composite(
                                sign_manifest_items(period_manifest["s2_before"]),
                                grid.geobox,
                                config.sentinel2,
                                config.dask,
                                export_config=config.export,
                                stage_dir=context.staging_dir,
                                period_id=period.id,
                                phase="before",
                            )
                            save_s2_composite(
                                s2_before,
                                context,
                                period.id,
                                "before",
                                grid.crs,
                                config.export,
                                config.sentinel2.use_bsi,
                            )
                        period_report["s2_before_source"] = "computed"
                    if s2_after is None and s2_items_available:
                        with log_timing(
                            LOGGER,
                            "Build and save S2 composite",
                            period_id=period.id,
                            phase="after",
                            item_count=len(period_manifest["s2_after"]),
                        ):
                            s2_after = build_s2_composite(
                                sign_manifest_items(period_manifest["s2_after"]),
                                grid.geobox,
                                config.sentinel2,
                                config.dask,
                                export_config=config.export,
                                stage_dir=context.staging_dir,
                                period_id=period.id,
                                phase="after",
                            )
                            save_s2_composite(
                                s2_after,
                                context,
                                period.id,
                                "after",
                                grid.crs,
                                config.export,
                                config.sentinel2.use_bsi,
                            )
                        period_report["s2_after_source"] = "computed"
                    elif not config.sentinel2.allow_unavailable:
                        raise RuntimeError(f"Sentinel-2 support is unavailable for period {period.id}.")

                if stop_after == "build_composites":
                    continue

                if stage_status(report, "detect_s1") == "pending":
                    mark_stage(report, "detect_s1", "running")
                    _persist_report(report, context)
                s1_detection = None if overwrite else load_s1_detection(context, period.id)
                period_report["s1_detection_source"] = "artifact" if s1_detection is not None else "pending"
                if s1_detection is None:
                    s1_detection = detect_s1_change(s1_before, s1_after, config.detection, grid.resolution_m)
                    save_s1_detection(s1_detection, context, period.id, grid.crs, config.export)
                    period_report["s1_detection_source"] = "computed"
                raw_masks.append((period.id, s1_detection.candidate))
                period_report["s1_candidate_pixels"] = binary_pixel_count(s1_detection.candidate)
                period_report["s1_candidate_area_ha"] = binary_area_ha(s1_detection.candidate, grid.resolution_m)
                period_report["s1_candidate_fraction"] = mask_fraction(s1_detection.candidate)
                period_report["s1_ratio_vv_mean_on_candidate"] = mean_over_mask(s1_detection.ratio_vv, s1_detection.candidate)
                period_report["s1_ratio_vh_mean_on_candidate"] = mean_over_mask(s1_detection.ratio_vh, s1_detection.candidate)

                if stop_after == "detect_s1":
                    continue

                if stage_status(report, "refine_s2") == "pending":
                    mark_stage(report, "refine_s2", "running")
                    _persist_report(report, context)
                refinement_artifacts = None if overwrite else load_refinement(context, period.id, config.sentinel2)
                if refinement_artifacts is None:
                    period_report["refinement_source"] = "pending"
                    if s2_before is None or s2_after is None:
                        s2_support = build_unavailable_s2_support_like(s1_detection.candidate)
                        period_report["s2_support_mode"] = "unavailable_keep_s1"
                    else:
                        s2_support = build_s2_support(s2_before, s2_after, config.sentinel2)
                        period_report["s2_support_mode"] = "soft_refinement"
                    fusion = apply_soft_refinement(s1_detection, s2_support, config.sentinel2)
                    save_refinement(s2_support, fusion, context, period.id, grid.crs, config.export)
                    period_report["refinement_source"] = "computed"
                else:
                    s2_support, fusion = refinement_artifacts
                    period_report["refinement_source"] = "artifact"
                    period_report["s2_support_mode"] = (
                        "unavailable_keep_s1" if not period_report["s2_items_available"] else "soft_refinement"
                    )
                refined_masks.append((period.id, fusion.refined))
                period_report["s2_reliable_fraction"] = mask_fraction(s2_support.reliable)
                period_report["s2_support_fraction"] = mask_fraction(s2_support.support)
                period_report["refined_pixels"] = binary_pixel_count(fusion.refined)
                period_report["refined_area_ha"] = binary_area_ha(fusion.refined, grid.resolution_m)
                period_report["removed_by_refinement_pixels"] = max(
                    0,
                    period_report["s1_candidate_pixels"] - period_report["refined_pixels"],
                )
                period_report["removed_by_refinement_fraction"] = (
                    0.0
                    if period_report["s1_candidate_pixels"] == 0
                    else period_report["removed_by_refinement_pixels"] / period_report["s1_candidate_pixels"]
                )
                period_report["fusion_decisions"] = decision_histogram(fusion.decision)

                _persist_report(report, context)

            mark_stage(report, "build_composites", "completed")
            _persist_report(report, context)
            if stage_status(report, "detect_s1") == "running":
                mark_stage(report, "detect_s1", "completed")
                _persist_report(report, context)

            if stop_after == "build_composites":
                finalize_run_report(report, "completed")
                _persist_report(report, context)
                return context
            if stop_after == "detect_s1":
                finalize_run_report(report, "completed")
                _persist_report(report, context)
                return context

            if stage_status(report, "refine_s2") == "running":
                mark_stage(report, "refine_s2", "completed")
                _persist_report(report, context)
            if stop_after == "refine_s2":
                finalize_run_report(report, "completed")
                _persist_report(report, context)
                return context

            mark_stage(report, "polygonize", "running")
            _persist_report(report, context)
            postprocess = None if overwrite else load_postprocess_rasters(context)
            if postprocess is None:
                cumulative_raw = build_cumulative_first_change(raw_masks)
                cumulative_refined = build_cumulative_first_change(refined_masks)
                density, zone_mask = build_density_zone_mask(cumulative_refined, config.density, grid.resolution_m)
                save_postprocess_rasters(
                    context,
                    cumulative_raw,
                    cumulative_refined,
                    density,
                    zone_mask,
                    grid.crs,
                    config.export,
                )
                report["postprocess_source"] = "computed"
            else:
                cumulative_raw = postprocess["cumulative_raw"]
                cumulative_refined = postprocess["cumulative_refined"]
                density = postprocess["density"]
                zone_mask = postprocess["zone"]
                report["postprocess_source"] = "artifact"

            polygons = None if overwrite else load_polygon_artifact(context)
            if polygons is None:
                polygons = polygonize_mask(zone_mask, grid.crs, config.polygonization)
                save_polygon_artifact(polygons, context)
                report["polygon_source"] = "computed"
            else:
                report["polygon_source"] = "artifact"
            report["cumulative_raw_area_ha"] = binary_area_ha(cumulative_raw > 0, grid.resolution_m)
            report["cumulative_refined_area_ha"] = binary_area_ha(cumulative_refined > 0, grid.resolution_m)
            report["zone_mask_area_ha"] = binary_area_ha(zone_mask > 0, grid.resolution_m)
            report["polygon_count"] = int(len(polygons))
            report["zone_area_ha"] = float(polygons["area_ha"].sum()) if not polygons.empty else 0.0
            mark_stage(report, "polygonize", "completed")
            _persist_report(report, context)
            if stop_after == "polygonize":
                finalize_run_report(report, "completed")
                _persist_report(report, context)
                return context

            if stop_after in {"export", None}:
                mark_stage(report, "export", "running")
                _persist_report(report, context)
                for vector_format in config.export.vector_formats:
                    export_path = export_vector_path(context, vector_format)
                    if export_path.exists() and not overwrite:
                        continue
                    write_vector(polygons, export_path)
                mark_stage(report, "export", "completed")
                _persist_report(report, context)

            finalize_run_report(report, "completed")
            _persist_report(report, context)
            LOGGER.info("Completed run %s", context.run_id)
            return context
    except (KeyboardInterrupt, PipelineInterruptedError) as exc:
        if report:
            mark_running_stages(report, "interrupted")
            report["error"] = {"type": exc.__class__.__name__, "message": str(exc)}
            finalize_run_report(report, "interrupted")
            _persist_report(report, context)
        LOGGER.warning("Pipeline interrupted for run %s", context.run_id)
        raise
    except Exception as exc:
        if report:
            mark_running_stages(report, "failed")
            report["error"] = {"type": exc.__class__.__name__, "message": str(exc)}
            finalize_run_report(report, "failed")
            _persist_report(report, context)
        LOGGER.exception("Pipeline failed for run %s", context.run_id)
        raise
