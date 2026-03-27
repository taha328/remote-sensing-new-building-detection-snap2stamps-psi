from __future__ import annotations

from pathlib import Path
from typing import Any
import logging

import pandas as pd
import geopandas as gpd

from casablanca_builtup.io import write_json, write_vector, write_yaml
from casablanca_builtup.qa import base_run_report, finalize_run_report, mark_running_stages, mark_stage
from casablanca_builtup.runtime import PipelineInterruptedError, interruption_guard
from casablanca_psi.acquisition import build_manifests, download_stack_scenes, load_aoi_geometry
from casablanca_psi.artifact_lifecycle import CleanupRecord, CleanupWarning, cleanup_stamps_workspace
from casablanca_psi.config import PipelineConfig, load_config
from casablanca_psi.export import export_fusion_outputs, write_run_summary
from casablanca_psi.fusion import FusionOutputs, fuse_evidence
from casablanca_psi.logging_utils import configure_logging
from casablanca_psi.psi_results import detect_emergent_ps, load_ps_points
from casablanca_psi.run_context import RunContext
from casablanca_psi.snap import SnapGraphRunner
from casablanca_psi.stamps import StaMPSRunner

LOGGER = logging.getLogger(__name__)
STAGES = ("acquire", "download_slc", "snap_preprocess", "stamps", "parse_psi", "fuse", "export")


def _project_root_from_config_path(config_path: Path) -> Path:
    if config_path.parent.name == "configs":
        return config_path.parent.parent
    return config_path.parent


def _persist_report(report: dict[str, Any], context: RunContext) -> None:
    write_json(report, context.reports_dir / "run_report.json")


def _empty_report(run_id: str, config_path: Path, context: RunContext) -> dict[str, Any]:
    report = base_run_report(run_id, str(config_path), grid={"crs": None, "resolution_m": None, "bounds": None, "width": None, "height": None})
    report["stages"] = {name: report["stages"].get("acquire", {"status": "pending", "started_at_utc": None, "completed_at_utc": None, "duration_s": None}).copy() for name in STAGES}
    report["run_group_id"] = context.group_id
    report["attempt_id"] = context.attempt_id
    report["run_root"] = str(context.root)
    report["stack_reports"] = []
    report["cleanup_events"] = []
    report["cleanup_warnings"] = []
    report["cleanup_summary"] = {"event_count": 0, "bytes_reclaimed": 0, "warning_count": 0}
    return report


def _record_cleanup_events(report: dict[str, Any], records: list[CleanupRecord] | tuple[CleanupRecord, ...]) -> None:
    if not records:
        return
    cleanup_events = report.setdefault("cleanup_events", [])
    summary = report.setdefault("cleanup_summary", {"event_count": 0, "bytes_reclaimed": 0, "warning_count": 0})
    for record in records:
        cleanup_events.append(record.as_dict())
        summary["event_count"] += 1
        summary["bytes_reclaimed"] += record.bytes_reclaimed


def _record_cleanup_warnings(
    report: dict[str, Any],
    warnings: list[CleanupWarning] | tuple[CleanupWarning, ...],
) -> None:
    if not warnings:
        return
    cleanup_warnings = report.setdefault("cleanup_warnings", [])
    summary = report.setdefault("cleanup_summary", {"event_count": 0, "bytes_reclaimed": 0, "warning_count": 0})
    for warning in warnings:
        cleanup_warnings.append(warning.as_dict())
        summary["warning_count"] += 1


def _cleanup_report_observer(report: dict[str, Any], context: RunContext):
    def observer(records: tuple[CleanupRecord, ...]) -> None:
        _record_cleanup_events(report, records)
        _persist_report(report, context)

    return observer


def run_pipeline(
    config_path: str | Path,
    *,
    stop_after: str | None = None,
    run_dir: str | Path | None = None,
    resume_latest: bool = False,
) -> RunContext:
    config_file = Path(config_path).resolve()
    config = load_config(config_file)
    project_root = _project_root_from_config_path(config_file)
    context = RunContext.create(config, project_root, run_dir=Path(run_dir).resolve() if run_dir else None, resume_latest=resume_latest)
    context.ensure_directories()
    configure_logging(config.run.log_level, context.logs_dir / "psi_pipeline.log" if config.run.write_log_file else None)
    write_yaml(config.model_dump(mode="json", exclude_none=True), context.reports_dir / "resolved_config.yaml")
    report = _empty_report(context.run_id, config_file, context)
    _persist_report(report, context)

    try:
        with interruption_guard():
            geometry = load_aoi_geometry(config.aoi)
            aoi_wkt = geometry.wkt

            mark_stage(report, "acquire", "running")
            _persist_report(report, context)
            manifests = build_manifests(config, context)
            report["stack_reports"] = [{"stack_id": key, "scene_count": len(value.scenes)} for key, value in manifests.items()]
            mark_stage(report, "acquire", "completed")
            _persist_report(report, context)
            if stop_after == "acquire":
                finalize_run_report(report, "completed")
                _persist_report(report, context)
                return context

            mark_stage(report, "download_slc", "running")
            _persist_report(report, context)
            for manifest in manifests.values():
                download_stack_scenes(config, context, manifest)
            mark_stage(report, "download_slc", "completed")
            _persist_report(report, context)
            if stop_after == "download_slc":
                finalize_run_report(report, "completed")
                _persist_report(report, context)
                return context

            snap_runner = SnapGraphRunner(config)
            stamps_runner = StaMPSRunner(config)
            stack_points = []
            report["runtime_policies"] = {
                "snap_gpt": snap_runner.describe_runtime_policy(),
                "stamps": {stack.id: stamps_runner.describe_execution_plan(stack) for stack in config.stacks},
            }
            _persist_report(report, context)

            mark_stage(report, "snap_preprocess", "running")
            _persist_report(report, context)
            snap_outputs: dict[str, Any] = {}
            snap_cleanup_observer = _cleanup_report_observer(report, context)
            for stack_config in config.stacks:
                manifest = manifests[stack_config.id]
                snap_outputs[stack_config.id] = snap_runner.run_stack(
                    context,
                    manifest,
                    stack_config,
                    aoi_wkt,
                    cleanup_observer=snap_cleanup_observer,
                )
                _record_cleanup_warnings(report, getattr(snap_outputs[stack_config.id], "cleanup_warnings", ()))
                _persist_report(report, context)
            mark_stage(report, "snap_preprocess", "completed")
            _persist_report(report, context)
            if stop_after == "snap_preprocess":
                finalize_run_report(report, "completed")
                _persist_report(report, context)
                return context

            mark_stage(report, "stamps", "running")
            _persist_report(report, context)
            stamps_outputs: dict[str, Any] = {}
            stamps_cleanup_observer = _cleanup_report_observer(report, context)
            for stack_config in config.stacks:
                manifest = manifests[stack_config.id]
                stamps_outputs[stack_config.id] = stamps_runner.run_stack(
                    context,
                    manifest,
                    stack_config,
                    snap_outputs[stack_config.id].stamps_export_dir,
                    cleanup_observer=stamps_cleanup_observer,
                )
                _persist_report(report, context)
            mark_stage(report, "stamps", "completed")
            _persist_report(report, context)
            if stop_after == "stamps":
                finalize_run_report(report, "completed")
                _persist_report(report, context)
                return context

            mark_stage(report, "parse_psi", "running")
            _persist_report(report, context)
            for stack_config in config.stacks:
                raw_points = load_ps_points(stamps_outputs[stack_config.id].ps_points_csv)
                raw_points_path = context.points_dir / f"{stack_config.id}_ps_raw.parquet"
                write_vector(raw_points, raw_points_path)
                emergent_points = detect_emergent_ps(raw_points, config.psi)
                emergent_points["stack_id"] = stack_config.id
                emergent_points_path = context.points_dir / f"{stack_config.id}_ps_emergent.parquet"
                write_vector(emergent_points, emergent_points_path)
                if config.artifact_lifecycle.enabled and config.artifact_lifecycle.purge_stamps_workspace_after_parse:
                    cleanup_records = cleanup_stamps_workspace(
                        stamps_outputs[stack_config.id].root,
                        keep_paths=(
                            stamps_outputs[stack_config.id].ps_points_csv,
                            stamps_outputs[stack_config.id].ps_timeseries_csv,
                        ),
                        logger=LOGGER,
                    )
                    _record_cleanup_events(report, cleanup_records)
                    _persist_report(report, context)
                stack_points.append(emergent_points)
            psi_points = (
                gpd.GeoDataFrame(pd.concat(stack_points, ignore_index=True), crs=stack_points[0].crs)
                if len(stack_points) > 1
                else stack_points[0]
            )
            mark_stage(report, "parse_psi", "completed")
            _persist_report(report, context)
            if stop_after == "parse_psi":
                finalize_run_report(report, "completed")
                _persist_report(report, context)
                return context

            mark_stage(report, "fuse", "running")
            _persist_report(report, context)
            outputs: FusionOutputs = fuse_evidence(psi_points, config.fusion, cluster_buffer_m=config.psi.cluster_buffer_m)
            mark_stage(report, "fuse", "completed")
            _persist_report(report, context)
            if stop_after == "fuse":
                finalize_run_report(report, "completed")
                _persist_report(report, context)
                return context

            mark_stage(report, "export", "running")
            _persist_report(report, context)
            export_fusion_outputs(outputs, context.vectors_dir, config.export)
            write_run_summary(
                {
                    "run_id": context.run_id,
                    "candidate_points": int(len(outputs.points)),
                    "candidate_polygons": int(len(outputs.polygons)),
                    "stacks": [stack.id for stack in config.stacks],
                },
                context.reports_dir / "psi_summary.json",
            )
            mark_stage(report, "export", "completed")
            finalize_run_report(report, "completed")
            _persist_report(report, context)
            return context
    except (KeyboardInterrupt, PipelineInterruptedError) as exc:
        mark_running_stages(report, "interrupted")
        report["error"] = {"type": exc.__class__.__name__, "message": str(exc)}
        finalize_run_report(report, "interrupted")
        _persist_report(report, context)
        raise
    except Exception as exc:
        mark_running_stages(report, "failed")
        report["error"] = {"type": exc.__class__.__name__, "message": str(exc)}
        finalize_run_report(report, "failed")
        _persist_report(report, context)
        LOGGER.exception("PSI pipeline failed")
        raise
