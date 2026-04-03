from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from casablanca_psi.cdpsi import plan_cdpsi_stack
from casablanca_psi.artifact_lifecycle import CleanupWarning
from casablanca_psi.config import load_config
from casablanca_psi.manifests import SlcScene, StackManifest
from casablanca_psi.pipeline import run_pipeline


def _build_manifest() -> StackManifest:
    scenes = []
    for index, acquisition_date in enumerate(["2023-08-17", "2023-08-29", "2023-09-10", "2023-09-22", "2023-10-04", "2023-10-16"], start=1):
        scenes.append(
            SlcScene(
                scene_id=f"SCENE_{index}",
                product_name=f"SCENE_{index}",
                acquisition_start=f"{acquisition_date}T00:00:00Z",
                acquisition_stop=f"{acquisition_date}T00:00:10Z",
                acquisition_date=acquisition_date,
                direction="ascending",
                relative_orbit=147,
                polarization="VV",
                swath_mode="IW",
                product_type="IW_SLC__1S",
                processing_level="L1",
                platform="Sentinel-1A",
                asset_name="product",
                href="https://download.example.invalid/scene.zip",
            )
        )
    return StackManifest(
        stack_id="asc_rel147_vv",
        direction="ascending",
        relative_orbit=147,
        product_type="SLC",
        scenes=scenes,
    )


def _write_emergence_ready_stamps_outputs(export_dir: Path) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)
    (export_dir / "ps_points.csv").write_text(
        "point_id,x_local_m,y_local_m,azimuth_index,range_index,lon,lat,temporal_coherence,scene_elevation_m,dem_error_phase_per_m,mean_velocity_mm_yr,master_day,n_ifg,n_image\n"
        "1,10.0,20.0,101,202,-7.60,33.57,0.95,12.0,0.01,NaN,739151,5,5\n",
        encoding="utf-8",
    )
    (export_dir / "ps_timeseries.csv").write_text("point_id,epoch,metric_name,value\n", encoding="utf-8")


def _write_raw_only_stamps_outputs(export_dir: Path) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)
    (export_dir / "ps_points.csv").write_text(
        "point_id,lon,lat,temporal_coherence,scene_elevation_m,dem_error_phase_per_m,mean_velocity_mm_yr,master_day,n_ifg,n_image\n"
        "1,-7.60,33.57,0.95,12.0,0.01,NaN,739151,5,5\n",
        encoding="utf-8",
    )
    (export_dir / "ps_timeseries.csv").write_text("point_id,epoch,metric_name,value\n", encoding="utf-8")


def _write_snap_export(export_dir: Path) -> None:
    for name in ("rslc", "diff0", "geo"):
        path = export_dir / name
        path.mkdir(parents=True, exist_ok=True)
        (path / "ok.txt").write_text("ok", encoding="utf-8")
    diff_dir = export_dir / "diff0"
    (diff_dir / "master_secondary.diff").write_text("ok", encoding="utf-8")
    (diff_dir / "master_secondary.base").write_text(
        "initial_baseline(TCN): 0.0 0.0 0.0\ninitial_baseline_rate: 0.0 0.0 0.0\n",
        encoding="utf-8",
    )


def test_run_pipeline_persists_runtime_policies(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "runs_psi" / "group-001" / "attempt-001"
    manifest = _build_manifest()

    monkeypatch.setattr("casablanca_psi.pipeline.configure_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr("casablanca_psi.stamps.StaMPSRunner._resolve_snaphu_binary", lambda self: Path("/tmp/snaphu"))
    monkeypatch.setattr("casablanca_psi.stamps.StaMPSRunner._resolve_triangle_binary", lambda self: Path("/tmp/triangle"))
    monkeypatch.setattr("casablanca_psi.pipeline.load_aoi_geometry", lambda *_args, **_kwargs: SimpleNamespace(wkt="POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))"))
    monkeypatch.setattr("casablanca_psi.pipeline.build_manifests", lambda *_args, **_kwargs: {manifest.stack_id: manifest})
    monkeypatch.setattr("casablanca_psi.pipeline.download_stack_scenes", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "casablanca_psi.pipeline.SnapGraphRunner.run_stack",
        lambda *_args, **_kwargs: SimpleNamespace(stamps_export_dir=Path("/tmp/stamps_export")),
    )

    context = run_pipeline(
        Path("configs/psi_casablanca_slc_minimal.yaml"),
        stop_after="snap_preprocess",
        run_dir=run_dir,
    )

    report = json.loads((context.reports_dir / "run_report.json").read_text(encoding="utf-8"))

    assert report["status"] == "completed"
    assert report["runtime_policies"]["snap_gpt"]["gpt_vmoptions_path"] == "/Applications/esa-snap/bin/gpt.vmoptions"
    assert "-Xmx8G" in report["runtime_policies"]["snap_gpt"]["effective_java_options"]
    assert report["runtime_policies"]["stamps"]["asc_rel147_vv"]["mode"] == "serial_patch_batches_then_serial_merge"
    assert report["runtime_policies"]["stamps"]["asc_rel147_vv"]["effective_parallel_patch_workers"] == 1
    assert report["runtime_policies"]["stamps"]["asc_rel147_vv"]["range_patches"] == 4
    assert report["runtime_policies"]["stamps"]["asc_rel147_vv"]["azimuth_patches"] == 3
    assert report["runtime_policies"]["stamps"]["asc_rel147_vv"]["patch_batch_count"] == 12
    assert report["runtime_policies"]["stamps"]["asc_rel147_vv"]["merge_resample_size"] == 100
    assert report["runtime_policies"]["stamps"]["asc_rel147_vv"]["snaphu_path"] == "/tmp/snaphu"
    assert report["runtime_policies"]["stamps"]["asc_rel147_vv"]["triangle_path"] == "/tmp/triangle"
    assert report["runtime_policies"]["cdpsi"]["asc_rel147_vv"]["valid_break_count"] == 1
    assert report["runtime_policies"]["cdpsi"]["asc_rel147_vv"]["subset_run_count"] == 2


def test_run_pipeline_persists_snap_cleanup_warnings(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "runs_psi" / "group-001" / "attempt-001"
    manifest = _build_manifest()

    monkeypatch.setattr("casablanca_psi.pipeline.configure_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr("casablanca_psi.pipeline.load_aoi_geometry", lambda *_args, **_kwargs: SimpleNamespace(wkt="POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))"))
    monkeypatch.setattr("casablanca_psi.pipeline.build_manifests", lambda *_args, **_kwargs: {manifest.stack_id: manifest})
    monkeypatch.setattr("casablanca_psi.pipeline.download_stack_scenes", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "casablanca_psi.pipeline.SnapGraphRunner.run_stack",
        lambda *_args, **_kwargs: SimpleNamespace(
            stamps_export_dir=Path("/tmp/stamps_export"),
            cleanup_warnings=(
                CleanupWarning(
                    category="snap_intermediate",
                    checkpoint="stamps_export_validated",
                    reason="validated export already exists",
                    path=Path("/tmp/stamps_export_inputs"),
                    message="OSError: [Errno 66] Directory not empty",
                ),
            ),
        ),
    )

    context = run_pipeline(
        Path("configs/psi_casablanca_slc_minimal.yaml"),
        stop_after="snap_preprocess",
        run_dir=run_dir,
    )

    report = json.loads((context.reports_dir / "run_report.json").read_text(encoding="utf-8"))

    assert report["status"] == "completed"
    assert report["cleanup_summary"]["warning_count"] == 3
    assert report["cleanup_warnings"][0]["checkpoint"] == "stamps_export_validated"
    assert "Directory not empty" in report["cleanup_warnings"][0]["message"]


def test_run_pipeline_persists_stamps_cleanup_warnings(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "runs_psi" / "group-001" / "attempt-001"
    manifest = _build_manifest()

    monkeypatch.setattr("casablanca_psi.pipeline.configure_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr("casablanca_psi.pipeline.load_aoi_geometry", lambda *_args, **_kwargs: SimpleNamespace(wkt="POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))"))
    monkeypatch.setattr("casablanca_psi.pipeline.build_manifests", lambda *_args, **_kwargs: {manifest.stack_id: manifest})
    monkeypatch.setattr("casablanca_psi.pipeline.download_stack_scenes", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "casablanca_psi.pipeline.SnapGraphRunner.run_stack",
        lambda *_args, **_kwargs: SimpleNamespace(stamps_export_dir=Path("/tmp/stamps_export")),
    )
    monkeypatch.setattr(
        "casablanca_psi.pipeline.StaMPSRunner.run_stack",
        lambda *_args, **_kwargs: SimpleNamespace(
            ps_points_csv=Path("/tmp/ps_points.csv"),
            ps_timeseries_csv=Path("/tmp/ps_timeseries.csv"),
            cleanup_warnings=(
                CleanupWarning(
                    category="snap_export",
                    checkpoint="stamps_outputs_validated",
                    reason="validated outputs already exist",
                    path=Path("/tmp/stamps_export"),
                    message="OSError: [Errno 66] Directory not empty",
                ),
            ),
        ),
    )

    context = run_pipeline(
        Path("configs/psi_casablanca_slc_minimal.yaml"),
        stop_after="stamps",
        run_dir=run_dir,
    )

    report = json.loads((context.reports_dir / "run_report.json").read_text(encoding="utf-8"))

    assert report["status"] == "completed"
    assert report["cleanup_summary"]["warning_count"] == 3
    assert report["cleanup_warnings"][0]["checkpoint"] == "stamps_outputs_validated"
    assert "Directory not empty" in report["cleanup_warnings"][0]["message"]


def test_run_pipeline_skips_snap_when_reusable_stamps_outputs_exist(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "runs_psi" / "group-001" / "attempt-001"
    manifest = _build_manifest()
    export_dir = run_dir / "stamps" / manifest.stack_id / "export"
    _write_emergence_ready_stamps_outputs(export_dir)
    config = load_config(Path("configs/psi_casablanca_slc_minimal.yaml"))
    plan = plan_cdpsi_stack(manifest, config.stacks[0], config.psi)
    for subset_plan in plan.subset_runs:
        _write_snap_export(run_dir / "snap" / subset_plan.stack_id / "stamps_export")

    monkeypatch.setattr("casablanca_psi.pipeline.configure_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr("casablanca_psi.pipeline.load_aoi_geometry", lambda *_args, **_kwargs: SimpleNamespace(wkt="POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))"))
    monkeypatch.setattr("casablanca_psi.pipeline.build_manifests", lambda *_args, **_kwargs: {manifest.stack_id: manifest})
    monkeypatch.setattr("casablanca_psi.pipeline.download_stack_scenes", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "casablanca_psi.pipeline.SnapGraphRunner.run_stack",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("SNAP should be skipped when reusable StaMPS outputs exist")),
    )
    monkeypatch.setattr(
        "casablanca_psi.pipeline.StaMPSRunner.run_stack",
        lambda *_args, **_kwargs: SimpleNamespace(
            ps_points_csv=export_dir / "ps_points.csv",
            ps_timeseries_csv=export_dir / "ps_timeseries.csv",
            cleanup_warnings=(),
        ),
    )

    context = run_pipeline(
        Path("configs/psi_casablanca_slc_minimal.yaml"),
        stop_after="stamps",
        run_dir=run_dir,
    )

    report = json.loads((context.reports_dir / "run_report.json").read_text(encoding="utf-8"))

    assert report["status"] == "completed"
    assert report["stages"]["snap_preprocess"]["status"] == "completed"
    assert report["stages"]["stamps"]["status"] == "completed"


def test_run_pipeline_does_not_skip_snap_for_raw_only_stamps_outputs(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "runs_psi" / "group-001" / "attempt-001"
    manifest = _build_manifest()
    export_dir = run_dir / "stamps" / manifest.stack_id / "export"
    _write_raw_only_stamps_outputs(export_dir)

    monkeypatch.setattr("casablanca_psi.pipeline.configure_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr("casablanca_psi.pipeline.load_aoi_geometry", lambda *_args, **_kwargs: SimpleNamespace(wkt="POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))"))
    monkeypatch.setattr("casablanca_psi.pipeline.build_manifests", lambda *_args, **_kwargs: {manifest.stack_id: manifest})
    monkeypatch.setattr("casablanca_psi.pipeline.download_stack_scenes", lambda *_args, **_kwargs: None)
    snap_calls: list[str] = []

    def fake_snap_run_stack(*_args, **_kwargs):
        snap_calls.append("called")
        return SimpleNamespace(stamps_export_dir=Path("/tmp/stamps_export"), cleanup_warnings=())

    monkeypatch.setattr("casablanca_psi.pipeline.SnapGraphRunner.run_stack", fake_snap_run_stack)
    monkeypatch.setattr(
        "casablanca_psi.pipeline.StaMPSRunner.run_stack",
        lambda *_args, **_kwargs: SimpleNamespace(
            ps_points_csv=export_dir / "ps_points.csv",
            ps_timeseries_csv=export_dir / "ps_timeseries.csv",
            cleanup_warnings=(),
        ),
    )

    context = run_pipeline(
        Path("configs/psi_casablanca_slc_minimal.yaml"),
        stop_after="stamps",
        run_dir=run_dir,
    )

    report = json.loads((context.reports_dir / "run_report.json").read_text(encoding="utf-8"))

    assert report["status"] == "completed"
    assert len(snap_calls) == 3
    assert report["stages"]["snap_preprocess"]["status"] == "completed"
    assert report["stages"]["stamps"]["status"] == "completed"
