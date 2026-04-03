from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from casablanca_psi.artifact_lifecycle import CleanupWarning
from casablanca_psi.config import load_config
from casablanca_psi.manifests import SlcScene, StackManifest
from casablanca_psi.run_context import RunContext
from casablanca_psi.stamps import StaMPSRunner, _PatchWorkerProcess


def _build_manifest() -> StackManifest:
    scenes = []
    for index, acquisition_date in enumerate(["2023-08-29", "2023-09-10", "2023-09-22", "2023-10-04", "2023-10-16"], start=1):
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


def _write_snap_export(export_dir: Path) -> None:
    for name in ("rslc", "diff0", "geo"):
        path = export_dir / name
        path.mkdir(parents=True, exist_ok=True)
        (path / "ok.txt").write_text("ok")


def _write_raw_only_stamps_outputs(export_dir: Path) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)
    (export_dir / "ps_points.csv").write_text(
        "point_id,lon,lat,temporal_coherence,scene_elevation_m,dem_error_phase_per_m,mean_velocity_mm_yr,master_day,n_ifg,n_image\n"
        "1,-7.60,33.57,0.95,12.0,0.01,NaN,739151,5,5\n",
        encoding="utf-8",
    )
    (export_dir / "ps_timeseries.csv").write_text("point_id,epoch,metric_name,value\n", encoding="utf-8")


def _write_emergence_ready_stamps_outputs(export_dir: Path) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)
    (export_dir / "ps_points.csv").write_text(
        "point_id,x_local_m,y_local_m,azimuth_index,range_index,lon,lat,temporal_coherence,scene_elevation_m,dem_error_phase_per_m,mean_velocity_mm_yr,master_day,n_ifg,n_image\n"
        "1,10.0,20.0,101,202,-7.60,33.57,0.95,12.0,0.01,NaN,739151,5,5\n",
        encoding="utf-8",
    )
    (export_dir / "ps_timeseries.csv").write_text("point_id,epoch,metric_name,value\n", encoding="utf-8")


def test_run_stack_reuses_existing_stamps_outputs_and_preserves_snap_export_for_cdpsi(tmp_path, monkeypatch) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "psi_casablanca_slc_minimal.yaml")
    runner = StaMPSRunner(config)
    monkeypatch.setattr(runner, "validate_environment", lambda: None)
    monkeypatch.setattr(runner, "_run_mt_prep_snap", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("mt_prep_snap should not run")))
    monkeypatch.setattr(runner, "_run_matlab_batch", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("matlab batch should not run")))

    context = RunContext.create(config, tmp_path)
    context.ensure_directories()
    manifest = _build_manifest()
    stack = config.stacks[0]
    root_dir = context.stamps_dir / manifest.stack_id
    export_dir = root_dir / "export"
    _write_emergence_ready_stamps_outputs(export_dir)

    snap_export_dir = context.snap_dir / manifest.stack_id / "stamps_export"
    _write_snap_export(snap_export_dir)

    outputs = runner.run_stack(context, manifest, stack, snap_export_dir)

    assert outputs.ps_points_csv.exists()
    assert outputs.ps_timeseries_csv.exists()
    assert snap_export_dir.exists()
    assert not any(record.category == "snap_export" for record in outputs.cleanup_records)


def test_run_stack_reuses_existing_outputs_without_snap_export_cleanup_warning_in_cdpsi_mode(tmp_path, monkeypatch) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "psi_casablanca_slc_minimal.yaml")
    runner = StaMPSRunner(config)
    monkeypatch.setattr(runner, "validate_environment", lambda: None)
    monkeypatch.setattr(runner, "_run_mt_prep_snap", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("mt_prep_snap should not run")))
    monkeypatch.setattr(runner, "_run_matlab_batch", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("matlab batch should not run")))

    delete_called = False

    def fake_delete_paths(*args, **kwargs):
        nonlocal delete_called
        delete_called = True
        return []

    monkeypatch.setattr("casablanca_psi.stamps.delete_paths", fake_delete_paths)

    context = RunContext.create(config, tmp_path)
    context.ensure_directories()
    manifest = _build_manifest()
    stack = config.stacks[0]
    root_dir = context.stamps_dir / manifest.stack_id
    export_dir = root_dir / "export"
    _write_emergence_ready_stamps_outputs(export_dir)

    snap_export_dir = context.snap_dir / manifest.stack_id / "stamps_export"
    _write_snap_export(snap_export_dir)

    outputs = runner.run_stack(context, manifest, stack, snap_export_dir)

    assert outputs.ps_points_csv.exists()
    assert snap_export_dir.exists()
    assert outputs.cleanup_warnings == ()
    assert delete_called is False


def test_run_stack_preserves_snap_export_after_valid_stamps_outputs_in_cdpsi_mode(tmp_path, monkeypatch) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "psi_casablanca_slc_minimal.yaml")
    runner = StaMPSRunner(config)
    monkeypatch.setattr(runner, "validate_environment", lambda: None)

    def fake_mt_prep(_master_date: str, _snap_export_dir: Path, cwd: Path) -> None:
        (cwd / "parms.mat").write_text("ok", encoding="utf-8")

    monkeypatch.setattr(runner, "_run_mt_prep_snap", fake_mt_prep)

    def fake_matlab_batch(_script: str, cwd: Path, *, log_name: str = "matlab_batch.log") -> None:
        export_dir = cwd / "export"
        _write_emergence_ready_stamps_outputs(export_dir)

    monkeypatch.setattr(runner, "_run_matlab_batch", fake_matlab_batch)

    context = RunContext.create(config, tmp_path)
    context.ensure_directories()
    manifest = _build_manifest()
    stack = config.stacks[0]
    snap_export_dir = context.snap_dir / manifest.stack_id / "stamps_export"
    _write_snap_export(snap_export_dir)

    outputs = runner.run_stack(context, manifest, stack, snap_export_dir)

    assert outputs.ps_points_csv.exists()
    assert outputs.ps_timeseries_csv.exists()
    assert snap_export_dir.exists()
    assert not any(record.category == "snap_export" for record in outputs.cleanup_records)


def test_run_stack_preserves_snap_export_when_only_raw_psi_outputs_exist(tmp_path, monkeypatch) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "psi_casablanca_slc_minimal.yaml")
    runner = StaMPSRunner(config)
    monkeypatch.setattr(runner, "validate_environment", lambda: None)

    def fake_mt_prep(_master_date: str, _snap_export_dir: Path, cwd: Path) -> None:
        (cwd / "parms.mat").write_text("ok", encoding="utf-8")

    monkeypatch.setattr(runner, "_run_mt_prep_snap", fake_mt_prep)

    def fake_matlab_batch(_script: str, cwd: Path, *, log_name: str = "matlab_batch.log") -> None:
        _write_raw_only_stamps_outputs(cwd / "export")

    monkeypatch.setattr(runner, "_run_matlab_batch", fake_matlab_batch)

    context = RunContext.create(config, tmp_path)
    context.ensure_directories()
    manifest = _build_manifest()
    stack = config.stacks[0]
    snap_export_dir = context.snap_dir / manifest.stack_id / "stamps_export"
    _write_snap_export(snap_export_dir)

    outputs = runner.run_stack(context, manifest, stack, snap_export_dir)

    assert outputs.ps_points_csv.exists()
    assert outputs.ps_timeseries_csv.exists()
    assert snap_export_dir.exists()
    assert not any(record.category == "snap_export" for record in outputs.cleanup_records)


def test_run_stack_emits_cleanup_records_to_observer(tmp_path, monkeypatch) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "psi_casablanca_slc_minimal.yaml")
    runner = StaMPSRunner(config)
    monkeypatch.setattr(runner, "validate_environment", lambda: None)

    def fake_mt_prep(_master_date: str, _snap_export_dir: Path, cwd: Path) -> None:
        (cwd / "parms.mat").write_text("ok", encoding="utf-8")

    monkeypatch.setattr(runner, "_run_mt_prep_snap", fake_mt_prep)

    def fake_matlab_batch(_script: str, cwd: Path, *, log_name: str = "matlab_batch.log") -> None:
        _write_emergence_ready_stamps_outputs(cwd / "export")

    monkeypatch.setattr(runner, "_run_matlab_batch", fake_matlab_batch)

    context = RunContext.create(config, tmp_path)
    context.ensure_directories()
    manifest = _build_manifest()
    stack = config.stacks[0]
    snap_export_dir = context.snap_dir / manifest.stack_id / "stamps_export"
    _write_snap_export(snap_export_dir)
    observed: list[tuple[str, ...]] = []

    outputs = runner.run_stack(
        context,
        manifest,
        stack,
        snap_export_dir,
        cleanup_observer=lambda records: observed.append(tuple(record.category for record in records)),
    )

    assert outputs.ps_points_csv.exists()
    assert observed == []
    assert outputs.cleanup_records == ()


def test_environment_includes_stamps_bin_dependency_bins_and_interpreter_dir(monkeypatch) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "psi_casablanca_slc_minimal.yaml")
    runner = StaMPSRunner(config)

    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setattr("shutil.which", lambda command: str(Path("/Applications/MATLAB_R2025b.app/bin/matlab")))
    monkeypatch.setattr(runner, "_resolve_snaphu_binary", lambda: Path("/tmp/tools/snaphu/bin/snaphu"))
    monkeypatch.setattr(runner, "_resolve_triangle_binary", lambda: Path("/tmp/tools/triangle/bin/triangle"))

    env = runner._environment()

    path_parts = env["PATH"].split(":")
    assert path_parts[0] == str(config.stamps.install_root / "bin")
    assert path_parts[1] == "/tmp/tools/snaphu/bin"
    assert path_parts[2] == "/tmp/tools/triangle/bin"
    assert path_parts[3] == "/Applications/MATLAB_R2025b.app/bin"
    assert env["STAMPS"] == str(config.stamps.install_root)
    assert env["SNAPHU_BIN"] == "/tmp/tools/snaphu/bin/snaphu"
    assert env["TRIANGLE_BIN"] == "/tmp/tools/triangle/bin/triangle"


def test_validate_environment_requires_snaphu(monkeypatch) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "psi_casablanca_slc_minimal.yaml")
    runner = StaMPSRunner(config)

    monkeypatch.setattr("shutil.which", lambda command: str(Path("/Applications/MATLAB_R2025b.app/bin/matlab")))
    monkeypatch.setattr(runner, "_resolve_snaphu_binary", lambda: None)
    monkeypatch.setattr(runner, "_resolve_triangle_binary", lambda: Path("/tmp/tools/triangle/bin/triangle"))

    with pytest.raises(FileNotFoundError, match="SNAPHU binary not found"):
        runner.validate_environment()


def test_validate_environment_requires_triangle(monkeypatch) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "psi_casablanca_slc_minimal.yaml")
    runner = StaMPSRunner(config)

    monkeypatch.setattr("shutil.which", lambda command: str(Path("/Applications/MATLAB_R2025b.app/bin/matlab")))
    monkeypatch.setattr(runner, "_resolve_snaphu_binary", lambda: Path("/tmp/tools/snaphu/bin/snaphu"))
    monkeypatch.setattr(runner, "_resolve_triangle_binary", lambda: None)

    with pytest.raises(FileNotFoundError, match="Triangle binary not found"):
        runner.validate_environment()


def test_run_mt_prep_snap_requires_candidate_files(tmp_path, monkeypatch) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "psi_casablanca_slc_minimal.yaml")
    runner = StaMPSRunner(config)

    monkeypatch.setattr(runner, "_environment", lambda: {})
    commands: list[list[str]] = []

    def fake_run(command: list[str], cwd: Path, env: dict[str, str], check: bool) -> None:
        commands.append(command)
        (cwd / "patch.list").write_text("PATCH_1\n", encoding="utf-8")
        (cwd / "PATCH_1").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("subprocess.run", fake_run)

    try:
        runner._run_mt_prep_snap("2023-09-22", tmp_path / "snap_export", cwd=tmp_path)
    except RuntimeError as exc:
        assert "pscands.1.ij" in str(exc)
    else:
        raise AssertionError("mt_prep_snap validation should fail when candidate files are missing")

    assert commands


def test_validate_mt_prep_outputs_accepts_complete_patch(tmp_path) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "psi_casablanca_slc_minimal.yaml")
    runner = StaMPSRunner(config)

    (tmp_path / "patch.list").write_text("PATCH_1\n", encoding="utf-8")
    patch_dir = tmp_path / "PATCH_1"
    patch_dir.mkdir()
    for name in ("pscands.1.ij", "pscands.1.ll", "pscands.1.hgt", "pscands.1.ph"):
        (patch_dir / name).write_text("ok", encoding="utf-8")

    runner._validate_mt_prep_outputs(tmp_path)


def test_run_stack_cleans_partial_workspace_before_rerun(tmp_path, monkeypatch) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "psi_casablanca_slc_minimal.yaml")
    runner = StaMPSRunner(config)
    monkeypatch.setattr(runner, "validate_environment", lambda: None)

    seen_state: dict[str, bool] = {}

    def fake_mt_prep(_master_date: str, _snap_export_dir: Path, cwd: Path) -> None:
        seen_state["patch_removed"] = not (cwd / "PATCH_1").exists()
        seen_state["stale_export_removed"] = not (cwd / "export" / "partial.txt").exists()
        (cwd / "parms.mat").write_text("ok", encoding="utf-8")

    def fake_matlab_batch(_script: str, cwd: Path, *, log_name: str = "matlab_batch.log") -> None:
        export_dir = cwd / "export"
        export_dir.mkdir(parents=True, exist_ok=True)
        (export_dir / "ps_points.csv").write_text("point_id,velocity\n1,1.0\n", encoding="utf-8")
        (export_dir / "ps_timeseries.csv").write_text("point_id,epoch,metric_name,value\n", encoding="utf-8")

    monkeypatch.setattr(runner, "_run_mt_prep_snap", fake_mt_prep)
    monkeypatch.setattr(runner, "_run_matlab_batch", fake_matlab_batch)

    context = RunContext.create(config, tmp_path)
    context.ensure_directories()
    manifest = _build_manifest()
    stack = config.stacks[0]
    root_dir = context.stamps_dir / manifest.stack_id
    (root_dir / "PATCH_1").mkdir(parents=True, exist_ok=True)
    (root_dir / "PATCH_1" / "pscands.1.ph").write_text("stale", encoding="utf-8")
    (root_dir / "export").mkdir(parents=True, exist_ok=True)
    (root_dir / "export" / "partial.txt").write_text("stale", encoding="utf-8")
    (root_dir / "matlab_batch.log").write_text("old log", encoding="utf-8")

    snap_export_dir = context.snap_dir / manifest.stack_id / "stamps_export"
    _write_snap_export(snap_export_dir)

    outputs = runner.run_stack(context, manifest, stack, snap_export_dir)

    assert outputs.ps_points_csv.exists()
    assert seen_state == {"patch_removed": True, "stale_export_removed": True}
    assert any(record.category == "stamps_workspace" for record in outputs.cleanup_records)


def test_run_matlab_batch_uses_logfile(tmp_path, monkeypatch) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "psi_casablanca_slc_minimal.yaml")
    runner = StaMPSRunner(config)

    commands: list[list[str]] = []
    monkeypatch.setattr(runner, "_environment", lambda: {})

    def fake_run(command: list[str], cwd: Path, env: dict[str, str], check: bool) -> None:
        commands.append(command)

    monkeypatch.setattr("subprocess.run", fake_run)

    runner._run_matlab_batch("disp('ok')", cwd=tmp_path)

    assert commands
    assert "-logfile" in commands[0]
    assert str(tmp_path / "matlab_batch.log") in commands[0]


def test_matlab_startup_script_injects_project_helper_dir() -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "psi_casablanca_slc_minimal.yaml")
    runner = StaMPSRunner(config)

    startup = runner._matlab_startup_script()

    assert f"addpath('{runner._matlab_dir.as_posix()}');" in startup
    assert "isempty(which('gausswin'))" in startup
    assert "isempty(which('interp'))" in startup
    assert "isempty(which('nanmean'))" in startup
    assert runner._matlab_helpers_dir.as_posix() in startup


def test_describe_execution_plan_is_patch_aware(monkeypatch) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "psi_casablanca_slc_minimal.yaml")
    runner = StaMPSRunner(config)
    monkeypatch.setattr(runner, "_resolve_snaphu_binary", lambda: Path("/tmp/tools/snaphu/bin/snaphu"))
    monkeypatch.setattr(runner, "_resolve_triangle_binary", lambda: Path("/tmp/tools/triangle/bin/triangle"))

    plan = runner.describe_execution_plan(config.stacks[0])

    assert plan["range_patches"] == 4
    assert plan["azimuth_patches"] == 3
    assert plan["planned_total_patches"] == 12
    assert plan["patch_batch_count"] == 12
    assert plan["requested_parallel_patch_workers"] == 1
    assert plan["effective_parallel_patch_workers"] == 1
    assert plan["parallel_patch_phase_enabled"] is False
    assert plan["serial_patch_batch_execution"] is True
    assert plan["merge_resample_size"] == 100
    assert plan["snaphu_path"] == "/tmp/tools/snaphu/bin/snaphu"
    assert plan["triangle_path"] == "/tmp/tools/triangle/bin/triangle"
    assert plan["mode"] == "serial_patch_batches_then_serial_merge"


def test_run_stack_uses_parallel_patch_phase_then_serial_merge(tmp_path, monkeypatch) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "psi_casablanca_slc_minimal.yaml")
    config.stamps.max_parallel_patch_workers = 2
    runner = StaMPSRunner(config)
    monkeypatch.setattr(runner, "validate_environment", lambda: None)

    parallel_patch_lists: list[list[str]] = []
    serial_scripts: list[tuple[str, str]] = []

    def fake_mt_prep(_master_date: str, _snap_export_dir: Path, cwd: Path) -> None:
        (cwd / "patch.list").write_text("PATCH_1\nPATCH_2\nPATCH_3\nPATCH_4\n", encoding="utf-8")
        (cwd / "parms.mat").write_text("ok", encoding="utf-8")
        for index in range(1, 5):
            (cwd / f"PATCH_{index}").mkdir(parents=True, exist_ok=True)

    def fake_parallel(root: Path, patch_list_paths: list[Path], *, max_parallel_workers: int) -> None:
        parallel_patch_lists.append([path.name for path in patch_list_paths])
        assert max_parallel_workers == 2

    def fake_matlab_batch(script: str, cwd: Path, *, log_name: str = "matlab_batch.log") -> None:
        serial_scripts.append((script, log_name))
        export_dir = cwd / "export"
        export_dir.mkdir(parents=True, exist_ok=True)
        (export_dir / "ps_points.csv").write_text("point_id,velocity\n1,1.0\n", encoding="utf-8")
        (export_dir / "ps_timeseries.csv").write_text("point_id,epoch,metric_name,value\n", encoding="utf-8")

    monkeypatch.setattr(runner, "_run_mt_prep_snap", fake_mt_prep)
    monkeypatch.setattr(runner, "_run_parallel_patch_batches", fake_parallel)
    monkeypatch.setattr(runner, "_run_matlab_batch", fake_matlab_batch)

    context = RunContext.create(config, tmp_path)
    context.ensure_directories()
    manifest = _build_manifest()
    stack = config.stacks[0]
    snap_export_dir = context.snap_dir / manifest.stack_id / "stamps_export"
    _write_snap_export(snap_export_dir)

    outputs = runner.run_stack(context, manifest, stack, snap_export_dir)

    root_dir = context.stamps_dir / manifest.stack_id
    assert outputs.ps_points_csv.exists()
    assert parallel_patch_lists == [["patch_list_split_1", "patch_list_split_2"]]
    assert (root_dir / "patch_list_split_1").read_text(encoding="utf-8").splitlines() == ["PATCH_1", "PATCH_2"]
    assert (root_dir / "patch_list_split_2").read_text(encoding="utf-8").splitlines() == ["PATCH_3", "PATCH_4"]
    assert (root_dir / "PATCH_1" / "parms.mat").exists()
    assert (root_dir / "PATCH_4" / "parms.mat").exists()
    assert serial_scripts
    assert [log_name for _script, log_name in serial_scripts] == ["matlab_setparm.log", "matlab_batch.log"]
    assert "setparm('merge_resample_size',100,1);" in serial_scripts[0][0]
    assert "stamps(5,8,[],0,[],2);" in serial_scripts[1][0]
    assert "stamps(1,8);" not in serial_scripts[1][0]


def test_run_stack_executes_patch_batches_serially_when_parallel_limit_is_one(tmp_path, monkeypatch) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "psi_casablanca_slc_minimal.yaml")
    runner = StaMPSRunner(config)
    monkeypatch.setattr(runner, "validate_environment", lambda: None)

    serial_scripts: list[tuple[str, str]] = []

    def fake_mt_prep(_master_date: str, _snap_export_dir: Path, cwd: Path) -> None:
        (cwd / "patch.list").write_text("PATCH_1\nPATCH_2\nPATCH_3\nPATCH_4\n", encoding="utf-8")
        (cwd / "parms.mat").write_text("ok", encoding="utf-8")
        for index in range(1, 5):
            (cwd / f"PATCH_{index}").mkdir(parents=True, exist_ok=True)

    def fail_parallel(*_args, **_kwargs) -> None:
        raise AssertionError("parallel patch batches should not run when max_parallel_patch_workers=1")

    def fake_matlab_batch(script: str, cwd: Path, *, log_name: str = "matlab_batch.log") -> None:
        serial_scripts.append((script, log_name))
        export_dir = cwd / "export"
        if "stamps(5,8,[],0,[],2);" in script:
            export_dir.mkdir(parents=True, exist_ok=True)
            (export_dir / "ps_points.csv").write_text("point_id,velocity\n1,1.0\n", encoding="utf-8")
            (export_dir / "ps_timeseries.csv").write_text("point_id,epoch,metric_name,value\n", encoding="utf-8")

    monkeypatch.setattr(runner, "_run_mt_prep_snap", fake_mt_prep)
    monkeypatch.setattr(runner, "_run_parallel_patch_batches", fail_parallel)
    monkeypatch.setattr(runner, "_run_matlab_batch", fake_matlab_batch)

    context = RunContext.create(config, tmp_path)
    context.ensure_directories()
    manifest = _build_manifest()
    stack = config.stacks[0]
    snap_export_dir = context.snap_dir / manifest.stack_id / "stamps_export"
    _write_snap_export(snap_export_dir)

    outputs = runner.run_stack(context, manifest, stack, snap_export_dir)

    assert outputs.ps_points_csv.exists()
    assert [log_name for _script, log_name in serial_scripts] == [
        "matlab_setparm.log",
        "matlab_patch_worker_1.log",
        "matlab_patch_worker_2.log",
        "matlab_patch_worker_3.log",
        "matlab_patch_worker_4.log",
        "matlab_batch.log",
    ]
    assert "setparm('merge_resample_size',100,1);" in serial_scripts[0][0]
    assert "stamps(1,5,[],0,'patch_list_split_1',1);" in serial_scripts[1][0]
    assert "stamps(1,5,[],0,'patch_list_split_4',1);" in serial_scripts[4][0]
    assert "stamps(5,8,[],0,[],2);" in serial_scripts[5][0]
    assert "stamps(1,8);" not in "\n".join(script for script, _log_name in serial_scripts)


def test_parallel_patch_failure_terminates_siblings(tmp_path, monkeypatch) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "psi_casablanca_slc_minimal.yaml")
    config.stamps.max_parallel_patch_workers = 2
    runner = StaMPSRunner(config)
    monkeypatch.setattr(runner, "_environment", lambda: {})

    class FakeProcess:
        def __init__(self, poll_sequence: list[int | None], pid: int) -> None:
            self._poll_sequence = list(poll_sequence)
            self.pid = pid

        def poll(self) -> int | None:
            if self._poll_sequence:
                return self._poll_sequence.pop(0)
            return None

        def wait(self, timeout: float | None = None) -> int:
            return 0

    workers = [
        _PatchWorkerProcess(
            index=1,
            patch_list_path=tmp_path / "patch_list_split_1",
            command=["worker-1"],
            process=FakeProcess([9], 101),
        ),
        _PatchWorkerProcess(
            index=2,
            patch_list_path=tmp_path / "patch_list_split_2",
            command=["worker-2"],
            process=FakeProcess([None, None], 102),
        ),
    ]

    spawn_index = {"value": 0}
    terminated: list[tuple[list[int], int | None]] = []

    def fake_spawn(_root: Path, *, patch_list_path: Path, index: int, env: dict[str, str]) -> _PatchWorkerProcess:
        worker = workers[spawn_index["value"]]
        spawn_index["value"] += 1
        return worker

    def fake_terminate(active_workers: list[_PatchWorkerProcess], *, failed_worker: _PatchWorkerProcess | None = None) -> None:
        terminated.append(([worker.index for worker in active_workers], failed_worker.index if failed_worker is not None else None))

    monkeypatch.setattr(runner, "_spawn_patch_worker", fake_spawn)
    monkeypatch.setattr(runner, "_terminate_patch_workers", fake_terminate)

    patch_lists = [tmp_path / "patch_list_split_1", tmp_path / "patch_list_split_2"]

    with pytest.raises(subprocess.CalledProcessError, match="worker-1"):
        runner._run_parallel_patch_batches(tmp_path, patch_lists, max_parallel_workers=2)

    assert terminated[0] == ([2], 1)


def test_run_stack_reuses_completed_patch_workspace_for_merge_rerun(tmp_path, monkeypatch) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "psi_casablanca_slc_minimal.yaml")
    runner = StaMPSRunner(config)
    monkeypatch.setattr(runner, "validate_environment", lambda: None)
    monkeypatch.setattr(runner, "_run_mt_prep_snap", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("mt_prep_snap should not rerun")))
    monkeypatch.setattr(
        runner,
        "_run_serial_patch_batches",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("patch batches should not rerun")),
    )
    monkeypatch.setattr(
        runner,
        "_run_parallel_patch_batches",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("parallel patch batches should not rerun")),
    )

    matlab_scripts: list[tuple[str, str]] = []

    def fake_matlab_batch(script: str, cwd: Path, *, log_name: str = "matlab_batch.log") -> None:
        matlab_scripts.append((script, log_name))
        if "stamps(5,8,[],0,[],2);" in script:
            export_dir = cwd / "export"
            export_dir.mkdir(parents=True, exist_ok=True)
            (export_dir / "ps_points.csv").write_text("point_id,velocity\n1,1.0\n", encoding="utf-8")
            (export_dir / "ps_timeseries.csv").write_text("point_id,epoch,metric_name,value\n", encoding="utf-8")

    monkeypatch.setattr(runner, "_run_matlab_batch", fake_matlab_batch)

    context = RunContext.create(config, tmp_path)
    context.ensure_directories()
    manifest = _build_manifest()
    stack = config.stacks[0]
    root_dir = context.stamps_dir / manifest.stack_id
    root_dir.mkdir(parents=True, exist_ok=True)
    (root_dir / "patch.list").write_text("PATCH_1\nPATCH_2\n", encoding="utf-8")
    (root_dir / "patch.list_old").write_text("stale\n", encoding="utf-8")
    (root_dir / "parms.mat").write_text("ok", encoding="utf-8")
    (root_dir / "ps2.mat").write_text("stale merged artifact", encoding="utf-8")
    (root_dir / "export").mkdir(parents=True, exist_ok=True)
    (root_dir / "export" / "partial.txt").write_text("stale", encoding="utf-8")
    for patch_name in ("PATCH_1", "PATCH_2"):
        patch_dir = root_dir / patch_name
        patch_dir.mkdir(parents=True, exist_ok=True)
        for filename in ("ps2.mat", "pm2.mat", "rc2.mat", "bp2.mat", "patch_noover.in", "no_ps_info.mat"):
            (patch_dir / filename).write_text("ok", encoding="utf-8")
        (patch_dir / "STAMPS.log").write_text("PS_CORRECT_PHASE Finished\n", encoding="utf-8")

    snap_export_dir = context.snap_dir / manifest.stack_id / "stamps_export"
    _write_snap_export(snap_export_dir)

    outputs = runner.run_stack(context, manifest, stack, snap_export_dir)

    assert outputs.ps_points_csv.exists()
    assert matlab_scripts
    assert [log_name for _script, log_name in matlab_scripts] == ["matlab_setparm.log", "matlab_batch.log"]
    assert "setparm('merge_resample_size',100,1);" in matlab_scripts[0][0]
    assert "stamps(5,8,[],0,[],2);" in matlab_scripts[1][0]
    assert not (root_dir / "patch.list_old").exists()
    assert not (root_dir / "ps2.mat").exists()
    assert (root_dir / "PATCH_1").exists()
    assert (root_dir / "PATCH_2").exists()
    assert any(record.category == "stamps_merge_stage" for record in outputs.cleanup_records)


def test_run_stack_reuses_step6_workspace_for_late_stage_rerun(tmp_path, monkeypatch) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "psi_casablanca_slc_minimal.yaml")
    runner = StaMPSRunner(config)
    monkeypatch.setattr(runner, "validate_environment", lambda: None)
    monkeypatch.setattr(runner, "_run_mt_prep_snap", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("mt_prep_snap should not rerun")))
    monkeypatch.setattr(
        runner,
        "_run_serial_patch_batches",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("patch batches should not rerun")),
    )
    monkeypatch.setattr(
        runner,
        "_run_parallel_patch_batches",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("parallel patch batches should not rerun")),
    )

    matlab_scripts: list[tuple[str, str]] = []

    def fake_matlab_batch(script: str, cwd: Path, *, log_name: str = "matlab_batch.log") -> None:
        matlab_scripts.append((script, log_name))
        export_dir = cwd / "export"
        export_dir.mkdir(parents=True, exist_ok=True)
        (export_dir / "ps_points.csv").write_text("point_id,velocity\n1,1.0\n", encoding="utf-8")
        (export_dir / "ps_timeseries.csv").write_text("point_id,epoch,metric_name,value\n", encoding="utf-8")

    monkeypatch.setattr(runner, "_run_matlab_batch", fake_matlab_batch)

    context = RunContext.create(config, tmp_path)
    context.ensure_directories()
    manifest = _build_manifest()
    stack = config.stacks[0]
    root_dir = context.stamps_dir / manifest.stack_id
    root_dir.mkdir(parents=True, exist_ok=True)
    (root_dir / "patch.list").write_text("PATCH_1\nPATCH_2\n", encoding="utf-8")
    (root_dir / "parms.mat").write_text("ok", encoding="utf-8")
    (root_dir / "STAMPS.log").write_text("PS_UNWRAP        Finished\n", encoding="utf-8")
    for filename in ("ps2.mat", "ph2.mat", "rc2.mat", "pm2.mat", "bp2.mat", "la2.mat", "inc2.mat", "hgt2.mat", "ifgstd2.mat", "phuw2.mat", "uw_grid.mat", "uw_interp.mat", "uw_space_time.mat"):
        (root_dir / filename).write_text("ok", encoding="utf-8")
    (root_dir / "scla2.mat").write_text("stale", encoding="utf-8")
    (root_dir / "export").mkdir(parents=True, exist_ok=True)
    (root_dir / "export" / "partial.txt").write_text("stale", encoding="utf-8")

    snap_export_dir = context.snap_dir / manifest.stack_id / "stamps_export"
    _write_snap_export(snap_export_dir)

    outputs = runner.run_stack(context, manifest, stack, snap_export_dir)

    assert outputs.ps_points_csv.exists()
    assert [log_name for _script, log_name in matlab_scripts] == ["matlab_batch.log"]
    assert "stamps(7,8,[],0,[],2);" in matlab_scripts[0][0]
    assert not (root_dir / "scla2.mat").exists()
    assert any(record.category == "stamps_late_stage" for record in outputs.cleanup_records)


def test_run_stack_reuses_step7_workspace_for_step8_rerun(tmp_path, monkeypatch) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "psi_casablanca_slc_minimal.yaml")
    runner = StaMPSRunner(config)
    monkeypatch.setattr(runner, "validate_environment", lambda: None)
    monkeypatch.setattr(runner, "_run_mt_prep_snap", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("mt_prep_snap should not rerun")))
    monkeypatch.setattr(
        runner,
        "_run_serial_patch_batches",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("patch batches should not rerun")),
    )
    monkeypatch.setattr(
        runner,
        "_run_parallel_patch_batches",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("parallel patch batches should not rerun")),
    )

    matlab_scripts: list[tuple[str, str]] = []

    def fake_matlab_batch(script: str, cwd: Path, *, log_name: str = "matlab_batch.log") -> None:
        matlab_scripts.append((script, log_name))
        export_dir = cwd / "export"
        export_dir.mkdir(parents=True, exist_ok=True)
        (export_dir / "ps_points.csv").write_text("point_id,velocity\n1,1.0\n", encoding="utf-8")
        (export_dir / "ps_timeseries.csv").write_text("point_id,epoch,metric_name,value\n", encoding="utf-8")

    monkeypatch.setattr(runner, "_run_matlab_batch", fake_matlab_batch)

    context = RunContext.create(config, tmp_path)
    context.ensure_directories()
    manifest = _build_manifest()
    stack = config.stacks[0]
    root_dir = context.stamps_dir / manifest.stack_id
    root_dir.mkdir(parents=True, exist_ok=True)
    (root_dir / "patch.list").write_text("PATCH_1\nPATCH_2\n", encoding="utf-8")
    (root_dir / "parms.mat").write_text("ok", encoding="utf-8")
    (root_dir / "STAMPS.log").write_text("PS_SMOOTH_SCLA   Finished\n", encoding="utf-8")
    for filename in (
        "ps2.mat",
        "ph2.mat",
        "rc2.mat",
        "pm2.mat",
        "bp2.mat",
        "la2.mat",
        "inc2.mat",
        "hgt2.mat",
        "ifgstd2.mat",
        "phuw2.mat",
        "uw_grid.mat",
        "uw_interp.mat",
        "uw_space_time.mat",
        "scla2.mat",
        "scla_smooth2.mat",
    ):
        (root_dir / filename).write_text("ok", encoding="utf-8")
    (root_dir / "scnfilt.1.node").write_text("stale", encoding="utf-8")
    (root_dir / "triangle_scn.log").write_text("stale", encoding="utf-8")
    (root_dir / "export").mkdir(parents=True, exist_ok=True)
    (root_dir / "export" / "partial.txt").write_text("stale", encoding="utf-8")

    snap_export_dir = context.snap_dir / manifest.stack_id / "stamps_export"
    _write_snap_export(snap_export_dir)

    outputs = runner.run_stack(context, manifest, stack, snap_export_dir)

    assert outputs.ps_points_csv.exists()
    assert [log_name for _script, log_name in matlab_scripts] == ["matlab_batch.log"]
    assert "stamps(8,8,[],0,[],2);" in matlab_scripts[0][0]
    assert not (root_dir / "scnfilt.1.node").exists()
    assert not (root_dir / "triangle_scn.log").exists()
    assert any(record.category == "stamps_step8_stage" for record in outputs.cleanup_records)


def test_run_stack_reuses_step8_workspace_for_export_only_rerun(tmp_path, monkeypatch) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "psi_casablanca_slc_minimal.yaml")
    runner = StaMPSRunner(config)
    monkeypatch.setattr(runner, "validate_environment", lambda: None)
    monkeypatch.setattr(runner, "_run_mt_prep_snap", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("mt_prep_snap should not rerun")))
    monkeypatch.setattr(
        runner,
        "_run_serial_patch_batches",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("patch batches should not rerun")),
    )
    monkeypatch.setattr(
        runner,
        "_run_parallel_patch_batches",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("parallel patch batches should not rerun")),
    )

    matlab_scripts: list[tuple[str, str]] = []

    def fake_matlab_batch(script: str, cwd: Path, *, log_name: str = "matlab_batch.log") -> None:
        matlab_scripts.append((script, log_name))
        export_dir = cwd / "export"
        export_dir.mkdir(parents=True, exist_ok=True)
        (export_dir / "ps_points.csv").write_text("point_id,velocity\n1,1.0\n", encoding="utf-8")
        (export_dir / "ps_timeseries.csv").write_text("point_id,epoch,metric_name,value\n", encoding="utf-8")

    monkeypatch.setattr(runner, "_run_matlab_batch", fake_matlab_batch)

    context = RunContext.create(config, tmp_path)
    context.ensure_directories()
    manifest = _build_manifest()
    stack = config.stacks[0]
    root_dir = context.stamps_dir / manifest.stack_id
    root_dir.mkdir(parents=True, exist_ok=True)
    (root_dir / "patch.list").write_text("PATCH_1\nPATCH_2\n", encoding="utf-8")
    (root_dir / "parms.mat").write_text("ok", encoding="utf-8")
    (root_dir / "STAMPS.log").write_text("PS_SCN_FILT      Finished\n", encoding="utf-8")
    for filename in (
        "ps2.mat",
        "ph2.mat",
        "rc2.mat",
        "pm2.mat",
        "bp2.mat",
        "la2.mat",
        "inc2.mat",
        "hgt2.mat",
        "ifgstd2.mat",
        "phuw2.mat",
        "uw_grid.mat",
        "uw_interp.mat",
        "uw_space_time.mat",
        "scla2.mat",
        "scla_smooth2.mat",
        "scn2.mat",
    ):
        (root_dir / filename).write_text("ok", encoding="utf-8")
    (root_dir / "export").mkdir(parents=True, exist_ok=True)
    (root_dir / "export" / "partial.txt").write_text("stale", encoding="utf-8")

    snap_export_dir = context.snap_dir / manifest.stack_id / "stamps_export"
    _write_snap_export(snap_export_dir)

    outputs = runner.run_stack(context, manifest, stack, snap_export_dir)

    assert outputs.ps_points_csv.exists()
    assert [log_name for _script, log_name in matlab_scripts] == ["matlab_batch.log"]
    assert "stamps(" not in matlab_scripts[0][0]
    assert "feval('export_ps_points');" in matlab_scripts[0][0]
    assert not (root_dir / "export" / "partial.txt").exists()
    assert any(record.category == "stamps_export_tail" for record in outputs.cleanup_records)
