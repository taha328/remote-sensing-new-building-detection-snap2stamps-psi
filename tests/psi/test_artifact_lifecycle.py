from __future__ import annotations

from pathlib import Path

import pytest

from casablanca_psi import artifact_lifecycle
from casablanca_psi.artifact_lifecycle import CleanupWarning, cleanup_stamps_workspace, delete_paths, is_valid_snap_export_dir


def test_is_valid_snap_export_dir_requires_non_empty_required_directories(tmp_path) -> None:
    export_dir = tmp_path / "stamps_export"
    for name in ("rslc", "diff0", "geo"):
        (export_dir / name).mkdir(parents=True, exist_ok=True)

    assert is_valid_snap_export_dir(export_dir) is False

    for name in ("rslc", "diff0", "geo"):
        (export_dir / name / "ok.txt").write_text("ok")

    diff_dir = export_dir / "diff0"
    (diff_dir / "20230922_20230829.diff").write_text("ok")
    (diff_dir / "20230922_20230829.base").write_text(
        "initial_baseline(TCN): 0.0000000 NaN NaN\n"
        "initial_baseline_rate: 0.0000000 NaN NaN\n",
        encoding="utf-8",
    )

    assert is_valid_snap_export_dir(export_dir) is False

    (diff_dir / "20230922_20230829.base").write_text(
        "initial_baseline(TCN): 0.0000000 12.5000000 -3.2500000\n"
        "initial_baseline_rate: 0.0000000 0.5000000 0.1250000\n",
        encoding="utf-8",
    )

    assert is_valid_snap_export_dir(export_dir) is True


def test_delete_paths_removes_dimap_sidecar_data_directory(tmp_path) -> None:
    product = tmp_path / "product.dim"
    product.write_text("ok")
    data_dir = product.with_suffix(".data")
    data_dir.mkdir()
    (data_dir / "band.img").write_text("ok")

    records = delete_paths(
        [product],
        category="snap_prepared",
        checkpoint="unit_test",
        reason="test cleanup",
    )

    assert not product.exists()
    assert not data_dir.exists()
    assert records
    assert records[0].bytes_reclaimed > 0


def test_delete_paths_retries_hidden_file_cleanup_before_warning(tmp_path, monkeypatch) -> None:
    directory = tmp_path / "stale_dir"
    directory.mkdir()
    (directory / ".DS_Store").write_text("finder", encoding="utf-8")
    calls = {"count": 0}
    original_rmtree = artifact_lifecycle.shutil.rmtree

    def flaky_rmtree(path, ignore_errors=False):
        if Path(path) == directory and calls["count"] == 0:
            calls["count"] += 1
            raise OSError(66, "Directory not empty")
        return original_rmtree(path, ignore_errors=ignore_errors)

    monkeypatch.setattr(artifact_lifecycle.shutil, "rmtree", flaky_rmtree)
    warnings: list[CleanupWarning] = []

    records = delete_paths(
        [directory],
        category="snap_intermediate",
        checkpoint="unit_test",
        reason="test cleanup",
        best_effort=True,
        retry_hidden_files=True,
        warning_records=warnings,
    )

    assert not directory.exists()
    assert records
    assert warnings == []


def test_delete_paths_keeps_strict_failures_fatal(tmp_path, monkeypatch) -> None:
    directory = tmp_path / "required_dir"
    directory.mkdir()

    def always_fail(*_args, **_kwargs):
        raise OSError(66, "Directory not empty")

    monkeypatch.setattr(artifact_lifecycle.shutil, "rmtree", always_fail)

    with pytest.raises(OSError, match="Directory not empty"):
        delete_paths(
            [directory],
            category="snap_intermediate",
            checkpoint="unit_test",
            reason="required cleanup must remain fatal",
        )


def test_delete_paths_best_effort_records_warning_when_cleanup_still_fails(tmp_path, monkeypatch) -> None:
    directory = tmp_path / "stale_dir"
    directory.mkdir()
    (directory / ".DS_Store").write_text("finder", encoding="utf-8")

    def always_fail(*_args, **_kwargs):
        raise OSError(66, "Directory not empty")

    monkeypatch.setattr(artifact_lifecycle.shutil, "rmtree", always_fail)
    warnings: list[CleanupWarning] = []

    records = delete_paths(
        [directory],
        category="snap_intermediate",
        checkpoint="stamps_export_validated",
        reason="validated export already exists",
        best_effort=True,
        retry_hidden_files=True,
        warning_records=warnings,
    )

    assert directory.exists()
    assert records == []
    assert len(warnings) == 1
    assert warnings[0].checkpoint == "stamps_export_validated"
    assert "Directory not empty" in warnings[0].message


def test_cleanup_stamps_workspace_preserves_export_csvs_only(tmp_path) -> None:
    root = tmp_path / "stamps"
    export_dir = root / "export"
    export_dir.mkdir(parents=True)
    points_csv = export_dir / "ps_points.csv"
    ts_csv = export_dir / "ps_timeseries.csv"
    points_csv.write_text("point_id,velocity\n1,1.0\n", encoding="utf-8")
    ts_csv.write_text("point_id,epoch,metric_name,value\n", encoding="utf-8")
    (root / "PATCH_1" / "patch.in").mkdir(parents=True)
    (root / "PATCH_1" / "patch.in" / "tmp.txt").write_text("tmp")
    (root / "geo").mkdir(parents=True)
    (root / "geo" / "geomap.txt").write_text("tmp")
    (export_dir / "debug.txt").write_text("tmp")

    records = cleanup_stamps_workspace(root, keep_paths=(points_csv, ts_csv))

    assert records
    assert points_csv.exists()
    assert ts_csv.exists()
    assert not (root / "PATCH_1").exists()
    assert not (root / "geo").exists()
    assert not (export_dir / "debug.txt").exists()
