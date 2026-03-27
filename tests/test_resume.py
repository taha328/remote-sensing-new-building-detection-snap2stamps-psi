from __future__ import annotations

from pathlib import Path

import pytest

from casablanca_builtup.io import read_json
from casablanca_builtup.pipeline import run_pipeline
import casablanca_builtup.pipeline as pipeline_module

from tests.helpers import install_fake_pipeline, write_test_config


def test_resume_reuses_existing_detection_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = write_test_config(tmp_path)
    install_fake_pipeline(monkeypatch, pipeline_module, with_s2=True)

    context = run_pipeline(config_path, stop_after="detect_s1")

    def fail_detect(*args, **kwargs):
        raise AssertionError("detect_s1_change should not be called when artifacts already exist")

    monkeypatch.setattr(pipeline_module, "detect_s1_change", fail_detect)
    resumed_context = run_pipeline(config_path, stop_after="detect_s1", run_dir=context.root)
    report = read_json(resumed_context.reports_dir / "run_report.json")

    assert resumed_context.root == context.root
    assert report["periods"][0]["s1_detection_source"] == "artifact"
    assert report["stages"]["detect_s1"]["status"] == "completed"


def test_resume_ignores_zero_byte_s2_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = write_test_config(tmp_path)
    install_fake_pipeline(monkeypatch, pipeline_module, with_s2=True)

    context = run_pipeline(config_path, stop_after="build_composites")
    broken_path = context.rasters_dir / "2023_2025_s2_before_blue.tif"
    broken_path.write_bytes(b"")

    calls = {"count": 0}
    original = pipeline_module.build_s2_composite

    def tracked_build_s2(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(pipeline_module, "build_s2_composite", tracked_build_s2)
    resumed_context = run_pipeline(config_path, stop_after="build_composites", run_dir=context.root)
    report = read_json(resumed_context.reports_dir / "run_report.json")

    assert resumed_context.root == context.root
    assert calls["count"] >= 1
    assert broken_path.exists()
    assert broken_path.stat().st_size > 0
    assert report["periods"][0]["s2_before_source"] == "computed"
