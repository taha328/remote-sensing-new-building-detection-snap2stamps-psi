from __future__ import annotations

from pathlib import Path

import pytest

from aoi_builtup.io import read_json
from aoi_builtup.pipeline import run_pipeline
import aoi_builtup.pipeline as pipeline_module

from tests.helpers import install_fake_pipeline, write_test_config


def test_pipeline_marks_report_interrupted_on_keyboard_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = write_test_config(tmp_path)
    install_fake_pipeline(monkeypatch, pipeline_module, with_s2=True)

    def interrupt_s2(*args, **kwargs):
        raise KeyboardInterrupt("simulated interruption")

    monkeypatch.setattr(pipeline_module, "build_s2_composite", interrupt_s2)

    with pytest.raises(KeyboardInterrupt):
        run_pipeline(config_path)

    run_root = next((tmp_path / "project" / "runs").glob("synthetic-builtup-*/attempt-*"))
    report = read_json(run_root / "reports" / "run_report.json")

    assert report["status"] == "interrupted"
    assert report["stages"]["build_composites"]["status"] == "interrupted"
    assert report["error"]["type"] == "KeyboardInterrupt"
    assert report["periods"][0]["s2_before_source"] == "pending"
