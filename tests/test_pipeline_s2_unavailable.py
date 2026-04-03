from __future__ import annotations

from pathlib import Path

import pytest

from aoi_builtup.io import read_json
from aoi_builtup.pipeline import run_pipeline
import aoi_builtup.pipeline as pipeline_module

from tests.helpers import install_fake_pipeline, write_test_config


def test_pipeline_keeps_s1_when_s2_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = write_test_config(tmp_path, allow_unavailable=True)
    install_fake_pipeline(monkeypatch, pipeline_module, with_s2=False)

    context = run_pipeline(config_path, stop_after="refine_s2")
    report = read_json(context.reports_dir / "run_report.json")
    period = report["periods"][0]

    assert period["s2_items_available"] is False
    assert period["s2_support_mode"] == "unavailable_keep_s1"
    assert period["s2_reliable_fraction"] == 0.0
    assert period["refined_pixels"] == period["s1_candidate_pixels"]
    assert report["stages"]["refine_s2"]["status"] == "completed"
