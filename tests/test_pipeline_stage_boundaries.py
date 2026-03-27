from __future__ import annotations

from pathlib import Path

import pytest

from casablanca_builtup.io import read_json
from casablanca_builtup.pipeline import run_pipeline
import casablanca_builtup.pipeline as pipeline_module

from tests.helpers import install_fake_pipeline, write_test_config


@pytest.mark.parametrize(
    ("stop_after", "expected_statuses", "expected_paths", "unexpected_paths"),
    [
        (
            "build_composites",
            {
                "acquire": "completed",
                "build_composites": "completed",
                "detect_s1": "pending",
                "refine_s2": "pending",
                "polygonize": "pending",
                "export": "pending",
            },
            ["2023_2025_s1_before_vv.tif", "2023_2025_s2_before_ndvi.tif"],
            ["2023_2025_s1_candidate.tif", "zone_mask.tif"],
        ),
        (
            "detect_s1",
            {
                "acquire": "completed",
                "build_composites": "completed",
                "detect_s1": "completed",
                "refine_s2": "pending",
                "polygonize": "pending",
                "export": "pending",
            },
            ["2023_2025_s1_candidate.tif"],
            ["2023_2025_refined_mask.tif", "zone_mask.tif"],
        ),
        (
            "refine_s2",
            {
                "acquire": "completed",
                "build_composites": "completed",
                "detect_s1": "completed",
                "refine_s2": "completed",
                "polygonize": "pending",
                "export": "pending",
            },
            ["2023_2025_refined_mask.tif", "2023_2025_fusion_decision.tif"],
            ["zone_mask.tif", "builtup_zones.parquet"],
        ),
        (
            "polygonize",
            {
                "acquire": "completed",
                "build_composites": "completed",
                "detect_s1": "completed",
                "refine_s2": "completed",
                "polygonize": "completed",
                "export": "pending",
            },
            ["zone_mask.tif"],
            ["builtup_zones.parquet"],
        ),
    ],
)
def test_pipeline_stage_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stop_after: str,
    expected_statuses: dict[str, str],
    expected_paths: list[str],
    unexpected_paths: list[str],
) -> None:
    config_path = write_test_config(tmp_path)
    install_fake_pipeline(monkeypatch, pipeline_module, with_s2=True)

    context = run_pipeline(config_path, stop_after=stop_after)
    report = read_json(context.reports_dir / "run_report.json")

    for stage, expected_status in expected_statuses.items():
        assert report["stages"][stage]["status"] == expected_status

    for relative_path in expected_paths:
        candidate_path = context.rasters_dir / relative_path
        if not candidate_path.exists():
            candidate_path = context.vectors_dir / relative_path
        assert candidate_path.exists()

    for relative_path in unexpected_paths:
        candidate_path = context.rasters_dir / relative_path
        if not candidate_path.exists():
            candidate_path = context.vectors_dir / relative_path
        assert not candidate_path.exists()
