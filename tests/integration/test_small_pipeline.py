from __future__ import annotations

from pathlib import Path

import pytest

from casablanca_builtup.evaluation import evaluate_run
from casablanca_builtup.io import read_json
from casablanca_builtup.pipeline import run_pipeline
import casablanca_builtup.pipeline as pipeline_module

from tests.helpers import install_fake_pipeline, write_test_config


def test_small_pipeline_full_run_and_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = write_test_config(tmp_path)
    install_fake_pipeline(monkeypatch, pipeline_module, with_s2=True)

    context = run_pipeline(config_path)
    report = read_json(context.reports_dir / "run_report.json")

    assert report["status"] == "completed"
    assert report["polygon_count"] >= 1
    assert (context.vectors_dir / "zone_polygons.parquet").exists()
    assert (context.vectors_dir / "builtup_zones.parquet").exists()
    assert (context.vectors_dir / "builtup_zones.gpkg").exists()

    evaluation = evaluate_run(
        run_dir=context.root,
        reference_raster=context.rasters_dir / "cumulative_refined.tif",
    )

    assert evaluation["against_reference"]["iou"] == 1.0
    assert (context.reports_dir / "evaluation" / "evaluation_summary.json").exists()
