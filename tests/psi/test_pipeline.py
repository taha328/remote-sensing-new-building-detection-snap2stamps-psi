from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from casablanca_psi.artifact_lifecycle import CleanupWarning
from casablanca_psi.manifests import SlcScene, StackManifest
from casablanca_psi.pipeline import run_pipeline


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


def test_run_pipeline_persists_runtime_policies(tmp_path, monkeypatch) -> None:
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
    assert report["cleanup_summary"]["warning_count"] == 1
    assert report["cleanup_warnings"][0]["checkpoint"] == "stamps_export_validated"
    assert "Directory not empty" in report["cleanup_warnings"][0]["message"]
