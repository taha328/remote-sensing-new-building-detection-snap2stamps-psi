from __future__ import annotations

from pathlib import Path

from casablanca_psi.config import load_config
from casablanca_psi.run_context import RunContext


def test_load_psi_config() -> None:
    config = load_config(Path("configs/psi_casablanca_slc.yaml"))
    assert config.project == "Casablanca PSI New-Building Detection"
    assert len(config.stacks) == 2
    assert config.stacks[0].direction == "ascending"
    assert config.stamps.export_script == Path("resources/stamps/export_ps_points.m")
    assert config.acquisition.auth.access_token_env == "ACCESS_TOKEN"
    assert config.acquisition.auth.refresh_margin_seconds == 120
    assert config.acquisition.download_transport == "auto"
    assert config.acquisition.s3.endpoint_url == "https://eodata.dataspace.copernicus.eu"
    assert config.acquisition.s3.bucket == "eodata"
    assert config.artifact_lifecycle.enabled is True
    assert config.snap.gpt_vmoptions_path == Path("/Applications/esa-snap/bin/gpt.vmoptions")
    assert config.snap.user_dir == Path("data/cache/snap-gpt-userdir")
    assert config.snap.default_tile_size_px == 512
    assert config.stamps.range_patches == 2
    assert config.stamps.azimuth_patches == 2
    assert config.stamps.max_parallel_patch_workers == 2
    assert config.artifact_lifecycle.purge_master_prepared_after_pair is True
    assert config.artifact_lifecycle.purge_secondary_prepared_after_pair is True
    assert config.artifact_lifecycle.purge_pair_products_after_merged_coreg is True
    assert config.artifact_lifecycle.purge_prepared_after_coreg is True
    assert config.artifact_lifecycle.purge_snap_intermediates_after_export is True
    assert config.artifact_lifecycle.purge_snap_export_after_stamps is True
    assert config.artifact_lifecycle.purge_stamps_workspace_after_parse is True


def test_load_psi_config_applies_s3_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("CDSE_S3_ENDPOINT", "eodata.dataspace.copernicus.eu")
    monkeypatch.setenv("CDSE_S3_BUCKET", "custom-bucket")
    monkeypatch.setenv(
        "CDSE_S3_FALLBACK_ENDPOINTS",
        "https://eodata.dataspace.copernicus.eu, eodata.ams.dataspace.copernicus.eu, eodata.ams.dataspace.copernicus.eu",
    )

    config = load_config(Path("configs/psi_casablanca_slc.yaml"))

    assert config.acquisition.s3.endpoint_url == "https://eodata.dataspace.copernicus.eu"
    assert config.acquisition.s3.bucket == "custom-bucket"
    assert config.acquisition.s3.fallback_endpoint_urls == ("https://eodata.ams.dataspace.copernicus.eu",)


def test_run_context_uses_explicit_run_dir_identity(tmp_path) -> None:
    config = load_config(Path("configs/psi_casablanca_slc.yaml"))
    project_root = tmp_path
    run_dir = project_root / "runs_psi" / "existing-group" / "attempt-007"
    context = RunContext.create(config, project_root, run_dir=run_dir)
    assert context.group_id == "existing-group"
    assert context.attempt_id == "attempt-007"
    assert context.run_id == "existing-group-attempt-007"


def test_load_minimal_psi_config_uses_single_capped_stack() -> None:
    config = load_config(Path("configs/psi_casablanca_slc_minimal.yaml"))

    assert len(config.stacks) == 1
    assert config.stacks[0].id == "asc_rel147_vv"
    assert config.stacks[0].min_scenes == 5
    assert config.stacks[0].scene_limit == 5
    assert config.snap.cache_size_mb == 2048
    assert config.snap.java_options == ("-Xms2G", "-Xmx8G")
    assert config.snap.clear_tile_cache_after_row is True
    assert config.snap.gpt_vmoptions_path == Path("/Applications/esa-snap/bin/gpt.vmoptions")
    assert config.snap.user_dir == Path("data/cache/snap-gpt-userdir")
    assert config.snap.default_tile_size_px == 512
    assert config.snap.workers == 1
    assert config.stamps.range_patches == 4
    assert config.stamps.azimuth_patches == 3
    assert config.stamps.max_parallel_patch_workers == 1
