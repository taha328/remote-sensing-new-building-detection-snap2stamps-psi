from __future__ import annotations

from datetime import date
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from casablanca_psi.cdpsi import build_cdpsi_artifacts, plan_cdpsi_stack
from casablanca_psi.config import OrbitStackConfig, PsiDetectionConfig
from casablanca_psi.manifests import SlcScene, StackManifest
from casablanca_psi.psi_results import load_ps_points


def _scene(acquisition_date: str, index: int) -> SlcScene:
    return SlcScene(
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


def _manifest(scene_dates: list[str]) -> StackManifest:
    return StackManifest(
        stack_id="asc_rel147_vv",
        direction="ascending",
        relative_orbit=147,
        product_type="SLC",
        scenes=[_scene(value, index) for index, value in enumerate(scene_dates, start=1)],
    )


def _stack(master_date: str) -> OrbitStackConfig:
    return OrbitStackConfig(
        id="asc_rel147_vv",
        direction="ascending",
        relative_orbit=147,
        iw_swaths=("IW1", "IW2", "IW3"),
        polarization="VV",
        master_date=date.fromisoformat(master_date),
        min_scenes=5,
        scene_limit=5,
    )


def _points(frame: pd.DataFrame) -> gpd.GeoDataFrame:
    geometry = gpd.points_from_xy(frame["lon"], frame["lat"], crs="EPSG:4326")
    return gpd.GeoDataFrame(frame, geometry=geometry).to_crs("EPSG:32629")


def test_load_ps_points_preserves_cdpsi_pixel_keys(tmp_path: Path) -> None:
    csv_path = tmp_path / "ps_points.csv"
    csv_path.write_text(
        "\n".join(
            [
                "point_id,x_local_m,y_local_m,azimuth_index,range_index,lon,lat,temporal_coherence,scene_elevation_m,dem_error_phase_per_m,mean_velocity_mm_yr,master_day,n_ifg,n_image",
                "1,10.0,20.0,101,202,-7.60,33.57,0.85,12.0,0.01,0.1,739151,5,5",
            ]
        ),
        encoding="utf-8",
    )

    points = load_ps_points(csv_path)

    assert len(points) == 1
    assert int(points.iloc[0]["azimuth_index"]) == 101
    assert int(points.iloc[0]["range_index"]) == 202


def test_plan_cdpsi_stack_rejects_five_scene_stack_with_minimum_three_images_per_side() -> None:
    manifest = _manifest(["2023-08-29", "2023-09-10", "2023-09-22", "2023-10-04", "2023-10-16"])
    plan = plan_cdpsi_stack(manifest, _stack("2023-09-22"), PsiDetectionConfig())

    assert plan.breaks == ()
    with pytest.raises(ValueError, match="cannot support a published CDPSI run"):
        plan.validate()


def test_plan_cdpsi_stack_builds_valid_breaks_for_six_scene_stack() -> None:
    manifest = _manifest(["2023-08-17", "2023-08-29", "2023-09-10", "2023-09-22", "2023-10-04", "2023-10-16"])
    plan = plan_cdpsi_stack(manifest, _stack("2023-09-22"), PsiDetectionConfig())

    assert len(plan.breaks) == 1
    assert plan.breaks[0].break_after_date.isoformat() == "2023-09-10"
    assert plan.breaks[0].break_before_date.isoformat() == "2023-09-22"
    assert plan.breaks[0].front.stack_id.endswith("20230817_20230910")
    assert plan.breaks[0].back.stack_id.endswith("20230922_20231016")


def test_build_cdpsi_artifacts_computes_change_indices_from_exact_pixel_matches() -> None:
    config = PsiDetectionConfig()
    manifest = _manifest(["2023-08-17", "2023-08-29", "2023-09-10", "2023-09-22", "2023-10-04", "2023-10-16"])
    plan = plan_cdpsi_stack(manifest, _stack("2023-09-22"), config)

    stable_count = 40
    emergence_count = 5
    count = stable_count + emergence_count
    azimuth = np.arange(1, count + 1)
    ranges = np.arange(1001, 1001 + count)
    base = pd.DataFrame(
        {
            "point_id": np.arange(1, count + 1),
            "azimuth_index": azimuth,
            "range_index": ranges,
            "lon": -7.60 + np.arange(count) * 0.0001,
            "lat": 33.57 + np.arange(count) * 0.0001,
        }
    )

    complete = _points(base.assign(temporal_coherence=np.r_[np.full(stable_count, 0.82), np.full(emergence_count, 0.20)]))
    front = _points(base.assign(point_id=np.arange(101, 101 + count), temporal_coherence=np.r_[np.full(stable_count, 0.83), np.full(emergence_count, 0.18)]))
    back = _points(base.assign(point_id=np.arange(201, 201 + count), temporal_coherence=np.r_[np.full(stable_count, 0.81), np.full(emergence_count, 0.95)]))

    artifacts = build_cdpsi_artifacts(
        complete,
        plan,
        {
            plan.breaks[0].front.stack_id: front,
            plan.breaks[0].back.stack_id: back,
        },
        config,
    )

    assert len(artifacts.change_points) == count
    assert len(artifacts.emergence_points) == emergence_count
    assert set(artifacts.emergence_points["cdpsi_class"]) == {"emergence_candidate"}
    assert artifacts.emergence_points["ci_emergence"].min() > artifacts.emergence_points["ci_emergence_threshold"].max()


def test_export_script_contract_does_not_emit_placeholder_emergence_columns() -> None:
    script_path = Path("resources/stamps/export_ps_points.m")
    script = script_path.read_text(encoding="utf-8")

    assert "pre_stability_fraction = nan" not in script
    assert "post_stability_fraction = nan" not in script
    assert "residual_height_m = nan" not in script
    assert "azimuth_index" in script
    assert "range_index" in script
