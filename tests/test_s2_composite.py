from __future__ import annotations

from pathlib import Path

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from aoi_builtup.config import DaskConfig, ExportConfig, Sentinel2Config
from aoi_builtup.s2 import composite as composite_module


class _GeoBox:
    crs = "EPSG:32629"


def test_build_s2_composite_materializes_dask_stack(monkeypatch) -> None:
    time = np.array(["2023-01-01", "2023-01-11"], dtype="datetime64[ns]")
    coords = {"time": time, "y": [0, 1], "x": [0, 1]}
    reflectance = np.array(
        [
            [[1000, 2000], [3000, 4000]],
            [[2000, 3000], [4000, 5000]],
        ],
        dtype=np.uint16,
    )
    scl = np.array(
        [
            [[4, 4], [4, 4]],
            [[4, 4], [4, 4]],
        ],
        dtype=np.uint8,
    )
    dataset = xr.Dataset(
        {
            "B02": xr.DataArray(da.from_array(reflectance, chunks=(1, 1, 2)), dims=("time", "y", "x"), coords=coords),
            "B03": xr.DataArray(da.from_array(reflectance, chunks=(1, 1, 2)), dims=("time", "y", "x"), coords=coords),
            "B04": xr.DataArray(da.from_array(reflectance, chunks=(1, 1, 2)), dims=("time", "y", "x"), coords=coords),
            "B08": xr.DataArray(da.from_array(reflectance, chunks=(1, 1, 2)), dims=("time", "y", "x"), coords=coords),
            "B11": xr.DataArray(da.from_array(reflectance, chunks=(1, 1, 2)), dims=("time", "y", "x"), coords=coords),
            "SCL": xr.DataArray(da.from_array(scl, chunks=(1, 1, 2)), dims=("time", "y", "x"), coords=coords),
        }
    )

    monkeypatch.setattr(composite_module, "stac_load", lambda *args, **kwargs: dataset)

    result = composite_module.build_s2_composite(
        items=[object(), object()],
        geobox=_GeoBox(),
        config=Sentinel2Config(),
        dask_config=DaskConfig(enabled=True, chunks={"x": 2, "y": 1}),
        period_id="p1",
        phase="before",
    )

    assert set(result.data_vars) == {
        "blue",
        "green",
        "red",
        "nir",
        "swir1",
        "ndvi",
        "ndbi",
        "mndwi",
        "bsi",
        "clear_count",
        "valid_count",
        "clear_fraction",
        "valid_fraction",
    }
    assert result["blue"].dtype == np.float32
    assert result["clear_count"].dtype == np.uint16
    assert result["clear_count"].shape == (2, 2)
    assert not hasattr(result["blue"].data, "dask")


def test_build_s2_composite_reuses_staged_local_stack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    time = np.array(["2023-01-01", "2023-01-11"], dtype="datetime64[ns]")
    coords = {"time": time, "y": [0, 1], "x": [0, 1]}
    reflectance = np.array(
        [
            [[1000, 2000], [3000, 4000]],
            [[2000, 3000], [4000, 5000]],
        ],
        dtype=np.uint16,
    )
    scl = np.array(
        [
            [[4, 4], [4, 4]],
            [[4, 4], [4, 4]],
        ],
        dtype=np.uint8,
    )
    dataset = xr.Dataset(
        {
            "B02": xr.DataArray(da.from_array(reflectance, chunks=(1, 1, 2)), dims=("time", "y", "x"), coords=coords),
            "B03": xr.DataArray(da.from_array(reflectance, chunks=(1, 1, 2)), dims=("time", "y", "x"), coords=coords),
            "B04": xr.DataArray(da.from_array(reflectance, chunks=(1, 1, 2)), dims=("time", "y", "x"), coords=coords),
            "B08": xr.DataArray(da.from_array(reflectance, chunks=(1, 1, 2)), dims=("time", "y", "x"), coords=coords),
            "B11": xr.DataArray(da.from_array(reflectance, chunks=(1, 1, 2)), dims=("time", "y", "x"), coords=coords),
            "SCL": xr.DataArray(da.from_array(scl, chunks=(1, 1, 2)), dims=("time", "y", "x"), coords=coords),
        }
    )

    monkeypatch.setattr(composite_module, "stac_load", lambda *args, **kwargs: dataset)

    result = composite_module.build_s2_composite(
        items=[object(), object()],
        geobox=_GeoBox(),
        config=Sentinel2Config(),
        dask_config=DaskConfig(enabled=True, chunks={"x": 2, "y": 1}),
        export_config=ExportConfig(),
        stage_dir=tmp_path,
        period_id="p1",
        phase="before",
    )

    assert (tmp_path / "s2" / "p1" / "before" / "blue_stack.tif").exists()
    assert result["blue"].shape == (2, 2)

    def fail_stac_load(*args, **kwargs):
        raise AssertionError("stac_load should not be called when staged S2 stacks already exist")

    monkeypatch.setattr(composite_module, "stac_load", fail_stac_load)
    resumed = composite_module.build_s2_composite(
        items=[object(), object()],
        geobox=_GeoBox(),
        config=Sentinel2Config(),
        dask_config=DaskConfig(enabled=True, chunks={"x": 2, "y": 1}),
        export_config=ExportConfig(),
        stage_dir=tmp_path,
        period_id="p1",
        phase="before",
    )

    assert resumed["blue"].equals(result["blue"])


def test_build_s2_composite_restages_invalid_local_stack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    time = np.array(["2023-01-01", "2023-01-11"], dtype="datetime64[ns]")
    coords = {"time": time, "y": [0, 1], "x": [0, 1]}
    reflectance = np.array(
        [
            [[1000, 2000], [3000, 4000]],
            [[2000, 3000], [4000, 5000]],
        ],
        dtype=np.uint16,
    )
    scl = np.array(
        [
            [[4, 4], [4, 4]],
            [[4, 4], [4, 4]],
        ],
        dtype=np.uint8,
    )
    dataset = xr.Dataset(
        {
            "B02": xr.DataArray(da.from_array(reflectance, chunks=(1, 1, 2)), dims=("time", "y", "x"), coords=coords),
            "B03": xr.DataArray(da.from_array(reflectance, chunks=(1, 1, 2)), dims=("time", "y", "x"), coords=coords),
            "B04": xr.DataArray(da.from_array(reflectance, chunks=(1, 1, 2)), dims=("time", "y", "x"), coords=coords),
            "B08": xr.DataArray(da.from_array(reflectance, chunks=(1, 1, 2)), dims=("time", "y", "x"), coords=coords),
            "B11": xr.DataArray(da.from_array(reflectance, chunks=(1, 1, 2)), dims=("time", "y", "x"), coords=coords),
            "SCL": xr.DataArray(da.from_array(scl, chunks=(1, 1, 2)), dims=("time", "y", "x"), coords=coords),
        }
    )
    monkeypatch.setattr(composite_module, "stac_load", lambda *args, **kwargs: dataset)
    composite_module.build_s2_composite(
        items=[object(), object()],
        geobox=_GeoBox(),
        config=Sentinel2Config(),
        dask_config=DaskConfig(enabled=True, chunks={"x": 2, "y": 1}),
        export_config=ExportConfig(),
        stage_dir=tmp_path,
        period_id="p1",
        phase="before",
    )

    broken = tmp_path / "s2" / "p1" / "before" / "blue_stack.tif"
    broken.write_bytes(b"")

    calls = {"count": 0}

    def tracked_stac_load(*args, **kwargs):
        calls["count"] += 1
        return dataset

    monkeypatch.setattr(composite_module, "stac_load", tracked_stac_load)
    result = composite_module.build_s2_composite(
        items=[object(), object()],
        geobox=_GeoBox(),
        config=Sentinel2Config(),
        dask_config=DaskConfig(enabled=True, chunks={"x": 2, "y": 1}),
        export_config=ExportConfig(),
        stage_dir=tmp_path,
        period_id="p1",
        phase="before",
    )

    assert calls["count"] >= 1
    assert broken.exists()
    assert broken.stat().st_size > 0
    assert result["blue"].shape == (2, 2)
