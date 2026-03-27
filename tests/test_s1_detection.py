import numpy as np
import xarray as xr

from casablanca_builtup.config import DetectionConfig
from casablanca_builtup.s1.detection import detect_s1_change


def _dataset(vv: np.ndarray, vh: np.ndarray) -> xr.Dataset:
    coords = {"y": np.arange(vv.shape[0]) * -10.0, "x": np.arange(vv.shape[1]) * 10.0}
    return xr.Dataset(
        {
            "vv": xr.DataArray(vv, dims=("y", "x"), coords=coords),
            "vh": xr.DataArray(vh, dims=("y", "x"), coords=coords),
        }
    )


def test_detect_s1_change_keeps_cluster() -> None:
    before_vv = np.ones((8, 8), dtype=np.float32)
    before_vh = np.ones((8, 8), dtype=np.float32)
    after_vv = before_vv.copy()
    after_vh = before_vh.copy()
    after_vv[2:5, 2:5] = 8.0
    after_vh[2:5, 2:5] = 1.3

    artifacts = detect_s1_change(
        _dataset(before_vv, before_vh),
        _dataset(after_vv, after_vh),
        DetectionConfig(),
        resolution_m=10.0,
    )

    assert int(artifacts.candidate.sum()) >= 9
    assert bool(artifacts.candidate.sel(y=-20.0, x=20.0).item()) is True
    assert bool(artifacts.candidate.sel(y=0.0, x=0.0).item()) is False


def test_detect_s1_change_removes_single_pixel_noise() -> None:
    before = np.ones((8, 8), dtype=np.float32)
    after = before.copy()
    after[3, 3] = 8.0

    artifacts = detect_s1_change(
        _dataset(before, before),
        _dataset(after, after),
        DetectionConfig(min_connected_pixels=4),
        resolution_m=10.0,
    )

    assert int(artifacts.candidate.sum()) == 0
