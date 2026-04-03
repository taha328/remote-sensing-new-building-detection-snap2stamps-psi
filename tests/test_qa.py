import xarray as xr

from aoi_builtup.qa import binary_area_ha, decision_histogram


def test_binary_area_ha_uses_resolution() -> None:
    mask = xr.DataArray([[1, 0], [1, 1]], dims=("y", "x"))
    assert binary_area_ha(mask.astype(bool), resolution_m=10.0) == 0.03


def test_decision_histogram_uses_named_buckets() -> None:
    decision = xr.DataArray([[0, 1, 2], [3, 4, 4]], dims=("y", "x"))
    histogram = decision_histogram(decision)

    assert histogram["not_candidate"] == 1
    assert histogram["kept_s1_s2_unreliable"] == 1
    assert histogram["kept_s1_s2_supported"] == 1
    assert histogram["kept_strong_s1_override"] == 1
    assert histogram["dropped_s2_reliable_unsupported"] == 2
