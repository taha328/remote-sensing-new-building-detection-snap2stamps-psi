import numpy as np
import xarray as xr

from casablanca_builtup.config import Sentinel2Config
from casablanca_builtup.fusion import apply_soft_refinement
from casablanca_builtup.s1.detection import S1DetectionArtifacts
from casablanca_builtup.s2.refinement import S2SupportArtifacts, build_s2_support, build_unavailable_s2_support_like


def _data(values: list[float], name: str) -> xr.DataArray:
    coords = {"y": [0.0], "x": [0.0, 10.0, 20.0]}
    return xr.DataArray(np.asarray([values]), dims=("y", "x"), coords=coords, name=name)


def test_soft_refinement_keeps_unreliable_and_strong_override() -> None:
    s1 = S1DetectionArtifacts(
        pvalue_vv=_data([0.02, 0.02, 0.0005], "pvalue_vv"),
        ratio_vv=_data([1.6, 1.6, 2.2], "ratio_vv"),
        ratio_vh=_data([1.06, 1.06, 1.2], "ratio_vh"),
        candidate=_data([1, 1, 1], "construction").astype(bool),
    )
    s2 = S2SupportArtifacts(
        score=_data([1, 3, 1], "s2_score").astype("uint8"),
        support=_data([0, 1, 0], "s2_support").astype(bool),
        reliable=_data([0, 1, 1], "s2_reliable").astype(bool),
    )

    fused = apply_soft_refinement(s1, s2, Sentinel2Config())

    assert fused.refined.values.tolist() == [[True, True, True]]
    assert fused.decision.values.tolist() == [[1, 2, 3]]


def test_build_unavailable_s2_support_like_marks_everything_unreliable() -> None:
    template = _data([1, 1, 1], "construction").astype(bool)
    support = build_unavailable_s2_support_like(template)

    assert support.score.values.tolist() == [[0, 0, 0]]
    assert support.support.values.tolist() == [[False, False, False]]
    assert support.reliable.values.tolist() == [[False, False, False]]


def test_build_s2_support_requires_clear_fraction_even_with_clear_counts() -> None:
    coords = {"y": [0.0], "x": [0.0, 10.0]}
    before = xr.Dataset(
        {
            "ndvi": xr.DataArray([[0.2, 0.2]], dims=("y", "x"), coords=coords),
            "ndbi": xr.DataArray([[0.0, 0.0]], dims=("y", "x"), coords=coords),
            "mndwi": xr.DataArray([[-0.1, -0.1]], dims=("y", "x"), coords=coords),
            "bsi": xr.DataArray([[0.03, 0.03]], dims=("y", "x"), coords=coords),
            "clear_count": xr.DataArray([[2, 2]], dims=("y", "x"), coords=coords),
            "valid_count": xr.DataArray([[2, 2]], dims=("y", "x"), coords=coords),
            "clear_fraction": xr.DataArray([[0.30, 0.05]], dims=("y", "x"), coords=coords),
            "valid_fraction": xr.DataArray([[0.30, 0.05]], dims=("y", "x"), coords=coords),
        }
    )
    after = xr.Dataset(
        {
            "ndvi": xr.DataArray([[0.2, 0.2]], dims=("y", "x"), coords=coords),
            "ndbi": xr.DataArray([[0.1, 0.1]], dims=("y", "x"), coords=coords),
            "mndwi": xr.DataArray([[-0.2, -0.2]], dims=("y", "x"), coords=coords),
            "bsi": xr.DataArray([[0.05, 0.05]], dims=("y", "x"), coords=coords),
            "clear_count": xr.DataArray([[2, 2]], dims=("y", "x"), coords=coords),
            "valid_count": xr.DataArray([[2, 2]], dims=("y", "x"), coords=coords),
            "clear_fraction": xr.DataArray([[0.30, 0.05]], dims=("y", "x"), coords=coords),
            "valid_fraction": xr.DataArray([[0.30, 0.05]], dims=("y", "x"), coords=coords),
        }
    )

    support = build_s2_support(before, after, Sentinel2Config(min_clear_fraction=0.15))

    assert support.support.values.tolist() == [[True, True]]
    assert support.reliable.values.tolist() == [[True, False]]
