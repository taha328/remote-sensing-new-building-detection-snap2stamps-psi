import numpy as np
import xarray as xr

from aoi_builtup.config import PolygonizationConfig
from aoi_builtup.postprocess.vectorize import polygonize_mask


def test_polygonize_mask_returns_expected_area() -> None:
    data = np.zeros((6, 6), dtype=np.uint8)
    data[1:4, 1:4] = 1
    mask = xr.DataArray(
        data,
        dims=("y", "x"),
        coords={
            "y": np.array([55.0, 45.0, 35.0, 25.0, 15.0, 5.0]),
            "x": np.array([5.0, 15.0, 25.0, 35.0, 45.0, 55.0]),
        },
        name="zone",
    )

    polygons = polygonize_mask(mask, "EPSG:32629", PolygonizationConfig(min_area_ha=0.05))

    assert len(polygons) == 1
    assert round(float(polygons.iloc[0]["area_ha"]), 2) == 0.09


def test_polygonize_mask_tiled_merges_geometry_across_tile_edges() -> None:
    data = np.zeros((8, 8), dtype=np.uint8)
    data[2:6, 2:6] = 1
    mask = xr.DataArray(
        data,
        dims=("y", "x"),
        coords={
            "y": np.array([75.0, 65.0, 55.0, 45.0, 35.0, 25.0, 15.0, 5.0]),
            "x": np.array([5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0]),
        },
        name="zone",
    )

    polygons = polygonize_mask(
        mask,
        "EPSG:32629",
        PolygonizationConfig(
            min_area_ha=0.05,
            use_tiled_polygonization=True,
            tile_size_pixels=3,
            tile_overlap_pixels=1,
        ),
    )

    assert len(polygons) == 1
    assert round(float(polygons.iloc[0]["area_ha"]), 2) == 0.16
