from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor

import geopandas as gpd
from odc.geo.geobox import GeoBox
from shapely.geometry import box

from casablanca_builtup.config import AOIConfig, GridConfig


@dataclass(frozen=True)
class GridDefinition:
    geobox: GeoBox
    bounds: tuple[float, float, float, float]
    width: int
    height: int
    resolution_m: float


def load_aoi_frame(aoi_config: AOIConfig) -> gpd.GeoDataFrame:
    if aoi_config.path is not None:
        gdf = gpd.read_file(aoi_config.path)
    else:
        minx, miny, maxx, maxy = aoi_config.bbox  # type: ignore[misc]
        gdf = gpd.GeoDataFrame(
            {"name": [aoi_config.name]},
            geometry=[box(minx, miny, maxx, maxy)],
            crs=aoi_config.crs,
        )
    if gdf.crs is None:
        gdf = gdf.set_crs(aoi_config.crs)
    gdf = gdf.to_crs(aoi_config.crs)
    geometry = gdf.geometry.union_all() if hasattr(gdf.geometry, "union_all") else gdf.unary_union
    return gpd.GeoDataFrame({"name": [aoi_config.name]}, geometry=[geometry], crs=gdf.crs)


def build_grid(aoi: gpd.GeoDataFrame, grid_config: GridConfig) -> GridDefinition:
    projected = aoi.to_crs(grid_config.crs)
    minx, miny, maxx, maxy = projected.total_bounds
    resolution = grid_config.resolution_m

    aligned_bounds = (
        floor(minx / resolution) * resolution,
        floor(miny / resolution) * resolution,
        ceil(maxx / resolution) * resolution,
        ceil(maxy / resolution) * resolution,
    )
    geobox = GeoBox.from_bbox(
        aligned_bounds,
        crs=grid_config.crs,
        resolution=resolution,
        anchor=grid_config.alignment,
    )
    height, width = geobox.shape
    return GridDefinition(
        geobox=geobox,
        bounds=aligned_bounds,
        width=width,
        height=height,
        resolution_m=resolution,
    )
