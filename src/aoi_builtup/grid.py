from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor
from typing import Iterable

import geopandas as gpd
from odc.geo.geobox import GeoBox
from shapely.geometry.base import BaseGeometry
from shapely.geometry import box
from shapely import wkt

from aoi_builtup.config import AOIConfig, GridConfig


@dataclass(frozen=True)
class GridDefinition:
    geobox: GeoBox
    crs: str
    bounds: tuple[float, float, float, float]
    width: int
    height: int
    resolution_m: float


def infer_metric_crs_from_bounds(bounds: Iterable[float]) -> str:
    minx, miny, maxx, maxy = tuple(bounds)
    lon = (minx + maxx) / 2.0
    lat = (miny + maxy) / 2.0

    # UTM does not cover the polar caps. Fall back to a generic metric CRS there.
    if lat <= -80.0 or lat >= 84.0:
        return "EPSG:3857"

    zone = int((lon + 180.0) // 6.0) + 1
    zone = max(1, min(zone, 60))
    epsg = 32600 + zone if lat >= 0.0 else 32700 + zone
    return f"EPSG:{epsg}"


def infer_metric_crs(aoi: gpd.GeoDataFrame) -> str:
    geographic = aoi.to_crs("EPSG:4326")
    return infer_metric_crs_from_bounds(geographic.total_bounds)


def infer_metric_crs_from_geometry(geometry: BaseGeometry) -> str:
    if not hasattr(geometry, "bounds"):
        if hasattr(geometry, "wkt"):
            geometry = wkt.loads(geometry.wkt)
        else:
            raise TypeError("geometry must provide either bounds or a WKT representation.")
    minx, miny, maxx, maxy = geometry.bounds
    return infer_metric_crs_from_bounds((minx, miny, maxx, maxy))


def resolve_grid_crs(aoi: gpd.GeoDataFrame, grid_config: GridConfig) -> str:
    requested = grid_config.crs
    if requested and requested.lower() != "auto":
        return requested
    return infer_metric_crs(aoi)


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
    resolved_crs = resolve_grid_crs(aoi, grid_config)
    projected = aoi.to_crs(resolved_crs)
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
        crs=resolved_crs,
        resolution=resolution,
        anchor=grid_config.alignment,
    )
    height, width = geobox.shape
    return GridDefinition(
        geobox=geobox,
        crs=resolved_crs,
        bounds=aligned_bounds,
        width=width,
        height=height,
        resolution_m=resolution,
    )
