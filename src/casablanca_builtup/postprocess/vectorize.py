from __future__ import annotations

from math import pi
from typing import Sequence

import geopandas as gpd
import numpy as np
import rasterio.features
import scipy.ndimage as ndi
import xarray as xr
from affine import Affine
from shapely.geometry import shape
from shapely.geometry import box as shapely_box
from skimage.morphology import disk

from casablanca_builtup.config import DensityConfig, PolygonizationConfig
from casablanca_builtup.io import ensure_spatial_metadata


def _radius_to_pixels(radius_m: float, resolution_m: float) -> int:
    return max(1, int(round(radius_m / resolution_m)))


def _remove_small_components(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    structure = ndi.generate_binary_structure(2, 2)
    labels, count = ndi.label(mask, structure=structure)
    if count == 0:
        return mask
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_pixels
    keep[0] = False
    return keep[labels]


def build_cumulative_first_change(period_masks: Sequence[tuple[str, xr.DataArray]]) -> xr.DataArray:
    if not period_masks:
        raise ValueError("At least one period mask is required.")
    template = period_masks[0][1].astype("uint8")
    cumulative = xr.zeros_like(template, dtype=np.uint8).rename("cumulative_change")
    for period_index, (_, mask) in enumerate(period_masks, start=1):
        available = cumulative == 0
        cumulative = xr.where(available & mask.astype(bool), period_index, cumulative)
    return cumulative.astype("uint8")


def build_density_zone_mask(
    cumulative: xr.DataArray,
    density_config: DensityConfig,
    resolution_m: float,
) -> tuple[xr.DataArray, xr.DataArray]:
    binary = (cumulative > 0).compute().values.astype(bool)
    binary = _remove_small_components(binary, density_config.min_connected_pixels)

    radius_pixels = _radius_to_pixels(density_config.radius_m, resolution_m)
    kernel = disk(radius_pixels).astype(np.float32)
    kernel /= kernel.sum()
    density = ndi.convolve(binary.astype(np.float32), kernel, mode="constant", cval=0.0)
    dense = density >= density_config.min_density

    closing_pixels = _radius_to_pixels(density_config.closing_radius_m, resolution_m)
    dense = ndi.binary_dilation(dense, structure=disk(closing_pixels))
    dense = ndi.binary_erosion(dense, structure=disk(closing_pixels))

    density_da = xr.DataArray(
        density.astype(np.float32),
        coords=cumulative.coords,
        dims=cumulative.dims,
        name="density",
    )
    zone_da = xr.DataArray(
        dense.astype(np.uint8),
        coords=cumulative.coords,
        dims=cumulative.dims,
        name="zone",
    )
    return density_da, zone_da


def _empty_polygons(crs: str) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"zone_id": [], "area_ha": [], "perim_m": [], "compactness": []},
        geometry=[],
        crs=crs,
    )


def _finalize_polygons(
    geometries: list,
    crs: str,
    polygon_config: PolygonizationConfig,
) -> gpd.GeoDataFrame:
    if not geometries:
        return _empty_polygons(crs)

    frame = gpd.GeoDataFrame({"zone_id": range(1, len(geometries) + 1)}, geometry=geometries, crs=crs)
    unioned = frame.geometry.union_all() if hasattr(frame.geometry, "union_all") else frame.unary_union
    exploded = gpd.GeoSeries([unioned], crs=crs).explode(index_parts=False).reset_index(drop=True)
    frame = gpd.GeoDataFrame({"zone_id": range(1, len(exploded) + 1)}, geometry=exploded, crs=crs)
    frame["geometry"] = frame.geometry.buffer(0)
    frame["geometry"] = frame.geometry.simplify(polygon_config.simplify_tolerance_m, preserve_topology=True)
    frame = frame[frame.geometry.notna() & ~frame.geometry.is_empty].copy()
    if frame.empty:
        return _empty_polygons(crs)

    frame["area_ha"] = frame.geometry.area / 10000.0
    frame["perim_m"] = frame.geometry.length
    frame["compactness"] = (4.0 * pi * frame.geometry.area) / frame["perim_m"].clip(lower=1.0).pow(2)
    frame = frame[
        (frame["area_ha"] >= polygon_config.min_area_ha)
        & (frame["compactness"] >= polygon_config.min_compactness)
    ].copy()
    frame.reset_index(drop=True, inplace=True)
    frame["zone_id"] = np.arange(1, len(frame) + 1)
    return frame


def _polygonize_array(data: np.ndarray, transform: Affine) -> list:
    features = rasterio.features.shapes(data, mask=data == 1, transform=transform)
    return [shape(geometry) for geometry, value in features if int(value) == 1]


def _polygonize_mask_single_pass(mask: xr.DataArray, crs: str, polygon_config: PolygonizationConfig) -> gpd.GeoDataFrame:
    raster = ensure_spatial_metadata(mask.astype("uint8"), crs)
    data = raster.values.astype(np.uint8)
    transform = raster.rio.transform()
    return _finalize_polygons(_polygonize_array(data, transform), crs, polygon_config)


def _polygonize_mask_tiled(mask: xr.DataArray, crs: str, polygon_config: PolygonizationConfig) -> gpd.GeoDataFrame:
    raster = ensure_spatial_metadata(mask.astype("uint8"), crs)
    data = raster.values.astype(np.uint8)
    transform = raster.rio.transform()
    height, width = data.shape
    tile_size = max(1, polygon_config.tile_size_pixels)
    overlap = max(0, polygon_config.tile_overlap_pixels)
    geometries: list = []

    for core_y0 in range(0, height, tile_size):
        core_y1 = min(core_y0 + tile_size, height)
        for core_x0 in range(0, width, tile_size):
            core_x1 = min(core_x0 + tile_size, width)
            tile_y0 = max(0, core_y0 - overlap)
            tile_y1 = min(height, core_y1 + overlap)
            tile_x0 = max(0, core_x0 - overlap)
            tile_x1 = min(width, core_x1 + overlap)
            tile = data[tile_y0:tile_y1, tile_x0:tile_x1]
            if np.count_nonzero(tile) == 0:
                continue

            tile_transform = transform * Affine.translation(tile_x0, tile_y0)
            tile_geometries = _polygonize_array(tile, tile_transform)
            if not tile_geometries:
                continue

            left, top = transform * (core_x0, core_y0)
            right, bottom = transform * (core_x1, core_y1)
            core_bbox = shapely_box(min(left, right), min(top, bottom), max(left, right), max(top, bottom))
            for geometry in tile_geometries:
                clipped = geometry.intersection(core_bbox)
                if not clipped.is_empty:
                    geometries.append(clipped)

    return _finalize_polygons(geometries, crs, polygon_config)


def polygonize_mask(
    mask: xr.DataArray,
    crs: str,
    polygon_config: PolygonizationConfig,
) -> gpd.GeoDataFrame:
    if polygon_config.use_tiled_polygonization:
        return _polygonize_mask_tiled(mask, crs, polygon_config)
    return _polygonize_mask_single_pass(mask, crs, polygon_config)
