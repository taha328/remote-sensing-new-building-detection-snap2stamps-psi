from __future__ import annotations

import json
import logging
import os
import shutil
from contextlib import suppress
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import geopandas as gpd
import pyogrio
import rasterio
import rioxarray  # noqa: F401
import xarray as xr
import yaml
from affine import Affine
from geopandas import GeoDataFrame

from casablanca_builtup.config import ExportConfig

LOGGER = logging.getLogger(__name__)


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_yaml(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def _infer_transform(data: xr.DataArray) -> Affine:
    x = np.asarray(data["x"].values)
    y = np.asarray(data["y"].values)
    xres = float(np.median(np.diff(x)))
    yres = float(np.median(np.diff(y)))
    return Affine.translation(float(x[0] - (xres / 2.0)), float(y[0] - (yres / 2.0))) * Affine.scale(
        xres,
        yres,
    )


def ensure_spatial_metadata(data: xr.DataArray, crs: str) -> xr.DataArray:
    raster = data.copy()
    raster = raster.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
    if raster.rio.crs is None:
        raster = raster.rio.write_crs(crs, inplace=False)
    try:
        raster.rio.transform()
    except Exception:
        raster = raster.rio.write_transform(_infer_transform(raster), inplace=False)
    return raster


def read_raster(path: Path, band_as_variable: bool = False) -> xr.DataArray:
    raster = rioxarray.open_rasterio(path, masked=True)
    if "band" in raster.dims and raster.sizes.get("band", 1) == 1 and not band_as_variable:
        raster = raster.squeeze("band", drop=True)
    return raster


def _temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.stem}.{os.getpid()}.{uuid4().hex}{path.suffix}")


def remove_file(path: Path) -> None:
    with suppress(FileNotFoundError):
        path.unlink()


def is_valid_raster_artifact(path: Path, *, cleanup: bool = False) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        if cleanup:
            remove_file(path)
        return False
    try:
        with rasterio.open(path) as dataset:
            valid = dataset.count > 0 and dataset.width > 0 and dataset.height > 0
    except Exception:
        valid = False
    if not valid and cleanup:
        LOGGER.warning("Removing incomplete raster artifact %s", path)
        remove_file(path)
    return valid


def is_valid_vector_artifact(path: Path, *, cleanup: bool = False) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        if cleanup:
            remove_file(path)
        return False
    try:
        if path.suffix.lower() == ".parquet":
            gpd.read_parquet(path)
        else:
            pyogrio.read_dataframe(path, max_features=1)
        valid = True
    except Exception:
        valid = False
    if not valid and cleanup:
        LOGGER.warning("Removing incomplete vector artifact %s", path)
        remove_file(path)
    return valid


def write_raster(data: xr.DataArray, path: Path, crs: str, export_config: ExportConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raster = ensure_spatial_metadata(data, crs)
    temp_path = _temporary_path(path)
    remove_file(temp_path)
    try:
        raster.rio.to_raster(
            temp_path,
            driver=export_config.raster_driver,
            compress=export_config.raster_compress,
            predictor=export_config.raster_predictor,
            tiled=True,
        )
        if not is_valid_raster_artifact(temp_path):
            raise IOError(f"Raster write did not produce a valid artifact: {temp_path}")
        temp_path.replace(path)
    finally:
        remove_file(temp_path)


def write_vector(frame: GeoDataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _temporary_path(path)
    remove_file(temp_path)
    suffix = path.suffix.lower()
    try:
        if suffix == ".parquet":
            frame.to_parquet(temp_path, index=False)
        elif suffix == ".gpkg":
            pyogrio.write_dataframe(frame, temp_path, driver="GPKG", layer=path.stem.replace(".", "_"))
        elif suffix == ".geojson":
            pyogrio.write_dataframe(frame, temp_path, driver="GeoJSON")
        else:
            raise ValueError(f"Unsupported vector output format for path: {path}")
        if not is_valid_vector_artifact(temp_path):
            raise IOError(f"Vector write did not produce a valid artifact: {temp_path}")
        temp_path.replace(path)
    finally:
        remove_file(temp_path)


def read_vector(path: Path) -> GeoDataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return gpd.read_parquet(path)
    return gpd.read_file(path)


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
