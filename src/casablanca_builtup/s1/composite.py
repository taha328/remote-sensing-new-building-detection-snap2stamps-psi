from __future__ import annotations

import xarray as xr
from odc.stac import load as stac_load

from casablanca_builtup.config import DaskConfig, Sentinel1Config


def build_s1_composite(
    items: list,
    geobox,
    config: Sentinel1Config,
    dask_config: DaskConfig,
) -> xr.Dataset:
    if not items:
        raise ValueError("No Sentinel-1 items available for composite construction.")

    dataset = stac_load(
        items,
        bands=[config.asset_names.vv, config.asset_names.vh],
        geobox=geobox,
        chunks=dask_config.chunks if dask_config.enabled else None,
        resampling="nearest",
    )
    dataset = dataset.rename(
        {
            config.asset_names.vv: "vv",
            config.asset_names.vh: "vh",
        }
    )
    return dataset.median(dim="time", skipna=True).astype("float32")
