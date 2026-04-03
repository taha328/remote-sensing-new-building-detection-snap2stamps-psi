from __future__ import annotations

import logging
from pathlib import Path

import dask
from dask.base import is_dask_collection
import numpy as np
import xarray as xr
from odc.stac import load as stac_load

from aoi_builtup.config import DaskConfig, ExportConfig, Sentinel2Config
from aoi_builtup.io import is_valid_raster_artifact, read_raster, write_raster
from aoi_builtup.runtime import dataset_profile, log_timing

LOGGER = logging.getLogger(__name__)
REFLECTANCE_BANDS = ("blue", "green", "red", "nir", "swir1")


def _s2_load_chunks(dask_config: DaskConfig) -> dict[str, int] | None:
    if not dask_config.enabled:
        return None
    chunks = dict(dask_config.chunks)
    if "x" in chunks:
        chunks["x"] = min(int(chunks["x"]), 1024)
    if "y" in chunks:
        chunks["y"] = min(int(chunks["y"]), 1024)
    chunks.setdefault("time", 1)
    return chunks


def _s2_compute_scheduler(dask_config: DaskConfig) -> str:
    if not dask_config.enabled:
        return "synchronous"
    return dask_config.scheduler


def _materialize_dataarray(
    data: xr.DataArray,
    *,
    logger: logging.Logger,
    event: str,
    period_id: str | None,
    phase: str | None,
    scheduler: str,
    band: str | None = None,
) -> xr.DataArray:
    fields = {"period_id": period_id, "phase": phase}
    if band is not None:
        fields["band"] = band
    if is_dask_collection(data.data):
        with dask.config.set(scheduler=scheduler):
            with log_timing(logger, event, scheduler=scheduler, **fields):
                return data.compute()
    with log_timing(logger, event, **fields):
        return data.load()


def _compute_indices(dataset: xr.Dataset, use_bsi: bool) -> xr.Dataset:
    ndvi = ((dataset["nir"] - dataset["red"]) / (dataset["nir"] + dataset["red"]).clip(min=1e-6)).rename(
        "ndvi"
    )
    ndbi = (
        (dataset["swir1"] - dataset["nir"]) / (dataset["swir1"] + dataset["nir"]).clip(min=1e-6)
    ).rename("ndbi")
    mndwi = (
        (dataset["green"] - dataset["swir1"]) / (dataset["green"] + dataset["swir1"]).clip(min=1e-6)
    ).rename("mndwi")

    outputs = [ndvi.astype("float32"), ndbi.astype("float32"), mndwi.astype("float32")]
    if use_bsi:
        bsi = (
            ((dataset["swir1"] + dataset["red"]) - (dataset["nir"] + dataset["blue"]))
            / ((dataset["swir1"] + dataset["red"]) + (dataset["nir"] + dataset["blue"])).clip(min=1e-6)
        ).rename("bsi")
        outputs.append(bsi.astype("float32"))
    return xr.merge(outputs)


def _load_s2_stack(
    items: list,
    geobox,
    *,
    band_assets: list[str],
    rename_map: dict[str, str],
    resampling: dict[str, str],
    load_chunks: dict[str, int] | None,
    period_id: str | None,
    phase: str | None,
) -> xr.Dataset:
    with log_timing(
        LOGGER,
        "S2 STAC load graph",
        period_id=period_id,
        phase=phase,
        item_count=len(items),
        bands=band_assets,
        chunks=load_chunks,
    ):
        dataset = stac_load(
            items,
            bands=band_assets,
            geobox=geobox,
            chunks=load_chunks,
            resampling=resampling,
        )
    dataset = dataset.rename(rename_map)
    LOGGER.info(
        "S2 source stack ready | period_id=%s phase=%s bands=%s profile=%s",
        period_id,
        phase,
        list(rename_map.values()),
        dataset_profile(dataset),
    )
    return dataset


def _stage_path(stage_dir: Path, period_id: str, phase: str, band: str) -> Path:
    return stage_dir / "s2" / period_id / phase / f"{band}_stack.tif"


def _prepare_stack_for_raster(data: xr.DataArray, band: str) -> xr.DataArray:
    stack = data.rename(band)
    if "time" in stack.dims:
        stack = stack.rename(time="band")
    if "band" not in stack.dims:
        stack = stack.expand_dims(dim={"band": [1]})
    else:
        stack = stack.assign_coords(band=np.arange(1, stack.sizes["band"] + 1, dtype=np.int32))
    return stack


def _load_staged_stack(path: Path, band: str) -> xr.DataArray:
    stack = read_raster(path).rename(band)
    if "band" not in stack.dims:
        stack = stack.expand_dims(dim={"band": [1]})
    return stack


def _stage_or_load_stack(
    items: list,
    geobox,
    *,
    asset_name: str,
    output_name: str,
    resampling: str,
    load_chunks: dict[str, int] | None,
    compute_scheduler: str,
    stage_dir: Path,
    export_config: ExportConfig,
    period_id: str | None,
    phase: str | None,
) -> xr.DataArray:
    if period_id is None or phase is None:
        raise ValueError("period_id and phase are required when using staged Sentinel-2 caching.")
    path = _stage_path(stage_dir, period_id, phase, output_name)
    if is_valid_raster_artifact(path, cleanup=True):
        LOGGER.info(
            "Reusing staged S2 stack artifact | period_id=%s phase=%s band=%s path=%s bytes=%s",
            period_id,
            phase,
            output_name,
            path,
            path.stat().st_size,
        )
        return _load_staged_stack(path, output_name)

    dataset = _load_s2_stack(
        items,
        geobox,
        band_assets=[asset_name],
        rename_map={asset_name: output_name},
        resampling={asset_name: resampling},
        load_chunks=load_chunks,
        period_id=period_id,
        phase=phase,
    )
    stack = _prepare_stack_for_raster(dataset[output_name], output_name)
    stack = _materialize_dataarray(
        stack,
        logger=LOGGER,
        event="S2 stage stack compute",
        period_id=period_id,
        phase=phase,
        scheduler=compute_scheduler,
        band=output_name,
    )
    stage_export_config = ExportConfig(
        raster_driver=export_config.raster_driver,
        raster_compress="NONE",
        raster_predictor=1,
        save_intermediates=export_config.save_intermediates,
        vector_formats=export_config.vector_formats,
    )
    with log_timing(
        LOGGER,
        "S2 stage stack write",
        period_id=period_id,
        phase=phase,
        band=output_name,
        path=path,
        compress=stage_export_config.raster_compress,
    ):
        write_raster(stack, path, str(geobox.crs), stage_export_config)
    LOGGER.info(
        "Staged S2 stack artifact ready | period_id=%s phase=%s band=%s path=%s bytes=%s slices=%s",
        period_id,
        phase,
        output_name,
        path,
        path.stat().st_size,
        stack.sizes.get("band", 1),
    )
    return _load_staged_stack(path, output_name)


def build_s2_composite(
    items: list,
    geobox,
    config: Sentinel2Config,
    dask_config: DaskConfig,
    export_config: ExportConfig | None = None,
    stage_dir: Path | None = None,
    *,
    period_id: str | None = None,
    phase: str | None = None,
) -> xr.Dataset:
    if not items:
        raise ValueError("No Sentinel-2 items available for composite construction.")
    load_chunks = _s2_load_chunks(dask_config)
    compute_scheduler = _s2_compute_scheduler(dask_config)
    export_cfg = export_config or ExportConfig()
    stage_root = stage_dir
    if stage_root is not None:
        stage_root.mkdir(parents=True, exist_ok=True)

    asset_by_band = {
        "scl": config.asset_names.scl,
        "blue": config.asset_names.blue,
        "green": config.asset_names.green,
        "red": config.asset_names.red,
        "nir": config.asset_names.nir,
        "swir1": config.asset_names.swir1,
    }
    resampling_by_band = {
        "scl": "nearest",
        "blue": "bilinear",
        "green": "bilinear",
        "red": "bilinear",
        "nir": "bilinear",
        "swir1": "bilinear",
    }

    def load_stack(band: str) -> xr.DataArray:
        if stage_root is not None:
            return _stage_or_load_stack(
                items,
                geobox,
                asset_name=asset_by_band[band],
                output_name=band,
                resampling=resampling_by_band[band],
                load_chunks=load_chunks,
                compute_scheduler=compute_scheduler,
                stage_dir=stage_root,
                export_config=export_cfg,
                period_id=period_id,
                phase=phase,
            )
        dataset = _load_s2_stack(
            items,
            geobox,
            band_assets=[asset_by_band[band]],
            rename_map={asset_by_band[band]: band},
            resampling={asset_by_band[band]: resampling_by_band[band]},
            load_chunks=load_chunks,
            period_id=period_id,
            phase=phase,
        )
        return _prepare_stack_for_raster(dataset[band], band)

    scl_stack = load_stack("scl")
    clear_mask = _materialize_dataarray(
        scl_stack.isin(np.asarray(config.clear_scl_classes)).rename("clear_mask"),
        logger=LOGGER,
        event="S2 clear mask compute",
        period_id=period_id,
        phase=phase,
        scheduler=compute_scheduler,
    ).astype(bool).load()
    usable_mask = clear_mask.copy(deep=True).rename("usable_mask")

    for band in REFLECTANCE_BANDS:
        band_stack = load_stack(band)
        band_valid = _materialize_dataarray(
            band_stack.notnull().rename("valid_mask"),
            logger=LOGGER,
            event="S2 reflectance valid compute",
            period_id=period_id,
            phase=phase,
            scheduler=compute_scheduler,
            band=band,
        ).astype(bool)
        usable_mask = (usable_mask & band_valid).rename("usable_mask")
    usable_mask = usable_mask.load()

    with log_timing(
        LOGGER,
        "S2 count products build",
        period_id=period_id,
        phase=phase,
        item_count=len(items),
    ):
        clear_count = clear_mask.sum(dim="band").astype("uint16").rename("clear_count")
        valid_count = usable_mask.sum(dim="band").astype("uint16").rename("valid_count")
        clear_fraction = (clear_count / max(len(items), 1)).astype("float32").rename("clear_fraction")
        valid_fraction = (valid_count / max(len(items), 1)).astype("float32").rename("valid_fraction")
        count_products = xr.merge([clear_count, valid_count, clear_fraction, valid_fraction]).load()
    LOGGER.info(
        "S2 count products ready | period_id=%s phase=%s profile=%s",
        period_id,
        phase,
        dataset_profile(count_products),
    )

    composite_bands: list[xr.DataArray] = []
    for band in REFLECTANCE_BANDS:
        band_stack = load_stack(band)
        with log_timing(
            LOGGER,
            "S2 reflectance composite graph",
            period_id=period_id,
            phase=phase,
            item_count=len(items),
            band=band,
        ):
            masked_band = ((band_stack.astype("float32") / 10000.0).where(usable_mask)).rename(band)
            composite_band = masked_band.median(dim="band", skipna=True).astype("float32").rename(band)
        composite_bands.append(
            _materialize_dataarray(
                composite_band,
                logger=LOGGER,
                event="S2 reflectance composite compute",
                period_id=period_id,
                phase=phase,
                scheduler=compute_scheduler,
                band=band,
            )
        )

    composite = xr.merge(composite_bands)
    merged = xr.merge([composite, _compute_indices(composite, config.use_bsi), count_products]).load()
    LOGGER.info(
        "S2 composite ready for export | period_id=%s phase=%s profile=%s",
        period_id,
        phase,
        dataset_profile(merged),
    )
    return merged
