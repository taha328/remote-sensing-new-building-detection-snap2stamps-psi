from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class AOIConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    path: Path | None = None
    bbox: tuple[float, float, float, float] | None = None
    crs: str = "EPSG:4326"

    @model_validator(mode="after")
    def validate_source(self) -> "AOIConfig":
        if (self.path is None) == (self.bbox is None):
            raise ValueError("Exactly one of aoi.path or aoi.bbox must be provided.")
        return self


class TimeWindow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: date
    end: date

    @model_validator(mode="after")
    def validate_dates(self) -> "TimeWindow":
        if self.end < self.start:
            raise ValueError("TimeWindow.end must be on or after TimeWindow.start.")
        return self

    @property
    def stac_datetime(self) -> str:
        return f"{self.start.isoformat()}/{self.end.isoformat()}"


class ChangePeriod(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    before: TimeWindow
    after: TimeWindow


class Sentinel1AssetNames(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vv: str = "vv"
    vh: str = "vh"


class Sentinel1Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    collection: str = "sentinel-1-rtc"
    platform: str = "SENTINEL-1A"
    instrument_mode: str = "IW"
    polarizations: tuple[str, str] = ("vv", "vh")
    orbit_state: Literal["ascending", "descending"] | None = None
    asset_names: Sentinel1AssetNames = Field(default_factory=Sentinel1AssetNames)


class Sentinel2AssetNames(BaseModel):
    model_config = ConfigDict(extra="forbid")

    blue: str = "B02"
    green: str = "B03"
    red: str = "B04"
    nir: str = "B08"
    swir1: str = "B11"
    scl: str = "SCL"


class StrongOverrideConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    vv_ratio_min: float = 2.0
    vh_ratio_min: float = 1.15
    pvalue_max: float = 0.001


class Sentinel2Thresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ndvi_after_max: float = 0.35
    mndwi_after_max: float = 0.0
    ndbi_after_min: float = -0.02
    delta_ndbi_min: float = 0.02
    bsi_after_min: float = 0.02
    delta_bsi_min: float = 0.01


class Sentinel2Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    collection: str = "sentinel-2-l2a"
    max_cloud_cover: float = 80.0
    asset_names: Sentinel2AssetNames = Field(default_factory=Sentinel2AssetNames)
    clear_scl_classes: tuple[int, ...] = (2, 4, 5, 6, 7)
    min_clear_observations: int = 2
    min_clear_fraction: float = 0.15
    support_score_min: int = 2
    thresholds: Sentinel2Thresholds = Field(default_factory=Sentinel2Thresholds)
    use_bsi: bool = True
    allow_unavailable: bool = True
    strong_override: StrongOverrideConfig = Field(default_factory=StrongOverrideConfig)


class GridConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    crs: str = "EPSG:32629"
    resolution_m: float = 10.0
    alignment: Literal["edge", "center"] = "edge"


class DetectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alpha: float = 0.01
    enl: float = 4.4
    vv_ratio_min: float = 1.5
    vh_ratio_min: float = 1.05
    min_connected_pixels: int = 4
    closing_radius_m: float = 10.0


class DensityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    radius_m: float = 20.0
    min_density: float = 0.15
    min_connected_pixels: int = 6
    closing_radius_m: float = 5.0


class PolygonizationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_area_ha: float = 0.05
    min_compactness: float = 0.0
    simplify_tolerance_m: float = 5.0
    use_tiled_polygonization: bool = True
    tile_size_pixels: int = 4096
    tile_overlap_pixels: int = 32


class ExportConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raster_driver: str = "GTiff"
    raster_compress: str = "DEFLATE"
    raster_predictor: int = 2
    save_intermediates: bool = True
    vector_formats: tuple[str, ...] = ("parquet", "gpkg")


class CacheConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reuse_manifests: bool = True
    overwrite: bool = False


class DaskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    chunks: dict[str, int] = Field(default_factory=lambda: {"x": 2048, "y": 2048})
    scheduler: Literal["threads", "synchronous"] = "threads"


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_root: str = "runs"
    log_level: str = "INFO"
    write_log_file: bool = True


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project: str
    aoi: AOIConfig
    periods: list[ChangePeriod]
    sentinel1: Sentinel1Config = Field(default_factory=Sentinel1Config)
    sentinel2: Sentinel2Config = Field(default_factory=Sentinel2Config)
    grid: GridConfig = Field(default_factory=GridConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    density: DensityConfig = Field(default_factory=DensityConfig)
    polygonization: PolygonizationConfig = Field(default_factory=PolygonizationConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    dask: DaskConfig = Field(default_factory=DaskConfig)
    run: RunConfig = Field(default_factory=RunConfig)

    @model_validator(mode="after")
    def validate_period_ids(self) -> "PipelineConfig":
        period_ids = [period.id for period in self.periods]
        if len(period_ids) != len(set(period_ids)):
            raise ValueError("period.id values must be unique.")
        return self


def load_config(path: str | Path) -> PipelineConfig:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text())
    return PipelineConfig.model_validate(payload)
