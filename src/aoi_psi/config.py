from __future__ import annotations

from datetime import date
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


def _env_override(name: str | None) -> str | None:
    if not name:
        return None
    value = os.environ.get(name)
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _normalize_s3_endpoint_url(value: str) -> str:
    endpoint = value.strip().rstrip("/")
    if "://" not in endpoint:
        endpoint = f"https://{endpoint}"
    return endpoint


def _csv_env_override(name: str | None) -> tuple[str, ...] | None:
    value = _env_override(name)
    if value is None:
        return None
    parts = tuple(part.strip() for part in value.split(",") if part.strip())
    return parts or ()


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
    def iso_range(self) -> str:
        return f"{self.start.isoformat()}/{self.end.isoformat()}"


class AcquisitionAuthConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    username_env: str | None = "CDSE_USERNAME"
    password_env: str | None = "CDSE_PASSWORD"
    bearer_token_env: str | None = "CDSE_ACCESS_TOKEN"
    access_token_env: str | None = "ACCESS_TOKEN"
    refresh_token_env: str | None = "REFRESH_TOKEN"
    totp_env: str | None = None
    token_endpoint: str = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    client_id: str = "cdse-public"
    refresh_margin_seconds: int = 120


class AcquisitionS3Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    endpoint_url: str = "https://eodata.dataspace.copernicus.eu"
    fallback_endpoint_urls: tuple[str, ...] = ("https://eodata.ams.dataspace.copernicus.eu",)
    bucket: str = "eodata"
    region_name: str = "default"
    endpoint_env: str | None = "CDSE_S3_ENDPOINT"
    fallback_endpoints_env: str | None = "CDSE_S3_FALLBACK_ENDPOINTS"
    bucket_env: str | None = "CDSE_S3_BUCKET"
    access_key_env: str = "CDSE_S3_ACCESS_KEY"
    secret_key_env: str = "CDSE_S3_SECRET_KEY"
    session_token_env: str | None = None
    max_attempts: int = 5
    download_attempts: int = 4
    retry_mode: Literal["legacy", "standard", "adaptive"] = "standard"
    connect_timeout_seconds: int = 15
    read_timeout_seconds: int | None = None
    max_pool_connections: int = 8
    addressing_style: Literal["auto", "path", "virtual"] = "path"
    tcp_keepalive: bool = True

    @model_validator(mode="after")
    def apply_environment_overrides(self) -> "AcquisitionS3Config":
        endpoint = _env_override(self.endpoint_env)
        if endpoint:
            self.endpoint_url = _normalize_s3_endpoint_url(endpoint)
        else:
            self.endpoint_url = _normalize_s3_endpoint_url(self.endpoint_url)

        fallback_endpoints = _csv_env_override(self.fallback_endpoints_env)
        candidates = fallback_endpoints if fallback_endpoints is not None else self.fallback_endpoint_urls
        normalized_fallbacks: list[str] = []
        for candidate in candidates:
            normalized = _normalize_s3_endpoint_url(candidate)
            if normalized == self.endpoint_url or normalized in normalized_fallbacks:
                continue
            normalized_fallbacks.append(normalized)
        self.fallback_endpoint_urls = tuple(normalized_fallbacks)

        bucket = _env_override(self.bucket_env)
        if bucket:
            self.bucket = bucket
        return self


class AcquisitionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["cdse-stac", "asf-search"] = "cdse-stac"
    s1_catalog_url: str = "https://stac.dataspace.copernicus.eu/v1"
    odata_catalog_url: str = "https://catalogue.dataspace.copernicus.eu/odata/v1"
    s1_collection: str = "sentinel-1-slc"
    s2_catalog_url: str = "https://earth-search.aws.element84.com/v1"
    s2_collection: str = "sentinel-2-l2a"
    time_window: TimeWindow
    s2_time_window: TimeWindow | None = None
    auth: AcquisitionAuthConfig = Field(default_factory=AcquisitionAuthConfig)
    s3: AcquisitionS3Config = Field(default_factory=AcquisitionS3Config)
    download_transport: Literal["auto", "s3", "odata"] = "auto"
    download_workers: int = 2
    timeout_seconds: int = 180
    slc_asset_priority: tuple[str, ...] = ("product", "safe_manifest")


class DEMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: Path
    vertical_datum: str = "EGM96"
    resampling: Literal["BILINEAR_INTERPOLATION", "NEAREST_NEIGHBOUR"] = "BILINEAR_INTERPOLATION"


class OrbitStackConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    direction: Literal["ascending", "descending"]
    relative_orbit: int
    iw_swaths: tuple[Literal["IW1", "IW2", "IW3"], ...] = ("IW1", "IW2", "IW3")
    polarization: Literal["VV", "VH", "VV+VH"] = "VV"
    master_date: date | None = None
    min_scenes: int = 18
    scene_limit: int | None = None

    @model_validator(mode="after")
    def validate_scene_limit(self) -> "OrbitStackConfig":
        if self.scene_limit is not None and self.scene_limit < self.min_scenes:
            raise ValueError("OrbitStackConfig.scene_limit must be greater than or equal to min_scenes.")
        return self


class SnapGraphConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gpt_path: str = "gpt"
    gpt_vmoptions_path: Path | None = None
    graph_root: Path = Path("resources/snap_graphs")
    orbit_source: Literal["precise", "restituted"] = "precise"
    user_dir: Path | None = None
    cache_size_mb: int = Field(default=8192, ge=1)
    java_options: tuple[str, ...] = ("-Xms4G", "-Xmx24G")
    clear_tile_cache_after_row: bool = False
    default_tile_size_px: int | None = Field(default=512, ge=1)
    subset_to_aoi: bool = True
    write_intermediate_geotiff: bool = False
    workers: int = Field(default=1, ge=1)


class StaMPSConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    install_root: Path
    matlab_command: str | None = "matlab"
    octave_command: str | None = None
    use_octave: bool = False
    export_script: Path | None = None
    amplitude_dispersion_threshold: float = 0.4
    weed_standard_dev: float = 1.0
    unwrap_method: Literal["3D", "snaphu"] = "3D"
    small_baseline_mode: bool = False
    range_patches: int = Field(default=1, ge=1)
    azimuth_patches: int = Field(default=1, ge=1)
    max_parallel_patch_workers: int = Field(default=1, ge=1)
    range_overlap: int = Field(default=50, ge=0)
    azimuth_overlap: int = Field(default=50, ge=0)
    merge_resample_size: int = Field(default=0, ge=0)
    mask_file: Path | None = None

    @model_validator(mode="before")
    @classmethod
    def _upgrade_legacy_patch_worker_field(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        legacy_value = normalized.pop("max_patch_workers", None)
        if "max_parallel_patch_workers" not in normalized and legacy_value is not None:
            normalized["max_parallel_patch_workers"] = legacy_value
        return normalized


class PsiDetectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["cdpsi"] = "cdpsi"
    minimum_subset_images: int = Field(default=3, ge=3)
    thresholding_mode: Literal["gaussian_3sigma"] = "gaussian_3sigma"
    sigma_multiplier: float = Field(default=3.0, gt=0.0)
    cluster_buffer_m: float = 12.0


class AmplitudeBranchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    ratio_min: float = 1.35
    log_ratio_min: float = 0.3
    min_connected_pixels: int = 4


class Sentinel2Thresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ndvi_after_max: float = 0.35
    mndwi_after_max: float = 0.0
    ndbi_after_min: float = -0.02
    delta_ndbi_min: float = 0.02
    bsi_after_min: float = 0.02
    delta_bsi_min: float = 0.01


class Sentinel2RefinementConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    min_clear_observations: int = 2
    max_cloud_cover: float = 70.0
    clear_scl_classes: tuple[int, ...] = (2, 4, 5, 6, 7)
    min_clear_fraction: float = 0.2
    support_score_min: int = 2
    thresholds: Sentinel2Thresholds = Field(default_factory=Sentinel2Thresholds)
    use_bsi: bool = True
    allow_unavailable: bool = True


class ContextConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parcel_path: Path | None = None
    building_footprints_path: Path | None = None
    land_use_path: Path | None = None


class FusionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    psi_primary_weight: float = 0.65
    amplitude_weight: float = 0.2
    optical_weight: float = 0.1
    context_weight: float = 0.05
    high_confidence_threshold: float = 0.7
    medium_confidence_threshold: float = 0.45
    allow_s2_non_support_penalty: bool = True
    s2_non_support_penalty: float = 0.08


class ExportConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raster_driver: str = "GTiff"
    raster_compress: str = "DEFLATE"
    raster_predictor: int = 2
    vector_formats: tuple[str, ...] = ("parquet", "gpkg")
    save_intermediate_points: bool = True


class CacheConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    overwrite: bool = False
    reuse_manifests: bool = True
    reuse_downloads: bool = True
    reuse_snap_outputs: bool = True
    reuse_stamps_outputs: bool = True


class ArtifactLifecycleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    purge_master_prepared_after_pair: bool = True
    purge_secondary_prepared_after_pair: bool = True
    purge_pair_products_after_merged_coreg: bool = True
    purge_prepared_after_coreg: bool = True
    purge_snap_intermediates_after_export: bool = True
    purge_snap_export_after_stamps: bool = True
    purge_stamps_workspace_after_parse: bool = True
    cleanup_obsolete_snap_backups: bool = True


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_root: str = "runs_psi"
    log_level: str = "INFO"
    write_log_file: bool = True


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project: str
    aoi: AOIConfig
    acquisition: AcquisitionConfig
    dem: DEMConfig
    stacks: list[OrbitStackConfig]
    snap: SnapGraphConfig
    stamps: StaMPSConfig
    psi: PsiDetectionConfig = Field(default_factory=PsiDetectionConfig)
    amplitude: AmplitudeBranchConfig = Field(default_factory=AmplitudeBranchConfig)
    s2_refinement: Sentinel2RefinementConfig = Field(default_factory=Sentinel2RefinementConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    artifact_lifecycle: ArtifactLifecycleConfig = Field(default_factory=ArtifactLifecycleConfig)
    run: RunConfig = Field(default_factory=RunConfig)

    @model_validator(mode="after")
    def validate_stacks(self) -> "PipelineConfig":
        ids = [stack.id for stack in self.stacks]
        if len(ids) != len(set(ids)):
            raise ValueError("stack ids must be unique.")
        keys = [(stack.direction, stack.relative_orbit) for stack in self.stacks]
        if len(keys) != len(set(keys)):
            raise ValueError("Each direction/relative_orbit combination must be unique.")
        if any(stack.master_date is None for stack in self.stacks):
            raise ValueError("Each PSI stack must define master_date for a reproducible inversion.")
        return self


def load_config(path: str | Path) -> PipelineConfig:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text())
    return PipelineConfig.model_validate(payload)
