from __future__ import annotations

import base64
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
import errno
import json
from pathlib import Path
import logging
import os
import random
import re
import shutil
import socket
import time
from urllib.parse import urlparse
import zipfile

import geopandas as gpd
import requests
from pystac import Item
from pystac_client import Client
from pystac_client.exceptions import APIError
from shapely.geometry import mapping, box

from casablanca_psi.config import AOIConfig, OrbitStackConfig, PipelineConfig
from casablanca_psi.manifests import (
    SlcScene,
    StackManifest,
    read_stack_manifest,
    slc_scene_zip_path,
    stack_manifest_path,
    update_scene_s3_path,
    write_stack_manifest,
)
from casablanca_psi.run_context import RunContext

LOGGER = logging.getLogger(__name__)
QUERY_RETRIES = 3
QUERY_CHUNK_DAYS = 45
DOWNLOAD_CHUNK_SIZE = 8 * 1024 * 1024
DOWNLOAD_RETRIES = 3
DOWNLOAD_RETRYABLE_STATUS_CODES = {401, 429, 500, 502, 503, 504}
ODATA_QUERY_RETRIES = 3
STAC_LOOKUP_BATCH_SIZE = 100
S3_MEMBER_MIN_ENDPOINT_ATTEMPTS = 16
S3_MEMBER_TAIL_BONUS_ATTEMPTS = 16
S3_MEMBER_TAIL_REMAINING_BYTES = 256 * 1024 * 1024
S3_MEMBER_MIN_CHUNK_SIZE = 512 * 1024
S3_RETRYABLE_ERROR_CODES = {
    "RequestTimeout",
    "RequestTimeTooSkewed",
    "InternalError",
    "InternalServerError",
    "SlowDown",
    "ServiceUnavailable",
    "Throttling",
}
PRODUCT_ID_RE = re.compile(r"Products\(([^)]+)\)/\$value")
S1_PRODUCT_NAME_RE = re.compile(r"^S1[A-Z]_")
S3_PATH_CACHE: dict[str, str] = {}


@dataclass(frozen=True)
class DownloadRecord:
    scene_id: str
    href: str
    path: Path


@dataclass(frozen=True)
class S3ObjectInfo:
    key: str
    size: int
    member_name: str


@dataclass(frozen=True)
class ProviderFieldValidation:
    catalog_url: str
    collection: str
    item_id: str
    property_mapping: dict[str, object]
    asset_mapping: dict[str, object]


def load_aoi_geometry(aoi: AOIConfig):
    if aoi.path is not None:
        return gpd.read_file(aoi.path).to_crs("EPSG:4326").geometry.union_all()
    return box(*aoi.bbox)


def _item_direction(item: Item) -> str | None:
    props = item.properties
    return props.get("sat:orbit_state") or props.get("orbit_state") or props.get("s1:orbit_state")


def _item_relative_orbit(item: Item) -> int | None:
    props = item.properties
    value = props.get("sat:relative_orbit") or props.get("relativeOrbitNumber")
    return None if value is None else int(value)


def _item_polarization(item: Item) -> str:
    props = item.properties
    value = props.get("sar:polarizations") or props.get("polarisationChannels") or []
    if isinstance(value, str):
        return value
    return "+".join(sorted(str(part) for part in value))


def _item_product_type(item: Item) -> str | None:
    props = item.properties
    return props.get("product:type") or props.get("productType")


def _item_processing_level(item: Item) -> str | None:
    props = item.properties
    return props.get("processing:level") or props.get("processing_level")


def _asset_href(item: Item, asset_name: str) -> str:
    asset = item.assets[asset_name]
    alternate = getattr(asset, "extra_fields", {}).get("alternate", {})
    https_alt = alternate.get("https", {})
    if isinstance(https_alt, dict) and https_alt.get("href"):
        return str(https_alt["href"])
    return asset.href


def _scene_href(item: Item, preferred_assets: tuple[str, ...]) -> tuple[str, str]:
    for key in preferred_assets:
        if key in item.assets:
            return _asset_href(item, key), key
    if item.assets:
        key = next(iter(item.assets.keys()))
        return _asset_href(item, key), key
    raise KeyError(f"No downloadable asset found for item {item.id}")


def _product_uuid_from_href(href: str) -> str | None:
    match = PRODUCT_ID_RE.search(href)
    if not match:
        return None
    return match.group(1)


def _asset_s3_href(item: Item, asset_name: str) -> str | None:
    asset = item.assets.get(asset_name)
    if asset is None:
        return None
    alternate = getattr(asset, "extra_fields", {}).get("alternate", {})
    candidates: list[str] = []
    if isinstance(asset.href, str):
        candidates.append(asset.href)
    if isinstance(alternate, dict):
        for key in ("s3", "s3+https"):
            value = alternate.get(key)
            if isinstance(value, str):
                candidates.append(value)
            elif isinstance(value, dict):
                href = value.get("href") or value.get("url")
                if isinstance(href, str):
                    candidates.append(href)
    for href in candidates:
        if href.startswith("s3://"):
            return href
    return None


def _s3_path_from_href(href: str) -> str | None:
    parsed = urlparse(href)
    if parsed.scheme != "s3" or not parsed.netloc:
        return None
    path = f"/{parsed.netloc}{parsed.path}".rstrip("/")
    if not path:
        return None
    parts = path.split("/")
    for index, part in enumerate(parts):
        if part.endswith(".SAFE"):
            return "/".join(parts[: index + 1])
    if parts[-1].lower() == "manifest.safe" and len(parts) > 1:
        return "/".join(parts[:-1])
    return path


def _scene_s3_path_from_item(item: Item, preferred_assets: tuple[str, ...]) -> str | None:
    candidates = list(dict.fromkeys((*preferred_assets, "safe_manifest", *item.assets.keys())))
    for asset_name in candidates:
        href = _asset_s3_href(item, asset_name)
        if not href:
            continue
        s3_path = _s3_path_from_href(href)
        if s3_path:
            return s3_path
    return None


def _derived_scene_s3_path(scene: SlcScene) -> str | None:
    if not S1_PRODUCT_NAME_RE.match(scene.product_name):
        return None
    safe_name = scene.product_name if scene.product_name.endswith(".SAFE") else f"{scene.product_name}.SAFE"
    return f"/eodata/Sentinel-1/SAR/SLC/{scene.acquisition_date.replace('-', '/')}/{safe_name}"


def _date_chunks(start: date, end: date, days: int = QUERY_CHUNK_DAYS) -> list[tuple[date, date]]:
    chunks: list[tuple[date, date]] = []
    current = start
    while current <= end:
        chunk_end = min(current + timedelta(days=days - 1), end)
        chunks.append((current, chunk_end))
        current = chunk_end + timedelta(days=1)
    return chunks


def _chunked(values: list[str], size: int) -> list[list[str]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def _iter_search_items(
    client: Client,
    *,
    collection: str,
    geometry,
    start: date,
    end: date,
) -> list[Item]:
    items: list[Item] = []
    for chunk_start, chunk_end in _date_chunks(start, end):
        iso_range = f"{chunk_start.isoformat()}T00:00:00Z/{chunk_end.isoformat()}T23:59:59Z"
        for attempt in range(1, QUERY_RETRIES + 1):
            try:
                search = client.search(
                    collections=[collection],
                    intersects=mapping(geometry),
                    datetime=iso_range,
                    limit=200,
                )
                items.extend(list(search.items()))
                break
            except APIError:
                if attempt == QUERY_RETRIES:
                    raise
                LOGGER.warning(
                    "Retrying SLC search chunk after provider error | datetime=%s attempt=%s",
                    iso_range,
                    attempt,
                )
    return items


def validate_s1_slc_provider_fields(config: PipelineConfig) -> ProviderFieldValidation:
    stack = config.stacks[0]
    geometry = load_aoi_geometry(config.aoi)
    client = Client.open(config.acquisition.s1_catalog_url)
    item = next(
        item
        for item in client.search(
            collections=[config.acquisition.s1_collection],
            intersects=mapping(geometry),
            datetime=config.acquisition.time_window.iso_range,
            limit=20,
        ).items()
        if (_item_direction(item) or "").lower() == stack.direction
        and _item_relative_orbit(item) == stack.relative_orbit
        and (item.properties.get("sar:instrument_mode") or item.properties.get("sensorMode")) == "IW"
        and stack.polarization in _item_polarization(item)
    )
    href, asset_name = _scene_href(item, config.acquisition.slc_asset_priority)
    return ProviderFieldValidation(
        catalog_url=config.acquisition.s1_catalog_url,
        collection=config.acquisition.s1_collection,
        item_id=item.id,
        property_mapping={
            "direction": _item_direction(item),
            "relative_orbit": _item_relative_orbit(item),
            "instrument_mode": item.properties.get("sar:instrument_mode"),
            "polarizations": item.properties.get("sar:polarizations"),
            "product_type": _item_product_type(item),
            "processing_level": _item_processing_level(item),
            "platform": item.properties.get("platform"),
            "datetime": item.properties.get("datetime"),
        },
        asset_mapping={
            "asset_name": asset_name,
            "asset_href": href,
            "available_assets": sorted(item.assets.keys()),
        },
    )


def _select_stack_scenes(scenes: list[SlcScene], stack: OrbitStackConfig) -> list[SlcScene]:
    if len(scenes) < stack.min_scenes:
        raise RuntimeError(
            f"Stack {stack.id} only has {len(scenes)} scenes; need at least {stack.min_scenes} for PSI."
        )

    if stack.scene_limit is None or len(scenes) <= stack.scene_limit:
        return scenes

    limit = stack.scene_limit
    if stack.master_date is None:
        selected = scenes[:limit]
        LOGGER.info(
            "Selected earliest scenes for capped PSI stack | stack_id=%s total=%s selected=%s",
            stack.id,
            len(scenes),
            len(selected),
        )
        return selected

    master_index = next(
        (index for index, scene in enumerate(scenes) if date.fromisoformat(scene.acquisition_date) == stack.master_date),
        None,
    )
    if master_index is None:
        raise ValueError(
            f"Configured master_date {stack.master_date.isoformat()} is not present in stack {stack.id}."
        )

    start = max(min(master_index - (limit // 2), len(scenes) - limit), 0)
    selected = scenes[start : start + limit]
    LOGGER.info(
        "Selected master-centered scenes for capped PSI stack | stack_id=%s total=%s selected=%s master_date=%s start_index=%s",
        stack.id,
        len(scenes),
        len(selected),
        stack.master_date.isoformat(),
        start,
    )
    return selected


def query_s1_slc_stack(config: PipelineConfig, stack: OrbitStackConfig) -> StackManifest:
    geometry = load_aoi_geometry(config.aoi)
    client = Client.open(config.acquisition.s1_catalog_url)
    scenes: list[SlcScene] = []
    items = _iter_search_items(
        client,
        collection=config.acquisition.s1_collection,
        geometry=geometry,
        start=config.acquisition.time_window.start,
        end=config.acquisition.time_window.end,
    )
    for item in sorted(items, key=lambda current: current.datetime):
        direction = _item_direction(item)
        relative_orbit = _item_relative_orbit(item)
        product_type = _item_product_type(item)
        processing_level = _item_processing_level(item)
        if direction is None or relative_orbit is None:
            continue
        if direction.lower() != stack.direction:
            continue
        if relative_orbit != stack.relative_orbit:
            continue
        swath_mode = item.properties.get("sar:instrument_mode") or item.properties.get("sensorMode")
        if swath_mode != "IW":
            continue
        if processing_level not in (None, "L1"):
            continue
        if product_type is not None and "SLC" not in str(product_type):
            continue
        polarization = _item_polarization(item)
        if stack.polarization == "VV" and "VV" not in polarization:
            continue
        if stack.polarization == "VH" and "VH" not in polarization:
            continue
        href, asset_name = _scene_href(item, config.acquisition.slc_asset_priority)
        acquisition_start = item.properties.get("start_datetime") or item.datetime.isoformat()
        acquisition_stop = item.properties.get("end_datetime") or item.datetime.isoformat()
        acquisition_date = datetime.fromisoformat(acquisition_start.replace("Z", "+00:00")).date().isoformat()
        scenes.append(
            SlcScene(
                scene_id=item.id,
                product_name=item.properties.get("title", item.id),
                acquisition_start=acquisition_start,
                acquisition_stop=acquisition_stop,
                acquisition_date=acquisition_date,
                direction=direction.lower(),
                relative_orbit=relative_orbit,
                polarization=polarization,
                swath_mode=swath_mode,
                product_type=product_type or "SLC",
                processing_level=processing_level or "L1",
                platform=str(item.properties.get("platform", "")),
                asset_name=asset_name,
                href=href,
                product_uuid=_product_uuid_from_href(href),
                s3_path=_scene_s3_path_from_item(item, config.acquisition.slc_asset_priority),
            )
        )
    scenes = _select_stack_scenes(scenes, stack)
    return StackManifest(
        stack_id=stack.id,
        direction=stack.direction,
        relative_orbit=stack.relative_orbit,
        product_type="SLC",
        scenes=scenes,
    )


def build_manifests(config: PipelineConfig, context: RunContext) -> dict[str, StackManifest]:
    manifests: dict[str, StackManifest] = {}
    for stack in config.stacks:
        path = stack_manifest_path(context, stack.id)
        if path.exists() and config.cache.reuse_manifests:
            manifests[stack.id] = read_stack_manifest(path)
            continue
        manifest = query_s1_slc_stack(config, stack)
        write_stack_manifest(manifest, path)
        manifests[stack.id] = manifest
    return manifests


def _auth_headers(config: PipelineConfig) -> dict[str, str]:
    session, _ = _authorized_session(config)
    auth = session.headers.get("Authorization")
    session.close()
    if auth:
        return {"Authorization": auth}
    return {}


def _env_candidates(primary: str | None, *fallbacks: str) -> list[str]:
    names: list[str] = []
    if primary:
        names.append(primary)
    for name in fallbacks:
        if name and name not in names:
            names.append(name)
    return names


def _env_value(*names: str) -> tuple[str | None, str | None]:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value, name
    return None, None


def _access_token_candidates(config: PipelineConfig) -> list[str]:
    auth = config.acquisition.auth
    return _env_candidates(auth.bearer_token_env, auth.access_token_env or "ACCESS_TOKEN", "ACCESS_TOKEN")


def _refresh_token_candidates(config: PipelineConfig) -> list[str]:
    auth = config.acquisition.auth
    return _env_candidates(auth.refresh_token_env or "REFRESH_TOKEN", "CDSE_REFRESH_TOKEN", "REFRESH_TOKEN")


def _username_candidates(config: PipelineConfig) -> list[str]:
    auth = config.acquisition.auth
    return _env_candidates(auth.username_env or "CDSE_USERNAME", "CDSE_USERNAME")


def _password_candidates(config: PipelineConfig) -> list[str]:
    auth = config.acquisition.auth
    return _env_candidates(auth.password_env or "CDSE_PASSWORD", "CDSE_PASSWORD")


def _totp_candidates(config: PipelineConfig) -> list[str]:
    auth = config.acquisition.auth
    return _env_candidates(auth.totp_env, "CDSE_TOTP", "TOTP")


def _decode_jwt_payload(token: str) -> dict[str, object] | None:
    parts = token.split(".")
    if len(parts) < 2:
        return None
    payload = parts[1]
    payload += "=" * (-len(payload) % 4)
    try:
        decoded = base64.urlsafe_b64decode(payload.encode("ascii"))
        data = json.loads(decoded.decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _token_seconds_to_expiry(token: str) -> int | None:
    payload = _decode_jwt_payload(token)
    if payload is None:
        return None
    exp = payload.get("exp")
    if not isinstance(exp, (int, float)):
        return None
    return int(exp) - int(time.time())


def _token_is_valid_enough(token: str, margin_seconds: int) -> bool:
    remaining_seconds = _token_seconds_to_expiry(token)
    if remaining_seconds is None:
        return True
    return remaining_seconds > margin_seconds


def _resolve_download_token(config: PipelineConfig, *, require_valid: bool = False) -> tuple[str | None, str | None]:
    margin = config.acquisition.auth.refresh_margin_seconds
    names = _access_token_candidates(config)
    for name in names:
        token = os.environ.get(name)
        if token:
            if require_valid and not _token_is_valid_enough(token, margin):
                continue
            return token, name
    return None, None


def _stale_download_token_env(config: PipelineConfig) -> str | None:
    margin = config.acquisition.auth.refresh_margin_seconds
    for name in _access_token_candidates(config):
        token = os.environ.get(name)
        if token and not _token_is_valid_enough(token, margin):
            return name
    return None


def ensure_download_auth(config: PipelineConfig) -> None:
    if config.acquisition.download_transport in {"auto", "s3"} and _has_s3_credentials(config):
        return
    token, token_env = _resolve_download_token(config)
    if token:
        return
    refresh_token, _ = _env_value(*_refresh_token_candidates(config))
    username, _ = _env_value(*_username_candidates(config))
    password, _ = _env_value(*_password_candidates(config))
    if refresh_token or (username and password):
        return
    raise RuntimeError(
        "CDSE download authentication is required for Sentinel-1 SLC product access. "
        f"Set the bearer token environment variable {config.acquisition.auth.bearer_token_env!r} "
        "or 'ACCESS_TOKEN', or provide refresh-token / username-password environment variables "
        "before running download-slc."
    )


def _partial_download_path(target: Path) -> Path:
    return target.with_name(f"{target.name}.part")


def _stage_download_dir(target: Path) -> Path:
    return target.with_name(f"{target.name}.parts")


def _staged_member_path(stage_dir: Path, member_name: str) -> Path:
    return stage_dir / Path(member_name)


def _staged_member_partial_path(member_path: Path) -> Path:
    return member_path.with_name(f"{member_path.name}.part")


def _remove_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def _store_access_token(config: PipelineConfig, access_token: str, refresh_token: str | None = None) -> str:
    names = _access_token_candidates(config)
    for name in names:
        os.environ[name] = access_token
    if refresh_token:
        for name in _refresh_token_candidates(config):
            os.environ[name] = refresh_token
    return names[0]


def _has_s3_credentials(config: PipelineConfig) -> bool:
    s3 = config.acquisition.s3
    access_key, _ = _env_value(s3.access_key_env)
    secret_key, _ = _env_value(s3.secret_key_env)
    return bool(s3.enabled and access_key and secret_key)


def _require_boto3():
    try:
        import boto3
        from botocore.config import Config as BotoConfig
        from botocore.exceptions import (
            BotoCoreError,
            ClientError,
            ConnectionClosedError,
            ConnectTimeoutError,
            ConnectionError as BotoConnectionError,
            EndpointConnectionError,
            HTTPClientError,
            ReadTimeoutError,
            SSLError,
        )
    except ImportError as exc:
        raise RuntimeError(
            "boto3 is required for CDSE S3 downloads. Install boto3 or use download_transport='odata'."
        ) from exc
    return (
        boto3,
        BotoConfig,
        BotoCoreError,
        ClientError,
        EndpointConnectionError,
        ConnectTimeoutError,
        ReadTimeoutError,
        ConnectionClosedError,
        HTTPClientError,
        SSLError,
        BotoConnectionError,
    )


def _s3_client(config: PipelineConfig, *, endpoint_url: str | None = None):
    (
        boto3,
        BotoConfig,
        _boto_core_error,
        _client_error,
        _endpoint_connection_error,
        _connect_timeout_error,
        _read_timeout_error,
        _connection_closed_error,
        _http_client_error,
        _ssl_error,
        _boto_connection_error,
    ) = _require_boto3()
    s3 = config.acquisition.s3
    access_key, access_env = _env_value(s3.access_key_env)
    secret_key, secret_env = _env_value(s3.secret_key_env)
    if not access_key or not secret_key:
        raise RuntimeError(
            "CDSE S3 download is enabled but S3 credentials are missing. "
            f"Set {s3.access_key_env!r} and {s3.secret_key_env!r}."
        )
    session_token, _ = _env_value(s3.session_token_env) if s3.session_token_env else (None, None)
    resolved_endpoint = endpoint_url or s3.endpoint_url
    LOGGER.info(
        "Creating CDSE S3 client | endpoint=%s bucket=%s access_key_env=%s secret_key_env=%s",
        resolved_endpoint,
        s3.bucket,
        access_env,
        secret_env,
    )
    client = boto3.session.Session().client(
        "s3",
        endpoint_url=resolved_endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        aws_session_token=session_token,
        region_name=s3.region_name,
        config=BotoConfig(
            signature_version="s3v4",
            retries={"max_attempts": s3.max_attempts, "mode": s3.retry_mode},
            connect_timeout=s3.connect_timeout_seconds,
            read_timeout=s3.read_timeout_seconds or config.acquisition.timeout_seconds,
            max_pool_connections=s3.max_pool_connections,
            tcp_keepalive=s3.tcp_keepalive,
            s3={"addressing_style": s3.addressing_style},
        ),
    )
    return client


def _s3_endpoint_candidates(config: PipelineConfig) -> tuple[str, ...]:
    s3 = config.acquisition.s3
    candidates: list[str] = []
    for endpoint in (s3.endpoint_url, *s3.fallback_endpoint_urls):
        if endpoint and endpoint not in candidates:
            candidates.append(endpoint)
    return tuple(candidates)


def _endpoint_resolution_summary(endpoint_url: str) -> str:
    host = urlparse(endpoint_url).hostname
    if not host:
        return "missing-host"
    try:
        results = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    except OSError as exc:
        return f"{host}:unresolved:{exc.__class__.__name__}"
    addresses = sorted({entry[4][0] for entry in results if len(entry) > 4 and entry[4]})
    return f"{host}:resolved:{len(addresses)}"


def _retry_sleep_seconds(attempt_number: int) -> float:
    return min(2 ** (attempt_number - 1), 30) + random.uniform(0.0, 0.5)


def _is_retryable_s3_error(exc: Exception) -> bool:
    try:
        (
            _boto3,
            _boto_config,
            BotoCoreError,
            ClientError,
            EndpointConnectionError,
            ConnectTimeoutError,
            ReadTimeoutError,
            ConnectionClosedError,
            HTTPClientError,
            SSLError,
            BotoConnectionError,
        ) = _require_boto3()
    except RuntimeError:
        return False

    if isinstance(exc, ClientError):
        code = exc.response.get("Error", {}).get("Code")
        return isinstance(code, str) and code in S3_RETRYABLE_ERROR_CODES
    return isinstance(
        exc,
        (
            EndpointConnectionError,
            ConnectTimeoutError,
            ReadTimeoutError,
            ConnectionClosedError,
            HTTPClientError,
            SSLError,
            BotoConnectionError,
            BotoCoreError,
        ),
    )


def _persist_scene_s3_path(
    context: RunContext,
    manifest: StackManifest,
    scene: SlcScene,
    s3_path: str,
) -> tuple[StackManifest, SlcScene]:
    if scene.s3_path == s3_path:
        return manifest, scene
    updated_scene = replace(scene, s3_path=s3_path)
    updated_manifest = update_scene_s3_path(manifest, scene.scene_id, s3_path)
    if updated_manifest is not manifest:
        write_stack_manifest(updated_manifest, stack_manifest_path(context, manifest.stack_id))
    return updated_manifest, updated_scene


def _stac_scene_s3_paths(config: PipelineConfig, scene_ids: list[str]) -> dict[str, str]:
    resolved: dict[str, str] = {}
    if not scene_ids:
        return resolved

    client = Client.open(config.acquisition.s1_catalog_url)
    for batch in _chunked(scene_ids, STAC_LOOKUP_BATCH_SIZE):
        for attempt in range(1, QUERY_RETRIES + 1):
            try:
                search = client.search(
                    collections=[config.acquisition.s1_collection],
                    ids=batch,
                    limit=len(batch),
                )
                for item in search.items():
                    s3_path = _scene_s3_path_from_item(item, config.acquisition.slc_asset_priority)
                    if s3_path:
                        resolved[item.id] = s3_path
                break
            except APIError:
                if attempt == QUERY_RETRIES:
                    raise
                LOGGER.warning(
                    "Retrying STAC S3-path lookup after provider error | batch_size=%s attempt=%s",
                    len(batch),
                    attempt,
                )
                time.sleep(min(2**attempt, 10))
    return resolved


def _enrich_manifest_s3_paths_from_stac(
    config: PipelineConfig,
    context: RunContext,
    manifest: StackManifest,
) -> StackManifest:
    missing_scenes = [scene for scene in manifest.scenes if not scene.s3_path]
    if not missing_scenes:
        return manifest

    resolved = _stac_scene_s3_paths(config, [scene.scene_id for scene in missing_scenes])
    if not resolved:
        return manifest

    updated_manifest = manifest
    updated_count = 0
    for scene in missing_scenes:
        s3_path = resolved.get(scene.scene_id)
        if not s3_path:
            continue
        cache_key = scene.product_uuid or scene.product_name
        S3_PATH_CACHE[cache_key] = s3_path
        next_manifest = update_scene_s3_path(updated_manifest, scene.scene_id, s3_path)
        if next_manifest is updated_manifest:
            continue
        updated_manifest = next_manifest
        updated_count += 1

    if updated_manifest is manifest:
        return manifest

    LOGGER.info(
        "Backfilled S3 paths from STAC manifest assets | stack_id=%s resolved=%s missing=%s",
        manifest.stack_id,
        updated_count,
        len(missing_scenes) - updated_count,
    )
    write_stack_manifest(updated_manifest, stack_manifest_path(context, manifest.stack_id))
    return updated_manifest


def _odata_product_lookup_url(config: PipelineConfig, product_id: str) -> str:
    base = config.acquisition.odata_catalog_url.rstrip("/")
    return f"{base}/Products({product_id})?$select=Id,Name,S3Path,Online"


def _odata_product_search_url(config: PipelineConfig, product_name: str) -> str:
    base = config.acquisition.odata_catalog_url.rstrip("/")
    safe_name = product_name if product_name.endswith(".SAFE") else f"{product_name}.SAFE"
    escaped = safe_name.replace("'", "''")
    return f"{base}/Products?$filter=Name eq '{escaped}'&$select=Id,Name,S3Path,Online"


def _odata_json(url: str, *, timeout_seconds: int) -> dict[str, object]:
    last_error: Exception | None = None
    for attempt in range(1, ODATA_QUERY_RETRIES + 1):
        try:
            response = requests.get(url, timeout=(15, timeout_seconds))
            if response.status_code >= 400:
                raise requests.HTTPError(
                    f"{response.status_code} Client Error: {response.reason} for url: {url}",
                    response=response,
                )
            payload = response.json()
            if not isinstance(payload, dict):
                raise RuntimeError("OData metadata response was not a JSON object.")
            return payload
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            last_error = exc
            if attempt == ODATA_QUERY_RETRIES:
                break
            LOGGER.warning("Retrying OData metadata lookup | attempt=%s error=%s", attempt, exc.__class__.__name__)
            time.sleep(min(2**attempt, 10))
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"OData metadata lookup failed for url: {url}")


def _scene_s3_path(config: PipelineConfig, scene: SlcScene) -> str:
    if scene.s3_path:
        return scene.s3_path
    cache_key = scene.product_uuid or scene.product_name
    if cache_key in S3_PATH_CACHE:
        return S3_PATH_CACHE[cache_key]

    derived_path = _derived_scene_s3_path(scene)
    if derived_path:
        return derived_path

    stac_paths = _stac_scene_s3_paths(config, [scene.scene_id])
    stac_path = stac_paths.get(scene.scene_id)
    if stac_path:
        S3_PATH_CACHE[cache_key] = stac_path
        return stac_path

    payload: dict[str, object]
    if scene.product_uuid:
        payload = _odata_json(
            _odata_product_lookup_url(config, scene.product_uuid),
            timeout_seconds=config.acquisition.timeout_seconds,
        )
        s3_path = payload.get("S3Path")
        if isinstance(s3_path, str) and s3_path:
            S3_PATH_CACHE[cache_key] = s3_path
            return s3_path

    payload = _odata_json(
        _odata_product_search_url(config, scene.product_name),
        timeout_seconds=config.acquisition.timeout_seconds,
    )
    values = payload.get("value")
    if isinstance(values, list):
        for item in values:
            if not isinstance(item, dict):
                continue
            name = item.get("Name")
            s3_path = item.get("S3Path")
            if isinstance(name, str) and isinstance(s3_path, str):
                product_name = scene.product_name if scene.product_name.endswith(".SAFE") else f"{scene.product_name}.SAFE"
                if name == product_name:
                    S3_PATH_CACHE[cache_key] = s3_path
                    return s3_path
    raise FileNotFoundError(f"Could not resolve S3Path for product {scene.product_name}")


def _normalize_s3_prefix(s3_path: str) -> str:
    normalized = s3_path.strip()
    if normalized.startswith("/"):
        normalized = normalized[1:]
    if normalized.startswith("eodata/"):
        normalized = normalized[len("eodata/") :]
    return normalized.rstrip("/")


def _relative_zip_member_name(key: str, prefix: str) -> str:
    parent = prefix.rsplit("/", 1)[0]
    return key[len(parent) + 1 :] if parent else key


def _iter_s3_objects(client, *, bucket: str, prefix: str) -> list[S3ObjectInfo]:
    paginator = client.get_paginator("list_objects_v2")
    objects: list[S3ObjectInfo] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/"):
        for entry in page.get("Contents", []):
            key = entry.get("Key")
            if not isinstance(key, str) or key.endswith("/"):
                continue
            size = entry.get("Size")
            if not isinstance(size, int):
                head = client.head_object(Bucket=bucket, Key=key)
                size = int(head["ContentLength"])
            objects.append(
                S3ObjectInfo(
                    key=key,
                    size=size,
                    member_name=_relative_zip_member_name(key, prefix),
                )
            )
    return sorted(objects, key=lambda current: current.key)


def _finalize_staged_member(member_path: Path, *, expected_size: int) -> bool:
    partial_path = _staged_member_partial_path(member_path)

    if member_path.exists():
        actual_size = member_path.stat().st_size
        if actual_size == expected_size:
            return True
        LOGGER.warning(
            "Removing invalid staged S3 member with unexpected size | path=%s expected=%s actual=%s",
            member_path,
            expected_size,
            actual_size,
        )
        member_path.unlink()

    if partial_path.exists():
        partial_size = partial_path.stat().st_size
        if partial_size == expected_size:
            member_path.parent.mkdir(parents=True, exist_ok=True)
            partial_path.replace(member_path)
            return True
        if partial_size > expected_size:
            LOGGER.warning(
                "Removing oversized staged S3 member partial | path=%s expected=%s actual=%s",
                partial_path,
                expected_size,
                partial_size,
            )
            partial_path.unlink()

    return False


def _resume_offset_for_member(member_path: Path, *, expected_size: int) -> int:
    partial_path = _staged_member_partial_path(member_path)
    if not partial_path.exists():
        return 0
    current_size = partial_path.stat().st_size
    if current_size > expected_size:
        LOGGER.warning(
            "Discarding oversized partial S3 member before retry | path=%s expected=%s actual=%s",
            partial_path,
            expected_size,
            current_size,
        )
        partial_path.unlink()
        return 0
    return current_size


def _response_status_code(response: dict[str, object]) -> int | None:
    metadata = response.get("ResponseMetadata")
    if isinstance(metadata, dict):
        status_code = metadata.get("HTTPStatusCode")
        if isinstance(status_code, int):
            return status_code
    return None


def _staged_member_bytes(member_path: Path, *, expected_size: int) -> int:
    if member_path.exists():
        return min(member_path.stat().st_size, expected_size)
    partial_path = _staged_member_partial_path(member_path)
    if partial_path.exists():
        return min(partial_path.stat().st_size, expected_size)
    return 0


def _member_endpoint_attempt_budget(
    config: PipelineConfig,
    *,
    remaining_bytes: int,
    endpoint_count: int,
) -> int:
    configured_attempts = max(config.acquisition.s3.download_attempts, 1) * max(endpoint_count, 1)
    budget = max(configured_attempts, S3_MEMBER_MIN_ENDPOINT_ATTEMPTS)
    if remaining_bytes <= S3_MEMBER_TAIL_REMAINING_BYTES:
        budget += S3_MEMBER_TAIL_BONUS_ATTEMPTS
    return budget


def _member_download_chunk_size(failure_count: int) -> int:
    if failure_count >= 12:
        return S3_MEMBER_MIN_CHUNK_SIZE
    if failure_count >= 8:
        return 1 * 1024 * 1024
    if failure_count >= 4:
        return 2 * 1024 * 1024
    if failure_count >= 2:
        return 4 * 1024 * 1024
    return DOWNLOAD_CHUNK_SIZE


def _member_retry_sleep_seconds(attempt_number: int, *, remaining_bytes: int) -> float:
    base = min(1.6 ** max(attempt_number - 1, 0), 20.0)
    if remaining_bytes <= S3_MEMBER_TAIL_REMAINING_BYTES:
        base = min(base, 8.0)
    return base + random.uniform(0.0, min(base * 0.25, 1.5))


def _range_resume_supported(response: dict[str, object], *, offset: int, expected_size: int) -> bool:
    if offset <= 0:
        return True
    status_code = _response_status_code(response)
    content_range = response.get("ContentRange")
    if status_code != 206 or not isinstance(content_range, str):
        return False
    return content_range.startswith(f"bytes {offset}-") and content_range.endswith(f"/{expected_size}")


def _download_s3_object_with_resume(
    config: PipelineConfig,
    *,
    bucket: str,
    object_info: S3ObjectInfo,
    stage_dir: Path,
    stack_id: str,
    scene_id: str,
) -> None:
    member_path = _staged_member_path(stage_dir, object_info.member_name)
    if _finalize_staged_member(member_path, expected_size=object_info.size):
        return

    endpoints = _s3_endpoint_candidates(config)
    endpoint_attempt = 0
    last_error: Exception | None = None

    endpoint_count = max(len(endpoints), 1)
    while True:
        if _finalize_staged_member(member_path, expected_size=object_info.size):
            return

        preserved_bytes = _staged_member_bytes(member_path, expected_size=object_info.size)
        remaining_bytes = max(object_info.size - preserved_bytes, 0)
        total_endpoint_attempts = _member_endpoint_attempt_budget(
            config,
            remaining_bytes=remaining_bytes,
            endpoint_count=endpoint_count,
        )
        if endpoint_attempt >= total_endpoint_attempts:
            break

        endpoint_index = endpoint_attempt % endpoint_count
        endpoint_url = endpoints[endpoint_index]
        endpoint_attempt += 1
        attempt_number = endpoint_attempt
        failure_count = attempt_number - 1
        chunk_size = _member_download_chunk_size(failure_count)
        client = _s3_client(config, endpoint_url=endpoint_url)
        body = None
        partial_path = _staged_member_partial_path(member_path)
        try:
            offset = _resume_offset_for_member(member_path, expected_size=object_info.size)
            request = {"Bucket": bucket, "Key": object_info.key}
            if offset > 0:
                request["Range"] = f"bytes={offset}-"

            LOGGER.info(
                "%s staged S3 member download | stack_id=%s scene_id=%s member=%s preserved=%s total=%s remaining=%s chunk_size=%s endpoint=%s attempt=%s/%s endpoint_index=%s/%s",
                "Resuming" if offset > 0 else "Starting",
                stack_id,
                scene_id,
                object_info.member_name,
                offset,
                object_info.size,
                max(object_info.size - offset, 0),
                chunk_size,
                endpoint_url,
                attempt_number,
                total_endpoint_attempts,
                endpoint_index + 1,
                endpoint_count,
            )

            response = client.get_object(**request)
            if offset > 0 and not _range_resume_supported(response, offset=offset, expected_size=object_info.size):
                raise RuntimeError(
                    "CDSE S3 endpoint did not honor ranged get_object resume request. "
                    f"member={object_info.member_name!r} offset={offset} endpoint={endpoint_url!r}"
                )

            body = response["Body"]
            partial_path.parent.mkdir(parents=True, exist_ok=True)
            with partial_path.open("ab" if offset > 0 else "wb") as handle:
                while True:
                    chunk = body.read(chunk_size)
                    if not chunk:
                        break
                    handle.write(chunk)

            actual_size = partial_path.stat().st_size
            if actual_size != object_info.size:
                raise OSError(
                    "Incomplete staged S3 member download. "
                    f"member={object_info.member_name!r} expected={object_info.size} actual={actual_size}"
                )

            partial_path.replace(member_path)
            return
        except RuntimeError as exc:
            last_error = exc
            preserved_after = _staged_member_bytes(member_path, expected_size=object_info.size)
            remaining_after = max(object_info.size - preserved_after, 0)
            updated_budget = _member_endpoint_attempt_budget(
                config,
                remaining_bytes=remaining_after,
                endpoint_count=endpoint_count,
            )
            if "did not honor ranged get_object resume request" not in str(exc) or attempt_number >= updated_budget:
                LOGGER.error(
                    "Exhausted staged S3 member retries after range-resume mismatch | stack_id=%s scene_id=%s member=%s preserved=%s total=%s remaining=%s endpoint=%s attempt=%s/%s",
                    stack_id,
                    scene_id,
                    object_info.member_name,
                    preserved_after,
                    object_info.size,
                    remaining_after,
                    endpoint_url,
                    attempt_number,
                    updated_budget,
                )
                raise
            LOGGER.warning(
                "Retrying staged S3 member on alternate endpoint after range-resume mismatch | stack_id=%s scene_id=%s member=%s preserved=%s total=%s remaining=%s endpoint=%s attempt=%s/%s endpoint_index=%s/%s",
                stack_id,
                scene_id,
                object_info.member_name,
                preserved_after,
                object_info.size,
                remaining_after,
                endpoint_url,
                attempt_number,
                updated_budget,
                endpoint_index + 1,
                endpoint_count,
            )
            time.sleep(_member_retry_sleep_seconds(attempt_number, remaining_bytes=remaining_after))
        except Exception as exc:
            last_error = exc
            preserved_after = _staged_member_bytes(member_path, expected_size=object_info.size)
            remaining_after = max(object_info.size - preserved_after, 0)
            updated_budget = _member_endpoint_attempt_budget(
                config,
                remaining_bytes=remaining_after,
                endpoint_count=endpoint_count,
            )
            if not _is_retryable_s3_error(exc) or attempt_number >= updated_budget:
                LOGGER.error(
                    "Exhausted staged S3 member retries after transport/storage error | stack_id=%s scene_id=%s member=%s preserved=%s total=%s remaining=%s endpoint=%s attempt=%s/%s dns=%s error=%s",
                    stack_id,
                    scene_id,
                    object_info.member_name,
                    preserved_after,
                    object_info.size,
                    remaining_after,
                    endpoint_url,
                    attempt_number,
                    updated_budget,
                    _endpoint_resolution_summary(endpoint_url),
                    exc.__class__.__name__,
                )
                raise
            LOGGER.warning(
                "Retrying staged S3 member after transport/storage error | stack_id=%s scene_id=%s member=%s preserved=%s total=%s remaining=%s endpoint=%s attempt=%s/%s endpoint_index=%s/%s dns=%s error=%s",
                stack_id,
                scene_id,
                object_info.member_name,
                preserved_after,
                object_info.size,
                remaining_after,
                endpoint_url,
                attempt_number,
                updated_budget,
                endpoint_index + 1,
                endpoint_count,
                _endpoint_resolution_summary(endpoint_url),
                exc.__class__.__name__,
            )
            time.sleep(_member_retry_sleep_seconds(attempt_number, remaining_bytes=remaining_after))
        finally:
            if body is not None:
                body.close()
            close_client = getattr(client, "close", None)
            if callable(close_client):
                close_client()

    if last_error is not None:
        raise last_error


def _validate_scene_zip(zip_path: Path, *, objects: list[S3ObjectInfo]) -> None:
    expected_sizes = {current.member_name: current.size for current in objects}
    with zipfile.ZipFile(zip_path) as archive:
        observed_sizes = {info.filename: info.file_size for info in archive.infolist()}

    if set(observed_sizes) != set(expected_sizes):
        missing = sorted(set(expected_sizes) - set(observed_sizes))
        extra = sorted(set(observed_sizes) - set(expected_sizes))
        raise RuntimeError(
            "Final scene ZIP members do not match staged SAFE contents. "
            f"missing={missing!r} extra={extra!r}"
        )

    for member_name, expected_size in expected_sizes.items():
        actual_size = observed_sizes.get(member_name)
        if actual_size != expected_size:
            raise RuntimeError(
                "Final scene ZIP member size mismatch. "
                f"member={member_name!r} expected={expected_size} actual={actual_size}"
            )


def _assemble_scene_zip_from_stage(
    *,
    stage_dir: Path,
    objects: list[S3ObjectInfo],
    partial: Path,
) -> None:
    _remove_if_exists(partial)
    with zipfile.ZipFile(partial, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for object_info in objects:
            member_path = _staged_member_path(stage_dir, object_info.member_name)
            if not member_path.exists():
                raise FileNotFoundError(
                    f"Missing staged S3 member {object_info.member_name!r} while assembling final ZIP."
                )
            with member_path.open("rb") as handle, archive.open(object_info.member_name, mode="w", force_zip64=True) as zip_member:
                shutil.copyfileobj(handle, zip_member, DOWNLOAD_CHUNK_SIZE)
    _validate_scene_zip(partial, objects=objects)


def _list_s3_scene_objects(
    config: PipelineConfig,
    *,
    bucket: str,
    prefix: str,
    stack_id: str,
    scene_id: str,
) -> tuple[list[S3ObjectInfo], str]:
    endpoints = _s3_endpoint_candidates(config)
    total_attempts = max(config.acquisition.s3.download_attempts, 1)
    total_endpoint_attempts = max(len(endpoints), 1) * total_attempts
    endpoint_attempt = 0
    last_error: Exception | None = None

    for attempt in range(1, total_attempts + 1):
        for endpoint_index, endpoint_url in enumerate(endpoints, start=1):
            endpoint_attempt += 1
            client = _s3_client(config, endpoint_url=endpoint_url)
            try:
                objects = _iter_s3_objects(client, bucket=bucket, prefix=prefix)
                if not objects:
                    raise FileNotFoundError(
                        f"CDSE S3 path {prefix!r} did not return any objects in bucket {bucket!r}."
                    )
                LOGGER.info(
                    "Downloading SLC scene via CDSE S3 | stack_id=%s scene_id=%s object_count=%s prefix=%s endpoint=%s attempt=%s/%s endpoint_index=%s/%s",
                    stack_id,
                    scene_id,
                    len(objects),
                    prefix,
                    endpoint_url,
                    endpoint_attempt,
                    total_endpoint_attempts,
                    endpoint_index,
                    len(endpoints),
                )
                return objects, endpoint_url
            except Exception as exc:
                last_error = exc
                if not _is_retryable_s3_error(exc) or endpoint_attempt >= total_endpoint_attempts:
                    raise
                LOGGER.warning(
                    "Retrying S3 scene listing after transport/storage error | stack_id=%s scene_id=%s attempt=%s/%s endpoint=%s endpoint_index=%s/%s prefix=%s dns=%s error=%s",
                    stack_id,
                    scene_id,
                    endpoint_attempt,
                    total_endpoint_attempts,
                    endpoint_url,
                    endpoint_index,
                    len(endpoints),
                    prefix,
                    _endpoint_resolution_summary(endpoint_url),
                    exc.__class__.__name__,
                )
                time.sleep(_retry_sleep_seconds(endpoint_attempt))
            finally:
                close_client = getattr(client, "close", None)
                if callable(close_client):
                    close_client()

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Failed to list CDSE S3 scene objects for {scene_id}")


def _download_scene_via_s3(config: PipelineConfig, scene: SlcScene, destination: Path, *, stack_id: str) -> None:
    partial = _partial_download_path(destination)
    stage_dir = _stage_download_dir(destination)
    prefix = _normalize_s3_prefix(_scene_s3_path(config, scene))
    bucket = config.acquisition.s3.bucket
    objects, _endpoint_url = _list_s3_scene_objects(
        config,
        bucket=bucket,
        prefix=prefix,
        stack_id=stack_id,
        scene_id=scene.scene_id,
    )
    stage_dir.mkdir(parents=True, exist_ok=True)
    for object_info in objects:
        _download_s3_object_with_resume(
            config,
            bucket=bucket,
            object_info=object_info,
            stage_dir=stage_dir,
            stack_id=stack_id,
            scene_id=scene.scene_id,
        )
    _assemble_scene_zip_from_stage(stage_dir=stage_dir, objects=objects, partial=partial)
    _remove_if_exists(destination)
    partial.replace(destination)
    shutil.rmtree(stage_dir, ignore_errors=True)


def _request_cdse_token(config: PipelineConfig, payload: dict[str, str], *, grant_name: str) -> tuple[str, str | None]:
    try:
        response = requests.post(
            config.acquisition.auth.token_endpoint,
            data=payload,
            timeout=(10, 30),
        )
    except requests.RequestException as exc:
        raise RuntimeError(
            f"CDSE token acquisition failed via {grant_name} grant: {exc.__class__.__name__}"
        ) from None
    if response.status_code >= 400:
        raise RuntimeError(
            f"CDSE token acquisition failed via {grant_name} grant with HTTP {response.status_code}."
        )
    try:
        body = response.json()
    except ValueError:
        raise RuntimeError(f"CDSE token acquisition via {grant_name} grant returned non-JSON content.") from None
    access_token = body.get("access_token")
    if not isinstance(access_token, str) or not access_token:
        raise RuntimeError(f"CDSE token acquisition via {grant_name} grant did not return an access_token.")
    refresh_token = body.get("refresh_token")
    return access_token, refresh_token if isinstance(refresh_token, str) and refresh_token else None


def _acquire_fresh_access_token(config: PipelineConfig, *, reason: str) -> tuple[str, str]:
    auth = config.acquisition.auth
    refresh_token, refresh_env = _env_value(*_refresh_token_candidates(config))
    if refresh_token:
        LOGGER.info(
            "Refreshing CDSE access token via refresh-token grant | reason=%s refresh_env=%s",
            reason,
            refresh_env,
        )
        access_token, new_refresh_token = _request_cdse_token(
            config,
            {
                "grant_type": "refresh_token",
                "client_id": auth.client_id,
                "refresh_token": refresh_token,
            },
            grant_name="refresh_token",
        )
        access_env = _store_access_token(config, access_token, new_refresh_token or refresh_token)
        return access_token, access_env

    username, username_env = _env_value(*_username_candidates(config))
    password, password_env = _env_value(*_password_candidates(config))
    if username and password:
        LOGGER.info(
            "Refreshing CDSE access token via password grant | reason=%s username_env=%s password_env=%s",
            reason,
            username_env,
            password_env,
        )
        payload = {
            "grant_type": "password",
            "client_id": auth.client_id,
            "username": username,
            "password": password,
        }
        totp, _ = _env_value(*_totp_candidates(config))
        if totp:
            payload["totp"] = totp
        access_token, refresh_token = _request_cdse_token(
            config,
            payload,
            grant_name="password",
        )
        access_env = _store_access_token(config, access_token, refresh_token)
        return access_token, access_env

    raise RuntimeError(
        "CDSE access token is expired or near expiry and automatic renewal is not configured. "
        "Provide a refresh token or username/password in environment variables."
    )


def _authorized_session(config: PipelineConfig, *, force_refresh: bool = False) -> tuple[requests.Session, str | None]:
    if force_refresh:
        token, token_env = _acquire_fresh_access_token(config, reason="forced_refresh")
    else:
        token, token_env = _resolve_download_token(config, require_valid=True)
        if not token:
            current_env = _stale_download_token_env(config)
            reason = "missing_access_token"
            if current_env:
                reason = "expired_or_near_expiry"
                LOGGER.info(
                    "Renewing CDSE access token before download | token_env=%s refresh_margin_seconds=%s",
                    current_env,
                    config.acquisition.auth.refresh_margin_seconds,
                )
            token, token_env = _acquire_fresh_access_token(config, reason=reason)
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {token}"})
    return session, token_env


def _stream_download(
    href: str,
    destination: Path,
    timeout_seconds: int,
    *,
    config: PipelineConfig,
    scene_id: str,
    stack_id: str,
) -> None:
    partial = _partial_download_path(destination)
    last_error: Exception | None = None
    force_refresh = False

    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        _remove_if_exists(partial)
        session, token_env = _authorized_session(config, force_refresh=force_refresh)
        try:
            with session.get(
                href,
                stream=True,
                allow_redirects=True,
                timeout=(30, timeout_seconds),
            ) as response:
                status_code = response.status_code
                if status_code in DOWNLOAD_RETRYABLE_STATUS_CODES and attempt < DOWNLOAD_RETRIES:
                    force_refresh = status_code == 401
                    LOGGER.warning(
                        "Retrying SLC download after HTTP status | stack_id=%s scene_id=%s status=%s attempt=%s token_env=%s",
                        stack_id,
                        scene_id,
                        status_code,
                        attempt,
                        token_env,
                    )
                    time.sleep(min(2**attempt, 10))
                    continue
                if status_code >= 400:
                    reason = getattr(response, "reason", "HTTP Error")
                    message = f"{status_code} Client Error: {reason} for url: {href}"
                    session.close()
                    token_env = None
                    raise requests.HTTPError(message, response=response)
                with partial.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                        if chunk:
                            handle.write(chunk)
            _remove_if_exists(destination)
            partial.replace(destination)
            return
        except requests.HTTPError as exc:
            last_error = exc
            _remove_if_exists(partial)
            status_code = exc.response.status_code if exc.response is not None else None
            if status_code in DOWNLOAD_RETRYABLE_STATUS_CODES and attempt < DOWNLOAD_RETRIES:
                force_refresh = status_code == 401
                LOGGER.warning(
                    "Retrying SLC download after HTTP error | stack_id=%s scene_id=%s status=%s attempt=%s",
                    stack_id,
                    scene_id,
                    status_code,
                    attempt,
                )
                time.sleep(min(2**attempt, 10))
                continue
            raise
        except (requests.RequestException, OSError) as exc:
            last_error = exc
            _remove_if_exists(partial)
            if attempt < DOWNLOAD_RETRIES:
                LOGGER.warning(
                    "Retrying SLC download after transport error | stack_id=%s scene_id=%s attempt=%s error=%s",
                    stack_id,
                    scene_id,
                    attempt,
                    exc.__class__.__name__,
                )
                time.sleep(min(2**attempt, 10))
                continue
            raise
        finally:
            session.close()

    if last_error is not None:
        raise last_error


def _download_transport_for_scene(config: PipelineConfig, scene: SlcScene) -> str:
    mode = config.acquisition.download_transport
    if mode == "s3":
        return "s3"
    if mode == "odata":
        return "odata"
    if _has_s3_credentials(config):
        return "s3"
    return "odata"


def _candidate_reused_zip_paths(context: RunContext, stack_id: str, scene: SlcScene) -> list[Path]:
    target = slc_scene_zip_path(context, stack_id, scene).resolve()
    output_root = context.root.parent.parent
    candidates: list[Path] = []
    for candidate in output_root.glob(f"**/raw/slc/{stack_id}/{scene.product_name}.zip"):
        resolved = candidate.resolve()
        if resolved == target:
            continue
        try:
            if not resolved.is_file() or resolved.stat().st_size <= 0:
                continue
        except OSError:
            continue
        candidates.append(resolved)
    candidates.sort(key=lambda path: (path.stat().st_mtime, str(path)), reverse=True)
    return candidates


def _materialize_reused_zip(source: Path, target: Path) -> str:
    try:
        os.link(source, target)
        return "hardlink"
    except OSError as exc:
        if exc.errno not in {errno.EXDEV, errno.EPERM, errno.EACCES, errno.EMLINK, errno.ENOTSUP}:
            raise
    try:
        target.symlink_to(source)
        return "symlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def _reuse_existing_scene_download(
    context: RunContext,
    stack_id: str,
    scene: SlcScene,
) -> tuple[Path, str] | None:
    target = slc_scene_zip_path(context, stack_id, scene)
    for candidate in _candidate_reused_zip_paths(context, stack_id, scene):
        mode = _materialize_reused_zip(candidate, target)
        return candidate, mode
    return None


def download_stack_scenes(config: PipelineConfig, context: RunContext, manifest: StackManifest) -> list[DownloadRecord]:
    records: list[DownloadRecord] = []
    pending_scene_ids: list[str] = []
    for scene in manifest.scenes:
        target = slc_scene_zip_path(context, manifest.stack_id, scene)
        target.parent.mkdir(parents=True, exist_ok=True)
        records.append(DownloadRecord(scene_id=scene.scene_id, href=scene.href, path=target))
        if target.exists() and target.stat().st_size > 0 and config.cache.reuse_downloads:
            LOGGER.info("Reusing SLC download %s", target)
            continue
        if config.cache.reuse_downloads:
            reused = _reuse_existing_scene_download(context, manifest.stack_id, scene)
            if reused is not None:
                source, mode = reused
                LOGGER.info(
                    "Reused SLC download from prior run | stack_id=%s scene_id=%s source=%s target=%s mode=%s",
                    manifest.stack_id,
                    scene.scene_id,
                    source,
                    target,
                    mode,
                )
                continue
        pending_scene_ids.append(scene.scene_id)

    if pending_scene_ids:
        ensure_download_auth(config)

    scenes_by_id = {scene.scene_id: scene for scene in manifest.scenes}
    for scene_id in pending_scene_ids:
        scene = scenes_by_id[scene_id]
        target = slc_scene_zip_path(context, manifest.stack_id, scene)
        partial = _partial_download_path(target)
        if partial.exists():
            LOGGER.warning("Removing stale partial SLC download %s", partial)
            partial.unlink()
        transport = _download_transport_for_scene(config, scene)
        LOGGER.info(
            "Downloading SLC scene | stack_id=%s scene_id=%s transport=%s href=%s",
            manifest.stack_id,
            scene.scene_id,
            transport,
            scene.href,
        )
        if transport == "s3":
            s3_path = _scene_s3_path(config, scene)
            scene_for_download = replace(scene, s3_path=s3_path)
            _download_scene_via_s3(config, scene_for_download, target, stack_id=manifest.stack_id)
            manifest, _scene = _persist_scene_s3_path(context, manifest, scene, s3_path)
            scenes_by_id[scene.scene_id] = replace(scene, s3_path=s3_path)
        else:
            _stream_download(
                scene.href,
                target,
                config.acquisition.timeout_seconds,
                config=config,
                scene_id=scene.scene_id,
                stack_id=manifest.stack_id,
            )
    return records
