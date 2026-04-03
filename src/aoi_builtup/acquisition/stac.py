from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import planetary_computer
import pystac
from pystac_client import Client
from shapely.geometry import mapping

from aoi_builtup.config import PipelineConfig, Sentinel1Config, Sentinel2Config, TimeWindow

LOGGER = logging.getLogger(__name__)
STAC_API_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


def open_catalog() -> Client:
    return Client.open(STAC_API_URL)


def _normalize_token(value: str | None) -> str:
    if value is None:
        return ""
    return value.replace("-", "").replace("_", "").lower()


def _sort_items(items: Iterable[pystac.Item]) -> list[pystac.Item]:
    return sorted(items, key=lambda item: (item.datetime or datetime.min, item.id))


def query_sentinel1(
    catalog: Client,
    geometry: Any,
    config: Sentinel1Config,
    window: TimeWindow,
) -> list[pystac.Item]:
    query: dict[str, Any] = {"sar:instrument_mode": {"eq": config.instrument_mode}}
    if config.orbit_state is not None:
        query["sat:orbit_state"] = {"eq": config.orbit_state}

    search = catalog.search(
        collections=[config.collection],
        intersects=mapping(geometry),
        datetime=window.stac_datetime,
        query=query,
    )
    items = _sort_items(search.items())
    filtered: list[pystac.Item] = []
    required_pols = {_normalize_token(pol) for pol in config.polarizations}
    platform_target = _normalize_token(config.platform)

    for item in items:
        platform = _normalize_token(item.properties.get("platform"))
        polarizations = {
            _normalize_token(value) for value in item.properties.get("sar:polarizations", [])
        }
        if platform_target and platform != platform_target:
            continue
        if not required_pols.issubset(polarizations):
            continue
        filtered.append(item)

    LOGGER.info("Selected %s Sentinel-1 items for %s", len(filtered), window.stac_datetime)
    return filtered


def query_sentinel2(
    catalog: Client,
    geometry: Any,
    config: Sentinel2Config,
    window: TimeWindow,
) -> list[pystac.Item]:
    search = catalog.search(
        collections=[config.collection],
        intersects=mapping(geometry),
        datetime=window.stac_datetime,
        query={"eo:cloud_cover": {"lt": config.max_cloud_cover}},
    )
    items = _sort_items(search.items())
    LOGGER.info("Selected %s Sentinel-2 items for %s", len(items), window.stac_datetime)
    return items


def save_manifest(items: list[pystac.Item], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"items": [item.to_dict() for item in items]}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def load_manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    return list(payload["items"])


def load_or_query_manifest(
    path: Path,
    reuse_existing: bool,
    query_fn: Callable[[], list[pystac.Item]],
) -> list[dict[str, Any]]:
    if reuse_existing and path.exists():
        LOGGER.info("Reusing manifest %s", path)
        return load_manifest(path)
    items = query_fn()
    save_manifest(items, path)
    return [item.to_dict() for item in items]


def sign_manifest_items(item_dicts: list[dict[str, Any]]) -> list[pystac.Item]:
    signed_items: list[pystac.Item] = []
    for item_dict in item_dicts:
        signed_item = planetary_computer.sign(pystac.Item.from_dict(item_dict))
        signed_items.append(signed_item)
    return signed_items


def manifest_path(root: Path, sensor: str, period_id: str, phase: str) -> Path:
    return root / f"{sensor}_{period_id}_{phase}.json"


def build_period_manifests(
    config: PipelineConfig,
    manifests_dir: Path,
    geometry: Any,
    include_s2: bool = True,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    catalog = open_catalog()
    manifest_index: dict[str, dict[str, list[dict[str, Any]]]] = {}

    for period in config.periods:
        manifest_index[period.id] = {}
        manifest_index[period.id]["s1_before"] = load_or_query_manifest(
            manifest_path(manifests_dir, "s1", period.id, "before"),
            reuse_existing=config.cache.reuse_manifests,
            query_fn=lambda period=period: query_sentinel1(
                catalog,
                geometry,
                config.sentinel1,
                period.before,
            ),
        )
        manifest_index[period.id]["s1_after"] = load_or_query_manifest(
            manifest_path(manifests_dir, "s1", period.id, "after"),
            reuse_existing=config.cache.reuse_manifests,
            query_fn=lambda period=period: query_sentinel1(
                catalog,
                geometry,
                config.sentinel1,
                period.after,
            ),
        )
        if include_s2:
            manifest_index[period.id]["s2_before"] = load_or_query_manifest(
                manifest_path(manifests_dir, "s2", period.id, "before"),
                reuse_existing=config.cache.reuse_manifests,
                query_fn=lambda period=period: query_sentinel2(
                    catalog,
                    geometry,
                    config.sentinel2,
                    period.before,
                ),
            )
            manifest_index[period.id]["s2_after"] = load_or_query_manifest(
                manifest_path(manifests_dir, "s2", period.id, "after"),
                reuse_existing=config.cache.reuse_manifests,
                query_fn=lambda period=period: query_sentinel2(
                    catalog,
                    geometry,
                    config.sentinel2,
                    period.after,
                ),
            )
    return manifest_index
