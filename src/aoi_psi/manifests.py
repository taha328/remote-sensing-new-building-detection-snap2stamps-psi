from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import date, datetime
from pathlib import Path
from typing import Any
import json

from aoi_psi.run_context import RunContext


@dataclass(frozen=True)
class SlcScene:
    scene_id: str
    product_name: str
    acquisition_start: str
    acquisition_stop: str
    acquisition_date: str
    direction: str
    relative_orbit: int
    polarization: str
    swath_mode: str
    product_type: str
    processing_level: str
    platform: str
    asset_name: str
    href: str
    product_uuid: str | None = None
    s3_path: str | None = None
    local_path: str | None = None


@dataclass(frozen=True)
class StackManifest:
    stack_id: str
    direction: str
    relative_orbit: int
    product_type: str
    scenes: list[SlcScene]


def stack_manifest_path(context: RunContext, stack_id: str) -> Path:
    return context.manifests_dir / f"{stack_id}.json"


def write_stack_manifest(manifest: StackManifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stack_id": manifest.stack_id,
        "direction": manifest.direction,
        "relative_orbit": manifest.relative_orbit,
        "product_type": manifest.product_type,
        "scene_count": len(manifest.scenes),
        "scenes": [asdict(scene) for scene in manifest.scenes],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def read_stack_manifest(path: Path) -> StackManifest:
    payload = json.loads(path.read_text())
    return StackManifest(
        stack_id=payload["stack_id"],
        direction=payload["direction"],
        relative_orbit=int(payload["relative_orbit"]),
        product_type=payload["product_type"],
        scenes=[SlcScene(**scene) for scene in payload["scenes"]],
    )


def update_scene_s3_path(manifest: StackManifest, scene_id: str, s3_path: str) -> StackManifest:
    updated = False
    scenes: list[SlcScene] = []
    for scene in manifest.scenes:
        if scene.scene_id == scene_id and scene.s3_path != s3_path:
            scenes.append(replace(scene, s3_path=s3_path))
            updated = True
            continue
        scenes.append(scene)
    if not updated:
        return manifest
    return StackManifest(
        stack_id=manifest.stack_id,
        direction=manifest.direction,
        relative_orbit=manifest.relative_orbit,
        product_type=manifest.product_type,
        scenes=scenes,
    )


def slc_scene_zip_path(context: RunContext, stack_id: str, scene: SlcScene) -> Path:
    return context.slc_dir / stack_id / f"{scene.product_name}.zip"


def snap_stack_dir(context: RunContext, stack_id: str) -> Path:
    return context.snap_dir / stack_id


def stamps_stack_dir(context: RunContext, stack_id: str) -> Path:
    return context.stamps_dir / stack_id


def manifest_summary(manifest: StackManifest) -> dict[str, Any]:
    return {
        "stack_id": manifest.stack_id,
        "direction": manifest.direction,
        "relative_orbit": manifest.relative_orbit,
        "scene_count": len(manifest.scenes),
        "start": manifest.scenes[0].acquisition_start if manifest.scenes else None,
        "end": manifest.scenes[-1].acquisition_stop if manifest.scenes else None,
    }


def scene_start_datetime(scene: SlcScene) -> datetime:
    return datetime.fromisoformat(scene.acquisition_start.replace("Z", "+00:00"))


def select_master_scene(manifest: StackManifest, master_date: date) -> SlcScene:
    for scene in manifest.scenes:
        if date.fromisoformat(scene.acquisition_date) == master_date:
            return scene
    raise ValueError(
        f"Configured master_date {master_date.isoformat()} is not present in stack {manifest.stack_id}."
    )


def secondary_scenes(manifest: StackManifest, master_scene: SlcScene) -> list[SlcScene]:
    return [scene for scene in manifest.scenes if scene.scene_id != master_scene.scene_id]
