from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd

from aoi_psi.config import OrbitStackConfig, PsiDetectionConfig
from aoi_psi.manifests import SlcScene, StackManifest, scene_start_datetime

PIXEL_KEY_COLUMNS = ("azimuth_index", "range_index")
REQUIRED_RAW_POINT_FIELDS = {
    "point_id",
    "temporal_coherence",
    "azimuth_index",
    "range_index",
}


@dataclass(frozen=True)
class CdpsiSubsetPlan:
    stack_id: str
    role: Literal["front", "back"]
    subset_start_date: date
    subset_end_date: date
    master_date: date
    scene_dates: tuple[date, ...]
    manifest: StackManifest
    stack: OrbitStackConfig


@dataclass(frozen=True)
class CdpsiBreakPlan:
    break_after_date: date
    break_before_date: date
    front: CdpsiSubsetPlan
    back: CdpsiSubsetPlan


@dataclass(frozen=True)
class CdpsiPlan:
    stack_id: str
    scene_dates: tuple[date, ...]
    minimum_subset_images: int
    breaks: tuple[CdpsiBreakPlan, ...]

    @property
    def subset_runs(self) -> tuple[CdpsiSubsetPlan, ...]:
        runs: list[CdpsiSubsetPlan] = []
        for item in self.breaks:
            runs.extend((item.front, item.back))
        return tuple(runs)

    def describe(self) -> dict[str, object]:
        return {
            "method": "cdpsi",
            "scene_dates": [value.isoformat() for value in self.scene_dates],
            "scene_count": len(self.scene_dates),
            "minimum_subset_images": self.minimum_subset_images,
            "valid_break_count": len(self.breaks),
            "subset_run_count": len(self.subset_runs),
            "break_intervals": [
                {
                    "break_after_date": item.break_after_date.isoformat(),
                    "break_before_date": item.break_before_date.isoformat(),
                    "front_stack_id": item.front.stack_id,
                    "back_stack_id": item.back.stack_id,
                    "front_scene_count": len(item.front.scene_dates),
                    "back_scene_count": len(item.back.scene_dates),
                    "front_master_date": item.front.master_date.isoformat(),
                    "back_master_date": item.back.master_date.isoformat(),
                }
                for item in self.breaks
            ],
            "valid": bool(self.breaks),
        }

    def validate(self) -> None:
        if self.breaks:
            return
        raise ValueError(
            "CDPSI requires at least one break interval with non-degenerate front and back subsets. "
            f"Stack {self.stack_id} has {len(self.scene_dates)} scenes, while config.psi.minimum_subset_images="
            f"{self.minimum_subset_images}. Under the current single-master PSI workflow, a 2-image subset yields a "
            "single interferogram and degenerate temporal coherence, so both front and back subsets must contain at "
            "least 3 acquisitions. This 5-scene same-attempt stack therefore cannot support a published CDPSI run."
        )


@dataclass(frozen=True)
class CdpsiArtifacts:
    change_points: gpd.GeoDataFrame
    emergence_points: gpd.GeoDataFrame


def _scene_date(scene: SlcScene) -> date:
    return date.fromisoformat(scene.acquisition_date)


def _select_subset_master_scene(scenes: list[SlcScene]) -> SlcScene:
    subset_times = [scene_start_datetime(scene) for scene in scenes]
    midpoint = subset_times[0] + (subset_times[-1] - subset_times[0]) / 2
    return min(scenes, key=lambda scene: abs(scene_start_datetime(scene) - midpoint))


def _subset_stack_id(base_stack_id: str, role: Literal["front", "back"], start_date: date, end_date: date) -> str:
    return f"{base_stack_id}_cdpsi_{role}_{start_date:%Y%m%d}_{end_date:%Y%m%d}"


def _build_subset_plan(
    manifest: StackManifest,
    stack: OrbitStackConfig,
    role: Literal["front", "back"],
    scenes: list[SlcScene],
) -> CdpsiSubsetPlan:
    subset_dates = tuple(_scene_date(scene) for scene in scenes)
    master_scene = _select_subset_master_scene(scenes)
    subset_stack_id = _subset_stack_id(stack.id, role, subset_dates[0], subset_dates[-1])
    subset_manifest = StackManifest(
        stack_id=subset_stack_id,
        direction=manifest.direction,
        relative_orbit=manifest.relative_orbit,
        product_type=manifest.product_type,
        scenes=list(scenes),
    )
    subset_stack = OrbitStackConfig(
        id=subset_stack_id,
        direction=stack.direction,
        relative_orbit=stack.relative_orbit,
        iw_swaths=stack.iw_swaths,
        polarization=stack.polarization,
        master_date=_scene_date(master_scene),
        min_scenes=len(scenes),
        scene_limit=len(scenes),
    )
    return CdpsiSubsetPlan(
        stack_id=subset_stack_id,
        role=role,
        subset_start_date=subset_dates[0],
        subset_end_date=subset_dates[-1],
        master_date=_scene_date(master_scene),
        scene_dates=subset_dates,
        manifest=subset_manifest,
        stack=subset_stack,
    )


def plan_cdpsi_stack(manifest: StackManifest, stack: OrbitStackConfig, config: PsiDetectionConfig) -> CdpsiPlan:
    scenes = list(manifest.scenes)
    breaks: list[CdpsiBreakPlan] = []
    minimum_subset_images = config.minimum_subset_images
    for break_index in range(1, len(scenes)):
        front_scenes = scenes[:break_index]
        back_scenes = scenes[break_index:]
        if len(front_scenes) < minimum_subset_images or len(back_scenes) < minimum_subset_images:
            continue
        breaks.append(
            CdpsiBreakPlan(
                break_after_date=_scene_date(front_scenes[-1]),
                break_before_date=_scene_date(back_scenes[0]),
                front=_build_subset_plan(manifest, stack, "front", front_scenes),
                back=_build_subset_plan(manifest, stack, "back", back_scenes),
            )
        )
    return CdpsiPlan(
        stack_id=stack.id,
        scene_dates=tuple(_scene_date(scene) for scene in scenes),
        minimum_subset_images=minimum_subset_images,
        breaks=tuple(breaks),
    )


def validate_raw_points_contract(points: gpd.GeoDataFrame) -> None:
    missing = REQUIRED_RAW_POINT_FIELDS - set(points.columns)
    if missing:
        raise ValueError(
            "StaMPS raw PSI export is missing the fields required for CDPSI correspondence: "
            f"{sorted(missing)}. Export azimuth_index and range_index from ps.ij before running CDPSI."
        )


def _deduplicate_points(points: gpd.GeoDataFrame, *, label: str) -> gpd.GeoDataFrame:
    validate_raw_points_contract(points)
    frame = points.copy()
    for column in PIXEL_KEY_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)
    duplicates = frame.duplicated(subset=list(PIXEL_KEY_COLUMNS), keep=False)
    if duplicates.any():
        sample = frame.loc[duplicates, list(PIXEL_KEY_COLUMNS)].head(5).to_dict(orient="records")
        raise ValueError(f"{label} PSI points contain duplicate pixel keys, so CDPSI correspondence is ambiguous: {sample}")
    return frame


def _central_mode(values: np.ndarray) -> float:
    if values.size == 1:
        return float(values[0])
    bins = max(20, int(np.sqrt(values.size)))
    counts, edges = np.histogram(values, bins=bins)
    peak_index = int(np.argmax(counts))
    return float((edges[peak_index] + edges[peak_index + 1]) / 2.0)


def gaussian_3sigma_threshold(values: pd.Series, *, sigma_multiplier: float) -> dict[str, float]:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    numeric = numeric[np.isfinite(numeric)]
    if numeric.size < 2:
        raise ValueError("CDPSI threshold estimation requires at least two finite change-index values.")

    mode = _central_mode(numeric)
    stable_side = numeric[numeric <= mode]
    if stable_side.size == 0:
        stable_side = numeric
    mirrored = np.concatenate((stable_side, 2.0 * mode - stable_side[stable_side < mode]))
    mu_first = float(mirrored.mean())
    sigma_first = float(mirrored.std(ddof=0))
    if sigma_first == 0.0:
        sigma_second = 0.0
        mu_second = mu_first
    else:
        inliers = mirrored[(mirrored >= mu_first - 3.0 * sigma_first) & (mirrored <= mu_first + 3.0 * sigma_first)]
        if inliers.size == 0:
            inliers = mirrored
        mu_second = float(inliers.mean())
        sigma_second = float(inliers.std(ddof=0))
    threshold = mu_second + sigma_multiplier * sigma_second
    return {
        "mode": mode,
        "mu": mu_second,
        "sigma": sigma_second,
        "threshold": float(threshold),
        "sample_size": int(numeric.size),
    }


def _empty_like(points: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    empty = points.iloc[0:0].copy()
    empty["break_after_date"] = pd.Series(dtype="object")
    empty["break_before_date"] = pd.Series(dtype="object")
    empty["occurrence_interval_start"] = pd.Series(dtype="object")
    empty["occurrence_interval_end"] = pd.Series(dtype="object")
    empty["tc_complete"] = pd.Series(dtype="float64")
    empty["tc_front"] = pd.Series(dtype="float64")
    empty["tc_back"] = pd.Series(dtype="float64")
    empty["ci_vanishment"] = pd.Series(dtype="float64")
    empty["ci_emergence"] = pd.Series(dtype="float64")
    empty["ci_threshold_method"] = pd.Series(dtype="object")
    empty["ci_sigma_multiplier"] = pd.Series(dtype="float64")
    empty["ci_vanishment_threshold"] = pd.Series(dtype="float64")
    empty["ci_emergence_threshold"] = pd.Series(dtype="float64")
    empty["ci_vanishment_mu"] = pd.Series(dtype="float64")
    empty["ci_emergence_mu"] = pd.Series(dtype="float64")
    empty["ci_vanishment_sigma"] = pd.Series(dtype="float64")
    empty["ci_emergence_sigma"] = pd.Series(dtype="float64")
    empty["ci_vanishment_mode"] = pd.Series(dtype="float64")
    empty["ci_emergence_mode"] = pd.Series(dtype="float64")
    empty["front_point_id"] = pd.Series(dtype="float64")
    empty["back_point_id"] = pd.Series(dtype="float64")
    empty["cdpsi_class"] = pd.Series(dtype="object")
    return empty


def _build_break_change_points(
    complete_points: gpd.GeoDataFrame,
    front_points: gpd.GeoDataFrame,
    back_points: gpd.GeoDataFrame,
    break_plan: CdpsiBreakPlan,
    config: PsiDetectionConfig,
) -> gpd.GeoDataFrame:
    complete = _deduplicate_points(complete_points, label="complete").rename(
        columns={
            "point_id": "complete_point_id",
            "temporal_coherence": "tc_complete",
        }
    )
    front = _deduplicate_points(front_points, label=break_plan.front.stack_id)[list(PIXEL_KEY_COLUMNS) + ["point_id", "temporal_coherence"]].rename(
        columns={"point_id": "front_point_id", "temporal_coherence": "tc_front"}
    )
    back = _deduplicate_points(back_points, label=break_plan.back.stack_id)[list(PIXEL_KEY_COLUMNS) + ["point_id", "temporal_coherence"]].rename(
        columns={"point_id": "back_point_id", "temporal_coherence": "tc_back"}
    )

    matched = complete.merge(front, on=list(PIXEL_KEY_COLUMNS), how="inner").merge(back, on=list(PIXEL_KEY_COLUMNS), how="inner")
    if matched.empty:
        return _empty_like(complete_points)

    emergence_stats = gaussian_3sigma_threshold(matched["ci_emergence"] if "ci_emergence" in matched else matched["tc_back"] - matched["tc_complete"], sigma_multiplier=config.sigma_multiplier)
    vanishment_stats = gaussian_3sigma_threshold(matched["ci_vanishment"] if "ci_vanishment" in matched else matched["tc_front"] - matched["tc_complete"], sigma_multiplier=config.sigma_multiplier)

    matched["point_id"] = matched["complete_point_id"]
    matched["break_after_date"] = break_plan.break_after_date.isoformat()
    matched["break_before_date"] = break_plan.break_before_date.isoformat()
    matched["occurrence_interval_start"] = break_plan.break_after_date.isoformat()
    matched["occurrence_interval_end"] = break_plan.break_before_date.isoformat()
    matched["tc_complete"] = pd.to_numeric(matched["tc_complete"], errors="coerce")
    matched["tc_front"] = pd.to_numeric(matched["tc_front"], errors="coerce")
    matched["tc_back"] = pd.to_numeric(matched["tc_back"], errors="coerce")
    matched["ci_vanishment"] = matched["tc_front"] - matched["tc_complete"]
    matched["ci_emergence"] = matched["tc_back"] - matched["tc_complete"]
    matched["ci_threshold_method"] = config.thresholding_mode
    matched["ci_sigma_multiplier"] = float(config.sigma_multiplier)
    matched["ci_vanishment_threshold"] = vanishment_stats["threshold"]
    matched["ci_emergence_threshold"] = emergence_stats["threshold"]
    matched["ci_vanishment_mu"] = vanishment_stats["mu"]
    matched["ci_emergence_mu"] = emergence_stats["mu"]
    matched["ci_vanishment_sigma"] = vanishment_stats["sigma"]
    matched["ci_emergence_sigma"] = emergence_stats["sigma"]
    matched["ci_vanishment_mode"] = vanishment_stats["mode"]
    matched["ci_emergence_mode"] = emergence_stats["mode"]

    emergence_candidate = matched["ci_emergence"] >= matched["ci_emergence_threshold"]
    vanishment_candidate = matched["ci_vanishment"] >= matched["ci_vanishment_threshold"]
    matched["cdpsi_class"] = np.select(
        [
            emergence_candidate & ~vanishment_candidate,
            vanishment_candidate & ~emergence_candidate,
            emergence_candidate & vanishment_candidate,
        ],
        [
            "emergence_candidate",
            "vanishment_candidate",
            "ambiguous_change",
        ],
        default="stable",
    )
    return gpd.GeoDataFrame(matched, geometry="geometry", crs=complete_points.crs)


def build_cdpsi_artifacts(
    complete_points: gpd.GeoDataFrame,
    plan: CdpsiPlan,
    subset_points_by_stack_id: dict[str, gpd.GeoDataFrame],
    config: PsiDetectionConfig,
) -> CdpsiArtifacts:
    plan.validate()
    break_frames: list[gpd.GeoDataFrame] = []
    for break_plan in plan.breaks:
        front_points = subset_points_by_stack_id.get(break_plan.front.stack_id)
        back_points = subset_points_by_stack_id.get(break_plan.back.stack_id)
        if front_points is None or back_points is None:
            raise ValueError(
                "CDPSI subset outputs are missing for break interval "
                f"{break_plan.break_after_date.isoformat()} -> {break_plan.break_before_date.isoformat()}."
            )
        break_frame = _build_break_change_points(complete_points, front_points, back_points, break_plan, config)
        if not break_frame.empty:
            break_frames.append(break_frame)
    if not break_frames:
        raise ValueError("CDPSI produced no matched complete/front/back PS intersections for any valid break interval.")

    change_points = gpd.GeoDataFrame(pd.concat(break_frames, ignore_index=True), geometry="geometry", crs=complete_points.crs)
    emergence_points = change_points.loc[change_points["cdpsi_class"] == "emergence_candidate"].copy()
    if emergence_points.empty:
        return CdpsiArtifacts(change_points=change_points, emergence_points=emergence_points)

    emergence_points = emergence_points.sort_values(
        by=["ci_emergence", "break_before_date", "break_after_date"],
        ascending=[False, True, True],
    )
    emergence_points = emergence_points.drop_duplicates(subset=list(PIXEL_KEY_COLUMNS), keep="first").copy()
    return CdpsiArtifacts(change_points=change_points, emergence_points=emergence_points)
