from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import pandas as pd

from casablanca_psi.config import PsiDetectionConfig

EMERGENCE_FIELDS = {
    "temporal_coherence",
    "pre_stability_fraction",
    "post_stability_fraction",
}


@dataclass(frozen=True)
class PsiArtifacts:
    points: gpd.GeoDataFrame
    emergent_points: gpd.GeoDataFrame


def load_ps_points(points_csv: Path, *, target_crs: str = "EPSG:32629") -> gpd.GeoDataFrame:
    frame = pd.read_csv(points_csv)
    if "point_id" not in frame.columns:
        raise ValueError("PSI export is missing point_id.")

    if {"lon", "lat"}.issubset(frame.columns):
        geometry = gpd.points_from_xy(frame["lon"], frame["lat"], crs="EPSG:4326")
        points = gpd.GeoDataFrame(frame, geometry=geometry)
        return points.to_crs(target_crs)

    if {"x", "y"}.issubset(frame.columns):
        geometry = gpd.points_from_xy(frame["x"], frame["y"])
        return gpd.GeoDataFrame(frame, geometry=geometry, crs=target_crs)

    if {"x_local_m", "y_local_m"}.issubset(frame.columns):
        raise ValueError(
            "PSI export only contains local StaMPS XY coordinates. Export lon/lat from StaMPS before parsing in Python."
        )

    raise ValueError("PSI export must contain lon/lat or projected x/y coordinates.")


def detect_emergent_ps(points: gpd.GeoDataFrame, config: PsiDetectionConfig) -> gpd.GeoDataFrame:
    missing = EMERGENCE_FIELDS - set(points.columns)
    if missing:
        raise ValueError(
            "PSI export does not yet contain the fields required for emergence detection: "
            f"{sorted(missing)}. Produce real PSI points first, then derive emergence timing/stability fields."
        )

    frame = points.copy()
    frame["stability_gain"] = frame["post_stability_fraction"] - frame["pre_stability_fraction"]
    if "residual_height_m" in frame.columns:
        frame["height_support"] = frame["residual_height_m"].fillna(0.0) >= config.min_residual_height_support_m
    else:
        frame["height_support"] = False
    frame["psi_emergent"] = (
        (frame["temporal_coherence"] >= config.min_temporal_coherence)
        & (frame["post_stability_fraction"] >= config.min_post_stability_fraction)
        & (frame["pre_stability_fraction"] <= config.max_pre_stability_fraction)
        & (frame["stability_gain"] >= config.min_stability_gain)
    )
    return frame.loc[frame["psi_emergent"]].copy()


def load_and_detect(points_csv: Path, config: PsiDetectionConfig, *, target_crs: str = "EPSG:32629") -> PsiArtifacts:
    points = load_ps_points(points_csv, target_crs=target_crs)
    emergent = detect_emergent_ps(points, config)
    return PsiArtifacts(points=points, emergent_points=emergent)
