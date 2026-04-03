from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import pandas as pd


@dataclass(frozen=True)
class PsiArtifacts:
    points: gpd.GeoDataFrame


def load_ps_points(points_csv: Path, *, target_crs: str = "EPSG:32629") -> gpd.GeoDataFrame:
    frame = pd.read_csv(points_csv)
    if "point_id" not in frame.columns:
        raise ValueError("PSI export is missing point_id.")

    for column in ("azimuth_index", "range_index"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

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


def load_points_only(points_csv: Path, *, target_crs: str = "EPSG:32629") -> PsiArtifacts:
    return PsiArtifacts(points=load_ps_points(points_csv, target_crs=target_crs))
