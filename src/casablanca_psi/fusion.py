from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.ops import unary_union

from casablanca_psi.config import FusionConfig


@dataclass(frozen=True)
class FusionOutputs:
    points: gpd.GeoDataFrame
    polygons: gpd.GeoDataFrame


def _joined_bool(joined: gpd.GeoDataFrame, column: str) -> np.ndarray:
    if column not in joined.columns:
        return np.zeros(len(joined), dtype=bool)
    series = joined[column]
    if not isinstance(series, pd.Series):
        return np.zeros(len(joined), dtype=bool)
    return series.fillna(False).astype(bool).to_numpy()


def _assign_confidence(frame: gpd.GeoDataFrame, config: FusionConfig) -> gpd.GeoDataFrame:
    result = frame.copy()
    result["confidence_score"] = (
        config.psi_primary_weight * result["psi_primary"].astype(float)
        + config.amplitude_weight * result["amplitude_support"].astype(float)
        + config.optical_weight * result["optical_support"].astype(float)
        + config.context_weight * result["context_support"].astype(float)
    )
    if config.allow_s2_non_support_penalty:
        result["confidence_score"] = result["confidence_score"] - (
            result["optical_reliable"] & ~result["optical_support"]
        ).astype(float) * config.s2_non_support_penalty
    result["confidence_score"] = result["confidence_score"].clip(lower=0.0, upper=1.0)
    result["confidence_class"] = np.select(
        [
            result["confidence_score"] >= config.high_confidence_threshold,
            result["confidence_score"] >= config.medium_confidence_threshold,
        ],
        ["high", "medium"],
        default="low",
    )
    return result


def fuse_evidence(
    psi_points: gpd.GeoDataFrame,
    config: FusionConfig,
    amplitude_support: gpd.GeoDataFrame | None = None,
    optical_support: gpd.GeoDataFrame | None = None,
    context_support: gpd.GeoDataFrame | None = None,
    cluster_buffer_m: float = 12.0,
) -> FusionOutputs:
    points = psi_points.copy()
    points["psi_primary"] = True
    points["amplitude_support"] = False
    points["optical_support"] = False
    points["optical_reliable"] = False
    points["context_support"] = False

    if amplitude_support is not None and not amplitude_support.empty:
        joined = gpd.sjoin(points, amplitude_support[["geometry"]], how="left", predicate="intersects")
        points["amplitude_support"] = joined.index_right.notnull().to_numpy()

    if optical_support is not None and not optical_support.empty:
        support_cols = [column for column in ("optical_support", "optical_reliable", "geometry") if column in optical_support]
        joined = gpd.sjoin(points, optical_support[support_cols], how="left", predicate="intersects")
        points["optical_support"] = _joined_bool(joined, "optical_support")
        points["optical_reliable"] = _joined_bool(joined, "optical_reliable")

    if context_support is not None and not context_support.empty:
        joined = gpd.sjoin(points, context_support[["geometry"]], how="left", predicate="intersects")
        points["context_support"] = joined.index_right.notnull().to_numpy()

    points = _assign_confidence(points, config)
    buffers = points.buffer(cluster_buffer_m)
    merged = gpd.GeoSeries([unary_union(buffers.to_list())], crs=points.crs).explode(index_parts=False).reset_index(drop=True)
    polygons = gpd.GeoDataFrame({"geometry": merged}, crs=points.crs)
    polygons = polygons[~polygons.geometry.is_empty].copy()
    if polygons.empty:
        polygons["confidence_class"] = []
        polygons["point_count"] = []
        polygons["mean_confidence"] = []
        return FusionOutputs(points=points, polygons=polygons)

    polygons["polygon_id"] = range(1, len(polygons) + 1)
    joined = gpd.sjoin(points, polygons, how="left", predicate="intersects")
    stats = (
        joined.groupby("polygon_id")
        .agg(point_count=("point_id", "count"), mean_confidence=("confidence_score", "mean"))
        .reset_index()
    )
    polygons = polygons.merge(stats, on="polygon_id", how="left")
    polygons["confidence_class"] = np.select(
        [
            polygons["mean_confidence"] >= config.high_confidence_threshold,
            polygons["mean_confidence"] >= config.medium_confidence_threshold,
        ],
        ["high", "medium"],
        default="low",
    )
    return FusionOutputs(points=points, polygons=polygons)
