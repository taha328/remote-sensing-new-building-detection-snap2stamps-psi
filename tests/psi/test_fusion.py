from __future__ import annotations

import geopandas as gpd
from shapely.geometry import Point, Polygon

from casablanca_psi.config import FusionConfig
from casablanca_psi.fusion import fuse_evidence


def test_fuse_evidence_scores_and_clusters() -> None:
    points = gpd.GeoDataFrame(
        {
            "point_id": [1, 2],
            "temporal_coherence": [0.9, 0.88],
            "pre_stability_fraction": [0.0, 0.1],
            "post_stability_fraction": [0.9, 0.8],
        },
        geometry=[Point(0, 0), Point(5, 0)],
        crs="EPSG:32629",
    )
    amplitude = gpd.GeoDataFrame({"geometry": [Polygon([(-1, -1), (8, -1), (8, 2), (-1, 2)])]}, crs=points.crs)
    optical = gpd.GeoDataFrame(
        {"optical_support": [True], "optical_reliable": [True], "geometry": [Polygon([(-1, -1), (8, -1), (8, 2), (-1, 2)])]},
        crs=points.crs,
    )

    outputs = fuse_evidence(points, FusionConfig(), amplitude_support=amplitude, optical_support=optical, cluster_buffer_m=6.0)

    assert len(outputs.points) == 2
    assert outputs.points["confidence_class"].isin({"high", "medium", "low"}).all()
    assert len(outputs.polygons) >= 1
