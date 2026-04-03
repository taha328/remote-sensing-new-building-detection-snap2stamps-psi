from __future__ import annotations

import geopandas as gpd
from shapely.geometry import Point, Polygon

from aoi_psi.evaluation import evaluate_points_against_reference, evaluate_polygons_against_reference


def test_evaluation_metrics() -> None:
    points = gpd.GeoDataFrame({"point_id": [1, 2]}, geometry=[Point(0, 0), Point(10, 10)], crs="EPSG:32629")
    reference_points = gpd.GeoDataFrame({"ref_id": [1]}, geometry=[Point(0, 0).buffer(1)], crs="EPSG:32629")
    point_metrics = evaluate_points_against_reference(points, reference_points)

    polygons = gpd.GeoDataFrame({"polygon_id": [1]}, geometry=[Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])], crs="EPSG:32629")
    reference_polygons = gpd.GeoDataFrame({"ref_id": [1]}, geometry=[Polygon([(2, 2), (7, 2), (7, 7), (2, 7)])], crs="EPSG:32629")
    polygon_metrics = evaluate_polygons_against_reference(polygons, reference_polygons)

    assert point_metrics["tp"] == 1
    assert point_metrics["fp"] == 1
    assert point_metrics["fn"] == 0
    assert polygon_metrics["intersection_area_m2"] == 9.0
    assert polygon_metrics["iou"] is not None
