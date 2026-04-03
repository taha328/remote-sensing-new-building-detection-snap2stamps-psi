from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd

from aoi_builtup.io import write_json


def evaluate_points_against_reference(points: gpd.GeoDataFrame, reference: gpd.GeoDataFrame) -> dict[str, Any]:
    joined = gpd.sjoin(points, reference[["geometry"]], how="left", predicate="intersects")
    tp = int(joined.index_right.notnull().sum())
    fp = int(joined.index_right.isnull().sum())
    fn = max(int(len(reference) - tp), 0)
    precision = None if (tp + fp) == 0 else tp / (tp + fp)
    recall = None if (tp + fn) == 0 else tp / (tp + fn)
    f1 = None if precision is None or recall is None or (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def evaluate_polygons_against_reference(polygons: gpd.GeoDataFrame, reference: gpd.GeoDataFrame) -> dict[str, Any]:
    intersections = gpd.overlay(polygons, reference, how="intersection")
    intersection_area = float(intersections.area.sum()) if not intersections.empty else 0.0
    predicted_area = float(polygons.area.sum()) if not polygons.empty else 0.0
    truth_area = float(reference.area.sum()) if not reference.empty else 0.0
    union_area = predicted_area + truth_area - intersection_area
    iou = None if union_area == 0 else intersection_area / union_area
    return {
        "predicted_area_m2": predicted_area,
        "reference_area_m2": truth_area,
        "intersection_area_m2": intersection_area,
        "area_delta_m2": predicted_area - truth_area,
        "iou": iou,
    }


def write_evaluation(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(result, output_dir / "evaluation_summary.json")
