from __future__ import annotations

from pathlib import Path

import geopandas as gpd
from shapely.geometry import box

from casablanca_builtup.io import write_json, write_vector
from casablanca_psi.config import ExportConfig
from casablanca_psi.fusion import FusionOutputs


def export_fusion_outputs(outputs: FusionOutputs, root: Path, export_config: ExportConfig) -> None:
    root.mkdir(parents=True, exist_ok=True)
    if export_config.save_intermediate_points:
        for extension in export_config.vector_formats:
            write_vector(outputs.points, root / f"cdpsi_emergence_candidates.{extension}")
    for extension in export_config.vector_formats:
        write_vector(outputs.polygons, root / f"fused_change_candidates.{extension}")


def write_run_summary(summary: dict, path: Path) -> None:
    write_json(summary, path)


def parcel_summary(points: gpd.GeoDataFrame, parcels: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    joined = gpd.sjoin(points, parcels, how="inner", predicate="intersects")
    if joined.empty:
        return gpd.GeoDataFrame(columns=["geometry", "candidate_count"], geometry="geometry", crs=parcels.crs)
    id_column = "parcel_id" if "parcel_id" in parcels.columns else parcels.index.name or "index_right"
    stats = joined.groupby(id_column).size().rename("candidate_count")
    result = parcels.copy()
    result[id_column] = result[id_column] if id_column in result.columns else result.index
    return result.merge(stats, left_on=id_column, right_index=True, how="left").fillna({"candidate_count": 0})


def bbox_polygon(bounds: tuple[float, float, float, float], crs: str) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"geometry": [box(*bounds)]}, crs=crs)
