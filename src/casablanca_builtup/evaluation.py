from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio.enums
import xarray as xr

from casablanca_builtup.config import ExportConfig
from casablanca_builtup.io import read_raster, write_json, write_raster


def to_binary_mask(data: xr.DataArray) -> xr.DataArray:
    return data.fillna(0).astype(np.float32) > 0


def align_to_reference(local: xr.DataArray, reference: xr.DataArray) -> xr.DataArray:
    if local.shape == reference.shape and str(local.rio.crs) == str(reference.rio.crs):
        return local
    return local.rio.reproject_match(reference, resampling=rasterio.enums.Resampling.nearest)


def confusion_counts(predicted: xr.DataArray, truth: xr.DataArray) -> dict[str, int]:
    predicted_mask = predicted.astype(bool)
    truth_mask = truth.astype(bool)
    tp = int((predicted_mask & truth_mask).astype("uint8").sum().compute().item())
    fp = int((predicted_mask & ~truth_mask).astype("uint8").sum().compute().item())
    fn = int((~predicted_mask & truth_mask).astype("uint8").sum().compute().item())
    tn = int((~predicted_mask & ~truth_mask).astype("uint8").sum().compute().item())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _metrics_from_counts(counts: dict[str, int], pixel_area_ha: float) -> dict[str, Any]:
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    tn = counts["tn"]
    predicted = tp + fp
    truth = tp + fn
    union = tp + fp + fn
    precision = None if predicted == 0 else tp / predicted
    recall = None if truth == 0 else tp / truth
    iou = None if union == 0 else tp / union
    f1 = None if precision is None or recall is None or (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)
    return {
        **counts,
        "predicted_area_ha": predicted * pixel_area_ha,
        "truth_area_ha": truth * pixel_area_ha,
        "area_delta_ha": (predicted - truth) * pixel_area_ha,
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "f1": f1,
        "support_pixels": tp + fp + fn + tn,
    }


def _confusion_raster(predicted: xr.DataArray, truth: xr.DataArray) -> xr.DataArray:
    confusion = xr.zeros_like(predicted.astype("uint8"), dtype=np.uint8).rename("confusion")
    confusion = xr.where(predicted & truth, 1, confusion)
    confusion = xr.where(predicted & ~truth, 2, confusion)
    confusion = xr.where(~predicted & truth, 3, confusion)
    return confusion


def evaluate_binary_rasters(
    local_raster: Path,
    reference_raster: Path,
    output_dir: Path | None = None,
    prefix: str = "reference",
) -> dict[str, Any]:
    local = to_binary_mask(read_raster(local_raster))
    reference = to_binary_mask(read_raster(reference_raster))
    local = align_to_reference(local, reference)

    pixel_area_ha = abs(local.rio.resolution()[0] * local.rio.resolution()[1]) / 10000.0
    counts = confusion_counts(local, reference)
    metrics = _metrics_from_counts(counts, pixel_area_ha)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(metrics, output_dir / f"{prefix}_metrics.json")
        write_raster(
            _confusion_raster(local, reference),
            output_dir / f"{prefix}_confusion.tif",
            str(reference.rio.crs),
            ExportConfig(),
        )
    return metrics


def evaluate_run(
    run_dir: Path,
    reference_raster: Path,
    labels_raster: Path | None = None,
    local_raster_name: str = "cumulative_refined.tif",
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    local_raster = run_dir / "rasters" / local_raster_name
    if not local_raster.exists():
        raise FileNotFoundError(f"Local raster not found: {local_raster}")

    evaluation_dir = run_dir / "reports" / "evaluation"
    result = {
        "local_raster": str(local_raster),
        "reference_raster": str(reference_raster.resolve()),
        "against_reference": evaluate_binary_rasters(
            local_raster,
            reference_raster.resolve(),
            output_dir=evaluation_dir,
            prefix="reference",
        ),
    }
    if labels_raster is not None:
        result["labels_raster"] = str(labels_raster.resolve())
        result["against_labels"] = evaluate_binary_rasters(
            local_raster,
            labels_raster.resolve(),
            output_dir=evaluation_dir,
            prefix="labels",
        )

    write_json(result, evaluation_dir / "evaluation_summary.json")
    return result
