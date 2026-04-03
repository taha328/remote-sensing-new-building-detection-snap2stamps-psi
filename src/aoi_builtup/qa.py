from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import xarray as xr


FUSION_DECISION_LABELS = {
    0: "not_candidate",
    1: "kept_s1_s2_unreliable",
    2: "kept_s1_s2_supported",
    3: "kept_strong_s1_override",
    4: "dropped_s2_reliable_unsupported",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _stage_entry() -> dict[str, Any]:
    return {
        "status": "pending",
        "started_at_utc": None,
        "completed_at_utc": None,
        "duration_s": None,
    }


def binary_pixel_count(mask: xr.DataArray) -> int:
    return int(mask.fillna(False).astype("uint8").sum().compute().item())


def binary_area_ha(mask: xr.DataArray, resolution_m: float) -> float:
    return float(binary_pixel_count(mask) * (resolution_m**2) / 10000.0)


def mask_fraction(mask: xr.DataArray) -> float:
    return float(mask.fillna(False).astype("uint8").mean().compute().item())


def mean_over_mask(values: xr.DataArray, mask: xr.DataArray) -> float | None:
    masked = values.where(mask.astype(bool))
    if int(mask.astype("uint8").sum().compute().item()) == 0:
        return None
    result = masked.mean(skipna=True).compute().item()
    if result is None or (isinstance(result, float) and np.isnan(result)):
        return None
    return float(result)


def decision_histogram(decision: xr.DataArray) -> dict[str, int]:
    result: dict[str, int] = {}
    for code, label in FUSION_DECISION_LABELS.items():
        count = int((decision == code).astype("uint8").sum().compute().item())
        result[label] = count
    return result


def base_run_report(run_id: str, config_path: str, grid: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "config_path": config_path,
        "run_group_id": None,
        "attempt_id": None,
        "run_root": None,
        "started_at_utc": utc_now_iso(),
        "completed_at_utc": None,
        "status": "running",
        "grid": grid,
        "stages": {
            "acquire": _stage_entry(),
            "build_composites": _stage_entry(),
            "detect_s1": _stage_entry(),
            "refine_s2": _stage_entry(),
            "polygonize": _stage_entry(),
            "export": _stage_entry(),
        },
        "periods": [],
    }


def mark_stage(report: dict[str, Any], stage: str, status: str) -> None:
    entry = report["stages"][stage]
    now = _utc_now()
    entry["status"] = status
    if status == "running":
        entry["started_at_utc"] = now.isoformat()
        entry["completed_at_utc"] = None
        entry["duration_s"] = None
        return
    if entry["started_at_utc"] is None:
        entry["started_at_utc"] = now.isoformat()
    entry["completed_at_utc"] = now.isoformat()
    started_at = datetime.fromisoformat(entry["started_at_utc"])
    entry["duration_s"] = max(0.0, (now - started_at).total_seconds())


def stage_status(report: dict[str, Any], stage: str) -> str:
    return str(report["stages"][stage]["status"])


def finalize_run_report(report: dict[str, Any], status: str) -> dict[str, Any]:
    report["status"] = status
    report["completed_at_utc"] = utc_now_iso()
    return report


def mark_running_stages(report: dict[str, Any], status: str) -> None:
    for stage_name, stage in report.get("stages", {}).items():
        if isinstance(stage, dict) and stage.get("status") == "running":
            mark_stage(report, stage_name, status)
