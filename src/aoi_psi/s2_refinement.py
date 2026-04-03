from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from aoi_psi.config import Sentinel2RefinementConfig


@dataclass(frozen=True)
class OpticalSupportArtifacts:
    score: xr.DataArray
    support: xr.DataArray
    reliable: xr.DataArray


def build_optical_support(before: xr.Dataset, after: xr.Dataset, config: Sentinel2RefinementConfig) -> OpticalSupportArtifacts:
    thresholds = config.thresholds
    delta_ndbi = (after["ndbi"] - before["ndbi"]).rename("delta_ndbi")
    score_layers = [
        after["ndvi"] < thresholds.ndvi_after_max,
        after["mndwi"] < thresholds.mndwi_after_max,
        after["ndbi"] > thresholds.ndbi_after_min,
        delta_ndbi > thresholds.delta_ndbi_min,
    ]
    if config.use_bsi and "bsi" in after and "bsi" in before:
        delta_bsi = after["bsi"] - before["bsi"]
        score_layers.append((after["bsi"] > thresholds.bsi_after_min) | (delta_bsi > thresholds.delta_bsi_min))

    score = xr.zeros_like(after["ndvi"], dtype=np.uint8).rename("s2_score")
    for layer in score_layers:
        score = (score + layer.fillna(False).astype(np.uint8)).astype(np.uint8)

    reliable = (
        (before["clear_count"] >= config.min_clear_observations)
        & (after["clear_count"] >= config.min_clear_observations)
        & (before["clear_fraction"] >= config.min_clear_fraction)
        & (after["clear_fraction"] >= config.min_clear_fraction)
        & (before["valid_fraction"] >= config.min_clear_fraction)
        & (after["valid_fraction"] >= config.min_clear_fraction)
    ).rename("s2_reliable")
    support = (score >= config.support_score_min).rename("s2_support")
    return OpticalSupportArtifacts(score=score, support=support, reliable=reliable)
