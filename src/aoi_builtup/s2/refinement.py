from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from aoi_builtup.config import Sentinel2Config


@dataclass(frozen=True)
class S2SupportArtifacts:
    score: xr.DataArray
    support: xr.DataArray
    reliable: xr.DataArray


def build_unavailable_s2_support_like(template: xr.DataArray) -> S2SupportArtifacts:
    score = xr.zeros_like(template, dtype=np.uint8).rename("s2_score")
    support = xr.zeros_like(template, dtype=bool).rename("s2_support")
    reliable = xr.zeros_like(template, dtype=bool).rename("s2_reliable")
    return S2SupportArtifacts(score=score, support=support, reliable=reliable)


def build_s2_support(
    before: xr.Dataset,
    after: xr.Dataset,
    config: Sentinel2Config,
) -> S2SupportArtifacts:
    thresholds = config.thresholds
    delta_ndbi = (after["ndbi"] - before["ndbi"]).rename("delta_ndbi")

    score_layers = [
        (after["ndvi"] < thresholds.ndvi_after_max),
        (after["mndwi"] < thresholds.mndwi_after_max),
        (after["ndbi"] > thresholds.ndbi_after_min),
        (delta_ndbi > thresholds.delta_ndbi_min),
    ]

    if config.use_bsi and "bsi" in after and "bsi" in before:
        delta_bsi = after["bsi"] - before["bsi"]
        bsi_support = (after["bsi"] > thresholds.bsi_after_min) | (delta_bsi > thresholds.delta_bsi_min)
        score_layers.append(bsi_support)

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
        & before["ndvi"].notnull()
        & after["ndvi"].notnull()
        & before["ndbi"].notnull()
        & after["ndbi"].notnull()
        & before["mndwi"].notnull()
        & after["mndwi"].notnull()
    ).rename("s2_reliable")
    support = (score >= config.support_score_min).rename("s2_support")
    return S2SupportArtifacts(score=score, support=support, reliable=reliable)
