from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from aoi_builtup.config import Sentinel2Config
from aoi_builtup.s1.detection import S1DetectionArtifacts
from aoi_builtup.s2.refinement import S2SupportArtifacts


@dataclass(frozen=True)
class FusionArtifacts:
    refined: xr.DataArray
    decision: xr.DataArray


def apply_soft_refinement(
    s1: S1DetectionArtifacts,
    s2: S2SupportArtifacts,
    config: Sentinel2Config,
) -> FusionArtifacts:
    override = xr.zeros_like(s1.candidate, dtype=bool)
    if config.strong_override.enabled:
        override = (
            (s1.ratio_vv >= config.strong_override.vv_ratio_min)
            & (s1.ratio_vh >= config.strong_override.vh_ratio_min)
            & (s1.pvalue_vv <= config.strong_override.pvalue_max)
        )

    refined = (s1.candidate & (~s2.reliable | s2.support | override)).rename("construction")

    decision = xr.full_like(s1.candidate.astype(np.uint8), fill_value=0).rename("fusion_decision")
    decision = xr.where(s1.candidate & ~s2.reliable, 1, decision)
    decision = xr.where(s1.candidate & s2.reliable & s2.support, 2, decision)
    decision = xr.where(s1.candidate & s2.reliable & ~s2.support & override, 3, decision)
    decision = xr.where(s1.candidate & s2.reliable & ~s2.support & ~override, 4, decision)

    return FusionArtifacts(refined=refined.astype(bool), decision=decision.astype("uint8"))
