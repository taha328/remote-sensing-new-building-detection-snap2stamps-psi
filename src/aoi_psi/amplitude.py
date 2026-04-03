from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from aoi_psi.config import AmplitudeBranchConfig


@dataclass(frozen=True)
class AmplitudeArtifacts:
    ratio: xr.DataArray
    log_ratio: xr.DataArray
    candidate: xr.DataArray


def amplitude_change(before: xr.DataArray, after: xr.DataArray, config: AmplitudeBranchConfig) -> AmplitudeArtifacts:
    ratio = (after / before.clip(min=1e-6)).astype("float32").rename("amplitude_ratio")
    log_ratio = xr.apply_ufunc(np.log1p, ratio).astype("float32").rename("amplitude_log_ratio")
    candidate = ((ratio >= config.ratio_min) & (log_ratio >= config.log_ratio_min)).rename("amplitude_candidate")
    return AmplitudeArtifacts(ratio=ratio, log_ratio=log_ratio, candidate=candidate)
