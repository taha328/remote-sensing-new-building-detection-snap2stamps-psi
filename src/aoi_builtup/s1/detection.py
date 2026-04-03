from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.ndimage as ndi
import xarray as xr
from scipy.stats import chi2
from skimage.morphology import disk

from aoi_builtup.config import DetectionConfig


@dataclass(frozen=True)
class S1DetectionArtifacts:
    pvalue_vv: xr.DataArray
    ratio_vv: xr.DataArray
    ratio_vh: xr.DataArray
    candidate: xr.DataArray


def _radius_to_pixels(radius_m: float, resolution_m: float) -> int:
    return max(0, int(round(radius_m / resolution_m)))


def _remove_small_components(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    if min_pixels <= 1:
        return mask
    structure = ndi.generate_binary_structure(2, 2)
    labels, num_labels = ndi.label(mask, structure=structure)
    if num_labels == 0:
        return mask
    counts = np.bincount(labels.ravel())
    keep = counts >= min_pixels
    keep[0] = False
    return keep[labels]


def _binary_closing(mask: np.ndarray, radius_pixels: int) -> np.ndarray:
    if radius_pixels <= 0:
        return mask
    footprint = disk(radius_pixels)
    dilated = ndi.binary_dilation(mask, structure=footprint)
    return ndi.binary_erosion(dilated, structure=footprint)


def compute_lrt_pvalue(before: xr.DataArray, after: xr.DataArray, enl: float) -> xr.DataArray:
    total = (before + after).clip(min=1e-6)
    statistic = -2.0 * np.log(
        np.clip(
            np.power((before / total), enl)
            * np.power((after / total), enl)
            * np.power(2.0, 2.0 * enl),
            1e-12,
            None,
        )
    )
    return xr.apply_ufunc(
        lambda values: chi2.sf(values, df=1),
        statistic,
        dask="parallelized",
        output_dtypes=[np.float32],
    ).rename("pvalue_vv")


def detect_s1_change(
    before: xr.Dataset,
    after: xr.Dataset,
    config: DetectionConfig,
    resolution_m: float,
) -> S1DetectionArtifacts:
    pvalue_vv = compute_lrt_pvalue(before["vv"], after["vv"], enl=config.enl)
    ratio_vv = (after["vv"] / before["vv"].clip(min=1e-6)).rename("ratio_vv")
    ratio_vh = (after["vh"] / before["vh"].clip(min=1e-6)).rename("ratio_vh")

    candidate = (
        (pvalue_vv < config.alpha)
        & (ratio_vv > config.vv_ratio_min)
        & (ratio_vh > config.vh_ratio_min)
    ).fillna(False)

    candidate_np = candidate.compute().values.astype(bool)
    candidate_np = _remove_small_components(candidate_np, min_pixels=config.min_connected_pixels)
    candidate_np = _binary_closing(
        candidate_np,
        radius_pixels=_radius_to_pixels(config.closing_radius_m, resolution_m),
    )

    candidate_clean = xr.DataArray(
        candidate_np,
        coords=candidate.coords,
        dims=candidate.dims,
        name="construction",
    )
    return S1DetectionArtifacts(
        pvalue_vv=pvalue_vv.astype("float32"),
        ratio_vv=ratio_vv.astype("float32"),
        ratio_vh=ratio_vh.astype("float32"),
        candidate=candidate_clean,
    )
