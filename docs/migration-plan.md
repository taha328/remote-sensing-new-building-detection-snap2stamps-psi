# Migration Plan

## Phase 1: Freeze the notebook baseline

- Export notebook period-level outputs:
  - S1 VV ratio
  - S1 VH ratio
  - S1 p-value
  - raw S1 candidate mask
  - refined S1+S2 mask
  - cumulative raster
  - final zone vectors
- Record the exact AOI, date windows, and thresholds from the notebook.
- Treat those outputs as non-regression references, not as ground truth.

## Phase 2: Reproduce S1 locally

- Replace GEE acquisition with `sentinel-1-rtc` from Planetary Computer.
- Match the notebook:
  - summer windows
  - Sentinel-1A pinning
  - VV/VH bands
  - LRT `alpha=0.01`
  - `VV ratio > 1.5`
  - `VH ratio > 1.05`
- Validate local S1 rasters against notebook patterns and total candidate area.

## Phase 3: Add S2 soft refinement

- Replace notebook S2 logic with local `sentinel-2-l2a` loading.
- Use SCL-based masking plus clear-observation counts.
- Keep S1 where S2 is unavailable.
- Require moderate S2 support only where S2 is reliable.
- Add strong-S1 override to preserve recall.

## Phase 4: Add cumulative zoning and vector outputs

- Build cumulative first-change raster.
- Compute local density and morphology locally.
- Polygonize with rasterio, then compute vector metrics with GeoPandas/Shapely.
- Export GeoParquet and GeoPackage instead of depending on fragile GeoJSON-only flows.

## Phase 5: Production hardening

- Add CLI stage commands.
- Persist manifests and run reports.
- Add unit tests and synthetic raster regression tests.
- Add optional reference-label evaluation if labelled polygons become available.
