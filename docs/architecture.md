# Built-Up Architecture

## Scope

The `aoi-builtup` workflow is a raster-first built-up change detector. It identifies probable new built-up areas, not guaranteed building footprints.

## Stage Order

1. Load and validate the YAML configuration.
2. Resolve the AOI geometry and derive the analysis grid.
3. Build deterministic Sentinel-1 and Sentinel-2 STAC manifests.
4. Create before/after Sentinel-1 composites on the analysis grid.
5. Run Sentinel-1 change detection.
6. Create Sentinel-2 support composites when optical data is available.
7. Apply soft S1/S2 fusion.
8. Build cumulative change rasters and density zones.
9. Polygonize the final refined mask.
10. Export rasters, vectors, manifests, and run reports.

## Main Data Flow

`config`
-> `RunContext`
-> `AOI geometry`
-> `resolved metric grid`
-> `STAC manifests`
-> `S1 composites`
-> `S1 candidate masks`
-> `S2 support layers`
-> `refined masks`
-> `cumulative raster`
-> `zone mask`
-> `polygons`
-> `reports and exports`

## Module Responsibilities

- `src/aoi_builtup/config.py`: schema validation and config loading.
- `src/aoi_builtup/grid.py`: AOI loading, automatic metric CRS selection, and grid alignment.
- `src/aoi_builtup/acquisition/stac.py`: STAC search, manifest freezing, and signed item reconstruction.
- `src/aoi_builtup/s1/composite.py`: Sentinel-1 loading and temporal compositing.
- `src/aoi_builtup/s1/detection.py`: likelihood-ratio, ratio thresholds, morphology, and connected-component cleanup.
- `src/aoi_builtup/s2/composite.py`: Sentinel-2 cloud masking, indices, and clear-observation accounting.
- `src/aoi_builtup/s2/refinement.py`: optical support and reliability masks.
- `src/aoi_builtup/fusion.py`: soft refinement logic combining S1 and S2.
- `src/aoi_builtup/postprocess/vectorize.py`: cumulative rasters, density zoning, polygonization, and geometry metrics.
- `src/aoi_builtup/resume.py`: artifact reuse and resume support.
- `src/aoi_builtup/qa.py`: run-report helpers and quantitative QA metrics.
- `src/aoi_builtup/pipeline.py`: orchestration.
- `src/aoi_builtup/cli.py`: CLI surface.

## Grid Policy

- `grid.crs: auto` resolves a local metric CRS from the AOI bounds.
- The example AOI currently resolves to `EPSG:32629`, but that is no longer hardcoded as the global default.
- Bounds are snapped outward to the configured grid resolution.
- Sentinel-1 and Sentinel-2 are resampled onto the same analysis grid.
- `B11` and `SCL` retain their lower native information content even when resampled to the analysis grid.

## Runtime Properties

- Manifests and stage outputs are persisted to disk for reproducibility and resume.
- Missing Sentinel-2 support does not abort the run when `allow_unavailable=true`.
- Polygonization is tiled by default to reduce memory pressure.
- The final run report records stage timings, counts, source reuse, and area metrics.
