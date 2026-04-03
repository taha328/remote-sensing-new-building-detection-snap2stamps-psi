# Architecture

## Product definition

The final product is a set of:

- probable built-up change rasters
- refined built-up change zones
- built-up change polygons

It is not a guaranteed building footprint extractor.

## Processing stages

1. Load and validate YAML configuration.
2. Resolve AOI geometry and build a deterministic analysis grid.
3. Query Planetary Computer STAC for Sentinel-1 RTC and Sentinel-2 L2A.
4. Freeze raw STAC item manifests to disk for reproducibility.
5. Load Sentinel-1 items onto the target grid and build before/after seasonal composites.
6. Run Sentinel-1 change detection with:
   - VV likelihood-ratio p-value
   - VV ratio increase
   - VH support
   - connected-component cleanup
7. Load Sentinel-2 composites on the same grid.
8. Build optical support layers:
   - NDVI
   - NDBI
   - MNDWI
   - optional BSI
   - clear observation counts
9. Fuse S1 and S2 with a soft rule:
   - keep S1 where S2 is unreliable
   - require moderate S2 support where S2 is reliable
   - allow strong-S1 override for very strong radar evidence
10. Build cumulative first-change raster.
11. Compute density zones and morphology on the cumulative refined mask.
12. Polygonize locally, then filter polygons by area and compactness.
13. Export rasters, vectors, manifests, metrics, and a run report.
14. Persist stage status and quantitative QA metrics per period.

## Data flow

`config.yaml`
-> `RunContext`
-> `AOI geometry`
-> `aligned grid`
-> `STAC manifests`
-> `period composites`
-> `S1 candidate masks`
-> `S2 support layers`
-> `refined masks`
-> `cumulative raster`
-> `zone mask`
-> `polygons`
-> `reports / exports`

## Module responsibilities

- `config.py`: schema validation and YAML loading.
- `run_context.py`: deterministic run IDs and output path management.
- `grid.py`: CRS, bounds, transform, and alignment policy.
- `acquisition/stac.py`: STAC search, manifest persistence, and signed item reconstruction.
- `s1/composite.py`: deterministic S1 loading and temporal median compositing.
- `s1/detection.py`: LRT, ratios, morphology, and CC cleanup.
- `s2/composite.py`: S2 cloud masking, indices, and clear-observation accounting.
- `s2/refinement.py`: support score and reliability mask.
- `fusion.py`: soft refinement logic.
- `postprocess/vectorize.py`: cumulative rasters, density zoning, polygonization, geometry metrics.
- `pipeline.py`: stage orchestration and run reports.
- `qa.py`: run report schema and quantitative QA helpers.
- `resume.py`: artifact registry and artifact-level resume/reuse.
- `evaluation.py`: notebook-vs-local and label-vs-local raster comparison utilities.
- `cli.py`: Typer CLI entry points.

## Sentinel-1 vs Sentinel-2 responsibilities

- Sentinel-1 is the primary detector.
- Sentinel-2 only refines S1 candidates.
- Fusion occurs after S1 candidate generation and before cumulative zoning.
- Polygonization happens only on the final cleaned refined mask.
- QA runs after each stage and is summarized in the final report.

## Grid and alignment policy

- Target CRS: `EPSG:32629` for Casablanca.
- Target resolution: `10 m`.
- Bounds: AOI bounds snapped outward to the `10 m` grid.
- Transform: north-up affine transform, anchored on pixel edges.
- S1 loading: direct to the 10 m master grid.
- S2 loading: all support bands aligned to the same 10 m grid.
- Important caveat: `B11` and `SCL` remain 20 m information resampled onto the 10 m grid; they are used only for support scoring, not footprint delineation.

## Scalability decisions

- Time stacks are loaded lazily with Dask.
- Post-composite 2D masks are intentionally materialized before connected components and morphology.
- Vectorization is delayed until the final cleaned mask, which is much smaller than the raw candidate space.
- Intermediate outputs are written to disk so later stages can resume without recomputing acquisitions.
- Missing Sentinel-2 support does not abort the run; the pipeline keeps S1 candidates and records the degraded support mode in the report.
- Polygonization is tiled by default to reduce peak memory pressure before final dissolve.
