# PSI Architecture

## Scope

The `aoi-psi` workflow is a local Sentinel-1 SLC to SNAP to StaMPS to Python orchestration chain for PSI and CDPSI-based emergence detection.

## Stage Order

1. Load configuration and create the run context.
2. Resolve the AOI and build stack manifests.
3. Download SLC scenes and organize them per orbit stack.
4. Run SNAP preprocessing for each stack and each CDPSI subset stack.
5. Run StaMPS for each exported stack.
6. Parse the StaMPS point export in Python.
7. Build CDPSI emergence artifacts from complete and subset stacks.
8. Fuse all stack outputs into final point and polygon candidates.
9. Export reports, point files, polygon files, and evaluation products.

## Ownership Boundary

SNAP handles:

- orbit correction
- TOPS split and deburst
- co-registration and interferometric preparation
- topographic phase removal
- StaMPS export preparation

StaMPS handles:

- persistent scatterer candidate selection
- PSI inversion and temporal coherence estimation
- time-series and per-point metrics

Python handles:

- acquisition manifests and downloads
- run IDs, caching, resume logic, and reports
- CDPSI subset planning
- PSI export parsing
- emergence scoring
- optional amplitude and Sentinel-2 fusion
- vector export and evaluation

## Key Runtime Details

- Ascending and descending stacks are processed independently.
- CDPSI is implemented by generating subset stacks around candidate break dates and comparing them to the complete stack.
- Parsed PSI outputs are reprojected into a metric CRS inferred from the AOI before clustering and buffering.
- Artifact cleanup is configurable through `artifact_lifecycle`.
- Reuse can happen independently for manifests, downloads, SNAP exports, and StaMPS outputs.

## Run Layout

Each PSI attempt writes:

- `manifests/`
- `raw/slc/`
- `raw/dem/`
- `snap/`
- `stamps/`
- `staging/`
- `points/`
- `vectors/`
- `reports/`
- `logs/`

## Scientific Caveats

- PSI emergence evidence is not a direct building-footprint extractor.
- Residual height and DEM error are supportive signals, not final labels.
- The repository orchestrates the workflow, but final scientific quality still depends on the correctness of the SNAP graphs, the StaMPS setup, and the export contract in `resources/stamps/export_ps_points.m`.
