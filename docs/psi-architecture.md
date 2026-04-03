# PSI-Based New-Building Detection

This scaffold defines a technically correct local PSI-first workflow for Casablanca:

1. acquire Sentinel-1 SLC scenes in separate ascending/descending stacks
2. preprocess each consistent-orbit stack in SNAP
3. export StaMPS-compatible inputs
4. run PSI in StaMPS
5. parse standardized PSI point exports in Python
6. optionally add amplitude-change support
7. optionally add Sentinel-2 L2A built-up support
8. fuse evidence into point and polygon candidate outputs

## Processing ownership

- SNAP:
  - orbit correction
  - TOPS split/deburst
  - co-registration / back-geocoding
  - interferometric preparation
  - topographic phase removal
  - StaMPS export
- StaMPS:
  - persistent scatterer candidate selection
  - PSI inversion / phase stability estimation
  - temporal coherence and DEM error estimation
  - PSI time-series and point metrics export
- Python:
  - acquisition manifests and downloads
  - run IDs, resume, logging, QA
  - PSI export parsing
  - emergence timing logic
  - amplitude branch
  - Sentinel-2 refinement
  - fusion
  - vector/raster/report export

## Disk layout

Each run writes:

- `manifests/`: frozen SLC stack manifests
- `raw/slc/`: downloaded `.zip` or SAFE packages
- `raw/dem/`: DEM copies or links
- `snap/`: stack-specific SNAP workspace
- `stamps/`: stack-specific StaMPS workspace and exports
- `staging/`: intermediate gridded artifacts
- `points/`: PSI candidate point products
- `vectors/`: fused polygon outputs
- `reports/`: run report, QA report, evaluation report
- `logs/`: run logs

## Scientific caution

- Residual height / DEM error is supportive evidence only.
- PSI-supported emergence is not equivalent to guaranteed building footprints.
- Ascending and descending stacks are processed separately and only fused downstream at the evidence layer.
