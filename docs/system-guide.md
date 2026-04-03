# System Guide

## 1. What This Repository Is

This repository is an AOI-agnostic local remote-sensing pipeline for new-building and built-up change detection.

It contains two production-oriented workflows:

- `aoi-builtup`: a raster pipeline based on Sentinel-1 RTC as the primary detector with Sentinel-2 L2A as an optional refinement layer.
- `aoi-psi`: a Sentinel-1 SLC orchestration pipeline that runs SNAP preprocessing, StaMPS PSI, CDPSI subset analysis, and downstream evidence fusion.

The repository is designed for local execution, reproducibility, and stage-level resume. It is not a hosted service, not a notebook-only prototype, and not a guaranteed building-footprint extractor.

## 2. End Products

The built-up workflow produces:

- probable change rasters
- refined cumulative change rasters
- density-zone rasters
- polygon outputs in GeoParquet and GeoPackage
- run reports, manifests, and evaluation outputs

The PSI workflow produces:

- stack manifests and download records
- StaMPS-compatible SNAP exports
- parsed PSI point tables
- CDPSI emergence candidate points
- fused polygon candidates
- run reports and evaluation outputs

## 3. Top-Level Repository Layout

- `configs/`: example configuration files.
- `docs/`: architecture, operations, onboarding, and this guide.
- `resources/`: SNAP graph templates, StaMPS export script, and MATLAB helpers.
- `src/aoi_builtup/`: built-up workflow package.
- `src/aoi_psi/`: PSI and CDPSI workflow package.
- `tests/`: unit and integration tests.

## 4. The Built-Up Workflow From Start To Finish

### 4.1 Entry Points

The CLI lives in `src/aoi_builtup/cli.py`.

Main commands:

- `aoi-builtup acquire-data`
- `aoi-builtup build-composites`
- `aoi-builtup detect-s1`
- `aoi-builtup refine-s2`
- `aoi-builtup polygonize`
- `aoi-builtup export`
- `aoi-builtup run-pipeline`
- `aoi-builtup evaluate-run`

### 4.2 Configuration Loading

`src/aoi_builtup/config.py` defines the complete schema:

- AOI definition
- before/after time windows
- Sentinel-1 parameters
- Sentinel-2 parameters
- grid controls
- change-detection thresholds
- polygonization thresholds
- export settings
- cache policy
- Dask settings
- run settings

The built-up example config is `configs/aoi_builtup.yaml`.

### 4.3 AOI And Grid Resolution

`src/aoi_builtup/grid.py` loads the AOI from either:

- `aoi.path`
- `aoi.bbox`

The grid is then built as follows:

1. The AOI is normalized to the configured AOI CRS.
2. The analysis CRS is resolved.
3. If `grid.crs` is `auto`, the code infers a local metric CRS from the AOI bounds.
4. Bounds are snapped outward to the configured resolution.
5. A deterministic `GeoBox` is created for all raster stages.

This removes the previous hardcoded dependency on a single city-specific UTM zone.

### 4.4 Acquisition And Manifests

`src/aoi_builtup/acquisition/stac.py` searches:

- Planetary Computer Sentinel-1 RTC
- Planetary Computer Sentinel-2 L2A

The pipeline freezes manifest files to disk before processing. That makes reruns deterministic and allows later stages to reuse acquisitions without re-querying STAC.

### 4.5 Sentinel-1 Processing

The core S1 path is:

- `src/aoi_builtup/s1/composite.py`
- `src/aoi_builtup/s1/detection.py`

The pipeline:

1. signs STAC items
2. loads them directly onto the analysis grid
3. builds temporal median composites
4. computes ratio-based and statistical change signals
5. cleans the result with morphology and connected-component filtering

This is the primary detector. Everything else refines it.

### 4.6 Sentinel-2 Refinement

The optical branch is implemented in:

- `src/aoi_builtup/s2/composite.py`
- `src/aoi_builtup/s2/refinement.py`
- `src/aoi_builtup/fusion.py`

The logic is intentionally conservative:

- S2 does not generate candidate buildings by itself.
- S2 only refines S1 candidates.
- If S2 is unavailable and `allow_unavailable=true`, the pipeline keeps the S1 candidates instead of aborting.

The support logic uses:

- NDVI
- NDBI
- MNDWI
- optional BSI
- clear observation counts
- reliability masks

### 4.7 Postprocessing And Polygonization

`src/aoi_builtup/postprocess/vectorize.py` handles:

- cumulative first-change rasters
- density zoning
- morphology
- raster-to-vector polygonization
- compactness and area filtering

Polygonization is tiled by default to avoid excessive peak memory.

### 4.8 Resume, Reports, And Evaluation

- `src/aoi_builtup/resume.py` handles artifact reuse.
- `src/aoi_builtup/qa.py` defines run-report updates and quantitative metrics.
- `src/aoi_builtup/evaluation.py` compares local results against reference rasters.

Each run records:

- stage status
- source reuse versus recomputation
- candidate counts
- area metrics
- output locations
- failure state if a stage aborts

## 5. The PSI Workflow From Start To Finish

### 5.1 Entry Points

The CLI lives in `src/aoi_psi/cli.py`.

Main commands:

- `aoi-psi validate-provider`
- `aoi-psi run-pipeline`
- `aoi-psi acquire`
- `aoi-psi download-slc`
- `aoi-psi run-snap`
- `aoi-psi run-stamps`
- `aoi-psi parse-psi`
- `aoi-psi fuse`
- `aoi-psi evaluate`

### 5.2 Configuration Model

`src/aoi_psi/config.py` defines:

- AOI definition
- SLC acquisition settings
- authentication and S3 download controls
- DEM path
- orbit stack definitions
- SNAP runtime parameters
- StaMPS runtime parameters
- PSI/CDPSI controls
- optional amplitude and S2 refinement controls
- fusion weights
- cache and artifact lifecycle policies
- run settings

The main examples are:

- `configs/aoi_psi_slc.yaml`
- `configs/aoi_psi_slc_minimal.yaml`
- `configs/aoi_psi_slc_cdpsi_min6.yaml`

### 5.3 Acquisition And Stack Planning

`src/aoi_psi/acquisition.py` is responsible for:

- AOI geometry loading
- STAC search over Sentinel-1 SLC
- provider field validation
- scene filtering per orbit stack
- manifest writing
- downloads via CDSE HTTP or S3

Each stack is defined explicitly by direction, relative orbit, polarization, and stack-length constraints.

### 5.4 SNAP Stage

`src/aoi_psi/snap.py` wraps the SNAP `gpt` runner and graph execution.

Responsibilities:

- prepare graph parameters
- run stack preprocessing
- export StaMPS-compatible products
- support CDPSI subset stacks
- optionally clean intermediate artifacts

The graph templates live in `resources/snap_graphs/`.

### 5.5 StaMPS Stage

`src/aoi_psi/stamps.py` manages:

- StaMPS workspace creation
- patch layout
- unwrap tool resolution
- MATLAB or Octave execution
- final export collection

The export contract depends on `resources/stamps/export_ps_points.m`.

### 5.6 PSI Parsing And CDPSI

The downstream Python logic uses:

- `src/aoi_psi/psi_results.py`
- `src/aoi_psi/cdpsi.py`

This stage:

1. reads the StaMPS point export
2. preserves pixel keys such as `azimuth_index` and `range_index`
3. reprojects lon/lat outputs into a metric CRS resolved from the AOI
4. compares the full stack with subset stacks
5. scores emergence candidates

### 5.7 Fusion And Export

`src/aoi_psi/fusion.py` clusters PSI points into polygons and assigns confidence using weighted evidence:

- PSI primary evidence
- optional amplitude evidence
- optional Sentinel-2 evidence
- optional context layers

The pipeline then exports:

- point candidates
- polygon candidates
- summary JSON
- run reports

## 6. Run Context And Directory Structure

Both workflows create a deterministic group ID from the project slug and config hash, then allocate attempt directories.

Built-up attempts contain:

- `manifests/`
- `rasters/`
- `staging/`
- `vectors/`
- `reports/`
- `logs/`

PSI attempts contain:

- `manifests/`
- `raw/`
- `snap/`
- `stamps/`
- `staging/`
- `rasters/`
- `points/`
- `vectors/`
- `reports/`
- `logs/`

## 7. How To Run The Repository

### 7.1 Install

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.lock
python -m pip install -e .
```

### 7.2 Run The Built-Up Workflow

```bash
aoi-builtup run-pipeline --config configs/aoi_builtup.yaml
```

### 7.3 Run The PSI Workflow

```bash
aoi-psi run-pipeline --config configs/aoi_psi_slc.yaml
```

### 7.4 Test The Repository

```bash
PYTHONPATH=src python -m pytest
```

## 8. How To Adapt It To A New AOI

1. Copy the example config you want.
2. Replace the AOI geometry.
3. Replace the time windows.
4. Replace the DEM path.
5. Check the external SNAP and StaMPS paths.
6. Run the minimal PSI config before the full stack.
7. Freeze the first working run as your reproducibility baseline.

## 9. Important Operational Caveats

- The built-up pipeline is local and artifact-heavy by design; disk usage matters.
- The PSI workflow depends on external software not vendored in this repository.
- Scientific quality depends on the validity of the SNAP graphs and the StaMPS export contract.
- PSI outputs and built-up polygons are candidate products; they still need domain-specific validation before operational deployment.
