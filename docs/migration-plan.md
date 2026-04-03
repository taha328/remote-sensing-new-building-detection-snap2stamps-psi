# AOI Onboarding Checklist

## 1. Duplicate The Example Config You Need

- built-up workflow: start from `configs/aoi_builtup.yaml`
- full PSI workflow: start from `configs/aoi_psi_slc.yaml`
- minimal PSI smoke test: start from `configs/aoi_psi_slc_minimal.yaml`

## 2. Replace The AOI Definition

- set `aoi.name` to a stable slug for the target AOI
- replace the example `bbox` with your own bbox, or switch to `aoi.path`
- keep `aoi.crs` aligned with the AOI source data

## 3. Verify The Grid Strategy

- keep `grid.crs: auto` unless you have a specific projected CRS requirement
- keep `resolution_m` aligned with the analysis you want to perform

## 4. Replace Example Time Windows

- set built-up before/after windows for your seasonal comparison
- set PSI acquisition windows for the stack length you need
- verify scene density before running expensive preprocessing

## 5. Prepare Reference Inputs

- update DEM paths under `dem.path`
- update evaluation references if you plan to run evaluation commands
- review optional context layers for PSI fusion

## 6. Validate External Tooling

- confirm SNAP `gpt_path` and `gpt_vmoptions_path`
- confirm StaMPS `install_root`
- confirm MATLAB or Octave configuration
- confirm Copernicus credentials

## 7. Smoke Test First

- run `aoi-builtup acquire-data` before full built-up processing
- run `aoi-psi validate-provider`
- run `aoi-psi run-pipeline --config configs/aoi_psi_slc_minimal.yaml` before the full PSI stack

## 8. Freeze The Working Configuration

- once the configuration works, keep the resolved config, manifests, and run report from the first successful attempt as your reproducibility baseline
