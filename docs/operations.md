# Operations

## Environment Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.lock
python -m pip install -e .
```

Supported interpreter range:

- Python `3.11`
- Python `3.12`

## Required External Tools For PSI

- ESA SNAP with a working `gpt` executable
- StaMPS
- MATLAB or Octave, depending on your `stamps` configuration
- Copernicus Data Space credentials

Common credential environment variables:

- `CDSE_ACCESS_TOKEN`
- `ACCESS_TOKEN`
- `REFRESH_TOKEN`
- `CDSE_S3_ACCESS_KEY`
- `CDSE_S3_SECRET_KEY`

## Test Commands

```bash
PYTHONPATH=src python -m pytest
PYTHONPATH=src python -m pytest tests/integration/test_small_pipeline.py
PYTHONPATH=src python -m pytest tests/psi
```

## Built-Up Commands

Fresh run:

```bash
aoi-builtup run-pipeline --config configs/aoi_builtup.yaml
```

Stop after acquisition:

```bash
aoi-builtup acquire-data --config configs/aoi_builtup.yaml
```

Resume the latest attempt for the same config hash:

```bash
aoi-builtup run-pipeline --config configs/aoi_builtup.yaml --resume-latest
```

Evaluate a completed run:

```bash
aoi-builtup evaluate-run \
  --run-dir runs/<project-slug>-<hash>/attempt-001 \
  --reference-raster data/reference/notebook_cumulative_refined.tif
```

## PSI Commands

Validate provider mapping:

```bash
aoi-psi validate-provider --config configs/aoi_psi_slc.yaml
```

Run the full PSI workflow:

```bash
aoi-psi run-pipeline --config configs/aoi_psi_slc.yaml
```

Run stage by stage:

```bash
aoi-psi acquire --config configs/aoi_psi_slc.yaml
aoi-psi download-slc --config configs/aoi_psi_slc.yaml
aoi-psi run-snap --config configs/aoi_psi_slc.yaml
aoi-psi run-stamps --config configs/aoi_psi_slc.yaml
aoi-psi parse-psi --config configs/aoi_psi_slc.yaml
aoi-psi fuse --config configs/aoi_psi_slc.yaml
```

Evaluate PSI outputs:

```bash
aoi-psi evaluate \
  --output-dir runs_psi/<project-slug>-<hash>/attempt-001/reports/evaluation \
  --points runs_psi/<project-slug>-<hash>/attempt-001/vectors/psi_candidates.parquet \
  --reference-points data/reference/psi_points.gpkg
```

## Run Directory Policy

Built-up runs:

- `runs/<project-slug>-<config-hash>/attempt-###/`

PSI runs:

- `runs_psi/<project-slug>-<config-hash>/attempt-###/`

Both workflows support:

- new-attempt execution by default
- artifact reuse when `overwrite=false`
- explicit resume via `--resume-latest`
- explicit resume via `--run-dir`
