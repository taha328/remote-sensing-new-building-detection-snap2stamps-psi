# Operations

## Environment

Create and activate an isolated environment, then install the pinned runtime:

```bash
cd /Users/tahaelouali/casablanca-builtup-pipeline
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.lock
python -m pip install -e .
```

Supported interpreter range:

- Python `3.11` or `3.12`
- Python `3.13` is intentionally not declared as supported in `pyproject.toml`

## Test commands

```bash
cd /Users/tahaelouali/casablanca-builtup-pipeline
source .venv/bin/activate
PYTHONPATH=src python -m pytest
PYTHONPATH=src python -m pytest tests/integration/test_small_pipeline.py
```

## Run modes

Fresh run:

```bash
casablanca-builtup run-pipeline --config configs/casablanca_city.yaml
```

Rerun same config as a new attempt:

```bash
casablanca-builtup run-pipeline --config configs/casablanca_city.yaml
```

Resume the latest attempt for the same config hash:

```bash
casablanca-builtup run-pipeline --config configs/casablanca_city.yaml --resume-latest
```

Resume a specific attempt:

```bash
casablanca-builtup run-pipeline --config configs/casablanca_city.yaml --run-dir runs/casablanca-builtup-<hash>/attempt-001
```

## Run directory policy

- Group root: `runs/<project-slug>-<config-hash>/`
- Attempt root: `runs/<project-slug>-<config-hash>/attempt-###/`
- Default behavior: create a new attempt directory.
- Resume behavior: reuse artifacts in an existing attempt when `overwrite=false`.
- Overwrite policy: if `cache.overwrite=true`, existing artifacts in the chosen attempt may be recomputed and overwritten.
- Resume policy: artifact presence, not report state, decides reuse.

## Evaluation

Compare the local cumulative refined raster against a frozen notebook raster:

```bash
casablanca-builtup evaluate-run \
  --run-dir runs/casablanca-builtup-<hash>/attempt-001 \
  --reference-raster data/reference/notebook_cumulative_refined.tif
```

Optional labelled comparison:

```bash
casablanca-builtup evaluate-run \
  --run-dir runs/casablanca-builtup-<hash>/attempt-001 \
  --reference-raster data/reference/notebook_cumulative_refined.tif \
  --labels-raster data/reference/reference_labels.tif
```

Outputs:

- `reports/evaluation/evaluation_summary.json`
- `reports/evaluation/reference_metrics.json`
- `reports/evaluation/reference_confusion.tif`
- `reports/evaluation/labels_metrics.json` when labels are provided
- `reports/evaluation/labels_confusion.tif` when labels are provided
