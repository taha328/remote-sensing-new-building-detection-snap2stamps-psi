# Casablanca Built-Up Pipeline

Local production-grade pipeline for probable built-up change detection over Casablanca.

Core design:

- Sentinel-1 RTC is the primary detector.
- Sentinel-2 L2A is a soft support layer, not a hard veto.
- Planetary Computer STAC replaces Google Earth Engine acquisition.
- Local raster processing replaces server-side GEE reductions.
- Reproducibility is enforced through config hashing, frozen STAC manifests, and on-disk intermediate artifacts.
- Run reports are written as JSON and pipeline logs are written per run.
- Sentinel-2 absence degrades gracefully to S1-only retention instead of aborting the run.
- Polygonization uses a tiled strategy by default to reduce peak memory pressure.

Entry points:

- `casablanca-builtup run-pipeline --config configs/casablanca_city.yaml`
- `casablanca-builtup acquire-data --config configs/casablanca_city.yaml`
- `casablanca-builtup polygonize --config configs/casablanca_city.yaml`
- `casablanca-builtup run-pipeline --config configs/casablanca_city.yaml --resume-latest`
- `casablanca-builtup evaluate-run --run-dir runs/<group>/attempt-001 --reference-raster data/reference/notebook_cumulative_refined.tif`

PSI entry points:

- `casablanca-psi run-pipeline --config configs/psi_casablanca_slc.yaml`
- `casablanca-psi acquire --config configs/psi_casablanca_slc.yaml`
- `casablanca-psi download-slc --config configs/psi_casablanca_slc.yaml`
- `casablanca-psi run-snap --config configs/psi_casablanca_slc.yaml`
- `casablanca-psi run-stamps --config configs/psi_casablanca_slc.yaml`
- `casablanca-psi parse-psi --config configs/psi_casablanca_slc.yaml`
- `casablanca-psi fuse --config configs/psi_casablanca_slc.yaml`
- `casablanca-psi evaluate --points runs_psi/<group>/attempt-001/vectors/psi_candidates.parquet --reference-points data/reference/psi_points.gpkg --output-dir runs_psi/<group>/attempt-001/reports/evaluation`

PSI notes:

- The PSI workflow is Sentinel-1 SLC based and expects external SNAP + StaMPS tooling.
- Ascending and descending stacks are processed separately and fused only after PSI outputs are parsed.
- The repository includes a production-grade Python scaffold, placeholder SNAP graph templates, and a placeholder StaMPS export script at [export_ps_points.m](/Users/tahaelouali/casablanca-builtup-pipeline/resources/stamps/export_ps_points.m). Those resources must be replaced with a validated SNAP/StaMPS implementation before operational use.

Environment setup:

```bash
cd /Users/tahaelouali/casablanca-builtup-pipeline
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.lock
python -m pip install -e .
```

Test execution:

```bash
cd /Users/tahaelouali/casablanca-builtup-pipeline
source .venv/bin/activate
PYTHONPATH=src python -m pytest
```

See [docs/architecture.md](/Users/tahaelouali/casablanca-builtup-pipeline/docs/architecture.md), [docs/migration-plan.md](/Users/tahaelouali/casablanca-builtup-pipeline/docs/migration-plan.md), [docs/operations.md](/Users/tahaelouali/casablanca-builtup-pipeline/docs/operations.md), and [docs/psi-architecture.md](/Users/tahaelouali/casablanca-builtup-pipeline/docs/psi-architecture.md).
