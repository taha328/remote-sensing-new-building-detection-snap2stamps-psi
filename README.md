# Remote-Sensing New-Building Detection Pipeline

AOI-agnostic local remote-sensing repository for new-building and built-up change detection.

The codebase exposes two complementary workflows:

- `aoi-builtup`: Sentinel-1 RTC primary detection, Sentinel-2 refinement, cumulative zoning, and polygon export.
- `aoi-psi`: Sentinel-1 SLC acquisition, SNAP preprocessing, StaMPS PSI/CDPSI parsing, and evidence fusion.

## Repository Structure

- `src/aoi_builtup/`: built-up change detection package.
- `src/aoi_psi/`: PSI, CDPSI, SNAP, and StaMPS orchestration package.
- `configs/aoi_builtup.yaml`: built-up example configuration.
- `configs/aoi_psi_slc.yaml`: full PSI example configuration.
- `configs/aoi_psi_slc_minimal.yaml`: reduced-cost PSI smoke-test configuration.
- `configs/aoi_psi_slc_cdpsi_min6.yaml`: smallest valid CDPSI example stack.
- `resources/snap_graphs/`: SNAP graph templates.
- `resources/stamps/export_ps_points.m`: StaMPS export contract consumed by the parser.

## Environment Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.lock
python -m pip install -e .
```

The PSI workflow also requires external SNAP and StaMPS installations plus valid Copernicus Data Space credentials.

## Quick Start

Built-up workflow:

```bash
aoi-builtup run-pipeline --config configs/aoi_builtup.yaml
```

PSI workflow:

```bash
aoi-psi run-pipeline --config configs/aoi_psi_slc.yaml
```

Run the test suite:

```bash
PYTHONPATH=src python -m pytest
```

## Documentation

- [System Guide](docs/system-guide.md)
- [Built-Up Architecture](docs/architecture.md)
- [PSI Architecture](docs/psi-architecture.md)
- [Operations Guide](docs/operations.md)
- [AOI Onboarding Checklist](docs/migration-plan.md)
