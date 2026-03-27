from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
import yaml


def write_test_config(tmp_path: Path, *, overwrite: bool = False, allow_unavailable: bool = True) -> Path:
    project_root = tmp_path / "project"
    config_dir = project_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "project": "synthetic-builtup",
        "aoi": {
            "name": "synthetic-aoi",
            "bbox": [-7.80, 33.40, -7.70, 33.50],
            "crs": "EPSG:4326",
        },
        "periods": [
            {
                "id": "2023_2025",
                "before": {"start": "2023-06-01", "end": "2023-06-30"},
                "after": {"start": "2025-06-01", "end": "2025-06-30"},
            }
        ],
        "sentinel2": {
            "allow_unavailable": allow_unavailable,
            "use_bsi": True,
            "min_clear_observations": 2,
            "min_clear_fraction": 0.15,
        },
        "export": {
            "save_intermediates": True,
            "vector_formats": ["parquet", "gpkg"],
        },
        "cache": {
            "reuse_manifests": True,
            "overwrite": overwrite,
        },
        "run": {
            "output_root": "runs",
            "log_level": "INFO",
            "write_log_file": False,
        },
    }
    config_path = config_dir / "test_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return config_path


def fake_manifests(with_s2: bool = True) -> dict[str, dict[str, list[dict[str, Any]]]]:
    manifests = {
        "2023_2025": {
            "s1_before": [{"phase": "before", "sensor": "s1"}],
            "s1_after": [{"phase": "after", "sensor": "s1"}],
        }
    }
    if with_s2:
        manifests["2023_2025"]["s2_before"] = [{"phase": "before", "sensor": "s2"}]
        manifests["2023_2025"]["s2_after"] = [{"phase": "after", "sensor": "s2"}]
    else:
        manifests["2023_2025"]["s2_before"] = []
        manifests["2023_2025"]["s2_after"] = []
    return manifests


def _coords(size: int = 8) -> dict[str, np.ndarray]:
    return {
        "y": np.array([75.0, 65.0, 55.0, 45.0, 35.0, 25.0, 15.0, 5.0][:size]),
        "x": np.array([5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0][:size]),
    }


def synthetic_s1_dataset(phase: str) -> xr.Dataset:
    coords = _coords()
    vv = np.ones((8, 8), dtype=np.float32)
    vh = np.ones((8, 8), dtype=np.float32)
    if phase == "after":
        vv[2:5, 2:5] = 8.0
        vh[2:5, 2:5] = 1.3
    return xr.Dataset(
        {
            "vv": xr.DataArray(vv, dims=("y", "x"), coords=coords),
            "vh": xr.DataArray(vh, dims=("y", "x"), coords=coords),
        }
    )


def synthetic_s2_dataset(phase: str) -> xr.Dataset:
    coords = _coords()
    ndvi = np.full((8, 8), 0.55, dtype=np.float32)
    ndbi = np.full((8, 8), -0.20, dtype=np.float32)
    mndwi = np.full((8, 8), 0.10, dtype=np.float32)
    bsi = np.full((8, 8), -0.10, dtype=np.float32)
    if phase == "after":
        ndvi[2:5, 2:5] = 0.20
        ndbi[2:5, 2:5] = 0.20
        mndwi[2:5, 2:5] = -0.20
        bsi[2:5, 2:5] = 0.10
    data = {
        "blue": xr.DataArray(np.full((8, 8), 0.1, dtype=np.float32), dims=("y", "x"), coords=coords),
        "green": xr.DataArray(np.full((8, 8), 0.1, dtype=np.float32), dims=("y", "x"), coords=coords),
        "red": xr.DataArray(np.full((8, 8), 0.1, dtype=np.float32), dims=("y", "x"), coords=coords),
        "nir": xr.DataArray(np.full((8, 8), 0.2, dtype=np.float32), dims=("y", "x"), coords=coords),
        "swir1": xr.DataArray(np.full((8, 8), 0.2, dtype=np.float32), dims=("y", "x"), coords=coords),
        "ndvi": xr.DataArray(ndvi, dims=("y", "x"), coords=coords),
        "ndbi": xr.DataArray(ndbi, dims=("y", "x"), coords=coords),
        "mndwi": xr.DataArray(mndwi, dims=("y", "x"), coords=coords),
        "bsi": xr.DataArray(bsi, dims=("y", "x"), coords=coords),
        "clear_count": xr.DataArray(np.full((8, 8), 3, dtype=np.uint16), dims=("y", "x"), coords=coords),
        "valid_count": xr.DataArray(np.full((8, 8), 3, dtype=np.uint16), dims=("y", "x"), coords=coords),
        "clear_fraction": xr.DataArray(np.full((8, 8), 0.6, dtype=np.float32), dims=("y", "x"), coords=coords),
        "valid_fraction": xr.DataArray(np.full((8, 8), 0.6, dtype=np.float32), dims=("y", "x"), coords=coords),
    }
    return xr.Dataset(data)


def install_fake_pipeline(monkeypatch, pipeline_module, *, with_s2: bool = True) -> None:
    monkeypatch.setattr(pipeline_module, "build_period_manifests", lambda *args, **kwargs: fake_manifests(with_s2=with_s2))
    monkeypatch.setattr(pipeline_module, "sign_manifest_items", lambda items: items)
    monkeypatch.setattr(
        pipeline_module,
        "build_s1_composite",
        lambda items, geobox, config, dask_config, **kwargs: synthetic_s1_dataset(items[0]["phase"]),
    )
    monkeypatch.setattr(
        pipeline_module,
        "build_s2_composite",
        lambda items, geobox, config, dask_config, **kwargs: synthetic_s2_dataset(items[0]["phase"]),
    )
