from pathlib import Path

from aoi_builtup.config import load_config
from aoi_builtup.grid import build_grid, load_aoi_frame


def test_load_example_config() -> None:
    config = load_config(Path("configs/aoi_builtup.yaml"))
    assert config.project == "aoi-builtup"
    assert config.grid.crs == "auto"
    assert config.periods[0].id == "2023_2025"
    assert config.sentinel1.collection == "sentinel-1-rtc"

    grid = build_grid(load_aoi_frame(config.aoi), config.grid)
    assert grid.crs == "EPSG:32629"
