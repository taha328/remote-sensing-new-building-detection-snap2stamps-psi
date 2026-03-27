from pathlib import Path

from casablanca_builtup.config import load_config


def test_load_example_config() -> None:
    config = load_config(Path("configs/casablanca_city.yaml"))
    assert config.project == "casablanca-builtup"
    assert config.grid.crs == "EPSG:32629"
    assert config.periods[0].id == "2023_2025"
    assert config.sentinel1.collection == "sentinel-1-rtc"
