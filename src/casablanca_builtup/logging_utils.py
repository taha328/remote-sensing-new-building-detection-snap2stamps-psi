from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(level: str, log_path: Path | None = None) -> None:
    log_level = getattr(logging, level.upper(), logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(log_level)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
