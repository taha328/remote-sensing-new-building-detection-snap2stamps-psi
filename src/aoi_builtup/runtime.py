from __future__ import annotations

import logging
import signal
import time
from contextlib import contextmanager
from typing import Any, Iterator

from dask.base import is_dask_collection
import xarray as xr


class PipelineInterruptedError(RuntimeError):
    """Raised when pipeline execution is interrupted by a signal."""


def _format_fields(fields: dict[str, Any]) -> str:
    parts = [f"{key}={value}" for key, value in fields.items() if value is not None]
    return " ".join(parts)


@contextmanager
def log_timing(logger: logging.Logger, event: str, **fields: Any) -> Iterator[None]:
    details = _format_fields(fields)
    prefix = f"{event} started"
    logger.info("%s%s", prefix, f" | {details}" if details else "")
    started = time.perf_counter()
    try:
        yield
    except BaseException:
        duration_s = time.perf_counter() - started
        logger.exception(
            "%s failed after %.2fs%s",
            event,
            duration_s,
            f" | {details}" if details else "",
        )
        raise
    duration_s = time.perf_counter() - started
    logger.info(
        "%s completed in %.2fs%s",
        event,
        duration_s,
        f" | {details}" if details else "",
    )


def dataset_profile(dataset: xr.Dataset) -> dict[str, Any]:
    chunks: dict[str, Any] = {}
    for name, data in dataset.data_vars.items():
        if getattr(data, "chunksizes", None):
            chunks[name] = {dim: list(values) for dim, values in data.chunksizes.items()}
    return {
        "sizes": dict(dataset.sizes),
        "vars": list(dataset.data_vars),
        "dask_backed": any(is_dask_collection(data.data) for data in dataset.data_vars.values()),
        "chunks": chunks or None,
    }


@contextmanager
def interruption_guard() -> Iterator[None]:
    previous_handlers: dict[int, Any] = {}

    def _handler(signum: int, frame: Any) -> None:
        signal_name = signal.Signals(signum).name
        raise PipelineInterruptedError(f"Received {signal_name}")

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, _handler)
    try:
        yield
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
