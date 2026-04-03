from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import logging
import math
import re
import shutil


LOGGER = logging.getLogger(__name__)
BASELINE_LINE_PATTERN = re.compile(r"^[^\t:]+:\s*([^\s]+)\s+([^\s]+)\s+([^\s]+)")


@dataclass(frozen=True)
class CleanupRecord:
    category: str
    checkpoint: str
    reason: str
    path: Path
    bytes_reclaimed: int

    def as_dict(self) -> dict[str, object]:
        return {
            "category": self.category,
            "checkpoint": self.checkpoint,
            "reason": self.reason,
            "path": str(self.path),
            "bytes_reclaimed": self.bytes_reclaimed,
        }


@dataclass(frozen=True)
class CleanupWarning:
    category: str
    checkpoint: str
    reason: str
    path: Path
    message: str

    def as_dict(self) -> dict[str, object]:
        return {
            "category": self.category,
            "checkpoint": self.checkpoint,
            "reason": self.reason,
            "path": str(self.path),
            "message": self.message,
        }


def path_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for current in path.rglob("*"):
        if current.is_file():
            total += current.stat().st_size
    return total


def directory_has_files(path: Path) -> bool:
    return path.is_dir() and any(current.is_file() for current in path.rglob("*"))


def dimap_data_dir(product: Path) -> Path:
    return product.with_suffix(".data")


def is_valid_dimap_product(product: Path) -> bool:
    if not product.exists() or product.stat().st_size == 0:
        return False
    data_dir = dimap_data_dir(product)
    return directory_has_files(data_dir)


def are_valid_dimap_products(products: Iterable[Path]) -> bool:
    materialized = tuple(products)
    return bool(materialized) and all(is_valid_dimap_product(product) for product in materialized)


def _has_finite_baseline_metadata(base_file: Path) -> bool:
    try:
        lines = base_file.read_text(encoding="utf-8").splitlines()
    except OSError:
        return False

    required_prefixes = ("initial_baseline(TCN):", "initial_baseline_rate:")
    for prefix in required_prefixes:
        line = next((current for current in lines if current.startswith(prefix)), None)
        if line is None:
            return False
        match = BASELINE_LINE_PATTERN.match(line)
        if match is None:
            return False
        try:
            values = tuple(float(token) for token in match.groups())
        except ValueError:
            return False
        if not all(math.isfinite(value) for value in values):
            return False
    return True


def is_valid_snap_export_dir(export_dir: Path) -> bool:
    required = (export_dir / "rslc", export_dir / "diff0", export_dir / "geo")
    if not all(directory_has_files(path) for path in required):
        return False
    diff_dir = export_dir / "diff0"
    base_files = sorted(diff_dir.glob("*.base"))
    diff_files = sorted(diff_dir.glob("*.diff"))
    if not base_files or len(base_files) != len(diff_files):
        return False
    return all(_has_finite_baseline_metadata(base_file) for base_file in base_files)


def is_valid_stamps_outputs(points_csv: Path, timeseries_csv: Path) -> bool:
    return (
        points_csv.exists()
        and points_csv.stat().st_size > 0
        and timeseries_csv.exists()
        and timeseries_csv.stat().st_size > 0
    )


def delete_paths(
    paths: Iterable[Path],
    *,
    category: str,
    checkpoint: str,
    reason: str,
    logger: logging.Logger | None = None,
    best_effort: bool = False,
    retry_hidden_files: bool = False,
    warning_records: list[CleanupWarning] | None = None,
) -> list[CleanupRecord]:
    records: list[CleanupRecord] = []
    seen: set[Path] = set()
    log = logger or LOGGER
    for path in paths:
        resolved = path.resolve(strict=False)
        if resolved in seen or not path.exists():
            continue
        seen.add(resolved)
        reclaimed = path_size_bytes(path)
        dimap_sidecar = None
        if path.is_file() and path.suffix.lower() == ".dim":
            dimap_sidecar = dimap_data_dir(path)
            reclaimed += path_size_bytes(dimap_sidecar)
        if reclaimed == 0 and path.is_dir():
            log.info(
                "Skipping empty artifact cleanup target | category=%s checkpoint=%s path=%s",
                category,
                checkpoint,
                path,
            )
        else:
            log.info(
                "Deleting ephemeral artifact | category=%s checkpoint=%s bytes=%s path=%s reason=%s",
                category,
                checkpoint,
                reclaimed,
                path,
                reason,
            )
        try:
            _delete_path(path, dimap_sidecar=dimap_sidecar, retry_hidden_files=retry_hidden_files)
        except OSError as exc:
            if not best_effort:
                raise
            message = f"{exc.__class__.__name__}: {exc}"
            log.warning(
                "Keeping superseded artifact after non-fatal cleanup failure | category=%s checkpoint=%s path=%s error=%s",
                category,
                checkpoint,
                path,
                message,
            )
            if warning_records is not None:
                warning_records.append(
                    CleanupWarning(
                        category=category,
                        checkpoint=checkpoint,
                        reason=reason,
                        path=path,
                        message=message,
                    )
                )
            continue
        records.append(
            CleanupRecord(
                category=category,
                checkpoint=checkpoint,
                reason=reason,
                path=path,
                bytes_reclaimed=reclaimed,
            )
        )
    return records


def _delete_path(path: Path, *, dimap_sidecar: Path | None, retry_hidden_files: bool) -> None:
    if path.is_dir():
        _delete_directory(path, retry_hidden_files=retry_hidden_files)
        return

    path.unlink()
    if dimap_sidecar is not None and dimap_sidecar.exists():
        _delete_directory(dimap_sidecar, retry_hidden_files=retry_hidden_files)


def _delete_directory(path: Path, *, retry_hidden_files: bool) -> None:
    try:
        shutil.rmtree(path, ignore_errors=False)
    except OSError:
        if not retry_hidden_files or not path.exists():
            raise
        _delete_hidden_entries(path)
        shutil.rmtree(path, ignore_errors=False)


def _delete_hidden_entries(path: Path) -> None:
    hidden_entries = sorted(
        (current for current in path.rglob("*") if current.name.startswith(".")),
        key=lambda current: len(current.parts),
        reverse=True,
    )
    for current in hidden_entries:
        if not current.exists():
            continue
        if current.is_dir():
            shutil.rmtree(current, ignore_errors=True)
        else:
            try:
                current.unlink()
            except OSError:
                continue


def delete_matching_direct_children(
    root: Path,
    *,
    category: str,
    checkpoint: str,
    reason: str,
    name_contains: str | None = None,
    name_prefix: str | None = None,
    logger: logging.Logger | None = None,
) -> list[CleanupRecord]:
    if not root.exists():
        return []
    candidates: list[Path] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        if name_contains and name_contains not in child.name:
            continue
        if name_prefix and not child.name.startswith(name_prefix):
            continue
        candidates.append(child)
    return delete_paths(candidates, category=category, checkpoint=checkpoint, reason=reason, logger=logger)


def cleanup_stamps_workspace(root: Path, *, keep_paths: Sequence[Path], logger: logging.Logger | None = None) -> list[CleanupRecord]:
    if not root.exists():
        return []

    resolved_keeps = tuple(path.resolve(strict=False) for path in keep_paths)

    def should_preserve(path: Path) -> bool:
        resolved = path.resolve(strict=False)
        return any(resolved == keep or resolved in keep.parents for keep in resolved_keeps)

    reclaimed = 0
    log = logger or LOGGER
    for child in sorted(root.iterdir(), key=lambda current: current.name):
        if should_preserve(child):
            for nested in sorted(child.rglob("*"), key=lambda current: len(current.parts), reverse=True):
                if should_preserve(nested):
                    continue
                reclaimed += path_size_bytes(nested)
                if nested.is_dir():
                    shutil.rmtree(nested, ignore_errors=False)
                else:
                    nested.unlink()
            continue
        reclaimed += path_size_bytes(child)
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=False)
        else:
            child.unlink()

    if reclaimed == 0:
        return []

    log.info(
        "Deleting ephemeral artifact subtree | category=stamps_workspace checkpoint=parse_points_validated bytes=%s path=%s reason=StaMPS workspace is no longer needed once parsed point products exist.",
        reclaimed,
        root,
    )
    return [
        CleanupRecord(
            category="stamps_workspace",
            checkpoint="parse_points_validated",
            reason="StaMPS workspace is no longer needed once parsed point products exist.",
            path=root,
            bytes_reclaimed=reclaimed,
        )
    ]
