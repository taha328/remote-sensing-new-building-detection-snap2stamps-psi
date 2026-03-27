from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from casablanca_builtup.config import PipelineConfig
from casablanca_builtup.utils.hashing import stable_config_hash, slugify


@dataclass(frozen=True)
class RunContext:
    group_id: str
    attempt_id: str
    run_id: str
    output_root: Path
    group_root: Path
    root: Path
    manifests_dir: Path
    rasters_dir: Path
    staging_dir: Path
    vectors_dir: Path
    reports_dir: Path
    logs_dir: Path

    @classmethod
    def create(
        cls,
        config: PipelineConfig,
        project_root: Path,
        run_dir: Path | None = None,
        resume_latest: bool = False,
    ) -> "RunContext":
        output_root = project_root / config.run.output_root
        group_id = f"{slugify(config.project)}-{stable_config_hash(config)}"
        group_root = output_root / group_id
        group_root.mkdir(parents=True, exist_ok=True)

        if run_dir is not None:
            root = Path(run_dir).resolve()
            attempt_id = root.name
            group_root = root.parent
        elif resume_latest:
            root = latest_attempt_dir(group_root)
            attempt_id = root.name
        else:
            attempt_id = next_attempt_id(group_root)
            root = group_root / attempt_id

        run_id = f"{group_id}-{attempt_id}"
        return cls(
            group_id=group_id,
            attempt_id=attempt_id,
            run_id=run_id,
            output_root=output_root,
            group_root=group_root,
            root=root,
            manifests_dir=root / "manifests",
            rasters_dir=root / "rasters",
            staging_dir=root / "staging",
            vectors_dir=root / "vectors",
            reports_dir=root / "reports",
            logs_dir=root / "logs",
        )

    def ensure_directories(self) -> None:
        for path in (
            self.root,
            self.manifests_dir,
            self.rasters_dir,
            self.staging_dir,
            self.vectors_dir,
            self.reports_dir,
            self.logs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


ATTEMPT_RE = re.compile(r"^attempt-(\d{3})$")


def list_attempt_dirs(group_root: Path) -> list[Path]:
    if not group_root.exists():
        return []
    attempts = [path for path in group_root.iterdir() if path.is_dir() and ATTEMPT_RE.match(path.name)]
    return sorted(attempts)


def next_attempt_id(group_root: Path) -> str:
    attempts = list_attempt_dirs(group_root)
    if not attempts:
        return "attempt-001"
    latest = max(int(ATTEMPT_RE.match(path.name).group(1)) for path in attempts if ATTEMPT_RE.match(path.name))
    return f"attempt-{latest + 1:03d}"


def latest_attempt_dir(group_root: Path) -> Path:
    attempts = list_attempt_dirs(group_root)
    if not attempts:
        raise FileNotFoundError(f"No attempt directories exist under {group_root}")
    return attempts[-1]
