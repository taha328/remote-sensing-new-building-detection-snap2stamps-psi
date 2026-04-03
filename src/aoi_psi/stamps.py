from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import logging
import math
import os
import signal
import shutil
import subprocess
import time
from typing import Callable, TextIO

from aoi_psi.artifact_lifecycle import CleanupRecord, CleanupWarning, delete_paths, is_valid_stamps_outputs
from aoi_psi.config import OrbitStackConfig, PipelineConfig
from aoi_psi.manifests import StackManifest, stamps_stack_dir
from aoi_psi.run_context import RunContext

LOGGER = logging.getLogger(__name__)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_SNAPHU_BINARY = _PROJECT_ROOT / ".tools" / "snaphu" / "bin" / "snaphu"
_LOCAL_TRIANGLE_BINARY = _PROJECT_ROOT / ".tools" / "triangle" / "bin" / "triangle"

_MERGE_STAGE_ROOT_ARTIFACTS = (
    "ps2.mat",
    "ph2.mat",
    "rc2.mat",
    "pm2.mat",
    "bp2.mat",
    "la2.mat",
    "inc2.mat",
    "hgt2.mat",
    "phuw2.mat",
    "scla2.mat",
    "scla_sb2.mat",
    "scn2.mat",
    "ifgstd2.mat",
    "patch.list_old",
)
_MERGE_READY_PATCH_FILES = ("ps2.mat", "pm2.mat", "rc2.mat", "bp2.mat", "patch_noover.in", "no_ps_info.mat")
_STEP6_READY_ROOT_FILES = (
    "ps2.mat",
    "ph2.mat",
    "rc2.mat",
    "pm2.mat",
    "bp2.mat",
    "la2.mat",
    "inc2.mat",
    "hgt2.mat",
    "ifgstd2.mat",
    "phuw2.mat",
    "uw_grid.mat",
    "uw_interp.mat",
    "uw_space_time.mat",
)
_STEP7_READY_ROOT_FILES = (
    *_STEP6_READY_ROOT_FILES,
    "scla2.mat",
    "scla_smooth2.mat",
)
_STEP8_READY_ROOT_FILES = (
    *_STEP7_READY_ROOT_FILES,
    "scn2.mat",
)
_LATE_STAGE_ROOT_ARTIFACTS = (
    "scla2.mat",
    "scla_sb2.mat",
    "scla_smooth2.mat",
    "scla_smooth_sb2.mat",
    "scn2.mat",
    "aps2.mat",
    "aps_sb2.mat",
    "tca2.mat",
    "tca_sb2.mat",
)
_STEP8_ROOT_ARTIFACTS = (
    "scn2.mat",
    "aps2.mat",
    "aps_sb2.mat",
    "tca2.mat",
    "tca_sb2.mat",
    "scnfilt.1.node",
    "scnfilt.2.edge",
    "scnfilt.2.node",
    "scnfilt.2.ele",
    "scnfilt.2.poly",
    "scnfilt.2.neigh",
    "triangle_scn.log",
)
_RAW_EXPORT_FIELDS = (
    "point_id",
    "lon",
    "lat",
    "temporal_coherence",
    "azimuth_index",
    "range_index",
)


@dataclass(frozen=True)
class StampsOutputs:
    stack_id: str
    root: Path
    export_dir: Path
    ps_points_csv: Path
    ps_timeseries_csv: Path
    cleanup_records: tuple[CleanupRecord, ...] = ()
    cleanup_warnings: tuple[CleanupWarning, ...] = ()


@dataclass
class _PatchWorkerProcess:
    index: int
    patch_list_path: Path
    command: list[str]
    process: subprocess.Popen[bytes] | subprocess.Popen[str]
    stream: TextIO | None = None


class StaMPSRunner:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    @staticmethod
    def _extend_cleanup_records(
        cleanup_records: list[CleanupRecord],
        new_records: list[CleanupRecord],
        cleanup_observer: Callable[[tuple[CleanupRecord, ...]], None] | None,
    ) -> None:
        if not new_records:
            return
        cleanup_records.extend(new_records)
        if cleanup_observer is not None:
            cleanup_observer(tuple(new_records))

    @property
    def _bin_dir(self) -> Path:
        return self.config.stamps.install_root / "bin"

    @property
    def _matlab_dir(self) -> Path:
        return self.config.stamps.install_root / "matlab"

    @property
    def _matlab_helpers_dir(self) -> Path:
        if self.config.stamps.export_script is None:
            raise ValueError("stamps.export_script must point to a MATLAB/Octave export script.")
        return self.config.stamps.export_script.resolve().parent.parent / "matlab_helpers"

    @property
    def _export_script_path(self) -> Path:
        if self.config.stamps.export_script is None:
            raise ValueError("stamps.export_script must point to a MATLAB/Octave export script.")
        return self.config.stamps.export_script.resolve()

    @property
    def _mt_prep_snap(self) -> Path:
        return self._bin_dir / "mt_prep_snap"

    @staticmethod
    def _resolve_binary(command_name: str, local_binary: Path) -> Path | None:
        candidates: list[Path] = []
        if local_binary.exists() and os.access(local_binary, os.X_OK):
            candidates.append(local_binary)
        resolved = shutil.which(command_name)
        if resolved is not None:
            resolved_path = Path(resolved)
            if resolved_path.exists() and os.access(resolved_path, os.X_OK):
                candidates.append(resolved_path)
        for candidate in candidates:
            try:
                return candidate.resolve()
            except OSError:
                continue
        return None

    def _resolve_snaphu_binary(self) -> Path | None:
        return self._resolve_binary("snaphu", _LOCAL_SNAPHU_BINARY)

    def _resolve_triangle_binary(self) -> Path | None:
        return self._resolve_binary("triangle", _LOCAL_TRIANGLE_BINARY)

    def _interpreter_command(self) -> str:
        if self.config.stamps.use_octave:
            if not self.config.stamps.octave_command:
                raise ValueError("stamps.octave_command is required when use_octave=true")
            return self.config.stamps.octave_command
        if not self.config.stamps.matlab_command:
            raise ValueError("stamps.matlab_command is required when use_octave=false")
        return self.config.stamps.matlab_command

    def validate_environment(self) -> None:
        if shutil.which(self._interpreter_command()) is None:
            raise FileNotFoundError(f"StaMPS interpreter not found: {self._interpreter_command()}")
        if not self.config.stamps.install_root.exists():
            raise FileNotFoundError(f"StaMPS install_root does not exist: {self.config.stamps.install_root}")
        if not self._mt_prep_snap.exists():
            raise FileNotFoundError(f"StaMPS mt_prep_snap not found: {self._mt_prep_snap}")
        if not self._matlab_dir.exists():
            raise FileNotFoundError(f"StaMPS matlab directory not found: {self._matlab_dir}")
        if self.config.stamps.export_script is None:
            raise ValueError("stamps.export_script must point to a MATLAB/Octave export script.")
        if not self.config.stamps.export_script.exists():
            raise FileNotFoundError(f"StaMPS export script not found: {self.config.stamps.export_script}")
        snaphu_binary = self._resolve_snaphu_binary()
        if snaphu_binary is None:
            raise FileNotFoundError(
                "SNAPHU binary not found. Install it on PATH or provision it at "
                f"{_LOCAL_SNAPHU_BINARY}"
            )
        LOGGER.info("Using SNAPHU binary | path=%s", snaphu_binary)
        triangle_binary = self._resolve_triangle_binary()
        if triangle_binary is None:
            raise FileNotFoundError(
                "Triangle binary not found. Install it on PATH or provision it at "
                f"{_LOCAL_TRIANGLE_BINARY}"
            )
        LOGGER.info("Using Triangle binary | path=%s", triangle_binary)

    def _environment(self) -> dict[str, str]:
        env = os.environ.copy()
        env["STAMPS"] = str(self.config.stamps.install_root)
        path_entries = [str(self._bin_dir)]
        snaphu_binary = self._resolve_snaphu_binary()
        if snaphu_binary is not None:
            path_entries.append(str(snaphu_binary.parent))
            env["SNAPHU_BIN"] = str(snaphu_binary)
        triangle_binary = self._resolve_triangle_binary()
        if triangle_binary is not None:
            path_entries.append(str(triangle_binary.parent))
            env["TRIANGLE_BIN"] = str(triangle_binary)
        interpreter_location = shutil.which(self._interpreter_command())
        if interpreter_location is not None:
            path_entries.append(str(Path(interpreter_location).resolve().parent))
        current_path = env.get("PATH")
        if current_path:
            path_entries.append(current_path)
        env["PATH"] = os.pathsep.join(path_entries)
        return env

    @staticmethod
    def _patch_names(root: Path) -> list[str]:
        patch_list = root / "patch.list"
        if not patch_list.exists():
            return []
        return [line.strip() for line in patch_list.read_text(encoding="utf-8").splitlines() if line.strip()]

    def _validate_mt_prep_outputs(self, root: Path) -> None:
        patch_names = self._patch_names(root)
        if not patch_names:
            raise RuntimeError(f"StaMPS mt_prep_snap did not create any patch entries under {root}")

        required_files = ("pscands.1.ij", "pscands.1.ll", "pscands.1.hgt", "pscands.1.ph")
        missing: list[str] = []
        for patch_name in patch_names:
            patch_dir = root / patch_name
            for filename in required_files:
                path = patch_dir / filename
                if not path.exists() or path.stat().st_size == 0:
                    missing.append(str(path))
        if missing:
            raise RuntimeError(
                "StaMPS mt_prep_snap did not produce the candidate files required by stamps(1,8): "
                + ", ".join(missing)
            )

    def describe_execution_plan(
        self,
        stack: OrbitStackConfig | None = None,
        *,
        patch_names: list[str] | None = None,
    ) -> dict[str, int | str | bool]:
        total_patches = len(patch_names) if patch_names is not None else self.config.stamps.range_patches * self.config.stamps.azimuth_patches
        total_patches = max(total_patches, 1)
        requested_parallel_patch_workers = self.config.stamps.max_parallel_patch_workers
        effective_parallel_patch_workers = min(requested_parallel_patch_workers, total_patches)
        parallel_patch_phase = effective_parallel_patch_workers > 1 and total_patches > 1
        patch_batch_count = effective_parallel_patch_workers if parallel_patch_phase else total_patches
        plan: dict[str, int | str | bool] = {
            "range_patches": self.config.stamps.range_patches,
            "azimuth_patches": self.config.stamps.azimuth_patches,
            "planned_total_patches": self.config.stamps.range_patches * self.config.stamps.azimuth_patches,
            "observed_total_patches": total_patches if patch_names is not None else 0,
            "patch_batch_count": patch_batch_count,
            "requested_parallel_patch_workers": requested_parallel_patch_workers,
            "effective_parallel_patch_workers": effective_parallel_patch_workers,
            "requested_patch_workers": requested_parallel_patch_workers,
            "effective_patch_workers": effective_parallel_patch_workers,
            "parallel_patch_phase_enabled": parallel_patch_phase,
            "serial_patch_batch_execution": not parallel_patch_phase,
            "parallel_patch_steps_end": 5,
            "serial_steps_start": 5,
            "merge_resample_size": self.config.stamps.merge_resample_size,
            "snaphu_path": str(self._resolve_snaphu_binary() or ""),
            "triangle_path": str(self._resolve_triangle_binary() or ""),
            "mode": "parallel_patch_batches_then_serial_merge" if parallel_patch_phase else "serial_patch_batches_then_serial_merge",
        }
        if stack is not None:
            plan["stack_id"] = stack.id
        return plan

    def _run_mt_prep_snap(self, master_date: str, datadir: Path, cwd: Path) -> None:
        command = [
            str(self._mt_prep_snap),
            master_date.replace("-", ""),
            str(datadir),
            str(self.config.stamps.amplitude_dispersion_threshold),
            str(self.config.stamps.range_patches),
            str(self.config.stamps.azimuth_patches),
            str(self.config.stamps.range_overlap),
            str(self.config.stamps.azimuth_overlap),
        ]
        if self.config.stamps.mask_file is not None:
            command.append(str(self.config.stamps.mask_file))
        LOGGER.info("Running mt_prep_snap | cwd=%s datadir=%s", cwd, datadir)
        subprocess.run(command, cwd=cwd, env=self._environment(), check=True)
        self._validate_mt_prep_outputs(cwd)

    def _configure_merge_stage_parameters(self, root: Path) -> None:
        if self.config.stamps.merge_resample_size == 0:
            LOGGER.info("Keeping StaMPS merge_resample_size at upstream default | root=%s", root)
            return
        parms_path = root / "parms.mat"
        if not parms_path.exists():
            raise FileNotFoundError(f"StaMPS parms.mat is required before configuring merge-stage parameters: {parms_path}")
        LOGGER.info(
            "Configuring StaMPS merge-stage parameters | root=%s merge_resample_size=%s",
            root,
            self.config.stamps.merge_resample_size,
        )
        script = (
            f"{self._matlab_startup_script()}"
            f"cd('{root.as_posix()}');"
            f"setparm('merge_resample_size',{self.config.stamps.merge_resample_size},1);"
        )
        self._run_matlab_batch(script, cwd=root, log_name="matlab_setparm.log")

    def _matlab_batch_command(self, script: str, *, cwd: Path, log_path: Path) -> list[str]:
        command_name = self._interpreter_command()
        if self.config.stamps.use_octave:
            return [command_name, "--quiet", "--eval", script]
        return [command_name, "-logfile", str(log_path), "-batch", script]

    def _matlab_startup_script(self) -> str:
        helper_dir = self._matlab_helpers_dir
        startup = [f"addpath('{self._matlab_dir.as_posix()}');"]
        if helper_dir.exists():
            startup.append(
                f"if (isempty(which('gausswin')) || isempty(which('interp')) || isempty(which('nanmean'))) && exist('{helper_dir.as_posix()}','dir');"
                f"addpath('{helper_dir.as_posix()}');"
                "end;"
            )
        return "".join(startup)

    def _export_script_invocation(self) -> str:
        export_script_path = self._export_script_path
        return (
            f"addpath('{export_script_path.parent.as_posix()}');"
            f"feval('{export_script_path.stem}');"
        )

    def _run_matlab_batch(self, script: str, cwd: Path, *, log_name: str = "matlab_batch.log") -> None:
        log_path = cwd / log_name
        command = self._matlab_batch_command(script, cwd=cwd, log_path=log_path)
        LOGGER.info("Running StaMPS batch | cwd=%s", cwd)
        if self.config.stamps.use_octave:
            with log_path.open("w", encoding="utf-8") as stream:
                subprocess.run(
                    command,
                    cwd=cwd,
                    env=self._environment(),
                    check=True,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                )
            return
        subprocess.run(command, cwd=cwd, env=self._environment(), check=True)

    def _write_patch_worker_lists(self, root: Path, patch_names: list[str], split_count: int) -> list[Path]:
        if not patch_names:
            return []
        if split_count < 1:
            raise ValueError("StaMPS patch split count must be at least 1.")
        parms_path = root / "parms.mat"
        if not parms_path.exists():
            raise FileNotFoundError(f"StaMPS parms.mat is required before launching patch workers: {parms_path}")
        patches_per_worker = math.ceil(len(patch_names) / split_count)
        patch_list_paths: list[Path] = []
        for index, start in enumerate(range(0, len(patch_names), patches_per_worker), start=1):
            assigned_patches = patch_names[start : start + patches_per_worker]
            if not assigned_patches:
                continue
            patch_list_path = root / f"patch_list_split_{index}"
            patch_list_path.write_text("\n".join(assigned_patches) + "\n", encoding="utf-8")
            for patch_name in assigned_patches:
                patch_dir = root / patch_name
                if not patch_dir.exists():
                    raise FileNotFoundError(f"StaMPS patch directory referenced by patch.list does not exist: {patch_dir}")
                shutil.copy2(parms_path, patch_dir / "parms.mat")
            patch_list_paths.append(patch_list_path)
        return patch_list_paths

    @staticmethod
    def _patch_step5_completed(patch_dir: Path) -> bool:
        log_path = patch_dir / "STAMPS.log"
        if not log_path.exists():
            return False
        try:
            log_text = log_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return False
        return "PS_CORRECT_PHASE Finished" in log_text or "No PS left in step 4, so will skip step 5" in log_text

    @staticmethod
    def _root_log_contains(root: Path, marker: str) -> bool:
        log_path = root / "STAMPS.log"
        if not log_path.exists():
            return False
        try:
            log_text = log_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return False
        return marker in log_text

    def _has_merge_ready_patch_workspace(self, root: Path) -> bool:
        parms_path = root / "parms.mat"
        if not parms_path.exists():
            return False
        patch_names = self._patch_names(root)
        if not patch_names:
            return False
        for patch_name in patch_names:
            patch_dir = root / patch_name
            if not patch_dir.is_dir():
                return False
            for filename in _MERGE_READY_PATCH_FILES:
                artifact = patch_dir / filename
                if not artifact.exists() or artifact.stat().st_size == 0:
                    return False
            if not self._patch_step5_completed(patch_dir):
                return False
        return True

    def _has_step6_ready_workspace(self, root: Path) -> bool:
        if not (root / "parms.mat").exists():
            return False
        if not self._patch_names(root):
            return False
        for filename in _STEP6_READY_ROOT_FILES:
            artifact = root / filename
            if not artifact.exists() or artifact.stat().st_size == 0:
                return False
        return self._root_log_contains(root, "PS_UNWRAP        Finished")

    def _has_step7_ready_workspace(self, root: Path) -> bool:
        if not (root / "parms.mat").exists():
            return False
        if not self._patch_names(root):
            return False
        for filename in _STEP7_READY_ROOT_FILES:
            artifact = root / filename
            if not artifact.exists() or artifact.stat().st_size == 0:
                return False
        return self._root_log_contains(root, "PS_SMOOTH_SCLA   Finished")

    def _has_step8_complete_workspace(self, root: Path) -> bool:
        if not (root / "parms.mat").exists():
            return False
        if not self._patch_names(root):
            return False
        for filename in _STEP8_READY_ROOT_FILES:
            artifact = root / filename
            if not artifact.exists() or artifact.stat().st_size == 0:
                return False
        return self._root_log_contains(root, "PS_SCN_FILT      Finished")

    def _cleanup_partial_merge_stage_for_rerun(self, root: Path, *, export_dir: Path) -> list[CleanupRecord]:
        targets = [path for name in _MERGE_STAGE_ROOT_ARTIFACTS if (path := root / name).exists()]
        if export_dir.exists():
            targets.append(export_dir)
        if not targets:
            return []
        return delete_paths(
            targets,
            category="stamps_merge_stage",
            checkpoint="stamps_merge_rerun_preflight",
            reason="Failed merged-stage StaMPS artifacts must be cleared before rerunning the global merge on the same attempt.",
            logger=LOGGER,
        )

    def _cleanup_partial_late_stage_for_rerun(self, root: Path, *, export_dir: Path) -> list[CleanupRecord]:
        targets = [path for name in _LATE_STAGE_ROOT_ARTIFACTS if (path := root / name).exists()]
        if export_dir.exists():
            targets.append(export_dir)
        if not targets:
            return []
        return delete_paths(
            targets,
            category="stamps_late_stage",
            checkpoint="stamps_late_stage_rerun_preflight",
            reason="Failed late-stage StaMPS artifacts must be cleared before rerunning Steps 7-8 on the same attempt.",
            logger=LOGGER,
        )

    def _cleanup_partial_step8_for_rerun(self, root: Path, *, export_dir: Path) -> list[CleanupRecord]:
        targets = [path for name in _STEP8_ROOT_ARTIFACTS if (path := root / name).exists()]
        if export_dir.exists():
            targets.append(export_dir)
        if not targets:
            return []
        return delete_paths(
            targets,
            category="stamps_step8_stage",
            checkpoint="stamps_step8_rerun_preflight",
            reason="Failed Step 8 StaMPS artifacts must be cleared before rerunning Step 8 on the same attempt.",
            logger=LOGGER,
        )

    def _cleanup_partial_export_for_rerun(self, root: Path, *, export_dir: Path) -> list[CleanupRecord]:
        targets = [export_dir] if export_dir.exists() else []
        if not targets:
            return []
        return delete_paths(
            targets,
            category="stamps_export_tail",
            checkpoint="stamps_export_rerun_preflight",
            reason="Failed export-script outputs must be cleared before rerunning the export tail on the same attempt.",
            logger=LOGGER,
        )

    @staticmethod
    def _csv_header(path: Path) -> tuple[str, ...]:
        if not path.exists() or path.stat().st_size == 0:
            return ()
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream)
            header = next(reader, ())
        return tuple(header)

    @staticmethod
    def _csv_has_data_rows(path: Path) -> bool:
        if not path.exists() or path.stat().st_size == 0:
            return False
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream)
            next(reader, None)
            return next(reader, None) is not None

    def _supports_raw_export_contract(self, points_csv: Path) -> bool:
        header = self._csv_header(points_csv)
        if not header:
            return False
        return set(_RAW_EXPORT_FIELDS).issubset(header)

    def _maybe_cleanup_snap_export_after_stamps(
        self,
        *,
        manifest: StackManifest,
        snap_export_dir: Path,
        points_csv: Path,
        timeseries_csv: Path,
        cleanup_records: list[CleanupRecord],
        cleanup_warnings: list[CleanupWarning],
        cleanup_observer: Callable[[tuple[CleanupRecord, ...]], None] | None,
    ) -> None:
        if not (self.config.artifact_lifecycle.enabled and self.config.artifact_lifecycle.purge_snap_export_after_stamps):
            return
        if self.config.psi.method == "cdpsi":
            LOGGER.warning(
                "Keeping SNAP StaMPS export because CDPSI may require subset-specific reruns from the same attempt lineage | stack_id=%s snap_export_dir=%s",
                manifest.stack_id,
                snap_export_dir,
            )
            return
        if not self._supports_raw_export_contract(points_csv):
            LOGGER.warning(
                "Keeping SNAP StaMPS export because the raw PSI export contract is incomplete | stack_id=%s snap_export_dir=%s points_csv=%s",
                manifest.stack_id,
                snap_export_dir,
                points_csv,
            )
            return
        self._extend_cleanup_records(
            cleanup_records,
            delete_paths(
                [snap_export_dir],
                category="snap_export",
                checkpoint="stamps_outputs_validated",
                reason="SNAP StaMPS export is no longer needed once reusable StaMPS CSV outputs exist.",
                logger=LOGGER,
                best_effort=True,
                retry_hidden_files=True,
                warning_records=cleanup_warnings,
            ),
            cleanup_observer,
        )

    def _spawn_patch_worker(
        self,
        root: Path,
        *,
        patch_list_path: Path,
        index: int,
        env: dict[str, str],
    ) -> _PatchWorkerProcess:
        script = (
            f"{self._matlab_startup_script()}"
            f"cd('{root.as_posix()}');"
            f"stamps(1,5,[],0,'{patch_list_path.name}',1);"
        )
        log_path = root / f"matlab_patch_worker_{index}.log"
        command = self._matlab_batch_command(script, cwd=root, log_path=log_path)
        LOGGER.info(
            "Running StaMPS patch batch | cwd=%s worker=%s patch_list=%s",
            root,
            index,
            patch_list_path.name,
        )
        if self.config.stamps.use_octave:
            stream = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                command,
                cwd=root,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
        else:
            stream = None
            process = subprocess.Popen(command, cwd=root, env=env, text=True, start_new_session=True)
        return _PatchWorkerProcess(
            index=index,
            patch_list_path=patch_list_path,
            command=command,
            process=process,
            stream=stream,
        )

    @staticmethod
    def _close_patch_worker_stream(worker: _PatchWorkerProcess) -> None:
        if worker.stream is not None and not worker.stream.closed:
            worker.stream.close()

    @staticmethod
    def _close_patch_worker_streams(workers: list[_PatchWorkerProcess]) -> None:
        for worker in workers:
            StaMPSRunner._close_patch_worker_stream(worker)

    @staticmethod
    def _signal_patch_worker(worker: _PatchWorkerProcess, sig: signal.Signals) -> None:
        if worker.process.poll() is not None:
            return
        try:
            os.killpg(worker.process.pid, sig)
        except ProcessLookupError:
            return
        except OSError:
            try:
                worker.process.send_signal(sig)
            except ProcessLookupError:
                return

    def _terminate_patch_workers(
        self,
        workers: list[_PatchWorkerProcess],
        *,
        failed_worker: _PatchWorkerProcess | None = None,
    ) -> None:
        active_workers = [worker for worker in workers if worker.process.poll() is None]
        if not active_workers:
            return
        if failed_worker is not None:
            LOGGER.warning(
                "Stopping sibling StaMPS patch workers after failure | failed_worker=%s failed_patch_list=%s sibling_workers=%s",
                failed_worker.index,
                failed_worker.patch_list_path.name,
                [worker.index for worker in active_workers],
            )
        for worker in active_workers:
            self._signal_patch_worker(worker, signal.SIGTERM)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if all(worker.process.poll() is not None for worker in active_workers):
                return
            time.sleep(0.2)
        for worker in active_workers:
            if worker.process.poll() is None:
                self._signal_patch_worker(worker, signal.SIGKILL)
        for worker in active_workers:
            try:
                worker.process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                LOGGER.warning(
                    "StaMPS patch worker did not exit after forced termination | worker=%s patch_list=%s",
                    worker.index,
                    worker.patch_list_path.name,
                )

    def _run_serial_patch_batches(self, root: Path, patch_list_paths: list[Path]) -> None:
        for index, patch_list_path in enumerate(patch_list_paths, start=1):
            script = (
                f"{self._matlab_startup_script()}"
                f"cd('{root.as_posix()}');"
                f"stamps(1,5,[],0,'{patch_list_path.name}',1);"
            )
            self._run_matlab_batch(script, cwd=root, log_name=f"matlab_patch_worker_{index}.log")

    def _run_parallel_patch_batches(
        self,
        root: Path,
        patch_list_paths: list[Path],
        *,
        max_parallel_workers: int,
    ) -> None:
        env = self._environment()
        if max_parallel_workers < 1:
            raise ValueError("StaMPS max_parallel_workers must be at least 1.")
        pending_batches = list(enumerate(patch_list_paths, start=1))
        active_workers: list[_PatchWorkerProcess] = []
        try:
            while pending_batches or active_workers:
                while pending_batches and len(active_workers) < max_parallel_workers:
                    index, patch_list_path = pending_batches.pop(0)
                    active_workers.append(
                        self._spawn_patch_worker(
                            root,
                            patch_list_path=patch_list_path,
                            index=index,
                            env=env,
                        )
                    )

                progress_made = False
                for worker in list(active_workers):
                    return_code = worker.process.poll()
                    if return_code is None:
                        continue
                    progress_made = True
                    active_workers.remove(worker)
                    self._close_patch_worker_stream(worker)
                    if return_code != 0:
                        self._terminate_patch_workers(active_workers, failed_worker=worker)
                        raise subprocess.CalledProcessError(return_code, worker.command)
                if not progress_made and active_workers:
                    time.sleep(0.2)
        finally:
            self._terminate_patch_workers(active_workers)
            self._close_patch_worker_streams(active_workers)

    def _cleanup_partial_workspace_for_rerun(self, root: Path, *, export_dir: Path) -> list[CleanupRecord]:
        targets = [child for child in sorted(root.iterdir(), key=lambda path: path.name) if child != export_dir]
        if export_dir.exists():
            targets.append(export_dir)
        if not targets:
            return []
        return delete_paths(
            targets,
            category="stamps_workspace",
            checkpoint="stamps_rerun_preflight",
            reason="Failed or stale StaMPS workspace must be cleared before rerunning mt_prep_snap on the same attempt.",
            logger=LOGGER,
        )

    def _has_structurally_reusable_outputs(self, root: Path) -> bool:
        export_dir = root / "export"
        points_csv = export_dir / "ps_points.csv"
        ts_csv = export_dir / "ps_timeseries.csv"
        return is_valid_stamps_outputs(points_csv, ts_csv)

    def _has_reusable_outputs(self, root: Path) -> bool:
        export_dir = root / "export"
        points_csv = export_dir / "ps_points.csv"
        return self._has_structurally_reusable_outputs(root) and self._supports_raw_export_contract(points_csv)

    def has_reusable_outputs(self, context: RunContext, stack_id: str) -> bool:
        if not self.config.cache.reuse_stamps_outputs:
            return False
        return self._has_reusable_outputs(stamps_stack_dir(context, stack_id))

    def run_stack(
        self,
        context: RunContext,
        manifest: StackManifest,
        stack: OrbitStackConfig,
        snap_export_dir: Path,
        *,
        cleanup_observer: Callable[[tuple[CleanupRecord, ...]], None] | None = None,
    ) -> StampsOutputs:
        self.validate_environment()
        root = stamps_stack_dir(context, manifest.stack_id)
        root.mkdir(parents=True, exist_ok=True)
        export_dir = root / "export"
        points_csv = export_dir / "ps_points.csv"
        ts_csv = export_dir / "ps_timeseries.csv"
        cleanup_records: list[CleanupRecord] = []
        cleanup_warnings: list[CleanupWarning] = []

        if self.config.cache.reuse_stamps_outputs and self._has_reusable_outputs(root):
            LOGGER.info("Reusing StaMPS outputs | stack_id=%s root=%s", manifest.stack_id, root)
            self._maybe_cleanup_snap_export_after_stamps(
                manifest=manifest,
                snap_export_dir=snap_export_dir,
                points_csv=points_csv,
                timeseries_csv=ts_csv,
                cleanup_records=cleanup_records,
                cleanup_warnings=cleanup_warnings,
                cleanup_observer=cleanup_observer,
            )
            return StampsOutputs(
                stack_id=manifest.stack_id,
                root=root,
                export_dir=export_dir,
                ps_points_csv=points_csv,
                ps_timeseries_csv=ts_csv,
                cleanup_records=tuple(cleanup_records),
                cleanup_warnings=tuple(cleanup_warnings),
            )

        start_step = 5
        export_only = False
        if self._has_step8_complete_workspace(root):
            self._extend_cleanup_records(
                cleanup_records,
                self._cleanup_partial_export_for_rerun(root, export_dir=export_dir),
                cleanup_observer,
            )
            patch_names = self._patch_names(root)
            execution_plan = self.describe_execution_plan(stack, patch_names=patch_names)
            export_only = True
            LOGGER.info(
                "Reusing Step 8-complete StaMPS workspace for export-only rerun | stack_id=%s root=%s plan=%s",
                manifest.stack_id,
                root,
                execution_plan,
            )
        elif self._has_step7_ready_workspace(root):
            self._extend_cleanup_records(
                cleanup_records,
                self._cleanup_partial_step8_for_rerun(root, export_dir=export_dir),
                cleanup_observer,
            )
            patch_names = self._patch_names(root)
            execution_plan = self.describe_execution_plan(stack, patch_names=patch_names)
            start_step = 8
            LOGGER.info(
                "Reusing Step 7-complete StaMPS workspace for Step 8 rerun | stack_id=%s root=%s plan=%s",
                manifest.stack_id,
                root,
                execution_plan,
            )
        elif self._has_step6_ready_workspace(root):
            self._extend_cleanup_records(
                cleanup_records,
                self._cleanup_partial_late_stage_for_rerun(root, export_dir=export_dir),
                cleanup_observer,
            )
            patch_names = self._patch_names(root)
            execution_plan = self.describe_execution_plan(stack, patch_names=patch_names)
            start_step = 7
            LOGGER.info(
                "Reusing Step 6-complete StaMPS workspace for late-stage rerun | stack_id=%s root=%s plan=%s",
                manifest.stack_id,
                root,
                execution_plan,
            )
        elif self._has_merge_ready_patch_workspace(root):
            self._extend_cleanup_records(
                cleanup_records,
                self._cleanup_partial_merge_stage_for_rerun(root, export_dir=export_dir),
                cleanup_observer,
            )
            self._configure_merge_stage_parameters(root)
            patch_names = self._patch_names(root)
            execution_plan = self.describe_execution_plan(stack, patch_names=patch_names)
            LOGGER.info(
                "Reusing completed StaMPS patch workspace for merged-stage rerun | stack_id=%s root=%s plan=%s",
                manifest.stack_id,
                root,
                execution_plan,
            )
        else:
            self._extend_cleanup_records(
                cleanup_records,
                self._cleanup_partial_workspace_for_rerun(root, export_dir=export_dir),
                cleanup_observer,
            )
            self._run_mt_prep_snap(stack.master_date.isoformat(), snap_export_dir, cwd=root)
            self._configure_merge_stage_parameters(root)
            patch_names = self._patch_names(root)
            execution_plan = self.describe_execution_plan(stack, patch_names=patch_names)
            LOGGER.info("Running StaMPS execution plan | stack_id=%s plan=%s", manifest.stack_id, execution_plan)
            patch_list_paths = self._write_patch_worker_lists(
                root,
                patch_names,
                int(execution_plan["patch_batch_count"]),
            )
            if execution_plan["parallel_patch_phase_enabled"]:
                self._run_parallel_patch_batches(
                    root,
                    patch_list_paths,
                    max_parallel_workers=int(execution_plan["effective_parallel_patch_workers"]),
                )
            else:
                self._run_serial_patch_batches(root, patch_list_paths)
        export_dir.mkdir(parents=True, exist_ok=True)
        if export_only:
            script = (
                f"{self._matlab_startup_script()}"
                f"cd('{root.as_posix()}');"
                f"{self._export_script_invocation()}"
            )
        else:
            script = (
                f"{self._matlab_startup_script()}"
                f"cd('{root.as_posix()}');"
                f"stamps({start_step},8,[],0,[],2);"
                f"{self._export_script_invocation()}"
            )
        self._run_matlab_batch(script, cwd=root)

        if not points_csv.exists():
            raise FileNotFoundError(f"StaMPS export did not produce PSI points: {points_csv}")
        if not ts_csv.exists():
            LOGGER.warning("StaMPS export did not produce ps_timeseries.csv; writing an empty placeholder")
            ts_csv.write_text("point_id,epoch,metric_name,value\n", encoding="utf-8")
        if not self._has_structurally_reusable_outputs(root):
            raise RuntimeError(f"StaMPS outputs are not structurally reusable under {root}")
        self._maybe_cleanup_snap_export_after_stamps(
            manifest=manifest,
            snap_export_dir=snap_export_dir,
            points_csv=points_csv,
            timeseries_csv=ts_csv,
            cleanup_records=cleanup_records,
            cleanup_warnings=cleanup_warnings,
            cleanup_observer=cleanup_observer,
        )

        return StampsOutputs(
            stack_id=manifest.stack_id,
            root=root,
            export_dir=export_dir,
            ps_points_csv=points_csv,
            ps_timeseries_csv=ts_csv,
            cleanup_records=tuple(cleanup_records),
            cleanup_warnings=tuple(cleanup_warnings),
        )
