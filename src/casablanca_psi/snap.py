from __future__ import annotations

from datetime import date as calendar_date, datetime
from dataclasses import dataclass
from pathlib import Path
import importlib
import logging
import re
import shutil
import subprocess
from typing import Any, Callable
from xml.etree import ElementTree as ET
import zipfile

from casablanca_psi.artifact_lifecycle import (
    CleanupRecord,
    CleanupWarning,
    are_valid_dimap_products,
    delete_matching_direct_children,
    delete_paths,
    directory_has_files,
    is_valid_snap_export_dir,
)
from casablanca_psi.config import OrbitStackConfig, PipelineConfig
from casablanca_psi.manifests import (
    SlcScene,
    StackManifest,
    secondary_scenes,
    select_master_scene,
    slc_scene_zip_path,
    snap_stack_dir,
)
from casablanca_psi.run_context import RunContext

LOGGER = logging.getLogger(__name__)
_EGM96_ZIP_NAME = "ww15mgh_b.zip"
_S1_ORBIT_FILENAME_PATTERN = re.compile(
    r"^(?P<platform>S1[AB])_OPER_AUX_(?P<orbit_kind>POEORB|RESORB)_OPOD_"
    r"(?P<generated>\d{8}T\d{6})_V(?P<valid_from>\d{8}T\d{6})_(?P<valid_to>\d{8}T\d{6})"
    r"\.EOF(?:\.zip)?$"
)
COREG_STACK_BAND_PATTERN = re.compile(
    r"^(?P<prefix>[iq]_[^_]+)_slv1_(?P<slave_date>\d{2}[A-Za-z]{3}\d{4})_slv(?P<stack_index>\d+)_(?P<master_date>\d{2}[A-Za-z]{3}\d{4})$"
)
IFG_STACK_BAND_PATTERN = re.compile(
    r"^(?P<prefix>[iq]_ifg_[^_]+)_(?P<master_date>\d{2}[A-Za-z]{3}\d{4})_(?P<slave_date>\d{2}[A-Za-z]{3}\d{4})_slv\d+_(?P=master_date)$"
)
FINAL_COREG_MASTER_BAND_PATTERN = re.compile(
    r"^(?P<component>[iq])_(?P<polarization>[^_]+)_mst_(?P<date>\d{2}[A-Za-z]{3}\d{4})$"
)
FINAL_COREG_SLAVE_BAND_PATTERN = re.compile(
    r"^(?P<component>[iq])_(?P<polarization>[^_]+)_slv\d+_(?P<date>\d{2}[A-Za-z]{3}\d{4})$"
)
FINAL_IFG_BAND_PATTERN = re.compile(
    r"^(?P<component>[iq])_ifg_(?P<polarization>[^_]+)_(?P<master_date>\d{2}[A-Za-z]{3}\d{4})_(?P<slave_date>\d{2}[A-Za-z]{3}\d{4})$"
)
METADATA_DATE_PATTERN = re.compile(r"^(?P<day>\d{2})-(?P<month>[A-Z]{3})-(?P<year>\d{4})\b")


class SnapNoIntersectionError(RuntimeError):
    """Raised when a SNAP job does not intersect the configured AOI."""


class SnapEsdNotApplicableError(RuntimeError):
    """Raised when SNAP ESD cannot run for the prepared burst geometry."""


@dataclass(frozen=True)
class SnapStackOutputs:
    stack_id: str
    stack_dir: Path
    prepared_dir: Path
    coreg_dir: Path
    interferogram_dir: Path
    stamps_export_dir: Path
    master_scene_id: str
    coreg_list: Path
    ifg_list: Path
    cleanup_records: tuple[CleanupRecord, ...] = ()
    cleanup_warnings: tuple[CleanupWarning, ...] = ()


@dataclass(frozen=True)
class SnapGraphJob:
    name: str
    graph_path: Path
    parameters: dict[str, str]
    work_dir: Path


class SnapGraphRunner:
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

    def validate_environment(self) -> None:
        executable = shutil.which(self.config.snap.gpt_path)
        if executable is None:
            raise FileNotFoundError(f"SNAP GPT executable not found: {self.config.snap.gpt_path}")
        probe = subprocess.run(
            [executable, "-h"],
            capture_output=True,
            text=True,
            check=False,
        )
        output = f"{probe.stdout}\n{probe.stderr}"
        if "Graph Processing Tool" not in output and "Usage:\n  gpt" not in output and "Usage: gpt" not in output:
            raise RuntimeError(
                f"Configured gpt_path does not resolve to SNAP GPT: {executable}. "
                "On macOS this often means /usr/sbin/gpt was found instead of SNAP."
            )
        if self.config.snap.gpt_vmoptions_path is not None and not self._gpt_vmoptions_path().exists():
            raise FileNotFoundError(
                f"Configured SNAP gpt.vmoptions path does not exist: {self._gpt_vmoptions_path()}"
            )
        if not self.config.dem.path.exists():
            raise FileNotFoundError(f"Configured DEM path does not exist: {self.config.dem.path}")
        user_dir = self._resolved_snap_user_dir()
        if user_dir is not None:
            user_dir.mkdir(parents=True, exist_ok=True)
        if self.config.dem.vertical_datum.strip().upper() == "EGM96":
            egm96_path = self._ensure_egm96_auxdata()
            LOGGER.info("Using SNAP EGM96 auxdata | path=%s", egm96_path)

    def _graph_path(self, graph_name: str) -> Path:
        path = (self.config.snap.graph_root / graph_name).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Required SNAP graph template not found: {path}")
        return path

    def _resolved_gpt_path(self) -> Path:
        configured = Path(self.config.snap.gpt_path).expanduser()
        if configured.is_absolute():
            return configured.resolve()
        executable = shutil.which(self.config.snap.gpt_path)
        if executable is None:
            return configured
        return Path(executable).resolve()

    def _gpt_vmoptions_path(self) -> Path:
        if self.config.snap.gpt_vmoptions_path is not None:
            return self.config.snap.gpt_vmoptions_path.expanduser().resolve()
        gpt_path = self._resolved_gpt_path()
        return gpt_path.with_name(f"{gpt_path.name}.vmoptions")

    @staticmethod
    def _read_vmoptions(path: Path) -> list[str]:
        if not path.exists():
            return []
        return [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]

    @staticmethod
    def _java_option_key(option: str) -> str:
        normalized = option.strip()
        if normalized.startswith("-D"):
            return normalized.split("=", 1)[0]
        for prefix in ("-Xms", "-Xmx", "-Xss"):
            if normalized.startswith(prefix):
                return prefix
        if normalized.startswith("-XX:") and "=" in normalized:
            return normalized.split("=", 1)[0]
        return normalized

    @staticmethod
    def _merge_java_options(installed_options: list[str], runtime_options: list[str]) -> list[str]:
        merged: dict[str, str] = {}
        order: list[str] = []
        for option in [*installed_options, *runtime_options]:
            key = SnapGraphRunner._java_option_key(option)
            if key not in merged:
                order.append(key)
            merged[key] = option
        return [merged[key] for key in order]

    def _resolved_snap_user_dir(self) -> Path | None:
        if self.config.snap.user_dir is None:
            return None
        return self.config.snap.user_dir.expanduser().resolve()

    def _resolved_auxdata_root(self) -> Path | None:
        user_dir = self._resolved_snap_user_dir()
        if user_dir is None:
            return None
        return user_dir / "auxdata"

    def _resolved_orbit_auxdata_root(self) -> Path | None:
        auxdata_root = self._resolved_auxdata_root()
        if auxdata_root is None:
            return None
        return auxdata_root / "Orbits" / "Sentinel-1"

    def _egm96_target_path(self) -> Path | None:
        auxdata_root = self._resolved_auxdata_root()
        if auxdata_root is None:
            return None
        return auxdata_root / "dem" / "egm96" / _EGM96_ZIP_NAME

    def _egm96_candidate_paths(self) -> tuple[Path, ...]:
        candidates: list[Path] = []
        target = self._egm96_target_path()
        if target is not None:
            candidates.append(target)
        home = Path.home()
        for candidate in (
            home / ".snap" / "auxdata" / "dem" / "egm96" / _EGM96_ZIP_NAME,
            home / "Library" / "Application Support" / "SNAP" / "auxdata" / "dem" / "egm96" / _EGM96_ZIP_NAME,
        ):
            if candidate not in candidates:
                candidates.append(candidate)
        return tuple(candidates)

    def _orbit_candidate_roots(self) -> tuple[Path, ...]:
        candidates: list[Path] = []
        target = self._resolved_orbit_auxdata_root()
        if target is not None:
            candidates.append(target)
        home = Path.home()
        for candidate in (
            home / ".snap" / "auxdata" / "Orbits" / "Sentinel-1",
            home / "Library" / "Application Support" / "SNAP" / "auxdata" / "Orbits" / "Sentinel-1",
        ):
            if candidate not in candidates:
                candidates.append(candidate)
        return tuple(candidates)

    @staticmethod
    def _is_valid_egm96_zip(path: Path) -> bool:
        return path.exists() and path.is_file() and zipfile.is_zipfile(path)

    @staticmethod
    def _parse_scene_acquisition_start(scene: SlcScene) -> datetime:
        return datetime.fromisoformat(scene.acquisition_start.replace("Z", "+00:00"))

    @staticmethod
    def _scene_platform_id(scene: SlcScene) -> str:
        normalized = scene.platform.replace("_", "-").upper()
        if normalized.endswith("1A"):
            return "S1A"
        if normalized.endswith("1B"):
            return "S1B"
        return normalized

    @staticmethod
    def _parse_orbit_file_validity(path: Path) -> tuple[str, str, datetime, datetime] | None:
        match = _S1_ORBIT_FILENAME_PATTERN.match(path.name)
        if match is None:
            return None
        return (
            match.group("platform"),
            match.group("orbit_kind"),
            datetime.strptime(match.group("valid_from"), "%Y%m%dT%H%M%S"),
            datetime.strptime(match.group("valid_to"), "%Y%m%dT%H%M%S"),
        )

    @staticmethod
    def _is_orbit_file_covering_scene(path: Path, scene: SlcScene, orbit_kind: str) -> bool:
        parsed = SnapGraphRunner._parse_orbit_file_validity(path)
        if parsed is None:
            return False
        platform, parsed_kind, valid_from, valid_to = parsed
        if platform != SnapGraphRunner._scene_platform_id(scene) or parsed_kind != orbit_kind:
            return False
        acquisition = SnapGraphRunner._parse_scene_acquisition_start(scene).replace(tzinfo=None)
        return valid_from <= acquisition <= valid_to

    def _orbit_target_dir(self, scene: SlcScene, orbit_kind: str) -> Path | None:
        orbit_root = self._resolved_orbit_auxdata_root()
        if orbit_root is None:
            return None
        acquisition = self._parse_scene_acquisition_start(scene)
        return orbit_root / orbit_kind / self._scene_platform_id(scene) / acquisition.strftime("%Y") / acquisition.strftime("%m")

    def _find_local_orbit_file(self, scene: SlcScene, orbit_kind: str) -> Path | None:
        target_dir = self._orbit_target_dir(scene, orbit_kind)
        candidate_roots = self._orbit_candidate_roots()
        search_dirs: list[Path] = []
        if target_dir is not None:
            search_dirs.append(target_dir)
        acquisition = self._parse_scene_acquisition_start(scene)
        year = acquisition.strftime("%Y")
        for root in candidate_roots:
            month_root = root / orbit_kind / self._scene_platform_id(scene) / year
            if not month_root.exists():
                continue
            for month_dir in sorted(path for path in month_root.iterdir() if path.is_dir()):
                if month_dir not in search_dirs:
                    search_dirs.append(month_dir)
        for directory in search_dirs:
            for candidate in sorted(directory.glob("*.EOF*")):
                if self._is_orbit_file_covering_scene(candidate, scene, orbit_kind):
                    return candidate.resolve()
        return None

    def _official_orbit_s3_prefix(self, scene: SlcScene, orbit_kind: str) -> str:
        acquisition = self._parse_scene_acquisition_start(scene)
        return f"Sentinel-1/AUX/AUX_{orbit_kind}/{acquisition:%Y/%m}/"

    def _official_orbit_s3_key(self, scene: SlcScene, orbit_kind: str) -> tuple[str, str] | None:
        acquisition_module = importlib.import_module("casablanca_psi.acquisition")
        s3_client_factory = getattr(acquisition_module, "_s3_client")
        endpoints = (
            self.config.acquisition.s3.endpoint_url,
            *self.config.acquisition.s3.fallback_endpoint_urls,
        )
        bucket = self.config.acquisition.s3.bucket
        prefix = self._official_orbit_s3_prefix(scene, orbit_kind)
        for endpoint in endpoints:
            client = None
            try:
                client = s3_client_factory(self.config, endpoint_url=endpoint)
                paginator = client.get_paginator("list_objects_v2")
                matches: list[str] = []
                for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                    for entry in page.get("Contents", ()):
                        key = entry.get("Key")
                        if not isinstance(key, str):
                            continue
                        if self._is_orbit_file_covering_scene(Path(key), scene, orbit_kind):
                            matches.append(key)
                if matches:
                    matches.sort()
                    return endpoint, matches[0]
            except Exception as exc:
                LOGGER.warning(
                    "Unable to query official Sentinel-1 orbit auxdata | scene=%s orbit_kind=%s endpoint=%s prefix=%s error=%s",
                    scene.scene_id,
                    orbit_kind,
                    endpoint,
                    prefix,
                    exc.__class__.__name__,
                )
            finally:
                close_client = getattr(client, "close", None)
                if callable(close_client):
                    close_client()
        return None

    def _download_official_orbit_auxdata(self, scene: SlcScene, orbit_kind: str, target_dir: Path) -> Path | None:
        official_key = self._official_orbit_s3_key(scene, orbit_kind)
        if official_key is None:
            return None
        endpoint, key = official_key
        target = target_dir / Path(key).name
        if target.exists() and self._is_orbit_file_covering_scene(target, scene, orbit_kind):
            return target.resolve()

        acquisition_module = importlib.import_module("casablanca_psi.acquisition")
        s3_client_factory = getattr(acquisition_module, "_s3_client")
        client = None
        body = None
        temp_path = target.with_suffix(f"{target.suffix}.tmp")
        try:
            client = s3_client_factory(self.config, endpoint_url=endpoint)
            response = client.get_object(Bucket=self.config.acquisition.s3.bucket, Key=key)
            body = response["Body"]
            with temp_path.open("wb") as stream:
                shutil.copyfileobj(body, stream)
            temp_path.replace(target)
            LOGGER.info(
                "Seeded Sentinel-1 orbit auxdata from official CDSE S3 | scene=%s orbit_kind=%s endpoint=%s key=%s path=%s",
                scene.scene_id,
                orbit_kind,
                endpoint,
                key,
                target,
            )
            return target.resolve()
        finally:
            if body is not None:
                body.close()
            close_client = getattr(client, "close", None)
            if callable(close_client):
                close_client()
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)

    def _ensure_egm96_auxdata(self) -> Path:
        target = self._egm96_target_path()
        if target is None:
            raise FileNotFoundError(
                "SNAP EGM96 auxdata requires snap.userdir so AuxDataPath can be resolved."
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        if self._is_valid_egm96_zip(target):
            return target.resolve()
        for candidate in self._egm96_candidate_paths():
            if candidate == target:
                continue
            if not self._is_valid_egm96_zip(candidate):
                continue
            shutil.copy2(candidate, target)
            return target.resolve()
        raise FileNotFoundError(
            "SNAP EGM96 auxdata is missing. Expected a valid "
            f"{_EGM96_ZIP_NAME} at {target} or in one of: "
            + ", ".join(str(path) for path in self._egm96_candidate_paths() if path != target)
        )

    def _ensure_orbit_auxdata_for_scene(self, scene: SlcScene) -> Path:
        orbit_kind = "POEORB" if self.config.snap.orbit_source == "precise" else "RESORB"
        target_dir = self._orbit_target_dir(scene, orbit_kind)
        if target_dir is None:
            raise FileNotFoundError(
                "SNAP orbit auxdata requires snap.userdir so AuxDataPath can be resolved."
            )
        target_dir.mkdir(parents=True, exist_ok=True)
        existing = self._find_local_orbit_file(scene, orbit_kind)
        if existing is None:
            existing = self._download_official_orbit_auxdata(scene, orbit_kind, target_dir)
        if existing is None:
            raise FileNotFoundError(
                f"No local Sentinel-1 {orbit_kind} orbit file covers {scene.acquisition_start} for {scene.platform}. "
                f"Expected it under {target_dir} or one of: "
                + ", ".join(str(root) for root in self._orbit_candidate_roots() if root != self._resolved_orbit_auxdata_root())
                + f". Also checked official CDSE AUX_{orbit_kind} S3 path prefix "
                + self._official_orbit_s3_prefix(scene, orbit_kind)
            )
        if existing.parent != target_dir:
            target = target_dir / existing.name
            if not target.exists():
                shutil.copy2(existing, target)
            return target.resolve()
        return existing

    def _runtime_property_java_options(self) -> list[str]:
        properties = [f"-Dsnap.parallelism={self.config.snap.workers}"]
        user_dir = self._resolved_snap_user_dir()
        if user_dir is not None:
            properties.append(f"-Dsnap.userdir={user_dir}")
        auxdata_root = self._resolved_auxdata_root()
        if auxdata_root is not None:
            properties.append(f"-DAuxDataPath={auxdata_root}")
        if self.config.snap.default_tile_size_px is not None:
            properties.append(f"-Dsnap.jai.defaultTileSize={self.config.snap.default_tile_size_px}")
        return properties

    def _effective_java_options(self) -> tuple[list[str], list[str], list[dict[str, str]]]:
        installed_options = self._read_vmoptions(self._gpt_vmoptions_path())
        runtime_options = [*self.config.snap.java_options, *self._runtime_property_java_options()]
        overrides: list[dict[str, str]] = []
        installed_by_key = {self._java_option_key(option): option for option in installed_options}
        for option in runtime_options:
            key = self._java_option_key(option)
            installed = installed_by_key.get(key)
            if installed is not None and installed != option:
                overrides.append({"key": key, "installed": installed, "runtime": option})
        return installed_options, runtime_options, overrides

    def describe_runtime_policy(self) -> dict[str, Any]:
        installed_options, runtime_options, overrides = self._effective_java_options()
        user_dir = self._resolved_snap_user_dir()
        auxdata_root = self._resolved_auxdata_root()
        egm96_target = self._egm96_target_path()
        return {
            "gpt_path": str(self._resolved_gpt_path()),
            "gpt_vmoptions_path": str(self._gpt_vmoptions_path()),
            "gpt_vmoptions_exists": self._gpt_vmoptions_path().exists(),
            "installed_vmoptions": installed_options,
            "pipeline_java_options": list(self.config.snap.java_options),
            "runtime_property_java_options": runtime_options[len(self.config.snap.java_options) :],
            "effective_java_options": self._merge_java_options(installed_options, runtime_options),
            "vmoptions_overrides": overrides,
            "cache_size_mb": self.config.snap.cache_size_mb,
            "workers": self.config.snap.workers,
            "clear_tile_cache_after_row": self.config.snap.clear_tile_cache_after_row,
            "snap_user_dir": str(user_dir) if user_dir is not None else None,
            "auxdata_root": str(auxdata_root) if auxdata_root is not None else None,
            "orbit_auxdata_root": str(self._resolved_orbit_auxdata_root()) if self._resolved_orbit_auxdata_root() is not None else None,
            "egm96_auxdata_path": str(egm96_target) if egm96_target is not None else None,
            "egm96_auxdata_exists": self._is_valid_egm96_zip(egm96_target) if egm96_target is not None else False,
            "default_tile_size_px": self.config.snap.default_tile_size_px,
        }

    def _run_graph(self, job: SnapGraphJob) -> None:
        self.validate_environment()
        job.work_dir.mkdir(parents=True, exist_ok=True)
        command = [self.config.snap.gpt_path, str(job.graph_path)]
        installed_options, runtime_options, _overrides = self._effective_java_options()
        user_dir = self._resolved_snap_user_dir()
        if user_dir is not None:
            user_dir.mkdir(parents=True, exist_ok=True)
        for option in runtime_options:
            command.append(f"-J{option}")
        command.extend(["-c", f"{self.config.snap.cache_size_mb}M"])
        command.extend(["-q", str(self.config.snap.workers)])
        if self.config.snap.clear_tile_cache_after_row:
            command.append("-x")
        for key, value in sorted(job.parameters.items()):
            command.append(f"-P{key}={value}")
        LOGGER.info(
            "Running SNAP graph | job=%s graph=%s gpt_path=%s gpt_vmoptions_path=%s installed_vmoptions=%s effective_java_options=%s cache_mb=%s parallelism=%s clear_tile_cache_after_row=%s snap_user_dir=%s tile_size_px=%s",
            job.name,
            job.graph_path,
            self._resolved_gpt_path(),
            self._gpt_vmoptions_path(),
            ",".join(installed_options) or "<none>",
            ",".join(self._merge_java_options(installed_options, runtime_options)),
            self.config.snap.cache_size_mb,
            self.config.snap.workers,
            self.config.snap.clear_tile_cache_after_row,
            user_dir,
            self.config.snap.default_tile_size_px,
        )
        try:
            subprocess.run(command, cwd=job.work_dir, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            self._cleanup_failed_job_outputs(job)
            combined_output = "\n".join(part for part in (exc.stdout, exc.stderr) if part).strip()
            if combined_output:
                LOGGER.error("SNAP graph failed | job=%s output=\n%s", job.name, combined_output)
            if self._is_no_intersection_output(combined_output):
                raise SnapNoIntersectionError(
                    f"SNAP job {job.name!r} does not intersect the configured AOI."
                ) from exc
            if self._is_esd_not_applicable_output(combined_output):
                raise SnapEsdNotApplicableError(
                    f"SNAP job {job.name!r} cannot use Enhanced Spectral Diversity for the prepared burst geometry."
                ) from exc
            raise

    @staticmethod
    def _is_no_intersection_output(output: str) -> bool:
        if not output:
            return False
        lowered = output.lower()
        return (
            "no intersection with source product boundary" in lowered
            or "must intersect with the image`s bounds" in lowered
            or "must intersect with the image's bounds" in lowered
        )

    @staticmethod
    def _is_esd_not_applicable_output(output: str) -> bool:
        if not output:
            return False
        return "registration window width should not be grater than burst width 0" in output.lower()

    def _cleanup_failed_job_outputs(self, job: SnapGraphJob) -> None:
        for key in ("outputFile", "coregOutputFile", "ifgOutputFile"):
            value = job.parameters.get(key)
            if not value:
                continue
            path = Path(value)
            data_dir = path.with_suffix(".data")
            if path.exists():
                LOGGER.warning("Removing partial SNAP output after failed job | job=%s path=%s", job.name, path)
                path.unlink()
            if data_dir.exists():
                LOGGER.warning("Removing partial SNAP data directory after failed job | job=%s path=%s", job.name, data_dir)
                shutil.rmtree(data_dir, ignore_errors=True)

        output_dir = job.parameters.get("outputDir")
        if output_dir:
            path = Path(output_dir)
            if path.exists():
                LOGGER.warning("Removing partial SNAP export directory after failed job | job=%s path=%s", job.name, path)
                shutil.rmtree(path, ignore_errors=True)
                path.mkdir(parents=True, exist_ok=True)

    def _cleanup_prepared_swath(self, prepared_dir: Path, scenes: list[SlcScene], iw_swath: str) -> None:
        for scene in scenes:
            product = self._prepared_product(prepared_dir, scene, iw_swath)
            data_dir = product.with_suffix(".data")
            if product.exists():
                LOGGER.warning("Removing skipped SNAP prepared product | swath=%s path=%s", iw_swath, product)
                product.unlink()
            if data_dir.exists():
                LOGGER.warning("Removing skipped SNAP prepared data directory | swath=%s path=%s", iw_swath, data_dir)
                shutil.rmtree(data_dir, ignore_errors=True)

    def _cleanup_coreg_swath(self, coreg_dir: Path, ifg_dir: Path, master: SlcScene, secondaries: list[SlcScene], iw_swath: str) -> None:
        for secondary in secondaries:
            coreg_output = self._coreg_product(coreg_dir, master, secondary, iw_swath)
            ifg_output = self._ifg_product(ifg_dir, master, secondary, iw_swath)
            for product in (coreg_output, ifg_output):
                data_dir = product.with_suffix(".data")
                if product.exists():
                    LOGGER.warning("Removing skipped SNAP coreg/interferogram product | swath=%s path=%s", iw_swath, product)
                    product.unlink()
                if data_dir.exists():
                    LOGGER.warning("Removing skipped SNAP coreg/interferogram data directory | swath=%s path=%s", iw_swath, data_dir)
                    shutil.rmtree(data_dir, ignore_errors=True)

    def _stack_dirs(self, context: RunContext, stack_id: str) -> tuple[Path, Path, Path, Path]:
        stack_dir = snap_stack_dir(context, stack_id)
        prepared_dir = stack_dir / "prepared"
        coreg_dir = stack_dir / "coreg"
        ifg_dir = stack_dir / "interferograms"
        export_dir = stack_dir / "stamps_export"
        for path in (prepared_dir, coreg_dir, ifg_dir, export_dir):
            path.mkdir(parents=True, exist_ok=True)
        return stack_dir, prepared_dir, coreg_dir, ifg_dir

    @staticmethod
    def _scene_name(scene: SlcScene) -> str:
        return scene.product_name.replace(".SAFE", "").replace(".zip", "")

    @staticmethod
    def _snap_polarization(scene_or_polarization: SlcScene | str) -> str:
        raw = scene_or_polarization if isinstance(scene_or_polarization, str) else scene_or_polarization.polarization
        normalized = raw.replace(",", "+")
        parts = {part.strip().upper() for part in normalized.split("+") if part.strip()}
        if parts == {"VV", "VH"}:
            return "VV,VH"
        return ",".join(sorted(parts))

    @staticmethod
    def _single_stack_polarization(polarization: str) -> str:
        normalized = polarization.replace(",", "+")
        parts = tuple(dict.fromkeys(part.strip().upper() for part in normalized.split("+") if part.strip()))
        if len(parts) != 1:
            raise RuntimeError(
                "SNAP StaMPS export requires a single configured polarization per stack. "
                f"Got: {polarization!r}"
            )
        return parts[0]

    def _prepared_product(self, prepared_dir: Path, scene: SlcScene, iw_swath: str) -> Path:
        return prepared_dir / f"{scene.acquisition_date}_{self._scene_name(scene)}_{iw_swath}.dim"

    def _coreg_product(self, coreg_dir: Path, master: SlcScene, secondary: SlcScene, iw_swath: str) -> Path:
        return coreg_dir / f"{master.acquisition_date}_{secondary.acquisition_date}_{iw_swath}_coreg.dim"

    def _ifg_product(self, ifg_dir: Path, master: SlcScene, secondary: SlcScene, iw_swath: str) -> Path:
        return ifg_dir / f"{master.acquisition_date}_{secondary.acquisition_date}_{iw_swath}_ifg.dim"

    def _product_list_file(self, stack_dir: Path, name: str, products: list[Path]) -> Path:
        path = stack_dir / name
        path.write_text("\n".join(str(product) for product in products) + "\n", encoding="utf-8")
        return path

    @staticmethod
    def _product_list_parameter(products: list[Path]) -> str:
        return ",".join(str(product) for product in products)

    def _resolve_scene_zip_path(
        self,
        *,
        context: RunContext,
        manifest: StackManifest,
        scene: SlcScene,
    ) -> Path:
        if scene.local_path:
            candidate = Path(scene.local_path).expanduser()
            if candidate.exists() and candidate.is_file() and candidate.stat().st_size > 0:
                return candidate.resolve()

        expected = slc_scene_zip_path(context, manifest.stack_id, scene)
        if expected.exists() and expected.is_file() and expected.stat().st_size > 0:
            return expected.resolve()

        candidates: list[Path] = []
        for candidate in sorted(context.slc_dir.glob(f"**/{scene.product_name}.zip"), key=lambda path: (len(path.parts), str(path))):
            if candidate == expected:
                continue
            try:
                if not candidate.is_file() or candidate.stat().st_size <= 0:
                    continue
            except OSError:
                continue
            candidates.append(candidate.resolve())
        if candidates:
            return candidates[0]

        raise FileNotFoundError(
            "No local SLC ZIP is available for SNAP preprocessing | "
            f"stack_id={manifest.stack_id} scene_id={scene.scene_id} expected={expected}"
        )

    def _prepare_job(
        self,
        *,
        context: RunContext,
        manifest: StackManifest,
        scene: SlcScene,
        polarization: str,
        iw_swath: str,
        output_file: Path,
        aoi_wkt: str,
    ) -> SnapGraphJob:
        return SnapGraphJob(
            name=f"prepare-{scene.scene_id}-{iw_swath}",
            graph_path=self._graph_path("prepare_slc_stack.xml"),
            parameters={
                "inputFile": str(self._resolve_scene_zip_path(context=context, manifest=manifest, scene=scene)),
                "orbitType": "Sentinel Precise (Auto Download)"
                if self.config.snap.orbit_source == "precise"
                else "Sentinel Restituted (Auto Download)",
                "subswath": iw_swath,
                "polarization": self._snap_polarization(polarization),
                "aoiWkt": aoi_wkt,
                "outputFile": str(output_file),
            },
            work_dir=output_file.parent,
        )

    def _coreg_job(
        self,
        *,
        master_file: Path,
        secondary_file: Path,
        iw_swath: str,
        coreg_output: Path,
        ifg_output: Path,
        aoi_wkt: str,
        use_esd: bool = True,
    ) -> SnapGraphJob:
        dem_path = self.config.dem.path.resolve()
        graph_name = "coregister_stack.xml" if use_esd else "coregister_stack_no_esd.xml"
        job_suffix = iw_swath if use_esd else f"{iw_swath}-no-esd"
        return SnapGraphJob(
            name=f"coreg-{master_file.stem}-{secondary_file.stem}-{job_suffix}",
            graph_path=self._graph_path(graph_name),
            parameters={
                "masterFile": str(master_file),
                "secondaryFile": str(secondary_file),
                "demPath": str(dem_path),
                "demNoDataValue": "0.0",
                "demResamplingMethod": self.config.dem.resampling,
                "aoiWkt": aoi_wkt,
                "coregOutputFile": str(coreg_output),
                "ifgOutputFile": str(ifg_output),
            },
            work_dir=coreg_output.parent,
        )

    def _export_job(
        self,
        *,
        stack_dir: Path,
        coreg_product: Path,
        ifg_product: Path,
        aoi_wkt: str,
        output_dir: Path,
    ) -> SnapGraphJob:
        return SnapGraphJob(
            name=f"stamps-export-{stack_dir.name}",
            graph_path=self._graph_path("stamps_export.xml"),
            parameters={
                "coregFile": str(coreg_product),
                "ifgFile": str(ifg_product),
                "outputDir": str(output_dir),
            },
            work_dir=output_dir,
        )

    def _band_merge_product_set_job(
        self,
        *,
        stack_id: str,
        job_name: str,
        products: list[Path],
        output_file: Path,
    ) -> SnapGraphJob:
        return SnapGraphJob(
            name=f"band-merge-product-set-{stack_id}-{job_name}",
            graph_path=self._graph_path("band_merge_product_set.xml"),
            parameters={
                "fileList": self._product_list_parameter(products),
                "outputFile": str(output_file),
            },
            work_dir=output_file.parent,
        )

    def _create_stack_product_job(
        self,
        *,
        stack_id: str,
        job_name: str,
        products: list[Path],
        output_file: Path,
        master_bands: list[str] | None = None,
        source_bands: list[str] | None = None,
    ) -> SnapGraphJob:
        return SnapGraphJob(
            name=f"create-stack-product-{stack_id}-{job_name}",
            graph_path=self._graph_path("create_stack_product.xml"),
            parameters={
                "fileList": self._product_list_parameter(products),
                "masterBands": ",".join(master_bands or ()),
                "sourceBands": ",".join(source_bands or ()),
                "outputFile": str(output_file),
            },
            work_dir=output_file.parent,
        )

    def _merge_product_set_job(
        self,
        *,
        stack_id: str,
        job_name: str,
        products: list[Path],
        output_file: Path,
    ) -> SnapGraphJob:
        return SnapGraphJob(
            name=f"merge-product-set-{stack_id}-{job_name}",
            graph_path=self._graph_path("merge_product_set.xml"),
            parameters={
                "fileList": self._product_list_parameter(products),
                "outputFile": str(output_file),
            },
            work_dir=output_file.parent,
        )

    def _ifg_from_coreg_job(
        self,
        *,
        stack_id: str,
        job_name: str,
        input_file: Path,
        output_file: Path,
    ) -> SnapGraphJob:
        dem_path = self.config.dem.path.resolve()
        return SnapGraphJob(
            name=f"derive-ifg-from-coreg-{stack_id}-{job_name}",
            graph_path=self._graph_path("derive_ifg_from_coreg_stack.xml"),
            parameters={
                "inputFile": str(input_file),
                "demPath": str(dem_path),
                "demNoDataValue": "0.0",
                "outputFile": str(output_file),
            },
            work_dir=output_file.parent,
        )

    def _select_export_bands_job(
        self,
        *,
        stack_id: str,
        job_name: str,
        input_file: Path,
        source_bands: list[str],
        output_file: Path,
    ) -> SnapGraphJob:
        return SnapGraphJob(
            name=f"select-export-bands-{stack_id}-{job_name}",
            graph_path=self._graph_path("select_export_bands.xml"),
            parameters={
                "inputFile": str(input_file),
                "sourceBands": ",".join(source_bands),
                "outputFile": str(output_file),
            },
            work_dir=output_file.parent,
        )

    def _has_reusable_export(self, output_dir: Path) -> bool:
        return is_valid_snap_export_dir(output_dir)

    @staticmethod
    def _valid_dimap_products_in_dir(path: Path) -> list[Path]:
        if not path.exists():
            return []
        return [product for product in sorted(path.glob("*.dim")) if are_valid_dimap_products((product,))]

    def _has_started_export_assembly(self, export_inputs_dir: Path) -> bool:
        if not export_inputs_dir.exists():
            return False
        if self._valid_dimap_products_in_dir(export_inputs_dir / "coreg_pairs_merged"):
            return True
        if self._valid_dimap_products_in_dir(export_inputs_dir / "ifg_pairs_merged"):
            return True
        if self._valid_dimap_products_in_dir(export_inputs_dir / "coreg_pairs_export"):
            return True
        if self._valid_dimap_products_in_dir(export_inputs_dir / "ifg_pairs_export"):
            return True
        if self._valid_dimap_products_in_dir(export_inputs_dir / "coreg_stack_inputs"):
            return True
        if self._valid_dimap_products_in_dir(export_inputs_dir / "ifg_stack_inputs"):
            return True
        if self._valid_dimap_products_in_dir(export_inputs_dir / "final_products"):
            return True
        return False

    def _prepared_swath_products(self, prepared_dir: Path, scenes: list[SlcScene], iw_swath: str) -> list[Path]:
        return [self._prepared_product(prepared_dir, scene, iw_swath) for scene in scenes]

    def _pair_products_complete(self, *, coreg_output: Path, ifg_output: Path) -> bool:
        return are_valid_dimap_products((coreg_output, ifg_output))

    @staticmethod
    def _secondary_date_from_pair_product(product: Path) -> str:
        parts = product.stem.split("_")
        if len(parts) < 4:
            raise ValueError(f"Unexpected pair product name: {product.name}")
        return parts[1]

    @staticmethod
    def _iw_swath_from_pair_product(product: Path) -> str:
        parts = product.stem.split("_")
        if len(parts) < 4:
            raise ValueError(f"Unexpected pair product name: {product.name}")
        return parts[2]

    def _merged_coreg_product(self, merged_coreg_dir: Path, master: SlcScene, secondary: SlcScene) -> Path:
        return merged_coreg_dir / f"{master.acquisition_date}_{secondary.acquisition_date}_coreg_merged.dim"

    def _merged_ifg_product(self, merged_ifg_dir: Path, master: SlcScene, secondary: SlcScene) -> Path:
        return merged_ifg_dir / f"{master.acquisition_date}_{secondary.acquisition_date}_ifg_merged.dim"

    def _export_ready_coreg_product(self, export_coreg_dir: Path, master: SlcScene, secondary: SlcScene) -> Path:
        return export_coreg_dir / f"{master.acquisition_date}_{secondary.acquisition_date}_coreg_export.dim"

    def _export_ready_ifg_product(self, export_ifg_dir: Path, master: SlcScene, secondary: SlcScene) -> Path:
        return export_ifg_dir / f"{master.acquisition_date}_{secondary.acquisition_date}_ifg_export.dim"

    def _coreg_stack_input_product(self, stack_input_dir: Path, master: SlcScene, secondary: SlcScene) -> Path:
        return stack_input_dir / f"{master.acquisition_date}_{secondary.acquisition_date}_coreg_stack_input.dim"

    def _ifg_stack_input_product(self, stack_input_dir: Path, master: SlcScene, secondary: SlcScene) -> Path:
        return stack_input_dir / f"{master.acquisition_date}_{secondary.acquisition_date}_ifg_stack_input.dim"

    def _final_coreg_export_product(self, final_product_dir: Path, master: SlcScene) -> Path:
        return final_product_dir / f"{master.acquisition_date}_coreg_final.dim"

    def _final_ifg_export_product(self, final_product_dir: Path, master: SlcScene) -> Path:
        return final_product_dir / f"{master.acquisition_date}_ifg_final.dim"

    @staticmethod
    def _scene_date_token(acquisition_date: str) -> str:
        return calendar_date.fromisoformat(acquisition_date).strftime("%d%b%Y")

    @staticmethod
    def _product_band_names(product: Path) -> list[str]:
        root = ET.parse(product).getroot()
        band_names: list[str] = []
        for band in root.findall(".//Spectral_Band_Info"):
            name = band.findtext("BAND_NAME")
            if name:
                band_names.append(name)
        return band_names

    def _final_coreg_contract_errors(
        self,
        *,
        product: Path,
        master_scene: SlcScene,
        secondaries: list[SlcScene],
        polarization: str,
    ) -> list[str]:
        expected_master_date = self._scene_date_token(master_scene.acquisition_date)
        expected_slave_dates = {self._scene_date_token(scene.acquisition_date) for scene in secondaries}
        band_names = self._product_band_names(product)
        master_components: set[str] = set()
        slave_components: dict[str, set[str]] = {}
        errors: list[str] = []

        for band_name in band_names:
            if not band_name.startswith(("i_", "q_")):
                errors.append(f"Unexpected non-RSLC band in final coreg product: {band_name}")
                continue

            master_match = FINAL_COREG_MASTER_BAND_PATTERN.match(band_name)
            if master_match is not None:
                if master_match.group("polarization") != polarization:
                    errors.append(f"Unexpected polarization in final coreg master band: {band_name}")
                    continue
                if master_match.group("date") != expected_master_date:
                    errors.append(f"Unexpected master date in final coreg product: {band_name}")
                    continue
                component = master_match.group("component")
                if component in master_components:
                    errors.append(f"Duplicate master component in final coreg product: {band_name}")
                    continue
                master_components.add(component)
                continue

            slave_match = FINAL_COREG_SLAVE_BAND_PATTERN.match(band_name)
            if slave_match is None:
                errors.append(f"Unsupported final coreg band name: {band_name}")
                continue
            if slave_match.group("polarization") != polarization:
                errors.append(f"Unexpected polarization in final coreg slave band: {band_name}")
                continue
            slave_date = slave_match.group("date")
            if slave_date not in expected_slave_dates:
                errors.append(f"Unexpected slave date in final coreg product: {band_name}")
                continue
            component = slave_match.group("component")
            components = slave_components.setdefault(slave_date, set())
            if component in components:
                errors.append(f"Duplicate final coreg slave component for {slave_date}: {band_name}")
                continue
            components.add(component)

        if master_components != {"i", "q"}:
            errors.append(
                "Final coreg product must contain exactly one master RSLC band pair "
                f"for {expected_master_date}; found components={sorted(master_components)}"
            )
        for slave_date in sorted(expected_slave_dates):
            components = slave_components.get(slave_date, set())
            if components != {"i", "q"}:
                errors.append(
                    "Final coreg product must contain exactly one slave RSLC band pair "
                    f"for {slave_date}; found components={sorted(components)}"
                )

        expected_band_count = 2 * (1 + len(expected_slave_dates))
        if len(band_names) != expected_band_count:
            errors.append(
                f"Final coreg product must contain exactly {expected_band_count} RSLC bands; found {len(band_names)}"
            )
        return errors

    def _final_coreg_metadata_contract_errors(
        self,
        *,
        product: Path,
        master_scene: SlcScene,
        secondaries: list[SlcScene],
        polarization: str,
    ) -> list[str]:
        expected_master_date = self._scene_date_token(master_scene.acquisition_date)
        expected_slave_dates = {self._scene_date_token(scene.acquisition_date) for scene in secondaries}
        try:
            root = ET.parse(product).getroot()
        except ET.ParseError as exc:
            return [f"Final coreg product metadata is not parseable XML: {exc}"]

        errors: list[str] = []
        baselines = root.find(".//MDElem[@name='Baselines']")
        if baselines is None:
            return [f"Final coreg product is missing Abstracted_Metadata/Baselines: {product}"]

        reference = baselines.find(f"./MDElem[@name='Ref_{expected_master_date}']")
        if reference is None:
            errors.append(f"Final coreg product is missing Ref_{expected_master_date} baseline metadata.")
        else:
            available_secondary_dates = {
                element.attrib.get("name", "").removeprefix("Secondary_")
                for element in reference.findall("./MDElem")
                if element.attrib.get("name", "").startswith("Secondary_")
            }
            missing_secondary_dates = sorted(expected_slave_dates - available_secondary_dates)
            if missing_secondary_dates:
                errors.append(
                    "Final coreg product baseline metadata is missing secondary dates: "
                    + ", ".join(missing_secondary_dates)
                )

        slave_band_entries = [
            (attribute.text or "")
            for attribute in root.findall(".//MDElem[@name='Slave_Metadata']//MDATTR[@name='Slave_bands']")
        ]
        if not slave_band_entries:
            errors.append("Final coreg product is missing Slave_Metadata/Slave_bands entries.")
        else:
            for slave_date in sorted(expected_slave_dates):
                if not any(slave_date in entry for entry in slave_band_entries):
                    errors.append(
                        f"Final coreg product metadata is missing Slave_bands mapping for {slave_date}."
                    )

        slave_metadata = root.find(".//MDElem[@name='Slave_Metadata']")
        if slave_metadata is None:
            errors.append("Final coreg product is missing Slave_Metadata.")
            return errors

        master_bands_attribute = slave_metadata.find("./MDATTR[@name='Master_bands']")
        expected_master_bands = {
            f"i_{polarization}_mst_{expected_master_date}",
            f"q_{polarization}_mst_{expected_master_date}",
        }
        if master_bands_attribute is None or not (master_bands_attribute.text or "").strip():
            errors.append("Final coreg product is missing Slave_Metadata/Master_bands.")
        else:
            master_bands = set(master_bands_attribute.text.split())
            if master_bands != expected_master_bands:
                errors.append(
                    "Final coreg product has an invalid Slave_Metadata/Master_bands mapping: "
                    f"expected={sorted(expected_master_bands)} actual={sorted(master_bands)}"
                )

        slave_child_dates: set[str] = set()
        for child in slave_metadata.findall("./MDElem"):
            slave_bands_attribute = child.find("./MDATTR[@name='Slave_bands']")
            if slave_bands_attribute is None or not (slave_bands_attribute.text or "").strip():
                errors.append(
                    f"Final coreg product Slave_Metadata child is missing Slave_bands: {child.attrib.get('name')!r}"
                )
                continue
            child_band_names = slave_bands_attribute.text.split()
            child_components: dict[str, str] = {}
            child_slave_dates: set[str] = set()
            for band_name in child_band_names:
                match = FINAL_COREG_SLAVE_BAND_PATTERN.match(band_name)
                if match is None or match.group("polarization") != polarization:
                    errors.append(
                        f"Final coreg product Slave_Metadata child contains an invalid slave band name: {band_name}"
                    )
                    continue
                child_slave_dates.add(match.group("date"))
                child_components[match.group("component")] = band_name
            if len(child_slave_dates) != 1:
                errors.append(
                    "Final coreg product Slave_Metadata child must map exactly one slave date; "
                    f"child={child.attrib.get('name')!r} dates={sorted(child_slave_dates)}"
                )
                continue
            child_slave_date = next(iter(child_slave_dates))
            slave_child_dates.add(child_slave_date)
            if set(child_components) != {"i", "q"}:
                errors.append(
                    "Final coreg product Slave_Metadata child must map exactly one I/Q slave pair; "
                    f"child={child.attrib.get('name')!r} components={sorted(child_components)}"
                )
            metadata_slave_date = self._metadata_date_token(
                child.findtext("./MDATTR[@name='first_line_time']")
            )
            if metadata_slave_date != child_slave_date:
                errors.append(
                    "Final coreg product Slave_Metadata child acquisition date does not match its slave band date: "
                    f"child={child.attrib.get('name')!r} metadata_date={metadata_slave_date} slave_date={child_slave_date}"
                )

        missing_slave_child_dates = sorted(expected_slave_dates - slave_child_dates)
        if missing_slave_child_dates:
            errors.append(
                "Final coreg product Slave_Metadata children are missing expected secondary dates: "
                + ", ".join(missing_slave_child_dates)
            )

        return errors

    def _has_valid_final_coreg_product_contract(
        self,
        *,
        product: Path,
        master_scene: SlcScene,
        secondaries: list[SlcScene],
        polarization: str,
    ) -> bool:
        return not self._final_coreg_contract_errors(
            product=product,
            master_scene=master_scene,
            secondaries=secondaries,
            polarization=polarization,
        ) and not self._final_coreg_metadata_contract_errors(
            product=product,
            master_scene=master_scene,
            secondaries=secondaries,
            polarization=polarization,
        )

    def _validate_final_coreg_product_contract(
        self,
        *,
        product: Path,
        master_scene: SlcScene,
        secondaries: list[SlcScene],
        polarization: str,
    ) -> None:
        errors = self._final_coreg_contract_errors(
            product=product,
            master_scene=master_scene,
            secondaries=secondaries,
            polarization=polarization,
        )
        errors.extend(
            self._final_coreg_metadata_contract_errors(
                product=product,
                master_scene=master_scene,
                secondaries=secondaries,
                polarization=polarization,
            )
        )
        if errors:
            raise RuntimeError(
                "Final SNAP coreg StaMPS export product has an invalid contract: "
                f"{product} | errors={' ; '.join(errors)}"
            )

    @staticmethod
    def _clone_xml_element(element: ET.Element) -> ET.Element:
        return ET.fromstring(ET.tostring(element, encoding="unicode"))

    @staticmethod
    def _metadata_date_token(value: str | None) -> str | None:
        if value is None:
            return None
        match = METADATA_DATE_PATTERN.match(value.strip())
        if match is None:
            return None
        return f"{match.group('day')}{match.group('month').title()}{match.group('year')}"

    @staticmethod
    def _set_mdattr_text(parent: ET.Element, *, name: str, text: str) -> None:
        attribute = parent.find(f"./MDATTR[@name='{name}']")
        if attribute is None:
            attribute = ET.Element("MDATTR", {"name": name, "type": "ascii", "mode": "rw"})
            parent.insert(0, attribute)
        attribute.text = text

    def _final_coreg_band_map_by_date(
        self,
        *,
        product: Path,
        master_scene: SlcScene,
        secondaries: list[SlcScene],
        polarization: str,
    ) -> tuple[list[str], dict[str, list[str]]]:
        master_date = self._scene_date_token(master_scene.acquisition_date)
        expected_slave_dates = [self._scene_date_token(scene.acquisition_date) for scene in secondaries]
        master_bands: dict[str, str] = {}
        slave_bands: dict[str, dict[str, str]] = {date: {} for date in expected_slave_dates}

        for band_name in self._product_band_names(product):
            master_match = FINAL_COREG_MASTER_BAND_PATTERN.match(band_name)
            if master_match is not None:
                if (
                    master_match.group("polarization") == polarization
                    and master_match.group("date") == master_date
                ):
                    master_bands[master_match.group("component")] = band_name
                continue

            slave_match = FINAL_COREG_SLAVE_BAND_PATTERN.match(band_name)
            if slave_match is None or slave_match.group("polarization") != polarization:
                continue
            slave_date = slave_match.group("date")
            if slave_date not in slave_bands:
                continue
            slave_bands[slave_date][slave_match.group("component")] = band_name

        if set(master_bands) != {"i", "q"}:
            raise RuntimeError(
                f"Final coreg product is missing the expected master I/Q band pair: {product}"
            )

        ordered_slave_bands: dict[str, list[str]] = {}
        for slave_date in expected_slave_dates:
            components = slave_bands.get(slave_date, {})
            if set(components) != {"i", "q"}:
                raise RuntimeError(
                    "Final coreg product is missing the expected slave I/Q band pair for "
                    f"{slave_date}: {product}"
                )
            ordered_slave_bands[slave_date] = [components["i"], components["q"]]

        return [master_bands["i"], master_bands["q"]], ordered_slave_bands

    def _merged_final_coreg_baseline_refs(
        self,
        *,
        source_products: list[Path],
        master_scene: SlcScene,
        secondaries: list[SlcScene],
    ) -> list[ET.Element]:
        master_date = self._scene_date_token(master_scene.acquisition_date)
        expected_slave_dates = [self._scene_date_token(scene.acquisition_date) for scene in secondaries]
        master_ref_name = f"Ref_{master_date}"
        master_secondary_elements: dict[str, ET.Element] = {}
        slave_ref_elements: dict[str, ET.Element] = {}

        for source_product in source_products:
            root = ET.parse(source_product).getroot()
            baselines = root.find(".//MDElem[@name='Abstracted_Metadata']/MDElem[@name='Baselines']")
            if baselines is None:
                raise RuntimeError(
                    f"Reusable coreg export checkpoint is missing Abstracted_Metadata/Baselines: {source_product}"
                )
            for reference in baselines.findall("./MDElem"):
                reference_name = reference.attrib.get("name")
                if reference_name is None:
                    continue
                if reference_name == master_ref_name:
                    for secondary in reference.findall("./MDElem"):
                        secondary_name = secondary.attrib.get("name")
                        if secondary_name is None or secondary_name in master_secondary_elements:
                            continue
                        master_secondary_elements[secondary_name] = self._clone_xml_element(secondary)
                    continue
                if not reference_name.startswith("Ref_"):
                    continue
                slave_date = reference_name.removeprefix("Ref_")
                if slave_date not in expected_slave_dates or slave_date in slave_ref_elements:
                    continue
                slave_ref_elements[slave_date] = self._clone_xml_element(reference)

        required_master_secondary_names = [f"Secondary_{master_date}", *[f"Secondary_{date}" for date in expected_slave_dates]]
        missing_master_secondary_names = [
            name for name in required_master_secondary_names if name not in master_secondary_elements
        ]
        if missing_master_secondary_names:
            raise RuntimeError(
                "Final coreg baseline repair could not find all required master-reference baseline entries in the "
                f"reusable pair checkpoints: missing={missing_master_secondary_names}"
            )

        missing_slave_refs = [date for date in expected_slave_dates if date not in slave_ref_elements]
        if missing_slave_refs:
            raise RuntimeError(
                "Final coreg baseline repair could not find all required slave-reference baseline entries in the "
                f"reusable pair checkpoints: missing={missing_slave_refs}"
            )

        merged_master_reference = ET.Element("MDElem", {"name": master_ref_name})
        for secondary_name in required_master_secondary_names:
            merged_master_reference.append(self._clone_xml_element(master_secondary_elements[secondary_name]))

        return [
            merged_master_reference,
            *[slave_ref_elements[slave_date] for slave_date in expected_slave_dates],
        ]

    def _repair_final_coreg_baseline_metadata(
        self,
        *,
        product: Path,
        source_products: list[Path],
        master_scene: SlcScene,
        secondaries: list[SlcScene],
    ) -> None:
        tree = ET.parse(product)
        root = tree.getroot()
        baselines = root.find(".//MDElem[@name='Abstracted_Metadata']/MDElem[@name='Baselines']")
        if baselines is None:
            raise RuntimeError(f"Final coreg product is missing Abstracted_Metadata/Baselines: {product}")

        repaired_references = self._merged_final_coreg_baseline_refs(
            source_products=source_products,
            master_scene=master_scene,
            secondaries=secondaries,
        )
        for child in list(baselines):
            baselines.remove(child)
        for reference in repaired_references:
            baselines.append(reference)
        tree.write(product, encoding="UTF-8")

    def _repair_final_coreg_slave_metadata(
        self,
        *,
        product: Path,
        source_products: list[Path],
        master_scene: SlcScene,
        secondaries: list[SlcScene],
        polarization: str,
    ) -> None:
        tree = ET.parse(product)
        root = tree.getroot()
        slave_metadata = root.find(".//MDElem[@name='Slave_Metadata']")
        if slave_metadata is None:
            raise RuntimeError(f"Final coreg product is missing Slave_Metadata: {product}")

        master_bands, slave_bands_by_date = self._final_coreg_band_map_by_date(
            product=product,
            master_scene=master_scene,
            secondaries=secondaries,
            polarization=polarization,
        )
        self._set_mdattr_text(slave_metadata, name="Master_bands", text=" ".join(master_bands))

        for child in list(slave_metadata.findall("./MDElem")):
            slave_metadata.remove(child)

        for source_product, secondary in zip(source_products, secondaries, strict=True):
            source_root = ET.parse(source_product).getroot()
            source_child = source_root.find(".//MDElem[@name='Slave_Metadata']/MDElem")
            if source_child is None:
                raise RuntimeError(
                    f"Reusable coreg export checkpoint is missing Slave_Metadata child metadata: {source_product}"
                )
            repaired_child = self._clone_xml_element(source_child)
            slave_date = self._scene_date_token(secondary.acquisition_date)
            slave_band_text = " ".join(slave_bands_by_date[slave_date])
            self._set_mdattr_text(repaired_child, name="Slave_bands", text=slave_band_text)

            for band_metadata in repaired_child.findall("./MDElem"):
                band_names_attribute = band_metadata.find("./MDATTR[@name='band_names']")
                if band_names_attribute is not None:
                    band_names_attribute.text = slave_band_text

            slave_metadata.append(repaired_child)

        tree.write(product, encoding="UTF-8")

    def _final_ifg_contract_errors(
        self,
        *,
        product: Path,
        master_scene: SlcScene,
        secondaries: list[SlcScene],
        polarization: str,
    ) -> list[str]:
        expected_master_date = self._scene_date_token(master_scene.acquisition_date)
        expected_slave_dates = {self._scene_date_token(scene.acquisition_date) for scene in secondaries}
        band_names = self._product_band_names(product)
        ancillary_counts = {"elevation": 0, "orthorectifiedLat": 0, "orthorectifiedLon": 0}
        ifg_components: dict[str, set[str]] = {}
        errors: list[str] = []

        for band_name in band_names:
            if band_name in ancillary_counts:
                ancillary_counts[band_name] += 1
                continue
            match = FINAL_IFG_BAND_PATTERN.match(band_name)
            if match is None:
                continue
            if match.group("polarization") != polarization:
                errors.append(f"Unexpected polarization in final interferogram product: {band_name}")
                continue
            if match.group("master_date") != expected_master_date:
                errors.append(f"Unexpected master date in final interferogram product: {band_name}")
                continue
            slave_date = match.group("slave_date")
            if slave_date not in expected_slave_dates:
                errors.append(f"Unexpected slave date in final interferogram product: {band_name}")
                continue
            components = ifg_components.setdefault(slave_date, set())
            component = match.group("component")
            if component in components:
                errors.append(f"Duplicate final interferogram component for {slave_date}: {band_name}")
                continue
            components.add(component)

        for band_name, count in ancillary_counts.items():
            if count != 1:
                errors.append(f"Final interferogram product must contain exactly one {band_name} band; found {count}")
        for slave_date in sorted(expected_slave_dates):
            components = ifg_components.get(slave_date, set())
            if components != {"i", "q"}:
                errors.append(
                    "Final interferogram product must contain exactly one IFG band pair "
                    f"for {slave_date}; found components={sorted(components)}"
                )
        return errors

    def _has_valid_final_ifg_product_contract(
        self,
        *,
        product: Path,
        master_scene: SlcScene,
        secondaries: list[SlcScene],
        polarization: str,
    ) -> bool:
        return not self._final_ifg_contract_errors(
            product=product,
            master_scene=master_scene,
            secondaries=secondaries,
            polarization=polarization,
        )

    def _validate_final_ifg_product_contract(
        self,
        *,
        product: Path,
        master_scene: SlcScene,
        secondaries: list[SlcScene],
        polarization: str,
    ) -> None:
        errors = self._final_ifg_contract_errors(
            product=product,
            master_scene=master_scene,
            secondaries=secondaries,
            polarization=polarization,
        )
        if errors:
            raise RuntimeError(
                "Final SNAP interferogram StaMPS export product has an invalid band contract: "
                f"{product} | errors={' ; '.join(errors)}"
            )

    def _selected_export_band_names(
        self,
        *,
        product: Path,
        polarization: str,
        product_kind: str,
    ) -> list[str]:
        band_names = self._product_band_names(product)
        if product_kind == "coreg":
            selected = [
                name
                for name in band_names
                if (name.startswith("i_") or name.startswith("q_")) and f"_{polarization}_" in name
            ]
        elif product_kind == "ifg":
            selected = [
                name
                for name in band_names
                if (
                    ((name.startswith("i_ifg_") or name.startswith("q_ifg_")) and f"_{polarization}_" in name)
                    or name in {"elevation", "orthorectifiedLat", "orthorectifiedLon"}
                )
            ]
        else:
            raise ValueError(f"Unsupported export product kind: {product_kind}")

        if not selected:
            raise RuntimeError(
                f"Could not derive export band selection for {product_kind} product {product} "
                f"and polarization {polarization}."
            )
        return selected

    def _selected_final_stack_input_band_names(
        self,
        *,
        product: Path,
        polarization: str,
        product_kind: str,
    ) -> list[str]:
        band_names = self._product_band_names(product)
        if product_kind == "coreg":
            selected = [
                name
                for name in band_names
                if (name.startswith("i_") or name.startswith("q_"))
                and f"_{polarization}_" in name
                and "_slv" in name
            ]
        elif product_kind == "ifg":
            selected = [
                name
                for name in band_names
                if (name.startswith("i_ifg_") or name.startswith("q_ifg_")) and f"_{polarization}_" in name
            ]
            if not selected:
                selected = [
                    name
                    for name in band_names
                    if (name.startswith("i_") or name.startswith("q_"))
                    and f"_{polarization}_" in name
                    and "_mst_" not in name
                ]
        else:
            raise ValueError(f"Unsupported final stack input product kind: {product_kind}")

        if not selected:
            raise RuntimeError(
                f"Could not derive final stack input band selection for {product_kind} product {product} "
                f"and polarization {polarization}."
            )
        return selected

    def _selected_final_coreg_stack_band_names(
        self,
        *,
        product: Path,
        master_scene: SlcScene,
        secondaries: list[SlcScene],
        polarization: str,
    ) -> list[str]:
        master_token = self._scene_date_token(master_scene.acquisition_date)
        expected_slave_dates = [self._scene_date_token(scene.acquisition_date) for scene in secondaries]
        band_names = self._product_band_names(product)

        selected_bands: list[str] = []
        master_components: set[str] = set()
        slave_components: dict[str, set[str]] = {date: set() for date in expected_slave_dates}

        for band_name in band_names:
            master_match = FINAL_COREG_MASTER_BAND_PATTERN.match(band_name)
            if master_match is not None:
                if (
                    master_match.group("polarization") == polarization
                    and master_match.group("date") == master_token
                    and master_match.group("component") not in master_components
                ):
                    selected_bands.append(band_name)
                    master_components.add(master_match.group("component"))
                continue

            exact_slave_match = FINAL_COREG_SLAVE_BAND_PATTERN.match(band_name)
            if exact_slave_match is not None:
                slave_date = exact_slave_match.group("date")
                component = exact_slave_match.group("component")
                if (
                    exact_slave_match.group("polarization") == polarization
                    and slave_date in slave_components
                    and component not in slave_components[slave_date]
                ):
                    selected_bands.append(band_name)
                    slave_components[slave_date].add(component)
                continue

            stacked_slave_match = COREG_STACK_BAND_PATTERN.match(band_name)
            if stacked_slave_match is None:
                continue
            prefix_component, prefix_polarization = stacked_slave_match.group("prefix").split("_", 1)
            if prefix_polarization != polarization:
                continue
            slave_date = stacked_slave_match.group("slave_date")
            component = prefix_component
            if slave_date not in slave_components or component in slave_components[slave_date]:
                continue
            selected_bands.append(band_name)
            slave_components[slave_date].add(component)

        missing_components: list[str] = []
        if master_components != {"i", "q"}:
            missing_components.append(f"master={sorted({'i', 'q'} - master_components)}")
        for slave_date in expected_slave_dates:
            components = slave_components.get(slave_date, set())
            if components != {"i", "q"}:
                missing_components.append(f"{slave_date}={sorted({'i', 'q'} - components)}")
        if missing_components:
            raise RuntimeError(
                "Final coreg stack assembly could not identify one master pair and one unique slave pair per date in "
                f"{product}: {'; '.join(missing_components)}"
            )

        return selected_bands

    def _prune_dimap_product_bands(self, *, product: Path, keep_band_names: list[str]) -> None:
        tree = ET.parse(product)
        root = tree.getroot()
        keep_set = set(keep_band_names)
        data_access = root.find("./Data_Access")
        image_interpretation = root.find("./Image_Interpretation")
        raster_dimensions = root.find("./Raster_Dimensions/NBANDS")
        if data_access is None or image_interpretation is None or raster_dimensions is None:
            raise RuntimeError(f"Cannot prune DIMAP product bands because required sections are missing: {product}")

        spectral_infos = image_interpretation.findall("Spectral_Band_Info")
        data_files = data_access.findall("Data_File")
        data_file_by_index: dict[int, ET.Element] = {}
        for data_file in data_files:
            band_index = data_file.findtext("BAND_INDEX")
            if band_index is None:
                continue
            data_file_by_index[int(band_index)] = data_file

        kept_infos: list[ET.Element] = []
        kept_data_files: list[ET.Element] = []
        removed_hdr_paths: list[Path] = []
        for spectral_info in spectral_infos:
            band_name = spectral_info.findtext("BAND_NAME")
            band_index_text = spectral_info.findtext("BAND_INDEX")
            if band_name is None or band_index_text is None:
                continue
            band_index = int(band_index_text)
            data_file = data_file_by_index.get(band_index)
            if band_name in keep_set:
                kept_infos.append(spectral_info)
                if data_file is not None:
                    kept_data_files.append(data_file)
                continue
            if data_file is not None:
                href_element = data_file.find("DATA_FILE_PATH")
                if href_element is not None:
                    href = href_element.attrib.get("href")
                    if href:
                        removed_hdr_paths.append(product.parent / href)

        if len(kept_infos) != len(keep_band_names):
            raise RuntimeError(
                f"Failed to retain the expected number of bands while pruning {product}: "
                f"expected={len(keep_band_names)} kept={len(kept_infos)}"
            )

        for spectral_info in list(image_interpretation.findall("Spectral_Band_Info")):
            image_interpretation.remove(spectral_info)
        for data_file in list(data_access.findall("Data_File")):
            data_access.remove(data_file)

        tie_point_files = [child for child in list(data_access) if child.tag != "Data_File"]
        for child in tie_point_files:
            data_access.remove(child)

        for new_index, spectral_info in enumerate(kept_infos):
            band_index = spectral_info.find("BAND_INDEX")
            if band_index is not None:
                band_index.text = str(new_index)
            image_interpretation.append(spectral_info)
            data_file = kept_data_files[new_index]
            data_file_band_index = data_file.find("BAND_INDEX")
            if data_file_band_index is not None:
                data_file_band_index.text = str(new_index)
            data_access.append(data_file)

        for child in tie_point_files:
            data_access.append(child)

        raster_dimensions.text = str(len(kept_infos))
        tree.write(product, encoding="UTF-8")

        for hdr_path in removed_hdr_paths:
            for candidate in (hdr_path, hdr_path.with_suffix(".img")):
                if candidate.exists():
                    candidate.unlink()

    @staticmethod
    def _normalized_final_stack_band_name(name: str, *, product_kind: str) -> str:
        if product_kind == "coreg":
            match = COREG_STACK_BAND_PATTERN.match(name)
            if match is None:
                return name
            return f"{match.group('prefix')}_slv{match.group('stack_index')}_{match.group('slave_date')}"
        if product_kind == "ifg":
            match = IFG_STACK_BAND_PATTERN.match(name)
            if match is None:
                return name
            return f"{match.group('prefix')}_{match.group('master_date')}_{match.group('slave_date')}"
        raise ValueError(f"Unsupported final stack product kind: {product_kind}")

    @staticmethod
    def _replace_band_name_tokens(text: str, rename_map: dict[str, str]) -> tuple[str, bool]:
        parts = re.split(r"(\s+|,)", text)
        changed = False
        for index, part in enumerate(parts):
            replacement = rename_map.get(part)
            if replacement is None or replacement == part:
                continue
            parts[index] = replacement
            changed = True
        return "".join(parts), changed

    def _normalize_final_stack_product_band_names(
        self,
        *,
        product: Path,
        product_kind: str,
    ) -> None:
        tree = ET.parse(product)
        root = tree.getroot()
        rename_map: dict[str, str] = {}

        for band_name in root.findall(".//Spectral_Band_Info/BAND_NAME"):
            if not band_name.text:
                continue
            normalized = self._normalized_final_stack_band_name(band_name.text, product_kind=product_kind)
            if normalized != band_name.text:
                rename_map[band_name.text] = normalized

        if not rename_map:
            return

        seen_band_names: set[str] = set()
        for band_name in root.findall(".//Spectral_Band_Info/BAND_NAME"):
            if not band_name.text:
                continue
            band_name.text = rename_map.get(band_name.text, band_name.text)
            if band_name.text in seen_band_names:
                raise RuntimeError(
                    f"Normalized SNAP {product_kind} final stack product contains duplicate band name {band_name.text!r}: {product}"
                )
            seen_band_names.add(band_name.text)

        for element in root.iter():
            if element.text:
                replaced_text, changed = self._replace_band_name_tokens(element.text, rename_map)
                if changed:
                    element.text = replaced_text
            if element.tail:
                replaced_tail, changed = self._replace_band_name_tokens(element.tail, rename_map)
                if changed:
                    element.tail = replaced_tail

        tree.write(product, encoding="UTF-8")

    def _pair_product_group_map(self, products: list[Path]) -> dict[str, dict[str, Path]]:
        grouped: dict[str, dict[str, Path]] = {}
        for product in products:
            secondary_date = self._secondary_date_from_pair_product(product)
            iw_swath = self._iw_swath_from_pair_product(product)
            grouped.setdefault(secondary_date, {})[iw_swath] = product
        return grouped

    def _group_pair_products_for_export(
        self,
        *,
        grouped: dict[str, dict[str, Path]],
        secondaries: list[SlcScene],
        iw_swaths: tuple[str, ...] | list[str],
    ) -> list[list[Path] | None]:
        ordered_groups: list[list[Path] | None] = []
        for secondary in secondaries:
            swath_map = grouped.get(secondary.acquisition_date)
            if swath_map is None:
                ordered_groups.append(None)
                continue
            ordered_group = [swath_map[iw_swath] for iw_swath in iw_swaths if iw_swath in swath_map]
            if not ordered_group:
                ordered_groups.append(None)
                continue
            ordered_groups.append(ordered_group)
        return ordered_groups

    def _cleanup_pair_source_products_after_merged_coreg(
        self,
        *,
        coreg_group: list[Path] | None,
        ifg_group: list[Path] | None,
        master: SlcScene,
        secondary: SlcScene,
    ) -> list[CleanupRecord]:
        paths = [path for group in (coreg_group, ifg_group) if group is not None for path in group if path.exists()]
        if not paths:
            return []
        return delete_paths(
            paths,
            category="snap_export_checkpoint",
            checkpoint="merged_coreg_pair_validated",
            reason=(
                f"Pair-level SNAP coreg/interferogram products for {master.acquisition_date}/{secondary.acquisition_date} "
                "are no longer needed once a reusable merged coreg export checkpoint exists."
            ),
            logger=LOGGER,
        )

    def _cleanup_superseded_export_checkpoint(
        self,
        *,
        superseded_product: Path,
        replacement_product: Path,
        description: str,
    ) -> list[CleanupRecord]:
        if not superseded_product.exists() or not are_valid_dimap_products((replacement_product,)):
            return []
        return delete_paths(
            [superseded_product],
            category="snap_export_checkpoint",
            checkpoint="export_ready_pair_validated",
            reason=(
                f"{description} is superseded by a reusable single-polarization export checkpoint: "
                f"{replacement_product.name}"
            ),
            logger=LOGGER,
        )

    def _cleanup_superseded_final_stack_source(
        self,
        *,
        superseded_product: Path,
        replacement_product: Path,
        description: str,
    ) -> list[CleanupRecord]:
        if not superseded_product.exists() or not are_valid_dimap_products((replacement_product,)):
            return []
        return delete_paths(
            [superseded_product],
            category="snap_export_checkpoint",
            checkpoint="final_stack_input_validated",
            reason=(
                f"{description} is superseded by a reusable deduplicated StaMPS stack input checkpoint: "
                f"{replacement_product.name}"
            ),
            logger=LOGGER,
        )

    def _cleanup_superseded_final_export_sources(
        self,
        *,
        superseded_products: list[Path],
        replacement_product: Path,
        description: str,
    ) -> list[CleanupRecord]:
        paths = [path for path in superseded_products if path.exists()]
        if not paths or not are_valid_dimap_products((replacement_product,)):
            return []
        return delete_paths(
            paths,
            category="snap_export_checkpoint",
            checkpoint="final_export_product_validated",
            reason=(
                f"{description} is superseded by a reusable final StaMPS export checkpoint product: "
                f"{replacement_product.name}"
            ),
            logger=LOGGER,
        )

    def _build_final_stamps_export_inputs(
        self,
        *,
        assembly_dir: Path,
        stack_id: str,
        master_scene: SlcScene,
        secondaries: list[SlcScene],
        polarization: str,
        cleanup_records: list[CleanupRecord],
        cleanup_observer: Callable[[tuple[CleanupRecord, ...]], None] | None,
    ) -> tuple[list[Path], list[Path]]:
        export_coreg_dir = assembly_dir / "coreg_pairs_export"
        export_ifg_dir = assembly_dir / "ifg_pairs_export"
        coreg_stack_input_dir = assembly_dir / "coreg_stack_inputs"
        ifg_stack_input_dir = assembly_dir / "ifg_stack_inputs"
        final_product_dir = assembly_dir / "final_products"
        for path in (coreg_stack_input_dir, ifg_stack_input_dir, final_product_dir):
            path.mkdir(parents=True, exist_ok=True)

        final_coreg = self._final_coreg_export_product(final_product_dir, master_scene)
        final_ifg = self._final_ifg_export_product(final_product_dir, master_scene)
        reuse_final_coreg = (
            self.config.cache.reuse_snap_outputs
            and are_valid_dimap_products((final_coreg,))
            and self._has_valid_final_coreg_product_contract(
                product=final_coreg,
                master_scene=master_scene,
                secondaries=secondaries,
                polarization=polarization,
            )
        )
        reuse_final_ifg = (
            self.config.cache.reuse_snap_outputs
            and are_valid_dimap_products((final_ifg,))
            and self._has_valid_final_ifg_product_contract(
                product=final_ifg,
                master_scene=master_scene,
                secondaries=secondaries,
                polarization=polarization,
            )
        )
        if self.config.cache.reuse_snap_outputs and final_coreg.exists() and not reuse_final_coreg:
            self._extend_cleanup_records(
                cleanup_records,
                self._cleanup_invalid_final_export_product(
                    product=final_coreg,
                    description="Final SNAP coreg export checkpoint",
                ),
                cleanup_observer,
            )
        if self.config.cache.reuse_snap_outputs and final_ifg.exists() and not reuse_final_ifg:
            self._extend_cleanup_records(
                cleanup_records,
                self._cleanup_invalid_final_export_product(
                    product=final_ifg,
                    description="Final SNAP interferogram export checkpoint",
                ),
                cleanup_observer,
            )
        need_final_coreg = not reuse_final_coreg
        need_final_ifg = not reuse_final_ifg
        if not need_final_coreg and not need_final_ifg:
            return [final_coreg], [final_ifg]

        final_ifg_sources: list[Path] = []
        coreg_export_products: list[Path] = []
        export_ifg_products: list[Path] = []
        for index, secondary in enumerate(secondaries):
            export_coreg = self._export_ready_coreg_product(export_coreg_dir, master_scene, secondary)
            export_ifg = self._export_ready_ifg_product(export_ifg_dir, master_scene, secondary)
            export_ifg_products.append(export_ifg)
            if need_final_coreg and not are_valid_dimap_products((export_coreg,)):
                raise RuntimeError(
                    f"Missing reusable coreg export checkpoint required to build the final StaMPS stack product: {export_coreg}"
                )
            if need_final_ifg and not are_valid_dimap_products((export_ifg,)):
                raise RuntimeError(
                    f"Missing reusable interferogram export checkpoint required to build the final StaMPS stack product: {export_ifg}"
                )
            if need_final_coreg or reuse_final_coreg:
                coreg_export_products.append(export_coreg)
            if need_final_ifg:
                if index == 0:
                    final_ifg_sources.append(export_ifg)
                else:
                    ifg_stack_input = self._ifg_stack_input_product(ifg_stack_input_dir, master_scene, secondary)
                    if not (self.config.cache.reuse_snap_outputs and are_valid_dimap_products((ifg_stack_input,))):
                        if ifg_stack_input.exists():
                            self._extend_cleanup_records(
                                cleanup_records,
                                self._cleanup_invalid_final_export_product(
                                    product=ifg_stack_input,
                                    description=(
                                        f"Final interferogram stack input checkpoint for {master_scene.acquisition_date}/{secondary.acquisition_date}"
                                    ),
                                ),
                                cleanup_observer,
                            )
                        self._run_graph(
                            self._select_export_bands_job(
                                stack_id=stack_id,
                                job_name=f"{secondary.acquisition_date}-ifg-stack-input",
                                input_file=export_ifg,
                                source_bands=self._selected_final_stack_input_band_names(
                                    product=export_ifg,
                                    polarization=polarization,
                                    product_kind="ifg",
                                ),
                                output_file=ifg_stack_input,
                            )
                        )
                    if not are_valid_dimap_products((ifg_stack_input,)):
                        raise RuntimeError(
                            "SNAP export assembly did not produce a structurally valid final interferogram stack input: "
                            f"{ifg_stack_input}"
                        )
                    if self.config.artifact_lifecycle.enabled:
                        self._extend_cleanup_records(
                            cleanup_records,
                            self._cleanup_superseded_final_stack_source(
                                superseded_product=export_ifg,
                                replacement_product=ifg_stack_input,
                                description=(
                                    "Final SNAP interferogram export checkpoint for "
                                    f"{master_scene.acquisition_date}/{secondary.acquisition_date}"
                                ),
                            ),
                            cleanup_observer,
                        )
                    final_ifg_sources.append(ifg_stack_input)

        if need_final_coreg:
            self._run_graph(
                self._create_stack_product_job(
                    stack_id=stack_id,
                    job_name="coreg-final",
                    products=coreg_export_products,
                    output_file=final_coreg,
                )
            )
            self._prune_dimap_product_bands(
                product=final_coreg,
                keep_band_names=self._selected_final_coreg_stack_band_names(
                    product=final_coreg,
                    master_scene=master_scene,
                    secondaries=secondaries,
                    polarization=polarization,
                ),
            )
            self._normalize_final_stack_product_band_names(product=final_coreg, product_kind="coreg")
            self._repair_final_coreg_baseline_metadata(
                product=final_coreg,
                source_products=coreg_export_products,
                master_scene=master_scene,
                secondaries=secondaries,
            )
            self._repair_final_coreg_slave_metadata(
                product=final_coreg,
                source_products=coreg_export_products,
                master_scene=master_scene,
                secondaries=secondaries,
                polarization=polarization,
            )
            self._validate_final_coreg_product_contract(
                product=final_coreg,
                master_scene=master_scene,
                secondaries=secondaries,
                polarization=polarization,
            )
            if self.config.artifact_lifecycle.enabled:
                self._extend_cleanup_records(
                    cleanup_records,
                    self._cleanup_superseded_final_export_sources(
                        superseded_products=coreg_export_products,
                        replacement_product=final_coreg,
                        description="Final SNAP coreg export assembly source checkpoints",
                    ),
                    cleanup_observer,
                )
        if need_final_coreg and not are_valid_dimap_products((final_coreg,)):
            raise RuntimeError(f"SNAP export assembly did not produce a structurally valid final coreg product: {final_coreg}")
        if (
            reuse_final_coreg
            and self.config.artifact_lifecycle.enabled
            and are_valid_dimap_products((final_coreg,))
        ):
            self._extend_cleanup_records(
                cleanup_records,
                self._cleanup_superseded_final_export_sources(
                    superseded_products=coreg_export_products,
                    replacement_product=final_coreg,
                    description="Final SNAP coreg export assembly source checkpoints",
                ),
                cleanup_observer,
            )

        if need_final_ifg:
            self._run_graph(
                self._band_merge_product_set_job(
                    stack_id=stack_id,
                    job_name="ifg-final",
                    products=final_ifg_sources,
                    output_file=final_ifg,
                )
            )
            self._validate_final_ifg_product_contract(
                product=final_ifg,
                master_scene=master_scene,
                secondaries=secondaries,
                polarization=polarization,
            )
        if need_final_ifg and not are_valid_dimap_products((final_ifg,)):
            raise RuntimeError(f"SNAP export assembly did not produce a structurally valid final interferogram product: {final_ifg}")

        if are_valid_dimap_products((final_ifg,)):
            self._extend_cleanup_records(
                cleanup_records,
                self._cleanup_superseded_final_export_sources(
                    superseded_products=export_ifg_products,
                    replacement_product=final_ifg,
                    description="Final SNAP interferogram export assembly source checkpoints",
                ),
                cleanup_observer,
            )
            self._extend_cleanup_records(
                cleanup_records,
                self._cleanup_superseded_final_export_sources(
                    superseded_products=final_ifg_sources[1:],
                    replacement_product=final_ifg,
                    description="Final SNAP interferogram stack input checkpoints",
                ),
                cleanup_observer,
            )

        return [final_coreg], [final_ifg]

    def _assemble_export_source_products(
        self,
        *,
        stack_dir: Path,
        stack_id: str,
        master_scene: SlcScene,
        secondaries: list[SlcScene],
        iw_swaths: tuple[str, ...] | list[str],
        polarization: str,
        coreg_products: list[Path],
        ifg_products: list[Path],
        cleanup_records: list[CleanupRecord],
        cleanup_observer: Callable[[tuple[CleanupRecord, ...]], None] | None = None,
    ) -> tuple[list[Path], list[Path]]:
        assembly_dir = stack_dir / "stamps_export_inputs"
        merged_coreg_dir = assembly_dir / "coreg_pairs_merged"
        merged_ifg_dir = assembly_dir / "ifg_pairs_merged"
        export_coreg_dir = assembly_dir / "coreg_pairs_export"
        export_ifg_dir = assembly_dir / "ifg_pairs_export"
        for path in (merged_coreg_dir, merged_ifg_dir, export_coreg_dir, export_ifg_dir):
            path.mkdir(parents=True, exist_ok=True)

        coreg_group_map = self._pair_product_group_map(coreg_products)
        ifg_group_map = self._pair_product_group_map(ifg_products)
        coreg_groups = self._group_pair_products_for_export(
            grouped=coreg_group_map,
            secondaries=secondaries,
            iw_swaths=iw_swaths,
        )
        ifg_groups = self._group_pair_products_for_export(
            grouped=ifg_group_map,
            secondaries=secondaries,
            iw_swaths=iw_swaths,
        )

        for secondary, coreg_group, ifg_group in zip(secondaries, coreg_groups, ifg_groups, strict=True):
            merged_coreg = self._merged_coreg_product(merged_coreg_dir, master_scene, secondary)
            merged_ifg = self._merged_ifg_product(merged_ifg_dir, master_scene, secondary)
            export_coreg = self._export_ready_coreg_product(export_coreg_dir, master_scene, secondary)
            export_ifg = self._export_ready_ifg_product(export_ifg_dir, master_scene, secondary)
            reuse_export_coreg = self.config.cache.reuse_snap_outputs and are_valid_dimap_products((export_coreg,))
            reuse_export_ifg = self.config.cache.reuse_snap_outputs and are_valid_dimap_products((export_ifg,))

            if not reuse_export_coreg:
                if not (self.config.cache.reuse_snap_outputs and are_valid_dimap_products((merged_coreg,))):
                    if coreg_group is None:
                        raise RuntimeError(
                            f"Missing SNAP coreg pair products for export assembly | secondary={secondary.acquisition_date}"
                        )
                    self._run_graph(
                        self._merge_product_set_job(
                            stack_id=stack_id,
                            job_name=f"{secondary.acquisition_date}-coreg",
                            products=coreg_group,
                            output_file=merged_coreg,
                        )
                    )
                if not are_valid_dimap_products((merged_coreg,)):
                    raise RuntimeError(
                        f"SNAP export assembly did not produce a structurally valid merged coreg product: {merged_coreg}"
                    )
                self._run_graph(
                    self._select_export_bands_job(
                        stack_id=stack_id,
                        job_name=f"{secondary.acquisition_date}-coreg",
                        input_file=merged_coreg,
                        source_bands=self._selected_export_band_names(
                            product=merged_coreg,
                            polarization=polarization,
                            product_kind="coreg",
                        ),
                        output_file=export_coreg,
                    )
                )
            if not are_valid_dimap_products((export_coreg,)):
                raise RuntimeError(
                    f"SNAP export assembly did not produce a structurally valid export-ready coreg product: {export_coreg}"
                )
            if (
                not reuse_export_coreg
                and
                self.config.artifact_lifecycle.enabled
                and self.config.artifact_lifecycle.purge_pair_products_after_merged_coreg
            ):
                self._extend_cleanup_records(
                    cleanup_records,
                    self._cleanup_pair_source_products_after_merged_coreg(
                        coreg_group=coreg_group,
                        ifg_group=ifg_group,
                        master=master_scene,
                        secondary=secondary,
                    ),
                        cleanup_observer,
                    )
            if self.config.artifact_lifecycle.enabled:
                self._extend_cleanup_records(
                    cleanup_records,
                    self._cleanup_superseded_export_checkpoint(
                        superseded_product=merged_coreg,
                        replacement_product=export_coreg,
                        description=(
                            f"Merged SNAP coreg export checkpoint for {master_scene.acquisition_date}/{secondary.acquisition_date}"
                        ),
                    ),
                        cleanup_observer,
                    )
            if not reuse_export_ifg:
                self._run_graph(
                    self._ifg_from_coreg_job(
                        stack_id=stack_id,
                        job_name=f"{secondary.acquisition_date}-ifg",
                        input_file=export_coreg,
                        output_file=export_ifg,
                    )
                )
            if not are_valid_dimap_products((export_ifg,)):
                raise RuntimeError(
                    f"SNAP export assembly did not produce a structurally valid export-ready interferogram product: {export_ifg}"
                )
            if self.config.artifact_lifecycle.enabled:
                self._extend_cleanup_records(
                    cleanup_records,
                    self._cleanup_superseded_export_checkpoint(
                        superseded_product=merged_ifg,
                        replacement_product=export_ifg,
                        description=(
                            f"Merged SNAP interferogram export checkpoint for {master_scene.acquisition_date}/{secondary.acquisition_date}"
                        ),
                    ),
                    cleanup_observer,
                )

        return self._build_final_stamps_export_inputs(
            assembly_dir=assembly_dir,
            stack_id=stack_id,
            master_scene=master_scene,
            secondaries=secondaries,
            polarization=polarization,
            cleanup_records=cleanup_records,
            cleanup_observer=cleanup_observer,
        )

    def _swath_pair_products_complete(
        self,
        *,
        coreg_dir: Path,
        ifg_dir: Path,
        master: SlcScene,
        secondaries: list[SlcScene],
        iw_swath: str,
    ) -> bool:
        coreg_products = [self._coreg_product(coreg_dir, master, secondary, iw_swath) for secondary in secondaries]
        ifg_products = [self._ifg_product(ifg_dir, master, secondary, iw_swath) for secondary in secondaries]
        return are_valid_dimap_products(coreg_products) and are_valid_dimap_products(ifg_products)

    def _cleanup_prepared_products(
        self,
        *,
        prepared_dir: Path,
        scenes: list[SlcScene],
        iw_swath: str,
    ) -> list[CleanupRecord]:
        return delete_paths(
            [product for product in self._prepared_swath_products(prepared_dir, scenes, iw_swath) if product.exists()],
            category="snap_prepared",
            checkpoint="swath_coreg_products_validated",
            reason=f"Remaining prepared SNAP products for {iw_swath} are no longer needed once all swath pair products exist.",
            logger=LOGGER,
        )

    def _cleanup_secondary_prepared_product(
        self,
        *,
        prepared_dir: Path,
        secondary: SlcScene,
        iw_swath: str,
    ) -> list[CleanupRecord]:
        product = self._prepared_product(prepared_dir, secondary, iw_swath)
        if not product.exists():
            return []
        return delete_paths(
            [product],
            category="snap_prepared",
            checkpoint="pair_products_validated",
            reason=(
                f"Prepared SNAP product for secondary {secondary.acquisition_date} {iw_swath} "
                "is no longer needed once its reusable pair products exist."
            ),
            logger=LOGGER,
        )

    def _cleanup_master_prepared_product(
        self,
        *,
        context: RunContext,
        manifest: StackManifest,
        prepared_dir: Path,
        master: SlcScene,
        iw_swath: str,
    ) -> list[CleanupRecord]:
        product = self._prepared_product(prepared_dir, master, iw_swath)
        if not product.exists():
            return []
        try:
            source_zip = self._resolve_scene_zip_path(context=context, manifest=manifest, scene=master)
        except FileNotFoundError:
            LOGGER.warning(
                "Keeping prepared SNAP master because raw SLC source is unavailable for safe regeneration | stack_id=%s swath=%s master=%s source_zip=%s",
                manifest.stack_id,
                iw_swath,
                master.scene_id,
                slc_scene_zip_path(context, manifest.stack_id, master),
            )
            return []
        return delete_paths(
            [product],
            category="snap_prepared",
            checkpoint="pair_products_validated",
            reason=(
                f"Prepared SNAP master product for {master.acquisition_date} {iw_swath} "
                "will be regenerated on demand from the preserved raw SLC source."
            ),
            logger=LOGGER,
        )

    def _cleanup_secondary_prepared_for_on_demand_regeneration(
        self,
        *,
        context: RunContext,
        manifest: StackManifest,
        prepared_dir: Path,
        secondary: SlcScene,
        iw_swath: str,
    ) -> list[CleanupRecord]:
        product = self._prepared_product(prepared_dir, secondary, iw_swath)
        if not product.exists():
            return []
        try:
            source_zip = self._resolve_scene_zip_path(context=context, manifest=manifest, scene=secondary)
        except FileNotFoundError:
            LOGGER.warning(
                "Keeping prepared SNAP secondary because raw SLC source is unavailable for safe regeneration | stack_id=%s swath=%s secondary=%s source_zip=%s",
                manifest.stack_id,
                iw_swath,
                secondary.scene_id,
                slc_scene_zip_path(context, manifest.stack_id, secondary),
            )
            return []
        return delete_paths(
            [product],
            category="snap_prepared",
            checkpoint="pair_prepare_on_demand",
            reason=(
                f"Prepared SNAP product for secondary {secondary.acquisition_date} {iw_swath} "
                "will be regenerated on demand from the preserved raw SLC source."
            ),
            logger=LOGGER,
        )

    def _cleanup_obsolete_pair_quarantine(
        self,
        *,
        stack_dir: Path,
        master: SlcScene,
        secondary: SlcScene,
        iw_swath: str,
        coreg_output: Path,
        ifg_output: Path,
    ) -> list[CleanupRecord]:
        if not self._pair_products_complete(coreg_output=coreg_output, ifg_output=ifg_output):
            return []

        cleanup_records: list[CleanupRecord] = []
        for child in sorted(stack_dir.iterdir(), key=lambda current: current.name):
            if not child.is_dir() or not child.name.startswith("interrupted_pair_backup_"):
                continue

            pair_paths = [
                child / "coreg" / self._coreg_product(Path("."), master, secondary, iw_swath).name,
                child / "interferograms" / self._ifg_product(Path("."), master, secondary, iw_swath).name,
            ]
            cleanup_records.extend(
                delete_paths(
                    [path for path in pair_paths if path.exists()],
                    category="snap_quarantine",
                    checkpoint="pair_products_validated",
                    reason=(
                        f"Interrupted quarantine for {master.acquisition_date}/{secondary.acquisition_date} {iw_swath} "
                        "is obsolete because the active pair products are now structurally valid."
                    ),
                    logger=LOGGER,
                )
            )
            if child.exists() and not directory_has_files(child):
                cleanup_records.extend(
                    delete_paths(
                        [child],
                        category="snap_quarantine",
                        checkpoint="pair_products_validated",
                        reason="Interrupted pair quarantine is empty after removing obsolete pair products.",
                        logger=LOGGER,
                    )
                )
        return cleanup_records

    def _cleanup_snap_intermediates(
        self,
        *,
        prepared_dir: Path,
        coreg_dir: Path,
        ifg_dir: Path,
        export_inputs_dir: Path | None = None,
        warning_records: list[CleanupWarning] | None = None,
    ) -> list[CleanupRecord]:
        return delete_paths(
            [path for path in (prepared_dir, coreg_dir, ifg_dir, export_inputs_dir) if path is not None and path.exists()],
            category="snap_intermediate",
            checkpoint="stamps_export_validated",
            reason="Prepared, coregistered, interferogram, and export-assembly SNAP products are no longer needed once a reusable StaMPS export exists.",
            logger=LOGGER,
            best_effort=True,
            retry_hidden_files=True,
            warning_records=warning_records,
        )

    def _cleanup_invalid_export_checkpoint_state(self, export_dir: Path, export_inputs_dir: Path) -> list[CleanupRecord]:
        if not export_dir.exists() or self._has_reusable_export(export_dir):
            return []
        return delete_paths(
            [export_dir],
            category="snap_export_checkpoint",
            checkpoint="invalid_stamps_export_detected",
            reason=(
                "Invalid SNAP StaMPS export output must be regenerated before the same run can continue."
            ),
            logger=LOGGER,
        )

    def _cleanup_invalid_final_export_product(
        self,
        *,
        product: Path,
        description: str,
    ) -> list[CleanupRecord]:
        if not product.exists():
            return []
        return delete_paths(
            [product],
            category="snap_export_checkpoint",
            checkpoint="invalid_final_export_product_detected",
            reason=(
                f"{description} must be regenerated because its band contract is invalid for the SNAP StaMPS export step."
            ),
            logger=LOGGER,
        )

    def _cleanup_obsolete_backups(self, context: RunContext, stack_dir: Path) -> list[CleanupRecord]:
        cleanup_records: list[CleanupRecord] = []
        cleanup_records.extend(
            delete_matching_direct_children(
                context.snap_dir,
                category="snap_backup",
                checkpoint="stamps_export_validated",
                reason="Obsolete SNAP backup lineage is not required once the active stack export is validated.",
                name_contains="_backup_",
                logger=LOGGER,
            )
        )
        cleanup_records.extend(
            delete_matching_direct_children(
                stack_dir,
                category="snap_quarantine",
                checkpoint="stamps_export_validated",
                reason="Interrupted pair quarantine is not required once the active stack export is validated.",
                name_prefix="interrupted_pair_backup_",
                logger=LOGGER,
            )
        )
        return cleanup_records

    def run_stack(
        self,
        context: RunContext,
        manifest: StackManifest,
        stack: OrbitStackConfig,
        aoi_wkt: str,
        *,
        cleanup_observer: Callable[[tuple[CleanupRecord, ...]], None] | None = None,
    ) -> SnapStackOutputs:
        stack_dir, prepared_dir, coreg_dir, ifg_dir = self._stack_dirs(context, manifest.stack_id)
        export_dir = stack_dir / "stamps_export"
        export_inputs_dir = stack_dir / "stamps_export_inputs"
        cleanup_records: list[CleanupRecord] = []
        cleanup_warnings: list[CleanupWarning] = []
        self._extend_cleanup_records(
            cleanup_records,
            self._cleanup_invalid_export_checkpoint_state(export_dir, export_inputs_dir),
            cleanup_observer,
        )
        export_assembly_started = self._has_started_export_assembly(export_inputs_dir)

        master_scene = select_master_scene(manifest, stack.master_date)
        secondaries = secondary_scenes(manifest, master_scene)
        stack_polarization = self._single_stack_polarization(stack.polarization)
        if len(secondaries) < 2:
            raise RuntimeError(
                f"Stack {manifest.stack_id} has only {len(secondaries)} secondary scenes. "
                "This single-master PSI/CDPSI workflow requires one reference plus at least two secondary images."
            )

        if self.config.cache.reuse_snap_outputs and self._has_reusable_export(export_dir):
            if self.config.artifact_lifecycle.enabled:
                if self.config.artifact_lifecycle.purge_snap_intermediates_after_export:
                    self._extend_cleanup_records(
                        cleanup_records,
                        self._cleanup_snap_intermediates(
                            prepared_dir=prepared_dir,
                            coreg_dir=coreg_dir,
                            ifg_dir=ifg_dir,
                            export_inputs_dir=export_inputs_dir,
                            warning_records=cleanup_warnings,
                        ),
                        cleanup_observer,
                    )
                if self.config.artifact_lifecycle.cleanup_obsolete_snap_backups:
                    self._extend_cleanup_records(
                        cleanup_records,
                        self._cleanup_obsolete_backups(context, stack_dir),
                        cleanup_observer,
                    )
            coreg_list = stack_dir / "coreg_products.txt"
            ifg_list = stack_dir / "ifg_products.txt"
            LOGGER.info("Reusing SNAP StaMPS export | stack_id=%s export_dir=%s", manifest.stack_id, export_dir)
            return SnapStackOutputs(
                stack_id=manifest.stack_id,
                stack_dir=stack_dir,
                prepared_dir=prepared_dir,
                coreg_dir=coreg_dir,
                interferogram_dir=ifg_dir,
                stamps_export_dir=export_dir,
                master_scene_id=master_scene.scene_id,
                coreg_list=coreg_list,
                ifg_list=ifg_list,
                cleanup_records=tuple(cleanup_records),
                cleanup_warnings=tuple(cleanup_warnings),
            )

        all_scenes = [master_scene, *secondaries]
        seeded_orbit_paths: set[Path] = set()
        for scene in all_scenes:
            orbit_path = self._ensure_orbit_auxdata_for_scene(scene)
            if orbit_path not in seeded_orbit_paths:
                LOGGER.info(
                    "Using SNAP orbit auxdata | scene=%s orbit_source=%s path=%s",
                    scene.scene_id,
                    self.config.snap.orbit_source,
                    orbit_path,
                )
                seeded_orbit_paths.add(orbit_path)
        coreg_products: list[Path] = self._valid_dimap_products_in_dir(coreg_dir) if export_assembly_started else []
        ifg_products: list[Path] = self._valid_dimap_products_in_dir(ifg_dir) if export_assembly_started else []
        if export_assembly_started:
            LOGGER.info(
                "Reusing export-assembly checkpoint and skipping SNAP pair regeneration | stack_id=%s export_inputs_dir=%s",
                manifest.stack_id,
                export_inputs_dir,
            )
        else:
            for iw_swath in stack.iw_swaths:
                master_prepared = self._prepared_product(prepared_dir, master_scene, iw_swath)
                swath_coreg_products: list[Path] = []
                swath_ifg_products: list[Path] = []
                try:
                    if (
                        self.config.cache.reuse_snap_outputs
                        and self._swath_pair_products_complete(
                            coreg_dir=coreg_dir,
                            ifg_dir=ifg_dir,
                            master=master_scene,
                            secondaries=secondaries,
                            iw_swath=iw_swath,
                        )
                    ):
                        LOGGER.info(
                            "Reusing complete SNAP swath pair products | stack_id=%s swath=%s",
                            manifest.stack_id,
                            iw_swath,
                        )
                        swath_coreg_products.extend(
                            [self._coreg_product(coreg_dir, master_scene, secondary, iw_swath) for secondary in secondaries]
                        )
                        swath_ifg_products.extend(
                            [self._ifg_product(ifg_dir, master_scene, secondary, iw_swath) for secondary in secondaries]
                        )
                        if self.config.artifact_lifecycle.enabled and self.config.artifact_lifecycle.purge_secondary_prepared_after_pair:
                            for secondary in secondaries:
                                self._extend_cleanup_records(
                                    cleanup_records,
                                    self._cleanup_secondary_prepared_product(
                                        prepared_dir=prepared_dir,
                                        secondary=secondary,
                                        iw_swath=iw_swath,
                                        ),
                                        cleanup_observer,
                                    )
                        if self.config.artifact_lifecycle.enabled and self.config.artifact_lifecycle.cleanup_obsolete_snap_backups:
                            for secondary in secondaries:
                                self._extend_cleanup_records(
                                    cleanup_records,
                                    self._cleanup_obsolete_pair_quarantine(
                                        stack_dir=stack_dir,
                                        master=master_scene,
                                        secondary=secondary,
                                        iw_swath=iw_swath,
                                        coreg_output=self._coreg_product(coreg_dir, master_scene, secondary, iw_swath),
                                        ifg_output=self._ifg_product(ifg_dir, master_scene, secondary, iw_swath),
                                    ),
                                    cleanup_observer,
                                )
                        if self.config.artifact_lifecycle.enabled and self.config.artifact_lifecycle.purge_master_prepared_after_pair:
                            self._extend_cleanup_records(
                                cleanup_records,
                                self._cleanup_master_prepared_product(
                                    context=context,
                                    manifest=manifest,
                                    prepared_dir=prepared_dir,
                                    master=master_scene,
                                    iw_swath=iw_swath,
                                ),
                                cleanup_observer,
                            )
                    else:
                        if self.config.artifact_lifecycle.enabled and self.config.artifact_lifecycle.purge_secondary_prepared_after_pair:
                            for secondary in secondaries:
                                coreg_output = self._coreg_product(coreg_dir, master_scene, secondary, iw_swath)
                                ifg_output = self._ifg_product(ifg_dir, master_scene, secondary, iw_swath)
                                if self.config.cache.reuse_snap_outputs and self._pair_products_complete(
                                    coreg_output=coreg_output,
                                    ifg_output=ifg_output,
                                ):
                                    self._extend_cleanup_records(
                                        cleanup_records,
                                        self._cleanup_secondary_prepared_product(
                                            prepared_dir=prepared_dir,
                                            secondary=secondary,
                                            iw_swath=iw_swath,
                                        ),
                                        cleanup_observer,
                                    )
                                else:
                                    self._extend_cleanup_records(
                                        cleanup_records,
                                        self._cleanup_secondary_prepared_for_on_demand_regeneration(
                                            context=context,
                                            manifest=manifest,
                                            prepared_dir=prepared_dir,
                                            secondary=secondary,
                                            iw_swath=iw_swath,
                                        ),
                                        cleanup_observer,
                                    )

                        for secondary in secondaries:
                            secondary_prepared = self._prepared_product(prepared_dir, secondary, iw_swath)
                            coreg_output = self._coreg_product(coreg_dir, master_scene, secondary, iw_swath)
                            ifg_output = self._ifg_product(ifg_dir, master_scene, secondary, iw_swath)
                            if self.config.cache.reuse_snap_outputs and self._pair_products_complete(
                                coreg_output=coreg_output,
                                ifg_output=ifg_output,
                            ):
                                swath_coreg_products.append(coreg_output)
                                swath_ifg_products.append(ifg_output)
                                if (
                                    self.config.artifact_lifecycle.enabled
                                    and self.config.artifact_lifecycle.purge_secondary_prepared_after_pair
                                ):
                                    self._extend_cleanup_records(
                                        cleanup_records,
                                        self._cleanup_secondary_prepared_product(
                                            prepared_dir=prepared_dir,
                                            secondary=secondary,
                                            iw_swath=iw_swath,
                                        ),
                                        cleanup_observer,
                                    )
                                if (
                                    self.config.artifact_lifecycle.enabled
                                    and self.config.artifact_lifecycle.purge_master_prepared_after_pair
                                ):
                                    self._extend_cleanup_records(
                                        cleanup_records,
                                        self._cleanup_master_prepared_product(
                                            context=context,
                                            manifest=manifest,
                                            prepared_dir=prepared_dir,
                                            master=master_scene,
                                            iw_swath=iw_swath,
                                        ),
                                        cleanup_observer,
                                    )
                                if (
                                    self.config.artifact_lifecycle.enabled
                                    and self.config.artifact_lifecycle.cleanup_obsolete_snap_backups
                                ):
                                    self._extend_cleanup_records(
                                        cleanup_records,
                                        self._cleanup_obsolete_pair_quarantine(
                                            stack_dir=stack_dir,
                                            master=master_scene,
                                            secondary=secondary,
                                            iw_swath=iw_swath,
                                            coreg_output=coreg_output,
                                            ifg_output=ifg_output,
                                        ),
                                        cleanup_observer,
                                    )
                                continue

                            if not master_prepared.exists() or not self.config.cache.reuse_snap_outputs:
                                self._run_graph(
                                    self._prepare_job(
                                        context=context,
                                        manifest=manifest,
                                        scene=master_scene,
                                        polarization=stack_polarization,
                                        iw_swath=iw_swath,
                                        output_file=master_prepared,
                                        aoi_wkt=aoi_wkt,
                                    )
                                )

                            if not secondary_prepared.exists() or not self.config.cache.reuse_snap_outputs:
                                self._run_graph(
                                    self._prepare_job(
                                        context=context,
                                        manifest=manifest,
                                        scene=secondary,
                                        polarization=stack_polarization,
                                        iw_swath=iw_swath,
                                        output_file=secondary_prepared,
                                        aoi_wkt=aoi_wkt,
                                    )
                                )

                            try:
                                self._run_graph(
                                    self._coreg_job(
                                        master_file=master_prepared,
                                        secondary_file=secondary_prepared,
                                        iw_swath=iw_swath,
                                        coreg_output=coreg_output,
                                        ifg_output=ifg_output,
                                        aoi_wkt=aoi_wkt,
                                    )
                                )
                            except SnapEsdNotApplicableError:
                                LOGGER.warning(
                                    "Retrying SNAP coregistration without ESD | stack_id=%s swath=%s master=%s secondary=%s",
                                    manifest.stack_id,
                                    iw_swath,
                                    master_scene.scene_id,
                                    secondary.scene_id,
                                )
                                self._run_graph(
                                    self._coreg_job(
                                        master_file=master_prepared,
                                        secondary_file=secondary_prepared,
                                        iw_swath=iw_swath,
                                        coreg_output=coreg_output,
                                        ifg_output=ifg_output,
                                        aoi_wkt=aoi_wkt,
                                        use_esd=False,
                                    )
                                )

                            swath_coreg_products.append(coreg_output)
                            swath_ifg_products.append(ifg_output)
                            if (
                                self.config.artifact_lifecycle.enabled
                                and self.config.artifact_lifecycle.purge_secondary_prepared_after_pair
                                and self._pair_products_complete(coreg_output=coreg_output, ifg_output=ifg_output)
                            ):
                                self._extend_cleanup_records(
                                    cleanup_records,
                                    self._cleanup_secondary_prepared_product(
                                        prepared_dir=prepared_dir,
                                        secondary=secondary,
                                        iw_swath=iw_swath,
                                    ),
                                    cleanup_observer,
                                )
                            if (
                                self.config.artifact_lifecycle.enabled
                                and self.config.artifact_lifecycle.purge_master_prepared_after_pair
                                and self._pair_products_complete(coreg_output=coreg_output, ifg_output=ifg_output)
                            ):
                                self._extend_cleanup_records(
                                    cleanup_records,
                                    self._cleanup_master_prepared_product(
                                        context=context,
                                        manifest=manifest,
                                        prepared_dir=prepared_dir,
                                        master=master_scene,
                                        iw_swath=iw_swath,
                                    ),
                                    cleanup_observer,
                                )
                            if (
                                self.config.artifact_lifecycle.enabled
                                and self.config.artifact_lifecycle.cleanup_obsolete_snap_backups
                                and self._pair_products_complete(coreg_output=coreg_output, ifg_output=ifg_output)
                            ):
                                self._extend_cleanup_records(
                                    cleanup_records,
                                    self._cleanup_obsolete_pair_quarantine(
                                        stack_dir=stack_dir,
                                        master=master_scene,
                                        secondary=secondary,
                                        iw_swath=iw_swath,
                                        coreg_output=coreg_output,
                                        ifg_output=ifg_output,
                                    ),
                                    cleanup_observer,
                                )
                except SnapNoIntersectionError:
                    LOGGER.warning(
                        "Skipping SNAP swath with no AOI intersection | stack_id=%s swath=%s",
                        manifest.stack_id,
                        iw_swath,
                    )
                    self._cleanup_prepared_swath(prepared_dir, all_scenes, iw_swath)
                    self._cleanup_coreg_swath(coreg_dir, ifg_dir, master_scene, secondaries, iw_swath)
                    continue

                if not swath_coreg_products or not swath_ifg_products:
                    continue
                coreg_products.extend(swath_coreg_products)
                ifg_products.extend(swath_ifg_products)
                if (
                    self.config.artifact_lifecycle.enabled
                    and self.config.artifact_lifecycle.purge_prepared_after_coreg
                    and are_valid_dimap_products(swath_coreg_products)
                    and are_valid_dimap_products(swath_ifg_products)
                ):
                    self._extend_cleanup_records(
                        cleanup_records,
                        self._cleanup_prepared_products(
                            prepared_dir=prepared_dir,
                            scenes=all_scenes,
                            iw_swath=iw_swath,
                        ),
                        cleanup_observer,
                    )

        if not export_assembly_started and (not coreg_products or not ifg_products):
            raise RuntimeError(
                f"Stack {manifest.stack_id} did not produce any usable coregistered/interferogram products for the configured AOI."
            )

        coreg_list = self._product_list_file(stack_dir, "coreg_products.txt", sorted(coreg_products))
        ifg_list = self._product_list_file(stack_dir, "ifg_products.txt", sorted(ifg_products))
        final_coreg_products, final_ifg_products = self._assemble_export_source_products(
            stack_dir=stack_dir,
            stack_id=manifest.stack_id,
            master_scene=master_scene,
            secondaries=secondaries,
            iw_swaths=stack.iw_swaths,
            polarization=stack_polarization,
            coreg_products=sorted(coreg_products),
            ifg_products=sorted(ifg_products),
            cleanup_records=cleanup_records,
            cleanup_observer=cleanup_observer,
        )
        self._run_graph(
            self._export_job(
                stack_dir=stack_dir,
                coreg_product=final_coreg_products[0],
                ifg_product=final_ifg_products[0],
                aoi_wkt=aoi_wkt,
                output_dir=export_dir,
            )
        )
        if not self._has_reusable_export(export_dir):
            raise RuntimeError(
                f"Stack {manifest.stack_id} did not produce a structurally valid StaMPS export directory: {export_dir}"
            )
        if self.config.artifact_lifecycle.enabled:
            if self.config.artifact_lifecycle.purge_snap_intermediates_after_export:
                    self._extend_cleanup_records(
                        cleanup_records,
                        self._cleanup_snap_intermediates(
                            prepared_dir=prepared_dir,
                            coreg_dir=coreg_dir,
                            ifg_dir=ifg_dir,
                            export_inputs_dir=export_inputs_dir,
                            warning_records=cleanup_warnings,
                        ),
                        cleanup_observer,
                    )
            if self.config.artifact_lifecycle.cleanup_obsolete_snap_backups:
                self._extend_cleanup_records(
                    cleanup_records,
                    self._cleanup_obsolete_backups(context, stack_dir),
                    cleanup_observer,
                )

        return SnapStackOutputs(
            stack_id=manifest.stack_id,
            stack_dir=stack_dir,
            prepared_dir=prepared_dir,
            coreg_dir=coreg_dir,
            interferogram_dir=ifg_dir,
            stamps_export_dir=export_dir,
            master_scene_id=master_scene.scene_id,
            coreg_list=coreg_list,
            ifg_list=ifg_list,
            cleanup_records=tuple(cleanup_records),
            cleanup_warnings=tuple(cleanup_warnings),
        )
