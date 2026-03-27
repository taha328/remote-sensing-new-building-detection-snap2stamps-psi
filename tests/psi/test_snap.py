from __future__ import annotations

from datetime import date as calendar_date
from pathlib import Path
from xml.sax.saxutils import escape
from xml.etree import ElementTree as ET

import pytest

from casablanca_psi.artifact_lifecycle import CleanupWarning
from casablanca_psi.config import load_config
from casablanca_psi.manifests import SlcScene, StackManifest, slc_scene_zip_path
from casablanca_psi.run_context import RunContext
from casablanca_psi.snap import (
    SnapEsdNotApplicableError,
    SnapGraphJob,
    SnapGraphRunner,
    SnapNoIntersectionError,
)


def _build_manifest() -> StackManifest:
    scenes = []
    for index, acquisition_date in enumerate(["2023-08-29", "2023-09-10", "2023-09-22", "2023-10-04", "2023-10-16"], start=1):
        scenes.append(
            SlcScene(
                scene_id=f"SCENE_{index}",
                product_name=f"SCENE_{index}",
                acquisition_start=f"{acquisition_date}T00:00:00Z",
                acquisition_stop=f"{acquisition_date}T00:00:10Z",
                acquisition_date=acquisition_date,
                direction="ascending",
                relative_orbit=147,
                polarization="VV+VH",
                swath_mode="IW",
                product_type="IW_SLC__1S",
                processing_level="L1",
                platform="Sentinel-1A",
                asset_name="product",
                href="https://download.example.invalid/scene.zip",
            )
        )
    return StackManifest(
        stack_id="asc_rel147_vv",
        direction="ascending",
        relative_orbit=147,
        product_type="SLC",
        scenes=scenes,
    )


def _write_fake_dimap(output: Path, *, band_names: list[str] | None = None) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    names = band_names or _default_fake_band_names(output)
    data_dir = output.with_suffix(".data")
    data_dir.mkdir(exist_ok=True)
    spectral_band_infos = []
    data_files = []
    for index, name in enumerate(names):
        hdr_name = f"band_{index}.hdr"
        img_name = f"band_{index}.img"
        spectral_band_infos.append(
            "<Spectral_Band_Info>"
            f"<BAND_INDEX>{index}</BAND_INDEX>"
            f"<BAND_NAME>{escape(name)}</BAND_NAME>"
            "</Spectral_Band_Info>"
        )
        data_files.append(
            "<Data_File>"
            f"<BAND_INDEX>{index}</BAND_INDEX>"
            f"<DATA_FILE_PATH href=\"{escape(hdr_name)}\" />"
            "</Data_File>"
        )
        (data_dir / hdr_name).write_text("ENVI\n", encoding="utf-8")
        (data_dir / img_name).write_text("ok", encoding="utf-8")
    output.write_text(
        "<Dimap_Document>"
        f"<Raster_Dimensions><NBANDS>{len(names)}</NBANDS></Raster_Dimensions>"
        "<Data_Access>"
        + "".join(data_files)
        + "</Data_Access>"
        + "<Image_Interpretation>"
        + "".join(spectral_band_infos)
        + "</Image_Interpretation>"
        + _fake_metadata_xml(names)
        + "</Dimap_Document>",
        encoding="utf-8",
    )


def _default_fake_band_names(output: Path) -> list[str]:
    stem = output.stem
    if stem.endswith("_coreg_final"):
        return _fake_final_coreg_bands()
    if stem.endswith("_ifg_final"):
        return _fake_final_ifg_bands()
    if stem.endswith("_coreg_export"):
        suffix = _scene_date_token(stem.split("_")[1])
        return [
            "i_VV_mst_22Sep2023",
            "q_VV_mst_22Sep2023",
            f"i_VV_slv1_{suffix}",
            f"q_VV_slv1_{suffix}",
        ]
    if stem.endswith("_coreg_stack_input"):
        suffix = _scene_date_token(stem.split("_")[1])
        return [f"i_VV_slv1_{suffix}", f"q_VV_slv1_{suffix}"]
    if stem.endswith("_ifg_stack_input"):
        suffix = _scene_date_token(stem.split("_")[1])
        return [
            f"i_ifg_VV_22Sep2023_{suffix}",
            f"q_ifg_VV_22Sep2023_{suffix}",
        ]
    if stem.endswith("_ifg_export"):
        return _fake_export_ifg_bands(stem.split("_")[1])
    return ["i_VV_mst_22Sep2023", "q_VV_mst_22Sep2023", "i_VV_slv1_29Aug2023", "q_VV_slv1_29Aug2023"]


def _fake_dual_pol_coreg_bands(secondary_date: str) -> list[str]:
    suffix = _scene_date_token(secondary_date)
    return [
        "i_VH_mst_22Sep2023",
        "q_VH_mst_22Sep2023",
        "i_VV_mst_22Sep2023",
        "q_VV_mst_22Sep2023",
        f"i_VH_slv1_{suffix}",
        f"q_VH_slv1_{suffix}",
        f"i_VV_slv1_{suffix}",
        f"q_VV_slv1_{suffix}",
    ]


def _fake_export_ifg_bands(secondary_date: str) -> list[str]:
    suffix = _scene_date_token(secondary_date)
    return [
        f"i_ifg_VV_22Sep2023_{suffix}",
        f"q_ifg_VV_22Sep2023_{suffix}",
        "elevation",
        "orthorectifiedLat",
        "orthorectifiedLon",
    ]


def _scene_date_token(acquisition_date: str) -> str:
    return calendar_date.fromisoformat(acquisition_date).strftime("%d%b%Y")


def _fake_final_coreg_bands() -> list[str]:
    return [
        "i_VV_mst_22Sep2023",
        "q_VV_mst_22Sep2023",
        "i_VV_slv1_29Aug2023",
        "q_VV_slv1_29Aug2023",
        "i_VV_slv1_10Sep2023",
        "q_VV_slv1_10Sep2023",
        "i_VV_slv1_04Oct2023",
        "q_VV_slv1_04Oct2023",
        "i_VV_slv1_16Oct2023",
        "q_VV_slv1_16Oct2023",
    ]


def _fake_final_ifg_bands() -> list[str]:
    return [
        "i_ifg_VV_22Sep2023_29Aug2023",
        "q_ifg_VV_22Sep2023_29Aug2023",
        "elevation",
        "orthorectifiedLat",
        "orthorectifiedLon",
        "i_ifg_VV_22Sep2023_10Sep2023",
        "q_ifg_VV_22Sep2023_10Sep2023",
        "i_ifg_VV_22Sep2023_04Oct2023",
        "q_ifg_VV_22Sep2023_04Oct2023",
        "i_ifg_VV_22Sep2023_16Oct2023",
        "q_ifg_VV_22Sep2023_16Oct2023",
    ]


def _fake_metadata_xml(band_names: list[str]) -> str:
    master_bands = [
        name
        for name in band_names
        if name.startswith(("i_", "q_")) and "_mst_" in name
    ]
    slave_bands = [
        name
        for name in band_names
        if name.startswith(("i_", "q_")) and "_slv" in name and "_ifg_" not in name
    ]
    if not master_bands and not slave_bands:
        return ""

    master_date = next(
        (name.split("_mst_", 1)[1].split("_", 1)[0] for name in master_bands if "_mst_" in name),
        "22Sep2023",
    )
    slave_dates: list[str] = []
    for band_name in slave_bands:
        token = band_name.split("_slv", 1)[1].split("_", 1)[1].split("_", 1)[0]
        if token not in slave_dates:
            slave_dates.append(token)

    baselines = (
        "<MDElem name=\"Abstracted_Metadata\">"
        "<MDATTR name=\"pulse_repetition_frequency\" type=\"float64\" mode=\"rw\">1.0</MDATTR>"
        "<MDElem name=\"Baselines\">"
        f"<MDElem name=\"Ref_{master_date}\">"
        f"<MDElem name=\"Secondary_{master_date}\" />"
        + "".join(f"<MDElem name=\"Secondary_{date}\" />" for date in slave_dates)
        + "</MDElem>"
        + "".join(
            f"<MDElem name=\"Ref_{date}\">"
            f"<MDElem name=\"Secondary_{master_date}\" />"
            f"<MDElem name=\"Secondary_{date}\" />"
            "</MDElem>"
            for date in slave_dates
        )
        + "</MDElem>"
        + "</MDElem>"
    )
    slave_metadata = (
        "<MDElem name=\"Slave_Metadata\">"
        f"<MDATTR name=\"Master_bands\" type=\"ascii\" mode=\"rw\">{' '.join(master_bands)}</MDATTR>"
        + "".join(
            f"<MDElem name=\"slave_{date}\">"
            f"<MDATTR name=\"PRODUCT\" type=\"ascii\" mode=\"rw\">FAKE_{date}</MDATTR>"
            f"<MDATTR name=\"first_line_time\" type=\"ascii\" mode=\"rw\">{date[:2]}-{date[2:5].upper()}-{date[5:]} 00:00:00.000000</MDATTR>"
            f"<MDATTR name=\"last_line_time\" type=\"ascii\" mode=\"rw\">{date[:2]}-{date[2:5].upper()}-{date[5:]} 00:00:10.000000</MDATTR>"
            f"<MDATTR name=\"ABS_ORBIT\" type=\"ascii\" mode=\"rw\">1</MDATTR>"
            f"<MDATTR name=\"REL_ORBIT\" type=\"ascii\" mode=\"rw\">147</MDATTR>"
            f"<MDATTR name=\"STATE_VECTOR_TIME\" type=\"ascii\" mode=\"rw\">{date[:2]}-{date[2:5].upper()}-{date[5:]} 00:00:00.000000</MDATTR>"
            f"<MDATTR name=\"PASS\" type=\"ascii\" mode=\"rw\">ASCENDING</MDATTR>"
            f"<MDATTR name=\"mds1_tx_rx_polar\" type=\"ascii\" mode=\"rw\">VV</MDATTR>"
            f"<MDATTR name=\"radar_frequency\" type=\"ascii\" mode=\"rw\">5405.000454334349</MDATTR>"
            f"<MDATTR name=\"Slave_bands\" type=\"ascii\" mode=\"rw\">{' '.join([band for band in slave_bands if date in band])}</MDATTR>"
            "<MDElem name=\"Band_IW1_VV\">"
            f"<MDATTR name=\"band_names\" type=\"ascii\" mode=\"rw\">{' '.join([band for band in slave_bands if date in band])}</MDATTR>"
            "</MDElem>"
            "</MDElem>"
            for date in slave_dates
        )
        + "</MDElem>"
    )
    return baselines + slave_metadata


def _has_fake_export(output_dir: Path) -> bool:
    return output_dir.exists() and all((output_dir / name / "ok.txt").exists() for name in ("rslc", "diff0", "geo"))


def test_validate_environment_rejects_non_snap_gpt(monkeypatch) -> None:
    config = load_config(Path("configs/psi_casablanca_slc.yaml"))
    runner = SnapGraphRunner(config)

    monkeypatch.setattr("shutil.which", lambda _: "/usr/sbin/gpt")

    class Result:
        stdout = "usage: gpt add partition"
        stderr = ""

    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: Result())

    with pytest.raises(RuntimeError, match="does not resolve to SNAP GPT"):
        runner.validate_environment()


def test_validate_environment_requires_dem(monkeypatch, tmp_path) -> None:
    config = load_config(Path("configs/psi_casablanca_slc.yaml"))
    config.dem.path = tmp_path / "missing_dem.tif"
    runner = SnapGraphRunner(config)

    monkeypatch.setattr("shutil.which", lambda _: "/Applications/esa-snap/bin/gpt")

    class Result:
        stdout = "Graph Processing Tool"
        stderr = ""

    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: Result())

    with pytest.raises(FileNotFoundError, match="Configured DEM path does not exist"):
        runner.validate_environment()


def test_graph_path_is_resolved_to_absolute_path() -> None:
    config = load_config(Path("configs/psi_casablanca_slc.yaml"))
    runner = SnapGraphRunner(config)

    graph_path = runner._graph_path("prepare_slc_stack.xml")

    assert graph_path.is_absolute()
    assert graph_path.name == "prepare_slc_stack.xml"


def test_coreg_and_export_jobs_use_absolute_dem_path() -> None:
    config = load_config(Path("configs/psi_casablanca_slc_minimal.yaml"))
    runner = SnapGraphRunner(config)
    coreg_job = runner._coreg_job(
        master_file=Path("/tmp/master.dim"),
        secondary_file=Path("/tmp/secondary.dim"),
        iw_swath="IW1",
        coreg_output=Path("/tmp/coreg.dim"),
        ifg_output=Path("/tmp/ifg.dim"),
        aoi_wkt="POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))",
    )
    export_job = runner._export_job(
        stack_dir=Path("/tmp/stack"),
        coreg_product=Path("/tmp/coreg_final.dim"),
        ifg_product=Path("/tmp/ifg_final.dim"),
        aoi_wkt="POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))",
        output_dir=Path("/tmp/export"),
    )

    assert Path(coreg_job.parameters["demPath"]).is_absolute()
    assert export_job.parameters["coregFile"] == "/tmp/coreg_final.dim"
    assert export_job.parameters["ifgFile"] == "/tmp/ifg_final.dim"


def test_snap_polarization_normalizes_dual_pol_combination_order() -> None:
    config = load_config(Path("configs/psi_casablanca_slc.yaml"))
    runner = SnapGraphRunner(config)

    class Scene:
        polarization = "VH+VV"

    assert runner._snap_polarization(Scene()) == "VV,VH"


def test_prepare_job_uses_configured_stack_polarization(tmp_path) -> None:
    config = load_config(Path("configs/psi_casablanca_slc_minimal.yaml"))
    runner = SnapGraphRunner(config)
    context = RunContext.create(config, tmp_path)
    manifest = _build_manifest()
    scene = manifest.scenes[0]

    job = runner._prepare_job(
        context=context,
        manifest=manifest,
        scene=scene,
        polarization=config.stacks[0].polarization,
        iw_swath="IW1",
        output_file=tmp_path / "prepared" / "scene.dim",
        aoi_wkt="POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))",
    )

    assert job.parameters["polarization"] == "VV"


def test_cleanup_failed_job_outputs_removes_partial_products(tmp_path) -> None:
    config = load_config(Path("configs/psi_casablanca_slc.yaml"))
    runner = SnapGraphRunner(config)
    output_file = tmp_path / "prepared" / "master.dim"
    data_dir = output_file.with_suffix(".data")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("partial")
    data_dir.mkdir()
    (data_dir / "band.img").write_text("partial")
    export_dir = tmp_path / "stamps_export"
    export_dir.mkdir()
    (export_dir / "partial.txt").write_text("partial")
    job = SnapGraphJob(
        name="failed-job",
        graph_path=Path("/tmp/graph.xml"),
        parameters={
            "outputFile": str(output_file),
            "outputDir": str(export_dir),
        },
        work_dir=tmp_path,
    )

    runner._cleanup_failed_job_outputs(job)

    assert not output_file.exists()
    assert not data_dir.exists()


def test_run_graph_passes_configured_parallelism_and_tile_cache_flags(monkeypatch, tmp_path) -> None:
    config = load_config(Path("configs/psi_casablanca_slc_minimal.yaml"))
    runner = SnapGraphRunner(config)
    job = SnapGraphJob(
        name="test-job",
        graph_path=tmp_path / "graph.xml",
        parameters={"outputFile": str(tmp_path / "out.dim")},
        work_dir=tmp_path,
    )
    job.graph_path.write_text("<graph />", encoding="utf-8")

    commands: list[list[str]] = []

    monkeypatch.setattr(runner, "validate_environment", lambda: None)

    class Result:
        stdout = ""
        stderr = ""

    def fake_run(command, **kwargs):
        commands.append(command)
        return Result()

    monkeypatch.setattr("subprocess.run", fake_run)

    runner._run_graph(job)

    assert len(commands) == 1
    command = commands[0]
    assert "-J-Xms2G" in command
    assert "-J-Xmx8G" in command
    assert any(entry.startswith("-J-Dsnap.userdir=") for entry in command)
    assert "-J-Dsnap.jai.defaultTileSize=512" in command
    assert "-J-Dsnap.parallelism=1" in command
    cache_index = command.index("-c")
    parallelism_index = command.index("-q")
    assert command[cache_index + 1] == "2048M"
    assert command[parallelism_index + 1] == "1"
    assert "-x" in command


def test_describe_runtime_policy_audits_installed_vmoptions() -> None:
    config = load_config(Path("configs/psi_casablanca_slc_minimal.yaml"))
    runner = SnapGraphRunner(config)

    policy = runner.describe_runtime_policy()

    assert policy["gpt_path"] == "/Applications/esa-snap/bin/gpt"
    assert policy["gpt_vmoptions_path"] == "/Applications/esa-snap/bin/gpt.vmoptions"
    assert policy["gpt_vmoptions_exists"] is True
    assert "-Xmx11G" in policy["installed_vmoptions"]
    assert "-Xmx8G" in policy["effective_java_options"]
    assert {
        "key": "-Xmx",
        "installed": "-Xmx11G",
        "runtime": "-Xmx8G",
    } in policy["vmoptions_overrides"]
    assert policy["snap_user_dir"].endswith("data/cache/snap-gpt-userdir")
    assert policy["auxdata_root"].endswith("data/cache/snap-gpt-userdir/auxdata")
    assert policy["default_tile_size_px"] == 512


def test_is_no_intersection_output_matches_snap_messages() -> None:
    config = load_config(Path("configs/psi_casablanca_slc.yaml"))
    runner = SnapGraphRunner(config)

    assert runner._is_no_intersection_output("WARNING: No intersection with source product boundary")
    assert runner._is_no_intersection_output("The specified region, if not null, must intersect with the image`s bounds.")
    assert not runner._is_no_intersection_output("Registration window width should not be grater than burst width 0")


def test_is_esd_not_applicable_output_matches_snap_message() -> None:
    config = load_config(Path("configs/psi_casablanca_slc.yaml"))
    runner = SnapGraphRunner(config)

    assert runner._is_esd_not_applicable_output("Registration window width should not be grater than burst width 0")


def test_run_stack_skips_non_intersecting_swath_and_continues(tmp_path, monkeypatch) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "psi_casablanca_slc_minimal.yaml")
    runner = SnapGraphRunner(config)
    monkeypatch.setattr(runner, "validate_environment", lambda: None)
    monkeypatch.setattr(runner, "_has_reusable_export", _has_fake_export)
    context = RunContext.create(config, tmp_path)
    context.ensure_directories()

    manifest = _build_manifest()
    stack = config.stacks[0]

    def fake_run_graph(job):
        if job.parameters.get("subswath") == "IW2":
            raise SnapNoIntersectionError("no overlap")
        for key in ("outputFile", "coregOutputFile", "ifgOutputFile"):
            if key in job.parameters:
                _write_fake_dimap(Path(job.parameters[key]))
        if "outputDir" in job.parameters:
            output_dir = Path(job.parameters["outputDir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            for name in ("rslc", "diff0", "geo"):
                subdir = output_dir / name
                subdir.mkdir(exist_ok=True)
                (subdir / "ok.txt").write_text("ok")

    monkeypatch.setattr(runner, "_run_graph", fake_run_graph)

    outputs = runner.run_stack(context, manifest, stack, "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))")

    coreg_list = outputs.coreg_list.read_text().splitlines()
    ifg_list = outputs.ifg_list.read_text().splitlines()
    assert len(coreg_list) == 8
    assert len(ifg_list) == 8
    assert all("_IW2_" not in line for line in coreg_list)
    assert all("_IW2_" not in line for line in ifg_list)


def test_run_stack_retries_coreg_without_esd_on_esd_error(tmp_path, monkeypatch) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "psi_casablanca_slc_minimal.yaml")
    runner = SnapGraphRunner(config)
    monkeypatch.setattr(runner, "validate_environment", lambda: None)
    monkeypatch.setattr(runner, "_has_reusable_export", _has_fake_export)
    context = RunContext.create(config, tmp_path)
    context.ensure_directories()

    manifest = _build_manifest()
    stack = config.stacks[0]
    coreg_attempts: list[str] = []

    def fake_run_graph(job):
        graph_name = job.graph_path.name
        if graph_name == "coregister_stack.xml":
            coreg_attempts.append(graph_name)
            raise SnapEsdNotApplicableError("esd not applicable")
        if graph_name == "coregister_stack_no_esd.xml":
            coreg_attempts.append(graph_name)
        for key in ("outputFile", "coregOutputFile", "ifgOutputFile"):
            if key in job.parameters:
                _write_fake_dimap(Path(job.parameters[key]))
        if "outputDir" in job.parameters:
            output_dir = Path(job.parameters["outputDir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            for name in ("rslc", "diff0", "geo"):
                subdir = output_dir / name
                subdir.mkdir(exist_ok=True)
                (subdir / "ok.txt").write_text("ok")

    monkeypatch.setattr(runner, "_run_graph", fake_run_graph)

    outputs = runner.run_stack(context, manifest, stack, "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))")

    coreg_list = outputs.coreg_list.read_text().splitlines()
    ifg_list = outputs.ifg_list.read_text().splitlines()
    assert len(coreg_list) == 12
    assert len(ifg_list) == 12
    assert "coregister_stack.xml" in coreg_attempts
    assert "coregister_stack_no_esd.xml" in coreg_attempts


def test_run_stack_purges_prepared_products_after_validated_swath_coreg(tmp_path, monkeypatch) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "psi_casablanca_slc_minimal.yaml")
    config.artifact_lifecycle.purge_pair_products_after_merged_coreg = False
    config.artifact_lifecycle.purge_prepared_after_coreg = True
    config.artifact_lifecycle.purge_snap_intermediates_after_export = False
    runner = SnapGraphRunner(config)
    monkeypatch.setattr(runner, "validate_environment", lambda: None)
    monkeypatch.setattr(runner, "_has_reusable_export", _has_fake_export)
    context = RunContext.create(config, tmp_path)
    context.ensure_directories()
    manifest = _build_manifest()
    stack = config.stacks[0]

    for scene in manifest.scenes:
        zip_path = slc_scene_zip_path(context, manifest.stack_id, scene)
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        zip_path.write_text("placeholder")

    def fake_run_graph(job):
        for key in ("outputFile", "coregOutputFile", "ifgOutputFile"):
            if key in job.parameters:
                _write_fake_dimap(Path(job.parameters[key]))
        if "outputDir" in job.parameters:
            output_dir = Path(job.parameters["outputDir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            for name in ("rslc", "diff0", "geo"):
                subdir = output_dir / name
                subdir.mkdir(exist_ok=True)
                (subdir / "ok.txt").write_text("ok")

    monkeypatch.setattr(runner, "_run_graph", fake_run_graph)

    outputs = runner.run_stack(context, manifest, stack, "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))")

    assert not any(outputs.prepared_dir.glob("*.dim"))
    assert any(outputs.coreg_dir.glob("*.dim"))
    assert any(outputs.interferogram_dir.glob("*.dim"))
    assert any(record.category == "snap_prepared" for record in outputs.cleanup_records)


def test_run_stack_prepares_secondaries_just_in_time_per_pair(tmp_path, monkeypatch) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "psi_casablanca_slc_minimal.yaml")
    config.artifact_lifecycle.purge_master_prepared_after_pair = False
    config.artifact_lifecycle.purge_secondary_prepared_after_pair = False
    config.artifact_lifecycle.purge_prepared_after_coreg = False
    config.artifact_lifecycle.purge_snap_intermediates_after_export = False
    runner = SnapGraphRunner(config)
    monkeypatch.setattr(runner, "validate_environment", lambda: None)
    monkeypatch.setattr(runner, "_has_reusable_export", _has_fake_export)
    context = RunContext.create(config, tmp_path)
    context.ensure_directories()
    manifest = _build_manifest()
    stack = config.stacks[0]
    graph_calls: list[tuple[str, str | None, str | None]] = []

    def fake_run_graph(job):
        graph_calls.append((job.graph_path.name, job.parameters.get("subswath"), job.parameters.get("secondaryFile")))
        for key in ("outputFile", "coregOutputFile", "ifgOutputFile"):
            if key in job.parameters:
                _write_fake_dimap(Path(job.parameters[key]))
        if "outputDir" in job.parameters:
            output_dir = Path(job.parameters["outputDir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            for name in ("rslc", "diff0", "geo"):
                subdir = output_dir / name
                subdir.mkdir(exist_ok=True)
                (subdir / "ok.txt").write_text("ok")

    monkeypatch.setattr(runner, "_run_graph", fake_run_graph)

    runner.run_stack(context, manifest, stack, "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))")

    first_coreg_index = next(index for index, (graph_name, _, _) in enumerate(graph_calls) if graph_name.startswith("coregister_stack"))
    prepare_before_first_coreg = [
        (graph_name, subswath)
        for graph_name, subswath, _secondary_file in graph_calls[:first_coreg_index]
        if graph_name == "prepare_slc_stack.xml"
    ]
    assert len(prepare_before_first_coreg) == 2
    assert prepare_before_first_coreg == [
        ("prepare_slc_stack.xml", "IW1"),
        ("prepare_slc_stack.xml", "IW1"),
    ]


def test_run_stack_purges_secondary_prepared_products_after_each_valid_pair(tmp_path, monkeypatch) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "psi_casablanca_slc_minimal.yaml")
    config.artifact_lifecycle.purge_master_prepared_after_pair = False
    config.artifact_lifecycle.purge_secondary_prepared_after_pair = True
    config.artifact_lifecycle.purge_prepared_after_coreg = False
    config.artifact_lifecycle.purge_snap_intermediates_after_export = False
    runner = SnapGraphRunner(config)
    monkeypatch.setattr(runner, "validate_environment", lambda: None)
    monkeypatch.setattr(runner, "_has_reusable_export", _has_fake_export)
    context = RunContext.create(config, tmp_path)
    context.ensure_directories()
    manifest = _build_manifest()
    stack = config.stacks[0]
    master = manifest.scenes[2]

    def fake_run_graph(job):
        for key in ("outputFile", "coregOutputFile", "ifgOutputFile"):
            if key in job.parameters:
                _write_fake_dimap(Path(job.parameters[key]))
        if "outputDir" in job.parameters:
            output_dir = Path(job.parameters["outputDir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            for name in ("rslc", "diff0", "geo"):
                subdir = output_dir / name
                subdir.mkdir(exist_ok=True)
                (subdir / "ok.txt").write_text("ok")

    monkeypatch.setattr(runner, "_run_graph", fake_run_graph)

    outputs = runner.run_stack(context, manifest, stack, "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))")

    remaining_prepared = sorted(path.name for path in outputs.prepared_dir.glob("*.dim"))
    expected_master_prepared = sorted(
        f"{master.acquisition_date}_{master.product_name}_{iw_swath}.dim" for iw_swath in stack.iw_swaths
    )
    assert remaining_prepared == expected_master_prepared
    assert any(record.checkpoint == "pair_products_validated" for record in outputs.cleanup_records)


def test_run_stack_purges_master_prepared_products_after_each_valid_pair(tmp_path, monkeypatch) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "psi_casablanca_slc_minimal.yaml")
    config.artifact_lifecycle.purge_master_prepared_after_pair = True
    config.artifact_lifecycle.purge_secondary_prepared_after_pair = True
    config.artifact_lifecycle.purge_prepared_after_coreg = False
    config.artifact_lifecycle.purge_snap_intermediates_after_export = False
    runner = SnapGraphRunner(config)
    monkeypatch.setattr(runner, "validate_environment", lambda: None)
    monkeypatch.setattr(runner, "_has_reusable_export", _has_fake_export)
    context = RunContext.create(config, tmp_path)
    context.ensure_directories()
    manifest = _build_manifest()
    stack = config.stacks[0]
    for scene in manifest.scenes:
        zip_path = slc_scene_zip_path(context, manifest.stack_id, scene)
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        zip_path.write_text("placeholder")

    def fake_run_graph(job):
        for key in ("outputFile", "coregOutputFile", "ifgOutputFile"):
            if key in job.parameters:
                _write_fake_dimap(Path(job.parameters[key]))
        if "outputDir" in job.parameters:
            output_dir = Path(job.parameters["outputDir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            for name in ("rslc", "diff0", "geo"):
                subdir = output_dir / name
                subdir.mkdir(exist_ok=True)
                (subdir / "ok.txt").write_text("ok")

    monkeypatch.setattr(runner, "_run_graph", fake_run_graph)

    outputs = runner.run_stack(context, manifest, stack, "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))")

    assert not any(outputs.prepared_dir.glob("*.dim"))
    assert any(
        "master product" in record.reason
        for record in outputs.cleanup_records
        if record.category == "snap_prepared" and record.checkpoint == "pair_products_validated"
    )


def test_run_stack_purges_stale_secondary_prepared_products_before_on_demand_regeneration(tmp_path, monkeypatch) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "psi_casablanca_slc_minimal.yaml")
    config.artifact_lifecycle.purge_master_prepared_after_pair = False
    config.artifact_lifecycle.purge_secondary_prepared_after_pair = True
    config.artifact_lifecycle.purge_prepared_after_coreg = False
    config.artifact_lifecycle.purge_snap_intermediates_after_export = False
    runner = SnapGraphRunner(config)
    monkeypatch.setattr(runner, "validate_environment", lambda: None)
    monkeypatch.setattr(runner, "_has_reusable_export", _has_fake_export)
    context = RunContext.create(config, tmp_path)
    context.ensure_directories()
    manifest = _build_manifest()
    stack = config.stacks[0]
    master = manifest.scenes[2]

    for scene in manifest.scenes:
        zip_path = slc_scene_zip_path(context, manifest.stack_id, scene)
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        zip_path.write_text("placeholder")

    for secondary in [manifest.scenes[0], manifest.scenes[1], manifest.scenes[3], manifest.scenes[4]]:
        _write_fake_dimap(runner._prepared_product(context.snap_dir / manifest.stack_id / "prepared", secondary, "IW1"))
    _write_fake_dimap(runner._prepared_product(context.snap_dir / manifest.stack_id / "prepared", master, "IW1"))

    def fake_run_graph(job):
        for key in ("outputFile", "coregOutputFile", "ifgOutputFile"):
            if key in job.parameters:
                _write_fake_dimap(Path(job.parameters[key]))
        if "outputDir" in job.parameters:
            output_dir = Path(job.parameters["outputDir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            for name in ("rslc", "diff0", "geo"):
                subdir = output_dir / name
                subdir.mkdir(exist_ok=True)
                (subdir / "ok.txt").write_text("ok")

    monkeypatch.setattr(runner, "_run_graph", fake_run_graph)

    outputs = runner.run_stack(context, manifest, stack, "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))")

    remaining_iw1_prepared = sorted(path.name for path in outputs.prepared_dir.glob("*_IW1.dim"))
    assert remaining_iw1_prepared == [f"{master.acquisition_date}_{master.product_name}_IW1.dim"]
    assert any(record.checkpoint == "pair_prepare_on_demand" for record in outputs.cleanup_records)


def test_run_stack_recreates_master_prepared_on_demand_between_pairs(tmp_path, monkeypatch) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "psi_casablanca_slc_minimal.yaml")
    config.artifact_lifecycle.purge_master_prepared_after_pair = True
    config.artifact_lifecycle.purge_secondary_prepared_after_pair = True
    config.artifact_lifecycle.purge_prepared_after_coreg = False
    config.artifact_lifecycle.purge_snap_intermediates_after_export = False
    runner = SnapGraphRunner(config)
    monkeypatch.setattr(runner, "validate_environment", lambda: None)
    monkeypatch.setattr(runner, "_has_reusable_export", _has_fake_export)
    context = RunContext.create(config, tmp_path)
    context.ensure_directories()
    manifest = _build_manifest()
    stack = config.stacks[0]
    master = manifest.scenes[2]
    prepare_outputs: list[str] = []

    for scene in manifest.scenes:
        zip_path = slc_scene_zip_path(context, manifest.stack_id, scene)
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        zip_path.write_text("placeholder")

    def fake_run_graph(job):
        if "outputFile" in job.parameters:
            prepare_outputs.append(Path(job.parameters["outputFile"]).name)
        for key in ("outputFile", "coregOutputFile", "ifgOutputFile"):
            if key in job.parameters:
                _write_fake_dimap(Path(job.parameters[key]))
        if "outputDir" in job.parameters:
            output_dir = Path(job.parameters["outputDir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            for name in ("rslc", "diff0", "geo"):
                subdir = output_dir / name
                subdir.mkdir(exist_ok=True)
                (subdir / "ok.txt").write_text("ok")

    monkeypatch.setattr(runner, "_run_graph", fake_run_graph)

    runner.run_stack(context, manifest, stack, "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))")

    master_prepare_count = sum(name.startswith(f"{master.acquisition_date}_{master.product_name}_") for name in prepare_outputs)
    assert master_prepare_count == len(stack.iw_swaths) * 4


def test_run_stack_removes_obsolete_pair_quarantine_after_active_pair_is_valid(tmp_path, monkeypatch) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "psi_casablanca_slc_minimal.yaml")
    config.artifact_lifecycle.purge_master_prepared_after_pair = False
    config.artifact_lifecycle.purge_secondary_prepared_after_pair = True
    config.artifact_lifecycle.purge_prepared_after_coreg = False
    config.artifact_lifecycle.purge_snap_intermediates_after_export = False
    config.artifact_lifecycle.cleanup_obsolete_snap_backups = True
    runner = SnapGraphRunner(config)
    monkeypatch.setattr(runner, "validate_environment", lambda: None)
    monkeypatch.setattr(runner, "_has_reusable_export", _has_fake_export)
    context = RunContext.create(config, tmp_path)
    context.ensure_directories()
    manifest = _build_manifest()
    stack = config.stacks[0]
    master = manifest.scenes[2]
    secondary = manifest.scenes[1]

    active_coreg = runner._coreg_product(context.snap_dir / manifest.stack_id / "coreg", master, secondary, "IW1")
    active_ifg = runner._ifg_product(context.snap_dir / manifest.stack_id / "interferograms", master, secondary, "IW1")
    _write_fake_dimap(active_coreg)
    _write_fake_dimap(active_ifg)

    backup_root = context.snap_dir / manifest.stack_id / "interrupted_pair_backup_20260324T000000"
    backup_coreg = backup_root / "coreg" / active_coreg.name
    backup_ifg = backup_root / "interferograms" / active_ifg.name
    _write_fake_dimap(backup_coreg)
    _write_fake_dimap(backup_ifg)

    def fake_run_graph(job):
        for key in ("outputFile", "coregOutputFile", "ifgOutputFile"):
            if key in job.parameters:
                _write_fake_dimap(Path(job.parameters[key]))
        if "outputDir" in job.parameters:
            output_dir = Path(job.parameters["outputDir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            for name in ("rslc", "diff0", "geo"):
                subdir = output_dir / name
                subdir.mkdir(exist_ok=True)
                (subdir / "ok.txt").write_text("ok")

    monkeypatch.setattr(runner, "_run_graph", fake_run_graph)

    outputs = runner.run_stack(context, manifest, stack, "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))")

    assert not backup_root.exists()
    assert any(record.category == "snap_quarantine" for record in outputs.cleanup_records)


def test_run_stack_emits_cleanup_records_to_observer(tmp_path, monkeypatch) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "psi_casablanca_slc_minimal.yaml")
    config.artifact_lifecycle.purge_master_prepared_after_pair = True
    config.artifact_lifecycle.purge_secondary_prepared_after_pair = True
    config.artifact_lifecycle.purge_prepared_after_coreg = False
    config.artifact_lifecycle.purge_snap_intermediates_after_export = False
    runner = SnapGraphRunner(config)
    monkeypatch.setattr(runner, "validate_environment", lambda: None)
    monkeypatch.setattr(runner, "_has_reusable_export", _has_fake_export)
    context = RunContext.create(config, tmp_path)
    context.ensure_directories()
    manifest = _build_manifest()
    stack = config.stacks[0]
    observed: list[tuple[str, ...]] = []

    def fake_run_graph(job):
        for key in ("outputFile", "coregOutputFile", "ifgOutputFile"):
            if key in job.parameters:
                _write_fake_dimap(Path(job.parameters[key]))
        if "outputDir" in job.parameters:
            output_dir = Path(job.parameters["outputDir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            for name in ("rslc", "diff0", "geo"):
                subdir = output_dir / name
                subdir.mkdir(exist_ok=True)
                (subdir / "ok.txt").write_text("ok")

    monkeypatch.setattr(runner, "_run_graph", fake_run_graph)

    outputs = runner.run_stack(
        context,
        manifest,
        stack,
        "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))",
        cleanup_observer=lambda records: observed.append(tuple(record.checkpoint for record in records)),
    )

    assert observed
    assert sum(len(batch) for batch in observed) == len(outputs.cleanup_records)


def test_run_stack_purges_snap_intermediates_after_validated_export(tmp_path, monkeypatch) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "psi_casablanca_slc_minimal.yaml")
    config.artifact_lifecycle.purge_prepared_after_coreg = False
    config.artifact_lifecycle.purge_snap_intermediates_after_export = True
    runner = SnapGraphRunner(config)
    monkeypatch.setattr(runner, "validate_environment", lambda: None)
    monkeypatch.setattr(runner, "_has_reusable_export", _has_fake_export)
    context = RunContext.create(config, tmp_path)
    context.ensure_directories()
    manifest = _build_manifest()
    stack = config.stacks[0]

    def fake_run_graph(job):
        for key in ("outputFile", "coregOutputFile", "ifgOutputFile"):
            if key in job.parameters:
                _write_fake_dimap(Path(job.parameters[key]))
        if "outputDir" in job.parameters:
            output_dir = Path(job.parameters["outputDir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            for name in ("rslc", "diff0", "geo"):
                subdir = output_dir / name
                subdir.mkdir(exist_ok=True)
                (subdir / "ok.txt").write_text("ok")

    monkeypatch.setattr(runner, "_run_graph", fake_run_graph)

    outputs = runner.run_stack(context, manifest, stack, "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))")

    assert outputs.stamps_export_dir.exists()
    assert not outputs.prepared_dir.exists()
    assert not outputs.coreg_dir.exists()
    assert not outputs.interferogram_dir.exists()
    assert not (outputs.stack_dir / "stamps_export_inputs").exists()
    assert any(record.category == "snap_intermediate" for record in outputs.cleanup_records)


def test_run_stack_reuses_valid_export_and_downgrades_cleanup_warning(tmp_path, monkeypatch) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "psi_casablanca_slc_minimal.yaml")
    config.artifact_lifecycle.purge_prepared_after_coreg = False
    config.artifact_lifecycle.purge_snap_intermediates_after_export = True
    runner = SnapGraphRunner(config)
    monkeypatch.setattr(runner, "validate_environment", lambda: None)
    monkeypatch.setattr(runner, "_has_reusable_export", _has_fake_export)
    context = RunContext.create(config, tmp_path)
    context.ensure_directories()
    manifest = _build_manifest()
    stack = config.stacks[0]

    stack_dir = context.snap_dir / manifest.stack_id
    prepared_dir = stack_dir / "prepared"
    coreg_dir = stack_dir / "coreg"
    ifg_dir = stack_dir / "interferograms"
    export_inputs_dir = stack_dir / "stamps_export_inputs"
    export_dir = stack_dir / "stamps_export"
    for path in (prepared_dir, coreg_dir, ifg_dir, export_inputs_dir / "coreg_stack_inputs"):
        path.mkdir(parents=True, exist_ok=True)
    (export_inputs_dir / "coreg_stack_inputs" / ".DS_Store").write_text("finder", encoding="utf-8")
    for name in ("rslc", "diff0", "geo"):
        subdir = export_dir / name
        subdir.mkdir(parents=True, exist_ok=True)
        (subdir / "ok.txt").write_text("ok", encoding="utf-8")

    def fake_delete_paths(paths, **kwargs):
        assert kwargs["best_effort"] is True
        assert kwargs["retry_hidden_files"] is True
        kwargs["warning_records"].append(
            CleanupWarning(
                category=kwargs["category"],
                checkpoint=kwargs["checkpoint"],
                reason=kwargs["reason"],
                path=list(paths)[-1],
                message="OSError: [Errno 66] Directory not empty",
            )
        )
        return []

    monkeypatch.setattr("casablanca_psi.snap.delete_paths", fake_delete_paths)

    outputs = runner.run_stack(context, manifest, stack, "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))")

    assert outputs.stamps_export_dir.exists()
    assert outputs.cleanup_records == ()
    assert len(outputs.cleanup_warnings) == 1
    assert outputs.cleanup_warnings[0].checkpoint == "stamps_export_validated"


def test_run_stack_skips_prepare_when_pair_products_already_cover_swath(tmp_path, monkeypatch) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "psi_casablanca_slc_minimal.yaml")
    config.artifact_lifecycle.purge_prepared_after_coreg = True
    config.artifact_lifecycle.purge_snap_intermediates_after_export = False
    runner = SnapGraphRunner(config)
    monkeypatch.setattr(runner, "validate_environment", lambda: None)
    monkeypatch.setattr(runner, "_has_reusable_export", _has_fake_export)
    context = RunContext.create(config, tmp_path)
    context.ensure_directories()
    manifest = _build_manifest()
    stack = config.stacks[0]
    master = manifest.scenes[2]
    secondaries = [manifest.scenes[0], manifest.scenes[1], manifest.scenes[3], manifest.scenes[4]]
    graph_calls: list[tuple[str, str | None]] = []

    for secondary in secondaries:
        _write_fake_dimap(runner._coreg_product(context.snap_dir / manifest.stack_id / "coreg", master, secondary, "IW1"))
        _write_fake_dimap(runner._ifg_product(context.snap_dir / manifest.stack_id / "interferograms", master, secondary, "IW1"))

    def fake_run_graph(job):
        graph_calls.append((job.graph_path.name, job.parameters.get("subswath")))
        if "outputDir" in job.parameters:
            output_dir = Path(job.parameters["outputDir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            for name in ("rslc", "diff0", "geo"):
                subdir = output_dir / name
                subdir.mkdir(exist_ok=True)
                (subdir / "ok.txt").write_text("ok")
        else:
            for key in ("outputFile", "coregOutputFile", "ifgOutputFile"):
                if key in job.parameters:
                    _write_fake_dimap(Path(job.parameters[key]))

    monkeypatch.setattr(runner, "_run_graph", fake_run_graph)

    runner.run_stack(context, manifest, stack, "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))")

    prepare_swaths = [subswath for graph_name, subswath in graph_calls if graph_name == "prepare_slc_stack.xml"]
    assert "IW1" not in prepare_swaths
    assert "IW2" in prepare_swaths
    assert "IW3" in prepare_swaths


def test_run_stack_builds_two_product_stamps_export_inputs(tmp_path, monkeypatch) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "psi_casablanca_slc_minimal.yaml")
    config.artifact_lifecycle.purge_prepared_after_coreg = False
    config.artifact_lifecycle.purge_snap_intermediates_after_export = False
    runner = SnapGraphRunner(config)
    monkeypatch.setattr(runner, "validate_environment", lambda: None)
    monkeypatch.setattr(runner, "_has_reusable_export", _has_fake_export)
    context = RunContext.create(config, tmp_path)
    context.ensure_directories()
    manifest = _build_manifest()
    stack = config.stacks[0]
    graph_calls: list[tuple[str, dict[str, str]]] = []

    def fake_run_graph(job):
        graph_calls.append((job.graph_path.name, dict(job.parameters)))
        graph_name = job.graph_path.name
        if graph_name == "merge_product_set.xml":
            output = Path(job.parameters["outputFile"])
            secondary_date = output.stem.split("_")[1]
            _write_fake_dimap(output, band_names=_fake_dual_pol_coreg_bands(secondary_date))
        elif graph_name == "select_export_bands.xml":
            output = Path(job.parameters["outputFile"])
            _write_fake_dimap(output, band_names=job.parameters["sourceBands"].split(","))
        elif graph_name == "derive_ifg_from_coreg_stack.xml":
            output = Path(job.parameters["outputFile"])
            secondary_date = output.stem.split("_")[1]
            _write_fake_dimap(output, band_names=_fake_export_ifg_bands(secondary_date))
        elif graph_name == "create_stack_product.xml":
            output = Path(job.parameters["outputFile"])
            if output.name.endswith("_coreg_final.dim"):
                _write_fake_dimap(output, band_names=_fake_final_coreg_bands())
            else:
                _write_fake_dimap(output)
        elif graph_name == "band_merge_product_set.xml":
            output = Path(job.parameters["outputFile"])
            if output.name.endswith("_coreg_final.dim"):
                _write_fake_dimap(output, band_names=_fake_final_coreg_bands())
            else:
                _write_fake_dimap(output, band_names=_fake_final_ifg_bands())
        else:
            for key in ("outputFile", "coregOutputFile", "ifgOutputFile"):
                if key in job.parameters:
                    _write_fake_dimap(Path(job.parameters[key]))
        if "outputDir" in job.parameters:
            output_dir = Path(job.parameters["outputDir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            for name in ("rslc", "diff0", "geo"):
                subdir = output_dir / name
                subdir.mkdir(exist_ok=True)
                (subdir / "ok.txt").write_text("ok")

    monkeypatch.setattr(runner, "_run_graph", fake_run_graph)

    outputs = runner.run_stack(context, manifest, stack, "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))")

    merge_calls = [params for graph_name, params in graph_calls if graph_name == "merge_product_set.xml"]
    create_stack_calls = [params for graph_name, params in graph_calls if graph_name == "create_stack_product.xml"]
    band_merge_calls = [params for graph_name, params in graph_calls if graph_name == "band_merge_product_set.xml"]
    select_calls = [params for graph_name, params in graph_calls if graph_name == "select_export_bands.xml"]
    derive_ifg_calls = [params for graph_name, params in graph_calls if graph_name == "derive_ifg_from_coreg_stack.xml"]
    export_calls = [params for graph_name, params in graph_calls if graph_name == "stamps_export.xml"]

    assert len(merge_calls) == 4
    coreg_export_calls = [params for params in select_calls if params["outputFile"].endswith("_coreg_export.dim")]
    ifg_stack_input_calls = [params for params in select_calls if params["outputFile"].endswith("_ifg_stack_input.dim")]

    assert len(select_calls) == 7
    assert len(coreg_export_calls) == 4
    assert len(ifg_stack_input_calls) == 3
    assert len(derive_ifg_calls) == 4
    assert len(create_stack_calls) == 1
    assert len(band_merge_calls) == 1
    assert len(export_calls) == 1
    final_coreg_stack = next(params for params in create_stack_calls if params["outputFile"].endswith("_coreg_final.dim"))
    final_ifg_stack = next(params for params in band_merge_calls if params["outputFile"].endswith("_ifg_final.dim"))
    final_coreg_sources = final_coreg_stack["fileList"].split(",")
    final_ifg_sources = final_ifg_stack["fileList"].split(",")
    assert len(final_coreg_sources) == 4
    assert len(final_ifg_sources) == 4
    assert final_coreg_sources[0].endswith("_coreg_export.dim")
    assert all(path.endswith("_coreg_export.dim") for path in final_coreg_sources[1:])
    assert final_coreg_stack["masterBands"] == ""
    assert final_coreg_stack["sourceBands"] == ""
    assert final_ifg_sources[0].endswith("_ifg_export.dim")
    assert all(path.endswith("_ifg_stack_input.dim") for path in final_ifg_sources[1:])
    assert export_calls[0]["coregFile"].endswith("_coreg_final.dim")
    assert export_calls[0]["ifgFile"].endswith("_ifg_final.dim")

    final_coreg_product = outputs.stack_dir / "stamps_export_inputs" / "final_products" / "2023-09-22_coreg_final.dim"
    final_ifg_product = outputs.stack_dir / "stamps_export_inputs" / "final_products" / "2023-09-22_ifg_final.dim"
    assert runner._product_band_names(final_coreg_product) == [
        "i_VV_mst_22Sep2023",
        "q_VV_mst_22Sep2023",
        "i_VV_slv1_29Aug2023",
        "q_VV_slv1_29Aug2023",
        "i_VV_slv1_10Sep2023",
        "q_VV_slv1_10Sep2023",
        "i_VV_slv1_04Oct2023",
        "q_VV_slv1_04Oct2023",
        "i_VV_slv1_16Oct2023",
        "q_VV_slv1_16Oct2023",
    ]
    assert runner._product_band_names(final_ifg_product) == [
        "i_ifg_VV_22Sep2023_29Aug2023",
        "q_ifg_VV_22Sep2023_29Aug2023",
        "elevation",
        "orthorectifiedLat",
        "orthorectifiedLon",
        "i_ifg_VV_22Sep2023_10Sep2023",
        "q_ifg_VV_22Sep2023_10Sep2023",
        "i_ifg_VV_22Sep2023_04Oct2023",
        "q_ifg_VV_22Sep2023_04Oct2023",
        "i_ifg_VV_22Sep2023_16Oct2023",
        "q_ifg_VV_22Sep2023_16Oct2023",
    ]
    assert not (outputs.stack_dir / "stamps_export_inputs" / "stack_products").exists()


def test_final_coreg_contract_validation_rejects_repeated_master_aliases(tmp_path) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "psi_casablanca_slc_minimal.yaml")
    runner = SnapGraphRunner(config)
    manifest = _build_manifest()
    master = manifest.scenes[2]
    secondaries = [manifest.scenes[0], manifest.scenes[1], manifest.scenes[3], manifest.scenes[4]]
    invalid_product = tmp_path / "coreg_final.dim"
    _write_fake_dimap(
        invalid_product,
        band_names=[
            "i_VV_mst_22Sep2023",
            "q_VV_mst_22Sep2023",
            "i_VV_slv1_29Aug2023",
            "q_VV_slv1_29Aug2023",
            "i_VV_mst_22Sep2023_slv2_22Sep2023",
            "q_VV_mst_22Sep2023_slv2_22Sep2023",
            "i_VV_slv1_10Sep2023_slv3_22Sep2023",
            "q_VV_slv1_10Sep2023_slv3_22Sep2023",
            "i_VV_mst_22Sep2023_slv4_22Sep2023",
            "q_VV_mst_22Sep2023_slv4_22Sep2023",
            "i_VV_slv1_04Oct2023_slv5_22Sep2023",
            "q_VV_slv1_04Oct2023_slv5_22Sep2023",
            "i_VV_mst_22Sep2023_slv6_22Sep2023",
            "q_VV_mst_22Sep2023_slv6_22Sep2023",
            "i_VV_slv1_16Oct2023_slv7_22Sep2023",
            "q_VV_slv1_16Oct2023_slv7_22Sep2023",
        ],
    )

    with pytest.raises(RuntimeError, match="invalid contract"):
        runner._validate_final_coreg_product_contract(
            product=invalid_product,
            master_scene=master,
            secondaries=secondaries,
            polarization="VV",
        )


def test_selected_final_coreg_stack_band_names_prune_and_normalize_to_one_master_and_one_slave_per_date(tmp_path) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "psi_casablanca_slc_minimal.yaml")
    runner = SnapGraphRunner(config)
    manifest = _build_manifest()
    master = manifest.scenes[2]
    secondaries = [manifest.scenes[0], manifest.scenes[1], manifest.scenes[3], manifest.scenes[4]]
    export_dir = tmp_path / "coreg_pairs_export"
    source_products: list[Path] = []
    for secondary in secondaries:
        source_product = export_dir / f"{master.acquisition_date}_{secondary.acquisition_date}_coreg_export.dim"
        _write_fake_dimap(source_product)
        source_products.append(source_product)
    product = tmp_path / "coreg_create_stack_full.dim"
    _write_fake_dimap(
        product,
        band_names=[
            "i_VV_mst_22Sep2023",
            "q_VV_mst_22Sep2023",
            "i_VV_slv1_29Aug2023",
            "q_VV_slv1_29Aug2023",
            "i_VV_mst_22Sep2023_slv2_22Sep2023",
            "q_VV_mst_22Sep2023_slv2_22Sep2023",
            "i_VV_slv1_10Sep2023_slv3_22Sep2023",
            "q_VV_slv1_10Sep2023_slv3_22Sep2023",
            "i_VV_mst_22Sep2023_slv4_22Sep2023",
            "q_VV_mst_22Sep2023_slv4_22Sep2023",
            "i_VV_slv1_04Oct2023_slv5_22Sep2023",
            "q_VV_slv1_04Oct2023_slv5_22Sep2023",
            "i_VV_mst_22Sep2023_slv6_22Sep2023",
            "q_VV_mst_22Sep2023_slv6_22Sep2023",
            "i_VV_slv1_16Oct2023_slv7_22Sep2023",
            "q_VV_slv1_16Oct2023_slv7_22Sep2023",
        ],
    )

    keep_band_names = runner._selected_final_coreg_stack_band_names(
        product=product,
        master_scene=master,
        secondaries=secondaries,
        polarization="VV",
    )

    runner._prune_dimap_product_bands(product=product, keep_band_names=keep_band_names)
    runner._normalize_final_stack_product_band_names(product=product, product_kind="coreg")
    runner._repair_final_coreg_slave_metadata(
        product=product,
        source_products=source_products,
        master_scene=master,
        secondaries=secondaries,
        polarization="VV",
    )

    runner._validate_final_coreg_product_contract(
        product=product,
        master_scene=master,
        secondaries=secondaries,
        polarization="VV",
    )


def test_final_coreg_contract_validation_rejects_missing_slave_metadata(tmp_path) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "psi_casablanca_slc_minimal.yaml")
    runner = SnapGraphRunner(config)
    manifest = _build_manifest()
    master = manifest.scenes[2]
    secondaries = [manifest.scenes[0], manifest.scenes[1], manifest.scenes[3], manifest.scenes[4]]
    product = tmp_path / "coreg_final.dim"
    _write_fake_dimap(product, band_names=_fake_final_coreg_bands())
    product.write_text(
        product.read_text(encoding="utf-8").replace('<MDElem name="Secondary_10Sep2023" />', ""),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="missing secondary dates"):
        runner._validate_final_coreg_product_contract(
            product=product,
            master_scene=master,
            secondaries=secondaries,
            polarization="VV",
        )


def test_final_coreg_contract_validation_rejects_slave_metadata_date_mismatch(tmp_path) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "psi_casablanca_slc_minimal.yaml")
    runner = SnapGraphRunner(config)
    manifest = _build_manifest()
    master = manifest.scenes[2]
    secondaries = [manifest.scenes[0], manifest.scenes[1], manifest.scenes[3], manifest.scenes[4]]
    product = tmp_path / "coreg_final.dim"
    _write_fake_dimap(product, band_names=_fake_final_coreg_bands())

    tree = ET.parse(product)
    child = tree.getroot().find(".//MDElem[@name='Slave_Metadata']/MDElem[@name='slave_10Sep2023']")
    assert child is not None
    child.find("./MDATTR[@name='first_line_time']").text = "22-SEP-2023 00:00:00.000000"
    tree.write(product, encoding="utf-8")

    with pytest.raises(RuntimeError, match="acquisition date does not match"):
        runner._validate_final_coreg_product_contract(
            product=product,
            master_scene=master,
            secondaries=secondaries,
            polarization="VV",
        )


def test_repair_final_coreg_baseline_metadata_merges_top_level_baselines(tmp_path) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "psi_casablanca_slc_minimal.yaml")
    runner = SnapGraphRunner(config)
    manifest = _build_manifest()
    master = manifest.scenes[2]
    secondaries = [manifest.scenes[0], manifest.scenes[1], manifest.scenes[3], manifest.scenes[4]]
    export_dir = tmp_path / "coreg_pairs_export"
    source_products: list[Path] = []
    for secondary in secondaries:
        product = export_dir / f"{master.acquisition_date}_{secondary.acquisition_date}_coreg_export.dim"
        _write_fake_dimap(product)
        source_products.append(product)

    final_product = tmp_path / "coreg_final.dim"
    _write_fake_dimap(final_product, band_names=_fake_final_coreg_bands())
    tree = ET.parse(final_product)
    baselines = tree.getroot().find(".//MDElem[@name='Abstracted_Metadata']/MDElem[@name='Baselines']")
    assert baselines is not None
    for child in list(baselines):
        baselines.remove(child)
    baselines.append(
        ET.fromstring(
            "<MDElem name=\"Ref_22Sep2023\">"
            "<MDElem name=\"Secondary_22Sep2023\" />"
            "<MDElem name=\"Secondary_29Aug2023\" />"
            "</MDElem>"
        )
    )
    baselines.append(
        ET.fromstring(
            "<MDElem name=\"Ref_29Aug2023\">"
            "<MDElem name=\"Secondary_22Sep2023\" />"
            "<MDElem name=\"Secondary_29Aug2023\" />"
            "</MDElem>"
        )
    )
    tree.write(final_product, encoding="utf-8")

    runner._repair_final_coreg_baseline_metadata(
        product=final_product,
        source_products=source_products,
        master_scene=master,
        secondaries=secondaries,
    )

    repaired_root = ET.parse(final_product).getroot()
    reference = repaired_root.find(".//MDElem[@name='Baselines']/MDElem[@name='Ref_22Sep2023']")
    assert reference is not None
    assert [child.attrib.get("name") for child in reference.findall("./MDElem")] == [
        "Secondary_22Sep2023",
        "Secondary_29Aug2023",
        "Secondary_10Sep2023",
        "Secondary_04Oct2023",
        "Secondary_16Oct2023",
    ]
    runner._validate_final_coreg_product_contract(
        product=final_product,
        master_scene=master,
        secondaries=secondaries,
        polarization="VV",
    )


def test_repair_final_coreg_slave_metadata_replaces_master_date_children(tmp_path) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "psi_casablanca_slc_minimal.yaml")
    runner = SnapGraphRunner(config)
    manifest = _build_manifest()
    master = manifest.scenes[2]
    secondaries = [manifest.scenes[0], manifest.scenes[1], manifest.scenes[3], manifest.scenes[4]]
    export_dir = tmp_path / "coreg_pairs_export"
    source_products: list[Path] = []
    for secondary in secondaries:
        product = export_dir / f"{master.acquisition_date}_{secondary.acquisition_date}_coreg_export.dim"
        _write_fake_dimap(product)
        source_products.append(product)

    final_product = tmp_path / "coreg_final.dim"
    _write_fake_dimap(
        final_product,
        band_names=[
            "i_VV_mst_22Sep2023",
            "q_VV_mst_22Sep2023",
            "i_VV_slv1_29Aug2023",
            "q_VV_slv1_29Aug2023",
            "i_VV_slv3_10Sep2023",
            "q_VV_slv3_10Sep2023",
            "i_VV_slv5_04Oct2023",
            "q_VV_slv5_04Oct2023",
            "i_VV_slv7_16Oct2023",
            "q_VV_slv7_16Oct2023",
        ],
    )
    tree = ET.parse(final_product)
    slave_metadata = tree.getroot().find(".//MDElem[@name='Slave_Metadata']")
    assert slave_metadata is not None
    slave_metadata.find("./MDATTR[@name='Master_bands']").text = (
        "i_VV_mst_22Sep2023 q_VV_mst_22Sep2023 i_VV_slv1_29Aug2023 q_VV_slv1_29Aug2023"
    )
    for name, bad_bands in {
        "slave_10Sep2023": "i_VV_mst_22Sep2023_slv2_22Sep2023 q_VV_mst_22Sep2023_slv2_22Sep2023 i_VV_slv3_10Sep2023 q_VV_slv3_10Sep2023",
        "slave_04Oct2023": "i_VV_mst_22Sep2023_slv4_22Sep2023 q_VV_mst_22Sep2023_slv4_22Sep2023 i_VV_slv5_04Oct2023 q_VV_slv5_04Oct2023",
        "slave_16Oct2023": "i_VV_mst_22Sep2023_slv6_22Sep2023 q_VV_mst_22Sep2023_slv6_22Sep2023 i_VV_slv7_16Oct2023 q_VV_slv7_16Oct2023",
    }.items():
        child = slave_metadata.find(f"./MDElem[@name='{name}']")
        assert child is not None
        child.find("./MDATTR[@name='PRODUCT']").text = "FAKE_22Sep2023"
        child.find("./MDATTR[@name='first_line_time']").text = "22-SEP-2023 00:00:00.000000"
        child.find("./MDATTR[@name='Slave_bands']").text = bad_bands
        child.find("./MDElem[@name='Band_IW1_VV']/MDATTR[@name='band_names']").text = bad_bands
    tree.write(final_product, encoding="utf-8")

    runner._repair_final_coreg_slave_metadata(
        product=final_product,
        source_products=source_products,
        master_scene=master,
        secondaries=secondaries,
        polarization="VV",
    )

    repaired_root = ET.parse(final_product).getroot()
    repaired_slave_metadata = repaired_root.find(".//MDElem[@name='Slave_Metadata']")
    assert repaired_slave_metadata is not None
    assert repaired_slave_metadata.find("./MDATTR[@name='Master_bands']").text == (
        "i_VV_mst_22Sep2023 q_VV_mst_22Sep2023"
    )
    child = repaired_slave_metadata.find("./MDElem[@name='slave_10Sep2023']")
    assert child is not None
    assert child.find("./MDATTR[@name='PRODUCT']").text == "FAKE_10Sep2023"
    assert child.find("./MDATTR[@name='first_line_time']").text == "10-SEP-2023 00:00:00.000000"
    assert child.find("./MDATTR[@name='Slave_bands']").text == "i_VV_slv3_10Sep2023 q_VV_slv3_10Sep2023"
    assert child.find("./MDElem[@name='Band_IW1_VV']/MDATTR[@name='band_names']").text == (
        "i_VV_slv3_10Sep2023 q_VV_slv3_10Sep2023"
    )

    runner._validate_final_coreg_product_contract(
        product=final_product,
        master_scene=master,
        secondaries=secondaries,
        polarization="VV",
    )


def test_create_stack_graph_uses_cli_supported_parameters() -> None:
    graph = Path("resources/snap_graphs/create_stack_product.xml").read_text(encoding="utf-8")

    assert "<initialOffsetMethod>Orbit</initialOffsetMethod>" in graph
    assert "<masterBands>${masterBands}</masterBands>" in graph
    assert "<sourceBands>${sourceBands}</sourceBands>" in graph
    assert "includeMaster" not in graph


def test_stamps_export_graph_reads_final_assembled_products() -> None:
    graph = Path("resources/snap_graphs/stamps_export.xml").read_text(encoding="utf-8")

    assert "<file>${coregFile}</file>" in graph
    assert "<file>${ifgFile}</file>" in graph
    assert "<node id=\"ReadCoregProduct\">" in graph
    assert "<node id=\"ReadIfgProduct\">" in graph
    assert "<sourceProduct refid=\"ReadCoregProduct\" />" in graph
    assert "<sourceProduct.1 refid=\"ReadIfgProduct\" />" in graph
    assert "CreateCoregStack" not in graph
    assert "CreateIfgStack" not in graph


def test_run_stack_purges_pair_products_after_merged_coreg_checkpoint(tmp_path, monkeypatch) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "psi_casablanca_slc_minimal.yaml")
    config.artifact_lifecycle.purge_pair_products_after_merged_coreg = True
    config.artifact_lifecycle.purge_prepared_after_coreg = False
    config.artifact_lifecycle.purge_snap_intermediates_after_export = False
    runner = SnapGraphRunner(config)
    monkeypatch.setattr(runner, "validate_environment", lambda: None)
    monkeypatch.setattr(runner, "_has_reusable_export", _has_fake_export)
    context = RunContext.create(config, tmp_path)
    context.ensure_directories()
    manifest = _build_manifest()
    stack = config.stacks[0]
    master = manifest.scenes[2]
    first_secondary = manifest.scenes[0]

    def fake_run_graph(job):
        graph_name = job.graph_path.name
        if graph_name == "merge_product_set.xml":
            output = Path(job.parameters["outputFile"])
            secondary_date = output.stem.split("_")[1]
            _write_fake_dimap(output, band_names=_fake_dual_pol_coreg_bands(secondary_date))
        elif graph_name == "select_export_bands.xml":
            output = Path(job.parameters["outputFile"])
            _write_fake_dimap(output, band_names=job.parameters["sourceBands"].split(","))
        elif graph_name == "derive_ifg_from_coreg_stack.xml":
            output = Path(job.parameters["outputFile"])
            secondary_date = output.stem.split("_")[1]
            _write_fake_dimap(output, band_names=_fake_export_ifg_bands(secondary_date))
        elif graph_name == "create_stack_product.xml":
            output = Path(job.parameters["outputFile"])
            if output.name.endswith("_coreg_final.dim"):
                _write_fake_dimap(output, band_names=_fake_final_coreg_bands())
            else:
                _write_fake_dimap(output)
        else:
            for key in ("outputFile", "coregOutputFile", "ifgOutputFile"):
                if key in job.parameters:
                    _write_fake_dimap(Path(job.parameters[key]))
        if "outputDir" in job.parameters:
            output_dir = Path(job.parameters["outputDir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            for name in ("rslc", "diff0", "geo"):
                subdir = output_dir / name
                subdir.mkdir(exist_ok=True)
                (subdir / "ok.txt").write_text("ok")

    monkeypatch.setattr(runner, "_run_graph", fake_run_graph)

    outputs = runner.run_stack(context, manifest, stack, "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))")

    for iw_swath in stack.iw_swaths:
        assert not runner._coreg_product(outputs.coreg_dir, master, first_secondary, iw_swath).exists()
        assert not runner._ifg_product(outputs.interferogram_dir, master, first_secondary, iw_swath).exists()
    assert not (
        outputs.stack_dir / "stamps_export_inputs" / "coreg_pairs_merged" / f"{master.acquisition_date}_{first_secondary.acquisition_date}_coreg_merged.dim"
    ).exists()
    assert (
        outputs.stack_dir / "stamps_export_inputs" / "final_products" / f"{master.acquisition_date}_coreg_final.dim"
    ).exists()
    assert any(record.checkpoint == "merged_coreg_pair_validated" for record in outputs.cleanup_records)


def test_run_stack_reuses_export_checkpoint_without_rebuilding_pair_products(tmp_path, monkeypatch) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "psi_casablanca_slc_minimal.yaml")
    config.artifact_lifecycle.purge_pair_products_after_merged_coreg = True
    config.artifact_lifecycle.purge_prepared_after_coreg = False
    config.artifact_lifecycle.purge_snap_intermediates_after_export = False
    runner = SnapGraphRunner(config)
    monkeypatch.setattr(runner, "validate_environment", lambda: None)
    monkeypatch.setattr(runner, "_has_reusable_export", _has_fake_export)
    context = RunContext.create(config, tmp_path)
    context.ensure_directories()
    manifest = _build_manifest()
    stack = config.stacks[0]
    master = manifest.scenes[2]
    secondaries = [manifest.scenes[0], manifest.scenes[1], manifest.scenes[3], manifest.scenes[4]]
    assembly_dir = context.snap_dir / manifest.stack_id / "stamps_export_inputs" / "coreg_pairs_merged"
    reused_secondary = secondaries[0]
    _write_fake_dimap(
        runner._merged_coreg_product(assembly_dir, master, reused_secondary),
        band_names=_fake_dual_pol_coreg_bands(reused_secondary.acquisition_date),
    )

    for secondary in secondaries[1:]:
        for iw_swath in stack.iw_swaths:
            _write_fake_dimap(runner._coreg_product(context.snap_dir / manifest.stack_id / "coreg", master, secondary, iw_swath))
            _write_fake_dimap(runner._ifg_product(context.snap_dir / manifest.stack_id / "interferograms", master, secondary, iw_swath))

    graph_names: list[str] = []

    def fake_run_graph(job):
        graph_names.append(job.graph_path.name)
        if job.graph_path.name in {"prepare_slc_stack.xml", "coregister_stack.xml", "coregister_stack_no_esd.xml"}:
            raise AssertionError(f"Unexpected pair-regeneration graph call: {job.graph_path.name}")
        graph_name = job.graph_path.name
        if graph_name == "merge_product_set.xml":
            output = Path(job.parameters["outputFile"])
            secondary_date = output.stem.split("_")[1]
            _write_fake_dimap(output, band_names=_fake_dual_pol_coreg_bands(secondary_date))
        elif graph_name == "select_export_bands.xml":
            output = Path(job.parameters["outputFile"])
            _write_fake_dimap(output, band_names=job.parameters["sourceBands"].split(","))
        elif graph_name == "derive_ifg_from_coreg_stack.xml":
            output = Path(job.parameters["outputFile"])
            secondary_date = output.stem.split("_")[1]
            _write_fake_dimap(output, band_names=_fake_export_ifg_bands(secondary_date))
        elif graph_name == "create_stack_product.xml":
            output = Path(job.parameters["outputFile"])
            if output.name.endswith("_coreg_final.dim"):
                _write_fake_dimap(output, band_names=_fake_final_coreg_bands())
            else:
                _write_fake_dimap(output)
        else:
            for key in ("outputFile", "coregOutputFile", "ifgOutputFile"):
                if key in job.parameters:
                    _write_fake_dimap(Path(job.parameters[key]))
        if "outputDir" in job.parameters:
            output_dir = Path(job.parameters["outputDir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            for name in ("rslc", "diff0", "geo"):
                subdir = output_dir / name
                subdir.mkdir(exist_ok=True)
                (subdir / "ok.txt").write_text("ok")

    monkeypatch.setattr(runner, "_run_graph", fake_run_graph)

    runner.run_stack(context, manifest, stack, "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))")

    assert "merge_product_set.xml" in graph_names
    assert "select_export_bands.xml" in graph_names
    assert "derive_ifg_from_coreg_stack.xml" in graph_names
    assert "create_stack_product.xml" in graph_names
    assert "band_merge_product_set.xml" in graph_names
    assert "stamps_export.xml" in graph_names


def test_cleanup_invalid_export_checkpoint_state_removes_bad_export_only(tmp_path) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "psi_casablanca_slc_minimal.yaml")
    runner = SnapGraphRunner(config)
    export_dir = tmp_path / "snap" / "asc_rel147_vv" / "stamps_export"
    for name in ("rslc", "diff0", "geo"):
        (export_dir / name).mkdir(parents=True, exist_ok=True)
        (export_dir / name / "ok.txt").write_text("ok")
    (export_dir / "diff0" / "20230922_20230829.diff").write_text("ok")
    (export_dir / "diff0" / "20230922_20230829.base").write_text(
        "initial_baseline(TCN): 0.0000000 NaN NaN\n"
        "initial_baseline_rate: 0.0000000 NaN NaN\n",
        encoding="utf-8",
    )

    export_inputs_dir = tmp_path / "snap" / "asc_rel147_vv" / "stamps_export_inputs"
    _write_fake_dimap(export_inputs_dir / "coreg_stack_inputs" / "stale_coreg.dim")
    _write_fake_dimap(export_inputs_dir / "ifg_stack_inputs" / "stale_ifg.dim")
    _write_fake_dimap(export_inputs_dir / "final_products" / "stale_final.dim")
    _write_fake_dimap(export_inputs_dir / "coreg_pairs_export" / "keep_coreg.dim")

    records = runner._cleanup_invalid_export_checkpoint_state(export_dir, export_inputs_dir)

    assert not export_dir.exists()
    assert (export_inputs_dir / "coreg_stack_inputs" / "stale_coreg.dim").exists()
    assert (export_inputs_dir / "ifg_stack_inputs" / "stale_ifg.dim").exists()
    assert (export_inputs_dir / "final_products" / "stale_final.dim").exists()
    assert (export_inputs_dir / "coreg_pairs_export" / "keep_coreg.dim").exists()
    assert any(record.checkpoint == "invalid_stamps_export_detected" for record in records)
