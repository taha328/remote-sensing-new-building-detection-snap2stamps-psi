from __future__ import annotations

from pathlib import Path
import typer

from aoi_builtup.evaluation import evaluate_run
from aoi_builtup.io import read_json, write_json
from aoi_builtup.pipeline import run_pipeline
from aoi_builtup.runtime import PipelineInterruptedError

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _validate_optional_file_path(path: Path | None, *, must_exist: bool, dir_okay: bool = True) -> Path | None:
    if path is None:
        return None
    if must_exist and not path.exists():
        raise typer.BadParameter(f"Path does not exist: {path}")
    if not dir_okay and path.is_dir():
        raise typer.BadParameter(f"Expected a file path, got directory: {path}")
    return path


def _run(
    config: Path,
    stop_after: str | None,
    run_dir: Path | None,
    resume_latest: bool,
) -> None:
    try:
        run_pipeline(
            config,
            stop_after=stop_after,
            run_dir=run_dir,
            resume_latest=resume_latest,
        )
    except (KeyboardInterrupt, PipelineInterruptedError):
        raise typer.Exit(code=130) from None


@app.command("acquire-data")
def acquire_data(
    config: Path = typer.Option(..., exists=True, readable=True),
    run_dir: Path = typer.Option(None),
    resume_latest: bool = typer.Option(False),
) -> None:
    _run(config, "acquire", _validate_optional_file_path(run_dir, must_exist=False), resume_latest)


@app.command("build-composites")
def build_composites(
    config: Path = typer.Option(..., exists=True, readable=True),
    run_dir: Path = typer.Option(None),
    resume_latest: bool = typer.Option(False),
) -> None:
    _run(config, "build_composites", _validate_optional_file_path(run_dir, must_exist=False), resume_latest)


@app.command("detect-s1")
def detect_s1(
    config: Path = typer.Option(..., exists=True, readable=True),
    run_dir: Path = typer.Option(None),
    resume_latest: bool = typer.Option(False),
) -> None:
    _run(config, "detect_s1", _validate_optional_file_path(run_dir, must_exist=False), resume_latest)


@app.command("refine-s2")
def refine_s2(
    config: Path = typer.Option(..., exists=True, readable=True),
    run_dir: Path = typer.Option(None),
    resume_latest: bool = typer.Option(False),
) -> None:
    _run(config, "refine_s2", _validate_optional_file_path(run_dir, must_exist=False), resume_latest)


@app.command("polygonize")
def polygonize(
    config: Path = typer.Option(..., exists=True, readable=True),
    run_dir: Path = typer.Option(None),
    resume_latest: bool = typer.Option(False),
) -> None:
    _run(config, "polygonize", _validate_optional_file_path(run_dir, must_exist=False), resume_latest)


@app.command("export")
def export_outputs(
    config: Path = typer.Option(..., exists=True, readable=True),
    run_dir: Path = typer.Option(None),
    resume_latest: bool = typer.Option(False),
) -> None:
    _run(config, "export", _validate_optional_file_path(run_dir, must_exist=False), resume_latest)


@app.command("run-pipeline")
def run_pipeline_command(
    config: Path = typer.Option(..., exists=True, readable=True),
    run_dir: Path = typer.Option(None),
    resume_latest: bool = typer.Option(False),
) -> None:
    _run(config, None, _validate_optional_file_path(run_dir, must_exist=False), resume_latest)


@app.command("evaluate-run")
def evaluate_run_command(
    run_dir: Path = typer.Option(..., exists=True, file_okay=False),
    reference_raster: Path = typer.Option(..., exists=True, dir_okay=False, readable=True),
    labels_raster: Path = typer.Option(None),
    local_raster_name: str = typer.Option("cumulative_refined.tif"),
) -> None:
    result = evaluate_run(
        run_dir=run_dir,
        reference_raster=reference_raster,
        labels_raster=_validate_optional_file_path(labels_raster, must_exist=True, dir_okay=False),
        local_raster_name=local_raster_name,
    )
    report_path = run_dir / "reports" / "run_report.json"
    if report_path.exists():
        report = read_json(report_path)
        report["evaluation"] = result
        write_json(report, report_path)


if __name__ == "__main__":
    app()
