from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from casablanca_builtup.io import read_vector
from casablanca_psi.acquisition import validate_s1_slc_provider_fields
from casablanca_psi.config import load_config
from casablanca_psi.evaluation import (
    evaluate_points_against_reference,
    evaluate_polygons_against_reference,
    write_evaluation,
)
from casablanca_psi.pipeline import run_pipeline

app = typer.Typer(add_completion=False, no_args_is_help=True, pretty_exceptions_show_locals=False)


def _run(config: Path, stop_after: str | None, run_dir: Path | None, resume_latest: bool) -> None:
    run_pipeline(config, stop_after=stop_after, run_dir=run_dir, resume_latest=resume_latest)


@app.command("validate-provider")
def validate_provider(
    config: Path = typer.Option(..., exists=True, readable=True),
) -> None:
    validation = validate_s1_slc_provider_fields(load_config(config))
    typer.echo(f"catalog_url={validation.catalog_url}")
    typer.echo(f"collection={validation.collection}")
    typer.echo(f"item_id={validation.item_id}")
    for key, value in validation.property_mapping.items():
        typer.echo(f"{key}={value}")
    typer.echo(f"asset_name={validation.asset_mapping['asset_name']}")
    typer.echo(f"asset_href={validation.asset_mapping['asset_href']}")


@app.command("run-pipeline")
def run_pipeline_command(
    config: Path = typer.Option(..., exists=True, readable=True),
    run_dir: Path = typer.Option(None),
    resume_latest: bool = typer.Option(False),
) -> None:
    _run(config, None, run_dir, resume_latest)


@app.command("acquire")
def acquire(
    config: Path = typer.Option(..., exists=True, readable=True),
    run_dir: Path = typer.Option(None),
    resume_latest: bool = typer.Option(False),
) -> None:
    _run(config, "acquire", run_dir, resume_latest)


@app.command("download-slc")
def download_slc(
    config: Path = typer.Option(..., exists=True, readable=True),
    run_dir: Path = typer.Option(None),
    resume_latest: bool = typer.Option(False),
) -> None:
    _run(config, "download_slc", run_dir, resume_latest)


@app.command("run-snap")
def run_snap(
    config: Path = typer.Option(..., exists=True, readable=True),
    run_dir: Path = typer.Option(None),
    resume_latest: bool = typer.Option(False),
) -> None:
    _run(config, "snap_preprocess", run_dir, resume_latest)


@app.command("run-stamps")
def run_stamps(
    config: Path = typer.Option(..., exists=True, readable=True),
    run_dir: Path = typer.Option(None),
    resume_latest: bool = typer.Option(False),
) -> None:
    _run(config, "stamps", run_dir, resume_latest)


@app.command("parse-psi")
def parse_psi(
    config: Path = typer.Option(..., exists=True, readable=True),
    run_dir: Path = typer.Option(None),
    resume_latest: bool = typer.Option(False),
) -> None:
    _run(config, "parse_psi", run_dir, resume_latest)


@app.command("fuse")
def fuse(
    config: Path = typer.Option(..., exists=True, readable=True),
    run_dir: Path = typer.Option(None),
    resume_latest: bool = typer.Option(False),
) -> None:
    _run(config, "fuse", run_dir, resume_latest)


@app.command("evaluate")
def evaluate(
    output_dir: Path = typer.Option(...),
    points: Optional[Path] = typer.Option(None, exists=True, readable=True),
    reference_points: Optional[Path] = typer.Option(None, exists=True, readable=True),
    polygons: Optional[Path] = typer.Option(None, exists=True, readable=True),
    reference_polygons: Optional[Path] = typer.Option(None, exists=True, readable=True),
) -> None:
    if not any((points and reference_points, polygons and reference_polygons)):
        raise typer.BadParameter(
            "Provide either --points with --reference-points, or --polygons with --reference-polygons."
        )
    result: dict[str, object] = {}
    if points and reference_points:
        result["points"] = evaluate_points_against_reference(read_vector(points), read_vector(reference_points))
    if polygons and reference_polygons:
        result["polygons"] = evaluate_polygons_against_reference(read_vector(polygons), read_vector(reference_polygons))
    write_evaluation(result, output_dir)


if __name__ == "__main__":
    app()
