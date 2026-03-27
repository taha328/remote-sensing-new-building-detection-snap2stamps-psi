from __future__ import annotations

from pathlib import Path

from casablanca_psi.config import PsiDetectionConfig
from casablanca_psi.psi_results import load_and_detect


def test_load_and_detect_emergent_points(tmp_path: Path) -> None:
    csv_path = tmp_path / "ps_points.csv"
    csv_path.write_text(
        "\n".join(
            [
                "point_id,lon,lat,temporal_coherence,pre_stability_fraction,post_stability_fraction,residual_height_m",
                "1,-7.60,33.57,0.85,0.1,0.8,7.0",
                "2,-7.61,33.58,0.6,0.2,0.5,1.0",
            ]
        ),
        encoding="utf-8",
    )

    artifacts = load_and_detect(csv_path, PsiDetectionConfig())

    assert len(artifacts.points) == 2
    assert len(artifacts.emergent_points) == 1
    assert artifacts.emergent_points.iloc[0]["point_id"] == 1
