from __future__ import annotations

from pathlib import Path

from casablanca_psi.manifests import SlcScene, StackManifest, read_stack_manifest, update_scene_s3_path, write_stack_manifest


def test_manifest_roundtrip(tmp_path: Path) -> None:
    manifest = StackManifest(
        stack_id="asc_rel154_vv",
        direction="ascending",
        relative_orbit=154,
        product_type="SLC",
        scenes=[
            SlcScene(
                scene_id="S1A_TEST",
                product_name="S1A_TEST_PRODUCT",
                acquisition_start="2024-01-01T00:00:00Z",
                acquisition_stop="2024-01-01T00:00:24Z",
                acquisition_date="2024-01-01",
                direction="ascending",
                relative_orbit=154,
                polarization="VV",
                swath_mode="IW",
                product_type="IW_SLC__1S",
                processing_level="L1",
                platform="sentinel-1a",
                asset_name="product",
                href="https://example.invalid/product.zip",
            )
        ],
    )
    path = tmp_path / "manifest.json"
    write_stack_manifest(manifest, path)
    restored = read_stack_manifest(path)
    assert restored.stack_id == manifest.stack_id
    assert restored.scenes[0].scene_id == "S1A_TEST"


def test_update_scene_s3_path_returns_updated_manifest() -> None:
    manifest = StackManifest(
        stack_id="asc_rel154_vv",
        direction="ascending",
        relative_orbit=154,
        product_type="SLC",
        scenes=[
            SlcScene(
                scene_id="S1A_TEST",
                product_name="S1A_TEST_PRODUCT",
                acquisition_start="2024-01-01T00:00:00Z",
                acquisition_stop="2024-01-01T00:00:24Z",
                acquisition_date="2024-01-01",
                direction="ascending",
                relative_orbit=154,
                polarization="VV",
                swath_mode="IW",
                product_type="IW_SLC__1S",
                processing_level="L1",
                platform="sentinel-1a",
                asset_name="product",
                href="https://example.invalid/product.zip",
            )
        ],
    )

    updated = update_scene_s3_path(manifest, "S1A_TEST", "/eodata/Sentinel-1/SAR/SLC/2024/01/01/S1A_TEST_PRODUCT.SAFE")

    assert updated.scenes[0].s3_path == "/eodata/Sentinel-1/SAR/SLC/2024/01/01/S1A_TEST_PRODUCT.SAFE"
    assert manifest.scenes[0].s3_path is None
