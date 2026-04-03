from __future__ import annotations

import json
import sys
from pathlib import Path
from textwrap import dedent


def markdown_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(text).strip("\n").splitlines(keepends=True),
    }


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(text).strip("\n").splitlines(keepends=True),
    }


VISUALS_MD = """
## Interactive visualization

The next cells add **zoomable maps** inside Kaggle so you can inspect:

- `t1` and `t2` Wayback imagery with a swipe comparison
- the derived `new_building_mask` over the newer Wayback image
- the `change_probability` surface over the newer Wayback image

All maps keep the georeferenced raster bounds, so zooming and panning stay spatially correct.
"""


VISUALS_CODE = """
import shutil
import zipfile

import folium
from folium.plugins import SideBySideLayers
from IPython.display import display
from matplotlib import colors


def save_multiband_like(reference_path: Path, output_path: Path, array: np.ndarray, dtype: str, compress: str = "LZW") -> Path:
    with rasterio.open(reference_path) as src:
        profile = src.profile.copy()

    if array.ndim == 2:
        array = array[:, :, None]

    profile.update(
        driver="GTiff",
        count=array.shape[2],
        dtype=dtype,
        compress=compress,
    )

    with rasterio.open(output_path, "w", **profile) as dst:
        for band_idx in range(array.shape[2]):
            dst.write(array[:, :, band_idx].astype(dtype), band_idx + 1)
    return output_path


def raster_bounds_latlon(reference_path: Path) -> dict:
    with rasterio.open(reference_path) as src:
        left, bottom, right, top = src.bounds
    west, south = TO_4326.transform(left, bottom)
    east, north = TO_4326.transform(right, top)
    return {
        "west": float(west),
        "south": float(south),
        "east": float(east),
        "north": float(north),
    }


def rgb_for_folium(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def scalar_to_rgba(
    array: np.ndarray,
    *,
    cmap_name: str = "magma",
    vmin: float = 0.0,
    vmax: float = 1.0,
    alpha_scale: float = 0.75,
    transparent_below: float | None = None,
) -> np.ndarray:
    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    rgba = plt.get_cmap(cmap_name)(norm(array)).astype(np.float32)
    if transparent_below is None:
        rgba[..., 3] = rgba[..., 3] * alpha_scale
    else:
        rgba[..., 3] = np.where(array >= transparent_below, rgba[..., 3] * alpha_scale, 0.0)
    return rgba


def binary_mask_to_rgba(mask: np.ndarray, *, color=(0.0, 1.0, 0.0), alpha: float = 0.80) -> np.ndarray:
    rgba = np.zeros(mask.shape + (4,), dtype=np.float32)
    rgba[..., 0] = float(color[0])
    rgba[..., 1] = float(color[1])
    rgba[..., 2] = float(color[2])
    rgba[..., 3] = np.where(mask, alpha, 0.0)
    return rgba


def blend_rgb_mask(base_rgb: np.ndarray, mask: np.ndarray, *, color=(0, 255, 0), alpha: float = 0.55) -> np.ndarray:
    base = base_rgb.astype(np.float32).copy()
    color_arr = np.array(color, dtype=np.float32)
    base[mask] = (1.0 - alpha) * base[mask] + alpha * color_arr
    return np.clip(base, 0, 255).astype(np.uint8)


def build_side_by_side_map(
    left_rgb: np.ndarray,
    right_rgb: np.ndarray,
    reference_path: Path,
    *,
    left_name: str,
    right_name: str,
    zoom_start: int = 16,
):
    bounds = raster_bounds_latlon(reference_path)
    folium_bounds = [[bounds["south"], bounds["west"]], [bounds["north"], bounds["east"]]]
    center = [(bounds["south"] + bounds["north"]) / 2.0, (bounds["west"] + bounds["east"]) / 2.0]

    fmap = folium.Map(location=center, zoom_start=zoom_start, tiles="OpenStreetMap", control_scale=True)
    left_layer = folium.raster_layers.ImageOverlay(
        image=rgb_for_folium(left_rgb),
        bounds=folium_bounds,
        name=left_name,
        opacity=1.0,
        interactive=True,
        cross_origin=False,
    )
    right_layer = folium.raster_layers.ImageOverlay(
        image=rgb_for_folium(right_rgb),
        bounds=folium_bounds,
        name=right_name,
        opacity=1.0,
        interactive=True,
        cross_origin=False,
    )
    left_layer.add_to(fmap)
    right_layer.add_to(fmap)
    SideBySideLayers(left_layer, right_layer).add_to(fmap)
    fmap.fit_bounds(folium_bounds)
    return fmap


def build_overlay_map(
    base_rgb: np.ndarray,
    overlays: list[tuple[str, np.ndarray]],
    reference_path: Path,
    *,
    zoom_start: int = 16,
):
    bounds = raster_bounds_latlon(reference_path)
    folium_bounds = [[bounds["south"], bounds["west"]], [bounds["north"], bounds["east"]]]
    center = [(bounds["south"] + bounds["north"]) / 2.0, (bounds["west"] + bounds["east"]) / 2.0]

    fmap = folium.Map(location=center, zoom_start=zoom_start, tiles="OpenStreetMap", control_scale=True)
    folium.raster_layers.ImageOverlay(
        image=rgb_for_folium(base_rgb),
        bounds=folium_bounds,
        name="Wayback RGB",
        opacity=1.0,
        interactive=True,
        cross_origin=False,
    ).add_to(fmap)

    for overlay_name, overlay_rgba in overlays:
        folium.raster_layers.ImageOverlay(
            image=overlay_rgba,
            bounds=folium_bounds,
            name=overlay_name,
            opacity=1.0,
            interactive=True,
            cross_origin=False,
        ).add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.fit_bounds(folium_bounds)
    return fmap


t1_t2_swipe_map = build_side_by_side_map(
    mosaic_t1["array"],
    mosaic_t2["array"],
    mosaic_t2["geotiff_path"],
    left_name=f"T1 {scene_t1['release']['identifier']}",
    right_name=f"T2 {scene_t2['release']['identifier']}",
    zoom_start=16,
)

new_building_zoom_map = build_overlay_map(
    mosaic_t2["array"],
    [
        ("New building mask", binary_mask_to_rgba(new_building_mask, color=(0.0, 1.0, 0.0), alpha=0.85)),
    ],
    mosaic_t2["geotiff_path"],
    zoom_start=16,
)

change_probability_zoom_map = build_overlay_map(
    mosaic_t2["array"],
    [
        ("Change probability", scalar_to_rgba(change_prob, cmap_name="magma", alpha_scale=0.80, transparent_below=0.15)),
        ("New building mask", binary_mask_to_rgba(new_building_mask, color=(0.0, 1.0, 0.0), alpha=0.65)),
    ],
    mosaic_t2["geotiff_path"],
    zoom_start=16,
)

print("Zoomable T1/T2 swipe map:")
display(t1_t2_swipe_map)
print("Zoomable new-building map:")
display(new_building_zoom_map)
print("Zoomable change-probability map:")
display(change_probability_zoom_map)
"""


QGIS_MD = """
## QGIS export

The next cell exports a clean georeferenced bundle for QGIS:

- the two Wayback RGB GeoTIFFs
- change and building probability GeoTIFFs
- the binary new-building mask GeoTIFF
- a labeled component GeoTIFF
- a georeferenced RGB overlay GeoTIFF for quick inspection
- the final `GeoJSON` and `CSV`
- a manifest and a single `.zip` bundle
"""


QGIS_CODE = """
import shutil
import zipfile

def safe_save_single_band_like(reference_path: Path, output_path: Path, array: np.ndarray, dtype: str) -> Path:
    with rasterio.open(reference_path) as src:
        profile = src.profile.copy()
    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)
    profile.pop("tiled", None)
    profile.update(driver="GTiff", count=1, dtype=dtype)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(array.astype(dtype), 1)
    return output_path


def safe_save_multiband_like(reference_path: Path, output_path: Path, array: np.ndarray, dtype: str, compress: str = "LZW") -> Path:
    with rasterio.open(reference_path) as src:
        profile = src.profile.copy()
    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)
    profile.pop("tiled", None)
    if array.ndim == 2:
        array = array[:, :, None]
    profile.update(driver="GTiff", count=array.shape[2], dtype=dtype, compress=compress)
    with rasterio.open(output_path, "w", **profile) as dst:
        for band_idx in range(array.shape[2]):
            dst.write(array[:, :, band_idx].astype(dtype), band_idx + 1)
    return output_path


def blend_rgb_mask(base_rgb: np.ndarray, mask: np.ndarray, *, color=(0, 255, 0), alpha: float = 0.55) -> np.ndarray:
    base = base_rgb.astype(np.float32).copy()
    color_arr = np.array(color, dtype=np.float32)
    base[mask] = (1.0 - alpha) * base[mask] + alpha * color_arr
    return np.clip(base, 0, 255).astype(np.uint8)


qgis_dir = OUTPUT_DIR / "qgis_export"
qgis_dir.mkdir(parents=True, exist_ok=True)

t1_rgb_qgis_path = qgis_dir / "t1_wayback_rgb.tif"
t2_rgb_qgis_path = qgis_dir / "t2_wayback_rgb.tif"
shutil.copy2(mosaic_t1["geotiff_path"], t1_rgb_qgis_path)
shutil.copy2(mosaic_t2["geotiff_path"], t2_rgb_qgis_path)

change_prob_qgis_path = qgis_dir / "change_probability.tif"
t1_prob_qgis_path = qgis_dir / "t1_building_probability.tif"
t2_prob_qgis_path = qgis_dir / "t2_building_probability.tif"
new_mask_qgis_path = qgis_dir / "new_building_mask.tif"
new_labels_qgis_path = qgis_dir / "new_building_labels.tif"
overlay_qgis_path = qgis_dir / "t2_new_building_overlay.tif"
geojson_qgis_path = qgis_dir / "new_buildings.geojson"
csv_qgis_path = qgis_dir / "new_buildings.csv"
summary_qgis_path = qgis_dir / "wayback_pair_summary.csv"

shutil.copy2(change_prob_path, change_prob_qgis_path)
shutil.copy2(t1_prob_path, t1_prob_qgis_path)
shutil.copy2(t2_prob_path, t2_prob_qgis_path)
shutil.copy2(new_building_mask_path, new_mask_qgis_path)
shutil.copy2(csv_path, csv_qgis_path)
shutil.copy2(release_summary_path, summary_qgis_path)
geojson_qgis_path.write_text(json.dumps(new_buildings_geojson))

safe_save_single_band_like(
    mosaic_t2["geotiff_path"],
    new_labels_qgis_path,
    new_building_labels.astype(np.uint16),
    "uint16",
)

t2_overlay_rgb = blend_rgb_mask(mosaic_t2["array"], new_building_mask, color=(0, 255, 0), alpha=0.55)
safe_save_multiband_like(
    mosaic_t2["geotiff_path"],
    overlay_qgis_path,
    t2_overlay_rgb,
    "uint8",
)

manifest = {
    "release_t1": {
        "identifier": scene_t1["release"]["identifier"],
        "release_date": str(scene_t1["release"]["release_date"]),
        "dominant_src_date": str(scene_t1["dominant_src_date"]),
    },
    "release_t2": {
        "identifier": scene_t2["release"]["identifier"],
        "release_date": str(scene_t2["release"]["release_date"]),
        "dominant_src_date": str(scene_t2["dominant_src_date"]),
    },
    "files": {
        "t1_wayback_rgb": str(t1_rgb_qgis_path),
        "t2_wayback_rgb": str(t2_rgb_qgis_path),
        "change_probability": str(change_prob_qgis_path),
        "t1_building_probability": str(t1_prob_qgis_path),
        "t2_building_probability": str(t2_prob_qgis_path),
        "new_building_mask": str(new_mask_qgis_path),
        "new_building_labels": str(new_labels_qgis_path),
        "t2_new_building_overlay": str(overlay_qgis_path),
        "new_buildings_geojson": str(geojson_qgis_path),
        "new_buildings_csv": str(csv_qgis_path),
        "wayback_pair_summary": str(summary_qgis_path),
    },
}

manifest_path = qgis_dir / "qgis_manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2))

zip_path = OUTPUT_DIR / "qgis_export_bundle.zip"
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for path in sorted(qgis_dir.rglob("*")):
        if path.is_file():
            zf.write(path, arcname=str(path.relative_to(qgis_dir)))

print("QGIS export directory:", qgis_dir)
print("QGIS bundle:", zip_path)
print("Files:")
for path in sorted(qgis_dir.rglob("*")):
    if path.is_file():
        print("-", path)
"""


def ensure_folium_install(cell: dict) -> None:
    src = "".join(cell.get("source", []))
    if '"folium",' in src or '"folium"' in src:
        return
    updated = src.replace(
        '"opencv-python-headless",\n',
        '"opencv-python-headless",\n        "folium",\n',
    )
    cell["source"] = updated.splitlines(keepends=True)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: patch_kaggle_changestar_visuals.py <notebook.ipynb>")

    notebook_path = Path(sys.argv[1])
    nb = json.loads(notebook_path.read_text())

    ensure_folium_install(nb["cells"][2])

    sources = ["".join(cell.get("source", [])) for cell in nb["cells"]]
    if any("## Interactive visualization" in src for src in sources):
        print(f"Notebook already contains interactive visualization cells: {notebook_path}")
        return

    insert_idx = next(
        i for i, cell in enumerate(nb["cells"])
        if cell["cell_type"] == "markdown" and "## Result interpretation" in "".join(cell.get("source", []))
    )

    nb["cells"][insert_idx:insert_idx] = [
        markdown_cell(VISUALS_MD),
        code_cell(VISUALS_CODE),
        markdown_cell(QGIS_MD),
        code_cell(QGIS_CODE),
    ]

    notebook_path.write_text(json.dumps(nb, indent=2))
    print(f"Patched {notebook_path}")


if __name__ == "__main__":
    main()
