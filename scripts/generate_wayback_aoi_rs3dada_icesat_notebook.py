from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "aoi_building_height_wayback_kaggle.ipynb"


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


def build_notebook() -> dict:
    cells = [
        markdown_cell(
            """
            # AOI Building Height with Wayback + RS3DAda + ICESat-2 Calibration

            This notebook replaces the earlier `YOLOv7 + shadow` reconstruction with the core architecture from:

            - Jian Song, Hongruixuan Chen, Naoto Yokoya, *Enhancing Monocular Height Estimation via Sparse LiDAR-Guided Correction* (`arXiv:2505.06905v1`)

            The pipeline is:

            1. Download a single **georeferenced VHR optical image** from **ArcGIS Wayback**.
            2. Run the public **RS3DAda** monocular height model to obtain a dense base height raster.
            3. Run the public **RS3DAda** land-cover model to obtain semantic labels.
            4. Retrieve **ICESat-2 ATL03/ATL08** photons with **SlideRule**.
            5. Apply the paper's preprocessing stages:
               - horizontal offset correction
               - ground interpolation / normalization
               - land-cover-aware filtering
            6. Extract shallow encoder features from the RS3DAda height model.
            7. Train a **RandomForestRegressor** on residuals between RS3DAda height and clean ICESat-2 heights.
            8. Apply the learned residual correction densely over the whole image.
            9. Convert the corrected height raster into **per-building estimates** using connected components on the RS3DAda building mask.

            ## Important correctness notes

            - This notebook uses the **public RS3DAda code path and public checkpoints** from the official SynRS3D repository.
            - It uses **Wayback** as the required single georeferenced VHR optical image source. That is a valid input substitution, but it is not the exact Bing/Tokyo/GeoSampa imagery from the case studies.
            - The PDF text says `srt=3` in the SlideRule paragraph, but the official SlideRule docs define `srt=0` for **land** and `srt=3` for **land ice**. For AOI, this notebook correctly uses `srt=0`.
            - The paper does not publish the exact random-forest hyperparameters or the exact shallow ViT block index used for residual features. This notebook uses the **shallowest of the four DINOv2 intermediate layers exposed by the public RS3DAda implementation** and a fixed, reproducible scikit-learn random forest.
            - The paper's core output is a corrected dense height map. The **per-building table** at the end is a practical post-processing layer added here because your goal is building-height estimation.
            """
        ),
        markdown_cell(
            """
            ## Primary sources

            - Paper preprint: [arXiv:2505.06905](https://arxiv.org/abs/2505.06905)
            - Official RS3DAda / SynRS3D repository: [JTRNEO/SynRS3D](https://github.com/JTRNEO/SynRS3D)
            - Official RS3DAda checkpoints: [JTRNEO/RS3DAda](https://huggingface.co/JTRNEO/RS3DAda)
            - ArcGIS Wayback metadata overview: [Wayback with World Imagery Metadata](https://www.esri.com/arcgis-blog/products/arcgis-living-atlas/imagery/wayback-with-world-imagery-metadata)
            - Live ArcGIS Wayback WMTS capabilities: [WMTSCapabilities.xml](https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/MapServer/WMTS/1.0.0/WMTSCapabilities.xml)
            - Official SlideRule ICESat-2 docs: [ICESat-2 Module](https://slideruleearth.io/rtd/user_guide/icesat2.html)

            ## Kaggle requirements

            - Turn **Internet** on.
            - Use a **T4 GPU** on Kaggle.
            - Do **not** use `P100` in the current Kaggle image: the shipped PyTorch build does not support `sm_60`, so RS3DAda will fail with `no kernel image is available for execution on the device`.
            - Keep the default AOI moderate. The paper's correction is track-driven, so extremely tiny AOIs may have no ICESat-2 coverage.
            """
        ),
        code_cell(
            """
            import subprocess
            import sys
            from pathlib import Path


            def run(cmd, cwd=None):
                printable = " ".join(str(x) for x in cmd)
                print("+", printable)
                subprocess.run([str(x) for x in cmd], cwd=cwd, check=True)


            KAGGLE_WORKING = Path("/kaggle/working")
            SYNRS3D_DIR = KAGGLE_WORKING / "SynRS3D"

            packages = [
                "lxml",
                "rasterio",
                "pyproj",
                "shapely",
                "scikit-learn",
                "scipy",
                "sliderule",
                "huggingface_hub",
                "albumentations",
                "opencv-python-headless",
            ]
            run([sys.executable, "-m", "pip", "install", "-q", *packages])

            if not SYNRS3D_DIR.exists():
                run(["git", "clone", "--depth", "1", "https://github.com/JTRNEO/SynRS3D.git", str(SYNRS3D_DIR)])
            """
        ),
        code_cell(
            """
            import io
            import json
            import math
            import os
            import re
            import sys
            import warnings
            from collections import Counter, defaultdict
            from datetime import date, datetime, timedelta, timezone
            from pathlib import Path

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import rasterio
            import requests
            import torch
            from albumentations import Compose, Normalize
            from albumentations.pytorch import ToTensorV2
            from huggingface_hub import hf_hub_download
            from lxml import etree
            from PIL import Image
            from pyproj import Geod, Transformer
            from rasterio.features import shapes
            from rasterio.transform import from_bounds
            from rasterio.warp import Resampling, reproject
            from scipy import ndimage
            from scipy.spatial import cKDTree
            from shapely.geometry import Polygon, mapping, shape
            from shapely.ops import transform as shapely_transform, unary_union
            from sklearn.ensemble import RandomForestRegressor
            from sliderule import icesat2, sliderule

            if str(SYNRS3D_DIR) not in sys.path:
                sys.path.append(str(SYNRS3D_DIR))

            from models.dpt import DPT_DINOv2

            %config InlineBackend.figure_format = "retina"

            pd.set_option("display.max_columns", 200)
            pd.set_option("display.max_rows", 200)
            warnings.filterwarnings("ignore", category=FutureWarning)
            """
        ),
        code_cell(
            """
            OUTPUT_DIR = Path("/kaggle/working/aoi_rs3dada_icesat")
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

            WMTS_CAPABILITIES_URL = (
                "https://wayback.maptiles.arcgis.com/arcgis/rest/services/"
                "World_Imagery/MapServer/WMTS/1.0.0/WMTSCapabilities.xml"
            )

            CASABLANCA_CENTER = {"lon": -7.62003, "lat": 33.59451}
            AOI_HALF_SIZE_M = 800.0
            RELEASE_SELECTOR = "latest"
            ZOOM = 19
            MAX_TILES = 800

            PATCH_SIZE = 1022
            STRIDE = 511
            USE_TTA = False
            FEATURE_LAYER_INDEX = 0  # shallowest of the 4 public DINOv2 intermediate layers
            DOWNSAMPLE_TO_SOURCE_RESOLUTION = True
            DOWNSAMPLE_TRIGGER_RATIO = 1.15

            HEIGHT_CHECKPOINT_NAME = "RS3DAda_vitl_DPT_height.pth"
            SEGMENTATION_CHECKPOINT_NAME = "RS3DAda_vitl_DPT_segmentation.pth"

            LAND_CLASS = 0
            TREE_CLASS = 1
            BUILDING_CLASS = 2
            TREE_CLASS_INDEX = 4
            BUILDING_CLASS_INDEX = 7
            MIN_ABOVEGROUND_HEIGHT_M = 2.0

            MAX_HORIZONTAL_OFFSET_M = 6.5
            COARSE_OFFSET_STEP_M = 1.0
            FINE_WINDOW_M = 1.0
            FINE_OFFSET_STEP_M = 0.1

            ICESAT_WINDOW_DAYS = [365, 730, 1460, 3650]
            ATL03_CONF_MIN = 3
            ATL08_CLASS_NAMES = ["atl08_ground", "atl08_canopy", "atl08_top_of_canopy"]
            IDW_POWER = 2.0
            IDW_K = 8

            RF_RANDOM_STATE = 42
            MIN_RF_TRAIN_SAMPLES = 25

            MIN_BUILDING_COMPONENT_PIXELS = 40

            def resolve_device() -> str:
                if not torch.cuda.is_available():
                    return "cpu"
                major, minor = torch.cuda.get_device_capability(0)
                gpu_name = torch.cuda.get_device_name(0)
                if major < 7:
                    raise RuntimeError(
                        "The current Kaggle PyTorch build does not support the active GPU "
                        f"({gpu_name}, compute capability sm_{major}{minor}). "
                        "Switch the notebook accelerator to T4 and rerun from the top."
                    )
                return "cuda"


            DEVICE = resolve_device()
            GEOD = Geod(ellps="WGS84")
            TO_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
            TO_4326 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

            session = requests.Session()
            session.headers.update(
                {
                    "User-Agent": "Kaggle-Wayback-RS3DAda-ICESat2/1.0",
                    "Accept": "*/*",
                }
            )


            def bbox_from_center_m(center_lon, center_lat, half_size_m):
                lon_east, lat_east, _ = GEOD.fwd(center_lon, center_lat, 90.0, half_size_m)
                lon_west, lat_west, _ = GEOD.fwd(center_lon, center_lat, 270.0, half_size_m)
                lon_north, lat_north, _ = GEOD.fwd(center_lon, center_lat, 0.0, half_size_m)
                lon_south, lat_south, _ = GEOD.fwd(center_lon, center_lat, 180.0, half_size_m)
                return {
                    "west": min(lon_west, lon_east),
                    "east": max(lon_west, lon_east),
                    "south": min(lat_south, lat_north),
                    "north": max(lat_south, lat_north),
                }


            CASABLANCA_BBOX = bbox_from_center_m(
                CASABLANCA_CENTER["lon"],
                CASABLANCA_CENTER["lat"],
                AOI_HALF_SIZE_M,
            )
            CASABLANCA_BBOX
            """
        ),
        code_cell(
            """
            def get_text(url: str, *, params=None, timeout=120) -> str:
                response = session.get(url, params=params, timeout=timeout)
                response.raise_for_status()
                return response.text


            def get_json(url: str, *, params=None, timeout=120) -> dict:
                response = session.get(url, params=params, timeout=timeout)
                response.raise_for_status()
                return response.json()


            def parse_wmts_capabilities(url: str = WMTS_CAPABILITIES_URL) -> pd.DataFrame:
                xml = get_text(url)
                ns = {
                    "wmts": "https://www.opengis.net/wmts/1.0",
                    "ows": "https://www.opengis.net/ows/1.1",
                }
                root = etree.fromstring(xml.encode("utf-8"))

                rows = []
                for layer in root.xpath(".//wmts:Layer", namespaces=ns):
                    title = layer.xpath("./ows:Title/text()", namespaces=ns)
                    identifier = layer.xpath("./ows:Identifier/text()", namespaces=ns)
                    resource_urls = layer.xpath("./wmts:ResourceURL", namespaces=ns)
                    tile_matrix_sets = layer.xpath(
                        "./wmts:TileMatrixSetLink/wmts:TileMatrixSet/text()", namespaces=ns
                    )

                    if not title or not identifier or not resource_urls:
                        continue

                    title = title[0]
                    identifier = identifier[0]
                    template = resource_urls[0].attrib.get("template")
                    if "Wayback" not in title or not template:
                        continue

                    release_date_match = re.search(r"(\\d{4}-\\d{2}-\\d{2})", title)
                    release_num_match = re.search(r"/tile/(\\d+)/\\{TileMatrix\\}/", template)

                    rows.append(
                        {
                            "title": title,
                            "identifier": identifier,
                            "release_date": (
                                pd.to_datetime(release_date_match.group(1)).date()
                                if release_date_match
                                else None
                            ),
                            "release_num": int(release_num_match.group(1)) if release_num_match else None,
                            "tile_matrix_sets": tile_matrix_sets,
                            "resource_url_template": template,
                        }
                    )

                return pd.DataFrame(rows).sort_values("release_date").reset_index(drop=True)


            def select_release(releases: pd.DataFrame, selector="latest") -> pd.Series:
                if selector == "latest":
                    return releases.iloc[-1]

                selector = str(selector).strip()
                identifier_match = releases["identifier"] == selector
                if identifier_match.any():
                    return releases.loc[identifier_match].iloc[-1]

                try:
                    release_date = pd.to_datetime(selector).date()
                    date_match = releases["release_date"] == release_date
                    if date_match.any():
                        return releases.loc[date_match].iloc[-1]
                except Exception:
                    pass

                raise ValueError("RELEASE_SELECTOR must be 'latest', a Wayback identifier, or YYYY-MM-DD.")


            def metadata_base_url_from_identifier(identifier: str) -> str:
                match = re.fullmatch(r"WB_(\\d{4})_R(\\d{2})", identifier)
                if not match:
                    raise ValueError(f"Unexpected Wayback identifier format: {identifier}")
                year, release_index = match.groups()
                service_name = f"World_Imagery_Metadata_{year}_r{release_index.lower()}"
                return (
                    "https://metadata.maptiles.arcgis.com/arcgis/rest/services/"
                    f"{service_name}/MapServer"
                )


            def get_metadata_service_summary(metadata_base_url: str) -> dict:
                return get_json(f"{metadata_base_url}?f=pjson")


            def get_metadata_layers(metadata_base_url: str) -> pd.DataFrame:
                service = get_metadata_service_summary(metadata_base_url)
                return pd.DataFrame(service["layers"])


            def query_metadata_point(
                metadata_base_url: str,
                lon: float,
                lat: float,
                *,
                out_fields=None,
                layer_ids=None,
            ) -> dict | None:
                service = get_metadata_service_summary(metadata_base_url)
                layers = service["layers"]
                layer_lookup = {layer["id"]: layer["name"] for layer in layers}

                if layer_ids is None:
                    layer_ids = [layer["id"] for layer in layers]
                if out_fields is None:
                    out_fields = [
                        "SRC_DATE",
                        "SRC_DATE2",
                        "SRC_RES",
                        "SRC_ACC",
                        "SAMP_RES",
                        "NICE_NAME",
                        "NICE_DESC",
                        "ReleaseName",
                    ]

                params = {
                    "geometry": json.dumps({"x": lon, "y": lat}),
                    "geometryType": "esriGeometryPoint",
                    "inSR": 4326,
                    "spatialRel": "esriSpatialRelIntersects",
                    "outFields": ",".join(out_fields),
                    "returnGeometry": "false",
                    "f": "pjson",
                }

                for layer_id in layer_ids:
                    payload = get_json(f"{metadata_base_url}/{layer_id}/query", params=params)
                    features = payload.get("features", [])
                    if features:
                        attrs = dict(features[0]["attributes"])
                        attrs["metadata_layer_id"] = layer_id
                        attrs["metadata_layer_name"] = layer_lookup.get(layer_id)
                        return attrs
                return None


            def sample_wayback_metadata_grid(metadata_base_url: str, bbox: dict, n: int = 5) -> pd.DataFrame:
                lons = np.linspace(bbox["west"], bbox["east"], n)
                lats = np.linspace(bbox["south"], bbox["north"], n)
                rows = []
                for lat in lats:
                    for lon in lons:
                        item = query_metadata_point(metadata_base_url, float(lon), float(lat))
                        if item:
                            item["query_lon"] = float(lon)
                            item["query_lat"] = float(lat)
                            rows.append(item)
                return pd.DataFrame(rows)


            def lonlat_to_tile_fraction(lon: float, lat: float, zoom: int) -> tuple[float, float]:
                lat = max(min(lat, 85.05112878), -85.05112878)
                n = 2**zoom
                x = (lon + 180.0) / 360.0 * n
                lat_rad = math.radians(lat)
                y = (
                    (1.0 - math.log(math.tan(lat_rad) + (1.0 / math.cos(lat_rad))) / math.pi)
                    / 2.0
                    * n
                )
                return x, y


            def tile_range_for_bbox(bbox: dict, zoom: int) -> tuple[int, int, int, int]:
                x0f, y1f = lonlat_to_tile_fraction(bbox["west"], bbox["south"], zoom)
                x1f, y0f = lonlat_to_tile_fraction(bbox["east"], bbox["north"], zoom)
                x_min = math.floor(min(x0f, x1f))
                x_max = math.floor(max(x0f, x1f))
                y_min = math.floor(min(y0f, y1f))
                y_max = math.floor(max(y0f, y1f))
                return x_min, x_max, y_min, y_max


            def tile_bounds_3857(x: int, y: int, zoom: int) -> tuple[float, float, float, float]:
                n = 2**zoom
                lon_left = x / n * 360.0 - 180.0
                lon_right = (x + 1) / n * 360.0 - 180.0
                lat_top = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
                lat_bottom = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))

                minx, miny = TO_3857.transform(lon_left, lat_bottom)
                maxx, maxy = TO_3857.transform(lon_right, lat_top)
                return minx, miny, maxx, maxy


            def build_tile_url(template: str, tile_matrix_set: str, zoom: int, x: int, y: int) -> str:
                return (
                    template.replace("{TileMatrixSet}", tile_matrix_set)
                    .replace("{TileMatrix}", str(zoom))
                    .replace("{TileRow}", str(y))
                    .replace("{TileCol}", str(x))
                )


            def download_wayback_mosaic(
                release_row: pd.Series,
                bbox: dict,
                *,
                zoom: int = ZOOM,
                max_tiles: int = MAX_TILES,
                out_dir: Path = OUTPUT_DIR,
            ) -> dict:
                tile_matrix_set = "default028mm"
                if tile_matrix_set not in release_row["tile_matrix_sets"]:
                    raise ValueError(
                        f"{tile_matrix_set} is not available for release {release_row['identifier']}"
                    )

                x_min, x_max, y_min, y_max = tile_range_for_bbox(bbox, zoom)
                tile_count = (x_max - x_min + 1) * (y_max - y_min + 1)
                if tile_count > max_tiles:
                    raise ValueError(
                        f"AOI would download {tile_count} tiles at z={zoom}; "
                        "reduce AOI, lower ZOOM, or raise MAX_TILES."
                    )

                width = (x_max - x_min + 1) * 256
                height = (y_max - y_min + 1) * 256
                canvas = Image.new("RGB", (width, height))

                for y in range(y_min, y_max + 1):
                    for x in range(x_min, x_max + 1):
                        tile_url = build_tile_url(
                            release_row["resource_url_template"], tile_matrix_set, zoom, x, y
                        )
                        response = session.get(tile_url, timeout=120)
                        response.raise_for_status()
                        tile = Image.open(io.BytesIO(response.content)).convert("RGB")
                        canvas.paste(tile, ((x - x_min) * 256, (y - y_min) * 256))

                left, _, _, top = tile_bounds_3857(x_min, y_min, zoom)
                _, bottom, right, _ = tile_bounds_3857(x_max, y_max, zoom)
                bounds_3857 = (left, bottom, right, top)

                png_path = out_dir / f"{release_row['identifier']}_aoi_z{zoom}.png"
                tif_path = out_dir / f"{release_row['identifier']}_aoi_z{zoom}.tif"
                canvas.save(png_path)

                arr = np.asarray(canvas)
                transform = from_bounds(*bounds_3857, width=arr.shape[1], height=arr.shape[0])
                with rasterio.open(
                    tif_path,
                    "w",
                    driver="GTiff",
                    width=arr.shape[1],
                    height=arr.shape[0],
                    count=3,
                    dtype=arr.dtype,
                    crs="EPSG:3857",
                    transform=transform,
                ) as dst:
                    for band_index in range(3):
                        dst.write(arr[:, :, band_index], band_index + 1)

                return {
                    "identifier": release_row["identifier"],
                    "release_num": int(release_row["release_num"]),
                    "tile_count": tile_count,
                    "tile_range": (x_min, x_max, y_min, y_max),
                    "bounds_3857": bounds_3857,
                    "png_path": png_path,
                    "geotiff_path": tif_path,
                    "array": arr,
                    "transform": transform,
                }


            def pixel_to_lonlat(col: float, row: float, mosaic: dict) -> tuple[float, float]:
                x, y = rasterio.transform.xy(mosaic["transform"], row, col, offset="center")
                lon, lat = TO_4326.transform(x, y)
                return lon, lat


            def lonlat_to_pixel(lon: float, lat: float, mosaic: dict) -> tuple[float, float]:
                x, y = TO_3857.transform(lon, lat)
                col, row = (~mosaic["transform"]) * (x, y)
                return float(col), float(row)


            def local_pixel_size_m(mosaic: dict) -> tuple[float, float]:
                h, w = mosaic["array"].shape[:2]
                center_col = w / 2
                center_row = h / 2
                lon1, lat1 = pixel_to_lonlat(center_col, center_row, mosaic)
                lon2, lat2 = pixel_to_lonlat(center_col + 1, center_row, mosaic)
                lon3, lat3 = pixel_to_lonlat(center_col, center_row + 1, mosaic)
                _, _, dx = GEOD.inv(lon1, lat1, lon2, lat2)
                _, _, dy = GEOD.inv(lon1, lat1, lon3, lat3)
                return abs(dx), abs(dy)


            def downsample_mosaic_to_target_resolution(
                mosaic: dict,
                target_resolution_m: float,
                *,
                out_dir: Path = OUTPUT_DIR,
            ) -> dict:
                px_x_m, px_y_m = local_pixel_size_m(mosaic)
                current_resolution_m = float((px_x_m + px_y_m) / 2.0)
                if target_resolution_m <= 0:
                    return mosaic
                if current_resolution_m >= target_resolution_m / DOWNSAMPLE_TRIGGER_RATIO:
                    return mosaic

                with rasterio.open(mosaic["geotiff_path"]) as src:
                    left, bottom, right, top = src.bounds
                    scale = target_resolution_m / current_resolution_m
                    dst_width = max(1, int(round(src.width / scale)))
                    dst_height = max(1, int(round(src.height / scale)))
                    dst_transform = from_bounds(left, bottom, right, top, dst_width, dst_height)
                    dst_arr = np.zeros((3, dst_height, dst_width), dtype=np.uint8)

                    for band_idx in range(1, 4):
                        reproject(
                            source=rasterio.band(src, band_idx),
                            destination=dst_arr[band_idx - 1],
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=dst_transform,
                            dst_crs=src.crs,
                            resampling=Resampling.bilinear,
                        )

                dst_path = out_dir / f"{Path(mosaic['geotiff_path']).stem}_srcres.tif"
                with rasterio.open(
                    dst_path,
                    "w",
                    driver="GTiff",
                    width=dst_width,
                    height=dst_height,
                    count=3,
                    dtype="uint8",
                    crs="EPSG:3857",
                    transform=dst_transform,
                ) as dst:
                    for band_idx in range(3):
                        dst.write(dst_arr[band_idx], band_idx + 1)

                downsampled = {
                    **mosaic,
                    "geotiff_path": dst_path,
                    "array": np.moveaxis(dst_arr, 0, -1),
                    "transform": dst_transform,
                    "downsampled_to_src_res": True,
                    "original_geotiff_path": mosaic["geotiff_path"],
                }
                return downsampled
            """
        ),
        code_cell(
            """
            releases = parse_wmts_capabilities()
            selected_release = select_release(releases, RELEASE_SELECTOR)
            metadata_base_url = metadata_base_url_from_identifier(selected_release["identifier"])
            metadata_layers = get_metadata_layers(metadata_base_url)
            metadata_grid = sample_wayback_metadata_grid(metadata_base_url, CASABLANCA_BBOX, n=5)

            mosaic = download_wayback_mosaic(
                selected_release,
                CASABLANCA_BBOX,
                zoom=ZOOM,
                max_tiles=MAX_TILES,
            )

            print("Selected Wayback release:")
            display(pd.DataFrame([selected_release]))
            print("\\nMetadata layers:")
            display(metadata_layers)
            print("\\nSampled metadata inside AOI:")
            display(metadata_grid.head(10))

            if metadata_grid.empty or metadata_grid["SRC_DATE"].dropna().empty:
                raise RuntimeError("Wayback metadata sampling did not return SRC_DATE values for the AOI.")

            dominant_src_date = int(metadata_grid["SRC_DATE"].dropna().mode().iloc[0])
            dominant_src_date_dt = pd.to_datetime(str(dominant_src_date), format="%Y%m%d").date()
            dominant_src_res = float(metadata_grid["SRC_RES"].dropna().median())

            if DOWNSAMPLE_TO_SOURCE_RESOLUTION:
                mosaic = downsample_mosaic_to_target_resolution(mosaic, dominant_src_res)

            print("\\nDominant SRC_DATE:", dominant_src_date_dt)
            print("Median SRC_RES (m):", dominant_src_res)
            print("Wayback image path:", mosaic["geotiff_path"])
            print("Tile count:", mosaic["tile_count"])
            print("Local pixel size near AOI center (m):", local_pixel_size_m(mosaic))
            print("DEVICE:", DEVICE)
            if torch.cuda.is_available():
                print("GPU:", torch.cuda.get_device_name(0))
            print("USE_TTA:", USE_TTA)
            print("Downsampled to source resolution:", mosaic.get("downsampled_to_src_res", False))

            plt.figure(figsize=(8, 8))
            plt.imshow(mosaic["array"])
            plt.title(f"{selected_release['identifier']} | AOI Wayback AOI")
            plt.axis("off")
            plt.show()
            """
        ),
        markdown_cell(
            """
            ## RS3DAda inference

            The next cells use the **public official RS3DAda implementation** from `JTRNEO/SynRS3D` and the official Hugging Face checkpoints:

            - `RS3DAda_vitl_DPT_height.pth`
            - `RS3DAda_vitl_DPT_segmentation.pth`

            The patch size, overlap, normalization, and blending logic below match the public `infer_height.py` and `infer_segmentation.py` scripts.

            For practicality on Kaggle, the default notebook settings differ from the public demo script in three ways:

            - `USE_TTA=False` by default, because test-time augmentation multiplies runtime by about 4.
            - if the Wayback web tiles are finer than the metadata-reported `SRC_RES`, the mosaic is automatically downsampled to approximately the source resolution before inference.
            - the default AOI is larger than before so the ICESat-2 calibration stage is more likely to have enough photons after filtering.

            These changes reduce compute without changing the model architecture or the correction pipeline.
            """
        ),
        code_cell(
            """
            def load_rgb_tiff(path: Path) -> np.ndarray:
                with rasterio.open(path) as src:
                    arr = np.moveaxis(src.read(), 0, -1)
                if arr.shape[2] > 3:
                    arr = arr[:, :, :3]
                return arr.astype(np.uint8)


            def save_raster_like(reference_path: Path, output_path: Path, array: np.ndarray, dtype=None) -> Path:
                with rasterio.open(reference_path) as src:
                    profile = src.profile.copy()

                if array.ndim == 2:
                    count = 1
                else:
                    count = array.shape[2]

                profile.update(
                    driver="GTiff",
                    count=count,
                    dtype=dtype or str(array.dtype),
                )

                with rasterio.open(output_path, "w", **profile) as dst:
                    if array.ndim == 2:
                        dst.write(array.astype(profile["dtype"]), 1)
                    else:
                        for band_idx in range(count):
                            dst.write(array[:, :, band_idx].astype(profile["dtype"]), band_idx + 1)
                return output_path


            def build_blend_weights(patch_size: int, stride: int) -> np.ndarray:
                w = patch_size
                if patch_size > stride:
                    s1 = stride
                    s2 = w - s1
                    d = 1.0 / (1.0 + s2)
                    b1 = np.ones((w, w), dtype=np.float32)
                    b1[:, s1:] = np.dot(
                        np.ones((w, 1), dtype=np.float32),
                        (-np.arange(1, s2 + 1, dtype=np.float32) * d + 1).reshape(1, s2),
                    )
                    b2 = np.flip(b1)
                    b3 = b1.T
                    b4 = np.flip(b3)
                    return (b1 * b2 * b3 * b4).astype(np.float32)
                return np.ones((w, w), dtype=np.float32)


            PATCH_TRANSFORM = Compose(
                [
                    Normalize(
                        mean=(123.675, 116.28, 103.53),
                        std=(58.395, 57.12, 57.375),
                        max_pixel_value=1,
                        always_apply=True,
                    ),
                    ToTensorV2(),
                ]
            )


            def build_rs3dada_model(checkpoint_path: Path) -> DPT_DINOv2:
                head_configs = [
                    {"name": "regression", "nclass": 1},
                    {"name": "segmentation", "nclass": 8},
                ]
                model = DPT_DINOv2(encoder="vitl", head_configs=head_configs, pretrained=False)
                state = torch.load(checkpoint_path, map_location="cpu")
                model.load_state_dict(state)
                model.eval()
                model.to(DEVICE)
                return model


            def sliding_positions(full_size: int, patch_size: int, stride: int) -> list[int]:
                if full_size <= patch_size:
                    return [0]
                positions = list(range(0, full_size - patch_size + 1, stride))
                if positions[-1] != full_size - patch_size:
                    positions.append(full_size - patch_size)
                return positions


            def run_rs3dada_inference(
                model: DPT_DINOv2,
                image_tif: Path,
                *,
                task: str,
                patch_size: int = PATCH_SIZE,
                stride: int = STRIDE,
                use_tta: bool = USE_TTA,
            ) -> np.ndarray:
                img0 = load_rgb_tiff(image_tif)
                height, width = img0.shape[:2]

                x_positions = sliding_positions(width, patch_size, stride)
                y_positions = sliding_positions(height, patch_size, stride)
                padded_h = patch_size + stride * (len(y_positions) - 1)
                padded_w = patch_size + stride * (len(x_positions) - 1)

                img1 = np.zeros((padded_h, padded_w, 3), dtype=np.uint8)
                img1[:height, :width] = img0.copy()

                weights = build_blend_weights(patch_size, stride)
                blend = np.zeros((padded_h, padded_w), dtype=np.float32)

                if task == "regression":
                    pred_all = np.zeros((padded_h, padded_w), dtype=np.float32)
                else:
                    pred_all = np.zeros((8, padded_h, padded_w), dtype=np.float32)

                for y0 in y_positions:
                    for x0 in x_positions:
                        img = img1[y0 : y0 + patch_size, x0 : x0 + patch_size].copy().astype(np.float32)
                        if use_tta:
                            views = [
                                img.copy(),
                                img[:, ::-1, :].copy(),
                                img[::-1, :, :].copy(),
                                img[::-1, ::-1, :].copy(),
                            ]
                            inputs = torch.cat(
                                [
                                    PATCH_TRANSFORM(image=view)["image"].unsqueeze(0)
                                    for view in views
                                ],
                                dim=0,
                            ).float().to(DEVICE)
                        else:
                            inputs = PATCH_TRANSFORM(image=img)["image"].unsqueeze(0).float().to(DEVICE)

                        with torch.no_grad():
                            outputs = model(inputs)

                        if task == "regression":
                            if use_tta:
                                pred = outputs["regression"].detach().cpu().numpy()
                                pred = (
                                    pred[0, :, :, :]
                                    + pred[1, :, :, ::-1]
                                    + pred[2, :, ::-1, :]
                                    + pred[3, :, ::-1, ::-1]
                                ) / 4.0
                                pred = pred.squeeze()
                            else:
                                pred = outputs["regression"].detach().cpu().numpy().squeeze()

                            pred_all[y0 : y0 + patch_size, x0 : x0 + patch_size] += pred * weights
                        else:
                            if use_tta:
                                pred = outputs["segmentation"].detach().cpu().numpy()
                                pred = (
                                    pred[0, :, :, :]
                                    + pred[1, :, :, ::-1]
                                    + pred[2, :, ::-1, :]
                                    + pred[3, :, ::-1, ::-1]
                                ) / 4.0
                            else:
                                pred = outputs["segmentation"].detach().cpu().numpy().squeeze(0)
                            pred_all[:, y0 : y0 + patch_size, x0 : x0 + patch_size] += pred * weights

                        blend[y0 : y0 + patch_size, x0 : x0 + patch_size] += weights

                if task == "regression":
                    pred_all = pred_all / blend
                    return pred_all[:height, :width]

                pred_all = pred_all / blend[None, :, :]
                pred_all = np.argmax(pred_all, axis=0).astype(np.uint8)
                return pred_all[:height, :width]


            def landcover_to_ground_tree_building(segmentation: np.ndarray) -> np.ndarray:
                out = np.zeros_like(segmentation, dtype=np.uint8)
                out[segmentation == TREE_CLASS_INDEX] = TREE_CLASS
                out[segmentation == BUILDING_CLASS_INDEX] = BUILDING_CLASS
                return out


            def download_rs3dada_checkpoint(filename: str) -> Path:
                return Path(
                    hf_hub_download(
                        repo_id="JTRNEO/RS3DAda",
                        filename=filename,
                        local_dir=OUTPUT_DIR / "checkpoints",
                        local_dir_use_symlinks=False,
                    )
                )


            height_ckpt = download_rs3dada_checkpoint(HEIGHT_CHECKPOINT_NAME)
            segmentation_ckpt = download_rs3dada_checkpoint(SEGMENTATION_CHECKPOINT_NAME)

            print("Height checkpoint:", height_ckpt)
            print("Segmentation checkpoint:", segmentation_ckpt)
            """
        ),
        code_cell(
            """
            height_model = build_rs3dada_model(height_ckpt)
            base_height = run_rs3dada_inference(
                height_model,
                mosaic["geotiff_path"],
                task="regression",
                patch_size=PATCH_SIZE,
                stride=STRIDE,
                use_tta=USE_TTA,
            ).astype(np.float32)
            del height_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            segmentation_model = build_rs3dada_model(segmentation_ckpt)
            segmentation = run_rs3dada_inference(
                segmentation_model,
                mosaic["geotiff_path"],
                task="segmentation",
                patch_size=PATCH_SIZE,
                stride=STRIDE,
                use_tta=USE_TTA,
            ).astype(np.uint8)
            del segmentation_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            landcover_gtb = landcover_to_ground_tree_building(segmentation)

            base_height_path = save_raster_like(
                mosaic["geotiff_path"],
                OUTPUT_DIR / "rs3dada_base_height.tif",
                base_height,
                dtype="float32",
            )
            segmentation_path = save_raster_like(
                mosaic["geotiff_path"],
                OUTPUT_DIR / "rs3dada_segmentation.tif",
                segmentation,
                dtype="uint8",
            )
            gtb_path = save_raster_like(
                mosaic["geotiff_path"],
                OUTPUT_DIR / "rs3dada_ground_tree_building.tif",
                landcover_gtb,
                dtype="uint8",
            )

            print("Base height map:", base_height_path)
            print("Segmentation map:", segmentation_path)
            print("Collapsed ground/tree/building map:", gtb_path)

            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            axes[0].imshow(mosaic["array"])
            axes[0].set_title("Wayback RGB")
            axes[0].axis("off")

            im1 = axes[1].imshow(base_height, cmap="viridis")
            axes[1].set_title("RS3DAda base height (m)")
            axes[1].axis("off")
            plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

            im2 = axes[2].imshow(landcover_gtb, cmap="Accent", vmin=0, vmax=2)
            axes[2].set_title("Ground / Tree / Building")
            axes[2].axis("off")
            plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
            plt.show()
            """
        ),
        markdown_cell(
            """
            ## ICESat-2 retrieval and preprocessing

            The paper's preprocessing stages are implemented below:

            1. Retrieve ATL03 photons through SlideRule with ATL08 classes attached.
            2. Filter to `atl03_cnf >= 3` and `quality_ph == 0`.
            3. Search a horizontal offset up to `±6.5 m` with coarse and fine grids.
            4. Normalize canopy photons by IDW interpolation of ground photons.
            5. Apply land-cover-aware filtering using RS3DAda ground/tree/building labels.

            ## Important correction to the PDF text

            The PDF retrieval paragraph writes `srt=3`, but the official SlideRule docs define:

            - `srt=0` = land
            - `srt=3` = land ice

            AOI is land, so this notebook correctly uses `srt=0`.
            """
        ),
        code_cell(
            """
            def bbox_to_sliderule_poly(bbox: dict) -> list[dict]:
                return [
                    {"lon": bbox["west"], "lat": bbox["south"]},
                    {"lon": bbox["east"], "lat": bbox["south"]},
                    {"lon": bbox["east"], "lat": bbox["north"]},
                    {"lon": bbox["west"], "lat": bbox["north"]},
                    {"lon": bbox["west"], "lat": bbox["south"]},
                ]


            def datetime_to_iso_z(dt: datetime) -> str:
                return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


            def initialize_sliderule():
                sliderule.init("slideruleearth.io", verbose=False)
                if hasattr(icesat2, "init"):
                    icesat2.init("slideruleearth.io", verbose=False)


            def normalize_icesat_dataframe(gdf) -> pd.DataFrame:
                if gdf is None or len(gdf) == 0:
                    return pd.DataFrame()

                df = pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))

                if "longitude" not in df.columns:
                    if "lon" in df.columns:
                        df["longitude"] = df["lon"]
                    elif hasattr(gdf, "geometry") and getattr(gdf, "geometry", None) is not None:
                        try:
                            df["longitude"] = gdf.geometry.x
                        except Exception:
                            pass

                if "latitude" not in df.columns:
                    if "lat" in df.columns:
                        df["latitude"] = df["lat"]
                    elif hasattr(gdf, "geometry") and getattr(gdf, "geometry", None) is not None:
                        try:
                            df["latitude"] = gdf.geometry.y
                        except Exception:
                            pass

                rename_map = {}
                if "h_mean" in df.columns and "height" not in df.columns:
                    rename_map["h_mean"] = "height"
                if "cnf" in df.columns and "atl03_cnf" not in df.columns:
                    rename_map["cnf"] = "atl03_cnf"
                if rename_map:
                    df = df.rename(columns=rename_map)

                required = ["latitude", "longitude", "height", "atl08_class", "atl03_cnf", "quality_ph"]
                missing = [col for col in required if col not in df.columns]
                if missing:
                    raise KeyError(
                        "SlideRule output is missing required columns: "
                        f"{missing}. Available columns: {sorted(df.columns.tolist())}"
                    )

                numeric_cols = required
                for col in numeric_cols:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                return df


            def fetch_icesat2_photons(bbox: dict, t0: datetime, t1: datetime) -> pd.DataFrame:
                parms = {
                    "poly": bbox_to_sliderule_poly(bbox),
                    "t0": datetime_to_iso_z(t0),
                    "t1": datetime_to_iso_z(t1),
                    "srt": 0,
                    "cnf": 0,
                    "quality_ph": 0,
                    "atl08_class": ATL08_CLASS_NAMES,
                    "len": 20.0,
                    "res": 20.0,
                }
                gdf = icesat2.atl03sp(parms)
                if gdf is None or len(gdf) == 0:
                    return pd.DataFrame()
                df = normalize_icesat_dataframe(gdf)
                df = df[
                    df["atl08_class"].isin([1, 2, 3])
                    & (df["atl03_cnf"] >= ATL03_CONF_MIN)
                    & (df["quality_ph"] == 0)
                ].copy()
                if df.empty:
                    return df
                df = df.reset_index(drop=True)

                cols_rows = df.apply(
                    lambda row: lonlat_to_pixel(row["longitude"], row["latitude"], mosaic),
                    axis=1,
                    result_type="expand",
                )
                cols_rows.columns = ["col", "row"]
                cols_rows = cols_rows.reset_index(drop=True)
                df = pd.concat([df, cols_rows], axis=1)
                h, w = base_height.shape
                df = df[
                    (df["col"] >= 0)
                    & (df["col"] < w)
                    & (df["row"] >= 0)
                    & (df["row"] < h)
                ].reset_index(drop=True)
                return df


            def bilinear_sample(array: np.ndarray, cols: np.ndarray, rows: np.ndarray) -> np.ndarray:
                h, w = array.shape
                cols = np.asarray(cols, dtype=np.float64)
                rows = np.asarray(rows, dtype=np.float64)
                x0 = np.floor(cols).astype(int)
                y0 = np.floor(rows).astype(int)
                x1 = x0 + 1
                y1 = y0 + 1
                valid = (x0 >= 0) & (x1 < w) & (y0 >= 0) & (y1 < h)
                out = np.full(cols.shape, np.nan, dtype=np.float64)
                if not np.any(valid):
                    return out
                xv = cols[valid]
                yv = rows[valid]
                x0v = x0[valid]
                y0v = y0[valid]
                x1v = x1[valid]
                y1v = y1[valid]
                wa = (x1v - xv) * (y1v - yv)
                wb = (xv - x0v) * (y1v - yv)
                wc = (x1v - xv) * (yv - y0v)
                wd = (xv - x0v) * (yv - y0v)
                out[valid] = (
                    wa * array[y0v, x0v]
                    + wb * array[y0v, x1v]
                    + wc * array[y1v, x0v]
                    + wd * array[y1v, x1v]
                )
                return out


            def nearest_sample(array: np.ndarray, cols: np.ndarray, rows: np.ndarray, fill_value=255) -> np.ndarray:
                h, w = array.shape
                c = np.rint(cols).astype(int)
                r = np.rint(rows).astype(int)
                out = np.full(c.shape, fill_value, dtype=array.dtype if np.isscalar(fill_value) else np.uint8)
                valid = (c >= 0) & (c < w) & (r >= 0) & (r < h)
                out[valid] = array[r[valid], c[valid]]
                return out


            def idw_predict(
                sample_xy: np.ndarray,
                sample_values: np.ndarray,
                query_xy: np.ndarray,
                *,
                k: int = IDW_K,
                power: float = IDW_POWER,
            ) -> np.ndarray:
                if len(sample_xy) == 0:
                    return np.full(len(query_xy), np.nan, dtype=np.float64)
                k = max(1, min(k, len(sample_xy)))
                tree = cKDTree(sample_xy)
                distances, indices = tree.query(query_xy, k=k)
                if k == 1:
                    distances = distances[:, None]
                    indices = indices[:, None]
                distances = np.maximum(distances.astype(np.float64), 1e-6)
                weights = 1.0 / np.power(distances, power)
                values = sample_values[indices]
                return np.sum(weights * values, axis=1) / np.sum(weights, axis=1)


            def normalize_shifted_photons(shifted: pd.DataFrame) -> pd.DataFrame:
                ground = shifted[shifted["atl08_class"] == 1].copy()
                canopy = shifted[shifted["atl08_class"].isin([2, 3])].copy()
                if ground.empty or canopy.empty:
                    return pd.DataFrame()

                jitter = np.linspace(0.0, 1e-6, len(ground), dtype=np.float64)
                ground_xy = np.column_stack(
                    [
                        ground["col"].to_numpy(dtype=np.float64) + jitter,
                        ground["row"].to_numpy(dtype=np.float64) + jitter,
                    ]
                )
                ground_z = ground["height"].to_numpy(dtype=np.float64)
                canopy_xy = np.column_stack(
                    [
                        canopy["col"].to_numpy(dtype=np.float64),
                        canopy["row"].to_numpy(dtype=np.float64),
                    ]
                )
                canopy_ground = idw_predict(ground_xy, ground_z, canopy_xy, k=IDW_K, power=IDW_POWER)
                canopy["h_ndsm"] = canopy["height"].to_numpy(dtype=np.float64) - canopy_ground
                canopy = canopy[np.isfinite(canopy["h_ndsm"]) & (canopy["h_ndsm"] >= 0)].copy()

                ground["h_ndsm"] = 0.0
                out = pd.concat([ground, canopy], ignore_index=True)
                return out.reset_index(drop=True)


            def build_clean_cell_reference(normalized: pd.DataFrame, landcover_gtb: np.ndarray) -> pd.DataFrame:
                if normalized.empty:
                    return pd.DataFrame()

                work = normalized.copy()
                work["cell_col"] = np.rint(work["col"]).astype(int)
                work["cell_row"] = np.rint(work["row"]).astype(int)
                work["pred_class"] = nearest_sample(
                    landcover_gtb,
                    work["cell_col"].to_numpy(),
                    work["cell_row"].to_numpy(),
                    fill_value=255,
                )
                work = work[work["pred_class"] != 255].copy()
                if work.empty:
                    return pd.DataFrame()

                all_groups = (
                    work.groupby(["cell_row", "cell_col"], as_index=False)
                    .agg(pred_class=("pred_class", lambda x: int(pd.Series(x).mode().iloc[0])))
                    .sort_values(["cell_row", "cell_col"])
                    .reset_index(drop=True)
                )

                records = []
                for (cell_row, cell_col), grp in work.groupby(["cell_row", "cell_col"]):
                    pred_class = int(pd.Series(grp["pred_class"]).mode().iloc[0])
                    if pred_class == LAND_CLASS:
                        ground_grp = grp[grp["atl08_class"] == 1]
                        if not ground_grp.empty:
                            records.append(
                                {
                                    "cell_row": int(cell_row),
                                    "cell_col": int(cell_col),
                                    "pred_class": pred_class,
                                    "clean_height_m": 0.0,
                                    "source": "ground",
                                }
                            )
                    elif pred_class in (TREE_CLASS, BUILDING_CLASS):
                        canopy_grp = grp[
                            grp["atl08_class"].isin([2, 3]) & (grp["h_ndsm"] >= MIN_ABOVEGROUND_HEIGHT_M)
                        ]
                        if not canopy_grp.empty:
                            records.append(
                                {
                                    "cell_row": int(cell_row),
                                    "cell_col": int(cell_col),
                                    "pred_class": pred_class,
                                    "clean_height_m": float(canopy_grp["h_ndsm"].mean()),
                                    "source": "measured",
                                }
                            )

                clean = pd.DataFrame(records)
                if clean.empty:
                    return clean

                missing_rows = []
                merged = all_groups.merge(clean, on=["cell_row", "cell_col", "pred_class"], how="left")
                for pred_class in (TREE_CLASS, BUILDING_CLASS):
                    class_clean = clean[clean["pred_class"] == pred_class]
                    class_missing = merged[
                        (merged["pred_class"] == pred_class) & merged["clean_height_m"].isna()
                    ]
                    if class_clean.empty or class_missing.empty:
                        continue
                    sample_xy = class_clean[["cell_col", "cell_row"]].to_numpy(dtype=np.float64)
                    sample_values = class_clean["clean_height_m"].to_numpy(dtype=np.float64)
                    query_xy = class_missing[["cell_col", "cell_row"]].to_numpy(dtype=np.float64)
                    preds = idw_predict(sample_xy, sample_values, query_xy, k=IDW_K, power=IDW_POWER)
                    for item, value in zip(class_missing.itertuples(index=False), preds):
                        missing_rows.append(
                            {
                                "cell_row": int(item.cell_row),
                                "cell_col": int(item.cell_col),
                                "pred_class": int(item.pred_class),
                                "clean_height_m": float(value),
                                "source": "interpolated",
                            }
                        )

                if missing_rows:
                    clean = pd.concat([clean, pd.DataFrame(missing_rows)], ignore_index=True)

                clean["base_height_m"] = bilinear_sample(
                    base_height,
                    clean["cell_col"].to_numpy(dtype=np.float64),
                    clean["cell_row"].to_numpy(dtype=np.float64),
                )
                clean = clean[np.isfinite(clean["base_height_m"])].reset_index(drop=True)
                return clean


            def evaluate_offset_rmse(dx_m: float, dy_m: float, photons: pd.DataFrame, pixel_size_x_m: float, pixel_size_y_m: float):
                dcol = dx_m / pixel_size_x_m
                drow = -dy_m / pixel_size_y_m
                shifted = photons.copy()
                shifted["col"] = shifted["col"] + dcol
                shifted["row"] = shifted["row"] + drow
                normalized = normalize_shifted_photons(shifted)
                clean = build_clean_cell_reference(normalized, landcover_gtb)
                if clean.empty:
                    return np.inf, clean
                residual = clean["base_height_m"] - clean["clean_height_m"]
                rmse = float(np.sqrt(np.mean(np.square(residual))))
                return rmse, clean


            def search_best_horizontal_offset(photons: pd.DataFrame, pixel_size_x_m: float, pixel_size_y_m: float):
                coarse_values = np.arange(-MAX_HORIZONTAL_OFFSET_M, MAX_HORIZONTAL_OFFSET_M + 1e-9, COARSE_OFFSET_STEP_M)
                best = {"dx_m": 0.0, "dy_m": 0.0, "rmse": np.inf, "clean": pd.DataFrame()}
                coarse_rows = []
                for dx_m in coarse_values:
                    for dy_m in coarse_values:
                        rmse, clean = evaluate_offset_rmse(dx_m, dy_m, photons, pixel_size_x_m, pixel_size_y_m)
                        coarse_rows.append({"dx_m": dx_m, "dy_m": dy_m, "rmse": rmse, "n_clean": len(clean)})
                        if rmse < best["rmse"]:
                            best = {"dx_m": dx_m, "dy_m": dy_m, "rmse": rmse, "clean": clean}

                fine_x = np.arange(best["dx_m"] - FINE_WINDOW_M, best["dx_m"] + FINE_WINDOW_M + 1e-9, FINE_OFFSET_STEP_M)
                fine_y = np.arange(best["dy_m"] - FINE_WINDOW_M, best["dy_m"] + FINE_WINDOW_M + 1e-9, FINE_OFFSET_STEP_M)
                fine_rows = []
                for dx_m in fine_x:
                    for dy_m in fine_y:
                        rmse, clean = evaluate_offset_rmse(dx_m, dy_m, photons, pixel_size_x_m, pixel_size_y_m)
                        fine_rows.append({"dx_m": dx_m, "dy_m": dy_m, "rmse": rmse, "n_clean": len(clean)})
                        if rmse < best["rmse"]:
                            best = {"dx_m": dx_m, "dy_m": dy_m, "rmse": rmse, "clean": clean}

                return best, pd.DataFrame(coarse_rows).sort_values("rmse"), pd.DataFrame(fine_rows).sort_values("rmse")
            """
        ),
        code_cell(
            """
            initialize_sliderule()

            src_date_anchor = datetime.combine(dominant_src_date_dt, datetime.min.time(), tzinfo=timezone.utc)
            photon_windows = []
            photon_df = pd.DataFrame()
            selected_window_days = None
            best_count = -1
            for days in ICESAT_WINDOW_DAYS:
                t0 = src_date_anchor - timedelta(days=days)
                t1 = src_date_anchor + timedelta(days=days)
                candidate = fetch_icesat2_photons(CASABLANCA_BBOX, t0, t1)
                count = len(candidate)
                photon_windows.append({"window_days": days, "count": count})
                if count > best_count:
                    photon_df = candidate
                    selected_window_days = days
                    best_count = count

            photon_window_df = pd.DataFrame(photon_windows)
            print("Photon retrieval attempts:")
            display(photon_window_df)

            if photon_df.empty:
                raise RuntimeError("No ICESat-2 photons were retrieved for the default AOI and search windows.")

            pixel_size_x_m, pixel_size_y_m = local_pixel_size_m(mosaic)
            best_offset, coarse_search, fine_search = search_best_horizontal_offset(
                photon_df,
                pixel_size_x_m,
                pixel_size_y_m,
            )

            clean_cells = best_offset.get("clean", pd.DataFrame()).copy().reset_index(drop=True)
            required_cols = {"base_height_m", "clean_height_m", "cell_col", "cell_row"}
            if clean_cells.empty or not required_cols.issubset(clean_cells.columns):
                raise RuntimeError(
                    "ICESat-2 photons were retrieved, but 0 usable clean calibration cells remained after "
                    "offset correction, normalization, and land-cover-aware filtering. "
                    "Increase AOI_HALF_SIZE_M and rerun from the Wayback section."
                )
            if len(clean_cells) < MIN_RF_TRAIN_SAMPLES:
                raise RuntimeError(
                    f"Only {len(clean_cells)} clean calibration cells were produced; need at least "
                    f"{MIN_RF_TRAIN_SAMPLES}. Increase AOI_HALF_SIZE_M and rerun from the Wayback section."
                )
            clean_cells["residual_m"] = clean_cells["base_height_m"] - clean_cells["clean_height_m"]
            clean_cells_path = OUTPUT_DIR / "icesat2_clean_cells.csv"
            clean_cells.to_csv(clean_cells_path, index=False)

            print("Selected ICESat search window (days):", selected_window_days)
            print("Raw photons:", len(photon_df))
            print("Best horizontal offset (m):", (best_offset["dx_m"], best_offset["dy_m"]))
            print("Best offset RMSE (m):", best_offset["rmse"])
            print("Clean cell references:", len(clean_cells))
            print("Clean cell CSV:", clean_cells_path)

            display(coarse_search.head(10))
            display(fine_search.head(10))
            display(clean_cells.head(20))

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            axes[0].imshow(base_height, cmap="viridis")
            axes[0].scatter(photon_df["col"], photon_df["row"], s=6, c="red", alpha=0.7)
            axes[0].set_title("Raw filtered ICESat-2 photons on base height")
            axes[0].axis("off")

            axes[1].imshow(base_height, cmap="viridis")
            axes[1].scatter(clean_cells["cell_col"], clean_cells["cell_row"], s=12, c=clean_cells["clean_height_m"], cmap="magma")
            axes[1].set_title("Clean cell references after offset + filtering")
            axes[1].axis("off")
            plt.show()
            """
        ),
        markdown_cell(
            """
            ## Shallow-feature residual correction

            The paper extracts shallow visual features from the MHE encoder, then learns residuals:

            - target residual = `h_pred - h_photon`
            - regressor = random forest
            - final height = `h_pred - r_hat`
            - tree pixels keep the original prediction

            The public RS3DAda repository exposes four DINOv2 intermediate layers through `get_intermediate_layers(..., 4, ...)`. Because the paper does not publish the exact shallow-block index used for calibration, this notebook uses the **shallowest of those 4 public layers** by default.
            """
        ),
        code_cell(
            """
            def patch_token_features(model: DPT_DINOv2, patch_rgb: np.ndarray, layer_index: int = FEATURE_LAYER_INDEX) -> np.ndarray:
                tensor = PATCH_TRANSFORM(image=patch_rgb.astype(np.float32))["image"].unsqueeze(0).float().to(DEVICE)
                with torch.no_grad():
                    features = model.pretrained.get_intermediate_layers(tensor, 4, return_class_token=True)
                selected = features[layer_index][0]
                patch_h = patch_rgb.shape[0] // 14
                patch_w = patch_rgb.shape[1] // 14
                feat = (
                    selected.squeeze(0)
                    .reshape(patch_h, patch_w, selected.shape[-1])
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                return feat


            def extract_point_features(
                model: DPT_DINOv2,
                image_tif: Path,
                points: pd.DataFrame,
                *,
                patch_size: int = PATCH_SIZE,
                stride: int = STRIDE,
                layer_index: int = FEATURE_LAYER_INDEX,
            ) -> np.ndarray:
                img0 = load_rgb_tiff(image_tif)
                height, width = img0.shape[:2]
                x_positions = sliding_positions(width, patch_size, stride)
                y_positions = sliding_positions(height, patch_size, stride)

                feature_sums = defaultdict(lambda: None)
                feature_counts = Counter()

                for y0 in y_positions:
                    for x0 in x_positions:
                        inside = points[
                            (points["cell_col"] >= x0)
                            & (points["cell_col"] < x0 + patch_size)
                            & (points["cell_row"] >= y0)
                            & (points["cell_row"] < y0 + patch_size)
                        ]
                        if inside.empty:
                            continue

                        patch = img0[
                            y0 : min(y0 + patch_size, height),
                            x0 : min(x0 + patch_size, width),
                        ].copy()
                        if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                            padded = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
                            padded[: patch.shape[0], : patch.shape[1]] = patch
                            patch = padded

                        feat = patch_token_features(model, patch, layer_index=layer_index)
                        patch_h, patch_w, dim = feat.shape

                        for row in inside.itertuples():
                            local_col = float(row.cell_col - x0)
                            local_row = float(row.cell_row - y0)
                            token_col = int(np.clip(math.floor(local_col / 14.0), 0, patch_w - 1))
                            token_row = int(np.clip(math.floor(local_row / 14.0), 0, patch_h - 1))
                            vec = feat[token_row, token_col]
                            key = int(row.Index)
                            if feature_sums[key] is None:
                                feature_sums[key] = vec.copy()
                            else:
                                feature_sums[key] += vec
                            feature_counts[key] += 1

                features = np.full((len(points), 1024), np.nan, dtype=np.float32)
                for idx in range(len(points)):
                    if feature_counts[idx] > 0:
                        features[idx] = feature_sums[idx] / feature_counts[idx]
                return features


            def predict_dense_residual_map(
                model: DPT_DINOv2,
                rf: RandomForestRegressor,
                image_tif: Path,
                *,
                patch_size: int = PATCH_SIZE,
                stride: int = STRIDE,
                layer_index: int = FEATURE_LAYER_INDEX,
            ) -> np.ndarray:
                img0 = load_rgb_tiff(image_tif)
                height, width = img0.shape[:2]

                x_positions = sliding_positions(width, patch_size, stride)
                y_positions = sliding_positions(height, patch_size, stride)
                padded_h = patch_size + stride * (len(y_positions) - 1)
                padded_w = patch_size + stride * (len(x_positions) - 1)

                img1 = np.zeros((padded_h, padded_w, 3), dtype=np.uint8)
                img1[:height, :width] = img0.copy()

                weights = build_blend_weights(patch_size, stride)
                blend = np.zeros((padded_h, padded_w), dtype=np.float32)
                residual_all = np.zeros((padded_h, padded_w), dtype=np.float32)

                for y0 in y_positions:
                    for x0 in x_positions:
                        patch = img1[y0 : y0 + patch_size, x0 : x0 + patch_size].copy()
                        feat = patch_token_features(model, patch, layer_index=layer_index)
                        patch_h, patch_w, dim = feat.shape
                        token_pred = rf.predict(feat.reshape(-1, dim)).reshape(patch_h, patch_w).astype(np.float32)
                        patch_pred = np.repeat(np.repeat(token_pred, 14, axis=0), 14, axis=1)
                        residual_all[y0 : y0 + patch_size, x0 : x0 + patch_size] += patch_pred * weights
                        blend[y0 : y0 + patch_size, x0 : x0 + patch_size] += weights

                residual_all = residual_all / blend
                return residual_all[:height, :width]


            height_model = build_rs3dada_model(height_ckpt)
            point_features = extract_point_features(
                height_model,
                mosaic["geotiff_path"],
                clean_cells.reset_index(drop=True),
                patch_size=PATCH_SIZE,
                stride=STRIDE,
                layer_index=FEATURE_LAYER_INDEX,
            )
            feature_valid = np.isfinite(point_features).all(axis=1)
            train_df = clean_cells.loc[feature_valid].reset_index(drop=True).copy()
            train_x = point_features[feature_valid]
            train_y = (train_df["base_height_m"] - train_df["clean_height_m"]).to_numpy(dtype=np.float32)

            if len(train_df) < MIN_RF_TRAIN_SAMPLES:
                raise RuntimeError(
                    f"Only {len(train_df)} RF training samples were available after feature extraction; "
                    f"need at least {MIN_RF_TRAIN_SAMPLES}."
                )

            rf = RandomForestRegressor(
                random_state=RF_RANDOM_STATE,
                n_jobs=-1,
            )
            rf.fit(train_x, train_y)

            residual_map = predict_dense_residual_map(
                height_model,
                rf,
                mosaic["geotiff_path"],
                patch_size=PATCH_SIZE,
                stride=STRIDE,
                layer_index=FEATURE_LAYER_INDEX,
            ).astype(np.float32)
            del height_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            corrected_height = (base_height - residual_map).astype(np.float32)
            corrected_height[landcover_gtb == TREE_CLASS] = base_height[landcover_gtb == TREE_CLASS]
            corrected_height = np.clip(corrected_height, a_min=0.0, a_max=None)

            residual_path = save_raster_like(
                mosaic["geotiff_path"],
                OUTPUT_DIR / "rs3dada_residual_map.tif",
                residual_map,
                dtype="float32",
            )
            corrected_path = save_raster_like(
                mosaic["geotiff_path"],
                OUTPUT_DIR / "rs3dada_corrected_height.tif",
                corrected_height,
                dtype="float32",
            )

            rf_train_path = OUTPUT_DIR / "rf_training_points.csv"
            train_df.assign(target_residual_m=train_y).to_csv(rf_train_path, index=False)

            print("RF training samples:", len(train_df))
            print("Residual raster:", residual_path)
            print("Corrected height raster:", corrected_path)
            print("RF training table:", rf_train_path)

            fig, axes = plt.subplots(1, 3, figsize=(20, 6))
            im0 = axes[0].imshow(base_height, cmap="viridis")
            axes[0].set_title("Base height")
            axes[0].axis("off")
            plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

            im1 = axes[1].imshow(residual_map, cmap="coolwarm")
            axes[1].set_title("Learned residual")
            axes[1].axis("off")
            plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

            im2 = axes[2].imshow(corrected_height, cmap="viridis")
            axes[2].set_title("Corrected height")
            axes[2].axis("off")
            plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
            plt.show()
            """
        ),
        markdown_cell(
            """
            ## Per-building export

            The paper's core output is a corrected dense height raster. For building-height estimation, this notebook turns the RS3DAda **building class** into connected components and computes per-component roof-height statistics from the corrected raster.

            This final step is a practical post-processing layer, not part of the published calibration architecture itself.
            """
        ),
        code_cell(
            """
            def component_lonlat_centroid(rows: np.ndarray, cols: np.ndarray, mosaic: dict) -> tuple[float, float]:
                row = float(np.mean(rows))
                col = float(np.mean(cols))
                return pixel_to_lonlat(col, row, mosaic)


            def extract_building_instances(segmentation: np.ndarray, corrected_height: np.ndarray) -> pd.DataFrame:
                building_mask = segmentation == BUILDING_CLASS_INDEX
                structure = np.ones((3, 3), dtype=np.uint8)
                labels, num = ndimage.label(building_mask.astype(np.uint8), structure=structure)

                rows = []
                features = []
                for label_id in range(1, num + 1):
                    component = labels == label_id
                    pixel_count = int(component.sum())
                    if pixel_count < MIN_BUILDING_COMPONENT_PIXELS:
                        continue

                    vals = corrected_height[component]
                    vals = vals[np.isfinite(vals)]
                    if vals.size == 0:
                        continue

                    rr, cc = np.where(component)
                    centroid_lon, centroid_lat = component_lonlat_centroid(rr, cc, mosaic)
                    px_x_m, px_y_m = local_pixel_size_m(mosaic)
                    area_m2 = pixel_count * px_x_m * px_y_m

                    component_shapes = list(
                        shapes(
                            component.astype(np.uint8),
                            mask=component.astype(bool),
                            transform=mosaic["transform"],
                        )
                    )
                    geoms = [shape(geom) for geom, value in component_shapes if value == 1]
                    geom = unary_union(geoms) if geoms else None

                    rows.append(
                        {
                            "building_id": len(rows),
                            "pixel_count": pixel_count,
                            "area_m2": float(area_m2),
                            "height_m_median": float(np.median(vals)),
                            "height_m_mean": float(np.mean(vals)),
                            "height_m_p90": float(np.percentile(vals, 90)),
                            "height_m_max": float(np.max(vals)),
                            "centroid_lon": float(centroid_lon),
                            "centroid_lat": float(centroid_lat),
                        }
                    )
                    if geom is not None:
                        geom = shapely_transform(lambda x, y, z=None: TO_4326.transform(x, y), geom)
                        features.append(
                            {
                                "type": "Feature",
                                "properties": {
                                    "building_id": len(rows) - 1,
                                    "height_m_median": float(np.median(vals)),
                                    "height_m_p90": float(np.percentile(vals, 90)),
                                    "height_m_max": float(np.max(vals)),
                                    "area_m2": float(area_m2),
                                },
                                "geometry": mapping(geom),
                            }
                        )

                geojson = {"type": "FeatureCollection", "features": features}
                with open(OUTPUT_DIR / "building_instances.geojson", "w") as f:
                    json.dump(geojson, f)

                df = pd.DataFrame(rows).sort_values("height_m_p90", ascending=False).reset_index(drop=True)
                df.to_csv(OUTPUT_DIR / "building_height_estimates.csv", index=False)
                return df


            building_instances = extract_building_instances(segmentation, corrected_height)
            print("Per-building CSV:", OUTPUT_DIR / "building_height_estimates.csv")
            print("Per-building GeoJSON:", OUTPUT_DIR / "building_instances.geojson")
            display(building_instances.head(20))

            plt.figure(figsize=(8, 8))
            plt.imshow(mosaic["array"])
            if not building_instances.empty:
                for item in building_instances.head(20).itertuples():
                    col, row = lonlat_to_pixel(item.centroid_lon, item.centroid_lat, mosaic)
                    plt.text(col, row, f"{item.height_m_p90:.1f} m", color="yellow", fontsize=8)
            plt.title("Top building components with corrected p90 heights")
            plt.axis("off")
            plt.show()

            print("Artifacts in", OUTPUT_DIR)
            print("- Wayback geotiff:", mosaic["geotiff_path"])
            print("- Base height:", base_height_path)
            print("- Segmentation:", segmentation_path)
            print("- Ground/tree/building:", gtb_path)
            print("- Clean ICESat-2 cells:", clean_cells_path)
            print("- Residual map:", residual_path)
            print("- Corrected height:", corrected_path)
            print("- Building heights:", OUTPUT_DIR / "building_height_estimates.csv")
            """
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(build_notebook(), indent=2))
    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
