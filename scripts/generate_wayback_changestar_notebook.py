from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


NOTEBOOK_PATH = Path(
    "/Users/tahaelouali/casablanca-builtup-pipeline/notebooks/"
    "casablanca_new_building_wayback_changestar_kaggle.ipynb"
)


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
            # Casablanca New-Building Detection with Wayback + Changen2-Pretrained ChangeStar

            This notebook uses the **official public Changen2-pretrained ChangeStar inference path** to detect **new buildings** between two ArcGIS Wayback releases over Casablanca.

            Core model sources:

            - `torchange` / `pytorch-change-models`
            - `s1_init_s1c1_changestar_vitl_1x256()`
            - official checkpoint family `EVER-Z/Changen2-ChangeStar1x256`
            - requested building-change checkpoint family containing `s1c1_cstar_vitl_1x256.pth`

            Pipeline:

            1. Discover ArcGIS Wayback releases from the live WMTS capabilities document.
            2. Select two releases and download two georeferenced RGB mosaics over the same AOI.
            3. Load the official **Changen2-pretrained ChangeStar ViT-L** building-change model.
            4. Run tiled inference with the **official preprocessing** from the author demo:
               - `A.Normalize()`
               - `ToTensorV2()`
               - concatenate `t1` and `t2` into a 6-channel tensor
            5. Recover the official three outputs:
               - `change_prediction`
               - `t1_semantic_prediction`
               - `t2_semantic_prediction`
            6. Derive a **new-building mask** as:
               - changed
               - non-building at `t1`
               - building at `t2`
            7. Export rasters, polygons, and a CSV summary.

            ## Important correctness notes

            - The **model loading path** matches the public author README and demo notebook:
              - `from torchange.models.changen2 import s1_init_s1c1_changestar_vitl_1x256`
            - The **checkpoint family** matches the public Changen2 README, which lists the building-change models and points to the inference demo.
            - The public demo is a **single-pair inference demo**, not a complete geospatial Wayback notebook. So this notebook keeps the official model usage intact and adds the geospatial glue needed for:
              - Wayback acquisition
              - tiled inference over larger mosaics
              - polygon export
            - The **new-building mask** is a deterministic post-processing layer built from the model's own outputs:
              - `new = change & (~t1_building) & t2_building`
              This is the correct practical way to convert ChangeStar's change + semantic outputs into a directional `new building` product.
            """
        ),
        markdown_cell(
            """
            ## Primary sources

            - Official `torchange` README: [Z-Zheng/pytorch-change-models](https://github.com/Z-Zheng/pytorch-change-models/blob/main/README.md)
            - Official Changen2 README: [torchange/models/changen2/README.md](https://github.com/Z-Zheng/pytorch-change-models/blob/main/torchange/models/changen2/README.md)
            - Official ChangeStar 1xd implementation: [changestar_1xd.py](https://github.com/Z-Zheng/pytorch-change-models/blob/main/torchange/models/changestar_1xd.py)
            - Official Changen2 pretrained ChangeStar loader: [_changestar_1x256.py](https://github.com/Z-Zheng/pytorch-change-models/blob/main/torchange/models/changen2/_changestar_1x256.py)
            - Official inference demo: [changen2_pretrained_changestar1x256_inference_demo.ipynb](https://github.com/Z-Zheng/pytorch-change-models/blob/main/examples/changen2_pretrained_changestar1x256_inference_demo.ipynb)
            - Official Changen2 checkpoint repo: [EVER-Z/Changen2-ChangeStar1x256](https://huggingface.co/EVER-Z/Changen2-ChangeStar1x256)
            - Official ChangeStar project page: [ChangeStar](https://zhuozheng.top/changestar/)
            - ArcGIS Wayback metadata overview: [Wayback with World Imagery Metadata](https://www.esri.com/arcgis-blog/products/arcgis-living-atlas/imagery/wayback-with-world-imagery-metadata)
            - Live ArcGIS Wayback WMTS capabilities: [WMTSCapabilities.xml](https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/MapServer/WMTS/1.0.0/WMTSCapabilities.xml)

            ## Kaggle requirements

            - Turn **Internet** on.
            - Use a **T4 GPU** on Kaggle.
            - Do **not** use `P100` in the current Kaggle image: the shipped PyTorch build does not support `sm_60`.
            - Keep the AOI moderate on first run. The ViT-L model is heavy, and two large Wayback mosaics increase runtime.
            """
        ),
        code_cell(
            """
            import subprocess
            import sys


            def run(cmd):
                printable = " ".join(str(x) for x in cmd)
                print("+", printable)
                subprocess.run([str(x) for x in cmd], check=True)


            run([sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/Z-Zheng/pytorch-change-models"])
            run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-q",
                    "lxml",
                    "rasterio",
                    "pyproj",
                    "shapely",
                    "scipy",
                    "opencv-python-headless",
                ]
            )
            """
        ),
        code_cell(
            """
            import io
            import json
            import math
            import re
            import warnings
            from pathlib import Path

            import albumentations as A
            from albumentations.pytorch import ToTensorV2
            import ever as er
            from huggingface_hub import hf_hub_download
            from lxml import etree
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            from PIL import Image
            import rasterio
            from rasterio.features import shapes
            from rasterio.transform import from_bounds
            from rasterio.warp import reproject
            from rasterio.enums import Resampling
            from pyproj import Geod, Transformer
            import requests
            from scipy import ndimage
            from shapely.geometry import mapping, shape
            from shapely.ops import transform as shapely_transform
            import torch
            from torchange.models.changen2 import s1_init_s1c1_changestar_vitl_1x256

            %config InlineBackend.figure_format = "retina"

            pd.set_option("display.max_columns", 200)
            pd.set_option("display.max_rows", 200)
            warnings.filterwarnings("ignore", category=FutureWarning)
            torch.set_grad_enabled(False)
            """
        ),
        code_cell(
            """
            OUTPUT_DIR = Path("/kaggle/working/casablanca_wayback_changestar")
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

            WMTS_CAPABILITIES_URL = (
                "https://wayback.maptiles.arcgis.com/arcgis/rest/services/"
                "World_Imagery/MapServer/WMTS/1.0.0/WMTSCapabilities.xml"
            )

            CASABLANCA_CENTER = {"lon": -7.62003, "lat": 33.59451}
            AOI_HALF_SIZE_M = 500.0
            T1_RELEASE_SELECTOR = "latest_minus_12"
            T2_RELEASE_SELECTOR = "latest"
            ZOOM = 19
            MAX_TILES = 500

            PATCH_SIZE = 1024
            STRIDE = 768

            CHANGE_THRESHOLD = 0.50
            SEMANTIC_THRESHOLD = 0.50
            MIN_NEW_BUILDING_PIXELS = 40


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
            AUTO_DEVICE = torch.device(DEVICE)
            GEOD = Geod(ellps="WGS84")
            TO_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
            TO_4326 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
            PREPROCESS = A.Compose(
                [A.Normalize(), ToTensorV2()],
                additional_targets={"image2": "image"},
            )

            session = requests.Session()
            session.headers.update(
                {
                    "User-Agent": "Kaggle-Wayback-ChangeStar/1.0",
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

            print("DEVICE:", DEVICE)
            if torch.cuda.is_available():
                print("GPU:", torch.cuda.get_device_name(0))
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
                selector = str(selector).strip()

                if selector == "latest":
                    return releases.iloc[-1]

                latest_minus = re.fullmatch(r"latest_minus_(\\d+)", selector)
                if latest_minus:
                    offset = int(latest_minus.group(1))
                    idx = len(releases) - 1 - offset
                    if idx < 0:
                        raise ValueError(f"Selector {selector!r} is older than the first available release.")
                    return releases.iloc[idx]

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

                raise ValueError(
                    "Release selector must be 'latest', 'latest_minus_N', a Wayback identifier, or YYYY-MM-DD."
                )


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
                label: str = "scene",
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

                png_path = out_dir / f"{label}_{release_row['identifier']}_z{zoom}.png"
                tif_path = out_dir / f"{label}_{release_row['identifier']}_z{zoom}.tif"
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
                    "release_date": release_row["release_date"],
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


            def summarize_scene(release_row: pd.Series, label: str) -> dict:
                metadata_base_url = metadata_base_url_from_identifier(release_row["identifier"])
                metadata_layers = get_metadata_layers(metadata_base_url)
                metadata_grid = sample_wayback_metadata_grid(metadata_base_url, CASABLANCA_BBOX, n=5)
                mosaic = download_wayback_mosaic(release_row, CASABLANCA_BBOX, label=label)

                dominant_src_date = None
                dominant_src_res = None
                if not metadata_grid.empty and metadata_grid["SRC_DATE"].dropna().shape[0] > 0:
                    dominant_src_date = int(metadata_grid["SRC_DATE"].dropna().mode().iloc[0])
                    dominant_src_date = pd.to_datetime(str(dominant_src_date), format="%Y%m%d").date()
                if not metadata_grid.empty and metadata_grid["SRC_RES"].dropna().shape[0] > 0:
                    dominant_src_res = float(metadata_grid["SRC_RES"].dropna().median())

                return {
                    "release": release_row,
                    "metadata_base_url": metadata_base_url,
                    "metadata_layers": metadata_layers,
                    "metadata_grid": metadata_grid,
                    "mosaic": mosaic,
                    "dominant_src_date": dominant_src_date,
                    "dominant_src_res": dominant_src_res,
                }
            """
        ),
        code_cell(
            """
            releases = parse_wmts_capabilities()
            release_t1 = select_release(releases, T1_RELEASE_SELECTOR)
            release_t2 = select_release(releases, T2_RELEASE_SELECTOR)

            if release_t1["identifier"] == release_t2["identifier"]:
                raise RuntimeError("T1 and T2 selected the same Wayback release. Choose two different releases.")

            ordered = pd.DataFrame([release_t1, release_t2]).sort_values("release_date").reset_index(drop=True)
            release_t1 = ordered.iloc[0]
            release_t2 = ordered.iloc[1]

            scene_t1 = summarize_scene(release_t1, "t1")
            scene_t2 = summarize_scene(release_t2, "t2")

            mosaic_t1 = scene_t1["mosaic"]
            mosaic_t2 = scene_t2["mosaic"]

            if mosaic_t1["tile_range"] != mosaic_t2["tile_range"]:
                raise RuntimeError("The two Wayback mosaics landed on different tile ranges. Keep the same AOI and zoom.")
            if mosaic_t1["array"].shape != mosaic_t2["array"].shape:
                raise RuntimeError("The two Wayback mosaics have different shapes. This notebook expects identical grids.")

            release_summary = pd.DataFrame(
                [
                    {
                        "label": "t1",
                        "identifier": scene_t1["release"]["identifier"],
                        "release_date": scene_t1["release"]["release_date"],
                        "dominant_src_date": scene_t1["dominant_src_date"],
                        "dominant_src_res_m": scene_t1["dominant_src_res"],
                        "tile_count": mosaic_t1["tile_count"],
                        "pixel_size_m_x_y": local_pixel_size_m(mosaic_t1),
                        "geotiff_path": str(mosaic_t1["geotiff_path"]),
                    },
                    {
                        "label": "t2",
                        "identifier": scene_t2["release"]["identifier"],
                        "release_date": scene_t2["release"]["release_date"],
                        "dominant_src_date": scene_t2["dominant_src_date"],
                        "dominant_src_res_m": scene_t2["dominant_src_res"],
                        "tile_count": mosaic_t2["tile_count"],
                        "pixel_size_m_x_y": local_pixel_size_m(mosaic_t2),
                        "geotiff_path": str(mosaic_t2["geotiff_path"]),
                    },
                ]
            )
            release_summary_path = OUTPUT_DIR / "wayback_pair_summary.csv"
            release_summary.to_csv(release_summary_path, index=False)

            print("Selected Wayback pair:")
            display(release_summary)

            fig, axes = plt.subplots(1, 2, figsize=(14, 7))
            axes[0].imshow(mosaic_t1["array"])
            axes[0].set_title(f"T1 | {scene_t1['release']['identifier']} | SRC_DATE {scene_t1['dominant_src_date']}")
            axes[0].axis("off")
            axes[1].imshow(mosaic_t2["array"])
            axes[1].set_title(f"T2 | {scene_t2['release']['identifier']} | SRC_DATE {scene_t2['dominant_src_date']}")
            axes[1].axis("off")
            plt.show()
            """
        ),
        markdown_cell(
            """
            ## Official Changen2-ChangeStar model loading

            The next cell follows the public author materials directly:

            - loader: `s1_init_s1c1_changestar_vitl_1x256()`
            - checkpoint family: `EVER-Z/Changen2-ChangeStar1x256`
            - official preprocessing from the demo notebook:
              - `A.Normalize()`
              - `ToTensorV2()`

            The `vitl` loader already points to the requested building-change model family. The cell below also resolves the exact public checkpoint filename for transparency.
            """
        ),
        code_cell(
            """
            exact_weight_path = hf_hub_download(
                "EVER-Z/Changen2-ChangeStar1x256",
                "s1c1_cstar_vitl_1x256.pth",
            )
            print("Resolved official checkpoint:", exact_weight_path)

            model = s1_init_s1c1_changestar_vitl_1x256()
            model.eval()
            model.to(AUTO_DEVICE)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            """
        ),
        code_cell(
            """
            def save_single_band_like(reference_path: Path, output_path: Path, array: np.ndarray, dtype: str) -> Path:
                with rasterio.open(reference_path) as src:
                    profile = src.profile.copy()
                profile.update(driver="GTiff", count=1, dtype=dtype)
                with rasterio.open(output_path, "w", **profile) as dst:
                    dst.write(array.astype(dtype), 1)
                return output_path


            def sliding_positions(length: int, patch_size: int, stride: int) -> list[int]:
                if length <= patch_size:
                    return [0]
                positions = list(range(0, length - patch_size + 1, stride))
                if positions[-1] != length - patch_size:
                    positions.append(length - patch_size)
                return positions


            def pad_patch_rgb(patch: np.ndarray, patch_size: int) -> np.ndarray:
                if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                    return patch
                padded = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
                padded[: patch.shape[0], : patch.shape[1], :] = patch
                return padded


            @torch.inference_mode()
            def predict_patch_probabilities(model, patch_t1: np.ndarray, patch_t2: np.ndarray) -> dict[str, np.ndarray]:
                data = PREPROCESS(image=patch_t1, image2=patch_t2)
                img = torch.cat([data["image"], data["image2"]], dim=0)
                prediction = model(img.unsqueeze(0).to(AUTO_DEVICE))

                def to_numpy(tensor: torch.Tensor) -> np.ndarray:
                    return tensor.detach().squeeze().to(torch.float32).cpu().numpy().astype(np.float32)

                return {
                    "change_prediction": to_numpy(prediction["change_prediction"]),
                    "t1_semantic_prediction": to_numpy(prediction["t1_semantic_prediction"]),
                    "t2_semantic_prediction": to_numpy(prediction["t2_semantic_prediction"]),
                }


            @torch.inference_mode()
            def run_tiled_inference(model, arr_t1: np.ndarray, arr_t2: np.ndarray) -> dict[str, np.ndarray]:
                if arr_t1.shape != arr_t2.shape:
                    raise ValueError("T1 and T2 arrays must have identical shapes.")

                height, width = arr_t1.shape[:2]
                xs = sliding_positions(width, PATCH_SIZE, STRIDE)
                ys = sliding_positions(height, PATCH_SIZE, STRIDE)

                change_sum = np.zeros((height, width), dtype=np.float32)
                t1_sum = np.zeros((height, width), dtype=np.float32)
                t2_sum = np.zeros((height, width), dtype=np.float32)
                counts = np.zeros((height, width), dtype=np.float32)

                total_patches = len(xs) * len(ys)
                patch_index = 0

                for y0 in ys:
                    for x0 in xs:
                        patch_index += 1
                        y1 = min(y0 + PATCH_SIZE, height)
                        x1 = min(x0 + PATCH_SIZE, width)

                        patch_t1 = arr_t1[y0:y1, x0:x1].copy()
                        patch_t2 = arr_t2[y0:y1, x0:x1].copy()
                        orig_h, orig_w = patch_t1.shape[:2]

                        patch_t1 = pad_patch_rgb(patch_t1, PATCH_SIZE)
                        patch_t2 = pad_patch_rgb(patch_t2, PATCH_SIZE)

                        pred = predict_patch_probabilities(model, patch_t1, patch_t2)
                        change = pred["change_prediction"][:orig_h, :orig_w]
                        t1_sem = pred["t1_semantic_prediction"][:orig_h, :orig_w]
                        t2_sem = pred["t2_semantic_prediction"][:orig_h, :orig_w]

                        change_sum[y0:y1, x0:x1] += change
                        t1_sum[y0:y1, x0:x1] += t1_sem
                        t2_sum[y0:y1, x0:x1] += t2_sem
                        counts[y0:y1, x0:x1] += 1.0

                        if patch_index == 1 or patch_index == total_patches or patch_index % 5 == 0:
                            print(f"Processed patch {patch_index}/{total_patches}")

                counts = np.maximum(counts, 1.0)
                return {
                    "change_prediction": change_sum / counts,
                    "t1_semantic_prediction": t1_sum / counts,
                    "t2_semantic_prediction": t2_sum / counts,
                }


            def remove_small_components(mask: np.ndarray, min_pixels: int) -> tuple[np.ndarray, np.ndarray]:
                labeled, num = ndimage.label(mask.astype(np.uint8))
                keep = np.zeros_like(mask, dtype=bool)
                for label_id in range(1, num + 1):
                    component = labeled == label_id
                    if int(component.sum()) >= min_pixels:
                        keep |= component
                kept_labels, _ = ndimage.label(keep.astype(np.uint8))
                return keep, kept_labels


            def vectorize_new_buildings(mask: np.ndarray, reference_path: Path) -> tuple[pd.DataFrame, dict]:
                records = []
                features_out = []

                with rasterio.open(reference_path) as src:
                    transform = src.transform

                for idx, (geom, value) in enumerate(
                    shapes(mask.astype(np.uint8), mask=mask.astype(np.uint8), transform=transform),
                    start=1,
                ):
                    if int(value) != 1:
                        continue

                    geom_3857 = shape(geom)
                    geom_wgs84 = shapely_transform(lambda x, y, z=None: TO_4326.transform(x, y), geom_3857)
                    area_m2 = abs(GEOD.geometry_area_perimeter(geom_wgs84)[0])
                    centroid = geom_wgs84.centroid

                    record = {
                        "building_id": idx,
                        "area_m2": float(area_m2),
                        "centroid_lon": float(centroid.x),
                        "centroid_lat": float(centroid.y),
                        "release_t1": scene_t1["release"]["identifier"],
                        "release_t2": scene_t2["release"]["identifier"],
                        "src_date_t1": str(scene_t1["dominant_src_date"]),
                        "src_date_t2": str(scene_t2["dominant_src_date"]),
                    }
                    records.append(record)
                    features_out.append(
                        {
                            "type": "Feature",
                            "geometry": mapping(geom_wgs84),
                            "properties": record,
                        }
                    )

                return pd.DataFrame(records), {"type": "FeatureCollection", "features": features_out}
            """
        ),
        code_cell(
            """
            probs = run_tiled_inference(model, mosaic_t1["array"], mosaic_t2["array"])

            change_prob = probs["change_prediction"]
            t1_building_prob = probs["t1_semantic_prediction"]
            t2_building_prob = probs["t2_semantic_prediction"]

            change_mask = change_prob >= CHANGE_THRESHOLD
            t1_building_mask = t1_building_prob >= SEMANTIC_THRESHOLD
            t2_building_mask = t2_building_prob >= SEMANTIC_THRESHOLD
            new_building_mask_raw = change_mask & (~t1_building_mask) & t2_building_mask
            new_building_mask, new_building_labels = remove_small_components(
                new_building_mask_raw,
                MIN_NEW_BUILDING_PIXELS,
            )

            change_prob_path = save_single_band_like(
                mosaic_t2["geotiff_path"],
                OUTPUT_DIR / "change_probability.tif",
                change_prob,
                "float32",
            )
            t1_prob_path = save_single_band_like(
                mosaic_t2["geotiff_path"],
                OUTPUT_DIR / "t1_building_probability.tif",
                t1_building_prob,
                "float32",
            )
            t2_prob_path = save_single_band_like(
                mosaic_t2["geotiff_path"],
                OUTPUT_DIR / "t2_building_probability.tif",
                t2_building_prob,
                "float32",
            )
            new_building_mask_path = save_single_band_like(
                mosaic_t2["geotiff_path"],
                OUTPUT_DIR / "new_building_mask.tif",
                new_building_mask.astype(np.uint8),
                "uint8",
            )

            print("Output rasters:")
            print("-", change_prob_path)
            print("-", t1_prob_path)
            print("-", t2_prob_path)
            print("-", new_building_mask_path)
            print("Detected new-building components:", int(new_building_labels.max()))

            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes[0, 0].imshow(mosaic_t1["array"])
            axes[0, 0].set_title("T1 Wayback RGB")
            axes[0, 0].axis("off")

            axes[0, 1].imshow(mosaic_t2["array"])
            axes[0, 1].set_title("T2 Wayback RGB")
            axes[0, 1].axis("off")

            im = axes[0, 2].imshow(change_prob, cmap="magma", vmin=0.0, vmax=1.0)
            axes[0, 2].set_title("Change probability")
            axes[0, 2].axis("off")
            plt.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)

            im = axes[1, 0].imshow(t1_building_prob, cmap="viridis", vmin=0.0, vmax=1.0)
            axes[1, 0].set_title("T1 building probability")
            axes[1, 0].axis("off")
            plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

            im = axes[1, 1].imshow(t2_building_prob, cmap="viridis", vmin=0.0, vmax=1.0)
            axes[1, 1].set_title("T2 building probability")
            axes[1, 1].axis("off")
            plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

            axes[1, 2].imshow(mosaic_t2["array"])
            axes[1, 2].imshow(new_building_mask, cmap="autumn", alpha=0.45)
            axes[1, 2].set_title("Derived new-building mask")
            axes[1, 2].axis("off")

            plt.tight_layout()
            plt.show()
            """
        ),
        code_cell(
            """
            new_buildings_df, new_buildings_geojson = vectorize_new_buildings(
                new_building_mask,
                mosaic_t2["geotiff_path"],
            )

            csv_path = OUTPUT_DIR / "new_buildings.csv"
            geojson_path = OUTPUT_DIR / "new_buildings.geojson"
            new_buildings_df.to_csv(csv_path, index=False)
            geojson_path.write_text(json.dumps(new_buildings_geojson))

            print("Vector outputs:")
            print("-", csv_path)
            print("-", geojson_path)
            print("Total new-building polygons:", len(new_buildings_df))

            if not new_buildings_df.empty:
                display(new_buildings_df.sort_values("area_m2", ascending=False).head(20))
            else:
                print("No new-building polygons survived the thresholds for the current Wayback pair and AOI.")
            """
        ),
        markdown_cell(
            """
            ## Result interpretation

            This notebook outputs:

            - `change_probability.tif`
            - `t1_building_probability.tif`
            - `t2_building_probability.tif`
            - `new_building_mask.tif`
            - `new_buildings.csv`
            - `new_buildings.geojson`

            The directional `new building` result is intentionally stricter than a plain change mask:

            - high change probability
            - low building probability at `t1`
            - high building probability at `t2`

            If you want a more permissive detector, lower:

            - `CHANGE_THRESHOLD`
            - `SEMANTIC_THRESHOLD`
            - `MIN_NEW_BUILDING_PIXELS`
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
