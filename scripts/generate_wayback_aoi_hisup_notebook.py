from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "building_polygon_extraction_wayback_hisup_kaggle.ipynb"


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
            # Building Polygon Extraction from ESRI Wayback VHR Imagery using HiSup (Hierarchical Supervision)

            ## Section 1 — Introduction

            **HiSup** predicts three complementary building-shape signals from a single very-high-resolution RGB image: a building mask, a junction (corner) map, and an attraction-field / line-structure signal. The central idea is **hierarchical supervision**: low-level corner and boundary cues help the network refine high-level building masks, while the refined masks constrain polygon reconstruction. In the public code this appears as the `mask`, `jloc`, and `afm` prediction branches plus the final remasking stage in `hisup/detector.py`.

            Corner detection matters because a plain semantic segmentation network usually returns soft blob-like masks. HiSup instead predicts vertices explicitly and uses them during polygon generation, which preserves sharp turns and right-angle corners that are needed for GIS-ready building footprints.

            The three supervision heads learn:

            - **Mask head**: building interior / foreground occupancy.
            - **Junction head**: concave and convex corner evidence used to localize polygon vertices.
            - **AFM head**: attraction-field / boundary structure that guides the final remasking branch and polygon regularity.

            HiSup also ships a **PoLiS** implementation (`hisup/utils/metrics/polis.py`). PoLiS is a polygon-to-polygon boundary distance metric: lower values mean closer geometric agreement between predicted and reference polygons.

            This notebook uses:

            - **Input**: a single RGB ESRI Wayback VHR mosaic over the target AOI.
            - **Output**: a `GeoDataFrame`, `GeoJSON`, and `GeoPackage` of building polygons georeferenced to `EPSG:4326`, plus pixel-space centroid columns that preserve the image coordinate frame.

            ### Important correctness notes

            The public HiSup repo does **not** use some paths and defaults from the prompt. To stay scientifically correct, the notebook follows the actual public implementation:

            - official config file: `config-files/crowdai_hrnet48.yaml`
            - official pretrained-loading logic from `hisup/detector.py`
            - official transform path from `hisup/dataset/build.py`
            - official polygon reconstruction from `BuildingDetector.forward_test()`

            **Sources**

            - HiSup paper: [ISPRS JPRS 2023](https://doi.org/10.1016/j.isprsjprs.2023.03.006)
            - Official HiSup repo: [SarahwXU/HiSup](https://github.com/SarahwXU/HiSup)
            - Official demo path: [scripts/demo.py](https://github.com/SarahwXU/HiSup/blob/main/scripts/demo.py)
            """
        ),
        markdown_cell(
            """
            ## Section 2 — Environment setup

            This notebook installs only the packages needed for the public HiSup inference path and for the required geospatial export / visualization steps. Two additional packages beyond the user prompt are installed because the public repo imports them directly:

            - `yacs` for `hisup.config.cfg`
            - `tqdm` for progress bars

            `matplotlib` and `descartes` are also required for the notebook's polygon visualization and match the public repo requirements file.

            **Sources**

            - Official requirements: [requirements.txt](https://github.com/SarahwXU/HiSup/blob/main/requirements.txt)
            - Official README install section: [README.md](https://github.com/SarahwXU/HiSup/blob/main/README.md)
            """
        ),
        code_cell(
            """
            import importlib
            import subprocess
            import sys
            from pathlib import Path


            def ensure_packages(packages: dict[str, str]) -> None:
                \"\"\"Install only missing Python packages needed by this notebook.\"\"\"
                missing = []
                for module_name, package_name in packages.items():
                    try:
                        importlib.import_module(module_name)
                    except ImportError:
                        missing.append(package_name)
                if missing:
                    print("Installing missing packages:", missing)
                    subprocess.run(
                        [sys.executable, "-m", "pip", "install", "-q", *missing],
                        check=True,
                    )
                else:
                    print("All required packages are already available.")


            REQUIRED_PACKAGES = {
                "torch": "torch",
                "torchvision": "torchvision",
                "cv2": "opencv-python",
                "shapely": "shapely",
                "pycocotools": "pycocotools",
                "scipy": "scipy",
                "skimage": "scikit-image",
                "geopandas": "geopandas",
                "rasterio": "rasterio",
                "pyproj": "pyproj",
                "requests": "requests",
                "PIL": "Pillow",
                "yacs": "yacs",
                "tqdm": "tqdm",
                "matplotlib": "matplotlib",
                "descartes": "descartes",
            }

            ensure_packages(REQUIRED_PACKAGES)

            HISUP_REPO_DIR = Path("/kaggle/working/HiSup")
            if not HISUP_REPO_DIR.exists():
                subprocess.run(
                    ["git", "clone", "--depth", "1", "https://github.com/SarahwXU/HiSup.git", str(HISUP_REPO_DIR)],
                    check=True,
                )
            else:
                print("Reusing cloned repo:", HISUP_REPO_DIR)

            if str(HISUP_REPO_DIR) not in sys.path:
                sys.path.insert(0, str(HISUP_REPO_DIR))

            print("HiSup repo:", HISUP_REPO_DIR)
            print("Config file exists:", (HISUP_REPO_DIR / "config-files" / "crowdai_hrnet48.yaml").exists())
            """
        ),
        markdown_cell(
            """
            ## Section 3 — Configuration

            The prompt asked for a stale Wayback release number and for COCO normalization values. To keep the notebook executable and faithful to the public HiSup demo, the notebook preserves those prompt fields in `CONFIG` but:

            - falls back to a live Wayback release if `WAYBACK_RELEASE_NUM` is no longer present in WMTS capabilities
            - uses the official HiSup transform path and image statistics from the repo / demo during inference, not the prompt's COCO preprocessing

            **Sources**

            - Official WMTS capabilities: [WMTSCapabilities.xml](https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/MapServer/WMTS/1.0.0/WMTSCapabilities.xml)
            - Official HiSup config: [config-files/crowdai_hrnet48.yaml](https://github.com/SarahwXU/HiSup/blob/main/config-files/crowdai_hrnet48.yaml)
            - Official detector weights URL in code: [hisup/detector.py](https://github.com/SarahwXU/HiSup/blob/main/hisup/detector.py)
            """
        ),
        code_cell(
            """
            import io
            import json
            import math
            import time
            import warnings
            from pathlib import Path

            import cv2
            import geopandas as gpd
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            from PIL import Image
            import rasterio
            from rasterio.transform import from_bounds
            from pyproj import Geod, Transformer
            import requests
            from shapely.affinity import translate
            from shapely.geometry import MultiPolygon, Polygon
            from shapely.ops import unary_union
            import torch
            from tqdm.auto import tqdm


            warnings.filterwarnings("ignore", category=FutureWarning)
            pd.set_option("display.max_columns", 200)
            pd.set_option("display.max_rows", 200)
            torch.set_grad_enabled(False)

            # NumPy compatibility shim for the public HiSup codebase on modern Kaggle images.
            # HiSup's HRNet backbones still reference deprecated aliases such as np.int.
            if not hasattr(np, "int"):
                np.int = int
            if not hasattr(np, "float"):
                np.float = float
            if not hasattr(np, "bool"):
                np.bool = bool
            if not hasattr(np, "int0"):
                np.int0 = np.intp

            CONFIG = {
                "WAYBACK_RELEASE_NUM": 25285,
                "WAYBACK_ZOOM": 19,
                "GSD_M": 0.298,
                "BBOX": {
                    "west": -7.6400,
                    "south": 33.5910,
                    "east": -7.6280,
                    "north": 33.5990,
                },
                "MAX_TILES": 200,
                "CHIP_SIZE": 512,
                "CHIP_OVERLAP": 112,
                "SCORE_THRESHOLD": 0.5,
                "OUTPUT_DIR": Path("/kaggle/working/hisup_aoi"),
                "PRETRAINED_WEIGHTS_URL": "https://github.com/XJKunnn/pretrained_model/releases/download/pretrained_model/crowdai_hrnet48_e100.pth",
                "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
                "COCO_MEAN": [123.675, 116.28, 103.53],
                "COCO_STD": [58.395, 57.12, 57.375],
            }
            CONFIG["OUTPUT_DIR"].mkdir(parents=True, exist_ok=True)

            WMTS_CAPABILITIES_URL = (
                "https://wayback.maptiles.arcgis.com/arcgis/rest/services/"
                "World_Imagery/MapServer/WMTS/1.0.0/WMTSCapabilities.xml"
            )
            WAYBACK_TILE_TEMPLATE = (
                "https://wayback.maptiles.arcgis.com/arcgis/rest/services/"
                "World_Imagery/WMTS/1.0.0/default028mm/MapServer/tile/"
                "{release_num}/{z}/{y}/{x}"
            )
            GEOD = Geod(ellps="WGS84")
            TO_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
            TO_4326 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

            SESSION = requests.Session()
            SESSION.headers.update(
                {
                    "User-Agent": "Kaggle-HiSup-Wayback/1.0",
                    "Referer": "https://www.arcgis.com/",
                    "Accept": "*/*",
                }
            )

            print(json.dumps({k: (str(v) if isinstance(v, Path) else v) for k, v in CONFIG.items()}, indent=2))
            """
        ),
        markdown_cell(
            """
            ## Section 4 — ESRI Wayback tile download

            The notebook resolves the requested release number against the live WMTS capabilities document, downloads all tiles covering the configured WGS84 bounding box, then writes both a stitched PNG and a georeferenced GeoTIFF in `EPSG:3857`.

            **Sources**

            - ArcGIS Wayback WMTS capabilities: [WMTSCapabilities.xml](https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/MapServer/WMTS/1.0.0/WMTSCapabilities.xml)
            - ArcGIS Wayback overview: [Wayback with World Imagery Metadata](https://www.esri.com/arcgis-blog/products/arcgis-living-atlas/imagery/wayback-with-world-imagery-metadata)
            """
        ),
        code_cell(
            """
            def get_text(url: str, *, params: dict | None = None) -> str:
                \"\"\"Fetch a text response with strict timeout and status handling.\"\"\"
                response = SESSION.get(url, params=params, timeout=120)
                response.raise_for_status()
                return response.text


            def get_json(url: str, *, params: dict | None = None) -> dict:
                \"\"\"Fetch a JSON response with strict timeout and status handling.\"\"\"
                response = SESSION.get(url, params=params, timeout=120)
                response.raise_for_status()
                return response.json()


            def parse_wayback_releases() -> pd.DataFrame:
                \"\"\"Parse live Wayback WMTS capabilities into a release table.\"\"\"
                import re
                import xml.etree.ElementTree as ET

                xml_text = get_text(WMTS_CAPABILITIES_URL)
                root = ET.fromstring(xml_text)
                ns = {
                    "wmts": "https://www.opengis.net/wmts/1.0",
                    "ows": "https://www.opengis.net/ows/1.1",
                }

                rows = []
                for layer in root.findall(".//wmts:Layer", ns):
                    identifier = layer.findtext("ows:Identifier", default="", namespaces=ns)
                    title = layer.findtext("ows:Title", default="", namespaces=ns)
                    resource = layer.find("wmts:ResourceURL", ns)
                    if not identifier.startswith("WB_") or resource is None:
                        continue
                    template = resource.attrib.get("template", "")
                    match = re.search(r"/tile/(\\d+)/\\{TileMatrix\\}/", template)
                    if not match:
                        continue
                    release_num = int(match.group(1))
                    date_match = re.search(r"(\\d{4}-\\d{2}-\\d{2})", title)
                    release_date = pd.to_datetime(date_match.group(1)).date() if date_match else None
                    rows.append(
                        {
                            "identifier": identifier,
                            "title": title,
                            "release_num": release_num,
                            "release_date": release_date,
                            "template": template,
                        }
                    )
                return pd.DataFrame(rows).sort_values("release_date").reset_index(drop=True)


            WAYBACK_RELEASES = parse_wayback_releases()
            display(WAYBACK_RELEASES.tail(10))


            def resolve_wayback_release(release_num: int) -> dict:
                \"\"\"Resolve a requested release number against live WMTS capabilities.

                If the configured release number is stale, the function falls back to the
                latest live release and prints the exact release used.
                \"\"\"
                matches = WAYBACK_RELEASES[WAYBACK_RELEASES["release_num"] == int(release_num)]
                if not matches.empty:
                    row = matches.iloc[-1].to_dict()
                    print(
                        f"Using configured Wayback release {row['identifier']} "
                        f"(release_num={row['release_num']}, date={row['release_date']})."
                    )
                    return row

                latest = WAYBACK_RELEASES.iloc[-1].to_dict()
                print(
                    f"Configured WAYBACK_RELEASE_NUM={release_num} was not found in live WMTS capabilities on "
                    f"{pd.Timestamp.utcnow().date()}. Falling back to latest live release "
                    f"{latest['identifier']} (release_num={latest['release_num']}, date={latest['release_date']})."
                )
                return latest


            def lonlat_to_tile(lat: float, lon: float, zoom: int) -> tuple[int, int]:
                \"\"\"Convert WGS84 latitude/longitude to slippy-map tile indices.

                Inputs:
                    lat: Latitude in decimal degrees.
                    lon: Longitude in decimal degrees.
                    zoom: Integer WMTS/slippy zoom level.

                Returns:
                    A tuple `(x, y)` of tile indices at the requested zoom.

                Assumptions:
                    Latitude is clamped to the valid Web Mercator domain.
                \"\"\"
                lat = max(min(float(lat), 85.05112878), -85.05112878)
                lon = float(lon)
                n = 2 ** int(zoom)
                x = int((lon + 180.0) / 360.0 * n)
                lat_rad = math.radians(lat)
                y = int(
                    (1.0 - math.log(math.tan(lat_rad) + (1.0 / math.cos(lat_rad))) / math.pi)
                    / 2.0
                    * n
                )
                x = min(max(x, 0), n - 1)
                y = min(max(y, 0), n - 1)
                return x, y


            def tile_to_lonlat_bounds(x: int, y: int, zoom: int) -> dict:
                \"\"\"Return WGS84 bounds for a slippy-map tile.

                Inputs:
                    x: Tile column.
                    y: Tile row.
                    zoom: Integer zoom level.

                Returns:
                    A dict with `west`, `south`, `east`, and `north` in decimal degrees.
                \"\"\"
                n = 2 ** int(zoom)
                west = x / n * 360.0 - 180.0
                east = (x + 1) / n * 360.0 - 180.0
                north = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n))))
                south = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * (y + 1) / n))))
                return {"west": west, "south": south, "east": east, "north": north}


            def download_wayback_mosaic(config: dict) -> dict:
                \"\"\"Download and stitch an ESRI Wayback RGB mosaic over the configured AOI.

                Inputs:
                    config: Notebook configuration dict containing release number, AOI,
                        zoom, output directory, and tile cap.

                Returns:
                    A dict with mosaic array, PNG path, GeoTIFF path, bounds in EPSG:3857,
                    width, and height.

                Assumptions:
                    The AOI is given in WGS84 and the live WMTS capabilities document is reachable.
                \"\"\"
                release = resolve_wayback_release(config["WAYBACK_RELEASE_NUM"])
                bbox = config["BBOX"]
                zoom = int(config["WAYBACK_ZOOM"])

                west_x, north_y = lonlat_to_tile(bbox["north"], bbox["west"], zoom)
                east_x, south_y = lonlat_to_tile(bbox["south"], bbox["east"], zoom)
                x_min, x_max = sorted((west_x, east_x))
                y_min, y_max = sorted((north_y, south_y))
                tile_count = (x_max - x_min + 1) * (y_max - y_min + 1)
                if tile_count > int(config["MAX_TILES"]):
                    raise ValueError(
                        f"AOI would download {tile_count} tiles at z={zoom}; "
                        "reduce AOI, lower WAYBACK_ZOOM, or raise MAX_TILES."
                    )

                width = (x_max - x_min + 1) * 256
                height = (y_max - y_min + 1) * 256
                canvas = Image.new("RGB", (width, height))

                tile_iter = [(x, y) for y in range(y_min, y_max + 1) for x in range(x_min, x_max + 1)]
                for x, y in tqdm(tile_iter, desc="Downloading Wayback tiles"):
                    tile_url = WAYBACK_TILE_TEMPLATE.format(
                        release_num=release["release_num"],
                        z=zoom,
                        y=y,
                        x=x,
                    )
                    response = SESSION.get(tile_url, timeout=120)
                    response.raise_for_status()
                    tile = Image.open(io.BytesIO(response.content)).convert("RGB")
                    canvas.paste(tile, ((x - x_min) * 256, (y - y_min) * 256))
                    time.sleep(0.15)

                left_lonlat = tile_to_lonlat_bounds(x_min, y_min, zoom)
                right_lonlat = tile_to_lonlat_bounds(x_max, y_max, zoom)
                left, bottom = TO_3857.transform(left_lonlat["west"], right_lonlat["south"])
                right, top = TO_3857.transform(right_lonlat["east"], left_lonlat["north"])
                bounds_3857 = (left, bottom, right, top)

                out_dir = config["OUTPUT_DIR"]
                png_path = out_dir / f"wayback_{release['identifier']}_z{zoom}.png"
                geotiff_path = out_dir / f"wayback_{release['identifier']}_z{zoom}.tif"
                canvas.save(png_path)

                array = np.asarray(canvas)
                transform = from_bounds(left, bottom, right, top, width=array.shape[1], height=array.shape[0])
                with rasterio.open(
                    geotiff_path,
                    "w",
                    driver="GTiff",
                    width=array.shape[1],
                    height=array.shape[0],
                    count=3,
                    dtype=array.dtype,
                    crs="EPSG:3857",
                    transform=transform,
                ) as dst:
                    for band_idx in range(3):
                        dst.write(array[:, :, band_idx], band_idx + 1)

                return {
                    "release": release,
                    "array": array,
                    "png_path": png_path,
                    "geotiff_path": geotiff_path,
                    "bounds_3857": bounds_3857,
                    "width": array.shape[1],
                    "height": array.shape[0],
                    "transform": transform,
                }
            """
        ),
        code_cell(
            """
            mosaic = download_wayback_mosaic(CONFIG)
            print("PNG:", mosaic["png_path"])
            print("GeoTIFF:", mosaic["geotiff_path"])
            print("Size (HxW):", mosaic["height"], mosaic["width"])
            print("Bounds EPSG:3857:", mosaic["bounds_3857"])

            plt.figure(figsize=(12, 12))
            plt.imshow(mosaic["array"])
            plt.title(f"Wayback mosaic | {mosaic['release']['identifier']} | z={CONFIG['WAYBACK_ZOOM']}")
            plt.axis("off")
            plt.show()
            """
        ),
        markdown_cell(
            """
            ## Section 5 — Retrieve image capture metadata

            ArcGIS Wayback metadata must be queried through the release-specific metadata service, not through the stale generic `World_Imagery_Metadata/MapServer/{release_num}` pattern. The notebook derives the correct service name from the resolved Wayback release identifier and queries the first metadata layer that returns a hit for the AOI center point.

            **Sources**

            - ArcGIS Wayback overview: [Wayback with World Imagery Metadata](https://www.esri.com/arcgis-blog/products/arcgis-living-atlas/imagery/wayback-with-world-imagery-metadata)
            - Live metadata service pattern visible from current services, for example: [World_Imagery_Metadata_2026_r03](https://metadata.maptiles.arcgis.com/arcgis/rest/services/World_Imagery_Metadata_2026_r03/MapServer?f=pjson)
            """
        ),
        code_cell(
            """
            def release_identifier_to_metadata_service(identifier: str) -> str:
                \"\"\"Convert a Wayback identifier like `WB_2026_R03` to its metadata service URL.\"\"\"
                import re

                match = re.fullmatch(r"WB_(\\d{4})_R(\\d{2})", identifier)
                if not match:
                    raise ValueError(f"Unexpected Wayback identifier format: {identifier}")
                year, release_idx = match.groups()
                service_name = f"World_Imagery_Metadata_{year}_r{release_idx.lower()}"
                return (
                    "https://metadata.maptiles.arcgis.com/arcgis/rest/services/"
                    f"{service_name}/MapServer"
                )


            def query_wayback_capture_metadata(release_num: int, lon: float, lat: float) -> dict:
                \"\"\"Query ESRI Wayback source metadata for a point location.

                Inputs:
                    release_num: Numeric Wayback release number. If stale, the notebook uses the
                        live resolved release instead.
                    lon: Longitude in decimal degrees.
                    lat: Latitude in decimal degrees.

                Returns:
                    The first matching metadata attribute dict, or an empty dict if no hit is found.

                Assumptions:
                    Metadata is queried from the release-specific metadata service corresponding
                    to the resolved Wayback identifier.
                \"\"\"
                release = resolve_wayback_release(release_num)
                base_url = release_identifier_to_metadata_service(release["identifier"])
                service_info = get_json(f"{base_url}", params={"f": "pjson"})
                layers = service_info.get("layers", [])
                params = {
                    "geometry": json.dumps({"x": float(lon), "y": float(lat), "spatialReference": {"wkid": 4326}}),
                    "geometryType": "esriGeometryPoint",
                    "inSR": 4326,
                    "spatialRel": "esriSpatialRelIntersects",
                    "outFields": "SRC_DATE,SRC_RES,SRC_ACC,NICE_DESC,NICE_NAME",
                    "returnGeometry": "false",
                    "f": "pjson",
                }
                for layer in layers:
                    payload = get_json(f"{base_url}/{layer['id']}/query", params=params)
                    features = payload.get("features", [])
                    if features:
                        attrs = dict(features[0]["attributes"])
                        attrs["_metadata_layer_id"] = layer["id"]
                        attrs["_metadata_layer_name"] = layer["name"]
                        attrs["_release_identifier"] = release["identifier"]
                        attrs["_release_num"] = release["release_num"]
                        return attrs
                return {}


            bbox = CONFIG["BBOX"]
            center_lon = (bbox["west"] + bbox["east"]) / 2.0
            center_lat = (bbox["south"] + bbox["north"]) / 2.0
            capture_metadata = query_wayback_capture_metadata(CONFIG["WAYBACK_RELEASE_NUM"], center_lon, center_lat)
            if capture_metadata:
                print("Resolved release identifier:", capture_metadata["_release_identifier"])
                print("Metadata layer:", capture_metadata["_metadata_layer_name"])
                print("SRC_DATE:", capture_metadata.get("SRC_DATE"))
                print("SRC_RES:", capture_metadata.get("SRC_RES"))
                print("SRC_ACC:", capture_metadata.get("SRC_ACC"))
                print("NICE_DESC:", capture_metadata.get("NICE_DESC"))
                print("NICE_NAME:", capture_metadata.get("NICE_NAME"))
                SRC_DATE = capture_metadata.get("SRC_DATE")
            else:
                print("No metadata feature was found at the AOI center point.")
                SRC_DATE = None
            """
        ),
        markdown_cell(
            """
            ## Section 6 — HiSup model loading

            The public quickstart calls `get_pretrained_model(cfg, dataset, device, pretrained=True)`. Internally, that function builds `BuildingDetector(cfg, test=True)` and loads the CrowdAI checkpoint. To keep the notebook reproducible on Kaggle, this notebook caches the official CrowdAI checkpoint in the output directory and then reproduces the same public loading logic manually.

            **Sources**

            - Official detector loader: [hisup/detector.py](https://github.com/SarahwXU/HiSup/blob/main/hisup/detector.py)
            - Official config file: [config-files/crowdai_hrnet48.yaml](https://github.com/SarahwXU/HiSup/blob/main/config-files/crowdai_hrnet48.yaml)
            """
        ),
        code_cell(
            """
            def download_file(url: str, destination: Path) -> Path:
                \"\"\"Download a file with streaming I/O and strict HTTP error handling.\"\"\"
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    print("Using cached file:", destination)
                    return destination

                with SESSION.get(url, stream=True, timeout=120) as response:
                    response.raise_for_status()
                    total = int(response.headers.get("content-length", 0))
                    with destination.open("wb") as f, tqdm(
                        total=total,
                        unit="B",
                        unit_scale=True,
                        desc=f"Downloading {destination.name}",
                    ) as pbar:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                return destination


            def load_hisup_model(config: dict) -> torch.nn.Module:
                \"\"\"Load the public HiSup CrowdAI pretrained model.

                Inputs:
                    config: Notebook configuration dict with output directory, device,
                        and the official pretrained checkpoint URL.

                Returns:
                    A `torch.nn.Module` in `eval()` mode on the configured device.

                Assumptions:
                    The official HiSup repo has been cloned into `HISUP_REPO_DIR` and
                    `config-files/crowdai_hrnet48.yaml` exists there.
                \"\"\"
                weights_path = download_file(
                    config["PRETRAINED_WEIGHTS_URL"],
                    config["OUTPUT_DIR"] / Path(config["PRETRAINED_WEIGHTS_URL"]).name,
                )

                try:
                    from hisup.config import cfg as hisup_cfg
                except ImportError as exc:
                    print(
                        "ImportError while importing 'from hisup.config import cfg'. "
                        f"Expected module under: {HISUP_REPO_DIR / 'hisup' / 'config'}. Exact error: {exc}"
                    )
                    raise

                try:
                    from hisup.detector import BuildingDetector
                except ImportError as exc:
                    print(
                        "ImportError while importing 'from hisup.detector import BuildingDetector'. "
                        f"Expected module file: {HISUP_REPO_DIR / 'hisup' / 'detector.py'}. Exact error: {exc}"
                    )
                    raise

                cfg_path = HISUP_REPO_DIR / "config-files" / "crowdai_hrnet48.yaml"
                if not cfg_path.exists():
                    raise FileNotFoundError(
                        f"Expected HiSup config file was not found: {cfg_path}"
                    )

                cfg = hisup_cfg.clone()
                cfg.defrost()
                cfg.merge_from_file(str(cfg_path))
                cfg.MODEL.DEVICE = config["DEVICE"]

                model = BuildingDetector(cfg, test=True)

                # weights_only=False is required here because the official checkpoint is a
                # full training checkpoint dict containing a nested 'model' state dict.
                checkpoint = torch.load(
                    weights_path,
                    map_location=config["DEVICE"],
                    weights_only=False,
                )
                state_dict = checkpoint["model"]
                if all(key.startswith("module.") for key in state_dict.keys()):
                    state_dict = {key[7:]: value for key, value in state_dict.items()}
                model.load_state_dict(state_dict, strict=True)
                model = model.to(config["DEVICE"]).eval()

                config["_HISUP_CFG"] = cfg
                config["_HISUP_WEIGHTS_PATH"] = weights_path
                return model


            model = load_hisup_model(CONFIG)
            HISUP_CFG = CONFIG["_HISUP_CFG"]
            print("Loaded HiSup weights from:", CONFIG["_HISUP_WEIGHTS_PATH"])
            print("Model device:", next(model.parameters()).device)
            """
        ),
        markdown_cell(
            """
            ## Sections 7–9 — Official preprocessing, tiled inference, polygon merge, and georeferencing

            The public HiSup demo does not manually post-process mask logits and junction maps outside the detector. Instead, `BuildingDetector.forward_test()` already returns:

            - `polys_pred`
            - `mask_pred`
            - `scores`
            - `juncs_pred`

            This notebook therefore uses the public forward path per chip, then merges overlapping chip detections and georeferences them to `EPSG:4326`.

            **Sources**

            - Official demo: [scripts/demo.py](https://github.com/SarahwXU/HiSup/blob/main/scripts/demo.py)
            - Official transform builder: [hisup/dataset/build.py](https://github.com/SarahwXU/HiSup/blob/main/hisup/dataset/build.py)
            - Official polygon generation path: [hisup/detector.py](https://github.com/SarahwXU/HiSup/blob/main/hisup/detector.py)
            """
        ),
        code_cell(
            """
            def set_hisup_image_statistics(image_array: np.ndarray, cfg) -> None:
                \"\"\"Match the public demo by computing per-image channel mean/std before inference.\"\"\"
                cfg.defrost()
                pixel_mean = [float(np.mean(image_array[:, :, channel])) for channel in range(3)]
                pixel_std = [float(np.std(image_array[:, :, channel])) for channel in range(3)]
                cfg.DATASETS.IMAGE.PIXEL_MEAN = pixel_mean
                cfg.DATASETS.IMAGE.PIXEL_STD = pixel_std


            def preprocess_image(image_array: np.ndarray, config: dict) -> torch.Tensor:
                \"\"\"Preprocess one chip with the official HiSup transform path.

                Inputs:
                    image_array: RGB chip as a NumPy array in HxWx3 format.
                    config: Notebook configuration dict containing `_HISUP_CFG`.

                Returns:
                    A normalized `torch.Tensor` in CxHxW format.

                Assumptions:
                    For scientific correctness this function uses HiSup's public
                    `build_transform(cfg)` path rather than the prompt's COCO/no-resize variant.
                \"\"\"
                try:
                    from hisup.dataset.build import build_transform
                except ImportError as exc:
                    print(
                        "ImportError while importing 'from hisup.dataset.build import build_transform'. "
                        f"Expected module file: {HISUP_REPO_DIR / 'hisup' / 'dataset' / 'build.py'}. Exact error: {exc}"
                    )
                    raise

                cfg = config["_HISUP_CFG"]
                transform = build_transform(cfg)
                tensor = transform(image_array.astype(float))
                return tensor


            def tile_image(image_array: np.ndarray, chip_size: int, overlap: int) -> list[dict]:
                \"\"\"Split an RGB mosaic into overlapping chips with zero padding at edges.

                Inputs:
                    image_array: Mosaic array in HxWx3 format.
                    chip_size: Chip width/height in pixels.
                    overlap: Overlap in pixels between adjacent chips.

                Returns:
                    A list of dicts with chip array, x/y offsets, and original unpadded chip size.

                Assumptions:
                    The overlap is smaller than the chip size.
                \"\"\"
                if overlap >= chip_size:
                    raise ValueError("CHIP_OVERLAP must be smaller than CHIP_SIZE.")

                stride = chip_size - overlap
                height, width = image_array.shape[:2]
                chips = []
                y_positions = list(range(0, max(height - chip_size, 0) + 1, stride))
                x_positions = list(range(0, max(width - chip_size, 0) + 1, stride))
                if not y_positions or y_positions[-1] != max(height - chip_size, 0):
                    y_positions.append(max(height - chip_size, 0))
                if not x_positions or x_positions[-1] != max(width - chip_size, 0):
                    x_positions.append(max(width - chip_size, 0))

                for y_offset in y_positions:
                    for x_offset in x_positions:
                        chip = image_array[y_offset:y_offset + chip_size, x_offset:x_offset + chip_size, :]
                        orig_h, orig_w = chip.shape[:2]
                        padded = np.zeros((chip_size, chip_size, 3), dtype=image_array.dtype)
                        padded[:orig_h, :orig_w, :] = chip
                        chips.append(
                            {
                                "array": padded,
                                "x_offset": int(x_offset),
                                "y_offset": int(y_offset),
                                "w": int(orig_w),
                                "h": int(orig_h),
                            }
                        )
                return chips


            @torch.inference_mode()
            def run_hisup_inference(model: torch.nn.Module, chips: list[dict], config: dict) -> list[dict]:
                \"\"\"Run official HiSup forward inference on tiled chips.

                Inputs:
                    model: Loaded HiSup `BuildingDetector`.
                    chips: Output of `tile_image`.
                    config: Notebook configuration dict containing `_HISUP_CFG`, score threshold,
                        and device.

                Returns:
                    A list of chip-level polygon dicts containing polygons, scores, junctions,
                    and chip offsets.

                Assumptions:
                    Polygon generation is delegated to `BuildingDetector.forward_test()` exactly
                    as in the public repo.
                \"\"\"
                try:
                    from hisup.utils.comm import to_single_device
                except ImportError as exc:
                    print(
                        "ImportError while importing 'from hisup.utils.comm import to_single_device'. "
                        f"Expected module file: {HISUP_REPO_DIR / 'hisup' / 'utils' / 'comm.py'}. Exact error: {exc}"
                    )
                    raise

                cfg = config["_HISUP_CFG"]
                results = []
                for chip in tqdm(chips, desc="Running HiSup on chips"):
                    chip_img = chip["array"][: chip["h"], : chip["w"], :]
                    model.origin_height = chip["h"]
                    model.origin_width = chip["w"]
                    tensor = preprocess_image(chip_img, config).unsqueeze(0).to(config["DEVICE"])
                    meta = {"height": chip["h"], "width": chip["w"]}
                    output, _ = model(tensor, [meta])
                    output = to_single_device(output, "cpu")

                    polygons = output["polys_pred"][0] if output["polys_pred"] else []
                    scores = output["scores"][0] if output["scores"] else []
                    junctions = output["juncs_pred"][0] if output["juncs_pred"] else np.empty((0, 2), dtype=np.float32)
                    mask_pred = output["mask_pred"][0] if output["mask_pred"] else None

                    kept_polygons = []
                    kept_scores = []
                    for polygon, score in zip(polygons, scores):
                        polygon = np.asarray(polygon, dtype=np.float32)
                        if polygon.ndim != 2 or polygon.shape[0] < 3:
                            continue
                        polygon[:, 0] = np.clip(polygon[:, 0], 0, chip["w"] - 1)
                        polygon[:, 1] = np.clip(polygon[:, 1], 0, chip["h"] - 1)
                        if float(score) >= float(config["SCORE_THRESHOLD"]):
                            kept_polygons.append(polygon)
                            kept_scores.append(float(score))

                    results.append(
                        {
                            "polygons": kept_polygons,
                            "scores": kept_scores,
                            "junctions": np.asarray(junctions, dtype=np.float32),
                            "mask_pred": mask_pred,
                            "chip_x_offset": chip["x_offset"],
                            "chip_y_offset": chip["y_offset"],
                            "chip_w": chip["w"],
                            "chip_h": chip["h"],
                        }
                    )
                return results


            def polygon_iou(poly_a: Polygon, poly_b: Polygon) -> float:
                \"\"\"Compute polygon IoU in image-pixel space.\"\"\"
                inter = poly_a.intersection(poly_b).area
                union = poly_a.union(poly_b).area
                if union <= 0:
                    return 0.0
                return float(inter / union)


            def pixel_polygon_to_3857_geometry(pixel_polygon: Polygon, mosaic: dict) -> Polygon:
                \"\"\"Convert a pixel-space polygon to projected EPSG:3857 coordinates using the mosaic affine transform.\"\"\"
                transform = mosaic["transform"]

                def map_xy(x, y, z=None):
                    return transform * (x, y)

                coords = [map_xy(x, y) for x, y in np.asarray(pixel_polygon.exterior.coords)]
                holes = []
                for ring in pixel_polygon.interiors:
                    holes.append([map_xy(x, y) for x, y in np.asarray(ring.coords)])
                return Polygon(coords, holes)


            def utm_epsg_for_lonlat(lon: float, lat: float) -> int:
                \"\"\"Estimate a UTM EPSG code from longitude and latitude.\"\"\"
                zone = int((lon + 180.0) // 6.0) + 1
                if lat >= 0:
                    return 32600 + zone
                return 32700 + zone


            def merge_chip_detections(chip_results: list[dict], mosaic: dict, config: dict) -> gpd.GeoDataFrame:
                \"\"\"Merge tiled HiSup polygons, deduplicate overlaps, and georeference results.

                Inputs:
                    chip_results: Output of `run_hisup_inference`.
                    mosaic: Output dict from `download_wayback_mosaic`.
                    config: Notebook configuration dict.

                Returns:
                    A `GeoDataFrame` in `EPSG:4326` with geometry, area/perimeter, centroid,
                    pixel-centroid, and detection score columns.

                Assumptions:
                    Duplicate detections arise mainly from overlapping chips and are removed using
                    an IoU threshold of 0.5 before unary-union cleanup inside duplicate clusters.
                \"\"\"
                candidates = []
                for chip_result in chip_results:
                    for polygon, score in zip(chip_result["polygons"], chip_result["scores"]):
                        geom_px = Polygon(polygon)
                        if geom_px.is_empty:
                            continue
                        if not geom_px.is_valid:
                            geom_px = geom_px.buffer(0)
                        if geom_px.is_empty or geom_px.area <= 1.0:
                            continue
                        geom_px = translate(
                            geom_px,
                            xoff=chip_result["chip_x_offset"],
                            yoff=chip_result["chip_y_offset"],
                        )
                        candidates.append({"geom_px": geom_px, "score": float(score)})

                if not candidates:
                    return gpd.GeoDataFrame(
                        columns=[
                            "geometry",
                            "area_m2",
                            "perimeter_m",
                            "centroid_lon",
                            "centroid_lat",
                            "pixel_centroid_x",
                            "pixel_centroid_y",
                            "score",
                            "num_vertices",
                        ],
                        geometry=[],
                        crs="EPSG:4326",
                    )

                candidates = sorted(candidates, key=lambda item: item["score"], reverse=True)
                clusters: list[list[dict]] = []
                for candidate in candidates:
                    assigned = False
                    for cluster in clusters:
                        if any(polygon_iou(candidate["geom_px"], item["geom_px"]) > 0.5 for item in cluster):
                            cluster.append(candidate)
                            assigned = True
                            break
                    if not assigned:
                        clusters.append([candidate])

                merged_records = []
                for cluster in clusters:
                    cluster_score = max(item["score"] for item in cluster)
                    merged_geom = unary_union([item["geom_px"] for item in cluster])
                    if merged_geom.is_empty:
                        continue
                    if isinstance(merged_geom, Polygon):
                        parts = [merged_geom]
                    elif isinstance(merged_geom, MultiPolygon):
                        parts = list(merged_geom.geoms)
                    else:
                        continue
                    for part in parts:
                        if not part.is_valid:
                            part = part.buffer(0)
                        if part.is_empty or part.area <= 1.0:
                            continue
                        merged_records.append({"geom_px": part, "score": cluster_score})

                rows = []
                for record in merged_records:
                    geom_3857 = pixel_polygon_to_3857_geometry(record["geom_px"], mosaic)
                    geom_4326 = gpd.GeoSeries([geom_3857], crs="EPSG:3857").to_crs("EPSG:4326").iloc[0]
                    centroid_4326 = geom_4326.centroid
                    centroid_px = record["geom_px"].centroid
                    rows.append(
                        {
                            "geometry": geom_4326,
                            "geometry_3857": geom_3857,
                            "score": record["score"],
                            "centroid_lon": float(centroid_4326.x),
                            "centroid_lat": float(centroid_4326.y),
                            "pixel_centroid_x": float(centroid_px.x),
                            "pixel_centroid_y": float(centroid_px.y),
                            "num_vertices": len(record["geom_px"].exterior.coords) - 1,
                        }
                    )

                gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
                if gdf.empty:
                    return gdf

                lon0 = float(gdf.iloc[0]["centroid_lon"])
                lat0 = float(gdf.iloc[0]["centroid_lat"])
                utm_epsg = utm_epsg_for_lonlat(lon0, lat0)
                gdf_metric = gdf.set_geometry("geometry_3857").copy()
                gdf_metric = gdf_metric.set_crs("EPSG:3857").to_crs(f"EPSG:{utm_epsg}")
                gdf["area_m2"] = gdf_metric.geometry.area.astype(float)
                gdf["perimeter_m"] = gdf_metric.geometry.length.astype(float)
                gdf = gdf.drop(columns=["geometry_3857"])
                return gdf
            """
        ),
        code_cell(
            """
            set_hisup_image_statistics(mosaic["array"], HISUP_CFG)
            chips = tile_image(mosaic["array"], CONFIG["CHIP_SIZE"], CONFIG["CHIP_OVERLAP"])
            print("Total chips:", len(chips))

            chip_results = run_hisup_inference(model, chips, CONFIG)
            buildings_gdf = merge_chip_detections(chip_results, mosaic, CONFIG)

            print("Detected polygons:", len(buildings_gdf))
            if not buildings_gdf.empty:
                display(buildings_gdf.head(10))
            """
        ),
        markdown_cell(
            """
            ## Section 10 — Visualization

            The left panel shows the full Wayback mosaic with georeferenced HiSup polygons projected back into pixel space. The right panel shows a 400×400 pixel crop around the largest detected building and highlights polygon vertices in red.

            **Sources**

            - Official visualization helpers: [hisup/utils/visualizer.py](https://github.com/SarahwXU/HiSup/blob/main/hisup/utils/visualizer.py)
            - Official polygon output path: [hisup/detector.py](https://github.com/SarahwXU/HiSup/blob/main/hisup/detector.py)
            """
        ),
        code_cell(
            """
            def geometry_4326_to_pixel_polygon(geometry_4326: Polygon, mosaic: dict) -> Polygon:
                \"\"\"Convert a georeferenced EPSG:4326 polygon back to mosaic pixel coordinates for plotting.\"\"\"
                projected = gpd.GeoSeries([geometry_4326], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]
                inverse = ~mosaic["transform"]

                def map_xy(x, y, z=None):
                    return inverse * (x, y)

                coords = [map_xy(x, y) for x, y in np.asarray(projected.exterior.coords)]
                holes = []
                for ring in projected.interiors:
                    holes.append([map_xy(x, y) for x, y in np.asarray(ring.coords)])
                return Polygon(coords, holes)


            def clip_crop_bounds(cx: float, cy: float, width: int, height: int, crop_size: int = 400) -> tuple[int, int, int, int]:
                \"\"\"Compute a bounded square crop window around a pixel-space center.\"\"\"
                half = crop_size // 2
                x1 = int(max(min(cx - half, width - crop_size), 0))
                y1 = int(max(min(cy - half, height - crop_size), 0))
                x2 = int(min(x1 + crop_size, width))
                y2 = int(min(y1 + crop_size, height))
                return x1, y1, x2, y2


            def summarize_polygons(gdf: gpd.GeoDataFrame) -> dict:
                \"\"\"Compute notebook summary statistics from the polygon GeoDataFrame.\"\"\"
                if gdf.empty:
                    return {
                        "total_buildings": 0,
                        "mean_area_m2": 0.0,
                        "mean_vertices": 0.0,
                        "pct_ge_4_vertices": 0.0,
                    }
                return {
                    "total_buildings": int(len(gdf)),
                    "mean_area_m2": float(gdf["area_m2"].mean()),
                    "mean_vertices": float(gdf["num_vertices"].mean()),
                    "pct_ge_4_vertices": float((gdf["num_vertices"] >= 4).mean() * 100.0),
                }


            summary_stats = summarize_polygons(buildings_gdf)
            print("Total buildings detected:", summary_stats["total_buildings"])
            print("Mean polygon area (m²):", round(summary_stats["mean_area_m2"], 3))
            print("Mean number of vertices:", round(summary_stats["mean_vertices"], 3))
            print("Percentage with >= 4 vertices:", round(summary_stats["pct_ge_4_vertices"], 2), "%")

            fig, axes = plt.subplots(1, 2, figsize=(18, 9))
            axes[0].imshow(mosaic["array"])
            axes[0].set_title("Wayback mosaic with HiSup polygons")
            axes[0].axis("off")

            if not buildings_gdf.empty:
                pixel_polygons = [geometry_4326_to_pixel_polygon(geom, mosaic) for geom in buildings_gdf.geometry]
                for poly in pixel_polygons:
                    coords = np.asarray(poly.exterior.coords)
                    axes[0].fill(coords[:, 0], coords[:, 1], color="lime", alpha=0.25)
                    axes[0].plot(coords[:, 0], coords[:, 1], color="yellow", linewidth=1.5)

                largest_idx = buildings_gdf["area_m2"].astype(float).idxmax()
                highlight_poly = pixel_polygons[buildings_gdf.index.get_loc(largest_idx)]
                cx = float(highlight_poly.centroid.x)
                cy = float(highlight_poly.centroid.y)
                x1, y1, x2, y2 = clip_crop_bounds(cx, cy, mosaic["width"], mosaic["height"], crop_size=400)
                crop = mosaic["array"][y1:y2, x1:x2, :]

                axes[1].imshow(crop)
                axes[1].set_title("400×400 px crop with polygon corners")
                axes[1].axis("off")
                for poly in pixel_polygons:
                    coords = np.asarray(poly.exterior.coords)
                    if coords[:, 0].max() < x1 or coords[:, 0].min() > x2 or coords[:, 1].max() < y1 or coords[:, 1].min() > y2:
                        continue
                    local = coords.copy()
                    local[:, 0] -= x1
                    local[:, 1] -= y1
                    axes[1].plot(local[:, 0], local[:, 1], color="yellow", linewidth=1.3)
                    axes[1].scatter(local[:-1, 0], local[:-1, 1], c="red", s=12)
            else:
                axes[1].imshow(mosaic["array"][:400, :400, :])
                axes[1].set_title("No polygons detected in current AOI")
                axes[1].axis("off")

            plt.tight_layout()
            plt.show()
            """
        ),
        markdown_cell(
            """
            ## Section 11 — Export results

            The notebook exports the final georeferenced building polygons as:

            - `buildings.geojson`
            - `buildings.gpkg`
            - `summary.csv`

            `summary.csv` carries every tabular column from the exported `GeoDataFrame`, including the per-polygon HiSup confidence score.

            **Sources**

            - Official HiSup repo: [SarahwXU/HiSup](https://github.com/SarahwXU/HiSup)
            """
        ),
        code_cell(
            """
            def export_results(gdf: gpd.GeoDataFrame, output_dir: Path) -> dict:
                \"\"\"Export polygon outputs as GeoJSON, GeoPackage, and summary CSV.\"\"\"
                output_dir.mkdir(parents=True, exist_ok=True)
                geojson_path = output_dir / "buildings.geojson"
                gpkg_path = output_dir / "buildings.gpkg"
                csv_path = output_dir / "summary.csv"

                export_gdf = gdf.copy()
                if export_gdf.empty:
                    export_gdf = gpd.GeoDataFrame(
                        export_gdf,
                        geometry="geometry",
                        crs="EPSG:4326",
                    )

                export_gdf.to_file(geojson_path, driver="GeoJSON")
                export_gdf.to_file(gpkg_path, driver="GPKG")

                summary_df = pd.DataFrame(export_gdf.drop(columns="geometry"))
                summary_df["geometry_wkt"] = export_gdf.geometry.to_wkt()
                summary_df.to_csv(csv_path, index=False)

                return {
                    "geojson_path": geojson_path,
                    "gpkg_path": gpkg_path,
                    "csv_path": csv_path,
                }


            export_paths = export_results(buildings_gdf, CONFIG["OUTPUT_DIR"])
            print("GeoJSON:", export_paths["geojson_path"])
            print("GeoPackage:", export_paths["gpkg_path"])
            print("CSV:", export_paths["csv_path"])
            """
        ),
        markdown_cell(
            """
            ## Section 12 — Interpretation and limits

            HiSup was trained by the authors on **AICrowd / CrowdAI building extraction imagery**, which is also very-high-resolution RGB imagery around the same order of magnitude as ESRI Wayback urban scenes. That makes zero-shot transfer to Wayback more plausible than transfer from coarse-resolution Sentinel products, but it is still a dataset shift.

            Known limitations for AOI:

            - touching or attached buildings can be merged by the mask-to-polygon stage
            - dense medina-style blocks can produce under-segmentation because adjacent roofs violate clean boundary separation
            - polygon quality should ideally be validated against local reference footprints, not only by visual inspection

            **PoLiS** is a boundary-distance metric, so lower is better. In practical validation, a low PoLiS together with high IoU means both the building extent and the corner geometry are well matched.

            These polygons can replace a YOLO building-detection phase in a later **shadow-based height-estimation** workflow:

            1. extract polygons with HiSup
            2. compute polygon orientation / roof footprint
            3. associate shadows to polygon edges
            4. estimate building height from shadow geometry and solar metadata

            To validate against OSM building footprints, use per-building IoU after reprojecting OSM footprints to the same CRS, spatially matching candidate polygons, and reporting IoU statistics per building and over the AOI.

            **Sources**

            - HiSup paper: [ISPRS JPRS 2023](https://doi.org/10.1016/j.isprsjprs.2023.03.006)
            - Official HiSup repo: [SarahwXU/HiSup](https://github.com/SarahwXU/HiSup)
            - Official PoLiS implementation: [hisup/utils/metrics/polis.py](https://github.com/SarahwXU/HiSup/blob/main/hisup/utils/metrics/polis.py)
            """
        ),
        code_cell(
            """
            print("✅ HiSup inference complete.")
            print("Output files:")
            print("-", mosaic["png_path"])
            print("-", mosaic["geotiff_path"])
            print("-", export_paths["geojson_path"])
            print("-", export_paths["gpkg_path"])
            print("-", export_paths["csv_path"])
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
