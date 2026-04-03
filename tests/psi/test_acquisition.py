from __future__ import annotations

import base64
from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
import time

import logging
import pytest
from pystac import Asset, Item

from aoi_psi.acquisition import (
    S3ObjectInfo,
    _download_transport_for_scene,
    _item_direction,
    _item_polarization,
    _item_processing_level,
    _item_product_type,
    _item_relative_orbit,
    _normalize_s3_prefix,
    _product_uuid_from_href,
    _select_stack_scenes,
    _scene_s3_path,
    _scene_s3_path_from_item,
    _authorized_session,
    _resolve_download_token,
    _token_seconds_to_expiry,
    _download_s3_object_with_resume,
    _download_scene_via_s3,
    _stream_download,
    _scene_href,
    download_stack_scenes,
    ensure_download_auth,
)
from aoi_psi.config import OrbitStackConfig, load_config
from aoi_psi.manifests import SlcScene, StackManifest, read_stack_manifest, stack_manifest_path, write_stack_manifest
from aoi_psi.run_context import RunContext


def _jwt_with_expiry(seconds_from_now: int) -> str:
    header = base64.urlsafe_b64encode(json.dumps({"alg": "none", "typ": "JWT"}).encode("utf-8")).decode("ascii").rstrip("=")
    payload = base64.urlsafe_b64encode(
        json.dumps({"exp": int(time.time()) + seconds_from_now}).encode("utf-8")
    ).decode("ascii").rstrip("=")
    return f"{header}.{payload}."


def test_scene_href_prefers_product_https_asset() -> None:
    item = Item(
        id="S1A_TEST",
        geometry={"type": "Point", "coordinates": [0.0, 0.0]},
        bbox=[0.0, 0.0, 0.0, 0.0],
        datetime=datetime(2024, 1, 1, tzinfo=timezone.utc),
        properties={
            "sat:orbit_state": "ascending",
            "sat:relative_orbit": 154,
            "sar:instrument_mode": "IW",
            "sar:polarizations": ["VV", "VH"],
            "product:type": "IW_SLC__1S",
            "processing:level": "L1",
        },
    )
    item.add_asset(
        "product",
        Asset(
            href="https://download.example.invalid/product.zip",
            extra_fields={"alternate": {"https": {"href": "https://download.example.invalid/product.zip"}}},
        ),
    )
    item.add_asset("safe_manifest", Asset(href="s3://example.invalid/manifest.safe"))

    href, asset_name = _scene_href(item, ("product", "safe_manifest"))

    assert asset_name == "product"
    assert href == "https://download.example.invalid/product.zip"
    assert _item_direction(item) == "ascending"
    assert _item_relative_orbit(item) == 154
    assert _item_polarization(item) == "VH+VV"
    assert _item_product_type(item) == "IW_SLC__1S"
    assert _item_processing_level(item) == "L1"


def test_product_uuid_is_parsed_from_odata_href() -> None:
    href = "https://download.dataspace.copernicus.eu/odata/v1/Products(57425c35-b04c-5073-a149-496049b701e3)/$value"
    assert _product_uuid_from_href(href) == "57425c35-b04c-5073-a149-496049b701e3"


def test_download_auth_is_required(monkeypatch) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "aoi_psi_slc.yaml")
    monkeypatch.delenv(config.acquisition.auth.bearer_token_env, raising=False)
    monkeypatch.delenv("ACCESS_TOKEN", raising=False)
    monkeypatch.delenv(config.acquisition.s3.access_key_env, raising=False)
    monkeypatch.delenv(config.acquisition.s3.secret_key_env, raising=False)
    with pytest.raises(RuntimeError, match="CDSE download authentication is required"):
        ensure_download_auth(config)


def test_download_auth_accepts_s3_credentials(monkeypatch) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "aoi_psi_slc.yaml")
    monkeypatch.delenv(config.acquisition.auth.bearer_token_env, raising=False)
    monkeypatch.delenv("ACCESS_TOKEN", raising=False)
    monkeypatch.setenv(config.acquisition.s3.access_key_env, "access-key")
    monkeypatch.setenv(config.acquisition.s3.secret_key_env, "secret-key")
    ensure_download_auth(config)


def test_resolve_download_token_falls_back_to_access_token(monkeypatch) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "aoi_psi_slc.yaml")
    monkeypatch.delenv(config.acquisition.auth.bearer_token_env, raising=False)
    monkeypatch.setenv("ACCESS_TOKEN", "token-from-access")
    token, env_name = _resolve_download_token(config)
    assert token == "token-from-access"
    assert env_name == "ACCESS_TOKEN"


def test_normalize_s3_prefix_strips_bucket_prefix() -> None:
    assert _normalize_s3_prefix("/eodata/Sentinel-1/SAR/SLC/2019/01/10/SCENE.SAFE") == "Sentinel-1/SAR/SLC/2019/01/10/SCENE.SAFE"


def test_scene_s3_path_uses_product_lookup(monkeypatch) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "aoi_psi_slc.yaml")
    scene = SlcScene(
        scene_id="SCENE_5",
        product_name="SCENE_5",
        acquisition_start="2024-01-01T00:00:00Z",
        acquisition_stop="2024-01-01T00:00:10Z",
        acquisition_date="2024-01-01",
        direction="ascending",
        relative_orbit=147,
        polarization="VV+VH",
        swath_mode="IW",
        product_type="IW_SLC__1S",
        processing_level="L1",
        platform="Sentinel-1A",
        asset_name="product",
        href="https://download.example.invalid/odata/v1/Products(abc-123)/$value",
        product_uuid="abc-123",
    )

    import aoi_psi.acquisition as acquisition_module

    monkeypatch.setattr(acquisition_module, "_stac_scene_s3_paths", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        acquisition_module,
        "_odata_json",
        lambda url, timeout_seconds: {"Id": "abc-123", "Name": "SCENE_5.SAFE", "S3Path": "/eodata/Sentinel-1/SAR/SLC/2024/01/01/SCENE_5.SAFE"},
    )

    assert _scene_s3_path(config, scene) == "/eodata/Sentinel-1/SAR/SLC/2024/01/01/SCENE_5.SAFE"


def test_scene_s3_path_prefers_stac_asset_lookup(monkeypatch) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "aoi_psi_slc.yaml")
    scene = SlcScene(
        scene_id="SCENE_STAC",
        product_name="SCENE_STAC",
        acquisition_start="2024-01-01T00:00:00Z",
        acquisition_stop="2024-01-01T00:00:10Z",
        acquisition_date="2024-01-01",
        direction="ascending",
        relative_orbit=147,
        polarization="VV+VH",
        swath_mode="IW",
        product_type="IW_SLC__1S",
        processing_level="L1",
        platform="Sentinel-1A",
        asset_name="product",
        href="https://download.example.invalid/odata/v1/Products(stac-123)/$value",
        product_uuid="stac-123",
    )

    import aoi_psi.acquisition as acquisition_module

    monkeypatch.setattr(
        acquisition_module,
        "_stac_scene_s3_paths",
        lambda *_args, **_kwargs: {"SCENE_STAC": "/eodata/Sentinel-1/SAR/SLC/2024/01/01/SCENE_STAC.SAFE"},
    )
    monkeypatch.setattr(
        acquisition_module,
        "_odata_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("OData lookup should not run")),
    )

    assert _scene_s3_path(config, scene) == "/eodata/Sentinel-1/SAR/SLC/2024/01/01/SCENE_STAC.SAFE"


def test_scene_s3_path_from_item_prefers_s3_safe_manifest_asset() -> None:
    item = Item(
        id="S1A_TEST_S3",
        geometry={"type": "Point", "coordinates": [0.0, 0.0]},
        bbox=[0.0, 0.0, 0.0, 0.0],
        datetime=datetime(2024, 1, 1, tzinfo=timezone.utc),
        properties={},
    )
    item.add_asset("product", Asset(href="https://download.example.invalid/product.zip"))
    item.add_asset(
        "safe_manifest",
        Asset(href="s3://eodata/Sentinel-1/SAR/SLC/2024/01/01/SCENE_5.SAFE/manifest.safe"),
    )

    assert (
        _scene_s3_path_from_item(item, ("product", "safe_manifest"))
        == "/eodata/Sentinel-1/SAR/SLC/2024/01/01/SCENE_5.SAFE"
    )


def test_select_stack_scenes_uses_master_centered_window() -> None:
    scenes = [
        SlcScene(
            scene_id=f"SCENE_{index}",
            product_name=f"SCENE_{index}",
            acquisition_start=f"2024-01-0{index}T00:00:00Z",
            acquisition_stop=f"2024-01-0{index}T00:00:10Z",
            acquisition_date=f"2024-01-0{index}",
            direction="ascending",
            relative_orbit=147,
            polarization="VV",
            swath_mode="IW",
            product_type="IW_SLC__1S",
            processing_level="L1",
            platform="Sentinel-1A",
            asset_name="product",
            href=f"https://download.example.invalid/{index}.zip",
        )
        for index in range(1, 8)
    ]
    stack = OrbitStackConfig(
        id="asc_rel147_vv",
        direction="ascending",
        relative_orbit=147,
        polarization="VV",
        master_date=date(2024, 1, 4),
        min_scenes=5,
        scene_limit=5,
    )

    selected = _select_stack_scenes(scenes, stack)

    assert [scene.acquisition_date for scene in selected] == [
        "2024-01-02",
        "2024-01-03",
        "2024-01-04",
        "2024-01-05",
        "2024-01-06",
    ]


def test_select_stack_scenes_uses_earliest_window_without_master_date() -> None:
    scenes = [
        SlcScene(
            scene_id=f"SCENE_{index}",
            product_name=f"SCENE_{index}",
            acquisition_start=f"2024-01-{index:02d}T00:00:00Z",
            acquisition_stop=f"2024-01-{index:02d}T00:00:10Z",
            acquisition_date=f"2024-01-{index:02d}",
            direction="ascending",
            relative_orbit=147,
            polarization="VV",
            swath_mode="IW",
            product_type="IW_SLC__1S",
            processing_level="L1",
            platform="Sentinel-1A",
            asset_name="product",
            href=f"https://download.example.invalid/{index}.zip",
        )
        for index in range(1, 8)
    ]
    stack = OrbitStackConfig(
        id="asc_rel147_vv",
        direction="ascending",
        relative_orbit=147,
        polarization="VV",
        master_date=None,
        min_scenes=5,
        scene_limit=5,
    )

    selected = _select_stack_scenes(scenes, stack)

    assert [scene.acquisition_date for scene in selected] == [
        "2024-01-01",
        "2024-01-02",
        "2024-01-03",
        "2024-01-04",
        "2024-01-05",
    ]


def test_token_seconds_to_expiry_detects_expired_token() -> None:
    assert _token_seconds_to_expiry(_jwt_with_expiry(-30)) <= -25


def test_authorized_session_refreshes_near_expiry_token(monkeypatch) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "aoi_psi_slc.yaml")
    old_token = _jwt_with_expiry(30)
    new_token = _jwt_with_expiry(900)
    monkeypatch.setenv(config.acquisition.auth.bearer_token_env, old_token)
    monkeypatch.setenv(config.acquisition.auth.refresh_token_env, "refresh-token")

    class Response:
        status_code = 200

        def json(self):
            return {"access_token": new_token, "refresh_token": "refresh-token-next"}

    class Session:
        def __init__(self):
            self.headers = {}

        def close(self):
            return None

    import aoi_psi.acquisition as acquisition_module

    def fake_post(url, data, timeout):
        assert data["grant_type"] == "refresh_token"
        assert data["refresh_token"] == "refresh-token"
        return Response()

    monkeypatch.setattr(acquisition_module.requests, "post", fake_post)
    monkeypatch.setattr(acquisition_module.requests, "Session", lambda: Session())

    session, token_env = _authorized_session(config)

    assert session.headers["Authorization"] == f"Bearer {new_token}"
    assert token_env == config.acquisition.auth.bearer_token_env
    assert _resolve_download_token(config, require_valid=True) == (new_token, config.acquisition.auth.bearer_token_env)
    assert os.environ[config.acquisition.auth.refresh_token_env] == "refresh-token-next"


def test_stream_download_writes_final_file_atomically(tmp_path, monkeypatch) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "aoi_psi_slc.yaml")
    monkeypatch.setenv(config.acquisition.auth.bearer_token_env, "token-one")

    class Response:
        def __init__(self, status_code: int, payload: bytes):
            self.status_code = status_code
            self._payload = payload
            self.headers = {}
            self.reason = "Unauthorized" if status_code == 401 else "OK"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def raise_for_status(self):
            if self.status_code >= 400:
                import requests

                raise requests.HTTPError(response=self)

        def iter_content(self, chunk_size: int):
            yield self._payload

    class Session:
        def __init__(self):
            self.headers = {}

        def get(self, href, stream, allow_redirects, timeout):
            assert self.headers["Authorization"] == "Bearer token-one"
            assert stream is True
            assert allow_redirects is True
            assert timeout == (30, 123)
            return Response(200, b"psi-test-bytes")

        def close(self):
            return None

    import aoi_psi.acquisition as acquisition_module

    monkeypatch.setattr(acquisition_module.requests, "Session", lambda: Session())
    destination = tmp_path / "scene.zip"
    _stream_download(
        "https://download.example.invalid/product.zip",
        destination,
        123,
        config=config,
        scene_id="SCENE_1",
        stack_id="STACK_A",
    )

    assert destination.read_bytes() == b"psi-test-bytes"
    assert not destination.with_name("scene.zip.part").exists()


def test_stream_download_retries_on_401_with_refreshed_token(tmp_path, monkeypatch) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "aoi_psi_slc.yaml")
    old_token = _jwt_with_expiry(900)
    new_token = _jwt_with_expiry(900)
    monkeypatch.setenv(config.acquisition.auth.bearer_token_env, old_token)
    monkeypatch.setenv(config.acquisition.auth.refresh_token_env, "refresh-old")

    class Response:
        def __init__(self, status_code: int, payload: bytes = b""):
            self.status_code = status_code
            self._payload = payload
            self.headers = {}
            self.reason = "Unauthorized" if status_code == 401 else "OK"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def raise_for_status(self):
            if self.status_code >= 400:
                import requests

                raise requests.HTTPError(response=self)

        def iter_content(self, chunk_size: int):
            if self._payload:
                yield self._payload

    call_tokens: list[str] = []

    class Session:
        def __init__(self):
            self.headers = {}

        def get(self, href, stream, allow_redirects, timeout):
            token = self.headers["Authorization"].split()[-1]
            call_tokens.append(token)
            if len(call_tokens) == 1:
                return Response(401)
            return Response(200, b"ok")

        def close(self):
            return None

    import aoi_psi.acquisition as acquisition_module

    class TokenResponse:
        status_code = 200

        def json(self):
            return {"access_token": new_token, "refresh_token": "refresh-new"}

    def fake_post(url, data, timeout):
        assert data["grant_type"] == "refresh_token"
        assert data["refresh_token"] in {"refresh-old", "refresh-new"}
        return TokenResponse()

    monkeypatch.setattr(acquisition_module.requests, "Session", lambda: Session())
    monkeypatch.setattr(acquisition_module.requests, "post", fake_post)
    monkeypatch.setattr(acquisition_module.time, "sleep", lambda *_args, **_kwargs: None)
    destination = tmp_path / "scene.zip"
    _stream_download(
        "https://download.example.invalid/product.zip",
        destination,
        123,
        config=config,
        scene_id="SCENE_2",
        stack_id="STACK_B",
    )

    assert destination.read_bytes() == b"ok"
    assert call_tokens == [old_token, new_token]
    assert os.environ[config.acquisition.auth.refresh_token_env] == "refresh-new"


def test_stream_download_does_not_log_token_values(tmp_path, monkeypatch, caplog) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "aoi_psi_slc.yaml")
    old_token = "secret-old-token"
    new_token = "secret-new-token"
    monkeypatch.setenv(config.acquisition.auth.bearer_token_env, old_token)
    monkeypatch.setenv(config.acquisition.auth.refresh_token_env, "refresh-secret")

    class Response:
        def __init__(self, status_code: int, payload: bytes = b""):
            self.status_code = status_code
            self._payload = payload
            self.headers = {}
            self.reason = "Unauthorized" if status_code == 401 else "OK"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def iter_content(self, chunk_size: int):
            if self._payload:
                yield self._payload

    class Session:
        def __init__(self):
            self.headers = {}

        def get(self, href, stream, allow_redirects, timeout):
            token = self.headers["Authorization"].split()[-1]
            if token == old_token:
                return Response(401)
            return Response(200, b"ok")

        def close(self):
            return None

    class TokenResponse:
        status_code = 200

        def json(self):
            return {"access_token": new_token, "refresh_token": "refresh-secret-next"}

    import aoi_psi.acquisition as acquisition_module

    monkeypatch.setattr(acquisition_module.requests, "Session", lambda: Session())
    monkeypatch.setattr(acquisition_module.requests, "post", lambda url, data, timeout: TokenResponse())
    monkeypatch.setattr(acquisition_module.time, "sleep", lambda *_args, **_kwargs: None)
    caplog.set_level(logging.INFO)

    _stream_download(
        "https://download.example.invalid/product.zip",
        tmp_path / "scene.zip",
        123,
        config=config,
        scene_id="SCENE_3",
        stack_id="STACK_C",
    )

    assert old_token not in caplog.text
    assert new_token not in caplog.text


def test_download_scene_via_s3_streams_prefix_into_zip(tmp_path, monkeypatch) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "aoi_psi_slc.yaml")
    scene = SlcScene(
        scene_id="SCENE_6",
        product_name="SCENE_6",
        acquisition_start="2024-01-01T00:00:00Z",
        acquisition_stop="2024-01-01T00:00:10Z",
        acquisition_date="2024-01-01",
        direction="ascending",
        relative_orbit=147,
        polarization="VV+VH",
        swath_mode="IW",
        product_type="IW_SLC__1S",
        processing_level="L1",
        platform="Sentinel-1A",
        asset_name="product",
        href="https://download.example.invalid/scene.zip",
        product_uuid="scene-uuid",
        s3_path="/eodata/Sentinel-1/SAR/SLC/2024/01/01/SCENE_6.SAFE",
    )

    class Body:
        def __init__(self, payload: bytes):
            self.payload = payload
            self.offset = 0

        def read(self, size: int):
            if self.offset >= len(self.payload):
                return b""
            chunk = self.payload[self.offset : self.offset + size]
            self.offset += len(chunk)
            return chunk

        def close(self):
            return None

    class Client:
        def get_object(self, Bucket, Key, Range=None):
            payload = b"manifest" if Key.endswith("manifest.safe") else b"tiff-data"
            return {"Body": Body(payload)}

        def close(self):
            return None

    import zipfile
    import aoi_psi.acquisition as acquisition_module

    monkeypatch.setattr(
        acquisition_module,
        "_list_s3_scene_objects",
        lambda *_args, **_kwargs: (
            [
                S3ObjectInfo(
                    key="Sentinel-1/SAR/SLC/2024/01/01/SCENE_6.SAFE/manifest.safe",
                    size=len(b"manifest"),
                    member_name="SCENE_6.SAFE/manifest.safe",
                ),
                S3ObjectInfo(
                    key="Sentinel-1/SAR/SLC/2024/01/01/SCENE_6.SAFE/measurement/file.tiff",
                    size=len(b"tiff-data"),
                    member_name="SCENE_6.SAFE/measurement/file.tiff",
                ),
            ],
            config.acquisition.s3.endpoint_url,
        ),
    )
    monkeypatch.setattr(acquisition_module, "_s3_client", lambda _config, *, endpoint_url=None: Client())
    destination = tmp_path / "scene.zip"
    _download_scene_via_s3(config, scene, destination, stack_id="STACK_S3")

    with zipfile.ZipFile(destination) as archive:
        assert sorted(archive.namelist()) == ["SCENE_6.SAFE/manifest.safe", "SCENE_6.SAFE/measurement/file.tiff"]
        assert archive.read("SCENE_6.SAFE/manifest.safe") == b"manifest"


def test_s3_client_uses_explicit_transport_config_without_logging_secret_values(monkeypatch, caplog) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "aoi_psi_slc.yaml")
    monkeypatch.setenv(config.acquisition.s3.access_key_env, "secret-access-value")
    monkeypatch.setenv(config.acquisition.s3.secret_key_env, "secret-secret-value")
    captured: dict[str, object] = {}

    class FakeBotoConfig:
        def __init__(self, **kwargs):
            captured["config_kwargs"] = kwargs

    class FakeSession:
        def client(self, service_name, **kwargs):
            captured["service_name"] = service_name
            captured["client_kwargs"] = kwargs
            return object()

    class FakeBoto3:
        class session:
            @staticmethod
            def Session():
                return FakeSession()

    import aoi_psi.acquisition as acquisition_module

    monkeypatch.setattr(
        acquisition_module,
        "_require_boto3",
        lambda: (
            FakeBoto3,
            FakeBotoConfig,
            Exception,
            Exception,
            Exception,
            Exception,
            Exception,
            Exception,
            Exception,
            Exception,
            Exception,
        ),
    )
    caplog.set_level(logging.INFO)

    acquisition_module._s3_client(config, endpoint_url="https://eodata.ams.dataspace.copernicus.eu")

    assert captured["service_name"] == "s3"
    assert captured["client_kwargs"]["endpoint_url"] == "https://eodata.ams.dataspace.copernicus.eu"
    assert captured["config_kwargs"]["signature_version"] == "s3v4"
    assert captured["config_kwargs"]["retries"] == {
        "max_attempts": config.acquisition.s3.max_attempts,
        "mode": config.acquisition.s3.retry_mode,
    }
    assert captured["config_kwargs"]["connect_timeout"] == config.acquisition.s3.connect_timeout_seconds
    assert captured["config_kwargs"]["read_timeout"] == config.acquisition.timeout_seconds
    assert captured["config_kwargs"]["max_pool_connections"] == config.acquisition.s3.max_pool_connections
    assert captured["config_kwargs"]["tcp_keepalive"] is True
    assert captured["config_kwargs"]["s3"] == {"addressing_style": config.acquisition.s3.addressing_style}
    assert "secret-access-value" not in caplog.text
    assert "secret-secret-value" not in caplog.text


def test_download_scene_via_s3_retries_endpoint_connection_error_on_fallback_endpoint(tmp_path, monkeypatch) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "aoi_psi_slc.yaml")
    config.acquisition.s3.download_attempts = 1
    config.acquisition.s3.fallback_endpoint_urls = ("https://eodata.ams.dataspace.copernicus.eu",)
    scene = SlcScene(
        scene_id="SCENE_6B",
        product_name="SCENE_6B",
        acquisition_start="2024-01-01T00:00:00Z",
        acquisition_stop="2024-01-01T00:00:10Z",
        acquisition_date="2024-01-01",
        direction="ascending",
        relative_orbit=147,
        polarization="VV+VH",
        swath_mode="IW",
        product_type="IW_SLC__1S",
        processing_level="L1",
        platform="Sentinel-1A",
        asset_name="product",
        href="https://download.example.invalid/scene.zip",
        product_uuid="scene-uuid",
        s3_path="/eodata/Sentinel-1/SAR/SLC/2024/01/01/SCENE_6B.SAFE",
    )

    class FakeEndpointConnectionError(Exception):
        pass

    class Body:
        def __init__(self, payload: bytes):
            self.payload = payload
            self.offset = 0

        def read(self, size: int):
            if self.offset >= len(self.payload):
                return b""
            chunk = self.payload[self.offset : self.offset + size]
            self.offset += len(chunk)
            return chunk

        def close(self):
            return None

    class FailingClient:
        def get_object(self, Bucket, Key, Range=None):
            raise FakeEndpointConnectionError("primary endpoint failed")

        def close(self):
            return None

    class SuccessClient:
        def get_object(self, Bucket, Key, Range=None):
            return {"Body": Body(b"manifest")}

        def close(self):
            return None

    endpoint_calls: list[str] = []

    import zipfile
    import aoi_psi.acquisition as acquisition_module

    monkeypatch.setattr(
        acquisition_module,
        "_list_s3_scene_objects",
        lambda *_args, **_kwargs: (
            [
                S3ObjectInfo(
                    key="Sentinel-1/SAR/SLC/2024/01/01/SCENE_6B.SAFE/manifest.safe",
                    size=len(b"manifest"),
                    member_name="SCENE_6B.SAFE/manifest.safe",
                )
            ],
            config.acquisition.s3.endpoint_url,
        ),
    )
    monkeypatch.setattr(
        acquisition_module,
        "_require_boto3",
        lambda: (
            object(),
            object(),
            Exception,
            type("FakeClientError", (Exception,), {}),
            FakeEndpointConnectionError,
            type("FakeConnectTimeoutError", (Exception,), {}),
            type("FakeReadTimeoutError", (Exception,), {}),
            type("FakeConnectionClosedError", (Exception,), {}),
            type("FakeHTTPClientError", (Exception,), {}),
            type("FakeSSLError", (Exception,), {}),
            type("FakeBotoConnectionError", (Exception,), {}),
        ),
    )

    def fake_s3_client(_config, *, endpoint_url=None):
        endpoint_calls.append(endpoint_url)
        if endpoint_url == config.acquisition.s3.endpoint_url:
            return FailingClient()
        return SuccessClient()

    monkeypatch.setattr(acquisition_module, "_s3_client", fake_s3_client)
    monkeypatch.setattr(acquisition_module.time, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(acquisition_module.random, "uniform", lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr(acquisition_module, "_endpoint_resolution_summary", lambda endpoint_url: "resolved")

    destination = tmp_path / "scene.zip"
    _download_scene_via_s3(config, scene, destination, stack_id="STACK_S3")

    assert endpoint_calls == [
        config.acquisition.s3.endpoint_url,
        "https://eodata.ams.dataspace.copernicus.eu",
    ]
    with zipfile.ZipFile(destination) as archive:
        assert archive.namelist() == ["SCENE_6B.SAFE/manifest.safe"]
        assert archive.read("SCENE_6B.SAFE/manifest.safe") == b"manifest"
    assert not destination.with_name("scene.zip.part").exists()


def test_download_scene_via_s3_retries_transient_read_timeout_with_fresh_client(tmp_path, monkeypatch) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "aoi_psi_slc.yaml")
    config.acquisition.s3.download_attempts = 2
    config.acquisition.s3.fallback_endpoint_urls = ()
    scene = SlcScene(
        scene_id="SCENE_6C",
        product_name="SCENE_6C",
        acquisition_start="2024-01-01T00:00:00Z",
        acquisition_stop="2024-01-01T00:00:10Z",
        acquisition_date="2024-01-01",
        direction="ascending",
        relative_orbit=147,
        polarization="VV+VH",
        swath_mode="IW",
        product_type="IW_SLC__1S",
        processing_level="L1",
        platform="Sentinel-1A",
        asset_name="product",
        href="https://download.example.invalid/scene.zip",
        product_uuid="scene-uuid",
        s3_path="/eodata/Sentinel-1/SAR/SLC/2024/01/01/SCENE_6C.SAFE",
    )

    class FakeReadTimeoutError(Exception):
        pass

    class Body:
        def __init__(self, payload: bytes):
            self.payload = payload
            self.offset = 0

        def read(self, size: int):
            if self.offset >= len(self.payload):
                return b""
            chunk = self.payload[self.offset : self.offset + size]
            self.offset += len(chunk)
            return chunk

        def close(self):
            return None

    client_creations: list[str] = []

    class Client:
        def __init__(self, fail: bool):
            self.fail = fail

        def get_object(self, Bucket, Key, Range=None):
            if self.fail:
                raise FakeReadTimeoutError("temporary timeout")
            return {"Body": Body(b"manifest")}

        def close(self):
            return None

    import zipfile
    import aoi_psi.acquisition as acquisition_module

    monkeypatch.setattr(
        acquisition_module,
        "_list_s3_scene_objects",
        lambda *_args, **_kwargs: (
            [
                S3ObjectInfo(
                    key="Sentinel-1/SAR/SLC/2024/01/01/SCENE_6C.SAFE/manifest.safe",
                    size=len(b"manifest"),
                    member_name="SCENE_6C.SAFE/manifest.safe",
                )
            ],
            config.acquisition.s3.endpoint_url,
        ),
    )
    monkeypatch.setattr(
        acquisition_module,
        "_require_boto3",
        lambda: (
            object(),
            object(),
            Exception,
            type("FakeClientError", (Exception,), {}),
            type("FakeEndpointConnectionError", (Exception,), {}),
            type("FakeConnectTimeoutError", (Exception,), {}),
            FakeReadTimeoutError,
            type("FakeConnectionClosedError", (Exception,), {}),
            type("FakeHTTPClientError", (Exception,), {}),
            type("FakeSSLError", (Exception,), {}),
            type("FakeBotoConnectionError", (Exception,), {}),
        ),
    )

    def fake_s3_client(_config, *, endpoint_url=None):
        client_creations.append(endpoint_url)
        return Client(fail=len(client_creations) == 1)

    monkeypatch.setattr(acquisition_module, "_s3_client", fake_s3_client)
    monkeypatch.setattr(acquisition_module.time, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(acquisition_module.random, "uniform", lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr(acquisition_module, "_endpoint_resolution_summary", lambda endpoint_url: "resolved")

    destination = tmp_path / "scene.zip"
    _download_scene_via_s3(config, scene, destination, stack_id="STACK_S3")

    assert client_creations == [config.acquisition.s3.endpoint_url, config.acquisition.s3.endpoint_url]
    with zipfile.ZipFile(destination) as archive:
        assert archive.read("SCENE_6C.SAFE/manifest.safe") == b"manifest"
    assert not destination.with_name("scene.zip.part").exists()


def test_download_s3_object_with_resume_recovers_after_midstream_stream_error(tmp_path, monkeypatch) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "aoi_psi_slc.yaml")
    config.acquisition.s3.download_attempts = 2
    config.acquisition.s3.fallback_endpoint_urls = ()
    payload = b"manifest-data"
    object_info = S3ObjectInfo(
        key="Sentinel-1/SAR/SLC/2024/01/01/SCENE_6D.SAFE/manifest.safe",
        size=len(payload),
        member_name="SCENE_6D.SAFE/manifest.safe",
    )

    class FakeBotoCoreError(Exception):
        pass

    class FakeResponseStreamingError(FakeBotoCoreError):
        pass

    class FailingBody:
        def __init__(self, chunk: bytes):
            self.chunk = chunk
            self.read_count = 0

        def read(self, size: int):
            self.read_count += 1
            if self.read_count == 1:
                return self.chunk
            raise FakeResponseStreamingError("stream broke mid-member")

        def close(self):
            return None

    class Body:
        def __init__(self, payload: bytes):
            self.payload = payload
            self.offset = 0

        def read(self, size: int):
            if self.offset >= len(self.payload):
                return b""
            chunk = self.payload[self.offset : self.offset + size]
            self.offset += len(chunk)
            return chunk

        def close(self):
            return None

    range_calls: list[str | None] = []

    class Client:
        def get_object(self, Bucket, Key, Range=None):
            range_calls.append(Range)
            if len(range_calls) == 1:
                return {
                    "Body": FailingBody(payload[:5]),
                    "ResponseMetadata": {"HTTPStatusCode": 200},
                }
            assert Range == "bytes=5-"
            return {
                "Body": Body(payload[5:]),
                "ResponseMetadata": {"HTTPStatusCode": 206},
                "ContentRange": f"bytes 5-{len(payload) - 1}/{len(payload)}",
            }

        def close(self):
            return None

    import aoi_psi.acquisition as acquisition_module

    monkeypatch.setattr(
        acquisition_module,
        "_require_boto3",
        lambda: (
            object(),
            object(),
            FakeBotoCoreError,
            type("FakeClientError", (Exception,), {}),
            type("FakeEndpointConnectionError", (Exception,), {}),
            type("FakeConnectTimeoutError", (Exception,), {}),
            type("FakeReadTimeoutError", (Exception,), {}),
            type("FakeConnectionClosedError", (Exception,), {}),
            type("FakeHTTPClientError", (Exception,), {}),
            type("FakeSSLError", (Exception,), {}),
            type("FakeBotoConnectionError", (Exception,), {}),
        ),
    )
    monkeypatch.setattr(acquisition_module, "_s3_client", lambda _config, *, endpoint_url=None: Client())
    monkeypatch.setattr(acquisition_module, "_member_endpoint_attempt_budget", lambda *_args, **_kwargs: 2)
    monkeypatch.setattr(acquisition_module.time, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(acquisition_module.random, "uniform", lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr(acquisition_module, "_endpoint_resolution_summary", lambda endpoint_url: "resolved")

    stage_dir = tmp_path / "scene.zip.parts"
    _download_s3_object_with_resume(
        config,
        bucket="eodata",
        object_info=object_info,
        stage_dir=stage_dir,
        stack_id="STACK_S3",
        scene_id="SCENE_6D",
    )

    member_path = stage_dir / "SCENE_6D.SAFE" / "manifest.safe"
    assert member_path.read_bytes() == payload
    assert not member_path.with_name("manifest.safe.part").exists()
    assert range_calls == [None, "bytes=5-"]


def test_download_s3_object_with_resume_survives_many_stream_breaks_before_success(tmp_path, monkeypatch) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "aoi_psi_slc.yaml")
    config.acquisition.s3.download_attempts = 1
    config.acquisition.s3.fallback_endpoint_urls = ()
    payload = b"0123456789"
    object_info = S3ObjectInfo(
        key="Sentinel-1/SAR/SLC/2024/01/01/SCENE_6DD.SAFE/manifest.safe",
        size=len(payload),
        member_name="SCENE_6DD.SAFE/manifest.safe",
    )

    class FakeBotoCoreError(Exception):
        pass

    class FakeResponseStreamingError(FakeBotoCoreError):
        pass

    class FlakyBody:
        def __init__(self, payload: bytes, *, fail_after_chunk: bool):
            self.payload = payload
            self.fail_after_chunk = fail_after_chunk
            self.offset = 0
            self.sent_once = False

        def read(self, size: int):
            if self.fail_after_chunk and self.sent_once:
                raise FakeResponseStreamingError("stream broke again")
            if self.offset >= len(self.payload):
                return b""
            self.sent_once = True
            chunk = self.payload[self.offset : self.offset + 1]
            self.offset += len(chunk)
            return chunk

        def close(self):
            return None

    range_calls: list[str | None] = []

    class Client:
        def get_object(self, Bucket, Key, Range=None):
            range_calls.append(Range)
            if len(range_calls) < len(payload):
                start = 0 if Range is None else int(Range.removeprefix("bytes=").removesuffix("-"))
                return {
                    "Body": FlakyBody(payload[start : start + 1], fail_after_chunk=True),
                    "ResponseMetadata": {"HTTPStatusCode": 206 if Range else 200},
                    "ContentRange": f"bytes {start}-{len(payload) - 1}/{len(payload)}" if Range else None,
                }
            start = int(range_calls[-1].removeprefix("bytes=").removesuffix("-"))
            return {
                "Body": FlakyBody(payload[start:], fail_after_chunk=False),
                "ResponseMetadata": {"HTTPStatusCode": 206},
                "ContentRange": f"bytes {start}-{len(payload) - 1}/{len(payload)}",
            }

        def close(self):
            return None

    import aoi_psi.acquisition as acquisition_module

    monkeypatch.setattr(
        acquisition_module,
        "_require_boto3",
        lambda: (
            object(),
            object(),
            FakeBotoCoreError,
            type("FakeClientError", (Exception,), {}),
            type("FakeEndpointConnectionError", (Exception,), {}),
            type("FakeConnectTimeoutError", (Exception,), {}),
            type("FakeReadTimeoutError", (Exception,), {}),
            type("FakeConnectionClosedError", (Exception,), {}),
            type("FakeHTTPClientError", (Exception,), {}),
            type("FakeSSLError", (Exception,), {}),
            type("FakeBotoConnectionError", (Exception,), {}),
        ),
    )
    monkeypatch.setattr(acquisition_module, "_s3_client", lambda _config, *, endpoint_url=None: Client())
    monkeypatch.setattr(acquisition_module, "_member_endpoint_attempt_budget", lambda *_args, **_kwargs: 10)
    monkeypatch.setattr(acquisition_module.time, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(acquisition_module.random, "uniform", lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr(acquisition_module, "_endpoint_resolution_summary", lambda endpoint_url: "resolved")

    stage_dir = tmp_path / "scene.zip.parts"
    _download_s3_object_with_resume(
        config,
        bucket="eodata",
        object_info=object_info,
        stage_dir=stage_dir,
        stack_id="STACK_S3",
        scene_id="SCENE_6DD",
    )

    member_path = stage_dir / "SCENE_6DD.SAFE" / "manifest.safe"
    assert member_path.read_bytes() == payload
    assert not member_path.with_name("manifest.safe.part").exists()
    assert range_calls == [
        None,
        "bytes=1-",
        "bytes=2-",
        "bytes=3-",
        "bytes=4-",
        "bytes=5-",
        "bytes=6-",
        "bytes=7-",
        "bytes=8-",
        "bytes=9-",
    ]


def test_download_s3_object_with_resume_uses_existing_partial_file(tmp_path, monkeypatch) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "aoi_psi_slc.yaml")
    config.acquisition.s3.download_attempts = 1
    config.acquisition.s3.fallback_endpoint_urls = ()
    payload = b"manifest-data"
    object_info = S3ObjectInfo(
        key="Sentinel-1/SAR/SLC/2024/01/01/SCENE_6E.SAFE/manifest.safe",
        size=len(payload),
        member_name="SCENE_6E.SAFE/manifest.safe",
    )

    class Body:
        def __init__(self, payload: bytes):
            self.payload = payload
            self.offset = 0

        def read(self, size: int):
            if self.offset >= len(self.payload):
                return b""
            chunk = self.payload[self.offset : self.offset + size]
            self.offset += len(chunk)
            return chunk

        def close(self):
            return None

    range_calls: list[str | None] = []

    class Client:
        def get_object(self, Bucket, Key, Range=None):
            range_calls.append(Range)
            assert Range == "bytes=4-"
            return {
                "Body": Body(payload[4:]),
                "ResponseMetadata": {"HTTPStatusCode": 206},
                "ContentRange": f"bytes 4-{len(payload) - 1}/{len(payload)}",
            }

        def close(self):
            return None

    import aoi_psi.acquisition as acquisition_module

    monkeypatch.setattr(acquisition_module, "_s3_client", lambda _config, *, endpoint_url=None: Client())
    monkeypatch.setattr(acquisition_module, "_member_endpoint_attempt_budget", lambda *_args, **_kwargs: 1)
    stage_dir = tmp_path / "scene.zip.parts"
    partial_path = stage_dir / "SCENE_6E.SAFE" / "manifest.safe.part"
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path.write_bytes(payload[:4])

    _download_s3_object_with_resume(
        config,
        bucket="eodata",
        object_info=object_info,
        stage_dir=stage_dir,
        stack_id="STACK_S3",
        scene_id="SCENE_6E",
    )

    member_path = stage_dir / "SCENE_6E.SAFE" / "manifest.safe"
    assert member_path.read_bytes() == payload
    assert not partial_path.exists()
    assert range_calls == ["bytes=4-"]


def test_download_s3_object_with_resume_preserves_partial_when_retries_exhaust(tmp_path, monkeypatch) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "aoi_psi_slc.yaml")
    config.acquisition.s3.download_attempts = 2
    config.acquisition.s3.fallback_endpoint_urls = ()
    payload = b"manifest-data"
    object_info = S3ObjectInfo(
        key="Sentinel-1/SAR/SLC/2024/01/01/SCENE_6F.SAFE/manifest.safe",
        size=len(payload),
        member_name="SCENE_6F.SAFE/manifest.safe",
    )

    class FakeBotoCoreError(Exception):
        pass

    class FakeResponseStreamingError(FakeBotoCoreError):
        pass

    class FailingBody:
        def __init__(self, payload: bytes):
            self.payload = payload
            self.sent = False

        def read(self, size: int):
            if not self.sent:
                self.sent = True
                return self.payload[:5]
            raise FakeResponseStreamingError("stream broke again")

        def close(self):
            return None

    class Client:
        def get_object(self, Bucket, Key, Range=None):
            return {
                "Body": FailingBody(payload),
                "ResponseMetadata": {"HTTPStatusCode": 206 if Range else 200},
                "ContentRange": f"bytes 5-{len(payload) - 1}/{len(payload)}" if Range else None,
            }

        def close(self):
            return None

    import aoi_psi.acquisition as acquisition_module

    monkeypatch.setattr(
        acquisition_module,
        "_require_boto3",
        lambda: (
            object(),
            object(),
            FakeBotoCoreError,
            type("FakeClientError", (Exception,), {}),
            type("FakeEndpointConnectionError", (Exception,), {}),
            type("FakeConnectTimeoutError", (Exception,), {}),
            type("FakeReadTimeoutError", (Exception,), {}),
            type("FakeConnectionClosedError", (Exception,), {}),
            type("FakeHTTPClientError", (Exception,), {}),
            type("FakeSSLError", (Exception,), {}),
            type("FakeBotoConnectionError", (Exception,), {}),
        ),
    )
    monkeypatch.setattr(acquisition_module, "_s3_client", lambda _config, *, endpoint_url=None: Client())
    monkeypatch.setattr(acquisition_module, "_member_endpoint_attempt_budget", lambda *_args, **_kwargs: 2)
    monkeypatch.setattr(acquisition_module.time, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(acquisition_module.random, "uniform", lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr(acquisition_module, "_endpoint_resolution_summary", lambda endpoint_url: "resolved")

    stage_dir = tmp_path / "scene.zip.parts"
    with pytest.raises(FakeResponseStreamingError, match="stream broke again"):
        _download_s3_object_with_resume(
            config,
            bucket="eodata",
            object_info=object_info,
            stage_dir=stage_dir,
            stack_id="STACK_S3",
            scene_id="SCENE_6F",
        )

    partial_path = stage_dir / "SCENE_6F.SAFE" / "manifest.safe.part"
    assert partial_path.exists()
    assert partial_path.stat().st_size == 10
    assert not (stage_dir / "SCENE_6F.SAFE" / "manifest.safe").exists()


def test_download_stack_scenes_uses_s3_and_persists_resolved_path(tmp_path, monkeypatch) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "aoi_psi_slc.yaml")
    config.acquisition.download_transport = "auto"
    monkeypatch.setenv(config.acquisition.s3.access_key_env, "access-key")
    monkeypatch.setenv(config.acquisition.s3.secret_key_env, "secret-key")
    monkeypatch.setenv(config.acquisition.auth.bearer_token_env, _jwt_with_expiry(900))
    context = RunContext.create(config, tmp_path)
    context.ensure_directories()
    scene = SlcScene(
        scene_id="SCENE_7",
        product_name="SCENE_7",
        acquisition_start="2024-01-01T00:00:00Z",
        acquisition_stop="2024-01-01T00:00:10Z",
        acquisition_date="2024-01-01",
        direction="ascending",
        relative_orbit=147,
        polarization="VV+VH",
        swath_mode="IW",
        product_type="IW_SLC__1S",
        processing_level="L1",
        platform="Sentinel-1A",
        asset_name="product",
        href="https://download.example.invalid/scene.zip",
        product_uuid="scene-uuid",
    )
    manifest = StackManifest(
        stack_id="asc_rel147_vv",
        direction="ascending",
        relative_orbit=147,
        product_type="SLC",
        scenes=[scene],
    )
    write_stack_manifest(manifest, stack_manifest_path(context, manifest.stack_id))
    resolved_path = "/eodata/Sentinel-1/SAR/SLC/2024/01/01/SCENE_7.SAFE"
    calls: list[str] = []

    import aoi_psi.acquisition as acquisition_module

    monkeypatch.setattr(
        acquisition_module,
        "_stac_scene_s3_paths",
        lambda *_args, **_kwargs: {"SCENE_7": resolved_path},
    )

    def fake_download_scene_via_s3(_config, s3_scene, destination, *, stack_id):
        calls.append("s3")
        assert s3_scene.s3_path == resolved_path
        destination.write_bytes(b"s3")

    monkeypatch.setattr(acquisition_module, "_download_scene_via_s3", fake_download_scene_via_s3)

    download_stack_scenes(config, context, manifest)
    restored = read_stack_manifest(stack_manifest_path(context, manifest.stack_id))

    assert calls == ["s3"]
    assert restored.scenes[0].s3_path == resolved_path
    assert (context.slc_dir / manifest.stack_id / "SCENE_7.zip").read_bytes() == b"s3"


def test_download_stack_scenes_surfaces_s3_errors_in_auto_mode(tmp_path, monkeypatch) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "aoi_psi_slc.yaml")
    config.acquisition.download_transport = "auto"
    monkeypatch.setenv(config.acquisition.s3.access_key_env, "access-key")
    monkeypatch.setenv(config.acquisition.s3.secret_key_env, "secret-key")
    context = RunContext.create(config, tmp_path)
    context.ensure_directories()
    scene = SlcScene(
        scene_id="SCENE_8",
        product_name="SCENE_8",
        acquisition_start="2024-01-01T00:00:00Z",
        acquisition_stop="2024-01-01T00:00:10Z",
        acquisition_date="2024-01-01",
        direction="ascending",
        relative_orbit=147,
        polarization="VV+VH",
        swath_mode="IW",
        product_type="IW_SLC__1S",
        processing_level="L1",
        platform="Sentinel-1A",
        asset_name="product",
        href="https://download.example.invalid/scene.zip",
        product_uuid="scene-uuid",
        s3_path="/eodata/Sentinel-1/SAR/SLC/2024/01/01/SCENE_8.SAFE",
    )
    manifest = StackManifest(
        stack_id="asc_rel147_vv",
        direction="ascending",
        relative_orbit=147,
        product_type="SLC",
        scenes=[scene],
    )

    import aoi_psi.acquisition as acquisition_module

    monkeypatch.setattr(acquisition_module, "_download_scene_via_s3", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("s3 down")))

    with pytest.raises(RuntimeError, match="s3 down"):
        download_stack_scenes(config, context, manifest)


def test_download_stack_scenes_reuses_completed_zip(tmp_path, monkeypatch) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "aoi_psi_slc.yaml")
    config.cache.reuse_downloads = True
    monkeypatch.setenv(config.acquisition.auth.bearer_token_env, _jwt_with_expiry(900))
    context = RunContext.create(config, tmp_path)
    context.ensure_directories()
    scene = SlcScene(
        scene_id="SCENE_4",
        product_name="SCENE_4",
        acquisition_start="2024-01-01T00:00:00Z",
        acquisition_stop="2024-01-01T00:00:10Z",
        acquisition_date="2024-01-01",
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
    manifest = StackManifest(
        stack_id="asc_rel147_vv",
        direction="ascending",
        relative_orbit=147,
        product_type="SLC",
        scenes=[scene],
    )
    target = context.slc_dir / manifest.stack_id / f"{scene.product_name}.zip"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"completed")
    called = {"count": 0}

    import aoi_psi.acquisition as acquisition_module

    def fake_stream_download(*args, **kwargs):
        called["count"] += 1

    monkeypatch.setattr(acquisition_module, "_stream_download", fake_stream_download)

    records = download_stack_scenes(config, context, manifest)

    assert called["count"] == 0
    assert len(records) == 1
    assert records[0].path == target


def test_download_stack_scenes_reuses_completed_zip_without_auth_when_nothing_is_pending(tmp_path, monkeypatch) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "aoi_psi_slc.yaml")
    config.cache.reuse_downloads = True
    context = RunContext.create(config, tmp_path)
    context.ensure_directories()
    scene = SlcScene(
        scene_id="SCENE_4B",
        product_name="SCENE_4B",
        acquisition_start="2024-01-01T00:00:00Z",
        acquisition_stop="2024-01-01T00:00:10Z",
        acquisition_date="2024-01-01",
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
    manifest = StackManifest(
        stack_id="asc_rel147_vv",
        direction="ascending",
        relative_orbit=147,
        product_type="SLC",
        scenes=[scene],
    )
    target = context.slc_dir / manifest.stack_id / f"{scene.product_name}.zip"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"completed")

    import aoi_psi.acquisition as acquisition_module

    def fail_if_called(*args, **kwargs):
        raise AssertionError("ensure_download_auth should not run when every ZIP is already present")

    monkeypatch.setattr(acquisition_module, "ensure_download_auth", fail_if_called)

    records = download_stack_scenes(config, context, manifest)

    assert len(records) == 1
    assert records[0].path == target
    assert target.read_bytes() == b"completed"


def test_download_stack_scenes_reuses_completed_zip_from_prior_attempt_without_auth(tmp_path, monkeypatch) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "aoi_psi_slc.yaml")
    config.cache.reuse_downloads = True

    source_context = RunContext.create(
        config,
        tmp_path,
        run_dir=tmp_path / "runs_psi" / "previous-group" / "attempt-001",
    )
    source_context.ensure_directories()
    context = RunContext.create(
        config,
        tmp_path,
        run_dir=tmp_path / "runs_psi" / "new-group" / "attempt-001",
    )
    context.ensure_directories()

    scene = SlcScene(
        scene_id="SCENE_REUSE",
        product_name="SCENE_REUSE",
        acquisition_start="2024-01-01T00:00:00Z",
        acquisition_stop="2024-01-01T00:00:10Z",
        acquisition_date="2024-01-01",
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
    manifest = StackManifest(
        stack_id="asc_rel147_vv",
        direction="ascending",
        relative_orbit=147,
        product_type="SLC",
        scenes=[scene],
    )
    source = source_context.slc_dir / manifest.stack_id / f"{scene.product_name}.zip"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"completed-from-prior-attempt")

    import aoi_psi.acquisition as acquisition_module

    def fail_if_called(*args, **kwargs):
        raise AssertionError("ensure_download_auth should not run when every ZIP is satisfied by cross-attempt reuse")

    monkeypatch.setattr(acquisition_module, "ensure_download_auth", fail_if_called)

    records = download_stack_scenes(config, context, manifest)

    target = context.slc_dir / manifest.stack_id / f"{scene.product_name}.zip"
    assert len(records) == 1
    assert records[0].path == target
    assert target.exists()
    assert target.read_bytes() == b"completed-from-prior-attempt"
    assert target.stat().st_ino == source.stat().st_ino
    assert target.stat().st_nlink >= 2


def test_download_stack_scenes_reuses_prior_attempt_zip_and_downloads_only_missing_scene(tmp_path, monkeypatch) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "aoi_psi_slc.yaml")
    config.cache.reuse_downloads = True

    source_context = RunContext.create(
        config,
        tmp_path,
        run_dir=tmp_path / "runs_psi" / "previous-group" / "attempt-001",
    )
    source_context.ensure_directories()
    context = RunContext.create(
        config,
        tmp_path,
        run_dir=tmp_path / "runs_psi" / "new-group" / "attempt-001",
    )
    context.ensure_directories()

    reused_scene = SlcScene(
        scene_id="SCENE_REUSED",
        product_name="SCENE_REUSED",
        acquisition_start="2024-01-01T00:00:00Z",
        acquisition_stop="2024-01-01T00:00:10Z",
        acquisition_date="2024-01-01",
        direction="ascending",
        relative_orbit=147,
        polarization="VV+VH",
        swath_mode="IW",
        product_type="IW_SLC__1S",
        processing_level="L1",
        platform="Sentinel-1A",
        asset_name="product",
        href="https://download.example.invalid/reused.zip",
    )
    missing_scene = SlcScene(
        scene_id="SCENE_MISSING",
        product_name="SCENE_MISSING",
        acquisition_start="2024-01-13T00:00:00Z",
        acquisition_stop="2024-01-13T00:00:10Z",
        acquisition_date="2024-01-13",
        direction="ascending",
        relative_orbit=147,
        polarization="VV+VH",
        swath_mode="IW",
        product_type="IW_SLC__1S",
        processing_level="L1",
        platform="Sentinel-1A",
        asset_name="product",
        href="https://download.example.invalid/missing.zip",
    )
    manifest = StackManifest(
        stack_id="asc_rel147_vv",
        direction="ascending",
        relative_orbit=147,
        product_type="SLC",
        scenes=[reused_scene, missing_scene],
    )
    source = source_context.slc_dir / manifest.stack_id / f"{reused_scene.product_name}.zip"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"completed-from-prior-attempt")

    import aoi_psi.acquisition as acquisition_module

    auth_calls = {"count": 0}
    stream_calls: list[str] = []

    monkeypatch.setattr(acquisition_module, "ensure_download_auth", lambda *_args, **_kwargs: auth_calls.__setitem__("count", auth_calls["count"] + 1))

    def fake_stream_download(href, target, *_args, **_kwargs):
        stream_calls.append(href)
        target.write_bytes(b"downloaded-now")

    monkeypatch.setattr(acquisition_module, "_stream_download", fake_stream_download)

    records = download_stack_scenes(config, context, manifest)

    reused_target = context.slc_dir / manifest.stack_id / f"{reused_scene.product_name}.zip"
    missing_target = context.slc_dir / manifest.stack_id / f"{missing_scene.product_name}.zip"
    assert len(records) == 2
    assert auth_calls["count"] == 1
    assert stream_calls == ["https://download.example.invalid/missing.zip"]
    assert reused_target.exists()
    assert reused_target.read_bytes() == b"completed-from-prior-attempt"
    assert reused_target.stat().st_ino == source.stat().st_ino
    assert missing_target.exists()
    assert missing_target.read_bytes() == b"downloaded-now"


def test_download_stack_scenes_removes_stale_partial_before_s3_retry(tmp_path, monkeypatch) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "aoi_psi_slc.yaml")
    config.acquisition.download_transport = "s3"
    monkeypatch.setenv(config.acquisition.s3.access_key_env, "access-key")
    monkeypatch.setenv(config.acquisition.s3.secret_key_env, "secret-key")
    context = RunContext.create(config, tmp_path)
    context.ensure_directories()
    scene = SlcScene(
        scene_id="SCENE_9",
        product_name="SCENE_9",
        acquisition_start="2024-01-01T00:00:00Z",
        acquisition_stop="2024-01-01T00:00:10Z",
        acquisition_date="2024-01-01",
        direction="ascending",
        relative_orbit=147,
        polarization="VV+VH",
        swath_mode="IW",
        product_type="IW_SLC__1S",
        processing_level="L1",
        platform="Sentinel-1A",
        asset_name="product",
        href="https://download.example.invalid/scene.zip",
        product_uuid="scene-uuid",
        s3_path="/eodata/Sentinel-1/SAR/SLC/2024/01/01/SCENE_9.SAFE",
    )
    manifest = StackManifest(
        stack_id="asc_rel147_vv",
        direction="ascending",
        relative_orbit=147,
        product_type="SLC",
        scenes=[scene],
    )
    target = context.slc_dir / manifest.stack_id / "SCENE_9.zip"
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name("SCENE_9.zip.part")
    partial.write_bytes(b"stale")
    calls = {"count": 0}

    import aoi_psi.acquisition as acquisition_module

    def fake_download_scene_via_s3(_config, _scene, destination, *, stack_id):
        calls["count"] += 1
        assert not partial.exists()
        destination.write_bytes(b"s3")

    monkeypatch.setattr(acquisition_module, "_download_scene_via_s3", fake_download_scene_via_s3)

    download_stack_scenes(config, context, manifest)

    assert calls["count"] == 1
    assert target.read_bytes() == b"s3"
    assert not partial.exists()
