from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from casablanca_builtup.config import PipelineConfig


def stable_config_hash(config: PipelineConfig, length: int = 12) -> str:
    payload = json.dumps(
        config.model_dump(mode="json", exclude_none=True),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length]


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def stable_hash(payload: dict[str, Any], length: int = 12) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:length]
