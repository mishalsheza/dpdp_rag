"""On-disk response cache for development: one JSON file per (config_hash, question, date)."""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import date
from pathlib import Path
from typing import Any


def cache_key(config_hash: str, question: str, as_of: date) -> str:
    normalized = re.sub(r"\s+", " ", question).strip()
    raw = json.dumps([config_hash, normalized, as_of.isoformat()], ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


class ResponseCache:
    def __init__(self, directory: Path) -> None:
        self.dir = directory

    def get(self, key: str) -> dict[str, Any] | None:
        path = self.dir / f"{key}.json"
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def put(self, key: str, payload: dict[str, Any]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.dir / f".{key}.{os.getpid()}.tmp"
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.dir / f"{key}.json")  # atomic, so readers never see partial files
