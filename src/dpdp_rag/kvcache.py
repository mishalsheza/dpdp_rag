"""A tiny persistent key-value store on SQLite, used for embedding and LLM-call caches.

One file per cache keeps CI caching simple (actions/cache restores a single path).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any


def digest(*parts: Any) -> str:
    """Stable sha256 of JSON-serialisable parts."""
    raw = json.dumps(parts, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


class SqliteKV:
    def __init__(self, path: Path, table: str) -> None:
        if not table.isidentifier():
            raise ValueError(f"bad table name {table!r}")
        self.path = path
        self.table = table
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute(f"CREATE TABLE IF NOT EXISTS {table} (key TEXT PRIMARY KEY, value TEXT)")
        self._conn.commit()

    def get(self, key: str) -> Any | None:
        with self._lock:
            row = self._conn.execute(
                f"SELECT value FROM {self.table} WHERE key = ?", (key,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def get_many(self, keys: list[str]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        with self._lock:
            for start in range(0, len(keys), 500):
                part = keys[start : start + 500]
                marks = ",".join("?" * len(part))
                rows = self._conn.execute(
                    f"SELECT key, value FROM {self.table} WHERE key IN ({marks})", part
                ).fetchall()
                out.update({k: json.loads(v) for k, v in rows})
        return out

    def put_many(self, items: dict[str, Any]) -> None:
        with self._lock:
            self._conn.executemany(
                f"INSERT OR REPLACE INTO {self.table} (key, value) VALUES (?, ?)",
                [(k, json.dumps(v)) for k, v in items.items()],
            )
            self._conn.commit()

    def put(self, key: str, value: Any) -> None:
        self.put_many({key: value})

    def __len__(self) -> int:
        with self._lock:
            return int(self._conn.execute(f"SELECT COUNT(*) FROM {self.table}").fetchone()[0])
