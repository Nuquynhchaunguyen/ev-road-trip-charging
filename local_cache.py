"""Small expiring disk cache for successful public routing/place responses.

This stores road estimates and amenity records, never charger availability or
live prices. Failure to read/write the cache leaves normal network lookup intact.
"""
from contextlib import closing
import json
import sqlite3
import time
from pathlib import Path


class LocalCache:
    def __init__(self, path):
        self.path = Path(path)

    def get(self, kind, key, ttl):
        if not self.path.exists():
            return None
        try:
            with closing(sqlite3.connect(self.path, timeout=2)) as connection:
                row = connection.execute(
                    "SELECT payload, saved_at FROM responses WHERE kind=? AND lookup_key=?",
                    (kind, json.dumps(key, separators=(",", ":"))),
                ).fetchone()
            if row is None or time.time() - row[1] >= ttl:
                return None
            return json.loads(row[0])
        except (OSError, sqlite3.Error, ValueError):
            return None

    def put(self, kind, key, value):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with closing(sqlite3.connect(self.path, timeout=2)) as connection:
                connection.execute("CREATE TABLE IF NOT EXISTS responses ("
                    "kind TEXT, lookup_key TEXT, payload TEXT, saved_at REAL, "
                    "PRIMARY KEY (kind, lookup_key))")
                connection.execute("INSERT OR REPLACE INTO responses VALUES (?, ?, ?, ?)",
                    (kind, json.dumps(key, separators=(",", ":")),
                     json.dumps(value, allow_nan=False), time.time()))
                connection.commit()
        except (OSError, sqlite3.Error, TypeError, ValueError):
            pass
