"""Tiny append-only JSONL cache so long AI runs can be stopped and resumed."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def text_key(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:20]


class JsonlCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.data: dict[str, dict] = {}
        if self.path.exists():
            with self.path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        self.data[rec["_key"]] = rec
                    except Exception:
                        continue

    def __contains__(self, k: str) -> bool:
        return k in self.data

    def get(self, k: str):
        return self.data.get(k)

    def put(self, k: str, rec: dict):
        rec = dict(rec, _key=k)
        self.data[k] = rec
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    def put_many(self, items: list[tuple[str, dict]]):
        if not items:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            for k, rec in items:
                rec = dict(rec, _key=k)
                self.data[k] = rec
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
