"""Content-addressed response cache.

The probes overlap heavily. Leave-one-out, the sufficient-set search and the
counterfactual search all evaluate overlapping subsets of the same segments,
and the baseline is requested by every one of them. Caching on the exact
request body removes that duplication, which matters more than usual while
capacity is the scarce resource rather than money.

The key is the canonical request: model, state and questions. Nothing else can
change the answer, and the model id is part of the key so a version diff never
reads a cached answer from the wrong version.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

from .types import SystemOneRequest

__all__ = ["CacheKey", "ResponseCache", "MemoryCache", "DiskCache", "NullCache", "cache_key"]

CacheKey = str


def cache_key(request: SystemOneRequest) -> CacheKey:
    """Hash of the exact bytes that will be sent.

    Deliberately **not** key-sorted. Sorting looks like the right way to make the
    key robust to irrelevant dictionary ordering, and it is wrong here: JSON
    object order is transmitted, so the order you list a Choice's options in is
    part of the request. The option-order probe varies exactly that, and a sorted
    key would serve every permutation from cache and report perfect stability no
    matter how order-sensitive the model actually is.

    Two requests share an answer only when they are byte-identical.
    """
    canonical = json.dumps(
        request.wire(), separators=(",", ":"), ensure_ascii=False, default=str
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@runtime_checkable
class ResponseCache(Protocol):
    """Stores raw response bodies. Parsing happens above the cache so a schema
    fix does not invalidate recorded traffic."""

    def get(self, key: CacheKey) -> Mapping[str, Any] | None: ...

    def put(self, key: CacheKey, body: Mapping[str, Any]) -> None: ...


class NullCache:
    """Caches nothing. Use when measuring real latency or true request counts."""

    __slots__ = ()

    def get(self, key: CacheKey) -> Mapping[str, Any] | None:
        return None

    def put(self, key: CacheKey, body: Mapping[str, Any]) -> None:
        return None


class MemoryCache:
    """Process-local cache. The default: an explanation is one process."""

    __slots__ = ("_entries",)

    def __init__(self) -> None:
        self._entries: dict[CacheKey, Mapping[str, Any]] = {}

    def get(self, key: CacheKey) -> Mapping[str, Any] | None:
        return self._entries.get(key)

    def put(self, key: CacheKey, body: Mapping[str, Any]) -> None:
        self._entries[key] = body

    def __len__(self) -> int:
        return len(self._entries)


class DiskCache:
    """Persists bodies as JSON under a directory.

    Turns a recorded session into a fixture you can replay offline and into CI,
    which is how the stability probes stay reproducible across model versions.

    Recorded bodies contain the answers to your real state. Treat the directory
    as sensitive; the shipped .gitignore already excludes the default location.
    """

    __slots__ = ("_root", "_memory")

    def __init__(self, root: str | Path, *, memoize: bool = True) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._memory = MemoryCache() if memoize else None

    def _path(self, key: CacheKey) -> Path:
        # Shard by the first two hex chars to keep directories small.
        return self._root / key[:2] / f"{key}.json"

    def get(self, key: CacheKey) -> Mapping[str, Any] | None:
        if self._memory is not None:
            hit = self._memory.get(key)
            if hit is not None:
                return hit
        path = self._path(key)
        if not path.exists():
            return None
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None  # a corrupt entry is a miss, not a crash
        if self._memory is not None:
            self._memory.put(key, body)
        return body

    def put(self, key: CacheKey, body: Mapping[str, Any]) -> None:
        if self._memory is not None:
            self._memory.put(key, body)
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)  # atomic, so a killed run leaves no half-written entry
