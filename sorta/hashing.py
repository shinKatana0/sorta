"""File hashing: blake3 (if installed) with a sha256 fallback."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

try:
    import blake3  # type: ignore
    _ALGO = "blake3"

    def _new():
        return blake3.blake3()
except ImportError:  # pragma: no cover
    _ALGO = "sha256"

    def _new():
        return hashlib.sha256()

_CHUNK = 1 << 20  # 1 MiB


def file_hash(path: str | Path) -> tuple[str, str]:
    """Returns (hex hash, algorithm name)."""
    h = _new()
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest(), _ALGO


def resolve_workers(raw: dict | None) -> int:
    """Number of threads for parallel per-file operations (indexing, pHash).

    `index.workers` in config.yaml (available via `cfg.raw` — we do not add a
    typed field here); default min(8, cpu_count). Measured empirically (see F11):
    blake3-py and Pillow decoding release the GIL during read/decode, so a
    ThreadPoolExecutor scales without a ProcessPoolExecutor.
    """
    default = min(8, os.cpu_count() or 1)
    idx = (raw or {}).get("index") or {}
    workers = idx.get("workers")
    if workers is None:
        return default
    try:
        n = int(workers)
    except (TypeError, ValueError):
        return default
    return n if n > 0 else default
