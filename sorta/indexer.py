"""FR-1: scanning, metadata, hashes, incrementality.

Invariant: original files are never modified.
A re-run skips files with matching path+size+mtime.
"""
from __future__ import annotations

import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator

from .config import Config
from .dates import resolve_taken_at
from .exif import read_batch
from .hashing import file_hash, resolve_workers

_BATCH = 200

# F17: movie/series release names — a strong, reliable "not personal" signal.
# Personal videos (VID_/PXL_/camera/messenger names) do not match these patterns.
_SEASON_EPISODE_RE = re.compile(r"(?i)\bS\d{1,2}E\d{1,3}\b|\b\d{1,2}x\d{2}\b")
_RESOLUTION_RE = re.compile(r"(?i)\b(720p|1080p|2160p|4k)\b")
_SOURCE_RE = re.compile(r"(?i)\b(webrip|web-?dl|bluray|bdrip|hdtv|dvdrip)\b")
_CODEC_RE = re.compile(r"(?i)\b(x264|x265|hevc|h\.?264|h\.?265)\b")
_GROUP_RE = re.compile(r"\[[^\[\]]{2,30}\]")
# Dot-separated release names: 3+ dot-separated tokens, then a 4-digit year
# (Movie.Name.2021.mp4) — weaker than the other signals on its own, but the brief
# treats it as a release pattern; size is deliberately not used as a signal — a very
# large file means nothing by itself (4K family video is large too).
_DOTTED_RELEASE_RE = re.compile(r"(?:[A-Za-z0-9]+\.){3,}(?:19|20)\d{2}\.")

_RELEASE_PATTERNS = (
    _SEASON_EPISODE_RE, _RESOLUTION_RE, _SOURCE_RE, _CODEC_RE,
    _GROUP_RE, _DOTTED_RELEASE_RE,
)


def is_not_personal_video(name: str, size: int = 0) -> bool:
    """Pure heuristic: a movie/series release name -> not personal media."""
    return any(p.search(name) for p in _RELEASE_PATTERNS)


@dataclass
class IndexStats:
    scanned: int = 0
    added: int = 0
    updated: int = 0
    skipped: int = 0
    errors: int = 0


def _walk(cfg: Config) -> Iterator[Path]:
    skip = set(cfg.index.skip_dirs)
    min_size = cfg.index.min_file_size_kb * 1024
    for src in cfg.sources:
        for p in sorted(src.rglob("*")):
            if any(part in skip or part.startswith(".") for part in p.parts):
                continue
            if not p.is_file():
                continue
            if cfg.index.media_type_of(p.suffix) is None:
                continue
            try:
                if p.stat().st_size < min_size:
                    continue
            except OSError:
                continue
            yield p


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    return any(r["name"] == column for r in conn.execute(f"PRAGMA table_info({table})"))


def _needs_update(conn: sqlite3.Connection, path: str, size: int, mtime: float) -> str | None:
    """None = skip; 'add' | 'update' = process."""
    row = conn.execute("SELECT size, mtime FROM files WHERE path = ?", (path,)).fetchone()
    if row is None:
        return "add"
    if row["size"] == size and abs(row["mtime"] - mtime) < 1e-6:
        return None
    return "update"


@dataclass
class _HashResult:
    """Result of the heavy per-file work (stat + blake3) computed in the thread pool."""
    path: Path
    action: str
    size: int = 0
    mtime: float = 0.0
    hash: str | None = None
    algo: str | None = None
    error: str | None = None


def _hash_one(item: tuple[Path, str]) -> _HashResult:
    p, action = item
    try:
        st = p.stat()
        h, algo = file_hash(p)
        return _HashResult(p, action, st.st_size, st.st_mtime, h, algo)
    except Exception as e:  # corrupt/vanished file — does not crash the pool
        return _HashResult(p, action, error=f"{type(e).__name__}: {e}")


def index(cfg: Config, conn: sqlite3.Connection,
          progress: Callable[[IndexStats], None] | None = None) -> IndexStats:
    stats = IndexStats()
    pending: list[tuple[Path, str]] = []  # (path, 'add'|'update')
    # Orientation is always extracted, but written only if the column has already been
    # added to the schema (schema migration runs separately).
    has_orientation = _has_column(conn, "files", "orientation")
    has_not_personal = _has_column(conn, "files", "not_personal")
    workers = resolve_workers(cfg.raw)

    def flush(pool: ThreadPoolExecutor):
        if not pending:
            return
        exif_map = read_batch([p for p, _ in pending])
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        # stat + blake3 — in the thread pool (I/O and hashing release the GIL); the
        # write to SQLite — only on the main thread (single-writer, one transaction per batch).
        results = list(pool.map(_hash_one, pending))
        with conn:  # one transaction per batch — Ctrl+C does not break consistency
            for r in results:
                path = str(r.path.resolve())
                if r.error is not None:
                    stats.errors += 1
                    conn.execute(
                        """INSERT INTO files (path, size, mtime, ext, media_type, error, indexed_at)
                           VALUES (?,?,?,?,?,?,?)
                           ON CONFLICT(path) DO UPDATE SET error=excluded.error,
                               indexed_at=excluded.indexed_at""",
                        (path, r.size, r.mtime, r.path.suffix.lower().lstrip("."), "photo",
                         r.error, now),
                    )
                    continue
                try:
                    ex = exif_map.get(path)
                    ta = resolve_taken_at(
                        ex.datetime_original if ex else None, r.path.name, r.mtime,
                        cfg.dates.min_year, cfg.dates.max_year,
                    )
                    mtype = cfg.index.media_type_of(r.path.suffix) or "photo"
                    conn.execute(
                        """INSERT INTO files (path, size, mtime, ext, media_type, hash, hash_algo,
                               phash, taken_at, taken_at_source, taken_at_confidence,
                               gps_lat, gps_lon, camera_make, camera_model, width, height,
                               dup_of, error, indexed_at)
                           VALUES (?,?,?,?,?,?,?,NULL,?,?,?,?,?,?,?,?,?,NULL,NULL,?)
                           ON CONFLICT(path) DO UPDATE SET
                               size=excluded.size, mtime=excluded.mtime, hash=excluded.hash,
                               hash_algo=excluded.hash_algo,
                               taken_at=excluded.taken_at, taken_at_source=excluded.taken_at_source,
                               taken_at_confidence=excluded.taken_at_confidence,
                               gps_lat=excluded.gps_lat, gps_lon=excluded.gps_lon,
                               camera_make=excluded.camera_make, camera_model=excluded.camera_model,
                               width=excluded.width, height=excluded.height,
                               -- phash is invalidated only when the content changes (different hash);
                               -- on an mtime-only reindex it is kept (no needless recompute)
                               phash=CASE WHEN hash=excluded.hash THEN phash ELSE NULL END,
                               dup_of=NULL, error=NULL, indexed_at=excluded.indexed_at""",
                        (path, r.size, r.mtime, r.path.suffix.lower().lstrip("."), mtype,
                         r.hash, r.algo, ta.dt.isoformat(timespec="seconds"), ta.source,
                         ta.confidence, ex.gps_lat if ex else None, ex.gps_lon if ex else None,
                         ex.make if ex else None, ex.model if ex else None,
                         ex.width if ex else None, ex.height if ex else None, now),
                    )
                    # phash is computed by compute_phashes() (F11): INSERT — NULL; UPDATE —
                    # kept when the hash is unchanged, otherwise NULL (recomputed).
                    if has_orientation:
                        conn.execute("UPDATE files SET orientation = ? WHERE path = ?",
                                     (ex.orientation if ex else None, path))
                    if has_not_personal:
                        not_personal = mtype == "video" and is_not_personal_video(
                            r.path.name, r.size)
                        conn.execute("UPDATE files SET not_personal = ? WHERE path = ?",
                                     (int(not_personal), path))
                    stats.added += r.action == "add"
                    stats.updated += r.action == "update"
                except Exception as e:  # a corrupt file does not crash the process
                    stats.errors += 1
                    conn.execute(
                        """INSERT INTO files (path, size, mtime, ext, media_type, error, indexed_at)
                           VALUES (?,?,?,?,?,?,?)
                           ON CONFLICT(path) DO UPDATE SET error=excluded.error,
                               indexed_at=excluded.indexed_at""",
                        (path, r.size, r.mtime, r.path.suffix.lower().lstrip("."), "photo",
                         f"{type(e).__name__}: {e}", now),
                    )
        pending.clear()
        if progress:
            progress(stats)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for p in _walk(cfg):
            stats.scanned += 1
            st = p.stat()
            action = _needs_update(conn, str(p.resolve()), st.st_size, st.st_mtime)
            if action is None:
                stats.skipped += 1
                continue
            pending.append((p, action))
            if len(pending) >= _BATCH:
                flush(pool)
        flush(pool)
    return stats
