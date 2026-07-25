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
from .exif import ExifData, read_batch, resolve_exif_workers
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


@dataclass
class RefreshStats:
    """Result of refresh_exif — deliberately measurable, not "it got better"."""
    scanned: int = 0          # rows the selection picked up
    updated: int = 0          # rows where at least one metadata column actually changed
    recovered_gps: int = 0    # rows that had no GPS and got coordinates
    recovered_date: int = 0   # rows whose taken_at now comes from EXIF (it did not before)
    still_empty: int = 0      # rows that really have no EXIF (png/wallpapers/downloads)
    errors: int = 0           # vanished/unreadable files — counted, never fatal


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
    exif_workers = resolve_exif_workers(cfg.raw)

    def flush(pool: ThreadPoolExecutor):
        if not pending:
            return
        # stat + blake3 — in the thread pool (I/O and hashing release the GIL); the
        # write to SQLite — only on the main thread (single-writer, one transaction per batch).
        # pool.map queues the work and returns a lazy iterator right away, so exiftool
        # (separate processes) runs alongside the hashing and the batch costs the longer
        # of the two phases instead of their sum (F72).
        results_it = pool.map(_hash_one, pending)
        exif_map = read_batch([p for p, _ in pending], exif_workers)
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        results = list(results_it)
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


# Metadata columns refresh_exif is allowed to rewrite. Everything derived from the file
# CONTENT (hash, hash_algo, phash, dup_of, size, mtime, not_personal, indexed_at) is out
# of scope: the content did not change, only what we managed to read out of it.
_REFRESH_COLUMNS = [
    "gps_lat", "gps_lon", "camera_make", "camera_model", "width", "height",
    "taken_at", "taken_at_source", "taken_at_confidence",
]


def refresh_exif(cfg: Config, conn: sqlite3.Connection, *, only_missing: bool = True,
                 progress: Callable[[int, int], None] | None = None) -> RefreshStats:
    """F71: re-read metadata of already-indexed files without reindexing them.

    A plain `index()` run will not touch these files: `_needs_update` compares
    path+size+mtime, and none of them changed — only the exiftool flag did (`-fast2`
    silently dropped the whole metadata block of most HEIC files, see exif._QUERY_ARGS).

    `only_missing=True` picks rows with no EXIF trace at all (camera_make, gps_lat and
    width all NULL). The criterion is deliberately wide: re-reading a png screenshot
    costs ~8 ms and breaks nothing, while missing one file with GPS is exactly the cost
    this feature exists to undo. `only_missing=False` re-reads everything (for the next
    time the flag changes).

    Only metadata is written; taken_at is recomputed through the same `resolve_taken_at`
    with the same cfg.dates bounds as `index()` — the two paths must not disagree on
    dates. Files are never read whole, no hashing happens.
    """
    stats = RefreshStats()
    has_orientation = _has_column(conn, "files", "orientation")
    columns = _REFRESH_COLUMNS + (["orientation"] if has_orientation else [])
    # media types exiftool is asked about — the same set the indexer accepts
    media_types = tuple(cfg.index.extensions)
    where = ["error IS NULL", f"media_type IN ({','.join('?' * len(media_types))})"]
    if only_missing:
        where.append("camera_make IS NULL AND gps_lat IS NULL AND width IS NULL")
    rows = conn.execute(
        f"SELECT id, path, mtime, {', '.join(columns)} FROM files"
        f" WHERE {' AND '.join(where)} ORDER BY id",
        media_types,
    ).fetchall()

    total = len(rows)
    exif_workers = resolve_exif_workers(cfg.raw)
    for start in range(0, total, _BATCH):
        batch = rows[start:start + _BATCH]
        paths = [Path(r["path"]) for r in batch]
        exif_map = read_batch(paths, exif_workers)
        with conn:  # one transaction per batch — Ctrl+C does not break consistency
            for row, path in zip(batch, paths):
                stats.scanned += 1
                try:
                    _refresh_row(conn, cfg, row, path, exif_map, columns, stats)
                except Exception:  # a broken row never takes the whole operation down
                    stats.errors += 1
        if progress:
            progress(min(start + _BATCH, total), total)
    return stats


def _refresh_row(conn: sqlite3.Connection, cfg: Config, row: sqlite3.Row, path: Path,
                 exif_map: dict[str, ExifData], columns: list[str],
                 stats: RefreshStats) -> None:
    ex = exif_map.get(str(path.resolve()))
    if ex is None or ex == ExifData():
        # Nothing was read: either the file has no EXIF at all, or it is gone/corrupt.
        # The stat only happens on this branch, so healthy files do not pay for it.
        try:
            missing = not path.exists()
        except OSError:
            missing = True
        if missing:
            stats.errors += 1
            return
        ex = ExifData()
    ta = resolve_taken_at(ex.datetime_original, path.name, row["mtime"],
                          cfg.dates.min_year, cfg.dates.max_year)
    new: dict[str, object] = {
        "gps_lat": ex.gps_lat, "gps_lon": ex.gps_lon,
        "camera_make": ex.make, "camera_model": ex.model,
        "width": ex.width, "height": ex.height,
        "taken_at": ta.dt.isoformat(timespec="seconds"),
        "taken_at_source": ta.source, "taken_at_confidence": ta.confidence,
        "orientation": ex.orientation,
    }
    changed = {c: new[c] for c in columns if row[c] != new[c]}
    if changed:
        conn.execute(
            f"UPDATE files SET {', '.join(f'{c} = ?' for c in changed)} WHERE id = ?",
            (*changed.values(), row["id"]),
        )
        stats.updated += 1
    if row["gps_lat"] is None and ex.gps_lat is not None:
        stats.recovered_gps += 1
    if ta.source == "exif" and row["taken_at_source"] != "exif":
        stats.recovered_date += 1
    # the same criterion the selection uses: such a row will be picked up again on a
    # repeated run (the selection deliberately does not remember "already tried")
    if ex.gps_lat is None and ex.make is None and ex.width is None:
        stats.still_empty += 1
