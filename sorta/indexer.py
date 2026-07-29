"""FR-1: scanning, metadata, hashes, incrementality.

Invariant: original files are never modified.
A re-run skips files with matching path+size+mtime.

F81: source folders can be excluded BEFORE the walk reaches them (`excludes.yaml`,
keyed by source root — see `load_excludes`). An excluded subtree is never entered, so
its files cost no stat, no hash and no later stage; rows already indexed under such a
path are deleted from the index at the start of `index()`, because "do not scan" and
"is in the index" cannot both be true.

F82: the same file also carries the OTHER kind of exclusion — "do not lay out"
(`skip_layout`), which the indexer never looks at: those files are scanned and indexed
as usual, `sorter._resolve_excludes` is what leaves them where they lie. Two sections
of one file rather than two files, so a folder can be moved between the two meanings
without moving between formats.
"""
from __future__ import annotations

import logging
import os
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Iterator

import yaml

from .config import Config
from .dates import resolve_taken_at
from .exif import ExifData, read_batch, resolve_exif_workers
from .hashing import file_hash, resolve_workers

_log = logging.getLogger(__name__)

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
    # F81: what the walk refused to enter, and what it evicted because of that.
    excluded_dirs: int = 0      # pruned subtrees
    excluded_files: int = 0     # files inside them (counted from directory entries only)
    removed_excluded: int = 0   # already-indexed rows deleted because they now sit under an exclusion


# --- F81/F82: excluded source folders --------------------------------------------

_EXCLUDES_FILENAME = "excludes.yaml"

# The two sections of a root's entry. `skip_scan` is F81 ("never entered by the walk"),
# `skip_layout` is F82 ("indexed as usual, but never laid out" — read by sorter.py).
_SKIP_SCAN = "skip_scan"
_SKIP_LAYOUT = "skip_layout"

# Tables that reference files(id). A file leaving the index must not leave a row
# behind in any of them. The move journal (moves/move_batches) is deliberately NOT
# here: it is the history of operations that really happened, not index state.
_DEPENDENT_TABLES = (
    "places", "media_class", "faces", "event_files", "dedup_choice", "manual_overrides",
    "manual_places", "frame_quality",
)


@dataclass
class Excludes:
    """Excluded directories, keyed by SOURCE ROOT, in two independent meanings.

    An exclusion is meaningless outside its root ("Movies" belongs to D:/Photos, not
    to the world), so the file groups the relative paths per root. Changing the source
    therefore needs no migration question: the new root has its own set, the old one
    keeps its own, and coming back restores it.

    `by_root` is "do not scan" (F81): the walk never enters it, its files are not in the
    index at all. `layout_by_root` is "do not lay out" (F82): the files ARE indexed —
    they take part in dedup, statistics and the web app — they are only left where they
    lie by `sorter`. The two are mutually exclusive per folder; `load_excludes` resolves
    a hand-written overlap in favour of "do not scan", the stronger of the two.
    """
    by_root: dict[str, list[str]] = field(default_factory=dict)  # normalized root -> rel paths
    layout_by_root: dict[str, list[str]] = field(default_factory=dict)

    def for_root(self, root: str | Path) -> frozenset[str]:
        return frozenset(self.by_root.get(_norm_root(root), ()))

    def layout_for_root(self, root: str | Path) -> frozenset[str]:
        return frozenset(self.layout_by_root.get(_norm_root(root), ()))

    def __bool__(self) -> bool:
        return any(self.by_root.values()) or any(self.layout_by_root.values())


def excludes_path(cfg: Config) -> Path:
    """Location of the exclusion file.

    `index.excludes_file` is read straight from `cfg.raw` — the same arrangement as
    `index.workers` in `hashing.resolve_workers`: no typed field is added to
    config.py for it. Default: `excludes.yaml` next to the database file.
    """
    idx = (cfg.raw or {}).get("index")
    value = idx.get("excludes_file") if isinstance(idx, dict) else None
    if isinstance(value, str) and value.strip():
        return Path(value.strip()).expanduser()
    return Path(cfg.database).expanduser().resolve().parent / _EXCLUDES_FILENAME


def _norm_root(root: str | Path) -> str:
    """Lookup key of a source root: resolved + normcase.

    The same root reaches us from config.yaml, from the CLI and from the web app, in
    whatever spelling the user typed; on Windows it also differs in case and in the
    separator. One canonical form is what makes those the same key.
    """
    try:
        resolved = Path(root).expanduser().resolve()
    except OSError:  # pragma: no cover — resolve() barely ever raises with strict=False
        resolved = Path(root)
    return os.path.normcase(str(resolved))


def _display_root(root: str | Path) -> str:
    """How a root is written INTO the file: absolute, POSIX separators (`D:/Photos`) —
    the file is machine-written but has to stay readable."""
    return Path(root).expanduser().resolve().as_posix()


def normalize_exclude(value: object) -> str | None:
    """One list entry -> a relative POSIX path, or None if it is rejected.

    The value comes from a file the web app writes, i.e. from OUTSIDE the program (the
    same class of risk as `manual_overrides.target` in F77). An exclusion may only
    NARROW the walk, never move it elsewhere, so anything that could point out of the
    root is refused: a non-string, an empty value, a backslash or a colon (`..\\x`,
    `C:/windows`, UNC), a leading `/` (`/etc`), and any `..` segment.
    """
    if not isinstance(value, str):
        return None
    raw = value.strip()
    if (not raw or "\\" in raw or ":" in raw or raw.startswith("/")
            or ".." in [seg.strip() for seg in raw.split("/")]):
        return None
    parts = [seg.strip() for seg in raw.split("/")]
    parts = [seg for seg in parts if seg and seg != "."]
    if not parts:
        return None
    return "/".join(parts)


def _read_excludes_file(path: Path) -> dict:
    """The raw mapping from disk. A missing file is not an error; a broken one is a
    warning and an empty result — losing a whole run over a damaged settings file is
    not an acceptable trade."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        _log.warning("index: файл исключений %s не прочитан (%s) — работаем без исключений",
                     path, exc)
        return {}
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        _log.warning("index: файл исключений %s испорчен (%s) — работаем без исключений",
                     path, exc)
        return {}
    if data is None:
        return {}
    if not isinstance(data, dict):
        _log.warning("index: файл исключений %s имеет неожиданную структуру (%s вместо "
                     "словаря по корням) — работаем без исключений", path, type(data).__name__)
        return {}
    return data


def _root_sections(value: object, root: str, path: Path) -> tuple[list, list]:
    """One root's raw value -> its (skip_scan, skip_layout) entries, still unvalidated.

    A plain LIST is the F81 spelling of the file and means `skip_scan`. It has to keep
    working: the file on the user's disk was written before `skip_layout` existed and is
    the only record of what they already excluded, so reading it as anything else (or
    not at all) would silently throw that away.
    """
    if isinstance(value, list):
        return value, []
    if not isinstance(value, dict):
        _log.warning("index: значение для корня %s в %s не список и не разделы "
                     "%s/%s — пропущено", root, path, _SKIP_SCAN, _SKIP_LAYOUT)
        return [], []
    sections: list[list] = []
    for key in (_SKIP_SCAN, _SKIP_LAYOUT):
        section = value.get(key, [])
        if section is None:
            section = []
        if not isinstance(section, list):
            _log.warning("index: раздел %s корня %s в %s не список — пропущен",
                         key, root, path)
            section = []
        sections.append(section)
    return sections[0], sections[1]


def _accept_all(values: Iterable[object], root: str) -> list[str]:
    """Validated, de-duplicated relative paths; rejected entries are logged, not fatal."""
    accepted: list[str] = []
    for value in values:
        rel = normalize_exclude(value)
        if rel is None:
            _log.warning("index: исключение %r для корня %s отклонено — оно уводит "
                         "обход за пределы корня", value, root)
            continue
        if rel not in accepted:
            accepted.append(rel)
    return accepted


def load_excludes(path: str | Path) -> Excludes:
    """Read the exclusion file (§1 of F81/F82) into a root-keyed, validated set."""
    p = Path(path)
    by_root: dict[str, list[str]] = {}
    layout_by_root: dict[str, list[str]] = {}
    for root, value in _read_excludes_file(p).items():
        if not isinstance(root, str) or not root.strip():
            _log.warning("index: ключ %r в %s не похож на корень источника — пропущен", root, p)
            continue
        raw_scan, raw_layout = _root_sections(value, root, p)
        key = _norm_root(root)
        scan = by_root.setdefault(key, [])
        scan += [rel for rel in _accept_all(raw_scan, root) if rel not in scan]
        layout = layout_by_root.setdefault(key, [])
        # a folder listed in both sections is not scanned: the file may be hand-edited,
        # and "not in the index" cannot be reconciled with "laid out, only differently"
        layout += [rel for rel in _accept_all(raw_layout, root)
                   if rel not in layout and rel not in scan]
    return Excludes(by_root, layout_by_root)


def save_excludes(path: str | Path, root: str | Path, values: Iterable[object],
                  layout: Iterable[object] | None = None) -> list[str]:
    """Write the exclusions of ONE root, preserving every other root's entry.

    `values` is the "do not scan" section, `layout` the "do not lay out" one;
    `layout=None` leaves whatever the file already had there (that is what
    `sorta index --exclude-dir` does — it has no opinion about the other section).
    Returns the accepted (normalized) "do not scan" list; entries `normalize_exclude`
    rejects are dropped. The write is atomic — a temp file next to the target +
    `os.replace`, like `imaging._write_preview`: the file is read by a run that may
    start at any moment, and nobody may ever observe half of it.
    """
    p = Path(path)
    data = _read_excludes_file(p)
    display = _display_root(root)
    if layout is None:
        layout = load_excludes(p).layout_for_root(root)
    # drop whatever spelling of this root the file already had — one root, one key
    kept = {k: v for k, v in data.items()
            if not (isinstance(k, str) and _norm_root(k) == _norm_root(root))}
    accepted = _accept_all(values, display)
    # the two states exclude each other (§2): "do not scan" wins, so ticking it clears
    # a "do not lay out" left on the same folder instead of writing a contradiction
    accepted_layout = [rel for rel in _accept_all(layout, display) if rel not in accepted]
    sections: dict[str, list[str]] = {}
    if accepted:
        sections[_SKIP_SCAN] = sorted(accepted)
    if accepted_layout:
        sections[_SKIP_LAYOUT] = sorted(accepted_layout)
    if sections:
        kept[display] = sections
    text = yaml.safe_dump(kept, allow_unicode=True, sort_keys=True, default_flow_style=False)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, p)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    return sorted(accepted)


def _excluded_prefixes(cfg: Config, excludes: Excludes) -> list[str]:
    """Absolute, normcase'd paths of every excluded subtree of every source."""
    prefixes: list[str] = []
    for src in cfg.sources:
        root = Path(src).expanduser().resolve()
        for rel in sorted(excludes.for_root(src)):
            prefixes.append(os.path.normcase(str(root.joinpath(*rel.split("/")))))
    return prefixes


def _under_any(path: str, prefixes: list[str]) -> bool:
    return any(path == prefix or path.startswith(prefix + os.sep) for prefix in prefixes)


def _existing_tables(conn: sqlite3.Connection) -> set[str]:
    return {r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'")}


def drop_excluded_rows(cfg: Config, conn: sqlite3.Connection, excludes: Excludes) -> int:
    """Delete indexed rows that now sit under an exclusion, with their dependents.

    "Do not scan" means "not in the index": leaving the 847 rows of a folder the user
    just excluded would make the state contradict the setting. Dependent rows go with
    them (see `_DEPENDENT_TABLES`), and `files.dup_of` references to a deleted row are
    cleared — a surviving file must not point at an id that no longer exists.
    """
    prefixes = _excluded_prefixes(cfg, excludes)
    if not prefixes:
        return 0
    doomed = [row["id"] for row in conn.execute("SELECT id, path FROM files")
              if _under_any(os.path.normcase(row["path"]), prefixes)]
    if not doomed:
        return 0
    tables = [t for t in _DEPENDENT_TABLES if t in _existing_tables(conn)]
    # `moves.file_id` also references files(id), but the move journal is history, not
    # index state, and must survive (§3) — with foreign keys enforced the DELETE below
    # would be refused because of it. So the constraint is lifted for this operation
    # only, and every table that IS index state is cleaned explicitly above. The
    # PRAGMA is a no-op inside a transaction, hence outside the `with`.
    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        with conn:  # one transaction — a Ctrl+C leaves no half-deleted file
            for start in range(0, len(doomed), _BATCH):
                chunk = doomed[start:start + _BATCH]
                ph = ",".join("?" * len(chunk))
                for table in tables:
                    conn.execute(f"DELETE FROM {table} WHERE file_id IN ({ph})", chunk)
                conn.execute(f"UPDATE files SET dup_of = NULL WHERE dup_of IN ({ph})", chunk)
                conn.execute(f"DELETE FROM files WHERE id IN ({ph})", chunk)
    finally:
        conn.execute("PRAGMA foreign_keys = ON")
    _log.info("index: удалено из индекса %d строк под исключёнными папками", len(doomed))
    return len(doomed)


@dataclass
class RefreshStats:
    """Result of refresh_exif — deliberately measurable, not "it got better"."""
    scanned: int = 0          # rows the selection picked up
    updated: int = 0          # rows where at least one metadata column actually changed
    recovered_gps: int = 0    # rows that had no GPS and got coordinates
    recovered_date: int = 0   # rows whose taken_at now comes from EXIF (it did not before)
    still_empty: int = 0      # rows that really have no EXIF (png/wallpapers/downloads)
    errors: int = 0           # vanished/unreadable files — counted, never fatal


def _walk(cfg: Config, excludes: Excludes | None = None,
          stats: IndexStats | None = None) -> Iterator[Path]:
    skip = set(cfg.index.skip_dirs)
    min_size = cfg.index.min_file_size_kb * 1024
    for src in cfg.sources:
        excluded = excludes.for_root(src) if excludes is not None else frozenset()
        for p in _walk_root(src, skip, excluded, stats):
            if cfg.index.media_type_of(p.suffix) is None:
                continue
            try:
                if p.stat().st_size < min_size:
                    continue
            except OSError:
                continue
            yield p


def _walk_root(src: Path, skip: set[str], excluded: frozenset[str],
               stats: IndexStats | None) -> Iterator[Path]:
    """Files under one source root, with excluded subtrees pruned.

    Replaces the previous `sorted(src.rglob("*"))`: rglob offers no way to stop before
    descending, and the whole point of F81 is that an excluded subtree is never
    entered. The name filter is unchanged — every component of the FULL path is still
    matched against skip_dirs / a leading dot, the components of the root included
    (hence the check on `src.parts` here).
    """
    if any(part in skip or part.startswith(".") for part in src.parts):
        return
    yield from _walk_dir(src, "", skip, excluded, stats)


def _walk_dir(directory: Path, prefix: str, skip: set[str], excluded: frozenset[str],
              stats: IndexStats | None) -> Iterator[Path]:
    try:
        with os.scandir(directory) as it:
            entries = sorted(it, key=lambda e: e.name)
    except OSError:  # unreadable directory — the run does not stop over one folder
        return
    for entry in entries:
        rel = f"{prefix}/{entry.name}" if prefix else entry.name
        if rel in excluded:
            # Excluded first, before is_file()/stat()/open(): from here on the subtree
            # costs nothing but the directory entries that count it.
            _count_excluded(entry, stats)
            continue
        if entry.name in skip or entry.name.startswith("."):
            continue
        if entry.is_dir(follow_symlinks=False):
            yield from _walk_dir(Path(entry.path), rel, skip, excluded, stats)
        elif entry.is_file():
            yield Path(entry.path)


def _count_excluded(entry: os.DirEntry[str], stats: IndexStats | None) -> None:
    """Count what the walk refused to enter — without numbers the effect of an
    exclusion is not observable (§2).

    Only directory ENTRIES are read (names from `scandir`, `is_dir` off the entry the
    listing already carries): no `stat`, no `open`, no hashing, and no later stage
    ever sees these files. That is the cost this feature exists to remove; listing
    names is what makes the removal reportable.
    """
    if stats is None:
        return
    if not entry.is_dir(follow_symlinks=False):
        stats.excluded_files += 1
        return
    stats.excluded_dirs += 1
    stats.excluded_files += _count_entries(Path(entry.path))


def _count_entries(directory: Path) -> int:
    total = 0
    try:
        with os.scandir(directory) as it:
            entries = list(it)
    except OSError:
        return 0
    for entry in entries:
        if entry.is_dir(follow_symlinks=False):
            total += _count_entries(Path(entry.path))
        else:
            total += 1
    return total


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
    # F81: the exclusions are read here, not in cli.py — every entry point (CLI, web
    # app) goes through index() and must obey the same file.
    excludes = load_excludes(excludes_path(cfg))
    stats.removed_excluded = drop_excluded_rows(cfg, conn, excludes)
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
        for p in _walk(cfg, excludes, stats):
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
