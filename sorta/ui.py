"""U1/U3/U4/F31/F32/F35/F36: a local web server — a live sort-plan report +
Duplicates (incl. batch saving) + deleting a single frame + a "People" tab (managing
face clusters) + person/event albums ("Collect into folder", on top of the F34
engine) + the "Process" entry point — running the pipeline
index→geo→landmarks→faces→events→junk→phash from the web, on a background server thread.

Most routes are READ-ONLY (reading originals/decoding thumbnails by file_id from the
index). Writes go through six narrowly-scoped paths: (1) `dedup_choice` — the user's
decisions on near-duplicates (keep/to_delete), a soft mark, does not touch files;
this also includes the batch `POST /api/dupes/choices` (F32) — the same effect as
`_apply_choice`/`_skip_group` over many groups in one transaction (the whole body is
validated before the first write — an invalid item causes no partial write); (2)
`POST /api/dupes/trash` — the non-keeper frames of a group physically go to the OS
trash; (3) `POST /api/photo/trash` (U4) — one arbitrary frame (a Cities-leaf or a
Duplicates frame) to the trash. (2) and (3) use the same `_trash_files` — a single
trash path: `send2trash` (not permanent deletion) + DELETE of the `files`/`dedup_choice`
rows so the index does not diverge from the disk. Original files are otherwise not
modified. (4) `POST /api/clusters/label` and (5) `POST /api/clusters/merge` (F31) —
naming/merging face clusters via `faces.label_cluster`/`faces.merge` (the public
faces.py API, used read-only from a code-ownership standpoint; the functions
themselves write to `face_clusters`). Both accept only int ids from the JSON body,
never a path. (6) `POST /api/album` (F35) — exporting a person/event slice
(link/copy/move) via `sorter.plan_album` (a public API, used read-only from a
code-ownership standpoint; it writes `moves`/`move_batches`); the body accepts only
kind/selector/mode/where/name/apply — strings/ints/bool, never a path (the server
resolves dest itself from `cfg.sort.album_dir`, `plan_album` resolves file_id -> path
from the DB itself). `apply=False` — a preview (writes nothing), the client confirms
and re-sends with `apply=True`.

(7) `POST /api/process` (F36) — starts a background THREAD running the stages
index→geo→landmarks→faces→events→junk→phash (the leaf functions indexer/geo/
landmarks/faces/events/junk/dedup/naming — NOT imported from cli.py, to avoid a
cycle); the body accepts `source_dir: str` (required) + optional
`deep: bool`/`geo_online: bool` (F50/#34, default False) — which override
`cfg.sources`/`cfg.naming.vlm_enabled`/`cfg.geo.provider` ONLY in this run's cfg copy
(`dataclasses.replace`) — the shared cfg read by the other routes' handlers is not
mutated. The thread opens its own sqlite connection (not transferable between
ThreadingHTTPServer threads). One run per server — a repeated `POST` while running ->
409 (`_ProcessState.try_start` is atomic under a shared lock). `GET /api/process/status`
— a thread-safe progress snapshot (polling); `POST /api/process/cancel` sets a flag
checked BETWEEN stages (not mid-stage). The pipeline moves no files — it only reads
source_dir and writes the index, so the layout FS invariants (the moves.jsonl journal,
hash verification) do not apply here.

(8) `POST /api/process/reset` (F42, the "Start over" button) — wipes the ENTIRE index
via the ready `db.reset_index(conn)` (the same tables as the CLI `sorta reset`:
metadata, geo, faces/clusters with names, events with names, junk, dedup_choice,
moves). Blocked with 409 while `/api/process` is still `running` (the same
`_ProcessState.snapshot()`). Does not touch files on disk or already-sorted folders —
only the DB contents. PlanCache is recomputed with the same conn right after the reset
(an empty DB -> an empty plan, see PlanCache).

(9) `POST /api/sort` (F43, the "Cities" tab, the "Sort" button) — the real layout of
the collection: calls `sorter.plan_and_sort(cfg, conn, "city", dest, apply=True,
copy=..., progress=...)` on a background thread with its own sqlite connection (the
`_ProcessState`/`_run_pipeline` pattern, but its own `_SortState` — no stages, one
operation). The body `{"dest": str|null|"", "mode": "move"|"copy"}`: `dest` empty/null
-> in-place (restructuring the source tree, `dest=None` in `plan_and_sort`, F28);
`mode` outside {move, copy} -> 400. The `moves`/`move_batches` journal, blake3
verification and name-conflict resolution — entirely in `plan_and_sort`, ui.py does
not duplicate this logic. Cross-locking with `/api/process`: while a sort is running —
`POST /api/process` and `POST /api/process/reset` answer 409 (and vice versa); a
repeated `POST /api/sort` while sorting — 409 (`_SortState.try_start`). A `ValueError`
from `plan_and_sort` (e.g. in-place with multiple `cfg.sources`) is caught and stored
in the state as an error, without crashing the thread/server. `GET /api/sort/status` —
a snapshot for polling. After a successful apply — `PlanCache.rebuild` with the same
conn (the city plan reads the new paths); the "Moves" tab learns about it from a reset
of `movesLoaded` in JS.

(10) `POST /api/browse` (F51, the "Browse…" button — next to the "Process" path field
and next to the layout destination field on the "Cities" tab) — opens a native
folder-picker dialog and returns `{"path": str}` (an empty string on cancel/error/no
GUI — not a 500, the button is just a convenience, manual path entry always works).
The dialog — tkinter `askdirectory` in a SEPARATE subprocess (`_browse_for_folder`,
`subprocess.run([sys.executable, "-c", ...])`): tkinter is not thread-safe, and the
POST handler runs on a ThreadingHTTPServer thread, not the process's main thread; a
fresh process = its own main thread, without a conflict with the server. The returned
path is not processed at all on the server — `POST /api/process` already validates
`source_dir` as an existing directory (no extra checks needed: the path is chosen by
the user in a native dialog on their own machine, there is no injection).

(11) `GET /api/sort/suggest-dest` — the default destination path for the city layout:
`{"dest": "<source>_sorted"}` (the source — `cfg.sources[0]` or the common root of the
indexed files; see `_suggested_sort_dest`). JS prefills the `#sort-dest` field only if
the user has not entered anything yet.

Security: the only entry to a file on disk for reading (`/thumb`, `/photo`) is a
file_id, resolved strictly via `SELECT path FROM files WHERE id = ?`. These routes
never accept a path directly from the request, so an arbitrary path (incl. `../..`)
does not resolve — a non-numeric/unknown id simply finds no row in files and answers
404. The write endpoints (`POST /api/dupes/*`, `POST /api/photo/trash`) also operate
only on a file_id from the JSON body (no paths from the client); before deleting a
`files` row or sending a path to the trash, the id is resolved by the same query
`SELECT ... FROM files WHERE id IN (...)` — unknown ids are silently ignored, not
substituted as a path. The server binds only to 127.0.0.1.

plan_and_sort (sorter, dry-run) — the single source of the plan; besides report.plan
it, as a side effect, writes CSV/HTML into report_output/ (a plan-only entry point
without side files is possible later). PlanCache calls it at most once per mode over
the server process's lifetime.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import io
import json
import logging
import mimetypes
import os
import sqlite3
import subprocess
import sys
import threading
import webbrowser
from collections import OrderedDict, defaultdict
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, urlsplit

from send2trash import send2trash as send_to_trash

from . import db, faces, i18n, imaging
from .config import Config, save_language
from .dedup import assign_duplicates, compute_phashes, near_duplicate_groups
from .events import build_events
from .faces import detect_and_cluster
from .geo import resolve_places
from .indexer import index as run_index
from .junk import classify as classify_junk
from .landmarks import Classifier, clip_classifier, detect_landmarks
from .naming import name_events, naming_settings
from .sorter import ALBUM_KINDS, ALBUM_MODES, AlbumReport, PlanItem, plan_album, plan_and_sort

_log = logging.getLogger(__name__)

DEFAULT_PORT = 8756
_THUMB_MAX_EDGE = 200
_CLUSTER_SAMPLE_LIMIT = 6
_EVENT_SAMPLE_LIMIT = 8
_SUPPORTED_MODES = ("city", "person", "event")
_DEFAULT_ALBUM_DIRNAME = "_Альбомы"

# F39: UI switcher languages — the same three as i18n.Lang; self-names for the
# selector options (not translated — this is a language's name in that language).
_UI_LANGS: tuple[str, ...] = ("ru", "en", "ja")
_LANG_SELF_NAMES: dict[str, str] = {"ru": "Русский", "en": "English", "ja": "日本語"}

_ProgressCB = Callable[[int, "int | None"], None]  # (done, total|None) — compatible with progress.ProgressCB


def _plan_item_to_json(item: PlanItem) -> dict:
    geo = "/".join(p for p in (item.country, item.city) if p) or None
    return {
        "file_id": item.file_id,
        "name": item.src.name,
        "target_rel": item.target_rel,
        "reason": item.reason,
        "date": item.taken_at,
        "geo": geo,
        "category": item.reason,
        "thumb_url": f"/thumb/{item.file_id}",
    }


class PlanCache:
    """An in-memory cache of report.plan by mode, built at server startup and
    recomputed explicitly (`rebuild`) — on an empty DB right away and again after
    `/api/process` (F36) finishes, and NOT on every external DB update.

    sqlite3 connections are not transferable between threads (`check_same_thread`),
    and ThreadingHTTPServer serves each request on a new thread — so the plan is
    built right here, on the caller's thread (at startup — before serve_forever, on
    rebuild — on the background pipeline thread), using the passed cfg/conn;
    subsequent requests read the ready list from memory without touching the DB,
    until someone explicitly calls rebuild.
    """

    def __init__(self, cfg: Config, conn: sqlite3.Connection, dest: Path) -> None:
        self._dest = dest
        self._by_mode: dict[str, list[PlanItem]] = {}
        self.rebuild(cfg, conn)

    def rebuild(self, cfg: Config, conn: sqlite3.Connection) -> None:
        by_mode: dict[str, list[PlanItem]] = {}
        for mode in _SUPPORTED_MODES:
            report = plan_and_sort(cfg, conn, mode, self._dest, apply=False)
            by_mode[mode] = report.plan
        self._by_mode = by_mode

    def get(self, mode: str) -> list[PlanItem] | None:
        """The list of PlanItem for a mode, or None for an unsupported mode."""
        return self._by_mode.get(mode)


def _resolve_path(db_path: Path, file_id: int) -> Path | None:
    """The only legitimate way to reach a file on disk — by id from files.

    Opens a short-lived connection per call: ThreadingHTTPServer request handlers
    each run on their own thread, and an sqlite3 connection from another (calling)
    thread must not be passed here (see PlanCache).
    """
    conn = sqlite3.connect(str(db_path))
    try:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT path FROM files WHERE id = ?", (file_id,)).fetchone()
    finally:
        conn.close()
    return Path(row["path"]) if row is not None else None


def _parse_file_id(raw: str) -> int | None:
    try:
        return int(raw)
    except ValueError:
        return None


# F42: the People tab renders ~48 cluster cards at once (with
# _CLUSTER_SAMPLE_LIMIT previews each) -> ~288 concurrent GET /thumb/<id>.
# ThreadingHTTPServer spawns a thread per request — without a cache each request
# re-runs decode_rgb + JPEG-encode, hundreds of parallel decodes saturate the CPU,
# the server stops responding. Two independent measures:
# (1) _thumb_cache — an LRU of ready JPEG bytes by (file_id, mtime): a repeated/
#     concurrent request for the same frame never reaches imaging at all;
# (2) _thumb_decode_semaphore — limits the number of decode+encode running
#     CONCURRENTLY (not the total number of requests) — while the cache warms up,
#     a request spike does not spawn hundreds of CPU-heavy decodes at once.
_THUMB_CACHE_MAX_ITEMS = 512
_THUMB_DECODE_CONCURRENCY = max(2, min(8, os.cpu_count() or 4))
# Lightbox (F42/follow-up): a large DECODED JPEG instead of the raw original
# (`/photo`) — the browser cannot do HEIC/RAW, but decode_rgb can. Frames are viewed
# one at a time, so the cache is smaller than the thumbnail one; the edge is larger.
_PREVIEW_MAX_EDGE = 1600
_PREVIEW_CACHE_MAX_ITEMS = 64

_ImgCacheKey = tuple[int, float]
_ThumbCacheKey = _ImgCacheKey  # name backward-compatibility
_thumb_cache: OrderedDict[_ImgCacheKey, bytes] = OrderedDict()
_thumb_cache_lock = threading.Lock()
_preview_cache: OrderedDict[_ImgCacheKey, bytes] = OrderedDict()
_preview_cache_lock = threading.Lock()
# a shared semaphore: limits the TOTAL number of concurrent decode+encode (thumb and
# preview together), so a request spike does not spawn hundreds of CPU-heavy decodes.
_thumb_decode_semaphore = threading.Semaphore(_THUMB_DECODE_CONCURRENCY)


def _thumb_cache_clear() -> None:
    """Clear the in-process caches of decoded images (thumbnails + previews).
    Tests — isolation between cases; a DB reset — so a frame of a wiped id is not
    served (the mtime key almost rules out a collision anyway, but we clear for rigor)."""
    with _thumb_cache_lock:
        _thumb_cache.clear()
    with _preview_cache_lock:
        _preview_cache.clear()


def _encode_jpeg_cached(
    file_id: int, path: Path, *, max_edge: int, quality: int,
    cache: OrderedDict[_ImgCacheKey, bytes], cache_lock: threading.Lock,
    cache_max: int,
) -> bytes | None:
    """Ready JPEG bytes of a frame (decoded to max_edge), from cache or by decoding.

    The key (file_id, mtime) — a change of mtime naturally invalidates the entry.
    A cache miss is rechecked AFTER acquiring the semaphore (another thread may have
    decoded and cached the same key while the current one waited in the queue) —
    avoids a needless re-decode under a request spike for one frame.
    """
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    key: _ImgCacheKey = (file_id, mtime)
    with cache_lock:
        cached = cache.get(key)
        if cached is not None:
            cache.move_to_end(key)
            return cached

    with _thumb_decode_semaphore:
        with cache_lock:
            cached = cache.get(key)
            if cached is not None:
                cache.move_to_end(key)
                return cached
        img = imaging.decode_rgb(path, max_edge=max_edge)
        if img is None:
            return None
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=quality)
        data = buf.getvalue()

    with cache_lock:
        cache[key] = data
        cache.move_to_end(key)
        while len(cache) > cache_max:
            cache.popitem(last=False)
    return data


def _thumb_bytes(file_id: int, path: Path) -> bytes | None:
    """Ready JPEG thumbnail bytes for file_id (the _thumb_cache cache, F42)."""
    return _encode_jpeg_cached(
        file_id, path, max_edge=_THUMB_MAX_EDGE, quality=85,
        cache=_thumb_cache, cache_lock=_thumb_cache_lock,
        cache_max=_THUMB_CACHE_MAX_ITEMS)


def _preview_bytes(file_id: int, path: Path) -> bytes | None:
    """A large decoded JPEG for the lightbox (HEIC/RAW are rendered too)."""
    return _encode_jpeg_cached(
        file_id, path, max_edge=_PREVIEW_MAX_EDGE, quality=88,
        cache=_preview_cache, cache_lock=_preview_cache_lock,
        cache_max=_PREVIEW_CACHE_MAX_ITEMS)


def _connect(db_path: Path) -> sqlite3.Connection:
    """A short-lived per-call connection (see _resolve_path — the same reason:
    sqlite3 connections are not transferable between ThreadingHTTPServer threads)."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _dupes_payload(db_path: Path, max_distance: int) -> list[dict]:
    """near_duplicate_groups -> JSON-compatible groups for the Duplicates tab.

    recommended (F14): the best frame of the group by (width*height, then size) desc.
    action — the current decision from dedup_choice (keep/to_delete/None).
    """
    conn = _connect(db_path)
    try:
        groups = near_duplicate_groups(conn, max_distance=max_distance)
        if not groups:
            return []
        all_ids = [r["id"] for g in groups for r in g]
        placeholders = ",".join("?" * len(all_ids))
        wh = {
            r["id"]: (r["width"], r["height"])
            for r in conn.execute(
                f"SELECT id, width, height FROM files WHERE id IN ({placeholders})",
                all_ids,
            ).fetchall()
        }
        choices = {
            r["file_id"]: r["action"]
            for r in conn.execute(
                f"SELECT file_id, action FROM dedup_choice WHERE file_id IN ({placeholders})",
                all_ids,
            ).fetchall()
        }
    finally:
        conn.close()

    result = []
    for idx, group in enumerate(groups):
        frames = []
        for r in group:
            w, h = wh.get(r["id"], (None, None))
            frames.append({
                "file_id": r["id"],
                "name": Path(r["path"]).name,
                "thumb_url": f"/thumb/{r['id']}",
                "width": w,
                "height": h,
                "size": r["size"],
                "action": choices.get(r["id"]),
                "recommended": False,
            })
        best = min(
            frames,
            key=lambda f: (
                -((f["width"] or 0) * (f["height"] or 0)),
                -(f["size"] or 0),
                f["file_id"],
            ),
        )
        best["recommended"] = True
        result.append({"group": idx, "frames": frames})
    return result


def _validate_group_payload(payload: object) -> tuple[list[int], int | None] | None:
    """Parse the body `{"group": [file_id,...], "keep_file_id": int?}`.

    None -> the body is invalid (not a JSON object / group is not a non-empty list of
    int / keep_file_id, if present, is not int). keep_file_id may be absent (skip).
    """
    if not isinstance(payload, dict):
        return None
    group = payload.get("group")
    if (not isinstance(group, list) or not group
            or not all(isinstance(x, int) and not isinstance(x, bool) for x in group)):
        return None
    keep = payload.get("keep_file_id")
    if keep is not None and (not isinstance(keep, int) or isinstance(keep, bool)):
        return None
    return group, keep


def _apply_choice(db_path: Path, group: list[int], keep_file_id: int) -> None:
    """keeper -> action='keep', the other frames of the group -> 'to_delete'.

    Idempotent: ON CONFLICT overwrites the old decision (e.g. when moving the keeper
    to another frame of the same group).
    """
    now = datetime.now(timezone.utc).isoformat()
    conn = _connect(db_path)
    try:
        with conn:
            for fid in group:
                action = "keep" if fid == keep_file_id else "to_delete"
                conn.execute(
                    """INSERT INTO dedup_choice (file_id, action, updated_at)
                       VALUES (?, ?, ?)
                       ON CONFLICT(file_id) DO UPDATE SET
                           action = excluded.action, updated_at = excluded.updated_at""",
                    (fid, action, now),
                )
    finally:
        conn.close()


def _skip_group(db_path: Path, group: list[int]) -> None:
    """"Do not delete this group" — clears dedup_choice of the group's frames."""
    conn = _connect(db_path)
    try:
        placeholders = ",".join("?" * len(group))
        with conn:
            conn.execute(
                f"DELETE FROM dedup_choice WHERE file_id IN ({placeholders})", group
            )
    finally:
        conn.close()


def _validate_batch_choices_payload(
    payload: object,
) -> tuple[list[tuple[list[int], int]], list[list[int]]] | None:
    """Parse the body `{"groups": [{"group": [...], "keep_file_id": int}, ...],
    "skip": [[file_id,...], ...]}`. `skip` is optional (default []).

    None -> the body is invalid: `groups` is not a non-empty list / any entry does not
    pass `_validate_group_payload` or its `keep_file_id` is absent/not in `group` /
    `skip` is not a list of lists of int. The whole body is validated, before any DB
    write (F32: atomicity — 400 without a partial write).
    """
    if not isinstance(payload, dict):
        return None
    raw_groups = payload.get("groups")
    if not isinstance(raw_groups, list) or not raw_groups:
        return None
    groups: list[tuple[list[int], int]] = []
    for entry in raw_groups:
        parsed = _validate_group_payload(entry)
        if parsed is None:
            return None
        group, keep = parsed
        if keep is None or keep not in group:
            return None
        groups.append((group, keep))
    raw_skip = payload.get("skip", [])
    if not isinstance(raw_skip, list):
        return None
    skip: list[list[int]] = []
    for entry in raw_skip:
        if (not isinstance(entry, list) or not entry
                or not all(isinstance(x, int) and not isinstance(x, bool) for x in entry)):
            return None
        skip.append(entry)
    return groups, skip


def _apply_batch_choices(
    db_path: Path, groups: list[tuple[list[int], int]], skip: list[list[int]]
) -> int:
    """Apply the keeper choice over all groups + clear the skipped ones, atomically.

    One transaction for the whole batch: either all groups are applied and all skips
    are cleared, or (on an exception before the call — validation already passed in
    _validate_batch_choices_payload) nothing changes. Returns the number of saved
    (not skipped) groups.
    """
    now = datetime.now(timezone.utc).isoformat()
    conn = _connect(db_path)
    try:
        with conn:
            for group, keep in groups:
                for fid in group:
                    action = "keep" if fid == keep else "to_delete"
                    conn.execute(
                        """INSERT INTO dedup_choice (file_id, action, updated_at)
                           VALUES (?, ?, ?)
                           ON CONFLICT(file_id) DO UPDATE SET
                               action = excluded.action, updated_at = excluded.updated_at""",
                        (fid, action, now),
                    )
            for group in skip:
                placeholders = ",".join("?" * len(group))
                conn.execute(
                    f"DELETE FROM dedup_choice WHERE file_id IN ({placeholders})", group
                )
    finally:
        conn.close()
    return len(groups)


def _target_rel(dst: str, dest_root: str) -> str:
    """dst relative to dest_root, as in PlanItem.target_rel (see sorter.py).

    ValueError (a path-case divergence on Windows, etc.) -> the full dst, the same
    fallback as in sorter._target_parts/plan_and_sort.
    """
    try:
        return Path(dst).relative_to(Path(dest_root)).as_posix()
    except ValueError:
        return Path(dst).as_posix()


def _moves_payload(db_path: Path, batch_id: int | None) -> dict:
    """The sort --apply batch manifest: batch metadata + the list of moves.

    batch_id=None -> the last batch (MAX(id) in move_batches). No batches ->
    {"batch": None, "moves": []}, without crashing. name/target_rel are computed from
    dst — independent of the current files row (a trashed file after a move still
    shows its path in the manifest, just without a preview).
    """
    conn = _connect(db_path)
    try:
        if batch_id is None:
            row = conn.execute(
                "SELECT id, mode, dest_root, started_at, finished_at, operation "
                "FROM move_batches ORDER BY id DESC LIMIT 1"
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT id, mode, dest_root, started_at, finished_at, operation "
                "FROM move_batches WHERE id = ?", (batch_id,)
            ).fetchone()
        if row is None:
            return {"batch": None, "moves": []}
        batch = dict(row)
        move_rows = conn.execute(
            "SELECT file_id, src, dst, status FROM moves "
            "WHERE batch_id = ? ORDER BY dst", (batch["id"],)
        ).fetchall()
    finally:
        conn.close()

    dest_root = batch["dest_root"]
    moves = [
        {
            "file_id": r["file_id"],
            "name": Path(r["dst"]).name,
            "src": r["src"],
            "dst": r["dst"],
            "target_rel": _target_rel(r["dst"], dest_root),
            "status": r["status"],
            "thumb_url": f"/thumb/{r['file_id']}",
        }
        for r in move_rows
    ]
    return {"batch": batch, "moves": moves}


def _trash_files(db_path: Path, ids: list[int]) -> list[dict]:
    """The single trash path: ids -> OS trash + DELETE of their files/dedup_choice rows.

    Reused by group deletion of duplicates (`_trash_group`, U3) and by deletion of a
    single frame (`/api/photo/trash`, U4). An id outside the current files (already
    deleted/unknown) is silently skipped — idempotent on a repeated call.
    """
    if not ids:
        return []
    conn = _connect(db_path)
    try:
        placeholders = ",".join("?" * len(ids))
        rows = conn.execute(
            f"SELECT id, path FROM files WHERE id IN ({placeholders})", ids
        ).fetchall()
        trashed = []
        for r in rows:
            send_to_trash(r["path"])
            trashed.append({"file_id": r["id"], "name": Path(r["path"]).name})
        found_ids = [r["id"] for r in rows]
        if found_ids:
            ph2 = ",".join("?" * len(found_ids))
            with conn:
                conn.execute(f"DELETE FROM dedup_choice WHERE file_id IN ({ph2})", found_ids)
                conn.execute(f"DELETE FROM files WHERE id IN ({ph2})", found_ids)
    finally:
        conn.close()
    return trashed


def _trash_group(db_path: Path, group: list[int], keep_file_id: int) -> list[dict]:
    """The group's non-keepers -> trash (see `_trash_files` — the shared trash path)."""
    conn = _connect(db_path)
    try:
        placeholders = ",".join("?" * len(group))
        rows = conn.execute(
            f"SELECT id FROM files WHERE id IN ({placeholders})", group
        ).fetchall()
        ids_to_trash = [r["id"] for r in rows if r["id"] != keep_file_id]
    finally:
        conn.close()
    return _trash_files(db_path, ids_to_trash)


def _validate_file_id_payload(payload: object) -> int | None:
    """Parse the body `{"file_id": int}`. None -> invalid (not dict / not int / bool)."""
    if not isinstance(payload, dict):
        return None
    file_id = payload.get("file_id")
    if not isinstance(file_id, int) or isinstance(file_id, bool):
        return None
    return file_id


def _validate_file_ids_payload(payload: object) -> list[int] | None:
    """Parse the body `{"file_ids": [int, ...]}` (bulk deletion of the selected).

    None -> invalid (not dict / not a non-empty list of int without bool). Duplicates
    are collapsed, order is preserved — `_trash_files` itself ignores ids outside the DB.
    """
    if not isinstance(payload, dict):
        return None
    raw = payload.get("file_ids")
    if not isinstance(raw, list) or not raw:
        return None
    seen: set[int] = set()
    ids: list[int] = []
    for v in raw:
        if not isinstance(v, int) or isinstance(v, bool):
            return None
        if v not in seen:
            seen.add(v)
            ids.append(v)
    return ids


def _clusters_payload(db_path: Path, sample_limit: int = _CLUSTER_SAMPLE_LIMIT) -> list[dict]:
    """Root clusters (`merged_into IS NULL`) with size/label/samples.

    size — the number of faces in the whole merge chain (the root + everything merged
    into it), not just faces whose `faces.cluster_id` points directly to the root
    (after `merge` it keeps pointing to the original cluster — see `faces.merge`).
    samples — up to `sample_limit` distinct file_ids, ordered by `faces.id`
    (deterministic, stable between requests). Noise clusters (`faces.cluster_id IS
    NULL`) are naturally excluded by the `WHERE cluster_id IS NOT NULL` filter. Sorted
    by descending size.
    """
    conn = _connect(db_path)
    try:
        cluster_rows = conn.execute(
            "SELECT id, label, merged_into FROM face_clusters"
        ).fetchall()
        face_rows = conn.execute(
            "SELECT cluster_id, file_id FROM faces "
            "WHERE cluster_id IS NOT NULL ORDER BY id"
        ).fetchall()
    finally:
        conn.close()

    merged_into = {r["id"]: r["merged_into"] for r in cluster_rows}
    labels = {r["id"]: r["label"] for r in cluster_rows}
    root_ids = [r["id"] for r in cluster_rows if r["merged_into"] is None]

    def root_of(cid: int) -> int:
        seen: set[int] = set()
        while merged_into.get(cid) is not None and cid not in seen:
            seen.add(cid)
            cid = merged_into[cid]
        return cid

    size: dict[int, int] = defaultdict(int)
    samples: dict[int, list[int]] = defaultdict(list)
    sample_seen: dict[int, set[int]] = defaultdict(set)
    for r in face_rows:
        root = root_of(r["cluster_id"])
        size[root] += 1
        seen_files = sample_seen[root]
        if r["file_id"] not in seen_files and len(samples[root]) < sample_limit:
            seen_files.add(r["file_id"])
            samples[root].append(r["file_id"])

    result = [
        {
            "cluster_id": rid,
            "size": size.get(rid, 0),
            "label": labels.get(rid),
            "samples": samples.get(rid, []),
        }
        for rid in root_ids
    ]
    result.sort(key=lambda c: (-c["size"], c["cluster_id"]))
    return result


def _validate_cluster_label_payload(payload: object) -> tuple[int, str] | None:
    """Parse `{"cluster_id": int, "name": str}`. None -> invalid."""
    if not isinstance(payload, dict):
        return None
    cluster_id = payload.get("cluster_id")
    name = payload.get("name")
    if not isinstance(cluster_id, int) or isinstance(cluster_id, bool):
        return None
    if not isinstance(name, str):
        return None
    return cluster_id, name


def _validate_cluster_merge_payload(payload: object) -> tuple[int, int] | None:
    """Parse `{"src": int, "dst": int}`. None -> invalid."""
    if not isinstance(payload, dict):
        return None
    src = payload.get("src")
    dst = payload.get("dst")
    if not isinstance(src, int) or isinstance(src, bool):
        return None
    if not isinstance(dst, int) or isinstance(dst, bool):
        return None
    return src, dst


def _album_dest(cfg: Config, db_path: Path) -> Path:
    """The album root: `cfg.sort.album_dir` if set in the config, otherwise the default next to the DB."""
    album_dir = getattr(cfg.sort, "album_dir", None)
    if album_dir:
        return Path(album_dir)
    return db_path.resolve().parent / _DEFAULT_ALBUM_DIRNAME


def _suggested_sort_dest(cfg: Config, db_path: Path) -> str:
    """The default destination path for the city layout: `<source>_sorted`.

    The source — the first `cfg.sources` (config.yaml); if empty — the common root of
    the indexed files from the DB. Nothing found → an empty string (the field stays
    for manual entry). A POSIX path (like sources in config).
    """
    root: Path | None = None
    if cfg.sources:
        root = Path(cfg.sources[0])
    else:
        try:
            conn = _connect(db_path)
            try:
                paths = [r[0] for r in conn.execute(
                    "SELECT path FROM files WHERE error IS NULL").fetchall()]
            finally:
                conn.close()
            if paths:
                common = os.path.commonpath(paths)
                # commonpath over files returns an ancestor directory; if it matched a
                # single file (the only path) — take its parent
                root = Path(common)
                if root.suffix:  # this is a file, not a directory
                    root = root.parent
        except (ValueError, OSError):
            root = None
    if root is None:
        return ""
    return (root.parent / (root.name + "_sorted")).as_posix()


def _events_payload(db_path: Path,
                    sample_limit: int = _EVENT_SAMPLE_LIMIT) -> list[dict]:
    """The event list for the "Events" tab: id/name/count/dates + up to
    `sample_limit` preview file_ids (clickable -> lightbox), by descending count."""
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            """SELECT e.id, e.name, e.started_at, e.ended_at,
                      COUNT(ef.file_id) AS count
               FROM events e LEFT JOIN event_files ef ON ef.event_id = e.id
               GROUP BY e.id
               ORDER BY count DESC, e.id"""
        ).fetchall()
        # samples in a separate pass: the event's canonical frames by time,
        # up to sample_limit per event (as _clusters_payload accumulates in Python)
        samples: dict[int, list[int]] = defaultdict(list)
        for s in conn.execute(
            """SELECT ef.event_id, ef.file_id
               FROM event_files ef JOIN files f ON f.id = ef.file_id
               WHERE f.dup_of IS NULL AND f.error IS NULL
               ORDER BY ef.event_id, f.taken_at, f.id"""
        ):
            bucket = samples[s["event_id"]]
            if len(bucket) < sample_limit:
                bucket.append(s["file_id"])
    finally:
        conn.close()
    return [
        {
            "id": r["id"],
            "name": r["name"],
            "count": r["count"],
            "started_at": r["started_at"],
            "ended_at": r["ended_at"],
            "samples": samples.get(r["id"], []),
        }
        for r in rows
    ]


def _tabs_visibility_payload(db_path: Path) -> dict[str, bool]:
    """F54: visibility of the "People"/"Events" tabs — by data presence (variant B,
    without a meta table). person ⇔ there is a faces row with a non-empty cluster_id
    (the same source as `_clusters_payload`); event ⇔ non-empty `events`. Light
    EXISTS queries, we do not build the full payload."""
    conn = _connect(db_path)
    try:
        person = bool(conn.execute(
            "SELECT EXISTS(SELECT 1 FROM faces WHERE cluster_id IS NOT NULL)"
        ).fetchone()[0])
        event = bool(conn.execute(
            "SELECT EXISTS(SELECT 1 FROM events)"
        ).fetchone()[0])
    finally:
        conn.close()
    return {"person": person, "event": event}


def _validate_album_payload(
    payload: object,
) -> tuple[str, str, str, list[str], str | None, bool, str | None] | None:
    """Parse the body `POST /api/album`. None -> invalid (400).

    kind/mode — from `ALBUM_KINDS`/`ALBUM_MODES` (sorter.py), selector — a non-empty
    string, `where` (opt.) — a list of strings, `name` (opt.) — a string (empty after
    strip is treated as absent — the default name is used), `apply` (opt., default
    False) — bool, `dest` (opt., F60) — the album destination path as a string;
    empty/absent -> None (the server resolves the default itself via `_album_dest`).
    """
    if not isinstance(payload, dict):
        return None
    kind = payload.get("kind")
    if kind not in ALBUM_KINDS:
        return None
    mode = payload.get("mode")
    if mode not in ALBUM_MODES:
        return None
    selector = payload.get("selector")
    if not isinstance(selector, str) or not selector.strip():
        return None
    where = payload.get("where", [])
    if not isinstance(where, list) or not all(isinstance(w, str) for w in where):
        return None
    name = payload.get("name")
    if name is not None:
        if not isinstance(name, str):
            return None
        name = name.strip() or None
    apply_ = payload.get("apply", False)
    if not isinstance(apply_, bool):
        return None
    dest = payload.get("dest")
    if dest is not None:
        if not isinstance(dest, str):
            return None
        dest = dest.strip() or None
    return kind, selector, mode, where, name, apply_, dest


def _album_report_to_json(report: AlbumReport, applied: bool) -> dict:
    """`AlbumReport` -> the JSON response body of `POST /api/album`.

    For a preview (`applied=False`) `plan_album` does not compute `blocked_multi`
    (that is a side effect of the apply loop for mode='move') — here it is recomputed
    from `report.plan` with the same logic (`item.multi_person`), so the preview shows
    the expected blocking before the real move.
    """
    blocked = report.blocked_multi
    if not applied and report.mode == "move":
        blocked = sum(1 for it in report.plan if it.multi_person)
    return {
        "album_name": report.album_name,
        "dest": str(report.dest),
        "mode": report.mode,
        "kind": report.kind,
        "count": len(report.plan),
        "blocked_multi": blocked,
        "transferred": report.transferred,
        "failed": report.failed,
        "applied": applied,
    }


# --- F36: "Process" — the background pipeline index→geo→landmarks→faces→events→
# junk→phash from the web (POST /api/process), pollable progress (GET
# /api/process/status), cancel (POST /api/process/cancel). NOT imported from cli.py
# (to avoid a cli<->ui cycle) — the same leaf functions as `cli._pipeline_steps` are
# called directly from indexer/geo/landmarks/faces/events/junk/dedup/naming, +
# compute_phashes (dedup) as the last step.

_PIPELINE_STAGE_NAMES = ("index", "geo", "landmarks", "faces", "events", "junk", "phash")

# F53/#39: faces and events — the heaviest/longest steps, opt-in via the "Process"
# checkboxes, default off. `_pipeline_steps()` still builds the FULL list (see the
# assert above by _PIPELINE_STAGE_NAMES) — filtering is up to the caller
# (`_run_pipeline`), with the same name list as `cli._OPTIONAL_STAGES`.
_OPTIONAL_STAGES = ("faces", "events")


class _LazyClassifierHolder:
    """Builds the CLIP classifier on the first call, reuses it between landmarks and
    junk within ONE `/api/process` run (the same reason as
    `cli._LazySharedClassifier`, F19: a shared image-feature cache for the whole run).
    Laziness preserves incrementality — a run without new unknown places and without
    new files for junk does not load the CLIP model at all.
    """

    def __init__(self, factory: Callable[[], Classifier]) -> None:
        self._factory = factory
        self._real: Classifier | None = None

    def __call__(self, paths: list[str], prompts: list[str]):
        if self._real is None:
            self._real = self._factory()
        return self._real(paths, prompts)


def _pipeline_steps() -> list[tuple[str, Callable[[Config, sqlite3.Connection, _ProgressCB], None]]]:
    """Processing steps in dependency order — the same as `cli._pipeline_steps`, plus
    `phash` last (canonically from cli _pipeline_steps).
    A fresh holder per call — a separate run does not share the CLIP classifier with
    the previous/next run.
    """
    holder: dict[str, _LazyClassifierHolder] = {}

    def _clip(cfg: Config) -> _LazyClassifierHolder:
        clf = holder.get("clip")
        if clf is None:
            clf = holder["clip"] = _LazyClassifierHolder(
                lambda: clip_classifier(naming_settings(cfg)))
        return clf

    def _index(cfg: Config, conn: sqlite3.Connection, cb: _ProgressCB) -> None:
        run_index(cfg, conn, progress=lambda s: cb(s.scanned, None))
        assign_duplicates(conn, cfg.dedup.canonical_strategy)

    def _geo(cfg: Config, conn: sqlite3.Connection, cb: _ProgressCB) -> None:
        resolve_places(cfg, conn, progress=cb)

    def _landmarks(cfg: Config, conn: sqlite3.Connection, cb: _ProgressCB) -> None:
        detect_landmarks(cfg, conn, classifier=_clip(cfg), progress=cb)

    def _faces(cfg: Config, conn: sqlite3.Connection, cb: _ProgressCB) -> None:
        detect_and_cluster(cfg, conn, progress=cb)

    def _events(cfg: Config, conn: sqlite3.Connection, cb: _ProgressCB) -> None:
        build_events(cfg, conn, progress=cb)
        name_events(cfg, conn)

    def _junk(cfg: Config, conn: sqlite3.Connection, cb: _ProgressCB) -> None:
        classify_junk(cfg, conn, classifier=_clip(cfg), progress=cb)

    def _phash(cfg: Config, conn: sqlite3.Connection, cb: _ProgressCB) -> None:
        compute_phashes(cfg, conn, progress=cb)

    steps: list[tuple[str, Callable[[Config, sqlite3.Connection, _ProgressCB], None]]] = [
        ("index", _index), ("geo", _geo), ("landmarks", _landmarks),
        ("faces", _faces), ("events", _events), ("junk", _junk), ("phash", _phash),
    ]
    assert tuple(name for name, _fn in steps) == _PIPELINE_STAGE_NAMES
    return steps


class _PipelineCancelled(BaseException):
    """Pipeline cancellation from the progress callback (mid-stage). BaseException,
    not Exception, so an `except Exception` inside stages does not swallow it;
    caught only in `_run_pipeline`."""


class _ProcessState:
    """Thread-safe state of the background `/api/process` pipeline (F36).

    One run per server: `try_start` under the same `_lock` as all other mutations
    atomically rejects a repeated start while the previous one is still `running` —
    the `POST /api/process` handler turns False into 409. Updated by the stages'
    progress callbacks from the pipeline thread; read by `GET /api/process/status`
    from ThreadingHTTPServer request threads — hence a lock on every operation, not
    just a dataclass of fields.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._reset_locked()

    def _reset_locked(self) -> None:
        self.running = False
        self.stage: str | None = None
        self.stage_index = 0
        self.stage_total = 0
        self.done = 0
        self.total = 0
        self.error: str | None = None
        self.finished = False
        self.source_dir: str | None = None
        self._cancel_requested = False

    def try_start(self, source_dir: str) -> bool:
        """True and switches to running if nothing is going now; otherwise False (409)."""
        with self._lock:
            if self.running:
                return False
            self._reset_locked()
            self.running = True
            self.source_dir = source_dir
            return True

    def set_stage_total(self, total: int) -> None:
        with self._lock:
            self.stage_total = total

    def set_stage(self, index: int, name: str) -> None:
        with self._lock:
            self.stage_index = index
            self.stage = name
            self.done = 0
            self.total = 0

    def set_progress(self, done: int, total: int | None = None) -> None:
        """A signature superset of all stage ProgressCB variants (done, total|None).

        If cancellation is requested — raises _PipelineCancelled right from the
        callback: stages call progress often, so cancellation fires almost
        immediately (mid-stage), not only between stages."""
        with self._lock:
            cancel = self._cancel_requested
            if not cancel:
                self.done = done
                if total is not None:
                    self.total = total
        if cancel:
            raise _PipelineCancelled()

    def request_cancel(self) -> None:
        with self._lock:
            if self.running:
                self._cancel_requested = True

    def cancel_requested(self) -> bool:
        with self._lock:
            return self._cancel_requested

    def finish(self, error: str | None) -> None:
        with self._lock:
            self.running = False
            self.finished = True
            self.error = error

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "running": self.running,
                "stage": self.stage,
                "stage_index": self.stage_index,
                "stage_total": self.stage_total,
                "done": self.done,
                "total": self.total,
                "error": self.error,
                "finished": self.finished,
                "cancel_requested": self._cancel_requested,
                "source_dir": self.source_dir,
            }


_BROWSE_DIALOG_TIMEOUT_S = 120

_BROWSE_DIALOG_SCRIPT = (
    "import tkinter, tkinter.filedialog, sys\n"
    "root = tkinter.Tk()\n"
    "root.withdraw()\n"
    "root.attributes('-topmost', True)\n"
    "path = tkinter.filedialog.askdirectory()\n"
    "root.destroy()\n"
    "sys.stdout.write(path or '')\n"
)


def _browse_for_folder() -> str:
    """F51: a native folder-picker dialog for the "Browse…" button.

    tkinter is not thread-safe and requires the process's main thread — the
    POST /api/browse handler runs on a ThreadingHTTPServer thread, so the dialog is
    opened in a SEPARATE process (its own main thread, without a conflict with the web
    server). Any failure (no display/GUI, cancel, timeout, exception) -> an empty
    string, not an error — the button is just a convenience, manual path entry always
    works."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", _BROWSE_DIALOG_SCRIPT],
            capture_output=True, text=True, timeout=_BROWSE_DIALOG_TIMEOUT_S,
            check=False,
        )
    except Exception:
        _log.exception("не удалось открыть диалог выбора папки")
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _process_defaults_payload(cfg: Config) -> dict:
    """F57: defaults for the "Process" checkboxes — JS sets .checked by these values
    on page init (otherwise the checkboxes always start empty regardless of
    config.yaml). `vlm_available` — whether the `transformers` package is installed
    (`find_spec`, WITHOUT importing the module/loading the model)."""
    return {
        "deep": bool(cfg.naming.vlm_enabled),
        "geo_online": cfg.geo.provider == "online",
        "vlm_available": importlib.util.find_spec("transformers") is not None,
    }


def _env_payload() -> dict:
    """F64: the environment for the UI banner. `gpu_profile` — whether the GPU profile
    is installed (the nvidia-* packages exist only in the `gpu` extra; `find_spec`
    without importing torch). CPU profile -> False -> a reduced-speed banner on the
    "Process" tab. (Detects the chosen profile, not "whether CUDA works right now" —
    on a broken GPU profile the runtime fallback fires, which is a separate symptom.)"""
    return {"gpu_profile": importlib.util.find_spec("nvidia") is not None}


def _validate_process_payload(payload: object) -> tuple[str, bool, bool, bool, bool] | None:
    """Parse `{"source_dir": str, "deep": bool=False, "geo_online": bool=False,
    "faces": bool=False, "events": bool=False}` (F50/#34: opt-in VLM tier /
    online geo for THIS run, without editing config.yaml; F53/#39: opt-in steps
    faces/events, the same principle — default False).
    None -> invalid: not dict / `source_dir` not a string or empty after strip /
    `deep`, `geo_online`, `faces`, `events` given but not bool."""
    if not isinstance(payload, dict):
        return None
    source_dir = payload.get("source_dir")
    if not isinstance(source_dir, str) or not source_dir.strip():
        return None
    deep = payload.get("deep", False)
    if not isinstance(deep, bool):
        return None
    geo_online = payload.get("geo_online", False)
    if not isinstance(geo_online, bool):
        return None
    faces = payload.get("faces", False)
    if not isinstance(faces, bool):
        return None
    events = payload.get("events", False)
    if not isinstance(events, bool):
        return None
    return source_dir.strip(), deep, geo_online, faces, events


def _validate_rerun_optional_payload(payload: object) -> tuple[bool, bool, bool] | None:
    """Parse `{"faces": bool=False, "events": bool=False, "deep": bool=False}`
    for F62/F63 `POST /api/process/rerun-optional` (re-running the SELECTED on an
    already-built index: faces / events / junk-with-VLM when deep). None ->
    invalid: not dict / a field is given but not bool / all three False (nothing to
    re-run)."""
    if not isinstance(payload, dict):
        return None
    faces = payload.get("faces", False)
    if not isinstance(faces, bool):
        return None
    events = payload.get("events", False)
    if not isinstance(events, bool):
        return None
    deep = payload.get("deep", False)
    if not isinstance(deep, bool):
        return None
    if not faces and not events and not deep:
        return None
    return faces, events, deep


def _run_pipeline(db_path: Path, cfg: Config, source_dir: str | None,
                  state: _ProcessState, cache: PlanCache,
                  deep: bool = False, geo_online: bool = False,
                  faces: bool = False, events: bool = False,
                  only_optional: bool = False) -> None:
    """The body of the `POST /api/process` background thread: its own sqlite
    connection (not transferable between threads), source_dir overrides cfg.sources
    only for this run (F28-style, like `cli._cmd_index` with a positional src) — the
    original cfg shared with request handlers is not mutated. `source_dir=None` (F62:
    opt-in re-run over the existing index) leaves `cfg.sources` as-is — `Path(None)`
    is not called.

    `deep`/`geo_online` (F50/#34, a full override since F57/#57) — authoritatively set
    `naming.vlm_enabled`/`geo.provider` on this run_cfg regardless of what is in
    config.yaml: `deep=False` forces the VLM off even if `cfg.naming.vlm_enabled=True`
    (similarly `geo_online=False` forces `provider="offline"`). So the UI checkboxes
    (initialized from cfg via `/api/process/defaults`) can be unchecked to disable what
    is enabled in config.yaml — previously an unchecked box did not force OFF but
    quietly took cfg (the F57 bug). The server cfg/config.yaml is not re-read or
    mutated — the override lives only in this run's run_cfg.

    `faces`/`events` (F53/#39) — opt-in steps, default off: without the checkboxes the
    run builds only `index/geo/landmarks/junk/phash`, the heaviest steps are skipped.
    `stage_total`/the "stage i/N" numbering are computed from the actual filtered list.

    `only_optional` (F62/F63: "Re-run selected" — POST
    `/api/process/rerun-optional`) — steps are narrowed to the SELECTED stages over the
    already-built index: `faces` (with faces), `events` (with events), `junk` (with
    deep — reclassification with the VLM, `naming.vlm_enabled=deep`). The other base
    ones (index/geo/landmarks/phash) are not run at all.

    Cancellation is checked BETWEEN stages (not mid-stage — MVP). After a successful
    finish (without an error/cancel) the plan cache (the Cities tab) is recomputed
    with the same conn so the tabs show the new data right away; Duplicates/People/
    Events read the DB directly on each request and need no refresh.
    """
    conn = _connect(db_path)
    error: str | None = None
    try:
        naming = dataclasses.replace(cfg.naming, vlm_enabled=deep)
        geo = dataclasses.replace(cfg.geo, provider="online" if geo_online else "offline")
        sources = [Path(source_dir).resolve()] if source_dir is not None else cfg.sources
        run_cfg = dataclasses.replace(cfg, sources=sources, naming=naming, geo=geo)
        enabled_optional = {"faces": faces, "events": events}
        if only_optional:
            # F63: re-run the selected — faces/events by flags + junk with deep
            # (reclassification with the VLM). The order from _pipeline_steps is kept.
            rerun = {name for name in _OPTIONAL_STAGES if enabled_optional[name]}
            if deep:
                rerun.add("junk")
            steps = [(name, fn) for name, fn in _pipeline_steps() if name in rerun]
        else:
            steps = [(name, fn) for name, fn in _pipeline_steps()
                     if name not in _OPTIONAL_STAGES or enabled_optional[name]]
        state.set_stage_total(len(steps))
        completed = True
        for i, (name, fn) in enumerate(steps, 1):
            if state.cancel_requested():
                completed = False
                break
            state.set_stage(i, name)
            try:
                fn(run_cfg, conn, state.set_progress)
            except _PipelineCancelled:
                completed = False  # mid-stage cancellation via the progress callback
                break
            except Exception as exc:  # noqa: BLE001 — report via status, do not crash the thread
                error = str(exc)
                _log.exception("sorta ui: этап пайплайна %r упал", name)
                completed = False
                break
        if completed and error is None:
            try:
                cache.rebuild(cfg, conn)
            except Exception as exc:  # noqa: BLE001
                error = f"план не обновлён: {exc}"
    finally:
        conn.close()
        state.finish(error)


# --- F43: apply the city layout from the UI (`POST /api/sort`) — reuses the
# sorter.plan_and_sort(apply=True) engine one-to-one with the CLI `sort --by city
# --apply`; ui.py here is only background/progress (the _ProcessState/_run_pipeline
# pattern from F36) and request-body validation. The moves/move_batches journal,
# blake3 verification, name-conflict resolution and in-place semantics (dest=None) —
# entirely in sorter.py, not duplicated.

class _SortState:
    """Thread-safe state of the background `/api/sort` apply (F43) — modelled on
    `_ProcessState`, but without stages (one `plan_and_sort` operation)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._reset_locked()

    def _reset_locked(self) -> None:
        self.running = False
        self.done = 0
        self.total = 0
        self.error: str | None = None
        self.finished = False
        self.result: dict | None = None

    def try_start(self) -> bool:
        """True and switches to running if nothing is going now; otherwise False (409)."""
        with self._lock:
            if self.running:
                return False
            self._reset_locked()
            self.running = True
            return True

    def set_progress(self, done: int, total: int) -> None:
        with self._lock:
            self.done = done
            self.total = total

    def finish(self, error: str | None, result: dict | None) -> None:
        with self._lock:
            self.running = False
            self.finished = True
            self.error = error
            self.result = result

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "running": self.running,
                "done": self.done,
                "total": self.total,
                "error": self.error,
                "finished": self.finished,
                "result": self.result,
            }


def _validate_sort_payload(payload: object) -> tuple[str | None, str] | None:
    """Parse the body `POST /api/sort`: `{"dest": str|null|"", "mode": "move"|"copy"}`.

    None -> invalid (400): not dict / `mode` not in {move, copy} / `dest` not a string
    and not null. `dest` an empty/whitespace string or null -> None (in-place — layout
    inside the source folder, see `plan_and_sort` F28).
    """
    if not isinstance(payload, dict):
        return None
    mode = payload.get("mode")
    if mode not in ("move", "copy"):
        return None
    dest = payload.get("dest")
    if dest is not None and not isinstance(dest, str):
        return None
    dest = dest.strip() if isinstance(dest, str) else None
    return (dest or None), mode


def _validate_language_payload(payload: object) -> str | None:
    """Parse the body `POST /api/config/language`: `{"language": "ru"|"en"|"ja"}`.

    None -> invalid (400): not a dict / `language` not one of the supported codes."""
    if not isinstance(payload, dict):
        return None
    lang = payload.get("language")
    if not isinstance(lang, str):
        return None
    lang = lang.strip().lower()
    return lang if lang in _UI_LANGS else None


def _run_sort(db_path: Path, cfg: Config, dest: str | None, mode: str,
             state: _SortState, cache: PlanCache) -> None:
    """The body of the `POST /api/sort` background thread: its own sqlite connection
    (not transferable between threads, like `_run_pipeline`). Calls the ready
    `sorter.plan_and_sort(..., apply=True)` — the moves/move_batches journal, blake3
    verification and name-conflict resolution are the engine, here only
    progress/status and rebuilding PlanCache after a successful apply.

    `plan_and_sort` may raise `ValueError` (e.g. in-place with ≠1 source in
    `cfg.sources`) — caught and stored in the state as an error, the thread does not
    crash and the server stays alive.
    """
    conn = _connect(db_path)
    error: str | None = None
    result: dict | None = None
    try:
        dest_path = Path(dest) if dest else None
        try:
            report = plan_and_sort(cfg, conn, "city", dest_path, apply=True,
                                   copy=(mode == "copy"), progress=state.set_progress)
        except ValueError as exc:
            error = str(exc)
        else:
            result = {
                "moved": report.moved,
                "failed": report.failed,
                "skipped_in_place": report.skipped_in_place,
                "dirs": report.dirs,
                "dest": str(report.dest),
                "in_place": report.in_place,
                "mode": mode,
            }
            # F45: rebuild is only an update of the cities-tree preview cache, the
            # apply already happened (files laid out, the moves journal written) —
            # a rebuild failure is NOT a layout error, only a soft signal for the UI.
            try:
                cache.rebuild(cfg, conn)
            except Exception:  # noqa: BLE001
                _log.exception("sorta ui: план не обновлён после apply раскладки")
                result["preview_stale"] = True
    finally:
        conn.close()
        state.finish(error, result)


_UI_STRINGS: dict[str, dict[str, str]] = {
    "tab_process": {"ru": "Обработать", "en": "Process", "ja": "処理"},
    "tab_city": {"ru": "Города", "en": "Cities", "ja": "都市"},
    "tab_dupes": {"ru": "Дубли", "en": "Duplicates", "ja": "重複"},
    "tab_person": {"ru": "Люди", "en": "People", "ja": "人物"},
    "tab_event": {"ru": "События", "en": "Events", "ja": "イベント"},
    "tab_moves": {"ru": "Перемещения", "en": "Moves", "ja": "移動"},
    "process_intro": {
        "ru": "Укажите папку с фото и нажмите «Обработать» — индекс наполнится "
              "(гео, лица, события, мусор, почти-дубликаты). Файлы не перемещаются.",
        "en": "Enter a photo folder and click Process — the index fills in "
              "(geo, faces, events, junk, near-duplicates). Files are not moved.",
        "ja": "写真フォルダを指定して「処理する」を押すと、インデックスが作成されます"
              "（位置情報、顔、イベント、不要写真、類似写真）。ファイルは移動されません。",
    },
    "process_path_placeholder": {
        "ru": "Путь к папке с фото", "en": "Path to photo folder",
        "ja": "写真フォルダのパス",
    },
    "process_start_button": {"ru": "Обработать", "en": "Process", "ja": "処理する"},
    "process_browse_button": {"ru": "Обзор…", "en": "Browse…", "ja": "参照…"},
    "process_deep_label": {
        "ru": "Глубокий анализ (VLM)", "en": "Deep analysis (VLM)",
        "ja": "詳細分析（VLM）",
    },
    "process_deep_hint": {
        "ru": "Медленнее; нужен `uv sync --extra vlm` (иначе автоматический откат "
              "на быстрый анализ).",
        "en": "Slower; requires `uv sync --extra vlm` (otherwise falls back to "
              "the fast tier automatically).",
        "ja": "処理が遅くなります。`uv sync --extra vlm` が必要です"
              "（なければ自動的に高速分析にフォールバックします）。",
    },
    "process_deep_vlm_missing": {
        "ru": "VLM не установлен — будет использован быстрый ярус (CLIP). "
              "Доустановите: `uv sync --extra vlm`.",
        "en": "VLM is not installed — the fast tier (CLIP) will be used instead. "
              "Install it: `uv sync --extra vlm`.",
        "ja": "VLM がインストールされていません。代わりに高速ティア（CLIP）が"
              "使用されます。インストール: `uv sync --extra vlm`。",
    },
    "process_geo_online_label": {
        "ru": "Онлайн-гео (точнее заграница)", "en": "Online geo (more accurate abroad)",
        "ja": "オンライン位置情報（海外でより正確）",
    },
    "process_geo_online_hint": {
        "ru": "Точнее определяет места за границей, но отправляет GPS-координаты "
              "фото на сервер геокодирования (сами фото никуда не отправляются).",
        "en": "More accurate place names abroad, but sends photo GPS coordinates "
              "to a geocoding server (the photos themselves are never sent).",
        "ja": "海外の地名をより正確に特定しますが、写真のGPS座標をジオコーディング"
              "サーバーに送信します（写真自体は送信されません）。",
    },
    "process_faces_label": {
        "ru": "Разбор по лицам", "en": "Detect faces",
        "ja": "顔の検出",
    },
    "process_faces_hint": {
        "ru": "Самый долгий шаг (детекция + кластеризация); включай, если "
              "нужна раскладка/альбомы по людям.",
        "en": "The slowest step (detection + clustering); enable it if you "
              "need sorting or albums by person.",
        "ja": "最も時間のかかるステップです（検出とクラスタリング）。人物ごとの"
              "整理やアルバムが必要な場合に有効にしてください。",
    },
    "process_events_label": {
        "ru": "Разбор по событиям", "en": "Detect events",
        "ja": "イベントの検出",
    },
    "process_events_hint": {
        "ru": "Группировка в поездки/события по времени и месту (нужен geo); "
              "для раскладки/альбомов по событиям.",
        "en": "Groups photos into trips/events by time and place (needs "
              "geo); for sorting or albums by event.",
        "ja": "時間と場所に基づいて旅行やイベントにグループ化します"
              "（位置情報が必要）。イベントごとの整理やアルバムに使います。",
    },
    "process_rerun_optional_button": {
        "ru": "Дозапустить выбранное",
        "en": "Re-run selected",
        "ja": "選択項目を再実行",
    },
    "process_rerun_optional_hint": {
        "ru": "по активному индексу, без переиндексации/гео: лица (при «Разбор по "
              "лицам»), события (при «Разбор по событиям»), VLM-классификация "
              "(при «Глубокий анализ»)",
        "en": "on the current index, without re-indexing/geo: faces (if «Detect "
              "faces»), events (if «Detect events»), VLM classification (if «Deep "
              "analysis»)",
        "ja": "既存のインデックスに対して（再インデックス・位置情報なし）: 顔（「顔の"
              "検出」時）、イベント（「イベントの検出」時）、VLM 分類（「詳細分析」時）",
    },
    "env_cpu_warning": {
        "ru": "Установлен CPU-профиль: обработка идёт на процессоре — распознавание "
              "людей, VLM и большие коллекции заметно медленнее. Для скорости "
              "поставьте GPU-профиль: uv tool install --force \".[gpu]\".",
        "en": "CPU profile installed: processing runs on the CPU — face recognition, "
              "VLM and large collections are noticeably slower. For speed, install "
              "the GPU profile: uv tool install --force \".[gpu]\".",
        "ja": "CPU プロファイルがインストールされています: 処理は CPU で実行され、"
              "顔認識・VLM・大規模なコレクションは著しく遅くなります。高速化するには "
              "GPU プロファイルをインストールしてください: uv tool install --force \".[gpu]\"。",
    },
    "process_cancel_button": {"ru": "Отмена", "en": "Cancel", "ja": "キャンセル"},
    "process_enter_path": {
        "ru": "Введите путь к папке.", "en": "Enter a folder path.",
        "ja": "フォルダのパスを入力してください。",
    },
    "process_stage_progress": {
        "ru": "Этап {stage} ({index}/{total}): {done} из {all}",
        "en": "Stage {stage} ({index}/{total}): {done} of {all}",
        "ja": "ステージ {stage}（{index}/{total}）: {done}/{all}",
    },
    "process_stage_progress_indeterminate": {  # #37: total not yet known (e.g. indexing)
        "ru": "Этап {stage} ({index}/{total}): обработано {done}",
        "en": "Stage {stage} ({index}/{total}): {done} processed",
        "ja": "ステージ {stage}（{index}/{total}）: {done} 件処理済み",
    },
    "process_done": {
        "ru": "Обработка завершена.", "en": "Processing complete.",
        "ja": "処理が完了しました。",
    },
    "process_cancelled": {
        "ru": "Обработка остановлена.", "en": "Processing stopped.",
        "ja": "処理が中止されました。",
    },
    "process_cancel_requested": {
        "ru": "Отмена запрошена — остановка после текущего шага…",
        "en": "Cancel requested — stopping after the current step…",
        "ja": "キャンセルを要求しました — 現在のステップ後に停止します…",
    },
    "process_error_prefix": {
        "ru": "Ошибка обработки: ", "en": "Processing error: ", "ja": "処理エラー: ",
    },
    "process_start_error_prefix": {
        "ru": "Не удалось запустить: ", "en": "Failed to start: ", "ja": "開始できません: ",
    },
    "process_stage_index": {"ru": "индексация", "en": "indexing", "ja": "インデックス作成"},
    "process_stage_geo": {"ru": "гео", "en": "geo", "ja": "位置情報"},
    "process_stage_landmarks": {"ru": "места", "en": "landmarks", "ja": "ランドマーク"},
    "process_stage_faces": {"ru": "лица", "en": "faces", "ja": "顔"},
    "process_stage_events": {"ru": "события", "en": "events", "ja": "イベント"},
    "process_stage_junk": {"ru": "классификация", "en": "classification", "ja": "分類"},
    "process_stage_phash": {"ru": "почти-дубликаты", "en": "near-duplicates", "ja": "類似写真"},
    "process_reset_button": {
        "ru": "Начать заново", "en": "Start over", "ja": "最初からやり直す",
    },
    "process_reset_confirm": {
        "ru": "Сотрёт индекс, включая имена людей/событий и решения по дублям. "
              "Фото и уже разложенные папки НЕ тронет. Продолжить?",
        "en": "This will erase the index, including people/event names and "
              "duplicate decisions. Photos and already-sorted folders are NOT "
              "touched. Continue?",
        "ja": "人物名・イベント名・重複の判定を含むインデックスを消去します。"
              "写真や既に整理済みのフォルダには触れません。続行しますか?",
    },
    "process_reset_done": {
        "ru": "Индекс сброшен.", "en": "Index reset.", "ja": "インデックスをリセットしました。",
    },
    "process_reset_error_prefix": {
        "ru": "Не удалось сбросить: ", "en": "Failed to reset: ", "ja": "リセットできません: ",
    },
    "lightbox_close": {"ru": "Закрыть", "en": "Close", "ja": "閉じる"},
    "lightbox_open": {"ru": "Открыть превью", "en": "Open preview", "ja": "プレビューを開く"},
    "delete_remember_label": {
        "ru": "Не спрашивать подтверждение удаления в этой сессии",
        "en": "Don't ask for delete confirmation this session",
        "ja": "このセッション中は削除の確認をしない",
    },
    "expand_all": {"ru": "Развернуть всё", "en": "Expand all", "ja": "すべて展開"},
    "collapse_all": {"ru": "Свернуть всё", "en": "Collapse all", "ja": "すべて折りたたむ"},
    "back_to_top": {"ru": "Наверх", "en": "Top", "ja": "上へ"},
    "loading": {"ru": "Загрузка...", "en": "Loading...", "ja": "読み込み中..."},
    "save_all_choices": {
        "ru": "Сохранить весь выбор", "en": "Save all choices", "ja": "すべての選択を保存",
    },
    "merge_selected": {"ru": "Слить выбранные", "en": "Merge selected", "ja": "選択を統合"},
    "theme_light": {"ru": "Светлая", "en": "Light", "ja": "ライト"},
    "theme_dark": {"ru": "Тёмная", "en": "Dark", "ja": "ダーク"},
    "error_loading_plan": {
        "ru": "Ошибка загрузки плана: ", "en": "Error loading plan: ",
        "ja": "プラン読み込みエラー: ",
    },
    "error_loading_moves": {
        "ru": "Ошибка загрузки перемещений: ", "en": "Error loading moves: ",
        "ja": "移動読み込みエラー: ",
    },
    "error_loading_dupes": {
        "ru": "Ошибка загрузки дублей: ", "en": "Error loading duplicates: ",
        "ja": "重複読み込みエラー: ",
    },
    "error_loading_clusters": {
        "ru": "Ошибка загрузки кластеров: ", "en": "Error loading clusters: ",
        "ja": "クラスター読み込みエラー: ",
    },
    "confirm_delete_photo": {
        "ru": "Удалить этот файл в корзину?", "en": "Move this file to trash?",
        "ja": "このファイルをごみ箱に移動しますか?",
    },
    "delete": {"ru": "Удалить", "en": "Delete", "ja": "削除"},
    "delete_selected": {
        "ru": "Удалить выбранное", "en": "Delete selected", "ja": "選択を削除",
    },
    "select_for_delete": {
        "ru": "Выбрать для удаления", "en": "Select for deletion", "ja": "削除対象に選択",
    },
    "confirm_delete_selected": {
        "ru": "Удалить {n} файлов в корзину?", "en": "Move {n} files to trash?",
        "ja": "{n} 件のファイルをごみ箱に移動しますか?",
    },
    "status_planned": {"ru": "запланировано", "en": "planned", "ja": "予定"},
    "status_done": {"ru": "выполнено", "en": "done", "ja": "完了"},
    "status_undone": {"ru": "отменено", "en": "undone", "ja": "取消"},
    "status_failed": {"ru": "ошибка", "en": "failed", "ja": "失敗"},
    "status_deleted": {"ru": "удалено", "en": "deleted", "ja": "削除済み"},
    "batch_label": {"ru": "Батч", "en": "Batch", "ja": "バッチ"},
    "started_label": {"ru": "начат", "en": "started", "ja": "開始"},
    "finished_label": {"ru": "завершён", "en": "finished", "ja": "終了"},
    "in_progress_label": {"ru": "в процессе", "en": "in progress", "ja": "進行中"},
    "files_count_label": {"ru": "файлов", "en": "files", "ja": "ファイル数"},
    "no_moves_yet": {
        "ru": "Перемещений ещё не выполнялось.", "en": "No moves have been made yet.",
        "ja": "まだ移動は実行されていません。",
    },
    "unnamed": {"ru": "без имени", "en": "unnamed", "ja": "名前なし"},
    "faces_unit": {"ru": "лиц", "en": "faces", "ja": "顔"},
    "person_name_placeholder": {"ru": "Имя человека", "en": "Person's name", "ja": "人物名"},
    "name_button": {"ru": "Назвать", "en": "Name", "ja": "名前を設定"},
    "alert_enter_name": {
        "ru": "Введите имя.", "en": "Enter a name.", "ja": "名前を入力してください。",
    },
    "select_for_merge": {
        "ru": "выбрать для слияния", "en": "select for merge", "ja": "統合対象として選択",
    },
    "no_clusters": {
        "ru": "Кластеры лиц не найдены.", "en": "No face clusters found.",
        "ja": "顔クラスターが見つかりません。",
    },
    "recommended_badge": {
        "ru": "★ рекомендовано", "en": "★ recommended", "ja": "★ おすすめ",
    },
    "action_keep": {"ru": "оставить", "en": "keep", "ja": "保持"},
    "action_to_delete": {"ru": "к удалению", "en": "to delete", "ja": "削除予定"},
    "skip_group_label": {
        "ru": "не удалять эту группу", "en": "don't delete this group",
        "ja": "このグループを削除しない",
    },
    "delete_dupes_button": {
        "ru": "Удалить дубли", "en": "Delete duplicates", "ja": "重複を削除",
    },
    "confirm_trash_group": {
        "ru": "Удалить в корзину все кадры группы {n}, кроме выбранного?",
        "en": "Move all frames in group {n} to trash, except the selected one?",
        "ja": "選択したもの以外、グループ{n}のすべてのフレームをごみ箱に移動しますか?",
    },
    "alert_choose_keeper": {
        "ru": "Выберите кадр, который нужно оставить.", "en": "Select the frame to keep.",
        "ja": "残すフレームを選択してください。",
    },
    "no_dupes": {
        "ru": "Почти-дубликаты не найдены.", "en": "No near-duplicates found.",
        "ja": "ほぼ重複が見つかりません。",
    },
    "select_group_to_save": {
        "ru": "Отметьте хотя бы одну группу для сохранения.",
        "en": "Mark at least one group to save.",
        "ja": "保存するグループを少なくとも1つ選択してください。",
    },
    "saved_groups": {
        "ru": "Сохранено групп: {n}", "en": "Groups saved: {n}", "ja": "保存したグループ数: {n}",
    },
    "group_title": {
        "ru": "Группа {n} ({count} кадра)", "en": "Group {n} ({count} frames)",
        "ja": "グループ{n}（{count}枚）",
    },
    "album_button": {
        "ru": "Собрать в папку", "en": "Gather into folder", "ja": "フォルダにまとめる",
    },
    "album_mode_link": {"ru": "Ссылка (link)", "en": "Link", "ja": "リンク"},
    "album_mode_copy": {"ru": "Копия", "en": "Copy", "ja": "コピー"},
    "album_mode_move": {"ru": "Перемещение", "en": "Move", "ja": "移動"},
    "album_where_placeholder": {
        "ru": "Фильтр, напр. city=Барселона", "en": "Filter, e.g. city=Barcelona",
        "ja": "フィルター（例: city=Barcelona）",
    },
    "album_name_placeholder": {
        "ru": "Имя папки альбома", "en": "Album folder name", "ja": "アルバムフォルダ名",
    },
    "album_dest_placeholder": {
        "ru": "Путь назначения альбома", "en": "Album destination path",
        "ja": "アルバムの保存先パス",
    },
    "album_name_first_hint": {
        "ru": "Сначала назовите кластер", "en": "Name the cluster first",
        "ja": "先にクラスターに名前を付けてください",
    },
    "album_preview_text": {
        "ru": "{n} файлов → {dest}", "en": "{n} files → {dest}", "ja": "{n} ファイル → {dest}",
    },
    "album_blocked_text": {
        "ru": "; move заблокирует {k} мульти-кадров",
        "en": "; move will block {k} multi-person frames",
        "ja": "；moveは{k}件のマルチ人物フレームをブロックします",
    },
    "album_confirm_move": {
        "ru": "Внимание: перемещение изымет файлы из общего пула сортировки. Продолжить?",
        "en": "Warning: moving will remove files from the common sorting pool. Continue?",
        "ja": "警告: 移動するとファイルは共通の振り分けプールから除外されます。続行しますか?",
    },
    "album_confirm_generic": {
        "ru": "Собрать альбом?", "en": "Gather the album?", "ja": "アルバムをまとめますか?",
    },
    "album_result_text": {
        "ru": "Собрано {n}, ошибок {f}", "en": "Gathered {n}, errors {f}",
        "ja": "収集済み{n}、エラー{f}",
    },
    "album_in_progress": {
        "ru": "Идёт сбор альбома...", "en": "Gathering album...", "ja": "アルバムを収集中...",
    },
    "no_events": {
        "ru": "События не найдены.", "en": "No events found.", "ja": "イベントが見つかりません。",
    },
    "error_loading_events": {
        "ru": "Ошибка загрузки событий: ", "en": "Error loading events: ",
        "ja": "イベント読み込みエラー: ",
    },
    # --- F43: apply the city layout (the "Cities" tab) -----------------
    "sort_dest_placeholder": {
        "ru": "Папка назначения (пусто = в исходной папке)",
        "en": "Destination folder (empty = in the source folder)",
        "ja": "移動先フォルダ（空欄 = 元のフォルダ内）",
    },
    "sort_dest_hint": {
        "ru": "Пусто — коллекция раскладывается внутри исходной папки (in-place).",
        "en": "Empty — the collection is sorted inside the source folder (in-place).",
        "ja": "空欄の場合、コレクションは元のフォルダ内で振り分けられます（in-place）。",
    },
    "sort_dest_inplace_label": {
        "ru": "исходная папка (in-place)", "en": "source folder (in-place)",
        "ja": "元のフォルダ（in-place）",
    },
    "sort_mode_move": {"ru": "Переместить", "en": "Move", "ja": "移動"},
    "sort_mode_copy": {"ru": "Копировать", "en": "Copy", "ja": "コピー"},
    "sort_apply_button": {"ru": "Разложить", "en": "Apply", "ja": "振り分ける"},
    "folder_lang_label": {
        "ru": "Язык папок", "en": "Folder language", "ja": "フォルダの言語",
    },
    "folder_lang_saved": {
        "ru": "Язык папок сохранён — план пересчитан.",
        "en": "Folder language saved — the plan was recomputed.",
        "ja": "フォルダの言語を保存しました — プランを再計算しました。",
    },
    "sort_confirm_summary": {
        "ru": "{n} файлов, {dirs} папок → {dest}",
        "en": "{n} files, {dirs} folders → {dest}",
        "ja": "{n} ファイル、{dirs} フォルダ → {dest}",
    },
    "sort_confirm_move": {
        "ru": "ВНИМАНИЕ: оригиналы будут ПЕРЕМЕЩЕНЫ. Откат — команда sorta undo.",
        "en": "WARNING: originals will be MOVED. Roll back with the sorta undo command.",
        "ja": "警告: オリジナルファイルが移動されます。元に戻すには sorta undo コマンドを使用してください。",
    },
    "sort_confirm_inplace": {
        "ru": "ВНИМАНИЕ: реструктурируется ИСХОДНОЕ дерево коллекции, а не копия "
              "в отдельной папке.",
        "en": "WARNING: this restructures the SOURCE tree of the collection, "
              "not a copy in a separate folder.",
        "ja": "警告: これは別フォルダのコピーではなく、コレクションの元のツリー"
              "構造そのものを再編成します。",
    },
    "sort_confirm_copy": {
        "ru": "Оригиналы останутся на месте — будут созданы копии.",
        "en": "Originals stay in place — copies will be created.",
        "ja": "オリジナルはそのまま残り、コピーが作成されます。",
    },
    "sort_progress_line": {
        "ru": "Готово {done} из {all}", "en": "Done {done} of {all}",
        "ja": "完了 {done}/{all}",
    },
    "sort_done_text": {
        "ru": "Разложено {n}, ошибок {f} (+ пропущено {p} на месте)",
        "en": "Sorted {n}, errors {f} (+ {p} skipped in place)",
        "ja": "振り分け済み {n}、エラー {f}（+ その場でスキップ {p}）",
    },
    "sort_error_prefix": {
        "ru": "Ошибка раскладки: ", "en": "Sort error: ", "ja": "振り分けエラー: ",
    },
    "sort_preview_stale_warning": {
        "ru": "Превью плана не обновилось — обновите вкладку.",
        "en": "Plan preview did not refresh — reload the tab.",
        "ja": "プレビューが更新されませんでした — タブを再読み込みしてください。",
    },
    "sort_start_error_prefix": {
        "ru": "Не удалось запустить: ", "en": "Failed to start: ", "ja": "開始できません: ",
    },
}


def _t(key: str, lang: i18n.Lang) -> str:
    """Resolve a chrome UI string: exact language -> en -> the key itself (see F33)."""
    entry = _UI_STRINGS.get(key)
    if entry is None:
        return key
    return entry.get(lang) or entry.get("en") or key


_INDEX_HTML_TEMPLATE = """<!doctype html>
<html lang="{{lang}}"><head><meta charset="utf-8">
<title>Sorta UI</title>
<style>
:root {
  color-scheme: light;
  --bg: #F7F8FB;
  --surface: #FFFFFF;
  --head-bg: #FBFCFE;
  --card: #FFFFFF;
  --chip: #F3F5F9;
  --field: #FFFFFF;
  --track: #E7EBF2;
  --ink: #1A2230;
  --muted: #5B6675;
  --line: #E3E7EE;
  --accent: #2F5BD0;
  --accent-soft: #B9C8EF;
  --on-accent: #FFFFFF;
  --tab-active-ink: #1A2230;
  --tab-active-bg: #FFFFFF;
  --tab-active-line: #DBE1EA;
  --good: #1E9E6A;
  --good-soft: #BFE7D5;
  --danger: #D14343;
  --danger-soft: #EAB6B6;
  --radius-sm: 5px;
  --radius-md: 8px;
  --radius-lg: 10px;
  --radius-pill: 999px;
  --shadow-sm: 0 1px 2px rgba(20,30,50,.06);
  --shadow-lg: 0 8px 24px rgba(20,30,50,.05);
  --space-xs: 4px;
  --space-sm: 8px;
  --space-md: 12px;
  --space-lg: 16px;
  --space-xl: 24px;
  --font-sans: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --bg: #141A22;
    --surface: #181F29;
    --head-bg: #171E27;
    --card: #181F29;
    --chip: #1E2731;
    --field: #121821;
    --track: #232D39;
    --ink: #E6EAF0;
    --muted: #8A96A6;
    --line: #28323F;
    --accent: #6E9BFF;
    --accent-soft: #31456E;
    --on-accent: #0B1220;
    --tab-active-ink: #E6EAF0;
    --tab-active-bg: #212B37;
    --tab-active-line: #334053;
    --good: #3ECB95;
    --good-soft: #204A3A;
    --danger: #F0736F;
    --danger-soft: #5A2C2C;
    --shadow-sm: none;
    --shadow-lg: none;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --bg: #141A22;
  --surface: #181F29;
  --head-bg: #171E27;
  --card: #181F29;
  --chip: #1E2731;
  --field: #121821;
  --track: #232D39;
  --ink: #E6EAF0;
  --muted: #8A96A6;
  --line: #28323F;
  --accent: #6E9BFF;
  --accent-soft: #31456E;
  --on-accent: #0B1220;
  --tab-active-ink: #E6EAF0;
  --tab-active-bg: #212B37;
  --tab-active-line: #334053;
  --good: #3ECB95;
  --good-soft: #204A3A;
  --danger: #F0736F;
  --danger-soft: #5A2C2C;
  --shadow-sm: none;
  --shadow-lg: none;
}
:root[data-theme="light"] {
  color-scheme: light;
  --bg: #F7F8FB;
  --surface: #FFFFFF;
  --head-bg: #FBFCFE;
  --card: #FFFFFF;
  --chip: #F3F5F9;
  --field: #FFFFFF;
  --track: #E7EBF2;
  --ink: #1A2230;
  --muted: #5B6675;
  --line: #E3E7EE;
  --accent: #2F5BD0;
  --accent-soft: #B9C8EF;
  --on-accent: #FFFFFF;
  --tab-active-ink: #1A2230;
  --tab-active-bg: #FFFFFF;
  --tab-active-line: #DBE1EA;
  --good: #1E9E6A;
  --good-soft: #BFE7D5;
  --danger: #D14343;
  --danger-soft: #EAB6B6;
  --shadow-sm: 0 1px 2px rgba(20,30,50,.06);
  --shadow-lg: 0 8px 24px rgba(20,30,50,.05);
}
* { box-sizing: border-box; }
html, body { max-width: 100%; overflow-x: hidden; }
body {
  font-family: var(--font-sans);
  margin: 0;
  padding: var(--space-lg) var(--space-xl) var(--space-xl);
  background: var(--bg);
  color: var(--ink);
  font-size: 14px;
  line-height: 1.45;
}
h1, h2, h3 { font-weight: 600; }
a { color: var(--accent); }
:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; border-radius: var(--radius-sm); }
@media (prefers-reduced-motion: no-preference) {
  .btn, .tab-btn, .top-btn, details > summary, .stage-chip, .thumb-skel img { transition: background .12s ease, border-color .12s ease, color .12s ease, opacity .12s ease, transform .12s ease; }
}

/* --- таблицы -------------------------------------------------------- */
.table-wrap { width: 100%; max-width: 100%; overflow-x: auto; border-radius: var(--radius-md); border: 1px solid var(--line); }
table { border-collapse: collapse; width: 100%; background: var(--surface); font-variant-numeric: tabular-nums; }
td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--line); vertical-align: middle; }
tr:last-child td { border-bottom: none; }
tbody tr:nth-child(even), table tr:nth-child(even) { background: var(--chip); }
table tr:hover { background: var(--accent-soft); }
img { width: 56px; height: 56px; object-fit: cover; border-radius: var(--radius-sm); border: 1px solid var(--line);
      vertical-align: middle; margin-right: var(--space-sm); background: var(--chip); }
details { margin-left: var(--space-md); }
summary { cursor: pointer; font-weight: 600; margin: var(--space-sm) 0; overflow-wrap: anywhere; list-style-position: outside; }
details .table-wrap { margin: 0.3rem 0 0.8rem var(--space-md); width: calc(100% - 1rem); }

/* --- кнопки ----------------------------------------------------------- */
.btn {
  display: inline-flex; align-items: center; gap: 6px;
  font-family: var(--font-sans); font-size: 13px; font-weight: 500; line-height: 1;
  padding: 7px 12px; margin: 0; cursor: pointer;
  background: var(--chip); color: var(--ink);
  border: 1px solid var(--line); border-radius: var(--radius-md);
}
.btn:hover { border-color: var(--accent); }
.btn:active { transform: translateY(1px); }
.btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
.btn:disabled:hover { border-color: var(--line); }
.btn svg { width: 14px; height: 14px; flex: none; }
.btn-primary { background: var(--accent); color: var(--on-accent); border-color: var(--accent); font-weight: 600; }
.btn-primary:hover { filter: brightness(1.06); border-color: var(--accent); }
.btn-ghost { background: transparent; }
.btn-danger { background: transparent; color: var(--danger); border-color: var(--danger-soft); }
.btn-danger:hover { background: var(--danger-soft); border-color: var(--danger); }
.btn-sm { padding: 4px 9px; font-size: 12px; }

/* --- шапка / вордмарк --------------------------------------------------- */
.header-bar { display: flex; align-items: center; justify-content: space-between;
      margin: 0 0 var(--space-lg) 0; gap: var(--space-md); flex-wrap: wrap; }
.brand { display: flex; align-items: center; gap: 8px; }
.brand-mark { width: 26px; height: 26px; color: var(--accent); flex: none; }
.brand-name { font-size: 1.15rem; font-weight: 700; letter-spacing: 0.01em; }
.header-controls { display: flex; align-items: center; gap: var(--space-sm); flex-wrap: wrap; }
.lang-field { display: inline-flex; align-items: center; gap: 5px; background: var(--chip);
      border: 1px solid var(--line); border-radius: var(--radius-md); padding: 0 4px 0 8px; }
.lang-field svg { width: 14px; height: 14px; color: var(--muted); flex: none; }
.lang-select { padding: 6px 4px; cursor: pointer; border: none; background: transparent; color: var(--ink);
      font-family: var(--font-sans); font-size: 13px; }
/* нативный option-попап в тёмной теме иначе белый со светлым текстом (плохой
   контраст) — явные цвета по токенам темы + color-scheme выше делают его читаемым */
.lang-select option { background: var(--surface); color: var(--ink); }
.theme-toggle-btn svg { width: 15px; height: 15px; }

/* --- вкладки ------------------------------------------------------------ */
.tabs { display: flex; gap: 4px; margin-bottom: var(--space-lg); border-bottom: 1px solid var(--line);
      overflow-x: auto; overflow-y: hidden; scrollbar-width: thin; }
.tab-btn {
  flex: none; padding: 8px 16px; cursor: pointer; font-family: var(--font-sans); font-size: 13.5px;
  font-weight: 500; color: var(--muted); background: transparent;
  border: 1px solid transparent; border-bottom: none; border-radius: var(--radius-md) var(--radius-md) 0 0;
}
.tab-btn:hover { color: var(--ink); background: var(--chip); }
.tab-btn.active { background: var(--tab-active-bg); color: var(--tab-active-ink);
      border-color: var(--tab-active-line); font-weight: 600; }
.tab-panel { display: none; }
.tab-panel.active { display: block; }

/* --- общие карточки ------------------------------------------------------ */
.card { border: 1px solid var(--line); border-radius: var(--radius-lg); padding: var(--space-md) var(--space-lg);
      margin-bottom: var(--space-md); background: var(--card); box-shadow: var(--shadow-sm); }
.card.named { border-color: var(--accent-soft); border-width: 1.5px; }
.card h3 { margin: 0 0 var(--space-sm) 0; font-size: 0.95rem; }

/* --- бейджи / чипы -------------------------------------------------------- */
.badge { display: inline-flex; align-items: center; gap: 3px; color: var(--good); font-weight: 600;
      margin-left: 6px; font-size: 0.85em; }
.badge svg { width: 12px; height: 12px; }
.chip { display: inline-flex; align-items: center; padding: 2px 8px; border-radius: var(--radius-pill);
      font-size: 0.78rem; font-weight: 500; background: var(--chip); color: var(--muted); }
.chip-good { background: var(--good-soft); color: var(--good); }
.chip-accent { background: var(--accent-soft); color: var(--accent); }
.chip-danger { background: var(--danger-soft); color: var(--danger); }

/* --- инпуты/селекты --------------------------------------------------- */
input[type="text"], select {
  font-family: var(--font-sans); font-size: 13px; color: var(--ink); background: var(--field);
  border: 1px solid var(--line); border-radius: var(--radius-md); padding: 7px 9px;
}
input[type="text"]:focus-visible, select:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
input[type="checkbox"], input[type="radio"] { accent-color: var(--accent); width: 15px; height: 15px; cursor: pointer; }
label { cursor: pointer; }

/* --- состояния: пусто/загрузка/ошибка ---------------------------------- */
.state-msg { display: flex; align-items: center; gap: 8px; padding: var(--space-md) var(--space-lg);
      border-radius: var(--radius-md); color: var(--muted); background: var(--chip); }
.state-error { color: var(--danger); background: var(--danger-soft); }
.state-msg svg { width: 15px; height: 15px; flex: none; }
@media (prefers-reduced-motion: no-preference) {
  .state-loading svg { animation: sorta-spin 0.9s linear infinite; }
}
@keyframes sorta-spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }

.tree-controls { margin: 0 0 var(--space-md) 0; display: flex; gap: var(--space-sm); }
.top-btn { position: fixed; right: 1.2rem; bottom: 1.2rem; padding: 9px 14px;
      cursor: pointer; border-radius: var(--radius-pill); opacity: 0.9; z-index: 1000;
      background: var(--surface); box-shadow: var(--shadow-lg); }
.top-btn:hover { opacity: 1; }
.dupes-controls { margin: 0 0 var(--space-md) 0; display: flex; align-items: center; gap: var(--space-sm); }
.dupes-controls #dupes-save-status { color: var(--good); font-size: 0.85rem; }
.dupe-group .table-wrap { margin-bottom: var(--space-sm); }
.skip-label { display: inline-flex; align-items: center; gap: 5px; font-size: 0.85rem; color: var(--muted);
      margin-right: var(--space-md); }
.cluster-controls { margin: 0 0 var(--space-md) 0; }
#clusters-grid, #events-list { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
      gap: var(--space-md); }
.cluster-thumbs { display: flex; flex-wrap: wrap; gap: 3px; margin-bottom: var(--space-sm); }
.thumb-skel { width: 44px; height: 44px; border-radius: var(--radius-sm); background: var(--track);
      overflow: hidden; }
.thumb-skel img { width: 100%; height: 100%; margin: 0; object-fit: cover; cursor: zoom-in;
      display: block; opacity: 0; }
.thumb-skel.loaded { background: transparent; }
.thumb-skel.loaded img { opacity: 1; }
.thumb-skel img:hover { outline: 2px solid var(--accent); outline-offset: -2px; }
.cluster-meta { font-size: 0.85rem; color: var(--muted); margin: 0 0 var(--space-sm) 0; }
.cluster-name-form { display: flex; gap: 5px; margin-bottom: var(--space-sm); }
.cluster-name-form input { flex: 1; min-width: 0; }
.cluster-merge-select { font-size: 0.8rem; display: flex; align-items: center; gap: 5px; color: var(--muted); }
.album-controls { display: flex; align-items: center; gap: 5px; margin-top: var(--space-sm); flex-wrap: wrap; }
.album-controls select { font-size: 0.8rem; padding: 6px 7px; }
.album-controls input[type="text"] { flex: 1; min-width: 90px; font-size: 0.8rem; padding: 6px 7px; }
.album-status { font-size: 0.8rem; color: var(--good); margin-left: 2px; }
.album-hint { font-size: 0.8rem; color: var(--muted); margin-top: var(--space-sm); font-style: italic; }
.event-meta { font-size: 0.85rem; color: var(--muted); margin: 0 0 var(--space-sm) 0; }
.event-thumbs { display: flex; flex-wrap: wrap; gap: 3px; margin: 0 0 var(--space-sm) 0; }
/* единый вид кликабельной миниатюры-превью (Города/Дубли/Перемещения/События) */
/* фон-плейсхолдер виден, пока lazy-<img> не загрузился — отклик вместо «пусто» */
.clickable-thumb { cursor: zoom-in; background: var(--track); }
.clickable-thumb:hover { outline: 2px solid var(--accent); outline-offset: -2px; }
.thumb-name { display: block; font-size: 0.8rem; color: var(--muted); word-break: break-all; margin-top: 2px; }
.event-name-input { width: 100%; margin-bottom: var(--space-sm); box-sizing: border-box; }
.process-intro { max-width: 46rem; color: var(--muted); }
/* F51: вертикальные группы (путь / каждый тумблер+hint / кнопки), а не один
   плоский flex — там .process-toggle-hint с flex-basis:100% уезжал в конец
   контейнера, после всех кнопок, оторвано от своего чекбокса. */
.process-controls { display: flex; flex-direction: column; gap: var(--space-sm);
      margin: var(--space-md) 0; max-width: 42rem; }
.process-path-row { display: flex; gap: var(--space-sm); align-items: center; flex-wrap: wrap; }
.process-path-row input[type="text"] { flex: 1; min-width: 220px; padding: 8px 10px; }
.process-option { display: flex; flex-direction: column; gap: 2px; }
.process-toggle-label { font-size: 0.85rem; display: inline-flex; align-items: center; gap: 4px; }
.process-toggle-hint { font-size: 0.8rem; color: var(--muted); margin-left: 20px; }
.process-toggle-warn { color: var(--danger); }
.process-actions { display: flex; gap: var(--space-sm); flex-wrap: wrap; align-items: center; }
.process-rerun-block { display: flex; flex-direction: column; align-items: flex-start; gap: 3px; margin-top: var(--space-sm); }
.process-rerun-hint { font-size: 0.8rem; color: var(--muted); margin: 0; max-width: 40rem; }
.process-progress { width: 100%; max-width: 40rem; display: block; margin: var(--space-sm) 0; height: 8px;
      appearance: none; border: none; border-radius: var(--radius-pill); overflow: hidden; background: var(--track); }
.process-progress::-webkit-progress-bar { background: var(--track); border-radius: var(--radius-pill); }
.process-progress::-webkit-progress-value { background: var(--accent); border-radius: var(--radius-pill); }
.process-progress::-moz-progress-bar { background: var(--accent); border-radius: var(--radius-pill); }
/* #37: total ещё неизвестен (индексация сканирует дерево) — вместо «0 из 0»
   бегущая полоса «идёт работа». Определённый прогресс (total>0) заполняется как
   обычно (::progress-value выше). */
.process-progress.indeterminate { background-image: linear-gradient(90deg,
      var(--track) 0%, var(--accent-soft) 40%, var(--accent) 50%, var(--accent-soft) 60%, var(--track) 100%);
      background-size: 240% 100%; background-repeat: no-repeat; }
.process-progress.indeterminate::-webkit-progress-bar { background: transparent; }
.process-progress.indeterminate::-webkit-progress-value { background: transparent; }
.process-progress.indeterminate::-moz-progress-bar { background: transparent; }
@media (prefers-reduced-motion: no-preference) {
  .process-progress.indeterminate { animation: process-indeterminate 1.2s linear infinite; }
}
@keyframes process-indeterminate { from { background-position: 120% 0; } to { background-position: -120% 0; } }
.process-status { margin: var(--space-sm) 0; color: var(--muted); }
/* F64: инфо-баннер о CPU-профиле (амбер, читается в обеих темах через --ink) */
.env-warning { margin-top: var(--space-md); padding: 10px 13px; font-size: 0.85rem;
      border-radius: var(--radius-md); color: var(--ink); line-height: 1.45;
      background: rgba(214, 158, 46, 0.13); border: 1px solid rgba(214, 158, 46, 0.42); }
.stage-chips { display: flex; flex-wrap: wrap; gap: 6px; margin: var(--space-sm) 0; }
.stage-chip { display: inline-flex; align-items: center; gap: 5px; padding: 3px 10px; border-radius: var(--radius-pill);
      font-size: 0.78rem; font-weight: 500; background: var(--chip); color: var(--muted); border: 1px solid var(--line); }
.stage-chip svg { width: 11px; height: 11px; }
.stage-chip.done { background: var(--good-soft); color: var(--good); border-color: transparent; }
.stage-chip.now { background: var(--accent-soft); color: var(--accent); border-color: transparent; font-weight: 600; }

.sort-controls { display: flex; gap: var(--space-sm); flex-wrap: wrap; align-items: center; margin: var(--space-md) 0; }
.sort-controls input[type="text"] { flex: 1; min-width: 220px; padding: 8px 10px; }
.sort-dest-hint { flex-basis: 100%; font-size: 0.8rem; color: var(--muted); }
.sort-mode-label { font-size: 0.85rem; display: inline-flex; align-items: center; gap: 4px; }

/* --- лайтбокс (F42): один переиспользуемый оверлей для крупного просмотра --- */
.lightbox { position: fixed; inset: 0; z-index: 2000; display: flex; align-items: center;
      justify-content: center; padding: var(--space-xl); background: rgba(10,14,22,.86);
      cursor: zoom-out; }
.lightbox[hidden] { display: none; }
.lightbox img { width: auto; height: auto; max-width: 100%; max-height: 100%;
      object-fit: contain; cursor: default;
      border-radius: var(--radius-md); box-shadow: var(--shadow-lg); background: var(--surface); }

@media (max-width: 640px) {
  body { padding: var(--space-md); }
  #clusters-grid, #events-list { grid-template-columns: 1fr; }
  .process-path-row { flex-direction: column; align-items: stretch; }
  .process-path-row input[type="text"] { min-width: 100%; }
}
</style></head><body>
<div class="header-bar">
<div class="brand">
<svg class="brand-mark" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"
     stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
<path d="M4 7l8-3 8 3v10l-8 3-8-3V7z"/><path d="M4 7l8 3 8-3"/><path d="M12 10v10"/>
</svg>
<span class="brand-name">Sorta</span>
</div>
<div class="header-controls">
<label class="lang-field">
<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true">
<circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3c2.5 2.7 3.8 6 3.8 9s-1.3 6.3-3.8 9c-2.5-2.7-3.8-6-3.8-9s1.3-6.3 3.8-9z"/>
</svg>
<select id="lang-select" class="lang-select">{{lang_options}}</select>
</label>
<button type="button" id="theme-toggle-btn" class="btn btn-ghost theme-toggle-btn">
<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"
     stroke-linejoin="round" aria-hidden="true"><path d="M20 14.5A8.5 8.5 0 1 1 9.5 4a7 7 0 0 0 10.5 10.5z"/></svg>
<span id="theme-toggle-label">{{theme_dark}}</span></button>
</div>
</div>
<div class="tabs" role="tablist">
<button type="button" class="tab-btn active" id="tab-btn-process">{{tab_process}}</button>
<button type="button" class="tab-btn" id="tab-btn-city">{{tab_city}}</button>
<button type="button" class="tab-btn" id="tab-btn-dupes">{{tab_dupes}}</button>
<button type="button" class="tab-btn" id="tab-btn-person" style="display:none">{{tab_person}}</button>
<button type="button" class="tab-btn" id="tab-btn-event" style="display:none">{{tab_event}}</button>
<button type="button" class="tab-btn" id="tab-btn-moves">{{tab_moves}}</button>
</div>
<p id="delete-remember-row" style="display:none"><label><input type="checkbox" id="delete-remember">
{{delete_remember_label}}</label></p>

<section id="tab-process" class="tab-panel active">
<p class="process-intro">{{process_intro}}</p>
<div class="process-controls">
<div class="process-path-row">
<input type="text" id="process-source-dir" placeholder="{{process_path_placeholder}}">
<button type="button" id="process-browse-btn" class="btn btn-ghost">{{process_browse_button}}</button>
</div>
<div class="process-option">
<label class="process-toggle-label"><input type="checkbox" id="process-deep-checkbox"> {{process_deep_label}}</label>
<span class="process-toggle-hint">{{process_deep_hint}}</span>
<span id="process-deep-vlm-missing" class="process-toggle-hint process-toggle-warn" style="display:none">{{process_deep_vlm_missing}}</span>
</div>
<div class="process-option">
<label class="process-toggle-label"><input type="checkbox" id="process-geo-online-checkbox"> {{process_geo_online_label}}</label>
<span class="process-toggle-hint">{{process_geo_online_hint}}</span>
</div>
<div class="process-option">
<label class="process-toggle-label"><input type="checkbox" id="process-faces-checkbox"> {{process_faces_label}}</label>
<span class="process-toggle-hint">{{process_faces_hint}}</span>
</div>
<div class="process-option">
<label class="process-toggle-label"><input type="checkbox" id="process-events-checkbox"> {{process_events_label}}</label>
<span class="process-toggle-hint">{{process_events_hint}}</span>
</div>
<div class="process-actions">
<button type="button" id="process-start-btn" class="btn btn-primary">{{process_start_button}}</button>
<button type="button" id="process-cancel-btn" class="btn btn-ghost process-cancel-btn" style="display:none">{{process_cancel_button}}</button>
<button type="button" id="process-reset-btn" class="btn btn-danger">{{process_reset_button}}</button>
</div>
<div class="process-rerun-block">
<button type="button" id="process-rerun-optional-btn" class="btn btn-ghost" disabled>{{process_rerun_optional_button}}</button>
<span class="process-rerun-hint">{{process_rerun_optional_hint}}</span>
</div>
</div>
<progress id="process-progress" class="process-progress" max="0" value="0" style="display:none"></progress>
<div id="process-stages" class="stage-chips"></div>
<div id="process-status" class="process-status"></div>
<div id="env-cpu-warning" class="env-warning" style="display:none">⚠ {{env_cpu_warning}}</div>
</section>

<section id="tab-city" class="tab-panel">
<div class="sort-controls">
<label class="sort-mode-label" for="folder-lang-select">{{folder_lang_label}}
<select id="folder-lang-select"><option value="ru">Русский</option><option value="en">English</option><option value="ja">日本語</option></select></label>
<input type="text" id="sort-dest" placeholder="{{sort_dest_placeholder}}">
<button type="button" id="sort-browse-btn" class="btn btn-ghost">{{process_browse_button}}</button>
<label class="sort-mode-label"><input type="radio" name="sort-mode" value="move" checked> {{sort_mode_move}}</label>
<label class="sort-mode-label"><input type="radio" name="sort-mode" value="copy"> {{sort_mode_copy}}</label>
<button type="button" id="sort-apply-btn" class="btn btn-primary">{{sort_apply_button}}</button>
<span class="sort-dest-hint">{{sort_dest_hint}}</span>
</div>
<progress id="sort-progress" class="process-progress" max="0" value="0" style="display:none"></progress>
<div id="sort-status" class="process-status"></div>
<div id="sort-warning" class="process-status"></div>
<div class="tree-controls">
<button type="button" class="btn btn-ghost expand-all-btn">{{expand_all}}</button>
<button type="button" class="btn btn-ghost collapse-all-btn">{{collapse_all}}</button>
<button type="button" id="city-delete-selected-btn" class="btn btn-danger" disabled>{{delete_selected}}<span id="city-delete-selected-count"></span></button>
</div>
<div id="tree-city"><div class="state-msg state-loading">{{loading}}</div></div>
</section>

<section id="tab-dupes" class="tab-panel">
<div class="dupes-controls">
<button type="button" id="dupes-save-all-btn" class="btn btn-primary">{{save_all_choices}}</button>
<span id="dupes-save-status"></span>
</div>
<div id="dupes-list"><div class="state-msg state-loading">{{loading}}</div></div>
</section>

<section id="tab-person" class="tab-panel">
<div class="cluster-controls">
<button type="button" id="clusters-merge-btn" class="btn btn-primary" disabled>
<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"
     stroke-linejoin="round" aria-hidden="true"><circle cx="6" cy="4.5" r="1.6"/><circle cx="18" cy="4.5" r="1.6"/>
<circle cx="12" cy="19.5" r="1.6"/><path d="M6 6v3c0 2.5 2 4 4 4h1M18 6v3c0 2.5-2 4-4 4h-1M12 13v5"/></svg>
{{merge_selected}}</button>
</div>
<div id="clusters-grid"><div class="state-msg state-loading">{{loading}}</div></div>
</section>

<section id="tab-event" class="tab-panel">
<div id="events-list"><div class="state-msg state-loading">{{loading}}</div></div>
</section>

<section id="tab-moves" class="tab-panel">
<div id="moves-summary"></div>
<div class="tree-controls">
<button type="button" class="btn btn-ghost expand-all-btn">{{expand_all}}</button>
<button type="button" class="btn btn-ghost collapse-all-btn">{{collapse_all}}</button>
</div>
<div id="tree-moves"><div class="state-msg state-loading">{{loading}}</div></div>
</section>

<button type="button" id="top-btn" class="btn top-btn" title="{{back_to_top}}">
<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"
     stroke-linejoin="round" aria-hidden="true"><path d="M12 19V5M5 12l7-7 7 7"/></svg>
{{back_to_top}}</button>
<div id="lightbox" class="lightbox" hidden title="{{lightbox_close}}">
<img id="lightbox-img" src="" alt="">
</div>
<script>window.I18N = {{i18n_json}};</script>
<script>
(function () {
  var I18N = window.I18N;
  var THEME_KEY = "sorta-ui-theme";

  // --- инлайн-SVG иконки (U1: без иконочных шрифтов/эмодзи) --------------
  var ICONS = {
    folder: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" ' +
        'stroke-linecap="round" stroke-linejoin="round"><path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 ' +
        '2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7z"/><path d="M12 12v4M10 14h4"/></svg>',
    tag: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" ' +
        'stroke-linecap="round" stroke-linejoin="round"><path d="M20.6 13.4 12 22l-9-9V4a1 1 ' +
        '0 0 1 1-1h9l7.6 7.6a2 2 0 0 1 0 2.8z"/><circle cx="7.5" cy="7.5" r="1.2"/></svg>',
    merge: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" ' +
        'stroke-linecap="round" stroke-linejoin="round"><circle cx="6" cy="4.5" r="1.6"/>' +
        '<circle cx="18" cy="4.5" r="1.6"/><circle cx="12" cy="19.5" r="1.6"/>' +
        '<path d="M6 6v3c0 2.5 2 4 4 4h1M18 6v3c0 2.5-2 4-4 4h-1M12 13v5"/></svg>',
    trash: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" ' +
        'stroke-linecap="round" stroke-linejoin="round"><path d="M4 7h16M9 7V4h6v3M6 7l1 13a2 ' +
        '2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-13"/><path d="M10 11v6M14 11v6"/></svg>',
    check: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" ' +
        'stroke-linecap="round" stroke-linejoin="round"><path d="M5 12.5l4.5 4.5L19 7"/></svg>',
    spinner: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" ' +
        'stroke-linecap="round"><circle cx="12" cy="12" r="9" opacity="0.25"/>' +
        '<path d="M21 12a9 9 0 0 0-9-9"/></svg>',
    warn: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" ' +
        'stroke-linecap="round" stroke-linejoin="round"><path d="M12 3 22 20H2L12 3z"/>' +
        '<path d="M12 10v4M12 17h.01"/></svg>',
    info: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" ' +
        'stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/>' +
        '<path d="M12 8h.01M11 11.5h1v5.5h1"/></svg>',
  };

  function icon(name) {
    var tmp = document.createElement("div");
    tmp.innerHTML = ICONS[name] || "";
    var el = tmp.firstElementChild;
    if (el) el.setAttribute("aria-hidden", "true");
    return el;
  }

  // Кнопка с опциональной иконкой: variant — "primary"/"ghost"/"danger"/null.
  function makeBtn(variant, iconName, label, extraClass) {
    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "btn" + (variant ? " btn-" + variant : "") + (extraClass ? " " + extraClass : "");
    if (iconName) btn.appendChild(icon(iconName));
    btn.appendChild(document.createTextNode(label));
    return btn;
  }

  // Единый спокойный вид для пустых/загрузочных/ошибочных состояний вкладок.
  function stateEl(kind, text) {
    var div = document.createElement("div");
    div.className = "state-msg state-" + kind;
    var iconName = kind === "error" ? "warn" : kind === "loading" ? "spinner" : "info";
    var ic = icon(iconName);
    if (ic) div.appendChild(ic);
    div.appendChild(document.createTextNode(text));
    return div;
  }

  function wrapTable(table) {
    var wrap = document.createElement("div");
    wrap.className = "table-wrap";
    wrap.appendChild(table);
    return wrap;
  }

  function fmt(template, vals) {
    return template.replace(/\\{(\\w+)\\}/g, function (_, key) {
      return Object.prototype.hasOwnProperty.call(vals, key) ? vals[key] : "";
    });
  }

  function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    document.getElementById("theme-toggle-label").textContent =
        theme === "dark" ? I18N.theme_light : I18N.theme_dark;
  }

  function initTheme() {
    var saved = null;
    try { saved = window.localStorage.getItem(THEME_KEY); } catch (e) { saved = null; }
    var theme = saved || ((window.matchMedia &&
        window.matchMedia("(prefers-color-scheme: dark)").matches) ? "dark" : "light");
    applyTheme(theme);
  }

  document.getElementById("theme-toggle-btn").addEventListener("click", function () {
    var current = document.documentElement.getAttribute("data-theme") === "dark" ? "dark" : "light";
    var next = current === "dark" ? "light" : "dark";
    applyTheme(next);
    try { window.localStorage.setItem(THEME_KEY, next); } catch (e) { /* ignore */ }
  });

  initTheme();

  var LANG_KEY = "sorta_lang";
  var SUPPORTED_LANGS = ["ru", "en", "ja"];

  function urlWithLang(lang) {
    var url = new URL(window.location.href);
    url.searchParams.set("lang", lang);
    return url.toString();
  }

  function initLang() {
    var select = document.getElementById("lang-select");
    var currentLang = document.documentElement.lang;
    var saved = null;
    try { saved = window.localStorage.getItem(LANG_KEY); } catch (e) { saved = null; }
    if (saved && SUPPORTED_LANGS.indexOf(saved) !== -1 && saved !== currentLang) {
      window.location.replace(urlWithLang(saved));
      return;
    }
    if (select) {
      select.addEventListener("change", function () {
        var next = select.value;
        try { window.localStorage.setItem(LANG_KEY, next); } catch (e) { /* ignore */ }
        window.location.href = urlWithLang(next);
      });
    }
  }

  initLang();

  // F65: the "Folder language" selector (Cities tab) — the OUTPUT language of
  // folders/names, separate from the interface language. Reads the current value
  // from /api/config, and on change persists it (POST /api/config/language) and
  // re-renders the city plan preview with the new folder names.
  function initFolderLang() {
    var select = document.getElementById("folder-lang-select");
    if (!select) return;
    fetch("/api/config")
      .then(function (r) { return r.json(); })
      .then(function (cfg) { if (cfg && cfg.language) select.value = cfg.language; })
      .catch(function () { /* keep the default option */ });
    select.addEventListener("change", function () {
      var next = select.value;
      var statusEl = document.getElementById("sort-status");
      select.disabled = true;
      postJson("/api/config/language", { language: next }).then(function (resp) {
        select.disabled = false;
        if (resp && resp.ok) {
          renderPlanTab("city", "tree-city");
          if (statusEl) statusEl.textContent = I18N.folder_lang_saved;
        } else if (statusEl) {
          statusEl.textContent = (resp && resp.error) ? resp.error : "error";
        }
      }).catch(function () { select.disabled = false; });
    });
  }

  initFolderLang();

  function countFiles(node) {
    var n = node.files.length;
    Object.keys(node.children).forEach(function (k) { n += countFiles(node.children[k]); });
    return n;
  }

  // F43: число папок текущего плана — для саммари подтверждения apply.
  function countDirs(node) {
    var keys = Object.keys(node.children);
    var n = keys.length;
    keys.forEach(function (k) { n += countDirs(node.children[k]); });
    return n;
  }

  function buildTree(items) {
    var root = { files: [], children: {} };
    items.forEach(function (item) {
      var parts = (item.target_rel || "").split("/");
      parts.pop();
      var node = root;
      parts.forEach(function (part) {
        if (!node.children[part]) node.children[part] = { files: [], children: {} };
        node = node.children[part];
      });
      node.files.push(item);
    });
    return root;
  }

  // --- удаление отдельного кадра (общий путь для обеих вкладок) --------

  function deletePhoto(fileId, onSuccess) {
    var remember = document.getElementById("delete-remember").checked;
    if (!remember && !window.confirm(I18N.confirm_delete_photo)) return;
    postJson("/api/photo/trash", { file_id: fileId }).then(function (resp) {
      if (resp.trashed && resp.trashed.length) onSuccess();
    });
  }

  // Массовое удаление выбранного (общий путь _trash_files, что и одиночный).
  // onSuccess получает список реально отправленных в корзину file_id.
  function deletePhotos(fileIds, onSuccess) {
    postJson("/api/photos/trash", { file_ids: fileIds }).then(function (resp) {
      if (resp.trashed) {
        onSuccess(resp.trashed.map(function (t) { return t.file_id; }));
      }
    });
  }

  // Переиспользуемый множественный выбор + «Удалить выбранное» для любого
  // контейнера со строками, где есть чекбокс `.row-select` (value=file_id).
  // Делегирование на контейнер — работает и с лениво построенными строками.
  function wireBulkDelete(containerId, buttonId, countId) {
    var container = document.getElementById(containerId);
    var button = document.getElementById(buttonId);
    var countEl = countId ? document.getElementById(countId) : null;
    function checked() {
      return Array.prototype.slice.call(container.querySelectorAll(".row-select:checked"));
    }
    function refresh() {
      var n = checked().length;
      if (countEl) countEl.textContent = n ? " (" + n + ")" : "";
      button.disabled = n === 0;
    }
    container.addEventListener("change", function (e) {
      if (e.target && e.target.classList && e.target.classList.contains("row-select")) refresh();
    });
    button.addEventListener("click", function () {
      var boxes = checked();
      if (!boxes.length) return;
      var ids = boxes.map(function (b) { return parseInt(b.value, 10); });
      if (!window.confirm(fmt(I18N.confirm_delete_selected, { n: ids.length }))) return;
      deletePhotos(ids, function (trashedIds) {
        var done = {};
        trashedIds.forEach(function (id) { done[id] = true; });
        boxes.forEach(function (b) {
          if (done[parseInt(b.value, 10)]) {
            var tr = b.closest("tr");
            if (tr) tr.remove();
          }
        });
        refresh();
      });
    });
    refresh();
  }

  // Единое поведение превью по всему UI: клик по миниатюре (Города/Дубли/
  // Перемещения/События/Люди) открывает лайтбокс с крупным /preview, а не новую
  // вкладку с сырым /photo. samples/index позволяют листать соседние кадры (для
  // одиночных строк — [fileId]/0). thumbUrl опционален (по умолчанию /thumb/id).
  function clickableThumb(fileId, samples, index, thumbUrl) {
    var img = document.createElement("img");
    img.src = thumbUrl || ("/thumb/" + fileId);
    img.loading = "lazy";
    img.alt = "";
    img.className = "clickable-thumb";
    img.title = I18N.lightbox_open;
    img.addEventListener("click", function () {
      openLightbox(samples || [fileId], index || 0);
    });
    return img;
  }

  function renderFiles(files) {
    var table = document.createElement("table");
    files.forEach(function (item) {
      var tr = document.createElement("tr");
      var tdSelect = document.createElement("td");
      var checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.className = "row-select";
      checkbox.value = String(item.file_id);
      checkbox.title = I18N.select_for_delete;
      tdSelect.appendChild(checkbox);
      tr.appendChild(tdSelect);
      var tdThumb = document.createElement("td");
      tdThumb.appendChild(clickableThumb(item.file_id, null, 0, item.thumb_url));
      var nameEl = document.createElement("span");
      nameEl.className = "thumb-name";
      nameEl.textContent = item.name;
      tdThumb.appendChild(nameEl);
      tr.appendChild(tdThumb);
      var tdMeta = document.createElement("td");
      tdMeta.textContent = [item.date, item.geo, item.category]
          .filter(Boolean).join(" \\u00b7 ");
      tr.appendChild(tdMeta);
      var tdActions = document.createElement("td");
      var btnDelete = makeBtn("danger", "trash", I18N.delete, "btn-sm");
      btnDelete.addEventListener("click", function () {
        deletePhoto(item.file_id, function () { tr.remove(); });
      });
      tdActions.appendChild(btnDelete);
      tr.appendChild(tdActions);
      table.appendChild(tr);
    });
    return wrapTable(table);
  }

  // Ленивое построение узла дерева: содержимое папки (строки файлов + дочерние
  // папки) создаётся ТОЛЬКО при первом раскрытии <details>. План города/
  // перемещений — до тысяч кадров; строить все строки+<img> сразу подвешивало
  // вкладку. Теперь начальный рендер = только сводки папок верхнего уровня,
  // миниатюры грузятся (lazy) лишь для раскрытых папок в зоне видимости.
  function renderNode(name, node, depth, renderFilesFn) {
    var renderFn = renderFilesFn || renderFiles;
    var details = document.createElement("details");
    var summary = document.createElement("summary");
    summary.textContent = name + " (" + countFiles(node) + ")";
    details.appendChild(summary);
    var built = false;
    details.addEventListener("toggle", function () {
      if (!details.open || built) return;
      built = true;
      if (node.files.length) details.appendChild(renderFn(node.files));
      Object.keys(node.children).sort().forEach(function (childName) {
        details.appendChild(renderNode(childName, node.children[childName], depth + 1, renderFn));
      });
    });
    return details;
  }

  // F43: счётчики последнего city-плана — используются саммари подтверждения
  // apply (не отдельным превью-запросом, дерево уже загружено вкладкой).
  var cityPlanCount = 0;
  var cityPlanDirCount = 0;

  // renderPlanTab: живое дерево плана режима (city/person/event) — общий код,
  // переиспользуемый всеми план-вкладками (U2).
  function renderPlanTab(mode, containerId) {
    var container = document.getElementById(containerId);
    fetch("/api/plan?mode=" + mode)
      .then(function (r) { return r.json(); })
      .then(function (items) {
        var root = buildTree(items);
        if (mode === "city") {
          cityPlanCount = items.length;
          cityPlanDirCount = countDirs(root);
        }
        container.textContent = "";
        if (root.files.length) container.appendChild(renderFiles(root.files));
        Object.keys(root.children).sort().forEach(function (name) {
          container.appendChild(renderNode(name, root.children[name], 0));
        });
      })
      .catch(function (err) {
        container.textContent = "";
        container.appendChild(stateEl("error", I18N.error_loading_plan + err));
      });
  }

  renderPlanTab("city", "tree-city");
  wireBulkDelete("tree-city", "city-delete-selected-btn", "city-delete-selected-count");

  document.querySelectorAll(".expand-all-btn").forEach(function (btn) {
    btn.addEventListener("click", function () {
      document.querySelectorAll("details").forEach(function (d) { d.open = true; });
    });
  });
  document.querySelectorAll(".collapse-all-btn").forEach(function (btn) {
    btn.addEventListener("click", function () {
      document.querySelectorAll("details").forEach(function (d) { d.open = false; });
    });
  });
  document.getElementById("top-btn").addEventListener("click", function () {
    window.scrollTo({ top: 0, behavior: "smooth" });
  });

  // --- вкладки ---------------------------------------------------------

  var dupesLoaded = false;
  var movesLoaded = false;
  var clustersLoaded = false;
  var eventsLoaded = false;

  function activateTab(name) {
    ["process", "city", "dupes", "person", "event", "moves"].forEach(function (t) {
      document.getElementById("tab-btn-" + t).classList.toggle("active", t === name);
      document.getElementById("tab-" + t).classList.toggle("active", t === name);
    });
    // #36: чекбокс «не спрашивать удаление» релевантен только там, где удаляют
    // (Города/Дубли) — на остальных вкладках это шум, прячем.
    document.getElementById("delete-remember-row").style.display =
        (name === "city" || name === "dupes") ? "" : "none";
    if (name === "dupes" && !dupesLoaded) {
      dupesLoaded = true;
      loadDupes();
    }
    if (name === "event" && !eventsLoaded) {
      eventsLoaded = true;
      loadEvents();
    }
    if (name === "person" && !clustersLoaded) {
      clustersLoaded = true;
      loadClusters();
    }
    if (name === "moves" && !movesLoaded) {
      movesLoaded = true;
      loadMoves();
    }
  }

  ["process", "city", "dupes", "person", "event", "moves"].forEach(function (t) {
    document.getElementById("tab-btn-" + t).addEventListener("click", function () {
      activateTab(t);
    });
  });

  // F54: «Люди»/«События» скрыты по умолчанию (без мигания) и раскрываются
  // по факту наличия данных в БД (вариант B, stateless) — фетч дешёвых
  // EXISTS-проверок, вызывается при инициализации и после каждого прогона
  // (refreshTabsAfterProcess), т.к. прогон мог впервые породить кластеры/события.
  function applyTabVisibility() {
    fetch("/api/tabs/visibility")
      .then(function (r) { return r.json(); })
      .then(function (data) {
        document.getElementById("tab-btn-person").style.display =
            data.person ? "" : "none";
        document.getElementById("tab-btn-event").style.display =
            data.event ? "" : "none";
        var activeBtn = document.querySelector(".tab-btn.active");
        var activeName = activeBtn ? activeBtn.id.replace("tab-btn-", "") : null;
        if ((activeName === "person" && !data.person) ||
            (activeName === "event" && !data.event)) {
          activateTab("process");
        }
      })
      .catch(function () {});
  }

  applyTabVisibility();

  // --- вкладка «Обработать» (F36: запуск пайплайна из веба + polling) ----

  // F57: чекбоксы deep/geo-online должны стартовать по факту config.yaml
  // (cfg.naming.vlm_enabled / cfg.geo.provider), а не всегда пустыми — иначе
  // сложно понять, что реально включено, и нельзя увидеть текущее состояние
  // до первого клика. vlmAvailable — установлен ли пакет transformers;
  // приглушённая пометка «VLM не установлен» показывается только когда
  // чекбокс отмечен, но пакета нет (запрос VLM ≠ его реальный запуск —
  // junk.classify штатно фолбэчит на CLIP).
  var vlmAvailable = true;

  function updateVlmMissingWarning() {
    var checked = document.getElementById("process-deep-checkbox").checked;
    document.getElementById("process-deep-vlm-missing").style.display =
        (checked && !vlmAvailable) ? "" : "none";
  }

  function applyProcessDefaults() {
    fetch("/api/process/defaults")
      .then(function (r) { return r.json(); })
      .then(function (data) {
        document.getElementById("process-deep-checkbox").checked = !!data.deep;
        document.getElementById("process-geo-online-checkbox").checked = !!data.geo_online;
        vlmAvailable = !!data.vlm_available;
        updateVlmMissingWarning();
      })
      .catch(function () {});
  }

  applyProcessDefaults();
  document.getElementById("process-deep-checkbox")
      .addEventListener("change", updateVlmMissingWarning);

  // F64: баннер о CPU-профиле (обработка на процессоре — медленно для лиц/VLM).
  fetch("/api/env").then(function (r) { return r.json(); })
    .then(function (data) {
      if (data && !data.gpu_profile) {
        document.getElementById("env-cpu-warning").style.display = "";
      }
    }).catch(function () {});

  var PROCESS_POLL_MS = 1500;
  var processPollTimer = null;

  function processStageLabel(stage) {
    return stage ? (I18N["process_stage_" + stage] || stage) : "";
  }

  // Чипы-этапы (F41): done/now/pending по стадиям пайплайна — тот же порядок,
  // что и сервер (_PIPELINE_STAGE_NAMES), только для отображения. F53/#39:
  // faces/events opt-in — currentProcessStages фиксируется по чекбоксам в
  // момент запуска (сервер фильтрует steps так же), иначе индексы чипов
  // разъедутся со stage_index отфильтрованного прогона.
  var ALL_PROCESS_STAGES = ["index", "geo", "landmarks", "faces", "events", "junk", "phash"];
  var OPTIONAL_PROCESS_STAGES = { faces: true, events: true };
  var currentProcessStages = ALL_PROCESS_STAGES.slice();

  function filterProcessStages(faces, events) {
    var enabled = { faces: faces, events: events };
    return ALL_PROCESS_STAGES.filter(function (name) {
      return !OPTIONAL_PROCESS_STAGES[name] || enabled[name];
    });
  }

  // F62/F63: «Дозапустить выбранное» — в отличие от filterProcessStages
  // (базовые + включённые опциональные), здесь ТОЛЬКО выбранное: faces/events
  // по флагам + junk при deep (переклассификация с VLM). Базовые index/geo/
  // landmarks/phash сервер не запускает. Порядок из ALL_PROCESS_STAGES.
  function filterRerunStages(faces, events, deep) {
    var enabled = { faces: faces, events: events, junk: deep };
    return ALL_PROCESS_STAGES.filter(function (name) { return enabled[name]; });
  }

  function rerunSelectedAllowed() {
    return document.getElementById("process-faces-checkbox").checked ||
        document.getElementById("process-events-checkbox").checked ||
        document.getElementById("process-deep-checkbox").checked;
  }

  function updateRerunSelectedDisabled() {
    document.getElementById("process-rerun-optional-btn").disabled = !rerunSelectedAllowed();
  }

  ["process-faces-checkbox", "process-events-checkbox", "process-deep-checkbox"]
      .forEach(function (id) {
        document.getElementById(id).addEventListener("change", updateRerunSelectedDisabled);
      });
  updateRerunSelectedDisabled();

  function renderStageChips(data) {
    var container = document.getElementById("process-stages");
    container.textContent = "";
    if (!data.running && !data.finished) return;
    var success = !data.running && data.finished && !data.error;
    currentProcessStages.forEach(function (name, idx) {
      var stepIndex = idx + 1;
      var cls = "pending";
      if (success || stepIndex < data.stage_index) cls = "done";
      else if (data.running && stepIndex === data.stage_index) cls = "now";
      var chip = document.createElement("span");
      chip.className = "stage-chip " + cls;
      if (cls === "done") chip.appendChild(icon("check"));
      chip.appendChild(document.createTextNode(processStageLabel(name)));
      container.appendChild(chip);
    });
  }

  function refreshTabsAfterProcess() {
    dupesLoaded = false;
    clustersLoaded = false;
    eventsLoaded = false;
    movesLoaded = false;
    renderPlanTab("city", "tree-city");
    applyTabVisibility();
  }

  function renderProcessStatus(data) {
    var startBtn = document.getElementById("process-start-btn");
    var cancelBtn = document.getElementById("process-cancel-btn");
    var rerunBtn = document.getElementById("process-rerun-optional-btn");
    var bar = document.getElementById("process-progress");
    var statusEl = document.getElementById("process-status");
    startBtn.disabled = !!data.running;
    rerunBtn.disabled = !!data.running || !rerunSelectedAllowed();
    cancelBtn.style.display = data.running ? "" : "none";
    cancelBtn.disabled = !!data.cancel_requested;
    bar.style.display = data.running ? "" : "none";
    if (!data.running) bar.classList.remove("indeterminate");
    renderStageChips(data);
    if (data.running) {
      if (data.cancel_requested) {
        // отмена запрошена — показываем фидбэк, пока стадия прерывается/дорабатывает
        bar.classList.add("indeterminate");
        bar.max = 1;
        bar.removeAttribute("value");
        statusEl.textContent = I18N.process_cancel_requested;
        return;
      }
      // #37: total>0 -> определённый прогресс (заполняется); total<=0 (индексация,
      // total неизвестен) -> бегущая indeterminate-полоса + «обработано X».
      if (data.total > 0) {
        bar.classList.remove("indeterminate");
        bar.max = data.total;
        bar.value = data.done || 0;
      } else {
        bar.classList.add("indeterminate");
        bar.max = 1;
        bar.removeAttribute("value");
      }
      statusEl.textContent = fmt(
        data.total > 0 ? I18N.process_stage_progress : I18N.process_stage_progress_indeterminate, {
        stage: processStageLabel(data.stage),
        index: data.stage_index,
        total: data.stage_total,
        done: data.done,
        all: data.total,
      });
      return;
    }
    if (!data.finished) {
      statusEl.textContent = "";
      return;
    }
    if (data.error) {
      statusEl.textContent = I18N.process_error_prefix + data.error;
    } else if (data.cancel_requested) {
      statusEl.textContent = I18N.process_cancelled;
      refreshTabsAfterProcess();
    } else {
      statusEl.textContent = I18N.process_done;
      refreshTabsAfterProcess();
    }
  }

  function pollProcessStatus() {
    fetch("/api/process/status")
      .then(function (r) { return r.json(); })
      .then(function (data) {
        renderProcessStatus(data);
        if (data.running) {
          processPollTimer = setTimeout(pollProcessStatus, PROCESS_POLL_MS);
        }
      });
  }

  document.getElementById("process-start-btn").addEventListener("click", function () {
    var input = document.getElementById("process-source-dir");
    var path = input.value.trim();
    if (!path) { window.alert(I18N.process_enter_path); return; }
    var deep = document.getElementById("process-deep-checkbox").checked;
    var geoOnline = document.getElementById("process-geo-online-checkbox").checked;
    var faces = document.getElementById("process-faces-checkbox").checked;
    var events = document.getElementById("process-events-checkbox").checked;
    currentProcessStages = filterProcessStages(faces, events);
    postJson("/api/process", {
      source_dir: path, deep: deep, geo_online: geoOnline, faces: faces, events: events,
    }).then(function (resp) {
      if (resp && resp.error) {
        document.getElementById("process-status").textContent =
            I18N.process_start_error_prefix + resp.error;
        return;
      }
      if (processPollTimer) clearTimeout(processPollTimer);
      pollProcessStatus();
    });
  });

  document.getElementById("process-rerun-optional-btn").addEventListener("click", function () {
    var faces = document.getElementById("process-faces-checkbox").checked;
    var events = document.getElementById("process-events-checkbox").checked;
    var deep = document.getElementById("process-deep-checkbox").checked;
    currentProcessStages = filterRerunStages(faces, events, deep);
    postJson("/api/process/rerun-optional", { faces: faces, events: events, deep: deep })
        .then(function (resp) {
      if (resp && resp.error) {
        document.getElementById("process-status").textContent =
            I18N.process_start_error_prefix + resp.error;
        return;
      }
      if (processPollTimer) clearTimeout(processPollTimer);
      pollProcessStatus();
    });
  });

  document.getElementById("process-browse-btn").addEventListener("click", function () {
    postJson("/api/browse", {}).then(function (resp) {
      if (resp && resp.path) {
        document.getElementById("process-source-dir").value = resp.path;
      }
    });
  });

  document.getElementById("process-cancel-btn").addEventListener("click", function () {
    this.disabled = true;  // мгновенный фидбэк, не ждём следующего polling-тика
    document.getElementById("process-status").textContent = I18N.process_cancel_requested;
    postJson("/api/process/cancel", {});
  });

  document.getElementById("process-reset-btn").addEventListener("click", function () {
    if (!window.confirm(I18N.process_reset_confirm)) return;
    postJson("/api/process/reset", {}).then(function (resp) {
      var statusEl = document.getElementById("process-status");
      if (resp && resp.error) {
        statusEl.textContent = I18N.process_reset_error_prefix + resp.error;
        return;
      }
      statusEl.textContent = I18N.process_reset_done;
      refreshTabsAfterProcess();
    });
  });

  pollProcessStatus();

  // --- вкладка «Города»: apply раскладки (F43) ----------------------------
  // Дерево-превью вкладки — уже dry-run; кнопка сразу открывает подтверждение
  // (текст зависит от режима/dest), только потом POST /api/sort. Фон +
  // прогресс — тот же паттерн polling, что и «Обработать» (F36) выше.

  var SORT_POLL_MS = 1500;
  var sortPollTimer = null;

  function updateSortApplyBtnStyle() {
    var btn = document.getElementById("sort-apply-btn");
    var checked = document.querySelector('input[name="sort-mode"]:checked');
    var move = !checked || checked.value === "move";
    btn.classList.toggle("btn-danger", move);
    btn.classList.toggle("btn-primary", !move);
  }

  document.querySelectorAll('input[name="sort-mode"]').forEach(function (r) {
    r.addEventListener("change", updateSortApplyBtnStyle);
  });
  updateSortApplyBtnStyle();

  function sortConfirmText(dest, mode) {
    var destLabel = dest || I18N.sort_dest_inplace_label;
    var text = fmt(I18N.sort_confirm_summary,
        { n: cityPlanCount, dirs: cityPlanDirCount, dest: destLabel });
    if (!dest) text += "\\n" + I18N.sort_confirm_inplace;
    else if (mode === "move") text += "\\n" + I18N.sort_confirm_move;
    else text += "\\n" + I18N.sort_confirm_copy;
    return text;
  }

  function renderSortStatus(data) {
    var btn = document.getElementById("sort-apply-btn");
    var bar = document.getElementById("sort-progress");
    var statusEl = document.getElementById("sort-status");
    var warnEl = document.getElementById("sort-warning");
    btn.disabled = !!data.running;
    bar.style.display = data.running ? "" : "none";
    if (data.running) {
      bar.max = data.total || 0;
      bar.value = data.done || 0;
      statusEl.textContent = fmt(I18N.sort_progress_line, { done: data.done, all: data.total });
      warnEl.textContent = "";
      return;
    }
    if (!data.finished) { statusEl.textContent = ""; warnEl.textContent = ""; return; }
    if (data.error) {
      statusEl.textContent = I18N.sort_error_prefix + data.error;
      warnEl.textContent = "";
      return;
    }
    var r = data.result || {};
    statusEl.textContent = fmt(I18N.sort_done_text,
        { n: r.moved || 0, f: r.failed || 0, p: r.skipped_in_place || 0 });
    warnEl.textContent = r.preview_stale ? I18N.sort_preview_stale_warning : "";
    movesLoaded = false;
    renderPlanTab("city", "tree-city");
  }

  function pollSortStatus() {
    fetch("/api/sort/status")
      .then(function (r) { return r.json(); })
      .then(function (data) {
        renderSortStatus(data);
        if (data.running) sortPollTimer = setTimeout(pollSortStatus, SORT_POLL_MS);
      });
  }

  document.getElementById("sort-apply-btn").addEventListener("click", function () {
    var dest = document.getElementById("sort-dest").value.trim();
    var checked = document.querySelector('input[name="sort-mode"]:checked');
    var mode = checked ? checked.value : "move";
    if (!window.confirm(sortConfirmText(dest, mode))) return;
    postJson("/api/sort", { dest: dest || null, mode: mode }).then(function (resp) {
      if (resp && resp.error) {
        document.getElementById("sort-status").textContent =
            I18N.sort_start_error_prefix + resp.error;
        return;
      }
      if (sortPollTimer) clearTimeout(sortPollTimer);
      pollSortStatus();
    });
  });

  document.getElementById("sort-browse-btn").addEventListener("click", function () {
    postJson("/api/browse", {}).then(function (resp) {
      if (resp && resp.path) {
        document.getElementById("sort-dest").value = resp.path;
      }
    });
  });

  // Дефолт пути назначения = <источник>_sorted (сервер знает источник); только
  // если пользователь ещё ничего не ввёл — свой ввод не затираем.
  fetch("/api/sort/suggest-dest").then(function (r) { return r.json(); })
    .then(function (resp) {
      var input = document.getElementById("sort-dest");
      if (resp && resp.dest && !input.value.trim()) input.value = resp.dest;
    }).catch(function () {});

  pollSortStatus();

  // --- вкладка «Перемещения» (U5, read-only манифест sort --apply) -------

  var MOVE_STATUS_LABELS = {
    planned: I18N.status_planned, done: I18N.status_done, undone: I18N.status_undone,
    failed: I18N.status_failed, deleted: I18N.status_deleted,
  };

  function moveStatusLabel(status) {
    return MOVE_STATUS_LABELS[status] || status;
  }

  var MOVE_STATUS_CHIP_CLASS = {
    done: "chip-good", planned: "chip-accent", failed: "chip-danger", deleted: "chip-danger",
    undone: "chip",
  };

  function renderMoveFiles(files) {
    var table = document.createElement("table");
    files.forEach(function (item) {
      var tr = document.createElement("tr");
      var tdThumb = document.createElement("td");
      tdThumb.appendChild(clickableThumb(item.file_id, null, 0, item.thumb_url));
      var nameEl = document.createElement("span");
      nameEl.className = "thumb-name";
      nameEl.textContent = item.name;
      tdThumb.appendChild(nameEl);
      tr.appendChild(tdThumb);
      var tdMeta = document.createElement("td");
      var pathLine = document.createElement("div");
      pathLine.textContent = item.src + " → " + item.dst;
      tdMeta.appendChild(pathLine);
      var statusChip = document.createElement("span");
      statusChip.className = "chip " + (MOVE_STATUS_CHIP_CLASS[item.status] || "chip");
      statusChip.textContent = moveStatusLabel(item.status);
      tdMeta.appendChild(statusChip);
      tr.appendChild(tdMeta);
      table.appendChild(tr);
    });
    return wrapTable(table);
  }

  function batchSummaryText(batch, count) {
    var parts = [I18N.batch_label + " #" + batch.id, batch.mode, batch.operation || "move",
        I18N.started_label + " " + batch.started_at];
    parts.push(batch.finished_at ? I18N.finished_label + " " + batch.finished_at
        : I18N.in_progress_label);
    parts.push(I18N.files_count_label + ": " + count);
    return parts.join(" · ");
  }

  function loadMoves() {
    var container = document.getElementById("tree-moves");
    var summary = document.getElementById("moves-summary");
    fetch("/api/moves")
      .then(function (r) { return r.json(); })
      .then(function (data) {
        container.textContent = "";
        summary.textContent = "";
        if (!data.batch) {
          summary.appendChild(stateEl("empty", I18N.no_moves_yet));
          return;
        }
        summary.textContent = batchSummaryText(data.batch, data.moves.length);
        var root = buildTree(data.moves);
        if (root.files.length) container.appendChild(renderMoveFiles(root.files));
        Object.keys(root.children).sort().forEach(function (name) {
          container.appendChild(renderNode(name, root.children[name], 0, renderMoveFiles));
        });
      })
      .catch(function (err) {
        container.textContent = "";
        container.appendChild(stateEl("error", I18N.error_loading_moves + err));
      });
  }

  // --- альбомы (F35): кнопка «Собрать в папку» на карточках Люди/События ---

  function albumModeSelect() {
    var select = document.createElement("select");
    ["link", "copy", "move"].forEach(function (m) {
      var opt = document.createElement("option");
      opt.value = m;
      opt.textContent = I18N["album_mode_" + m];
      select.appendChild(opt);
    });
    return select;
  }

  // Поле пути назначения альбома + «Обзор…» (F60, тот же мотив, что и
  // sort-dest/process-source-dir): дефолт = <источник>_sorted с сервера,
  // префилл только если поле ещё пустое (свой ввод не затираем).
  function appendAlbumDestControls(box) {
    var input = document.createElement("input");
    input.type = "text";
    input.className = "album-dest-input";
    input.placeholder = I18N.album_dest_placeholder;
    box.appendChild(input);
    var browseBtn = makeBtn("ghost", null, I18N.process_browse_button, "btn-sm album-browse-btn");
    browseBtn.addEventListener("click", function () {
      postJson("/api/browse", {}).then(function (resp) {
        if (resp && resp.path) input.value = resp.path;
      });
    });
    box.appendChild(browseBtn);
    fetch("/api/sort/suggest-dest").then(function (r) { return r.json(); })
      .then(function (resp) {
        if (resp && resp.dest && !input.value.trim()) input.value = resp.dest;
      }).catch(function () {});
    return input;
  }

  function albumPreviewText(resp) {
    var txt = fmt(I18N.album_preview_text, { n: resp.count, dest: resp.dest });
    if (resp.mode === "move" && resp.blocked_multi) {
      txt += fmt(I18N.album_blocked_text, { k: resp.blocked_multi });
    }
    return txt;
  }

  // Превью (apply=false) -> подтверждение (текст зависит от режима, move явно
  // предупреждает об изъятии из пула) -> apply=true. statusEl получает
  // прогресс/результат; при успешном apply сбрасывается кэш вкладки
  // «Перемещения», чтобы следующий заход её перезагрузил (F35 п.4).
  function gatherAlbum(kind, selector, mode, where, name, dest, statusEl) {
    var body = { kind: kind, selector: selector, mode: mode, apply: false };
    if (where) body.where = [where];
    if (name) body.name = name;
    if (dest) body.dest = dest;
    statusEl.textContent = I18N.album_in_progress;
    postJson("/api/album", body).then(function (resp) {
      if (resp.error) { statusEl.textContent = resp.error; return; }
      var confirmMsg = albumPreviewText(resp) + "\\n" +
          (mode === "move" ? I18N.album_confirm_move : I18N.album_confirm_generic);
      if (!window.confirm(confirmMsg)) { statusEl.textContent = ""; return; }
      body.apply = true;
      statusEl.textContent = I18N.album_in_progress;
      postJson("/api/album", body).then(function (resp2) {
        if (resp2.error) { statusEl.textContent = resp2.error; return; }
        statusEl.textContent = fmt(I18N.album_result_text,
            { n: resp2.transferred, f: resp2.failed });
        movesLoaded = false;
      });
    });
  }

  // --- лайтбокс (F42): один переиспользуемый оверлей поверх /photo/<id> ---
  // Заполняется по клику (не N скрытых оверлеев). Клик по фону/Esc закрывает;
  // стрелки ←/→ листают переданный список sample-кадров (опц., F42).

  var lightboxEl = document.getElementById("lightbox");
  var lightboxImg = document.getElementById("lightbox-img");
  var lightboxSamples = null;
  var lightboxIndex = 0;

  function showLightboxAt(index) {
    lightboxIndex = index;
    // /preview — крупный ДЕКОДИРОВАННЫЙ JPEG (HEIC/RAW рендерятся), не сырой /photo
    lightboxImg.src = "/preview/" + lightboxSamples[index];
  }

  function openLightbox(samples, index) {
    lightboxSamples = samples;
    showLightboxAt(index);
    lightboxEl.hidden = false;
  }

  function closeLightbox() {
    lightboxEl.hidden = true;
    lightboxImg.src = "";
    lightboxSamples = null;
  }

  lightboxEl.addEventListener("click", closeLightbox);
  lightboxImg.addEventListener("click", function (e) { e.stopPropagation(); });
  document.addEventListener("keydown", function (e) {
    if (lightboxEl.hidden) return;
    if (e.key === "Escape") { closeLightbox(); return; }
    if (!lightboxSamples || lightboxSamples.length < 2) return;
    if (e.key === "ArrowRight") showLightboxAt((lightboxIndex + 1) % lightboxSamples.length);
    else if (e.key === "ArrowLeft") {
      showLightboxAt((lightboxIndex - 1 + lightboxSamples.length) % lightboxSamples.length);
    }
  });

  // --- вкладка «Люди» (F31, управление кластерами лиц) --------------------

  var clustersById = {};
  var selectedForMerge = {};
  var selectedForMergeCount = 0;

  function updateMergeButton() {
    document.getElementById("clusters-merge-btn").disabled = selectedForMergeCount !== 2;
  }

  function toggleMergeSelection(clusterId, checked) {
    if (checked) {
      if (!(clusterId in selectedForMerge)) selectedForMergeCount += 1;
      selectedForMerge[clusterId] = true;
    } else {
      if (clusterId in selectedForMerge) selectedForMergeCount -= 1;
      delete selectedForMerge[clusterId];
    }
    updateMergeButton();
  }

  function renderClusterCard(c) {
    var card = document.createElement("div");
    card.className = "card" + (c.label ? " named" : "");

    var thumbs = document.createElement("div");
    thumbs.className = "cluster-thumbs";
    // Скелетон рисуется сразу (карточка отзывчива, пока идёт /thumb) —
    // сама миниатюра грузится лениво и фоном; onload плавно проявляет её и
    // снимает скелетон-заглушку (F42).
    c.samples.forEach(function (fileId, idx) {
      var skel = document.createElement("div");
      skel.className = "thumb-skel";
      var img = document.createElement("img");
      img.loading = "lazy";
      img.alt = "";
      img.addEventListener("load", function () { skel.className = "thumb-skel loaded"; });
      img.addEventListener("click", function () { openLightbox(c.samples, idx); });
      img.src = "/thumb/" + fileId;
      skel.appendChild(img);
      thumbs.appendChild(skel);
    });
    card.appendChild(thumbs);

    var meta = document.createElement("div");
    meta.className = "cluster-meta";
    meta.textContent = (c.label ? c.label : I18N.unnamed) + " \\u00b7 " + c.size + " " +
        I18N.faces_unit;
    card.appendChild(meta);

    var form = document.createElement("div");
    form.className = "cluster-name-form";
    var input = document.createElement("input");
    input.type = "text";
    input.value = c.label || "";
    input.placeholder = I18N.person_name_placeholder;
    form.appendChild(input);
    var btnName = makeBtn("primary", "tag", I18N.name_button, "btn-sm");
    btnName.addEventListener("click", function () {
      var name = input.value.trim();
      if (!name) { window.alert(I18N.alert_enter_name); return; }
      postJson("/api/clusters/label", { cluster_id: c.cluster_id, name: name })
        .then(function (resp) { if (resp && resp.ok) loadClusters(); });
    });
    form.appendChild(btnName);
    card.appendChild(form);

    var mergeLabel = document.createElement("label");
    mergeLabel.className = "cluster-merge-select";
    var checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.addEventListener("change", function () {
      toggleMergeSelection(c.cluster_id, checkbox.checked);
    });
    mergeLabel.appendChild(checkbox);
    mergeLabel.appendChild(document.createTextNode(" " + I18N.select_for_merge));
    card.appendChild(mergeLabel);

    if (c.label) {
      var albumBox = document.createElement("div");
      albumBox.className = "album-controls";
      var modeSelect = albumModeSelect();
      albumBox.appendChild(modeSelect);
      var destInput = appendAlbumDestControls(albumBox);
      var whereInput = document.createElement("input");
      whereInput.type = "text";
      whereInput.placeholder = I18N.album_where_placeholder;
      albumBox.appendChild(whereInput);
      var albumBtn = makeBtn("primary", "folder", I18N.album_button, "btn-sm album-gather-btn");
      var albumStatus = document.createElement("span");
      albumStatus.className = "album-status";
      albumBtn.addEventListener("click", function () {
        var where = whereInput.value.trim();
        gatherAlbum("person", c.label, modeSelect.value, where || null, null,
            destInput.value.trim() || null, albumStatus);
      });
      albumBox.appendChild(albumBtn);
      albumBox.appendChild(albumStatus);
      card.appendChild(albumBox);
    } else {
      var hint = document.createElement("div");
      hint.className = "album-hint";
      hint.textContent = I18N.album_name_first_hint;
      card.appendChild(hint);
    }

    return card;
  }

  function loadClusters() {
    var container = document.getElementById("clusters-grid");
    fetch("/api/clusters")
      .then(function (r) { return r.json(); })
      .then(function (clusters) {
        container.textContent = "";
        clustersById = {};
        selectedForMerge = {};
        selectedForMergeCount = 0;
        updateMergeButton();
        if (!clusters.length) {
          container.appendChild(stateEl("empty", I18N.no_clusters));
          return;
        }
        var named = clusters.filter(function (c) { return c.label; });
        var unnamed = clusters.filter(function (c) { return !c.label; });
        named.concat(unnamed).forEach(function (c) {
          clustersById[c.cluster_id] = c;
          container.appendChild(renderClusterCard(c));
        });
      })
      .catch(function (err) {
        container.textContent = "";
        container.appendChild(stateEl("error", I18N.error_loading_clusters + err));
      });
  }

  document.getElementById("clusters-merge-btn").addEventListener("click", function () {
    var ids = Object.keys(selectedForMerge).map(Number);
    if (ids.length !== 2) return;
    var a = clustersById[ids[0]];
    var b = clustersById[ids[1]];
    var dst = a.size >= b.size ? a.cluster_id : b.cluster_id;
    var src = dst === a.cluster_id ? b.cluster_id : a.cluster_id;
    postJson("/api/clusters/merge", { src: src, dst: dst })
      .then(function (resp) { if (resp && resp.ok) loadClusters(); });
  });

  // --- вкладка «События» (F35: список событий + «Собрать в папку») --------

  function renderEventCard(e) {
    var card = document.createElement("div");
    card.className = "card";

    var meta = document.createElement("div");
    meta.className = "event-meta";
    meta.textContent = e.count + " " + I18N.files_count_label + " \\u00b7 " +
        [e.started_at, e.ended_at].filter(Boolean).join(" \\u2013 ");
    card.appendChild(meta);

    // превью-кадры события (клик -> лайтбокс, стрелки листают кадры события)
    if (e.samples && e.samples.length) {
      var thumbs = document.createElement("div");
      thumbs.className = "event-thumbs";
      e.samples.forEach(function (fileId, idx) {
        thumbs.appendChild(clickableThumb(fileId, e.samples, idx));
      });
      card.appendChild(thumbs);
    }

    var nameInput = document.createElement("input");
    nameInput.type = "text";
    nameInput.className = "event-name-input";
    nameInput.value = e.name || "";
    nameInput.placeholder = I18N.album_name_placeholder;
    card.appendChild(nameInput);

    var albumBox = document.createElement("div");
    albumBox.className = "album-controls";
    var modeSelect = albumModeSelect();
    albumBox.appendChild(modeSelect);
    var destInput = appendAlbumDestControls(albumBox);
    var albumBtn = makeBtn("primary", "folder", I18N.album_button, "btn-sm album-gather-btn");
    var albumStatus = document.createElement("span");
    albumStatus.className = "album-status";
    albumBtn.addEventListener("click", function () {
      var name = nameInput.value.trim();
      gatherAlbum("event", String(e.id), modeSelect.value, null, name || null,
          destInput.value.trim() || null, albumStatus);
    });
    albumBox.appendChild(albumBtn);
    albumBox.appendChild(albumStatus);
    card.appendChild(albumBox);

    return card;
  }

  function loadEvents() {
    var container = document.getElementById("events-list");
    fetch("/api/events")
      .then(function (r) { return r.json(); })
      .then(function (events) {
        container.textContent = "";
        if (!events.length) {
          container.appendChild(stateEl("empty", I18N.no_events));
          return;
        }
        events.forEach(function (e) { container.appendChild(renderEventCard(e)); });
      })
      .catch(function (err) {
        container.textContent = "";
        container.appendChild(stateEl("error", I18N.error_loading_events + err));
      });
  }

  // --- вкладка «Дубли» ---------------------------------------------------

  function postJson(url, data) {
    return fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    }).then(function (r) { return r.json(); });
  }

  var currentGroups = [];

  function groupFileIds(g) {
    return g.frames.map(function (f) { return f.file_id; });
  }

  function selectedKeeper(g) {
    var radios = document.getElementsByName("keep-" + g.group);
    for (var i = 0; i < radios.length; i++) {
      if (radios[i].checked) return parseInt(radios[i].value, 10);
    }
    return null;
  }

  function groupSkipped(g) {
    var checkbox = document.getElementById("skip-" + g.group);
    return !!(checkbox && checkbox.checked);
  }

  function actionLabel(action) {
    if (action === "keep") return I18N.action_keep;
    if (action === "to_delete") return I18N.action_to_delete;
    return "";
  }

  function renderGroup(g) {
    var box = document.createElement("div");
    box.className = "card dupe-group";

    var title = document.createElement("h3");
    title.textContent = fmt(I18N.group_title, { n: g.group + 1, count: g.frames.length });
    box.appendChild(title);

    var table = document.createElement("table");
    // клик по кадру группы -> лайтбокс; стрелки листают кадры этого дубль-набора
    var groupSamples = g.frames.map(function (fr) { return fr.file_id; });
    g.frames.forEach(function (f, frameIdx) {
      var tr = document.createElement("tr");

      var tdRadio = document.createElement("td");
      var radio = document.createElement("input");
      radio.type = "radio";
      radio.name = "keep-" + g.group;
      radio.value = String(f.file_id);
      radio.checked = f.action === "keep" || (!f.action && f.recommended);
      tdRadio.appendChild(radio);
      tr.appendChild(tdRadio);

      var tdThumb = document.createElement("td");
      tdThumb.appendChild(clickableThumb(f.file_id, groupSamples, frameIdx, f.thumb_url));
      var nameEl = document.createElement("span");
      nameEl.className = "thumb-name";
      nameEl.textContent = f.name;
      tdThumb.appendChild(nameEl);
      if (f.recommended) {
        var badge = document.createElement("span");
        badge.className = "badge";
        badge.appendChild(icon("check"));
        badge.appendChild(document.createTextNode(I18N.recommended_badge));
        tdThumb.appendChild(badge);
      }
      tr.appendChild(tdThumb);

      var tdMeta = document.createElement("td");
      var dims = f.width && f.height ? f.width + "×" + f.height : "?";
      var kb = Math.round((f.size || 0) / 1024) + " KB";
      tdMeta.textContent = [dims, kb, actionLabel(f.action)].filter(Boolean).join(" · ");
      tr.appendChild(tdMeta);

      var tdActions = document.createElement("td");
      var btnFrameDelete = makeBtn("danger", "trash", I18N.delete, "btn-sm");
      btnFrameDelete.addEventListener("click", function () {
        deletePhoto(f.file_id, function () { tr.remove(); });
      });
      tdActions.appendChild(btnFrameDelete);
      tr.appendChild(tdActions);

      table.appendChild(tr);
    });
    box.appendChild(wrapTable(table));

    var skipLabel = document.createElement("label");
    skipLabel.className = "skip-label";
    var skipCheckbox = document.createElement("input");
    skipCheckbox.type = "checkbox";
    skipCheckbox.id = "skip-" + g.group;
    skipLabel.appendChild(skipCheckbox);
    skipLabel.appendChild(document.createTextNode(" " + I18N.skip_group_label));
    box.appendChild(skipLabel);

    var btnTrash = makeBtn("danger", "trash", I18N.delete_dupes_button);
    btnTrash.addEventListener("click", function () {
      var keep = selectedKeeper(g);
      if (keep === null) { window.alert(I18N.alert_choose_keeper); return; }
      var remember = document.getElementById("delete-remember").checked;
      if (!remember && !window.confirm(fmt(I18N.confirm_trash_group, { n: g.group + 1 }))) {
        return;
      }
      postJson("/api/dupes/trash", { group: groupFileIds(g), keep_file_id: keep })
        .then(loadDupes);
    });
    box.appendChild(btnTrash);

    return box;
  }

  function loadDupes() {
    document.getElementById("dupes-save-status").textContent = "";
    fetch("/api/dupes")
      .then(function (r) { return r.json(); })
      .then(function (groups) {
        currentGroups = groups;
        var container = document.getElementById("dupes-list");
        container.textContent = "";
        if (!groups.length) {
          container.appendChild(stateEl("empty", I18N.no_dupes));
          return;
        }
        groups.forEach(function (g) { container.appendChild(renderGroup(g)); });
      })
      .catch(function (err) {
        var container = document.getElementById("dupes-list");
        container.textContent = "";
        container.appendChild(stateEl("error", I18N.error_loading_dupes + err));
      });
  }

  document.getElementById("dupes-save-all-btn").addEventListener("click", function () {
    var statusEl = document.getElementById("dupes-save-status");
    var groups = [];
    var skip = [];
    currentGroups.forEach(function (g) {
      if (groupSkipped(g)) {
        skip.push(groupFileIds(g));
        return;
      }
      var keep = selectedKeeper(g);
      if (keep === null) return;
      groups.push({ group: groupFileIds(g), keep_file_id: keep });
    });
    if (!groups.length) {
      statusEl.textContent = I18N.select_group_to_save;
      return;
    }
    postJson("/api/dupes/choices", { groups: groups, skip: skip }).then(function (resp) {
      if (resp && typeof resp.saved === "number") {
        statusEl.textContent = fmt(I18N.saved_groups, { n: resp.saved });
      }
      loadDupes();
    });
  });
})();
</script>
</body></html>
"""


def _render_index_html(lang: i18n.Lang) -> str:
    """Fills the chrome `{{key}}` placeholders and the `window.I18N` JSON (F33).

    Placeholders are literal `{{...}}` tokens, replaced via `str.replace` (not
    `.format`): the CSS/JS in the template is full of single `{`/`}`, which `.format`
    would interpret as substitution fields.
    """
    i18n_map = {key: _t(key, lang) for key in _UI_STRINGS}
    lang_options = "".join(
        f'<option value="{code}"{" selected" if code == lang else ""}>{name}</option>'
        for code, name in _LANG_SELF_NAMES.items()
    )
    html = _INDEX_HTML_TEMPLATE.replace("{{lang}}", lang)
    html = html.replace("{{lang_options}}", lang_options)
    html = html.replace("{{i18n_json}}", json.dumps(i18n_map, ensure_ascii=False))
    for key, value in i18n_map.items():
        html = html.replace("{{" + key + "}}", value)
    return html


def _make_handler(db_path: Path, cache: PlanCache, cfg: Config,
                  process_state: _ProcessState,
                  sort_state: _SortState,
                  busy_lock: threading.Lock,
                  config_path: str | Path | None = None) -> type[BaseHTTPRequestHandler]:
    default_lang = i18n.normalize_lang(cfg.raw.get("language"))
    _index_html_cache: dict[i18n.Lang, bytes] = {
        default_lang: _render_index_html(default_lang).encode("utf-8"),
    }

    def _resolve_query_lang(raw_values: list[str] | None) -> i18n.Lang:
        """`?lang=` from the query -> a valid code (ru/en/ja), otherwise `default_lang`
        (F39: an invalid/absent lang does not crash, just the config default)."""
        raw = (raw_values or [""])[0].strip().lower()
        if raw in _UI_LANGS:
            return raw  # type: ignore[return-value]
        return default_lang

    def _index_html_for(lang: i18n.Lang) -> bytes:
        html = _index_html_cache.get(lang)
        if html is None:
            html = _render_index_html(lang).encode("utf-8")
            _index_html_cache[lang] = html
        return html

    class Handler(BaseHTTPRequestHandler):
        server_version = "SortaUI/1"

        def log_message(self, fmt: str, *args: object) -> None:
            _log.debug("%s - %s", self.address_string(), fmt % args)

        def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler contract)
            parts = urlsplit(self.path)
            path = parts.path
            if path == "/":
                self._serve_index(parse_qs(parts.query))
            elif path == "/api/plan":
                self._serve_plan(parse_qs(parts.query))
            elif path == "/api/dupes":
                self._serve_dupes()
            elif path == "/api/moves":
                self._serve_moves(parse_qs(parts.query))
            elif path == "/api/clusters":
                self._serve_clusters()
            elif path == "/api/events":
                self._serve_events()
            elif path == "/api/process/status":
                self._serve_process_status()
            elif path == "/api/process/defaults":
                self._send_json(_process_defaults_payload(cfg))
            elif path == "/api/config":
                self._send_json({"language": i18n.normalize_lang(cfg.raw.get("language"))})
            elif path == "/api/env":
                self._send_json(_env_payload())
            elif path == "/api/sort/status":
                self._serve_sort_status()
            elif path == "/api/sort/suggest-dest":
                self._send_json({"dest": _suggested_sort_dest(cfg, db_path)})
            elif path == "/api/tabs/visibility":
                self._send_json(_tabs_visibility_payload(db_path))
            elif path.startswith("/thumb/"):
                self._serve_thumb(path[len("/thumb/"):])
            elif path.startswith("/preview/"):
                self._serve_preview(path[len("/preview/"):])
            elif path.startswith("/photo/"):
                self._serve_photo(path[len("/photo/"):])
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler contract)
            parts = urlsplit(self.path)
            path = parts.path
            if path == "/api/dupes/choice":
                self._handle_dupes_choice()
            elif path == "/api/dupes/choices":
                self._handle_dupes_choices()
            elif path == "/api/dupes/skip":
                self._handle_dupes_skip()
            elif path == "/api/dupes/trash":
                self._handle_dupes_trash()
            elif path == "/api/photo/trash":
                self._handle_photo_trash()
            elif path == "/api/photos/trash":
                self._handle_photos_trash()
            elif path == "/api/clusters/label":
                self._handle_cluster_label()
            elif path == "/api/clusters/merge":
                self._handle_cluster_merge()
            elif path == "/api/album":
                self._handle_album()
            elif path == "/api/process":
                self._handle_process_start()
            elif path == "/api/process/rerun-optional":
                self._handle_process_rerun_optional()
            elif path == "/api/process/cancel":
                self._handle_process_cancel()
            elif path == "/api/process/reset":
                self._handle_process_reset()
            elif path == "/api/config/language":
                self._handle_set_language()
            elif path == "/api/browse":
                self._handle_browse()
            elif path == "/api/sort":
                self._handle_sort_start()
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def _serve_index(self, query: dict[str, list[str]]) -> None:
            lang = _resolve_query_lang(query.get("lang"))
            self._send_bytes(_index_html_for(lang), "text/html; charset=utf-8")

        def _serve_plan(self, query: dict[str, list[str]]) -> None:
            mode = (query.get("mode") or [""])[0]
            items = cache.get(mode)
            if items is None:
                self._send_json({"error": f"unsupported mode: {mode!r}"},
                                status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json([_plan_item_to_json(it) for it in items])

        def _serve_dupes(self) -> None:
            self._send_json(_dupes_payload(db_path, cfg.index.phash_max_distance))

        def _serve_moves(self, query: dict[str, list[str]]) -> None:
            raw_batch = (query.get("batch") or [""])[0]
            batch_id = None
            if raw_batch:
                try:
                    batch_id = int(raw_batch)
                except ValueError:
                    batch_id = None
            self._send_json(_moves_payload(db_path, batch_id))

        def _serve_clusters(self) -> None:
            self._send_json(_clusters_payload(db_path))

        def _serve_events(self) -> None:
            self._send_json(_events_payload(db_path))

        def _read_json_body(self) -> object | None:
            try:
                length = int(self.headers.get("Content-Length", 0))
            except ValueError:
                return None
            if length <= 0:
                return None
            try:
                return json.loads(self.rfile.read(length))
            except json.JSONDecodeError:
                return None

        def _handle_dupes_choice(self) -> None:
            parsed = _validate_group_payload(self._read_json_body())
            if parsed is None:
                self._send_json({"error": "invalid body"}, status=HTTPStatus.BAD_REQUEST)
                return
            group, keep = parsed
            if keep is None or keep not in group:
                self._send_json({"error": "keep_file_id must be in group"},
                                status=HTTPStatus.BAD_REQUEST)
                return
            _apply_choice(db_path, group, keep)
            self._send_json({"ok": True})

        def _handle_dupes_choices(self) -> None:
            parsed = _validate_batch_choices_payload(self._read_json_body())
            if parsed is None:
                self._send_json({"error": "invalid body"}, status=HTTPStatus.BAD_REQUEST)
                return
            groups, skip = parsed
            saved = _apply_batch_choices(db_path, groups, skip)
            self._send_json({"saved": saved})

        def _handle_dupes_skip(self) -> None:
            parsed = _validate_group_payload(self._read_json_body())
            if parsed is None:
                self._send_json({"error": "invalid body"}, status=HTTPStatus.BAD_REQUEST)
                return
            group, _keep = parsed
            _skip_group(db_path, group)
            self._send_json({"ok": True})

        def _handle_dupes_trash(self) -> None:
            parsed = _validate_group_payload(self._read_json_body())
            if parsed is None:
                self._send_json({"error": "invalid body"}, status=HTTPStatus.BAD_REQUEST)
                return
            group, keep = parsed
            if keep is None or keep not in group:
                self._send_json({"error": "keep_file_id must be in group"},
                                status=HTTPStatus.BAD_REQUEST)
                return
            trashed = _trash_group(db_path, group, keep)
            self._send_json({"trashed": trashed})

        def _handle_photo_trash(self) -> None:
            file_id = _validate_file_id_payload(self._read_json_body())
            if file_id is None:
                self._send_json({"error": "invalid body"}, status=HTTPStatus.BAD_REQUEST)
                return
            trashed = _trash_files(db_path, [file_id])
            self._send_json({"trashed": trashed})

        def _handle_photos_trash(self) -> None:
            # bulk deletion of the selected (the shared _trash_files path, same as single)
            ids = _validate_file_ids_payload(self._read_json_body())
            if ids is None:
                self._send_json({"error": "invalid body"}, status=HTTPStatus.BAD_REQUEST)
                return
            trashed = _trash_files(db_path, ids)
            self._send_json({"trashed": trashed})

        def _handle_cluster_label(self) -> None:
            parsed = _validate_cluster_label_payload(self._read_json_body())
            if parsed is None:
                self._send_json({"error": "invalid body"}, status=HTTPStatus.BAD_REQUEST)
                return
            cluster_id, name = parsed
            name = name.strip()
            if not name:
                self._send_json({"error": "name must not be empty"},
                                status=HTTPStatus.BAD_REQUEST)
                return
            conn = _connect(db_path)
            try:
                root = faces.label_cluster(conn, cluster_id, name)
            except ValueError:
                self._send_json({"error": "cluster not found"}, status=HTTPStatus.NOT_FOUND)
                return
            finally:
                conn.close()
            self._send_json({"ok": True, "cluster_id": root, "label": name})

        def _handle_cluster_merge(self) -> None:
            parsed = _validate_cluster_merge_payload(self._read_json_body())
            if parsed is None:
                self._send_json({"error": "invalid body"}, status=HTTPStatus.BAD_REQUEST)
                return
            src, dst = parsed
            conn = _connect(db_path)
            try:
                root = faces.merge(conn, src, dst)
            except ValueError:
                self._send_json({"error": "cluster not found"}, status=HTTPStatus.NOT_FOUND)
                return
            finally:
                conn.close()
            self._send_json({"ok": True, "cluster_id": root})

        def _handle_album(self) -> None:
            parsed = _validate_album_payload(self._read_json_body())
            if parsed is None:
                self._send_json({"error": "invalid body"}, status=HTTPStatus.BAD_REQUEST)
                return
            kind, selector, mode, where, name, apply_, dest_str = parsed
            dest = Path(dest_str) if dest_str else _album_dest(cfg, db_path)
            conn = _connect(db_path)
            try:
                report = plan_album(cfg, conn, kind, selector, dest, mode=mode,
                                    where=where, apply=apply_, album_name=name)
            finally:
                conn.close()
            self._send_json(_album_report_to_json(report, apply_))

        def _serve_process_status(self) -> None:
            self._send_json(process_state.snapshot())

        def _handle_process_start(self) -> None:
            parsed = _validate_process_payload(self._read_json_body())
            if parsed is None:
                self._send_json({"error": "invalid body"}, status=HTTPStatus.BAD_REQUEST)
                return
            source_dir, deep, geo_online, faces, events = parsed
            if not Path(source_dir).is_dir():
                self._send_json({"error": "not a directory"}, status=HTTPStatus.BAD_REQUEST)
                return
            # F43/F45: sort (layout apply) and process — both heavy operations
            # write/move files; the shared busy_lock makes the "the other is not
            # running" check + its own try_start an atomic critical section (otherwise
            # a TOCTOU between two parallel POSTs).
            with busy_lock:
                if sort_state.snapshot()["running"]:
                    self._send_json({"error": "sort is running"}, status=HTTPStatus.CONFLICT)
                    return
                if not process_state.try_start(source_dir):
                    self._send_json({"error": "already running"}, status=HTTPStatus.CONFLICT)
                    return
            thread = threading.Thread(
                target=_run_pipeline,
                args=(db_path, cfg, source_dir, process_state, cache, deep, geo_online,
                      faces, events),
                daemon=True,
            )
            thread.start()
            self._send_json({"ok": True})

        def _handle_process_rerun_optional(self) -> None:
            # F62/F63: "Re-run selected" — the same _ProcessState/busy_lock as
            # /api/process; no source_dir from the client — indexing is not
            # overridden (_run_pipeline(source_dir=None) leaves cfg.sources).
            # deep -> junk with the VLM (naming.vlm_enabled=deep).
            parsed = _validate_rerun_optional_payload(self._read_json_body())
            if parsed is None:
                self._send_json({"error": "invalid body"}, status=HTTPStatus.BAD_REQUEST)
                return
            faces, events, deep = parsed
            with busy_lock:
                if sort_state.snapshot()["running"]:
                    self._send_json({"error": "sort is running"}, status=HTTPStatus.CONFLICT)
                    return
                if not process_state.try_start(""):
                    self._send_json({"error": "already running"}, status=HTTPStatus.CONFLICT)
                    return
            thread = threading.Thread(
                target=_run_pipeline,
                args=(db_path, cfg, None, process_state, cache),
                kwargs={"faces": faces, "events": events, "deep": deep,
                        "only_optional": True},
                daemon=True,
            )
            thread.start()
            self._send_json({"ok": True})

        def _handle_process_cancel(self) -> None:
            process_state.request_cancel()
            self._send_json({"ok": True})

        def _handle_process_reset(self) -> None:
            # F45: the reset also writes to the DB — hold busy_lock for the whole
            # reset, not just the check, otherwise sort/process could start in the
            # window between the check and db.reset_index itself.
            with busy_lock:
                if process_state.snapshot()["running"] or sort_state.snapshot()["running"]:
                    self._send_json({"error": "already running"}, status=HTTPStatus.CONFLICT)
                    return
                conn = _connect(db_path)
                try:
                    db.reset_index(conn)
                    cache.rebuild(cfg, conn)
                finally:
                    conn.close()
            self._send_json({"ok": True})

        def _handle_set_language(self) -> None:
            # F65: the "Folder language" selector — sets the OUTPUT language (folders/
            # names) for the plan preview and apply, separate from the interface `?lang`.
            # Persists into config.yaml (if known) so it survives restarts and CLI runs.
            lang = _validate_language_payload(self._read_json_body())
            if lang is None:
                self._send_json({"error": "invalid language"},
                                status=HTTPStatus.BAD_REQUEST)
                return
            # hold busy_lock: the rebuild must not race a running sort/process that
            # reads cfg (the same guard as /api/process/reset).
            with busy_lock:
                if process_state.snapshot()["running"] or sort_state.snapshot()["running"]:
                    self._send_json({"error": "already running"},
                                    status=HTTPStatus.CONFLICT)
                    return
                cfg.raw["language"] = lang
                cfg.language = lang
                if config_path is not None:
                    try:
                        save_language(config_path, lang)
                    except OSError as exc:
                        self._send_json({"error": f"could not save config: {exc}"},
                                        status=HTTPStatus.INTERNAL_SERVER_ERROR)
                        return
                conn = _connect(db_path)
                try:
                    cache.rebuild(cfg, conn)
                finally:
                    conn.close()
            self._send_json({"ok": True, "language": lang})

        def _handle_browse(self) -> None:
            self._send_json({"path": _browse_for_folder()})

        def _serve_sort_status(self) -> None:
            self._send_json(sort_state.snapshot())

        def _handle_sort_start(self) -> None:
            parsed = _validate_sort_payload(self._read_json_body())
            if parsed is None:
                self._send_json({"error": "invalid body"}, status=HTTPStatus.BAD_REQUEST)
                return
            dest, mode = parsed
            # F45: see the comment in _handle_process_start — the same shared
            # busy_lock, the same "other running -> own try_start" order.
            with busy_lock:
                if process_state.snapshot()["running"]:
                    self._send_json({"error": "process is running"}, status=HTTPStatus.CONFLICT)
                    return
                if not sort_state.try_start():
                    self._send_json({"error": "already running"}, status=HTTPStatus.CONFLICT)
                    return
            thread = threading.Thread(
                target=_run_sort, args=(db_path, cfg, dest, mode, sort_state, cache),
                daemon=True,
            )
            thread.start()
            self._send_json({"ok": True})

        def _serve_thumb(self, raw_id: str) -> None:
            file_id = _parse_file_id(raw_id)
            path = self._resolve(raw_id)
            if file_id is None or path is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            data = _thumb_bytes(file_id, path)
            if data is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._send_bytes(data, "image/jpeg")

        def _serve_photo(self, raw_id: str) -> None:
            path = self._resolve(raw_id)
            if path is None or not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self._send_bytes(path.read_bytes(), ctype)

        def _serve_preview(self, raw_id: str) -> None:
            # a large DECODED JPEG for the lightbox: HEIC/RAW, which the browser does
            # not render from the raw /photo, arrive here as JPEG (decode_rgb).
            file_id = _parse_file_id(raw_id)
            path = self._resolve(raw_id)
            if file_id is None or path is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            data = _preview_bytes(file_id, path)
            if data is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._send_bytes(data, "image/jpeg")

        def _resolve(self, raw_id: str) -> Path | None:
            """file_id (integer only) -> the path from files; otherwise None.

            A non-numeric/arbitrary segment (incl. with `../`) does not parse into an
            id and never reaches an FS read — the only path to a file is via
            SELECT path FROM files WHERE id = ?.
            """
            file_id = _parse_file_id(raw_id)
            if file_id is None:
                return None
            return _resolve_path(db_path, file_id)

        def _send_bytes(self, body: bytes, content_type: str) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, data: object, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(data).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def build_server(cfg: Config, conn: sqlite3.Connection, *,
                 port: int = DEFAULT_PORT,
                 config_path: str | Path | None = None) -> ThreadingHTTPServer:
    """Build (but do not start) the server bound to 127.0.0.1:port.

    port=0 asks the OS to pick a free port (used by tests and able to report the
    real port via server.server_port). `config_path` — the config.yaml to persist the
    folder-language choice into (POST /api/config/language); None disables the write
    (the running cfg is still updated in memory).
    """
    dest = Path(cfg.database).resolve().parent / "_sorta_ui_preview"
    cache = PlanCache(cfg, conn, dest)
    process_state = _ProcessState()
    sort_state = _SortState()
    busy_lock = threading.Lock()
    handler_cls = _make_handler(Path(cfg.database).resolve(), cache, cfg,
                                process_state, sort_state, busy_lock,
                                config_path=config_path)
    return ThreadingHTTPServer(("127.0.0.1", port), handler_cls)


def serve(cfg: Config, conn: sqlite3.Connection, *,
         port: int = DEFAULT_PORT, open_browser: bool = True,
         config_path: str | Path | None = None) -> None:
    """Start the local read-only plan server and block until Ctrl+C.

    127.0.0.1 only. A busy port -> RuntimeError with a clear message (the caller
    cli.py decides how to show it to the user). `config_path` is threaded to the
    server so the folder-language selector can persist into config.yaml.
    """
    try:
        httpd = build_server(cfg, conn, port=port, config_path=config_path)
    except OSError as exc:
        raise RuntimeError(f"sorta ui: порт {port} занят или недоступен: {exc}") from exc
    url = f"http://127.0.0.1:{httpd.server_port}/"
    print(f"sorta ui: {url} (Ctrl+C для остановки)")
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
