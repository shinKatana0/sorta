"""F182: what more than one tab of the web app needs.

The package is split by TAB, not by layer, because a feature lives in one tab and
would otherwise touch every layer. This module is the remainder of that cut: the
sqlite connection, the paging window, the thumbnail and preview caches, the
destination of a frame, the payload validators shared by several tabs. Everything
here is imported by two or more tab modules; anything used by exactly one belongs
with that tab instead.

`sorta.ui` re-exports all of it, so `ui._connect`, `ui._page_payload`,
`ui._thumb_cache_clear` and the rest keep the names they were imported by.
"""
from __future__ import annotations

import io
import logging
import os
import sqlite3
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Callable

from send2trash import send2trash as send_to_trash

from .. import imaging
from ..config import Config
from ..sorter import Destination, destinations


# Named, not `__name__`: the web app logs under one name however many modules it is cut
# into, and that name is the one every log line carried before F182 split the file.
_log = logging.getLogger("sorta.ui")

DEFAULT_PORT = 8756
_THUMB_MAX_EDGE = 200
_CLUSTER_SAMPLE_LIMIT = 6
_EVENT_SAMPLE_LIMIT = 8
_SUPPORTED_MODES = ("city", "person", "event")
_DEFAULT_ALBUM_DIRNAME = "_Альбомы"
# F70: `/api/plan` never serves a whole mode again — a category page is bounded by a
# default and a hard maximum, so no query can ask the server for 26k items at once.
_PLAN_PAGE_DEFAULT_LIMIT = 200
_PLAN_PAGE_MAX_LIMIT = 1000

# F39: UI switcher languages — the same three as i18n.Lang; self-names for the
# selector options (not translated — this is a language's name in that language).
_UI_LANGS: tuple[str, ...] = ("ru", "en", "ja")
_LANG_SELF_NAMES: dict[str, str] = {"ru": "Русский", "en": "English", "ja": "日本語"}

_ProgressCB = Callable[[int, "int | None"], None]  # (done, total|None) — compatible with progress.ProgressCB


def _parse_page_window(query: dict[str, list[str]],
                       default_limit: int = _PLAN_PAGE_DEFAULT_LIMIT
                       ) -> tuple[int, int] | None:
    """(offset, limit) for any paged route, or None -> 400.

    A missing parameter falls back to the default; a non-integer or negative one is
    rejected rather than coerced — the one outcome that must never happen is quietly
    serving the whole category. A limit above the maximum is clamped, not rejected:
    an over-eager client gets less data, not an error.

    F173: `default_limit` is an argument because one route's page size is a setting rather
    than a constant — search opens to `features.search_page`. Everything else about the
    window is the same rule for every list, which is the point: a slice added tomorrow
    gets a validated window by calling this, not by writing a fourth copy of it.
    """
    raw_offset = (query.get("offset") or ["0"])[0]
    raw_limit = (query.get("limit") or [str(default_limit)])[0].strip()
    try:
        offset, limit = int(raw_offset), int(raw_limit or default_limit)
    except ValueError:
        return None
    if offset < 0 or limit < 0:
        return None
    return offset, min(limit, _PLAN_PAGE_MAX_LIMIT)


def _page_payload(items: list[dict], *, total: int, offset: int, limit: int) -> dict:
    """The five keys every paged slice answers with — F173's shared half on the server.

    Two of them are the feature. `total` is the length of the LIST, never the length of
    this page: "showing 200" and "there are 200" read identically, and for a ranking the
    second is almost never true. `has_more` is computed here, from the window the server
    actually served, so the button on the screen cannot disagree with the data behind it —
    a client deciding for itself would have to keep a running count and would be wrong the
    first time a page came back short.

    A slice merges its own keys into the result (`animals`, `counts`, the state of the
    search index): what is shared is the paging, not the payload.
    """
    return {
        "items": items,
        "total": int(total),
        "offset": int(offset),
        "limit": int(limit),
        "has_more": int(offset) + len(items) < int(total),
    }


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

# F80: the key carries the frame index too — a clip has one tile but a whole
# filmstrip behind the lightbox, and every frame of it is a separate JPEG. Photos and
# tiles are simply always frame 0.
_ImgCacheKey = tuple[int, float, int]
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
    cache_max: int, frame: int = 0,
) -> bytes | None:
    """Ready JPEG bytes of a frame (decoded to max_edge), from cache or by decoding.

    The key (file_id, mtime, frame) — a change of mtime naturally invalidates the
    entry. A cache miss is rechecked AFTER acquiring the semaphore (another thread may
    have decoded and cached the same key while the current one waited in the queue) —
    avoids a needless re-decode under a request spike for one frame.
    """
    try:
        stat = path.stat()
    except OSError:
        return None
    mtime = stat.st_mtime
    key: _ImgCacheKey = (file_id, mtime, frame)
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
        # F67: a gallery of thousands of tiles used to pay a full decode of the
        # ORIGINAL per tile (180-470 ms) — the preview cache turns that into a few ms
        # once the frame has been touched by any stage.
        # F80: video_frame with frame=0 IS decode_rgb_preview (photos included), so
        # every tile and the whole photo path stay on exactly the previous code.
        img = imaging.video_frame(
            path, mtime, stat.st_size, frame, max_edge=max_edge)
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


def _preview_bytes(file_id: int, path: Path, frame: int = 0) -> bytes | None:
    """A large decoded JPEG for the lightbox (HEIC/RAW are rendered too).

    F80: `frame` > 0 asks for that frame of a clip's filmstrip — the same cache, one
    entry per frame (a strip is at most SORTA_VIDEO_FRAMES of them).
    """
    return _encode_jpeg_cached(
        file_id, path, max_edge=_PREVIEW_MAX_EDGE, quality=88,
        cache=_preview_cache, cache_lock=_preview_cache_lock,
        cache_max=_PREVIEW_CACHE_MAX_ITEMS, frame=frame)


def _connect(db_path: Path) -> sqlite3.Connection:
    """A short-lived per-call connection (see _resolve_path — the same reason:
    sqlite3 connections are not transferable between ThreadingHTTPServer threads)."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _trash_files(db_path: Path, ids: list[int]) -> list[dict]:
    """The single trash path: ids -> OS trash + DELETE of their files/dedup_choice rows.

    Reused by group deletion of duplicates (`_trash_group`, U3) and by deletion of a
    single frame (`/api/photo/trash`, U4). An id outside the current files (already
    deleted/unknown) is silently skipped — idempotent on a repeated call.

    F210: the frame's PREVIEW goes with it. The preview key is a hash of
    (path, mtime, size), and after `send2trash` not one of the three can be read off the
    disk any more — so the row, which still holds all three, is what the key is computed
    from, and the removal itself happens once the original is in the bin (nothing can
    regenerate a preview of a file that is no longer there). A preview that will not go —
    missing, locked, unwritable — is not an error: `imaging.preview_delete` never raises,
    because the tidying of a derivative may not stop the deletion of the original.
    """
    if not ids:
        return []
    conn = _connect(db_path)
    try:
        placeholders = ",".join("?" * len(ids))
        rows = conn.execute(
            f"SELECT id, path, mtime, size FROM files WHERE id IN ({placeholders})", ids
        ).fetchall()
        trashed = []
        for r in rows:
            send_to_trash(r["path"])
            imaging.preview_delete(r["path"], r["mtime"], r["size"])
            trashed.append({"file_id": r["id"], "name": Path(r["path"]).name})
        found_ids = [r["id"] for r in rows]
        if found_ids:
            ph2 = ",".join("?" * len(found_ids))
            with conn:
                conn.execute(f"DELETE FROM dedup_choice WHERE file_id IN ({ph2})", found_ids)
                # F149: both directions. Trashing a processed copy has to forget that it
                # existed (otherwise the button keeps answering "you already have one" for
                # a file that is gone), and trashing an ORIGINAL leaves its copy an
                # ordinary photograph — the derivation is a fact about a pair, and one half
                # of it is no longer there.
                conn.execute(
                    f"DELETE FROM restored_files "
                    f"WHERE file_id IN ({ph2}) OR source_file_id IN ({ph2})",
                    found_ids + found_ids)
                conn.execute(f"DELETE FROM files WHERE id IN ({ph2})", found_ids)
    finally:
        conn.close()
    return trashed


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


# --- F174: an action says WHERE the frame goes --------------------------------------
# Two of the marks the slices offer read as one movement to the person making it ("this
# frame does not belong in this slice"), and neither of them said where the frame ends
# up. Worse, they are not the same movement at all: taking an animal mark off changes
# a MEMBERSHIP and moves no file, while returning a product to the photos is a real
# transfer out of «_Товары» into a city on the next `sort --apply`. The fix is language,
# not storage — `manual_pet` and `manual_overrides` stay two tables.
#
# The folder name comes from `sorter.destinations`, i.e. from the code that builds the
# plan, never from a rule spelled a second time here. `city` is the mode because it is
# the mode the web app applies (see `_run_sort`), so the caption is about the layout the
# button will actually produce.
_DEST_MODE = "city"

# The plan's reason codes, grouped into the handful of answers a BULK caption can state:
# "12 frames will return: 7 into cities, 5 into no_place" is what the person needs before
# selecting dozens at once, and one folder name out of twelve would simply mislead them.
# A reason nobody grouped lands in `other` rather than being dropped — a group that
# silently loses frames would make the counts stop adding up to the selection.
_DEST_GROUPS: dict[str, str] = {
    "city": "city",
    "manual_reassign": "city",
    "country_only": "country",
    # F202: the third level of the place layout is a group of its own — folding it into
    # `country` would make the caption say "to the country level" about frames that land
    # a folder deeper, in a region the user named themselves.
    "region_only": "region",
    "no_place": "no_place",
    "low_date": "undated",
    "downloaded": "undated",
}


def _destination_json(dest: Destination | None) -> dict:
    """The three fields a card needs to name its destination, or empty for an unknown id.

    `folder` is what the caption prints, `reason` is what the explanation under it is
    looked up by (`dest_why_<reason>`, the `junk_bucket_<verdict>` pattern), and `group`
    is what the bulk breakdown counts. All three are decided HERE: a client that derived
    the group from the folder name would be a second copy of the layout rules, in JS.
    """
    if dest is None:
        return {}
    return {
        "dest": dest.folder,
        "dest_reason": dest.reason,
        "dest_group": _DEST_GROUPS.get(dest.reason, "other"),
    }


def _destinations_for(cfg: Config, conn: sqlite3.Connection, rows: list[sqlite3.Row],
                      assume_action: str | None = None) -> dict[int, Destination]:
    """`sorter.destinations` over the ids of one PAGE of cards, on the open connection.

    Bounded by the page the client asked for, so the cost does not grow with the archive.
    A failure to compute it is not a failure to show the page: geo data may be missing
    (`GeoResolver`) or the layout may raise on a config the slice has no say over, and a
    grid that 500s because a caption could not be phrased is worse than a grid without
    the caption. The cards then simply carry no `dest` field.
    """
    if not rows:
        return {}
    try:
        return destinations(cfg, conn, _DEST_MODE, [int(r["id"]) for r in rows],
                            assume_action)
    except (ValueError, sqlite3.Error, OSError) as exc:
        _log.warning("ui: не удалось вычислить назначение кадров: %s", exc)
        return {}


def _parse_file_id_query(query: dict[str, list[str]]) -> int | None:
    """`?file_id=` as a positive int, or None -> 400 (the same rule as the POST body)."""
    raw = (query.get("file_id") or [""])[0].strip()
    if not raw.isdigit():
        return None
    return int(raw)


def _is_under(path: str, directory: str) -> bool:
    """Is `path` inside `directory`? A comparison of two strings, never of the disk.

    `files.path` is written by the indexer with the separators of the machine that
    indexed it, and the folder arrives from the client's own tree, so both are
    normalized (case and separator) before the prefix test. The boundary character is
    required — `/Photos/Greece2019` must not count as being inside `/Photos/Greece`.
    """
    root = os.path.normcase(directory.rstrip("\\/"))
    target = os.path.normcase(path)
    if not root:
        return False
    return target.startswith(root + os.sep) or target.startswith(root + "/")


# The population every per-file number is counted over — exactly the files the sorter
# lays out (`plan_and_sort`), so a counter here matches what an apply will carry off.
_OVERVIEW_LIVE = "f.dup_of IS NULL AND f.error IS NULL"
