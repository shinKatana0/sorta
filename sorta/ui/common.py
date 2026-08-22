"""F182: what more than one tab of the web app needs.

The package is split by TAB, not by layer. Everything here is imported by two or more
tab modules; anything used by exactly one belongs with that tab instead. `sorta.ui`
re-exports all of it under the old names.
"""
from __future__ import annotations

import errno
import io
import logging
import os
import sqlite3
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Callable

from send2trash import send2trash as send_to_trash
from send2trash.exceptions import TrashPermissionError

from .. import imaging
from ..config import Config
from ..sorter import Destination, destinations


# Named, not `__name__`: the web app logs under one name however many modules F182 cut
# it into, and that name is the one every log line carried before the split.
_log = logging.getLogger("sorta.ui")

DEFAULT_PORT = 8756
_THUMB_MAX_EDGE = 200
_CLUSTER_SAMPLE_LIMIT = 6
_EVENT_SAMPLE_LIMIT = 8
_SUPPORTED_MODES = ("city", "person", "event")
_DEFAULT_ALBUM_DIRNAME = "_Альбомы"
# F70: bounded by a default and a hard maximum, so no query can ask the server for the
# whole mode — 26k items at once, before this.
_PLAN_PAGE_DEFAULT_LIMIT = 200
_PLAN_PAGE_MAX_LIMIT = 1000

# F39: the same three as i18n.Lang. Self-names are not translated — this is a
# language's name in that language.
_UI_LANGS: tuple[str, ...] = ("ru", "en", "ja")
_LANG_SELF_NAMES: dict[str, str] = {"ru": "Русский", "en": "English", "ja": "日本語"}

_ProgressCB = Callable[[int, "int | None"], None]  # (done, total|None) — compatible with progress.ProgressCB


def _parse_page_window(query: dict[str, list[str]],
                       default_limit: int = _PLAN_PAGE_DEFAULT_LIMIT
                       ) -> tuple[int, int] | None:
    """(offset, limit) for any paged route, or None -> 400.

    A bad parameter is rejected rather than coerced; a limit over the maximum is
    clamped instead, so an over-eager client gets less data, not an error.
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
    """The five keys every paged slice answers with (F173).

    `total` is the length of the LIST, never of this page; `has_more` is computed from
    the window the server actually served, so the button cannot disagree with the data.
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

    Opens its own connection: a ThreadingHTTPServer handler runs on its own thread and
    an sqlite3 connection may not cross threads (see PlanCache).
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


# F42: the People tab renders ~48 cluster cards with _CLUSTER_SAMPLE_LIMIT previews
# each -> ~288 concurrent GET /thumb/<id>, one thread per request. Without a cache each
# one re-runs decode_rgb + JPEG-encode and the parallel decodes saturate the CPU until
# the server stops responding. Hence two independent measures: the LRU below, and a
# semaphore that bounds how many decodes run AT ONCE while the cache is still cold.
_THUMB_CACHE_MAX_ITEMS = 512
_THUMB_DECODE_CONCURRENCY = max(2, min(8, os.cpu_count() or 4))
# The lightbox serves a large DECODED JPEG rather than the raw original (`/photo`):
# the browser cannot do HEIC/RAW, decode_rgb can. Frames are viewed one at a time, so
# fewer entries at a bigger edge than the thumbnails.
_PREVIEW_MAX_EDGE = 1600
_PREVIEW_CACHE_MAX_ITEMS = 64

# F80: the key carries the frame index — a clip has one tile but a whole filmstrip
# behind the lightbox, each frame a separate JPEG. Photos and tiles are always frame 0.
_ImgCacheKey = tuple[int, float, int]
_ThumbCacheKey = _ImgCacheKey  # name backward-compatibility
_thumb_cache: OrderedDict[_ImgCacheKey, bytes] = OrderedDict()
_thumb_cache_lock = threading.Lock()
_preview_cache: OrderedDict[_ImgCacheKey, bytes] = OrderedDict()
_preview_cache_lock = threading.Lock()
# Shared: the bound is on thumb and preview decodes TOGETHER.
_thumb_decode_semaphore = threading.Semaphore(_THUMB_DECODE_CONCURRENCY)


def _thumb_cache_clear() -> None:
    """Clear the in-process caches of decoded images (thumbnails + previews)."""
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

    Keyed by (file_id, mtime, frame), so a changed mtime invalidates the entry. The
    miss is rechecked AFTER the semaphore: another thread may have decoded the same key
    while this one queued, and a spike on one frame must not decode it twice.
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
        # F67: a full decode of the ORIGINAL per tile cost 180-470 ms; the preview
        # cache turns that into a few ms once any stage has touched the frame.
        # F80: video_frame with frame=0 IS decode_rgb_preview, photos included.
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
    entry per frame (at most SORTA_VIDEO_FRAMES of them).
    """
    return _encode_jpeg_cached(
        file_id, path, max_edge=_PREVIEW_MAX_EDGE, quality=88,
        cache=_preview_cache, cache_lock=_preview_cache_lock,
        cache_max=_PREVIEW_CACHE_MAX_ITEMS, frame=frame)


def _connect(db_path: Path) -> sqlite3.Connection:
    """A short-lived per-call connection (see _resolve_path for why per call)."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


# F241: a volume with nowhere to put the file (read-only mount, a share without a trash)
# makes `send2trash` raise. Deleting it anyway would turn the only promise this button
# makes — that it is reversible — into an `unlink` nobody asked for, so a volume that
# cannot take a file into the trash loses none.
TRASH_REFUSED_NO_BIN = "no_trash_on_volume"
TRASH_REFUSED_PERMISSION = "permission"
TRASH_REFUSED_IN_USE = "in_use"
TRASH_REFUSED_FAILED = "failed"

_TRASH_PROBE_PREFIX = ".sorta-trash-probe-"
# ERROR_SHARING_VIOLATION — a file another process holds open comes back as a plain
# PermissionError on Windows, and it is the one refusal that goes away by itself.
_WIN_SHARING_VIOLATION = 32


def _trash_volume_key(path: str) -> str:
    """The volume `path` lives on — the unit the preflight probe is cached by.

    Neither the platform nor the directory: `send2trash` decides by where the file is,
    so `C:\\` says nothing about `\\\\nas\\photos`, while two folders of one disk must
    not cost two probes.
    """
    full = os.path.abspath(path)
    drive = os.path.splitdrive(full)[0]
    if drive:
        return os.path.normcase(drive)
    directory = os.path.dirname(full)
    while directory and not os.path.ismount(directory):
        parent = os.path.dirname(directory)
        if parent == directory:
            break
        directory = parent
    return os.path.normcase(directory or os.sep)


def _volume_accepts_trash(directory: Path) -> bool | None:
    """Does the OS trash take files from `directory`? Asked by really trashing one.

    None means the probe could not be placed at all: the volume was never asked, so
    nothing about it may be cached and the real attempt has to answer instead.
    """
    probe = directory / f"{_TRASH_PROBE_PREFIX}{os.getpid()}-{threading.get_ident()}"
    try:
        probe.write_bytes(b"")
    except OSError as exc:
        _log.warning("ui: the trash probe could not be written to %s: %s", directory, exc)
        return None
    try:
        send_to_trash(str(probe))
        return True
    except OSError as exc:
        _log.warning("ui: %s does not accept files into the trash: %s", directory, exc)
        return False
    finally:
        # Both ways round: the probe outlives a refusal, and under a test double it
        # outlives the success too.
        try:
            probe.unlink()
        except OSError:
            pass


def _refusal_reason(exc: OSError) -> str:
    """The machine code for a failed `send_to_trash`, never the text of the exception.

    The text goes to the log; the screen speaks from the string catalog.
    """
    if isinstance(exc, TrashPermissionError):
        return TRASH_REFUSED_NO_BIN
    if (getattr(exc, "winerror", None) == _WIN_SHARING_VIOLATION
            or exc.errno in (errno.EBUSY, errno.ETXTBSY)):
        return TRASH_REFUSED_IN_USE
    if isinstance(exc, PermissionError):
        return TRASH_REFUSED_PERMISSION
    return TRASH_REFUSED_FAILED


def _trash_files(db_path: Path, ids: list[int]) -> tuple[list[dict], list[dict]]:
    """The single trash path: ids -> OS trash + DELETE of their files/dedup_choice rows.

    Returns (trashed, refused). A frame the trash would not take stays on disk AND in
    `files`, keeps its preview, and comes back in `refused` with a reason code — the
    DELETE runs over what actually left, never over what was asked for.

    An id outside the current files is silently skipped — idempotent on a repeat.
    F210: the frame's preview goes with it, keyed off the ROW (path, mtime, size),
    because after `send2trash` none of the three can be read off the disk any more.
    """
    if not ids:
        return [], []
    conn = _connect(db_path)
    try:
        placeholders = ",".join("?" * len(ids))
        rows = conn.execute(
            f"SELECT id, path, mtime, size FROM files WHERE id IN ({placeholders})", ids
        ).fetchall()
        trashed: list[dict] = []
        refused: list[dict] = []
        accepts: dict[str, bool] = {}
        for r in rows:
            path = Path(r["path"])
            entry = {"file_id": r["id"], "name": path.name}
            key = _trash_volume_key(r["path"])
            verdict = accepts.get(key)
            if verdict is None:
                verdict = _volume_accepts_trash(path.parent)
                if verdict is not None:
                    accepts[key] = verdict
            if verdict is False:
                _log.warning("ui: %s was not deleted (%s)", r["path"], TRASH_REFUSED_NO_BIN)
                refused.append({**entry, "reason": TRASH_REFUSED_NO_BIN})
                continue
            try:
                send_to_trash(r["path"])
            except OSError as exc:
                reason = _refusal_reason(exc)
                _log.warning("ui: %s was not deleted (%s): %s", r["path"], reason, exc)
                refused.append({**entry, "reason": reason})
                continue
            imaging.preview_delete(r["path"], r["mtime"], r["size"])
            trashed.append(entry)
        gone_ids = [t["file_id"] for t in trashed]
        if gone_ids:
            ph2 = ",".join("?" * len(gone_ids))
            with conn:
                conn.execute(f"DELETE FROM dedup_choice WHERE file_id IN ({ph2})", gone_ids)
                # F149: both directions. The derivation is a fact about a PAIR, so
                # either half going to the bin ends it — otherwise the button keeps
                # answering "you already have one" about a file that is gone.
                conn.execute(
                    f"DELETE FROM restored_files "
                    f"WHERE file_id IN ({ph2}) OR source_file_id IN ({ph2})",
                    gone_ids + gone_ids)
                conn.execute(f"DELETE FROM files WHERE id IN ({ph2})", gone_ids)
    finally:
        conn.close()
    return trashed, refused


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
    are collapsed, order is preserved.
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


# F174: `city` is the mode the web app applies (see `_run_sort`), so the caption is
# about the layout the button will actually produce. The folder name itself comes from
# `sorter.destinations` — never from a rule spelled a second time here.
_DEST_MODE = "city"

# The plan's reason codes, grouped into the few answers a BULK caption can state
# ("12 frames will return: 7 into cities, 5 into no_place"). A reason nobody grouped
# lands in `other` rather than being dropped: a group that silently loses frames would
# make the counts stop adding up to the selection.
_DEST_GROUPS: dict[str, str] = {
    "city": "city",
    "manual_reassign": "city",
    "country_only": "country",
    # F202: its own group — folded into `country` the caption would say "to the country
    # level" about frames that land a folder deeper, in a region the user named.
    "region_only": "region",
    "no_place": "no_place",
    "low_date": "undated",
    "downloaded": "undated",
}


def _destination_json(dest: Destination | None) -> dict:
    """The three fields a card needs to name its destination, or empty for an unknown id.

    All three are decided here: a client deriving `dest_group` from the folder name
    would be a second copy of the layout rules, in JS.
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

    Failing to compute it is not failing to show the page — the cards simply carry no
    `dest` field. Geo data may be missing, or the layout may raise on a config the
    slice has no say over.
    """
    if not rows:
        return {}
    try:
        return destinations(cfg, conn, _DEST_MODE, [int(r["id"]) for r in rows],
                            assume_action)
    except (ValueError, sqlite3.Error, OSError) as exc:
        _log.warning("ui: the destination of the frames could not be computed: %s", exc)
        return {}


def _parse_file_id_query(query: dict[str, list[str]]) -> int | None:
    """`?file_id=` as a positive int, or None -> 400 (the same rule as the POST body)."""
    raw = (query.get("file_id") or [""])[0].strip()
    if not raw.isdigit():
        return None
    return int(raw)


def _is_under(path: str, directory: str) -> bool:
    """Is `path` inside `directory`? A comparison of two strings, never of the disk.

    Both sides are normalized (case and separator) first: `files.path` carries the
    separators of the machine that indexed it. The boundary character is required —
    `/Photos/Greece2019` is not inside `/Photos/Greece`.
    """
    root = os.path.normcase(directory.rstrip("\\/"))
    target = os.path.normcase(path)
    if not root:
        return False
    return target.startswith(root + os.sep) or target.startswith(root + "/")


# The population every per-file number is counted over — exactly the files the sorter
# lays out, so a counter here matches what an apply will carry off.
_OVERVIEW_LIVE = "f.dup_of IS NULL AND f.error IS NULL"


# --- F227: what the launch is doing, while it is still doing it ----------------------
#
# The tray entry point binds the port FIRST and runs its diagnostics after, because none
# of them is needed to answer an HTTP request: `warn_if_gpu_mismatch` alone was 3.76 s of
# a 5.65 s start-up, all of it the torch import. The cost is a tab that opens onto a
# program still getting ready, so the launch says where it is: `sorta/tray.py` writes
# this object, `GET /api/startup` reads it. It lives in `common` because a state the
# route owned would have to be handed backwards to the code that runs before the route.
#
# Deliberately NO percentage: the steps differ in length by two orders of magnitude, so
# "step 5 of 7" would sit at 71% through the four-second torch import and then jump to
# done. Deliberately not the model download either (F222/F225) — that has its own line,
# its own megabytes and its own failure on the run screen.

STARTUP_CONFIG = "config"
STARTUP_PORT = "port"
STARTUP_DATABASE = "database"
STARTUP_SERVER = "server"
STARTUP_ENVIRONMENT = "environment"
STARTUP_GPU = "gpu"
STARTUP_GEO = "geo"

# In the order the launch walks them: the first four before the port answers, the last
# three the diagnostics that used to precede it. The page numbers the current step
# against this list, and the string catalog owes each name a caption.
STARTUP_STEPS: tuple[str, ...] = (
    STARTUP_CONFIG, STARTUP_PORT, STARTUP_DATABASE, STARTUP_SERVER,
    STARTUP_ENVIRONMENT, STARTUP_GPU, STARTUP_GEO,
)


class _StartupState:
    """Which step of the launch is running now, and what each finished one cost.

    The default is READY — `sorta ui` never declares a launch, so a server started any
    other way never shows a waiting screen it would have no way out of.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._steps: tuple[str, ...] = ()
        self._step: str | None = None
        self._done: list[tuple[str, float]] = []
        self._ready = True
        self._started: float | None = None
        self._total: float | None = None

    def expect(self, steps: tuple[str, ...] = STARTUP_STEPS) -> None:
        """This process is launching, and these are the steps it will take."""
        with self._lock:
            self._steps = tuple(steps)
            self._step = None
            self._done = []
            self._ready = False
            self._started = time.monotonic()
            self._total = None

    def enter(self, step: str) -> None:
        """The launch has started `step`.

        A step entered after `ready` does not take readiness back: the diagnostics behind
        the bind are still steps, and a page already showing the program must keep it.
        """
        with self._lock:
            self._step = step
            if self._started is not None and self._total is None:
                self._ready = False

    def leave(self, step: str, seconds: float) -> None:
        """`step` is over, and it took `seconds`."""
        with self._lock:
            self._done.append((step, float(seconds)))
            if self._step == step:
                self._step = None

    def ready(self) -> None:
        """Everything the launch had to do is done — the page may show the program."""
        with self._lock:
            self._step = None
            self._ready = True
            # Frozen here: "how long did the launch take" must stop being a clock that
            # keeps running while the program serves.
            self._total = self._seconds()

    def reset(self) -> None:
        """Back to "nothing is starting" — the state a fresh process opens with."""
        with self._lock:
            self._steps = ()
            self._step = None
            self._done = []
            self._ready = True
            self._started = None
            self._total = None

    def elapsed(self) -> float:
        """Seconds since the launch was declared — its total once it is over."""
        with self._lock:
            return self._seconds()

    def _seconds(self) -> float:
        """The clock of the launch. Call under the lock."""
        if self._total is not None:
            return self._total
        if self._started is None:
            return 0.0
        return max(0.0, time.monotonic() - self._started)

    def snapshot(self) -> dict:
        """What `GET /api/startup` answers: the state, in one consistent read."""
        with self._lock:
            return {
                "ready": self._ready,
                "step": self._step,
                "steps": list(self._steps),
                "done": [{"step": name, "seconds": round(seconds, 3)}
                         for name, seconds in self._done],
                "elapsed": round(self._seconds(), 3),
            }


# Reached through the accessor, never imported by value: a module holding its own
# reference would keep answering about a state the tests have since replaced.
_startup_state = _StartupState()


def startup_state() -> _StartupState:
    """The launch state of THIS process."""
    return _startup_state


def _startup_payload() -> dict:
    """The body of `GET /api/startup`."""
    return _startup_state.snapshot()
