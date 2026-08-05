"""F182: the run screen — the pipeline, the source tree, and what a run costs.

Everything the "Process" screen owns: the stage list index -> geo -> landmarks ->
classify -> faces -> events -> phash and the background thread that walks it, the
folder picker and the source tree with its excludes, the estimate F138/F159 quote
before a run starts, and the caches a person may clear afterwards.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import os
import sqlite3
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .. import imaging
from ..config import Config
from ..dedup import assign_duplicates, compute_phashes
from ..events import build_events
from ..faces import detect_and_cluster
from ..geo import geo_cache_size, resolve_places
from ..indexer import excludes_path, index as run_index, load_excludes
from ..junk import classify as classify_junk
from ..junk import CLASSIFY_PHASE_VLM, CLASSIFY_STAGE, VERDICTS_STAGE
from ..landmarks import Classifier, clip_classifier, detect_landmarks
from ..naming import name_events, naming_settings
from ..runlog import (
    Measurement, measurement_files, measurement_unit, read_measurements, stage_timer,
)
from .common import _ProgressCB, _connect, _log
from .layout import PlanCache
from .review import _db_fingerprint


# --- F36: "Process" — the background pipeline index→geo→landmarks→classify→faces→
# events→junk→phash from the web (POST /api/process), pollable progress (GET
# /api/process/status), cancel (POST /api/process/cancel). NOT imported from cli.py
# (to avoid a cli<->ui cycle) — the same leaf functions as `cli._pipeline_steps` are
# called directly from indexer/geo/landmarks/faces/events/junk/dedup/naming, +
# compute_phashes (dedup) as the last step.

# F165: `classify` — the front half of the junk stage (the verdicts, `verdicts_only`),
# placed before `faces` so that the faces stage skips the frames the classifier has
# already called screenshots, documents, memes or products. The back half keeps its
# place: everything left in it reads what `faces` writes.
_PIPELINE_STAGE_NAMES = ("index", "geo", "landmarks", "classify", "faces", "events",
                         "junk", "phash")

# F53/#39: faces and events — the heaviest/longest steps, opt-in via the "Process"
# checkboxes, default off. `_pipeline_steps()` still builds the FULL list (see the
# assert above by _PIPELINE_STAGE_NAMES) — filtering is up to the caller
# (`_run_pipeline`), with the same name list as `cli._OPTIONAL_STAGES`.
_OPTIONAL_STAGES = ("faces", "events")

# F135: with one button the run always walks the whole pipeline, and a stage that
# skipped everything looks exactly like a stage that did nothing. A step may report
# `{"processed": n, "skipped": m}` — the same two numbers the CLI prints ("skipped as
# already processed") — and the status snapshot carries them to the client. `None`
# means the stage cannot tell the two apart, and then nothing is claimed about it.
_StageStats = dict[str, int] | None
_StageFn = Callable[[Config, sqlite3.Connection, "_ProgressCB"], _StageStats]


def _stage_stats(stats: object, processed: tuple[str, ...], skipped: str) -> _StageStats:
    """Sum the `processed` counters of a stage's stats object and read `skipped` off it.

    None when any of the names is missing or does not hold a number. Stages are
    replaceable (tests swap the whole leaf function, a future one may stop returning
    stats at all), and a caption at the bottom of the page is worth neither an
    exception in the pipeline thread nor a fabricated zero — "skipped: 0" would claim
    a stage skipped nothing where in truth it said nothing.
    """
    values: list[int] = []
    for name in (*processed, skipped):
        value = getattr(stats, name, None)
        if not isinstance(value, int):
            return None
        values.append(value)
    return {"processed": sum(values[:-1]), "skipped": values[-1]}


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

    def features(self, paths: list[str]) -> list[Any]:
        """The CLIP vectors of the paths already scored — the F128 half of the junk stage.

        F146: without this method the holder is not the classifier that stage expects.
        `junk.classify` decides whether it can fill `clip_embeddings` by looking for
        `features` on the object it was handed, so a wrapper forwarding `__call__` alone
        turned the whole half off — silently, and for every run started from the web app,
        which is where most runs are started.

        Laziness is untouched: a classifier that has not been built has scored nothing, so
        its cache holds nothing and every path is None — the same answer
        `landmarks.CachingFeatureClassifier` gives for a path nobody has scored, and no
        model is loaded to give it.
        """
        features_of = getattr(self._real, "features", None)
        if not callable(features_of):
            return [None] * len(paths)
        return list(features_of(paths))


def _pipeline_steps() -> list[tuple[str, _StageFn]]:
    """Processing steps in dependency order — the same as `cli._pipeline_steps`, plus
    `phash` last (canonically from cli _pipeline_steps).
    A fresh holder per call — a separate run does not share the CLIP classifier with
    the previous/next run.

    F135: a step returns `{"processed": n, "skipped": m}` where the stage's own stats
    can separate new work from what it recognised as already done — `index` (unchanged
    files) and `junk` (the F68 incremental skip). The rest return None: inventing a
    zero for a stage that does not count skips would claim something untrue.
    """
    holder: dict[str, _LazyClassifierHolder] = {}

    def _clip(cfg: Config) -> _LazyClassifierHolder:
        clf = holder.get("clip")
        if clf is None:
            clf = holder["clip"] = _LazyClassifierHolder(
                lambda: clip_classifier(naming_settings(cfg)))
        return clf

    def _index(cfg: Config, conn: sqlite3.Connection, cb: _ProgressCB) -> _StageStats:
        stats = run_index(cfg, conn, progress=lambda s: cb(s.scanned, None))
        assign_duplicates(conn, cfg.dedup.canonical_strategy)
        # `added + updated` is the work; `skipped` is what path+mtime+size recognised
        # as unchanged — the same split `cli._summarize_index` prints.
        return _stage_stats(stats, ("added", "updated"), "skipped")

    def _geo(cfg: Config, conn: sqlite3.Connection, cb: _ProgressCB) -> _StageStats:
        resolve_places(cfg, conn, progress=cb)
        return None

    def _landmarks(cfg: Config, conn: sqlite3.Connection, cb: _ProgressCB) -> _StageStats:
        detect_landmarks(cfg, conn, classifier=_clip(cfg), progress=cb)
        return None

    def _faces(cfg: Config, conn: sqlite3.Connection, cb: _ProgressCB) -> _StageStats:
        detect_and_cluster(cfg, conn, progress=cb)
        return None

    def _events(cfg: Config, conn: sqlite3.Connection, cb: _ProgressCB) -> _StageStats:
        build_events(cfg, conn, progress=cb)
        name_events(cfg, conn)
        return None

    def _classify(cfg: Config, conn: sqlite3.Connection, cb: _ProgressCB) -> _StageStats:
        stats = classify_junk(cfg, conn, classifier=_clip(cfg), verdicts_only=True,
                              progress=cb)
        return _stage_stats(stats, ("processed",), "skipped_incremental")

    def _junk(cfg: Config, conn: sqlite3.Connection, cb: _ProgressCB) -> _StageStats:
        stats = classify_junk(cfg, conn, classifier=_clip(cfg), progress=cb)
        return _stage_stats(stats, ("processed",), "skipped_incremental")

    def _phash(cfg: Config, conn: sqlite3.Connection, cb: _ProgressCB) -> _StageStats:
        compute_phashes(cfg, conn, progress=cb)
        return None

    steps: list[tuple[str, _StageFn]] = [
        ("index", _index), ("geo", _geo), ("landmarks", _landmarks),
        ("classify", _classify), ("faces", _faces), ("events", _events),
        ("junk", _junk), ("phash", _phash),
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
        # F191: WHICH stage the error belongs to. `stage` cannot answer that on its
        # own — a run that dies while rebuilding the plan cache is still standing on
        # the last stage that succeeded — and the collapsed stage row has to name the
        # failure without being opened.
        self.error_stage: str | None = None
        self.finished = False
        self.source_dir: str | None = None
        self.phase: str | None = None
        self._phase_started = 0.0
        self._cancel_requested = False
        # F135: per-stage {"processed", "skipped"} of THIS run — see `_stage_stats`.
        self.stage_stats: dict[str, dict[str, int]] = {}

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
            self.phase = None
            self._phase_started = 0.0

    def set_stage_stats(self, name: str, stats: dict[str, int]) -> None:
        """F135: what the finished stage `name` processed and what it skipped."""
        with self._lock:
            self.stage_stats[name] = dict(stats)

    def set_progress(self, done: int, total: int | None = None) -> None:
        """A signature superset of all stage ProgressCB variants (done, total|None).

        If cancellation is requested — raises _PipelineCancelled right from the
        callback: stages call progress often, so cancellation fires almost
        immediately (mid-stage), not only between stages.

        `total=None` zeroes the total instead of keeping the previous one (F84): a
        stage can go from a measurable phase to an unmeasurable one (faces: detection
        by frames -> HDBSCAN), and a total left over from the previous phase would
        keep drawing a filled bar with numbers that mean nothing.
        """
        with self._lock:
            cancel = self._cancel_requested
            if not cancel:
                self.done = done
                self.total = total if total is not None else 0
        if cancel:
            raise _PipelineCancelled()

    def set_phase(self, phase: str | None) -> None:
        """The named sub-phase of the current stage (F84), or None — no phase.

        The clock starts over on every change: on a phase without a percent the
        elapsed time is the only honest sign of life the bar can show.
        """
        with self._lock:
            self.phase = phase
            self._phase_started = time.monotonic() if phase else 0.0

    def request_cancel(self) -> None:
        with self._lock:
            if self.running:
                self._cancel_requested = True

    def cancel_requested(self) -> bool:
        with self._lock:
            return self._cancel_requested

    def finish(self, error: str | None, error_stage: str | None = None) -> None:
        with self._lock:
            self.running = False
            self.finished = True
            self.error = error
            self.error_stage = error_stage
            self.phase = None  # a finished run is not in any phase (F84)
            self._phase_started = 0.0

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
                # F191: the stage the error came from, so the collapsed stage row can
                # say which one fell over. None when the run failed outside a stage.
                "error_stage": self.error_stage,
                "finished": self.finished,
                "cancel_requested": self._cancel_requested,
                # F135: also what puts the source of the last run back into an empty
                # field — with one button the path has to come back by itself, and the
                # browser's own memory is not there in a fresh profile.
                "source_dir": self.source_dir,
                # F135: {stage: {"processed", "skipped"}} for the stages that can tell
                # new work from work recognised as already done.
                "stage_stats": {name: dict(values)
                                for name, values in self.stage_stats.items()},
                # F84: the sub-phase of the current stage and how long it has been
                # running. phase=None -> the stage reports no phases (every stage but
                # faces), and the client draws exactly what it drew before.
                "phase": self.phase,
                "phase_elapsed": (round(time.monotonic() - self._phase_started, 1)
                                  if self.phase else 0.0),
            }


class _StageProgress:
    """The callback a pipeline stage gets: `(done, total)` plus a `phase` channel (F84).

    Stages that know nothing about phases just call it, exactly as they called
    `state.set_progress` before. `faces` reports the phases of clustering through
    `.phase(name)` — the same duck-typed channel `progress.TaskProgress` gives the CLI.
    """

    def __init__(self, state: _ProcessState) -> None:
        self._state = state

    def __call__(self, done: int, total: int | None = None) -> None:
        self._state.set_progress(done, total)

    def phase(self, name: str) -> None:
        self._state.set_phase(name)


_BROWSE_DIALOG_TIMEOUT_S = 120
# Serialises the folder dialog — see _browse_for_folder.
_browse_lock = threading.Lock()

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
    works.

    Only one dialog at a time: the subprocess takes a second or two to show a window,
    and every request that arrives meanwhile used to spawn another Explorer. The
    client disables its button too, but that cannot cover a second browser tab or a
    click that races the disable — the guard belongs here as well. A refused call
    returns "" (same contract as cancel), so the already-open dialog stays the one
    the user is talking to.
    """
    if not _browse_lock.acquire(blocking=False):
        return ""
    try:
        return _run_browse_dialog()
    finally:
        _browse_lock.release()


def _run_browse_dialog() -> str:
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


# --- F81: the source folder tree ("do not scan") --------------------------------

# The response is bounded on purpose: on a pathological tree it must not blow up.
# Sizes are still summed over everything below, so the numbers stay truthful — only
# the node LIST is cut, and the answer says so instead of silently shortening.
_TREE_MAX_NODES = 2000
_TREE_MAX_DEPTH = 12


def _validate_tree_root(raw: object) -> Path | None:
    """The tree root arrives from the client, so it is checked before anything is read.

    The same rule the path behind the "Browse…" button meets in
    `_handle_process_start`: a non-empty ABSOLUTE path to an existing directory.
    Anything else (a relative path, a file, a directory that is not there) -> None ->
    400. The server never walks an arbitrary path just because it was asked to.
    """
    if not isinstance(raw, str) or not raw.strip():
        return None
    path = Path(raw.strip()).expanduser()
    if not path.is_absolute():
        return None
    try:
        if not path.is_dir():
            return None
        return path.resolve()
    except OSError:
        return None


def _sum_dir(directory: Path) -> tuple[int, int]:
    """(files, bytes) of a whole subtree — metadata only (`scandir`/`stat`)."""
    files = size = 0
    try:
        with os.scandir(directory) as it:
            entries = list(it)
    except OSError:
        return 0, 0
    for entry in entries:
        try:
            if entry.is_dir(follow_symlinks=False):
                sub_files, sub_size = _sum_dir(Path(entry.path))
                files += sub_files
                size += sub_size
                continue
            files += 1
            size += entry.stat(follow_symlinks=False).st_size
        except OSError:  # a vanished/unreadable entry is not worth failing the tree over
            continue
    return files, size


def _scan_dir(directory: Path, rel: str, name: str, depth: int,
              budget: list[int], max_depth: int) -> dict:
    node: dict = {"name": name, "rel": rel, "files": 0, "size": 0,
                  "children": [], "truncated": False}
    try:
        with os.scandir(directory) as it:
            entries = sorted(it, key=lambda e: e.name.lower())
    except OSError:
        return node
    for entry in entries:
        try:
            is_dir = entry.is_dir(follow_symlinks=False)
        except OSError:
            continue
        if not is_dir:
            node["files"] += 1
            try:
                node["size"] += entry.stat(follow_symlinks=False).st_size
            except OSError:
                pass
            continue
        child_rel = f"{rel}/{entry.name}" if rel else entry.name
        if depth < max_depth and budget[0] > 0:
            budget[0] -= 1
            child = _scan_dir(Path(entry.path), child_rel, entry.name, depth + 1,
                              budget, max_depth)
            node["children"].append(child)
            node["files"] += child["files"]
            node["size"] += child["size"]
        else:
            # over the limit: the folder is not sent, but its files still count
            sub_files, sub_size = _sum_dir(Path(entry.path))
            node["files"] += sub_files
            node["size"] += sub_size
            node["truncated"] = True
    return node


def _any_truncated(node: dict) -> bool:
    return bool(node["truncated"]) or any(_any_truncated(c) for c in node["children"])


def _source_tree_payload(root: Path, skip_scan: list[str], skip_layout: list[str],
                         max_nodes: int = _TREE_MAX_NODES,
                         max_depth: int = _TREE_MAX_DEPTH) -> dict:
    """§4: the directory structure under `root` — FOLDERS only, each with the number
    of files and the total size of its subtree. File contents are never read.

    Both exclusion lists ride along so the tree can be drawn with the state each folder
    already has (F82), in one request instead of two."""
    budget = [max_nodes]
    tree = _scan_dir(root, "", root.name or str(root), 0, budget, max_depth)
    return {
        "root": root.as_posix(),
        "tree": tree,
        "nodes": max_nodes - budget[0] + 1,
        "limit": max_nodes,
        "max_depth": max_depth,
        "truncated": _any_truncated(tree),
        "skip_scan": skip_scan,
        "skip_layout": skip_layout,
    }


def _excludes_payload(cfg: Config, root: Path) -> dict:
    """What is currently left out under `root`, in both meanings — the collapsed
    one-line summary of the source block shows the two numbers separately (§3).

    The size is measured for "do not scan" only: that is disk work the run will not do.
    A "do not lay out" folder is read and indexed exactly as before, so its size saves
    nothing and printing it would suggest otherwise.
    """
    excludes = load_excludes(excludes_path(cfg))
    scan = sorted(excludes.for_root(root))
    layout = sorted(excludes.layout_for_root(root))
    files = size = 0
    for rel in scan:
        sub_files, sub_size = _sum_dir(root.joinpath(*rel.split("/")))
        files += sub_files
        size += sub_size
    return {"root": root.as_posix(), "skip_scan": scan, "count": len(scan),
            "files": files, "size": size,
            "skip_layout": layout, "layout_count": len(layout)}


def _validate_excludes_payload(
        payload: object) -> tuple[str, list[object], list[object]] | None:
    """Parse `{"root": str, "skip_scan": [str, ...], "skip_layout": [str, ...]}`.
    None -> invalid body.

    The entries themselves are not judged here — `indexer.normalize_exclude` is the
    single place that decides whether a path may narrow the walk, and the handler
    reports back which ones it refused.
    """
    if not isinstance(payload, dict):
        return None
    root = payload.get("root")
    if not isinstance(root, str) or not root.strip():
        return None
    sections: list[list[object]] = []
    for key in ("skip_scan", "skip_layout"):
        values = payload.get(key, [])
        if not isinstance(values, list):
            return None
        sections.append(values)
    return root, sections[0], sections[1]


def _process_defaults_payload(cfg: Config) -> dict:
    """F57: defaults for the "Process" checkboxes — JS sets .checked by these values
    on page init (otherwise the checkboxes always start empty regardless of
    config.yaml). `vlm_available` — whether the `transformers` package is installed
    (`find_spec`, WITHOUT importing the module/loading the model).

    F123: `pets` rides here for the same reason and from the same place — the config
    (`features.pets`), which the settings column also edits. Two entry points, one
    source of truth, exactly as `deep` lives next to `naming.vlm_enabled`.

    F138: the knobs that moved onto this screen out of the settings column ride here too,
    from the same place — `features.pets_verify`. The column no longer offers them, so the
    file is now their ONLY home and this is what a run starts from. F186 retired three of
    that set (`vlm.quality`, `vlm.quality_scope`, `dedup.keeper_vlm`) with the two
    questions they switched on.

    F161: `products` joins them from `vlm.products`, and its default is the reason the
    key exists — a file that never heard of it answers True here, so the screen opens
    showing the run that file has always described.
    """
    return {
        "deep": bool(cfg.naming.vlm_enabled),
        "products": bool(cfg.vlm.products),
        "geo_online": cfg.geo.provider == "online",
        "pets": bool(cfg.features.pets),
        "pets_verify": bool(cfg.features.pets_verify),
        "vlm_available": importlib.util.find_spec("transformers") is not None,
    }


# --- F138: what this run costs, said before it starts -------------------------
#
# Moving four expensive knobs onto the run screen risks bringing back the console of
# switches F133 took away. What stops it is that the list means something: every line
# carries its price and the sum stands under them, so the screen is a budget a person
# assembles rather than a row of toggles.
#
# A price is only worth showing if it is COMPUTED. The same checkbox is four hours on
# one collection and four minutes on another, so nothing here is a constant in the
# markup: each number is a measured rate multiplied by a count taken out of THIS index.
# Where a count cannot be taken — a fresh collection, a stage that has never run — the
# answer is None and the screen draws a dash. A zero would read as "free", and the one
# thing an estimate may not do is promise twenty minutes with two hours coming.
#
# F159: the rates below are no longer THE price. They are the price until this machine
# has measured its own, which after F147 it does on every run — the run log holds
# `stage=<s>[ phase=<p>] elapsed=<sec> processed=<n>` for everything the pipeline does,
# and a second per frame taken from there beats a second per frame measured once on
# somebody else's collection and shipped in a wheel. The screen says which of the two it
# used, because a person deciding whether to wait four hours needs to tell "this is how
# it went for YOU last time" from "this is how it went for the developer".
#
# The defaults, each with the measurement it comes from:
_SEC_PER_VLM_FRAME = 0.78    # F113: one frame in one prompt
# The faces stage over the reference collection — the ~17 minutes the changelog and the
# F123 note both quote — spread over its 19 757 photographs.
_SEC_PER_FACES_FRAME = 17 * 60 / 19757
# index + geo + landmarks + phash, the four that always run: ~5 minutes over the same
# collection.
_SEC_PER_BASE_FRAME = 5 * 60 / 19757
# events: a grouping pass over rows the DB already holds — under a minute there, and it
# is scaled per frame for the same reason as the others rather than pinned at "fast".
_SEC_PER_EVENTS_FRAME = 15.0 / 19757

# Where a rate comes from, as it travels to the browser next to the seconds it produced.
# `fixed` is neither: the animal line costs 0 because the prompts ride inside a CLIP call
# that runs anyway, and a structural zero has no pedigree to state.
_RATE_MEASURED = "measured"
_RATE_DEFAULT = "default"
_RATE_FIXED = "fixed"

# Which units of the run log price which rate, and the default each falls back to. A rate
# counts as measured only when EVERY unit behind it is: `base` covers four stages, and
# three measured ones plus a guessed fourth is a guess wearing a measurement's clothes.
#
# The model questions are read from TWO units, because F165 split the stage that asks them
# in half: the deep tier decides what a frame IS and runs ahead of faces (`classify`),
# while the quality and animal questions read what faces wrote and stay behind it
# (`junk`). Both phases are called `junk_vlm`, so pricing the deep line off the wrong one
# would quietly charge it the rate of a different population.
#
# F186 removed a fourth reader of that phase — the keeper question, which was priced from
# `estimate:` because the log could not tell its seconds from the per-frame ones. It is not
# asked any more, so nothing quotes a price for it.
_RATE_UNITS: dict[str, tuple[str, ...]] = {
    "base": tuple(measurement_unit(stage)
                  for stage in ("index", "geo", "landmarks", "phash")),
    "faces": (measurement_unit("faces"),),
    "events": (measurement_unit("events"),),
    "vlm_verdict": (measurement_unit(VERDICTS_STAGE, CLASSIFY_PHASE_VLM),),
    "vlm_frame": (measurement_unit(CLASSIFY_STAGE, CLASSIFY_PHASE_VLM),),
}
_DEFAULT_RATES: dict[str, float] = {
    "base": _SEC_PER_BASE_FRAME,
    "faces": _SEC_PER_FACES_FRAME,
    "events": _SEC_PER_EVENTS_FRAME,
    "vlm_verdict": _SEC_PER_VLM_FRAME,
    "vlm_frame": _SEC_PER_VLM_FRAME,
}


@dataclasses.dataclass(frozen=True)
class _Rate:
    """Seconds per unit, and where that number came from (F159)."""

    seconds: float
    source: str
    at: datetime | None = None


def _resolve_rates(measurements: dict[str, Measurement]) -> dict[str, _Rate]:
    """The run log's rates where it has them, the shipped defaults where it does not."""
    rates: dict[str, _Rate] = {}
    for name, units in _RATE_UNITS.items():
        found = [measurements[unit] for unit in units if unit in measurements]
        if len(found) == len(units):
            rates[name] = _Rate(sum(m.seconds_per_unit for m in found),
                                _RATE_MEASURED, max(m.at for m in found))
        else:
            rates[name] = _Rate(_DEFAULT_RATES[name], _RATE_DEFAULT)
    return rates


# The photographs a run actually works on: `sorta` skips a duplicate and a file it could
# not read, so counting them in would price frames nobody looks at. Same predicate the
# faces measurement script samples by.
_LIVE_PHOTOS_SQL = ("SELECT COUNT(*) FROM files "
                    "WHERE dup_of IS NULL AND error IS NULL AND media_type = 'photo'")


def _positive_or_none(value: int) -> int | None:
    """A count of zero from a stage that has never run is "unknown", not "nothing"."""
    return value or None


# The estimate is asked for on every open of the first tab, and one of its counts is the
# near-duplicate grouping, which costs seconds over tens of thousands of pHashes (F66).
# Keyed like the Duplicates payload — any write to the index changes the fingerprint —
# plus the config values the arithmetic reads, so moving a threshold in the settings
# column re-prices immediately instead of serving the number the old one produced.
# F159 adds the run log to that list for the same reason: a run that has just written its
# own timings is exactly the moment the old prices stop being the right answer.
_ESTIMATE_CACHE_MAX_ITEMS = 2
_estimate_cache: OrderedDict[tuple, dict] = OrderedDict()
_estimate_cache_lock = threading.Lock()


def _estimate_cache_clear() -> None:
    """Drop the cached estimates (test isolation)."""
    with _estimate_cache_lock:
        _estimate_cache.clear()


def _run_log_fingerprint() -> tuple:
    """(mtime, size) of every file the measurements are read out of (F159)."""
    stats: list[tuple[str, int, int]] = []
    for path in measurement_files():
        try:
            st = path.stat()
        except OSError:
            continue
        stats.append((str(path), st.st_mtime_ns, st.st_size))
    return tuple(stats)


def _process_estimate_payload(cfg: Config, db_path: Path) -> dict:
    """`GET /api/process/estimate` — the seconds behind every line of the run budget.

    `counts` travels next to `seconds` on purpose: a number a person is asked to plan
    an evening around should be checkable against the collection it was derived from,
    not taken on faith. Both dicts use the same keys, and `None` in either means "this
    index does not know" — the screen draws a dash and the sum says so too.

    `pets` is 0.0 rather than None when there is anything to count: the animal prompts
    ride inside the CLIP call the junk stage makes anyway (F123), so the line genuinely
    adds nothing to the run — one of the two places a zero here is the truth. The other
    is `deep` since F161: a master switch that only grants permission does no work, and
    saying so with a number is the point of taking its old effect out into `products`.

    F159 adds `sources` and `measured_at`, on the same keys again. A rate is either
    `measured` — read out of this machine's own run log — or `default`, a number measured
    once elsewhere and shipped with the tool, and the difference is the whole point:
    somebody deciding whether to start a four-hour run is entitled to know whose four
    hours the estimate is describing. `fixed` is the third value and belongs to the one
    line that is structurally free.
    """
    key = (str(db_path), _db_fingerprint(db_path), cfg.index.phash_max_distance,
           float(cfg.features.pet_candidate_threshold),
           bool(cfg.features.junk_rescue), float(cfg.features.junk_rescue_threshold),
           float(cfg.estimate.measurement_max_age_days), _run_log_fingerprint())
    with _estimate_cache_lock:
        cached = _estimate_cache.get(key)
        if cached is not None:
            _estimate_cache.move_to_end(key)
            return cached
    conn = _connect(db_path)
    try:
        photos = int(conn.execute(_LIVE_PHOTOS_SQL).fetchone()[0])
        # The deep tier's gate picks its candidates from the CLIP probabilities of the
        # run in progress, so the only honest source for "how many frames it asks
        # about" is how many it answered on last time (`source='vlm'`).
        products = _positive_or_none(int(conn.execute(
            "SELECT COUNT(*) FROM media_class WHERE source = 'vlm'").fetchone()[0]))
        # F161: unless the F140 selection is on, and then the tier is shown the frames
        # that cleared `features.junk_rescue_threshold` instead of the whole candidate
        # list — 955 of the live collection's 24 196 against ~7 300, twelve minutes
        # against an hour and a half. The screen has to show the price of the run that
        # WILL happen, so the population follows the config rather than averaging the
        # two. A collection nobody has scored yet says nothing: no `junk_score` at all
        # is a dash, the same answer the pet check gives before its own pass has run.
        if cfg.features.junk_rescue:
            scored = int(conn.execute(
                "SELECT COUNT(*) FROM frame_quality"
                " WHERE junk_score IS NOT NULL").fetchone()[0])
            products = None if not scored else int(conn.execute(
                "SELECT COUNT(*) FROM frame_quality WHERE junk_score >= ?",
                (float(cfg.features.junk_rescue_threshold),)).fetchone()[0])
        # The pet check is shown the frames CLIP scored above the candidate threshold —
        # a number that exists only once the CLIP pet group has run at all.
        pet_scored = int(conn.execute(
            "SELECT COUNT(*) FROM frame_quality WHERE pet_score IS NOT NULL"
        ).fetchone()[0])
        pets_verify = None if not pet_scored else int(conn.execute(
            "SELECT COUNT(*) FROM frame_quality WHERE pet_score >= ?",
            (float(cfg.features.pet_candidate_threshold),)).fetchone()[0])
    finally:
        conn.close()
    rates = _resolve_rates(read_measurements(
        max_age_days=float(cfg.estimate.measurement_max_age_days)))
    counts: dict[str, int | None] = {
        "base": _positive_or_none(photos),
        "faces": _positive_or_none(photos),
        "events": _positive_or_none(photos),
        "pets": _positive_or_none(photos),
        "pets_verify": pets_verify,
        # F161: the master switch is priced over the frames of the run it permits, and
        # the rate is a structural zero — permission costs nothing. The line that costs
        # what this one used to is `products`.
        "deep": _positive_or_none(photos),
        "products": products,
    }
    per_line: dict[str, _Rate] = {
        "base": rates["base"],
        "faces": rates["faces"],
        "events": rates["events"],
        "pets": _Rate(0.0, _RATE_FIXED),
        "pets_verify": rates["vlm_frame"],
        # F161: the master switch itself. Zero and `fixed`, like the animal line and for
        # a kinder reason — that one rides on a pass that runs anyway, this one has no
        # pass at all.
        "deep": _Rate(0.0, _RATE_FIXED),
        # F165 moved the deep tier ahead of faces, into a stage of its own — so this is
        # the one model line whose rate comes from `classify` rather than from `junk`.
        "products": rates["vlm_verdict"],
    }
    seconds: dict[str, float | None] = {}
    for name, rate in per_line.items():
        count = counts[name]
        seconds[name] = None if count is None else round(count * rate.seconds, 1)
    measured = [rate.at for rate in per_line.values() if rate.at is not None]
    payload = {
        "seconds": seconds,
        "counts": counts,
        "sources": {name: rate.source for name, rate in per_line.items()},
        "measured_at": max(measured).date().isoformat() if measured else None,
    }
    with _estimate_cache_lock:
        _estimate_cache[key] = payload
        _estimate_cache.move_to_end(key)
        while len(_estimate_cache) > _ESTIMATE_CACHE_MAX_ITEMS:
            _estimate_cache.popitem(last=False)
    return payload


def _env_payload() -> dict:
    """F64: the environment for the UI banner. `gpu_profile` — whether the GPU profile
    is installed (the nvidia-* packages exist only in the `gpu` extra; `find_spec`
    without importing torch). CPU profile -> False -> a reduced-speed banner on the
    "Process" tab. (Detects the chosen profile, not "whether CUDA works right now" —
    on a broken GPU profile the runtime fallback fires, which is a separate symptom.)"""
    return {"gpu_profile": importlib.util.find_spec("nvidia") is not None}


# F94: the two caches the web app may look at and empty. The CLI (`sorta cache`) knows
# the same pair; the names are what travels in the body of `POST /api/cache/clear`.
_CACHE_TARGETS = ("preview", "geo")


def _cache_payload(db_path: Path) -> dict:
    """F94: what the preview and geo caches occupy — the numbers `sorta cache` prints.

    The preview side is a metadata-only walk (`_sum_dir`) of a directory that holds one
    JPEG per frame — tens of thousands of them on a real collection, which is exactly
    why this is its own route and not a field of the status snapshot. The geo side is a
    `COUNT(*)`, the unit `sorta cache` reports for it: rows, not bytes.

    A cache that was never written is not an error — a missing directory sums to
    (0, 0) and an empty table counts 0.
    """
    directory = imaging.preview_dir()
    files, size = _sum_dir(directory)
    conn = _connect(db_path)
    try:
        entries = geo_cache_size(conn)
    finally:
        conn.close()
    return {
        # F117: `max_gb` is 0 when no ceiling is set, and the front end renders that as
        # a state rather than as a limit of zero. The share is computed here so the two
        # entry points cannot disagree on the arithmetic.
        "preview": {"dir": str(directory), "files": files, "bytes": size,
                    "max_gb": imaging.preview_cache_max_gb()},
        "geo": {"entries": entries},
    }


def _validate_cache_clear_payload(payload: object) -> str | None:
    """Parse `{"target": "preview"|"geo"}` (F94). None -> invalid: not a dict, or a
    target outside the pair — deleting is not something to guess an object for."""
    if not isinstance(payload, dict):
        return None
    target = payload.get("target")
    if not isinstance(target, str) or target not in _CACHE_TARGETS:
        return None
    return target


@dataclasses.dataclass(frozen=True)
class _RunOptions:
    """The knobs of ONE run, exactly as the run screen sends them.

    Each is an override applied to a COPY of the config for this run and never written
    back to config.yaml: the screen starts from the file (`/api/process/defaults`) and
    what a person changes on it decides this run alone. That is F123's rule for `deep`
    and `pets`, and F138 extends it to the three knobs it took out of the settings
    column — a value with two homes acquires two truths and a question about which of
    them is the real one.

    F138 fields are `None` when the body did not carry them, meaning "the config
    decides" — the convention `cli._quality_overrides` already follows for
    `--quality/--no-quality`. The run screen always sends all four, so an unticked box
    there forces OFF (the F57 rule) rather than quietly falling back to config.yaml;
    `/api/process/rerun-optional`, which has no interface for them, leaves them alone.

    F161 adds `products` with the same convention, and the "config decides" half of it
    carries the compatibility promise: `/api/process/rerun-optional` sends `deep` and no
    `products`, so re-running the junk stage with the model does what it did before this
    key existed.
    """
    deep: bool = False
    products: bool | None = None
    geo_online: bool = False
    faces: bool = False
    events: bool = False
    pets: bool = False
    pets_verify: bool | None = None


def _validate_process_payload(payload: object) -> tuple[str, _RunOptions] | None:
    """Parse `{"source_dir": str, "deep": bool=False, "geo_online": bool=False,
    "faces": bool=False, "events": bool=False, "pets": bool=False,
    "products": bool?, "pets_verify": bool?}`
    (F50/#34: opt-in VLM tier / online geo for THIS run, without editing config.yaml;
    F53/#39: opt-in steps faces/events, the same principle — default False; F123:
    `pets` is an opt-in of the THIRD shape — neither a tier nor a step, but a config
    override on the junk stage, `features.pets`; F138: the same third shape for
    `features.pets_verify`. F186 retired the other three of that set — `vlm.quality`,
    the scope select and `dedup.keeper_vlm` — with the questions behind them.)
    None -> invalid: not dict / `source_dir` not a string or empty after strip / a flag
    given but not bool."""
    if not isinstance(payload, dict):
        return None
    source_dir = payload.get("source_dir")
    if not isinstance(source_dir, str) or not source_dir.strip():
        return None
    flags: dict[str, object] = {}
    for key in ("deep", "geo_online", "faces", "events", "pets"):
        value = payload.get(key, False)
        if not isinstance(value, bool):
            return None
        flags[key] = value
    for key in ("products", "pets_verify"):
        value = payload.get(key)
        if value is not None and not isinstance(value, bool):
            return None
        flags[key] = value
    return source_dir.strip(), _RunOptions(**flags)  # type: ignore[arg-type]


def _validate_rerun_optional_payload(
        payload: object) -> tuple[bool, bool, bool, bool] | None:
    """Parse `{"faces": bool=False, "events": bool=False, "deep": bool=False,
    "pets": bool=False}` for F62/F63 `POST /api/process/rerun-optional` (re-running the
    SELECTED on an already-built index: faces / events / junk-with-VLM when deep).
    F123: `pets` re-runs the junk stage too — the animals are counted inside it — so
    `deep` and `pets` together still mean ONE junk run, not two. None -> invalid: not
    dict / a field is given but not bool / all four False (nothing to re-run)."""
    if not isinstance(payload, dict):
        return None
    flags: list[bool] = []
    for key in ("faces", "events", "deep", "pets"):
        value = payload.get(key, False)
        if not isinstance(value, bool):
            return None
        flags.append(value)
    if not any(flags):
        return None
    faces, events, deep, pets = flags
    return faces, events, deep, pets


def _run_cfg(cfg: Config, source_dir: str | None, opts: _RunOptions) -> Config:
    """A COPY of the config with this run's overrides on it — the original, shared with
    the request handlers, is not mutated and config.yaml is not written (F138 §2).

    `deep`/`geo_online`/`pets` are full overrides (see `_run_pipeline`); the F138 knobs
    are applied only when the body carried them, so the one caller without an interface
    for them (`/api/process/rerun-optional`) keeps running by the file.
    """
    naming = dataclasses.replace(cfg.naming, vlm_enabled=opts.deep)
    geo = dataclasses.replace(cfg.geo,
                              provider="online" if opts.geo_online else "offline")
    features = dataclasses.replace(cfg.features, pets=opts.pets)
    if opts.pets_verify is not None:
        features = dataclasses.replace(features, pets_verify=opts.pets_verify)
    vlm_changed: dict[str, Any] = {}
    if opts.products is not None:
        vlm_changed["products"] = opts.products
    vlm = dataclasses.replace(cfg.vlm, **vlm_changed) if vlm_changed else cfg.vlm
    sources = [Path(source_dir).resolve()] if source_dir is not None else cfg.sources
    return dataclasses.replace(cfg, sources=sources, naming=naming, geo=geo,
                               features=features, vlm=vlm)


def _run_pipeline(db_path: Path, cfg: Config, source_dir: str | None,
                  state: _ProcessState, cache: PlanCache,
                  options: _RunOptions | None = None,
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

    `pets` (F123) — the same kind of override as `deep`, on `features.pets`, and NOT a
    stage: animals are three extra prompts inside the CLIP call the `junk` stage makes
    anyway, so the flag changes what that stage computes and leaves the list of stages
    exactly as it was. Making it an `_OPTIONAL_STAGES` entry would put a stage that does
    not exist into the run.

    `pets_verify`/`quality`/`quality_scope`/`keeper` (F138) — four more of that same
    third shape, all of them settings of the `junk` stage (`features.pets_verify`,
    `vlm.quality`, `vlm.quality_scope`, `dedup.keeper_vlm`), so the list of stages is
    again untouched and only what one of them computes changes. They are what the run
    screen prices: between a quarter of an hour and four hours each.

    `products` (F161) — the fifth of that shape, on `vlm.products`, and the one that took
    an effect away from `deep`: with it off the classify half runs its cheap tiers and
    asks the model nothing, whatever `deep` says. `deep` remains what decides whether a
    model may be raised at all.

    `only_optional` (F62/F63: "Re-run selected" — POST
    `/api/process/rerun-optional`) — steps are narrowed to the SELECTED stages over the
    already-built index: `faces` (with faces), `events` (with events), `junk` (with
    deep — reclassification with the VLM, `naming.vlm_enabled=deep` — or with `pets`,
    which recomputes the animal verdicts). `deep` and `pets` together are still ONE junk
    run: they are two settings of one stage. The other base ones
    (index/geo/landmarks/phash) are not run at all.

    F135: a step that returns `{"processed", "skipped"}` has it recorded into the
    state, so the finished run can say what it did and what it recognised as already
    done instead of showing the same "Done." for both.

    Cancellation is checked BETWEEN stages (not mid-stage — MVP). After a successful
    finish (without an error/cancel) the plan cache (the Cities tab) is recomputed
    with the same conn so the tabs show the new data right away; Duplicates/People/
    Events read the DB directly on each request and need no refresh.
    """
    opts = options or _RunOptions()
    conn = _connect(db_path)
    error: str | None = None
    error_stage: str | None = None
    try:
        run_cfg = _run_cfg(cfg, source_dir, opts)
        enabled_optional = {"faces": opts.faces, "events": opts.events}
        if only_optional:
            # F63: re-run the selected — faces/events by flags + junk with deep
            # (reclassification with the VLM). The order from _pipeline_steps is kept.
            # F123: pets asks for the same junk stage — a set, so two reasons to run it
            # still add up to one entry.
            rerun = {name for name in _OPTIONAL_STAGES if enabled_optional[name]}
            if opts.deep or opts.pets:
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
                # F69: the UI pipeline runs for hours in a background thread with
                # nobody watching the console — the per-stage timing has to reach the
                # run log, or "which stage ate the time" stays a guess.
                with stage_timer(name):
                    stats = fn(run_cfg, conn, _StageProgress(state))
                if stats is not None:
                    state.set_stage_stats(name, stats)
            except _PipelineCancelled:
                completed = False  # mid-stage cancellation via the progress callback
                break
            except Exception as exc:  # noqa: BLE001 — report via status, do not crash the thread
                error = str(exc)
                error_stage = name  # F191: named in the collapsed row, not behind a click
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
        state.finish(error, error_stage)
