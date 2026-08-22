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
from typing import Any, Callable, Sequence

from .. import imaging
from ..config import Config, skipped_stage_notes
from ..dedup import assign_duplicates, compute_phashes
from ..diagnostics import memory_health, nvidia_gpu_present
from ..events import build_events
from ..faces import detect_and_cluster
from ..faults import Fault, fault_code, fault_params
from ..geo import geo_cache_size, resolve_places
from ..i18n import Lang, normalize_lang
from ..indexer import excludes_path, index as run_index, load_excludes
from ..junk import classify as classify_junk
from ..junk import (
    CLASSIFY_PHASE_PETS_VLM,
    CLASSIFY_PHASE_RESCUE_VLM,
    CLASSIFY_PHASE_VLM,
    CLASSIFY_STAGE,
    VERDICTS_STAGE,
)
from ..landmarks import _SCAN_KEY as _LANDMARK_SCAN_KEY
from ..landmarks import Classifier, clip_classifier, detect_landmarks
from ..naming import name_events, naming_settings
from ..runlog import (
    Measurement, measurement_files, measurement_unit, read_measurements, stage_timer,
)
from ..offline import ENV_ALLOW_DOWNLOAD, offline_by_us
from ..tiers import (
    PartState, TierState, download_failure, megabytes, run_parts, stage_downloads,
    tier_states, watch_download, weights_size_mb,
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
#
# F222: `landmarks` joins them. It had no checkbox at all, which is what the whole
# feature is about — a stage nobody chose, downloading 1.6 GB and producing 0.55% of the
# places of the owner's collection. Same mechanism as the other two, deliberately: a
# second way to skip a stage is a second way to get it wrong.
_OPTIONAL_STAGES = ("landmarks", "faces", "events")

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


class _DownloadRefused(Fault, RuntimeError):
    """F222: the weights would not come down, said in words.

    A distinct class so that the message travelling to the browser is known to be the
    finished sentence and not something to be dressed up a second time. What a person got
    before it existed was `<urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] ...>` in the
    red box of the run screen — no stage, no model, no size, and nothing to do about it.
    The traceback still goes to the log.

    F245: the sentence it carries is now the ENGLISH one and the page builds its own from
    `params`. Written in the interface language here, it put a Russian paragraph into the
    traceback the log keeps of it.
    """

    codes = ("download_refused",)


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


def _run_language(cfg: Config) -> Lang:
    """The language a run writes its own sentences in — the configured one.

    The web app renders its chrome from `ui/strings.py`, but these two sentences (F222:
    the refusal of a download, and the note about a stage skipped while its settings sit
    in the file) are built on the server by the same functions the command line calls, so
    that the two entry points cannot word the same fact differently.
    """
    return normalize_lang(getattr(cfg, "language", None))


def _pipeline_steps(
        notify: Callable[..., None] | None = None
        ) -> list[tuple[str, _StageFn]]:
    """Processing steps in dependency order — the same as `cli._pipeline_steps`, plus
    `phash` last (canonically from cli _pipeline_steps).
    A fresh holder per call — a separate run does not share the CLIP classifier with
    the previous/next run.

    F135: a step returns `{"processed": n, "skipped": m}` where the stage's own stats
    can separate new work from what it recognised as already done — `index` (unchanged
    files) and `junk` (the F68 incremental skip). The rest return None: inventing a
    zero for a stage that does not count skips would claim something untrue.

    F222: `notify` is called with (stage, weights) when the CLIP model is about to be
    downloaded and with (None, ()) when the attempt is over. It hangs on the FACTORY and
    not on the stage: a run with no unknown places and no new frames never builds a
    classifier, and announcing a download there would be a sentence about nothing. The
    caller turns it into the line the run screen shows — 1.6 GB with no line at all is
    indistinguishable from a hang, which is exactly what the owner reported.

    F225: and while it runs it is called again, (stage, weights, bytes so far), at least
    every `tiers.PROGRESS_SECONDS`. Naming the model was not enough — the second report,
    on 2026-08-08, is of the same 1.6 GB arriving under a line that never changed, which
    reads as a hang exactly the way silence did. The bytes are measured on the disk by
    `tiers.watch_download`, the measurement the wizard's console prints from.
    """
    holder: dict[str, _LazyClassifierHolder] = {}
    # Which stage asked last: three of them share one classifier (F19) and whichever
    # arrives first pays for the download, so the sentence has to name that one.
    asked_by = {"stage": "landmarks"}

    def _build(cfg: Config) -> Classifier:
        stage = asked_by["stage"]
        pending = stage_downloads(stage)
        if not pending:
            # Already on disk: nothing to watch, nothing to announce, and a failure here
            # failed for some other reason and is said as-is.
            return clip_classifier(naming_settings(cfg))
        if notify is not None:
            notify(stage, pending)
        built: list[Classifier] = []
        try:
            # F225: the build runs on a thread of its own so that this one is free to
            # keep saying how much has arrived. The classifier lands in `built` — a
            # thread has no return value to hand back.
            failure = watch_download(
                lambda: built.append(clip_classifier(naming_settings(cfg))),
                lambda done: None if notify is None else notify(stage, pending, done))
            if failure is not None:
                _log.error("sorta ui: the weights for stage %r were not downloaded", stage,
                           exc_info=failure)
                raise _DownloadRefused(
                    download_failure(stage, pending, "en", failure), "download_refused",
                    stage=stage, weights=", ".join(pending) or "-",
                    size_mb=weights_size_mb(pending),
                    error=str(failure).strip() or failure.__class__.__name__,
                    offline_variable=ENV_ALLOW_DOWNLOAD if offline_by_us() else "",
                ) from failure
            return built[0]
        finally:
            if notify is not None:
                notify(None, ())

    def _clip(cfg: Config, stage: str) -> _LazyClassifierHolder:
        asked_by["stage"] = stage
        clf = holder.get("clip")
        if clf is None:
            clf = holder["clip"] = _LazyClassifierHolder(lambda: _build(cfg))
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
        detect_landmarks(cfg, conn, classifier=_clip(cfg, "landmarks"), progress=cb)
        return None

    def _faces(cfg: Config, conn: sqlite3.Connection, cb: _ProgressCB) -> _StageStats:
        detect_and_cluster(cfg, conn, progress=cb)
        return None

    def _events(cfg: Config, conn: sqlite3.Connection, cb: _ProgressCB) -> _StageStats:
        build_events(cfg, conn, progress=cb)
        name_events(cfg, conn)
        return None

    def _classify(cfg: Config, conn: sqlite3.Connection, cb: _ProgressCB) -> _StageStats:
        stats = classify_junk(cfg, conn, classifier=_clip(cfg, "classify"),
                              verdicts_only=True, progress=cb)
        return _stage_stats(stats, ("processed",), "skipped_incremental")

    def _junk(cfg: Config, conn: sqlite3.Connection, cb: _ProgressCB) -> _StageStats:
        stats = classify_junk(cfg, conn, classifier=_clip(cfg, "junk"), progress=cb)
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
        self.error_code: str | None = None
        self.error_params: dict[str, object] = {}
        self.finished = False
        self.source_dir: str | None = None
        self.phase: str | None = None
        self._phase_started = 0.0
        self._cancel_requested = False
        # F135: per-stage {"processed", "skipped"} of THIS run — see `_stage_stats`.
        self.stage_stats: dict[str, dict[str, int]] = {}
        # F217: was the deep tier asked for, and did it run. The fall back to the fast
        # tier is silent on purpose (`junk.py` catches everything around building the
        # classifier, and a missing package may not kill a four-hour run) — which leaves
        # a person who ticked "Deep analysis" with a finished run, an unchanged
        # collection and the reason in a log. `None` means the question was not asked or
        # could not be answered; see `_deep_tier_ran`.
        self.deep_requested = False
        self.deep_ran: bool | None = None
        # F222: the model this run is fetching right now, or None between downloads.
        # 1.6 GB with nothing on screen reads as a hang — the owner's report says
        # "it hung on landmarks", and nothing had hung. How much has arrived is known to
        # the download library and not to us; what is stated here is what we do know, and
        # a named model beats a silent hour.
        self.download_stage: str | None = None
        self.download_weights: tuple[str, ...] = ()
        # F225: ...and how much of it has arrived, measured on the disk. A model that is
        # named but never moves is read as a hang just as reliably as one that is not
        # named at all — which is what the second report of it said.
        self.download_done: int = 0
        # F222: stages this run skipped whose settings are still in config.yaml. The
        # sentence is built by `config.skipped_stage_notes`, the same one `sorta run`
        # prints, so a person cannot be told two different things about one file.
        self.skipped_notes: tuple[str, ...] = ()

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

    def set_deep_requested(self, requested: bool) -> None:
        """F217: this run was started with the deep tier ticked."""
        with self._lock:
            self.deep_requested = requested

    def set_deep_ran(self, ran: bool | None) -> None:
        """F217: whether the deep tier actually handled anything (None — unknown)."""
        with self._lock:
            self.deep_ran = ran

    def set_download(self, stage: str | None, weights: tuple[str, ...] = (),
                     done: int = 0) -> None:
        """F222: a model is being fetched for `stage` — or, with None, is not any more.

        F225: `done` is how many bytes have arrived since this download started. The
        line goes away when the download ends, so a finished one leaves no bar behind.
        """
        with self._lock:
            self.download_stage = stage if weights else None
            self.download_weights = weights
            self.download_done = done if weights else 0

    def _waiting_for_download_locked(self) -> bool:
        """F229: the current stage cannot count frames — its model is still arriving.

        No new state: both fields it reads are the ones F222/F225 already keep, and the
        question is asked here so that the run screen and this module cannot come to two
        different answers about one run. The stage line draws a FRAME counter, and a
        stage whose weights are not on disk has not processed one frame and cannot —
        "0 of 8" standing still for the whole download is what the owner read as a hang
        on a collection of eight photographs. The number was true; the unit was not.
        """
        return (self.running and bool(self.download_weights)
                and self.download_stage == self.stage)

    def set_skipped_notes(self, notes: Sequence[str]) -> None:
        """F222: what this run left out while the file still configures it."""
        with self._lock:
            self.skipped_notes = tuple(notes)

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

    def finish(self, error: str | None, error_stage: str | None = None,
               error_code: str | None = None,
               error_params: dict[str, object] | None = None) -> None:
        """End the run. `error_code`/`error_params` are the F245 personality of the
        failure, absent for anything that is not ours — the page then has only the
        English sentence, which is what it shows."""
        with self._lock:
            self.running = False
            self.finished = True
            self.error = error
            self.error_stage = error_stage
            self.error_code = error_code
            self.error_params = dict(error_params or {})
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
                # F245: which failure it was and the values it names, for the page to say
                # the same thing in its own language. Null for `sqlite3`, `OSError` and
                # everything else not ours — those are shown as they arrived.
                "error_code": self.error_code,
                "error_params": dict(self.error_params),
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
                # F217: the deep tier was asked for and the run went through on the fast
                # one — the state the log used to be the only witness of.
                "deep_requested": self.deep_requested,
                "deep_ran": self.deep_ran,
                # F222: what is coming down the wire, while it is coming. Null between
                # downloads and on a run that needs none — the screen draws nothing then,
                # rather than a bar about a finished download.
                # F225: `done_mb` is what has arrived, so the line can say "X of Y" and
                # keep changing while the gigabytes come down.
                "download": ({"stage": self.download_stage,
                              "weights": list(self.download_weights),
                              "mb": weights_size_mb(self.download_weights),
                              "done_mb": megabytes(self.download_done)}
                             if self.download_weights else None),
                # F229: ...and that THIS stage is waiting for it, which is what turns the
                # frame counter next to the line above off. False for a download that
                # belongs to another stage and for every run that needs none — then the
                # counter is the most useful thing on the screen and stays.
                "stage_waiting_download": self._waiting_for_download_locked(),
                # F222: the stages this run skipped whose settings the file still holds.
                # Almost always empty — a note on every run is noise, and noise is what
                # gets learned and then not read.
                "skipped_notes": list(self.skipped_notes),
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


def _browse_for_folder() -> tuple[str, str]:
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
        return "", BROWSE_CANCELLED
    try:
        return _run_browse_dialog()
    finally:
        _browse_lock.release()


# What the picker answered. "Cancelled" and "this machine has no picker" used to be one
# empty string, so on Ubuntu — where the system python has no tkinter until somebody
# installs python3-tk — the button did nothing at all and wrote nothing anywhere: the
# failing branch below did not even log. Met on 2026-08-09.
BROWSE_CANCELLED = ""
BROWSE_UNAVAILABLE = "unavailable"


def _run_browse_dialog() -> tuple[str, str]:
    """-> (path, problem). An empty problem means the answer is the path, cancel included."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", _BROWSE_DIALOG_SCRIPT],
            capture_output=True, text=True, timeout=_BROWSE_DIALOG_TIMEOUT_S,
            check=False,
        )
    except Exception:
        _log.exception("browse: could not start the folder dialog")
        return "", BROWSE_UNAVAILABLE
    if result.returncode != 0:
        _log.warning("browse: the folder dialog exited %s: %s",
                     result.returncode, (result.stderr or "").strip()[:400])
        return "", BROWSE_UNAVAILABLE
    return result.stdout.strip(), BROWSE_CANCELLED


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
    config.yaml).

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

    F204: and the last two questions the model is asked from anywhere in the pipeline —
    `features.junk_rescue` and `features.landmarks_verify`. They were on no screen at
    all: the file decided them, the run spent the hours, and the estimate below was
    already pricing one of them. An option nobody can see is not a default, it is a
    thing that happens.

    F217 took `vlm_available` out of here. It answered "is `transformers` importable",
    which is one half of one tier read a second way — and a second reading of the same
    question is what this feature exists to remove: `/api/env` now carries the state of
    every tier from the probe `sorta doctor` uses, and the note next to the deep checkbox
    is drawn from that.
    """
    return {
        "deep": bool(cfg.naming.vlm_enabled),
        "products": bool(cfg.vlm.products),
        "geo_online": cfg.geo.provider == "online",
        # F222: the landmark stage, which is off in a fresh config and stays exactly what
        # the file says in one that switched it on. The checkbox showing the saved value
        # is the whole point: a person who enabled it a month ago must not find it
        # silently cleared.
        "landmarks": bool(cfg.features.landmarks),
        "pets": bool(cfg.features.pets),
        "pets_verify": bool(cfg.features.pets_verify),
        "junk_rescue": bool(cfg.features.junk_rescue),
        "landmarks_verify": bool(cfg.features.landmarks_verify),
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
# index + geo + landmarks + phash, the four that always run. The shipped figure was
# ~5 minutes over the reference collection and it UNDERSTATED the stage by about a third:
# the full run of 2026-07-27 (offline geo, 26 135 canonical photographs) spent
#
#     index 5.3 min + landmarks 2.8 min + phash 0.7 min + geo 2.4 s  =  ~8.8 min
#
# and landmarks alone is more than half of the five minutes this used to claim for all
# four. Checked against the same run's other two defaults, which held up: faces was within
# 8% and the VLM's 0.78 s/frame reproduced to the second decimal a year and three models
# later. An estimate that runs long is a warning; one that runs short is a broken promise,
# so the number is the measured one.
#
# This is the DEFAULT — what the screen quotes before it has ever seen a run. A real
# measurement out of the run log replaces it, and the screen says which of the two it used.
#
# F222 took landmarks OUT of this line, because the stage now has a checkbox and a line
# of its own. Leaving it in would break the rule the whole block is written under: the
# price shown has to be the price of the run that will happen, and a run with the stage
# cleared does not spend those minutes. The split is the same measured run — index 5.3 +
# phash 0.7 + geo 2.4 s is the ~6.0 minutes below, landmarks the 2.8 that used to hide
# inside it. (The owner's run of 2026-08-07 spent 4.3 minutes there; both are real runs
# over different collections, and the machine's own measurement replaces either.)
_SEC_PER_BASE_FRAME = 6.0 * 60 / 26135
_SEC_PER_LANDMARKS_FRAME = 2.8 * 60 / 26135
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
# The model is asked in three passes and each is read from its OWN phase (F205). They
# used to share the name `junk_vlm`, and the price of a frame is not the same in all
# three: measured 2026-08-05, the deep tier runs pipelined at 1.4 frames/s while the
# animal check and the rescue ask one frame at a time at ~0.42 — so a line priced off
# another pass's phase is charged the rate of a different population, threefold wrong.
# The stage each phase belongs to is part of the unit key because that is the log's own
# spelling, not because it is what tells the three apart any more: the deep tier runs
# ahead of faces in `classify` (F165), the other two behind it in `junk`.
#
# A log written before this split holds only `junk_vlm`, so the two new units are simply
# absent there and their lines fall back to the shipped default — the same answer a
# machine that has never run gets, and the screen says so.
#
# F186 removed a fourth reader of that phase — the keeper question, which was priced from
# `estimate:` because the log could not tell its seconds from the per-frame ones. It is not
# asked any more, so nothing quotes a price for it.
_RATE_UNITS: dict[str, tuple[str, ...]] = {
    # F222: `landmarks` moved out of here into a line of its own — see
    # `_SEC_PER_LANDMARKS_FRAME`. A log written before the split holds all four units, so
    # nothing has to be re-measured for either line to read `measured`.
    "base": tuple(measurement_unit(stage) for stage in ("index", "geo", "phash")),
    "landmarks": (measurement_unit("landmarks"),),
    "faces": (measurement_unit("faces"),),
    "events": (measurement_unit("events"),),
    "vlm_verdict": (measurement_unit(VERDICTS_STAGE, CLASSIFY_PHASE_VLM),),
    "vlm_pets": (measurement_unit(CLASSIFY_STAGE, CLASSIFY_PHASE_PETS_VLM),),
    "vlm_rescue": (measurement_unit(CLASSIFY_STAGE, CLASSIFY_PHASE_RESCUE_VLM),),
}
_DEFAULT_RATES: dict[str, float] = {
    "base": _SEC_PER_BASE_FRAME,
    "landmarks": _SEC_PER_LANDMARKS_FRAME,
    "faces": _SEC_PER_FACES_FRAME,
    "events": _SEC_PER_EVENTS_FRAME,
    "vlm_verdict": _SEC_PER_VLM_FRAME,
    "vlm_pets": _SEC_PER_VLM_FRAME,
    "vlm_rescue": _SEC_PER_VLM_FRAME,
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
        #
        # F204: the band is counted whether or not the config asks for the rescue, because
        # it is now a line of the screen and a line has to state its price BEFORE it is
        # ticked. A collection nobody has scored yet says nothing either way — the score
        # is written by the rescue itself, so before the first run with it on this index
        # genuinely cannot tell, and a dash says exactly that.
        scored = int(conn.execute(
            "SELECT COUNT(*) FROM frame_quality"
            " WHERE junk_score IS NOT NULL").fetchone()[0])
        junk_rescue = None if not scored else int(conn.execute(
            "SELECT COUNT(*) FROM frame_quality WHERE junk_score >= ?",
            (float(cfg.features.junk_rescue_threshold),)).fetchone()[0])
        if cfg.features.junk_rescue:
            products = junk_rescue
        # F204: the landmark check, priced the way `products` is — off what it asked
        # about last time. `landmark_checks` holds one row per proposal shown to the
        # model, next to the stage's own scan rows (F136), which are keyed by a reserved
        # name and are not questions. Nothing asked yet -> a dash: the candidate band of
        # a run that has never widened its gate is not in this index to be counted.
        landmarks_verify = _positive_or_none(int(conn.execute(
            "SELECT COUNT(*) FROM landmark_checks WHERE landmark != ?",
            (_LANDMARK_SCAN_KEY,)).fetchone()[0]))
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
        # F222: the stage that used to hide inside `base`. Priced over the same frames —
        # what it walks is the place-less subset, but that subset is not knowable before
        # geo has run on the frames this run will add, and quoting the whole collection
        # is the direction that warns rather than the one that breaks a promise.
        "landmarks": _positive_or_none(photos),
        "faces": _positive_or_none(photos),
        "events": _positive_or_none(photos),
        "pets": _positive_or_none(photos),
        "pets_verify": pets_verify,
        # F161: the master switch is priced over the frames of the run it permits, and
        # the rate is a structural zero — permission costs nothing. The line that costs
        # what this one used to is `products`.
        "deep": _positive_or_none(photos),
        "products": products,
        # F204: the two questions that had no line. Both are one model call per frame of
        # a band this index can count, and both are a dash until it can — see above.
        "junk_rescue": junk_rescue,
        "landmarks_verify": landmarks_verify,
    }
    per_line: dict[str, _Rate] = {
        "base": rates["base"],
        "landmarks": rates["landmarks"],
        "faces": rates["faces"],
        "events": rates["events"],
        "pets": _Rate(0.0, _RATE_FIXED),
        "pets_verify": rates["vlm_pets"],
        # F161: the master switch itself. Zero and `fixed`, like the animal line and for
        # a kinder reason — that one rides on a pass that runs anyway, this one has no
        # pass at all.
        "deep": _Rate(0.0, _RATE_FIXED),
        # F165 moved the deep tier ahead of faces, into a stage of its own — so this is
        # the one model line whose rate comes from `classify` rather than from `junk`.
        "products": rates["vlm_verdict"],
        # F204: one question per candidate, the same shape as the animal check — the
        # rescue asks in the back half of the junk stage, the landmark check in
        # `landmarks`, and neither is a pass over the collection.
        # F205: the rescue is now priced off its own seconds rather than off whichever
        # pass happened to write the shared phase last. The landmark check has no phase of
        # its own to be priced from, so it keeps the nearest measurement there is — the
        # animal check, which is the same shape: one frame, one question, one at a time.
        "junk_rescue": rates["vlm_rescue"],
        "landmarks_verify": rates["vlm_pets"],
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


# --- F217: the install tiers, on the screen where the checkbox is -------------
#
# A person who installed Sorta with the installer and cleared the "set it up at the end"
# box lives in this web app and never opens a terminal — and until now the app never told
# them that a tier is missing. Worse, the refusal is silent BY DESIGN and rightly so:
# `junk.py` catches everything around building the deep classifier and falls back to the
# fast tier, so a run with "Deep analysis" ticked finishes, changes nothing, and says why
# only in a log nobody reads.
#
# The three states are the doctor's three lines and are named after them, because they
# are the same probe's answer (`sorta/tiers.py`) rather than a second reading of it. Two
# tiers of the catalog install no packages at all — their weights are downloaded by the
# stage on first use — so calling those "not installed" would send a person to the wizard
# for something that happens by itself. That is the distinction F216 built the middle
# state for, and it is the one this feature could most easily get wrong.
TIER_READY = "ready"
TIER_WEIGHTS = "weights"
TIER_ABSENT = "absent"


def _tier_state_name(state: TierState) -> str:
    """Which of the doctor's three sentences this tier gets — in its order of severity.

    Packages first: a tier missing both is missing the half a person has to act on, and
    the weights of a tier whose code is not installed will never be asked for.
    """
    if state.missing_packages:
        return TIER_ABSENT
    if state.missing_weights:
        return TIER_WEIGHTS
    return TIER_READY


def _tiers_payload(states: list[TierState] | None = None) -> dict[str, dict]:
    """Every tier of the catalog as the browser needs it: a state and what is missing.

    `missing` carries the model names for the middle state — the page renders the
    doctor's own sentence about them, and the size next to it comes from the catalog.
    The package names of an absent tier are not sent: the screen names the way out
    (`sorta-setup`), and a list of distributions is a repair a person does not perform
    from a browser.
    """
    return {
        state.key: {"state": _tier_state_name(state),
                    "missing": list(state.missing_weights)}
        for state in (tier_states() if states is None else states)
    }


# Whether this machine has an NVIDIA card is asked ONCE per process. Two reasons, and
# the second one is not cosmetic: a card does not arrive or leave while a server is up,
# and `nvidia-smi` on a half-installed driver may take the full 3 s its probe allows —
# which is longer than the tray gives this whole route when it asks "is the program on
# this port ours?" (`tray.PROBE_TIMEOUT`, 2 s). A second launch would then be told a
# stranger holds the port. The lock is held across the call so that two requests arriving
# together run one `nvidia-smi` and not two.
_gpu_present_cache: dict[str, bool] = {}
_gpu_present_lock = threading.Lock()


def _gpu_present_cache_clear() -> None:
    """Forget the hardware answer (test isolation)."""
    with _gpu_present_lock:
        _gpu_present_cache.clear()


def _gpu_present() -> bool:
    with _gpu_present_lock:
        if "answer" not in _gpu_present_cache:
            _gpu_present_cache["answer"] = nvidia_gpu_present()
        return _gpu_present_cache["answer"]


# --- F222: what THIS run will download, before the button is pressed ----------
#
# F217 put a note next to two checkboxes. It could not do more, because a note hangs on
# an OPTION and the stages that download the most had none: the classification pulls
# 1.6 GB of CLIP on a fresh machine and there is no tick in front of it, by decision —
# without the verdicts, screenshots, documents and product shots ride into the city
# folders among the photographs.
#
# So the screen states the sum before the run, over the lines that WILL run, the ones
# without a checkbox included. The numbers come from `tiers.run_parts`, i.e. from the one
# probe `sorta doctor` and the F217 notes read — a second copy would answer differently
# inside a release, which is the failure F211 and F217 both exist against.


def _parts_payload(parts: list[PartState] | None = None) -> dict[str, dict]:
    """Every line of the run: which tiers it needs, and what it would download.

    `missing` is per WEIGHT rather than per tier so the browser can add up a run without
    counting a shared model twice — landmarks, the animals and the classification all
    raise the same ViT-L-14, and three lines quoting 1.6 GB each would promise 4.8 GB of
    downloads for one file.
    """
    return {
        part.key: {
            "tiers": list(part.tiers),
            "weights": list(part.weights),
            "missing": list(part.missing),
            "mb": part.download_mb,
            "always": not part.optional,
            # F222 §6b: the tier is not installable from here and not installed. The
            # checkbox goes dead rather than lying, and the note beside it says why.
            "available": part.available,
        }
        for part in (run_parts() if parts is None else parts)
    }


def _weights_payload(parts: list[PartState] | None = None) -> dict[str, int]:
    """What each model that is still missing weighs — the megabytes of the summary."""
    sizes: dict[str, int] = {}
    for part in (run_parts() if parts is None else parts):
        for name in part.missing:
            sizes[name] = weights_size_mb((name,))
    return sizes


def _memory_payload() -> dict:
    """F237: whether this machine has the memory for a run, and the two numbers the line
    above the button states. Asked on every request rather than cached like the card
    (`_gpu_present`): free memory is the one thing on this route that changes while the
    page is open, which is what lets the line go away by itself when it does."""
    health = memory_health()
    return {"low": health.low, "free_mb": health.available_mb,
            "needed_mb": health.needed_mb}


def _env_payload() -> dict:
    """F64: the environment for the UI banner. `gpu_profile` — whether the GPU profile
    is installed (the nvidia-* packages exist only in the `gpu` extra; `find_spec`
    without importing torch). CPU profile -> False -> a reduced-speed banner on the
    "Process" tab. (Detects the chosen profile, not "whether CUDA works right now" —
    on a broken GPU profile the runtime fallback fires, which is a separate symptom.)

    F217: `gpu_present` — whether there is a card at all, and it is what decides whether
    that banner is shown. `gpu_profile` alone cannot: it answers "are the nvidia-*
    packages here", so a machine with no NVIDIA card was being advised to download a
    2.5 GB CUDA profile it has no use for. The probe is `nvidia-smi`, cheap and without
    importing torch, and everything that is not a successful listing is "no card"; it is
    asked once per process (see `_gpu_present`).

    `tiers` is the same answer `sorta doctor` gives, from the same probe, so the run
    screen can say next to a checkbox that the tier behind it is not installed.

    F222: `parts` and `weights` come out of that SAME reading — one `tier_states()` call
    feeds the notes, the per-option availability and the download summary, so the three
    cannot disagree about the same machine. The route stays a GET that reads and changes
    nothing; installing is still the wizard's job and nobody else's.
    """
    states = tier_states()
    parts = run_parts(states)
    return {
        "gpu_profile": importlib.util.find_spec("nvidia") is not None,
        "gpu_present": _gpu_present(),
        "tiers": _tiers_payload(states),
        "parts": _parts_payload(parts),
        "weights": _weights_payload(parts),
        "memory": _memory_payload(),
    }


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


# F217: "did the deep tier handle anything at all" — the Overview tab's own question
# (`vlm_ran`), asked on the run screen, where the person who ticked the box is standing.
# `media_class.tier` holds which tier produced a verdict, and `junk.classify` writes
# 'vlm' only when the model was actually raised (a fall back to CLIP writes 'clip'), so
# this is the signal the database already carries — nothing new is recorded for it.
_DEEP_TIER_SQL = "SELECT 1 FROM media_class WHERE tier = 'vlm' LIMIT 1"


def _deep_tier_ran(conn: sqlite3.Connection) -> bool | None:
    """Has any frame of this index been classified by the deep tier? None — cannot tell.

    A statement about the INDEX and not about this run alone, deliberately: an
    incremental run over a collection the deep tier has already been through processes
    nothing new, and calling that a fall back would be an alarm about a run that did
    exactly what it should. What it catches is the case the feature is for — the tier was
    asked for, it has never handled a single frame, and the answer was in a log.
    """
    try:
        return conn.execute(_DEEP_TIER_SQL).fetchone() is not None
    except sqlite3.Error:  # a schema older than the column — nothing is claimed
        return None


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

    F204 adds the last two of that shape — `features.junk_rescue` and
    `features.landmarks_verify`, both settings of a stage rather than stages. Same
    convention again, and for the rescue the "config decides" half matters as much as the
    override: a run started from outside the browser must keep the selection the file
    asks for, because that is what the estimate above has been pricing all along.
    """
    deep: bool = False
    products: bool | None = None
    geo_online: bool = False
    faces: bool = False
    events: bool = False
    # F222: an opt-in STAGE like the two above, and False by default for the same reason
    # — an unticked box has to force it off (the F57 rule). The config key behind it
    # (`features.landmarks`) is what the checkbox STARTS from, through
    # `/api/process/defaults`, exactly as `deep` starts from `naming.vlm_enabled`.
    landmarks: bool = False
    pets: bool = False
    pets_verify: bool | None = None
    junk_rescue: bool | None = None
    landmarks_verify: bool | None = None


def _validate_process_payload(payload: object) -> tuple[str, _RunOptions] | None:
    """Parse `{"source_dir": str, "deep": bool=False, "geo_online": bool=False,
    "faces": bool=False, "events": bool=False, "landmarks": bool=False,
    "pets": bool=False,
    "products": bool?, "pets_verify": bool?, "junk_rescue": bool?,
    "landmarks_verify": bool?}`
    (F50/#34: opt-in VLM tier / online geo for THIS run, without editing config.yaml;
    F53/#39: opt-in steps faces/events, the same principle — default False; F123:
    `pets` is an opt-in of the THIRD shape — neither a tier nor a step, but a config
    override on the junk stage, `features.pets`; F138: the same third shape for
    `features.pets_verify`. F186 retired the other three of that set — `vlm.quality`,
    the scope select and `dedup.keeper_vlm` — with the questions behind them. F204:
    `junk_rescue` and `landmarks_verify`, the same shape over `features`. F222:
    `landmarks` — a STAGE of the F53 kind, not a setting, and the first of those with a
    config key behind it.)
    None -> invalid: not dict / `source_dir` not a string or empty after strip / a flag
    given but not bool."""
    if not isinstance(payload, dict):
        return None
    source_dir = payload.get("source_dir")
    if not isinstance(source_dir, str) or not source_dir.strip():
        return None
    flags: dict[str, object] = {}
    for key in ("deep", "geo_online", "faces", "events", "landmarks", "pets"):
        value = payload.get(key, False)
        if not isinstance(value, bool):
            return None
        flags[key] = value
    for key in ("products", "pets_verify", "junk_rescue", "landmarks_verify"):
        value = payload.get(key)
        if value is not None and not isinstance(value, bool):
            return None
        flags[key] = value
    # `~` expanded here, at the boundary: it is a convention of the shell, and Python
    # leaves it alone. A person on Linux types `~/Downloads/photos` because that is what
    # every other program takes, and the run then refused a folder that exists (met
    # 2026-08-09). `indexer` expands too — this is the same answer one layer earlier, so
    # the path stored on the screen is the path the run will use.
    return (str(Path(source_dir.strip()).expanduser()),
            _RunOptions(**flags))  # type: ignore[arg-type]


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
    # F222: `landmarks` decides a STAGE and is applied here as well, so that anything
    # reading the config of this run (the stage's own settings, a log line) sees the run
    # that is actually happening rather than what the file asks for.
    features = dataclasses.replace(cfg.features, pets=opts.pets,
                                   landmarks=opts.landmarks)
    if opts.pets_verify is not None:
        features = dataclasses.replace(features, pets_verify=opts.pets_verify)
    # F204: the two that had no interface. Applied here and nowhere else — the thresholds
    # they select by stay in the file, where they were measured (F140/F131), because what
    # was missing was the choice to run them at all, not a way to retune them.
    if opts.junk_rescue is not None:
        features = dataclasses.replace(features, junk_rescue=opts.junk_rescue)
    if opts.landmarks_verify is not None:
        features = dataclasses.replace(features,
                                       landmarks_verify=opts.landmarks_verify)
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

    `faces`/`events`/`landmarks` (F53/#39, F222) — opt-in steps, default off: without the
    checkboxes the run builds only `index/geo/classify/junk/phash`, the heaviest steps are
    skipped. `stage_total`/the "stage i/N" numbering are computed from the actual filtered
    list. Skipping `landmarks` takes nothing away from a database that already has its
    `visual` places: the stage only ever WRITES places for rows geo left as 'unknown', so
    a run without it leaves those rows exactly as the last run left them.

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

    `junk_rescue`/`landmarks_verify` (F204) — two more of that third shape, on
    `features.junk_rescue` and `features.landmarks_verify`. Neither adds a stage: the
    rescue is a question the back half of `junk` asks about the frames its own score
    selected, the check is a question `landmarks` asks about a proposal CLIP made. Both
    are under `deep` like every other model question (F145): a subordinate flag raises
    no weights by itself, so ticking one with the master clear changes nothing.

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
    state.set_deep_requested(opts.deep)
    conn = _connect(db_path)
    error: str | None = None
    error_stage: str | None = None
    error_code: str | None = None
    error_params: dict[str, object] = {}
    try:
        run_cfg = _run_cfg(cfg, source_dir, opts)
        enabled_optional = {"faces": opts.faces, "events": opts.events,
                            "landmarks": opts.landmarks}
        # F222: the download line of the run screen, fed from the factory that does the
        # downloading — see `_pipeline_steps`.
        steps_of = _pipeline_steps(state.set_download)
        if only_optional:
            # F63: re-run the selected — faces/events by flags + junk with deep
            # (reclassification with the VLM). The order from _pipeline_steps is kept.
            # F123: pets asks for the same junk stage — a set, so two reasons to run it
            # still add up to one entry.
            rerun = {name for name in _OPTIONAL_STAGES if enabled_optional[name]}
            if opts.deep or opts.pets:
                rerun.add("junk")
            steps = [(name, fn) for name, fn in steps_of if name in rerun]
        else:
            steps = [(name, fn) for name, fn in steps_of
                     if name not in _OPTIONAL_STAGES or enabled_optional[name]]
            # F222: said about the config FILE and not about this run's overrides, so
            # `cfg` rather than `run_cfg` — the question is what a person wrote down and
            # is about to get nothing out of.
            state.set_skipped_notes(skipped_stage_notes(
                cfg, [name for name in _OPTIONAL_STAGES if not enabled_optional[name]],
                _run_language(cfg)))
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
                # F245: what the page says the same thing in its own language from.
                error_code = fault_code(exc)
                error_params = fault_params(exc)
                _log.exception("sorta ui: pipeline stage %r failed", name)
                completed = False
                break
        if completed and error is None:
            # F217: only for a run that finished. A cancelled or failed one did not get
            # to the classifier, and "the deep tier did not run" would describe the
            # cancel rather than the missing tier.
            if opts.deep:
                state.set_deep_ran(_deep_tier_ran(conn))
            try:
                cache.rebuild(cfg, conn)
            except Exception as exc:  # noqa: BLE001
                error = f"the plan was not rebuilt: {exc}"
                error_code = "plan_not_rebuilt"
                error_params = {"error": str(exc)}
    finally:
        conn.close()
        state.finish(error, error_stage, error_code, error_params)
