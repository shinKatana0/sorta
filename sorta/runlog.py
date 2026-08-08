"""F69: a run log file — right now nothing of a run survives the console.

Logging is wired to the console only (`config.configure_logging`), and nobody reads
that console: `sorta ui` lives for hours in the background. On the 373 GB run of
2026-07-25 the VLM fell back to the CLIP tier and the geo database did not load —
both warnings went nowhere, so neither failure is reproducible after the fact, and
the per-stage timings never existed at all.

This module only ADDS a file sink: a rotating UTF-8 file on the ROOT logger, so the
`_log` of every module (`geo`, `junk`, `faces`, …) lands in it. The rich console
output is left exactly as it is — this is a second sink, not a replacement.

Note for wiring up entry points: the file handler sees a record only if the logger
that emitted it lets the record through first. `config.configure_logging` sets the
level of the `sorta` logger (default `WARNING`), so warnings/errors always reach the
file, while the INFO of `stage_timer`/`log_environment` needs `log_level: INFO`
(the `logging:` config section is the orchestrator's part of F69).

No DB, no schema, no config: the overrides live in the env vars `SORTA_LOG_FILE` /
`SORTA_LOG_LEVEL`. Never fatal — a log file that cannot be opened must not take the
tool down with it.

F166 adds the other half of the same idea: the file has to say what is happening NOW,
not only what happened. Every timed unit of a run — a stage and each phase inside it —
writes the same three kinds of line: `started`, a periodic `progress` (interval from
`logging.progress_interval_sec`), and a summary written the moment that unit is over.
The summary in particular is no longer held back until the stage ends, which is what
used to make a cut-short run lose the timings of phases that had long finished.

F159 closes the loop by READING those lines back (`read_measurements`). Once a machine
has run a stage, the file holds how fast that stage is HERE — and a rate read from there
beats any constant measured once on somebody else's collection and shipped in a wheel.
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import platform
import re
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Iterator

_LOG = logging.getLogger(__name__)

ENV_LOG_FILE = "SORTA_LOG_FILE"
ENV_LOG_LEVEL = "SORTA_LOG_LEVEL"

_MAX_BYTES = 5 * 1024 * 1024
_BACKUP_COUNT = 5
_DEFAULT_LEVEL = logging.INFO

# Time (ISO, local), level, logger, thread, message. The thread matters: in `ui.py`
# the pipeline runs in a background thread next to the HTTP handlers, and without the
# thread name the file is an unreadable interleaving of the two.
_FORMAT = "%(asctime)s.%(msecs)03d %(levelname)-8s %(name)s [%(threadName)s] %(message)s"
_DATEFMT = "%Y-%m-%dT%H:%M:%S"

# Marks our handlers on the root logger (the same trick as `_sorta_handler` in
# config.py) — lets tests and repeated calls tell them from foreign handlers.
_HANDLER_MARK = "_sorta_runlog_handler"

# Mirrors `geodata._DEFAULT_DATA_DIR` and its legacy fallback, in that order.
# Deliberately not imported: `geodata` pulls in numpy/scipy at import time, and the
# environment header must stay cheap and safe. The package directory comes FIRST — it
# is where F65 ships the data; probing only the repository-root path (which is what
# this did until 2026-07-26) made the header report a missing base on every correct
# install, which is worse than useless in the one line meant to catch a real one.
_GEO_DATA_DIR = Path(__file__).resolve().parent / "data" / "geo"
_LEGACY_GEO_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "geo"
_PLACES_FILE = "places.tsv"


def default_log_path() -> Path:
    """`%LOCALAPPDATA%\\sorta\\logs\\sorta.log` on Windows, `~/.cache/sorta/logs/...` elsewhere."""
    local_appdata = os.environ.get("LOCALAPPDATA") if os.name == "nt" else None
    base = Path(local_appdata) if local_appdata else Path.home() / ".cache"
    return base / "sorta" / "logs" / "sorta.log"


def _coerce_level(value: int | str | None) -> int | None:
    """A level name or a number -> int; None — nothing usable was given."""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    if text.lstrip("-").isdigit():
        return int(text)
    named = logging.getLevelName(text.upper())
    return named if isinstance(named, int) else None


def _resolve_level(level: int | str | None) -> int:
    """Argument -> env `SORTA_LOG_LEVEL` -> INFO. Garbage is skipped, not fatal."""
    for candidate in (level, os.environ.get(ENV_LOG_LEVEL)):
        resolved = _coerce_level(candidate)
        if resolved is not None:
            return resolved
    return _DEFAULT_LEVEL


def _resolve_path(path: str | Path | None) -> Path:
    """Argument -> env `SORTA_LOG_FILE` -> `default_log_path()`."""
    if path is not None:
        return Path(path)
    from_env = os.environ.get(ENV_LOG_FILE)
    if from_env:
        return Path(from_env)
    return default_log_path()


def _key(path: str | Path) -> str:
    """A comparable form of the path: like logging's own `os.path.abspath`, but
    case-insensitive on Windows (`C:\\Sorta` and `c:\\sorta` are one file)."""
    return os.path.normcase(os.path.abspath(str(path)))


def _existing_handler(root: logging.Logger, path: Path) -> logging.Handler | None:
    target = _key(path)
    for handler in root.handlers:
        base = getattr(handler, "baseFilename", None)
        if base is not None and _key(base) == target:
            return handler
    return None


def _lower_root_level(root: logging.Logger, level: int) -> None:
    """The root logger is WARNING by default — INFO records emitted on it would be
    dropped before any handler. Only ever lowers the threshold, never raises it."""
    if root.level == logging.NOTSET or root.level > level:
        root.setLevel(level)


def file_log_level(level: int | str | None = None) -> int:
    """Level the file sink will record at (argument, then SORTA_LOG_LEVEL, then INFO).

    Callers need this to decide how far to lower their own logger: a record is
    dropped by the level of the logger it was emitted on, long before any handler is
    consulted, so a WARNING console setting would silently swallow the INFO stage
    timings this module exists to write.
    """
    return _resolve_level(level)


def setup_file_logging(
    path: str | Path | None = None, level: int | str | None = None
) -> Path:
    """Attach a rotating UTF-8 file sink (5 MB x 5) to the root logger.

    Idempotent: a repeated call for the same path does not add a second handler
    (otherwise every line would be written twice). Returns the path actually used.

    Never raises: if the directory cannot be created / there are no rights / the disk
    is full, the problem is reported to the console and work continues without a file.
    """
    target = _resolve_path(path)
    resolved_level = _resolve_level(level)
    root = logging.getLogger()

    # Third-party `warnings.warn` (transformers, torch, easyocr) go through the
    # `py.warnings` logger and land in the file too.
    logging.captureWarnings(True)

    existing = _existing_handler(root, target)
    if existing is not None:
        existing.setLevel(resolved_level)
        _lower_root_level(root, resolved_level)
        return target

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        handler: logging.Handler = logging.handlers.RotatingFileHandler(
            target,
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUP_COUNT,
            # utf-8 is mandatory: the messages are full of Cyrillic, and the default
            # file encoding on a Windows machine (cp1251) breaks the write — the same
            # rake as the console crash behind `verbose=False` in junk.py.
            encoding="utf-8",
        )
    except Exception as exc:
        _LOG.warning(
            "runlog: не удалось открыть файл лога %s (%s) — работаем без файла", target, exc
        )
        return target

    handler.setLevel(resolved_level)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    setattr(handler, _HANDLER_MARK, True)
    root.addHandler(handler)
    _lower_root_level(root, resolved_level)
    return target


# --- F166: how often a running unit repeats its counters ----------------------------

# Seconds between the periodic `progress` lines. 60 is the default because a stage this
# tool runs is measured in tens of minutes: one line a minute is a heartbeat next to the
# `stage=junk elapsed=1521.005` it explains, and not a channel of its own. 0 switches the
# periodic line off entirely — `started` and the summaries are not affected by it, they
# are the record of the run rather than its heartbeat.
DEFAULT_PROGRESS_INTERVAL_SEC = 60.0

_progress_interval = DEFAULT_PROGRESS_INTERVAL_SEC


def set_progress_interval(seconds: object) -> None:
    """Set the interval between periodic `progress` lines (`logging:` in config.yaml).

    Pushed in from `config.load_config` rather than read from there: this module is a
    leaf that every other one imports, and importing the config back would be a cycle.
    Hence `object` and not a number — whatever the YAML happened to hold arrives here,
    and garbage is ignored rather than fatal, the same rule as the level and the path.
    A negative interval reads as "off" and not as "always".
    """
    global _progress_interval
    try:
        value = float(seconds)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return
    _progress_interval = max(0.0, value)


def progress_interval() -> float:
    """The interval currently in force, in seconds. 0 — no periodic lines."""
    return _progress_interval


class _Throttle:
    """One periodic line per interval, and no more (F166 requirement 7).

    The interval is read at every check instead of being captured at construction, so
    a config loaded after the first stage started still takes effect. The clock starts
    at construction: the first `progress` line comes an interval after the `started`
    line rather than right next to it.
    """

    def __init__(self) -> None:
        self._last = time.perf_counter()

    def reset(self) -> None:
        self._last = time.perf_counter()

    def due(self) -> bool:
        if _progress_interval <= 0:
            return False
        now = time.perf_counter()
        if now - self._last < _progress_interval:
            return False
        self._last = now
        return True


def _counters(processed: int | None, elapsed: float, total: int | None = None) -> str:
    """The ` processed=<n>[ total=<n>] rate=<n>/s` tail — empty if nothing was reported.

    `total` is only ever passed by the periodic line: the summary of a finished unit
    states what it did, and a denominator next to a final count would just repeat it.
    """
    if processed is None:
        return ""
    tail = f" processed={processed}"
    if total is not None:
        tail += f" total={total}"
    if elapsed > 0:
        tail += f" rate={processed / elapsed:.1f}/s"
    return tail


def _log_started(label: str, total: int | None) -> None:
    """`<label> started[ total=<n>]` — the line a unit opens with (F166)."""
    if total is None:
        _LOG.info("%s started", label)
    else:
        _LOG.info("%s started total=%d", label, total)


def _log_progress(label: str, elapsed: float, done: int | None,
                  total: int | None) -> None:
    """`<label> progress elapsed=<sec> processed=<n>[ total=<n>] rate=<n>/s` (F166).

    One form for a stage and for a phase — `label` is `stage=<s>` or
    `stage=<s> phase=<p>` — because the two are read together and a reader should not
    have to learn a second shape to follow a run down into its phases.
    """
    _LOG.info("%s progress elapsed=%.3f%s", label, elapsed,
              _counters(done, elapsed, total))


def log_phase(stage: str, phase: str, elapsed: float,
              processed: int | None = None) -> None:
    """Write the timing of ONE phase inside a stage (F147).

    `stage=<name> phase=<name> elapsed=<sec> processed=<n> rate=<n>/s` — the same
    key=value shape, the same logger and the same INFO level as the stage summary
    above, so one grep collects both and both reach the file at the settings a long
    production run is actually started with. A breakdown that only existed under DEBUG
    would be a breakdown nobody has when they need it: the junk stage of 2026-08-02
    took 2 070 seconds and the log held one line about it.

    The unit count is not optional decoration — eighteen minutes over 1 362 model calls
    and eighteen minutes over 22 096 frames are different news, and without the counter
    they read the same.

    A phase that did not run is simply never passed here. `elapsed=0` would read as
    "it happened instantly"; absence reads as "it did not happen".

    F166: written by `StagePhases` the moment the phase is over, not collected into a
    batch at the end of the stage. The shape of the line is deliberately untouched —
    every grep and every estimate built on F147 keeps working.
    """
    _LOG.info("stage=%s phase=%s elapsed=%.3f%s", stage, phase, elapsed,
              _counters(processed, elapsed))


@dataclass
class _PhaseTiming:
    """F147: how long one phase of a stage ran, and over how many units.

    Both halves are needed to price a phase. Seconds alone cannot tell an expensive
    question asked rarely from a cheap one asked of every frame — eighteen minutes over
    1 362 model calls and eighteen minutes over 22 096 frames look identical until the
    denominator is written down next to them.
    """

    seconds: float = 0.0
    processed: int = 0


class StagePhases:
    """The phases of ONE stage: their clocks, their counters and their lines (F166).

    F147 put this stopwatch inside the junk stage and read it once, at the end. That
    made it an instrument that answers "where did the time go" exactly when the time
    has already gone — and, worse, one that answers nothing at all if the run is cut
    short: the orchestrator interrupted `junk` on 2026-08-03 and lost the numbers of
    three phases that had finished long before. Here the same measurement writes as it
    goes: a phase announces itself, repeats its counters once an interval, and is
    written out the moment it is over.

    Two ways in, because a stage moves between its phases in two different ways:

    `enter` relabels INSIDE the current pass. The fast tier of `junk` interleaves
    CLIP -> OCR -> write per chunk (F73) over one shared counter of frames, so those
    three accumulate seconds in parallel buckets and none of them is finished until the
    pass is; re-entering a phase is the same phase, and its clock keeps adding up.

    `start` opens a NEW pass over a population of its own. That is the boundary at
    which the phases of the previous pass really are over, so it is where their
    summaries are written. Consecutive passes under the same name (the deep tier asks
    the VLM over three different candidate lists) keep sharing one bucket, exactly as
    F147 decided — the name is what the reader prices, not the call site.

    The stage owns the object and drives it; `stage_timer` only needs to be able to
    close it, which is what `track_phases` registers it for.
    """

    def __init__(self, stage: str) -> None:
        self._stage = stage
        # Insertion-ordered: the lines come out in the order the stage first entered
        # each phase, which is the order somebody reading them expects.
        self._timings: dict[str, _PhaseTiming] = {}
        self._open: str | None = None
        self._since = 0.0
        # The denominator and the position of the CURRENT pass — shared by the phases
        # that interleave inside it, which is why they live here and not in a bucket.
        # These are the very numbers `/api/process/status` reports as done/total.
        self._total: int | None = None
        self._done: int | None = None
        self._throttle = _Throttle()

    @property
    def open(self) -> str | None:
        """The phase the clock is on, or None — the stage is between passes."""
        return self._open

    def _label(self, phase: str) -> str:
        return f"stage={self._stage} phase={phase}"

    def _switch(self, phase: str) -> None:
        """Stop the clock of the phase being timed and start `phase`'s (F147)."""
        if phase == self._open:
            return
        now = time.perf_counter()
        if self._open is not None:
            self._timings[self._open].seconds += now - self._since
        self._timings.setdefault(phase, _PhaseTiming())
        self._open, self._since = phase, now

    def _elapsed(self, phase: str) -> float:
        """Seconds `phase` has cost so far, including the stretch still running."""
        seconds = self._timings[phase].seconds
        if phase == self._open:
            seconds += time.perf_counter() - self._since
        return seconds

    def enter(self, phase: str) -> None:
        """Relabel to `phase` inside the current pass; every other bucket is kept."""
        new = phase not in self._timings
        self._switch(phase)
        if new:
            # No total: a phase reached mid-pass knows the size of the pass, not of its
            # own share of it, and a denominator that does not belong to the numerator
            # is worse than none.
            _log_started(self._label(phase), None)

    def start(self, phase: str, total: int | None = None) -> None:
        """Open a new pass over `total` items of its own — the previous one is over."""
        self._retire(keep=phase)
        self._switch(phase)
        self._total, self._done = total, 0
        self._throttle.reset()
        _log_started(self._label(phase), total)

    def count(self, phase: str, units: int) -> None:
        """Add `units` to what `phase` has processed (F147).

        Separate from `enter` because the number is rarely known when the phase opens:
        the CLIP phase begins at the top of a chunk and only decides a few lines later
        how many of its frames actually need encoding.
        """
        self._timings.setdefault(phase, _PhaseTiming()).processed += units

    def step(self, done: int, total: int | None = None) -> None:
        """Where the current pass is NOW — throttled into one line per interval.

        `done`/`total` are the pass counters and not the bucket's own `processed`: they
        are what the stage is asked about while it runs, and they are the same pair
        `/api/process/status` serves, which is the point of doing this here rather than
        in a counter of the log's own.
        """
        self._done = done
        if total is not None:
            self._total = total
        if self._open is not None and self._throttle.due():
            _log_progress(self._label(self._open), self._elapsed(self._open),
                          done, self._total)

    def _retire(self, keep: str | None = None, reason: str | None = None) -> None:
        """Write out every phase that is over and forget it.

        `reason` (`failed`, `interrupted (...)`) applies to the phase that was still on
        the clock — it did not finish, and a plain summary would claim it did. The ones
        that finished earlier get their ordinary summary either way: keeping THOSE is
        the whole reason this happens on the way out of a broken run at all.
        """
        stopped: str | None = None
        if self._open is not None and self._open != keep:
            self._timings[self._open].seconds += time.perf_counter() - self._since
            stopped, self._open = self._open, None
        for phase in [p for p in self._timings if p != keep]:
            timing = self._timings.pop(phase)
            if phase == stopped and reason is not None:
                _LOG.info("%s %s elapsed=%.3f%s", self._label(phase), reason,
                          timing.seconds, _counters(timing.processed, timing.seconds))
            else:
                log_phase(self._stage, phase, timing.seconds, timing.processed)
        if keep is None:
            self._total = self._done = None

    def close(self, reason: str | None = None) -> None:
        """The stage is over: write out everything still on the books (F166).

        Idempotent — a stage has several exits, and `stage_timer` closes it again on
        the way out, so this must not be able to report the same seconds twice.
        """
        self._retire(reason=reason)
        self._throttle.reset()


# The phases of the stage currently running, by stage name. The registry exists for one
# reason: `stage_timer` wraps a stage from the OUTSIDE (cli/ui), while the phases are
# driven from the inside (junk), and the interrupted path has to be able to reach them
# without threading an object through every stage signature.
_PHASES: dict[str, StagePhases] = {}
_PHASES_LOCK = threading.Lock()


def track_phases(stage: str) -> StagePhases:
    """Open phase bookkeeping for `stage` and register it under that name (F166)."""
    tracker = StagePhases(stage)
    with _PHASES_LOCK:
        _PHASES[stage] = tracker
    return tracker


def _tracked(stage: str) -> StagePhases | None:
    with _PHASES_LOCK:
        return _PHASES.get(stage)


def _drop_phases(stage: str) -> StagePhases | None:
    """Unregister the phases of `stage` without writing anything."""
    with _PHASES_LOCK:
        return _PHASES.pop(stage, None)


def _close_phases(stage: str, reason: str | None) -> None:
    """Unregister the phases of `stage` and write out whatever they still hold."""
    tracker = _drop_phases(stage)
    if tracker is not None:
        tracker.close(reason)


@dataclass
class StageResult:
    """The handle a stage gets from `stage_timer`.

    The caller fills in `processed` when the count is known only at the end
    (`result.processed = 1234`) — the summary line then also carries the rate.

    F166: `progress` does the same job while the stage is still running, so a stage
    with no phases of its own is no less readable in the log than one that has them.
    """

    name: str
    total: int | None = None
    processed: int | None = None
    started: float = field(default_factory=time.perf_counter, compare=False)
    _throttle: _Throttle = field(default_factory=_Throttle, compare=False, repr=False)

    def progress(self, done: int, total: int | None = None) -> None:
        """Report where the stage is NOW — throttled into one line per interval.

        Silent while a phase of this stage is open: the phase line carries the very
        same counters and a name on top of them, and two heartbeats for one stage
        would be noise rather than detail (F166 requirement 7).
        """
        self.processed = done
        if total is not None:
            self.total = total
        phases = _tracked(self.name)
        if phases is not None and phases.open is not None:
            return
        if self._throttle.due():
            _log_progress(f"stage={self.name}", time.perf_counter() - self.started,
                          done, self.total)


class _Observed:
    """A stage's progress callback with the run log tapped into it (F166).

    A tap and not a counter of its own: the log is fed by the same call that moves the
    bar, so the two cannot come to disagree about where a run is — the rule F147 set
    for the phase names, applied to the numbers next to them. Everything the wrapped
    callback did still happens, including raising the pipeline's cancellation from
    inside `ui`.
    """

    def __init__(self, result: StageResult, callback: Callable[..., None]) -> None:
        self._result = result
        self._callback = callback
        inner = getattr(callback, "phase", None)
        self._phase: Callable[[str], None] | None = inner if callable(inner) else None

    def __call__(self, done: int, total: int | None = None) -> None:
        self._result.progress(done, total)
        self._callback(done, total)

    def phase(self, name: str) -> None:
        if self._phase is not None:
            self._phase(name)


def observe(result: StageResult, callback: Callable[..., None]) -> _Observed:
    """Tap the run log into the progress callback a stage is about to be handed."""
    return _Observed(result, callback)


@contextmanager
def stage_timer(name: str, *, total: int | None = None) -> Iterator[StageResult]:
    """Time a pipeline stage and write a machine-greppable summary line.

    The summary keeps a stable `stage=<name> elapsed=<sec> processed=<n>` prefix, so a
    profile of a run can be collected from the file without parsing prose. Nothing is
    ever swallowed — whatever came out is re-raised.

    Failures are logged as ERROR with a traceback; a BaseException that is NOT an
    Exception is control flow, not breakage (KeyboardInterrupt, SystemExit, and the
    pipeline's own cancellation, which subclasses BaseException precisely so that an
    `except Exception` inside a stage cannot eat it). Pressing "Cancel" used to print
    a full ERROR traceback, which reads as a crash for something the user asked for.

    F166: whichever of the three ways out it takes, the phases of the stage are closed
    FIRST — their lines belong above the total they add up to, and on the interrupted
    path they are the numbers the run would otherwise have taken down with it.
    """
    result = StageResult(name=name, total=total)
    # Anything a previous run of this stage left registered is stale by definition: the
    # stage is starting over, and its old phases must not be written into this run.
    _drop_phases(name)
    started = result.started = time.perf_counter()
    _log_started(f"stage={name}", total)
    try:
        yield result
    except Exception:
        elapsed = time.perf_counter() - started
        _close_phases(name, "failed")
        _LOG.error(
            "stage=%s failed elapsed=%.3f%s", name, elapsed, _counters(result.processed, elapsed),
            exc_info=True,
        )
        raise
    except BaseException as exc:
        elapsed = time.perf_counter() - started
        _close_phases(name, f"interrupted ({type(exc).__name__})")
        _LOG.info(
            "stage=%s interrupted (%s) elapsed=%.3f%s",
            name, type(exc).__name__, elapsed, _counters(result.processed, elapsed),
        )
        raise
    elapsed = time.perf_counter() - started
    _close_phases(name, None)
    _LOG.info("stage=%s elapsed=%.3f%s", name, elapsed, _counters(result.processed, elapsed))


# --- F159: reading the timings back, so an estimate stops carrying constants --------
#
# Everything above WRITES timings; this reads them. The run screen used to price a stage
# with a number measured once, on a developer's collection, and baked into `ui.py` — 1.32
# seconds for a comparative question that really costs 0.45 s plus 1.03 s per frame in it,
# a 3.7x understatement on the collection it was checked against. The file already holds
# the true rate of every stage ON THIS MACHINE, which is the number a person deciding
# whether to wait four hours actually wants.
#
# Only the SUMMARY lines are read. `started` and `progress` say nothing about a finished
# unit; `failed` and `interrupted (...)` describe one that stopped early, where the
# seconds are real but the denominator is not — a rate built from those would promise a
# run faster than any run has ever been. The three are told apart by shape: a summary is
# the only line where `elapsed=` follows the unit immediately.
_MEASUREMENT_RE = re.compile(
    r"^(?P<at>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\.\d+\s+\w+\s+\S+\s+\[[^\]]*\]\s+"
    r"stage=(?P<stage>[^\s=]+)(?:\s+phase=(?P<phase>[^\s=]+))?"
    r"\s+elapsed=(?P<elapsed>\d+(?:\.\d+)?)"
    r"(?:\s+processed=(?P<processed>\d+))?"
)
# The `  sorta: <version>` line of the environment header. It has no timestamp prefix —
# `log_environment` emits the whole header as ONE record — which is also why it can never
# collide with the pattern above.
_BUILD_RE = re.compile(r"^\s+sorta:\s*(?P<build>\S+)\s*$")

# How long a timing is worth trusting, in days. Ninety is deliberately generous: the guard
# that actually matters is the build below, and this one only catches the case that guard
# cannot see — the same version of the tool, running months later on a machine whose disk,
# GPU or collection has moved on since.
DEFAULT_MEASUREMENT_MAX_AGE_DAYS = 90.0


def measurement_unit(stage: str, phase: str | None = None) -> str:
    """The key one timed unit is remembered under — the log's own `stage=`/`phase=`.

    Callers name what they want to price in the same words the file uses, so there is no
    second vocabulary to keep in step with `log_phase`.
    """
    return f"stage={stage}" if phase is None else f"stage={stage} phase={phase}"


@dataclass(frozen=True)
class Measurement:
    """How fast one timed unit ran here, the last time it ran (F159).

    `processed` is not decoration: seconds alone cannot tell an expensive question asked
    rarely from a cheap one asked of every frame, which is exactly why F147 wrote the
    denominator next to the numerator in the first place.
    """

    unit: str
    seconds: float
    processed: int
    at: datetime
    build: str

    @property
    def seconds_per_unit(self) -> float:
        """Seconds per item — the rate an estimate multiplies a count by."""
        return self.seconds / self.processed


def _usable(seconds: float, processed: int) -> bool:
    """A rate needs both halves, and neither of them may be zero.

    `processed=0` cannot be divided by. `elapsed=0.000` reads as "instant", and the far
    likelier reading is a stage that recognised its whole population as already done
    (the F68 incremental skip) — pricing the next run at nothing on the strength of that
    is the one thing an estimate may not do. Falling back to the shipped default is the
    conservative direction, and it costs only accuracy.
    """
    return seconds > 0 and processed > 0


def _measurement_files(path: Path) -> list[Path]:
    """The log and its most recent backup, oldest first.

    The backup is read because rotation is not aware of runs: the 5 MB boundary can fall
    between the environment header of a run and the stage summaries that belong to it,
    and a measurement whose build is unknown is a measurement this module refuses to use.
    One backup is enough for that — going further back would only offer timings older
    than the ones already in hand.
    """
    return [p for p in (path.with_name(path.name + ".1"), path) if p.is_file()]


def measurement_files(path: str | Path | None = None) -> list[Path]:
    """The files `read_measurements` would read, oldest first; empty — there are none.

    Exposed so a caller that CACHES an answer built on them can key that cache on their
    state, the way the web app keys the run estimate on the state of the index: a run
    that has just written its own timings must not be answered with the old prices.
    """
    return _measurement_files(_resolve_path(path))


def read_measurements(
    path: str | Path | None = None,
    *,
    build: str | None = None,
    max_age_days: float = DEFAULT_MEASUREMENT_MAX_AGE_DAYS,
    now: datetime | None = None,
) -> dict[str, Measurement]:
    """The latest usable timing of every stage and phase in the run log, by unit name.

    A stale measurement is worse than none, because it is believed. Two guards, and both
    are the `frame_quality.source` device — an answer is kept only while the question
    behind it is still the same one:

    * the BUILD. Every run opens with an environment header carrying `sorta: <version>`,
      and a timing from another version is a timing of a stage that may since have been
      rewritten. Deliberately blunt: it discards timings that were still valid rather
      than keep one that is not, and a discarded timing costs only the default estimate.
      A timing no header vouches for is discarded on the same rule.
    * the AGE, `max_age_days`. 0 or less switches it off.

    Never raises and never blocks: an unreadable, missing or half-written log is simply a
    machine with no measurements yet, which is a case the caller has to handle anyway.
    """
    wanted = build if build is not None else _running_build()
    cutoff: datetime | None = None
    if max_age_days > 0:
        cutoff = (now or datetime.now()) - timedelta(days=max_age_days)
    found: dict[str, Measurement] = {}
    for source in _measurement_files(_resolve_path(path)):
        current_build: str | None = None
        try:
            with source.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    header = _BUILD_RE.match(line)
                    if header is not None:
                        current_build = header.group("build")
                        continue
                    if current_build != wanted:
                        continue
                    parsed = _parse_measurement(line, current_build, cutoff)
                    if parsed is not None:
                        found[parsed.unit] = parsed
        except OSError as exc:
            _LOG.debug("runlog: run log %s is unreadable (%s)", source, exc)
    return found


def _parse_measurement(line: str, build: str, cutoff: datetime | None) -> Measurement | None:
    """One log line -> a Measurement, or None if it is not a usable summary."""
    match = _MEASUREMENT_RE.match(line)
    if match is None or match.group("processed") is None:
        return None
    seconds, processed = float(match.group("elapsed")), int(match.group("processed"))
    if not _usable(seconds, processed):
        return None
    try:
        at = datetime.strptime(match.group("at"), _DATEFMT)
    except ValueError:
        return None
    if cutoff is not None and at < cutoff:
        return None
    return Measurement(
        unit=measurement_unit(match.group("stage"), match.group("phase")),
        seconds=seconds, processed=processed, at=at, build=build,
    )


def _running_build() -> str:
    """The version of the package doing the asking — see `read_measurements`."""
    try:
        from . import __version__

        return str(__version__)
    except Exception:  # a source tree without the package metadata is not fatal
        return "unknown"


def _package_origin() -> str:
    """Where the running `sorta` package was imported from.

    This is the line that tells an installed uv-tool apart from the repository working
    tree — the root cause of F65 (the geo database was missing from the wheel while the
    same command worked from the repo).
    """
    package_dir = Path(__file__).resolve().parent
    parts = {part.lower() for part in package_dir.parts}
    if (package_dir.parent / "pyproject.toml").exists():
        origin = "repo working tree"
    elif "site-packages" in parts or "dist-packages" in parts:
        origin = "installed (site-packages)"
    else:
        origin = "unknown"
    return f"{package_dir} ({origin})"


def _gpu_line() -> str:
    """torch/onnxruntime CUDA availability — but only if somebody has already loaded them.

    Asking the diagnostics layer costs 13.96 s (measured 2026-08-08, warm cache, fast
    machine) because the answer means importing torch, and the base tier needs torch for
    nothing. `sorta doctor` probes for real, on request.
    """
    if "torch" not in sys.modules and "onnxruntime" not in sys.modules:
        return "not loaded (ask `sorta doctor` for the real answer)"
    try:
        from . import diagnostics

        return diagnostics.gpu_health().summary
    except Exception as exc:  # diagnostics/torch/onnxruntime unavailable — not fatal
        return f"unavailable ({exc})"


def _geo_data_dir() -> Path:
    """Where the resolver will actually read the base from — package dir, else legacy."""
    if (_LEGACY_GEO_DATA_DIR / _PLACES_FILE).is_file() \
            and not (_GEO_DATA_DIR / _PLACES_FILE).is_file():
        return _LEGACY_GEO_DATA_DIR
    return _GEO_DATA_DIR


def _geo_line() -> str:
    """The bundled geo data: path + whether it is actually there (F65)."""
    try:
        data_dir = _geo_data_dir()
        directory = "yes" if data_dir.is_dir() else "no"
        places = "yes" if (data_dir / _PLACES_FILE).is_file() else "no"
        return f"{data_dir} (каталог: {directory}, places.tsv: {places})"
    except Exception as exc:
        return f"{_GEO_DATA_DIR} (проверка не удалась: {exc})"


def _safe(getter: Callable[[], object]) -> str:
    """Call a no-argument getter, turning any failure into a readable value."""
    try:
        return str(getter())
    except Exception as exc:
        return f"недоступно ({exc})"


def log_environment() -> None:
    """Write the environment header once at the start of a run.

    Exactly what was missing to catch F65 and the VLM fallback right away. Never
    raises and never loads ML models: an unavailable library becomes a "недоступно"
    line. Emitted as ONE record so it is not interleaved with the pipeline thread.
    """
    try:
        from . import __version__

        version = __version__
    except Exception:
        version = "unknown"

    try:
        gpu = _gpu_line().replace("\n", "; ")  # the diagnostics summary is multi-line
        lines = [
            "environment:",
            f"  sorta: {version}",
            f"  python: {sys.version.split()[0]} ({sys.executable})",
            f"  platform: {_safe(platform.platform)}",
            f"  package: {_safe(_package_origin)}",
            f"  gpu: {gpu}",
            f"  geo data: {_geo_line()}",
            f"  cwd: {_safe(os.getcwd)}",
        ]
        _LOG.info("\n".join(lines))
    except Exception as exc:  # the header must never take a run down
        _LOG.warning("runlog: could not assemble the environment header (%s)", exc)
