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
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import platform
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
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

# Mirrors `geodata._DEFAULT_DATA_DIR`. Deliberately not imported: `geodata` pulls in
# numpy/scipy at import time, and the environment header must stay cheap and safe.
_GEO_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "geo"


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


@dataclass
class StageResult:
    """The handle a stage gets from `stage_timer`.

    The caller fills in `processed` when the count is known only at the end
    (`result.processed = 1234`) — the summary line then also carries the rate.
    """

    name: str
    total: int | None = None
    processed: int | None = None


def _counters(result: StageResult, elapsed: float) -> str:
    """The ` processed=<n> rate=<n>/s` tail — empty if the caller reported nothing."""
    if result.processed is None:
        return ""
    tail = f" processed={result.processed}"
    if elapsed > 0:
        tail += f" rate={result.processed / elapsed:.1f}/s"
    return tail


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
    """
    result = StageResult(name=name, total=total)
    started = time.perf_counter()
    if total is None:
        _LOG.info("stage=%s started", name)
    else:
        _LOG.info("stage=%s started total=%d", name, total)
    try:
        yield result
    except Exception:
        elapsed = time.perf_counter() - started
        _LOG.error(
            "stage=%s failed elapsed=%.3f%s", name, elapsed, _counters(result, elapsed),
            exc_info=True,
        )
        raise
    except BaseException as exc:
        elapsed = time.perf_counter() - started
        _LOG.info(
            "stage=%s interrupted (%s) elapsed=%.3f%s",
            name, type(exc).__name__, elapsed, _counters(result, elapsed),
        )
        raise
    elapsed = time.perf_counter() - started
    _LOG.info("stage=%s elapsed=%.3f%s", name, elapsed, _counters(result, elapsed))


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
    """torch/onnxruntime CUDA availability via the existing diagnostics layer."""
    try:
        from . import diagnostics

        return diagnostics.gpu_health().summary
    except Exception as exc:  # diagnostics/torch/onnxruntime unavailable — not fatal
        return f"недоступны ({exc})"


def _geo_line() -> str:
    """The bundled geo data: path + whether it is actually there (F65)."""
    try:
        directory = "yes" if _GEO_DATA_DIR.is_dir() else "no"
        places = "yes" if (_GEO_DATA_DIR / "places.tsv").is_file() else "no"
        return f"{_GEO_DATA_DIR} (каталог: {directory}, places.tsv: {places})"
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
        _LOG.warning("runlog: не удалось собрать заголовок окружения (%s)", exc)
