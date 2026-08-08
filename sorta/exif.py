"""Metadata reading: exiftool (preferred) or Pillow (fallback).

exiftool covers HEIC/RAW/video, Pillow only jpeg/png/tiff/webp. Which exiftool that is —
PATH or the one the installer shipped — is `resolve_exiftool` below. It runs through a
pool of long-lived processes (-stay_open), so startup is not paid per batch, and a
session that dies costs its own slice a one-shot call.
The interface is uniform: read_batch(paths) -> dict[path, ExifData].
"""
from __future__ import annotations

import atexit
import json
import math
import os
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import install, launch

_EXIFTOOL_TAGS = [
    "-DateTimeOriginal", "-CreateDate", "-GPSLatitude", "-GPSLongitude",
    "-Make", "-Model", "-ImageWidth", "-ImageHeight", "-Orientation",
]


@dataclass
class ExifData:
    datetime_original: str | None = None
    gps_lat: float | None = None
    gps_lon: float | None = None
    make: str | None = None
    model: str | None = None
    width: int | None = None
    height: int | None = None
    orientation: int | None = None  # EXIF 274: 1..8, numeric value (-n)


# --- which exiftool this copy uses (F226) --------------------------------------------
# PATH first, the 25 MB binary the Windows installer carries second: a machine where
# somebody installed exiftool on purpose keeps the copy it can update, and the shipped one
# answers for the machine that has nothing — every machine the installer is FOR. No config
# key in front of the two. PATH itself is untouched: the command carries an absolute path,
# so `exiftool` in a shell still means what it did.

# `exiftool -ver` on a healthy binary answers immediately; this bound exists so a wedged
# one costs a diagnostic pause rather than a hung index.
_PROBE_TIMEOUT = 15


def _starts(binary: str) -> bool:
    """Does this file actually run as exiftool? `Path.exists()` is not that question: the
    Windows build is `exiftool.exe` PLUS an `exiftool_files\\` directory beside it, and
    without the directory the .exe is still there, named right, and does not start."""
    try:
        proc = launch.run([binary, "-ver"], capture_output=True, text=True,
                          timeout=_PROBE_TIMEOUT)
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0 and bool(proc.stdout.strip())


def resolve_exiftool(*, which: Callable[[str], str | None] | None = None,
                     runs: Callable[[str], bool] | None = None,
                     manifest: dict | None = None) -> str | None:
    """The exiftool to run, by the order above — or None when there is none. Every
    dependency resolves at CALL time rather than as a default, so a caller can ask the
    question for a machine other than this one; `sorta doctor` does."""
    finder = shutil.which if which is None else which
    on_path = finder("exiftool")
    if on_path:
        return on_path
    shipped = install.tool_path(
        install.load_manifest() if manifest is None else manifest, "exiftool")
    probe = _starts if runs is None else runs
    return shipped if shipped and probe(shipped) else None


# Resolved once per process: `read_batch` asks per batch and the shipped branch costs a
# subprocess. A one-element tuple, so "resolved to nothing" and "not resolved yet" stay
# different states.
_resolved: tuple[str | None] | None = None


def exiftool_binary(*, refresh: bool = False) -> str | None:
    """The path of the exiftool this process uses, or None if it has none."""
    global _resolved
    if refresh or _resolved is None:
        _resolved = (resolve_exiftool(),)
    return _resolved[0]


def exiftool_available() -> bool:
    return exiftool_binary() is not None


def _parse_records(records: list[dict]) -> dict[str, ExifData]:
    out: dict[str, ExifData] = {}
    for rec in records:
        out[str(Path(rec["SourceFile"]).resolve())] = ExifData(
            datetime_original=rec.get("DateTimeOriginal") or rec.get("CreateDate"),
            gps_lat=_to_float(rec.get("GPSLatitude")),
            gps_lon=_to_float(rec.get("GPSLongitude")),
            make=rec.get("Make"),
            model=rec.get("Model"),
            width=rec.get("ImageWidth"),
            height=rec.get("ImageHeight"),
            orientation=_to_int(rec.get("Orientation")),
        )
    return out


# exiftool command; in tests it is replaced with a fake script to check the protocol.
# None means "nobody has overridden it" — then it is whatever `exiftool_binary()` found.
_EXIFTOOL_CMD: list[str] | None = None


def _exiftool_cmd() -> list[str]:
    if _EXIFTOOL_CMD is not None:
        return _EXIFTOOL_CMD
    # The bare name is a last resort, reached only by a caller that skipped
    # `exiftool_available()`: letting the OS answer is closer to the old behaviour.
    return [exiftool_binary() or "exiftool"]

# Arguments for each query; in a -stay_open session we additionally declare the
# stdin-argfile encoding (a Windows-only exiftool option) since we write it in UTF-8.
#
# F71 — do NOT put `-fast2` back here for speed: it stops reading before the metadata
# block, which in HEIC sits AFTER the image data. On the production collection (40 287
# files) that cost 11 584 files (29%, 62.7% of all HEIC) their whole block: no
# camera_make and GPS for exactly 0 of them. On 150 such files metadata was read for 0
# with `-fast2`, 109 with `-fast`, 109 with no flag. `-fast` costs 1.6x and not the 7.5x
# of a full read (17.0 -> 27.1 -> 127.3 ms per file), and on 250 mixed files all nine
# requested tags matched a full read exactly.
_QUERY_ARGS = ["-json", "-n", "-fast"]
_SESSION_ARGS = _QUERY_ARGS + (
    ["-charset", "filename=utf8"] if sys.platform == "win32" else []
)


class UnsafeExifPath(ValueError):
    """A path that must not be handed to exiftool (F208) — see `_require_absolute`. Its
    own type so the fallback below can tell a refusal apart from a dead session: that one
    is retried one-shot, this one must not be retried at all."""


def _require_absolute(paths: list[Path]) -> None:
    """Refuse a relative path before it can become an exiftool argument (F208).

    exiftool reads any argument starting with `-` as an OPTION and has no `--` separator
    to stop it; one of those options is `-config`, which loads a Perl file — a file NAMED
    like an option would be executed instead of read. Nothing can reach here with such a
    name today (the indexer resolves its root), but that invariant is held elsewhere and
    checked nowhere, so both ways into exiftool check it at the boundary.
    """
    for path in paths:
        if not Path(path).is_absolute():
            raise UnsafeExifPath(f"exiftool: path must be absolute, got {str(path)!r}")


def _close_pipes(proc: subprocess.Popen) -> None:
    """A dead exiftool still owns its pipes — flushing them later raises EINVAL."""
    for pipe in (proc.stdin, proc.stdout):
        if pipe is not None:
            try:
                pipe.close()
            except OSError:
                pass


class ExifToolSession:
    """Long-lived process `exiftool -stay_open True -@ -` (FR-1 item 7).

    Protocol: query arguments to stdin one per line, then `-execute`; the response is read
    up to the `{ready}` marker. The pipes are binary (UTF-8 by hand) to avoid the
    text-mode \\n -> \\r\\n translation. A dead process is restarted on the next read().

    One request at a time (F72): the pipes are a single request/response stream, and two
    threads in one session would interleave queries and answers.
    """

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    def _ensure(self) -> subprocess.Popen:
        if self._proc is None or self._proc.poll() is not None:
            if self._proc is not None:
                _close_pipes(self._proc)
            # F228: `launch.popen` and not `subprocess.Popen` — one per read worker, so a
            # run from the shortcut used to open up to eight console windows in a row.
            self._proc = launch.popen(
                [*_exiftool_cmd(), "-stay_open", "True", "-@", "-"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        return self._proc

    def read(self, paths: list[Path]) -> dict[str, ExifData]:
        if not paths:
            return {}
        _require_absolute(paths)
        with self._lock:
            proc = self._ensure()
            assert proc.stdin is not None and proc.stdout is not None
            args = [*_SESSION_ARGS, *_EXIFTOOL_TAGS, *map(str, paths)]
            proc.stdin.write(("\n".join(args) + "\n-execute\n").encode("utf-8"))
            proc.stdin.flush()
            buf = bytearray()
            while True:
                line = proc.stdout.readline()
                if not line:
                    raise RuntimeError("exiftool -stay_open: process exited before {ready}")
                if line.strip().startswith(b"{ready"):
                    break
                buf += line
        payload = buf.decode("utf-8", errors="replace").strip()
        return _parse_records(json.loads(payload)) if payload else {}

    def close(self) -> None:
        with self._lock:
            proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.poll() is None:
                assert proc.stdin is not None
                proc.stdin.write(b"-stay_open\nFalse\n")
                proc.stdin.flush()
                proc.wait(timeout=5)
        except Exception:
            proc.kill()
        finally:
            _close_pipes(proc)


# Below this many paths per session the split only adds scheduling noise: the sessions
# are warm, but exiftool still parses an argfile and re-emits JSON per slice.
_MIN_PATHS_PER_SESSION = 32


def resolve_exif_workers(raw: dict | None) -> int:
    """Number of parallel exiftool sessions — same shape as hashing.resolve_workers.

    `index.exif_workers` in config.yaml (read straight from `cfg.raw`); default
    min(8, cpu_count). A separate process is not capped by the GIL and scales nearly
    linearly: on the production collection (40 287 files) 11.8 ms/file with one session,
    5.8 with two, 3.2 with four, 2.0 with eight (F72).
    """
    default = min(8, os.cpu_count() or 1)
    idx = (raw or {}).get("index") or {}
    workers = idx.get("exif_workers")
    if workers is None:
        return default
    try:
        n = int(workers)
    except (TypeError, ValueError):
        return default
    return n if n > 0 else default


def _split(paths: list[Path], parts: int) -> list[list[Path]]:
    """Contiguous slices of near-equal size: every path lands in exactly one of them."""
    size, rest = divmod(len(paths), parts)
    out: list[list[Path]] = []
    start = 0
    for i in range(parts):
        end = start + size + (1 if i < rest else 0)
        out.append(paths[start:end])
        start = end
    return out


def _slice_count(n_paths: int, workers: int) -> int:
    return max(1, min(workers, math.ceil(n_paths / _MIN_PATHS_PER_SESSION)))


def _read_slice(session: ExifToolSession, paths: list[Path]) -> dict[str, ExifData]:
    """One slice through its own session; a broken session only costs its own slice."""
    try:
        return session.read(paths)
    except UnsafeExifPath:
        raise  # F208: a refused path is not a broken session — do not restart, do not retry
    except Exception:
        session.close()  # _ensure() starts a fresh process on the next call
        return read_batch_exiftool(paths)


class ExifToolPool:
    """N long-lived exiftool sessions serving one read_batch in parallel (F72). Created
    once for the whole run — re-spawning per batch would throw away the point of
    -stay_open — and lazily, so a command that reads no metadata spawns nothing."""

    def __init__(self) -> None:
        self._sessions: list[ExifToolSession] = []
        self._lock = threading.Lock()

    def sessions(self, count: int) -> list[ExifToolSession]:
        with self._lock:
            while len(self._sessions) < count:
                self._sessions.append(ExifToolSession())
            return self._sessions[:count]

    def read(self, paths: list[Path], workers: int) -> dict[str, ExifData]:
        if not paths:
            return {}
        chunks = _split(paths, _slice_count(len(paths), workers))
        sessions = self.sessions(len(chunks))
        if len(chunks) == 1:
            return _read_slice(sessions[0], chunks[0])
        out: dict[str, ExifData] = {}
        with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
            for part in pool.map(_read_slice, sessions, chunks):
                out.update(part)  # keyed by resolved path — merge order does not matter
        return out

    def close(self) -> None:
        with self._lock:
            sessions, self._sessions = self._sessions, []
        for session in sessions:
            session.close()


_pool = ExifToolPool()
atexit.register(_pool.close)  # leftover `exiftool -stay_open` processes would hang around


def read_batch_exiftool(paths: list[Path], chunk: int = 200) -> dict[str, ExifData]:
    """One-shot batch exiftool call (fallback if the -stay_open session broke)."""
    _require_absolute(paths)
    out: dict[str, ExifData] = {}
    for i in range(0, len(paths), chunk):
        batch = [str(p) for p in paths[i:i + chunk]]
        proc = launch.run(
            [*_exiftool_cmd(), *_QUERY_ARGS, *_EXIFTOOL_TAGS, *batch],
            capture_output=True, text=True,
        )
        if not proc.stdout.strip():
            continue
        out.update(_parse_records(json.loads(proc.stdout)))
    return out


def _to_int(v) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _to_float(v) -> float | None:
    """GPS may arrive from exiftool as an empty string/garbage — coerce to float|None."""
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _to_deg(v, ref) -> float | None:
    try:
        d, m, s = (float(x) for x in v)
        deg = d + m / 60 + s / 3600
        return -deg if ref in ("S", "W") else deg
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def read_one_pillow(path: Path) -> ExifData:
    from PIL import Image
    data = ExifData()
    try:
        with Image.open(path) as img:
            data.width, data.height = img.size
            exif = img.getexif()
            if not exif:
                return data
            data.datetime_original = exif.get(36867) or exif.get(306)  # DateTimeOriginal | DateTime
            data.make, data.model = exif.get(271), exif.get(272)
            data.orientation = _to_int(exif.get(274))
            gps = exif.get_ifd(34853) if hasattr(exif, "get_ifd") else None
            if gps:
                data.gps_lat = _to_deg(gps.get(2), gps.get(1))
                data.gps_lon = _to_deg(gps.get(4), gps.get(3))
    except Exception:
        pass  # the error is recorded by the indexer at its level
    return data


def read_batch(paths: list[Path], workers: int | None = None) -> dict[str, ExifData]:
    """Metadata for a batch of paths, keyed by the resolved absolute path.

    `workers` — how many sessions may share the batch; None (and anything <= 0) means the
    default. The caller passes the configured value in, which keeps this module
    independent of Config.
    """
    if exiftool_available():
        n = workers if workers is not None and workers > 0 else resolve_exif_workers(None)
        return _pool.read(paths, n)
    return {str(p.resolve()): read_one_pillow(p) for p in paths}
