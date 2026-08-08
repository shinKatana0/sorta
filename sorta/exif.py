"""Metadata reading: exiftool (preferred) or Pillow (fallback).

exiftool covers HEIC/RAW/video; Pillow — only jpeg/png/tiff/webp. Which exiftool that
is — the one on PATH or the one the installer shipped — is `resolve_exiftool` below.
exiftool runs through a pool of long-lived processes (-stay_open) — the
process-startup cost is not paid per batch; on a session failure that session's
slice falls back to a one-shot subprocess call.
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
# The Windows installer carries a 25 MB exiftool and the manifest names it — and until
# F226 this module looked for the binary by ONE name on PATH, which nothing in the
# install ever puts there (neither `sorta.iss` nor `sitecustomize.py` mentions exiftool).
# So an installed copy fell straight through to Pillow and read no HEIC/RAW/video dates,
# no GPS and no orientation at all, while the wizard's screen said in as many words that
# "exiftool ships with the program". The resolution below is one half of making that
# sentence true; the other half is in `scripts/build_installer.py`, which shipped the
# .exe without the `exiftool_files\` directory it cannot start without.
#
# The order is PATH first, the shipped binary second:
#
#   * a machine where somebody installed exiftool on purpose (this project's own, via
#     winget) has to keep behaving exactly as it did — that copy is also the one a person
#     can update, and it is the one they will expect to be used;
#   * the shipped binary is the answer for the machine that has nothing, which is every
#     machine the installer is FOR.
#
# There is deliberately no config key in front of those two: `config.py` has none today,
# and a setting invented here would be one nobody asked for and nobody maintains.
#
# The machine's PATH is left alone in either case — what goes into the command is the
# absolute path, so nothing this program does changes what `exiftool` means in a shell.

# `exiftool -ver` on a healthy binary answers immediately; this bound exists so a wedged
# one costs a diagnostic pause rather than a hung index.
_PROBE_TIMEOUT = 15


def _starts(binary: str) -> bool:
    """Does this file actually run as exiftool? — `Path.exists()` is not that question.

    The Windows build is `exiftool.exe` PLUS an `exiftool_files\\` directory beside it,
    and without that directory the .exe is still there, still named right, and does not
    start. A manifest that names a binary whose payload half went missing would otherwise
    win over the working copy on PATH, which is the opposite of the point. Asking for the
    version is the cheap way to tell the two apart, and it is asked once per process.
    """
    try:
        proc = launch.run([binary, "-ver"], capture_output=True, text=True,
                          timeout=_PROBE_TIMEOUT)
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0 and bool(proc.stdout.strip())


def resolve_exiftool(*, which: Callable[[str], str | None] | None = None,
                     runs: Callable[[str], bool] | None = None,
                     manifest: dict | None = None) -> str | None:
    """The exiftool to run, by the order above — or None when there is none.

    Every dependency is injectable and resolved at CALL time rather than bound as a
    default, so a caller (and the suite) can ask the question for a machine other than
    this one; `sorta doctor` does exactly that.
    """
    finder = shutil.which if which is None else which
    on_path = finder("exiftool")
    if on_path:
        return on_path
    shipped = install.tool_path(
        install.load_manifest() if manifest is None else manifest, "exiftool")
    probe = _starts if runs is None else runs
    return shipped if shipped and probe(shipped) else None


# Resolved once per process and remembered: `read_batch` asks per batch, and the shipped
# branch of the answer costs a subprocess. A one-element tuple rather than a bare string
# so that "resolved to nothing" and "not resolved yet" stay different states.
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
# None means "nobody has overridden it", and then the command is whatever
# `exiftool_binary()` resolved — the shipped binary's absolute path on an installed copy.
_EXIFTOOL_CMD: list[str] | None = None


def _exiftool_cmd() -> list[str]:
    if _EXIFTOOL_CMD is not None:
        return _EXIFTOOL_CMD
    # The bare name is the last resort and not a fallback anyone should reach: a caller
    # that got here past `exiftool_available()` is a one-shot read on a machine where
    # nothing was found, and letting the OS answer is closer to the old behaviour than
    # raising would be.
    return [exiftool_binary() or "exiftool"]

# Arguments for each query; in a -stay_open session we additionally declare the
# stdin-argfile encoding (a Windows-only exiftool option) since we write it in UTF-8.
#
# F71 — do NOT put `-fast2` back here for speed. `-fast2` makes exiftool stop reading
# very early, and in HEIC the metadata block sits AFTER the image data: it is never
# reached. Measured on the production collection (40 287 files): 11 584 files (29%)
# had no camera_make in the index and exactly 0 of them had GPS — the whole block was
# lost, not single tags; 62.7% of all HEIC. On 150 such files, metadata was read for
# 0 of them with `-fast2`, 109 with `-fast`, 109 with no flag at all. The cost of
# completeness is 1.6x, not the 7.5x of a full read (17.0 -> 27.1 -> 127.3 ms per file),
# and on 250 mixed files all nine requested tags matched a full read exactly.
_QUERY_ARGS = ["-json", "-n", "-fast"]
_SESSION_ARGS = _QUERY_ARGS + (
    ["-charset", "filename=utf8"] if sys.platform == "win32" else []
)


class UnsafeExifPath(ValueError):
    """A path that must not be handed to exiftool (F208) — see `_require_absolute`.

    A ValueError so an ordinary caller need not know the name; its own type so the
    fallback below can tell a refusal apart from a session that died (that one is
    retried one-shot, this one must not be retried at all).
    """


def _require_absolute(paths: list[Path]) -> None:
    """Refuse a relative path before it can become an exiftool argument (F208).

    exiftool reads any argument starting with `-` as an OPTION, and the paths are passed
    to it with no `--` separator in front of them (it has none). One of those options is
    `-config`, which loads a Perl file — so a file NAMED like an option would be executed
    instead of read.

    Today nothing can reach here with such a name: the indexer resolves its root
    (`Path(src).expanduser().resolve()`), so every path is absolute and cannot begin with
    a dash. But that is an invariant held somewhere else and checked nowhere, and a single
    future caller passing a relative path would open the hole with nothing to notice it.
    Hence the check at the boundary, on both ways into exiftool.
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

    Protocol: query arguments are written to stdin one per line, then `-execute`;
    the response is read up to the `{ready}` marker. The pipes are binary (UTF-8 by
    hand) — this avoids the text-mode \\n -> \\r\\n translation. A dead process is
    restarted transparently on the next read().

    One request at a time: the pipes are a single request/response stream, so two
    threads writing into the same session would interleave queries and answers.
    The lock makes that impossible instead of merely unlikely (F72).
    """

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    def _ensure(self) -> subprocess.Popen:
        if self._proc is None or self._proc.poll() is not None:
            if self._proc is not None:
                _close_pipes(self._proc)
            # F228: `launch.popen` and not `subprocess.Popen` — there are as many of these
            # as there are read workers, so a run from the shortcut used to open up to
            # eight console windows in a row on the index stage.
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


# Below this many paths per session the split is not worth it: eight processes on five
# files only add scheduling noise (the sessions are already warm, but exiftool still
# parses the argfile and re-emits JSON per slice).
_MIN_PATHS_PER_SESSION = 32


def resolve_exif_workers(raw: dict | None) -> int:
    """Number of parallel exiftool sessions — same shape as hashing.resolve_workers.

    `index.exif_workers` in config.yaml (read straight from `cfg.raw`, no typed field
    is added for it); default min(8, cpu_count). exiftool is a separate process, so the
    GIL does not cap it and it scales nearly linearly: measured on the production
    collection (40 287 files) 11.8 ms/file with one session, 5.8 with two, 3.2 with
    four, 2.0 with eight (F72).
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
    """N long-lived exiftool sessions serving one read_batch in parallel (F72).

    The processes are created once for the whole run and reused across calls —
    re-spawning them per batch would throw away the point of -stay_open. Creation is
    lazy: a command that never reads metadata never spawns a single exiftool.
    """

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

    `workers` — how many exiftool sessions may share the batch; None (and any value
    <= 0) means the default of resolve_exif_workers. The caller passes the configured
    value in (the indexer does) so this module stays independent of Config.
    """
    if exiftool_available():
        n = workers if workers is not None and workers > 0 else resolve_exif_workers(None)
        return _pool.read(paths, n)
    return {str(p.resolve()): read_one_pillow(p) for p in paths}
