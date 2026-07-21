"""Metadata reading: exiftool (preferred) or Pillow (fallback).

exiftool covers HEIC/RAW/video; Pillow — only jpeg/png/tiff/webp.
exiftool runs through one long-lived process (-stay_open) — the process-startup
cost is not paid per batch; on a session failure it falls back to a one-shot
subprocess call.
The interface is uniform: read_batch(paths) -> dict[path, ExifData].
"""
from __future__ import annotations

import atexit
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

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


def exiftool_available() -> bool:
    return shutil.which("exiftool") is not None


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
_EXIFTOOL_CMD = ["exiftool"]

# Arguments for each query; in a -stay_open session we additionally declare the
# stdin-argfile encoding (a Windows-only exiftool option) since we write it in UTF-8.
_QUERY_ARGS = ["-json", "-n", "-fast2"]
_SESSION_ARGS = _QUERY_ARGS + (
    ["-charset", "filename=utf8"] if sys.platform == "win32" else []
)


class ExifToolSession:
    """Long-lived process `exiftool -stay_open True -@ -` (FR-1 item 7).

    Protocol: query arguments are written to stdin one per line, then `-execute`;
    the response is read up to the `{ready}` marker. The pipes are binary (UTF-8 by
    hand) — this avoids the text-mode \\n -> \\r\\n translation. A dead process is
    restarted transparently on the next read().
    """

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None

    def _ensure(self) -> subprocess.Popen:
        if self._proc is None or self._proc.poll() is not None:
            self._proc = subprocess.Popen(
                [*_EXIFTOOL_CMD, "-stay_open", "True", "-@", "-"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        return self._proc

    def read(self, paths: list[Path]) -> dict[str, ExifData]:
        if not paths:
            return {}
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
        proc, self._proc = self._proc, None
        if proc is None or proc.poll() is not None:
            return
        try:
            assert proc.stdin is not None
            proc.stdin.write(b"-stay_open\nFalse\n")
            proc.stdin.flush()
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


_session = ExifToolSession()
atexit.register(_session.close)


def read_batch_exiftool(paths: list[Path], chunk: int = 200) -> dict[str, ExifData]:
    """One-shot batch exiftool call (fallback if the -stay_open session broke)."""
    out: dict[str, ExifData] = {}
    for i in range(0, len(paths), chunk):
        batch = [str(p) for p in paths[i:i + chunk]]
        proc = subprocess.run(
            [*_EXIFTOOL_CMD, *_QUERY_ARGS, *_EXIFTOOL_TAGS, *batch],
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


def read_batch(paths: list[Path]) -> dict[str, ExifData]:
    if exiftool_available():
        try:
            return _session.read(paths)
        except Exception:
            _session.close()
            return read_batch_exiftool(paths)
    return {str(p.resolve()): read_one_pillow(p) for p in paths}
