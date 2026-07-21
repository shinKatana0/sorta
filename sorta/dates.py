"""Capture-date cascade: EXIF -> filename -> mtime.

Filename heuristics cover: IMG_YYYYMMDD_HHMMSS, PXL_, VID_, YYYYMMDD_HHMMSS,
IMG-YYYYMMDD-WAxxxx (WhatsApp), "WhatsApp Image YYYY-MM-DD at HH.MM.SS",
photo_YYYY-MM-DD_HH-MM-SS (Telegram), Screenshot_YYYYMMDD-HHMMSS, bare dates.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

# (pattern, has_time). Order matters: from specific to general.
_PATTERNS: list[tuple[re.Pattern, bool]] = [
    # 20190705_123456 / IMG_20190705_123456 / Screenshot_20200101-101112,
    # and also PXL_20210101_101112000 (Pixel, optional milliseconds)
    (re.compile(r"(?<!\d)(\d{4})(\d{2})(\d{2})[-_](\d{2})(\d{2})(\d{2})(?:\d{3})?(?!\d)"), True),
    # WhatsApp Image 2020-05-01 at 12.34.56 / photo_2019-07-05_12-34-56 (Telegram)
    (re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})[ _](?:at[ _])?(\d{2})[.\-:](\d{2})[.\-:](\d{2})(?!\d)"), True),
    # IMG-20190705-WA0001 (WhatsApp, date only)
    (re.compile(r"(?<!\d)(\d{4})(\d{2})(\d{2})-WA\d+", re.IGNORECASE), False),
    # 2019-07-05 (date only)
    (re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)"), False),
    # 20190705 (date only, the riskiest — kept last)
    (re.compile(r"(?<!\d)(\d{4})(\d{2})(\d{2})(?!\d)"), False),
]


@dataclass
class TakenAt:
    dt: datetime
    source: str       # exif | filename | mtime
    confidence: str   # high | medium | low


def _valid(dt: datetime, min_year: int, max_year: int) -> bool:
    return min_year <= dt.year <= max_year


def parse_exif_datetime(value: str | None, min_year: int = 1990, max_year: int = 2035) -> datetime | None:
    """EXIF format 'YYYY:MM:DD HH:MM:SS' with variations."""
    if not value:
        return None
    v = value.strip()
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y:%m:%d"):
        try:
            dt = datetime.strptime(v[:19], fmt)
            if _valid(dt, min_year, max_year):
                return dt
        except ValueError:
            continue
    return None


def parse_filename_datetime(name: str, min_year: int = 1990, max_year: int = 2035) -> datetime | None:
    for pattern, has_time in _PATTERNS:
        m = pattern.search(name)
        if not m:
            continue
        g = [int(x) for x in m.groups() if x is not None]
        try:
            dt = (datetime(g[0], g[1], g[2], g[3], g[4], g[5]) if has_time
                  else datetime(g[0], g[1], g[2]))
        except ValueError:
            continue
        if _valid(dt, min_year, max_year):
            return dt
    return None


def resolve_taken_at(
    exif_value: str | None,
    filename: str,
    mtime: float,
    min_year: int = 1990,
    max_year: int = 2035,
) -> TakenAt:
    """Cascade: EXIF (high) -> filename (medium) -> mtime (low)."""
    if dt := parse_exif_datetime(exif_value, min_year, max_year):
        return TakenAt(dt, "exif", "high")
    if dt := parse_filename_datetime(filename, min_year, max_year):
        return TakenAt(dt, "filename", "medium")
    return TakenAt(datetime.fromtimestamp(mtime), "mtime", "low")
