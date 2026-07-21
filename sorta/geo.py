"""F2/G2 (Phase 2): the geo layer.

Contract: reads files (gps_lat/gps_lon, taken_at), writes ONLY into places.
Does not touch files, faces, events, moves.

Confidence levels:
- exact_gps        — coordinates from EXIF, offline resolve via geodata.GeoResolver
- session_inferred — place inherited from a file with GPS in the same time session
- unknown          — could not resolve (visual — landmarks.py, Phase 5/F6)

The provider is chosen by `cfg.geo.provider`: offline (default) — bundled GeoNames
via geodata.GeoResolver; online (G2b) — Nominatim/OSM reverse geocoding, names as
text already in cfg.language (no geonameids).

Canonically we write geonameid (city_geonameid/district_geonameid) + the English/
asciiname anchor `city` (for --where/CSV/landmark fallback). Localizing names into
the target language is sort's job (G3), not this module's. `region` — DEPRECATED, no
longer written (stays NULL). `district_name` — online only (district name as text,
offline leaves it NULL and writes geonameid into district_geonameid).

Idempotency: a re-run fully recomputes places.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Protocol

from .config import Config
from .geodata import GeoResolver
from .i18n import Lang

_log = logging.getLogger(__name__)

_PROGRESS_EVERY = 1000
_INHERIT_CONFIDENCE = ("high", "medium")
_CANONICAL_LANG: Lang = "en"  # the city anchor is always English/asciiname — not localized here
_NOMINATIM_MIN_INTERVAL = 1.0  # OSM policy: no more than 1 request/sec
# coordinate rounding for the in-memory cache — now from cfg.geo.cache_coord_digits


@dataclass
class GeoStats:
    total: int = 0
    exact_gps: int = 0
    session_inferred: int = 0
    unknown: int = 0


@dataclass(frozen=True)
class _Place:
    country: str | None
    city_geonameid: int | None
    district_geonameid: int | None
    city: str | None
    district_name: str | None = None
    country_name: str | None = None  # v10 (online): full country name from Nominatim; offline None


_UNKNOWN_PLACE = _Place(country=None, city_geonameid=None, district_geonameid=None, city=None)


class _PlaceBatchResolver(Protocol):
    def resolve_places(
        self, coords: list[tuple[float, float]],
        progress: Callable[[int, int], None] | None = None,
    ) -> list[_Place]:
        ...  # pragma: no cover — protocol signature


def _coord(v: object) -> float | None:
    """Coordinate → float or None (guard against '' / garbage in the index)."""
    if v is None or v == "":
        return None
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


class _OfflineBatchResolver:
    """A wrapper over geodata.GeoResolver: resolve coordinates + the canonical (en) city anchor.

    A geonameid → name cache — on a batch of photos of the same city/district we do
    not call name() repeatedly.
    """

    def __init__(self, resolver: GeoResolver) -> None:
        self._resolver = resolver
        self._name_cache: dict[int, str] = {}

    def _city_name(self, geonameid: int | None) -> str | None:
        if geonameid is None:
            return None
        if geonameid not in self._name_cache:
            self._name_cache[geonameid] = self._resolver.name(geonameid, _CANONICAL_LANG)
        return self._name_cache[geonameid]

    def resolve_places(
        self, coords: list[tuple[float, float]],
        progress: Callable[[int, int], None] | None = None,
    ) -> list[_Place]:
        # offline resolve — bundled data, no network: the stage is fast anyway, and
        # total is already visible from the initial progress(0, len(rows)) before the
        # write loop below (see resolve_places) — no extra ticks needed here.
        del progress
        places = []
        for lat, lon in coords:
            res = self._resolver.resolve(lat, lon)
            places.append(_Place(
                country=res.country_cc,
                city_geonameid=res.city_id,
                district_geonameid=res.district_id,
                city=self._city_name(res.city_id),
            ))
        return places


class _NominatimResolver:
    """Online resolve via Nominatim/OSM reverse geocoding (variant B: names as text).

    No geonameids — city/district are returned as ready names in the language
    cfg.language. Respects the OSM policy: a mandatory User-Agent and no more than
    1 request/sec; repeated coordinates (rounded to cfg.geo.cache_coord_digits
    digits) within a run are taken from the in-memory cache without a new request.
    """

    def __init__(self, cfg: Config) -> None:
        self._url = cfg.geo.nominatim_url.rstrip("/") + "/reverse"
        self._user_agent = cfg.geo.nominatim_user_agent
        self._timeout = cfg.geo.nominatim_timeout
        self._language = cfg.language
        self._coord_digits = cfg.geo.cache_coord_digits  # cache-key rounding (speed)
        self._cache: dict[tuple[float, float], _Place] = {}
        self._last_request: float | None = None

    def _rate_limit(self) -> None:
        if self._last_request is not None:
            elapsed = time.monotonic() - self._last_request
            wait = _NOMINATIM_MIN_INTERVAL - elapsed
            if wait > 0:
                time.sleep(wait)
        self._last_request = time.monotonic()

    def _fetch(self, lat: float, lon: float) -> _Place:
        query = urllib.parse.urlencode({
            "lat": lat, "lon": lon, "format": "jsonv2", "zoom": 14,
            "accept-language": self._language,
        })
        req = urllib.request.Request(
            f"{self._url}?{query}", headers={"User-Agent": self._user_agent},
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
            _log.warning("geo: nominatim reverse не удался для (%s, %s): %s", lat, lon, exc)
            return _UNKNOWN_PLACE

        address = data.get("address") if isinstance(data, dict) else None
        if not address:
            _log.warning("geo: nominatim пустой address для (%s, %s)", lat, lon)
            return _UNKNOWN_PLACE

        country_code = address.get("country_code")
        city = address.get("city") or address.get("town") or address.get("village") \
            or address.get("municipality")
        district = address.get("suburb") or address.get("city_district") \
            or address.get("neighbourhood") or address.get("quarter")
        return _Place(
            country=country_code.upper() if country_code else None,
            city_geonameid=None,
            district_geonameid=None,
            city=city,
            district_name=district,
            country_name=address.get("country"),  # full name in the accept-language language
        )

    def resolve_places(
        self, coords: list[tuple[float, float]],
        progress: Callable[[int, int], None] | None = None,
    ) -> list[_Place]:
        # The network phase itself (~1 request/sec, most of the run can go here):
        # progress on EVERY coordinate, not rarely — otherwise the counter hangs at
        # "0 of N" for all those minutes and then instantly races to the end.
        places = []
        for i, (lat, lon) in enumerate(coords, 1):
            key = (round(lat, self._coord_digits), round(lon, self._coord_digits))
            if key not in self._cache:
                self._rate_limit()
                self._cache[key] = self._fetch(lat, lon)
            places.append(self._cache[key])
            if progress:
                progress(i, len(coords))
        return places


def _resolver_for(cfg: Config) -> _PlaceBatchResolver:
    """Provider abstraction by `cfg.geo.provider`.

    offline -> geodata.GeoResolver (bundled GeoNames, no network).
    online  -> Nominatim/OSM reverse geocoding (G2b) — names as text, no geonameids.
    """
    provider = cfg.geo.provider
    if provider == "offline":
        return _OfflineBatchResolver(GeoResolver())
    if provider == "online":
        return _NominatimResolver(cfg)
    raise ValueError(f"geo: неизвестный geo.provider={provider!r} (ожидается offline|online)")


def _parse_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        # drop tzinfo: taken_at is local capture time, a mix of aware/naive would
        # break sorting
        return datetime.fromisoformat(s).replace(tzinfo=None)
    except ValueError:
        return None


def _split_sessions(
    timed: list[tuple[datetime, sqlite3.Row]], gap_hours: float,
) -> list[list[tuple[datetime, sqlite3.Row]]]:
    """Files sorted by time; a gap > gap_hours starts a new session."""
    sessions: list[list[tuple[datetime, sqlite3.Row]]] = []
    gap_sec = gap_hours * 3600
    for item in sorted(timed, key=lambda t: t[0]):
        if sessions and (item[0] - sessions[-1][-1][0]).total_seconds() <= gap_sec:
            sessions[-1].append(item)
        else:
            sessions.append([item])
    return sessions


def resolve_places(
    cfg: Config, conn: sqlite3.Connection,
    progress: Callable[[int, int], None] | None = None,
) -> GeoStats:
    """Resolve the place of each canonical file and fully recompute places."""
    gap_hours = float(cfg.geo.session_gap_hours)

    rows = conn.execute(
        """SELECT id, taken_at, taken_at_confidence, gps_lat, gps_lon
           FROM files WHERE dup_of IS NULL AND error IS NULL"""
    ).fetchall()

    # 1) exact_gps: all files with valid coordinates.
    #    Coordinates may be garbage ('' from broken EXIF), so we coerce to float and
    #    skip the unparsable ones (otherwise geodata/scipy crashes).
    gps_rows: list[sqlite3.Row] = []
    coords: list[tuple[float, float]] = []
    for r in rows:
        lat, lon = _coord(r["gps_lat"]), _coord(r["gps_lon"])
        if lat is not None and lon is not None:
            gps_rows.append(r)
            coords.append((lat, lon))
    resolved: dict[int, tuple[_Place, str]] = {}
    if coords:
        resolver = _resolver_for(cfg)
        # online: the entire network phase sits right here (in the resolve) (~1
        # request/sec to Nominatim, minutes on a real collection) — progress must
        # move here, not in the write loop below (which is instant for online: the
        # network already ran, the rest is pure SQLite).
        places = resolver.resolve_places(coords, progress=progress)
        for r, place in zip(gps_rows, places):
            resolved[r["id"]] = (place, "exact_gps")

    # 2) session_inferred: inheritance of the FULL place (country + both geonameids
    #    + city) within a time session.
    timed = [(dt, r) for r in rows if (dt := _parse_dt(r["taken_at"])) is not None]
    for session in _split_sessions(timed, gap_hours):
        sources = [(dt, resolved[r["id"]][0]) for dt, r in session if r["id"] in resolved]
        if not sources:
            continue
        for dt, r in session:
            if r["id"] in resolved or r["taken_at_confidence"] not in _INHERIT_CONFIDENCE:
                continue
            # several cities in a session → take the nearest-in-time file with GPS
            _, place = min(sources, key=lambda s: abs((s[0] - dt).total_seconds()))
            resolved[r["id"]] = (place, "session_inferred")

    # 3) write: full recomputation of the places table in one transaction
    stats = GeoStats(total=len(rows))
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if progress:
        progress(0, len(rows))  # total right away, even if the stage is small/fast (#37)
    # The write loop is pure SQLite (the network, if any, already ran in the resolve
    # above), the throttle does not depend on the provider.
    with conn:
        conn.execute("DELETE FROM places")
        for i, r in enumerate(rows, 1):
            place, confidence = resolved.get(r["id"], (_UNKNOWN_PLACE, "unknown"))
            setattr(stats, confidence, getattr(stats, confidence) + 1)
            conn.execute(
                """INSERT INTO places
                       (file_id, country, country_name, city, city_geonameid,
                        district_geonameid, district_name, confidence, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (r["id"], place.country, place.country_name, place.city,
                 place.city_geonameid, place.district_geonameid, place.district_name,
                 confidence, now),
            )
            if i % _PROGRESS_EVERY == 0:
                if progress:
                    progress(i, len(rows))
                else:
                    print(f"geo: {i}/{len(rows)} файлов")
        if progress and rows:
            progress(len(rows), len(rows))
    return stats
