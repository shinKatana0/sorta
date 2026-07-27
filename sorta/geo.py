"""F2/G2 (Phase 2): the geo layer.

Contract: reads files (gps_lat/gps_lon, taken_at), writes ONLY into places.
Does not touch files, faces, events, moves.

Confidence levels:
- exact_gps        — coordinates from EXIF, offline resolve via geodata.GeoResolver
- session_inferred — place inherited from a file with GPS in the same time session
- trip_inferred    — place inherited from the TRIP the file belongs to (F85a), when the
                     GPS frames of that trip agree about the city
- unknown          — could not resolve (visual — landmarks.py, Phase 5/F6)

`exact_gps` requires a place that ACTUALLY resolved (F65): coordinates whose resolve
came back empty stay `unknown` and are counted in `GeoStats.gps_unresolved` — such a
file must not look confidently placed nor become a donor for session inheritance.

The provider is chosen by `cfg.geo.provider`: offline (default) — bundled GeoNames
via geodata.GeoResolver; online (G2b) — Nominatim/OSM reverse geocoding, names as
text already in cfg.language (no geonameids). Online answers that carry a country but
no city are completed from the bundled offline base (F86, see _CityFallbackResolver) —
the online provider stays the primary source for NAMES; the offline base is asked
for every coordinate anyway since F93 — it supplies the cache key — and only
completes a city the provider did not give.

Canonically we write geonameid (city_geonameid/district_geonameid) + the English/
asciiname anchor `city` (for --where/CSV/landmark fallback). Localizing names into
the target language is sort's job (G3), not this module's. `region` — DEPRECATED, no
longer written (stays NULL). `district_name` — online only (district name as text,
offline leaves it NULL and writes geonameid into district_geonameid).

F85a: inheritance has a second level. A time session is six hours wide, and 1 758 files
of the live collection sat in a session where nobody had GPS while the TRIP around them
was placed perfectly well. Widening the session window is the wrong knob — it stretches
an arbitrary interval of time, and twelve hours later the camera may be in another city.
A trip is a unit of MEANING (sessions merged by time AND geographic proximity), so the
place of a trip is inheritable inside it — but only when the trip's own GPS frames agree
about the city AND the file lies between two of them in time (see _inherit_trip_places).

F93: the ONLINE answers live in the `geo_cache` table, not in the process. A re-run
still recomputes places from scratch (session inheritance needs the whole collection),
but it no longer re-asks the network about coordinates it already knows — and it stores
all three languages side by side, so switching folder language costs no requests at
all. The offline path never touches that table: recomputing it takes two seconds.

Idempotency: a re-run fully recomputes places.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from math import asin, cos, radians, sin, sqrt
from pathlib import Path
from typing import Callable, Protocol

from .config import Config
from .geodata import GeoResolver
from .i18n import Lang

_log = logging.getLogger(__name__)

_PROGRESS_EVERY = 1000
_INHERIT_CONFIDENCE = ("high", "medium")
_CANONICAL_LANG: Lang = "en"  # the city anchor is always English/asciiname — not localized here
_NOMINATIM_MIN_INTERVAL = 1.0  # OSM policy: no more than 1 request/sec
# coordinate rounding for the grid fallback key — from cfg.geo.cache_coord_digits
# F93: a cached answer holds every interface language at once. Language is a property
# of the DATA, not of the run: the user switched folders to Japanese and the cities
# stayed Russian until the next full geo pass, i.e. 35 minutes of network. Completing
# the missing languages from the bundled base is not an option — measured on the live
# collection, GeoNames has ja names for 36 of its 83 cities.
_CACHE_LANGS: tuple[Lang, ...] = ("ru", "en", "ja")
_PROVIDER_ONLINE = "online"  # the only provider that writes into geo_cache
# F86: how often the "an answer came back without a city" warning is written. Per file
# it would be thousands of lines on a real collection; silence is what let the defect
# live through a full production run (zero warnings for 1 596 lost cities).
_CITY_MISSING_WARN_EVERY = 50
# F85a: defaults for the trip thresholds, read from the `events:` section (see the
# block comment above _TripLocality). Same numbers as events._DEFAULT_* — the config
# normally supplies them, these are only the fallback for an EventsConfig without them.
_DEFAULT_TRIP_MERGE_GAP_HOURS = 48.0
_DEFAULT_TRIP_MERGE_MAX_KM = 120.0
_EARTH_RADIUS_KM = 6371.0088


@dataclass
class GeoStats:
    total: int = 0
    exact_gps: int = 0
    session_inferred: int = 0
    trip_inferred: int = 0  # F85a: inherited from the trip, not from the time session
    unknown: int = 0
    # F65: files with valid coordinates whose place did not resolve. Non-zero means
    # the geo data is broken/missing, not that the user's photos are unusual.
    gps_unresolved: int = 0


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


def _is_null_island(lat: float, lon: float) -> bool:
    """Exactly (0, 0) — a camera writing zeros because it never got a fix.

    The pair matters, not the individual values: latitude 0 is the equator and
    longitude 0 is Greenwich, both perfectly real on their own. Only both at once is
    the sentinel. Left unfiltered it is worse than a missing place, because the
    nearest land to 0°N 0°E is Ghana and a nearest-neighbour resolver answers
    confidently — 35 files landed in a country the user has never visited, and 16
    more inherited it through the time-session rule.
    """
    return lat == 0.0 and lon == 0.0


class _OfflineBatchResolver:
    """A wrapper over geodata.GeoResolver: resolve coordinates + the canonical (en) city anchor.

    A geonameid → name cache — on a batch of photos of the same city/district we do
    not call name() repeatedly.
    """

    def __init__(self, resolver: GeoResolver) -> None:
        self._resolver = resolver
        self._name_cache: dict[int, str] = {}

    @property
    def data_dir(self) -> Path | None:
        """Where the bundled data is read from — for the "nothing resolved" warning."""
        return getattr(self._resolver, "data_dir", None)

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


class _NominatimClient:
    """One Nominatim/OSM reverse-geocoding request in ONE language (variant B: names as text).

    No geonameids — city/district come back as ready names in the language that was
    asked for. Respects the OSM policy: a mandatory User-Agent and no more than
    1 request/sec. Deduplicating coordinates, the three languages and the persistent
    cache all live one level up (_CachedOnlineResolver) — this class only knows how to
    ask politely.
    """

    def __init__(self, cfg: Config) -> None:
        self._url = cfg.geo.nominatim_url.rstrip("/") + "/reverse"
        self._user_agent = cfg.geo.nominatim_user_agent
        self._timeout = cfg.geo.nominatim_timeout
        self._last_request: float | None = None
        self.requests = 0  # for the run summary: how much network the stage actually cost

    def _rate_limit(self) -> None:
        if self._last_request is not None:
            elapsed = time.monotonic() - self._last_request
            wait = _NOMINATIM_MIN_INTERVAL - elapsed
            if wait > 0:
                time.sleep(wait)
        self._last_request = time.monotonic()

    def fetch(self, lat: float, lon: float, lang: str) -> _Place:
        self._rate_limit()
        self.requests += 1
        query = urllib.parse.urlencode({
            "lat": lat, "lon": lon, "format": "jsonv2", "zoom": 14,
            "accept-language": lang,
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
        # F86: outside cities Nominatim names the settlement with whatever key fits it —
        # hamlet/locality/isolated_dwelling for the countryside. `county`/`state` are
        # deliberately NOT read: an administrative region is not a city, and putting a
        # file into a folder named after an oblast would be worse than the country level.
        city = address.get("city") or address.get("town") or address.get("village") \
            or address.get("municipality") or address.get("hamlet") \
            or address.get("locality") or address.get("isolated_dwelling")
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


def _pick_lang(values: dict[str, str | None], lang: str) -> str | None:
    """The variant for `lang`, falling back to en and then to any language present.

    An honest fallback, not a substitution: OSM has no `name:ja` for a Balinese
    village, so its ja answer is the local latin name — which is exactly the string the
    sorter used to write anyway. Returning nothing instead would hide a resolved place.
    """
    for candidate in (lang, _CANONICAL_LANG, *_CACHE_LANGS):
        value = values.get(candidate)
        if value:
            return value
    return None


@dataclass(frozen=True)
class _CachedAnswer:
    """One geo_cache row: what the provider said about a key, in all three languages."""

    country: str | None
    country_name: dict[str, str | None]
    city: dict[str, str | None]
    district: dict[str, str | None]

    @classmethod
    def of(cls, places: dict[str, _Place]) -> "_CachedAnswer":
        """Fold the per-language answers about ONE key into a single row."""
        return cls(
            country=next((p.country for p in places.values() if p.country), None),
            country_name={lang: p.country_name for lang, p in places.items()},
            city={lang: p.city for lang, p in places.items()},
            district={lang: p.district_name for lang, p in places.items()},
        )

    def place(self, lang: str) -> _Place:
        """The place as the current run needs it — the variant for `lang`."""
        return _Place(
            country=self.country,
            city_geonameid=None,
            district_geonameid=None,
            city=_pick_lang(self.city, lang),
            district_name=_pick_lang(self.district, lang),
            country_name=_pick_lang(self.country_name, lang),
        )


def _all_answered(places: dict[str, _Place]) -> bool:
    """Did EVERY language come back with a place? Only then may the row be cached.

    A one-off network failure (or the "nominatim пустой address" of a bad minute — two
    of them in one live run) must not be frozen into the collection forever, and a
    half-written row would pin the missing language until the expiry date. Either all
    three languages, or nothing: the files still get the answer of this run, and the
    next run tries again.
    """
    return bool(places) and all(p.country is not None for p in places.values())


class _GeoCacheTable:
    """Access to `geo_cache` (schema v13): provider answers that outlive the run.

    The key has two shapes, built by the code (SQLite would count NULLs in a composite
    primary key as distinct rows):
      `c:<city_geonameid>/<district_geonameid>` — the normal one;
      `g:<lat>/<lon>` — the fallback for coordinates the local base cannot place.
    The provider is part of the key: offline and online give different answers and
    mixing them would be a silent misplacement. The language is NOT — it became a
    dimension of the value.
    """

    def __init__(self, conn: sqlite3.Connection, provider: str, max_age_days: int) -> None:
        self._conn = conn
        self._provider = provider
        self._max_age_days = max_age_days
        self.hits = 0
        self.misses = 0
        self.expired = 0

    def _fresh(self, updated_at: str | None) -> bool:
        """Is the row still within cfg.geo.cache_max_age_days? (0 — the expiry is off.)

        City and district borders move rarely, but not never; an unreadable timestamp
        counts as expired — asking again is cheap next to trusting a row we cannot date.
        """
        if self._max_age_days <= 0:
            return True
        written = _parse_dt(updated_at)
        if written is None:
            return False
        age = datetime.now(timezone.utc).replace(tzinfo=None) - written
        return age <= timedelta(days=self._max_age_days)

    def get(self, key: str) -> _CachedAnswer | None:
        row = self._conn.execute(
            """SELECT country, country_name_ru, country_name_en, country_name_ja,
                      city_ru, city_en, city_ja,
                      district_ru, district_en, district_ja, updated_at
               FROM geo_cache WHERE provider = ? AND key = ?""",
            (self._provider, key),
        ).fetchone()
        if row is None:
            self.misses += 1
            return None
        if not self._fresh(row["updated_at"]):
            self.expired += 1
            return None
        self.hits += 1
        return _CachedAnswer(
            country=row["country"],
            country_name={lang: row[f"country_name_{lang}"] for lang in _CACHE_LANGS},
            city={lang: row[f"city_{lang}"] for lang in _CACHE_LANGS},
            district={lang: row[f"district_{lang}"] for lang in _CACHE_LANGS},
        )

    def put(self, key: str, answer: _CachedAnswer) -> None:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._conn:
            self._conn.execute(
                """INSERT OR REPLACE INTO geo_cache
                       (provider, key, country,
                        country_name_ru, country_name_en, country_name_ja,
                        city_ru, city_en, city_ja,
                        district_ru, district_en, district_ja, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (self._provider, key, answer.country,
                 *(answer.country_name.get(lang) for lang in _CACHE_LANGS),
                 *(answer.city.get(lang) for lang in _CACHE_LANGS),
                 *(answer.district.get(lang) for lang in _CACHE_LANGS), now),
            )


def geo_cache_size(conn: sqlite3.Connection) -> int:
    """How many provider answers are cached (for `sorta cache`)."""
    row = conn.execute("SELECT COUNT(*) AS n FROM geo_cache").fetchone()
    return int(row["n"]) if row is not None else 0


def clear_geo_cache(conn: sqlite3.Connection) -> int:
    """Drop every cached provider answer; returns how many rows went away.

    The escape hatch of F93: a cache can freeze a WRONG answer of the provider, and
    "Start over" deliberately no longer wipes it — so there has to be one command that
    does (`sorta cache --clear-geo`, `sorta reset --clear-geo`, the checkbox in the
    reset dialog of the web app).
    """
    removed = geo_cache_size(conn)
    with conn:
        conn.execute("DELETE FROM geo_cache")
    return removed


def _median_coord(points: list[tuple[float, float]]) -> tuple[float, float]:
    """The representative of a group — the median latitude and longitude.

    The median and not the mean: one frame shot from a plane over the same district
    would drag an average out of the place entirely. For a district of an awkward shape
    even the median can land outside its border — the same caveat F92 carries about its
    own trip centres.
    """
    return (statistics.median(p[0] for p in points),
            statistics.median(p[1] for p in points))


class _CachedOnlineResolver:
    """F93: the online provider behind a cache keyed by the place of the LOCAL base.

    Two things used to die with the process. The answers — `geo` recomputes places from
    scratch every run (session inheritance looks at neighbours in time, so a partial
    recompute gives a different result), and the in-memory cache went with the resolver:
    adding 200 photos cost the same ~35 minutes of Nominatim as a full run. And the
    language — it was asked for once, at `accept-language`, so switching folder language
    left the cities in the old one until the next full pass.

    The key is the pair (city_geonameid, district_geonameid) that every coordinate gets
    for free from the bundled KD-tree. It beats a coordinate grid on both axes at once —
    measured on 14 254 GPS files: 603 requests against 6 219 for a 110 m grid, and zero
    localities mixed against 0.9% of districts. The reason is that the local base has
    already partitioned the map by MEANING, while a grid invents squares that do not
    know where a district ends. Coordinates the local base cannot place fall back to a
    grid key (`g:`) — there are few of them, and the cost of a mistake there is higher.

    A side effect worth as much as the speed: an online place is now anchored to a city
    of the local base, so event names find a dominant locality again. Nominatim answers
    with a village or a suburb, dozens per trip, and the name used to fall back to the
    COUNTRY («Тайланд» instead of «Пхангнга» for 1 359 files).

    On a miss the provider is asked THREE times (ru/en/ja) about ONE representative
    coordinate — the median of the group — and the row is written once.
    """

    def __init__(self, cfg: Config, conn: sqlite3.Connection,
                 client: _NominatimClient, local: GeoResolver | None) -> None:
        self._client = client
        self._local = local
        self._language = cfg.language
        self._grid_digits = cfg.geo.cache_coord_digits
        self._cache = _GeoCacheTable(conn, _PROVIDER_ONLINE, cfg.geo.cache_max_age_days)

    def _key(self, lat: float, lon: float) -> str:
        """The cache key of a coordinate: the local base's place, or a grid cell."""
        if self._local is not None:
            res = self._local.resolve(lat, lon)
            if res.city_id is not None:
                district = "-" if res.district_id is None else str(res.district_id)
                return f"c:{res.city_id}/{district}"
        return f"g:{round(lat, self._grid_digits)}/{round(lon, self._grid_digits)}"

    def _ask_provider(self, point: tuple[float, float]) -> tuple[_CachedAnswer, bool]:
        """Three requests (ru/en/ja) about one point -> the row + may it be cached."""
        lat, lon = point
        places: dict[str, _Place] = {
            lang: self._client.fetch(lat, lon, lang) for lang in _CACHE_LANGS}
        return _CachedAnswer.of(places), _all_answered(places)

    def resolve_places(
        self, coords: list[tuple[float, float]],
        progress: Callable[[int, int], None] | None = None,
    ) -> list[_Place]:
        groups: dict[str, list[int]] = {}
        keys: list[str] = []
        for i, (lat, lon) in enumerate(coords):
            key = self._key(lat, lon)
            keys.append(key)
            groups.setdefault(key, []).append(i)

        # The network phase itself (~1 request/sec, most of the run can go here):
        # progress after every GROUP, not once at the end — otherwise the counter hangs
        # at "0 of N" for all those minutes and then instantly races to the end.
        by_key: dict[str, _Place] = {}
        done = 0
        for key, indices in groups.items():
            answer = self._cache.get(key)
            if answer is None:
                answer, cacheable = self._ask_provider(
                    _median_coord([coords[i] for i in indices]))
                if cacheable:
                    self._cache.put(key, answer)
            by_key[key] = answer.place(self._language)
            done += len(indices)
            if progress:
                progress(done, len(coords))
        _log.info(
            "geo: онлайн-кэш: групп %d (из %d координат), попаданий %d, просрочено %d, "
            "запросов к провайдеру %d",
            len(groups), len(coords), self._cache.hits, self._cache.expired,
            self._client.requests,
        )
        return [by_key[key] for key in keys]


class _CityFallbackResolver:
    """F86: the online provider first, the bundled offline base as insurance for the city.

    Nominatim answers for suburbs and the countryside without any of the city keys it
    is read by, and the place used to end up as "country known, city NULL". On the live
    collection that is 1 596 files (1 471 exact_gps + 125 inherited) that the sorter then
    hid in _Unsorted/no_place — while the SAME coordinates resolve to a city in the
    bundled GeoNames data (55.4138, 37.8976 -> Домодедово; -8.79806, 115.2349 ->
    Jabajero). Turning the online provider on made the result worse than offline; here
    the online answer stays primary and only a missing city is completed offline. Since
    F93 the offline lookup happens for every coordinate regardless (it is what the cache
    key is built from), so this costs nothing extra — but it is no longer true that the
    base is "only touched on a miss".

    Country, city and district always come from ONE source: an offline city replaces the
    whole place, so a Nominatim country name never gets glued onto a GeoNames city. If
    the two providers disagree about the country, the offline answer is dropped — a
    nearest-neighbour city in the wrong country is exactly the silent misplacement F75
    guards against; the file keeps the online country and is laid out at country level.

    A place with no country at all (a failed request, an empty address) is NOT asked
    offline: there the provider gave no answer to complete, and F65 keeps its meaning —
    such coordinates stay `unknown` instead of quietly switching provider mid-run.
    """

    def __init__(self, primary: _PlaceBatchResolver, offline: _OfflineBatchResolver) -> None:
        self._primary = primary
        self._offline = offline
        self._city_missing = 0
        self._city_recovered = 0

    def _offline_place(self, coord: tuple[float, float], online: _Place) -> _Place:
        """The offline place for the coordinate, or `online` unchanged if it does not help."""
        offline = self._offline.resolve_places([coord])[0]
        if offline.city is None or offline.country != online.country:
            return online
        return offline

    def resolve_places(
        self, coords: list[tuple[float, float]],
        progress: Callable[[int, int], None] | None = None,
    ) -> list[_Place]:
        places = list(self._primary.resolve_places(coords, progress=progress))
        for i, place in enumerate(places):
            if place.city is not None or place.country is None:
                continue
            self._city_missing += 1
            completed = self._offline_place(coords[i], place)
            if completed is not place:
                self._city_recovered += 1
                places[i] = completed
            if (self._city_missing == 1
                    or self._city_missing % _CITY_MISSING_WARN_EVERY == 0):
                _log.warning(
                    "geo: провайдер вернул ответ без города для (%s, %s), страна %s — "
                    "случай %d, из них с городом из оффлайн-базы: %d",
                    coords[i][0], coords[i][1], place.country,
                    self._city_missing, self._city_recovered,
                )
        if self._city_missing:
            _log.warning(
                "geo: ответов без города: %d, из них город найден в оффлайн-базе: %d "
                "(остальные останутся на уровне страны)",
                self._city_missing, self._city_recovered,
            )
        return places


def _resolver_for(cfg: Config, conn: sqlite3.Connection) -> _PlaceBatchResolver:
    """Provider abstraction by `cfg.geo.provider`.

    offline -> geodata.GeoResolver (bundled GeoNames, no network) — never reads or
               writes geo_cache: a full offline recompute takes two seconds.
    online  -> Nominatim/OSM reverse geocoding (G2b) — names as text, no geonameids,
               behind the persistent cache of F93 and wrapped in the offline city
               fallback of F86 (both only if the bundled data is actually there: online
               must keep working on an install without it).
    """
    provider = cfg.geo.provider
    if provider == "offline":
        return _OfflineBatchResolver(GeoResolver())
    if provider == "online":
        offline = GeoResolver()
        available = offline.data_available()
        if not available:
            _log.warning(
                "geo: оффлайн-база недоступна (%s) — города, которых нет в ответе "
                "провайдера, восстановить не получится, а кэш ответов будет "
                "группировать координаты по сетке, а не по городу и району",
                offline.data_dir,
            )
        online = _CachedOnlineResolver(cfg, conn, _NominatimClient(cfg),
                                      offline if available else None)
        if not available:
            return online
        return _CityFallbackResolver(online, _OfflineBatchResolver(offline))
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


# --- F85a: the trip level of inheritance --------------------------------------------
# DUPLICATED RULE. The twin lives in events.py (_split_sessions/_merge_sessions/
# _same_trip) — change a threshold on one side and look at the other.
#
# It has to be duplicated. The stages run index -> geo -> landmarks -> faces -> events,
# so when geo works the `events` table does not exist yet on a clean run and there is
# nothing to read; moving the inheritance into the events stage is not allowed either,
# because `places` has exactly one writer (docs/ARCHITECTURE.md §2), and the stage order
# cannot change (events needs `country` from places). So geo groups its own sessions
# into trips by the same rule, reading the SAME thresholds — cfg.events.trip_merge_gap_hours
# and cfg.events.trip_merge_max_km.
#
# Two deliberate differences, both in the direction of "inherit less":
# * no admin1-region branch. events has a loaded GeoResolver at hand; geo would have to
#   load the bundled base a second time just for this. A trip here can therefore only
#   come out SHORTER than the events one, never wider — no file is placed further away
#   than the rule above allows.
# * a session where nobody has GPS does not break a trip: it neither confirms nor denies
#   the locality, and those sessions are exactly what this feature exists for. In events
#   such a session starts a new group, because there a group IS the event.


@dataclass(frozen=True)
class _TripLocality:
    """Where a session was, as far as its own GPS files know."""

    key: tuple[str, object] | None  # ("i", geonameid) | ("s", casefolded city name)
    country: str | None
    coords: tuple[float, float] | None  # F92: the median GPS of the files of `key`


def _city_key(place: _Place) -> tuple[str, object] | None:
    """The locality key of a resolved place: geonameid, else the city name, else none.

    events._city_key prefers `district_name` for the string fallback — an event wants
    the most specific name it can print. Here the only question is which CITY the trip
    agreed on, so a suburb must not split one city into three localities.
    """
    if place.city_geonameid is not None:
        return ("i", place.city_geonameid)
    if place.city and place.city.strip():
        return ("s", place.city.strip().casefold())
    return None


def _haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = radians(a[0]), radians(a[1]), radians(b[0]), radians(b[1])
    h = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * asin(sqrt(h))


# a resolved GPS file: when it was taken, the place it resolved to, its own coordinates
_Donor = tuple[datetime, _Place, tuple[float, float] | None]


def _session_donors(
    session: list[tuple[datetime, sqlite3.Row]],
    resolved: dict[int, tuple[_Place, str]], gps: dict[int, tuple[float, float]],
) -> list[_Donor]:
    """The exact_gps files of a session — the only ones that may vouch for a place.

    Session-inherited files are deliberately left out: their place is a copy of these
    same coordinates, and counting it again would let one GPS frame outvote another
    just by having more dateless neighbours.
    """
    donors: list[_Donor] = []
    for dt, r in session:
        entry = resolved.get(r["id"])
        if entry is not None and entry[1] == "exact_gps":
            donors.append((dt, entry[0], gps.get(r["id"])))
    return donors


def _session_locality(donors: list[_Donor]) -> _TripLocality | None:
    """The dominant locality of a session; None — nobody in it has a place at all."""
    if not donors:
        return None
    keys = Counter(k for _dt, p, _c in donors if (k := _city_key(p)) is not None)
    countries = Counter(p.country for _dt, p, _c in donors if p.country)
    country = countries.most_common(1)[0][0] if countries else None
    if not keys:
        return _TripLocality(key=None, country=country, coords=None)
    best_key = keys.most_common(1)[0][0]
    own = [c for _dt, p, c in donors if _city_key(p) == best_key and c is not None]
    return _TripLocality(key=best_key, country=country,
                         coords=_median_coord(own) if own else None)


def _same_trip(anchor: _TripLocality, cand: _TripLocality, max_km: float) -> bool:
    """The same country AND (the same city OR centers closer than max_km).

    The centers are the medians of the files' OWN GPS (F92), so the distance branch
    works under any geo provider, geonameid or not.
    """
    if not anchor.country or not cand.country or anchor.country != cand.country:
        return False
    if anchor.key is not None and anchor.key == cand.key:
        return True
    if anchor.key is None or cand.key is None:  # an unknown locality confirms nothing
        return False
    if max_km > 0 and anchor.coords is not None and cand.coords is not None:
        return _haversine_km(anchor.coords, cand.coords) <= max_km
    return False


def _merge_trips(
    sessions: list[list[tuple[datetime, sqlite3.Row]]],
    localities: list[_TripLocality | None],
    trip_gap_hours: float, max_km: float,
) -> list[list[int]]:
    """Adjacent sessions -> trips (as lists of session indices).

    The anchor is the locality of the first session of the trip that HAS one, and it is
    not recomputed on further merges (as in events): the comparison always runs against
    the place of the trip, not against the last session added to it.
    """
    trips: list[list[int]] = []
    anchors: list[_TripLocality | None] = []
    trip_gap = timedelta(hours=trip_gap_hours)
    for i, session in enumerate(sessions):
        locality = localities[i]
        if trips:
            anchor = anchors[-1]
            prev = sessions[trips[-1][-1]]
            if (session[0][0] - prev[-1][0] < trip_gap
                    and (locality is None or anchor is None
                         or _same_trip(anchor, locality, max_km))):
                trips[-1].append(i)
                if anchor is None:
                    anchors[-1] = locality
                continue
        trips.append([i])
        anchors.append(locality)
    return trips


def _brackets(first: datetime, last: datetime, dt: datetime) -> bool:
    """Is `dt` between the first and the last frame that vouches for the trip's city?

    The evidence a trip offers is not uniform over its length. In the MIDDLE it is an
    alibi: the camera was in this city before the file and in the same city after it, and
    nothing in between could have taken it a thousand kilometres away and back. Past the
    ends there is no such alibi — the last GPS frame does not say when the trip's owner
    left, and a day-trip out of town lands exactly there.

    The measurement agrees, and not by a small margin: on the validation collection the
    span holds 414 of the 554 inferences and 4 of the 32 mistakes — the other 28 all sat
    past an end. Precision inside 99.0%, outside 80.0%.
    """
    return first <= dt <= last


def _trip_place(places: list[_Place]) -> _Place:
    """The place a trip lends: the dominant one of its city, minus the district.

    What the trip agreed on is the CITY; a district would be a finer claim than the
    evidence supports — the frame may well have been shot in another part of town.
    """
    rep = Counter(places).most_common(1)[0][0]
    return replace(rep, district_geonameid=None, district_name=None)


def _inherit_trip_places(
    cfg: Config, sessions: list[list[tuple[datetime, sqlite3.Row]]],
    resolved: dict[int, tuple[_Place, str]], gps: dict[int, tuple[float, float]],
) -> None:
    """F85a: a file with no place inherits the place of its TRIP. Mutates `resolved`.

    Runs AFTER session inheritance — that one is more precise and keeps its priority;
    only what it did not reach is considered here.

    Two conditions, both measured on scripts/measure_place_inference.py (hide the GPS of
    files that have it, infer, compare with the truth — 554 trip-level cases):

    1. the trip's own GPS frames agree about the city: the dominant city holds MORE than
       half of them (a frame whose place came back as country-only counts in the
       denominator — it is a GPS frame that does not confirm the city). A trip across
       three cities leaves its place-less files as they are: a foreign city is worse than
       an empty folder, because the user will not look there (the principle of F75/F86);
    2. the file lies BETWEEN two frames of that city in time — see `_brackets`.

    Rule 1 alone measured 94.2% precision, below the 95% this feature has to clear; every
    single mistake was a file outside the span of the trip's GPS frames. Adding rule 2
    brings it to 99.0% and costs about a quarter of the reach.
    """
    trip_gap_hours = float(getattr(cfg.events, "trip_merge_gap_hours",
                                   _DEFAULT_TRIP_MERGE_GAP_HOURS))
    max_km = float(getattr(cfg.events, "trip_merge_max_km", _DEFAULT_TRIP_MERGE_MAX_KM))
    donors_of = [_session_donors(s, resolved, gps) for s in sessions]
    localities = [_session_locality(d) for d in donors_of]
    for trip in _merge_trips(sessions, localities, trip_gap_hours, max_km):
        donors = [d for i in trip for d in donors_of[i]]
        keys = Counter(k for _dt, p, _c in donors if (k := _city_key(p)) is not None)
        if not keys:
            continue
        best_key, best_n = keys.most_common(1)[0]
        if best_n * 2 <= len(donors):
            continue
        vouching = [d for d in donors if _city_key(d[1]) == best_key]
        first, last = min(d[0] for d in vouching), max(d[0] for d in vouching)
        place = _trip_place([p for _dt, p, _c in vouching])
        for i in trip:
            for dt, r in sessions[i]:
                if r["id"] in resolved or r["taken_at_confidence"] not in _INHERIT_CONFIDENCE:
                    continue
                if not _brackets(first, last, dt):
                    continue
                resolved[r["id"]] = (place, "trip_inferred")


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
        if lat is not None and lon is not None and not _is_null_island(lat, lon):
            gps_rows.append(r)
            coords.append((lat, lon))
    resolved: dict[int, tuple[_Place, str]] = {}
    # F85a: the coordinates of the GPS files by id — a trip locality's center is the
    # median of the files' OWN GPS (F92), so the distance check needs them again below.
    gps_by_id = {r["id"]: c for r, c in zip(gps_rows, coords)}
    gps_unresolved = 0
    if coords:
        resolver = _resolver_for(cfg, conn)
        # online: the entire network phase sits right here (in the resolve) (~1
        # request/sec to Nominatim, minutes on a real collection) — progress must
        # move here, not in the write loop below (which is instant for online: the
        # network already ran, the rest is pure SQLite).
        places = resolver.resolve_places(coords, progress=progress)
        for r, place in zip(gps_rows, places):
            # F65: honest confidence — coordinates alone do not make an exact_gps.
            # An empty place (missing geo data offline, a failed request online) stays
            # "unknown" instead of filling the DB with confident-looking NULLs, and
            # never becomes a donor for session inheritance below.
            if place.country is None:
                gps_unresolved += 1
                continue
            resolved[r["id"]] = (place, "exact_gps")
        if gps_unresolved:
            # the offline resolver knows where it read from; online (Nominatim) has
            # already logged each failed request itself — here it is the total
            data_dir = getattr(resolver, "data_dir", None)
            _log.warning(
                "geo: %d из %d файлов с координатами не разрезолвились в место%s — "
                "проверьте гео-данные (places.tsv)",
                gps_unresolved, len(coords),
                f" (гео-данные: {data_dir})" if data_dir else "",
            )

    # 2) session_inferred: inheritance of the FULL place (country + both geonameids
    #    + city) within a time session.
    timed = [(dt, r) for r in rows if (dt := _parse_dt(r["taken_at"])) is not None]
    sessions = _split_sessions(timed, gap_hours)
    for session in sessions:
        sources = [(dt, resolved[r["id"]][0]) for dt, r in session if r["id"] in resolved]
        if not sources:
            continue
        for dt, r in session:
            if r["id"] in resolved or r["taken_at_confidence"] not in _INHERIT_CONFIDENCE:
                continue
            # several cities in a session → take the nearest-in-time file with GPS
            _, place = min(sources, key=lambda s: abs((s[0] - dt).total_seconds()))
            resolved[r["id"]] = (place, "session_inferred")

    # 2b) trip_inferred (F85a): what the six-hour session could not reach — a session
    #     where nobody has GPS inherits from the trip around it. Deliberately second:
    #     session inheritance is the more precise of the two and keeps priority.
    _inherit_trip_places(cfg, sessions, resolved, gps_by_id)

    # 3) write: full recomputation of the places table in one transaction
    stats = GeoStats(total=len(rows), gps_unresolved=gps_unresolved)
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
