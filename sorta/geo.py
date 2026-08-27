"""F2/G2 (Phase 2): the geo layer.

Contract: reads files (gps_lat/gps_lon, taken_at), writes ONLY into places. A re-run
recomputes the table in full.

Confidence levels, in the order they are tried:
- exact_gps        — coordinates from EXIF, resolved through the chosen provider
- session_inferred — inherited from a file with GPS in the same time session
- trip_inferred    — inherited from the TRIP (F85a), when its GPS frames agree on a city
- path_inferred    — the COUNTRY read off a folder name (F85c); the only signal here
                     that is not geometry, and deliberately the last one
- unknown          — could not resolve (visual — landmarks.py, Phase 5/F6)

`exact_gps` requires a place that ACTUALLY resolved (F65): coordinates whose resolve came
back empty stay `unknown` and are counted in `GeoStats.gps_unresolved`, so such a file
neither looks confidently placed nor becomes a donor for inheritance.

`cfg.geo.provider` picks between the bundled GeoNames base (offline, default) and
Nominatim/OSM (online, G2b — names as text, no geonameids). Since F93 the offline base is
asked for every coordinate either way, because it supplies the cache key.

We write geonameid + the city NAME in `cfg.language`; the folder segment is localized by
sort (G3) out of the geonameid, so a change of `language` costs no geo run for a place the
bundled base knows. `region` is DEPRECATED (stays NULL); `district_name` is online-only.

F172: what a place is CALLED is decided in `_place_name` alone. Before it the two sources
disagreed about the language of `places.city` — a live ru collection held «Сочи» and
«Sochi» for one geonameid, with the 179 Samara / 382 Nizhny Novgorod files filed in Latin
under a Russian country folder.

F85a: 1 758 files of the live collection sat in a six-hour session where nobody had GPS
while the TRIP around them was placed perfectly well. Widening the session window is the
wrong knob — twelve hours later the camera may be in another city — so a trip, a unit of
MEANING, lends its place instead (_inherit_trip_places).
"""
from __future__ import annotations

import json
import logging
import re
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

from . import i18n
from .config import Config
from .geodata import GeoDataMissing, GeoResolver
from .i18n import Lang

_log = logging.getLogger(__name__)

_PROGRESS_EVERY = 1000
_INHERIT_CONFIDENCE = ("high", "medium")
_NAME_FALLBACK_LANG: Lang = "en"  # F172: the second step of the naming chain, after cfg.language
_NOMINATIM_MIN_INTERVAL = 1.0  # OSM policy: no more than 1 request/sec
# F93: a cached answer holds every interface language at once, because language is a
# property of the DATA, not of the run — switching folders to Japanese used to leave the
# cities Russian until the next full geo pass, 35 minutes of network. Completing the
# missing languages from the bundled base is not an option: GeoNames has ja names for 36
# of the live collection's 83 cities.
_CACHE_LANGS: tuple[Lang, ...] = ("ru", "en", "ja")
_PROVIDER_ONLINE = "online"  # the only provider that writes into geo_cache
# F86: per file this warning would be thousands of lines on a real collection — and
# silence is what let the defect live through a full run (1 596 cities lost, 0 warnings).
_CITY_MISSING_WARN_EVERY = 50
# F85a: the fallback for an EventsConfig without the trip thresholds; the config normally
# supplies them. Same numbers as events._DEFAULT_*.
_DEFAULT_TRIP_MERGE_GAP_HOURS = 48.0
_DEFAULT_TRIP_MERGE_MAX_KM = 120.0
_EARTH_RADIUS_KM = 6371.0088
# F85c: one WORD of a folder name («Тайланд 2023» -> «Тайланд»). Digits and punctuation
# are separators, so a year glued to the name does not hide the country behind it.
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
# A word inside a longer name is only tried as a country from this length up — one notch
# stricter than the rule that was scored, because the short country names («Чад», «Мали»,
# «Того») double as ordinary words. The WHOLE segment is still matched at any length, so
# a folder actually named «Чад» resolves.
_PATH_HINT_MIN_WORD = 4


@dataclass
class GeoStats:
    total: int = 0
    exact_gps: int = 0
    session_inferred: int = 0
    trip_inferred: int = 0  # F85a: inherited from the trip, not from the time session
    path_inferred: int = 0  # F85c: the country came from a folder name, nothing else
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

    Only the PAIR is the sentinel: latitude 0 and longitude 0 are each perfectly real on
    their own. Unfiltered it is worse than a missing place — the nearest land to 0°N 0°E
    is Ghana, and a nearest-neighbour resolver answers confidently: 35 files landed in a
    country the user has never visited and 16 more inherited it through the session rule.
    """
    return lat == 0.0 and lon == 0.0


def _place_name(lang: Lang, *, geonameid: int | None = None,
                resolver: GeoResolver | None = None,
                provider_name: str | None = None) -> str | None:
    """F172: the ONE rule for what a place is called: `lang` -> en -> the native name.

    A geonameid outranks the text, so two files of one city cannot land in two folders —
    except that `GeoResolver.name` ends its own chain with the geonameid itself, and a
    folder called `498817` explains nothing; that answer is refused here in favour of the
    text (sorter._city_display_name refuses it too).

    The online provider passes through WITHOUT a geonameid on purpose: the id the local
    base gives for the same coordinates is not its id — the base answers with the nearest
    city of cities1000, Nominatim names the hamlet the photo was taken in (F86/F93).
    Substituting one for the other would MOVE the file; this only renames.
    """
    if geonameid is not None and resolver is not None:
        name = resolver.name(geonameid, lang)
        if name and name != str(geonameid):
            return name
    return provider_name


class _OfflineBatchResolver:
    """A wrapper over geodata.GeoResolver: resolve coordinates + name the city (F172).

    Caches geonameid -> name: a batch of photos of one city must not call name() again
    for every frame.
    """

    def __init__(self, resolver: GeoResolver, lang: Lang) -> None:
        self._resolver = resolver
        self._lang = lang
        self._name_cache: dict[int, str | None] = {}

    @property
    def data_dir(self) -> Path | None:
        """Where the bundled data is read from — for the "nothing resolved" warning."""
        return getattr(self._resolver, "data_dir", None)

    def _city_name(self, geonameid: int | None) -> str | None:
        if geonameid is None:
            return None
        if geonameid not in self._name_cache:
            self._name_cache[geonameid] = _place_name(
                self._lang, geonameid=geonameid, resolver=self._resolver)
        return self._name_cache[geonameid]

    def resolve_places(
        self, coords: list[tuple[float, float]],
        progress: Callable[[int, int], None] | None = None,
    ) -> list[_Place]:
        # No network here, so no ticks are needed: the total is already on screen from the
        # progress(0, len(rows)) that resolve_places emits before its write loop.
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
    """One Nominatim/OSM reverse-geocoding request in ONE language (names as text).

    No geonameids: city/district come back as ready names in the language asked for.
    Respects the OSM policy — a mandatory User-Agent and at most 1 request/sec.
    Deduplication, the three languages and the persistent cache live one level up
    (_CachedOnlineResolver).
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
            _log.warning("geo: nominatim reverse failed for (%s, %s): %s", lat, lon, exc)
            return _UNKNOWN_PLACE

        address = data.get("address") if isinstance(data, dict) else None
        if not address:
            _log.warning("geo: nominatim returned an empty address for (%s, %s)", lat, lon)
            return _UNKNOWN_PLACE

        country_code = address.get("country_code")
        # F86: outside cities Nominatim names the settlement with whatever key fits —
        # hamlet/locality/isolated_dwelling. `county`/`state` are deliberately NOT read:
        # a folder named after an oblast would be worse than the country level.
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

    The provider's half of the `_place_name` chain, over the three cached answers rather
    than over names.tsv. OSM has no `name:ja` for a Balinese village, so its ja answer is
    the local latin name — returning nothing instead would hide a resolved place.
    """
    for candidate in (lang, _NAME_FALLBACK_LANG, *_CACHE_LANGS):
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

    def place(self, lang: Lang) -> _Place:
        """The place as the current run needs it — the variant for `lang`.

        F172: the city goes through `_place_name` like every other source, even though the
        provider answers with text and no geonameid to translate by.
        """
        return _Place(
            country=self.country,
            city_geonameid=None,
            district_geonameid=None,
            city=_place_name(lang, provider_name=_pick_lang(self.city, lang)),
            district_name=_pick_lang(self.district, lang),
            country_name=_pick_lang(self.country_name, lang),
        )


def _all_answered(places: dict[str, _Place]) -> bool:
    """Did EVERY language come back with a place? Only then may the row be cached.

    A one-off network failure (two in one live run) must not be frozen into the collection:
    a half-written row would pin the missing language until the expiry date. The files
    still get the answer of this run, and the next run tries again.
    """
    return bool(places) and all(p.country is not None for p in places.values())


class _GeoCacheTable:
    """Access to `geo_cache` (schema v13): provider answers that outlive the run.

    The key is built by the code, because SQLite counts NULLs in a composite primary key
    as distinct rows. Two shapes: `c:<city_geonameid>/<district_geonameid>`, and
    `g:<lat>/<lon>` for coordinates the local base cannot place. The provider is part of
    the key (mixing offline and online answers would be a silent misplacement); the
    language is not — it is a dimension of the value.
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

        An unreadable timestamp counts as expired: asking again is cheap next to trusting
        a row we cannot date.
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

    The escape hatch of F93: a cache can freeze a WRONG provider answer, and "Start over"
    deliberately no longer wipes it (`sorta cache --clear-geo`, `sorta reset --clear-geo`,
    the checkbox in the web app's reset dialog).
    """
    removed = geo_cache_size(conn)
    with conn:
        conn.execute("DELETE FROM geo_cache")
    return removed


def _median_coord(points: list[tuple[float, float]]) -> tuple[float, float]:
    """The representative of a group — the median latitude and longitude.

    Median and not mean: one frame shot from a plane would drag an average out of the
    place entirely. For a district of an awkward shape even the median can land outside
    its border — the caveat F92 carries about its own trip centres.
    """
    return (statistics.median(p[0] for p in points),
            statistics.median(p[1] for p in points))


class _CachedOnlineResolver:
    """F93: the online provider behind a cache keyed by the place of the LOCAL base.

    Before it, adding 200 photos cost the same ~35 minutes of Nominatim as a full run, and
    the language was fixed at `accept-language`, so switching folder language left the
    cities in the old one until the next full pass.

    The key is the (city_geonameid, district_geonameid) pair every coordinate gets for free
    from the bundled KD-tree, and it beats a coordinate grid on both axes at once —
    measured on 14 254 GPS files: 603 requests against 6 219 for a 110 m grid, and zero
    localities mixed against 0.9% of districts. The base has partitioned the map by
    MEANING; a grid invents squares that do not know where a district ends. Coordinates the
    base cannot place fall back to a grid key (`g:`).

    A side effect worth as much as the speed: an online place is now anchored to a city of
    the local base, so event names find a dominant locality again instead of falling back
    to the COUNTRY («Тайланд» instead of «Пхангнга» for 1 359 files).

    On a miss the provider is asked THREE times (ru/en/ja) about the median coordinate of
    the group, and the row is written once.
    """

    def __init__(self, cfg: Config, conn: sqlite3.Connection,
                 client: _NominatimClient, local: GeoResolver | None) -> None:
        self._client = client
        self._local = local
        self._language = i18n.normalize_lang(cfg.language)
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

        # Progress after every GROUP, not once at the end: this loop is ~1 request/sec and
        # can be most of the run, so a counter that ticks only at the end hangs at "0 of N"
        # for those minutes and then races.
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
            "geo: online cache: %d groups (out of %d coordinates), %d hits, %d expired, "
            "%d requests to the provider",
            len(groups), len(coords), self._cache.hits, self._cache.expired,
            self._client.requests,
        )
        return [by_key[key] for key in keys]


class _CityFallbackResolver:
    """F86: the online provider first, the bundled offline base as insurance for the city.

    Nominatim answers for suburbs and the countryside without any of the city keys it is
    read by, so the place used to end up as "country known, city NULL" — 1 596 files of a
    full run (1 471 exact_gps + 125 inherited) that the sorter then hid in
    _Unsorted/no_place, while the SAME coordinates resolve offline to a city.

    Country, city and district always come from ONE source: an offline city replaces the
    whole place, so a Nominatim country name is never glued onto a GeoNames city. If the
    two disagree about the country the offline answer is dropped — a nearest-neighbour city
    in the wrong country is the silent misplacement F75 guards against.

    A place with no country at all (a failed request, an empty address) is NOT asked
    offline: there is no answer to complete, and F65 keeps its meaning — such coordinates
    stay `unknown` instead of quietly switching provider mid-run.
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
                    "geo: the provider answered without a city for (%s, %s), country %s — "
                    "case %d, of which %d got a city from the offline base",
                    coords[i][0], coords[i][1], place.country,
                    self._city_missing, self._city_recovered,
                )
        if self._city_missing:
            _log.warning(
                "geo: answers without a city: %d, of which %d found a city in the offline "
                "base (the rest stay at the country level)",
                self._city_missing, self._city_recovered,
            )
        return places


def _resolver_for(cfg: Config, conn: sqlite3.Connection,
                  offline: GeoResolver) -> _PlaceBatchResolver:
    """Provider abstraction by `cfg.geo.provider`.

    offline -> geodata.GeoResolver, which never touches geo_cache: a full offline
               recompute takes two seconds.
    online  -> Nominatim/OSM (G2b), behind the F93 cache and the F86 city fallback — both
               only if the bundled data is actually there, since online must keep working
               on an install without it.

    `offline` is passed in rather than created here so the F85c path hint shares one
    loaded base with this stage.
    """
    provider = cfg.geo.provider
    lang = i18n.normalize_lang(cfg.language)
    if provider == "offline":
        return _OfflineBatchResolver(offline, lang)
    if provider == "online":
        available = offline.data_available()
        if not available:
            _log.warning(
                "geo: the offline base is unavailable (%s) — cities the provider leaves "
                "out of its answer cannot be recovered, and the answer cache will group "
                "coordinates by grid cell rather than by city and district",
                offline.data_dir,
            )
        online = _CachedOnlineResolver(cfg, conn, _NominatimClient(cfg),
                                      offline if available else None)
        if not available:
            return online
        return _CityFallbackResolver(online, _OfflineBatchResolver(offline, lang))
    raise ValueError(f"geo: unknown geo.provider={provider!r} (expected offline|online)")


def _parse_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        # drop tzinfo: taken_at is local capture time, and a mix of aware and naive
        # values would break sorting
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
# It has to be duplicated: the stages run index -> geo -> landmarks -> faces -> events, so
# on a clean run the `events` table does not exist yet when geo works; and the inheritance
# cannot move into events either, because `places` has exactly one writer
# (docs/ARCHITECTURE.md §2) and the order cannot change (events needs `country` from
# places). So geo groups its own sessions by the same rule and the SAME thresholds.
#
# Two deliberate differences, both toward "inherit less":
# * no admin1-region branch — geo would have to load the bundled base a second time for
#   it, so a trip here can only come out SHORTER than the events one, never wider;
# * a session where nobody has GPS does not break a trip (it neither confirms nor denies
#   the locality, and those sessions are what this feature exists for). In events such a
#   session starts a new group, because there a group IS the event.


@dataclass(frozen=True)
class _TripLocality:
    """Where a session was, as far as its own GPS files know."""

    key: tuple[str, object] | None  # ("i", geonameid) | ("s", casefolded city name)
    country: str | None
    coords: tuple[float, float] | None  # F92: the median GPS of the files of `key`


def _city_key(place: _Place) -> tuple[str, object] | None:
    """The locality key of a resolved place: geonameid, else the city name, else none.

    events._city_key prefers `district_name` for the string fallback, wanting the most
    specific name it can print. Here the question is which CITY the trip agreed on, so a
    suburb must not split one city into three localities.
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

    Session-inherited files are left out: their place is a copy of these same coordinates,
    and counting it again would let one GPS frame outvote another just by having more
    dateless neighbours.
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

    Inside the span the trip is an alibi: the camera was in this city before the file and
    after it. Past an end there is none — the last GPS frame does not say when its owner
    left, and a day-trip out of town lands exactly there. Measured on the validation
    collection: the span holds 414 of the 554 inferences and 4 of the 32 mistakes, the
    other 28 all sitting past an end. Precision inside 99.0%, outside 80.0%.
    """
    return first <= dt <= last


def _trip_place(places: list[_Place]) -> _Place:
    """The place a trip lends: the dominant one of its city, minus the district.

    The trip agreed on a CITY; a district would be a finer claim than the evidence
    supports — the frame may have been shot in another part of town.
    """
    rep = Counter(places).most_common(1)[0][0]
    return replace(rep, district_geonameid=None, district_name=None)


def _inherit_trip_places(
    cfg: Config, sessions: list[list[tuple[datetime, sqlite3.Row]]],
    resolved: dict[int, tuple[_Place, str]], gps: dict[int, tuple[float, float]],
) -> None:
    """F85a: a file with no place inherits the place of its TRIP. Mutates `resolved`.

    Runs AFTER session inheritance, which is more precise and keeps its priority. Two
    conditions, both measured on scripts/measure_place_inference.py (hide the GPS of files
    that have it, infer, compare with the truth — 554 trip-level cases):

    1. the trip's own GPS frames agree about the city: the dominant city holds MORE than
       half of them, a country-only frame counting in the denominator. A trip across three
       cities leaves its place-less files alone — a foreign city is worse than an empty
       folder, because the user will not look there (F75/F86);
    2. the file lies BETWEEN two frames of that city in time — see `_brackets`.

    Rule 1 alone measured 94.2% precision, below the 95% this feature has to clear, and
    every mistake was a file outside the span. Rule 2 brings it to 99.0% and costs about a
    quarter of the reach.
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


# --- F85c: the country a folder NAME gives away -------------------------------------


def _path_segments(path: str) -> list[str]:
    """The DIRECTORY names of a file path, deepest first.

    Split by hand rather than through `Path`: the index stores whatever separator the
    machine that wrote it used, and a POSIX interpreter reads `D:\\Фото\\Греция\\a.jpg` as
    a single segment. The file name is dropped — a camera names files, a person names
    folders.
    """
    parts = [p for p in re.split(r"[\\/]+", path.strip()) if p]
    return list(reversed(parts[:-1]))


def _name_candidates(segment: str) -> list[str]:
    """What in one folder name may be a country: the whole name, then its words.

    The whole name first, being the more specific claim («Коста-Рика» is one name and two
    words). Words shorter than _PATH_HINT_MIN_WORD are left out.
    """
    name = segment.strip()
    out = [name] if name else []
    for word in _WORD.findall(name):
        if len(word) >= _PATH_HINT_MIN_WORD and word != name:
            out.append(word)
    return out


class _CountryFromPath:
    """F85c: country (never city) from the names of the folders a file lies in.

    COUNTRY ONLY. Measured on the GPS files of the live collection — hide it, guess from
    the path, compare: a country from a folder name is right 99.5% of 2 105 hints, a city
    4.3% of 1 152, because the bundled base holds 150 000 settlements and any ordinary word
    resolves to some hamlet. `city_ids_by_name` is not called here and must not be added.

    DEEPEST FIRST. The folder nearest the file is the most specific thing the user said
    about it: under «Отпуска/Греция 2019» the answer is Greece, and a collection root named
    after the country the owner lives in must not outvote it.

    Two bundled offline name sources: the curated i18n dictionary and the resolver's
    GeoNames country names in all three languages (~250 countries, and the spellings people
    actually type — «Тайланд» is in there). Without the bundled data the hint produces
    nothing; it never becomes a network call or a guess.
    """

    def __init__(self, resolver: object) -> None:
        # getattr, not a direct call: a resolver without the reverse lookups (tests inject
        # a mini one) must degrade to the curated dictionary rather than crash the stage.
        self._lookup = getattr(resolver, "country_cc_by_name", None)
        self._cache: dict[str, str | None] = {}

    def _cc_of(self, name: str) -> str | None:
        cc = i18n.country_cc_by_name(name)
        if cc is not None:
            return cc
        if self._lookup is None:
            return None
        for lang in _CACHE_LANGS:
            try:
                found = self._lookup(name, lang)
            except GeoDataMissing:
                self._lookup = None  # the base is not there — do not ask 6 000 more times
                return None
            if found:
                return str(found).upper()
        return None

    def country_of(self, path: str) -> str | None:
        """The ISO cc a folder on this path names, or None if none of them names one."""
        for segment in _path_segments(path):
            for candidate in _name_candidates(segment):
                key = candidate.casefold()
                if key not in self._cache:
                    self._cache[key] = self._cc_of(candidate)
                cc = self._cache[key]
                if cc is not None:
                    return cc
        return None


def _inherit_path_countries(
    rows: list[sqlite3.Row], resolved: dict[int, tuple[_Place, str]],
    hint: _CountryFromPath,
) -> None:
    """F85c: fill the still-place-less files with the country of their folder name.

    Runs LAST: the hint is a person's label, not a measurement, and may never overwrite
    what GPS or a neighbour in time established. `taken_at_confidence` is not consulted
    (unlike the two inheritance rules) — a folder name says nothing about time, and an
    undated file goes down the sorter's own undated branch anyway.

    The place is country-only by construction; the sorter has a branch for exactly that
    shape (`country_only`, F86) — `<Country>/<year>/`.
    """
    for r in rows:
        if r["id"] in resolved:
            continue
        cc = hint.country_of(r["path"])
        if cc is None:
            continue
        resolved[r["id"]] = (
            _Place(country=cc, city_geonameid=None, district_geonameid=None, city=None),
            "path_inferred",
        )


def resolve_places(
    cfg: Config, conn: sqlite3.Connection,
    progress: Callable[[int, int], None] | None = None,
) -> GeoStats:
    """Resolve the place of each canonical file and fully recompute places."""
    gap_hours = float(cfg.geo.session_gap_hours)

    rows = conn.execute(
        """SELECT id, path, taken_at, taken_at_confidence, gps_lat, gps_lon
           FROM files WHERE dup_of IS NULL AND error IS NULL"""
    ).fetchall()
    # One bundled resolver for the whole stage: the batch resolver below and the F85c path
    # hint read the same 12 MB of GeoNames. Construction loads nothing (GeoResolver reads
    # lazily), so this costs nothing when neither is reached.
    local = GeoResolver()

    # 1) exact_gps. Coordinates may be garbage ('' from broken EXIF), so the unparsable
    #    ones are skipped — geodata/scipy would crash on them.
    gps_rows: list[sqlite3.Row] = []
    coords: list[tuple[float, float]] = []
    for r in rows:
        lat, lon = _coord(r["gps_lat"]), _coord(r["gps_lon"])
        if lat is not None and lon is not None and not _is_null_island(lat, lon):
            gps_rows.append(r)
            coords.append((lat, lon))
    resolved: dict[int, tuple[_Place, str]] = {}
    # F85a: a trip locality's center is the median of the files' OWN GPS (F92), so the
    # distance check needs the coordinates again below.
    gps_by_id = {r["id"]: c for r, c in zip(gps_rows, coords)}
    gps_unresolved = 0
    if coords:
        resolver = _resolver_for(cfg, conn, local)
        # Online, the whole network phase is this call (~1 request/sec, minutes on a real
        # collection), so progress moves here and not in the write loop below.
        places = resolver.resolve_places(coords, progress=progress)
        for r, place in zip(gps_rows, places):
            # F65: coordinates alone do not make an exact_gps. An empty place (missing geo
            # data offline, a failed request online) stays "unknown" rather than filling
            # the DB with confident-looking NULLs, and never becomes an inheritance donor.
            if place.country is None:
                gps_unresolved += 1
                continue
            resolved[r["id"]] = (place, "exact_gps")
        if gps_unresolved:
            # Online has already logged each failed request; this is the total.
            data_dir = getattr(resolver, "data_dir", None)
            _log.warning(
                "geo: %d of %d files with coordinates did not resolve to a place%s — "
                "check the geo data (places.tsv)",
                gps_unresolved, len(coords),
                f" (geo data: {data_dir})" if data_dir else "",
            )

    # 2) session_inferred: the FULL place (country + both geonameids + city) inherited
    #    within a time session.
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

    # 2b) trip_inferred (F85a): a session where nobody has GPS inherits from the trip
    #     around it. Second on purpose — session inheritance is the more precise of the two.
    _inherit_trip_places(cfg, sessions, resolved, gps_by_id)

    # 2c) path_inferred (F85c): the COUNTRY off a folder name, for what no geometric
    #     signal reached at all. Last in the queue by design.
    _inherit_path_countries(rows, resolved, _CountryFromPath(local))

    # 3) write: full recomputation of the places table in one transaction
    stats = GeoStats(total=len(rows), gps_unresolved=gps_unresolved)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if progress:
        progress(0, len(rows))  # total right away, even if the stage is small/fast (#37)
    # Pure SQLite from here — the network, if any, already ran in the resolve above.
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
                    print(i18n.cli_text(
                        "cli.geo.progress",
                        i18n.normalize_lang(cfg.raw.get("language")),
                        done=i, total=len(rows)))
        if progress and rows:
            progress(len(rows), len(rows))
    return stats
