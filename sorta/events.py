"""F4/F30/F44 (Phase 4): events.

Contract: reads files + places, writes ONLY into events and event_files.

Rules:
- auto events (origin='auto'): sessions by a gap > events.gap_hours (default 6);
  adjacent sessions merge into one event (a large trip) when the gap is <
  events.trip_merge_gap_hours (default 48) AND the same "trip locality":
  the same country (places.country), AND (the same city OR the same admin1 region OR
  cities closer than events.trip_merge_max_km, F44/#19). City — places.city_geonameid;
  if it is NULL (online provider, G2b), the string fallback
  places.district_name/city (F44/#19-A1) is used — such cities have no
  coordinates/region, so merging works only by string equality for them. An unknown
  locality does not confirm a merge. Files with taken_at_confidence='low' do not
  enter auto events.
- Size threshold (F30): groups (after merging) with a file count <
  events.min_event_size (default 5) do not become an auto event — their files
  do not enter event_files (the sorter routes them down the no_event branch).
- The event name and `place_city` — the localized locality of the group (F44/#19-B):
  one city per group → the city name (geodata.GeoResolver.name/string fallback);
  several cities → the admin1 region of the DOMINANT (by file count) city
  (GeoResolver.region_name), and if there is no region — the country (country_name),
  and if not that either — the name of the dominant city. No file has a locality
  → no city.
- Recomputation recreates auto events; a manual name (name_is_manual=1) is carried
  over to a new event if it overlaps the old one by files > 50% (of the old one).
- Manual events (origin='manual', add_manual_event) are NOT recreated by
  recomputation — only the canonical files of the range are reattached by taken_at
  (including 'low': an explicit user instruction outranks heuristics). Their files
  are excluded from auto-clustering. Overlapping ranges of two manual events is an
  error. Manual events are not subject to the size threshold.

Parameters — the typed config.yaml `events:` section (cfg.events). The
`min_event_size`/`trip_merge_gap_hours` fields — F30, still read via getattr with a
default (EventsConfig does not type them yet).
"""
from __future__ import annotations

import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import asin, cos, radians, sin, sqrt
from typing import Callable

from . import i18n
from .config import Config
from .geodata import GeoResolver

_PROGRESS_EVERY = 100
_DEFAULT_MIN_EVENT_SIZE = 5
_DEFAULT_TRIP_MERGE_GAP_HOURS = 48.0
_DEFAULT_TRIP_MERGE_MAX_KM = 120.0
_EARTH_RADIUS_KM = 6371.0088


@dataclass
class EventStats:
    auto_events: int = 0
    manual_events: int = 0
    auto_files: int = 0
    manual_files: int = 0
    names_preserved: int = 0


@dataclass(frozen=True)
class _File:
    id: int
    dt: datetime
    confidence: str | None
    city_id: int | None  # places.city_geonameid — the city itself (G2), not a district/string
    city_str: str | None  # F44/#19-A1: the string fallback (district_name/city) when city_id is NULL
    country_cc: str | None  # places.country (ISO cc) — for the "same country" check (F44/#19-B)


@dataclass(frozen=True)
class _Locality:
    """The locality (city) of a session/file group — for merging and the trip name."""

    city_id: int | None
    city_str: str | None  # meaningful only when city_id is None
    key: tuple[str, object] | None  # locality equality: ("i", geonameid) | ("s", casefold)
    country_cc: str | None


def _parse_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        # drop tzinfo: taken_at is local capture time, a mix of aware/naive would
        # break sorting and comparisons
        return datetime.fromisoformat(s).replace(tzinfo=None)
    except ValueError:
        return None


def _load_files(conn: sqlite3.Connection) -> list[_File]:
    """Canonical files with a date, sorted by time.

    City — geonameid from places.city_geonameid; when it is NULL (online provider,
    G2b does not resolve geonameid), city_str is the string fallback district_name/city
    (F44/#19-A1) so the online path does not lose the place.
    """
    rows = conn.execute(
        """SELECT f.id, f.taken_at, f.taken_at_confidence AS confidence,
                  p.city_geonameid, p.district_name, p.city, p.country
           FROM files f LEFT JOIN places p ON p.file_id = f.id
           WHERE f.dup_of IS NULL AND f.error IS NULL AND f.taken_at IS NOT NULL"""
    ).fetchall()
    out = []
    for r in rows:
        dt = _parse_dt(r["taken_at"])
        if dt is None:
            continue
        city_id = r["city_geonameid"]
        city_str = None
        if city_id is None:
            raw = r["district_name"] or r["city"]
            city_str = raw.strip() if raw and raw.strip() else None
        out.append(_File(r["id"], dt, r["confidence"], city_id, city_str, r["country"]))
    out.sort(key=lambda f: (f.dt, f.id))
    return out


def _split_sessions(files: list[_File], gap_hours: float) -> list[list[_File]]:
    """Files sorted by dt; a gap > gap_hours starts a new session."""
    sessions: list[list[_File]] = []
    gap = timedelta(hours=gap_hours)
    for f in files:
        if sessions and f.dt - sessions[-1][-1].dt <= gap:
            sessions[-1].append(f)
        else:
            sessions.append([f])
    return sessions


def _dominant_city_id(files: list[_File]) -> int | None:
    """By city_geonameid only (no string fallback) — for manual events, whose logic
    brief F44 says not to change."""
    ids = Counter(f.city_id for f in files if f.city_id is not None)
    return ids.most_common(1)[0][0] if ids else None


def _city_name(resolver: GeoResolver, lang: i18n.Lang, city_id: int | None) -> str | None:
    """geonameid -> localized city name; None -> no city (F30)."""
    return resolver.name(city_id, lang) if city_id is not None else None


def _city_key(f: _File) -> tuple[str, object] | None:
    """A file's locality key: geonameid, else the normalized string fallback, else None."""
    if f.city_id is not None:
        return ("i", f.city_id)
    if f.city_str:
        return ("s", f.city_str.strip().casefold())
    return None


def _file_city_name(resolver: GeoResolver, lang: i18n.Lang, f: _File) -> str | None:
    """A file's display city name: geonameid -> resolver; else the string fallback as-is."""
    if f.city_id is not None:
        return resolver.name(f.city_id, lang)
    return f.city_str


def _dominant_locality(files: list[_File]) -> _Locality:
    """The dominant (by file count) locality among files + the dominant country."""
    keys = Counter(k for f in files if (k := _city_key(f)) is not None)
    country_ccs = Counter(f.country_cc for f in files if f.country_cc)
    country_cc = country_ccs.most_common(1)[0][0] if country_ccs else None
    if not keys:
        return _Locality(city_id=None, city_str=None, key=None, country_cc=country_cc)
    best_key = keys.most_common(1)[0][0]
    rep = next(f for f in files if _city_key(f) == best_key)
    return _Locality(city_id=rep.city_id, city_str=rep.city_str, key=best_key, country_cc=country_cc)


def _haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = radians(a[0]), radians(a[1]), radians(b[0]), radians(b[1])
    h = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * asin(sqrt(h))


def _same_trip(
    anchor: _Locality, cand: _Locality, resolver: GeoResolver, max_km: float,
) -> bool:
    """F44/#19-B: the same country AND (the same city OR the same admin1 region OR

    cities closer than max_km). Cities without a geonameid (online strings) have no
    coordinates/region — for them only string-key equality applies.
    """
    if not anchor.country_cc or not cand.country_cc or anchor.country_cc != cand.country_cc:
        return False
    if anchor.key is not None and anchor.key == cand.key:
        return True
    if anchor.city_id is None or cand.city_id is None:
        return False
    region_a = resolver.region_key_of(anchor.city_id)
    if region_a is not None and region_a == resolver.region_key_of(cand.city_id):
        return True
    if max_km > 0:
        coords_a, coords_b = resolver.coords_of(anchor.city_id), resolver.coords_of(cand.city_id)
        if coords_a is not None and coords_b is not None:
            return _haversine_km(coords_a, coords_b) <= max_km
    return False


def _merge_sessions(
    sessions: list[list[_File]], trip_gap_hours: float,
    resolver: GeoResolver, trip_merge_max_km: float,
) -> list[tuple[list[_File], _Locality]]:
    """Merge adjacent sessions into a trip: gap < trip_gap_hours AND _same_trip.

    The group's anchor locality is taken from the FIRST session and is not
    recomputed on further merges (as before for city_id) — the comparison always
    runs against it, not against the last added session.
    """
    trip_gap = timedelta(hours=trip_gap_hours)
    groups: list[tuple[list[_File], _Locality]] = []
    for session in sessions:
        locality = _dominant_locality(session)
        if groups:
            prev_files, anchor = groups[-1]
            if (session[0].dt - prev_files[-1].dt < trip_gap
                    and _same_trip(anchor, locality, resolver, trip_merge_max_km)):
                prev_files.extend(session)
                continue
        groups.append((session, locality))
    return groups


def _group_place_name(files: list[_File], resolver: GeoResolver, lang: i18n.Lang) -> str | None:
    """The group locality AFTER merging (F44/#19-B): one city -> its name;

    several cities -> the admin1 region of the dominant (by file count) city,
    else the country, else the name of the dominant city. No known city at all
    -> None.
    """
    keys = Counter(k for f in files if (k := _city_key(f)) is not None)
    if not keys:
        return None
    dom_key, _n = keys.most_common(1)[0]
    dom_file = next(f for f in files if _city_key(f) == dom_key)
    if len(keys) == 1:
        return _file_city_name(resolver, lang, dom_file)
    if dom_file.city_id is not None:
        region_key = resolver.region_key_of(dom_file.city_id)
        if region_key is not None:
            region_name = resolver.region_name(region_key[0], region_key[1], lang)
            if region_name:
                return region_name
    if dom_file.country_cc:
        country_name = resolver.country_name(dom_file.country_cc, lang)
        if country_name:
            return country_name
    return _file_city_name(resolver, lang, dom_file)


def _event_name(start: datetime, end: datetime, place_name: str | None) -> str:
    """`YYYY-MM-DD <Locality>`; multi-day — `YYYY-MM-DD..MM-DD`; no locality — without it."""
    d1, d2 = start.date(), end.date()
    if d1 == d2:
        base = d1.isoformat()
    elif d1.year == d2.year:
        base = f"{d1.isoformat()}..{d2.strftime('%m-%d')}"
    else:
        base = f"{d1.isoformat()}..{d2.isoformat()}"
    return f"{base} {place_name}" if place_name else base


def build_events(
    cfg: Config, conn: sqlite3.Connection,
    progress: Callable[[int, int], None] | None = None,
) -> EventStats:
    """Full recomputation of auto events; for manual ones only reattach range files."""
    gap_hours = float(cfg.events.gap_hours)
    trip_gap_hours = float(
        getattr(cfg.events, "trip_merge_gap_hours", _DEFAULT_TRIP_MERGE_GAP_HOURS))
    min_event_size = int(getattr(cfg.events, "min_event_size", _DEFAULT_MIN_EVENT_SIZE))
    trip_merge_max_km = float(
        getattr(cfg.events, "trip_merge_max_km", _DEFAULT_TRIP_MERGE_MAX_KM))
    lang = i18n.normalize_lang(cfg.raw.get("language"))
    resolver = GeoResolver()  # F30: one resolver per run, lazy loading of bundled data

    files = _load_files(conn)
    stats = EventStats()
    with conn:
        # 1) manual events: not recreated; the range files (including low) are
        #    reattached — newly indexed ones are picked up
        manual_ids: set[int] = set()
        manual_rows = conn.execute(
            "SELECT id, started_at, ended_at FROM events WHERE origin = 'manual'"
        ).fetchall()
        for r in manual_rows:
            m_start, m_end = _parse_dt(r["started_at"]), _parse_dt(r["ended_at"])
            in_range = [
                f.id for f in files
                if m_start is not None and m_end is not None and m_start <= f.dt <= m_end
            ]
            conn.execute("DELETE FROM event_files WHERE event_id = ?", (r["id"],))
            conn.executemany(
                "INSERT INTO event_files (event_id, file_id) VALUES (?, ?)",
                [(r["id"], fid) for fid in in_range],
            )
            manual_ids.update(in_range)
        stats.manual_events = len(manual_rows)
        stats.manual_files = len(manual_ids)

        # 2) manual names of old auto events — candidates for carry-over (brief item 4)
        named_old: list[tuple[str, set[int]]] = []
        for r in conn.execute(
            "SELECT id, name FROM events WHERE origin = 'auto' AND name_is_manual = 1"
        ).fetchall():
            old_files = {
                row["file_id"] for row in conn.execute(
                    "SELECT file_id FROM event_files WHERE event_id = ?", (r["id"],))
            }
            if old_files:
                named_old.append((r["name"], old_files))

        # 3) recreation of auto events
        conn.execute(
            """DELETE FROM event_files WHERE event_id IN
               (SELECT id FROM events WHERE origin = 'auto')"""
        )
        conn.execute("DELETE FROM events WHERE origin = 'auto'")

        auto = [f for f in files if f.confidence != "low" and f.id not in manual_ids]
        groups = _merge_sessions(
            _split_sessions(auto, gap_hours), trip_gap_hours, resolver, trip_merge_max_km)
        # F30: small groups (< min_event_size) do not become an auto event —
        # their files simply do not enter event_files (sorter: the no_event branch)
        created = [g for g in groups if len(g[0]) >= min_event_size]
        for group, _anchor in created:
            start, end = group[0].dt, group[-1].dt
            place_city = _group_place_name(group, resolver, lang)
            name, is_manual, best = _event_name(start, end, place_city), 0, 0.0
            new_ids = {f.id for f in group}
            for old_name, old_files in named_old:
                overlap = len(old_files & new_ids) / len(old_files)
                if overlap > 0.5 and overlap > best:
                    name, is_manual, best = old_name, 1, overlap
            stats.names_preserved += is_manual
            cur = conn.execute(
                """INSERT INTO events
                   (started_at, ended_at, place_city, name, name_is_manual, origin)
                   VALUES (?, ?, ?, ?, ?, 'auto')""",
                (start.isoformat(), end.isoformat(), place_city, name, is_manual),
            )
            eid = cur.lastrowid
            conn.executemany(
                "INSERT INTO event_files (event_id, file_id) VALUES (?, ?)",
                [(eid, f.id) for f in group],
            )
            stats.auto_events += 1
            stats.auto_files += len(group)
            if progress and stats.auto_events % _PROGRESS_EVERY == 0:
                progress(stats.auto_events, len(created))
    return stats


def rename_event(conn: sqlite3.Connection, event_id: int, name: str) -> None:
    """Manual name: survives recomputation (carried over on a > 50% file overlap)."""
    with conn:
        cur = conn.execute(
            "UPDATE events SET name = ?, name_is_manual = 1 WHERE id = ?",
            (name, event_id),
        )
        if cur.rowcount == 0:
            raise ValueError(f"events: событие с id={event_id} не найдено")


def _parse_bound(s: str, *, end: bool) -> datetime:
    try:
        dt = datetime.fromisoformat(s).replace(tzinfo=None)
    except ValueError as exc:
        raise ValueError(
            f"events: не удалось разобрать дату {s!r} — ожидается ISO 8601, "
            "например 2026-01-10"
        ) from exc
    if end and len(s) <= 10:  # a date without time → end of day inclusive
        dt = dt.replace(hour=23, minute=59, second=59, microsecond=999999)
    return dt


def add_manual_event(
    conn: sqlite3.Connection, name: str, date_from: str, date_to: str,
) -> int:
    """A manual event (origin='manual', name_is_manual=1) over a date range.

    Attaches ALL canonical files of the range by taken_at, including
    taken_at_confidence='low', and takes them from existing auto events.
    Manual-event ranges cannot overlap. Does not take cfg (the same call from cli.py
    without a config) — `place_city` is localized with the default language
    (i18n.normalize_lang(None)), not the config.yaml language.
    """
    start = _parse_bound(date_from, end=False)
    end = _parse_bound(date_to, end=True)
    if start > end:
        raise ValueError(f"events: date_from {date_from!r} позже date_to {date_to!r}")

    for r in conn.execute(
        "SELECT id, name, started_at, ended_at FROM events WHERE origin = 'manual'"
    ).fetchall():
        o_start, o_end = _parse_dt(r["started_at"]), _parse_dt(r["ended_at"])
        if o_start is not None and o_end is not None and start <= o_end and end >= o_start:
            raise ValueError(
                f"events: диапазон {date_from}..{date_to} пересекается с ручным "
                f"событием '{r['name']}' (id={r['id']}, "
                f"{r['started_at']}..{r['ended_at']}) — сначала измените его"
            )

    files = [f for f in _load_files(conn) if start <= f.dt <= end]
    city_id = _dominant_city_id(files)
    place_city = _city_name(GeoResolver(), i18n.normalize_lang(None), city_id)
    with conn:
        cur = conn.execute(
            """INSERT INTO events
               (started_at, ended_at, place_city, name, name_is_manual, origin)
               VALUES (?, ?, ?, ?, 1, 'manual')""",
            (start.isoformat(), end.isoformat(), place_city, name),
        )
        event_id = cur.lastrowid
        assert event_id is not None
        conn.executemany(
            "INSERT INTO event_files (event_id, file_id) VALUES (?, ?)",
            [(event_id, f.id) for f in files],
        )
        # manual-event priority: the range files leave auto events
        if files:
            marks = ",".join("?" * len(files))
            conn.execute(
                f"""DELETE FROM event_files WHERE file_id IN ({marks})
                    AND event_id IN (SELECT id FROM events WHERE origin = 'auto')""",
                [f.id for f in files],
            )
            conn.execute(
                """DELETE FROM events WHERE origin = 'auto'
                   AND id NOT IN (SELECT event_id FROM event_files)"""
            )
    return event_id
