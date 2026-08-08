"""F182: the "Layout" tab — the plan, the canon, the places, the settings.

One canon and a physical move: the plan of where every frame goes, the overrides a
person puts on top of it, the place assigned to a whole group at once, the albums
built beside the canon, the people clusters they can be built from, and the settings
column that governs the run. `PlanCache` is here because the plan is what this tab
is: the other tabs read it, none of them build it.

F192: the tab asks two questions — where the collection goes and by what it is
grouped — and the second one is the CRITERION (`sorter.MODES`: city, person, event),
which `PlanCache` has always keyed its modes by and which `_run_sort` used to
hard-code to "city". Everything else the tab can do moved behind a gear on the screen;
nothing moved in this module, because none of it was ever about placement.
"""
from __future__ import annotations

import dataclasses
import os
import sqlite3
import threading
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .. import i18n, imaging
from ..config import Config
from ..geodata import GeoDataMissing, GeoResolver
from ..sorter import (
    ALBUM_KINDS, ALBUM_MODES, SELECTORLESS_ALBUM_KINDS, AlbumReport, PlanItem, plan_and_sort,
)
# F104: the pre-apply summary has to say what the apply will DO, so it asks the two
# functions the apply itself uses rather than re-deriving the rule here — the moment
# the two answers can differ, the dialog is quoting numbers nobody has to honour.
# `_fs`: the long-path form a filesystem call needs on Windows; `_is_the_same_file`:
# "the file already lying at the target is byte-for-byte the one we would put there".
from ..sorter import _fs, _is_the_same_file
# F203: a reassign target is now typed rather than picked from the plan, so the route has
# to refuse a bad one — with the layout's own naming rule, not a second copy of it.
from ..sorter import manual_target_parts
from .common import (
    _CLUSTER_SAMPLE_LIMIT, _DEFAULT_ALBUM_DIRNAME, _EVENT_SAMPLE_LIMIT, _SUPPORTED_MODES,
    _UI_LANGS, _connect, _is_under, _log, _validate_file_ids_payload,
)
# F202: a region option has to SAY it is a region — one word can be a city and a region
# at once («Алтай» is both) — and that word is a caption, so it comes from the catalog
# through the same resolver the page fills its placeholders with.
from .page import _t


def _plan_item_to_json(item: PlanItem,
                       override: tuple[str, str | None] | None = None) -> dict:
    # G3: `item.city` already comes in the folder language (sorter._city_display_name)
    # — the grid of the "Cities"/"Events" tabs must not label a frame «St Petersburg»
    # while the target folder right next to it reads «Санкт-Петербург». One function
    # decides both, so the plan and the card can never disagree.
    geo = "/".join(p for p in (item.country, item.city) if p) or None
    payload = {
        "file_id": item.file_id,
        "name": item.src.name,
        # Where the file came FROM. Only the basename used to reach the UI, yet the
        # source folder is often the best evidence there is about a frame: 41% of this
        # collection sits in hand-named directories ("Тайланд 04.2025",
        # "Турция. Белек") — a person's own labelling of place and date. It is also
        # what you need in order to judge a wrong guess: a Colosseum match is plainly
        # wrong once you can see the file lives under "карелия".
        "src_dir": item.src.parent.name,
        "src_path": str(item.src.parent),
        "target_rel": item.target_rel,
        "reason": item.reason,
        "date": item.taken_at,
        "geo": geo,
        # F85c: how confidently the place was determined — and, for `manual`, that it
        # was not determined at all but chosen by the user. The grid draws its own mark
        # off this, so a hand-assigned place never reads as something the program found.
        "place_confidence": item.place_confidence,
        "category": item.reason,
        "thumb_url": f"/thumb/{item.file_id}",
        # F80: video and photo tiles used to be indistinguishable in the grid. The
        # extension is enough (the indexer decides media_type the same way) and costs
        # no query — the plan carries no media_type of its own.
        "video": imaging.is_video_path(item.src),
    }
    if override is not None:
        # F77: only a corrected file carries the mark — the frontend draws a frame off
        # the presence of the key, so an uncorrected row must not carry a null.
        payload["override"] = override[0]
        payload["override_target"] = override[1]
    return payload


def _plan_category(item: PlanItem) -> str:
    """The target FOLDER of a plan item — the aggregation key of `/api/plan` (F70).

    `target_rel` is POSIX and always carries at least one directory segment (see
    sorter._target_parts — every branch returns a non-empty folder list), so the key
    is never empty; a pathological item without a folder falls back to target_rel.
    """
    head, sep, _name = item.target_rel.rpartition("/")
    return head if sep else item.target_rel


def _dest_occupancy(items: list[PlanItem], dest: Path | None) -> tuple[int, int]:
    """(taken, identical) target paths of `items` inside `dest` — F104.

    `taken` — the plan item's target name is already occupied; `identical` — by a
    byte-for-byte copy of that very file, i.e. the apply will SKIP it (F97) instead of
    writing a `_1` twin next to it. The difference between the two numbers is the file
    that will be written after all, under another name.

    The rule is asked of `sorter._is_the_same_file` rather than re-implemented: the
    dialog states what the apply is going to do, and the moment the two can disagree
    the numbers stop being a promise. `dest=None` — the destination could not be
    resolved (see `_summary_dest`), so nothing is claimed about it.
    """
    if dest is None:
        return 0, 0
    taken = identical = 0
    for item in items:
        head, sep, _name = item.target_rel.rpartition("/")
        target_dir = dest.joinpath(*head.split("/")) if sep else dest
        # The name the apply TRIES first — the `_1` suffixes come after this one.
        target = target_dir / item.src.name
        if not _fs(target).exists():
            continue
        taken += 1
        # src == dst is the in-place layout: the file IS the one lying at the target,
        # and it is skipped just as surely as an identical copy would be.
        if (os.path.normcase(str(target)) == os.path.normcase(str(item.src))
                or _is_the_same_file(target, item.src, item.db_hash, item.db_algo)):
            identical += 1
    return taken, identical


def _overrides_map(db_path: Path) -> dict[int, tuple[str, str | None]]:
    """F77: file_id -> (action, target) from `manual_overrides` — the live marks.

    Read per request instead of being stored in the built plan: a correction must be
    visible right after it is saved, and invalidating the plan of a mode would make the
    next request pay for a full rebuild (see PlanCache). The table holds one row per
    corrected file, so it is tiny next to the plan itself.
    """
    conn = _connect(db_path)
    try:
        rows = conn.execute("SELECT file_id, action, target FROM manual_overrides").fetchall()
    finally:
        conn.close()
    return {int(r["file_id"]): (r["action"], r["target"]) for r in rows}


class _ModePlan:
    """One built mode: the plan items plus the per-folder index the routes serve.

    Both the aggregate rows and the per-category buckets are computed once, at build
    time — a request then only slices a ready list, so `/api/plan` costs milliseconds
    regardless of the collection size.
    """

    def __init__(self, items: list[PlanItem], sizes: dict[int, int]) -> None:
        self.items = items
        # F104: kept, not only folded into the rows below — the pre-apply summary sums
        # the volume of the files that will actually move, which is not the sum of the
        # folder rows (those include what the user marked "leave alone").
        self.sizes = sizes
        buckets: dict[str, list[PlanItem]] = defaultdict(list)
        for item in items:
            buckets[_plan_category(item)].append(item)
        self.by_category: dict[str, list[PlanItem]] = dict(buckets)
        self.categories: list[dict] = [
            {
                "category": name,
                "count": len(group),
                "size": sum(sizes.get(it.file_id, 0) for it in group),
            }
            for name, group in sorted(self.by_category.items())
        ]


class PlanCache:
    """An in-memory cache of report.plan by mode, built LAZILY — one mode on its
    first request — and dropped explicitly (`rebuild`) after `/api/process` (F36),
    a reset, an apply or a folder-language change, and NOT on every external DB update.

    F70: building all three modes eagerly cost ~40 s on a 26k collection, both at
    `sorta ui` start and on every rebuild, with the user staring at a dead window.
    Now `__init__`/`rebuild` only record what to build; the work happens on the
    thread that first asks for that mode, and only for the mode actually opened.

    sqlite3 connections are not transferable between threads (`check_same_thread`),
    and ThreadingHTTPServer serves each request on a new thread — so a lazy build
    opens its own short-lived connection from `cfg.database` instead of reusing the
    connection of whoever created the cache (see `_connect`, the same reason).

    Thread safety: a mode is built under its own lock, so a request burst from
    several ThreadingHTTPServer threads produces one build and one shared result.
    A `rebuild` that lands mid-build bumps the generation counter, and the finished
    (now stale) plan is simply not stored.
    """

    def __init__(self, cfg: Config, conn: sqlite3.Connection, dest: Path) -> None:
        self._dest = dest
        self._cfg = cfg
        self._db_path = Path(cfg.database).resolve()
        self._by_mode: dict[str, _ModePlan] = {}
        self._generation = 0
        self._state_lock = threading.Lock()
        self._build_locks = {mode: threading.Lock() for mode in _SUPPORTED_MODES}

    def rebuild(self, cfg: Config, conn: sqlite3.Connection) -> None:
        """Invalidate every built mode — the next request recomputes what it needs.

        The signature is kept as-is (the pipeline/sort threads call it with their own
        cfg/conn), but nothing is computed here anymore: a rebuild that blocks the
        caller for ~40 s is exactly what F70 removed. `conn` is deliberately unused —
        it belongs to the calling thread, and the lazy build runs on another one.
        """
        with self._state_lock:
            self._cfg = cfg
            self._db_path = Path(cfg.database).resolve()
            self._by_mode = {}
            self._generation += 1

    def _plan(self, mode: str) -> _ModePlan | None:
        """The built mode (building it if needed), or None for an unsupported mode."""
        if mode not in _SUPPORTED_MODES:
            return None
        with self._state_lock:
            built = self._by_mode.get(mode)
            if built is not None:
                return built
        with self._build_locks[mode]:
            with self._state_lock:
                built = self._by_mode.get(mode)
                if built is not None:
                    return built
                cfg, generation = self._cfg, self._generation
            built = self._build(cfg, mode)
            with self._state_lock:
                if generation == self._generation:
                    self._by_mode[mode] = built
            return built

    def _build(self, cfg: Config, mode: str) -> _ModePlan:
        """One dry-run plan + the file sizes the aggregate reports, in one connection.

        keep_manual_excluded=True (F77): a file marked "leave alone" is not moved by
        `sort --apply` (the sorter drops it from any plan that moves anything), but it
        must stay VISIBLE and unmarkable here — otherwise marking a frame would make it
        vanish from the grid on the next rebuild, with no way back.
        """
        conn = _connect(self._db_path)
        try:
            report = plan_and_sort(cfg, conn, mode, self._dest, apply=False,
                                   write_reports=False, keep_manual_excluded=True)
            sizes = {int(row["id"]): int(row["size"] or 0)
                     for row in conn.execute("SELECT id, size FROM files")}
        finally:
            conn.close()
        return _ModePlan(report.plan, sizes)

    def get(self, mode: str) -> list[PlanItem] | None:
        """The list of PlanItem for a mode, or None for an unsupported mode."""
        built = self._plan(mode)
        return None if built is None else built.items

    def aggregate(self, mode: str) -> dict | None:
        """`GET /api/plan?mode=` — target folders with counts/sizes, no file list.

        F77: the totals also say how many of the plan's files carry a manual correction
        (`overridden`) and how many of those are "leave alone" (`excluded`). The latter
        are LISTED (see `_build`) but will not be moved, so the apply confirmation counts
        `total - excluded`. Counted per request from the live table; the per-folder rows
        keep their existing shape (folder/count/size) — the marks themselves travel with
        the files, on the category page.
        """
        built = self._plan(mode)
        if built is None:
            return None
        marks = _overrides_map(self._db_path)
        actions = [marks[it.file_id][0] for it in built.items if it.file_id in marks]
        return {"mode": mode, "total": len(built.items),
                "overridden": len(actions),
                "excluded": sum(1 for a in actions if a == "exclude"),
                "categories": built.categories}

    def summary(self, mode: str, dest: Path | None) -> dict | None:
        """`GET /api/sort/summary` — the numbers the pre-apply dialog states (F104).

        Everything is read off the SAME built plan the "Cities" tree draws, so the
        dialog cannot quote a number the tab does not show: `files`/`dirs` leave out
        what the user marked "leave alone" (exactly as `aggregate` does), `bytes` is
        the volume of precisely those files, and the two review folders are counted by
        the plan's own reason codes — a folder NAME changes with the folder language,
        a reason does not.

        `dest` is the destination the form is about to send (None — it could not be
        resolved, see `_summary_dest`). What is already lying there is asked of the
        filesystem with the rule `sorter._resolve_dst` applies at apply time, so
        "already there, will be skipped" in the dialog means the same event that
        `report.skipped_already_copied`/`skipped_in_place` will count. That costs a
        stat per file (plus a hash where the size matches), which is why this is a
        request of its own and not part of every `/api/plan`.
        """
        built = self._plan(mode)
        if built is None:
            return None
        marks = _overrides_map(self._db_path)
        items = [it for it in built.items
                 if marks.get(it.file_id, ("", None))[0] != "exclude"]
        existing, same = _dest_occupancy(items, dest)
        return {
            "mode": mode,
            "dest": str(dest) if dest is not None else None,
            "files": len(items),
            "dirs": len({_plan_category(it) for it in items}),
            "bytes": sum(built.sizes.get(it.file_id, 0) for it in items),
            "products": sum(1 for it in items if it.reason == "product"),
            "documents": sum(1 for it in items if it.reason == "document"),
            "dest_existing": existing,
            "dest_same": same,
        }

    def page(self, mode: str, category: str, offset: int, limit: int) -> dict | None:
        """`GET /api/plan?mode=&category=&offset=&limit=` — one page of one folder.

        An unknown category is an empty page with `total: 0` (not an error): a folder
        can disappear between an aggregate and a click on it.
        """
        built = self._plan(mode)
        if built is None:
            return None
        items = built.by_category.get(category, [])
        page = items[offset:offset + limit]
        marks = _overrides_map(self._db_path) if page else {}
        return {
            "mode": mode,
            "category": category,
            "total": len(items),
            "offset": offset,
            "limit": limit,
            "items": [_plan_item_to_json(it, marks.get(it.file_id)) for it in page],
        }


_OVERRIDE_ACTIONS = ("exclude", "reassign", "clear", "photo")


def _reassign_target_refusal(payload: object) -> str | None:
    """F203: why this body's reassign target is no folder name, or None when it is one.

    The target is TYPED now, not picked from the plan's folders, so the answer to a bad
    one has to be a refusal with a reason — before F203 the route stored anything and the
    sorter dropped it hours later, at apply time, into a log nobody reads. The rule
    itself is `sorter.manual_target_parts`, the same function the layout cleans its own
    folder names with: two rules for one name is how the folder shown here and the folder
    written there start disagreeing.

    Asked of the RAW body, before its shape is validated (the `class_album_refusal`
    arrangement of F193): that is what lets `../../evil` come back saying it leaves the
    sort root instead of the flat "invalid body" every other malformed field earns.
    Anything that is not a reassign is None and travels on to the ordinary validation.
    """
    if not isinstance(payload, dict) or payload.get("action") != "reassign":
        return None
    target = payload.get("target")
    if not isinstance(target, str):
        return None  # a missing or non-string target is a malformed body, not a bad name
    _parts, refusal = manual_target_parts(target)
    return refusal


def _validate_overrides_payload(payload: object) -> tuple[list[int], str, str | None] | None:
    """Parse the body `POST /api/overrides` (F77):
    `{"file_ids": [int,...], "action": "exclude"|"reassign"|"clear"|"photo",
    "target": str?}`.

    None -> invalid (400): not an object, an unknown/absent action, file_ids that is not
    a non-empty list of ints (bool excluded, like everywhere else), or `reassign`
    without a non-empty target. The target is NOT resolved into a path here — it is a
    folder of the layout, checked as a NAME by `_reassign_target_refusal` (F203) and
    stored as the user typed it; sorter._manual_target_parts asks the same question again
    before a destination is built from it.

    F103: `photo` ("the classifier is wrong, this IS a personal photo") carries no
    target — the whole point is that the file goes back to the AUTOMATIC city layout,
    not to a folder someone had to name.
    """
    if not isinstance(payload, dict):
        return None
    action = payload.get("action")
    if action not in _OVERRIDE_ACTIONS:
        return None
    ids = _validate_file_ids_payload(payload)
    if ids is None:
        return None
    if action == "reassign":
        target = payload.get("target")
        if not isinstance(target, str) or not target.strip():
            return None
        return ids, action, target.strip()
    return ids, action, None


def _apply_overrides(db_path: Path, file_ids: list[int], action: str,
                     target: str | None) -> list[int]:
    """Write (or, for 'clear', delete) the manual marks; returns the affected file_ids.

    One row per file: a repeated correction of the same file overwrites it via ON
    CONFLICT rather than adding a second row. Ids outside `files` are silently skipped
    (the same rule as `_trash_files`; the FK on manual_overrides.file_id would reject
    them anyway). One transaction for the whole selection — a bulk correction either
    lands entirely or not at all.
    """
    now = datetime.now(timezone.utc).isoformat()
    conn = _connect(db_path)
    try:
        placeholders = ",".join("?" * len(file_ids))
        known = [r["id"] for r in conn.execute(
            f"SELECT id FROM files WHERE id IN ({placeholders})", file_ids)]
        if not known:
            return []
        ph = ",".join("?" * len(known))
        with conn:
            if action == "clear":
                conn.execute(
                    f"DELETE FROM manual_overrides WHERE file_id IN ({ph})", known)
            else:
                conn.executemany(
                    """INSERT INTO manual_overrides (file_id, action, target, updated_at)
                       VALUES (?, ?, ?, ?)
                       ON CONFLICT(file_id) DO UPDATE SET
                           action = excluded.action, target = excluded.target,
                           updated_at = excluded.updated_at""",
                    [(fid, action, target, now) for fid in known])
    finally:
        conn.close()
    return known


# --- F85c: assigning a place to a whole group at once -------------------------------
# About 6 300 files of the live collection carry no place signal at all — no GPS, no
# neighbour in time with one, no landmark, and nothing readable in the folder name. No
# model will place them: the information is not in them. It is in the person who took
# them, and the only thing that stands between them and a correct place is that clicking
# six thousand times is not a thing anyone will do. Hence: pick a GROUP the user already
# thinks in (a whole event, a whole source folder), pick a place from the bundled base,
# one action.

_PLACE_KINDS = ("event", "source_dir")
_PLACE_ACTIONS = ("assign", "clear")
_PLACE_SEARCH_LIMIT = 12
# F201: the country half of the answer never crowds the cities out of a list this
# short — a two-letter prefix matches a dozen country names on its own ("ma": Macao,
# Madagascar, Malawi, Malaysia, Maldives, Mali, Malta, Marshall Islands...).
_PLACE_COUNTRY_LIMIT = 4
# F202: and the region half is capped for the same reason — «Se» begins the name of a
# dozen regions, and the cities they would push out are what most searches are about.
_PLACE_REGION_LIMIT = 4
# Below this many characters the picker does not search at all. One letter is not a
# request: it matches thousands of settlements, so the answer would be noise, and an
# empty answer to it must not be read as "no such place" either (see
# `_places_search_payload`). Two rather than three because a name can be that short —
# «東京» is the whole of Tokyo, and «Мо» is a town in Norway.
_PLACE_SEARCH_MIN_QUERY = 2

_geo_resolver_cache: GeoResolver | None = None


def _geo_resolver() -> GeoResolver:
    """The bundled GeoNames resolver, loaded at most once per server process.

    The place picker asks it on every keystroke (debounced), and the data behind it is
    12 MB plus a KD-tree — building that per request would make the field unusable.
    """
    global _geo_resolver_cache
    if _geo_resolver_cache is None:
        _geo_resolver_cache = GeoResolver()
    return _geo_resolver_cache


@dataclasses.dataclass(frozen=True)
class _ManualPlace:
    """What a `manual_places` row holds: a country, narrowed to one region or one city.

    F202: the three levels are one choice, never a mix — a row carries a city, or a
    region, or neither. That is F85c's rule («the place comes from ONE source») applied
    inside the manual row itself: a region the user named must not acquire the city the
    program inferred, and the validator refuses a body that asks for both.
    """

    country: str
    city: str | None = None
    city_geonameid: int | None = None
    region_geonameid: int | None = None


def _country_label(cc: str, lang: i18n.Lang) -> str:
    """The country name to SHOW: the curated dictionary first, then the bundled base."""
    curated = i18n.country(cc, lang)
    if curated != cc:
        return curated
    try:
        return _geo_resolver().country_name(cc, lang) or cc
    except GeoDataMissing:
        return cc


def _city_option(gid: int, cc: str, lang: i18n.Lang) -> dict:
    """One city of the answer: the geonameid the DB stores, told apart in the label.

    Same-named cities are separated by region and country rather than picked between —
    picking for the user would be guessing.
    """
    resolver = _geo_resolver()
    region = resolver.region_key_of(gid)
    region_name = resolver.region_name(cc, region[1], lang) if region else None
    city_name = resolver.name(gid, lang)
    details = ", ".join(p for p in (region_name, _country_label(cc, lang)) if p)
    return {
        "kind": "city", "country": cc, "city_geonameid": gid, "region_geonameid": None,
        "city": resolver.name(gid, "en"),
        "label": f"{city_name} ({details})" if details else city_name,
    }


def _region_option(gid: int, cc: str, lang: i18n.Lang) -> dict:
    """One admin1 region of the answer — F202.

    The label names the LEVEL and not only the country, because one word is regularly
    both a city and a region: «Алтай» is a Russian republic and two Mongolian towns, and
    a list that shows the three of them without saying which is which asks the user to
    guess. The country still follows, for the same reason a city label carries it.
    """
    resolver = _geo_resolver()
    details = ", ".join((_t("place_kind_region", lang), _country_label(cc, lang)))
    return {
        "kind": "region", "country": cc, "city_geonameid": None, "city": None,
        "region_geonameid": gid,
        "label": f"{resolver.name(gid, lang)} ({details})",
    }


def _region_candidates(query: str, lang: i18n.Lang,
                       limit: int = _PLACE_REGION_LIMIT) -> list[dict]:
    """Admin1 regions whose name in ANY of the three languages starts with `query`.

    The order follows `_city_candidates`: a finished name first (typing «Крым» in full
    means that region, whatever else begins with it), then alphabetically by the shown
    name — regions carry no population to rank by, and an arbitrary order in a list of
    four is a list that reshuffles itself between two identical searches.

    A region with no country is dropped like a city with none: the layout starts at the
    country folder, so such an option could not be laid out at all.
    """
    resolver = _geo_resolver()
    exact: set[int] = set()
    found: set[int] = set()
    for search_lang in _UI_LANGS:
        exact.update(resolver.region_ids_by_name(query, search_lang))  # type: ignore[arg-type]
        found.update(resolver.region_ids_by_prefix(query, search_lang))  # type: ignore[arg-type]
    found |= exact
    with_country: list[tuple[int, str]] = []
    for gid in found:
        key = resolver.region_key_by_id(gid)
        if key is not None and key[0]:
            with_country.append((gid, key[0]))
    with_country.sort(key=lambda pair: (pair[0] not in exact,
                                        resolver.name(pair[0], lang)))
    return [_region_option(gid, cc, lang) for gid, cc in with_country[:limit]]


def _city_candidates(query: str, lang: i18n.Lang, limit: int = _PLACE_SEARCH_LIMIT,
                     ) -> list[dict]:
    """Cities of the bundled base whose name in ANY of the three languages STARTS with
    `query` — the whole name, or any word inside a composite one («Новг» -> «Нижний
    Новгород»). The geonameids are the ones `city_ids_by_name` (F46) answers with and the
    ones that land in `places.city_geonameid`; only the way they are LOOKED UP is wider.

    The order is part of the answer (F201), and it is:

    1. an exact full-name match first — if the user finished typing a name, that name is
       what they meant, whatever else begins with it («Мо» is a town in Norway before it
       is the start of «Москва»);
    2. then by population, descending — the base holds 150 000 settlements, so every
       prefix finds hamlets, and «Моск» has to answer «Москва» before «Москаленки»;
    3. then by the shown name, alphabetically — a predictable tail for the places the
       base gives no population for.

    Labels are built only for the `limit` that survive the cut: a two-letter prefix
    matches thousands of cities, and every label costs a region and a country lookup.
    """
    resolver = _geo_resolver()
    exact: set[int] = set()
    found: set[int] = set()
    for search_lang in _UI_LANGS:
        exact.update(resolver.city_ids_by_name(query, search_lang))  # type: ignore[arg-type]
        found.update(resolver.city_ids_by_prefix(query, search_lang))  # type: ignore[arg-type]
    found |= exact
    # Without a country the place cannot be laid out (the layout starts at the country
    # folder), so such a city is not offered at all.
    with_country = [(gid, cc) for gid in found if (cc := resolver.country_of(gid))]
    with_country.sort(key=lambda pair: (pair[0] not in exact,
                                        -resolver.population_of(pair[0]),
                                        resolver.name(pair[0], lang)))
    return [_city_option(gid, cc, lang) for gid, cc in with_country[:limit]]


def _country_candidates(query: str, lang: i18n.Lang,
                        limit: int = _PLACE_COUNTRY_LIMIT) -> list[dict]:
    """Countries whose name starts with the typed text, the exact match first.

    The curated dictionary (`i18n.country_cc_by_name`) is asked for a full name first:
    it is hand-checked and its spellings are the ones the layout writes. The bundled
    base then adds what it knows — by full name, then by prefix, alphabetically by the
    label the user will read.
    """
    resolver = _geo_resolver()
    ordered: list[str] = []
    seen: set[str] = set()

    def add(cc: str | None) -> None:
        if cc and cc.upper() not in seen:
            seen.add(cc.upper())
            ordered.append(cc.upper())

    add(i18n.country_cc_by_name(query))
    for search_lang in _UI_LANGS:
        add(resolver.country_cc_by_name(query, search_lang))  # type: ignore[arg-type]
    by_prefix = {cc for search_lang in _UI_LANGS
                 for cc in resolver.country_ccs_by_prefix(query, search_lang)}  # type: ignore[arg-type]
    for cc in sorted(by_prefix, key=lambda c: _country_label(c, lang)):
        add(cc)
    return [{"kind": "country", "country": cc, "city_geonameid": None, "city": None,
             "region_geonameid": None, "label": _country_label(cc, lang)}
            for cc in ordered[:limit]]


def _places_search(query: str, lang: i18n.Lang) -> list[dict]:
    """`GET /api/places/search` — what the typed text may mean, widest level first.

    Country first because it is the safer answer: a wrong country is a mistake the user
    can see in one glance at the plan, and the country level is where a file with no
    other signal belongs anyway. F202 puts REGIONS second by the same measure: the
    bigger the miss, the more visible it is in the plan, and a wrong region reads almost
    as loudly as a wrong country while a wrong city disappears among the right ones.
    All three halves read ONLY the bundled base — no network, no model, and nothing is
    written until the user picks one and confirms.

    The text is treated as a PREFIX, because that is what a combobox promises: the field
    is typed into letter by letter, and an answer that arrives only on the finished name
    arrives too late to help. Below `_PLACE_SEARCH_MIN_QUERY` characters nothing is
    searched at all — see `_places_search_payload` for the difference that makes.
    """
    text = query.strip()
    if len(text) < _PLACE_SEARCH_MIN_QUERY:
        return []
    try:
        results = _country_candidates(text, lang)
        results.extend(_region_candidates(text, lang))
        results.extend(_city_candidates(text, lang,
                                        limit=_PLACE_SEARCH_LIMIT - len(results)))
    except GeoDataMissing:
        # The bundled base is the only source here; without it the picker offers
        # nothing rather than pretending an empty answer means "no such place".
        _log.warning("ui: гео-данные недоступны — поиск места вернёт пустой список")
        return []
    return results


def _places_search_payload(query: str, lang: i18n.Lang) -> dict:
    """The body of `GET /api/places/search`: the results, and whether we SEARCHED.

    "Nothing found" and "not asked yet" are different answers, and telling them apart is
    the whole point of F201: the first says the name is wrong, the second says the name
    is not finished. An empty list looks the same from the client, so the server says
    which one it is — and the threshold stays in one place, here, instead of being
    copied into the page.
    """
    text = query.strip()
    return {"query": text,
            "searched": len(text) >= _PLACE_SEARCH_MIN_QUERY,
            "results": _places_search(text, lang)}


def _validate_place_payload(
    payload: object,
) -> tuple[str, str, str, _ManualPlace | None, bool] | None:
    """Parse the body of `POST /api/place`:
    `{"kind": "event"|"source_dir", "selector": str, "action": "assign"|"clear",
      "country": str?, "city_geonameid": int?, "region_geonameid": int?,
      "include_gps": bool?}`.

    None -> invalid (400). `assign` needs a country (a city alone would leave the layout
    without its top folder); `city_geonameid` is optional and narrows it to one city,
    `region_geonameid` (F202) narrows it to a region instead. The two are mutually
    exclusive: a body carrying both asks for a place at two levels at once, and quietly
    honouring one of them is how a hand-picked region ends up with an inferred city
    under it. An id that is not a region of the bundled base is refused as well — the
    layout would have nothing to name the folder with.
    The selector is NOT resolved here — an event id is looked up in the DB, and a source
    folder is only ever COMPARED against `files.path`, never opened (see
    `_place_target_ids`).
    """
    if not isinstance(payload, dict):
        return None
    kind = payload.get("kind")
    action = payload.get("action")
    selector = payload.get("selector")
    if kind not in _PLACE_KINDS or action not in _PLACE_ACTIONS:
        return None
    if not isinstance(selector, str) or not selector.strip():
        return None
    include_gps = bool(payload.get("include_gps"))
    if action == "clear":
        return kind, selector.strip(), action, None, include_gps
    country = payload.get("country")
    if not isinstance(country, str) or not country.strip():
        return None
    gid = payload.get("city_geonameid")
    region_gid = payload.get("region_geonameid")
    for value in (gid, region_gid):
        if value is not None and (not isinstance(value, int) or isinstance(value, bool)):
            return None
    if gid is not None and region_gid is not None:
        return None
    city = None
    try:
        if gid is not None:
            city = _geo_resolver().name(gid, "en")
        if region_gid is not None and _geo_resolver().region_key_by_id(region_gid) is None:
            return None
    except GeoDataMissing:
        return None
    return (kind, selector.strip(), action,
            _ManualPlace(country=country.strip().upper(), city=city, city_geonameid=gid,
                         region_geonameid=region_gid),
            include_gps)


def _place_target_ids(conn: sqlite3.Connection, kind: str, selector: str) -> list[int]:
    """The canonical files of the chosen group — one event, or one source folder.

    Only these two kinds exist on purpose: both are groups the user already sees as a
    thing (a card on the "Events" tab, a folder in the plan), and both are BOUNDED. "The
    whole collection in one action" is deliberately not offered — the larger the grab,
    the higher the price of a wrong pick, and undoing it means finding the files again.
    """
    if kind == "event":
        try:
            event_id = int(selector)
        except ValueError:
            return []
        rows = conn.execute(
            """SELECT f.id FROM event_files ef JOIN files f ON f.id = ef.file_id
               WHERE ef.event_id = ? AND f.dup_of IS NULL AND f.error IS NULL""",
            (event_id,),
        ).fetchall()
        return [int(r["id"]) for r in rows]
    rows = conn.execute(
        "SELECT id, path FROM files WHERE dup_of IS NULL AND error IS NULL").fetchall()
    return [int(r["id"]) for r in rows if _is_under(r["path"], selector)]


def _apply_bulk_place(db_path: Path, kind: str, selector: str, action: str,
                      place: _ManualPlace | None, include_gps: bool) -> dict:
    """Write (or drop) the manual place of a whole group. Returns what happened.

    Files with `confidence='exact_gps'` are SKIPPED unless `include_gps` is set: those
    were placed by the camera at the moment of the shot, and a memory of which city a
    trip was in is not better evidence than a coordinate. They are counted and reported
    back, so the client can offer to include them — an explicit second decision, never a
    silent overwrite. `clear` skips nothing: dropping a manual row can only restore what
    the program itself worked out.

    One transaction for the whole group — a bulk assignment either lands entirely or not
    at all, which is what makes "undo" a single action too.
    """
    now = datetime.now(timezone.utc).isoformat()
    conn = _connect(db_path)
    try:
        ids = _place_target_ids(conn, kind, selector)
        skipped_gps = 0
        if ids and action == "assign" and not include_gps:
            ph = ",".join("?" * len(ids))
            with_gps = {int(r["file_id"]) for r in conn.execute(
                f"""SELECT file_id FROM places
                    WHERE confidence = 'exact_gps' AND file_id IN ({ph})""", ids)}
            skipped_gps = len(with_gps)
            ids = [fid for fid in ids if fid not in with_gps]
        if ids:
            ph = ",".join("?" * len(ids))
            with conn:
                if action == "clear":
                    conn.execute(
                        f"DELETE FROM manual_places WHERE file_id IN ({ph})", ids)
                else:
                    assert place is not None  # guaranteed by _validate_place_payload
                    # Every column of the place is written on a conflict, the empty ones
                    # included: an assignment replaces the whole place (F85c), so a
                    # region assigned over a city has to CLEAR that city rather than
                    # leave it standing one level down (F202).
                    conn.executemany(
                        """INSERT INTO manual_places
                               (file_id, country, city, city_geonameid,
                                region_geonameid, updated_at)
                           VALUES (?, ?, ?, ?, ?, ?)
                           ON CONFLICT(file_id) DO UPDATE SET
                               country = excluded.country, city = excluded.city,
                               city_geonameid = excluded.city_geonameid,
                               region_geonameid = excluded.region_geonameid,
                               updated_at = excluded.updated_at""",
                        [(fid, place.country, place.city, place.city_geonameid,
                          place.region_geonameid, now) for fid in ids])
    finally:
        conn.close()
    return {
        "ok": True, "action": action, "kind": kind, "selector": selector,
        "affected": len(ids), "skipped_gps": skipped_gps,
        "country": place.country if place else None,
        "city_geonameid": place.city_geonameid if place else None,
        "region_geonameid": place.region_geonameid if place else None,
    }


def _clusters_payload(db_path: Path, sample_limit: int = _CLUSTER_SAMPLE_LIMIT) -> list[dict]:
    """Root clusters (`merged_into IS NULL`) with size/label/samples.

    size — the number of faces in the whole merge chain (the root + everything merged
    into it), not just faces whose `faces.cluster_id` points directly to the root
    (after `merge` it keeps pointing to the original cluster — see `faces.merge`).
    samples — up to `sample_limit` distinct file_ids, ordered by `faces.id`
    (deterministic, stable between requests). Noise clusters (`faces.cluster_id IS
    NULL`) are naturally excluded by the `WHERE cluster_id IS NOT NULL` filter. Sorted
    by descending size.
    """
    conn = _connect(db_path)
    try:
        cluster_rows = conn.execute(
            "SELECT id, label, merged_into FROM face_clusters"
        ).fetchall()
        face_rows = conn.execute(
            "SELECT cluster_id, file_id FROM faces "
            "WHERE cluster_id IS NOT NULL ORDER BY id"
        ).fetchall()
    finally:
        conn.close()

    merged_into = {r["id"]: r["merged_into"] for r in cluster_rows}
    labels = {r["id"]: r["label"] for r in cluster_rows}
    root_ids = [r["id"] for r in cluster_rows if r["merged_into"] is None]

    def root_of(cid: int) -> int:
        seen: set[int] = set()
        while merged_into.get(cid) is not None and cid not in seen:
            seen.add(cid)
            cid = merged_into[cid]
        return cid

    size: dict[int, int] = defaultdict(int)
    samples: dict[int, list[int]] = defaultdict(list)
    sample_seen: dict[int, set[int]] = defaultdict(set)
    for r in face_rows:
        root = root_of(r["cluster_id"])
        size[root] += 1
        seen_files = sample_seen[root]
        if r["file_id"] not in seen_files and len(samples[root]) < sample_limit:
            seen_files.add(r["file_id"])
            samples[root].append(r["file_id"])

    result = [
        {
            "cluster_id": rid,
            "size": size.get(rid, 0),
            "label": labels.get(rid),
            "samples": samples.get(rid, []),
        }
        for rid in root_ids
    ]
    result.sort(key=lambda c: (-c["size"], c["cluster_id"]))
    return result


def _validate_cluster_label_payload(payload: object) -> tuple[int, str] | None:
    """Parse `{"cluster_id": int, "name": str}`. None -> invalid."""
    if not isinstance(payload, dict):
        return None
    cluster_id = payload.get("cluster_id")
    name = payload.get("name")
    if not isinstance(cluster_id, int) or isinstance(cluster_id, bool):
        return None
    if not isinstance(name, str):
        return None
    return cluster_id, name


def _validate_cluster_merge_payload(payload: object) -> tuple[int, int] | None:
    """Parse `{"src": int, "dst": int}`. None -> invalid."""
    if not isinstance(payload, dict):
        return None
    src = payload.get("src")
    dst = payload.get("dst")
    if not isinstance(src, int) or isinstance(src, bool):
        return None
    if not isinstance(dst, int) or isinstance(dst, bool):
        return None
    return src, dst


def _album_dest(cfg: Config, db_path: Path) -> Path:
    """The album root: `cfg.sort.album_dir` if set in the config, otherwise the default next to the DB."""
    album_dir = getattr(cfg.sort, "album_dir", None)
    if album_dir:
        return Path(album_dir)
    return db_path.resolve().parent / _DEFAULT_ALBUM_DIRNAME


def _suggested_sort_dest(cfg: Config, db_path: Path) -> str:
    """The default destination path for the city layout: `<source>_sorted`.

    The source — the first `cfg.sources` (config.yaml); if empty — the common root of
    the indexed files from the DB. Nothing found → an empty string (the field stays
    for manual entry). A POSIX path (like sources in config).

    A source that is not on THIS disk suggests nothing: the installer writes config.yaml
    from the shipped example, whose `sources` is the sample `D:/Photos`, and a default
    built from a folder that does not exist is wrong on every machine.
    """
    root: Path | None = None
    if cfg.sources and Path(cfg.sources[0]).is_dir():
        root = Path(cfg.sources[0])
    else:
        try:
            conn = _connect(db_path)
            try:
                paths = [r[0] for r in conn.execute(
                    "SELECT path FROM files WHERE error IS NULL").fetchall()]
            finally:
                conn.close()
            if paths:
                common = os.path.commonpath(paths)
                # commonpath over files returns an ancestor directory; if it matched a
                # single file (the only path) — take its parent
                root = Path(common)
                if root.suffix:  # this is a file, not a directory
                    root = root.parent
        except (ValueError, OSError):
            root = None
    if root is None:
        return ""
    return (root.parent / (root.name + "_sorted")).as_posix()


def _events_payload(db_path: Path,
                    sample_limit: int = _EVENT_SAMPLE_LIMIT) -> list[dict]:
    """The event list for the "Events" tab: id/name/count/dates + up to
    `sample_limit` preview file_ids (clickable -> lightbox), by descending count."""
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            """SELECT e.id, e.name, e.started_at, e.ended_at,
                      COUNT(ef.file_id) AS count
               FROM events e LEFT JOIN event_files ef ON ef.event_id = e.id
               GROUP BY e.id
               ORDER BY count DESC, e.id"""
        ).fetchall()
        # samples in a separate pass: the event's canonical frames by time,
        # up to sample_limit per event (as _clusters_payload accumulates in Python)
        samples: dict[int, list[int]] = defaultdict(list)
        for s in conn.execute(
            """SELECT ef.event_id, ef.file_id
               FROM event_files ef JOIN files f ON f.id = ef.file_id
               WHERE f.dup_of IS NULL AND f.error IS NULL
               ORDER BY ef.event_id, f.taken_at, f.id"""
        ):
            bucket = samples[s["event_id"]]
            if len(bucket) < sample_limit:
                bucket.append(s["file_id"])
    finally:
        conn.close()
    return [
        {
            "id": r["id"],
            "name": r["name"],
            "count": r["count"],
            "started_at": r["started_at"],
            "ended_at": r["ended_at"],
            "samples": samples.get(r["id"], []),
        }
        for r in rows
    ]


def _validate_album_payload(
    payload: object,
) -> tuple[str, str, str, list[str], str | None, bool, str | None] | None:
    """Parse the body `POST /api/album`. None -> invalid (400).

    kind/mode — from `ALBUM_KINDS`/`ALBUM_MODES` (sorter.py), selector — a non-empty
    string, `where` (opt.) — a list of strings, `name` (opt.) — a string (empty after
    strip is treated as absent — the default name is used), `apply` (opt., default
    False) — bool, `dest` (opt., F60) — the album destination path as a string;
    empty/absent -> None (the server resolves the default itself via `_album_dest`).

    F123: `kind='animal'` is the one kind with nothing to select — the collection has a
    single animal slice — so an empty selector is accepted there (and only there: for a
    person or an event an empty selector is a client that lost its subject, and
    gathering "everything" would be the wrong answer to it).
    F139: the class and quality slices join it, and F152 the three face slices, by the
    same rule and through the same shared list (`SELECTORLESS_ALBUM_KINDS`).

    Whether a KIND may be gathered at all is not decided here: that answer depends on
    `vlm.exclude_classes` and is given by the route, which has the config (F133).
    """
    if not isinstance(payload, dict):
        return None
    kind = payload.get("kind")
    if kind not in ALBUM_KINDS:
        return None
    mode = payload.get("mode")
    if mode not in ALBUM_MODES:
        return None
    selectorless = kind in SELECTORLESS_ALBUM_KINDS
    selector = payload.get("selector", "" if selectorless else None)
    if not isinstance(selector, str):
        return None
    if not selectorless and not selector.strip():
        return None
    where = payload.get("where", [])
    if not isinstance(where, list) or not all(isinstance(w, str) for w in where):
        return None
    name = payload.get("name")
    if name is not None:
        if not isinstance(name, str):
            return None
        name = name.strip() or None
    apply_ = payload.get("apply", False)
    if not isinstance(apply_, bool):
        return None
    dest = payload.get("dest")
    if dest is not None:
        if not isinstance(dest, str):
            return None
        dest = dest.strip() or None
    return kind, selector, mode, where, name, apply_, dest


def _album_report_to_json(report: AlbumReport, applied: bool) -> dict:
    """`AlbumReport` -> the JSON response body of `POST /api/album`.

    For a preview (`applied=False`) `plan_album` does not compute `blocked_multi`
    (that is a side effect of the apply loop for mode='move') — here it is recomputed
    from `report.plan` with the same logic (`item.multi_person`), so the preview shows
    the expected blocking before the real move.
    """
    blocked = report.blocked_multi
    if not applied and report.mode == "move":
        blocked = sum(1 for it in report.plan if it.multi_person)
    return {
        "album_name": report.album_name,
        "dest": str(report.dest),
        "mode": report.mode,
        "kind": report.kind,
        "count": len(report.plan),
        "blocked_multi": blocked,
        "transferred": report.transferred,
        "failed": report.failed,
        "applied": applied,
    }


# --- F43: apply the city layout from the UI (`POST /api/sort`) — reuses the
# sorter.plan_and_sort(apply=True) engine one-to-one with the CLI `sort --by city
# --apply`; ui.py here is only background/progress (the _ProcessState/_run_pipeline
# pattern from F36) and request-body validation. The moves/move_batches journal,
# blake3 verification, name-conflict resolution and in-place semantics (dest=None) —
# entirely in sorter.py, not duplicated.

class _SortState:
    """Thread-safe state of the background `/api/sort` apply (F43) — modelled on
    `_ProcessState`, but without stages (one `plan_and_sort` operation).

    F97: it also carries a cancel flag now. Unlike `_ProcessState`, the flag is only
    READ (`cancel_requested` is handed to `plan_and_sort` as `should_cancel`) — it
    never raises out of a callback. The layout has a batch to close before it may
    stop, so the engine decides when to break, not the state object.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._reset_locked()

    def _reset_locked(self) -> None:
        self.running = False
        self.done = 0
        self.total = 0
        self.error: str | None = None
        self.finished = False
        self.result: dict | None = None
        self._cancel_requested = False

    def try_start(self) -> bool:
        """True and switches to running if nothing is going now; otherwise False (409)."""
        with self._lock:
            if self.running:
                return False
            self._reset_locked()
            self.running = True
            return True

    def set_progress(self, done: int, total: int) -> None:
        with self._lock:
            self.done = done
            self.total = total

    def request_cancel(self) -> None:
        with self._lock:
            if self.running:
                self._cancel_requested = True

    def cancel_requested(self) -> bool:
        with self._lock:
            return self._cancel_requested

    def finish(self, error: str | None, result: dict | None) -> None:
        with self._lock:
            self.running = False
            self.finished = True
            self.error = error
            self.result = result

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "running": self.running,
                "done": self.done,
                "total": self.total,
                "error": self.error,
                "finished": self.finished,
                "result": self.result,
                "cancel_requested": self._cancel_requested,
            }


def _validate_sort_payload(payload: object) -> tuple[str | None, str, str] | None:
    """Parse the body `POST /api/sort`:
    `{"dest": str|null|"", "mode": "move"|"copy"?, "by": "city"|"person"|"event"?}`.

    None -> invalid (400): not dict / `mode` present but not in {move, copy} / `by`
    outside `_SUPPORTED_MODES` / `dest` not a string and not null. `dest` an
    empty/whitespace string or null -> None (in-place — layout inside the source folder,
    see `plan_and_sort` F28).

    F200: `mode` is optional and falls back to "copy" — the same value the radio in
    `page.html` now carries `checked`. The screen and the parser are two defaults for
    one question, and a body that omits the field has to land where the screen says it
    would; copy is the answer that destroys nothing if the click was careless.

    F192: `by` is the criterion the layout screen now asks for — the same three values
    `sorta sort --by` and `GET /api/plan?mode=` have taken since F5. It is optional and
    falls back to "city": that was the only criterion the web app could apply before, so
    a client that does not send the field keeps meaning exactly what it used to.

    `mode` and `by` are deliberately two fields and not one: move-or-copy is HOW the
    files travel, the criterion is WHERE they land, and a single key covering both is
    how a copy by person becomes a move by city.
    """
    if not isinstance(payload, dict):
        return None
    mode = payload.get("mode", "copy")
    if mode not in ("move", "copy"):
        return None
    by = payload.get("by", "city")
    if by not in _SUPPORTED_MODES:
        return None
    dest = payload.get("dest")
    if dest is not None and not isinstance(dest, str):
        return None
    dest = dest.strip() if isinstance(dest, str) else None
    return (dest or None), mode, by


def _validate_language_payload(payload: object) -> str | None:
    """Parse the body `POST /api/config/language`: `{"language": "ru"|"en"|"ja"}`.

    None -> invalid (400): not a dict / `language` not one of the supported codes."""
    if not isinstance(payload, dict):
        return None
    lang = payload.get("language")
    if not isinstance(lang, str):
        return None
    lang = lang.strip().lower()
    return lang if lang in _UI_LANGS else None


# --- F104: the settings column of the "Cities" tab (`/api/settings`) ----------
# A toggle in the interface has to change what the tool DOES, not just what a file
# says — so for every knob here the question "what does writing it invalidate?" is
# answered explicitly, and the answer is what makes a restart unnecessary:
#
#   vlm.model    — which weights to load. Read when the model is first needed, i.e.
#                  inside the next run. Nothing to invalidate.
#   vlm.workers  — the frame-preparation pool. Read when the VLM pass starts.
#   vlm.max_edge — the input size of a frame. Read per frame from that run's config.
#
# The folder language is the one setting with a consequence — the plan preview is
# BUILT in that language — and it keeps its own endpoint (`POST /api/config/language`,
# F65), which rebuilds the plan cache. It is not folded in here precisely because its
# answer to the question above is different.


@dataclasses.dataclass(frozen=True)
class _SettingSpec:
    """What a settings key accepts.

    `minimum`/`maximum` apply to `kind` of "int" and "float"; `choices` to "choice",
    which is a string restricted to a fixed set (a select in the form, not a text box —
    a scope the server would refuse is not worth offering).
    """
    kind: str  # bool | str | int | float | choice
    minimum: float = 0
    maximum: float = 0
    choices: tuple[str, ...] = ()


# The bounds are sanity rails, not tuning advice: 0 threads or a 4-pixel frame is a
# typo, and a 40 000-pixel one is a typo that costs the whole VRAM budget. The `min`/
# `max` attributes of the number inputs in the template carry the same numbers — a test
# pins the two together, because a form that offers a value the server refuses is worse
# than no form.
_SETTINGS_SPEC: dict[str, _SettingSpec] = {
    "vlm.model": _SettingSpec("str"),
    "vlm.workers": _SettingSpec("int", 1, 32),
    "vlm.max_edge": _SettingSpec("int", 128, 4096),
    # F138: `vlm.enabled`, `vlm.quality`, `vlm.quality_scope` and `features.pets` are
    # NOT here any more. They decide what THIS run costs — between a quarter of an hour
    # and four hours each — so they live on the run screen with their price next to
    # them, and a knob that moved there leaves this column: two entry points for one
    # value give two truths and a question about which one is in force. What stays is
    # what costs a run nothing — the thresholds, the model, the pool, the input size,
    # the cache ceiling. The config FILE remains their home; the screen starts from it
    # (`/api/process/defaults`) and overrides it for one run only.
    "features.pet_threshold": _SettingSpec("float", 0.0, 1.0),
    "features.sharpness_max_edge": _SettingSpec("int", 64, 4096),
    "features.sharpness_band_min": _SettingSpec("float", 0.0, 10000.0),
    "features.sharpness_band_max": _SettingSpec("float", 0.0, 10000.0),
    "features.subject_score_min": _SettingSpec("float", 0.0, 1.0),
    # F117: the preview-cache ceiling in GB. 0 is a legal value and the default — it
    # means "no ceiling", the behaviour since F67, so the minimum cannot be 1. The
    # upper rail is a typo guard: nobody caps a preview cache at four terabytes.
    "imaging.preview_cache_max_gb": _SettingSpec("int", 0, 4096),
}

# Which config object each section's keys live on. `imaging:` is the exception and maps
# to the environment instead (config._IMAGING_ENV — imaging.py is a leaf module that
# pool workers call with a path and nothing else), so applying it means setting the
# variable rather than replacing a dataclass field.
_SETTING_SECTIONS = ("vlm", "features")
_IMAGING_SETTING_ENV = {
    "imaging.preview_cache_max_gb": imaging.ENV_PREVIEW_MAX_GB,
}


def _settings_payload(cfg: Config) -> dict:
    """`GET /api/settings` — the current values, straight out of the RUNNING config."""
    values = {
        key: getattr(getattr(cfg, key.split(".", 1)[0]), key.split(".", 1)[1])
        for key in _SETTINGS_SPEC if key.split(".", 1)[0] in _SETTING_SECTIONS
    }
    return {
        **values,
        # Read through imaging, not off cfg: the environment is the source of truth for
        # this one, and a shell export legitimately overrides the file.
        "imaging.preview_cache_max_gb": int(imaging.preview_cache_max_gb()),
    }


def _validate_settings_payload(payload: object) -> dict[str, object] | None:
    """Parse the body of `POST /api/settings`: `{"<key>": <value>, …}`. None -> 400.

    The WHOLE body is rejected on the first bad key or value — a half-applied save
    would leave the file and the running config disagreeing about which half of the
    form the user is looking at. An empty body is invalid too: it would answer "ok"
    without having done anything.
    """
    if not isinstance(payload, dict) or not payload:
        return None
    values: dict[str, object] = {}
    for key, raw in payload.items():
        spec = _SETTINGS_SPEC.get(key)
        if spec is None:
            return None
        if spec.kind == "bool":
            if not isinstance(raw, bool):
                return None
            values[key] = raw
        elif spec.kind == "str":
            if not isinstance(raw, str) or not raw.strip():
                return None
            values[key] = raw.strip()
        elif spec.kind == "choice":
            # F119: a fixed set, so a misspelling is a 400 rather than a silent
            # fallback. `vlm.quality_scope` is the one where that matters: `all` is the
            # 4.3-hour option, and drifting into it by accident is expensive.
            if not isinstance(raw, str) or raw not in spec.choices:
                return None
            values[key] = raw
        elif spec.kind == "float":
            # bool is an int is not a float here either; ints are accepted and widened,
            # because a form posting `1` for a threshold of 1.0 is not an error.
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                return None
            if not spec.minimum <= float(raw) <= spec.maximum:
                return None
            values[key] = float(raw)
        else:
            # bool is an int in Python — `workers: true` is garbage, not 1.
            if isinstance(raw, bool) or not isinstance(raw, int):
                return None
            if not spec.minimum <= raw <= spec.maximum:
                return None
            values[key] = raw
    return values


def _apply_settings(cfg: Config, values: dict[str, object]) -> None:
    """Put validated values into the RUNNING config (see the note above: nothing else
    has to be invalidated — every one of them is read when the next run starts)."""
    fields = {key: key.split(".", 1)[1] for key in _SETTINGS_SPEC}
    # F117: the imaging keys live in the environment rather than on a dataclass, so they
    # are applied separately and taken OUT before the vlm replace below — passing one to
    # dataclasses.replace would raise on an unknown field.
    # `values` is NOT mutated here: the caller iterates the same dict afterwards to
    # persist each key into config.yaml, and removing one would save a setting that
    # applied but never reached the file.
    for key, env_name in _IMAGING_SETTING_ENV.items():
        if key not in values:
            continue
        os.environ[env_name] = str(values[key])
        section = cfg.raw.get("imaging")
        if not isinstance(section, dict):
            section = {}
            cfg.raw["imaging"] = section
        section[fields[key]] = values[key]
    # F119: one loop per section instead of a hard-coded `cfg.vlm`. `features:` (the
    # F113 quality cascade) is a second dataclass section and behaves identically —
    # replace the fields on the running config, then mirror them into cfg.raw so a later
    # save writes the same values the form is showing.
    touched_vlm = False
    for name in _SETTING_SECTIONS:
        picked = {k: v for k, v in values.items() if k.startswith(f"{name}.")}
        if not picked:
            continue
        touched_vlm = touched_vlm or name == "vlm"
        # The values were type-checked one by one against _SETTINGS_SPEC, which mypy
        # cannot follow through a dict[str, object] — the cast says so rather than
        # widening the spec into something the validator would have to trust.
        changed: Any = {fields[key]: value for key, value in picked.items()}
        setattr(cfg, name, dataclasses.replace(getattr(cfg, name), **changed))
        section = cfg.raw.get(name)
        if not isinstance(section, dict):  # absent, or present and left empty
            section = {}
            cfg.raw[name] = section
        for key, value in picked.items():
            section[fields[key]] = value
    if touched_vlm:
        # F102: `naming.vlm_enabled`/`classify_vlm_model` are the effective per-run
        # toggle the junk stage reads, and load_config holds them equal to the `vlm:`
        # section. A write that skipped this would be a setting that saved and did not
        # apply.
        cfg.naming = dataclasses.replace(cfg.naming, vlm_enabled=cfg.vlm.enabled,
                                         classify_vlm_model=cfg.vlm.model)


def _summary_dest(cfg: Config, dest: str | None) -> Path | None:
    """The destination root the pre-apply summary must look into (F104).

    An empty destination means the in-place layout, whose root `plan_and_sort` takes
    from the single configured source (F28) — resolved the same way here, and None
    when that rule does not apply, so the dialog can say "the numbers about the
    destination are unknown" instead of inventing them.
    """
    if dest:
        return Path(dest)
    if len(cfg.sources) == 1:
        return Path(cfg.sources[0])
    return None


def _run_sort(db_path: Path, cfg: Config, dest: str | None, mode: str,
             state: _SortState, cache: PlanCache, by: str = "city") -> None:
    """The body of the `POST /api/sort` background thread: its own sqlite connection
    (not transferable between threads, like `_run_pipeline`). Calls the ready
    `sorter.plan_and_sort(..., apply=True)` — the moves/move_batches journal, blake3
    verification and name-conflict resolution are the engine, here only
    progress/status and rebuilding PlanCache after a successful apply.

    `plan_and_sort` may raise `ValueError` (e.g. in-place with ≠1 source in
    `cfg.sources`) — caught and stored in the state as an error, the thread does not
    crash and the server stays alive.

    F97: `should_cancel` is the state's own flag, so `POST /api/sort/cancel` stops the
    copying between files. A cancelled run is NOT an error — it returns a result like
    any other, with `cancelled` set and `moved` telling how far it got.

    F192: `by` is the criterion of the layout — the argument `plan_and_sort` has always
    taken and this function used to hard-code to "city". Nothing else changes with it:
    the same engine, the same journal, the same undo.
    """
    conn = _connect(db_path)
    error: str | None = None
    result: dict | None = None
    try:
        dest_path = Path(dest) if dest else None
        try:
            report = plan_and_sort(cfg, conn, by, dest_path, apply=True,
                                   copy=(mode == "copy"), progress=state.set_progress,
                                   should_cancel=state.cancel_requested)
        except ValueError as exc:
            error = str(exc)
        else:
            result = {
                "moved": report.moved,
                "failed": report.failed,
                "skipped_in_place": report.skipped_in_place,
                "skipped_already_copied": report.skipped_already_copied,
                "cancelled": report.cancelled,
                "total": len(report.plan),
                "dirs": report.dirs,
                "dest": str(report.dest),
                "in_place": report.in_place,
                "mode": mode,
                "by": by,
            }
            # F45: rebuild is only an update of the cities-tree preview cache, the
            # apply already happened (files laid out, the moves journal written) —
            # a rebuild failure is NOT a layout error, only a soft signal for the UI.
            try:
                cache.rebuild(cfg, conn)
            except Exception:  # noqa: BLE001
                _log.exception("sorta ui: план не обновлён после apply раскладки")
                result["preview_stale"] = True
    finally:
        conn.close()
        state.finish(error, result)
