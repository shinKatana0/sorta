"""F5: sorting by moving files.

Contract: reads files/places/manual_places/faces/face_clusters/events/event_files/
media_class/manual_overrides, writes to move_batches/moves and to the FS. The only exception:
after a successful move, files.path is updated so the index stays valid; undo
restores the old value.

Invariants (must not be broken):
  - without apply=True, no FS operation except writing the CSV plan next to the DB;
  - a moves row (status='planned') is committed BEFORE the file is moved; after
    the move and verification (dst exists, size matches) — 'done';
  - an existing dst is never overwritten: suffixes _1, _2, ...;
  - cross-device (os.rename -> OSError): copy -> blake3 verify -> delete src;
    on a hash mismatch the copy is deleted, move.status='failed', the process
    continues;
  - undo: reverse journal order, dst -> src, status='undone'; a missing dst is
    logged, the rollback continues.
  - copy mode (C16, --copy): src is NOT deleted and files.path does NOT change;
    move_batches.operation='copy' lets undo distinguish it (deletes dst instead of dst -> src).

F77 (manual_overrides, written by the web app): a correction the user made by eye
outranks every automatic rule here. action='exclude' — the file drops out of the plan
entirely (it is not moved anywhere, it stays exactly where it lies), counted in
SortReport.manual_excluded separately from the --exclude directories; action='reassign'
— the layout target is the folder the user picked, ahead of the dedup choice, the junk
verdict, not_personal and geo. The reassign target comes from the DB, i.e. from
outside, so it is validated against the sort root before a path is built from it (see
_manual_target_parts) — an invalid target is ignored with a warning, never followed.
F103 adds a third action, 'photo': "this is an ordinary photo, the classifier is
wrong" — it neutralizes the junk/document/product verdict for ROUTING only (the file
goes down the normal mode/date branch) and never touches media_class, so re-running the
junk tier cannot wipe the correction.

The low_date rule: any mode's layout includes the year (YYYY), so a file without
taken_at or with taken_at_confidence='low' (a date only from mtime — often the copy
time, not the capture time) goes to _Unsorted/low_date/. The exception is
event mode (F5.1): a file that fell into an event (auto or manual) takes the year
from events.started_at, not from its own date, so low-confidence/undated files of
manual events are laid out under <event_year>/<name>/, not low_date; low_date for
event mode remains only as a fallback in case of an unparsable started_at (should
not happen — the column is NOT NULL, ISO).

F78 splits that undated bucket in two. Measured on the live collection, 1057 of the
1059 undated files carried no camera trace at all (no camera_make/camera_model, no
GPS) and had numeric messenger-cache names — the bucket is not "my shots whose date
was lost", it is forwarded and downloaded pictures. So a file with any camera trace
(see _looks_like_a_camera_shot) stays in _Unsorted/low_date/ as before, and one
without goes to _Unsorted/downloaded/ with its own reason code, so the two cases are
distinguishable in the CSV. Only the branch where the year could not be determined is
affected; the order of the checks above it (dedup_delete -> not_personal -> document
-> product -> junk) is untouched, so a screenshot or a document never reaches here.

F83 does the same for the one verdict that cannot be trusted on those same files: a
`meme` on a file with no camera trace is routed as `photo` (see
_is_indistinguishable_meme), because CLIP decides meme-vs-photo there on content alone
and errs both ways. It is a routing change, not a classification one — the verdict in
media_class stays as it is — and it deliberately leaves `document` and `screenshot`
alone: a scan has no camera EXIF either, and those two folders are the ones the user
reviews by hand.

F86 does the same for city mode: a file whose country resolved but whose city no
provider knows goes to <Country>/<year>/ (reason `country_only`) instead of
_Unsorted/no_place/. Only a file without a country at all (place_confidence='unknown')
still lands in no_place.

F85c (manual_places, written by the web app): a place the user assigned to a whole event
or a whole source folder. It is read here and NOT in geo, because `places` has a single
writer and is recomputed from scratch on every geo run — a manual place stored there
would live until the next run. The row replaces the automatic place as a WHOLE (country,
city and district together, never a mix of the two sources) and the file is reported with
place_confidence='manual', so the CSV, the HTML report and the web app all tell a place
the user chose from one the program inferred. Everything above it in _target_parts still
outranks it: a file marked "leave alone" or reassigned by hand (F77), a to_delete
duplicate, a document or a junk verdict does not go to the assigned city — those are
decisions about what the file IS, and the assignment only says where it was taken.
One deliberate limit: `--where country=/city=` still selects on the AUTOMATIC place. It
is a SQL filter over `places`, applied before the manual row replaces anything, and a
selection language that answered about hand-assigned places would have to run twice.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import re
import shutil
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Callable, Sequence
from urllib.parse import quote

from . import i18n, imaging
from .config import Config
from .dedup import near_duplicate_groups
from .geodata import GeoResolver
from .hashing import file_hash
from .indexer import excludes_path, load_excludes
from .search import TextEncoder, search_text

_log = logging.getLogger(__name__)


def _report_dir(cfg: Config) -> Path:
    """Directory for sort plan reports (CSV/HTML/thumbs). `cfg.sort.report_dir` if
    set; otherwise `report_output/` next to the DB (F56). Isolates one-off reports
    with real place names/paths from the DB/repo directory and keeps them gitignored
    (`report_output/`). The directory is created on access."""
    d = getattr(cfg.sort, "report_dir", None)
    base = Path(d) if d else Path(cfg.database).resolve().parent / "report_output"
    base.mkdir(parents=True, exist_ok=True)
    return base


MODES = ("city", "person", "event")
_MULTI_PERSON = ("primary", "shared_folder")

_CSV_COLUMNS = [
    "path", "taken_at", "taken_at_confidence", "country", "city",
    "place_confidence", "persons", "event", "junk_verdict", "junk_source",
    "target", "reason",
]

# The root of the merged_into chain for each cluster (the effective cluster, like
# faces.resolve_root but in SQL) + people labels on files. Clusters that cannot be
# reached from a root (a broken cyclic chain) do not enter _roots and are simply
# ignored. casefold() — a UDF, see _sql_casefold: a case-insensitive comparison that
# works for Cyrillic too (SQLite NOCASE is ASCII-only).
_CTE = """WITH RECURSIVE _roots(id, root) AS (
    SELECT id, id FROM face_clusters WHERE merged_into IS NULL
    UNION ALL
    SELECT fc.id, r.root FROM face_clusters fc JOIN _roots r ON fc.merged_into = r.id
), _person_files(file_id, label, bbox) AS (
    SELECT fa.file_id, cl.label, fa.bbox
    FROM faces fa
    JOIN _roots r ON fa.cluster_id = r.id
    JOIN face_clusters cl ON cl.id = r.root
    WHERE cl.label IS NOT NULL AND fa.bbox != '[]'
)
"""

def _layout_excluded_dirs(cfg: Config) -> list[Path]:
    """F82: the `skip_layout` section of the exclusion file, as absolute directories.

    Entries there are relative to a source root (the same key the "do not scan" section
    uses), so they are resolved against every source. Reading only — the file belongs to
    the indexer and to the web app; `sort` never writes it.
    """
    excludes = load_excludes(excludes_path(cfg))
    dirs: list[Path] = []
    for src in cfg.sources:
        root = Path(src).expanduser().resolve()
        for rel in sorted(excludes.layout_for_root(src)):
            dirs.append(root.joinpath(*rel.split("/")))
    return dirs


def _resolve_excludes(cfg: Config, exclude: Sequence[str] | None) -> list[Path]:
    """Exclude directories from --exclude (repeatable) + config sort.exclude_dirs
    + the `skip_layout` folders ticked in the web app (F82).

    All three sources are COMBINED — the tree in the UI adds to what config.yaml and
    the command line say, it does not replace either. Paths are coerced to absolute
    resolved form for comparison by directory boundary (see _is_excluded).
    """
    dirs = list(exclude or [])
    dirs += list(cfg.sort.exclude_dirs)
    return [Path(d).resolve() for d in dirs] + _layout_excluded_dirs(cfg)


def _is_excluded(path: Path, excludes: list[Path]) -> bool:
    """True if path is inside any of excludes (including excludes itself).

    Path.is_relative_to compares path parts via each platform's flavour — on
    Windows this is a case-insensitive comparison (ntpath casefold), so no separate
    case normalization is needed.
    """
    return any(path.is_relative_to(ex) for ex in excludes)


_WHERE_FIELDS = ("city", "country", "event", "person", "year")
_YEAR_OPS = ("=", "!=", ">=", "<=", ">", "<")
_EXPR_RE = re.compile(r"^\s*([A-Za-z_]+)\s*(>=|<=|!=|=|>|<)\s*(.+?)\s*$")
_STR_CONDS = {
    "country": "casefold(p.country) = casefold(?)",
    "city": "casefold(p.city) = casefold(?)",
    "person": ("f.id IN (SELECT file_id FROM _person_files "
               "WHERE casefold(label) = casefold(?))"),
    "event": ("f.id IN (SELECT ef.file_id FROM event_files ef "
              "JOIN events e ON e.id = ef.event_id "
              "WHERE casefold(e.name) = casefold(?))"),
}


def _sql_casefold(s: str | None) -> str | None:
    return s.casefold() if isinstance(s, str) else s


def parse_where(exprs: Sequence[str], lang: i18n.Lang = "en",
                resolver: GeoResolver | None = None) -> tuple[str, list[str | int]]:
    """Parse --where conditions into a SQL condition (joined by AND) + parameters.

    The condition assumes the aliases f (files), p (places) and the CTE
    _person_files — the query in plan_and_sort injects them. String fields are
    compared case-insensitively (the casefold UDF); for year all operators are allowed.

    F46: country/city with a value in the config language (lang) are resolved via
    resolver (GeoResolver.country_cc_by_name/city_ids_by_name) into a canonical
    ISO cc / list of geonameid — so config-language folders («Россия»/«Москва») and
    --where in the same language stay in sync. resolver=None (as before, without it)
    or a non-resolving value — a fallback to the previous string comparison (canonical
    country=RU/city=Moscow keeps working). A city with several geonameid (same-named
    cities) matches any of them.
    """
    conds: list[str] = []
    params: list[str | int] = []
    for expr in exprs:
        m = _EXPR_RE.match(expr)
        if not m:
            raise ValueError(
                f"--where: не разобрано условие {expr!r}; формат <поле><оп><значение>, "
                f"поля: {', '.join(_WHERE_FIELDS)}")
        fld, op, value = m.group(1).lower(), m.group(2), m.group(3)
        if fld == "year":
            try:
                params.append(int(value))
            except ValueError:
                raise ValueError(f"--where: year сравнивается с целым числом, "
                                 f"получено {value!r}") from None
            conds.append(f"CAST(substr(f.taken_at, 1, 4) AS INTEGER) {op} ?")
        elif fld in _STR_CONDS:
            if op != "=":
                raise ValueError(
                    f"--where: для поля {fld} допустим только оператор '='; "
                    f"операторы {' '.join(_YEAR_OPS)} — только для year")
            if fld == "country":
                cc = resolver.country_cc_by_name(value, lang) if resolver else None
                conds.append(_STR_CONDS["country"])
                params.append(cc if cc else value)
            elif fld == "city":
                ids = resolver.city_ids_by_name(value, lang) if resolver else []
                if ids:
                    # OR with a string match: does not lose files where
                    # city_geonameid is not set (online G2b — only the text
                    # p.city from Nominatim), even if value resolves in the
                    # bundled data.
                    qmarks = ",".join("?" * len(ids))
                    conds.append(f"(p.city_geonameid IN ({qmarks}) OR "
                                 f"{_STR_CONDS['city']})")
                    params.extend(ids)
                    params.append(value)
                else:
                    conds.append(_STR_CONDS["city"])
                    params.append(value)
            else:
                conds.append(_STR_CONDS[fld])
                params.append(value)
        else:
            raise ValueError(f"--where: неизвестное поле {fld!r}; "
                             f"допустимы: {', '.join(_WHERE_FIELDS)}")
    return (" AND ".join(conds) if conds else "1"), params


# --- Layout ------------------------------------------------------------------

_FORBIDDEN_CHARS = set('<>:"/\\|?*')
_RESERVED_NAMES = {"CON", "PRN", "AUX", "NUL",
                   *(f"COM{i}" for i in range(1, 10)),
                   *(f"LPT{i}" for i in range(1, 10))}


def _sanitize(name: str) -> str:
    """A directory name safe for Windows/NTFS."""
    s = "".join("_" if c in _FORBIDDEN_CHARS or ord(c) < 32 else c for c in name)
    s = s.strip().rstrip(" .")
    if not s:
        return "_"
    if s.split(".")[0].upper() in _RESERVED_NAMES:
        return "_" + s
    return s


def _manual_target_parts(target: str | None, src: str) -> list[str] | None:
    """F77: `manual_overrides.target` -> layout segments under the sort root, or None.

    The value is written by the web app, i.e. it reaches the sorter from OUTSIDE, and
    it is the one input of this feature that becomes a WRITE path — so it is treated as
    untrusted. Accepted: a relative POSIX path of plain segments. Rejected (None + a
    warning; the caller then lays the file out automatically, rather than writing
    outside the root): an empty value, a backslash or a colon (`..\\..\\x`, `C:/win`,
    UNC), a leading `/`, and any `..` segment. What survives still goes through
    _sanitize, like every other folder name in the layout.
    """
    if not isinstance(target, str):
        return None
    raw = target.strip()
    rejected = (not raw or "\\" in raw or ":" in raw or raw.startswith("/")
                or ".." in [seg.strip() for seg in raw.split("/")])
    if rejected:
        _log.warning("sort: ручная правка проигнорирована — target выходит за корень "
                     "раскладки или не является относительным путём: %r (%s)", target, src)
        return None
    parts = [_sanitize(seg) for seg in raw.split("/") if seg.strip() not in ("", ".")]
    if not parts:
        _log.warning("sort: ручная правка проигнорирована — пустой target: %r (%s)",
                     target, src)
        return None
    return parts


def _looks_like_a_camera_shot(row: sqlite3.Row) -> bool:
    """F78: does the file carry ANY trace of having been shot by a camera?

    Any one of camera_make / camera_model / gps_lat is enough — messengers strip most
    of the EXIF from a forwarded photo, so demanding several signals would push real
    (if metadata-poor) shots into the downloaded folder. The mirror image of
    junk._is_real_photo, minus the face signal: here the question is only "was this
    taken by a device", not "is this a memory".

    gps_lat is compared to None on purpose — 0.0 is a valid latitude (see the
    Null Island guard in geo.py), so a truthiness test would drop it.
    """
    return bool(row["camera_make"] or row["camera_model"]
                or row["gps_lat"] is not None)


def _is_indistinguishable_meme(row: sqlite3.Row) -> bool:
    """F83: a `meme` verdict on a file that carries nothing to judge it by.

    Measured on the live collection: 3437 files have no camera trace at all
    (camera_make, camera_model and gps_lat all NULL) — forwarded and downloaded
    pictures whose metadata the messenger stripped. For those, `photo` vs `meme` is
    decided by CLIP on content alone, and it is wrong in BOTH directions, so files of
    one and the same origin end up in different folders for no reason the user can see.

    This is NOT an accuracy improvement. There is nothing left in these files to tell
    the two apart; the rule only replaces two coin flips with one predictable outcome.
    The real fix for them is a manual correction by eye (F77). Do not read this as a
    classifier and do not try to "improve" it.

    Only `meme` collapses. `document` and `screenshot` keep their own folders: a
    scanner writes no camera EXIF either, so a scanned document has no trace either,
    and those are exactly the categories the user opens by hand — documents for
    privacy, screenshots to delete. Files WITH a camera trace are untouched: among
    20743 camera shots there were 0 false `meme`.
    """
    return row["junk_verdict"] == "meme" and not _looks_like_a_camera_shot(row)


def _undated_parts(row: sqlite3.Row, lang: i18n.Lang) -> tuple[list[str], str]:
    """F78: the target for a file whose year could not be determined.

    A real shot with an unread date keeps the historical _Unsorted/low_date/; anything
    without a camera trace — the overwhelming majority of this bucket — goes to
    _Unsorted/downloaded/ under its own reason, so the report tells the two apart.
    """
    if _looks_like_a_camera_shot(row):
        return [i18n.folder("unsorted", lang), i18n.folder("low_date", lang)], "low_date"
    return [i18n.folder("unsorted", lang), i18n.folder("downloaded", lang)], "downloaded"


def _city_display_name(city: str | None, city_gid: int | None,
                       lang: i18n.Lang, resolver: GeoResolver) -> str | None:
    """G3: the city name to lay out and to show, in `lang`.

    `places` holds the English anchor plus `city_geonameid` (geo.py writes exactly
    that and calls localizing sort's job) — so the translation happens HERE, once per
    row, and switching the folder language changes every city name without a single
    geo query or a write into `places`.

    Without a geonameid the DB text is the only name there is and is used as-is: a
    landmark/visual place, an online provider's answer (already in the config
    language) and a place the user assigned by hand carry no id to translate by.

    `GeoResolver.name` ends its fallback chain with the geonameid itself — a folder
    called `498817` explains nothing to anyone, so that answer is refused in favour of
    the anchor: an English city name is an honest answer, a number is not.
    """
    if city_gid is None:
        return city
    name = resolver.name(city_gid, lang)
    if name == str(city_gid) and city:
        return city
    return name


def _year_of(taken_at: str | None, confidence: str | None) -> str | None:
    if not taken_at or len(taken_at) < 4 or not taken_at[:4].isdigit():
        return None
    if confidence == "low":
        return None
    return taken_at[:4]


def _target_parts(mode: str, strategy: str, row: sqlite3.Row,
                  persons: list[tuple[str, float]],
                  event: tuple[str, str | None] | None,
                  lang: i18n.Lang, resolver: GeoResolver,
                  drop_unlocalized_district: bool = True) -> tuple[list[str], str]:
    """The relative target directory (a list of segments) + a reason for the CSV.

    Path segments (service folders, country) are localized via i18n.folder/
    i18n.country by lang (F27); reason — a stable English code, not localized.
    City/district (G3) — via resolver.name(geonameid, lang) if the geonameid is
    known (G2); otherwise (landmark/visual without geonameid) — the original text
    row["city"] as-is. See _city_display_name: neither name may end up being the
    geonameid itself.
    """
    if row["manual_action"] == "reassign":
        # F77: the user dragged this frame into a folder by hand — that outranks EVERY
        # rule below (dedup choice, not_personal, junk/document/product verdict, geo,
        # event/person layout): they looked at the frame, the classifier did not.
        # A target that does not validate falls through to the automatic layout.
        manual_parts = _manual_target_parts(row["manual_target"], row["path"])
        if manual_parts is not None:
            return manual_parts, "manual_reassign"
    if row["dedup_action"] == "to_delete":
        # U3b: an explicit user decision from the web app (sorta ui) — the highest
        # priority of all (city/junk/document/not_personal), the file goes to the
        # to_delete folder («_удалить» in a ru layout) regardless of the sort mode.
        return [i18n.folder("to_delete", lang)], "dedup_delete"
    if row["not_personal"]:
        # F17: a downloaded movie/series (release name, marked at indexing) — not
        # personal media, past the city/date/people layout, into a separate folder.
        return [i18n.folder("unsorted", lang), i18n.folder("not_personal", lang)], "not_personal"
    verdict = row["junk_verdict"]
    if row["manual_action"] == "photo":
        # F103: the user opened the "Not personal photos" view, looked at the frame and
        # said it is an ordinary photo. Only the ROUTE changes here — media_class keeps
        # the model's verdict, because that verdict is a measurement and a correction by
        # eye is a separate layer on top of it. Overwriting the measurement would mean a
        # re-run of the junk tier silently wipes the correction, with nothing to tell the
        # user why their decisions disappeared. Dropping the verdict for routing purposes
        # lets the file fall through to the ordinary mode/date layout below — exactly
        # where a `photo` verdict would have taken it.
        verdict = None
    if verdict == "document":
        # F15: a photographed document — a separate review category (not junk), its
        # own top-level folder regardless of the sort mode.
        return [i18n.folder("documents", lang)], "document"
    if verdict == "product":
        # F37-B (deep VLM tier): an item for sale — its own review folder («_Товары»),
        # not junk and not a memory. Only with vlm_enabled (the fast tier gives no product).
        return [i18n.folder("products", lang)], "product"
    if verdict is not None and verdict != "photo" and not _is_indistinguishable_meme(row):
        # F83: a meme with no camera trace falls through here and is routed like a
        # photo — down the normal branch below, i.e. into the mode/date layout or, with
        # no reliable date, into the downloaded folder of F78. media_class keeps the
        # verdict as it is (reports and the UI still need it), only the route changes.
        return [i18n.folder("unsorted", lang), i18n.folder("junk", lang),
               _sanitize(verdict)], "junk"
    if mode == "event":
        if event is None:
            # F30: the file did not fall into an event (a small group < min_event_size
            # or no event) → lay it out by date Year/month, not into a flat service
            # folder.
            year = _year_of(row["taken_at"], row["taken_at_confidence"])
            if year is None:
                return _undated_parts(row, lang)
            taken_at = row["taken_at"] or ""
            month = taken_at[5:7] if len(taken_at) >= 7 and taken_at[5:7].isdigit() else None
            return ([year, month] if month else [year]), "no_event"
        event_name, event_year = event
        year = event_year or _year_of(row["taken_at"], row["taken_at_confidence"])
        if year is None:
            return _undated_parts(row, lang)
        return [year, _sanitize(event_name)], "event"
    year = _year_of(row["taken_at"], row["taken_at_confidence"])
    if year is None:
        return _undated_parts(row, lang)
    if mode == "city":
        country_known = bool(row["country"] or row["country_name"])
        if (row["place_confidence"] or "unknown") == "unknown" or (
                row["city"] is None and not country_known):
            return [i18n.folder("unsorted", lang), i18n.folder("no_place", lang)], "no_place"
        # online (G6): the full country name from Nominatim is already in the config
        # language; offline — localize the ISO cc via the curated dict i18n.country
        country_name = row["country_name"] or (
            i18n.country(row["country"], lang) if row["country"] else "Unknown")
        if row["city"] is None:
            # F86: the country is resolved, only the city is missing (no provider knows a
            # settlement for these coordinates — mid-ocean, a desert road). The file goes
            # to the country level: hiding it in _Unsorted/no_place would throw away the
            # one place signal we do have. Guessing a city by the nearest one is not an
            # option here — that is the F75 misplacement.
            return [_sanitize(country_name), year], "country_only"
        city_name = _city_display_name(row["city"], row["city_geonameid"], lang, resolver)
        assert city_name is not None  # row["city"] is not None here (guarded above)
        parts = [_sanitize(country_name), _sanitize(city_name), year]
        district_gid = row["district_geonameid"]
        if district_gid is not None:
            # F49: a foreign transliterated district (no localized name in
            # names.tsv) is dropped — only Country/City/Year. RU and localized
            # foreign districts (named «Убуд»/«Кута» in the base) stay.
            if not drop_unlocalized_district or resolver.has_localized_name(district_gid, lang):
                district_name = resolver.name(district_gid, lang)
                # G3: the same refusal as in _city_display_name — a district the
                # bundled base does not know resolves to its own geonameid, and there
                # is no anchor text to fall back to here (an online district comes
                # WITHOUT an id, see below), so the segment is simply left out.
                if district_name != str(district_gid):
                    parts.append(_sanitize(district_name))
        elif row["district_name"]:
            # G2b online: the district as a name from Nominatim (no geonameid)
            parts.append(_sanitize(row["district_name"]))
        return parts, "city"
    # person
    if not persons:
        return [i18n.folder("unsorted", lang), i18n.folder("no_faces", lang)], "no_faces"
    if len(persons) == 1:
        return [_sanitize(persons[0][0]), year], "person"
    if strategy == "shared_folder":
        return [i18n.folder("shared", lang), year], "person_shared"
    # primary: persons are sorted by descending bbox area
    return [_sanitize(persons[0][0]), year], "person_primary"


def _load_persons(conn: sqlite3.Connection) -> dict[int, list[tuple[str, float]]]:
    """file_id -> [(label, max bbox area)], by descending area."""
    acc: dict[int, dict[str, float]] = {}
    for r in conn.execute(_CTE + "SELECT file_id, label, bbox FROM _person_files"):
        try:
            x1, y1, x2, y2 = json.loads(r["bbox"])
            area = abs((x2 - x1) * (y2 - y1))
        except (ValueError, TypeError):
            area = 0.0
        d = acc.setdefault(r["file_id"], {})
        d[r["label"]] = max(d.get(r["label"], 0.0), area)
    return {fid: sorted(d.items(), key=lambda kv: -kv[1]) for fid, d in acc.items()}


def _load_events(conn: sqlite3.Connection) -> dict[int, tuple[str, str | None]]:
    """file_id -> (event name, event year); with several — the earliest by started_at.

    Year — the first 4 chars of `events.started_at` (ISO). The column is NOT NULL,
    but in case of an unparsable value the year is None — the caller falls back to
    the file's date (see _target_parts).
    """
    out: dict[int, tuple[str, str | None]] = {}
    for r in conn.execute(
        """SELECT ef.file_id, e.name, e.started_at FROM event_files ef
           JOIN events e ON e.id = ef.event_id ORDER BY e.started_at"""):
        started = r["started_at"]
        year = started[:4] if started and started[:4].isdigit() else None
        out.setdefault(r["file_id"], (r["name"], year))
    return out


# --- F14: near-duplicates (--dedupe) -----------------------------------------

def _quality_key(file_id: int, size: int | None,
                 dims: dict[int, tuple[int, int]]) -> tuple[int, int, int]:
    """The "quality" sort key: -(width*height), -size, id (determinism)."""
    w, h = dims.get(file_id, (0, 0))
    return -(w * h), -(size or 0), file_id


def _select_best(members: list[sqlite3.Row],
                 dims: dict[int, tuple[int, int]]) -> sqlite3.Row:
    return sorted(members, key=lambda r: _quality_key(r["id"], r["size"], dims))[0]


def _resolve_near_dup_roles(
    conn: sqlite3.Connection, cfg: Config, selected_ids: set[int],
) -> tuple[dict[int, int], dict[int, int]]:
    """file_id -> group index (1-based), separately for the best and the other group members.

    Groups from near_duplicate_groups are trimmed to the files of the current plan
    selection (--where may have excluded part of the group); groups left with one
    file (or none) after trimming are not meaningful for dedup and are skipped.
    width/height are read from files (near_duplicate_groups does not return them).
    """
    groups = near_duplicate_groups(conn, cfg.index.phash_max_distance)
    trimmed: list[list[sqlite3.Row]] = []
    candidate_ids: set[int] = set()
    for group in groups:
        members = [r for r in group if r["id"] in selected_ids]
        if len(members) < 2:
            continue
        trimmed.append(members)
        candidate_ids.update(r["id"] for r in members)

    dims: dict[int, tuple[int, int]] = {}
    if candidate_ids:
        qmarks = ",".join("?" * len(candidate_ids))
        for r in conn.execute(
            f"SELECT id, width, height FROM files WHERE id IN ({qmarks})",
            tuple(candidate_ids)):
            dims[r["id"]] = (r["width"] or 0, r["height"] or 0)

    best_of: dict[int, int] = {}
    worse_of: dict[int, int] = {}
    for gi, members in enumerate(trimmed, 1):
        best = _select_best(members, dims)
        best_of[best["id"]] = gi
        for r in members:
            if r["id"] != best["id"]:
                worse_of[r["id"]] = gi
    return best_of, worse_of


# --- Transfer ---------------------------------------------------------------

_LONG_PATH_PREFIX = "\\" * 2 + "?" + "\\"   # \\?\ — written this way to survive escaping


def _fs(path: Path) -> Path:
    """F97: the form a path must take at the boundary of a FILESYSTEM call on Windows.

    Windows resolves an ordinary path against MAX_PATH (260 characters); the `\\\\?\\`
    prefix lifts that limit. Without it a destination root plus a country/city folder
    plus the original file name goes past 260 easily, and such a file lands silently
    in `failed` — measured on the live collection, not hypothetical.

    The prefix also switches OFF Windows' own path normalization, so what is handed to
    it must ALREADY be absolute, with backslash separators and without `.`/`..`
    segments — hence os.path.abspath (it normalizes too) before the prefix is glued
    on. A UNC path takes its own form: `\\\\server\\share` -> `\\\\?\\UNC\\server\\share`.

    Paths stored in the DB (moves.src/dst, files.path) NEVER carry the prefix — the
    project's convention is plain absolute paths and this function exists only at the
    call boundary. On non-Windows it returns the path unchanged: the branching lives
    here, not in every caller.

    The per-component limit of 255 characters survives the prefix (verified: WinError
    123 on a 300-character folder name). Nothing here defends against it, because
    every component in the layout is either a short city/year folder or a file name
    that already exists on a filesystem that enforces the same limit.
    """
    if os.name != "nt":
        return path
    text = str(path)
    if text.startswith(_LONG_PATH_PREFIX):
        return path
    full = os.path.abspath(text)
    if full.startswith(_LONG_PATH_PREFIX[:2]):
        return Path(_LONG_PATH_PREFIX + "UNC" + full[1:])
    return Path(_LONG_PATH_PREFIX + full)


class TransferError(RuntimeError):
    """Transferring a single file failed; the caller marks the move failed."""


def _copy_and_verify(src: Path, dst: Path, expected_hash: str) -> None:
    """copy2 src -> dst, blake3 verify; on failure dst is deleted, TransferError."""
    try:
        shutil.copy2(_fs(src), _fs(dst))
    except OSError as exc:
        _fs(dst).unlink(missing_ok=True)
        raise TransferError(f"копирование не удалось: {src} -> {dst}: {exc}") from None
    if file_hash(_fs(dst))[0] != expected_hash:
        _fs(dst).unlink(missing_ok=True)
        raise TransferError(f"хэш копии не совпал, копия удалена: {src} -> {dst}")


def _transfer(src: Path, dst: Path, src_hash: str | None = None,
             copy: bool = False, link: bool = False) -> None:
    """Move (copy=False), copy (copy=True) or link (link=True) src -> dst.

    dst is not overwritten. move: os.rename; on OSError (different device/volume)
    copy -> blake3 verify -> delete src. copy (C16): always copy2 -> blake3 verify,
    src is NOT touched (neither on success nor on failure). link (F34): os.link
    (a hardlink to the same data); on OSError (different volumes, FAT/exFAT,
    cross-disk) — an auto-fallback to the copy path (the same as copy=True), the
    album is materialized anyway. After any path — a check: dst exists and the size
    matches.

    F97: every FS call here goes through `_fs` — the long-path form on Windows. The
    plain `src`/`dst` stay in the log/exception texts (a `\\\\?\\` prefix in a message
    to the user means nothing) and in whatever the caller writes to the DB.
    """
    size = _fs(src).stat().st_size
    _fs(dst.parent).mkdir(parents=True, exist_ok=True)
    if _fs(dst).exists():
        raise TransferError(f"dst уже существует, перезапись запрещена: {dst}")
    if link:
        try:
            os.link(_fs(src), _fs(dst))
        except OSError as exc:
            _log.warning("album: hardlink недоступен (%s), фолбэк на copy: %s -> %s",
                        exc, src, dst)
            _copy_and_verify(src, dst, src_hash or file_hash(_fs(src))[0])
    elif copy:
        _copy_and_verify(src, dst, src_hash or file_hash(_fs(src))[0])
    else:
        try:
            os.rename(_fs(src), _fs(dst))
        except OSError:
            _copy_and_verify(src, dst, src_hash or file_hash(_fs(src))[0])
            os.remove(_fs(src))
    if not _fs(dst).exists() or _fs(dst).stat().st_size != size:
        raise TransferError(f"проверка после перемещения не прошла: {dst}")


def _is_the_same_file(dst: Path, src: Path, src_hash: str | None,
                      src_algo: str | None) -> bool:
    """F97: is the file already lying at `dst` the very one we would put there?

    Size first (a stat, free), the blake3 of `dst` only when the size matches —
    hashing every same-named file in the target would cost a full read per candidate.
    The SOURCE hash comes from the index (`files.hash`, already computed at index
    time) rather than being recomputed: a resumed apply must not pay for re-reading
    the sources it has already read once.

    Without a hash in the index, or under a different algorithm, or on any OSError,
    the answer is "no" — the file then takes the usual `_1` suffix. That is wasteful,
    never destructive; the opposite mistake would skip a file that was never copied.
    """
    if not src_hash:
        return False
    try:
        if _fs(dst).stat().st_size != _fs(src).stat().st_size:
            return False
        dst_hash, algo = file_hash(_fs(dst))
    except OSError:
        return False
    if src_algo and algo != src_algo:
        return False
    return dst_hash == src_hash


def _resolve_dst(target_dir: Path, src: Path, claimed: set[str],
                 src_hash: str | None = None,
                 src_algo: str | None = None) -> tuple[Path, bool, bool]:
    """dst without overwriting -> (dst, in_place, already_copied).

    Suffixes _1, _2 against the disk and the names other plan items have claimed.

    in_place=True — the file is already AT the target path (src == dst), an in-place
    layout; nothing to do.

    already_copied=True (F97) — `dst` is occupied by a file that is byte-for-byte our
    source (see _is_the_same_file), i.e. a previous apply into this same dest already
    copied it. Before F97 that case was indistinguishable from "another file happens
    to share this name" and got a `_1` suffix: measured on the live collection, a
    second `sort --apply` into the same dest re-copied 10 021 files and 140.9 GB of
    duplicates. `_1` for a DIFFERENT file with the same name is still correct and
    still happens — the two cases only had to stop being one.

    The decision is made while the plan is built, so a dry-run plan shows exactly the
    targets an apply would use.
    """
    dst = target_dir / src.name
    if os.path.normcase(str(dst)) == os.path.normcase(str(src)):
        return src, True, False
    n = 0
    cand = dst
    while True:
        key = os.path.normcase(str(cand))
        if key not in claimed:
            if not _fs(cand).exists():
                claimed.add(key)
                return cand, False, False
            if _is_the_same_file(cand, src, src_hash, src_algo):
                # claimed as well: a later file of the same name must not be handed
                # this path either, it goes on to _1 as it always did
                claimed.add(key)
                return cand, False, True
        n += 1
        cand = dst.with_name(f"{dst.stem}_{n}{dst.suffix}")


# --- Plan and apply ---------------------------------------------------------

@dataclass
class PlanItem:
    file_id: int
    src: Path
    dst: Path
    in_place: bool
    target_rel: str            # path relative to dest, POSIX separators
    reason: str                # city|country_only|person|person_primary|person_shared
    #                            | event|no_place|no_faces|no_event|junk|low_date|downloaded
    #                            | dedup_delete|manual_reassign
    #                            | manual_exclude (preview only, see keep_manual_excluded)
    taken_at: str | None
    taken_at_confidence: str | None
    country: str | None
    city: str | None
    place_confidence: str | None
    gps_lat: float | None      # F23: for the report's Geo column (places gives only city/country)
    gps_lon: float | None
    persons: list[str]         # labels by descending bbox area
    event: str | None
    junk_verdict: str | None
    junk_source: str | None
    db_hash: str | None
    db_algo: str | None
    near_dup_group: int | None = None   # F14: near-duplicate group index (1-based)
    near_dup_role: str | None = None    # kept | moved | deleted
    # F97: dst already holds a byte-for-byte copy of this file (a previous apply into
    # the same dest) — apply skips it instead of writing a `_1` twin. Distinct from
    # in_place, which is "src and dst are one and the same path".
    already_copied: bool = False


@dataclass
class SortReport:
    mode: str
    dest: Path
    csv_path: Path
    html_path: Path
    plan: list[PlanItem] = field(default_factory=list)
    dirs: int = 0
    batch_id: int | None = None
    moved: int = 0
    failed: int = 0
    skipped_in_place: int = 0
    # F97: deliberately NOT folded into skipped_in_place. "Skipped because source and
    # target are the same path" and "skipped because the copy is already there" are
    # different events, and one number for both turns diagnosing an interrupted run
    # back into guesswork — which is exactly how the 140.9 GB of duplicates went
    # unnoticed.
    skipped_already_copied: int = 0
    # F97: apply stopped on request (should_cancel) — the report says "cancelled,
    # N of M", never a bare "done".
    cancelled: bool = False
    deleted: int = 0   # F14: --delete-worse-dupes, permanently deleted worse near-dups
    excluded: int = 0  # F16: files skipped because of --exclude/sort.exclude_dirs
    in_place: bool = False  # F28: dest not set explicitly — layout inside the source root
    # F77: manual corrections from the web app — deliberately NOT folded into
    # `excluded` above: --exclude is "this directory is already sorted by hand", a
    # manual override is "leave this frame alone"; one report number for both would
    # hide which mechanism dropped a file.
    manual_excluded: int = 0    # action='exclude': files left where they lie
    manual_reassigned: int = 0  # action='reassign': files laid out into a chosen folder


@dataclass
class UndoStats:
    batch_id: int = 0
    undone: int = 0
    missing: int = 0
    failed: int = 0
    # F97: the rollback stopped on request (should_cancel). What was already undone
    # stays undone — the rest keeps its status and a repeated undo finishes the job.
    cancelled: bool = False
    # F97: files found in the result that are OURS by journal but NOT byte-for-byte
    # what we wrote (a copy interrupted mid-write). They are never deleted — the user
    # is told the path instead. Without this they stayed in the result silently,
    # looking like ordinary photos with truncated insides.
    stray: list[str] = field(default_factory=list)


_CSV_DEDUPE_COLUMNS = ["near_dup_group", "near_dup_role"]


def _write_plan_csv(csv_path: Path, plan: list[PlanItem]) -> None:
    """The CSV diagnosis — the approval document before --apply (utf-8-sig and ';' for Excel).

    The near_dup_group/near_dup_role columns are added only if the plan contains
    near-duplicates (--dedupe) — without it the CSV is no different from F5.
    """
    has_dedupe = any(it.near_dup_group is not None for it in plan)
    columns = _CSV_COLUMNS + _CSV_DEDUPE_COLUMNS if has_dedupe else _CSV_COLUMNS
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(columns)
        for it in plan:
            row = [
                str(it.src), it.taken_at or "", it.taken_at_confidence or "",
                it.country or "", it.city or "", it.place_confidence or "",
                ";".join(it.persons), it.event or "", it.junk_verdict or "",
                it.junk_source or "", it.target_rel, it.reason,
            ]
            if has_dedupe:
                row += [
                    str(it.near_dup_group) if it.near_dup_group is not None else "",
                    it.near_dup_role or "",
                ]
            w.writerow(row)


# --- HTML report -------------------------------------------------------------
# The precedent for sanitization/file:// links — faces.export_contact_sheet (modules
# do not import each other, so _file_uri is duplicated right here).

_THUMB_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"}
_THUMB_SIZE = (200, 200)


def _file_uri(path: str) -> str:
    try:
        return Path(path).as_uri()
    except ValueError:  # a POSIX path without a Windows drive
        return "file://" + quote(path)


_CONFIDENCE_LABEL = {"low": "низкая точность", "medium": "средняя точность"}


def _format_date_cell(item: PlanItem) -> str:
    """F23: the Date/time column — a human-readable date, low-confidence is marked."""
    if not item.taken_at:
        return "без даты"
    d = item.taken_at[:10]
    label = _CONFIDENCE_LABEL.get(item.taken_at_confidence or "")
    if label:
        d += f" ({label})"
    return d


def _format_geo_cell(item: PlanItem) -> str:
    """F23: the Geo column — country/city + place_confidence, coordinates (if any).

    Empty if there is neither a place nor coordinates (must not crash).
    """
    parts: list[str] = []
    place = "/".join(p for p in (item.country, item.city) if p)
    if place:
        if item.place_confidence:
            place += f" ({item.place_confidence})"
        parts.append(place)
    if item.gps_lat is not None and item.gps_lon is not None:
        parts.append(f"{item.gps_lat:.4f}, {item.gps_lon:.4f}")
    return " · ".join(parts)


def _format_people_event_cell(item: PlanItem) -> str:
    """F23: the People/Event column — do not lose the info from the former _diagnosis."""
    parts: list[str] = []
    if item.persons:
        parts.append(", ".join(item.persons))
    if item.event:
        parts.append(item.event)
    return " · ".join(parts)


def _format_category_cell(item: PlanItem) -> str:
    """F23: the Category column — reason, + junk/document verdict, + near-dup role."""
    parts = [item.reason]
    if item.junk_verdict and item.reason in ("junk", "document"):
        parts.append(item.junk_verdict)
    if item.near_dup_role:
        parts.append(_NEAR_DUP_ROLE_LABEL.get(item.near_dup_role, item.near_dup_role))
    return " · ".join(parts)


def _make_thumbnail(src: Path, dst: Path) -> bool:
    """Decode+resize src -> dst (JPEG, thumbs_dir). True on success.

    Decode — via the shared imaging layer (HEIC-lazy, draft downscale, error->None,
    F18); any failure (unrecognized format, corrupt file, no pillow-heif) — False,
    without crashing; the report row stays without a preview.
    """
    if src.suffix.lower() not in _THUMB_EXTS:
        return False
    img = imaging.decode_rgb(src, max_edge=_THUMB_SIZE[0])
    if img is None:
        return False
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst, "JPEG", quality=85)
        return True
    except Exception as exc:
        _log.warning("sort: миниатюра не создана для %s: %s", src, exc)
        return False


def _generate_thumbnails(plan: list[PlanItem], thumbs_dir: Path,
                         workers: int) -> set[int]:
    """Generate plan thumbnails in parallel; returns file_ids that succeeded.

    Decode — the heaviest report step (~288s serially on 2k photos); we spread it
    across a pool (Pillow releases the GIL in the C decode, like
    faces._prefetch_decode). One thumb per file_id; order does not matter.
    """
    def _one(item: PlanItem) -> tuple[int, bool]:
        thumb_file = thumbs_dir / f"{item.file_id}.jpg"
        return item.file_id, _make_thumbnail(item.src, thumb_file)

    ok: set[int] = set()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for fid, success in pool.map(_one, plan):
            if success:
                ok.add(fid)
    return ok


_NEAR_DUP_ROLE_LABEL = {"kept": "оставлен", "moved": "в дубли", "deleted": "удалён"}


def _render_near_dup_section(plan: list[PlanItem]) -> str:
    """F14: the "Near-duplicates" section — by group, who is kept and who is moved/deleted.

    Empty if the plan has no near-dup items (an ordinary run without --dedupe).
    """
    groups: dict[int, list[PlanItem]] = {}
    for item in plan:
        if item.near_dup_group is not None:
            groups.setdefault(item.near_dup_group, []).append(item)
    if not groups:
        return ""
    rows: list[str] = []
    for gi in sorted(groups):
        for item in sorted(groups[gi], key=lambda it: it.near_dup_role != "kept"):
            label = _NEAR_DUP_ROLE_LABEL.get(item.near_dup_role or "", item.near_dup_role or "")
            rows.append(
                f'<tr><td>{gi}</td>'
                f'<td><a href="{escape(_file_uri(str(item.src)))}">{escape(item.src.name)}</a></td>'
                f'<td>{escape(label)}</td>'
                f'<td>{escape(item.target_rel)}</td></tr>')
    return (
        f'<section><h2>Почти-дубликаты <small>({len(groups)} групп)</small></h2>\n'
        f'<table><thead><tr><th>Группа</th><th>Файл</th><th>Статус</th>'
        f'<th>Куда</th></tr></thead>\n<tbody>\n{"".join(rows)}\n</tbody></table></section>')


def _tree_sort_key(name: str) -> tuple[int, int | str]:
    """Sort key of a tree segment: a year (4-digit number) — ascending, everything
    else — alphabetically (casefold). A year is not mixed with strings at the same
    level (see _target_parts — a year is always its own level); the numeric branch is
    only there so we do not implicitly rely on the lexicographic order of 4-digit
    numbers matching the numeric one.
    """
    return (0, int(name)) if name.isdigit() else (1, name.casefold())


def _build_tree(plan: list[PlanItem]) -> dict:
    """A directory tree by target_rel segments: {"files": [...], "children": {...}}.

    A file goes into the node of its parent directory (the target_rel parent);
    intermediate path segments are container nodes without files of their own.
    """
    root: dict = {"files": [], "children": {}}
    for item in plan:
        node = root
        for part in Path(item.target_rel).parent.parts:
            node = node["children"].setdefault(part, {"files": [], "children": {}})
        node["files"].append(item)
    return root


_LEAF_COLUMNS = (
    ("Файл", "text"), ("Дата/время", "date"), ("Гео", "text"),
    ("Люди/Событие", "text"), ("Категория", "text"),
)


def _render_leaf_header() -> str:
    """F24: leaf headers are clickable — sort their own table (sortaSort in <script>).

    data-sort-type distinguishes the sort key: 'date' takes the cell's data-sort
    (ISO taken_at), 'text' — textContent. onclick passes the specific <th> (this),
    not a global list — so sorting only affects its own table.
    """
    ths = "".join(
        f'<th data-sort-type="{kind}" onclick="sortaSort(this)">{label}'
        f'<span class="sorta-sort-ind"></span></th>'
        for label, kind in _LEAF_COLUMNS
    )
    return f"<tr>{ths}</tr>"


def _render_file_rows(items: list[PlanItem], thumbs_dir: Path | None,
                      thumb_ok: set[int]) -> str:
    """A tree-leaf table: File (thumbnail F18 + file:// link) / Date·time / Geo /
    People·Event / Category (F23); headers are clickable to sort the rows of ONLY
    their own table (F24)."""
    rows: list[str] = []
    for item in items:
        img_tag = ""
        if thumbs_dir is not None and item.file_id in thumb_ok:
            thumb_file = thumbs_dir / f"{item.file_id}.jpg"
            img_tag = (f'<img src="{escape(f"{thumbs_dir.name}/{thumb_file.name}")}" '
                      f'loading="lazy" alt="">')
        # data-sort on the date cell — the full ISO taken_at (lexicographic = chronological);
        # empty if there is no date — sortaSort always pushes empty keys to the end.
        date_sort = escape(item.taken_at or "")
        rows.append(
            f'<tr><td>{img_tag}<a href="{escape(_file_uri(str(item.src)))}">'
            f'{escape(item.src.name)}</a></td>'
            f'<td data-sort="{date_sort}">{escape(_format_date_cell(item))}</td>'
            f'<td>{escape(_format_geo_cell(item))}</td>'
            f'<td>{escape(_format_people_event_cell(item))}</td>'
            f'<td>{escape(_format_category_cell(item))}</td></tr>')
    return (f'<table><thead>{_render_leaf_header()}</thead>\n'
            f'<tbody>\n{"".join(rows)}\n</tbody></table>')


def _render_tree_node(name: str, node: dict, depth: int, thumbs_dir: Path | None,
                      thumb_ok: set[int]) -> tuple[str, int]:
    """Recursively render a tree node as <details>; returns (html, subtree count).

    Collapsing — via native <details>/<summary>, no JS. The top level (depth=0) is
    expanded (<details open>), deeper — collapsed (F21 #4: so there is no wall of
    text by default). The count in <summary> — the sum of files of the whole subtree,
    including nested nodes.
    """
    children_html: list[str] = []
    total = len(node["files"])
    for child_name in sorted(node["children"], key=_tree_sort_key):
        child_html, child_count = _render_tree_node(
            child_name, node["children"][child_name], depth + 1, thumbs_dir, thumb_ok)
        children_html.append(child_html)
        total += child_count
    files_html = _render_file_rows(node["files"], thumbs_dir, thumb_ok) if node["files"] else ""
    open_attr = " open" if depth == 0 else ""
    html = (f'<details{open_attr}><summary>{escape(name)} <small>({total})</small></summary>\n'
            f'{files_html}{"".join(children_html)}</details>\n')
    return html, total


def _write_plan_html(html_path: Path, plan: list[PlanItem], dest: Path,
                     thumbnails: bool = False, thumbnail_workers: int = 8) -> Path | None:
    """The plan HTML report: a collapsible tree by target_rel segments (F21).

    Tested separately from FS moves — does not touch the DB and does not move files,
    only writes html_path (and thumbs_dir next to it, if thumbnails=True).
    Returns thumbs_dir if thumbnails are on, otherwise None. Thumbnails are
    generated BEFORE assembling the HTML, in parallel (F18, thumbnail_workers).
    """
    thumbs_dir = html_path.parent / f"{html_path.stem}_thumbs" if thumbnails else None
    thumb_ok: set[int] = set()
    if thumbnails and thumbs_dir is not None:
        thumb_ok = _generate_thumbnails(plan, thumbs_dir, thumbnail_workers)

    tree = _build_tree(plan)
    top_html: list[str] = []
    if tree["files"]:  # defensive case: a file without a directory segment — should not happen
        top_html.append(_render_file_rows(tree["files"], thumbs_dir, thumb_ok))
    for name in sorted(tree["children"], key=_tree_sort_key):
        node_html, _count = _render_tree_node(
            name, tree["children"][name], 0, thumbs_dir, thumb_ok)
        top_html.append(node_html)

    html = f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<title>План сортировки: {escape(dest.name)} ({len(plan)} файлов)</title>
<style>
body {{ font-family: sans-serif; margin: 1rem; }}
h2 {{ margin-top: 2rem; overflow-wrap: anywhere; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ text-align: left; padding: 4px 8px; border-bottom: 1px solid #ddd; vertical-align: middle; }}
img {{ width: 64px; height: 64px; object-fit: cover; border-radius: 4px;
      vertical-align: middle; margin-right: 6px; }}
details {{ margin-left: 1rem; }}
summary {{ cursor: pointer; font-weight: bold; margin: 0.3rem 0; overflow-wrap: anywhere; }}
details table {{ margin: 0.2rem 0 0.6rem 1rem; width: calc(100% - 1rem); }}
.tree-controls {{ margin: 0.5rem 0; }}
.tree-controls button {{ margin-right: 0.5rem; padding: 4px 10px; cursor: pointer; }}
th[data-sort-type] {{ cursor: pointer; user-select: none; }}
.sorta-sort-ind {{ font-size: 0.75em; }}
.sorta-top {{ position: fixed; right: 1.2rem; bottom: 1.2rem; padding: 8px 12px;
      cursor: pointer; border-radius: 6px; opacity: 0.85; z-index: 1000; }}
.sorta-top:hover {{ opacity: 1; }}
</style></head><body>
<h1>План сортировки: {escape(dest.name)} <small>({len(plan)} файлов)</small></h1>
{_render_near_dup_section(plan)}
<div class="tree-controls">
<button type="button" id="sorta-expand-all">Развернуть всё</button>
<button type="button" id="sorta-collapse-all">Свернуть всё</button>
</div>
<div class="tree">
{"".join(top_html)}
</div>
<button type="button" id="sorta-top" class="sorta-top" title="Наверх">↑ Наверх</button>
<script>
document.getElementById('sorta-expand-all').addEventListener('click', function () {{
  document.querySelectorAll('details').forEach(function (d) {{ d.open = true; }});
}});
document.getElementById('sorta-collapse-all').addEventListener('click', function () {{
  document.querySelectorAll('details').forEach(function (d) {{ d.open = false; }});
}});
document.getElementById('sorta-top').addEventListener('click', function () {{
  window.scrollTo({{ top: 0, behavior: 'smooth' }});
}});
function sortaSort(th) {{
  // F24: сортировка только СВОЕЙ таблицы — находим её от кликнутого <th>
  // (this), а не через глобальный querySelectorAll по документу.
  var table = th.closest('table');
  var tbody = table.querySelector('tbody');
  var headCells = th.parentNode.children;
  var idx = Array.prototype.indexOf.call(headCells, th);
  var type = th.getAttribute('data-sort-type');
  var dir = th.getAttribute('data-sort-dir') === 'asc' ? 'desc' : 'asc';
  Array.prototype.forEach.call(headCells, function (h) {{
    h.removeAttribute('data-sort-dir');
    var ind = h.querySelector('.sorta-sort-ind');
    if (ind) {{ ind.textContent = ''; }}
  }});
  th.setAttribute('data-sort-dir', dir);
  var ownInd = th.querySelector('.sorta-sort-ind');
  if (ownInd) {{ ownInd.textContent = dir === 'asc' ? ' ▲' : ' ▼'; }}
  function sortaKey(row) {{
    var cell = row.children[idx];
    if (!cell) {{ return ''; }}
    if (type === 'date') {{ return cell.getAttribute('data-sort') || ''; }}
    return cell.textContent.trim().toLowerCase();
  }}
  var rows = Array.prototype.slice.call(tbody.children);
  rows.sort(function (a, b) {{
    var ka = sortaKey(a), kb = sortaKey(b);
    var ea = ka === '', eb = kb === '';
    if (ea || eb) {{ return ea === eb ? 0 : (ea ? 1 : -1); }}
    if (ka === kb) {{ return 0; }}
    var cmp = ka < kb ? -1 : 1;
    return dir === 'asc' ? cmp : -cmp;
  }});
  rows.forEach(function (r) {{ tbody.appendChild(r); }});
}}
</script>
</body></html>
"""
    html_path.write_text(html, encoding="utf-8")
    return thumbs_dir


def _record_failed(conn: sqlite3.Connection, batch_id: int, item: PlanItem,
                   hash_value: str) -> None:
    conn.execute(
        "INSERT INTO moves (batch_id, file_id, src, dst, hash, status) "
        "VALUES (?, ?, ?, ?, ?, 'failed')",
        (batch_id, item.file_id, str(item.src), str(item.dst), hash_value))
    conn.commit()


def _precheck_hash(conn: sqlite3.Connection, batch_id: int, item: PlanItem,
                   report: SortReport) -> str | None:
    """Hash verification before the move — a safeguard against a stale index."""
    try:
        src_hash, algo = file_hash(_fs(item.src))
    except OSError as exc:
        _log.warning("sort: источник недоступен, пропуск: %s (%s)", item.src, exc)
        _record_failed(conn, batch_id, item, item.db_hash or "")
        report.failed += 1
        return None
    if item.db_hash and item.db_algo == algo and src_hash != item.db_hash:
        _log.warning("sort: файл изменился после индексации, пропуск: %s", item.src)
        _record_failed(conn, batch_id, item, src_hash)
        report.failed += 1
        return None
    return src_hash


def plan_and_sort(cfg: Config, conn: sqlite3.Connection, mode: str,
                  dest: Path | None, apply: bool = False,
                  copy: bool = False,
                  where: Sequence[str] | None = None,
                  thumbnails: bool = False,
                  dedupe: bool = False,
                  delete_worse_dupes: bool = False,
                  exclude: Sequence[str] | None = None,
                  write_reports: bool = True,
                  keep_manual_excluded: bool = False,
                  progress: Callable[[int, int], None] | None = None,
                  should_cancel: Callable[[], bool] | None = None) -> SortReport:
    """Build a layout plan; with apply=True move files with journaling.

    write_reports=False skips the CSV/HTML artefacts (the returned SortReport still
    carries the paths they WOULD have had). The UI calls this purely to read the plan
    into memory, and every rebuild was silently dropping six report files into
    report_output/ as a side effect of someone opening a tab.

    Dry-run (default): prints a summary, writes the CSV and HTML plan next to the DB
    and performs no FS or journal operation. where — conditions from --where.
    thumbnails=True — additionally puts thumbnails into a cache folder next to the
    HTML report (decode is heavy, hence behind a flag).

    F28: dest=None — in-place layout, the target root = the source root
    (cfg.sources[0]). Requires exactly one source in cfg.sources — otherwise
    ValueError (a common parent for several sources cannot be guessed, an explicit
    error is safer). With apply=True a warning is printed that the ORIGINAL tree is
    being restructured (unlike a layout into a separate --dest). Idempotency (a
    repeated apply touches nothing for already-sorted files) is provided by the
    existing _resolve_dst / PlanItem.in_place mechanism — it does not depend on
    whether dest matches the source.

    copy=True (C16): instead of moving, files are COPIED into the target structure,
    the originals stay in place — files.path is NOT updated. Journaled as
    move_batches.operation='copy'; undo of such a batch deletes the copies (dst)
    rather than restoring src. Only the apply stage differs — plan/CSV/HTML are the same.

    F16 (--exclude): directories of the already-manually-sorted part of the
    collection (a folder + all subfolders, by path boundary) drop out of the plan
    entirely — before layout, near-dup grouping (F14) and writing CSV/HTML; the files
    stay in the index. exclude is combined with config sort.exclude_dirs; the number
    excluded is in report.excluded.

    F77 (manual_overrides, written by `sorta ui`): a per-file correction made by eye.
    action='exclude' — the file is dropped from the plan before layout/near-dup
    grouping/reports (it is not moved anywhere; counted in report.manual_excluded,
    separately from --exclude); action='reassign' — its target folder is
    `dest/<target>/<name>`, ahead of every automatic rule, with the usual name-conflict
    suffixes (report.manual_reassigned). An invalid target (`..`, absolute, a drive) is
    ignored with a warning and the file is laid out automatically — a correction can
    never write outside dest. Files without a correction are laid out exactly as before.
    action='photo' (F103) — the junk/document/product verdict is ignored for this file
    and it is laid out like any other photo (by city/date); media_class is untouched.

    keep_manual_excluded=True — for the web app's PREVIEW only: files marked "leave
    alone" stay in the returned plan, carrying the target they WOULD have had and
    reason='manual_exclude', so the UI can keep showing (and unmarking) them instead of
    losing them from the grid the moment they are marked. Forced off whenever
    apply=True, so a plan that actually moves files can never contain them.

    F14 (--dedupe): among near-duplicates (pHash, only in the current --where
    selection) the best by quality (width*height, then size) is sorted normally, the
    rest are moved to _Duplicates/ (reason near_dup). Requires a computed pHash —
    otherwise a hint and an empty plan (nothing is written to disk), like in
    `dupes --near`. delete_worse_dupes=True (only with dedupe) instead of moving to
    _Duplicates/ PERMANENTLY deletes the worse ones — not undoable via undo, the
    status in moves is 'deleted' (audit).

    F97 (should_cancel): a predicate polled at the START of each file's iteration,
    before the moves row is written. On True the loop BREAKS — it does not raise, the
    way the UI pipeline cancels itself out of a progress callback. An exception here
    would fly past the code that closes the batch, and a batch left with
    finished_at=NULL is exactly what undo (the tool the user reaches for after a
    cancel) has to be able to read. The check is deliberately not inside `_transfer`:
    a half-copied file must either finish or be removed, and `_copy_and_verify`
    already guarantees that. report.cancelled says so, so "copied 4 000 of 22 364" can
    be told apart from a plain "done".

    F97 (resuming): a second apply into the same dest no longer duplicates what the
    first one copied — see `_resolve_dst`/`report.skipped_already_copied`. Without it
    cancelling would be useless: the run could be stopped but not continued.
    """
    if mode not in MODES:
        raise ValueError(f"неизвестный режим {mode!r}; допустимы: {', '.join(MODES)}")
    strategy = str(cfg.sort.multi_person)
    if strategy not in _MULTI_PERSON:
        raise ValueError(f"sort.multi_person: {strategy!r}; "
                         f"допустимы: {', '.join(_MULTI_PERSON)}")
    if delete_worse_dupes and not dedupe:
        raise ValueError("--delete-worse-dupes требует --dedupe")
    lang = i18n.normalize_lang(cfg.raw.get("language"))
    # sort.drop_unlocalized_district is not yet typed in SortConfig —
    # read via getattr with a default of True.
    drop_unlocalized_district = bool(
        getattr(cfg.sort, "drop_unlocalized_district", True))
    if delete_worse_dupes:
        print(i18n.cli_text("cli.sort.warn_delete_dupes", lang))
    in_place_run = dest is None
    if dest is None:
        if len(cfg.sources) != 1:
            raise ValueError(
                "in-place раскладка требует единственного источника; "
                "задайте --dest или оставьте один каталог в sources")
        dest = cfg.sources[0]
    if in_place_run and apply:
        print(i18n.cli_text("cli.sort.warn_in_place", lang,
                            path=Path(dest).resolve()))
    conn.create_function("casefold", 1, _sql_casefold, deterministic=True)
    resolver = GeoResolver()  # G3: lazy loading of bundled data on first access
    cond, params = parse_where(where or [], lang, resolver)
    dest = Path(dest).resolve()

    if dedupe:
        have_phash = conn.execute(
            "SELECT COUNT(*) FROM files WHERE phash IS NOT NULL").fetchone()[0]
        if not have_phash:
            print(i18n.cli_text("cli.dupes.no_phash", lang))  # same sentence as `dupes`
            placeholder = _report_dir(cfg) / "sort_plan_no_phash"
            return SortReport(mode=mode, dest=dest,
                              csv_path=placeholder.with_suffix(".csv"),
                              html_path=placeholder.with_suffix(".html"))

    # F85c: `mp.file_id IS NULL` (not COALESCE per column) decides between the two
    # sources of a place — a manual row wins as a WHOLE, so a hand-picked city can never
    # end up under an inferred country, and the district of the automatic place cannot
    # survive under a city the user replaced. `country_name` is dropped for a manual row
    # on purpose: it is the provider's spelling of the OLD country, and _target_parts
    # localizes the cc through i18n.country when it is absent.
    rows = conn.execute(
        _CTE + f"""SELECT f.id, f.path, f.taken_at, f.taken_at_confidence,
               f.hash, f.hash_algo, f.not_personal, f.gps_lat, f.gps_lon,
               f.camera_make, f.camera_model,
               CASE WHEN mp.file_id IS NULL THEN p.country ELSE mp.country END AS country,
               CASE WHEN mp.file_id IS NULL THEN p.country_name END AS country_name,
               CASE WHEN mp.file_id IS NULL THEN p.city ELSE mp.city END AS city,
               CASE WHEN mp.file_id IS NULL THEN p.confidence
                    ELSE 'manual' END AS place_confidence,
               CASE WHEN mp.file_id IS NULL THEN p.city_geonameid
                    ELSE mp.city_geonameid END AS city_geonameid,
               CASE WHEN mp.file_id IS NULL THEN p.district_geonameid END AS district_geonameid,
               CASE WHEN mp.file_id IS NULL THEN p.district_name END AS district_name,
               mc.verdict AS junk_verdict, mc.source AS junk_source,
               dc.action AS dedup_action,
               mo.action AS manual_action, mo.target AS manual_target
           FROM files f
           LEFT JOIN places p ON p.file_id = f.id
           LEFT JOIN manual_places mp ON mp.file_id = f.id
           LEFT JOIN media_class mc ON mc.file_id = f.id
           LEFT JOIN dedup_choice dc ON dc.file_id = f.id
           LEFT JOIN manual_overrides mo ON mo.file_id = f.id
           WHERE f.dup_of IS NULL AND f.error IS NULL AND {cond}
           ORDER BY f.path""", params).fetchall()

    excludes = _resolve_excludes(cfg, exclude)
    excluded_count = 0
    if excludes:
        kept_rows = []
        for r in rows:
            if _is_excluded(Path(r["path"]).resolve(), excludes):
                excluded_count += 1
            else:
                kept_rows.append(r)
        rows = kept_rows

    # F77: "leave alone" is not a target folder — such a file is not moved at all, so
    # it leaves the plan here, before layout, near-dup grouping and the reports. Its
    # own counter, separate from the --exclude directories above. keep_manual_excluded
    # (preview only, never with apply — see the docstring) keeps the rows so the web app
    # can still show the mark; they are then flagged reason='manual_exclude' below and
    # take part in nothing that moves a file.
    manual_excluded_count = 0
    keep_excluded = keep_manual_excluded and not apply
    if any(r["manual_action"] == "exclude" for r in rows):
        kept_rows = []
        for r in rows:
            if r["manual_action"] == "exclude":
                manual_excluded_count += 1
                if keep_excluded:
                    kept_rows.append(r)
            else:
                kept_rows.append(r)
        rows = kept_rows

    persons_by_file = _load_persons(conn)
    events_by_file = _load_events(conn)

    row_targets: list[tuple[sqlite3.Row, list[str], str,
                           list[tuple[str, float]], tuple[str, str | None] | None]] = []
    manual_reassigned_count = 0
    for r in rows:
        persons = persons_by_file.get(r["id"], [])
        event = events_by_file.get(r["id"])
        parts, reason = _target_parts(mode, strategy, r, persons, event, lang, resolver,
                                      drop_unlocalized_district)
        if reason == "manual_reassign":
            manual_reassigned_count += 1
        if r["manual_action"] == "exclude":
            # only reachable with keep_excluded (preview): the automatic target stays,
            # the reason says the file is not going anywhere
            reason = "manual_exclude"
        row_targets.append((r, parts, reason, persons, event))

    near_dup_best: dict[int, int] = {}
    near_dup_worse: dict[int, int] = {}
    if dedupe:
        # junk/document/not_personal/dedup_delete are excluded from grouping BEFORE
        # picking the best: otherwise such a file with a higher resolution could
        # "win" the group and pull a normal photo into _Duplicates instead of its
        # usual layout (they are all sorted separately, independent of near-dups);
        # dedup_delete — an explicit manual user decision (U3b), it must not pull a
        # near-dup group onto itself; manual_reassign (F77) — the same reasoning, and
        # the near-dup role must not overwrite the folder the user chose by hand.
        sortable_ids = {r["id"] for r, _parts, reason, _p, _e in row_targets
                        if reason not in ("junk", "document", "not_personal",
                                          "dedup_delete", "manual_reassign",
                                          "manual_exclude")}
        near_dup_best, near_dup_worse = _resolve_near_dup_roles(conn, cfg, sortable_ids)

    claimed: set[str] = set()
    plan: list[PlanItem] = []
    for r, parts, reason, persons, event in row_targets:
        near_dup_group: int | None = None
        near_dup_role: str | None = None
        if r["id"] in near_dup_best:
            near_dup_group, near_dup_role = near_dup_best[r["id"]], "kept"
        elif r["id"] in near_dup_worse:
            near_dup_group = near_dup_worse[r["id"]]
            near_dup_role = "deleted" if delete_worse_dupes else "moved"
            parts = [i18n.folder("duplicates", lang)]
            reason = "near_dup_delete" if delete_worse_dupes else "near_dup"
        src = Path(r["path"])
        dst, in_place, already_copied = _resolve_dst(
            dest.joinpath(*parts), src, claimed, r["hash"], r["hash_algo"])
        try:
            target_rel = dst.relative_to(dest).as_posix()
        except ValueError:  # only on a path-case divergence on Windows
            target_rel = dst.as_posix()
        plan.append(PlanItem(
            file_id=r["id"], src=src, dst=dst, in_place=in_place,
            target_rel=target_rel, reason=reason,
            taken_at=r["taken_at"], taken_at_confidence=r["taken_at_confidence"],
            # G3: the city is carried in the layout language, not as the English
            # anchor of `places` — the CSV/HTML reports and the web app's cards all
            # read it from here, and a plan that lays a frame into «Санкт-Петербург»
            # while its own Geo column says «St Petersburg» reads as two places.
            # The country stays as the DB has it (an ISO cc or the online provider's
            # full name): it is localized where it is FORMATTED, via i18n.country.
            country=r["country"],
            city=_city_display_name(r["city"], r["city_geonameid"], lang, resolver),
            place_confidence=r["place_confidence"],
            gps_lat=r["gps_lat"], gps_lon=r["gps_lon"],
            persons=[label for label, _area in persons],
            event=event[0] if event else None,
            junk_verdict=r["junk_verdict"], junk_source=r["junk_source"],
            db_hash=r["hash"], db_algo=r["hash_algo"],
            near_dup_group=near_dup_group, near_dup_role=near_dup_role,
            already_copied=already_copied))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"sort_plan_{mode}_{stamp}"
    csv_path = _report_dir(cfg) / f"{stem}.csv"
    html_path = csv_path.with_name(f"{stem}.html")
    if write_reports:
        _write_plan_csv(csv_path, plan)
        thumb_workers = cfg.sort.thumbnail_workers or min(8, os.cpu_count() or 4)
        _write_plan_html(html_path, plan, dest, thumbnails=thumbnails,
                         thumbnail_workers=thumb_workers)
    report = SortReport(mode=mode, dest=dest, csv_path=csv_path, html_path=html_path,
                        plan=plan, dirs=len({it.dst.parent for it in plan}),
                        excluded=excluded_count, in_place=in_place_run,
                        manual_excluded=manual_excluded_count,
                        manual_reassigned=manual_reassigned_count)
    excluded_note = (i18n.cli_text("cli.sort.plan_excluded", lang, n=excluded_count)
                     if excludes else "")
    # F77: manual corrections are reported on their own, not merged into the excluded
    # count — they are a person's decision, not a rule from the config.
    manual_note = ""
    if manual_excluded_count or manual_reassigned_count:
        manual_note = i18n.cli_text("cli.sort.plan_manual", lang,
                                    reassigned=manual_reassigned_count,
                                    excluded=manual_excluded_count)
    where_note = (i18n.cli_text("cli.sort.plan_paths", lang,
                                csv=csv_path, html=html_path) if write_reports else "")
    # The command echo stays untranslated on purpose: it is what the reader would type.
    print(f"sort --by {mode}{' --apply' if apply else ' (dry-run)'}: "
          + i18n.cli_text("cli.sort.plan_counts", lang,
                          files=len(plan), dirs=report.dirs)
          + f"{where_note}{excluded_note}{manual_note}")
    if not apply:
        return report

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cur = conn.execute(
        "INSERT INTO move_batches (mode, dest_root, started_at, operation) VALUES (?, ?, ?, ?)",
        (mode, str(dest), now, "copy" if copy else "move"))
    batch_id = cur.lastrowid
    conn.commit()
    assert batch_id is not None
    report.batch_id = batch_id

    for i, item in enumerate(plan, 1):
        if should_cancel is not None and should_cancel():
            # F97: break, never raise — the batch below MUST get its finished_at
            _log.info("sort: отмена по запросу, перенесено %d из %d", report.moved, len(plan))
            report.cancelled = True
            break
        if progress:
            progress(i, len(plan))
        if item.in_place:
            report.skipped_in_place += 1
            continue
        if item.already_copied:
            # F97: the copy is already in the target from an earlier apply — leaving
            # it alone is the whole point; a moves row here would only journal a
            # non-event.
            report.skipped_already_copied += 1
            continue
        src_hash = _precheck_hash(conn, batch_id, item, report)
        if src_hash is None:
            continue
        cur = conn.execute(
            "INSERT INTO moves (batch_id, file_id, src, dst, hash, status) "
            "VALUES (?, ?, ?, ?, ?, 'planned')",
            (batch_id, item.file_id, str(item.src), str(item.dst), src_hash))
        move_id = cur.lastrowid
        conn.commit()  # invariant: the journal is committed BEFORE the FS operation
        if item.near_dup_role == "deleted":
            try:
                _fs(item.src).unlink()
            except OSError as exc:
                _log.warning("sort: удаление не удалось, пропуск: %s (%s)", item.src, exc)
                conn.execute("UPDATE moves SET status = 'failed' WHERE id = ?", (move_id,))
                conn.commit()
                report.failed += 1
                continue
            conn.execute("UPDATE moves SET status = 'deleted' WHERE id = ?", (move_id,))
            conn.commit()
            report.deleted += 1
            continue
        try:
            _transfer(item.src, item.dst, src_hash, copy=copy)
        except TransferError as exc:
            _log.warning("sort: %s", exc)
            conn.execute("UPDATE moves SET status = 'failed' WHERE id = ?", (move_id,))
            conn.commit()
            report.failed += 1
            continue
        conn.execute("UPDATE moves SET status = 'done' WHERE id = ?", (move_id,))
        if not copy:
            # copy: the original is untouched, files.path keeps pointing to src
            conn.execute("UPDATE files SET path = ? WHERE id = ?",
                         (str(item.dst), item.file_id))
        conn.commit()
        report.moved += 1

    conn.execute(
        "UPDATE move_batches SET finished_at = ? WHERE id = ?",
        (datetime.now(timezone.utc).isoformat(timespec="seconds"), batch_id))
    conn.commit()
    return report


def undo(conn: sqlite3.Connection, batch_id: int | None = None,
         progress: Callable[[int, int], None] | None = None,
         should_cancel: Callable[[], bool] | None = None) -> UndoStats:
    """Undo a batch by the journal in reverse order.

    batch_id=None — the last batch that still has moves with status='done' (repeated
    calls pop batches like a stack). A missing dst is logged (the status stays
    'done'), the rollback continues. An occupied src is not overwritten — the file is
    restored with a suffix _1, _2, ...

    move_batches.operation='copy' (C16) — a different rollback: dst (the copy) is just
    deleted after a hash check, files.path and src are not touched (the original
    never moved). On a hash mismatch the copy is NOT deleted (the status stays
    'done', failed++), since it is unclear what exactly changed.
    operation='link' (F34) — the same path as 'copy': dst is a hardlink (or a copy
    fallback), deleting dst is safe and does not touch the source data.

    F97 (should_cancel): polled at the start of each row, `break` and never an
    exception — the same discipline as plan_and_sort, and for a stronger reason:
    undoing a copy batch re-hashes every copy (blake3 over 220 GB is minutes to tens
    of minutes), so it cannot be an operation the user is unable to stop. Rows already
    processed keep their new status, the rest keep the old one, and a repeated undo
    finishes the job — idempotency here matters more than speed.

    F97 (the tail of an interrupted copy): rows still in status='planned' whose dst
    exists. The journal is committed BEFORE the FS operation, so a run killed between
    the two leaves a fully written file that undo used to walk straight past — an
    orphan in the result that looks like an ordinary photo. Such a row is now handled
    exactly like a 'done' one of the same operation, on one condition: the blake3 of
    dst must match moves.hash. On a match it is our own complete file (deleted for
    copy/link, moved back for move); on a mismatch it is a copy interrupted mid-write
    — NOT deleted, reported in `stats.stray` for the user to look at by hand. A
    'planned' row without a dst on disk means the operation never started; there is
    nothing to undo and nothing to report.

    F97: a batch left with finished_at=NULL (an interrupted apply) is closed here —
    otherwise it goes on looking like "running right now" forever.
    """
    if batch_id is None:
        row = conn.execute(
            "SELECT MAX(batch_id) AS last_id FROM moves WHERE status = 'done'"
        ).fetchone()
        if row is None or row["last_id"] is None:
            raise ValueError("undo: нет завершённых перемещений для отката")
        batch_id = int(row["last_id"])
    batch = conn.execute(
        "SELECT operation FROM move_batches WHERE id = ?", (batch_id,)).fetchone()
    operation = batch["operation"] if batch else "move"
    rows = conn.execute(
        "SELECT id, file_id, src, dst, hash, status FROM moves "
        "WHERE batch_id = ? AND status IN ('done', 'planned') ORDER BY id DESC",
        (batch_id,)).fetchall()
    stats = UndoStats(batch_id=batch_id)
    for i, r in enumerate(rows, 1):
        if should_cancel is not None and should_cancel():
            _log.info("undo: отмена по запросу, откачено %d из %d", stats.undone, len(rows))
            stats.cancelled = True
            break
        if progress:
            progress(i, len(rows))
        src, dst = Path(r["src"]), Path(r["dst"])
        tail = r["status"] == "planned"
        if not _fs(dst).exists():
            if tail:
                continue  # the FS operation never started — nothing was written
            _log.warning("undo: dst отсутствует, статус остаётся 'done': %s", dst)
            stats.missing += 1
            continue
        if operation in ("copy", "link") or tail:
            # A tail row goes through the hash check whatever the operation is: only a
            # full match proves the file at dst is ours and complete.
            try:
                dst_hash = file_hash(_fs(dst))[0]
            except OSError as exc:
                _log.warning("undo: копия недоступна, пропуск: %s (%s)", dst, exc)
                stats.failed += 1
                continue
            if dst_hash != r["hash"]:
                if tail:
                    _log.warning("undo: битая копия прерванного переноса, НЕ удалена: %s", dst)
                    stats.stray.append(str(dst))
                    continue
                _log.warning("undo: хэш копии не совпал, копия НЕ удалена: %s", dst)
                stats.failed += 1
                continue
            if operation in ("copy", "link"):
                _fs(dst).unlink()
                conn.execute("UPDATE moves SET status = 'undone' WHERE id = ?", (r["id"],))
                conn.commit()
                stats.undone += 1
                continue
            # a move batch's tail: the file belongs at src, not in the bin
        restore, n = src, 0
        while _fs(restore).exists():
            n += 1
            restore = src.with_name(f"{src.stem}_{n}{src.suffix}")
        if n:
            _log.warning("undo: %s занят, восстановление как %s", src, restore.name)
        try:
            _transfer(dst, restore)
        except TransferError as exc:
            _log.warning("undo: %s", exc)
            stats.failed += 1
            continue
        conn.execute("UPDATE moves SET status = 'undone' WHERE id = ?", (r["id"],))
        conn.execute("UPDATE files SET path = ? WHERE id = ?",
                     (str(restore), r["file_id"]))
        conn.commit()
        stats.undone += 1
    conn.execute(
        "UPDATE move_batches SET finished_at = ? WHERE id = ? AND finished_at IS NULL",
        (datetime.now(timezone.utc).isoformat(timespec="seconds"), batch_id))
    conn.commit()
    return stats


# --- F34: album engine (export a person/event slice into a named folder) ----------
#
# An album is a targeted export of an index slice (not a full layout): all canonical
# files of a person (accounting for cluster merges, F31) or an event, optionally
# narrowed by --where, flat into dest/<album_name>/. The base city layout is not
# touched; an album is an additional "view" (link/copy) or, on explicit request, a
# removal from the pool (move). Journal/undo — the shared move_batches/moves
# mechanism, operation='link'|'copy'|'move' (undo for 'link' — see _transfer/undo above).

# F123: `animal` joins the pair. It is a slice like the other two — canonical files the
# frame-quality stage marked as holding an animal — and the only thing that makes it
# different is that there is nothing to select INSIDE it: the whole collection has one
# animal view, so the selector is accepted and ignored.
#
# F129: `query` is the one whose slice is not written down anywhere. The selector is the
# words a person typed, and what it selects is the top of a RANKING over the stored CLIP
# vectors (search.py) — `features.search_limit` frames, best first. There is no threshold
# in it and there will not be one: the score orders frames against each other and says
# nothing in absolute terms, so the album is a sample to look through, not a claim that
# each of its frames holds a cake. Everything else about it is an album like any other.
#
# F139: the rest of the slices the interface already draws. Nothing about them needed
# inventing — the engine has gathered any slice into a folder since F34, and products,
# screenshots, memes, blurred frames, closed eyes and "no subject" were left with a
# counter and a delete button only because their views arrived after the album did.
#
# The class slices are one `media_class.verdict` each, over the same canonical, readable
# population every other counter uses — so the album and the bucket's counter are the
# same number by construction. `document` is deliberately absent from the tuple, and it
# is a config question rather than a constant: `vlm.exclude_classes` (F133) already means
# "this class is private", and the guard in `plan_album` reads that key, so a class moved
# INTO it loses its album along with its preview instead of keeping one of the two.
CLASS_ALBUM_KINDS = ("product", "screenshot", "meme")

# The quality slices of the "Review" workspace (F126). They select on `frame_quality`,
# and `blurred` selects through the SAME window the workspace lists — see
# `quality_slice_where`.
QUALITY_ALBUM_KINDS = ("blurred", "eyes_closed", "no_subject")

ALBUM_KINDS = (("person", "event", "animal", "query")
               + CLASS_ALBUM_KINDS + QUALITY_ALBUM_KINDS)
ALBUM_MODES = ("link", "copy", "move")

# The kinds with nothing to select INSIDE them: the collection has exactly one animal
# view, one products bucket, one blurred list. An empty selector is accepted for these
# (and only for these — for a person, an event or a query it is the subject itself, and
# gathering "everything" would be the wrong answer to a client that lost it).
SELECTORLESS_ALBUM_KINDS = ("animal",) + CLASS_ALBUM_KINDS + QUALITY_ALBUM_KINDS

# The default album name of a slice that cannot name itself after a selector: a folder
# name like any other the layout creates, so it comes from the catalog and follows
# `language:`. `product` reuses the `products` folder the city layout already files that
# bucket into — the album a person gathers by hand and the folder the plan builds should
# not be two differently named things.
ALBUM_FOLDER_KEYS = {
    "animal": "animals",
    "product": "products",
    "screenshot": "screenshots",
    "meme": "memes",
    "blurred": "blurred",
    "eyes_closed": "eyes_closed",
    "no_subject": "no_subject",
}

# F124: THE rule for "is there an animal in this frame", written down once. The user's
# verdict (`manual_pet`, they looked at the frame) outranks the model's
# (`frame_quality.pet`, it looked at the frame), and COALESCE keeps the automatic answer
# wherever no manual row exists — a file nobody has touched behaves exactly as it did
# before this feature.
#
# The rule is applied WHEN READ, never when written: `junk` still recomputes
# `frame_quality` from scratch on every run and knows nothing about this table, which is
# precisely why a manual mark survives a change of model, of prompts or of the threshold.
#
# Both joins are LEFT joins, and that is not decoration: a frame the user marked as an
# animal need not have a `frame_quality` row at all (the stage never reached it), and it
# belongs in the slice all the same.
#
# A SELECT of ids rather than a CTE, because it has to compose in two places — the album
# query already opens with the recursive `_CTE`, and the tab needs the same rule inside a
# SELECT list. It takes no parameters, so it can be interpolated into either. The one
# consumer outside this module is ui.py (the "Animals" tab and the Overview counter):
# two independent spellings of this expression would drift, and the day they did the
# counter, the tab and the album would each report a different collection.
ANIMAL_IDS_SQL = """SELECT af.id FROM files af
    LEFT JOIN frame_quality afq ON afq.file_id = af.id
    LEFT JOIN manual_pet amp ON amp.file_id = af.id
    WHERE COALESCE(amp.is_animal, afq.pet IS NOT NULL)"""

# F139: the same idea as ANIMAL_IDS_SQL, for the three quality slices — the membership
# rule written down ONCE, in terms of the aliases `f` (files), `fq` (frame_quality) and
# `mc` (media_class), and read by both consumers: the album here and the "Review"
# workspace in ui.py, which draws the list and its counter from it. Two spellings would
# drift, and the day they did the chip, the list and the album would each report a
# different set of frames — which is the one thing this feature must not do, because the
# whole point of these slices is that the decision is taken by eye, on what was shown.
#
# Photographs only (F120: sharpness and open eyes mean nothing on a screenshot or a
# receipt), canonical and readable.
QUALITY_FROM = ("FROM files f JOIN frame_quality fq ON fq.file_id = f.id "
                "JOIN media_class mc ON mc.file_id = f.id")
QUALITY_POPULATION = "mc.verdict = 'photo' AND f.dup_of IS NULL AND f.error IS NULL"

# `eyes_open`/`has_subject` are `= 0` and never `IS NOT 1`: NULL there means "not asked"
# (schema), and a frame nobody looked at must not be shown to a user as an answer.
_QUALITY_MEMBER = {
    "blurred": "fq.sharpness IS NOT NULL",
    "eyes_closed": "fq.eyes_open = 0",
    "no_subject": "fq.has_subject = 0",
}


def quality_slice_where(kind: str, blur_max: float | None) -> tuple[str, list[object]]:
    """The WHERE of one quality slice + its parameters, against `QUALITY_FROM`.

    `blur_max` is the window the blurred list opens to (`features.blur_review_max`) and
    applies to that slice alone; None — no ceiling (the workspace's "show more", which
    continues past the window). The window is a prefix of the same ordering, so
    continuing past it neither repeats a frame nor skips one.

    An album is ALWAYS gathered inside the window: the button collects what was shown,
    and past the window sit thousands of frames nobody has looked at.
    """
    where = f"{QUALITY_POPULATION} AND {_QUALITY_MEMBER[kind]}"
    params: list[object] = []
    if kind == "blurred" and blur_max is not None:
        where += " AND fq.sharpness < ?"
        params.append(float(blur_max))
    return where, params


@dataclass
class AlbumPlanItem:
    file_id: int
    src: Path
    dst: Path
    persons: list[str]     # labels of all named people on the file (for the move check)
    # (kept for every kind: the multi-person block on `move` is about who is in the
    # frame, not about what selected it — an animal photo with two named people in it
    # is exactly as ambiguous to carry off as a person album's.)
    multi_person: bool     # len(persons) >= 2 — with mode='move' such a file is blocked
    already_copied: bool = False   # F97: the album already holds this exact file


@dataclass
class AlbumReport:
    kind: str               # person | event | animal
    selector: str            # as passed by the caller (person name / event name|id;
    #                          empty and ignored for animal)
    album_name: str          # the final folder name (before _sanitize)
    dest: Path               # dest/<album_name> (already resolved)
    mode: str                # link | copy | move
    plan: list[AlbumPlanItem] = field(default_factory=list)
    batch_id: int | None = None
    transferred: int = 0
    failed: int = 0
    blocked_multi: int = 0   # mode='move': files skipped due to multi-membership
    # F97: re-running an album into the same folder no longer re-materializes what is
    # already there — the same `_resolve_dst` rule as the city layout, on purpose: an
    # album has no separate logic of its own.
    skipped_already_copied: int = 0


def _resolve_event_ids_and_name(conn: sqlite3.Connection, selector: str) -> tuple[list[int], str]:
    """selector -> (event id, if selector is a number and such an id exists) | (all ids
    with a casefold-matching name). Also returns the canonical name for the default
    album_name: the exact event name on an unambiguous match, otherwise the original
    selector (several differently-named events with the same id are impossible here,
    but several ids can share one name — then the name is still unambiguous).
    """
    if selector.isdigit():
        row = conn.execute(
            "SELECT id, name FROM events WHERE id = ?", (int(selector),)).fetchone()
        if row is not None:
            return [row["id"]], row["name"]
    rows = conn.execute(
        "SELECT id, name FROM events WHERE casefold(name) = casefold(?)", (selector,)).fetchall()
    if not rows:
        return [], selector
    names = {r["name"] for r in rows}
    name = next(iter(names)) if len(names) == 1 else selector
    return [r["id"] for r in rows], name


def plan_album(cfg: Config, conn: sqlite3.Connection, kind: str, selector: str,
               dest: Path, mode: str = "link",
               where: Sequence[str] | None = None, apply: bool = False,
               album_name: str | None = None,
               encoder: TextEncoder | None = None) -> AlbumReport:
    """Build an album export plan; with apply=True materialize it (link/copy/move).

    kind='person': selector — a person's name; the slice = canonical files (dup_of IS
    NULL) that have a face in a cluster whose merged_into chain root (F31, via the
    shared _CTE/_person_files) has label==selector (casefold).
    kind='event': selector — an event name OR id; the slice = the event(s)' event_files.
    kind='animal' (F123): the slice = files with a `frame_quality.pet` verdict, corrected
    by the user's own marks (F124, `ANIMAL_IDS_SQL` — the same expression the web app's
    tab and counter read, never a second copy of it). The selector is not used — the
    collection has exactly one animal slice — and an empty string is accepted for it; the
    default album name is the localized `animals` folder.
    kind='query' (F129): selector — the words themselves; the slice = the top
    `features.search_limit` canonical photographs of the CLIP ranking for those words
    (search.py), and the default album name is the query. `encoder` is the CLIP text
    encoder and exists for the same reason `detect_landmarks` takes a classifier: the real
    one is loaded on demand, tests hand in a fake. It is ignored by every other kind.
    kind in CLASS_ALBUM_KINDS (F139, `product`/`screenshot`/`meme`): the slice = the
    frames the classifier filed under that verdict — `media_class.verdict = kind`, which
    is what the bucket's counter counts, so the two cannot disagree. No selector; the
    default album name is the class's folder from the catalog. A class listed in
    `vlm.exclude_classes` is REFUSED here (ValueError): that key means "this class is
    private" (F133) and a private bucket keeps its counter and gets neither a preview nor
    an album — gathering somebody's passports into one folder in one click is exactly
    what it exists to prevent.
    kind in QUALITY_ALBUM_KINDS (F139, `blurred`/`eyes_closed`/`no_subject`): the slice =
    the "Review" workspace's flat list of that name (`quality_slice_where`, the shared
    rule), blurred inside the `features.blur_review_max` window. No selector; the default
    album name comes from the catalog.
    where (opt.) reuses parse_where as an additional AND condition on top of the slice
    (person here is the subject, not a where field; --where can still carry its own
    city/country/event/year/person conditions). junk is NOT filtered (these are the
    person's/event's photos), but files.error IS NOT NULL is always excluded, as are
    duplicates (dup_of).

    dry-run (apply=False, default) only prints the plan, writes nothing to the DB/FS.
    apply=True journals into move_batches/moves BEFORE each operation
    (move_batches.mode='album_<kind>', operation=mode) and calls _transfer.

    mode='move' — a warning is always printed (dry-run and apply): the file leaves the
    sort canon. Files with 2+ named people in the frame are NOT moved with move
    (blocked, blocked_multi++) — it is ambiguous whose album it is; link/copy have no
    such restriction.

    F97: an album goes through the same `_resolve_dst` as the city layout, so it
    inherits the same rule — a file already sitting in the album folder byte-for-byte
    is left alone (skipped_already_copied) instead of being re-materialized under a
    `_1` name. No album-specific logic: gathering the same album twice was the same
    bug as applying the same layout twice.
    """
    if kind not in ALBUM_KINDS:
        raise ValueError(f"неизвестный тип альбома {kind!r}; допустимы: {', '.join(ALBUM_KINDS)}")
    if mode not in ALBUM_MODES:
        raise ValueError(f"неизвестный режим альбома {mode!r}; допустимы: {', '.join(ALBUM_MODES)}")
    # F118: this function printed Russian whatever `language:` said — plan_and_sort read
    # the language and plan_album never did.
    lang = i18n.normalize_lang(cfg.raw.get("language"))
    conn.create_function("casefold", 1, _sql_casefold, deterministic=True)

    subject_params: list[object]
    if kind == "person":
        resolved_name = selector
        subject_cond = ("f.id IN (SELECT file_id FROM _person_files "
                        "WHERE casefold(label) = casefold(?))")
        subject_params = [selector]
    elif kind == "animal":
        # F123: `pet IS NOT NULL` — the same "NULL means NOT ASKED" rule the whole
        # frame_quality table lives by: a row exists for every frame the stage touched,
        # and only the ones whose pet score cleared `features.pet_threshold` carry a
        # verdict. F124: with the user's own verdict on top of it, once, via
        # ANIMAL_IDS_SQL. dup_of/error are excluded below, with the other kinds.
        resolved_name = i18n.folder(ALBUM_FOLDER_KEYS["animal"], lang)
        subject_cond = f"f.id IN ({ANIMAL_IDS_SQL})"
        subject_params = []
    elif kind in CLASS_ALBUM_KINDS:
        # F139/F133: the privacy guard is here rather than in the web app, because a
        # button hidden in the browser is not a rule — a request sent past the interface
        # would gather the folder all the same.
        if kind in set(cfg.vlm.exclude_classes):
            raise ValueError(
                f"альбом класса {kind!r} запрещён: класс указан в vlm.exclude_classes")
        resolved_name = i18n.folder(ALBUM_FOLDER_KEYS[kind], lang)
        subject_cond = "f.id IN (SELECT file_id FROM media_class WHERE verdict = ?)"
        subject_params = [kind]
    elif kind in QUALITY_ALBUM_KINDS:
        # The inner query brings its own `f`/`fq`/`mc`, which shadow the outer `f` for
        # the length of the subquery — it is a plain uncorrelated set of ids, and the
        # outer WHERE keeps applying `dup_of`/`error` to the file being planned.
        resolved_name = i18n.folder(ALBUM_FOLDER_KEYS[kind], lang)
        quality_cond, quality_params = quality_slice_where(
            kind, cfg.features.blur_review_max)
        subject_cond = f"f.id IN (SELECT f.id {QUALITY_FROM} WHERE {quality_cond})"
        subject_params = list(quality_params)
    elif kind == "query":
        # F129: the ranking runs FIRST and the ids it returns are the slice. The ids are
        # interpolated rather than bound because `features.search_limit` is a user-set
        # number and SQLite has a ceiling on bound parameters (the reason
        # `read_clip_embeddings` batches its own reads); they come straight out of
        # `files.id`, so the int() is the whole sanitization there is to do.
        # `search_text` raises when there is nothing to rank at all — a caller gets the
        # reason (no vectors / vectors of another model) instead of an empty album.
        resolved_name = selector
        ids = [fid for fid, _score in search_text(cfg, conn, selector, encoder=encoder)]
        subject_cond = f"f.id IN ({','.join(str(int(i)) for i in ids)})" if ids else "0"
        subject_params = []
    else:
        event_ids, resolved_name = _resolve_event_ids_and_name(conn, selector)
        if event_ids:
            qmarks = ",".join("?" * len(event_ids))
            subject_cond = f"f.id IN (SELECT file_id FROM event_files WHERE event_id IN ({qmarks}))"
            subject_params = list(event_ids)
        else:
            subject_cond, subject_params = "0", []  # an intentionally empty slice, without IN ()

    where_cond, where_params = parse_where(where or [])
    full_cond = f"({subject_cond}) AND ({where_cond})"
    full_params = subject_params + where_params

    rows = conn.execute(
        _CTE + f"""SELECT f.id, f.path, f.hash, f.hash_algo FROM files f
               LEFT JOIN places p ON p.file_id = f.id
               WHERE f.dup_of IS NULL AND f.error IS NULL AND {full_cond}
               ORDER BY f.path""", full_params).fetchall()

    final_name = album_name or resolved_name
    album_dir = Path(dest).resolve() / _sanitize(final_name)
    report = AlbumReport(kind=kind, selector=selector, album_name=final_name,
                         dest=album_dir, mode=mode)

    if not rows:
        print(f"album {kind} {selector!r}: "
              + i18n.cli_text("cli.album.empty", lang))
        return report

    persons_by_file = _load_persons(conn)
    if mode == "move":
        print(i18n.cli_text("cli.album.warn_move", lang))

    claimed: set[str] = set()
    for r in rows:
        src = Path(r["path"])
        persons = [label for label, _area in persons_by_file.get(r["id"], [])]
        dst, _in_place, already_copied = _resolve_dst(
            album_dir, src, claimed, r["hash"], r["hash_algo"])
        report.plan.append(AlbumPlanItem(file_id=r["id"], src=src, dst=dst,
                                         persons=persons, multi_person=len(persons) >= 2,
                                         already_copied=already_copied))

    if mode == "move":
        blocked = [it for it in report.plan if it.multi_person]
        if blocked:
            names = ", ".join(str(it.src) for it in blocked[:5])
            more = " …" if len(blocked) > 5 else ""
            print(i18n.cli_text("cli.album.warn_blocked_multi", lang,
                                n=len(blocked), names=names, more=more))

    print(f"album {kind} {selector!r}{' --apply' if apply else ' (dry-run)'} "
          f"[{mode}]: "
          + i18n.cli_text("cli.album.plan_counts", lang,
                          files=len(report.plan), dest=album_dir))

    if not apply:
        return report

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cur = conn.execute(
        "INSERT INTO move_batches (mode, dest_root, started_at, operation) VALUES (?, ?, ?, ?)",
        (f"album_{kind}", str(Path(dest).resolve()), now, mode))
    batch_id = cur.lastrowid
    conn.commit()
    assert batch_id is not None
    report.batch_id = batch_id

    for item in report.plan:
        if mode == "move" and item.multi_person:
            report.blocked_multi += 1
            continue
        if item.already_copied:
            report.skipped_already_copied += 1
            continue
        try:
            src_hash, _algo = file_hash(_fs(item.src))
        except OSError as exc:
            _log.warning("album: источник недоступен, пропуск: %s (%s)", item.src, exc)
            report.failed += 1
            continue
        cur = conn.execute(
            "INSERT INTO moves (batch_id, file_id, src, dst, hash, status) "
            "VALUES (?, ?, ?, ?, ?, 'planned')",
            (batch_id, item.file_id, str(item.src), str(item.dst), src_hash))
        move_id = cur.lastrowid
        conn.commit()  # invariant: the journal is committed BEFORE the FS operation
        try:
            _transfer(item.src, item.dst, src_hash,
                     copy=(mode == "copy"), link=(mode == "link"))
        except TransferError as exc:
            _log.warning("album: %s", exc)
            conn.execute("UPDATE moves SET status = 'failed' WHERE id = ?", (move_id,))
            conn.commit()
            report.failed += 1
            continue
        conn.execute("UPDATE moves SET status = 'done' WHERE id = ?", (move_id,))
        if mode == "move":
            # like plan_and_sort: only a real move updates files.path;
            # link/copy leave the original canonical (files.path untouched).
            conn.execute("UPDATE files SET path = ? WHERE id = ?",
                         (str(item.dst), item.file_id))
        conn.commit()
        report.transferred += 1

    conn.execute(
        "UPDATE move_batches SET finished_at = ? WHERE id = ?",
        (datetime.now(timezone.utc).isoformat(timespec="seconds"), batch_id))
    conn.commit()
    return report
