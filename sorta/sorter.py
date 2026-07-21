"""F5: sorting by moving files.

Contract: reads files/places/faces/face_clusters/events/event_files/media_class,
writes to move_batches/moves and to the FS. The only exception: after a successful
move, files.path is updated so the index stays valid; undo restores the old value.

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

The low_date rule: any mode's layout includes the year (YYYY), so a file without
taken_at or with taken_at_confidence='low' (a date only from mtime — often the copy
time, not the capture time) goes to _Unsorted/low_date/. The exception is
event mode (F5.1): a file that fell into an event (auto or manual) takes the year
from events.started_at, not from its own date, so low-confidence/undated files of
manual events are laid out under <event_year>/<name>/, not low_date; low_date for
event mode remains only as a fallback in case of an unparsable started_at (should
not happen — the column is NOT NULL, ISO).
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

def _resolve_excludes(cfg: Config, exclude: Sequence[str] | None) -> list[Path]:
    """Exclude directories from --exclude (repeatable) + config sort.exclude_dirs.

    Both sources are combined, paths are coerced to absolute resolved form for
    comparison by directory boundary (see _is_excluded).
    """
    dirs = list(exclude or [])
    dirs += list(cfg.sort.exclude_dirs)
    return [Path(d).resolve() for d in dirs]


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
    ISO cc / list of geonameid — so config-language folders (Россия/Москва) and
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
    row["city"] as-is.
    """
    if row["dedup_action"] == "to_delete":
        # U3b: an explicit user decision from the web app (sorta ui) — the highest
        # priority of all (city/junk/document/not_personal), the file goes to
        # _удалить regardless of the sort mode.
        return [i18n.folder("to_delete", lang)], "dedup_delete"
    if row["not_personal"]:
        # F17: a downloaded movie/series (release name, marked at indexing) — not
        # personal media, past the city/date/people layout, into a separate folder.
        return [i18n.folder("unsorted", lang), i18n.folder("not_personal", lang)], "not_personal"
    verdict = row["junk_verdict"]
    if verdict == "document":
        # F15: a photographed document — a separate review category (not junk), its
        # own top-level folder regardless of the sort mode.
        return [i18n.folder("documents", lang)], "document"
    if verdict == "product":
        # F37-B (deep VLM tier): an item for sale — its own review folder _Товары,
        # not junk and not a memory. Only with vlm_enabled (the fast tier gives no product).
        return [i18n.folder("products", lang)], "product"
    if verdict is not None and verdict != "photo":
        return [i18n.folder("unsorted", lang), i18n.folder("junk", lang),
               _sanitize(verdict)], "junk"
    if mode == "event":
        if event is None:
            # F30: the file did not fall into an event (a small group < min_event_size
            # or no event) → lay it out by date Year/month, not into a flat service
            # folder.
            year = _year_of(row["taken_at"], row["taken_at_confidence"])
            if year is None:
                return [i18n.folder("unsorted", lang), i18n.folder("low_date", lang)], "low_date"
            taken_at = row["taken_at"] or ""
            month = taken_at[5:7] if len(taken_at) >= 7 and taken_at[5:7].isdigit() else None
            return ([year, month] if month else [year]), "no_event"
        event_name, event_year = event
        year = event_year or _year_of(row["taken_at"], row["taken_at_confidence"])
        if year is None:
            return [i18n.folder("unsorted", lang), i18n.folder("low_date", lang)], "low_date"
        return [year, _sanitize(event_name)], "event"
    year = _year_of(row["taken_at"], row["taken_at_confidence"])
    if year is None:
        return [i18n.folder("unsorted", lang), i18n.folder("low_date", lang)], "low_date"
    if mode == "city":
        if row["city"] is None or (row["place_confidence"] or "unknown") == "unknown":
            return [i18n.folder("unsorted", lang), i18n.folder("no_place", lang)], "no_place"
        # online (G6): the full country name from Nominatim is already in the config
        # language; offline — localize the ISO cc via the curated dict i18n.country
        country_name = row["country_name"] or (
            i18n.country(row["country"], lang) if row["country"] else "Unknown")
        city_gid = row["city_geonameid"]
        city_name = resolver.name(city_gid, lang) if city_gid is not None else row["city"]
        parts = [_sanitize(country_name), _sanitize(city_name), year]
        district_gid = row["district_geonameid"]
        if district_gid is not None:
            # F49: a foreign transliterated district (no localized name in
            # names.tsv) is dropped — only Country/City/Year. RU and localized
            # foreign districts (Убуд/Кута) stay.
            if not drop_unlocalized_district or resolver.has_localized_name(district_gid, lang):
                parts.append(_sanitize(resolver.name(district_gid, lang)))
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

class TransferError(RuntimeError):
    """Transferring a single file failed; the caller marks the move failed."""


def _copy_and_verify(src: Path, dst: Path, expected_hash: str) -> None:
    """copy2 src -> dst, blake3 verify; on failure dst is deleted, TransferError."""
    try:
        shutil.copy2(src, dst)
    except OSError as exc:
        Path(dst).unlink(missing_ok=True)
        raise TransferError(f"копирование не удалось: {src} -> {dst}: {exc}") from None
    if file_hash(dst)[0] != expected_hash:
        dst.unlink(missing_ok=True)
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
    """
    size = src.stat().st_size
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise TransferError(f"dst уже существует, перезапись запрещена: {dst}")
    if link:
        try:
            os.link(src, dst)
        except OSError as exc:
            _log.warning("album: hardlink недоступен (%s), фолбэк на copy: %s -> %s",
                        exc, src, dst)
            _copy_and_verify(src, dst, src_hash or file_hash(src)[0])
    elif copy:
        _copy_and_verify(src, dst, src_hash or file_hash(src)[0])
    else:
        try:
            os.rename(src, dst)
        except OSError:
            _copy_and_verify(src, dst, src_hash or file_hash(src)[0])
            os.remove(src)
    if not dst.exists() or dst.stat().st_size != size:
        raise TransferError(f"проверка после перемещения не прошла: {dst}")


def _resolve_dst(target_dir: Path, src: Path, claimed: set[str]) -> tuple[Path, bool]:
    """dst without overwriting: suffixes _1, _2 against the disk and plan-claimed names.

    If the file is already at the target location (a repeated apply after an
    interruption) — returns (src, True): no move needed.
    """
    dst = target_dir / src.name
    if os.path.normcase(str(dst)) == os.path.normcase(str(src)):
        return src, True
    n = 0
    cand = dst
    while os.path.normcase(str(cand)) in claimed or cand.exists():
        n += 1
        cand = dst.with_name(f"{dst.stem}_{n}{dst.suffix}")
    claimed.add(os.path.normcase(str(cand)))
    return cand, False


# --- Plan and apply ---------------------------------------------------------

@dataclass
class PlanItem:
    file_id: int
    src: Path
    dst: Path
    in_place: bool
    target_rel: str            # path relative to dest, POSIX separators
    reason: str                # city|person|person_primary|person_shared|event
    #                            | no_place|no_faces|no_event|junk|low_date
    #                            | dedup_delete
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
    deleted: int = 0   # F14: --delete-worse-dupes, permanently deleted worse near-dups
    excluded: int = 0  # F16: files skipped because of --exclude/sort.exclude_dirs
    in_place: bool = False  # F28: dest not set explicitly — layout inside the source root


@dataclass
class UndoStats:
    batch_id: int = 0
    undone: int = 0
    missing: int = 0
    failed: int = 0


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
        src_hash, algo = file_hash(item.src)
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
                  progress: Callable[[int, int], None] | None = None) -> SortReport:
    """Build a layout plan; with apply=True move files with journaling.

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

    F14 (--dedupe): among near-duplicates (pHash, only in the current --where
    selection) the best by quality (width*height, then size) is sorted normally, the
    rest are moved to _Duplicates/ (reason near_dup). Requires a computed pHash —
    otherwise a hint and an empty plan (nothing is written to disk), like in
    `dupes --near`. delete_worse_dupes=True (only with dedupe) instead of moving to
    _Duplicates/ PERMANENTLY deletes the worse ones — not undoable via undo, the
    status in moves is 'deleted' (audit).
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
        print("ВНИМАНИЕ: --delete-worse-dupes БЕЗВОЗВРАТНО удаляет худшие почти-дубликаты "
              "(не подлежит откату через sorta undo)")
    in_place_run = dest is None
    if dest is None:
        if len(cfg.sources) != 1:
            raise ValueError(
                "in-place раскладка требует единственного источника; "
                "задайте --dest или оставьте один каталог в sources")
        dest = cfg.sources[0]
    if in_place_run and apply:
        print(f"ВНИМАНИЕ: --dest не задан — реструктурируется ИСХОДНОЕ дерево "
              f"в {Path(dest).resolve()} (in-place раскладка)")
    conn.create_function("casefold", 1, _sql_casefold, deterministic=True)
    resolver = GeoResolver()  # G3: lazy loading of bundled data on first access
    cond, params = parse_where(where or [], lang, resolver)
    dest = Path(dest).resolve()

    if dedupe:
        have_phash = conn.execute(
            "SELECT COUNT(*) FROM files WHERE phash IS NOT NULL").fetchone()[0]
        if not have_phash:
            print("pHash ещё не посчитан — запустите: sorta phash")
            placeholder = _report_dir(cfg) / "sort_plan_no_phash"
            return SortReport(mode=mode, dest=dest,
                              csv_path=placeholder.with_suffix(".csv"),
                              html_path=placeholder.with_suffix(".html"))

    rows = conn.execute(
        _CTE + f"""SELECT f.id, f.path, f.taken_at, f.taken_at_confidence,
               f.hash, f.hash_algo, f.not_personal, f.gps_lat, f.gps_lon,
               p.country, p.country_name, p.city, p.confidence AS place_confidence,
               p.city_geonameid, p.district_geonameid, p.district_name,
               mc.verdict AS junk_verdict, mc.source AS junk_source,
               dc.action AS dedup_action
           FROM files f
           LEFT JOIN places p ON p.file_id = f.id
           LEFT JOIN media_class mc ON mc.file_id = f.id
           LEFT JOIN dedup_choice dc ON dc.file_id = f.id
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

    persons_by_file = _load_persons(conn)
    events_by_file = _load_events(conn)

    row_targets: list[tuple[sqlite3.Row, list[str], str,
                           list[tuple[str, float]], tuple[str, str | None] | None]] = []
    for r in rows:
        persons = persons_by_file.get(r["id"], [])
        event = events_by_file.get(r["id"])
        parts, reason = _target_parts(mode, strategy, r, persons, event, lang, resolver,
                                      drop_unlocalized_district)
        row_targets.append((r, parts, reason, persons, event))

    near_dup_best: dict[int, int] = {}
    near_dup_worse: dict[int, int] = {}
    if dedupe:
        # junk/document/not_personal/dedup_delete are excluded from grouping BEFORE
        # picking the best: otherwise such a file with a higher resolution could
        # "win" the group and pull a normal photo into _Duplicates instead of its
        # usual layout (they are all sorted separately, independent of near-dups);
        # dedup_delete — an explicit manual user decision (U3b), it must not pull a
        # near-dup group onto itself.
        sortable_ids = {r["id"] for r, _parts, reason, _p, _e in row_targets
                        if reason not in ("junk", "document", "not_personal", "dedup_delete")}
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
        dst, in_place = _resolve_dst(dest.joinpath(*parts), src, claimed)
        try:
            target_rel = dst.relative_to(dest).as_posix()
        except ValueError:  # only on a path-case divergence on Windows
            target_rel = dst.as_posix()
        plan.append(PlanItem(
            file_id=r["id"], src=src, dst=dst, in_place=in_place,
            target_rel=target_rel, reason=reason,
            taken_at=r["taken_at"], taken_at_confidence=r["taken_at_confidence"],
            country=r["country"], city=r["city"],
            place_confidence=r["place_confidence"],
            gps_lat=r["gps_lat"], gps_lon=r["gps_lon"],
            persons=[label for label, _area in persons],
            event=event[0] if event else None,
            junk_verdict=r["junk_verdict"], junk_source=r["junk_source"],
            db_hash=r["hash"], db_algo=r["hash_algo"],
            near_dup_group=near_dup_group, near_dup_role=near_dup_role))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"sort_plan_{mode}_{stamp}"
    csv_path = _report_dir(cfg) / f"{stem}.csv"
    html_path = csv_path.with_name(f"{stem}.html")
    _write_plan_csv(csv_path, plan)
    thumb_workers = cfg.sort.thumbnail_workers or min(8, os.cpu_count() or 4)
    _write_plan_html(html_path, plan, dest, thumbnails=thumbnails,
                     thumbnail_workers=thumb_workers)
    report = SortReport(mode=mode, dest=dest, csv_path=csv_path, html_path=html_path,
                        plan=plan, dirs=len({it.dst.parent for it in plan}),
                        excluded=excluded_count, in_place=in_place_run)
    excluded_note = f"; исключено: {excluded_count}" if excludes else ""
    print(f"sort --by {mode}{' --apply' if apply else ' (dry-run)'}: "
          f"{len(plan)} файлов -> {report.dirs} каталогов; план: {csv_path}, {html_path}"
          f"{excluded_note}")
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
        if progress:
            progress(i, len(plan))
        if item.in_place:
            report.skipped_in_place += 1
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
                item.src.unlink()
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
         progress: Callable[[int, int], None] | None = None) -> UndoStats:
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
        "SELECT id, file_id, src, dst, hash FROM moves "
        "WHERE batch_id = ? AND status = 'done' ORDER BY id DESC",
        (batch_id,)).fetchall()
    stats = UndoStats(batch_id=batch_id)
    for i, r in enumerate(rows, 1):
        if progress:
            progress(i, len(rows))
        src, dst = Path(r["src"]), Path(r["dst"])
        if not dst.exists():
            _log.warning("undo: dst отсутствует, статус остаётся 'done': %s", dst)
            stats.missing += 1
            continue
        if operation in ("copy", "link"):
            try:
                dst_hash = file_hash(dst)[0]
            except OSError as exc:
                _log.warning("undo: копия недоступна, пропуск: %s (%s)", dst, exc)
                stats.failed += 1
                continue
            if dst_hash != r["hash"]:
                _log.warning("undo: хэш копии не совпал, копия НЕ удалена: %s", dst)
                stats.failed += 1
                continue
            dst.unlink()
            conn.execute("UPDATE moves SET status = 'undone' WHERE id = ?", (r["id"],))
            conn.commit()
            stats.undone += 1
            continue
        restore, n = src, 0
        while restore.exists():
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
    return stats


# --- F34: album engine (export a person/event slice into a named folder) ----------
#
# An album is a targeted export of an index slice (not a full layout): all canonical
# files of a person (accounting for cluster merges, F31) or an event, optionally
# narrowed by --where, flat into dest/<album_name>/. The base city layout is not
# touched; an album is an additional "view" (link/copy) or, on explicit request, a
# removal from the pool (move). Journal/undo — the shared move_batches/moves
# mechanism, operation='link'|'copy'|'move' (undo for 'link' — see _transfer/undo above).

ALBUM_KINDS = ("person", "event")
ALBUM_MODES = ("link", "copy", "move")


@dataclass
class AlbumPlanItem:
    file_id: int
    src: Path
    dst: Path
    persons: list[str]     # labels of all named people on the file (for the move check)
    multi_person: bool     # len(persons) >= 2 — with mode='move' such a file is blocked


@dataclass
class AlbumReport:
    kind: str               # person | event
    selector: str            # as passed by the caller (person name / event name|id)
    album_name: str          # the final folder name (before _sanitize)
    dest: Path               # dest/<album_name> (already resolved)
    mode: str                # link | copy | move
    plan: list[AlbumPlanItem] = field(default_factory=list)
    batch_id: int | None = None
    transferred: int = 0
    failed: int = 0
    blocked_multi: int = 0   # mode='move': files skipped due to multi-membership


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
               album_name: str | None = None) -> AlbumReport:
    """Build an album export plan; with apply=True materialize it (link/copy/move).

    kind='person': selector — a person's name; the slice = canonical files (dup_of IS
    NULL) that have a face in a cluster whose merged_into chain root (F31, via the
    shared _CTE/_person_files) has label==selector (casefold).
    kind='event': selector — an event name OR id; the slice = the event(s)' event_files.
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
    """
    if kind not in ALBUM_KINDS:
        raise ValueError(f"неизвестный тип альбома {kind!r}; допустимы: {', '.join(ALBUM_KINDS)}")
    if mode not in ALBUM_MODES:
        raise ValueError(f"неизвестный режим альбома {mode!r}; допустимы: {', '.join(ALBUM_MODES)}")
    conn.create_function("casefold", 1, _sql_casefold, deterministic=True)

    subject_params: list[str | int]
    if kind == "person":
        resolved_name = selector
        subject_cond = ("f.id IN (SELECT file_id FROM _person_files "
                        "WHERE casefold(label) = casefold(?))")
        subject_params = [selector]
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
        _CTE + f"""SELECT f.id, f.path FROM files f
               LEFT JOIN places p ON p.file_id = f.id
               WHERE f.dup_of IS NULL AND f.error IS NULL AND {full_cond}
               ORDER BY f.path""", full_params).fetchall()

    final_name = album_name or resolved_name
    album_dir = Path(dest).resolve() / _sanitize(final_name)
    report = AlbumReport(kind=kind, selector=selector, album_name=final_name,
                         dest=album_dir, mode=mode)

    if not rows:
        print(f"album {kind} {selector!r}: срез пуст, ничего не выгружено")
        return report

    persons_by_file = _load_persons(conn)
    if mode == "move":
        print("ВНИМАНИЕ: --move изымает файлы альбома из общего пула сортировки "
             "(канон города/другие альбомы больше не увидят эти файлы)")

    claimed: set[str] = set()
    for r in rows:
        src = Path(r["path"])
        persons = [label for label, _area in persons_by_file.get(r["id"], [])]
        dst, _in_place = _resolve_dst(album_dir, src, claimed)
        report.plan.append(AlbumPlanItem(file_id=r["id"], src=src, dst=dst,
                                         persons=persons, multi_person=len(persons) >= 2))

    if mode == "move":
        blocked = [it for it in report.plan if it.multi_person]
        if blocked:
            names = ", ".join(str(it.src) for it in blocked[:5])
            more = " …" if len(blocked) > 5 else ""
            print(f"ВНИМАНИЕ: {len(blocked)} файл(ов) с 2+ названными людьми на кадре — "
                 f"move для них заблокирован (неясно, чей это альбом), используйте "
                 f"--link/--copy: {names}{more}")

    print(f"album {kind} {selector!r}{' --apply' if apply else ' (dry-run)'} "
         f"[{mode}]: {len(report.plan)} файлов -> {album_dir}")

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
        try:
            src_hash, _algo = file_hash(item.src)
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
