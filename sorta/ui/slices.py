"""F182: the "Slices" tab — the queries, the pins and the built-in slices.

A slice is a question asked of the index. None of them moves a file — a slice is
hardlinks, free to make and to drop — which is what keeps them out of `layout`.
"""
from __future__ import annotations

import dataclasses
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .. import imaging
from ..config import Config, SavedSlice
from ..detect import detector_settings
from ..junk import faces_stage_ran, search_index_model, search_index_settings
from ..landmarks import batched
from ..naming import naming_settings
from ..search import (
    REASON_EMPTY, REASON_OTHER_MODEL, EmbeddingsMissing, TextEncoder, match_person, person_page,
    rank_queries, rank_text, text_encoder,
)
from ..sorter import (
    CLASS_ALBUM_KINDS, FACE_SLICES, Destination, animal_auto_sql, animal_ids_sql,
    face_slice_ids_sql,
)
from .common import (
    _OVERVIEW_LIVE, _connect, _destination_json, _destinations_for, _page_payload,
    _parse_page_window, _validate_file_ids_payload,
)
from .layout import _overrides_map


# --- F103: the "Utility frames" slice ------------------------------------------------
# The deep VLM tier carries away roughly every tenth frame into service folders (2 202
# `product` alone on the live 24k run), so "a handful of wrong verdicts" is dozens of
# frames nobody could find. This view reclassifies nothing: the fix is a row in
# `manual_overrides` (F77), and `media_class` keeps whatever the model measured.

# A `document` verdict is never decoded for display — a preview is a derived copy of the
# contents. Returning one to the photos is still allowed: the person knows what is in
# their own file, they just do not need it rendered to decide.
# F133: WHICH classes those are is a config question — `vlm.exclude_classes` already
# carries the list and defaults to `["document"]`, and one visible list of sensitive
# classes beats two. The tuple below is only the fallback for a caller that passes
# nothing: a privacy guard must never switch itself off through an omission.
_JUNK_NO_PREVIEW = ("document",)


# --- F193: one answer to "may this slice be gathered into a folder" -------------------
# Every bucket carries an album row, and a bucket that may not be gathered carries the
# REASON instead of a button — a hidden control forbids nothing (the F133 lesson), and a
# request sent past the interface would have gathered the folder all the same. One
# function decides it, read by both the payload and the route: two spellings of "this
# class is private" is how a guard grows a hole.
#
# `document` is not an album kind and emptying `vlm.exclude_classes` does not make one:
# that key decides what is SHOWN, never that a folder of somebody's passports may be
# assembled in one click. If the owner decides the other way, this tuple is the change.
_NEVER_ALBUM_CLASSES = ("document",)

# Codes rather than sentences: the reason has to be read in the interface language, so
# the server sends the word and the catalog holds the sentences (the F156 `_PIN_*`
# arrangement). Four and not one because the remedies differ — one is an edit of a config
# key, one is a decision nobody has taken, and two are not remedies at all.
_ALBUM_BLOCKED_DOCUMENTS = "documents"      # never gathered, whatever the key says
_ALBUM_BLOCKED_SENSITIVE = "sensitive"      # the class sits in `vlm.exclude_classes`
_ALBUM_BLOCKED_NO_KIND = "no_kind"          # a bucket the album engine has no kind for
_ALBUM_BLOCKED_ALL_BUCKETS = "all_buckets"  # the "everything non-photo" view, not a slice


def _album_refusal(kind: object, sensitive: frozenset[str]) -> str | None:
    """Why this media class has no album, or None when it has one.

    Only a CLASS can be refused here; everything else is None and travels on to the
    ordinary validation. That is what lets the route ask this of a raw request body
    before it knows the body is well-formed: `document` has to come back with "documents
    are never gathered", not the "invalid body" a name outside `ALBUM_KINDS` earns.
    """
    if not isinstance(kind, str):
        return None
    if kind in _NEVER_ALBUM_CLASSES:
        return _ALBUM_BLOCKED_DOCUMENTS
    if kind in sensitive:
        return _ALBUM_BLOCKED_SENSITIVE
    return None


def class_album_refusal(cfg: Config, kind: object) -> str | None:
    """`_album_refusal` against the LIVE `vlm.exclude_classes` — what the route asks.

    Read per request: the settings panel can change the key without a restart, and a
    guard reading a startup value would be a guard about a configuration nobody runs.
    """
    return _album_refusal(kind, frozenset(cfg.vlm.exclude_classes))


def _bucket_album(bucket: str | None,
                  sensitive: frozenset[str]) -> tuple[str | None, str | None]:
    """(the album kind of this junk bucket, the reason there is none) — never both None.

    `_ALBUM_BLOCKED_NO_KIND` is the one that matters for tomorrow: a class added to the
    classifier without an album kind would otherwise fall silently out of the interface.
    """
    if bucket is None:
        return None, _ALBUM_BLOCKED_ALL_BUCKETS
    refusal = _album_refusal(bucket, sensitive)
    if refusal is not None:
        return None, refusal
    if bucket not in CLASS_ALBUM_KINDS:
        return None, _ALBUM_BLOCKED_NO_KIND
    return bucket, None


# F193: the frames a person ticked. `POST /api/album` takes them next to the kind, and the
# rule is the same for every slice — the list narrows the slice (`sorter.plan_album`) and
# an empty one is a refusal rather than an album of nothing.
_ALBUM_NO_SELECTION = "empty_selection"


def album_selection(payload: object) -> tuple[list[int] | None, str | None]:
    """(the ticked ids or None, the refusal code or None) for `POST /api/album`.

    An ABSENT `file_ids` is not a refusal: it is how the whole slice is gathered. An
    EMPTY list is `_ALBUM_NO_SELECTION` — a client that has a selection holding nothing,
    and a folder of zero files would read as "this slice is empty", which is a statement
    about somebody's archive that is not true. Anything else malformed is a plain 400.
    """
    if not isinstance(payload, dict) or payload.get("file_ids") is None:
        return None, None
    raw = payload.get("file_ids")
    if isinstance(raw, list) and not raw:
        return None, _ALBUM_NO_SELECTION
    ids = _validate_file_ids_payload(payload)
    if ids is None:
        return None, "invalid"
    return ids, None


def _junk_item_to_json(row: sqlite3.Row, restored: bool,
                       no_preview: frozenset[str] = frozenset(_JUNK_NO_PREVIEW),
                       dest: Destination | None = None) -> dict:
    """One card of the junk view. `thumb_url` is ABSENT for a no-preview bucket."""
    path = Path(row["path"])
    verdict = row["verdict"]
    payload = {
        "file_id": int(row["id"]),
        "verdict": verdict,
        "name": path.name,
        "date": row["taken_at"],
        # F175: said out loud rather than inferred from the missing `thumb_url` — a card
        # the person must not delete has to be visible AS one before "select everything"
        # is pressed, and the inference would be a second copy of the privacy rule in JS.
        "sensitive": verdict in no_preview,
        # F77/F103: the frame already carries a manual "this is a photo" correction —
        # the card says so instead of offering the same action twice.
        "restored": restored,
        # F174: where the frame lands if it IS returned — the folder the plan will build
        # once the `photo` mark is in the table, not one named by this file.
        **_destination_json(dest),
    }
    if verdict not in no_preview:
        payload["thumb_url"] = f"/thumb/{int(row['id'])}"
        payload["video"] = imaging.is_video_path(path)
    return payload


# F171: the order INSIDE one bucket — the model's own estimate, most confident first.
# NULL means "no estimate", never "unsure", so those frames keep the path order at the
# END of the list rather than sinking to a score they were never given. `f.path` is
# unique and already breaks every tie, so no id is needed. Never applied to the "all"
# view: four classes are four separate softmaxes, and an order across them would be a
# comparison nobody measured.
_JUNK_ORDER = "(mc.score IS NULL), mc.score DESC, f.path"


def _junk_payload(db_path: Path, cfg: Config, bucket: str | None,
                  offset: int, limit: int,
                  sensitive: frozenset[str] = frozenset(_JUNK_NO_PREVIEW)) -> dict:
    """`GET /api/junk` — the buckets with their counts + one page of one bucket.

    F133: a class in `sensitive` (= `vlm.exclude_classes`) keeps its counter, its cards
    and the way back to the photos, and loses exactly one thing: `thumb_url`. Enforced
    HERE rather than in the markup — a card the browser was given a preview link for is
    a card whose contents have already been decoded and sent.

    The `<> 'photo'` guard sits in the query itself, not in the parameter check, so no
    value of `bucket` can turn this route into a way of listing personal photos.
    `bucket=None` is every non-photo frame; an unknown bucket is an empty page, not an
    error. `buckets` is always the full set of counters, independent of the filter;
    `total` is the size of the CURRENT selection.

    F139/F193: exactly one of `album_kind` and `album_blocked` is set, always, so the row
    over the grid is drawn either way. The server decides, because a client working it
    out for itself would be a second copy of the privacy rule.

    F171: `ordered_by_score` says whether the page really is the ranking `_JUNK_ORDER`
    promises, so the caption promises one exactly where there is one.
    """
    conn = _connect(db_path)
    try:
        counts = conn.execute(
            """SELECT mc.verdict AS verdict, COUNT(*) AS n
               FROM files f JOIN media_class mc ON mc.file_id = f.id
               WHERE mc.verdict <> 'photo' AND f.dup_of IS NULL AND f.error IS NULL
               GROUP BY mc.verdict"""
        ).fetchall()
        params: list[object] = []
        clause = ""
        if bucket is not None:
            clause = " AND mc.verdict = ?"
            params.append(bucket)
        total = conn.execute(
            f"""SELECT COUNT(*) FROM files f JOIN media_class mc ON mc.file_id = f.id
                WHERE mc.verdict <> 'photo' AND f.dup_of IS NULL AND f.error IS NULL
                      {clause}""", params).fetchone()[0]
        scored = 0 if bucket is None else int(conn.execute(
            f"""SELECT COUNT(*) FROM files f JOIN media_class mc ON mc.file_id = f.id
                WHERE mc.verdict <> 'photo' AND f.dup_of IS NULL AND f.error IS NULL
                      AND mc.score IS NOT NULL {clause}""", params).fetchone()[0])
        rows = conn.execute(
            f"""SELECT f.id, f.path, f.taken_at, mc.verdict
                FROM files f JOIN media_class mc ON mc.file_id = f.id
                WHERE mc.verdict <> 'photo' AND f.dup_of IS NULL AND f.error IS NULL
                      {clause}
                ORDER BY {_JUNK_ORDER if bucket is not None else 'f.path'}
                LIMIT ? OFFSET ?""", [*params, limit, offset]).fetchall()
        # F174: asked with the correction the button writes already assumed, so the
        # caption names the city the frame goes back to and not the service folder it is
        # sitting in right now.
        dests = _destinations_for(cfg, conn, rows, "photo")
    finally:
        conn.close()
    marks = _overrides_map(db_path) if rows else {}
    buckets = [{"verdict": r["verdict"], "count": int(r["n"])} for r in counts]
    buckets.sort(key=lambda b: (-b["count"], b["verdict"]))
    album_kind, album_blocked = _bucket_album(bucket, sensitive)
    return {
        "bucket": bucket,
        "buckets": buckets,
        "album_kind": album_kind,
        "album_blocked": album_blocked,
        # A card without `thumb_url` is a class the server refuses to render, not a
        # preview that failed to build, and the two need different words on the screen.
        "sensitive": sorted(sensitive),
        # F171: `False` for the "all" view, and for a bucket the classifier settled
        # without a number of its own — a heuristics-only run, or the frames the deep
        # tier rewrote, both of which store NULL rather than a confidence.
        "ordered_by_score": bool(scored),
        "total": int(total),
        "offset": offset,
        "limit": limit,
        "items": [
            _junk_item_to_json(
                r, (marks.get(int(r["id"])) or ("", None))[0] == "photo", sensitive,
                dests.get(int(r["id"])))
            for r in rows
        ],
    }


# --- F123: the "Animals" tab — the pet verdicts of the frame-quality stage ----------
# Calibrated in F122: 805 frames of the live collection at 92% precision. The order is by
# CONFIDENCE, not by path — about 64 of those 805 are not animals, and reading top-down
# until the quality runs out is how a person finds where that border sits, so the score
# travels to the card.


def _animal_item_to_json(row: sqlite3.Row, dest: Destination | None = None) -> dict:
    """One card of the animal view: a thumbnail, a name, a date and the pet score.

    F124: `is_animal` comes straight out of the shared rule, never recomputed here in
    Python, and `manual` says whether a person decided it. A frame the user has taken
    the mark off stays on the card, struck through: it must be visible as marked BY
    HAND, or the counter moves for no reason anybody can see and cannot be taken back.
    """
    path = Path(row["path"])
    return {
        "file_id": int(row["id"]),
        "name": path.name,
        "date": row["taken_at"],
        # A payload that pretended 0.0 was measured would lie about exactly the number
        # this tab exists to show.
        "score": None if row["pet_score"] is None else float(row["pet_score"]),
        "is_animal": bool(row["is_animal"]),
        "manual": None if row["manual"] is None else bool(row["manual"]),
        "thumb_url": f"/thumb/{int(row['id'])}",
        "video": imaging.is_video_path(path),
        # F174: where the frame ALREADY lies. The mark changes a membership and moves no
        # file, and the card can only say so convincingly by naming that folder.
        **_destination_json(dest),
    }


_ANIMALS_JOIN = ("FROM files f LEFT JOIN frame_quality fq ON fq.file_id = f.id "
                 "LEFT JOIN manual_pet mp ON mp.file_id = f.id")


# F160: the helpers of this slice take the WHOLE live config and resolve the detector's
# two switches (`detect.enabled` and the model that wrote the boxes) through the one
# function that ANDs them. Reading half of the pair is the mistake F145 was written
# about, in either direction.
def _animals_population(cfg: Config) -> str:
    """What the TAB LISTS: the model's marks plus every frame a person has touched.

    Deliberately wider than the slice — a frame marked "not an animal" is no longer in
    the album and is still on this page, struck through, because a card that vanishes
    takes the undo button with it. F137: "the model's marks" is the automatic half of
    the shared rule, not the `frame_quality.pet` cache, so a threshold edit takes frames
    off this page too.
    """
    return (f"({animal_auto_sql(cfg.features, 'fq', detector_settings(cfg))} "
            "OR mp.file_id IS NOT NULL) AND f.dup_of IS NULL AND f.error IS NULL")


def _animals_count_sql(cfg: Config) -> str:
    """What COUNTS as an animal: `sorter.animal_ids_sql` and nothing else, over the
    canonical, readable files. Used by this tab and by the "Overview" number, so the two
    cannot disagree with the album or with each other."""
    ids = animal_ids_sql(cfg.features, detector_settings(cfg))
    return f"""SELECT COUNT(*) FROM files f
    WHERE f.dup_of IS NULL AND f.error IS NULL AND f.id IN ({ids})"""


def _animals_select(cfg: Config) -> str:
    """One row shape: the page and the answer to a mark are the same SELECT, so a card
    redrawn after an edit says what the same card would say on a reload."""
    ids = animal_ids_sql(cfg.features, detector_settings(cfg))
    return f"""SELECT f.id, f.path, f.taken_at, fq.pet_score,
           mp.is_animal AS manual, f.id IN ({ids}) AS is_animal
    {_ANIMALS_JOIN}"""


def _animals_payload(db_path: Path, cfg: Config, offset: int, limit: int) -> dict:
    """`GET /api/animals` — one page of the animal slice, most confident first.

    Two numbers, two questions: `total` is the length of the LIST the paging walks
    (model marks plus manual decisions), `animals` is how many frames count as animals
    by the shared rule — the number "Overview" shows and the album gathers.

    The id breaks ties in the order, so equal scores keep a stable place between pages
    instead of being shown twice or never. A manual decision does NOT move a card: a
    list that reshuffles under the frame just marked is one nobody can finish reading.

    `cfg` is the LIVE config (F137): the thresholds this page is drawn with are the ones
    in force at the moment of the request, not the ones some run wrote into the database.
    """
    population = _animals_population(cfg)
    conn = _connect(db_path)
    try:
        total = conn.execute(
            f"SELECT COUNT(*) {_ANIMALS_JOIN} WHERE {population}").fetchone()[0]
        animals = conn.execute(_animals_count_sql(cfg)).fetchone()[0]
        rows = conn.execute(
            f"""{_animals_select(cfg)}
                WHERE {population}
                ORDER BY fq.pet_score DESC, f.id
                LIMIT ? OFFSET ?""", (limit, offset)).fetchall()
        # F174: no assumed correction — the mark changes a membership and not a route,
        # so where the frame lies now is where it will lie after it.
        dests = _destinations_for(cfg, conn, rows)
    finally:
        conn.close()
    return {
        "animals": int(animals),
        **_page_payload([_animal_item_to_json(r, dests.get(int(r["id"])))
                         for r in rows],
                        total=int(total), offset=offset, limit=limit),
    }


# F124: `clear` drops the row and hands the frame back to the automatic verdict, which is
# not the same as `not_animal` — the difference is why the row is two-valued rather than
# a presence flag.
_ANIMAL_MARK_ACTIONS = ("animal", "not_animal", "clear")


def _validate_animal_mark_payload(payload: object) -> tuple[list[int], str] | None:
    """Parse the body `POST /api/animals/mark`:
    `{"file_ids": [int,...], "action": "animal"|"not_animal"|"clear"}`.

    None -> invalid (400). The ids go through the same `_validate_file_ids_payload` as
    every other write route — ints only, never a path.
    """
    if not isinstance(payload, dict):
        return None
    ids = _validate_file_ids_payload(payload)
    if ids is None:
        return None
    action = payload.get("action")
    if action not in _ANIMAL_MARK_ACTIONS:
        return None
    return ids, action


def _apply_animal_mark(db_path: Path, cfg: Config,
                       ids: list[int], action: str) -> dict:
    """Write the user's verdict into `manual_pet`; answer with the redrawn cards.

    One row per file, so marking the same frame twice overwrites. Nothing here touches
    `frame_quality`: the model's own table keeps being recomputed from scratch and this
    mark is read on top of it (`sorter.animal_ids_sql`). An id outside the current index
    is skipped rather than written.

    The answer carries the redrawn cards and the fresh count so the client can update in
    place — a reload would send a reader walking down a confidence ranking back to the
    first screen after every decision. `items` may come back SHORTER than the ids (a
    `clear` on a frame the model never marked leaves the list), and those cards are
    dropped.
    """
    now = datetime.now(timezone.utc).isoformat()
    count_sql = _animals_count_sql(cfg)
    conn = _connect(db_path)
    try:
        placeholders = ",".join("?" * len(ids))
        known = [int(r["id"]) for r in conn.execute(
            f"SELECT id FROM files WHERE id IN ({placeholders})", ids).fetchall()]
        if not known:
            return {"marked": 0, "items": [],
                    "animals": int(conn.execute(count_sql).fetchone()[0])}
        known_placeholders = ",".join("?" * len(known))
        with conn:
            if action == "clear":
                conn.execute(
                    f"DELETE FROM manual_pet WHERE file_id IN ({known_placeholders})",
                    known)
            else:
                is_animal = 1 if action == "animal" else 0
                conn.executemany(
                    """INSERT INTO manual_pet (file_id, is_animal, updated_at)
                       VALUES (?, ?, ?)
                       ON CONFLICT(file_id) DO UPDATE SET
                           is_animal = excluded.is_animal,
                           updated_at = excluded.updated_at""",
                    [(fid, is_animal, now) for fid in known])
        rows = conn.execute(
            f"""{_animals_select(cfg)}
                WHERE {_animals_population(cfg)}
                  AND f.id IN ({known_placeholders})""",
            known).fetchall()
        animals = conn.execute(count_sql).fetchone()[0]
        # F174: a caption that vanished on the first click would look like the mark had
        # moved the file after all.
        dests = _destinations_for(cfg, conn, rows)
    finally:
        conn.close()
    return {
        "marked": len(known),
        "animals": int(animals),
        "items": [_animal_item_to_json(r, dests.get(int(r["id"]))) for r in rows],
    }


# --- F152: the face slices — with people / group photos / portraits ----------------
# The three largest populations of the archive (people are 27.5% of a hand-labelled
# sample of 200 frames), off a signal already on disk: 12 952 real faces over 7 341
# photographs. The rules live in `sorter.face_slice_ids_sql`, one copy, so the album,
# this panel and the "Overview" counters talk about one collection.
#
# Membership here is a fact of a detector's output, not a place in a ranking, so there is
# no score to show. And without a faces run the counters travel as `null` with a
# `reason`, never as zeros (the F125 rule): a zero would read as "no photograph of yours
# has a person on it", a conclusion drawn from a table nobody filled.

# `media_type` is not filtered: the faces stage only ever writes rows for photographs.
_FACE_LIVE = "f.dup_of IS NULL AND f.error IS NULL"

# `media_class` rides along for the F133 privacy rule alone — a frame of a sensitive
# class is listed but never given a `thumb_url`.
_FACE_FROM = "FROM files f LEFT JOIN media_class mc ON mc.file_id = f.id"

# The same `bbox != '[]'` rule the slices themselves are built on.
_FACE_COUNT_SQL = ("(SELECT COUNT(*) FROM faces fa WHERE fa.file_id = f.id "
                   "AND fa.bbox != '[]')")


def _face_slice_where(cfg: Config, slice_: str) -> tuple[str, list[object]]:
    """The WHERE of one face slice + its parameters, over the canonical population."""
    ids_sql, params = face_slice_ids_sql(cfg, slice_)
    return f"{_FACE_LIVE} AND f.id IN ({ids_sql})", params


def _face_slice_count(conn: sqlite3.Connection, cfg: Config, slice_: str) -> int:
    """How many frames one face slice holds, under the WHERE its page uses."""
    where, params = _face_slice_where(cfg, slice_)
    return int(conn.execute(
        f"SELECT COUNT(*) FROM files f WHERE {where}", params).fetchone()[0])


def _face_item_to_json(row: sqlite3.Row, sensitive: frozenset[str]) -> dict:
    """One card: a thumbnail, a name, a date and how many faces the frame holds.

    No score, because there is none to invent: the frame is in the slice because a box
    was found on it. The face count is what makes the group slice checkable by eye.
    """
    path = Path(row["path"])
    payload = {
        "file_id": int(row["id"]),
        "name": path.name,
        "date": row["taken_at"],
        "faces": int(row["faces"]),
    }
    verdict = row["verdict"]
    if verdict is None or str(verdict) not in sensitive:
        payload["thumb_url"] = f"/thumb/{int(row['id'])}"
        payload["video"] = imaging.is_video_path(path)
    return payload


def _face_slices_payload(cfg: Config, db_path: Path, slice_: str, offset: int,
                         limit: int, sensitive: frozenset[str]) -> dict:
    """`GET /api/face-slices` — the three counters + one bounded page of the current one.

    `counts` is always the full set, and every entry is `null` when the faces stage has
    not run: unmeasured, not zero, and `reason` says which. Once the stage has run a zero
    IS the answer. `ORDER BY f.id` because these slices have no ranking of their own and
    index order is stable, which is what paging needs.
    """
    conn = _connect(db_path)
    try:
        ran = faces_stage_ran(conn)
        counts: dict[str, int | None] = {name: None for name in FACE_SLICES}
        items: list[dict] = []
        total = 0
        if ran:
            for name in FACE_SLICES:
                counts[name] = _face_slice_count(conn, cfg, name)
            where, params = _face_slice_where(cfg, slice_)
            total = int(counts[slice_] or 0)
            rows = conn.execute(
                f"""SELECT f.id, f.path, f.taken_at, mc.verdict AS verdict,
                           {_FACE_COUNT_SQL} AS faces
                    {_FACE_FROM} WHERE {where}
                    ORDER BY f.id LIMIT ? OFFSET ?""",
                [*params, limit, offset]).fetchall()
            items = [_face_item_to_json(r, sensitive) for r in rows]
    finally:
        conn.close()
    return {
        "slice": slice_,
        "counts": [{"slice": name, "count": counts[name]} for name in FACE_SLICES],
        "reason": None if ran else "no_faces_run",
        # The thresholds travel with the answer so the hint above the grid states the
        # rule the numbers came from instead of repeating a default in JS.
        "group_min": int(cfg.features.group_photo_faces),
        "portrait_share": float(cfg.features.portrait_face_share),
        **_page_payload(items, total=total, offset=offset, limit=limit),
    }


def _parse_face_slice_query(query: dict[str, list[str]]) -> tuple[str, int, int] | None:
    """(slice, offset, limit) for `GET /api/face-slices`, or None -> 400.

    An unknown slice is refused rather than answered with an empty page — the
    `_parse_review_query` rule.
    """
    window = _parse_page_window(query)
    if window is None:
        return None
    slice_ = ((query.get("slice") or [FACE_SLICES[0]])[0].strip() or FACE_SLICES[0])
    if slice_ not in FACE_SLICES:
        return None
    return slice_, window[0], window[1]


def _parse_junk_query(query: dict[str, list[str]]) -> tuple[str | None, int, int] | None:
    """(bucket, offset, limit) for `GET /api/junk`, or None -> 400.

    An empty/absent `bucket` means "every non-photo frame".
    """
    window = _parse_page_window(query)
    if window is None:
        return None
    raw_bucket = (query.get("bucket") or [""])[0].strip()
    return (raw_bucket or None), window[0], window[1]


# --- F156: why a built-in slice is empty --------------------------------------------
# The F125 rule (NULL is "not asked", not "no") applied to a whole slice. Each of the
# three exact slices answers with one of three things, never with a bare emptiness:
#
#   None          the slice holds photographs
#   not_run       the stage that fills it never ran — the interface links to the run
#                 screen, which is where that is fixed
#   none_found    the stage ran over this collection and there is nothing of the kind
#
# Two reasons and not one: only one of them is a fact about the person's photographs,
# and only the other has an action attached. `not_run` is also what a stage SWITCHED OFF
# looks like (`features.pets: false`), and that is right rather than a compromise — the
# run screen holds that checkbox, so the sentence and the link lead to the same place.
_SLICE_NOT_RUN = "not_run"
_SLICE_NONE_FOUND = "none_found"


def _tabs_visibility_payload(db_path: Path, cfg: Config) -> dict[str, object]:
    """F54: visibility of the "People"/"Events"/"Animals" tabs, by data presence.

    Light EXISTS queries, never the full payload. person ⇔ a faces row with a non-empty
    cluster_id; event ⇔ non-empty `events`; animal (F123) ⇔ some frame counts as one.

    F156: ...or the slice has something to SAY. A slice whose stage never ran appears
    too, because its emptiness is a sentence with a link in it and a pin that hides
    itself never gets to say it; `reasons` is which of the two empty states it is in. A
    slice that ran and found nothing keeps hiding — there the zero IS the fact.

    The animal question is asked of what the tab would LIST and not of what it would
    count: a user who has taken the mark off every frame emptied the slice but not the
    tab, and the tab is where the undo button lives. F137 is why it is that expression
    rather than "some `frame_quality.pet` is set" — the cache column can claim a verdict
    the thresholds in force have withdrawn.

    F152: `face` is deliberately NOT asked of its own data. The three face slices appear
    as soon as the index holds a photograph the stage could have looked at, because
    without a run they have to be able to SAY there was none.

    `indexed` rides along for the same cost: "re-run the selected stage" only makes sense
    over files that exist, and right after "Start over" ticking "faces" used to light the
    button up over nothing at all.
    """
    conn = _connect(db_path)
    try:
        person = bool(conn.execute(
            "SELECT EXISTS(SELECT 1 FROM faces WHERE cluster_id IS NOT NULL)"
        ).fetchone()[0])
        event = bool(conn.execute(
            "SELECT EXISTS(SELECT 1 FROM events)"
        ).fetchone()[0])
        animal = bool(conn.execute(
            f"SELECT EXISTS(SELECT 1 {_ANIMALS_JOIN} "
            f"WHERE {_animals_population(cfg)})"
        ).fetchone()[0])
        face = bool(conn.execute(
            "SELECT EXISTS(SELECT 1 FROM files WHERE dup_of IS NULL AND error IS NULL "
            "AND media_type = 'photo')"
        ).fetchone()[0])
        indexed = bool(conn.execute(
            "SELECT EXISTS(SELECT 1 FROM files)"
        ).fetchone()[0])
        # F156: each EXISTS asks whether the STAGE left anything behind — not whether the
        # slice came out non-empty, which is answered above.
        #
        # faces: a real box (`faces_stage_ran` excludes the "processed, none here" marker
        #   row). events: the stage groups every canonical frame carrying a date, so its
        #   own output is the only marker — with no dated frames there was nothing to
        #   group. animals: a STORED `pet_score`, which the stage writes whether or not
        #   it reached the threshold and never writes with `features.pets` off. A fact of
        #   the table rather than of the switch as it stands now (F137): the question is
        #   what was asked of THIS collection, and the switch may have moved since.
        dated = bool(conn.execute(
            f"SELECT EXISTS(SELECT 1 FROM files f WHERE {_OVERVIEW_LIVE} "
            "AND f.taken_at IS NOT NULL)").fetchone()[0])
        stage_ran = {
            "person": faces_stage_ran(conn),
            "event": bool(conn.execute(
                "SELECT EXISTS(SELECT 1 FROM events)").fetchone()[0]) or not dated,
            "animal": bool(conn.execute(
                "SELECT EXISTS(SELECT 1 FROM frame_quality WHERE pet_score IS NOT NULL)"
            ).fetchone()[0]),
        }
    finally:
        conn.close()
    found = {"person": person, "event": event, "animal": animal}
    reasons = {
        name: None if has else (_SLICE_NONE_FOUND if stage_ran[name]
                                else _SLICE_NOT_RUN)
        for name, has in found.items()
    }
    # Offered when it holds photographs OR has something to say and a collection to say
    # it over — the population its own stage walks (canonical photographs for faces and
    # animals, any indexed file for events).
    over = {"person": face, "event": indexed, "animal": face}
    visible = {name: has or (over[name] and reasons[name] == _SLICE_NOT_RUN)
               for name, has in found.items()}
    return {**visible, "face": face, "indexed": indexed, "reasons": reasons}


# --- F134: the search line of the "Slices" tab (`GET /api/search`) ------------------
# One idea: an interface that cannot search says WHY, and never by showing an empty
# result list. "Nothing was found for cake" and "nothing was ever encoded" are the same
# empty list on screen, and only one of them is a fact about the archive. So the state of
# the index travels with every answer:
#
#   empty         no vectors at all           -> process the collection (an ordinary run)
#   other_model   vectors of another model    -> process it again, that index is not
#                                                comparable with this query
#   partial       some of the collection      -> searchable, and it says N of M out loud
#   ready         all of it                   -> an ordinary search line
#
# The two unavailable states are deliberately two: "run it" and "run it AGAIN because the
# model changed" are different instructions. The partial state is a denominator and not a
# warning — an incremental run is the normal way to live with a growing archive.
#
# NOT here: a similarity threshold. The score orders frames against each other and means
# nothing in absolute terms (see search.py), so it travels to the card and the reader
# stops where the quality runs out.

_SEARCH_READY = "ready"
_SEARCH_PARTIAL = "partial"
# The unavailable states are the engine's own codes, not a second spelling: the route can
# be reached before and after `search_text` raises, and the two paths must not be able to
# disagree about which state the index is in.
_SEARCH_AVAILABLE_STATES = (_SEARCH_READY, _SEARCH_PARTIAL)

# The denominator of "N of M" — the same rule `search._CANDIDATES_SQL` selects on,
# counted here rather than imported because this is a COUNT of it.
_SEARCH_PHOTOS_SQL = """SELECT COUNT(*) FROM files
    WHERE dup_of IS NULL AND error IS NULL AND media_type = 'photo'"""

# The numerator. Joined to `files` on purpose: a row whose frame has since become a
# duplicate or gone unreadable is not something a search can return.
#
# F141: `search_embeddings`, the multilingual index the engine actually reads — NOT
# `clip_embeddings`, whose classification vectors cannot answer a query. Counting that
# table would make the line say "searching all 19 753 photographs" over an index the
# search will refuse to use.
_SEARCH_COVERED_SQL = """SELECT COUNT(*) FROM search_embeddings e
    JOIN files f ON f.id = e.file_id
    WHERE e.model = ? AND f.dup_of IS NULL AND f.error IS NULL AND f.media_type = 'photo'"""

# F189: whether anybody in this collection has a NAME — the roots of the `merged_into`
# chains, where `search.match_person` looks. It travels with the state because the line is
# DISABLED while the index cannot rank, and a name needs no index at all
# (`features.search_index` is off by default): without this a person would be typing the
# name of their own daughter into a dead field.
_SEARCH_NAMES_SQL = """SELECT EXISTS(
    SELECT 1 FROM face_clusters WHERE merged_into IS NULL AND label IS NOT NULL)"""

# One card shape, whichever state produced it. LEFT JOIN because a photograph usually has
# no `media_class` row at all — the class is what the privacy rule reads.
_SEARCH_ROWS_SQL = """SELECT f.id, f.path, f.taken_at, mc.verdict
    FROM files f LEFT JOIN media_class mc ON mc.file_id = f.id
    WHERE f.id IN ({marks})"""


def _search_index_state(conn: sqlite3.Connection, model: str) -> dict:
    """Which of the four states the index is in, plus the numbers that state it.

    `index_model` names the model that actually produced the stored vectors, taken as the
    one with the most rows: a table can hold leftovers of several, and only the dominant
    one is worth putting in front of a reader. `indexed`/`photos` is the "searching N of
    M photographs" line, computed here once so it cannot disagree with `available`.
    """
    counts = {str(r["model"]): int(r["n"]) for r in conn.execute(
        "SELECT model, COUNT(*) AS n FROM search_embeddings GROUP BY model")}
    stored = counts.get(model, 0)
    photos = int(conn.execute(_SEARCH_PHOTOS_SQL).fetchone()[0])
    indexed = int(conn.execute(
        _SEARCH_COVERED_SQL, (model,)).fetchone()[0]) if stored else 0
    others = [(n, name) for name, n in counts.items() if name != model]
    if not counts:
        state = REASON_EMPTY
    elif not stored:
        state = REASON_OTHER_MODEL
    elif not indexed:
        # Vectors of this model exist and not one belongs to a frame a search may
        # return: nothing to rank, and running the stage again is the fix.
        state = REASON_EMPTY
    else:
        state = _SEARCH_PARTIAL if indexed < photos else _SEARCH_READY
    return {
        "state": state,
        "available": state in _SEARCH_AVAILABLE_STATES,
        "model": model,
        "index_model": model if stored else (max(others)[1] if others else None),
        "indexed": indexed,
        # F189: deliberately not part of `available` — the index stays in whatever state
        # it is in; this only says the line has something to answer even so.
        "names": bool(conn.execute(_SEARCH_NAMES_SQL).fetchone()[0]),
        # F173: `photos`, not `total`, which everywhere on this server means the length
        # of the list being walked. Two numbers under one name is how a counter starts
        # saying "showing 200 of 19 753 photographs" about a list of 4 000.
        "photos": photos,
    }


def _search_item_to_json(row: sqlite3.Row, score: float, sensitive: frozenset[str],
                         scored: bool = True) -> dict:
    """One card of the ranking: the score is always on it, the thumbnail sometimes.

    F189: `scored=False` for a card of a SELECTION — a person's frames — and then the key
    is ABSENT rather than zero. The number explains an order and this list has none, so a
    "closeness 0.000" under every frame would be a measurement nobody made.

    F133: a frame whose class sits in `vlm.exclude_classes` gets no `thumb_url`, so the
    browser never asks `/thumb` for it. The guard is on the server because a search that
    answered with a link would be the way around a protection the slices already apply.
    """
    path = Path(row["path"])
    payload: dict = {
        "file_id": int(row["id"]),
        "name": path.name,
        "date": row["taken_at"],
    }
    if scored:
        # A ranking, not a filter: the number is what lets a reader see where the
        # relevance ran out.
        payload["score"] = float(score)
    verdict = row["verdict"]
    if verdict is None or str(verdict) not in sensitive:
        payload["thumb_url"] = f"/thumb/{int(row['id'])}"
        payload["video"] = imaging.is_video_path(path)
    return payload


def _search_items(conn: sqlite3.Connection, hits: Sequence[tuple[int, float]],
                  sensitive: frozenset[str], scored: bool = True) -> list[dict]:
    """The engine's (file_id, score) pairs -> cards, IN THE RANKING'S ORDER.

    Fetched in chunks (a limit is user-set and SQLite has a ceiling on bound parameters)
    and then re-ordered by the ranking, never by what SQLite returned: the order is the
    answer here.
    """
    rows: dict[int, sqlite3.Row] = {}
    for part in batched([fid for fid, _score in hits], 500):
        marks = ",".join("?" * len(part))
        rows.update({int(r["id"]): r for r in conn.execute(
            _SEARCH_ROWS_SQL.format(marks=marks), tuple(part))})
    return [_search_item_to_json(rows[fid], score, sensitive, scored)
            for fid, score in hits if fid in rows]


# --- F189: the search line answers a NAME with the person ------------------------------
# A parse of the query string and nothing else — no index, no threshold, no cluster work
# — in ONE place for the whole server: the typed line (`/api/search`) and a pinned slice
# of the same words (`/api/saved-slices`) have to answer identically, or a pin becomes a
# second engine with a name.
#
# Two flags travel to the client rather than a merged list:
#
#     person   the name this string is, whenever it is one — even when the ranking is
#              what gets served, because the offer of the other answer is the point
#     exact    whether THIS payload is the person's frames. It decides the caption, which
#              is how a reader tells an exact selection from the top of a ranking
#
# So a name that is also an ordinary word shows the person first and keeps the ranking
# one click away: the second answer never disappears silently.


def _person_payload(conn: sqlite3.Connection, cfg: Config, label: str, offset: int,
                    limit: int) -> dict:
    """One page of a person's frames, in the shape every paged slice of this server has.

    `exact: true` is the whole difference on the wire. No score on the cards: there is no
    order here to explain.
    """
    page = person_page(conn, label, limit=limit, offset=offset)
    return {
        "person": label,
        "exact": True,
        **_page_payload(
            _search_items(conn, page.hits, frozenset(cfg.vlm.exclude_classes),
                          scored=False),
            total=page.total, offset=page.offset, limit=page.limit),
    }


def _search_payload(cfg: Config, db_path: Path, text: str, offset: int, limit: int,
                    encoder: TextEncoder | None = None, words: bool = False) -> dict:
    """`GET /api/search` — the state of the index always, a page of the ranking when
    there is one.

    An empty query and an unavailable index both return before `rank_text`, which keeps
    a stray keystroke from loading CLIP. `EmbeddingsMissing` is still caught: the state
    was read a moment earlier and a run can empty the table in between, and the answer
    then carries the engine's own reason rather than an empty `items` list.

    F173: `total` is the length of the RANKING and comes back from the engine with the
    page, so the counter and the "show more" button cannot be out of step with it.

    F189: a string that IS somebody's name is answered with that person's frames before
    the index is consulted at all — a selection out of `face_clusters` needs no vector,
    so a name finds the person on a collection nobody has indexed yet. `words=True` is
    how the client asks for the ranking anyway: the name only goes first, it never takes
    the word search away.
    """
    conn = _connect(db_path)
    try:
        model = search_index_model(cfg)  # F141: the search model, not the classifier's
        payload = _search_index_state(conn, model)
        payload.update({"query": text, "person": None, "exact": False,
                        **_page_payload([], total=0, offset=offset, limit=limit)})
        # Computed even when the ranking is what gets served: the client can only offer
        # the other answer if the payload names it.
        person = match_person(conn, text)
        payload["person"] = person
        if person is not None and not words:
            payload.update(_person_payload(conn, cfg, person, offset, limit))
            return payload
        if not text.strip() or not payload["available"]:
            return payload
        try:
            page = rank_text(cfg, conn, text, limit=limit, offset=offset, encoder=encoder)
        except EmbeddingsMissing as exc:
            payload["state"] = exc.reason
            payload["available"] = False
            return payload
        payload.update(_page_payload(
            _search_items(conn, page.hits, frozenset(cfg.vlm.exclude_classes)),
            total=page.total, offset=page.offset, limit=page.limit))
        return payload
    finally:
        conn.close()


def _parse_search_query(query: dict[str, list[str]],
                        default_limit: int) -> tuple[str, int, int, bool] | None:
    """(query text, offset, limit, words) for `GET /api/search`, or None -> 400.

    An absent/empty `q` is NOT an error: the client asks with one on purpose, to learn
    the state of the index without spending a model on it.

    F189: `words=1` asks for the ranking even when the string names somebody. Anything
    else (absent, `0`, a typo) means the person — a malformed flag must not be a 400 on
    a route whose whole job is to answer.
    """
    window = _parse_page_window(query, default_limit)
    if window is None:
        return None
    return ((query.get("q") or [""])[0], window[0], window[1],
            (query.get("words") or [""])[0] == "1")


# --- F151: the pinned queries of the "Slices" tab (`GET /api/saved-slices`) ------------
# A slice is a saved query. The measurement of 2026-08-02 (200 frames out of 22 096,
# labelled by hand, the first time RECALL was measured rather than the precision of the
# top): the six hand-written filters find 6% of the blurred frames, 33% of the animals,
# 0% of the products and nothing at all for children — while the SAME vectors, asked in
# words, give 61% for children, 65% for products and 60% for animals at the same depth,
# and 89% / 95% / 87% at twice it.
#
# So this route adds no model, no pass and no table. The only new thing on the server is
# WHERE the words come from — `features.saved_slices`, a config entry rather than code,
# so a slice can be retuned or added without a release.
#
# Three decisions:
#
# * these lists are ESTIMATES and are labelled apart from the exact ones. The `pet` label
#   beside them is 71% precise and verified by a model; this ranking is 60% and verified
#   by nobody. Both stay — they answer different questions — and matching captions would
#   let a reader take one for the other;
# * no count on a pin, and no threshold anywhere. A ranking covers the whole index, so
#   its length is not a number of children;
# * depth is the lever: "show more" continues the same ranking, the one handle the
#   measurement confirmed (61% -> 89%).
#
# Not here on purpose: PEOPLE (the signal is `faces`, 7 341 frames, exact and free) and
# BLURRED (the sharpness filter is 100% precise on the sample and the query 36%; the
# exact half has to come first or it drowns).


def _saved_slice_by_name(cfg: Config, name: str) -> SavedSlice | None:
    for slice_ in cfg.features.saved_slices:
        if slice_.name == name:
            return slice_
    return None


def _saved_slices_payload(cfg: Config, db_path: Path, name: str | None, offset: int,
                          limit: int, encoder: TextEncoder | None = None) -> dict:
    """`GET /api/saved-slices` — the pins always, one page of the asked-for slice.

    The shape is `_search_payload`'s: a pinned slice IS a search, so the state of the
    index travels with every answer. The rule is worth more here than in the search line
    — nobody types "children" into a pin, so an empty page would read as a fact about the
    archive rather than as a question that missed.

    `name=None` is the tab's own call on open: the pins and the state, no ranking, no
    model. The phrases travel with the page because a slice whose words are invisible
    cannot be edited by the person it is wrong for.

    F189: a pin whose SINGLE phrase is somebody's name answers with that person's frames,
    exactly as the search line does — otherwise a pin ranking a name by CLIP while the
    line selected her cluster would be two silently different answers under one word. A
    pin of several phrases stays a query: a name averaged with other words is not a name.
    """
    # The LIVE config, in the file's own order — that order is the order of the pins.
    slices = cfg.features.saved_slices
    conn = _connect(db_path)
    try:
        model = search_index_model(cfg)
        current = _saved_slice_by_name(cfg, name) if name else None
        payload = _search_index_state(conn, model)
        payload.update({
            "slices": [{"slice": s.name, "queries": list(s.queries)} for s in slices],
            "slice": current.name if current else None,
            "queries": list(current.queries) if current else [],
            # What captions these lists apart from the exact slices beside them. A
            # constant rather than a per-slice flag: everything this route serves is a
            # ranking.
            "approximate": True,
            # F156: travels with every answer so the "pin this" button can say the limit
            # is reached BEFORE somebody names a slice that will be refused.
            "max_pinned": int(cfg.features.max_pinned_slices),
            # F189: the same two flags the search line sends, so the panel captions a
            # pinned person the way it captions a typed one.
            "person": None,
            "exact": False,
            **_page_payload([], total=0, offset=offset, limit=limit),
        })
        if current is None:
            return payload
        person = (match_person(conn, current.queries[0])
                  if len(current.queries) == 1 else None)
        if person is not None:
            payload.update(_person_payload(conn, cfg, person, offset, limit))
            # A fact, not an estimate — the panel prints the word beside every ranking.
            payload["approximate"] = False
            return payload
        if not payload["available"]:
            return payload
        try:
            page = rank_queries(cfg, conn, current.queries, limit=limit, offset=offset,
                                encoder=encoder)
        except EmbeddingsMissing as exc:
            payload["state"] = exc.reason
            payload["available"] = False
            return payload
        payload.update(_page_payload(
            _search_items(conn, page.hits, frozenset(cfg.vlm.exclude_classes)),
            total=page.total, offset=page.offset, limit=page.limit))
        return payload
    finally:
        conn.close()


def _parse_saved_slice_query(cfg: Config, query: dict[str, list[str]],
                             default_limit: int) -> tuple[str | None, int, int] | None:
    """(slice name or None, offset, limit) for `GET /api/saved-slices`, or None -> 400.

    An absent `slice` is NOT an error — it is how the pin row is asked for. A `slice` not
    in the config IS one: an empty page would show a slice that does not exist as one
    holding no photographs.
    """
    window = _parse_page_window(query, default_limit)
    if window is None:
        return None
    name = (query.get("slice") or [""])[0].strip()
    if name and _saved_slice_by_name(cfg, name) is None:
        return None
    return (name or None), window[0], window[1]


# --- F156: pinning a query of one's own (`POST /api/saved-slices/{pin,unpin,move}`) ----
# The measurement that turned this feature around (2026-08-02, a random sample of 200
# frames): 65 of them — a third — fall into no class at all, and the ten candidate slices
# for those 65 cover 26%, 23%, 22%, 20%, 18%, 17%, 15%, 12%, 12%, 6%. Not one reaches a
# third of a third, and food — which both the user and the author had in mind as a large
# slice — came out at 8 frames, smaller than sky or signage. So the product stops
# guessing which facets matter; what is new is WHO writes the list.
#
# Three properties:
#
# * the list lives in `config.yaml`, which survives `reset` and a re-processing where the
#   index does not — a slice somebody named is not something to lose to a re-index;
# * a pin is a SAVED QUERY and nothing else, so a pinned slice is indistinguishable from
#   a built-in one on screen, and unpinning deletes a config entry and touches no file;
# * the number of pins is bounded (`features.max_pinned_slices`) and reaching the bound
#   is SAID. A pin that silently does not appear is worse than no pin.
#
# No suggestions, ever: a "you might want to pin «food»" would be the guessing this
# replaces, wearing a friendlier hat.

# Why a pin was refused, in one word the client can caption — the reason has to be shown
# in the interface language, so the server sends the code and the catalog the sentence.
_PIN_EMPTY = "empty"          # nothing was typed — there is no query to save
_PIN_DUPLICATE = "duplicate"  # a pin of that name is already there
_PIN_LIMIT = "limit"          # `features.max_pinned_slices` is reached


def _validate_pin_payload(payload: object) -> tuple[str, str] | None:
    """Parse the body of `POST /api/saved-slices/pin` -> (name, query). None -> 400.

    `name` is optional and defaults to the query. An empty query is refused HERE rather
    than in the interface: a slice with no words would rank the collection by an
    arbitrary direction and look exactly like an answer.
    """
    if not isinstance(payload, dict):
        return None
    query = payload.get("query")
    if not isinstance(query, str) or not query.strip():
        return None
    name = payload.get("name")
    if name is not None and not isinstance(name, str):
        return None
    return ((name or "").strip() or query.strip()), query.strip()


def _validate_slice_name_payload(payload: object) -> str | None:
    """`{"slice": "<name>"}` -> the name, for unpin and move. None -> 400."""
    if not isinstance(payload, dict):
        return None
    name = payload.get("slice")
    if not isinstance(name, str) or not name.strip():
        return None
    return name.strip()


def _validate_move_payload(payload: object) -> tuple[str, int] | None:
    """`{"slice": …, "delta": -1|1}` -> (name, delta), the arrows. None -> 400.

    Arrows and not drag-and-drop: the order is a list of at most a dozen names, a
    keyboard reaches an arrow, and a drop target would be a second way to say it.
    """
    name = _validate_slice_name_payload(payload)
    if name is None or not isinstance(payload, dict):
        return None
    delta = payload.get("delta")
    if isinstance(delta, bool) or delta not in (-1, 1):
        return None
    return name, int(delta)


def _pinned_with(cfg: Config, name: str, query: str) -> tuple[SavedSlice, ...] | str:
    """The pinned list with one more slice in it, or the code saying why it cannot be.

    The new pin goes to the END, where the person who made it will look for it. Emptiness
    is not one of the answers: `_validate_pin_payload` has already refused it.
    """
    slices = tuple(cfg.features.saved_slices)
    if any(existing.name == name for existing in slices):
        return _PIN_DUPLICATE
    if len(slices) >= int(cfg.features.max_pinned_slices):
        return _PIN_LIMIT
    return (*slices, SavedSlice(name, (query.strip(),)))


def _pinned_without(cfg: Config, name: str) -> tuple[SavedSlice, ...] | None:
    """The pinned list with that slice gone, or None when there is no such slice.

    Nothing but the config entry is removed: the frames the slice ranked are the
    collection's and were never the slice's to hold.
    """
    slices = tuple(cfg.features.saved_slices)
    kept = tuple(s for s in slices if s.name != name)
    return kept if len(kept) != len(slices) else None


def _pinned_moved(cfg: Config, name: str, delta: int) -> tuple[SavedSlice, ...] | None:
    """The pinned list with that slice one step up or down. None -> no such slice.

    A step off either end is a no-op rather than an error: the arrow at the top of the list
    does nothing, which is what an arrow at the top of a list does.
    """
    slices = list(cfg.features.saved_slices)
    index = next((i for i, s in enumerate(slices) if s.name == name), None)
    if index is None:
        return None
    target = index + delta
    if 0 <= target < len(slices):
        slices[index], slices[target] = slices[target], slices[index]
    return tuple(slices)


def _apply_saved_slices(cfg: Config, slices: tuple[SavedSlice, ...]) -> None:
    """Put the new pin list into the RUNNING config, `raw` included.

    `raw` is mirrored for the reason `_apply_settings` mirrors its own section: a later
    save of anything else must not write back the mapping this call just replaced.
    """
    cfg.features = dataclasses.replace(cfg.features, saved_slices=slices)
    section = cfg.raw.get("features")
    if not isinstance(section, dict):
        section = {}
        cfg.raw["features"] = section
    section["saved_slices"] = {s.name: list(s.queries) for s in slices}


class _LazyTextEncoder:
    """The CLIP text tower of this server: loaded on the first query, then reused.

    Two reasons, as with `_LazyClassifierHolder`: the model must not be loaded by merely
    starting the UI (most sessions never search), and not twice, since the search route
    and the album route both encode text.
    """

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._encoder: TextEncoder | None = None
        self._lock = threading.Lock()

    def __call__(self, texts: Sequence[str]) -> Any:
        with self._lock:
            if self._encoder is None:
                # F141: the SEARCH model's text tower — a query has to land in the space
                # the stored vectors live in, and since F141 that space is not the
                # classification model's.
                self._encoder = text_encoder(search_index_settings(
                    naming_settings(self._cfg), search_index_model(self._cfg)))
            encoder = self._encoder
        return encoder(texts)
