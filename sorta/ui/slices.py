"""F182: the "Slices" tab — the queries, the pins and the built-in slices.

A slice is a question asked of the index: the utility frames, the animals, the face
slices, the search line, and the pinned queries a person saved. None of them moves a
file — a slice is hardlinks, free to make and to drop — and that is what keeps them
in one module while the canon lives in `layout`.
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
# The deep VLM tier carries away roughly every tenth frame of the collection into
# service folders (2 202 `product` alone on the live 24k run), and until now those
# buckets were visible only indirectly, as folders of the layout plan. A handful of
# those verdicts are wrong, and "a handful out of 2 202" is dozens of frames nobody
# could find. This view shows the buckets AS buckets and lets the wrong ones go back in
# one action. It reclassifies nothing: the fix is a row in `manual_overrides` (F77),
# `media_class` keeps whatever the model measured.

# The `document` bucket is passports, medical forms and bank papers. Those frames get a
# card with a name and a date and NO thumbnail — the project rule is that a document
# verdict is never decoded for display (a preview is a derived copy of the contents).
# Returning one to the photos is still allowed: the person knows what is in their own
# file, they just do not need it rendered to decide.
# F133: which classes those are is a CONFIG question, not a constant — `vlm.exclude_classes`
# already carries the list ("do not show this to the model") and defaults to
# `["document"]`. One visible list of sensitive classes beats two, of which the second
# gets forgotten. The tuple below is only the fallback for a caller that passes nothing:
# a privacy guard must never switch itself off through an omission (the F120 lesson,
# where a typo in the same key would silently have sent documents to the VLM).
_JUNK_NO_PREVIEW = ("document",)


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
        # F175: said out loud, per card, for the same reason the page-level list is —
        # a card the person must not delete has to be visible AS one before the "select
        # everything" button is pressed, and a client inferring it from the missing
        # `thumb_url` would be a second copy of the privacy rule in JS.
        "sensitive": verdict in no_preview,
        # F77/F103: the frame already carries a manual "this is a photo" correction —
        # the card says so instead of offering the same action twice.
        "restored": restored,
        # F174: where the frame lands if it IS returned — the folder the plan will build
        # for it once the `photo` mark is in the table, not a folder named by this file.
        **_destination_json(dest),
    }
    if verdict not in no_preview:
        payload["thumb_url"] = f"/thumb/{int(row['id'])}"
        payload["video"] = imaging.is_video_path(path)
    return payload


# F171: the order INSIDE one bucket — the model's own estimate, most confident first.
# `media_class.score` is the number the verdict was decided by (the CLIP probability of
# the winning class, or the text density for a document); NULL means "no estimate", never
# "unsure", so those frames keep the old path order at the END of the list instead of
# sinking to a score they were never given. The id is not needed as a tie-break: `f.path`
# is unique and already breaks every tie, so a card keeps its place between pages.
#
# It is applied to one bucket and never to the "all" view, for the reason F175 gives about
# the captions: four classes are four separate softmaxes, and an order across them would
# be a comparison nobody measured.
_JUNK_ORDER = "(mc.score IS NULL), mc.score DESC, f.path"


def _junk_payload(db_path: Path, cfg: Config, bucket: str | None,
                  offset: int, limit: int,
                  sensitive: frozenset[str] = frozenset(_JUNK_NO_PREVIEW)) -> dict:
    """`GET /api/junk` — the buckets with their counts + one page of one bucket.

    F133: `sensitive` is `vlm.exclude_classes` — the config list that already means
    "handle this class as private", and whose default is `["document"]`. A class in it
    keeps its counter, its cards and the way back to the photos, and loses exactly one
    thing: `thumb_url`. That is the whole of the rule, and it has to be enforced HERE
    rather than in the markup — a card the browser was given a preview link for is a
    card whose contents have already been decoded and sent, whatever the page then
    chooses to draw. The card still carries a name and a date, which is what "open the
    documents in the common grid, do not enlarge them" (the brief) asks for.

    Reusing the VLM key instead of adding a second one is a deliberate trade: one
    visible list of sensitive classes beats two, of which the second gets forgotten.
    Emptying it therefore lifts both protections at once — the guide entry for the key
    is what has to say so.

    The selection is `media_class.verdict <> 'photo'` over canonical, readable files —
    the same `dup_of IS NULL AND error IS NULL` population `junk.classify` writes and
    the sorter lays out, so a bucket counter here matches what the plan will carry off.

    `bucket=None` — every non-photo frame; otherwise exactly the requested verdict. The
    `<> 'photo'` guard sits in the query itself rather than in the parameter check, so
    no value of `bucket` can turn this route into a way of listing personal photos.

    `buckets` is always the full set of counters (it is what the filter chips are drawn
    from), independent of the current filter; `total` is the size of the CURRENT
    selection. An unknown bucket is an empty page, not an error — the same rule as an
    unknown category in `PlanCache.page`.

    F139: `album_kind` is the album this bucket can be gathered into, or None — the
    server decides, because the answer depends on `sensitive` and a client that worked it
    out for itself would be a second copy of the privacy rule. It is None for the "all"
    view (an album of "everything the classifier carried off" is not a slice anybody
    asked for) and for a class in `vlm.exclude_classes`, which keeps its counter and gets
    neither a preview nor an album.

    F171: a bucket is a LIST IN ORDER — `_JUNK_ORDER`, the model's own estimate first —
    and `ordered_by_score` says whether it really was one, so the caption promises a
    ranking exactly where there is one to promise.
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
        # F174: what the button on these cards will do — asked with the correction it
        # writes already assumed, so the caption names the city the frame goes back to
        # and not the service folder it is sitting in right now.
        dests = _destinations_for(cfg, conn, rows, "photo")
    finally:
        conn.close()
    marks = _overrides_map(db_path) if rows else {}
    buckets = [{"verdict": r["verdict"], "count": int(r["n"])} for r in counts]
    buckets.sort(key=lambda b: (-b["count"], b["verdict"]))
    return {
        "bucket": bucket,
        "buckets": buckets,
        "album_kind": (
            bucket if (bucket in CLASS_ALBUM_KINDS and bucket not in sensitive)
            else None),
        # Said out loud rather than inferred from a missing field: a card without
        # `thumb_url` is a class the server refuses to render, not a preview that failed
        # to build, and the two need different words on the screen.
        "sensitive": sorted(sensitive),
        # F171: whether this page really is the ranking its caption promises. `False` for
        # the "all" view (no ordering across four buckets) and for a bucket the classifier
        # settled without a number of its own — a heuristics-only run, or the frames the
        # deep tier rewrote, both of which store NULL rather than a confidence.
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
# The signal has been computed since F113 and calibrated in F122 (805 frames of the live
# collection at 92% precision), and until now nobody could see a single one of them. The
# view is the junk grid's twin — a page of thumbnails over a read-only query — with one
# deliberate difference: the order is by CONFIDENCE, not by path. About 64 of those 805
# frames are not animals, and reading top-down until the quality runs out is how a person
# finds where that border sits, so the score travels to the card and is shown on it.


def _animal_item_to_json(row: sqlite3.Row, dest: Destination | None = None) -> dict:
    """One card of the animal view: a thumbnail, a name, a date and the pet score.

    F124: plus the two facts a card has to state about the mark itself — whether the
    frame counts as an animal right now (`is_animal`, straight out of the shared rule,
    never recomputed here in Python) and whether that answer came from a person
    (`manual`, the value of the `manual_pet` row, or None if there is none). A frame the
    user has taken the mark off stays on the card, struck through: it must be visible as
    marked BY HAND, otherwise the counter moves for no reason anybody can see and the
    decision cannot be taken back.
    """
    path = Path(row["path"])
    return {
        "file_id": int(row["id"]),
        "name": path.name,
        "date": row["taken_at"],
        # NULL is impossible for a frame that carries a verdict (junk writes the score
        # alongside it) — but a payload that pretends 0.0 was measured would lie about
        # exactly the number this tab exists to show.
        "score": None if row["pet_score"] is None else float(row["pet_score"]),
        "is_animal": bool(row["is_animal"]),
        "manual": None if row["manual"] is None else bool(row["manual"]),
        "thumb_url": f"/thumb/{int(row['id'])}",
        "video": imaging.is_video_path(path),
        # F174: where the frame ALREADY lies. This slice is a view over the canon, not an
        # extraction from it, so the mark moves no file — and the card can only say so
        # convincingly by naming the folder the frame is in either way.
        **_destination_json(dest),
    }


_ANIMALS_JOIN = ("FROM files f LEFT JOIN frame_quality fq ON fq.file_id = f.id "
                 "LEFT JOIN manual_pet mp ON mp.file_id = f.id")


# F160: the animal rule now has a tier whose switches live outside `features:` — the
# detector's master switch `detect.enabled` (F145) and the model that wrote the boxes. So
# the helpers of this slice take the WHOLE live config, the way `_overview_payload` already
# does, and resolve both switches through the one function that ANDs them
# (`detector_settings`).
# Reading half of the pair here is exactly the mistake F145 was written about, and a slice
# still counting the boxes of a detector the user has switched off is the same bug in the
# other direction.
def _animals_population(cfg: Config) -> str:
    """What the TAB LISTS: the model's marks plus every frame a person has touched.

    Deliberately wider than the slice — a frame marked "not an animal" is no longer in the
    album and is still on this page, struck through, because a card that vanishes takes the
    undo button with it.

    F137: "the model's marks" is the automatic half of the shared rule (`animal_auto_sql`),
    not the `frame_quality.pet` cache — a threshold edit has to take frames OFF this page
    too, or the list and the counter it carries would disagree about the same collection.
    """
    return (f"({animal_auto_sql(cfg.features, 'fq', detector_settings(cfg))} "
            "OR mp.file_id IS NOT NULL) AND f.dup_of IS NULL AND f.error IS NULL")


def _animals_count_sql(cfg: Config) -> str:
    """What COUNTS as an animal: `sorter.animal_ids_sql` and nothing else, over the
    canonical, readable files every other counter in this file is built on. Used by this
    tab and by the "Overview" number, so the two cannot disagree with the album or with
    each other."""
    ids = animal_ids_sql(cfg.features, detector_settings(cfg))
    return f"""SELECT COUNT(*) FROM files f
    WHERE f.dup_of IS NULL AND f.error IS NULL AND f.id IN ({ids})"""


def _animals_select(cfg: Config) -> str:
    """One card, one row shape — the page and the answer to a mark are the same SELECT, so
    a card redrawn after an edit says exactly what the same card would say on a reload."""
    ids = animal_ids_sql(cfg.features, detector_settings(cfg))
    return f"""SELECT f.id, f.path, f.taken_at, fq.pet_score,
           mp.is_animal AS manual, f.id IN ({ids}) AS is_animal
    {_ANIMALS_JOIN}"""


def _animals_payload(db_path: Path, cfg: Config, offset: int, limit: int) -> dict:
    """`GET /api/animals` — one page of the animal slice, most confident first.

    Two numbers, because after F124 they are two different questions: `total` is the
    length of the LIST (what the paging walks — model marks plus manual decisions), and
    `animals` is how many frames actually count as animals, by the one shared rule. The
    second is the number "Overview" shows and the album gathers; the first is what
    "showing 200 of N" is about.

    `ORDER BY pet_score DESC, f.id` — the id breaks ties, so two frames with an equal
    score keep a stable place between pages instead of swapping and being shown twice
    (or never) as the reader pages down. A manual decision does NOT move a card: the
    reader is walking down a list sorted by confidence, and a list that reshuffles under
    the frame just marked is a list nobody can finish reading.

    `cfg` is the LIVE config, for the reason `/api/junk` reads its sensitive classes off
    it: the thresholds this page is drawn with — and, since F160, whether the detector's
    tier counts at all — are the ones in force at the moment of the request, not the ones
    some run wrote into the database (F137).
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
        # F174: no assumed correction — the question here is where the frame lies NOW,
        # which is the same folder it will lie in after the mark, because the mark
        # changes a membership and not a route.
        dests = _destinations_for(cfg, conn, rows)
    finally:
        conn.close()
    return {
        "animals": int(animals),
        **_page_payload([_animal_item_to_json(r, dests.get(int(r["id"])))
                         for r in rows],
                        total=int(total), offset=offset, limit=limit),
    }


# F124: "the model is wrong about this frame", the only three answers there are. `clear`
# drops the row and hands the frame back to the automatic verdict — which is not the same
# as `not_animal`, and the difference is the reason the row is two-valued rather than a
# presence flag.
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

    One row per file (PRIMARY KEY file_id), so marking the same frame twice overwrites
    rather than piling up. Nothing here touches `frame_quality` — the whole point of the
    feature is that the model's own table keeps being recomputed from scratch and this
    mark is read on top of it (`sorter.animal_ids_sql`).

    An id outside the current index is skipped rather than written (the rule
    `_apply_review_mark`/`_trash_files` follow): a decision about a file the program does
    not know is not a decision about anything, and the FK would reject it anyway.

    The answer carries `items` (the marked frames as the tab's own cards) and `animals`
    (the fresh count by the shared rule) so the client can redraw one card and the
    counter in place. It could reload the page instead, and that is exactly what it must
    not do: this list is read top-down until the confidence runs out, and a reload sends
    the reader back to the first screen after every decision. `items` may come back
    SHORTER than the ids — a `clear` on a frame the model never marked leaves the list
    altogether — and the client drops those cards.
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
        # F174: the redrawn card has to say what a reload would say, and after F174 that
        # includes the folder the frame lies in — a caption that vanished on the first
        # click would look like the mark had moved the file after all.
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
# sample of 200 frames) had no slice at all, while the signal for them has been on disk
# since the faces stage: 12 952 real faces over 7 341 photographs. The rules themselves
# live in `sorter.face_slice_ids_sql`, exactly one copy of them, for the reason
# `ANIMAL_IDS_SQL` lives there — the album, this panel and the "Overview" counters must
# be talking about one collection.
#
# What is different from the slices around it is the CAPTION rather than the query:
# membership here is a fact of a detector's output, not a place in a ranking, so the
# panel says so and says nothing about confidence — there is no score to show.
#
# The one state that is not a number: without a faces run the honest answer is WHY there
# is nothing (`reason='no_faces_run'`, the F125 rule) and the counters travel as `null`
# rather than as zeros. A zero here reads as "no photograph of yours has a person on
# it" — a conclusion about somebody's own archive, drawn from a table nobody filled.

# Canonical and readable, the population every other counter in this file is built on.
# `media_type` is not filtered: the faces stage only ever writes rows for photographs,
# so a video cannot be in these slices anyway.
_FACE_LIVE = "f.dup_of IS NULL AND f.error IS NULL"

# `media_class` rides along for the F133 privacy rule alone — a frame of a sensitive
# class is listed but never given a `thumb_url`, so no preview of a document with a face
# on it is ever decoded.
_FACE_FROM = "FROM files f LEFT JOIN media_class mc ON mc.file_id = f.id"

# How many real faces this frame carries — the one number a card of these slices shows,
# and the same `bbox != '[]'` rule the slices themselves are built on.
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
    was found on it. The face count is on the card all the same — it is what makes the
    group slice checkable by eye, and on a portrait it says "one" out loud.
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

    `counts` is always the full set (it is what the pins draw), and every entry is `null`
    when the faces stage has not run: the counters are then not zero, they are unmeasured,
    and `reason` says which. Once the stage has run a zero IS the answer — "no group
    photographs were found" is a fact about the collection — and it is shown as one.

    `ORDER BY f.id`: these slices have no ranking of their own (there is no confidence in
    them to rank by), and index order is stable, which is what paging needs.
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
        # The thresholds travel with the answer so the hint above the grid can state the
        # rule the numbers were produced by instead of repeating a default in JS.
        "group_min": int(cfg.features.group_photo_faces),
        "portrait_share": float(cfg.features.portrait_face_share),
        **_page_payload(items, total=total, offset=offset, limit=limit),
    }


def _parse_face_slice_query(query: dict[str, list[str]]) -> tuple[str, int, int] | None:
    """(slice, offset, limit) for `GET /api/face-slices`, or None -> 400.

    An unknown slice is refused rather than answered with an empty page, the
    `_parse_review_query` rule: there are exactly three, so anything else is a client
    that has lost track of what it is asking for.
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

    An empty/absent `bucket` means "every non-photo frame"; the window is parsed by the
    same rules (and with the same bounds) as a plan page — a bad number is refused, an
    over-eager limit is clamped rather than rejected.
    """
    window = _parse_page_window(query)
    if window is None:
        return None
    raw_bucket = (query.get("bucket") or [""])[0].strip()
    return (raw_bucket or None), window[0], window[1]


# --- F156: why a built-in slice is empty --------------------------------------------
# A zero with no explanation reads as "your archive holds none of these", and far more
# often it means "nobody has looked yet" — the `frame_quality` rule of F125 (NULL is "not
# asked", not "no") applied to a whole slice. So each of the three exact slices answers
# with one of three things, and never with a bare emptiness:
#
#   None          the slice holds photographs
#   not_run       the stage that fills it never ran — the run screen is where that is
#                 fixed, and the interface links straight to it
#   none_found    the stage ran over this collection and there is nothing of the kind
#
# The two reasons are two on purpose: only one of them is a fact about the person's
# photographs, and only the other one has an action attached to it.
#
# `not_run` is also what a stage SWITCHED OFF looks like (`features.pets: false` — the
# quality stage runs and never asks about animals), and that is right rather than a
# compromise: the run screen holds that checkbox, so the sentence and the link lead to the
# same place either way. Which is the whole reason the standard slices are not made
# hideable a second time — one control for "do not compute animals", not two.
_SLICE_NOT_RUN = "not_run"
_SLICE_NONE_FOUND = "none_found"


def _tabs_visibility_payload(db_path: Path, cfg: Config) -> dict[str, object]:
    """F54: visibility of the "People"/"Events"/"Animals" tabs — by data presence
    (variant B, without a meta table). person ⇔ there is a faces row with a non-empty
    cluster_id (the same source as `_clusters_payload`); event ⇔ non-empty `events`;
    animal (F123) ⇔ some `frame_quality` row counts as an animal, which is false for
    every collection processed with `features.pets` off. Light EXISTS queries, we do not
    build the full payload.

    F156: ...or there is something to SAY. A slice whose stage has never run appears too,
    exactly as the face slices have since F152, because its emptiness is a sentence with a
    link in it and a pin that hides itself never gets to say it. `reasons` carries which
    of the two empty states each slice is in (`_SLICE_NOT_RUN` / `_SLICE_NONE_FOUND`, and
    `None` when the slice holds something) — the answer the panel captions itself with.
    A slice that ran and found nothing keeps hiding: there the zero IS the fact, the
    collection has already said it, and a pin over an empty page teaches nothing.

    The animal question is deliberately asked of what the tab would LIST
    (`_animals_population`) and not of what it would count: a user who has taken the mark
    off every frame has emptied the slice but not the tab, and the tab is where the undo
    button lives. F137 is the reason it is that expression rather than the older "some
    `frame_quality.pet` is set" — the cache column can claim a verdict the thresholds in
    force have withdrawn, and a tab shown for an empty page is exactly the drift this
    feature is about.

    F152: `face` is the one that is NOT asked of its own data, and that is the whole
    point. The three face slices appear as soon as the index holds a photograph the faces
    stage could have looked at — because without a run they have to be able to SAY there
    was no run (`no_faces_run`), and a pin that hides itself says nothing at all. It is
    the same question phase 3 asks of a collection (`faces._CANONICAL`), minus the join
    to the faces table.

    `indexed` rides along for the same cost: "re-run the selected stage" only makes
    sense over files that exist. Right after "Start over" the index is empty and
    ticking "faces" used to light the button up — offering to catch up a stage on
    nothing at all.
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
        # F156: which of the two empty states each of the three is in. One EXISTS per
        # slice, and each one asks whether the STAGE left anything behind — not whether
        # the slice came out non-empty, which is the question already answered above.
        #
        # faces: a real box (`faces_stage_ran` excludes the "processed, none here" marker
        #   row). events: the stage groups every canonical frame that carries a date, so
        #   its own output is the only marker there is — with dated frames and no events
        #   nothing has grouped them, and with no dated frames there was nothing to group.
        #   animals: a STORED `pet_score`, which the stage writes whether or not it reached
        #   the threshold and never writes with `features.pets` off. A fact of the table
        #   rather than the switch as it stands right now (F137's rule): the question is
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
    # A slice is offered when it holds photographs OR when it has something to say and a
    # collection to say it over — the population being the one its own stage walks
    # (canonical photographs for faces and animals, any indexed file for events).
    over = {"person": face, "event": indexed, "animal": face}
    visible = {name: has or (over[name] and reasons[name] == _SLICE_NOT_RUN)
               for name, has in found.items()}
    return {**visible, "face": face, "indexed": indexed, "reasons": reasons}


# --- F134: the search line of the "Slices" tab (`GET /api/search`) ------------------
# F129 built the engine and F133 left the line drawn but disabled; this is the wiring in
# between. It carries one idea and everything else follows from it: an interface that
# cannot search says WHY, and never by showing an empty result list.
#
# `clip_embeddings` is filled by the junk stage of an ordinary run, so a fresh collection
# — and any collection last processed before F128 — has nothing to rank. "Nothing was
# found for cake" and "nothing was ever encoded" are the same empty list on screen, and
# only one of them is a fact about the archive. A person who reads the first when the
# second is true concludes something false about their own photographs, which is the
# single most expensive mistake this feature can make. So the state of the index travels
# with every answer, the line is disabled while there is nothing to search, and the
# reason stands next to it:
#
#   empty         no vectors at all           -> process the collection (an ordinary run)
#   other_model   vectors of another model    -> process it again, that index is not
#                                                comparable with this query
#   partial       some of the collection      -> searchable, and it says N of M out loud
#   ready         all of it                   -> an ordinary search line
#
# The two unavailable states are deliberately two: "run it" and "run it AGAIN because the
# model changed" are different instructions, and a single sentence covering both teaches
# the reader nothing. The partial state is not a warning but an honest denominator — an
# incremental run is the normal way to live with a growing archive, and a person has to be
# able to tell "it is not in the collection" from "it is not in the index yet".
#
# What this route does NOT do: introduce a similarity threshold. The score orders frames
# against each other and means nothing in absolute terms (see search.py), so it travels to
# the card and the reader stops where the quality runs out — the same arrangement the
# animal slice and the sharpness list already use.

_SEARCH_READY = "ready"
_SEARCH_PARTIAL = "partial"
# The unavailable states are the engine's own codes, not a second spelling of them: the
# route can be reached before and after `search_text` raises, and the two paths must not
# be able to disagree about which state the index is in.
_SEARCH_AVAILABLE_STATES = (_SEARCH_READY, _SEARCH_PARTIAL)

# The population a search ranks over and the denominator of "N of M" — the same
# `dup_of IS NULL AND error IS NULL AND media_type = 'photo'` rule `search._CANDIDATES_SQL`
# selects on, counted here rather than imported as SQL because this is a COUNT of it.
_SEARCH_PHOTOS_SQL = """SELECT COUNT(*) FROM files
    WHERE dup_of IS NULL AND error IS NULL AND media_type = 'photo'"""

# How much of that population this model has a vector for. Joined to `files` on purpose:
# a row whose frame has since become a duplicate or gone unreadable is not something a
# search can return, so counting it would inflate the numerator of a fraction whose whole
# job is to be honest.
#
# F141: the table is `search_embeddings`, the multilingual index the engine actually reads
# — not `clip_embeddings`, which holds the classification model's vectors and cannot
# answer a query. Counting the other table would make this line say "searching all 19 753
# photographs" over an index the search will refuse to use, which is the one thing this
# route exists to prevent.
_SEARCH_COVERED_SQL = """SELECT COUNT(*) FROM search_embeddings e
    JOIN files f ON f.id = e.file_id
    WHERE e.model = ? AND f.dup_of IS NULL AND f.error IS NULL AND f.media_type = 'photo'"""

# F189: whether anybody in this collection has a NAME — the roots of the `merged_into`
# chains, which is where `search.match_person` looks. It travels with the state because the
# line is DISABLED while the index cannot rank, and a name needs no index at all:
# `features.search_index` is off by default, so without this the feature would be invisible
# on a fresh collection — a person typing the name of their own daughter into a dead field.
_SEARCH_NAMES_SQL = """SELECT EXISTS(
    SELECT 1 FROM face_clusters WHERE merged_into IS NULL AND label IS NOT NULL)"""

# One card, and the same shape whichever state produced it. LEFT JOIN because a photograph
# usually has no `media_class` row at all — the class is what the privacy rule below reads.
_SEARCH_ROWS_SQL = """SELECT f.id, f.path, f.taken_at, mc.verdict
    FROM files f LEFT JOIN media_class mc ON mc.file_id = f.id
    WHERE f.id IN ({marks})"""

# F173: a limit is a SAMPLE SIZE (search.py) and a page of one at that, so the ceiling on
# how much may be rendered at once is `_PLAN_PAGE_MAX_LIMIT` like everywhere else — a
# client asking for more gets less rather than an error. It used to be a constant of its
# own with the same value, which is one more place a rule could drift.


def _search_index_state(conn: sqlite3.Connection, model: str) -> dict:
    """Which of the four states the index is in, plus the numbers that state it.

    `index_model` is what a person is told when the answer is "another model": the name of
    the model that actually produced the stored vectors, taken as the one with the most
    rows. Naming it is the difference between a sentence somebody can act on and a shrug —
    and the row count is how the name is chosen, because a table can hold leftovers of
    several models and only the dominant one is worth putting in front of a reader.

    `indexed` counts vectors of THIS model within the searchable population, `photos` the
    population itself. The pair is the "we are searching N of M photographs" line, and it
    is computed here, once, so the line and the availability of the field cannot disagree.
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
        # Vectors of this model exist and not one of them belongs to a frame a search may
        # return. There is nothing to rank and running the stage again is the fix, so this
        # is the empty state — exactly what `search._nothing_to_rank` calls it.
        state = REASON_EMPTY
    else:
        state = _SEARCH_PARTIAL if indexed < photos else _SEARCH_READY
    return {
        "state": state,
        "available": state in _SEARCH_AVAILABLE_STATES,
        "model": model,
        "index_model": model if stored else (max(others)[1] if others else None),
        "indexed": indexed,
        # F189: not part of `available` — the index is still in whatever state it is in,
        # and the sentence about it does not change. What this adds is that the line has
        # something to answer even so.
        "names": bool(conn.execute(_SEARCH_NAMES_SQL).fetchone()[0]),
        # F173: `photos`, not `total`. This route answers with a PAGE of a ranking now, and
        # in every paged payload of this server `total` means the length of the list being
        # walked. Two numbers called the same thing in one answer is how a counter starts
        # saying "showing 200 of 19 753 photographs in the collection" about a list of
        # 4 000 — so the coverage line's denominator got the name of what it counts.
        "photos": photos,
    }


def _search_item_to_json(row: sqlite3.Row, score: float, sensitive: frozenset[str],
                         scored: bool = True) -> dict:
    """One card of the ranking: the score is always on it, the thumbnail sometimes.

    F189: `scored=False` for a card of a SELECTION — a person's frames — and then the key
    is absent rather than zero. The number explains an order; this list has no order to
    explain, and «близость 0.000» under every frame of somebody's daughter would be a
    measurement nobody made.

    F133's rule, unchanged: a frame whose class sits in `vlm.exclude_classes` (documents
    by default) gets no `thumb_url`, so the browser never asks `/thumb` for it and no
    preview of a passport is ever decoded. The guard is here, on the server, for the
    reason it is there — a search that answered with a link would turn this route into
    the way around a protection the slices already apply.
    """
    path = Path(row["path"])
    payload: dict = {
        "file_id": int(row["id"]),
        "name": path.name,
        "date": row["taken_at"],
    }
    if scored:
        # A ranking, not a filter: the number is what lets a reader see where the
        # relevance ran out, and a card without it would hide exactly that.
        payload["score"] = float(score)
    verdict = row["verdict"]
    if verdict is None or str(verdict) not in sensitive:
        payload["thumb_url"] = f"/thumb/{int(row['id'])}"
        payload["video"] = imaging.is_video_path(path)
    return payload


def _search_items(conn: sqlite3.Connection, hits: Sequence[tuple[int, float]],
                  sensitive: frozenset[str], scored: bool = True) -> list[dict]:
    """The engine's (file_id, score) pairs -> cards, IN THE RANKING'S ORDER.

    The rows are fetched in chunks (a limit is user-set and SQLite has a ceiling on bound
    parameters — the `search.file_paths` reason) and then re-ordered by the ranking, never
    by whatever order SQLite returned: the order is the answer here.
    """
    rows: dict[int, sqlite3.Row] = {}
    for part in batched([fid for fid, _score in hits], 500):
        marks = ",".join("?" * len(part))
        rows.update({int(r["id"]): r for r in conn.execute(
            _SEARCH_ROWS_SQL.format(marks=marks), tuple(part))})
    return [_search_item_to_json(rows[fid], score, sensitive, scored)
            for fid, score in hits if fid in rows]


# --- F189: the search line answers a NAME with the person ------------------------------
# The question this closes: a cluster somebody named, and merged another cluster into, was
# reachable by `album person <name>` and by `sort --by person` and by no query anybody could
# type. «Ирина» in the search line asked CLIP for frames resembling a WORD.
#
# The bridge is a parse of the query string and nothing else — no index, no threshold, no
# cluster work — and it is deliberately in ONE place for the whole server: the typed line
# (`/api/search`) and a pinned slice of the same words (`/api/saved-slices`, F156) have to
# answer identically, or a pin becomes a second engine with a name.
#
# What travels to the client is two flags rather than a merged list:
#
#     person   the name this string is, whenever it is one — even when the answer being
#              served is the ranking, because the offer of the other answer is the point
#     exact    whether THIS payload is the person's frames. It decides the caption, and a
#              caption is how a reader tells an exact selection from the top of a ranking
#
# Requirement 4 lives in that pair: a name that is also an ordinary word («Роза», «Марк»)
# shows the person first and keeps the ranking one click away — the second answer never
# disappears silently, which is what a search line quietly hijacked by a name would do.


def _person_payload(conn: sqlite3.Connection, cfg: Config, label: str, offset: int,
                    limit: int) -> dict:
    """One page of a person's frames, in the shape every paged slice of this server has.

    `exact: true` is the whole difference on the wire, and the client draws a different
    sentence from it. The cards carry no score (`scored=False`): there is no order here to
    explain.
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
    """`GET /api/search` — the state of the index always, a page of the ranking when there
    is one.

    The model is not asked anything unless there is a reason to: an empty query and an
    unavailable index both return before `rank_text`, which is what keeps a stray
    keystroke from loading CLIP and what makes "the line is disabled" cheap to render.

    `EmbeddingsMissing` is still caught, because the state was read a moment earlier and a
    run can empty the table in between; the answer then carries the engine's own reason
    rather than an empty `items` list, which is the one thing this route must never send.

    F173: a page rather than the whole answer, in the shape `_page_payload` gives every
    other paged slice. `total` is the length of the RANKING — the number the counter says
    and the number the "show more" button is decided by — and it comes back from the engine
    with the page, so the two cannot be computed out of step with each other. A state that
    ranks nothing still carries `total: 0` and `has_more: false`: the client draws the same
    controls whatever happened, and they are simply not there when there is nothing below.

    F189: a string that IS somebody's name is answered with that person's frames, before
    the index is consulted at all — a selection out of `face_clusters` needs no vector, so
    a name still finds the person on a collection nobody has indexed yet. `words=True` is
    how the client asks for the ranking anyway, which is the other half of the same rule:
    the name never takes the word search away, it only goes first.
    """
    conn = _connect(db_path)
    try:
        model = search_index_model(cfg)  # F141: the search model, not the classifier's
        payload = _search_index_state(conn, model)
        payload.update({"query": text, "person": None, "exact": False,
                        **_page_payload([], total=0, offset=offset, limit=limit)})
        # Computed even when the ranking is what gets served: the client offers the other
        # answer, and it can only offer what the payload names.
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

    An absent/empty `q` is NOT an error: the client asks with one on purpose, to learn the
    state of the index without spending a model on it. The window is the shared
    `_parse_page_window` — a non-integer or a negative number is rejected, an over-eager
    limit is clamped — with `features.search_page` as the default size of a page.

    F189: `words=1` asks for the ranking even when the string names somebody. Anything else
    (absent, `0`, a typo) means the default, which is the person — a malformed flag must not
    be a 400 on a route whose whole job is to answer.
    """
    window = _parse_page_window(query, default_limit)
    if window is None:
        return None
    return ((query.get("q") or [""])[0], window[0], window[1],
            (query.get("words") or [""])[0] == "1")


# --- F151: the pinned queries of the "Slices" tab (`GET /api/saved-slices`) ------------
# A slice is a saved query. The measurement of 2026-08-02 (200 frames out of 22 096,
# labelled by hand, the first time RECALL was measured rather than the precision of the
# top) is what turned the feature around: the six hand-written filters find 6% of the
# blurred frames, 33% of the animals, 0% of the products and have nothing at all for
# children — while the SAME vectors, asked in words, give 61% for children, 65% for
# products and 60% for animals at the same depth, and 89% / 95% / 87% at twice it.
#
# So this route adds no model, no pass and no table: the vectors are the junk stage's
# (F128/F141), the ranking is F129's, the paging is F173's, and the only new thing on the
# server is WHERE the words come from — `features.saved_slices`, a config entry rather
# than code, so a slice can be retuned or added without a release.
#
# Three properties are the feature and each is a decision:
#
# * these lists are ESTIMATES and are labelled apart from the exact ones. The `pet` label
#   next to them is 71% precise and verified by a model; this ranking is 60% and verified
#   by nobody. Both slices stay, because they answer different questions ("is this
#   confidently an animal" against "show me every animal"), and if their captions matched
#   a reader would take one for the other;
# * no count on a pin, and no threshold anywhere. A ranking covers the whole index, so its
#   length is not a number of children; where the list stops being about the query is a
#   judgement, and the person reading it makes it;
# * depth is the lever. The page is `features.search_page` and "show more" continues the
#   same ranking — the one handle the measurement confirmed (61% -> 89%).
#
# Not here on purpose: PEOPLE (the signal is `faces`, 7 341 frames, exact and free — F152
# already draws it) and BLURRED (the sharpness filter is 100% precise on the sample and
# the query 36%; merging them is a different feature, and the exact half has to come
# first or it drowns).


def _saved_slice_by_name(cfg: Config, name: str) -> SavedSlice | None:
    for slice_ in cfg.features.saved_slices:
        if slice_.name == name:
            return slice_
    return None


def _saved_slices_payload(cfg: Config, db_path: Path, name: str | None, offset: int,
                          limit: int, encoder: TextEncoder | None = None) -> dict:
    """`GET /api/saved-slices` — the pins always, one page of the asked-for slice.

    The shape is `_search_payload`'s and deliberately so: a pinned slice IS a search, so
    the state of the index travels with every answer and an index that cannot rank says
    which of the two unavailable states it is in instead of coming back as an empty list.
    That rule is worth more here than in the search line — nobody types "children" into a
    pin, so an empty page would be read as a fact about the archive rather than as a
    question that missed.

    `name=None` is the tab's own call on open: the pins and the state, no ranking, no
    model. The phrases travel with the page because the panel prints them — a slice whose
    words are invisible cannot be edited by the person it is wrong for.

    F189: a pin whose single phrase is somebody's NAME answers with that person's frames,
    exactly as the search line does for the same string. Pinning is how a named person
    becomes an ordinary tab and it was supposed to cost nothing — but a pin that ranked
    «Ирина» by CLIP while the search line selected her cluster would be two answers under
    one word, and the divergence would be silent. A pin of SEVERAL phrases is a query and
    stays one: a name averaged with other words is not a name.
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
            # The one word the client needs to caption these lists apart from the exact
            # slices beside them. A constant rather than a per-slice flag: everything this
            # route serves is a ranking, and the day one of them is not, it will not be
            # served from here.
            "approximate": True,
            # F156: how many pins the interface may add (`features.max_pinned_slices`).
            # It travels with every answer so the "pin this" button can say the limit is
            # reached BEFORE somebody types a name for a slice that will be refused.
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
            # This list is a fact and not an estimate, and the word that says so is the
            # one the panel prints beside every ranking on this tab.
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

    An absent `slice` is NOT an error — it is how the pin row is asked for. A `slice` that
    is not in the config IS one, the `_parse_face_slice_query` rule: answering it with an
    empty page would show a slice that does not exist as one holding no photographs.
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
# for those 65 cover 26%, 23%, 22%, 20%, 18%, 17%, 15%, 12%, 12%, 6%. Not one of them
# reaches a third of a third. Ten slices for 65 frames out of 200 is the thirteen-control
# remote F133 took apart, and food — which both the user and the author had in mind as a
# large slice — came out at 8 frames, smaller than sky or signage.
#
# So the product stops guessing which facets matter. For one person they are mountains and
# children, for another receipts and cars, and nobody but the owner of the archive knows
# which. The mechanism is unchanged (F129 ranks, F151 pins) — what is new is WHO writes
# the list.
#
# Three properties are the feature:
#
# * the list lives in `config.yaml`, beside the slices that ship. The index does not
#   survive `reset` or a re-processing and the config file does, and a slice somebody named
#   is not something to lose to a re-index;
# * a pin is a SAVED QUERY and nothing else, so a pinned slice is indistinguishable from a
#   built-in one on screen — the same grid, the same album, the same counter — and it is
#   removed by unpinning it, which deletes a config entry and touches no file;
# * the number of pins is bounded (`features.max_pinned_slices`) and reaching the bound is
#   SAID. A pin that silently does not appear is worse than no pin.
#
# No suggestions, ever: the product does not offer to pin anything for you. That is the
# whole point of the feature, and a "you might want to pin «food»" would be the guessing
# it replaces, wearing a friendlier hat.

# Why a pin was refused, in one word the client can caption. Not a sentence: the reason
# has to be shown in the interface language, so the server sends the code and the catalog
# holds the three sentences.
_PIN_EMPTY = "empty"          # nothing was typed — there is no query to save
_PIN_DUPLICATE = "duplicate"  # a pin of that name is already there
_PIN_LIMIT = "limit"          # `features.max_pinned_slices` is reached


def _validate_pin_payload(payload: object) -> tuple[str, str] | None:
    """Parse the body of `POST /api/saved-slices/pin` -> (name, query). None -> 400.

    `query` is the text that was typed and is required; `name` is optional and defaults to
    the query itself, which is what the field is pre-filled with. Both are stripped, and an
    empty query is refused HERE rather than in the interface: a slice with no words would
    rank the collection by an arbitrary direction and look exactly like an answer
    (`search.encode_queries` refuses it for the same reason).
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

    Arrows and not drag-and-drop, one of the two and deliberately the smaller: the order
    is a list of at most a dozen names, a keyboard reaches an arrow, and a drop target is a
    second way to say the same thing that would then have to agree with the first.
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

    The new pin goes to the END of the list, where the person who made it will look for
    it: the order is theirs to change afterwards, and inserting somewhere clever would be
    the product having an opinion about a list it does not own.

    Emptiness is not one of the answers here — `_validate_pin_payload` has already
    refused it with `_PIN_EMPTY`, and a second copy of that rule is a second place for it
    to drift.
    """
    slices = tuple(cfg.features.saved_slices)
    if any(existing.name == name for existing in slices):
        return _PIN_DUPLICATE
    if len(slices) >= int(cfg.features.max_pinned_slices):
        return _PIN_LIMIT
    return (*slices, SavedSlice(name, (query.strip(),)))


def _pinned_without(cfg: Config, name: str) -> tuple[SavedSlice, ...] | None:
    """The pinned list with that slice gone, or None when there is no such slice.

    Nothing but the config entry is removed. Unpinning is not a deletion of anything on
    disk, and the confirmation the interface asks for says so — the frames the slice ranked
    are the collection's and were never the slice's to hold.
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

    The same arrangement as `_LazyClassifierHolder` and for the same two reasons — the
    model must not be loaded by merely starting the UI (most sessions never search), and
    it must not be loaded twice, since the search route and the album route both encode
    text. Tests replace `ui.text_encoder`, so the whole feature runs without a model.
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
