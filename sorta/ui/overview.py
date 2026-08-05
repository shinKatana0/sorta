"""F182: the "Overview" tab — the state of the collection in one screen.

Read-only by construction: it counts what the other tabs produced. That is why it
imports from `review` and `slices` and nothing imports from it — a number on this
screen must be the number that tab would show, so it asks that tab's own SQL.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from ..config import Config
from ..junk import faces_stage_ran
from ..sorter import FACE_SLICES
from .common import _OVERVIEW_LIVE, _connect
from .slices import _animals_count_sql, _face_slice_count
from .review import _review_flat_counts


# --- F108: the "Overview" tab — the state of the collection in one screen -----------
# Every number below is a plain aggregate over the index, and the plan is deliberately
# NOT built: a layout of 24k frames costs minutes, while this is the screen a user opens
# right AFTER a run to see what changed. Nothing is cached either — a number that is one
# run out of date answers the question wrongly, which is worse than not answering it.
#
# Privacy: aggregates only. No file path and no file id leaves this endpoint; the single
# path in the payload is the destination FOLDER of the last layout, because "where did it
# go" is one of the four questions the layout group exists to answer.

# The order the place groups are shown in: from the place we know exactly, through the
# ones inherited from a neighbour, down to no place at all.
_PLACE_CONFIDENCE_ORDER = ("manual", "exact_gps", "session_inferred", "trip_inferred",
                           "path_inferred", "visual", "unknown")


def _media_class_breakdown(conn: sqlite3.Connection, column: str) -> list[dict]:
    """`verdict`/`source`/`tier` -> [{"key": …, "count": n}], the biggest group first.

    The three breakdowns are counted over the same population, so each of them sums to
    the same `classes.total` — a `tier` split that does not add up to the number of
    classified files is exactly the confusion this tab exists to remove. `tier` is NULL
    for rows written before v11; that group travels as `key: null` and the view labels it.

    The column name is interpolated into the SQL — it never comes from a request, the
    three call sites below pass literals.
    """
    rows = conn.execute(
        f"""SELECT mc.{column} AS key, COUNT(*) AS n
            FROM files f JOIN media_class mc ON mc.file_id = f.id
            WHERE {_OVERVIEW_LIVE}
            GROUP BY mc.{column}""").fetchall()
    out = [{"key": r["key"], "count": int(r["n"])} for r in rows]
    out.sort(key=lambda b: (-b["count"], b["key"] or ""))
    return out


def _overview_place(conn: sqlite3.Connection) -> dict:
    """The place group: how each frame got its place, and how many have none at all.

    A manual place (F85c) wins over `places` as a whole, exactly as the sorter reads it —
    otherwise a frame the user placed by hand would be counted here as placeless. The
    `no_place` rule is `sorter._target_parts` verbatim: an unknown confidence, or neither
    a city nor a country. Every one of those frames ends up in `_Unsorted/no_place`, which
    is why this is the one number of the group that is shown even when it is zero.
    """
    total = conn.execute(
        f"SELECT COUNT(*) FROM files f WHERE {_OVERVIEW_LIVE}").fetchone()[0]
    rows = conn.execute(
        f"""SELECT CASE WHEN mp.file_id IS NOT NULL THEN 'manual'
                        ELSE COALESCE(p.confidence, 'unknown') END AS conf,
                   COUNT(*) AS n
            FROM files f
            LEFT JOIN places p ON p.file_id = f.id
            LEFT JOIN manual_places mp ON mp.file_id = f.id
            WHERE {_OVERVIEW_LIVE}
            GROUP BY conf""").fetchall()
    no_place = conn.execute(
        f"""SELECT COUNT(*) FROM files f
            LEFT JOIN places p ON p.file_id = f.id
            LEFT JOIN manual_places mp ON mp.file_id = f.id
            WHERE {_OVERVIEW_LIVE} AND mp.file_id IS NULL
                  AND (COALESCE(p.confidence, 'unknown') = 'unknown'
                       OR (p.city IS NULL AND p.country IS NULL
                           AND p.country_name IS NULL))""").fetchone()[0]
    counts = {r["conf"]: int(r["n"]) for r in rows}
    confidence = []
    for key in _PLACE_CONFIDENCE_ORDER:
        count = counts.pop(key, 0)
        if count:
            confidence.append({"key": key, "count": count})
    # A confidence value this list does not know about is still shown, under its raw name:
    # a place the index carries must never be invisible here.
    confidence += [{"key": key, "count": count}
                   for key, count in sorted(counts.items()) if count]
    return {
        "total": int(total),
        "confidence": confidence,
        "no_place": int(no_place),
        "no_place_percent": round(100.0 * no_place / total, 1) if total else 0.0,
    }


def _overview_layout(conn: sqlite3.Connection) -> dict:
    """The layout group: was anything moved, when, where, how, and was it finished.

    Only the LAST batch is described. `finished_at IS NULL` is the trace of an interrupted
    run — the tab says so explicitly instead of showing a batch that merely looks normal.
    """
    batches = conn.execute("SELECT COUNT(*) FROM move_batches").fetchone()[0]
    unfinished = conn.execute(
        "SELECT COUNT(*) FROM move_batches WHERE finished_at IS NULL").fetchone()[0]
    last = conn.execute(
        """SELECT id, mode, operation, dest_root, started_at, finished_at
           FROM move_batches ORDER BY started_at DESC, id DESC LIMIT 1""").fetchone()
    payload: dict = {"batches": int(batches), "unfinished": int(unfinished), "last": None}
    if last is None:
        return payload
    counted = conn.execute(
        """SELECT COUNT(*) AS files, COALESCE(SUM(status = 'done'), 0) AS done
           FROM moves WHERE batch_id = ?""", (last["id"],)).fetchone()
    payload["last"] = {
        "mode": last["mode"],
        "operation": last["operation"],
        "dest_root": last["dest_root"],
        "started_at": last["started_at"],
        "finished_at": last["finished_at"],
        "unfinished": last["finished_at"] is None,
        "files": int(counted["files"]),
        "done": int(counted["done"]),
    }
    return payload


def _overview_payload(db_path: Path, cfg: Config) -> dict:
    """`GET /api/overview` — the four groups of numbers the tab draws.

    `empty` is the whole answer for a fresh index: the view then invites the user to pick
    a folder instead of drawing a table of zeros.

    F152: the three face slices are counted here by the same `sorter.face_slice_ids_sql`
    the panel and the albums use, and they are the one group of rows that can answer
    `null` — without a faces run they are unmeasured, not empty, and `faces_reason` says
    so. `cfg` (rather than the single `blur_max` this used to take) is what carries the
    thresholds those three rules read.

    F126: the flat review slices are counted here too, by the SAME queries the
    workspace itself uses (`_review_flat_counts`) — a counter that disagrees with the
    list it links to is worse than no counter. The blur window comes from the same
    `features` (`blur_review_max`), so this row and that list say one number. The
    duplicates row above stays what it always was: exact copies found by hash, not the
    phash groups of the workspace, which cost seconds to build and have no place on a
    tab made of plain aggregates.
    """
    # F137 needs the thresholds and F152 needs them too, so the whole config comes in and
    # the features are unpacked once here rather than threaded as a second argument.
    features = cfg.features
    conn = _connect(db_path)
    try:
        files = conn.execute(
            """SELECT COUNT(*) AS files,
                      COALESCE(SUM(media_type <> 'video'), 0) AS photos,
                      COALESCE(SUM(media_type = 'video'), 0) AS videos,
                      COALESCE(SUM(dup_of IS NOT NULL), 0) AS duplicates,
                      COALESCE(SUM(error IS NOT NULL), 0) AS errors
               FROM files""").fetchone()
        events = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        # F123: counted over the same population as the "Animals" tab and the animal
        # album, so the three cannot disagree. F124: which now means the one shared rule
        # (`_animals_count_sql` -> `sorter.animal_ids_sql`) — a frame the user unmarked
        # leaves this number exactly as it leaves the album. F137: and a threshold the
        # user edited moves it here, in the tab and in the album together.
        animals = conn.execute(_animals_count_sql(cfg)).fetchone()[0]
        faces_ran = faces_stage_ran(conn)
        faces_counts: dict[str, int | None] = {
            name: (_face_slice_count(conn, cfg, name) if faces_ran else None)
            for name in FACE_SLICES
        }
        review = _review_flat_counts(conn, features)
        place = _overview_place(conn)
        classes_total = conn.execute(
            f"""SELECT COUNT(*) FROM files f JOIN media_class mc ON mc.file_id = f.id
                WHERE {_OVERVIEW_LIVE}""").fetchone()[0]
        updated_at = conn.execute(
            f"""SELECT MAX(mc.updated_at) FROM files f
                JOIN media_class mc ON mc.file_id = f.id
                WHERE {_OVERVIEW_LIVE}""").fetchone()[0]
        tiers = _media_class_breakdown(conn, "tier")
        classes = {
            "total": int(classes_total),
            "verdicts": _media_class_breakdown(conn, "verdict"),
            "sources": _media_class_breakdown(conn, "source"),
            "tiers": tiers,
            # "Did the deep tier run at all" — the question that used to be answered by a
            # query into the database. A file the vlm tier deliberately skipped keeps
            # source='clip' but tier='vlm', so the TIER is what answers it (schema v11).
            "vlm_ran": any(t["key"] == "vlm" for t in tiers),
            "updated_at": updated_at,
        }
        layout = _overview_layout(conn)
    finally:
        conn.close()
    return {
        "empty": int(files["files"]) == 0,
        "collection": {
            "files": int(files["files"]),
            "photos": int(files["photos"]),
            "videos": int(files["videos"]),
            "duplicates": int(files["duplicates"]),
            "errors": int(files["errors"]),
            "events": int(events),
            "animals": int(animals),
            # F152: `null` where the faces stage never ran — the F125 rule, and the same
            # distinction `/api/face-slices` draws between "none" and "not asked".
            "with_people": faces_counts["people"],
            "group_photos": faces_counts["group"],
            "portraits": faces_counts["portrait"],
            "faces_reason": None if faces_ran else "no_faces_run",
            "blurred": review["blurred"],
            "eyes_closed": review["eyes"],
            # F150: the same query the slice itself runs, so the row and the list it
            # links to cannot say two different numbers.
            "low_resolution": review["low_resolution"],
        },
        "place": place,
        "classes": classes,
        "layout": layout,
    }
