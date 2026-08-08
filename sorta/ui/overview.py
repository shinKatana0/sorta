"""F182: the "Overview" tab — the state of the collection in one screen.

Imports from `review` and `slices` and is imported by nothing: a number on this
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


# F108: plain aggregates, no plan and no cache. Building a layout costs minutes on 24k
# frames, and a cached number one run out of date answers the question wrongly.
#
# Privacy: aggregates only. No file path and no file id leaves this endpoint; the one
# path in the payload is the destination FOLDER of the last layout.

# Shown in this order: from the place we know exactly, through the ones inherited from
# a neighbour, down to no place at all.
_PLACE_CONFIDENCE_ORDER = ("manual", "exact_gps", "session_inferred", "trip_inferred",
                           "path_inferred", "visual", "unknown")


def _media_class_breakdown(conn: sqlite3.Connection, column: str) -> list[dict]:
    """`verdict`/`source`/`tier` -> [{"key": …, "count": n}], the biggest group first.

    All three are counted over the same population, so each sums to `classes.total`.
    `tier` is NULL for rows written before schema v11; that group travels as
    `key: null`. The column name is interpolated into the SQL — it never comes from a
    request, the three call sites pass literals.
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

    A manual place (F85c) wins over `places` as a whole, as the sorter reads it —
    otherwise a frame the user placed by hand would count as placeless. The `no_place`
    rule is `sorter._target_parts` verbatim: an unknown confidence, or neither a city
    nor a country. Those frames land in `_Unsorted/no_place`, which is why this number
    is shown even when it is zero.
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
    # A confidence this list does not know is still shown, under its raw name: a place
    # the index carries must never be invisible here.
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

    Only the LAST batch is described. `finished_at IS NULL` is the trace of an
    interrupted run, and the tab says so rather than showing a batch that looks normal.
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

    Every counted row runs the query its own tab runs — face slices through
    `sorter.face_slice_ids_sql` (F152), the flat review slices through
    `_review_flat_counts` (F126). The face slices are the one group that can answer
    `null`: without a faces run they are unmeasured, not empty. `duplicates` is exact
    copies found by hash, not the phash groups of the workspace, which cost seconds to
    build and have no place on a tab made of plain aggregates.
    """
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
        # F123/F124/F137: one shared rule (`_animals_count_sql` ->
        # `sorter.animal_ids_sql`), so an unmarked frame or an edited threshold moves
        # this number, the "Animals" tab and the animal album together.
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
            # A file the vlm tier deliberately skipped keeps source='clip' but
            # tier='vlm', so the TIER answers "did the deep tier run" (schema v11).
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
            # F152: `null` where the faces stage never ran — "none" and "not asked" are
            # different answers, as in `/api/face-slices`.
            "with_people": faces_counts["people"],
            "group_photos": faces_counts["group"],
            "portraits": faces_counts["portrait"],
            "faces_reason": None if faces_ran else "no_faces_run",
            "blurred": review["blurred"],
            "eyes_closed": review["eyes"],
            "low_resolution": review["low_resolution"],
        },
        "place": place,
        "classes": classes,
        "layout": layout,
    }
