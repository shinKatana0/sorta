"""F182: the "Review" workspace — duplicates, blur, closed eyes, restoring.

Marks here decide what leaves for `_delete` during the layout, so everything that
writes `dedup_choice` is in this one module.
"""
from __future__ import annotations

import sqlite3
import threading
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

from .. import imaging, restore
from ..config import FeaturesConfig
from ..dedup import (KEEPER_SOURCE_SHARPNESS, TIER_SAME_IMAGE, TIER_SIMILAR,
                     GroupKeeper, exact_duplicate_summary, group_key, group_tier,
                     near_duplicate_groups, read_group_keepers)
# The blur list (F157) reads F155's `face_sharpness` through the indexer's own
# optional-column check — the two features could be merged in either order.
from ..indexer import _has_column
from ..junk import faces_stage_ran
from ..sorter import quality_slice_from, quality_slice_where
from .common import _connect, _parse_page_window, _trash_files, _validate_file_ids_payload
from .slices import _JUNK_NO_PREVIEW


# F66: near_duplicate_groups over tens of thousands of pHashes costs seconds and the
# tab re-requests it on every open. The payload is a few MB of JSON, hence two entries.
_DUPES_CACHE_MAX_ITEMS = 2
_DupesFingerprint = tuple[tuple[int, int], ...]
_DupesCacheKey = tuple[str, int, _DupesFingerprint]
_dupes_cache: OrderedDict[_DupesCacheKey, dict] = OrderedDict()
_dupes_cache_lock = threading.Lock()


def _dupes_cache_clear() -> None:
    """Drop the cached Duplicates payloads (test isolation)."""
    with _dupes_cache_lock:
        _dupes_cache.clear()


def _db_fingerprint(db_path: Path) -> _DupesFingerprint:
    """(st_mtime_ns, st_size) of the DB file AND its `-wal` sidecar.

    WAL mode: a commit can land entirely in `<db>-wal` and leave the main file
    untouched, so the `.db` stat alone would serve stale groups after a run. A missing
    file contributes (-1, -1).
    """
    fingerprint: list[tuple[int, int]] = []
    for p in (db_path, Path(f"{db_path}-wal")):
        try:
            st = p.stat()
        except OSError:
            fingerprint.append((-1, -1))
        else:
            fingerprint.append((st.st_mtime_ns, st.st_size))
    return tuple(fingerprint)


# F199: the answer names the captions, so "every group says which tier it is" is a
# property of the payload rather than of the markup. String KEYS, not text — the catalog
# is `strings.py` in three languages, as with `blur_order` and the restore refusals.
_TIER_CAPTIONS = {
    TIER_SAME_IMAGE: ("dupe_tier_same_image", "dupe_same_image_note"),
    TIER_SIMILAR: ("dupe_tier_similar", "dupe_similar_note"),
}


def _tier_captions(tier: str) -> dict[str, str]:
    """The two caption keys one tier is said with: the line, and the reasoning.

    An unknown tier falls back to SIMILAR — of the two, the one that promises nothing
    is the safe thing to say about a group the table does not recognise.
    """
    caption, why = _TIER_CAPTIONS.get(tier, _TIER_CAPTIONS[TIER_SIMILAR])
    return {"tier_caption": caption, "tier_why": why}


def _dupes_payload(db_path: Path, max_distance: int) -> dict:
    """The Duplicates screen: the three tiers of sameness, each with its own default.

    F194 — `{"exact": {...}, "groups": [...]}`, one answer holding all three
    (`dedup.TIER_*` states the measurement):

    * `exact` is a pair of NUMBERS, not a list: over byte-identical copies "choose which
      to keep" is a question about nothing. Collapsed means shown as a number, not
      deleted — no route on this path removes a file by itself;
    * `same_image` is one picture stored more than once, so the largest frame carries
      `recommended`: resolution and weight are facts, not taste;
    * `similar` carries NO recommendation — measured blind on 111 groups, no signal we
      have beats picking at random, and a highlighted frame reads as an answer. What it
      carries is an ORDER (`order`: `sharpness` or `size`), which is what sharpness
      honestly is.

    `action` is the current decision from `dedup_choice` — the human's own table, which
    nothing here writes or reorders. Cached (F66) under (db path, max_distance,
    `_db_fingerprint`): any write to the index recomputes the payload.
    """
    key: _DupesCacheKey = (str(db_path), max_distance, _db_fingerprint(db_path))
    with _dupes_cache_lock:
        cached = _dupes_cache.get(key)
        if cached is not None:
            _dupes_cache.move_to_end(key)
            return cached

    def remember(payload: dict) -> dict:
        with _dupes_cache_lock:
            _dupes_cache[key] = payload
            _dupes_cache.move_to_end(key)
            while len(_dupes_cache) > _DUPES_CACHE_MAX_ITEMS:
                _dupes_cache.popitem(last=False)
        return payload

    conn = _connect(db_path)
    try:
        exact = exact_duplicate_summary(conn)
        exact_json = {"copies": exact.copies, "originals": exact.originals}
        groups = near_duplicate_groups(conn, max_distance=max_distance)
        if not groups:
            return remember({"exact": exact_json, "groups": []})
        all_ids = [r["id"] for g in groups for r in g]
        placeholders = ",".join("?" * len(all_ids))
        wh = {
            r["id"]: (r["width"], r["height"])
            for r in conn.execute(
                f"SELECT id, width, height FROM files WHERE id IN ({placeholders})",
                all_ids,
            ).fetchall()
        }
        choices = {
            r["file_id"]: r["action"]
            for r in conn.execute(
                f"SELECT file_id, action FROM dedup_choice WHERE file_id IN ({placeholders})",
                all_ids,
            ).fetchall()
        }
        # F120: sharpness is only comparable INSIDE a group. Across the collection a
        # screenshot averages 2854 against a photograph's 1253, so a global ranking
        # sorts by content type rather than by focus.
        sharp = {
            r["file_id"]: r["sharpness"]
            for r in conn.execute(
                f"SELECT file_id, sharpness FROM frame_quality "
                f"WHERE file_id IN ({placeholders}) AND sharpness IS NOT NULL",
                all_ids,
            ).fetchall()
        }
        # F148: a group is addressed by a hash of its membership (dedup.group_key), so a
        # missing key means it was never asked about (a pair under
        # `keeper_min_group_size`) or has gained/lost a frame since. Either way it falls
        # back to its own sharpness, which is always available.
        keepers = read_group_keepers(
            conn, [group_key([r["id"] for r in g]) for g in groups])
    finally:
        conn.close()

    result = []
    for idx, group in enumerate(groups):
        frames = []
        for r in group:
            w, h = wh.get(r["id"], (None, None))
            frames.append({
                "file_id": r["id"],
                "name": Path(r["path"]).name,
                # Deciding which of two identical frames to keep is mostly a question of
                # WHERE they lie — the copy in "Sorted" beats the one in "Downloads".
                "src_dir": Path(r["path"]).parent.name,
                "src_path": str(Path(r["path"]).parent),
                "thumb_url": f"/thumb/{r['id']}",
                "width": w,
                "height": h,
                "size": r["size"],
                "sharpness": sharp.get(r["id"]),
                "action": choices.get(r["id"]),
                "recommended": False,
            })
        tier = group_tier([r["phash"] for r in group])
        keeper = keepers.get(group_key([f["file_id"] for f in frames]))
        if tier == TIER_SAME_IMAGE:
            frames, order = _order_by_size(frames), "size"
            # The one place in this payload where anything is proposed, and it is
            # checkable: same picture, so the larger file is the better copy of it.
            frames[0]["recommended"] = True
            recommended_by: str | None = "size"
        else:
            frames, order = _order_similar(frames, keeper)
            recommended_by = None
        result.append({"group": idx, "tier": tier, "frames": frames,
                       **_tier_captions(tier),
                       # What the frames are SORTED by — never who is best.
                       "order": order,
                       # Set only where a rule holds (`same_image`); None is the honest
                       # value elsewhere, since no signal we have beats a coin.
                       "recommended_by": recommended_by})
    return remember({"exact": exact_json, "groups": result})


def _order_by_size(frames: list[dict]) -> list[dict]:
    """The same picture, biggest copy first — resolution, then weight, then id.

    `file_id` closes the order so two runs over an unchanged group answer the same way.
    """
    return sorted(frames, key=lambda f: (-((f["width"] or 0) * (f["height"] or 0)),
                                         -(f["size"] or 0), f["file_id"]))


def _order_similar(frames: list[dict],
                   keeper: GroupKeeper | None) -> tuple[list[dict], str]:
    """Similar frames in an ORDER, and the name of what ordered them. Nothing is chosen.

    Sharpness is a fine order and a measured non-answer: blind labelling of 111 groups
    put it at 27% against 30.4% for random, so it decides what a person looks at FIRST
    and nothing else. It leads only when EVERY frame has it — a partial comparison would
    prefer whichever frames happened to be measured, and since F120 only personal
    photographs are measured at all. Otherwise the group falls back to size.

    A `group_keeper` row is honoured only when its source is sharpness. A row from the
    retired model question (F186) is ignored: that answer was measured to be a coin
    toss, and turning it into a position would smuggle it back as advice.
    """
    by_sharpness = all(f["sharpness"] is not None for f in frames)
    order = "sharpness" if by_sharpness else "size"
    ranked = sorted(frames, key=lambda f: (
        -(f["sharpness"] or 0.0) if by_sharpness else 0.0,
        -((f["width"] or 0) * (f["height"] or 0)), -(f["size"] or 0), f["file_id"]))
    if keeper is not None and keeper.source == KEEPER_SOURCE_SHARPNESS:
        lead = next((f for f in ranked if f["file_id"] == keeper.keeper_id), None)
        if lead is not None:
            ranked = [lead] + [f for f in ranked if f is not lead]
    return ranked, order


def _validate_group_payload(payload: object) -> tuple[list[int], int | None] | None:
    """Parse the body `{"group": [file_id,...], "keep_file_id": int?}`, None -> 400.

    keep_file_id may be absent (skip).
    """
    if not isinstance(payload, dict):
        return None
    group = payload.get("group")
    if (not isinstance(group, list) or not group
            or not all(isinstance(x, int) and not isinstance(x, bool) for x in group)):
        return None
    keep = payload.get("keep_file_id")
    if keep is not None and (not isinstance(keep, int) or isinstance(keep, bool)):
        return None
    return group, keep


def _apply_choice(db_path: Path, group: list[int], keep_file_id: int) -> None:
    """keeper -> action='keep', the other frames of the group -> 'to_delete'.

    Idempotent: ON CONFLICT overwrites the old decision, e.g. on moving the keeper.
    """
    now = datetime.now(timezone.utc).isoformat()
    conn = _connect(db_path)
    try:
        with conn:
            for fid in group:
                action = "keep" if fid == keep_file_id else "to_delete"
                conn.execute(
                    """INSERT INTO dedup_choice (file_id, action, updated_at)
                       VALUES (?, ?, ?)
                       ON CONFLICT(file_id) DO UPDATE SET
                           action = excluded.action, updated_at = excluded.updated_at""",
                    (fid, action, now),
                )
    finally:
        conn.close()


def _skip_group(db_path: Path, group: list[int]) -> None:
    """"Do not delete this group" — clears dedup_choice of the group's frames."""
    conn = _connect(db_path)
    try:
        placeholders = ",".join("?" * len(group))
        with conn:
            conn.execute(
                f"DELETE FROM dedup_choice WHERE file_id IN ({placeholders})", group
            )
    finally:
        conn.close()


def _validate_keep_ids(entry: dict, group: list[int],
                       legacy: int | None) -> list[int] | None:
    """The frames of one group a person chose to KEEP — several of them, since F194.

    `keep_file_ids: [int,...]` is the new shape; the F32 `keep_file_id: int` still
    works, since it is also the shape of the three single-keeper routes beside it.
    None -> invalid (400), and an EMPTY list counts as invalid: "keep none of them" is
    the one sentence this route must not be able to say — a group nobody chose in is
    simply not sent. A frame named twice is one keeper, not an error.
    """
    raw = entry.get("keep_file_ids")
    if raw is None:
        return None if legacy is None or legacy not in group else [legacy]
    if (not isinstance(raw, list) or not raw
            or not all(isinstance(x, int) and not isinstance(x, bool) for x in raw)
            or not all(x in group for x in raw)):
        return None
    return list(dict.fromkeys(raw))


def _validate_batch_choices_payload(
    payload: object,
) -> tuple[list[tuple[list[int], list[int]]], list[list[int]]] | None:
    """Parse the body `{"groups": [{"group": [...], "keep_file_ids": [int,...]}, ...],
    "skip": [[file_id,...], ...]}`. `skip` is optional (default []).

    None -> invalid. The whole body is validated BEFORE any DB write (F32: a 400 never
    leaves a partial write behind).
    """
    if not isinstance(payload, dict):
        return None
    raw_groups = payload.get("groups")
    if not isinstance(raw_groups, list) or not raw_groups:
        return None
    groups: list[tuple[list[int], list[int]]] = []
    for entry in raw_groups:
        if not isinstance(entry, dict):
            return None
        parsed = _validate_group_payload(entry)
        if parsed is None:
            return None
        group, keep = parsed
        keeps = _validate_keep_ids(entry, group, keep)
        if keeps is None:
            return None
        groups.append((group, keeps))
    raw_skip = payload.get("skip", [])
    if not isinstance(raw_skip, list):
        return None
    skip: list[list[int]] = []
    for entry in raw_skip:
        if (not isinstance(entry, list) or not entry
                or not all(isinstance(x, int) and not isinstance(x, bool) for x in entry)):
            return None
        skip.append(entry)
    return groups, skip


def _apply_batch_choices(
    db_path: Path, groups: list[tuple[list[int], list[int]]], skip: list[list[int]]
) -> int:
    """Apply the kept frames over all groups + clear the skipped ones, atomically.

    One transaction for the whole batch. Returns the number of saved (not skipped)
    groups. F194: a group keeps as MANY frames as the person named, and exactly those.
    """
    now = datetime.now(timezone.utc).isoformat()
    conn = _connect(db_path)
    try:
        with conn:
            for group, keeps in groups:
                kept = set(keeps)
                for fid in group:
                    action = "keep" if fid in kept else "to_delete"
                    conn.execute(
                        """INSERT INTO dedup_choice (file_id, action, updated_at)
                           VALUES (?, ?, ?)
                           ON CONFLICT(file_id) DO UPDATE SET
                               action = excluded.action, updated_at = excluded.updated_at""",
                        (fid, action, now),
                    )
            for group in skip:
                placeholders = ",".join("?" * len(group))
                conn.execute(
                    f"DELETE FROM dedup_choice WHERE file_id IN ({placeholders})", group
                )
    finally:
        conn.close()
    return len(groups)


def _trash_group(db_path: Path, group: list[int], keep_file_id: int
                 ) -> tuple[list[dict], list[dict]]:
    """The group's non-keepers -> (trashed, refused), the shared `_trash_files` path."""
    conn = _connect(db_path)
    try:
        placeholders = ",".join("?" * len(group))
        rows = conn.execute(
            f"SELECT id FROM files WHERE id IN ({placeholders})", group
        ).fetchall()
        ids_to_trash = [r["id"] for r in rows if r["id"] != keep_file_id]
    finally:
        conn.close()
    return _trash_files(db_path, ids_to_trash)


# --- F126: the "Review" workspace — duplicates, blur, closed eyes ------------------
# Two rules the slices are built on:
#
# * a decision is a row in `dedup_choice` and nothing else — a second deletion path in a
#   program that moves 300 GB of somebody's photographs is a second way to lose them.
#   `file_id` is its primary key, so a frame in two slices carries ONE decision;
# * nothing is ever marked automatically, and the measurement is why: reviewed by eye in
#   bands, blurred frames turn up in every band up to 400, and the blurred frame that
#   gets kept is the only photograph of a person or a place.
#
# F177 removed a fourth slice, "no subject": the model called 212 of 6 111 frames
# subjectless and by eye those 212 are ordinary photographs. Deleted rather than hidden.
#
# F150's "low resolution" is not folded INTO the blurred list — measured on 22 095
# photographs the two intersect by 3% (682 of the 706 frames under a megapixel are
# formally sharp), so mixing them would hide each inside the other.
_REVIEW_SLICES = ("dupes", "blurred", "eyes", "low_resolution")

# F139: which album kind each flat slice gathers into — the map that keeps the list and
# the album on one rule. The names differ because the switcher's are older than the
# album's, and renaming either half would move an API parameter for nothing. Duplicates
# have no kind: gathering them into a folder is not what they are for.
_REVIEW_SLICE_KIND = {"blurred": "blurred", "eyes": "eyes_closed",
                      "low_resolution": "low_resolution"}

# ASCENDING for all three, so the most damaged frame comes first. None of the orderings
# is a verdict — each decides what to look at first, not what to delete. `f.id` closes
# every one of them: without it, paging would drop and repeat frames of equal sharpness,
# openness or size at the seam.
_REVIEW_SLICE_ORDER = {
    "blurred": "fq.sharpness ASC, f.id",
    "eyes": "fq.eye_openness ASC, f.id",
    "low_resolution": "f.width * f.height ASC, f.id",
}

# F157 + F155: a frame with a sharpness measured inside its face is ordered by that
# number, and BEFORE every frame that has none. The two numbers are not on one scale (a
# variance over a whole preview against one over a 100-200 px crop), so they must never
# meet inside one comparison — hence `face_sharpness IS NULL` first, then each group by
# its own number. The face number is also the better signal: on frames that have a face
# it finds 62% of the blurred ones against 15% (F155, 68 labelled frames). NULL keeps
# its schema meaning — "not measured", never "sharp".
_BLURRED_ORDER_WITH_FACE = ("(fq.face_sharpness IS NULL), fq.face_sharpness ASC, "
                            "fq.sharpness ASC, f.id")


def _blurred_order_column(conn: sqlite3.Connection) -> str:
    """Which number orders the blur list on THIS database — the F155 column, or the frame.

    Asked of the schema rather than assumed: a database from before v25 has no
    `face_sharpness` at all, and the list has to open on it as it does anywhere else.
    """
    return ("face_sharpness" if _has_column(conn, "frame_quality", "face_sharpness")
            else "sharpness")


def _review_order(conn: sqlite3.Connection, slice_: str) -> str:
    """The ORDER BY of one flat slice, against `_review_from`."""
    if slice_ == "blurred" and _blurred_order_column(conn) == "face_sharpness":
        return _BLURRED_ORDER_WITH_FACE
    return _REVIEW_SLICE_ORDER[slice_]


# A card shows the number its slice is ABOUT. The absent one is selected as NULL rather
# than left out, so one row shape feeds one `_review_item_to_json`; and `low_resolution`
# has no `fq` alias to read at all (`quality_slice_from`).
_REVIEW_SLICE_COLUMNS = {
    "blurred": "fq.sharpness AS sharpness, NULL AS width, NULL AS height",
    "eyes": "fq.sharpness AS sharpness, NULL AS width, NULL AS height",
    "low_resolution": "NULL AS sharpness, f.width AS width, f.height AS height",
}

# The membership rule lives in sorter.py (`quality_slice_where`, `quality_slice_from`)
# and is read from there: the album of a slice and the list of it must be one set.


def _review_from(slice_: str) -> str:
    """The FROM of one flat slice — the shared rule, by slice name."""
    return quality_slice_from(_REVIEW_SLICE_KIND[slice_])


def _review_where(slice_: str, features: FeaturesConfig, *,
                  beyond: bool = False) -> tuple[str, list[object]]:
    """The WHERE of one flat slice + its parameters — the shared rule, by slice name.

    `beyond` is "show more": the blurred list opens to `features.blur_review_max` and
    the closed-eyes list to `features.eye_openness_max` (F179), and each runs on without
    a ceiling once asked.
    """
    return quality_slice_where(_REVIEW_SLICE_KIND[slice_], features, beyond=beyond)


def _review_count(conn: sqlite3.Connection, slice_: str,
                  features: FeaturesConfig) -> int:
    """How many frames one flat slice holds, under the same WHERE the page uses."""
    where, params = _review_where(slice_, features)
    return int(conn.execute(
        f"SELECT COUNT(*) {_review_from(slice_)} WHERE {where}", params).fetchone()[0])


def _review_flat_counts(conn: sqlite3.Connection,
                        features: FeaturesConfig) -> dict[str, int]:
    """The flat slice counters — plain aggregates, cheap enough for "Overview".

    EVERY slice is counted INSIDE its own window, so the chip, the "Overview" row and
    the length of the list the tab opens with are one number per slice. For the eyes
    (F179) a counter ignoring the window would advertise every frame a face was measured
    on — the whole face population, not the closed eyes.
    """
    return {name: _review_count(conn, name, features)
            for name in _REVIEW_SLICES if name != "dupes"}


# F133: "decided" is a row in `dedup_choice` and nothing else, so a slice empties as the
# person works through it and the warning on the "Layout" tab disappears on its own.
_PENDING_JOIN = " LEFT JOIN dedup_choice dc ON dc.file_id = f.id"


def _review_pending_count(conn: sqlite3.Connection, slice_: str,
                          features: FeaturesConfig) -> int:
    """How many frames of one flat slice still carry no decision."""
    where, params = _review_where(slice_, features)
    return int(conn.execute(
        f"SELECT COUNT(*) {_review_from(slice_)}{_PENDING_JOIN} "
        f"WHERE {where} AND dc.action IS NULL", params).fetchone()[0])


def _review_pending_counts(conn: sqlite3.Connection,
                           features: FeaturesConfig) -> dict[str, int]:
    """The undecided part of each flat slice, under the same WHERE the page uses."""
    return {name: _review_pending_count(conn, name, features)
            for name in _REVIEW_SLICES if name != "dupes"}


def _pending_dupe_groups(groups: list[dict]) -> int:
    """Duplicate groups carrying no decision — no query, the payload already says so.

    Decided as soon as ONE frame carries an action. "Do not delete this group" CLEARS
    those rows (`_skip_group`), so such a group counts as undecided again.
    """
    return sum(
        1 for g in groups
        if not any(f.get("action") for f in g.get("frames", []))
    )


def _review_item_to_json(row: sqlite3.Row, action: str | None) -> dict:
    """One card of a flat slice: a thumbnail, a name, a date, the slice's number, the
    decision."""
    path = Path(row["path"])
    return {
        "file_id": int(row["id"]),
        "name": path.name,
        "date": row["taken_at"],
        # With a burst of similar frames the folder is often the only thing that tells
        # them apart.
        "src_dir": path.parent.name,
        "src_path": str(path.parent),
        "sharpness": None if row["sharpness"] is None else float(row["sharpness"]),
        # F150: a thumbnail is the same 200 px whatever it was made from, so the pixel
        # count is the one thing a person cannot see and the one they are deciding on.
        "width": None if row["width"] is None else int(row["width"]),
        "height": None if row["height"] is None else int(row["height"]),
        "action": action,
        "thumb_url": f"/thumb/{int(row['id'])}",
        "video": imaging.is_video_path(path),
    }


def _review_payload(db_path: Path, slice_: str, offset: int, limit: int, *,
                    beyond: bool, features: FeaturesConfig,
                    max_distance: int) -> dict:
    """`GET /api/review` — the slice counters + one bounded page of the current slice.

    `counts` is always the full set: a slice with nothing in it stays in the list showing
    a zero, because "you have no closed eyes" is an answer and a vanished entry is a
    riddle. `dupes` counts GROUPS and comes from the cached `_dupes_payload`, so opening
    the workspace pays for it once; `slice='dupes'` carries no items, since forcing the
    one grouped slice into the flat shape would cost the keeper choice.

    `eyes_reason='no_faces_run'` (F125): the eye number is measured only where a face was
    found, so without a faces run the honest answer is why there is no data, not a zero.

    The thresholds (`blur_max`, `eye_max`, `low_resolution_mp`) travel with the answer so
    the hint above the grid states the rule the list was built by, instead of repeating a
    default in JS. `window_total` is the count of the CURRENT slice's window, which
    "show more" walks past into the frames the ranking is less sure about.
    """
    conn = _connect(db_path)
    try:
        counts = _review_flat_counts(conn, features)
        pending = _review_pending_counts(conn, features)
        eyes_reason = None if faces_stage_ran(conn) else "no_faces_run"
        window_total = counts.get(slice_, counts["blurred"])
        blur_order = _blurred_order_column(conn)
        items: list[dict] = []
        total = 0
        if slice_ != "dupes":
            source = _review_from(slice_)
            where, params = _review_where(slice_, features, beyond=beyond)
            total = int(conn.execute(
                f"SELECT COUNT(*) {source} WHERE {where}", params).fetchone()[0])
            rows = conn.execute(
                f"""SELECT f.id, f.path, f.taken_at, {_REVIEW_SLICE_COLUMNS[slice_]}
                    {source} WHERE {where}
                    ORDER BY {_review_order(conn, slice_)}
                    LIMIT ? OFFSET ?""", [*params, limit, offset]).fetchall()
            actions: dict[int, str] = {}
            if rows:
                ids = [int(r["id"]) for r in rows]
                placeholders = ",".join("?" * len(ids))
                actions = {
                    int(r["file_id"]): r["action"]
                    for r in conn.execute(
                        f"SELECT file_id, action FROM dedup_choice "
                        f"WHERE file_id IN ({placeholders})", ids).fetchall()
                }
            items = [_review_item_to_json(r, actions.get(int(r["id"]))) for r in rows]
    finally:
        conn.close()
    # F194: the number of GROUPS on screen, i.e. the second and third tiers. The first
    # is a number and not a list, so it cannot be part of a count of things to look at.
    groups = _dupes_payload(db_path, max_distance)["groups"]
    counts["dupes"] = len(groups)
    pending["dupes"] = _pending_dupe_groups(groups)
    if slice_ == "dupes":
        total = counts["dupes"]
    return {
        "slice": slice_,
        "grouped": slice_ == "dupes",
        # F139: the client draws its "gather into a folder" row from this and never from
        # a table of its own. None for the duplicates.
        "album_kind": _REVIEW_SLICE_KIND.get(slice_),
        "counts": [{"slice": name, "count": counts[name]} for name in _REVIEW_SLICES],
        # F133: what the "Layout" tab warns about. The per-slice breakdown rides along
        # because it costs nothing and says WHERE the work is left.
        "pending": [{"slice": name, "count": pending[name]} for name in _REVIEW_SLICES],
        "pending_total": sum(pending.values()),
        "eyes_reason": eyes_reason,
        "blur_max": float(features.blur_review_max),
        # F157: the caption says which number ordered the list out loud — "frames with a
        # face are ordered by the sharpness of the face" is the one thing that explains
        # why a visibly sharp street can sit above a soft portrait.
        "blur_order": blur_order,
        # F179: the caption shown with this states the PRECISION measured at it, not a
        # count — 62% right is the fact a person needs before looking at the list.
        "eye_max": float(features.eye_openness_max),
        "low_resolution_mp": float(features.low_resolution_mp),
        "window_total": window_total,
        "beyond": bool(beyond),
        "total": total,
        "offset": offset,
        "limit": limit,
        "items": items,
    }


def _parse_review_query(
    query: dict[str, list[str]],
) -> tuple[str, int, int, bool] | None:
    """(slice, offset, limit, beyond) for `GET /api/review`, or None -> 400.

    An unknown slice is refused rather than answered with an empty page: the switcher
    offers exactly `_REVIEW_SLICES`, so anything else is a client that has lost track.
    """
    window = _parse_page_window(query)
    if window is None:
        return None
    slice_ = ((query.get("slice") or [_REVIEW_SLICES[0]])[0].strip()
              or _REVIEW_SLICES[0])
    if slice_ not in _REVIEW_SLICES:
        return None
    beyond = (query.get("beyond") or ["0"])[0].strip() in ("1", "true")
    return slice_, window[0], window[1], beyond


_REVIEW_MARK_ACTIONS = ("keep", "to_delete", "clear")


def _validate_review_mark_payload(payload: object) -> tuple[list[int], str] | None:
    """Parse the body `POST /api/review/mark`:
    `{"file_ids": [int,...], "action": "keep"|"to_delete"|"clear"}`.

    None -> invalid (400). The ids go through the same `_validate_file_ids_payload` as
    every other bulk route — ints only, never a path.
    """
    if not isinstance(payload, dict):
        return None
    ids = _validate_file_ids_payload(payload)
    if ids is None:
        return None
    action = payload.get("action")
    if action not in _REVIEW_MARK_ACTIONS:
        return None
    return ids, action


def _apply_review_mark(db_path: Path, ids: list[int], action: str) -> int:
    """Write the decision of a flat slice into `dedup_choice`; returns how many landed.

    The same table and the same two values the duplicates half writes: one decision per
    file, understood by one consumer (`sorter`). `clear` removes the row — "I have not
    decided", which is not `keep`; `keep` is what survives the next run, so the two or
    three blurred frames a person keeps for the memory are not asked about again.

    Nothing here touches a file on disk. An id outside the current index is skipped.
    """
    now = datetime.now(timezone.utc).isoformat()
    conn = _connect(db_path)
    try:
        placeholders = ",".join("?" * len(ids))
        known = [int(r["id"]) for r in conn.execute(
            f"SELECT id FROM files WHERE id IN ({placeholders})", ids).fetchall()]
        if not known:
            return 0
        known_placeholders = ",".join("?" * len(known))
        with conn:
            if action == "clear":
                conn.execute(
                    f"DELETE FROM dedup_choice WHERE file_id IN ({known_placeholders})",
                    known)
            else:
                conn.executemany(
                    """INSERT INTO dedup_choice (file_id, action, updated_at)
                       VALUES (?, ?, ?)
                       ON CONFLICT(file_id) DO UPDATE SET
                           action = excluded.action, updated_at = excluded.updated_at""",
                    [(fid, action, now) for fid in known])
    finally:
        conn.close()
    return len(known)


# --- F149: "try to improve" — one frame, by request, a copy beside it -----------------
# What this path deliberately does NOT do (see `restore` for the measurement):
#
# * more than ONE frame per press. `{"file_id": int}` and no list shape at all, and no
#   CLI command either — a model that draws plausible detail, applied in bulk, turns an
#   archive into a collection of convincing forgeries;
# * open the original for writing. The copy carries `_restored` in its name;
# * make a second copy on a repeat press (`restore.existing_copy` returns the first);
# * mark the original for deletion. Nothing on this path writes `dedup_choice` — the
#   copy simply becomes a frame the existing marking route can be used on.


def _restored_item_to_json(row: sqlite3.Row, source_file_id: int) -> dict:
    """One card for the processed copy — the shape of a review card, plus what it is.

    `action` is always None: the copy has just been created, so it must not arrive with
    a decision attached.
    """
    item = _review_item_to_json(row, None)
    item["restored"] = True
    item["source_file_id"] = int(source_file_id)
    return item


# --- F168: the second entrance — the expanded frame, in every slice ------------------
# F149 drew the button in the "blurred" slice alone, and the measurement of 2026-08-03
# says that place is almost empty: the Laplacian filter at its threshold finds 8% of the
# frames a person calls soft. F169 (80 blind pairs) says the gain is not about blur at
# all — it is about SIZE:
#
#     < 640 px    66% |  640-1024  58%  |  1024-1280  52%
#
# — a clean win on small frames, a coin toss by 1280. Hence ONE input, on the frame a
# person has already expanded, offered only below `features.restore_max_edge`.
#
# The bans are enforced HERE, in the route, not by not drawing a button (the F133 rule:
# a hidden control is not a rule, and a request made past the interface collects the
# same thing). F198 made the ceiling one of them: F169 refused nothing above it and
# warned after the fact, and the measurement of 2026-08-04 came back at 35/35/30 on
# blind pairs above the ceiling — i.e. nothing, for a run of the model and a useless
# file beside the original.
RESTORE_ERROR_SENSITIVE = "sensitive_class"
RESTORE_ERROR_VIDEO = "video"
RESTORE_ERROR_TOO_LARGE = "too_large"


def _restore_refusal(path: Path, verdict: str | None, media_type: str | None,
                     sensitive: frozenset[str]) -> str | None:
    """The code this frame may not be processed under, or None — the server-side bans.

    A private class (`vlm.exclude_classes`, `document` by default) is refused because
    processing one means decoding a passport or a medical form and drawing it four times
    larger. Video is refused because a clip has no single frame to be the answer.
    """
    if verdict is not None and verdict in sensitive:
        return RESTORE_ERROR_SENSITIVE
    if media_type == "video" or imaging.is_video_path(path):
        return RESTORE_ERROR_VIDEO
    return None


def _restore_source_row(conn: sqlite3.Connection, file_id: int) -> sqlite3.Row | None:
    """The source's path and the two facts the bans are decided from, or None.

    A LEFT JOIN on purpose: a frame nobody has classified yet (`media_class` is written by
    a run that may not have happened) is an ordinary photograph, not a refusal.
    """
    return conn.execute(
        """SELECT f.id, f.path, f.media_type, mc.verdict AS verdict
           FROM files f LEFT JOIN media_class mc ON mc.file_id = f.id
           WHERE f.id = ?""", (file_id,)).fetchone()


def _restore_notice(src: Path, max_edge: int) -> dict:
    """F169: what the answer owes about the ceiling — `rebuilt` and the two numbers.

    Recomputed from the source rather than remembered: the press that REUSES a copy owes
    the same sentence, and must not quietly drop the warning the first one carried.
    """
    edge = restore.source_edge(src)
    return {"rebuilt": edge > int(max_edge) > 0, "source_edge": edge,
            "max_edge": int(max_edge)}


def _restore_decision(path: Path, verdict: str | None, media_type: str | None,
                      sensitive: frozenset[str], max_edge: int) -> dict:
    """F198: may this frame be processed — the ONE answer both entrances read.

    `_restore_offer` shows it and `_restore_frame` enforces it; a second place deciding
    "is this allowed" drifted apart within a day last time.

    Above `features.restore_max_edge` the answer is NO: the model is x4, so a big frame
    would be reduced first and blown back up to about its own size, and the measurement
    of 2026-08-04 found nothing there (35/35/30 on blind pairs). Below the ceiling
    nothing is narrowed — that is where the gain was measured (62% against 10% for
    bicubic on small frames).

    The ORDER matters: a frame refused as a personal document is never measured, because
    the size comes off the file's header and that is a file this program does not open.
    """
    refusal = _restore_refusal(path, verdict, media_type, sensitive)
    if refusal is not None:
        return {"available": False, "reason": refusal, "rebuilt": False,
                "source_edge": 0, "max_edge": int(max_edge)}
    notice = _restore_notice(path, max_edge)
    if notice["rebuilt"]:
        return {"available": False, "reason": RESTORE_ERROR_TOO_LARGE, **notice}
    return {"available": True, "reason": None, **notice}


def _restore_frame(db_path: Path, features: FeaturesConfig, file_id: int,
                   sensitive: frozenset[str] = frozenset(_JUNK_NO_PREVIEW)) -> dict:
    """`POST /api/review/restore` for ONE id -> the card of the copy, or the reason.

    Reads the source's path from the index, never from the request. A reason travels as
    a CODE (`restore.ERROR_*`) the client translates: the weights come from the network
    and offline is an ordinary state, so "the model is not here" has to be readable.

    `rebuilt` travels with a successful answer too — the engine measures what it
    actually processed, so a file replaced between the check and the work is reported
    rather than passed off as untouched.

    F168: the default `sensitive` is the fallback list for the same reason
    `_junk_payload` has one — a privacy guard must not switch itself off through an
    omission. Every refusal is an ordinary reason (200 + a code), not an error.
    """
    conn = _connect(db_path)
    try:
        row = _restore_source_row(conn, file_id)
        if row is None:
            return {"ok": False, "error": "file not found"}
        # The same function the offer called: what is drawn and what is enforced cannot
        # be two decisions (F198).
        decision = _restore_decision(Path(row["path"]), row["verdict"], row["media_type"],
                                     sensitive, features.restore_max_edge)
        if not decision["available"]:
            return {"ok": False, "reason": decision["reason"],
                    "rebuilt": decision["rebuilt"],
                    "source_edge": decision["source_edge"],
                    "max_edge": decision["max_edge"]}
        model = features.restore_model
        notice = {key: decision[key] for key in ("rebuilt", "source_edge", "max_edge")}
        existing = restore.existing_copy(conn, file_id, model)
        if existing is not None:
            copy_id, copy_path = existing
            if Path(copy_path).exists():
                return {"ok": True, "reused": True, **notice,
                        "item": _restored_item_to_json(_restored_row(conn, copy_id), file_id)}
            # The person deleted it in their file manager: drawing a card for a file
            # that is gone is worse than doing the work again.
            restore.forget_copy(conn, copy_id)
        result = restore.restore_frame(Path(row["path"]), model,
                                       max_edge=features.restore_max_edge)
        if not result.ok or result.path is None:
            return {"ok": False, "reason": result.error, "detail": result.detail}
        notice = {"rebuilt": result.rebuilt, "source_edge": result.source_edge,
                  "max_edge": int(features.restore_max_edge)}
        copy_id = restore.record_restored(conn, file_id, result.path, model=model)
        item = _restored_item_to_json(_restored_row(conn, copy_id), file_id)
    finally:
        conn.close()
    # The copy is a new canonical file, so the cached duplicate payload no longer
    # describes the collection.
    _dupes_cache_clear()
    return {"ok": True, "reused": False, "item": item, **notice}


def _restored_row(conn: sqlite3.Connection, file_id: int) -> sqlite3.Row:
    """The copy's row in the shape `_review_item_to_json` reads.

    `sharpness` is selected as NULL rather than joined: the copy has no `frame_quality`
    row until the next run measures it, and a printed zero would claim a measurement
    nobody made. F150: the size IS real (`record_restored` measures the copy it wrote),
    and on the low-resolution slice — where ×4 turns 640×480 into 2560×1920 — the change
    in size is the whole result of the operation.
    """
    return conn.execute(
        "SELECT id, path, taken_at, NULL AS sharpness, width, height "
        "FROM files WHERE id = ?", (file_id,)).fetchone()


def _restored_source_json(conn: sqlite3.Connection, file_id: int) -> dict | None:
    """Where this frame was processed FROM, or None if it is not a copy at all.

    The link comes out of `restored_files`, never out of the name: the copy is an
    ordinary member of the collection — it lies in the city folder beside its source and
    can be gathered into an album — so wherever it turns up it has to say what it is.
    """
    row = conn.execute(
        """SELECT r.source_file_id AS file_id, f.path AS path
           FROM restored_files r JOIN files f ON f.id = r.source_file_id
           WHERE r.file_id = ?""", (file_id,)).fetchone()
    if row is None:
        return None
    return {"file_id": int(row["file_id"]), "name": Path(row["path"]).name}


def _restore_offer(db_path: Path, features: FeaturesConfig, file_id: int,
                   sensitive: frozenset[str] = frozenset(_JUNK_NO_PREVIEW)) -> dict | None:
    """`GET /api/restore/offer` — what the expanded frame affords; None -> 404.

    A VIEW of `_restore_decision`, which is also what the route enforces — never a
    second implementation. `available` is the bans, the ceiling among them (F198);
    `rebuilt` says the frame is above `features.restore_max_edge`, where the answer is
    `available: False` with `reason` `too_large` and the two numbers the sentence is
    built from. `restored_from` is the other direction: this frame IS a copy.
    """
    conn = _connect(db_path)
    try:
        row = _restore_source_row(conn, file_id)
        if row is None:
            return None
        decision = _restore_decision(Path(row["path"]), row["verdict"], row["media_type"],
                                     sensitive, features.restore_max_edge)
        return {
            "file_id": int(row["id"]),
            "restored_from": _restored_source_json(conn, file_id),
            **decision,
        }
    finally:
        conn.close()
