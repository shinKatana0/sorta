"""F182: the "Review" workspace — duplicates, blur, closed eyes, restoring.

The four things a person opens the tab to go through, plus the second entrance F168
added: an expanded frame that can be improved by request. Marks here decide what
leaves for `_delete` during the layout, so everything that writes `dedup_choice` or
`review_mark` is in this one module.
"""
from __future__ import annotations

import sqlite3
import threading
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from .. import imaging, restore
from ..config import FeaturesConfig
from ..dedup import KEEPER_SOURCE_SHARPNESS, group_key, near_duplicate_groups, read_group_keepers
from ..indexer import _has_column
from ..junk import faces_stage_ran
from ..sorter import quality_slice_from, quality_slice_where
from .common import _connect, _parse_page_window, _trash_files, _validate_file_ids_payload
from .slices import _JUNK_NO_PREVIEW


# F66: near_duplicate_groups over tens of thousands of pHashes costs seconds, and the
# Duplicates tab re-requests it on every open. The payload is a few MB of JSON, so a
# couple of entries is all we keep (one per max_distance in practice).
_DUPES_CACHE_MAX_ITEMS = 2
_DupesFingerprint = tuple[tuple[int, int], ...]
_DupesCacheKey = tuple[str, int, _DupesFingerprint]
_dupes_cache: OrderedDict[_DupesCacheKey, list[dict]] = OrderedDict()
_dupes_cache_lock = threading.Lock()


def _dupes_cache_clear() -> None:
    """Drop the cached Duplicates payloads (test isolation)."""
    with _dupes_cache_lock:
        _dupes_cache.clear()


def _db_fingerprint(db_path: Path) -> _DupesFingerprint:
    """(st_mtime_ns, st_size) of the DB file AND its `-wal` sidecar.

    The schema runs in WAL mode, so a commit can land entirely in `<db>-wal` and
    leave the main file untouched — keying on the `.db` stat alone would serve stale
    groups after a pipeline run. A missing file contributes (-1, -1).
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


def _dupes_payload(db_path: Path, max_distance: int) -> list[dict]:
    """near_duplicate_groups -> JSON-compatible groups for the Duplicates tab.

    recommended (F14): the best frame of the group by (width*height, then size) desc.
    action — the current decision from dedup_choice (keep/to_delete/None).

    keeper_id/keeper_source (F148): the STORED recommendation of the group, if it has
    one — the row `group_keeper` has been getting since F132 and which nothing read.
    Where it exists it names the recommended frame (the star and the preselected radio
    follow it), and `keeper_source` says who chose: `model` or `sharpness`. A group
    without a row — a pair, or one whose membership changed since it was asked about —
    carries `None` in both and is ranked here exactly as it was before.

    Cached (F66) under (db path, max_distance, _db_fingerprint): any write to the
    index changes the fingerprint and the payload is recomputed.
    """
    key: _DupesCacheKey = (str(db_path), max_distance, _db_fingerprint(db_path))
    with _dupes_cache_lock:
        cached = _dupes_cache.get(key)
        if cached is not None:
            _dupes_cache.move_to_end(key)
            return cached

    def remember(payload: list[dict]) -> list[dict]:
        with _dupes_cache_lock:
            _dupes_cache[key] = payload
            _dupes_cache.move_to_end(key)
            while len(_dupes_cache) > _DUPES_CACHE_MAX_ITEMS:
                _dupes_cache.popitem(last=False)
        return payload

    conn = _connect(db_path)
    try:
        groups = near_duplicate_groups(conn, max_distance=max_distance)
        if not groups:
            return remember([])
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
        # F120: sharpness, where it is finally comparable. Across the collection it is
        # not — a screenshot averages 2854 against a photograph's 1253, so a global
        # ranking sorts by content type rather than by focus. Inside a near-duplicate
        # group the frames ARE the same picture, which is the one place the number
        # answers the question it was measured for: which of these five is in focus.
        sharp = {
            r["file_id"]: r["sharpness"]
            for r in conn.execute(
                f"SELECT file_id, sharpness FROM frame_quality "
                f"WHERE file_id IN ({placeholders}) AND sharpness IS NOT NULL",
                all_ids,
            ).fetchall()
        }
        # F148: a group is addressed by a hash of its membership (dedup.group_key), so a
        # key that is missing here means the group has never been asked about (a pair
        # under `keeper_min_group_size`) or has gained/lost a frame since it was. Both
        # readings lead to the same behaviour: no stored recommendation, the ranking
        # below decides, and the tab looks like it did before this feature.
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
                # Where the frame lies in the source, as the Cities tab shows it:
                # `src_dir` in the line, the full `src_path` in the tooltip. Deciding
                # which of two identical frames to keep is mostly a question of WHERE
                # they lie — the copy in "Sorted" beats the one in "Downloads".
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
        # Sharpness leads only when EVERY frame of the group has it. A partial comparison
        # would quietly prefer whichever frames happened to be measured — and after F120
        # only personal photographs are measured at all, so a mixed group is a real case,
        # not a corner one.
        by_sharpness = all(f["sharpness"] is not None for f in frames)
        best = min(
            frames,
            key=lambda f: (
                -(f["sharpness"] or 0.0) if by_sharpness else 0.0,
                -((f["width"] or 0) * (f["height"] or 0)),
                -(f["size"] or 0),
                f["file_id"],
            ),
        )
        # F148: the stored recommendation wins over the local ranking when the group has
        # one — that is the whole point of having computed it. It never widens what is
        # marked: it moves the star and the preselected keeper radio from one frame to
        # another, and `dedup_choice` is still written by the user's hand alone.
        keeper = keepers.get(group_key([f["file_id"] for f in frames]))
        keeper_source = None
        if keeper is not None:
            named = next((f for f in frames if f["file_id"] == keeper.keeper_id), None)
            if named is not None:
                best = named
                # Two words, not the prompt fingerprint the row carries: the user needs
                # to know WHO advises (trust in the advice depends on it), not which
                # revision of the question was asked.
                keeper_source = ("sharpness" if keeper.source == KEEPER_SOURCE_SHARPNESS
                                 else "model")
        best["recommended"] = True
        result.append({"group": idx, "frames": frames,
                       # Why this one — so the tab can say it instead of asking the user
                       # to trust a star. This is the LOCAL ranking's basis; when
                       # `keeper_source` is set, that is who named the starred frame.
                       "recommended_by": "sharpness" if by_sharpness else "resolution",
                       "keeper_id": best["file_id"] if keeper_source else None,
                       "keeper_source": keeper_source})
    return remember(result)


def _validate_group_payload(payload: object) -> tuple[list[int], int | None] | None:
    """Parse the body `{"group": [file_id,...], "keep_file_id": int?}`.

    None -> the body is invalid (not a JSON object / group is not a non-empty list of
    int / keep_file_id, if present, is not int). keep_file_id may be absent (skip).
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

    Idempotent: ON CONFLICT overwrites the old decision (e.g. when moving the keeper
    to another frame of the same group).
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


def _validate_batch_choices_payload(
    payload: object,
) -> tuple[list[tuple[list[int], int]], list[list[int]]] | None:
    """Parse the body `{"groups": [{"group": [...], "keep_file_id": int}, ...],
    "skip": [[file_id,...], ...]}`. `skip` is optional (default []).

    None -> the body is invalid: `groups` is not a non-empty list / any entry does not
    pass `_validate_group_payload` or its `keep_file_id` is absent/not in `group` /
    `skip` is not a list of lists of int. The whole body is validated, before any DB
    write (F32: atomicity — 400 without a partial write).
    """
    if not isinstance(payload, dict):
        return None
    raw_groups = payload.get("groups")
    if not isinstance(raw_groups, list) or not raw_groups:
        return None
    groups: list[tuple[list[int], int]] = []
    for entry in raw_groups:
        parsed = _validate_group_payload(entry)
        if parsed is None:
            return None
        group, keep = parsed
        if keep is None or keep not in group:
            return None
        groups.append((group, keep))
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
    db_path: Path, groups: list[tuple[list[int], int]], skip: list[list[int]]
) -> int:
    """Apply the keeper choice over all groups + clear the skipped ones, atomically.

    One transaction for the whole batch: either all groups are applied and all skips
    are cleared, or (on an exception before the call — validation already passed in
    _validate_batch_choices_payload) nothing changes. Returns the number of saved
    (not skipped) groups.
    """
    now = datetime.now(timezone.utc).isoformat()
    conn = _connect(db_path)
    try:
        with conn:
            for group, keep in groups:
                for fid in group:
                    action = "keep" if fid == keep else "to_delete"
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


def _trash_group(db_path: Path, group: list[int], keep_file_id: int) -> list[dict]:
    """The group's non-keepers -> trash (see `_trash_files` — the shared trash path)."""
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
# Three signals, one job: look at a frame and decide whether it stays. Duplicates have had
# a tab with the whole viewing-and-deleting machinery since U3; the other two have been
# computed into `frame_quality` since F113 and were not visible anywhere. So this is one
# place with SLICES rather than tabs — and the duplicates half is deliberately untouched:
# `/api/dupes` and its four write routes answer exactly as before, because that
# is the one path in the product that deletes files and it is the one path that has been
# run against the live collection.
#
# F177 removed a fourth slice, "no subject". The model was asked about 6 111 frames and
# called 212 of them subjectless; looked at by eye, those 212 are ordinary photographs, so
# the slice was showing a list assembled by nothing. It is deleted rather than hidden: a
# hidden slice comes back at the first edit of this file.
#
# Two rules the slices are built on:
#
# * a decision is a row in `dedup_choice` and nothing else. `to_delete` already means
#   "move into `_delete` on the next `sort --apply`" (sorter.py), and a second deletion
#   path in a program that moves 300 GB of somebody's photographs is a second way to lose
#   them. `file_id` is the primary key there, so a frame that shows up in two slices
#   carries ONE decision and shows it in both;
# * nothing is ever marked automatically. There is no "delete everything below the
#   threshold" route here, and the measurement is why: reviewed by eye in bands, blurred
#   frames turn up in every band up to 400, and the blurred frame that gets kept is the
#   only photograph of a person or a place. Sharpness ranks the list; a human decides.
#
# F150 adds a fifth, "low resolution", and it sits here rather than in a tab of its own
# for the same reason the other four share this one: all of them are "look at this and
# decide whether it stays". It is not folded INTO the blurred list either — measured on
# 22 095 photographs, the two populations intersect by 3% (682 of the 706 frames under a
# megapixel are formally sharp), so mixing them would hide each inside the other and leave
# a person sorting blur wondering why sharp little pictures keep appearing.
_REVIEW_SLICES = ("dupes", "blurred", "eyes", "low_resolution")

# F139: which album kind each flat slice gathers into — and, read the other way, the map
# that keeps the list and the album on one rule. The names differ because the switcher's
# are older than the album's (`eyes` is a chip label, `eyes_closed` is a folder), and
# renaming either half would move an API parameter for nothing. Duplicates have no kind:
# they are the grouped slice, the one where a keeper is chosen, and the one path in the
# program that deletes files — collecting them into a folder is not what they are for.
_REVIEW_SLICE_KIND = {"blurred": "blurred", "eyes": "eyes_closed",
                      "low_resolution": "low_resolution"}

# Every flat slice is ranked by the number it exists for, and for all three that means
# ASCENDING, so the most damaged frame is the first one a person sees: a blurred frame has
# little variance, a closed eye is a thin slit, a low-resolution frame has few pixels. None
# of the three orderings is a verdict — each decides what to look at first, not what to
# delete. F179 gave the eyes such a number; before it they went in index order, because the
# VLM answer behind them was a yes/no with nothing to sort by. `f.id` closes every one of
# them: frames of equal sharpness, equal openness or equal size must come back in the same
# order on every page, or paging would drop and repeat them at the seam.
_REVIEW_SLICE_ORDER = {
    "blurred": "fq.sharpness ASC, f.id",
    "eyes": "fq.eye_openness ASC, f.id",
    "low_resolution": "f.width * f.height ASC, f.id",
}

# F157 + F155: where a frame HAS a sharpness measured inside its face, that number orders
# it — and it orders it BEFORE every frame that has none. Two reasons, and neither is a
# preference:
#
# * the two numbers are not on one scale (a variance over a whole preview against one over
#   a 100-200 px crop, `features.face_sharpness_max` says why no factor converts them), so
#   they must never meet inside one comparison. `face_sharpness IS NULL` first, then each
#   group by its own number, is the only ordering that keeps that promise;
# * on frames that have a face the face number finds 62% of the blurred ones against 15%
#   for the whole-frame number (F155, 68 labelled frames). Reading the better signal first
#   is what a ranking is for.
#
# NULL keeps its schema meaning throughout — "not measured", never "sharp" — so a frame
# with no face, or one from a run before the column existed, simply sorts by the frame
# number in the second half of the list instead of dropping out of it.
_BLURRED_ORDER_WITH_FACE = ("(fq.face_sharpness IS NULL), fq.face_sharpness ASC, "
                            "fq.sharpness ASC, f.id")


def _blurred_order_column(conn: sqlite3.Connection) -> str:
    """Which number orders the blur list on THIS database — the F155 column, or the frame.

    The column is asked of the schema rather than assumed, because the order of F155 and
    F157 was never fixed: a database from before v25 has no `face_sharpness` at all, and
    the list has to open on it exactly as it does anywhere else. `_has_column` is the
    indexer's, which reads its own optional columns the same way (`files.orientation`);
    a second spelling of one PRAGMA would be a second thing to keep true.
    """
    return ("face_sharpness" if _has_column(conn, "frame_quality", "face_sharpness")
            else "sharpness")


def _review_order(conn: sqlite3.Connection, slice_: str) -> str:
    """The ORDER BY of one flat slice, against `_review_from`."""
    if slice_ == "blurred" and _blurred_order_column(conn) == "face_sharpness":
        return _BLURRED_ORDER_WITH_FACE
    return _REVIEW_SLICE_ORDER[slice_]


# The two extra columns a card carries, by slice — a card shows the number its slice is
# ABOUT and not every number the row happens to hold. The absent one is selected as NULL
# rather than left out so that one row shape feeds one `_review_item_to_json`; and for
# `low_resolution` there is no `fq` alias to read at all (`quality_slice_from`).
_REVIEW_SLICE_COLUMNS = {
    "blurred": "fq.sharpness AS sharpness, NULL AS width, NULL AS height",
    "eyes": "fq.sharpness AS sharpness, NULL AS width, NULL AS height",
    "low_resolution": "NULL AS sharpness, f.width AS width, f.height AS height",
}

# The membership rule itself lives in sorter.py (`quality_slice_where`,
# `quality_slice_from`) and is read from there rather than restated here: the album of a
# slice and the list of it must be the same set of frames, and two spellings of one
# condition drift.


def _review_from(slice_: str) -> str:
    """The FROM of one flat slice — the shared rule, by slice name."""
    return quality_slice_from(_REVIEW_SLICE_KIND[slice_])


def _review_where(slice_: str, features: FeaturesConfig, *,
                  beyond: bool = False) -> tuple[str, list[object]]:
    """The WHERE of one flat slice + its parameters — the shared rule, by slice name.

    `beyond` is "show more": the blurred list opens to `features.blur_review_max` and the
    closed-eyes list to `features.eye_openness_max` (F179), and each runs on without a
    ceiling once asked. Each window bounds its own slice alone.
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

    EVERY slice is counted INSIDE its own window, so the chip, the "Overview" row and the
    length of the list the tab opens with are one number per slice. F179 made that true of
    the eyes too: the slice is a ranking now, and a counter that ignored the window would
    advertise every frame a face was measured on — the whole face population, not the
    closed eyes.
    """
    return {name: _review_count(conn, name, features)
            for name in _REVIEW_SLICES if name != "dupes"}


# F133: the same flat slices again, counting only the frames NOBODY has decided about.
# "Decided" is a row in `dedup_choice` and nothing else — the rule the marks are written
# by — so a slice empties as the person works through it, which is what makes the warning
# on the "Layout" tab disappear on its own.
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

    A group counts as decided as soon as ONE of its frames carries an action: choosing a
    keeper writes `keep` on it and `to_delete` on the rest. "Do not delete this group"
    CLEARS those rows (`_skip_group`), so such a group is undecided again — which is the
    literal truth about it and the same thing the slice counters say.
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
        # Where it lies, as on the Cities and Duplicates lists: with a burst of similar
        # frames the folder is often the only thing that tells them apart.
        "src_dir": path.parent.name,
        "src_path": str(path.parent),
        "sharpness": None if row["sharpness"] is None else float(row["sharpness"]),
        # F150: the size of the picture, on the card of the slice that is about it. A
        # thumbnail is the same 200 px whatever it was made from, so the pixels are the
        # one thing a person cannot see and the one thing they are deciding on.
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

    `counts` is always the full set (it is what the switcher draws, and a slice with
    nothing in it stays in the list showing a zero: "you have no closed eyes" is an
    answer, a vanished entry is a riddle). `dupes` counts GROUPS and comes from the
    cached `_dupes_payload` — the same payload the duplicates half of the tab renders
    from, so opening the workspace pays for it once.

    `slice='dupes'` carries no items: duplicates are the one grouped slice, and forcing
    them into the flat shape would cost the keeper choice that the whole view is for.
    The client renders that slice from `/api/dupes`, exactly as it did when it was a tab
    of its own.

    `eyes_reason='no_faces_run'` (F125) — the eye number is measured only where a face was
    found, so without a faces run the honest answer is why there is no data, not a zero
    that looks like "your subjects all had their eyes open".

    F150: `low_resolution_mp` travels with the answer for the same reason `blur_max`
    does — and `eye_max` with it since F179 — the hint above the grid states the rule the
    list was built by instead of repeating a default in JS.

    F179: `window_total` is the count of the CURRENT slice's window, because every flat
    slice has one now — blurred down to `features.blur_review_max`, closed eyes down to
    `features.eye_openness_max` — and "show more" walks either of them past its window
    into the frames the ranking is less sure about.

    F157: for the blurred slice that window is the depth of the FIRST PAGE, so
    `window_total`, the chip's counter and the length of the list the tab opens with are
    one number — a length, not a population. `blur_order` says which column ordered it.
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
    groups = _dupes_payload(db_path, max_distance)
    counts["dupes"] = len(groups)
    pending["dupes"] = _pending_dupe_groups(groups)
    if slice_ == "dupes":
        total = counts["dupes"]
    return {
        "slice": slice_,
        "grouped": slice_ == "dupes",
        # F139: the album kind of the CURRENT slice, or None for the duplicates. The
        # client draws its "gather into a folder" row from this and never from a table of
        # its own — see `_REVIEW_SLICE_KIND`.
        "album_kind": _REVIEW_SLICE_KIND.get(slice_),
        "counts": [{"slice": name, "count": counts[name]} for name in _REVIEW_SLICES],
        # F133: what the "Layout" tab warns about — the part of the workspace nobody has
        # answered yet. `pending_total` is the one number the warning shows; the per-slice
        # breakdown rides along because it costs nothing and says WHERE the work is left.
        "pending": [{"slice": name, "count": pending[name]} for name in _REVIEW_SLICES],
        "pending_total": sum(pending.values()),
        "eyes_reason": eyes_reason,
        "blur_max": float(features.blur_review_max),
        # F157: which number ordered the blur list — `face_sharpness` where F155's column
        # exists, `sharpness` where it does not. The caption says so out loud, because
        # "frames with a face are ordered by the sharpness of the face" is the one thing
        # that explains why a visibly sharp street can sit above a soft portrait.
        "blur_order": blur_order,
        # F179: the number the closed-eyes caption is shown with — and that caption states
        # the PRECISION measured at it, not a count, because 62% right is the fact a person
        # needs before looking at the list.
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
    offers exactly `_REVIEW_SLICES`, so anything else is a client that has lost track of
    what it is asking for. The window is parsed by the plan-page rules
    (`_parse_page_window`).
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

    The same table and the same two values the duplicates half writes, on purpose: one
    decision per file, understood by one consumer (`sorter`, which moves `to_delete`
    into `_delete` on `--apply`). `clear` removes the row, i.e. "I have not decided",
    which is not the same as `keep` — and `keep` is what survives the next run, so the
    two or three blurred frames a person keeps for the memory are not asked about again.

    Nothing here touches a file on disk. An id outside the current index is skipped
    rather than written (`_trash_files` resolves ids the same way): a decision about a
    file the program does not know is not a decision about anything.
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
# The third action of the Review tab, next to the two it has had ("mark for deletion",
# "keep"). What it does NOT do is most of the design (see `restore` for the measurement
# and the reasoning):
#
# * ONE frame per press. `{"file_id": int}` and no list shape at all — a body carrying
#   `file_ids` is refused by the validator like any other malformed one. There is no CLI
#   command either. A model that draws plausible detail, applied in bulk, turns an archive
#   into a collection of convincing forgeries;
# * the original is never opened for writing, and the copy carries `_restored` in its name;
# * a repeat press returns the copy that already exists (`restore.existing_copy`) instead
#   of making a second one;
# * keeping the copy does NOT mark the original for deletion. Two decisions, and the
#   second one is the person's — the same line between advice and action as F148. Nothing
#   on this path writes `dedup_choice`; the copy simply becomes a frame the existing
#   marking route can be used on, exactly like its source.


def _restored_item_to_json(row: sqlite3.Row, source_file_id: int) -> dict:
    """One card for the processed copy — the shape of a review card, plus what it is.

    `restored` and `source_file_id` are what the client draws the badge from and where it
    inserts the card: beside the original, not at the end of the list and not in a dialog
    of its own. `action` is always None: the copy has just been created, so nobody has
    decided anything about it yet, and it must not arrive with a decision attached.
    """
    item = _review_item_to_json(row, None)
    item["restored"] = True
    item["source_file_id"] = int(source_file_id)
    return item


# --- F168: the second entrance — the expanded frame, in every slice ------------------
# F149 drew the button in ONE place, the "blurred" slice, and the measurement of
# 2026-08-03 says that place is almost empty: the Laplacian filter at its threshold finds
# 8% of the frames a person calls soft (it answers "how much detail is in the frame", not
# "is it in focus"). So the action sat behind a detector we had measured to be nearly
# blind, and the only way to reach it was to be lucky enough to be in that list.
#
# The second measurement (F169, 80 blind pairs) says where the action really belongs. The
# gain is not about blur at all — it is about SIZE:
#
#     < 640 px    66% |  640-1024  58%  |  1024-1280  52%
#
# — a clean win on small frames, a coin toss by 1280. Hence the shape of this entrance:
# ONE input, on the frame a person has already expanded (the lightbox, which every slice
# opens), and offered only while the frame is below `features.restore_max_edge`. Above the
# ceiling the offer is withdrawn AND the reason is said out loud (`_restore_offer`): a
# button there would promise what the measurement did not find, and a frame silently
# rebuilt from a quarter of itself is exactly the trade F169 exists to disclose.
#
# The two bans below are enforced HERE, in the route, and not by not drawing a button —
# the F133 rule: a hidden control is not a rule, and a request made past the interface
# collects the same thing.
RESTORE_ERROR_SENSITIVE = "sensitive_class"
RESTORE_ERROR_VIDEO = "video"


def _restore_refusal(path: Path, verdict: str | None, media_type: str | None,
                     sensitive: frozenset[str]) -> str | None:
    """The code this frame may not be processed under, or None — the server-side bans.

    A private class (`vlm.exclude_classes`, `document` by default) is refused because
    processing one means decoding a passport or a medical form and drawing it four times
    larger — the one thing the product deliberately never renders. Video is refused
    because the engine is about images: a clip has no single frame to be the answer.
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

    Recomputed from the source rather than remembered, because the same sentence is owed
    on the press that REUSES a copy: the frame and the ceiling are what they are, so the
    second press must not quietly drop the warning the first one carried.
    """
    edge = restore.source_edge(src)
    return {"rebuilt": edge > int(max_edge) > 0, "source_edge": edge,
            "max_edge": int(max_edge)}


def _restore_frame(db_path: Path, features: FeaturesConfig, file_id: int,
                   sensitive: frozenset[str] = frozenset(_JUNK_NO_PREVIEW)) -> dict:
    """`POST /api/review/restore` for ONE id -> the card of the copy, or the reason.

    Reads the source's path from the index (never from the request — the same rule every
    other route follows), hands it to `restore.restore_frame`, then indexes the result. A
    reason travels as a CODE (`restore.ERROR_*`), which the client translates: the weights
    come from the network and offline is an ordinary state for this product, so "the model
    is not here" has to be an answer a person can read rather than an empty result.

    F169: the ceiling comes from `features.restore_max_edge` and is PASSED — it used to be
    a constant the engine defaulted to, i.e. one number for every frame with nobody told —
    and the answer carries `rebuilt` whenever the frame was larger than it. The action is
    not refused for such a frame: what should happen to a 12 Mpx one is the measurement's
    decision (`scripts/measure_restore.py`), and until it is made the honest thing is to
    do the work and say what was done.

    F168: `sensitive` is `vlm.exclude_classes`, and the two bans it and `media_type` carry
    are refused HERE rather than by not drawing a button — this route is now reachable
    from every slice, and a rule that lives in the markup is a rule a request made past
    the interface never meets. The default is the fallback list for the same reason
    `_junk_payload` has one: a privacy guard must not switch itself off through an
    omission. Both refusals are ordinary reasons (200 + a code the client translates),
    not errors: the person pointed at a frame this action does not apply to, which is
    something the interface has to be able to say.
    """
    conn = _connect(db_path)
    try:
        row = _restore_source_row(conn, file_id)
        if row is None:
            return {"ok": False, "error": "file not found"}
        refusal = _restore_refusal(Path(row["path"]), row["verdict"], row["media_type"],
                                   sensitive)
        if refusal is not None:
            return {"ok": False, "reason": refusal}
        model = features.restore_model
        notice = _restore_notice(Path(row["path"]), features.restore_max_edge)
        existing = restore.existing_copy(conn, file_id, model)
        if existing is not None:
            copy_id, copy_path = existing
            if Path(copy_path).exists():
                return {"ok": True, "reused": True, **notice,
                        "item": _restored_item_to_json(_restored_row(conn, copy_id), file_id)}
            # The person deleted it in their file manager. Answering "you already have one"
            # and drawing a card for a file that is gone is worse than doing the work again.
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
    # The copy is a new canonical file, so the cached duplicate payload and the cached
    # layout no longer describe the collection. It is never a duplicate of its source
    # (`dedup`), which is a statement about the GROUPS and not about the cache.
    _dupes_cache_clear()
    return {"ok": True, "reused": False, "item": item, **notice}


def _restored_row(conn: sqlite3.Connection, file_id: int) -> sqlite3.Row:
    """The copy's row in the shape `_review_item_to_json` reads.

    `sharpness` is selected as NULL rather than joined: the copy has no `frame_quality`
    row and will not have one until the next run measures it, and a card that printed a
    zero would be claiming a measurement nobody made.

    F150: the size, on the other hand, is REAL and is selected as such. `record_restored`
    measures the copy it just wrote, and on the low-resolution slice — the model's proper
    addressee, where ×4 turns 640×480 into 2560×1920 — the change in size is the whole
    result of the operation. A card that hid it would leave the person comparing two
    thumbnails of identical width on screen.
    """
    return conn.execute(
        "SELECT id, path, taken_at, NULL AS sharpness, width, height "
        "FROM files WHERE id = ?", (file_id,)).fetchone()


def _restored_source_json(conn: sqlite3.Connection, file_id: int) -> dict | None:
    """Where this frame was processed FROM, or None if it is not a copy at all.

    The badge on the expanded frame is drawn from this, and the link comes out of
    `restored_files` rather than out of the name: the copy is an ordinary member of the
    collection now — it lies in the city folder beside its source, it can be gathered
    into an album — so wherever it turns up it has to say what it is, or it reads as a
    second similar photograph that came from nowhere.
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

    Read-only, and it is NOT a second implementation of the action: pressing still goes
    to the one route, and a reason still travels as the same code. This answers the two
    questions the expanded frame has to answer before anything is offered —

    * `available`: may this frame be processed at all (the bans the route enforces). A
      client that worked that out for itself would be a second copy of the privacy rule,
      which is the mistake F133 named;
    * `rebuilt`: is the frame ABOVE `features.restore_max_edge`, i.e. would the copy be
      rebuilt from a reduced version of itself. The measurement found the gain on small
      frames and nothing by 1280 px, so above the ceiling the offer is withdrawn and the
      two numbers are handed over for the sentence that says why. Silence there would be
      a promise the measurement does not support.

    `restored_from` is the other direction: this frame IS a copy, and here is the frame it
    was made from.

    A refused frame is not measured: the size comes off the file's header, and a frame
    classed as a personal document is one this program does not open for any purpose. The
    two numbers are what the "too large" sentence is built from, and there is no such
    sentence to build when the answer is already no.
    """
    conn = _connect(db_path)
    try:
        row = _restore_source_row(conn, file_id)
        if row is None:
            return None
        path = Path(row["path"])
        refusal = _restore_refusal(path, row["verdict"], row["media_type"], sensitive)
        notice = ({"rebuilt": False, "source_edge": 0,
                   "max_edge": int(features.restore_max_edge)} if refusal is not None
                  else _restore_notice(path, features.restore_max_edge))
        return {
            "file_id": int(row["id"]),
            "available": refusal is None,
            "reason": refusal,
            "restored_from": _restored_source_json(conn, file_id),
            **notice,
        }
    finally:
        conn.close()
