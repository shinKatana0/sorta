"""F182: the "Moves" tab — what the last batch did, and rolling it back.

The manifest of a batch and the undo that reads it. `_UndoState` extends the sort
state of `layout` because a rollback is the same kind of long job as the layout it
undoes, watched through the same progress protocol.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from .. import imaging
from ..config import Config
from ..sorter import undo
from .common import _connect, _log
from .layout import PlanCache, _SortState


def _target_rel(dst: str, dest_root: str) -> str:
    """dst relative to dest_root, as in PlanItem.target_rel (see sorter.py).

    ValueError (a path-case divergence on Windows, etc.) -> the full dst, the same
    fallback as in sorter._target_parts/plan_and_sort.
    """
    try:
        return Path(dst).relative_to(Path(dest_root)).as_posix()
    except ValueError:
        return Path(dst).as_posix()


def _moves_payload(db_path: Path, batch_id: int | None) -> dict:
    """The sort --apply batch manifest: batch metadata + the list of moves.

    batch_id=None -> the last batch (MAX(id) in move_batches). No batches ->
    {"batch": None, "moves": []}, without crashing. name/target_rel are computed from
    dst — independent of the current files row (a trashed file after a move still
    shows its path in the manifest, just without a preview).
    """
    conn = _connect(db_path)
    try:
        if batch_id is None:
            row = conn.execute(
                "SELECT id, mode, dest_root, started_at, finished_at, operation "
                "FROM move_batches ORDER BY id DESC LIMIT 1"
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT id, mode, dest_root, started_at, finished_at, operation "
                "FROM move_batches WHERE id = ?", (batch_id,)
            ).fetchone()
        if row is None:
            return {"batch": None, "moves": []}
        batch = dict(row)
        move_rows = conn.execute(
            "SELECT file_id, src, dst, status FROM moves "
            "WHERE batch_id = ? ORDER BY dst", (batch["id"],)
        ).fetchall()
    finally:
        conn.close()

    dest_root = batch["dest_root"]
    moves = [
        {
            "file_id": r["file_id"],
            "name": Path(r["dst"]).name,
            "src": r["src"],
            "dst": r["dst"],
            "target_rel": _target_rel(r["dst"], dest_root),
            "status": r["status"],
            "thumb_url": f"/thumb/{r['file_id']}",
            "video": imaging.is_video_path(r["dst"]),  # F80, as in _plan_item_to_json
        }
        for r in move_rows
    ]
    return {"batch": batch, "moves": moves}


class _UndoState(_SortState):
    """Thread-safe state of the background `/api/undo` rollback (F97).

    Deliberately the same shape as `_SortState` (running/done/total/error/finished/
    result + a cancel flag): the client polls it with the same code, and a rollback is
    the same kind of thing as a layout — one long operation over a file list that has
    to be stoppable. A separate class rather than a second `_SortState` instance so
    the cross-lock in the handlers reads as what it is: sort, process and undo are
    three named things that may not run at the same time.
    """


# --- F97: roll the last batch back from the UI (`POST /api/undo`) -------------
# The engine is `sorter.undo`, exactly the one behind the CLI `sorta undo` — the
# blake3 verification before deleting a copy, the interrupted tail and the closing of
# a batch left with finished_at=NULL all live there. Here, as with `/api/sort`, only
# the background thread, the progress snapshot and the cancel flag.
#
# The batch is resolved the same way `_moves_payload` resolves it — the LAST batch in
# move_batches, i.e. the very one the "Moves" tab is showing. `sorter.undo(None)` picks
# the last batch that has a 'done' move instead, which is a different batch in exactly
# the case this button exists for: a run interrupted before its first file finished.
# The button and the manifest next to it must talk about the same thing.


def _last_batch_id(conn: sqlite3.Connection) -> int | None:
    row = conn.execute("SELECT id FROM move_batches ORDER BY id DESC LIMIT 1").fetchone()
    return int(row["id"]) if row is not None else None


def _run_undo(db_path: Path, cfg: Config, state: _UndoState, cache: PlanCache) -> None:
    """The body of the `POST /api/undo` background thread: its own sqlite connection
    (not transferable between threads, like `_run_sort`).

    No batches at all -> an error in the state, not an exception: the button is only
    reachable when the manifest shows a batch, so this is a race, not a user mistake.
    A cancelled rollback is a normal result with `cancelled` set — what was undone
    stays undone and pressing the button again finishes the rest.

    `stray` (copies of an interrupted transfer whose hash does not match) travels to
    the client as a list of paths: those files are still lying in the result and only
    a human can decide what they are.
    """
    conn = _connect(db_path)
    error: str | None = None
    result: dict | None = None
    try:
        batch_id = _last_batch_id(conn)
        if batch_id is None:
            error = "no batches to undo"
        else:
            try:
                stats = undo(conn, batch_id, progress=state.set_progress,
                             should_cancel=state.cancel_requested)
            except ValueError as exc:
                error = str(exc)
            else:
                result = {
                    "batch_id": stats.batch_id,
                    "undone": stats.undone,
                    "missing": stats.missing,
                    "failed": stats.failed,
                    "cancelled": stats.cancelled,
                    "stray": stats.stray,
                }
                # As in _run_sort: the rollback already happened, so a preview cache
                # that would not rebuild is a soft signal, never a rollback error.
                try:
                    cache.rebuild(cfg, conn)
                except Exception:  # noqa: BLE001
                    _log.exception("sorta ui: план не обновлён после отката")
                    result["preview_stale"] = True
    finally:
        conn.close()
        state.finish(error, result)
