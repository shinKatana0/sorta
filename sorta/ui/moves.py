"""F182: the "Moves" tab — what the last batch did, and rolling it back."""
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
    fallback as sorter._target_parts.
    """
    try:
        return Path(dst).relative_to(Path(dest_root)).as_posix()
    except ValueError:
        return Path(dst).as_posix()


def _moves_payload(db_path: Path, batch_id: int | None) -> dict:
    """The sort --apply batch manifest: batch metadata + the list of moves.

    batch_id=None -> the last batch. No batches -> {"batch": None, "moves": []}.
    name/target_rel come from dst, not from the current files row: a file trashed
    after a move still shows its path in the manifest, just without a preview.
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

    A distinct class rather than a second `_SortState` instance so the cross-lock in
    the handlers reads as what it is: sort, process and undo may not run at once.
    """


# F97: the batch is resolved as `_moves_payload` resolves it — the LAST batch in
# move_batches, the one the "Moves" tab is showing. `sorter.undo(None)` picks the last
# batch with a 'done' move instead, a different batch in exactly the case this button
# exists for: a run interrupted before its first file finished.


def _last_batch_id(conn: sqlite3.Connection) -> int | None:
    row = conn.execute("SELECT id FROM move_batches ORDER BY id DESC LIMIT 1").fetchone()
    return int(row["id"]) if row is not None else None


def _run_undo(db_path: Path, cfg: Config, state: _UndoState, cache: PlanCache) -> None:
    """The body of the `POST /api/undo` background thread.

    Opens its own sqlite connection (a connection is not transferable between
    threads). No batches at all -> an error in the state, not an exception: the button
    is only reachable when the manifest shows a batch, so this is a race. A cancelled
    rollback is a normal result — what was undone stays undone and pressing the button
    again finishes the rest. `stray` (copies of an interrupted transfer whose hash
    does not match) travels to the client as paths: only a human can decide what they
    are.
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
                # The rollback already happened: a preview cache that would not
                # rebuild is a soft signal, never a rollback error.
                try:
                    cache.rebuild(cfg, conn)
                except Exception:  # noqa: BLE001
                    _log.exception("sorta ui: the plan was not rebuilt after the rollback")
                    result["preview_stale"] = True
    finally:
        conn.close()
        state.finish(error, result)
