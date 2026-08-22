"""F242: the collection moved — carry the index over instead of indexing it again.

The index is keyed by absolute path and incrementality is path + mtime + size, so a file
whose path changed is a NEW file. A different drive letter or a renamed folder therefore
costs a full re-run — an hour, recoverable — and every decision hanging off the old rows:
face names, manual places, animal marks, duplicate choices, layout overrides. Those are
the only things in the product a person typed by hand, and nothing recovers them.

Rewriting the prefix keeps every `files.id`, so nothing that references an id notices the
move at all.
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from .config import Config
from .db import connect
from .faults import Fault

_log = logging.getLogger(__name__)

_EXAMPLES = 3

# SQLite has no escape sequences in string literals, so this really is one backslash.
_FOLD = "replace({column}, '\\', '/')"


class RelocateError(Fault, RuntimeError):
    """A refusal: the move cannot be applied, and nothing has been written."""

    codes = ("relocate_same_prefix", "relocate_no_index", "relocate_target_missing",
             "relocate_no_rows", "relocate_collisions")


class CollectionMoved(Fault, RuntimeError):
    """Every source root is gone while the index is full — the move this module undoes."""

    codes = ("relocate_collection_moved",)


_MOVED_HINT = (
    "the collection was not found: none of the source folders exist ({roots}), while the "
    "index already holds {rows} files (for example {sample}). The usual cause is a move — "
    "another drive letter, a renamed folder, a volume mounted somewhere else. Indexing "
    "from here would call every file new and lose the face names, places, marks and "
    "duplicate decisions entered by hand, so it stops instead. Carry the index over with "
    "`sorta relocate --from <old> --to <new>`, or point `sources` at the folder that "
    "exists."
)


def refuse_if_the_collection_moved(cfg: Config, conn: sqlite3.Connection) -> None:
    """Raise before a moved collection is silently indexed from scratch.

    The threshold is EVERY root missing and the index not empty. One missing root is an
    ordinary thing — a second source folder named before it is plugged in — and an empty
    index is a first run, which has nothing to lose and so says nothing.

    Here rather than in the indexer because it is the same fact as the rest of the
    module, seen from the other end; the indexer only calls it.
    """
    roots = [Path(src).expanduser() for src in cfg.sources]
    if not roots or any(root.exists() for root in roots):
        return
    row = conn.execute("SELECT count(*) AS n, min(path) AS sample FROM files").fetchone()
    if not row["n"]:
        return
    listed = ", ".join(str(root) for root in roots)
    raise CollectionMoved(
        _MOVED_HINT.format(roots=listed, rows=row["n"], sample=row["sample"]),
        "relocate_collection_moved",
        roots=listed, rows=row["n"], sample=row["sample"])


@dataclass(frozen=True)
class ColumnHits:
    table: str
    column: str
    rows: int


@dataclass
class RelocatePlan:
    """What a move would change. `applied` says whether it already did."""

    old_prefix: str
    new_prefix: str
    rows: int = 0
    columns: list[ColumnHits] = field(default_factory=list)
    examples: list[tuple[str, str]] = field(default_factory=list)
    new_prefix_exists: bool = False
    applied: bool = False


def normalize_prefix(value: str | Path) -> str:
    """A prefix in the one spelling the comparison happens in: absolute, POSIX, no
    trailing separator.

    The same expanduser + resolve a source root goes through, so `D:\\Photos`,
    `D:/Photos/` and `~/photos` all reduce to the one form the database is matched
    against. It does not touch case: a prefix that differs from the stored one in case
    matches nothing, and the run is refused for having found nothing rather than moving
    the wrong rows.
    """
    return Path(value).expanduser().resolve().as_posix().rstrip("/")


def _moved(value: str, old: str, new: str) -> str | None:
    """`value` with `old` swapped for `new`, or None if it does not sit under `old`.

    Two things the obvious `str.replace` would get wrong. The match ends on a component
    boundary, so `/photos` leaves `/photos-backup` alone. And separators are folded for
    the comparison, because a path is stored as `str(Path)` — backslashes on Windows —
    while the prefixes arrive normalized to POSIX; the row then keeps its own style, as
    half a database written in the other one is worse than either.
    """
    folded = value.replace("\\", "/")
    if folded != old and not folded.startswith(old + "/"):
        return None
    head, tail = value[:len(old)], value[len(old):]
    prefix = new.replace("/", "\\") if "\\" in head else new
    return prefix + tail


def _tables(conn: sqlite3.Connection) -> list[str]:
    return [row["name"] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'")]


def _is_text(declared: str | None) -> bool:
    """SQLite's own TEXT-affinity rule, over the declared type."""
    upper = (declared or "").upper()
    return any(token in upper for token in ("CHAR", "CLOB", "TEXT"))


def text_columns(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    """Every column that COULD hold a path: TEXT affinity, in every table.

    Discovered, never listed. A list here is the guard that is true and useless at the
    same time — the next feature adds a table with a path in it and the move goes on
    being silently partial. Nothing here decides by column NAME either: what selects a
    row is that its value sits under the old prefix.
    """
    columns: list[tuple[str, str]] = []
    for table in _tables(conn):
        for row in conn.execute(f'PRAGMA table_info("{table}")'):
            if _is_text(row["type"]):
                columns.append((table, row["name"]))
    return columns


def _hits(conn: sqlite3.Connection, table: str, column: str,
          old: str, new: str) -> Iterator[tuple[int, str, str]]:
    """(rowid, before, after) for every value of one column that sits under `old`."""
    folded = _FOLD.format(column=f'"{column}"')
    sql = (f'SELECT rowid AS rid, "{column}" AS value FROM "{table}" '
           f'WHERE "{column}" IS NOT NULL AND substr({folded}, 1, ?) = ?')
    for row in conn.execute(sql, (len(old), old)):
        before = row["value"]
        if not isinstance(before, str):
            continue
        after = _moved(before, old, new)
        if after is not None and after != before:
            yield row["rid"], before, after


def _update_column(conn: sqlite3.Connection, table: str, column: str,
                   old: str, new: str) -> int:
    updates = [(rid, after) for rid, _before, after in _hits(conn, table, column, old, new)]
    for rid, after in updates:
        conn.execute(f'UPDATE "{table}" SET "{column}" = ? WHERE rowid = ?', (after, rid))
    return len(updates)


def _conflicts(conn: sqlite3.Connection, old: str, new: str) -> list[str]:
    """Paths that two `files` rows would share after the move.

    Compared with separators folded rather than byte for byte: the UNIQUE index on
    `files.path` only catches the identical spelling, and two rows that differ by a
    slash are still one file arriving twice.
    """
    seen: set[str] = set()
    clashes: list[str] = []
    for row in conn.execute("SELECT path FROM files"):
        after = _moved(row["path"], old, new)
        final = row["path"] if after is None else after
        key = final.replace("\\", "/")
        if key in seen:
            clashes.append(final)
        else:
            seen.add(key)
    return clashes


def _plan(conn: sqlite3.Connection, old: str, new: str) -> RelocatePlan:
    plan = RelocatePlan(old_prefix=old, new_prefix=new, new_prefix_exists=Path(new).exists())
    for table, column in text_columns(conn):
        rows = 0
        for _rid, before, after in _hits(conn, table, column, old, new):
            rows += 1
            if len(plan.examples) < _EXAMPLES:
                plan.examples.append((before, after))
        if rows:
            plan.columns.append(ColumnHits(table, column, rows))
            plan.rows += rows
    return plan


def _refuse_unless_applicable(conn: sqlite3.Connection, plan: RelocatePlan) -> None:
    if not plan.new_prefix_exists:
        raise RelocateError(
            f"{plan.new_prefix} does not exist — nothing was written. The new location "
            "has to be there before the index is pointed at it.",
            "relocate_target_missing", prefix=plan.new_prefix)
    if not plan.rows:
        raise RelocateError(
            f"no value in the index starts with {plan.old_prefix} — nothing was written. "
            "Check the old prefix against a path the index actually holds; the match is "
            "case-sensitive.",
            "relocate_no_rows", prefix=plan.old_prefix)
    clashes = _conflicts(conn, plan.old_prefix, plan.new_prefix)
    if clashes:
        raise RelocateError(
            f"{len(clashes)} paths would collide with rows that are already there "
            f"(for example {clashes[0]}) — nothing was written.",
            "relocate_collisions", count=len(clashes), sample=clashes[0])


def relocate(db_path: str | Path, old_prefix: str | Path, new_prefix: str | Path, *,
             apply: bool = False) -> RelocatePlan:
    """Swap one path prefix for another across the whole index, keeping every row id.

    A dry run unless `apply` — the plan says how many values would change, in which
    columns, and shows three of them. With `apply` the whole change is one transaction:
    every column or none.

    What it will not do: work out the new location by itself (both prefixes are named by
    hand), touch a single file on disk, or match a prefix whose case differs from what
    the database holds.
    """
    old, new = normalize_prefix(old_prefix), normalize_prefix(new_prefix)
    if old == new:
        raise RelocateError(f"the old and the new prefix are the same path ({old}).",
                            "relocate_same_prefix", prefix=old)
    db = Path(db_path).expanduser()
    if not db.exists():
        raise RelocateError(f"no index at {db}.", "relocate_no_index", path=str(db))
    conn = connect(db)
    try:
        plan = _plan(conn, old, new)
        if not apply:
            return plan
        _refuse_unless_applicable(conn, plan)
        with conn:  # all the columns or none — a half-moved index points nowhere twice
            for hit in plan.columns:
                _update_column(conn, hit.table, hit.column, old, new)
        plan.applied = True
        _log.info("relocate: %d values moved from %s to %s", plan.rows, old, new)
        return plan
    finally:
        conn.close()


def format_plan(plan: RelocatePlan) -> str:
    """The plan as the terminal prints it: the count, the columns, three examples."""
    verb = "moved" if plan.applied else "would move"
    lines = [f"relocate: {plan.old_prefix} -> {plan.new_prefix}",
             f"{plan.rows} values in {len(plan.columns)} columns {verb}"]
    lines += [f"  {hit.table}.{hit.column}: {hit.rows}" for hit in plan.columns]
    lines += [f"  {before} -> {after}" for before, after in plan.examples]
    if not plan.applied:
        if not plan.new_prefix_exists:
            lines.append(f"{plan.new_prefix} does not exist — --apply would refuse")
        lines.append("nothing written — run it again with --apply")
    return "\n".join(lines)
