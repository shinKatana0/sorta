"""A database shaped as an OLDER schema version — the knowledge, in one place.

Every migration test needs the same thing: a database as some earlier version left it,
so that the migration under test has something to migrate. Each of them used to build
its own — create a fresh database, drop the table this feature adds, write
`PRAGMA user_version = N` — and that recipe is wrong in a way that stays hidden until
another feature touches the schema. A FRESH database carries every LATER column too, so
the fixture is a shape no released version ever had; the next feature's
`ALTER TABLE ... ADD COLUMN` then runs against a column that already exists and fails.

That is not hypothetical (F124): the "v16" fixture kept `frame_quality.pet_vlm`, added by
a feature merged in parallel, and the migration died on `duplicate column name` — under
the mask of a `PermissionError`, because the still-open connection kept the temp
directory from being removed, so the real error never reached the report at all.

So a simulated old database has to be old in EVERY respect, not only in the one the
calling test cares about. What each version added is recorded below once, and rolling
back undoes all of it. What makes the database old stays visible at the call site — the
test says which feature it predates, which is the content of such a test.

Adding a schema version? Add what it introduced to `SCHEMA_HISTORY`. Nothing else in the
suite has to learn the number: it lives in `schema.sql` and reaches the tests as
`sorta.db.SCHEMA_VERSION`.
"""
from __future__ import annotations

import sqlite3

# version -> what that version INTRODUCED. "table" for a whole table, "table.column" for
# a column added to one that already existed. Only additions are listed: nothing has ever
# been dropped, and a version that dropped something would have to say how to put it back.
# A version that changed DATA rather than shape (v26) introduces nothing and is listed as
# an empty tuple — rolling back to it is rolling back to the shape of the version before.
SCHEMA_HISTORY: dict[int, tuple[str, ...]] = {
    2: ("files.orientation",),
    3: ("media_class",),
    4: ("events.origin",),
    5: ("files.not_personal",),
    6: ("places.city_geonameid", "places.district_geonameid"),
    7: ("dedup_choice",),
    8: ("move_batches.operation",),
    9: ("places.district_name",),
    10: ("places.country_name",),
    11: ("media_class.tier",),
    12: ("manual_overrides",),
    13: ("geo_cache",),
    14: ("manual_places",),
    15: ("frame_quality",),
    16: ("clip_embeddings",),
    17: ("frame_quality.pet_vlm",),
    18: ("manual_pet",),
    19: ("group_keeper",),
    20: ("landmark_checks",),
    21: ("frame_quality.junk_score",),
    22: ("search_embeddings",),
    23: ("detections",),
    24: ("restored_files",),
    25: ("frame_quality.face_sharpness",),
    26: (),  # F177: no new shape — it empties frame_quality.has_subject.
}


def version_that_added(item: str) -> int:
    """The schema version `item` ("table" or "table.column") first appeared in."""
    for version, added in sorted(SCHEMA_HISTORY.items()):
        if item in added:
            return version
    raise KeyError(f"{item!r} is not recorded in SCHEMA_HISTORY")


def roll_back_to(conn: sqlite3.Connection, version: int) -> None:
    """Undo everything the schema gained after `version`, and say so in `user_version`.

    Descending order matters: a column added to a table that itself arrived later is
    dropped with the table, and asking for the column afterwards would raise.
    """
    for later in sorted(SCHEMA_HISTORY, reverse=True):
        if later <= version:
            break
        for item in SCHEMA_HISTORY[later]:
            table, _, column = item.partition(".")
            if column:
                conn.execute(f"ALTER TABLE {table} DROP COLUMN {column}")
            else:
                conn.execute(f"DROP TABLE {table}")
    conn.execute(f"PRAGMA user_version = {int(version)}")  # PRAGMA takes no parameters


def roll_back_before(conn: sqlite3.Connection, item: str) -> int:
    """Shape `conn` as the last version that did NOT have `item` yet; return that version.

    The call reads as what the test means — "a database from before `manual_pet`
    existed" — instead of a number that has to be looked up, and that stops being true
    the moment the schema moves on.
    """
    version = version_that_added(item) - 1
    roll_back_to(conn, version)
    return version


def items_after(version: int) -> list[str]:
    """Everything the schema gained after `version` — what a rollback must have removed."""
    return [item
            for later in sorted(SCHEMA_HISTORY)
            if later > version
            for item in SCHEMA_HISTORY[later]]
