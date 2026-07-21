"""SQLite connection and schema application."""
from __future__ import annotations

import sqlite3
from importlib.resources import files
from pathlib import Path

SCHEMA = files("sorta.db").joinpath("schema.sql").read_text(encoding="utf-8")


def _migrate(conn: sqlite3.Connection) -> None:
    """Migrate existing DBs before executescript (which sets the new user_version).

    A fresh DB (user_version = 0) gets the full current schema — migrations are
    only needed for tables already created by previous versions.
    """
    (version,) = conn.execute("PRAGMA user_version").fetchone()
    if version == 1:  # v2: files.orientation
        conn.execute("ALTER TABLE files ADD COLUMN orientation INTEGER")
    # v3 (media_class) — a new table, created by executescript below
    if 1 <= version <= 3:  # v4: events.origin
        conn.execute("ALTER TABLE events ADD COLUMN origin TEXT NOT NULL DEFAULT 'auto'")
    if 1 <= version <= 4:  # v5: files.not_personal
        conn.execute("ALTER TABLE files ADD COLUMN not_personal INTEGER NOT NULL DEFAULT 0")
    if 1 <= version <= 5:  # v6: places.city_geonameid/district_geonameid (G2)
        conn.execute("ALTER TABLE places ADD COLUMN city_geonameid INTEGER")
        conn.execute("ALTER TABLE places ADD COLUMN district_geonameid INTEGER")
    # v7 (dedup_choice) — a new table, created by executescript below
    if 1 <= version <= 7:  # v8: move_batches.operation (C16 copy mode)
        conn.execute("ALTER TABLE move_batches ADD COLUMN operation TEXT NOT NULL DEFAULT 'move'")
    if 1 <= version <= 8:  # v9: places.district_name (G2b online provider)
        conn.execute("ALTER TABLE places ADD COLUMN district_name TEXT")
    if 1 <= version <= 9:  # v10: places.country_name (G6 online — full country name)
        conn.execute("ALTER TABLE places ADD COLUMN country_name TEXT")


def connect(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    _migrate(conn)
    conn.executescript(SCHEMA)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def reset_index(conn: sqlite3.Connection) -> None:
    """Wipe the ENTIRE index (all tables) and recreate the empty schema — "start over".

    Deletes ONLY DB data: metadata, geo, faces/clusters (and people names!),
    events (and manual names!), junk classification, dup decisions, the move
    journal. FILES on disk and already-sorted folders are NOT touched (they are not
    in the DB). Used by the `sorta reset` command and the "Start over" button in
    `sorta ui`.
    """
    with conn:
        conn.execute("PRAGMA foreign_keys = OFF")
        tables = [r["name"] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")]
        for name in tables:
            conn.execute(f'DROP TABLE IF EXISTS "{name}"')
    conn.executescript(SCHEMA)  # recreates empty tables + user_version
    conn.execute("PRAGMA foreign_keys = ON")
