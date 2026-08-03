"""SQLite connection and schema application."""
from __future__ import annotations

import re
import sqlite3
from importlib.resources import files
from pathlib import Path

SCHEMA = files("sorta.db").joinpath("schema.sql").read_text(encoding="utf-8")

_USER_VERSION_PRAGMA = re.compile(r"^\s*PRAGMA\s+user_version\s*=\s*(\d+)\s*;", re.MULTILINE)


def _declared_version(schema: str) -> int:
    """The schema version, read off the schema itself.

    The number is written ONCE, in `schema.sql`, where `executescript` applies it to the
    database; everything that has to know it reads `SCHEMA_VERSION` instead of repeating
    the literal. It used to be repeated in every test that checks a migration, and the
    version is the one thing parallel features cannot settle between themselves: F124,
    F132 and F131 each raised it, and each had to chase the same number through five or
    six test files — the last of them with the RIGHT number and still red, because the
    literals that broke belonged to its NEIGHBOURS' tests, which compare a fresh database
    against a hard-coded figure.

    A missing or duplicated PRAGMA is an error rather than a default: a second one would
    be a second source of truth, and silently guessing 0 would make every migration run
    against a database that claims to predate the schema.
    """
    found = _USER_VERSION_PRAGMA.findall(schema)
    if len(found) != 1:
        raise RuntimeError(
            "schema.sql must declare `PRAGMA user_version = N` exactly once, "
            f"found {len(found)}"
        )
    return int(found[0])


SCHEMA_VERSION = _declared_version(SCHEMA)


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
    # media_class itself only appeared in v3 — a v1/v2 DB has no such table yet
    # (it gets created by executescript below), so the range starts at 3.
    if 3 <= version <= 10:  # v11: media_class.tier (F68 — incrementality marker)
        conn.execute("ALTER TABLE media_class ADD COLUMN tier TEXT")
        # Backfill so the upgrade does not reclassify the whole collection: 'ocr' is
        # a verdict of the fast (clip) tier, not a tier of its own.
        conn.execute(
            "UPDATE media_class SET tier = CASE source"
            " WHEN 'ocr' THEN 'clip' WHEN 'vlm' THEN 'vlm'"
            " WHEN 'heuristic' THEN 'heuristic' ELSE 'clip' END"
        )
    # v12 (manual_overrides) — a new table, created by executescript below
    # v13 (geo_cache, F93) — a new table, created by executescript below
    # v14 (manual_places, F85c) — a new table, created by executescript below
    # v15 (frame_quality, F113) — a new table, created by executescript below
    # v16 (clip_embeddings, F128) — a new table, created by executescript below
    # frame_quality itself only appeared in v15, so the range starts there: an older DB
    # gets the column with the table from executescript below.
    if 15 <= version <= 16:  # v17: frame_quality.pet_vlm (F130 — the pet check's answer)
        conn.execute("ALTER TABLE frame_quality ADD COLUMN pet_vlm TEXT")
    # v18 (manual_pet, F124) — a new table, created by executescript below.
    # F124 was written against a main standing at v16 and took v17, exactly as its brief
    # said to; F130 merged first and took that number. Renumbered here by the
    # orchestrator — the version is the one thing two parallel workers cannot settle
    # between themselves, because neither can see the other's worktree, and each branch
    # is self-consistent right up until the second merge.
    # v19 (group_keeper, F132) — a new table, created by executescript below.
    # The third renumbering of the day, for the same reason. F124/F130/F132 all took the
    # next free number their own main showed them; only the merge order can decide.
    if 15 <= version <= 20:  # v21: frame_quality.junk_score (F140 — the rescue score)
        conn.execute("ALTER TABLE frame_quality ADD COLUMN junk_score REAL")
    # v20 (landmark_checks, F131) — a new table, created by executescript below.
    # v22 (search_embeddings, F141) — a new table, created by executescript below.
    # v23 (restored_files, F149) — a new table, created by executescript below.


def connect(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    _migrate(conn)
    conn.executescript(SCHEMA)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# F93: the ONE table "start over" spares. A reset is about the user's files; the name
# of a point on the map does not depend on which files are lying around, and re-asking
# Nominatim for it costs ~10 minutes of network per collection. An explicit exception
# list, not "every table in sqlite_master": a new table must be wiped by default, and
# surviving a reset has to be a deliberate decision per table.
_KEPT_ON_RESET = ("geo_cache",)


def reset_index(conn: sqlite3.Connection, *, clear_geo: bool = False) -> None:
    """Wipe the index (all tables but the geo cache) and recreate the empty schema.

    Deletes ONLY DB data: metadata, geo, faces/clusters (and people names!),
    events (and manual names!), junk classification, dup decisions, the move
    journal. FILES on disk and already-sorted folders are NOT touched (they are not
    in the DB). Used by the `sorta reset` command and the "Start over" button in
    `sorta ui`.

    `clear_geo=True` additionally drops the cached provider answers (`geo_cache`,
    F93). It must stay reachable: the cache can hold a WRONG answer, and a user who
    sees a wrong city presses "Start over" expecting to redo everything from nothing —
    without the flag they would get the very same wrong city back.
    """
    kept = () if clear_geo else _KEPT_ON_RESET
    with conn:
        conn.execute("PRAGMA foreign_keys = OFF")
        tables = [r["name"] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'") if r["name"] not in kept]
        for name in tables:
            conn.execute(f'DROP TABLE IF EXISTS "{name}"')
    conn.executescript(SCHEMA)  # recreates empty tables + user_version
    conn.execute("PRAGMA foreign_keys = ON")
