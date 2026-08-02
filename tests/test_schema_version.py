"""F143: the schema version is written in ONE place, and the suite reads it from there.

The number moved v15 -> v21 in two days, and every step cost the same repair: the feature
raised `schema.sql`, and five or six test files still compared a fresh database against
the old figure. F131 is the case that settles it — it took the RIGHT number, checked
against `db/__init__.py` exactly as its brief said, and still turned the gate red,
because the literals it broke were in its NEIGHBOURS' tests. A warning in a brief cannot
fix that; only having one number can.

So this module states the two facts the arrangement rests on — the constant agrees with
the database sqlite actually creates, and it agrees with the text of `schema.sql` — plus
the guard that keeps the literals from coming back, and the tests of the shared old-database
fixture that replaced every hand-built "a database as of version N".
"""
from __future__ import annotations

import re
import sqlite3
import tempfile
import unittest
from pathlib import Path

import sorta.db
from sorta.db import SCHEMA, SCHEMA_VERSION, connect

from tests.schema_history import (
    SCHEMA_HISTORY,
    items_after,
    roll_back_before,
    roll_back_to,
    version_that_added,
)

_ROOT = Path(__file__).resolve().parent.parent
_TESTS = _ROOT / "tests"
_SCHEMA_SQL = Path(sorta.db.__file__).resolve().parent / "schema.sql"


def pragma_versions(schema: str) -> list[int]:
    """Every `PRAGMA user_version = N` in the text, found WITHOUT the module's own regex.

    Deliberately a second, dumber reader: reusing `db._USER_VERSION_PRAGMA` would make
    the test agree with the parser instead of with the file.
    """
    found = []
    for line in schema.splitlines():
        statement = line.strip().rstrip(";").replace(" ", "").lower()
        if statement.startswith("pragmauser_version="):
            found.append(int(statement.split("=")[1]))
    return found


def table_names(conn: sqlite3.Connection) -> set[str]:
    return {r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}


def has_item(conn: sqlite3.Connection, item: str) -> bool:
    """Whether the database holds `item` — a table, or a column of one ("table.column")."""
    table, _, column = item.partition(".")
    if table not in table_names(conn):
        return False
    if not column:
        return True
    return column in {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}


class TestTheVersionHasOneSource(unittest.TestCase):
    def test_the_constant_is_what_a_fresh_database_reports(self):
        """Brief test 1: sqlite executes the PRAGMA, the constant only reads it — if the
        reading were wrong, this is where it shows, on the database itself."""
        with tempfile.TemporaryDirectory() as tmp:
            conn = connect(Path(tmp) / "fresh.db")
            (version,) = conn.execute("PRAGMA user_version").fetchone()
            conn.close()
        self.assertEqual(version, SCHEMA_VERSION)

    def test_the_constant_is_the_number_written_in_schema_sql(self):
        """Brief test 2, the main one: without it a second source of truth appears and
        the whole cost comes back. Exactly one PRAGMA, and it is the constant."""
        self.assertEqual(pragma_versions(SCHEMA), [SCHEMA_VERSION])

    def test_the_schema_the_code_loads_is_the_file_on_disk(self):
        """`SCHEMA` comes through importlib.resources; the test above reads what it got.
        This one ties that text back to the file a person edits."""
        self.assertEqual(SCHEMA, _SCHEMA_SQL.read_text(encoding="utf-8"))

    def test_a_schema_without_the_pragma_is_an_error_not_a_default(self):
        """Guessing 0 would send every migration through a database claiming to predate
        the schema; a second PRAGMA would be the second source of truth again."""
        with self.assertRaises(RuntimeError):
            sorta.db._declared_version("CREATE TABLE files (id INTEGER PRIMARY KEY);")
        with self.assertRaises(RuntimeError):
            sorta.db._declared_version("PRAGMA user_version = 3;\nPRAGMA user_version = 4;")


class TestNoTestRepeatsTheVersion(unittest.TestCase):
    """Brief test 3: the guard against the literals coming back.

    Two shapes were found in the suite, and they fail differently. A line that compares
    something to the CURRENT number breaks on the very next feature that raises the
    schema — that is the four-times-repeated repair, and it has no exceptions. A
    hand-written `PRAGMA user_version = N` is the other half: it is how a fixture claims
    to be old while carrying columns from the future, and it belongs in
    `tests/schema_history.py` instead.
    """

    # The pointwise exceptions, one reason each.
    #   test_db.py         — writes a minimal v1 out of raw SQL, never from the current
    #                        schema, so it cannot inherit a column from the future and its
    #                        number cannot go stale. It is also the independent check on
    #                        the shared fixture: it proves a real old database migrates
    #                        without using the history the fixture is built from.
    #   test_junk_tier.py  — assembles a pre-v11 `media_class` the same way. It DOES start
    #                        from `connect()`, and would be worth moving onto the shared
    #                        fixture; it is outside this feature's ownership, and it
    #                        survives today only because no migration below v11 has been
    #                        added since.
    #   this file          — the two shapes are its subject matter; the examples below
    #                        have to be written out to be checked at all.
    _WRITE_A_VERSION_BY_HAND = {"test_db.py", "test_junk_tier.py", "test_schema_version.py"}

    _NUMBER = re.compile(r"\b\d+\b")
    _HAND_WRITTEN = re.compile(r"PRAGMA\s+user_version\s*=\s*\d")

    def sources(self) -> list[Path]:
        return sorted(_TESTS.rglob("*.py"))

    def lines_comparing_to_the_current_version(self, source: str) -> list[int]:
        """Lines that talk about a version AND carry the current one as a bare number."""
        return [number
                for number, line in enumerate(source.splitlines(), start=1)
                if "version" in line.lower()
                and any(int(n) == SCHEMA_VERSION for n in self._NUMBER.findall(line))]

    def test_no_test_compares_against_the_current_version(self):
        for path in self.sources():
            source = path.read_text(encoding="utf-8")
            with self.subTest(file=path.name):
                self.assertEqual(self.lines_comparing_to_the_current_version(source), [])

    def test_no_test_writes_a_version_by_hand(self):
        for path in self.sources():
            if path.name in self._WRITE_A_VERSION_BY_HAND:
                continue
            source = path.read_text(encoding="utf-8")
            with self.subTest(file=path.name):
                self.assertIsNone(self._HAND_WRITTEN.search(source))

    def test_the_guard_would_notice_either_shape(self):
        """A rule that never fires is not a guard. Both shapes, and the forms that are
        fine: an OLD number in an assertion stays true across a bump, and the fixture
        writes the pragma from a variable."""
        self.assertEqual(
            self.lines_comparing_to_the_current_version(
                f"self.assertEqual(version, {SCHEMA_VERSION})"), [1])
        self.assertEqual(
            self.lines_comparing_to_the_current_version(
                "self.assertGreaterEqual(version, 11)"), [])
        self.assertIsNotNone(self._HAND_WRITTEN.search("conn.execute('PRAGMA user_version = 16')"))
        self.assertIsNone(
            self._HAND_WRITTEN.search('conn.execute(f"PRAGMA user_version = {version}")'))

    def test_every_test_file_is_actually_scanned(self):
        """A wrong directory would make the whole guard silently vacuous."""
        scanned = self.sources()
        self.assertGreater(len(scanned), 50)
        self.assertIn("test_db.py", {p.name for p in scanned})


class TestSchemaHistory(unittest.TestCase):
    """The record of what each version added — the thing a rollback is built out of."""

    def test_every_recorded_item_exists_in_a_fresh_database(self):
        """A typo would leave the rollback silently skipping what it meant to remove."""
        with tempfile.TemporaryDirectory() as tmp:
            conn = connect(Path(tmp) / "fresh.db")
            missing = [item for added in SCHEMA_HISTORY.values() for item in added
                       if not has_item(conn, item)]
            conn.close()
        self.assertEqual(missing, [])

    def test_no_item_is_recorded_twice(self):
        items = [item for added in SCHEMA_HISTORY.values() for item in added]
        self.assertEqual(sorted(items), sorted(set(items)))

    def test_the_history_claims_no_version_the_schema_does_not_have(self):
        self.assertLessEqual(max(SCHEMA_HISTORY), SCHEMA_VERSION)

    def test_version_that_added_finds_tables_and_columns(self):
        self.assertEqual(version_that_added("frame_quality") + 2,
                         version_that_added("frame_quality.pet_vlm"))
        with self.assertRaises(KeyError):
            version_that_added("no_such_table")


class TestOldDatabaseFixture(unittest.TestCase):
    """Brief test 4: what it builds is old in EVERY respect, and it still migrates."""

    def old_database(self, tmp: str, item: str) -> tuple[Path, int]:
        db = Path(tmp) / "old.db"
        conn = connect(db)
        conn.execute("INSERT INTO files (path, size, mtime, ext, media_type, indexed_at) "
                     "VALUES ('/a.jpg', 1, 0.0, 'jpg', 'photo', 'x')")
        version = roll_back_before(conn, item)
        conn.commit()
        conn.close()
        return db, version

    def test_nothing_from_a_later_version_is_left_behind(self):
        for item in ("frame_quality", "clip_embeddings", "frame_quality.pet_vlm",
                     "manual_pet", "group_keeper"):
            with self.subTest(before=item), tempfile.TemporaryDirectory() as tmp:
                db, version = self.old_database(tmp, item)
                conn = sqlite3.connect(db)  # raw: connect() would migrate it first
                conn.row_factory = sqlite3.Row
                left = [later for later in items_after(version) if has_item(conn, later)]
                (declared,) = conn.execute("PRAGMA user_version").fetchone()
                conn.close()
                self.assertEqual(left, [])
                self.assertEqual(declared, version)

    def test_what_the_version_did_have_is_still_there(self):
        """The other half: a rollback that emptied the database would pass the test above
        and prove nothing."""
        with tempfile.TemporaryDirectory() as tmp:
            db, version = self.old_database(tmp, "manual_pet")
            conn = sqlite3.connect(db)
            conn.row_factory = sqlite3.Row
            missing = [item
                       for at, added in SCHEMA_HISTORY.items() if at <= version
                       for item in added if not has_item(conn, item)]
            conn.close()
            self.assertEqual(missing, [])

    def test_the_migration_runs_on_it_and_keeps_the_rows(self):
        for item in ("frame_quality", "clip_embeddings", "frame_quality.pet_vlm",
                     "manual_pet", "group_keeper"):
            with self.subTest(before=item), tempfile.TemporaryDirectory() as tmp:
                db, _ = self.old_database(tmp, item)
                conn = connect(db)
                (version,) = conn.execute("PRAGMA user_version").fetchone()
                files = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
                current = has_item(conn, item)
                conn.close()
                self.assertEqual(version, SCHEMA_VERSION)
                self.assertEqual(files, 1)
                self.assertTrue(current)

    def test_rolling_back_to_the_very_first_version_leaves_a_v1_database(self):
        """The extreme case, and the one that exercises the whole chain of migrations."""
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "v1.db"
            conn = connect(db)
            roll_back_to(conn, min(SCHEMA_HISTORY) - 1)
            conn.commit()
            conn.close()

            conn = connect(db)
            (version,) = conn.execute("PRAGMA user_version").fetchone()
            missing = [item for added in SCHEMA_HISTORY.values() for item in added
                       if not has_item(conn, item)]
            conn.close()
        self.assertEqual(version, SCHEMA_VERSION)
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
