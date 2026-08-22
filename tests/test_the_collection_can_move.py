"""F242: the index survives the collection moving to another path.

The thing being protected is not the hour of re-indexing, which comes back on its own.
It is what does not: the face names, the manual places, the animal marks and the
duplicate decisions that hang off `files.id` and are the only data in the product a
person typed. So most of what is asserted below is that the ids did not move.
"""
from __future__ import annotations

import sqlite3
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from sorta import relocate as R
from sorta.config import Config, IndexConfig
from sorta.db import connect
from sorta.indexer import index
from sorta.relocate import CollectionMoved


def fill(conn: sqlite3.Connection, paths: list[str]) -> None:
    """Files, a named cluster, a face on each file and a manual mark — the whole point
    of the feature in four tables."""
    conn.execute("INSERT INTO face_clusters (id, label) VALUES (1, 'Anna')")
    for i, path in enumerate(paths, start=1):
        conn.execute(
            "INSERT INTO files (id, path, size, mtime, ext, media_type, indexed_at) "
            "VALUES (?, ?, 100, 1.0, 'jpg', 'photo', '2026-08-22')", (i, path))
        conn.execute("INSERT INTO faces (file_id, bbox, embedding, cluster_id) "
                     "VALUES (?, '[1,2,3,4]', X'00', 1)", (i,))
        conn.execute("INSERT INTO manual_pet (file_id, is_animal, updated_at) "
                     "VALUES (?, 1, '2026-08-22')", (i,))
    conn.commit()


def paths_of(db: Path) -> list[str]:
    conn = connect(db)
    try:
        return [row["path"] for row in conn.execute("SELECT path FROM files ORDER BY id")]
    finally:
        conn.close()


class MovedCollection(unittest.TestCase):
    """One database, four files under `old`, one under a sibling that must not move."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.old = self.root / "photos"
        self.new = self.root / "pictures"
        self.new.mkdir()
        self.db = self.root / "photos.db"
        conn = connect(self.db)
        self.stored = [
            str(self.old / "2019" / "a.jpg"),
            str(self.old / "2019" / "b.jpg"),
            str(self.old / "c.jpg"),
            str(self.root / "photos-backup" / "d.jpg"),
        ]
        fill(conn, self.stored)
        conn.close()

    def tearDown(self):
        self.tmp.cleanup()

    def move(self, **kwargs) -> R.RelocatePlan:
        return R.relocate(self.db, self.old, self.new, **kwargs)


class TestDryRunWritesNothing(MovedCollection):
    def test_the_default_run_leaves_every_path_where_it_was(self):
        plan = self.move()
        self.assertEqual(plan.rows, 3)
        self.assertFalse(plan.applied)
        self.assertEqual(paths_of(self.db), self.stored)

    def test_the_output_names_the_number_of_rows_and_three_examples(self):
        text = R.format_plan(self.move())
        self.assertIn("3 values", text)
        self.assertIn("files.path: 3", text)
        self.assertEqual(3, sum(1 for line in text.splitlines() if " -> " in line
                                and line.startswith("  ")))
        self.assertIn("--apply", text)


class TestTheMoveKeepsTheIds(MovedCollection):
    def test_paths_point_at_the_new_prefix(self):
        plan = self.move(apply=True)
        self.assertTrue(plan.applied)
        self.assertEqual(paths_of(self.db)[:3], [
            str(self.new / "2019" / "a.jpg"),
            str(self.new / "2019" / "b.jpg"),
            str(self.new / "c.jpg"),
        ])

    def test_the_manual_work_is_still_attached_to_the_same_files(self):
        before = self.faces_by_path()
        self.move(apply=True)
        after = self.faces_by_path()
        self.assertEqual(sorted(before), [1, 2, 3, 4])
        self.assertEqual(sorted(before), sorted(after))
        conn = connect(self.db)
        try:
            named = conn.execute(
                "SELECT f.path FROM files f JOIN faces fa ON fa.file_id = f.id "
                "JOIN face_clusters c ON c.id = fa.cluster_id "
                "JOIN manual_pet m ON m.file_id = f.id WHERE c.label = 'Anna'").fetchall()
        finally:
            conn.close()
        self.assertEqual(len(named), 4)

    def faces_by_path(self) -> list[int]:
        conn = connect(self.db)
        try:
            return [row["file_id"] for row in conn.execute("SELECT file_id FROM faces")]
        finally:
            conn.close()


class TestTheBoundaryIsAPathComponent(MovedCollection):
    def test_photos_backup_is_a_different_folder(self):
        self.move(apply=True)
        self.assertEqual(paths_of(self.db)[3], self.stored[3])

    def test_a_prefix_written_with_a_trailing_slash_matches_all_the_same(self):
        plan = R.relocate(self.db, f"{self.old.as_posix()}/", self.new, apply=True)
        self.assertEqual(plan.rows, 3)


class TestSeparatorStyle(MovedCollection):
    def test_a_row_stored_in_posix_keeps_posix(self):
        conn = connect(self.db)
        conn.execute("UPDATE files SET path = ? WHERE id = 1",
                     ((self.old / "2019" / "a.jpg").as_posix(),))
        conn.commit()
        conn.close()
        self.move(apply=True)
        self.assertEqual(paths_of(self.db)[0], (self.new / "2019" / "a.jpg").as_posix())


class TestNothingIsWrittenOnAFailure(MovedCollection):
    def test_an_exception_halfway_through_leaves_the_database_as_it_was(self):
        conn = connect(self.db)
        conn.execute("INSERT INTO move_batches (id, mode, dest_root, started_at) "
                     "VALUES (1, 'city', ?, '2026-08-22')", (str(self.old / "sorted"),))
        conn.commit()
        conn.close()
        calls = []

        def explode(*args, **kwargs):
            calls.append(args)
            if len(calls) > 1:
                raise sqlite3.OperationalError("disk went away")
            return R._update_column(*args, **kwargs)

        with unittest.mock.patch.object(R, "_update_column", explode):
            with self.assertRaises(sqlite3.OperationalError):
                self.move(apply=True)
        self.assertGreater(len(calls), 1)
        self.assertEqual(paths_of(self.db), self.stored)


class TestRefusals(MovedCollection):
    def test_a_new_prefix_that_is_not_on_disk_is_refused(self):
        with self.assertRaises(R.RelocateError) as caught:
            R.relocate(self.db, self.old, self.root / "nowhere", apply=True)
        self.assertIn("does not exist", str(caught.exception))
        self.assertEqual(paths_of(self.db), self.stored)

    def test_an_old_prefix_that_matches_nothing_is_refused(self):
        with self.assertRaises(R.RelocateError) as caught:
            R.relocate(self.db, self.root / "elsewhere", self.new, apply=True)
        self.assertIn("no value in the index", str(caught.exception))

    def test_a_move_that_would_put_two_rows_on_one_path_is_refused(self):
        conn = connect(self.db)
        conn.execute("INSERT INTO files (id, path, size, mtime, ext, media_type, indexed_at)"
                     " VALUES (9, ?, 100, 1.0, 'jpg', 'photo', '2026-08-22')",
                     (str(self.new / "c.jpg"),))
        conn.commit()
        conn.close()
        with self.assertRaises(R.RelocateError) as caught:
            self.move(apply=True)
        self.assertIn("collide", str(caught.exception))
        self.assertEqual(paths_of(self.db)[:4], self.stored)

    def test_the_same_prefix_twice_is_refused(self):
        with self.assertRaises(R.RelocateError):
            R.relocate(self.db, self.old, self.old, apply=True)

    def test_a_database_that_is_not_there_is_refused(self):
        with self.assertRaises(R.RelocateError):
            R.relocate(self.root / "absent.db", self.old, self.new)

    def test_a_dry_run_over_nothing_reports_instead_of_raising(self):
        plan = R.relocate(self.db, self.root / "elsewhere", self.root / "nowhere")
        self.assertEqual(plan.rows, 0)
        self.assertIn("does not exist", R.format_plan(plan))


def every_text_value(conn: sqlite3.Connection) -> list[tuple[str, str, str]]:
    """(table, column, value) for every TEXT value in the database — the sweep the
    guard below is made of, and deliberately not the same code path as the move."""
    found = []
    for table, column in R.text_columns(conn):
        for row in conn.execute(f'SELECT "{column}" AS v FROM "{table}"'):
            if isinstance(row["v"], str):
                found.append((table, column, row["v"]))
    return found


class TestEveryColumnThatHoldsAPathIsCovered(unittest.TestCase):
    """The guard that FINDS instead of listing.

    Every TEXT column of the real schema is filled with a path under the old prefix and
    the whole database is swept afterwards. A feature that adds a table, or a column, is
    covered by this test on the day it is written and without anyone editing it — which
    is the only kind of guard worth having here, because the failure it prevents is
    silent.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.old, self.new = self.root / "photos", self.root / "pictures"
        self.new.mkdir()
        self.db = self.root / "photos.db"

    def tearDown(self):
        self.tmp.cleanup()

    def fill_every_text_column(self, conn: sqlite3.Connection) -> int:
        # Foreign keys off: the fixture is one row per table with every id set to 1, and
        # the order sqlite_master hands the tables back in is not an insertion order.
        conn.execute("PRAGMA foreign_keys = OFF")
        filled = 0
        for table, columns in self.schema(conn).items():
            names = ", ".join(f'"{name}"' for name, _ in columns)
            values = []
            for name, declared in columns:
                if R._is_text(declared):
                    values.append(f"{self.old.as_posix()}/{table}_{name}.jpg")
                    filled += 1
                elif "INT" in declared.upper() or "REAL" in declared.upper():
                    values.append(1)
                else:
                    values.append(b"\x00")
            conn.execute(f'INSERT INTO "{table}" ({names}) VALUES '
                         f'({", ".join("?" * len(values))})', values)
        conn.commit()
        return filled

    def schema(self, conn: sqlite3.Connection) -> dict[str, list[tuple[str, str]]]:
        return {table: [(row["name"], row["type"])
                        for row in conn.execute(f'PRAGMA table_info("{table}")')]
                for table in R._tables(conn)}

    def left_behind(self) -> list[tuple[str, str, str]]:
        conn = connect(self.db)
        try:
            old = R.normalize_prefix(self.old)
            return [hit for hit in every_text_value(conn)
                    if hit[2].replace("\\", "/").startswith(old + "/")]
        finally:
            conn.close()

    def test_no_text_value_anywhere_is_left_under_the_old_prefix(self):
        conn = connect(self.db)
        filled = self.fill_every_text_column(conn)
        conn.close()
        self.assertGreater(filled, 40)
        plan = R.relocate(self.db, self.old, self.new, apply=True)
        self.assertEqual(plan.rows, filled)
        self.assertEqual(self.left_behind(), [])

    def test_a_column_added_after_this_test_was_written_is_moved_too(self):
        conn = connect(self.db)
        self.fill_every_text_column(conn)
        conn.execute("ALTER TABLE files ADD COLUMN thumb_path TEXT")
        conn.execute("CREATE TABLE sidecars (file_id INTEGER, sidecar TEXT)")
        conn.execute("UPDATE files SET thumb_path = ?", (str(self.old / "thumb.jpg"),))
        conn.execute("INSERT INTO sidecars VALUES (1, ?)", (str(self.old / "a.xmp"),))
        conn.commit()
        conn.close()
        R.relocate(self.db, self.old, self.new, apply=True)
        self.assertEqual(self.left_behind(), [])

    def test_the_sweep_goes_red_when_one_column_is_missed(self):
        conn = connect(self.db)
        self.fill_every_text_column(conn)
        conn.close()
        keep = R.text_columns

        def without_manual_places(c):
            return [pair for pair in keep(c) if pair != ("manual_places", "updated_at")]

        with unittest.mock.patch.object(R, "text_columns", without_manual_places):
            R.relocate(self.db, self.old, self.new, apply=True)
        self.assertEqual([(t, c) for t, c, _ in self.left_behind()],
                         [("manual_places", "updated_at")])


class TestTheIndexerStopsInsteadOfStartingOver(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.gone = self.root / "gone"
        self.db = self.root / "photos.db"
        self.cfg = Config(sources=[self.gone], database=self.db,
                          index=IndexConfig(min_file_size_kb=0, compute_phash=False))
        self.conn = connect(self.db)

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def test_a_full_index_and_no_root_on_disk_stops_the_run(self):
        fill(self.conn, [str(self.gone / "a.jpg")])
        with self.assertRaises(CollectionMoved) as caught:
            index(self.cfg, self.conn)
        message = str(caught.exception)
        self.assertIn("The usual cause is a move", message)
        self.assertIn("sorta relocate --from", message)
        self.assertIn(str(self.gone), message)
        self.assertIn(str(self.gone / "a.jpg"), message)

    def test_an_empty_index_says_nothing_and_runs(self):
        stats = index(self.cfg, self.conn)
        self.assertEqual(stats.scanned, 0)

    def test_one_root_that_still_exists_is_not_a_move(self):
        here = self.root / "here"
        here.mkdir()
        self.cfg.sources = [self.gone, here]
        fill(self.conn, [str(self.gone / "a.jpg")])
        self.assertEqual(index(self.cfg, self.conn).scanned, 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
