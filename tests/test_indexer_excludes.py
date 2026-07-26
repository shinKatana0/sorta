"""F81: source folders excluded BEFORE indexing, not cleaned up afterwards.

The point of the feature is what does NOT happen: an excluded subtree is never
stat'ed, never opened, never hashed, and never reaches a later stage. So most of the
tests below assert absence — of disk calls, of rows, of dangling references.
"""
from __future__ import annotations

import builtins
import logging
import os
import pathlib
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from PIL import Image

from sorta.config import Config, IndexConfig
from sorta.db import connect
from sorta.indexer import (
    Excludes,
    _walk,
    drop_excluded_rows,
    excludes_path,
    index,
    load_excludes,
    normalize_exclude,
    save_excludes,
)


def make_jpeg(path: Path, color=(255, 0, 0), size=(64, 64)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path, "JPEG")


class ExcludesTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.src = self.root / "photos"
        self.excludes_file = self.root / "excludes.yaml"
        self.cfg = Config(
            sources=[self.src],
            database=self.root / "test.db",
            index=IndexConfig(min_file_size_kb=0, compute_phash=False),
            raw={"index": {"excludes_file": str(self.excludes_file)}},
        )
        self.conn = connect(self.cfg.database)

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def write_excludes(self, text: str) -> None:
        self.excludes_file.write_text(text, encoding="utf-8")

    def indexed_names(self) -> set[str]:
        return {Path(r["path"]).name
                for r in self.conn.execute("SELECT path FROM files")}


class TestExcludedSubtreeIsNotIndexed(ExcludesTestBase):
    def test_excluded_folder_and_its_subtree_are_skipped(self):
        make_jpeg(self.src / "keep.jpg")
        make_jpeg(self.src / "Screenshots" / "shot.jpg")
        make_jpeg(self.src / "Movies" / "poster.jpg")
        make_jpeg(self.src / "Movies" / "deep" / "deeper" / "frame.jpg")
        self.write_excludes(f'"{self.src.as_posix()}":\n  - Movies\n')

        stats = index(self.cfg, self.conn)

        self.assertEqual(self.indexed_names(), {"keep.jpg", "shot.jpg"})
        self.assertEqual(stats.added, 2)
        self.assertEqual(stats.scanned, 2)  # the excluded files never entered the walk

    def test_nested_exclusion_keeps_its_siblings(self):
        make_jpeg(self.src / "trip" / "a.jpg")
        make_jpeg(self.src / "trip" / "temp" / "b.jpg")
        self.write_excludes(f'"{self.src.as_posix()}":\n  - trip/temp\n')

        index(self.cfg, self.conn)

        self.assertEqual(self.indexed_names(), {"a.jpg"})


class TestExcludedSubtreeIsNotTouched(ExcludesTestBase):
    """The feature is worth nothing if the disk is read anyway."""

    def test_no_stat_and_no_open_inside_the_excluded_subtree(self):
        make_jpeg(self.src / "keep.jpg")
        make_jpeg(self.src / "Movies" / "big.jpg")
        make_jpeg(self.src / "Movies" / "deep" / "bigger.jpg")
        self.write_excludes(f'"{self.src.as_posix()}":\n  - Movies\n')
        movies = str(self.src / "Movies")
        touched: list[str] = []

        real_stat, real_open, real_path_stat = os.stat, builtins.open, pathlib.Path.stat

        def record(path):
            text = str(path)
            if text.startswith(movies):
                touched.append(text)

        def fake_stat(path, *a, **kw):
            record(path)
            return real_stat(path, *a, **kw)

        def fake_path_stat(self_path, *a, **kw):
            record(self_path)
            return real_path_stat(self_path, *a, **kw)

        def fake_open(file, *a, **kw):
            record(file)
            return real_open(file, *a, **kw)

        with unittest.mock.patch.object(os, "stat", fake_stat), \
                unittest.mock.patch.object(pathlib.Path, "stat", fake_path_stat), \
                unittest.mock.patch.object(builtins, "open", fake_open):
            stats = index(self.cfg, self.conn)

        self.assertEqual(touched, [], f"исключённое поддерево читалось с диска: {touched}")
        self.assertEqual(self.indexed_names(), {"keep.jpg"})
        # ...and the skip is still reported
        self.assertEqual((stats.excluded_dirs, stats.excluded_files), (1, 2))


class TestAlreadyIndexedRowsAreRemoved(ExcludesTestBase):
    def _dependents_of(self, file_id: int) -> None:
        self.conn.execute(
            "INSERT INTO places (file_id, country, city, confidence, updated_at)"
            " VALUES (?, 'ru', 'Moscow', 'exact_gps', '2026-01-01')", (file_id,))
        self.conn.execute(
            "INSERT INTO media_class (file_id, verdict, source, updated_at)"
            " VALUES (?, 'photo', 'clip', '2026-01-01')", (file_id,))
        self.conn.execute(
            "INSERT INTO faces (file_id, bbox, embedding) VALUES (?, '[0,0,1,1]', x'00')",
            (file_id,))
        self.conn.execute(
            "INSERT OR IGNORE INTO events (id, started_at, ended_at, name)"
            " VALUES (1, '2022-01-01', '2022-01-02', 'trip')")
        self.conn.execute("INSERT OR IGNORE INTO event_files (event_id, file_id)"
                          " VALUES (1, ?)", (file_id,))
        self.conn.execute(
            "INSERT INTO dedup_choice (file_id, action, updated_at)"
            " VALUES (?, 'keep', '2026-01-01')", (file_id,))
        self.conn.execute(
            "INSERT INTO manual_overrides (file_id, action, target, updated_at)"
            " VALUES (?, 'exclude', NULL, '2026-01-01')", (file_id,))
        self.conn.commit()

    def test_rows_under_a_new_exclusion_are_deleted_without_dangling_dependents(self):
        make_jpeg(self.src / "keep.jpg")
        make_jpeg(self.src / "Movies" / "a.jpg")
        make_jpeg(self.src / "Movies" / "deep" / "b.jpg")
        first = index(self.cfg, self.conn)
        self.assertEqual(first.added, 3)
        for row in self.conn.execute("SELECT id FROM files").fetchall():
            self._dependents_of(row["id"])

        self.write_excludes(f'"{self.src.as_posix()}":\n  - Movies\n')
        second = index(self.cfg, self.conn)

        self.assertEqual(second.removed_excluded, 2)
        self.assertEqual(self.indexed_names(), {"keep.jpg"})
        alive = {r["id"] for r in self.conn.execute("SELECT id FROM files")}
        for table in ("places", "media_class", "faces", "event_files", "dedup_choice",
                      "manual_overrides"):
            orphans = [r["file_id"] for r in self.conn.execute(
                f"SELECT file_id FROM {table}") if r["file_id"] not in alive]
            self.assertEqual(orphans, [], f"висячие строки в {table}: {orphans}")

    def test_dup_of_reference_to_a_removed_row_is_cleared(self):
        make_jpeg(self.src / "keep.jpg")
        make_jpeg(self.src / "Movies" / "canonical.jpg")
        index(self.cfg, self.conn)
        canonical = self.conn.execute(
            "SELECT id FROM files WHERE path LIKE '%canonical.jpg'").fetchone()["id"]
        self.conn.execute("UPDATE files SET dup_of = ? WHERE path LIKE '%keep.jpg'",
                          (canonical,))
        self.conn.commit()

        self.write_excludes(f'"{self.src.as_posix()}":\n  - Movies\n')
        index(self.cfg, self.conn)

        row = self.conn.execute("SELECT dup_of FROM files").fetchone()
        self.assertIsNone(row["dup_of"])

    def test_move_journal_is_not_touched(self):
        make_jpeg(self.src / "Movies" / "a.jpg")
        index(self.cfg, self.conn)
        file_id = self.conn.execute("SELECT id FROM files").fetchone()["id"]
        self.conn.execute(
            "INSERT INTO move_batches (id, mode, dest_root, started_at)"
            " VALUES (1, 'city', 'D:/out', '2026-01-01')")
        self.conn.execute(
            "INSERT INTO moves (batch_id, file_id, src, dst, hash, status)"
            " VALUES (1, ?, 'a', 'b', 'h', 'done')", (file_id,))
        self.conn.commit()

        self.write_excludes(f'"{self.src.as_posix()}":\n  - Movies\n')
        index(self.cfg, self.conn)

        # history of what really happened stays, even though the index row is gone
        self.assertEqual(self.conn.execute("SELECT COUNT(*) c FROM moves").fetchone()["c"], 1)
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) c FROM move_batches").fetchone()["c"], 1)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) c FROM files").fetchone()["c"], 0)


class TestKeyedByRoot(ExcludesTestBase):
    def test_two_roots_do_not_interfere(self):
        other = self.root / "archive"
        make_jpeg(self.src / "Movies" / "a.jpg")
        make_jpeg(self.src / "temp" / "b.jpg")
        make_jpeg(other / "Movies" / "c.jpg")
        make_jpeg(other / "temp" / "d.jpg")
        self.write_excludes(
            f'"{self.src.as_posix()}":\n  - Movies\n'
            f'"{other.as_posix()}":\n  - temp\n')

        self.cfg.sources = [self.src, other]
        index(self.cfg, self.conn)

        self.assertEqual(self.indexed_names(), {"b.jpg", "c.jpg"})

    def test_switching_the_source_picks_up_its_own_set(self):
        other = self.root / "archive"
        make_jpeg(self.src / "Movies" / "a.jpg")
        make_jpeg(other / "Movies" / "c.jpg")
        self.write_excludes(f'"{self.src.as_posix()}":\n  - Movies\n')

        # the other root has no entry of its own -> nothing is excluded there
        self.cfg.sources = [other]
        index(self.cfg, self.conn)
        self.assertEqual(self.indexed_names(), {"c.jpg"})

        # coming back to the first root restores its set
        self.cfg.sources = [self.src]
        stats = index(self.cfg, self.conn)
        self.assertEqual(stats.added, 0)
        self.assertEqual(self.indexed_names(), {"c.jpg"})


class TestBrokenFileIsSurvivable(ExcludesTestBase):
    def _assert_indexes_everything(self):
        make_jpeg(self.src / "a.jpg")
        make_jpeg(self.src / "Movies" / "b.jpg")
        stats = index(self.cfg, self.conn)
        self.assertEqual(stats.added, 2)
        self.assertEqual((stats.excluded_dirs, stats.excluded_files), (0, 0))

    def test_missing_file_is_not_an_error(self):
        self.assertFalse(self.excludes_file.exists())
        with self.assertNoLogs("sorta.indexer", level=logging.WARNING):
            self._assert_indexes_everything()

    def test_broken_yaml_warns_and_keeps_going(self):
        self.write_excludes("this: [is: not\n  valid yaml\n")
        with self.assertLogs("sorta.indexer", level=logging.WARNING):
            self._assert_indexes_everything()

    def test_unexpected_structure_warns_and_keeps_going(self):
        self.write_excludes("- Movies\n- Screenshots\n")
        with self.assertLogs("sorta.indexer", level=logging.WARNING):
            self._assert_indexes_everything()

    def test_root_value_that_is_not_a_list_warns_and_keeps_going(self):
        self.write_excludes(f'"{self.src.as_posix()}": Movies\n')
        with self.assertLogs("sorta.indexer", level=logging.WARNING):
            self._assert_indexes_everything()

    def test_load_never_raises_on_a_directory_in_place_of_the_file(self):
        directory = self.root / "as-a-dir"
        directory.mkdir()
        with self.assertLogs("sorta.indexer", level=logging.WARNING):
            self.assertFalse(load_excludes(directory))


class TestPathValidation(ExcludesTestBase):
    def test_escaping_values_are_rejected(self):
        for value in ("..", "../..", "Movies/../..", "/etc", "C:/windows",
                      "\\\\server\\share", "..\\windows", "", "   ", 5, None, ["Movies"]):
            with self.subTest(value=value):
                self.assertIsNone(normalize_exclude(value))

    def test_plain_relative_values_are_accepted(self):
        self.assertEqual(normalize_exclude("Movies"), "Movies")
        self.assertEqual(normalize_exclude(" trip/temp/ "), "trip/temp")
        self.assertEqual(normalize_exclude("./Movies/"), "Movies")

    def test_an_escaping_entry_does_not_widen_the_walk(self):
        outside = self.root / "outside"
        make_jpeg(outside / "secret.jpg")
        make_jpeg(self.src / "a.jpg")
        self.write_excludes(
            f'"{self.src.as_posix()}":\n'
            '  - "../outside"\n'
            '  - "/etc"\n'
            '  - "C:/windows"\n'
            '  - "\\\\\\\\server\\\\share"\n')

        with self.assertLogs("sorta.indexer", level=logging.WARNING):
            stats = index(self.cfg, self.conn)

        self.assertEqual(self.indexed_names(), {"a.jpg"})  # walk unchanged
        self.assertEqual((stats.excluded_dirs, stats.excluded_files), (0, 0))
        self.assertTrue((outside / "secret.jpg").exists())  # nothing removed outside


class TestStatsAreCountedRight(ExcludesTestBase):
    def test_excluded_and_removed_counters(self):
        make_jpeg(self.src / "keep.jpg")
        make_jpeg(self.src / "Movies" / "a.jpg")
        make_jpeg(self.src / "Movies" / "b.jpg")
        make_jpeg(self.src / "Movies" / "deep" / "c.jpg")
        (self.src / "Movies" / "notes.txt").write_text("not media", encoding="utf-8")
        make_jpeg(self.src / "Screenshots" / "s.jpg")
        index(self.cfg, self.conn)  # everything indexed first

        self.write_excludes(
            f'"{self.src.as_posix()}":\n  - Movies\n  - Screenshots\n')
        stats = index(self.cfg, self.conn)

        self.assertEqual(stats.excluded_dirs, 2)
        self.assertEqual(stats.excluded_files, 5)  # 4 under Movies (incl. notes.txt) + 1
        self.assertEqual(stats.removed_excluded, 4)  # only media had rows
        self.assertEqual(stats.scanned, 1)
        self.assertEqual(stats.skipped, 1)

    def test_counters_stay_zero_without_an_excludes_file(self):
        make_jpeg(self.src / "a.jpg")
        stats = index(self.cfg, self.conn)
        self.assertEqual(
            (stats.excluded_dirs, stats.excluded_files, stats.removed_excluded), (0, 0, 0))


class TestRegressionWithoutExcludes(ExcludesTestBase):
    def test_walk_yields_exactly_what_rglob_used_to(self):
        make_jpeg(self.src / "a.jpg")
        make_jpeg(self.src / "sub" / "b.jpg")
        make_jpeg(self.src / "sub" / "deep" / "c.jpg")
        make_jpeg(self.src / ".hidden" / "d.jpg")
        make_jpeg(self.src / "@eaDir" / "e.jpg")
        (self.src / "notes.txt").write_text("skip", encoding="utf-8")

        skip = set(self.cfg.index.skip_dirs)
        expected = {
            p for p in sorted(self.src.rglob("*"))
            if not any(part in skip or part.startswith(".") for part in p.parts)
            and p.is_file() and self.cfg.index.media_type_of(p.suffix) is not None
        }
        self.assertEqual(set(_walk(self.cfg)), expected)
        self.assertEqual({p.name for p in expected}, {"a.jpg", "b.jpg", "c.jpg"})

    def test_index_without_the_file_behaves_as_before(self):
        make_jpeg(self.src / "a.jpg")
        make_jpeg(self.src / "sub" / "b.jpg")
        first = index(self.cfg, self.conn)
        second = index(self.cfg, self.conn)
        self.assertEqual((first.added, first.updated, first.skipped), (2, 0, 0))
        self.assertEqual((second.added, second.updated, second.skipped), (0, 0, 2))
        self.assertEqual(self.indexed_names(), {"a.jpg", "b.jpg"})


class TestExcludesFileLocation(ExcludesTestBase):
    def test_default_is_next_to_the_database(self):
        cfg = Config(sources=[self.src], database=self.root / "db" / "sorta.db", raw={})
        self.assertEqual(excludes_path(cfg), self.root / "db" / "excludes.yaml")

    def test_config_value_wins(self):
        cfg = Config(sources=[self.src], database=self.root / "sorta.db",
                     raw={"index": {"excludes_file": str(self.root / "custom.yaml")}})
        self.assertEqual(excludes_path(cfg), self.root / "custom.yaml")

    def test_unusable_config_value_falls_back_to_the_default(self):
        for value in ("", "   ", None, 7, ["x"]):
            with self.subTest(value=value):
                cfg = Config(sources=[self.src], database=self.root / "sorta.db",
                             raw={"index": {"excludes_file": value}})
                self.assertEqual(excludes_path(cfg), self.root / "excludes.yaml")


class TestSaveExcludes(ExcludesTestBase):
    def test_saving_one_root_keeps_the_others(self):
        other = self.root / "archive"
        save_excludes(self.excludes_file, self.src, ["Movies", "Screenshots"])
        save_excludes(self.excludes_file, other, ["temp"])
        save_excludes(self.excludes_file, self.src, ["Movies"])

        loaded = load_excludes(self.excludes_file)
        self.assertEqual(loaded.for_root(self.src), {"Movies"})
        self.assertEqual(loaded.for_root(other), {"temp"})

    def test_empty_list_drops_the_root_key(self):
        save_excludes(self.excludes_file, self.src, ["Movies"])
        save_excludes(self.excludes_file, self.src, [])
        self.assertFalse(load_excludes(self.excludes_file))
        self.assertNotIn(self.src.name, self.excludes_file.read_text(encoding="utf-8"))

    def test_rejected_values_are_not_written(self):
        with self.assertLogs("sorta.indexer", level=logging.WARNING):
            accepted = save_excludes(self.excludes_file, self.src, ["Movies", "../x", "/etc"])
        self.assertEqual(accepted, ["Movies"])
        self.assertEqual(load_excludes(self.excludes_file).for_root(self.src), {"Movies"})

    def test_no_temporary_file_is_left_behind(self):
        save_excludes(self.excludes_file, self.src, ["Movies"])
        leftovers = [p.name for p in self.root.iterdir() if p.name.endswith(".tmp")]
        self.assertEqual(leftovers, [])


class TestDropExcludedRowsIsIdempotent(ExcludesTestBase):
    def test_second_call_removes_nothing(self):
        make_jpeg(self.src / "Movies" / "a.jpg")
        index(self.cfg, self.conn)
        excludes = Excludes({os.path.normcase(str(self.src.resolve())): ["Movies"]})
        self.assertEqual(drop_excluded_rows(self.cfg, self.conn, excludes), 1)
        self.assertEqual(drop_excluded_rows(self.cfg, self.conn, excludes), 0)


if __name__ == "__main__":
    unittest.main()
