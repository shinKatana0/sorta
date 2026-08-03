"""F164: the three measurement scripts — the arithmetic that decides a default.

Each of them prints a table and then ONE sentence saying whether a number in the product
moves. That sentence is the whole point of the tools (the F90 rule: a threshold is
changed in front of a table, never quietly), so it is what is tested here, together with
the two properties every measurement in this project has to have — the population is the
one the stage really works on, and nothing printed identifies a frame.

No model, no GPU and no photograph: the sweeps themselves are I/O over hardware, and what
is checked below is the part that is not.
"""
from __future__ import annotations

import importlib.util
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

from sorta.db import connect

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load(name: str):
    """Import scripts/<name>.py — a script, not a package module (the F140 pattern)."""
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


write = _load("measure_junk_write")
ocr = _load("measure_ocr_workers")
vlm = _load("measure_vlm_workers")

# A path that must never reach a report. Every table below is checked against it — the
# rule that a measurement about documents must not become a list of where they are.
SECRET = "паспорт_ивановой.jpg"


class TestJunkWriteMeasurement(unittest.TestCase):
    """The commit strategies, run for real against a throwaway database."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = Path(self.tmp.name) / "m.db"

    def rows_written(self, conn: sqlite3.Connection) -> int:
        return int(conn.execute("SELECT COUNT(*) FROM media_class").fetchone()[0])

    def test_each_strategy_writes_every_row(self):
        conn = write._prepare_db(self.db, 20)
        self.addCleanup(conn.close)
        for mode in ("stage", "chunk", "row"):
            with self.subTest(mode=mode):
                conn.execute("DELETE FROM media_class")
                conn.commit()
                result = write.measure_upsert(conn, 20, 8, mode)
                self.assertEqual(self.rows_written(conn), 20)
                self.assertEqual(result.rows, 20)
                self.assertGreater(result.seconds, 0.0)

    def test_the_number_of_commits_is_what_the_strategy_says(self):
        conn = write._prepare_db(self.db, 20)
        self.addCleanup(conn.close)
        self.assertEqual(write.measure_upsert(conn, 20, 8, "stage").commits, 1)
        self.assertEqual(write.measure_upsert(conn, 20, 8, "chunk").commits, 3)
        self.assertEqual(write.measure_upsert(conn, 20, 8, "row").commits, 20)

    def test_the_embedding_write_stores_a_vector_per_row(self):
        conn = write._prepare_db(self.db, 5)
        self.addCleanup(conn.close)
        result = write.measure_embedding_write(conn, 5)
        self.assertEqual(result.commits, 1)
        stored = conn.execute("SELECT dim, LENGTH(vec) FROM clip_embeddings").fetchall()
        self.assertEqual(len(stored), 5)
        for dim, blob in stored:
            self.assertEqual(dim, write.EMBEDDING_DIM)
            self.assertEqual(blob, 4 * write.EMBEDDING_DIM)  # float32

    def test_a_cheap_row_says_the_phase_is_not_sqlite(self):
        rows = [write.WriteRow("stage", 1000, 0.005, 1),
                write.WriteRow("row", 1000, 2.0, 1000)]
        self.assertIn("не SQLite", write.verdict(rows))

    def test_an_expensive_row_says_to_keep_looking(self):
        """If one transaction really did cost milliseconds a row, batching is not the fix."""
        rows = [write.WriteRow("stage", 1000, 20.0, 1),
                write.WriteRow("row", 1000, 25.0, 1000)]
        self.assertIn("искать причину дальше", write.verdict(rows))

    def test_one_strategy_alone_decides_nothing(self):
        self.assertIn("сравнивать не с чем",
                      write.verdict([write.WriteRow("stage", 10, 0.1, 1)]))

    def test_the_tables_are_aggregates_and_name_no_frame(self):
        rows = [write.WriteRow("stage", 10, 0.1, 1), write.WriteRow("row", 10, 0.5, 10)]
        table = write.format_write_table(rows, 16)
        self.assertNotIn(SECRET, table)
        self.assertIn("одна транзакция", table)
        # The breakdown names COSTS, never frames — and it prints the phase it is being
        # compared with, which is the only way a share column means anything.
        frames = [write.FrameRow("sharpness", 10, 0.1)]
        self.assertNotIn(SECRET, write.format_frame_table(frames, 19.4))
        self.assertIn("19.4", write.format_frame_table(frames, 19.4))


class TestOcrWorkerSweep(unittest.TestCase):
    """Which frames it measures on, and when it says the ceiling may move."""

    def row(self, workers: int, seconds: float, detectors: int | None = None):
        return ocr.WorkerRow(workers=workers, frames=100, seconds=seconds,
                             detectors=workers if detectors is None else detectors,
                             answered=100, peak_vram_mb=None)

    def test_the_population_is_canonical_photos_without_faces(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "m.db"
            conn = connect(db)
            here = (Path(tmp) / "real.jpg")
            here.write_bytes(b"x")
            rows = [("kept", str(here), None, None), ("gone", "/nope/missing.jpg", None, None),
                    ("dup", str(here) + "2", 1, None), ("bad", str(here) + "3", None, "boom")]
            for name, path, dup, error in rows:
                conn.execute(
                    """INSERT INTO files (path, size, mtime, ext, media_type, width,
                           height, dup_of, error, indexed_at)
                       VALUES (?, 1, 0, 'jpg', 'photo', 10, 10, ?, ?, '2026-01-01')""",
                    (path, dup, error))
            faced = Path(tmp) / "faced.jpg"
            faced.write_bytes(b"x")
            cur = conn.execute(
                """INSERT INTO files (path, size, mtime, ext, media_type, width, height,
                       indexed_at) VALUES (?, 1, 0, 'jpg', 'photo', 10, 10, '2026-01-01')""",
                (str(faced),))
            conn.execute(
                "INSERT INTO faces (file_id, bbox, embedding) VALUES (?, '[1,2,3,4]', ?)",
                (cur.lastrowid, b"\x00"))
            conn.commit()
            conn.close()
            jobs = ocr.sample_jobs(str(db), 10, seed=1)
        self.assertEqual([job[1] for job in jobs], [str(here)])

    def test_the_sample_is_capped(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "m.db"
            conn = connect(db)
            for i in range(5):
                path = Path(tmp) / f"{i}.jpg"
                path.write_bytes(b"x")
                conn.execute(
                    """INSERT INTO files (path, size, mtime, ext, media_type, width,
                           height, indexed_at)
                       VALUES (?, 1, 0, 'jpg', 'photo', 10, 10, '2026-01-01')""",
                    (str(path),))
            conn.commit()
            conn.close()
            self.assertEqual(len(ocr.sample_jobs(str(db), 3, seed=1)), 3)

    def test_a_real_win_raises_the_ceiling(self):
        rows = [self.row(1, 40.0), self.row(4, 10.0), self.row(8, 5.0)]
        self.assertIn("потолок поднимать", ocr.outcome(rows, default=4))

    def test_a_small_win_does_not(self):
        rows = [self.row(1, 40.0), self.row(4, 10.0), self.row(8, 9.5)]
        self.assertIn("потолок оставить", ocr.outcome(rows, default=4))

    def test_a_shrunken_pool_cannot_raise_the_ceiling(self):
        """Its row measures a smaller pool than its label — VRAM is the whole question."""
        rows = [self.row(4, 10.0), self.row(8, 2.0, detectors=4)]
        self.assertIn("сравнивать не с чем", ocr.outcome(rows, default=4))

    def test_the_table_marks_the_default_and_flags_a_shrunken_pool(self):
        table = ocr.format_table([self.row(4, 10.0), self.row(8, 9.0, detectors=5)],
                                 default=4)
        self.assertIn("4*", table)
        self.assertIn("5 !", table)
        self.assertNotIn(SECRET, table)  # counts only — never which frames were read

    def test_speedup_is_measured_against_the_first_row(self):
        table = ocr.format_table([self.row(1, 40.0), self.row(4, 10.0)], default=4)
        self.assertIn("x4.00", table)


class TestVlmWorkerSweep(unittest.TestCase):
    """The busy share, the F101 invariant and the sentence about the default."""

    def row(self, workers: int, wall: float, gen: float, labels=("a", "b")):
        return vlm.WorkerRow(workers=workers, labels=tuple(labels), wall_sec=wall,
                             gen_sec=gen, cpu_cores=1.0, peak_rss_mb=100.0)

    def test_the_busy_share_is_the_model_half_over_the_wall_clock(self):
        self.assertAlmostEqual(self.row(4, 10.0, 5.0).gpu_busy_pct, 50.0)
        self.assertEqual(self.row(4, 0.0, 0.0).gpu_busy_pct, 0.0)

    def test_frames_per_second_counts_the_labels(self):
        self.assertAlmostEqual(self.row(4, 2.0, 1.0).frames_per_sec, 1.0)

    def test_identical_labels_are_the_invariant_holding(self):
        rows = [self.row(1, 10.0, 5.0), self.row(4, 8.0, 5.0)]
        report, ok = vlm.format_invariant(rows)
        self.assertTrue(ok)
        self.assertIn("совпадение полное", report)

    def test_a_moved_label_stops_the_measurement(self):
        rows = [self.row(1, 10.0, 5.0), self.row(4, 8.0, 5.0, labels=("a", "c"))]
        report, ok = vlm.format_invariant(rows)
        self.assertFalse(ok)
        self.assertIn("РАСХОЖДЕНИЙ 1", report)

    def test_a_missing_label_counts_as_a_divergence(self):
        rows = [self.row(1, 10.0, 5.0), self.row(4, 8.0, 5.0, labels=("a",))]
        _report, ok = vlm.format_invariant(rows)
        self.assertFalse(ok)

    def test_a_lone_row_compares_with_nothing(self):
        report, ok = vlm.format_invariant([self.row(4, 10.0, 5.0)])
        self.assertTrue(ok)
        self.assertIn("сравнивать не с чем", report)

    def test_a_slower_pool_keeps_the_default(self):
        rows = [self.row(4, 10.0, 5.0), self.row(8, 12.0, 5.0)]
        self.assertIn("дефолт оставить 4", vlm.outcome(rows, default=4))

    def test_a_real_win_names_the_knee_and_not_the_maximum(self):
        """A default should be the smallest count that is already as fast as it gets."""
        rows = [self.row(4, 10.0, 5.0), self.row(8, 5.0, 5.0), self.row(12, 4.9, 5.0)]
        answer = vlm.outcome(rows, default=4)
        self.assertIn("дефолт поднимать до 8", answer)

    def test_the_table_holds_no_paths_and_marks_the_default(self):
        table = vlm.format_table([self.row(4, 10.0, 5.0)], default=4, mode="тест")
        self.assertIn("4*", table)
        self.assertNotIn(SECRET, table)

    def test_the_peak_memory_reads_as_a_number_or_says_nothing(self):
        peak = vlm.peak_rss_mb()
        self.assertTrue(peak is None or peak > 0.0)


if __name__ == "__main__":
    unittest.main()
