"""F113: the frame-quality measurement — the arithmetic behind the tables.

The script exists because three thresholds shipped as guesses and a human has to replace
them by looking at a distribution. So what is tested is that the tables tell the truth:
the sweep counts what somebody counting by hand would count, the percentiles come off the
real values, the band is split by the reason a frame is in it, and a cache round-trips.

No model, no GPU, no photo — everything below is arithmetic over per-frame aggregates.
And, as with every measurement in this project, nothing the script prints may identify a
frame.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

from sorta import junk

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "measure_frame_quality.py"


def _load_script():
    """Import scripts/measure_frame_quality.py — it is a script, not a package module."""
    spec = importlib.util.spec_from_file_location("measure_frame_quality", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


measure = _load_script()


def frame(fid=1, pet_class=None, pet_score=0.0, sharpness=None, subject_score=0.99):
    return measure.Frame(file_id=fid, pet_class=pet_class, pet_score=pet_score,
                         sharpness=sharpness, subject_score=subject_score)


def settings(**kwargs):
    base = dict(pets=True, pet_threshold=0.6, sharpness_max_edge=512,
                sharpness_band=(30.0, 300.0), subject_score_min=0.9,
                vlm_quality=True, vlm_scope="groups")
    base.update(kwargs)
    return junk.QualitySettings(**base)


class TestPetSweep(unittest.TestCase):
    def test_counts_what_a_threshold_would_fire_on(self):
        frames = [frame(1, "cat", 0.95), frame(2, "dog", 0.65),
                  frame(3, "cat", 0.35), frame(4, None, 0.0)]
        rows = {r.threshold: r for r in measure.sweep_pets(frames, [0.3, 0.6, 0.9])}
        self.assertEqual(rows[0.3].fired, 3)
        self.assertEqual(rows[0.6].fired, 2)
        self.assertEqual(rows[0.9].fired, 1)

    def test_splits_by_class(self):
        frames = [frame(1, "cat", 0.95), frame(2, "dog", 0.95), frame(3, "cat", 0.95)]
        (row,) = measure.sweep_pets(frames, [0.5])
        self.assertEqual(row.by_class, {"cat": 2, "dog": 1})

    def test_a_frame_without_a_pet_class_never_fires(self):
        (row,) = measure.sweep_pets([frame(1, None, 1.0)], [0.0])
        self.assertEqual(row.fired, 0)

    def test_thresholds_below_the_configured_one_are_answerable(self):
        """The reason the cache stores the class unfiltered: a lower threshold must be
        askable after the fact, which a class already cut at the configured one cannot
        answer."""
        (row,) = measure.sweep_pets([frame(1, "cat", 0.4)], [0.3])
        self.assertEqual(row.fired, 1)


class TestPercentiles(unittest.TestCase):
    def test_nearest_rank_returns_values_that_exist(self):
        values = [float(v) for v in range(1, 101)]
        got = dict(measure.percentiles(values, (50, 100)))
        self.assertIn(got[50], values)
        self.assertEqual(got[100], 100.0)

    def test_empty_input_is_empty_output(self):
        self.assertEqual(measure.percentiles([]), [])

    def test_single_value(self):
        self.assertEqual(measure.percentiles([7.0], (5, 95)), [(5, 7.0), (95, 7.0)])


class TestBandSummary(unittest.TestCase):
    """The band block has to separate the two reasons — they are two different knobs."""

    def test_counts_each_reason_and_the_union(self):
        q = settings()
        frames = [
            frame(1, sharpness=100.0, subject_score=0.99),   # by sharpness only
            frame(2, sharpness=5000.0, subject_score=0.20),  # by subject only
            frame(3, sharpness=100.0, subject_score=0.20),   # by both
            frame(4, sharpness=5000.0, subject_score=0.99),  # by neither
        ]
        text = measure.format_band(frames, q)
        self.assertIn("по резкости", text)
        # 2 by sharpness, 2 by subject, 3 in the union — the union is not the sum
        self.assertRegex(text, r"по резкости.*\s2\s")
        self.assertRegex(text, r"по сюжету.*\s2\s")
        self.assertRegex(text, r"итого в полосе.*\s3\s")

    def test_uses_the_pipelines_own_band_function(self):
        # a band wide enough to swallow everything must report everything
        q = settings(sharpness_band=(0.0, 1e9))
        text = measure.format_band([frame(1, sharpness=1.0), frame(2, sharpness=2.0)], q)
        self.assertRegex(text, r"итого в полосе.*\s2\s")


class TestFormattingIsAnonymous(unittest.TestCase):
    """Privacy: the tables are aggregates. No path, no basename, ever."""

    def test_no_identifier_reaches_the_output(self):
        q = settings()
        frames = [frame(101, "cat", 0.9, 100.0, 0.5), frame(102, None, 0.1, 900.0, 0.99)]
        text = "\n".join([
            measure.format_pets(frames, measure.sweep_pets(frames, [0.5, 0.9]), 0.6),
            measure.format_sharpness(frames, q),
            measure.format_band(frames, q),
        ])
        for forbidden in ("/photos", ".jpg", "101", "102"):
            self.assertNotIn(forbidden, text)

    def test_the_configured_threshold_is_marked_in_the_pet_table(self):
        frames = [frame(1, "cat", 0.9)]
        text = measure.format_pets(frames, measure.sweep_pets(frames, [0.5, 0.6]), 0.6)
        self.assertIn(" 0.60*", text)

    def test_undecodable_frames_are_reported_as_null_not_zero(self):
        frames = [frame(1, sharpness=None), frame(2, sharpness=50.0)]
        text = measure.format_sharpness(frames, settings())
        self.assertIn("не декодировалось: 1", text)


class TestCache(unittest.TestCase):
    def test_round_trip(self):
        frames = [frame(1, "cat", 0.9, 100.0, 0.5), frame(2, None, 0.1, None, 0.99)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.json"
            measure.save_cache(path, frames)
            self.assertEqual(measure.load_cache(path), frames)

    def test_the_cache_holds_no_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.json"
            measure.save_cache(path, [frame(1, "cat", 0.9, 100.0, 0.5)])
            self.assertNotIn("photos", path.read_text(encoding="utf-8"))

    def test_a_cache_of_another_version_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.json"
            path.write_text(json.dumps({"version": 999, "frames": []}), encoding="utf-8")
            with self.assertRaises(SystemExit):
                measure.load_cache(path)


class TestSampling(unittest.TestCase):
    def test_only_files_that_exist_are_sampled(self):
        from sorta.db import connect

        with tempfile.TemporaryDirectory() as tmp:
            real = Path(tmp) / "real.jpg"
            real.write_bytes(b"x")
            db = Path(tmp) / "s.db"
            conn = connect(db)
            for path in (str(real), str(Path(tmp) / "gone.jpg")):
                conn.execute(
                    "INSERT INTO files (path, size, mtime, ext, media_type, indexed_at) "
                    "VALUES (?, 1, 0.0, 'jpg', 'photo', 'x')", (path,))
            conn.commit()
            conn.close()
            rows = measure.sample_rows(str(db), 10, seed=1)
        self.assertEqual([r["path"] for r in rows], [str(real)])

    def test_the_floor_is_stated_and_is_the_briefs_number(self):
        self.assertEqual(measure.MIN_SAMPLE, 200)
        self.assertGreaterEqual(measure.DEFAULT_SAMPLE, measure.MIN_SAMPLE)


if __name__ == "__main__":
    unittest.main()
