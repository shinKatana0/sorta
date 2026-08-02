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
import sqlite3
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


def frame(fid=1, pet_class=None, pet_score=0.0, sharpness=None, subject_score=0.99,
          pet_vlm=None):
    return measure.Frame(file_id=fid, pet_class=pet_class, pet_score=pet_score,
                         sharpness=sharpness, subject_score=subject_score,
                         pet_vlm=pet_vlm)


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


class TestCandidateTable(unittest.TestCase):
    """F130: the table the candidate threshold is chosen from — a count and its price."""

    def test_counts_what_each_threshold_would_send_to_the_model(self):
        frames = [frame(1, "cat", 0.95), frame(2, "cat", 0.45), frame(3, None, 0.05)]
        text = measure.format_candidates(frames, [0.1, 0.5], 0.3)
        self.assertRegex(text, r"0\.10\s+2\s")   # 0.05 is below even the lowest cut
        self.assertRegex(text, r"0\.50\s+1\s")

    def test_the_price_is_the_measured_one(self):
        frames = [frame(i, "cat", 0.9) for i in range(1, 101)]
        text = measure.format_candidates(frames, [0.5], 0.3)
        # 100 frames x 0.78 s = 78 s = 1.3 min
        self.assertIn("1.3 мин", text)

    def test_the_configured_threshold_is_marked(self):
        text = measure.format_candidates([frame(1, "cat", 0.9)], [0.3, 0.5], 0.3)
        self.assertIn(" 0.30*", text)


class TestAnswerSummary(unittest.TestCase):
    """What the stored answers changed — counted in both directions, never as a total."""

    def q(self):
        return settings(pet_threshold=0.7)

    def test_it_says_so_when_the_check_has_not_run(self):
        text = measure.format_answers([frame(1, "cat", 0.9)], self.q())
        self.assertIn("features.pets_verify", text)

    def test_both_directions_are_counted_separately(self):
        frames = [
            frame(1, "cat", 0.95, pet_vlm="depiction"),  # marked before, not after
            frame(2, "cat", 0.35, pet_vlm="real"),       # not marked before, marked after
            frame(3, "cat", 0.90, pet_vlm="real"),       # marked both times
            frame(4, "cat", 0.10, pet_vlm=None),         # never asked, never marked
        ]
        text = measure.format_answers(frames, self.q())
        self.assertRegex(text, r"до проверки.*:\s+2")
        self.assertRegex(text, r"снято проверкой.*:\s+1")
        self.assertRegex(text, r"добавлено проверкой.*:\s+1")
        self.assertRegex(text, r"после проверки:\s+2")

    def test_the_answers_are_broken_down_by_class(self):
        frames = [frame(1, "cat", 0.9, pet_vlm="real"),
                  frame(2, "cat", 0.9, pet_vlm="real"),
                  frame(3, "cat", 0.9, pet_vlm="none")]
        self.assertIn("none 1, real 2", measure.format_answers(frames, self.q()))


class TestAccuracy(unittest.TestCase):
    """The block the feature is accepted or rejected on — F122's arithmetic, repeatable."""

    def test_precision_and_recall_of_an_unweighted_sample(self):
        # one band, every frame labelled: the weights are 1 and the numbers are the plain
        # counts, which is the case a reader can check by hand.
        frames = [frame(1, "cat", 0.95), frame(2, "cat", 0.90),
                  frame(3, "cat", 0.85), frame(4, "cat", 0.80)]
        labels = {1: True, 2: True, 3: False, 4: True}
        a = measure.accuracy(frames, labels, 0.7, verified=False)
        self.assertAlmostEqual(a.precision, 0.75)  # 3 of the 4 marked are animals
        self.assertAlmostEqual(a.recall, 1.0)      # and every animal is marked

    def test_the_check_is_what_moves_the_numbers(self):
        frames = [frame(1, "cat", 0.95, pet_vlm="depiction"),  # a plush toy, rejected
                  frame(2, "cat", 0.95, pet_vlm="real"),
                  frame(3, "cat", 0.95, pet_vlm="real")]
        labels = {1: False, 2: True, 3: True}
        before = measure.accuracy(frames, labels, 0.7, verified=False)
        after = measure.accuracy(frames, labels, 0.7, verified=True)
        self.assertAlmostEqual(before.precision, 2 / 3)
        self.assertAlmostEqual(after.precision, 1.0)
        self.assertAlmostEqual(after.recall, 1.0)

    def test_recall_counts_the_animals_the_threshold_never_reached(self):
        """The half of the cascade a precision figure cannot show: a labelled animal below
        the threshold is a miss until the check marks it."""
        frames = [frame(1, "cat", 0.35, pet_vlm="real")]
        labels = {1: True}
        self.assertAlmostEqual(
            measure.accuracy(frames, labels, 0.7, verified=False).recall, 0.0)
        self.assertAlmostEqual(
            measure.accuracy(frames, labels, 0.7, verified=True).recall, 1.0)

    def test_labels_are_weighted_back_to_their_band(self):
        """A stratified sample only answers a question about the collection once each
        label carries the size of the band it was drawn from."""
        # band 0.9-1.0: 2 frames, 1 labelled (weight 2). Band 0.0-0.1: 10 frames, 1
        # labelled (weight 10). Both labelled frames are animals; only the high one is
        # marked, so recall is 2 / 12 and not 1 / 2.
        frames = ([frame(1, "cat", 0.95), frame(2, "cat", 0.95)]
                  + [frame(10 + i, "cat", 0.05) for i in range(10)])
        labels = {1: True, 10: True}
        a = measure.accuracy(frames, labels, 0.7, verified=False)
        self.assertAlmostEqual(a.precision, 1.0)
        self.assertAlmostEqual(a.recall, 2 / 12)

    def test_an_unlabelled_band_contributes_nothing_rather_than_zero(self):
        weights = measure.band_weights(
            [frame(1, "cat", 0.95), frame(2, "cat", 0.05)], {1})
        self.assertEqual(set(weights), {1})

    def test_an_empty_sample_is_not_a_division_by_zero(self):
        a = measure.accuracy([frame(1, "cat", 0.9)], {}, 0.7, verified=False)
        self.assertEqual((a.precision, a.recall), (0.0, 0.0))

    def test_the_block_warns_when_the_sample_is_below_the_floor(self):
        frames = [frame(1, "cat", 0.9)]
        text = measure.format_accuracy(frames, {1: True}, settings(pet_threshold=0.7))
        self.assertIn("ВНИМАНИЕ", text)
        self.assertIn("92%", text)  # the F122 numbers the result is compared against

    def test_the_block_names_both_rows(self):
        frames = [frame(i, "cat", 0.9, pet_vlm="real") for i in range(1, 4)]
        text = measure.format_accuracy(frames, {1: True, 2: True, 3: True},
                                       settings(pet_threshold=0.7))
        self.assertIn("до проверки", text)
        self.assertIn("после проверки", text)


class TestLabelWorksheet(unittest.TestCase):
    """The worksheet: stratified, file ids only, and usable half-filled."""

    def test_it_is_stratified_over_the_score_bands(self):
        frames = ([frame(i, "cat", 0.95) for i in range(1, 21)]
                  + [frame(100 + i, "cat", 0.05) for i in range(20)])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.json"
            written = measure.write_label_template(path, frames, per_band=3, seed=1)
            data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(written, 6)  # three from each of the two non-empty bands
        self.assertEqual(set(data.values()), {None})
        high = sum(1 for fid in data if int(fid) <= 20)
        self.assertEqual(high, 3)

    def test_it_holds_no_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.json"
            measure.write_label_template(path, [frame(1, "cat", 0.9)], 5, seed=1)
            text = path.read_text(encoding="utf-8")
        self.assertNotIn("photos", text)
        self.assertNotIn(".jpg", text)

    def test_a_half_filled_sheet_drops_the_unanswered_frames(self):
        """An unanswered frame is not a `false`: reading it as one would invent the very
        labels the sheet exists to collect."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.json"
            path.write_text(json.dumps({"1": True, "2": False, "3": None}),
                            encoding="utf-8")
            self.assertEqual(measure.load_labels(path), {1: True, 2: False})


class TestStoredAnswersComeFromTheIndex(unittest.TestCase):
    """The answers are read from the DB on every run — a cache must not describe the past."""

    def test_the_answers_of_a_live_run_are_attached(self):
        from sorta.db import connect

        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "a.db"
            conn = connect(db)
            conn.execute(
                "INSERT INTO files (path, size, mtime, ext, media_type, indexed_at) "
                "VALUES ('/a.jpg', 1, 0.0, 'jpg', 'photo', 'x')")
            conn.execute(
                "INSERT INTO frame_quality (file_id, pet, pet_score, pet_vlm, source,"
                " updated_at) VALUES (1, NULL, 0.9, 'depiction', 'vlm#1', 'x')")
            conn.commit()
            conn.close()
            frames = measure.read_pet_vlm(str(db), [frame(1, "cat", 0.9), frame(2)])
        self.assertEqual(frames[0].pet_vlm, "depiction")
        self.assertIsNone(frames[1].pet_vlm)

    def test_an_index_without_the_column_answers_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "old.db"
            raw = sqlite3.connect(db)
            raw.execute("CREATE TABLE frame_quality (file_id INTEGER PRIMARY KEY)")
            raw.commit()
            raw.close()
            frames = measure.read_pet_vlm(str(db), [frame(1, "cat", 0.9)])
        self.assertIsNone(frames[0].pet_vlm)


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
