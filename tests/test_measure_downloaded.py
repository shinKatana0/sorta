"""F163 phase 0: the tables "not your photograph" would be decided from — their arithmetic.

The script exists because the class has no measurement at all: thirteen frames looked at by
eye, a metadata rule that separated them 13 of 13 on a sample selected by that same rule,
and 31-41% precision as soon as 500 honest frames were labelled. So what has to be right
here is that the numbers mean what the columns say:

* the population table counts the metadata slices and crosses them with the verdicts the
  pipeline writes today — the question of how much of the class is already caught;
* the sample is stratified over the POOLED ranking (the best rank over all wordings), so it
  is not drawn by the one wording it would then flatter, and every precision and recall is
  WEIGHTED by the design — a band sampled at one in six hundred must not count the same as
  one sampled at one in four;
* recall keeps the whole class in its denominator even when the candidate pool is narrowed
  to "no camera in EXIF": a selector that loses frames has to show it;
* the baseline rules are the brief's own, and a rule about size never fires on a frame
  whose size the index does not know;
* nothing printed identifies a frame, and the worksheet holds file ids alone.

No model, no GPU, no photo: everything below is arithmetic over labels, ranks and rows.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

from sorta import junk
from sorta.db import connect

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "measure_downloaded.py"


def _load_script():
    """Import scripts/measure_downloaded.py — it is a script, not a package module."""
    spec = importlib.util.spec_from_file_location("measure_downloaded", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


measure = _load_script()


def frame(file_id: int, verdict: str | None = "photo", camera: bool = True,
          mp: float | None = 12.0):
    return measure.Frame(file_id=file_id, verdict=verdict, has_camera=camera, megapixels=mp)


class TestTheMetadataSignal(unittest.TestCase):
    """The two facts the brief's rule is built on, read off the index the way it stores."""

    def test_either_exif_field_names_a_camera(self):
        self.assertTrue(measure.has_camera_exif("Apple", None))
        self.assertTrue(measure.has_camera_exif(None, "iPhone 13 Pro"))
        self.assertTrue(measure.has_camera_exif("vivo", "X90 Pro+"))

    def test_an_absent_or_blank_tag_is_not_a_camera(self):
        """A messenger strips EXIF and exiftool hands back empty tags — that is the whole
        weakness of the signal, and it must not read as a camera."""
        self.assertFalse(measure.has_camera_exif(None, None))
        self.assertFalse(measure.has_camera_exif("", ""))
        self.assertFalse(measure.has_camera_exif("   ", None))

    def test_megapixels_come_from_the_stored_size(self):
        self.assertAlmostEqual(measure.megapixels(4000, 3000), 12.0)

    def test_an_unknown_size_is_none_and_never_zero(self):
        for width, height in ((None, 3000), (4000, None), (0, 3000), ("x", 3)):
            with self.subTest(size=(width, height)):
                self.assertIsNone(measure.megapixels(width, height))

    def test_a_frame_of_unknown_size_is_not_small(self):
        """Otherwise "not measured" would silently join the band the rules fire in."""
        self.assertFalse(frame(1, mp=None).smaller_than(1.0))
        self.assertTrue(frame(1, mp=0.5).smaller_than(1.0))
        self.assertFalse(frame(1, mp=1.0).smaller_than(1.0))


class TestThePopulationTable(unittest.TestCase):
    """Table 1: the slices, and what the classifier calls the frames inside them."""

    def frames(self):
        return [
            frame(1, camera=True, mp=12.0),                       # an ordinary photograph
            frame(2, camera=False, mp=0.5),                       # small, no camera
            frame(3, verdict="meme", camera=False, mp=0.4),       # already caught
            frame(4, verdict="screenshot", camera=False, mp=2.0),  # already caught
            frame(5, verdict=None, camera=False, mp=None),        # never classified
        ]

    def rows(self):
        return {r.name: r for r in measure.population_table(self.frames())}

    def test_every_slice_is_counted_against_the_whole_collection(self):
        rows = self.rows()
        self.assertEqual(rows["вся коллекция"].frames, 5)
        self.assertAlmostEqual(rows["вся коллекция"].share, 1.0)
        self.assertEqual(rows["нет камеры"].frames, 4)
        self.assertAlmostEqual(rows["нет камеры"].share, 0.8)

    def test_the_size_slices_are_the_briefs_own(self):
        rows = self.rows()
        self.assertEqual(rows["нет камеры и меньше 1 Мп"].frames, 2)   # files 2 and 3
        self.assertEqual(rows["нет камеры и меньше 3 Мп"].frames, 3)   # ...and file 4
        self.assertEqual(rows["меньше 1 Мп"].frames, 2)

    def test_the_slice_says_how_much_of_it_is_already_caught(self):
        """The reading the brief asks for first: if `meme` and `screenshot` already hold
        the frames, the feature is smaller than it looks."""
        verdicts = self.rows()["нет камеры"].verdicts
        self.assertEqual(verdicts["meme"], 1)
        self.assertEqual(verdicts["screenshot"], 1)
        self.assertEqual(verdicts["нет класса"], 1)

    def test_an_empty_collection_is_not_a_division_by_zero(self):
        (row, *_rest) = measure.population_table([])
        self.assertEqual((row.frames, row.share), (0, 0.0))


class TestTheBands(unittest.TestCase):
    """The sampling design: the bands, their sizes, and the pooling behind them."""

    def test_the_tail_of_the_ranking_is_a_band_of_its_own(self):
        """Without it the worksheet would hold no frame the query dislikes, and a recall
        measured on such a sample cannot come out wrong — the trap of the thirteen."""
        bands = measure.rank_bands(20000)
        self.assertEqual(bands[0], (0, 100))
        self.assertEqual(bands[-1], (5000, 20000))

    def test_a_short_ranking_gets_the_bands_it_supports(self):
        self.assertEqual(measure.rank_bands(60), [(0, 60)])
        self.assertEqual(measure.rank_bands(150), [(0, 100), (100, 150)])
        self.assertEqual(measure.rank_bands(0), [])

    def test_the_pooled_rank_is_the_best_one_over_the_wordings(self):
        ranks = measure.pooled_ranks({"a": [1, 2, 3], "b": [3, 1, 2]})
        self.assertEqual(ranks, {1: 0, 2: 1, 3: 0})

    def test_bands_are_measured_by_counting_frames_and_not_by_their_edges(self):
        """Pooled ranks are not a permutation — two frames can both be first, each for a
        different wording — so a band holds fewer frames than its width."""
        ranks = {10: 0, 11: 0, 12: 0}
        sizes = measure.band_sizes(ranks)
        self.assertEqual(sizes[(0, 3)], 3)
        self.assertEqual(sum(sizes.values()), len(ranks))


class TestTheWeights(unittest.TestCase):
    """Why the numbers are weighted at all: the bands are sampled at different rates."""

    def test_a_labelled_frame_stands_for_its_whole_band(self):
        ranks = {file_id: file_id - 1 for file_id in range(1, 201)}  # 200 ranked frames
        labels = {1: True, 101: False}   # one from (0, 100), one from (100, 200)
        weights = measure.band_weights(ranks, labels)
        self.assertAlmostEqual(weights[1], 100.0)
        self.assertAlmostEqual(weights[101], 100.0)

    def test_a_denser_labelled_band_weighs_less_per_frame(self):
        ranks = {file_id: file_id - 1 for file_id in range(1, 201)}
        weights = measure.band_weights(ranks, {1: True, 2: True, 101: False})
        self.assertAlmostEqual(weights[1], 50.0)    # two labels for a hundred frames
        self.assertAlmostEqual(weights[101], 100.0)

    def test_a_labelled_frame_outside_the_ranking_gets_no_weight(self):
        """No stored vector, or reclassified since the worksheet was written: it takes
        part in nothing rather than being folded in at some invented rate."""
        weights = measure.band_weights({1: 0}, {1: True, 999: True})
        self.assertEqual(list(weights), [1])

    def test_without_labels_there_are_no_weights(self):
        self.assertEqual(measure.band_weights({1: 0, 2: 1}, {}), {})


class TestTheQueryTable(unittest.TestCase):
    """Tables 2 and 4: precision and recall of the query, as estimates of the collection."""

    def setUp(self):
        # Two labelled frames near the top standing for ten frames each, one far down
        # standing for a hundred: the shape that makes an unweighted average wrong.
        self.labels = {1: True, 2: False, 3: True}
        self.weights = {1: 10.0, 2: 10.0, 3: 100.0}
        self.name = measure.WORDINGS[0].name

    def rows(self, ranking, depths=(1, 2)):
        """The rows of the one wording ranked here — the other three get theirs too (and
        empty, see below), because the table always holds every wording."""
        return [row for row in measure.query_rows(
            {self.name: ranking}, self.labels, self.weights, depths)
            if row.wording == self.name]

    def test_precision_and_recall_are_weighted_by_the_design(self):
        first, second = self.rows([1, 2])
        self.assertAlmostEqual(first.estimate.precision, 1.0)     # frame 1 alone, correct
        self.assertAlmostEqual(first.estimate.recall, 10 / 110)   # of the whole class
        self.assertAlmostEqual(second.estimate.precision, 0.5)    # 1 right, 2 wrong
        self.assertAlmostEqual(second.estimate.recall, 10 / 110)

    def test_the_raw_counts_are_printed_next_to_the_estimate(self):
        (row,) = self.rows([1, 2], depths=(2,))
        self.assertEqual((row.estimate.labelled, row.estimate.correct), (2, 1))
        self.assertEqual(row.candidates, 2)

    def test_doubling_the_depth_is_the_lever_the_table_shows(self):
        low, high = self.rows([2, 3], depths=(1, 2))
        self.assertAlmostEqual(low.estimate.recall, 0.0)
        self.assertAlmostEqual(high.estimate.recall, 100 / 110)

    def test_an_unlabelled_frame_takes_no_part_in_either_number(self):
        (row,) = self.rows([1, 77], depths=(2,))
        self.assertEqual(row.candidates, 2)
        self.assertEqual(row.estimate.labelled, 1)
        self.assertAlmostEqual(row.estimate.precision, 1.0)

    def test_narrowing_the_pool_does_not_narrow_the_recall_denominator(self):
        """The whole question of table 4: does "no camera" concentrate the class or only
        lose it? A denominator that shrank with the pool could not answer it."""
        (everywhere,) = self.rows([1, 2, 3], depths=(3,))
        (gated,) = self.rows([1], depths=(3,))
        self.assertAlmostEqual(everywhere.estimate.recall, 1.0)
        self.assertAlmostEqual(gated.estimate.recall, 10 / 110)
        self.assertAlmostEqual(gated.estimate.precision, 1.0)

    def test_every_wording_gets_a_row_at_every_depth(self):
        rows = measure.query_rows({}, self.labels, self.weights, (1, 2))
        self.assertEqual(len(rows), 2 * len(measure.WORDINGS))
        self.assertTrue(all(r.candidates == 0 for r in rows))
        self.assertTrue(all(r.estimate.precision is None for r in rows))

    def test_an_empty_sample_leaves_the_numbers_unstated_rather_than_zero(self):
        row = measure.query_rows({self.name: [1]}, {}, {}, (1,))[0]
        self.assertIsNone(row.estimate.precision)
        self.assertIsNone(row.estimate.recall)


class TestTheBaseline(unittest.TestCase):
    """Table 5: today's classes and the metadata rules, on the very same frames."""

    def setUp(self):
        self.frames = [
            frame(1, camera=False, mp=0.5),                 # downloaded: no camera, small
            frame(2, camera=False, mp=12.0),                # a photograph sent by a friend
            frame(3, camera=True, mp=12.0),                 # a photograph, EXIF intact
            frame(4, camera=False, mp=None),                # downloaded, size unknown
        ]
        self.labels = {1: True, 2: False, 3: False, 4: True}
        self.weights = {file_id: 1.0 for file_id in self.labels}

    def rows(self):
        return {r.name: r for r in
                measure.rule_rows(self.frames, self.labels, self.weights)}

    def test_the_metadata_rule_is_measured_as_the_brief_states_it(self):
        row = self.rows()["нет камеры"]
        self.assertEqual((row.marked, row.estimate.correct), (3, 2))
        self.assertAlmostEqual(row.estimate.precision, 2 / 3)
        self.assertAlmostEqual(row.estimate.recall, 1.0)

    def test_a_size_rule_never_fires_on_a_frame_of_unknown_size(self):
        row = self.rows()["нет камеры и меньше 1 Мп"]
        self.assertEqual((row.marked, row.estimate.correct), (1, 1))
        self.assertAlmostEqual(row.estimate.recall, 0.5)  # frame 4 is lost by the rule

    def test_todays_classes_are_on_the_table_whatever_they_score(self):
        """Zero by construction here — only frames the stage calls `photo` are ranked —
        and the row is printed anyway: without it every number above reads as a gain."""
        row = self.rows()["нынешние классы (не «photo»)"]
        self.assertEqual((row.marked, row.estimate.correct), (0, 0))
        self.assertIsNone(row.estimate.precision)

    def test_a_reclassified_frame_is_what_that_row_counts(self):
        self.frames.append(frame(5, verdict="meme", camera=False, mp=0.2))
        self.labels[5] = True
        self.weights[5] = 1.0
        row = self.rows()["нынешние классы (не «photo»)"]
        self.assertEqual((row.marked, row.estimate.correct), (1, 1))

    def test_the_population_estimate_is_the_weighted_class(self):
        self.assertAlmostEqual(
            measure.population_estimate({1: 10.0, 2: 100.0}, {1: True, 2: False}), 10.0)


class TestTheWorksheet(unittest.TestCase):
    """File ids and nothing else, stratified over the pooled ranking — the privacy rule."""

    def ranks(self, total=6000):
        return {file_id: file_id - 1 for file_id in range(1, total + 1)}

    def test_it_holds_ids_and_nulls_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.json"
            written = measure.write_sample(path, self.ranks(), per_band=25, seed=1)
            data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(written, len(data))
        self.assertTrue(all(value is None for value in data.values()))
        self.assertTrue(all(key.isdigit() for key in data))

    def test_the_worksheet_is_the_size_the_brief_asks_for(self):
        """150-200 frames: thirteen is less than the fifteen animals on which the detector
        was wrong by twenty points."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.json"
            written = measure.write_sample(path, self.ranks(20000), per_band=25, seed=1)
        self.assertGreaterEqual(written, 150)
        self.assertLessEqual(written, 200)

    def test_every_band_of_the_ranking_is_represented(self):
        ranks = self.ranks()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.json"
            measure.write_sample(path, ranks, per_band=4, seed=1)
            picked = [int(key) for key in json.loads(path.read_text(encoding="utf-8"))]
        for low, high in measure.design_bands(ranks):
            with self.subTest(band=(low, high)):
                self.assertTrue(any(low <= ranks[file_id] < high for file_id in picked))

    def test_it_holds_no_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.json"
            measure.write_sample(path, {7: 0}, per_band=5, seed=1)
            text = path.read_text(encoding="utf-8")
        self.assertNotIn("photos", text)
        self.assertNotIn(".jpg", text)

    def test_a_worksheet_still_full_of_nulls_measures_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.json"
            path.write_text(json.dumps({"1": None, "2": True, "3": False}),
                            encoding="utf-8")
            labels = measure.read_labels(str(path))
        self.assertEqual(labels, {2: True, 3: False})

    def test_no_worksheet_is_no_labels(self):
        self.assertEqual(measure.read_labels(None), {})


class IndexCase(unittest.TestCase):
    """A small index, written by hand: files with their EXIF, sizes and verdicts."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = Path(self.tmp.name) / "m.db"
        self.conn = connect(self.db)
        self.addCleanup(self.conn.close)

    def add(self, name, verdict="photo", make=None, model=None, width=4000, height=3000,
            dup_of=None, error=None):
        cur = self.conn.execute(
            "INSERT INTO files (path, size, mtime, ext, media_type, camera_make,"
            " camera_model, width, height, dup_of, error, indexed_at)"
            " VALUES (?, 1, 0.0, 'jpg', 'photo', ?, ?, ?, ?, ?, ?, 'x')",
            (f"/photos/{name}", make, model, width, height, dup_of, error))
        file_id = cur.lastrowid
        if verdict is not None:
            self.conn.execute(
                "INSERT INTO media_class (file_id, verdict, source, updated_at, tier)"
                " VALUES (?, ?, 'clip', 'x', 'clip')", (file_id, verdict))
        self.conn.commit()
        return file_id


class TestReadingTheIndex(IndexCase):
    """What the script takes from the database — and that it may only take."""

    def test_the_database_is_opened_read_only(self):
        self.add("a.jpg")
        conn = measure.open_ro(str(self.db))
        self.addCleanup(conn.close)
        with self.assertRaises(Exception):
            conn.execute("DELETE FROM files")

    def test_the_frames_carry_their_metadata_and_their_verdict(self):
        self.add("a.jpg", make="Apple", model="iPhone 13 Pro", width=4032, height=3024)
        self.add("b.png", verdict="meme", width=800, height=600)
        conn = measure.open_ro(str(self.db))
        self.addCleanup(conn.close)
        photo, meme = measure.read_frames(conn)
        self.assertTrue(photo.has_camera)
        self.assertAlmostEqual(photo.megapixels, 4032 * 3024 / 1e6)
        self.assertEqual((meme.verdict, meme.has_camera), ("meme", False))

    def test_a_frame_the_stage_never_classified_has_no_verdict(self):
        self.add("c.jpg", verdict=None)
        conn = measure.open_ro(str(self.db))
        self.addCleanup(conn.close)
        (row,) = measure.read_frames(conn)
        self.assertIsNone(row.verdict)

    def test_duplicates_and_unreadable_files_are_not_the_population(self):
        canonical = self.add("a.jpg")
        self.add("a-copy.jpg", dup_of=canonical)
        self.add("broken.jpg", error="decode failed")
        conn = measure.open_ro(str(self.db))
        self.addCleanup(conn.close)
        self.assertEqual([f.file_id for f in measure.read_frames(conn)], [canonical])

    def test_only_frames_the_stage_calls_photographs_can_be_ranked(self):
        """`clip_embeddings` is purged of every other verdict on each run, so a query
        cannot reach a screenshot — which is exactly why the baseline row is zero."""
        frames = [frame(1), frame(2, verdict="screenshot"), frame(3, verdict=None)]
        self.assertEqual(measure.ranked_ids(frames), [1])


class TestTheTablesAreAnonymous(unittest.TestCase):
    """Privacy: the tables are aggregates. No path, no basename, no file id, ever."""

    def tables(self):
        labels = {1: True, 2: False}
        weights = {1: 10.0, 2: 10.0}
        name = measure.WORDINGS[0].name
        rows = measure.query_rows({name: [1, 2]}, labels, weights, (1, 2))
        return "\n".join([
            measure.format_population(measure.population_table(
                [frame(1, camera=False, mp=0.5), frame(2, verdict="meme")])),
            measure.format_query(rows, "ФОРМУЛИРОВКИ"),
            measure.format_gate(rows, rows),
            measure.format_baseline(
                measure.rule_rows([frame(1, camera=False, mp=0.5)], labels, weights), 570.0),
        ])

    def test_no_identifier_reaches_the_output(self):
        text = self.tables()
        for forbidden in ("/photos", ".jpg", "file_id"):
            self.assertNotIn(forbidden, text)

    def test_the_population_table_says_what_it_is_for(self):
        self.assertIn("ЧИТАТЬ ПЕРВОЙ", self.tables())

    def test_the_query_table_states_what_is_closed_and_what_is_the_lever(self):
        """The two things a reader must not have to rediscover: the ensemble was measured
        and gave nothing, and depth is the one confirmed lever of recall."""
        text = measure.format_query([], "ФОРМУЛИРОВКИ")
        self.assertIn("АНСАМБЛЬ ФОРМУЛИРОВОК ЗАКРЫТ", text)

    def test_the_gate_table_says_the_denominator_is_the_whole_class(self):
        """The claim table 4 exists to make: a selector that loses frames shows it."""
        text = measure.format_gate([], [])
        self.assertIn("от ВСЕГО класса", text)

    def test_the_baseline_states_the_estimated_population(self):
        text = measure.format_baseline([], 570.0)
        self.assertIn("~570", text)

    def test_the_baseline_explains_its_own_zero(self):
        self.assertIn("ПО ПОСТРОЕНИЮ", measure.format_baseline([], 0.0))


class TestItMeasuresThePipelinesOwnQuery(unittest.TestCase):
    """Nothing is reimplemented here: the ranking and the vectors are the stage's."""

    def test_the_ranking_is_the_stages_own(self):
        self.assertEqual(measure.detect.rank_candidates.__module__, "sorta.detect")
        self.assertEqual(measure.junk.read_clip_embeddings.__module__, "sorta.junk")

    def test_the_population_verdict_is_the_stages_own(self):
        self.assertEqual(measure.ranked_ids([frame(1, verdict=junk.QUALITY_VERDICT)]), [1])

    def test_the_wordings_are_the_four_the_brief_names(self):
        self.assertEqual(len(measure.WORDINGS), 4)
        self.assertEqual(len({w.name for w in measure.WORDINGS}), 4)
        self.assertEqual(len({w.prompt for w in measure.WORDINGS}), 4)

    def test_a_wording_is_one_prompt_and_never_an_ensemble(self):
        """Closed by measurement: an ensemble of wordings gives no effect, and it would
        hide which single phrasing the model actually answers."""
        for wording in measure.WORDINGS:
            with self.subTest(wording=wording.name):
                self.assertIsInstance(wording.prompt, str)

    def test_the_depths_double(self):
        """Depth is the one confirmed lever of recall, so the rows have to read as pairs."""
        for low, high in zip(measure.DEFAULT_DEPTHS, measure.DEFAULT_DEPTHS[1:]):
            self.assertEqual(high, 2 * low)

    def test_the_size_grid_is_the_one_the_brief_measured(self):
        self.assertEqual(measure.SMALL_MEGAPIXELS, (1.0, 3.0, 5.0))


if __name__ == "__main__":
    unittest.main()
