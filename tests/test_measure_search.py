"""F129: the search measurement — the arithmetic, and the privacy of what it prints.

The script exists because the accuracy of a CLIP search on this collection has never been
measured, and F121/F122 is what happens when a number is assumed instead: a class looked
like it worked until the labels arrived. So what is tested here is that the tables cannot
flatter the feature — the bands have to add up to the collection, an unlabelled frame must
not be counted as a miss, a thin sample has to say so out loud — and that a report about
someone's photographs does not print where they are unless it was asked to.

No model, no GPU, no photo: everything below is arithmetic over (file_id, score) pairs.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "measure_search.py"


def _load_script():
    """Import scripts/measure_search.py — it is a script, not a package module."""
    spec = importlib.util.spec_from_file_location("measure_search", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


measure = _load_script()


def result(query: str = "cake", hits=((1, 0.33), (2, 0.29), (3, 0.21)),
           scores=None):
    ranked = [(int(fid), float(score)) for fid, score in hits]
    return measure.Result(query=query, hits=ranked,
                          scores=[s for _f, s in ranked] if scores is None
                          else list(scores))


class TestBands(unittest.TestCase):
    def test_a_score_falls_into_exactly_one_band(self):
        low, high = measure.band_of(0.285)
        self.assertLessEqual(low, 0.285)
        self.assertLess(0.285, high)

    def test_the_lowest_band_swallows_everything_below_the_grid(self):
        self.assertEqual(measure.band_of(-1.0), measure.band_of(0.0))

    def test_the_bands_account_for_every_frame(self):
        scores = [0.05, 0.21, 0.215, 0.27, 0.31, 0.36, 0.99]
        counts = measure.band_counts(scores)
        self.assertEqual(sum(count for _low, _high, count in counts), len(scores))

    def test_an_empty_collection_gives_empty_bands_rather_than_a_crash(self):
        self.assertEqual(sum(c for _l, _h, c in measure.band_counts([])), 0)


class TestPrecision(unittest.TestCase):
    def test_precision_is_computed_over_the_labelled_frames_of_the_prefix(self):
        hits = [(i, 0.3 - 0.001 * i) for i in range(1, 11)]
        labels = {1: True, 2: True, 3: False, 4: True}
        rows = {depth: (labelled, correct, precision)
                for depth, labelled, correct, precision in measure.precision_at(
                    hits, labels, depths=(2, 4, 10))}
        self.assertEqual(rows[2], (2, 2, 1.0))
        self.assertEqual(rows[4], (4, 3, 0.75))
        self.assertEqual(rows[10], (4, 3, 0.75))  # the unlabelled six are not misses

    def test_a_depth_with_nothing_labelled_is_empty_and_not_zero_accuracy(self):
        rows = measure.precision_at([(1, 0.3)], {}, depths=(1,))
        self.assertEqual(rows, [(1, 0, 0, 0.0)])

    def test_precision_by_band_splits_the_marks_by_score(self):
        hits = [(1, 0.33), (2, 0.31), (3, 0.21)]
        rows = measure.precision_by_band(hits, {1: True, 2: False, 3: False})
        by_band = {(low, high): (labelled, correct, precision)
                   for low, high, labelled, correct, precision in rows}
        self.assertEqual(by_band[measure.band_of(0.33)], (1, 1, 1.0))
        self.assertEqual(by_band[measure.band_of(0.31)], (1, 0, 0.0))
        self.assertEqual(by_band[measure.band_of(0.21)], (1, 0, 0.0))

    def test_a_band_nobody_labelled_is_left_out_instead_of_reported_as_zero(self):
        rows = measure.precision_by_band([(1, 0.33), (2, 0.21)], {1: True})
        self.assertEqual([(low, high) for low, high, *_rest in rows],
                         [measure.band_of(0.33)])


class TestTheTablesReadHonestly(unittest.TestCase):
    def test_the_top_table_prints_ids_and_no_path_by_default(self):
        printed = measure.format_top(result(), None)
        self.assertIn("file_id", printed)
        self.assertNotIn("secret", printed)
        self.assertIn("0.330", printed)

    def test_paths_are_printed_only_when_asked_for(self):
        printed = measure.format_top(result(), {1: "C:/photos/secret.jpg"})
        self.assertIn("C:/photos/secret.jpg", printed)

    def test_the_band_table_shows_the_share_of_the_collection(self):
        printed = measure.format_bands(
            result(scores=[0.33, 0.29, 0.21] + [0.1] * 97))
        self.assertIn("«cake»", printed)
        self.assertIn("97", printed)  # the pile the top was drawn from

    def test_a_thin_sample_is_flagged_before_its_precision_is_read(self):
        printed = measure.format_precision(result(), {1: True})
        self.assertIn(str(measure.MIN_LABELS), printed)
        self.assertIn("ВНИМАНИЕ", printed)

    def test_a_full_sample_is_not_flagged(self):
        hits = [(i, 0.3) for i in range(1, 21)]
        printed = measure.format_precision(
            result(hits=hits), {i: True for i in range(1, 21)})
        self.assertNotIn("ВНИМАНИЕ", printed)


class TestTheWorksheet(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_template_holds_file_ids_and_nulls_only(self):
        path = self.root / "marks.json"
        written = measure.write_label_template(path, [result(), result(query="snow")])
        self.assertEqual(written, 6)
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(set(data), {"cake", "snow"})
        self.assertEqual(set(data["cake"].values()), {None})
        self.assertEqual(set(data["cake"]), {"1", "2", "3"})

    def test_a_partially_filled_sheet_drops_the_unanswered_frames(self):
        path = self.root / "marks.json"
        path.write_text(json.dumps({"cake": {"1": True, "2": None, "3": False}}),
                        encoding="utf-8")
        self.assertEqual(measure.load_labels(path), {"cake": {1: True, 3: False}})


class TestTheQueryList(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def args(self, queries=None, queries_file=None):
        return type("Args", (), {"queries": queries, "queries_file": queries_file})()

    def test_queries_come_from_the_command_line_and_the_file_without_repeats(self):
        path = self.root / "queries.txt"
        path.write_text("# a comment\nsnow\ncake\n\n", encoding="utf-8")
        self.assertEqual(
            measure.read_queries(self.args(["cake"], str(path))), ["cake", "snow"])

    def test_no_queries_at_all_is_an_empty_list_rather_than_a_crash(self):
        self.assertEqual(measure.read_queries(self.args()), [])


if __name__ == "__main__":
    unittest.main()
