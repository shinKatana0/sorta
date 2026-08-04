"""F129: the search measurement — the arithmetic, and the privacy of what it prints.

The script exists because the accuracy of a CLIP search on this collection has never been
measured, and F121/F122 is what happens when a number is assumed instead: a class looked
like it worked until the labels arrived. So what is tested here is that the tables cannot
flatter the feature — the bands have to add up to the collection, an unlabelled frame must
not be counted as a miss, a thin sample has to say so out loud — and that a report about
someone's photographs does not print where they are unless it was asked to.

F153 adds the other half of the numbers and the comparison that needs them. Precision was
never going to decide whether merging the two indexes is worth anything — both models are
at 98% at top-5 already — so what is tested here is that RECALL is computed against a
stated denominator (the marked frames, not the collection), that an unlabelled frame
counts as not relevant in it while precision still ignores it, and that a table which
cannot support a conclusion says so instead of printing one.

No model, no GPU, no photo: everything below is arithmetic over (file_id, score) pairs.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

from sorta import search

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


class TestRecall(unittest.TestCase):
    """F153: the half nobody measured — and the denominator it is honest about."""

    def test_recall_counts_the_marked_frames_the_prefix_reached(self):
        hits = [(i, 0.3 - 0.001 * i) for i in range(1, 11)]
        labels = {1: True, 3: False, 5: True, 99: True}  # three positives in the pool
        rows = {depth: (found, pool, recall)
                for depth, found, pool, recall in measure.recall_at(
                    hits, labels, depths=(2, 10))}
        self.assertEqual(rows[2], (1, 3, 1 / 3))
        self.assertEqual(rows[10], (2, 3, 2 / 3))

    def test_an_unlabelled_frame_is_not_relevant_here_although_precision_ignores_it(self):
        # The two rules differ on purpose: precision asks what a person checked, recall
        # asks what the variant surfaced. Counting an unjudged frame as relevant would let
        # a variant raise its recall by returning frames nobody has looked at.
        hits = [(1, 0.3), (2, 0.3)]
        labels = {1: True, 7: True}
        self.assertEqual(measure.recall_at(hits, labels, depths=(2,)), [(2, 1, 2, 0.5)])
        self.assertEqual(measure.precision_at(hits, labels, depths=(2,)), [(2, 1, 1, 1.0)])

    def test_a_query_with_no_positive_marks_is_an_empty_pool_and_not_an_accuracy(self):
        self.assertEqual(measure.recall_at([(1, 0.3)], {1: False}, depths=(1,)),
                         [(1, 0, 0, 0.0)])

    def test_the_recall_table_states_that_the_denominator_is_the_marks(self):
        printed = measure.format_recall(result(), {1: True, 2: True, 3: False})
        self.assertIn("ПОЛНОТА", printed)
        self.assertIn("не вся коллекция", printed)

    def test_a_thin_pool_is_flagged_before_its_recall_is_read(self):
        self.assertIn("ВНИМАНИЕ", measure.format_recall(result(), {1: True}))


class TestTheVariantComparison(unittest.TestCase):
    """F153: L14, XLM and both merges in one table, over one set of marks."""

    def variants(self):
        # XLM has the two relevant frames deeper down; L14 has one of them first. A merge
        # is supposed to be able to beat both, which is exactly what needs measuring.
        return measure.with_fusions({
            measure.VARIANT_SEARCH: [(3, 0.31), (1, 0.30), (2, 0.29)],
            measure.VARIANT_CLASS: [(2, 0.25), (4, 0.24), (1, 0.23)],
        })

    def test_all_four_variants_are_measured(self):
        variants = self.variants()
        self.assertEqual(set(variants), {measure.VARIANT_SEARCH, measure.VARIANT_CLASS,
                                         *measure.FUSION_VARIANTS})

    def test_the_merges_come_from_the_features_own_function(self):
        # Not a copy of the arithmetic living in the script: a private reimplementation
        # would measure this file instead of the feature.
        variants = self.variants()
        for mode in measure.FUSION_VARIANTS:
            with self.subTest(mode=mode):
                self.assertEqual(
                    variants[mode],
                    search.fuse([[3, 1, 2], [2, 4, 1]], mode, measure.FUSION_DEPTH))

    def test_an_index_that_could_not_rank_leaves_the_others_measurable(self):
        variants = measure.with_fusions({measure.VARIANT_SEARCH: [(1, 0.3), (2, 0.2)]})
        self.assertEqual([fid for fid, _w in variants[measure.SEARCH_FUSION_RANK]], [1, 2])
        self.assertEqual(measure.with_fusions({})[measure.SEARCH_FUSION_UNION], [])

    def test_every_variant_gets_a_cell_per_depth_from_the_same_marks(self):
        scores = measure.compare(self.variants(), {1: True, 2: True, 3: False},
                                 depths=(1, 3))
        self.assertEqual(len(scores), 4 * 2)
        by_key = {(s.variant, s.depth): s for s in scores}
        xlm = by_key[(measure.VARIANT_SEARCH, 3)]
        self.assertEqual((xlm.labelled, xlm.correct), (3, 2))
        self.assertEqual((xlm.found, xlm.pool), (2, 2))
        self.assertEqual(xlm.recall, 1.0)
        self.assertEqual(by_key[(measure.VARIANT_CLASS, 1)].found, 1)

    def test_the_unjudged_part_of_a_prefix_is_counted_and_not_hidden(self):
        scores = measure.compare({measure.VARIANT_SEARCH: [(1, 0.3), (9, 0.2)]},
                                 {1: True}, depths=(2,))
        self.assertEqual((scores[0].labelled, scores[0].unlabelled), (1, 1))

    def test_the_table_prints_a_line_per_variant_and_depth(self):
        printed = measure.format_comparison(
            "cake", measure.compare(self.variants(), {1: True, 2: True, 3: False},
                                    depths=(3,)))
        self.assertIn("«cake»", printed)
        for variant in (measure.VARIANT_SEARCH, measure.VARIANT_CLASS,
                        *measure.FUSION_VARIANTS):
            self.assertIn(variant, printed)
        self.assertIn("ОТНОСИТЕЛЬНАЯ", printed)

    def test_a_thin_pool_of_positives_is_flagged_in_the_comparison_too(self):
        printed = measure.format_comparison(
            "cake", measure.compare(self.variants(), {1: True}, depths=(3,)))
        self.assertIn("ВНИМАНИЕ", printed)
        self.assertIn(str(measure.MIN_LABELS), printed)

    def test_a_mostly_unlabelled_output_says_the_marks_are_not_enough(self):
        printed = measure.format_comparison(
            "cake", measure.compare(
                {measure.VARIANT_SEARCH: [(i, 0.3) for i in range(1, 21)]},
                {i: True for i in range(1, 6)}, depths=(20,)))
        self.assertIn("доразметьте", printed)


class TestTheWorksheet(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_template_holds_file_ids_and_nulls_only(self):
        path = self.root / "marks.json"
        written = measure.write_label_template(
            path, measure.top_ids([result(), result(query="snow")]))
        self.assertEqual(written, 6)
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(set(data), {"cake", "snow"})
        self.assertEqual(set(data["cake"].values()), {None})
        self.assertEqual(set(data["cake"]), {"1", "2", "3"})

    def test_a_fusion_worksheet_covers_the_top_of_every_variant(self):
        # The frames only the merge surfaced are the ones the comparison stands on, and
        # they are unlabelled by construction — a sheet holding one variant's top would
        # measure the merge against a sample its competitor chose.
        ids = measure.merged_ids({"XLM": [(1, 0.3), (2, 0.2)],
                                  "rank": [(2, 0.02), (5, 0.01)]}, top=2)
        self.assertEqual(ids, [1, 2, 5])
        path = self.root / "marks.json"
        self.assertEqual(measure.write_label_template(path, {"cake": ids}), 3)
        self.assertEqual(set(json.loads(path.read_text(encoding="utf-8"))["cake"]),
                         {"1", "2", "5"})

    def test_the_worksheet_only_holds_the_top_asked_for(self):
        self.assertEqual(measure.merged_ids({"XLM": [(1, 0.3), (2, 0.2)]}, top=1), [1])

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
