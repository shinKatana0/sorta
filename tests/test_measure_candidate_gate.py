"""F106: the candidate-gate curve — the arithmetic and the pre-registered verdict.

The script answers one question with numbers: what would raising
`naming.product_candidate_min` cost? The brief fixed the shape of the answer before the
first run — two criteria, both required, and a named outcome for "no" — precisely so the
curve cannot talk anybody into a threshold afterwards. That is what is tested here:
given frames, the sweep must count what a human counting by hand would count, and the
recommendation must be the one the criteria give, including when it has to be C.

No model, no GPU, no photo: every function below is arithmetic over per-frame
aggregates. And, as with every measurement in this project, nothing the script prints may
identify a frame.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "measure_candidate_gate.py"


def _load_script():
    """Import scripts/measure_candidate_gate.py — it is a script, not a package module."""
    spec = importlib.util.spec_from_file_location("measure_candidate_gate", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gate = _load_script()

PHOTO = "photo"
PRODUCT = "product"
DOC = "document"


def frame(score, before=PHOTO, after=PHOTO, forced=False):
    """One measured frame: its product score and what the tier did to its verdict."""
    return gate.Frame(product_score=score, forced=forced, before=before, after=after)


def changed(score, before=PHOTO, after=PRODUCT, forced=False):
    """A frame the deep tier was useful on — the benefit the curve is about."""
    return frame(score, before=before, after=after, forced=forced)


def row_at(rows, threshold):
    return next(r for r in rows if abs(r.threshold - threshold) < 1e-9)


class TestCurveCounters(unittest.TestCase):
    """Test 1 of the brief: the counters per threshold against numbers computed by hand."""

    def setUp(self):
        # Five frames, scores 0.30 / 0.45 / 0.50 / 0.70 / 0.90; the tier changed the
        # verdict of the 0.45, the 0.70 and the 0.90 one.
        self.frames = [
            frame(0.30),
            changed(0.45),
            frame(0.50),
            changed(0.70),
            changed(0.90, after=DOC),
        ]
        self.rows = gate.sweep(self.frames, [0.4, 0.6, 0.8])

    def test_candidates_are_the_frames_at_or_above_the_threshold(self):
        self.assertEqual([r.candidates for r in self.rows], [4, 2, 1])

    def test_the_threshold_is_inclusive(self):
        """>= , exactly as junk.classify gates it — 0.45 is a candidate at 0.45."""
        self.assertEqual(gate.sweep([frame(0.45)], [0.45])[0].candidates, 1)

    def test_benefit_kept_and_lost_split_the_changed_verdicts(self):
        self.assertEqual([(r.kept, r.lost) for r in self.rows], [(3, 0), (2, 1), (1, 2)])
        for r in self.rows:
            self.assertEqual(r.changed_total, 3)

    def test_the_time_column_is_the_measured_seconds_per_frame(self):
        self.assertAlmostEqual(row_at(self.rows, 0.6).seconds, 2 * gate.SEC_PER_FRAME)

    def test_the_population_cut_is_measured_against_the_current_threshold(self):
        base = gate.baseline_row(self.rows, 0.4)
        self.assertEqual(base.threshold, 0.4)
        self.assertAlmostEqual(gate.population_cut(base, row_at(self.rows, 0.6)), 0.5)
        self.assertAlmostEqual(gate.benefit_kept(base, row_at(self.rows, 0.8)), 1 / 3)


class TestGateIsMoreThanTheProductScore(unittest.TestCase):
    """A frame the document branch pulls in stays a candidate at every threshold.

    Counting it as lost would invent a loss that raising the knob never causes.
    """

    def test_a_forced_frame_survives_any_threshold(self):
        rows = gate.sweep([changed(0.05, forced=True)], [0.4, 0.9])
        self.assertEqual([(r.candidates, r.kept, r.lost) for r in rows], [(1, 1, 0)] * 2)

    def test_a_frame_the_gate_never_looks_at_is_a_candidate_nowhere(self):
        """Faces / screenshot / meme: in the collection, in no population."""
        rows = gate.sweep([frame(None), changed(0.9)], [0.4])
        self.assertEqual((rows[0].candidates, rows[0].kept), (1, 1))


class TestEmptyAndFullPopulations(unittest.TestCase):
    """Test 2 of the brief: the ends of the curve."""

    def setUp(self):
        self.frames = [changed(0.3), frame(0.5), changed(0.9)]

    def test_a_threshold_above_every_score_gates_nobody(self):
        row = gate.sweep(self.frames, [1.01])[0]
        self.assertEqual((row.candidates, row.kept, row.lost), (0, 0, 2))
        self.assertEqual(row.seconds, 0.0)

    def test_a_threshold_below_every_score_gates_everybody(self):
        row = gate.sweep(self.frames, [0.0])[0]
        self.assertEqual((row.candidates, row.kept, row.lost), (3, 2, 0))

    def test_an_empty_benefit_is_not_a_division_by_zero(self):
        row = gate.sweep([frame(0.5)], [0.9])[0]
        self.assertEqual(row.kept_frac, 1.0)
        self.assertEqual(gate.benefit_kept(row, row), 1.0)

    def test_an_empty_population_is_not_a_division_by_zero(self):
        base = gate.sweep([frame(0.1)], [0.9])[0]
        self.assertEqual(gate.population_cut(base, base), 0.0)


class TestFramesWithoutABaseline(unittest.TestCase):
    """Test 3 of the brief: a frame indexed after the snapshot was taken.

    Whether the tier changed anything for it is not knowable, so it is counted apart
    instead of being guessed in either direction.
    """

    def setUp(self):
        self.frames = [frame(0.9, before=None, after=PRODUCT), changed(0.9), frame(0.9)]
        self.row = gate.sweep(self.frames, [0.4])[0]

    def test_it_still_counts_as_a_candidate(self):
        self.assertEqual(self.row.candidates, 3)

    def test_it_is_reported_on_its_own_line(self):
        self.assertEqual(self.row.unknown, 1)
        self.assertIn("нет базы для сравнения",
                      gate.format_table([self.row], total=3, current=0.4))

    def test_it_is_not_counted_as_benefit(self):
        self.assertEqual((self.row.kept, self.row.lost), (1, 0))

    def test_it_is_not_counted_as_a_loss_when_the_threshold_drops_it(self):
        row = gate.sweep(self.frames, [0.95])[0]
        self.assertEqual((row.candidates, row.kept, row.lost, row.unknown), (0, 0, 1, 0))


class TestUnchangedVerdictsAreNotBenefit(unittest.TestCase):
    """Test 4 of the brief: the model answering is not the model being useful."""

    def test_a_frame_the_tier_confirmed_is_not_benefit(self):
        rows = gate.sweep([frame(0.9, before=PHOTO, after=PHOTO)], [0.4, 0.95])
        self.assertEqual([(r.candidates, r.kept, r.lost) for r in rows], [(1, 0, 0), (0, 0, 0)])

    def test_dropping_it_costs_nothing_but_still_saves_the_time(self):
        rows = gate.sweep([frame(0.9)] * 10, [0.4, 0.95])
        base, higher = rows
        self.assertEqual(gate.benefit_kept(base, higher), 1.0)
        self.assertEqual(gate.population_cut(base, higher), 1.0)


class TestLossBreakdown(unittest.TestCase):
    """Test 5 of the brief: what exactly is given up, per label pair."""

    def setUp(self):
        self.frames = [
            changed(0.5, PHOTO, PRODUCT), changed(0.5, PHOTO, PRODUCT),
            changed(0.5, PHOTO, DOC),
            changed(0.5, DOC, PHOTO),
            changed(0.9, PHOTO, PRODUCT),   # survives the sweep below
        ]
        self.row = gate.sweep(self.frames, [0.7])[0]

    def test_the_pairs_sum_to_the_total_loss(self):
        self.assertEqual(sum(self.row.lost_pairs.values()), self.row.lost)
        self.assertEqual(self.row.lost, 4)

    def test_every_pair_is_counted_with_its_direction(self):
        self.assertEqual(dict(self.row.lost_pairs), {
            (PHOTO, PRODUCT): 2, (PHOTO, DOC): 1, (DOC, PHOTO): 1})

    def test_only_a_found_document_counts_as_a_lost_document(self):
        """`photo -> document` is papers in a city folder; `document -> photo` is not."""
        self.assertEqual(self.row.documents_lost, 1)

    def test_the_report_names_the_pairs(self):
        report = gate.format_losses(gate.sweep(self.frames, [0.4, 0.7]), current=0.4)
        self.assertIn("photo -> product: 2", report)
        self.assertIn("document -> photo: 1", report)

    def test_the_threshold_of_the_first_lost_document_is_reported(self):
        rows = gate.sweep(self.frames, [0.4, 0.6, 0.7])
        hit = gate.first_document_loss(rows, gate.baseline_row(rows, 0.4))
        self.assertEqual(hit.threshold, 0.6)
        self.assertIn("первый потерянный документ: порог 0.60",
                      gate.format_losses(rows, current=0.4))

    def test_a_grid_that_loses_no_document_says_so(self):
        rows = gate.sweep([changed(0.5, PHOTO, PRODUCT)], [0.4, 0.9])
        self.assertIsNone(gate.first_document_loss(rows, gate.baseline_row(rows, 0.4)))
        self.assertIn("ни на одном пороге", gate.format_losses(rows, current=0.4))


class TestPreRegisteredCriteria(unittest.TestCase):
    """The numbers themselves — written down before the run, and they must stay put."""

    def test_the_criteria_are_the_ones_the_brief_registered(self):
        self.assertEqual(gate.MIN_BENEFIT_KEPT, 0.95)
        self.assertEqual(gate.MIN_POPULATION_CUT, 0.25)

    def test_the_time_per_frame_is_the_measured_one(self):
        self.assertEqual(gate.SEC_PER_FRAME, 0.78)

    def test_the_grid_contains_the_threshold_in_force(self):
        self.assertIn(0.4, gate.DEFAULT_GRID)


class TestRecommendation(unittest.TestCase):
    """Test 6 of the brief: a threshold when both criteria are met, outcome C when not."""

    def collection(self, useful_high, useful_low, useless_low):
        """Frames split by where the benefit sits relative to a 0.6 threshold."""
        return ([changed(0.9)] * useful_high + [changed(0.45)] * useful_low
                + [frame(0.45)] * useless_low)

    def test_a_threshold_that_meets_both_criteria_is_recommended(self):
        # 100 changed verdicts, 98 of them above 0.6; 300 candidates at 0.4, 98 at 0.6.
        rows = gate.sweep(self.collection(98, 2, 200), [0.4, 0.6])
        letter, why = gate.recommend(rows, current=0.4)
        self.assertEqual(letter, "A")
        self.assertIn("0.60", why)
        self.assertIn("naming.product_candidate_min", why)

    def test_losing_too_much_benefit_is_outcome_c(self):
        # the population halves, but only 90% of the changed verdicts survive
        rows = gate.sweep(self.collection(90, 10, 100), [0.4, 0.6])
        letter, why = gate.recommend(rows, current=0.4)
        self.assertEqual(letter, "C")
        self.assertIn("гейт настроен хорошо", why)

    def test_a_population_that_barely_moves_is_outcome_c(self):
        # every verdict survives, but the gate only sheds 10% of its candidates
        rows = gate.sweep([changed(0.9)] * 90 + [frame(0.45)] * 10, [0.4, 0.6])
        letter, why = gate.recommend(rows, current=0.4)
        self.assertEqual(letter, "C")
        self.assertIn("популяция сокращается", why)

    def test_the_biggest_cut_among_the_acceptable_thresholds_wins(self):
        frames = [changed(0.9)] * 96 + [changed(0.5)] * 4 + [frame(0.45)] * 300
        letter, why = gate.recommend(gate.sweep(frames, [0.4, 0.5, 0.8]), current=0.4)
        self.assertEqual(letter, "A")
        self.assertIn("0.80", why)

    def test_a_grid_with_nothing_above_the_current_threshold_is_outcome_c(self):
        letter, why = gate.recommend(gate.sweep([changed(0.9)], [0.3, 0.4]), current=0.4)
        self.assertEqual(letter, "C")
        self.assertIn("сравнивать не с чем", why)

    def test_the_outcome_line_carries_the_letter(self):
        rows = gate.sweep(self.collection(98, 2, 200), [0.4, 0.6])
        self.assertTrue(gate.format_outcome(rows, current=0.4).startswith("ИСХОД A"))

    def test_a_grid_without_the_configured_value_still_has_a_baseline(self):
        """The nearest row is the baseline — a grid off by 0.01 must not crash."""
        rows = gate.sweep([changed(0.9)], [0.41, 0.7])
        self.assertEqual(gate.baseline_row(rows, 0.4).threshold, 0.41)


class TestBeforeSnapshot(unittest.TestCase):
    """The snapshot is produced outside this repo, so its shapes are what they are."""

    def load(self, payload):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "before.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            return gate.load_before(path)

    def test_a_list_of_rows(self):
        self.assertEqual(
            self.load([{"file_id": 1, "verdict": PHOTO}, {"file_id": 2, "verdict": DOC}]),
            {1: PHOTO, 2: DOC})

    def test_a_list_of_pairs(self):
        self.assertEqual(self.load([[1, PHOTO], [2, DOC]]), {1: PHOTO, 2: DOC})

    def test_a_mapping_of_id_to_verdict(self):
        self.assertEqual(self.load({"1": PHOTO, "2": DOC}), {1: PHOTO, 2: DOC})

    def test_rows_under_a_key(self):
        self.assertEqual(self.load({"rows": [{"file_id": 7, "verdict": DOC}]}), {7: DOC})

    def test_an_empty_snapshot_is_an_error_not_an_empty_baseline(self):
        with self.assertRaises(SystemExit):
            self.load([])

    def test_something_that_is_not_a_snapshot_is_an_error(self):
        with self.assertRaises(SystemExit):
            self.load("media_class")


class TestThresholdGrid(unittest.TestCase):
    def test_a_comma_separated_grid_is_parsed_sorted_and_deduplicated(self):
        self.assertEqual(gate.parse_thresholds("0.5,0.3,0.4,0.3"), [0.3, 0.4, 0.5])

    def test_spaces_are_allowed_too(self):
        self.assertEqual(gate.parse_thresholds("0.3 0.4"), [0.3, 0.4])

    def test_an_empty_grid_is_an_error(self):
        with self.assertRaises(SystemExit):
            gate.parse_thresholds("  ")


class TestReportIdentifiesNothing(unittest.TestCase):
    """Privacy: a table about documents must not become a list of where they are."""

    def test_no_frame_identity_reaches_the_output(self):
        frames = [changed(0.5, PHOTO, DOC), changed(0.9), frame(0.45)]
        rows = gate.sweep(frames, [0.4, 0.7])
        text = "\n".join([gate.format_table(rows, len(frames), 0.4),
                          gate.format_losses(rows, 0.4),
                          gate.format_outcome(rows, 0.4)])
        for leak in ("/photos", ".jpg", "file_id", "IMG_", "\\"):
            self.assertNotIn(leak, text)


if __name__ == "__main__":
    unittest.main()
