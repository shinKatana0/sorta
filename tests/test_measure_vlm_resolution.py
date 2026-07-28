"""F102: the resolution measurement — the arithmetic and the pre-registered verdict.

The script exists to answer one question with numbers: may the default input size of the
deep tier be lowered? The brief fixed the answer's shape before the first run — two
criteria, both required, and three outcomes — precisely so that the table cannot talk
anybody into a number afterwards. That is what is tested here: given results, the script
must reach the outcome the brief says it must, including the cases where it has to say
no.

No model, no GPU, no photo: every function below is arithmetic over per-frame
aggregates. And, as with every measurement in this project, nothing the script prints
may identify a frame.
"""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "measure_vlm_resolution.py"


def _load_script():
    """Import scripts/measure_vlm_resolution.py — it is a script, not a package module."""
    spec = importlib.util.spec_from_file_location("measure_vlm_resolution", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


res = _load_script()

DOC = "document"
PHOTO = "personal_photo"
PRODUCT = "product"


def edge(max_edge, labels, wall=1.0, frame_ms=(10.0,)):
    return res.EdgeResult(max_edge=max_edge, labels=tuple(labels),
                          frame_ms=tuple(frame_ms), wall_sec=wall, peak_vram_mb=23100.0)


def sample(n, documents=0, label=PHOTO):
    """`n` labels of which `documents` are documents — a baseline of a given shape."""
    return [DOC] * documents + [label] * (n - documents)


def faster(base_labels, other_labels, times):
    """A baseline and a candidate that ran `times` faster over the same frames."""
    return [edge(896, base_labels, wall=times),
            edge(448, other_labels, wall=1.0)]


class TestAgreement(unittest.TestCase):
    def test_identical_passes_agree_completely(self):
        labels = [DOC, PHOTO, PRODUCT]
        self.assertEqual(res.agreement(edge(896, labels), edge(448, labels)), 1.0)

    def test_one_moved_label_in_four(self):
        self.assertEqual(
            res.agreement(edge(896, [DOC] * 4), edge(448, [DOC, DOC, DOC, PHOTO])), 0.75)

    def test_a_missing_frame_counts_against_agreement(self):
        """Fewer answers is not the same as agreeing — a crashed pass must not win."""
        self.assertEqual(
            res.agreement(edge(896, [DOC, DOC]), edge(448, [DOC])), 0.5)

    def test_an_empty_pass_is_not_a_division_by_zero(self):
        self.assertEqual(res.agreement(edge(896, []), edge(448, [])), 0.0)


class TestDocumentLoss(unittest.TestCase):
    """The directional criterion: a document read as a photo is the failure that matters."""

    def test_documents_that_became_photos_are_counted(self):
        base = edge(896, [DOC, DOC, DOC, PHOTO])
        other = edge(448, [DOC, PHOTO, DOC, PHOTO])
        self.assertEqual(res.document_loss(base, other), (1, 3))

    def test_a_document_read_as_a_product_is_not_counted_as_loss(self):
        """It is a disagreement — `agreement` has it — but not a photo album leak."""
        base = edge(896, [DOC, DOC])
        self.assertEqual(res.document_loss(base, edge(448, [DOC, PRODUCT])), (0, 2))

    def test_the_reverse_direction_is_not_a_loss(self):
        base = edge(896, [PHOTO, PHOTO])
        self.assertEqual(res.document_loss(base, edge(448, [DOC, DOC])), (0, 0))

    def test_a_sample_without_documents_has_no_loss_fraction(self):
        check = res.assess(edge(896, [PHOTO] * 4), edge(448, [PHOTO] * 4))
        self.assertEqual(check.document_loss_frac, 0.0)


class TestSpeedup(unittest.TestCase):
    def test_speedup_is_frames_per_second_against_the_baseline(self):
        base, other = faster([PHOTO] * 10, [PHOTO] * 10, times=2.0)
        self.assertAlmostEqual(res.speedup(base, other), 2.0)

    def test_a_baseline_that_measured_nothing_is_not_a_division_by_zero(self):
        self.assertEqual(res.speedup(edge(896, [], wall=0.0), edge(448, [PHOTO])), 0.0)


class TestPreRegisteredThresholds(unittest.TestCase):
    """The numbers themselves — written down before the run, and they must stay put."""

    def test_the_criteria_are_the_ones_the_brief_registered(self):
        self.assertEqual(res.MIN_SAMPLE, 300)
        self.assertEqual(res.MIN_AGREEMENT, 0.98)
        self.assertEqual(res.MAX_DOCUMENT_LOSS, 0.02)
        self.assertEqual(res.MIN_SPEEDUP, 1.40)

    def test_the_baseline_is_the_resolution_that_shipped(self):
        self.assertEqual(res.DEFAULT_EDGES[0], 896)


class TestOutcome(unittest.TestCase):
    """The verdict: A only when both criteria are met, B and C for the ways they are not."""

    def outcome_of(self, base_labels, other_labels, times):
        return res.outcome(faster(base_labels, other_labels, times))

    def test_a_fast_and_faithful_pass_is_outcome_a(self):
        labels = sample(300, documents=50)
        letter, why = self.outcome_of(labels, labels, times=1.6)
        self.assertEqual(letter, "A")
        self.assertIn("448", why)
        self.assertIn("меняем дефолт", why)

    def test_a_slow_pass_is_outcome_c_even_when_every_verdict_matches(self):
        labels = sample(300, documents=50)
        letter, why = self.outcome_of(labels, labels, times=1.2)
        self.assertEqual(letter, "C")
        self.assertIn("закрыт", why)

    def test_fast_but_disagreeing_is_outcome_b(self):
        base = sample(300, documents=50)
        other = list(base)
        for i in range(50, 70):  # 20 photos of 300 read differently — 93.3% agreement
            other[i] = PRODUCT
        letter, why = self.outcome_of(base, other, times=2.0)
        self.assertEqual(letter, "B")
        self.assertIn("согласие", why)

    def test_losing_documents_is_outcome_b_even_at_high_overall_agreement(self):
        """The criterion the brief added on purpose: an average hides a class."""
        base = sample(300, documents=50)
        other = list(base)
        for i in range(3):  # 3 of 50 documents = 6% > 2%, while agreement is 99%
            other[i] = PHOTO
        letter, why = self.outcome_of(base, other, times=2.0)
        self.assertGreaterEqual(res.agreement(*faster(base, other, 2.0)), res.MIN_AGREEMENT)
        self.assertEqual(letter, "B")
        self.assertIn("документов потеряно", why)

    def test_a_sample_below_the_registered_minimum_cannot_reach_a(self):
        labels = sample(299, documents=50)
        letter, why = self.outcome_of(labels, labels, times=2.0)
        self.assertEqual(letter, "B")
        self.assertIn("299", why)

    def test_the_fastest_acceptable_resolution_is_the_one_picked(self):
        labels = sample(300, documents=50)
        results = [edge(896, labels, wall=2.0), edge(672, labels, wall=1.2),
                   edge(448, labels, wall=1.0)]
        letter, why = res.outcome(results)
        self.assertEqual(letter, "A")
        self.assertIn("448", why)

    def test_only_the_acceptable_ones_are_candidates_for_the_default(self):
        """672 keeps its verdicts, 448 does not — the default may only move to 672."""
        labels = sample(300, documents=50)
        broken = [PRODUCT] * 300
        results = [edge(896, labels, wall=2.0), edge(672, labels, wall=1.2),
                   edge(448, broken, wall=1.0)]
        letter, why = res.outcome(results)
        self.assertEqual(letter, "A")
        self.assertIn("672", why)

    def test_a_single_resolution_has_nothing_to_compare(self):
        letter, why = res.outcome([edge(896, sample(300))])
        self.assertEqual(letter, "C")
        self.assertIn("сравнивать не с чем", why)


class TestReport(unittest.TestCase):
    def test_the_table_states_the_speedup_against_the_baseline(self):
        table = res.format_table(faster([PHOTO] * 10, [PHOTO] * 10, times=2.0))
        self.assertIn("x1.00", table)
        self.assertIn("x2.00", table)
        self.assertIn("896", table)
        self.assertIn("448", table)

    def test_the_verdict_block_names_every_threshold_it_judges_by(self):
        base = sample(300, documents=50)
        other = list(base)
        other[0] = PHOTO
        report = res.format_verdicts(faster(base, other, times=2.0))
        self.assertIn("согласие", report)
        self.assertIn("документов", report)
        self.assertIn("document -> personal_photo: 1", report)

    def test_the_outcome_line_carries_the_letter(self):
        labels = sample(300, documents=50)
        self.assertTrue(
            res.format_outcome(faster(labels, labels, times=1.6)).startswith("ИСХОД A"))

    def test_a_single_resolution_says_so_in_the_verdict_block(self):
        self.assertIn("сравнивать не с чем",
                      res.format_verdicts([edge(896, [DOC])]))


class TestReportIdentifiesNothing(unittest.TestCase):
    """Privacy: a table about documents must not become a list of where they are."""

    def test_no_frame_identity_reaches_the_output(self):
        base = sample(300, documents=50)
        other = list(base)
        other[0] = PHOTO
        results = faster(base, other, times=2.0)
        text = "\n".join([res.format_table(results), res.format_verdicts(results),
                          res.format_outcome(results)])
        for leak in ("/photos", ".jpg", "file_id", "IMG_"):
            self.assertNotIn(leak, text)


if __name__ == "__main__":
    unittest.main()
