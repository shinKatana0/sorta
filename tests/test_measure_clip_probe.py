"""F107: the CLIP probe — the honesty of the split, the arithmetic, the pre-registered verdict.

The script asks whether a light classifier over CLIP features reproduces what the VLM
decided, and the brief fixed the shape of the answer before the first run: three outcomes
with numbers attached to them. That is what is tested here — that the measurement cannot
flatter itself. The split has to be stratified, reproducible and disjoint; the metrics have
to come from the held-out part alone (a probe trained on noise must score like noise); the
matrix has to add up; the gate curve has to be monotone; and the "do nothing" row has to be
computed from the snapshot rather than from the labels it is supposed to be a baseline for.

No model, no GPU, no photo: the feature vectors below are synthetic, everything else is
arithmetic over per-frame aggregates. And, as with every measurement in this project,
nothing the script prints may identify a frame.
"""
from __future__ import annotations

import importlib.util
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "measure_clip_probe.py"


def _load_script():
    """Import scripts/measure_clip_probe.py — it is a script, not a package module."""
    spec = importlib.util.spec_from_file_location("measure_clip_probe", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


probe = _load_script()

PHOTO = "photo"
PRODUCT = "product"
DOC = "document"


def sample(features, label, before=PHOTO):
    return probe.Sample(features=tuple(features), label=label, before=before)


def answer(label=PHOTO, predicted=None, confidence=0.9, before=PHOTO):
    """One held-out frame after the probe answered — `predicted=None` means it agreed."""
    return probe.Answer(label=label, predicted=label if predicted is None else predicted,
                        confidence=confidence, before=before)


def separable(n_per_class=40, seed=0):
    """Three classes CLIP could not confuse: a different corner of the space each."""
    rng = random.Random(seed)
    corners = {PHOTO: (1.0, 0.0, 0.0), PRODUCT: (0.0, 1.0, 0.0), DOC: (0.0, 0.0, 1.0)}
    return [sample([c + rng.uniform(-0.02, 0.02) for c in corner], label)
            for label, corner in corners.items() for _ in range(n_per_class)]


def noise(n_per_class=60, features=6, seed=0):
    """Features that carry nothing about the label — the floor any metric must fall to."""
    rng = random.Random(seed)
    return [sample([rng.random() for _ in range(features)], label)
            for label in (PHOTO, PRODUCT, DOC) for _ in range(n_per_class)]


class TestSplitIsHonest(unittest.TestCase):
    """Test 1 of the brief: stratified, reproducible by seed, and the parts are disjoint."""

    def setUp(self):
        self.labels = [PHOTO] * 100 + [PRODUCT] * 50 + [DOC] * 10

    def test_the_parts_are_disjoint_and_cover_the_sample(self):
        train, test = probe.stratified_split(self.labels, 0.3, seed=1)
        self.assertEqual(set(train) & set(test), set())
        self.assertEqual(sorted(train + test), list(range(len(self.labels))))

    def test_every_class_is_split_in_the_same_proportion(self):
        _train, test = probe.stratified_split(self.labels, 0.3, seed=1)
        held = [self.labels[i] for i in test]
        self.assertEqual(held.count(PHOTO), 30)
        self.assertEqual(held.count(PRODUCT), 15)
        self.assertEqual(held.count(DOC), 3)

    def test_the_same_seed_gives_the_same_split(self):
        first = probe.stratified_split(self.labels, 0.3, seed=7)
        self.assertEqual(first, probe.stratified_split(self.labels, 0.3, seed=7))

    def test_another_seed_gives_another_split(self):
        a = probe.stratified_split(self.labels, 0.3, seed=1)
        b = probe.stratified_split(self.labels, 0.3, seed=2)
        self.assertNotEqual(a, b)

    def test_a_class_of_one_frame_stays_in_training(self):
        """A class the probe never saw is a column of zeros — a sample-size artefact."""
        _train, test = probe.stratified_split([PHOTO] * 10 + [DOC], 0.3, seed=1)
        self.assertNotIn(10, test)

    def test_a_share_outside_the_unit_interval_is_an_error(self):
        for bad in (0.0, 1.0, -0.1, 1.5):
            with self.assertRaises(SystemExit):
                probe.stratified_split(self.labels, bad, seed=1)


class TestProbeMeasuresTheHeldOutPart(unittest.TestCase):
    """Tests 2 and 3 of the brief: perfect data scores 100%, noise scores like noise."""

    def test_only_the_held_out_part_is_answered(self):
        samples = separable()
        run = probe.run_probe(samples, test_size=0.3, seed=1)
        train, test = probe.stratified_split([s.label for s in samples], 0.3, 1)
        self.assertEqual(len(run.answers), len(test))
        self.assertEqual(run.trained_on, len(train))

    def test_perfectly_separable_classes_are_reproduced_exactly(self):
        run = probe.run_probe(separable(), test_size=0.3, seed=1)
        self.assertEqual(probe.confusion(run.answers).agreement, 1.0)

    def test_noise_scores_near_chance_not_near_one(self):
        """If this ever passed at 95%, the metric would be reading the training part."""
        run = probe.run_probe(noise(), test_size=0.3, seed=1)
        agreement = probe.confusion(run.answers).agreement
        self.assertLess(agreement, 0.55)   # chance for three balanced classes is ~0.33
        self.assertLess(agreement, probe.MIN_AGREEMENT)

    def test_the_run_is_reproducible(self):
        first = probe.run_probe(separable(), test_size=0.3, seed=1)
        second = probe.run_probe(separable(), test_size=0.3, seed=1)
        self.assertEqual(first.answers, second.answers)

    def test_an_empty_sample_is_an_error_not_an_empty_report(self):
        with self.assertRaises(SystemExit):
            probe.run_probe([], test_size=0.3, seed=1)

    def test_a_single_class_is_an_error(self):
        samples = [sample([0.5, 0.5], PHOTO) for _ in range(10)]
        with self.assertRaises(SystemExit):
            probe.run_probe(samples, test_size=0.3, seed=1)

    def test_the_confidence_is_a_probability(self):
        run = probe.run_probe(separable(), test_size=0.3, seed=1)
        for a in run.answers:
            self.assertGreaterEqual(a.confidence, 0.0)
            self.assertLessEqual(a.confidence, 1.0)


class TestConfusionMatrix(unittest.TestCase):
    """Test 4 of the brief: the matrix adds up to the number of frames."""

    def setUp(self):
        self.answers = ([answer(PHOTO)] * 5 + [answer(PHOTO, PRODUCT)] * 2
                        + [answer(DOC, PHOTO)] * 3 + [answer(DOC)] * 1)
        self.evaluation = probe.confusion(self.answers)

    def test_the_pairs_sum_to_the_held_out_frames(self):
        self.assertEqual(self.evaluation.total, len(self.answers))
        self.assertEqual(sum(self.evaluation.pairs.values()), len(self.answers))

    def test_the_agreement_is_the_diagonal(self):
        self.assertEqual(self.evaluation.agreed, 6)
        self.assertAlmostEqual(self.evaluation.agreement, 6 / 11)

    def test_every_class_sums_to_its_own_frames(self):
        per_class = self.evaluation.per_class()
        self.assertEqual(per_class[PHOTO], (5, 7))
        self.assertEqual(per_class[DOC], (1, 4))
        self.assertEqual(sum(total for _agreed, total in per_class.values()), 11)

    def test_an_empty_held_out_part_is_not_a_division_by_zero(self):
        self.assertEqual(probe.confusion([]).agreement, 1.0)

    def test_the_matrix_is_printed_with_counts(self):
        text = probe.format_confusion(self.evaluation)
        self.assertIn("photo -> photo: 5, product: 2", text)
        self.assertIn("document -> photo: 3", text)


class TestDocumentsGetTheirOwnLine(unittest.TestCase):
    """`document -> anything` is papers in a city folder, not "slightly worse filing"."""

    def test_only_the_losing_direction_counts(self):
        evaluation = probe.confusion([answer(DOC)] * 8 + [answer(DOC, PHOTO)] * 2
                                     + [answer(PHOTO, DOC)] * 20)
        self.assertEqual(evaluation.document_leak(), (2, 10))

    def test_a_sample_without_documents_is_not_a_division_by_zero(self):
        self.assertEqual(probe.confusion([answer(PHOTO)]).document_leak(), (0, 0))
        self.assertIn("0 из 0", probe.format_documents(probe.confusion([answer(PHOTO)])))

    def test_the_report_names_what_the_documents_became(self):
        text = probe.format_documents(probe.confusion([answer(DOC, PHOTO)] * 3
                                                      + [answer(DOC, PRODUCT)]))
        self.assertIn("document -> photo: 3", text)
        self.assertIn("document -> product: 1", text)
        self.assertIn("ВЫШЕ", text)   # 100% leak, well above the 2% criterion


class TestGateCurve(unittest.TestCase):
    """Test 5 of the brief: monotone in N, and N=100% preserves every change."""

    def setUp(self):
        # 10 frames, the changed verdicts sitting at the low-confidence end and answered
        # wrongly by the probe — exactly the frames a smart gate is supposed to catch.
        self.answers = (
            [answer(PRODUCT, PHOTO, confidence=0.30 + 0.01 * i, before=PHOTO)
             for i in range(2)]
            + [answer(PHOTO, PHOTO, confidence=0.50 + 0.01 * i, before=PHOTO)
               for i in range(6)]
            + [answer(PRODUCT, PHOTO, confidence=0.90 + 0.01 * i, before=PHOTO)
               for i in range(2)]
        )

    def test_the_gate_takes_the_least_confident_frames_first(self):
        row = probe.gate_curve(self.answers, (0.2,))[0]
        self.assertEqual((row.to_vlm, row.kept, row.changed_total), (2, 2, 4))

    def test_the_curve_is_monotone_in_n(self):
        rows = probe.gate_curve(self.answers, (0.1, 0.2, 0.3, 0.5, 1.0))
        kept = [r.kept for r in rows]
        self.assertEqual(kept, sorted(kept))
        self.assertEqual([r.to_vlm for r in rows], sorted(r.to_vlm for r in rows))

    def test_sending_everything_preserves_everything(self):
        row = probe.gate_curve(self.answers, (1.0,))[0]
        self.assertEqual(row.to_vlm, len(self.answers))
        self.assertEqual(row.kept, row.changed_total)
        self.assertEqual(row.kept_frac, 1.0)
        self.assertEqual(row.lost, 0)

    def test_a_change_the_probe_gets_right_survives_without_the_model(self):
        answers = [answer(PRODUCT, PRODUCT, confidence=0.99, before=PHOTO)]
        self.assertEqual(probe.gate_curve(answers, (0.0,))[0].kept, 1)

    def test_an_unchanged_verdict_is_not_benefit(self):
        answers = [answer(PHOTO, PRODUCT, confidence=0.1, before=PHOTO)]
        row = probe.gate_curve(answers, (1.0,))[0]
        self.assertEqual((row.changed_total, row.kept), (0, 0))
        self.assertEqual(row.kept_frac, 1.0)

    def test_an_empty_held_out_part_gates_nobody(self):
        row = probe.gate_curve([], (0.3,))[0]
        self.assertEqual((row.to_vlm, row.kept, row.changed_total), (0, 0, 0))

    def test_the_brief_grid_is_the_one_reported(self):
        self.assertEqual(probe.GATE_GRID[:4], (0.10, 0.20, 0.30, 0.50))
        self.assertIn(1.00, probe.GATE_GRID)   # the control row


class TestDoNothingBaseline(unittest.TestCase):
    """Test 6 of the brief: the baseline is the snapshot, not the labels."""

    def test_it_compares_the_fast_verdict_with_the_vlm_label(self):
        answers = [answer(PRODUCT, PRODUCT, before=PHOTO)] * 3 + [answer(PHOTO)] * 7
        self.assertEqual(probe.do_nothing(answers), (7, 10, 0))

    def test_a_perfect_probe_does_not_improve_the_baseline(self):
        """The probe answering every frame right says nothing about the fast tier."""
        answers = [answer(PRODUCT, PRODUCT, before=PHOTO)] * 10
        matched, known, _unknown = probe.do_nothing(answers)
        self.assertEqual((matched, known), (0, 10))
        self.assertEqual(probe.confusion(answers).agreement, 1.0)

    def test_the_report_carries_both_numbers(self):
        answers = [answer(PRODUCT, PHOTO, before=PHOTO)] * 4 + [answer(PHOTO)] * 6
        text = probe.format_baseline(answers)
        self.assertIn("совпало бы с VLM: 6 из 10", text)
        self.assertIn("изменённых вердиктов: 4", text)


class TestFramesWithoutABaseline(unittest.TestCase):
    """Test 7 of the brief: a frame in the DB that the snapshot does not have.

    Whether the tier changed anything for it is not knowable, so it is counted apart
    instead of being guessed in either direction.
    """

    def setUp(self):
        self.answers = [answer(PHOTO, before=None), answer(PRODUCT, PRODUCT, before=PHOTO),
                        answer(PHOTO)]

    def test_it_is_counted_on_its_own_line(self):
        self.assertEqual(probe.do_nothing(self.answers), (1, 2, 1))
        self.assertIn("нет в снимке «до»: 1", probe.format_baseline(self.answers))

    def test_it_is_not_a_changed_verdict(self):
        self.assertFalse(self.answers[0].changed)
        self.assertEqual(probe.gate_curve(self.answers, (1.0,))[0].changed_total, 1)

    def test_it_still_counts_in_the_agreement(self):
        self.assertEqual(probe.confusion(self.answers).total, 3)

    def test_a_report_without_any_missing_frame_stays_quiet(self):
        self.assertNotIn("нет в снимке", probe.format_baseline(self.answers[1:]))


class TestPreRegisteredCriteria(unittest.TestCase):
    """The numbers themselves — written down before the run, and they must stay put."""

    def test_the_criteria_are_the_ones_the_brief_registered(self):
        self.assertEqual(probe.MIN_AGREEMENT, 0.95)
        self.assertEqual(probe.MAX_DOCUMENT_LEAK, 0.02)
        self.assertEqual(probe.MIN_CHANGES_KEPT, 0.98)
        self.assertEqual(probe.MAX_SMART_GATE_SHARE, 0.30)

    def test_the_default_split_is_the_one_the_brief_asked_for(self):
        self.assertEqual(probe.DEFAULT_TEST_SIZE, 0.3)

    def test_the_features_are_the_prompt_classes_of_the_pipeline(self):
        """Imported, not copied: a private copy would measure the script against itself."""
        from sorta import junk
        groups = dict(probe.FEATURE_GROUPS)
        self.assertIs(groups["base"], junk._CLIP_CLASSES)
        self.assertIs(groups["document"], junk._DOCUMENT_CLASSES)
        self.assertIs(groups["product"], junk._PRODUCT_CLASSES)
        self.assertEqual(probe.N_FEATURES,
                         len(junk._CLIP_CLASSES) + len(junk._DOCUMENT_CLASSES)
                         + len(junk._PRODUCT_CLASSES))


class TestOutcome(unittest.TestCase):
    """A, B or C — decided by the criteria above and by nothing else."""

    def curve(self, kept_at_30, changed_total=100):
        """A curve where N=30% preserves `kept_at_30` of the changed verdicts."""
        return [probe.GateRow(share=0.3, to_vlm=30, kept=kept_at_30,
                              changed_total=changed_total),
                probe.GateRow(share=1.0, to_vlm=100, kept=changed_total,
                              changed_total=changed_total)]

    def test_high_agreement_and_safe_documents_is_outcome_a(self):
        evaluation = probe.confusion([answer(PHOTO)] * 96 + [answer(PHOTO, PRODUCT)] * 4
                                     + [answer(DOC)] * 100)
        letter, why = probe.decide(evaluation, self.curve(50))
        self.assertEqual(letter, "A")
        self.assertIn("согласие с VLM", why)

    def test_leaking_documents_blocks_outcome_a(self):
        """96% agreement, but a tenth of the documents became photos."""
        evaluation = probe.confusion([answer(PHOTO)] * 96 + [answer(DOC)] * 90
                                     + [answer(DOC, PHOTO)] * 10)
        letter, _why = probe.decide(evaluation, self.curve(50))
        self.assertNotEqual(letter, "A")

    def test_a_cheap_gate_that_keeps_the_changes_is_outcome_b(self):
        evaluation = probe.confusion([answer(PHOTO)] * 80 + [answer(PHOTO, PRODUCT)] * 20)
        letter, why = probe.decide(evaluation, self.curve(99))
        self.assertEqual(letter, "B")
        self.assertIn("N=30%", why)

    def test_a_gate_that_loses_changes_is_outcome_c(self):
        evaluation = probe.confusion([answer(PHOTO)] * 80 + [answer(PHOTO, PRODUCT)] * 20)
        letter, why = probe.decide(evaluation, self.curve(90))
        self.assertEqual(letter, "C")
        self.assertIn("ни A, ни B", why)

    def test_a_gate_that_is_too_expensive_is_outcome_c(self):
        """Preserving everything at 50% of the population is not the cut B is about."""
        curve = [probe.GateRow(share=0.5, to_vlm=50, kept=100, changed_total=100)]
        evaluation = probe.confusion([answer(PHOTO)] * 80 + [answer(PHOTO, PRODUCT)] * 20)
        letter, why = probe.decide(evaluation, curve)
        self.assertEqual(letter, "C")
        self.assertIn(f"нет N <= {probe.MAX_SMART_GATE_SHARE:.0%}", why)

    def test_the_outcome_line_carries_the_letter(self):
        evaluation = probe.confusion([answer(PHOTO)] * 100)
        self.assertTrue(
            probe.format_outcome(evaluation, self.curve(50)).startswith("ИСХОД A"))

    def test_noise_ends_in_outcome_c(self):
        """The end-to-end shape of the failure the brief calls a normal result."""
        run = probe.run_probe(noise(), test_size=0.3, seed=1)
        evaluation = probe.confusion(run.answers)
        letter, _why = probe.decide(evaluation, probe.gate_curve(run.answers))
        self.assertEqual(letter, "C")


class TestBeforeSnapshot(unittest.TestCase):
    """The snapshot is produced outside this repo, so its shapes are what they are."""

    def load(self, payload):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "before.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            return probe.load_before(path)

    def test_a_flat_mapping_of_id_to_verdict(self):
        self.assertEqual(self.load({"1": PHOTO, "2": DOC}), {1: PHOTO, 2: DOC})

    def test_a_list_of_rows(self):
        self.assertEqual(
            self.load([{"file_id": 1, "verdict": PHOTO}, {"file_id": 2, "verdict": DOC}]),
            {1: PHOTO, 2: DOC})

    def test_a_list_of_pairs(self):
        self.assertEqual(self.load([[1, PHOTO], [2, DOC]]), {1: PHOTO, 2: DOC})

    def test_rows_under_a_key(self):
        self.assertEqual(self.load({"rows": [{"file_id": 7, "verdict": DOC}]}), {7: DOC})

    def test_an_empty_snapshot_is_an_error_not_an_empty_baseline(self):
        with self.assertRaises(SystemExit):
            self.load([])

    def test_something_that_is_not_a_snapshot_is_an_error(self):
        with self.assertRaises(SystemExit):
            self.load("media_class")


class TestReportIdentifiesNothing(unittest.TestCase):
    """Privacy: a table about documents must not become a list of where they are."""

    def test_no_frame_identity_reaches_the_output(self):
        samples = separable()
        run = probe.run_probe(samples, test_size=0.3, seed=1)
        evaluation = probe.confusion(run.answers)
        curve = probe.gate_curve(run.answers)
        text = "\n".join([
            probe.format_header(run, len(samples), 0.3, 1),
            probe.format_agreement(evaluation),
            probe.format_confusion(evaluation),
            probe.format_documents(evaluation),
            probe.format_gate_curve(curve),
            probe.format_baseline(run.answers),
            probe.format_outcome(evaluation, curve),
        ])
        for leak in ("/photos", ".jpg", "file_id", "IMG_", "\\"):
            self.assertNotIn(leak, text)

    def test_the_sample_carries_no_identity_at_all(self):
        self.assertEqual(set(probe.Sample.__dataclass_fields__),
                         {"features", "label", "before"})
        self.assertEqual(set(probe.Answer.__dataclass_fields__),
                         {"label", "predicted", "confidence", "before"})


if __name__ == "__main__":
    unittest.main()
