"""F110: the measurement of what the candidate gate never shows the model.

The script answers one question with numbers: of the frames the gate keeps away from the
deep tier, how many would the model have judged differently — and how many of those are
documents now lying in city folders. The brief fixed the shape of that answer before the
first run (two pre-registered thresholds, three named outcomes), precisely so the table
cannot talk anybody into a conclusion afterwards.

That is what is tested here: the population must be the unseen one and not the whole
collection, the sample must be random and reproducible, the counters must add up the way
somebody counting by hand would add them, and the two ways a frame can carry no
comparison — an answer that names no label, a fast verdict the model has no label for —
must be visible on their own lines instead of quietly becoming agreement.

No model, no GPU, no photo: everything below is arithmetic over per-frame aggregates and
a fake classifier. And, as with every measurement in this project, nothing the script
prints may identify a frame.
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from sorta import imaging, junk
from sorta.db import connect

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "measure_gate_misses.py"


def _load_script():
    """Import scripts/measure_gate_misses.py — it is a script, not a package module."""
    spec = importlib.util.spec_from_file_location("measure_gate_misses", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


misses = _load_script()

PHOTO = "photo"
DOC = "document"
PRODUCT = "product"
SCREENSHOT = "screenshot"


def answer(fast=PHOTO, deep=PHOTO):
    return misses.Answer(fast=fast, deep=deep)


def tally(answers, population=1000):
    return misses.tally(answers, population=population)


class TestPopulationIsTheUnseenOne(unittest.TestCase):
    """Test 1 of the brief: the sample comes from `tier='vlm' AND source != 'vlm'`.

    Those are exactly the frames the model was never asked about. A frame with
    source='vlm' is one it answered, and a frame with no deep tier at all was never in
    the question.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = Path(self.tmp.name) / "test.db"
        self.conn = connect(self.db)
        self.addCleanup(self.conn.close)

    def add(self, name, source="clip", tier="vlm", verdict=PHOTO, on_disk=True,
            dup_of=None, error=None):
        path = Path(self.tmp.name) / name
        if on_disk:
            path.write_bytes(b"x")
        cur = self.conn.execute(
            """INSERT INTO files (path, size, mtime, ext, media_type, indexed_at,
                                  dup_of, error)
               VALUES (?, 1, 0, 'jpg', 'photo', '2026-01-01', ?, ?)""",
            (str(path), dup_of, error))
        if source is not None:
            self.conn.execute(
                """INSERT INTO media_class (file_id, verdict, source, score, updated_at,
                       tier) VALUES (?, ?, ?, NULL, '2026-01-01', ?)""",
                (cur.lastrowid, verdict, source, tier))
        self.conn.commit()
        return str(path)

    def paths(self):
        return {r["path"] for r in misses.unseen_rows(str(self.db))}

    def test_the_frames_the_gate_did_not_let_through_are_the_population(self):
        unseen = {self.add("a.jpg"), self.add("b.jpg", source="ocr"),
                  self.add("c.jpg", source="heuristic")}
        self.assertEqual(self.paths(), unseen)

    def test_a_frame_the_model_answered_is_not_in_the_population(self):
        self.add("answered.jpg", source="vlm")
        kept = self.add("unseen.jpg")
        self.assertEqual(self.paths(), {kept})

    def test_a_frame_no_deep_run_touched_is_not_in_the_population(self):
        self.add("fast_only.jpg", tier="clip")
        self.add("no_tier.jpg", tier=None)
        self.add("unclassified.jpg", source=None)
        self.assertEqual(self.paths(), set())

    def test_duplicates_and_broken_files_stay_out(self):
        keeper = self.add("keep.jpg")
        self.add("dup.jpg", dup_of=1)
        self.add("broken.jpg", error="decode failed")
        self.assertEqual(self.paths(), {keeper})

    def test_the_verdict_travels_with_the_row(self):
        self.add("doc.jpg", verdict=DOC)
        rows = misses.unseen_rows(str(self.db))
        self.assertEqual([r["verdict"] for r in rows], [DOC])


class TestSample(unittest.TestCase):
    """Test 2 of the brief: random, reproducible, and not the first N rows by id."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.rows = [self.row(i) for i in range(200)]

    def row(self, i, on_disk=True):
        path = Path(self.tmp.name) / f"{i:04d}.jpg"
        if on_disk:
            path.write_bytes(b"x")
        return {"id": i, "path": str(path), "verdict": PHOTO}

    def test_the_same_seed_draws_the_same_frames(self):
        first = misses.take_sample(self.rows, 20, seed=misses.DEFAULT_SEED)
        second = misses.take_sample(self.rows, 20, seed=misses.DEFAULT_SEED)
        self.assertEqual([r["id"] for r in first], [r["id"] for r in second])

    def test_a_different_seed_draws_different_frames(self):
        first = misses.take_sample(self.rows, 20, seed=1)
        second = misses.take_sample(self.rows, 20, seed=2)
        self.assertNotEqual([r["id"] for r in first], [r["id"] for r in second])

    def test_the_sample_is_not_the_first_n_rows_by_id(self):
        """A prefix of the collection is one trip: the report would describe that trip."""
        drawn = [r["id"] for r in misses.take_sample(self.rows, 20, misses.DEFAULT_SEED)]
        self.assertNotEqual(drawn, list(range(20)))
        self.assertNotEqual(sorted(drawn), list(range(20)))

    def test_the_sample_size_is_respected_and_the_frames_are_distinct(self):
        drawn = misses.take_sample(self.rows, 30, seed=3)
        self.assertEqual(len(drawn), 30)
        self.assertEqual(len({r["id"] for r in drawn}), 30)

    def test_a_sample_larger_than_the_population_takes_all_of_it(self):
        self.assertEqual(len(misses.take_sample(self.rows, 500, seed=3)), 200)

    def test_a_frame_that_is_gone_from_disk_is_not_shown_to_the_model(self):
        rows = [self.row(900, on_disk=False), *self.rows[:5]]
        drawn = misses.take_sample(rows, 10, seed=3)
        self.assertNotIn(900, [r["id"] for r in drawn])
        self.assertEqual(len(drawn), 5)

    def test_an_empty_population_gives_an_empty_sample(self):
        self.assertEqual(misses.take_sample([], 10, seed=3), [])


class TestAnswerParsing(unittest.TestCase):
    """Test 5 of the brief: an answer that names no label is not silent agreement.

    The tier maps an unrecognized answer to `personal_photo` — conservative, and right
    for a pipeline. Here it would mean the worse the model behaves, the quieter the
    report gets.
    """

    def test_the_three_labels_map_to_the_verdicts_of_the_tier(self):
        self.assertEqual(misses.model_verdict("personal_photo"), PHOTO)
        self.assertEqual(misses.model_verdict("document"), DOC)
        self.assertEqual(misses.model_verdict("product"), PRODUCT)

    def test_the_parser_is_the_one_the_tier_uses(self):
        """A wordy answer is read exactly as junk._vlm_label reads it."""
        wordy = "This image is a document, I think."
        self.assertEqual(junk._vlm_label(wordy), DOC)
        self.assertEqual(misses.model_verdict(wordy), DOC)

    def test_an_answer_that_names_no_label_is_not_a_verdict(self):
        for garbage in ("", "  ", "hmm", "I cannot answer that", "42"):
            self.assertIsNone(misses.model_verdict(garbage))

    def test_the_tier_would_have_called_the_same_garbage_a_personal_photo(self):
        """The difference this function exists for, stated as a test."""
        self.assertEqual(junk._vlm_label("hmm"), junk._VLM_FALLBACK_LABEL)
        self.assertIsNone(misses.model_verdict("hmm"))

    def test_an_unparsed_answer_is_counted_apart_and_never_as_agreement(self):
        t = tally([answer(deep=None), answer(fast=DOC, deep=None), answer()])
        self.assertEqual((t.unparsed, t.agreed, t.mismatched, t.compared), (2, 1, 0, 1))


class TestCounters(unittest.TestCase):
    """Test 3 of the brief: the breakdown must add up to the number of disagreements."""

    def setUp(self):
        self.answers = [
            answer(PHOTO, DOC), answer(PHOTO, DOC),
            answer(PHOTO, PRODUCT), answer(PHOTO, PRODUCT), answer(PHOTO, PRODUCT),
            answer(DOC, PHOTO),
            answer(PHOTO, PHOTO), answer(DOC, DOC), answer(PRODUCT, PRODUCT),
        ]
        self.t = tally(self.answers)

    def test_the_pairs_sum_to_the_number_of_disagreements(self):
        self.assertEqual(sum(self.t.pairs.values()), self.t.mismatched)
        self.assertEqual(self.t.mismatched, 6)

    def test_every_pair_is_counted_with_its_direction(self):
        self.assertEqual(dict(self.t.pairs), {
            (PHOTO, DOC): 2, (PHOTO, PRODUCT): 3, (DOC, PHOTO): 1})

    def test_agreement_and_disagreement_split_the_compared_frames(self):
        self.assertEqual((self.t.agreed, self.t.compared), (3, 9))
        self.assertAlmostEqual(self.t.mismatch_frac, 6 / 9)
        self.assertAlmostEqual(self.t.agreement, 3 / 9)

    def test_only_photo_to_document_counts_as_a_missed_document(self):
        """`document -> photo` leaves a document a document; the other way is papers
        in a city folder."""
        self.assertEqual(self.t.documents, 2)
        self.assertAlmostEqual(self.t.document_frac, 2 / 9)

    def test_the_breakdown_names_the_pairs_and_says_it_adds_up(self):
        report = misses.format_pairs(self.t)
        self.assertIn("photo -> product: 3", report)
        self.assertIn("photo -> document: 2", report)
        self.assertIn("document -> photo: 1", report)
        self.assertIn("сумма по парам: 6", report)


class TestOutOfVocabulary(unittest.TestCase):
    """A screenshot cannot be confirmed: the model has no such label.

    Those frames are in the population — the tier processed them — and counting them as
    disagreement would price the vocabulary gap and call it a gate that is too narrow.
    """

    def setUp(self):
        self.t = tally([answer(SCREENSHOT, PHOTO), answer("meme", PHOTO), answer()])

    def test_they_are_counted_on_their_own_line(self):
        self.assertEqual(self.t.out_of_vocabulary, 2)
        self.assertIn("вне словаря модели: 2", misses.format_skipped(self.t))

    def test_they_are_neither_agreement_nor_disagreement(self):
        self.assertEqual((self.t.agreed, self.t.mismatched, self.t.compared), (1, 0, 1))
        self.assertEqual(self.t.mismatch_frac, 0.0)

    def test_the_buckets_partition_the_sample(self):
        t = tally([answer(SCREENSHOT, PHOTO), answer(deep=None), answer(),
                   answer(PHOTO, DOC)])
        self.assertEqual(t.sample, 4)
        self.assertEqual(
            t.agreed + t.mismatched + t.unparsed + t.out_of_vocabulary, t.sample)


class TestForecast(unittest.TestCase):
    """Test 4 of the brief: the forecast is the share times the size of the population."""

    def test_the_document_forecast_is_the_share_over_the_population(self):
        # 3 of 300 compared frames are photo -> document; the population is 16 300.
        answers = ([answer(PHOTO, DOC)] * 3 + [answer()] * 297)
        t = tally(answers, population=16_300)
        self.assertAlmostEqual(t.document_frac, 0.01)
        self.assertAlmostEqual(t.documents_forecast, 163.0)

    def test_it_is_computed_by_hand_the_same_way(self):
        t = tally([answer(PHOTO, DOC)] * 2 + [answer()] * 8, population=500)
        self.assertAlmostEqual(t.documents_forecast, 2 / 10 * 500)

    def test_the_price_of_the_whole_population_is_the_measured_seconds_per_frame(self):
        t = tally([answer()], population=16_300)
        self.assertAlmostEqual(t.population_minutes, 16_300 * 0.78 / 60.0)
        self.assertIn("мин GPU", misses.format_price(t))

    def test_frames_that_carry_no_comparison_do_not_dilute_the_share(self):
        """The denominator is the compared frames — stated as a test, not as a comment."""
        answers = [answer(PHOTO, DOC)] + [answer()] * 9 + [answer(deep=None)] * 10
        t = tally(answers, population=100)
        self.assertAlmostEqual(t.document_frac, 0.1)
        self.assertAlmostEqual(t.documents_forecast, 10.0)


class TestPreRegisteredCriteria(unittest.TestCase):
    """The numbers themselves — written down before the run, and they must stay put."""

    def test_the_thresholds_are_the_ones_the_brief_registered(self):
        self.assertEqual(misses.MISMATCH_MIN, 0.05)
        self.assertEqual(misses.DOCUMENT_MISS_MIN, 0.01)

    def test_the_sample_and_the_seed_are_the_ones_the_brief_fixed(self):
        self.assertEqual(misses.MIN_SAMPLE, 300)
        self.assertEqual(misses.DEFAULT_SEED, 20260729)

    def test_the_time_per_frame_is_the_measured_one(self):
        self.assertEqual(misses.SEC_PER_FRAME, 0.78)


class TestOutcome(unittest.TestCase):
    """A / B / C by the pre-registered criteria, including at their boundaries."""

    def collection(self, documents, other_mismatches, agreed):
        return ([answer(PHOTO, DOC)] * documents
                + [answer(PHOTO, PRODUCT)] * other_mismatches
                + [answer()] * agreed)

    def letter(self, *args, population=16_300, **kwargs):
        return misses.decide(tally(self.collection(*args, **kwargs), population))

    def test_many_disagreements_are_outcome_a(self):
        letter, why = self.letter(documents=0, other_mismatches=30, agreed=270)
        self.assertEqual(letter, "A")
        self.assertIn("гейт узок", why)
        self.assertIn("мин", why)  # the price of widening it, in minutes

    def test_documents_alone_are_outcome_b(self):
        # 1% documents, 2% disagreements overall — under A, over the document threshold
        letter, why = self.letter(documents=3, other_mismatches=3, agreed=294)
        self.assertEqual(letter, "B")
        self.assertIn("отдельный дешёвый вопрос про документы", why)
        self.assertIn("163", why)  # the forecast over the population, in frames

    def test_a_quiet_gate_is_outcome_c(self):
        letter, why = self.letter(documents=2, other_mismatches=3, agreed=295)
        self.assertEqual(letter, "C")
        self.assertIn("гейт настроен верно", why)

    def test_the_thresholds_are_inclusive(self):
        """">= 5%" and ">= 1%" — exactly as the brief writes them."""
        at_a = tally([answer(PHOTO, PRODUCT)] * 5 + [answer()] * 95)
        self.assertEqual(misses.decide(at_a)[0], "A")
        at_b = tally([answer(PHOTO, DOC)] * 1 + [answer()] * 99)
        self.assertEqual(misses.decide(at_b)[0], "B")

    def test_documents_inside_a_wide_disagreement_still_report_a(self):
        """A is the OR of the brief; B is its narrower case and is checked first."""
        letter, _why = self.letter(documents=10, other_mismatches=20, agreed=270)
        self.assertEqual(letter, "A")

    def test_the_outcome_line_carries_the_letter(self):
        t = tally(self.collection(documents=2, other_mismatches=3, agreed=295))
        self.assertTrue(misses.format_outcome(t).startswith("ИСХОД C"))

    def test_a_sample_that_compared_nothing_is_not_a_division_by_zero(self):
        t = tally([answer(deep=None), answer(SCREENSHOT, PHOTO)], population=16_300)
        self.assertEqual((t.mismatch_frac, t.agreement, t.documents_forecast),
                         (0.0, 1.0, 0.0))
        letter, why = misses.decide(t)
        self.assertEqual(letter, "C")
        self.assertIn("сравнивать не с чем", why)


class TestEmptyPopulation(unittest.TestCase):
    """Test 7 of the brief: no deep run — a clear message, not a division by zero."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = Path(self.tmp.name) / "test.db"
        self.conn = connect(self.db)
        self.conn.close()

    def patch(self, target, name, value):
        original = getattr(target, name)
        setattr(target, name, value)
        self.addCleanup(setattr, target, name, original)

    def run_main(self, argv=()):
        cfg = type("Cfg", (), {"database": str(self.db), "vlm": type(
            "Vlm", (), {"model": "Qwen/test", "workers": 1, "max_edge": 896})()})()
        self.patch(misses, "load_config", lambda _path: cfg)
        self.patch(sys, "argv", ["measure_gate_misses.py", *argv])
        return misses.main()

    def test_an_empty_population_stops_with_a_message(self):
        with self.assertRaises(SystemExit) as caught:
            self.run_main()
        self.assertIn("глубокий ярус", str(caught.exception))

    def test_a_population_with_no_files_left_on_disk_stops_too(self):
        conn = connect(self.db)
        self.addCleanup(conn.close)
        cur = conn.execute(
            """INSERT INTO files (path, size, mtime, ext, media_type, indexed_at)
               VALUES (?, 1, 0, 'jpg', 'photo', '2026-01-01')""",
            (str(Path(self.tmp.name) / "gone.jpg"),))
        conn.execute(
            """INSERT INTO media_class (file_id, verdict, source, score, updated_at, tier)
               VALUES (?, 'photo', 'clip', NULL, '2026-01-01', 'vlm')""",
            (cur.lastrowid,))
        conn.commit()
        with self.assertRaises(SystemExit) as caught:
            self.run_main()
        self.assertIn("нет на диске", str(caught.exception))

    def test_an_empty_tally_prints_a_report_instead_of_crashing(self):
        t = tally([], population=0)
        text = "\n".join([misses.format_summary(t), misses.format_pairs(t),
                          misses.format_documents(t), misses.format_skipped(t),
                          misses.format_price(t), misses.format_outcome(t)])
        self.assertIn("расхождений нет", text)
        self.assertIn("ИСХОД C", text)


class TestAskModel(unittest.TestCase):
    """The loop that shows frames to the model: the tier's classifier, the raw answer.

    The classifier is built exactly as the script builds it — `junk.vlm_classifier_from`
    over a runtime wrapped by `recording` — with two things stood in for: the runtime
    (a function that answers with text) and the decode (swapped out, as the junk tests
    do, so no real JPEG and no preview cache are involved).
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        original = imaging.decode_rgb_preview
        imaging.decode_rgb_preview = (  # type: ignore[assignment]
            lambda path, mtime, size, max_edge: Image.new("RGB", (8, 8)))
        self.addCleanup(setattr, imaging, "decode_rgb_preview", original)

    def classifier(self, describe):
        """The tier's classifier over `describe` -> (classifier, sink)."""
        sink: list[str] = []
        return junk.vlm_classifier_from(misses.recording(describe, sink),
                                        max_edge=448), sink

    def answering(self, replies, seen=None):
        """A runtime that hands out `replies` in order and checks the prompt it is given."""
        it = iter(replies)

        def describe(frames, prompt, max_new_tokens):
            if seen is not None:
                seen.append(prompt)
            return next(it)

        return describe

    def rows(self, verdicts, on_disk=True):
        out = []
        for i, verdict in enumerate(verdicts):
            path = Path(self.tmp.name) / f"{i}.jpg"
            if on_disk:
                path.write_bytes(b"x")  # never decoded — see setUp
            out.append({"id": i, "path": str(path), "verdict": verdict})
        return out

    def test_every_frame_gets_its_answer_paired_with_the_fast_verdict(self):
        seen: list[str] = []
        classifier, sink = self.classifier(
            self.answering(["document", "personal_photo"], seen))
        answers = misses.ask_model(classifier, sink, self.rows([PHOTO, PHOTO]))
        self.assertEqual([(a.fast, a.deep) for a in answers],
                         [(PHOTO, DOC), (PHOTO, PHOTO)])

    def test_the_model_is_asked_the_tiers_own_question(self):
        """The prompt is imported, not copied: a copy would measure another question."""
        seen: list[str] = []
        classifier, sink = self.classifier(self.answering(["product"], seen))
        misses.ask_model(classifier, sink, self.rows([PHOTO]))
        self.assertEqual(seen, [junk._VLM_PROMPT])

    def test_a_garbage_answer_arrives_as_no_verdict(self):
        classifier, sink = self.classifier(self.answering(["I would rather not say"]))
        answers = misses.ask_model(classifier, sink, self.rows([PHOTO]))
        self.assertIsNone(answers[0].deep)

    def test_a_frame_that_does_not_decode_never_reaches_the_model(self):
        seen: list[str] = []
        classifier, sink = self.classifier(self.answering(["document"], seen))
        answers = misses.ask_model(classifier, sink, self.rows([PHOTO], on_disk=False))
        self.assertEqual((answers[0].fast, answers[0].deep), (PHOTO, None))
        self.assertEqual(seen, [])

    def test_a_model_error_on_one_frame_does_not_end_the_measurement(self):
        def describe(frames, prompt, max_new_tokens):
            raise RuntimeError("CUDA out of memory")

        classifier, sink = self.classifier(describe)
        answers = misses.ask_model(classifier, sink, self.rows([PHOTO, DOC]))
        self.assertEqual([a.deep for a in answers], [None, None])
        self.assertEqual([a.fast for a in answers], [PHOTO, DOC])

    def test_the_recording_wrapper_hands_the_answer_through_unchanged(self):
        sink: list[str] = []
        describe = misses.recording(self.answering(["document"]), sink)
        self.assertEqual(describe([Image.new("RGB", (8, 8))], "prompt", 8), "document")
        self.assertEqual(sink, ["document"])

    def test_no_transcript_of_the_answers_survives_the_pass(self):
        """The sink holds one answer at a time and is empty when the pass is over."""
        classifier, sink = self.classifier(self.answering(["document", "product"]))
        misses.ask_model(classifier, sink, self.rows([PHOTO, PHOTO]))
        self.assertEqual(sink, [])


class TestReportIdentifiesNothing(unittest.TestCase):
    """Test 6 of the brief: no path, no id, no recognized content in the output.

    The model is shown documents exactly as the product tier already shows it documents.
    Only flags come back out.
    """

    def test_no_frame_identity_reaches_the_output(self):
        t = tally([answer(PHOTO, DOC), answer(PHOTO, PRODUCT), answer(SCREENSHOT, PHOTO),
                   answer(deep=None), answer()], population=16_300)
        text = "\n".join([misses.format_summary(t), misses.format_pairs(t),
                          misses.format_documents(t), misses.format_skipped(t),
                          misses.format_price(t), misses.format_outcome(t)])
        for leak in ("/photos", ".jpg", "file_id", "IMG_", "\\", "id="):
            self.assertNotIn(leak, text)

    def test_the_answer_text_is_not_reported_anywhere(self):
        """Only the parsed label survives an answer — the reply itself never leaves."""
        self.assertEqual(
            [f.name for f in misses.Answer.__dataclass_fields__.values()],
            ["fast", "deep"])


if __name__ == "__main__":
    unittest.main()
