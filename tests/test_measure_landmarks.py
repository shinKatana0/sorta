"""F131 phase 0: the landmark probe — the sample, the two forms, the verdict.

The feature this belongs to may never be built: the probe exists to find out whether a
3B VLM knows a landmark or merely agrees with CLIP about one, and "it agrees" closes the
feature. So what is tested here is that the measurement cannot flatter itself. The
ground truth must come from the pipeline's own verdicts rather than from a copy of them;
a wrong proposal must stay wrong; an answer the model did not give must never count as a
confirmation; the verdict must follow the pre-registered numbers and nothing else; and,
as with every measurement in this project, nothing the report prints may identify a
frame.

No model, no GPU: the VLM is a function returning strings, CLIP is a table of scores.
"""
from __future__ import annotations

import importlib.util
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from sorta.db import connect
from sorta.landmarks import Landmark

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "measure_landmarks.py"


def _load_script():
    """Import scripts/measure_landmarks.py — it is a script, not a package module."""
    spec = importlib.util.spec_from_file_location("measure_landmarks", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


probe = _load_script()

# Two bridges on purpose: "bridge" belongs to neither of them alone, which is the case
# the uniqueness rule of match_named_landmark exists for.
LANDMARKS = [
    Landmark(prompt="a photo of the Charles Bridge in Prague", name="Карлов мост",
             country="CZ", city="Prague", geonameid=3067696),
    Landmark(prompt="a photo of Tower Bridge in London", name="Тауэрский мост",
             country="GB", city="London", geonameid=2643741),
    Landmark(prompt="a photo of the Eiffel Tower in Paris", name="Эйфелева башня",
             country="FR", city="Paris", geonameid=2988507),
]
PRAGUE, LONDON, PARIS = 0, 1, 2


def answer(group=probe.GROUP_CONFIRMED, correct=True, confirmed=True,
           named=probe.NAMED_PROPOSED):
    return probe.Answer(group=group, correct=correct, confirmed=confirmed, named=named)


def right(**kwargs):
    """One frame whose proposal IS the right landmark."""
    return answer(group=probe.GROUP_CONFIRMED, correct=True, **kwargs)


def wrong(group=probe.GROUP_REJECTED, **kwargs):
    """One frame whose proposal is NOT the right landmark."""
    return answer(group=group, correct=False, **kwargs)


class TestLandmarkPhrase(unittest.TestCase):
    """The question is asked in English, because that is what the model was trained in."""

    def test_the_clip_prompt_becomes_the_wording_of_the_question(self):
        self.assertEqual(probe.landmark_phrase(LANDMARKS[PRAGUE]),
                         "the Charles Bridge in Prague")
        self.assertEqual(probe.landmark_phrase(LANDMARKS[PARIS]),
                         "the Eiffel Tower in Paris")

    def test_the_russian_name_is_never_what_is_asked(self):
        """`name` is the interface language; a question in it would measure Russian."""
        for landmark in LANDMARKS:
            self.assertNotIn(landmark.name, probe.landmark_phrase(landmark))

    def test_a_prompt_without_the_usual_opening_is_left_alone(self):
        odd = Landmark(prompt="Red Square in Moscow", name="Красная площадь",
                       country="RU", city="Moscow")
        self.assertEqual(probe.landmark_phrase(odd), "Red Square in Moscow")

    def test_an_empty_prompt_falls_back_to_the_name(self):
        empty = Landmark(prompt="a photo of ", name="Место", country="RU", city="Moscow")
        self.assertEqual(probe.landmark_phrase(empty), "Место")


class TestYesNo(unittest.TestCase):
    """The verification form: read leniently, but never invent a confirmation."""

    def test_a_one_word_answer(self):
        self.assertIs(probe.parse_yes_no("yes"), True)
        self.assertIs(probe.parse_yes_no("No."), False)

    def test_the_model_explaining_itself_still_parses(self):
        self.assertIs(probe.parse_yes_no("Yes, this is the Charles Bridge."), True)
        self.assertIs(probe.parse_yes_no("no, it is some other bridge"), False)

    def test_the_word_that_answered_first_wins(self):
        self.assertIs(probe.parse_yes_no("No — yes it is a bridge, but not that one"),
                      False)

    def test_no_is_not_found_inside_another_word(self):
        """"not", "none", "nothing" are not the answer "no"."""
        self.assertIsNone(probe.parse_yes_no("nothing here is recognizable"))
        self.assertIs(probe.parse_yes_no("It is not the Eiffel Tower, no"), False)

    def test_an_answer_that_says_neither_is_not_a_confirmation(self):
        for text in ("", "   ", "a bridge over a river", "Charles Bridge"):
            with self.subTest(text=text):
                self.assertIsNone(probe.parse_yes_no(text))


class TestNamedLandmark(unittest.TestCase):
    """The naming form: a free answer mapped back onto the list, or onto nothing."""

    def match(self, text):
        return probe.match_named_landmark(text, LANDMARKS)

    def test_a_distinctive_word_names_the_landmark(self):
        self.assertEqual(self.match("The Eiffel Tower"), PARIS)
        self.assertEqual(self.match("charles bridge"), PRAGUE)

    def test_the_city_alone_names_it(self):
        self.assertEqual(self.match("somewhere in Prague, Czech Republic"), PRAGUE)

    def test_a_shared_word_needs_a_second_one(self):
        """"bridge" belongs to two entries here, so alone it names neither."""
        self.assertIsNone(self.match("a bridge over a river"))
        self.assertEqual(self.match("Tower Bridge"), LONDON)

    def test_a_word_shared_with_another_landmark_does_not_decide_the_match(self):
        self.assertEqual(self.match("the Eiffel Tower in Paris"), PARIS)
        self.assertEqual(self.match("Tower Bridge in London"), LONDON)

    def test_an_unknown_place_names_nothing(self):
        for text in ("none", "", "the Brooklyn Bridge", "a street in some old town"):
            with self.subTest(text=text):
                self.assertIsNone(self.match(text))

    def test_a_city_is_not_matched_inside_a_longer_word(self):
        self.assertIsNone(self.match("a house in Londonderry"))

    def test_uniqueness_is_computed_from_the_list_in_hand(self):
        """With only one bridge in the list, "bridge" is evidence again."""
        one_bridge = [LANDMARKS[PRAGUE], LANDMARKS[PARIS]]
        self.assertEqual(probe.match_named_landmark("a bridge", one_bridge), 0)


class TestQuestionAskers(unittest.TestCase):
    """Both forms over a fake runtime: the prompts, and a frame that will not decode."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = str(Path(self.tmp.name) / "frame.jpg")
        Image.new("RGB", (64, 48), (30, 90, 150)).save(self.path, "JPEG")
        self.asked: list[tuple[int, str, int]] = []

        def describe(frames, prompt, max_new_tokens):
            self.asked.append((len(frames), prompt, max_new_tokens))
            return "yes"

        self.verify, self.naming = probe.question_askers(describe, max_edge=64)

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_verification_question_names_the_proposed_place(self):
        self.assertEqual(self.verify(self.path, LANDMARKS[PRAGUE]), "yes")
        frames, prompt, tokens = self.asked[0]
        self.assertEqual(frames, 1)
        self.assertIn("the Charles Bridge in Prague", prompt)
        self.assertIn("yes or no", prompt)
        self.assertEqual(tokens, probe.PROBE_MAX_NEW_TOKENS)

    def test_the_naming_question_names_no_place_at_all(self):
        """It is the control question: a hint in it would be the answer."""
        self.naming(self.path)
        _frames, prompt, _tokens = self.asked[0]
        for landmark in LANDMARKS:
            self.assertNotIn(landmark.city, prompt)
            self.assertNotIn("Charles", prompt)

    def test_a_frame_that_will_not_decode_is_not_asked_about(self):
        missing = str(Path(self.tmp.name) / "gone.jpg")
        self.assertEqual(self.verify(missing, LANDMARKS[PRAGUE]), "")
        self.assertEqual(self.naming(missing), "")
        self.assertEqual(self.asked, [])

    def test_an_answer_that_never_happened_is_not_a_confirmation(self):
        missing = str(Path(self.tmp.name) / "gone.jpg")
        self.assertIsNone(probe.parse_yes_no(self.verify(missing, LANDMARKS[PRAGUE])))


class ProbeDatabaseCase(unittest.TestCase):
    """A DB shaped like a real index: files, and a place row per file."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.conn = connect(Path(self.tmp.name) / "test.db")
        self._n = 0

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def add(self, folder="/photos", confidence="unknown", country=None, city=None,
            media_type="photo", error=None, dup_of=None):
        self._n += 1
        path = f"{folder}/photo{self._n}.jpg"
        cur = self.conn.execute(
            """INSERT INTO files (path, size, mtime, ext, media_type, indexed_at,
                   error, dup_of)
               VALUES (?, 1000, 0, 'jpg', ?, '2026-01-01', ?, ?)""",
            (path, media_type, error, dup_of))
        file_id = int(cur.lastrowid or 0)
        self.conn.execute(
            """INSERT INTO places (file_id, country, city, confidence, updated_at)
               VALUES (?, ?, ?, ?, '2026-01-01')""",
            (file_id, country, city, confidence))
        self.conn.commit()
        return file_id, path


class TestConfirmedGroup(ProbeDatabaseCase):
    """The frames the stage already resolved — the only ones whose proposal is right."""

    def test_only_visual_rows_are_taken(self):
        self.add(confidence="visual", country="CZ", city="Prague")
        self.add(confidence="exact_gps", country="CZ", city="Prague")
        self.add(confidence="unknown")
        picked = probe.confirmed_candidates(self.conn, LANDMARKS, 10, _rng())
        self.assertEqual(len(picked), 1)
        self.assertEqual(picked[0].proposed, PRAGUE)
        self.assertTrue(picked[0].correct)
        self.assertEqual(picked[0].group, probe.GROUP_CONFIRMED)

    def test_a_city_no_longer_in_the_list_is_skipped_not_guessed(self):
        self.add(confidence="visual", country="IT", city="Rome")
        self.assertEqual(probe.confirmed_candidates(self.conn, LANDMARKS, 10, _rng()), [])

    def test_duplicates_and_broken_files_are_not_sampled(self):
        first, _path = self.add(confidence="visual", country="CZ", city="Prague")
        self.add(confidence="visual", country="CZ", city="Prague", dup_of=first)
        self.add(confidence="visual", country="GB", city="London", error="boom")
        self.add(confidence="visual", country="FR", city="Paris", media_type="video")
        picked = probe.confirmed_candidates(self.conn, LANDMARKS, 10, _rng())
        self.assertEqual([c.proposed for c in picked], [PRAGUE])

    def test_the_sample_is_bounded_and_reproducible(self):
        for _ in range(20):
            self.add(confidence="visual", country="CZ", city="Prague")
        first = probe.confirmed_candidates(self.conn, LANDMARKS, 5, _rng())
        self.assertEqual(len(first), 5)
        self.assertEqual([c.path for c in first],
                         [c.path for c in probe.confirmed_candidates(
                             self.conn, LANDMARKS, 5, _rng())])


def _rng(seed=17):
    import random
    return random.Random(seed)


def _row(file_id, path):
    return {"id": file_id, "path": path}


class TestCorroborationDrivesTheGroundTruth(unittest.TestCase):
    """The `rejected` group is the stage's own verdict, re-derived, not a copy of it."""

    def rows(self, folder, n, start=1):
        return [_row(start + i, f"{folder}/photo{start + i}.jpg") for i in range(n)]

    def test_proposals_below_the_threshold_are_not_proposals(self):
        rows = self.rows("/trip", 2)
        kept, dropped = probe.corroborated(
            rows, [(PRAGUE, 0.9), (LONDON, 0.4)], LANDMARKS, 0.85,
            min_group=5, dominance=0.6, resolver=None, lang="en")
        self.assertEqual(len(kept) + len(dropped), 1)
        self.assertEqual(kept[0][0]["id"], rows[0]["id"])

    def test_the_odd_city_out_of_a_folder_is_the_rejected_group(self):
        """The F75 group rule: one card dump, one trip — the minority is the mistake."""
        rows = self.rows("/trip", 6)
        results = [(PRAGUE, 0.9)] * 5 + [(LONDON, 0.9)]
        kept, dropped = probe.corroborated(rows, results, LANDMARKS, 0.85,
                                           min_group=5, dominance=0.6,
                                           resolver=None, lang="en")
        self.assertEqual(len(kept), 5)
        self.assertEqual([best for _row, best in dropped], [LONDON])

    def test_a_rejected_candidate_is_wrong_by_construction(self):
        rows = self.rows("/trip", 6)
        results = [(PRAGUE, 0.9)] * 5 + [(LONDON, 0.9)]
        _kept, dropped = probe.corroborated(rows, results, LANDMARKS, 0.85,
                                            min_group=5, dominance=0.6,
                                            resolver=None, lang="en")
        candidates = probe.rejected_candidates(dropped, 10, _rng())
        self.assertEqual(len(candidates), 1)
        self.assertFalse(candidates[0].correct)
        self.assertEqual(candidates[0].group, probe.GROUP_REJECTED)
        self.assertEqual(candidates[0].proposed, LONDON)

    def test_the_band_shrinks_as_the_threshold_rises(self):
        rows = self.rows("/trip", 4)
        results = [(PRAGUE, 0.55), (PRAGUE, 0.75), (PRAGUE, 0.88), (PRAGUE, 0.97)]
        band = probe.band_curve(rows, results, LANDMARKS, (0.50, 0.80, 0.95),
                                min_group=5, dominance=0.6, resolver=None, lang="en")
        self.assertEqual([r.proposals for r in band], [4, 2, 1])
        self.assertEqual([r.threshold for r in band], [0.50, 0.80, 0.95])

    def test_every_band_row_splits_into_kept_and_dropped(self):
        rows = self.rows("/trip", 6)
        results = [(PRAGUE, 0.9)] * 5 + [(LONDON, 0.9)]
        for row in probe.band_curve(rows, results, LANDMARKS, (0.5, 0.85),
                                    min_group=5, dominance=0.6,
                                    resolver=None, lang="en"):
            with self.subTest(threshold=row.threshold):
                self.assertEqual(row.kept + row.dropped, row.proposals)
                self.assertEqual(row.dropped, 1)


class TestForeignGroup(unittest.TestCase):
    """Frames from countries the list cannot match: every fire on them is wrong."""

    def setUp(self):
        self.rows = [_row(i, f"/thailand/photo{i}.jpg") for i in range(1, 5)]
        self.results = [(PRAGUE, 0.71), (LONDON, 0.95), (PARIS, 0.60), (PRAGUE, 0.88)]

    def test_the_most_confident_proposals_are_the_ones_asked_about(self):
        picked = probe.foreign_candidates(self.rows, self.results, 2)
        self.assertEqual([c.proposed for c in picked], [LONDON, PRAGUE])

    def test_they_are_all_wrong_by_construction(self):
        for candidate in probe.foreign_candidates(self.rows, self.results, 4):
            self.assertFalse(candidate.correct)
            self.assertEqual(candidate.group, probe.GROUP_FOREIGN)

    def test_an_empty_pool_is_an_empty_group_not_a_crash(self):
        self.assertEqual(probe.foreign_candidates([], [], 5), [])


class TestAsking(unittest.TestCase):
    """Both questions per frame, and what the naming answer is counted as."""

    def candidates(self):
        return [
            probe.Candidate(path="/a.jpg", group=probe.GROUP_CONFIRMED,
                            proposed=PRAGUE, correct=True),
            probe.Candidate(path="/b.jpg", group=probe.GROUP_REJECTED,
                            proposed=LONDON, correct=False),
        ]

    def ask(self, verify_answers, naming_answers):
        asked: list[str] = []

        def verify(path, landmark):
            asked.append(path)
            return verify_answers[path]

        def naming(path):
            return naming_answers[path]

        answers = probe.ask_all(self.candidates(), verify, naming, LANDMARKS)
        return answers, asked

    def test_every_frame_is_asked_both_questions(self):
        answers, asked = self.ask({"/a.jpg": "yes", "/b.jpg": "no"},
                                  {"/a.jpg": "Charles Bridge", "/b.jpg": "none"})
        self.assertEqual(asked, ["/a.jpg", "/b.jpg"])
        self.assertEqual([a.confirmed for a in answers], [True, False])
        self.assertEqual([a.named for a in answers],
                         [probe.NAMED_PROPOSED, probe.NAMED_NONE])

    def test_naming_the_proposed_landmark_backs_the_proposal(self):
        answers, _asked = self.ask({"/a.jpg": "yes", "/b.jpg": "yes"},
                                   {"/a.jpg": "Prague", "/b.jpg": "Tower Bridge"})
        self.assertEqual([a.named for a in answers],
                         [probe.NAMED_PROPOSED, probe.NAMED_PROPOSED])

    def test_naming_a_different_place_is_not_a_confirmation(self):
        answers, _asked = self.ask({"/a.jpg": "yes", "/b.jpg": "yes"},
                                   {"/a.jpg": "the Eiffel Tower", "/b.jpg": "Prague"})
        self.assertEqual([a.named for a in answers],
                         [probe.NAMED_OTHER, probe.NAMED_OTHER])
        self.assertFalse(probe._confirmed(answers[0], probe.FORM_NAMING))

    def test_the_group_and_the_truth_travel_with_the_answer(self):
        answers, _asked = self.ask({"/a.jpg": "yes", "/b.jpg": "yes"},
                                   {"/a.jpg": "none", "/b.jpg": "none"})
        self.assertEqual([(a.group, a.correct) for a in answers],
                         [(probe.GROUP_CONFIRMED, True), (probe.GROUP_REJECTED, False)])


class TestFormStats(unittest.TestCase):
    """The arithmetic of the decisive number."""

    def setUp(self):
        self.answers = (
            [right(confirmed=True)] * 8
            + [right(confirmed=False)] * 2
            + [wrong(confirmed=True)] * 3
            + [wrong(confirmed=False)] * 7
        )

    def test_the_shares_are_over_their_own_side_of_the_sample(self):
        stats = probe.form_stats(self.answers, probe.FORM_VERIFY)
        self.assertEqual((stats.right_total, stats.right_confirmed), (10, 8))
        self.assertEqual((stats.wrong_total, stats.wrong_confirmed), (10, 3))
        self.assertAlmostEqual(stats.true_confirm, 0.8)
        self.assertAlmostEqual(stats.false_confirm, 0.3)

    def test_the_accuracy_counts_a_rejected_wrong_proposal_as_right(self):
        stats = probe.form_stats(self.answers, probe.FORM_VERIFY)
        self.assertAlmostEqual(stats.accuracy, (8 + 7) / 20)
        self.assertEqual(stats.total, 20)

    def test_an_answer_that_did_not_parse_never_confirms_anything(self):
        answers = [wrong(confirmed=None, named=probe.NAMED_NONE)] * 4
        stats = probe.form_stats(answers, probe.FORM_VERIFY)
        self.assertEqual(stats.wrong_confirmed, 0)
        self.assertEqual(stats.unparsed, 4)

    def test_the_naming_form_reads_the_naming_field(self):
        answers = [right(confirmed=False, named=probe.NAMED_PROPOSED),
                   wrong(confirmed=True, named=probe.NAMED_OTHER)]
        stats = probe.form_stats(answers, probe.FORM_NAMING)
        self.assertEqual((stats.right_confirmed, stats.wrong_confirmed), (1, 0))
        self.assertEqual(stats.named_other, 1)

    def test_an_empty_sample_is_not_a_division_by_zero(self):
        stats = probe.form_stats([], probe.FORM_VERIFY)
        self.assertEqual((stats.accuracy, stats.true_confirm, stats.false_confirm),
                         (0.0, 0.0, 0.0))


class TestVerdict(unittest.TestCase):
    """The pre-registered criteria decide, and nothing else does."""

    def sample(self, true_confirm, false_confirm, frames=60):
        """A sample of `frames` split in half, with the two rates dialled in."""
        half = frames // 2
        confirmed_right = round(true_confirm * half)
        confirmed_wrong = round(false_confirm * half)
        return (
            [right(confirmed=True)] * confirmed_right
            + [right(confirmed=False)] * (half - confirmed_right)
            + [wrong(confirmed=True)] * confirmed_wrong
            + [wrong(confirmed=False)] * (half - confirmed_wrong)
        )

    def decide(self, answers):
        stats = [probe.form_stats(answers, probe.FORM_VERIFY),
                 probe.form_stats(answers, probe.FORM_NAMING)]
        return probe.decide(stats, len(answers))

    def test_a_model_that_separates_opens_phase_1(self):
        verdict, why = self.decide(self.sample(0.9, 0.03))
        self.assertEqual(verdict, probe.VERDICT_GO)
        self.assertIn(probe.FORM_VERIFY, why)

    def test_a_model_that_confirms_wrong_cities_closes_the_feature(self):
        """The outcome the brief calls a normal result: measured, not guessed."""
        verdict, why = self.decide(self.sample(0.9, 0.4))
        self.assertEqual(verdict, probe.VERDICT_CLOSE)
        self.assertIn("40%", why)

    def test_a_gate_that_would_lose_todays_finds_also_closes_it(self):
        """Rejecting wrong proposals is not enough if it rejects the right ones too."""
        verdict, _why = self.decide(self.sample(0.3, 0.0))
        self.assertEqual(verdict, probe.VERDICT_CLOSE)

    def test_a_sample_below_the_brief_minimum_decides_nothing(self):
        verdict, why = self.decide(self.sample(0.9, 0.0, frames=20))
        self.assertEqual(verdict, probe.VERDICT_UNCLEAR)
        self.assertIn(str(probe.MIN_PROBE_FRAMES), why)

    def test_a_sample_with_only_one_side_decides_nothing(self):
        verdict, _why = self.decide([right(confirmed=True)] * 60)
        self.assertEqual(verdict, probe.VERDICT_UNCLEAR)

    def test_either_form_may_carry_the_feature(self):
        """The cascade would use whichever question works; the verdict names it."""
        answers = (
            [right(confirmed=False, named=probe.NAMED_PROPOSED)] * 30
            + [wrong(confirmed=True, named=probe.NAMED_NONE)] * 30
        )
        verdict, why = self.decide(answers)
        self.assertEqual(verdict, probe.VERDICT_GO)
        self.assertIn(probe.FORM_NAMING, why)

    def test_the_criteria_are_the_ones_written_down_before_the_run(self):
        self.assertEqual(probe.MIN_PROBE_FRAMES, 50)
        self.assertEqual(probe.MAX_FALSE_CONFIRM, 0.10)
        self.assertEqual(probe.MIN_TRUE_CONFIRM, 0.70)

    def test_the_boundary_values_pass_rather_than_fail(self):
        verdict, _why = self.decide(self.sample(probe.MIN_TRUE_CONFIRM,
                                                probe.MAX_FALSE_CONFIRM))
        self.assertEqual(verdict, probe.VERDICT_GO)


class TestReport(unittest.TestCase):
    """What the run prints — and what it must never print."""

    def setUp(self):
        self.answers = (
            [right(confirmed=True, named=probe.NAMED_PROPOSED)] * 20
            + [wrong(confirmed=True, named=probe.NAMED_OTHER)] * 15
            + [wrong(group=probe.GROUP_FOREIGN, confirmed=False,
                     named=probe.NAMED_NONE)] * 15
        )
        self.counts = {probe.GROUP_CONFIRMED: 20, probe.GROUP_REJECTED: 15,
                       probe.GROUP_FOREIGN: 15}
        self.band = [probe.BandRow(threshold=0.5, proposals=900, kept=300),
                     probe.BandRow(threshold=0.85, proposals=200, kept=144)]
        self.text = probe.probe_report(self.answers, self.counts, self.band, 0.85)

    def test_the_decisive_number_is_in_the_report(self):
        self.assertIn("подтвердила НЕВЕРНЫХ предложений: 15 из 30 (50%)", self.text)
        self.assertIn("решающее число", self.text)

    def test_both_forms_are_reported(self):
        self.assertIn(f"Форма «{probe.FORM_VERIFY}»", self.text)
        self.assertIn(f"Форма «{probe.FORM_NAMING}»", self.text)

    def test_the_band_table_is_what_phase_1_would_pick_a_threshold_from(self):
        self.assertIn("Полоса неуверенности", self.text)
        self.assertIn("900", self.text)
        self.assertIn("144", self.text)

    def test_the_verdict_line_carries_the_verdict(self):
        """The fixture is a model whose naming form separates — so: phase 1.

        The verification form on the same frames confirms half the wrong proposals,
        which is what makes the pair worth reporting separately at all.
        """
        self.assertIn(f"ВЕРДИКТ ФАЗЫ 0: {probe.VERDICT_GO}", self.text)
        self.assertIn(f"форма «{probe.FORM_NAMING}»", self.text)

    def test_every_group_gets_a_line_with_its_expected_answer(self):
        for group in probe.PROBE_GROUPS:
            self.assertIn(group, self.text)
        self.assertIn("правильный ответ — «да»", self.text)
        self.assertIn("правильный ответ — «нет»", self.text)

    def test_a_small_sample_says_so_instead_of_pretending(self):
        text = probe.format_sample({probe.GROUP_CONFIRMED: 3})
        self.assertIn(f"< {probe.MIN_PROBE_FRAMES}", text)
        self.assertNotIn("мало", probe.format_sample(self.counts))

    def test_nothing_in_the_report_identifies_a_frame(self):
        for leak in ("/photos", ".jpg", "file_id", "IMG_", "\\"):
            self.assertNotIn(leak, self.text)

    def test_the_answer_carries_no_identity_at_all(self):
        """The whole field set on purpose: it is what would catch a path added later."""
        self.assertEqual(set(probe.Answer.__dataclass_fields__),
                         {"group", "correct", "confirmed", "named"})


class TestScriptStillMeasuresTheThreshold(unittest.TestCase):
    """The probe is an addition — the F75 precision measurement must stay intact."""

    def test_the_threshold_grid_is_unchanged(self):
        self.assertEqual(probe.THRESHOLDS, (0.50, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99))

    def test_the_probe_is_opt_in(self):
        source = _SCRIPT.read_text(encoding="utf-8")
        self.assertIn('ap.add_argument("--probe", action="store_true"', source)

    def test_the_prompts_come_from_the_pipeline(self):
        """Imported, not copied: a private prompt set would measure another stage."""
        from sorta import landmarks as stage
        self.assertIs(probe.landmark_prompts, stage.landmark_prompts)
        self.assertIs(probe._corroborate, stage._corroborate)


class TestClassifyHelper(unittest.TestCase):
    """The CLIP helper both halves of the script share."""

    def test_argmax_is_taken_over_the_landmarks_only(self):
        import numpy as np

        def classifier(paths, prompts):
            row = np.zeros((1, len(prompts)), dtype=np.float32)
            row[0, len(LANDMARKS)] = 0.99   # a distractor column takes the mass
            row[0, LONDON] = 0.4
            return row

        result = probe._classify(classifier, ["/a.jpg"],
                                 ["p"] * (len(LANDMARKS) + 3), len(LANDMARKS), 8)
        (best, score), = result
        self.assertEqual(best, LONDON)
        self.assertAlmostEqual(score, 0.4, places=5)


class TestSqliteRowsWork(ProbeDatabaseCase):
    """The helpers take real sqlite3.Row objects, not only the dicts of the tests."""

    def test_corroboration_reads_a_row_object(self):
        file_id, path = self.add(folder="/trip")
        rows = self.conn.execute("SELECT id, path FROM files").fetchall()
        self.assertIsInstance(rows[0], sqlite3.Row)
        kept, dropped = probe.corroborated(rows, [(PRAGUE, 0.9)], LANDMARKS, 0.85,
                                           min_group=5, dominance=0.6,
                                           resolver=None, lang="en")
        self.assertEqual([r["id"] for r, _b in kept], [file_id])
        self.assertEqual(dropped, [])
        self.assertEqual(probe.foreign_candidates(rows, [(PRAGUE, 0.9)], 1)[0].path, path)


if __name__ == "__main__":
    unittest.main()
