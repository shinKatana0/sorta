"""F131: the local VLM checks a landmark CLIP proposed, before corroboration sees it.

The feature was measured before it was written, because it could easily have been worth
less than nothing. CLIP's failure on landmarks is not the perceptual one the animal
cascade cures — the wrong cities scored 0.980 against 0.991 for the right one — and a 3B
model could share exactly that weakness, in which case the check would confirm wrong
cities with authority. The probe said otherwise (zero wrong proposals confirmed on 104
frames, 24 of them hard), and these tests pin the properties that verdict rests on:

* with the toggle off nothing at all changes, down to the gate the proposals are
  collected at — a separately stated acceptance criterion, not a nicety;
* the model is a FILTER placed before F75, never a way around it: a country named in the
  path still refutes a match the model confirmed, and a rejected frame stays `unknown`
  rather than moving to the place the model did name;
* silence is a rejection and an unavailable model is not — the two failures look alike
  and mean opposite things, and getting them the wrong way round would either throw the
  feature away or lose finds the cheap tier makes today;
* an answer already paid for is not paid for twice.

CLIP is a table of scores and the VLM is a function returning strings — no model, no GPU.
The fixture (a tiny geo base, four landmarks, a DB of place-less photos) is the F75 one,
reused rather than rebuilt: this feature is a step inside that pipeline.
"""
from __future__ import annotations

import dataclasses
import unittest

from sorta.config import FeaturesConfig, _naming_from
from sorta.landmarks import (
    CHECK_CONFIRMED,
    CHECK_REJECTED,
    LANDMARK_NAMING_PROMPT,
    detect_landmarks,
    landmark_check_model,
)
from tests.test_landmark_corroboration import (
    BERLIN,
    PARIS,
    PRAGUE,
    CorroborationCase,
    PathClassifier,
)

# What the fake model answers. The words are the ones a real one gives back — the parser
# is `match_named_landmark`, shared with the phase-0 probe and tested there.
SAYS_PRAGUE = "Charles Bridge, Prague"
SAYS_BERLIN = "the Brandenburg Gate in Berlin"
SAYS_NOTHING = "none"
SILENCE = ""            # the model produced nothing at all — the common case in the probe


class Boom(RuntimeError):
    """The model raised: it was asked and nothing came back."""


class VerifyCase(CorroborationCase):
    """The F75 fixture, plus the check: a threshold pair and a scripted model."""

    threshold = 0.85          # naming.landmark_threshold — today's gate
    candidate = 0.5           # features.landmark_candidate_threshold — the check's gate
    verify = True

    def setUp(self) -> None:
        super().setUp()
        self.cfg = dataclasses.replace(
            self.cfg,
            naming=_naming_from({"landmarks_file": self.cfg.naming.landmarks_file,
                                 "landmark_threshold": self.threshold}),
            features=FeaturesConfig(landmarks_verify=self.verify,
                                    landmark_candidate_threshold=self.candidate),
        )
        self.answers: dict[str, str] = {}
        self.asked: list[str] = []
        self.raises: set[str] = set()

    def says(self, path: str, answer: str) -> str:
        self.answers[path] = answer
        return path

    def ask(self, path: str) -> str:
        """The model. Answers what the test scripted, or names nothing."""
        self.asked.append(path)
        if path in self.raises:
            raise Boom("no runtime")
        return self.answers.get(path, SAYS_NOTHING)

    def run_stage(self, asker=..., **kwargs):
        if asker is ...:
            asker = self.ask
        return detect_landmarks(self.cfg, self.conn,
                                classifier=PathClassifier(self.scores),
                                resolver=self.resolver, asker=asker, **kwargs)

    def checks(self) -> dict[tuple[str, str], tuple[str, float]]:
        """(path, proposed landmark) -> (verdict, score), out of `landmark_checks`."""
        by_id = {file_id: path for path, file_id in self.ids.items()}
        return {(by_id[r["file_id"]], r["landmark"]): (r["verdict"], r["score"])
                for r in self.conn.execute(
                    "SELECT file_id, landmark, verdict, score FROM landmark_checks")}


class TestToggleOff(VerifyCase):
    """Criterion one: with the check off the run is today's, in every respect."""

    verify = False

    def test_the_model_is_never_asked(self) -> None:
        self.add("/photos/DCIM", PRAGUE, prob=0.95)
        self.add("/photos/DCIM", PRAGUE, prob=0.60)
        self.run_stage()
        self.assertEqual(self.asked, [])

    def test_the_gate_stays_at_todays_threshold(self) -> None:
        """A proposal in the band the check would have opened must not slip through."""
        band = self.add("/photos/DCIM", PRAGUE, prob=0.60)
        strong = self.add("/photos/DCIM", PRAGUE, prob=0.95)
        stats = self.run_stage()
        self.assertEqual(stats.matched, 1)
        self.assertEqual(self.place_of(band)[3], "unknown")
        self.assertEqual(self.place_of(strong)[3], "visual")

    def test_nothing_is_remembered_because_nothing_was_asked(self) -> None:
        self.add("/photos/DCIM", PRAGUE, prob=0.95)
        self.run_stage()
        self.assertEqual(self.checks(), {})

    def test_the_check_counters_are_all_zero(self) -> None:
        self.add("/photos/DCIM", PRAGUE, prob=0.95)
        stats = self.run_stage()
        self.assertEqual(
            (stats.checked, stats.checks_reused, stats.confirmed_by_model,
             stats.rejected_by_model, stats.checks_failed), (0, 0, 0, 0, 0))

    def test_the_same_input_gives_the_same_places_as_with_the_check_on(self) -> None:
        """The one comparison the criterion is actually about: identical results.

        The model confirms everything here, so the check changes nothing it is allowed to
        change — and the two runs must then agree file for file.
        """
        for prob in (0.95, 0.90):
            self.says(self.add("/photos/DCIM", PRAGUE, prob=prob), SAYS_PRAGUE)
        without = self.run_stage()
        self.conn.execute("UPDATE places SET country = NULL, city = NULL, "
                          "city_geonameid = NULL, confidence = 'unknown'")
        self.conn.commit()
        self.cfg = dataclasses.replace(
            self.cfg, features=FeaturesConfig(landmarks_verify=True,
                                              landmark_candidate_threshold=0.5))
        with_check = self.run_stage()
        self.assertEqual(without.matched, with_check.matched)
        self.assertEqual(self.cities(), {"Prague": 2})


class TestOnlyTheBandIsAsked(VerifyCase):
    """Criterion two: the check runs over proposals above the NEW threshold."""

    def test_a_proposal_below_the_candidate_gate_is_not_a_proposal(self) -> None:
        weak = self.add("/photos/DCIM", PRAGUE, prob=0.40)
        self.run_stage()
        self.assertEqual(self.asked, [])
        self.assertEqual(self.place_of(weak)[3], "unknown")

    def test_the_band_the_old_threshold_would_have_dropped_is_asked_about(self) -> None:
        band = self.says(self.add("/photos/DCIM", PRAGUE, prob=0.60), SAYS_PRAGUE)
        stats = self.run_stage()
        self.assertEqual(self.asked, [band])
        self.assertEqual(self.place_of(band), ("CZ", "Prague", 3067696, "visual"))
        self.assertEqual(stats.matched, 1)

    def test_a_proposal_above_the_old_threshold_is_checked_too(self) -> None:
        """"Every proposal is checked" — the confident ones are not exempt."""
        strong = self.says(self.add("/photos/DCIM", PRAGUE, prob=0.99), SAYS_PRAGUE)
        self.run_stage()
        self.assertEqual(self.asked, [strong])

    def test_the_candidate_gate_never_narrows_the_population(self) -> None:
        """A gate set above today's threshold is clamped back down to it.

        Otherwise switching the check on would LOSE frames the stage already places,
        which is the one thing it must not do.
        """
        self.cfg = dataclasses.replace(
            self.cfg, features=FeaturesConfig(landmarks_verify=True,
                                              landmark_candidate_threshold=0.99))
        strong = self.says(self.add("/photos/DCIM", PRAGUE, prob=0.90), SAYS_PRAGUE)
        stats = self.run_stage()
        self.assertEqual(self.asked, [strong])
        self.assertEqual(stats.matched, 1)

    def test_the_proposals_are_counted_before_the_check_removes_any(self) -> None:
        self.says(self.add("/photos/DCIM", PRAGUE, prob=0.60), SAYS_PRAGUE)
        self.add("/photos/DCIM", PRAGUE, prob=0.60)          # answers "none"
        stats = self.run_stage()
        self.assertEqual((stats.proposals, stats.checked), (2, 2))
        self.assertEqual((stats.confirmed_by_model, stats.rejected_by_model), (1, 1))


class TestSilenceIsARejection(VerifyCase):
    """Brief §2a: an empty answer is "not confirmed" — the common, healthy outcome.

    71 of the 104 probe answers named nothing, and that is precisely why no wrong city was
    ever confirmed. Reading silence as "could not parse" and falling back to the CLIP rule
    would hand the whole band back to the threshold that cannot split it.
    """

    def test_an_empty_answer_drops_even_a_confident_proposal(self) -> None:
        strong = self.says(self.add("/photos/DCIM", PRAGUE, prob=0.99), SILENCE)
        stats = self.run_stage()
        self.assertEqual(stats.matched, 0)
        self.assertEqual(self.place_of(strong)[3], "unknown")

    def test_saying_none_is_the_same_as_saying_nothing(self) -> None:
        quiet = self.says(self.add("/photos/DCIM", PRAGUE, prob=0.99), SAYS_NOTHING)
        self.run_stage()
        self.assertEqual(self.place_of(quiet)[3], "unknown")

    def test_a_rejection_is_remembered_as_one(self) -> None:
        path = self.says(self.add("/photos/DCIM", PRAGUE, prob=0.99), SILENCE)
        self.run_stage()
        verdict, score = self.checks()[(path, "Карлов мост")]
        self.assertEqual(verdict, CHECK_REJECTED)
        self.assertAlmostEqual(score, 0.99, places=5)


class TestModelFailureFallsBackToTheOldRule(VerifyCase):
    """Criterion three: an unavailable tier must not lose what the cheap one finds."""

    def test_a_raising_model_leaves_the_clip_threshold_in_charge(self) -> None:
        strong = self.add("/photos/DCIM", PRAGUE, prob=0.95)
        self.raises.add(strong)
        stats = self.run_stage()
        self.assertEqual(stats.checks_failed, 1)
        self.assertEqual(self.place_of(strong), ("CZ", "Prague", 3067696, "visual"))

    def test_the_band_below_the_old_threshold_is_not_kept_on_a_failure(self) -> None:
        """The fallback is the OLD rule, not "keep everything": 0.60 was never enough."""
        band = self.add("/photos/DCIM", PRAGUE, prob=0.60)
        self.raises.add(band)
        stats = self.run_stage()
        self.assertEqual(stats.matched, 0)
        self.assertEqual(self.place_of(band)[3], "unknown")

    def test_a_failure_stores_nothing_because_nothing_was_learned(self) -> None:
        path = self.add("/photos/DCIM", PRAGUE, prob=0.95)
        self.raises.add(path)
        self.run_stage()
        self.assertEqual(self.checks(), {})

    def test_one_frame_failing_does_not_cost_the_others(self) -> None:
        broken = self.add("/photos/DCIM", PRAGUE, prob=0.95)
        self.raises.add(broken)
        answered = self.says(self.add("/photos/DCIM", PRAGUE, prob=0.60), SAYS_PRAGUE)
        stats = self.run_stage()
        self.assertEqual(stats.matched, 2)
        self.assertEqual(self.place_of(answered)[3], "visual")

    def test_a_model_that_will_not_build_leaves_the_gate_where_it_was(self) -> None:
        """The widening is tied to the check EXISTING: a wide band nobody checks is worse
        than no feature at all."""
        def factory(model_name):
            raise Boom("no weights")

        band = self.add("/photos/DCIM", PRAGUE, prob=0.60)
        strong = self.add("/photos/DCIM", PRAGUE, prob=0.95)
        stats = self.run_stage(asker=None, asker_factory=factory)
        self.assertEqual(self.asked, [])
        self.assertEqual(stats.matched, 1)
        self.assertEqual(self.place_of(band)[3], "unknown")
        self.assertEqual(self.place_of(strong)[3], "visual")


class TestCorroborationStillDecides(VerifyCase):
    """The main safety test: the model is a filter BEFORE F75, never a way past it."""

    language = "ru"

    def test_a_country_in_the_path_refutes_what_the_model_confirmed(self) -> None:
        """The case the user caught by eye. No agreement of two models overrules it."""
        path = self.says(self.add("/photos/Франция/DCIM", PRAGUE, prob=0.99), SAYS_PRAGUE)
        stats = self.run_stage()
        self.assertEqual(self.asked, [path])
        self.assertEqual(stats.confirmed_by_model, 1)
        self.assertEqual(stats.dropped_by_folder_name, 1)
        self.assertEqual(stats.matched, 0)
        self.assertEqual(self.place_of(path), (None, None, None, "unknown"))

    def test_the_group_rule_still_discards_the_odd_city_out(self) -> None:
        for _ in range(8):
            self.says(self.add("/photos/DCIM/100D3300", PRAGUE, prob=0.95), SAYS_PRAGUE)
        odd = self.says(self.add("/photos/DCIM/100D3300", BERLIN, prob=0.95), SAYS_BERLIN)
        stats = self.run_stage()
        self.assertEqual(stats.confirmed_by_model, 9)
        self.assertEqual(stats.dropped_by_group, 1)
        self.assertEqual(self.cities(), {"Prague": 8})
        self.assertEqual(self.place_of(odd), (None, None, None, "unknown"))

    def test_a_refuted_frame_is_remembered_as_confirmed_not_as_rejected(self) -> None:
        """The table records what the MODEL said; where the file went is `places`.

        Keeping the two apart is what stops a corroboration verdict from being re-read
        later as the model's opinion.
        """
        path = self.says(self.add("/photos/Франция/DCIM", PRAGUE, prob=0.99), SAYS_PRAGUE)
        self.run_stage()
        self.assertEqual(self.checks()[(path, "Карлов мост")][0], CHECK_CONFIRMED)


class TestARejectedFrameStaysUnknown(VerifyCase):
    """Criterion six, and the F75 invariant underneath the whole stage."""

    def test_the_place_the_model_named_is_not_written_anywhere(self) -> None:
        path = self.says(self.add("/photos/DCIM", PRAGUE, prob=0.99), SAYS_BERLIN)
        stats = self.run_stage()
        self.assertEqual(stats.rejected_by_model, 1)
        self.assertEqual(stats.matched, 0)
        self.assertEqual(self.place_of(path), (None, None, None, "unknown"))
        self.assertEqual(self.cities(), {})

    def test_naming_a_third_place_is_a_rejection_too(self) -> None:
        path = self.says(self.add("/photos/DCIM", PRAGUE, prob=0.99),
                         "the Eiffel Tower in Paris")
        self.run_stage()
        self.assertEqual(self.place_of(path), (None, None, None, "unknown"))

    def test_a_rejected_frame_does_not_block_a_confirmed_neighbour(self) -> None:
        wrong = self.says(self.add("/photos/DCIM", PRAGUE, prob=0.99), SAYS_BERLIN)
        right = self.says(self.add("/photos/DCIM", PARIS, prob=0.99),
                          "the Eiffel Tower in Paris")
        self.run_stage()
        self.assertEqual(self.place_of(wrong)[3], "unknown")
        self.assertEqual(self.place_of(right)[3], "visual")


class TestTheAnswerIsRemembered(VerifyCase):
    """Criterion five: a rejected frame comes back every run — it must not be re-asked."""

    def test_a_second_run_asks_nothing_new(self) -> None:
        rejected = self.says(self.add("/photos/DCIM", PRAGUE, prob=0.99), SAYS_BERLIN)
        self.run_stage()
        self.assertEqual(self.asked, [rejected])
        stats = self.run_stage()
        self.assertEqual(self.asked, [rejected])          # not asked a second time
        self.assertEqual((stats.checked, stats.checks_reused), (0, 1))
        self.assertEqual(self.place_of(rejected)[3], "unknown")

    def test_a_remembered_confirmation_still_carries_the_whole_decision(self) -> None:
        """A reused verdict places the file — it is not merely a "do not ask" marker.

        The frame gets here the only way a confirmed one can come back at all: the group
        rule dropped it while its eight neighbours were placed. On the second run those
        eight are no longer `unknown`, so the group is too small for the rule to fire and
        the stored confirmation is all there is to go on.
        """
        for _ in range(8):
            self.says(self.add("/photos/DCIM/100D3300", PRAGUE, prob=0.95), SAYS_PRAGUE)
        odd = self.says(self.add("/photos/DCIM/100D3300", BERLIN, prob=0.95), SAYS_BERLIN)
        self.run_stage()
        self.assertEqual(self.place_of(odd)[3], "unknown")
        self.asked.clear()
        stats = self.run_stage()
        self.assertEqual(self.asked, [])
        self.assertEqual((stats.checked, stats.checks_reused), (0, 1))
        self.assertEqual(self.place_of(odd), ("DE", "Berlin", 2950159, "visual"))

    def test_a_different_proposal_for_the_same_frame_is_a_new_question(self) -> None:
        path = self.says(self.add("/photos/DCIM", PRAGUE, prob=0.99), SAYS_BERLIN)
        self.run_stage()
        self.scores[path] = (PARIS, 0.99)                  # CLIP changed its mind
        self.run_stage()
        self.assertEqual(self.asked, [path, path])
        self.assertEqual(set(self.checks()),
                         {(path, "Карлов мост"), (path, "Эйфелева башня")})

    def test_a_reworded_question_invalidates_the_stored_answers(self) -> None:
        path = self.says(self.add("/photos/DCIM", PRAGUE, prob=0.99), SAYS_BERLIN)
        self.run_stage()
        self.conn.execute("UPDATE landmark_checks SET model = 'other#deadbeef'")
        self.conn.commit()
        self.run_stage()
        self.assertEqual(self.asked, [path, path])


class TestTheStoredMarker(unittest.TestCase):
    """What makes a stored answer stale — the group_keeper device (F120/F130/F132)."""

    def test_the_marker_names_the_model_and_fingerprints_the_question(self) -> None:
        marker = landmark_check_model("Qwen/Qwen2.5-VL-3B-Instruct")
        model, _, fingerprint = marker.partition("#")
        self.assertEqual(model, "Qwen/Qwen2.5-VL-3B-Instruct")
        self.assertEqual(len(fingerprint), 8)

    def test_two_models_are_two_markers(self) -> None:
        self.assertNotEqual(landmark_check_model("a"), landmark_check_model("b"))

    def test_the_same_model_and_question_give_the_same_marker(self) -> None:
        self.assertEqual(landmark_check_model("a"), landmark_check_model("a"))


class TestTheQuestion(unittest.TestCase):
    """The wording is the measured half of this feature, so it is pinned here."""

    def test_the_question_never_names_a_place(self) -> None:
        """Naming the proposal is what turns a check into an agreement."""
        for word in ("Charles", "Prague", "Eiffel", "Berlin", "{"):
            self.assertNotIn(word, LANDMARK_NAMING_PROMPT)

    def test_it_is_the_open_form_and_not_the_yes_no_one(self) -> None:
        """`verify` was half as good on the right proposals in all three probe runs."""
        self.assertNotIn("yes or no", LANDMARK_NAMING_PROMPT)
        self.assertIn("name of that place", LANDMARK_NAMING_PROMPT)


if __name__ == "__main__":
    unittest.main()
