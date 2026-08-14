"""F137: "there is an animal here" is DERIVED when read, not stored when written.

`frame_quality.pet` is a cache from now on. The verdict is computed at the moment of the
read out of what the stage stores — `pet_score`, `pet_vlm` — and out of the thresholds the
config holds right then, because that is the property those two columns are kept for: a
threshold chosen from a distribution has to be re-choosable without another pass over the
collection. The thresholds are deliberately not part of `quality_prompt_fingerprint`
(hashing them would send sharpness, CLIP and the VLM round again on every edit), so before
this feature an edited threshold left the database behind without a word — the live archive
kept 966 animals selected at a 0.30 candidate gate long after the gate went back to 0.50,
where the stored answers say 848.

So the file is built around one question: does the answer follow the CONFIG or the stored
label. Every case here writes a `frame_quality` row once, never re-runs anything, and only
moves thresholds; the last class writes a deliberately WRONG `pet` and checks that no
consumer notices.
"""
from __future__ import annotations

import dataclasses
import io
import json
import sqlite3
import unittest
from contextlib import redirect_stdout

from sorta import ui
from sorta.junk import PET_VLM_DEPICTION, PET_VLM_NONE, PET_VLM_REAL
from sorta.sorter import plan_album

from tests import waiting
from tests.test_sorter_album_animal import AnimalAlbumTestBase
from tests.test_ui_animals import AnimalsTestBase

# The thresholds a case moves through. Wide on purpose: "at any threshold" is the claim
# the model's answer makes, and a single value cannot pin it.
_THRESHOLD_GRID = (0.05, 0.3, 0.5, 0.7, 0.95)


def write_quality(conn: sqlite3.Connection, file_id: int, *, score: float | None,
                  vlm: str | None = None, pet: str | None = "animal") -> None:
    """One `frame_quality` row, as the junk stage leaves it.

    `pet` is written explicitly rather than derived, because half of this file is about a
    row whose cached label and stored numbers disagree — the state a threshold edit leaves
    behind, and the one the feature has to make harmless.
    """
    conn.execute(
        """INSERT INTO frame_quality (file_id, sharpness, pet, pet_score, pet_vlm,
               source, updated_at)
           VALUES (?, 100.0, ?, ?, ?, 'clip', '2026-01-01')""",
        (file_id, pet, score, vlm))
    conn.commit()


class DerivedRuleCase(AnimalAlbumTestBase):
    """The album slice over frames written once and read at moving thresholds."""

    def store(self, name: str, *, score: float | None, vlm: str | None = None,
              pet: str | None = "animal") -> int:
        file_id = self.add_file(name)
        write_quality(self.conn, file_id, score=score, vlm=vlm, pet=pet)
        return file_id

    def thresholds(self, **kwargs: float) -> None:
        """Edit the config, and nothing else — no stage runs in this file."""
        self.cfg.features = dataclasses.replace(self.cfg.features, **kwargs)

    def slice_ids(self) -> list[int]:
        with redirect_stdout(io.StringIO()):
            report = plan_album(self.cfg, self.conn, "animal", "", self.dest)
        return sorted(it.file_id for it in report.plan)

    def quality_snapshot(self) -> list[tuple]:
        return [tuple(r) for r in self.conn.execute(
            "SELECT file_id, pet, pet_score, pet_vlm, source, updated_at "
            "FROM frame_quality ORDER BY file_id")]


class TestTheThresholdIsAppliedAtRead(DerivedRuleCase):
    """Brief test 1 — the main test of the feature."""

    def test_one_frame_between_two_thresholds_moves_with_the_config(self):
        fid = self.store("cat.jpg", score=0.65, pet=None)
        self.thresholds(pet_threshold=0.60)
        self.assertEqual(self.slice_ids(), [fid])
        self.thresholds(pet_threshold=0.70)
        self.assertEqual(self.slice_ids(), [])

    def test_and_moves_it_without_a_single_write(self):
        """"Without a recompute" is the claim, so it is checked as one: the stored row is
        the same row, `pet` included, on both sides of the answer changing."""
        self.store("cat.jpg", score=0.65, pet=None)
        self.thresholds(pet_threshold=0.60)
        before = self.quality_snapshot()
        self.assertEqual(len(self.slice_ids()), 1)
        self.thresholds(pet_threshold=0.70)
        self.assertEqual(self.slice_ids(), [])
        self.assertEqual(self.quality_snapshot(), before)

    def test_the_threshold_is_inclusive_here_too(self):
        """`pet_score >= threshold`, the same comparison `junk.pet_label` writes with."""
        fid = self.store("cat.jpg", score=0.70, pet=None)
        self.thresholds(pet_threshold=0.70)
        self.assertEqual(self.slice_ids(), [fid])

    def test_a_frame_the_stage_never_scored_is_not_an_animal_at_any_threshold(self):
        self.store("no_score.jpg", score=None, pet=None)
        self.add_file("no_row_at_all.jpg")
        for threshold in _THRESHOLD_GRID:
            with self.subTest(threshold=threshold):
                self.thresholds(pet_threshold=threshold)
                self.assertEqual(self.slice_ids(), [])


class TestTheCandidateThresholdIsAppliedAtRead(DerivedRuleCase):
    """The 966-vs-848 case the feature was written for.

    The answers of the check are stored for every frame it was ever shown, and F130 chose
    its gate by re-reading them at a higher one (its 0.30 → 0.50 rows are a replay, not a
    second pass). The gate belongs to the read for exactly that reason: a stored answer is
    in force for a frame the CURRENT `pet_candidate_threshold` would still show the model,
    and the frames of a gate that has since been raised fall back to the score. In a
    database whose answers came from the config now in force this changes nothing — every
    frame that has an answer cleared the gate to get one.
    """

    def test_an_answer_from_a_lower_gate_stops_counting_when_the_gate_goes_back_up(self):
        below = self.store("dark_cat.jpg", score=0.35, vlm=PET_VLM_REAL)
        self.thresholds(pet_threshold=0.70, pet_candidate_threshold=0.30)
        self.assertEqual(self.slice_ids(), [below])
        self.thresholds(pet_candidate_threshold=0.50)
        self.assertEqual(self.slice_ids(), [])
        self.thresholds(pet_candidate_threshold=0.30)
        self.assertEqual(self.slice_ids(), [below])

    def test_a_gated_out_answer_falls_back_to_the_score_rather_than_to_no(self):
        """The fallback is the rule that ran before the check existed, never a guess."""
        fid = self.store("cat.jpg", score=0.80, vlm=PET_VLM_REAL)
        self.thresholds(pet_threshold=0.70, pet_candidate_threshold=0.95)
        self.assertEqual(self.slice_ids(), [fid])

    def test_the_frames_above_the_gate_are_untouched_by_it(self):
        toy = self.store("toy.jpg", score=0.95, vlm=PET_VLM_DEPICTION)
        cat = self.store("cat.jpg", score=0.90, vlm=PET_VLM_REAL)
        for gate in (0.1, 0.3, 0.5, 0.7):
            with self.subTest(gate=gate):
                self.thresholds(pet_threshold=0.70, pet_candidate_threshold=gate)
                self.assertEqual(self.slice_ids(), [cat])
                self.assertNotIn(toy, self.slice_ids())


class TestTheAnswerOutranksTheScore(DerivedRuleCase):
    """Brief tests 2 and 3: the F130 cascade, unchanged by this feature."""

    def test_a_live_animal_below_the_threshold_is_an_animal_at_any_threshold(self):
        fid = self.store("dark_cat.jpg", score=0.35, vlm=PET_VLM_REAL, pet="animal")
        for threshold in _THRESHOLD_GRID:
            with self.subTest(threshold=threshold):
                self.thresholds(pet_threshold=threshold, pet_candidate_threshold=0.30)
                self.assertEqual(self.slice_ids(), [fid])

    def test_a_rejected_frame_above_the_threshold_is_not_an_animal_at_any_threshold(self):
        for name, answer in (("toy.jpg", PET_VLM_DEPICTION), ("coat.jpg", PET_VLM_NONE)):
            self.store(name, score=0.95, vlm=answer, pet=None)
        for threshold in _THRESHOLD_GRID:
            with self.subTest(threshold=threshold):
                self.thresholds(pet_threshold=threshold, pet_candidate_threshold=0.30)
                self.assertEqual(self.slice_ids(), [])

    def test_an_unanswered_frame_falls_back_to_the_threshold(self):
        """NULL is "not asked", not "no" — the distinction the check is built on."""
        fid = self.store("cat.jpg", score=0.80, vlm=None, pet=None)
        self.thresholds(pet_threshold=0.70)
        self.assertEqual(self.slice_ids(), [fid])
        self.thresholds(pet_threshold=0.90)
        self.assertEqual(self.slice_ids(), [])


class TestTheManualVerdictStillWins(DerivedRuleCase):
    """Brief test 4: F124 is not broken, in either direction and at any threshold."""

    def mark_by_hand(self, file_id: int, is_animal: bool) -> None:
        self.conn.execute(
            "INSERT INTO manual_pet (file_id, is_animal, updated_at) VALUES (?, ?, 'x')",
            (file_id, 1 if is_animal else 0))
        self.conn.commit()

    def test_not_an_animal_beats_a_confident_score_and_a_real_answer(self):
        fid = self.store("fur_coat.jpg", score=0.95, vlm=PET_VLM_REAL)
        self.mark_by_hand(fid, is_animal=False)
        for threshold in _THRESHOLD_GRID:
            with self.subTest(threshold=threshold):
                self.thresholds(pet_threshold=threshold, pet_candidate_threshold=0.30)
                self.assertEqual(self.slice_ids(), [])

    def test_it_is_an_animal_beats_a_low_score_and_a_rejection(self):
        fid = self.store("dark_cat.jpg", score=0.10, vlm=PET_VLM_NONE, pet=None)
        self.mark_by_hand(fid, is_animal=True)
        for threshold in _THRESHOLD_GRID:
            with self.subTest(threshold=threshold):
                self.thresholds(pet_threshold=threshold, pet_candidate_threshold=0.30)
                self.assertEqual(self.slice_ids(), [fid])

    def test_a_frame_with_no_quality_row_is_still_added_by_hand(self):
        fid = self.add_file("never_asked.jpg")
        self.mark_by_hand(fid, is_animal=True)
        self.assertEqual(self.slice_ids(), [fid])


class TestTheStoredLabelDecidesNothing(DerivedRuleCase):
    """Brief test 5, the album half: `pet` is a cache, and a stale one changes no answer.

    Both directions of staleness are here, because they are two different mistakes: a label
    left behind by a threshold that has since risen, and a missing label under a threshold
    that has since fallen.
    """

    def test_a_stale_animal_label_selects_nothing_on_its_own(self):
        self.store("desk.jpg", score=0.20, pet="animal")
        self.thresholds(pet_threshold=0.70)
        self.assertEqual(self.slice_ids(), [])

    def test_a_missing_label_hides_nothing(self):
        fid = self.store("cat.jpg", score=0.90, pet=None)
        self.thresholds(pet_threshold=0.70)
        self.assertEqual(self.slice_ids(), [fid])

    def test_a_label_that_contradicts_the_stored_answer_is_ignored(self):
        """The cache of a run whose check answered after the label was written."""
        self.store("toy.jpg", score=0.95, vlm=PET_VLM_DEPICTION, pet="animal")
        cat = self.store("dark_cat.jpg", score=0.35, vlm=PET_VLM_REAL, pet=None)
        self.thresholds(pet_threshold=0.70, pet_candidate_threshold=0.30)
        self.assertEqual(self.slice_ids(), [cat])


class DerivedRuleUiCase(AnimalsTestBase):
    """The same rule through the web app: the tab, its counter and "Overview"."""

    def store(self, name: str, *, score: float | None, vlm: str | None = None,
              pet: str | None = "animal") -> int:
        file_id, _path, _content = self.add_photo_file(name)
        write_quality(self.conn, file_id, score=score, vlm=vlm, pet=pet)
        return file_id

    def thresholds(self, **kwargs: float) -> None:
        """Edit the RUNNING config — the object the server reads per request. Nothing is
        restarted and nothing is re-run: that is the whole claim being made."""
        self.cfg.features = dataclasses.replace(self.cfg.features, **kwargs)

    def listed_ids(self) -> list[int]:
        return sorted(it["file_id"] for it in self.animals()["items"])

    def tab_animal_ids(self) -> list[int]:
        return sorted(it["file_id"] for it in self.animals()["items"] if it["is_animal"])

    def overview_animals(self) -> int:
        _status, body, _ctype = self.get("/api/overview")
        return json.loads(body)["collection"]["animals"]

    def tab_visible(self) -> bool:
        _status, body, _ctype = self.get("/api/tabs/visibility")
        return bool(json.loads(body)["animal"])

    def album_ids(self) -> list[int]:
        with redirect_stdout(io.StringIO()):
            report = plan_album(self.cfg, self.conn, "animal", "", self.root / "album")
        return sorted(it.file_id for it in report.plan)


class TestTheWebAppReadsTheSameRule(DerivedRuleUiCase):
    """Brief test 6: one collection, four answers, and they are one number."""

    def collection(self) -> dict[str, int]:
        """A frame of every kind the rule decides, and one stale label among them."""
        return {
            "cat.jpg": self.store("cat.jpg", score=0.95),
            "toy.jpg": self.store("toy.jpg", score=0.93, vlm=PET_VLM_DEPICTION,
                                  pet="animal"),          # stale: the answer says no
            "dark_cat.jpg": self.store("dark_cat.jpg", score=0.55, vlm=PET_VLM_REAL,
                                       pet=None),         # stale: the answer says yes
            "coat.jpg": self.store("coat.jpg", score=0.65, pet=None),
            "desk.jpg": self.store("desk.jpg", score=0.05, pet=None),
        }

    def test_the_tab_the_counter_and_the_album_agree(self):
        ids = self.collection()
        self.start_server()
        expected = sorted([ids["cat.jpg"], ids["dark_cat.jpg"]])
        self.assertEqual(self.tab_animal_ids(), expected)
        self.assertEqual(self.album_ids(), expected)
        self.assertEqual(self.animals()["animals"], len(expected))
        self.assertEqual(self.overview_animals(), len(expected))

    def test_a_lower_threshold_moves_all_three_at_once_and_without_a_run(self):
        """The acceptance criterion of the feature, spelled as a test: the number the
        user sees follows `features.pet_threshold` with nothing re-run in between."""
        ids = self.collection()
        self.start_server()
        self.assertEqual(self.overview_animals(), 2)
        self.thresholds(pet_threshold=0.60)     # the coat clears it now
        expected = sorted([ids["cat.jpg"], ids["dark_cat.jpg"], ids["coat.jpg"]])
        self.assertEqual(self.tab_animal_ids(), expected)
        self.assertEqual(self.album_ids(), expected)
        self.assertEqual(self.animals()["animals"], 3)
        self.assertEqual(self.overview_animals(), 3)

    def test_a_higher_candidate_threshold_moves_them_too(self):
        """The 0.30 → 0.50 rollback: the frame the check reached under the old gate goes
        back to being decided by its score, everywhere at once."""
        ids = self.collection()
        self.start_server()
        self.assertIn(ids["dark_cat.jpg"], self.tab_animal_ids())
        self.thresholds(pet_candidate_threshold=0.60)
        expected = [ids["cat.jpg"]]
        self.assertEqual(self.tab_animal_ids(), expected)
        self.assertEqual(self.album_ids(), expected)
        self.assertEqual(self.overview_animals(), 1)

    def test_the_settings_panel_is_the_same_edit(self):
        """`features.pet_threshold` is a knob of the settings column (F104), and it puts
        the value into the RUNNING config — so the whole chain from the form to the
        counter has no run in it either. `_apply_settings` is the function the route
        calls; going through it is what makes this a test of the product and not of
        `dataclasses.replace`."""
        ids = self.collection()
        self.start_server()
        self.assertEqual(self.overview_animals(), 2)
        ui._apply_settings(self.cfg, {"features.pet_threshold": 0.60})
        self.assertEqual(self.overview_animals(), 3)
        self.assertIn(ids["coat.jpg"], self.tab_animal_ids())

    def test_the_page_lists_what_the_thresholds_select(self):
        """The list follows the rule as well as the counter does — a card the thresholds
        have withdrawn must not sit on the page claiming to be an animal."""
        ids = self.collection()
        self.start_server()
        self.assertEqual(self.listed_ids(),
                         sorted([ids["cat.jpg"], ids["dark_cat.jpg"]]))
        self.thresholds(pet_threshold=0.60)
        self.assertEqual(
            self.listed_ids(),
            sorted([ids["cat.jpg"], ids["dark_cat.jpg"], ids["coat.jpg"]]))

    def test_a_marked_frame_stays_on_the_page_at_every_threshold(self):
        """F124's own rule: a hand-made decision is visible so it can be taken back."""
        ids = self.collection()
        self.start_server()
        status, _resp = self.post_mark(ids["coat.jpg"], "not_animal")
        self.assertEqual(status, 200)
        for threshold in (0.10, 0.60, 0.95):
            with self.subTest(threshold=threshold):
                self.thresholds(pet_threshold=threshold)
                self.assertIn(ids["coat.jpg"], self.listed_ids())
                self.assertNotIn(ids["coat.jpg"], self.tab_animal_ids())

    def post_mark(self, file_id: int, action: str) -> tuple[int, dict]:
        answer = waiting.post_json(f"{self.base_url}/api/animals/mark",
                                   {"file_ids": [file_id], "action": action})
        return answer.status, answer.json()


class TestTheTabAppearsByTheSameRule(DerivedRuleUiCase):
    def test_hidden_when_the_thresholds_leave_nothing_to_show(self):
        """A stale `pet` used to open a tab with an empty page under it."""
        self.store("desk.jpg", score=0.20, pet="animal")
        self.start_server()
        self.assertFalse(self.tab_visible())
        self.assertEqual(self.animals()["total"], 0)

    def test_shown_again_once_the_threshold_reaches_the_frame(self):
        self.store("desk.jpg", score=0.20, pet=None)
        self.start_server()
        self.assertFalse(self.tab_visible())
        self.thresholds(pet_threshold=0.15)
        self.assertTrue(self.tab_visible())
        self.assertEqual(self.animals()["total"], 1)


if __name__ == "__main__":
    unittest.main()
