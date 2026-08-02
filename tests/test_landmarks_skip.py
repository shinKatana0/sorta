"""F136: the stage stops recomputing what has not changed.

`landmarks` was incremental in its SELECTION only — a match leaves as
`confidence='visual'`, everything else keeps `'unknown'` and was handed to CLIP again on
every run: 7 619 frames and 138 s of a 176 s run on the live collection, for an answer
that could not have come out differently. The interface used to work around it with a
button; F135 removed the button, so the stage has to do it itself.

The two properties that decide whether this is a saving or a silent bug:

* a stored answer must not outlive what produced it — a new file, an edited list, a moved
  threshold, a reworded prompt all have to bring the frame back to CLIP (the F120 device);
* corroboration must see the SAME set of matches a full run would have built. It is not a
  per-file rule: the group rule reads the company a match keeps, so skipping a file that
  proposed something thins out its folder and changes the verdict of its neighbours (F75).
  That is the one failure this feature could hide, and `TestCorroborationOverThePartialSet`
  is the test written against it — a run that skips is compared with a full run over the
  very same selection, not with a guess about what it should say.

CLIP is a table of scores, as everywhere in these tests; here it also counts the frames it
was shown, because "the stage did not run" is the actual claim.
"""
from __future__ import annotations

import dataclasses
import sqlite3
from pathlib import Path
from unittest import mock

import numpy as np

from sorta.config import _naming_from
from sorta.landmarks import (
    _SCAN_KEY as SCAN_KEY,
    _SCAN_NONE as SCAN_NONE,
    _stage_fingerprint,
    detect_landmarks,
    landmark_prompts,
    load_landmarks,
)
from tests.test_landmark_corroboration import (
    BERLIN,
    PARIS,
    PRAGUE,
    CorroborationCase,
    PathClassifier,
)
from tests.test_landmarks_verify import SAYS_BERLIN, SAYS_PRAGUE, VerifyCase


class CountingClassifier(PathClassifier):
    """The F75 fake CLIP, plus a record of every frame it was handed."""

    def __init__(self, scores: dict[str, tuple[int, float]]) -> None:
        super().__init__(scores)
        self.seen: list[str] = []

    def __call__(self, image_paths: list[str], prompts: list[str]) -> np.ndarray:
        self.seen.extend(image_paths)
        return super().__call__(image_paths, prompts)


def _places(conn: sqlite3.Connection) -> dict[str, tuple]:
    """path -> what the stage decided about it, for the whole DB."""
    return {r["path"]: (r["country"], r["city"], r["city_geonameid"], r["confidence"])
            for r in conn.execute(
                """SELECT f.path, p.country, p.city, p.city_geonameid, p.confidence
                   FROM places p JOIN files f ON f.id = p.file_id ORDER BY f.id""")}


class SkipCase(CorroborationCase):
    """The F75 fixture with a CLIP that counts, and the small edits a run can survive."""

    def run_stage(self, **kwargs):
        self.clip = CountingClassifier(self.scores)
        return detect_landmarks(self.cfg, self.conn, classifier=self.clip,
                                resolver=self.resolver, **kwargs)

    @property
    def seen(self) -> list[str]:
        """The frames the last run showed CLIP."""
        return self.clip.seen

    def touch(self, path: str) -> None:
        """The file changed on disk — the index says so, which is where we read it."""
        self.conn.execute("UPDATE files SET mtime = mtime + 1, size = size + 1"
                          " WHERE id = ?", (self.ids[path],))
        self.conn.commit()

    def scans(self) -> dict[str, tuple[str, float | None]]:
        """path -> (what CLIP proposed, the score), out of the stage's own rows."""
        by_id = {file_id: path for path, file_id in self.ids.items()}
        return {by_id[r["file_id"]]: (r["verdict"], r["score"])
                for r in self.conn.execute(
                    "SELECT file_id, verdict, score FROM landmark_checks WHERE landmark = ?",
                    (SCAN_KEY,))}


class TestASecondRunLooksAtNothing(SkipCase):
    """Criterion one: nothing changed, so nothing is computed — and nothing moves."""

    def _collection(self) -> None:
        """One of each: placed, matched-nothing, and refuted by the path.

        The last two are what the selection consists of on every later run, and they are
        decided by rules that do not depend on who else is in it — so "the same result"
        is a claim about this feature and not about the group rule seeing a smaller
        folder, which is how a re-run has always behaved.
        """
        self.add("/photos/Чехия/DCIM", PRAGUE, n=8)      # placed: the path confirms CZ
        self.add("/photos/Финляндия/DCIM", -1, prob=0.95, n=2)   # nothing above the gate
        self.add("/photos/Финляндия/DCIM", BERLIN, n=1)  # the path says Finland

    def test_the_classifier_is_not_called_at_all(self) -> None:
        self._collection()
        self.run_stage()
        self.assertEqual(len(self.seen), 11)
        stats = self.run_stage()
        self.assertEqual(self.seen, [])
        self.assertEqual((stats.scanned, stats.skipped), (3, 3))  # the 8 matches left

    def test_the_result_is_the_same(self) -> None:
        self._collection()
        self.run_stage()
        before = _places(self.conn)
        self.run_stage()
        self.assertEqual(_places(self.conn), before)

    def test_a_frame_that_matched_nothing_is_remembered_as_such(self) -> None:
        """The population this feature is for: the frames that never match.

        They keep `unknown` and come back into the selection on every run — a marker
        written only for the survivors would save nothing at all.
        """
        quiet = self.add("/photos/DCIM", -1, prob=0.95)   # an anti-class took the mass
        self.run_stage()
        self.assertEqual(self.scans()[quiet], (SCAN_NONE, None))
        self.run_stage()
        self.assertEqual(self.seen, [])

    def test_a_proposal_is_remembered_with_its_score(self) -> None:
        berlin = self.add("/photos/Финляндия", BERLIN, prob=0.77)  # refuted by the path
        self.run_stage()
        verdict, score = self.scans()[berlin]
        self.assertEqual(verdict, "Бранденбургские ворота")
        self.assertAlmostEqual(float(score or 0.0), 0.77, places=5)


class TestOnlyWhatChangedIsRecomputed(SkipCase):
    """Criterion two: a new or edited file costs its own CLIP pass, not everyone's."""

    def test_a_new_file_is_the_only_one_scanned(self) -> None:
        self.add("/photos/DCIM", -1, prob=0.95, n=3)
        self.run_stage()
        fresh = self.add("/photos/DCIM", -1, prob=0.95)
        stats = self.run_stage()
        self.assertEqual(self.seen, [fresh])
        self.assertEqual((stats.scanned, stats.skipped), (4, 3))

    def test_an_edited_file_comes_back_to_clip(self) -> None:
        edited = self.add("/photos/DCIM", -1, prob=0.95)
        self.add("/photos/DCIM", -1, prob=0.95, n=2)
        self.run_stage()
        self.touch(edited)
        self.run_stage()
        self.assertEqual(self.seen, [edited])

    def test_a_replaced_photo_gets_a_new_answer(self) -> None:
        """The point of keeping the file's identity in the marker, end to end."""
        path = self.add("/photos/DCIM", -1, prob=0.95)
        self.run_stage()
        self.assertEqual(self.place_of(path)[3], "unknown")
        self.scores[path] = (PARIS, 0.99)
        self.touch(path)
        self.run_stage()
        self.assertEqual(self.place_of(path), ("FR", "Paris", 2988507, "visual"))


class TestASettingChangeRecomputesEverything(SkipCase):
    """Criterion three and four: the marker fingerprints what decides the answer."""

    def _two_runs_after(self, change) -> list[str]:
        """Run once, apply `change`, run again -> the frames the second run scanned."""
        self.add("/photos/DCIM", -1, prob=0.95, n=3)
        self.run_stage()
        change()
        self.run_stage()
        return self.seen

    def test_a_moved_threshold_recomputes_all(self) -> None:
        def change() -> None:
            self.cfg.naming = dataclasses.replace(self.cfg.naming,
                                                  landmark_threshold=0.9)
        self.assertEqual(len(self._two_runs_after(change)), 3)

    def test_an_edited_landmark_list_recomputes_all(self) -> None:
        def change() -> None:
            path = Path(self.cfg.naming.landmarks_file)
            path.write_text(path.read_text(encoding="utf-8").replace(
                "a photo of Red Square in Moscow",
                "a photo of Saint Basil's Cathedral in Moscow"), encoding="utf-8")
        self.assertEqual(len(self._two_runs_after(change)), 3)

    def test_a_renamed_landmark_recomputes_all(self) -> None:
        """An edit that leaves every prompt alone still changes what would be written."""
        def change() -> None:
            path = Path(self.cfg.naming.landmarks_file)
            path.write_text(path.read_text(encoding="utf-8").replace(
                "geonameid: 524901", "geonameid: 524902"), encoding="utf-8")
        self.assertEqual(len(self._two_runs_after(change)), 3)

    def test_a_reworded_prompt_recomputes_all(self) -> None:
        """The distractors are code, not config — and they move the scores too."""
        anti = ("a screenshot from a video game", "a drawing of a famous building")
        with mock.patch("sorta.landmarks._ANTI_PROMPTS", anti):
            self.add("/photos/DCIM", -1, prob=0.95, n=3)
            self.run_stage()
        self.run_stage()
        self.assertEqual(len(self.seen), 3)

    def test_a_moved_group_threshold_recomputes_all(self) -> None:
        def change() -> None:
            self.cfg.naming = dataclasses.replace(self.cfg.naming,
                                                  landmark_group_min=3)
        self.assertEqual(len(self._two_runs_after(change)), 3)

    def test_a_moved_dominance_recomputes_all(self) -> None:
        def change() -> None:
            self.cfg.naming = dataclasses.replace(self.cfg.naming,
                                                  landmark_group_dominance=0.9)
        self.assertEqual(len(self._two_runs_after(change)), 3)

    def test_a_wider_candidate_gate_recomputes_all(self) -> None:
        """Turning the F131 check on widens the band proposals are collected at."""
        self.add("/photos/DCIM", PRAGUE, prob=0.2, n=3)     # below today's threshold
        self.cfg.naming = _naming_from({"landmarks_file": self.cfg.naming.landmarks_file,
                                        "landmark_threshold": 0.85})
        self.run_stage()
        self.cfg.features = dataclasses.replace(self.cfg.features,
                                                landmarks_verify=True,
                                                landmark_candidate_threshold=0.1)
        self.run_stage(asker=lambda path: SAYS_PRAGUE)
        self.assertEqual(len(self.seen), 3)


class TestTheFingerprint(SkipCase):
    """The digest itself: the same inputs give the same string, any of them moves it."""

    def _fingerprint(self, **over) -> str:
        landmarks = load_landmarks(self.cfg.naming.landmarks_file)
        args = dict(landmarks=landmarks, prompts=landmark_prompts(landmarks),
                    threshold=0.85, gate=0.5, min_group=5, dominance=0.6)
        args.update(over)
        return _stage_fingerprint(**args)  # type: ignore[arg-type]

    def test_the_same_inputs_give_the_same_fingerprint(self) -> None:
        self.assertEqual(self._fingerprint(), self._fingerprint())

    def test_every_input_moves_it(self) -> None:
        base = self._fingerprint()
        for field, value in (("threshold", 0.9), ("gate", 0.4), ("min_group", 6),
                             ("dominance", 0.5), ("prompts", ["a photo of nothing"])):
            with self.subTest(field):
                self.assertNotEqual(self._fingerprint(**{field: value}), base)


class TestCorroborationOverThePartialSet(SkipCase):
    """The main test of the feature: a partial run decides what a full one would.

    The set-up is the case a per-file skip gets wrong. After the first run the three
    Berlins are `unknown` — the group rule dropped them — and their seven Paris
    neighbours have left the selection as `visual`. Two new frames then arrive in the
    same folder. Whether they are placed depends entirely on whether the three raised
    Berlins are counted with them: together they are a group of five with a dominant
    city, and the newcomers are its minority; on their own they are a group of two,
    which the rule does not touch at all.
    """

    def _scenario(self) -> tuple[list[str], list[str]]:
        berlin = [self.add("/photos/DCIM/100D3300", BERLIN) for _ in range(3)]
        self.add("/photos/DCIM/100D3300", PARIS, n=7)
        self.run_stage()
        for path in berlin:
            self.assertEqual(self.place_of(path)[3], "unknown")
        newcomers = [self.add("/photos/DCIM/100D3300", PRAGUE) for _ in range(2)]
        return berlin, newcomers

    def _reference(self) -> dict[str, tuple]:
        """The same selection, computed WITHOUT the skip — the run to match.

        A copy of the database with the stage's scan rows dropped: same files, same
        settings, same `unknown` set, and every frame of it goes back through CLIP. That
        is the definition of "the verdicts a full run gives", and comparing against it
        beats writing the expected places out by hand — an expectation copied from the
        implementation would agree with a bug just as happily.
        """
        self.conn.commit()
        ref = sqlite3.connect(Path(self.tmp.name) / "reference.db")
        self.conn.backup(ref)
        ref.row_factory = sqlite3.Row
        ref.execute("DELETE FROM landmark_checks WHERE landmark = ?", (SCAN_KEY,))
        ref.commit()
        detect_landmarks(self.cfg, ref, classifier=PathClassifier(self.scores),
                         resolver=self.resolver)
        snapshot = _places(ref)
        ref.close()
        return snapshot

    def test_the_partial_run_agrees_with_the_full_one(self) -> None:
        self._scenario()
        reference = self._reference()
        stats = self.run_stage()
        self.assertEqual(stats.skipped, 3)          # the raised Berlins
        self.assertEqual(_places(self.conn), reference)

    def test_the_raised_matches_decide_the_newcomers(self) -> None:
        """Spelled out, so a run that quietly dropped them fails loudly here."""
        berlin, newcomers = self._scenario()
        self.run_stage()
        for path in newcomers:
            self.assertEqual(self.place_of(path), (None, None, None, "unknown"))
        for path in berlin:
            self.assertEqual(self.place_of(path), ("DE", "Berlin", 2950159, "visual"))

    def test_the_skipped_frames_are_still_counted(self) -> None:
        self._scenario()
        stats = self.run_stage()
        self.assertEqual(stats.scanned, 5)          # 3 raised + 2 new
        self.assertEqual(stats.proposals, 5)
        self.assertEqual(stats.dropped_by_group, 2)


class TestARefutedFrameStaysUnknown(SkipCase):
    """Criterion six: the F75 invariant survives the skip, run after run.

    A country named in the path refutes the match every time, because corroboration is
    recomputed for the raised proposal as well — the skip is over the CLIP pass, never
    over the rules that decide where a file goes.
    """

    def test_the_refutation_is_recomputed_for_a_skipped_frame(self) -> None:
        berlin = self.add("/photos/Финляндия/Хельсинки", BERLIN)
        self.run_stage()
        self.assertEqual(self.place_of(berlin), (None, None, None, "unknown"))
        stats = self.run_stage()
        self.assertEqual(self.seen, [])
        self.assertEqual((stats.skipped, stats.dropped_by_folder_name), (1, 1))
        self.assertEqual(self.place_of(berlin), (None, None, None, "unknown"))

    def test_a_dropped_frame_never_takes_the_place_of_its_neighbours(self) -> None:
        """The one thing worse than a missed place: a wrong one, arrived at by skipping."""
        self.add("/photos/Франция/Париж", PRAGUE, n=9)
        odd = self.add("/photos/Франция/Париж", PARIS, n=1)
        self.run_stage()
        self.run_stage()
        self.assertEqual(self.place_of(odd), ("FR", "Paris", 2988507, "visual"))
        self.assertEqual(self.cities(), {"Paris": 1})


class TestTheCheckedFrameIsNotAskedAgain(VerifyCase):
    """Criterion five: with F131 on, a skipped frame is not re-scored NOR re-asked."""

    def run_stage(self, asker=..., **kwargs):
        if asker is ...:
            asker = self.ask
        self.clip = CountingClassifier(self.scores)
        return detect_landmarks(self.cfg, self.conn, classifier=self.clip,
                                resolver=self.resolver, asker=asker, **kwargs)

    def test_a_rejected_frame_costs_nothing_on_the_second_run(self) -> None:
        rejected = self.says(self.add("/photos/DCIM", PRAGUE, prob=0.99), SAYS_BERLIN)
        self.run_stage()
        self.assertEqual((self.asked, self.clip.seen), ([rejected], [rejected]))
        self.asked.clear()
        stats = self.run_stage()
        self.assertEqual((self.asked, self.clip.seen), ([], []))
        self.assertEqual((stats.skipped, stats.checks_reused), (1, 1))
        self.assertEqual(self.place_of(rejected), (None, None, None, "unknown"))

    def test_a_reused_confirmation_still_places_the_frame(self) -> None:
        """The raised proposal carries the whole decision, not just "do not ask again".

        The frame gets here the only way a confirmed one can come back: the group rule
        dropped it while its eight neighbours were placed. On the second run neither CLIP
        nor the model is consulted, and it is still placed — from the stored proposal and
        the stored answer together.
        """
        for _ in range(8):
            self.says(self.add("/photos/DCIM/100D3300", PRAGUE, prob=0.95), SAYS_PRAGUE)
        odd = self.says(self.add("/photos/DCIM/100D3300", BERLIN, prob=0.95), SAYS_BERLIN)
        self.run_stage()
        self.assertEqual(self.place_of(odd)[3], "unknown")
        self.asked.clear()
        stats = self.run_stage()
        self.assertEqual((self.asked, self.clip.seen), ([], []))
        self.assertEqual((stats.skipped, stats.checks_reused), (1, 1))
        self.assertEqual(self.place_of(odd), ("DE", "Berlin", 2950159, "visual"))
