"""F125: `vlm.quality_scope: faces` — ask about the face markup, or do not ask at all.

The question "are the eyes open" only has a population where a face was found. Measured
on the live collection: 7 341 photographs carry one, against 19 757 frames for `all` —
95 minutes of GPU instead of 4.3 hours, and the frames dropped are the ones where the
question means nothing.

Two properties carry the feature and both are easy to get wrong:

* `faces` rows with `bbox = '[]'` are the marker "processed, no face here", not a face.
  Nearly every file in a processed index has one (24 195 of 24 196), so a predicate that
  forgets to exclude them turns "by faces" into "by everything";
* no faces run means no population, and the user's rule is strict — no faces pass, no
  question. The model half is then not built at all, with a reason in the log, while the
  cheap tiers (sharpness, animals) go on measuring: an optional add-on must not take the
  stage down with it.
"""
from __future__ import annotations

from sorta import junk
from sorta.junk import classify, quality_scope_ids

from tests.test_frame_quality import (
    CONFIDENT,
    NO_OCR,
    _CAT_IDX,
    _PHOTO_IDX,
    FrameQualityCase,
    QualityClassifier,
    flat_sharpness,
)

ANSWER = "eyes_open subject"


class FacesScopeCase(FrameQualityCase):
    """A collection where every frame is in the uncertain band, so only scope decides."""

    def setUp(self):
        super().setUp()
        self.vlm(quality=True, quality_scope="faces")
        self.deep_analysis_on()  # F145: `vlm.quality` alone raises no model
        self.asked: list[str] = []

    def mark_processed(self, fid: int) -> None:
        """The faces stage's "this file was processed and had no face in it" row."""
        self.conn.execute(
            "INSERT INTO faces (file_id, bbox, embedding) VALUES (?, '[]', ?)",
            (fid, b""))
        self.conn.commit()

    def ask(self, path: str) -> str:
        self.asked.append(path)
        return ANSWER

    def run_junk(self, logits=None, **kwargs):
        """One classify() run over frames CLIP is confident about (band = sharpness)."""
        clf = QualityClassifier(logits=logits)
        return classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR,
                        sharpness_detector=flat_sharpness(100.0), **kwargs)


class TestOnlyRealFacesAreAsked(FacesScopeCase):
    def test_the_population_is_the_frames_with_a_face(self):
        with_face = self.add_file("face.jpg", has_face=True)
        without = self.add_file("landscape.jpg")
        self.run_junk(quality_vlm=self.ask)

        self.assertEqual(self.asked, ["/photos/face.jpg"])
        self.assertEqual(self.quality(with_face)["eyes_open"], 1)
        self.assertIsNone(self.quality(without)["eyes_open"])

    def test_the_processed_marker_is_not_a_face(self):
        """The trap. `bbox = '[]'` stands on nearly every processed file, and taking it
        for a face makes `faces` mean `all` while looking like it works."""
        with_face = self.add_file("face.jpg", has_face=True)
        marked = self.add_file("no_face.jpg")
        self.mark_processed(marked)

        self.run_junk(quality_vlm=self.ask)
        self.assertEqual(self.asked, ["/photos/face.jpg"])
        self.assertEqual(self.quality(with_face)["eyes_open"], 1)
        self.assertIsNone(self.quality(marked)["eyes_open"])

    def test_a_frame_that_is_not_a_photograph_is_left_out(self):
        """F120 still applies on top: a screenshot with a face in it is not asked about
        (and keeps no quality row at all)."""
        photo = self.add_file("face.jpg", has_face=True)
        # no camera in the EXIF: with one, `_is_real_photo` protects the frame from the
        # screenshot verdict and the case would be testing something else
        shot = self.add_file("Screenshot_1.png", camera_make=None, camera_model=None,
                             has_face=True)

        self.run_junk(logits={"face.jpg": {_PHOTO_IDX: CONFIDENT},
                              "Screenshot_1.png": {1: CONFIDENT}},
                      quality_vlm=self.ask)
        self.assertEqual(self.media_class(shot)["verdict"], "screenshot")
        self.assertEqual(self.asked, ["/photos/face.jpg"])
        self.assertIsNotNone(self.quality(photo))
        self.assertIsNone(self.quality(shot))

    def test_the_second_run_does_not_ask_again(self):
        self.add_file("face.jpg", has_face=True)
        self.run_junk(quality_vlm=self.ask)
        self.assertEqual(len(self.asked), 1)
        self.run_junk(quality_vlm=self.ask)
        self.assertEqual(len(self.asked), 1)


class TestWithoutAFacesRun(FacesScopeCase):
    """Brief test 3: the hard dependency. No faces, no question — and no crash."""

    def factory(self, _model):
        raise AssertionError("no model may be built when the faces scope has no faces")

    def test_the_model_is_not_built_and_the_reason_is_logged(self):
        fid = self.add_file("landscape.jpg")
        self.mark_processed(fid)  # the stage ran; it just found nothing

        with self.assertLogs("sorta.junk", level="WARNING") as logged:
            stats = self.run_junk(quality_vlm_factory=self.factory)

        self.assertIn("faces", "\n".join(logged.output))
        self.assertEqual(stats.quality_candidates, 0)
        self.assertEqual(stats.quality_answered, 0)

    def test_the_cheap_tiers_still_run(self):
        self.features(pets=True, pet_threshold=0.5)
        fid = self.add_file("cat.jpg")

        with self.assertLogs("sorta.junk", level="WARNING"):
            stats = self.run_junk(logits={"cat.jpg": {_CAT_IDX: CONFIDENT}},
                                  quality_vlm_factory=self.factory)

        row = self.quality(fid)
        self.assertAlmostEqual(row["sharpness"], 100.0)
        self.assertEqual(row["pet"], junk.PET_CLASS)
        self.assertEqual(stats.pets_found, 1)
        self.assertIsNone(row["eyes_open"])
        # the row is marked by the tier that actually ran, so a run after `faces` picks
        # it up instead of considering the question answered
        self.assertEqual(junk.quality_tier(row["source"]), junk.QUALITY_SOURCE_CLIP)

    def test_a_faces_run_afterwards_asks_the_question(self):
        """The other half of the same promise: the dependency is satisfiable."""
        fid = self.add_file("face.jpg")
        self.mark_processed(fid)
        with self.assertLogs("sorta.junk", level="WARNING"):
            self.run_junk(quality_vlm_factory=self.factory)
        self.assertIsNone(self.quality(fid)["eyes_open"])

        # `faces` runs and finds one
        self.conn.execute(
            "INSERT INTO faces (file_id, bbox, embedding) VALUES (?, '[1,2,3,4]', ?)",
            (fid, b"\x00" * 4))
        self.conn.commit()
        self.run_junk(quality_vlm=self.ask)
        self.assertEqual(self.asked, ["/photos/face.jpg"])
        self.assertEqual(self.quality(fid)["eyes_open"], 1)


class TestAnimalsKeepTheirOwnPopulation(FacesScopeCase):
    """Brief test 6: the scope is the VLM's, not the cascade's.

    Animals are a prompt group inside the CLIP call the stage already makes, and they are
    a question about every photograph. Narrowing them to the frames with a face would cost
    a signal that was already paid for.
    """

    def test_pets_are_measured_on_frames_the_model_never_sees(self):
        cat = self.add_file("cat.jpg")
        person = self.add_file("face.jpg", has_face=True)
        self.features(pets=True, pet_threshold=0.5)

        stats = self.run_junk(logits={"cat.jpg": {_CAT_IDX: CONFIDENT}},
                              quality_vlm=self.ask)

        self.assertEqual(self.asked, ["/photos/face.jpg"])
        self.assertEqual(self.quality(cat)["pet"], junk.PET_CLASS)
        self.assertIsNone(self.quality(cat)["eyes_open"])
        self.assertEqual(stats.quality_rows, 2)   # both frames measured
        self.assertEqual(stats.pets_found, 1)
        self.assertEqual(stats.quality_candidates, 1)  # one asked about
        self.assertIsNotNone(self.quality(person))


class TestTheOtherScopesAreUnchanged(FacesScopeCase):
    """Brief test 4: `faces` is a fourth value, not a change to the three that existed."""

    def populate(self):
        """A frame with a face outside any group, and a group of two without faces."""
        lonely_face = self.add_file("face.jpg", has_face=True,
                                    phash="0000000000000000")
        group_a = self.add_file("g1.jpg", phash="ffffffffffffffff")
        group_b = self.add_file("g2.jpg", phash="ffffffffffffffff")
        return lonely_face, group_a, group_b

    def test_groups_still_means_the_near_duplicate_groups(self):
        self.vlm(quality=True)  # the default scope
        self.populate()
        self.run_junk(quality_vlm=self.ask)
        self.assertEqual(sorted(self.asked), ["/photos/g1.jpg", "/photos/g2.jpg"])

    def test_events_still_means_the_event_frames(self):
        self.vlm(quality=True, quality_scope="events")
        lonely_face, group_a, _group_b = self.populate()
        self.conn.execute(
            "INSERT INTO events (id, started_at, ended_at, name) "
            "VALUES (1, '2026-01-01', '2026-01-02', 'x')")
        self.conn.execute(
            "INSERT INTO event_files (event_id, file_id) VALUES (1, ?)", (group_a,))
        self.conn.commit()

        self.run_junk(quality_vlm=self.ask)
        self.assertEqual(self.asked, ["/photos/g1.jpg"])
        self.assertIsNone(self.quality(lonely_face)["eyes_open"])

    def test_all_still_means_all(self):
        self.vlm(quality=True, quality_scope="all")
        self.populate()
        self.run_junk(quality_vlm=self.ask)
        self.assertEqual(sorted(self.asked),
                         ["/photos/face.jpg", "/photos/g1.jpg", "/photos/g2.jpg"])

    def test_faces_asks_about_the_face_alone(self):
        self.populate()
        self.run_junk(quality_vlm=self.ask)
        self.assertEqual(self.asked, ["/photos/face.jpg"])


class TestScopeHelpers(FacesScopeCase):
    """The two functions the stage decides on, asked directly."""

    def test_the_id_set_holds_photographs_with_a_face(self):
        with_face = self.add_file("face.jpg", has_face=True)
        unclassified = self.add_file("fresh.jpg", has_face=True)
        marked = self.add_file("no_face.jpg")
        self.mark_processed(marked)
        bare = self.add_file("never_processed.jpg")
        not_a_photo = self.add_file("shot.png", has_face=True)
        for fid, verdict in ((with_face, "photo"), (marked, "photo"), (bare, "photo"),
                             (not_a_photo, "screenshot")):
            self.conn.execute(
                """INSERT INTO media_class (file_id, verdict, source, score, updated_at,
                       tier) VALUES (?, ?, 'clip', 0.9, '2026-01-01', 'clip')""",
                (fid, verdict))
        self.conn.commit()

        ids = quality_scope_ids(self.cfg, self.conn, "faces")
        # `unclassified` has no verdict yet — a first run has classified nothing, and
        # excluding it would make the scope empty exactly when the whole index is new
        self.assertEqual(ids, {with_face, unclassified})

    def test_the_marker_alone_is_not_a_faces_run(self):
        fid = self.add_file("no_face.jpg")
        self.assertFalse(junk.faces_stage_ran(self.conn))
        self.mark_processed(fid)
        self.assertFalse(junk.faces_stage_ran(self.conn))
        self.add_file("face.jpg", has_face=True)
        self.assertTrue(junk.faces_stage_ran(self.conn))

    def test_only_the_faces_scope_can_be_unsatisfiable(self):
        for scope in ("groups", "events", "all"):
            with self.subTest(scope=scope):
                self.assertTrue(junk.quality_scope_ready(self.conn, scope))
        with self.assertLogs("sorta.junk", level="WARNING"):
            self.assertFalse(junk.quality_scope_ready(self.conn, "faces"))
        self.add_file("face.jpg", has_face=True)
        self.assertTrue(junk.quality_scope_ready(self.conn, "faces"))
