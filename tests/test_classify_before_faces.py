"""F165: the verdicts run before faces, and faces skip what they call junk.

The faces stage is 46% of a full run and it used to walk every canonical photograph —
screenshots, receipts and memes included — because the classifier that knows about them
ran AFTER it. So the classification is split in two by dependency: `classify` (verdicts,
which need nothing from faces) goes ahead of the detector, and `junk` keeps everything
that reads what the detector writes.

What the cases below check is the feature and the three ways it could quietly go wrong:

* screenshots, documents and memes never reach face detection (the whole point), and the
  number of faces on a collection of real photographs does not move;
* NULL is not a verdict. A frame nobody has classified is detected exactly as before —
  otherwise `sorta faces` on a fresh index would find nothing at all — and a frame the
  deep tier moves back to `photo` gets its faces on the next run, so the economy cannot
  turn into lost data;
* what `junk` reads out of the `faces` table still works, because that half did NOT move:
  `frame_quality.face_sharpness` (F155) is measured inside the boxes the faces stage
  wrote. A naive swap of the two stages would have broken that silently, which is why the
  split is by dependency and the test for it is a regression test for F155. (The second
  reader of those boxes was `vlm.quality_scope: faces`, and F186 retired it with the
  question it chose a population for.)

No model is loaded anywhere: the classifier, the sharpness detector and the face analyzer
are all injected, as everywhere else in this suite.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from sorta import cli, faces, junk
from sorta.faces import EMBED_DIM, detect_faces
from sorta.junk import QUALITY_VERDICT, classify
from tests.test_frame_quality import (
    FrameQualityCase,
    QualityClassifier,
    flat_sharpness,
)
from tests.test_junk import NO_OCR, FakeClassifier

_CLASSES = [cls for cls, _prompt in junk._CLIP_CLASSES]
_SCREENSHOT_IDX = _CLASSES.index("screenshot")
_PHOTO_IDX = _CLASSES.index("photo")


def _embedding() -> np.ndarray:
    return np.full(EMBED_DIM, 0.1, dtype=np.float32)


def _hit() -> tuple[list[float], float, np.ndarray]:
    return ([10.0, 10.0, 130.0, 130.0], 0.95, _embedding())


class ClassifyBeforeFacesCase(FrameQualityCase):
    """A temp index, plus the two halves of the stage and a recording face detector."""

    def add_junk_file(self, name: str) -> int:
        """A frame CLIP is free to call junk: no camera, no GPS, no face to veto it."""
        return self.add_file(name, camera_make=None, camera_model=None)

    def run_classify(self, classifier=None, **kwargs) -> junk.JunkStats:
        """The `classify` stage — the verdicts alone, the half that runs before faces."""
        return classify(self.cfg, self.conn,
                        classifier=classifier or FakeClassifier({}),
                        text_detector=NO_OCR, verdicts_only=True, **kwargs)

    def run_junk(self, classifier=None, sharpness=None, **kwargs) -> junk.JunkStats:
        """The `junk` stage — everything that needs the face signal."""
        return classify(self.cfg, self.conn,
                        classifier=classifier or QualityClassifier(),
                        text_detector=NO_OCR,
                        sharpness_detector=sharpness or flat_sharpness(100.0), **kwargs)

    def detect(self, hits=None, **kwargs):
        """Run the faces stage with a mock analyzer; returns the paths it was shown."""
        seen: list[str] = []
        found = hits or {}

        def analyzer(path, _orientation):
            seen.append(path)
            return found.get(path, [])

        detect_faces(self.cfg, self.conn, analyzer=analyzer, **kwargs)
        return seen

    def verdict(self, fid: int) -> str | None:
        row = self.media_class(fid)
        return None if row is None else row["verdict"]

    def face_rows(self, fid: int) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) FROM faces WHERE file_id = ?", (fid,)).fetchone()[0]

    def set_verdict(self, fid: int, verdict: str) -> None:
        """What the deep tier or the rescue does to a frame between two runs."""
        self.conn.execute(
            "INSERT INTO media_class (file_id, verdict, source, updated_at, tier)"
            " VALUES (?, ?, 'vlm', '2026-01-02', 'vlm')"
            " ON CONFLICT(file_id) DO UPDATE SET verdict = excluded.verdict",
            (fid, verdict))
        self.conn.commit()


class TestFacesSkipWhatTheClassifierRejected(ClassifyBeforeFacesCase):
    """Brief test 1 (the main one) and test 7: what is skipped, and what is not."""

    def test_a_screenshot_never_reaches_face_detection(self):
        photo = self.add_file("IMG_0001.jpg")
        shot = self.add_junk_file("shot.png")
        clf = FakeClassifier({"shot.png": (_SCREENSHOT_IDX, 0.99)})

        self.run_classify(classifier=clf)
        self.assertEqual(self.verdict(shot), "screenshot")
        self.assertEqual(self.verdict(photo), QUALITY_VERDICT)

        seen = self.detect()
        self.assertEqual(seen, ["/photos/IMG_0001.jpg"])
        # Not even the "processed, no faces" marker: the frame was never looked at, and
        # the next run must be free to look if the verdict changes.
        self.assertEqual(self.face_rows(shot), 0)

    def test_documents_and_memes_are_skipped_too(self):
        meme = self.add_junk_file("meme.jpg")
        doc = self.add_junk_file("doc.jpg")
        self.set_verdict(meme, "meme")
        self.set_verdict(doc, "document")
        self.assertEqual(self.detect(), [])

    def test_the_faces_of_an_all_photo_collection_are_unchanged(self):
        # Brief test 7: the economy must come out of frames nobody wanted, not out of the
        # answer. The same collection classified and unclassified detects the same faces.
        ids = [self.add_file(f"IMG_{i}.jpg") for i in range(3)]
        hits = {f"/photos/IMG_{i}.jpg": [_hit()] for i in range(3)}
        before = self.detect(hits=hits)
        found_before = self.conn.execute(
            "SELECT COUNT(*) FROM faces WHERE bbox != '[]'").fetchone()[0]

        self.conn.execute("DELETE FROM faces")
        self.conn.commit()
        self.run_classify()
        for fid in ids:
            self.assertEqual(self.verdict(fid), QUALITY_VERDICT)
        after = self.detect(hits=hits)
        found_after = self.conn.execute(
            "SELECT COUNT(*) FROM faces WHERE bbox != '[]'").fetchone()[0]

        self.assertEqual(after, before)
        self.assertEqual(found_after, found_before)


class TestNullIsNotAVerdict(ClassifyBeforeFacesCase):
    """Brief tests 2, 3 and 6: the rule that keeps the economy from losing data."""

    def test_a_frame_with_no_row_in_media_class_is_detected(self):
        fid = self.add_file("IMG_0001.jpg")
        self.assertIsNone(self.verdict(fid))
        self.assertEqual(self.detect(), ["/photos/IMG_0001.jpg"])

    def test_faces_alone_walks_the_whole_collection(self):
        # Brief test 6: `sorta faces` without ever running the classification. Junk-looking
        # names and all — nobody has been asked about them, so nobody may skip them.
        paths = [f"/photos/{name}" for name in ("IMG_0001.jpg", "shot.png", "doc.jpg")]
        for name in ("IMG_0001.jpg", "shot.png", "doc.jpg"):
            self.add_junk_file(name)
        self.assertEqual(sorted(self.detect()), sorted(paths))

    def test_a_frame_reclassified_to_photo_gets_faces_on_the_next_run(self):
        # Brief test 3: the deep tier moved 2 592 frames of 24 196 on the reference run.
        # A frame that becomes a photograph after the faces stage has passed it by has no
        # `faces` row, so the ordinary incrementality of the stage picks it up.
        fid = self.add_junk_file("shot.png")
        self.set_verdict(fid, "screenshot")
        self.assertEqual(self.detect(), [])

        self.set_verdict(fid, QUALITY_VERDICT)
        self.assertEqual(self.detect(hits={"/photos/shot.png": [_hit()]}),
                         ["/photos/shot.png"])
        self.assertEqual(self.face_rows(fid), 1)

    def test_a_frame_reclassified_to_junk_keeps_the_faces_already_found(self):
        # The other direction, and it is not an error: work already paid for is not
        # undone by a later verdict. The row stays, the stage does not go back to it.
        fid = self.add_file("IMG_0001.jpg")
        self.detect(hits={"/photos/IMG_0001.jpg": [_hit()]})
        self.set_verdict(fid, "product")
        self.assertEqual(self.detect(), [])
        self.assertEqual(self.face_rows(fid), 1)

    def test_a_rescan_is_narrowed_by_the_verdicts_as_well(self):
        photo = self.add_file("IMG_0001.jpg")
        shot = self.add_junk_file("shot.png")
        self.set_verdict(photo, QUALITY_VERDICT)
        self.set_verdict(shot, "screenshot")
        self.detect(hits={"/photos/IMG_0001.jpg": [_hit()]})
        self.assertEqual(self.detect(rescan=True), ["/photos/IMG_0001.jpg"])

    def test_the_verdict_the_faces_stage_reads_is_the_one_junk_writes(self):
        """The two spellings of "a personal photograph" cannot drift apart.

        `faces` names the value itself instead of importing the classification stage (it
        keeps a three-import module three imports wide), so the pin is here.
        """
        self.assertEqual(faces._PHOTO_VERDICT, QUALITY_VERDICT)


class TestTheHalfAfterFacesStillHasIts(ClassifyBeforeFacesCase):
    """Brief tests 4 and 5: the dependencies a naive stage swap would have broken."""

    def test_face_sharpness_is_still_measured(self):
        # REGRESSION TEST FOR F155. `frame_quality.face_sharpness` is the laplacian inside
        # `faces.bbox`; with the stages swapped instead of split it would silently stop
        # being computed on every first run, because NULL there means "not measured".
        fid = self.add_file("IMG_0001.jpg")
        self.run_classify()
        self.detect(hits={"/photos/IMG_0001.jpg": [_hit()]})

        def sharpness(_path, boxes=junk.NO_FACES):
            # The face number is answered only when the boxes actually arrive — a detector
            # that returns a constant would pass this test with the dependency broken.
            return junk.Sharpness(frame=100.0, face=42.0 if boxes.usable else None)

        self.run_junk(sharpness=sharpness)
        row = self.quality(fid)
        self.assertEqual(row["sharpness"], 100.0)
        self.assertEqual(row["face_sharpness"], 42.0)


class TestTheSplitItself(ClassifyBeforeFacesCase):
    """One function, two stages: what each half does, and what it leaves alone."""

    def test_the_verdicts_half_writes_no_frame_quality_row(self):
        fid = self.add_file("IMG_0001.jpg")

        def sharpness(_path, _boxes=junk.NO_FACES):
            raise AssertionError("the verdicts half must not measure frames")

        stats = self.run_classify(sharpness_detector=sharpness)
        self.assertEqual(self.verdict(fid), QUALITY_VERDICT)
        self.assertEqual(stats.processed, 1)
        self.assertEqual(stats.quality_rows, 0)
        self.assertIsNone(self.quality(fid))

    def test_the_verdicts_half_asks_no_model_of_the_cascades(self):
        # F145 in the small: every question of the half after faces belongs to that half,
        # and a factory called here would mean weights loaded before the faces stage.
        self.deep_analysis_on()
        self.features(pets=True, pets_verify=True, junk_rescue=True)
        self.add_file("IMG_0001.jpg")

        def factory(_model):
            raise AssertionError("no cascade model may be built by the verdicts half")

        self.run_classify(pet_vlm_factory=factory, junk_rescue_vlm_factory=factory)

    def test_the_second_half_does_not_reclassify_what_the_first_one_settled(self):
        # The economy of the split: incrementality is `media_class.tier`, so the frames
        # the `classify` stage settled cost the `junk` stage nothing but their quality row.
        fid = self.add_file("IMG_0001.jpg")
        first = self.run_classify()
        second = self.run_junk()
        self.assertEqual((first.processed, first.skipped_incremental), (1, 0))
        self.assertEqual((second.processed, second.skipped_incremental), (0, 1))
        self.assertEqual(self.verdict(fid), QUALITY_VERDICT)
        self.assertEqual(second.quality_rows, 1)
        self.assertIsNotNone(self.quality(fid))

    def test_junk_on_its_own_is_still_the_whole_stage(self):
        # Nobody is forced through the split: one call with no flag classifies AND
        # measures, exactly as `sorta junk` did before the stage was cut in two.
        shot = self.add_junk_file("shot.png")
        photo = self.add_file("IMG_0001.jpg")
        stats = self.run_junk(classifier=FakeClassifier(
            {"shot.png": (_SCREENSHOT_IDX, 0.99)}))
        self.assertEqual(self.verdict(shot), "screenshot")
        self.assertEqual(self.verdict(photo), QUALITY_VERDICT)
        self.assertEqual(stats.processed, 2)
        self.assertIsNone(self.quality(shot))  # F120: a screenshot keeps no quality row
        self.assertIsNotNone(self.quality(photo))

    def test_the_phases_are_filed_under_the_stage_that_ran_them(self):
        # F166: `stage_timer` closes the phases registered under ITS name, so a half
        # filing them under the other stage's name would lose the breakdown of both.
        self.add_file("IMG_0001.jpg")
        with patch.object(junk, "track_phases") as tracked:
            self.run_classify()
            self.assertEqual(tracked.call_args.args, (junk.VERDICTS_STAGE,))
            self.run_junk()
            self.assertEqual(tracked.call_args.args, (junk.CLASSIFY_STAGE,))


class TestTheCommandLine(unittest.TestCase):
    """`sorta classify` — the front half, reachable on its own."""

    def setUp(self):
        if cli.app is None:  # pragma: no cover — the argparse fallback, no typer
            self.skipTest("typer is not installed")

    def test_the_command_runs_the_verdicts_half(self):
        captured: dict[str, object] = {}

        def fake_classify(cfg, conn, verdicts_only=False, progress=None, **kwargs):
            captured["verdicts_only"] = verdicts_only
            return junk.JunkStats()

        with patch.object(cli, "classify_junk", fake_classify), \
                patch.object(cli, "load_config", lambda _path: _config_for_cli()), \
                patch.object(cli, "connect", lambda _db: None):
            cli._cmd_classify("config.yaml")
        self.assertIs(captured["verdicts_only"], True)

    def test_the_argparse_fallback_offers_it_too(self):
        self.assertIn("classify", cli._FALLBACK_COMMANDS)


def _config_for_cli():
    from pathlib import Path

    from sorta.config import Config

    return Config(sources=[Path(".")], database=Path("unused.db"))


if __name__ == "__main__":
    unittest.main()
