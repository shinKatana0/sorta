"""F121: the eyes answer is believed only where a face was actually detected.

The prompt already says "use neither word if there are no people" and the model does not
obey it — a review of the first live run found cats answered as `eyes_open` and people in
glasses answered as `eyes_closed`. The detector knows where a face is, so the cheap fix
is to stop believing the answer elsewhere. Asking stays free (one prompt, one call);
believing is what costs.

The trap this guards against is the other direction: if `faces` has never run, "no face
on this frame" and "nobody has looked" are the same row, and dropping the answer on both
would switch the signal off for every user who skipped the faces stage — silently.
"""
from __future__ import annotations

from sorta.junk import classify

from tests.test_frame_quality import (
    CONFIDENT,
    NO_OCR,
    _PHOTO_IDX,
    FrameQualityCase,
    QualityClassifier,
    flat_sharpness,
)


def answer_all(_path: str) -> str:
    """The model answering every keyword it has ever known, eyes included — as it does
    on a cat. The retired words (F122, F177) are there on purpose: a model still says
    them, and they must not turn into a stored answer."""
    return "eyes_closed no_subject deliberate"


class TestEyesAreBelievedOnlyWhereAFaceIs(FrameQualityCase):
    def setUp(self):
        super().setUp()
        self.vlm(quality=True, quality_scope="all")
        self.features(sharpness_band_min=30.0, sharpness_band_max=300.0)
        self.deep_analysis_on()  # F145: `vlm.quality` alone raises no model

    def run_junk(self) -> None:
        clf = QualityClassifier(logits={"with_face.jpg": {_PHOTO_IDX: CONFIDENT},
                                        "no_face.jpg": {_PHOTO_IDX: CONFIDENT}})
        classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR,
                 sharpness_detector=flat_sharpness(100.0), quality_vlm=answer_all)

    def test_a_frame_with_a_face_keeps_the_eyes_answer(self):
        fid = self.add_file("with_face.jpg", has_face=True)
        self.run_junk()
        row = self.quality(fid)
        self.assertEqual(row["eyes_open"], 0)  # the model said eyes_closed

    def test_a_frame_without_a_face_drops_it(self):
        """The cat. The model answers, and the answer is not written."""
        with_face = self.add_file("with_face.jpg", has_face=True)
        no_face = self.add_file("no_face.jpg")
        self.run_junk()
        self.assertEqual(self.quality(with_face)["eyes_open"], 0)
        self.assertIsNone(self.quality(no_face)["eyes_open"])

    def test_a_frame_without_a_face_carries_no_answer_at_all(self):
        """F177: the eyes are the whole answer now, so dropping them on a faceless frame
        leaves nothing behind. Until F177 the row kept `has_subject` here; the question
        is retired and the column must stay NULL whatever the model volunteers."""
        with_face = self.add_file("with_face.jpg", has_face=True)
        no_face = self.add_file("no_face.jpg")
        self.run_junk()
        row = self.quality(no_face)
        self.assertIsNone(row["eyes_open"])
        self.assertIsNone(row["has_subject"])
        self.assertIsNone(row["is_accidental"])
        # the row itself is still written by the cheap tier — only the answers are absent
        self.assertIsNotNone(row["sharpness"])
        self.assertIsNotNone(self.quality(with_face))

    def test_without_a_faces_run_the_answer_is_kept(self):
        """No face rows at all: "no face here" cannot be told from "nobody looked", so
        the signal stays rather than vanishing for everyone who skipped the stage."""
        fid = self.add_file("no_face.jpg")
        self.run_junk()
        self.assertEqual(self.quality(fid)["eyes_open"], 0)
