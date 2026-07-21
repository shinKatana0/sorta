"""F6 (FR-7): junk classification — heuristics, CLIP override, incrementality.

F13: junk is conservative — the veto is extended to faces, heuristics reduced to the
explicit Screenshot_ name, the CLIP "document" class removed.
F15: verdict='document' came back as a separate review category — only without faces
and by a separate, higher CLIP threshold (naming.document_threshold).
F22: the "street/building" anti-class narrows document FP on travel photos; the
explicit Screenshot_ name overrides both document detection and the face veto.
"""
import sys
import tempfile
import types
import unittest
import unittest.mock
from pathlib import Path

import numpy as np
from PIL import Image

from sorta import imaging
from sorta.config import Config, _naming_from
from sorta.db import connect
from sorta.junk import (
    _DOCUMENT_CLASSES,
    _N_PROD_ANTI,
    _PRODUCT_CLASSES,
    _document_score,
    _in_screenshots_dir,
    classify,
    heuristic_verdict,
)
from sorta.landmarks import CachingFeatureClassifier

# F22: _DOCUMENT_CLASSES is now 5 anti-classes (photo + street/building) + 4
# document subclasses; doc_scores mocks should index the positive class "receipt",
# not the old index 1 (F15) — that is now the "building" anti-class.
_RECEIPT_IDX = [cls for cls, _prompt in _DOCUMENT_CLASSES].index("receipt")
# #14/V1: the first product subclass (after the personal-photo anti-classes) — for
# prod_scores mocks that make a file a VLM candidate by the product signal.
_PRODUCT_POS_IDX = _N_PROD_ANTI


def NO_OCR(_path, _width, _height):
    """F37 (Phase A): a text_detector mock for tests that do not check OCR —
    None means "signal unknown", the gate/rescue in classify() does not touch it,
    the verdict stays exactly what CLIP/heuristic/veto gave.
    """
    return None


class FakeClassifier:
    """Mock CLIP: class index + score by the file basename.

    F15/F22: junk.classify does two independent CLIP passes per chunk — the main 3
    classes (photo/screenshot/meme, len(prompts) == 3) and, for files without faces,
    the document classes (len(prompts) == len(_DOCUMENT_CLASSES), 5 anti + 4
    positive). Separate signal maps by prompt count keep the passes from confusing
    each other: by default (basename not in doc_scores) the document pass confidently
    returns the "photo" anti-class (idx=0) — not a document.
    """

    def __init__(self, scores, doc_scores=None, prod_scores=None):  # {basename: (class_index, score)}
        self.scores = scores
        self.doc_scores = doc_scores or {}
        self.prod_scores = prod_scores or {}
        self.seen_paths = []

    def __call__(self, image_paths, prompts):
        self.seen_paths.extend(image_paths)
        if len(prompts) == len(_DOCUMENT_CLASSES):
            table = self.doc_scores
        elif len(prompts) == len(_PRODUCT_CLASSES):
            table = self.prod_scores
        else:
            table = self.scores
        out = np.zeros((len(image_paths), len(prompts)), dtype=np.float32)
        for i, p in enumerate(image_paths):
            idx, score = table.get(Path(p).name, (0, 0.99))
            out[i, idx] = score
            # distribute the remainder so argmax stays with idx
            remainder = max(0.0, (1.0 - score) / max(1, len(prompts) - 1))
            for j in range(len(prompts)):
                if j != idx:
                    out[i, j] = remainder
        return out


def _make_caching_classifier(scores, doc_scores=None):
    """CachingFeatureClassifier (F19) with scoring logic equivalent to FakeClassifier
    above, but with a real encode()/score() split: the feature is just an id assigned
    to a basename on first encode; score() looks at the prompt count (3 — main
    classes, 5 — document pass), as before.
    """
    scores = scores
    doc_scores = doc_scores or {}
    name_to_id: dict = {}
    id_to_name: dict = {}

    def encode(paths):
        result = []
        for p in paths:
            name = Path(p).name
            if name not in name_to_id:
                name_to_id[name] = len(name_to_id)
                id_to_name[name_to_id[name]] = name
            result.append(np.array([name_to_id[name]], dtype=np.float32))
        return result

    def score(feats, prompts):
        table = doc_scores if len(prompts) == len(_DOCUMENT_CLASSES) else scores
        out = np.zeros((len(feats), len(prompts)), dtype=np.float32)
        for i, f in enumerate(feats):
            name = id_to_name[int(f[0])]
            idx, sc = table.get(name, (0, 0.99))
            out[i, idx] = sc
            remainder = max(0.0, (1.0 - sc) / max(1, len(prompts) - 1))
            for j in range(len(prompts)):
                if j != idx:
                    out[i, j] = remainder
        return out

    return CachingFeatureClassifier(encode=encode, score=score)


class TestHeuristicVerdict(unittest.TestCase):
    def test_camera_photo_is_not_junk(self):
        self.assertIsNone(heuristic_verdict(
            "/x/IMG_0001.jpg", 4032, 3024, "Apple", "iPhone 13"))

    def test_screenshot_filename_pattern(self):
        self.assertEqual(heuristic_verdict(
            "/x/Screenshot_20230501-120000.png", 1080, 2340, None, None),
            "screenshot")

    def test_screenshot_filename_pattern_russian(self):
        self.assertEqual(heuristic_verdict(
            "/x/снимок_экрана_2023.png", 1080, 2340, None, None),
            "screenshot")

    def test_screen_ratio_alone_is_no_longer_screenshot(self):
        # F13: the screen-ratio heuristic was removed — 3:4/4:3/9:16 are usual
        # phone-photo proportions, not a junk sign.
        self.assertIsNone(heuristic_verdict(
            "/x/random_name.jpg", 1080, 1920, None, None))

    def test_messenger_name_alone_is_no_longer_meme(self):
        # F13: the messenger-name→meme heuristic was removed — a forwarded photo is
        # often a real one.
        self.assertIsNone(heuristic_verdict(
            "/x/IMG-20230501-WA0007.jpg", 800, 533, None, None))
        self.assertIsNone(heuristic_verdict(
            "/x/photo_2023-05-01_12-00-00.jpg", 800, 533, None, None))

    def test_camera_photo_with_screenshot_name_still_not_junk(self):
        # camera_make is present — heuristics do not fire at all
        self.assertIsNone(heuristic_verdict(
            "/x/Screenshot_2023.png", 4000, 3000, "Canon", "EOS"))

    def test_no_signal_is_photo(self):
        self.assertIsNone(heuristic_verdict(
            "/x/regular_name.jpg", 4000, 3000, None, None))


class TestDocumentScore(unittest.TestCase):
    """F22: document-score is computed only over the positive subclasses
    (receipt/paper/meter/scan) — the anti-classes (photo + street/building) are not
    counted, however much probability mass CLIP gives them."""

    def test_ignores_anti_class_mass(self):
        n = len(_DOCUMENT_CLASSES)
        probs = np.full(n, 0.02, dtype=np.float32)
        probs[1] = 0.7  # the "building" anti-class confidently dominates
        self.assertLess(_document_score(probs), 0.1)

    def test_uses_positive_document_classes_only(self):
        n = len(_DOCUMENT_CLASSES)
        probs = np.full(n, 0.01, dtype=np.float32)
        receipt_idx = [cls for cls, _p in _DOCUMENT_CLASSES].index("receipt")
        probs[receipt_idx] = 0.9
        self.assertAlmostEqual(float(_document_score(probs)), 0.9, places=5)


class TestInScreenshotsDir(unittest.TestCase):
    def test_backslash_path(self):
        self.assertTrue(_in_screenshots_dir(r"C:\photos\Screenshots\foo.png"))

    def test_forward_slash_path(self):
        self.assertTrue(_in_screenshots_dir("/photos/Screenshots/foo.png"))

    def test_singular_screenshot_dir(self):
        self.assertTrue(_in_screenshots_dir("/photos/Screenshot/foo.png"))

    def test_case_insensitive(self):
        self.assertTrue(_in_screenshots_dir("/photos/SCREENSHOTS/foo.png"))
        self.assertTrue(_in_screenshots_dir("/photos/Screenshot/foo.png"))

    def test_nested_below_screenshots(self):
        self.assertTrue(_in_screenshots_dir("/photos/DCIM/Screenshots/sub/x.png"))

    def test_no_false_positive_on_partial_segment_match(self):
        # the segment must match fully, not be a substring
        self.assertFalse(_in_screenshots_dir("/photos/My Screenshots Album/x.jpg"))

    def test_no_match_outside_screenshots(self):
        self.assertFalse(_in_screenshots_dir("/photos/DCIM/foo.png"))


class TestClassify(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.naming = {}
        self.cfg = Config(sources=[Path(self.tmp.name)],
                          database=Path(self.tmp.name) / "test.db",
                          naming=_naming_from(self.naming))
        self.conn = connect(self.cfg.database)
        self._n = 0

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def add_file(self, name, media_type="photo", dup_of=None, error=None,
                width=4000, height=3000, camera_make="Canon", camera_model="EOS",
                gps_lat=None, has_face=False):
        self._n += 1
        cur = self.conn.execute(
            """INSERT INTO files (path, size, mtime, ext, media_type, width, height,
                   camera_make, camera_model, gps_lat, dup_of, error, indexed_at)
               VALUES (?, 1000, 0, 'jpg', ?, ?, ?, ?, ?, ?, ?, ?, '2026-01-01')""",
            (f"/photos/{name}", media_type, width, height, camera_make, camera_model,
             gps_lat, dup_of, error))
        fid = cur.lastrowid
        if has_face:
            self.conn.execute(
                "INSERT INTO faces (file_id, bbox, embedding) VALUES (?, '[1,2,3,4]', ?)",
                (fid, b"\x00" * 4))
        self.conn.commit()
        return fid

    def media_class(self, fid):
        return self.conn.execute(
            "SELECT verdict, source, score FROM media_class WHERE file_id = ?",
            (fid,)).fetchone()

    def test_heuristic_only_no_clip(self):
        shot = self.add_file("Screenshot_1.png", camera_make=None, camera_model=None)
        photo = self.add_file("IMG_0002.jpg")
        stats = classify(self.cfg, self.conn, use_clip=False)
        self.assertFalse(stats.clip_used)
        row = self.media_class(shot)
        self.assertEqual(row["verdict"], "screenshot")
        self.assertEqual(row["source"], "heuristic")
        self.assertIsNone(row["score"])
        self.assertEqual(self.media_class(photo)["verdict"], "photo")

    def test_photo_verdict_is_also_recorded(self):
        fid = self.add_file("IMG_0001.jpg")
        classify(self.cfg, self.conn, use_clip=False)
        row = self.media_class(fid)
        self.assertEqual(row["verdict"], "photo")
        self.assertEqual(row["source"], "heuristic")

    def test_clip_overrides_heuristic(self):
        # without camera EXIF a confident CLIP overrides the heuristic
        fid = self.add_file("IMG_0003.jpg", camera_make=None, camera_model=None)
        clf = FakeClassifier({"IMG_0003.jpg": (1, 0.9)})  # index 1 = screenshot, >= 0.85
        stats = classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR)
        self.assertTrue(stats.clip_used)
        row = self.media_class(fid)
        self.assertEqual(row["verdict"], "screenshot")
        self.assertEqual(row["source"], "clip")
        self.assertAlmostEqual(row["score"], 0.9, places=5)

    def test_document_class_no_longer_produced(self):
        # F13: the CLIP "document" class was removed — only photo/screenshot/meme
        from sorta.junk import _CLIP_CLASSES
        self.assertEqual([c for c, _p in _CLIP_CLASSES], ["photo", "screenshot", "meme"])

    def test_camera_exif_vetoes_clip_junk(self):
        # real data: a photo with camera EXIF must not be marked meme/screenshot by CLIP
        fid = self.add_file("160A3747.jpg")  # camera_make=Canon by default
        clf = FakeClassifier({"160A3747.jpg": (2, 0.99)})  # CLIP confidently "meme"
        classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR)
        self.assertEqual(self.media_class(fid)["verdict"], "photo")

    def test_gps_vetoes_clip_junk(self):
        fid = self.add_file("geo.jpg", camera_make=None, camera_model=None, gps_lat=55.75)
        clf = FakeClassifier({"geo.jpg": (2, 0.99)})  # CLIP confidently "meme"
        classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR)
        self.assertEqual(self.media_class(fid)["verdict"], "photo")

    def test_faces_veto_clip_junk(self):
        # F13: faces in a photo — also a veto, a messenger strips EXIF from forwards
        fid = self.add_file("random.jpg", camera_make=None, camera_model=None,
                            has_face=True)
        clf = FakeClassifier({"random.jpg": (2, 0.99)})  # CLIP confidently "meme"
        classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR)
        self.assertEqual(self.media_class(fid)["verdict"], "photo")

    def test_messenger_name_with_faces_vetoed_to_photo(self):
        # F13: a WA name is no longer a meme heuristic, and with faces + the CLIP veto — photo
        fid = self.add_file("IMG-20230501-WA0007.jpg", camera_make=None,
                            camera_model=None, has_face=True)
        clf = FakeClassifier({"IMG-20230501-WA0007.jpg": (2, 0.99)})  # CLIP "meme"
        classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR)
        self.assertEqual(self.media_class(fid)["verdict"], "photo")

    def test_screen_ratio_without_camera_or_faces_weak_clip_is_photo(self):
        # 3:4/4:3 without camera/faces + a low-confidence CLIP → the heuristic is silent → photo
        fid = self.add_file("random_name.jpg", camera_make=None, camera_model=None,
                            width=1080, height=1920)
        clf = FakeClassifier({"random_name.jpg": (1, 0.4)})  # below the 0.85 threshold
        classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR)
        self.assertEqual(self.media_class(fid)["verdict"], "photo")

    def test_high_confidence_clip_screenshot_without_camera_gps_faces(self):
        fid = self.add_file("odd_name.jpg", camera_make=None, camera_model=None)
        clf = FakeClassifier({"odd_name.jpg": (1, 0.9)})  # index 1 = screenshot, >= 0.85
        classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR)
        self.assertEqual(self.media_class(fid)["verdict"], "screenshot")

    def test_clip_below_threshold_keeps_heuristic_verdict_but_source_clip(self):
        fid = self.add_file("Screenshot_low.png", camera_make=None, camera_model=None)
        # CLIP is low-confidence (< the default 0.85 threshold) → use the heuristic
        clf = FakeClassifier({"Screenshot_low.png": (0, 0.4)})  # index 0 = photo, weak
        classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR)
        row = self.media_class(fid)
        self.assertEqual(row["verdict"], "screenshot")  # the heuristic won
        self.assertEqual(row["source"], "clip")  # but marked as checked by CLIP

    def test_threshold_from_config(self):
        self.naming["junk_threshold"] = 0.95
        self.cfg.naming = _naming_from(self.naming)
        # without camera/GPS/faces, so the veto does not fire and the threshold is what's checked
        fid = self.add_file("IMG_0004.jpg", camera_make=None, camera_model=None)
        clf = FakeClassifier({"IMG_0004.jpg": (1, 0.8)})  # below the custom 0.95 threshold
        classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR)
        self.assertEqual(self.media_class(fid)["verdict"], "photo")  # heuristic: photo

    def test_incrementality_skips_clip_rows(self):
        self.add_file("IMG_0005.jpg")
        clf = FakeClassifier({"IMG_0005.jpg": (0, 0.99)})
        classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR)
        # F15: without faces the classifier is called twice per pass (the main
        # 3 classes + a separate document pass) — both records for the same file.
        self.assertEqual(clf.seen_paths, ["/photos/IMG_0005.jpg"] * 2)
        stats2 = classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR)
        self.assertEqual(stats2.processed, 0)
        self.assertEqual(len(clf.seen_paths), 2)  # the second pass added nothing

    def test_heuristic_only_row_reprocessed_by_clip_later(self):
        fid = self.add_file("IMG_0006.jpg", camera_make=None, camera_model=None)
        classify(self.cfg, self.conn, use_clip=False)
        self.assertEqual(self.media_class(fid)["source"], "heuristic")
        clf = FakeClassifier({"IMG_0006.jpg": (2, 0.9)})  # index 2 = meme, >= 0.85
        stats2 = classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR)
        self.assertEqual(stats2.processed, 1)
        row = self.media_class(fid)
        self.assertEqual(row["verdict"], "meme")
        self.assertEqual(row["source"], "clip")

    def test_no_faces_high_confidence_document_is_document_even_with_camera(self):
        # camera_make/model are set by default (Canon/EOS) — a photographed document
        # is the target F15 case, document detection runs BEFORE the camera veto.
        fid = self.add_file("receipt.jpg")
        clf = FakeClassifier({}, doc_scores={"receipt.jpg": (_RECEIPT_IDX, 0.95)})
        classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR)
        row = self.media_class(fid)
        self.assertEqual(row["verdict"], "document")
        self.assertEqual(row["source"], "clip")
        self.assertAlmostEqual(row["score"], 0.95, places=5)

    def test_faces_veto_document_even_with_high_confidence_clip(self):
        # F13: a face in a photo — a portrait, not a document, even if CLIP is confident
        fid = self.add_file("portrait_with_paper.jpg", has_face=True)
        clf = FakeClassifier({}, doc_scores={"portrait_with_paper.jpg": (_RECEIPT_IDX, 0.99)})
        classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR)
        self.assertEqual(self.media_class(fid)["verdict"], "photo")

    def test_no_faces_document_score_below_threshold_stays_photo(self):
        fid = self.add_file("maybe_doc.jpg", camera_make=None, camera_model=None)
        clf = FakeClassifier({}, doc_scores={"maybe_doc.jpg": (_RECEIPT_IDX, 0.5)})  # < 0.9
        classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR)
        self.assertEqual(self.media_class(fid)["verdict"], "photo")

    def test_document_threshold_from_config(self):
        self.cfg.naming = _naming_from({"document_threshold": 0.6})
        fid = self.add_file("doc2.jpg", camera_make=None, camera_model=None)
        clf = FakeClassifier({}, doc_scores={"doc2.jpg": (_RECEIPT_IDX, 0.7)})  # < 0.9, >= 0.6
        classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR)
        self.assertEqual(self.media_class(fid)["verdict"], "document")

    def test_document_verdict_is_incremental(self):
        fid = self.add_file("receipt2.jpg")
        clf = FakeClassifier({}, doc_scores={"receipt2.jpg": (_RECEIPT_IDX, 0.95)})
        classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR)
        stats2 = classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR)
        self.assertEqual(stats2.processed, 0)
        self.assertEqual(self.media_class(fid)["verdict"], "document")

    def test_building_street_anti_class_keeps_photo_not_document(self):
        # F22: a travel photo of a building with a sign — CLIP is confident in the
        # "building" anti-class, the residual mass on "paper" is still below the
        # threshold, since document-score is computed only over the positive subclasses.
        fid = self.add_file("IMG_20230223_120308.jpg", gps_lat=55.75)

        class StreetClf:
            def __call__(self, paths, prompts):
                if len(prompts) == len(_DOCUMENT_CLASSES):
                    row = np.full(len(prompts), 0.02, dtype=np.float32)
                    row[1] = 0.6   # the "building" anti-class dominates
                    row[6] = 0.15  # a bit of "paper", but below the 0.9 threshold
                    return np.tile(row, (len(paths), 1))
                out = np.zeros((len(paths), len(prompts)), dtype=np.float32)
                out[:, 0] = 0.99  # confidently "photo" over the main 3 classes
                return out

        classify(self.cfg, self.conn, classifier=StreetClf(), text_detector=NO_OCR)
        self.assertEqual(self.media_class(fid)["verdict"], "photo")

    def test_screenshot_name_with_faces_overrides_veto(self):
        # F22: an avatar (face) on a LinkedIn/Telegram screenshot does not make the
        # file a real photo — the explicit Screenshot_ name overrides the F13 veto.
        fid = self.add_file("Screenshot_20231104_133830_LinkedIn.jpg",
                            camera_make=None, camera_model=None, has_face=True)
        clf = FakeClassifier({})
        classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR)
        self.assertEqual(self.media_class(fid)["verdict"], "screenshot")

    def test_screenshot_name_with_gps_and_faces_overrides_veto(self):
        fid = self.add_file("Screenshot_gps.png", camera_make=None, camera_model=None,
                            gps_lat=55.75, has_face=True)
        clf = FakeClassifier({})
        classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR)
        self.assertEqual(self.media_class(fid)["verdict"], "screenshot")

    def test_screenshot_name_without_face_weak_clip_is_screenshot(self):
        # F13 behaviour preserved: without a face, low CLIP → the heuristic.
        fid = self.add_file("Screenshot_plain.png", camera_make=None, camera_model=None)
        clf = FakeClassifier({"Screenshot_plain.png": (0, 0.4)})  # below the threshold
        classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR)
        self.assertEqual(self.media_class(fid)["verdict"], "screenshot")

    def test_screenshot_name_overrides_high_confidence_document(self):
        # the name outranks document detection (brief F22, item 4): a screenshot of a
        # document in a messenger — still a screenshot, not a document.
        fid = self.add_file("Screenshot_doc.png", camera_make=None, camera_model=None)
        clf = FakeClassifier({}, doc_scores={"Screenshot_doc.png": (_RECEIPT_IDX, 0.99)})
        classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR)
        self.assertEqual(self.media_class(fid)["verdict"], "screenshot")

    def test_skips_duplicates_errors_and_videos(self):
        canon = self.add_file("IMG_0007.jpg")
        self.add_file("dup.jpg", dup_of=canon)
        self.add_file("broken.jpg", error="boom")
        self.add_file("clip.mp4", media_type="video")
        stats = classify(self.cfg, self.conn, use_clip=False)
        self.assertEqual(stats.total, 1)
        self.assertEqual(stats.processed, 1)

    def test_faces_signal_prevents_false_screenshot_verdict(self):
        # without CLIP: the name does not match the Screenshot_ pattern → photo
        # regardless of size/faces (the screen-ratio heuristic was removed in F13)
        fid = self.add_file("random.jpg", camera_make=None, camera_model=None,
                            width=1080, height=1920, has_face=True)
        classify(self.cfg, self.conn, use_clip=False)
        self.assertEqual(self.media_class(fid)["verdict"], "photo")

    def test_progress_first_call_has_full_total_heuristic_only(self):
        # F52 (#37): use_clip=False used not to call progress at all.
        self.add_file("IMG_0100.jpg")
        self.add_file("IMG_0101.jpg")
        calls = []
        classify(self.cfg, self.conn, use_clip=False,
                progress=lambda done, total: calls.append((done, total)))
        self.assertTrue(calls)
        self.assertEqual(calls[0], (0, 2))
        self.assertEqual(calls[-1], (2, 2))

    def test_progress_first_call_has_full_total_with_clip(self):
        # F52 (#37): a small stage (< clip_batch_size) — total must be known from the
        # first call, not "0 of 0".
        self.add_file("IMG_0200.jpg", camera_make=None, camera_model=None)
        calls = []
        clf = FakeClassifier({"IMG_0200.jpg": (0, 0.99)})
        classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR,
                progress=lambda done, total: calls.append((done, total)))
        self.assertTrue(calls)
        self.assertEqual(calls[0], (0, 1))
        self.assertEqual(calls[-1], (1, 1))


class TestPolygonArea(unittest.TestCase):
    def test_rectangle(self):
        from sorta.junk import _polygon_area
        self.assertAlmostEqual(_polygon_area([[0, 0], [10, 0], [10, 5], [0, 5]]), 50.0)

    def test_degenerate_is_zero(self):
        from sorta.junk import _polygon_area
        self.assertEqual(_polygon_area([[0, 0], [0, 0], [0, 0], [0, 0]]), 0.0)


class TestEasyocrTextFracDetectorDecode(unittest.TestCase):
    """F48: the detector must call imaging.decode_rgb with an aggressive draft_margin
    (perf fix: the default margin=2× kills JPEG-draft for typical max_edge/camera
    frame sizes, see junk.py/imaging.py) and pass downscale_px as max_edge, as before (F40)."""

    def test_decode_rgb_called_with_aggressive_draft_margin_and_maxpx(self):
        from sorta.junk import easyocr_text_frac_detector

        fake_easyocr = types.ModuleType("easyocr")

        class FakeReader:
            def __init__(self, *args, **kwargs):
                pass

            def detect(self, arr):
                return ([[]], [[]])  # no boxes — text_frac=0.0

        fake_easyocr.Reader = FakeReader

        calls = []
        real_decode_rgb = imaging.decode_rgb

        def spy_decode_rgb(path, max_edge=None, **kwargs):
            calls.append((max_edge, kwargs))
            return real_decode_rgb(path, max_edge=max_edge, **kwargs)

        with unittest.mock.patch.dict(sys.modules, {"easyocr": fake_easyocr}), \
                unittest.mock.patch.object(imaging, "decode_rgb", spy_decode_rgb):
            detector = easyocr_text_frac_detector(800)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "x.jpg"
                Image.new("RGB", (64, 64)).save(path, "JPEG")
                frac = detector(str(path), 64, 64)

        self.assertEqual(frac, 0.0)
        self.assertEqual(len(calls), 1)
        max_edge, kwargs = calls[0]
        self.assertEqual(max_edge, 800)
        self.assertEqual(kwargs.get("draft_margin"), imaging._DRAFT_MARGIN_AGGRESSIVE)


class TestOcrTextFrac(unittest.TestCase):
    """F37 (Phase A): the FP gate (document->photo at low text_frac) and the FN rescue
    (photo->document at high text_frac) on top of the CLIP verdict.
    Screenshot/meme are not touched. easyocr is not called — text_detector is mocked
    (no model downloads in the gates)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.naming = {}
        self.cfg = Config(sources=[Path(self.tmp.name)],
                          database=Path(self.tmp.name) / "test.db",
                          naming=_naming_from(self.naming))
        self.conn = connect(self.cfg.database)
        self._n = 0

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def add_file(self, name, camera_make="Canon", camera_model="EOS",
                 has_face=False, width=4000, height=3000):
        self._n += 1
        cur = self.conn.execute(
            """INSERT INTO files (path, size, mtime, ext, media_type, width, height,
                   camera_make, camera_model, gps_lat, indexed_at)
               VALUES (?, 1000, 0, 'jpg', 'photo', ?, ?, ?, ?, NULL, '2026-01-01')""",
            (f"/photos/{name}", width, height, camera_make, camera_model))
        fid = cur.lastrowid
        if has_face:
            self.conn.execute(
                "INSERT INTO faces (file_id, bbox, embedding) VALUES (?, '[1,2,3,4]', ?)",
                (fid, b"\x00" * 4))
        self.conn.commit()
        return fid

    def media_class(self, fid):
        return self.conn.execute(
            "SELECT verdict, source, score FROM media_class WHERE file_id = ?",
            (fid,)).fetchone()

    def test_low_text_frac_downgrades_clip_document_to_photo(self):
        # beach/scene: CLIP confidently "document", but almost no text in the frame
        fid = self.add_file("beach.jpg")
        clf = FakeClassifier({}, doc_scores={"beach.jpg": (_RECEIPT_IDX, 0.95)})
        classify(self.cfg, self.conn, classifier=clf,
                 text_detector=lambda p, w, h: 0.01)
        row = self.media_class(fid)
        self.assertEqual(row["verdict"], "photo")
        self.assertEqual(row["source"], "ocr")
        self.assertAlmostEqual(row["score"], 0.01, places=5)

    def test_high_text_frac_rescues_clip_photo_to_document(self):
        # a medical form: CLIP is unsure (stays 'photo' via the camera veto), but the
        # frame is densely covered with text; doc_score 0.5 in the rescue zone (>= the
        # default 0.3 text_rescue_docscore_min, F38) — OCR is called.
        fid = self.add_file("medform.jpg")
        clf = FakeClassifier({}, doc_scores={"medform.jpg": (_RECEIPT_IDX, 0.5)})
        classify(self.cfg, self.conn, classifier=clf,
                 text_detector=lambda p, w, h: 0.5)
        row = self.media_class(fid)
        self.assertEqual(row["verdict"], "document")
        self.assertEqual(row["source"], "ocr")
        self.assertAlmostEqual(row["score"], 0.5, places=5)

    def test_mid_text_frac_does_not_change_verdict(self):
        # between the thresholds — neither the gate nor the rescue fires
        fid = self.add_file("plain.jpg")
        clf = FakeClassifier({"plain.jpg": (0, 0.99)})  # confidently "photo"
        classify(self.cfg, self.conn, classifier=clf,
                 text_detector=lambda p, w, h: 0.2)
        row = self.media_class(fid)
        self.assertEqual(row["verdict"], "photo")
        self.assertEqual(row["source"], "clip")

    def test_none_text_frac_is_treated_as_no_signal(self):
        # the detector could not compute the signal (e.g. width/height unknown)
        # -> we do not touch the verdict
        fid = self.add_file("unknown_size.jpg")
        clf = FakeClassifier({}, doc_scores={"unknown_size.jpg": (_RECEIPT_IDX, 0.95)})
        classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR)
        row = self.media_class(fid)
        self.assertEqual(row["verdict"], "document")
        self.assertEqual(row["source"], "clip")

    def test_screenshot_not_affected_by_high_text_frac(self):
        # the explicit Screenshot_ name -> screenshot; a high text_frac must not
        # turn it into a document
        fid = self.add_file("Screenshot_receipt.png", camera_make=None,
                            camera_model=None)
        clf = FakeClassifier({})
        classify(self.cfg, self.conn, classifier=clf,
                 text_detector=lambda p, w, h: 0.9)
        self.assertEqual(self.media_class(fid)["verdict"], "screenshot")

    def test_meme_not_affected_by_low_text_frac(self):
        # CLIP confidently "meme" (no veto) -> a low text_frac must not turn it
        # into photo (OCR touches only document<->photo)
        fid = self.add_file("meme.jpg", camera_make=None, camera_model=None)
        clf = FakeClassifier({"meme.jpg": (2, 0.99)})  # index 2 = meme
        classify(self.cfg, self.conn, classifier=clf,
                 text_detector=lambda p, w, h: 0.01)
        self.assertEqual(self.media_class(fid)["verdict"], "meme")

    def test_faces_veto_ocr_rescue(self):
        # a face in a photo -> the OCR rescue does not apply, even with dense text
        # (the same veto as document-CLIP: portraits are never documents)
        fid = self.add_file("portrait_with_text.jpg", has_face=True)
        clf = FakeClassifier({"portrait_with_text.jpg": (0, 0.99)})
        classify(self.cfg, self.conn, classifier=clf,
                 text_detector=lambda p, w, h: 0.9)
        self.assertEqual(self.media_class(fid)["verdict"], "photo")

    def test_thresholds_from_config(self):
        # F37/F38: text_frac_document is read via getattr(cfg.naming, ..., default) —
        # attached directly to the frozen instance (the same pattern by which the
        # field will later become a real dataclass attribute). The custom threshold
        # 0.4 is HIGHER than the default (0.15, F38) — text_frac=0.2 would pass the
        # default but not the custom one: proving the config is read, not the default constant.
        object.__setattr__(self.cfg.naming, "text_frac_document", 0.4)
        fid = self.add_file("custom_threshold.jpg")
        # doc_score 0.5 in the rescue zone (>= the default 0.3 text_rescue_docscore_min,
        # F38), otherwise the gate blocks the OCR call before the text_frac_document threshold.
        clf = FakeClassifier({"custom_threshold.jpg": (0, 0.99)},
                             doc_scores={"custom_threshold.jpg": (_RECEIPT_IDX, 0.5)})
        classify(self.cfg, self.conn, classifier=clf,
                 text_detector=lambda p, w, h: 0.2)  # >= the default 0.15, but < the custom 0.4
        row = self.media_class(fid)
        self.assertEqual(row["verdict"], "photo")
        self.assertEqual(row["source"], "clip")

    def test_text_detector_receives_width_and_height(self):
        seen = []

        def spy(path, width, height):
            seen.append((path, width, height))
            return None

        fid = self.add_file("dims.jpg", width=1234, height=5678)
        # doc_score in the rescue zone (the F38 gate), otherwise the detector is not called.
        clf = FakeClassifier({"dims.jpg": (0, 0.99)},
                             doc_scores={"dims.jpg": (_RECEIPT_IDX, 0.5)})
        classify(self.cfg, self.conn, classifier=clf, text_detector=spy)
        self.assertEqual(seen, [("/photos/dims.jpg", 1234, 5678)])
        self.assertEqual(self.media_class(fid)["verdict"], "photo")


class TestOcrRescueDocscoreGate(unittest.TestCase):
    """F38: the rescue branch (verdict='photo' -> 'document') calls OCR only if the
    document-CLIP already "doubts" (doc_score >= text_rescue_docscore_min) — a clear
    scene (doc_score≈0) spends no OCR call. The FP gate (verdict='document' -> 'photo')
    is not limited by this threshold."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.naming = {}
        self.cfg = Config(sources=[Path(self.tmp.name)],
                          database=Path(self.tmp.name) / "test.db",
                          naming=_naming_from(self.naming))
        self.conn = connect(self.cfg.database)
        self._n = 0

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def add_file(self, name, camera_make="Canon", camera_model="EOS"):
        self._n += 1
        cur = self.conn.execute(
            """INSERT INTO files (path, size, mtime, ext, media_type, width, height,
                   camera_make, camera_model, gps_lat, indexed_at)
               VALUES (?, 1000, 0, 'jpg', 'photo', 4000, 3000, ?, ?, NULL, '2026-01-01')""",
            (f"/photos/{name}", camera_make, camera_model))
        fid = cur.lastrowid
        self.conn.commit()
        return fid

    def media_class(self, fid):
        return self.conn.execute(
            "SELECT verdict, source, score FROM media_class WHERE file_id = ?",
            (fid,)).fetchone()

    def test_low_doc_score_skips_ocr_rescue(self):
        # scene: doc_score clearly below the default text_rescue_docscore_min (0.3) ->
        # the OCR detector is not called at all, even if text_frac were deliberately
        # high. The verdict stays photo (via the camera veto).
        fid = self.add_file("scene.jpg")
        calls = []

        def counting_detector(p, w, h):
            calls.append(p)
            return 0.9  # a deliberately high text_frac — must not be used

        clf = FakeClassifier({}, doc_scores={"scene.jpg": (_RECEIPT_IDX, 0.02)})
        classify(self.cfg, self.conn, classifier=clf, text_detector=counting_detector)
        self.assertEqual(calls, [])
        row = self.media_class(fid)
        self.assertEqual(row["verdict"], "photo")
        self.assertEqual(row["source"], "clip")

    def test_high_doc_score_calls_ocr_rescue(self):
        # doc_score in the rescue zone (0.3..document_threshold) -> the detector is
        # called, a high text_frac rescues into document.
        fid = self.add_file("doubt.jpg")
        calls = []

        def counting_detector(p, w, h):
            calls.append(p)
            return 0.5

        clf = FakeClassifier({}, doc_scores={"doubt.jpg": (_RECEIPT_IDX, 0.5)})
        classify(self.cfg, self.conn, classifier=clf, text_detector=counting_detector)
        self.assertEqual(len(calls), 1)
        row = self.media_class(fid)
        self.assertEqual(row["verdict"], "document")
        self.assertEqual(row["source"], "ocr")

    def test_text_rescue_docscore_min_from_config(self):
        # the threshold is read via getattr(cfg.naming, "text_rescue_docscore_min",
        # default) — attached directly to the frozen instance (the F30 pattern, later
        # a real dataclass field). The default 0.3 would block this doc_score (0.15);
        # a custom low 0.1 lets it through.
        object.__setattr__(self.cfg.naming, "text_rescue_docscore_min", 0.1)
        fid = self.add_file("lowdoc.jpg")
        clf = FakeClassifier({}, doc_scores={"lowdoc.jpg": (_RECEIPT_IDX, 0.15)})
        classify(self.cfg, self.conn, classifier=clf,
                 text_detector=lambda p, w, h: 0.5)
        row = self.media_class(fid)
        self.assertEqual(row["verdict"], "document")
        self.assertEqual(row["source"], "ocr")

    def test_fp_gate_not_limited_by_rescue_threshold(self):
        # the FP gate (document -> photo at low text_frac) works independently of
        # text_rescue_docscore_min — even a deliberately huge rescue threshold (0.99)
        # does not block OCR for the document branch.
        object.__setattr__(self.cfg.naming, "text_rescue_docscore_min", 0.99)
        fid = self.add_file("beach2.jpg")
        clf = FakeClassifier({}, doc_scores={"beach2.jpg": (_RECEIPT_IDX, 0.95)})
        classify(self.cfg, self.conn, classifier=clf,
                 text_detector=lambda p, w, h: 0.01)
        row = self.media_class(fid)
        self.assertEqual(row["verdict"], "photo")
        self.assertEqual(row["source"], "ocr")


class TestScreenshotsDirOverride(unittest.TestCase):
    """F29: a file in a Screenshots/Screenshot folder that CLIP gave 'photo' is
    overridden to 'screenshot'. document/meme are not touched."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.naming = {}
        self.cfg = Config(sources=[Path(self.tmp.name)],
                          database=Path(self.tmp.name) / "test.db",
                          naming=_naming_from(self.naming))
        self.conn = connect(self.cfg.database)
        self._n = 0

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def add_file(self, path, camera_make=None, camera_model=None,
                 gps_lat=None, has_face=False):
        self._n += 1
        cur = self.conn.execute(
            """INSERT INTO files (path, size, mtime, ext, media_type, width, height,
                   camera_make, camera_model, gps_lat, indexed_at)
               VALUES (?, 1000, 0, 'jpg', 'photo', 4000, 3000, ?, ?, ?, '2026-01-01')""",
            (path, camera_make, camera_model, gps_lat))
        fid = cur.lastrowid
        if has_face:
            self.conn.execute(
                "INSERT INTO faces (file_id, bbox, embedding) VALUES (?, '[1,2,3,4]', ?)",
                (fid, b"\x00" * 4))
        self.conn.commit()
        return fid

    def media_class(self, fid):
        return self.conn.execute(
            "SELECT verdict, source, score FROM media_class WHERE file_id = ?",
            (fid,)).fetchone()

    def test_clip_photo_in_screenshots_dir_becomes_screenshot(self):
        fid = self.add_file("/photos/Screenshots/foo.png")
        clf = FakeClassifier({"foo.png": (0, 0.99)})  # index 0 = photo
        classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR)
        row = self.media_class(fid)
        self.assertEqual(row["verdict"], "screenshot")
        self.assertEqual(row["source"], "clip")

    def test_veto_photo_in_screenshots_dir_becomes_screenshot(self):
        # camera EXIF vetoes CLIP into 'photo' -> the folder overrides it further
        fid = self.add_file("/photos/Screenshots/cam.jpg",
                            camera_make="Canon", camera_model="EOS")
        clf = FakeClassifier({"cam.jpg": (2, 0.99)})  # CLIP would want 'meme'
        classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR)
        self.assertEqual(self.media_class(fid)["verdict"], "screenshot")

    def test_document_in_screenshots_dir_not_overridden(self):
        fid = self.add_file("/photos/Screenshots/doc.png")
        clf = FakeClassifier({}, doc_scores={"doc.png": (_RECEIPT_IDX, 0.95)})
        classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR)
        self.assertEqual(self.media_class(fid)["verdict"], "document")

    def test_meme_in_screenshots_dir_not_overridden(self):
        fid = self.add_file("/photos/Screenshots/m.png",
                            camera_make=None, camera_model=None)
        clf = FakeClassifier({"m.png": (2, 0.99)})  # index 2 = meme
        classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR)
        self.assertEqual(self.media_class(fid)["verdict"], "meme")

    def test_photo_outside_screenshots_dir_unaffected(self):
        fid = self.add_file("/photos/DCIM/foo.png")
        clf = FakeClassifier({"foo.png": (0, 0.99)})  # index 0 = photo
        classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR)
        self.assertEqual(self.media_class(fid)["verdict"], "photo")

    def test_case_insensitive_and_nested_screenshots_dir(self):
        fid = self.add_file("/photos/DCIM/SCREENSHOTS/sub/x.png")
        clf = FakeClassifier({"x.png": (0, 0.99)})  # index 0 = photo
        classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR)
        self.assertEqual(self.media_class(fid)["verdict"], "screenshot")


class TestCachingEquivalence(unittest.TestCase):
    """F19: classify() on CachingFeatureClassifier (image-feature cache) gives the
    same media_class verdicts as on the plain (non-caching) classifier — the
    optimization does not change the veto/threshold/document-detection logic."""

    def _run(self, classifier, db_name):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        cfg = Config(sources=[Path(tmp.name)], database=Path(tmp.name) / db_name,
                    naming=_naming_from({}))
        conn = connect(cfg.database)
        self.addCleanup(conn.close)
        files = [
            # (name, camera_make, camera_model, has_face)
            ("odd_name.jpg", None, None, False),        # screenshot, CLIP confident
            ("random.jpg", None, None, True),            # face veto -> photo
            ("160A3747.jpg", "Canon", "EOS", False),      # camera veto -> photo
            ("receipt.jpg", "Canon", "EOS", False),       # document (no faces)
        ]
        for name, make, model, has_face in files:
            cur = conn.execute(
                """INSERT INTO files (path, size, mtime, ext, media_type, width,
                       height, camera_make, camera_model, gps_lat, indexed_at)
                   VALUES (?, 1000, 0, 'jpg', 'photo', 4000, 3000, ?, ?, NULL,
                           '2026-01-01')""",
                (f"/photos/{name}", make, model))
            fid = cur.lastrowid
            if has_face:
                conn.execute(
                    "INSERT INTO faces (file_id, bbox, embedding) VALUES (?, '[1,2,3,4]', ?)",
                    (fid, b"\x00" * 4))
        conn.commit()
        classify(cfg, conn, classifier=classifier, text_detector=NO_OCR)
        return {
            r["path"]: (r["verdict"], r["source"])
            for r in conn.execute(
                """SELECT f.path, mc.verdict, mc.source FROM files f
                   JOIN media_class mc ON mc.file_id = f.id ORDER BY f.path""")
        }

    def test_caching_classifier_matches_plain(self):
        scores = {"odd_name.jpg": (1, 0.9), "160A3747.jpg": (2, 0.99)}
        doc_scores = {"receipt.jpg": (_RECEIPT_IDX, 0.95)}
        plain = FakeClassifier(scores, doc_scores)
        caching = _make_caching_classifier(scores, doc_scores)
        plain_result = self._run(plain, "plain.db")
        caching_result = self._run(caching, "caching.db")
        self.assertEqual(plain_result, caching_result)
        self.assertEqual(plain_result["/photos/odd_name.jpg"], ("screenshot", "clip"))
        self.assertEqual(plain_result["/photos/random.jpg"], ("photo", "clip"))
        self.assertEqual(plain_result["/photos/160A3747.jpg"], ("photo", "clip"))
        self.assertEqual(plain_result["/photos/receipt.jpg"], ("document", "clip"))


class TestVlmTier(unittest.TestCase):
    """F37 (Phase B): the deep tier (VLM), opt-in via naming.vlm_enabled.

    We mock classify_media entirely — no transformers/model download in tests.
    vlm_enabled is read via getattr(cfg.naming, ...) — the field is attached directly
    to the frozen instance (the F30/F37-A pattern), later added to NamingConfig.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = Config(sources=[Path(self.tmp.name)],
                          database=Path(self.tmp.name) / "test.db",
                          naming=_naming_from({}))
        self.conn = connect(self.cfg.database)
        self._n = 0

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def _enable_vlm(self, enabled=True):
        object.__setattr__(self.cfg.naming, "vlm_enabled", enabled)

    def add_file(self, name, camera_make="Canon", camera_model="EOS",
                 has_face=False, width=4000, height=3000):
        self._n += 1
        cur = self.conn.execute(
            """INSERT INTO files (path, size, mtime, ext, media_type, width, height,
                   camera_make, camera_model, gps_lat, indexed_at)
               VALUES (?, 1000, 0, 'jpg', 'photo', ?, ?, ?, ?, NULL, '2026-01-01')""",
            (f"/photos/{name}", width, height, camera_make, camera_model))
        fid = cur.lastrowid
        if has_face:
            self.conn.execute(
                "INSERT INTO faces (file_id, bbox, embedding) VALUES (?, '[1,2,3,4]', ?)",
                (fid, b"\x00" * 4))
        self.conn.commit()
        return fid

    def media_class(self, fid):
        return self.conn.execute(
            "SELECT verdict, source, score FROM media_class WHERE file_id = ?",
            (fid,)).fetchone()

    def _candidate_clf(self, name, main=(0, 0.99)):
        # #14/V1: doc_score 0.5 -> the file becomes a VLM candidate (>= text_rescue
        # 0.3), but not 'document' (< document_threshold 0.9); the fast verdict is
        # photo, the VLM decides the final one.
        return FakeClassifier({name: main}, doc_scores={name: (_RECEIPT_IDX, 0.5)})

    def test_vlm_disabled_does_not_build_or_call_vlm(self):
        # vlm_enabled=False (default) -> the Phase A/CLIP path, the factory is not called.
        fid = self.add_file("IMG_0100.jpg", camera_make=None, camera_model=None)
        factory_calls = []

        def factory(model_name):
            factory_calls.append(model_name)
            raise AssertionError("the VLM factory must not be built when vlm_enabled=False")

        clf = FakeClassifier({"IMG_0100.jpg": (0, 0.99)})
        classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR,
                 vlm_classifier_factory=factory)
        self.assertEqual(factory_calls, [])
        self.assertEqual(self.media_class(fid)["source"], "clip")

    def test_clean_photo_not_a_candidate_vlm_not_called(self):
        # #14/V1 the gate's essence: a clean photo (doc_score/product_score low) is NOT
        # a candidate -> the VLM is not called, the fast verdict stays (source='clip').
        self._enable_vlm()
        fid = self.add_file("beach.jpg", camera_make=None, camera_model=None)
        clf = FakeClassifier({"beach.jpg": (0, 0.99)})  # doc/prod low by default

        def vlm(_path):
            raise AssertionError("the VLM must not be called for a non-candidate (a clean photo)")

        stats = classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR,
                         vlm_classifier=vlm)
        row = self.media_class(fid)
        self.assertEqual(row["verdict"], "photo")
        self.assertEqual(row["source"], "clip")
        self.assertEqual(stats.vlm_candidates, 0)

    def test_vlm_enabled_mock_personal_photo(self):
        self._enable_vlm()
        fid = self.add_file("holiday.jpg", camera_make=None, camera_model=None)
        clf = self._candidate_clf("holiday.jpg")
        classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR,
                 vlm_classifier=lambda path: "personal_photo")
        row = self.media_class(fid)
        self.assertEqual(row["verdict"], "photo")
        self.assertEqual(row["source"], "vlm")  # a candidate is marked vlm even without a verdict change

    def test_vlm_enabled_mock_document(self):
        self._enable_vlm()
        fid = self.add_file("scan.jpg", camera_make=None, camera_model=None)
        clf = self._candidate_clf("scan.jpg")
        classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR,
                 vlm_classifier=lambda path: "document")
        row = self.media_class(fid)
        self.assertEqual(row["verdict"], "document")
        self.assertEqual(row["source"], "vlm")

    def test_vlm_enabled_mock_product(self):
        self._enable_vlm()
        fid = self.add_file("listing.jpg", camera_make=None, camera_model=None)
        clf = self._candidate_clf("listing.jpg")
        classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR,
                 vlm_classifier=lambda path: "product")
        row = self.media_class(fid)
        self.assertEqual(row["verdict"], "product")
        self.assertEqual(row["source"], "vlm")

    def test_product_signal_makes_candidate(self):
        # #14/V1 product prefilter: doc_score low, but product_score high ->
        # the file is still a candidate (a product "as a photo" is not missed) -> VLM -> product.
        self._enable_vlm()
        fid = self.add_file("shoe.jpg", camera_make=None, camera_model=None)
        clf = FakeClassifier({"shoe.jpg": (0, 0.99)},
                             prod_scores={"shoe.jpg": (_PRODUCT_POS_IDX, 0.8)})
        classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR,
                 vlm_classifier=lambda path: "product")
        row = self.media_class(fid)
        self.assertEqual(row["verdict"], "product")
        self.assertEqual(row["source"], "vlm")

    def test_unknown_vlm_label_falls_back_to_fast_verdict(self):
        self._enable_vlm()
        fid = self.add_file("odd.jpg", camera_make=None, camera_model=None)
        clf = self._candidate_clf("odd.jpg")
        classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR,
                 vlm_classifier=lambda path: "garbage")
        row = self.media_class(fid)
        self.assertEqual(row["verdict"], "photo")  # an unknown label -> the fast verdict
        self.assertEqual(row["source"], "vlm")

    def test_heuristic_screenshot_name_overrides_vlm(self):
        self._enable_vlm()
        fid = self.add_file("Screenshot_20240101.png", camera_make=None,
                            camera_model=None)
        clf = FakeClassifier({"Screenshot_20240101.png": (0, 0.99)})

        def vlm(_path):
            raise AssertionError("the VLM must not be called for a screenshot (not a candidate)")

        classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR,
                 vlm_classifier=vlm)
        row = self.media_class(fid)
        self.assertEqual(row["verdict"], "screenshot")
        # single flow: a screenshot goes the CLIP path (source='clip'), not the old
        # VLM branch; the point — the VLM does not touch it (not a candidate), the verdict is kept.
        self.assertEqual(row["source"], "clip")

    def test_factory_raises_falls_back_to_fast_tier(self):
        # GRACEFUL FALLBACK: the factory (real model build) raises ->
        # classify() does not crash, falls back to the fast tier (CLIP), media_class
        # is still written.
        self._enable_vlm()
        fid = self.add_file("IMG_0200.jpg", camera_make=None, camera_model=None)

        def broken_factory(model_name):
            raise RuntimeError("no CUDA / transformers not installed")

        clf = FakeClassifier({"IMG_0200.jpg": (0, 0.99)})
        stats = classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR,
                         vlm_classifier_factory=broken_factory)
        row = self.media_class(fid)
        self.assertIsNotNone(row)
        self.assertEqual(row["source"], "clip")
        self.assertEqual(stats.by_verdict.get("photo", 0)
                          + stats.by_verdict.get("screenshot", 0)
                          + stats.by_verdict.get("meme", 0)
                          + stats.by_verdict.get("document", 0), 1)

    def test_vlm_error_on_one_file_does_not_abort_run(self):
        # #31: a VLM runtime error on a frame (e.g. a CUDA assert) does NOT crash the
        # run — the file keeps the fast verdict (source='clip'), classify returns.
        self._enable_vlm()
        fid = self.add_file("boom.jpg", camera_make=None, camera_model=None)
        clf = self._candidate_clf("boom.jpg")

        def vlm(_path):
            raise RuntimeError("CUDA error: device-side assert triggered")

        stats = classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR,
                         vlm_classifier=vlm)
        row = self.media_class(fid)
        self.assertIsNotNone(row)
        self.assertEqual(row["source"], "clip")  # the VLM failed -> the fast verdict is kept
        self.assertEqual(stats.vlm_candidates, 1)
        self.assertEqual(stats.vlm_applied, 0)

    def test_vlm_enabled_none_returned_vlm_classifier_not_built_when_use_clip_false(self):
        # use_clip=False — an explicit heuristics-only mode, deep does not enter it.
        self._enable_vlm()
        fid = self.add_file("Screenshot_off.png", camera_make=None, camera_model=None)

        def factory(model_name):
            raise AssertionError("the VLM must not be built when use_clip=False")

        classify(self.cfg, self.conn, use_clip=False, vlm_classifier_factory=factory)
        self.assertEqual(self.media_class(fid)["source"], "heuristic")

    def test_incrementality_skips_vlm_candidates(self):
        self._enable_vlm()
        self.add_file("IMG_0300.jpg", camera_make=None, camera_model=None)
        calls = []

        def vlm(path):
            calls.append(path)
            return "personal_photo"

        clf = self._candidate_clf("IMG_0300.jpg")
        classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR,
                 vlm_classifier=vlm)
        self.assertEqual(len(calls), 1)
        stats2 = classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR,
                          vlm_classifier=vlm)
        self.assertEqual(stats2.processed, 0)  # a candidate marked 'vlm' -> not in todo
        self.assertEqual(len(calls), 1)  # the second run did not call the VLM again

    def test_switching_from_fast_to_deep_reprocesses_clip_rows(self):
        # F37 Phase B (item 9): a deep re-run reclassifies non-'vlm' rows.
        fid = self.add_file("IMG_0400.jpg", camera_make=None, camera_model=None)
        clf_fast = FakeClassifier({"IMG_0400.jpg": (0, 0.99)})
        classify(self.cfg, self.conn, classifier=clf_fast, text_detector=NO_OCR)
        self.assertEqual(self.media_class(fid)["source"], "clip")

        self._enable_vlm()
        clf = self._candidate_clf("IMG_0400.jpg")
        stats2 = classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR,
                          vlm_classifier=lambda path: "document")
        self.assertEqual(stats2.processed, 1)
        row = self.media_class(fid)
        self.assertEqual(row["verdict"], "document")
        self.assertEqual(row["source"], "vlm")

    def test_switching_from_deep_to_fast_reprocesses_vlm_rows(self):
        # switching deep -> fast: 'vlm' rows are reprocessed by CLIP
        # (the active-tier marker switches back to 'clip').
        self._enable_vlm()
        fid = self.add_file("IMG_0500.jpg", camera_make=None, camera_model=None)
        clf = self._candidate_clf("IMG_0500.jpg")
        classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR,
                 vlm_classifier=lambda path: "product")
        self.assertEqual(self.media_class(fid)["source"], "vlm")

        self._enable_vlm(False)
        clf_fast = FakeClassifier({"IMG_0500.jpg": (0, 0.99)})
        stats2 = classify(self.cfg, self.conn, classifier=clf_fast, text_detector=NO_OCR)
        self.assertEqual(stats2.processed, 1)
        self.assertEqual(self.media_class(fid)["source"], "clip")

    def test_existing_junk_tests_use_clip_source_marker_unaffected(self):
        # sanity: vlm_enabled is absent from config by default -> getattr gives
        # False, the behaviour matches the rest of the test suite (F13/F15/F22/F29).
        self.assertFalse(bool(getattr(self.cfg.naming, "vlm_enabled", False)))


if __name__ == "__main__":
    unittest.main()
