"""F155: the laplacian measured INSIDE the face, and the rescaling that makes it real.

The whole-frame number answers "how much detail is in this picture", which is not the
question the blur filter asks — a detailed sharp street and a smooth blurred face score
alike, and the filter built on it caught 2 of 33 blurred frames on a hand-checked sample.
A face is the one object comparable across frames, so the same variance taken inside it
means the same thing twice.

What the cases below check is the feature and its one lethal detail:

* the column fills where a real face is and stays NULL everywhere else — no face, only the
  `bbox = '[]'` marker, or a crop too small for the number to mean anything;
* THE COORDINATES ARE RESCALED. `faces.bbox` is written in pixels of the full original and
  the laplacian is taken over a preview several times smaller, so a box used as written
  falls off the frame. The measurement this feature came from made exactly that mistake and
  reported 100% recall over the 29 crops that happened to survive instead of the real 62% —
  a broken crop flatters the result rather than failing. Two cases pin it: the arithmetic
  of `face_crop_boxes`, and an end-to-end frame whose only sharp region sits where the
  rescaled box lands and nowhere else;
* several faces give the sharpest one;
* it costs NO EXTRA DECODE — a run over frames with faces decodes as many times as one over
  frames without;
* and the whole-frame number does not move because of any of it.

These cases use real files and the real detector: the feature is about pixels and
coordinates, and a mocked detector would test the mock.
"""
from __future__ import annotations

import unittest
import unittest.mock
from pathlib import Path

import numpy as np
from PIL import Image

from sorta import imaging, junk
from sorta.junk import FaceBoxes, face_crop_boxes, laplacian_variance, read_face_boxes
from tests.test_frame_quality import FrameQualityCase, QualityClassifier
from tests.test_junk import NO_OCR

# The preview the cases below are measured at. Smaller than the source on purpose — the
# rescaling is what is under test, and at a preview the size of the original every wrong
# scale would still land on the face.
PREVIEW_EDGE = 500
SOURCE_SIZE = (1000, 750)  # exactly twice the preview: the scale is 0.5 and checkable by eye


def flat_image(size: tuple[int, int] = SOURCE_SIZE, value: int = 128) -> Image.Image:
    """A frame with no detail anywhere — its laplacian is 0."""
    return Image.new("RGB", size, (value, value, value))


def no_landmark_model():
    """A 106-point model that finds no contour — what the suite runs with (see run_stage)."""
    return lambda _img, _box: None


def add_noise(img: Image.Image, box: tuple[int, int, int, int], seed: int = 7) -> None:
    """Paste high-frequency noise into `box` — the only sharp region of the frame."""
    left, top, right, bottom = box
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 255, (bottom - top, right - left, 3), dtype=np.uint8)
    img.paste(Image.fromarray(noise), (left, top))


class FaceSharpnessCase(FrameQualityCase):
    """Real JPEGs on disk, real face rows, and the pipeline's own detector."""

    def setUp(self):
        super().setUp()
        self.features(sharpness_max_edge=PREVIEW_EDGE)

    def add_photo(self, name, img, faces=(), orientation=None):
        """Write `img` as a JPEG and index it, with `faces` as ORIGINAL-frame boxes."""
        path = Path(self.tmp.name) / name
        params = {}
        if orientation is not None:
            exif = Image.Exif()
            exif[274] = orientation
            params["exif"] = exif
        img.save(path, "JPEG", quality=95, **params)
        width, height = img.size
        cur = self.conn.execute(
            """INSERT INTO files (path, size, mtime, ext, media_type, width, height,
                   camera_make, camera_model, gps_lat, indexed_at)
               VALUES (?, 1000, 0, 'jpg', 'photo', ?, ?, 'Canon', 'EOS', NULL,
                       '2026-01-01')""",
            (str(path), width, height))
        fid = cur.lastrowid
        for box in faces:
            self.conn.execute(
                "INSERT INTO faces (file_id, bbox, embedding) VALUES (?, ?, ?)",
                (fid, str(list(box)), b"\x00" * 4))
        self.conn.commit()
        return fid

    def add_no_face_marker(self, fid):
        """The `bbox = '[]'` row: "processed, nothing found" — not a face (F125's trap)."""
        self.conn.execute(
            "INSERT INTO faces (file_id, bbox, embedding) VALUES (?, '[]', ?)",
            (fid, b""))
        self.conn.commit()

    def run_stage(self, **kwargs):
        """classify() with the REAL sharpness detector — the pixels are the point here.

        F179: the real 106-point model is NOT one of them. It lives inside the `buffalo_l`
        set and `FaceAnalysis` downloads that set when it is missing, so a default here
        would put a few hundred megabytes of network into the test suite; the stub answers
        "no contour", which is the same path a machine without the [faces] extra takes.
        The eye geometry has a suite of its own (tests/test_closed_eyes_slice.py), and it
        injects a contour it can predict.
        """
        kwargs.setdefault("classifier", QualityClassifier())
        kwargs.setdefault("text_detector", NO_OCR)
        kwargs.setdefault("eye_landmarks_factory", no_landmark_model)
        return junk.classify(self.cfg, self.conn, **kwargs)

    def preview_of(self, fid):
        """The very frame the detector measures — for computing an expected value.

        F179: the stage decodes in COLOUR once a face has ever been found in this index —
        the 106-point contour is measured on colour pixels — and grayscale otherwise, so
        this mirrors that choice off the same fact (`faces_stage_ran`). Both paths measure
        the luma of the same preview and the two laplacians agree to ~0.15%, but the cases
        below compare to four decimal places, which is a claim about the ARRAY and not
        about the signal.
        """
        path = self.conn.execute(
            "SELECT path FROM files WHERE id = ?", (fid,)).fetchone()["path"]
        img = imaging.decode_rgb_preview(
            path, Path(path).stat().st_mtime, Path(path).stat().st_size,
            max_edge=PREVIEW_EDGE, grayscale=not junk.faces_stage_ran(self.conn),
            apply_orientation=True)
        assert img is not None
        return img


class TestCropRescaling(unittest.TestCase):
    """The arithmetic of the rescaling, on its own — brief test 2, the main one."""

    def test_a_preview_half_the_size_halves_the_coordinates(self):
        boxes = face_crop_boxes(FaceBoxes(((200.0, 100.0, 400.0, 300.0),), 1000.0),
                                (500, 375))
        self.assertEqual(boxes, [(100, 50, 200, 150)])

    def test_a_box_running_past_the_edge_is_clamped_inside_the_preview(self):
        # A face at the border, or rounding in the scale: the crop is of the part that is
        # inside the frame, and never of coordinates the array does not have.
        boxes = face_crop_boxes(FaceBoxes(((800.0, 600.0, 1100.0, 900.0),), 1000.0),
                                (500, 375))
        (left, top, right, bottom) = boxes[0]
        self.assertEqual((left, top), (400, 300))
        self.assertLessEqual(right, 500)
        self.assertLessEqual(bottom, 375)

    def test_boxes_below_the_minimum_size_are_left_out(self):
        # 40 px of a 1000 px original is 20 px of a 500 px preview — under the floor.
        self.assertEqual(
            face_crop_boxes(FaceBoxes(((0.0, 0.0, 40.0, 40.0),), 1000.0), (500, 375)), [])

    def test_boxes_without_a_scale_are_not_boxes(self):
        self.assertEqual(
            face_crop_boxes(FaceBoxes(((0.0, 0.0, 400.0, 400.0),), 0.0), (500, 375)), [])

    def test_a_preview_the_size_of_the_original_keeps_the_coordinates(self):
        boxes = face_crop_boxes(FaceBoxes(((100.0, 50.0, 300.0, 250.0),), 500.0),
                                (500, 375))
        self.assertEqual(boxes, [(100, 50, 300, 250)])


class TestTheCropLandsOnTheFace(FaceSharpnessCase):
    """Brief test 2, end to end: the number comes off the face and not off somewhere else."""

    def test_the_value_is_the_laplacian_of_the_rescaled_crop(self):
        img = flat_image()
        add_noise(img, (400, 200, 600, 400))  # the only sharp region of the frame
        fid = self.add_photo("face.jpg", img, faces=[(400, 200, 600, 400)])
        self.run_stage()

        # in the preview (half the size) that region is (200, 100)-(300, 200)
        expected = laplacian_variance(self.preview_of(fid).crop((200, 100, 300, 200)))
        row = self.quality(fid)
        self.assertAlmostEqual(row["face_sharpness"], expected, places=4)
        # and it is emphatically not what an unrescaled box would have measured: the same
        # coordinates used as written land on the flat part of the preview.
        self.assertGreater(row["face_sharpness"], row["sharpness"])

    def test_an_unrescaled_box_would_have_measured_flat_pixels(self):
        """The failure mode named in the brief, spelled out: it does not raise, it lies."""
        img = flat_image()
        add_noise(img, (400, 200, 600, 400))
        fid = self.add_photo("face.jpg", img, faces=[(400, 200, 600, 400)])
        self.run_stage()

        preview = self.preview_of(fid)
        naive = face_crop_boxes(FaceBoxes(((400.0, 200.0, 600.0, 400.0),), 500.0),
                                preview.size)
        self.assertLess(laplacian_variance(preview.crop(naive[0])),
                        self.quality(fid)["face_sharpness"] / 10)


class TestPopulation(FaceSharpnessCase):
    """Brief tests 1, 3 and 5: who gets a number and who gets NULL."""

    def test_a_frame_with_a_real_face_gets_the_number(self):
        img = flat_image()
        add_noise(img, (400, 200, 600, 400))
        fid = self.add_photo("face.jpg", img, faces=[(400, 200, 600, 400)])
        self.run_stage()
        self.assertIsNotNone(self.quality(fid)["face_sharpness"])

    def test_a_frame_with_no_faces_row_gets_null(self):
        fid = self.add_photo("landscape.jpg", flat_image())
        self.run_stage()
        row = self.quality(fid)
        self.assertIsNotNone(row["sharpness"])   # the frame WAS measured
        self.assertIsNone(row["face_sharpness"])  # there was simply no face on it

    def test_the_no_face_marker_is_not_a_face(self):
        """`bbox = '[]'` means "processed, nothing found" — 24 195 of 24 196 files have one."""
        fid = self.add_photo("nobody.jpg", flat_image())
        self.add_no_face_marker(fid)
        self.run_stage()
        self.assertIsNone(self.quality(fid)["face_sharpness"])

    def test_a_face_below_the_minimum_crop_size_is_null_not_zero(self):
        # 40 px of the original -> 20 px of the preview: too little for the variance to
        # mean anything, and 0.0 would read as "perfectly smooth", a different statement.
        img = flat_image()
        add_noise(img, (100, 100, 140, 140))
        fid = self.add_photo("tiny.jpg", img, faces=[(100, 100, 140, 140)])
        self.run_stage()
        self.assertIsNone(self.quality(fid)["face_sharpness"])

    def test_a_frame_without_width_is_not_guessed_at(self):
        """No size, no scale: the boxes are in units nothing can convert."""
        img = flat_image()
        add_noise(img, (400, 200, 600, 400))
        fid = self.add_photo("face.jpg", img, faces=[(400, 200, 600, 400)])
        self.conn.execute("UPDATE files SET width = NULL WHERE id = ?", (fid,))
        self.conn.commit()
        self.run_stage()
        self.assertIsNone(self.quality(fid)["face_sharpness"])


class TestSeveralFaces(FaceSharpnessCase):
    """Brief test 4: the sharpest face answers for the frame."""

    def test_the_sharpest_face_wins(self):
        img = flat_image()
        add_noise(img, (400, 200, 600, 400))  # one face in focus
        fid = self.add_photo("two.jpg", img, faces=[(0, 0, 200, 200),          # flat
                                                    (400, 200, 600, 400)])     # sharp
        self.run_stage()
        expected = laplacian_variance(self.preview_of(fid).crop((200, 100, 300, 200)))
        self.assertAlmostEqual(self.quality(fid)["face_sharpness"], expected, places=4)

    def test_the_order_of_the_rows_does_not_decide(self):
        img = flat_image()
        add_noise(img, (400, 200, 600, 400))
        fid = self.add_photo("two.jpg", img, faces=[(400, 200, 600, 400),
                                                    (0, 0, 200, 200)])
        self.run_stage()
        expected = laplacian_variance(self.preview_of(fid).crop((200, 100, 300, 200)))
        self.assertAlmostEqual(self.quality(fid)["face_sharpness"], expected, places=4)


class TestNoSecondPass(FaceSharpnessCase):
    """Brief test 6: the face number rides in the decode the stage already paid for."""

    def _decodes_for(self, names_with_faces):
        calls: list[str] = []
        real = imaging.decode_rgb_preview

        def counting(path, *args, **kwargs):
            calls.append(str(path))
            return real(path, *args, **kwargs)

        for i, has_face in enumerate(names_with_faces):
            img = flat_image()
            add_noise(img, (400, 200, 600, 400))
            self.add_photo(f"f{i}.jpg", img,
                           faces=[(400, 200, 600, 400)] if has_face else [])
        with unittest.mock.patch.object(junk.imaging, "decode_rgb_preview", counting):
            self.run_stage()
        return calls

    def test_frames_with_faces_decode_no_more_often_than_frames_without(self):
        with_faces = self._decodes_for([True, True, True])
        self.setUp()  # a second, independent index — the same three frames without faces
        without = self._decodes_for([False, False, False])
        self.assertEqual(len(with_faces), len(without))
        self.assertEqual(len(with_faces), 3)  # one decode per frame, faces or no faces

    def test_every_frame_is_decoded_exactly_once(self):
        calls = self._decodes_for([True, False, True])
        self.assertEqual(sorted(calls), sorted(set(calls)))


class TestTheFrameNumberDoesNotMove(FaceSharpnessCase):
    """Brief test 7: `sharpness` is what it was — the face half is added, not swapped in."""

    def test_the_same_pixels_give_the_same_frame_sharpness_with_and_without_a_face(self):
        img = flat_image()
        add_noise(img, (400, 200, 600, 400))
        with_face = self.add_photo("a.jpg", img, faces=[(400, 200, 600, 400)])
        without = self.add_photo("b.jpg", img)
        self.run_stage()
        self.assertAlmostEqual(self.quality(with_face)["sharpness"],
                               self.quality(without)["sharpness"], places=4)

    def test_the_frame_number_is_the_laplacian_of_the_whole_preview(self):
        img = flat_image()
        add_noise(img, (400, 200, 600, 400))
        fid = self.add_photo("a.jpg", img, faces=[(400, 200, 600, 400)])
        self.run_stage()
        self.assertAlmostEqual(self.quality(fid)["sharpness"],
                               laplacian_variance(self.preview_of(fid)), places=4)

    def test_an_exif_rotation_does_not_change_it(self):
        """The decode now applies the orientation (the boxes live in the rotated space).

        The laplacian kernel and the interior it is taken over are symmetric under every
        rotation and mirror an EXIF orientation can express, so the transform itself moves
        the variance by nothing. What is left is the resample: a 1000x750 frame scaled to
        500 px and then turned is not pixel-identical to the same frame turned and then
        scaled. That difference is ~0.01% here, four orders of magnitude below the width
        of any band this number is read against.
        """
        img = flat_image()
        add_noise(img, (400, 200, 600, 400))
        upright = self.add_photo("a.jpg", img)
        rotated = self.add_photo("b.jpg", img, orientation=6)  # 90° clockwise
        self.run_stage()
        self.assertAlmostEqual(
            self.quality(upright)["sharpness"] / self.quality(rotated)["sharpness"],
            1.0, places=3)


class TestReadFaceBoxes(FaceSharpnessCase):
    """The reader: what reaches the detector, and what deliberately does not."""

    def test_boxes_come_back_with_the_long_edge_of_their_frame(self):
        fid = self.add_photo("a.jpg", flat_image(), faces=[(10, 20, 110, 120)])
        boxes = read_face_boxes(self.conn, [fid])
        self.assertEqual(boxes[fid].boxes, ((10.0, 20.0, 110.0, 120.0),))
        self.assertEqual(boxes[fid].long_edge, 1000.0)  # max(1000, 750)

    def test_the_no_face_marker_never_reaches_the_detector(self):
        fid = self.add_photo("a.jpg", flat_image())
        self.add_no_face_marker(fid)
        self.assertEqual(read_face_boxes(self.conn, [fid]), {})

    def test_an_unreadable_bbox_costs_that_frame_its_face_number_and_nothing_else(self):
        fid = self.add_photo("a.jpg", flat_image())
        self.conn.execute(
            "INSERT INTO faces (file_id, bbox, embedding) VALUES (?, 'not json', ?)",
            (fid, b"\x00" * 4))
        self.conn.commit()
        self.assertEqual(read_face_boxes(self.conn, [fid]), {})
        self.run_stage()
        self.assertIsNotNone(self.quality(fid)["sharpness"])
        self.assertIsNone(self.quality(fid)["face_sharpness"])

    def test_a_frame_that_was_never_asked_about_is_absent(self):
        fid = self.add_photo("a.jpg", flat_image(), faces=[(10, 20, 110, 120)])
        self.assertEqual(read_face_boxes(self.conn, []), {})
        self.assertIn(fid, read_face_boxes(self.conn, [fid]))


class TestTheColumnIsReadableBack(FaceSharpnessCase):
    """`read_frame_quality` is the one place the NULL/number distinction is decided."""

    def test_the_number_survives_the_round_trip(self):
        img = flat_image()
        add_noise(img, (400, 200, 600, 400))
        measured = self.add_photo("a.jpg", img, faces=[(400, 200, 600, 400)])
        unmeasured = self.add_photo("b.jpg", flat_image())
        self.run_stage()
        rows = junk.read_frame_quality(self.conn)
        self.assertAlmostEqual(rows[measured].face_sharpness,
                               self.quality(measured)["face_sharpness"], places=6)
        self.assertIsNone(rows[unmeasured].face_sharpness)

    def test_a_rerun_that_finds_nothing_new_leaves_the_number_alone(self):
        """Incrementality: the column belongs to the same marker as the rest of the row."""
        img = flat_image()
        add_noise(img, (400, 200, 600, 400))
        fid = self.add_photo("a.jpg", img, faces=[(400, 200, 600, 400)])
        self.run_stage()
        before = self.quality(fid)["face_sharpness"]
        self.run_stage()
        self.assertAlmostEqual(self.quality(fid)["face_sharpness"], before, places=6)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
