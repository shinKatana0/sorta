"""F179: closed eyes out of geometry — the 948 frames nobody could see.

The question "are the eyes open" was asked of a local VLM and retired: 60% precision over
9% of the frames it was meant to find, for 92 minutes a run. The population stayed — ~948
frames, 15.6% of everything with a face in it — and F178 priced three cheaper ways to see
it against the SAME 249 hand labels. Eyelid geometry won at 62% precision and 48% recall:
five times the recall at slightly better precision, out of arithmetic over contour points
rather than a network's opinion.

What the cases below are about, in the order the risk runs:

* THE COORDINATES ARE RESCALED. `faces.bbox` is in pixels of the full original and the
  contour is fitted on a preview a few hundred pixels wide, so a box used as written lands
  on empty sky. This is the mistake F155 made once and the F178 measurement made again — 39
  of 68 crops off the frame, and the 29 survivors reporting 100% recall instead of 62%. A
  broken crop FLATTERS the result rather than failing, which is why it gets the first and
  the loudest case here;
* the column fills where a real face is and stays NULL everywhere else — no face, only the
  `bbox = '[]'` marker (F125's trap), a crop below the minimum, or no contour model at all;
* several faces -> the LARGEST one decides. A frame where somebody at the back blinked is
  not a portrait with closed eyes;
* it costs NO EXTRA DECODE: a run that measures eyes decodes exactly as often as one that
  does not;
* the slice reads its threshold from `features.eye_openness_max` at QUERY time, so moving
  the number moves the list with nothing re-run (the F137 lesson), and the list is ORDERED
  from the most closed with "show more" walking past the window into the doubtful part;
* the caption states the MEASURED PRECISION and not a count of what was "found".

No model is loaded anywhere below. The real 106-point contour comes out of the `buffalo_l`
set, which `FaceAnalysis` downloads when it is missing — a test suite must not do that — so
the contour is injected, and it is injected as an ELLIPSE whose openness the case chose, so
every expected number is arithmetic a reader can check by hand.
"""
from __future__ import annotations

import unittest
import unittest.mock

import numpy as np

from sorta import junk, sorter, ui
from sorta.junk import eye_openness, largest_face_box
from sorta.junk import FaceBoxes
from tests.test_face_sharpness import (
    SOURCE_SIZE,
    FaceSharpnessCase,
    flat_image,
)
from tests.test_ui_review import ReviewTestBase

# The ring `2d106det` gives per eye: eight contour points. The fake below lays them on an
# ellipse, where `eye_openness` is exactly the ratio of the semi-axes — see `ellipse_ring`.
RING_POINTS = 8
# A face box in ORIGINAL pixels, and the same box in the preview, which is half the size
# (SOURCE_SIZE is exactly twice PREVIEW_EDGE — see tests.test_face_sharpness).
FACE_BOX = (400, 200, 600, 400)
FACE_BOX_IN_PREVIEW = (200, 100, 300, 200)


def ellipse_ring(centre: tuple[float, float], half_width: float,
                 openness: float) -> np.ndarray:
    """`RING_POINTS` points on an ellipse — a ring whose `eye_openness` is `openness`.

    The two points furthest apart are the ends of the long axis (2 * half_width), and the
    ring's spread across that axis is twice the short semi-axis, so the ratio the function
    under test computes is the short semi-axis over the long one — exactly what was asked
    for. That identity is what makes the expected values below arithmetic instead of a
    number copied out of a run.
    """
    angles = np.arange(RING_POINTS) * (2.0 * np.pi / RING_POINTS)
    return np.stack([centre[0] + half_width * np.cos(angles),
                     centre[1] + half_width * openness * np.sin(angles)], axis=1)


def contour(openness: float, second_eye: float | None = None) -> np.ndarray:
    """106 points whose two eye rings have exactly the openness asked for.

    Everything outside the two rings stays at the origin: the feature reads the rings by
    index and nothing else, and a case that filled the other 90 points would be describing
    a face rather than testing a measurement.
    """
    points = np.zeros((106, 2), dtype=np.float64)
    values = (openness, openness if second_eye is None else second_eye)
    for ring, value in zip(junk.EYE_RINGS, values):
        points[list(ring)] = ellipse_ring((30.0, 40.0), 10.0, value)
    return points


class FakeContours:
    """A stand-in for `2d106det`: it records every box it was asked about.

    `openness` is either one number for every face or a function of the box, which is how
    the several-faces case makes the two faces of a frame disagree. `built` counts how
    many times the model was constructed — the feature promises once per run, and only on a
    run that has a face to fit it to.
    """

    def __init__(self, openness=0.05, points=None):
        self._openness = openness
        self._points = points
        self.boxes: list[tuple[int, int, int, int]] = []
        self.built = 0

    def factory(self):
        self.built += 1
        return self

    def __call__(self, img, box):
        self.boxes.append(box)
        if self._points is not None:
            return self._points
        value = (self._openness(box) if callable(self._openness) else self._openness)
        return None if value is None else contour(value)


class ClosedEyesCase(FaceSharpnessCase):
    """A real JPEG, real `faces` rows, the real decode — and an injected contour."""

    def run_with(self, contours: FakeContours, **kwargs):
        return self.run_stage(eye_landmarks_factory=contours.factory, **kwargs)

    def openness_of(self, fid):
        return self.quality(fid)["eye_openness"]


class TestTheGeometryItself(unittest.TestCase):
    """`eye_openness` — the number, with no database and no pixels around it."""

    def test_a_ring_answers_the_ratio_of_its_axes(self):
        for expected in (0.05, 0.18, 0.5, 1.0):
            with self.subTest(openness=expected):
                ring = ellipse_ring((0.0, 0.0), 10.0, expected)
                self.assertAlmostEqual(eye_openness(ring), expected, places=9)

    def test_the_number_does_not_care_how_the_head_is_tilted(self):
        """The corners define the eye's own axis, so a rotation of the ring changes nothing.

        This is why the function finds the two furthest points instead of naming four
        indices "upper lid" and "lower lid": a tilted head is an ordinary photograph.
        """
        ring = ellipse_ring((0.0, 0.0), 10.0, 0.12)
        angle = np.deg2rad(37.0)
        rotation = np.array([[np.cos(angle), -np.sin(angle)],
                             [np.sin(angle), np.cos(angle)]])
        self.assertAlmostEqual(eye_openness(ring @ rotation.T),
                               eye_openness(ring), places=9)

    def test_the_order_of_the_points_does_not_decide(self):
        ring = ellipse_ring((0.0, 0.0), 10.0, 0.12)
        rng = np.random.default_rng(3)
        shuffled = ring[rng.permutation(len(ring))]
        self.assertAlmostEqual(eye_openness(shuffled), eye_openness(ring), places=9)

    def test_a_degenerate_ring_is_not_measured_rather_than_measured_as_zero(self):
        # 0.0 would sort itself to the very top of a list ordered by "most closed first".
        self.assertIsNone(eye_openness(np.zeros((RING_POINTS, 2))))
        self.assertIsNone(eye_openness(np.zeros((2, 2))))
        self.assertIsNone(eye_openness(np.zeros((RING_POINTS, 3))))

    def test_the_more_closed_eye_answers_for_the_face(self):
        """A wink is a frame the person wants to look at, and an average would hide it."""
        img = flat_image((100, 100))
        faces = FaceBoxes(((10.0, 10.0, 90.0, 90.0),), 100.0)
        model = FakeContours(points=contour(0.4, second_eye=0.06))
        self.assertAlmostEqual(junk.face_eye_openness(img, faces, model), 0.06, places=9)

    def test_without_a_model_there_is_no_number(self):
        img = flat_image((100, 100))
        faces = FaceBoxes(((10.0, 10.0, 90.0, 90.0),), 100.0)
        self.assertIsNone(junk.face_eye_openness(img, faces, None))

    def test_an_answer_that_is_not_106_points_is_not_read_as_a_ring(self):
        img = flat_image((100, 100))
        faces = FaceBoxes(((10.0, 10.0, 90.0, 90.0),), 100.0)
        short = FakeContours(points=np.zeros((5, 2)))
        self.assertIsNone(junk.face_eye_openness(img, faces, short))


class TestTheLargestFace(unittest.TestCase):
    """`largest_face_box` — the `largest` rule of the measurement, as arithmetic."""

    def test_the_biggest_box_is_the_one_that_comes_back(self):
        faces = FaceBoxes(((0.0, 0.0, 100.0, 100.0), (200.0, 200.0, 600.0, 600.0)), 1000.0)
        self.assertEqual(largest_face_box(faces, (500, 375)), (100, 100, 300, 300))

    def test_the_order_of_the_rows_does_not_decide(self):
        big = (200.0, 200.0, 600.0, 600.0)
        small = (0.0, 0.0, 100.0, 100.0)
        self.assertEqual(largest_face_box(FaceBoxes((big, small), 1000.0), (500, 375)),
                         largest_face_box(FaceBoxes((small, big), 1000.0), (500, 375)))

    def test_a_frame_with_no_faces_has_no_box(self):
        self.assertIsNone(largest_face_box(junk.NO_FACES, (500, 375)))

    def test_a_box_too_small_after_rescaling_is_not_a_face_to_fit_to(self):
        # 40 px of a 1000 px original is 20 px of a 500 px preview — below the floor the
        # crop of F155 already refuses to measure over.
        faces = FaceBoxes(((0.0, 0.0, 40.0, 40.0),), 1000.0)
        self.assertIsNone(largest_face_box(faces, (500, 375)))


class TestTheContourIsFittedToTheRescaledBox(ClosedEyesCase):
    """Brief test 1 — the main one: the box handed to the model is in PREVIEW pixels."""

    def test_a_preview_half_the_size_halves_the_coordinates(self):
        fid = self.add_photo("face.jpg", flat_image(), faces=[FACE_BOX])
        model = FakeContours()
        self.run_with(model)
        self.assertEqual(model.boxes, [FACE_BOX_IN_PREVIEW])
        # and that is exactly half of what the `faces` row says, on every coordinate
        self.assertEqual(model.boxes[0], tuple(v // 2 for v in FACE_BOX))
        self.assertIsNotNone(self.openness_of(fid))

    def test_the_box_never_leaves_the_array_it_is_measured_on(self):
        """The failure this feature is built around: it does not raise, it lies.

        A box used as written runs off a preview half the size, and what comes back is a
        contour fitted to whatever the model finds in the corner — a number that looks like
        a measurement of somebody's eye.
        """
        self.add_photo("face.jpg", flat_image(), faces=[FACE_BOX])
        model = FakeContours()
        self.run_with(model)
        preview_width, preview_height = self.preview_of(
            self.conn.execute("SELECT id FROM files").fetchone()["id"]).size
        left, top, right, bottom = model.boxes[0]
        self.assertLessEqual(right, preview_width)
        self.assertLessEqual(bottom, preview_height)
        self.assertGreaterEqual(left, 0)
        self.assertGreaterEqual(top, 0)

    def test_the_stored_number_is_the_openness_of_the_contour(self):
        fid = self.add_photo("face.jpg", flat_image(), faces=[FACE_BOX])
        self.run_with(FakeContours(openness=0.07))
        self.assertAlmostEqual(self.openness_of(fid), 0.07, places=9)


class TestPopulation(ClosedEyesCase):
    """Brief tests 2 and 3: who gets a number, and who gets NULL."""

    def test_a_frame_with_a_real_face_gets_the_number(self):
        fid = self.add_photo("face.jpg", flat_image(), faces=[FACE_BOX])
        self.run_with(FakeContours())
        self.assertIsNotNone(self.openness_of(fid))

    def test_a_frame_with_no_faces_row_gets_null(self):
        withface = self.add_photo("face.jpg", flat_image(), faces=[FACE_BOX])
        landscape = self.add_photo("landscape.jpg", flat_image())
        self.run_with(FakeContours())
        self.assertIsNotNone(self.quality(landscape)["sharpness"])   # it WAS measured
        self.assertIsNone(self.openness_of(landscape))               # just no face on it
        self.assertIsNotNone(self.openness_of(withface))

    def test_the_no_face_marker_is_not_a_face(self):
        """`bbox = '[]'` means "processed, nothing found" — F125's trap, again."""
        marked = self.add_photo("nobody.jpg", flat_image())
        self.add_no_face_marker(marked)
        self.add_photo("face.jpg", flat_image(), faces=[FACE_BOX])   # so the model is built
        model = FakeContours()
        self.run_with(model)
        self.assertIsNone(self.openness_of(marked))
        self.assertEqual(model.boxes, [FACE_BOX_IN_PREVIEW])   # it was never asked about

    def test_a_face_below_the_minimum_crop_size_is_null_not_a_small_number(self):
        fid = self.add_photo("tiny.jpg", flat_image(), faces=[(100, 100, 140, 140)])
        self.run_with(FakeContours())
        self.assertIsNone(self.openness_of(fid))

    def test_a_contour_the_model_could_not_fit_is_null(self):
        fid = self.add_photo("face.jpg", flat_image(), faces=[FACE_BOX])
        self.run_with(FakeContours(openness=lambda _box: None))
        self.assertIsNotNone(self.quality(fid)["sharpness"])   # the rest of the row stands
        self.assertIsNone(self.openness_of(fid))

    def test_a_model_that_cannot_be_built_costs_this_column_and_nothing_else(self):
        """The graceful fallback: no [faces] extra, no weights, a broken runtime."""
        fid = self.add_photo("face.jpg", flat_image(), faces=[FACE_BOX])

        def explode():
            raise RuntimeError("no onnxruntime here")

        self.run_stage(eye_landmarks_factory=explode)
        self.assertIsNotNone(self.quality(fid)["sharpness"])
        self.assertIsNotNone(self.quality(fid)["face_sharpness"])
        self.assertIsNone(self.openness_of(fid))

    def test_the_number_survives_the_round_trip_out_of_the_table(self):
        measured = self.add_photo("a.jpg", flat_image(), faces=[FACE_BOX])
        unmeasured = self.add_photo("b.jpg", flat_image())
        self.run_with(FakeContours(openness=0.11))
        rows = junk.read_frame_quality(self.conn)
        self.assertAlmostEqual(rows[measured].eye_openness, 0.11, places=9)
        self.assertIsNone(rows[unmeasured].eye_openness)


class TestSeveralFaces(ClosedEyesCase):
    """Brief test 4: the LARGEST face decides, and only it is asked about."""

    SMALL = (0, 0, 200, 200)          # 200 x 200 in the original
    BIG = (400, 200, 800, 600)        # 400 x 400 — twice the side, four times the area

    def test_the_largest_face_answers_for_the_frame(self):
        fid = self.add_photo("two.jpg", flat_image(SOURCE_SIZE),
                             faces=[self.SMALL, self.BIG])
        # the big face has its eyes open, the small one in the background has them shut
        model = FakeContours(
            openness=lambda box: 0.4 if (box[2] - box[0]) > 150 else 0.03)
        self.run_with(model)
        self.assertAlmostEqual(self.openness_of(fid), 0.4, places=9)

    def test_a_blink_in_the_background_does_not_make_it_a_closed_eyes_frame(self):
        """The whole point of the `largest` rule, stated as the frame it protects."""
        fid = self.add_photo("two.jpg", flat_image(SOURCE_SIZE),
                             faces=[self.SMALL, self.BIG])
        model = FakeContours(
            openness=lambda box: 0.4 if (box[2] - box[0]) > 150 else 0.03)
        self.run_with(model)
        self.assertGreater(self.openness_of(fid), self.cfg.features.eye_openness_max)

    def test_only_the_largest_face_is_ever_asked_about(self):
        self.add_photo("two.jpg", flat_image(SOURCE_SIZE), faces=[self.SMALL, self.BIG])
        model = FakeContours()
        self.run_with(model)
        self.assertEqual(len(model.boxes), 1)
        self.assertEqual(model.boxes[0], tuple(v // 2 for v in self.BIG))

    def test_the_order_of_the_rows_does_not_decide(self):
        first = self.add_photo("a.jpg", flat_image(SOURCE_SIZE),
                               faces=[self.SMALL, self.BIG])
        second = self.add_photo("b.jpg", flat_image(SOURCE_SIZE),
                                faces=[self.BIG, self.SMALL])
        self.run_with(FakeContours(
            openness=lambda box: 0.4 if (box[2] - box[0]) > 150 else 0.03))
        self.assertEqual(self.openness_of(first), self.openness_of(second))


class TestNoSecondPass(ClosedEyesCase):
    """Brief test 5: the eye number rides in the decode the stage already paid for."""

    def _decodes(self, *, with_contours: bool) -> list[str]:
        calls: list[str] = []
        real = junk.imaging.decode_rgb_preview

        def counting(path, *args, **kwargs):
            calls.append(str(path))
            return real(path, *args, **kwargs)

        for i in range(3):
            self.add_photo(f"f{i}.jpg", flat_image(), faces=[FACE_BOX])
        with unittest.mock.patch.object(junk.imaging, "decode_rgb_preview", counting):
            if with_contours:
                self.run_with(FakeContours())
            else:
                self.run_stage()
        return calls

    def test_measuring_the_eyes_adds_no_decode(self):
        with_eyes = self._decodes(with_contours=True)
        self.setUp()   # a second, independent index — the same three frames
        without = self._decodes(with_contours=False)
        self.assertEqual(len(with_eyes), len(without))
        self.assertEqual(len(with_eyes), 3)   # one decode per frame, eyes or no eyes

    def test_every_frame_is_decoded_exactly_once(self):
        calls = self._decodes(with_contours=True)
        self.assertEqual(sorted(calls), sorted(set(calls)))

    def test_the_contour_model_is_built_once_for_the_whole_run(self):
        for i in range(3):
            self.add_photo(f"f{i}.jpg", flat_image(), faces=[FACE_BOX])
        model = FakeContours()
        self.run_with(model)
        self.assertEqual(model.built, 1)
        self.assertEqual(len(model.boxes), 3)

    def test_a_collection_with_no_faces_never_builds_the_model(self):
        """Two thirds of an archive have no face, and a first run has no `faces` rows."""
        self.add_photo("landscape.jpg", flat_image())
        model = FakeContours()
        self.run_with(model)
        self.assertEqual(model.built, 0)


class TestTheOtherNumbersDoNotMove(ClosedEyesCase):
    """The eye number is ADDED to the row, not swapped into it."""

    def test_the_two_laplacians_are_what_they_were(self):
        from tests.test_face_sharpness import add_noise, laplacian_variance

        img = flat_image()
        add_noise(img, FACE_BOX)
        fid = self.add_photo("a.jpg", img, faces=[FACE_BOX])
        self.run_with(FakeContours())
        preview = self.preview_of(fid)
        self.assertAlmostEqual(self.quality(fid)["sharpness"],
                               laplacian_variance(preview), places=4)
        self.assertAlmostEqual(self.quality(fid)["face_sharpness"],
                               laplacian_variance(preview.crop(FACE_BOX_IN_PREVIEW)),
                               places=4)

    def test_a_rerun_that_finds_nothing_new_leaves_the_number_alone(self):
        fid = self.add_photo("a.jpg", flat_image(), faces=[FACE_BOX])
        self.run_with(FakeContours(openness=0.09))
        before = self.openness_of(fid)
        second = FakeContours(openness=0.09)
        self.run_with(second)
        self.assertEqual(self.openness_of(fid), before)
        self.assertEqual(second.boxes, [])   # incrementality: nothing was measured again


class ClosedEyesSliceCase(ReviewTestBase):
    """The slice over a ready `frame_quality`: no stage, no pixels — just the rule."""

    def payload(self, slice_="eyes", *, offset=0, limit=50, beyond=False,
                eye_max=None):
        return ui._review_payload(
            self.cfg.database, slice_, offset, limit, beyond=beyond,
            blur_max=self.cfg.features.blur_review_max,
            # Brief test 6: the threshold comes off the CONFIG. A literal here would test
            # the number this file happened to be written on rather than the setting.
            eye_max=self.cfg.features.eye_openness_max if eye_max is None else eye_max,
            max_distance=self.cfg.index.phash_max_distance)

    def ids(self, data):
        return [item["file_id"] for item in data["items"]]

    def closed(self, name, openness):
        return self.add_reviewable(name, sharpness=500.0, eye_openness=openness)


class TestTheSliceReadsTheConfig(ClosedEyesSliceCase):
    """Brief test 6: `features.eye_openness_max` decides, and it decides WHEN READ."""

    def test_the_window_is_the_configured_threshold(self):
        limit = self.cfg.features.eye_openness_max
        inside = self.closed("closed.jpg", limit / 2)
        self.closed("open.jpg", limit * 2)
        self.assertEqual(self.ids(self.payload()), [inside])

    def test_a_threshold_the_user_moved_moves_the_slice_with_nothing_re_run(self):
        """The F137 lesson: a verdict frozen into the row would need a fresh pass."""
        limit = self.cfg.features.eye_openness_max
        inside = self.closed("closed.jpg", limit / 2)
        outside = self.closed("open.jpg", limit * 2)
        self.assertEqual(self.ids(self.payload()), [inside])
        self.assertEqual(self.ids(self.payload(eye_max=limit * 3)), [inside, outside])

    def test_a_frame_that_was_never_measured_is_not_an_answer(self):
        self.closed("unmeasured.jpg", None)
        self.assertEqual(self.payload()["total"], 0)

    def test_the_counter_and_the_list_are_the_same_number(self):
        limit = self.cfg.features.eye_openness_max
        self.closed("a.jpg", limit / 2)
        self.closed("b.jpg", limit / 3)
        self.closed("c.jpg", limit * 2)
        data = self.payload()
        counts = {row["slice"]: row["count"] for row in data["counts"]}
        self.assertEqual(counts["eyes"], 2)
        self.assertEqual(data["total"], 2)
        self.assertEqual(len(self.ids(data)), 2)

    def test_the_album_gathers_the_very_frames_the_list_shows(self):
        """One rule, two spellings would drift — the album reads `quality_slice_where`."""
        limit = self.cfg.features.eye_openness_max
        listed = [self.closed("a.jpg", limit / 2), self.closed("b.jpg", limit / 3)]
        self.closed("c.jpg", limit * 2)
        where, params = sorter.quality_slice_where("eyes_closed", None, limit)
        rows = self.conn.execute(
            f"SELECT f.id {sorter.QUALITY_FROM} WHERE {where}", params).fetchall()
        self.assertEqual(sorted(int(r["id"]) for r in rows), sorted(listed))


class TestTheSliceIsARanking(ClosedEyesSliceCase):
    """Brief test 7's other half: ordered from the confident, continued on demand."""

    def test_the_most_closed_come_first(self):
        limit = self.cfg.features.eye_openness_max
        middle = self.closed("middle.jpg", limit * 0.5)
        tightest = self.closed("tight.jpg", limit * 0.1)
        widest = self.closed("wide.jpg", limit * 0.9)
        self.assertEqual(self.ids(self.payload()), [tightest, middle, widest])

    def test_show_more_continues_past_the_window_into_the_doubtful_part(self):
        limit = self.cfg.features.eye_openness_max
        inside = [self.closed(f"in{i}.jpg", limit * (0.1 + 0.1 * i)) for i in range(3)]
        outside = [self.closed(f"out{i}.jpg", limit * (1.1 + 0.1 * i)) for i in range(2)]
        self.assertEqual(self.ids(self.payload()), inside)
        self.assertEqual(self.ids(self.payload(beyond=True)), inside + outside)

    def test_the_seam_neither_repeats_a_frame_nor_skips_one(self):
        limit = self.cfg.features.eye_openness_max
        everything = [self.closed(f"a{i}.jpg", limit * (0.1 + 0.05 * i))
                      for i in range(6)]
        first = self.ids(self.payload(limit=3))
        second = self.ids(self.payload(offset=3, limit=3))
        self.assertEqual(first + second, everything)
        self.assertEqual(len(set(first + second)), len(everything))

    def test_the_window_total_is_the_current_slice(self):
        """What the client's "show more" compares against before stepping outside."""
        limit = self.cfg.features.eye_openness_max
        self.closed("in.jpg", limit / 2)
        self.closed("out.jpg", limit * 2)
        self.assertEqual(self.payload()["window_total"], 1)
        self.assertEqual(self.payload(beyond=True)["window_total"], 1)


class TestTheCaptionStatesTheMeasurement(unittest.TestCase):
    """Brief test 7: the caption names the precision, not a count of what was "found"."""

    LANGS = ("ru", "en", "ja")

    def caption(self, lang):
        return ui._UI_STRINGS["review_hint_eyes"][lang]

    def test_every_language_states_the_measured_precision(self):
        for lang in self.LANGS:
            with self.subTest(lang=lang):
                self.assertIn("62", self.caption(lang))

    def test_every_language_says_what_the_rest_of_the_list_is(self):
        """62% right means one frame in three is not — a reader has to be told that."""
        for lang in self.LANGS:
            with self.subTest(lang=lang):
                caption = self.caption(lang)
                self.assertTrue(any(word in caption for word in
                                    ("каждый третий", "one frame in three", "3 コマに 1")),
                                caption)

    def test_the_caption_carries_the_threshold_it_is_shown_with(self):
        for lang in self.LANGS:
            with self.subTest(lang=lang):
                self.assertIn("{max}", self.caption(lang))

    def test_it_does_not_advertise_a_count_of_findings(self):
        # "Found 730 frames" reads as a verdict about 730 photographs; this list is not one.
        for lang in self.LANGS:
            with self.subTest(lang=lang):
                self.assertNotIn("{n}", self.caption(lang))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
