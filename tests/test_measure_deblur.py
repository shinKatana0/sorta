"""F183, phase 0: the measurement that decides whether a 1:1 model is the right tool.

Everything this feature will or will not become is read off this script's output, so what
is checked here is the script's honesty:

* THE QUESTION IS FIDELITY. Not "better", not "sharper" — the substituted question is what
  gave the wrong answer twice in one day, in both directions, and the verdict block has to
  ask the right one out loud;
* THE POPULATION IS SPLIT BY DEGRADATION. Motion smear, missed focus, general softness are
  three questions; a model trained on camera shake can repair one of them and do nothing
  for the other two, and one average would report that as "it sort of works";
* ONLY FRAMES ABOVE THE CEILING. Small frames are decided (F169: the x4 model beats bicubic
  62% against 10%) and nothing here may pull them in;
* THE CANDIDATE IS CHECKED BEFORE IT IS PRICED. A model that enlarges is answering F169's
  question, and a model that returns its input untouched is a null result dressed as "no
  harm" — F149's first probe flattered itself in exactly that way;
* THE PREMISE IS CHECKED FIRST: the cost must grow with the pixels rather than with the
  square of a multiplier, and a run where it does not says to stop;
* the pairs for the eyes are BLIND, and blind across ARMS too — two instruments on the same
  frames laid out in order would tell a person which model they are looking at by the third
  sheet;
* the counts come off the product's OWN slice rules, so the measurement cannot describe a
  population no button can show;
* nothing printed identifies a frame, and the originals are only ever read.

No model, no GPU, no collection: the candidates are stubs of the shape `restore.UpscaleFn`
has, the frames are Pillow images, and every expected number is arithmetic.
"""
from __future__ import annotations

import importlib.util
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from sorta import db
from sorta.config import FeaturesConfig

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "measure_deblur.py"


def _load_script():
    """Import scripts/measure_deblur.py — it is a script, not a package module."""
    spec = importlib.util.spec_from_file_location("measure_deblur", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


md = _load_script()

FEATURES = FeaturesConfig()


def noise(size=(600, 450), seed: int = 1) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8),
                           "RGB")


def smeared(image: Image.Image, length: int = 9) -> Image.Image:
    """The same picture with a HORIZONTAL smear — a hand that moved, in one line of numpy."""
    a = np.asarray(image.convert("L"), dtype=np.float32)
    shifts = range(-(length // 2), length // 2 + 1)
    blurred = np.mean([np.roll(a, s, axis=1) for s in shifts], axis=0)
    return Image.fromarray(blurred.astype(np.uint8)).convert("RGB")


def identity(image: Image.Image) -> Image.Image:
    return image.copy()


def softening(image: Image.Image) -> Image.Image:
    """A stand-in for a 1:1 restoration model: same size in, same size out, pixels moved."""
    return image.filter(ImageFilter.GaussianBlur(1.5))


def doubling(image: Image.Image) -> Image.Image:
    """A stand-in for the x4 arm (x2 keeps the fixtures small)."""
    return image.resize((image.width * 2, image.height * 2))


def make_jpeg(path: Path, size=(2400, 1800), color=(90, 120, 160)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path, "JPEG")
    return path


def run(file_id: int, arm: str, degradation: str, max_edge: int | None, *,
        seconds: float = 1.0, source=(4000, 3000), input_size=(4000, 3000),
        output=(4000, 3000), weight: int = 1024 * 1024, vram: float | None = None,
        error: str | None = None):
    return md.FrameRun(file_id=file_id, arm=arm, degradation=degradation,
                       max_edge=max_edge, source_size=source, input_size=input_size,
                       output_size=output, weight_bytes=weight, seconds=seconds,
                       peak_vram_mb=vram, error=error)


def cost(arm: str, megapixels: float, seconds: float, *, vram: float | None = None,
         error: str | None = None):
    return md.CostRow(arm=arm, megapixels=megapixels, input_size=(1000, 750),
                      output_size=(1000, 750), seconds=seconds, peak_vram_mb=vram,
                      error=error)


class TestWhichKindOfSoftTheFrameIs(unittest.TestCase):
    """Three degradations, because they are three accidents repaired by three training
    sets — and a deblurring model is trained mostly on the first of them."""

    def test_direction_is_what_separates_a_smear_from_everything_else(self):
        # A picture detailed in every direction has no direction; one made of vertical
        # lines has nothing but. The number is the same scale-free ratio in both cases.
        self.assertLess(md.anisotropy(noise()), 0.05)
        stripes = np.zeros((200, 200), dtype=np.uint8)
        stripes[:, ::4] = 255
        self.assertAlmostEqual(md.anisotropy(Image.fromarray(stripes)), 1.0, places=2)

    def test_a_flat_frame_has_no_direction_and_is_not_called_isotropic(self):
        """None is "not measured". 0.0 would read as "equally detailed everywhere", which
        is a statement about a frame that has no detail at all."""
        self.assertIsNone(md.anisotropy(Image.new("RGB", (100, 100), (20, 20, 20))))
        self.assertIsNone(md.anisotropy(Image.new("RGB", (2, 2), (10, 200, 10))))

    def test_a_hand_that_moved_lands_in_motion_and_a_missed_focus_does_not(self):
        motion = md.degradation_of(smeared(noise()), blur_max=90.0, sharpness_edge=512)
        self.assertEqual(motion.kind, md.MOTION)
        self.assertGreaterEqual(motion.anisotropy or 0.0, md.MOTION_ANISOTROPY)

        defocus = md.degradation_of(noise().filter(ImageFilter.GaussianBlur(4)),
                                    blur_max=90.0, sharpness_edge=512)
        self.assertEqual(defocus.kind, md.DEFOCUS)
        self.assertLess(defocus.anisotropy or 1.0, md.MOTION_ANISOTROPY)

    def test_deep_in_the_window_is_a_missed_focus_and_near_its_top_is_merely_soft(self):
        """The line between the two is a DEGREE on the scale the blurred list is already
        ranked by, so it is a fraction of `features.blur_review_max` — a person who moves
        the window moves this with it."""
        frame = noise().filter(ImageFilter.GaussianBlur(4))
        sharpness = md.degradation_of(frame, blur_max=90.0, sharpness_edge=512).sharpness
        assert sharpness is not None
        # The same frame, read against a window it sits near the top of instead of deep in.
        wide = md.degradation_of(frame, blur_max=sharpness / md.DEFOCUS_SHARE * 0.9,
                                 sharpness_edge=512)
        self.assertEqual(wide.kind, md.SOFT)

    def test_a_frame_that_cannot_be_measured_is_unknown_and_not_quietly_soft(self):
        blank = md.degradation_of(Image.new("RGB", (2, 2)), blur_max=90.0,
                                  sharpness_edge=512)
        self.assertEqual(blank.kind, md.UNKNOWN)

    def test_every_degradation_has_a_label_a_person_can_read(self):
        self.assertEqual(set(md.DEGRADATION_LABEL), set(md.DEGRADATIONS))
        for kind in md.DEGRADATIONS:
            self.assertTrue(md.DEGRADATION_LABEL[kind].strip())

    def test_the_two_numbers_are_taken_at_the_scales_they_belong_to(self):
        """Direction in the frame's OWN pixels (a three-pixel smear does not survive a
        downscale); sharpness on the preview the index measures, because that is the only
        scale `blur_review_max` may be compared against."""
        self.assertEqual(md.fit_edge(Image.new("RGB", (2048, 1536)), 512).size, (512, 384))
        # ...and never the other way round: a small frame is not enlarged to reach a ceiling.
        self.assertEqual(md.fit_edge(Image.new("RGB", (300, 200)), 512).size, (300, 200))
        self.assertEqual(md.fit_edge(Image.new("RGB", (300, 200)), None).size, (300, 200))


class TestTheCandidateIsCheckedBeforeItIsPriced(unittest.TestCase):
    """The F149 lesson made mechanical: a broken instrument flatters the result."""

    def test_a_model_that_returns_the_same_size_and_moves_the_pixels_is_measured(self):
        probe = md.probe_one_to_one(softening)
        self.assertTrue(probe.usable)
        self.assertEqual(probe.reason, "")
        self.assertAlmostEqual(probe.scale, 1.0)

    def test_a_model_that_enlarges_is_answering_the_other_feature_s_question(self):
        probe = md.probe_one_to_one(doubling)
        self.assertFalse(probe.usable)
        self.assertAlmostEqual(probe.scale, 2.0)
        self.assertIn("один к одному", probe.reason)

    def test_a_model_that_hands_the_frame_back_untouched_is_not_a_result(self):
        probe = md.probe_one_to_one(identity)
        self.assertFalse(probe.usable)
        self.assertEqual(probe.changed, 0.0)
        self.assertIn("без изменений", probe.reason)

    def test_a_candidate_that_will_not_run_is_a_reason_and_not_a_traceback(self):
        def broken(_image: Image.Image) -> Image.Image:
            raise RuntimeError("CUDA out of memory")

        probe = md.probe_one_to_one(broken)
        self.assertFalse(probe.usable)
        self.assertIn("CUDA out of memory", probe.reason)


class TestThePriceAndThePremise(unittest.TestCase):
    """Brief item 1: if the cost does not grow with the pixels, nothing else matters."""

    def test_the_row_prices_one_frame_size_through_one_arm(self):
        arm = md.Arm(name="cand", max_edge=None, process=softening)
        row = md.cost_row(arm, 0.1)
        self.assertIsNone(row.error)
        self.assertEqual(row.input_size, row.output_size)
        self.assertGreater(row.seconds, 0.0)
        self.assertGreater(row.ms_per_megapixel, 0.0)

    def test_the_x4_arm_is_priced_through_its_ceiling_the_way_it_really_runs(self):
        arm = md.Arm(name=md.ARM_BASELINE, max_edge=256, process=doubling)
        row = md.cost_row(arm, 1.0)
        self.assertEqual(max(row.input_size), 256)
        self.assertEqual(max(row.output_size), 512)

    def test_a_size_that_does_not_fit_is_a_row_and_the_premise_is_declared_wrong(self):
        def dies(_image: Image.Image) -> Image.Image:
            raise RuntimeError("CUDA out of memory")

        row = md.cost_row(md.Arm(name="cand", max_edge=None, process=dies), 0.1)
        self.assertIn("CUDA out of memory", row.error or "")

        linear, message = md.growth_verdict([cost("cand", 1, 0.1), row])
        self.assertFalse(linear)
        self.assertIn("посылка неверна", message)

    def test_a_cost_that_follows_the_pixels_is_called_linear(self):
        rows = [cost("cand", 1, 0.10, vram=500), cost("cand", 4, 0.40, vram=2000),
                cost("cand", 12, 1.32, vram=6000)]
        linear, message = md.growth_verdict(rows)
        self.assertTrue(linear)
        self.assertIn("линейный", message)

    def test_a_cost_that_grows_with_the_square_stops_the_run(self):
        rows = [cost("x4", 1, 0.10), cost("x4", 4, 1.60), cost("x4", 12, 14.4)]
        linear, message = md.growth_verdict(rows)
        self.assertFalse(linear)
        self.assertIn("время растёт", message)
        self.assertIn("посылка неверна", message)

    def test_memory_that_outgrows_the_pixels_stops_it_just_as_time_does(self):
        rows = [cost("cand", 1, 0.10, vram=500), cost("cand", 12, 1.20, vram=24000)]
        linear, message = md.growth_verdict(rows)
        self.assertFalse(linear)
        self.assertIn("память растёт", message)

    def test_one_point_is_not_a_growth(self):
        linear, message = md.growth_verdict([cost("cand", 1, 0.1)])
        self.assertFalse(linear)
        self.assertIn("не на чем", message)

    def test_the_synthetic_frame_is_the_size_it_says_it_is(self):
        frame = md.synthetic_frame(1)
        self.assertAlmostEqual(frame.width * frame.height / 1_000_000, 1.0, places=1)
        self.assertAlmostEqual(frame.width / frame.height, 4 / 3, places=1)

    def test_the_price_table_prints_the_sizes_the_verdict_and_no_frame(self):
        table = md.format_cost_table([cost("cand", 1, 0.10, vram=500),
                                      cost("cand", 12, 1.20, vram=6000),
                                      cost("x4", 12, 0.0, error="RuntimeError: no memory")])
        self.assertIn("cand", table)
        self.assertIn("500 МБ", table)
        self.assertIn("линейный", table)
        self.assertIn("RuntimeError: no memory", table)
        for leak in (".jpg", "/", "\\"):
            self.assertNotIn(leak, table)


class TestOneFrameThroughOneArm(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_the_1_to_1_arm_is_shown_the_frame_as_it_lies(self):
        src = make_jpeg(self.root / "shot.jpg", size=(2400, 1800))
        arm = md.Arm(name="cand", max_edge=None, process=softening)

        row, processed = md.run_frame(arm, src, 7, md.SOFT)

        self.assertIsNone(row.error)
        self.assertEqual((row.file_id, row.arm, row.degradation), (7, "cand", md.SOFT))
        self.assertEqual(row.source_size, (2400, 1800))
        self.assertEqual(row.input_size, (2400, 1800))
        self.assertEqual(row.output_size, (2400, 1800))
        self.assertEqual(md.truth_kept(row.source_edge, row.max_edge), 1.0)
        assert processed is not None

    def test_the_baseline_arm_is_shown_a_quarter_of_the_side_the_way_it_ships(self):
        src = make_jpeg(self.root / "shot.jpg", size=(2400, 1800))
        arm = md.Arm(name=md.ARM_BASELINE, max_edge=600, process=doubling)

        row, _ = md.run_frame(arm, src, 1, md.SOFT)

        self.assertEqual(row.input_size, (600, 450))
        self.assertAlmostEqual(md.truth_kept(row.source_edge, row.max_edge), 0.0625)

    def test_a_model_that_runs_out_of_memory_is_a_row_and_not_a_crash(self):
        src = make_jpeg(self.root / "shot.jpg", size=(1200, 900))

        def dies(_image: Image.Image) -> Image.Image:
            raise RuntimeError("CUDA out of memory")

        row, processed = md.run_frame(md.Arm("cand", None, dies), src, 3, md.MOTION)

        self.assertIsNone(processed)
        self.assertIn("CUDA out of memory", row.error or "")
        self.assertEqual(row.output_size, (0, 0))

    def test_a_frame_that_will_not_read_is_a_row_too(self):
        broken = self.root / "broken.jpg"
        broken.write_bytes(b"not an image")
        row, processed = md.run_frame(md.Arm("cand", None, softening), broken, 4, md.SOFT)
        self.assertIsNone(processed)
        self.assertTrue(row.error)

    def test_the_original_is_only_ever_read(self):
        """This script measures the action; it does not perform it. Nothing is written
        beside anybody's photograph — not a copy, not a re-encode."""
        src = make_jpeg(self.root / "shot.jpg", size=(1200, 900))
        before, mtime = src.read_bytes(), src.stat().st_mtime

        md.run_frame(md.Arm("cand", None, softening), src, 1, md.SOFT)
        md.original_run(src, 1, md.SOFT)

        self.assertEqual(src.read_bytes(), before)
        self.assertEqual(src.stat().st_mtime, mtime)
        self.assertEqual([p.name for p in self.root.iterdir()], ["shot.jpg"])

    def test_the_original_row_is_the_file_as_it_lies(self):
        src = make_jpeg(self.root / "shot.jpg", size=(3000, 2000))
        row = md.original_run(src, 9, md.MOTION)
        self.assertEqual(row.arm, md.ARM_ORIGINAL)
        self.assertEqual(row.source_size, (3000, 2000))
        self.assertEqual(row.weight_bytes, src.stat().st_size)
        self.assertEqual(row.seconds, 0.0)

    def test_every_frame_meets_every_arm_and_carries_its_own_degradation(self):
        smear = self.root / "smear.jpg"
        smeared(noise((1200, 900))).save(smear, "JPEG", quality=95)
        arms = [md.Arm(md.ARM_ORIGINAL, None), md.Arm("cand", None, softening),
                md.Arm(md.ARM_BASELINE, 300, doubling)]

        runs, pairs = md.measure(arms, [(5, str(smear))], blur_max=90.0, sharpness_edge=512)

        self.assertEqual([r.arm for r in runs], [md.ARM_ORIGINAL, "cand", md.ARM_BASELINE])
        self.assertEqual({r.degradation for r in runs}, {md.MOTION})
        # One pair per instrument, and never one for the original against itself.
        self.assertEqual([p[0].arm for p in pairs], ["cand", md.ARM_BASELINE])


class TestTheTables(unittest.TestCase):
    ARMS = [md.Arm(md.ARM_ORIGINAL, None), md.Arm("cand", None, softening),
            md.Arm(md.ARM_BASELINE, 1024, doubling)]

    def rows(self, kind: str = md.MOTION, count: int = 5) -> list:
        out = []
        for n in range(count):
            out += [
                run(n, md.ARM_ORIGINAL, kind, None, seconds=0.0, weight=3 * 1024 * 1024),
                run(n, "cand", kind, None, seconds=1.0, vram=1980.0),
                run(n, md.ARM_BASELINE, kind, 1024, seconds=3.4, input_size=(1024, 768),
                    output=(4096, 3072), vram=7100.0),
            ]
        return out

    def test_the_original_is_the_first_row_of_every_table(self):
        table = md.format_degradation_table(md.MOTION, self.rows(), self.ARMS)
        body = [line for line in table.splitlines() if md.ARM_ORIGINAL in line
                or "cand" in line]
        self.assertIn(md.ARM_ORIGINAL, body[0])

    def test_the_table_prints_time_weight_memory_and_what_was_kept(self):
        table = md.format_degradation_table(md.MOTION, self.rows(), self.ARMS)
        self.assertIn("1.00 с", table)
        self.assertIn("1980 МБ", table)
        self.assertIn("3.00 МБ", table)     # the original's own weight
        self.assertIn("100%", table)        # the 1:1 arm gives up nothing
        self.assertIn("7%", table)          # 1024 of a 4000 px frame, areally

    def test_each_degradation_gets_its_own_table_and_never_one_average(self):
        runs = self.rows(md.MOTION) + self.rows(md.DEFOCUS)
        text = md.format_degradation_tables(runs, self.ARMS)
        for kind in (md.MOTION, md.DEFOCUS, md.SOFT):
            self.assertIn(md.DEGRADATION_LABEL[kind], text)
        # A type nothing landed in says so instead of being quietly merged into another.
        self.assertIn("нет кадров этого типа", text)

    def test_the_unknown_bucket_appears_only_when_something_lands_in_it(self):
        self.assertNotIn(md.DEGRADATION_LABEL[md.UNKNOWN],
                         md.format_degradation_tables(self.rows(), self.ARMS))
        self.assertIn(md.DEGRADATION_LABEL[md.UNKNOWN],
                      md.format_degradation_tables(self.rows(md.UNKNOWN), self.ARMS))

    def test_a_handful_of_frames_is_called_an_anecdote_rather_than_an_answer(self):
        thin = md.format_degradation_table(md.MOTION, self.rows(count=2), self.ARMS)
        self.assertIn("анекдот", thin)
        self.assertNotIn("анекдот",
                         md.format_degradation_table(md.MOTION, self.rows(), self.ARMS))

    def test_a_failed_run_is_counted_and_never_averaged_into_a_time(self):
        summary = md.summarize([run(1, "cand", md.SOFT, None, seconds=2.0),
                                run(2, "cand", md.SOFT, None, error="RuntimeError: nope")])
        self.assertEqual((summary.frames, summary.failed), (1, 1))
        self.assertAlmostEqual(summary.seconds, 2.0)

    def test_an_arm_that_failed_on_every_frame_says_why(self):
        table = md.format_degradation_table(
            md.SOFT, [run(1, "cand", md.SOFT, None, error="RuntimeError: no memory")],
            self.ARMS)
        self.assertIn("RuntimeError: no memory", table)

    def test_nothing_in_the_tables_identifies_a_frame(self):
        table = md.format_degradation_tables(self.rows(), self.ARMS)
        for leak in (".jpg", "shot", "/", "\\"):
            with self.subTest(leak=leak):
                self.assertNotIn(leak, table)


class TestHowManyFramesThisTouches(unittest.TestCase):
    """Brief item 5, off the product's OWN slice rules — a measurement that described a
    population no button can show would be describing nothing."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "index.db"
        self.conn = db.connect(self.path)
        self.addCleanup(self.conn.close)
        self.next_id = 0

    def add(self, *, width: int, height: int, sharpness: float | None,
            verdict: str = "photo", exists: bool = True) -> int:
        self.next_id += 1
        file_id = self.next_id
        path = Path(self.tmp.name) / f"f{file_id}.jpg"
        if exists:
            make_jpeg(path, size=(8, 6))
        self.conn.execute(
            "INSERT INTO files (id, path, size, mtime, ext, media_type, width, height,"
            " indexed_at) VALUES (?, ?, 1, 1.0, 'jpg', 'photo', ?, ?, '2026-08-04')",
            (file_id, str(path), width, height))
        self.conn.execute(
            "INSERT INTO media_class (file_id, verdict, source, updated_at)"
            " VALUES (?, ?, 'clip', '2026-08-04')", (file_id, verdict))
        if sharpness is not None:
            self.conn.execute(
                # `source` is NOT NULL: it carries the fingerprint of the prompts an
                # answer came from, so a row without one could never be invalidated when
                # the wording changes. A hand-built row has to satisfy the whole schema,
                # not the part this test cares about.
                "INSERT INTO frame_quality (file_id, sharpness, source, updated_at)"
                " VALUES (?, ?, 'test', '2026-08-04')", (file_id, sharpness))
        self.conn.commit()
        return file_id

    def test_the_count_separates_the_slice_from_the_part_this_feature_is_about(self):
        self.add(width=4000, height=3000, sharpness=20.0)    # blurred and big — the case
        self.add(width=4000, height=3000, sharpness=30.0)    # ...and another
        self.add(width=800, height=600, sharpness=20.0)      # blurred but small — F169's
        self.add(width=4000, height=3000, sharpness=400.0)   # big and sharp
        self.add(width=4000, height=3000, sharpness=20.0, verdict="screenshot")

        reach = md.slice_reach(self.conn, FEATURES, 1024)

        self.assertEqual(reach.photos, 4)                  # the screenshot is not one
        self.assertEqual(reach.above_ceiling, 3)
        self.assertEqual(reach.blurred, 3)
        self.assertEqual(reach.blurred_above_ceiling, 2)
        self.assertAlmostEqual(reach.share_of_blurred, 2 / 3)

    def test_the_printed_block_says_the_number_is_a_floor_and_not_a_size(self):
        text = md.format_reach(md.slice_reach(self.conn, FEATURES, 1024), 1024)
        self.assertIn("8%", text)               # what the filter actually finds
        self.assertIn("СНИЗУ", text)
        self.assertIn("мелких", text)           # ...and that small frames are decided

    def test_the_sample_holds_only_frames_of_this_feature_s_population(self):
        wanted = {self.add(width=4000, height=3000, sharpness=20.0),
                  self.add(width=3000, height=4000, sharpness=80.0)}
        self.add(width=800, height=600, sharpness=20.0)      # small: F169 decided it
        self.add(width=4000, height=3000, sharpness=400.0)   # sharp: not in the slice
        self.add(width=4000, height=3000, sharpness=None)    # never measured
        gone = self.add(width=4000, height=3000, sharpness=10.0, exists=False)

        picked = md.sample_frames(str(self.path), FEATURES, 1024, 10, seed=1)

        self.assertEqual({file_id for file_id, _ in picked}, wanted)
        self.assertNotIn(gone, {file_id for file_id, _ in picked})

    def test_the_sample_is_seeded_so_two_candidates_meet_the_same_frames(self):
        for n in range(12):
            self.add(width=4000, height=3000, sharpness=10.0 + n)
        first = md.sample_frames(str(self.path), FEATURES, 1024, 5, seed=7)
        again = md.sample_frames(str(self.path), FEATURES, 1024, 5, seed=7)
        self.assertEqual(first, again)
        self.assertEqual(len(first), 5)
        other = md.sample_frames(str(self.path), FEATURES, 1024, 5, seed=8)
        self.assertNotEqual([f for f, _ in first], [f for f, _ in other])

    def test_the_index_is_opened_read_only(self):
        self.add(width=4000, height=3000, sharpness=20.0)
        md.sample_frames(str(self.path), FEATURES, 1024, 5, seed=1)
        conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        self.addCleanup(conn.close)
        with self.assertRaises(sqlite3.OperationalError):
            conn.execute("DELETE FROM files")


class TestTheBlindPairs(unittest.TestCase):
    """A person who can tell which picture is which is not judging the pictures."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.original = Image.new("RGB", (400, 300), (30, 60, 90))
        self.processed = Image.new("RGB", (400, 300), (200, 180, 160))

    def pairs(self, count: int = 8) -> list:
        out = []
        for n in range(count):
            arm = "cand" if n % 2 == 0 else md.ARM_BASELINE
            out.append((run(n, arm, md.MOTION, None), self.original, self.processed))
        return out

    def test_the_sheets_are_numbered_and_never_named_after_frame_or_model(self):
        key = md.write_blind_pairs(self.pairs(2), self.root / "pairs", seed=7)

        names = sorted(p.name for p in (self.root / "pairs").iterdir())
        self.assertEqual([n for n in names if not n.endswith("_crop.jpg")],
                         ["pair_01.jpg", "pair_02.jpg"])
        for entry in key:
            self.assertNotIn("cand", entry["sheets"]["full"])
            self.assertIn(entry["left"], (md.ORIGINAL, md.PROCESSED))
            self.assertNotEqual(entry["left"], entry["right"])

    def test_the_key_says_which_instrument_and_which_degradation_each_pair_was(self):
        key = md.write_blind_pairs(self.pairs(2), self.root / "pairs", seed=7)
        self.assertEqual({entry["arm"] for entry in key}, {"cand", md.ARM_BASELINE})
        self.assertEqual({entry["degradation"] for entry in key}, {md.MOTION})
        self.assertEqual(sorted(entry["file_id"] for entry in key), [0, 1])

    def test_the_sheets_are_shuffled_across_the_instruments(self):
        """Two arms on the same frames, laid out in the order they ran, would alternate —
        and by the third sheet a person would be scoring a model, not a picture."""
        key = md.write_blind_pairs(self.pairs(8), self.root / "pairs", seed=3)
        arms = [entry["arm"] for entry in key]
        alternating = [("cand" if n % 2 == 0 else md.ARM_BASELINE) for n in range(8)]
        self.assertNotEqual(arms, alternating)
        self.assertEqual(sorted(arms), sorted(alternating))

    def test_the_layout_is_seeded_so_a_second_run_lays_out_the_same_sheets(self):
        first = md.write_blind_pairs(self.pairs(), self.root / "one", seed=11)
        again = md.write_blind_pairs(self.pairs(), self.root / "two", seed=11)
        self.assertEqual([(e["file_id"], e["arm"], e["left"]) for e in first],
                         [(e["file_id"], e["arm"], e["left"]) for e in again])

    def test_both_halves_are_the_same_size_and_the_sides_swap_when_flipped(self):
        straight = md.blind_sheet(self.original, self.processed, flipped=False)
        flipped = md.blind_sheet(self.original, self.processed, flipped=True)
        self.assertEqual(straight.size, flipped.size)
        self.assertEqual(straight.getpixel((5, 5)),
                         flipped.getpixel((flipped.width - 5, 5)))

    def test_a_copy_the_size_of_its_original_gets_a_1_to_1_crop_as_well(self):
        """Which is every arm here: the 1:1 candidate by construction, and the x4 arm
        because it comes back at about the size it was given. A 12 Mpx frame shrunk onto a
        sheet is a frame whose lost detail cannot be seen."""
        original = Image.effect_noise((3000, 2000), 40).convert("RGB")
        rebuilt = original.resize((3008, 2005))

        key = md.write_blind_pairs([(run(1, "cand", md.SOFT, None), original, rebuilt)],
                                   self.root / "pairs", seed=1)

        self.assertEqual(key[0]["sheets"]["crop"], "pair_01_crop.jpg")
        with Image.open(self.root / "pairs" / "pair_01_crop.jpg") as crop:
            self.assertEqual(crop.size, (md.CROP_BOX * 2 + 16, md.CROP_BOX))

    def test_both_sheets_of_one_pair_agree_on_which_side_is_which(self):
        dark = Image.new("RGB", (3000, 2000), (20, 20, 20))
        bright = Image.new("RGB", (3008, 2005), (220, 220, 220))

        key = md.write_blind_pairs([(run(1, "cand", md.SOFT, None), dark, bright)],
                                   self.root / "pairs", seed=1)

        corners = []
        for name in (key[0]["sheets"]["full"], key[0]["sheets"]["crop"]):
            with Image.open(self.root / "pairs" / name) as sheet:
                corners.append(sheet.getpixel((5, 5))[0] > 128)
        self.assertEqual(corners[0], corners[1])
        self.assertEqual(key[0]["left"] == md.PROCESSED, corners[0])


class TestTheVerdictBelongsToAPerson(unittest.TestCase):
    def test_the_question_asked_is_fidelity_and_not_which_one_is_better(self):
        """The substituted question is what gave the wrong answer twice in one day, in
        both directions. It is asked once, in the words a person will read."""
        text = md.format_verdict_prompt(Path("measure_deblur"))
        self.assertIn("ближе к тому, что было", text)
        self.assertIn("ЧЕЛОВЕК", text)
        self.assertIn("key.json", text)
        self.assertIn("Метрика", text)

    def test_all_three_outcomes_of_the_brief_are_named_before_the_looking_starts(self):
        text = md.format_verdict_prompt(Path("measure_deblur"))
        # The stem, not the nominative: the prose declines it ("только на смазе
        # движения"), and demanding a case that flowing Russian does not use
        # would make the test about grammar rather than about the outcome.
        self.assertIn("смаз", text)             # works on one degradation only
        self.assertIn("realworld-sr-x4", text)  # ...and where the current model stays
        self.assertIn("F168", text)             # ...and what a "no" costs

    def test_the_split_is_offered_as_a_heuristic_a_person_may_correct(self):
        text = md.format_verdict_prompt(Path("measure_deblur"))
        self.assertIn("эвристика", text)


class TestTheArms(unittest.TestCase):
    def test_the_original_comes_first_the_candidates_next_and_the_baseline_last(self):
        arms = md.build_arms({"cand": softening}, doubling, 1024)
        self.assertEqual([a.name for a in arms],
                         [md.ARM_ORIGINAL, "cand", md.ARM_BASELINE])
        self.assertTrue(arms[0].is_original)
        self.assertIsNone(arms[1].max_edge)     # the frame as it lies — the whole point
        self.assertEqual(arms[2].max_edge, 1024)

    def test_dropping_the_baseline_is_possible_and_leaves_the_original_in_place(self):
        arms = md.build_arms({"cand": softening}, None, 1024)
        self.assertEqual([a.name for a in arms], [md.ARM_ORIGINAL, "cand"])

    def test_a_model_name_is_shortened_to_something_a_column_can_hold(self):
        self.assertEqual(md.short_name("owner/some-weights"), "some-weights")
        self.assertLessEqual(len(md.short_name("owner/" + "x" * 40)), 14)


if __name__ == "__main__":
    unittest.main()
