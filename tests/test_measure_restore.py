"""F169, phase 0: the measurement that decides what happens to a full-sized frame.

The script's own honesty is what is checked here, because everything that follows in this
feature is read off its output:

* the three populations are three, and they split where the ceiling does — an average
  over "a downloaded picture" and "a 12 Mpx camera shot" would answer neither question;
* the BASELINE is the original itself, in every table. Without it "it got sharper" has
  nothing to be sharper than;
* a run that does not fit into memory is a ROW, not a crash: "the full frame does not fit"
  is one of the answers this measurement exists to produce;
* the pairs for the eyes are BLIND — both halves the same size, the order seeded, and
  nothing in a file name saying which is which;
* nothing printed identifies a frame (the rule of every measurement script here);
* the originals are read and never written.

No model, no GPU, no collection: the upscaler is a stub of the same shape the feature's
own tests use, and every function below is arithmetic or Pillow.
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "measure_restore.py"


def _load_script():
    """Import scripts/measure_restore.py — it is a script, not a package module."""
    spec = importlib.util.spec_from_file_location("measure_restore", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mr = _load_script()


def doubling(image: Image.Image) -> Image.Image:
    """A stand-in for Swin2SR: an image in, a bigger image out (x2 keeps fixtures small)."""
    return image.resize((image.width * 2, image.height * 2))


def make_jpeg(path: Path, size=(2400, 1800), color=(90, 120, 160)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path, "JPEG")
    return path


def run(file_id: int, population: str, max_edge: int | None, *, seconds: float = 1.0,
        source=(4000, 3000), input_size=(1024, 768), output=(4096, 3072),
        weight: int = 1024 * 1024, vram: float | None = None,
        error: str | None = None) -> object:
    return mr.FrameRun(file_id=file_id, population=population, max_edge=max_edge,
                       source_size=source, input_size=input_size, output_size=output,
                       weight_bytes=weight, seconds=seconds, peak_vram_mb=vram,
                       error=error)


class TestTheThreePopulations(unittest.TestCase):
    """They ask different questions, and the split is where the ceiling is."""

    def test_the_bands_split_at_the_ceiling_and_at_a_full_camera_frame(self):
        for edge, expected in ((320, mr.SMALL), (1023, mr.SMALL), (1024, mr.SMALL),
                               (1025, mr.MID), (2500, mr.MID),
                               (2501, mr.BIG), (4032, mr.BIG)):
            with self.subTest(edge=edge):
                self.assertEqual(mr.population_of(edge), expected)

    def test_every_band_has_a_label_that_states_its_own_sizes(self):
        self.assertEqual(set(mr.POPULATION_LABEL), set(mr.POPULATIONS))
        for population in mr.POPULATIONS:
            self.assertTrue(mr.POPULATION_LABEL[population].strip())


class TestHowMuchOfTheOriginalSurvives(unittest.TestCase):
    """The one number here that needs no eye: the share of the frame's own pixels the
    model was even shown. Areal, because the ceiling cuts both sides."""

    def test_a_frame_under_the_ceiling_keeps_everything(self):
        self.assertEqual(mr.truth_kept(800, 1024), 1.0)
        self.assertEqual(mr.truth_kept(1024, 1024), 1.0)

    def test_halving_the_side_keeps_a_quarter_of_the_pixels(self):
        self.assertAlmostEqual(mr.truth_kept(2048, 1024), 0.25)

    def test_a_12_megapixel_frame_at_the_shipped_ceiling(self):
        # 4032 -> 1024 is the case the whole feature is about: about 6% of what was there.
        self.assertAlmostEqual(mr.truth_kept(4032, 1024), 0.0645, places=3)

    def test_the_full_frame_and_the_baseline_give_up_nothing(self):
        self.assertEqual(mr.truth_kept(4032, mr.FULL_FRAME), 1.0)
        self.assertEqual(mr.truth_kept(4032, None), 1.0)

    def test_a_frame_with_no_size_is_not_a_division_by_zero(self):
        self.assertEqual(mr.truth_kept(0, 1024), 0.0)


class TestOneFrameThroughTheModel(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_the_row_says_what_went_in_and_what_came_out(self):
        src = make_jpeg(self.root / "shot.jpg", size=(2400, 1800))

        row, processed = mr.run_frame(doubling, src, 7, 600)

        self.assertIsNone(row.error)
        self.assertEqual(row.file_id, 7)
        self.assertEqual(row.population, mr.MID)
        self.assertEqual(row.source_size, (2400, 1800))
        self.assertEqual(row.input_size, (600, 450))
        self.assertEqual(row.output_size, (1200, 900))
        self.assertGreater(row.weight_bytes, 0)
        self.assertGreater(row.seconds, 0.0)
        assert processed is not None
        self.assertEqual(processed.size, (1200, 900))

    def test_the_full_frame_is_the_frame_itself(self):
        """`0` is "no ceiling" — the variant the brief expects to hit the memory wall,
        and a wall has to be measured rather than assumed."""
        src = make_jpeg(self.root / "shot.jpg", size=(1200, 900))
        row, _ = mr.run_frame(doubling, src, 1, mr.FULL_FRAME)
        self.assertEqual(row.input_size, (1200, 900))
        self.assertEqual(mr.truth_kept(row.source_edge, row.max_edge), 1.0)

    def test_a_model_that_runs_out_of_memory_is_a_row_and_not_a_crash(self):
        src = make_jpeg(self.root / "shot.jpg", size=(1200, 900))

        def failing(_image: Image.Image) -> Image.Image:
            raise RuntimeError("CUDA out of memory")

        row, processed = mr.run_frame(failing, src, 3, mr.FULL_FRAME)

        self.assertIsNone(processed)
        self.assertIn("CUDA out of memory", row.error or "")
        self.assertEqual(row.output_size, (0, 0))

    def test_a_frame_that_will_not_read_is_a_row_too(self):
        broken = self.root / "broken.jpg"
        broken.write_bytes(b"not an image")
        row, processed = mr.run_frame(doubling, broken, 4, 1024)
        self.assertIsNone(processed)
        self.assertTrue(row.error)

    def test_the_original_is_only_ever_read(self):
        """This script measures the action; it does not perform it. Nothing is written
        beside anybody's photograph — not a copy, not a re-encode."""
        src = make_jpeg(self.root / "shot.jpg", size=(1200, 900))
        before, mtime = src.read_bytes(), src.stat().st_mtime

        mr.run_frame(doubling, src, 1, 600)
        mr.baseline_run(src, 1)

        self.assertEqual(src.read_bytes(), before)
        self.assertEqual(src.stat().st_mtime, mtime)
        self.assertEqual([p.name for p in self.root.iterdir()], ["shot.jpg"])

    def test_the_baseline_row_is_the_file_as_it_lies(self):
        src = make_jpeg(self.root / "shot.jpg", size=(3000, 2000))
        row = mr.baseline_run(src, 9)
        self.assertTrue(row.is_baseline)
        self.assertEqual(row.population, mr.BIG)
        self.assertEqual(row.source_size, (3000, 2000))
        self.assertEqual(row.weight_bytes, src.stat().st_size)
        self.assertEqual(row.seconds, 0.0)

    def test_the_weight_is_the_one_the_feature_really_writes(self):
        small = mr.weigh_jpeg(Image.new("RGB", (64, 48), (10, 20, 30)))
        big = mr.weigh_jpeg(Image.effect_noise((600, 400), 60).convert("RGB"))
        self.assertGreater(small, 0)
        self.assertGreater(big, small)


class TestTheTables(unittest.TestCase):
    def rows(self) -> list:
        return [
            run(1, mr.BIG, None, seconds=0.0, input_size=(4000, 3000),
                output=(4000, 3000), weight=3 * 1024 * 1024),
            run(1, mr.BIG, 1024, seconds=1.0, vram=1980.0),
            run(1, mr.BIG, 2048, seconds=3.4, input_size=(2048, 1536),
                output=(8192, 6144), vram=7100.0),
            run(1, mr.BIG, mr.FULL_FRAME, error="RuntimeError: CUDA out of memory"),
        ]

    def table(self) -> str:
        return mr.format_population_table(mr.BIG, self.rows(), [1024, 2048, mr.FULL_FRAME])

    def test_the_original_is_the_first_row(self):
        lines = [line for line in self.table().splitlines() if line.strip()]
        body = [line for line in lines if "оригинал" in line or "1024" in line]
        self.assertIn("оригинал", body[0])

    def test_every_ceiling_gets_a_row_and_the_failure_says_why(self):
        table = self.table()
        self.assertIn("1024", table)
        self.assertIn("2048", table)
        self.assertIn("целиком", table)
        self.assertIn("CUDA out of memory", table)

    def test_the_table_prints_time_weight_memory_and_what_was_kept(self):
        table = self.table()
        self.assertIn("1.00 с", table)
        self.assertIn("1980 МБ", table)
        self.assertIn("3.00 МБ", table)   # the original's own weight
        self.assertIn("26%", table)       # 1024 of a 4000 px frame, areally

    def test_a_population_with_no_frames_says_so_instead_of_printing_nothing(self):
        table = mr.format_population_table(mr.SMALL, [], [1024])
        self.assertIn("нет кадров", table)

    def test_nothing_in_the_tables_identifies_a_frame(self):
        """The rule every measurement here follows: counts and sizes, never a path."""
        table = self.table()
        for leak in (".jpg", "shot", "/", "\\"):
            with self.subTest(leak=leak):
                self.assertNotIn(leak, table)

    def test_a_failed_run_is_counted_and_never_averaged_into_a_time(self):
        summary = mr.summarize([run(1, mr.BIG, 2048, seconds=2.0),
                                run(2, mr.BIG, 2048, error="RuntimeError: nope")])
        self.assertEqual((summary.frames, summary.failed), (1, 1))
        self.assertAlmostEqual(summary.seconds, 2.0)

    def test_the_rows_come_baseline_first_then_the_ceilings_as_asked(self):
        order = [s.max_edge for s in mr.summaries(self.rows(), [1024, 2048, mr.FULL_FRAME])]
        self.assertEqual(order, [None, 1024, 2048, mr.FULL_FRAME])

    def test_a_ceiling_nobody_measured_is_not_invented_as_an_empty_row(self):
        order = [s.max_edge for s in mr.summaries(self.rows(), [1024, 4096])]
        self.assertEqual(order, [None, 1024])


class TestTheBlindPairs(unittest.TestCase):
    """The only honest judge of "better" is a person, and a person who can tell which
    picture is which is not judging the pictures."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.original = Image.new("RGB", (400, 300), (30, 60, 90))
        self.processed = Image.new("RGB", (1600, 1200), (200, 180, 160))

    def test_both_halves_are_drawn_at_exactly_the_same_size(self):
        sheet = mr.blind_sheet(self.original, self.processed, flipped=False)
        half = mr.sheet_half_size(self.original.size)
        self.assertEqual(sheet.size, (half[0] * 2 + mr.SHEET_GAP, half[1]))

    def test_the_sides_swap_when_the_pair_is_flipped(self):
        straight = mr.blind_sheet(self.original, self.processed, flipped=False)
        flipped = mr.blind_sheet(self.original, self.processed, flipped=True)
        self.assertEqual(straight.size, flipped.size)
        self.assertEqual(straight.getpixel((5, 5)), flipped.getpixel(
            (flipped.width - 5, 5)))
        self.assertNotEqual(straight.getpixel((5, 5)), flipped.getpixel((5, 5)))

    def test_the_sheets_are_numbered_and_never_named_after_the_frame(self):
        pairs = [(run(11, mr.BIG, 1024), self.original, self.processed),
                 (run(12, mr.MID, 2048), self.original, self.processed)]

        key = mr.write_blind_pairs(pairs, self.root / "pairs", seed=7)

        names = sorted(p.name for p in (self.root / "pairs").iterdir())
        self.assertEqual(names, ["pair_01.jpg", "pair_02.jpg"])
        for entry, name in zip(key, names):
            self.assertEqual(entry["sheets"]["full"], name)
            self.assertIn(entry["left"], (mr.ORIGINAL, mr.PROCESSED))
            self.assertNotEqual(entry["left"], entry["right"])
        self.assertEqual([entry["file_id"] for entry in key], [11, 12])
        self.assertEqual([entry["max_edge"] for entry in key], [1024, 2048])

    def test_a_copy_the_size_of_its_original_also_gets_a_1_to_1_crop(self):
        """The population this measurement is about: the copy came back the size it went
        in, so the difference is visible at native scale and nowhere else."""
        original = Image.effect_noise((3000, 2000), 40).convert("RGB")
        rebuilt = original.resize((3008, 2005))   # what x4 over a reduced frame gives back

        key = mr.write_blind_pairs([(run(1, mr.BIG, 1024), original, rebuilt)],
                                   self.root / "pairs", seed=1)

        self.assertEqual(sorted(p.name for p in (self.root / "pairs").iterdir()),
                         ["pair_01.jpg", "pair_01_crop.jpg"])
        with Image.open(self.root / "pairs" / "pair_01_crop.jpg") as crop:
            # Two crops at native scale, side by side — no resampling of either half.
            self.assertEqual(crop.size, (mr.CROP_BOX * 2 + mr.SHEET_GAP, mr.CROP_BOX))
        self.assertEqual(key[0]["sheets"]["crop"], "pair_01_crop.jpg")

    def test_a_real_enlargement_gets_no_crop_sheet(self):
        """Below the ceiling the copy IS four times bigger, and there is no 1:1 pair to
        show — cropping would compare two different fields of view."""
        key = mr.write_blind_pairs([(run(1, mr.SMALL, 1024), self.original,
                                     self.processed)], self.root / "pairs", seed=1)
        self.assertNotIn("crop", key[0]["sheets"])
        self.assertEqual([p.name for p in (self.root / "pairs").iterdir()],
                         ["pair_01.jpg"])

    def test_both_sheets_of_one_pair_agree_on_which_side_is_which(self):
        """Two sheets of one frame disagreeing would hand the answer over: whichever of
        them a person opened second would tell them what the first one hid."""
        dark = Image.new("RGB", (3000, 2000), (20, 20, 20))
        bright = Image.new("RGB", (3008, 2005), (220, 220, 220))

        key = mr.write_blind_pairs([(run(1, mr.BIG, 1024), dark, bright)],
                                   self.root / "pairs", seed=1)

        corners = []
        for name in (key[0]["sheets"]["full"], key[0]["sheets"]["crop"]):
            with Image.open(self.root / "pairs" / name) as sheet:
                corners.append(sheet.getpixel((5, 5))[0] > 128)
        self.assertEqual(corners[0], corners[1])
        # ...and the key says the same thing the sheets do.
        self.assertEqual(key[0]["left"] == mr.PROCESSED, corners[0])

    def test_what_counts_as_the_same_size_and_what_does_not(self):
        self.assertTrue(mr.same_scale((4032, 3024), (4096, 3072)))
        self.assertFalse(mr.same_scale((1000, 750), (4000, 3000)))
        self.assertFalse(mr.same_scale((0, 0), (4000, 3000)))

    def test_the_crop_is_the_middle_and_never_bigger_than_the_frame(self):
        crop = mr.centre_crop(Image.new("RGB", (200, 150)), box=700)
        self.assertEqual(crop.size, (150, 150))
        self.assertEqual(mr.centre_crop(Image.new("RGB", (3000, 2000))).size,
                         (mr.CROP_BOX, mr.CROP_BOX))

    def test_the_order_is_seeded_and_not_always_the_same(self):
        pairs = [(run(n, mr.BIG, 1024), self.original, self.processed)
                 for n in range(20)]
        key = mr.write_blind_pairs(pairs, self.root / "pairs", seed=20260804)
        sides = {entry["left"] for entry in key}
        self.assertEqual(sides, {mr.ORIGINAL, mr.PROCESSED})
        again = mr.write_blind_pairs(pairs, self.root / "again", seed=20260804)
        self.assertEqual([e["left"] for e in again], [e["left"] for e in key])


class TestTheVerdictBelongsToAPerson(unittest.TestCase):
    def test_the_prompt_names_both_outcomes_and_hands_the_choice_over(self):
        text = mr.format_verdict_prompt(Path("measure_restore"))
        self.assertIn("ЧЕЛОВЕК", text)
        self.assertIn("key.json", text)
        self.assertIn("D:", text)   # worse than the original -> close the action
        self.assertIn("C:", text)   # comparable -> tiling plus a return to the size
        self.assertIn("F168", text)


if __name__ == "__main__":
    unittest.main()
