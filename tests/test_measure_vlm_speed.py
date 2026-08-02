"""F101/F105: the VLM speed measurement — everything about it that is not the model.

The script exists to answer two questions the feature is accepted on: how much faster
the pass got, and whether a single verdict moved. Both answers are arithmetic over
per-frame aggregates, so both are testable here with a fake runtime — no transformers,
no GPU, no photo.

Three of these tests are about the brief rather than about code:

* a mismatch of even one label must make the report say STOP and the process exit
  non-zero — "ну почти" was ruled out in writing before the measurement existed;
* a batched pass must produce the labels of the unbatched one, in the same order, for
  the same files — the batch is the same arithmetic, so a moved label is a defect;
* nothing the script prints may identify a frame — a table about documents must not
  become a list of where the documents are (the rule of measure_ocr_gate.py before it).

F144 reopened the batch half of the question under a condition F105 itself changed, and
added `--per-call` to the same script. Its tests are the last classes of this file and
are about a different property: not "did a verdict move" but "is the arithmetic of
seconds per image right, and is a fast call that answers rubbish counted as rubbish".
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from sorta import naming
from sorta.db import connect
from sorta.junk import PreparedFrame, SplitVlmClassifier

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "measure_vlm_speed.py"


def _load_script():
    """Import scripts/measure_vlm_speed.py — it is a script, not a package module."""
    spec = importlib.util.spec_from_file_location("measure_vlm_speed", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


speed = _load_script()

# The label a frame of this width gets from the fake runtime below. Real labels, because
# the script parses the answer with the stage's own parser.
MARK_LABELS = {20: "document", 21: "product", 22: "personal_photo", 23: "document",
               24: "product", 25: "personal_photo", 26: "document"}


def result(name, labels, frame_ms=(10.0,), wall=1.0, workers=1, batch=1,
           attn="default", vram=7890.0):
    return speed.ModeResult(
        name=name, workers=workers, labels=tuple(labels), frame_ms=tuple(frame_ms),
        wall_sec=wall, cpu_cores=0.8, gpu_util_pct=26.0, peak_vram_mb=vram,
        attn=attn, kernels="sdpa/eager", batch=batch)


def batch_runtime(fail_prepare=(), fail_generate=(), fail_batch=False,
                  batch_answers=None):
    """A naming.BatchVlm whose answer for a frame is the label of its WIDTH.

    The width is the identity of the frame all the way through the pass — that is what
    makes it visible when a batch hands the answer of one file to another.

    The failures are the three shapes the pass has to survive: a frame the processor
    chokes on (`fail_prepare`), a frame the model dies on (`fail_generate`) and a batch
    that fails only BECAUSE it is a batch (`fail_batch` — no VRAM for N at once), which
    is the case where retrying frame by frame must recover every one of them.
    """
    def prepare_batch(groups, prompt):
        marks = [group[0].width for group in groups]
        if any(mark in fail_prepare for mark in marks):
            raise RuntimeError("процессор подавился кадром")
        if fail_batch and len(groups) > 1:
            raise RuntimeError("нет памяти на батч целиком")
        return marks

    def generate_batch(prepared, max_new_tokens):
        if any(mark in fail_generate for mark in prepared):
            raise RuntimeError("CUDA out of memory")
        if batch_answers is not None and len(prepared) > 1:
            return list(batch_answers)
        return [MARK_LABELS[mark] for mark in prepared]

    def prepare(frames, prompt):
        return prepare_batch([frames], prompt)

    def generate(prepared, max_new_tokens):
        return generate_batch(prepared, max_new_tokens)[0]

    return naming.BatchVlm(prepare=prepare, generate=generate,
                           prepare_batch=prepare_batch, generate_batch=generate_batch)


class FramesOnDisk(unittest.TestCase):
    """Real JPEGs: the batched pass decodes the frames itself, as the stage does."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def frames(self, marks):
        """One JPEG per mark, the mark being its width; returns the paths in order."""
        paths = []
        for i, mark in enumerate(marks):
            path = Path(self.tmp.name) / f"f_{i}_{mark}.jpg"
            Image.new("RGB", (mark, 4), (10, 100, 200)).save(path, "JPEG")
            paths.append(str(path))
        return paths


class TestPercentile(unittest.TestCase):
    """Nearest rank: a p90 must be a frame that really took that long."""

    def test_p90_is_a_real_observation(self):
        values = [float(i) for i in range(1, 11)]
        self.assertEqual(speed.percentile(values, 0.9), 9.0)
        self.assertEqual(speed.percentile(values, 0.5), 5.0)

    def test_edges(self):
        self.assertEqual(speed.percentile([], 0.9), 0.0)
        self.assertEqual(speed.percentile([7.0], 0.9), 7.0)
        self.assertEqual(speed.percentile([3.0, 1.0, 2.0], 1.0), 3.0)
        self.assertEqual(speed.percentile([3.0, 1.0, 2.0], 0.0), 1.0)


class TestSpreadOverBatch(unittest.TestCase):
    """A batch answers every frame at the same instant — the cost is shared, not zero."""

    def test_the_batch_time_is_divided_between_its_frames(self):
        self.assertEqual(speed.spread_over_batch([400.0, 0.0, 0.0, 0.0], 4),
                         [100.0] * 4)

    def test_a_short_last_batch_keeps_its_own_frames(self):
        self.assertEqual(speed.spread_over_batch([400.0, 0.0, 60.0], 2),
                         [200.0, 200.0, 60.0])

    def test_without_a_batch_the_timings_are_untouched(self):
        self.assertEqual(speed.spread_over_batch([5.0, 7.0], 1), [5.0, 7.0])


class TestModeGrid(unittest.TestCase):
    """The modes are the grid, and the baseline — no attention request, no batch."""

    def test_the_grid_is_attention_major(self):
        specs = speed.mode_specs(["default", "sdpa"], [1, 4])
        self.assertEqual([s.name for s in specs],
                         ["default/b1", "default/b4", "sdpa/b1", "sdpa/b4"])

    def test_the_default_mode_asks_for_nothing(self):
        self.assertIsNone(speed.ATTN_SPECS["default"])
        self.assertEqual(speed.requested_kernels("default"), {})

    def test_a_named_kernel_covers_both_halves(self):
        self.assertEqual(speed.requested_kernels("sdpa"),
                         {"language": "sdpa", "vision": "sdpa"})

    def test_the_tower_can_be_asked_for_alone(self):
        self.assertEqual(speed.ATTN_SPECS["vision-sdpa"], {"vision_config": "sdpa"})
        self.assertEqual(speed.requested_kernels("vision-sdpa"), {"vision": "sdpa"})


class TestUnmetRequest(unittest.TestCase):
    """A kernel transformers could not give is a table that lies — it has to be seen."""

    def test_a_request_that_arrived_is_silent(self):
        self.assertEqual(
            speed.unmet_request("sdpa", {"language": "sdpa", "vision": "sdpa"}), {})

    def test_a_tower_still_on_eager_is_reported(self):
        self.assertEqual(
            speed.unmet_request("sdpa", {"language": "sdpa", "vision": "eager"}),
            {"vision": ("sdpa", "eager")})

    def test_nothing_was_asked_so_nothing_is_unmet(self):
        self.assertEqual(
            speed.unmet_request("default", {"language": "sdpa", "vision": "eager"}), {})


class TestModeResultStats(unittest.TestCase):
    def test_median_p90_and_rate(self):
        r = result("x", ["document"] * 4, frame_ms=(100.0, 200.0, 300.0, 1000.0),
                   wall=1.6)
        self.assertEqual(r.median_ms, 250.0)
        self.assertEqual(r.p90_ms, 1000.0)
        self.assertAlmostEqual(r.frames_per_sec, 2.5)

    def test_empty_pass_does_not_divide_by_zero(self):
        r = result("x", [], frame_ms=(), wall=0.0)
        self.assertEqual((r.median_ms, r.p90_ms, r.frames_per_sec), (0.0, 0.0, 0.0))


class TestLabelMismatches(unittest.TestCase):
    def test_identical_labels_are_no_mismatch(self):
        labels = ["document", "product", "personal_photo"]
        self.assertEqual(
            speed.label_mismatches(result("a", labels), result("b", labels)), {})

    def test_mismatches_are_counted_per_label_pair(self):
        base = result("a", ["document", "document", "product"])
        other = result("b", ["product", "document", "product"])
        self.assertEqual(speed.label_mismatches(base, other),
                         {("document", "product"): 1})

    def test_a_missing_frame_is_a_mismatch(self):
        base = result("a", ["document", "product"])
        self.assertEqual(speed.label_mismatches(base, result("b", ["document"])),
                         {("<нет кадра>", "<нет кадра>"): 1})


class TestVerdictReport(unittest.TestCase):
    """The acceptance criterion in report form: one moved label is a stop, not a note."""

    def test_full_match_is_reported_and_accepted(self):
        labels = ["document", "product"]
        report, ok = speed.format_verdicts(
            [result("default/b1", labels), result("sdpa/b4", labels)])
        self.assertTrue(ok)
        self.assertIn("совпадение полное", report)
        self.assertNotIn("СТОП", report)

    def test_one_moved_label_is_a_stop(self):
        report, ok = speed.format_verdicts([
            result("default/b1", ["document", "product", "product"]),
            result("sdpa/b4", ["document", "product", "personal_photo"]),
        ])
        self.assertFalse(ok)
        self.assertIn("СТОП", report)
        self.assertIn("product -> personal_photo: 1", report)

    def test_a_single_mode_has_nothing_to_compare(self):
        report, ok = speed.format_verdicts([result("default/b1", ["document"])])
        self.assertTrue(ok)
        self.assertIn("сравнивать не с чем", report)


class TestCriteria(unittest.TestCase):
    """The pre-registered thresholds, applied — and none of them applied loosely."""

    def base(self, n=speed.MIN_SAMPLE):
        return result("default/b1", ["document"] * n, wall=10.0)

    def candidate(self, labels=None, wall=5.0, batch=4, attn="sdpa", vram=7890.0,
                  n=speed.MIN_SAMPLE):
        return result(f"{attn}/b{batch}", labels or ["document"] * n, wall=wall,
                      batch=batch, attn=attn, vram=vram)

    def test_a_mode_that_is_faster_and_agrees_is_accepted(self):
        check = speed.assess(self.base(), self.candidate())
        self.assertTrue(check.accepted)
        self.assertAlmostEqual(check.speedup, 2.0)
        self.assertEqual(check.mismatches, 0)

    def test_a_single_moved_label_rejects_the_mode(self):
        labels = ["document"] * speed.MIN_SAMPLE
        labels[7] = "product"
        check = speed.assess(self.base(), self.candidate(labels=labels))
        self.assertEqual(check.mismatches, 1)
        self.assertFalse(check.verdicts_match)
        self.assertFalse(check.accepted)

    def test_a_speedup_below_the_threshold_rejects_the_mode(self):
        check = speed.assess(self.base(), self.candidate(wall=10.0 / 1.10))
        self.assertFalse(check.fast_enough)
        self.assertFalse(check.accepted)

    def test_a_mode_that_does_not_fit_in_vram_is_rejected_however_fast(self):
        check = speed.assess(self.base(), self.candidate(wall=1.0, vram=18000.0))
        self.assertTrue(check.fast_enough)
        self.assertFalse(check.fits_in_vram)
        self.assertFalse(check.accepted)

    def test_a_sample_below_the_minimum_proves_nothing(self):
        small = 10
        check = speed.assess(self.base(n=small), self.candidate(n=small))
        self.assertFalse(check.sample_enough)
        self.assertFalse(check.accepted)

    def test_a_missing_vram_reading_does_not_reject(self):
        """No CUDA — no number; that is not evidence against the mode."""
        check = speed.assess(self.base(), self.candidate(vram=None))
        self.assertTrue(check.fits_in_vram)


class TestOutcome(unittest.TestCase):
    """A / B / C — the brief's three endings, decided by the numbers alone."""

    def modes(self, *specs):
        n = speed.MIN_SAMPLE
        out = [result("default/b1", ["document"] * n, wall=10.0)]
        for attn, batch, wall, labels, vram in specs:
            out.append(result(f"{attn}/b{batch}", labels or ["document"] * n, wall=wall,
                              batch=batch, attn=attn, vram=vram))
        return out

    def test_a_batched_mode_that_passed_is_outcome_a(self):
        letter, why = speed.outcome(self.modes(("sdpa", 1, 8.0, None, 7890.0),
                                               ("sdpa", 4, 5.0, None, 9000.0)))
        self.assertEqual(letter, "A")
        self.assertIn("sdpa/b4", why)

    def test_only_the_kernel_passing_is_outcome_b(self):
        letter, why = speed.outcome(self.modes(("sdpa", 1, 5.0, None, 7890.0),
                                               ("sdpa", 4, 4.0, None, 20000.0)))
        self.assertEqual(letter, "B")
        self.assertIn("sdpa/b1", why)

    def test_nothing_passing_is_outcome_c(self):
        letter, why = speed.outcome(self.modes(("sdpa", 1, 9.5, None, 7890.0),
                                               ("sdpa", 4, 9.0, None, 7890.0)))
        self.assertEqual(letter, "C")
        self.assertIn("закрываем", why)

    def test_a_faster_mode_with_a_moved_label_is_not_an_outcome_a(self):
        moved = ["document"] * speed.MIN_SAMPLE
        moved[3] = "personal_photo"
        letter, _why = speed.outcome(self.modes(("sdpa", 4, 2.0, moved, 7890.0)))
        self.assertEqual(letter, "C")

    def test_one_mode_alone_decides_nothing(self):
        letter, why = speed.outcome([result("default/b1", ["document"])])
        self.assertEqual(letter, "C")
        self.assertIn("сравнивать не с чем", why)

    def test_the_report_line_names_the_outcome(self):
        text = speed.format_outcome(self.modes(("sdpa", 4, 5.0, None, 9000.0)))
        self.assertTrue(text.startswith("ИСХОД A"))


class TestReportIdentifiesNothing(unittest.TestCase):
    """Privacy: the report is aggregates — no path, no basename, no file id."""

    def test_no_frame_identity_reaches_the_output(self):
        results = [result("default/b1", ["document", "product"]),
                   result("sdpa/b4", ["product", "product"], batch=4, attn="sdpa")]
        text = "\n".join([speed.format_table(results),
                          speed.format_verdicts(results)[0],
                          speed.format_criteria(results),
                          speed.format_outcome(results)])
        for leak in ("/photos", ".jpg", "file_id", "IMG_"):
            self.assertNotIn(leak, text)


class TestFormatTable(unittest.TestCase):
    def test_speedup_is_measured_against_the_first_mode(self):
        base = result("default/b1", ["document"] * 10, wall=10.0, workers=1)
        fast = result("sdpa/b4", ["document"] * 10, wall=2.5, workers=4, batch=4,
                      attn="sdpa")
        table = speed.format_table([base, fast])
        self.assertIn("x1.00", table)
        self.assertIn("x4.00", table)
        self.assertIn("sdpa/eager", table)  # the kernels that really ran

    def test_a_mode_over_the_vram_budget_is_marked(self):
        base = result("default/b1", ["document"], wall=1.0)
        greedy = result("sdpa/b16", ["document"], wall=0.5, batch=16, vram=20000.0)
        self.assertIn("20000 МБ !", speed.format_table([base, greedy]))

    def test_the_criteria_block_says_why_a_mode_was_rejected(self):
        base = result("default/b1", ["document"] * speed.MIN_SAMPLE, wall=10.0)
        greedy = result("sdpa/b16", ["document"] * speed.MIN_SAMPLE, wall=1.0, batch=16,
                        vram=20000.0)
        text = speed.format_criteria([base, greedy])
        self.assertIn("отклонён", text)
        self.assertIn("не влезает", text)


class TestRunMode(unittest.TestCase):
    """run_mode times the stream it is given and never raises on a failed frame."""

    def stream(self, items):
        return iter(items)

    def test_labels_are_kept_in_order_with_a_timing_per_frame(self):
        spec = speed.ModeSpec(attn="default", batch=1)
        r = speed.run_mode(spec, self.stream(["product", "document"] * 3), 3,
                           kernels="sdpa/eager")
        self.assertEqual(list(r.labels), ["product", "document"] * 3)
        self.assertEqual(len(r.frame_ms), 6)
        self.assertGreater(r.wall_sec, 0.0)
        self.assertEqual((r.name, r.kernels, r.batch), ("default/b1", "sdpa/eager", 1))

    def test_a_failed_frame_is_recorded_not_raised(self):
        spec = speed.ModeSpec(attn="sdpa", batch=1)
        r = speed.run_mode(spec, self.stream(["product", RuntimeError("CUDA")]), 2)
        self.assertEqual(list(r.labels), ["product", "ERROR"])

    def test_a_batched_mode_shares_its_time_between_the_frames(self):
        spec = speed.ModeSpec(attn="default", batch=2)
        r = speed.run_mode(spec, self.stream(["document"] * 4), 2)
        self.assertEqual(r.frame_ms[0], r.frame_ms[1])
        self.assertEqual(r.frame_ms[2], r.frame_ms[3])


class TestDecodeFrame(FramesOnDisk):
    def test_a_frame_on_disk_decodes(self):
        image = speed.decode_frame(self.frames([20])[0], 896)
        self.assertIsNotNone(image)
        self.assertEqual(image.width, 20)

    def test_a_missing_file_is_not_an_error(self):
        self.assertIsNone(speed.decode_frame(str(Path(self.tmp.name) / "gone.jpg"), 896))

    def test_an_undecodable_file_is_not_an_error(self):
        broken = Path(self.tmp.name) / "broken.jpg"
        broken.write_bytes(b"not an image at all")
        self.assertIsNone(speed.decode_frame(str(broken), 896))


class TestPreparedChunk(FramesOnDisk):
    """The CPU half of a batch: what decoded, where it sat, and never an exception."""

    def test_every_frame_of_the_chunk_is_prepared_together(self):
        chunk = speed.prepare_chunk(batch_runtime(), self.frames([20, 21, 22]), 896)
        self.assertEqual(chunk.size, 3)
        self.assertEqual(chunk.kept, [0, 1, 2])
        self.assertEqual(chunk.prepared, [20, 21, 22])

    def test_a_frame_that_would_not_decode_keeps_its_position(self):
        paths = self.frames([20, 21])
        paths.insert(1, str(Path(self.tmp.name) / "gone.jpg"))
        chunk = speed.prepare_chunk(batch_runtime(), paths, 896)
        self.assertEqual((chunk.size, chunk.kept), (3, [0, 2]))
        self.assertEqual(chunk.prepared, [20, 21])

    def test_a_preparation_that_raises_does_not_leave_the_worker(self):
        chunk = speed.prepare_chunk(batch_runtime(fail_prepare={21}),
                                    self.frames([20, 21]), 896)
        self.assertIsNone(chunk.prepared)
        self.assertEqual(chunk.kept, [0, 1])


class TestChunkLabels(FramesOnDisk):
    """The GPU half: the answers go back BY POSITION, and a bad frame costs one frame."""

    def labels(self, marks, runtime=None, missing=()):
        paths = self.frames(marks)
        for position in missing:
            paths.insert(position, str(Path(self.tmp.name) / "gone.jpg"))
        runtime = runtime or batch_runtime()
        return speed.chunk_labels(runtime, speed.prepare_chunk(runtime, paths, 896))

    def test_the_answers_line_up_with_the_frames(self):
        self.assertEqual(self.labels([20, 21, 22]),
                         ["document", "product", "personal_photo"])

    def test_an_undecodable_frame_takes_the_tiers_conservative_label(self):
        """junk answers `personal_photo` for a frame it could not show the model."""
        self.assertEqual(self.labels([20, 21], missing=[1]),
                         ["document", "personal_photo", "product"])

    def test_a_batch_that_will_not_prepare_is_retried_frame_by_frame(self):
        """No VRAM for N frames at once must cost the pass nothing but the batching."""
        self.assertEqual(self.labels([20, 21], runtime=batch_runtime(fail_batch=True)),
                         ["document", "product"])

    def test_a_frame_the_processor_chokes_on_costs_only_itself(self):
        got = self.labels([20, 21], runtime=batch_runtime(fail_prepare={21}))
        self.assertEqual(got[0], "document")
        self.assertIsInstance(got[1], BaseException)

    def test_a_frame_that_cannot_be_generated_loses_only_itself(self):
        got = self.labels([20, 21], runtime=batch_runtime(fail_generate={21}))
        self.assertEqual(got[0], "document")
        self.assertIsInstance(got[1], BaseException)

    def test_a_model_that_answers_the_wrong_number_of_times_is_not_trusted(self):
        """The alignment is unknowable, so the batch is dropped and each frame re-asked."""
        got = self.labels([20, 21], runtime=batch_runtime(batch_answers=["document"]))
        self.assertEqual(got, ["document", "product"])

    def test_a_chunk_with_nothing_readable_never_reaches_the_model(self):
        runtime = batch_runtime(fail_prepare=set(MARK_LABELS))
        chunk = speed.prepare_chunk(runtime, [str(Path(self.tmp.name) / "gone.jpg")], 896)
        self.assertEqual(speed.chunk_labels(runtime, chunk), ["personal_photo"])


class TestBatchedPass(FramesOnDisk):
    """The property the whole feature rests on: the batch answers what the frames are."""

    def test_labels_come_back_in_input_order(self):
        marks = [20, 21, 22, 23, 24, 25, 26]
        got = list(speed.batched_labels(batch_runtime(), self.frames(marks), 3, 2, 896))
        self.assertEqual(got, [MARK_LABELS[m] for m in marks])

    def test_a_batch_larger_than_the_sample_is_one_chunk(self):
        marks = [20, 21]
        got = list(speed.batched_labels(batch_runtime(), self.frames(marks), 16, 4, 896))
        self.assertEqual(got, [MARK_LABELS[m] for m in marks])

    def test_batching_changes_no_label_of_the_unbatched_pass(self):
        marks = [20, 21, 22, 23, 24, 25, 26]
        paths = self.frames(marks)
        runtime = batch_runtime()
        serial = list(speed.mode_items(runtime, paths, speed.ModeSpec("default", 1),
                                       2, 896))
        for batch in (2, 4, 8):
            with self.subTest(batch=batch):
                batched = list(speed.mode_items(
                    runtime, paths, speed.ModeSpec("default", batch), 2, 896))
                self.assertEqual(batched, serial)

    def test_the_unbatched_mode_goes_through_the_stages_own_pass(self):
        """Batch 1 is not a re-implementation: it is junk._vlm_labels over the runtime."""
        marks = [20, 21, 22]
        got = list(speed.mode_items(batch_runtime(), self.frames(marks),
                                    speed.ModeSpec("default", 1), 1, 896))
        self.assertEqual(got, [MARK_LABELS[m] for m in marks])

    def test_a_single_bad_frame_does_not_end_the_pass(self):
        marks = [20, 21, 22, 23]
        got = list(speed.batched_labels(batch_runtime(fail_generate={22}),
                                        self.frames(marks), 2, 2, 896))
        self.assertEqual([got[0], got[1], got[3]], ["document", "product", "document"])
        self.assertIsInstance(got[2], BaseException)


class TestArguments(unittest.TestCase):
    """Test 6: the modes are turned on by arguments, and the defaults are the grid."""

    def parse(self, argv):
        return speed.build_parser().parse_args(argv)

    def test_the_default_run_measures_the_grid_against_the_shipped_path(self):
        args = self.parse([])
        self.assertEqual(args.attn[0], "default")
        self.assertEqual(args.batch[0], 1)
        self.assertEqual(speed.mode_specs(args.attn, args.batch)[0].name, "default/b1")
        self.assertEqual(args.sample, speed.MIN_SAMPLE)

    def test_the_levers_are_asked_for_by_name(self):
        args = self.parse(["--attn", "eager", "sdpa", "--batch", "1", "8"])
        self.assertEqual([s.name for s in speed.mode_specs(args.attn, args.batch)],
                         ["eager/b1", "eager/b8", "sdpa/b1", "sdpa/b8"])

    def test_an_unknown_kernel_is_refused(self):
        with self.assertRaises(SystemExit):
            self.parse(["--attn", "flash_attention_2"])


class TestMainExitCode(unittest.TestCase):
    """A measurement whose verdicts moved must not report success to the caller."""

    def run_main(self, results, argv=("--sample", "2")):
        cfg = type("Cfg", (), {"database": ":memory:", "vlm": type(
            "Vlm", (), {"model": "Qwen/test", "workers": 2, "max_edge": 896})()})()
        self.patch(speed, "load_config", lambda _path: cfg)
        self.patch(speed, "sample_paths", lambda *_a: (["/photos/a.jpg"], "тест"))
        self.patch(speed, "measure", lambda *_a: results)
        self.patch(sys, "argv", ["measure_vlm_speed.py", *argv])
        return speed.main()

    def patch(self, target, name, value):
        original = getattr(target, name)
        setattr(target, name, value)
        self.addCleanup(setattr, target, name, original)

    def test_matching_verdicts_return_zero(self):
        labels = ["document", "product"]
        self.assertEqual(
            self.run_main([result("default/b1", labels),
                           result("sdpa/b4", labels, batch=4, attn="sdpa")]), 0)

    def test_a_moved_verdict_returns_non_zero(self):
        self.assertEqual(
            self.run_main([result("default/b1", ["document", "product"]),
                           result("sdpa/b4", ["document", "document"], batch=4)]), 1)

    def test_a_batch_below_one_is_refused(self):
        with self.assertRaises(SystemExit):
            self.run_main([], argv=("--batch", "0"))


class TestSamplePaths(unittest.TestCase):
    """The sample is the deep tier's own kind of frame, and it must exist on disk."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = Path(self.tmp.name) / "test.db"
        self.conn = connect(self.db)
        self.addCleanup(self.conn.close)

    def add(self, name, source=None, has_face=False, on_disk=True):
        path = Path(self.tmp.name) / name
        if on_disk:
            path.write_bytes(b"x")
        cur = self.conn.execute(
            """INSERT INTO files (path, size, mtime, ext, media_type, indexed_at)
               VALUES (?, 1, 0, 'jpg', 'photo', '2026-01-01')""", (str(path),))
        if source is not None:
            self.conn.execute(
                """INSERT INTO media_class (file_id, verdict, source, score, updated_at,
                       tier) VALUES (?, 'photo', ?, NULL, '2026-01-01', 'vlm')""",
                (cur.lastrowid, source))
        if has_face:
            self.conn.execute(
                "INSERT INTO faces (file_id, bbox, embedding) VALUES (?, '[1,2,3,4]', ?)",
                (cur.lastrowid, b"\x00" * 4))
        self.conn.commit()
        return str(path)

    def test_previous_deep_candidates_are_preferred(self):
        self.add("clip_only.jpg", source="clip")
        deep = {self.add(f"deep_{i}.jpg", source="vlm") for i in range(3)}
        paths, origin = speed.sample_paths(str(self.db), 10, seed=1)
        self.assertEqual(set(paths), deep)
        self.assertIn("source='vlm'", origin)

    def test_falls_back_to_canonical_photos_without_faces(self):
        plain = {self.add(f"plain_{i}.jpg") for i in range(2)}
        self.add("portrait.jpg", has_face=True)
        paths, origin = speed.sample_paths(str(self.db), 10, seed=1)
        self.assertEqual(set(paths), plain)
        self.assertIn("без лиц", origin)

    def test_missing_files_and_the_sample_size_are_respected(self):
        for i in range(4):
            self.add(f"deep_{i}.jpg", source="vlm")
        self.add("gone.jpg", source="vlm", on_disk=False)
        paths, _origin = speed.sample_paths(str(self.db), 2, seed=1)
        self.assertEqual(len(paths), 2)
        self.assertTrue(all(Path(p).exists() for p in paths))

    def test_sampling_is_deterministic_for_a_seed(self):
        for i in range(6):
            self.add(f"deep_{i}.jpg", source="vlm")
        first, _ = speed.sample_paths(str(self.db), 3, seed=7)
        second, _ = speed.sample_paths(str(self.db), 3, seed=7)
        self.assertEqual(first, second)


class TestSplitClassifierStillWorks(unittest.TestCase):
    """The pipelined pass of F101 is what batch 1 measures — it must still be usable."""

    def test_the_stages_own_pipeline_yields_in_input_order(self):
        from sorta import junk

        def prepare(path):
            return PreparedFrame(inputs=Path(path).name)

        def classify_prepared(prepared):
            return "document" if prepared.inputs.endswith("1.jpg") else "product"

        classifier = SplitVlmClassifier(prepare=prepare,
                                        classify_prepared=classify_prepared)
        paths = [f"/photos/f_{i}.jpg" for i in range(4)]
        self.assertEqual(list(junk._vlm_labels(classifier, paths, 3)),
                         ["product", "document", "product", "product"])


# --- F144: the price of a call by the number of images in it -----------------

# What the fake runtime answers about a frame of this width. Three of them are on
# purpose not answers to the question at all: an unreadable answer is the thing the
# report has to count, or a batch that started producing rubbish would look like a win.
CALL_ANSWERS = {20: "indoor", 21: "outdoor", 22: "unclear", 23: "Indoor.",
                24: "hard to say, honestly", 25: "outdoor", 26: ""}


class Clock:
    """A clock that only moves while the model is being asked — one tick per call.

    Wall time is what the measurement is about, so it must not be real here: a fake
    clock that advances by a known amount inside `generate_batch` and nowhere else makes
    every per-call number in the table exact instead of "greater than zero".
    """

    def __init__(self, per_call=0.5):
        self.now = 0.0
        self.per_call = per_call

    def __call__(self):
        return self.now

    def tick(self):
        self.now += self.per_call


def call_runtime(clock=None, fail_from=None, short_by=0):
    """A naming.BatchVlm answering the measurement's question by the frame's WIDTH.

    The width is the identity of the frame, as in `batch_runtime` above. `fail_from` is
    the call size at which generate dies (no VRAM for that many at once); `short_by` is
    how many answers the model swallows — the failure whose alignment can never be
    guessed back, and which therefore has to be counted rather than interpreted.
    """
    def prepare_batch(groups, prompt):
        return [group[0].width for group in groups]

    def generate_batch(prepared, max_new_tokens):
        if clock is not None:
            clock.tick()
        if fail_from is not None and len(prepared) >= fail_from:
            raise RuntimeError("CUDA out of memory")
        answers = [CALL_ANSWERS[mark] for mark in prepared]
        return answers[:len(answers) - short_by] if short_by else answers

    return naming.BatchVlm(
        prepare=lambda frames, prompt: prepare_batch([frames], prompt),
        generate=lambda prepared, tokens: generate_batch(prepared, tokens)[0],
        prepare_batch=prepare_batch, generate_batch=generate_batch)


class TestSecondsPerImage(unittest.TestCase):
    """Test 1: the arithmetic of the answer, on planted timings."""

    def test_seconds_per_image_divides_by_the_images_really_shown(self):
        stats = speed.CallStats(batch=4, seconds=(4.0, 2.0), images=(4, 2), asked=6,
                                parsed=6)
        self.assertEqual((stats.calls, stats.total_images), (2, 6))
        self.assertAlmostEqual(stats.total_sec, 6.0)
        self.assertAlmostEqual(stats.sec_per_image, 1.0)
        self.assertAlmostEqual(stats.parsed_share, 1.0)

    def test_median_mean_min_and_max_are_per_call(self):
        stats = speed.CallStats(batch=2, seconds=(1.0, 2.0, 6.0), images=(2, 2, 2))
        self.assertEqual((stats.median_sec, stats.min_sec, stats.max_sec),
                         (2.0, 1.0, 6.0))
        self.assertAlmostEqual(stats.mean_sec, 3.0)

    def test_a_short_last_call_is_not_counted_as_a_full_one(self):
        """Eight asked for, five shown: the rate divides by five, not by eight."""
        stats = speed.CallStats(batch=8, seconds=(10.0,), images=(5,), asked=5, parsed=5)
        self.assertAlmostEqual(stats.sec_per_image, 2.0)

    def test_an_empty_pass_divides_by_nothing(self):
        stats = speed.CallStats(batch=4)
        self.assertEqual(
            (stats.calls, stats.sec_per_image, stats.median_sec, stats.mean_sec,
             stats.min_sec, stats.max_sec, stats.parsed_share),
            (0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    def test_the_speedup_is_measured_per_image_and_not_per_call(self):
        """Four calls of a second against one call of four seconds is x2, not x4."""
        base = speed.CallStats(batch=1, seconds=(1.0,) * 4, images=(1,) * 4)
        eight = speed.CallStats(batch=8, seconds=(4.0,), images=(8,))
        self.assertAlmostEqual(speed.call_speedup(base, eight), 2.0)

    def test_a_size_that_measured_nothing_has_no_speedup(self):
        base = speed.CallStats(batch=1, seconds=(1.0,), images=(1,))
        self.assertEqual(speed.call_speedup(base, speed.CallStats(batch=8)), 0.0)
        self.assertEqual(speed.call_speedup(speed.CallStats(batch=1), base), 0.0)


class TestCallSizesComeFromTheArgument(unittest.TestCase):
    """Test 2: the sizes are an argument, and the default grid is 1 2 4 8."""

    def parse(self, argv):
        return speed.build_parser().parse_args(argv)

    def test_the_default_sizes_are_one_two_four_eight(self):
        self.assertEqual(speed.DEFAULT_BATCHES, (1, 2, 4, 8))
        self.assertEqual(self.parse([]).batch, [1, 2, 4, 8])

    def test_the_sizes_are_taken_from_the_argument(self):
        args = self.parse(["--per-call", "--batch", "1", "3", "16"])
        self.assertEqual(args.batch, [1, 3, 16])
        self.assertTrue(args.per_call)

    def test_the_per_call_measurement_is_off_unless_asked_for(self):
        self.assertFalse(self.parse([]).per_call)


class TestMeasureCalls(FramesOnDisk):
    """The pass itself: one generate per chunk, and what came back out of it."""

    def test_every_chunk_is_one_call_and_the_answers_are_counted(self):
        clock = Clock(0.5)
        stats = speed.measure_calls(call_runtime(clock), self.frames([20, 21, 22, 25]),
                                    2, 896, clock=clock)
        self.assertEqual((stats.calls, stats.total_images, stats.asked), (2, 4, 4))
        self.assertEqual(stats.seconds, (0.5, 0.5))
        self.assertAlmostEqual(stats.sec_per_image, 0.25)
        self.assertEqual((stats.parsed, stats.failed, stats.skipped), (4, 0, 0))

    def test_the_whole_sample_is_one_call_when_it_fits_in_the_size(self):
        clock = Clock(2.0)
        stats = speed.measure_calls(call_runtime(clock), self.frames([20, 21]), 8, 896,
                                    clock=clock)
        self.assertEqual((stats.calls, stats.total_images), (1, 2))
        self.assertAlmostEqual(stats.sec_per_image, 1.0)

    def test_the_size_is_the_only_thing_that_changes_the_answers(self):
        """The property the whole comparison rests on: same frames, same words back."""
        paths = self.frames([20, 21, 22, 23, 25])
        for size in (1, 2, 4, 8):
            with self.subTest(size=size):
                clock = Clock()
                stats = speed.measure_calls(call_runtime(clock), paths, size, 896,
                                            clock=clock)
                self.assertEqual((stats.asked, stats.parsed, stats.failed), (5, 5, 0))


class TestAFrameThatWouldNotDecode(FramesOnDisk):
    """Test 3: a frame that does not decode is skipped and breaks nothing."""

    def paths_with_two_bad_frames(self):
        paths = self.frames([20, 21, 22])
        broken = Path(self.tmp.name) / "broken.jpg"
        broken.write_bytes(b"not an image at all")
        paths.insert(1, str(broken))
        paths.insert(3, str(Path(self.tmp.name) / "gone.jpg"))
        return paths

    def test_the_call_goes_on_without_it(self):
        clock = Clock()
        stats = speed.measure_calls(call_runtime(clock), self.paths_with_two_bad_frames(),
                                    5, 896, clock=clock)
        self.assertEqual(stats.skipped, 2)
        self.assertEqual((stats.calls, stats.total_images, stats.asked), (1, 3, 3))
        self.assertEqual((stats.parsed, stats.failed), (3, 0))

    def test_the_skipped_frames_do_not_inflate_the_rate(self):
        """Three images shown in one call of a second is 0.33 s per image, not 0.20."""
        clock = Clock(1.0)
        stats = speed.measure_calls(call_runtime(clock), self.paths_with_two_bad_frames(),
                                    5, 896, clock=clock)
        self.assertAlmostEqual(stats.sec_per_image, 1.0 / 3.0)

    def test_a_chunk_with_nothing_readable_makes_no_call_at_all(self):
        gone = [str(Path(self.tmp.name) / f"gone_{i}.jpg") for i in range(2)]
        clock = Clock()
        stats = speed.measure_calls(call_runtime(clock), gone, 2, 896, clock=clock)
        self.assertEqual((stats.calls, stats.skipped, stats.asked), (0, 2, 0))
        self.assertEqual(stats.sec_per_image, 0.0)

    def test_the_report_says_how_many_were_lost(self):
        clock = Clock()
        stats = speed.measure_calls(call_runtime(clock), self.paths_with_two_bad_frames(),
                                    5, 896, clock=clock)
        self.assertIn("пропущены", speed.format_call_table([stats]))


class TestUnreadableAnswersAreCounted(FramesOnDisk):
    """Test 5: speed bought with rubbish is not speed, so the rubbish is in the table."""

    def test_an_answer_that_does_not_parse_is_counted_and_printed(self):
        clock = Clock()
        stats = speed.measure_calls(call_runtime(clock), self.frames([20, 24, 26]), 4,
                                    896, clock=clock)
        self.assertEqual((stats.asked, stats.parsed), (3, 1))
        self.assertAlmostEqual(stats.parsed_share, 1 / 3)
        self.assertIn("1/3", speed.format_call_table([stats]))

    def test_a_call_that_raises_costs_its_answers_and_says_so(self):
        clock = Clock()
        stats = speed.measure_calls(call_runtime(clock, fail_from=4),
                                    self.frames([20, 21, 22, 25]), 4, 896, clock=clock)
        self.assertEqual((stats.calls, stats.asked, stats.failed, stats.parsed),
                         (1, 4, 1, 0))
        self.assertIn("не ответило", speed.format_call_table([stats]))

    def test_a_model_that_answers_fewer_times_than_asked_is_a_failure(self):
        """The alignment is unknowable, so nothing about that call may be believed."""
        clock = Clock()
        stats = speed.measure_calls(call_runtime(clock, short_by=1),
                                    self.frames([20, 21]), 2, 896, clock=clock)
        self.assertEqual((stats.failed, stats.parsed, stats.asked), (1, 0, 2))

    def test_a_failed_call_still_costs_its_seconds(self):
        clock = Clock(0.5)
        stats = speed.measure_calls(call_runtime(clock, fail_from=2),
                                    self.frames([20, 21]), 2, 896, clock=clock)
        self.assertEqual(stats.seconds, (0.5,))

    def test_the_words_are_read_the_way_the_stage_reads_its_own(self):
        self.assertEqual(speed.parse_call_answer("Indoor."), "indoor")
        self.assertEqual(speed.parse_call_answer("I think it is outdoor"), "outdoor")
        self.assertEqual(speed.parse_call_answer("unclear"), "unclear")
        self.assertIsNone(speed.parse_call_answer("hard to say, honestly"))
        self.assertIsNone(speed.parse_call_answer(""))

    def test_the_measurement_asks_its_own_question_not_the_stages(self):
        """The brief's boundary, stated as a fact: no stage prompt is re-timed here."""
        from sorta import junk

        self.assertNotIn(speed._CALL_PROMPT, (junk._VLM_PROMPT, junk._PET_VLM_PROMPT))
        for word in speed._CALL_ANSWERS:
            self.assertNotIn(word, junk._VLM_PROMPT)


class TestTheIndexIsOnlyRead(unittest.TestCase):
    """Test 4: the tool measures — it must not be able to write to the collection."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = Path(self.tmp.name) / "test.db"
        conn = connect(self.db)
        self.addCleanup(conn.close)
        for i in range(3):
            frame = Path(self.tmp.name) / f"f_{i}.jpg"
            Image.new("RGB", (20, 4), (10, 100, 200)).save(frame, "JPEG")
            conn.execute(
                """INSERT INTO files (path, size, mtime, ext, media_type, indexed_at)
                   VALUES (?, 1, 0, 'jpg', 'photo', '2026-01-01')""", (str(frame),))
        conn.commit()

    def test_the_index_is_opened_read_only(self):
        opened = []
        original = speed.sqlite3.connect

        def spy(target, *args, **kwargs):
            opened.append(str(target))
            return original(target, *args, **kwargs)

        speed.sqlite3.connect = spy
        self.addCleanup(setattr, speed.sqlite3, "connect", original)
        speed.sample_paths(str(self.db), 3, seed=1)
        self.assertTrue(opened)
        for target in opened:
            self.assertIn("mode=ro", target)

    def test_a_whole_pass_leaves_the_database_byte_for_byte(self):
        before = self.db.read_bytes()
        paths, _origin = speed.sample_paths(str(self.db), 3, seed=1)
        self.assertEqual(len(paths), 3)
        clock = Clock()
        stats = speed.measure_calls(call_runtime(clock), paths, 2, 896, clock=clock)
        self.assertEqual(stats.asked, 3)
        self.assertEqual(self.db.read_bytes(), before)


class TestCallReport(unittest.TestCase):
    """The table the acceptance criterion names, and the line printed under it."""

    def stats(self, batch, sec_per_call, calls=4, parsed=None, asked=None):
        asked = batch * calls if asked is None else asked
        return speed.CallStats(
            batch=batch, seconds=(sec_per_call,) * calls, images=(batch,) * calls,
            asked=asked, parsed=asked if parsed is None else parsed,
            cpu_cores=0.84, gpu_util_pct=26.0, peak_vram_mb=7890.0)

    def test_the_table_names_the_four_columns_of_the_brief(self):
        table = speed.format_call_table([self.stats(1, 0.78), self.stats(8, 5.0)])
        for column in ("картинок", "медиана", "с/изобр", "разобрано"):
            self.assertIn(column, table)

    def test_the_speedup_column_compares_the_cost_of_an_image(self):
        """0.78 s for one against 5.00 s for eight — 0.625 s an image, x1.25."""
        table = speed.format_call_table([self.stats(1, 0.78), self.stats(8, 5.0)])
        self.assertIn("x1.00", table)
        self.assertIn("x1.25", table)

    def test_the_load_of_the_machine_is_in_the_table(self):
        """The old verdict was made of these two numbers; a new one needs them too."""
        table = speed.format_call_table([self.stats(1, 0.78)])
        self.assertIn("26%", table)
        self.assertIn("0.84", table)

    def test_a_cheaper_call_that_still_answers_is_a_win(self):
        win, why = speed.call_outcome([self.stats(1, 0.78), self.stats(8, 5.0)])
        self.assertTrue(win)
        self.assertIn("выигрыш есть", why)
        self.assertIn("8 изображений", why)

    def test_a_speedup_below_the_threshold_is_not_a_win(self):
        """0.78 against 0.75 an image is x1.04 — under the x1.15 the brief asks for."""
        win, why = speed.call_outcome([self.stats(1, 0.78), self.stats(8, 6.0)])
        self.assertFalse(win)
        self.assertIn("F105", why)

    def test_speed_bought_with_unreadable_answers_is_not_a_win(self):
        confused = self.stats(8, 3.0, parsed=8)
        self.assertGreater(speed.call_speedup(self.stats(1, 0.78), confused),
                           speed.MIN_SPEEDUP)
        win, _why = speed.call_outcome([self.stats(1, 0.78), confused])
        self.assertFalse(win)

    def test_one_frame_of_noise_does_not_reject_a_faster_call(self):
        """The tolerance, applied: a share that moved by less than a point is noise."""
        almost = self.stats(8, 5.0, calls=40, parsed=319)
        win, _why = speed.call_outcome([self.stats(1, 0.78), almost])
        self.assertTrue(win)

    def test_one_size_alone_decides_nothing(self):
        win, why = speed.call_outcome([self.stats(1, 0.78)])
        self.assertFalse(win)
        self.assertIn("сравнивать не с чем", why)

    def test_the_outcome_line_is_printed_either_way(self):
        for other in (self.stats(8, 5.0), self.stats(8, 6.0)):
            with self.subTest(seconds=other.median_sec):
                text = speed.format_call_outcome([self.stats(1, 0.78), other])
                self.assertTrue(text.startswith("ИТОГ"))

    def test_the_report_identifies_no_frame(self):
        stats = [self.stats(1, 0.78), self.stats(4, 2.0)]
        text = "\n".join([speed.format_call_table(stats),
                          speed.format_call_outcome(stats)])
        for leak in ("/photos", ".jpg", "file_id", "IMG_"):
            self.assertNotIn(leak, text)


class TestPerCallMain(unittest.TestCase):
    """`--per-call` end to end: the table is printed, and the sizes reach the model."""

    def patch(self, target, name, value):
        original = getattr(target, name)
        setattr(target, name, value)
        self.addCleanup(setattr, target, name, original)

    def stats(self, batch, sec_per_call):
        return speed.CallStats(batch=batch, seconds=(sec_per_call,) * 4,
                               images=(batch,) * 4, asked=batch * 4, parsed=batch * 4)

    def run_main(self, measure, argv=("--per-call", "--sample", "4")):
        cfg = type("Cfg", (), {"database": ":memory:", "vlm": type(
            "Vlm", (), {"model": "Qwen/test", "workers": 2, "max_edge": 896})()})()
        self.patch(speed, "load_config", lambda _path: cfg)
        self.patch(speed, "sample_paths", lambda *_a: (["/photos/a.jpg"], "тест"))
        self.patch(speed, "measure_per_call", measure)
        self.patch(sys, "argv", ["measure_vlm_speed.py", *argv])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = speed.main()
        return code, out.getvalue()

    def test_the_table_and_the_outcome_are_printed(self):
        code, out = self.run_main(lambda *_a: [self.stats(1, 0.78), self.stats(8, 5.0)])
        self.assertEqual(code, 0)
        self.assertIn("с/изобр", out)
        self.assertIn("ИТОГ", out)

    def test_a_measurement_that_made_no_call_reports_failure(self):
        code, out = self.run_main(lambda *_a: [speed.CallStats(batch=1)])
        self.assertEqual(code, 1)
        self.assertIn("замер не состоялся", out)

    def test_the_sizes_reach_the_measurement_sorted_and_without_repeats(self):
        seen: dict[str, list[int]] = {}

        def measure(_model, _paths, batches, _max_edge):
            seen["batches"] = list(batches)
            return [self.stats(1, 1.0)]

        code, _out = self.run_main(
            measure, argv=("--per-call", "--batch", "8", "1", "8", "2"))
        self.assertEqual(code, 0)
        self.assertEqual(seen["batches"], [1, 2, 8])

    def test_a_size_below_one_is_refused_before_the_model_is_loaded(self):
        with self.assertRaises(SystemExit):
            self.run_main(lambda *_a: [], argv=("--per-call", "--batch", "0"))


if __name__ == "__main__":
    unittest.main()
