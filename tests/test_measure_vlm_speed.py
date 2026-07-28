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
"""
from __future__ import annotations

import importlib.util
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


if __name__ == "__main__":
    unittest.main()
