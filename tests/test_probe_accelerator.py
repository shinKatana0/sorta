"""F214: the probe that runs where the hardware is — its reporting, not its models.

The script exists because a faked accelerator proves nothing about a real one, so what
can be checked here is everything EXCEPT the weights: that a tier which does not fit is
reported as a line rather than a traceback, that `--strict` is what turns a miss into a
failure, and that the machine's own answer is printed before any tier is attempted.

The tiers themselves (buffalo_l, CLIP ViT-L) download gigabytes and need a device this
machine does not have — they are driven through the same `attempt` seam with stand-ins,
which is the seam that decides whether a runner goes red for the right reason.
"""
from __future__ import annotations

import importlib.util
import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "probe_accelerator.py"


def _load_script():
    """Import scripts/probe_accelerator.py — it is a script, not a package module."""
    spec = importlib.util.spec_from_file_location("probe_accelerator", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe = _load_script()


class ATierThatDoesNotFitIsAnAnswerTest(unittest.TestCase):
    """The whole point of the script: "no room for this" is output, not a crash."""

    def test_a_raising_tier_is_reported_as_skipped(self):
        def no_room() -> tuple[str, str]:
            raise OSError("No space left on device")

        status, detail = probe.attempt(no_room)
        self.assertEqual(status, probe.SKIPPED)
        self.assertIn("No space left on device", detail)
        self.assertIn("OSError", detail, "the reason has to be nameable afterwards")

    def test_a_tier_that_fits_passes_its_own_words_through(self):
        status, detail = probe.attempt(lambda: (probe.FITTED, "ran on CoreML"))
        self.assertEqual((status, detail), (probe.FITTED, "ran on CoreML"))

    def test_the_line_names_the_tier_and_the_verdict(self):
        line = probe.report("clip (ViT-L-14)", probe.SKIPPED, "no room")
        self.assertIn("clip (ViT-L-14)", line)
        self.assertIn(probe.SKIPPED, line)
        self.assertIn("no room", line)


class WhatTheRunnerPrintsTest(unittest.TestCase):
    """The machine describes itself before anything heavy is attempted."""

    def run_main(self, *argv: str) -> tuple[int, str]:
        out = io.StringIO()
        with redirect_stdout(out):
            code = probe.main(list(argv))
        return code, out.getvalue()

    def test_the_bare_run_reports_the_machine_and_touches_no_tier(self):
        code, output = self.run_main()
        self.assertEqual(code, 0)
        self.assertIn("free disk:", output)
        self.assertIn("torch device:", output)
        self.assertNotIn("tier ", output)

    def test_free_disk_is_a_real_number_for_this_machine(self):
        self.assertGreater(probe.free_gb(str(Path(__file__).parent)), 0.0)

    def test_a_missed_tier_does_not_fail_the_command_by_default(self):
        with mock.patch.object(probe, "probe_faces",
                               side_effect=RuntimeError("no onnxruntime here")):
            code, output = self.run_main("--faces")
        self.assertEqual(code, 0, "reporting a miss is the job, not failing on it")
        self.assertIn(probe.SKIPPED, output)
        self.assertIn("no onnxruntime here", output)

    def test_strict_is_what_turns_a_miss_into_a_failure(self):
        with mock.patch.object(probe, "probe_clip",
                               side_effect=RuntimeError("out of disk")):
            code, output = self.run_main("--clip", "--strict")
        self.assertEqual(code, 1)
        self.assertIn("did not fit", output)

    def test_strict_says_nothing_when_every_asked_tier_fitted(self):
        with mock.patch.object(probe, "probe_faces",
                               return_value=(probe.FITTED, "CoreMLExecutionProvider")):
            code, output = self.run_main("--faces", "--strict")
        self.assertEqual(code, 0)
        self.assertIn(probe.FITTED, output)

    def test_all_asks_for_every_tier_that_has_a_flag(self):
        with mock.patch.object(probe, "probe_faces",
                               return_value=(probe.FITTED, "faces ok")) as faces, \
                mock.patch.object(probe, "probe_clip",
                                  return_value=(probe.FITTED, "clip ok")) as clip:
            code, output = self.run_main("--all")
        self.assertEqual(code, 0)
        self.assertEqual((faces.call_count, clip.call_count), (1, 1))
        self.assertIn("faces ok", output)
        self.assertIn("clip ok", output)

    def test_the_deep_tier_has_no_flag_and_is_not_pretended_to_be_checked(self):
        """7 GB does not fit a runner; a flag for it would read as a check somebody ran."""
        with self.assertRaises(SystemExit):
            with redirect_stdout(io.StringIO()):
                probe.main(["--vlm"])


class ComparingTheDevicesTest(unittest.TestCase):
    """`--clip` skips itself where there is nothing to compare against."""

    def test_a_cpu_only_machine_says_there_is_nothing_to_compare(self):
        with mock.patch.object(probe.accel, "torch_device", return_value="cpu"):
            status, detail = probe.probe_clip()
        self.assertEqual(status, probe.SKIPPED)
        self.assertIn("nothing to compare", detail)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
