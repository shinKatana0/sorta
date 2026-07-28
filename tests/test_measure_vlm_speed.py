"""F101: the VLM speed measurement — everything about it that is not the model.

The script exists to answer two questions the feature is accepted on: how much faster
the pass got, and whether a single verdict moved. Both answers are arithmetic over
per-frame aggregates, so both are testable here with a fake classifier — no
transformers, no GPU, no photo.

Two of these tests are about the brief rather than about code:

* a mismatch of even one label must make the report say STOP and the process exit
  non-zero — "ну почти" was ruled out in writing before the measurement existed;
* nothing the script prints may identify a frame — a table about documents must not
  become a list of where the documents are (the rule of measure_ocr_gate.py before it).
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

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


def result(name, labels, frame_ms=(10.0,), wall=1.0, workers=1, fast=True):
    return speed.ModeResult(
        name=name, workers=workers, fast_processor=fast, labels=tuple(labels),
        frame_ms=tuple(frame_ms), wall_sec=wall, cpu_cores=0.8,
        gpu_util_pct=26.0, peak_vram_mb=23100.0)


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
            [result("baseline", labels), result("pipelined", labels)])
        self.assertTrue(ok)
        self.assertIn("совпадение полное", report)
        self.assertNotIn("СТОП", report)

    def test_one_moved_label_is_a_stop(self):
        report, ok = speed.format_verdicts([
            result("baseline", ["document", "product", "product"]),
            result("pipelined", ["document", "product", "personal_photo"]),
        ])
        self.assertFalse(ok)
        self.assertIn("СТОП", report)
        self.assertIn("product -> personal_photo: 1", report)

    def test_a_single_mode_has_nothing_to_compare(self):
        report, ok = speed.format_verdicts([result("baseline", ["document"])])
        self.assertTrue(ok)
        self.assertIn("сравнивать не с чем", report)


class TestReportIdentifiesNothing(unittest.TestCase):
    """Privacy: the report is aggregates — no path, no basename, no file id."""

    def test_no_frame_identity_reaches_the_output(self):
        results = [result("baseline", ["document", "product"]),
                   result("pipelined", ["product", "product"])]
        text = speed.format_table(results) + "\n" + speed.format_verdicts(results)[0]
        for leak in ("/photos", ".jpg", "file_id", "IMG_"):
            self.assertNotIn(leak, text)


class TestFormatTable(unittest.TestCase):
    def test_speedup_is_measured_against_the_first_mode(self):
        base = result("baseline", ["document"] * 10, wall=10.0, workers=1, fast=False)
        fast = result("pipelined", ["document"] * 10, wall=2.5, workers=4)
        table = speed.format_table([base, fast])
        self.assertIn("x1.00", table)
        self.assertIn("x4.00", table)
        self.assertIn("медленный", table)
        self.assertIn("быстрый", table)


class TestRunMode(unittest.TestCase):
    """run_mode goes through the pipeline's own _vlm_labels — not a copy of it."""

    def classifier(self, labels, fail=frozenset()):
        def prepare(path):
            return PreparedFrame(inputs=Path(path).name)

        def classify_prepared(prepared):
            if prepared.inputs in fail:
                raise RuntimeError("CUDA error")
            return labels[prepared.inputs]

        return SplitVlmClassifier(prepare=prepare, classify_prepared=classify_prepared)

    def test_labels_come_back_in_input_order_with_a_timing_per_frame(self):
        paths = [f"/photos/f_{i}.jpg" for i in range(6)]
        labels = {f"f_{i}.jpg": ("document" if i % 2 else "product") for i in range(6)}
        r = speed.run_mode("pipelined", self.classifier(labels), paths, 3, True)
        self.assertEqual(list(r.labels),
                         ["product", "document"] * 3)
        self.assertEqual(len(r.frame_ms), 6)
        self.assertGreater(r.wall_sec, 0.0)

    def test_a_failed_frame_is_recorded_not_raised(self):
        paths = ["/photos/f_0.jpg", "/photos/f_1.jpg"]
        labels = {"f_0.jpg": "product", "f_1.jpg": "document"}
        r = speed.run_mode("pipelined", self.classifier(labels, fail={"f_1.jpg"}),
                           paths, 2, True)
        self.assertEqual(list(r.labels), ["product", "ERROR"])


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


if __name__ == "__main__":
    unittest.main()
