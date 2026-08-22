"""F243 phase 0: the rig that measures the CPU tier is itself measured here.

The brief's whole point is that the rig has to PROVE what it measured, so these tests are
about the proof at least as much as about the arithmetic:

* a stack that cannot be shown to be CPU-only makes the rig refuse — and refuse before it
  has created a database or written a report, because a half-run leaves a file somebody
  will read later;
* a production index is not somewhere a measurement may write, and an exception to that
  is a flag somebody typed rather than a default;
* a synthetic collection goes through every stage end to end, and every requested stage
  reaches the report — a stage that produced no timing says so instead of vanishing;
* the timings come from the run log of THIS run: an older line for the same stage, from
  the same build, must not be picked up.

No GPU, no models, no photographs of anybody: a few dozen generated JPEGs in a temporary
directory and two faked modules.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from PIL import Image

from sorta import __version__

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "measure_cpu_tier.py"


def _load_script():
    """Import scripts/measure_cpu_tier.py — it is a script, not a package module."""
    spec = importlib.util.spec_from_file_location("measure_cpu_tier", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rig = _load_script()


def fake_torch(version="2.13.0+cpu", built=None, available=False, raises=False):
    """A torch that answers about CUDA the way the test needs it to."""
    def is_available():
        if raises:
            raise RuntimeError("CUDA driver initialisation failed")
        return available

    return SimpleNamespace(__version__=version, version=SimpleNamespace(cuda=built),
                           cuda=SimpleNamespace(is_available=is_available))


def fake_onnx(*providers):
    return SimpleNamespace(get_available_providers=lambda: list(providers))


CPU_ONLY_STACK = {"torch": fake_torch(), "onnxruntime": fake_onnx("CPUExecutionProvider")}


def make_jpeg(path: Path, seed: int) -> None:
    """One synthetic frame — distinct per seed, and over the 5 KB the indexer's default
    `min_file_size_kb` refuses to look at, which a flat 64x64 colour is not."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.effect_mandelbrot((320, 320), (-2, -1.5, 1, 1.5), 20 + seed).convert("RGB").save(
        path, "JPEG")


class TestItRefusesWhatItCannotProve(unittest.TestCase):
    """Criterion 1: faked CUDA availability and the rig does not start, and says why."""

    def test_a_cpu_stack_proves_itself(self):
        checks = rig.cpu_only_checks(fake_torch(), fake_onnx("CPUExecutionProvider"))
        self.assertEqual(rig.refusals(checks), [])
        self.assertEqual([c.name for c in checks],
                         [rig.TORCH_RUNTIME, rig.TORCH_BUILD, rig.ONNX_PROVIDERS])

    def test_torch_seeing_a_card_is_a_refusal(self):
        checks = rig.cpu_only_checks(fake_torch(available=True), fake_onnx())
        refused = rig.refusals(checks)
        self.assertEqual([c.name for c in refused], [rig.TORCH_RUNTIME])
        self.assertIn("True", refused[0].detail)

    def test_a_cuda_build_with_no_card_is_still_a_refusal(self):
        """The report is about the install PROFILE, not about the weather on the card."""
        checks = rig.cpu_only_checks(fake_torch(version="2.13.0+cu130", built="13.0"),
                                     fake_onnx())
        self.assertEqual([c.name for c in rig.refusals(checks)], [rig.TORCH_BUILD])

    def test_a_cuda_provider_on_offer_is_a_refusal(self):
        checks = rig.cpu_only_checks(
            fake_torch(), fake_onnx("CUDAExecutionProvider", "CPUExecutionProvider"))
        refused = rig.refusals(checks)
        self.assertEqual([c.name for c in refused], [rig.ONNX_PROVIDERS])
        self.assertIn("CUDAExecutionProvider", refused[0].detail)

    def test_a_question_that_raises_is_not_taken_as_a_no(self):
        checks = rig.cpu_only_checks(fake_torch(raises=True), fake_onnx())
        refused = rig.refusals(checks)
        self.assertEqual([c.name for c in refused], [rig.TORCH_RUNTIME])
        self.assertIn("RuntimeError", refused[0].detail)

    def test_a_machine_without_either_package_has_no_cuda_to_hide(self):
        checks = rig.cpu_only_checks(None, None)
        self.assertEqual(rig.refusals(checks), [])
        self.assertTrue(all("not installed" in c.detail for c in checks))

    def test_every_answer_is_written_down_and_not_only_the_bad_ones(self):
        """Criterion 2: the report has to be readable by somebody who was not here."""
        checks = rig.cpu_only_checks(fake_torch(), fake_onnx("CPUExecutionProvider"))
        self.assertEqual([c.as_json()["check"] for c in checks],
                         [rig.TORCH_RUNTIME, rig.TORCH_BUILD, rig.ONNX_PROVIDERS])
        self.assertTrue(all(c.as_json()["cpu_only"] for c in checks))


class TestARefusalLeavesNothingBehind(unittest.TestCase):
    """A refusal that had already created files would leave somebody a number to find."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def run_rig(self, stack):
        args = ["--config", str(self.root / "config.yaml"), "--src", str(self.root),
                "--db", str(self.root / "measure.db"), "--out", str(self.root / "out.json")]
        with mock.patch.dict(sys.modules, stack), mock.patch.dict(os.environ, {}):
            return rig.main(args)

    def test_a_gpu_stack_stops_the_run(self):
        code = self.run_rig({"torch": fake_torch(available=True),
                             "onnxruntime": fake_onnx("CUDAExecutionProvider")})
        self.assertEqual(code, 2)

    def test_no_report_and_no_database_are_created(self):
        self.run_rig({"torch": fake_torch(available=True), "onnxruntime": fake_onnx()})
        self.assertFalse((self.root / "out.json").exists())
        self.assertFalse((self.root / "measure.db").exists())

    def test_a_report_path_that_cannot_be_written_stops_the_run_at_the_start(self):
        """Hours of run and then a typo in `--out` is not somewhere to find that out."""
        args = ["--config", str(self.root / "config.yaml"), "--src", str(self.root),
                "--db", str(self.root / "measure.db"),
                "--out", str(self.root / "nowhere" / "out.json")]
        with mock.patch.dict(sys.modules, CPU_ONLY_STACK), self.assertRaises(SystemExit):
            rig.main(args)
        self.assertFalse((self.root / "measure.db").exists())


class TestTheMeasurementGetsADatabaseOfItsOwn(unittest.TestCase):
    """Criterion 3: not photos.db, not sorta.db, and not what config.yaml points at."""

    def test_the_two_production_names_are_known(self):
        self.assertTrue(rig.is_production_db(Path("photos.db")))
        self.assertTrue(rig.is_production_db(Path("/data/sorta.db")))
        self.assertTrue(rig.is_production_db(Path("PHOTOS.DB")))

    def test_the_default_is_neither_of_them(self):
        self.assertFalse(rig.is_production_db(Path(rig.DEFAULT_DB)))

    def test_the_configured_index_counts_whatever_it_is_called(self):
        self.assertTrue(rig.is_production_db(Path("archive.db"), Path("archive.db")))
        self.assertFalse(rig.is_production_db(Path("archive.db"), Path("other.db")))

    def test_pointing_at_one_needs_the_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = ["--config", str(root / "config.yaml"), "--src", str(root),
                    "--db", str(root / "photos.db"), "--out", str(root / "out.json")]
            with mock.patch.dict(sys.modules, CPU_ONLY_STACK):
                self.assertEqual(rig.main(args), 2)
            self.assertFalse((root / "photos.db").exists())


class TestASyntheticCollectionGoesThroughWhole(unittest.TestCase):
    """Criterion 4: dozens of files in a temporary directory, every field in its place."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.src = cls.root / "collection"
        cls.files = 24
        for i in range(cls.files):
            make_jpeg(cls.src / f"IMG_2019070{i % 8}_1234{i:02d}.jpg", i)
        cls.out = cls.root / "report.json"
        cls.log = cls.root / "run.log"
        args = ["--config", str(cls.root / "config.yaml"), "--src", str(cls.src),
                "--db", str(cls.root / "measure.db"), "--out", str(cls.out),
                "--log", str(cls.log)]
        with mock.patch.dict(sys.modules, CPU_ONLY_STACK), \
                mock.patch.dict(os.environ, {}):
            cls.code = rig.main(args)
        cls.report = json.loads(cls.out.read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_the_run_finished(self):
        self.assertEqual(self.code, 0)

    def test_every_requested_stage_reached_the_report(self):
        self.assertEqual([s["name"] for s in self.report["stages"]],
                         list(rig.DEFAULT_STAGES))

    def test_every_stage_counted_its_population_and_explained_itself(self):
        """Seconds are not asserted per stage: over two dozen files `geo` can finish
        inside the log's millisecond, and the report then owes a note instead of a
        number. What it may never owe is silence."""
        for stage in self.report["stages"]:
            with self.subTest(stage=stage["name"]):
                self.assertGreater(stage["processed"], 0)
                if stage["status"] == rig.MEASURED:
                    self.assertGreater(stage["seconds"], 0)
                else:
                    self.assertEqual(stage["status"], rig.NO_TIMING)
                    self.assertTrue(stage["note"])

    def test_the_stage_that_does_the_real_work_was_timed(self):
        index = next(s for s in self.report["stages"] if s["name"] == "index")
        self.assertEqual(index["status"], rig.MEASURED)
        self.assertGreater(index["seconds"], 0)
        self.assertEqual(index["processed"], self.files)

    def test_the_total_is_the_sum_of_what_was_measured(self):
        measured = [s["seconds"] for s in self.report["stages"] if s["seconds"] is not None]
        self.assertAlmostEqual(self.report["measured_seconds_total"],
                               round(sum(measured), 3), places=2)

    def test_the_collection_is_the_one_that_was_measured(self):
        collection = self.report["collection"]
        self.assertEqual(collection["files"], self.files)
        self.assertEqual(collection["errors"], 0)
        self.assertEqual(collection["sources"], [str(self.src)])
        self.assertGreaterEqual(collection["workers"], 1)

    def test_the_proof_travels_with_the_numbers(self):
        self.assertTrue(self.report["cpu_only"])
        self.assertEqual([c["check"] for c in self.report["proof"]],
                         [rig.TORCH_RUNTIME, rig.TORCH_BUILD, rig.ONNX_PROVIDERS])

    def test_the_machine_is_named(self):
        machine = self.report["machine"]
        self.assertTrue(machine["processor"])
        self.assertGreaterEqual(machine["cores_logical"], 1)
        self.assertTrue(machine["platform"])
        self.assertEqual(machine["executable"], sys.executable)

    def test_the_versions_are_named(self):
        packages = self.report["packages"]
        self.assertEqual(packages["sorta"], __version__)
        self.assertIn("numpy", packages)
        self.assertIn("onnxruntime-gpu", packages)

    def test_what_was_deliberately_not_measured_is_named_too(self):
        self.assertEqual(self.report["not_measured"]["stages"], list(rig.OUT_OF_SCOPE))
        self.assertIn("faces", self.report["not_measured"]["stages"])
        self.assertTrue(self.report["not_measured"]["why"])

    def test_the_timings_really_came_from_the_run_log(self):
        written = self.log.read_text(encoding="utf-8", errors="replace")
        self.assertIn("stage=index elapsed=", written)
        self.assertIn(f"  sorta: {__version__}", written)

    def test_the_measurement_wrote_only_into_its_own_database(self):
        self.assertTrue((self.root / "measure.db").exists())
        self.assertFalse((self.root / "sorta.db").exists())
        self.assertFalse((self.src / "measure.db").exists())

    def test_the_console_form_says_the_same_thing(self):
        text = rig.format_report(self.report)
        for name in rig.DEFAULT_STAGES:
            self.assertIn(name, text)
        self.assertIn("not measured:", text)


class TestNoStageIsSkippedQuietly(unittest.TestCase):
    """Criterion 5: a stage with no timing is a row in the report, not a gap in it."""

    def test_a_stage_the_log_never_mentioned_is_still_reported(self):
        outcomes = rig.collect_outcomes({}, {"index": 40}, ["index", "geo"])
        self.assertEqual([o.name for o in outcomes], ["index", "geo"])
        self.assertTrue(all(o.status == rig.NO_TIMING for o in outcomes))
        self.assertTrue(all(o.note for o in outcomes))

    def test_a_stage_that_had_nothing_to_do_says_that_and_not_something_darker(self):
        outcomes = rig.collect_outcomes({}, {"phash": 0}, ["phash"])
        self.assertEqual(outcomes[0].processed, 0)
        self.assertIn("nothing to do", outcomes[0].note)

    def test_a_stage_that_ran_and_left_no_line_sends_the_reader_to_the_log(self):
        self.assertIn("read the log", rig.no_timing_note(500))

    def test_a_missing_timing_does_not_become_zero_seconds_in_the_total(self):
        outcomes = [rig.StageOutcome("index", rig.MEASURED, 12.5, 40),
                    rig.StageOutcome("geo", rig.NO_TIMING, None, 0, "x")]
        self.assertEqual(rig.measured_seconds(outcomes), 12.5)

    def test_a_measured_stage_carries_the_logs_own_numbers(self):
        found = {"stage=index": rig.runlog.Measurement(
            unit="stage=index", seconds=61.2345, processed=38485,
            at=datetime.now(), build=__version__)}
        outcome = rig.collect_outcomes(found, {"index": 38485}, ["index"])[0]
        self.assertEqual((outcome.status, outcome.seconds, outcome.processed),
                         (rig.MEASURED, 61.234, 38485))


class TestTheTimingsBelongToThisRun(unittest.TestCase):
    """An append-only log holds other runs; borrowing their seconds is the same defect."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.log = Path(self.tmp.name) / "run.log"
        self.addCleanup(self.tmp.cleanup)

    def write_log(self, *stamps):
        lines = [f"  sorta: {__version__}"]
        for at, stage, seconds in stamps:
            lines.append(f"{at.strftime('%Y-%m-%dT%H:%M:%S')}.000 INFO     sorta.runlog "
                         f"[MainThread] stage={stage} elapsed={seconds:.3f} processed=100")
        self.log.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_a_line_from_before_the_run_is_not_this_runs_timing(self):
        started = datetime.now()
        self.write_log((started - timedelta(hours=2), "index", 11.0))
        self.assertEqual(rig.measurements_since(self.log, started), {})

    def test_a_line_from_this_run_is_kept(self):
        started = datetime.now()
        self.write_log((started + timedelta(seconds=5), "index", 11.0))
        found = rig.measurements_since(self.log, started)
        self.assertEqual(found["stage=index"].seconds, 11.0)

    def test_a_stage_the_second_before_the_start_is_still_kept(self):
        """The log's stamps have a one-second resolution — the guard rounds the same way."""
        started = datetime.now().replace(microsecond=500000)
        self.write_log((started.replace(microsecond=0), "index", 11.0))
        self.assertIn("stage=index", rig.measurements_since(self.log, started))


class TestWhichStagesRun(unittest.TestCase):
    """The base tier and nothing heavier, in the order the pipeline needs."""

    def test_the_default_is_the_base_tier(self):
        self.assertEqual(rig.DEFAULT_STAGES, ("index", "geo", "phash", "dupes"))
        self.assertEqual([s.name for s in rig.select_stages(",".join(rig.DEFAULT_STAGES))],
                         list(rig.DEFAULT_STAGES))

    def test_no_heavy_stage_can_be_asked_for(self):
        for heavy in rig.OUT_OF_SCOPE:
            with self.subTest(stage=heavy), self.assertRaises(SystemExit):
                rig.select_stages(heavy)

    def test_the_order_is_the_pipelines_and_not_the_flags(self):
        """phash after index and dupes after phash is a dependency, not a preference."""
        self.assertEqual([s.name for s in rig.select_stages("dupes,phash,index")],
                         ["index", "phash", "dupes"])

    def test_a_stage_named_twice_runs_once(self):
        self.assertEqual([s.name for s in rig.select_stages("geo geo")], ["geo"])

    def test_an_empty_selection_is_an_error_and_not_an_empty_report(self):
        with self.assertRaises(SystemExit):
            rig.select_stages(" ")


if __name__ == "__main__":
    unittest.main()
