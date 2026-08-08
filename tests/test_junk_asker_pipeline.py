"""F206: the animal check and the rescue ask through the deep tier's pipeline.

F165 split the stage in two and the pipeline went with the verdicts: `classify` overlaps
the CPU half of one frame with the GPU half of the previous one (F101), while
`_frame_question` — the animal check and the screen-capture rescue — kept asking one frame
at a time. The run of 2026-08-05 priced that at 0.42 frames/s against the tier's 1.4 over
the same model, i.e. 116 minutes a run over 4 281 frames.

What is asserted here is what a change of SCHEDULE is allowed to do, and it is almost
nothing:

* the verdicts do not move — one worker and four write byte-identical `media_class` and
  `frame_quality` rows, which is the criterion the brief calls stricter than the speed;
* an answer lands on the file it is about, even when the preparations finish in the
  reverse order (a FIFO of futures, not "whatever finished first");
* a frame whose preparation or generation fails keeps its cheap-tier answer and takes no
  neighbour with it, and the progress bar still reaches its total;
* the pass makes ONE preparation per candidate — a pipeline overlaps work, it does not
  double it;
* preparation leaves the caller's thread, generation and every write stay on it, and the
  frames in flight are bounded (the RAM/VRAM argument F101 made: the prepared tensors are
  CPU tensors and at most `2 x workers` of them exist at once).

And the watchdog the brief will not accept the feature without: THE PRICE OF A FRAME,
measured against another phase of the SAME run rather than in seconds. `junk_pets_vlm` and
`junk_rescue_vlm` ask one model one question about one frame, exactly as `junk_vlm` does,
so a phase that costs several times what that one costs is a phase that stopped
overlapping — and that statement survives a slower machine, where an absolute number in
seconds does not. `TestTheGuardHasTeeth` runs the same fixture through an asker WITHOUT
halves (the pre-F206 shape) and asserts the guard fails there, because a watchdog that
cannot fail is not one.

No model is loaded anywhere: every asker below is injected, as everywhere else in the junk
suite.
"""
from __future__ import annotations

import importlib.util
import logging
import re
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

import pytest
from PIL import Image

from sorta import junk, runlog
from sorta.config import Config, FeaturesConfig, VlmConfig, _naming_from
from sorta.db import connect
from sorta.junk import (
    CLASSIFY_PHASE_PETS_VLM,
    CLASSIFY_PHASE_RESCUE_VLM,
    CLASSIFY_PHASE_VLM,
    CLASSIFY_STAGE,
    PET_VLM_REAL,
    PreparedFrame,
    SplitVlmClassifier,
    classify,
    vlm_junk_rescue_asker,
    vlm_pet_asker,
)
from sorta.naming import SplitVlm
from tests.test_junk import NO_OCR
from tests.test_junk_rescue import FakeTextEncoder
from tests.test_junk_vlm_pipeline import ThreadSpyConn
from tests.test_vlm_phase_names import ThreeAskersClassifier

WORKERS = 4
_RUNLOG = "sorta.runlog"
# The phase summary as `runlog` writes it — the two numbers the watchdog is built on.
_PHASE_LINE = re.compile(
    r"^stage=(?P<stage>\S+) phase=(?P<phase>\S+) elapsed=(?P<elapsed>[0-9.]+)"
    r"(?: processed=(?P<processed>\d+))?")

_MEASURE_SCRIPT = (Path(__file__).resolve().parent.parent
                   / "scripts" / "measure_asker_pipeline.py")


def _load_measure_script():
    """Import the acceptance script — it is a script, not a package module."""
    spec = importlib.util.spec_from_file_location("measure_asker_pipeline",
                                                  _MEASURE_SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module   # the dataclasses inside need to find their module
    spec.loader.exec_module(module)
    return module


# The synthetic shape of a frame's cost, and it is the production one in miniature: the
# CPU half (decode + the processor) is the long one and the GPU half is short, which is
# what the overlap exists for (F101 measured ~0.6 s against ~0.19 s on the live run).
PREPARE_SECONDS = 0.008
GENERATE_SECONDS = 0.002
# What the watchdog allows. Serial costs PREPARE+GENERATE a frame where the pipeline costs
# about max(PREPARE / workers, GENERATE) — a factor of four here, threefold on the live
# run — so a limit of two is comfortably clear of scheduling noise and nowhere near the
# regression it is here to catch.
MAX_PHASE_RATIO = 2.0


def frame_names(count: int) -> list[str]:
    """The names an `AskerRun` of that size holds — known without building one."""
    return [f"cat_{i}.jpg" for i in range(count)]


class SplitAsker:
    """One question over one frame with observable halves — what `_frame_question` returns.

    The same double as `FakeSplitVlm` (the deep tier's, F101) for the same reason: what a
    pass does with its threads has to be assertable without a model. `prepare` is the CPU
    half the pipeline moves off the caller's thread, `classify_prepared` the GPU half that
    must stay on it; either can be slowed down (so that completion order differs from
    input order) or made to fail. `max_alive` is how many prepared-but-unanswered frames
    existed at once — the bound the window keeps.
    """

    def __init__(self, answers: dict[str, str], default: str = "none",
                 prepare_seconds: float = 0.0, generate_seconds: float = 0.0,
                 prepare_delay: dict[str, float] | None = None,
                 fail_prepare: frozenset[str] = frozenset(),
                 fail_generate: frozenset[str] = frozenset()) -> None:
        self.answers = answers
        self.default = default
        self.prepare_seconds = prepare_seconds
        self.generate_seconds = generate_seconds
        self.prepare_delay = prepare_delay or {}
        self.fail_prepare = fail_prepare
        self.fail_generate = fail_generate
        self._lock = threading.Lock()
        self.prepare_threads: set[int] = set()
        self.generate_threads: set[int] = set()
        self.prepared: list[str] = []    # in completion order
        self.asked: list[str] = []       # in consumption order
        self._alive = 0
        self.max_alive = 0

    def prepare(self, path: str) -> PreparedFrame:
        name = Path(path).name
        time.sleep(self.prepare_delay.get(name, self.prepare_seconds))
        if name in self.fail_prepare:
            raise RuntimeError(f"decode failed on {name}")
        with self._lock:
            self.prepare_threads.add(threading.get_ident())
            self.prepared.append(name)
            self._alive += 1
            self.max_alive = max(self.max_alive, self._alive)
        return PreparedFrame(inputs=name)

    def classify_prepared(self, prepared: PreparedFrame) -> str:
        name = str(prepared.inputs)
        with self._lock:
            self.generate_threads.add(threading.get_ident())
            self.asked.append(name)
            self._alive -= 1
        time.sleep(self.generate_seconds)
        if name in self.fail_generate:
            raise RuntimeError(f"CUDA error on {name}")
        return self.answers.get(name, self.default)

    def asker(self) -> SplitVlmClassifier:
        return SplitVlmClassifier(prepare=self.prepare,
                                  classify_prepared=self.classify_prepared)

    def serial_asker(self):
        """The pre-F206 shape: one callable, both halves, no way to overlap them."""
        def ask(path: str) -> str:
            return self.classify_prepared(self.prepare(path))

        return ask


class AskerRun:
    """A throwaway index whose every frame is a candidate of all three model passes.

    One fixture for the three, because that is what makes the phases comparable: the deep
    tier takes the frame because the product prompts flag it, the rescue because its stored
    vector scores above the threshold, the animal check because its pet score does — the
    SAME frames, so a ratio between two of the phases is a ratio over one sample, which is
    the whole point of the watchdog below.

    The two answers that keep the sample intact are deliberate: the deep tier answers
    `personal_photo` and the rescue answers `photo`, so no pass reclassifies a frame out of
    the population of the next one (`_reclassified` would otherwise trim the animal
    candidates away and the last phase would have nothing to price).
    """

    def __init__(self, workers: int, frames: int = 12) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = Config(
            sources=[Path(self.tmp.name)],
            database=Path(self.tmp.name) / "test.db",
            naming=_naming_from({"clip": {"batch_size": 8}}),
            features=FeaturesConfig(pets=True, pets_verify=True, pet_threshold=0.7,
                                    pet_candidate_threshold=0.3, junk_rescue=True,
                                    junk_rescue_threshold=0.02),
            vlm=VlmConfig(enabled=True, workers=workers),
        )
        # F145: the master switch the stage actually reads (`vlm_allowed`).
        object.__setattr__(self.cfg.naming, "vlm_enabled", True)
        self.conn = connect(self.cfg.database)
        self.names = frame_names(frames)
        for name in self.names:
            self.conn.execute(
                """INSERT INTO files (path, size, mtime, ext, media_type, width, height,
                       camera_make, camera_model, gps_lat, indexed_at)
                   VALUES (?, 1000, 0, 'jpg', 'photo', 4000, 3000, 'Canon', 'EOS', NULL,
                           '2026-01-01')""",
                (f"/photos/{name}",))
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()
        self.tmp.cleanup()

    def classifier(self) -> ThreeAskersClassifier:
        return ThreeAskersClassifier({name: 0.9 for name in self.names},
                                     {name: 0.5 for name in self.names},
                                     products=self.names)

    def run(self, pet, rescue=None, deep=None, conn=None, progress=None):
        return classify(
            self.cfg, conn or self.conn, classifier=self.classifier(),
            text_detector=NO_OCR,
            vlm_classifier=deep or (lambda _path: "personal_photo"),
            pet_vlm=pet,
            junk_rescue_vlm=rescue if rescue is not None else (lambda _path: "photo"),
            junk_text_encoder=FakeTextEncoder(),
            sharpness_detector=lambda _path, _faces: junk.Sharpness(500.0),
            progress=progress)

    def run_logged(self, pet, rescue=None, deep=None, case=None):
        """The same run, with its phase lines parsed out of the run log."""
        assert case is not None
        with case.assertLogs(_RUNLOG, level=logging.INFO) as captured:
            with runlog.stage_timer(CLASSIFY_STAGE):
                self.run(pet, rescue=rescue, deep=deep)
        lines = [m for m in (_PHASE_LINE.match(r.getMessage())
                             for r in captured.records) if m is not None]
        return {m["phase"]: m for m in lines}

    def rows(self) -> dict[str, dict]:
        """Every column any of the three passes can write, keyed by file."""
        return {
            Path(r["path"]).name: {k: r[k] for k in r.keys() if k != "path"}
            for r in self.conn.execute(
                """SELECT f.path, mc.verdict, mc.source AS verdict_source, mc.score,
                          mc.tier, fq.pet, fq.pet_score, fq.pet_vlm, fq.junk_score,
                          fq.source AS quality_source
                   FROM files f
                   LEFT JOIN media_class mc ON mc.file_id = f.id
                   LEFT JOIN frame_quality fq ON fq.file_id = f.id
                   ORDER BY f.id""")
        }

    def pet_vlm(self) -> dict[str, str | None]:
        return {
            Path(r["path"]).name: r["pet_vlm"]
            for r in self.conn.execute(
                """SELECT f.path, fq.pet_vlm FROM files f
                   JOIN frame_quality fq ON fq.file_id = f.id ORDER BY f.id""")
        }


def reversing_delays(names: list[str]) -> dict[str, float]:
    """The later the frame, the faster it prepares — completion order is reversed.

    A pass that yielded whatever finished first would then label every frame with a
    neighbour's answer, which is the failure mode this shape is here to provoke.
    """
    return {name: (len(names) - i) * 0.004 for i, name in enumerate(names)}


class TestTheVerdictsDoNotMove(unittest.TestCase):
    """The main test: the schedule changed, the answers did not."""

    def rows_for(self, workers: int, delays: dict[str, float] | None = None):
        run = AskerRun(workers)
        try:
            answers = {name: (PET_VLM_REAL if i % 2 else "depiction")
                       for i, name in enumerate(run.names)}
            pet = SplitAsker(answers, prepare_delay=delays)
            stats = run.run(pet.asker())
            self.assertEqual(len(pet.asked), len(run.names))
            return run.rows(), stats
        finally:
            run.close()

    def test_one_worker_and_four_write_the_same_rows(self):
        serial, serial_stats = self.rows_for(1)
        parallel, parallel_stats = self.rows_for(WORKERS)
        self.assertEqual(serial, parallel)
        self.assertEqual(serial_stats.pet_verified, parallel_stats.pet_verified)
        self.assertEqual(serial_stats.pets_found, parallel_stats.pets_found)
        self.assertEqual(serial_stats.by_verdict, parallel_stats.by_verdict)
        # ...and the check really did decide: half the frames lost the CLIP label
        self.assertEqual(
            (serial["cat_0.jpg"]["pet"], serial["cat_0.jpg"]["pet_vlm"]),
            (None, "depiction"))
        self.assertEqual(
            (serial["cat_1.jpg"]["pet"], serial["cat_1.jpg"]["pet_vlm"]),
            ("animal", PET_VLM_REAL))

    def test_a_racing_preparation_writes_the_same_rows_as_a_serial_one(self):
        serial, _stats = self.rows_for(1)
        parallel, _stats = self.rows_for(WORKERS, reversing_delays(frame_names(12)))
        self.assertEqual(serial, parallel)

    def test_every_answer_lands_on_its_own_file(self):
        """Test 2 of the brief: a mixed-up order gives verdicts to the wrong files."""
        run = AskerRun(WORKERS)
        self.addCleanup(run.close)
        answers = {name: (PET_VLM_REAL if i % 3 == 0 else "none")
                   for i, name in enumerate(run.names)}
        pet = SplitAsker(answers, prepare_delay=reversing_delays(run.names))
        run.run(pet.asker())

        self.assertEqual(run.pet_vlm(), answers)
        self.assertEqual(pet.asked, run.names)          # consumed in candidate order
        self.assertNotEqual(pet.prepared, run.names)    # ...while preparation raced


class TestTheHalvesRunWhereTheyShould(unittest.TestCase):
    """Preparation off the caller's thread, generation and the writes on it."""

    def test_preparation_leaves_the_thread_and_generation_does_not(self):
        run = AskerRun(WORKERS, frames=4 * WORKERS)
        self.addCleanup(run.close)
        pet = SplitAsker({}, prepare_seconds=0.002)
        run.run(pet.asker())
        self.assertNotIn(threading.get_ident(), pet.prepare_threads)
        self.assertGreater(len(pet.prepare_threads), 1)
        self.assertLessEqual(len(pet.prepare_threads), WORKERS)
        self.assertEqual(pet.generate_threads, {threading.get_ident()})

    def test_one_worker_keeps_everything_on_the_caller_thread(self):
        run = AskerRun(1, frames=6)
        self.addCleanup(run.close)
        pet = SplitAsker({})
        run.run(pet.asker())
        self.assertEqual(pet.prepare_threads, {threading.get_ident()})
        self.assertEqual(pet.generate_threads, {threading.get_ident()})

    def test_the_rescue_halves_run_where_the_animal_ones_do(self):
        run = AskerRun(WORKERS, frames=4 * WORKERS)
        self.addCleanup(run.close)
        rescue = SplitAsker({}, default="photo", prepare_seconds=0.002)
        run.run(SplitAsker({}).asker(), rescue=rescue.asker())
        self.assertNotIn(threading.get_ident(), rescue.prepare_threads)
        self.assertEqual(rescue.generate_threads, {threading.get_ident()})
        self.assertEqual(len(rescue.asked), 4 * WORKERS)

    def test_frames_in_flight_are_bounded(self):
        # The GPU half is the slow one here — the opposite of production, and exactly the
        # case in which an unbounded pool would prepare the whole candidate list at once.
        run = AskerRun(WORKERS, frames=40)
        self.addCleanup(run.close)
        pet = SplitAsker({}, generate_seconds=0.002)
        run.run(pet.asker())
        self.assertLessEqual(pet.max_alive, 2 * WORKERS)
        self.assertGreater(pet.max_alive, 1, "the workers must actually run ahead")

    def test_writes_stay_on_the_caller_thread(self):
        run = AskerRun(WORKERS, frames=3 * WORKERS)
        self.addCleanup(run.close)
        spy = ThreadSpyConn(run.conn)
        pet = SplitAsker({}, prepare_seconds=0.001)
        run.run(pet.asker(), conn=spy)
        self.assertEqual(spy.threads, {threading.get_ident()})


class TestOneBadFrame(unittest.TestCase):
    """Test 3 of the brief: a frame that fails takes no neighbour with it."""

    def test_a_failed_preparation_keeps_the_cheap_label(self):
        run = AskerRun(WORKERS)
        self.addCleanup(run.close)
        answers = {name: "depiction" for name in run.names}
        pet = SplitAsker(answers, fail_prepare=frozenset({"cat_0.jpg"}))
        with self.assertLogs("sorta.junk", level="WARNING") as logs:
            run.run(pet.asker())

        rows = run.rows()
        # the frame nobody could decode: no answer stored, the CLIP label untouched
        self.assertEqual((rows["cat_0.jpg"]["pet"], rows["cat_0.jpg"]["pet_vlm"]),
                         ("animal", None))
        # ...and every neighbour answered for
        for name in run.names[1:]:
            with self.subTest(name=name):
                self.assertEqual((rows[name]["pet"], rows[name]["pet_vlm"]),
                                 (None, "depiction"))
        self.assertTrue(any("животных" in m for m in logs.output))

    def test_a_failed_generation_keeps_the_cheap_label(self):
        run = AskerRun(WORKERS)
        self.addCleanup(run.close)
        pet = SplitAsker({name: PET_VLM_REAL for name in run.names},
                         fail_generate=frozenset({"cat_2.jpg"}))
        with self.assertLogs("sorta.junk", level="WARNING"):
            run.run(pet.asker())

        rows = run.rows()
        self.assertIsNone(rows["cat_2.jpg"]["pet_vlm"])
        self.assertEqual(rows["cat_1.jpg"]["pet_vlm"], PET_VLM_REAL)
        self.assertEqual(rows["cat_3.jpg"]["pet_vlm"], PET_VLM_REAL)

    def test_a_failing_rescue_candidate_leaves_the_fast_verdict(self):
        run = AskerRun(WORKERS)
        self.addCleanup(run.close)
        rescue = SplitAsker({name: "screenshot" for name in run.names}, default="photo",
                            fail_prepare=frozenset({"cat_1.jpg"}))
        with self.assertLogs("sorta.junk", level="WARNING") as logs:
            run.run(SplitAsker({}).asker(), rescue=rescue.asker())

        rows = run.rows()
        self.assertEqual(rows["cat_1.jpg"]["verdict"], "photo")   # the fast tier's
        self.assertEqual(rows["cat_0.jpg"]["verdict"], "screenshot")  # the model's
        self.assertTrue(any("кандидату" in m for m in logs.output))

    def test_a_failing_last_frame_still_completes_the_bar(self):
        run = AskerRun(WORKERS, frames=6)
        self.addCleanup(run.close)
        seen: list[tuple[int, int | None]] = []
        pet = SplitAsker({}, fail_prepare=frozenset({"cat_5.jpg"}))
        with self.assertLogs("sorta.junk", level="WARNING"):
            run.run(pet.asker(),
                    progress=lambda done, total: seen.append((done, total)))
        self.assertEqual(seen[-1], (6, 6))


class TestTheWorkIsNotDoubled(unittest.TestCase):
    """Test 5 of the brief: a pipeline overlaps the decodes, it does not repeat them."""

    def prepared_by(self, workers: int) -> list[str]:
        run = AskerRun(workers)
        try:
            pet = SplitAsker({})
            run.run(pet.asker())
            self.assertEqual(len(pet.asked), len(run.names))
            return sorted(pet.prepared)
        finally:
            run.close()

    def test_one_preparation_per_candidate_on_both_paths(self):
        serial, parallel = self.prepared_by(1), self.prepared_by(WORKERS)
        self.assertEqual(parallel, serial)
        self.assertEqual(len(parallel), len(set(parallel)))


# Both subclasses below price a frame by SLEEPING for it — 8 ms of "prepare" against 2 ms
# of "generate" — and then assert about the ratio of the two. That is an assertion about
# time, and a machine running eight test processes and a worker session slows the
# pipelined path as readily as the serial one, which compresses the ratio and fails the
# claim on a run where nothing is wrong. Caught on 2026-08-08 in a gate for a feature that
# touches neither askers nor timing.
@pytest.mark.serial
class PhaseRatioCase(unittest.TestCase):
    """The shared arithmetic of the watchdog: seconds per frame, phase against phase."""

    FRAMES = 24

    def phase_seconds_per_frame(self, lines) -> dict[str, float]:
        out = {}
        for phase, match in lines.items():
            processed = int(match["processed"] or 0)
            if processed:
                out[phase] = float(match["elapsed"]) / processed
        return out

    def priced_run(self, *, split: bool):
        """One run of the fixture through askers of the given shape; its phase prices."""
        run = AskerRun(WORKERS, frames=self.FRAMES)
        self.addCleanup(run.close)
        deep = SplitAsker({name: "personal_photo" for name in run.names},
                          prepare_seconds=PREPARE_SECONDS,
                          generate_seconds=GENERATE_SECONDS)
        pet = SplitAsker({}, prepare_seconds=PREPARE_SECONDS,
                         generate_seconds=GENERATE_SECONDS)
        rescue = SplitAsker({}, default="photo", prepare_seconds=PREPARE_SECONDS,
                            generate_seconds=GENERATE_SECONDS)
        lines = run.run_logged(
            pet.asker() if split else pet.serial_asker(),
            rescue=rescue.asker() if split else rescue.serial_asker(),
            deep=deep.asker(), case=self)
        for asker in (deep, pet, rescue):
            self.assertEqual(len(asker.asked), self.FRAMES)
        return self.phase_seconds_per_frame(lines)


class TestTheFramePriceGuard(PhaseRatioCase):
    """The watchdog the brief will not accept the feature without.

    Relative and not absolute on purpose: 0.42 frames/s is a number about a card, while
    "three times what the phase next to it costs" is a number about this stage, and the
    second one survives being run on somebody else's machine. Both phases below ask one
    model one question about one frame — the deep tier's `junk_vlm` is the reference
    because it is the pass that has been pipelined since F101.
    """

    def setUp(self):
        self.prices = self.priced_run(split=True)

    def test_the_animal_phase_is_not_multiply_dearer_than_the_deep_tier(self):
        ratio = self.prices[CLASSIFY_PHASE_PETS_VLM] / self.prices[CLASSIFY_PHASE_VLM]
        self.assertLess(
            ratio, MAX_PHASE_RATIO,
            f"{CLASSIFY_PHASE_PETS_VLM} costs x{ratio:.2f} of {CLASSIFY_PHASE_VLM} per "
            f"frame — the same question about the same frame of the same model, so this "
            f"is the pass having stopped overlapping (F206)")

    def test_the_rescue_phase_is_not_multiply_dearer_than_the_deep_tier(self):
        ratio = self.prices[CLASSIFY_PHASE_RESCUE_VLM] / self.prices[CLASSIFY_PHASE_VLM]
        self.assertLess(
            ratio, MAX_PHASE_RATIO,
            f"{CLASSIFY_PHASE_RESCUE_VLM} costs x{ratio:.2f} of {CLASSIFY_PHASE_VLM} "
            f"per frame (F206)")

    def test_all_three_phases_were_actually_priced(self):
        """A watchdog over a phase that did not run would be green forever."""
        for phase in (CLASSIFY_PHASE_VLM, CLASSIFY_PHASE_PETS_VLM,
                      CLASSIFY_PHASE_RESCUE_VLM):
            with self.subTest(phase=phase):
                self.assertGreater(self.prices.get(phase, 0.0), 0.0)


class TestTheGuardHasTeeth(PhaseRatioCase):
    """The same fixture through the PRE-F206 asker: the guard above must fail on it.

    Without this the watchdog is a test that cannot go red, which is exactly what the
    three days the regression lived were made of — a green gate over a question nobody
    asked.
    """

    def test_an_asker_without_halves_breaks_the_ratio(self):
        prices = self.priced_run(split=False)
        ratio = prices[CLASSIFY_PHASE_PETS_VLM] / prices[CLASSIFY_PHASE_VLM]
        self.assertGreater(
            ratio, MAX_PHASE_RATIO,
            f"a serial asker cost only x{ratio:.2f} of the pipelined tier — the guard "
            f"in TestTheFramePriceGuard would not have caught F206")


class TestTheQuestionKeepsItsHalves(unittest.TestCase):
    """`_frame_question` over a runtime: the halves when it has them, the old path else."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = str(Path(self.tmp.name) / "frame.jpg")
        Image.new("RGB", (64, 48), (10, 100, 200)).save(self.path, "JPEG")

    def split_runtime(self, answer: str = "real"):
        """A SplitVlm that records what each half was given."""
        calls: dict[str, list] = {"prepare": [], "generate": []}

        def prepare(frames, prompt):
            calls["prepare"].append((len(frames), prompt))
            return {"frames": len(frames)}

        def generate(prepared, max_new_tokens):
            calls["generate"].append((prepared, max_new_tokens))
            return answer

        return SplitVlm(prepare=prepare, generate=generate), calls

    def test_a_split_runtime_gives_a_split_pet_asker(self):
        runtime, calls = self.split_runtime()
        ask = vlm_pet_asker(runtime, max_edge=128)
        self.assertIsInstance(ask, SplitVlmClassifier)
        prepared = ask.prepare(self.path)
        self.assertIsNone(prepared.label)        # the model still has to answer
        self.assertEqual(calls["generate"], [])  # ...and has not been asked yet
        self.assertEqual(ask.classify_prepared(prepared), "real")
        self.assertEqual(calls["prepare"], [(1, junk._PET_VLM_PROMPT)])
        self.assertEqual(calls["generate"][0][1], junk._PET_VLM_MAX_NEW_TOKENS)

    def test_a_split_runtime_gives_a_split_rescue_asker_with_its_own_prompt(self):
        runtime, calls = self.split_runtime("screenshot")
        ask = vlm_junk_rescue_asker(runtime, max_edge=128)
        self.assertIsInstance(ask, SplitVlmClassifier)
        self.assertEqual(ask(self.path), "screenshot")   # calling it does both halves
        self.assertEqual(calls["prepare"], [(1, junk._JUNK_RESCUE_PROMPT)])
        self.assertEqual(calls["generate"][0][1], junk._JUNK_RESCUE_MAX_NEW_TOKENS)

    def test_a_plain_runtime_still_gives_the_plain_asker(self):
        seen = []

        def describe(frames, prompt, max_new_tokens):
            seen.append((len(frames), prompt, max_new_tokens))
            return "real"

        ask = vlm_pet_asker(describe, max_edge=128)
        self.assertNotIsInstance(ask, SplitVlmClassifier)
        self.assertEqual(ask(self.path), "real")
        self.assertEqual(seen, [(1, junk._PET_VLM_PROMPT,
                                 junk._PET_VLM_MAX_NEW_TOKENS)])

    def test_a_missing_frame_never_reaches_the_model_on_either_path(self):
        gone = str(Path(self.tmp.name) / "gone.jpg")
        runtime, calls = self.split_runtime()
        ask = vlm_pet_asker(runtime, max_edge=128)
        prepared = ask.prepare(gone)
        self.assertEqual(prepared.label, "")     # "not asked", as the serial path answers
        self.assertEqual(ask.classify_prepared(prepared), "")
        self.assertEqual(calls["prepare"], [])
        self.assertEqual(calls["generate"], [])
        self.assertEqual(vlm_pet_asker(lambda *_a: "real", max_edge=128)(gone), "")

    def test_an_undecodable_frame_answers_nothing_on_either_path(self):
        broken = Path(self.tmp.name) / "broken.jpg"
        broken.write_bytes(b"not an image at all")
        runtime, calls = self.split_runtime()
        self.assertEqual(vlm_pet_asker(runtime, max_edge=128)(str(broken)), "")
        self.assertEqual(calls["prepare"], [])
        self.assertEqual(
            vlm_junk_rescue_asker(lambda *_a: "photo", max_edge=128)(str(broken)), "")

    def test_an_unreadable_answer_is_still_not_an_answer(self):
        """The parsers own that rule; what is pinned here is that the halves feed them."""
        runtime, _calls = self.split_runtime("I could not say")
        self.assertIsNone(
            junk.parse_pet_answer(vlm_pet_asker(runtime, max_edge=128)(self.path)))


class TestTheAcceptanceScript(unittest.TestCase):
    """`scripts/measure_asker_pipeline.py` — the halves of it that need no model.

    The script is what the sample acceptance is read off (rates, the VRAM peak of both
    arms, and how many frames answered differently), so the two things it must not get
    wrong are covered here: the population it samples is the one the stage gates on, and
    the pre-registered verdict puts the answers ABOVE the speed.
    """

    def setUp(self):
        self.script = _load_measure_script()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def arm(self, name, workers, seconds, answers, peak=None):
        return self.script.ArmRow(name=name, workers=workers, frames=len(answers),
                                  seconds=seconds, answers=answers, errors=0,
                                  peak_vram_mb=peak)

    def test_the_sample_is_the_candidate_population_of_the_question(self):
        db = Path(self.tmp.name) / "sample.db"
        conn = connect(db)
        existing = str(Path(self.tmp.name) / "here.jpg")
        Image.new("RGB", (8, 8)).save(existing, "JPEG")
        for path, pet in ((existing, 0.9), ("/photos/low.jpg", 0.1),
                          ("/photos/gone.jpg", 0.9)):
            cur = conn.execute(
                """INSERT INTO files (path, size, mtime, ext, media_type, indexed_at)
                   VALUES (?, 1, 0, 'jpg', 'photo', '2026-01-01')""", (path,))
            conn.execute(
                "INSERT INTO frame_quality (file_id, pet_score, source, updated_at)"
                " VALUES (?, ?, 'vlm', '2026-01-01')", (cur.lastrowid, pet))
        conn.commit()
        conn.close()

        # below the threshold — not a candidate; missing from disk — nothing to ask about
        self.assertEqual(
            self.script.sample_paths(str(db), "pets", 0.3, 10, seed=1), [existing])

    def test_a_disagreement_is_counted_on_the_parsed_answer(self):
        before = self.arm("serial", 1, 4.0, ["real", "  DEPICTION  ", "none"])
        after = self.arm("pipeline", 4, 1.0, ["real", "depiction", "real"])
        # the middle pair differs in wording alone and is not a disagreement; the last
        # pair is a different stored label and is
        self.assertEqual(self.script.disagreements(before, after, "pets"), 1)

    def test_answers_that_moved_outrank_a_speedup(self):
        before = self.arm("serial", 1, 10.0, ["real"] * 4)
        after = self.arm("pipeline", 4, 1.0, ["none"] * 4)
        self.assertIn("СТОП", self.script.outcome([before, after], moved=4))

    def test_a_pass_that_did_not_overlap_is_not_accepted(self):
        before = self.arm("serial", 1, 10.0, ["real"] * 4)
        after = self.arm("pipeline", 4, 9.5, ["real"] * 4)
        self.assertIn("не принято", self.script.outcome([before, after], moved=0))

    def test_a_clean_run_is_accepted_and_the_table_prints_both_vram_peaks(self):
        before = self.arm("serial", 1, 10.0, ["real"] * 4, peak=20500.0)
        after = self.arm("pipeline", 4, 2.5, ["real"] * 4, peak=20512.0)
        self.assertIn("принято", self.script.outcome([before, after], moved=0))
        table = self.script.format_table([before, after], moved=0)
        self.assertIn("20500 МБ", table)
        self.assertIn("20512 МБ", table)
        self.assertIn("вердикты разошлись: 0", table)


if __name__ == "__main__":
    unittest.main()
