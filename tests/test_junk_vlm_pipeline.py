"""F101: the deep VLM tier overlaps its CPU half with its GPU half — same verdicts.

No transformers here: the classifier is injected as a `SplitVlmClassifier` whose two
halves are fakes, so everything the feature actually consists of — preparation off the
caller's thread, generation on it, the candidate order preserved, the window bounded,
one frame's failure surviving, SQLite still single-writer — is covered without a model
or a GPU.

The test this feature stands or falls on is the first one: `vlm_workers=1` and
`vlm_workers=4` must write byte-identical media_class rows. F101 is a perf change, and
the brief is explicit that a speedup that moves one verdict is not a speedup.
"""
from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path

from PIL import Image

from sorta.config import Config, _naming_from
from sorta.db import connect
from sorta.junk import (
    _VLM_PROMPT,
    PreparedFrame,
    _vlm_labels,
    SplitVlmClassifier,
    classify,
    resolve_vlm_workers,
    vlm_classifier_from,
)
from sorta.naming import SplitVlm
from tests.test_junk import NO_OCR, _RECEIPT_IDX, FakeClassifier

WORKERS = 4
# #14/V1: doc_score inside the candidate zone (>= text_rescue_docscore_min 0.3, below
# document_threshold 0.9) — the fast tier says 'photo' and hands the frame to the VLM.
CANDIDATE_DOC_SCORE = 0.5


class FakeSplitVlm:
    """A split classifier with observable halves: what ran, where, and how many at once.

    `prepare` is the CPU half the pipeline moves onto worker threads; `classify_prepared`
    is the GPU half that must stay on the caller's. Either can be made slow (so that
    completion order differs from input order) or made to fail. `max_alive` is the number
    of prepared-but-not-yet-classified frames that existed at the same time — the RAM
    bound the brief asks for.
    """

    def __init__(self, labels: dict[str, str], prepare_delay: dict[str, float] | None = None,
                 fail_prepare: frozenset[str] = frozenset(),
                 fail_generate: frozenset[str] = frozenset(),
                 generate_delay: float = 0.0):
        self.labels = labels
        self.prepare_delay = prepare_delay or {}
        self.fail_prepare = fail_prepare
        self.fail_generate = fail_generate
        self.generate_delay = generate_delay
        self._lock = threading.Lock()
        self.prepare_threads: set[int] = set()
        self.generate_threads: set[int] = set()
        self.prepared: list[str] = []   # in completion order
        self.classified: list[str] = []  # in consumption order
        self._alive = 0
        self.max_alive = 0

    def prepare(self, path: str) -> PreparedFrame:
        name = Path(path).name
        time.sleep(self.prepare_delay.get(name, 0.0))
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
            self.classified.append(name)
            self._alive -= 1
        time.sleep(self.generate_delay)
        if name in self.fail_generate:
            raise RuntimeError(f"CUDA error on {name}")
        return self.labels.get(name, "personal_photo")

    def classifier(self) -> SplitVlmClassifier:
        return SplitVlmClassifier(prepare=self.prepare,
                                  classify_prepared=self.classify_prepared)


class ThreadSpyConn:
    """A connection proxy recording the thread of every execute() (single-writer)."""

    def __init__(self, conn):
        self._conn = conn
        self.threads: set[int] = set()

    def execute(self, *args, **kwargs):
        self.threads.add(threading.get_ident())
        return self._conn.execute(*args, **kwargs)

    def __enter__(self):
        return self._conn.__enter__()

    def __exit__(self, *exc):
        return self._conn.__exit__(*exc)


class Candidates:
    """A throwaway DB of canonical photos that all pass the VLM candidate gate."""

    def __init__(self, vlm_workers: int, batch_size: int = 16):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = Config(
            sources=[Path(self.tmp.name)],
            database=Path(self.tmp.name) / "test.db",
            naming=_naming_from({"clip": {"batch_size": batch_size}}),
            raw={"naming": {"vlm_workers": vlm_workers}},
        )
        object.__setattr__(self.cfg.naming, "vlm_enabled", True)
        self.conn = connect(self.cfg.database)
        self.names: list[str] = []

    def close(self) -> None:
        self.conn.close()
        self.tmp.cleanup()

    def add_files(self, n: int, prefix: str = "cand") -> None:
        for i in range(n):
            name = f"{prefix}_{i}.jpg"
            self.conn.execute(
                """INSERT INTO files (path, size, mtime, ext, media_type, width, height,
                       camera_make, camera_model, gps_lat, indexed_at)
                   VALUES (?, 1000, 0, 'jpg', 'photo', 4000, 3000, 'Canon', 'EOS', NULL,
                           '2026-01-01')""",
                (f"/photos/{name}",))
            self.names.append(name)
        self.conn.commit()

    def classifier(self) -> FakeClassifier:
        return FakeClassifier(
            {}, doc_scores={n: (_RECEIPT_IDX, CANDIDATE_DOC_SCORE) for n in self.names})

    def labels(self) -> dict[str, str]:
        """A different label per third of the sample — so an order bug cannot hide."""
        wheel = ("document", "product", "personal_photo")
        return {name: wheel[i % 3] for i, name in enumerate(self.names)}

    def run(self, fake: FakeSplitVlm, conn=None, progress=None):
        return classify(self.cfg, (conn or self.conn), classifier=self.classifier(),
                        text_detector=NO_OCR, vlm_classifier=fake.classifier(),
                        progress=progress)

    def rows(self) -> dict[str, tuple]:
        return {
            Path(r["path"]).name: (r["verdict"], r["source"], r["score"], r["tier"])
            for r in self.conn.execute(
                """SELECT f.path, mc.verdict, mc.source, mc.score, mc.tier
                   FROM media_class mc JOIN files f ON f.id = mc.file_id
                   ORDER BY f.id""")
        }


class TestResolveVlmWorkers(unittest.TestCase):
    """naming.vlm_workers comes straight out of cfg.raw, like naming.ocr_workers."""

    def test_value_from_raw_wins(self):
        self.assertEqual(resolve_vlm_workers({"naming": {"vlm_workers": 6}}), 6)

    def test_default_when_absent(self):
        default = resolve_vlm_workers(None)
        self.assertEqual(resolve_vlm_workers({}), default)
        self.assertEqual(resolve_vlm_workers({"naming": {}}), default)
        self.assertEqual(resolve_vlm_workers({"naming": None}), default)
        # modest on purpose: the machine running this may have two cores
        self.assertLessEqual(default, 4)
        self.assertGreaterEqual(default, 1)

    def test_zero_and_negative_fall_back_to_default(self):
        default = resolve_vlm_workers(None)
        self.assertEqual(resolve_vlm_workers({"naming": {"vlm_workers": 0}}), default)
        self.assertEqual(resolve_vlm_workers({"naming": {"vlm_workers": -2}}), default)

    def test_garbage_falls_back_to_default(self):
        default = resolve_vlm_workers(None)
        self.assertEqual(resolve_vlm_workers({"naming": {"vlm_workers": "many"}}), default)
        self.assertEqual(resolve_vlm_workers({"naming": {"vlm_workers": [4]}}), default)

    def test_never_below_one(self):
        for value in (1, 0, -1, "x", None):
            self.assertGreaterEqual(
                resolve_vlm_workers({"naming": {"vlm_workers": value}}), 1)


class TestPipelineEquivalence(unittest.TestCase):
    """The main F101 test: the number of preparation threads changes nothing."""

    def rows_for(self, workers: int, delays: dict[str, float] | None = None):
        col = Candidates(workers)
        try:
            col.add_files(12)
            fake = FakeSplitVlm(col.labels(), prepare_delay=delays)
            stats = col.run(fake)
            self.assertEqual(stats.vlm_candidates, 12)
            self.assertEqual(len(fake.classified), 12)
            return col.rows(), stats
        finally:
            col.close()

    def test_one_and_four_workers_agree(self):
        serial, serial_stats = self.rows_for(1)
        parallel, parallel_stats = self.rows_for(WORKERS)
        self.assertEqual(serial, parallel)
        self.assertEqual(serial_stats.by_verdict, parallel_stats.by_verdict)
        self.assertEqual(serial_stats.vlm_applied, parallel_stats.vlm_applied)
        # ...and the deep tier really did decide: a third of the frames became products
        self.assertEqual(serial["cand_0.jpg"][:2], ("document", "vlm"))
        self.assertEqual(serial["cand_1.jpg"][:2], ("product", "vlm"))
        self.assertEqual(serial["cand_2.jpg"][:2], ("photo", "vlm"))

    def test_out_of_order_preparation_still_writes_the_right_verdicts(self):
        # the later the frame, the faster it prepares: completion order is the reverse
        # of the candidate order, and a pipeline that yielded "whatever finished first"
        # would label every file with its neighbour's answer
        delays = {f"cand_{i}.jpg": (12 - i) * 0.005 for i in range(12)}
        parallel, _ = self.rows_for(WORKERS, delays)
        serial, _ = self.rows_for(1)
        self.assertEqual(serial, parallel)

    def test_labels_are_consumed_in_candidate_order(self):
        col = Candidates(WORKERS)
        self.addCleanup(col.close)
        col.add_files(10)
        fake = FakeSplitVlm(col.labels(),
                            prepare_delay={f"cand_{i}.jpg": (10 - i) * 0.005
                                           for i in range(10)})
        col.run(fake)
        self.assertEqual(fake.classified, col.names)
        self.assertNotEqual(fake.prepared, col.names)  # preparation really did race


class TestHalvesRunWhereTheyShould(unittest.TestCase):
    """Preparation off the caller's thread, generation (the single GPU) on it."""

    def setUp(self):
        self.col = Candidates(WORKERS)
        self.addCleanup(self.col.close)

    def test_prepare_leaves_the_thread_and_generate_does_not(self):
        self.col.add_files(4 * WORKERS)
        fake = FakeSplitVlm(self.col.labels())
        self.col.run(fake)
        self.assertNotIn(threading.get_ident(), fake.prepare_threads)
        self.assertGreater(len(fake.prepare_threads), 1)
        self.assertLessEqual(len(fake.prepare_threads), WORKERS)
        self.assertEqual(fake.generate_threads, {threading.get_ident()})

    def test_single_worker_keeps_everything_on_the_caller_thread(self):
        col = Candidates(1)
        self.addCleanup(col.close)
        col.add_files(6)
        fake = FakeSplitVlm(col.labels())
        col.run(fake)
        self.assertEqual(fake.prepare_threads, {threading.get_ident()})
        self.assertEqual(fake.generate_threads, {threading.get_ident()})

    def test_frames_in_flight_are_bounded(self):
        # RAM: the pass must not prepare the whole candidate list up front. The GPU half
        # is made the slow one here (which is the opposite of production, and exactly
        # the case in which an unbounded pool would run away with the whole list).
        self.col.add_files(40)
        fake = FakeSplitVlm(self.col.labels(), generate_delay=0.002)
        self.col.run(fake)
        self.assertLessEqual(fake.max_alive, 2 * WORKERS)
        self.assertGreater(fake.max_alive, 1, "the workers must actually run ahead")

    def test_writes_happen_only_on_the_main_thread(self):
        self.col.add_files(3 * WORKERS)
        fake = FakeSplitVlm(self.col.labels())
        spy = ThreadSpyConn(self.col.conn)
        self.col.run(fake, conn=spy)
        self.assertEqual(spy.threads, {threading.get_ident()})
        self.assertEqual(len(self.col.rows()), 3 * WORKERS)


class TestFrameErrors(unittest.TestCase):
    """One frame failing keeps its fast verdict — and the pass keeps its counter."""

    def rows_with(self, **kwargs):
        col = Candidates(WORKERS)
        self.addCleanup(col.close)
        col.add_files(6)
        fake = FakeSplitVlm(col.labels(), **kwargs)
        with self.assertLogs("sorta.junk", level="WARNING") as logs:
            stats = col.run(fake)
        return col.rows(), stats, logs.output

    def test_preparation_failure_keeps_the_fast_verdict(self):
        rows, stats, logs = self.rows_with(fail_prepare=frozenset({"cand_0.jpg"}))
        self.assertEqual(rows["cand_0.jpg"][:2], ("photo", "clip"))  # the fast verdict
        self.assertEqual(rows["cand_1.jpg"][:2], ("product", "vlm"))
        self.assertEqual(rows["cand_3.jpg"][:2], ("document", "vlm"))
        self.assertEqual(stats.vlm_candidates, 6)
        self.assertTrue(any("VLM-ошибка" in m for m in logs))

    def test_generation_failure_keeps_the_fast_verdict(self):
        rows, _stats, logs = self.rows_with(fail_generate=frozenset({"cand_1.jpg"}))
        self.assertEqual(rows["cand_1.jpg"][:2], ("photo", "clip"))
        self.assertEqual(rows["cand_0.jpg"][:2], ("document", "vlm"))
        self.assertTrue(any("VLM-ошибка" in m for m in logs))

    def test_a_failing_last_frame_still_completes_the_progress_bar(self):
        # F100: the counter must reach its total even when the model failed on the
        # final candidate — the pipeline must not reintroduce that freeze
        col = Candidates(WORKERS)
        self.addCleanup(col.close)
        col.add_files(6)
        seen: list[tuple[int, int | None]] = []
        fake = FakeSplitVlm(col.labels(), fail_prepare=frozenset({"cand_5.jpg"}))
        with self.assertLogs("sorta.junk", level="WARNING"):
            col.run(fake, progress=lambda done, total: seen.append((done, total)))
        self.assertEqual(seen[-1], (6, 6))
        self.assertIn((0, 6), seen)


class TestSerialFallback(unittest.TestCase):
    """A classifier without halves (every mock in the suite) takes the old path."""

    def test_plain_callable_classifier_never_leaves_the_thread(self):
        col = Candidates(WORKERS)
        self.addCleanup(col.close)
        col.add_files(5)
        threads: set[int] = set()

        def classify_media(path: str) -> str:
            threads.add(threading.get_ident())
            return "product"

        stats = classify(col.cfg, col.conn, classifier=col.classifier(),
                         text_detector=NO_OCR, vlm_classifier=classify_media)
        self.assertEqual(threads, {threading.get_ident()})
        self.assertEqual(stats.vlm_applied, 5)
        self.assertTrue(all(r[:2] == ("product", "vlm") for r in col.rows().values()))

    def test_a_failing_serial_classifier_keeps_the_fast_verdict(self):
        # the serial path owns the same contract as the pipelined one: the exception
        # belongs to the caller, which logs it and leaves the frame as the fast tier
        # classified it
        col = Candidates(1)
        self.addCleanup(col.close)
        col.add_files(3)

        def classify_media(path: str) -> str:
            if path.endswith("cand_1.jpg"):
                raise RuntimeError("CUDA out of memory")
            return "document"

        with self.assertLogs("sorta.junk", level="WARNING") as logs:
            classify(col.cfg, col.conn, classifier=col.classifier(),
                     text_detector=NO_OCR, vlm_classifier=classify_media)
        rows = col.rows()
        self.assertEqual(rows["cand_1.jpg"][:2], ("photo", "clip"))
        self.assertEqual(rows["cand_0.jpg"][:2], ("document", "vlm"))
        self.assertEqual(rows["cand_2.jpg"][:2], ("document", "vlm"))
        self.assertTrue(any("VLM-ошибка" in m for m in logs.output))


class TestVlmLabelsContract(unittest.TestCase):
    """_vlm_labels on its own: one item per path, in order, on both paths."""

    PATHS = [f"/photos/cand_{i}.jpg" for i in range(5)]

    def labels_of(self, fake_or_fn, workers):
        return list(_vlm_labels(fake_or_fn, self.PATHS, workers))

    def test_serial_path_yields_one_label_per_path_in_order(self):
        def classify_media(path):
            return "document" if path.endswith("_1.jpg") else "product"

        self.assertEqual(self.labels_of(classify_media, 1),
                         ["product", "document", "product", "product", "product"])

    def test_serial_path_yields_the_exception_of_a_bad_frame(self):
        def classify_media(path):
            if path.endswith("_2.jpg"):
                raise RuntimeError("boom")
            return "product"

        labels = self.labels_of(classify_media, 1)
        self.assertIsInstance(labels[2], RuntimeError)
        self.assertEqual([label for label in labels if label != labels[2]],
                         ["product"] * 4)

    def test_a_split_classifier_with_one_worker_stays_serial(self):
        fake = FakeSplitVlm({f"cand_{i}.jpg": "product" for i in range(5)})
        self.assertEqual(self.labels_of(fake.classifier(), 1), ["product"] * 5)
        self.assertEqual(fake.prepare_threads, {threading.get_ident()})

    def test_pipelined_path_yields_the_same_sequence(self):
        labels = {f"cand_{i}.jpg": ("document", "product", "personal_photo")[i % 3]
                  for i in range(5)}
        fake = FakeSplitVlm(labels, prepare_delay={"cand_0.jpg": 0.03})
        self.assertEqual(self.labels_of(fake.classifier(), WORKERS),
                         [labels[f"cand_{i}.jpg"] for i in range(5)])

    def test_an_empty_candidate_list_starts_no_threads(self):
        fake = FakeSplitVlm({})
        self.assertEqual(list(_vlm_labels(fake.classifier(), [], WORKERS)), [])
        self.assertEqual(fake.prepare_threads, set())


class TestPipelineSpeedup(unittest.TestCase):
    """Acceptance: with a synthetic CPU half, four preparation threads beat one."""

    def elapsed(self, workers: int, frames: int, delay: float) -> float:
        col = Candidates(workers)
        try:
            col.add_files(frames)
            fake = FakeSplitVlm(col.labels(),
                                prepare_delay={n: delay for n in col.names})
            started = time.perf_counter()
            col.run(fake)
            spent = time.perf_counter() - started
            self.assertEqual(len(fake.classified), frames)
            return spent
        finally:
            col.close()

    def test_four_workers_beat_one(self):
        frames, delay = 6 * WORKERS, 0.02  # ~0.48 s serial, ~0.12 s on 4 workers
        serial = self.elapsed(1, frames, delay)
        parallel = self.elapsed(WORKERS, frames, delay)
        self.assertLess(parallel, serial / 2,
                        f"vlm_workers={WORKERS} must be well faster than 1: "
                        f"{parallel:.3f}s vs {serial:.3f}s")


class TestVlmClassifierFromRuntime(unittest.TestCase):
    """vlm_classifier_from: the stage's decode/prompt/parsing over a loaded runtime."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = str(Path(self.tmp.name) / "frame.jpg")
        Image.new("RGB", (64, 48), (10, 100, 200)).save(self.path, "JPEG")

    def split_runtime(self, answer="product"):
        """A SplitVlm that records what each half was given."""
        calls: dict[str, list] = {"prepare": [], "generate": []}

        def prepare(frames, prompt):
            calls["prepare"].append((len(frames), prompt))
            return {"frames": len(frames)}

        def generate(prepared, max_new_tokens):
            calls["generate"].append((prepared, max_new_tokens))
            return answer

        return SplitVlm(prepare=prepare, generate=generate), calls

    def test_split_runtime_gives_a_split_classifier(self):
        runtime, calls = self.split_runtime()
        classifier = vlm_classifier_from(runtime)
        self.assertIsInstance(classifier, SplitVlmClassifier)
        prepared = classifier.prepare(self.path)
        self.assertIsNone(prepared.label)          # the model still has to answer
        self.assertEqual(calls["generate"], [])    # ...and has not been asked yet
        self.assertEqual(classifier.classify_prepared(prepared), "product")
        self.assertEqual(calls["prepare"], [(1, _VLM_PROMPT)])
        self.assertEqual(calls["generate"][0][0], {"frames": 1})

    def test_calling_the_split_classifier_does_both_halves(self):
        runtime, _calls = self.split_runtime("document")
        self.assertEqual(vlm_classifier_from(runtime)(self.path), "document")

    def test_plain_runtime_gives_a_plain_classifier(self):
        seen = []

        def describe(frames, prompt, max_new_tokens):
            seen.append((len(frames), prompt, max_new_tokens))
            return "  DOCUMENT  "

        classifier = vlm_classifier_from(describe)
        self.assertNotIsInstance(classifier, SplitVlmClassifier)
        self.assertEqual(classifier(self.path), "document")
        self.assertEqual(seen[0][:2], (1, _VLM_PROMPT))

    def test_plain_runtime_also_answers_conservatively_without_a_frame(self):
        seen = []

        def describe(frames, prompt, max_new_tokens):
            seen.append(prompt)
            return "document"

        classifier = vlm_classifier_from(describe)
        self.assertEqual(classifier(str(Path(self.tmp.name) / "gone.jpg")),
                         "personal_photo")
        self.assertEqual(seen, [])  # the model is not asked about a frame we do not have

    def test_unknown_answer_falls_back_to_personal_photo(self):
        runtime, _calls = self.split_runtime("I think this might be a nice picture")
        self.assertEqual(vlm_classifier_from(runtime)(self.path), "personal_photo")

    def test_missing_file_never_reaches_the_model(self):
        runtime, calls = self.split_runtime()
        classifier = vlm_classifier_from(runtime)
        prepared = classifier.prepare(str(Path(self.tmp.name) / "gone.jpg"))
        self.assertEqual(prepared.label, "personal_photo")
        self.assertEqual(classifier.classify_prepared(prepared), "personal_photo")
        self.assertEqual(calls["prepare"], [])
        self.assertEqual(calls["generate"], [])

    def test_undecodable_file_never_reaches_the_model(self):
        broken = Path(self.tmp.name) / "broken.jpg"
        broken.write_bytes(b"not an image at all")
        runtime, calls = self.split_runtime()
        self.assertEqual(vlm_classifier_from(runtime)(str(broken)), "personal_photo")
        self.assertEqual(calls["prepare"], [])


if __name__ == "__main__":
    unittest.main()
