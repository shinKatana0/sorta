"""F73: parallel OCR in the junk stage — one detector per worker, identical verdicts.

No easyocr here: the detector is injected as a FACTORY (`text_detector_factory`), so
the orchestration — a per-thread detector built once and reused, the pool shrinking
when VRAM runs out, single-writer DB access, the untouched `run_ocr` gate — is covered
without loading a model. The main test is equivalence: K=1 and K=4 must produce
byte-identical media_class rows, because F73 is a perf change and nothing else.
"""
from __future__ import annotations

import tempfile
import threading
import time
import unittest
import unittest.mock
from pathlib import Path

import pytest

from sorta import junk as junk_mod
from sorta.config import Config, _naming_from
from sorta.db import connect
from sorta.junk import (
    _OcrPool,
    _resolve_detector_factory,
    classify,
    resolve_ocr_workers,
)
from tests.test_junk import _RECEIPT_IDX, FakeClassifier

WORKERS = 4
# doc_score inside the F38 rescue zone (>= the default text_rescue_docscore_min 0.3,
# below document_threshold 0.9): the frame keeps verdict='photo' through the camera
# veto, and the OCR gate opens for it.
RESCUE_DOC_SCORE = 0.5
DOC_FRAC = 0.5    # >= text_frac_document (0.15) -> the FN rescue makes it a document
PLAIN_FRAC = 0.01  # between nothing and text_frac_document -> the verdict is untouched


class FakeDetectors:
    """Stands in for easyocr_text_frac_detector: a factory, one detector per thread.

    Counts factory calls (= how many Readers a run would build) and records which
    thread built each detector and ran each frame. `barrier` makes the assertions exact
    instead of timing-dependent: a build blocks until all K workers have reached theirs,
    so a worker cannot quietly process the whole chunk before its siblings start.
    """

    def __init__(self, fracs: dict[str, float], barrier: int = 0,
                 builds_ok: int | None = None, fail_frames: frozenset[str] = frozenset(),
                 delay: float = 0.0):
        self.fracs = fracs
        self.builds_ok = builds_ok      # the factory raises after this many builds (VRAM)
        self.fail_frames = fail_frames  # frames whose detector call raises
        self.delay = delay              # synthetic per-frame cost (the speedup measurement)
        self._barrier = threading.Barrier(barrier) if barrier else None
        self._lock = threading.Lock()
        self.build_threads: list[int] = []
        self.frame_threads: set[int] = set()
        self.seen: list[str] = []
        self.all_attempted = True

    @property
    def builds(self) -> int:
        """Factory calls, successful or not — one per worker that tried to build."""
        return len(self.build_threads)

    def __call__(self):
        with self._lock:
            n = len(self.build_threads) + 1
            self.build_threads.append(threading.get_ident())
        # The barrier is waited on BEFORE the VRAM verdict below, so the degradation
        # tests are deterministic too: every worker reaches its build attempt.
        if self._barrier is not None:
            try:
                self._barrier.wait(timeout=30)
            except threading.BrokenBarrierError:  # fewer live workers than requested
                self.all_attempted = False
        if self.builds_ok is not None and n > self.builds_ok:
            raise RuntimeError(f"out of VRAM for reader #{n}")

        def text_frac(path: str, width: int | None, height: int | None) -> float | None:
            name = Path(path).name
            with self._lock:
                self.frame_threads.add(threading.get_ident())
                self.seen.append(name)
            if self.delay:
                time.sleep(self.delay)
            if name in self.fail_frames:
                raise RuntimeError(f"detect failed on {name}")
            return self.fracs.get(name)

        return text_frac


class ThreadSpyConn:
    """A connection proxy that records the thread of every execute() (single-writer)."""

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


class Collection:
    """A throwaway DB with canonical photos of three kinds (see add_files).

    `doc_N`  — camera EXIF, doc_score in the rescue zone: the OCR gate is OPEN;
    `plain_N` — doc_score ~0: the gate is closed by the F38 doc-score threshold;
    `face_N` — a detected face: the gate is closed by the face veto.
    """

    def __init__(self, ocr_workers: int, batch_size: int = 16):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = Config(
            sources=[Path(self.tmp.name)],
            database=Path(self.tmp.name) / "test.db",
            naming=_naming_from({"clip": {"batch_size": batch_size}}),
            raw={"naming": {"ocr_workers": ocr_workers}},
        )
        self.conn = connect(self.cfg.database)
        self.names: list[str] = []

    def close(self) -> None:
        self.conn.close()
        self.tmp.cleanup()

    def _add(self, name: str, has_face: bool = False) -> int:
        cur = self.conn.execute(
            """INSERT INTO files (path, size, mtime, ext, media_type, width, height,
                   camera_make, camera_model, gps_lat, indexed_at)
               VALUES (?, 1000, 0, 'jpg', 'photo', 4000, 3000, 'Canon', 'EOS', NULL,
                       '2026-01-01')""",
            (f"/photos/{name}",))
        fid = cur.lastrowid
        if has_face:
            self.conn.execute(
                "INSERT INTO faces (file_id, bbox, embedding) VALUES (?, '[1,2,3,4]', ?)",
                (fid, b"\x00" * 4))
        self.conn.commit()
        self.names.append(name)
        return fid

    def add_files(self, n_docs: int, n_plain: int = 0, n_faces: int = 0) -> None:
        for i in range(n_docs):
            self._add(f"doc_{i}.jpg")
        for i in range(n_plain):
            self._add(f"plain_{i}.jpg")
        for i in range(n_faces):
            self._add(f"face_{i}.jpg", has_face=True)

    def classifier(self) -> FakeClassifier:
        """CLIP mock: the rescue zone for doc_*/face_*, a clear scene for plain_*."""
        doc_scores = {
            name: (_RECEIPT_IDX, RESCUE_DOC_SCORE)
            for name in self.names if not name.startswith("plain_")
        }
        return FakeClassifier({}, doc_scores=doc_scores)

    def fracs(self) -> dict[str, float]:
        """Half of the gated frames get a document-grade text_frac, half do not."""
        return {
            name: (DOC_FRAC if i % 2 == 0 else PLAIN_FRAC)
            for i, name in enumerate(self.names)
        }

    def run(self, detectors: FakeDetectors, conn=None):
        return classify(self.cfg, (conn or self.conn), classifier=self.classifier(),
                        text_detector_factory=detectors)

    def rows(self) -> dict[str, tuple]:
        return {
            Path(r["path"]).name: (r["verdict"], r["source"], r["score"], r["tier"])
            for r in self.conn.execute(
                """SELECT f.path, mc.verdict, mc.source, mc.score, mc.tier
                   FROM media_class mc JOIN files f ON f.id = mc.file_id
                   ORDER BY f.id""")
        }


class TestResolveOcrWorkers(unittest.TestCase):
    """naming.ocr_workers is read straight from cfg.raw (no typed field), like
    index.workers in hashing.resolve_workers."""

    def test_value_from_raw_wins(self):
        self.assertEqual(resolve_ocr_workers({"naming": {"ocr_workers": 7}}), 7)

    def test_default_when_absent(self):
        default = resolve_ocr_workers(None)
        self.assertEqual(resolve_ocr_workers({}), default)
        self.assertEqual(resolve_ocr_workers({"naming": {}}), default)
        self.assertEqual(resolve_ocr_workers({"naming": None}), default)
        # conservative: every worker holds a Reader. F164: the ceiling itself rather
        # than a copy of its value — the number lives next to the reasoning for it.
        self.assertLessEqual(default, junk_mod._DEFAULT_OCR_WORKERS_CAP)
        self.assertGreaterEqual(default, 1)

    def test_zero_and_negative_fall_back_to_default(self):
        default = resolve_ocr_workers(None)
        self.assertEqual(resolve_ocr_workers({"naming": {"ocr_workers": 0}}), default)
        self.assertEqual(resolve_ocr_workers({"naming": {"ocr_workers": -3}}), default)

    def test_garbage_falls_back_to_default(self):
        default = resolve_ocr_workers(None)
        self.assertEqual(resolve_ocr_workers({"naming": {"ocr_workers": "many"}}), default)
        self.assertEqual(resolve_ocr_workers({"naming": {"ocr_workers": {"a": 1}}}), default)

    def test_never_below_one(self):
        for raw in ({"naming": {"ocr_workers": v}} for v in (1, 0, -1, "x", None)):
            self.assertGreaterEqual(resolve_ocr_workers(raw), 1)


class TestParallelOcrEquivalence(unittest.TestCase):
    """The main F73 test: K=1 and K=4 must classify identically."""

    def rows_for(self, workers: int) -> dict[str, tuple]:
        col = Collection(workers)
        try:
            col.add_files(n_docs=10, n_plain=4, n_faces=2)
            stats = col.run(FakeDetectors(col.fracs()))
            self.assertEqual((stats.total, stats.processed), (16, 16))
            self.assertEqual(sum(stats.by_verdict.values()), 16)
            return col.rows()
        finally:
            col.close()

    def test_same_media_class_for_one_and_four_workers(self):
        serial = self.rows_for(1)
        parallel = self.rows_for(WORKERS)
        self.assertEqual(serial, parallel)
        # and the OCR really did something: half of the gated frames became documents
        verdicts = {name: row[0] for name, row in serial.items()}
        sources = {name: row[1] for name, row in serial.items()}
        self.assertEqual(verdicts["doc_0.jpg"], "document")
        self.assertEqual(sources["doc_0.jpg"], "ocr")
        self.assertEqual(verdicts["doc_1.jpg"], "photo")
        self.assertEqual(sources["doc_1.jpg"], "clip")

    def test_same_media_class_across_chunk_boundaries(self):
        # a batch size that does not divide the file count — the OCR phase runs per
        # chunk, so the chunk edges must not shift a verdict
        def rows(workers, batch):
            col = Collection(workers, batch_size=batch)
            try:
                col.add_files(n_docs=9, n_plain=3)
                col.run(FakeDetectors(col.fracs()))
                return col.rows()
            finally:
                col.close()

        self.assertEqual(rows(1, 16), rows(WORKERS, 4))


class TestDetectorPerThread(unittest.TestCase):
    """One detector per worker thread — built once, never per frame."""

    def setUp(self):
        self.col = Collection(WORKERS)
        self.addCleanup(self.col.close)

    def test_one_detector_per_worker_thread_not_per_frame(self):
        self.col.add_files(n_docs=5 * WORKERS)
        det = FakeDetectors(self.col.fracs(), barrier=WORKERS)
        self.col.run(det)
        self.assertTrue(det.all_attempted,
                        "every worker must build its own detector — none may be shared")
        self.assertEqual(det.builds, WORKERS)                 # not one per frame
        self.assertEqual(len(set(det.build_threads)), WORKERS)  # a distinct thread each
        self.assertEqual(len(det.seen), 5 * WORKERS)
        self.assertEqual(det.frame_threads, set(det.build_threads))
        self.assertNotIn(threading.get_ident(), det.frame_threads)

    def test_detector_is_reused_across_chunks(self):
        # 3 chunks of 4 gated frames: the counter must not grow after the first chunk
        col = Collection(WORKERS, batch_size=WORKERS)
        self.addCleanup(col.close)
        col.add_files(n_docs=3 * WORKERS)
        det = FakeDetectors(col.fracs(), barrier=WORKERS)
        col.run(det)
        self.assertEqual(det.builds, WORKERS)
        self.assertEqual(len(det.seen), 3 * WORKERS)
        self.assertEqual(len(set(det.build_threads)), WORKERS)

    def test_single_worker_runs_on_the_calling_thread(self):
        # K=1 keeps the pre-F73 path: the detector lives on the caller's thread
        col = Collection(1)
        self.addCleanup(col.close)
        col.add_files(n_docs=4)
        det = FakeDetectors(col.fracs())
        col.run(det)
        self.assertEqual(det.builds, 1)
        self.assertEqual(det.build_threads, [threading.get_ident()])
        self.assertEqual(det.frame_threads, {threading.get_ident()})

    def test_no_detector_is_built_when_the_gate_opens_for_nobody(self):
        # loading a Reader costs seconds — a run where no frame needs OCR must not
        self.col.add_files(n_docs=0, n_plain=6, n_faces=2)
        det = FakeDetectors({})
        self.col.run(det)
        self.assertEqual(det.builds, 0)
        self.assertEqual(det.seen, [])


class TestOcrGateUnchanged(unittest.TestCase):
    """Parallelizing must not widen the `run_ocr` gate (F38 + the face veto)."""

    def setUp(self):
        self.col = Collection(WORKERS)
        self.addCleanup(self.col.close)

    def test_only_gated_files_reach_the_detector(self):
        self.col.add_files(n_docs=6, n_plain=5, n_faces=3)
        det = FakeDetectors(self.col.fracs())
        self.col.run(det)
        self.assertEqual(sorted(det.seen), [f"doc_{i}.jpg" for i in range(6)])
        # frames the gate skipped keep their CLIP verdict, source stays 'clip'
        rows = self.col.rows()
        for i in range(5):
            self.assertEqual(rows[f"plain_{i}.jpg"][:2], ("photo", "clip"))
        for i in range(3):
            self.assertEqual(rows[f"face_{i}.jpg"][:2], ("photo", "clip"))


class TestOcrPoolDegradation(unittest.TestCase):
    """VRAM: a detector that fails to build shrinks the pool instead of killing the stage."""

    def test_only_the_first_reader_builds_stage_still_completes(self):
        col = Collection(WORKERS)
        self.addCleanup(col.close)
        col.add_files(n_docs=10, n_plain=2)
        det = FakeDetectors(col.fracs(), barrier=WORKERS, builds_ok=1)
        with self.assertLogs("sorta.junk", level="WARNING") as logs:
            stats = col.run(det)
        self.assertEqual(det.builds, WORKERS)  # all four tried, three ran out of VRAM
        self.assertTrue(any("уменьшен" in m for m in logs.output),
                        f"the shrink must be logged, not silent: {logs.output}")
        self.assertEqual(stats.processed, 12)
        # every frame was processed, all of them by the single surviving worker
        self.assertEqual(sorted(det.seen), [f"doc_{i}.jpg" for i in range(10)])
        self.assertEqual(len(det.frame_threads), 1)

    def test_degraded_result_equals_the_single_worker_result(self):
        def rows(workers, builds_ok):
            col = Collection(workers)
            try:
                col.add_files(n_docs=8, n_plain=2)
                col.run(FakeDetectors(col.fracs(), barrier=workers, builds_ok=builds_ok))
                return col.rows()
            finally:
                col.close()

        with self.assertLogs("sorta.junk", level="WARNING"):
            degraded = rows(WORKERS, 1)
        self.assertEqual(rows(1, None), degraded)

    def test_failed_builds_are_not_retried_on_every_chunk(self):
        # once the pool has shrunk, the dead workers must not call the factory again
        col = Collection(WORKERS, batch_size=WORKERS)
        self.addCleanup(col.close)
        col.add_files(n_docs=4 * WORKERS)
        det = FakeDetectors(col.fracs(), barrier=WORKERS, builds_ok=1)
        with self.assertLogs("sorta.junk", level="WARNING"):
            col.run(det)
        # exactly one attempt per worker over the whole run (4 chunks), not per chunk
        self.assertEqual(det.builds, WORKERS)
        self.assertEqual(len(det.seen), 4 * WORKERS)

    def test_no_detector_at_all_is_a_stage_error(self):
        # an unbuildable detector was a stage error before F73 too — degrading to
        # "OCR silently off for the whole collection" would hide the reason
        col = Collection(WORKERS)
        self.addCleanup(col.close)
        col.add_files(n_docs=4)
        det = FakeDetectors(col.fracs(), barrier=WORKERS, builds_ok=0)
        with self.assertLogs("sorta.junk", level="WARNING"), \
                self.assertRaises(RuntimeError):
            col.run(det)


class TestOcrPoolLimit(unittest.TestCase):
    """_OcrPool.detector on its own: the shrunk limit is a VRAM ceiling, not a hint."""

    def test_a_late_worker_does_not_build_beyond_the_shrunk_limit(self):
        built: list[int] = []

        def factory():
            built.append(len(built) + 1)
            if len(built) > 1:  # only the first Reader fits in memory
                raise RuntimeError("out of VRAM")
            return lambda path, width, height: 0.5

        pool = _OcrPool(factory, 4)
        got: list[object] = []
        # Sequential threads: A builds the only detector, B fails and shrinks the pool
        # to 1, C arrives afterwards — and must not even call the factory again,
        # otherwise a degraded pool would keep retrying a build that cannot succeed.
        with self.assertLogs("sorta.junk", level="WARNING"):
            for _ in range(3):
                t = threading.Thread(target=lambda: got.append(pool._detector()))
                t.start()
                t.join()
        self.assertEqual(built, [1, 2])   # C did not try
        self.assertIsNotNone(got[0])
        self.assertEqual(got[1:], [None, None])
        self.assertEqual(pool.detectors_built, 1)

    def test_single_worker_build_failure_is_a_stage_error(self):
        # K=1 is the pre-F73 path: an easyocr that cannot be built killed the stage
        # then and must keep doing so (there is nothing left to degrade to)
        col = Collection(1)
        self.addCleanup(col.close)
        col.add_files(n_docs=3)
        det = FakeDetectors(col.fracs(), builds_ok=0)
        with self.assertLogs("sorta.junk", level="WARNING"), \
                self.assertRaises(RuntimeError):
            col.run(det)
        self.assertEqual(det.seen, [])


class TestDefaultDetectorFactory(unittest.TestCase):
    """Without an injected detector every worker builds its own easyocr one, with the
    downscale from the config (F38/F40) — easyocr itself is never imported here."""

    def test_default_factory_builds_easyocr_with_the_configured_downscale(self):
        cfg = Config(database=Path("x.db"),
                     naming=_naming_from({"text_frac_downscale_px": 900}))
        factory = _resolve_detector_factory(cfg, None)
        with unittest.mock.patch.object(junk_mod, "easyocr_text_frac_detector") as build:
            factory()
            factory()
        self.assertEqual(build.call_args_list,
                         [unittest.mock.call(900), unittest.mock.call(900)])

    def test_injected_detector_is_shared_by_every_worker(self):
        def detector(path, width, height):
            return 0.42

        factory = _resolve_detector_factory(Config(database=Path("x.db")), detector)
        self.assertIs(factory(), detector)
        self.assertIs(factory(), detector)


class TestOcrFrameErrors(unittest.TestCase):
    """A detector error on one frame must not break the stage or its neighbours."""

    def test_failing_frame_keeps_the_clip_verdict_of_its_neighbours(self):
        col = Collection(WORKERS)
        self.addCleanup(col.close)
        col.add_files(n_docs=6)
        det = FakeDetectors(col.fracs(), fail_frames=frozenset({"doc_0.jpg"}))
        stats = col.run(det)
        self.assertEqual(stats.processed, 6)
        rows = col.rows()
        # doc_0 would have been rescued into a document; without a signal it stays
        # photo/clip — and doc_2/doc_4 are rescued as usual
        self.assertEqual(rows["doc_0.jpg"][:2], ("photo", "clip"))
        self.assertEqual(rows["doc_2.jpg"][:2], ("document", "ocr"))
        self.assertEqual(rows["doc_4.jpg"][:2], ("document", "ocr"))
        self.assertEqual(rows["doc_1.jpg"][:2], ("photo", "clip"))


class TestSingleWriter(unittest.TestCase):
    """Only the caller's thread touches SQLite (check_same_thread stays intact)."""

    def test_writes_happen_only_on_the_main_thread(self):
        col = Collection(WORKERS)
        self.addCleanup(col.close)
        col.add_files(n_docs=4 * WORKERS, n_plain=2)
        det = FakeDetectors(col.fracs(), barrier=WORKERS)
        spy = ThreadSpyConn(col.conn)
        # the connection was created on this thread with check_same_thread=True, so a
        # worker-thread write would raise ProgrammingError as well — assert it directly
        col.run(det, conn=spy)
        self.assertEqual(spy.threads, {threading.get_ident()})
        self.assertTrue(det.frame_threads)
        self.assertNotIn(threading.get_ident(), det.frame_threads)
        self.assertEqual(len(col.rows()), 4 * WORKERS + 2)


class TestParallelOcrSpeedup(unittest.TestCase):
    """Acceptance: with a synthetic per-frame cost, K=4 is measurably faster than K=1."""

    def elapsed(self, workers: int, frames: int, delay: float) -> float:
        col = Collection(workers)
        try:
            col.add_files(n_docs=frames)
            det = FakeDetectors(col.fracs(), delay=delay)
            t0 = time.perf_counter()
            col.run(det)
            elapsed = time.perf_counter() - t0
            self.assertEqual(len(det.seen), frames)
            return elapsed
        finally:
            col.close()

    # serial: asserts on ELAPSED TIME (K=4 beats K=1). This is the test that was caught
    # red on 2026-08-02 at 0.270 s against a 0.256 s bound, with three gates running at
    # once — i.e. under exactly the load the parallel half creates every time.
    @pytest.mark.serial
    def test_four_workers_beat_one(self):
        frames, delay = 6 * WORKERS, 0.02  # ~0.48 s serial, ~0.12 s on 4 workers
        serial = self.elapsed(1, frames, delay)
        parallel = self.elapsed(WORKERS, frames, delay)
        self.assertLess(parallel, serial / 2,
                        f"K={WORKERS} must be well faster than K=1: "
                        f"{parallel:.3f}s vs {serial:.3f}s")
