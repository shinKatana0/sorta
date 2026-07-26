"""F12.1: parallel faces inference — worker count, one session per thread, single writer.

No ML here: the insightface session is replaced by an injected `infer_factory` and
`_decode_for_faces` by a fake that packs the file index into the "image", so the
orchestration (thread-local sessions, error handling, progress, DB writes) is covered
without loading a model.
"""
from __future__ import annotations

import sys
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import sorta.faces as faces_mod
from sorta.config import Config
from sorta.faces import EMBED_DIM, _detect_parallel, _infer_workers, detect_faces
from tests.test_faces import FacesTestCase

WORKERS = 4
CUDA_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]
CPU_PROVIDERS = ["CPUExecutionProvider"]


def fake_ort(providers: list[str]) -> types.ModuleType:
    """A stand-in onnxruntime — the lazy import inside faces.py picks it up."""
    mod = types.ModuleType("onnxruntime")
    mod.get_available_providers = lambda: list(providers)
    return mod


def cfg_with(**faces_raw) -> Config:
    return Config(sources=[Path(".")], database=Path("x.db"), raw={"faces": faces_raw})


def path_index(path: str) -> int:
    """/photos/img_7.jpg -> 7 (the file counter of FacesTestCase.add_file)."""
    return int(Path(path).stem.split("_")[1])


def fake_decode(path: str, orientation: int | None) -> np.ndarray:
    """A 1×1 "frame" carrying the file index — enough for a deterministic fake infer."""
    return np.full((1, 1, 3), path_index(path), dtype=np.uint8)


def broken_decode_for(bad: int):
    def decode(path: str, orientation: int | None) -> np.ndarray:
        if path_index(path) == bad:
            raise ValueError("corrupt frame")
        return fake_decode(path, orientation)

    return decode


def expected_embedding(idx: int) -> np.ndarray:
    return np.full(EMBED_DIM, float(idx), dtype="<f4")


class FakeSessions:
    """Stands in for `_insightface_infer`: one independent session per worker thread.

    The barrier makes the "K sessions, one per thread" assertion exact instead of
    timing-dependent: a factory call blocks until all K workers have built theirs.
    """

    def __init__(self, workers: int, infer_fails_on: frozenset[int] = frozenset()):
        self.workers = workers
        self.infer_fails_on = infer_fails_on
        self._barrier = threading.Barrier(workers)
        self._lock = threading.Lock()
        self.session_threads: list[int] = []
        self.infer_threads: set[int] = set()
        self.seen: list[int] = []
        self.all_sessions_live = True

    def __call__(self):
        with self._lock:
            self.session_threads.append(threading.get_ident())
        try:
            self._barrier.wait(timeout=30)
        except threading.BrokenBarrierError:  # fewer live workers than requested
            self.all_sessions_live = False

        def infer(img: np.ndarray) -> list[tuple[list[float], float, np.ndarray]]:
            idx = int(img[0, 0, 0])
            with self._lock:
                self.infer_threads.add(threading.get_ident())
                self.seen.append(idx)
            if idx in self.infer_fails_on:
                raise RuntimeError(f"inference failed on {idx}")
            return [([0.0, 0.0, 100.0, 100.0], 0.95, expected_embedding(idx))]

        return infer


class TestInferWorkers(unittest.TestCase):
    def test_config_override_wins_over_autodetect(self):
        with mock.patch.dict(sys.modules, {"onnxruntime": fake_ort(CPU_PROVIDERS)}):
            self.assertEqual(_infer_workers(cfg_with(infer_workers=6)), 6)

    def test_auto_is_4_with_cuda_provider(self):
        with mock.patch.dict(sys.modules, {"onnxruntime": fake_ort(CUDA_PROVIDERS)}):
            self.assertEqual(_infer_workers(cfg_with()), 4)
            # no faces section at all — still the auto default
            self.assertEqual(_infer_workers(Config(database=Path("x.db"))), 4)

    def test_auto_is_1_without_cuda_provider(self):
        # on the CPU profile parallel sessions only oversubscribe the cores
        with mock.patch.dict(sys.modules, {"onnxruntime": fake_ort(CPU_PROVIDERS)}):
            self.assertEqual(_infer_workers(cfg_with(infer_workers=0)), 1)

    def test_auto_is_1_without_onnxruntime(self):
        # None in sys.modules makes `import onnxruntime` raise ImportError
        with mock.patch.dict(sys.modules, {"onnxruntime": None}):
            self.assertEqual(_infer_workers(cfg_with()), 1)

    def test_never_below_one(self):
        with mock.patch.dict(sys.modules, {"onnxruntime": fake_ort(CUDA_PROVIDERS)}):
            self.assertEqual(_infer_workers(cfg_with(infer_workers=-3)), 1)


class TestDetectParallelHelper(unittest.TestCase):
    """`_detect_parallel` on its own: rows in, (row, hits) out on the calling thread."""

    def rows(self, n: int) -> list[dict]:
        return [{"id": i, "path": f"/photos/img_{i}.jpg", "orientation": None}
                for i in range(1, n + 1)]

    def collect(self, rows, sessions, workers, decode=fake_decode, decode_workers=2):
        got: dict[int, object] = {}
        threads: set[int] = set()

        def on_result(r, hits):
            threads.add(threading.get_ident())
            got[r["id"]] = hits

        _detect_parallel(rows, decode, sessions, workers, decode_workers, on_result)
        return got, threads

    def test_every_row_once_and_results_on_the_calling_thread(self):
        sessions = FakeSessions(3)
        got, threads = self.collect(self.rows(12), sessions, 3)
        self.assertEqual(sorted(got), list(range(1, 13)))
        self.assertEqual(sorted(sessions.seen), list(range(1, 13)))
        # the single-writer invariant: results are handed over on the main thread only
        self.assertEqual(threads, {threading.get_ident()})

    def test_failed_frame_becomes_none_and_the_rest_survive(self):
        sessions = FakeSessions(3, infer_fails_on=frozenset({5}))
        got, _ = self.collect(self.rows(12), sessions, 3)
        self.assertIsNone(got[5])
        self.assertTrue(all(got[i] is not None for i in range(1, 13) if i != 5))

    def test_decode_failure_becomes_none(self):
        sessions = FakeSessions(2)
        got, _ = self.collect(self.rows(8), sessions, 2, decode=broken_decode_for(4))
        self.assertIsNone(got[4])
        self.assertTrue(all(got[i] is not None for i in range(1, 9) if i != 4))


class TestDetectFacesParallel(FacesTestCase):
    """The real path of `detect_faces` with an injected fake inference factory."""

    def setUp(self):
        super().setUp()
        self.cfg.raw = {"faces": {"infer_workers": WORKERS}}

    def detect(self, n_files: int, sessions: FakeSessions, decode=fake_decode,
               progress=None):
        ids = [self.add_file()[0] for _ in range(n_files)]
        with mock.patch("sorta.faces._decode_for_faces", decode):
            stats = detect_faces(self.cfg, self.conn, progress=progress,
                                 infer_factory=sessions)
        return ids, stats

    def faces_by_file(self):
        return {
            r["file_id"]: r for r in self.conn.execute(
                "SELECT file_id, bbox, embedding FROM faces")
        }

    def test_all_rows_processed_once_with_the_faked_hits(self):
        n = 5 * WORKERS
        ids, stats = self.detect(n, FakeSessions(WORKERS))
        self.assertEqual((stats.files_total, stats.files_processed), (n, n))
        self.assertEqual((stats.faces_found, stats.no_face_files, stats.errors), (n, 0, 0))
        rows = self.faces_by_file()
        # exactly one faces row per file; the order of insertion does not matter
        self.assertEqual(sorted(rows), sorted(ids))
        for idx, file_id in enumerate(ids, 1):
            r = rows[file_id]
            self.assertEqual(r["bbox"], "[0.0, 0.0, 100.0, 100.0]")
            np.testing.assert_array_equal(
                np.frombuffer(r["embedding"], dtype="<f4"), expected_embedding(idx))

    def test_one_session_per_worker_thread(self):
        sessions = FakeSessions(WORKERS)
        self.detect(5 * WORKERS, sessions)
        self.assertTrue(sessions.all_sessions_live,
                        "every worker must build its own session — none may be shared")
        self.assertEqual(len(sessions.session_threads), WORKERS)
        self.assertEqual(len(set(sessions.session_threads)), WORKERS)
        self.assertEqual(sessions.infer_threads, set(sessions.session_threads))
        self.assertNotIn(threading.get_ident(), sessions.infer_threads)

    def test_writes_happen_only_on_the_main_thread(self):
        write_threads: set[int] = set()
        real_write = faces_mod._write_hits

        def spy(conn, s, stats, r, hits):
            write_threads.add(threading.get_ident())
            real_write(conn, s, stats, r, hits)

        # sqlite3 connections are check_same_thread=True — a worker-thread write would
        # also raise, but we assert the invariant explicitly.
        with mock.patch("sorta.faces._write_hits", spy):
            self.detect(3 * WORKERS, FakeSessions(WORKERS))
        self.assertEqual(write_threads, {threading.get_ident()})

    def test_inference_error_is_counted_and_the_rest_are_written(self):
        n = 3 * WORKERS
        ids, stats = self.detect(n, FakeSessions(WORKERS, infer_fails_on=frozenset({3})))
        self.assertEqual((stats.errors, stats.files_processed), (1, n - 1))
        rows = self.faces_by_file()
        self.assertEqual(sorted(rows), sorted(fid for fid in ids if fid != ids[2]))
        # no row for the failed file — the next run retries it
        self.assertNotIn(ids[2], rows)

    def test_decode_error_is_counted_and_the_rest_are_written(self):
        n = 3 * WORKERS
        ids, stats = self.detect(n, FakeSessions(WORKERS), decode=broken_decode_for(2))
        self.assertEqual((stats.errors, stats.files_processed), (1, n - 1))
        self.assertNotIn(ids[1], self.faces_by_file())

    def test_progress_counts_every_frame_including_errors(self):
        n = 3 * WORKERS
        calls: list[tuple[int, int]] = []
        self.detect(n, FakeSessions(WORKERS, infer_fails_on=frozenset({1})),
                    progress=lambda i, total: calls.append((i, total)))
        self.assertEqual([i for i, _ in calls], list(range(1, n + 1)))
        self.assertTrue(all(total == n for _, total in calls))

    def test_incrementality_across_runs(self):
        ids, _ = self.detect(WORKERS, FakeSessions(WORKERS))
        sessions = FakeSessions(WORKERS)
        with mock.patch("sorta.faces._decode_for_faces", fake_decode):
            stats = detect_faces(self.cfg, self.conn, infer_factory=sessions)
        self.assertEqual((stats.files_total, stats.files_processed), (0, 0))
        self.assertEqual(sessions.seen, [])
        self.assertEqual(len(self.faces_by_file()), len(ids))


class TestDetectFacesSingleWorker(FacesTestCase):
    """infer_workers=1 (the CPU profile) keeps the prefetch-decode pipeline."""

    def setUp(self):
        super().setUp()
        self.cfg.raw = {"faces": {"infer_workers": 1, "decode_workers": 4}}

    def test_single_session_and_same_results(self):
        ids = [self.add_file()[0] for _ in range(6)]
        sessions = FakeSessions(1)
        with mock.patch("sorta.faces._decode_for_faces", fake_decode):
            stats = detect_faces(self.cfg, self.conn, infer_factory=sessions)
        self.assertEqual((stats.files_processed, stats.faces_found, stats.errors), (6, 6, 0))
        self.assertEqual(len(sessions.session_threads), 1)
        self.assertEqual(sessions.session_threads, [threading.get_ident()])
        rows = self.conn.execute("SELECT file_id, embedding FROM faces").fetchall()
        self.assertEqual(sorted(r["file_id"] for r in rows), sorted(ids))
        for idx, file_id in enumerate(ids, 1):
            got = next(r for r in rows if r["file_id"] == file_id)
            np.testing.assert_array_equal(
                np.frombuffer(got["embedding"], dtype="<f4"), expected_embedding(idx))

    def test_decode_error_counted(self):
        self.add_file()
        self.add_file()
        sessions = FakeSessions(1)
        with mock.patch("sorta.faces._decode_for_faces", broken_decode_for(1)):
            stats = detect_faces(self.cfg, self.conn, infer_factory=sessions)
        self.assertEqual((stats.errors, stats.files_processed), (1, 1))
