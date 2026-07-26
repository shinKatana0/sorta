"""F87: decode and inference are decoupled — a decode pool feeds the session pool.

Until F87 every inference worker decoded its own frame, so a session idled for as long
as its thread read a RAW (2-5% GPU load on a real run) and `faces.decode_workers` was
not read at all on the GPU path. The tests below pin the new shape without a model:
`_decode_for_faces` is faked (the "image" carries the file index) and the insightface
session is an injected factory, so what is exercised is the orchestration — which pool
runs what, on which thread results are handed over, what stays bounded in memory.
"""
from __future__ import annotations

import threading
import time
import unittest
from unittest import mock

import numpy as np

import sorta.faces as faces_mod
from sorta.faces import _detect_parallel, detect_faces
from tests.test_faces import FacesTestCase
from tests.test_faces_parallel import (
    FakeSessions,
    broken_decode_for,
    expected_embedding,
    fake_decode,
    path_index,
)

INFER_WORKERS = 2
DECODE_WORKERS = 3


def rows_of(n: int) -> list[dict]:
    return [{"id": i, "path": f"/photos/img_{i}.jpg", "orientation": None}
            for i in range(1, n + 1)]


def fake_hits(idx: int) -> list[tuple[list[float], float, np.ndarray]]:
    return [([0.0, 0.0, 100.0, 100.0], 0.95, expected_embedding(idx))]


class BarrierDecoder:
    """A decode fake that proves the pool really is `decode_workers` threads wide.

    The first `workers` calls block on a barrier: it trips only if that many decodes
    run at the same time, which is exactly the claim "decoding happens in its own pool
    of decode_workers threads" and not "inside the inference workers". Later calls
    skip the barrier — the number of frames left is not a multiple of the pool size.
    """

    def __init__(self, workers: int) -> None:
        self._barrier = threading.Barrier(workers)
        self._lock = threading.Lock()
        self._synced = False
        self.threads: set[int] = set()
        self.all_workers_live = True

    def __call__(self, path: str, orientation: int | None) -> np.ndarray:
        with self._lock:
            self.threads.add(threading.get_ident())
            synced = self._synced
        if not synced:
            try:
                self._barrier.wait(timeout=30)
            except threading.BrokenBarrierError:  # fewer live decoders than requested
                self.all_workers_live = False
            with self._lock:
                self._synced = True
        return fake_decode(path, orientation)


class FrameTracker:
    """Counts frames that are decoded but whose inference has not finished yet.

    Inference is the slow side (a `delay` per frame), so without a bounded window the
    decode pool would run away and hold the whole collection in memory as full-res
    arrays. `peak` is what the window test asserts on.
    """

    def __init__(self, delay: float = 0.002) -> None:
        self._lock = threading.Lock()
        self._delay = delay
        self.live = 0
        self.peak = 0

    def decode(self, path: str, orientation: int | None) -> np.ndarray:
        img = fake_decode(path, orientation)
        with self._lock:
            self.live += 1
            self.peak = max(self.peak, self.live)
        return img

    def __call__(self):  # the infer factory: one session per inference thread
        def infer(img: np.ndarray) -> list[tuple[list[float], float, np.ndarray]]:
            time.sleep(self._delay)
            with self._lock:
                self.live -= 1
            return fake_hits(int(img[0, 0, 0]))

        return infer


class DecoupledHelperTest(unittest.TestCase):
    """`_detect_parallel` on its own: rows in, (row, hits) out on the calling thread."""

    def collect(self, rows, sessions, decode=fake_decode,
                workers=INFER_WORKERS, decode_workers=DECODE_WORKERS):
        got: dict[int, object] = {}
        order: list[int] = []
        threads: set[int] = set()

        def on_result(r, hits):
            threads.add(threading.get_ident())
            order.append(r["id"])
            got[r["id"]] = hits

        _detect_parallel(rows, decode, sessions, workers, decode_workers, on_result)
        return got, order, threads

    def test_decoding_runs_in_its_own_pool_apart_from_the_sessions(self):
        decoder = BarrierDecoder(DECODE_WORKERS)
        sessions = FakeSessions(INFER_WORKERS)
        got, _order, _threads = self.collect(rows_of(40), sessions, decode=decoder)
        self.assertTrue(decoder.all_workers_live,
                        "decode must run in a pool of decode_workers threads")
        self.assertEqual(len(decoder.threads), DECODE_WORKERS)
        # the decoupling itself: no thread both decodes and infers, and neither job
        # happens on the caller's thread
        self.assertEqual(decoder.threads & sessions.infer_threads, set())
        self.assertNotIn(threading.get_ident(), decoder.threads)
        self.assertEqual(len(got), 40)

    def test_on_result_runs_only_on_the_calling_thread(self):
        sessions = FakeSessions(INFER_WORKERS)
        _got, _order, threads = self.collect(rows_of(24), sessions)
        # the single-writer invariant: SQLite is written from this thread only
        self.assertEqual(threads, {threading.get_ident()})

    def test_every_row_is_handed_over_exactly_once_in_any_order(self):
        rows = rows_of(16)

        def decode(path: str, orientation: int | None) -> np.ndarray:
            # the smaller the index, the slower the decode — later frames overtake
            time.sleep(0.003 * (17 - path_index(path)))
            return fake_decode(path, orientation)

        got, order, _threads = self.collect(rows, FakeSessions(INFER_WORKERS), decode=decode)
        self.assertEqual(sorted(order), list(range(1, 17)))
        self.assertEqual(len(order), len(set(order)), "a row must not be reported twice")
        self.assertEqual(sorted(got), list(range(1, 17)))
        self.assertNotEqual(order, list(range(1, 17)),
                            "results are reported as they complete, not in input order")

    def test_a_broken_frame_is_none_and_the_rest_are_processed(self):
        got, _order, _threads = self.collect(
            rows_of(12), FakeSessions(INFER_WORKERS), decode=broken_decode_for(5))
        self.assertIsNone(got[5])
        self.assertEqual(sorted(got), list(range(1, 13)))
        self.assertTrue(all(got[i] is not None for i in range(1, 13) if i != 5))

    def test_a_failing_session_does_not_take_the_other_frames_down(self):
        sessions = FakeSessions(INFER_WORKERS, infer_fails_on=frozenset({7}))
        got, _order, _threads = self.collect(rows_of(12), sessions)
        self.assertIsNone(got[7])
        self.assertTrue(all(got[i] is not None for i in range(1, 13) if i != 7))

    def test_frames_in_flight_stay_inside_the_window(self):
        tracker = FrameTracker()
        rows = rows_of(120)
        got, _order, _threads = self.collect(
            rows, tracker, decode=tracker.decode,
            workers=INFER_WORKERS, decode_workers=DECODE_WORKERS)
        self.assertEqual(len(got), 120)
        # bounded on both sides: ~2x decode_workers decoded frames waiting, plus
        # ~2x infer_workers of them handed to a session
        window = 2 * DECODE_WORKERS + 2 * INFER_WORKERS
        self.assertLessEqual(tracker.peak, window)
        self.assertEqual(tracker.live, 0)


class DetectFacesDecoupledTest(FacesTestCase):
    """`detect_faces` end to end on the GPU path (a fake factory instead of a model)."""

    def setUp(self):
        super().setUp()
        self.cfg.raw = {"faces": {"infer_workers": INFER_WORKERS,
                                  "decode_workers": DECODE_WORKERS}}

    def detect(self, n_files: int, sessions, decode=fake_decode):
        ids = [self.add_file()[0] for _ in range(n_files)]
        with mock.patch("sorta.faces._decode_for_faces", decode):
            stats = detect_faces(self.cfg, self.conn, infer_factory=sessions)
        return ids, stats

    def faces_by_file(self) -> dict[int, tuple[str, bytes]]:
        return {
            r["file_id"]: (r["bbox"], bytes(r["embedding"]))
            for r in self.conn.execute("SELECT file_id, bbox, embedding FROM faces")
        }

    def test_same_faces_as_the_serial_path_on_the_same_input(self):
        n = 12
        ids, stats = self.detect(n, FakeSessions(INFER_WORKERS))
        decoupled = self.faces_by_file()
        self.assertEqual((stats.files_processed, stats.faces_found, stats.errors), (n, n, 0))

        # the same files through the strictly serial analyzer path (decode + infer in
        # one call), which is what the parallel scheme must remain equivalent to
        with self.conn:
            self.conn.execute("DELETE FROM faces")
        serial = detect_faces(
            self.cfg, self.conn,
            analyzer=lambda path, orientation: fake_hits(path_index(path)))
        self.assertEqual(serial.files_processed, n)
        self.assertEqual(self.faces_by_file(), decoupled)
        self.assertEqual(sorted(decoupled), sorted(ids))

    def test_decode_workers_sizes_the_pool_on_the_parallel_path(self):
        # the regression this feature is about: before F87 the setting was read only
        # on the 1-session path, so tuning it did nothing on a GPU machine
        sizes: list[int] = []
        real_prefetch = faces_mod._prefetch_decode

        def spy(rows, decode, max_workers):
            sizes.append(max_workers)
            return real_prefetch(rows, decode, max_workers)

        with mock.patch("sorta.faces._prefetch_decode", spy):
            self.detect(6, FakeSessions(INFER_WORKERS))
        self.assertEqual(sizes, [DECODE_WORKERS])

    def test_writes_and_progress_stay_on_the_main_thread(self):
        write_threads: set[int] = set()
        real_write = faces_mod._write_hits

        def spy(conn, s, stats, r, hits):
            write_threads.add(threading.get_ident())
            real_write(conn, s, stats, r, hits)

        ids = [self.add_file()[0] for _ in range(8)]
        progress_threads: set[int] = set()
        with mock.patch("sorta.faces._write_hits", spy), \
                mock.patch("sorta.faces._decode_for_faces", fake_decode):
            detect_faces(self.cfg, self.conn, infer_factory=FakeSessions(INFER_WORKERS),
                         progress=lambda i, total: progress_threads.add(threading.get_ident()))
        self.assertEqual(write_threads, {threading.get_ident()})
        self.assertEqual(progress_threads, {threading.get_ident()})
        self.assertEqual(len(self.faces_by_file()), len(ids))

    def test_decode_error_leaves_no_row_and_the_rest_are_written(self):
        ids, stats = self.detect(6, FakeSessions(INFER_WORKERS), decode=broken_decode_for(3))
        self.assertEqual((stats.errors, stats.files_processed), (1, 5))
        rows = self.faces_by_file()
        self.assertNotIn(ids[2], rows)  # no marker row — the next run retries the file
        self.assertEqual(sorted(rows), sorted(fid for fid in ids if fid != ids[2]))


class SingleWorkerPathTest(FacesTestCase):
    """infer_workers=1 (the CPU profile) still takes the serial branch, untouched."""

    def setUp(self):
        super().setUp()
        self.cfg.raw = {"faces": {"infer_workers": 1, "decode_workers": DECODE_WORKERS}}

    def test_parallel_scheme_is_not_used_and_inference_stays_on_this_thread(self):
        ids = [self.add_file()[0] for _ in range(5)]
        sessions = FakeSessions(1)

        def never(*args, **kwargs):  # pragma: no cover — the assertion is that it is not called
            raise AssertionError("infer_workers=1 must not go through _detect_parallel")

        with mock.patch("sorta.faces._detect_parallel", never), \
                mock.patch("sorta.faces._decode_for_faces", fake_decode):
            stats = detect_faces(self.cfg, self.conn, infer_factory=sessions)
        self.assertEqual((stats.files_processed, stats.faces_found, stats.errors), (5, 5, 0))
        self.assertEqual(sessions.session_threads, [threading.get_ident()])
        self.assertEqual(sessions.infer_threads, {threading.get_ident()})
        rows = self.conn.execute("SELECT file_id FROM faces").fetchall()
        self.assertEqual(sorted(r["file_id"] for r in rows), sorted(ids))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
