"""F72: a pool of exiftool sessions instead of one serial process.

exiftool is a separate process, so the GIL does not cap it: on the production
collection one session read 11.8 ms/file and eight read 2.0 (x5.8). Everything here
runs against the fake exiftool from test_exif_flags — the real binary is never called
in the gate. The parts that matter are that N sessions return exactly what one session
returned, that a broken session only costs its own slice, and that no process is left
running afterwards.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import sorta.exif as exif
from tests.test_exif_flags import FakeExifTool


def meta_for(names: list[str]) -> dict:
    """Each file reports its own name as Make — cross-talk between slices is visible."""
    return {
        name: {"Make": name, "Model": "cam", "ImageWidth": 64, "ImageHeight": 48,
               "DateTimeOriginal": "2024:01:02 03:04:05", "GPSLatitude": 55.75,
               "GPSLongitude": 37.62, "Orientation": 6}
        for name in names
    }


def names(count: int, prefix: str = "img") -> list[str]:
    return [f"{prefix}_{i:04d}.jpg" for i in range(count)]


class TestSplit(unittest.TestCase):
    """Every path lands in exactly one slice — nothing lost, nothing read twice."""

    def test_slices_cover_input_exactly(self):
        for length in (0, 1, 7, 200, 1000):
            paths = [Path(f"/x/{n}") for n in names(length)]
            for workers in (1, 2, 4, 8):
                count = exif._slice_count(length, workers)
                chunks = exif._split(paths, count)
                with self.subTest(length=length, workers=workers):
                    self.assertEqual([p for c in chunks for p in c], paths)
                    self.assertEqual(len(chunks), count)

    def test_slice_count_never_below_one_and_never_above_workers(self):
        for length in (0, 1, 7, 200, 1000):
            for workers in (1, 2, 4, 8):
                count = exif._slice_count(length, workers)
                self.assertGreaterEqual(count, 1)
                self.assertLessEqual(count, workers)

    def test_small_batches_use_fewer_slices(self):
        self.assertEqual(exif._slice_count(5, 8), 1)
        self.assertEqual(exif._slice_count(31, 8), 1)
        self.assertEqual(exif._slice_count(64, 8), 2)
        self.assertEqual(exif._slice_count(1000, 8), 8)  # capped by the worker count

    def test_slices_are_balanced(self):
        chunks = exif._split([Path(str(i)) for i in range(200)], 7)
        self.assertEqual(sorted({len(c) for c in chunks}), [28, 29])


class TestResolveExifWorkers(unittest.TestCase):
    """index.exif_workers straight out of cfg.raw, like hashing.resolve_workers."""

    def test_raw_value_wins(self):
        self.assertEqual(exif.resolve_exif_workers({"index": {"exif_workers": 3}}), 3)

    def test_default_is_min_8_cpu(self):
        with mock.patch("os.cpu_count", return_value=16):
            self.assertEqual(exif.resolve_exif_workers({}), 8)
            self.assertEqual(exif.resolve_exif_workers(None), 8)
            self.assertEqual(exif.resolve_exif_workers({"index": {}}), 8)
            self.assertEqual(exif.resolve_exif_workers({"index": {"exif_workers": 0}}), 8)
            self.assertEqual(exif.resolve_exif_workers({"index": {"exif_workers": -4}}), 8)
            self.assertEqual(exif.resolve_exif_workers({"index": {"exif_workers": "x"}}), 8)

    def test_default_follows_a_small_cpu_count(self):
        with mock.patch("os.cpu_count", return_value=2):
            self.assertEqual(exif.resolve_exif_workers({}), 2)
        with mock.patch("os.cpu_count", return_value=None):
            self.assertEqual(exif.resolve_exif_workers({}), 1)

    def test_always_at_least_one(self):
        with mock.patch("os.cpu_count", return_value=None):
            for raw in ({}, {"index": {"exif_workers": 0}}, {"index": {"exif_workers": -1}}):
                self.assertGreaterEqual(exif.resolve_exif_workers(raw), 1)

    def test_index_workers_is_a_different_knob(self):
        # the hashing thread count must not be mistaken for the exiftool session count
        with mock.patch("os.cpu_count", return_value=16):
            self.assertEqual(exif.resolve_exif_workers({"index": {"workers": 20}}), 8)


class FakeExifToolTestCase(unittest.TestCase):
    """Base: a fake exiftool over `count` synthetic paths (no files on disk needed)."""

    count = 200
    crash_on: list[str] = []
    delay = 0.0

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.names = names(self.count)
        self.paths = [self.root / n for n in self.names]
        self.fake = FakeExifTool(self.root, meta_for(self.names),
                                 crash_on=self.crash_on, delay=self.delay)

    def tearDown(self):
        self.fake.restore()
        self.tmp.cleanup()

    def key(self, path: Path) -> str:
        return str(path.resolve())


class TestEquivalence(FakeExifToolTestCase):
    """The main guarantee: N sessions return exactly what one session returned."""

    def test_one_session_and_eight_agree(self):
        one = exif.read_batch(self.paths, 1)
        many = exif.read_batch(self.paths, 8)
        self.assertEqual(set(one), {self.key(p) for p in self.paths})
        self.assertEqual(one, many)

    def test_every_field_survives_the_split(self):
        out = exif.read_batch(self.paths, 8)
        for path in self.paths:
            data = out[self.key(path)]
            self.assertEqual(data.make, path.name)  # no answers swapped between slices
            self.assertEqual((data.width, data.height, data.orientation), (64, 48, 6))
            self.assertEqual((data.gps_lat, data.gps_lon), (55.75, 37.62))
            self.assertEqual(data.datetime_original, "2024:01:02 03:04:05")

    def test_empty_batch_spawns_nothing(self):
        self.assertEqual(exif.read_batch([], 8), {})
        self.assertEqual(self.fake.launches(), 0)


class TestSessionCount(FakeExifToolTestCase):
    count = 5

    def test_small_batch_uses_one_session(self):
        exif.read_batch(self.paths, 8)
        self.assertEqual(self.fake.launches(), 1)

    def test_sessions_are_lazy(self):
        # importing/patching alone must not spawn eight exiftool processes
        self.assertEqual(self.fake.launches(), 0)


class TestSessionReuse(FakeExifToolTestCase):
    def test_two_calls_do_not_spawn_new_processes(self):
        exif.read_batch(self.paths, 4)
        after_first = self.fake.launches()
        self.assertEqual(after_first, 4)
        exif.read_batch(self.paths, 4)
        # Not an exact equality: a session that dies is transparently restarted by
        # `_ensure`, which is deliberate behaviour, and under the load of the full
        # suite that occasionally happens. What must hold is that the second call
        # REUSES the pool rather than building a fresh one — anything below a second
        # full set of launches proves that.
        self.assertLess(self.fake.launches(), after_first * 2,
                        "второй вызов поднял новый пул вместо переиспользования")

    def test_a_wider_call_only_adds_the_missing_sessions(self):
        # A session that dies is transparently restarted by `_ensure` — deliberate
        # behaviour that fires now and then under the load of the full suite, so ANY
        # exact launch count here is a coin toss rather than a check. The neighbour
        # above was fixed for this long ago; this test kept two equalities and was the
        # flake that turned CI red at random (twice locally on 2026-07-28, then on
        # Windows CI twice more the day after — the second time because the first fix
        # only replaced the LOWER of its two exact counts and left this one).
        #
        # Both bounds are stated against what was actually launched, not against a
        # constant: at least as many sessions as were asked for, and no full rebuild of
        # the ones that already exist.
        exif.read_batch(self.paths, 2)
        after_first = self.fake.launches()
        self.assertGreaterEqual(after_first, 2, "обе сессии первого вызова не поднялись")

        exif.read_batch(self.paths, 4)
        self.assertGreaterEqual(self.fake.launches(), 4, "не все сессии подняты")
        self.assertLess(self.fake.launches(), after_first + 4,
                        f"широкий вызов пересоздал пул вместо добавления недостающих: "
                        f"{self.fake.launches()} запусков при {after_first} до него")


class TestFailureIsolation(FakeExifToolTestCase):
    """A session that dies takes its own slice to the one-shot fallback, nothing else."""

    crash_on = ["img_0000.jpg"]  # first slice

    def test_other_slices_keep_their_results(self):
        out = exif.read_batch(self.paths, 8)  # no exception escapes
        self.assertEqual(set(out), {self.key(p) for p in self.paths})
        for path in self.paths:
            self.assertEqual(out[self.key(path)].make, path.name)

    def test_broken_session_is_restarted_on_the_next_call(self):
        exif.read_batch(self.paths, 8)
        launches = self.fake.launches()
        out = exif.read_batch(self.paths, 8)
        self.assertGreater(self.fake.launches(), launches)  # the dead one came back
        self.assertEqual(set(out), {self.key(p) for p in self.paths})

    def test_survivors_stay_in_session(self):
        # only the crashing slice is closed; the other sessions keep their process
        exif.read_batch(self.paths, 4)  # 200 paths -> 4 slices, the first one crashes
        alive = sum(s._proc is not None for s in exif._pool.sessions(4))
        self.assertEqual(alive, 3)


class TestThreadSafety(FakeExifToolTestCase):
    """Concurrent read_batch calls share the sessions without mixing up the answers."""

    count = 400

    def test_parallel_callers_get_their_own_paths(self):
        groups = [self.paths[i::4] for i in range(4)]
        results: dict[int, dict] = {}
        errors: list[Exception] = []
        barrier = threading.Barrier(len(groups))

        def worker(idx: int) -> None:
            try:
                barrier.wait(timeout=30)
                results[idx] = exif.read_batch(groups[idx], 8)
            except Exception as e:  # reported by the assertions below, not swallowed
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(groups))]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        self.assertEqual(errors, [])
        for idx, group in enumerate(groups):
            self.assertEqual(set(results[idx]), {self.key(p) for p in group})
            for path in group:
                self.assertEqual(results[idx][self.key(path)].make, path.name)


class TestShutdown(FakeExifToolTestCase):
    """The launch count is a lower bound here for the same reason as in TestSessionReuse:
    `_ensure` transparently restarts a session that dies, which happens now and then under
    the load of the full suite, so an exact count is a coin toss. Both cases in this class
    are fixed together on purpose — the previous round of this bug was fixed in one of two
    neighbouring assertions and went red again on the other (ac1daf7).

    What actually has to hold is that nothing is left running: every session that was asked
    for exited, and the pool holds no sessions afterwards.
    """

    count = 100

    def test_close_stops_every_session(self):
        exif.read_batch(self.paths, 4)
        launched = self.fake.launches()
        self.assertGreaterEqual(launched, 4, "не все четыре сессии поднялись")
        self.assertLess(launched, 8, f"пул пересоздан целиком: {launched} запусков")
        exif._pool.close()
        self.assertGreaterEqual(self.fake.clean_exits(), 4,
                                f"close() оставил процессы: {self.fake.clean_exits()} "
                                f"выходов при {launched} запусках")
        self.assertEqual(exif._pool._sessions, [])

    def test_close_is_idempotent_and_the_pool_stays_usable(self):
        exif.read_batch(self.paths, 4)
        exif._pool.close()
        exif._pool.close()
        out = exif.read_batch(self.paths, 4)
        self.assertEqual(len(out), self.count)

    def test_atexit_leaves_no_exiftool_running(self):
        """A child process reads and exits normally: every session must exit with it."""
        repo = str(Path(exif.__file__).resolve().parents[1])
        child = self.root / "child.py"
        child.write_text(
            "import sys\n"
            "from pathlib import Path\n"
            f"sys.path.insert(0, {json.dumps(repo)})\n"
            "import sorta.exif as exif\n"
            f"exif._EXIFTOOL_CMD = [sys.executable, {json.dumps(str(self.fake.script))}]\n"
            "exif.exiftool_available = lambda: True\n"
            f"paths = [Path(p) for p in {json.dumps([str(p) for p in self.paths])}]\n"
            "out = exif.read_batch(paths, 4)\n"
            f"assert len(out) == {self.count}, len(out)\n",
            encoding="utf-8",
        )
        proc = subprocess.run([sys.executable, str(child)], capture_output=True, text=True,
                              timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        launched = self.fake.launches()
        self.assertGreaterEqual(launched, 4, "не все четыре сессии поднялись")
        self.assertLess(launched, 8, f"пул пересоздан целиком: {launched} запусков")
        # atexit closed them: a restarted session leaves an extra start without an exit,
        # so the four the child actually used are the bound, not the total launch count.
        self.assertGreaterEqual(self.fake.clean_exits(), 4,
                                f"atexit оставил процессы: {self.fake.clean_exits()} "
                                f"выходов при {launched} запусках")


class TestSpeedup(FakeExifToolTestCase):
    """Acceptance: with an artificial per-file delay, N sessions are measurably faster."""

    count = 256
    delay = 0.005  # per file inside the fake exiftool

    def measure(self, workers: int) -> float:
        pool = exif.ExifToolPool()
        try:
            for session in pool.sessions(workers):
                session.read([self.paths[0]])  # warm the processes; -stay_open is the point
            start = time.perf_counter()
            out = pool.read(self.paths, workers)
            elapsed = time.perf_counter() - start
        finally:
            pool.close()
        self.assertEqual(len(out), self.count)
        return elapsed

    def test_eight_sessions_beat_one(self):
        serial = self.measure(1)
        parallel = self.measure(8)
        print(f"\n[F72] {self.count} files @ {self.delay * 1000:.0f} ms: "
              f"1 session {serial:.2f}s -> 8 sessions {parallel:.2f}s "
              f"(x{serial / parallel:.1f})")
        self.assertLess(parallel, serial / 2)  # the ideal is x8; the margin absorbs load


if __name__ == "__main__":
    unittest.main()
