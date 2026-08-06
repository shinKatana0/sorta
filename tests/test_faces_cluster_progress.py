"""F84: cluster_faces reports its phases — the step no longer goes silent.

The point of the feature is honesty of the bar, not speed: HDBSCAN is one blocking
call, so the phase around it reports `total=None` (an indeterminate bar), and the
phases that CAN be measured report real counts. A call without a callback must behave
exactly as before — that is how the CLI and the rest of the suite call it.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from sorta.config import Config, FacesConfig
from sorta.db import connect
from sorta.faces import (
    CLUSTER_PHASE_CLUSTER,
    CLUSTER_PHASE_INHERIT,
    CLUSTER_PHASE_READ,
    CLUSTER_PHASE_WRITE,
    EMBED_DIM,
    _PROGRESS_EVERY,
    cluster_faces,
    detect_and_cluster,
)

RNG = np.random.default_rng(84)


def unit(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v)).astype(np.float32)


def group(axis: int, n: int) -> list[np.ndarray]:
    """n close unit vectors around a basis vector — one "person"."""
    center = np.zeros(EMBED_DIM, dtype=np.float64)
    center[axis] = 1.0
    return [unit(center + 0.05 * RNG.normal(size=EMBED_DIM)) for _ in range(n)]


class _Recorder:
    """A stage callback with the phase channel (like ui._StageProgress/TaskProgress)."""

    def __init__(self) -> None:
        self.calls: list[tuple[str | None, int, int | None]] = []
        self.current: str | None = None

    def __call__(self, done: int, total: int | None = None) -> None:
        self.calls.append((self.current, done, total))

    def phase(self, name: str) -> None:
        self.current = name

    @property
    def phases(self) -> list[str]:
        """Phase names in the order they were first reported."""
        seen: list[str] = []
        for name, _done, _total in self.calls:
            if name is not None and name not in seen:
                seen.append(name)
        return seen

    def totals_of(self, phase: str) -> list[int | None]:
        return [total for name, _done, total in self.calls if name == phase]

    def dones_of(self, phase: str) -> list[int]:
        return [done for name, done, _total in self.calls if name == phase]


class ClusterProgressTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = Config(sources=[Path(self.tmp.name)],
                          database=Path(self.tmp.name) / "test.db")
        self.conn = connect(self.cfg.database)
        self._n = 0

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def add_face(self, emb, cluster_id=None) -> int:
        self._n += 1
        cur = self.conn.execute(
            """INSERT INTO files (path, size, mtime, ext, media_type, indexed_at)
               VALUES (?, 1000, 0, 'jpg', 'photo', '2026-01-01')""",
            (f"/photos/img_{self._n}.jpg",),
        )
        cur = self.conn.execute(
            "INSERT INTO faces (file_id, bbox, embedding, cluster_id) VALUES (?, ?, ?, ?)",
            (cur.lastrowid, "[0, 0, 100, 100]",
             np.asarray(emb, dtype="<f4").tobytes(), cluster_id),
        )
        self.conn.commit()
        return cur.lastrowid

    def add_marker(self) -> None:
        """The "file processed, no faces" row — must not reach clustering."""
        self._n += 1
        cur = self.conn.execute(
            """INSERT INTO files (path, size, mtime, ext, media_type, indexed_at)
               VALUES (?, 1000, 0, 'jpg', 'photo', '2026-01-01')""",
            (f"/photos/img_{self._n}.jpg",),
        )
        self.conn.execute(
            "INSERT INTO faces (file_id, bbox, embedding) VALUES (?, '[]', ?)",
            (cur.lastrowid, b""),
        )
        self.conn.commit()

    def fill_two_groups(self) -> None:
        for emb in group(0, 6) + group(1, 6):
            self.add_face(emb)


class TestWithoutCallbackUnchanged(ClusterProgressTestCase):
    """Regression: the observability hooks change nothing for a plain call."""

    def test_clusters_are_built_without_progress(self):
        self.fill_two_groups()
        stats = cluster_faces(self.cfg, self.conn)
        self.assertEqual(stats.faces, 12)
        self.assertEqual(stats.clusters, 2)
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM face_clusters").fetchone()[0], 2)

    def test_same_result_with_and_without_progress(self):
        self.fill_two_groups()
        before = cluster_faces(self.cfg, self.conn)
        after = cluster_faces(self.cfg, self.conn, progress=_Recorder())
        self.assertEqual((before.faces, before.clusters, before.noise),
                         (after.faces, after.clusters, after.noise))

    def test_plain_callback_without_phase_channel_still_works(self):
        # The stage contract is (done, total|None); a callback that cannot show a
        # caption simply gets no phases — and must not break the run.
        self.fill_two_groups()
        seen: list[tuple[int, int | None]] = []
        stats = cluster_faces(self.cfg, self.conn,
                              progress=lambda done, total: seen.append((done, total)))
        self.assertEqual(stats.clusters, 2)
        self.assertTrue(seen)
        self.assertIn((0, None), seen)  # the unmeasurable phase still reports


class TestPhaseSequence(ClusterProgressTestCase):
    def test_all_phases_in_order(self):
        self.fill_two_groups()
        rec = _Recorder()
        cluster_faces(self.cfg, self.conn, progress=rec)
        self.assertEqual(
            rec.phases,
            [CLUSTER_PHASE_READ, CLUSTER_PHASE_CLUSTER,
             CLUSTER_PHASE_INHERIT, CLUSTER_PHASE_WRITE],
        )

    def test_hdbscan_phase_is_indeterminate(self):
        # No honest percent exists inside one blocking fit_predict — total stays None
        # so the UI draws a running bar instead of a made-up share.
        self.fill_two_groups()
        rec = _Recorder()
        cluster_faces(self.cfg, self.conn, progress=rec)
        self.assertTrue(rec.totals_of(CLUSTER_PHASE_CLUSTER))
        self.assertTrue(all(t is None for t in rec.totals_of(CLUSTER_PHASE_CLUSTER)))

    def test_measurable_phases_report_real_totals(self):
        self.fill_two_groups()
        rec = _Recorder()
        stats = cluster_faces(self.cfg, self.conn, progress=rec)
        self.assertEqual(set(rec.totals_of(CLUSTER_PHASE_READ)), {12})
        self.assertEqual(rec.dones_of(CLUSTER_PHASE_READ)[-1], 12)
        self.assertEqual(set(rec.totals_of(CLUSTER_PHASE_INHERIT)), {stats.clusters})
        self.assertEqual(set(rec.totals_of(CLUSTER_PHASE_WRITE)), {stats.clusters})
        self.assertEqual(rec.dones_of(CLUSTER_PHASE_WRITE)[-1], stats.clusters)

    def test_every_phase_starts_from_zero(self):
        # A new phase restarts the counter — otherwise the bar would jump backwards
        # from the previous phase's tail without an explanation.
        self.fill_two_groups()
        rec = _Recorder()
        cluster_faces(self.cfg, self.conn, progress=rec)
        for phase in rec.phases:
            self.assertEqual(rec.dones_of(phase)[0], 0, phase)

    def test_read_phase_ticks_while_reading(self):
        for emb in group(0, _PROGRESS_EVERY + 5):
            self.add_face(emb)
        rec = _Recorder()
        cluster_faces(self.cfg, self.conn, progress=rec)
        # 0, _PROGRESS_EVERY, and the final count — reading is not one silent call.
        self.assertIn(_PROGRESS_EVERY, rec.dones_of(CLUSTER_PHASE_READ))
        self.assertEqual(rec.dones_of(CLUSTER_PHASE_READ)[-1], _PROGRESS_EVERY + 5)

    def test_markers_are_not_counted_in_the_read_total(self):
        self.fill_two_groups()
        for _ in range(3):
            self.add_marker()
        rec = _Recorder()
        cluster_faces(self.cfg, self.conn, progress=rec)
        self.assertEqual(set(rec.totals_of(CLUSTER_PHASE_READ)), {12})


class TestDegenerateCollections(ClusterProgressTestCase):
    """Empty / single-face collections must not break the callback (or the run)."""

    def test_no_faces_at_all(self):
        rec = _Recorder()
        stats = cluster_faces(self.cfg, self.conn, progress=rec)
        self.assertEqual(stats.faces, 0)
        self.assertEqual(rec.phases, [CLUSTER_PHASE_READ])
        self.assertEqual(set(rec.totals_of(CLUSTER_PHASE_READ)), {0})

    def test_only_markers(self):
        self.add_marker()
        rec = _Recorder()
        stats = cluster_faces(self.cfg, self.conn, progress=rec)
        self.assertEqual(stats.faces, 0)
        self.assertEqual(set(rec.totals_of(CLUSTER_PHASE_READ)), {0})

    def test_single_face(self):
        self.add_face(group(0, 1)[0])
        rec = _Recorder()
        stats = cluster_faces(self.cfg, self.conn, progress=rec)
        self.assertEqual(stats.faces, 1)
        self.assertEqual(stats.clusters, 0)  # a lone face is noise, not a cluster
        self.assertEqual(
            rec.phases,
            [CLUSTER_PHASE_READ, CLUSTER_PHASE_CLUSTER,
             CLUSTER_PHASE_INHERIT, CLUSTER_PHASE_WRITE],
        )
        # no groups -> the measurable phases honestly report a total of 0
        self.assertEqual(set(rec.totals_of(CLUSTER_PHASE_WRITE)), {0})

    def test_only_malformed_embeddings(self):
        self.add_face(np.zeros(EMBED_DIM // 2, dtype=np.float32))
        rec = _Recorder()
        stats = cluster_faces(self.cfg, self.conn, progress=rec)
        self.assertEqual(stats.malformed, 1)
        self.assertEqual(rec.phases, [CLUSTER_PHASE_READ])


class TestCancellationThroughProgress(ClusterProgressTestCase):
    """The UI callback raises to cancel a run — clustering must not corrupt the DB."""

    def test_raise_in_write_phase_rolls_back(self):
        self.fill_two_groups()
        cluster_faces(self.cfg, self.conn)  # a good state to be preserved
        before = self.conn.execute(
            "SELECT COUNT(*) FROM face_clusters").fetchone()[0]

        class _Boom(BaseException):
            pass

        rec = _Recorder()

        def cb(done: int, total: int | None = None) -> None:
            rec(done, total)
            if rec.current == CLUSTER_PHASE_WRITE and done:
                raise _Boom()

        cb.phase = rec.phase  # type: ignore[attr-defined]
        # `force` because since F212 a second call over an unchanged set of faces does
        # not reach the write phase at all — and this case is about what happens when it
        # does. The cancellation is the subject here, not the skip.
        with self.assertRaises(_Boom):
            cluster_faces(self.cfg, self.conn, progress=cb, force=True)
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM face_clusters").fetchone()[0],
            before)


class TestDetectAndClusterPassesProgress(ClusterProgressTestCase):
    def test_cluster_phases_reach_the_same_callback_as_detection(self):
        # The whole point: one callback drives both halves of the step, so the bar
        # does not freeze on "N/N" the moment detection ends.
        self.fill_two_groups()
        rec = _Recorder()
        _face_stats, cl_stats = detect_and_cluster(
            self.cfg, self.conn, progress=rec, analyzer=lambda path, orient: [])
        self.assertEqual(cl_stats.clusters, 2)
        self.assertEqual(
            rec.phases,
            [CLUSTER_PHASE_READ, CLUSTER_PHASE_CLUSTER,
             CLUSTER_PHASE_INHERIT, CLUSTER_PHASE_WRITE],
        )


class TestThresholdsUntouched(ClusterProgressTestCase):
    """F84 is observability only — the config thresholds keep deciding the clusters."""

    def test_min_cluster_size_from_config_still_applies(self):
        for emb in group(0, 3) + group(1, 6):
            self.add_face(emb)
        with_default = cluster_faces(self.cfg, self.conn, progress=_Recorder())
        self.assertEqual(with_default.clusters, 1)  # the group of 3 is noise at 5
        self.cfg.faces = FacesConfig(min_cluster_size=2)
        lowered = cluster_faces(self.cfg, self.conn, progress=_Recorder())
        self.assertEqual(lowered.clusters, 2)


if __name__ == "__main__":
    unittest.main()
