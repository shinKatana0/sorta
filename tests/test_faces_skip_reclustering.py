"""F212: clustering that knows nothing has changed.

The measurement this grew out of (repeat run, 2026-08-06): a second pass over a ready
database took 4 min 16 s, of which `faces` was 171.9 s — 67% of the whole run — while
detection had exactly FOUR new frames to look at. `detect_and_cluster` always called
`cluster_faces`, and `cluster_faces` always ran HDBSCAN over all 24 477 faces, whether or
not a single one of them was new.

So the stage now stores what its clusters are an answer to (`cluster_state.fingerprint`,
schema v29) and compares before recomputing — the device `frame_quality.source` and
`landmark_checks.model` already use. The tests below are about the two ways that can be
wrong, and they are not symmetric:

* skipping when something DID change is silent corruption — the clusters go on describing
  a collection that no longer exists. Hence a case per input that must invalidate: a face
  added, a face deleted, one face swapped for another with the counter left where it was
  (the case a `COUNT(*)` fingerprint would miss), a threshold moved, the algorithm version
  raised, and `--rescan` asked for by hand.
* recomputing when nothing changed is only slow — but it is the whole feature, so the
  main case pins it by replacing the splitting function itself rather than by timing
  anything.

And the boundary that makes the feature safe to have at all: manual work on clusters — a
name, a merge, a split — changes the clusters and NOT the set of faces, so it must leave
the fingerprint alone. A recomputation triggered by naming somebody would wipe the name.
"""
from __future__ import annotations

import contextlib
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from sorta import faces
from sorta.config import Config, FacesConfig
from sorta.db import SCHEMA_VERSION, connect, reset_index
from sorta.faces import (
    EMBED_DIM,
    cluster_faces,
    detect_and_cluster,
    detect_faces,
    label_cluster,
    merge,
)

from tests.schema_history import roll_back_before

RNG = np.random.default_rng(212)


def unit(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v)).astype(np.float32)


def face_of(axis: int) -> np.ndarray:
    """One more shot of the person living on `axis` — close to the previous ones."""
    center = np.zeros(EMBED_DIM, dtype=np.float64)
    center[axis] = 1.0
    return unit(center + 0.05 * RNG.normal(size=EMBED_DIM))


def hit(emb: np.ndarray) -> tuple[list[float], float, np.ndarray]:
    return ([10.0, 10.0, 130.0, 130.0], 0.95, emb)


@contextlib.contextmanager
def watch_hdbscan():
    """Count the calls to the splitting function, keeping its real behaviour.

    The brief's requirement: prove the skip by the FUNCTION not being called, not by a
    run being quick. A timing assertion would pass on a fast machine with the skip
    removed, which is the failure this whole file exists to catch.
    """
    calls: list[int] = []
    real = faces._hdbscan_labels

    def spy(x, s):
        calls.append(int(x.shape[0]))
        return real(x, s)

    with mock.patch.object(faces, "_hdbscan_labels", spy):
        yield calls


class ClusterSkipTestCase(unittest.TestCase):
    """A small collection plus a fake detector — the fixture of tests/test_faces_rescan."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = self.config()
        self.conn = connect(self.cfg.database)
        self.hits: dict[str, list] = {}
        self.paths: dict[int, str] = {}
        self._n = 0

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def config(self, **faces_kwargs) -> Config:
        """The run's config; small synthetic groups have to be clusters, not noise."""
        settings = {"min_cluster_size": 2, **faces_kwargs}
        return Config(
            sources=[Path(self.tmp.name)],
            database=Path(self.tmp.name) / "test.db",
            faces=FacesConfig(**settings),
        )

    def analyzer(self, path, orientation):
        return self.hits[path]

    def add_photo(self, *embeddings: np.ndarray) -> int:
        self._n += 1
        path = f"/photos/img_{self._n}.jpg"
        cur = self.conn.execute(
            """INSERT INTO files (path, size, mtime, ext, media_type, indexed_at)
               VALUES (?, 1000, 0, 'jpg', 'photo', '2026-01-01')""", (path,))
        self.conn.commit()
        self.hits[path] = [hit(e) for e in embeddings]
        self.paths[cur.lastrowid] = path
        return cur.lastrowid

    def person(self, axis: int, n: int) -> list[int]:
        return [self.add_photo(face_of(axis)) for _ in range(n)]

    def run_stage(self, cfg: Config | None = None, **kwargs):
        return detect_and_cluster(cfg or self.cfg, self.conn,
                                  analyzer=self.analyzer, **kwargs)

    def run_stage_detect_only(self):
        return detect_faces(self.cfg, self.conn, analyzer=self.analyzer)

    def collection(self, people: int = 2, shots: int = 6) -> list[list[int]]:
        """A clustered collection: `people` people with `shots` photos each."""
        made = [self.person(axis, shots) for axis in range(people)]
        self.run_stage()
        return made

    def fingerprint(self) -> str:
        return faces._cluster_fingerprint(self.conn, faces._settings(self.cfg))

    def stored(self) -> str | None:
        return faces._stored_fingerprint(self.conn)

    def clustering(self) -> dict[int, int | None]:
        """face id -> its cluster, the answer the rest of the program reads."""
        return {r["id"]: r["cluster_id"] for r in self.conn.execute(
            "SELECT id, cluster_id FROM faces ORDER BY id")}

    def labels(self) -> dict[int, str | None]:
        return {r["id"]: r["label"] for r in self.conn.execute(
            "SELECT id, label FROM face_clusters ORDER BY id")}

    def cluster_of(self, *files: int) -> int:
        holes = ",".join("?" * len(files))
        rows = self.conn.execute(
            f"""SELECT DISTINCT cluster_id FROM faces
                WHERE cluster_id IS NOT NULL AND file_id IN ({holes})""", files).fetchall()
        self.assertEqual(len(rows), 1, f"expected one cluster over {files}, got {rows}")
        return rows[0][0]

    def a_face_id(self) -> int:
        return self.conn.execute(
            "SELECT MIN(id) FROM faces WHERE bbox != '[]'").fetchone()[0]

    def drop_face(self, face_id: int) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM faces WHERE id = ?", (face_id,))

    def insert_face(self, file_id: int, emb: np.ndarray) -> int:
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO faces (file_id, bbox, embedding) VALUES (?, ?, ?)",
                (file_id, "[0,0,10,10]", np.asarray(emb, dtype="<f4").tobytes()))
        return int(cur.lastrowid)


class TestTheSkipItself(ClusterSkipTestCase):
    """The main case: a repeat run over an unchanged collection does not split anything."""

    def test_a_second_run_with_nothing_new_never_reaches_hdbscan(self):
        self.collection()
        with watch_hdbscan() as calls:
            _faces, stats = self.run_stage()
        self.assertEqual(calls, [], "HDBSCAN ran over a set of faces that did not move")
        self.assertTrue(stats.skipped)

    def test_the_first_run_does_cluster(self):
        """The other half of the same statement — the spy sees a real first pass."""
        self.person(0, 6)
        self.person(1, 6)
        with watch_hdbscan() as calls:
            _faces, stats = self.run_stage()
        self.assertEqual(calls, [12])
        self.assertFalse(stats.skipped)

    def test_the_clusters_survive_the_skip_untouched(self):
        """Requirement 9: what is read out afterwards is what was there before."""
        self.collection()
        before, before_labels = self.clustering(), self.labels()
        self.run_stage()
        self.assertEqual(self.clustering(), before)
        self.assertEqual(self.labels(), before_labels)

    def test_a_skipped_run_reports_the_clusters_it_left_alone(self):
        """The stats a caller prints must describe the base, not report zero everywhere."""
        self.person(0, 6)
        self.person(1, 6)
        _faces, first = self.run_stage()
        _faces, second = self.run_stage()
        self.assertTrue(second.skipped)
        self.assertEqual(
            (second.faces, second.clusters, second.noise, second.malformed),
            (first.faces, first.clusters, first.noise, first.malformed))

    def test_the_marker_says_which_algorithm_produced_the_clusters(self):
        self.collection()
        stored = self.stored()
        assert stored is not None
        self.assertTrue(stored.startswith("hdbscan#"), stored)
        self.assertEqual(stored, self.fingerprint())

    def test_the_row_is_written_once_and_stays_one_row(self):
        self.collection()
        self.person(1, 2)
        self.run_stage()
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM cluster_state").fetchone()[0], 1)


class TestWhatMustInvalidateIt(ClusterSkipTestCase):
    """Every input whose change the clusters have to follow."""

    def test_a_new_face_brings_the_clustering_back(self):
        self.collection()
        self.person(0, 1)
        with watch_hdbscan() as calls:
            _faces, stats = self.run_stage()
        self.assertEqual(len(calls), 1)
        self.assertFalse(stats.skipped)

    def test_a_deleted_face_brings_the_clustering_back(self):
        """A frame sent to the trash takes its faces with it."""
        self.collection()
        self.drop_face(self.a_face_id())
        with watch_hdbscan() as calls:
            _faces, stats = self.run_stage()
        self.assertEqual(len(calls), 1)
        self.assertFalse(stats.skipped)

    def test_one_face_swapped_for_another_recomputes_although_the_count_is_the_same(self):
        """The case a `COUNT(*)` fingerprint would sleep through."""
        made = self.collection()
        before = self.conn.execute(
            "SELECT COUNT(*) FROM faces WHERE bbox != '[]'").fetchone()[0]
        self.drop_face(self.a_face_id())
        self.insert_face(made[0][0], face_of(1))
        after = self.conn.execute(
            "SELECT COUNT(*) FROM faces WHERE bbox != '[]'").fetchone()[0]
        self.assertEqual(before, after, "the fixture must leave the counter where it was")

        with watch_hdbscan() as calls:
            _faces, stats = self.run_stage()
        self.assertEqual(len(calls), 1)
        self.assertFalse(stats.skipped)

    def test_a_changed_min_cluster_size_recomputes(self):
        """Otherwise a person moves the threshold and sees no effect at all."""
        self.collection()
        changed = self.config(min_cluster_size=3)
        with watch_hdbscan() as calls:
            stats = cluster_faces(changed, self.conn)
        self.assertEqual(len(calls), 1)
        self.assertFalse(stats.skipped)

    def test_a_changed_max_distance_recomputes(self):
        """The other threshold the splitting reads — it becomes the selection epsilon."""
        self.collection()
        changed = self.config(max_distance=0.3)
        with watch_hdbscan() as calls:
            cluster_faces(changed, self.conn)
        self.assertEqual(len(calls), 1)

    def test_raising_the_algorithm_version_recomputes(self):
        """The part of the question no digest can read off the database."""
        self.collection()
        with mock.patch.object(faces, "CLUSTER_ALGO_VERSION",
                               faces.CLUSTER_ALGO_VERSION + 1), watch_hdbscan() as calls:
            cluster_faces(self.cfg, self.conn)
        self.assertEqual(len(calls), 1)

    def test_rescan_recomputes_even_though_the_fingerprint_matches(self):
        """A rescan writes new vectors under ids SQLite may hand straight back."""
        self.collection()
        self.assertEqual(self.stored(), self.fingerprint())
        with watch_hdbscan() as calls:
            _faces, stats = self.run_stage(rescan=True)
        self.assertEqual(len(calls), 1)
        self.assertFalse(stats.skipped)

    def test_force_recomputes_on_its_own(self):
        self.collection()
        with watch_hdbscan() as calls:
            cluster_faces(self.cfg, self.conn, force=True)
        self.assertEqual(len(calls), 1)

    def test_a_base_with_no_fingerprint_clusters_as_it_always_did(self):
        """An older database: the migration adds the table empty, and empty means CLUSTER."""
        self.collection()
        with self.conn:
            self.conn.execute("DELETE FROM cluster_state")
        before = self.clustering()
        with watch_hdbscan() as calls:
            _faces, stats = self.run_stage()
        self.assertEqual(len(calls), 1)
        self.assertFalse(stats.skipped)
        self.assertEqual(self.clustering(), before, "the same faces give the same split")

    def test_a_cancelled_write_does_not_leave_the_fingerprint_behind(self):
        """The marker and the clusters it describes are written in one transaction.

        Otherwise a run cancelled halfway through the write would leave a base whose
        fingerprint promises clusters nobody wrote, and every later run would agree with
        it and skip.
        """
        self.collection()
        before = self.stored()
        self.person(2, 6)
        self.run_stage_detect_only()

        class _Boom(BaseException):
            pass

        def cancel(done: int, total: int | None = None) -> None:
            if done and total == 3:            # the write phase, one cluster written
                raise _Boom()

        with self.assertRaises(_Boom):
            cluster_faces(self.cfg, self.conn, progress=cancel)
        self.assertEqual(self.stored(), before, "the cancelled run must promise nothing")
        with watch_hdbscan() as calls:
            cluster_faces(self.cfg, self.conn)
        self.assertEqual(len(calls), 1, "the next run has to finish the job")

    def test_starting_over_forgets_the_fingerprint(self):
        """`reset_index` wipes the clusters, so the marker must not outlive them."""
        self.collection()
        reset_index(self.conn)
        self.assertIsNone(self.stored())

    def test_faces_written_by_a_detection_run_are_clustered_in_the_same_pass(self):
        """The two halves in one call: detection writes, clustering must see the writes."""
        self.collection()
        self.person(2, 6)
        _faces, stats = self.run_stage()
        self.assertEqual(stats.clusters, 3)
        self.assertEqual(stats.faces, 18)


class TestManualWorkIsNotAnInput(ClusterSkipTestCase):
    """A name, a merge and a split change the clusters — never the set of faces.

    A recomputation triggered by any of them would undo the very edit that triggered it,
    which is worse than the slow run this feature removes.
    """

    def test_naming_a_cluster_moves_nothing(self):
        made = self.collection()
        before = self.fingerprint()
        label_cluster(self.conn, self.cluster_of(*made[0]), "Мама")
        self.assertEqual(self.fingerprint(), before)

        with watch_hdbscan() as calls:
            _faces, stats = self.run_stage()
        self.assertEqual(calls, [])
        self.assertTrue(stats.skipped)
        self.assertIn("Мама", set(self.labels().values()))

    def test_a_merge_moves_nothing(self):
        made = self.collection()
        before = self.fingerprint()
        root = merge(self.conn, self.cluster_of(*made[0]), self.cluster_of(*made[1]))
        label_cluster(self.conn, root, "Борис")
        self.assertEqual(self.fingerprint(), before)

        with watch_hdbscan() as calls:
            self.run_stage()
        self.assertEqual(calls, [])
        self.assertIn("Борис", set(self.labels().values()))
        self.assertEqual(
            self.conn.execute(
                "SELECT COUNT(*) FROM face_clusters WHERE merged_into IS NOT NULL"
            ).fetchone()[0], 1, "the merge itself must still be in the base")

    def test_a_split_moves_nothing(self):
        """Splitting is faces changing clusters — the ids they are keyed by do not move."""
        made = self.collection()
        before = self.fingerprint()
        cur = self.conn.execute("INSERT INTO face_clusters (label) VALUES ('Папа')")
        with self.conn:
            self.conn.execute(
                "UPDATE faces SET cluster_id = ? WHERE file_id = ?",
                (cur.lastrowid, made[0][0]))
        self.assertEqual(self.fingerprint(), before)

        split_off = self.clustering()
        with watch_hdbscan() as calls:
            self.run_stage()
        self.assertEqual(calls, [])
        self.assertEqual(self.clustering(), split_off, "the split must still be there")
        self.assertIn("Папа", set(self.labels().values()))


class TestDegenerateBases(ClusterSkipTestCase):
    """The paths that return before HDBSCAN — they must remember their answer too."""

    def test_an_empty_base_is_clustered_once_and_then_left_alone(self):
        stats = cluster_faces(self.cfg, self.conn)
        self.assertEqual((stats.faces, stats.skipped), (0, False))
        self.assertIsNotNone(self.stored())
        self.assertTrue(cluster_faces(self.cfg, self.conn).skipped)

    def test_a_base_of_malformed_embeddings_is_clustered_once(self):
        file_id = self.add_photo()
        with self.conn:
            self.conn.execute(
                "INSERT INTO faces (file_id, bbox, embedding) VALUES (?, '[0,0,9,9]', ?)",
                (file_id, b"\x00" * 8))
        first = cluster_faces(self.cfg, self.conn)
        self.assertEqual((first.malformed, first.skipped), (1, False))

        second = cluster_faces(self.cfg, self.conn)
        self.assertEqual((second.malformed, second.faces, second.skipped), (1, 1, True))
        self.assertEqual(second.noise, 0, "a malformed row is not counted twice")

    def test_the_no_faces_markers_are_not_part_of_the_question(self):
        """A frame with nobody on it changes nothing about how the others are split."""
        self.collection()
        before = self.fingerprint()
        self.add_photo()                       # detection will write the marker row
        detect_faces(self.cfg, self.conn, analyzer=self.analyzer)
        self.assertEqual(self.fingerprint(), before)
        with watch_hdbscan() as calls:
            self.assertTrue(cluster_faces(self.cfg, self.conn).skipped)
        self.assertEqual(calls, [])


class TestMigration(unittest.TestCase):
    """v29 on a database that predates the table."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db = Path(self.tmp.name) / "old.db"

    def tearDown(self):
        self.tmp.cleanup()

    def make_old_base(self) -> int:
        """A database shaped as the last version before `cluster_state` existed."""
        conn = connect(self.db)
        with conn:
            version = roll_back_before(conn, "cluster_state")
        conn.close()
        return version

    def test_the_old_shape_really_has_no_table(self):
        version = self.make_old_base()
        raw = sqlite3.connect(self.db)
        try:
            self.assertEqual(raw.execute("PRAGMA user_version").fetchone()[0], version)
            self.assertEqual(
                raw.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
                            " AND name='cluster_state'").fetchone()[0], 0)
        finally:
            raw.close()

    def test_connecting_creates_the_table_and_raises_the_version(self):
        self.make_old_base()
        conn = connect(self.db)
        try:
            self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0],
                             SCHEMA_VERSION)
            self.assertEqual(
                conn.execute("SELECT COUNT(*) FROM cluster_state").fetchone()[0], 0,
                "an upgraded base has no answer stored — so it clusters once")
        finally:
            conn.close()

    def test_connecting_twice_is_safe(self):
        self.make_old_base()
        for _ in range(2):
            conn = connect(self.db)
            conn.close()
        conn = connect(self.db)
        try:
            self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0],
                             SCHEMA_VERSION)
        finally:
            conn.close()

    def test_a_second_row_is_refused(self):
        """One clustering, one answer — the CHECK is what keeps that true."""
        conn = connect(self.db)
        try:
            conn.execute(
                "INSERT INTO cluster_state (id, fingerprint, updated_at)"
                " VALUES (1, 'hdbscan#a', '2026-08-06')")
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO cluster_state (id, fingerprint, updated_at)"
                    " VALUES (2, 'hdbscan#b', '2026-08-06')")
        finally:
            conn.close()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
