"""F89: `sorta faces --rescan [--limit N]` — recomputing a step that is incremental.

`detect_faces` only ever looked at files with no faces row, so after a full pass there
was nothing left to recompute: neither to measure the step on the real pipeline (F87
and F88 both had to be measured through a side script), nor to apply a detector change
to a collection that is already sorted.

The part worth testing hardest is not the flag but what it must not break. A rescan
gives every face a NEW id, and cluster labels used to be inherited by intersecting
face ids — which after a rescan is always empty, i.e. every name the user typed would
be dropped silently. Inheritance therefore switches to file ids, the identity that
survives detection; the tests below pin both that the names come across and that the
"> 50% overlap" rule still refuses to move a name onto somebody else's cluster.
"""
from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from sorta import cli
from sorta.config import Config, FacesConfig
from sorta.db import connect
from sorta.faces import (
    EMBED_DIM,
    ClusterSnapshot,
    cluster_faces,
    detect_and_cluster,
    detect_faces,
    label_cluster,
    merge,
    snapshot_clusters,
)

# One stream for the whole file: what `face_of` draws depends on how many draws the
# tests before it made. That is why the parallel half of the gate distributes by FILE
# (`--dist loadfile`, F219) — the tests here get the same vectors they got when the
# suite ran in one process, and a clustering verdict does not turn on scheduling.
# Reseeding per test would give every test its own stream and change those vectors,
# which is a re-calibration of the expectations below, not a fix.
RNG = np.random.default_rng(89)


def unit(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v)).astype(np.float32)


def face_of(axis: int) -> np.ndarray:
    """One more shot of the person living on `axis` — close to the previous ones."""
    center = np.zeros(EMBED_DIM, dtype=np.float64)
    center[axis] = 1.0
    return unit(center + 0.05 * RNG.normal(size=EMBED_DIM))


def hit(emb: np.ndarray) -> tuple[list[float], float, np.ndarray]:
    return ([10.0, 10.0, 130.0, 130.0], 0.95, emb)


class RescanTestCase(unittest.TestCase):
    """A collection of canonical photos plus a fake detector that remembers its calls."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = Config(
            sources=[Path(self.tmp.name)],
            database=Path(self.tmp.name) / "test.db",
            # small synthetic groups have to be clusters, not noise
            faces=FacesConfig(min_cluster_size=2),
        )
        self.conn = connect(self.cfg.database)
        self.hits: dict[str, list] = {}     # path -> what the detector "finds" there
        self.seen: list[str] = []           # paths the detector was actually asked about
        self.paths: dict[int, str] = {}     # file id -> its path
        self._n = 0

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def analyzer(self, path, orientation):
        self.seen.append(path)
        return self.hits[path]

    def add_photo(self, *embeddings: np.ndarray, media_type="photo",
                  dup_of=None, error=None) -> int:
        """A file in the index; the detector will report `embeddings` on it."""
        self._n += 1
        path = f"/photos/img_{self._n}.jpg"
        cur = self.conn.execute(
            """INSERT INTO files (path, size, mtime, ext, media_type, dup_of, error,
                   indexed_at)
               VALUES (?, 1000, 0, 'jpg', ?, ?, ?, '2026-01-01')""",
            (path, media_type, dup_of, error),
        )
        self.conn.commit()
        self.hits[path] = [hit(e) for e in embeddings]
        self.paths[cur.lastrowid] = path
        return cur.lastrowid

    def person(self, axis: int, n: int) -> list[int]:
        """n photos of one person — one face each. Returns the file ids."""
        return [self.add_photo(face_of(axis)) for _ in range(n)]

    def face_ids(self) -> dict[int, list[int]]:
        """file_id -> its faces rows, markers included."""
        out: dict[int, list[int]] = {}
        for r in self.conn.execute("SELECT id, file_id FROM faces ORDER BY id"):
            out.setdefault(r["file_id"], []).append(r["id"])
        return out

    def n_faces(self) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) FROM faces WHERE bbox != '[]'").fetchone()[0]

    def labels_of(self, *files: int) -> set[str | None]:
        """Labels of the clusters these files' faces landed in (noise has none)."""
        holes = ",".join("?" * len(files))
        return {
            r["label"] for r in self.conn.execute(
                f"""SELECT c.label FROM faces f JOIN face_clusters c ON c.id = f.cluster_id
                    WHERE f.file_id IN ({holes})""", files)
        }

    def split_in_two(self, left: list[int], right: list[int]) -> tuple[int, int]:
        """Put the faces of these two file sets into two clusters, by hand.

        Deliberately not through HDBSCAN: it would put one person into one cluster,
        and the case under test is a pair that the USER decided to merge (F3).
        """
        made: list[int] = []
        for files in (left, right):
            cur = self.conn.execute("INSERT INTO face_clusters (label) VALUES (NULL)")
            made.append(int(cur.lastrowid))
            holes = ",".join("?" * len(files))
            self.conn.execute(
                f"UPDATE faces SET cluster_id = ? WHERE file_id IN ({holes})",
                (made[-1], *files))
        self.conn.commit()
        return made[0], made[1]

    def cluster_of(self, *files: int) -> int:
        """The cluster these files' faces are in; HDBSCAN may leave single shots as noise."""
        holes = ",".join("?" * len(files))
        rows = self.conn.execute(
            f"""SELECT DISTINCT cluster_id FROM faces
                WHERE cluster_id IS NOT NULL AND file_id IN ({holes})""", files
        ).fetchall()
        self.assertEqual(len(rows), 1, f"expected one cluster over {files}, got {rows}")
        return rows[0][0]

    def all_labels(self) -> set[str | None]:
        return {r["label"] for r in self.conn.execute("SELECT label FROM face_clusters")}


class TestIncrementalUnchanged(RescanTestCase):
    """Regression: without the flag the step behaves exactly as before F89."""

    def test_second_run_without_the_flag_detects_nothing(self):
        self.person(0, 3)
        first = detect_faces(self.cfg, self.conn, analyzer=self.analyzer)
        self.assertEqual((first.files_total, first.files_processed), (3, 3))
        before = self.face_ids()

        self.seen.clear()
        second = detect_faces(self.cfg, self.conn, analyzer=self.analyzer)
        self.assertEqual(self.seen, [], "an incremental run must not touch the detector")
        self.assertEqual((second.files_total, second.files_processed), (0, 0))
        self.assertEqual(self.face_ids(), before, "the faces rows must stay untouched")

    def test_the_no_faces_marker_still_counts_as_processed(self):
        empty = self.add_photo()  # a photo with nobody on it
        detect_faces(self.cfg, self.conn, analyzer=self.analyzer)
        self.seen.clear()
        detect_faces(self.cfg, self.conn, analyzer=self.analyzer)
        self.assertEqual(self.seen, [])
        self.assertEqual(len(self.face_ids()[empty]), 1)

    def test_only_new_files_are_picked_up(self):
        self.person(0, 2)
        detect_faces(self.cfg, self.conn, analyzer=self.analyzer)
        fresh = self.add_photo(face_of(1))
        self.seen.clear()
        stats = detect_faces(self.cfg, self.conn, analyzer=self.analyzer)
        self.assertEqual(self.seen, [self.paths[fresh]])
        self.assertEqual(stats.files_total, 1)


class TestRescanRecomputes(RescanTestCase):
    """--rescan: every canonical photo goes through the detector again."""

    def test_all_files_are_detected_again(self):
        files = self.person(0, 4)
        detect_faces(self.cfg, self.conn, analyzer=self.analyzer)
        self.seen.clear()

        stats = detect_faces(self.cfg, self.conn, analyzer=self.analyzer, rescan=True)
        self.assertEqual(sorted(self.seen), sorted(self.paths[f] for f in files))
        self.assertEqual((stats.files_total, stats.files_processed), (4, 4))

    def test_old_rows_are_replaced_not_duplicated(self):
        self.person(0, 4)
        detect_faces(self.cfg, self.conn, analyzer=self.analyzer)
        before = self.face_ids()

        detect_faces(self.cfg, self.conn, analyzer=self.analyzer, rescan=True)
        after = self.face_ids()
        self.assertEqual(sorted(after), sorted(before), "the same files, one row each")
        self.assertEqual([len(v) for v in after.values()], [1, 1, 1, 1])
        old_ids = {i for ids in before.values() for i in ids}
        new_ids = {i for ids in after.values() for i in ids}
        self.assertFalse(old_ids & new_ids, "a rescan writes new rows, hence new ids")

    def test_a_changed_detector_result_lands_in_the_database(self):
        """The F88 case: the pinned det_size finds 297 faces where there were 300."""
        files = self.person(0, 3)
        detect_faces(self.cfg, self.conn, analyzer=self.analyzer)
        self.assertEqual(self.n_faces(), 3)

        self.hits[self.paths[files[0]]] = []          # this one is no longer detected
        self.hits[self.paths[files[1]]] = [hit(face_of(0)), hit(face_of(1))]  # one more
        detect_faces(self.cfg, self.conn, analyzer=self.analyzer, rescan=True)
        self.assertEqual(self.n_faces(), 3)
        self.assertEqual(len(self.face_ids()[files[1]]), 2)
        # the file with nothing on it keeps the "processed, no faces" marker
        marker = self.conn.execute(
            "SELECT bbox, embedding FROM faces WHERE file_id = ?", (files[0],)).fetchone()
        self.assertEqual((marker["bbox"], marker["embedding"]), ("[]", b""))

    def test_duplicates_videos_and_broken_files_stay_out(self):
        canon = self.add_photo(face_of(0))
        self.add_photo(face_of(0), media_type="video")
        self.add_photo(face_of(0), dup_of=canon)
        self.add_photo(face_of(0), error="boom")
        detect_faces(self.cfg, self.conn, analyzer=self.analyzer, rescan=True)
        self.assertEqual(self.seen, [self.paths[canon]])

    def test_a_failed_frame_keeps_its_previous_faces(self):
        """A read error must not be a way to lose data that is already in the base."""
        files = self.person(0, 2)
        detect_faces(self.cfg, self.conn, analyzer=self.analyzer)
        before = self.face_ids()

        def broken(path, orientation):
            if path == self.paths[files[0]]:
                raise ValueError("unreadable")
            return self.hits[path]

        stats = detect_faces(self.cfg, self.conn, analyzer=broken, rescan=True)
        self.assertEqual(stats.errors, 1)
        self.assertEqual(self.face_ids()[files[0]], before[files[0]])

    def test_limit_without_rescan_is_refused(self):
        with self.assertRaises(ValueError):
            detect_faces(self.cfg, self.conn, analyzer=self.analyzer, limit=2)


class TestRescanLimit(RescanTestCase):
    """--limit N: recompute N random files, leave the rest of the collection alone."""

    def test_exactly_n_files_are_touched(self):
        self.person(0, 8)
        detect_faces(self.cfg, self.conn, analyzer=self.analyzer)
        before = self.face_ids()
        self.seen.clear()

        stats = detect_faces(self.cfg, self.conn, analyzer=self.analyzer,
                             rescan=True, limit=3)
        self.assertEqual(len(self.seen), 3)
        self.assertEqual(len(set(self.seen)), 3, "the same file must not be picked twice")
        self.assertEqual((stats.files_total, stats.files_processed), (3, 3))

        after = self.face_ids()
        self.assertEqual(sorted(after), sorted(before), "all 8 files still have rows")
        # Which files were recomputed is what the detector was ASKED about, not a diff
        # of the row ids: a rescan deletes and re-inserts, and when the deleted rows
        # sat at the end of the table SQLite hands the same rowids straight back — so
        # a genuinely rewritten file can come back looking untouched (this comparison
        # failed roughly one run in six). The other direction is sound: a row that was
        # never deleted cannot change its id, so the rest is still checked by id.
        touched = {fid for fid, path in self.paths.items() if path in self.seen}
        self.assertEqual(len(touched), 3)
        untouched = set(before) - touched
        for fid in untouched:
            self.assertEqual(after[fid], before[fid], "row ids of the rest must not move")

    def test_a_limit_over_the_collection_size_takes_everything(self):
        self.person(0, 3)
        detect_faces(self.cfg, self.conn, analyzer=self.analyzer)
        self.seen.clear()
        stats = detect_faces(self.cfg, self.conn, analyzer=self.analyzer,
                             rescan=True, limit=50)
        self.assertEqual((stats.files_total, len(self.seen)), (3, 3))

    def test_the_sample_is_random_not_the_head_of_the_collection(self):
        """The first files by id are one folder from one camera — a biased measurement."""
        self.person(0, 40)
        detect_faces(self.cfg, self.conn, analyzer=self.analyzer)
        samples = []
        for _ in range(5):
            self.seen.clear()
            detect_faces(self.cfg, self.conn, analyzer=self.analyzer, rescan=True, limit=5)
            samples.append(tuple(sorted(self.seen)))
        self.assertGreater(len(set(samples)), 1, "five identical samples out of 40 files")


class TestLabelsSurviveRescan(RescanTestCase):
    """The point of the feature: names of people must not disappear silently."""

    def rescan(self):
        return detect_and_cluster(self.cfg, self.conn, analyzer=self.analyzer, rescan=True)

    def test_a_named_cluster_keeps_its_name(self):
        mother = self.person(0, 6)
        self.person(1, 6)
        detect_and_cluster(self.cfg, self.conn, analyzer=self.analyzer)
        label_cluster(self.conn, self.cluster_of(*mother), "Мама")

        _face_stats, cl_stats = self.rescan()
        self.assertEqual(cl_stats.clusters, 2)
        self.assertEqual(cl_stats.labels_kept, 1)
        self.assertEqual(self.labels_of(*mother), {"Мама"})

    def test_the_name_moves_with_the_files_even_when_the_faces_differ(self):
        """A detector change: one shot is lost, another gains a second face."""
        mother = self.person(0, 6)
        self.person(1, 6)
        detect_and_cluster(self.cfg, self.conn, analyzer=self.analyzer)
        label_cluster(self.conn, self.cluster_of(*mother), "Мама")

        self.hits[self.paths[mother[0]]] = []
        self.hits[self.paths[mother[1]]] = [hit(face_of(0)), hit(face_of(0))]
        self.rescan()
        self.assertEqual(self.labels_of(*mother), {"Мама"})

    def test_a_merged_pair_keeps_the_name_the_user_gave_it(self):
        """F3 merge: the chain is resolved to its root before the snapshot is taken."""
        left = self.person(0, 3)
        right = self.person(0, 3)   # the same person, split into two clusters
        detect_faces(self.cfg, self.conn, analyzer=self.analyzer)
        a, b = self.split_in_two(left, right)
        merge(self.conn, a, b)
        label_cluster(self.conn, a, "Борис")

        _face_stats, cl_stats = self.rescan()
        self.assertEqual(cl_stats.labels_kept, 1)
        self.assertEqual(self.labels_of(*left, *right), {"Борис"})

    def test_a_name_is_not_pulled_onto_a_cluster_that_is_mostly_new_files(self):
        """Same rule as before: below a half of the files in common — no inheritance."""
        known = self.person(0, 3)
        detect_and_cluster(self.cfg, self.conn, analyzer=self.analyzer)
        label_cluster(self.conn, self.cluster_of(*known), "Мама")
        for _ in range(7):          # seven more shots of the same person, never indexed
            self.add_photo(face_of(0))

        _face_stats, cl_stats = self.rescan()
        self.assertEqual(cl_stats.clusters, 1)
        self.assertEqual(cl_stats.labels_kept, 0, "3 of 10 files is not a majority")
        self.assertEqual(self.all_labels(), {None})

    def test_a_bare_majority_of_the_files_does_inherit(self):
        """The other side of the same threshold: 6 of 10 files carry the name over."""
        known = self.person(0, 6)
        detect_and_cluster(self.cfg, self.conn, analyzer=self.analyzer)
        label_cluster(self.conn, self.cluster_of(*known), "Мама")
        for _ in range(4):
            self.add_photo(face_of(0))

        _face_stats, cl_stats = self.rescan()
        self.assertEqual((cl_stats.clusters, cl_stats.labels_kept), (1, 1))
        self.assertEqual(self.labels_of(*known), {"Мама"})

    def test_a_rescan_of_a_slice_keeps_the_names_of_the_untouched_rest(self):
        """--limit: the recomputed faces are new, the others are not — both must match."""
        mother = self.person(0, 6)
        father = self.person(1, 6)
        detect_and_cluster(self.cfg, self.conn, analyzer=self.analyzer)
        for files, name in ((mother, "Мама"), (father, "Папа")):
            label_cluster(self.conn, self.cluster_of(*files), name)

        _face_stats, cl_stats = detect_and_cluster(
            self.cfg, self.conn, analyzer=self.analyzer, rescan=True, limit=4)
        self.assertEqual((cl_stats.clusters, cl_stats.labels_kept), (2, 2))
        self.assertEqual(self.labels_of(*mother), {"Мама"})
        self.assertEqual(self.labels_of(*father), {"Папа"})

    def test_without_a_snapshot_the_names_would_be_lost(self):
        """Why detect_and_cluster is the entry point: the halves alone cannot do it.

        Not a wish for this behaviour — a pin on the reason the snapshot exists, so
        that nobody 'simplifies' the pairing away.
        """
        mother = self.person(0, 6)
        self.person(1, 6)
        detect_and_cluster(self.cfg, self.conn, analyzer=self.analyzer)
        label_cluster(self.conn, self.cluster_of(*mother), "Мама")

        detect_faces(self.cfg, self.conn, analyzer=self.analyzer, rescan=True)
        cluster_faces(self.cfg, self.conn)          # no snapshot — nothing to match on
        self.assertEqual(self.all_labels(), {None})


class TestSnapshot(RescanTestCase):
    """`snapshot_clusters` — the state the inheritance is computed against."""

    def test_files_are_grouped_under_the_root_of_the_merge_chain(self):
        left = self.person(0, 3)
        right = self.person(0, 3)
        detect_and_cluster(self.cfg, self.conn, analyzer=self.analyzer)
        a, b = self.split_in_two(left, right)
        root = merge(self.conn, a, b)
        label_cluster(self.conn, a, "Борис")

        snap = snapshot_clusters(self.conn)
        self.assertEqual(set(snap.files), {root})
        self.assertEqual(snap.files[root], set(left + right))
        self.assertEqual(snap.labels[root], "Борис")

    def test_noise_and_markers_are_not_in_the_snapshot(self):
        self.person(0, 3)
        self.add_photo()                      # no faces at all
        detect_and_cluster(self.cfg, self.conn, analyzer=self.analyzer)
        snap = snapshot_clusters(self.conn)
        clustered = {
            r["file_id"] for r in self.conn.execute(
                "SELECT file_id FROM faces WHERE cluster_id IS NOT NULL")
        }
        self.assertEqual(set().union(*snap.files.values()), clustered)

    def test_an_empty_base_gives_an_empty_snapshot(self):
        snap = snapshot_clusters(self.conn)
        self.assertEqual((snap.files, snap.labels), ({}, {}))

    def test_an_empty_snapshot_simply_inherits_nothing(self):
        self.person(0, 4)
        detect_faces(self.cfg, self.conn, analyzer=self.analyzer)
        stats = cluster_faces(self.cfg, self.conn,
                              inherit_from=ClusterSnapshot(labels={}, files={}))
        self.assertEqual((stats.clusters, stats.labels_kept), (1, 0))


class TestCliWiring(unittest.TestCase):
    """`sorta faces --rescan --limit N` reaches detect_and_cluster, and only there."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        root = Path(self.tmp.name)
        (root / "src").mkdir()
        self.cfg_path = root / "config.yaml"
        self.cfg_path.write_text(
            f'sources: ["{(root / "src").as_posix()}"]\n'
            f'database: "{(root / "test.db").as_posix()}"\n',
            encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def call_cmd(self, **kwargs) -> dict:
        captured: dict = {}

        def fake(cfg, conn, progress=None, **kw):
            captured.update(kw)
            return (mock.Mock(files_processed=0, faces_found=0, no_face_files=0, errors=0),
                    mock.Mock(clusters=0, faces=0, noise=0, labels_kept=0, malformed=0))

        with mock.patch.object(cli, "detect_and_cluster", fake), \
                contextlib.redirect_stdout(io.StringIO()):
            cli._cmd_faces(str(self.cfg_path), **kwargs)
        return captured

    def test_default_call_is_incremental(self):
        self.assertEqual(self.call_cmd(), {"rescan": False, "limit": None})

    def test_flags_are_forwarded(self):
        self.assertEqual(self.call_cmd(rescan=True, limit=500),
                         {"rescan": True, "limit": 500})


class TestCliFlags(unittest.TestCase):
    """The typer surface: the flags exist, and --limit is refused on its own."""

    def setUp(self):
        if not hasattr(cli, "app"):  # pragma: no cover — the argparse fallback
            self.skipTest("typer is not installed")
        from typer.testing import CliRunner
        self.runner = CliRunner()

    def invoke(self, args: list[str]):
        with mock.patch.object(cli, "_cmd_faces") as cmd:
            result = self.runner.invoke(cli.app, ["faces", *args])
        return result, cmd

    def test_rescan_and_limit_reach_the_command(self):
        result, cmd = self.invoke(["--rescan", "--limit", "500", "-c", "config.yaml"])
        self.assertEqual(result.exit_code, 0, result.output)
        cmd.assert_called_once_with("config.yaml", rescan=True, limit=500)

    def test_no_flags_keeps_the_incremental_call(self):
        result, cmd = self.invoke(["-c", "config.yaml"])
        self.assertEqual(result.exit_code, 0, result.output)
        cmd.assert_called_once_with("config.yaml", rescan=False, limit=None)

    def test_limit_without_rescan_is_rejected(self):
        result, cmd = self.invoke(["--limit", "500"])
        self.assertNotEqual(result.exit_code, 0)
        cmd.assert_not_called()

    def test_a_non_positive_limit_is_rejected(self):
        for value in ("0", "-5"):
            with self.subTest(value=value):
                result, cmd = self.invoke(["--rescan", "--limit", value])
                self.assertNotEqual(result.exit_code, 0)
                cmd.assert_not_called()

    def test_the_subcommands_still_work_with_the_new_options(self):
        with mock.patch.object(cli, "_cmd_faces") as cmd:
            result = self.runner.invoke(cli.app, ["faces", "--help"])
        self.assertEqual(result.exit_code, 0)
        # A plain substring, and it stays plain: typer styles --help whenever
        # GITHUB_ACTIONS is set, and "--rescan" then arrives split by ANSI escapes.
        # conftest.py turns that styling off for the whole run — see the note there.
        self.assertIn("--rescan", result.output)
        cmd.assert_not_called()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
