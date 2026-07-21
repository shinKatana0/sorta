"""Faces: detection (mock analyzer), clustering of synthetic data, labels, merge, sheet."""
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np

from sorta.config import Config, FacesConfig
from sorta.db import connect
from sorta.faces import (
    EMBED_DIM,
    cluster_faces,
    detect_and_cluster,
    detect_faces,
    export_contact_sheet,
    label_cluster,
    merge,
    resolve_root,
)

RNG = np.random.default_rng(42)


def unit(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v)).astype(np.float32)


def group(axis: int, n: int) -> list[np.ndarray]:
    """n close unit vectors around the axis basis vector (one "person")."""
    center = np.zeros(EMBED_DIM, dtype=np.float64)
    center[axis] = 1.0
    return [unit(center + 0.05 * RNG.normal(size=EMBED_DIM)) for _ in range(n)]


def lone(axis: int) -> np.ndarray:
    """A single vector orthogonal to the groups — deliberate noise."""
    v = np.zeros(EMBED_DIM, dtype=np.float64)
    v[axis] = 1.0
    return unit(v)


class FacesTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = Config(sources=[Path(self.tmp.name)],
                          database=Path(self.tmp.name) / "test.db")
        self.conn = connect(self.cfg.database)
        self._n = 0

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def add_file(self, media_type="photo", dup_of=None, error=None, orientation=None):
        self._n += 1
        cur = self.conn.execute(
            """INSERT INTO files (path, size, mtime, ext, media_type, orientation,
                   dup_of, error, indexed_at)
               VALUES (?, 1000, 0, 'jpg', ?, ?, ?, ?, '2026-01-01')""",
            (f"/photos/img_{self._n}.jpg", media_type, orientation, dup_of, error),
        )
        self.conn.commit()
        return cur.lastrowid, f"/photos/img_{self._n}.jpg"

    def add_face(self, file_id, emb, cluster_id=None):
        cur = self.conn.execute(
            "INSERT INTO faces (file_id, bbox, embedding, cluster_id) VALUES (?, ?, ?, ?)",
            (file_id, "[0, 0, 100, 100]", np.asarray(emb, dtype="<f4").tobytes(), cluster_id),
        )
        self.conn.commit()
        return cur.lastrowid

    def faces_rows(self):
        return self.conn.execute(
            "SELECT id, file_id, bbox, embedding, cluster_id FROM faces ORDER BY id"
        ).fetchall()

    def cluster_of_face(self, face_id):
        return self.conn.execute(
            "SELECT cluster_id FROM faces WHERE id = ?", (face_id,)).fetchone()[0]


def hit(emb, size=120.0, score=0.95):
    return ([10.0, 10.0, 10.0 + size, 10.0 + size], score, emb)


class TestDetect(FacesTestCase):
    def test_saves_faces_and_no_face_marker(self):
        f1, p1 = self.add_file()
        f2, p2 = self.add_file()
        e1, e2 = unit(RNG.normal(size=EMBED_DIM)), unit(RNG.normal(size=EMBED_DIM))
        hits = {p1: [hit(e1), hit(e2)], p2: []}
        stats = detect_faces(self.cfg, self.conn,
                             analyzer=lambda path, orient: hits[path])
        self.assertEqual((stats.files_total, stats.files_processed), (2, 2))
        self.assertEqual((stats.faces_found, stats.no_face_files, stats.errors), (2, 1, 0))
        rows = self.faces_rows()
        self.assertEqual(len(rows), 3)
        # embedding: 512 float32 little-endian, restored losslessly
        got = np.frombuffer(rows[0]["embedding"], dtype="<f4")
        self.assertEqual(got.shape, (EMBED_DIM,))
        np.testing.assert_array_equal(got, e1)
        self.assertEqual(rows[0]["bbox"], "[10.0, 10.0, 130.0, 130.0]")
        # the "no faces" marker
        marker = [r for r in rows if r["file_id"] == f2]
        self.assertEqual([(r["bbox"], r["embedding"]) for r in marker], [("[]", b"")])

    def test_quality_filters_small_and_unsure(self):
        _, p = self.add_file()
        e = unit(RNG.normal(size=EMBED_DIM))
        hits = [hit(e, size=30.0),          # smaller than min_face_px=40
                hit(e, score=0.5),          # below det_threshold=0.7
                hit(e, size=200.0)]         # passes
        stats = detect_faces(self.cfg, self.conn, analyzer=lambda path, orient: hits)
        self.assertEqual(stats.faces_found, 1)
        self.assertEqual(len(self.faces_rows()), 1)

    def test_thresholds_come_from_config(self):
        self.cfg.faces = FacesConfig(min_face_px=150, det_threshold=0.99)
        _, p = self.add_file()
        e = unit(RNG.normal(size=EMBED_DIM))
        stats = detect_faces(self.cfg, self.conn,
                             analyzer=lambda path, orient: [hit(e, size=120.0, score=0.95)])
        self.assertEqual((stats.faces_found, stats.no_face_files), (0, 1))

    def test_incremental_skips_processed_files(self):
        _, p1 = self.add_file()
        f2, _ = self.add_file()
        calls = []

        def analyzer(path, orient):
            calls.append(path)
            return [hit(unit(RNG.normal(size=EMBED_DIM)))] if path == p1 else []

        detect_faces(self.cfg, self.conn, analyzer=analyzer)
        self.assertEqual(len(calls), 2)
        stats2 = detect_faces(self.cfg, self.conn, analyzer=analyzer)
        # repeated run: both the file with faces and the marker file are not recomputed
        self.assertEqual(len(calls), 2)
        self.assertEqual((stats2.files_total, len(self.faces_rows())), (0, 2))

    def test_skips_videos_duplicates_and_broken(self):
        canon, p = self.add_file()
        self.add_file(media_type="video")
        self.add_file(dup_of=canon)
        self.add_file(error="boom")
        seen = []
        detect_faces(self.cfg, self.conn,
                     analyzer=lambda path, orient: seen.append(path) or [])
        self.assertEqual(seen, [p])

    def test_analyzer_error_counted_and_retried(self):
        _, p = self.add_file()

        def broken(path, orient):
            raise ValueError("unreadable")

        stats = detect_faces(self.cfg, self.conn, analyzer=broken)
        self.assertEqual((stats.errors, len(self.faces_rows())), (1, 0))
        # no rows — the next run will try the file again
        stats2 = detect_faces(self.cfg, self.conn,
                              analyzer=lambda path, orient: [])
        self.assertEqual(stats2.files_processed, 1)


class TestCluster(FacesTestCase):
    def seed_groups(self, sizes=(8, 8, 6), noise=3):
        """Synthetic: len(sizes) "people" + noise singletons. Returns ids by group."""
        ids = []
        for g_idx, n in enumerate(sizes):
            ids.append([self.add_face(self.add_file()[0], e) for e in group(g_idx, n)])
        noise_ids = [self.add_face(self.add_file()[0], lone(100 + i)) for i in range(noise)]
        return ids, noise_ids

    def test_three_groups_plus_noise(self):
        ids, noise_ids = self.seed_groups()
        stats = cluster_faces(self.cfg, self.conn)
        self.assertEqual((stats.faces, stats.clusters, stats.noise), (25, 3, 3))
        for members in ids:
            cids = {self.cluster_of_face(fid) for fid in members}
            self.assertEqual(len(cids), 1, "a group must fall into one cluster")
            self.assertIsNotNone(cids.pop())
        # three groups — three distinct clusters
        self.assertEqual(len({self.cluster_of_face(m[0]) for m in ids}), 3)
        for fid in noise_ids:
            self.assertIsNone(self.cluster_of_face(fid))

    def test_min_cluster_size_from_config(self):
        small = [self.add_face(self.add_file()[0], e) for e in group(0, 3)]
        big = [self.add_face(self.add_file()[0], e) for e in group(1, 6)]
        stats = cluster_faces(self.cfg, self.conn)  # default min_cluster_size=5
        self.assertEqual(stats.clusters, 1)
        self.assertTrue(all(self.cluster_of_face(f) is None for f in small))
        self.assertTrue(all(self.cluster_of_face(f) is not None for f in big))
        self.cfg.faces = FacesConfig(min_cluster_size=2)
        stats = cluster_faces(self.cfg, self.conn)
        self.assertEqual(stats.clusters, 2)
        self.assertTrue(all(self.cluster_of_face(f) is not None for f in small))

    def test_recluster_preserves_labels(self):
        ids, _ = self.seed_groups(sizes=(8, 8), noise=0)
        cluster_faces(self.cfg, self.conn)
        anna_cluster = self.cluster_of_face(ids[0][0])
        label_cluster(self.conn, anna_cluster, "Анна")
        # new faces of Anna appeared and a new person
        for e in group(0, 3):
            self.add_face(self.add_file()[0], e)
        for e in group(2, 6):
            self.add_face(self.add_file()[0], e)
        stats = cluster_faces(self.cfg, self.conn)
        self.assertEqual(stats.clusters, 3)
        self.assertEqual(stats.labels_kept, 1)
        new_cluster = self.cluster_of_face(ids[0][0])
        label = self.conn.execute(
            "SELECT label FROM face_clusters WHERE id = ?", (new_cluster,)).fetchone()[0]
        self.assertEqual(label, "Анна")
        # old clusters do not accumulate: the table has exactly as many rows as clusters
        n_rows = self.conn.execute("SELECT COUNT(*) FROM face_clusters").fetchone()[0]
        self.assertEqual(n_rows, 3)

    def test_recluster_preserves_label_of_merged_pair(self):
        # two clusters merged manually and named: after recomputation, when HDBSCAN
        # merges their faces itself, the root label must be preserved
        a = [self.add_face(self.add_file()[0], e) for e in group(0, 5)]
        b = [self.add_face(self.add_file()[0], e) for e in group(0, 5)]
        ca = self.conn.execute("INSERT INTO face_clusters (label) VALUES (NULL)").lastrowid
        cb = self.conn.execute("INSERT INTO face_clusters (label) VALUES (NULL)").lastrowid
        for fid in a:
            self.conn.execute("UPDATE faces SET cluster_id=? WHERE id=?", (ca, fid))
        for fid in b:
            self.conn.execute("UPDATE faces SET cluster_id=? WHERE id=?", (cb, fid))
        self.conn.commit()
        merge(self.conn, ca, cb)
        label_cluster(self.conn, ca, "Борис")
        stats = cluster_faces(self.cfg, self.conn)
        self.assertEqual((stats.clusters, stats.labels_kept), (1, 1))
        clustered = {self.cluster_of_face(fid) for fid in a + b} - {None}
        self.assertEqual(len(clustered), 1)
        label = self.conn.execute(
            "SELECT label FROM face_clusters WHERE id = ?", (clustered.pop(),)).fetchone()[0]
        self.assertEqual(label, "Борис")

    def test_markers_do_not_participate(self):
        fid, p = self.add_file()
        self.conn.execute(
            "INSERT INTO faces (file_id, bbox, embedding) VALUES (?, '[]', ?)", (fid, b""))
        self.conn.commit()
        stats = cluster_faces(self.cfg, self.conn)
        self.assertEqual(stats.faces, 0)

    def add_malformed_face(self):
        """A faces row with a wrong-length embedding (not EMBED_DIM*4 bytes)."""
        fid, _ = self.add_file()
        cur = self.conn.execute(
            "INSERT INTO faces (file_id, bbox, embedding) VALUES (?, ?, ?)",
            (fid, "[0, 0, 100, 100]", b"\x00"),
        )
        self.conn.commit()
        return cur.lastrowid

    def test_malformed_embedding_does_not_crash_clustering(self):
        ids, noise_ids = self.seed_groups()
        bad_id = self.add_malformed_face()
        stats = cluster_faces(self.cfg, self.conn)
        self.assertEqual(stats.malformed, 1)
        self.assertEqual((stats.clusters, stats.noise), (3, 3))
        self.assertIsNone(self.cluster_of_face(bad_id))
        for members in ids:
            cids = {self.cluster_of_face(fid) for fid in members}
            self.assertEqual(len(cids), 1)
            self.assertIsNotNone(cids.pop())

    def test_only_malformed_embeddings_yields_no_clusters(self):
        bad_id = self.add_malformed_face()
        stats = cluster_faces(self.cfg, self.conn)
        self.assertEqual((stats.clusters, stats.malformed), (0, 1))
        self.assertIsNone(self.cluster_of_face(bad_id))
        n_rows = self.conn.execute("SELECT COUNT(*) FROM face_clusters").fetchone()[0]
        self.assertEqual(n_rows, 0)

    def test_detect_and_cluster_end_to_end(self):
        people = [group(0, 6), group(1, 6)]
        paths = {}
        for person in people:
            for e in person:
                _, p = self.add_file()
                paths[p] = [hit(e)]
        face_stats, cl_stats = detect_and_cluster(
            self.cfg, self.conn, analyzer=lambda path, orient: paths[path])
        self.assertEqual(face_stats.faces_found, 12)
        self.assertEqual((cl_stats.clusters, cl_stats.noise), (2, 0))


class TestManualOps(FacesTestCase):
    def new_cluster(self, label=None):
        cur = self.conn.execute("INSERT INTO face_clusters (label) VALUES (?)", (label,))
        self.conn.commit()
        return cur.lastrowid

    def test_merge_chain_resolves_to_root(self):
        a, b, c = self.new_cluster(), self.new_cluster(), self.new_cluster()
        merge(self.conn, a, b)
        merge(self.conn, b, c)
        self.assertEqual(resolve_root(self.conn, a), c)
        self.assertEqual(resolve_root(self.conn, b), c)
        self.assertEqual(resolve_root(self.conn, c), c)

    def test_merge_same_root_is_noop(self):
        a, b = self.new_cluster(), self.new_cluster()
        merge(self.conn, a, b)
        self.assertEqual(merge(self.conn, a, b), b)
        self.assertEqual(merge(self.conn, b, b), b)

    def test_merge_keeps_label(self):
        a, b = self.new_cluster(label="Вера"), self.new_cluster()
        merge(self.conn, a, b)
        label = self.conn.execute(
            "SELECT label FROM face_clusters WHERE id = ?", (b,)).fetchone()[0]
        self.assertEqual(label, "Вера")

    def test_merge_does_not_overwrite_dst_label(self):
        a, b = self.new_cluster(label="Вера"), self.new_cluster(label="Глеб")
        merge(self.conn, a, b)
        label = self.conn.execute(
            "SELECT label FROM face_clusters WHERE id = ?", (b,)).fetchone()[0]
        self.assertEqual(label, "Глеб")

    def test_label_goes_to_root_of_chain(self):
        a, b = self.new_cluster(), self.new_cluster()
        merge(self.conn, a, b)
        self.assertEqual(label_cluster(self.conn, a, "Дина"), b)
        label = self.conn.execute(
            "SELECT label FROM face_clusters WHERE id = ?", (b,)).fetchone()[0]
        self.assertEqual(label, "Дина")

    def test_unknown_cluster_raises(self):
        with self.assertRaises(ValueError):
            resolve_root(self.conn, 999)


class TestContactSheet(FacesTestCase):
    def test_sheet_includes_merged_members_and_escapes(self):
        ca = self.conn.execute("INSERT INTO face_clusters (label) VALUES ('О''Хара & Ко')")
        ca = ca.lastrowid
        cb = self.conn.execute("INSERT INTO face_clusters (label) VALUES (NULL)").lastrowid
        f1, p1 = self.add_file()
        f2, p2 = self.add_file()
        self.add_face(f1, unit(RNG.normal(size=EMBED_DIM)), cluster_id=ca)
        self.add_face(f2, unit(RNG.normal(size=EMBED_DIM)), cluster_id=cb)
        merge(self.conn, cb, ca)
        out = Path(self.tmp.name) / "sheet.html"
        n = export_contact_sheet(self.conn, cb, out)  # a merged id is accepted too
        self.assertEqual(n, 2)
        html = out.read_text(encoding="utf-8")
        self.assertIn("img_1.jpg", html)
        self.assertIn("img_2.jpg", html)   # a face from the merged cluster is present
        self.assertIn("file://", html)
        self.assertIn("О&#x27;Хара &amp; Ко", html)
        self.assertNotIn("<script", html)


class TestReadImage(unittest.TestCase):
    def test_decodes_jpeg_via_cv2(self):
        from PIL import Image

        from sorta.faces import _read_image_bgr
        with tempfile.TemporaryDirectory() as tmp:
            p = str(Path(tmp) / "img.jpg")
            Image.new("RGB", (32, 24), (200, 100, 50)).save(p, "JPEG")
            img = _read_image_bgr(p)
            self.assertEqual(img.shape, (24, 32, 3))

    def test_decodes_heic_via_pillow_fallback(self):
        # HEIC is not handled by cv2 — we check the fallback to pillow-heif
        try:
            import pillow_heif
        except ImportError:
            self.skipTest("pillow-heif not installed")
        from PIL import Image

        from sorta.faces import _read_image_bgr
        pillow_heif.register_heif_opener()
        with tempfile.TemporaryDirectory() as tmp:
            p = str(Path(tmp) / "iphone.heic")
            Image.new("RGB", (48, 32), (10, 220, 30)).save(p, "HEIF")
            img = _read_image_bgr(p)
            self.assertEqual(img.shape, (32, 48, 3))

    def test_undecodable_raises(self):
        from sorta.faces import _read_image_bgr
        with tempfile.TemporaryDirectory() as tmp:
            p = str(Path(tmp) / "broken.jpg")
            Path(p).write_bytes(b"not an image")
            with self.assertRaises(ValueError):
                _read_image_bgr(p)


class TestDecodeWorkers(unittest.TestCase):
    def test_default_is_min_8_cpu_count(self):
        import os

        from sorta.faces import _decode_workers
        cfg = Config(sources=[Path(".")], database=Path("x.db"))
        self.assertEqual(_decode_workers(cfg), min(8, os.cpu_count() or 4))

    def test_config_override_from_raw(self):
        from sorta.faces import _decode_workers
        cfg = Config(sources=[Path(".")], database=Path("x.db"), raw={"faces": {"decode_workers": 3}})
        self.assertEqual(_decode_workers(cfg), 3)


class TestPrefetchDecode(unittest.TestCase):
    def test_all_rows_yielded_order_by_readiness_not_input(self):
        from sorta.faces import _prefetch_decode
        rows = [{"id": i, "path": str(i), "orientation": None} for i in range(16)]

        def decode(path, orientation):
            # the smaller the id, the longer the decode — later-submitted ones overtake
            time.sleep(0.002 * (16 - int(path)))
            return int(path)

        results = list(_prefetch_decode(rows, decode, max_workers=4))
        self.assertEqual(sorted(r["id"] for r, _img, _err in results), list(range(16)))
        order = [r["id"] for r, _img, _err in results]
        self.assertNotEqual(order, list(range(16)),
                            "the order must be determined by decode readiness, not by input order")

    def test_bounded_window_backpressure(self):
        from sorta.faces import _prefetch_decode
        rows = [{"id": i, "path": str(i), "orientation": None} for i in range(12)]
        lock = threading.Lock()
        started = 0

        def decode(path, orientation):
            nonlocal started
            with lock:
                started += 1
            time.sleep(0.01)
            return path

        max_workers = 2
        window = max_workers * 2
        consumed = 0
        for _r, _img, _err in _prefetch_decode(rows, decode, max_workers):
            consumed += 1
            with lock:
                in_flight = started - consumed
            self.assertLessEqual(in_flight, window)
            time.sleep(0.02)  # slow "inference" — we check the decode does not race ahead

    def test_broken_frame_counted_and_pipeline_continues(self):
        from sorta.faces import _prefetch_decode
        rows = [{"id": i, "path": f"{i}.jpg", "orientation": None} for i in range(8)]

        def decode(path, orientation):
            if path == "3.jpg":
                raise ValueError("corrupt frame")
            return path

        results = list(_prefetch_decode(rows, decode, max_workers=3))
        self.assertEqual(len(results), 8)
        errors = {r["id"]: err for r, _img, err in results}
        self.assertIsInstance(errors[3], ValueError)
        self.assertTrue(all(errors[i] is None for i in range(8) if i != 3))


class TestOrientation(unittest.TestCase):
    def test_exif_rotations(self):
        from sorta.faces import _apply_orientation
        img = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
        np.testing.assert_array_equal(_apply_orientation(img, None), img)
        np.testing.assert_array_equal(_apply_orientation(img, 1), img)
        np.testing.assert_array_equal(_apply_orientation(img, 3), np.rot90(img, 2))
        np.testing.assert_array_equal(_apply_orientation(img, 6), np.rot90(img, 3))
        np.testing.assert_array_equal(_apply_orientation(img, 8), np.rot90(img, 1))


class TestSmokeML(unittest.TestCase):
    """The only ML test: real insightface, only if the model is already downloaded."""

    @staticmethod
    def _model_ready() -> bool:
        try:
            import insightface  # noqa: F401
        except ImportError:
            return False
        model_dir = Path.home() / ".insightface" / "models" / "buffalo_l"
        return model_dir.is_dir() and any(model_dir.glob("*.onnx"))

    @unittest.skipUnless(_model_ready(), "insightface/buffalo_l not installed locally")
    def test_analyzer_runs_on_blank_image(self):  # pragma: no cover
        import cv2

        from sorta.faces import FacesSettings, _insightface_analyzer
        with tempfile.TemporaryDirectory() as tmp:
            img_path = str(Path(tmp) / "blank.jpg")
            cv2.imwrite(img_path, np.full((128, 128, 3), 128, dtype=np.uint8))
            analyze = _insightface_analyzer(FacesSettings())
            hits = analyze(img_path, None)
            self.assertIsInstance(hits, list)  # no faces on a gray square
            self.assertEqual(hits, [])

    @unittest.skipUnless(_model_ready(), "insightface/buffalo_l not installed locally")
    def test_allowed_modules_does_not_change_embeddings(self):  # pragma: no cover
        """F47: embeddings with/without allowed_modules must match (recognition
        aligns by kps from detection, not by landmark/genderage).

        Needs a folder with real face photos (the unit path mocks the analyzer and
        synthetic data does not check this) — set via SORTA_FACES_SMOKE_DIR. Without
        the variable it is skipped: run it manually on a GPU, pointing at a path with
        several real frames containing faces.
        """
        sample_dir = os.environ.get("SORTA_FACES_SMOKE_DIR")
        if not sample_dir:
            self.skipTest("SORTA_FACES_SMOKE_DIR not set — no real photos for the smoke")
        exts = {".jpg", ".jpeg", ".png", ".heic"}
        paths = [str(p) for p in Path(sample_dir).iterdir() if p.suffix.lower() in exts]
        if not paths:
            self.skipTest(f"no images in {sample_dir}")

        from sorta.faces import compare_allowed_modules_embeddings
        report = compare_allowed_modules_embeddings(paths)
        self.assertGreater(report.faces_compared, 0, "no faces found in the test frames")
        self.assertEqual(report.mismatched_face_counts, [])
        for path, idx, cos in report.cosines:
            self.assertGreaterEqual(cos, 0.999, f"{path} face {idx}: cosine={cos}")


if __name__ == "__main__":
    unittest.main()
