"""F31: the People tab = cluster management — /api/clusters, label, merge."""
from __future__ import annotations

import json
import unittest
import urllib.error
import urllib.request

from tests.test_ui import UiServerTestBase


class ClustersTestBase(UiServerTestBase):
    """Cluster/face fixtures on top of the base U1 server."""

    def add_cluster(self, *, label: str | None = None, merged_into: int | None = None) -> int:
        cur = self.conn.execute(
            "INSERT INTO face_clusters (label, merged_into) VALUES (?, ?)",
            (label, merged_into),
        )
        self.conn.commit()
        return cur.lastrowid

    def add_face(self, file_id: int, cluster_id: int | None) -> int:
        cur = self.conn.execute(
            """INSERT INTO faces (file_id, bbox, embedding, cluster_id)
               VALUES (?, '[0,0,10,10]', ?, ?)""",
            (file_id, b"embedding", cluster_id),
        )
        self.conn.commit()
        return cur.lastrowid

    def post(self, path: str, data: dict) -> tuple[int, dict]:
        body = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}{path}", data=body, method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())


class TestApiClustersGet(ClustersTestBase):
    def test_root_clusters_with_size_label_samples(self):
        fid1, _p1, _c1 = self.add_photo_file("a.jpg")
        fid2, _p2, _c2 = self.add_photo_file("b.jpg")
        cluster = self.add_cluster(label="Alice")
        self.add_face(fid1, cluster)
        self.add_face(fid2, cluster)
        self.start_server()
        status, body, ctype = self.get("/api/clusters")
        self.assertEqual(status, 200)
        self.assertIn("application/json", ctype)
        clusters = json.loads(body)
        self.assertEqual(len(clusters), 1)
        c = clusters[0]
        self.assertEqual(c["cluster_id"], cluster)
        self.assertEqual(c["label"], "Alice")
        self.assertEqual(c["size"], 2)
        self.assertEqual(set(c["samples"]), {fid1, fid2})

    def test_sorted_by_size_descending(self):
        fid1, _p1, _c1 = self.add_photo_file("a.jpg")
        fid2, _p2, _c2 = self.add_photo_file("b.jpg")
        fid3, _p3, _c3 = self.add_photo_file("c.jpg")
        small = self.add_cluster(label=None)
        big = self.add_cluster(label=None)
        self.add_face(fid1, small)
        self.add_face(fid2, big)
        self.add_face(fid3, big)
        self.start_server()
        status, body, _ctype = self.get("/api/clusters")
        self.assertEqual(status, 200)
        clusters = json.loads(body)
        self.assertEqual([c["cluster_id"] for c in clusters], [big, small])
        self.assertEqual([c["size"] for c in clusters], [2, 1])

    def test_noise_faces_excluded(self):
        fid1, _p1, _c1 = self.add_photo_file("a.jpg")
        cluster = self.add_cluster(label="Bob")
        self.add_face(fid1, cluster)
        fid_noise, _p2, _c2 = self.add_photo_file("noise.jpg")
        self.add_face(fid_noise, None)
        self.start_server()
        status, body, _ctype = self.get("/api/clusters")
        self.assertEqual(status, 200)
        clusters = json.loads(body)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0]["size"], 1)

    def test_merged_cluster_not_shown_but_counted_in_root_size(self):
        fid1, _p1, _c1 = self.add_photo_file("a.jpg")
        fid2, _p2, _c2 = self.add_photo_file("b.jpg")
        dst = self.add_cluster(label="Carol")
        src = self.add_cluster(label=None, merged_into=dst)
        self.add_face(fid1, dst)
        self.add_face(fid2, src)
        self.start_server()
        status, body, _ctype = self.get("/api/clusters")
        self.assertEqual(status, 200)
        clusters = json.loads(body)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0]["cluster_id"], dst)
        self.assertEqual(clusters[0]["size"], 2)

    def test_samples_limited_to_six(self):
        cluster = self.add_cluster(label="Dan")
        for i in range(8):
            fid, _p, _c = self.add_photo_file(f"p{i}.jpg")
            self.add_face(fid, cluster)
        self.start_server()
        status, body, _ctype = self.get("/api/clusters")
        self.assertEqual(status, 200)
        clusters = json.loads(body)
        self.assertEqual(clusters[0]["size"], 8)
        self.assertEqual(len(clusters[0]["samples"]), 6)

    def test_no_clusters_returns_empty_list(self):
        self.start_server()
        status, body, _ctype = self.get("/api/clusters")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), [])


class TestApiClustersLabel(ClustersTestBase):
    def test_label_sets_name_in_db(self):
        cluster = self.add_cluster(label=None)
        self.start_server()
        status, payload = self.post(
            "/api/clusters/label", {"cluster_id": cluster, "name": "Eve"})
        self.assertEqual(status, 200)
        self.assertTrue(payload.get("ok"))
        row = self.conn.execute(
            "SELECT label FROM face_clusters WHERE id = ?", (cluster,)).fetchone()
        self.assertEqual(row["label"], "Eve")

    def test_empty_name_returns_400(self):
        cluster = self.add_cluster(label=None)
        self.start_server()
        status, payload = self.post(
            "/api/clusters/label", {"cluster_id": cluster, "name": "   "})
        self.assertEqual(status, 400)
        self.assertIn("error", payload)

    def test_missing_name_returns_400(self):
        cluster = self.add_cluster(label=None)
        self.start_server()
        status, payload = self.post("/api/clusters/label", {"cluster_id": cluster})
        self.assertEqual(status, 400)
        self.assertIn("error", payload)

    def test_unknown_cluster_id_returns_404(self):
        self.start_server()
        status, payload = self.post(
            "/api/clusters/label", {"cluster_id": 999999, "name": "Frank"})
        self.assertEqual(status, 404)
        self.assertIn("error", payload)

    def test_non_int_cluster_id_returns_400(self):
        self.start_server()
        status, payload = self.post(
            "/api/clusters/label", {"cluster_id": "1", "name": "Frank"})
        self.assertEqual(status, 400)
        self.assertIn("error", payload)


class TestApiClustersMerge(ClustersTestBase):
    def test_merge_sets_merged_into_on_source_root(self):
        src = self.add_cluster(label=None)
        dst = self.add_cluster(label="Grace")
        self.start_server()
        status, payload = self.post("/api/clusters/merge", {"src": src, "dst": dst})
        self.assertEqual(status, 200)
        self.assertTrue(payload.get("ok"))
        row = self.conn.execute(
            "SELECT merged_into FROM face_clusters WHERE id = ?", (src,)).fetchone()
        self.assertEqual(row["merged_into"], dst)

    def test_unknown_id_returns_4xx(self):
        dst = self.add_cluster(label=None)
        self.start_server()
        status, payload = self.post("/api/clusters/merge", {"src": 999999, "dst": dst})
        self.assertIn(status, (400, 404))
        self.assertIn("error", payload)

    def test_invalid_body_returns_400(self):
        self.start_server()
        status, payload = self.post("/api/clusters/merge", {"src": "a", "dst": 1})
        self.assertEqual(status, 400)
        self.assertIn("error", payload)


class TestIndexHtmlPeopleTab(ClustersTestBase):
    def test_people_tab_has_cluster_grid_and_no_external_resources(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn('id="tab-btn-person"', html)
        self.assertIn(">People<", html)
        self.assertIn('id="clusters-grid"', html)
        self.assertIn('id="clusters-merge-btn"', html)
        self.assertIn("loadClusters", html)
        self.assertIn("/api/clusters", html)
        self.assertNotIn("http://", html)
        self.assertNotIn("https://", html)
        self.assertNotIn("<link", html)


if __name__ == "__main__":
    unittest.main()
