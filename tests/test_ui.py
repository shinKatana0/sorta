"""U1: the local plan server — routes, file_id resolution, path traversal, smoke."""
from __future__ import annotations

import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image

from sorta import ui
from sorta.config import Config
from sorta.db import connect
from sorta.hashing import file_hash

_JPEG_MAGIC = b"\xff\xd8"


def _make_jpeg(path: Path, color=(200, 50, 50), size=(64, 48)) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path, "JPEG")
    return path.read_bytes()


class UiServerTestBase(unittest.TestCase):
    """The server does NOT start in setUp: PlanCache is built once on the
    build_server caller's thread (see ui.PlanCache) — fixtures must be added BEFORE
    start_server(), otherwise the plan stays empty (not a bug, but a consequence of
    the thread-safe design: the sqlite3 connection from setUp must not be touched from
    ThreadingHTTPServer threads).
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.src_dir = self.root / "src"
        self.src_dir.mkdir()
        self.cfg = Config(sources=[self.src_dir], database=self.root / "test.db",
                          raw={})
        self.conn = connect(self.cfg.database)
        self._n = 0
        self.server = None
        self.thread = None
        self.base_url = None
        # F42: file_ids from different tests overlap (the counter restarts in each
        # fresh tmp-db) — without a reset the /thumb cache could serve bytes cached
        # under the same (file_id, mtime) in a previous test.
        ui._thumb_cache_clear()

    def tearDown(self):
        if self.server is not None:
            self.server.shutdown()
            self.thread.join(timeout=5)
            self.server.server_close()
        self.conn.close()
        self.tmp.cleanup()

    def start_server(self) -> None:
        self.server = ui.build_server(self.cfg, self.conn, port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    # --- fixtures ------------------------------------------------------

    def add_photo_file(self, rel: str, country: str | None = None,
                       city: str | None = None) -> tuple[int, Path, bytes]:
        self._n += 1
        p = self.src_dir / rel
        content = _make_jpeg(p)
        digest, algo = file_hash(p)
        path = str(p.resolve())
        cur = self.conn.execute(
            """INSERT INTO files (path, size, mtime, ext, media_type, hash, hash_algo,
                   taken_at, taken_at_source, taken_at_confidence, indexed_at)
               VALUES (?, ?, 0, 'jpg', 'photo', ?, ?, '2022-05-01T10:00:00', 'exif',
                       'high', '2026-01-01')""",
            (path, len(content), digest, algo),
        )
        file_id = cur.lastrowid
        if country is not None or city is not None:
            self.conn.execute(
                """INSERT INTO places (file_id, country, region, city, confidence,
                       updated_at)
                   VALUES (?, ?, NULL, ?, 'exact_gps', '2026-01-01')""",
                (file_id, country, city))
        self.conn.commit()
        return file_id, p, content

    def get(self, path: str) -> tuple[int, bytes, str]:
        """(status, body, content_type); does not raise on 4xx/5xx — like urllib.request."""
        try:
            with urllib.request.urlopen(f"{self.base_url}{path}", timeout=5) as resp:
                return resp.status, resp.read(), resp.headers.get("Content-Type", "")
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read(), exc.headers.get("Content-Type", "")


class TestServerSmoke(UiServerTestBase):
    def test_binds_localhost_only(self):
        self.start_server()
        self.assertEqual(self.server.server_address[0], "127.0.0.1")

    def test_starts_and_serves_index(self):
        self.start_server()
        status, body, ctype = self.get("/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", ctype)
        self.assertIn(b"<html", body.lower())

    def test_stops_cleanly(self):
        self.start_server()
        # tearDown does shutdown/close; here we just make sure the server accepts a
        # request before stopping (the stop itself is verified in tearDown by the
        # absence of exceptions).
        status, _body, _ctype = self.get("/")
        self.assertEqual(status, 200)


class TestApiPlan(UiServerTestBase):
    def test_city_mode_returns_expected_fields(self):
        fid1, _p1, _c1 = self.add_photo_file("a.jpg", country="ru", city="Moscow")
        fid2, _p2, _c2 = self.add_photo_file("b.jpg", country="ru", city="Moscow")
        self.start_server()
        status, body, ctype = self.get("/api/plan?mode=city")
        self.assertEqual(status, 200)
        self.assertIn("application/json", ctype)
        items = json.loads(body)
        self.assertEqual({it["file_id"] for it in items}, {fid1, fid2})
        expected_keys = {"file_id", "name", "target_rel", "reason", "date",
                         "geo", "category", "thumb_url"}
        for item in items:
            self.assertEqual(expected_keys, set(item.keys()))
            self.assertEqual(item["thumb_url"], f"/thumb/{item['file_id']}")

    def test_unsupported_mode_returns_400(self):
        self.add_photo_file("a.jpg", country="ru", city="Moscow")
        self.start_server()
        status, body, _ctype = self.get("/api/plan?mode=unknown_mode")
        self.assertEqual(status, 400)
        payload = json.loads(body)
        self.assertIn("error", payload)

    def test_missing_mode_returns_400(self):
        self.start_server()
        status, _body, _ctype = self.get("/api/plan")
        self.assertEqual(status, 400)

    def test_plan_cached_not_recomputed_after_start(self):
        self.add_photo_file("a.jpg", country="ru", city="Moscow")
        self.start_server()
        status1, body1, _ = self.get("/api/plan?mode=city")
        # adding a file AFTER the server starts must not enter the cache —
        # PlanCache is built once at build_server and not recomputed afterwards.
        self.add_photo_file("b.jpg", country="ru", city="Moscow")
        status2, body2, _ = self.get("/api/plan?mode=city")
        self.assertEqual(status1, 200)
        self.assertEqual(status2, 200)
        self.assertEqual(len(json.loads(body1)), 1)
        self.assertEqual(len(json.loads(body1)), len(json.loads(body2)))


class TestThumbAndPhoto(UiServerTestBase):
    def test_thumb_resolves_id(self):
        fid, _p, _content = self.add_photo_file("a.jpg", country="ru", city="Moscow")
        self.start_server()
        status, body, ctype = self.get(f"/thumb/{fid}")
        self.assertEqual(status, 200)
        self.assertEqual(ctype, "image/jpeg")
        self.assertTrue(body.startswith(_JPEG_MAGIC))

    def test_photo_resolves_id_to_original_bytes(self):
        fid, _p, content = self.add_photo_file("a.jpg", country="ru", city="Moscow")
        self.start_server()
        status, body, ctype = self.get(f"/photo/{fid}")
        self.assertEqual(status, 200)
        self.assertIn("jpeg", ctype)
        self.assertEqual(body, content)

    def test_thumb_unknown_id_404(self):
        self.start_server()
        status, _body, _ctype = self.get("/thumb/999999")
        self.assertEqual(status, 404)

    def test_photo_unknown_id_404(self):
        self.start_server()
        status, _body, _ctype = self.get("/photo/999999")
        self.assertEqual(status, 404)

    def test_thumb_non_numeric_id_404(self):
        self.start_server()
        status, _body, _ctype = self.get("/thumb/not-a-number")
        self.assertEqual(status, 404)


class TestPathTraversal(UiServerTestBase):
    def test_thumb_path_traversal_not_resolved(self):
        # ../.. in a segment does not parse as a file_id -> 404, a file outside the
        # index is never read.
        self.start_server()
        status, _body, _ctype = self.get("/thumb/..%2f..%2f..%2fconfig.yaml")
        self.assertEqual(status, 404)

    def test_photo_path_traversal_not_resolved(self):
        self.start_server()
        status, body, _ctype = self.get("/photo/..%2f..%2fpyproject.toml")
        self.assertEqual(status, 404)
        self.assertNotIn(b"[project]", body)

    def test_photo_absolute_path_not_resolved(self):
        self.start_server()
        # an absolute path as an "id" — also does not parse to int, also 404.
        quoted = urllib.parse.quote(str(self.cfg.database), safe="")
        status, _body, _ctype = self.get(f"/photo/{quoted}")
        self.assertEqual(status, 404)


if __name__ == "__main__":
    unittest.main()
