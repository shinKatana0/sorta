"""F42: the cluster-preview lightbox (click -> /photo/<id>, Esc/background closes) +
thumbnail perf (/thumb cache + skeletons/lazy loading) + the "Start over" button
(POST /api/process/reset -> db.reset_index)."""
from __future__ import annotations

import json
import threading
import unittest
import urllib.error
import urllib.request
from unittest import mock

from sorta import imaging

from tests.test_ui import UiServerTestBase
from tests.test_ui_process import ProcessTestBase


class TestLightboxHtml(UiServerTestBase):
    def test_lightbox_overlay_present_in_html(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn('id="lightbox" class="lightbox" hidden', html)
        self.assertIn('id="lightbox-img"', html)

    def test_cluster_thumb_click_opens_lightbox_via_preview_endpoint(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn("openLightbox", html)
        # the lightbox loads the large DECODED /preview (HEIC/RAW too), not /photo
        self.assertIn('"/preview/" + lightboxSamples[index]', html)
        # a wrapper/click handler over the cluster preview (not N hidden overlays)
        self.assertIn('img.addEventListener("click", function () { openLightbox(', html)

    def test_escape_closes_lightbox_in_js(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn("closeLightbox", html)
        self.assertIn('e.key === "Escape"', html)
        # a click on the overlay background also closes it
        self.assertIn("lightboxEl.addEventListener(\"click\", closeLightbox)", html)

    def test_lightbox_img_overrides_global_thumb_size(self):
        # regression: the global `img { width:56px; height:56px }` (thumbnail default)
        # used to shrink the lightbox image to 56px — `.lightbox img` must reset
        # width/height to auto, otherwise max-width/height do not scale.
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertRegex(html, r"\.lightbox img \{[^}]*width: auto;[^}]*height: auto;")
        self.assertRegex(html, r"\.lightbox img \{[^}]*max-width: 100%;")

    def test_unified_clickable_thumb_opens_lightbox_not_new_tab(self):
        # uniform behaviour: a click on a thumbnail everywhere opens the lightbox via
        # the shared clickableThumb, not a new tab with the raw /photo.
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn("function clickableThumb(", html)
        self.assertIn("openLightbox(samples || [fileId], index || 0)", html)
        # the lists (Cities/Duplicates/Moves/Events) no longer open a new tab with
        # the raw /photo — target="_blank" was removed from the frames
        self.assertNotIn('a.target = "_blank"', html)
        # Events render clickable preview frames
        self.assertIn("event-thumbs", html)

    def test_tree_nodes_build_lazily_on_open(self):
        # perf: the Cities/Moves plan — up to thousands of frames; folder rows are
        # built ONLY on expanding <details> (toggle), not all at once (otherwise the
        # tab hung building tens of thousands of DOM nodes).
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn('details.addEventListener("toggle"', html)
        self.assertIn("if (!details.open || built) return;", html)

    def test_no_external_resources(self):
        # U1: invariant — the whole UI is inline, without external resources
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertNotIn("http://", html)
        self.assertNotIn("https://", html)
        self.assertNotIn("<link", html)


class TestPreviewEndpoint(UiServerTestBase):
    def test_preview_returns_jpeg_and_caches(self):
        fid, _p, _c = self.add_photo_file("a.jpg")
        self.start_server()
        real_decode = imaging.decode_rgb
        calls: list[int] = []

        def counting_decode(*args, **kwargs):
            calls.append(1)
            return real_decode(*args, **kwargs)

        with mock.patch.object(imaging, "decode_rgb", counting_decode):
            status1, body1, ctype1 = self.get(f"/preview/{fid}")
            self.assertEqual(status1, 200)
            self.assertIn("image/jpeg", ctype1)  # HEIC/RAW would also arrive as JPEG
            status2, body2, _c2 = self.get(f"/preview/{fid}")
            self.assertEqual(status2, 200)
            self.assertEqual(body1, body2)
        # the second request — from the cache, no re-decode
        self.assertEqual(len(calls), 1)

    def test_preview_unknown_id_404(self):
        self.start_server()
        status, _body, _ctype = self.get("/preview/999999")
        self.assertEqual(status, 404)


class TestThumbCache(UiServerTestBase):
    def test_repeated_get_thumb_does_not_redecode(self):
        fid, _p, _c = self.add_photo_file("a.jpg")
        self.start_server()
        real_decode = imaging.decode_rgb
        calls: list[int] = []

        def counting_decode(*args, **kwargs):
            calls.append(1)
            return real_decode(*args, **kwargs)

        with mock.patch.object(imaging, "decode_rgb", counting_decode):
            status1, body1, ctype1 = self.get(f"/thumb/{fid}")
            self.assertEqual(status1, 200)
            self.assertIn("image/jpeg", ctype1)
            status2, body2, ctype2 = self.get(f"/thumb/{fid}")
            self.assertEqual(status2, 200)
            self.assertIn("image/jpeg", ctype2)
            self.assertEqual(body1, body2)

        self.assertEqual(len(calls), 1)

    def test_different_files_are_cached_independently(self):
        fid1, _p1, _c1 = self.add_photo_file("a.jpg")
        fid2, _p2, _c2 = self.add_photo_file("b.jpg", country="ru", city="Moscow")
        self.start_server()
        real_decode = imaging.decode_rgb
        calls: list[int] = []

        def counting_decode(*args, **kwargs):
            calls.append(1)
            return real_decode(*args, **kwargs)

        with mock.patch.object(imaging, "decode_rgb", counting_decode):
            status1, _body1, _ctype1 = self.get(f"/thumb/{fid1}")
            status2, _body2, _ctype2 = self.get(f"/thumb/{fid2}")
            self.assertEqual(status1, 200)
            self.assertEqual(status2, 200)

        # different file_ids -> different cache keys -> both decoded (not confused
        # with each other), but each exactly once.
        self.assertEqual(len(calls), 2)


class TestClusterThumbSkeleton(UiServerTestBase):
    def test_skeleton_and_lazy_loading_present_in_cluster_card_js(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn('"thumb-skel"', html)
        self.assertIn('img.loading = "lazy"', html)
        self.assertIn('skel.className = "thumb-skel loaded"', html)


class ResetTestBase(ProcessTestBase):
    """Data fixtures in all tables that reset must wipe."""

    def post_reset(self):
        req = urllib.request.Request(
            f"{self.base_url}/api/process/reset", data=b"{}", method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def seed_full_index(self):
        fid, _p, _c = self.add_photo_file("a.jpg", country="ru", city="Moscow")
        cluster = self.conn.execute(
            "INSERT INTO face_clusters (label, merged_into) VALUES ('Alice', NULL)"
        ).lastrowid
        self.conn.execute(
            """INSERT INTO faces (file_id, bbox, embedding, cluster_id)
               VALUES (?, '[0,0,10,10]', ?, ?)""",
            (fid, b"embedding", cluster),
        )
        self.conn.execute(
            "INSERT INTO dedup_choice (file_id, action, updated_at) VALUES (?, 'keep', 'now')",
            (fid,),
        )
        batch = self.conn.execute(
            """INSERT INTO move_batches (mode, dest_root, started_at, finished_at, operation)
               VALUES ('city', ?, '2026-01-01T10:00:00', '2026-01-01T10:05:00', 'move')""",
            (str(self.root / "dest"),),
        ).lastrowid
        self.conn.execute(
            """INSERT INTO moves (batch_id, file_id, src, dst, hash, status)
               VALUES (?, ?, 'a', 'b', 'deadbeef', 'done')""",
            (batch, fid),
        )
        self.conn.commit()
        return fid


class TestProcessResetEndpoint(ResetTestBase):
    def test_reset_clears_all_tables_and_returns_200(self):
        self.seed_full_index()
        self.start_server()
        status, payload = self.post_reset()
        self.assertEqual(status, 200)
        self.assertTrue(payload.get("ok"))

        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) AS n FROM files").fetchone()["n"], 0)
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) AS n FROM face_clusters").fetchone()["n"], 0)
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) AS n FROM faces").fetchone()["n"], 0)
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) AS n FROM dedup_choice").fetchone()["n"], 0)
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) AS n FROM move_batches").fetchone()["n"], 0)
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) AS n FROM moves").fetchone()["n"], 0)

    def test_reset_refreshes_plan_cache_to_empty(self):
        self.seed_full_index()
        self.start_server()
        status_before, body_before, _ = self.get("/api/plan?mode=city")
        self.assertEqual(status_before, 200)
        self.assertEqual(len(json.loads(body_before)), 1)

        status, _payload = self.post_reset()
        self.assertEqual(status, 200)

        status_after, body_after, _ = self.get("/api/plan?mode=city")
        self.assertEqual(status_after, 200)
        self.assertEqual(json.loads(body_after), [])

    def test_reset_while_running_returns_409(self):
        block = threading.Event()
        self.patch_fast_stages(block_stage="index", block_event=block)
        self.start_server()
        try:
            status1, resp1 = self.post("/api/process", {"source_dir": str(self.src_dir)})
            self.assertEqual(status1, 200)
            self.assertTrue(resp1.get("ok"))
            self.assertTrue(self.status()["running"])

            status2, resp2 = self.post_reset()
            self.assertEqual(status2, 409)
            self.assertIn("error", resp2)
        finally:
            block.set()
        final_status = self._poll_finished()
        self.assertFalse(final_status["running"])

    def _poll_finished(self):
        import time
        deadline = time.monotonic() + 5.0
        last = None
        while time.monotonic() < deadline:
            last = self.status()
            if last["finished"]:
                return last
            time.sleep(0.02)
        raise AssertionError(f"the pipeline did not finish in time: {last}")

    def test_reset_after_finished_run_is_allowed(self):
        self.patch_fast_stages()
        self.start_server()
        status, _resp = self.post("/api/process", {"source_dir": str(self.src_dir)})
        self.assertEqual(status, 200)
        self._poll_finished()

        status2, payload2 = self.post_reset()
        self.assertEqual(status2, 200)
        self.assertTrue(payload2.get("ok"))


class TestProcessResetButtonHtml(UiServerTestBase):
    def test_reset_button_and_confirm_text_present(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn('id="process-reset-btn"', html)
        self.assertIn("btn-danger", html)
        self.assertIn("/api/process/reset", html)
        # an explicit warning: wipes the index (names/dupes), does not touch photos
        self.assertIn("process_reset_confirm", html)  # the i18n key is present in window.I18N
        i18n_start = html.index("window.I18N = ") + len("window.I18N = ")
        i18n_end = html.index(";</script>", i18n_start)
        i18n = json.loads(html[i18n_start:i18n_end])
        confirm_text = i18n["process_reset_confirm"]
        self.assertIn("people", confirm_text)
        self.assertIn("event", confirm_text)
        self.assertIn("duplicate", confirm_text)
        self.assertIn("NOT", confirm_text)

    def test_reset_button_in_process_controls_near_start_button(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        controls_start = html.index('class="process-controls"')
        section_end = html.index("</section>", controls_start)
        controls_html = html[controls_start:section_end]
        self.assertIn('id="process-start-btn"', controls_html)
        self.assertIn('id="process-reset-btn"', controls_html)


if __name__ == "__main__":
    unittest.main()
