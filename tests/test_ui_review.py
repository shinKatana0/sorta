"""F126: the "Review" workspace — four slices of one job, one decision per frame.

What this file is really guarding, in the order the risk runs:

* the duplicates half must not have moved. Its own tests (test_ui_dupes) still pass
  unchanged; here we only check that the list now lives inside the new tab and that its
  routes were not replaced by anything;
* a slice must contain its own frames and nobody else's — not a screenshot (F120), not
  an exact duplicate, not a file that failed to read;
* the blur window is a WINDOW: the list opens to `features.blur_review_max`, "show more"
  continues past it, and the seam neither loses a frame nor shows one twice;
* one decision per file, in the existing `dedup_choice`, surviving a recompute of
  `frame_quality` — and no route anywhere that marks a whole slice at once.
"""
from __future__ import annotations

import dataclasses
import json
import unittest
import urllib.error
import urllib.request
from unittest import mock

from sorta import ui

from tests.test_ui import UiServerTestBase


class ReviewTestBase(UiServerTestBase):
    """Photographs with `media_class`/`frame_quality` rows on top of the U1 server."""

    def add_reviewable(self, rel: str, *, verdict: str = "photo",
                       sharpness: float | None = 100.0,
                       eyes_open: int | None = None,
                       has_subject: int | None = None,
                       source: str = "vlm#aaaaaaaa") -> int:
        file_id, _p, _c = self.add_photo_file(rel)
        self.conn.execute(
            """INSERT INTO media_class (file_id, verdict, source, score, updated_at, tier)
               VALUES (?, ?, 'clip', 0.9, '2026-01-01', 'clip')""",
            (file_id, verdict))
        self.conn.execute(
            """INSERT INTO frame_quality (file_id, sharpness, eyes_open, has_subject,
                   source, updated_at)
               VALUES (?, ?, ?, ?, ?, '2026-01-01')""",
            (file_id, sharpness, eyes_open, has_subject, source))
        self.conn.commit()
        return file_id

    def add_real_face(self, file_id: int) -> None:
        """A found face — what makes the eyes question askable at all (F125)."""
        self.conn.execute(
            "INSERT INTO faces (file_id, bbox, embedding) VALUES (?, '[0,0,10,10]', ?)",
            (file_id, b"embedding"))
        self.conn.commit()

    def add_face_marker(self, file_id: int) -> None:
        """`bbox = '[]'` — "processed, no faces here", not a face."""
        self.conn.execute(
            "INSERT INTO faces (file_id, bbox, embedding) VALUES (?, '[]', ?)",
            (file_id, b""))
        self.conn.commit()

    def review(self, query: str = "") -> dict:
        status, body, ctype = self.get("/api/review" + query)
        self.assertEqual(status, 200, body)
        self.assertIn("application/json", ctype)
        return json.loads(body)

    def counts(self, data: dict) -> dict[str, int]:
        return {row["slice"]: row["count"] for row in data["counts"]}

    def actions(self) -> dict[int, str]:
        return {r["file_id"]: r["action"] for r in
                self.conn.execute("SELECT file_id, action FROM dedup_choice").fetchall()}

    def post(self, path: str, data: object) -> tuple[int, dict]:
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


class TestSliceSelection(ReviewTestBase):
    def test_each_slice_returns_its_own_frames(self):
        blurred = self.add_reviewable("blur.jpg", sharpness=10.0)
        eyes = self.add_reviewable("eyes.jpg", sharpness=500.0, eyes_open=0)
        subject = self.add_reviewable("nosubject.jpg", sharpness=500.0, has_subject=0)
        sharp = self.add_reviewable("fine.jpg", sharpness=900.0,
                                    eyes_open=1, has_subject=1)
        self.start_server()
        for slice_, expected in (("blurred", [blurred]), ("eyes", [eyes]),
                                 ("subject", [subject])):
            with self.subTest(slice=slice_):
                data = self.review(f"?slice={slice_}")
                self.assertEqual([it["file_id"] for it in data["items"]], expected)
                self.assertNotIn(sharp, [it["file_id"] for it in data["items"]])

    def test_a_null_answer_is_not_a_no(self):
        # NULL in eyes_open/has_subject means "not asked" — a frame nobody looked at
        # must never be offered as an answer.
        self.add_reviewable("unasked.jpg", sharpness=500.0,
                            eyes_open=None, has_subject=None)
        self.start_server()
        self.assertEqual(self.review("?slice=eyes")["total"], 0)
        self.assertEqual(self.review("?slice=subject")["total"], 0)

    def test_non_photos_are_in_no_slice(self):
        # F120: the quality signals mean nothing on a screenshot, a document or a
        # product shot, so those frames are not offered for review at all.
        for verdict in ("screenshot", "document", "product", "meme"):
            self.add_reviewable(f"{verdict}.jpg", verdict=verdict, sharpness=1.0,
                                eyes_open=0, has_subject=0)
        self.start_server()
        data = self.review("?slice=blurred")
        self.assertEqual(self.counts(data), {"dupes": 0, "blurred": 0, "eyes": 0,
                                             "subject": 0})
        for slice_ in ("blurred", "eyes", "subject"):
            self.assertEqual(self.review(f"?slice={slice_}")["items"], [])

    def test_duplicates_and_read_errors_are_in_no_slice(self):
        canonical = self.add_reviewable("a.jpg", sharpness=5.0, eyes_open=0,
                                        has_subject=0)
        duplicate = self.add_reviewable("b.jpg", sharpness=5.0, eyes_open=0,
                                        has_subject=0)
        broken = self.add_reviewable("c.jpg", sharpness=5.0, eyes_open=0, has_subject=0)
        self.conn.execute("UPDATE files SET dup_of = ? WHERE id = ?",
                          (canonical, duplicate))
        self.conn.execute("UPDATE files SET error = 'nope' WHERE id = ?", (broken,))
        self.conn.commit()
        self.start_server()
        for slice_ in ("blurred", "eyes", "subject"):
            with self.subTest(slice=slice_):
                data = self.review(f"?slice={slice_}")
                self.assertEqual([it["file_id"] for it in data["items"]], [canonical])
                self.assertEqual(data["total"], 1)

    def test_an_empty_slice_answers_zero_and_stays_in_the_list(self):
        self.add_reviewable("a.jpg", sharpness=5.0)
        self.start_server()
        data = self.review("?slice=eyes")
        counts = self.counts(data)
        self.assertEqual(set(counts), {"dupes", "blurred", "eyes", "subject"})
        self.assertEqual(counts["eyes"], 0)
        self.assertEqual(data["total"], 0)
        self.assertEqual(data["items"], [])

    def test_unknown_slice_is_a_400(self):
        self.start_server()
        status, body, _ctype = self.get("/api/review?slice=everything")
        self.assertEqual(status, 400)
        self.assertIn("error", json.loads(body))

    def test_bad_offset_is_a_400(self):
        self.start_server()
        status, body, _ctype = self.get("/api/review?slice=blurred&offset=nope")
        self.assertEqual(status, 400)
        self.assertIn("error", json.loads(body))

    def test_the_default_slice_is_duplicates(self):
        self.start_server()
        self.assertEqual(self.review()["slice"], "dupes")

    def test_the_card_carries_what_the_decision_needs(self):
        fid = self.add_reviewable("holiday/blur.jpg", sharpness=12.5)
        self.start_server()
        item = self.review("?slice=blurred")["items"][0]
        self.assertEqual(set(item), {"file_id", "name", "date", "src_dir", "src_path",
                                     "sharpness", "action", "thumb_url", "video"})
        self.assertEqual(item["file_id"], fid)
        self.assertEqual(item["name"], "blur.jpg")
        self.assertEqual(item["src_dir"], "holiday")
        self.assertAlmostEqual(item["sharpness"], 12.5)
        self.assertEqual(item["thumb_url"], f"/thumb/{fid}")
        self.assertIsNone(item["action"])
        self.assertFalse(item["video"])


class TestDuplicatesSlice(ReviewTestBase):
    """The grouped slice keeps its own route and its own shape."""

    def add_dupe(self, rel: str, *, phash: str) -> int:
        self._n += 1
        p = self.src_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"fake-image-bytes")
        cur = self.conn.execute(
            """INSERT INTO files (path, size, mtime, ext, media_type, hash, hash_algo,
                   phash, taken_at, taken_at_source, taken_at_confidence, width, height,
                   indexed_at)
               VALUES (?, 1000, 0, 'jpg', 'photo', ?, 'blake3', ?,
                       '2022-05-01T10:00:00', 'exif', 'high', 100, 100, '2026-01-01')""",
            (str(p.resolve()), f"hash-{self._n}", phash))
        self.conn.commit()
        return cur.lastrowid

    def test_the_counter_counts_groups_and_the_page_stays_empty(self):
        self.add_dupe("a.jpg", phash="0" * 16)
        self.add_dupe("b.jpg", phash="0" * 16)
        self.start_server()
        data = self.review("?slice=dupes")
        self.assertTrue(data["grouped"])
        self.assertEqual(self.counts(data)["dupes"], 1)
        self.assertEqual(data["total"], 1)
        # Duplicates are rendered from their own route, which is why this one carries
        # no items for them.
        self.assertEqual(data["items"], [])

    def test_the_duplicates_route_is_still_the_one_that_serves_them(self):
        self.add_dupe("a.jpg", phash="0" * 16)
        self.add_dupe("b.jpg", phash="0" * 16)
        self.start_server()
        status, body, _ctype = self.get("/api/dupes")
        self.assertEqual(status, 200)
        groups = json.loads(body)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]["frames"]), 2)

    def tearDown(self):
        ui._dupes_cache_clear()
        super().tearDown()


class TestBlurWindow(ReviewTestBase):
    def test_sorted_by_ascending_sharpness(self):
        mid = self.add_reviewable("mid.jpg", sharpness=50.0)
        low = self.add_reviewable("low.jpg", sharpness=5.0)
        high = self.add_reviewable("high.jpg", sharpness=80.0)
        self.start_server()
        data = self.review("?slice=blurred")
        self.assertEqual([it["file_id"] for it in data["items"]], [low, mid, high])

    def test_the_window_bounds_the_list(self):
        inside = [self.add_reviewable(f"in{i}.jpg", sharpness=10.0 + i)
                  for i in range(3)]
        self.add_reviewable("out.jpg", sharpness=300.0)
        self.start_server()
        data = self.review("?slice=blurred")
        self.assertEqual(data["blur_max"], 90.0)
        self.assertEqual(data["window_total"], 3)
        self.assertEqual(data["total"], 3)
        self.assertEqual([it["file_id"] for it in data["items"]], inside)
        self.assertEqual(self.counts(data)["blurred"], 3)

    def test_the_window_follows_the_config_key(self):
        self.add_reviewable("a.jpg", sharpness=50.0)
        self.add_reviewable("b.jpg", sharpness=150.0)
        self.cfg.features = dataclasses.replace(self.cfg.features, blur_review_max=200.0)
        self.start_server()
        data = self.review("?slice=blurred")
        self.assertEqual(data["blur_max"], 200.0)
        self.assertEqual(data["total"], 2)

    def test_show_more_continues_past_the_window_without_a_seam(self):
        # The window is a prefix of the same ordering, so the page that continues past
        # it must neither repeat a frame nor skip one.
        inside = [self.add_reviewable(f"in{i}.jpg", sharpness=10.0 + i)
                  for i in range(3)]
        outside = [self.add_reviewable(f"out{i}.jpg", sharpness=100.0 + i)
                   for i in range(3)]
        self.start_server()
        window = self.review("?slice=blurred")
        self.assertEqual([it["file_id"] for it in window["items"]], inside)
        beyond = self.review(f"?slice=blurred&beyond=1&offset={len(window['items'])}")
        self.assertTrue(beyond["beyond"])
        self.assertEqual(beyond["total"], 6)
        self.assertEqual(beyond["window_total"], 3)
        self.assertEqual([it["file_id"] for it in beyond["items"]], outside)
        seen = [it["file_id"] for it in window["items"] + beyond["items"]]
        self.assertEqual(len(seen), len(set(seen)))
        self.assertEqual(set(seen), set(inside + outside))

    def test_paging_inside_the_window_walks_it_once(self):
        for i in range(5):
            self.add_reviewable(f"a{i}.jpg", sharpness=10.0 + i)
        self.start_server()
        pages = [self.review(f"?slice=blurred&offset={off}&limit=2")
                 for off in (0, 2, 4)]
        seen = [it["file_id"] for page in pages for it in page["items"]]
        self.assertEqual(pages[0]["total"], 5)
        self.assertEqual(len(seen), 5)
        self.assertEqual(len(set(seen)), 5)

    def test_frames_without_a_sharpness_number_are_not_in_the_list(self):
        # NULL sharpness means the frame did not decode, not that it is blurred.
        self.add_reviewable("undecoded.jpg", sharpness=None)
        self.start_server()
        self.assertEqual(self.review("?slice=blurred")["total"], 0)


class TestEyesWithoutFacesRun(ReviewTestBase):
    def test_the_reason_travels_instead_of_a_bare_zero(self):
        self.add_reviewable("a.jpg", sharpness=500.0)
        self.start_server()
        data = self.review("?slice=eyes")
        self.assertEqual(data["eyes_reason"], "no_faces_run")
        self.assertEqual(data["total"], 0)

    def test_a_marker_only_row_is_not_a_faces_run(self):
        # `bbox = '[]'` stands on nearly every file and means "processed, no faces
        # here" — reading it as a faces run would hide the reason on every collection.
        fid = self.add_reviewable("a.jpg", sharpness=500.0)
        self.add_face_marker(fid)
        self.start_server()
        self.assertEqual(self.review("?slice=eyes")["eyes_reason"], "no_faces_run")

    def test_a_found_face_clears_the_reason(self):
        fid = self.add_reviewable("a.jpg", sharpness=500.0, eyes_open=0)
        self.add_real_face(fid)
        self.start_server()
        data = self.review("?slice=eyes")
        self.assertIsNone(data["eyes_reason"])
        self.assertEqual([it["file_id"] for it in data["items"]], [fid])


class TestOneDecisionPerFrame(ReviewTestBase):
    def test_marking_writes_dedup_choice_and_nothing_else(self):
        fid = self.add_reviewable("a.jpg", sharpness=10.0)
        self.start_server()
        with mock.patch("sorta.ui.send_to_trash") as trash:
            status, body = self.post(
                "/api/review/mark", {"file_ids": [fid], "action": "to_delete"})
        self.assertEqual(status, 200)
        self.assertEqual(body, {"ok": True, "marked": 1})
        self.assertEqual(self.actions(), {fid: "to_delete"})
        trash.assert_not_called()
        self.assertIsNotNone(
            self.conn.execute("SELECT id FROM files WHERE id = ?", (fid,)).fetchone())

    def test_keep_and_clear_are_different_answers(self):
        fid = self.add_reviewable("a.jpg", sharpness=10.0)
        self.start_server()
        self.post("/api/review/mark", {"file_ids": [fid], "action": "keep"})
        self.assertEqual(self.actions(), {fid: "keep"})
        self.post("/api/review/mark", {"file_ids": [fid], "action": "clear"})
        self.assertEqual(self.actions(), {})

    def test_a_frame_in_two_slices_carries_one_decision_visible_in_both(self):
        fid = self.add_reviewable("a.jpg", sharpness=10.0, eyes_open=0)
        self.add_real_face(fid)
        self.start_server()
        self.post("/api/review/mark", {"file_ids": [fid], "action": "to_delete"})
        rows = self.conn.execute(
            "SELECT COUNT(*) c FROM dedup_choice WHERE file_id = ?", (fid,)).fetchone()
        self.assertEqual(rows["c"], 1)
        for slice_ in ("blurred", "eyes"):
            with self.subTest(slice=slice_):
                item = self.review(f"?slice={slice_}")["items"][0]
                self.assertEqual(item["file_id"], fid)
                self.assertEqual(item["action"], "to_delete")

    def test_a_decision_made_on_the_duplicates_slice_shows_up_on_a_flat_one(self):
        fid = self.add_reviewable("a.jpg", sharpness=10.0)
        other = self.add_reviewable("b.jpg", sharpness=10.0)
        self.start_server()
        status, _body = self.post(
            "/api/dupes/choice", {"group": [fid, other], "keep_file_id": fid})
        self.assertEqual(status, 200)
        by_id = {it["file_id"]: it for it in self.review("?slice=blurred")["items"]}
        self.assertEqual(by_id[fid]["action"], "keep")
        self.assertEqual(by_id[other]["action"], "to_delete")

    def test_decisions_survive_a_recompute_of_frame_quality(self):
        # A re-run rewrites `frame_quality` (here: a new prompt fingerprint in `source`,
        # which is what invalidates the rows). The decisions live in another table for
        # exactly this reason — otherwise the couple of frames kept for the memory would
        # come back with every run.
        keep = self.add_reviewable("keep.jpg", sharpness=10.0, source="vlm#aaaaaaaa")
        drop = self.add_reviewable("drop.jpg", sharpness=11.0, source="vlm#aaaaaaaa")
        self.start_server()
        self.post("/api/review/mark", {"file_ids": [keep], "action": "keep"})
        self.post("/api/review/mark", {"file_ids": [drop], "action": "to_delete"})
        self.conn.execute("DELETE FROM frame_quality")
        for fid, sharp in ((keep, 12.0), (drop, 13.0)):
            self.conn.execute(
                """INSERT INTO frame_quality (file_id, sharpness, source, updated_at)
                   VALUES (?, ?, 'vlm#bbbbbbbb', '2026-02-02')""", (fid, sharp))
        self.conn.commit()
        self.assertEqual(self.actions(), {keep: "keep", drop: "to_delete"})
        by_id = {it["file_id"]: it for it in self.review("?slice=blurred")["items"]}
        self.assertEqual(by_id[keep]["action"], "keep")
        self.assertEqual(by_id[drop]["action"], "to_delete")

    def test_unknown_ids_are_skipped_rather_than_written(self):
        fid = self.add_reviewable("a.jpg", sharpness=10.0)
        self.start_server()
        status, body = self.post(
            "/api/review/mark", {"file_ids": [fid, 999999], "action": "to_delete"})
        self.assertEqual(status, 200)
        self.assertEqual(body["marked"], 1)
        self.assertEqual(self.actions(), {fid: "to_delete"})

    def test_a_bad_body_is_a_400_and_writes_nothing(self):
        fid = self.add_reviewable("a.jpg", sharpness=10.0)
        self.start_server()
        for payload in ({"file_ids": [fid], "action": "burn"},
                        {"file_ids": [], "action": "keep"},
                        {"file_ids": [fid]},
                        {"file_ids": ["1"], "action": "keep"},
                        {"action": "keep"},
                        []):
            with self.subTest(payload=payload):
                status, body = self.post("/api/review/mark", payload)
                self.assertEqual(status, 400)
                self.assertIn("error", body)
        self.assertEqual(self.actions(), {})


class TestNoBulkDeleteRoute(ReviewTestBase):
    """A safety requirement, not a matter of taste: sharpness ranks frames, it does not
    classify them, so nothing here may mark or delete a whole slice at once."""

    def post_status(self, path: str, data: object) -> int:
        """The status alone — an absent route answers with an HTML 404, not JSON."""
        body = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}{path}", data=body, method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status
        except urllib.error.HTTPError as exc:
            exc.read()
            return exc.code

    def test_no_route_deletes_below_a_threshold(self):
        self.add_reviewable("a.jpg", sharpness=1.0)
        self.start_server()
        for path in ("/api/review/trash", "/api/review/delete", "/api/review/mark-all",
                     "/api/review/mark_all", "/api/blurred/trash"):
            with self.subTest(path=path):
                self.assertEqual(
                    self.post_status(path, {"slice": "blurred", "max": 90}), 404)

    def test_the_mark_route_takes_ids_and_never_a_threshold(self):
        fid = self.add_reviewable("a.jpg", sharpness=1.0)
        self.start_server()
        status, body = self.post(
            "/api/review/mark", {"slice": "blurred", "max": 90, "action": "to_delete"})
        self.assertEqual(status, 400)
        self.assertIn("error", body)
        self.assertEqual(self.actions(), {})
        self.assertIsNotNone(
            self.conn.execute("SELECT id FROM files WHERE id = ?", (fid,)).fetchone())

    def test_the_server_has_no_bulk_marking_helper(self):
        self.assertFalse([name for name in dir(ui)
                          if "mark_all" in name or "mark_below" in name])
        self.assertEqual(ui._REVIEW_MARK_ACTIONS, ("keep", "to_delete", "clear"))


class TestOverviewCounters(ReviewTestBase):
    def test_the_overview_counts_the_same_slices_the_tab_does(self):
        self.add_reviewable("blur.jpg", sharpness=10.0)
        self.add_reviewable("eyes.jpg", sharpness=500.0, eyes_open=0)
        self.add_reviewable("nosubject.jpg", sharpness=500.0, has_subject=0)
        self.add_reviewable("screenshot.jpg", verdict="screenshot", sharpness=1.0,
                            eyes_open=0, has_subject=0)
        self.start_server()
        _status, body, _ctype = self.get("/api/overview")
        collection = json.loads(body)["collection"]
        counts = self.counts(self.review("?slice=blurred"))
        self.assertEqual(collection["blurred"], counts["blurred"])
        self.assertEqual(collection["eyes_closed"], counts["eyes"])
        self.assertEqual(collection["no_subject"], counts["subject"])
        self.assertEqual(collection["blurred"], 1)
        self.assertEqual(collection["eyes_closed"], 1)
        self.assertEqual(collection["no_subject"], 1)

    def test_the_blurred_counter_uses_the_same_window_as_the_list(self):
        self.add_reviewable("a.jpg", sharpness=50.0)
        self.add_reviewable("b.jpg", sharpness=150.0)
        self.cfg.features = dataclasses.replace(self.cfg.features, blur_review_max=100.0)
        self.start_server()
        _status, body, _ctype = self.get("/api/overview")
        self.assertEqual(json.loads(body)["collection"]["blurred"], 1)
        self.assertEqual(self.review("?slice=blurred")["total"], 1)

    def test_zero_when_nothing_was_measured(self):
        self.add_photo_file("a.jpg")
        self.start_server()
        _status, body, _ctype = self.get("/api/overview")
        collection = json.loads(body)["collection"]
        self.assertEqual(
            (collection["blurred"], collection["eyes_closed"], collection["no_subject"]),
            (0, 0, 0))


class TestReviewTabHtml(ReviewTestBase):
    def setUp(self):
        super().setUp()
        self.start_server()
        _status, body, _ctype = self.get("/")
        self.html = body.decode("utf-8")

    def test_the_tab_replaced_the_duplicates_tab(self):
        self.assertIn('id="tab-btn-review"', self.html)
        self.assertIn('id="tab-review"', self.html)
        self.assertNotIn('id="tab-btn-dupes"', self.html)
        self.assertNotIn('id="tab-dupes"', self.html)
        self.assertIn('"review", "person"', self.html)

    def test_the_duplicates_machinery_moved_in_unchanged(self):
        self.assertIn('id="dupes-list"', self.html)
        self.assertIn('id="dupes-save-all-btn"', self.html)
        self.assertIn('fetch("/api/dupes")', self.html)
        self.assertIn("/api/dupes/choices", self.html)
        self.assertIn("/api/dupes/trash", self.html)

    def test_all_four_slices_are_in_the_markup_with_a_counter_each(self):
        for slice_ in ("dupes", "blurred", "eyes", "subject"):
            with self.subTest(slice=slice_):
                self.assertIn(f'id="review-slice-{slice_}"', self.html)
                self.assertIn(f'id="review-count-{slice_}"', self.html)
        self.assertIn('var REVIEW_SLICES = ["dupes", "blurred", "eyes", "subject"];',
                      self.html)

    def test_the_grid_is_paged_rather_than_rendered_whole(self):
        # F70: 530 cards with previews must not land in the DOM at once.
        self.assertIn("var REVIEW_PAGE_SIZE = 200;", self.html)
        self.assertIn("fetchReview(reviewOffset, true)", self.html)
        self.assertIn('id="review-more-btn"', self.html)

    def test_no_external_resources(self):
        self.assertNotIn("http://", self.html)
        self.assertNotIn("https://", self.html)
        self.assertNotIn("<link", self.html)

    def test_the_quality_scope_select_offers_by_faces(self):
        self.assertIn('<option value="faces">', self.html)
        for lang, expected in (("ru", "По лицам"), ("en", "By faces"), ("ja", "顔で")):
            with self.subTest(lang=lang):
                _status, body, _ctype = self.get(f"/?lang={lang}")
                self.assertIn(expected, body.decode("utf-8"))

    def test_the_select_offers_nothing_the_server_would_refuse(self):
        offered = {"groups", "events", "faces", "all"}
        self.assertTrue(offered <= set(ui._SETTINGS_SPEC["vlm.quality_scope"].choices))

    def test_i18n_ru_en_ja(self):
        for lang, expected in (("ru", "Разбор"), ("en", "Review"), ("ja", "仕分け")):
            with self.subTest(lang=lang):
                _status, body, _ctype = self.get(f"/?lang={lang}")
                self.assertIn(expected, body.decode("utf-8"))

    def test_every_new_string_is_translated_three_ways(self):
        keys = ("tab_review", "review_intro", "review_slice_dupes",
                "review_slice_blurred", "review_slice_eyes", "review_slice_subject",
                "review_hint_blurred", "review_hint_eyes", "review_hint_subject",
                "review_eyes_no_faces", "review_empty", "review_sharpness_label",
                "review_mark_delete", "review_mark_keep", "review_mark_clear",
                "review_select_label", "review_select_all", "review_select_none",
                "review_marked_status", "review_load_more", "review_load_more_beyond",
                "review_shown_label", "review_error_prefix", "error_loading_review",
                "overview_blurred", "overview_eyes_closed", "overview_no_subject",
                "settings_scope_faces")
        for key in keys:
            with self.subTest(key=key):
                entry = ui._UI_STRINGS[key]
                self.assertEqual(set(entry), {"ru", "en", "ja"})
                for lang in ("ru", "en", "ja"):
                    self.assertTrue(entry[lang].strip())
        for lang in ("ru", "en", "ja"):
            self.assertIn("{max}", ui._UI_STRINGS["review_hint_blurred"][lang])
            self.assertIn("{value}", ui._UI_STRINGS["review_sharpness_label"][lang])
            self.assertIn("{shown}", ui._UI_STRINGS["review_shown_label"][lang])
            self.assertIn("{total}", ui._UI_STRINGS["review_shown_label"][lang])
            self.assertIn("{n}", ui._UI_STRINGS["review_marked_status"][lang])

    def test_the_view_says_nothing_is_deleted_by_the_number(self):
        # The hint is the feature: a threshold that decides nothing has to say so, or
        # the next reader will add the button this feature deliberately does not have.
        self.assertIn("delete everything below the", self.html)
        self.assertNotIn("delete-below-threshold", self.html)


if __name__ == "__main__":
    unittest.main()
