"""F103: the "Utility frames" slice — the classifier's buckets shown AS buckets, plus
the bulk way back for the frames it got wrong.

The deep VLM tier carries roughly every tenth frame of a live collection into service
folders and a few of those verdicts are wrong; until this view there was nowhere to look
at them. Two properties matter most here and both are pinned below:

* the correction is a row in `manual_overrides` (F77) and `media_class` is NOT rewritten
  — otherwise a re-run of the junk tier would silently wipe the user's decisions;
* a `document` card never carries a preview link. That bucket is passports, medical
  forms and bank papers, and the project rule is that such a frame is not decoded for
  display. Returning one to the photos is still allowed — only its preview is not built.

The sorter side of the `photo` action (a returned frame is laid out by city, not into a
service folder) is exercised here too, through `plan_and_sort`: the endpoint would be
pointless if the plan kept the frame where the classifier put it.
"""
from __future__ import annotations

import dataclasses
import io
import json
import unittest
import urllib.error
import urllib.parse
import urllib.request
from contextlib import redirect_stdout

from sorta import ui
from sorta.sorter import plan_and_sort

from tests import waiting
from tests.test_ui import UiServerTestBase


class JunkViewTestBase(UiServerTestBase):
    def post(self, path: str, data: object) -> tuple[int, dict]:
        answer = waiting.post_json(f"{self.base_url}{path}", data)
        return answer.status, answer.json()

    def junk(self, query: str = "") -> dict:
        _status, body, _ctype = self.get("/api/junk" + query)
        return json.loads(body)

    def add_classified(self, rel: str, verdict: str, *, country: str | None = "ru",
                       city: str | None = "Moscow", source: str = "vlm") -> int:
        """A file plus its `media_class` row — the fixture this whole view reads."""
        file_id, _path, _content = self.add_photo_file(rel, country=country, city=city)
        self.conn.execute(
            """INSERT INTO media_class (file_id, verdict, source, score, updated_at, tier)
               VALUES (?, ?, ?, NULL, '2026-07-28', 'vlm')""",
            (file_id, verdict, source))
        self.conn.commit()
        return file_id

    def override_rows(self) -> dict[int, tuple[str, str | None]]:
        return {r["file_id"]: (r["action"], r["target"]) for r in self.conn.execute(
            "SELECT file_id, action, target FROM manual_overrides")}

    def verdicts(self) -> dict[int, str]:
        return {r["file_id"]: r["verdict"]
                for r in self.conn.execute("SELECT file_id, verdict FROM media_class")}

    def plan_targets(self) -> dict[int, str]:
        """file_id -> target_rel of the city plan (a dry run, nothing is moved)."""
        with redirect_stdout(io.StringIO()):
            report = plan_and_sort(self.cfg, self.conn, "city", self.root / "dest",
                                   apply=False, write_reports=False)
        return {it.file_id: it.target_rel for it in report.plan}


class TestBucketsAndCounters(JunkViewTestBase):
    def test_only_non_photo_frames_are_listed(self):
        photo = self.add_classified("a.jpg", "photo")
        product = self.add_classified("b.jpg", "product")
        screenshot = self.add_classified("c.jpg", "screenshot")
        self.start_server()
        ids = {it["file_id"] for it in self.junk()["items"]}
        self.assertEqual(ids, {product, screenshot})
        self.assertNotIn(photo, ids)

    def test_a_file_without_a_verdict_is_not_listed(self):
        # junk has not run on it yet — it is not a bucket, it is simply unclassified.
        self.add_photo_file("a.jpg", country="ru", city="Moscow")
        self.add_classified("b.jpg", "meme")
        self.start_server()
        self.assertEqual(self.junk()["total"], 1)

    def test_bucket_counters_match_media_class(self):
        for i in range(3):
            self.add_classified(f"p{i}.jpg", "product")
        self.add_classified("d.jpg", "document")
        self.add_classified("s.jpg", "screenshot")
        self.add_classified("ok.jpg", "photo")
        self.start_server()
        buckets = {b["verdict"]: b["count"] for b in self.junk()["buckets"]}
        expected = {r["verdict"]: r["n"] for r in self.conn.execute(
            "SELECT verdict, COUNT(*) AS n FROM media_class "
            "WHERE verdict <> 'photo' GROUP BY verdict")}
        self.assertEqual(buckets, expected)
        self.assertEqual(buckets, {"product": 3, "document": 1, "screenshot": 1})
        self.assertEqual(self.junk()["total"], 5)

    def test_buckets_come_sorted_by_size(self):
        self.add_classified("d.jpg", "document")
        for i in range(2):
            self.add_classified(f"p{i}.jpg", "product")
        self.start_server()
        self.assertEqual([b["verdict"] for b in self.junk()["buckets"]],
                         ["product", "document"])

    def test_counters_do_not_depend_on_the_current_filter(self):
        self.add_classified("p.jpg", "product")
        self.add_classified("d.jpg", "document")
        self.start_server()
        data = self.junk("?bucket=product")
        self.assertEqual({b["verdict"] for b in data["buckets"]}, {"product", "document"})
        self.assertEqual(data["total"], 1)


class TestBucketFilter(JunkViewTestBase):
    def test_filter_returns_only_that_verdict(self):
        product = self.add_classified("p.jpg", "product")
        self.add_classified("d.jpg", "document")
        self.add_classified("s.jpg", "screenshot")
        self.start_server()
        data = self.junk("?bucket=product")
        self.assertEqual(data["bucket"], "product")
        self.assertEqual([it["file_id"] for it in data["items"]], [product])
        self.assertEqual(data["total"], 1)

    def test_every_bucket_filters_to_itself(self):
        ids = {v: self.add_classified(f"{v}.jpg", v)
               for v in ("product", "document", "screenshot", "meme")}
        self.start_server()
        for verdict, file_id in ids.items():
            with self.subTest(verdict=verdict):
                items = self.junk("?bucket=" + verdict)["items"]
                self.assertEqual([it["file_id"] for it in items], [file_id])
                self.assertEqual({it["verdict"] for it in items}, {verdict})

    def test_photo_cannot_be_asked_for_as_a_bucket(self):
        # The `<> 'photo'` guard is in the query, not in the parameter check — no value
        # of `bucket` may turn this route into a way of listing personal photos.
        self.add_classified("a.jpg", "photo")
        self.add_classified("b.jpg", "product")
        self.start_server()
        data = self.junk("?bucket=photo")
        self.assertEqual(data["items"], [])
        self.assertEqual(data["total"], 0)

    def test_unknown_bucket_is_an_empty_page_not_an_error(self):
        self.add_classified("b.jpg", "product")
        self.start_server()
        status, body, _ctype = self.get("/api/junk?bucket=nonsense")
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertEqual(data["items"], [])
        self.assertEqual(data["total"], 0)

    def test_empty_bucket_returns_a_correct_empty_answer(self):
        # Requirement 6: an empty bucket is a well-formed answer the UI can render as
        # "nothing here", not a request that never resolves.
        self.add_classified("b.jpg", "product")
        self.start_server()
        data = self.junk("?bucket=meme")
        self.assertEqual(data["items"], [])
        self.assertEqual(data["total"], 0)
        self.assertEqual(data["bucket"], "meme")
        self.assertEqual(data["offset"], 0)
        self.assertIn("product", [b["verdict"] for b in data["buckets"]])

    def test_an_empty_index_answers_with_empty_buckets(self):
        self.start_server()
        status, body, _ctype = self.get("/api/junk")
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertEqual(data["buckets"], [])
        self.assertEqual(data["items"], [])
        self.assertEqual(data["total"], 0)


class TestPagingWindow(JunkViewTestBase):
    def test_limit_and_offset_page_the_bucket(self):
        ids = [self.add_classified(f"p{i}.jpg", "product") for i in range(5)]
        self.start_server()
        first = self.junk("?bucket=product&limit=2")
        self.assertEqual([it["file_id"] for it in first["items"]], ids[:2])
        self.assertEqual(first["total"], 5)
        second = self.junk("?bucket=product&limit=2&offset=2")
        self.assertEqual([it["file_id"] for it in second["items"]], ids[2:4])

    def test_a_broken_window_is_refused(self):
        self.start_server()
        for query in ("?offset=-1", "?limit=abc"):
            with self.subTest(query=query):
                status, _body, _ctype = self.get("/api/junk" + query)
                self.assertEqual(status, 400)

    def test_an_over_eager_limit_is_clamped_not_refused(self):
        self.add_classified("p.jpg", "product")
        self.start_server()
        data = self.junk("?limit=999999")
        self.assertEqual(data["limit"], ui._PLAN_PAGE_MAX_LIMIT)


class TestDocumentsAreNotPreviewed(JunkViewTestBase):
    def test_a_document_card_carries_no_preview_link(self):
        self.add_classified("passport.jpg", "document")
        self.start_server()
        item = self.junk("?bucket=document")["items"][0]
        self.assertNotIn("thumb_url", item)
        self.assertNotIn("video", item)
        # what is left is enough to decide: the name and the date
        self.assertEqual(item["name"], "passport.jpg")
        self.assertEqual(item["date"], "2022-05-01T10:00:00")

    def test_a_product_card_does_carry_a_preview_link(self):
        product = self.add_classified("chair.jpg", "product")
        self.start_server()
        item = self.junk("?bucket=product")["items"][0]
        self.assertEqual(item["thumb_url"], f"/thumb/{product}")
        self.assertFalse(item["video"])

    def test_no_document_preview_leaks_through_the_all_bucket(self):
        self.add_classified("passport.jpg", "document")
        self.add_classified("chair.jpg", "product")
        self.start_server()
        by_verdict = {it["verdict"]: it for it in self.junk()["items"]}
        self.assertNotIn("thumb_url", by_verdict["document"])
        self.assertIn("thumb_url", by_verdict["product"])

    def test_the_whole_document_response_mentions_no_thumb_route(self):
        self.add_classified("passport.jpg", "document")
        self.start_server()
        _status, body, _ctype = self.get("/api/junk?bucket=document")
        self.assertNotIn(b"/thumb/", body)
        self.assertNotIn(b"/preview/", body)
        self.assertNotIn(b"/photo/", body)

    def test_the_list_comes_from_the_config_key_and_not_from_the_code(self):
        """F133: `vlm.exclude_classes` decides, so moving a class in or out of it moves
        the protection with it. Asserting the default alone cannot tell a config read
        from a hard-coded "document" — only changing the key can."""
        self.add_classified("passport.jpg", "document")
        self.add_classified("chair.jpg", "product")
        self.cfg.vlm = dataclasses.replace(
            self.cfg.vlm, exclude_classes=("product",))
        self.start_server()
        by_verdict = {it["verdict"]: it for it in self.junk()["items"]}
        # the class that JOINED the list lost its preview...
        self.assertNotIn("thumb_url", by_verdict["product"])
        # ...and the one that left it got its preview back
        self.assertIn("thumb_url", by_verdict["document"])

    def test_an_empty_list_lifts_the_protection_and_says_so_by_doing_it(self):
        """Emptying the key is a decision a person can make — it means "show the model
        everything", and since F133 it also means "render everything". The trade is
        recorded in the guide; the test pins that it is the KEY doing it."""
        self.add_classified("passport.jpg", "document")
        self.cfg.vlm = dataclasses.replace(self.cfg.vlm, exclude_classes=())
        self.start_server()
        item = self.junk("?bucket=document")["items"][0]
        self.assertIn("thumb_url", item)


class TestBulkReturnToPhotos(JunkViewTestBase):
    def restore(self, file_ids: list[int]) -> tuple[int, dict]:
        return self.post("/api/overrides", {"file_ids": file_ids, "action": "photo"})

    def test_n_selected_frames_write_n_override_rows(self):
        ids = [self.add_classified(f"p{i}.jpg", "product") for i in range(3)]
        self.add_classified("keep.jpg", "product")
        self.start_server()
        status, payload = self.restore(ids)
        self.assertEqual(status, 200)
        self.assertEqual(sorted(payload["file_ids"]), sorted(ids))
        self.assertEqual(self.override_rows(),
                         {file_id: ("photo", None) for file_id in ids})

    def test_media_class_is_not_rewritten(self):
        # The model's verdict is a measurement; the correction is a separate layer on
        # top of it. Overwriting it would let the next junk run wipe the correction.
        ids = [self.add_classified(f"p{i}.jpg", "product") for i in range(2)]
        before = self.verdicts()
        self.start_server()
        self.restore(ids)
        self.assertEqual(self.verdicts(), before)
        self.assertEqual(set(self.verdicts().values()), {"product"})

    def test_a_repeated_return_does_not_duplicate_rows(self):
        file_id = self.add_classified("p.jpg", "product")
        self.start_server()
        self.restore([file_id])
        self.restore([file_id])
        self.restore([file_id])
        count = self.conn.execute(
            "SELECT COUNT(*) FROM manual_overrides").fetchone()[0]
        self.assertEqual(count, 1)
        self.assertEqual(self.override_rows(), {file_id: ("photo", None)})

    def test_a_returned_frame_is_laid_out_by_city_not_into_a_service_folder(self):
        returned = self.add_classified("wrong.jpg", "product")
        left = self.add_classified("real.jpg", "product")
        self.start_server()
        before = self.plan_targets()
        self.assertTrue(before[returned].startswith("_Products/"), before[returned])
        self.restore([returned])
        after = self.plan_targets()
        self.assertEqual(after[returned], "Russia/Moscow/2022/wrong.jpg")
        # the frames nobody touched keep the classifier's route
        self.assertEqual(after[left], before[left])

    def test_returning_several_frames_moves_all_of_them(self):
        ids = [self.add_classified(f"p{i}.jpg", "product") for i in range(3)]
        self.start_server()
        self.restore(ids)
        targets = self.plan_targets()
        for file_id in ids:
            with self.subTest(file_id=file_id):
                self.assertTrue(targets[file_id].startswith("Russia/Moscow/2022/"),
                                targets[file_id])

    def test_a_returned_document_also_goes_back_to_the_city_layout(self):
        # Returning from the document bucket is allowed — the person knows what is in
        # their own file. Only the preview is refused, never the correction.
        file_id = self.add_classified("scan.jpg", "document")
        self.start_server()
        self.restore([file_id])
        self.assertEqual(self.plan_targets()[file_id], "Russia/Moscow/2022/scan.jpg")

    def test_every_bucket_can_be_returned(self):
        ids = {v: self.add_classified(f"{v}.jpg", v)
               for v in ("product", "document", "screenshot", "meme")}
        self.start_server()
        self.restore(list(ids.values()))
        targets = self.plan_targets()
        for verdict, file_id in ids.items():
            with self.subTest(verdict=verdict):
                self.assertEqual(targets[file_id], f"Russia/Moscow/2022/{verdict}.jpg")

    def test_the_card_reports_the_frame_as_returned(self):
        file_id = self.add_classified("p.jpg", "product")
        self.start_server()
        self.assertFalse(self.junk()["items"][0]["restored"])
        self.restore([file_id])
        item = self.junk()["items"][0]
        self.assertEqual(item["file_id"], file_id)
        self.assertTrue(item["restored"])
        # the frame stays in its bucket: media_class still says `product`
        self.assertEqual(item["verdict"], "product")

    def test_another_correction_does_not_read_as_a_return(self):
        file_id = self.add_classified("p.jpg", "product")
        self.start_server()
        self.post("/api/overrides", {"file_ids": [file_id], "action": "exclude"})
        self.assertFalse(self.junk()["items"][0]["restored"])

    def test_clearing_the_correction_sends_the_frame_back_to_its_bucket(self):
        file_id = self.add_classified("p.jpg", "product")
        self.start_server()
        self.restore([file_id])
        self.post("/api/overrides", {"file_ids": [file_id], "action": "clear"})
        self.assertEqual(self.override_rows(), {})
        self.assertFalse(self.junk()["items"][0]["restored"])
        self.assertTrue(self.plan_targets()[file_id].startswith("_Products/"))

    def test_a_second_junk_run_does_not_wipe_the_correction(self):
        # The acceptance criterion: `media_class` is the tier's to rewrite,
        # `manual_overrides` is not — so re-classifying leaves the correction standing.
        file_id = self.add_classified("p.jpg", "product")
        self.start_server()
        self.restore([file_id])
        self.conn.execute(
            "UPDATE media_class SET verdict = 'product', updated_at = '2026-07-29' "
            "WHERE file_id = ?", (file_id,))
        self.conn.commit()
        self.assertEqual(self.override_rows(), {file_id: ("photo", None)})
        self.assertEqual(self.plan_targets()[file_id], "Russia/Moscow/2022/p.jpg")

    def test_photo_with_a_target_is_still_accepted_and_stores_no_target(self):
        # `photo` carries no destination by design — the point is the AUTOMATIC layout.
        file_id = self.add_classified("p.jpg", "product")
        self.start_server()
        status, _payload = self.post(
            "/api/overrides", {"file_ids": [file_id], "action": "photo",
                               "target": "Франция/Париж"})
        self.assertEqual(status, 200)
        self.assertEqual(self.override_rows(), {file_id: ("photo", None)})

    def test_an_unknown_action_is_still_refused(self):
        file_id = self.add_classified("p.jpg", "product")
        self.start_server()
        status, _payload = self.post(
            "/api/overrides", {"file_ids": [file_id], "action": "photos"})
        self.assertEqual(status, 400)
        self.assertEqual(self.override_rows(), {})


class TestReturnedFrameIsMarkedInThePlan(JunkViewTestBase):
    def test_the_city_page_carries_the_photo_mark(self):
        file_id = self.add_classified("p.jpg", "product")
        self.start_server()
        _s, body, _c = self.get("/api/plan?mode=city")
        category = json.loads(body)["categories"][0]["category"]
        self.post("/api/overrides", {"file_ids": [file_id], "action": "photo"})
        _s2, page, _c2 = self.get(
            "/api/plan?mode=city&category=" + urllib.parse.quote(category))
        item = json.loads(page)["items"][0]
        self.assertEqual(item["override"], "photo")
        self.assertIsNone(item["override_target"])


class TestJunkMarkup(JunkViewTestBase):
    def setUp(self):
        super().setUp()
        self.start_server()
        _status, body, _ctype = self.get("/")
        self.html = body.decode("utf-8")

    def test_the_panel_exists_inside_the_slices_tab(self):
        # F133: the buckets became pinned slices — products, screenshots, documents and
        # the rest sit next to people, events and animals, and the panel that renders
        # them is the one this view has always used.
        self.assertIn('id="tab-junk" class="slice-panel"', self.html)
        self.assertNotIn('id="tab-btn-junk"', self.html)
        slices = self.html.split('id="tab-slices"', 1)[1].split("</section", 1)[0]
        self.assertIn('id="tab-junk"', slices)

    def test_the_bulk_controls_are_present(self):
        self.assertIn('id="junk-restore-btn"', self.html)
        self.assertIn('id="junk-select-all-btn"', self.html)
        self.assertIn('id="junk-select-none-btn"', self.html)
        # the bucket chips are the slice pins now, and the pin row is built from data
        self.assertIn('id="slice-pins"', self.html)
        self.assertIn('id="junk-grid"', self.html)

    def test_the_view_reads_the_new_route_and_writes_through_the_old_one(self):
        self.assertIn('"/api/junk?offset="', self.html)
        self.assertIn('postJson("/api/overrides", { file_ids: ids, action: action })',
                      self.html)

    def test_the_returned_state_has_its_own_row_style(self):
        # not "leave alone" (red) and not "moved to a folder" (dashed blue) — a third,
        # non-negative state
        self.assertIn("tr.override-photo", self.html)
        self.assertIn("outline: 2px solid var(--good)", self.html)

    def test_an_empty_bucket_renders_a_message_instead_of_a_spinner(self):
        self.assertIn('stateEl("empty", I18N.junk_empty)', self.html)

    def test_no_external_resources_added(self):
        self.assertNotIn("http://", self.html)
        self.assertNotIn("https://", self.html)
        self.assertNotIn("<link", self.html)


class TestJunkStringsAreTranslated(unittest.TestCase):
    KEYS = ("tab_junk", "junk_intro", "junk_bucket_product",
            "junk_bucket_document", "junk_bucket_screenshot", "junk_bucket_meme",
            "junk_empty", "slice_return_button", "junk_restore_confirm",
            "junk_undo_restore_button", "junk_restored_mark",
            "junk_select_all", "junk_select_none", "junk_load_more",
            "junk_shown_label", "junk_document_no_preview", "junk_document_hint",
            "junk_error_prefix", "error_loading_junk")

    def test_every_new_string_exists_in_all_three_languages(self):
        for key in self.KEYS:
            with self.subTest(key=key):
                entry = ui._UI_STRINGS[key]
                self.assertEqual(set(entry), {"ru", "en", "ja"})
                for lang, value in entry.items():
                    self.assertTrue(value.strip(), f"{key}/{lang} is empty")

    def test_a_bucket_label_exists_for_every_verdict_the_view_can_show(self):
        # The chip label is looked up as `junk_bucket_<verdict>` in JS — a verdict
        # without a key would render as a raw English code next to translated ones.
        for verdict in ("product", "document", "screenshot", "meme"):
            with self.subTest(verdict=verdict):
                self.assertIn(f"junk_bucket_{verdict}", ui._UI_STRINGS)

    def test_the_counted_placeholders_survive_translation(self):
        for lang in ("ru", "en", "ja"):
            with self.subTest(lang=lang):
                self.assertIn("{n}", ui._UI_STRINGS["junk_restore_confirm"][lang])
                self.assertIn("{shown}", ui._UI_STRINGS["junk_shown_label"][lang])
                self.assertIn("{total}", ui._UI_STRINGS["junk_shown_label"][lang])


if __name__ == "__main__":
    unittest.main()
