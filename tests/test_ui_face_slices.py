"""F152: the three face slices — with people / group photos / portraits.

What this file guards, in the order the risk runs:

* THE trap. A `faces` row with `bbox = '[]'` is the marker "this file was processed and
  had no face on it", and 24 195 of 24 196 live files carry one. A predicate that forgets
  to exclude it turns "photographs with people" into "every photograph", which is F125's
  mistake repeated. Every slice, every counter and every album is checked against it;
* the two geometric rules are the rules the config says: two faces are not a group and
  three are, one big face is a portrait and one small one is not, two faces are never a
  portrait however big they are;
* the population is the canonical photographs — no duplicates, no unreadable files —
  and it is the same population the "Overview" counter and the album gather;
* without a faces run the answer is the REASON and not a zero, everywhere it appears:
  the panel, the pin counters and the "Overview" rows.
"""
from __future__ import annotations

import dataclasses
import json
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from sorta import ui

from tests.test_ui import UiServerTestBase


class FaceSliceTestBase(UiServerTestBase):
    """Photographs with `faces` rows on top of the U1 server.

    `width`/`height` are set on every frame because the portrait rule is geometry over
    them — a file the index never measured cannot be in that slice, and a fixture that
    silently left them NULL would make the portrait cases pass for the wrong reason.
    """

    def add_frame(self, rel: str, *, width: int = 1000, height: int = 1000) -> int:
        file_id, _p, _c = self.add_photo_file(rel)
        self.conn.execute("UPDATE files SET width = ?, height = ? WHERE id = ?",
                          (width, height, file_id))
        self.conn.commit()
        return file_id

    def add_face(self, file_id: int, bbox: str = "[0,0,100,100]") -> None:
        self.conn.execute(
            "INSERT INTO faces (file_id, bbox, embedding) VALUES (?, ?, ?)",
            (file_id, bbox, b"embedding"))
        self.conn.commit()

    def add_face_marker(self, file_id: int) -> None:
        """`bbox = '[]'` — "processed, no faces here". Not a face."""
        self.conn.execute(
            "INSERT INTO faces (file_id, bbox, embedding) VALUES (?, '[]', ?)",
            (file_id, b""))
        self.conn.commit()

    def add_square_face(self, file_id: int, side: float, *, at: float = 0.0) -> None:
        """A face box of `side` x `side` pixels — the portrait share made readable."""
        self.add_face(file_id, f"[{at},{at},{at + side},{at + side}]")

    def slice_(self, name: str = "people", query: str = "") -> dict:
        status, body, ctype = self.get(f"/api/face-slices?slice={name}{query}")
        self.assertEqual(status, 200, body)
        self.assertIn("application/json", ctype)
        return json.loads(body)

    def counts(self, data: dict) -> dict:
        return {row["slice"]: row["count"] for row in data["counts"]}

    def ids(self, data: dict) -> list[int]:
        return [item["file_id"] for item in data["items"]]

    def post(self, path: str, data: object) -> tuple[int, dict]:
        body = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}{path}", data=body, method="POST",
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())


class TestTheMarkerRowIsNeverAFace(FaceSliceTestBase):
    """THE test of this feature. `bbox = '[]'` means "looked, found nothing"."""

    def test_a_frame_with_only_the_marker_is_not_in_the_people_slice(self):
        marked = self.add_frame("no-people.jpg")
        self.add_face_marker(marked)
        real = self.add_frame("people.jpg")
        self.add_face(real)
        self.start_server()
        data = self.slice_("people")
        self.assertEqual(self.ids(data), [real])
        self.assertEqual(self.counts(data)["people"], 1)

    def test_a_collection_of_markers_alone_gives_an_empty_slice_not_all_of_it(self):
        # The live shape of the failure: nearly every file carries a marker row, so a
        # predicate without the exclusion answers with the whole collection.
        for i in range(5):
            self.add_face_marker(self.add_frame(f"m{i}.jpg"))
        real = self.add_frame("real.jpg")
        self.add_face(real)
        self.start_server()
        data = self.slice_("people")
        self.assertEqual(data["total"], 1)
        self.assertEqual(self.ids(data), [real])

    def test_markers_do_not_count_towards_a_group(self):
        fid = self.add_frame("two-and-a-marker.jpg")
        self.add_face(fid)
        self.add_face(fid)
        self.add_face_marker(fid)   # cannot happen in practice; must not count if it does
        self.start_server()
        self.assertEqual(self.slice_("group")["total"], 0)

    def test_markers_do_not_make_a_frame_a_portrait(self):
        fid = self.add_frame("marker-and-one-face.jpg")
        self.add_square_face(fid, 500)
        self.add_face_marker(fid)
        self.start_server()
        # COUNT(*) = 1 has to be a count of REAL faces, not of rows.
        self.assertEqual(self.ids(self.slice_("portrait")), [fid])

    def test_the_marker_is_not_counted_on_the_card_either(self):
        fid = self.add_frame("one.jpg")
        self.add_face(fid)
        self.add_face_marker(fid)
        self.start_server()
        self.assertEqual(self.slice_("people")["items"][0]["faces"], 1)


class TestGroupPhotos(FaceSliceTestBase):
    def _with_faces(self, rel: str, n: int) -> int:
        fid = self.add_frame(rel)
        for _ in range(n):
            self.add_face(fid)
        return fid

    def test_two_faces_are_not_a_group_and_three_are(self):
        self._with_faces("pair.jpg", 2)
        trio = self._with_faces("trio.jpg", 3)
        self.start_server()
        data = self.slice_("group")
        self.assertEqual(self.ids(data), [trio])
        self.assertEqual(self.counts(data)["group"], 1)

    def test_the_threshold_comes_from_the_config(self):
        pair = self._with_faces("pair.jpg", 2)
        trio = self._with_faces("trio.jpg", 3)
        self.cfg.features = dataclasses.replace(self.cfg.features, group_photo_faces=2)
        self.start_server()
        self.assertEqual(sorted(self.ids(self.slice_("group"))), sorted([pair, trio]))

    def test_a_raised_threshold_narrows_the_slice(self):
        self._with_faces("trio.jpg", 3)
        many = self._with_faces("crowd.jpg", 6)
        self.cfg.features = dataclasses.replace(self.cfg.features, group_photo_faces=5)
        self.start_server()
        self.assertEqual(self.ids(self.slice_("group")), [many])

    def test_the_hint_carries_the_number_the_query_used(self):
        self.cfg.features = dataclasses.replace(self.cfg.features, group_photo_faces=4)
        self.add_face(self.add_frame("a.jpg"))
        self.start_server()
        self.assertEqual(self.slice_("group")["group_min"], 4)


class TestPortraits(FaceSliceTestBase):
    """One face, and it takes a noticeable share of the frame. Pure geometry."""

    def test_one_big_face_is_a_portrait_and_one_small_face_is_not(self):
        # 1000x1000 frames: 0.08 of the area is a 283 px side.
        big = self.add_frame("big.jpg")
        self.add_square_face(big, 400)          # 0.16 of the frame
        small = self.add_frame("small.jpg")
        self.add_square_face(small, 100)        # 0.01 of the frame
        self.start_server()
        data = self.slice_("portrait")
        self.assertEqual(self.ids(data), [big])
        self.assertEqual(self.counts(data)["portrait"], 1)

    def test_two_faces_are_never_a_portrait_however_big(self):
        fid = self.add_frame("two-big.jpg")
        self.add_square_face(fid, 500)
        self.add_square_face(fid, 500, at=500)
        self.start_server()
        self.assertEqual(self.slice_("portrait")["total"], 0)

    def test_the_share_comes_from_the_config(self):
        fid = self.add_frame("mid.jpg")
        self.add_square_face(fid, 200)          # 0.04 of the frame — below the default
        self.start_server()
        self.assertEqual(self.slice_("portrait")["total"], 0)
        # The threshold is read off the live config on every request (the `/api/junk`
        # rule), so lowering it takes effect without a restart.
        self.cfg.features = dataclasses.replace(
            self.cfg.features, portrait_face_share=0.03)
        self.assertEqual(self.ids(self.slice_("portrait")), [fid])

    def test_the_share_is_of_the_frame_and_not_of_a_fixed_size(self):
        # The same 400 px face is a portrait on a 1000 px frame and not on a 4000 px one.
        small_frame = self.add_frame("close.jpg", width=1000, height=1000)
        self.add_square_face(small_frame, 400)
        big_frame = self.add_frame("far.jpg", width=4000, height=4000)
        self.add_square_face(big_frame, 400)
        self.start_server()
        self.assertEqual(self.ids(self.slice_("portrait")), [small_frame])

    def test_a_frame_without_dimensions_is_not_in_the_slice(self):
        # The share cannot be computed for it, and inventing one would be a guess.
        fid, _p, _c = self.add_photo_file("nodims.jpg")
        self.add_square_face(fid, 900)
        self.start_server()
        self.assertEqual(self.slice_("portrait")["total"], 0)
        self.assertEqual(self.ids(self.slice_("people")), [fid])  # still has a person

    def test_the_hint_carries_the_share_the_query_used(self):
        self.add_face(self.add_frame("a.jpg"))
        self.start_server()
        self.assertAlmostEqual(self.slice_("portrait")["portrait_share"], 0.08)


class TestThePopulation(FaceSliceTestBase):
    def test_duplicates_and_read_errors_are_excluded(self):
        canonical = self.add_frame("a.jpg")
        duplicate = self.add_frame("b.jpg")
        broken = self.add_frame("c.jpg")
        for fid in (canonical, duplicate, broken):
            self.add_square_face(fid, 500)
        self.conn.execute("UPDATE files SET dup_of = ? WHERE id = ?",
                          (canonical, duplicate))
        self.conn.execute("UPDATE files SET error = 'nope' WHERE id = ?", (broken,))
        self.conn.commit()
        self.start_server()
        for name in ("people", "portrait"):
            with self.subTest(slice=name):
                data = self.slice_(name)
                self.assertEqual(self.ids(data), [canonical])
                self.assertEqual(data["total"], 1)

    def test_paging_walks_the_whole_slice_without_repeats(self):
        for i in range(5):
            self.add_face(self.add_frame(f"a{i}.jpg"))
        self.start_server()
        seen = []
        for offset in (0, 2, 4):
            page = self.slice_("people", f"&offset={offset}&limit=2")
            self.assertEqual(page["total"], 5)
            seen += self.ids(page)
        self.assertEqual(len(seen), 5)
        self.assertEqual(len(set(seen)), 5)

    def test_a_card_carries_the_number_of_faces_and_no_score(self):
        fid = self.add_frame("trio.jpg")
        for _ in range(3):
            self.add_face(fid)
        self.start_server()
        item = self.slice_("group")["items"][0]
        self.assertEqual(item["file_id"], fid)
        self.assertEqual(item["faces"], 3)
        self.assertEqual(item["thumb_url"], f"/thumb/{fid}")
        self.assertEqual(item["date"], "2022-05-01T10:00:00")
        # These slices are facts, not rankings: there is no confidence to print.
        self.assertNotIn("score", item)

    def test_a_sensitive_class_is_listed_without_a_thumbnail(self):
        # F133's rule: a document with a face on it is never decoded for display.
        fid = self.add_frame("passport.jpg")
        self.add_square_face(fid, 500)
        self.conn.execute(
            """INSERT INTO media_class (file_id, verdict, source, updated_at)
               VALUES (?, 'document', 'clip', '2026-01-01')""", (fid,))
        self.conn.commit()
        self.start_server()
        item = self.slice_("people")["items"][0]
        self.assertEqual(item["file_id"], fid)
        self.assertNotIn("thumb_url", item)

    def test_an_unknown_slice_is_a_400(self):
        self.start_server()
        status, body, _ctype = self.get("/api/face-slices?slice=children")
        self.assertEqual(status, 400)
        self.assertIn("error", json.loads(body))

    def test_a_bad_offset_is_a_400(self):
        self.start_server()
        status, body, _ctype = self.get("/api/face-slices?offset=nope")
        self.assertEqual(status, 400)
        self.assertIn("error", json.loads(body))


class TestWithoutAFacesRun(FaceSliceTestBase):
    """F125's rule: a reason, never a zero. Nothing was measured, so nothing is claimed."""

    def test_the_answer_is_a_reason_and_the_counters_are_null(self):
        self.add_frame("a.jpg")
        self.start_server()
        data = self.slice_("people")
        self.assertEqual(data["reason"], "no_faces_run")
        self.assertEqual(self.counts(data),
                         {"people": None, "group": None, "portrait": None})
        self.assertEqual(data["items"], [])

    def test_a_collection_of_markers_alone_still_counts_as_no_run(self):
        # The stage touched every file and found nothing: `faces_stage_ran` is about REAL
        # faces, so this is still "there is nothing to say", not "0 people".
        self.add_face_marker(self.add_frame("a.jpg"))
        self.start_server()
        self.assertEqual(self.slice_("people")["reason"], "no_faces_run")

    def test_one_real_face_switches_the_counters_on(self):
        self.add_face(self.add_frame("a.jpg"))
        self.add_face_marker(self.add_frame("b.jpg"))
        self.start_server()
        data = self.slice_("people")
        self.assertIsNone(data["reason"])
        self.assertEqual(self.counts(data), {"people": 1, "group": 0, "portrait": 0})

    def test_a_zero_after_a_run_is_an_answer_and_stays_a_zero(self):
        self.add_face(self.add_frame("a.jpg"))
        self.start_server()
        data = self.slice_("group")
        self.assertIsNone(data["reason"])
        self.assertEqual(data["total"], 0)


class TestOverviewCounters(FaceSliceTestBase):
    def overview(self) -> dict:
        _status, body, _ctype = self.get("/api/overview")
        return json.loads(body)["collection"]

    def test_every_counter_equals_the_length_of_its_slice(self):
        alone = self.add_frame("alone.jpg")
        self.add_square_face(alone, 500)
        trio = self.add_frame("trio.jpg")
        for i in range(3):
            self.add_square_face(trio, 50, at=i * 60)
        self.start_server()
        collection = self.overview()
        self.assertEqual(collection["with_people"], 2)
        self.assertEqual(collection["group_photos"], 1)
        self.assertEqual(collection["portraits"], 1)
        self.assertEqual(collection["with_people"], self.slice_("people")["total"])
        self.assertEqual(collection["group_photos"], self.slice_("group")["total"])
        self.assertEqual(collection["portraits"], self.slice_("portrait")["total"])

    def test_the_marker_row_does_not_inflate_the_counter(self):
        for i in range(4):
            self.add_face_marker(self.add_frame(f"m{i}.jpg"))
        self.add_face(self.add_frame("real.jpg"))
        self.start_server()
        self.assertEqual(self.overview()["with_people"], 1)

    def test_without_a_faces_run_the_rows_are_null_with_a_reason(self):
        self.add_frame("a.jpg")
        self.start_server()
        collection = self.overview()
        self.assertIsNone(collection["with_people"])
        self.assertIsNone(collection["group_photos"])
        self.assertIsNone(collection["portraits"])
        self.assertEqual(collection["faces_reason"], "no_faces_run")

    def test_the_thresholds_of_the_config_reach_the_overview_too(self):
        pair = self.add_frame("pair.jpg")
        self.add_face(pair)
        self.add_face(pair)
        self.cfg.features = dataclasses.replace(self.cfg.features, group_photo_faces=2)
        self.start_server()
        self.assertEqual(self.overview()["group_photos"], 1)


class TestAlbums(FaceSliceTestBase):
    """The same actions as every other slice (F139): link, copy, move."""

    def test_a_preview_counts_the_slice_without_writing_anything(self):
        fid = self.add_frame("a.jpg")
        self.add_square_face(fid, 500)
        self.start_server()
        status, body = self.post(
            "/api/album", {"kind": "people", "mode": "link", "apply": False})
        self.assertEqual(status, 200)
        self.assertEqual(body["count"], 1)
        self.assertFalse(body["applied"])
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM move_batches").fetchone()[0], 0)

    def test_every_kind_gathers_into_its_own_album_batch(self):
        alone = self.add_frame("alone.jpg")
        self.add_square_face(alone, 500)
        trio = self.add_frame("trio.jpg")
        for i in range(3):
            self.add_square_face(trio, 50, at=i * 60)
        self.start_server()
        for kind, expected in (("people", 2), ("group", 1), ("portrait", 1)):
            with self.subTest(kind=kind):
                status, body = self.post(
                    "/api/album", {"kind": kind, "mode": "link", "apply": True})
                self.assertEqual(status, 200)
                self.assertEqual(body["transferred"], expected)
                self.assertEqual(body["failed"], 0)
                batch = self.conn.execute(
                    "SELECT mode, operation FROM move_batches "
                    "ORDER BY id DESC LIMIT 1").fetchone()
                self.assertEqual(batch["mode"], f"album_{kind}")
                self.assertEqual(batch["operation"], "link")

    def test_the_marker_row_does_not_reach_the_album(self):
        self.add_face_marker(self.add_frame("marker.jpg"))
        real = self.add_frame("real.jpg")
        self.add_face(real)
        path = Path(self.conn.execute(
            "SELECT path FROM files WHERE id = ?", (real,)).fetchone()[0])
        self.start_server()
        status, body = self.post(
            "/api/album", {"kind": "people", "mode": "link", "apply": True})
        self.assertEqual(status, 200)
        self.assertEqual(body["transferred"], 1)
        self.assertTrue((Path(body["dest"]) / path.name).exists())

    def test_a_copy_album_is_the_same_slice_in_another_mode(self):
        fid = self.add_frame("a.jpg")
        self.add_square_face(fid, 500)
        self.start_server()
        status, body = self.post(
            "/api/album", {"kind": "portrait", "mode": "copy", "apply": True})
        self.assertEqual(status, 200)
        self.assertEqual(body["transferred"], 1)
        self.assertEqual(
            self.conn.execute(
                "SELECT operation FROM move_batches ORDER BY id DESC LIMIT 1"
            ).fetchone()[0], "copy")

    def test_a_missing_selector_is_accepted_for_these_kinds(self):
        self.add_face(self.add_frame("a.jpg"))
        self.start_server()
        for kind in ("people", "group", "portrait"):
            with self.subTest(kind=kind):
                status, _body = self.post("/api/album", {"kind": kind, "mode": "link"})
                self.assertEqual(status, 200)
        # and it is still refused where the selector IS the subject
        status, _body = self.post("/api/album", {"kind": "person", "mode": "link"})
        self.assertEqual(status, 400)

    def test_the_album_folder_follows_the_interface_language(self):
        self.add_face(self.add_frame("a.jpg"))
        self.cfg.raw = {"language": "ru"}
        self.start_server()
        _status, body = self.post("/api/album", {"kind": "people", "mode": "link"})
        self.assertEqual(Path(body["dest"]).name, "_С людьми")


class TestFaceSlicesHtml(FaceSliceTestBase):
    def test_the_panel_and_its_paging_are_in_the_markup(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn('id="tab-face"', html)
        self.assertIn('id="face-grid"', html)
        self.assertIn('id="face-more-btn"', html)
        self.assertIn('id="face-album"', html)
        self.assertIn('id="face-hint"', html)
        self.assertIn('"/api/face-slices?slice="', html)
        self.assertIn("var FACE_PAGE_SIZE = 200;", html)

    def test_the_pins_are_built_from_data_and_not_written_out(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertNotIn('id="slice-pin-face-people"', html)
        self.assertIn("face: !!data.face,", html)
        self.assertIn('pins.push({ key: "face:" + name', html)

    def test_i18n_ru_en_ja(self):
        self.start_server()
        for lang, expected in (("ru", "Групповые"), ("en", "Group photos"),
                               ("ja", "集合写真")):
            with self.subTest(lang=lang):
                _status, body, _ctype = self.get(f"/?lang={lang}")
                self.assertIn(expected, body.decode("utf-8"))

    def test_every_new_string_is_translated_three_ways(self):
        keys = ("face_slice_people", "face_slice_group", "face_slice_portrait",
                "face_slices_intro", "face_hint_people", "face_hint_group",
                "face_hint_portrait", "face_no_faces_run", "face_empty",
                "face_count_label", "face_load_more", "face_shown_label",
                "error_loading_face_slices", "overview_with_people",
                "overview_group_photos", "overview_portraits")
        for key in keys:
            with self.subTest(key=key):
                entry = ui._UI_STRINGS[key]
                self.assertEqual(set(entry), {"ru", "en", "ja"})
                for lang in ("ru", "en", "ja"):
                    self.assertTrue(entry[lang].strip())
        for lang in ("ru", "en", "ja"):
            self.assertIn("{n}", ui._UI_STRINGS["face_hint_group"][lang])
            self.assertIn("{share}", ui._UI_STRINGS["face_hint_portrait"][lang])
            self.assertIn("{n}", ui._UI_STRINGS["face_count_label"][lang])
            self.assertIn("{shown}", ui._UI_STRINGS["face_shown_label"][lang])
            self.assertIn("{total}", ui._UI_STRINGS["face_shown_label"][lang])


class TestTabVisibility(FaceSliceTestBase):
    def visibility(self) -> dict:
        _status, body, _ctype = self.get("/api/tabs/visibility")
        return json.loads(body)

    def test_hidden_on_an_empty_index(self):
        self.start_server()
        self.assertFalse(self.visibility()["face"])

    def test_shown_before_any_faces_run(self):
        # Deliberately NOT the F133 rule: the empty state of these slices is a sentence
        # ("the faces stage has not run"), and a pin that hides itself never says it.
        self.add_frame("a.jpg")
        self.start_server()
        self.assertTrue(self.visibility()["face"])

    def test_a_duplicate_alone_does_not_raise_the_pins(self):
        canonical = self.add_frame("a.jpg")
        duplicate = self.add_frame("b.jpg")
        self.conn.execute("UPDATE files SET dup_of = ? WHERE id = ?",
                          (canonical, duplicate))
        self.conn.execute("UPDATE files SET error = 'nope' WHERE id = ?", (canonical,))
        self.conn.commit()
        self.start_server()
        self.assertFalse(self.visibility()["face"])


if __name__ == "__main__":
    unittest.main()
