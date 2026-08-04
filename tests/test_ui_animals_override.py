"""F124: taking a false animal mark off a frame (and putting a missing one back) in the UI.

Three properties, in the order they matter:

* the correction is a row in `manual_pet` and NOTHING else — `frame_quality` keeps its
  single writer, so the next `junk` run cannot quietly undo the user;
* a corrected frame stays VISIBLE on the tab, marked as decided by hand. A card that
  vanishes moves the counter for no reason a reader can see and takes the undo with it;
* the tab, the counter and the album read ONE rule (`sorter.animal_ids_sql`). The case
  that pins this is the one with both directions of correction present at once: two
  expressions of that rule would disagree there first.
"""
from __future__ import annotations

import io
import json
import unittest
import urllib.error
import urllib.request
from contextlib import redirect_stdout

from sorta import ui
from sorta.sorter import plan_album

from tests.test_ui_animals import AnimalsTestBase


class AnimalMarkTestBase(AnimalsTestBase):
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

    def mark(self, file_id: int, action: str) -> tuple[int, dict]:
        return self.post("/api/animals/mark", {"file_ids": [file_id], "action": action})

    def album_ids(self) -> list[int]:
        """The animal album slice, with the CLI chatter swallowed."""
        with redirect_stdout(io.StringIO()):
            report = plan_album(self.cfg, self.conn, "animal", "", self.root / "album")
        return sorted(it.file_id for it in report.plan)

    def tab_animal_ids(self) -> list[int]:
        """The frames the tab presents AS animals (a card marked by hand stays in the
        list — it is `is_animal` that says whether it counts)."""
        return sorted(it["file_id"] for it in self.animals()["items"] if it["is_animal"])

    def overview_animals(self) -> int:
        _status, body, _ctype = self.get("/api/overview")
        return json.loads(body)["collection"]["animals"]

    def manual_rows(self) -> dict[int, int]:
        return {r["file_id"]: r["is_animal"]
                for r in self.conn.execute("SELECT file_id, is_animal FROM manual_pet")}


class TestMarkRoute(AnimalMarkTestBase):
    def test_not_an_animal_writes_one_row_and_leaves_the_frame_on_the_page(self):
        fid, _p, _c = self.add_photo_file("fur_coat.jpg")
        self.mark_animal(fid, score=0.72)
        self.start_server()
        status, resp = self.mark(fid, "not_animal")
        self.assertEqual(status, 200)
        self.assertTrue(resp["ok"])
        self.assertEqual(resp["marked"], 1)
        self.assertEqual(resp["animals"], 0)
        self.assertEqual(self.manual_rows(), {fid: 0})
        # still listed, and listed as decided by hand
        item = self.animals()["items"][0]
        self.assertEqual(item["file_id"], fid)
        self.assertFalse(item["is_animal"])
        self.assertFalse(item["manual"])
        self.assertAlmostEqual(item["score"], 0.72)   # the model's number is still shown

    def test_the_answer_carries_the_redrawn_card(self):
        """The client redraws one card instead of reloading the page, so the card has to
        come back from the server — a second rendering of the rule in JS is how the tab
        and the album start disagreeing."""
        fid, _p, _c = self.add_photo_file("cat.jpg")
        self.mark_animal(fid)
        self.start_server()
        _status, resp = self.mark(fid, "not_animal")
        self.assertEqual([it["file_id"] for it in resp["items"]], [fid])
        self.assertFalse(resp["items"][0]["is_animal"])
        self.assertFalse(resp["items"][0]["manual"])

    def test_it_is_an_animal_adds_a_frame_the_model_left_out(self):
        fid, _p, _c = self.add_photo_file("dark_cat.jpg")
        self.mark_animal(fid, pet=None, score=0.61)
        self.start_server()
        self.assertEqual(self.animals()["animals"], 0)
        status, resp = self.mark(fid, "animal")
        self.assertEqual(status, 200)
        self.assertEqual(resp["animals"], 1)
        self.assertEqual(self.manual_rows(), {fid: 1})
        item = self.animals()["items"][0]
        self.assertTrue(item["is_animal"])
        self.assertTrue(item["manual"])

    def test_clear_hands_the_frame_back_to_the_model(self):
        fid, _p, _c = self.add_photo_file("cat.jpg")
        self.mark_animal(fid)
        self.start_server()
        self.mark(fid, "not_animal")
        self.assertEqual(self.animals()["animals"], 0)
        status, resp = self.mark(fid, "clear")
        self.assertEqual(status, 200)
        self.assertEqual(resp["animals"], 1)
        self.assertEqual(self.manual_rows(), {})
        item = self.animals()["items"][0]
        self.assertTrue(item["is_animal"])
        self.assertIsNone(item["manual"])

    def test_a_cleared_frame_the_model_never_marked_leaves_the_list(self):
        """`items` comes back shorter than the ids, and the client drops that card."""
        fid, _p, _c = self.add_photo_file("never_asked.jpg")
        self.start_server()
        self.mark(fid, "animal")
        self.assertEqual(self.animals()["total"], 1)
        _status, resp = self.mark(fid, "clear")
        self.assertEqual(resp["items"], [])
        self.assertEqual(self.animals()["total"], 0)

    def test_marking_the_same_frame_twice_overwrites_the_row(self):
        fid, _p, _c = self.add_photo_file("cat.jpg")
        self.mark_animal(fid)
        self.start_server()
        self.mark(fid, "not_animal")
        self.mark(fid, "animal")
        self.assertEqual(self.manual_rows(), {fid: 1})
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM manual_pet").fetchone()[0], 1)

    def test_the_route_never_touches_frame_quality(self):
        fid, _p, _c = self.add_photo_file("cat.jpg")
        self.mark_animal(fid, score=0.88)
        self.start_server()
        before = dict(self.conn.execute(
            "SELECT pet, pet_score, source FROM frame_quality").fetchone())
        self.mark(fid, "not_animal")
        after = dict(self.conn.execute(
            "SELECT pet, pet_score, source FROM frame_quality").fetchone())
        self.assertEqual(before, after)

    def test_an_unknown_id_is_skipped_rather_than_written(self):
        self.start_server()
        status, resp = self.mark(9999, "not_animal")
        self.assertEqual(status, 200)
        self.assertEqual(resp["marked"], 0)
        self.assertEqual(self.manual_rows(), {})

    def test_an_unknown_action_is_a_400(self):
        fid, _p, _c = self.add_photo_file("cat.jpg")
        self.mark_animal(fid)
        self.start_server()
        status, resp = self.mark(fid, "maybe")
        self.assertEqual(status, 400)
        self.assertIn("error", resp)
        self.assertEqual(self.manual_rows(), {})

    def test_a_body_without_ids_is_a_400(self):
        self.start_server()
        for body in ({"action": "not_animal"},
                     {"file_ids": [], "action": "not_animal"},
                     {"file_ids": ["1"], "action": "not_animal"},
                     {"file_ids": [1]}):
            with self.subTest(body=body):
                status, resp = self.post("/api/animals/mark", body)
                self.assertEqual(status, 400)
                self.assertIn("error", resp)


class TestOneReadRule(AnimalMarkTestBase):
    """Brief test 6: the album slice and the tab answer with the same set."""

    def _mixed_collection(self) -> dict[str, int]:
        """One frame of every combination the rule has to decide."""
        ids = {}
        for name, pet, score in (("cat.jpg", "animal", 0.95),
                                 ("fur_coat.jpg", "animal", 0.72),
                                 ("dark_cat.jpg", None, 0.61),
                                 ("desk.jpg", None, 0.10)):
            fid, _p, _c = self.add_photo_file(name)
            self.mark_animal(fid, pet=pet, score=score)
            ids[name] = fid
        fid, _p, _c = self.add_photo_file("never_asked.jpg")
        ids["never_asked.jpg"] = fid
        return ids

    def test_both_directions_of_correction_leave_the_two_in_agreement(self):
        ids = self._mixed_collection()
        self.start_server()
        self.mark(ids["fur_coat.jpg"], "not_animal")   # a false mark taken off
        self.mark(ids["dark_cat.jpg"], "animal")       # a missing one put on
        expected = sorted([ids["cat.jpg"], ids["dark_cat.jpg"]])
        self.assertEqual(self.tab_animal_ids(), expected)
        self.assertEqual(self.album_ids(), expected)
        self.assertEqual(self.animals()["animals"], len(expected))
        self.assertEqual(self.overview_animals(), len(expected))

    def test_a_frame_added_without_any_frame_quality_row_reaches_the_album_too(self):
        ids = self._mixed_collection()
        self.start_server()
        self.mark(ids["never_asked.jpg"], "animal")
        expected = sorted([ids["cat.jpg"], ids["fur_coat.jpg"], ids["never_asked.jpg"]])
        self.assertEqual(self.tab_animal_ids(), expected)
        self.assertEqual(self.album_ids(), expected)

    def test_without_any_manual_row_everything_answers_as_before(self):
        ids = self._mixed_collection()
        self.start_server()
        expected = sorted([ids["cat.jpg"], ids["fur_coat.jpg"]])
        self.assertEqual(self.tab_animal_ids(), expected)
        self.assertEqual(self.album_ids(), expected)
        self.assertEqual(self.overview_animals(), 2)
        for item in self.animals()["items"]:
            self.assertIsNone(item["manual"])

    def test_the_list_is_longer_than_the_count_once_a_frame_is_unmarked(self):
        """`total` is the length of the list, `animals` is how many of it count — after
        a correction those stop being the same number, and the payload says both."""
        ids = self._mixed_collection()
        self.start_server()
        self.mark(ids["fur_coat.jpg"], "not_animal")
        data = self.animals()
        self.assertEqual(data["total"], 2)      # the coat is still on the page
        self.assertEqual(data["animals"], 1)

    def test_the_unmarked_frame_keeps_its_place_in_the_confidence_order(self):
        """A decision must not reshuffle the list under a reader walking down it."""
        ids = self._mixed_collection()
        self.start_server()
        before = [it["file_id"] for it in self.animals()["items"]]
        self.mark(ids["fur_coat.jpg"], "not_animal")
        self.assertEqual([it["file_id"] for it in self.animals()["items"]], before)

    def test_a_duplicate_marked_by_hand_stays_out_of_all_three(self):
        canonical, _p, _c = self.add_photo_file("a.jpg")
        duplicate, _p2, _c2 = self.add_photo_file("b.jpg")
        self.conn.execute("UPDATE files SET dup_of = ? WHERE id = ?",
                          (canonical, duplicate))
        self.conn.commit()
        self.start_server()
        self.mark(duplicate, "animal")
        self.assertEqual(self.tab_animal_ids(), [])
        self.assertEqual(self.album_ids(), [])
        self.assertEqual(self.overview_animals(), 0)


class TestTabStaysUsableAfterAMark(AnimalMarkTestBase):
    def test_the_tab_is_still_offered_when_every_frame_has_been_unmarked(self):
        """Otherwise the undo button leaves with the tab that holds it."""
        fid, _p, _c = self.add_photo_file("fur_coat.jpg")
        self.mark_animal(fid)
        self.start_server()
        self.mark(fid, "not_animal")
        _status, body, _ctype = self.get("/api/tabs/visibility")
        self.assertTrue(json.loads(body)["animal"])

    def test_paging_walks_the_list_including_the_unmarked_frames(self):
        ids = []
        for i in range(4):
            fid, _p, _c = self.add_photo_file(f"a{i}.jpg")
            self.mark_animal(fid, score=0.9 - i / 100)
            ids.append(fid)
        self.start_server()
        self.mark(ids[1], "not_animal")
        page1 = self.animals("?offset=0&limit=2")
        page2 = self.animals("?offset=2&limit=2")
        seen = [it["file_id"] for it in page1["items"] + page2["items"]]
        self.assertEqual(seen, ids)
        self.assertEqual(page1["total"], 4)
        self.assertEqual(page1["animals"], 3)


class TestMarkUiHtml(AnimalMarkTestBase):
    def test_the_card_carries_the_toggle_and_the_way_back(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn('"/api/animals/mark"', html)
        self.assertIn("animal-mark-btn", html)
        self.assertIn("animal-clear-btn", html)
        self.assertIn('id="animals-counted"', html)
        self.assertIn('id="animals-mark-status"', html)
        self.assertIn("markAnimal(item.file_id", html)

    def test_no_route_marks_a_whole_band(self):
        """The feature exists because somebody LOOKED at the frame; a threshold is
        already there for the other case."""
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertNotIn("animals-select-all", html)
        self.assertNotIn("animals-mark-all", html)

    def test_every_new_string_is_translated_three_ways(self):
        keys = ("slice_return_button", "animals_mark_animal", "animals_mark_clear",
                "animals_manual_excluded", "animals_manual_included",
                "animals_counted_label", "animals_error_prefix")
        for key in keys:
            with self.subTest(key=key):
                entry = ui._UI_STRINGS[key]
                self.assertEqual(set(entry), {"ru", "en", "ja"})
                for lang in ("ru", "en", "ja"):
                    self.assertTrue(entry[lang].strip())
        for lang in ("ru", "en", "ja"):
            self.assertIn("{n}", ui._UI_STRINGS["animals_counted_label"][lang])

    def test_the_buttons_are_rendered_in_the_page_language(self):
        # F174: the "take the mark off" half is the shared `slice_return_button` now —
        # one name for one intention, the same words the junk view uses.
        self.start_server()
        for lang, expected in (("ru", "Вернуть в раскладку"),
                               ("en", "Return to the layout"),
                               ("ja", "振り分けに戻す")):
            with self.subTest(lang=lang):
                _status, body, _ctype = self.get(f"/?lang={lang}")
                self.assertIn(expected, body.decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
