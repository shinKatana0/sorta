"""F193: every slice offers the same thing — pick some frames, name the folder, gather.

Three complaints of 2026-08-04 were ONE defect, and fixing them one at a time would have
meant answering "what is a slice and what can be done with it" three times, differently:

    4. there is a gather button but no way to pick photographs — so it is all or nothing
    5. Portraits / With people / Group differ: memes and screenshots let you name the
       folder — the behaviour has to be the same
    6. the documents slice: no button and no way to gather — the same again

So the tests here are about the PROPERTY rather than about three slices. The first one
enumerates — it walks the album kinds the engine knows and the panels the page draws,
instead of naming them one by one, because a test that names them is a test the next slice
falls out of.

Documents get their own class, because that is the one where it is easy to be wrong. The
decision is F139's and stands: `document` is not an album kind, and emptying
`vlm.exclude_classes` does not make one — that key decides what is SHOWN, never that a
folder of somebody's passports may be assembled in one click. What F193 changes is that
the refusal is now SAID: it comes out of the route with a reason, and the bucket carries an
album row like every other slice instead of a silence that forbade nothing.
"""
from __future__ import annotations

import dataclasses
import json
import re
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from sorta import ui
from sorta.sorter import (
    ALBUM_KINDS, CLASS_ALBUM_KINDS, FACE_ALBUM_KINDS, QUALITY_ALBUM_KINDS,
    SELECTORLESS_ALBUM_KINDS,
)

from tests.test_search import unit
from tests.test_ui_search import SearchUiTestBase


class SliceUniformityTestBase(SearchUiTestBase):
    """The search fixture's fake text tower comes along: `kind='query'` is a slice like
    any other here, and it must be asked the same questions as the rest without a model
    being loaded for it."""

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

    def album(self, kind: str, **extra) -> tuple[int, dict]:
        body: dict = {"kind": kind, "mode": "link", "apply": False}
        body.update(extra)
        return self.post("/api/album", body)

    def junk(self, query: str = "") -> dict:
        _status, body, _ctype = self.get("/api/junk" + query)
        return json.loads(body)

    def add_classified(self, rel: str, verdict: str) -> int:
        file_id, _path, _content = self.add_photo_file(rel, country="ru", city="Moscow")
        self.conn.execute(
            """INSERT INTO media_class (file_id, verdict, source, updated_at, tier)
               VALUES (?, ?, 'vlm', '2026-07-28', 'vlm')""", (file_id, verdict))
        self.conn.commit()
        return file_id

    def add_face_photo(self, rel: str, faces: int = 1) -> int:
        """A photograph the faces stage found `faces` real boxes on."""
        file_id, _path, _content = self.add_photo_file(rel)
        self.conn.execute("UPDATE files SET width = 100, height = 100 WHERE id = ?",
                          (file_id,))
        for i in range(faces):
            self.conn.execute(
                "INSERT INTO faces (file_id, bbox, embedding) VALUES (?, ?, ?)",
                (file_id, json.dumps([i, 0, i + 90, 90]), b"embedding"))
        self.conn.commit()
        return file_id

    def add_animal(self, rel: str, score: float = 0.9) -> int:
        file_id = self.add_classified(rel, "photo")
        self.conn.execute(
            """INSERT INTO frame_quality (file_id, pet_score, source, updated_at)
               VALUES (?, ?, 'clip', '2026-01-01')""", (file_id, score))
        self.conn.commit()
        return file_id


# --- the enumerating test: the property, not the three slices -------------------------


class TestEverySliceOffersTheSameThing(SliceUniformityTestBase):
    """Requirement 1: walk the slices, do not name them.

    `ALBUM_KINDS` is the engine's own list of what a slice can be gathered as, so a kind
    added tomorrow lands in this loop by existing. `person`, `event` and `query` need a
    subject and are asked with one; the rest are asked without, through the same shared
    `SELECTORLESS_ALBUM_KINDS` the route reads.
    """

    def setUp(self):
        super().setUp()
        # A frame for every population these kinds select over, so no kind is answered
        # "200, zero files" for want of a fixture — the ranking of `query` included, which
        # needs a vector in the search index before it will rank anything at all.
        self.product = self.add_classified("chair.jpg", "product")
        self.store_vector(self.product, unit(1.0))
        self.animal = self.add_animal("cat.jpg")
        self.face = self.add_face_photo("dad.jpg")
        self.start_server()

    def selector_for(self, kind: str) -> str:
        if kind in SELECTORLESS_ALBUM_KINDS:
            return ""
        return "1" if kind == "event" else "somebody"

    def test_every_kind_takes_a_folder_name(self):
        """Complaint 5: the folder name was a privilege of memes and screenshots."""
        for kind in ALBUM_KINDS:
            with self.subTest(kind=kind):
                status, body = self.album(
                    kind, selector=self.selector_for(kind), name="Моя папка")
                self.assertEqual(status, 200)
                self.assertEqual(body["album_name"], "Моя папка")
                self.assertEqual(Path(body["dest"]).name, "Моя папка")

    def test_every_kind_takes_a_selection_of_frames(self):
        """Complaint 4: it used to be the whole slice or nothing."""
        for kind in ALBUM_KINDS:
            with self.subTest(kind=kind):
                status, body = self.album(
                    kind, selector=self.selector_for(kind),
                    file_ids=[self.product, self.animal, self.face])
                self.assertEqual(status, 200)
                self.assertLessEqual(body["count"], 3)

    def test_every_kind_refuses_an_empty_selection_with_a_reason(self):
        """Requirement 3: a clear refusal, never an album of nothing.

        "Nobody ticked anything" and "this slice is empty" are different sentences and
        only one of them is about the collection — a folder of zero files states the
        second one silently.
        """
        for kind in ALBUM_KINDS:
            with self.subTest(kind=kind):
                status, body = self.album(
                    kind, selector=self.selector_for(kind), file_ids=[])
                self.assertEqual(status, 400)
                self.assertEqual(body["reason"], ui._ALBUM_NO_SELECTION)
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM move_batches").fetchone()[0], 0)

    def test_the_kinds_of_this_tab_are_all_of_the_ones_that_ship(self):
        # The loop above is only worth what its list is: if a slice existed outside
        # `ALBUM_KINDS`, it would be a slice with no album, which is the defect itself.
        self.assertEqual(
            set(ALBUM_KINDS),
            {"person", "event", "animal", "query"} | set(CLASS_ALBUM_KINDS)
            | set(QUALITY_ALBUM_KINDS) | set(FACE_ALBUM_KINDS))


class TestTheSelectionGathersExactlyTheTickedFrames(SliceUniformityTestBase):
    """Requirement 2, and the guard that has to come with it."""

    def setUp(self):
        super().setUp()
        self.chair = self.add_classified("chair.jpg", "product")
        self.lamp = self.add_classified("lamp.jpg", "product")
        self.table = self.add_classified("table.jpg", "product")
        self.start_server()

    def test_only_the_selected_frames_land_in_the_folder(self):
        status, body = self.album("product", file_ids=[self.chair, self.table],
                                  name="Выбранное", apply=True)
        self.assertEqual(status, 200)
        self.assertEqual(body["count"], 2)
        self.assertEqual(body["transferred"], 2)
        dest = Path(body["dest"])
        self.assertTrue((dest / "chair.jpg").exists())
        self.assertTrue((dest / "table.jpg").exists())
        self.assertFalse((dest / "lamp.jpg").exists())

    def test_the_whole_slice_is_still_gathered_without_a_selection(self):
        status, body = self.album("product")
        self.assertEqual(status, 200)
        self.assertEqual(body["count"], 3)

    def test_a_selection_narrows_the_slice_and_never_widens_it(self):
        """The property that makes this safe to expose: the ids are ANDed onto the
        membership rule, so a request past the interface cannot pull a frame out of a
        slice it is not in — nor out of a slice that has no album at all."""
        meme = self.add_classified("funny.jpg", "meme")
        status, body = self.album("product", file_ids=[self.chair, meme], apply=True)
        self.assertEqual(status, 200)
        self.assertEqual(body["count"], 1)
        self.assertFalse((Path(body["dest"]) / "funny.jpg").exists())

    def test_a_selection_of_frames_outside_the_slice_gathers_nothing(self):
        meme = self.add_classified("funny.jpg", "meme")
        status, body = self.album("product", file_ids=[meme], apply=True)
        self.assertEqual(status, 200)
        self.assertEqual(body["count"], 0)
        self.assertEqual(body["transferred"], 0)

    def test_a_malformed_selection_is_a_plain_bad_request(self):
        for bad in ("nope", [None], ["1"], [True], {"a": 1}):
            with self.subTest(bad=bad):
                status, body = self.album("product", file_ids=bad)
                self.assertEqual(status, 400)
                self.assertIn("error", body)

    def test_the_selection_survives_the_dry_run_confirm_apply_path(self):
        # The interface previews first and applies second, with the same body: the two
        # answers have to be about the same frames or the confirmation means nothing.
        _s1, preview = self.album("product", file_ids=[self.chair], name="Один")
        _s2, applied = self.album("product", file_ids=[self.chair], name="Один",
                                  apply=True)
        self.assertEqual(preview["count"], applied["count"])
        self.assertEqual(applied["transferred"], 1)


class TestDocumentsAnswerOutLoud(SliceUniformityTestBase):
    """Requirement 5, the one that is easy to get wrong.

    The decision: documents are NOT gathered into a folder, and the refusal comes from the
    route with a reason. A hidden button was never the rule — it forbade nothing, and a
    request sent past the interface would have gathered the folder all the same.
    """

    def setUp(self):
        super().setUp()
        self.passport = self.add_classified("passport.jpg", "document")
        self.start_server()

    def test_the_refusal_comes_from_the_route(self):
        """Asked past the interface, which is the only place a rule can live."""
        status, body = self.album("document")
        self.assertEqual(status, 403)
        self.assertEqual(body["reason"], ui._ALBUM_BLOCKED_DOCUMENTS)
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM move_batches").fetchone()[0], 0)

    def test_a_selection_of_documents_is_refused_by_the_same_route(self):
        status, body = self.album("document", file_ids=[self.passport], apply=True)
        self.assertEqual(status, 403)
        self.assertEqual(body["reason"], ui._ALBUM_BLOCKED_DOCUMENTS)
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM move_batches").fetchone()[0], 0)

    def test_the_bucket_carries_the_reason_instead_of_a_silence(self):
        data = self.junk("?bucket=document")
        self.assertIsNone(data["album_kind"])
        self.assertEqual(data["album_blocked"], ui._ALBUM_BLOCKED_DOCUMENTS)

    def test_emptying_the_sensitive_key_still_does_not_gather_documents(self):
        """`vlm.exclude_classes` decides what is SHOWN (the owner has emptied it for
        previews), never that a folder of passports may be assembled in one click."""
        self.cfg.vlm = dataclasses.replace(self.cfg.vlm, exclude_classes=())
        self.assertEqual(self.junk("?bucket=document")["album_blocked"],
                         ui._ALBUM_BLOCKED_DOCUMENTS)
        status, body = self.album("document")
        self.assertEqual(status, 403)
        self.assertEqual(body["reason"], ui._ALBUM_BLOCKED_DOCUMENTS)

    def test_the_frames_can_still_be_returned_to_the_photos(self):
        # The refusal is about the folder and nothing else: the one action this bucket
        # has always offered is untouched.
        status, body = self.post("/api/overrides",
                                 {"file_ids": [self.passport], "action": "photo"})
        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])


class TestEveryBucketSaysWhyOrOffersAnAlbum(SliceUniformityTestBase):
    """`album_kind` and `album_blocked`: exactly one of the two, for every bucket."""

    def setUp(self):
        super().setUp()
        self.add_classified("chair.jpg", "product")
        self.add_classified("shot.png", "screenshot")
        self.add_classified("funny.jpg", "meme")
        self.add_classified("passport.jpg", "document")
        self.start_server()

    def buckets(self) -> list[str]:
        return [b["verdict"] for b in self.junk()["buckets"]]

    def test_no_bucket_is_silent(self):
        """The enumerating half of requirement 1, on the payload side: the buckets come
        from the collection, not from a list in this test."""
        for bucket in self.buckets():
            with self.subTest(bucket=bucket):
                data = self.junk("?bucket=" + bucket)
                self.assertEqual(
                    (data["album_kind"] is None), (data["album_blocked"] is not None))

    def test_a_gatherable_bucket_names_its_kind_and_blocks_nothing(self):
        data = self.junk("?bucket=product")
        self.assertEqual(data["album_kind"], "product")
        self.assertIsNone(data["album_blocked"])

    def test_a_class_moved_into_the_key_says_which_rule_refused_it(self):
        self.cfg.vlm = dataclasses.replace(self.cfg.vlm, exclude_classes=("product",))
        data = self.junk("?bucket=product")
        self.assertIsNone(data["album_kind"])
        self.assertEqual(data["album_blocked"], ui._ALBUM_BLOCKED_SENSITIVE)
        status, body = self.album("product")
        self.assertEqual(status, 403)
        self.assertEqual(body["reason"], ui._ALBUM_BLOCKED_SENSITIVE)

    def test_the_all_buckets_view_says_why_it_has_no_album(self):
        data = self.junk()
        self.assertIsNone(data["album_kind"])
        self.assertEqual(data["album_blocked"], ui._ALBUM_BLOCKED_ALL_BUCKETS)

    def test_a_class_shipped_without_an_album_kind_is_named_rather_than_dropped(self):
        # The guard for tomorrow: a new verdict must not fall silently out of the
        # interface the way `document` did.
        self.add_classified("weird.jpg", "sticker")
        data = self.junk("?bucket=sticker")
        self.assertIsNone(data["album_kind"])
        self.assertEqual(data["album_blocked"], ui._ALBUM_BLOCKED_NO_KIND)

    def test_the_two_ends_read_one_rule(self):
        # The payload's reason and the route's reason come from the same function, so a
        # button that appears and a request that is refused cannot disagree.
        for kind, expected in (("document", ui._ALBUM_BLOCKED_DOCUMENTS),
                               ("product", None)):
            with self.subTest(kind=kind):
                self.assertEqual(ui.class_album_refusal(self.cfg, kind), expected)


class TestThePinnedSliceIsAnOrdinarySlice(SliceUniformityTestBase):
    """F156's pins are a slice and fall under the same uniformity, not under an
    exception: a pin is a saved query, and the album of one is `kind='query'`."""

    def test_a_pinned_query_gathers_a_named_folder_of_selected_frames(self):
        chair = self.add_classified("chair.jpg", "product")
        self.store_vector(chair, unit(1.0))
        lamp = self.add_classified("lamp.jpg", "product")
        self.store_vector(lamp, unit(1.0, 0.1))
        self.start_server()
        status, body = self.album("query", selector="стулья", name="Стулья",
                                  file_ids=[chair], apply=True)
        self.assertEqual(status, 200)
        self.assertEqual(body["album_name"], "Стулья")
        self.assertEqual(body["count"], 1)
        self.assertTrue((Path(body["dest"]) / "chair.jpg").exists())
        self.assertFalse((Path(body["dest"]) / "lamp.jpg").exists())

    def test_a_pin_is_refused_for_an_empty_selection_like_every_other_slice(self):
        self.start_server()
        status, body = self.album("query", selector="стулья", file_ids=[])
        self.assertEqual(status, 400)
        self.assertEqual(body["reason"], ui._ALBUM_NO_SELECTION)


# --- the browser side: one album row, and every panel of the tab draws it -------------


class TestOneAlbumRowForEverySlicePanel(SliceUniformityTestBase):
    """Requirement 1 again, at the end where the three complaints were made.

    The panels are ENUMERATED out of the served page — every `album-controls` box of the
    "Slices" tab — rather than listed here, so a slice panel added tomorrow is covered by
    this test the day it is added.
    """

    def setUp(self):
        super().setUp()
        self.start_server()
        _status, body, _ctype = self.get("/")
        self.html = body.decode("utf-8")

    def slice_album_boxes(self) -> list[str]:
        section = self.html.split('<section id="tab-slices"', 1)[1].split(
            "</section>", 1)[0]
        # `<slice>-album` is the convention every panel of the tab follows; the other
        # `album-controls` boxes of the tab are rows of a different kind (F189's link to
        # the other answer) and are not albums.
        boxes = re.findall(r'id="([\w-]+-album)" class="album-controls"', section)
        self.assertTrue(boxes, "the Slices tab draws no album row at all")
        return boxes

    def test_every_panel_of_the_tab_has_an_album_row(self):
        # The five panels that ship. Stated as a floor rather than as the list, so adding
        # one does not fail here — the loop below is what every panel has to pass.
        self.assertGreaterEqual(len(self.slice_album_boxes()), 5)

    def test_every_album_row_is_built_by_the_one_shared_function(self):
        for box in self.slice_album_boxes():
            with self.subTest(box=box):
                self.assertIn('box: "' + box + '"', self.html)

    def test_the_shared_row_carries_all_three_affordances(self):
        row = self.html.split("function renderAlbumRow", 1)[1][:3000]
        self.assertIn("albumModeSelect()", row)              # link / copy / move
        self.assertIn("I18N.album_name_placeholder", row)    # name the folder
        self.assertIn("appendAlbumDestControls(box)", row)
        self.assertIn("I18N.album_selected_only", row)       # gather only the ticked
        self.assertIn("album-gather-btn", row)
        self.assertIn("appendAlbumBusyHint(box)", row)

    def test_there_is_exactly_one_album_row_builder(self):
        self.assertEqual(self.html.count("function renderAlbumRow("), 1)
        # The per-slice copies it replaced. Their return would mean a slice went back to
        # answering "what can I do with you" on its own — which is the defect.
        for gone in ("function renderSearchAlbumControls(",
                     "function renderQuerySliceAlbum(",
                     "function renderAnimalsAlbumControls(",
                     "function renderFaceAlbumControls("):
            with self.subTest(gone=gone):
                self.assertNotIn(gone, self.html)

    def test_the_selection_is_one_mechanism_too(self):
        self.assertEqual(self.html.count("function makeSelection("), 1)
        for slice_ in ("searchSelection", "querySelection", "faceSelection",
                       "animalsSelection", "junkSelection"):
            with self.subTest(slice=slice_):
                self.assertIn("var " + slice_ + " = makeSelection(", self.html)

    def test_a_refused_slice_draws_the_reason_where_the_button_was(self):
        row = self.html.split("function renderAlbumRow", 1)[1][:3000]
        self.assertIn("albumBlockedText", row)
        self.assertIn("I18N.album_blocked_", self.html)

    def test_the_junk_panel_still_keeps_gathering_out_of_the_destructive_row(self):
        # F139's requirement 4, unchanged: one movement must not be able to both gather
        # and delete, however much the two rows now share a selection.
        controls = self.html.split('id="junk-restore-btn"', 1)[1].split("</div>", 1)[0]
        self.assertNotIn("album", controls)

    def test_no_external_resources_added(self):
        self.assertNotIn("http://", self.html)
        self.assertNotIn("https://", self.html)


class TestTheNewCaptionsAreInThreeLanguages(unittest.TestCase):
    """Requirement 7: every sentence this feature added goes through the catalog."""

    KEYS = ("album_select_label", "album_selected_only", "album_selection_hint",
            "album_error_empty_selection", "album_blocked_documents",
            "album_blocked_sensitive", "album_blocked_no_kind",
            "album_blocked_all_buckets")

    def test_each_caption_has_all_three(self):
        for key in self.KEYS:
            with self.subTest(key=key):
                entry = ui._UI_STRINGS[key]
                self.assertEqual(set(entry), {"ru", "en", "ja"})
                for lang, text in entry.items():
                    self.assertTrue(text.strip(), f"{key}/{lang} is empty")

    def test_every_refusal_code_the_server_sends_has_a_sentence(self):
        # The enumeration that matters here: a reason added to the server without a
        # sentence would reach the screen as a bare word.
        for code in (ui._ALBUM_BLOCKED_DOCUMENTS, ui._ALBUM_BLOCKED_SENSITIVE,
                     ui._ALBUM_BLOCKED_NO_KIND, ui._ALBUM_BLOCKED_ALL_BUCKETS):
            with self.subTest(code=code):
                self.assertIn("album_blocked_" + code, ui._UI_STRINGS)


if __name__ == "__main__":
    unittest.main()
