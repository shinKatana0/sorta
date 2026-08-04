"""F139: every slice of the web app can be gathered into a folder, not only people.

The engine could always do it (`sorter.plan_album`); the buttons existed only where the
slice happened to arrive first. What this file guards is the wiring and the one thing
the wiring must not get wrong — the privacy rule:

* the album of a class bucket and the album of a quality slice hold exactly what their
  own view SHOWS, counter for counter (they are compared on the same data below);
* a class listed in `vlm.exclude_classes` gets no button AND is refused by the route —
  both ends, because a button the page does not draw is not a rule;
* the marking rows ("back to photos", "to trash") are untouched and stay in their own
  block: one movement must never be able to both gather and delete.
"""
from __future__ import annotations

import dataclasses
import json
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from sorta import ui
from sorta.sorter import CLASS_ALBUM_KINDS, QUALITY_ALBUM_KINDS

from tests.test_ui import UiServerTestBase


class SliceAlbumTestBase(UiServerTestBase):
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
        body = {"kind": kind, "mode": "link", "apply": False}
        body.update(extra)
        return self.post("/api/album", body)

    def junk(self, query: str = "") -> dict:
        _status, body, _ctype = self.get("/api/junk" + query)
        return json.loads(body)

    def review(self, query: str = "") -> dict:
        _status, body, _ctype = self.get("/api/review" + query)
        return json.loads(body)

    def add_classified(self, rel: str, verdict: str) -> int:
        file_id, _path, _content = self.add_photo_file(rel, country="ru", city="Moscow")
        self.conn.execute(
            """INSERT INTO media_class (file_id, verdict, source, updated_at, tier)
               VALUES (?, ?, 'vlm', '2026-07-28', 'vlm')""", (file_id, verdict))
        self.conn.commit()
        return file_id

    def add_quality(self, rel: str, *, sharpness: float | None = 500.0,
                    eyes_open: int | None = None) -> int:
        file_id = self.add_classified(rel, "photo")
        self.conn.execute(
            """INSERT INTO frame_quality (file_id, sharpness, eyes_open,
                   source, updated_at)
               VALUES (?, ?, ?, 'clip', '2026-01-01')""",
            (file_id, sharpness, eyes_open))
        self.conn.commit()
        return file_id


class TestTheRouteGathersEveryNewKind(SliceAlbumTestBase):
    def test_a_class_bucket_previews_without_writing(self):
        self.add_classified("chair.jpg", "product")
        self.start_server()
        status, body = self.album("product")
        self.assertEqual(status, 200)
        self.assertEqual(body["count"], 1)
        self.assertEqual(body["kind"], "product")
        self.assertFalse(body["applied"])
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM move_batches").fetchone()[0], 0)

    def test_a_quality_slice_previews_without_writing(self):
        self.add_quality("blurred.jpg", sharpness=10.0)
        self.start_server()
        status, body = self.album("blurred")
        self.assertEqual(status, 200)
        self.assertEqual(body["count"], 1)
        self.assertEqual(body["kind"], "blurred")

    def test_the_selector_is_optional_for_every_new_kind(self):
        # These slices have nothing to select inside them, so a body without a selector
        # is the ordinary request and not a client that lost its subject.
        self.start_server()
        for kind in CLASS_ALBUM_KINDS + QUALITY_ALBUM_KINDS:
            with self.subTest(kind=kind):
                status, _body = self.album(kind)
                self.assertEqual(status, 200)

    def test_apply_links_the_products_into_their_own_batch(self):
        self.add_classified("chair.jpg", "product")
        self.start_server()
        status, body = self.album("product", apply=True)
        self.assertEqual(status, 200)
        self.assertEqual(body["transferred"], 1)
        self.assertEqual(body["failed"], 0)
        self.assertTrue((Path(body["dest"]) / "chair.jpg").exists())
        batch = self.conn.execute(
            "SELECT mode, operation FROM move_batches ORDER BY id DESC LIMIT 1").fetchone()
        self.assertEqual(batch["mode"], "album_product")
        self.assertEqual(batch["operation"], "link")

    def test_the_folder_can_be_named(self):
        # The acceptance criterion, literally: the "Products" slice becomes a folder
        # called whatever the person typed.
        self.add_classified("chair.jpg", "product")
        self.start_server()
        status, body = self.album("product", name="Товар", mode="copy", apply=True)
        self.assertEqual(status, 200)
        self.assertEqual(body["album_name"], "Товар")
        self.assertEqual(Path(body["dest"]).name, "Товар")
        self.assertTrue((Path(body["dest"]) / "chair.jpg").exists())

    def test_all_three_modes_reach_the_disk(self):
        self.start_server()
        for mode in ("link", "copy", "move"):
            with self.subTest(mode=mode):
                self.add_classified(f"{mode}.jpg", "product")
                status, body = self.album(
                    "product", mode=mode, name=f"album_{mode}", apply=True)
                self.assertEqual(status, 200)
                self.assertTrue((Path(body["dest"]) / f"{mode}.jpg").exists())


class TestTheAlbumMatchesWhatTheViewShows(SliceAlbumTestBase):
    """Requirement 1 of the tests: the counter and the album are the same set. They are
    compared on one fixture, through the two routes a person actually uses."""

    def setUp(self):
        super().setUp()
        # Two products, one of them a duplicate and one unreadable — neither belongs to
        # a bucket's counter nor to its album.
        self.chair = self.add_classified("chair.jpg", "product")
        self.lamp = self.add_classified("lamp.jpg", "product")
        duplicate = self.add_classified("copy.jpg", "product")
        broken = self.add_classified("broken.jpg", "product")
        self.conn.execute("UPDATE files SET dup_of = ? WHERE id = ?",
                          (self.chair, duplicate))
        self.conn.execute("UPDATE files SET error = 'cannot read' WHERE id = ?", (broken,))
        self.conn.commit()
        self.add_classified("meme.jpg", "meme")
        self.blurred = self.add_quality("blurred.jpg", sharpness=10.0)
        self.add_quality("sharp.jpg", sharpness=900.0)
        self.add_quality("eyes.jpg", eyes_open=0)
        self.start_server()

    def test_the_product_album_is_the_size_of_the_product_bucket(self):
        bucket = self.junk("?bucket=product&limit=0")
        _status, body = self.album("product")
        self.assertEqual(body["count"], bucket["total"])
        self.assertEqual(body["count"], 2)

    def test_the_meme_album_is_the_size_of_the_meme_bucket(self):
        bucket = self.junk("?bucket=meme&limit=0")
        _status, body = self.album("meme")
        self.assertEqual(body["count"], bucket["total"])
        self.assertEqual(body["count"], 1)

    def test_each_quality_album_is_the_size_of_its_chip(self):
        counts = {row["slice"]: row["count"] for row in self.review()["counts"]}
        for slice_, kind in (("blurred", "blurred"), ("eyes", "eyes_closed")):
            with self.subTest(slice=slice_):
                _status, body = self.album(kind)
                self.assertEqual(body["count"], counts[slice_])

    def test_the_blurred_album_holds_the_window_and_not_the_tail(self):
        _status, body = self.album("blurred")
        self.assertEqual(body["count"], 1)
        # The same key that bounds the list bounds the album: widen it and both grow.
        self.cfg.features = dataclasses.replace(
            self.cfg.features, blur_review_max=1000.0)
        counts = {row["slice"]: row["count"] for row in self.review()["counts"]}
        _status2, wider = self.album("blurred")
        self.assertGreater(wider["count"], 1)
        self.assertEqual(wider["count"], counts["blurred"])


class TestSensitiveClassesAreRefusedByTheServer(SliceAlbumTestBase):
    """F133's rule, at the end that matters: a request sent past the interface must not
    gather a folder of documents. Both ends are checked — the payload the page draws its
    button from, and the route itself."""

    def test_the_document_bucket_offers_no_album_kind(self):
        self.add_classified("passport.jpg", "document")
        self.start_server()
        self.assertIsNone(self.junk("?bucket=document")["album_kind"])

    def test_the_document_kind_is_refused_by_the_route(self):
        self.add_classified("passport.jpg", "document")
        self.start_server()
        status, body = self.album("document")
        self.assertEqual(status, 400)
        self.assertIn("error", body)
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM move_batches").fetchone()[0], 0)

    def test_a_class_moved_into_the_key_loses_both_ends(self):
        # Only changing the key can tell a config read from a hard-coded "document".
        self.add_classified("chair.jpg", "product")
        self.cfg.vlm = dataclasses.replace(self.cfg.vlm, exclude_classes=("product",))
        self.start_server()
        self.assertIsNone(self.junk("?bucket=product")["album_kind"])
        status, body = self.album("product", apply=True)
        self.assertEqual(status, 403)
        self.assertIn("error", body)
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM move_batches").fetchone()[0], 0)

    def test_emptying_the_key_does_not_hand_out_a_document_album(self):
        """Emptying `vlm.exclude_classes` lifts the preview rule (F133) — it does not
        make documents gatherable. `document` is not an album kind at all: the class is
        passports, medical forms and bank papers, and the config decides what is SHOWN,
        never that a folder of them may be assembled in one click."""
        self.add_classified("passport.jpg", "document")
        self.cfg.vlm = dataclasses.replace(self.cfg.vlm, exclude_classes=())
        self.start_server()
        self.assertIsNone(self.junk("?bucket=document")["album_kind"])
        status, _body = self.album("document")
        self.assertEqual(status, 400)

    def test_a_class_that_left_the_key_gets_its_album_back(self):
        self.add_classified("chair.jpg", "product")
        self.add_classified("meme.jpg", "meme")
        self.cfg.vlm = dataclasses.replace(self.cfg.vlm, exclude_classes=("product",))
        self.start_server()
        self.assertEqual(self.junk("?bucket=meme")["album_kind"], "meme")
        status, _body = self.album("meme")
        self.assertEqual(status, 200)

    def test_the_all_buckets_view_has_no_album_of_its_own(self):
        # "Everything the classifier carried off" is not a slice anybody asked for, and
        # it would mix a sensitive class into the folder besides.
        self.add_classified("passport.jpg", "document")
        self.add_classified("chair.jpg", "product")
        self.start_server()
        self.assertIsNone(self.junk()["album_kind"])


class TestThePayloadsTellTheClientWhatToDraw(SliceAlbumTestBase):
    def test_a_gatherable_bucket_names_its_kind(self):
        self.add_classified("chair.jpg", "product")
        self.start_server()
        self.assertEqual(self.junk("?bucket=product")["album_kind"], "product")

    def test_each_flat_review_slice_names_its_kind(self):
        self.start_server()
        for slice_, kind in (("blurred", "blurred"), ("eyes", "eyes_closed")):
            with self.subTest(slice=slice_):
                self.assertEqual(self.review("?slice=" + slice_)["album_kind"], kind)

    def test_the_duplicates_slice_names_none(self):
        # Duplicates are the one path in the program that deletes files and the one
        # slice where a keeper is chosen; a folder is not what they are for.
        self.start_server()
        self.assertIsNone(self.review("?slice=dupes")["album_kind"])


class TestSliceAlbumMarkup(SliceAlbumTestBase):
    def setUp(self):
        super().setUp()
        self.start_server()
        _status, body, _ctype = self.get("/")
        self.html = body.decode("utf-8")

    def test_both_panels_hold_an_album_row(self):
        self.assertIn('id="junk-album" class="album-controls"', self.html)
        self.assertIn('id="review-album" class="album-controls"', self.html)

    def test_the_rows_are_drawn_from_the_kind_the_server_sent(self):
        self.assertIn('renderSliceAlbumControls("junk-album", data.album_kind)',
                      self.html)
        self.assertIn('renderSliceAlbumControls("review-album", data.album_kind)',
                      self.html)

    def test_the_gather_row_carries_the_same_controls_as_the_others(self):
        row = self.html.split("function renderSliceAlbumControls", 1)[1][:1600]
        self.assertIn("albumModeSelect()", row)          # link / copy / move
        self.assertIn("appendAlbumDestControls(box)", row)
        self.assertIn("I18N.album_name_placeholder", row)  # name the folder
        self.assertIn("album-gather-btn", row)
        self.assertIn("appendAlbumBusyHint(box)", row)
        self.assertIn("gatherAlbum(kind", row)

    def test_gathering_stays_out_of_the_destructive_rows(self):
        # Requirement 4: "to trash" and "back to photos" keep their own block, and the
        # gather row is a separate element — one movement cannot do both.
        junk_controls = self.html.split('id="junk-restore-btn"', 1)[1].split("</div>", 1)[0]
        self.assertNotIn("album", junk_controls)
        review_controls = self.html.split(
            'id="review-delete-btn"', 1)[1].split("</div>", 1)[0]
        self.assertNotIn("album", review_controls)

    def test_no_external_resources_added(self):
        self.assertNotIn("http://", self.html)
        self.assertNotIn("https://", self.html)
        self.assertNotIn("<link", self.html)


class TestTheFolderNamesAreTranslated(unittest.TestCase):
    """The button label and the default folder name are product text, so they go through
    the catalogs in all three languages — the folder keys through `i18n.folder`, the
    button through the served app's strings."""

    def test_every_new_folder_key_has_three_distinct_names(self):
        from sorta.i18n import FOLDER_KEYS, folder
        for key in ("screenshots", "memes", "blurred", "eyes_closed"):
            with self.subTest(key=key):
                self.assertIn(key, FOLDER_KEYS)
                names = {lang: folder(key, lang) for lang in ("ru", "en", "ja")}
                self.assertEqual(len(set(names.values())), 3, names)
                for lang, value in names.items():
                    self.assertTrue(value.strip(), f"{key}/{lang} is empty")

    def test_the_gather_button_is_translated(self):
        for key in ("album_button", "album_name_placeholder", "album_mode_link",
                    "album_mode_copy", "album_mode_move"):
            with self.subTest(key=key):
                entry = ui._UI_STRINGS[key]
                self.assertEqual(set(entry), {"ru", "en", "ja"})


if __name__ == "__main__":
    unittest.main()
