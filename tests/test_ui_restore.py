"""F149: "try to improve" — the third action of the Review tab, on ONE frame.

`tests/test_restore.py` covers the module (the original is untouched, the copy is named
and never overwrites, a reason instead of an empty result, the input is bounded). This
file is about the part a person actually meets: the route that refuses anything but a
single id, the copy arriving as a second card beside the original and marked as
processed, and the two things the feature must NOT do —

* keeping the copy does not mark the original for deletion. Two decisions, and the second
  one is the person's (the F148 line between advice and action), so the case reads
  `dedup_choice` rather than the markup;
* the restored frame does not come back as a new duplicate pair to sort out.

The real weights are never loaded: `restore.load_swin2sr` is patched with a stub of the
same shape, which is also what makes "loaded on the first press, not at server start"
checkable at all.

F169 adds the third thing this route must not do: decide for everybody in silence. It
called the engine with no ceiling at all, so a constant in `restore.py` chose what every
person got back — including for a 12 Mpx frame, which came back the size it went in and
rebuilt out of a 1024 px copy of itself. The cases below check that the ceiling is the
config's and that such a frame comes back with an answer that SAYS so.
"""
from __future__ import annotations

import dataclasses
import json
import unittest
from unittest import mock

from PIL import Image

from sorta import restore, ui

from tests.test_ui_review import ReviewTestBase


def doubling_upscaler(_model_name: str) -> restore.UpscaleFn:
    """A stand-in for Swin2SR — an image in, a bigger image out."""
    def upscale(image: Image.Image) -> Image.Image:
        return image.resize((image.width * 2, image.height * 2))
    return upscale


class RestoreUiTestBase(ReviewTestBase):
    def setUp(self):
        super().setUp()
        restore.reset_upscalers()
        self.addCleanup(restore.reset_upscalers)
        ui._dupes_cache_clear()
        self.addCleanup(ui._dupes_cache_clear)
        self.loads: list[str] = []

    def patch_model(self, loader=doubling_upscaler):
        def counting(name: str) -> restore.UpscaleFn:
            self.loads.append(name)
            return loader(name)
        patcher = mock.patch.object(restore, "load_swin2sr", counting)
        patcher.start()
        self.addCleanup(patcher.stop)

    def restore_frame(self, payload: object) -> tuple[int, dict]:
        return self.post("/api/review/restore", payload)

    def choices(self) -> dict[int, str]:
        return {r["file_id"]: r["action"] for r in
                self.conn.execute("SELECT file_id, action FROM dedup_choice").fetchall()}

    def files(self) -> dict[int, str]:
        return {r["id"]: r["path"] for r in
                self.conn.execute("SELECT id, path FROM files").fetchall()}

    def html(self) -> str:
        _status, body, _ctype = self.get("/")
        return body.decode("utf-8")


class TestOneFrameOnly(RestoreUiTestBase):
    """No bulk route exists — not from the interface, not by hand."""

    def test_a_list_of_ids_is_refused(self):
        file_id = self.add_reviewable("blur.jpg", sharpness=10.0)
        self.patch_model()
        self.start_server()

        for body in ({"file_ids": [file_id, file_id + 1]},
                     {"file_ids": [file_id]},
                     {"file_id": [file_id]},
                     {"file_id": None},
                     []):
            with self.subTest(body=body):
                status, payload = self.restore_frame(body)
                self.assertEqual(status, 400, payload)
        self.assertEqual(self.loads, [])
        self.assertIsNone(self.conn.execute("SELECT 1 FROM restored_files").fetchone())

    def test_an_unknown_id_is_a_404_and_not_a_load(self):
        self.patch_model()
        self.start_server()
        status, payload = self.restore_frame({"file_id": 99999})
        self.assertEqual(status, 404, payload)
        self.assertEqual(self.loads, [])


class TestTheCopyArrivesBesideTheOriginal(RestoreUiTestBase):
    def test_the_answer_is_a_card_marked_as_processed(self):
        source = self.add_reviewable("blur.jpg", sharpness=10.0)
        source_path = self.files()[source]
        self.patch_model()
        self.start_server()

        status, payload = self.restore_frame({"file_id": source})

        self.assertEqual(status, 200, payload)
        self.assertTrue(payload["ok"])
        self.assertFalse(payload["reused"])
        item = payload["item"]
        self.assertTrue(item["restored"])
        self.assertEqual(item["source_file_id"], source)
        self.assertEqual(item["name"], "blur_restored.jpg")
        self.assertIsNone(item["action"])   # a new frame arrives with no decision on it
        self.assertEqual(item["thumb_url"], f"/thumb/{item['file_id']}")
        # The same shape as any other review card — that is what "the same actions" means.
        self.assertEqual(set(item) - {"restored", "source_file_id"},
                         {"file_id", "name", "date", "src_dir", "src_path", "sharpness",
                          "action", "thumb_url", "video"})
        # ...and the original is still exactly where and what it was.
        self.assertEqual(self.files()[source], source_path)

    def test_the_copy_is_an_indexed_file_carrying_the_date_of_the_original(self):
        """Not scanned but derived: a re-encoded JPEG has no EXIF, so metadata read off
        the copy would date it today and file it under this year."""
        source = self.add_reviewable("blur.jpg", sharpness=10.0)
        self.patch_model()
        self.start_server()

        _status, payload = self.restore_frame({"file_id": source})
        copy_id = payload["item"]["file_id"]

        copy = self.conn.execute("SELECT * FROM files WHERE id = ?", (copy_id,)).fetchone()
        original = self.conn.execute("SELECT * FROM files WHERE id = ?",
                                     (source,)).fetchone()
        self.assertEqual(copy["taken_at"], original["taken_at"])
        self.assertEqual(copy["media_type"], "photo")
        self.assertIsNone(copy["dup_of"])
        self.assertTrue(copy["path"].endswith("blur_restored.jpg"))
        link = self.conn.execute("SELECT * FROM restored_files").fetchone()
        self.assertEqual((link["file_id"], link["source_file_id"]), (copy_id, source))
        self.assertEqual(link["model"], self.cfg.features.restore_model)

    def test_pressing_twice_returns_the_copy_that_exists(self):
        source = self.add_reviewable("blur.jpg", sharpness=10.0)
        self.patch_model()
        self.start_server()

        _status, first = self.restore_frame({"file_id": source})
        _status, second = self.restore_frame({"file_id": source})

        self.assertFalse(first["reused"])
        self.assertTrue(second["reused"])
        self.assertEqual(second["item"]["file_id"], first["item"]["file_id"])
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM restored_files").fetchone()[0], 1)
        self.assertEqual(
            len([p for p in self.src_dir.iterdir() if "_restored" in p.name]), 1)

    def test_a_copy_deleted_from_disk_is_made_again(self):
        """Answering "you already have one" about a file that is gone is worse than
        doing the work a second time."""
        source = self.add_reviewable("blur.jpg", sharpness=10.0)
        self.patch_model()
        self.start_server()

        self.restore_frame({"file_id": source})
        (self.src_dir / "blur_restored.jpg").unlink()
        _status, second = self.restore_frame({"file_id": source})

        self.assertTrue(second["ok"])
        self.assertFalse(second["reused"])   # the work was done again, not reported as old
        self.assertTrue((self.src_dir / "blur_restored.jpg").exists())
        # The stale row went with the file it named: one copy, and it is the new one.
        rows = self.conn.execute("SELECT file_id FROM restored_files").fetchall()
        self.assertEqual([r["file_id"] for r in rows], [second["item"]["file_id"]])


class TestTheModelIsNotLoadedAtStartup(RestoreUiTestBase):
    def test_starting_the_server_loads_nothing(self):
        self.add_reviewable("blur.jpg", sharpness=10.0)
        self.patch_model()
        self.start_server()
        # Everything the tab does before the button is pressed.
        self.get("/")
        self.get("/api/review?slice=blurred")
        self.assertEqual(self.loads, [])
        self.assertEqual(restore.loaded_models(), ())

    def test_the_first_press_loads_it_and_the_second_does_not(self):
        first_frame = self.add_reviewable("a.jpg", sharpness=10.0)
        second_frame = self.add_reviewable("b.jpg", sharpness=11.0)
        self.patch_model()
        self.start_server()

        self.restore_frame({"file_id": first_frame})
        self.assertEqual(self.loads, [self.cfg.features.restore_model])
        self.restore_frame({"file_id": second_frame})
        self.assertEqual(self.loads, [self.cfg.features.restore_model])


class TestAReasonInsteadOfAnEmptyResult(RestoreUiTestBase):
    def test_a_model_that_will_not_load_is_named_and_writes_nothing(self):
        source = self.add_reviewable("blur.jpg", sharpness=10.0)

        def broken(_name: str) -> restore.UpscaleFn:
            raise ImportError("transformers is not installed")

        self.patch_model(broken)
        self.start_server()

        status, payload = self.restore_frame({"file_id": source})

        self.assertEqual(status, 200, payload)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["reason"], restore.ERROR_MODEL_UNAVAILABLE)
        self.assertIn("transformers", payload["detail"])
        self.assertIsNone(self.conn.execute("SELECT 1 FROM restored_files").fetchone())
        self.assertEqual([p.name for p in self.src_dir.iterdir()], ["blur.jpg"])

    def test_a_frame_that_does_not_decode_is_the_same_story(self):
        source = self.add_reviewable("blur.jpg", sharpness=10.0)
        (self.src_dir / "blur.jpg").write_bytes(b"not an image any more")
        self.patch_model()
        self.start_server()

        status, payload = self.restore_frame({"file_id": source})

        self.assertEqual(status, 200, payload)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["reason"], restore.ERROR_DECODE_FAILED)
        self.assertEqual(self.loads, [])   # no 400 MB load for a file that will not read
        self.assertIsNone(self.conn.execute("SELECT 1 FROM restored_files").fetchone())

    def test_every_reason_has_a_string_in_all_three_languages(self):
        for code in (restore.ERROR_MODEL_UNAVAILABLE, restore.ERROR_DECODE_FAILED,
                     restore.ERROR_WRITE_FAILED):
            with self.subTest(reason=code):
                entry = ui._UI_STRINGS[f"review_restore_error_{code}"]
                self.assertEqual(set(entry), {"ru", "en", "ja"})
                for lang in ("ru", "en", "ja"):
                    self.assertTrue(entry[lang].strip())


class TestKeepingTheCopyIsNotADecisionAboutTheOriginal(RestoreUiTestBase):
    """Requirement 6, and the same line F148 drew: choosing one frame is advice about it
    and nothing at all about the other."""

    def test_restoring_marks_nothing(self):
        source = self.add_reviewable("blur.jpg", sharpness=10.0)
        self.patch_model()
        self.start_server()

        self.restore_frame({"file_id": source})

        self.assertEqual(self.choices(), {})

    def test_keeping_the_copy_leaves_the_original_undecided(self):
        source = self.add_reviewable("blur.jpg", sharpness=10.0)
        self.patch_model()
        self.start_server()
        _status, payload = self.restore_frame({"file_id": source})
        copy_id = payload["item"]["file_id"]

        status, marked = self.post("/api/review/mark",
                                   {"file_ids": [copy_id], "action": "keep"})

        self.assertEqual(status, 200, marked)
        self.assertEqual(self.choices(), {copy_id: "keep"})

    def test_both_can_be_kept_and_so_can_neither(self):
        source = self.add_reviewable("blur.jpg", sharpness=10.0)
        self.patch_model()
        self.start_server()
        _status, payload = self.restore_frame({"file_id": source})
        copy_id = payload["item"]["file_id"]

        self.post("/api/review/mark", {"file_ids": [source, copy_id], "action": "keep"})
        self.assertEqual(self.choices(), {source: "keep", copy_id: "keep"})

        self.post("/api/review/mark", {"file_ids": [source, copy_id], "action": "clear"})
        self.assertEqual(self.choices(), {})


class TestTheCopyIsNoNewDuplicateTask(RestoreUiTestBase):
    """The pair a person created themselves must not come back as work to do."""

    def test_the_duplicates_list_does_not_grow(self):
        source = self.add_reviewable("blur.jpg", sharpness=10.0)
        self.patch_model()
        self.start_server()
        _status, payload = self.restore_frame({"file_id": source})
        copy_id = payload["item"]["file_id"]

        # What the next `phash` run would produce: the copy IS a near-duplicate of its
        # source, so give both the same hash and ask the tab.
        self.conn.execute("UPDATE files SET phash = ? WHERE id IN (?, ?)",
                          ("0" * 16, source, copy_id))
        self.conn.commit()
        ui._dupes_cache_clear()

        status, body, _ctype = self.get("/api/dupes")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), [])

        review = self.review("?slice=dupes")
        self.assertEqual([c["count"] for c in review["counts"] if c["slice"] == "dupes"],
                         [0])
        self.assertEqual(self.choices(), {})

    def test_trashing_the_copy_forgets_it(self):
        """...and the button may then be pressed again — the index must not claim a copy
        that is no longer anywhere."""
        source = self.add_reviewable("blur.jpg", sharpness=10.0)
        self.patch_model()
        self.start_server()
        _status, payload = self.restore_frame({"file_id": source})
        copy_id = payload["item"]["file_id"]

        status, trashed = self.post("/api/photo/trash", {"file_id": copy_id})

        self.assertEqual(status, 200, trashed)
        self.assertIsNone(self.conn.execute("SELECT 1 FROM restored_files").fetchone())
        _status, again = self.restore_frame({"file_id": source})
        self.assertTrue(again["ok"])
        self.assertFalse(again["reused"])


class TestTheCeilingIsPassedAndSaidOutLoud(RestoreUiTestBase):
    """F169. The route used to call the engine WITHOUT a ceiling, so one number in the
    code decided for every frame and nobody was told: a 12 Mpx shot came back the size it
    went in, rebuilt out of a 1024 px copy of itself. Two things are checked here — the
    ceiling is the config's, and a frame above it is told so in the answer."""

    def set_ceiling(self, max_edge: int) -> None:
        self.cfg = dataclasses.replace(
            self.cfg,
            features=dataclasses.replace(self.cfg.features, restore_max_edge=max_edge))

    def big_frame(self, rel: str = "big.jpg", size=(2400, 1800)) -> int:
        """A reviewable frame whose FILE is a full-sized one — the population the button
        was never measured on."""
        file_id = self.add_reviewable(rel, sharpness=10.0)
        Image.new("RGB", size, (90, 120, 160)).save(self.src_dir / rel, "JPEG")
        return file_id

    def watching(self, seen: list) -> None:
        def loader(_name: str) -> restore.UpscaleFn:
            def upscale(image: Image.Image) -> Image.Image:
                seen.append(image.size)
                return image
            return upscale
        self.patch_model(loader)

    def test_the_engine_is_given_the_ceiling_from_the_config(self):
        self.set_ceiling(600)
        file_id = self.big_frame()
        seen: list = []
        self.watching(seen)
        self.start_server()

        status, payload = self.restore_frame({"file_id": file_id})

        self.assertEqual(status, 200, payload)
        self.assertEqual(seen, [(600, 450)])

    def test_a_frame_above_the_ceiling_says_the_copy_was_rebuilt(self):
        self.set_ceiling(600)
        file_id = self.big_frame()
        self.patch_model()
        self.start_server()

        _status, payload = self.restore_frame({"file_id": file_id})

        self.assertTrue(payload["ok"], payload)
        self.assertTrue(payload["rebuilt"])
        self.assertEqual(payload["source_edge"], 2400)
        self.assertEqual(payload["max_edge"], 600)

    def test_a_frame_below_the_ceiling_claims_nothing_of_the_sort(self):
        """The ordinary case of this action — a small frame, a clean gain, no warning."""
        file_id = self.add_reviewable("small.jpg", sharpness=10.0)
        self.patch_model()
        self.start_server()

        _status, payload = self.restore_frame({"file_id": file_id})

        self.assertFalse(payload["rebuilt"])
        self.assertEqual(payload["source_edge"], 64)
        self.assertEqual(payload["max_edge"], self.cfg.features.restore_max_edge)

    def test_the_second_press_repeats_the_warning_with_the_reused_copy(self):
        """The frame and the ceiling have not changed, so neither has what is owed."""
        self.set_ceiling(600)
        file_id = self.big_frame()
        self.patch_model()
        self.start_server()

        self.restore_frame({"file_id": file_id})
        _status, second = self.restore_frame({"file_id": file_id})

        self.assertTrue(second["reused"])
        self.assertTrue(second["rebuilt"])
        self.assertEqual(second["source_edge"], 2400)

    def test_the_original_is_untouched_even_when_the_copy_is_rebuilt(self):
        self.set_ceiling(600)
        file_id = self.big_frame()
        before = (self.src_dir / "big.jpg").read_bytes()
        self.patch_model()
        self.start_server()

        _status, payload = self.restore_frame({"file_id": file_id})

        self.assertTrue(payload["rebuilt"])
        self.assertEqual((self.src_dir / "big.jpg").read_bytes(), before)

    def test_the_warning_reaches_the_screen_in_three_languages(self):
        self.start_server()
        html = self.html()
        self.assertIn("if (resp.rebuilt)", html)
        self.assertIn("fmt(I18N.review_restore_rebuilt", html)
        entry = ui._UI_STRINGS["review_restore_rebuilt"]
        self.assertEqual(set(entry), {"ru", "en", "ja"})
        for lang in ("ru", "en", "ja"):
            with self.subTest(lang=lang):
                # Both numbers are named: "too big" without saying too big for what is a
                # warning a person cannot act on, and the key that moves the limit is the
                # action they can take.
                self.assertIn("{max_edge}", entry[lang])
                self.assertIn("{source_edge}", entry[lang])
                self.assertIn("features.restore_max_edge", entry[lang])


class TestTheButtonAndTheBadge(RestoreUiTestBase):
    """The client half: a third action next to the two, alive for exactly one frame, and
    a card inserted where the original is rather than a "file saved" message."""

    def test_the_button_is_the_third_action_of_the_row(self):
        html = self.html_of_started_server()
        for marker in ('id="review-delete-btn"', 'id="review-keep-btn"',
                       'id="review-restore-btn"'):
            self.assertIn(marker, html)
        order = [html.index(f'id="review-{name}-btn"')
                 for name in ("delete", "keep", "restore")]
        self.assertEqual(order, sorted(order))

    def test_it_is_alive_for_exactly_one_selected_frame(self):
        html = self.html_of_started_server()
        self.assertIn("restoreBtn.disabled = uiBusy() || reviewRestoring || n !== 1;", html)

    def test_the_copy_is_inserted_next_to_its_original(self):
        html = self.html_of_started_server()
        self.assertIn('grid.querySelector(\'[data-file-id="\' + item.source_file_id', html)
        self.assertIn("grid.insertBefore(card, source.nextSibling)", html)

    def test_the_card_says_processed_and_carries_the_same_actions(self):
        html = self.html_of_started_server()
        self.assertIn('(item.restored ? " processed" : "")', html)
        self.assertIn("badge.textContent = I18N.review_restore_badge;", html)
        self.assertIn("badge.title = I18N.review_restore_badge_hint;", html)
        # The checkbox is built by the same `renderReviewCard`, so the copy affords
        # exactly what its original does — there is no second card renderer to drift.
        self.assertEqual(html.count("function renderReviewCard(item) {"), 1)

    def test_every_new_string_is_translated_three_ways(self):
        for key in ("review_restore", "review_restore_hint", "review_restore_running",
                    "review_restore_badge", "review_restore_badge_hint",
                    "review_restore_done", "review_restore_reused"):
            with self.subTest(key=key):
                entry = ui._UI_STRINGS[key]
                self.assertEqual(set(entry), {"ru", "en", "ja"})
                for lang in ("ru", "en", "ja"):
                    self.assertTrue(entry[lang].strip())

    def test_the_wording_never_calls_the_copy_an_improved_photograph(self):
        """The one thing the interface must not say. The button offers to TRY, and what
        comes back is described as processed by a model — never as a better photograph."""
        badge = ui._UI_STRINGS["review_restore_badge"]
        self.assertEqual(badge["ru"], "обработано моделью")
        self.assertEqual(badge["en"], "processed by a model")
        for lang in ("ru", "en", "ja"):
            with self.subTest(lang=lang):
                self.assertIn("model" if lang == "en" else
                              ("модел" if lang == "ru" else "モデル"),
                              ui._UI_STRINGS["review_restore_badge_hint"][lang])

    def html_of_started_server(self) -> str:
        self.start_server()
        return self.html()


if __name__ == "__main__":
    unittest.main()
