"""F168: "try to improve" reachable from the EXPANDED FRAME, in every slice.

F149 shipped the action behind one door — the "blurred" slice — and the measurement of
2026-08-03 says that door is nearly shut: the sharpness filter at its threshold holds 8%
of the frames a person calls soft. The second measurement (F169, 80 blind pairs) says
where the action really belongs, and it is not blur but SIZE — 66% under 640 px, a coin
toss by 1280. So this file is about the second entrance and its two edges:

* the frame a person has expanded, in ANY slice, is offered the action when it is small
  enough — and is told why it is not when it is not. A withdrawn offer with nothing said
  is the silent half of a promise the measurement does not support;
* the two things the action must never do are refused by the ROUTE and not by the absence
  of a button (the F133 rule: a hidden control is not a rule). A personal document must
  not be decoded and drawn four times larger even when the request arrives past the
  interface, and a clip has no frame for an image model to answer with.

`tests/test_ui_restore.py` keeps the first entrance and the engine's own contract (the
original untouched, a reason instead of an empty result, the ceiling said out loud); this
file never repeats those — it only checks that reaching the same route from a different
place changes none of them.
"""
from __future__ import annotations

import dataclasses
import json
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from sorta import restore, ui

from tests.test_ui_restore import RestoreUiTestBase


class RestoreAnywhereBase(RestoreUiTestBase):
    def offer(self, file_id: int) -> tuple[int, dict]:
        status, body, ctype = self.get(f"/api/restore/offer?file_id={file_id}")
        if status == 200:
            self.assertIn("application/json", ctype)
        return status, json.loads(body)

    def plain_photo(self, rel: str) -> int:
        """A photograph and nothing else: no `media_class`, no `frame_quality`, so it is
        in NO review slice at all — the population the first entrance could not reach."""
        file_id, _path, _content = self.add_photo_file(rel)
        return file_id

    def big_photo(self, rel: str, size: tuple[int, int] = (2400, 1800)) -> int:
        file_id = self.plain_photo(rel)
        Image.new("RGB", size, (90, 120, 160)).save(self.src_dir / rel, "JPEG")
        return file_id

    def exclude_classes(self, *classes: str) -> None:
        self.cfg = dataclasses.replace(
            self.cfg, vlm=dataclasses.replace(self.cfg.vlm, exclude_classes=classes))

    def restored_names(self) -> list[str]:
        return sorted(p.name for p in self.src_dir.iterdir() if "_restored" in p.name)

    def html(self) -> str:
        _status, body, _ctype = self.get("/")
        return body.decode("utf-8")


class TestTheExpandedFrameIsTheSecondEntrance(RestoreAnywhereBase):
    """Requirement 1: the action is reachable from the frame, not from a slice."""

    def test_a_frame_in_no_review_slice_is_offered_and_processed(self):
        file_id = self.plain_photo("holiday.jpg")
        self.patch_model()
        self.start_server()

        self.assertEqual(self.review("?slice=blurred")["items"], [])
        status, offer = self.offer(file_id)

        self.assertEqual(status, 200, offer)
        self.assertTrue(offer["available"], offer)
        self.assertIsNone(offer["reason"])
        self.assertFalse(offer["rebuilt"])

        status, payload = self.restore_frame({"file_id": file_id})
        self.assertEqual(status, 200, payload)
        self.assertTrue(payload["ok"], payload)
        self.assertEqual(self.restored_names(), ["holiday_restored.jpg"])

    def test_the_slices_of_people_animals_events_and_search_all_reach_it(self):
        """The route reads the FILE and no slice table, so belonging to one changes
        nothing. Stated by putting one frame in each population and asking."""
        people = self.plain_photo("people.jpg")
        self.add_real_face(people)
        animals = self.plain_photo("cat.jpg")
        self.conn.execute(
            "INSERT INTO manual_pet (file_id, is_animal, updated_at) VALUES (?, 1, 'now')",
            (animals,))
        trip = self.plain_photo("trip.jpg")
        event = self.conn.execute(
            """INSERT INTO events (started_at, ended_at, name)
               VALUES ('2022-05-01', '2022-05-02', 'Trip')""").lastrowid
        self.conn.execute("INSERT INTO event_files (event_id, file_id) VALUES (?, ?)",
                          (event, trip))
        self.conn.commit()
        searched = self.plain_photo("mountain.jpg")   # search ranks ordinary photographs
        self.patch_model()
        self.start_server()

        for name, file_id in (("people", people), ("animals", animals),
                              ("events", trip), ("search", searched)):
            with self.subTest(slice=name):
                _status, offer = self.offer(file_id)
                self.assertTrue(offer["available"], offer)
                _status, payload = self.restore_frame({"file_id": file_id})
                self.assertTrue(payload["ok"], payload)

        self.assertEqual(self.restored_names(),
                         ["cat_restored.jpg", "mountain_restored.jpg",
                          "people_restored.jpg", "trip_restored.jpg"])

    def test_an_unknown_id_is_a_404_and_a_bad_one_a_400(self):
        self.start_server()
        status, payload = self.offer(99999)
        self.assertEqual(status, 404, payload)
        for raw in ("", "abc", "-1", "1.5"):
            with self.subTest(file_id=raw):
                status, _body, _ctype = self.get(f"/api/restore/offer?file_id={raw}")
                self.assertEqual(status, 400)

    def test_asking_loads_no_model_and_writes_nothing(self):
        """Opening a photograph must cost what it always cost: the answer is a row of the
        index plus the header of the file."""
        file_id = self.plain_photo("holiday.jpg")
        self.patch_model()
        self.start_server()

        self.offer(file_id)

        self.assertEqual(self.loads, [])
        self.assertEqual(restore.loaded_models(), ())
        self.assertEqual(self.restored_names(), [])
        self.assertIsNone(self.conn.execute("SELECT 1 FROM restored_files").fetchone())


class TestPrivateClassesAreRefusedByTheRoute(RestoreAnywhereBase):
    """The main case. Restoring a document means decoding a passport or a medical form
    and drawing it four times larger — the one thing the product deliberately never
    renders — and the refusal has to survive a request that never met the interface."""

    def test_a_document_is_refused_even_though_the_request_skipped_the_page(self):
        file_id = self.add_reviewable("passport.jpg", verdict="document")
        self.patch_model()
        self.start_server()

        status, payload = self.restore_frame({"file_id": file_id})

        self.assertEqual(status, 200, payload)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["reason"], ui.RESTORE_ERROR_SENSITIVE)
        self.assertEqual(self.loads, [])          # the frame was never even decoded
        self.assertEqual(self.restored_names(), [])
        self.assertIsNone(self.conn.execute("SELECT 1 FROM restored_files").fetchone())

    def test_the_offer_says_so_too_so_the_button_is_never_drawn(self):
        file_id = self.add_reviewable("passport.jpg", verdict="document")
        self.start_server()

        _status, offer = self.offer(file_id)

        self.assertFalse(offer["available"])
        self.assertEqual(offer["reason"], ui.RESTORE_ERROR_SENSITIVE)

    def test_the_file_of_a_refused_frame_is_never_opened(self):
        """The size is read off the header, and a document is a file this program opens
        for no purpose at all — including for a sentence it will not print."""
        file_id = self.add_reviewable("passport.jpg", verdict="document")
        self.start_server()

        def refuse(src: Path) -> int:
            raise AssertionError(f"the document {src} was opened")

        with mock.patch.object(restore, "source_edge", refuse):
            _status, offer = self.offer(file_id)

        self.assertEqual(offer["source_edge"], 0)
        self.assertFalse(offer["rebuilt"])

    def test_the_list_is_the_config_key_and_not_a_constant(self):
        """`vlm.exclude_classes` is the one visible list of private classes (F133), read
        live: a class added to it is protected here without a restart."""
        self.exclude_classes("document", "screenshot")
        screenshot = self.add_reviewable("scan.jpg", verdict="screenshot")
        product = self.add_reviewable("shoe.jpg", verdict="product")
        self.patch_model()
        self.start_server()

        _status, refused = self.restore_frame({"file_id": screenshot})
        self.assertEqual(refused["reason"], ui.RESTORE_ERROR_SENSITIVE)
        _status, allowed = self.restore_frame({"file_id": product})
        self.assertTrue(allowed["ok"], allowed)

    def test_a_frame_nobody_classified_is_an_ordinary_photograph(self):
        """`media_class` is written by a run that may not have happened, and an absent
        verdict must not read as a private one — nothing would be restorable at all."""
        file_id = self.plain_photo("holiday.jpg")
        self.start_server()

        _status, offer = self.offer(file_id)

        self.assertTrue(offer["available"], offer)


class TestVideoIsRefused(RestoreAnywhereBase):
    """The engine is about images: a clip has no single frame to be the answer."""

    def test_a_clip_is_refused_by_the_route(self):
        file_id = self.plain_photo("clip.mp4")
        self.patch_model()
        self.start_server()

        status, payload = self.restore_frame({"file_id": file_id})

        self.assertEqual(status, 200, payload)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["reason"], ui.RESTORE_ERROR_VIDEO)
        self.assertEqual(self.loads, [])
        self.assertIsNone(self.conn.execute("SELECT 1 FROM restored_files").fetchone())

    def test_the_offer_refuses_it_as_well(self):
        file_id = self.plain_photo("clip.mp4")
        self.start_server()

        _status, offer = self.offer(file_id)

        self.assertFalse(offer["available"])
        self.assertEqual(offer["reason"], ui.RESTORE_ERROR_VIDEO)

    def test_the_index_is_believed_when_the_extension_does_not_say(self):
        """Both signals are consulted: a row the indexer called a video is one whatever
        its name looks like."""
        self.assertEqual(
            ui._restore_refusal(Path("holiday.jpg"), "photo", "video", frozenset()),
            ui.RESTORE_ERROR_VIDEO)
        self.assertIsNone(
            ui._restore_refusal(Path("holiday.jpg"), "photo", "photo", frozenset()))


class TestTheCeilingDecidesWhereItIsOffered(RestoreAnywhereBase):
    """F169's verdict, applied: the gain belongs to small frames, so the offer does too —
    and above the ceiling the reason is said instead of being left out."""

    def test_a_small_frame_is_offered_without_a_warning(self):
        file_id = self.plain_photo("small.jpg")
        self.start_server()

        _status, offer = self.offer(file_id)

        self.assertTrue(offer["available"])
        self.assertFalse(offer["rebuilt"])
        self.assertEqual(offer["max_edge"], self.cfg.features.restore_max_edge)
        self.assertLess(offer["source_edge"], offer["max_edge"])

    def test_a_frame_above_the_ceiling_carries_both_numbers(self):
        self.cfg = dataclasses.replace(
            self.cfg,
            features=dataclasses.replace(self.cfg.features, restore_max_edge=600))
        file_id = self.big_photo("big.jpg")
        self.start_server()

        _status, offer = self.offer(file_id)

        # F198: a refusal now, and the same one the route answers a press with — the two
        # numbers are what the sentence saying why is built from.
        self.assertFalse(offer["available"])
        self.assertEqual(offer["reason"], ui.RESTORE_ERROR_TOO_LARGE)
        self.assertTrue(offer["rebuilt"])
        self.assertEqual(offer["source_edge"], 2400)
        self.assertEqual(offer["max_edge"], 600)

    def test_the_page_withdraws_the_button_and_keeps_the_sentence(self):
        self.start_server()
        html = self.html()
        self.assertIn("var offered = offer.available;", html)
        self.assertIn("lightboxRestoreBtn.hidden = !offered;", html)
        self.assertIn("fmt(I18N.review_restore_error_too_large,", html)

    def test_the_sentence_names_both_numbers_and_the_key_in_three_languages(self):
        entry = ui._UI_STRINGS["review_restore_error_too_large"]
        self.assertEqual(set(entry), {"ru", "en", "ja"})
        for lang in ("ru", "en", "ja"):
            with self.subTest(lang=lang):
                self.assertIn("{max_edge}", entry[lang])
                self.assertIn("{source_edge}", entry[lang])
                self.assertIn("features.restore_max_edge", entry[lang])


class TestTheCopySaysWhereItCameFrom(RestoreAnywhereBase):
    """The copy is a canonical file — it lies in the city folder beside its source and
    turns up in every slice the source does. Wherever it is opened it has to say what it
    is, or it reads as a second similar photograph that came from nowhere."""

    def test_the_offer_for_a_copy_names_the_frame_it_was_made_from(self):
        source = self.plain_photo("holiday.jpg")
        self.patch_model()
        self.start_server()
        _status, payload = self.restore_frame({"file_id": source})
        copy_id = payload["item"]["file_id"]

        _status, offer = self.offer(copy_id)

        self.assertEqual(offer["restored_from"], {"file_id": source, "name": "holiday.jpg"})

    def test_an_ordinary_frame_carries_no_such_link(self):
        file_id = self.plain_photo("holiday.jpg")
        self.start_server()

        _status, offer = self.offer(file_id)

        self.assertIsNone(offer["restored_from"])

    def test_the_badge_is_drawn_from_it_and_never_calls_the_copy_a_photograph(self):
        self.start_server()
        html = self.html()
        self.assertIn("fmt(I18N.review_restore_source_badge,", html)
        self.assertIn("lightboxRestoreBadge.title = I18N.review_restore_badge_hint;", html)
        entry = ui._UI_STRINGS["review_restore_source_badge"]
        self.assertEqual(set(entry), {"ru", "en", "ja"})
        for lang in ("ru", "en", "ja"):
            with self.subTest(lang=lang):
                self.assertIn("{name}", entry[lang])
                self.assertIn("model" if lang == "en" else
                              ("модел" if lang == "ru" else "モデル"), entry[lang])


class TestPressingTwiceAndPressingWhileBusy(RestoreAnywhereBase):
    def test_the_second_press_from_the_same_place_returns_the_copy_that_exists(self):
        file_id = self.plain_photo("holiday.jpg")
        self.patch_model()
        self.start_server()

        _status, first = self.restore_frame({"file_id": file_id})
        _status, second = self.restore_frame({"file_id": file_id})

        self.assertFalse(first["reused"])
        self.assertTrue(second["reused"])
        self.assertEqual(second["item"]["file_id"], first["item"]["file_id"])
        self.assertEqual(self.loads, [self.cfg.features.restore_model])
        self.assertEqual(self.restored_names(), ["holiday_restored.jpg"])

    def test_the_route_is_still_one_of_the_guarded_ones(self):
        # F145: it writes a file beside the original AND a row in the index, and the run
        # it would race with rewrites the very tables that row needs. A second entrance
        # must not be a way around that.
        self.assertIn("/api/review/restore", ui.BUSY_REFUSED_ROUTES)

    def test_the_button_on_the_expanded_frame_is_dead_while_something_runs(self):
        self.start_server()
        html = self.html()
        self.assertIn("lightboxRestoreBtn.disabled = uiBusy() || lightboxRestoring;", html)
        self.assertIn("registerBusyRefresh(renderLightboxRestore);", html)


class TestTheOneEntranceIsTheExpandedFrame(RestoreAnywhereBase):
    """F133: not a control on every tile. Thirteen of those held a third of the screen and
    were reached for once a month; this one is reached for less often than that."""

    def test_the_action_lives_inside_the_lightbox_overlay(self):
        self.start_server()
        html = self.html()
        overlay = html[html.index('<div id="lightbox" '):]
        overlay = overlay[:overlay.index("<script>")]
        self.assertIn('id="lightbox-restore-btn"', overlay)
        self.assertIn('id="lightbox-restore-badge"', overlay)

    def test_no_tile_grows_a_button_of_its_own(self):
        self.start_server()
        html = self.html()
        # `clickableThumb` is the single opener of the expanded frame everywhere (cities,
        # people, animals, events, search), so one bar inside the overlay is one entrance
        # for all of them.
        self.assertEqual(html.count("function clickableThumb("), 1)
        self.assertEqual(html.count('id="lightbox-restore-btn"'), 1)
        self.assertEqual(html.count('id="review-restore-btn"'), 1)

    def test_it_asks_the_server_rather_than_deciding_for_itself(self):
        self.start_server()
        html = self.html()
        self.assertIn('fetch("/api/restore/offer?file_id="', html)
        self.assertIn('postJson("/api/review/restore", { file_id: id })', html)

    def test_a_clip_is_never_asked_about(self):
        self.start_server()
        html = self.html()
        self.assertIn("var id = lightboxFrames ? null : lightboxFrameId();", html)

    def test_the_copy_is_shown_where_the_press_happened(self):
        self.start_server()
        html = self.html()
        self.assertIn("lightboxSamples.splice(lightboxIndex + 1, 0, resp.item.file_id);",
                      html)
        self.assertIn("showLightboxAt(lightboxIndex + 1);", html)

    def test_the_reason_of_a_refusal_is_translated_by_the_existing_lookup(self):
        self.start_server()
        html = self.html()
        self.assertIn('I18N["review_restore_error_" + resp.reason]', html)
        for code in (ui.RESTORE_ERROR_SENSITIVE, ui.RESTORE_ERROR_VIDEO):
            with self.subTest(reason=code):
                entry = ui._UI_STRINGS[f"review_restore_error_{code}"]
                self.assertEqual(set(entry), {"ru", "en", "ja"})
                for lang in ("ru", "en", "ja"):
                    self.assertTrue(entry[lang].strip())

    def test_the_hint_of_the_expanded_frame_exists_in_three_languages(self):
        entry = ui._UI_STRINGS["review_restore_expanded_hint"]
        self.assertEqual(set(entry), {"ru", "en", "ja"})
        for lang in ("ru", "en", "ja"):
            with self.subTest(lang=lang):
                self.assertTrue(entry[lang].strip())
                # The one thing every string of this feature must say: the model draws,
                # it does not recover.
                self.assertIn("model" if lang == "en" else
                              ("модел" if lang == "ru" else "モデル"),
                              entry[lang].lower())


if __name__ == "__main__":
    unittest.main()
