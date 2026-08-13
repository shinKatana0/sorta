"""F198: above the ceiling the answer is NO — a refusal instead of work done for nothing.

F169 shipped the ceiling as a SETTING and left the question of what to do above it open on
purpose: it was written before the measurement, so it did the work and said afterwards that
the copy had been rebuilt from a reduced frame. The measurement came back on 2026-08-04 —
35/35/30 on blind pairs above the ceiling, which is nothing — and the code stayed in the
state of waiting for an answer that had arrived. On 2026-08-05 the owner pressed the button
on a 4320 px frame: the model ran, a near-duplicate file appeared beside the original, a row
appeared in the index, and the honest warning arrived AFTER all of it.

So this file is about three things:

* a frame above `features.restore_max_edge` is refused BY THE ROUTE, and nothing is written.
  Checked with a request that never met the interface, because a hidden button forbids
  nothing (the F133 rule the two other refusals already live by);
* the refusal names the limit, the size of this frame and the key that moves the limit —
  a person who disagrees with the threshold has to be able to see that it is theirs;
* the offer and the route read ONE answer. They disagreed for a day and cost a useless
  file; a second place deciding "may this frame be processed" is a second thing to forget.

Below the ceiling nothing here narrows the action: that is where the gain was measured, and
`tests/test_ui_restore.py` keeps that case.
"""
from __future__ import annotations

import dataclasses
import unittest
from unittest import mock

from sorta import ui

from tests import waiting
from tests.test_ui_restore_anywhere import RestoreAnywhereBase


class CeilingTestBase(RestoreAnywhereBase):
    def set_ceiling(self, max_edge: int) -> None:
        self.cfg = dataclasses.replace(
            self.cfg,
            features=dataclasses.replace(self.cfg.features, restore_max_edge=max_edge))

    def restart_server(self) -> None:
        """A changed ceiling needs a new server — the config is read when it is built,
        which is also what a person editing the key and restarting `sorta ui` does."""
        if self.server is not None:
            waiting.stop_server(self.server, self.thread)
            self.server = None
        self.start_server()

    def restored_rows(self) -> list[int]:
        return [r["file_id"] for r in
                self.conn.execute("SELECT file_id FROM restored_files").fetchall()]


class TestTheFrameAboveTheCeilingIsRefused(CeilingTestBase):
    """The main case, in the owner's own numbers: 4320 px against a 1024 px ceiling."""

    def test_the_route_refuses_and_leaves_nothing_on_disk(self):
        file_id = self.big_photo("owner.jpg", (4320, 2880))
        self.patch_model()
        self.start_server()

        status, payload = self.restore_frame({"file_id": file_id})

        self.assertEqual(status, 200, payload)
        self.assertFalse(payload["ok"], payload)
        self.assertEqual(payload["reason"], ui.RESTORE_ERROR_TOO_LARGE)
        # The point of the refusal: no model run, no file beside the original, no row in
        # the index. The old behaviour spent all three and produced an almost-duplicate.
        self.assertEqual(self.loads, [])
        self.assertEqual(self.restored_names(), [])
        self.assertEqual(self.restored_rows(), [])

    def test_the_refusal_arrives_even_though_the_request_skipped_the_page(self):
        """The button is withdrawn on the expanded frame, but the Review tab still offers
        the action on any single selected frame — and a request can be made by hand. The
        rule is the route's, so all three meet the same one."""
        file_id = self.big_photo("big.jpg")
        self.set_ceiling(600)
        self.patch_model()
        self.start_server()

        _status, offer = self.offer(file_id)
        _status, payload = self.restore_frame({"file_id": file_id})

        self.assertFalse(offer["available"], offer)
        self.assertFalse(payload["ok"], payload)
        self.assertEqual(self.restored_names(), [])

    def test_the_refusal_names_the_limit_and_the_size_of_this_frame(self):
        file_id = self.big_photo("owner.jpg", (4320, 2880))
        self.patch_model()
        self.start_server()

        _status, payload = self.restore_frame({"file_id": file_id})

        self.assertEqual(payload["source_edge"], 4320)
        self.assertEqual(payload["max_edge"], 1024)
        self.assertTrue(payload["rebuilt"])

    def test_both_numbers_reach_the_translated_sentence(self):
        """The wording is the one the warning used, and it is only worth anything filled
        in: "too large" without saying too large for WHAT is not something to act on."""
        file_id = self.big_photo("owner.jpg", (4320, 2880))
        self.patch_model()
        self.start_server()

        _status, payload = self.restore_frame({"file_id": file_id})

        entry = ui._UI_STRINGS[f"review_restore_error_{ui.RESTORE_ERROR_TOO_LARGE}"]
        self.assertEqual(set(entry), {"ru", "en", "ja"})
        for lang in ("ru", "en", "ja"):
            with self.subTest(lang=lang):
                sentence = entry[lang].format(**payload)
                self.assertIn("4320", sentence)
                self.assertIn("1024", sentence)
                self.assertIn("features.restore_max_edge", sentence)

    def test_the_ceiling_is_the_setting_and_the_same_frame_passes_when_it_is_raised(self):
        """The threshold is the person's: 1024 is where the measured usefulness ended, and
        the key is what says so — the same frame is refused under it and processed over."""
        self.set_ceiling(600)
        file_id = self.big_photo("big.jpg")
        self.patch_model()
        self.start_server()
        _status, refused = self.restore_frame({"file_id": file_id})

        self.set_ceiling(4000)
        self.restart_server()
        _status, done = self.restore_frame({"file_id": file_id})

        self.assertEqual(refused["reason"], ui.RESTORE_ERROR_TOO_LARGE)
        self.assertTrue(done["ok"], done)
        self.assertEqual(self.restored_names(), ["big_restored.jpg"])

    def test_a_frame_exactly_at_the_ceiling_is_processed(self):
        """`> max_edge`, not `>=`: the ceiling is the largest size the action still works
        at, and F169 hands such a frame to the model untouched."""
        self.set_ceiling(2400)
        file_id = self.big_photo("edge.jpg")
        self.patch_model()
        self.start_server()

        _status, payload = self.restore_frame({"file_id": file_id})

        self.assertTrue(payload["ok"], payload)
        self.assertFalse(payload["rebuilt"])
        self.assertEqual(payload["source_edge"], 2400)


class TestBelowTheCeilingNothingIsNarrowed(CeilingTestBase):
    """Requirement 4 of the brief. The gain is measured there — 62% against 10% for plain
    bicubic on small frames — and this feature is not allowed to touch it."""

    def test_a_small_frame_is_offered_and_processed_as_before(self):
        file_id = self.plain_photo("holiday.jpg")
        self.patch_model()
        self.start_server()

        _status, offer = self.offer(file_id)
        _status, payload = self.restore_frame({"file_id": file_id})

        self.assertTrue(offer["available"], offer)
        self.assertIsNone(offer["reason"])
        self.assertTrue(payload["ok"], payload)
        self.assertFalse(payload["rebuilt"])
        self.assertEqual(self.restored_names(), ["holiday_restored.jpg"])
        self.assertEqual(self.loads, [self.cfg.features.restore_model])


class TestTheOfferAndTheRouteAreOneAnswer(CeilingTestBase):
    """The defect itself. Two entrances answered "may this frame be processed" separately:
    the offer withdrew itself above the ceiling, the route did the work anyway. What is
    checked here is not the ceiling but the SHAPE — one function, two readers."""

    def frames(self) -> dict[str, int]:
        self.set_ceiling(600)
        return {
            "small": self.plain_photo("holiday.jpg"),
            "big": self.big_photo("big.jpg"),
            "document": self.add_reviewable("passport.jpg", verdict="document"),
            "video": self.plain_photo("clip.mp4"),
        }

    def test_every_frame_gets_the_same_verdict_from_both(self):
        frames = self.frames()
        self.patch_model()
        self.start_server()

        for name, file_id in frames.items():
            with self.subTest(frame=name):
                _status, offer = self.offer(file_id)
                _status, payload = self.restore_frame({"file_id": file_id})
                self.assertEqual(offer["available"], bool(payload.get("ok")))
                self.assertEqual(offer["reason"], payload.get("reason"))

    def test_the_reasons_are_the_ones_the_bans_of_F168_carry(self):
        """...and the ceiling did not swallow them: a private class is still refused
        because of what it is, not because of how large it is."""
        frames = self.frames()
        self.patch_model()
        self.start_server()

        _status, document = self.restore_frame({"file_id": frames["document"]})
        _status, video = self.restore_frame({"file_id": frames["video"]})
        _status, big = self.restore_frame({"file_id": frames["big"]})

        self.assertEqual(document["reason"], ui.RESTORE_ERROR_SENSITIVE)
        self.assertEqual(video["reason"], ui.RESTORE_ERROR_VIDEO)
        self.assertEqual(big["reason"], ui.RESTORE_ERROR_TOO_LARGE)

    def test_neither_entrance_has_an_opinion_of_its_own(self):
        """Stated by taking the one answer away: with the decision replaced, BOTH follow
        it. A route that reached its own conclusion would ignore this and process."""
        file_id = self.plain_photo("holiday.jpg")
        self.patch_model()
        self.start_server()
        invented = {"available": False, "reason": "invented", "rebuilt": False,
                    "source_edge": 7, "max_edge": 8}

        with mock.patch.object(ui.review, "_restore_decision", return_value=invented):
            _status, offer = self.offer(file_id)
            _status, payload = self.restore_frame({"file_id": file_id})

        self.assertEqual(offer["reason"], "invented")
        self.assertEqual(payload["reason"], "invented")
        self.assertFalse(payload["ok"], payload)
        self.assertEqual(self.restored_names(), [])

    def test_a_document_above_the_ceiling_is_refused_as_a_document(self):
        """The order of the two: the size is read off the file's header, and a personal
        document is a file this program opens for no purpose at all."""
        self.set_ceiling(600)
        file_id = self.add_reviewable("passport.jpg", verdict="document")
        self.patch_model()
        self.start_server()

        _status, payload = self.restore_frame({"file_id": file_id})

        self.assertEqual(payload["reason"], ui.RESTORE_ERROR_SENSITIVE)
        self.assertEqual(payload["source_edge"], 0)


class TestTheCopiesAlreadyMadeAreLeftAlone(CeilingTestBase):
    """Requirement 5. The useless copies of yesterday stay where they are: deleting them
    is a decision a person makes, not a side effect of a fix."""

    def test_lowering_the_ceiling_refuses_the_next_press_and_keeps_the_old_copy(self):
        self.set_ceiling(4000)
        file_id = self.big_photo("big.jpg")
        self.patch_model()
        self.start_server()
        _status, first = self.restore_frame({"file_id": file_id})
        copy_id = first["item"]["file_id"]

        self.set_ceiling(600)
        self.restart_server()
        _status, second = self.restore_frame({"file_id": file_id})

        self.assertEqual(second["reason"], ui.RESTORE_ERROR_TOO_LARGE)
        # The file and the row it is known by are exactly as they were.
        self.assertEqual(self.restored_names(), ["big_restored.jpg"])
        self.assertEqual(self.restored_rows(), [copy_id])
        self.assertTrue((self.src_dir / "big_restored.jpg").exists())


class TestTheRefusalReachesTheScreen(CeilingTestBase):
    def test_both_entrances_translate_it_with_the_numbers_of_the_answer(self):
        self.start_server()
        html = self.html()
        # The lookup is the one every other reason goes through — and it fills the
        # sentence from the answer, which is what makes "too large" say how large.
        self.assertEqual(
            html.count('fmt(I18N["review_restore_error_" + resp.reason] || "", resp)'), 2)

    def test_the_withdrawn_button_says_the_same_sentence(self):
        self.start_server()
        html = self.html()
        self.assertIn('var note = offer.reason === "too_large"', html)
        self.assertIn("fmt(I18N.review_restore_error_too_large,", html)
        self.assertEqual(ui.RESTORE_ERROR_TOO_LARGE, "too_large")


if __name__ == "__main__":
    unittest.main()
