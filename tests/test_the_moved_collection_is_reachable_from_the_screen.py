"""F248: the stop about a moved collection can be reached from the interface.

F242 wrote the stop and F244 the way out of it, and neither could be reached by the one
path they were written for: index a folder, rename it, press start. `POST /api/process`
checks `is_dir()` BEFORE the pipeline, so the run ended at `Failed to start: not a
directory` — no stage, no `error_stage == "index"`, and therefore no offer to carry the
index over.

The check itself is right — there is no sense walking a folder that is not there. What
was missing is that "the folder is not there" says two things: a mistyped path over an
empty index, and a collection that moved over a full one. So the refusal now tells them
apart, and the second one carries what draws the transfer panel.

The case that matters most here is `TestTheWholePathFromStartToTheAnswer`: every part of
this feature was already correct on its own, and what nobody had done was walk it end to
end.
"""
from __future__ import annotations

import json
import re
import unittest
from unittest import mock

import pytest

from sorta import ui
from sorta.indexer import index as real_index
from sorta.relocate import CollectionMoved
from sorta.ui import process as ui_process

from tests import waiting
from tests.test_ui_process import ProcessTestBase, _poll_until

_ROUTE = "/api/process"


class SourceRefusalTestBase(ProcessTestBase):
    """A server whose source folder can be taken away under it."""

    def start(self, source_dir: str) -> tuple[int, dict]:
        return self.post(_ROUTE, {"source_dir": source_dir})

    def gone(self) -> str:
        """A path under the temporary root that is not there — the renamed folder."""
        return str(self.root / "renamed-in-the-explorer")


class TestATypoIsStillATypo(SourceRefusalTestBase):
    """Acceptance criterion 1: an empty index has nothing to have moved."""

    def setUp(self):
        super().setUp()
        self.patch_fast_stages()
        self.start_server()

    def test_the_refusal_names_the_path_and_offers_no_move(self):
        status, resp = self.start(self.gone())
        self.assertEqual(status, 400, resp)
        self.assertEqual(resp["error"], ui.SOURCE_MISSING)
        self.assertEqual(resp["params"], {"path": self.gone()})
        self.assertNotIn("relocate", resp)

    def test_the_refusal_is_a_code_and_not_a_raw_english_string(self):
        _status, resp = self.start(self.gone())
        self.assertEqual(resp["code"], ui.SOURCE_MISSING)
        self.assertIn(f"fault_{ui.SOURCE_MISSING}", ui._UI_STRINGS)
        self.assertNotEqual(resp["error"], "not a directory")

    def test_a_file_is_not_a_folder_and_is_not_a_move_either(self):
        """`is_dir()` is false for a file that exists; the collection has not moved."""
        _file_id, path, _content = self.add_photo_file("a.jpg")
        _status, resp = self.start(str(path))
        self.assertEqual(resp["error"], ui.SOURCE_MISSING)
        self.assertNotIn("relocate", resp)

    def test_nothing_was_walked(self):
        self.start(self.gone())
        self.assertEqual(self.calls, [])
        self.assertFalse(self.status()["running"])


class TestAFullIndexMeansTheCollectionMoved(SourceRefusalTestBase):
    """Acceptance criteria 2, 3 and 5."""

    def setUp(self):
        super().setUp()
        self.patch_fast_stages()
        self.add_photo_file("a.jpg")
        self.add_photo_file("b.jpg")
        self.start_server()

    def test_the_answer_explains_the_move(self):
        status, resp = self.start(self.gone())
        self.assertEqual(status, 400, resp)
        self.assertEqual(resp["error"], "relocate_collection_moved")
        self.assertEqual(resp["code"], "relocate_collection_moved")
        self.assertEqual(resp["params"]["rows"], 2)
        self.assertIn("relocate", resp["reason"])

    def test_the_answer_carries_what_the_page_draws_the_offer_from(self):
        _status, resp = self.start(self.gone())
        self.assertIn("relocate", resp)
        self.assertEqual(resp["relocate"]["old_prefix"],
                         ui._indexed_prefix(self.cfg.database))

    def test_the_prefix_it_offers_is_the_one_the_move_accepts(self):
        """A prefilled path is worth nothing if the plan built from it is empty."""
        _status, resp = self.start(self.gone())
        answer = waiting.post_json(
            f"{self.base_url}/api/relocate",
            {"old_prefix": resp["relocate"]["old_prefix"],
             "new_prefix": str(self.root)})
        self.assertGreater(answer.json()["rows"], 0)

    def test_nothing_was_walked_here_either(self):
        self.start(self.gone())
        self.assertEqual(self.calls, [])
        self.assertFalse(self.status()["running"])

    def test_a_source_that_is_there_still_runs(self):
        status, _resp = self.start(str(self.src_dir))
        self.assertEqual(status, 200)
        _poll_until(self.status, lambda d: d["finished"])
        self.assertIn("index", self.calls)


class TestTheThresholdIsNotASecondCopy(SourceRefusalTestBase):
    """Requirement 1: "moved" is decided by `relocate` and asked of it, not re-derived."""

    def setUp(self):
        super().setUp()
        self.add_photo_file("a.jpg")

    def test_the_guard_of_relocate_is_the_one_that_answers(self):
        with mock.patch.object(
                ui_process, "refuse_if_the_collection_moved") as guard:
            refusal = ui_process._source_refusal(self.cfg.database, self.cfg, self.gone())
        guard.assert_called_once()
        cfg, _conn = guard.call_args.args
        self.assertEqual([str(src) for src in cfg.sources], [self.gone()])
        # With the guard silent there is no move to report, whatever the index holds.
        self.assertEqual(refusal["error"], ui.SOURCE_MISSING)

    def test_what_that_guard_raises_is_what_the_route_reports(self):
        moved = CollectionMoved("the collection was not found", "relocate_collection_moved",
                                roots="D:/photos", rows=17, sample="D:/photos/a.jpg")
        with mock.patch.object(
                ui_process, "refuse_if_the_collection_moved", side_effect=moved):
            refusal = ui_process._source_refusal(self.cfg.database, self.cfg, self.gone())
        self.assertEqual(refusal["reason"], "the collection was not found")
        self.assertEqual(refusal["params"], {"roots": "D:/photos", "rows": 17,
                                             "sample": "D:/photos/a.jpg"})

    def test_a_folder_that_is_there_is_not_asked_about_at_all(self):
        self.assertIsNone(
            ui_process._source_refusal(self.cfg.database, self.cfg, str(self.src_dir)))

    def test_the_refusal_survives_a_trip_through_json(self):
        """It is answered as a body, so a value the encoder cannot take would be a 500
        at the exact moment the person needs a sentence."""
        refusal = ui_process._source_refusal(self.cfg.database, self.cfg, self.gone())
        self.assertEqual(json.loads(json.dumps(refusal)), refusal)


# The real indexer runs inside the server process here, which is what makes this the one
# case that walks the whole feature; in the parallel half it costs its neighbours.
@pytest.mark.serial
class TestTheWholePathFromStartToTheAnswer(SourceRefusalTestBase):
    """Acceptance criterion 6 — the case whose absence let the defect through.

    Index a real folder through the real route, rename it the way a person does in the
    explorer, press start again, and read the answer the page reads.
    """

    def setUp(self):
        super().setUp()
        self.patch_fast_stages()
        # ...with the index stage put back: the whole point is that the rows in the
        # database were written by a run and not by a fixture.
        self._patch("run_index", real_index)
        self.start_server()

    def test_index_rename_start_and_the_offer_is_in_the_answer(self):
        self.add_photo_file("a.jpg")
        status, _resp = self.start(str(self.src_dir))
        self.assertEqual(status, 200)
        _poll_until(self.status, lambda d: d["finished"])
        indexed = self.conn.execute("SELECT count(*) AS n FROM files").fetchone()["n"]
        self.assertGreater(indexed, 0)

        self.src_dir.rename(self.root / "photos-2019")

        status, resp = self.start(str(self.src_dir))
        self.assertEqual(status, 400, resp)
        self.assertEqual(resp["code"], "relocate_collection_moved")
        self.assertTrue(resp["relocate"]["old_prefix"])
        # The sentence the page will draw exists in all three languages...
        for lang in ("ru", "en", "ja"):
            with self.subTest(lang=lang):
                self.assertTrue(
                    ui._UI_STRINGS[f"fault_{resp['code']}"][lang].format(**resp["params"]))
        # ...and the run this refused never started.
        self.assertFalse(self.status()["running"])


class TestTheOldPathIsStillRefusedTheOldWay(ProcessTestBase):
    """Acceptance criterion 7: the offer beside a failed `index` stage keeps working.

    That path is the one that fires when the walk DID begin — `cfg.sources` pointing
    somewhere gone while the request named a folder that exists.
    """

    def test_a_stage_that_stops_on_the_move_still_leaves_the_state_behind(self):
        state = ui._ProcessState()
        self.assertTrue(state.try_start(str(self.src_dir)))

        def boom(cfg, conn, progress=None):
            raise CollectionMoved("the collection was not found",
                                  "relocate_collection_moved", roots="D:/photos",
                                  rows=17, sample="D:/photos/a.jpg")

        with mock.patch.object(ui_process, "_pipeline_steps",
                                        lambda notify: [("index", boom)]):
            ui.process._run_pipeline(self.cfg.database, self.cfg, str(self.src_dir),
                                     state, ui.PlanCache(self.cfg, self.conn,
                                                         self.root / "_preview"))
        snapshot = state.snapshot()
        self.assertEqual(snapshot["error_stage"], "index")
        self.assertEqual(snapshot["error_code"], "relocate_collection_moved")


class ScreenCase(unittest.TestCase):
    """The rendered page, read as text — there is no engine here to run the script."""

    @classmethod
    def setUpClass(cls):
        cls.html = ui._render_index_html("ru")

    def body(self, name: str) -> str:
        start = self.html.index(f"function {name}(")
        depth = 0
        for j in range(self.html.index("{", start), len(self.html)):
            if self.html[j] == "{":
                depth += 1
            elif self.html[j] == "}":
                depth -= 1
                if depth == 0:
                    return self.html[start:j + 1]
        raise AssertionError(f"тело {name} не найдено")

    def start_handler(self) -> str:
        """The click handler of the run button, from its `postJson` to the end."""
        start = self.html.index('postJson("/api/process", {')
        return self.html[start:self.html.index("browseIntoField", start)]


class TestTheRefusalReachesTheSameOnePanel(ScreenCase):
    """Acceptance criteria 2, 3 and 7, on the page side."""

    def test_a_refused_start_opens_the_offer(self):
        self.assertIn("if (resp.relocate) showRelocateOffer(resp.relocate.old_prefix);",
                      self.start_handler())

    def test_the_offer_is_the_one_a_failed_index_stage_opens(self):
        opener = self.body("showRelocateOffer")
        self.assertIn("setRelocateOfferShown(true);", opener)
        self.assertIn("setRelocateOfferShown(failed === \"index\" "
                      "|| relocateOfferedByStart);", self.body("renderRelocateOffer"))

    def test_the_failed_stage_path_still_decides_on_the_stage(self):
        self.assertIn("data.error_stage || data.stage", self.body("renderRelocateOffer"))

    def test_the_old_prefix_from_the_answer_prefills_the_field(self):
        opener = self.body("showRelocateOffer")
        self.assertIn('document.getElementById("relocate-old")', opener)
        self.assertIn("field.value = oldPrefix", opener)

    def test_a_poll_does_not_take_the_offer_away_again(self):
        """The refusal reaches no stage, so the status snapshot knows nothing about it."""
        self.assertIn("relocateOfferedByStart = true;", self.body("showRelocateOffer"))
        self.assertIn("relocateOfferedByStart = false;", self.start_handler())
        self.assertIn("relocateOfferedByStart = false;", self.body("requestRelocate"))


class TestTheStartRefusalSpeaksTheInterfaceLanguage(ScreenCase):
    """Acceptance criterion 4: three languages, and no raw English on a localized screen."""

    def test_the_page_renders_the_refusal_from_the_catalog(self):
        self.assertIn("I18N.process_start_error_prefix + faultSentence(resp);",
                      self.start_handler())

    def test_both_refusals_are_in_all_three_languages(self):
        for code in (ui.SOURCE_MISSING, "relocate_collection_moved"):
            entry = ui._UI_STRINGS[f"fault_{code}"]
            with self.subTest(code=code):
                self.assertEqual(set(entry), {"ru", "en", "ja"})
                for lang, text in entry.items():
                    self.assertTrue(text.strip(), f"{code}/{lang} пуст")

    def test_the_english_column_is_the_sentence_the_server_sends(self):
        """The F245 rule: `en` IS the message, so an English screen shows what the log
        holds and the two cannot drift."""
        self.assertEqual(
            ui._UI_STRINGS[f"fault_{ui.SOURCE_MISSING}"]["en"].format(path="/a/b"),
            ui_process._SOURCE_MISSING_HINT.format(path="/a/b"))

    def test_the_browser_gets_the_key_for_every_language(self):
        for lang in ("ru", "en", "ja"):
            served = json.loads(re.search(r"window\.I18N = (\{.*\});",
                                          ui._render_index_html(lang)).group(1))
            with self.subTest(lang=lang):
                self.assertTrue(served[f"fault_{ui.SOURCE_MISSING}"])


if __name__ == "__main__":
    unittest.main()
