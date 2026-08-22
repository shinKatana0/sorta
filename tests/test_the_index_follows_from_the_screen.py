"""F244: the index is carried over from the screen, not only from the terminal.

F242 gave the product `sorta relocate` and the stop in front of a moved collection: with
rows in the index and not one source folder on disk, a run does not start, because
indexing from the new location would call every file new and lose the face names, the
places, the marks and the duplicate decisions somebody entered by hand. That sentence
already reaches the collapsed error row of the run screen. The only way to act on it was
a terminal — for a product whose web app is an equal-rights entry point, that is half the
users left holding a diagnosis and no cure.

So: `POST /api/relocate`, a writing route like every other, and a form on the screen that
names the problem. What is checked here is that the route CALLS the engine rather than
carrying a second copy of it, that a refusal is an answer and not a 500, that the guards
of a writing route see the new route at all, and that the button which writes is
unreachable until a plan has been shown.
"""
from __future__ import annotations

import json
import re
import tempfile
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

from sorta import ui
from sorta.db import connect
from sorta.relocate import RelocateError, RelocatePlan
from sorta.ui import process as ui_process

from tests import waiting
from tests.test_ui import UiServerTestBase
from tests.test_ui_master_switch import _BODIES, _post_routes

_ROUTE = "/api/relocate"

# Every caption this feature added. Read by the translation case below and by the one
# that proves no interface string sits in the markup.
_STRING_KEYS = (
    "relocate_offer_hint", "relocate_open_button", "relocate_title", "relocate_hint",
    "relocate_old_label", "relocate_new_label", "relocate_plan_button",
    "relocate_apply_button", "relocate_plan_first", "relocate_needs_both",
    "relocate_plan_summary", "relocate_plan_example", "relocate_dest_ok",
    "relocate_dest_missing", "relocate_plan_empty", "relocate_applied",
    "relocate_refused_prefix",
)


class RelocateServerTestBase(UiServerTestBase):
    """An index whose files sit under `src`, and a folder they could have moved to.

    Nothing is moved on disk: the engine rewrites a prefix in the database and opens no
    file, so what makes this a move is the pair of prefixes and not the state of `src`.
    """

    def setUp(self):
        super().setUp()
        self.new_dir = self.root / "moved"
        self.new_dir.mkdir()
        self.add_photo_file("a.jpg")
        self.add_photo_file("b.jpg")

    def paths(self) -> dict[int, str]:
        return {row["id"]: row["path"]
                for row in self.conn.execute("SELECT id, path FROM files")}

    def relocate(self, old: object = None, new: object = None,
                 **extra: object) -> tuple[int, dict]:
        body: dict[str, object] = {
            "old_prefix": str(self.src_dir) if old is None else old,
            "new_prefix": str(self.new_dir) if new is None else new,
        }
        body.update(extra)
        answer = waiting.post_json(f"{self.base_url}{_ROUTE}", body)
        return answer.status, answer.json()


class TestThePlanComesBackBeforeAnythingIsWritten(RelocateServerTestBase):
    """Acceptance criteria 1 and 3: a dry run by default, and it says what it found."""

    def setUp(self):
        super().setUp()
        self.start_server()

    def test_without_the_flag_nothing_is_written(self):
        before = self.paths()
        status, resp = self.relocate()
        self.assertEqual(status, 200, resp)
        self.assertFalse(resp["applied"])
        self.assertGreater(resp["rows"], 0)
        self.assertEqual(self.paths(), before)

    def test_the_plan_names_the_columns_and_shows_examples(self):
        _status, resp = self.relocate()
        self.assertTrue(resp["columns"])
        self.assertEqual({"table", "column", "rows"}, set(resp["columns"][0]))
        self.assertTrue(resp["examples"])
        self.assertLessEqual(len(resp["examples"]), 3)
        before, after = resp["examples"][0]["before"], resp["examples"][0]["after"]
        self.assertIn("src", before.replace("\\", "/"))
        self.assertIn("moved", after.replace("\\", "/"))

    def test_it_says_whether_the_new_place_is_there(self):
        _status, resp = self.relocate()
        self.assertTrue(resp["new_prefix_exists"])
        _status, absent = self.relocate(new=str(self.root / "nowhere"))
        self.assertFalse(absent["new_prefix_exists"])
        # ...and a dry run to a folder that is not there is still an answer, not a
        # refusal: the refusals belong to the apply, which is the thing that writes.
        self.assertNotIn("error", absent)

    def test_a_prefix_that_matches_nothing_plans_no_rows(self):
        _status, resp = self.relocate(old=str(self.root / "elsewhere"))
        self.assertEqual(resp["rows"], 0)
        self.assertEqual(resp["columns"], [])


class TestApplyingItMovesTheIndex(RelocateServerTestBase):
    """Acceptance criterion 7: the rows point at the new place and keep their ids."""

    def setUp(self):
        super().setUp()
        self.start_server()

    def test_the_paths_move_and_the_ids_do_not(self):
        before = self.paths()
        status, resp = self.relocate(apply=True)
        self.assertEqual(status, 200, resp)
        self.assertTrue(resp["applied"])
        after = self.paths()
        self.assertEqual(set(before), set(after))
        for file_id, old_path in before.items():
            self.assertEqual(
                after[file_id].replace("\\", "/"),
                old_path.replace("\\", "/").replace(self.src_dir.as_posix(),
                                                    self.new_dir.as_posix()))

    def test_the_answer_carries_the_same_plan_marked_as_done(self):
        _status, dry = self.relocate()
        _status, applied = self.relocate(apply=True)
        self.assertEqual(applied["rows"], dry["rows"])
        self.assertEqual(applied["old_prefix"], dry["old_prefix"])
        self.assertEqual(applied["new_prefix"], dry["new_prefix"])
        self.assertTrue(applied["applied"])


class TestARefusalIsAnAnswerAndNotACrash(RelocateServerTestBase):
    """Acceptance criterion 3: a reason on screen, never a 500 — and nothing written.

    The three refusals are the engine's own (`relocate._refuse_unless_applicable`); what
    is checked here is that they arrive as something the page can say out loud.
    """

    def setUp(self):
        super().setUp()
        self.start_server()

    def assertRefused(self, resp: dict) -> str:
        self.assertEqual(resp["error"], ui.RELOCATE_REFUSED)
        self.assertTrue(resp["reason"].strip())
        return resp["reason"]

    def test_a_destination_that_is_not_there(self):
        before = self.paths()
        status, resp = self.relocate(new=str(self.root / "nowhere"), apply=True)
        self.assertEqual(status, 200)
        self.assertIn("nowhere", self.assertRefused(resp))
        self.assertEqual(self.paths(), before)

    def test_an_old_prefix_that_matches_nothing(self):
        before = self.paths()
        status, resp = self.relocate(old=str(self.root / "elsewhere"), apply=True)
        self.assertEqual(status, 200)
        self.assertRefused(resp)
        self.assertEqual(self.paths(), before)

    def test_two_rows_that_would_land_on_one_path(self):
        """A file already indexed at the destination — the move would make two rows of
        the same photograph, and the engine refuses the whole thing."""
        self.conn.execute(
            """INSERT INTO files (path, size, mtime, ext, media_type, indexed_at)
               VALUES (?, 1, 0, 'jpg', 'photo', '2026-01-01')""",
            (str(self.new_dir / "a.jpg"),))
        self.conn.commit()
        before = self.paths()
        status, resp = self.relocate(apply=True)
        self.assertEqual(status, 200)
        self.assertRefused(resp)
        self.assertEqual(self.paths(), before)

    def test_the_same_path_twice(self):
        status, resp = self.relocate(new=str(self.src_dir), apply=True)
        self.assertEqual(status, 200)
        self.assertRefused(resp)


class TestTheBodyIsValidated(RelocateServerTestBase):
    """Acceptance criterion 5: rubbish earns a 400 with a reason, and the program lives."""

    def setUp(self):
        super().setUp()
        self.start_server()

    def test_every_shape_of_rubbish_is_a_400(self):
        bodies: list[object] = [
            [], "old", 5, None, {},
            {"old_prefix": "/a"},
            {"new_prefix": "/b"},
            {"old_prefix": "", "new_prefix": "/b"},
            {"old_prefix": "   ", "new_prefix": "/b"},
            {"old_prefix": "/a", "new_prefix": 7},
            {"old_prefix": ["/a"], "new_prefix": "/b"},
            {"old_prefix": "/a", "new_prefix": "/b", "apply": "yes"},
            {"old_prefix": "/a", "new_prefix": "/b", "apply": 1},
        ]
        for body in bodies:
            with self.subTest(body=body):
                answer = waiting.post_json(f"{self.base_url}{_ROUTE}", body)
                self.assertEqual(answer.status, 400, answer.json())
                self.assertIn("error", answer.json())

    def test_the_server_still_answers_afterwards(self):
        waiting.post_json(f"{self.base_url}{_ROUTE}", "nonsense")
        status, _body, _ctype = self.get("/api/process/status")
        self.assertEqual(status, 200)

    def test_the_validator_on_its_own(self):
        self.assertEqual(
            ui._validate_relocate_payload({"old_prefix": " /a ", "new_prefix": "/b"}),
            ("/a", "/b", False))
        self.assertEqual(
            ui._validate_relocate_payload({"old_prefix": "/a", "new_prefix": "/b",
                                           "apply": True}),
            ("/a", "/b", True))


class TestItIsAWritingRoute(RelocateServerTestBase):
    """Acceptance criterion 4: both guards see it — asked of the guards themselves.

    They walk a list of routes, so a route missing from the list is not refused and not
    reported either: it is simply never asked about.
    """

    def test_the_dispatcher_knows_it_and_the_busy_table_does_too(self):
        self.assertIn(_ROUTE, _post_routes())
        self.assertIn(_ROUTE, ui.BUSY_REFUSED_ROUTES)
        self.assertNotIn(_ROUTE, ui._BUSY_EXEMPT_ROUTES)

    def test_the_shared_bodies_carry_it(self):
        """The two guard suites POST a real body per route; without an entry there this
        route would be walked with nothing to send and prove nothing."""
        self.assertIn(_ROUTE, _BODIES)

    def test_it_refuses_while_a_layout_is_running(self):
        state = ui._SortState()
        self.assertTrue(state.try_start())
        with mock.patch.object(ui, "_SortState", return_value=state):
            self.start_server()
            before = self.paths()
            status, resp = self.relocate(apply=True)
            self.assertEqual(status, 409, resp)
            self.assertEqual(self.paths(), before)

    def test_a_page_that_is_not_ours_may_not_move_the_index(self):
        self.start_server()
        before = self.paths()
        request = urllib.request.Request(
            f"{self.base_url}{_ROUTE}", method="POST",
            data=json.dumps({"old_prefix": str(self.src_dir),
                             "new_prefix": str(self.new_dir),
                             "apply": True}).encode("utf-8"),
            headers={"Content-Type": "text/plain"})
        answer = waiting.fetch(request)
        self.assertEqual(answer.status, 403)
        self.assertEqual(self.paths(), before)


class TestTheEngineIsCalledAndNotCopied(RelocateServerTestBase):
    """Acceptance criterion 8: `relocate.relocate` does the work, this route does not.

    The stand-in is put where the name is USED (`sorta.ui.process`) and not on the
    re-export in `sorta.ui`: the re-export binds its own reference at import, so patching
    it intercepts nothing (F182).
    """

    def setUp(self):
        super().setUp()
        self.start_server()

    def test_the_route_calls_it_with_what_the_body_asked_for(self):
        plan = RelocatePlan(old_prefix="/old", new_prefix="/new", rows=7, applied=True)
        with mock.patch.object(ui_process, "relocate", return_value=plan) as engine:
            status, resp = self.relocate(apply=True)
        self.assertEqual(status, 200)
        self.assertEqual(resp["rows"], 7)
        engine.assert_called_once()
        args, kwargs = engine.call_args
        self.assertEqual(Path(args[0]), Path(self.cfg.database).resolve())
        self.assertEqual((args[1], args[2]), (str(self.src_dir), str(self.new_dir)))
        self.assertEqual(kwargs, {"apply": True})

    def test_with_the_engine_out_of_the_way_the_route_writes_nothing(self):
        """If a second copy of the move lived here, the paths would change anyway."""
        before = self.paths()
        plan = RelocatePlan(old_prefix="/old", new_prefix="/new", rows=1, applied=True)
        with mock.patch.object(ui_process, "relocate", return_value=plan):
            self.relocate(apply=True)
        self.assertEqual(self.paths(), before)

    def test_the_engines_refusal_is_the_one_the_route_reports(self):
        with mock.patch.object(ui_process, "relocate",
                               side_effect=RelocateError("no such folder")):
            status, resp = self.relocate(apply=True)
        self.assertEqual(status, 200)
        self.assertEqual(resp["error"], ui.RELOCATE_REFUSED)
        self.assertEqual(resp["reason"], "no such folder")


class TestTheOldPrefixIsOffered(RelocateServerTestBase):
    """Acceptance criterion 2: the field is prefilled from the index."""

    def suggested(self) -> str:
        status, body, _ctype = self.get("/api/relocate/suggest")
        self.assertEqual(status, 200)
        return json.loads(body)["old_prefix"]

    def test_it_is_the_folder_the_indexed_files_sit_under(self):
        self.start_server()
        suggestion = self.suggested()
        self.assertTrue(suggestion)
        self.assertEqual(Path(suggestion), self.src_dir.resolve())

    def test_what_is_offered_is_what_the_route_accepts(self):
        """The prefill is worth nothing if the plan it produces is empty."""
        self.start_server()
        _status, resp = self.relocate(old=self.suggested())
        self.assertGreater(resp["rows"], 0)


class TestTheOfferedPrefixOnItsOwn(unittest.TestCase):
    """`_indexed_prefix` — the cases a live index cannot easily be put into."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "index.db"
        self.conn = connect(self.db)

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def index(self, *paths: str) -> None:
        for path in paths:
            self.conn.execute(
                """INSERT INTO files (path, size, mtime, ext, media_type, indexed_at)
                   VALUES (?, 1, 0, 'jpg', 'photo', '2026-01-01')""", (path,))
        self.conn.commit()

    def test_an_empty_index_offers_nothing(self):
        self.assertEqual(ui._indexed_prefix(self.db), "")

    def test_one_file_offers_the_folder_it_is_in(self):
        self.index("D:/Photos/2019/a.jpg")
        self.assertEqual(ui._indexed_prefix(self.db), "D:/Photos/2019")

    def test_the_prefix_is_cut_at_a_separator_and_not_mid_name(self):
        """`a1.jpg` and `a2.jpg` share the letter `a`, which is not a folder."""
        self.index("D:/Photos/a1.jpg", "D:/Photos/a2.jpg")
        self.assertEqual(ui._indexed_prefix(self.db), "D:/Photos")

    def test_backslashes_are_answered_as_they_are_stored(self):
        self.index(r"D:\Photos\a.jpg", r"D:\Photos\b.jpg")
        self.assertEqual(ui._indexed_prefix(self.db), r"D:\Photos")

    def test_two_drives_offer_nothing(self):
        self.index("D:/Photos/a.jpg", "E:/Photos/b.jpg")
        self.assertEqual(ui._indexed_prefix(self.db), "")

    def test_a_bare_root_offers_nothing(self):
        """`D:` means "wherever that drive is standing" to `Path.resolve`, which is a
        different folder from `D:\\` — offering it would prefill the wrong thing."""
        self.index("D:/a.jpg", "D:/b.jpg")
        self.assertEqual(ui._indexed_prefix(self.db), "")
        self.conn.execute("DELETE FROM files")
        self.index("/a.jpg", "/b.jpg")
        self.assertEqual(ui._indexed_prefix(self.db), "")


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


class TestTheScreenOffersTheMove(ScreenCase):
    """Requirement 3: the action is where the person meets the problem."""

    def test_the_offer_is_drawn_next_to_the_failed_stage(self):
        self.assertIn('id="relocate-offer"', self.html)
        self.assertIn("renderRelocateOffer(data);", self.body("renderProcessStatus"))

    def test_it_appears_for_a_failed_index_stage_and_for_nothing_else(self):
        body = self.body("renderRelocateOffer")
        self.assertIn("data.error_stage || data.stage", body)
        self.assertIn('failed === "index" ? "" : "none"', body)

    def test_the_form_has_both_prefixes_and_the_folder_picker(self):
        for element in ('id="relocate-old"', 'id="relocate-new"',
                        'id="relocate-browse-btn"'):
            self.assertIn(element, self.html)
        self.assertIn('document.getElementById("relocate-new").value = path',
                      self.html)

    def test_the_old_prefix_is_asked_of_the_server(self):
        self.assertIn('fetch("/api/relocate/suggest")', self.html)


class TestTheMoveWaitsForItsPlan(ScreenCase):
    """Requirement: the button that writes is unreachable until the plan is on screen.

    The whole product works this way — `sort` is a dry run first, the run screen quotes
    its own price, the layout asks before it moves anything — and a button that moved the
    index on the first press would be the one exception.
    """

    def test_the_apply_button_starts_disabled(self):
        self.assertIn('id="relocate-apply-btn" class="btn btn-primary" disabled',
                      self.html)

    def test_only_a_shown_plan_enables_it(self):
        self.assertIn("setRelocatePlanned(!!resp.rows);", self.body("requestRelocate"))
        self.assertIn('.disabled = !relocatePlanned;', self.body("setRelocatePlanned"))

    def test_an_edited_path_takes_the_plan_away_again(self):
        self.assertIn("setRelocatePlanned(false);", self.body("resetRelocatePlan"))
        self.assertIn('.addEventListener("input", resetRelocatePlan)', self.html)

    def test_the_plan_button_asks_for_a_dry_run(self):
        self.assertIn("requestRelocate(false);", self.html)
        self.assertIn("apply: apply", self.body("requestRelocate"))


class TestThePageDoesNotKeepStaleNumbers(ScreenCase):
    """Requirement 5: after the move every count on the page is about the old paths."""

    def test_a_finished_move_refreshes_what_the_page_shows(self):
        body = self.body("requestRelocate")
        applied = body[body.index("resp.applied"):]
        self.assertIn("refreshTabsAfterProcess();", applied)
        self.assertIn("I18N.relocate_applied", applied)


class TestEveryCaptionGoesThroughI18n(ScreenCase):
    """Requirement 4: three languages, and not one string written into the markup."""

    def test_every_key_exists_in_all_three_languages(self):
        for key in _STRING_KEYS:
            with self.subTest(key=key):
                entry = ui._UI_STRINGS[key]
                self.assertEqual(set(entry), {"ru", "en", "ja"})
                for lang, value in entry.items():
                    self.assertTrue(value.strip(), f"{key}/{lang} пуст")

    def test_the_markup_holds_placeholders_and_no_captions(self):
        markup = (ui._WEB_DIR / "page.html").read_text(encoding="utf-8")
        panel = markup[markup.index('id="relocate-offer"'):
                       markup.index('id="process-summary"')]
        for key in ("relocate_offer_hint", "relocate_open_button", "relocate_title",
                    "relocate_hint", "relocate_old_label", "relocate_new_label",
                    "relocate_plan_button", "relocate_apply_button"):
            with self.subTest(key=key):
                self.assertIn("{{" + key + "}}", panel)
                self.assertNotIn(ui._UI_STRINGS[key]["ru"], panel)

    def test_the_script_says_nothing_in_a_language_of_its_own(self):
        """Every caption the relocate script shows comes out of `I18N`, so a person
        reading the page in Japanese is not shown one Russian line."""
        script = "".join(self.body(name) for name in
                         ("renderRelocatePlan", "requestRelocate", "resetRelocatePlan"))
        self.assertFalse(re.findall(r'"[^"]*[а-яА-Я\u3040-\u30ff\u4e00-\u9fff][^"]*"',
                                    script))
        for key in ("relocate_plan_summary", "relocate_plan_empty", "relocate_dest_ok",
                    "relocate_dest_missing", "relocate_applied",
                    "relocate_refused_prefix", "relocate_needs_both"):
            with self.subTest(key=key):
                self.assertIn("I18N." + key, script)

    def test_the_three_languages_reach_the_rendered_page(self):
        for lang in ("ru", "en", "ja"):
            with self.subTest(lang=lang):
                html = ui._render_index_html(lang)
                self.assertIn(ui._UI_STRINGS["relocate_open_button"][lang], html)
                self.assertNotIn("{{relocate_", html)


if __name__ == "__main__":
    unittest.main()
