"""F156: a person pins their own query as a slice.

The measurement that produced this feature (2026-08-02, a random sample of 200 frames):
65 of them — a third — fall into no class at all, and the ten candidate slices for those
65 cover 26%, 23%, 22%, 20%, 18%, 17%, 15%, 12%, 12% and 6%. Not one reaches a third of a
third, and food, which both the user and the author held in mind as a large slice, came
out at 8 frames — smaller than sky or signage. Ten slices for 65 frames out of 200 would
rebuild the thirteen-control remote F133 took apart. So the product stops guessing which
facets matter and the owner of the archive pins their own.

What the tests below are about:

* the pins live in `config.yaml` and nowhere else. The index does not survive `reset` or a
  re-processing and the config file does — a slice somebody named must not be one
  re-index away from gone;
* a pinned slice is indistinguishable from a built-in one: the same grid, the same album
  (`kind='query'`, `move_batches.mode='album_query'`), the same counter;
* unpinning removes a config entry and not one photograph;
* every refusal is a sentence — an empty query, a duplicate name, and above all the limit
  (`features.max_pinned_slices`), which is F133's bound and not a resource one;
* a pinned slice answers exactly what the search line answered for those words. If the two
  drifted, the pin would be a second engine, which is the thing F151 exists not to be;
* and the other half of the same idea: an empty BUILT-IN slice says WHICH empty it is.
  "The stage never ran" comes with a link to the run screen; "it ran and there is none of
  this" is a fact the collection has already stated. A bare zero reads as the second when
  it is nearly always the first — the `frame_quality` rule of F125, applied to a slice.

No model is loaded anywhere: `ui.text_encoder` is the fake of `tests.test_ui_search`, and
pinning does not rank anything at all — it saves words.
"""
from __future__ import annotations

import dataclasses
import json
import tempfile
import threading
import unittest
import urllib.parse
from pathlib import Path

from sorta import ui
from sorta.config import (
    DEFAULT_SAVED_SLICES,
    SavedSlice,
    load_config,
    save_saved_slices,
)

from tests import waiting
from tests.test_search import unit
from tests.test_ui import UiServerTestBase
from tests.test_ui_search import SearchUiTestBase

_ROOT = Path(__file__).resolve().parent.parent


class PinTestBase(SearchUiTestBase):
    """A UI server started WITH a config file, because the file is the storage."""

    def setUp(self):
        super().setUp()
        self.config_path = self.root / "config.yaml"
        self.write_config()

    def write_config(self, body: str = "") -> None:
        """A small config that points at this test's database, plus a comment to guard.

        The comment is the point of the line-level writer: the file belongs to the person
        and is full of their prose, and a save that dropped it would be a YAML round trip.
        """
        self.config_path.write_text(
            "# my own config\n"
            f'database: "{self.cfg.database.as_posix()}"\n'
            "language: en\n" + body, encoding="utf-8")

    def start_server(self, with_config: bool = True) -> None:
        self.server = ui.build_server(
            self.cfg, self.conn, port=0,
            config_path=str(self.config_path) if with_config else None)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def restart_from_the_file(self) -> None:
        """Stop the server, read `config.yaml` back, start again — a restart, honestly.

        This is the test the feature exists for: a pin that lives only in the running
        process is a pin the next morning does not have.
        """
        waiting.stop_server(self.server, self.thread)
        self.cfg = load_config(str(self.config_path))
        self.start_server()

    # --- requests ------------------------------------------------------

    def slice_(self, name: str = "", extra: str = "") -> dict:
        query = f"?slice={urllib.parse.quote(name)}" if name else "?"
        status, body, ctype = self.get(f"/api/saved-slices{query}{extra}")
        self.assertEqual(status, 200)
        self.assertIn("application/json", ctype)
        return json.loads(body)

    def pinned_names(self) -> list[str]:
        return [s["slice"] for s in self.slice_()["slices"]]

    def saved_names(self) -> list[str]:
        """The names in the FILE, read the way the next start of the tool reads them."""
        return [s.name for s in load_config(str(self.config_path)).features.saved_slices]

    def pin(self, query: str, name: str | None = None) -> tuple[int, dict]:
        body: dict[str, object] = {"query": query}
        if name is not None:
            body["name"] = name
        return self.post("/api/saved-slices/pin", body)

    def unpin(self, name: str) -> tuple[int, dict]:
        return self.post("/api/saved-slices/unpin", {"slice": name})

    def move(self, name: str, delta: int) -> tuple[int, dict]:
        return self.post("/api/saved-slices/move", {"slice": name, "delta": delta})

    def pin_nothing(self) -> None:
        """Start from an empty row — the shipped three are F151's and are not the subject."""
        self.cfg.features = dataclasses.replace(self.cfg.features, saved_slices=())

    def ids(self, data: dict) -> list[int]:
        return [it["file_id"] for it in data["items"]]


class TestAPinnedQueryBecomesASlice(PinTestBase):
    """Brief test 1: the pin appears among the slices and survives a restart."""

    def test_pinning_a_query_puts_it_in_the_row_of_slices(self):
        self.pin_nothing()
        self.add_indexed_photo("a.jpg", unit(1.0))
        self.start_server()
        status, resp = self.pin("mountains", "горы")
        self.assertEqual(status, 200)
        self.assertTrue(resp["ok"])
        self.assertEqual([s["slice"] for s in resp["slices"]], ["горы"])
        self.assertEqual(self.pinned_names(), ["горы"])
        self.assertEqual(self.slice_("горы")["queries"], ["mountains"])

    def test_the_new_pin_goes_to_the_end_of_the_row(self):
        # Where the person who just made it will look for it. The order is theirs to
        # change afterwards; the product does not have an opinion about a list it does
        # not own.
        self.start_server()
        _status, resp = self.pin("mountains")
        self.assertEqual([s["slice"] for s in resp["slices"]],
                         [s.name for s in DEFAULT_SAVED_SLICES] + ["mountains"])

    def test_the_name_defaults_to_the_query_itself(self):
        self.pin_nothing()
        self.start_server()
        _status, resp = self.pin("receipts")
        self.assertEqual([s["slice"] for s in resp["slices"]], ["receipts"])
        _status, resp = self.pin("cars", name="   ")
        self.assertEqual([s["slice"] for s in resp["slices"]], ["receipts", "cars"])

    def test_the_pin_is_still_there_after_the_server_restarts(self):
        self.pin_nothing()
        self.add_indexed_photo("a.jpg", unit(1.0))
        self.start_server()
        self.pin("mountains", "горы")
        self.restart_from_the_file()
        self.assertEqual(self.pinned_names(), ["горы"])
        self.assertEqual(self.slice_("горы")["queries"], ["mountains"])

    def test_a_duplicate_name_is_refused_rather_than_shadowing_the_first(self):
        self.pin_nothing()
        self.start_server()
        self.pin("mountains", "горы")
        status, resp = self.pin("hills", "горы")
        self.assertEqual(status, 400)
        self.assertEqual(resp["reason"], "duplicate")
        self.assertEqual(self.pinned_names(), ["горы"])
        self.assertEqual(self.saved_names(), ["горы"])

    def test_pinning_ranks_nothing_and_loads_no_model(self):
        # A pin saves WORDS. The ranking happens when the slice is opened, exactly as it
        # does for the slices that ship in the config file.
        self.add_indexed_photo("a.jpg", unit(1.0))
        self.start_server()
        self.pin("mountains")
        self.assertEqual(self.encoded, [])

    def test_the_arrows_reorder_the_row_and_the_file_agrees(self):
        self.pin_nothing()
        self.start_server()
        self.pin("mountains", "a")
        self.pin("cars", "b")
        status, resp = self.move("b", -1)
        self.assertEqual(status, 200)
        self.assertEqual([s["slice"] for s in resp["slices"]], ["b", "a"])
        self.assertEqual(self.saved_names(), ["b", "a"])
        # a step off the end is a no-op, not an error: that is what an arrow at the top
        # of a list does
        _status, resp = self.move("b", -1)
        self.assertEqual([s["slice"] for s in resp["slices"]], ["b", "a"])

    def test_moving_or_unpinning_an_unknown_slice_is_a_400(self):
        self.start_server()
        for path, body in (("/api/saved-slices/unpin", {"slice": "nope"}),
                           ("/api/saved-slices/move", {"slice": "nope", "delta": 1}),
                           ("/api/saved-slices/move", {"slice": "children", "delta": 3}),
                           ("/api/saved-slices/move", {"slice": "children"})):
            with self.subTest(path=path, body=body):
                status, resp = self.post(path, body)
                self.assertEqual(status, 400)
                self.assertIn("error", resp)

    def test_a_server_without_a_config_file_still_pins_for_this_session(self):
        # Nothing else can be offered there, and refusing would take the feature away
        # from a session that is otherwise complete.
        self.pin_nothing()
        self.start_server(with_config=False)
        status, resp = self.pin("mountains")
        self.assertEqual(status, 200)
        self.assertEqual([s["slice"] for s in resp["slices"]], ["mountains"])
        self.assertEqual(self.saved_names(),
                         [s.name for s in DEFAULT_SAVED_SLICES])  # the file is untouched


class TestTheStorageIsTheConfigFile(PinTestBase):
    """Brief test 2: written into `config.yaml`, and `reset` does not take it away."""

    def test_the_pin_is_written_into_the_file_with_the_rest_of_it_intact(self):
        self.pin_nothing()
        self.start_server()
        self.pin("mountains", "горы")
        text = self.config_path.read_text(encoding="utf-8")
        self.assertIn("saved_slices:", text)
        self.assertIn("mountains", text)
        self.assertIn("# my own config", text)     # the person's file, not a dump
        self.assertIn("language: en", text)
        self.assertEqual(self.saved_names(), ["горы"])

    def test_resetting_the_index_does_not_take_the_pins_with_it(self):
        """The whole reason the pins are not in the database."""
        self.pin_nothing()
        self.add_indexed_photo("a.jpg", unit(1.0))
        self.start_server()
        self.pin("mountains", "горы")
        status, resp = self.post("/api/process/reset", {})
        self.assertEqual(status, 200)
        self.assertTrue(resp["ok"])
        self.assertEqual(self.slice_()["photos"], 0)   # the index really is gone
        self.assertEqual(self.pinned_names(), ["горы"])
        self.assertEqual(self.saved_names(), ["горы"])

    def test_the_pins_of_the_file_are_what_a_fresh_process_reads(self):
        # The other direction: an edit made by hand reaches the interface, unchanged.
        self.write_config('features:\n  saved_slices:\n    snow: ["a photo of snow"]\n')
        self.cfg = load_config(str(self.config_path))
        self.start_server()
        self.assertEqual(self.pinned_names(), ["snow"])
        self.assertEqual(self.slice_("snow")["queries"], ["a photo of snow"])


class TestUnpinning(PinTestBase):
    """Brief test 3: the slice goes, the photographs stay."""

    def test_unpinning_removes_the_slice_from_the_row_and_the_file(self):
        self.pin_nothing()
        self.start_server()
        self.pin("mountains", "горы")
        self.pin("cars", "машины")
        status, resp = self.unpin("горы")
        self.assertEqual(status, 200)
        self.assertEqual([s["slice"] for s in resp["slices"]], ["машины"])
        self.assertEqual(self.pinned_names(), ["машины"])
        self.assertEqual(self.saved_names(), ["машины"])

    def test_unpinning_touches_no_file_and_no_row(self):
        self.pin_nothing()
        file_id = self.add_indexed_photo("a.jpg", unit(1.0))
        on_disk = sorted(p.name for p in self.src_dir.iterdir())
        self.start_server()
        self.pin("mountains", "горы")
        self.unpin("горы")
        self.assertEqual(sorted(p.name for p in self.src_dir.iterdir()), on_disk)
        self.assertEqual(
            [r[0] for r in self.conn.execute("SELECT id FROM files")], [file_id])
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM move_batches").fetchone()[0], 0)

    def test_unpinning_the_last_slice_leaves_an_empty_row_and_not_the_defaults(self):
        """`saved_slices: {}` is written on purpose: an absent key means "the three that
        ship", so unpinning the last pin would silently restore them."""
        self.pin_nothing()
        self.start_server()
        self.pin("mountains", "горы")
        self.unpin("горы")
        self.assertEqual(self.pinned_names(), [])
        self.assertEqual(self.saved_names(), [])

    def test_a_slice_that_ships_can_be_unpinned_like_any_other(self):
        # These are saved QUERIES and live in the same mapping. The exact slices (people,
        # events, animals, duplicates) are the ones that cannot be unpinned, and they are
        # not in this list at all.
        self.start_server()
        _status, resp = self.unpin("children")
        self.assertEqual([s["slice"] for s in resp["slices"]], ["products", "animals"])

    def test_no_exact_slice_is_reachable_through_this_route(self):
        self.start_server()
        for name in ("person", "event", "animal", "dupes", "people", "portrait"):
            with self.subTest(name=name):
                status, _resp = self.unpin(name)
                self.assertEqual(status, 400)


class TestTheLimitIsSaidOutLoud(PinTestBase):
    """Brief test 4: the thirteenth pin is refused, and the refusal is a sentence."""

    def test_the_default_limit_is_twelve(self):
        self.assertEqual(self.cfg.features.max_pinned_slices, 12)
        self.start_server()
        self.assertEqual(self.slice_()["max_pinned"], 12)

    def test_the_pin_past_the_limit_is_refused_with_the_reason_and_the_number(self):
        self.cfg.features = dataclasses.replace(
            self.cfg.features, saved_slices=(), max_pinned_slices=3)
        self.start_server()
        for i in range(3):
            status, _resp = self.pin(f"query {i}", f"slice {i}")
            self.assertEqual(status, 200)
        status, resp = self.pin("one too many")
        self.assertEqual(status, 400)
        self.assertEqual(resp["reason"], "limit")
        self.assertEqual(resp["max_pinned"], 3)
        self.assertIn("error", resp)
        # and nothing was pinned quietly
        self.assertEqual(len(self.pinned_names()), 3)
        self.assertEqual(len(self.saved_names()), 3)

    def test_unpinning_one_makes_room_again(self):
        self.cfg.features = dataclasses.replace(
            self.cfg.features, saved_slices=(), max_pinned_slices=1)
        self.start_server()
        self.pin("mountains", "горы")
        self.assertEqual(self.pin("cars", "машины")[0], 400)
        self.unpin("горы")
        self.assertEqual(self.pin("cars", "машины")[0], 200)

    def test_a_file_edited_by_hand_past_the_limit_keeps_every_slice(self):
        """The number governs what the INTERFACE adds, not what the file may say."""
        body = "features:\n  max_pinned_slices: 2\n  saved_slices:\n"
        body += "".join(f'    s{i}: ["q{i}"]\n' for i in range(5))
        self.write_config(body)
        self.cfg = load_config(str(self.config_path))
        self.start_server()
        self.assertEqual(len(self.pinned_names()), 5)
        self.assertEqual(self.pin("more")[0], 400)


class TestAnEmptyQueryIsNotASlice(PinTestBase):
    """Brief test 5: there is nothing to pin, and the answer says so."""

    def test_an_empty_query_is_refused(self):
        self.pin_nothing()
        self.start_server()
        for body in ({"query": ""}, {"query": "   "}, {"query": "  ", "name": "горы"},
                     {"name": "горы"}, {"query": 7}, {"query": "x", "name": 7}, []):
            with self.subTest(body=body):
                status, resp = self.post("/api/saved-slices/pin", body)
                self.assertEqual(status, 400)
                self.assertIn("error", resp)
        self.assertEqual(self.pinned_names(), [])
        self.assertEqual(self.encoded, [])

    def test_the_refusal_names_the_reason_the_catalog_has_a_sentence_for(self):
        self.start_server()
        _status, resp = self.post("/api/saved-slices/pin", {"query": "   "})
        self.assertEqual(resp["reason"], "empty")
        for lang in ("ru", "en", "ja"):
            self.assertTrue(ui._t("pin_error_empty", lang).strip())

    def test_the_client_does_not_offer_the_button_without_a_result(self):
        # The button appears when there is something to save: a query with nothing under
        # it is not a slice yet.
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        # F189 gave the condition a name (`some`) because the same count now decides the
        # album row and the offer of the other answer as well. The rule is unchanged.
        self.assertIn("var some = (data.items || []).length;", html)
        self.assertIn('showPinButton(some ? data.query : "");', html)
        self.assertIn('id="slice-pin-btn"', html)
        button = [ln for ln in html.splitlines() if 'id="slice-pin-btn"' in ln]
        self.assertEqual(len(button), 1)
        self.assertIn('style="display:none"', button[0])


class TestAPinnedSliceAnswersLikeTheSearch(PinTestBase):
    """Brief test 6: the pin is the same engine, asked the same words."""

    def test_the_pinned_slice_returns_what_the_search_returned(self):
        self.pin_nothing()
        self.vectors.update({"mountains": unit(0.0, 1.0)})
        near = self.add_indexed_photo("near.jpg", unit(0.0, 1.0))
        far = self.add_indexed_photo("far.jpg", unit(1.0, 0.0))
        self.start_server()
        searched = self.search("mountains")
        self.pin("mountains", "горы")
        pinned = self.slice_("горы")
        self.assertEqual(self.ids(searched), [near, far])
        self.assertEqual(self.ids(pinned), self.ids(searched))
        self.assertEqual([it["score"] for it in pinned["items"]],
                         [it["score"] for it in searched["items"]])
        self.assertEqual(pinned["total"], searched["total"])

    def test_the_pinned_slice_is_the_estimate_it_was_as_a_search(self):
        # The caption of F151 covers it unchanged — a ranking, no threshold, and the
        # depth button as the main control. A pin does not turn a ranking into a mark.
        self.add_indexed_photo("a.jpg", unit(1.0))
        self.start_server()
        self.pin("mountains", "горы")
        data = self.slice_("горы")
        self.assertTrue(data["approximate"])
        self.assertIn("score", data["items"][0])

    def test_an_unfilled_index_gives_the_reason_and_not_an_empty_pin(self):
        # The rule matters MORE for a pin than for a typed query: nobody typed anything
        # just now, so an empty list would read as a fact about the archive.
        self.pin_nothing()
        self.add_photo_file("a.jpg")
        self.start_server()
        self.pin("mountains", "горы")
        data = self.slice_("горы")
        self.assertEqual(data["state"], "empty")
        self.assertFalse(data["available"])
        self.assertEqual(data["items"], [])


class TestTheActionsOfAPinnedSlice(PinTestBase):
    """Brief test 7: the album works, and `move_batches.mode` is meaningful."""

    def album(self, selector: str, **extra) -> tuple[int, dict]:
        body: dict[str, object] = {"kind": "query", "selector": selector,
                                   "mode": "link", "apply": False}
        body.update(extra)
        return self.post("/api/album", body)

    def test_the_album_gathers_what_the_pinned_slice_shows(self):
        self.pin_nothing()
        self.vectors.update({"mountains": unit(0.0, 1.0)})
        self.add_indexed_photo("near.jpg", unit(0.0, 1.0))
        self.add_indexed_photo("far.jpg", unit(1.0, 0.0))
        self.start_server()
        self.pin("mountains", "горы")
        status, resp = self.album("mountains", name="горы")
        self.assertEqual(status, 200)
        self.assertEqual(resp["kind"], "query")
        self.assertEqual(resp["count"], len(self.slice_("горы")["items"]))
        self.assertEqual(resp["album_name"], "горы")

    def test_applying_it_journals_under_a_mode_that_says_what_it_was(self):
        self.pin_nothing()
        self.add_indexed_photo("near.jpg", unit(1.0))
        self.start_server()
        self.pin("mountains", "горы")
        status, resp = self.album("mountains", name="горы", apply=True,
                                  dest=str(self.root / "albums"))
        self.assertEqual(status, 200)
        self.assertTrue(resp["applied"])
        self.assertEqual(resp["transferred"], 1)
        modes = [r[0] for r in self.conn.execute("SELECT mode FROM move_batches")]
        self.assertEqual(modes, ["album_query"])

    def test_the_three_album_modes_are_all_offered(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        album = html[html.index("function renderQuerySliceAlbum"):
                     html.index("function renderQuerySliceActions")]
        # F193: a pin is an ordinary slice, so its row is the shared one — the modes, the
        # destination and the folder name come from there rather than from a copy here.
        self.assertIn('box: "query-album"', album)
        # F189: a pinned NAME gathers the person's album instead — same row, same modes,
        # and the kind follows what the slice actually answered.
        self.assertIn('kind: data.person ? "person" : "query"', album)
        self.assertIn("selector: data.person || one", album)
        self.assertIn('id="query-album"', html)
        row = html.split("function renderAlbumRow", 1)[1].split(
            "function renderSliceAlbumControls", 1)[0]
        self.assertIn("albumModeSelect()", row)          # link / copy / move
        self.assertIn("appendAlbumDestControls(box)", row)

    def test_a_slice_of_several_phrases_says_why_it_has_no_album(self):
        # The album gathers a single wording and this ranking is an average of three, so
        # a button here would gather a different list under the slice's own name.
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn("I18N.pin_album_one_query", html)
        for lang in ("ru", "en", "ja"):
            with self.subTest(lang=lang):
                self.assertIn("features.saved_slices",
                              ui._UI_STRINGS["pin_album_one_query"][lang])


class TestThePanelOfAPinnedSlice(PinTestBase):
    """Requirement three: a pinned slice is indistinguishable from a built-in one."""

    def test_the_panel_carries_the_unpin_and_the_two_arrows(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        panel = html[html.index('<div id="tab-query"'):html.index('<div id="tab-person"')]
        for element in ('id="query-unpin-btn"', 'id="query-left-btn"',
                        'id="query-right-btn"', 'id="query-album"', 'id="query-grid"'):
            with self.subTest(element=element):
                self.assertIn(element, panel)

    def test_unpinning_asks_first_and_says_the_files_stay(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        unpin = html[html.index(
            'document.getElementById("query-unpin-btn").addEventListener'):]
        unpin = unpin[:unpin.index("});")]
        self.assertIn("window.confirm(fmt(I18N.pin_unpin_confirm", unpin)
        for lang, promise in (("ru", "файлы"), ("en", "files"), ("ja", "ファイル")):
            with self.subTest(lang=lang):
                self.assertIn(promise, ui._t("pin_unpin_confirm", lang))

    def test_the_three_writes_are_dead_while_something_runs_and_say_so(self):
        # F145's rule: all three write `config.yaml`, the server refuses a config write
        # mid-run, and a control that is alive for an action that cannot happen teaches
        # that the interface lies.
        for route in ("/api/saved-slices/pin", "/api/saved-slices/unpin",
                      "/api/saved-slices/move"):
            with self.subTest(route=route):
                self.assertIn(route, ui.BUSY_REFUSED_ROUTES)
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn('"slice-pin-btn",', html)
        self.assertIn("registerBusyRefresh(refreshQuerySliceControls);", html)
        refresh = html[html.index("function refreshQuerySliceControls"):]
        refresh = refresh[:refresh.index("registerBusyRefresh")]
        for control in ("query-left-btn", "query-right-btn", "query-unpin-btn"):
            with self.subTest(control=control):
                self.assertIn(control, refresh)
        self.assertEqual(refresh.count("busy ||"), 3)

    def test_the_pin_row_is_still_built_from_the_route(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn('pins.push({ key: "query:" + s.slice', html)
        self.assertIn("savedSlicesMax = data.max_pinned || 0;", html)


class BuiltInEmptinessTestBase(UiServerTestBase):
    """The F156 half about the slices a person does NOT pin."""

    def visibility(self) -> dict:
        _status, body, _ctype = self.get("/api/tabs/visibility")
        return json.loads(body)

    def reasons(self) -> dict:
        return self.visibility()["reasons"]

    def add_face(self, file_id: int, *, cluster: bool = False) -> None:
        cluster_id = None
        if cluster:
            cluster_id = self.conn.execute(
                "INSERT INTO face_clusters (label) VALUES ('Alice')").lastrowid
        self.conn.execute(
            """INSERT INTO faces (file_id, bbox, embedding, cluster_id)
               VALUES (?, '[0,0,10,10]', ?, ?)""", (file_id, b"e", cluster_id))
        self.conn.commit()

    def measure_animals(self, file_id: int, score: float = 0.01) -> None:
        self.conn.execute(
            """INSERT INTO frame_quality (file_id, pet_score, source, updated_at)
               VALUES (?, ?, 'clip', '2026-01-01')""", (file_id, score))
        self.conn.commit()

    def add_event(self, file_id: int) -> None:
        event_id = self.conn.execute(
            """INSERT INTO events (started_at, ended_at, name, name_is_manual, origin)
               VALUES ('2022-05-01T10:00:00', '2022-05-01T10:00:00', 'e', 0, 'auto')"""
        ).lastrowid
        self.conn.execute("INSERT INTO event_files (event_id, file_id) VALUES (?, ?)",
                          (event_id, file_id))
        self.conn.commit()


class TestAnEmptyBuiltInSliceExplainsItself(BuiltInEmptinessTestBase):
    """Brief test 5a: two different answers, never a bare zero."""

    def test_a_stage_that_never_ran_answers_not_run(self):
        self.add_photo_file("a.jpg")
        self.start_server()
        self.assertEqual(self.reasons(),
                         {"person": "not_run", "event": "not_run", "animal": "not_run"})

    def test_a_stage_that_ran_and_found_nothing_answers_none_found(self):
        file_id, _p, _c = self.add_photo_file("a.jpg")
        self.add_face(file_id)              # the detector looked and clustered nothing
        self.measure_animals(file_id)       # the pet score was measured and is low
        self.add_event(file_id)             # events were built
        self.start_server()
        # events are not empty here, so that slice's reason is None — the other two ran
        self.assertEqual(self.reasons()["person"], "none_found")
        self.assertEqual(self.reasons()["animal"], "none_found")

    def test_the_two_answers_are_two(self):
        """The point of the whole section: "nobody looked" and "there are none" are
        different sentences, and only one of them is a fact about the photographs."""
        file_id, _p, _c = self.add_photo_file("a.jpg")
        self.start_server()
        self.assertEqual(self.reasons()["animal"], "not_run")
        self.measure_animals(file_id)
        self.assertEqual(self.reasons()["animal"], "none_found")
        self.assertNotEqual(self.reasons()["animal"], "not_run")

    def test_a_slice_that_holds_something_has_no_reason_to_give(self):
        file_id, _p, _c = self.add_photo_file("a.jpg")
        self.add_face(file_id, cluster=True)
        self.add_event(file_id)
        self.start_server()
        self.assertIsNone(self.reasons()["person"])
        self.assertIsNone(self.reasons()["event"])

    def test_an_uncomputed_slice_is_shown_so_that_it_can_say_so(self):
        # The F152 rule, extended: a pin that hides itself never gets to explain itself.
        self.add_photo_file("a.jpg")
        self.start_server()
        data = self.visibility()
        self.assertTrue(data["person"])
        self.assertTrue(data["animal"])
        self.assertTrue(data["event"])

    def test_a_computed_and_empty_slice_still_hides(self):
        # There the zero IS the fact and the collection has already stated it; a pin over
        # an empty page teaches nothing.
        file_id, _p, _c = self.add_photo_file("a.jpg")
        self.add_face(file_id)
        self.measure_animals(file_id)
        self.start_server()
        data = self.visibility()
        self.assertFalse(data["person"])
        self.assertFalse(data["animal"])

    def test_nothing_at_all_in_the_index_is_no_slice_and_no_sentence(self):
        self.start_server()
        data = self.visibility()
        self.assertFalse(data["person"])
        self.assertFalse(data["event"])
        self.assertFalse(data["animal"])

    def test_animals_switched_off_read_as_uncomputed_and_not_as_absent(self):
        """`features.pets: false` is the run screen's own way of removing the slice —
        the section's argument for not adding a second one."""
        file_id, _p, _c = self.add_photo_file("a.jpg")
        self.conn.execute(
            """INSERT INTO frame_quality (file_id, sharpness, source, updated_at)
               VALUES (?, 100.0, 'clip', '2026-01-01')""", (file_id,))
        self.conn.commit()
        self.start_server()
        self.assertEqual(self.reasons()["animal"], "not_run")

    def test_a_collection_with_no_dates_has_nothing_to_group(self):
        self.conn.execute(
            """INSERT INTO files (path, size, mtime, ext, media_type, hash, hash_algo,
                   indexed_at)
               VALUES ('/tmp/x.jpg', 1, 0, 'jpg', 'photo', 'h', 'sha256', '2026-01-01')""")
        self.conn.commit()
        self.start_server()
        self.assertEqual(self.reasons()["event"], "none_found")

    def test_the_client_links_to_the_run_screen_for_the_uncomputed_one_only(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        helper = html[html.index("function sliceEmptyState"):]
        helper = helper[:helper.index("function buildSlicePins")]
        self.assertIn('sliceReasons[key] !== "not_run"', helper)
        self.assertIn("I18N.slice_not_computed", helper)
        self.assertIn("I18N.slice_goto_process", helper)
        self.assertIn('activateTab("overview")', helper)
        # and each of the three panels goes through it
        self.assertIn('sliceEmptyState("person", I18N.no_clusters)', html)
        self.assertIn('sliceEmptyState("event", I18N.no_events)', html)
        self.assertIn('sliceEmptyState("animal", I18N.animals_empty)', html)
        self.assertIn("sliceReasons = data.reasons || {};", html)

    def test_the_two_sentences_are_different_in_every_language(self):
        for lang in ("ru", "en", "ja"):
            with self.subTest(lang=lang):
                not_run = ui._t("slice_not_computed", lang)
                for found_nothing in ("no_clusters", "no_events", "animals_empty"):
                    self.assertNotEqual(not_run, ui._t(found_nothing, lang))


class TestTheConfigWriter(unittest.TestCase):
    """`config.save_saved_slices` — a line-level edit of somebody's own file."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "config.yaml"

    def names(self) -> list[str]:
        return [s.name for s in load_config(str(self.path)).features.saved_slices]

    def test_it_round_trips_through_the_file(self):
        self.path.write_text("language: en\n", encoding="utf-8")
        slices = (SavedSlice("горы", ("mountains",)),
                  SavedSlice("receipts: paper", ("a photo of a receipt",)))
        save_saved_slices(self.path, slices)
        self.assertEqual(load_config(str(self.path)).features.saved_slices, slices)

    def test_the_comments_and_the_other_settings_survive(self):
        self.path.write_text(
            "# a file somebody wrote\n"
            "language: ru\n"
            "features:\n"
            "  # why these\n"
            "  saved_slices:\n"
            "    old: [\"a photo of something\"]\n"
            "  search_page: 42\n", encoding="utf-8")
        save_saved_slices(self.path, [SavedSlice("new", ("snow",))])
        text = self.path.read_text(encoding="utf-8")
        self.assertIn("# a file somebody wrote", text)
        self.assertIn("# why these", text)
        self.assertNotIn("old:", text)
        cfg = load_config(str(self.path))
        self.assertEqual(self.names(), ["new"])
        self.assertEqual(cfg.features.search_page, 42)
        self.assertEqual(cfg.language, "ru")

    def test_an_empty_list_is_written_as_an_empty_mapping(self):
        # An absent key means "the three that ship" — unpinning the last pin must not
        # bring them back.
        self.path.write_text("language: en\n", encoding="utf-8")
        save_saved_slices(self.path, [])
        self.assertIn("saved_slices: {}", self.path.read_text(encoding="utf-8"))
        self.assertEqual(load_config(str(self.path)).features.saved_slices, ())

    def test_a_file_without_the_section_grows_one(self):
        self.path.write_text("language: en\n", encoding="utf-8")
        save_saved_slices(self.path, [SavedSlice("snow", ("a photo of snow",))])
        self.assertEqual(self.names(), ["snow"])
        self.assertIn("features:", self.path.read_text(encoding="utf-8"))

    def test_a_missing_file_is_created(self):
        save_saved_slices(self.path, [SavedSlice("snow", ("a photo of snow",))])
        self.assertEqual(self.names(), ["snow"])

    def test_the_indentation_of_the_file_is_kept(self):
        self.path.write_text(
            "features:\n"
            "    search_page: 10\n"
            "    saved_slices:\n"
            "        old: [\"x\"]\n", encoding="utf-8")
        save_saved_slices(self.path, [SavedSlice("new", ("snow",))])
        text = self.path.read_text(encoding="utf-8")
        self.assertIn("    saved_slices:\n", text)
        self.assertEqual(self.names(), ["new"])
        self.assertEqual(load_config(str(self.path)).features.search_page, 10)

    def test_the_example_config_survives_being_rewritten(self):
        """The file every user starts from, and the one with thirty lines of comment
        above the key."""
        example = (_ROOT / "config.example.yaml").read_text(encoding="utf-8")
        self.path.write_text(example, encoding="utf-8")
        before = load_config(str(self.path))
        save_saved_slices(self.path,
                          [*DEFAULT_SAVED_SLICES, SavedSlice("горы", ("mountains",))])
        after = load_config(str(self.path))
        self.assertEqual(self.names(),
                         [s.name for s in DEFAULT_SAVED_SLICES] + ["горы"])
        self.assertEqual(after.features.search_page, before.features.search_page)
        self.assertEqual(after.features.group_photo_faces,
                         before.features.group_photo_faces)
        self.assertIn("# F156: THE WEB APP WRITES HERE TOO.",
                      self.path.read_text(encoding="utf-8"))


class TestTheSettingAndTheCatalog(unittest.TestCase):
    def test_the_example_config_documents_the_limit(self):
        example = (_ROOT / "config.example.yaml").read_text(encoding="utf-8")
        self.assertIn("max_pinned_slices: 12", example)
        cfg = load_config(str(_ROOT / "config.example.yaml"))
        self.assertEqual(cfg.features.max_pinned_slices, 12)

    def test_a_broken_limit_falls_back_rather_than_pinning_nothing(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            path = Path(tmp) / "config.yaml"
            for body in ("nope", "0", "-3"):
                with self.subTest(body=body):
                    path.write_text(f"features:\n  max_pinned_slices: {body}\n",
                                    encoding="utf-8")
                    self.assertEqual(
                        load_config(str(path)).features.max_pinned_slices, 12)

    def test_every_new_string_is_translated_three_ways(self):
        keys = ("pin_slice_button", "pin_slice_prompt", "pin_slice_language_warning",
                "pin_slice_done", "pin_error_empty", "pin_error_duplicate",
                "pin_error_limit", "pin_error_generic", "pin_unpin_button",
                "pin_unpin_confirm", "pin_move_left", "pin_move_right",
                "pin_album_one_query", "slice_not_computed", "slice_goto_process")
        for key in keys:
            with self.subTest(key=key):
                entry = ui._UI_STRINGS[key]
                self.assertEqual(set(entry), {"ru", "en", "ja"})
                for lang in ("ru", "en", "ja"):
                    self.assertTrue(entry[lang].strip())

    def test_the_captions_carry_their_placeholders(self):
        for lang in ("ru", "en", "ja"):
            with self.subTest(lang=lang):
                self.assertIn("{query}", ui._UI_STRINGS["pin_slice_prompt"][lang])
                self.assertIn("{name}", ui._UI_STRINGS["pin_slice_done"][lang])
                self.assertIn("{name}", ui._UI_STRINGS["pin_unpin_confirm"][lang])
                self.assertIn("{max}", ui._UI_STRINGS["pin_error_limit"][lang])

    def test_the_limit_sentence_names_the_setting_a_reader_can_raise(self):
        for lang in ("ru", "en", "ja"):
            with self.subTest(lang=lang):
                self.assertIn("features.max_pinned_slices",
                              ui._UI_STRINGS["pin_error_limit"][lang])

    def test_the_language_warning_is_given_before_the_pin_and_not_after(self):
        for lang in ("ru", "en", "ja"):
            with self.subTest(lang=lang):
                self.assertTrue(
                    ui._UI_STRINGS["pin_slice_language_warning"][lang].strip())
        html = ui._render_index_html("ru")
        prompt = html[html.index('document.getElementById("slice-pin-btn")'):]
        prompt = prompt[:prompt.index("window.prompt")]
        self.assertIn("isAscii(query)", prompt)
        self.assertIn("I18N.pin_slice_language_warning", prompt)


if __name__ == "__main__":
    unittest.main()
