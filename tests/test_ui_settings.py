"""F104: the settings column of the "Cities" tab — GET/POST /api/settings.

Until now every one of these knobs lived in config.yaml only, which means it was
switched with a text editor and a restart of `sorta ui`. A toggle that needs a restart
is not a toggle, so what this pins is not "the value was written" but the whole chain:
the RUNNING config changes, the file changes without losing anything else in it, and a
change that must not happen right now is refused instead of half-applied.
"""
from __future__ import annotations

import dataclasses
import json
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

import yaml

from sorta import ui
from sorta.config import load_config

from tests.test_ui_process import ProcessTestBase, _poll_until

_F104_SETTINGS_KEYS = (
    "settings_title", "settings_hint",
    "settings_vlm_enabled_label", "settings_vlm_enabled_hint",
    "settings_vlm_model_label",
    "settings_vlm_workers_label", "settings_vlm_workers_hint",
    "settings_vlm_max_edge_label", "settings_vlm_max_edge_hint",
    "settings_folders_title", "settings_folder_lang_hint",
    "settings_saved", "settings_error_prefix", "settings_busy",
)


class SettingsTestBase(ProcessTestBase):
    """A server started with a real config.yaml, so persistence can be verified."""

    def setUp(self):
        super().setUp()
        self.config_path = self.root / "config.yaml"
        self.config_path.write_text(
            "# a note of my own\n"
            "language: en\n"
            "vlm:\n"
            "  enabled: false\n"
            "  model: Qwen/Qwen2.5-VL-3B-Instruct\n"
            "  workers: 2\n"
            "  max_edge: 896\n"
            "naming:\n"
            "  provider: template\n",
            encoding="utf-8")
        # The running config is the one that file produces — which is what `sorta ui`
        # hands the server (cli loads the config and passes its path alongside).
        loaded = load_config(self.config_path)
        self.cfg.vlm, self.cfg.naming, self.cfg.raw = (
            loaded.vlm, loaded.naming, loaded.raw)

    def start_server(self, config_path: Path | bool | None = None) -> None:
        """config_path=False — no file at all (the write is disabled server-side)."""
        path = self.config_path if config_path is None else (config_path or None)
        self.server = ui.build_server(self.cfg, self.conn, port=0, config_path=path)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def settings(self) -> dict:
        status, body, ctype = self.get("/api/settings")
        self.assertEqual(status, 200)
        self.assertIn("application/json", ctype)
        return json.loads(body)

    def saved(self) -> dict:
        return yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}

    def post_raw(self, path: str, payload: object) -> tuple[int, dict]:
        """POST an arbitrary JSON body (ProcessTestBase.post wants a dict)."""
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(f"{self.base_url}{path}", data=data,
                                     headers={"Content-Type": "application/json"},
                                     method="POST")
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())


class TestReadSettings(SettingsTestBase):
    def test_get_returns_the_values_of_the_running_config(self):
        self.cfg.vlm = dataclasses.replace(
            self.cfg.vlm, enabled=True, model="some/other-vlm", workers=3, max_edge=640)
        self.start_server()
        self.assertEqual(self.settings(), {
            "vlm.enabled": True,
            "vlm.model": "some/other-vlm",
            "vlm.workers": 3,
            "vlm.max_edge": 640,
        })

    def test_get_reports_a_usable_value_for_every_knob(self):
        """The column renders straight from this answer — a missing or nonsensical
        value would show as an empty field the user could only fix by typing."""
        self.start_server()
        data = self.settings()
        self.assertEqual(set(data), set(ui._SETTINGS_SPEC))
        self.assertIs(data["vlm.enabled"], False)
        self.assertTrue(data["vlm.model"])
        self.assertGreaterEqual(data["vlm.workers"], 1)
        self.assertGreater(data["vlm.max_edge"], 0)


class TestWriteSettings(SettingsTestBase):
    def test_a_toggle_changes_the_running_config_and_the_file(self):
        self.start_server()
        status, resp = self.post_raw("/api/settings", {"vlm.enabled": True})
        self.assertEqual(status, 200)
        self.assertTrue(resp["ok"])
        self.assertIs(resp["settings"]["vlm.enabled"], True)
        # in memory — the run that starts next reads this, no restart involved
        self.assertIs(self.cfg.vlm.enabled, True)
        # F102: the field the junk stage actually consults is held equal to it
        self.assertIs(self.cfg.naming.vlm_enabled, True)
        # and on disk
        self.assertIs(self.saved()["vlm"]["enabled"], True)

    def test_the_rest_of_the_file_is_untouched(self):
        self.start_server()
        self.post_raw("/api/settings", {"vlm.max_edge": 512})
        text = self.config_path.read_text(encoding="utf-8")
        self.assertIn("# a note of my own", text)
        self.assertEqual(self.saved(), {
            "language": "en",
            "vlm": {"enabled": False, "model": "Qwen/Qwen2.5-VL-3B-Instruct",
                    "workers": 2, "max_edge": 512},
            "naming": {"provider": "template"},
        })

    def test_several_keys_in_one_body(self):
        self.start_server()
        status, _resp = self.post_raw(
            "/api/settings", {"vlm.workers": 4, "vlm.model": "some/other-vlm"})
        self.assertEqual(status, 200)
        self.assertEqual(self.cfg.vlm.workers, 4)
        self.assertEqual(self.cfg.vlm.model, "some/other-vlm")
        self.assertEqual(self.saved()["vlm"]["workers"], 4)
        self.assertEqual(self.saved()["vlm"]["model"], "some/other-vlm")

    def test_the_new_value_is_what_a_reload_of_the_config_would_give(self):
        """"Saved" has to mean the next `sorta index` sees it too, not just this tab."""
        self.start_server()
        self.post_raw("/api/settings", {"vlm.enabled": True, "vlm.max_edge": 640})
        reloaded = load_config(self.config_path)
        self.assertIs(reloaded.vlm.enabled, True)
        self.assertEqual(reloaded.vlm.max_edge, 640)

    def test_the_process_tab_default_follows_the_toggle(self):
        """The "Deep analysis (VLM)" checkbox is initialized from the same field — a
        toggle that did not move it would be a setting saved and not applied."""
        self.start_server()
        _s, body, _c = self.get("/api/process/defaults")
        self.assertIs(json.loads(body)["deep"], False)
        self.post_raw("/api/settings", {"vlm.enabled": True})
        _s2, body2, _c2 = self.get("/api/process/defaults")
        self.assertIs(json.loads(body2)["deep"], True)

    def test_without_a_config_path_the_change_is_memory_only(self):
        """`sorta ui` can be handed no config file at all (the CLI decides). The
        running config still moves — it just has nowhere to be remembered."""
        self.start_server(config_path=False)
        status, _resp = self.post_raw("/api/settings", {"vlm.workers": 5})
        self.assertEqual(status, 200)
        self.assertEqual(self.cfg.vlm.workers, 5)
        self.assertEqual(self.saved()["vlm"]["workers"], 2)  # the file never moved


class TestRejectedValues(SettingsTestBase):
    def test_garbage_is_400_and_the_file_is_not_touched(self):
        self.start_server()
        before = self.config_path.read_text(encoding="utf-8")
        for payload in (
            {"vlm.workers": "many"},        # a string where a number belongs
            {"vlm.workers": -1},            # negative
            {"vlm.workers": 0},             # zero threads is not a pool
            {"vlm.workers": 1.5},           # not a whole number
            {"vlm.workers": True},          # bool is an int in Python, not here
            {"vlm.max_edge": 0},
            {"vlm.max_edge": 99999},        # a typo that costs the whole VRAM budget
            {"vlm.enabled": "true"},        # the string, not the boolean
            {"vlm.model": ""},              # an empty model name
            {"vlm.model": 5},
            {"sort.exclude_dirs": ["/tmp"]},  # a key this endpoint does not own
            {},                             # an "ok" that changed nothing
            [],
            "vlm.enabled",
        ):
            with self.subTest(payload=payload):
                status, resp = self.post_raw("/api/settings", payload)
                self.assertEqual(status, 400)
                self.assertIn("error", resp)
        self.assertEqual(self.config_path.read_text(encoding="utf-8"), before)
        self.assertIs(self.cfg.vlm.enabled, False)
        self.assertEqual(self.cfg.vlm.workers, 2)

    def test_one_bad_key_rejects_the_whole_body(self):
        """A half-applied save leaves the form and the tool disagreeing about which
        half of it is real."""
        self.start_server()
        status, _resp = self.post_raw(
            "/api/settings", {"vlm.enabled": True, "vlm.workers": -3})
        self.assertEqual(status, 400)
        self.assertIs(self.cfg.vlm.enabled, False)
        self.assertIs(self.saved()["vlm"]["enabled"], False)

    def test_the_bounds_of_the_spec_are_the_ones_the_form_shows(self):
        """The number inputs carry min/max attributes; if they drifted from the
        server's range the user would be refused by a form that offered the value."""
        html = ui._render_index_html("en")
        spec = ui._SETTINGS_SPEC
        self.assertIn(f'id="setting-vlm-workers" min="{spec["vlm.workers"].minimum}" '
                      f'max="{spec["vlm.workers"].maximum}"', html)
        self.assertIn(f'id="setting-vlm-max-edge" min="{spec["vlm.max_edge"].minimum}" '
                      f'max="{spec["vlm.max_edge"].maximum}"', html)


class TestRefusedWhileBusy(SettingsTestBase):
    def test_409_while_the_pipeline_runs_and_nothing_is_written(self):
        """Swapping the model or the frame size in the middle of a classification is
        not a setting but an accident: the run would then be doing neither what the
        file says nor what the user saw."""
        block = threading.Event()
        self.addCleanup(block.set)
        self.patch_fast_stages(block_stage="geo", block_event=block)
        self.start_server()
        before = self.config_path.read_text(encoding="utf-8")
        start_status, _resp = self.post("/api/process",
                                        {"source_dir": str(self.src_dir)})
        self.assertEqual(start_status, 200)
        _poll_until(lambda: json.loads(self.get("/api/process/status")[1]),
                    lambda d: d["running"] and "geo" in self.calls)

        status, resp = self.post_raw("/api/settings", {"vlm.enabled": True})
        self.assertEqual(status, 409)
        self.assertEqual(resp["error"], "already running")
        self.assertEqual(self.config_path.read_text(encoding="utf-8"), before)
        self.assertIs(self.cfg.vlm.enabled, False)

        block.set()
        _poll_until(lambda: json.loads(self.get("/api/process/status")[1]),
                    lambda d: d["finished"])

    def test_the_save_works_again_once_the_run_is_over(self):
        self.patch_fast_stages()
        self.start_server()
        self.post("/api/process", {"source_dir": str(self.src_dir)})
        _poll_until(lambda: json.loads(self.get("/api/process/status")[1]),
                    lambda d: d["finished"])
        status, _resp = self.post_raw("/api/settings", {"vlm.enabled": True})
        self.assertEqual(status, 200)
        self.assertIs(self.saved()["vlm"]["enabled"], True)

    def test_a_sort_in_flight_also_refuses(self):
        state = ui._SortState()
        self.assertTrue(state.try_start())
        with mock.patch.object(ui, "_SortState", return_value=state):
            self.start_server()
            status, resp = self.post_raw("/api/settings", {"vlm.workers": 3})
        self.assertEqual(status, 409)
        self.assertEqual(resp["error"], "already running")
        self.assertEqual(self.cfg.vlm.workers, 2)

    def test_an_undo_in_flight_also_refuses(self):
        state = ui._UndoState()
        self.assertTrue(state.try_start())
        with mock.patch.object(ui, "_UndoState", return_value=state):
            self.start_server()
            status, resp = self.post_raw("/api/settings", {"vlm.workers": 3})
        self.assertEqual(status, 409)
        self.assertEqual(self.cfg.vlm.workers, 2)


class TestSettingsMarkup(SettingsTestBase):
    def setUp(self):
        super().setUp()
        self.html = ui._render_index_html("en")

    def test_the_column_holds_a_control_per_knob(self):
        for control in ("setting-vlm-enabled", "setting-vlm-model",
                        "setting-vlm-workers", "setting-vlm-max-edge"):
            self.assertIn(f'id="{control}"', self.html)
        self.assertIn('class="city-side"', self.html)
        self.assertIn("/api/settings", self.html)

    def test_every_function_has_its_own_toggle(self):
        """Explicitly asked for: no single "smart mode" switch that means four things
        at once — the deep tier is one checkbox, the model, the threads and the frame
        size are their own fields."""
        self.assertEqual(len(ui._SETTINGS_SPEC), 4)
        self.assertEqual(self.html.count('id="setting-vlm-enabled"'), 1)

    def test_the_folder_language_moved_into_the_column(self):
        """It was the odd one out in the action row: a setting standing next to the
        button that moves the collection."""
        side = self.html.split('class="city-side"', 1)[1].split("</aside>", 1)[0]
        self.assertIn('id="folder-lang-select"', side)
        row = self.html.split('class="sort-controls"', 1)[1].split("</div>", 1)[0]
        self.assertNotIn("folder-lang-select", row)

    def test_no_external_resources_added(self):
        self.assertNotIn("http://", self.html)
        self.assertNotIn("https://", self.html)
        self.assertNotIn("<link", self.html)


class TestSettingsStringsAreTranslated(unittest.TestCase):
    def test_every_new_string_exists_in_all_three_languages(self):
        for key in _F104_SETTINGS_KEYS:
            with self.subTest(key=key):
                entry = ui._UI_STRINGS[key]
                self.assertEqual(set(entry), {"ru", "en", "ja"})
                for lang, value in entry.items():
                    self.assertTrue(value.strip(), f"{key}/{lang} is empty")


if __name__ == "__main__":
    unittest.main()
