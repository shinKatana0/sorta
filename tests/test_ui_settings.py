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
import os
import re
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

import yaml

from sorta import imaging, ui
from sorta.config import load_config

from tests.test_ui_process import ProcessTestBase, _poll_until

_F104_SETTINGS_KEYS = (
    "settings_title", "settings_hint",
    # F138: the deep-tier toggle left this column for the run screen (with its price),
    # and its two strings went with it — what stays here is what costs a run nothing.
    "settings_costs_moved_hint",
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
        # F117: saving an `imaging:` key APPLIES it by setting an environment variable,
        # and a variable set by one test is still set for the next one — the preview
        # cache would then run with a ceiling nobody in that test asked for. Snapshot
        # and restore around every case in this file, not just the ones that write it:
        # the leak is invisible where it is caused and only shows up somewhere else.
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
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
            "vlm.model": "some/other-vlm",
            "vlm.workers": 3,
            "vlm.max_edge": 640,
            # F138: `vlm.enabled` and `features.pets` are not answered here any more —
            # they cost a run time, so they are priced on the run screen and this column
            # is not their second home. (The two that moved with them, `vlm.quality` and
            # `vlm.quality_scope`, are retired outright since F186.)
            "features.pet_threshold": 0.7,
            "features.sharpness_max_edge": 512,
            "features.sharpness_band_min": 30.0,
            "features.sharpness_band_max": 300.0,
            "features.subject_score_min": 0.9,
            # F117: read from the environment rather than from cfg — the ceiling has no
            # dataclass field, and 0 (no ceiling) is its default.
            "imaging.preview_cache_max_gb": 0,
        })

    def test_get_reports_a_usable_value_for_every_knob(self):
        """The column renders straight from this answer — a missing or nonsensical
        value would show as an empty field the user could only fix by typing."""
        self.start_server()
        data = self.settings()
        self.assertEqual(set(data), set(ui._SETTINGS_SPEC))
        self.assertTrue(data["vlm.model"])
        self.assertGreaterEqual(data["vlm.workers"], 1)
        self.assertGreater(data["vlm.max_edge"], 0)


class TestWriteSettings(SettingsTestBase):
    def test_a_toggle_changes_the_running_config_and_the_file(self):
        self.start_server()
        status, resp = self.post_raw("/api/settings", {"vlm.model": "some/other-vlm"})
        self.assertEqual(status, 200)
        self.assertTrue(resp["ok"])
        self.assertEqual(resp["settings"]["vlm.model"], "some/other-vlm")
        # in memory — the run that starts next reads this, no restart involved
        self.assertEqual(self.cfg.vlm.model, "some/other-vlm")
        # F102: the field the junk stage actually consults is held equal to it
        self.assertEqual(self.cfg.naming.classify_vlm_model, "some/other-vlm")
        # and on disk
        self.assertEqual(self.saved()["vlm"]["model"], "some/other-vlm")

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

    def test_the_preview_ceiling_applies_to_the_environment_and_the_file(self):
        """F117: the one key with no dataclass field behind it.

        `imaging:` is applied to the environment (imaging.py is a leaf module that pool
        workers call with a path and nothing else), so "applied" means the variable is
        set — and a saved setting that only reached the file would be a ceiling nothing
        enforces until the next restart.
        """
        self.start_server()
        status, resp = self.post_raw(
            "/api/settings", {"imaging.preview_cache_max_gb": 40})
        self.assertEqual(status, 200)
        self.assertEqual(resp["settings"]["imaging.preview_cache_max_gb"], 40)
        self.assertEqual(imaging.preview_cache_max_gb(), 40.0)
        self.assertEqual(self.saved()["imaging"]["preview_cache_max_gb"], 40)

    def test_the_preview_ceiling_travels_with_the_vlm_keys(self):
        """One body, two sections: the imaging key is taken out before the vlm fields
        reach dataclasses.replace, and must still be written to the file."""
        self.start_server()
        status, _resp = self.post_raw(
            "/api/settings", {"vlm.workers": 4, "imaging.preview_cache_max_gb": 25})
        self.assertEqual(status, 200)
        self.assertEqual(self.cfg.vlm.workers, 4)
        self.assertEqual(imaging.preview_cache_max_gb(), 25.0)
        self.assertEqual(self.saved()["vlm"]["workers"], 4)
        self.assertEqual(self.saved()["imaging"]["preview_cache_max_gb"], 25)

    def test_zero_is_a_legal_ceiling_meaning_none(self):
        """0 is the default and a value a person can choose — the minimum of the spec
        cannot be 1, or "no ceiling" would be unreachable from the form."""
        self.start_server()
        status, resp = self.post_raw(
            "/api/settings", {"imaging.preview_cache_max_gb": 0})
        self.assertEqual(status, 200)
        self.assertEqual(resp["settings"]["imaging.preview_cache_max_gb"], 0)
        self.assertEqual(imaging.preview_cache_max_gb(), 0.0)

    def test_the_quality_thresholds_apply_and_persist(self):
        """F119: `features:` is a second dataclass section, and F113 shipped it without
        an interface — the only way to set the cascade's thresholds was editing the
        file. F138: the cascade's TOGGLES moved to the run screen, the thresholds (which
        cost a run nothing) stayed here, and this is the half that still saves."""
        self.start_server()
        status, resp = self.post_raw("/api/settings", {
            "features.pet_threshold": 0.75,
            "features.subject_score_min": 0.8,
        })
        self.assertEqual(status, 200)
        self.assertEqual(self.cfg.features.pet_threshold, 0.75)
        self.assertEqual(self.cfg.features.subject_score_min, 0.8)
        self.assertEqual(self.saved()["features"]["pet_threshold"], 0.75)
        self.assertEqual(resp["settings"]["features.subject_score_min"], 0.8)

    def test_a_knob_that_moved_to_the_run_screen_is_refused_here(self):
        """F138 §2: a knob has one home. The column no longer offers `vlm.enabled`, so
        the endpoint must not quietly accept it either — two writable addresses for one
        value is exactly the pair of truths the move was made to end.

        F186 retired two of the four keys this case was written over. They stay in the
        list: a retired key is refused for a stronger reason than a moved one, and an
        endpoint that started accepting `vlm.quality` again would be writing a value
        nothing reads.
        """
        self.start_server()
        for key, value in (("vlm.enabled", True), ("vlm.quality", True),
                           ("vlm.quality_scope", "all"), ("features.pets", True)):
            with self.subTest(key=key):
                status, _resp = self.post_raw("/api/settings", {key: value})
                self.assertEqual(status, 400)
        self.assertIs(self.cfg.features.pets, False)

    def test_a_float_setting_takes_a_whole_number_too(self):
        """A form posting `1` for a threshold of 1.0 is not an error."""
        self.start_server()
        status, _resp = self.post_raw(
            "/api/settings", {"features.subject_score_min": 1})
        self.assertEqual(status, 200)
        self.assertEqual(self.cfg.features.subject_score_min, 1.0)

    def test_a_float_out_of_range_is_refused(self):
        self.start_server()
        before = self.cfg.features.pet_threshold  # not a literal: the default is tuned
        for value in (-0.1, 1.5):
            with self.subTest(value=value):
                status, _resp = self.post_raw(
                    "/api/settings", {"features.pet_threshold": value})
                self.assertEqual(status, 400)
        self.assertEqual(self.cfg.features.pet_threshold, before)

    def test_a_bool_is_not_accepted_as_a_float(self):
        """`pet_threshold: true` is garbage, not 1.0 — the same rule the int kind has."""
        self.start_server()
        status, _resp = self.post_raw(
            "/api/settings", {"features.pet_threshold": True})
        self.assertEqual(status, 400)

    def test_the_new_value_is_what_a_reload_of_the_config_would_give(self):
        """"Saved" has to mean the next `sorta index` sees it too, not just this tab."""
        self.start_server()
        self.post_raw("/api/settings", {"vlm.workers": 7, "vlm.max_edge": 640})
        reloaded = load_config(self.config_path)
        self.assertEqual(reloaded.vlm.workers, 7)
        self.assertEqual(reloaded.vlm.max_edge, 640)

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
            {"features.pet_threshold": "high"},  # a string where a number belongs
            {"vlm.model": ""},              # an empty model name
            {"vlm.model": 5},
            {"sort.exclude_dirs": ["/tmp"]},  # a key this endpoint does not own
            {},                             # an "ok" that changed nothing
            [],
            "vlm.model",
        ):
            with self.subTest(payload=payload):
                status, resp = self.post_raw("/api/settings", payload)
                self.assertEqual(status, 400)
                self.assertIn("error", resp)
        self.assertEqual(self.config_path.read_text(encoding="utf-8"), before)
        self.assertEqual(self.cfg.vlm.workers, 2)

    def test_one_bad_key_rejects_the_whole_body(self):
        """A half-applied save leaves the form and the tool disagreeing about which
        half of it is real."""
        self.start_server()
        status, _resp = self.post_raw(
            "/api/settings", {"vlm.max_edge": 640, "vlm.workers": -3})
        self.assertEqual(status, 400)
        self.assertEqual(self.cfg.vlm.max_edge, 896)
        self.assertEqual(self.saved()["vlm"]["max_edge"], 896)

    def test_the_bounds_of_the_spec_are_the_ones_the_form_shows(self):
        """The number inputs carry min/max attributes; if they drifted from the
        server's range the user would be refused by a form that offered the value."""
        html = ui._render_index_html("en")
        # Derived from the spec rather than listed: F119 added five more numeric
        # controls, and a case that names two of them only ever proves that somebody
        # remembered to extend it. Values are compared as numbers — the form writes
        # min="0" where the spec holds 0.0, and that is not a drift.
        for key, spec in ui._SETTINGS_SPEC.items():
            if spec.kind not in ("int", "float"):
                continue
            control = "setting-" + key.replace(".", "-").replace("_", "-")
            found = re.search(rf'id="{control}" min="([^"]+)" max="([^"]+)"', html)
            with self.subTest(key=key):
                self.assertIsNotNone(found, f"{control}: no min/max in the form")
                assert found is not None  # for mypy; assertIsNotNone already checked
                self.assertEqual(float(found.group(1)), float(spec.minimum))
                self.assertEqual(float(found.group(2)), float(spec.maximum))


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

        status, resp = self.post_raw("/api/settings", {"vlm.max_edge": 640})
        self.assertEqual(status, 409)
        self.assertEqual(resp["error"], "already running")
        self.assertEqual(self.config_path.read_text(encoding="utf-8"), before)
        self.assertEqual(self.cfg.vlm.max_edge, 896)

        block.set()
        _poll_until(lambda: json.loads(self.get("/api/process/status")[1]),
                    lambda d: d["finished"])

    def test_the_save_works_again_once_the_run_is_over(self):
        self.patch_fast_stages()
        self.start_server()
        self.post("/api/process", {"source_dir": str(self.src_dir)})
        _poll_until(lambda: json.loads(self.get("/api/process/status")[1]),
                    lambda d: d["finished"])
        status, _resp = self.post_raw("/api/settings", {"vlm.max_edge": 640})
        self.assertEqual(status, 200)
        self.assertEqual(self.saved()["vlm"]["max_edge"], 640)

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
        for control in ("setting-vlm-model",
                        "setting-vlm-workers", "setting-vlm-max-edge",
                        "setting-imaging-preview-cache-max-gb"):
            self.assertIn(f'id="{control}"', self.html)
        # F133: the same column, now behind the gear in the header instead of holding a
        # third of the "Layout" tab at all times. Only the place moved.
        self.assertIn('class="settings-side"', self.html)
        self.assertIn('id="settings-toggle-btn"', self.html)
        self.assertIn('id="settings-panel" class="settings-panel" hidden', self.html)
        self.assertIn("/api/settings", self.html)

    def test_the_column_is_no_longer_part_of_the_layout_tab(self):
        """The point of the move: configuration people return to about once a month is
        not a working surface and must not stand next to the plan."""
        layout = self.html.split('id="tab-layout"', 1)[1].split("</section", 1)[0]
        for key in ui._SETTINGS_SPEC:
            control = "setting-" + key.replace(".", "-").replace("_", "-")
            with self.subTest(key=key):
                self.assertNotIn(control, layout)
        panel = self.html.split('id="settings-panel"', 1)[1].split("</aside>", 1)[0]
        for key in ui._SETTINGS_SPEC:
            control = "setting-" + key.replace(".", "-").replace("_", "-")
            with self.subTest(key=key):
                self.assertIn(f'id="{control}"', panel)

    def test_every_function_has_its_own_toggle(self):
        """Explicitly asked for: no single "smart mode" switch that means four things
        at once — the deep tier is one checkbox, the model, the threads and the frame
        size are their own fields.

        Stated against the spec rather than against a count: the number was 4 and is now
        5 (F117 added the preview-cache ceiling), and a magic number only ever says that
        somebody added a knob — not that the knob got a control of its own.
        """
        for key in ui._SETTINGS_SPEC:
            control = "setting-" + key.replace(".", "-").replace("_", "-")
            with self.subTest(key=key):
                self.assertEqual(self.html.count(f'id="{control}"'), 1, control)

    def test_the_folder_language_moved_into_the_column(self):
        """It was the odd one out in the action row: a setting standing next to the
        button that moves the collection."""
        side = self.html.split('class="settings-side"', 1)[1].split("</aside>", 1)[0]
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
