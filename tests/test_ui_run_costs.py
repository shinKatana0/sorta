"""F138: the run screen says what a run costs, and the expensive knobs live on it.

Three knobs that decide between a quarter of an hour and four hours of a run sat in the
settings column next to the number of preparation threads (`vlm.quality` +
`vlm.quality_scope`, `features.pets_verify`, `dedup.keeper_vlm`). They moved onto the
screen where a run is started, and what makes that a budget rather than a longer row of
switches is the price on every line plus the sum under them.

What is pinned here is the whole of that:

* the four knobs reach the run and do NOT rewrite config.yaml (§2, test 1);
* each starts from the config (test 2);
* each is GONE from the settings column, explicitly — one home per value (test 3);
* the thresholds and the model stayed there (test 4);
* the scope select is shown only with its parent on (§3, test 5);
* the sum is the sum of the lines and moves with a toggle, in the browser (test 6);
* an empty collection shows a dash, never a zero (§1, test 7);
* the estimate is labelled as an estimate (§1, test 8).
"""
from __future__ import annotations

import dataclasses
import json
import re
import threading

import yaml

from sorta import ui
from sorta.config import load_config

from tests.test_ui_process import ProcessTestBase, _poll_until


class RunCostsTestBase(ProcessTestBase):
    """A server with a real config.yaml — "does not rewrite the file" needs a file."""

    def setUp(self):
        super().setUp()
        # The estimate is cached by (db, fingerprint, thresholds) like the Duplicates
        # payload — fixtures of a previous case must not answer this one.
        ui._estimate_cache_clear()
        self.addCleanup(ui._estimate_cache_clear)
        self.config_path = self.root / "config.yaml"
        self.config_path.write_text(
            "language: en\n"
            "vlm:\n"
            "  quality: false\n"
            "  quality_scope: groups\n"
            "features:\n"
            "  pets: false\n"
            "  pets_verify: false\n"
            "dedup:\n"
            "  keeper_vlm: false\n",
            encoding="utf-8")
        loaded = load_config(self.config_path)
        self.cfg.vlm, self.cfg.features = loaded.vlm, loaded.features
        self.cfg.dedup, self.cfg.raw = loaded.dedup, loaded.raw

    def start_server(self, config_path=None) -> None:
        path = self.config_path if config_path is None else (config_path or None)
        self.server = ui.build_server(self.cfg, self.conn, port=0, config_path=path)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def saved(self) -> dict:
        return yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}

    def run_once(self, body: dict) -> object:
        """Start a run with `body`, wait for it, and return the cfg the junk stage saw.

        The junk stage is the one every F138 knob is a setting of, so its config is
        where "the checkbox reached the run" is either true or not.
        """
        seen: list[object] = []

        def fake_junk(cfg, conn, classifier=None, use_clip=True, text_detector=None,
                      progress=None):
            self.calls.append("junk")
            seen.append(cfg)

        self.patch_fast_stages()
        self._patch("classify_junk", fake_junk)
        self.start_server()
        status, resp = self.post("/api/process",
                                 {"source_dir": str(self.src_dir), **body})
        self.assertEqual(status, 200, resp)
        _poll_until(self.status, lambda d: d["finished"])
        self.assertEqual(len(seen), 1)
        return seen[0]


class TestKnobsReachTheRun(RunCostsTestBase):
    """Test 1: each of the four drives the run — and leaves config.yaml alone."""

    def test_every_moved_knob_reaches_the_junk_stage(self):
        run_cfg = self.run_once({
            "pets": True, "pets_verify": True, "quality": True,
            "quality_scope": "events", "keeper": True,
        })
        self.assertIs(run_cfg.vlm.quality, True)
        self.assertEqual(run_cfg.vlm.quality_scope, "events")
        self.assertIs(run_cfg.features.pets_verify, True)
        self.assertIs(run_cfg.dedup.keeper_vlm, True)

    def test_the_file_is_not_rewritten_by_a_run(self):
        """§2's exception, the F123 shape: the screen starts from the config and
        overrides it for ONE run. A run that wrote the file back would make every
        checkbox a permanent setting nobody asked to change."""
        before = self.config_path.read_text(encoding="utf-8")
        self.run_once({"pets": True, "pets_verify": True, "quality": True,
                       "quality_scope": "all", "keeper": True})
        self.assertEqual(self.config_path.read_text(encoding="utf-8"), before)
        self.assertEqual(self.saved()["vlm"]["quality"], False)
        self.assertEqual(self.saved()["features"]["pets_verify"], False)
        self.assertEqual(self.saved()["dedup"]["keeper_vlm"], False)

    def test_the_shared_config_of_the_server_is_not_mutated(self):
        """The other routes read the same object; a run must not move it under them."""
        self.run_once({"quality": True, "keeper": True, "pets_verify": True,
                       "pets": True})
        self.assertIs(self.cfg.vlm.quality, False)
        self.assertIs(self.cfg.features.pets_verify, False)
        self.assertIs(self.cfg.dedup.keeper_vlm, False)

    def test_an_unticked_box_forces_off_what_the_config_switched_on(self):
        """The F57 rule, carried over: an empty checkbox means OFF, not "as the file
        says" — otherwise unticking something expensive would quietly do nothing."""
        self.cfg.vlm = dataclasses.replace(self.cfg.vlm, quality=True)
        self.cfg.features = dataclasses.replace(self.cfg.features, pets_verify=True)
        self.cfg.dedup = dataclasses.replace(self.cfg.dedup, keeper_vlm=True)
        run_cfg = self.run_once({"quality": False, "pets_verify": False,
                                 "keeper": False})
        self.assertIs(run_cfg.vlm.quality, False)
        self.assertIs(run_cfg.features.pets_verify, False)
        self.assertIs(run_cfg.dedup.keeper_vlm, False)

    def test_a_body_without_them_leaves_the_config_alone(self):
        """`/api/process/rerun-optional` and any caller outside the browser have no
        interface for these four; an absent field means "the file decides", the
        cli._quality_overrides convention — not a silent OFF."""
        self.cfg.vlm = dataclasses.replace(self.cfg.vlm, quality=True)
        self.cfg.dedup = dataclasses.replace(self.cfg.dedup, keeper_vlm=True)
        run_cfg = self.run_once({})
        self.assertIs(run_cfg.vlm.quality, True)
        self.assertIs(run_cfg.dedup.keeper_vlm, True)

    def test_an_unknown_scope_is_refused_rather_than_defaulted(self):
        """`all` is the 4.3-hour option: a misspelling must not be rounded into it or
        past it, so the set is closed and a miss is a 400."""
        self.patch_fast_stages()
        self.start_server()
        status, _resp = self.post("/api/process", {"source_dir": str(self.src_dir),
                                                   "quality_scope": "everything"})
        self.assertEqual(status, 400)

    def test_a_flag_that_is_not_a_boolean_is_refused(self):
        self.patch_fast_stages()
        self.start_server()
        for body in ({"quality": "yes"}, {"keeper": 1}, {"pets_verify": "true"}):
            with self.subTest(body=body):
                status, _resp = self.post(
                    "/api/process", {"source_dir": str(self.src_dir), **body})
                self.assertEqual(status, 400)


class TestDefaultsComeFromTheConfig(RunCostsTestBase):
    """Test 2: the starting state of every moved knob is what the file says."""

    def test_defaults_follow_the_config(self):
        self.cfg.vlm = dataclasses.replace(self.cfg.vlm, quality=True,
                                           quality_scope="faces")
        self.cfg.features = dataclasses.replace(self.cfg.features, pets_verify=True)
        self.cfg.dedup = dataclasses.replace(self.cfg.dedup, keeper_vlm=True)
        self.start_server()
        _status, body, _ctype = self.get("/api/process/defaults")
        data = json.loads(body)
        self.assertIs(data["quality"], True)
        self.assertEqual(data["quality_scope"], "faces")
        self.assertIs(data["pets_verify"], True)
        self.assertIs(data["keeper"], True)

    def test_the_script_sets_every_checkbox_from_that_answer(self):
        html = ui._render_index_html("en")
        for field, control in (("pets_verify", "process-pets-verify-checkbox"),
                               ("quality", "process-quality-checkbox"),
                               ("keeper", "process-keeper-checkbox")):
            with self.subTest(field=field):
                self.assertIn(
                    f'document.getElementById("{control}").checked = !!data.{field};',
                    html)
        self.assertIn('document.getElementById("process-quality-scope").value ='
                      ' data.quality_scope;', html)


class TestOneHomePerKnob(RunCostsTestBase):
    """Tests 3 and 4: the moved knobs left the column, the cheap ones stayed."""

    def test_the_moved_knobs_are_gone_from_the_settings_column(self):
        html = ui._render_index_html("en")
        panel = html.split('id="settings-panel"', 1)[1].split("</aside>", 1)[0]
        for control in ("setting-vlm-enabled", "setting-vlm-quality",
                        "setting-vlm-quality-scope", "setting-features-pets\""):
            with self.subTest(control=control):
                self.assertNotIn(control, panel)

    def test_the_endpoint_no_longer_accepts_them_either(self):
        """A control removed from the form but still writable through the route would
        leave the second home in place — just harder to find."""
        for key in ("vlm.enabled", "vlm.quality", "vlm.quality_scope",
                    "features.pets"):
            with self.subTest(key=key):
                self.assertNotIn(key, ui._SETTINGS_SPEC)

    def test_the_cheap_knobs_stayed_in_the_column(self):
        html = ui._render_index_html("en")
        panel = html.split('id="settings-panel"', 1)[1].split("</aside>", 1)[0]
        for control in ("setting-vlm-model", "setting-vlm-workers",
                        "setting-vlm-max-edge",
                        "setting-features-pet-threshold",
                        "setting-features-sharpness-band-min",
                        "setting-features-sharpness-band-max",
                        "setting-features-subject-score-min",
                        "setting-imaging-preview-cache-max-gb"):
            with self.subTest(control=control):
                self.assertIn(f'id="{control}"', panel)

    def test_each_moved_knob_has_exactly_one_control_in_the_page(self):
        html = ui._render_index_html("en")
        for control in ("process-quality-checkbox", "process-quality-scope",
                        "process-pets-verify-checkbox", "process-keeper-checkbox",
                        "process-deep-checkbox", "process-pets-checkbox"):
            with self.subTest(control=control):
                self.assertEqual(html.count(f'id="{control}"'), 1)


class TestTheBlockItself(RunCostsTestBase):
    """Tests 5, 6 and 8 plus the readability criterion — the markup and its script."""

    def setUp(self):
        super().setUp()
        self.html = ui._render_index_html("en")
        self.options = self.html.split('id="step-options"', 1)[1].split(
            'id="step-actions"', 1)[0]

    def test_the_block_holds_no_more_than_seven_lines(self):
        """The readability criterion, and the reason the four knobs could be moved at
        all: a longer list is the console of switches F133 took away."""
        block = self.html.split('id="process-costs"', 1)[1].split("</div>\n</div>", 1)[0]
        self.assertEqual(len(re.findall(r'class="cost-row"', block)), 7)

    def test_every_line_carries_a_price_slot(self):
        for key in ("base", "faces", "events", "pets", "pets_verify", "deep",
                    "quality", "keeper"):
            with self.subTest(key=key):
                self.assertIn(f'data-cost="{key}"', self.options)

    def test_the_scope_starts_hidden_and_is_shown_with_its_parent(self):
        """§3: a scope for a question nobody is asking is a choice about nothing."""
        row = self.options.split('id="process-quality-scope-row"', 1)[1][:60]
        self.assertIn('style="display:none"', row)
        self.assertIn('document.getElementById("process-quality-scope-row").style.display'
                      ' =\n        qualityOn ? "" : "none";', self.html)
        self.assertIn('var qualityOn = document.getElementById('
                      '"process-quality-checkbox").checked;', self.html)

    def test_the_only_nesting_is_one_level_deep(self):
        """A subordinate control of a subordinate control is where this block would
        stop being readable, so there is deliberately no second level."""
        self.assertEqual(self.options.count('class="cost-child"'), 2)
        for child in self.options.split('class="cost-child"')[1:]:
            self.assertNotIn("cost-child", child.split("</span>", 1)[0])

    def test_the_total_stands_between_the_list_and_the_run_button(self):
        """Where the eye is already going. Its own block, so collapsing the options
        does not take the price away with them."""
        actions = self.html.split('id="step-actions"', 1)[1]
        budget = actions.index('id="process-budget"')
        button = actions.index('id="process-start-btn"')
        self.assertLess(budget, button)
        self.assertLess(self.html.index('id="process-costs"'), self.html.index(
            'id="process-budget"'))

    def test_the_total_is_summed_in_the_browser_not_asked_for(self):
        """Test 6: a request between a person and a switch they are still deciding
        about is exactly what "updates immediately" rules out — and there is nothing
        to ask, since a checkbox does not change what the index holds."""
        self.assertIn("total += seconds;", self.html)
        self.assertIn('value.textContent = formatCost(total);', self.html)
        listeners = self.html.split(
            '"process-quality-scope", "process-keeper-checkbox"].forEach', 1)[1]
        self.assertIn('addEventListener("change", renderCosts)',
                      listeners.split("});", 1)[0])

    def test_the_estimate_says_it_is_an_estimate(self):
        """Test 8. A wrong exact number is worse than an honest approximate one."""
        self.assertIn(ui._t("costs_estimate_note", "en"), self.options)
        self.assertIn("estimate", ui._t("costs_estimate_note", "en"))

    def test_every_new_string_exists_in_all_three_languages(self):
        for key in ("costs_title", "costs_estimate_note", "costs_base_label",
                    "costs_always", "costs_total_label", "costs_total_at_least",
                    "costs_unknown", "costs_free", "costs_under_minute",
                    "costs_minutes", "costs_hours", "process_pets_verify_label",
                    "process_pets_verify_hint", "process_quality_label",
                    "process_quality_hint", "process_quality_scope_label",
                    "process_keeper_label", "process_keeper_hint",
                    "settings_costs_moved_hint"):
            with self.subTest(key=key):
                entry = ui._UI_STRINGS[key]
                self.assertEqual(set(entry), {"ru", "en", "ja"})
                for lang, value in entry.items():
                    self.assertTrue(value.strip(), f"{key}/{lang} is empty")


class TestEstimateEndpoint(RunCostsTestBase):
    """Test 7 and the arithmetic of §1 — measured rates over counts from this index."""

    def estimate(self) -> dict:
        status, body, ctype = self.get("/api/process/estimate")
        self.assertEqual(status, 200)
        self.assertIn("application/json", ctype)
        return json.loads(body)

    def add_photo(self, rel: str, *, phash: str | None = None) -> int:
        file_id, _p, _c = self.add_photo_file(rel)
        if phash is not None:
            self.conn.execute("UPDATE files SET phash = ? WHERE id = ?",
                              (phash, file_id))
            self.conn.commit()
        return file_id

    def test_an_empty_collection_shows_a_dash_not_a_zero(self):
        """§1: a zero reads as "free". Nothing has been indexed, so nothing is known
        — and the honest answer to "how long" is that this index cannot say."""
        self.start_server()
        data = self.estimate()
        for key, value in data["seconds"].items():
            with self.subTest(key=key):
                self.assertIsNone(value, key)
        self.assertTrue(all(v is None for v in data["counts"].values()))

    def test_a_stage_that_never_ran_is_unknown_rather_than_free(self):
        """The frames the deep tier asks about are chosen from the CLIP scores of the
        run in progress: before one has happened the number does not exist."""
        self.add_photo("a.jpg")
        self.start_server()
        data = self.estimate()
        self.assertIsNone(data["seconds"]["deep"])
        self.assertIsNone(data["seconds"]["pets_verify"])
        self.assertIsNone(data["seconds"]["keeper"])
        self.assertIsNotNone(data["seconds"]["base"])

    def test_the_lines_that_only_need_frames_are_priced_at_once(self):
        for i in range(4):
            self.add_photo(f"p{i}.jpg")
        self.start_server()
        data = self.estimate()
        self.assertEqual(data["counts"]["faces"], 4)
        self.assertAlmostEqual(data["seconds"]["faces"],
                               round(4 * ui._SEC_PER_FACES_FRAME, 1))
        # F123: the animals ride on a CLIP pass that runs anyway — the one line where
        # a zero is the truth rather than a missing number.
        self.assertEqual(data["seconds"]["pets"], 0.0)

    def test_the_deep_tier_is_priced_by_the_frames_it_answered_on(self):
        ids = [self.add_photo(f"p{i}.jpg") for i in range(3)]
        for file_id, source in zip(ids, ("clip", "vlm", "vlm")):
            self.conn.execute(
                "INSERT INTO media_class (file_id, verdict, source, tier, updated_at)"
                " VALUES (?, 'photo', ?, 'vlm', '2026-01-01')", (file_id, source))
        self.conn.commit()
        self.start_server()
        data = self.estimate()
        self.assertEqual(data["counts"]["deep"], 2)
        self.assertAlmostEqual(data["seconds"]["deep"],
                               round(2 * ui._SEC_PER_VLM_FRAME, 1))

    def test_the_pet_check_is_priced_by_its_candidate_threshold(self):
        ids = [self.add_photo(f"p{i}.jpg") for i in range(3)]
        for file_id, score in zip(ids, (0.9, 0.55, 0.1)):
            self.conn.execute(
                "INSERT INTO frame_quality (file_id, pet_score, source, updated_at)"
                " VALUES (?, ?, 'clip', '2026-01-01')", (file_id, score))
        self.conn.commit()
        self.cfg.features = dataclasses.replace(self.cfg.features,
                                                pet_candidate_threshold=0.5)
        self.start_server()
        data = self.estimate()
        self.assertEqual(data["counts"]["pets_verify"], 2)
        self.assertAlmostEqual(data["seconds"]["pets_verify"],
                               round(2 * ui._SEC_PER_VLM_FRAME, 1))

    def test_the_keeper_question_is_priced_per_group_not_per_frame(self):
        """F132 measured 1.32 s for a call carrying up to five frames — multiplying a
        per-frame rate by the frames of a group is the arithmetic that turns "ten
        minutes" into an hour.

        The pair is deliberate and so is the pinned `keeper_min_group_size`: this case is
        about the ARITHMETIC over a group, and it must not move when the product default
        moves (F144 raised it to 3 and pinned the mechanism tests the same way).
        """
        for i in range(2):
            self.add_photo(f"dup{i}.jpg", phash="f" * 16)
        self.add_photo("alone.jpg", phash="0" * 16)
        self.cfg.dedup = dataclasses.replace(self.cfg.dedup, keeper_min_group_size=2)
        self.start_server()
        data = self.estimate()
        self.assertEqual(data["counts"]["keeper"], 1)
        self.assertAlmostEqual(data["seconds"]["keeper"],
                               round(ui._SEC_PER_VLM_GROUP, 1))
        # The same grouping is what `quality_scope: groups` asks about — by frames.
        self.assertEqual(data["counts"]["quality_groups"], 2)

    def test_a_group_below_the_configured_size_is_not_asked_about(self):
        for i in range(2):
            self.add_photo(f"dup{i}.jpg", phash="f" * 16)
        self.cfg.dedup = dataclasses.replace(self.cfg.dedup, keeper_min_group_size=3)
        self.start_server()
        self.assertEqual(self.estimate()["counts"]["keeper"], 0)

    def test_every_scope_is_priced_so_the_select_costs_no_request(self):
        """The four scopes differ by hours, and the choice has to be answerable the
        moment it is made — so all four travel at once."""
        for i in range(3):
            self.add_photo(f"p{i}.jpg", phash="f" * 16)
        self.start_server()
        data = self.estimate()
        self.assertEqual(data["counts"]["quality_all"], 3)
        self.assertEqual(data["counts"]["quality_groups"], 3)
        # Neither events nor faces have been built — a dash, not "no frames".
        self.assertIsNone(data["seconds"]["quality_events"])
        self.assertIsNone(data["seconds"]["quality_faces"])

    def test_the_faces_scope_becomes_knowable_once_faces_have_been_found(self):
        first = self.add_photo("a.jpg")
        self.add_photo("b.jpg")
        self.conn.execute(
            "INSERT INTO faces (file_id, bbox, embedding) VALUES (?, '[1,2,3,4]', x'00')",
            (first,))
        self.conn.commit()
        self.start_server()
        data = self.estimate()
        self.assertEqual(data["counts"]["quality_faces"], 1)
        self.assertAlmostEqual(data["seconds"]["quality_faces"],
                               round(ui._SEC_PER_VLM_FRAME, 1))

    def test_the_answer_follows_the_index_rather_than_the_cache(self):
        """The near-duplicate grouping costs seconds, so the payload is cached the way
        the Duplicates one is — and, like it, keyed on the state of the DB: a run that
        indexes more frames must not be answered with the price of the old ones."""
        self.add_photo("a.jpg")
        self.start_server()
        first = self.estimate()["counts"]["faces"]
        self.add_photo("b.jpg")
        self.assertEqual(self.estimate()["counts"]["faces"], first + 1)

    def test_counts_travel_with_the_seconds(self):
        """A number somebody is asked to plan an evening around should be checkable
        against the collection it came from."""
        self.add_photo("a.jpg")
        self.start_server()
        data = self.estimate()
        self.assertEqual(set(data), {"seconds", "counts"})
        self.assertEqual(set(data["seconds"]), set(data["counts"]))
