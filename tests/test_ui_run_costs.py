"""F138: the run screen says what a run costs, and the expensive knobs live on it.

Three knobs that decide between a quarter of an hour and four hours of a run sat in the
settings column next to the number of preparation threads (`vlm.quality` +
`vlm.quality_scope`, `features.pets_verify`, `dedup.keeper_vlm`). They moved onto the
screen where a run is started, and what makes that a budget rather than a longer row of
switches is the price on every line plus the sum under them.

F186 retired two of those three questions, and the budget lost the keeper line and the
scope select with them. A budget must not quote a price for a stage that no longer runs,
so what is checked below is both that the retired lines are gone and that the ones beside
them did not move.

What is pinned here is the whole of that:

* the knobs reach the run and do NOT rewrite config.yaml (§2, test 1);
* each starts from the config (test 2);
* each is GONE from the settings column, explicitly — one home per value (test 3);
* the thresholds and the model stayed there (test 4);
* the budget names every line the run has and no line it does not (F186);
* the sum is the sum of the lines and moves with a toggle, in the browser (test 6);
* an empty collection shows a dash, never a zero (§1, test 7);
* the estimate is labelled as an estimate (§1, test 8).
"""
from __future__ import annotations

import dataclasses
import json
import os
import re
import threading
from unittest import mock

import yaml

from sorta import runlog, ui
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
        # F159: the estimate now also reads the run log. Every case gets a path of its
        # own and does not create the file, so these cases keep pricing a run the way
        # they always did — with the shipped defaults, on a machine that has no timings.
        self.log_path = self.root / "runlog" / "sorta.log"
        patcher = mock.patch.dict(os.environ,
                                  {runlog.ENV_LOG_FILE: str(self.log_path)})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.config_path = self.root / "config.yaml"
        self.config_path.write_text(
            "language: en\n"
            "vlm:\n"
            "  products: true\n"
            "features:\n"
            "  pets: false\n"
            "  pets_verify: false\n",
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
                      verdicts_only=False, progress=None):
            # F165: the knobs under test are settings of the half AFTER faces, so
            # that is the call whose config is captured.
            self.calls.append("classify" if verdicts_only else "junk")
            if not verdicts_only:
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
        run_cfg = self.run_once({"pets": True, "pets_verify": True})
        self.assertIs(run_cfg.features.pets, True)
        self.assertIs(run_cfg.features.pets_verify, True)

    def test_the_file_is_not_rewritten_by_a_run(self):
        """§2's exception, the F123 shape: the screen starts from the config and
        overrides it for ONE run. A run that wrote the file back would make every
        checkbox a permanent setting nobody asked to change."""
        before = self.config_path.read_text(encoding="utf-8")
        self.run_once({"pets": True, "pets_verify": True, "products": False})
        self.assertEqual(self.config_path.read_text(encoding="utf-8"), before)
        self.assertEqual(self.saved()["features"]["pets_verify"], False)
        self.assertEqual(self.saved()["vlm"]["products"], True)

    def test_the_shared_config_of_the_server_is_not_mutated(self):
        """The other routes read the same object; a run must not move it under them."""
        self.run_once({"pets_verify": True, "pets": True})
        self.assertIs(self.cfg.features.pets, False)
        self.assertIs(self.cfg.features.pets_verify, False)

    def test_an_unticked_box_forces_off_what_the_config_switched_on(self):
        """The F57 rule, carried over: an empty checkbox means OFF, not "as the file
        says" — otherwise unticking something expensive would quietly do nothing."""
        self.cfg.features = dataclasses.replace(self.cfg.features, pets=True,
                                                pets_verify=True)
        run_cfg = self.run_once({"pets": False, "pets_verify": False})
        self.assertIs(run_cfg.features.pets, False)
        self.assertIs(run_cfg.features.pets_verify, False)

    def test_a_body_without_them_leaves_the_config_alone(self):
        """`/api/process/rerun-optional` and any caller outside the browser have no
        interface for these four; an absent field means "the file decides", the
        cli._quality_overrides convention — not a silent OFF."""
        self.cfg.features = dataclasses.replace(self.cfg.features, pets_verify=True)
        run_cfg = self.run_once({})
        self.assertIs(run_cfg.features.pets_verify, True)

    def test_a_retired_flag_switches_nothing_on(self):
        """F186: the run route knew three flags this screen no longer has.

        A caller outside the browser may still send them — a script, a saved request —
        and the route reads them the way it reads any field it does not know: not at all.
        What matters is that the run then does what the file says, rather than the model
        being asked a question that no longer exists.
        """
        run_cfg = self.run_once({"quality": True, "quality_scope": "faces",
                                 "keeper": True})
        for section, key in (("vlm", "quality"), ("vlm", "quality_scope"),
                             ("dedup", "keeper_vlm")):
            with self.subTest(key=f"{section}.{key}"):
                self.assertFalse(hasattr(getattr(run_cfg, section), key))
        # and the run itself happened, with the settings of the file beside them
        self.assertIs(run_cfg.features.pets_verify, False)

    def test_a_flag_that_is_not_a_boolean_is_refused(self):
        self.patch_fast_stages()
        self.start_server()
        for body in ({"pets": "yes"}, {"products": 1}, {"pets_verify": "true"}):
            with self.subTest(body=body):
                status, _resp = self.post(
                    "/api/process", {"source_dir": str(self.src_dir), **body})
                self.assertEqual(status, 400)


class TestDefaultsComeFromTheConfig(RunCostsTestBase):
    """Test 2: the starting state of every moved knob is what the file says."""

    def test_defaults_follow_the_config(self):
        self.cfg.features = dataclasses.replace(self.cfg.features, pets=True,
                                                pets_verify=True)
        self.cfg.vlm = dataclasses.replace(self.cfg.vlm, products=False)
        self.start_server()
        _status, body, _ctype = self.get("/api/process/defaults")
        data = json.loads(body)
        self.assertIs(data["pets"], True)
        self.assertIs(data["pets_verify"], True)
        self.assertIs(data["products"], False)

    def test_the_script_sets_every_checkbox_from_that_answer(self):
        html = ui._render_index_html("en")
        for field, control in (("pets_verify", "process-pets-verify-checkbox"),
                               ("pets", "process-pets-checkbox"),
                               ("products", "process-products-checkbox")):
            with self.subTest(field=field):
                self.assertIn(
                    f'document.getElementById("{control}").checked = !!data.{field};',
                    html)


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
        for control in ("process-pets-verify-checkbox", "process-deep-checkbox",
                        "process-products-checkbox", "process-pets-checkbox"):
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
        all: a longer list is the console of switches F133 took away.

        Five top-level rows since F186 took two questions out of the budget — under the
        ceiling rather than at it, which is the direction this criterion is allowed to
        move in. Both numbers are asserted: the ceiling is the rule, the exact count is
        what makes a line appearing or vanishing unnoticed impossible.
        """
        block = self.html.split('id="process-costs"', 1)[1].split("</div>\n</div>", 1)[0]
        rows = len(re.findall(r'class="cost-row"', block))
        self.assertLessEqual(rows, 7)
        self.assertEqual(rows, 5)

    def test_every_line_carries_a_price_slot(self):
        for key in ("base", "faces", "events", "pets", "pets_verify", "deep",
                    "products"):
            with self.subTest(key=key):
                self.assertIn(f'data-cost="{key}"', self.options)

    def test_the_budget_names_no_line_the_run_does_not_have(self):
        """F186: the keeper line and the scope select are gone from the markup.

        A price slot for a stage nobody runs is worse than a missing one — it reads as a
        cost a person is about to pay, and the sum under the list would carry it.
        """
        for gone in ('data-cost="quality"', 'data-cost="keeper"',
                     'id="process-quality-scope-row"', 'id="process-quality-scope"',
                     'id="process-keeper-checkbox"'):
            with self.subTest(fragment=gone):
                self.assertNotIn(gone, self.html)

    def test_the_only_nesting_is_one_level_deep(self):
        """A subordinate control of a subordinate control is where this block would
        stop being readable, so there is deliberately no second level.

        Two children: the animal check under the animals, and the product line under the
        master switch (F161) — which is what keeps the block short while the master stops
        doing anything by itself. The third was the scope select of the quality question,
        and F186 retired it with the question.
        """
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
            '"process-products-checkbox"].forEach', 1)[1]
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
                    "process_pets_verify_hint", "settings_costs_moved_hint"):
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
        run in progress: before one has happened the number does not exist.

        F161 moved that line from `deep` to `products` — the master switch is priced at
        zero on any collection, which is a different statement and has a case of its own
        below.
        """
        self.add_photo("a.jpg")
        self.start_server()
        data = self.estimate()
        self.assertIsNone(data["seconds"]["products"])
        self.assertIsNone(data["seconds"]["pets_verify"])
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
        """F161: the line is `products` now, and it is the same arithmetic on the same
        population — what changed is which checkbox it stands next to."""
        ids = [self.add_photo(f"p{i}.jpg") for i in range(3)]
        for file_id, source in zip(ids, ("clip", "vlm", "vlm")):
            self.conn.execute(
                "INSERT INTO media_class (file_id, verdict, source, tier, updated_at)"
                " VALUES (?, 'photo', ?, 'vlm', '2026-01-01')", (file_id, source))
        self.conn.commit()
        self.start_server()
        data = self.estimate()
        self.assertEqual(data["counts"]["products"], 2)
        self.assertAlmostEqual(data["seconds"]["products"],
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

    def test_the_estimate_prices_the_lines_of_the_run_and_no_others(self):
        """F186: the retired questions took their lines out of the payload.

        Two of them were priced here — the keeper question per GROUP (a population this
        endpoint computed with a near-duplicate pass of its own) and the quality question
        per frame, once for each of its four scopes. Neither stage runs, so neither may
        appear in a budget; and the lines that stayed have to be exactly the ones the run
        still has, which is what the second half of this case says.
        """
        for i in range(2):
            self.add_photo(f"dup{i}.jpg", phash="f" * 16)
        self.start_server()
        data = self.estimate()
        self.assertEqual(set(data["seconds"]),
                         {"base", "faces", "events", "pets", "pets_verify", "deep",
                          "products"})
        for retired in ("keeper", "quality_all", "quality_groups", "quality_events",
                        "quality_faces"):
            with self.subTest(line=retired):
                self.assertNotIn(retired, data["seconds"])
                self.assertNotIn(retired, data["counts"])

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
        against the collection it came from — and, since F159, against the machine the
        rate behind it was measured on."""
        self.add_photo("a.jpg")
        self.start_server()
        data = self.estimate()
        self.assertEqual(set(data), {"seconds", "counts", "sources", "measured_at"})
        self.assertEqual(set(data["seconds"]), set(data["counts"]))
        self.assertEqual(set(data["seconds"]), set(data["sources"]))
