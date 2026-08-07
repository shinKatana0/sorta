"""F222: a run states what it will download, before it starts.

The report this feature comes from: the owner installed the program, chose almost
nothing in the wizard, started a run — and the run went `indexing → geo → landmarks →
verdicts`, hung on the landmarks and died on the verdicts. His question was "why did it
get in at all, when I ticked almost nothing in the setup?"

The answer was that those stages have no tick. `sorta run` made only `faces` and
`events` optional; landmarks, classification and near-duplicates ran always, downloaded
their weights silently, and F217's tier notes could not help because a note hangs on an
OPTION and those stages had none.

What is under test here:

* the landmark stage is a choice now, and it is off — 143 places of 26 137 (0.55%) for
  1.6 GB of weights;
* the regression that decision is not allowed to cost: `visual` places already in a
  database survive a run with the stage switched off, untouched;
* the pairing "option -> weights" is DERIVED and guarded. F217's two notes went to the
  two options whose checkbox happens to be named like a tier; the animals, which load
  the very same CLIP, got none. A checkbox that can raise a model and is missing from
  the table fails this suite;
* the download summary is built from the same probe as those notes, and it counts the
  stages that have no checkbox too;
* a refusal to download is a sentence naming the stage, the model and the size — not a
  traceback.
"""
from __future__ import annotations

import dataclasses
import json
import unittest
from pathlib import Path
from unittest import mock

from sorta import i18n, tiers, ui, wizard
from sorta.config import FeaturesConfig, _features_from
from sorta.ui import strings as ui_strings
from tests.test_ui_process import ProcessTestBase, _poll_until

_LANGS: tuple[i18n.Lang, ...] = ("ru", "en", "ja")
_ROOT = Path(__file__).resolve().parent.parent
_APP_JS = _ROOT / "sorta" / "web" / "app" / "app.js"


class TestTheLandmarkStageIsAChoice(unittest.TestCase):
    """Decision 1 of the brief: off by default, but still available."""

    def test_the_default_is_off(self):
        self.assertFalse(FeaturesConfig().landmarks)

    def test_a_config_can_switch_it_on_and_off(self):
        self.assertTrue(_features_from({"landmarks": True}).landmarks)
        self.assertFalse(_features_from({"landmarks": False}).landmarks)

    def test_a_config_that_never_heard_of_the_key_gets_the_default(self):
        self.assertFalse(_features_from({}).landmarks)

    def test_the_example_config_documents_it(self):
        text = (_ROOT / "config.example.yaml").read_text(encoding="utf-8")
        self.assertIn("landmarks: false", text)


class TestTheOptionToWeightsTableIsDerivedAndGuarded(unittest.TestCase):
    """§5 of the brief. The pairing was done by eye and it showed: of everything on the
    run screen exactly two options got a tier note, and they are the two whose caption
    matches a tier name. Animals, landmarks and the classification share ViT-L-14, which
    belongs to a tier called "Search by words", so nobody connected them."""

    def test_every_checkbox_of_the_run_screen_is_in_the_table(self):
        """The guard the brief asks for: an option without an entry fails the suite,
        instead of being found by a user two releases later."""
        html = (_ROOT / "sorta" / "web" / "page.html").read_text(encoding="utf-8")
        found = set()
        for chunk in html.split('id="process-')[1:]:
            name = chunk.split('"')[0]
            if name.endswith("-checkbox"):
                found.add(name[: -len("-checkbox")].replace("-", "_"))
        self.assertIn("landmarks", found)   # the case this feature adds
        self.assertIn("pets", found)        # the case it was found by
        for key in sorted(found):
            with self.subTest(option=key):
                self.assertIn(key, tiers.RUN_PARTS_BY_KEY)

    def test_the_animals_now_have_a_tier_and_it_is_the_clip_one(self):
        """Test 6 of the brief, named: today they have no note at all."""
        self.assertEqual(tiers.part_tiers("pets"), ("search",))
        self.assertEqual(tiers.RUN_PARTS_BY_KEY["pets"].weights, ("ViT-L-14",))

    def test_the_tier_of_a_part_is_read_off_the_catalog_not_off_its_name(self):
        for key, expected in (("landmarks", "search"), ("classify", "search"),
                              ("faces", "faces"), ("deep", "deep"),
                              ("products", "deep"), ("landmarks_verify", "deep")):
            with self.subTest(part=key):
                self.assertEqual(tiers.part_tiers(key), (expected,))

    def test_a_part_that_raises_nothing_names_no_tier(self):
        for key in ("base", "events", "geo_online"):
            with self.subTest(part=key):
                self.assertEqual(tiers.part_tiers(key), ())

    def test_every_weight_of_the_catalog_has_a_size(self):
        """The same rule the cache markers already live under: a weight named by the
        catalog and missing here would be priced at zero, and a run would promise a free
        download of 1.6 GB."""
        named = {weight for tier in wizard.TIERS for weight in tier.weights}
        self.assertEqual(named - set(tiers._WEIGHT_MB), set())

    def test_the_weight_sizes_add_up_to_what_the_catalog_states(self):
        """Per-weight numbers exist so a stage can quote what IT fetches — but they may
        not exceed the tier the wizard prices, or the two screens would disagree about
        the same download."""
        for tier in wizard.TIERS:
            if not tier.weights:
                continue
            with self.subTest(tier=tier.key):
                self.assertLessEqual(tiers.weights_size_mb(tier.weights),
                                     tier.download_mb)

    def test_the_classification_weighs_what_the_report_says(self):
        self.assertEqual(tiers.weights_size_mb(("ViT-L-14",)), 1600)

    def test_every_weight_a_line_names_is_carried_by_some_tier(self):
        """A part naming a model nobody installs would have no note and no way out — the
        F217 defect with the pairing done right and the catalog wrong."""
        for part in tiers.RUN_PARTS:
            for weight in part.weights:
                with self.subTest(part=part.key, weight=weight):
                    self.assertIsNotNone(tiers.weight_tier(weight))

    def test_every_stage_that_downloads_names_a_known_model(self):
        """The other half of the table — the one the run-time sentences are built from.
        A stage naming a model the catalog does not know would quote a size of zero."""
        for stage, weights in tiers.STAGE_WEIGHTS.items():
            for weight in weights:
                with self.subTest(stage=stage, weight=weight):
                    self.assertIn(weight, tiers._WEIGHT_MB)
                    self.assertIsNotNone(tiers.weight_tier(weight))


class TestWhatThisRunWillDownload(unittest.TestCase):
    """§2: the summary counts the stages nobody was asked about."""

    def _nothing_cached(self) -> list[tiers.TierState]:
        return tiers.tier_states(package_present=lambda _n: True,
                                 weights_cached=lambda _n: False)

    def _all_cached(self) -> list[tiers.TierState]:
        return tiers.tier_states(package_present=lambda _n: True,
                                 weights_cached=lambda _n: True)

    def test_the_stage_without_a_checkbox_is_in_the_summary(self):
        parts = {part.key: part for part in tiers.run_parts(self._nothing_cached())}
        classify = parts["classify"]
        self.assertFalse(classify.optional)          # nobody ticks it
        self.assertEqual(classify.missing, ("ViT-L-14",))
        self.assertEqual(classify.download_mb, 1600)

    def test_a_machine_that_has_the_weights_downloads_nothing(self):
        for part in tiers.run_parts(self._all_cached()):
            with self.subTest(part=part.key):
                self.assertEqual(part.missing, ())
                self.assertEqual(part.download_mb, 0)

    def test_the_summary_reads_the_same_probe_as_the_notes(self):
        """One probe, not two — the F217 rule this feature inherits and the reason
        `run_parts` takes the states rather than looking at the disk itself."""
        states = [tiers.TierState("search", missing_weights=("ViT-L-14",))]
        parts = {part.key: part for part in tiers.run_parts(states)}
        self.assertEqual(parts["landmarks"].missing, ("ViT-L-14",))
        # ...and a tier nobody probed is not called broken.
        self.assertTrue(parts["deep"].available)

    def test_a_tier_whose_packages_are_absent_is_not_available(self):
        states = [tiers.TierState("deep", missing_packages=("transformers",))]
        parts = {part.key: part for part in tiers.run_parts(states)}
        self.assertFalse(parts["deep"].available)
        self.assertFalse(parts["products"].available)
        # Weights are a different failure: they arrive by themselves on first use.
        weights_only = [tiers.TierState("faces", missing_weights=("buffalo_l",))]
        self.assertTrue({p.key: p for p in tiers.run_parts(weights_only)}["faces"]
                        .available)


class TestARefusalToDownloadIsWords(unittest.TestCase):
    """§4: what a person gets today is
    `<urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] ...>` — no stage, no model, no
    size, nothing to do about it."""

    def test_the_message_names_the_stage_the_model_and_the_size(self):
        for lang in _LANGS:
            with self.subTest(lang=lang):
                text = tiers.download_failure("landmarks", ("ViT-L-14",), lang,
                                              "SSL: CERTIFICATE_VERIFY_FAILED")
                self.assertIn("ViT-L-14", text)
                self.assertIn(tiers.stage_label("landmarks", lang), text)
                self.assertIn("1.6", text)
                self.assertIn("SSL: CERTIFICATE_VERIFY_FAILED", text)
                self.assertIn("sorta-setup", text)

    def test_the_notice_says_it_happens_once(self):
        for lang in _LANGS:
            with self.subTest(lang=lang):
                text = tiers.download_notice("classify", ("ViT-L-14",), lang)
                self.assertIn("ViT-L-14", text)
                self.assertIn(tiers.stage_label("classify", lang), text)
                self.assertIn("1.6", text)

    def test_a_stage_with_no_name_of_its_own_is_still_named(self):
        self.assertEqual(tiers.stage_label("phash", "en"), "phash")

    def test_the_stage_labels_exist_in_three_languages(self):
        for stage in tiers.STAGE_WEIGHTS:
            labels = {lang: tiers.stage_label(stage, lang) for lang in _LANGS}
            with self.subTest(stage=stage):
                self.assertEqual(len(set(labels.values())), 3, labels)


class TestTheOwnersRegressionIsNotTouched(ProcessTestBase):
    """The hard requirement of the brief. There is a live database of 26 137 frames with
    143 `visual` places in it, and switching the stage off may not cost one of them."""

    def _place(self, rel: str, confidence: str, city: str) -> int:
        file_id, _path, _content = self.add_photo_file(rel)
        self.conn.execute(
            """INSERT INTO places (file_id, country, region, city, confidence,
                   updated_at)
               VALUES (?, 'cz', NULL, ?, ?, '2026-08-01')""",
            (file_id, city, confidence))
        self.conn.commit()
        return int(file_id)

    def _places(self) -> list[tuple]:
        return [tuple(row) for row in self.conn.execute(
            "SELECT file_id, city, confidence, updated_at FROM places ORDER BY file_id")]

    def test_places_found_by_the_stage_survive_a_run_without_it(self):
        self._place("prague.jpg", "visual", "Prague")
        self._place("gps.jpg", "exact_gps", "Moscow")
        before = self._places()
        self.patch_fast_stages()
        self.start_server()
        status, _resp = self.post("/api/process", {"source_dir": str(self.src_dir)})
        self.assertEqual(status, 200)
        final = _poll_until(self.status, lambda d: d["finished"])
        self.assertIsNone(final["error"])
        self.assertNotIn("landmarks", self.calls)   # the stage really did not run
        self.assertEqual(self._places(), before)    # ...and nothing of it was undone

    def test_the_stage_only_ever_writes_where_geo_gave_up(self):
        """Why the case above holds by construction and not by luck: the selection of
        `detect_landmarks` is `places.confidence = 'unknown'`, so a row it has already
        marked `visual` is not in the population of the next run at all."""
        source = (_ROOT / "sorta" / "landmarks.py").read_text(encoding="utf-8")
        self.assertIn("AND p.confidence = 'unknown'", source)

    def test_a_config_that_switched_it_on_still_runs_it(self):
        """"config.yaml where landmarks are enabled works as before — with no edits by
        hand" — the brief, word for word."""
        self.cfg.features = dataclasses.replace(self.cfg.features, landmarks=True)
        self.start_server()
        _status, body, _ctype = self.get("/api/process/defaults")
        self.assertTrue(json.loads(body)["landmarks"])


class TestTheRunSummaryReachesTheBrowser(ProcessTestBase):
    """§2: the sum before the button, and the stage with no checkbox inside it."""

    def test_the_env_route_carries_the_lines_and_the_model_sizes(self):
        ui.process._gpu_present_cache_clear()
        self.addCleanup(ui.process._gpu_present_cache_clear)
        with mock.patch.object(ui.process, "nvidia_gpu_present", return_value=False):
            self.start_server()
            _status, body, _ctype = self.get("/api/env")
        data = json.loads(body)
        self.assertIn("landmarks", data["parts"])
        self.assertIn("classify", data["parts"])
        self.assertTrue(data["parts"]["classify"]["always"])
        self.assertFalse(data["parts"]["landmarks"]["always"])
        for weight, mb in data["weights"].items():
            with self.subTest(weight=weight):
                self.assertGreater(mb, 0)

    def test_the_summary_and_the_notes_read_one_probe(self):
        """Test 5 of the brief, as a guard. Two readings of "what is installed" answer
        differently within a release and nobody finds out from the code — the F211/F217
        lesson. `/api/env` builds the notes, the availability and the download summary
        from ONE `tier_states()` call.
        """
        with mock.patch.object(ui.process, "tier_states",
                               wraps=ui.process.tier_states) as probe:
            payload = ui.process._env_payload()
        self.assertEqual(probe.call_count, 1)
        self.assertTrue(payload["parts"])

    def test_the_animals_get_the_note_they_never_had(self):
        """Test 6, named after the case that exposed the whole defect: "why is there no
        warning at all on animal recognition?" Because its tier is called "Search by
        words" and nobody paired the two by hand."""
        nothing = tiers.tier_states(package_present=lambda _n: True,
                                    weights_cached=lambda _n: False)
        parts = ui.process._parts_payload(tiers.run_parts(nothing))
        self.assertEqual(parts["pets"]["tiers"], ["search"])
        self.assertEqual(parts["pets"]["missing"], ["ViT-L-14"])
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn('id="process-pets-tier-note"', html)
        self.assertIn('setPartNote("process-pets-tier-note", "pets"', html)

    def test_the_page_carries_the_summary_and_the_line_while_it_downloads(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        for element in ("process-download-total", "process-download-list",
                        "process-download-status"):
            with self.subTest(element=element):
                self.assertIn(f'id="{element}"', html)

    def test_the_summary_is_summed_by_model_and_not_by_line(self):
        """Landmarks, the animals and the classification raise ONE file. Three lines
        quoting 1.6 GB each would promise a download three times the real one."""
        source = _APP_JS.read_text(encoding="utf-8")
        self.assertIn("function renderDownloadPlan()", source)
        plan = source.split("function renderDownloadPlan()")[1].split("\n  }")[0]
        self.assertIn("forWhat[weight]", plan)

    def test_a_stage_that_downloads_says_so_while_it_does(self):
        """§3: the status carries what is being fetched — 1.6 GB with nothing on screen
        is what the owner reported as "it hung on landmarks"."""
        state = ui.process._ProcessState()
        self.assertIsNone(state.snapshot()["download"])
        state.set_download("landmarks", ("ViT-L-14",))
        snapshot = state.snapshot()
        self.assertEqual(snapshot["download"]["weights"], ["ViT-L-14"])
        self.assertEqual(snapshot["download"]["stage"], "landmarks")
        self.assertEqual(snapshot["download"]["mb"], 1600)
        state.set_download(None)
        self.assertIsNone(state.snapshot()["download"])

    def test_the_line_is_hung_on_the_factory_and_not_on_the_stage(self):
        """A run with nothing to do never builds a classifier, and announcing a download
        there would be a sentence about nothing."""
        seen: list[tuple] = []
        steps = dict(ui.process._pipeline_steps(
            lambda stage, weights: seen.append((stage, weights))))
        self.patch_fast_stages()
        steps["landmarks"](self.cfg, self.conn, lambda *_a, **_k: None)
        self.assertEqual(seen, [])


class TestAnUnavailableTierIsAnUnavailableCheckbox(ProcessTestBase):
    """§6b, which reverses one half of F217. "Explain, do not forbid" is right where the
    action is possible but unwise; here it is impossible — the tier is not on the machine,
    so the box changes nothing whatever it is set to, and F211 forbids a control that does
    nothing in as many words."""

    def test_the_missing_packages_make_the_line_unavailable(self):
        absent = [tiers.TierState("deep", missing_packages=("transformers",))]
        parts = ui.process._parts_payload(tiers.run_parts(absent))
        self.assertFalse(parts["deep"]["available"])

    def test_the_script_disables_by_the_probe_and_never_writes_checked(self):
        source = _APP_JS.read_text(encoding="utf-8")
        self.assertIn("function updateOptionAvailability()", source)
        block = source.split("function updateOptionAvailability()")[1].split("\n  }")[0]
        self.assertIn("partAvailable(key)", block)
        # The saved setting is not to be replaced by an empty one: somebody who ticked
        # deep analysis a month ago and wiped a cache keeps their answer.
        self.assertNotIn(".checked", block)

    def test_the_saved_setting_is_still_what_the_screen_shows(self):
        self.cfg.naming = dataclasses.replace(self.cfg.naming, vlm_enabled=True)
        self.start_server()
        _status, body, _ctype = self.get("/api/process/defaults")
        # The defaults route knows nothing about tiers — which is the point: a missing
        # tier cannot reach in and clear a person's answer.
        self.assertTrue(json.loads(body)["deep"])

    def test_the_reason_and_the_way_out_stand_next_to_it(self):
        for lang in _LANGS:
            with self.subTest(lang=lang):
                note = ui_strings._UI_STRINGS["tier_unavailable_note"][lang]
                self.assertTrue(note.strip())
        source = _APP_JS.read_text(encoding="utf-8")
        # ...built out of the doctor's own sentences, plus the wizard's way out.
        self.assertIn("I18N.tier_unavailable_note", source)
        self.assertIn("I18N.tier_add_hint", source)

    def test_the_tier_appearing_re_enables_it_without_a_restart(self):
        source = _APP_JS.read_text(encoding="utf-8")
        self.assertIn("function loadEnv()", source)
        self.assertIn("window.setInterval", source)

    def test_the_summary_does_not_promise_an_absent_tiers_download_either(self):
        """The same rule as the seconds: a checkbox left ticked for a tier that is not on
        the machine describes a run that will not happen, so neither its hours nor its
        gigabytes are quoted."""
        source = _APP_JS.read_text(encoding="utf-8")
        block = source.split("function partWillRun(")[1].split("\n  }")[0]
        self.assertIn("partAvailable(key)", block)

    def test_the_estimate_does_not_promise_an_absent_tiers_time(self):
        """§6: the deep tier's checkbox can be ticked with the tier missing, and
        `junk.classify` then falls back to the fast one — so the price shown has to be
        the price of the run that will happen."""
        source = _APP_JS.read_text(encoding="utf-8")
        master = source.split("function vlmMasterOn()")[1].split("}")[0]
        self.assertIn('partAvailable("deep")', master)


class TestASubOptionDiesWithItsStage(ProcessTestBase):
    """§7: the landmark check is a question about the landmark stage. With the stage out
    of the run there is nothing to check."""

    def test_the_check_is_a_child_of_the_stage_in_the_markup(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        row = html.split('id="process-landmarks-checkbox"')[1]
        self.assertIn('id="process-landmarks-verify-checkbox"',
                      row.split("</div>")[0])

    def test_the_script_switches_it_off_with_the_stage(self):
        source = _APP_JS.read_text(encoding="utf-8")
        self.assertIn("function landmarksStageOn()", source)
        self.assertIn("landmarks-off-hint", source)
        rows = source.split("var COST_ROWS")[1].split("];")[0]
        self.assertIn('parent: "process-landmarks-checkbox"', rows)

    def test_the_check_is_not_sent_for_a_run_without_the_stage(self):
        source = _APP_JS.read_text(encoding="utf-8")
        self.assertIn("landmarks_verify: landmarks", source)

    def test_the_other_deep_sub_options_belong_to_stages_that_always_run(self):
        """The brief asks for the rest of the deep tier's sub-options to be checked for
        the same defect. They are questions of `classify` and `junk`, which have no
        checkbox and always run — except the animal check, whose parent is the animal
        line and which was already handled by F138."""
        for key, parent in (("products", None), ("junk_rescue", None),
                            ("pets_verify", "process-pets-checkbox")):
            with self.subTest(option=key):
                self.assertIn(key, tiers.RUN_PARTS_BY_KEY)
                source = _APP_JS.read_text(encoding="utf-8")
                rows = source.split("var COST_ROWS")[1].split("];")[0]
                row = rows.split('key: "' + key + '"')[1].split("}")[0]
                if parent is None:
                    self.assertNotIn("parent:", row)
                else:
                    self.assertIn(parent, row)


class TestTheLandmarkOptionOnTheScreen(ProcessTestBase):
    """The checkbox itself, in the three languages the catalog carries."""

    def test_the_checkbox_and_its_price_are_on_the_page(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn('id="process-landmarks-checkbox"', html)
        self.assertIn('data-cost="landmarks"', html)
        self.assertIn('id="process-landmarks-tier-note"', html)

    def test_the_caption_exists_in_three_languages(self):
        for key in ("process_landmarks_label", "process_landmarks_hint",
                    "process_needs_landmarks_hint", "download_title", "download_line",
                    "download_none", "download_running", "download_always_part",
                    "tier_unavailable_note"):
            entry = ui_strings._UI_STRINGS[key]
            with self.subTest(key=key):
                self.assertEqual(set(entry), set(_LANGS))
                self.assertEqual(len(set(entry.values())), 3, entry)

    def test_the_hint_states_the_price_the_decision_was_made_on(self):
        for lang in _LANGS:
            with self.subTest(lang=lang):
                hint = ui_strings._UI_STRINGS["process_landmarks_hint"][lang]
                self.assertIn("143", hint)
                self.assertIn("1.6" if lang != "ru" else "1,6", hint)

    def test_the_run_is_priced_apart_from_the_stages_that_always_run(self):
        """§6 again, on the other side: the landmark minutes used to sit inside the
        "always" line, so a run without the stage was quoted its time anyway."""
        self.assertIn("landmarks", ui.process._RATE_UNITS)
        self.assertNotIn("landmarks", ui.process._RATE_UNITS["base"][0])
        self.assertIn("landmarks", ui.process._DEFAULT_RATES)


class TestTheWizardOffersTheComponent(unittest.TestCase):
    """§8: the weights belong to the tier called "Search by words", and the landmark
    stage raises the same file — so that tier NAMES the stage instead of a fifth tier
    being invented for one model the catalog already carries."""

    def test_no_tier_was_added_for_it(self):
        self.assertEqual([tier.key for tier in wizard.TIERS],
                         ["base", "faces", "search", "gpu", "deep"])

    def test_the_offer_names_what_the_stage_gives(self):
        for lang in _LANGS:
            with self.subTest(lang=lang):
                benefit = i18n.cli_text("cli.setup.tier.search.benefit", lang)
                without = i18n.cli_text("cli.setup.tier.search.without", lang)
                marker = {"ru": "мест", "en": "places", "ja": "場所"}[lang]
                self.assertIn(marker, benefit)
                self.assertIn(marker, without)


if __name__ == "__main__":
    unittest.main()
