"""F161: product recognition is a line of the run screen, not a side effect of a master.

"Deep analysis (VLM)" was the master switch of every question the model is asked (F145)
AND, on its own, the thing that switched on the deep junk tier. That made it the only
option on the run screen with an effect nobody had named and a price nobody had stated:

* the effect is products. The deep tier is the ONLY producer of the `product` class —
  the fast tier never emits one — and on the live run of 2026-07-28 it moved 2 202 of
  its 2 592 changed verdicts into exactly that class;
* the price is between ~12 and ~95 minutes depending on whether `features.junk_rescue`
  narrows the candidates, and the checkbox stated neither.

So the tier gets `vlm.products`, a line under the master, and a price out of the same
estimate every other line is priced from. The master keeps the veto and nothing else.

What the cases below pin, in the order the brief asks for them:

1. with the master clear the product line is dead and priced at zero (the F145 rule);
2. the master alone runs no deep tier — with the product line clear the verdicts are the
   ones the fast tier writes, and no factory is called;
3. the price differs between a run with `features.junk_rescue` and one without;
4. a config that never heard of the key behaves exactly as it did — THE COMPATIBILITY
   CASE, and the reason the default is `true` where every other subordinate key is
   `false`;
5. no caption names the technology where it can name the result;
6. the key is written up in all three guides (the F115 watchdog covers the table; this
   file also requires the sentence a reader needs).

Test 7 of the brief — the caret in the path field on an empty collection — lives with
the assertion it restores, in test_ui_three_layers.py.
"""
from __future__ import annotations

import dataclasses
import json
import re
import tempfile
import unittest
from pathlib import Path

from sorta import junk, ui
from sorta.config import (
    Config,
    VlmConfig,
    _naming_from,
    load_config,
    products_allowed,
    vlm_allowed,
)
from sorta.junk import classify

from tests.test_docs_guides import GUIDES
from tests.test_junk import NO_OCR
from tests.test_junk_tier import TierTestBase
from tests.test_pets_cascade import PetClassifier
from tests.test_ui_run_costs import RunCostsTestBase

_ROOT = Path(__file__).resolve().parent.parent


class TestTheKeyItself(unittest.TestCase):
    """`vlm.products` and the helper that reads it against the master switch."""

    def test_the_default_is_on_because_that_is_what_yesterday_did(self):
        """The one subordinate key defaulting to true, and the whole of requirement 4:
        before it existed, `vlm.enabled` alone meant the deep tier."""
        self.assertTrue(VlmConfig().products)

    def test_the_key_is_read_off_the_section(self):
        cfg = Config(vlm=VlmConfig(products=False))
        self.assertFalse(cfg.vlm.products)

    def test_the_master_keeps_the_veto(self):
        """Permission is required and permission is not an instruction: both halves."""
        cases = {(True, True): True, (True, False): False,
                 (False, True): False, (False, False): False}
        for (master, products), expected in cases.items():
            with self.subTest(master=master, products=products):
                cfg = Config(naming=_naming_from({"vlm_enabled": master}),
                             vlm=VlmConfig(enabled=master, products=products))
                self.assertIs(products_allowed(cfg), expected)
                self.assertIs(vlm_allowed(cfg), master)

    def test_a_config_that_never_heard_of_the_key_asks_about_products(self):
        """A measurement script's hand-built settings object included — `getattr` with
        a default of True, not False, or an old caller would quietly lose the tier."""
        class Old:
            naming = _naming_from({"vlm_enabled": True})

        self.assertTrue(products_allowed(Old()))  # type: ignore[arg-type]


class TestTheFileIsReadAsWritten(unittest.TestCase):
    """The YAML side: absent, false, and garbage."""

    def load(self, body: str, tmp: Path) -> Config:
        path = tmp / "config.yaml"
        path.write_text(body, encoding="utf-8")
        return load_config(str(path))

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_a_file_without_the_key_keeps_the_tier(self):
        self.assertTrue(self.load("vlm:\n  enabled: true\n", self.root).vlm.products)

    def test_the_key_switches_the_line_off(self):
        cfg = self.load("vlm:\n  enabled: true\n  products: false\n", self.root)
        self.assertFalse(cfg.vlm.products)
        self.assertTrue(cfg.vlm.enabled)

    def test_a_quoted_false_is_still_false(self):
        """bool("false") is True in Python, which is how a 20 GB tier stays on."""
        cfg = self.load('vlm:\n  products: "false"\n', self.root)
        self.assertFalse(cfg.vlm.products)

    def test_garbage_keeps_the_default(self):
        for value in ("[1]", "perhaps"):
            with self.subTest(value=value):
                self.assertTrue(
                    self.load(f"vlm:\n  products: {value}\n", self.root).vlm.products)


class ProductTierCase(TierTestBase):
    """One candidate frame the deep tier would call a product, and a counted factory."""

    def setUp(self):
        super().setUp()
        self.calls: list[str] = []

    def products(self, enabled: bool) -> None:
        self.cfg.vlm = dataclasses.replace(self.cfg.vlm, products=enabled)

    def vlm(self, path: str) -> str:
        self.calls.append(path)
        return "product"

    def run_classify(self, **kwargs):
        clf = self.candidate_clf("shoe.jpg")
        return classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR,
                        **kwargs)


class TestTheTierFollowsItsOwnKey(ProductTierCase):
    """Brief tests 1, 2 and 4 at the stage that does the work."""

    def test_the_master_alone_does_not_run_it(self):
        """Requirement 2. The master grants permission; with the product line clear the
        verdict is the one the fast tier wrote and the model was never asked."""
        fid = self.add_file("shoe.jpg", camera_make=None, camera_model=None)
        self.enable_vlm()
        self.products(False)
        self.run_classify(vlm_classifier=self.vlm)
        self.assertEqual(self.calls, [])
        row = self.media_class(fid)
        self.assertEqual((row["verdict"], row["source"], row["tier"]),
                         ("photo", "clip", "clip"))

    def test_no_factory_is_built_either(self):
        """"The model answered nothing" and "the model was never built" differ by five
        seconds and several gigabytes — so the count, not the verdict, is the assertion."""
        self.add_file("shoe.jpg", camera_make=None, camera_model=None)
        self.enable_vlm()
        self.products(False)
        built: list[str] = []
        self.run_classify(vlm_classifier_factory=lambda name: (
            built.append(name) or (lambda _path: "product")))
        self.assertEqual(built, [])

    def test_the_line_is_dead_without_the_master(self):
        """Requirement 1, the F145 rule: a subordinate key raises nothing by itself."""
        fid = self.add_file("shoe.jpg", camera_make=None, camera_model=None)
        self.products(True)  # and the master left off
        self.run_classify(vlm_classifier=self.vlm)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.media_class(fid)["tier"], "clip")

    def test_with_both_on_the_tier_is_the_one_it_always_was(self):
        """Requirement 4: the default state of a config that never heard of the key."""
        fid = self.add_file("shoe.jpg", camera_make=None, camera_model=None)
        self.enable_vlm()
        self.assertTrue(self.cfg.vlm.products)  # nobody set it — this is the default
        self.run_classify(vlm_classifier=self.vlm)
        self.assertEqual(self.calls, ["/photos/shoe.jpg"])
        row = self.media_class(fid)
        self.assertEqual((row["verdict"], row["source"], row["tier"]),
                         ("product", "vlm", "vlm"))

    def test_the_verdicts_are_the_ones_a_run_without_the_master_writes(self):
        """Not merely "no products" — the same row a master-off run produces, which is
        what makes "the master does nothing by itself" a statement about the result."""
        fid = self.add_file("shoe.jpg", camera_make=None, camera_model=None)
        self.enable_vlm()
        self.products(False)
        self.run_classify(vlm_classifier=self.vlm)
        with_master_on = tuple(self.media_class(fid))

        self.conn.execute("DELETE FROM media_class")
        self.conn.commit()
        self.enable_vlm(False)
        self.products(True)
        self.run_classify(vlm_classifier=self.vlm)
        self.assertEqual(with_master_on, tuple(self.media_class(fid)))

    def test_switching_the_line_off_reclassifies_the_rows(self):
        """The tier marker has to follow the line, or a run started without products
        would keep serving the verdicts of the run that had them (F68)."""
        fid = self.add_file("shoe.jpg", camera_make=None, camera_model=None)
        self.enable_vlm()
        self.run_classify(vlm_classifier=self.vlm)
        self.assertEqual(self.media_class(fid)["verdict"], "product")

        self.products(False)
        stats = self.run_classify(vlm_classifier=self.vlm)
        self.assertEqual(stats.processed, 1)
        self.assertEqual(self.media_class(fid)["verdict"], "photo")


class TestTheOtherQuestionsAreUntouched(ProductTierCase):
    """A boundary: `vlm.products` is the deep TIER, not the model.

    Clearing it must not take the four F145 questions down with it — they have their own
    keys and the master above them is still on.
    """

    def test_the_animal_check_still_runs_with_the_product_line_off(self):
        self.add_file("cat.jpg", camera_make=None, camera_model=None)
        self.enable_vlm()
        self.products(False)
        self.cfg.features = dataclasses.replace(
            self.cfg.features, pets=True, pets_verify=True, pet_threshold=0.7,
            pet_candidate_threshold=0.3)
        asked: list[str] = []
        classify(self.cfg, self.conn, classifier=PetClassifier({"cat.jpg": 0.95}),
                 text_detector=NO_OCR,
                 pet_vlm=lambda path: asked.append(path) or "real")
        self.assertEqual(asked, ["/photos/cat.jpg"])


class TestTheLineOnTheScreen(unittest.TestCase):
    """The markup and the script: a line, a price, and a master that costs nothing."""

    @classmethod
    def setUpClass(cls):
        cls.html = ui._render_index_html("en")

    def options(self) -> str:
        return self.html.split('id="step-options"', 1)[1].split('id="step-actions"', 1)[0]

    def test_the_line_stands_under_the_master_with_a_price_of_its_own(self):
        block = self.options().split('id="process-deep-checkbox"', 1)[1]
        row = block.split('class="cost-row"', 1)[0]
        self.assertIn('id="process-products-checkbox"', row)
        self.assertIn('data-cost="products"', row)
        self.assertIn('class="cost-child" id="process-products-row"', row)

    def test_it_is_visible_even_when_it_is_dead(self):
        """A hidden option reads as "there is no such feature", and products are the
        feature people came for. The animal check hides (its parent question is not
        being asked at all); this one goes grey."""
        row = self.options().split('id="process-products-row"', 1)[1][:60]
        self.assertNotIn("display:none", row)

    def test_the_master_says_its_own_price_is_zero(self):
        self.assertIn("row.master ? I18N.costs_permission_only", self.html)
        self.assertIn("if (row.master) return 0;", self.html)
        self.assertIn("{ key: \"deep\", id: \"process-deep-checkbox\", master: true }",
                      self.html)

    def test_the_box_is_sent_with_the_run_and_started_from_the_config(self):
        self.assertIn('products: document.getElementById("process-products-checkbox")'
                      ".checked,", self.html)
        self.assertIn('document.getElementById("process-products-checkbox").checked ='
                      " !!data.products;", self.html)


class TestTheCaptionsNameTheResult(unittest.TestCase):
    """Brief test 5. A person ticking a box is buying an outcome, not a technology."""

    RESULT = {"ru": "товар", "en": "product", "ja": "商品"}
    TECHNOLOGY = ("vlm", "clip", "qwen", "нейросет", "глубок", "deep analysis",
                  "詳細解析", "詳細分析")

    def test_the_label_is_the_outcome_in_every_language(self):
        for lang, word in self.RESULT.items():
            with self.subTest(lang=lang):
                label = ui._t("process_products_label", lang)
                self.assertIn(word, label.lower())
                for term in self.TECHNOLOGY:
                    self.assertNotIn(term, label.lower())

    def test_the_hint_says_where_the_outcome_shows_up(self):
        """Two places, and the third sentence is the one that decides: without the line
        products are not few, there are none."""
        for lang, word in self.RESULT.items():
            with self.subTest(lang=lang):
                hint = ui._t("process_products_hint", lang).lower()
                self.assertIn(word, hint)
                for term in self.TECHNOLOGY:
                    self.assertNotIn(term, hint)

    def test_the_master_no_longer_promises_time_it_does_not_take(self):
        """It used to open with "Slower". It is not slower — it is nothing at all, and
        the lines it unlocks state their own price."""
        for lang, promise in (("ru", "медленнее"), ("en", "slower"), ("ja", "遅く")):
            with self.subTest(lang=lang):
                self.assertNotIn(promise, ui._t("process_deep_hint", lang).lower())

    def test_every_new_string_exists_in_all_three_languages(self):
        for key in ("process_products_label", "process_products_hint",
                    "costs_permission_only", "process_deep_hint"):
            with self.subTest(key=key):
                entry = ui._UI_STRINGS[key]
                self.assertEqual(set(entry), {"ru", "en", "ja"})
                for lang, value in entry.items():
                    self.assertTrue(value.strip(), f"{key}/{lang} is empty")


class TestThePrice(RunCostsTestBase):
    """Requirements 1 and 2 of the brief: a price, and the right one of the two."""

    def estimate(self) -> dict:
        status, body, _ctype = self.get("/api/process/estimate")
        self.assertEqual(status, 200)
        return json.loads(body)

    def answered_by_the_tier(self, count: int) -> list[int]:
        ids = []
        for i in range(count):
            file_id, _p, _c = self.add_photo_file(f"p{i}.jpg")
            self.conn.execute(
                "INSERT INTO media_class (file_id, verdict, source, tier, updated_at)"
                " VALUES (?, 'photo', 'vlm', 'vlm', '2026-01-01')", (file_id,))
            ids.append(file_id)
        self.conn.commit()
        return ids

    def score(self, file_id: int, value: float) -> None:
        self.conn.execute(
            "INSERT INTO frame_quality (file_id, junk_score, source, updated_at)"
            " VALUES (?, ?, 'clip', '2026-01-01')", (file_id, value))
        self.conn.commit()

    def rescue(self, enabled: bool, threshold: float = 0.02) -> None:
        self.cfg.features = dataclasses.replace(
            self.cfg.features, junk_rescue=enabled, junk_rescue_threshold=threshold)

    def test_the_whole_pass_is_priced_when_nothing_narrows_it(self):
        self.answered_by_the_tier(3)
        self.start_server()
        data = self.estimate()
        self.assertEqual(data["counts"]["products"], 3)
        self.assertAlmostEqual(data["seconds"]["products"],
                               round(3 * ui._SEC_PER_VLM_FRAME, 1))

    def test_the_f140_selection_changes_the_price(self):
        """Requirement 2: the estimate has to show the price of the run that WILL
        happen, and with the selection on that is the band, not the pass."""
        ids = self.answered_by_the_tier(3)
        for file_id, value in zip(ids, (0.5, 0.03, -0.1)):
            self.score(file_id, value)
        self.start_server()
        whole_pass = self.estimate()

        self.rescue(True)
        narrowed = self.estimate()
        self.assertEqual(whole_pass["counts"]["products"], 3)
        self.assertEqual(narrowed["counts"]["products"], 2)
        self.assertLess(narrowed["seconds"]["products"],
                        whole_pass["seconds"]["products"])

    def test_the_threshold_is_the_one_in_the_config(self):
        ids = self.answered_by_the_tier(3)
        for file_id, value in zip(ids, (0.5, 0.03, -0.1)):
            self.score(file_id, value)
        self.rescue(True, threshold=0.4)
        self.start_server()
        self.assertEqual(self.estimate()["counts"]["products"], 1)

    def test_a_collection_nobody_has_scored_yet_says_so(self):
        """A dash, not a zero: with the selection on and no `junk_score` anywhere, the
        population is unknown rather than empty."""
        self.answered_by_the_tier(2)
        self.rescue(True)
        self.start_server()
        data = self.estimate()
        self.assertIsNone(data["counts"]["products"])
        self.assertIsNone(data["seconds"]["products"])

    def test_the_master_line_costs_nothing_at_all(self):
        """Requirement: its own price is zero. Not a dash — a dash means "unknown", and
        what a switch that only grants permission costs is known exactly."""
        self.answered_by_the_tier(2)
        self.start_server()
        data = self.estimate()
        self.assertEqual(data["seconds"]["deep"], 0.0)
        self.assertEqual(data["sources"]["deep"], ui._RATE_FIXED)

    def test_the_cache_is_keyed_on_the_selection(self):
        """The two prices differ by a factor of eight — serving one for the other is the
        estimate lying, which is the one thing this screen may not do."""
        ids = self.answered_by_the_tier(2)
        for file_id in ids:
            self.score(file_id, 0.5)
        self.start_server()
        first = self.estimate()["counts"]["products"]
        self.rescue(True, threshold=0.9)
        self.assertEqual(first, 2)
        self.assertEqual(self.estimate()["counts"]["products"], 0)


class TestTheRunTakesTheBox(RunCostsTestBase):
    """The override reaches the stage and does not touch config.yaml."""

    def test_an_unticked_box_forces_the_tier_off_for_this_run(self):
        self.cfg.vlm = dataclasses.replace(self.cfg.vlm, products=True)
        run_cfg = self.run_once({"deep": True, "products": False})
        self.assertFalse(run_cfg.vlm.products)
        # ...and the master is untouched: permission is a separate decision
        self.assertTrue(run_cfg.naming.vlm_enabled)

    def test_a_body_without_it_leaves_the_config_alone(self):
        """Requirement 4 over HTTP: `/api/process/rerun-optional` and every caller
        outside the browser send no `products`, and must keep getting the tier."""
        run_cfg = self.run_once({"deep": True})
        self.assertTrue(run_cfg.vlm.products)

    def test_the_defaults_route_answers_from_the_config(self):
        self.cfg.vlm = dataclasses.replace(self.cfg.vlm, products=False)
        self.start_server()
        _status, body, _ctype = self.get("/api/process/defaults")
        self.assertIs(json.loads(body)["products"], False)

    def test_the_default_of_a_file_that_never_heard_of_it_is_on(self):
        self.start_server()
        _status, body, _ctype = self.get("/api/process/defaults")
        self.assertIs(json.loads(body)["products"], True)

    def test_a_non_boolean_is_refused(self):
        self.patch_fast_stages()
        self.start_server()
        status, _resp = self.post("/api/process", {"source_dir": str(self.src_dir),
                                                   "products": "yes"})
        self.assertEqual(status, 400)

    def test_the_file_is_not_rewritten(self):
        before = self.config_path.read_text(encoding="utf-8")
        self.run_once({"deep": True, "products": False})
        self.assertEqual(self.config_path.read_text(encoding="utf-8"), before)


class TestTheGuidesSayWhatItGives(unittest.TestCase):
    """Brief test 6. The F115 watchdog requires `vlm.products` to appear at all; what a
    reader needs on top of that is the sentence saying the slice is EMPTY without it."""

    def test_the_key_is_named_where_the_run_is_described_and_where_it_is_configured(self):
        """Twice: the run screen's list of checkboxes (§8) and the `vlm:` table (§21).
        The F115 watchdog is satisfied by the table alone, and somebody deciding whether
        to tick a box is not reading the table."""
        for lang, path in GUIDES.items():
            with self.subTest(lang=lang):
                self.assertGreaterEqual(
                    path.read_text(encoding="utf-8").count("vlm.products"), 2)

    def test_the_example_config_carries_the_key(self):
        text = (_ROOT / "config.example.yaml").read_text(encoding="utf-8")
        self.assertIn("products: true", text)

    def test_the_master_row_describes_permission_rather_than_a_tier(self):
        """Requirement 5 in the reference table: `vlm.enabled` used to be written up as
        "turns the deep tier on for good", which is now the row below it."""
        for lang, word in (("ru", "азреша"), ("en", "permit"), ("ja", "許可")):
            with self.subTest(lang=lang):
                rows = [line for line in GUIDES[lang].read_text(
                    encoding="utf-8").splitlines()
                    if line.startswith("| `vlm.enabled` |")]
                self.assertEqual(len(rows), 1)
                self.assertIn(word, rows[0])

    def test_every_guide_says_the_slice_is_empty_without_it(self):
        """The fact a reader cannot infer: this is not "fewer products", it is none —
        the fast tier does not produce the class at all."""
        for lang, word in (("ru", "ноль"), ("en", "none"), ("ja", "ゼロ")):
            with self.subTest(lang=lang):
                text = GUIDES[lang].read_text(encoding="utf-8")
                paragraph = text.split("vlm.products", 1)[1][:1200]
                self.assertIn(word, paragraph)


class TestTheDeepTierIsStillNamedInJunk(unittest.TestCase):
    """The gate is one line in `junk.py` and it is the shared helper, not a second rule."""

    def test_the_tier_gate_reads_the_helper(self):
        import inspect
        src = inspect.getsource(junk.classify)
        self.assertIn("if use_clip and products_allowed(cfg):", src)
        self.assertIsNone(re.search(r"if use_clip and vlm_on:", src))


if __name__ == "__main__":
    unittest.main()
