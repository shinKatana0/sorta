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

import unittest
from pathlib import Path

from sorta import i18n, tiers, wizard
from sorta.config import FeaturesConfig, _features_from

_LANGS: tuple[i18n.Lang, ...] = ("ru", "en", "ja")
_ROOT = Path(__file__).resolve().parent.parent


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


if __name__ == "__main__":
    unittest.main()
