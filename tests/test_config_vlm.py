"""F102: the `vlm:` config section, and the old `naming.*` keys that still feed it.

The knobs of the local VLM used to live in the `naming:` section because there was no
other address, and the one that decides what the pass costs — the input resolution —
was not in the config at all. This is the section that fixes that. What these tests
guard is not the shape of a dataclass but the promise made to a user who already has a
config.yaml: a file written before this feature keeps meaning exactly what it meant.

The load-bearing case is `naming.vlm_enabled: false` alone. Reading the new key, finding
nothing and taking the default would still be `False` today — but the same silence on
`naming.vlm_enabled: true` would leave the deep tier off, and the same shape of bug with
the toggle inverted switches a 20 GB model on for somebody who never asked. So the old
address is read, and it says so once per run rather than once per frame.
"""
from __future__ import annotations

import logging
import os
import tempfile
import unittest
from pathlib import Path

from sorta import config as config_mod
from sorta.config import (
    DEFAULT_VLM_MAX_EDGE,
    DEFAULT_VLM_MODEL,
    VlmConfig,
    default_vlm_workers,
    load_config,
    resolve_vlm_workers,
)


class VlmConfigCase(unittest.TestCase):
    """A config.yaml on disk — the loader is what is under test, not a dict."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.cfg_path = Path(self.tmp.name) / "config.yaml"
        # "Once per run" is process-wide state; every test starts from a clean slate.
        config_mod._ALIAS_WARNED.clear()
        self.addCleanup(config_mod._ALIAS_WARNED.clear)

    def load(self, body: str):
        self.cfg_path.write_text(body, encoding="utf-8")
        return load_config(str(self.cfg_path))


class TestNewSection(VlmConfigCase):
    """Test 1: the `vlm:` section alone."""

    def test_every_key_is_read(self):
        cfg = self.load(
            "vlm:\n"
            "  enabled: true\n"
            "  model: \"Qwen/Qwen2.5-VL-7B-Instruct\"\n"
            "  workers: 6\n"
            "  max_edge: 672\n"
        )
        self.assertEqual(
            (cfg.vlm.enabled, cfg.vlm.model, cfg.vlm.workers, cfg.vlm.max_edge),
            (True, "Qwen/Qwen2.5-VL-7B-Instruct", 6, 672))

    def test_the_toggle_reaches_the_field_the_stages_read(self):
        """`--deep` and the UI checkbox replace cfg.naming.vlm_enabled — it has to agree."""
        cfg = self.load("vlm:\n  enabled: true\n  model: Qwen/other\n")
        self.assertTrue(cfg.naming.vlm_enabled)
        self.assertEqual(cfg.naming.classify_vlm_model, "Qwen/other")

    def test_a_partial_section_keeps_the_defaults_of_the_rest(self):
        cfg = self.load("vlm:\n  max_edge: 448\n")
        self.assertEqual(cfg.vlm.max_edge, 448)
        self.assertFalse(cfg.vlm.enabled)
        self.assertEqual(cfg.vlm.model, DEFAULT_VLM_MODEL)


class TestLegacyKeys(VlmConfigCase):
    """Test 2: a config written before F102 works, unchanged, and says so once."""

    LEGACY = ("naming:\n"
              "  vlm_enabled: true\n"
              "  classify_vlm_model: Qwen/legacy\n"
              "  vlm_workers: 7\n")

    def test_old_keys_are_still_read(self):
        cfg = self.load(self.LEGACY)
        self.assertEqual(
            (cfg.vlm.enabled, cfg.vlm.model, cfg.vlm.workers),
            (True, "Qwen/legacy", 7))
        self.assertEqual(cfg.vlm.max_edge, DEFAULT_VLM_MAX_EDGE)

    def test_an_old_false_is_not_quietly_upgraded_to_the_new_default(self):
        """The reason the aliases exist: a `false` on the old key must stay a `false`."""
        cfg = self.load("naming:\n  vlm_enabled: false\n")
        self.assertFalse(cfg.vlm.enabled)
        self.assertFalse(cfg.naming.vlm_enabled)

    def test_the_legacy_fields_of_naming_keep_working_as_before(self):
        cfg = self.load(self.LEGACY)
        self.assertTrue(cfg.naming.vlm_enabled)
        self.assertEqual(cfg.naming.classify_vlm_model, "Qwen/legacy")

    def test_one_warning_per_key_per_run_not_one_per_load(self):
        with self.assertLogs("sorta.config", level=logging.WARNING) as caught:
            self.load(self.LEGACY)
            self.load(self.LEGACY)  # the web app reloads the config on every request
        warned = [line for line in caught.output if "vlm_enabled" in line]
        self.assertEqual(len(warned), 1)
        self.assertIn("vlm.enabled", warned[0])
        # ...and the same holds for each of the other two addresses.
        self.assertEqual(len([line for line in caught.output if "vlm_workers" in line]), 1)
        self.assertEqual(
            len([line for line in caught.output if "classify_vlm_model" in line]), 1)


class TestBothGiven(VlmConfigCase):
    """Test 3: the new key wins, and there is nothing to warn about."""

    BOTH = ("naming:\n"
            "  vlm_enabled: false\n"
            "  classify_vlm_model: Qwen/old\n"
            "  vlm_workers: 2\n"
            "vlm:\n"
            "  enabled: true\n"
            "  model: Qwen/new\n"
            "  workers: 5\n")

    def test_the_new_key_wins(self):
        cfg = self.load(self.BOTH)
        self.assertEqual((cfg.vlm.enabled, cfg.vlm.model, cfg.vlm.workers),
                         (True, "Qwen/new", 5))

    def test_no_deprecation_warning_when_the_new_key_is_present(self):
        with self.assertLogs("sorta.config", level=logging.WARNING) as caught:
            logging.getLogger("sorta.config").warning("anchor")  # assertLogs needs one
            self.load(self.BOTH)
        self.assertEqual(caught.output, ["WARNING:sorta.config:anchor"])

    def test_a_key_given_only_in_the_old_place_is_still_taken(self):
        """Mixed configs are the realistic case — half migrated, half not."""
        cfg = self.load("naming:\n  vlm_workers: 3\nvlm:\n  enabled: true\n")
        self.assertEqual((cfg.vlm.enabled, cfg.vlm.workers), (True, 3))


class TestDefaults(VlmConfigCase):
    """Test 4: neither address given — the documented defaults, nothing else."""

    def test_an_empty_config_gives_the_shipped_values(self):
        cfg = self.load("")
        self.assertFalse(cfg.vlm.enabled)
        self.assertEqual(cfg.vlm.model, DEFAULT_VLM_MODEL)
        self.assertEqual(cfg.vlm.max_edge, 896)
        self.assertEqual(cfg.vlm.workers, min(4, os.cpu_count() or 1))

    def test_the_default_resolution_is_the_one_that_shipped(self):
        """896 is not a taste: measure_vlm_resolution.py compares against it."""
        self.assertEqual(DEFAULT_VLM_MAX_EDGE, 896)
        self.assertEqual(VlmConfig().max_edge, 896)

    def test_a_hand_built_section_is_usable_without_a_config_file(self):
        """Tests and scripts build a VlmConfig directly — its workers must be a real number."""
        self.assertEqual(VlmConfig().workers, default_vlm_workers())
        self.assertGreaterEqual(VlmConfig().workers, 1)


class TestGarbageValues(VlmConfigCase):
    """Test 5: a typo in a number is a typo, not a crash and not a silent zero."""

    def test_a_string_where_a_number_belongs_falls_back(self):
        cfg = self.load("vlm:\n  max_edge: большой\n  workers: много\n")
        self.assertEqual(cfg.vlm.max_edge, DEFAULT_VLM_MAX_EDGE)
        self.assertEqual(cfg.vlm.workers, default_vlm_workers())

    def test_zero_and_negative_fall_back(self):
        for value in ("0", "-896"):
            with self.subTest(value=value):
                cfg = self.load(f"vlm:\n  max_edge: {value}\n  workers: {value}\n")
                self.assertEqual(cfg.vlm.max_edge, DEFAULT_VLM_MAX_EDGE)
                self.assertEqual(cfg.vlm.workers, default_vlm_workers())

    def test_a_list_where_a_number_belongs_falls_back(self):
        cfg = self.load("vlm:\n  max_edge: [896]\n  workers: [4]\n")
        self.assertEqual(cfg.vlm.max_edge, DEFAULT_VLM_MAX_EDGE)
        self.assertEqual(cfg.vlm.workers, default_vlm_workers())

    def test_a_quoted_false_does_not_switch_the_tier_on(self):
        """bool("false") is True in Python — which is how a heavy tier gets switched on."""
        cfg = self.load('vlm:\n  enabled: "false"\n')
        self.assertFalse(cfg.vlm.enabled)
        self.assertFalse(cfg.naming.vlm_enabled)

    def test_a_quoted_true_is_still_understood(self):
        self.assertTrue(self.load('vlm:\n  enabled: "true"\n').vlm.enabled)

    def test_unrecognizable_truth_keeps_the_tier_off(self):
        self.assertFalse(self.load("vlm:\n  enabled: [1]\n").vlm.enabled)

    def test_a_number_reads_as_truth_the_way_yaml_means_it(self):
        self.assertTrue(self.load("vlm:\n  enabled: 1\n").vlm.enabled)
        self.assertFalse(self.load("vlm:\n  enabled: 0\n").vlm.enabled)

    def test_a_boolean_where_a_size_belongs_falls_back(self):
        """`true` is an int in Python — it must not become a 1-pixel frame."""
        cfg = self.load("vlm:\n  max_edge: true\n  workers: true\n")
        self.assertEqual(cfg.vlm.max_edge, DEFAULT_VLM_MAX_EDGE)
        self.assertEqual(cfg.vlm.workers, default_vlm_workers())

    def test_an_empty_or_non_string_model_falls_back(self):
        for body in ('vlm:\n  model: ""\n', "vlm:\n  model: 42\n",
                     "vlm:\n  model: '   '\n"):
            with self.subTest(body=body.strip()):
                self.assertEqual(self.load(body).vlm.model, DEFAULT_VLM_MODEL)

    def test_a_section_that_is_not_a_mapping_is_not_a_crash(self):
        cfg = self.load("vlm: true\n")  # must not raise
        self.assertEqual(cfg.vlm, VlmConfig())

    def test_a_number_written_as_a_string_is_still_a_number(self):
        """A quoted 448 is a YAML accident, not garbage — the intent is unambiguous."""
        self.assertEqual(self.load('vlm:\n  max_edge: "448"\n').vlm.max_edge, 448)


class TestResolveVlmWorkers(unittest.TestCase):
    """The raw-dict resolver the measurement scripts share with load_config."""

    def setUp(self):
        config_mod._ALIAS_WARNED.clear()
        self.addCleanup(config_mod._ALIAS_WARNED.clear)

    def test_the_new_key_is_read(self):
        self.assertEqual(resolve_vlm_workers({"vlm": {"workers": 6}}), 6)

    def test_the_old_key_is_still_read(self):
        self.assertEqual(resolve_vlm_workers({"naming": {"vlm_workers": 6}}), 6)

    def test_the_new_key_wins(self):
        self.assertEqual(
            resolve_vlm_workers({"vlm": {"workers": 6}, "naming": {"vlm_workers": 2}}), 6)

    def test_absent_gives_the_default(self):
        default = default_vlm_workers()
        for raw in (None, {}, {"vlm": {}}, {"vlm": None}, {"naming": None}):
            with self.subTest(raw=raw):
                self.assertEqual(resolve_vlm_workers(raw), default)

    def test_garbage_gives_the_default(self):
        default = default_vlm_workers()
        for value in (0, -2, "many", [4], None):
            with self.subTest(value=value):
                self.assertEqual(resolve_vlm_workers({"vlm": {"workers": value}}), default)

    def test_one_is_the_serial_pass_and_is_respected(self):
        self.assertEqual(resolve_vlm_workers({"vlm": {"workers": 1}}), 1)


class TestExampleConfig(unittest.TestCase):
    """config.example.yaml is what a user copies — it must document the new address."""

    def setUp(self):
        config_mod._ALIAS_WARNED.clear()
        self.addCleanup(config_mod._ALIAS_WARNED.clear)

    def test_the_example_carries_the_vlm_section_at_its_defaults(self):
        example = Path(__file__).resolve().parent.parent / "config.example.yaml"
        cfg = load_config(str(example))
        self.assertEqual(cfg.vlm, VlmConfig())
        self.assertFalse(cfg.naming.vlm_enabled)

    def test_the_example_uses_the_new_address_and_warns_about_nothing(self):
        example = Path(__file__).resolve().parent.parent / "config.example.yaml"
        with self.assertLogs("sorta.config", level=logging.WARNING) as caught:
            logging.getLogger("sorta.config").warning("anchor")  # assertLogs needs one
            load_config(str(example))
        self.assertEqual(caught.output, ["WARNING:sorta.config:anchor"])


if __name__ == "__main__":
    unittest.main()
