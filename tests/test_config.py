"""F52: log_level in config + configure_logging (level, idempotency, invalid input)."""
from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path

from sorta.config import FeaturesConfig, configure_logging, load_config


class TestLogLevelConfig(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.cfg_path = Path(self.tmp.name) / "config.yaml"

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, body: str) -> None:
        self.cfg_path.write_text(body, encoding="utf-8")

    def test_default_is_warning(self):
        self._write("")
        cfg = load_config(str(self.cfg_path))
        self.assertEqual(cfg.log_level, "WARNING")

    def test_explicit_level_loaded(self):
        self._write("log_level: DEBUG\n")
        cfg = load_config(str(self.cfg_path))
        self.assertEqual(cfg.log_level, "DEBUG")


class TestExampleConfigLoads(unittest.TestCase):
    """config.example.yaml — what the user copies into config.yaml.
    It must load without errors and carry the current schema keys."""

    def _example_path(self) -> Path:
        return Path(__file__).resolve().parent.parent / "config.example.yaml"

    def test_example_loads(self):
        cfg = load_config(str(self._example_path()))
        # keys added after the initial template (F30/F35/F37-B/F44/F49/F56)
        self.assertEqual(cfg.events.trip_merge_gap_hours, 48)
        self.assertEqual(cfg.events.min_event_size, 5)
        self.assertFalse(cfg.naming.vlm_enabled)
        self.assertIsNone(cfg.sort.report_dir)
        self.assertTrue(cfg.sort.drop_unlocalized_district)
        # F113: the example documents the frame-quality section at its defaults — the
        # toggles off, so copying it changes nothing about how a run behaves.
        self.assertEqual(cfg.features, FeaturesConfig())
        self.assertFalse(cfg.vlm.quality)
        self.assertEqual(cfg.vlm.quality_scope, "groups")


class TestRawOnlyKeysDoNotCrash(unittest.TestCase):
    """Config sections may carry keys read directly from cfg.raw
    (faces.decode_workers) or future-phase keys — they must not break the section
    constructor, but must be kept in Config.raw."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.cfg_path = Path(self.tmp.name) / "config.yaml"

    def tearDown(self):
        self.tmp.cleanup()

    def test_faces_decode_workers_does_not_crash(self):
        self.cfg_path.write_text(
            "faces:\n  min_face_px: 40\n  decode_workers: 3\n", encoding="utf-8")
        cfg = load_config(str(self.cfg_path))
        self.assertEqual(cfg.faces.min_face_px, 40)
        self.assertEqual((cfg.raw.get("faces") or {}).get("decode_workers"), 3)

    def test_unknown_future_key_ignored(self):
        self.cfg_path.write_text("geo:\n  future_phase_option: 1\n", encoding="utf-8")
        cfg = load_config(str(self.cfg_path))  # must not raise
        self.assertEqual((cfg.raw.get("geo") or {}).get("future_phase_option"), 1)


class TestFeaturesSection(unittest.TestCase):
    """F113: `features:` — the frame-quality toggle and the thresholds it needs.

    Every value is garbage-tolerant for the same reason the `vlm:` section is: a typo in a
    config file is a typo, not a reason to refuse to start — and a threshold silently read
    as 0 would switch a feature on across a whole collection.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.cfg_path = Path(self.tmp.name) / "config.yaml"

    def _load(self, body: str):
        self.cfg_path.write_text(body, encoding="utf-8")
        return load_config(str(self.cfg_path))

    def test_absent_section_gives_the_defaults(self):
        cfg = self._load("")
        self.assertEqual(cfg.features, FeaturesConfig())
        self.assertFalse(cfg.features.pets)  # a new feature is off until asked for

    def test_values_are_read(self):
        cfg = self._load(
            "features:\n"
            "  pets: true\n"
            "  pet_threshold: 0.42\n"
            "  sharpness_max_edge: 640\n"
            "  sharpness_band_min: 10\n"
            "  sharpness_band_max: 500\n"
            "  subject_score_min: 0.75\n")
        self.assertTrue(cfg.features.pets)
        self.assertAlmostEqual(cfg.features.pet_threshold, 0.42)
        self.assertEqual(cfg.features.sharpness_max_edge, 640)
        self.assertAlmostEqual(cfg.features.sharpness_band_min, 10.0)
        self.assertAlmostEqual(cfg.features.sharpness_band_max, 500.0)
        self.assertAlmostEqual(cfg.features.subject_score_min, 0.75)

    def test_a_quoted_false_does_not_switch_the_feature_on(self):
        self.assertFalse(self._load('features:\n  pets: "false"\n').features.pets)

    def test_garbage_falls_back_to_the_defaults(self):
        cfg = self._load("features:\n  pet_threshold: many\n  sharpness_max_edge: 0\n")
        d = FeaturesConfig()
        self.assertAlmostEqual(cfg.features.pet_threshold, d.pet_threshold)
        self.assertEqual(cfg.features.sharpness_max_edge, d.sharpness_max_edge)

    def test_an_empty_section_is_not_a_crash(self):
        self.assertEqual(self._load("features:\n").features, FeaturesConfig())


class TestVlmQualityKeys(unittest.TestCase):
    """F113: `vlm.quality` / `vlm.quality_scope` — the band's own toggle and population."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.cfg_path = Path(self.tmp.name) / "config.yaml"

    def _load(self, body: str):
        self.cfg_path.write_text(body, encoding="utf-8")
        return load_config(str(self.cfg_path))

    def test_defaults_are_off_and_groups(self):
        cfg = self._load("")
        self.assertFalse(cfg.vlm.quality)
        self.assertEqual(cfg.vlm.quality_scope, "groups")

    def test_the_quality_toggle_is_separate_from_the_deep_tier(self):
        cfg = self._load("vlm:\n  quality: true\n")
        self.assertTrue(cfg.vlm.quality)
        self.assertFalse(cfg.vlm.enabled)  # the deep junk tier stays off

    def test_every_scope_is_accepted(self):
        for scope in ("groups", "events", "all"):
            with self.subTest(scope=scope):
                self.assertEqual(
                    self._load(f"vlm:\n  quality_scope: {scope}\n").vlm.quality_scope,
                    scope)

    def test_a_misspelled_scope_does_not_become_the_expensive_one(self):
        with self.assertLogs("sorta.config", level=logging.WARNING):
            cfg = self._load("vlm:\n  quality_scope: everything\n")
        self.assertEqual(cfg.vlm.quality_scope, "groups")


class TestConfigureLogging(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger("sorta")
        self._orig_level = self.logger.level
        self._orig_handlers = list(self.logger.handlers)

    def tearDown(self):
        self.logger.handlers = self._orig_handlers
        self.logger.setLevel(self._orig_level)

    def test_sets_level(self):
        configure_logging("DEBUG")
        self.assertEqual(self.logger.level, logging.DEBUG)

    def test_idempotent_single_handler(self):
        configure_logging("DEBUG")
        configure_logging("DEBUG")
        configure_logging("INFO")
        sorta_handlers = [h for h in self.logger.handlers if getattr(h, "_sorta_handler", False)]
        self.assertEqual(len(sorta_handlers), 1)

    def test_invalid_level_falls_back_to_warning(self):
        configure_logging("BOGUS")
        # The assertion moved from the logger to the console handler on purpose: the
        # logger is now deliberately lowered to the file sink's level so the INFO
        # `stage=` lines are not dropped before reaching it (they were). What "falls
        # back to WARNING" has to mean is what the console prints, and that is the
        # handler's level.
        console = [h for h in self.logger.handlers if getattr(h, "_sorta_handler", False)]
        self.assertTrue(console)
        for handler in console:
            self.assertEqual(handler.level, logging.WARNING)

    def test_invalid_level_does_not_raise(self):
        try:
            configure_logging("nonsense")
        except Exception as exc:  # pragma: no cover — the test should fail if this triggers
            self.fail(f"configure_logging raised an exception: {exc}")

    def test_case_insensitive(self):
        configure_logging("info")
        self.assertEqual(self.logger.level, logging.INFO)


if __name__ == "__main__":
    unittest.main()
