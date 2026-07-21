"""F52: log_level in config + configure_logging (level, idempotency, invalid input)."""
from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path

from sorta.config import configure_logging, load_config


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
        self.assertEqual(self.logger.level, logging.WARNING)

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
