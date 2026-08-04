"""F52: log_level in config + configure_logging (level, idempotency, invalid input)."""
from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path

from sorta.config import (
    VLM_QUALITY_SCOPES,
    FeaturesConfig,
    VlmConfig,
    configure_logging,
    load_config,
)


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

    def test_the_face_sharpness_threshold_is_read_and_defaults_to_the_measured_one(self):
        # F155: 200 is the row of the measurement where recall quadruples (62% against 15%
        # for the whole frame) at a comparable number of frames flagged. Provisional by
        # construction — 13 blurred frames in the sample — so it has to be overridable.
        self.assertAlmostEqual(FeaturesConfig().face_sharpness_max, 200.0)
        cfg = self._load("features:\n  face_sharpness_max: 350\n")
        self.assertAlmostEqual(cfg.features.face_sharpness_max, 350.0)
        self.assertAlmostEqual(
            self._load("features:\n  face_sharpness_max: sharp\n"
                       ).features.face_sharpness_max, 200.0)

    def test_the_restore_ceiling_is_a_setting_and_not_a_constant(self):
        # F169: the longer side a frame is scaled to before the x4 model — the one number
        # that decides whether a person gets their own detail back or a plausible
        # redrawing of it. It lived in `restore.py` until this feature, where nobody could
        # move it and nobody was told it existed.
        self.assertEqual(FeaturesConfig().restore_max_edge, 1024)
        self.assertEqual(self._load("features:\n  restore_max_edge: 2048\n"
                                    ).features.restore_max_edge, 2048)
        for garbage in ("wide", 0, -1, "true"):
            with self.subTest(value=garbage):
                self.assertEqual(
                    self._load(f"features:\n  restore_max_edge: {garbage}\n"
                               ).features.restore_max_edge, 1024)

    def test_a_quoted_false_does_not_switch_the_feature_on(self):
        self.assertFalse(self._load('features:\n  pets: "false"\n').features.pets)

    def test_garbage_falls_back_to_the_defaults(self):
        cfg = self._load("features:\n  pet_threshold: many\n  sharpness_max_edge: 0\n")
        d = FeaturesConfig()
        self.assertAlmostEqual(cfg.features.pet_threshold, d.pet_threshold)
        self.assertEqual(cfg.features.sharpness_max_edge, d.sharpness_max_edge)

    def test_an_empty_section_is_not_a_crash(self):
        self.assertEqual(self._load("features:\n").features, FeaturesConfig())

    def test_the_pet_thresholds_have_the_measured_defaults(self):
        # F158: the gate of the cascade is 0.30 — measured on 500 random hand-labelled
        # frames, where it marks 28 at 82% precision / 64% recall against 20 at 90% / 50%
        # for the 0.50 F130 shipped. The display threshold answers a different question
        # (who is labelled with no check at all) and did NOT move with it.
        d = FeaturesConfig()
        self.assertAlmostEqual(d.pet_candidate_threshold, 0.3)
        self.assertAlmostEqual(d.pet_threshold, 0.7)

    def test_the_pet_candidate_threshold_is_read_from_the_file(self):
        cfg = self._load("features:\n  pet_candidate_threshold: 0.45\n")
        self.assertAlmostEqual(cfg.features.pet_candidate_threshold, 0.45)
        # and garbage in it must not open the gate to the whole collection
        cfg = self._load("features:\n  pet_candidate_threshold: wide\n")
        self.assertAlmostEqual(cfg.features.pet_candidate_threshold,
                               FeaturesConfig().pet_candidate_threshold)

    def test_the_detector_numbers_have_the_measured_defaults(self):
        # F162: both are rows of the tables `scripts/measure_detector.py` prints, re-read
        # on 500 hand-labelled frames (36 animals) after F154 shipped numbers taken on 200.
        # The confidence: 0.60 marks 25 of 29 frames correctly where 0.50 marks 25 of 32 —
        # the same recall for three false marks fewer, so it dominates instead of trading.
        # The depth: 4 000 candidates is where the query's own recall ceiling reaches 100%
        # (83% at 2 000), and costs 5.6 minutes at the measured 83.8 ms per frame.
        cfg = self._load("")
        self.assertAlmostEqual(cfg.features.detector_threshold, 0.6)
        self.assertEqual(cfg.features.detector_candidates, 4000)

    def test_the_detector_numbers_are_read_from_the_file(self):
        cfg = self._load("features:\n"
                         "  detector_threshold: 0.45\n"
                         "  detector_candidates: 250\n")
        self.assertAlmostEqual(cfg.features.detector_threshold, 0.45)
        self.assertEqual(cfg.features.detector_candidates, 250)

    def test_garbage_detector_numbers_fall_back_to_the_defaults(self):
        # A depth read as 0 would show the detector nothing at all, and a threshold read
        # as 0 would make every box above the storage floor an animal.
        cfg = self._load("features:\n"
                         "  detector_threshold: confident\n"
                         "  detector_candidates: 0\n")
        d = FeaturesConfig()
        self.assertAlmostEqual(cfg.features.detector_threshold, d.detector_threshold)
        self.assertEqual(cfg.features.detector_candidates, d.detector_candidates)

    def test_the_face_slice_numbers_have_the_documented_defaults(self):
        # F152: geometry, not confidence — a count of face boxes and a share of the frame.
        d = FeaturesConfig()
        self.assertEqual(d.group_photo_faces, 3)
        self.assertAlmostEqual(d.portrait_face_share, 0.08)

    def test_the_face_slice_numbers_are_read(self):
        cfg = self._load(
            "features:\n"
            "  group_photo_faces: 5\n"
            "  portrait_face_share: 0.2\n")
        self.assertEqual(cfg.features.group_photo_faces, 5)
        self.assertAlmostEqual(cfg.features.portrait_face_share, 0.2)

    def test_garbage_face_slice_numbers_fall_back_to_the_defaults(self):
        # A zero group size would put every photograph into the group slice, and a
        # non-number share would be read as a 0 that makes every single face a portrait.
        cfg = self._load(
            "features:\n"
            "  group_photo_faces: 0\n"
            "  portrait_face_share: wide\n")
        d = FeaturesConfig()
        self.assertEqual(cfg.features.group_photo_faces, d.group_photo_faces)
        self.assertAlmostEqual(cfg.features.portrait_face_share, d.portrait_face_share)


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
        # F125: `faces` joins the three that existed, and the list is what the UI select
        # is built from — a value the config refuses must never be offered there.
        self.assertEqual(VLM_QUALITY_SCOPES, ("groups", "events", "faces", "all"))
        for scope in VLM_QUALITY_SCOPES:
            with self.subTest(scope=scope):
                self.assertEqual(
                    self._load(f"vlm:\n  quality_scope: {scope}\n").vlm.quality_scope,
                    scope)

    def test_the_new_scope_does_not_move_the_default(self):
        """F125 adds a value, not a policy: which population a collection wants stays the
        user's decision."""
        self.assertEqual(VlmConfig().quality_scope, "groups")
        self.assertEqual(self._load("vlm:\n  quality: true\n").vlm.quality_scope, "groups")

    def test_a_misspelled_scope_does_not_become_the_expensive_one(self):
        with self.assertLogs("sorta.config", level=logging.WARNING):
            cfg = self._load("vlm:\n  quality_scope: everything\n")
        self.assertEqual(cfg.vlm.quality_scope, "groups")

    def test_a_near_miss_of_the_new_scope_is_refused_too(self):
        """`face`, `faces_only`, `Faces ` — the plural and the spelling are the contract,
        and a typo must not quietly select a different population."""
        for typo in ("face", "faces_only"):
            with self.subTest(typo=typo):
                with self.assertLogs("sorta.config", level=logging.WARNING):
                    cfg = self._load(f"vlm:\n  quality_scope: {typo}\n")
                self.assertEqual(cfg.vlm.quality_scope, "groups")
        # case and surrounding space are normalized, as they always were
        self.assertEqual(
            self._load("vlm:\n  quality_scope: ' Faces '\n").vlm.quality_scope, "faces")


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
