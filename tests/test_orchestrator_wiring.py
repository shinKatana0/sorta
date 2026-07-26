"""Wiring added by the orchestrator on top of F65/F67/F68/F69.

Each of those features shipped a mechanism but was forbidden from touching the
shared entry points (cli.py/config.py/sorter.py). These tests cover the seams.
"""
from __future__ import annotations

import os
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from sorta import imaging, landmarks
from sorta.config import _apply_imaging_config


class TestLandmarksFileResolution(unittest.TestCase):
    """F65 follow-up: landmarks.yaml had the same CWD-relative trap as the geo base."""

    def test_bundled_list_is_found_without_configuration(self):
        resolved = landmarks.resolve_landmarks_file(landmarks.DEFAULT_LANDMARKS_FILE)
        self.assertTrue(resolved.exists())
        self.assertGreater(len(landmarks.load_landmarks(resolved)), 0)

    def test_default_resolves_from_an_unrelated_working_directory(self):
        """The actual regression: the default only ever worked from the repo root."""
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                resolved = landmarks.resolve_landmarks_file(
                    landmarks.DEFAULT_LANDMARKS_FILE)
                self.assertTrue(resolved.exists())
            finally:
                os.chdir(original)

    def test_existing_custom_path_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            custom = Path(tmp) / "mine.yaml"
            custom.write_text("landmarks: []", encoding="utf-8")
            self.assertEqual(landmarks.resolve_landmarks_file(custom), custom)

    def test_missing_custom_path_raises_instead_of_falling_back(self):
        """Silently swapping in our list would be indistinguishable from the config
        having been applied — exactly the failure mode F65 was about."""
        with self.assertRaises(FileNotFoundError):
            landmarks.resolve_landmarks_file("no/such/list.yaml")

    def test_empty_value_uses_the_bundled_list(self):
        self.assertTrue(landmarks.resolve_landmarks_file("").exists())


class TestImagingConfigSection(unittest.TestCase):
    """F67 follow-up: the `imaging:` config section seeds the SORTA_PREVIEW_* vars."""

    def test_values_are_applied_to_the_environment(self):
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            _apply_imaging_config({"preview_max_edge": 2048, "preview_quality": 70})
            self.assertEqual(os.environ[imaging.ENV_PREVIEW_MAX_EDGE], "2048")
            self.assertEqual(os.environ[imaging.ENV_PREVIEW_QUALITY], "70")
            self.assertEqual(imaging.preview_max_edge(), 2048)

    def test_booleans_become_the_flag_form_imaging_expects(self):
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            _apply_imaging_config({"preview_cache": False})
            self.assertEqual(os.environ[imaging.ENV_PREVIEW_CACHE], "0")
            self.assertFalse(imaging.preview_cache_enabled())

    def test_environment_wins_over_the_config_file(self):
        """The documented contract: env is an override, not a default."""
        with unittest.mock.patch.dict(
                os.environ, {imaging.ENV_PREVIEW_MAX_EDGE: "512"}, clear=True):
            _apply_imaging_config({"preview_max_edge": 2048})
            self.assertEqual(os.environ[imaging.ENV_PREVIEW_MAX_EDGE], "512")

    def test_absent_keys_touch_nothing(self):
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            _apply_imaging_config({})
            self.assertNotIn(imaging.ENV_PREVIEW_DIR, os.environ)


class TestVideoConfigSection(unittest.TestCase):
    """F74/F80 follow-up: the video keys config.example.yaml documents are now read.

    They lived in imaging as env vars only, so `video_frames: 3` in the config file
    was silently ignored — the lightbox kept paging six frames.
    """

    def test_video_keys_reach_imaging(self):
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            _apply_imaging_config({"video_previews": False, "video_workers": 2,
                                   "video_frames": 3})
            self.assertEqual(os.environ[imaging.ENV_VIDEO_PREVIEWS], "0")
            self.assertEqual(os.environ[imaging.ENV_VIDEO_WORKERS], "2")
            self.assertEqual(os.environ[imaging.ENV_VIDEO_FRAMES], "3")
            self.assertFalse(imaging.video_previews_enabled())
            self.assertEqual(imaging.video_workers(), 2)
            self.assertEqual(imaging.video_frames(), 3)

    def test_environment_still_wins(self):
        with unittest.mock.patch.dict(
                os.environ, {imaging.ENV_VIDEO_FRAMES: "1"}, clear=True):
            _apply_imaging_config({"video_frames": 6})
            self.assertEqual(imaging.video_frames(), 1)

    def test_defaults_hold_when_the_section_says_nothing(self):
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            _apply_imaging_config({})
            self.assertNotIn(imaging.ENV_VIDEO_FRAMES, os.environ)
            self.assertEqual(imaging.video_frames(), imaging.VIDEO_FRAMES)


if __name__ == "__main__":
    unittest.main()
