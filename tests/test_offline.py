"""Model loading must not reach for the network once the weights are on disk.

Observed live: every stage that touches open_clip/easyocr/transformers printed
"You are sending unauthenticated requests to the HF Hub" — huggingface_hub calling
out to check a revision for weights that were already cached. Verified separately
that CLIP loads fine with HF_HUB_OFFLINE=1, so the call bought nothing.
"""
from __future__ import annotations

import os
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from sorta import offline


class TestConfigureModelOffline(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cache = Path(self.tmp.name) / "hub"
        self.cache.mkdir()

    def _populate(self):
        """A FINISHED download: since 2026-08-09 a bare directory is not one, because an
        interrupted fetch leaves exactly that and used to switch the machine offline."""
        snapshot = (self.cache / "models--timm--vit_large_patch14_clip_224.openai"
                    / "snapshots" / "abc123")
        snapshot.mkdir(parents=True)
        (snapshot / "open_clip_model.safetensors").write_bytes(b"x" * 16)

    def test_offline_is_enabled_when_the_cache_holds_models(self):
        self._populate()
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            self.assertTrue(offline.configure_model_offline(self.cache))
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
            self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "1")

    def test_empty_cache_leaves_the_first_download_possible(self):
        """A fresh machine has to be able to fetch the weights once."""
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(offline.configure_model_offline(self.cache))
            self.assertNotIn("HF_HUB_OFFLINE", os.environ)

    def test_missing_cache_directory_is_not_an_error(self):
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(offline.configure_model_offline(self.cache / "nope"))

    def test_explicit_user_setting_is_never_overridden(self):
        self._populate()
        with unittest.mock.patch.dict(os.environ, {"HF_HUB_OFFLINE": "0"}, clear=True):
            self.assertFalse(offline.configure_model_offline(self.cache))
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "0")

    def test_escape_hatch_disables_the_whole_thing(self):
        self._populate()
        for value in ("1", "true", "YES", "on"):
            with unittest.mock.patch.dict(
                    os.environ, {offline.ENV_ALLOW_DOWNLOAD: value}, clear=True):
                self.assertFalse(offline.configure_model_offline(self.cache))
                self.assertNotIn("HF_HUB_OFFLINE", os.environ)


class TestCacheDirResolution(unittest.TestCase):
    def test_hf_hub_cache_wins(self):
        with unittest.mock.patch.dict(os.environ, {"HF_HUB_CACHE": r"D:/models"}, clear=True):
            self.assertEqual(offline.hf_cache_dir(), Path(r"D:/models"))

    def test_hf_home_is_used_with_the_hub_suffix(self):
        with unittest.mock.patch.dict(os.environ, {"HF_HOME": r"D:/hf"}, clear=True):
            self.assertEqual(offline.hf_cache_dir(), Path(r"D:/hf") / "hub")

    def test_default_is_the_user_cache(self):
        # Only the HF variables are removed, not the whole environment: Path.home()
        # needs USERPROFILE/HOME and raises RuntimeError without them.
        # patch.dict snapshots the mapping and restores it on exit, so popping inside
        # the block is safe (passing None as a value would not be — os.environ only
        # accepts strings).
        with unittest.mock.patch.dict(os.environ, {}, clear=False):
            for var in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME"):
                os.environ.pop(var, None)
            self.assertEqual(offline.hf_cache_dir(),
                             Path.home() / ".cache" / "huggingface" / "hub")


class TestAHalfDownloadDoesNotMakeAMachineOffline(unittest.TestCase):
    """The trap this module set for itself, met on Linux 2026-08-09.

    One interrupted fetch leaves `models--…` in the cache. Counting that as "the machine
    has models" switched the loaders offline, and the next attempt met "cannot find the
    requested files in the local cache" — on a machine whose network was working. The
    person then spends the evening on the network, because that is what the message says.
    """

    def cache_with(self, tmp: Path, name: str, *, complete: bool) -> Path:
        entry = tmp / name / "snapshots" / "abc123"
        entry.mkdir(parents=True)
        (entry / ("model.safetensors" if complete
                  else "model.safetensors.incomplete")).write_bytes(b"x" * 16)
        return tmp

    def test_a_finished_model_still_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = self.cache_with(Path(tmp), "models--timm--vit", complete=True)
            self.assertTrue(offline.models_are_cached(cache))

    def test_an_interrupted_one_does_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = self.cache_with(Path(tmp), "models--timm--vit", complete=False)
            self.assertFalse(offline.models_are_cached(cache))

    def test_and_so_offline_mode_is_not_switched_on(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = self.cache_with(Path(tmp), "models--timm--vit", complete=False)
            with unittest.mock.patch.dict(os.environ, {}, clear=False):
                for var in (*offline._HF_VARS, offline.ENV_ALLOW_DOWNLOAD):
                    os.environ.pop(var, None)
                self.assertFalse(offline.configure_model_offline(cache))
                self.assertNotIn("HF_HUB_OFFLINE", os.environ)


if __name__ == "__main__":
    unittest.main()
