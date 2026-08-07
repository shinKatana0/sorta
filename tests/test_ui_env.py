"""F64/F217: the environment banner of the "Process" tab — GET /api/env + markup.

`gpu_profile` is whether the GPU profile is INSTALLED (find_spec("nvidia")). F217 added
the two things that made the banner wrong for the person it was written for: whether the
machine has a card at all (`gpu_present`, the nvidia-smi probe that was already written
and unused), and the state of every install tier, taken from the probe `sorta doctor`
uses. Without the first, a machine with no NVIDIA card was advised to download 2.5 GB of
CUDA wheels; without the second, the way out named a command an installed copy cannot
run.
"""
from __future__ import annotations

import json
import unittest
from unittest import mock

from sorta import i18n, tiers, ui, wizard
from tests.test_ui import UiServerTestBase


def _no_card():
    """The nvidia-smi probe answered for the test rather than for this machine.

    The answer is cached for the life of the process (see `_gpu_present`), so every case
    that cares about it clears the cache first.
    """
    ui.process._gpu_present_cache_clear()
    return mock.patch.object(ui.process, "nvidia_gpu_present", return_value=False)


def _with_card():
    ui.process._gpu_present_cache_clear()
    return mock.patch.object(ui.process, "nvidia_gpu_present", return_value=True)


class TestTheCardIsAskedAboutOnce(unittest.TestCase):
    """`/api/env` is also how a second launch asks whether the program on this port is
    ours (`tray.PROBE_TIMEOUT`, 2 s) — and `nvidia-smi` on a half-installed driver may
    take the 3 s its own probe allows. A card does not arrive while the server is up, so
    the question is asked once."""

    def setUp(self):
        ui.process._gpu_present_cache_clear()

    def tearDown(self):
        ui.process._gpu_present_cache_clear()

    def test_the_probe_runs_once_however_often_the_route_is_asked(self):
        with mock.patch.object(ui.process, "nvidia_gpu_present",
                               return_value=True) as probe:
            first = ui.process._gpu_present()
            second = ui.process._gpu_present()
        self.assertTrue(first)
        self.assertTrue(second)
        self.assertEqual(probe.call_count, 1)


class TestApiEnv(UiServerTestBase):
    def test_gpu_profile_true_when_nvidia_present(self):
        with mock.patch.object(ui.importlib.util, "find_spec", return_value=object()), \
                _with_card():
            self.start_server()
            status, body, ctype = self.get("/api/env")
        self.assertEqual(status, 200)
        self.assertIn("application/json", ctype)
        data = json.loads(body)
        self.assertEqual(set(data.keys()), {"gpu_profile", "gpu_present", "tiers"})
        self.assertTrue(data["gpu_profile"])
        self.assertTrue(data["gpu_present"])

    def test_gpu_profile_false_when_nvidia_missing(self):
        with mock.patch.object(ui.importlib.util, "find_spec", return_value=None), \
                _no_card():
            self.start_server()
            _status, body, _ctype = self.get("/api/env")
        self.assertFalse(json.loads(body)["gpu_profile"])

    def test_the_card_is_answered_for_separately_from_the_profile(self):
        """The two are different questions, and conflating them is the defect: packages
        named `nvidia-*` say which profile was chosen, not whether there is a card."""
        with mock.patch.object(ui.importlib.util, "find_spec", return_value=None), \
                _with_card():
            self.start_server()
            _status, body, _ctype = self.get("/api/env")
        data = json.loads(body)
        self.assertFalse(data["gpu_profile"])
        self.assertTrue(data["gpu_present"])

    def test_every_tier_of_the_catalog_reaches_the_browser(self):
        with _no_card():
            self.start_server()
            _status, body, _ctype = self.get("/api/env")
        tiers = json.loads(body)["tiers"]
        self.assertEqual(set(tiers), {tier.key for tier in wizard.TIERS})
        for key, info in tiers.items():
            with self.subTest(tier=key):
                self.assertIn(info["state"], ("ready", "weights", "absent"))
                self.assertIsInstance(info["missing"], list)


class TestEnvBannerMarkup(UiServerTestBase):
    def test_banner_hidden_by_default_and_js_fetches_env(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn('id="env-cpu-warning"', html)
        self.assertIn('id="env-cpu-warning" class="env-warning" style="display:none"', html)
        self.assertIn('fetch("/api/env")', html)
        self.assertIn('!data.gpu_profile', html)

    def test_the_banner_needs_a_card_as_well_as_a_missing_profile(self):
        """The three conversations of the brief: a card and no profile — say how to turn
        the acceleration on; no card — say nothing, the CPU profile is the right install
        and there is nothing to advise."""
        self.start_server()
        _status, body, _ctype = self.get("/")
        self.assertIn("data.gpu_present && !data.gpu_profile", body.decode("utf-8"))

    def test_warning_text_ru_en_ja(self):
        for lang, text in (
            ("ru", "установлен CPU-профиль"),
            ("en", "the CPU profile is installed"),
            ("ja", "CPU プロファイル"),
        ):
            self.cfg.raw = {"language": lang}
            self.start_server()
            _status, body, _ctype = self.get("/")
            self.assertIn(text, body.decode("utf-8"), msg=f"lang={lang}")
            self.tearDown()
            self.setUp()

    def test_the_banner_names_the_tier_and_the_wizard_rather_than_a_command(self):
        """`uv tool install --force ".[gpu]"` worked for nobody who used the installer:
        no `uv` on PATH, no sources for `.`, and the install was made with
        `uv pip install --target`. The way out is the wizard's own line, and the tier is
        named the way `sorta-setup` names it — so the line can be found."""
        for lang in ("ru", "en", "ja"):
            with self.subTest(lang=lang):
                self.cfg.raw = {"language": lang}
                self.start_server()
                _status, body, _ctype = self.get("/")
                html = body.decode("utf-8")
                self.assertNotIn("uv tool install", html)
                self.assertIn(wizard.TIERS_BY_KEY["gpu"].name(lang), html)
                self.assertIn(i18n.cli_text(tiers._tier_hint_key(), lang).strip(), html)
                self.tearDown()
                self.setUp()

    def test_no_external_resources(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertNotIn("http://", html)
        self.assertNotIn("https://", html)


if __name__ == "__main__":
    unittest.main()
