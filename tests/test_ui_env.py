"""F64: the CPU-profile banner on the "Process" tab — GET /api/env + markup.

gpu_profile = whether the GPU profile is installed (find_spec("nvidia")); CPU -> a
reduced-speed banner (faces/VLM/large collections)."""
from __future__ import annotations

import json
import unittest
from unittest import mock

from sorta import ui
from tests.test_ui import UiServerTestBase


class TestApiEnv(UiServerTestBase):
    def test_gpu_profile_true_when_nvidia_present(self):
        with mock.patch.object(ui.importlib.util, "find_spec", return_value=object()):
            self.start_server()
            status, body, ctype = self.get("/api/env")
        self.assertEqual(status, 200)
        self.assertIn("application/json", ctype)
        data = json.loads(body)
        self.assertEqual(set(data.keys()), {"gpu_profile"})
        self.assertTrue(data["gpu_profile"])

    def test_gpu_profile_false_when_nvidia_missing(self):
        with mock.patch.object(ui.importlib.util, "find_spec", return_value=None):
            self.start_server()
            _status, body, _ctype = self.get("/api/env")
        self.assertFalse(json.loads(body)["gpu_profile"])


class TestEnvBannerMarkup(UiServerTestBase):
    def test_banner_hidden_by_default_and_js_fetches_env(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn('id="env-cpu-warning"', html)
        self.assertIn('id="env-cpu-warning" class="env-warning" style="display:none"', html)
        self.assertIn('fetch("/api/env")', html)
        self.assertIn('!data.gpu_profile', html)

    def test_warning_text_ru_en_ja(self):
        for lang, text in (
            ("ru", "Установлен CPU-профиль"),
            ("en", "CPU profile installed"),
            ("ja", "CPU プロファイル"),
        ):
            self.cfg.raw = {"language": lang}
            self.start_server()
            _status, body, _ctype = self.get("/")
            self.assertIn(text, body.decode("utf-8"), msg=f"lang={lang}")
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
