"""F65: the "Folder language" selector — POST /api/config/language sets the OUTPUT
language (folders/names) separately from the interface `?lang`, persists into
config.yaml, and rebuilds the plan preview. Plus config.save_language (unit)."""
from __future__ import annotations

import json
import threading
import unittest
from pathlib import Path

from sorta import ui
from sorta.config import save_language

from tests.test_ui import UiServerTestBase


class TestSaveLanguage(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "config.yaml"

    def tearDown(self):
        self.tmp.cleanup()

    def test_replaces_existing_line_preserving_comments(self):
        self.path.write_text(
            "# a comment\nlanguage: en  # inline note\ndatabase: sorta.db\n",
            encoding="utf-8")
        save_language(self.path, "ru")
        text = self.path.read_text(encoding="utf-8")
        self.assertIn("language: ru", text)
        self.assertNotIn("language: en", text)
        # the surrounding lines survive
        self.assertIn("# a comment", text)
        self.assertIn("database: sorta.db", text)

    def test_appends_when_language_absent(self):
        self.path.write_text("database: sorta.db\n", encoding="utf-8")
        save_language(self.path, "ja")
        text = self.path.read_text(encoding="utf-8")
        self.assertIn("database: sorta.db", text)
        self.assertIn("language: ja", text)

    def test_creates_file_when_missing(self):
        self.assertFalse(self.path.exists())
        save_language(self.path, "ru")
        self.assertEqual(self.path.read_text(encoding="utf-8"), "language: ru\n")

    def test_invalid_value_normalized_to_default_en(self):
        save_language(self.path, "xx")
        self.assertIn("language: en", self.path.read_text(encoding="utf-8"))

    def test_only_top_level_language_line_is_replaced(self):
        # a nested/indented `language:` (a different key) must not be touched
        self.path.write_text(
            "language: en\nnested:\n  language: keep-me\n", encoding="utf-8")
        save_language(self.path, "ru")
        text = self.path.read_text(encoding="utf-8")
        self.assertIn("language: ru", text)
        self.assertIn("  language: keep-me", text)


class FolderLangServerBase(UiServerTestBase):
    """Starts the server with a config_path so persistence can be verified."""

    def start_with_config(self, config_path: str | Path | None) -> None:
        self.server = ui.build_server(self.cfg, self.conn, port=0,
                                      config_path=config_path)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def post(self, path: str, payload: dict) -> tuple[int, dict]:
        import urllib.error
        import urllib.request
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(f"{self.base_url}{path}", data=data,
                                     headers={"Content-Type": "application/json"},
                                     method="POST")
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())


class TestApiConfig(FolderLangServerBase):
    def test_get_config_returns_current_language(self):
        self.cfg.raw = {"language": "ja"}
        self.start_with_config(None)
        status, body, _ctype = self.get("/api/config")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), {"language": "ja"})

    def test_get_config_defaults_to_en_when_unset(self):
        self.start_with_config(None)
        _status, body, _ctype = self.get("/api/config")
        self.assertEqual(json.loads(body)["language"], "en")


class TestSetLanguage(FolderLangServerBase):
    def _config_file(self, lang: str = "en") -> Path:
        path = self.root / "config.yaml"
        path.write_text(f"# my config\nlanguage: {lang}\n", encoding="utf-8")
        return path

    def test_post_persists_and_relocalizes_plan(self):
        self.add_photo_file("a.jpg", country="ru", city="Moscow")
        self.cfg.raw = {"language": "en"}
        config_path = self._config_file("en")
        self.start_with_config(config_path)

        # before: English folders
        _s, body, _c = self.get("/api/plan?mode=city")
        self.assertTrue(json.loads(body)[0]["target_rel"].startswith("Russia/"))

        status, payload = self.post("/api/config/language", {"language": "ru"})
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"ok": True, "language": "ru"})

        # the running cfg is updated
        self.assertEqual(self.cfg.language, "ru")
        self.assertEqual(self.cfg.raw["language"], "ru")
        # persisted to config.yaml, comment preserved
        text = config_path.read_text(encoding="utf-8")
        self.assertIn("language: ru", text)
        self.assertIn("# my config", text)
        # the plan preview is rebuilt with Russian folder names
        _s2, body2, _c2 = self.get("/api/plan?mode=city")
        self.assertTrue(json.loads(body2)[0]["target_rel"].startswith("Россия/"))

    def test_invalid_language_is_400_and_no_change(self):
        self.cfg.raw = {"language": "en"}
        self.start_with_config(self._config_file("en"))
        status, payload = self.post("/api/config/language", {"language": "xx"})
        self.assertEqual(status, 400)
        self.assertIn("error", payload)
        self.assertEqual(self.cfg.language, "en")

    def test_no_config_path_updates_memory_only(self):
        self.cfg.raw = {"language": "en"}
        self.start_with_config(None)  # no file to write
        status, payload = self.post("/api/config/language", {"language": "ja"})
        self.assertEqual(status, 200)
        self.assertEqual(payload["language"], "ja")
        self.assertEqual(self.cfg.language, "ja")


class TestFolderLangMarkup(UiServerTestBase):
    def test_selector_and_label_present_in_index(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn('id="folder-lang-select"', html)
        self.assertIn("Folder language", html)  # default interface language is en
        # all three folder-language options are offered
        self.assertIn('<option value="ru">Русский</option>', html)
        self.assertIn('<option value="ja">日本語</option>', html)


if __name__ == "__main__":
    unittest.main()
