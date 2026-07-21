"""F33: interface i18n (ru/en/ja) + dark theme + a collapsed plan tree."""
from __future__ import annotations

import json
import unittest

from sorta import ui

from tests.test_ui import UiServerTestBase


class TestIndexHtmlLanguage(UiServerTestBase):
    def test_default_language_is_english(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn('<html lang="en">', html)
        self.assertIn(">Cities<", html)
        self.assertIn(">Duplicates<", html)
        self.assertIn("window.I18N", html)

    def test_language_en_renders_english_chrome(self):
        self.cfg.raw = {"language": "en"}
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn('<html lang="en">', html)
        self.assertIn(">Cities<", html)
        self.assertIn(">Duplicates<", html)
        self.assertIn(">People<", html)
        self.assertIn(">Events<", html)
        self.assertIn(">Moves<", html)
        self.assertIn("Expand all", html)
        self.assertIn("window.I18N", html)
        # Russian chrome must not leak through with language=en.
        self.assertNotIn(">Города<", html)

    def test_language_ja_renders_japanese_chrome(self):
        self.cfg.raw = {"language": "ja"}
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn('<html lang="ja">', html)
        self.assertIn(">都市<", html)
        self.assertIn(">重複<", html)
        self.assertIn("window.I18N", html)

    def test_unknown_language_falls_back_to_en(self):
        self.cfg.raw = {"language": "xx"}
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn('<html lang="en">', html)
        self.assertIn(">Cities<", html)

    def test_data_strings_from_server_not_translated(self):
        # Data (folders/names) does not go through _UI_STRINGS/_t — the server does
        # not touch it, regardless of the interface language.
        self.add_photo_file("a.jpg", country="ru", city="Moscow")
        self.cfg.raw = {"language": "en"}
        self.start_server()
        status, body, ctype = self.get("/api/plan?mode=city")
        self.assertEqual(status, 200)
        self.assertIn("application/json", ctype)
        items = json.loads(body)
        self.assertEqual(items[0]["geo"], "ru/Moscow")


class TestUiStringsFallback(unittest.TestCase):
    def test_t_falls_back_to_en_then_key(self):
        ui._UI_STRINGS["_f33_test_only_en"] = {"en": "hello"}
        try:
            self.assertEqual(ui._t("_f33_test_only_en", "ru"), "hello")
            self.assertEqual(ui._t("_f33_test_only_en", "ja"), "hello")
            self.assertEqual(ui._t("_f33_totally_unknown_key", "ru"),
                             "_f33_totally_unknown_key")
        finally:
            del ui._UI_STRINGS["_f33_test_only_en"]

    def test_t_known_key_exact_language(self):
        self.assertEqual(ui._t("tab_city", "en"), "Cities")
        self.assertEqual(ui._t("tab_city", "ja"), "都市")
        self.assertEqual(ui._t("tab_city", "ru"), "Города")


class TestDarkTheme(UiServerTestBase):
    def test_theme_toggle_present_in_html(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn('id="theme-toggle-btn"', html)
        self.assertIn("localStorage", html)
        self.assertIn("prefers-color-scheme", html)

    def test_css_variables_for_light_and_dark(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn("--bg", html)
        self.assertIn("--ink", html)
        self.assertIn("--line", html)
        self.assertIn("--card", html)
        self.assertIn("--accent", html)
        self.assertIn('data-theme="dark"', html)
        self.assertIn('data-theme="light"', html)


class TestLangSwitcher(UiServerTestBase):
    """F39: `?lang=` switches the chrome language without restarting the server."""

    def test_lang_query_en_overrides_default(self):
        self.start_server()
        _status, body, _ctype = self.get("/?lang=en")
        html = body.decode("utf-8")
        self.assertIn('<html lang="en">', html)
        self.assertIn(">Cities<", html)

    def test_lang_query_ja_overrides_default(self):
        self.start_server()
        _status, body, _ctype = self.get("/?lang=ja")
        html = body.decode("utf-8")
        self.assertIn('<html lang="ja">', html)
        self.assertIn(">都市<", html)

    def test_no_lang_query_uses_config_default(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn('<html lang="en">', html)
        self.assertIn(">Cities<", html)

    def test_invalid_lang_query_falls_back_to_config_default_not_ru(self):
        # An invalid ?lang must not hardcode ru — the default is cfg.language.
        self.cfg.raw = {"language": "en"}
        self.start_server()
        status, body, _ctype = self.get("/?lang=xx")
        html = body.decode("utf-8")
        self.assertEqual(status, 200)
        self.assertIn('<html lang="en">', html)
        self.assertIn(">Cities<", html)

    def test_invalid_lang_query_is_200_and_default_en(self):
        self.start_server()
        status, body, _ctype = self.get("/?lang=xx")
        html = body.decode("utf-8")
        self.assertEqual(status, 200)
        self.assertIn('<html lang="en">', html)

    def test_selector_present_with_all_three_options(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn('id="lang-select"', html)
        self.assertIn('<option value="ru"', html)
        self.assertIn('<option value="en"', html)
        self.assertIn('<option value="ja"', html)
        self.assertIn("Русский", html)
        self.assertIn("English", html)
        self.assertIn("日本語", html)

    def test_selector_marks_current_lang_selected(self):
        self.start_server()
        _status, body, _ctype = self.get("/?lang=en")
        html = body.decode("utf-8")
        self.assertIn('<option value="en" selected>', html)
        self.assertIn('<option value="ru">', html)

    def test_localstorage_switch_logic_in_script(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn("sorta_lang", html)
        self.assertIn("lang-select", html)
        self.assertIn("localStorage", html)

    def test_other_ui_invariants_still_hold(self):
        # U1: no external resources; the theme (F33) is not affected by the language switcher.
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertNotIn("http://", html)
        self.assertNotIn("https://", html)
        self.assertIn('id="theme-toggle-btn"', html)


class TestTreeCollapsedByDefault(UiServerTestBase):
    def test_plan_tree_nodes_render_without_forced_open(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        # renderNode used to do `if (depth === 0) details.open = true;` —
        # the top tree level opened automatically. That logic was removed, so the
        # pattern must no longer appear in the script.
        self.assertNotIn("depth === 0", html)
        self.assertIn("expand-all-btn", html)
        self.assertIn("collapse-all-btn", html)
        # The expand/collapse-all buttons still set .open explicitly.
        self.assertIn("d.open = true", html)
        self.assertIn("d.open = false", html)


if __name__ == "__main__":
    unittest.main()
