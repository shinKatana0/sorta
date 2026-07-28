"""F104: `config.save_setting` — one key of config.yaml rewritten in place.

The web app writes settings while the server runs, into a file that belongs to the
user: their comments, their ordering, their blank lines, and quite possibly keys this
version of the program knows nothing about. A YAML round-trip would silently throw all
of that away on the first click of a checkbox, so the saver is a LINE-level edit and
what is pinned here is exactly that — everything the call did not aim at survives it.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from sorta.config import load_config, save_setting


class SaveSettingTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "config.yaml"

    def write(self, text: str) -> None:
        self.path.write_text(text, encoding="utf-8")

    def text(self) -> str:
        return self.path.read_text(encoding="utf-8")

    def loaded(self) -> dict:
        return yaml.safe_load(self.text()) or {}


class TestNestedKey(SaveSettingTestBase):
    def test_replaces_the_value_inside_its_section(self):
        self.write("vlm:\n  enabled: false\n  max_edge: 896\n")
        save_setting(self.path, "vlm.enabled", True)
        self.assertEqual(self.loaded(), {"vlm": {"enabled": True, "max_edge": 896}})

    def test_comments_and_neighbours_survive(self):
        self.write(
            "# my own notes\n"
            "database: sorta.db\n"
            "vlm:\n"
            "  # why this is off\n"
            "  enabled: false\n"
            "  model: Qwen/Qwen2.5-VL-3B-Instruct\n"
            "naming:\n"
            "  provider: template\n")
        save_setting(self.path, "vlm.enabled", True)
        text = self.text()
        self.assertIn("# my own notes", text)
        self.assertIn("  # why this is off", text)
        self.assertIn("database: sorta.db", text)
        self.assertIn("  provider: template", text)
        self.assertEqual(self.loaded()["vlm"]["enabled"], True)
        self.assertEqual(self.loaded()["naming"], {"provider": "template"})

    def test_the_indentation_of_the_section_is_kept(self):
        self.write("vlm:\n    enabled: false\n")
        save_setting(self.path, "vlm.enabled", True)
        self.assertIn("    enabled: true", self.text())

    def test_a_missing_key_is_added_to_its_section(self):
        self.write("vlm:\n  enabled: true\nlanguage: ru\n")
        save_setting(self.path, "vlm.max_edge", 640)
        self.assertEqual(self.loaded(),
                         {"vlm": {"enabled": True, "max_edge": 640}, "language": "ru"})

    def test_a_missing_section_is_appended(self):
        self.write("# just a language\nlanguage: ru\n")
        save_setting(self.path, "vlm.workers", 3)
        self.assertEqual(self.loaded(), {"language": "ru", "vlm": {"workers": 3}})
        self.assertIn("# just a language", self.text())

    def test_an_empty_section_header_gets_its_first_key(self):
        self.write("vlm:\nlanguage: ru\n")
        save_setting(self.path, "vlm.workers", 2)
        self.assertEqual(self.loaded(), {"vlm": {"workers": 2}, "language": "ru"})

    def test_a_same_named_key_in_another_section_is_not_touched(self):
        """`enabled:` is not a rare name — only the one under `vlm:` may move."""
        self.write("other:\n  enabled: false\nvlm:\n  enabled: false\n")
        save_setting(self.path, "vlm.enabled", True)
        self.assertEqual(self.loaded(),
                         {"other": {"enabled": False}, "vlm": {"enabled": True}})

    def test_a_missing_file_is_created(self):
        save_setting(self.path, "vlm.enabled", True)
        self.assertEqual(self.loaded(), {"vlm": {"enabled": True}})


class TestTopLevelKey(SaveSettingTestBase):
    def test_replaces_an_existing_top_level_line(self):
        self.write("# note\nlanguage: en\ndatabase: sorta.db\n")
        save_setting(self.path, "language", "ru")
        self.assertEqual(self.loaded(), {"language": "ru", "database": "sorta.db"})
        self.assertIn("# note", self.text())

    def test_appends_when_absent(self):
        self.write("database: sorta.db\n")
        save_setting(self.path, "language", "ja")
        self.assertEqual(self.loaded(), {"database": "sorta.db", "language": "ja"})

    def test_an_indented_key_of_the_same_name_is_left_alone(self):
        self.write("language: en\nnested:\n  language: keep-me\n")
        save_setting(self.path, "language", "ru")
        self.assertEqual(self.loaded(),
                         {"language": "ru", "nested": {"language": "keep-me"}})


class TestScalarFormatting(SaveSettingTestBase):
    def test_booleans_are_yaml_booleans_not_python_ones(self):
        """`True` with a capital T reads back as the STRING "True" — and a string is
        truthy, so `enabled: False` would switch a 20 GB tier on."""
        self.write("vlm:\n  enabled: true\n")
        save_setting(self.path, "vlm.enabled", False)
        self.assertIn("enabled: false", self.text())
        self.assertIs(self.loaded()["vlm"]["enabled"], False)

    def test_a_model_id_survives_unquoted(self):
        save_setting(self.path, "vlm.model", "Qwen/Qwen2.5-VL-3B-Instruct")
        self.assertEqual(self.loaded()["vlm"]["model"], "Qwen/Qwen2.5-VL-3B-Instruct")

    def test_a_string_yaml_would_read_as_something_else_is_quoted(self):
        for value in ("no", "true", "null", "on"):
            with self.subTest(value=value):
                save_setting(self.path, "vlm.model", value)
                self.assertEqual(self.loaded()["vlm"]["model"], value)

    def test_a_string_with_yaml_punctuation_is_quoted(self):
        for value in ("a: b", "# not a comment", "- dash", 'quote"inside'):
            with self.subTest(value=value):
                save_setting(self.path, "vlm.model", value)
                self.assertEqual(self.loaded()["vlm"]["model"], value)

    def test_integers_stay_integers(self):
        save_setting(self.path, "vlm.max_edge", 640)
        self.assertEqual(self.loaded()["vlm"]["max_edge"], 640)


class TestTheSavedFileIsStillOurConfig(SaveSettingTestBase):
    def test_load_config_reads_back_what_was_written(self):
        """The point of the whole exercise: the value has to be there on the NEXT
        start, through the ordinary loader, not only inside the running process."""
        self.write("language: ru\nvlm:\n  enabled: false\n  workers: 1\n")
        save_setting(self.path, "vlm.enabled", True)
        save_setting(self.path, "vlm.max_edge", 512)
        cfg = load_config(self.path)
        self.assertTrue(cfg.vlm.enabled)
        self.assertEqual(cfg.vlm.max_edge, 512)
        self.assertEqual(cfg.vlm.workers, 1)
        self.assertEqual(cfg.language, "ru")
        # F102: the legacy mirror stays in step with the section
        self.assertTrue(cfg.naming.vlm_enabled)


if __name__ == "__main__":
    unittest.main()
