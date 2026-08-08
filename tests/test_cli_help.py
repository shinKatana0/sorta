"""F114: `--help` speaks the language from the config, like the rest of the CLI.

F112 moved the printed output into the catalog and left the help behind, for a real
reason: a `typer.Option(..., help=...)` runs when cli.py is imported, before anything
has read a config. F114 turns that around — the application is BUILT once the language
is known (`cli.build_app`), and the language is found by peeking at argv for
`--config`/`-c` beforehand (`cli._peek_config_path`).

Two properties matter more than the translation itself, and most of the cases below are
about them:

* `--help` has to work with no config and with a broken one. The person reading the
  help is the one who has not set anything up yet; a traceback there is worse than
  English help.
* the peek must not become a second parser. It looks, consumes nothing and validates
  nothing — typer still says what is wrong with a command line, in typer's words.

The `ru` expectations are GOLDEN in the same sense as in test_cli_i18n.py: they are the
strings cli.py used to carry in its decorators and docstrings, word for word.
"""
from __future__ import annotations

import contextlib
import io
import os
import re
import string
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sorta import __version__, cli, i18n, install

_LANGS = ("ru", "en", "ja")
_CYRILLIC = re.compile(r"[Ѐ-ӿ]")
# The two help texts that are an enumeration of literal option values ("city | person |
# event") and are therefore the same string in all three languages.
_UNTRANSLATABLE = ("cli.help.sort.by", "cli.help.album.kind")


def flat(text: str) -> str:
    """Help as one line: rich wraps it to the terminal width, and where exactly it
    wraps is not what any of these cases is about."""
    return " ".join(text.split())


@contextlib.contextmanager
def working_directory(path: Path):
    """`sorta --help` with no `--config` reads ./config.yaml — so the cases about the
    default path have to stand somewhere."""
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class _HelpCase(unittest.TestCase):
    def setUp(self):
        if cli.app is None:  # pragma: no cover — the argparse fallback
            self.skipTest("typer is not installed")
        from typer.testing import CliRunner
        self.runner = CliRunner()
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def config(self, lang: str, *, directory: Path | None = None) -> Path:
        path = (directory or self.root) / f"{lang}.yaml"
        path.write_text(f'database: "{(self.root / "t.db").as_posix()}"\n'
                        f'language: {lang}\n', encoding="utf-8")
        return path

    def help_for(self, argv: list[str]):
        """What `sorta <argv>` prints, decided the way `main()` decides it.

        A wide terminal on purpose: at 80 columns rich breaks a long help text across
        the border of its panel, and a case about wording would then fail over
        line breaks.
        """
        with patch.dict(os.environ, {"COLUMNS": "400"}):
            return self.runner.invoke(cli.build_app(cli._startup_lang(argv)), argv)


class TestHelpSpeaksTheConfigLanguage(_HelpCase):
    """The acceptance criterion: `sorta --help` and `sorta <command> --help` are in the
    language of the config, and the three languages really are three texts."""

    def test_the_root_help_follows_the_config_in_the_working_directory(self):
        printed = {}
        for lang in _LANGS:
            home = self.root / f"home-{lang}"
            home.mkdir()
            (home / "config.yaml").write_text(f"language: {lang}\n", encoding="utf-8")
            with working_directory(home):
                result = self.help_for(["--help"])
            self.assertEqual(result.exit_code, 0, result.output)
            printed[lang] = flat(result.output)
        # The version is taken from the package, not spelled out: this case is about the
        # three languages being three texts, and a literal here turns every release bump
        # into a red CI run (it did, on the v0.3.0 commit). The `v{version}` shape itself
        # is pinned by the catalog case below.
        self.assertIn(f"Sorta v{__version__} — сортировка фотоколлекции", printed["ru"])
        self.assertIn(f"Sorta v{__version__} — sorting a photo collection", printed["en"])
        self.assertIn(f"Sorta v{__version__} — 写真コレクションの整理", printed["ja"])
        self.assertEqual(len(set(printed.values())), 3)

    def test_a_command_help_follows_the_config_given_by_flag(self):
        for lang, phrase in (("ru", "Сканировать источники, извлечь метаданные"),
                             ("en", "Scan the sources, extract the metadata"),
                             ("ja", "ソースをスキャンし、メタデータを抽出し")):
            with self.subTest(lang=lang):
                result = self.help_for(
                    ["index", "--config", str(self.config(lang)), "--help"])
                self.assertEqual(result.exit_code, 0, result.output)
                self.assertIn(phrase, flat(result.output))

    def test_the_options_are_localized_and_not_only_the_description(self):
        result = self.help_for(["run", "--config", str(self.config("ja")), "--help"])
        self.assertEqual(result.exit_code, 0, result.output)
        printed = flat(result.output)
        self.assertIn("config.yaml へのパス", printed)
        self.assertIn("最後に dry-run のプランを作成します", printed)
        self.assertIsNone(_CYRILLIC.search(printed))

    def test_a_subcommand_group_is_localized_as_well(self):
        result = self.help_for(["faces", "--config", str(self.config("en")), "--help"])
        self.assertEqual(result.exit_code, 0, result.output)
        printed = flat(result.output)
        self.assertIn("Faces: detection, clusters, naming.", printed)
        self.assertIn("Export a contact sheet of the cluster into HTML.", printed)
        self.assertIsNone(_CYRILLIC.search(printed))

    def test_the_language_comes_from_the_file_the_flag_points_at(self):
        self.assertEqual(
            cli._startup_lang(["index", "--config", str(self.config("ja")), "--help"]),
            "ja")
        self.assertEqual(
            cli._startup_lang(["index", "-c", str(self.config("ru"))]), "ru")


class TestHelpWorksWithoutAUsableConfig(_HelpCase):
    """Requirement 1 of the brief: the main reader of the help is the person who has
    not configured anything yet, so nothing about the config may take it down."""

    def test_no_config_at_all_is_the_default_language(self):
        empty = self.root / "empty"
        empty.mkdir()
        with working_directory(empty):
            self.assertEqual(cli._startup_lang(["--help"]), "en")
            result = self.help_for(["--help"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("sorting a photo collection", flat(result.output))

    def _broken_paths(self) -> dict[str, Path]:
        invalid = self.root / "invalid.yaml"
        invalid.write_text("language: [oops\n", encoding="utf-8")
        empty_file = self.root / "empty.yaml"
        empty_file.write_text("", encoding="utf-8")
        unreadable = self.root / "a-directory.yaml"
        unreadable.mkdir()  # a directory where a file is expected: read_text raises
        return {"invalid": invalid, "empty": empty_file, "unreadable": unreadable,
                "missing": self.root / "nowhere.yaml"}

    def test_a_broken_config_gives_the_default_language_and_no_traceback(self):
        for name, path in self._broken_paths().items():
            with self.subTest(config=name):
                argv = ["index", "--config", str(path), "--help"]
                self.assertEqual(cli._startup_lang(argv), "en")
                result = self.help_for(argv)
                self.assertEqual(result.exit_code, 0, result.output)
                self.assertIsNone(result.exception)
                self.assertIn("Scan the sources", flat(result.output))


class TestThePeekIsNotASecondParser(_HelpCase):
    """Requirement 2: `_peek_config_path` only looks. Every spelling click accepts is
    recognised, everything else is left alone — including the errors."""

    def test_every_spelling_of_the_flag_is_found(self):
        path = "some/where/my.yaml"
        for argv in (["index", "--config", path],
                     [f"--config={path}", "index"],
                     ["index", "-c", path],
                     ["index", f"-c={path}"],
                     ["index", f"-c{path}"]):
            with self.subTest(argv=argv):
                self.assertEqual(cli._peek_config_path(argv), path)

    def test_what_is_not_a_config_flag_leaves_the_default(self):
        for argv in ([],
                     ["sort", "--copy", "--apply"],  # --copy also starts with -c
                     ["index", "--", "-c", "after-the-dashes.yaml"],
                     ["index", "--config"]):  # incomplete: typer's problem, not ours
            with self.subTest(argv=argv):
                self.assertEqual(cli._peek_config_path(argv), "config.yaml")

    def test_the_last_value_wins_like_in_click(self):
        self.assertEqual(
            cli._peek_config_path(["-c", "first.yaml", "--config", "second.yaml"]),
            "second.yaml")

    def test_a_normal_command_still_parses(self):
        app = cli.build_app("en")
        path = str(self.config("en"))
        with patch.object(cli, "_cmd_stats") as cmd:
            result = self.runner.invoke(app, ["stats", "--config", path])
        self.assertEqual(result.exit_code, 0, result.output)
        cmd.assert_called_once_with(path)

    def test_an_unknown_flag_is_still_typers_error(self):
        result = self.runner.invoke(cli.build_app("en"), ["stats", "--nope"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("--nope", flat(result.output))

    def test_the_peek_does_not_consume_the_flag(self):
        """The value has to reach the command as well — the peek is not a parser that
        eats what it read."""
        app = cli.build_app("en")
        path = str(self.config("ru"))
        with patch.object(cli, "_cmd_dupes") as cmd:
            result = self.runner.invoke(app, ["dupes", "--near", "-c", path])
        self.assertEqual(result.exit_code, 0, result.output)
        cmd.assert_called_once_with(path, near=True)


class TestTheArgparseFallbackIsLocalizedToo(_HelpCase):
    """Requirement 3: cli.py keeps a typer-free path for CI/sandboxes. Help whose
    language depends on which packages happen to be installed is the least predictable
    kind of help there is, so the fallback reads the same catalog."""

    def fallback_help(self, lang: str) -> str:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), self.assertRaises(SystemExit) as ctx:
            cli._argparse_main(lang, ["--help"])
        self.assertEqual(ctx.exception.code, 0)
        return flat(buf.getvalue())

    def test_the_fallback_help_speaks_each_language(self):
        russian = self.fallback_help("ru")
        self.assertIn("сортировка фотоколлекции", russian)
        self.assertIn("Путь к config.yaml", russian)
        english = self.fallback_help("en")
        self.assertIn("sorting a photo collection", english)
        self.assertIn("Path to config.yaml", english)
        self.assertIsNone(_CYRILLIC.search(english))
        self.assertIn("写真コレクションの整理", self.fallback_help("ja"))

    def test_the_fallback_still_runs_the_commands(self):
        with patch.object(cli, "_cmd_stats") as cmd:
            cli._argparse_main("en", ["stats", "-c", "x.yaml"])
        cmd.assert_called_once_with("x.yaml")
        with patch.object(cli, "_cmd_dupes") as cmd:
            cli._argparse_main("en", ["dupes", "--near"])
        cmd.assert_called_once_with("config.yaml", near=True)

    def test_main_takes_the_fallback_when_typer_is_missing(self):
        path = self.config("ja")
        with patch.object(cli, "_configure_runtime"), \
                patch.object(cli, "app", None), \
                patch.object(cli, "_argparse_main") as fallback, \
                patch.object(sys, "argv", ["sorta", "-c", str(path), "stats"]):
            cli.main()
        fallback.assert_called_once_with("ja")

    def test_main_runs_the_built_application_when_typer_is_there(self):
        with patch.object(cli, "_configure_runtime"), patch.object(cli, "app") as built:
            cli.main()
        built.assert_called_once_with()


class TestTheFlagsOfTheParityFeatureAreDocumented(_HelpCase):
    """F127: the new flags are help texts like any other, so the F114 watchdog widens
    to the commands that grew them — a key missing a language would otherwise reach the
    terminal as the key itself (`cli_text` falls back to it) and nothing would fail."""

    CHANGED = ("junk", "run", "cache", "album")

    def help_in(self, command: str, lang: str) -> str:
        result = self.help_for([command, "--config", str(self.config(lang)), "--help"])
        self.assertEqual(result.exit_code, 0, result.output)
        return flat(result.output)

    def test_every_changed_command_renders_in_three_languages(self):
        for command in self.CHANGED:
            printed = {lang: self.help_in(command, lang) for lang in _LANGS}
            for lang, text in printed.items():
                with self.subTest(command=command, lang=lang):
                    self.assertNotIn("cli.help.", text)  # no key came through raw
            with self.subTest(command=command):
                self.assertEqual(len(set(printed.values())), 3)

    def test_the_new_flags_are_listed_by_the_commands_that_take_them(self):
        # F186 retired `--quality`, `--no-quality` and `--quality-scope` with the
        # question they overrode; `--pets` is what is left of the F127 set on these two
        # commands, and it is still listed by both.
        expected = {
            "junk": ("--pets", "--no-pets"),
            "run": ("--pets", "--no-pets"),
            "cache": ("--preview-max-gb",),
        }
        for command, flags in expected.items():
            for lang in _LANGS:
                text = self.help_in(command, lang)
                for flag in flags:
                    with self.subTest(command=command, lang=lang, flag=flag):
                        self.assertIn(flag, text)

    def test_the_retired_flags_are_not_offered_in_any_language(self):
        """They used to be here with the price of each scope in the help text, so that
        nobody found out what `all` costs from a four-hour run. The question is retired
        (F186) and a help screen that still offered the flags would be describing a run
        the program cannot start."""
        for command in ("junk", "run"):
            for lang in _LANGS:
                text = self.help_in(command, lang)
                for flag in ("--quality", "--quality-scope"):
                    with self.subTest(command=command, lang=lang, flag=flag):
                        self.assertNotIn(flag, text)

    def test_the_album_help_mentions_the_animal_kind(self):
        for lang in _LANGS:
            with self.subTest(lang=lang):
                self.assertIn("animal", self.help_in("album", lang))


class TestRussianHelpIsUnchanged(unittest.TestCase):
    """Golden, as in test_cli_i18n.py: the `ru` variants ARE the texts cli.py carried
    before F114. A run with `language: ru` has to print the help it printed yesterday,
    so any rewording fails here instead of in a user's terminal.

    The line breaks inside the multi-paragraph texts are part of the expectation —
    Typer keeps them and only re-wraps what does not fit.
    """

    def assert_ru(self, key: str, text: str) -> None:
        self.assertEqual(i18n._CLI_STRINGS[key]["ru"], text, key)

    def test_the_short_texts(self):
        self.assert_ru("cli.help.app", "Sorta v{version} — сортировка фотоколлекции")
        self.assert_ru("cli.help.opt.config", "Путь к config.yaml")
        self.assert_ru("cli.help.index",
                       "Сканировать источники, извлечь метаданные, пометить дубликаты.")
        self.assert_ru("cli.help.index.src",
                       "Каталог с фото (рекурсивно); переопределяет config sources")
        self.assert_ru("cli.help.stats",
                       "Покрытие индекса: GPS, источники дат, дубликаты.")
        self.assert_ru("cli.help.dupes.near", "Показать почти-дубликаты (pHash)")
        self.assert_ru("cli.help.geo",
                       "Определить место каждого файла: GPS + наследование по сессиям.")
        self.assert_ru("cli.help.phash",
                       "Посчитать pHash для почти-дубликатов (для `dupes --near`).")
        self.assert_ru("cli.help.faces", "Лица: детекция, кластеры, именование.")
        self.assert_ru("cli.help.faces.label",
                       'Назвать кластер: sorta faces label 3 "Мама".')
        self.assert_ru("cli.help.events",
                       "События: автокластеризация, имена, ручные события.")
        self.assert_ru("cli.help.sort.by", "city | person | event")
        self.assert_ru("cli.help.sort.delete_worse_dupes",
                       "С --dedupe: БЕЗВОЗВРАТНО удалять худшие (не откатывается)")
        self.assert_ru("cli.help.album.where",
                       'Доп. фильтр среза: "city=Барселона", "year>=2020"')
        self.assert_ru("cli.help.reset.yes", "Без подтверждения")
        self.assert_ru("cli.help.undo.batch", "ID батча (по умолчанию последний)")

    def test_the_multi_paragraph_texts(self):
        self.assert_ru(
            "cli.help.reset",
            "Стереть индекс (БД) и начать с нуля. Фото и разложенные папки НЕ "
            "трогает.\n"
            "\n"
            "Внимание: пропадут имена людей/событий и решения по дублям. Кэш геоданных\n"
            "(F93) остаётся — названия точек на карте не зависят от того, какие файлы лежат\n"
            "у пользователя; стереть и его — `--clear-geo`.")
        # F165: the one deliberate edit to this golden — the list of steps is a FACT
        # about the pipeline, and the pipeline gained `classify` between `landmarks` and
        # `junk`. The rest of the text is pinned exactly as before.
        self.assert_ru(
            "cli.help.run",
            "Анализ одним прогоном: index -> geo -> landmarks -> classify -> junk "
            "(+faces/+events с флагами).\n"
            "\n"
            "Ничего не перемещает. С --by в конце строит dry-run план (в --dest либо\n"
            "in-place в корень источника, если --dest не задан).")
        self.assertTrue(i18n._CLI_STRINGS["cli.help.cache"]["ru"].startswith(
            "Кэши: показать путь и размер, при --clear/--clear-geo — удалить.\n\n"))


class TestTheHelpHalfOfTheCatalogIsComplete(unittest.TestCase):
    """Parity of the three languages over the help keys, the way the served UI has it
    — plus both directions of "the interface and the catalog agree"."""

    def test_every_help_key_carries_all_three_languages(self):
        keys = i18n.help_keys()
        self.assertGreater(len(keys), 40)  # the catalog is actually the whole surface
        for key in keys:
            with self.subTest(key=key):
                entry = i18n._CLI_STRINGS[key]
                self.assertEqual(set(entry), set(_LANGS))
                for lang in _LANGS:
                    self.assertTrue(entry[lang].strip(), lang)

    def test_the_interface_asks_for_exactly_the_keys_that_exist(self):
        source = Path(cli.__file__).read_text(encoding="utf-8")
        used = set(re.findall(r'"(cli\.help\.[a-z0-9_.]+)"', source))
        # F230: a key of `install.INSTALL_ADVICE` is a BASE — the literal in the source
        # stands for three keys, one per install kind, and `install.advice_key` turns it
        # into the one that gets printed (the help of `--deep` names a different command
        # in a checkout than on an installed copy).
        asked = set()
        for key in used:
            asked.update(install.advice_keys(key) if key in install.INSTALL_ADVICE
                         else {key})
        self.assertEqual(asked, set(i18n.help_keys()))

    def test_a_translation_is_a_translation_and_not_a_copy(self):
        for key in i18n.help_keys():
            if key in _UNTRANSLATABLE:
                continue
            entry = i18n._CLI_STRINGS[key]
            with self.subTest(key=key):
                self.assertNotEqual(entry["en"], entry["ru"])
                self.assertNotEqual(entry["en"], entry["ja"])

    def test_no_help_text_leaks_a_format_field_the_caller_does_not_pass(self):
        # Only the application title and the `--deep` help take a substitution; a stray
        # `{...}` anywhere else would raise KeyError inside `cli_text` at the worst
        # possible moment. F230: `cli.help.run.deep` carries `{how}`, filled with the way
        # THIS install adds the deep tier — `uv sync --extra vlm` in a checkout, the
        # wizard on an installed copy.
        takes = {"cli.help.app": {"version"}, "cli.help.run.deep": {"how"}}
        for key in i18n.help_keys():
            expected = takes.get(key, set())
            for lang in _LANGS:
                with self.subTest(key=key, lang=lang):
                    template = i18n._CLI_STRINGS[key][lang]
                    fields = {name for _, name, _, _
                              in string.Formatter().parse(template) if name}
                    self.assertEqual(fields, expected)


if __name__ == "__main__":
    unittest.main()
