"""F111: comments and docstrings in the Python sources are English.

The repository is public, so the prose that explains WHY the code is the way it is has
to be readable by someone who does not read Russian. Russian that is the SUBJECT of a
sentence stays — folder names the product really creates («_удалить»), the spellings
people really type for their trip folders («Тайланд 2023»), country names that double
as ordinary words («Чад») — because removing them would leave the paragraph proving
nothing. The mechanical form of that rule, and what this module checks, is:

* every Cyrillic run inside a comment or a docstring sits inside quotes — «», "", ''
  or backticks — i.e. it is cited as data, not written as language;
* there is no exemption. F111 had to make one for the command docstrings of
  `sorta/cli.py`, which Typer printed as the `--help` text of each command: program
  output in the interface language, not documentation about the code. F114 moved that
  output where the rest of the interface language already lived — the string catalog
  in `sorta/i18n.py` — so cli.py is now read like every other file. The second class
  below states what replaced the exemption.

Functional strings are out of scope by construction: this module only ever looks at
COMMENT tokens and at docstrings, never at the string literals the program prints.
"""
from __future__ import annotations

import ast
import io
import re
import tokenize
import unittest
from pathlib import Path

from sorta import cli, i18n

_ROOT = Path(__file__).resolve().parent.parent
_SOURCE_DIRS = ("sorta", "tests", "scripts")

# The whole Cyrillic block plus the numero sign — a Russian typographic mark that
# would otherwise slip past as punctuation. The class is matched by Python's `re` over
# decoded text, not by a shell tool over bytes: a byte-wise character class is how the
# audit for this feature first mistook an em dash for Russian.
_CYRILLIC = re.compile(r"[Ѐ-ӿ№]+")
# A quoted span, across line breaks: the guillemets the project uses for names people
# type, plain quotes, and backticks for identifiers and command lines. An apostrophe
# after a letter is a possessive ("a person's own labelling"), not an opening quote —
# treating it as one would let a whole sentence count as quoted and hide prose in it.
_QUOTED = re.compile(r"«[^»]*»|\"[^\"]*\"|(?<!\w)'[^']*'|`[^`]*`", re.DOTALL)


def source_files() -> list[Path]:
    return sorted(
        path
        for directory in _SOURCE_DIRS
        for path in (_ROOT / directory).rglob("*.py")
    )


def comment_blocks(source: str) -> list[tuple[int, str]]:
    """(first line, text) per run of consecutive comment lines.

    Consecutive lines are joined because a quotation may be wrapped across two of them
    — a per-line check would read the halves as unquoted Russian.
    """
    blocks: list[tuple[int, str]] = []
    previous_line = -2
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type != tokenize.COMMENT:
            continue
        text = token.string.lstrip("#").strip()
        if token.start[0] == previous_line + 1 and blocks:
            first, existing = blocks[-1]
            blocks[-1] = (first, f"{existing} {text}")
        else:
            blocks.append((token.start[0], text))
        previous_line = token.start[0]
    return blocks


def docstrings(source: str) -> list[tuple[int, str]]:
    """Every bare string-literal statement: docstrings and attribute docstrings."""
    found: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        for field in ("body", "orelse", "finalbody"):
            body = getattr(node, field, None)
            if not isinstance(body, list):
                continue
            for statement in body:
                if (
                    isinstance(statement, ast.Expr)
                    and isinstance(statement.value, ast.Constant)
                    and isinstance(statement.value.value, str)
                ):
                    found.append((statement.lineno, statement.value.value))
    return found


def unquoted_cyrillic(text: str) -> list[str]:
    """The Cyrillic runs of `text` that are NOT inside quotes."""
    return _CYRILLIC.findall(_QUOTED.sub(" ", text))


class TestCommentsAndDocstringsAreEnglish(unittest.TestCase):
    """Prose in English; Russian only where it is the thing being talked about."""

    def test_no_comment_holds_russian_prose(self):
        for path in source_files():
            source = path.read_text(encoding="utf-8")
            for line, text in comment_blocks(source):
                with self.subTest(file=path.name, line=line):
                    self.assertEqual(unquoted_cyrillic(text), [], text)

    def test_no_docstring_holds_russian_prose(self):
        for path in source_files():
            source = path.read_text(encoding="utf-8")
            for line, text in docstrings(source):
                with self.subTest(file=path.name, line=line):
                    self.assertEqual(unquoted_cyrillic(text), [], text)

    def test_the_rule_would_notice_a_russian_comment(self):
        """The check has to fail on prose and pass on a citation of the same word."""
        self.assertEqual(unquoted_cyrillic("# папка для мусора"), ["папка", "для", "мусора"])
        self.assertEqual(unquoted_cyrillic("the folder is named «_удалить»"), [])
        self.assertEqual(unquoted_cyrillic('a name like "Тайланд 2023"'), [])
        self.assertEqual(unquoted_cyrillic("`--where country=Россия`"), [])
        # A possessive apostrophe must not open a span that swallows the prose after it.
        self.assertEqual(
            unquoted_cyrillic("a person's own labelling: папка для мусора'"),
            ["папка", "для", "мусора"],
        )

    def test_a_quotation_wrapped_over_two_comment_lines_stays_one_block(self):
        source = '# a trip folder is named "Тайланд\n# 04.2025" by hand\nx = 1\n'
        self.assertEqual(
            comment_blocks(source),
            [(1, 'a trip folder is named "Тайланд 04.2025" by hand')],
        )

    def test_every_source_directory_is_actually_scanned(self):
        """A typo in the directory list would make the whole guard silently vacuous."""
        scanned = source_files()
        self.assertGreater(len(scanned), 100)
        for directory in _SOURCE_DIRS:
            with self.subTest(directory=directory):
                self.assertTrue(any(p.is_relative_to(_ROOT / directory) for p in scanned))


class TestHelpTextComesFromTheCatalog(unittest.TestCase):
    """What replaced the cli.py exemption, stated as a fact about the program rather
    than as a rule: the `--help` text of a command is a `cli.help.*` entry of the
    string catalog now, so the Russian that used to justify the exemption is a
    translation like any other and no longer lives in a docstring at all (F114)."""

    def setUp(self):
        if cli.app is None:  # pragma: no cover — the argparse fallback
            self.skipTest("typer is not installed")
        from typer.testing import CliRunner
        self.runner = CliRunner()

    def test_the_help_output_carries_no_terminal_styling(self):
        """The guard behind the assertion below, checked on output rather than on faith.

        conftest.py sets `_TYPER_FORCE_DISABLE_TERMINAL` because typer styles `--help`
        whenever GITHUB_ACTIONS is set, and the escapes cut a docstring into pieces:
        "--near" comes back as a bold "-" followed by a bold "-near". A switch that
        stops working — renamed upstream, or read after typer was already imported —
        would otherwise surface as a mystery failure on CI only; here it fails by name.
        """
        result = self.runner.invoke(cli.app, ["phash", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertNotIn("\x1b", result.output)

    def test_a_catalog_entry_is_what_help_prints(self):
        russian = cli.build_app("ru")
        for command in ("stats", "geo", "phash"):
            with self.subTest(command=command):
                text = i18n.cli_text(f"cli.help.{command}", "ru")
                self.assertTrue(unquoted_cyrillic(text), text)  # it IS Russian prose
                result = self.runner.invoke(russian, [command, "--help"])
                self.assertEqual(result.exit_code, 0)
                printed = " ".join(result.output.split())
                self.assertIn(" ".join(text.split()), printed)


if __name__ == "__main__":
    unittest.main()
