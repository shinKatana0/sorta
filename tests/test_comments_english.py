"""F111: comments and docstrings in the Python sources are English.

The repository is public, so the prose that explains WHY the code is the way it is has
to be readable by someone who does not read Russian. Russian that is the SUBJECT of a
sentence stays — folder names the product really creates («_удалить»), the spellings
people really type for their trip folders («Тайланд 2023»), country names that double
as ordinary words («Чад») — because removing them would leave the paragraph proving
nothing. The mechanical form of that rule, and what this module checks, is:

* every Cyrillic run inside a comment or a docstring sits inside quotes — «», "", ''
  or backticks — i.e. it is cited as data, not written as language;
* the one exemption is the command docstrings of `sorta/cli.py`: Typer prints them as
  the `--help` text of each command, so they are program output in the interface
  language, not documentation about the code. The second test below pins that claim
  down by reading a docstring back out of `--help`.

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

from sorta import cli

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

# Typer builds `sorta <command> --help` out of these docstrings — see the comment above
# the Typer block in cli.py. Everything else in that file follows the ordinary rule.
_HELP_TEXT_FILE = Path("sorta") / "cli.py"


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


def _is_command(node: ast.FunctionDef) -> bool:
    """Decorated with @app.command()/@app.callback() — i.e. Typer prints its docstring."""
    for decorator in node.decorator_list:
        call = decorator.func if isinstance(decorator, ast.Call) else decorator
        if isinstance(call, ast.Attribute) and call.attr in ("command", "callback"):
            return True
    return False


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
            if path.relative_to(_ROOT) == _HELP_TEXT_FILE:
                continue
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


class TestCliDocstringsAreHelpText(unittest.TestCase):
    """The cli.py exemption, stated as a fact about the program rather than a rule."""

    def setUp(self):
        if not hasattr(cli, "app"):  # pragma: no cover — the argparse fallback
            self.skipTest("typer is not installed")
        from typer.testing import CliRunner
        self.runner = CliRunner()

    def test_a_command_docstring_is_what_help_prints(self):
        for command, function in (("stats", "stats"), ("geo", "geo"), ("phash", "phash")):
            with self.subTest(command=command):
                doc = getattr(cli, function).__doc__
                self.assertIsNotNone(doc)
                result = self.runner.invoke(cli.app, [command, "--help"])
                self.assertEqual(result.exit_code, 0)
                printed = " ".join(result.output.split())
                self.assertIn(" ".join(doc.split()), printed)

    def test_the_exempted_docstrings_are_the_only_russian_ones_left(self):
        """Russian outside the command docstrings of cli.py would still be a bug."""
        source = (_ROOT / _HELP_TEXT_FILE).read_text(encoding="utf-8")
        tree = ast.parse(source)
        help_texts = {
            node.body[0].lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and _is_command(node)
            and ast.get_docstring(node) is not None
        }
        self.assertGreater(len(help_texts), 15)
        for line, text in docstrings(source):
            if not unquoted_cyrillic(text):
                continue
            with self.subTest(line=line):
                self.assertIn(
                    line, help_texts,
                    f"cli.py:{line} is Russian but is not a command's --help text",
                )


if __name__ == "__main__":
    unittest.main()
