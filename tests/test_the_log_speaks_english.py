"""F238: the log line and the text of a crash are English, everywhere in the package.

The default interface language is `en` (F112/F114), so a Russian log line is a message
the reader of that log cannot read. The log is asked for in three of the guides — it
arrives attached to a report written by someone whose product speaks English — and the
text of an exception is the same thing on a screenshot. So both are English, and neither
goes into the `i18n` catalog: a catalog entry costs three translations and a completeness
guard, and diagnostics are a tool for taking a run apart, not text of the product.

What is read here, because the definition IS the rule: every string literal handed to a
logging call (`.debug/.info/.warning/.error/.exception/.critical`) and every string
literal under a `raise`. An f-string is walked into, so the Russian around a `{value}`
counts on its own. Nothing else the program emits is read: the static HTML report, the
prompt sent to the model, a regex over Russian file names are all product or data, and
F238 does not reach them.

The package is walked, never listed. A guard with a hand-written list of files is true
and useless — it passes the moment someone adds the file it does not name, which is how
the English-documents guard was first got past.
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

from tests.test_comments_english import unquoted_cyrillic

_ROOT = Path(__file__).resolve().parent.parent
_PACKAGE = _ROOT / "sorta"

_LOG_METHODS = frozenset({"debug", "info", "warning", "error", "exception", "critical"})

# The two string catalogs: Russian in them is the ru translation of the interface, which
# is the whole point of the file. `tests/test_i18n.py` is what holds `i18n.py` to account
# for having all three languages; the web chrome of `ui/strings.py` answers to `test_ui_i18n`.
_CATALOGS = frozenset({"sorta/i18n.py", "sorta/ui/strings.py"})


def modules() -> list[Path]:
    return sorted(_PACKAGE.rglob("*.py"))


def module_name(path: Path) -> str:
    return path.relative_to(_ROOT).as_posix()


def _is_log_call(node: ast.Call) -> bool:
    return isinstance(node.func, ast.Attribute) and node.func.attr in _LOG_METHODS


def message_strings(source: str) -> list[tuple[int, str]]:
    """(line, text) for every string the program says to a human about a run.

    Only says WHICH strings are messages, not whether they are English — an f-string is
    reported one fragment at a time, so the answer is per literal and not per statement.
    """
    found: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and _is_log_call(node):
            carriers: list[ast.expr] = list(node.args) + [kw.value for kw in node.keywords]
        elif isinstance(node, ast.Raise):
            carriers = [part for part in (node.exc, node.cause) if part is not None]
        else:
            continue
        for carrier in carriers:
            for part in ast.walk(carrier):
                if isinstance(part, ast.Constant) and isinstance(part.value, str):
                    found.append((part.lineno, part.value))
    # Stable by line only: within one line the walk order is left to right, which is how
    # the fragments of an f-string read. Sorting by the text as well would reverse them.
    found.sort(key=lambda item: item[0])
    return found


def russian_messages(source: str) -> list[tuple[int, str]]:
    """The messages of `source` that are written in Russian rather than citing it.

    Quoted Russian passes, by the same rule as the comments in `CONTRIBUTING.md`: a
    folder name the product creates («_удалить») or a query someone typed is the SUBJECT
    of the message, and an English sentence about it would be a lie without it.
    """
    return [(line, text) for line, text in message_strings(source) if unquoted_cyrillic(text)]


def scanned() -> list[str]:
    """The modules `offenders` reads. Named apart from it because a scan that has
    quietly shrunk and a package that is clean produce the same green."""
    return [name for name in map(module_name, modules()) if name not in _CATALOGS]


def offenders() -> dict[str, list[tuple[int, str]]]:
    found = {}
    for name in scanned():
        russian = russian_messages((_ROOT / name).read_text(encoding="utf-8"))
        if russian:
            found[name] = russian
    return found


class TestNoRussianReachesTheLog(unittest.TestCase):
    def test_the_package_says_nothing_in_russian_to_a_reader_of_the_log(self):
        red = offenders()
        self.assertEqual(
            red, {},
            "\n".join([f"{name}:{line} {text!r}"
                       for name, lines in red.items() for line, text in lines]))

    def test_nothing_is_exempt_but_the_catalogs(self):
        """F239: the files F238 held back for F237 are read like the rest of the
        package. The list that let them past is gone, and this says so."""
        missing = [name for name in ("sorta/cli.py", "sorta/ui/process.py",
                                     "sorta/diagnostics.py") if name not in scanned()]
        self.assertEqual(missing, [])
        self.assertEqual(sorted(set(map(module_name, modules())) - set(scanned())),
                         sorted(_CATALOGS))

    def test_the_files_named_as_catalogs_exist(self):
        for name in sorted(_CATALOGS):
            with self.subTest(module=name):
                self.assertTrue((_ROOT / name).is_file())


class TestTheGuardGoesRed(unittest.TestCase):
    """Checked by putting the defect back: a watchdog nobody has seen bark is decoration."""

    def _added_to(self, module: str, line: str) -> list[str]:
        source = (_PACKAGE / module).read_text(encoding="utf-8")
        self.assertEqual(russian_messages(source), [], module)
        spoiled = f"{source}\n{line}\n"
        return [text for _, text in russian_messages(spoiled)]

    def test_a_russian_log_line_added_to_a_real_module_is_found(self):
        self.assertEqual(
            self._added_to("runlog.py", '_log.warning("runlog: не удалось открыть файл")'),
            ["runlog: не удалось открыть файл"])

    def test_a_russian_exception_added_to_a_real_module_is_found(self):
        self.assertEqual(
            self._added_to("dates.py", 'raise ValueError(f"дата {1} не разобрана")'),
            ["дата ", " не разобрана"])

    def test_a_russian_log_line_added_to_the_cli_is_found(self):
        """Spoiled in memory: writing to the real `cli.py` would race the parallel half
        of the gate, which imports it."""
        self.assertEqual(
            self._added_to("cli.py", '_log.error("cli: веса не скачались")'),
            ["cli: веса не скачались"])

    def test_a_russian_log_line_added_to_the_ui_pipeline_is_found(self):
        self.assertEqual(
            self._added_to("ui/process.py", '_log.exception("sorta ui: этап упал")'),
            ["sorta ui: этап упал"])

    def test_a_cited_folder_name_is_not_a_russian_message(self):
        source = '_log.info("sort: junk goes to «_удалить»")\n'
        self.assertEqual(russian_messages(source), [])
        self.assertEqual(len(message_strings(source)), 1)

    def test_a_string_the_program_does_not_say_is_not_read(self):
        source = 'PROMPT = "опиши кадр"\nrender("<h1>Отчёт</h1>")\n'
        self.assertEqual(message_strings(source), [])


class TestTheScanIsNotVacuous(unittest.TestCase):
    """A walk pointed at nothing finds nothing and looks exactly like a green gate."""

    def test_the_real_package_is_what_is_walked(self):
        names = [module_name(path) for path in modules()]
        self.assertGreater(len(names), 30)
        for expected in ("sorta/sorter.py", "sorta/ui/layout.py", "sorta/i18n.py"):
            self.assertIn(expected, names)

    def test_the_package_really_holds_messages_to_read(self):
        counted = sum(
            len(message_strings(path.read_text(encoding="utf-8"))) for path in modules()
        )
        self.assertGreater(counted, 200)

    def test_both_kinds_of_message_are_collected(self):
        source = '_log.warning("a: %s", x)\nraise ValueError("b")\n'
        self.assertEqual(message_strings(source), [(1, "a: %s"), (2, "b")])


if __name__ == "__main__":
    unittest.main()
