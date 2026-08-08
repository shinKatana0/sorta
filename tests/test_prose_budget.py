"""F231d, F231g: a file loses prose freely and gains it only on purpose.

What is counted, because the count IS the rule: per `*.py`, the physical lines held by
COMMENT tokens plus the lines spanned by every bare string-literal statement — module,
class and function docstrings, and the attribute docstrings written under a constant. A
string the program prints is never counted: only a string that is a statement of its own
is prose.

Two scopes, one counter: `tests/prose_budgets.txt` holds a number per module of `sorta/`,
`tests/prose_budgets_tests.txt` a number per file of `tests/`. Over the number is a
failure; under it is free, and the run says by how much. A budget that tracked the last
measurement would turn every unrelated edit into a red gate — the reason `fail_under` in
pyproject.toml sits below the coverage the suite actually reaches. Raising a number is an
edit of that file, so growth arrives in a diff, with a reason beside it, instead of
arriving on its own: the rule of 2026-08-08 lasted an hour against the person who wrote it.

The suite came under the ratchet holding as much prose as the whole package (13 532 lines
against 13 510) at a third of the density, and was fixed as it stood, nothing cleaned: a
test's docstring is usually the statement of a requirement, so it passes the rule already.
These numbers are a ceiling and were never meant to be a reduction.

There is deliberately no limit on a single block. A trap sometimes needs six lines, and a
per-block limit would get it split in two rather than deleted.
"""
from __future__ import annotations

import ast
import io
import sys
import tokenize
import unittest
import warnings
from pathlib import Path
from typing import NamedTuple

_ROOT = Path(__file__).resolve().parent.parent


class Scope(NamedTuple):
    """A tree of `*.py` and the file of numbers that holds its prose in place."""

    root: Path
    budget_file: Path
    header: str


PACKAGE = Scope(
    root=_ROOT / "sorta",
    budget_file=_ROOT / "tests" / "prose_budgets.txt",
    header="""\
# F231d: lines of comment and docstring per module of `sorta/`, the fact on 2026-08-08.
# tests/test_prose_budget.py counts and compares; it also writes this file, on
# `python tests/test_prose_budget.py --update`.
#
# Lowering a number needs nothing but the shrunken file. RAISING one is the point of the
# file: prose that grows has to be argued for in the diff that grows it.
""",
)

SUITE = Scope(
    root=_ROOT / "tests",
    budget_file=_ROOT / "tests" / "prose_budgets_tests.txt",
    header="""\
# F231g: lines of comment and docstring per file of `tests/`, the fact on 2026-08-08.
# A file of its own rather than more lines in prose_budgets.txt: two other features were
# editing that one the day this arrived, and separate files merge without a conflict.
# tests/test_prose_budget.py counts and compares; it also writes this file, on
# `python tests/test_prose_budget.py --update`.
#
# Lowering a number needs nothing but the shrunken file. RAISING one is the point of the
# file: prose that grows has to be argued for in the diff that grows it.
""",
)


def modules(scope: Scope) -> list[Path]:
    return sorted(scope.root.rglob("*.py"))


def module_name(path: Path) -> str:
    return path.relative_to(_ROOT).as_posix()


def comment_lines(source: str) -> int:
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    return sum(1 for token in tokens if token.type == tokenize.COMMENT)


def docstring_lines(source: str) -> int:
    return sum(
        node.end_lineno - node.lineno + 1
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
        and node.end_lineno is not None
    )


def prose_lines(source: str) -> int:
    return comment_lines(source) + docstring_lines(source)


def measured(scope: Scope) -> dict[str, int]:
    return {
        module_name(path): prose_lines(path.read_text(encoding="utf-8"))
        for path in modules(scope)
    }


def budgets(scope: Scope) -> dict[str, int]:
    found: dict[str, int] = {}
    for line in scope.budget_file.read_text(encoding="utf-8").splitlines():
        text = line.split("#", 1)[0].strip()
        if text:
            name, number = text.rsplit(maxsplit=1)
            found[name] = int(number)
    return found


def over_budget(fact: dict[str, int], budget: dict[str, int]) -> dict[str, str]:
    """Module -> why it is red, for every module that holds more prose than it may."""
    return {
        name: f"{name}: budget {budget[name]} lines of prose, now {count} (+{count - budget[name]})"
        for name, count in fact.items()
        if name in budget and count > budget[name]
    }


def can_be_lowered(fact: dict[str, int], budget: dict[str, int]) -> dict[str, int]:
    """Module -> the number its budget could say now. Never a failure, only an offer."""
    return {
        name: count
        for name, count in fact.items()
        if name in budget and count < budget[name]
    }


def write_budgets(scope: Scope, numbers: dict[str, int]) -> None:
    width = max(len(name) for name in numbers) + 2
    body = "".join(f"{name:<{width}}{count}\n" for name, count in sorted(numbers.items()))
    scope.budget_file.write_text(scope.header + body, encoding="utf-8")


class _RatchetOverAScope:
    """The three assertions every scope owes; the subclass says which tree it is about."""

    scope: Scope

    def test_nothing_in_the_scope_grew_past_its_budget(self):
        red = over_budget(measured(self.scope), budgets(self.scope))
        self.assertEqual(
            red, {},
            "\n".join(["prose grew:", *red.values(),
                       "delete prose until it fits, or raise the number in "
                       f"{module_name(self.scope.budget_file)} and say in the commit why "
                       "it had to grow"]))

    def test_every_file_has_a_number_and_every_number_a_file(self):
        fact, budget = measured(self.scope), budgets(self.scope)
        self.assertEqual(sorted(set(fact) - set(budget)), [],
                         "new files with no budget — run "
                         "`python tests/test_prose_budget.py --update`")
        self.assertEqual(sorted(set(budget) - set(fact)), [],
                         "budgets for files that are gone — same command")

    def test_a_file_that_shrank_is_offered_a_smaller_number(self):
        slack = can_be_lowered(measured(self.scope), budgets(self.scope))
        if slack:  # pragma: no cover — green once the budgets are the fact
            warnings.warn("prose budgets that can be lowered: " +
                          ", ".join(f"{name} -> {count}" for name, count in slack.items()),
                          stacklevel=1)


class TestThePackageIsWithinBudget(_RatchetOverAScope, unittest.TestCase):
    scope = PACKAGE


class TestTheSuiteIsWithinBudget(_RatchetOverAScope, unittest.TestCase):
    scope = SUITE


class TestTheRatchetGoesRed(unittest.TestCase):
    """A watchdog nobody has seen fail is not a watchdog."""

    def _ten_lines_of_prose_break(self, scope: Scope, path: Path) -> None:
        name = module_name(path)
        source = path.read_text(encoding="utf-8")
        fattened = source + "\n" + "# an essay about what the code above does\n" * 10
        fact = {name: prose_lines(fattened)}
        self.assertEqual(fact[name], prose_lines(source) + 10)
        red = over_budget(fact, budgets(scope))
        self.assertIn(name, red)
        self.assertIn(str(budgets(scope)[name]), red[name])
        self.assertIn(str(fact[name]), red[name])

    def test_ten_lines_of_prose_added_to_a_real_module_break_the_gate(self):
        self._ten_lines_of_prose_break(PACKAGE, PACKAGE.root / "tiers.py")

    def test_ten_lines_of_prose_added_to_a_real_test_break_the_gate(self):
        self._ten_lines_of_prose_break(SUITE, SUITE.root / "test_dates.py")

    def test_the_message_names_the_file_the_budget_and_the_fact(self):
        self.assertEqual(over_budget({"sorta/x.py": 12}, {"sorta/x.py": 10}),
                         {"sorta/x.py": "sorta/x.py: budget 10 lines of prose, now 12 (+2)"})

    def test_prose_under_the_budget_is_not_a_failure(self):
        self.assertEqual(over_budget({"sorta/x.py": 3}, {"sorta/x.py": 10}), {})

    def test_a_shrunken_file_is_the_only_thing_offered_a_smaller_number(self):
        self.assertEqual(can_be_lowered({"a.py": 3, "b.py": 9}, {"a.py": 9, "b.py": 9}),
                         {"a.py": 3})


class TestWhatCounts(unittest.TestCase):
    """The definition, written as examples: this is the whole of what a budget measures."""

    def test_a_comment_costs_a_line_wherever_it_sits(self):
        self.assertEqual(comment_lines("# one\n# two\nx = 1  # three\n"), 3)

    def test_a_docstring_costs_the_lines_it_spans(self):
        self.assertEqual(docstring_lines('def f():\n    """one\n    two\n    """\n'), 3)

    def test_an_attribute_docstring_counts_too(self):
        self.assertEqual(docstring_lines('X = 1\n"""what X is for."""\n'), 1)

    def test_a_string_the_program_uses_is_not_prose(self):
        source = 'MESSAGE = "not a docstring"\n\n\ndef f():\n    return log("nor this")\n'
        self.assertEqual(prose_lines(source), 0)

    def test_the_two_halves_are_added_up(self):
        self.assertEqual(prose_lines('"""a module."""\n# and a comment\nx = 1\n'), 2)


class TestTheScanIsNotVacuous(unittest.TestCase):
    """A counter pointed at nothing counts nothing and looks exactly like a green gate."""

    def test_the_real_package_is_what_is_being_read(self):
        names = measured(PACKAGE)
        self.assertGreater(len(names), 30)
        self.assertIn("sorta/junk.py", names)
        self.assertIn("sorta/ui/strings.py", names)

    def test_the_real_suite_is_what_is_being_read(self):
        names = measured(SUITE)
        self.assertGreater(len(names), 200)
        self.assertIn("tests/conftest.py", names)
        self.assertIn("tests/test_prose_budget.py", names)

    def test_both_budget_files_parse_into_numbers(self):
        for scope, least in ((PACKAGE, 30), (SUITE, 200)):
            budget = budgets(scope)
            self.assertGreater(len(budget), least)
            self.assertTrue(all(count >= 0 for count in budget.values()))

    def test_the_two_scopes_do_not_share_a_file_of_numbers(self):
        self.assertNotEqual(PACKAGE.budget_file, SUITE.budget_file)


if __name__ == "__main__":
    if "--update" in sys.argv:
        for updated in (PACKAGE, SUITE):
            numbers = measured(updated)
            write_budgets(updated, numbers)
            print(f"wrote {updated.budget_file} for {len(numbers)} files")
    else:
        unittest.main()
