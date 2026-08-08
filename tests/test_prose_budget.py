"""F231d: a module of `sorta/` loses prose freely and gains it only on purpose.

What is counted, because the count IS the rule: per `sorta/**/*.py`, the physical lines
held by COMMENT tokens plus the lines spanned by every bare string-literal statement —
module, class and function docstrings, and the attribute docstrings written under a
constant. A string the program prints is never counted: only a string that is a
statement of its own is prose.

`tests/prose_budgets.txt` holds one number per module. Over it is a failure; under it is
free, and the run says by how much. A budget that tracked the last measurement would
turn every unrelated edit into a red gate — the reason `fail_under` in pyproject.toml
sits below the coverage the suite actually reaches. Raising a number is an edit of that
file, so growth arrives in a diff, with a reason beside it, instead of arriving on its
own: the rule of 2026-08-08 lasted an hour against the person who wrote it.

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

_ROOT = Path(__file__).resolve().parent.parent
_PACKAGE = _ROOT / "sorta"
_BUDGETS = Path(__file__).resolve().parent / "prose_budgets.txt"

_HEADER = """\
# F231d: lines of comment and docstring per module of `sorta/`, the fact on 2026-08-08.
# tests/test_prose_budget.py counts and compares; it also writes this file, on
# `python tests/test_prose_budget.py --update`.
#
# Lowering a number needs nothing but the shrunken file. RAISING one is the point of the
# file: prose that grows has to be argued for in the diff that grows it.
"""


def modules() -> list[Path]:
    return sorted(_PACKAGE.rglob("*.py"))


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


def measured() -> dict[str, int]:
    return {
        module_name(path): prose_lines(path.read_text(encoding="utf-8"))
        for path in modules()
    }


def budgets() -> dict[str, int]:
    found: dict[str, int] = {}
    for line in _BUDGETS.read_text(encoding="utf-8").splitlines():
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


def write_budgets(numbers: dict[str, int]) -> None:
    width = max(len(name) for name in numbers) + 2
    body = "".join(f"{name:<{width}}{count}\n" for name, count in sorted(numbers.items()))
    _BUDGETS.write_text(_HEADER + body, encoding="utf-8")


class TestNoModuleGrowsPastItsBudget(unittest.TestCase):

    def test_the_package_is_within_budget(self):
        red = over_budget(measured(), budgets())
        self.assertEqual(
            red, {},
            "\n".join(["prose grew:", *red.values(),
                       "delete prose until it fits, or raise the number in "
                       "tests/prose_budgets.txt and say in the commit why it had to grow"]))

    def test_every_module_has_a_number_and_every_number_a_module(self):
        fact, budget = measured(), budgets()
        self.assertEqual(sorted(set(fact) - set(budget)), [],
                         "new modules with no budget — run "
                         "`python tests/test_prose_budget.py --update`")
        self.assertEqual(sorted(set(budget) - set(fact)), [],
                         "budgets for modules that are gone — same command")

    def test_a_module_that_shrank_is_offered_a_smaller_number(self):
        self.assertEqual(can_be_lowered({"a.py": 3, "b.py": 9}, {"a.py": 9, "b.py": 9}),
                         {"a.py": 3})
        slack = can_be_lowered(measured(), budgets())
        if slack:  # pragma: no cover — green once the budgets are the fact
            warnings.warn("prose budgets that can be lowered: " +
                          ", ".join(f"{name} -> {count}" for name, count in slack.items()),
                          stacklevel=1)


class TestTheRatchetGoesRed(unittest.TestCase):
    """A watchdog nobody has seen fail is not a watchdog."""

    def test_ten_lines_of_prose_added_to_a_real_module_break_the_gate(self):
        path = _PACKAGE / "tiers.py"
        name = module_name(path)
        source = path.read_text(encoding="utf-8")
        fattened = source + "\n" + "# an essay about what the code above does\n" * 10
        fact = {name: prose_lines(fattened)}
        self.assertEqual(fact[name], prose_lines(source) + 10)
        red = over_budget(fact, budgets())
        self.assertIn(name, red)
        self.assertIn(str(budgets()[name]), red[name])
        self.assertIn(str(fact[name]), red[name])

    def test_the_message_names_the_file_the_budget_and_the_fact(self):
        self.assertEqual(over_budget({"sorta/x.py": 12}, {"sorta/x.py": 10}),
                         {"sorta/x.py": "sorta/x.py: budget 10 lines of prose, now 12 (+2)"})

    def test_prose_under_the_budget_is_not_a_failure(self):
        self.assertEqual(over_budget({"sorta/x.py": 3}, {"sorta/x.py": 10}), {})


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
        names = measured()
        self.assertGreater(len(names), 30)
        self.assertIn("sorta/junk.py", names)
        self.assertIn("sorta/ui/strings.py", names)

    def test_the_budget_file_parses_into_numbers(self):
        budget = budgets()
        self.assertGreater(len(budget), 30)
        self.assertTrue(all(count >= 0 for count in budget.values()))


if __name__ == "__main__":
    if "--update" in sys.argv:
        write_budgets(measured())
        print(f"wrote {_BUDGETS} for {len(measured())} modules")
    else:
        unittest.main()
