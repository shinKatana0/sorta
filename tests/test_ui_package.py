"""F182: `sorta/ui.py` became the `sorta/ui/` package, and it stayed one module to import.

A day's queue of ten features stood on one file, because two workers inside `ui.py` is a
guaranteed conflict — F152 came back with 18 divergences across 10 files, F160 with an
import that vanished and that neither gate caught on its own. The cut is by TAB rather
than by layer, because a feature lives in one tab and would otherwise touch all four
layers anyway.

The move is only worth anything if it is provably a move, and that rests on two
properties this file pins:

* `sorta.ui` still answers to every name it answered to (tests 1-2). Fifty test files
  and `sorta.cli` import from it; a package that exported only what the dispatcher calls
  would have turned the move into an edit of fifty files, and then nothing would prove
  the behaviour was unchanged.
* The tab modules stay a DAG with `common` at the bottom (tests 3-4). A cycle would not
  fail here — Python would raise a partially-initialised import at server start, on the
  user's machine — and two tabs importing each other is exactly the coupling the split
  was made to remove.

The last one (test 5) is the small thing that would have gone unnoticed: log lines are
still filed under `sorta.ui`, not under whichever module now holds the call.
"""
from __future__ import annotations

import ast
import importlib
import logging
import subprocess
import sys
import unittest
from pathlib import Path

from sorta import ui

_PKG = Path(ui.__file__).resolve().parent
_TABS = ("common", "layout", "slices", "review", "overview", "moves", "process",
         "page", "strings")

# A hand-written sample of what the suite and the CLI actually reach for through
# `sorta.ui`. Test 1 below covers everything mechanically; this list is here so the
# failure says WHICH name went missing rather than "some name did".
_NAMES_IN_USE = (
    "serve", "build_server", "DEFAULT_PORT", "BUSY_REFUSED_ROUTES",
    "RESTORE_ERROR_SENSITIVE", "RESTORE_ERROR_VIDEO",
    "_UI_STRINGS", "_t", "_render_index_html", "_INDEX_HTML_TEMPLATE",
    "PlanCache", "_plan_item_to_json", "_page_payload", "_parse_page_window",
    "_junk_payload", "_review_payload", "_dupes_payload", "_overview_payload",
    "_moves_payload", "_source_tree_payload", "_process_estimate_payload",
    "_ProcessState", "_SortState", "_UndoState", "_PipelineCancelled",
    "_pipeline_steps", "_stage_stats", "_browse_for_folder", "_geo_resolver",
    "_thumb_cache_clear", "_estimate_cache_clear", "_dupes_cache_clear",
    "_destination_json", "_destinations_for", "_album_dest", "_apply_settings",
)


def _intra_package_imports(path: Path) -> dict[str, set[str]]:
    """`from .x import ...` / `from . import x`, as the module names they name."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 1:
            if node.module:
                out.add(node.module.split(".")[0])
            else:
                out.update(a.name for a in node.names)
    return {path.stem: out}


class TestTheePackageExportsWhatTheModuleDid(unittest.TestCase):
    def test_every_name_a_tab_module_defines_is_reachable_on_sorta_ui(self):
        """The mechanical half. A helper added to a tab tomorrow is re-exported or this
        fails — which is the only thing keeping `ui.<anything>` from rotting away one
        name at a time."""
        for tab in _TABS:
            module = importlib.import_module(f"sorta.ui.{tab}")
            defined = {
                name for name, value in vars(module).items()
                if not name.startswith("__")
                and getattr(value, "__module__", module.__name__) == module.__name__
            }
            for name in sorted(defined):
                with self.subTest(f"{tab}.{name}"):
                    self.assertTrue(hasattr(ui, name),
                                    f"sorta.ui does not re-export {tab}.{name}")
                    self.assertIs(getattr(ui, name), getattr(module, name))

    def test_the_names_the_suite_and_the_cli_import_are_all_there(self):
        for name in _NAMES_IN_USE:
            with self.subTest(name):
                self.assertTrue(hasattr(ui, name), f"sorta.ui lost {name}")

    def test_importing_a_tab_module_first_still_works(self):
        """A cycle can hide behind import order: `sorta.ui` pulls the tabs in a working
        sequence, and a fresh interpreter that reaches for one tab directly does not."""
        for tab in _TABS:
            with self.subTest(tab):
                done = subprocess.run(
                    [sys.executable, "-c", f"import sorta.ui.{tab}"],
                    cwd=_PKG.parents[1], capture_output=True, text=True,
                )
                self.assertEqual(done.returncode, 0, done.stderr)


class TestTheTabsAreADag(unittest.TestCase):
    def test_the_tab_modules_have_no_import_cycle(self):
        graph: dict[str, set[str]] = {}
        for tab in _TABS:
            graph.update(_intra_package_imports(_PKG / f"{tab}.py"))
        # Kahn's algorithm: whatever is left when nothing has an empty in-edge set is a
        # cycle, and it is named in the failure.
        pending = {k: {d for d in v if d in graph} for k, v in graph.items()}
        while True:
            ready = [k for k, v in pending.items() if not v]
            if not ready:
                break
            for k in ready:
                del pending[k]
            for v in pending.values():
                v.difference_update(ready)
        self.assertEqual(pending, {}, f"import cycle between tab modules: {pending}")

    def test_common_and_strings_depend_on_no_tab(self):
        """The bottom of the package. `common` is what more than one tab needs; the
        moment it imports a tab, "shared" has become "everything", and the queue that
        F182 was written to end is back."""
        for tab in ("common", "strings"):
            with self.subTest(tab):
                self.assertEqual(_intra_package_imports(_PKG / f"{tab}.py")[tab], set())

    def test_only_the_package_root_knows_every_tab(self):
        """The route table is the one place allowed to reach into all of them — that is
        what makes two features in two tabs independent."""
        root = _intra_package_imports(_PKG / "__init__.py")["__init__"]
        self.assertEqual(root, set(_TABS))
        for tab in _TABS:
            with self.subTest(tab):
                self.assertNotIn("__init__", _intra_package_imports(_PKG / f"{tab}.py")[tab])


class TestTheLogNameSurvivedTheSplit(unittest.TestCase):
    def test_the_web_app_still_logs_under_sorta_ui(self):
        """`getLogger(__name__)` in a package would file the same line under
        `sorta.ui.common` — every filter and every grep over an existing run log is
        written against the old name."""
        self.assertEqual(ui._log.name, "sorta.ui")
        self.assertIs(ui._log, logging.getLogger("sorta.ui"))


if __name__ == "__main__":
    unittest.main()
