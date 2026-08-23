"""`app.js` runs to its last line, so every handler in it is wired.

The page script is one long IIFE. A statement that throws part way through does not
fail loudly: the browser reports it in a console nobody has open, the handlers ABOVE the
throw work, and every one below is silently never attached. On 2026-08-23 that was a boot
action placed among the tab handlers, 880 lines above the `var` it read — so "Browse" and
"Start" did nothing at all, with not one line in the log, because no request was ever
made. It shipped in an installer and was found in a VM.

Nothing that reads the text can see that, so this executes it: node runs `app.js` against
a DOM stub that answers every lookup, and the probe reports whether the script reached its
end and whether the two handlers furthest down the file were attached. What it does NOT
do is test behaviour — `fetch` never settles here, so no async path runs at all.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_PROBE = _ROOT / "tests" / "js" / "run_page_script.js"
_APP = _ROOT / "sorta" / "web" / "app" / "app.js"

_NODE = shutil.which("node")


def run_probe(script: Path) -> dict[str, object]:
    """-> {"reached_end": bool, "browse": bool, "start": bool, "output": str}."""
    result = subprocess.run([_NODE, str(_PROBE), str(script)],
                            capture_output=True, text=True, timeout=120, check=False)
    out = result.stdout
    return {"reached_end": "script reached the end" in out,
            "browse": "browse handler wired: true" in out,
            "start": "start handler wired: true" in out,
            "output": out + result.stderr}


@unittest.skipIf(_NODE is None, "node is not on PATH: the page script cannot be executed")
class TestTheScriptRunsToItsEnd(unittest.TestCase):
    def test_it_reaches_the_last_line(self):
        report = run_probe(_APP)
        self.assertTrue(report["reached_end"], report["output"])

    def test_the_handlers_furthest_down_the_file_are_wired(self):
        """The two the owner pressed when nothing happened."""
        report = run_probe(_APP)
        self.assertTrue(report["browse"], report["output"])
        self.assertTrue(report["start"], report["output"])


@unittest.skipIf(_NODE is None, "node is not on PATH: the page script cannot be executed")
class TestTheProbeWouldNotice(unittest.TestCase):
    """Guards the guard. A probe that reports success on a script that cannot run is
    worse than no probe, and this project has met that four times."""

    def test_a_script_that_throws_is_reported_as_dead(self):
        broken = _ROOT / "tests" / "js" / "_broken_for_the_guard.js"
        broken.write_text("(function () { undefinedName.field; })();", encoding="utf-8")
        try:
            report = run_probe(broken)
        finally:
            broken.unlink(missing_ok=True)
        self.assertFalse(report["reached_end"], report["output"])
        self.assertIn("SCRIPT DIED", str(report["output"]))

    def test_a_script_that_wires_nothing_is_reported_as_such(self):
        empty = _ROOT / "tests" / "js" / "_empty_for_the_guard.js"
        empty.write_text("(function () { })();", encoding="utf-8")
        try:
            report = run_probe(empty)
        finally:
            empty.unlink(missing_ok=True)
        self.assertTrue(report["reached_end"], report["output"])
        self.assertFalse(report["browse"], report["output"])


class TestTheProbeIsThere(unittest.TestCase):
    """Runs without node: the file has to exist and travel, or the skips above hide it."""

    def test_the_probe_is_tracked_and_names_the_script_it_runs(self):
        self.assertTrue(_PROBE.exists(), _PROBE)
        tracked = subprocess.run(["git", "ls-files", "tests/js/run_page_script.js"],
                                 cwd=_ROOT, capture_output=True, text=True, check=True)
        self.assertTrue(tracked.stdout.strip(), "tests/js/ is outside the .gitignore allow-list")

    def test_the_two_handlers_it_looks_for_exist_in_the_markup(self):
        """A probe asking about buttons nobody has would pass for the wrong reason."""
        page = (_ROOT / "sorta" / "web" / "page.html").read_text(encoding="utf-8")
        for button in ("process-browse-btn", "process-start-btn"):
            with self.subTest(button=button):
                self.assertIn(f'id="{button}"', page)
        probe = _PROBE.read_text(encoding="utf-8")
        self.assertIn("process-browse-btn", probe)
        self.assertIn("process-start-btn", probe)


if __name__ == "__main__":
    unittest.main()
