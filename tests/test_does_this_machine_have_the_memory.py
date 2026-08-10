"""F237: a run says whether this machine has the memory for it, BEFORE it starts.

The report: on a virtual machine with 4 GB a run with the classification was killed by
the kernel — no traceback, no `except`, no last line in the log. The same profile on 6 GB
finished. Nothing can be caught after the fact, because the process is not the one that
decides to end, so the only place with anything to say is before the start.

What is under test here:

* the reading is per platform and without a new dependency (`/proc/meminfo` on Linux,
  `GlobalMemoryStatusEx` on Windows), and "I do not know" is a legitimate answer that
  produces silence rather than an invented number;
* the line appears only BELOW the threshold — a warning shown on every run is one nobody
  reads, which cost this project a note already (F233);
* the console and the run screen say the same sentence, out of the same catalog entry,
  with the same two numbers;
* it is a warning: nothing is refused and the start button is not touched.
"""
from __future__ import annotations

import contextlib
import io
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from sorta import cli, diagnostics, i18n, ui
from sorta.diagnostics import (
    MEMORY_FLOOR_MB, MemoryHealth, _linux_available_mb, _windows_available_mb,
    available_memory_mb, memory_health,
)
from sorta.ui.strings import _UI_STRINGS
from tests.test_ui import UiServerTestBase

_LANGS = ("ru", "en", "ja")
_APP_JS = Path(__file__).resolve().parent.parent / "sorta" / "web" / "app" / "app.js"

MEMINFO = """\
MemTotal:        8039140 kB
MemFree:          204152 kB
MemAvailable:    3405600 kB
Buffers:           64160 kB
"""


def _meminfo(text: str):
    """Point the Linux reader at a file of our own instead of this machine's."""
    tmp = tempfile.NamedTemporaryFile("w", suffix=".meminfo", delete=False,
                                      encoding="utf-8")
    tmp.write(text)
    tmp.close()
    return mock.patch.object(diagnostics, "_MEMINFO", Path(tmp.name))


class TestReadingTheFreeMemory(unittest.TestCase):
    def test_mem_available_is_read_in_megabytes(self):
        with _meminfo(MEMINFO):
            self.assertEqual(_linux_available_mb(), 3405600 * 1024 // (1024 * 1024))

    def test_a_kernel_without_mem_available_is_a_dont_know(self):
        """MemFree is not an answer to this question: on a machine that has been up for a
        day most of it is page cache, which a run may have back for the asking."""
        with _meminfo("MemTotal:  8039140 kB\nMemFree:  204152 kB\n"):
            self.assertIsNone(_linux_available_mb())

    def test_an_unreadable_meminfo_is_a_dont_know(self):
        with mock.patch.object(diagnostics, "_MEMINFO", Path("/no/such/meminfo")):
            self.assertIsNone(_linux_available_mb())

    def test_a_line_that_is_not_a_number_is_a_dont_know(self):
        with _meminfo("MemAvailable:    plenty kB\n"):
            self.assertIsNone(_linux_available_mb())

    def test_linux_is_asked_through_proc_meminfo(self):
        with mock.patch.object(sys, "platform", "linux"), \
                mock.patch.object(diagnostics, "_linux_available_mb", return_value=7):
            self.assertEqual(available_memory_mb(), 7)

    def test_windows_is_asked_through_the_kernel_call(self):
        with mock.patch.object(sys, "platform", "win32"), \
                mock.patch.object(diagnostics, "_windows_available_mb", return_value=9):
            self.assertEqual(available_memory_mb(), 9)

    def test_a_platform_with_no_reading_says_so(self):
        with mock.patch.object(sys, "platform", "darwin"):
            self.assertIsNone(available_memory_mb())

    def test_the_windows_call_never_raises_off_windows(self):
        self.assertIsInstance(_windows_available_mb(), (int, type(None)))

    def test_the_probe_of_this_machine_never_raises(self):
        answer = available_memory_mb()
        self.assertTrue(answer is None or answer > 0)


class TestTheThreshold(unittest.TestCase):
    def test_the_floor_is_the_size_that_died(self):
        """4 GB is the machine the kernel killed a run on (2026-08-09); 6 GB is the one it
        finished on. The check reads FREE memory, always below the total, so the floor is
        the first of the two — the second would fire on a machine known to be enough."""
        self.assertEqual(MEMORY_FLOOR_MB, 4000)

    def test_the_guides_state_the_same_two_numbers(self):
        guides = Path(__file__).resolve().parent.parent / "docs" / "guide"
        for name, floor in (("en", "4 GB"), ("ru", "4 ГБ"), ("ja", "4 GB")):
            with self.subTest(lang=name):
                text = (guides / f"user-guide.{name}.md").read_text(encoding="utf-8")
                self.assertIn(floor, text)

    def test_below_the_floor_is_low(self):
        self.assertTrue(MemoryHealth(MEMORY_FLOOR_MB - 1).low)

    def test_at_the_floor_is_not_low(self):
        self.assertFalse(MemoryHealth(MEMORY_FLOOR_MB).low)

    def test_an_unknown_amount_is_not_low(self):
        self.assertFalse(MemoryHealth(None).low)

    def test_the_health_of_this_machine_carries_the_floor(self):
        self.assertEqual(memory_health(available_mb=2048).needed_mb, MEMORY_FLOOR_MB)
        self.assertTrue(memory_health(available_mb=2048).low)


def _low(available_mb: int = 3325):
    return mock.patch.object(cli, "memory_health",
                             return_value=MemoryHealth(available_mb))


def _plenty():
    return mock.patch.object(cli, "memory_health",
                             return_value=MemoryHealth(MEMORY_FLOOR_MB * 4))


def _unknown():
    return mock.patch.object(cli, "memory_health", return_value=MemoryHealth(None))


def _printed(lang: str) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cli._print_low_memory_warning(lang)
    return buf.getvalue()


class TestTheConsoleSaysItBeforeTheRun(unittest.TestCase):
    def test_it_names_what_is_free_and_what_is_needed(self):
        with _low():
            printed = _printed("en")
        self.assertIn("3.3 GB", printed)
        self.assertIn("4.0 GB", printed)

    def test_it_names_both_switches_in_every_language(self):
        expected = {"ru": ("Глубокий анализ (VLM)", "классификацию кадров"),
                    "en": ("Deep analysis (VLM)", "frame classification"),
                    "ja": ("詳細分析（VLM）", "コマの分類")}
        for lang in _LANGS:
            with self.subTest(lang=lang), _low():
                printed = _printed(lang)
                for phrase in expected[lang]:
                    self.assertIn(phrase, printed)

    def test_enough_memory_says_nothing(self):
        with _plenty():
            self.assertEqual(_printed("en"), "")

    def test_an_unreadable_machine_says_nothing(self):
        """Silence, not a made-up number: the alternative was a threshold invented for a
        platform nobody measured."""
        with _unknown():
            self.assertEqual(_printed("en"), "")


class TestTheRunStartsAnyway(unittest.TestCase):
    """Memory moves — a browser closed a minute later gives a gigabyte back. Naming the
    risk is the whole job; deciding for the person is not."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        root = Path(self.tmp.name)
        (root / "src").mkdir()
        self.cfg_path = root / "config.yaml"
        self.cfg_path.write_text(
            f'sources: ["{(root / "src").as_posix()}"]\n'
            f'database: "{(root / "test.db").as_posix()}"\n'
            'language: en\n',
            encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self) -> str:
        steps = [("alpha", lambda cfg, conn, cb: "done")]
        buf = io.StringIO()
        with mock.patch.object(cli, "_pipeline_steps", lambda: steps), \
                contextlib.redirect_stdout(buf):
            cli._cmd_run(str(self.cfg_path))
        return buf.getvalue()

    def test_the_warning_stands_above_the_first_stage_and_the_stage_runs(self):
        with _low():
            out = self._run()
        self.assertIn("Little free memory", out)
        self.assertLess(out.index("Little free memory"), out.index("[stage 1/1] alpha"))
        self.assertIn("done", out)

    def test_a_run_on_a_roomy_machine_is_not_told_anything(self):
        with _plenty():
            self.assertNotIn("free memory", self._run())


class TestTheRunScreenSaysTheSame(UiServerTestBase):
    def test_the_env_route_carries_the_two_numbers(self):
        with mock.patch.object(ui.process, "memory_health",
                               return_value=MemoryHealth(3000)):
            self.start_server()
            _status, body, _ctype = self.get("/api/env")
        memory = json.loads(body)["memory"]
        self.assertEqual(memory, {"low": True, "free_mb": 3000,
                                  "needed_mb": MEMORY_FLOOR_MB})

    def test_a_machine_that_cannot_be_asked_carries_no_number(self):
        with mock.patch.object(ui.process, "memory_health",
                               return_value=MemoryHealth(None)):
            self.start_server()
            _status, body, _ctype = self.get("/api/env")
        memory = json.loads(body)["memory"]
        self.assertFalse(memory["low"])
        self.assertIsNone(memory["free_mb"])

    def test_the_line_is_hidden_until_the_answer_comes_back(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn('id="env-memory-warning" class="env-warning" style="display:none"',
                      html)
        self.assertIn("renderMemoryWarning(data.memory)", html)

    def test_the_screen_and_the_console_share_the_sentence(self):
        for lang in _LANGS:
            with self.subTest(lang=lang):
                template = i18n.cli_text("cli.run.low_memory", lang,
                                         free="{free}", needed="{needed}").strip()
                self.assertEqual(_UI_STRINGS["low_memory_warning"][lang], template)

    def test_the_sentence_reaches_the_page_in_every_language(self):
        for lang, phrase in (("ru", "Свободной памяти мало"),
                             ("en", "Little free memory"),
                             ("ja", "空きメモリが少なめです")):
            with self.subTest(lang=lang):
                self.cfg.raw = {"language": lang}
                self.start_server()
                _status, body, _ctype = self.get("/")
                self.assertIn(phrase, body.decode("utf-8"))
                self.tearDown()
                self.setUp()

    def test_the_start_button_is_left_alone(self):
        """The acceptance line "the start button works either way", read off the only
        function that could break it."""
        source = _APP_JS.read_text(encoding="utf-8")
        body = re.search(r"function renderMemoryWarning\(memory\) \{.*?\n  \}",
                         source, re.S)
        self.assertIsNotNone(body)
        self.assertNotIn("disabled", body.group(0))
        self.assertNotIn("process-start-btn", body.group(0))


if __name__ == "__main__":
    unittest.main()
