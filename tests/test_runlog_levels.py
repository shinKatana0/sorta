"""The run log must actually record what it was built to record.

Caught on a live run: `stage=... elapsed=...` lines never appeared in the file.
configure_logging set the `sorta` logger to the console level (WARNING by default),
and a record is dropped by the level of the logger it was emitted on — long before
any handler is consulted. The file sink was attached and idle.
"""
from __future__ import annotations

import logging
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from sorta import runlog
from sorta.config import configure_logging


class TestRunLogCapturesStageLines(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.log = Path(self.tmp.name) / "sorta.log"
        self.root = logging.getLogger()
        self.before = list(self.root.handlers)
        self.sorta_level = logging.getLogger("sorta").level
        self.addCleanup(self._restore)

    def _restore(self):
        for handler in list(self.root.handlers):
            if handler not in self.before:
                self.root.removeHandler(handler)
                handler.close()
        logging.getLogger("sorta").setLevel(self.sorta_level)

    def _configure(self, level: str = "WARNING"):
        with unittest.mock.patch.dict(
                "os.environ", {runlog.ENV_LOG_FILE: str(self.log)}):
            configure_logging(level)

    def test_stage_lines_reach_the_file_at_default_console_level(self):
        """The regression: log_level=WARNING used to swallow the INFO stage lines."""
        self._configure("WARNING")
        with runlog.stage_timer("junk") as result:
            result.processed = 7
        text = self.log.read_text(encoding="utf-8")
        self.assertIn("stage=junk", text)
        self.assertIn("processed=7", text)

    def test_console_handler_keeps_the_configured_level(self):
        """Lowering the logger must not make the console chattier."""
        self._configure("WARNING")
        console = [h for h in logging.getLogger("sorta").handlers
                   if getattr(h, "_sorta_handler", False)]
        self.assertTrue(console, "консольный обработчик пропал")
        for handler in console:
            self.assertEqual(handler.level, logging.WARNING)

    def test_logger_is_lowered_only_to_what_the_file_wants(self):
        self._configure("WARNING")
        self.assertLessEqual(logging.getLogger("sorta").level, runlog.file_log_level())

    def test_debug_console_level_is_not_raised_by_the_file_sink(self):
        self._configure("DEBUG")
        self.assertEqual(logging.getLogger("sorta").level, logging.DEBUG)


class TestTestSuiteIsolation(unittest.TestCase):
    def test_log_and_preview_paths_are_sandboxed(self):
        """conftest must keep the suite out of the user's real %LOCALAPPDATA%."""
        import os

        for var in (runlog.ENV_LOG_FILE, "SORTA_PREVIEW_DIR"):
            value = os.environ.get(var, "")
            self.assertTrue(value, f"{var} не задан — тесты пишут в боевой путь")
            self.assertIn("sorta-tests", value)


if __name__ == "__main__":
    unittest.main()
