"""F100: the phases of the junk stage reach the web UI.

`GET /api/process/status` already carries `phase` (F84); what is new here is that the
junk stage fills it — and that its phases are MEASURABLE, the deep one included. So the
status must hand out a phase together with numbers that mean something (`done`/`total`
over the VLM candidates), not a phase with a stopwatch: the whole complaint was a bar
standing at 100% with no way to tell a working model from a hung process.
"""
from __future__ import annotations

import threading
import unittest

from sorta import ui
from sorta.junk import (
    CLASSIFY_PHASE_CLIP,
    CLASSIFY_PHASE_VLM,
    CLASSIFY_PHASE_WRITE,
)

from tests.test_ui_process import ProcessTestBase, _poll_until


class TestJunkPhaseInStatus(ProcessTestBase):
    def _patch_junk(self, phase: str, done: int, total: int,
                    block: threading.Event) -> None:
        """Stands in for junk.classify: names a phase, reports its counts, then waits."""
        calls = self.calls

        def blocking_junk(cfg, conn, classifier=None, verdicts_only=False,
                          progress=None):
            # F165: the same function is also the `classify` stage of the run, and
            # the phases under test are the ones the half AFTER faces reports.
            if verdicts_only:
                calls.append("classify")
                return
            calls.append("junk")
            progress.phase(phase)
            progress(done, total)
            block.wait(timeout=5)

        self._patch("classify_junk", blocking_junk)

    def test_vlm_phase_reaches_the_status_with_its_own_counts(self):
        block = threading.Event()
        self.patch_fast_stages()
        self._patch_junk(CLASSIFY_PHASE_VLM, 120, 1843, block)
        self.start_server()
        try:
            self.post("/api/process", {"source_dir": str(self.src_dir), "deep": True})
            running = _poll_until(self.status,
                                  lambda d: d["phase"] == CLASSIFY_PHASE_VLM)
            self.assertEqual(running["stage"], "junk")
            # measurable, unlike HDBSCAN: a real share, not a clock
            self.assertEqual((running["done"], running["total"]), (120, 1843))
            self.assertGreaterEqual(running["phase_elapsed"], 0.0)
        finally:
            block.set()
        final = _poll_until(self.status, lambda d: d["finished"])
        self.assertIsNone(final["error"])
        self.assertIsNone(final["phase"])

    def test_fast_phase_reaches_the_status_too(self):
        block = threading.Event()
        self.patch_fast_stages()
        self._patch_junk(CLASSIFY_PHASE_CLIP, 3000, 24196, block)
        self.start_server()
        try:
            self.post("/api/process", {"source_dir": str(self.src_dir)})
            running = _poll_until(self.status,
                                  lambda d: d["phase"] == CLASSIFY_PHASE_CLIP)
            self.assertEqual((running["done"], running["total"]), (3000, 24196))
        finally:
            block.set()
        _poll_until(self.status, lambda d: d["finished"])

    def test_a_later_phase_replaces_the_previous_one(self):
        # The denominator of the deep pass is different from the fast one; the client
        # may only show them together with the caption that explains the switch.
        block = threading.Event()
        self.patch_fast_stages()
        calls = self.calls
        seen = threading.Event()

        def two_phase_junk(cfg, conn, classifier=None, verdicts_only=False,
                           progress=None):
            if verdicts_only:  # F165: see `_patch_junk` above
                calls.append("classify")
                return
            calls.append("junk")
            progress.phase(CLASSIFY_PHASE_WRITE)
            progress(24196, 24196)
            seen.wait(timeout=5)
            progress.phase(CLASSIFY_PHASE_VLM)
            progress(0, 1843)
            block.wait(timeout=5)

        self._patch("classify_junk", two_phase_junk)
        self.start_server()
        try:
            self.post("/api/process", {"source_dir": str(self.src_dir)})
            first = _poll_until(self.status,
                                lambda d: d["phase"] == CLASSIFY_PHASE_WRITE)
            self.assertEqual(first["total"], 24196)
            seen.set()
            second = _poll_until(self.status, lambda d: d["phase"] == CLASSIFY_PHASE_VLM)
            self.assertEqual((second["done"], second["total"]), (0, 1843))
        finally:
            seen.set()
            block.set()
        _poll_until(self.status, lambda d: d["finished"])


class TestJunkPhaseI18n(ProcessTestBase):
    """Requirement 6: captions for the new phases in all three languages."""

    _KEYS = ("process_phase_junk_clip", "process_phase_junk_ocr",
             "process_phase_junk_vlm", "process_phase_junk_write",
             # F205: the two model passes that stopped sharing the deep tier's caption
             "process_phase_junk_pets_vlm", "process_phase_junk_rescue_vlm")

    def test_every_junk_phase_has_a_ui_string(self):
        # The client looks the caption up as "process_phase_" + the key from the
        # status; a phase without a string would surface as a raw identifier.
        from sorta import junk
        for name, value in vars(junk).items():
            if name.startswith("CLASSIFY_PHASE_"):
                self.assertIn(f"process_phase_{value}", ui._UI_STRINGS, name)

    def test_all_phase_keys_have_three_languages(self):
        for key in self._KEYS:
            entry = ui._UI_STRINGS[key]
            self.assertEqual(set(entry), {"ru", "en", "ja"}, key)
            for lang, text in entry.items():
                self.assertTrue(text.strip(), f"{key}/{lang}")

    def _html_of(self, lang: str) -> str:
        self.cfg.raw = {"language": lang}
        self.start_server()
        _status, body, _ctype = self.get("/")
        return body.decode("utf-8")

    def test_english_captions_served(self):
        html = self._html_of("en")
        self.assertIn("fast pass (CLIP)", html)
        self.assertIn("text detection (OCR)", html)
        self.assertIn("deep analysis (VLM)", html)
        self.assertIn("saving verdicts", html)

    def test_russian_captions_served(self):
        html = self._html_of("ru")
        self.assertIn("быстрый разбор (CLIP)", html)
        self.assertIn("поиск текста (OCR)", html)
        self.assertIn("глубокий анализ (VLM)", html)
        self.assertIn("запись вердиктов", html)

    def test_japanese_captions_served(self):
        html = self._html_of("ja")
        self.assertIn("高速判定 (CLIP)", html)
        self.assertIn("テキスト検出 (OCR)", html)
        self.assertIn("詳細解析 (VLM)", html)
        self.assertIn("判定を保存中", html)


if __name__ == "__main__":
    unittest.main()
