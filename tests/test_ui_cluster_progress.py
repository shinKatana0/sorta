"""F84: the phase of the current stage reaches the web UI.

`GET /api/process/status` grows a `phase` field (+ how long it has been running);
`phase = null` means the stage reports no phases — then the screen is exactly what it
was before this feature, which is what most of these tests check.
"""
from __future__ import annotations

import threading
import time
import unittest

from sorta import ui
from sorta.faces import CLUSTER_PHASE_CLUSTER, CLUSTER_PHASE_READ

from tests.test_ui_process import ProcessTestBase, _poll_until


class TestProcessStatePhase(unittest.TestCase):
    """The state object itself: phase, its clock, and the reset points."""

    def setUp(self):
        self.state = ui._ProcessState()

    def test_idle_snapshot_has_no_phase(self):
        snap = self.state.snapshot()
        self.assertIsNone(snap["phase"])
        self.assertEqual(snap["phase_elapsed"], 0.0)

    def test_set_phase_shows_in_snapshot_with_elapsed(self):
        self.state.try_start("x")
        self.state.set_phase(CLUSTER_PHASE_CLUSTER)
        snap = self.state.snapshot()
        self.assertEqual(snap["phase"], CLUSTER_PHASE_CLUSTER)
        self.assertGreaterEqual(snap["phase_elapsed"], 0.0)

    def test_elapsed_counts_from_the_start_of_the_phase(self):
        self.state.try_start("x")
        self.state.set_phase(CLUSTER_PHASE_CLUSTER)
        self.state._phase_started = time.monotonic() - 7.0  # as if 7s have passed
        self.assertGreaterEqual(self.state.snapshot()["phase_elapsed"], 7.0)

    def test_new_phase_restarts_the_clock(self):
        self.state.try_start("x")
        self.state.set_phase(CLUSTER_PHASE_READ)
        first = self.state._phase_started
        self.state.set_phase(CLUSTER_PHASE_CLUSTER)
        self.assertGreaterEqual(self.state._phase_started, first)
        self.assertEqual(self.state.snapshot()["phase"], CLUSTER_PHASE_CLUSTER)

    def test_next_stage_clears_the_phase(self):
        self.state.try_start("x")
        self.state.set_phase(CLUSTER_PHASE_CLUSTER)
        self.state.set_stage(2, "events")
        snap = self.state.snapshot()
        self.assertIsNone(snap["phase"])
        self.assertEqual(snap["phase_elapsed"], 0.0)

    def test_finish_clears_the_phase(self):
        self.state.try_start("x")
        self.state.set_phase(CLUSTER_PHASE_CLUSTER)
        self.state.finish(None)
        self.assertIsNone(self.state.snapshot()["phase"])

    def test_set_phase_none_stops_the_clock(self):
        self.state.try_start("x")
        self.state.set_phase(CLUSTER_PHASE_CLUSTER)
        self.state.set_phase(None)
        self.assertEqual(self.state.snapshot()["phase_elapsed"], 0.0)

    def test_unknown_total_drops_the_previous_one(self):
        # F84: within one stage a measurable phase can be followed by an unmeasurable
        # one (frames -> HDBSCAN). A total left over from the previous phase would
        # keep the bar filled and the numbers meaningless.
        self.state.try_start("x")
        self.state.set_progress(100, 100)
        self.state.set_progress(0, None)
        snap = self.state.snapshot()
        self.assertEqual(snap["total"], 0)
        self.assertEqual(snap["done"], 0)


class TestStageProgressChannel(unittest.TestCase):
    """The object stages are handed: a callback with a phase channel."""

    def test_call_updates_progress_and_phase_updates_phase(self):
        state = ui._ProcessState()
        state.try_start("x")
        cb = ui._StageProgress(state)
        cb(3, 10)
        cb.phase(CLUSTER_PHASE_READ)
        snap = state.snapshot()
        self.assertEqual((snap["done"], snap["total"]), (3, 10))
        self.assertEqual(snap["phase"], CLUSTER_PHASE_READ)

    def test_still_cancels_the_run_from_the_callback(self):
        state = ui._ProcessState()
        state.try_start("x")
        state.request_cancel()
        with self.assertRaises(ui._PipelineCancelled):
            ui._StageProgress(state)(1, 2)


class TestStatusEndpointPhase(ProcessTestBase):
    def test_phase_null_for_a_stage_without_phases(self):
        block = threading.Event()
        self.patch_fast_stages(block_stage="geo", block_event=block)
        self.start_server()
        try:
            self.post("/api/process", {"source_dir": str(self.src_dir)})
            running = _poll_until(self.status, lambda d: d["stage"] == "geo")
            self.assertIsNone(running["phase"])
            self.assertEqual(running["phase_elapsed"], 0.0)
        finally:
            block.set()
        _poll_until(self.status, lambda d: d["finished"])

    def test_phase_reported_by_faces_reaches_the_status(self):
        block = threading.Event()
        self.patch_fast_stages()

        def blocking_faces(cfg, conn, progress=None):
            """Stands in for cluster_faces: names the phase, then goes quiet."""
            self.calls.append("faces")
            progress.phase(CLUSTER_PHASE_CLUSTER)
            progress(0, None)
            block.wait(timeout=5)
            return None

        self._patch("detect_and_cluster", blocking_faces)
        self.start_server()
        try:
            self.post("/api/process",
                      {"source_dir": str(self.src_dir), "faces": True})
            running = _poll_until(self.status,
                                  lambda d: d["phase"] == CLUSTER_PHASE_CLUSTER)
            self.assertEqual(running["stage"], "faces")
            self.assertEqual(running["total"], 0)  # unmeasurable -> running bar
            self.assertGreaterEqual(running["phase_elapsed"], 0.0)
        finally:
            block.set()
        final = _poll_until(self.status, lambda d: d["finished"])
        self.assertIsNone(final["error"])
        self.assertIsNone(final["phase"])


class TestPhaseMarkup(ProcessTestBase):
    def test_phase_element_present_and_hidden_by_default(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn('id="process-phase"', html)
        # nothing to show until a stage names a phase — the old screen, unchanged
        self.assertIn('<div id="process-phase" class="process-phase" style="display:none">',
                      html)

    def test_render_reads_phase_and_elapsed_from_the_status(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn("function renderProcessPhase(data)", html)
        self.assertIn('I18N["process_phase_" + key]', html)
        self.assertIn("data.phase_elapsed", html)
        self.assertIn("renderProcessPhase(data);", html)

    def test_status_line_of_other_stages_is_untouched(self):
        # Regression: the phase caption is a separate line; the stage status text and
        # the indeterminate-bar logic (#37) stay exactly as they were.
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn("I18N.process_stage_progress_indeterminate", html)
        self.assertIn('bar.classList.add("indeterminate")', html)


class TestPhaseI18n(ProcessTestBase):
    """The captions of all four phases exist in ru/en/ja (F84 requirement 4)."""

    _KEYS = ("process_phase_cluster_read", "process_phase_cluster_hdbscan",
             "process_phase_cluster_inherit", "process_phase_cluster_write",
             "process_phase_elapsed")

    def test_every_cluster_phase_has_a_ui_string(self):
        # The client looks the caption up as "process_phase_" + the key from the
        # status; a phase without a string would surface as a raw identifier.
        from sorta import faces
        for name, value in vars(faces).items():
            if name.startswith("CLUSTER_PHASE_"):
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
        self.assertIn("clusters: reading embeddings", html)
        self.assertIn("clusters: grouping faces", html)
        self.assertIn("clusters: carrying names over", html)
        self.assertIn("clusters: saving", html)
        self.assertIn("{seconds}s so far", html)

    def test_russian_captions_served(self):
        html = self._html_of("ru")
        self.assertIn("кластеры: чтение эмбеддингов", html)
        self.assertIn("кластеры: группировка лиц", html)
        self.assertIn("кластеры: перенос имён", html)
        self.assertIn("кластеры: запись", html)
        self.assertIn("идёт {seconds} с", html)

    def test_japanese_captions_served(self):
        html = self._html_of("ja")
        self.assertIn("クラスタ: 埋め込みを読み込み中", html)
        self.assertIn("クラスタ: 顔をグループ化中", html)
        self.assertIn("クラスタ: 名前を引き継ぎ中", html)
        self.assertIn("クラスタ: 保存中", html)
        self.assertIn("経過 {seconds} 秒", html)


if __name__ == "__main__":
    unittest.main()
