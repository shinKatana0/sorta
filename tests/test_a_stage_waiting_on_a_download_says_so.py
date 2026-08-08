"""F229: a stage waiting for a download says so instead of freezing a frame counter.

The owner's words, from the virtual machine of 2026-08-08:

    Stage verdicts (4/7):  0 of 8 — what is going on? there are only 8 photographs in
    the test, and as a user I do not understand what is stuck and what is happening.

Nothing was stuck: 1.6 GB of CLIP weights were coming down, and the line beside them
counted FRAMES — of which not one can be processed while the model is missing. The
number was true and the unit was not, and on eight photographs the download IS the whole
run, so the frozen counter was the entire screen.

F225 put the megabytes on that screen and stopped half way: the counter that cannot move
stayed next to them. Everything here is about that half — the stage line, in the browser
and in a terminal, for the stretch of time in which frames are impossible, and the frame
counter back the moment they are not.
"""
from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from sorta import cli, i18n, progress, tiers, ui
from sorta.config import Config
from sorta.ui import strings as ui_strings
from tests.test_ui_process import ProcessTestBase

_LANGS: tuple[i18n.Lang, ...] = ("ru", "en", "ja")
_ROOT = Path(__file__).resolve().parent.parent
_APP_JS = _ROOT / "sorta" / "web" / "app" / "app.js"


class _FakeBar:
    """What a stage is handed to report with: a callable with a `phase` channel.

    The same shape `progress.TaskProgress` and `runlog._Observed` have — the point of the
    test is that `_build_clip` talks through that channel and not to a rich object it
    would then have to know about.
    """

    def __init__(self) -> None:
        self.phases: list[str | None] = []

    def __call__(self, done: int, total: int | None = None) -> None:
        pass

    def phase(self, name: str | None) -> None:
        self.phases.append(name)


def _build_clip(bar: object | None = None, *, missing=("ViT-L-14",), build=None):
    """`cli._build_clip` with the network and the disk taken out; returns what it printed."""
    printed: list[str] = []
    factory = build if build is not None else (lambda settings: None)

    def say(*parts: object, **_kwargs: object) -> None:
        # The signature of the real `print`: the refusal path takes the traceback of the
        # failed download through it, and a stub that only accepts one argument would
        # fail the test for a reason that has nothing to do with what it asserts.
        printed.append(" ".join(str(part) for part in parts))

    with mock.patch.object(cli, "clip_classifier", factory), \
            mock.patch.object(cli.tiers, "stage_downloads", return_value=tuple(missing)), \
            mock.patch.object(tiers, "downloaded_bytes", return_value=0), \
            mock.patch("builtins.print", side_effect=say):
        cli._build_clip(Config(), "classify", bar)
    return printed


# --- the run screen ------------------------------------------------------------------


class TestTheStageLineNamesTheWait(ProcessTestBase):
    """§«Что делать»: while the model is coming, the stage line is about the model."""

    def _waiting(self, state) -> bool:
        return state.snapshot()["stage_waiting_download"]

    def _started_on(self, stage: str) -> ui.process._ProcessState:
        state = ui.process._ProcessState()
        state.try_start(str(self.src_dir))
        state.set_stage(4, stage)
        state.set_progress(0, 8)  # the eight frames of the report
        return state

    def test_a_stage_fetching_its_own_model_is_waiting(self):
        state = self._started_on("classify")
        state.set_download("classify", ("ViT-L-14",), 512 * 1_000_000)
        self.assertTrue(self._waiting(state))

    def test_and_the_frame_counter_comes_back_when_the_model_is_on_disk(self):
        state = self._started_on("classify")
        state.set_download("classify", ("ViT-L-14",), 512 * 1_000_000)
        state.set_download(None)
        snapshot = state.snapshot()
        self.assertFalse(snapshot["stage_waiting_download"])
        # ...to the very numbers it had: nothing about the wait touches the counter.
        self.assertEqual((snapshot["done"], snapshot["total"]), (0, 8))

    def test_a_run_that_downloads_nothing_never_waits(self):
        """The full cache — the line has to behave exactly as it did before F229."""
        self.assertFalse(self._waiting(self._started_on("classify")))

    def test_a_download_belonging_to_another_stage_does_not_stop_this_one_counting(self):
        """`download` is one field for the whole run; the claim is about ONE stage.

        A stage that is counting frames of its own must keep counting them, whatever is
        being fetched elsewhere — hiding a moving counter is the failure this feature is
        explicitly told not to trade for.
        """
        state = self._started_on("faces")
        state.set_download("classify", ("ViT-L-14",), 0)
        self.assertFalse(self._waiting(state))

    def test_a_finished_run_is_not_waiting_for_anything(self):
        state = self._started_on("classify")
        state.set_download("classify", ("ViT-L-14",), 0)
        state.finish(None)
        self.assertFalse(self._waiting(state))

    def test_the_state_travels_from_the_factory_that_does_the_downloading(self):
        """End to end: the same run, the same state object, no new machinery in between.

        The stage asks its classifier for something, the factory sees the weights are
        missing and reports the download — and it is at THAT moment, inside the build,
        that the status has to say the stage is waiting.
        """
        state = self._started_on("landmarks")
        seen: list[dict] = []

        def fake_clip(_settings):
            seen.append(state.snapshot())
            return lambda paths, prompts: []

        steps = dict(ui.process._pipeline_steps(state.set_download))
        self.patch_fast_stages()
        self._patch("detect_landmarks",
                    lambda cfg, conn, classifier=None, progress=None:
                    classifier(["a.jpg"], ["a photo"]))
        self._patch("clip_classifier", fake_clip)
        self._patch("stage_downloads", lambda stage, states=None: ("ViT-L-14",))
        with mock.patch.object(tiers, "downloaded_bytes", return_value=0):
            steps["landmarks"](self.cfg, self.conn, lambda *_a, **_k: None)
        self.assertTrue(seen[0]["stage_waiting_download"])
        self.assertEqual(seen[0]["download"]["weights"], ["ViT-L-14"])
        # ...and afterwards the stage is a stage counting frames again.
        self.assertFalse(state.snapshot()["stage_waiting_download"])


class TestThePageDrawsTheWaitInsteadOfTheZero(unittest.TestCase):
    """The client half — it reads the flag, and it says words rather than a number."""

    def setUp(self):
        self.source = _APP_JS.read_text(encoding="utf-8")

    def test_the_flag_is_read_and_the_sentence_is_shown(self):
        self.assertIn("data.stage_waiting_download", self.source)
        self.assertIn("I18N.process_stage_waiting_model", self.source)

    def test_the_wait_is_decided_before_the_frame_counter_is_drawn(self):
        """Order is the whole fix: the branch that draws "{done} of {all}" must never be
        reached while frames are impossible."""
        wait = self.source.index("data.stage_waiting_download")
        counter = self.source.index("I18N.process_stage_progress_indeterminate")
        self.assertLess(wait, counter)

    def test_the_counter_is_not_removed_from_the_page(self):
        """§«Чего НЕ делать»: once frames are going, their number is the most useful
        thing on the screen — only the stretch in which they are impossible changes."""
        self.assertIn("I18N.process_stage_progress", self.source)

    def test_no_percentage_of_the_download_is_invented_beside_it(self):
        """The megabytes are the F225 line and stay there alone: two quantities in one
        place stop being told apart within a month."""
        start = self.source.index("data.stage_waiting_download")
        block = self.source[start:start + 600]
        self.assertNotIn("done_mb", block)
        self.assertNotIn("info.mb", block)

    def test_the_sentence_exists_in_three_languages(self):
        for lang in _LANGS:
            with self.subTest(lang=lang):
                text = ui_strings._UI_STRINGS["process_stage_waiting_model"][lang]
                self.assertIn("{stage}", text)
                self.assertIn("{index}", text)
                self.assertIn("{total}", text)
                # The one thing it may not carry: a count of frames.
                self.assertNotIn("{done}", text)
                self.assertNotIn("{all}", text)


# --- the terminal ---------------------------------------------------------------------


class TestTheTerminalSaysTheSameThing(unittest.TestCase):
    """§«То же самое в терминале»: `sorta run` draws a bar over the same stretch."""

    def test_the_wait_is_printed_beside_the_announcement_of_the_download(self):
        printed = _build_clip()
        self.assertIn(i18n.cli_text("cli.download.waiting", "en"), printed)
        # ...after the sentence that says what is being fetched, not instead of it.
        self.assertIn(tiers.download_notice("classify", ("ViT-L-14",), "en"), printed)

    def test_the_stage_bar_carries_the_wait_and_then_stops_carrying_it(self):
        bar = _FakeBar()
        _build_clip(bar)
        self.assertEqual(bar.phases, [cli._DOWNLOAD_PHASE, None])

    def test_a_refused_download_still_takes_the_caption_down(self):
        """A stage that failed is not a stage still waiting — and the sentence a person
        reads next is the refusal, which must not have a stale caption over it."""
        def refuse(_settings):
            raise OSError("SSL: CERTIFICATE_VERIFY_FAILED")

        bar = _FakeBar()
        with self.assertRaises(SystemExit):
            _build_clip(bar, build=refuse)
        self.assertEqual(bar.phases, [cli._DOWNLOAD_PHASE, None])

    def test_a_full_cache_says_nothing_and_relabels_nothing(self):
        bar = _FakeBar()
        printed = _build_clip(bar, missing=())
        self.assertEqual(bar.phases, [])
        self.assertEqual(printed, [])

    def test_a_stage_with_no_bar_at_all_still_downloads(self):
        """`landmarks` run on its own, a piped run, a test with a plain lambda: the
        caption is a convenience and never a condition of the download."""
        self.assertIn(i18n.cli_text("cli.download.waiting", "en"), _build_clip(None))

    def test_the_bar_shows_the_sentence_and_not_the_raw_key(self):
        """An unlabelled phase key would put the word `download` on the bar — which is
        the sentence's job, and the reason the label is registered with every stage that
        can pay for a download."""
        seen: list[dict] = []
        labels = cli._download_phase_labels("en")
        task = progress.TaskProgress("classify", lambda **fields: seen.append(fields),
                                     labels)
        task.phase(cli._DOWNLOAD_PHASE)
        self.assertIn(i18n.cli_text("cli.download.waiting", "en"),
                      seen[-1]["description"])

    def test_the_pipeline_registers_the_label_with_every_stage_that_can_download(self):
        source = (_ROOT / "sorta" / "cli.py").read_text(encoding="utf-8")
        # Four bars can stand over a download: the pipeline of `sorta run` and the three
        # commands that build a CLIP model on their own (landmarks, junk, classify).
        self.assertEqual(source.count("**_download_phase_labels(lang)")
                         + source.count("phase_labels=_download_phase_labels(lang)"), 4)

    def test_the_sentence_exists_in_three_languages(self):
        for lang in _LANGS:
            with self.subTest(lang=lang):
                self.assertTrue(i18n.cli_text("cli.download.waiting", lang).strip())


class TestACaptionCanBeTakenBackDown(unittest.TestCase):
    """The mechanism under it: a phase that is over leaves the bar as it found it."""

    def test_a_phase_of_none_restores_the_bare_description(self):
        seen: list[dict] = []
        task = progress.TaskProgress("classify", lambda **fields: seen.append(fields),
                                     {"download": "waiting"})
        task.phase("download")
        task.phase(None)
        self.assertEqual(seen[-1], {"description": "classify"})

    def test_a_quiet_run_is_still_a_no_op(self):
        progress.TaskProgress("classify", None).phase(None)  # must not raise


if __name__ == "__main__":
    unittest.main()
