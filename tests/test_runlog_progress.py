"""F166: the run log says what is happening NOW, not what happened.

F147 gave the junk stage a breakdown by phase, and the live run of 2026-08-03 showed
what was wrong with it: all four phase lines carried the same timestamp, printed in one
batch just before the stage summary. An instrument that answers "where did the time go"
at the moment the time has already gone answers nothing for a two-hour stage — and if
the run is cut short it answers nothing at all, which is exactly what happened when the
orchestrator interrupted `junk` and lost the numbers of three phases that had finished
long before.

So every timed unit of a run — the stage and each phase inside it — now writes the same
three kinds of line: it announces itself, repeats its counters once an interval, and is
summarised the moment it is over. The tests below are the requirements of the brief in
the order it lists them, the first one being the one that cost real data.
"""
from __future__ import annotations

import logging
import re
import time
import unittest

from sorta import runlog
from sorta.junk import (
    CLASSIFY_PHASE_CLIP,
    CLASSIFY_PHASE_OCR,
    CLASSIFY_PHASE_VLM,
    CLASSIFY_PHASE_WRITE,
    CLASSIFY_STAGE,
    classify,
)
from tests.test_junk import NO_OCR
from tests.test_junk_phase_progress import JunkPhaseTestBase

_RUNLOG = "sorta.runlog"

# The three shapes a unit writes, stage and phase alike. `phase=` is optional in every
# one of them — that is the uniformity the brief asks for, expressed as a regex.
_UNIT = r"stage=(?P<stage>\S+)(?: phase=(?P<phase>\S+))?"
_STARTED = re.compile(rf"^{_UNIT} started(?: total=(?P<total>\d+))?$")
_PROGRESS = re.compile(
    rf"^{_UNIT} progress elapsed=(?P<elapsed>[0-9.]+) processed=(?P<processed>\d+)"
    r"(?: total=(?P<total>\d+))?(?: rate=(?P<rate>[0-9.]+)/s)?$")
_SUMMARY = re.compile(rf"^{_UNIT} elapsed=(?P<elapsed>[0-9.]+)")
_INTERRUPTED = re.compile(
    rf"^{_UNIT} interrupted \((?P<reason>\w+)\) elapsed=(?P<elapsed>[0-9.]+)")


class RunLogTestBase(unittest.TestCase):
    """Captures the run log and pins the interval — 60 s would make every test wait."""

    interval = 0.0  # no periodic lines unless the test is about them

    def setUp(self):
        self.addCleanup(runlog.set_progress_interval,
                        runlog.DEFAULT_PROGRESS_INTERVAL_SEC)
        runlog.set_progress_interval(self.interval)
        # A stage left registered by another test would be closed into this one's log.
        self.addCleanup(runlog._PHASES.clear)
        runlog._PHASES.clear()

    def capture(self):
        return self.assertLogs(_RUNLOG, level=logging.INFO)

    @staticmethod
    def messages(captured):
        return [record.getMessage() for record in captured.records]

    @staticmethod
    def matches(messages, pattern):
        return [m for m in (pattern.match(line) for line in messages) if m is not None]

    def index_of(self, messages, pattern, **fields):
        """Where in the log the first line matching `pattern` with `fields` sits."""
        for i, line in enumerate(messages):
            match = pattern.match(line)
            if match is not None and all(
                    match.groupdict().get(k) == v for k, v in fields.items()):
                return i
        self.fail(f"no {pattern.pattern!r} with {fields} in:\n" + "\n".join(messages))


class TestAnInterruptedStageKeepsWhatFinished(RunLogTestBase):
    """Requirement 5, and the reason the feature exists: an abort must not erase.

    Before this, `log_phase` was called for every phase at once at the end of the
    stage — so a cancel at minute forty took the finished phases down with the
    unfinished one, and the estimate of F159 got nothing from a run whose numbers
    were sitting in memory the whole time.
    """

    def test_the_phases_that_finished_are_in_the_log_and_the_open_one_is_marked(self):
        with self.capture() as captured:
            with self.assertRaises(KeyboardInterrupt):
                with runlog.stage_timer(CLASSIFY_STAGE):
                    phases = runlog.track_phases(CLASSIFY_STAGE)
                    phases.start(CLASSIFY_PHASE_CLIP, 24196)
                    phases.count(CLASSIFY_PHASE_CLIP, 24196)
                    phases.enter(CLASSIFY_PHASE_OCR)
                    phases.count(CLASSIFY_PHASE_OCR, 6793)
                    phases.start(CLASSIFY_PHASE_VLM, 7214)
                    phases.count(CLASSIFY_PHASE_VLM, 1876)
                    raise KeyboardInterrupt

        messages = self.messages(captured)
        summarised = {m["phase"] for m in self.matches(messages, _SUMMARY)}
        self.assertIn(CLASSIFY_PHASE_CLIP, summarised)
        self.assertIn(CLASSIFY_PHASE_OCR, summarised)
        # The one that did not finish says so instead of claiming a clean run — and
        # still carries its seconds, which is what the estimate reads.
        stopped = self.matches(messages, _INTERRUPTED)
        self.assertEqual([m["phase"] for m in stopped], [CLASSIFY_PHASE_VLM, None])
        self.assertGreaterEqual(float(stopped[0]["elapsed"]), 0.0)

    def test_a_failed_stage_keeps_them_too(self):
        with self.capture() as captured:
            with self.assertRaises(ValueError):
                with runlog.stage_timer(CLASSIFY_STAGE):
                    phases = runlog.track_phases(CLASSIFY_STAGE)
                    phases.start(CLASSIFY_PHASE_CLIP, 4)
                    phases.count(CLASSIFY_PHASE_CLIP, 4)
                    phases.start(CLASSIFY_PHASE_VLM, 2)
                    raise ValueError("the model died")

        messages = self.messages(captured)
        self.assertIn(CLASSIFY_PHASE_CLIP,
                      {m["phase"] for m in self.matches(messages, _SUMMARY)})
        self.assertTrue(any(
            line.startswith(f"stage={CLASSIFY_STAGE} phase={CLASSIFY_PHASE_VLM} failed")
            for line in messages), messages)

    def test_the_phase_lines_come_before_the_stage_line_they_add_up_to(self):
        with self.capture() as captured:
            with self.assertRaises(KeyboardInterrupt):
                with runlog.stage_timer(CLASSIFY_STAGE):
                    phases = runlog.track_phases(CLASSIFY_STAGE)
                    phases.start(CLASSIFY_PHASE_CLIP, 4)
                    phases.count(CLASSIFY_PHASE_CLIP, 4)
                    raise KeyboardInterrupt

        messages = self.messages(captured)
        self.assertLess(
            self.index_of(messages, _INTERRUPTED, phase=CLASSIFY_PHASE_CLIP),
            self.index_of(messages, _INTERRUPTED, phase=None))


class TestAnInterruptedJunkStageKeepsWhatFinished(JunkPhaseTestBase):
    """Requirement 5 on the real stage — the one that was actually cut short."""

    def setUp(self):
        super().setUp()
        self.addCleanup(runlog.set_progress_interval,
                        runlog.DEFAULT_PROGRESS_INTERVAL_SEC)
        runlog.set_progress_interval(0)
        self.addCleanup(runlog._PHASES.clear)

    def test_cancelling_the_deep_tier_leaves_the_fast_phases_behind(self):
        names = [f"scan_{i}.jpg" for i in range(3)]
        for name in names:
            self.add_file(name)
        self.enable_vlm()

        def cancel(_path):
            raise KeyboardInterrupt  # the Cancel button, mid-VLM

        with self.assertLogs(_RUNLOG, level=logging.INFO) as captured:
            with self.assertRaises(KeyboardInterrupt):
                with runlog.stage_timer(CLASSIFY_STAGE):
                    classify(self.cfg, self.conn,
                             classifier=self.candidate_clf(names),
                             text_detector=NO_OCR, vlm_classifier=cancel)

        messages = [r.getMessage() for r in captured.records]
        summarised = {m["phase"] for m in
                      (_SUMMARY.match(line) for line in messages) if m is not None}
        # The fast tier finished when the deep one opened: all three of its phases are
        # on record, with their unit counts, although the stage never returned.
        self.assertLessEqual(
            {CLASSIFY_PHASE_CLIP, CLASSIFY_PHASE_OCR, CLASSIFY_PHASE_WRITE}, summarised)
        self.assertTrue(any(
            line.startswith(
                f"stage={CLASSIFY_STAGE} phase={CLASSIFY_PHASE_VLM} interrupted")
            for line in messages), messages)


class TestAPhaseAnnouncesItself(RunLogTestBase):
    """Requirement 1: a phase opens with a line, exactly as a stage does."""

    def test_the_start_line_comes_before_the_summary(self):
        with self.capture() as captured:
            phases = runlog.track_phases(CLASSIFY_STAGE)
            phases.start(CLASSIFY_PHASE_CLIP, 24196)
            phases.count(CLASSIFY_PHASE_CLIP, 24196)
            phases.close()

        messages = self.messages(captured)
        self.assertLess(
            self.index_of(messages, _STARTED, phase=CLASSIFY_PHASE_CLIP),
            self.index_of(messages, _SUMMARY, phase=CLASSIFY_PHASE_CLIP))

    def test_a_known_population_is_announced_with_it(self):
        with self.capture() as captured:
            phases = runlog.track_phases(CLASSIFY_STAGE)
            phases.start(CLASSIFY_PHASE_VLM, 7214)
            phases.close()

        started = self.matches(self.messages(captured), _STARTED)[0]
        self.assertEqual(started["total"], "7214")

    def test_a_phase_reached_mid_pass_is_announced_without_a_total(self):
        # Its own population is not known — the denominator on the bar belongs to the
        # pass, not to this phase's share of it, and stating it here would be a lie.
        with self.capture() as captured:
            phases = runlog.track_phases(CLASSIFY_STAGE)
            phases.start(CLASSIFY_PHASE_CLIP, 10)
            phases.enter(CLASSIFY_PHASE_OCR)
            phases.close()

        started = self.matches(self.messages(captured), _STARTED)
        self.assertEqual([(m["phase"], m["total"]) for m in started],
                         [(CLASSIFY_PHASE_CLIP, "10"), (CLASSIFY_PHASE_OCR, None)])

    def test_re_entering_a_phase_does_not_announce_it_twice(self):
        # The fast tier goes CLIP -> OCR -> write once per chunk (F73); a start line
        # per chunk would bury the log under the very thing it exists to show.
        with self.capture() as captured:
            phases = runlog.track_phases(CLASSIFY_STAGE)
            phases.start(CLASSIFY_PHASE_CLIP, 10)
            for _chunk in range(5):
                phases.enter(CLASSIFY_PHASE_CLIP)
                phases.enter(CLASSIFY_PHASE_OCR)
                phases.enter(CLASSIFY_PHASE_WRITE)
            phases.close()

        started = self.matches(self.messages(captured), _STARTED)
        self.assertEqual([m["phase"] for m in started],
                         [CLASSIFY_PHASE_CLIP, CLASSIFY_PHASE_OCR,
                          CLASSIFY_PHASE_WRITE])


class TestAPhaseIsSummarisedWhenItEnds(RunLogTestBase):
    """Requirement 2/3: the summary lands at the end of the phase, not of the stage."""

    def test_the_first_phase_is_written_out_before_the_second_starts(self):
        with self.capture() as captured:
            phases = runlog.track_phases(CLASSIFY_STAGE)
            phases.start(CLASSIFY_PHASE_CLIP, 10)
            phases.count(CLASSIFY_PHASE_CLIP, 10)
            phases.start(CLASSIFY_PHASE_VLM, 3)
            phases.count(CLASSIFY_PHASE_VLM, 3)
            phases.close()

        messages = self.messages(captured)
        self.assertLess(
            self.index_of(messages, _SUMMARY, phase=CLASSIFY_PHASE_CLIP),
            self.index_of(messages, _STARTED, phase=CLASSIFY_PHASE_VLM))

    def test_the_phases_of_one_pass_are_written_out_together_at_its_end(self):
        # They interleave over one shared counter, so none of them is finished until
        # the pass is — and then all three are, which is where their lines belong.
        with self.capture() as captured:
            phases = runlog.track_phases(CLASSIFY_STAGE)
            phases.start(CLASSIFY_PHASE_CLIP, 10)
            phases.enter(CLASSIFY_PHASE_OCR)
            phases.enter(CLASSIFY_PHASE_WRITE)
            phases.start(CLASSIFY_PHASE_VLM, 3)
            phases.close()

        messages = self.messages(captured)
        for phase in (CLASSIFY_PHASE_CLIP, CLASSIFY_PHASE_OCR, CLASSIFY_PHASE_WRITE):
            self.assertLess(self.index_of(messages, _SUMMARY, phase=phase),
                            self.index_of(messages, _STARTED, phase=CLASSIFY_PHASE_VLM))

    def test_consecutive_passes_under_one_name_still_share_one_bucket(self):
        # F147's decision, kept: the deep tier asks the same model over three candidate
        # lists, and what the reader prices is the model, not the call site.
        with self.capture() as captured:
            phases = runlog.track_phases(CLASSIFY_STAGE)
            phases.start(CLASSIFY_PHASE_VLM, 3)
            phases.count(CLASSIFY_PHASE_VLM, 3)
            phases.start(CLASSIFY_PHASE_VLM, 2)
            phases.count(CLASSIFY_PHASE_VLM, 2)
            phases.close()

        summaries = self.matches(self.messages(captured), _SUMMARY)
        self.assertEqual([m["phase"] for m in summaries], [CLASSIFY_PHASE_VLM])
        self.assertIn("processed=5", summaries[0].string)

    def test_closing_twice_does_not_report_the_same_seconds_again(self):
        # The stage has several exits and `stage_timer` closes it once more on the way
        # out; a doubled line would double the price of the phase in every estimate.
        with self.capture() as captured:
            phases = runlog.track_phases(CLASSIFY_STAGE)
            phases.start(CLASSIFY_PHASE_CLIP, 4)
            phases.count(CLASSIFY_PHASE_CLIP, 4)
            phases.close()
            phases.close()

        summaries = self.matches(self.messages(captured), _SUMMARY)
        self.assertEqual([m["phase"] for m in summaries], [CLASSIFY_PHASE_CLIP])


class TestProgressKeepsToItsInterval(RunLogTestBase):
    """Requirement 3/7: a heartbeat, not a channel of its own."""

    interval = 0.05

    def test_a_burst_of_reports_produces_at_most_one_line_per_interval(self):
        with self.capture() as captured:
            phases = runlog.track_phases(CLASSIFY_STAGE)
            phases.start(CLASSIFY_PHASE_VLM, 1000)
            for i in range(200):  # a fast loop: far more reports than intervals
                phases.step(i)
            time.sleep(self.interval * 1.5)
            for i in range(200, 400):
                phases.step(i)
            phases.close()

        progress = self.matches(self.messages(captured), _PROGRESS)
        self.assertEqual(len(progress), 1, self.messages(captured))
        self.assertEqual(progress[0]["phase"], CLASSIFY_PHASE_VLM)
        self.assertEqual(progress[0]["total"], "1000")

    def test_zero_switches_the_periodic_line_off_entirely(self):
        runlog.set_progress_interval(0)
        with self.capture() as captured:
            phases = runlog.track_phases(CLASSIFY_STAGE)
            phases.start(CLASSIFY_PHASE_VLM, 10)
            for i in range(10):
                time.sleep(0.01)
                phases.step(i)
            phases.close()

        self.assertEqual(self.matches(self.messages(captured), _PROGRESS), [])

    def test_the_periodic_line_carries_the_position_of_the_pass(self):
        with self.capture() as captured:
            phases = runlog.track_phases(CLASSIFY_STAGE)
            phases.start(CLASSIFY_PHASE_VLM, 7214)
            time.sleep(self.interval * 1.5)
            phases.step(1876)
            phases.close()

        progress = self.matches(self.messages(captured), _PROGRESS)[0]
        # The pair `/api/process/status` serves, in the file, under the phase name.
        self.assertEqual((progress["processed"], progress["total"]), ("1876", "7214"))
        self.assertIsNotNone(progress["rate"])

    def test_a_configured_interval_reaches_the_run_log(self):
        runlog.set_progress_interval(12)
        self.assertEqual(runlog.progress_interval(), 12.0)
        runlog.set_progress_interval("not a number")  # garbage is ignored, not fatal
        self.assertEqual(runlog.progress_interval(), 12.0)
        runlog.set_progress_interval(-5)  # a negative interval is "off", not "always"
        self.assertEqual(runlog.progress_interval(), 0.0)


class TestAStageWithoutPhasesReportsTheSameWay(RunLogTestBase):
    """Requirement 6: index/geo/faces/... are no less readable than junk."""

    interval = 0.05

    def test_the_stage_line_has_the_shape_the_phase_line_has(self):
        with self.capture() as captured:
            with runlog.stage_timer("index", total=1000) as stage:
                time.sleep(self.interval * 1.5)
                stage.progress(400, 1000)

        messages = self.messages(captured)
        started = self.matches(messages, _STARTED)[0]
        progress = self.matches(messages, _PROGRESS)[0]
        self.assertEqual((started["stage"], started["phase"], started["total"]),
                         ("index", None, "1000"))
        self.assertEqual(
            (progress["stage"], progress["phase"], progress["processed"],
             progress["total"]), ("index", None, "400", "1000"))
        # And the count it reported last is what the summary prices the stage by.
        self.assertIn("processed=400", messages[-1])

    def test_the_callback_the_bar_reads_is_the_one_the_log_reads(self):
        seen: list[tuple[int, int | None]] = []
        with self.capture():
            with runlog.stage_timer("geo") as stage:
                callback = runlog.observe(stage, lambda done, total=None:
                                          seen.append((done, total)))
                callback(7, 40)

        self.assertEqual(seen, [(7, 40)])  # the bar still gets every call, unchanged
        self.assertEqual((stage.processed, stage.total), (7, 40))

    def test_the_stage_stays_quiet_while_one_of_its_phases_is_open(self):
        # Two heartbeats for one stage is noise: the phase line says everything the
        # stage line would, plus which phase it is.
        with self.capture() as captured:
            with runlog.stage_timer(CLASSIFY_STAGE) as stage:
                phases = runlog.track_phases(CLASSIFY_STAGE)
                phases.start(CLASSIFY_PHASE_VLM, 100)
                time.sleep(self.interval * 1.5)
                stage.progress(50, 100)

        progress = self.matches(self.messages(captured), _PROGRESS)
        self.assertEqual([m["phase"] for m in progress], [])

    def test_a_phase_channel_on_the_callback_is_forwarded_untouched(self):
        phases: list[str] = []

        class _Bar:
            def __call__(self, done, total=None):
                pass

            def phase(self, name):
                phases.append(name)

        with self.capture():
            with runlog.stage_timer("faces") as stage:
                runlog.observe(stage, _Bar()).phase("cluster_hdbscan")
        self.assertEqual(phases, ["cluster_hdbscan"])


class TestOneNameForTheCaptionAndTheStopwatch(JunkPhaseTestBase):
    """Requirement 4: the log and `/api/process/status` cannot drift apart."""

    def setUp(self):
        super().setUp()
        self.addCleanup(runlog.set_progress_interval,
                        runlog.DEFAULT_PROGRESS_INTERVAL_SEC)
        runlog.set_progress_interval(0)
        self.addCleanup(runlog._PHASES.clear)

    def test_the_phase_in_the_log_is_the_phase_the_status_api_serves(self):
        from sorta import ui

        state = ui._ProcessState()
        state.try_start(str(self.tmp.name))
        state.set_stage(1, CLASSIFY_STAGE)
        served: list[str | None] = []

        class _AsTheUiDoes:
            """`ui._StageProgress`, sampled the way `/api/process/status` samples it."""

            def __init__(self, inner):
                self._inner = inner

            def __call__(self, done, total=None):
                self._inner(done, total)

            def phase(self, name):
                self._inner.phase(name)
                served.append(state.snapshot()["phase"])

        names = ["scan.jpg"]
        self.add_file("scan.jpg")
        with self.assertLogs(_RUNLOG, level=logging.INFO) as captured:
            classify(self.cfg, self.conn, classifier=self.candidate_clf(names),
                     text_detector=NO_OCR,
                     progress=_AsTheUiDoes(ui._StageProgress(state)))

        logged = [m["phase"] for m in
                  (_STARTED.match(r.getMessage()) for r in captured.records)
                  if m is not None and m["phase"] is not None]
        self.assertEqual(logged, served)
        self.assertEqual(logged, [CLASSIFY_PHASE_CLIP, CLASSIFY_PHASE_OCR,
                                  CLASSIFY_PHASE_WRITE])


class TestTheOldShapeStillParses(RunLogTestBase):
    """Requirement 7: `stage=` and `elapsed=` did not move — F147's greps still work."""

    def test_the_phase_summary_is_byte_for_byte_the_line_f147_wrote(self):
        with self.capture() as captured:
            phases = runlog.track_phases(CLASSIFY_STAGE)
            phases.start(CLASSIFY_PHASE_CLIP, 4)
            phases.count(CLASSIFY_PHASE_CLIP, 4)
            phases.close()

        summary = self.matches(self.messages(captured), _SUMMARY)[0].string
        self.assertRegex(
            summary,
            rf"^stage={CLASSIFY_STAGE} phase={CLASSIFY_PHASE_CLIP} "
            r"elapsed=[0-9]+\.[0-9]{3} processed=4( rate=[0-9.]+/s)?$")

    def test_every_token_of_every_new_line_is_a_key_value(self):
        # Comparisons between runs are built from these lines — prose would not do.
        # The one exception is the parenthesised reason of an abort, which reads the
        # same as it always has on the stage line above it.
        runlog.set_progress_interval(0.01)
        with self.capture() as captured:
            with self.assertRaises(KeyboardInterrupt):
                with runlog.stage_timer(CLASSIFY_STAGE, total=10) as stage:
                    phases = runlog.track_phases(CLASSIFY_STAGE)
                    phases.start(CLASSIFY_PHASE_VLM, 10)
                    time.sleep(0.02)
                    phases.step(3)
                    stage.processed = 3
                    raise KeyboardInterrupt

        for line in self.messages(captured):
            tokens = [t for t in line.split() if not t.startswith("(")]
            for token in tokens:
                self.assertTrue("=" in token or token in ("started", "progress",
                                                          "interrupted"), line)

    def test_the_stage_summary_did_not_change_either(self):
        with self.capture() as captured:
            with runlog.stage_timer("index") as stage:
                stage.processed = 42

        self.assertRegex(self.messages(captured)[-1],
                         r"^stage=index elapsed=[0-9]+\.[0-9]{3} processed=42"
                         r"( rate=[0-9.]+/s)?$")
