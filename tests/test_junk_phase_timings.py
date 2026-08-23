"""F147: the junk stage breaks its own seconds down by phase in the run log.

On the run of 2026-08-02 the stage took 2 070 seconds — more than half of the whole
hour — and the log held exactly one line about it: `stage=junk elapsed=2070.208`. Six
different things work inside that stage (CLIP over every frame, OCR behind the gate,
the laplacian, the quality model, the animal cascade, the stored vectors), and which of
them ate the 34 minutes was a guess. This feature is the arithmetic that replaces the
guess, and nothing else: not one verdict, threshold or call count moves.

The phase names are the ones F100 already gave the progress bar (CLASSIFY_PHASE_*),
which is why the tests below compare against the constants rather than against string
literals of their own — one name for the caption and for the stopwatch, so the two can
never come to mean different things.
"""
from __future__ import annotations

import logging
import re

from sorta import runlog
from sorta.junk import (
    CLASSIFY_PHASE_CLIP,
    CLASSIFY_PHASE_OCR,
    CLASSIFY_PHASE_VLM,
    CLASSIFY_PHASE_WRITE,
    CLASSIFY_STAGE,
    classify,
)
from tests.test_junk import NO_OCR, FakeClassifier
from tests.test_junk_phase_progress import JunkPhaseTestBase, _Recorder

_RUNLOG = "sorta.runlog"

# The line the whole feature exists to produce. Parsed instead of matched so the tests
# can price a phase (seconds AND units), which is the point of requirement 2.
_PHASE_LINE = re.compile(
    r"^stage=(?P<stage>\S+) phase=(?P<phase>\S+) elapsed=(?P<elapsed>[0-9.]+)"
    r"(?: processed=(?P<processed>\d+))?")


class PhaseTimingTestBase(JunkPhaseTestBase):
    """Runs `classify` with the run log captured, and hands back the parsed lines."""

    def run_classify(self, **kwargs):
        """classify() under `assertLogs`; returns (stats, [parsed phase lines])."""
        with self.assertLogs(_RUNLOG, level=logging.INFO) as captured:
            stats = classify(self.cfg, self.conn, **kwargs)
        self.records = captured.records
        return stats, self.phase_lines(captured.records)

    @staticmethod
    def phase_lines(records):
        out = []
        for record in records:
            match = _PHASE_LINE.match(record.getMessage())
            if match is not None:
                out.append((record, match))
        return out

    @staticmethod
    def phases_of(lines):
        return [m["phase"] for _record, m in lines]

    @staticmethod
    def processed_of(lines):
        return {m["phase"]: int(m["processed"]) for _record, m in lines}

    def deep_run(self, names, **kwargs):
        """A run that exercises all four phases: CLIP -> OCR -> write -> VLM."""
        self.enable_vlm()
        kwargs.setdefault("classifier", self.candidate_clf(names))
        kwargs.setdefault("text_detector", NO_OCR)
        kwargs.setdefault("vlm_classifier", lambda path: "document")
        return self.run_classify(**kwargs)


class TestEveryPhaseThatRanIsTimed(PhaseTimingTestBase):
    """Test 1: a phase that ran left a line with its seconds AND its unit count."""

    def test_all_four_phases_report_time_and_units(self):
        names = [f"scan_{i}.jpg" for i in range(4)]
        for name in names:
            self.add_file(name)
        stats, lines = self.deep_run(names)

        self.assertEqual(
            self.phases_of(lines),
            [CLASSIFY_PHASE_CLIP, CLASSIFY_PHASE_OCR, CLASSIFY_PHASE_WRITE,
             CLASSIFY_PHASE_VLM])
        for _record, match in lines:
            self.assertEqual(match["stage"], CLASSIFY_STAGE)
            self.assertIsNotNone(match["processed"], match.string)
            self.assertGreaterEqual(float(match["elapsed"]), 0.0)
        # Requirement 2: seconds alone cannot tell 4 model calls from 4 encoded frames.
        self.assertEqual(
            self.processed_of(lines),
            {CLASSIFY_PHASE_CLIP: 4, CLASSIFY_PHASE_OCR: 4,
             CLASSIFY_PHASE_WRITE: 4, CLASSIFY_PHASE_VLM: 4})
        self.assertEqual(stats.vlm_candidates, 4)

    def test_the_deep_phase_counts_candidates_and_not_frames(self):
        # The one number the breakdown exists for: 18 minutes over 1 362 calls and 18
        # minutes over 22 096 frames are different news.
        names = [f"scan_{i}.jpg" for i in range(3)]
        for name in names:
            self.add_file(name)
        for i in range(5):
            self.add_file(f"beach_{i}.jpg")  # clean photos: no gate, no candidate
        _stats, lines = self.deep_run(names)

        processed = self.processed_of(lines)
        self.assertEqual(processed[CLASSIFY_PHASE_WRITE], 8)
        self.assertEqual(processed[CLASSIFY_PHASE_CLIP], 8)
        self.assertEqual(processed[CLASSIFY_PHASE_OCR], 3)
        self.assertEqual(processed[CLASSIFY_PHASE_VLM], 3)

    def test_units_add_up_over_chunks(self):
        # The fast phases interleave per chunk (F73) — each phase keeps ONE bucket
        # across all of them instead of one line per chunk.
        self.set_batch_size(2)
        names = [f"scan_{i}.jpg" for i in range(6)]
        for name in names:
            self.add_file(name)
        _stats, lines = self.run_classify(
            classifier=self.candidate_clf(names), text_detector=NO_OCR)

        self.assertEqual(len(lines), 3)  # clip, ocr, write — one line each, not per chunk
        self.assertEqual(self.processed_of(lines),
                         {CLASSIFY_PHASE_CLIP: 6, CLASSIFY_PHASE_OCR: 6,
                          CLASSIFY_PHASE_WRITE: 6})

    def test_the_heuristics_only_run_times_the_one_phase_it_has(self):
        self.add_file("Screenshot_1.png")
        self.add_file("IMG_1.jpg")
        _stats, lines = self.run_classify(use_clip=False)

        self.assertEqual(self.phases_of(lines), [CLASSIFY_PHASE_WRITE])
        self.assertEqual(self.processed_of(lines), {CLASSIFY_PHASE_WRITE: 2})

    def test_timings_do_not_depend_on_a_progress_callback(self):
        # The stopwatch hangs on the phase machinery, not on the bar: the CLI's quiet
        # mode and the web app's own reporter must all produce the same breakdown.
        for i in range(3):
            self.add_file(f"scan_{i}.jpg")
        names = [f"scan_{i}.jpg" for i in range(3)]
        _stats, without = self.run_classify(
            classifier=self.candidate_clf(names), text_detector=NO_OCR)
        self.conn.execute("DELETE FROM media_class")
        self.conn.commit()
        _stats, with_bar = self.run_classify(
            classifier=self.candidate_clf(names), text_detector=NO_OCR,
            progress=_Recorder())

        self.assertEqual(self.processed_of(without), self.processed_of(with_bar))
        self.assertEqual(self.phases_of(without), self.phases_of(with_bar))


class TestAPhaseThatDidNotRunIsSilent(PhaseTimingTestBase):
    """Test 2: absence means "it did not happen"; `elapsed=0` would mean the opposite."""

    def test_no_ocr_line_when_the_gate_never_opened(self):
        self.add_file("beach.jpg")
        _stats, lines = self.run_classify(
            classifier=FakeClassifier({}), text_detector=NO_OCR)

        self.assertEqual(self.phases_of(lines),
                         [CLASSIFY_PHASE_CLIP, CLASSIFY_PHASE_WRITE])
        self.assertNotIn(CLASSIFY_PHASE_OCR, self.processed_of(lines))

    def test_no_deep_line_when_the_gate_selects_nobody(self):
        self.add_file("beach.jpg")
        self.enable_vlm()

        def vlm(_path):
            raise AssertionError("the VLM must not be called for a clean photo")

        stats, lines = self.run_classify(
            classifier=FakeClassifier({}), text_detector=NO_OCR, vlm_classifier=vlm)

        self.assertEqual(stats.vlm_candidates, 0)
        self.assertNotIn(CLASSIFY_PHASE_VLM, self.phases_of(lines))

    def test_no_deep_line_when_the_model_could_not_be_built(self):
        # Graceful fallback (F37-B): the deep pass did not happen, so it is not timed.
        self.add_file("scan.jpg")
        self.enable_vlm()

        def broken_factory(_model_name):
            raise RuntimeError("no CUDA / transformers not installed")

        _stats, lines = self.run_classify(
            classifier=self.candidate_clf(["scan.jpg"]), text_detector=NO_OCR,
            vlm_classifier_factory=broken_factory)

        self.assertNotIn(CLASSIFY_PHASE_VLM, self.phases_of(lines))

    def test_a_run_with_nothing_to_do_writes_no_phase_lines_at_all(self):
        # Incrementality: the second run walks no frame, so the breakdown of a stage
        # that did nothing must be empty rather than four zeroes.
        self.add_file("scan.jpg")
        clf = self.candidate_clf(["scan.jpg"])
        classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR)
        with self.assertLogs(_RUNLOG, level=logging.INFO) as captured:
            logging.getLogger(_RUNLOG).info("marker")  # assertLogs needs one record
            stats = classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR)

        self.assertEqual(stats.processed, 0)
        self.assertEqual(self.phase_lines(captured.records), [])


class TestPhaseSecondsFitTheStage(PhaseTimingTestBase):
    """Test 3: the parts must add up to something close to the whole."""

    def test_sum_of_phases_is_within_the_stage_and_is_most_of_it(self):
        names = [f"scan_{i}.jpg" for i in range(20)]
        for name in names:
            self.add_file(name)
        self.enable_vlm()
        with self.assertLogs(_RUNLOG, level=logging.INFO) as captured:
            with runlog.stage_timer(CLASSIFY_STAGE):
                classify(self.cfg, self.conn, classifier=self.candidate_clf(names),
                         text_detector=NO_OCR,
                         vlm_classifier=lambda path: "document")

        messages = [r.getMessage() for r in captured.records]
        summary = [m for m in messages
                   if m.startswith(f"stage={CLASSIFY_STAGE} elapsed=")]
        self.assertEqual(len(summary), 1)
        stage_elapsed = float(summary[0].split("elapsed=")[1].split()[0])
        phases = sum(float(m["elapsed"])
                     for _r, m in self.phase_lines(captured.records))

        # Both sides are `elapsed=` values rounded to milliseconds by `runlog`, and the
        # left one is a SUM of them: three phases that exactly fill their stage add up to
        # 0.037000000000000005 against the stage's 0.037 and the assertion fails on the
        # last bit of a float. Caught 2026-08-23. The tolerance is a millisecond — the
        # resolution the two numbers are printed at, so nothing real hides under it.
        self.assertLessEqual(phases, stage_elapsed + 1e-3)
        # And not an order of magnitude short of it: the phases are meant to account
        # for the stage, not to sample it. What is left over is the setup before the
        # first phase opens (the collection query, the tier gates).
        self.assertGreater(phases, stage_elapsed / 10)


class TestPhaseNamesAreTheProgressNames(PhaseTimingTestBase):
    """Test 4: one name for the caption and for the stopwatch — compared to constants."""

    def test_logged_names_are_exactly_the_names_reported_to_the_bar(self):
        names = [f"scan_{i}.jpg" for i in range(2)]
        for name in names:
            self.add_file(name)
        rec = _Recorder()
        _stats, lines = self.deep_run(names, progress=rec)

        self.assertEqual(self.phases_of(lines), rec.phases)

    def test_names_come_from_the_progress_constants(self):
        names = ["scan.jpg"]
        self.add_file("scan.jpg")
        _stats, lines = self.deep_run(names)

        known = {CLASSIFY_PHASE_CLIP, CLASSIFY_PHASE_OCR, CLASSIFY_PHASE_VLM,
                 CLASSIFY_PHASE_WRITE}
        self.assertTrue(set(self.phases_of(lines)) <= known, self.phases_of(lines))

    def test_the_stage_name_is_the_one_the_pipeline_uses(self):
        from sorta.cli import _pipeline_steps

        self.assertIn(CLASSIFY_STAGE, [name for name, _fn in _pipeline_steps()])


class TestLinesReachTheOrdinaryLogLevel(PhaseTimingTestBase):
    """Test 5: the same level as the stage lines — not a DEBUG-only breakdown."""

    def test_phase_records_are_info_like_the_stage_summary(self):
        self.add_file("scan.jpg")
        with self.assertLogs(_RUNLOG, level=logging.INFO) as captured:
            with runlog.stage_timer(CLASSIFY_STAGE):
                classify(self.cfg, self.conn,
                         classifier=self.candidate_clf(["scan.jpg"]),
                         text_detector=NO_OCR)

        lines = self.phase_lines(captured.records)
        self.assertTrue(lines)
        summary = next(r for r in captured.records
                       if r.getMessage().startswith(f"stage={CLASSIFY_STAGE} elapsed="))
        for record, _match in lines:
            self.assertEqual(record.levelno, summary.levelno)
            self.assertEqual(record.levelno, logging.INFO)
            self.assertEqual(record.name, summary.name)

    def test_lines_are_machine_readable_key_values(self):
        # Comparisons between runs are built from these lines — prose would not do.
        self.add_file("scan.jpg")
        _stats, lines = self.run_classify(
            classifier=self.candidate_clf(["scan.jpg"]), text_detector=NO_OCR)

        for record, _match in lines:
            for token in record.getMessage().split():
                self.assertIn("=", token, record.getMessage())


class TestNothingElseMoved(PhaseTimingTestBase):
    """The instrument must not change what it measures."""

    def test_verdicts_are_identical_to_a_run_of_the_previous_shape(self):
        names = ["scan.jpg", "beach.jpg"]
        for name in names:
            self.add_file(name)
        self.enable_vlm()
        clf = self.candidate_clf(["scan.jpg"])
        classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR,
                 vlm_classifier=lambda path: "product")
        first = self.verdicts()
        self.conn.execute("DELETE FROM media_class")
        self.conn.commit()
        self.run_classify(classifier=clf, text_detector=NO_OCR,
                          vlm_classifier=lambda path: "product", progress=_Recorder())

        self.assertEqual(self.verdicts(), first)

    def test_the_bar_still_gets_its_phases_and_its_counter(self):
        names = [f"scan_{i}.jpg" for i in range(2)]
        for name in names:
            self.add_file(name)
        rec = _Recorder()
        self.deep_run(names, progress=rec)

        self.assertEqual(
            rec.phases,
            [CLASSIFY_PHASE_CLIP, CLASSIFY_PHASE_OCR, CLASSIFY_PHASE_WRITE,
             CLASSIFY_PHASE_VLM])
        self.assertEqual(rec.dones_of(CLASSIFY_PHASE_VLM), [0, 1, 2])
