"""F205: the three passes that ask the model report under three names.

The run of 2026-08-05 asked the model in three places and filed all three under one
phase, `junk_vlm`:

    stage=classify phase=junk_vlm  total=7 951   1.4 frames/s  (pipelined)
    stage=junk     phase=junk_vlm  total=2 997   0.42          (one at a time)
    stage=junk     phase=junk_vlm  total=1 284   0.41-0.49     (one at a time)

Three prices that differ threefold under one name. A phase name is the unit a timing is
filed under (`runlog.measurement_unit`), so one name meant the run screen could price at
most one of the three off its own seconds and charged the other two whatever that one
cost — for the animal check and the rescue, three times too little.

What is pinned here:

* the three passes write three different names, and `read_measurements` gives back three
  units (test 1);
* the estimate charges the animal check the animal check's rate, not the deep tier's,
  and the rescue its own (test 2);
* every one of the three names has a caption in all three languages, in the terminal
  catalog and in the served app's (test 3);
* a log from before the split — one `junk_vlm`, no other model phase — is read without
  an error, and the lines it says nothing about fall back to their defaults (test 4);
* the progress bar is still told which pass it is showing (test 5).

The deep tier KEEPS `junk_vlm`, which is the choice this file is built around: it was the
pass that dominated the shared bucket, so old logs price it correctly, and the two names
that are new are new for questions no old log could price separately anyway.
"""
from __future__ import annotations

import logging
import re
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from sorta import __version__, cli, i18n, junk, runlog, ui
from sorta.junk import (
    CLASSIFY_PHASE_PETS_VLM,
    CLASSIFY_PHASE_RESCUE_VLM,
    CLASSIFY_PHASE_VLM,
    CLASSIFY_STAGE,
    classify,
)
from tests.test_estimate_from_measurements import MEASURED, EstimateTestBase, log_line
from tests.test_frame_quality import FrameQualityCase
from tests.test_junk import NO_OCR
from tests.test_junk_phase_progress import _Recorder
from tests.test_junk_rescue import FakeTextEncoder, vector_for
from tests.test_pets_cascade import PetClassifier

_RUNLOG = "sorta.runlog"
# The summary line of a phase, as `runlog` writes it — parsed, because what this feature
# is about is the two fields inside it: the name and the population under that name.
_PHASE_LINE = re.compile(
    r"^stage=(?P<stage>\S+) phase=(?P<phase>\S+) elapsed=(?P<elapsed>[0-9.]+)"
    r"(?: processed=(?P<processed>\d+))?")


class ThreeAskersClassifier(PetClassifier):
    """The CLIP mock of the pet cascade, plus the vectors the rescue scores.

    One classifier because the three passes select from ONE run: the deep tier takes the
    frames the product prompts flag, the animal check the frames above the pet candidate
    threshold, and the rescue the frames whose stored vector scores above its own. Three
    populations of three different sizes, which is what makes the phase lines below tell
    each other apart by more than their names.
    """

    def __init__(self, pet_scores, rescue_scores, products=()):
        super().__init__(pet_scores, products=tuple(products))
        self.rescue_scores = rescue_scores

    def features(self, paths):
        # -0.5 for a frame the case says nothing about: comfortably below any threshold
        # a case here sets, so it is scored and is no candidate.
        return [vector_for(self.rescue_scores.get(Path(p).name, -0.5))
                for p in paths]


class ThreePassCase(FrameQualityCase):
    """One `classify()` in which all three model passes have work to do."""

    PRODUCTS = ("prod1.jpg", "prod2.jpg", "prod3.jpg")   # the deep tier: 3 questions
    PETS = ("cat1.jpg", "cat2.jpg")                      # the animal check: 2
    SHOTS = ("shot1.jpg",)                               # the rescue: 1

    def setUp(self):
        super().setUp()
        self.features(pets=True, pets_verify=True, pet_threshold=0.7,
                      pet_candidate_threshold=0.3,
                      junk_rescue=True, junk_rescue_threshold=0.02)
        self.deep_analysis_on()
        self.encoder = FakeTextEncoder()

    def run_three_passes(self, **kwargs):
        """The run itself; returns the parsed phase lines of its run log."""
        for name in self.PRODUCTS + self.PETS + self.SHOTS:
            self.add_file(name)
        clf = ThreeAskersClassifier(
            {name: 0.9 for name in self.PETS},
            {name: 0.5 for name in self.SHOTS},
            products=self.PRODUCTS)
        kwargs.setdefault("vlm_classifier", lambda _path: "personal_photo")
        kwargs.setdefault("pet_vlm", lambda _path: "real")
        kwargs.setdefault("junk_rescue_vlm", lambda _path: "screenshot")
        with self.assertLogs(_RUNLOG, level=logging.INFO) as captured:
            with runlog.stage_timer(CLASSIFY_STAGE):
                classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR,
                         junk_text_encoder=self.encoder,
                         sharpness_detector=lambda _p, _faces: junk.Sharpness(500.0),
                         **kwargs)
        return self.phase_lines(captured.records)

    @staticmethod
    def phase_lines(records):
        out = []
        for record in records:
            match = _PHASE_LINE.match(record.getMessage())
            if match is not None:
                out.append(match)
        return out

    @staticmethod
    def processed_of(lines):
        return {m["phase"]: int(m["processed"]) for m in lines}


class TestThreeNamesInTheLog(ThreePassCase):
    """Test 1: three passes, three names, three units to read back."""

    def write_measured_log(self, lines, seconds):
        """A run log holding these phase lines with the given seconds, ready to read."""
        at = datetime.now() - timedelta(hours=1)
        out = [log_line(at, "environment:"), f"  sorta: {__version__}",
               "  python: 3.10.0"]
        for match in lines:
            elapsed = seconds.get(match["phase"])
            if elapsed is None:
                continue
            out.append(log_line(
                at, f"stage={match['stage']} phase={match['phase']}"
                    f" elapsed={elapsed:.3f} processed={match['processed']}"))
        path = Path(self.tmp.name) / "sorta.log"
        path.write_text("\n".join(out) + "\n", encoding="utf-8")
        return path

    def test_each_pass_writes_its_own_phase_with_its_own_population(self):
        lines = self.run_three_passes()
        processed = self.processed_of(lines)

        self.assertEqual(processed[CLASSIFY_PHASE_VLM], len(self.PRODUCTS))
        self.assertEqual(processed[CLASSIFY_PHASE_PETS_VLM], len(self.PETS))
        self.assertEqual(processed[CLASSIFY_PHASE_RESCUE_VLM], len(self.SHOTS))
        # Three names, and none of them is a spelling of another.
        self.assertEqual(
            len({CLASSIFY_PHASE_VLM, CLASSIFY_PHASE_PETS_VLM, CLASSIFY_PHASE_RESCUE_VLM}),
            3)

    def test_read_measurements_gives_back_three_units(self):
        """The end the names exist for: a rate per pass, not one rate for three.

        The seconds are rewritten on the way into the file, and only the seconds: a mock
        model answers in microseconds, and `elapsed=0.000` is a timing `read_measurements`
        refuses (it reads as a stage that skipped its whole population). The names and the
        populations are the run's own.
        """
        lines = self.run_three_passes()
        seconds = {CLASSIFY_PHASE_VLM: 3.0, CLASSIFY_PHASE_PETS_VLM: 10.0,
                   CLASSIFY_PHASE_RESCUE_VLM: 21.0}
        log = self.write_measured_log(lines, seconds)

        found = runlog.read_measurements(log)
        units = {phase: runlog.measurement_unit(CLASSIFY_STAGE, phase)
                 for phase in seconds}
        self.assertEqual(len(set(units.values())), 3)
        for phase, unit in units.items():
            with self.subTest(phase=phase):
                self.assertIn(unit, found)
        # And they are three different prices per frame — the fact one name could not say.
        rates = [found[unit].seconds_per_unit for unit in units.values()]
        self.assertEqual(sorted(rates), [1.0, 5.0, 21.0])


class TestTheBarIsStillTold(ThreePassCase):
    """Test 5: the caption follows the pass, as it did when there was one name."""

    def test_the_progress_channel_hears_all_three_names(self):
        rec = _Recorder()
        self.run_three_passes(progress=rec)

        for phase in (CLASSIFY_PHASE_VLM, CLASSIFY_PHASE_PETS_VLM,
                      CLASSIFY_PHASE_RESCUE_VLM):
            with self.subTest(phase=phase):
                self.assertIn(phase, rec.phases)
                # A real denominator under each caption, not a bar left at its last one.
                self.assertTrue(rec.totals_of(phase))

    def test_the_names_reported_are_the_names_logged(self):
        rec = _Recorder()
        lines = self.run_three_passes(progress=rec)

        self.assertEqual([m["phase"] for m in lines], rec.phases)


class TestEveryPhaseIsSpoken(unittest.TestCase):
    """Test 3: a number without a word is what F100 removed — for all three names now."""

    PHASES = (CLASSIFY_PHASE_VLM, CLASSIFY_PHASE_PETS_VLM, CLASSIFY_PHASE_RESCUE_VLM)
    LANGS = ("ru", "en", "ja")

    def test_the_terminal_labels_every_model_phase(self):
        for lang in self.LANGS:
            labels = cli._junk_phase_labels(lang)
            for phase in self.PHASES:
                with self.subTest(lang=lang, phase=phase):
                    self.assertTrue(labels[phase].strip())

    def test_the_terminal_captions_exist_in_all_three_languages(self):
        for phase in self.PHASES:
            entry = i18n._CLI_STRINGS[f"cli.phase.{phase}"]
            self.assertEqual(set(entry), set(self.LANGS), phase)
            for lang, text in entry.items():
                self.assertTrue(text.strip(), f"{phase}/{lang}")

    def test_the_served_app_captions_exist_in_all_three_languages(self):
        for phase in self.PHASES:
            entry = ui._UI_STRINGS[f"process_phase_{phase}"]
            self.assertEqual(set(entry), set(self.LANGS), phase)
            for lang, text in entry.items():
                self.assertTrue(text.strip(), f"{phase}/{lang}")

    def test_the_three_captions_are_three_different_sentences(self):
        """One name gave three passes one caption; three names that repeat one wording
        would leave the reader of the bar exactly where they were."""
        for lang in self.LANGS:
            with self.subTest(lang=lang):
                captions = {ui._t(f"process_phase_{phase}", lang)
                            for phase in self.PHASES}
                self.assertEqual(len(captions), len(self.PHASES))


class TestEachLineIsPricedByItsOwnPass(EstimateTestBase):
    """Test 2: the animal check costs what the animal check costs."""

    def stage_a_run(self):
        """Two frames in every band the estimate prices off a model phase."""
        for i in range(2):
            file_id = self.add_photo(f"p{i}.jpg")
            self.conn.execute(
                "INSERT INTO media_class (file_id, verdict, source, tier, updated_at)"
                " VALUES (?, 'photo', 'vlm', 'vlm', '2026-01-01')", (file_id,))
            self.conn.execute(
                "INSERT INTO frame_quality"
                " (file_id, pet_score, junk_score, source, updated_at)"
                " VALUES (?, 0.9, 0.9, 'vlm', '2026-01-01')", (file_id,))
        self.conn.commit()

    def test_the_animal_line_is_charged_the_animal_rate(self):
        self.stage_a_run()
        self.write_run_log()
        self.start_server()
        data = self.estimate()

        self.assertEqual(data["counts"]["pets_verify"], 2)
        self.assertAlmostEqual(
            data["seconds"]["pets_verify"],
            round(2 * MEASURED["stage=junk phase=junk_pets_vlm"], 1))
        # Not the deep tier's rate, which is what it was charged before F205.
        self.assertNotAlmostEqual(
            data["seconds"]["pets_verify"],
            round(2 * MEASURED["stage=classify phase=junk_vlm"], 1))
        self.assertEqual(data["sources"]["pets_verify"], "measured")

    def test_the_rescue_line_is_charged_the_rescue_rate(self):
        self.stage_a_run()
        self.write_run_log()
        self.start_server()
        data = self.estimate()

        self.assertEqual(data["counts"]["junk_rescue"], 2)
        self.assertAlmostEqual(
            data["seconds"]["junk_rescue"],
            round(2 * MEASURED["stage=junk phase=junk_rescue_vlm"], 1))
        self.assertEqual(data["sources"]["junk_rescue"], "measured")

    def test_the_deep_tier_keeps_its_own(self):
        self.stage_a_run()
        self.write_run_log()
        self.start_server()
        data = self.estimate()

        self.assertAlmostEqual(
            data["seconds"]["products"],
            round(2 * MEASURED["stage=classify phase=junk_vlm"], 1))

    def test_the_three_lines_do_not_quote_one_price(self):
        """The acceptance criterion, in one assertion: three passes, three prices."""
        self.stage_a_run()
        self.write_run_log()
        self.start_server()
        seconds = self.estimate()["seconds"]

        self.assertEqual(
            len({seconds["products"], seconds["pets_verify"], seconds["junk_rescue"]}),
            3)


class TestALogFromBeforeTheSplit(EstimateTestBase):
    """Test 4: the old spelling is read without an error, and says nothing it cannot."""

    OLD = {"stage=faces": 2.00, "stage=junk phase=junk_vlm": 5.00,
           "stage=classify phase=junk_vlm": 3.00}

    def test_the_old_names_are_read_and_the_missing_ones_fall_back(self):
        for i in range(2):
            file_id = self.add_photo(f"p{i}.jpg")
            self.conn.execute(
                "INSERT INTO frame_quality"
                " (file_id, pet_score, junk_score, source, updated_at)"
                " VALUES (?, 0.9, 0.9, 'vlm', '2026-01-01')", (file_id,))
        self.conn.commit()
        self.write_run_log(rates=self.OLD)
        self.start_server()
        data = self.estimate()

        # What the old log DOES price: the deep tier, whose name did not move.
        self.assertEqual(data["sources"]["products"], "measured")
        self.assertEqual(data["sources"]["faces"], "measured")
        # And what it does not: the shipped default, said to be one.
        for key in ("pets_verify", "junk_rescue"):
            with self.subTest(key=key):
                self.assertEqual(data["sources"][key], "default")
                self.assertAlmostEqual(data["seconds"][key],
                                       round(2 * ui._SEC_PER_VLM_FRAME, 1))

    def test_a_log_with_no_model_phase_at_all_is_not_an_error(self):
        self.add_photo("a.jpg")
        self.write_run_log(rates={"stage=faces": 2.00})
        self.start_server()
        data = self.estimate()

        for key in ("products", "pets_verify", "junk_rescue"):
            with self.subTest(key=key):
                self.assertEqual(data["sources"][key], "default")


class TestTheUnitsAreReadPhaseByPhase(unittest.TestCase):
    """The rate table names a phase per pass — the arrangement the two tests above rest
    on, asserted where it is written so a future edit cannot quietly re-share a name."""

    def test_every_model_rate_reads_a_distinct_unit(self):
        units = [ui._RATE_UNITS[name][0]
                 for name in ("vlm_verdict", "vlm_pets", "vlm_rescue")]
        self.assertEqual(len(set(units)), 3)
        for unit in units:
            self.assertIn("phase=", unit)

    def test_the_phase_names_in_the_table_are_the_ones_the_stage_reports(self):
        for name, phase in (("vlm_verdict", CLASSIFY_PHASE_VLM),
                            ("vlm_pets", CLASSIFY_PHASE_PETS_VLM),
                            ("vlm_rescue", CLASSIFY_PHASE_RESCUE_VLM)):
            with self.subTest(name=name):
                self.assertTrue(ui._RATE_UNITS[name][0].endswith(f"phase={phase}"))


class TestTheKeysStayIdentifiers(unittest.TestCase):
    """The phase name travels into the log, the caption lookup and every grep over both."""

    def test_the_new_names_look_like_the_old_ones(self):
        for phase in (CLASSIFY_PHASE_PETS_VLM, CLASSIFY_PHASE_RESCUE_VLM):
            with self.subTest(phase=phase):
                self.assertTrue(phase.startswith("junk_"))
                self.assertTrue(phase.isascii() and phase.replace("_", "").isalnum())


if __name__ == "__main__":
    unittest.main()
