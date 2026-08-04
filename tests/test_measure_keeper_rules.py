"""F180 phase 0: the arithmetic the keeper verdict is made of.

The script decides whether the last VLM question of the junk stage stays, so what has to
be right here is not "the code runs" but that the measurement cannot flatter itself:

* the labelling is collected blind — the worksheet shuffles the frames of a group, so
  neither the sharpness order nor the model's answer can lead the person filling it in;
* "одинаковые" is a first-class answer: those groups leave the headline percentage
  entirely, because there every rule is right and counting them would lift all three
  variants by the same amount and hide the finding;
* the baseline is the CURRENT behaviour — the model's stored answers plus the sharpness
  fallback the stage really uses when it has none;
* the arithmetic with every knob off is exactly `dedup.rank_frames`, so the table starts
  at what the interface already does rather than at zero;
* the threshold is picked by a rule fixed before the run (`best_rule`), the cascade's
  margin by another one (`best_cascade`), and the verdict by criteria fixed before the
  run (`decide`);
* nothing printed identifies a frame — a near-duplicate group is a burst of one moment.

No model, no GPU and no photograph: every number below is arithmetic over labels.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

from sorta import dedup
from sorta.db import connect

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "measure_keeper_rules.py"


def _load_script():
    """Import scripts/measure_keeper_rules.py — it is a script, not a package module."""
    spec = importlib.util.spec_from_file_location("measure_keeper_rules", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


measure = _load_script()

SAME = measure.CHOICE_SAME
FRAME = measure.FOCUS_FRAME
FACE = measure.FOCUS_FACE


def frame(file_id, sharpness=None, face=None, eye=None, pixels=12_000_000, size=3_000_000):
    return measure.Frame(file_id=file_id, sharpness=sharpness, face_sharpness=face,
                         eye_openness=eye, pixels=pixels, size=size)


def group(*frames, key=None):
    """A group ranked as `dedup.keeper_groups` ranks it — best frame first."""
    frames = tuple(frames)
    return measure.Group(key=key or dedup.group_key([f.file_id for f in frames]),
                         frames=frames)


def rule(eye_min=0.0, focus=FRAME):
    return measure.Rule(eye_min=eye_min, focus=focus)


class TestTheRuleReadsThreeNumbers(unittest.TestCase):
    """The blink filter first, the sharpness after it — the brief's own order."""

    def test_the_sharpest_frame_wins_when_every_eye_is_open(self):
        picked = measure.judge([frame(1, sharpness=90.0, eye=0.30),
                                frame(2, sharpness=50.0, eye=0.30)], rule())
        self.assertEqual(picked.file_id, 1)

    def test_a_blink_loses_to_a_blurrier_frame_with_open_eyes(self):
        picked = measure.judge([frame(1, sharpness=90.0, eye=0.04),
                                frame(2, sharpness=50.0, eye=0.30)], rule(eye_min=0.18))
        self.assertEqual(picked.file_id, 2)

    def test_the_filter_switched_off_keeps_the_sharpest_blink(self):
        picked = measure.judge([frame(1, sharpness=90.0, eye=0.04),
                                frame(2, sharpness=50.0, eye=0.30)], rule(eye_min=0.0))
        self.assertEqual(picked.file_id, 1)

    def test_a_group_where_everybody_blinked_is_still_answered(self):
        picked = measure.judge([frame(1, sharpness=50.0, eye=0.02),
                                frame(2, sharpness=90.0, eye=0.03)], rule(eye_min=0.18))
        self.assertEqual(picked.file_id, 2)

    def test_a_frame_with_no_openness_measured_is_not_a_blink(self):
        picked = measure.judge([frame(1, sharpness=90.0, eye=None),
                                frame(2, sharpness=50.0, eye=0.30)], rule(eye_min=0.18))
        self.assertEqual(picked.file_id, 1)

    def test_the_face_focus_orders_by_the_face_when_the_whole_group_has_one(self):
        frames = [frame(1, sharpness=90.0, face=10.0), frame(2, sharpness=50.0, face=80.0)]
        self.assertEqual(measure.judge(frames, rule(focus=FACE)).file_id, 2)
        self.assertEqual(measure.judge(frames, rule(focus=FRAME)).file_id, 1)

    def test_a_half_measured_group_falls_back_to_the_frame_number(self):
        # The convention of `dedup.rank_frames`: a partial comparison would quietly prefer
        # whichever frames happened to be measured.
        frames = [frame(1, sharpness=90.0, face=None), frame(2, sharpness=50.0, face=80.0)]
        self.assertEqual(measure.judge(frames, rule(focus=FACE)).file_id, 1)

    def test_a_group_nobody_measured_is_ordered_by_resolution_then_size_then_id(self):
        frames = [frame(7, pixels=1000, size=10), frame(3, pixels=1000, size=10),
                  frame(9, pixels=4000, size=10)]
        self.assertEqual(measure.judge(frames, rule()).file_id, 9)
        self.assertEqual(measure.judge(frames[:2], rule()).file_id, 3)


class TestItIsTodaysRankingWithTheKnobsOff(unittest.TestCase):
    """The table has to start at what the interface already does, not at zero."""

    def frames(self):
        return [frame(4, sharpness=10.0, pixels=100, size=5),
                frame(2, sharpness=None, pixels=900, size=5),
                frame(8, sharpness=70.0, pixels=900, size=9)]

    def test_the_plain_rule_picks_what_dedup_rank_frames_picks(self):
        for sharpness in (True, False):
            with self.subTest(sharpness=sharpness):
                mine = [f if sharpness else frame(f.file_id, sharpness=None,
                                                  pixels=f.pixels, size=f.size)
                        for f in self.frames()]
                theirs = [dedup.GroupFrame(file_id=f.file_id, path="", sharpness=f.sharpness,
                                           pixels=f.pixels, size=f.size) for f in mine]
                self.assertEqual(measure.judge(mine, rule()).file_id,
                                 dedup.rank_frames(theirs)[0].file_id)


class TestHowFarAheadTheWinnerIs(unittest.TestCase):
    """The margin is the whole of the cascade: it decides what gets asked."""

    def test_two_frames_of_the_same_number_leave_no_margin(self):
        picked = measure.judge([frame(1, sharpness=50.0), frame(2, sharpness=50.0)], rule())
        self.assertEqual(picked.margin, 0.0)

    def test_the_margin_is_the_gap_as_a_share_of_the_winner(self):
        picked = measure.judge([frame(1, sharpness=100.0), frame(2, sharpness=75.0)], rule())
        self.assertAlmostEqual(picked.margin, 0.25)

    def test_the_last_frame_standing_after_the_blinks_is_certain(self):
        picked = measure.judge([frame(1, sharpness=50.0, eye=0.30),
                                frame(2, sharpness=90.0, eye=0.02),
                                frame(3, sharpness=95.0, eye=0.03)], rule(eye_min=0.18))
        self.assertEqual((picked.file_id, picked.margin), (1, 1.0))

    def test_a_group_no_column_can_order_is_not_confident(self):
        picked = measure.judge([frame(1, sharpness=None), frame(2, sharpness=None)], rule())
        self.assertEqual(picked.margin, 0.0)

    def test_a_single_frame_group_claims_no_confidence(self):
        self.assertEqual(measure.judge([frame(1, sharpness=50.0)], rule()).margin, 0.0)


class TestTheThreeVariants(unittest.TestCase):
    """Who answers what, and which of them would put a group to the model."""

    def group(self):
        return group(frame(1, sharpness=90.0), frame(2, sharpness=88.0))

    def test_the_arithmetic_never_asks(self):
        pick = measure.arithmetic(rule())(self.group())
        self.assertEqual((pick.file_id, pick.asked), (1, False))

    def test_the_model_answers_what_the_index_stored(self):
        g = self.group()
        pick = measure.model({g.key: 2})(g)
        self.assertEqual((pick.file_id, pick.asked, pick.silent), (2, True, False))

    def test_a_group_the_model_never_answered_falls_back_to_sharpness(self):
        # What the stage really does — and it is still counted as a call, because the
        # stage still makes one.
        pick = measure.model({})(self.group())
        self.assertEqual((pick.file_id, pick.asked, pick.silent), (1, True, True))

    def test_the_cascade_keeps_a_confident_group_to_itself(self):
        g = group(frame(1, sharpness=100.0), frame(2, sharpness=10.0))
        pick = measure.cascade(rule(), 0.10, {g.key: 2})(g)
        self.assertEqual((pick.file_id, pick.asked), (1, False))

    def test_the_cascade_asks_where_the_frames_are_alike(self):
        g = group(frame(1, sharpness=100.0), frame(2, sharpness=99.0))
        pick = measure.cascade(rule(), 0.10, {g.key: 2})(g)
        self.assertEqual((pick.file_id, pick.asked), (2, True))

    def test_the_ends_of_the_margin_grid_are_the_other_two_variants(self):
        """At 0 the cascade is the arithmetic; above 1 it is the model."""
        g = group(frame(1, sharpness=100.0), frame(2, sharpness=99.0))
        self.assertEqual(measure.MARGIN_GRID[0], 0.0)
        self.assertGreater(measure.MARGIN_GRID[-1], 1.0)
        self.assertFalse(measure.cascade(rule(), measure.MARGIN_GRID[0], {})(g).asked)
        self.assertTrue(measure.cascade(rule(), measure.MARGIN_GRID[-1], {})(g).asked)


class TestTheSameAnswerLeavesThePercentage(unittest.TestCase):
    """The finding of F132, carried into the arithmetic of this one."""

    def groups(self):
        return [group(frame(1, sharpness=90.0), frame(2, sharpness=10.0), key="a"),
                group(frame(3, sharpness=90.0), frame(4, sharpness=10.0), key="b"),
                group(frame(5, sharpness=90.0), frame(6, sharpness=10.0), key="c")]

    def score(self, labels):
        return measure.score("x", self.groups(), labels, measure.arithmetic(rule()),
                             measure.Cost())

    def test_a_group_called_same_is_not_a_win_and_not_a_miss(self):
        row = self.score({"a": 1, "b": SAME, "c": 5})
        self.assertEqual((row.decided, row.agreed, row.same), (2, 2, 1))
        self.assertEqual(row.agreement, 1.0)

    def test_the_lenient_reading_counts_it_as_a_win(self):
        row = self.score({"a": 2, "b": SAME, "c": 6})
        self.assertEqual(row.agreement, 0.0)
        self.assertAlmostEqual(row.lenient, 1 / 3)

    def test_an_unanswered_group_takes_part_in_nothing(self):
        row = self.score({"a": 1})
        self.assertEqual((row.decided, row.same, row.labelled), (1, 0, 1))

    def test_a_labelling_of_nothing_but_same_claims_no_agreement(self):
        row = self.score({"a": SAME, "b": SAME, "c": SAME})
        self.assertEqual((row.agreement, row.lenient), (0.0, 1.0))
        self.assertEqual(measure.same_share(row), 1.0)


class TestThePrice(unittest.TestCase):
    """Seconds over the live population, and only for the frames really shown."""

    def prices(self):
        return measure.Prices(call_s=0.5, frame_s=1.0, max_frames=5)

    def test_one_call_is_the_fixed_part_plus_a_part_per_frame(self):
        self.assertAlmostEqual(measure.call_seconds(3, self.prices()), 3.5)

    def test_a_burst_larger_than_the_question_costs_the_question(self):
        self.assertAlmostEqual(measure.call_seconds(38, self.prices()), 5.5)

    def test_a_group_of_one_frame_is_never_a_call(self):
        self.assertEqual(measure.call_seconds(1, self.prices()), 0.0)

    def test_the_model_pays_for_every_group_and_the_arithmetic_for_none(self):
        population = [group(frame(1, sharpness=90.0), frame(2, sharpness=10.0), key="a"),
                      group(frame(3, sharpness=90.0), frame(4, sharpness=10.0), key="b")]
        model = measure.cost(population, measure.model({}), self.prices())
        rules = measure.cost(population, measure.arithmetic(rule()), self.prices())
        self.assertEqual((model.asked, model.seconds), (2, 5.0))
        self.assertEqual((rules.asked, rules.seconds), (0, 0.0))

    def test_the_cascade_pays_only_for_what_it_asks(self):
        population = [group(frame(1, sharpness=100.0), frame(2, sharpness=10.0), key="a"),
                      group(frame(3, sharpness=100.0), frame(4, sharpness=99.0), key="b")]
        spent = measure.cost(population, measure.cascade(rule(), 0.10, {}), self.prices())
        self.assertEqual((spent.asked, spent.seconds), (1, 2.5))


class TestWhatTheCascadeGivesUp(unittest.TestCase):
    """Item 5 of the brief: it did not ask, and it was wrong — did the model know better?"""

    def groups(self):
        # Both groups look confident to the arithmetic and both times it picks frame 1.
        return [group(frame(1, sharpness=100.0), frame(2, sharpness=10.0), key="a"),
                group(frame(3, sharpness=100.0), frame(4, sharpness=10.0), key="b"),
                group(frame(5, sharpness=100.0), frame(6, sharpness=99.0), key="c")]

    def loss(self, labels, choices):
        groups = self.groups()
        return measure.loss(groups, labels, measure.cascade(rule(), 0.10, choices),
                            measure.model(choices))

    def test_a_quiet_miss_the_model_would_have_caught_is_the_number_that_matters(self):
        found = self.loss({"a": 2, "b": 3}, {"a": 2, "b": 4})
        self.assertEqual((found.quiet, found.wrong, found.model_right), (2, 1, 1))

    def test_a_miss_the_model_would_have_made_too_costs_nothing_to_stop_asking(self):
        found = self.loss({"a": 2}, {"a": 1})
        self.assertEqual((found.quiet, found.wrong, found.model_right), (1, 1, 0))

    def test_a_group_it_asked_about_is_not_its_loss(self):
        self.assertEqual(self.loss({"c": 6}, {"c": 5}).quiet, 0)

    def test_the_same_groups_are_left_out_of_the_count(self):
        self.assertEqual(self.loss({"a": SAME, "b": SAME}, {}).quiet, 0)


class TestTheModelsOwnCeiling(unittest.TestCase):
    """The stage shows the model five frames; the owner is shown the whole group."""

    def group(self):
        return group(*[frame(i, sharpness=100.0 - i) for i in range(1, 8)], key="a")

    def test_a_pick_outside_the_window_was_lost_before_the_question(self):
        self.assertEqual(measure.unseen_picks([self.group()], {"a": 7}, 5), 1)

    def test_a_pick_inside_it_is_not_counted(self):
        self.assertEqual(measure.unseen_picks([self.group()], {"a": 3}, 5), 0)

    def test_same_is_not_a_pick_at_all(self):
        self.assertEqual(measure.unseen_picks([self.group()], {"a": SAME}, 5), 0)


class TestTheThresholdsArePickedByARule(unittest.TestCase):
    """A bar chosen after seeing the table is not a bar (F131)."""

    def row(self, agreement, eye_min=0.0, focus=FRAME, decided=100):
        return measure.RuleRow(
            rule=rule(eye_min=eye_min, focus=focus),
            score=measure.Score(variant="x", decided=decided,
                                agreed=round(decided * agreement)))

    def test_it_takes_the_most_agreement(self):
        rows = [self.row(0.50), self.row(0.70, eye_min=0.18), self.row(0.60, eye_min=0.20)]
        self.assertEqual(measure.best_rule(rows).rule.eye_min, 0.18)

    def test_a_tie_goes_to_the_signal_that_exists_for_more_rows(self):
        rows = [self.row(0.70, focus=FACE), self.row(0.70, focus=FRAME)]
        self.assertEqual(measure.best_rule(rows).rule.focus, FRAME)

    def test_a_tie_goes_to_the_filter_that_is_switched_off(self):
        rows = [self.row(0.70, eye_min=0.25), self.row(0.70, eye_min=0.0)]
        self.assertEqual(measure.best_rule(rows).rule.eye_min, 0.0)

    def test_the_sweep_covers_every_threshold_of_both_focuses(self):
        self.assertEqual(len(measure.rules()),
                         len(measure.EYE_GRID) * len(measure.FOCUS))
        self.assertIn(rule(eye_min=0.0, focus=FRAME), measure.rules())


class TestTheCascadeIsPickedForItsPrice(unittest.TestCase):
    """It exists to buy back seconds, so the cheapest row that holds the bar wins."""

    def row(self, margin, agreement, asked, decided=100):
        return measure.CascadeRow(
            margin=margin,
            score=measure.Score(variant="x", decided=decided,
                                agreed=round(decided * agreement),
                                cost=measure.Cost(asked=asked, seconds=asked * 5.0)),
            loss=measure.Loss())

    def rows(self):
        return [self.row(0.0, 0.60, 0), self.row(0.10, 0.72, 30),
                self.row(0.30, 0.75, 80), self.row(2.0, 0.74, 100)]

    def test_the_cheapest_row_over_the_floor_wins(self):
        self.assertEqual(measure.best_cascade(self.rows(), floor=0.70).margin, 0.10)

    def test_a_lower_floor_buys_back_more_seconds(self):
        self.assertEqual(measure.best_cascade(self.rows(), floor=0.55).margin, 0.0)

    def test_a_floor_nothing_reaches_still_names_the_best_row_there_was(self):
        self.assertEqual(measure.best_cascade(self.rows(), floor=0.99).margin, 0.30)

    def test_no_rows_at_all_is_no_cascade(self):
        self.assertIsNone(measure.best_cascade([], floor=0.5))


class TestTheVerdict(unittest.TestCase):
    """Five outcomes, and the criteria are the ones written before the run."""

    def rule_row(self, agreement, decided=100, eye_min=0.18):
        return measure.RuleRow(
            rule=rule(eye_min=eye_min),
            score=measure.Score(variant="a", decided=decided,
                                agreed=round(decided * agreement)))

    def cascade_row(self, agreement, seconds, decided=100):
        return measure.CascadeRow(
            margin=0.10,
            score=measure.Score(variant="c", decided=decided,
                                agreed=round(decided * agreement),
                                cost=measure.Cost(asked=30, seconds=seconds)),
            loss=measure.Loss())

    def base(self, agreement=0.70, decided=100, same=10, seconds=120.0):
        return measure.Score(variant="m", decided=decided,
                             agreed=round(decided * agreement), same=same,
                             cost=measure.Cost(asked=115, seconds=seconds))

    def test_arithmetic_that_is_not_worse_removes_the_question(self):
        verdict, why = measure.decide(self.rule_row(0.68), self.cascade_row(0.70, 40.0),
                                      self.base())
        self.assertEqual(verdict, measure.VERDICT_ARITHMETIC)
        self.assertIn("0.18", why)

    def test_a_cheap_cascade_that_is_not_worse_is_taken_next(self):
        verdict, _why = measure.decide(self.rule_row(0.50), self.cascade_row(0.69, 40.0),
                                       self.base())
        self.assertEqual(verdict, measure.VERDICT_CASCADE)

    def test_a_cascade_that_saves_nothing_is_not_worth_its_complexity(self):
        verdict, why = measure.decide(self.rule_row(0.50), self.cascade_row(0.69, 110.0),
                                      self.base())
        self.assertEqual(verdict, measure.VERDICT_MODEL)
        # The reason has to be the price, not a quality gap it does not have.
        self.assertIn("качество держит", why)

    def test_a_model_clearly_ahead_of_both_stays(self):
        verdict, why = measure.decide(self.rule_row(0.40), self.cascade_row(0.50, 10.0),
                                      self.base())
        self.assertEqual(verdict, measure.VERDICT_MODEL)
        self.assertIn("70%", why)
        self.assertIn("разрыв с моделью", why)

    def test_half_the_groups_indistinguishable_is_itself_the_verdict(self):
        verdict, why = measure.decide(self.rule_row(0.90), self.cascade_row(0.90, 10.0),
                                      self.base(same=100))
        self.assertEqual(verdict, measure.VERDICT_SAME)
        self.assertIn("50%", why)

    def test_too_few_decided_groups_is_not_a_verdict(self):
        thin = measure.MIN_DECIDED_GROUPS - 1
        verdict, _why = measure.decide(self.rule_row(0.90, decided=thin),
                                       self.cascade_row(0.90, 10.0, decided=thin),
                                       self.base(decided=thin, same=0))
        self.assertEqual(verdict, measure.VERDICT_UNCLEAR)

    def test_the_bar_moves_with_the_baseline_rather_than_being_a_constant(self):
        # A model that does better makes the bar harder, by construction.
        verdict, _why = measure.decide(self.rule_row(0.68), self.cascade_row(0.68, 110.0),
                                       self.base(agreement=0.90))
        self.assertEqual(verdict, measure.VERDICT_MODEL)


class TestTheWorksheetIsBlind(unittest.TestCase):
    """A sheet that showed the answer would collect a labelling of the hint."""

    def groups(self):
        return [group(*[frame(i * 10 + n, sharpness=100.0 - n) for n in range(4)],
                      key=f"g{i}")
                for i in range(20)]

    def test_it_holds_the_frames_of_every_group_and_nothing_else(self):
        written = measure.sheet(self.groups(), seed=1)
        self.assertEqual(set(written), {f"g{i}" for i in range(20)})
        for key, cell in written.items():
            with self.subTest(group=key):
                self.assertEqual(sorted(cell), ["choice", "frames"])
                self.assertIsNone(cell["choice"])

    def test_the_frames_are_shuffled_out_of_the_sharpness_order(self):
        written = measure.sheet(self.groups(), seed=1)
        ranked = {g.key: [f.file_id for f in g.frames] for g in self.groups()}
        moved = [key for key, cell in written.items() if cell["frames"] != ranked[key]]
        self.assertGreater(len(moved), len(ranked) // 2)

    def test_a_shuffle_loses_no_frame_and_invents_none(self):
        written = measure.sheet(self.groups(), seed=1)
        for g in self.groups():
            with self.subTest(group=g.key):
                self.assertEqual(sorted(written[g.key]["frames"]),
                                 sorted(f.file_id for f in g.frames))

    def test_the_same_seed_writes_the_same_sheet(self):
        self.assertEqual(measure.sheet(self.groups(), seed=7),
                         measure.sheet(self.groups(), seed=7))


class TestReadingTheFilledInSheet(unittest.TestCase):
    """What the worksheet may say, and what it may not."""

    def read(self, payload):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sheet.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            return measure.read_sheet(str(path))

    def sheet(self, choice):
        return {"a": {"frames": [4, 9], "choice": choice}}

    def test_a_chosen_frame_comes_back_as_a_number(self):
        self.assertEqual(self.read(self.sheet(9)), {"a": 9})

    def test_the_third_answer_comes_back_as_itself(self):
        self.assertEqual(self.read(self.sheet(SAME)), {"a": SAME})

    def test_an_unanswered_group_is_dropped_rather_than_counted_as_a_miss(self):
        payload = {"a": {"frames": [4, 9], "choice": None},
                   "b": {"frames": [1, 2], "choice": 1}}
        self.assertEqual(self.read(payload), {"b": 1})

    def test_a_frame_from_another_group_is_refused(self):
        with self.assertRaises(SystemExit):
            self.read(self.sheet(77))

    def test_a_choice_that_is_neither_an_id_nor_the_word_is_refused(self):
        with self.assertRaises(SystemExit):
            self.read(self.sheet("the left one"))

    def test_a_group_without_its_frames_is_refused(self):
        with self.assertRaises(SystemExit):
            self.read({"a": {"choice": 4}})

    def test_an_empty_labelling_is_refused(self):
        with self.assertRaises(SystemExit):
            self.read(self.sheet(None))

    def test_something_that_is_not_a_worksheet_is_refused(self):
        with self.assertRaises(SystemExit):
            self.read([1, 2, 3])

    def test_a_missing_file_is_refused(self):
        with self.assertRaises(SystemExit):
            measure.read_sheet(str(Path(tempfile.gettempdir()) / "no-such-sheet.json"))

    def test_a_file_that_is_not_json_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sheet.json"
            path.write_text("not json at all", encoding="utf-8")
            with self.assertRaises(SystemExit):
                measure.read_sheet(str(path))

    def test_what_is_written_is_what_is_read_back(self):
        groups = [group(frame(1, sharpness=90.0), frame(2, sharpness=10.0), key="a")]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sheet.json"
            self.assertEqual(measure.write_sheet(str(path), groups, seed=3), 1)
            filled = json.loads(path.read_text(encoding="utf-8"))
            filled["a"]["choice"] = 2
            path.write_text(json.dumps(filled), encoding="utf-8")
            self.assertEqual(measure.read_sheet(str(path)), {"a": 2})


class TestItReadsTheCollectionItRunsOn(unittest.TestCase):
    """The groups, the three numbers and the model's answers all come from the index."""

    def db(self, tmp):
        conn = connect(Path(tmp) / "x.db")
        # Two pHashes one bit apart and one far away: a near-duplicate pair and a single.
        for i, (phash, sharpness, face, eye) in enumerate(
                [("f0f0f0f0f0f0f0f0", 100.0, 10.0, 0.30),
                 ("f0f0f0f0f0f0f0f1", 50.0, 80.0, 0.05),
                 ("0f0f0f0f0f0f0f0f", 70.0, None, None)], start=1):
            conn.execute(
                "INSERT INTO files (path, size, mtime, ext, media_type, indexed_at,"
                " width, height, phash) VALUES (?, 1, 0.0, 'jpg', 'photo', 'x', 10, 10, ?)",
                (f"/photos/{i}.jpg", phash))
            conn.execute(
                "INSERT INTO frame_quality (file_id, sharpness, face_sharpness,"
                " eye_openness, source, updated_at) VALUES (?, ?, ?, ?, 'classic', 'now')",
                (i, sharpness, face, eye))
        conn.commit()
        return conn

    def test_a_group_carries_all_three_numbers(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self.db(tmp)
            groups = measure.read_groups(conn, max_distance=5)
            conn.close()
        self.assertEqual(len(groups), 1)
        first, second = groups[0].frames
        self.assertEqual((first.file_id, first.sharpness, first.face_sharpness,
                          first.eye_openness), (1, 100.0, 10.0, 0.30))
        self.assertEqual((second.face_sharpness, second.eye_openness), (80.0, 0.05))

    def test_the_group_is_ranked_and_keyed_the_way_the_stage_keys_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self.db(tmp)
            groups = measure.read_groups(conn, max_distance=5)
            conn.close()
        self.assertEqual(groups[0].key, dedup.group_key([1, 2]))
        self.assertEqual([f.file_id for f in groups[0].frames], [1, 2])

    def test_a_narrower_population_leaves_the_pair_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self.db(tmp)
            groups = measure.read_groups(conn, max_distance=5, min_size=3)
            conn.close()
        self.assertEqual(groups, [])

    def test_only_the_models_own_answers_are_the_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self.db(tmp)
            dedup.store_group_keeper(conn, "vlm-group", 2, "vlm#1a2b3c4d", "now")
            dedup.store_group_keeper(conn, "cheap-group", 1,
                                     dedup.KEEPER_SOURCE_SHARPNESS, "now")
            conn.commit()
            found = measure.model_choices(conn)
            conn.close()
        self.assertEqual(found, {"vlm-group": 2})


class TestThePopulationBlock(unittest.TestCase):
    """Item 1: measuring on exact pHash equality would be measuring on nothing."""

    def text(self):
        return measure.format_population({2: 20, 3: 4}, {2: 90, 3: 20, 5: 5}, min_size=3,
                                         prices=measure.Prices(call_s=0.5, frame_s=1.0,
                                                               max_frames=5))

    def test_both_readings_are_printed_at_the_size_the_measurement_uses(self):
        # The comparison the brief makes: exact equality leaves a handful of groups of
        # three or more, real nearness leaves several times as many, and a table built on
        # the first would be a table about nothing.
        self.assertIn("групп по точному совпадению phash: 24, из них от 3 кадров: 4",
                      self.text())
        self.assertIn("настоящей близости", self.text())
        self.assertIn("из них от 3 кадров: 25", self.text())

    def test_the_population_of_the_measurement_is_the_wide_one(self):
        self.assertIn("популяция замера — групп: 25, кадров в них: 85", self.text())

    def test_the_histogram_counts_every_group(self):
        self.assertEqual(measure.histogram([(1, 2), (1, 2, 3), (4, 5)]), {2: 2, 3: 1})

    def test_the_baseline_price_is_printed_before_anybody_has_labelled_anything(self):
        # 20 groups of three at 3.5 s and 5 of five at 5.5 s; the pairs are not asked about.
        self.assertAlmostEqual(
            measure.baseline_seconds({2: 90, 3: 20, 5: 5}, 3,
                                     measure.Prices(call_s=0.5, frame_s=1.0, max_frames=5)),
            97.5)
        self.assertIn("цена нынешнего поведения (чистая модель) на ней: 98 с",
                      self.text())


class TestTheReport(unittest.TestCase):
    """The tables the brief asks for, and nothing that names a photograph."""

    def measurement(self):
        population = []
        labels = {}
        choices = {}
        for i in range(40):
            key = f"g{i}"
            best, worst = 2 * i + 1, 2 * i + 2
            population.append(group(frame(best, sharpness=100.0, face=90.0, eye=0.30),
                                    frame(worst, sharpness=60.0, face=40.0, eye=0.04),
                                    key=key))
            choices[key] = best if i % 3 else worst
            labels[key] = SAME if i % 5 == 0 else best
        return measure.measure(population, labels, choices,
                               measure.Prices(call_s=0.5, frame_s=1.0, max_frames=5))

    def text(self):
        return measure.report(self.measurement())

    def test_the_three_variants_get_a_line_each(self):
        text = self.text()
        for variant in (measure.VARIANT_ARITHMETIC, measure.VARIANT_CASCADE,
                        measure.VARIANT_MODEL, measure.VARIANT_SHARPNESS):
            with self.subTest(variant=variant):
                self.assertIn(variant, text)

    def test_the_baseline_is_printed_under_every_table(self):
        # Two arithmetic tables, the cascade table and the summary — see the brief's
        # item 6: without the current behaviour any figure reads as an improvement.
        self.assertEqual(self.text().count("<- базовая линия"), 4)

    def test_the_share_of_same_is_reported_before_the_tables(self):
        text = self.text()
        self.assertLess(text.index("«одинаковые»"),
                        text.index(f"«{measure.VARIANT_ARITHMETIC}»"))
        self.assertIn("считаются БЕЗ них", text)

    def test_the_prices_are_the_seconds_of_the_live_population(self):
        m = self.measurement()
        # 40 groups of two frames, at 0.5 s a call plus 1 s a frame.
        self.assertEqual(m.model.cost.seconds, 40 * 2.5)
        self.assertEqual(m.sharpness.cost.seconds, 0.0)
        self.assertIn("цена, с", measure.format_summary(m))

    def test_what_the_cascade_loses_is_printed_with_the_model_beside_it(self):
        text = self.text()
        self.assertIn("Что теряет каскад против модели", text)
        self.assertIn("из ошибок — те, где модель была права", text)

    def test_the_verdict_carries_the_criteria_that_produced_it(self):
        text = self.text()
        self.assertIn("ВЕРДИКТ ФАЗЫ 0", text)
        self.assertIn("best_rule", text)
        self.assertIn("best_cascade", text)

    def test_the_arithmetic_that_sees_the_blinks_wins_this_labelling(self):
        # The sample is built so the sharp frame is also the one with open eyes, which is
        # the brief's own claim; the rule that reads the eye must not do worse than one
        # that does not.
        m = self.measurement()
        self.assertEqual(m.best_rule.score.agreement, 1.0)
        self.assertGreater(m.model.decided, 0)

    def test_nothing_printed_identifies_a_frame(self):
        text = self.text()
        for forbidden in ("/photos", ".jpg", "file_id", "\\"):
            with self.subTest(token=forbidden):
                self.assertNotIn(forbidden, text)
        # The group keys are sha1 hex — a report that leaked one would leak the membership.
        for group_key in (g.key for g in self.measurement().population):
            self.assertNotIn(group_key, text)


if __name__ == "__main__":
    unittest.main()
