"""Who picks the keeper of a burst: arithmetic, the model, or a cascade (F180, phase 0).

The keeper question is asked of a VLM today (F132) and it is the last question of the junk
stage that costs real time. The brief's observation is that it may not have to be: closed
eyes are exactly what makes one frame of a burst worse than another, and since F179 the
eyelid geometry is a column of `frame_quality` like the two sharpness numbers next to it.
All three signals are already computed on every run:

    frame_quality.sharpness        the laplacian over the frame
    frame_quality.face_sharpness   the same laplacian inside the face (F155)
    frame_quality.eye_openness     the eye opening over the eye width (F179)

So this script measures three ways to answer the same question — a rule over those three
numbers, the model as it runs now, and a cascade that only asks where the rule is not
confident — and it chooses nothing by eye: it prints tables and a verdict computed from
criteria fixed in the code before the first run (the device of F178, the lesson of F131).

The truth here is human, and it is collected blind
-------------------------------------------------
"The best frame" cannot be derived from a metric: it is the frame a person calls best. So
the measurement needs a labelling, and `--write-sample` writes one that CANNOT be led by
the answer it is measuring — a group is listed by file id with its frames SHUFFLED, so
neither the model's pick nor the sharpness order is visible in the worksheet. A sheet that
showed either would collect a labelling of the hint.

The third answer is the point, not a footnote. `--write-sample` leaves `"choice": null`
and the owner writes a file id OR the word `same`, and F132 already measured why: the model
picked a frame other than the sharpest in 40 groups out of 73, and in those groups the
frames were indistinguishable. If the share of `same` is large again, every percentage
below is a comparison of rules on a question that does not matter, and the report says so
first (`SAME_SHARE_LOUD`) instead of quietly ranking three ways to answer it.

What the numbers here cannot be
-------------------------------
The thresholds are picked on the same groups they are scored on, by `best_rule` and not by
eye, so the arithmetic row is an upper bound on itself; a phase 1 owes a re-measurement on
fresh labels. The model row is not tuned at all — it is read from `group_keeper`, i.e. the
answers the stage really stored, so it is the baseline the brief demands rather than a
re-run of the question under kinder conditions. And the model's ceiling is printed with it:
the stage shows the model only the best `dedup.keeper_max_frames` frames of a group, so a
group where the owner picked a frame outside that window was lost by the model before it
was asked.

No model runs here. The price of the model is `KEEPER_CALL_S + KEEPER_FRAME_S x frames`,
measured on this collection and on the stage's own asker by the worksheet that priced the
keeper question, and carried in as a constant — a phase 0 that re-priced the baseline on
whatever machine it happened to run on would compare the arithmetic against seconds the
pipeline never paid. That worksheet went with the question it priced (F186 retired the
comparative keeper call); the number it produced stays here, as the record of what the
model cost when it was still asked.

Privacy: counts, seconds and group sizes only. No path, no basename and no file id reaches
the OUTPUT — a near-duplicate group is a burst of one moment, and the rule of
measure_eye_state.py holds here too. The worksheet is the one
place file ids are written, because a person has to be able to look the frames up, and it
is a file the owner asked for rather than a report. The database is opened read-only.

Usage (from the repo root; no GPU and no extras needed — this pass runs no model):
    python scripts/measure_keeper_rules.py                        # population and prices
    python scripts/measure_keeper_rules.py --write-sample keeper_sheet.json
    python scripts/measure_keeper_rules.py --labels keeper_sheet.json
    python scripts/measure_keeper_rules.py --labels keeper_sheet.json --min-size 3
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sorta import dedup, junk  # noqa: E402
from sorta.config import load_config  # noqa: E402

# --- the worksheet ------------------------------------------------------------------

# The third answer. Not a convenience: a group whose frames a person cannot tell apart is
# a group where every rule is right, and counting it as a win for all three would inflate
# each of them by the same amount and hide the finding.
CHOICE_SAME = "same"
# One filled-in cell: a file id of the group, or CHOICE_SAME. None is "not answered yet".
Choice = int | str

# --- the variants -------------------------------------------------------------------

VARIANT_ARITHMETIC = "арифметика"
VARIANT_CASCADE = "каскад"
VARIANT_MODEL = "модель"
# The fourth line of the table, and the cheapest thing that already exists: the ranking the
# Duplicates tab falls back to whenever the model produced no answer (`dedup.rank_frames`).
# It is the arithmetic with every knob off, and it is printed because a comparison that
# starts at the model reads any number as an improvement over doing nothing.
VARIANT_SHARPNESS = "резкость"

# Which sharpness the rule orders a group by. `face` means "the face number where the WHOLE
# group has it, the frame number otherwise" — the all-or-nothing convention of
# `dedup.rank_frames`, because a partial comparison quietly prefers whichever frames
# happened to be measured, and face_sharpness exists for 5 790 rows against 22 091.
FOCUS_FRAME = "резкость кадра"
FOCUS_FACE = "резкость лица"
FOCUS = (FOCUS_FRAME, FOCUS_FACE)

# The openness below which a frame counts as a blink and steps out of the comparison. Swept
# rather than chosen: 0.18 is what `features.eye_openness_max` uses for the closed-eyes
# slice (F179), but that threshold was fixed for a different job — showing a person a list
# — and inside a burst the question is a comparison. 0 is the filter switched off.
EYE_GRID = (0.0, 0.10, 0.13, 0.16, 0.18, 0.20, 0.22, 0.25)

# How far ahead the winner has to be, as a share of its own number, before the cascade
# keeps the answer to itself. The two ends of the grid are the other two variants: at 0 the
# gap is always large enough and the model is never asked, above 1 it never is and the
# model is asked about everything. A relative gap, because the laplacian has no scale that
# survives leaving a group — inside one it compares the same scene at the same size (F120).
MARGIN_GRID = (0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0, 2.0)

# --- the price of the model ----------------------------------------------------------
#
# One keeper call, in seconds: a fixed part and a part per frame in the prompt. Measured on
# this collection and on the stage's own asker, by a worksheet that no longer exists (F186
# retired the call itself); see the module docstring for why it is not re-measured here.
# `--call-seconds` / `--frame-seconds` carry a re-measurement in without touching the code.
KEEPER_CALL_S = 0.45
KEEPER_FRAME_S = 1.03

# --- pre-registered acceptance criteria (F178's device) -------------------------------
#
# 1. At least 30 groups where the owner actually named a frame. With n such groups the
#    standard error of a share near a half is 0.5/sqrt(n), so below 30 it is over 9
#    percentage points and "not worse" is not a statement the sample can carry.
# 2. "Not worse than the model" is within 5 percentage points. On the ~100 decided groups
#    the brief expects that is about one standard error — a smaller gap is not a difference
#    this labelling can see, and pretending otherwise is how a table picks a winner by noise.
# 3. A cascade is only worth its complexity if it costs at most half of the model. Anything
#    above that is the same two minutes with more code in the middle (the F130/F140 bar).
# 4. If half the labelled groups come back `same`, that IS the verdict. The number is the
#    brief's own: F132 found the frames indistinguishable in 40 groups of 73, i.e. 55%.
MIN_DECIDED_GROUPS = 30
NOT_WORSE = 0.05
CASCADE_CHEAP = 0.5
SAME_SHARE_LOUD = 0.5

VERDICT_UNCLEAR = "ВЕРДИКТА НЕТ"
VERDICT_SAME = "ВОПРОС МАЛОЗНАЧИМ"
VERDICT_ARITHMETIC = "УБРАТЬ ВОПРОС К МОДЕЛИ"
VERDICT_CASCADE = "БРАТЬ КАСКАД"
VERDICT_MODEL = "ОСТАВИТЬ МОДЕЛЬ"


@dataclass(frozen=True)
class Frame:
    """One frame of a group with everything a rule may read, and nothing else.

    The three numbers are `frame_quality`'s own columns; `pixels`, `size` and `file_id` are
    the tie-breakers `dedup.rank_frames` already uses, carried so that a rule which changes
    nothing produces exactly today's answer rather than a different one by accident.
    """
    file_id: int
    sharpness: float | None = None
    face_sharpness: float | None = None
    eye_openness: float | None = None
    pixels: int = 0
    size: int = 0


@dataclass(frozen=True)
class Group:
    """A near-duplicate group: its key, and its frames in the order the stage ranks them.

    `key` is `dedup.group_key` — a hash of the membership, so a worksheet keeps working
    exactly as long as the group it was written about still exists in that shape, and a
    group that gained or lost a frame is simply not found instead of being mislabelled.
    """
    key: str
    frames: tuple[Frame, ...] = ()


@dataclass(frozen=True)
class Rule:
    """One arithmetic rule: which frames the blink filter drops, and what orders the rest."""
    eye_min: float
    focus: str


@dataclass(frozen=True)
class Decision:
    """What a rule answered about a group, and how far ahead its answer was.

    `margin` is the whole of the cascade: it is the share of the winner's number that
    separates it from the runner-up, so 0 means "these two are the same picture" and 1
    means "everyone else blinked". A rule that could not order the group at all (no column
    complete) reports 0 — not confident, which sends the group to the model.
    """
    file_id: int
    margin: float


@dataclass(frozen=True)
class Pick:
    """One variant's answer about one group, with what it cost to arrive at it."""
    file_id: int
    asked: bool = False    # would this variant put the group to the model on a real run
    silent: bool = False   # it would, and the index holds no model answer for this group


Picker = Callable[[Group], Pick]


@dataclass(frozen=True)
class Cost:
    """What a variant costs on the LIVE population — groups asked about, and seconds."""
    asked: int = 0
    seconds: float = 0.0


@dataclass(frozen=True)
class Score:
    """How one variant did against the owner's choice, over the labelled groups."""
    variant: str
    decided: int = 0    # groups where the owner named a frame
    agreed: int = 0     # of those, the ones this variant named too
    same: int = 0       # groups where the owner said the frames are the same
    silent: int = 0     # groups it would have asked about and found no stored answer for
    cost: Cost = field(default_factory=Cost)

    @property
    def labelled(self) -> int:
        return self.decided + self.same

    @property
    def agreement(self) -> float:
        """The headline: the `same` groups are OUT, because there every rule is right."""
        return self.agreed / self.decided if self.decided else 0.0

    @property
    def lenient(self) -> float:
        """The other reading — `same` counted as a win, which it is for all three."""
        return (self.agreed + self.same) / self.labelled if self.labelled else 0.0


@dataclass(frozen=True)
class Loss:
    """What a cascade gives up against asking every time — item 5 of the brief."""
    quiet: int = 0         # groups it decided by arithmetic alone
    wrong: int = 0         # of those, the ones it got wrong
    model_right: int = 0   # of those, the ones the model would have got right


@dataclass(frozen=True)
class RuleRow:
    """One row of the arithmetic table: a rule and what it scored."""
    rule: Rule
    score: Score


@dataclass(frozen=True)
class CascadeRow:
    """One row of the cascade table: a confidence margin, its score and what it lost."""
    margin: float
    score: Score
    loss: Loss


@dataclass(frozen=True)
class Prices:
    """The cost model of one keeper question, and how many frames a question may hold."""
    call_s: float = KEEPER_CALL_S
    frame_s: float = KEEPER_FRAME_S
    max_frames: int = 5


# --- the arithmetic -------------------------------------------------------------------


def column(frames: Sequence[Frame], focus: str) -> list[float] | None:
    """The number this group is ordered by, one per frame — or None when there is none.

    All or nothing, the convention of `dedup.rank_frames`: the face number is used only
    when EVERY frame of the group carries it, and the frame number only when every frame
    carries that. A group where neither is complete is ordered by resolution and size
    alone, which is what the pipeline does today and not a special case invented here.
    """
    if not frames:
        return None
    tried = [[f.face_sharpness for f in frames]] if focus == FOCUS_FACE else []
    tried.append([f.sharpness for f in frames])
    for values in tried:
        if all(value is not None for value in values):
            return [float(value) for value in values if value is not None]
    return None


def judge(frames: Sequence[Frame], rule: Rule) -> Decision:
    """The rule's answer about one group: which frame, and how far ahead of the next.

    Two steps, in this order. The blink filter first, because that is the brief's whole
    observation — a frame where somebody's eyes are shut is not the best frame of the
    burst whatever its focus is. Then the sharpness, which inside a group is the one place
    the laplacian answers the question it was measured for (F120): the frames are the same
    scene at the same scale.

    A filter that would empty the group is not applied. If everybody blinked, the group is
    still a group and the comparison falls back to focus alone — dropping every frame would
    mean answering "none of them", which is not one of the answers.
    """
    kept = [f for f in frames if f.eye_openness is None or f.eye_openness >= rule.eye_min]
    if not kept:
        kept = list(frames)
    if not kept:
        return Decision(file_id=0, margin=0.0)
    values = column(kept, rule.focus)
    order = sorted(range(len(kept)),
                   key=lambda i: (-(values[i] if values else 0.0), -kept[i].pixels,
                                  -kept[i].size, kept[i].file_id))
    best = kept[order[0]]
    if len(kept) == 1:
        # Everyone else blinked out of the comparison: there is nothing left to be close to.
        margin = 1.0 if len(frames) > 1 else 0.0
    elif values and values[order[0]] > 0:
        top, second = values[order[0]], values[order[1]]
        margin = max(0.0, (top - second) / top)
    else:
        margin = 0.0
    return Decision(file_id=best.file_id, margin=margin)


def rules() -> list[Rule]:
    """Every rule of the sweep, in the order the tables print them."""
    return [Rule(eye_min=eye, focus=focus) for focus in FOCUS for eye in EYE_GRID]


# --- the three variants as pickers -------------------------------------------------------


def arithmetic(rule: Rule) -> Picker:
    """The rule alone. Costs nothing and asks nothing: all three numbers are on disk."""
    return lambda group: Pick(file_id=judge(group.frames, rule).file_id)


def model(choices: Mapping[str, int]) -> Picker:
    """The model as it runs now — its stored answer, or the fallback the stage really uses.

    A group with no row in `group_keeper` is not skipped: the stage falls back to the top
    of the sharpness ranking there (junk.py, `keeper, chosen_by = group[0].file_id, ...`),
    and the baseline the brief asks for is the CURRENT BEHAVIOUR, fallback included.
    Counting those groups as silent rather than dropping them keeps the model's column
    honest about how often it produced nothing.
    """
    def pick(group: Group) -> Pick:
        chosen = choices.get(group.key)
        fallback = group.frames[0].file_id if group.frames else 0
        return Pick(file_id=chosen if chosen is not None else fallback,
                    asked=True, silent=chosen is None)

    return pick


def cascade(rule: Rule, margin: float, choices: Mapping[str, int]) -> Picker:
    """The rule where it is confident, the model where it is not."""
    ask = model(choices)

    def pick(group: Group) -> Pick:
        decision = judge(group.frames, rule)
        if decision.margin >= margin:
            return Pick(file_id=decision.file_id)
        return ask(group)

    return pick


# --- what a run would pay ---------------------------------------------------------------


def call_seconds(frames: int, prices: Prices) -> float:
    """Seconds for one keeper question about a group of `frames` frames.

    Only the frames really sent are priced: the stage shows the best `keeper_max_frames`
    of a group (junk.py), so a burst of 38 frames costs a question about five. A group
    that would show fewer than two frames is never asked and costs nothing — one image is
    not a comparison.
    """
    shown = min(frames, prices.max_frames)
    return prices.call_s + prices.frame_s * shown if shown > 1 else 0.0


def cost(population: Sequence[Group], chose: Picker, prices: Prices) -> Cost:
    """The variant's price over the whole live population, not over the labelled sample."""
    asked = [g for g in population if chose(g).asked]
    return Cost(asked=len(asked),
                seconds=sum(call_seconds(len(g.frames), prices) for g in asked))


# --- scoring against the owner ------------------------------------------------------------


def score(variant: str, groups: Sequence[Group], labels: Mapping[str, Choice],
          chose: Picker, spent: Cost) -> Score:
    """One variant against the labelling: agreed, missed, and how often it said nothing."""
    decided = agreed = same = silent = 0
    for group in groups:
        answer = labels.get(group.key)
        if answer is None:
            continue
        pick = chose(group)
        silent += int(pick.silent)
        if answer == CHOICE_SAME:
            same += 1
            continue
        decided += 1
        agreed += int(pick.file_id == answer)
    return Score(variant=variant, decided=decided, agreed=agreed, same=same,
                 silent=silent, cost=spent)


def loss(groups: Sequence[Group], labels: Mapping[str, Choice],
         quiet: Picker, ask: Picker) -> Loss:
    """Where the cascade kept quiet and was wrong — and whether the model knew better.

    The second half is the one that decides anything: a group the cascade got wrong and the
    model would have got wrong too costs nothing to stop asking about.
    """
    decided = missed = recoverable = 0
    for group in groups:
        answer = labels.get(group.key)
        if answer is None or answer == CHOICE_SAME:
            continue
        pick = quiet(group)
        if pick.asked:
            continue
        decided += 1
        if pick.file_id != answer:
            missed += 1
            recoverable += int(ask(group).file_id == answer)
    return Loss(quiet=decided, wrong=missed, model_right=recoverable)


def unseen_picks(groups: Sequence[Group], labels: Mapping[str, Choice],
                 max_frames: int) -> int:
    """Groups where the owner picked a frame the model is never shown — its own ceiling."""
    count = 0
    for group in groups:
        answer = labels.get(group.key)
        if answer is None or answer == CHOICE_SAME:
            continue
        if answer not in {f.file_id for f in group.frames[:max_frames]}:
            count += 1
    return count


# --- the rules picked by rule, never by eye -------------------------------------------------


def best_rule(rows: Sequence[RuleRow]) -> RuleRow | None:
    """The most agreement the sweep reaches; ties go to the rule that assumes the least.

    Fixed here before the run for the reason F131 paid for: a threshold chosen by looking
    at the table measures the person reading it. On a tie the frame number wins over the
    face one (it exists for four times as many rows) and the lower blink threshold wins
    over the higher one (a knob that changes no number belongs switched off).
    """
    if not rows:
        return None
    return max(rows, key=lambda r: (r.score.agreement, r.rule.focus == FOCUS_FRAME,
                                    -r.rule.eye_min))


def best_cascade(rows: Sequence[CascadeRow], floor: float) -> CascadeRow | None:
    """The cascade that asks least while staying within `floor` of the model.

    The other way round from `best_rule`, and deliberately: a cascade exists to buy back
    seconds, so among the rows that hold the quality bar the right one is the cheapest, not
    the best. If nothing holds the bar the table has no cascade to offer and the row with
    the most agreement is returned so the report can say what the best one still was.
    """
    if not rows:
        return None
    usable = [r for r in rows if r.score.agreement >= floor]
    if not usable:
        return max(rows, key=lambda r: (r.score.agreement, -r.score.cost.asked))
    return min(usable, key=lambda r: (r.score.cost.asked, -r.score.agreement))


def same_share(base: Score) -> float:
    """The share of labelled groups whose frames the owner could not tell apart."""
    return base.same / base.labelled if base.labelled else 0.0


def decide(rule_row: RuleRow, cascade_row: CascadeRow, base: Score) -> tuple[str, str]:
    """The verdict, from the criteria above and from nothing else."""
    best = rule_row.rule
    if base.decided < MIN_DECIDED_GROUPS:
        return VERDICT_UNCLEAR, (f"групп с выбранным кадром {base.decided} < "
                                 f"{MIN_DECIDED_GROUPS} — совпадение не измеряется")
    share = same_share(base)
    if share >= SAME_SHARE_LOUD:
        return VERDICT_SAME, (
            f"«одинаковые» — {share:.0%} размеченных групп ({base.same} из "
            f"{base.labelled}): в них право любое правило, и оптимизировать выбор не "
            f"стоит вовсе — правильный ход перестать делать вид, что он важен")
    floor = base.agreement - NOT_WORSE
    if rule_row.score.agreement >= floor:
        return VERDICT_ARITHMETIC, (
            f"арифметика ({best.focus}, порог глаза {best.eye_min:g}) совпадает с "
            f"человеком в {rule_row.score.agreement:.0%} против {base.agreement:.0%} у "
            f"модели — вопрос к VLM убирается, а `group_keeper` заполняется бесплатно")
    if (cascade_row.score.agreement >= floor
            and cascade_row.score.cost.seconds <= base.cost.seconds * CASCADE_CHEAP):
        return VERDICT_CASCADE, (
            f"каскад с запасом {cascade_row.margin:g} совпадает в "
            f"{cascade_row.score.agreement:.0%} против {base.agreement:.0%} у модели, "
            f"спрашивая {cascade_row.score.cost.asked} из {base.cost.asked} групп "
            f"({cascade_row.score.cost.seconds:.0f} с против {base.cost.seconds:.0f} с)")
    # Two ways to land here, and they are not the same finding: either nothing held the
    # quality bar, or the cascade held it and bought back too little to be worth the code.
    gap = (f"каскад качество держит, но стоит {cascade_row.score.cost.seconds:.0f} с "
           f"против {base.cost.seconds:.0f} с у модели — дешевле "
           f"{CASCADE_CHEAP:.0%} он не вышел"
           if cascade_row.score.agreement >= floor
           else f"разрыв с моделью больше {NOT_WORSE:.0%} у обоих")
    return VERDICT_MODEL, (
        f"модель совпадает в {base.agreement:.0%}, лучшая арифметика — в "
        f"{rule_row.score.agreement:.0%}, лучший каскад — в "
        f"{cascade_row.score.agreement:.0%}; {gap}, а "
        f"{base.cost.seconds / 60.0:.0f} мин на коллекцию — не та цена, ради которой "
        f"стоит терять качество")


# --- the measurement --------------------------------------------------------------------


@dataclass(frozen=True)
class Measurement:
    """Everything the report prints, computed once and formatted separately."""
    population: tuple[Group, ...]
    labelled: tuple[Group, ...]
    labels: Mapping[str, Choice]
    rows: tuple[RuleRow, ...]
    cascades: tuple[CascadeRow, ...]
    model: Score
    sharpness: Score
    unseen: int
    prices: Prices

    @property
    def best_rule(self) -> RuleRow:
        row = best_rule(self.rows)
        assert row is not None  # `rules()` is never empty
        return row

    @property
    def best_cascade(self) -> CascadeRow:
        row = best_cascade(self.cascades, self.model.agreement - NOT_WORSE)
        assert row is not None  # MARGIN_GRID is never empty
        return row


def measure(population: Sequence[Group], labels: Mapping[str, Choice],
            choices: Mapping[str, int], prices: Prices) -> Measurement:
    """Every variant over every threshold, priced on the population and scored on the sheet.

    The cascade is swept over the arithmetic that won its own table rather than over all of
    them: which rule the cascade carries is not a free parameter, it is the answer of the
    step before, and sweeping both would be choosing a pair out of a hundred on a hundred
    groups.
    """
    labelled = tuple(g for g in population if g.key in labels)
    ask = model(choices)
    base = score(VARIANT_MODEL, labelled, labels, ask, cost(population, ask, prices))
    rows = tuple(RuleRow(rule=rule,
                         score=score(VARIANT_ARITHMETIC, labelled, labels,
                                     arithmetic(rule), Cost()))
                 for rule in rules())
    chosen = best_rule(rows)
    assert chosen is not None
    cascades = []
    for margin in MARGIN_GRID:
        mixed = cascade(chosen.rule, margin, choices)
        cascades.append(CascadeRow(
            margin=margin,
            score=score(VARIANT_CASCADE, labelled, labels, mixed,
                        cost(population, mixed, prices)),
            loss=loss(labelled, labels, mixed, ask)))
    today = arithmetic(Rule(eye_min=0.0, focus=FOCUS_FRAME))
    return Measurement(
        population=tuple(population), labelled=labelled, labels=labels, rows=rows,
        cascades=tuple(cascades), model=base,
        sharpness=score(VARIANT_SHARPNESS, labelled, labels, today, Cost()),
        unseen=unseen_picks(labelled, labels, prices.max_frames), prices=prices)


# --- the report ---------------------------------------------------------------------------


def histogram(groups: Sequence[Sequence[object]]) -> dict[int, int]:
    """Group size -> how many groups of it."""
    out: dict[int, int] = {}
    for group in groups:
        out[len(group)] = out.get(len(group), 0) + 1
    return dict(sorted(out.items()))


def baseline_seconds(near: Mapping[int, int], min_size: int, prices: Prices) -> float:
    """What the model costs on the whole population, straight off the size histogram.

    The same `call_seconds` the variants are priced with, computed here because it needs
    nothing but the pHashes: this is the one number of the brief a collection can answer
    before anybody has labelled a single group.
    """
    return sum(count * call_seconds(size, prices)
               for size, count in near.items() if size >= min_size)


def format_population(exact: Mapping[int, int], near: Mapping[int, int], min_size: int,
                      prices: Prices) -> str:
    """Item 1: how many groups there are to measure on, and by which reading of "duplicate".

    Both readings, because the difference is the whole population question: exact pHash
    equality gives a couple of dozen groups on this collection and real nearness gives
    over a hundred, and a table built on the first would be a table about nothing.
    """
    def wide_enough(hist: Mapping[int, int]) -> int:
        return sum(count for size, count in hist.items() if size >= min_size)

    wide = wide_enough(near)
    frames = sum(size * count for size, count in near.items() if size >= min_size)
    lines = ["Население",
             f"  групп по точному совпадению phash: {sum(exact.values())}, "
             f"из них от {min_size} кадров: {wide_enough(exact)}",
             f"  групп по настоящей близости (`index.phash_max_distance`): "
             f"{sum(near.values())}, из них от {min_size} кадров: {wide}",
             "  размер группы -> сколько таких (по настоящей близости):"]
    lines += [f"    {size:>3}: {count}" for size, count in sorted(near.items())]
    lines.append(f"  популяция замера — групп: {wide}, кадров в них: {frames}")
    seconds = baseline_seconds(near, min_size, prices)
    lines.append(f"  цена нынешнего поведения (чистая модель) на ней: {seconds:.0f} с "
                 f"({seconds / 60.0:.1f} мин); арифметика на той же популяции — 0 с")
    return "\n".join(lines)


def format_sample(m: Measurement) -> str:
    """Item 3: how much of the labelling says the frames cannot be told apart."""
    base = m.model
    share = same_share(base)
    lines = [f"\nРазметка: {base.labelled} из {len(m.population)} групп "
             f"(лист слепой: кадры вперемешку, ответ модели не показан)",
             f"  выбран кадр: {base.decided}",
             f"  «одинаковые»: {base.same} ({share:.0%}) — в этих группах право любое "
             f"правило,\n    поэтому все проценты ниже считаются БЕЗ них"]
    if m.unseen:
        lines.append(f"  выбор человека вне первых {m.prices.max_frames} кадров — "
                     f"групп: {m.unseen}; этих кадров модель не видела вовсе")
    if base.silent:
        lines.append(f"  без ответа модели в `group_keeper` — групп: {base.silent}; "
                     f"там она отвечает резкостью, как и на прогоне")
    return "\n".join(lines)


def format_rules(m: Measurement, focus: str) -> str:
    """One arithmetic table: the blink threshold swept, the baseline under it."""
    lines = [f"\n«{VARIANT_ARITHMETIC}», {focus} (порог глаза не выбран заранее — "
             f"его выбирает `best_rule`)",
             f"{'глаз':>7} | {'групп':>6} | {'совпало':>8} | {'совпадение':>11} | "
             f"{'с «одинаковыми»':>16}",
             "-" * 60]
    for row in m.rows:
        if row.rule.focus != focus:
            continue
        label = "выкл" if row.rule.eye_min <= 0 else f"{row.rule.eye_min:g}"
        lines.append(f"{label:>7} | {row.score.decided:>6} | {row.score.agreed:>8} | "
                     f"{row.score.agreement:>11.0%} | {row.score.lenient:>16.0%}")
    lines.append(f"{VARIANT_MODEL:>7} | {m.model.decided:>6} | {m.model.agreed:>8} | "
                 f"{m.model.agreement:>11.0%} | {m.model.lenient:>16.0%}   "
                 f"<- базовая линия")
    return "\n".join(lines)


def format_cascades(m: Measurement) -> str:
    """The cascade table: what each confidence margin asks for, and what it costs."""
    best = m.best_rule.rule
    lines = [f"\n«{VARIANT_CASCADE}» на лучшей арифметике ({best.focus}, порог глаза "
             f"{best.eye_min:g})",
             f"{'запас':>7} | {'спросит':>8} | {'цена, с':>9} | {'совпадение':>11} | "
             f"{'молча и мимо':>13} | {'из них модель права':>20}",
             "-" * 84]
    for row in m.cascades:
        lines.append(f"{row.margin:>7g} | {row.score.cost.asked:>8} | "
                     f"{row.score.cost.seconds:>9.0f} | {row.score.agreement:>11.0%} | "
                     f"{row.loss.wrong:>13} | {row.loss.model_right:>20}")
    lines.append(f"{VARIANT_MODEL:>7} | {m.model.cost.asked:>8} | "
                 f"{m.model.cost.seconds:>9.0f} | {m.model.agreement:>11.0%} | "
                 f"{'-':>13} | {'-':>20}   <- базовая линия")
    return "\n".join(lines)


def format_summary(m: Measurement) -> str:
    """Items 2, 4 and 6 in one table: the three variants and the do-nothing line."""
    best = m.best_rule
    cascade_row = m.best_cascade
    lines = [f"\nТри варианта на боевой популяции (групп: {len(m.population)}; "
             f"{m.prices.call_s:g} с на вызов + {m.prices.frame_s:g} с на кадр)",
             f"{'вариант':>11} | {'совпадение':>11} | {'с «одинаковыми»':>16} | "
             f"{'спросит':>8} | {'цена, с':>9}",
             "-" * 70]
    for label, row in ((VARIANT_SHARPNESS, m.sharpness),
                       (VARIANT_ARITHMETIC, best.score),
                       (VARIANT_CASCADE, cascade_row.score),
                       (VARIANT_MODEL, m.model)):
        tail = "   <- базовая линия" if label == VARIANT_MODEL else ""
        lines.append(f"{label:>11} | {row.agreement:>11.0%} | {row.lenient:>16.0%} | "
                     f"{row.cost.asked:>8} | {row.cost.seconds:>9.0f}{tail}")
    lines.append(f"  арифметика — {best.rule.focus}, порог глаза {best.rule.eye_min:g}; "
                 f"каскад — запас {cascade_row.margin:g}")
    lines.append(f"  «{VARIANT_SHARPNESS}» — это то, что интерфейс показывает и сегодня, "
                 f"когда модель не ответила")
    return "\n".join(lines)


def format_losses(m: Measurement) -> str:
    """Item 5: on how many groups the cascade did not ask and was wrong."""
    row = m.best_cascade
    return (f"\nЧто теряет каскад против модели (запас {row.margin:g})\n"
            f"  решил сам: {row.loss.quiet} из {m.model.decided} групп\n"
            f"  из них ошибся: {row.loss.wrong}\n"
            f"  из ошибок — те, где модель была права: {row.loss.model_right} "
            f"(остальные она тоже не угадала)")


def format_verdict(m: Measurement) -> str:
    verdict, why = decide(m.best_rule, m.best_cascade, m.model)
    return (f"\nВЕРДИКТ ФАЗЫ 0: {verdict}\n  {why}\n"
            f"  (критерии зафиксированы до прогона: групп с выбором "
            f">= {MIN_DECIDED_GROUPS}, «не хуже» — в пределах {NOT_WORSE:.0%},\n"
            f"   каскад берут не дороже {CASCADE_CHEAP:.0%} цены модели, доля "
            f"«одинаковых» от {SAME_SHARE_LOUD:.0%} — сама по себе вердикт;\n"
            f"   порог выбирает `best_rule`, запас каскада — `best_cascade`, не глаз)")


def report(m: Measurement) -> str:
    """The whole report except the population block, which is read before the sheet."""
    parts = [format_sample(m)]
    parts += [format_rules(m, focus) for focus in FOCUS]
    parts += [format_cascades(m), format_summary(m), format_losses(m), format_verdict(m)]
    return "\n".join(parts)


# --- the worksheet ---------------------------------------------------------------------


def sheet(groups: Sequence[Group], seed: int) -> dict[str, dict[str, object]]:
    """A blind worksheet: `{group_key: {"frames": [file ids], "choice": null}}`.

    Shuffled with a seeded RNG, and that is the point of the whole file: the frames arrive
    in the order the stage ranks them, so writing them out as they came would hand the
    person the sharpness answer and the model's first choice at the same time. Nothing else
    is written — no score, no path, no stored keeper.
    """
    rng = random.Random(seed)
    out: dict[str, dict[str, object]] = {}
    for group in groups:
        ids = [f.file_id for f in group.frames]
        rng.shuffle(ids)
        out[group.key] = {"frames": ids, "choice": None}
    return out


def write_sheet(path: str, groups: Sequence[Group], seed: int) -> int:
    """Write the worksheet and report how many groups it holds."""
    payload = sheet(groups, seed)
    Path(path).write_text(json.dumps(payload, indent=1, ensure_ascii=False),
                          encoding="utf-8")
    return len(payload)


def read_sheet(path: str) -> dict[str, Choice]:
    """The filled-in worksheet -> `{group_key: file id | "same"}`; nulls are dropped.

    An unanswered group is not an answer and takes part in nothing — the same rule
    measure_downloaded.py and measure_detector.py follow, and for the same reason: reading
    a blank as a miss would invent exactly the marks the sheet exists to collect. A choice
    that is not one of the group's own frames is refused outright rather than dropped: it
    means the sheet and the index have drifted apart, and every number below would then be
    computed on a labelling of something else.
    """
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"{path}: файл разметки не читается ({exc})") from None
    except ValueError as exc:
        raise SystemExit(f"{path}: это не JSON ({exc})") from None
    if not isinstance(raw, dict):
        raise SystemExit(f"{path}: ожидается отображение «ключ группы -> выбор»")
    out: dict[str, Choice] = {}
    for key, value in raw.items():
        if not isinstance(value, dict) or not isinstance(value.get("frames"), list):
            raise SystemExit(f"{path}: у группы «{key}» нет списка `frames`")
        choice = value.get("choice")
        if choice is None:
            continue
        if choice == CHOICE_SAME:
            out[str(key)] = CHOICE_SAME
            continue
        try:
            chosen = int(choice)
        except (TypeError, ValueError):
            raise SystemExit(f"{path}: выбор «{choice}» в группе «{key}» — не id кадра "
                             f"и не «{CHOICE_SAME}»") from None
        if chosen not in [int(i) for i in value["frames"]]:
            raise SystemExit(f"{path}: кадра {chosen} нет в группе «{key}» — лист и "
                             f"индекс разошлись, разметку надо переснять")
        out[str(key)] = chosen
    if not out:
        raise SystemExit(f"{path}: ни одна группа не размечена — мерить не на чем")
    return out


# --- reading the collection ---------------------------------------------------------------


def read_groups(conn: sqlite3.Connection, max_distance: int,
                min_size: int = 2) -> list[Group]:
    """The near-duplicate groups with everything a rule reads, ranked as the stage ranks them.

    `dedup.keeper_groups` supplies the groups, the ranking and two of the three numbers;
    the other two columns come from `junk.read_frame_quality`, which is the reading side of
    the "NULL is not False" rule — nothing here rebuilds either.
    """
    ranked = dedup.keeper_groups(conn, max_distance, min_size=min_size)
    if not ranked:
        return []
    try:
        quality = junk.read_frame_quality(conn, [f.file_id for g in ranked for f in g])
    except sqlite3.OperationalError as exc:
        # An index written before the F179 migration has no `eye_openness` column at all,
        # and without the third signal there is no third variant to measure. Said out loud
        # rather than as a traceback: the fix is a run of the junk stage, not of this script.
        raise SystemExit(f"индекс не отдаёт колонки `frame_quality` ({exc}) — похоже, он "
                         f"старше миграции F179; прогоните стадию junk и повторите") from None
    out: list[Group] = []
    for group in ranked:
        frames = []
        for f in group:
            row = quality.get(f.file_id)
            frames.append(Frame(file_id=f.file_id, sharpness=f.sharpness,
                                face_sharpness=row.face_sharpness if row else None,
                                eye_openness=row.eye_openness if row else None,
                                pixels=f.pixels, size=f.size))
        out.append(Group(key=dedup.group_key([f.file_id for f in group]),
                         frames=tuple(frames)))
    return out


def model_choices(conn: sqlite3.Connection) -> dict[str, int]:
    """What the MODEL answered, out of `group_keeper` — never out of a second file.

    Rows written by the sharpness fallback are not the model's answers and are left out:
    counting them would score the arithmetic against itself and report a suspiciously
    strong baseline.
    """
    return {key: row.keeper_id
            for key, row in dedup.read_group_keepers(conn).items()
            if row.source != dedup.KEEPER_SOURCE_SHARPNESS}


def run(args: argparse.Namespace) -> int:  # pragma: no cover — I/O over the live index
    cfg = load_config(args.config)
    min_size = args.min_size or int(cfg.dedup.keeper_min_group_size)
    prices = Prices(call_s=args.call_seconds, frame_s=args.frame_seconds,
                    max_frames=args.max_frames or int(cfg.dedup.keeper_max_frames))
    conn = sqlite3.connect(f"file:{cfg.database}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        # Item 1 first, and off `near_duplicate_groups` alone: the population question is
        # answered by the pHashes, so it has to be printed even on an index too old to carry
        # the third signal. The cost is one more union-find pass (`keeper_groups` repeats
        # it below), which is seconds on tens of thousands of files since F66.
        exact = histogram(dedup.near_duplicate_groups(conn, 0))
        near = histogram(dedup.near_duplicate_groups(conn, cfg.index.phash_max_distance))
        print(format_population(exact, near, min_size, prices))
        every = read_groups(conn, cfg.index.phash_max_distance, min_size=2)
        choices = model_choices(conn)
    finally:
        conn.close()
    population = [g for g in every if len(g.frames) >= min_size]
    if not population:
        raise SystemExit(f"групп от {min_size} кадров в индексе нет — мерить нечего")

    if args.write_sample:
        written = write_sheet(args.write_sample, population, args.seed)
        print(f"\nлист записан: {args.write_sample}; групп в нём: {written}\n"
              f"  в каждой группе `frames` перемешаны; впишите в `choice` id кадра "
              f"или «{CHOICE_SAME}»")
        return 0

    if not args.labels:
        print("\nразметки нет: совпадение с человеком не считается. Лист — "
              "`--write-sample`, затем `--labels`.")
        return 0
    labels = read_sheet(args.labels)
    known = {g.key for g in population}
    stale = [key for key in labels if key not in known]
    if stale:
        print(f"\nгрупп из листа больше нет в индексе: {len(stale)}; состав группы "
              f"изменился, они выпадают из замера")
    print(report(measure(population, labels, choices, prices)))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--labels", help="the filled-in worksheet (see --write-sample)")
    ap.add_argument("--write-sample", help="write a blind worksheet and exit")
    ap.add_argument("--min-size", type=int, default=0,
                    help="override dedup.keeper_min_group_size for the population")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="override dedup.keeper_max_frames (what one question shows)")
    ap.add_argument("--call-seconds", type=float, default=KEEPER_CALL_S,
                    help=f"seconds per keeper call, measured (default {KEEPER_CALL_S})")
    ap.add_argument("--frame-seconds", type=float, default=KEEPER_FRAME_S,
                    help=f"seconds per frame in the prompt (default {KEEPER_FRAME_S})")
    ap.add_argument("--seed", type=int, default=20260804,
                    help="the shuffle of the worksheet — the same seed, the same sheet")
    return run(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
