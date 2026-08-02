"""Price the keeper question (F132) BEFORE its total cost is quoted anywhere.

The brief of this feature says the worker may not name a total until the per-call cost
has been measured, and it says it because the one number in hand was measured for a
different question: 0.78 s per frame (F113) is a prompt with ONE image in it. The keeper
question carries up to five, and at 896 px a frame is over a thousand visual tokens — so
what a group costs is somewhere above 0.78 s and nobody has looked. Multiplying 791 groups
by a number measured on another shape is exactly the arithmetic that turns into "it will
take ten minutes" and then takes an hour.

So this script prints, in this order and never the other way round:

1. what the collection actually holds — groups by size, and the population each value of
   `dedup.keeper_min_group_size` leaves;
2. SECONDS PER CALL, measured on a sample of real groups with the stage's own asker,
   prompt and parser;
3. only then the projected cost of the full population, at that measured rate.

If the projection crosses the budget the brief set (30 minutes), the report says so and
names the setting that fixes it (`keeper_min_group_size: 3`, which drops the 85% of
groups that are pairs — the ones where sharpness compares two frames of one scene at one
scale honestly).

Nothing is reimplemented: `junk.qwen_vlm_keeper`, `junk.keeper_prompt`,
`junk.parse_keeper_answer` and `dedup.keeper_groups` are the pipeline's own, so the table
prices the question that actually runs (the lesson scripts/measure_ocr_gate.py opens with).

Privacy: nothing printed identifies a frame. No path, no basename, no file id — only
counts, seconds and the size of a group. The same rule the other measurement scripts
follow, and it matters more here: a near-duplicate group is a burst of one moment.

Usage (from the repo root, with a GPU venv — `uv sync --extra gpu --extra vlm`):
    python scripts/measure_group_keeper.py                  # 20 groups, config.yaml
    python scripts/measure_group_keeper.py --groups 40
    python scripts/measure_group_keeper.py --min-size 3     # price the narrow population
"""
from __future__ import annotations

import argparse
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sorta import dedup, junk  # noqa: E402
from sorta.config import load_config  # noqa: E402
from sorta.db import connect  # noqa: E402

# The budget the brief set for the whole pass, in minutes. Not a threshold the code acts
# on — the report states which side of it the projection falls, and narrowing the
# population stays a decision for a person in front of the table.
BUDGET_MINUTES = 30.0
DEFAULT_SAMPLE = 20


@dataclass
class Measurement:
    """What one sample of groups cost, and what came back from it."""
    seconds: list[float] = field(default_factory=list)   # one per call
    frames: list[int] = field(default_factory=list)      # frames shown in that call
    answered: int = 0        # answers that parsed into a frame number
    moved: int = 0           # of those, the ones that did NOT pick the sharpest frame

    @property
    def calls(self) -> int:
        return len(self.seconds)

    @property
    def per_call(self) -> float:
        """Seconds per call, the MEDIAN — a group that hit a cold cache is not the rate."""
        return statistics.median(self.seconds) if self.seconds else 0.0


def size_histogram(groups: Sequence[Sequence[dedup.GroupFrame]]) -> dict[int, int]:
    """Group size -> how many groups of it. The first thing the population question needs."""
    out: dict[int, int] = {}
    for group in groups:
        out[len(group)] = out.get(len(group), 0) + 1
    return dict(sorted(out.items()))


def measure(groups: Sequence[Sequence[dedup.GroupFrame]], ask: junk.KeeperAskFn,
            max_frames: int, clock: Callable[[], float] = time.perf_counter,
            ) -> Measurement:
    """Ask about each group, timing the calls — the stage's own prompt and parser.

    A group is shown the best `max_frames` frames by sharpness, exactly as the stage sends
    them, because the number of images in the prompt is the thing being priced. An answer
    that raises or does not parse still costs its seconds and is counted as a call: a
    projection that quietly dropped the failures would price a model that never fails.
    """
    out = Measurement()
    for group in groups:
        shown = list(group[:max_frames])
        if len(shown) < 2:
            continue
        started = clock()
        try:
            answer = ask([f.path for f in shown])
        except Exception as exc:  # noqa: BLE001 — a failure is a measurement, not a stop
            print(f"  вызов не удался: {exc}")
            answer = ""
        out.seconds.append(clock() - started)
        out.frames.append(len(shown))
        choice = junk.parse_keeper_answer(answer, len(shown))
        if choice is not None:
            out.answered += 1
            out.moved += choice != 1  # index 1 is the sharpest frame of the group
    return out


def population_lines(groups: Sequence[Sequence[dedup.GroupFrame]]) -> list[str]:
    """The collection as it is: sizes, and what each `keeper_min_group_size` leaves."""
    hist = size_histogram(groups)
    frames = sum(len(g) for g in groups)
    lines = [f"групп почти-дублей: {len(groups)}, кадров в них: {frames}",
             "размер группы -> сколько таких:"]
    lines += [f"  {size:>3}: {count}" for size, count in hist.items()]
    for min_size in (2, 3):
        kept = sum(count for size, count in hist.items() if size >= min_size)
        lines.append(f"keeper_min_group_size: {min_size} -> {kept} вызовов")
    return lines


def rate_lines(m: Measurement) -> list[str]:
    """The measured cost per call. Printed BEFORE any total — see the module docstring."""
    if not m.seconds:
        return ["замер не состоялся: ни одной группы не удалось спросить"]
    frames = statistics.median(m.frames)
    return [
        f"замер: {m.calls} вызовов, по {frames:.0f} кадра в вопросе",
        f"  секунд на вызов: медиана {m.per_call:.2f}, среднее "
        f"{statistics.fmean(m.seconds):.2f}, минимум {min(m.seconds):.2f}, "
        f"максимум {max(m.seconds):.2f}",
        f"  разобрано ответов: {m.answered}/{m.calls}; из них выбрали не самый резкий "
        f"кадр: {m.moved}",
    ]


def cost_lines(m: Measurement, groups: Sequence[Sequence[dedup.GroupFrame]]) -> list[str]:
    """The projection — only ever computed from the rate measured just above it."""
    if not m.seconds:
        return []
    hist = size_histogram(groups)
    lines = ["цена полного прохода при этой скорости:"]
    for min_size in (2, 3):
        calls = sum(count for size, count in hist.items() if size >= min_size)
        minutes = calls * m.per_call / 60.0
        verdict = "в бюджет" if minutes <= BUDGET_MINUTES else "ДОРОЖЕ БЮДЖЕТА"
        lines.append(f"  keeper_min_group_size: {min_size} -> {calls} вызовов x "
                     f"{m.per_call:.2f} с = {minutes:.1f} мин ({verdict})")
    widest = sum(count for size, count in hist.items() if size >= 2) * m.per_call / 60.0
    if widest > BUDGET_MINUTES:
        lines.append(f"  бюджет {BUDGET_MINUTES:.0f} мин пробит на всех группах — "
                     f"сузить популяцию до keeper_min_group_size: 3")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--groups", type=int, default=DEFAULT_SAMPLE,
                    help=f"groups to time (default {DEFAULT_SAMPLE})")
    ap.add_argument("--min-size", type=int, default=None,
                    help="override dedup.keeper_min_group_size for the sample")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="override dedup.keeper_max_frames")
    ap.add_argument("--seed", type=int, default=20260802)
    args = ap.parse_args()

    cfg = load_config(args.config)
    max_frames = args.max_frames or cfg.dedup.keeper_max_frames
    conn = connect(cfg.database)
    try:
        # The population is always read at min_size 2: the report prices BOTH settings and
        # cannot do that off a list one of them has already trimmed.
        groups = dedup.keeper_groups(conn, cfg.index.phash_max_distance, min_size=2)
    finally:
        conn.close()
    if not groups:
        raise SystemExit("в индексе нет групп почти-дублей — нечего мерить")
    for line in population_lines(groups):
        print(line)

    min_size = args.min_size if args.min_size is not None else cfg.dedup.keeper_min_group_size
    sample = [g for g in groups if len(g) >= min_size]
    random.Random(args.seed).shuffle(sample)
    sample = sample[:max(1, args.groups)]
    print(f"\nмеряю {len(sample)} групп (keeper_min_group_size: {min_size}, "
          f"до {max_frames} кадров в вопросе, модель {cfg.vlm.model})")
    ask = junk.qwen_vlm_keeper(cfg.vlm.model, cfg.vlm.max_edge)
    m = measure(sample, ask, max_frames)
    for line in rate_lines(m):
        print(line)
    print()
    for line in cost_lines(m, groups):
        print(line)


if __name__ == "__main__":
    main()
