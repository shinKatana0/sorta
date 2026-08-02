"""F132: the keeper measurement — everything about it that is not the model.

The brief makes one thing a condition of acceptance rather than a nicety: the total cost
of the pass may not be named until the SECONDS PER CALL have been measured, because the
number in hand (0.78 s) was measured on a prompt with one image and this question carries
up to five. So the cases below check the report's contract, not its prose:

* the rate is printed before any projection — asserted on the real output order;
* the projection is that rate times the real population, for both values of
  `dedup.keeper_min_group_size`, and it says out loud when the wide population busts the
  30-minute budget;
* a call that fails is still counted and still costs its seconds: a projection that
  dropped the failures would price a model that never fails;
* nothing printed identifies a frame — a near-duplicate group is a burst of one moment,
  and the other measurement scripts hold the same line.

No model and no photo: the asker and the clock are injected.
"""
from __future__ import annotations

import importlib.util
import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from sorta import dedup

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "measure_group_keeper.py"


def _load_script():
    """Import scripts/measure_group_keeper.py — it is a script, not a package module."""
    spec = importlib.util.spec_from_file_location("measure_group_keeper", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


keeper = _load_script()


def group(size: int, first_id: int = 1) -> list[dedup.GroupFrame]:
    """A ranked group of `size` frames, sharpest first — what keeper_groups returns."""
    return [dedup.GroupFrame(file_id=first_id + i, path=f"/photos/burst/{first_id + i}.jpg",
                             sharpness=100.0 - i, pixels=12_000_000, size=3_000_000)
            for i in range(size)]


class FakeClock:
    """A clock that advances by a fixed step on every second reading."""

    def __init__(self, step: float = 2.5):
        self.step = step
        self.now = 0.0
        self.reads = 0

    def __call__(self) -> float:
        self.reads += 1
        if self.reads % 2 == 0:  # the closing read of a call — the call took `step`
            self.now += self.step
        return self.now


class TestMeasure(unittest.TestCase):
    """The measurement itself: what is timed, what is counted, what is skipped."""

    def test_one_call_per_group_timed_at_the_measured_rate(self):
        clock = FakeClock(step=2.5)
        m = keeper.measure([group(3), group(4, 10)], lambda _paths: "1", 5, clock)
        self.assertEqual(m.calls, 2)
        self.assertEqual(m.per_call, 2.5)
        self.assertEqual(m.answered, 2)

    def test_the_question_holds_at_most_max_frames(self):
        seen = []
        keeper.measure([group(9)], lambda paths: seen.append(len(paths)) or "1", 4,
                       FakeClock())
        self.assertEqual(seen, [4])

    def test_a_group_too_small_to_compare_is_not_a_call(self):
        m = keeper.measure([group(1)], lambda _paths: "1", 5, FakeClock())
        self.assertEqual(m.calls, 0)
        self.assertEqual(keeper.rate_lines(m), [
            "замер не состоялся: ни одной группы не удалось спросить"])

    def test_a_failed_call_still_costs_its_seconds(self):
        def boom(_paths):
            raise RuntimeError("out of memory")

        out = io.StringIO()
        with redirect_stdout(out):
            m = keeper.measure([group(3)], boom, 5, FakeClock(step=4.0))
        self.assertEqual(m.calls, 1)
        self.assertEqual(m.per_call, 4.0)
        self.assertEqual(m.answered, 0)

    def test_an_unreadable_answer_is_a_call_but_not_an_answer(self):
        m = keeper.measure([group(3)], lambda _paths: "no idea", 5, FakeClock())
        self.assertEqual((m.calls, m.answered), (1, 0))

    def test_the_moved_count_is_the_answers_that_are_not_the_sharpest_frame(self):
        m = keeper.measure([group(3), group(3, 10)],
                           lambda paths: "1" if "1.jpg" in paths[0] else "3", 5,
                           FakeClock())
        self.assertEqual((m.answered, m.moved), (2, 1))


class TestReport(unittest.TestCase):
    """The order and the content of what the script prints."""

    def population(self):
        return [group(2), group(2, 10), group(3, 20), group(5, 30)]

    def test_the_population_block_names_both_settings(self):
        lines = keeper.population_lines(self.population())
        self.assertIn("групп почти-дублей: 4, кадров в них: 12", lines[0])
        self.assertIn("keeper_min_group_size: 2 -> 4 вызовов", lines)
        self.assertIn("keeper_min_group_size: 3 -> 2 вызовов", lines)

    def test_the_rate_is_printed_before_any_total(self):
        """The acceptance criterion of the brief, asserted on the output order."""
        m = keeper.measure(self.population(), lambda _paths: "1", 5, FakeClock(step=3.0))
        out = io.StringIO()
        with redirect_stdout(out):
            for line in keeper.rate_lines(m):
                print(line)
            for line in keeper.cost_lines(m, self.population()):
                print(line)
        text = out.getvalue()
        self.assertLess(text.index("секунд на вызов"), text.index("цена полного прохода"))

    def test_the_projection_is_the_measured_rate_times_the_population(self):
        m = keeper.measure(self.population(), lambda _paths: "1", 5, FakeClock(step=3.0))
        lines = keeper.cost_lines(m, self.population())
        self.assertIn("  keeper_min_group_size: 2 -> 4 вызовов x 3.00 с = 0.2 мин "
                      "(в бюджет)", lines)
        self.assertIn("  keeper_min_group_size: 3 -> 2 вызовов x 3.00 с = 0.1 мин "
                      "(в бюджет)", lines)

    def test_busting_the_budget_names_the_setting_that_fixes_it(self):
        groups = [group(2, 100 * i) for i in range(1, 800)] + [group(3)]
        m = keeper.measure([group(3)], lambda _paths: "1", 5, FakeClock(step=3.0))
        lines = keeper.cost_lines(m, groups)
        self.assertTrue(any("ДОРОЖЕ БЮДЖЕТА" in line for line in lines), lines)
        self.assertTrue(any("keeper_min_group_size: 3" in line and "сузить" in line
                            for line in lines), lines)

    def test_nothing_printed_identifies_a_frame(self):
        m = keeper.measure(self.population(), lambda _paths: "2", 5, FakeClock())
        text = "\n".join(keeper.population_lines(self.population())
                         + keeper.rate_lines(m) + keeper.cost_lines(m, self.population()))
        for forbidden in ("/photos", ".jpg", "burst"):
            with self.subTest(token=forbidden):
                self.assertNotIn(forbidden, text)


if __name__ == "__main__":
    unittest.main()
