"""F164: the two thread ceilings of the junk stage — read from the config, and harmless.

Both ceilings were set by taste and never by a measurement: `_DEFAULT_OCR_WORKERS_CAP`
at 4 "so a weak card is not knocked over" (F73), and `default_vlm_workers()` at
min(4, cores) with 24 cores under it. F164 built the sweeps that price them
(scripts/measure_ocr_workers.py, scripts/measure_vlm_workers.py) and the VLM one came
back saying four is already past the knee — the tables and both verdicts are recorded
next to the values. This file pins what no future sweep may quietly change:

* the number of threads comes from the CONFIG at both ends, is never below one, and
  never exceeds the machine's cores — a default is a function of the hardware, not a
  literal, and this file therefore never spells one;
* a weak machine keeps a weak pool. The whole reason a ceiling exists is the machine
  with four cores and a small card, and no measurement on a 24-core one may hand it
  more threads than it had;
* NOT ONE VERDICT MOVES. The same collection classified with 1, 4, 8 and 12 threads
  gives byte-identical `media_class` rows, at both ends of the stage;
* the deep tier's labels come back in the CANDIDATE ORDER whatever the thread count is
  (the F101 invariant), and no frame is prepared twice — more threads must buy overlap,
  not extra decodes.
"""
from __future__ import annotations

import os
import unittest
import unittest.mock

from sorta import config as config_mod
from sorta import junk as junk_mod
from sorta.config import default_vlm_workers, resolve_vlm_workers
from sorta.junk import _vlm_labels, resolve_ocr_workers
from tests.test_junk_parallel_ocr import Collection, FakeDetectors
from tests.test_junk_vlm_pipeline import Candidates, FakeSplitVlm

# The grid both halves are checked over: 1 (no pool at all), the shipped defaults, and
# the counts the F164 sweeps went up to.
GRID = (1, 2, 4, 6, 8, 12)
# Cores of the machines the defaults must behave on: a netbook, a laptop, the 24-core
# machine the sweeps ran on, and something absurd.
CORE_COUNTS = (1, 2, 4, 8, 12, 16, 24, 64, 128)
# The pool a weak machine had before F164, and must not exceed after it.
WEAK_MACHINE_CORES = 8
WEAK_MACHINE_MAX_WORKERS = 4


class WorkerDefaultCase(unittest.TestCase):
    """Helpers for asking a default what it would do on a machine of N cores."""

    def on_cores(self, resolve, cores: int | None) -> int:
        with unittest.mock.patch.object(os, "cpu_count", return_value=cores):
            return resolve()


class TestVlmWorkerDefault(WorkerDefaultCase):
    """`vlm.workers` when the config does not say — a function of the cores."""

    def default_on(self, cores: int | None) -> int:
        return self.on_cores(default_vlm_workers, cores)

    def test_never_below_one(self):
        for cores in (*CORE_COUNTS, 0, None):
            with self.subTest(cores=cores):
                self.assertGreaterEqual(self.default_on(cores), 1)

    def test_never_more_threads_than_cores(self):
        for cores in CORE_COUNTS:
            with self.subTest(cores=cores):
                self.assertLessEqual(self.default_on(cores), cores)

    def test_never_above_the_ceiling(self):
        for cores in CORE_COUNTS:
            with self.subTest(cores=cores):
                self.assertLessEqual(self.default_on(cores),
                                     config_mod._VLM_WORKERS_CAP)

    def test_a_weak_machine_keeps_the_pool_it_had(self):
        """The point of a ceiling: four cores and a small card get no twelve threads."""
        for cores in (1, 2, 4, WEAK_MACHINE_CORES):
            with self.subTest(cores=cores):
                self.assertLessEqual(self.default_on(cores), WEAK_MACHINE_MAX_WORKERS)

    def test_more_cores_never_mean_fewer_threads(self):
        values = [self.default_on(cores) for cores in CORE_COUNTS]
        self.assertEqual(values, sorted(values))

    def test_a_small_machine_gets_a_thread_per_core_and_no_more(self):
        """The half of "a function of the cores" that a ceiling cannot express."""
        for cores in (1, 2, 3):
            with self.subTest(cores=cores):
                self.assertEqual(self.default_on(cores), cores)


class TestVlmWorkersComeFromTheConfig(unittest.TestCase):
    """The value in use is the config's — the default is only what it falls back to."""

    def test_an_explicit_value_wins_over_the_default(self):
        for workers in GRID:
            with self.subTest(workers=workers):
                self.assertEqual(resolve_vlm_workers({"vlm": {"workers": workers}}),
                                 workers)

    def test_the_ceiling_is_a_default_not_a_limit_on_what_a_user_asks_for(self):
        """A user who has measured their own machine may ask for more — and gets it."""
        big = config_mod._VLM_WORKERS_CAP * 4
        self.assertEqual(resolve_vlm_workers({"vlm": {"workers": big}}), big)

    def test_absent_falls_back_to_the_default_of_this_machine(self):
        self.assertEqual(resolve_vlm_workers({}), default_vlm_workers())


class TestOcrWorkerDefault(WorkerDefaultCase):
    """`naming.ocr_workers` when the config does not say — the same rules."""

    def default_on(self, cores: int | None) -> int:
        return self.on_cores(lambda: resolve_ocr_workers(None), cores)

    def test_never_below_one(self):
        for cores in (*CORE_COUNTS, 0, None):
            with self.subTest(cores=cores):
                self.assertGreaterEqual(self.default_on(cores), 1)

    def test_never_more_threads_than_cores(self):
        for cores in CORE_COUNTS:
            with self.subTest(cores=cores):
                self.assertLessEqual(self.default_on(cores), cores)

    def test_never_above_the_ceiling(self):
        for cores in CORE_COUNTS:
            with self.subTest(cores=cores):
                self.assertLessEqual(self.default_on(cores),
                                     junk_mod._DEFAULT_OCR_WORKERS_CAP)

    def test_a_weak_machine_keeps_the_pool_it_had(self):
        """Every worker holds its own easyocr Reader — that is VRAM, per thread."""
        for cores in (1, 2, 4, WEAK_MACHINE_CORES):
            with self.subTest(cores=cores):
                self.assertLessEqual(self.default_on(cores), WEAK_MACHINE_MAX_WORKERS)

    def test_more_cores_never_mean_fewer_threads(self):
        values = [self.default_on(cores) for cores in CORE_COUNTS]
        self.assertEqual(values, sorted(values))

    def test_an_explicit_value_wins_over_the_default(self):
        for workers in GRID:
            with self.subTest(workers=workers):
                self.assertEqual(
                    resolve_ocr_workers({"naming": {"ocr_workers": workers}}), workers)


class TestLabelsDoNotDependOnTheThreadCount(unittest.TestCase):
    """The F101 invariant, over every thread count F164 considered.

    The delays are deliberately anti-sorted — the first frame is the slowest to prepare
    — so a pipeline that yielded "whatever finished first" would come back in visibly
    the wrong order at every count above one.
    """

    PATHS = [f"/photos/cand_{i}.jpg" for i in range(12)]

    def labels_with(self, workers: int) -> tuple[list[str], list[str]]:
        wheel = ("document", "product", "personal_photo")
        names = [p.rsplit("/", 1)[-1] for p in self.PATHS]
        fake = FakeSplitVlm(
            {name: wheel[i % 3] for i, name in enumerate(names)},
            prepare_delay={name: 0.02 * (len(names) - i)
                           for i, name in enumerate(names)})
        labels = list(_vlm_labels(fake.classifier(), self.PATHS, workers))
        return [str(label) for label in labels], fake.prepared

    def test_the_sequence_is_the_candidate_order_at_every_count(self):
        expected, _prepared = self.labels_with(1)
        for workers in GRID[1:]:
            with self.subTest(workers=workers):
                labels, _ = self.labels_with(workers)
                self.assertEqual(labels, expected)

    def test_no_frame_is_prepared_twice(self):
        """More threads must buy overlap, not extra decodes (the cost that would eat it)."""
        for workers in GRID:
            with self.subTest(workers=workers):
                _labels, prepared = self.labels_with(workers)
                self.assertEqual(sorted(prepared),
                                 sorted(p.rsplit("/", 1)[-1] for p in self.PATHS))


class TestVerdictsDoNotMoveWithTheThreadCount(unittest.TestCase):
    """The main test of F164: the same collection, more threads, identical rows."""

    def vlm_rows(self, workers: int):
        col = Candidates(workers)
        try:
            col.add_files(12)
            fake = FakeSplitVlm(col.labels(),
                                prepare_delay={name: 0.005 * (12 - i)
                                               for i, name in enumerate(col.names)})
            stats = col.run(fake)
            return col.rows(), dict(stats.by_verdict), stats.vlm_applied
        finally:
            col.close()

    def ocr_rows(self, workers: int):
        col = Collection(workers)
        try:
            col.add_files(n_docs=10, n_plain=4, n_faces=2)
            stats = col.run(FakeDetectors(col.fracs()))
            return col.rows(), dict(stats.by_verdict)
        finally:
            col.close()

    def test_the_deep_tier_writes_the_same_rows_at_every_count(self):
        expected = self.vlm_rows(1)
        for workers in GRID[1:]:
            with self.subTest(workers=workers):
                self.assertEqual(self.vlm_rows(workers), expected)

    def test_the_ocr_pool_writes_the_same_rows_at_every_count(self):
        expected = self.ocr_rows(1)
        for workers in GRID[1:]:
            with self.subTest(workers=workers):
                self.assertEqual(self.ocr_rows(workers), expected)
