"""F164: the fast pass writes its verdicts in ONE transaction — the property, pinned.

The brief that ordered this file suspected the opposite. `junk_write` was billed 19.4 ms
per frame by the F147 phase table, which is what a commit per row looks like, and the
proposed fix was to write in batches. The measurement said the writes are already
batched to the maximum: one transaction around the whole chunk loop, 0.005 ms per row
(scripts/measure_junk_write.py — the table is recorded at junk._MEDIA_CLASS_UPSERT).

So there is nothing to speed up here, and exactly one thing to protect: the property
that made the batching idea unnecessary is also the property that keeps a collection
consistent when a run dies. Half of today's verdicts and half of yesterday's is a
database nobody can reason about — the incrementality marker (`tier`) would call the
first half up to date — and it is precisely what a per-chunk commit would produce.

Nothing here measures anything. These are the invariants the numbers rest on.
"""
from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from sorta.config import Config, _naming_from
from sorta.db import connect
from sorta.junk import classify
from tests.test_junk import NO_OCR, FakeClassifier

# Three chunks of the default `naming.clip.batch_size` (16) and a bit: enough that a
# per-chunk commit would be visible, and that a failure can land in the middle.
FILES = 40
# The CLIP call the run dies on. Two calls per chunk (the junk classes, then the
# document pass over the faceless frames), so this lands inside the second chunk.
FAIL_AT_CALL = 3


class HookedClassifier:
    """A FakeClassifier that runs `hook()` before every CLIP call of the stage.

    Composition rather than a subclass: the hook has to fire on the calls the LOOP
    makes, and wrapping is the only way to be sure nothing else in the mock's own
    behaviour moved.
    """

    def __init__(self, hook) -> None:
        self._inner = FakeClassifier({})
        self._hook = hook
        self.calls = 0

    def __call__(self, image_paths, prompts):
        self.calls += 1
        self._hook(self.calls)
        return self._inner(image_paths, prompts)


class WriteTransactionCase(unittest.TestCase):
    """A collection of plain photographs — the verdict logic is not what is tested."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = Config(sources=[Path(self.tmp.name)],
                          database=Path(self.tmp.name) / "test.db",
                          naming=_naming_from({}))
        self.conn = connect(self.cfg.database)
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(self.conn.close)
        for i in range(FILES):
            self.conn.execute(
                """INSERT INTO files (path, size, mtime, ext, media_type, width, height,
                       camera_make, camera_model, indexed_at)
                   VALUES (?, 1000, 0, 'jpg', 'photo', 4000, 3000, 'Canon', 'EOS',
                           '2026-01-01')""",
                (f"/photos/IMG_{i:04d}.jpg",))
        self.conn.commit()

    def outside_count(self) -> int:
        """media_class as a SECOND connection sees it — i.e. what is committed."""
        conn = sqlite3.connect(f"file:{self.cfg.database}?mode=ro", uri=True)
        try:
            return int(conn.execute("SELECT COUNT(*) FROM media_class").fetchone()[0])
        finally:
            conn.close()

    def verdicts(self) -> dict[str, str]:
        return {Path(r["path"]).name: r["verdict"] for r in self.conn.execute(
            "SELECT f.path, mc.verdict FROM media_class mc"
            " JOIN files f ON f.id = mc.file_id")}


class TestOneTransactionPerPass(WriteTransactionCase):
    """The fast pass commits once, at the end — not per chunk and not per row."""

    def test_nothing_is_visible_from_outside_until_the_pass_is_done(self):
        seen: list[int] = []
        classifier = HookedClassifier(lambda _n: seen.append(self.outside_count()))
        classify(self.cfg, self.conn, classifier=classifier, text_detector=NO_OCR)
        self.assertGreater(classifier.calls, 2,
                           "one chunk only — a per-chunk commit would not show")
        self.assertEqual(set(seen), {0},
                         "a second connection saw rows mid-pass: the stage committed "
                         "before the pass was done")
        self.assertEqual(self.outside_count(), FILES)

    def test_every_frame_of_the_pass_is_written(self):
        classify(self.cfg, self.conn, classifier=FakeClassifier({}), text_detector=NO_OCR)
        self.assertEqual(len(self.verdicts()), FILES)


class TestAnInterruptedPass(WriteTransactionCase):
    """The reason the shape above is worth keeping: an interrupted run writes nothing."""

    def failing_run(self, at_call: int) -> None:
        def hook(call: int) -> None:
            if call >= at_call:
                raise RuntimeError("карта отвалилась посреди прогона")

        with self.assertRaises(RuntimeError):
            classify(self.cfg, self.conn, classifier=HookedClassifier(hook),
                     text_detector=NO_OCR)

    def test_a_failure_mid_pass_leaves_no_half_classified_collection(self):
        self.failing_run(FAIL_AT_CALL)
        self.assertEqual(self.verdicts(), {})
        self.assertEqual(self.outside_count(), 0)

    def test_a_failure_mid_pass_keeps_the_verdicts_of_the_previous_run(self):
        # A heuristics-only run first: every frame gets a verdict of its own tier, and
        # the CLIP run that dies must not leave a mixture of the two.
        classify(self.cfg, self.conn, use_clip=False)
        before = self.verdicts()
        self.assertEqual(len(before), FILES)
        self.failing_run(FAIL_AT_CALL)
        self.assertEqual(self.verdicts(), before)
        rows = self.conn.execute(
            "SELECT DISTINCT tier, source FROM media_class").fetchall()
        self.assertEqual([(r["tier"], r["source"]) for r in rows],
                         [("heuristic", "heuristic")])
