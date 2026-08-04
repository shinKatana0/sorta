"""F149: a restored frame does not come back as a duplicate to sort out.

This is the half of the feature that would otherwise be discovered a week later. A
model-processed copy resembles its source BY CONSTRUCTION — that is what it is for — so
the moment it becomes an ordinary indexed file, the next `phash` run sees a pair and puts
it in front of the person who made it. Every run. Forever.

`near_duplicate_groups` is where that is closed, and it is closed by EXCLUSION: a derived
file is not a frame to decide about, because the decision was taken when it was made. The
other option in the brief (put it in a group carrying an answer already) would mean
writing `dedup_choice` from a stage, and that table is the user's own hand alone — which
is the second thing checked here.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sorta.db import connect
from sorta.dedup import near_duplicate_groups


class RestoredDedupTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.conn = connect(Path(self.tmp.name) / "test.db")
        self.addCleanup(self.conn.close)

    def add(self, name: str, phash: str, size: int = 1000) -> int:
        cur = self.conn.execute(
            """INSERT INTO files (path, size, mtime, ext, media_type, phash, indexed_at)
               VALUES (?, ?, 0, 'jpg', 'photo', ?, '2026-01-01')""",
            (f"/src/{name}", size, phash))
        self.conn.commit()
        return int(cur.lastrowid or 0)

    def mark_restored(self, copy_id: int, source_id: int, model: str = "swin") -> None:
        self.conn.execute(
            """INSERT INTO restored_files (file_id, source_file_id, model, created_at)
               VALUES (?, ?, ?, '2026-08-03')""", (copy_id, source_id, model))
        self.conn.commit()

    def group_ids(self) -> list[list[int]]:
        return [[int(r["id"]) for r in group] for group in near_duplicate_groups(self.conn)]


class TestThePairIsNotANewTask(RestoredDedupTestBase):
    def test_a_copy_and_its_original_are_not_a_group(self):
        source = self.add("shot.jpg", "0" * 16, size=2000)
        copy = self.add("shot_restored.jpg", "0" * 16, size=9000)
        self.assertEqual(self.group_ids(), [[copy, source]])  # without the row: a pair

        self.mark_restored(copy, source)

        self.assertEqual(self.group_ids(), [])

    def test_the_original_is_still_compared_with_everything_else(self):
        """Excluding the copy must not take its source out of the duplicates work."""
        source = self.add("shot.jpg", "0" * 16, size=2000)
        burst = self.add("burst.jpg", "0" * 16, size=3000)
        copy = self.add("shot_restored.jpg", "0" * 16, size=9000)
        self.mark_restored(copy, source)

        self.assertEqual(self.group_ids(), [[burst, source]])

    def test_a_copy_of_a_different_model_is_excluded_too(self):
        """The exclusion is about being DERIVED, not about which model derived it."""
        source = self.add("shot.jpg", "0" * 16)
        first = self.add("shot_restored.jpg", "0" * 16)
        second = self.add("shot_restored_1.jpg", "0" * 16)
        self.mark_restored(first, source, model="swin")
        self.mark_restored(second, source, model="other")

        self.assertEqual(self.group_ids(), [])

    def test_nothing_here_marks_anything(self):
        """The alternative resolution would have written a decision; this one writes
        nothing at all — `dedup_choice` is the user's hand alone."""
        source = self.add("shot.jpg", "0" * 16)
        copy = self.add("shot_restored.jpg", "0" * 16)
        self.mark_restored(copy, source)

        near_duplicate_groups(self.conn)

        self.assertIsNone(self.conn.execute("SELECT 1 FROM dedup_choice").fetchone())

    def test_an_ordinary_collection_is_grouped_exactly_as_before(self):
        """The guard against the exclusion quietly emptying the duplicates tab."""
        a = self.add("a.jpg", "0" * 16, size=3000)
        b = self.add("b.jpg", "0" * 16, size=2000)
        self.add("c.jpg", "f" * 16, size=1000)

        self.assertEqual(self.group_ids(), [[a, b]])


if __name__ == "__main__":
    unittest.main()
