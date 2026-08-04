"""F132/F186: the keeper of a near-duplicate group — the mechanism, without the question.

F132 asked a model which frame of a burst was the one to keep. F186 retired that question:
measured on 2026-08-04 over 111 groups the owner labelled blind, the model agreed with the
person on 32% of them against 30.4% for picking a frame at random, for 451 seconds of GPU a
run. Nothing replaced it, because at that accuracy there was nothing to buy.

What retired is the QUESTION, and this file is what is left of it — the mechanism, which is
untouched:

* the group key is the group's membership, so a burst that gained or lost a frame is a
  different group and an unchanged one is not;
* the ranking inside a group is sharpness, then resolution, then size — the recommendation
  the Duplicates tab has shown since before any model was asked;
* `group_keeper` stores that recommendation with `source = 'sharpness'`, one row per group,
  overwritten and never duplicated — and a row an older run left behind under a model
  source is still read back exactly as it was written;
* the three `dedup:` keys that outlived the question are still read, and a config.yaml
  that still names the retired one loads without a word (tests/test_config.py states that
  one for all three retired keys at once).

The cases that drove a fake asker through `classify` went with the asker: one question per
group, the max-frames slice, the fallbacks, the population gate and the "nothing is written
to `dedup_choice`" guard. That last one is not lost — `classify` no longer touches either
table, and the tab's own writes are covered by the duplicates suite.
"""
from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path

from sorta import dedup, junk
from sorta.config import load_config
from sorta.db import SCHEMA_VERSION, connect
from tests.schema_history import roll_back_before

# A stored source from before F186: a collection that ran the retired question still
# carries rows like this one, and the interface still reads them (`keeper_source: model`).
LEGACY_MODEL_SOURCE = "vlm#abc12345"


class TestGroupKey(unittest.TestCase):
    """The identity of a group is its membership — that is what invalidates an answer."""

    def test_the_key_does_not_depend_on_the_order_of_the_ids(self):
        self.assertEqual(dedup.group_key([3, 1, 2]), dedup.group_key([1, 2, 3]))

    def test_a_changed_membership_is_a_different_group(self):
        self.assertNotEqual(dedup.group_key([1, 2]), dedup.group_key([1, 2, 3]))
        self.assertNotEqual(dedup.group_key([1, 2]), dedup.group_key([1, 3]))

    def test_ids_are_separated_rather_than_concatenated(self):
        """`1,23` and `12,3` are different groups, and a key must not confuse them."""
        self.assertNotEqual(dedup.group_key([1, 23]), dedup.group_key([12, 3]))


class TestRanking(unittest.TestCase):
    """The recommendation itself: sharpness inside a group, then resolution and size."""

    def frame(self, file_id, sharpness=None, pixels=0, size=0):
        return dedup.GroupFrame(file_id=file_id, path=f"/p/{file_id}.jpg",
                                sharpness=sharpness, pixels=pixels, size=size)

    def test_the_sharpest_frame_comes_first(self):
        ranked = dedup.rank_frames([self.frame(1, 10.0), self.frame(2, 90.0),
                                    self.frame(3, 50.0)])
        self.assertEqual([f.file_id for f in ranked], [2, 3, 1])

    def test_a_partly_measured_group_falls_back_to_resolution(self):
        """A partial comparison would prefer whichever frames happened to be measured."""
        ranked = dedup.rank_frames([self.frame(1, 10.0, pixels=100),
                                    self.frame(2, None, pixels=900)])
        self.assertEqual([f.file_id for f in ranked], [2, 1])

    def test_the_order_is_total_so_two_runs_agree(self):
        frames = [self.frame(7), self.frame(2), self.frame(5)]
        self.assertEqual([f.file_id for f in dedup.rank_frames(frames)], [2, 5, 7])


class TestTheQuestionIsGone(unittest.TestCase):
    """F186: the asker, its prompt and its parser left the stage together.

    Named rather than implied. The retirement is easy to half-do — leaving a prompt behind
    for something to start asking again — and a stage that can no longer be handed an asker
    cannot quietly regain one.
    """

    def test_the_stage_takes_no_keeper_asker(self):
        parameters = inspect.signature(junk.classify).parameters
        self.assertNotIn("keeper_vlm", parameters)
        self.assertNotIn("keeper_vlm_factory", parameters)

    def test_nothing_of_the_question_is_left_in_the_module(self):
        for name in ("keeper_prompt", "parse_keeper_answer", "keeper_source",
                     "keeper_prompt_fingerprint", "vlm_keeper_asker", "qwen_vlm_keeper",
                     "_KEEPER_PROMPT"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(junk, name))

    def test_the_mechanism_it_fed_is_still_there(self):
        """The other half — a guard that passed on an emptied module would be worse than
        none. `group_keeper` is written and read by the same three names it always was."""
        self.assertEqual(dedup.KEEPER_SOURCE_SHARPNESS, "sharpness")
        for name in ("store_group_keeper", "read_group_keepers", "keeper_groups"):
            with self.subTest(name=name):
                self.assertTrue(callable(getattr(dedup, name)))


class TestStorage(unittest.TestCase):
    """The table itself: the migration, the sharpness recommendation, reading it back."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = Path(self.tmp.name) / "test.db"

    def test_a_fresh_db_has_the_table_and_the_version(self):
        conn = connect(self.db)
        self.addCleanup(conn.close)
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(group_keeper)")}
        self.assertEqual(cols, {"group_key", "keeper_id", "source", "updated_at"})
        self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], SCHEMA_VERSION)

    def test_a_db_from_before_the_table_gains_it_and_keeps_its_rows(self):
        conn = connect(self.db)
        conn.execute(
            "INSERT INTO files (path, size, mtime, ext, media_type, indexed_at) "
            "VALUES ('/a.jpg', 1, 0.0, 'jpg', 'photo', 'x')")
        # The whole database goes back, not only this table: a database that predates
        # `group_keeper` predates F140's `frame_quality.junk_score` as well, and leaving
        # that column in place would make its migration add one that already exists.
        roll_back_before(conn, "group_keeper")
        conn.commit()
        conn.close()

        conn = connect(self.db)
        self.addCleanup(conn.close)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM files").fetchone()[0], 1)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM group_keeper").fetchone()[0], 0)
        self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], SCHEMA_VERSION)

    def test_the_sharpness_recommendation_is_what_is_stored_and_read_back(self):
        """The mechanism F186 kept, end to end: rank a group, store its best frame under
        `sharpness`, read it back. This is what the Duplicates tab shows a star for."""
        conn = connect(self.db)
        self.addCleanup(conn.close)
        for name, sharpness in (("blurred.jpg", 10.0), ("sharp.jpg", 90.0)):
            cur = conn.execute(
                """INSERT INTO files (path, size, mtime, ext, media_type, indexed_at)
                   VALUES (?, 1, 0.0, 'jpg', 'photo', 'x')""", (f"/photos/{name}",))
            conn.execute(
                "INSERT INTO frame_quality (file_id, sharpness, source, updated_at) "
                "VALUES (?, ?, 'classic', 'x')", (cur.lastrowid, sharpness))
        conn.commit()
        frames = [dedup.GroupFrame(file_id=r["id"], path=r["path"],
                                   sharpness=r["sharpness"], pixels=0, size=0)
                  for r in conn.execute(
                      """SELECT f.id, f.path, q.sharpness FROM files f
                         JOIN frame_quality q ON q.file_id = f.id ORDER BY f.id""")]
        ranked = dedup.rank_frames(frames)
        key = dedup.group_key([f.file_id for f in frames])
        with conn:
            dedup.store_group_keeper(conn, key, ranked[0].file_id,
                                     dedup.KEEPER_SOURCE_SHARPNESS, "t1")

        stored = dedup.read_group_keepers(conn)[key]
        self.assertEqual(stored.source, dedup.KEEPER_SOURCE_SHARPNESS)
        self.assertEqual(
            conn.execute("SELECT path FROM files WHERE id = ?",
                         (stored.keeper_id,)).fetchone()[0], "/photos/sharp.jpg")

    def test_a_stored_recommendation_is_overwritten_not_duplicated(self):
        conn = connect(self.db)
        self.addCleanup(conn.close)
        conn.execute(
            """INSERT INTO files (path, size, mtime, ext, media_type, indexed_at)
               VALUES ('/a.jpg', 1, 0.0, 'jpg', 'photo', 'x')""")
        (file_id,) = conn.execute("SELECT id FROM files").fetchone()
        key = dedup.group_key([file_id])
        with conn:
            dedup.store_group_keeper(conn, key, file_id, LEGACY_MODEL_SOURCE, "t1")
            dedup.store_group_keeper(conn, key, file_id,
                                     dedup.KEEPER_SOURCE_SHARPNESS, "t2")
        stored = dedup.read_group_keepers(conn)
        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[key].source, dedup.KEEPER_SOURCE_SHARPNESS)
        self.assertEqual(dedup.read_group_keepers(conn, [key]), stored)
        self.assertEqual(dedup.read_group_keepers(conn, ["missing"]), {})

    def test_a_row_from_a_run_that_asked_the_model_is_read_as_it_was_written(self):
        """F186 removed the writer, not the reader: a collection carrying answers of the
        retired question keeps them, and the interface goes on naming their source."""
        conn = connect(self.db)
        self.addCleanup(conn.close)
        conn.execute(
            """INSERT INTO files (path, size, mtime, ext, media_type, indexed_at)
               VALUES ('/a.jpg', 1, 0.0, 'jpg', 'photo', 'x')""")
        (file_id,) = conn.execute("SELECT id FROM files").fetchone()
        key = dedup.group_key([file_id])
        with conn:
            dedup.store_group_keeper(conn, key, file_id, LEGACY_MODEL_SOURCE, "t1")
        self.assertEqual(dedup.read_group_keepers(conn)[key].source, LEGACY_MODEL_SOURCE)


class TestConfigKeys(unittest.TestCase):
    """The two sizes that outlived the question: defaults, and garbage that cannot pass."""

    def load(self, text: str):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "config.yaml"
        path.write_text(text, encoding="utf-8")
        return load_config(path)

    def test_the_defaults(self):
        cfg = self.load("sources: []\n")
        self.assertEqual(cfg.dedup.keeper_max_frames, 5)
        # 3, not 2: on a live collection 85% of groups are pairs, and looking at 73 of
        # them showed the two frames are indistinguishable — 1.44 s of VLM to answer a
        # question with no answer. The measurement is in config.py next to the value.
        self.assertEqual(cfg.dedup.keeper_min_group_size, 3)

    def test_the_values_are_read(self):
        cfg = self.load("dedup:\n  keeper_max_frames: 3\n"
                        "  keeper_min_group_size: 3\n")
        self.assertEqual(cfg.dedup.keeper_max_frames, 3)
        self.assertEqual(cfg.dedup.keeper_min_group_size, 3)

    def test_garbage_numbers_fall_back_to_the_defaults(self):
        cfg = self.load("dedup:\n  keeper_max_frames: 0\n  keeper_min_group_size: nope\n")
        self.assertEqual(cfg.dedup.keeper_max_frames, 5)
        self.assertEqual(cfg.dedup.keeper_min_group_size, 3)

    def test_the_old_key_of_the_section_still_works(self):
        cfg = self.load("dedup:\n  canonical_strategy: largest\n")
        self.assertEqual(cfg.dedup.canonical_strategy, "largest")


if __name__ == "__main__":
    unittest.main()
