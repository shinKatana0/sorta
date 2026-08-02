"""F132: the best frame of a near-duplicate group — one comparative question, stored as advice.

What the brief promises, and therefore what is asserted below:

* with `dedup.keeper_vlm` off nothing changes at all — no model is built, no call is made,
  and the table stays empty, so the interface keeps recommending by sharpness;
* with it on the model is asked ONCE PER GROUP, not once per frame (counted, because that
  is the whole bet of the feature: a comparative question, answered in one call);
* a group larger than `dedup.keeper_max_frames` sends the best N by sharpness and no more,
  and the answer applies to the group as a whole;
* everything that can fail falls back to the sharpness recommendation and never to an
  empty one — an answer that does not parse, a number outside the group, a model that
  raises on one group, a model that will not build;
* the group key is the group's membership, so a burst that gained or lost a frame is asked
  again and an unchanged one is not;
* `keeper_min_group_size` keeps pairs away from the model;
* and NOTHING is ever written to `dedup_choice`. That case is not optional: it is the line
  between a recommendation and an action, and this feature is on the recommendation side.

No model is loaded anywhere: the classifier, the sharpness detector and the keeper asker
are injected, as everywhere else in the junk suite.
"""
from __future__ import annotations

import tempfile
import unittest
import unittest.mock
from pathlib import Path

from sorta import dedup, junk
from sorta.config import DedupConfig, load_config
from sorta.db import connect
from sorta.junk import classify, keeper_source, parse_keeper_answer
from tests.test_frame_quality import FrameQualityCase
from tests.test_junk import NO_OCR, FakeClassifier

# Group g, member i -> a pHash 2 bits away from its siblings and at least 16 from every
# other group's (the threshold is 5). The whole point of the layout is that the union-find
# in near_duplicate_groups produces exactly the groups the case describes.
_GROUP_BASES = (0x00000000, 0xFFFFFFFF, 0x0000FFFF, 0xFFFF0000)


def phash_of(group: int, member: int) -> str:
    return f"{(_GROUP_BASES[group] << 32) | (1 << member):016x}"


class Keeper:
    """A keeper asker that answers per group and remembers what it was shown.

    Keyed by the basenames of the frames in the question, so a case can both assert WHICH
    frames reached the model and give a different answer per group.
    """

    def __init__(self, answers: dict[str, str] | None = None, default: str = "1",
                 boom: tuple[str, ...] = ()):
        self.answers = answers or {}
        self.default = default
        self.boom = set(boom)
        self.asked: list[tuple[str, ...]] = []

    def __call__(self, paths):
        names = tuple(Path(p).name for p in paths)
        self.asked.append(names)
        for name in names:
            if name in self.boom:
                raise RuntimeError("CUDA error: device-side assert triggered")
        for name in names:
            if name in self.answers:
                return self.answers[name]
        return self.default


class KeeperCase(FrameQualityCase):
    """The fixture of the file: the keeper question on, everything else at its default."""

    def setUp(self):
        super().setUp()
        self.sharpness: dict[str, float] = {}
        self.dedup(keeper_vlm=True)

    def dedup(self, **kwargs):
        # The mechanism is exercised on PAIRS, so the group-size gate is pinned here
        # rather than inherited from the default. The default moved 2 -> 3 once the
        # pairs were looked at (they are indistinguishable, so the question has no
        # answer) — a test of "does the asker get called" must not move with it.
        kwargs.setdefault("keeper_min_group_size", 2)
        self.cfg.dedup = DedupConfig(**kwargs)

    def add_group(self, group: int, sharpness: list[float], prefix: str = "g",
                  screenshot: str | None = None) -> list[str]:
        """One near-duplicate group of `len(sharpness)` frames; returns their names."""
        names = []
        for member, value in enumerate(sharpness):
            name = f"{prefix}{group}_{member}.jpg"
            # A frame the case wants classified as junk gets no camera EXIF: camera make
            # and model are an unconditional veto over the CLIP verdict (brief F13).
            camera = (None, None) if name == screenshot else ("Canon", "EOS")
            self.add_file(name, camera_make=camera[0], camera_model=camera[1],
                          phash=phash_of(group, member))
            self.sharpness[name] = value
            names.append(name)
        self.screenshot = screenshot
        return names

    def run_stage(self, asker=None, **kwargs):
        scores = {self.screenshot: (1, 0.99)} if getattr(self, "screenshot", None) else {}
        clf = FakeClassifier(scores)
        return classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR,
                        sharpness_detector=lambda p: self.sharpness.get(Path(p).name),
                        keeper_vlm=asker, **kwargs)

    def keepers(self) -> dict[str, tuple[str, str]]:
        """Stored recommendations as {group key: (keeper basename, source)}."""
        out = {}
        for row in self.conn.execute(
                """SELECT gk.group_key, f.path, gk.source FROM group_keeper gk
                   JOIN files f ON f.id = gk.keeper_id"""):
            out[row["group_key"]] = (Path(row["path"]).name, row["source"])
        return out

    def keeper_of(self, group: int) -> tuple[str, str]:
        """The recommendation of the one group in the case, by its frames' names."""
        stored = list(self.keepers().values())
        self.assertEqual(len(stored), 1, stored)
        return stored[0]


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


class TestAnswerParsing(unittest.TestCase):
    """Read leniently, and refuse what cannot be a frame of this group."""

    def test_a_bare_number(self):
        self.assertEqual(parse_keeper_answer("2", 5), 2)
        self.assertEqual(parse_keeper_answer(" 3 ", 5), 3)

    def test_prose_around_the_number_still_parses(self):
        self.assertEqual(parse_keeper_answer("Photo 4 is the best one.", 5), 4)
        self.assertEqual(parse_keeper_answer("I would keep #2", 5), 2)

    def test_the_first_number_inside_the_group_wins(self):
        """A model that names two frames is comparing them; its own pick comes first."""
        self.assertEqual(parse_keeper_answer("2 is better than 3", 5), 2)

    def test_a_number_outside_the_group_is_not_a_choice(self):
        self.assertIsNone(parse_keeper_answer("6", 5))
        self.assertIsNone(parse_keeper_answer("0", 5))
        self.assertEqual(parse_keeper_answer("photo 9, I mean 3", 5), 3)

    def test_an_ordinal_word_is_accepted_when_no_number_is(self):
        self.assertEqual(parse_keeper_answer("the second one", 5), 2)
        self.assertEqual(parse_keeper_answer("First.", 5), 1)

    def test_cardinals_are_not_ordinals(self):
        """`the best one` is not a vote for frame 1 — that is why only ordinals parse."""
        self.assertIsNone(parse_keeper_answer("the best one", 5))
        self.assertIsNone(parse_keeper_answer("keep one of them", 5))

    def test_an_unreadable_answer_is_none(self):
        for answer in ("", "I cannot help with that", "все хороши"):
            with self.subTest(answer=answer):
                self.assertIsNone(parse_keeper_answer(answer, 5))

    def test_an_ordinal_past_the_group_is_not_a_choice(self):
        self.assertIsNone(parse_keeper_answer("the fourth one", 3))


class TestRanking(unittest.TestCase):
    """The fallback recommendation: sharpness inside a group, then resolution and size."""

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


class TestToggleOff(KeeperCase):
    """Brief test 1: with the toggle off the run is what it was, in every respect."""

    def setUp(self):
        super().setUp()
        self.dedup(keeper_vlm=False)

    def test_nothing_is_asked_and_nothing_is_stored(self):
        def never(_paths):
            raise AssertionError("no group may reach the model with keeper_vlm off")

        self.add_group(0, [10.0, 90.0])
        self.run_stage(asker=never)
        self.assertEqual(self.keepers(), {})

    def test_no_model_is_built_either(self):
        def factory(_model):
            raise AssertionError("no model may be built with keeper_vlm off")

        self.add_group(0, [10.0, 90.0])
        stats = self.run_stage(keeper_vlm_factory=factory)
        self.assertEqual(stats.keeper_groups, 0)


class TestOneCallPerGroup(KeeperCase):
    """Brief test 2: one question per group — not one per frame."""

    def test_a_group_of_four_costs_one_call(self):
        self.add_group(0, [10.0, 20.0, 30.0, 40.0])
        asker = Keeper(default="1")
        stats = self.run_stage(asker=asker)
        self.assertEqual(len(asker.asked), 1)
        self.assertEqual(len(asker.asked[0]), 4)
        self.assertEqual((stats.keeper_groups, stats.keeper_asked,
                          stats.keeper_answered), (1, 1, 1))

    def test_two_groups_cost_two_calls(self):
        self.add_group(0, [10.0, 20.0])
        self.add_group(1, [30.0, 40.0])
        asker = Keeper(default="1")
        stats = self.run_stage(asker=asker)
        self.assertEqual(len(asker.asked), 2)
        self.assertEqual(stats.keeper_groups, 2)
        self.assertEqual(len(self.keepers()), 2)

    def test_the_answer_picks_the_frame_at_that_position(self):
        self.add_group(0, [10.0, 90.0])  # the second file is the sharper one
        self.run_stage(asker=Keeper(default="2"))
        # the question is ordered by sharpness, so position 2 is the BLURRED frame
        self.assertEqual(self.keeper_of(0), ("g0_0.jpg", keeper_source()))


class TestMaxFrames(KeeperCase):
    """Brief test 3: the best N by sharpness go into the question, the rest do not."""

    def setUp(self):
        super().setUp()
        self.dedup(keeper_vlm=True, keeper_max_frames=2)

    def test_only_the_sharpest_n_are_shown_and_the_answer_holds_for_the_group(self):
        self.add_group(0, [10.0, 20.0, 30.0, 40.0])
        asker = Keeper(default="2")
        stats = self.run_stage(asker=asker)
        self.assertEqual(asker.asked, [("g0_3.jpg", "g0_2.jpg")])
        # the second frame shown is the one with sharpness 30 — and the group as a whole
        # now recommends it, the two frames never shown included
        self.assertEqual(self.keeper_of(0), ("g0_2.jpg", keeper_source()))
        self.assertEqual(stats.keeper_answered, 1)

    def test_a_question_of_one_frame_is_no_question_at_all(self):
        """`keeper_max_frames: 1` leaves nothing to compare — the ranking stands."""
        self.dedup(keeper_vlm=True, keeper_max_frames=1)
        self.add_group(0, [10.0, 90.0])
        asker = Keeper()
        self.run_stage(asker=asker)
        self.assertEqual(asker.asked, [])
        self.assertEqual(self.keeper_of(0), ("g0_1.jpg", dedup.KEEPER_SOURCE_SHARPNESS))


class TestFallbacks(KeeperCase):
    """Brief tests 4 and 5: a refusal costs the answer, never the recommendation."""

    def test_an_unreadable_answer_leaves_the_sharpness_recommendation(self):
        self.add_group(0, [10.0, 90.0])
        stats = self.run_stage(asker=Keeper(default="I cannot help with that"))
        self.assertEqual(self.keeper_of(0), ("g0_1.jpg", dedup.KEEPER_SOURCE_SHARPNESS))
        self.assertEqual((stats.keeper_asked, stats.keeper_answered), (1, 0))

    def test_a_number_outside_the_group_is_not_applied(self):
        self.add_group(0, [10.0, 90.0])
        self.run_stage(asker=Keeper(default="7"))
        self.assertEqual(self.keeper_of(0), ("g0_1.jpg", dedup.KEEPER_SOURCE_SHARPNESS))

    def test_a_model_that_raises_on_one_group_costs_only_that_group(self):
        self.add_group(0, [10.0, 90.0])
        self.add_group(1, [20.0, 80.0], prefix="h")
        asker = Keeper(default="1", boom=("g0_1.jpg",))
        stats = self.run_stage(asker=asker)
        stored = self.keepers()
        self.assertEqual(len(stored), 2)
        sources = sorted(source for _name, source in stored.values())
        self.assertEqual(sources, [dedup.KEEPER_SOURCE_SHARPNESS, keeper_source()])
        self.assertEqual(stats.keeper_answered, 1)

    def test_a_model_that_will_not_build_leaves_the_stage_running(self):
        def factory(_model):
            raise RuntimeError("no transformers")

        self.add_group(0, [10.0, 90.0])
        stats = self.run_stage(keeper_vlm_factory=factory)
        self.assertEqual(self.keepers(), {})
        self.assertEqual(stats.quality_rows, 2)  # the cheap tiers did their work

    def test_a_collection_without_groups_asks_nothing(self):
        self.add_file("alone.jpg", phash=phash_of(0, 0))
        self.sharpness["alone.jpg"] = 50.0
        asker = Keeper()
        stats = self.run_stage(asker=asker)
        self.assertEqual(asker.asked, [])
        self.assertEqual(stats.keeper_groups, 0)


class TestIncrementality(KeeperCase):
    """Brief tests 6 and 7: the membership is the key, so it invalidates itself."""

    def test_an_unchanged_group_is_not_asked_twice(self):
        self.add_group(0, [10.0, 90.0])
        asker = Keeper(default="1")
        self.run_stage(asker=asker)
        self.assertEqual(len(asker.asked), 1)
        self.run_stage(asker=asker)
        self.assertEqual(len(asker.asked), 1)
        self.assertEqual(len(self.keepers()), 1)

    def test_a_group_that_gained_a_frame_is_asked_again(self):
        self.add_group(0, [10.0, 90.0])
        asker = Keeper(default="1")
        self.run_stage(asker=asker)
        self.add_file("g0_2.jpg", phash=phash_of(0, 2))
        self.sharpness["g0_2.jpg"] = 50.0
        self.run_stage(asker=asker)
        self.assertEqual(len(asker.asked), 2)
        self.assertEqual(len(asker.asked[1]), 3)
        # the answer of the old group is not reused, and does not linger as a second row
        self.assertEqual(len(self.keepers()), 2)

    def test_editing_the_question_invalidates_the_answers_it_produced(self):
        self.add_group(0, [10.0, 90.0])
        asker = Keeper(default="1")
        self.run_stage(asker=asker)
        with unittest.mock.patch.object(junk, "_KEEPER_PROMPT",
                                        junk._KEEPER_PROMPT + " Please."):
            self.run_stage(asker=asker)
        self.assertEqual(len(asker.asked), 2)

    def test_the_pass_runs_when_nothing_else_in_the_stage_has_work(self):
        """Switching the toggle on for an already-classified collection must ask.

        This is the ordinary way the feature is switched on, and the stage returns early
        when no frame needs classifying — so the keeper half has to run on that path too.
        """
        self.dedup(keeper_vlm=False)
        self.add_group(0, [10.0, 90.0])
        self.run_stage()
        self.dedup(keeper_vlm=True)
        asker = Keeper(default="1")
        stats = self.run_stage(asker=asker)
        self.assertEqual(stats.processed, 0)  # nothing to reclassify
        self.assertEqual(len(asker.asked), 1)
        self.assertEqual(self.keeper_of(0), ("g0_1.jpg", keeper_source()))


class TestPopulation(KeeperCase):
    """Brief test 8, and the classes a model is never shown."""

    def test_min_group_size_keeps_pairs_away_from_the_model(self):
        self.dedup(keeper_vlm=True, keeper_min_group_size=3)
        self.add_group(0, [10.0, 90.0])              # a pair
        self.add_group(1, [10.0, 50.0, 90.0], prefix="h")
        asker = Keeper(default="1")
        stats = self.run_stage(asker=asker)
        self.assertEqual([len(a) for a in asker.asked], [3])
        self.assertEqual(stats.keeper_groups, 1)
        # and the pair gets no row at all: its recommendation is the interface's own
        stored = self.keepers()
        self.assertEqual(len(stored), 1)
        self.assertEqual(list(stored.values())[0][0], "h1_2.jpg")

    def test_a_group_holding_something_that_is_not_a_photograph_is_not_shown(self):
        """The F120 rule, and here also the privacy one: no group of scans goes to a model."""
        self.add_group(0, [10.0, 90.0], screenshot="g0_1.jpg")
        asker = Keeper(default="1")
        stats = self.run_stage(asker=asker)
        self.assertEqual(asker.asked, [])
        self.assertEqual((stats.keeper_groups, stats.keeper_asked), (1, 0))
        # a recommendation is still stored — a refusal must not be an empty screen
        self.assertEqual(self.keeper_of(0)[1], dedup.KEEPER_SOURCE_SHARPNESS)


class TestAdviceOnly(KeeperCase):
    """Brief test 9: the line between advice and action. Not optional.

    `dedup_choice` is what the sorter moves files by and what the trash button reads. The
    keeper pass must never put a row there, whatever the model answers.
    """

    def test_nothing_is_written_to_dedup_choice(self):
        self.add_group(0, [10.0, 90.0])
        self.add_group(1, [20.0, 80.0], prefix="h")
        self.run_stage(asker=Keeper(default="2"))
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM dedup_choice").fetchone()[0], 0)
        self.assertEqual(len(self.keepers()), 2)

    def test_a_users_own_choice_is_left_alone(self):
        names = self.add_group(0, [10.0, 90.0])
        self.run_stage(asker=Keeper(default="1"))
        blurred = self.conn.execute(
            "SELECT id FROM files WHERE path = ?", (f"/photos/{names[0]}",)).fetchone()[0]
        self.conn.execute(
            "INSERT INTO dedup_choice (file_id, action, updated_at) VALUES (?, 'keep', 'x')",
            (blurred,))
        self.conn.commit()
        # a second run may re-ask, but it may not touch the decision
        self.add_file("g0_2.jpg", phash=phash_of(0, 2))
        self.sharpness["g0_2.jpg"] = 50.0
        self.run_stage(asker=Keeper(default="1"))
        rows = self.conn.execute("SELECT file_id, action FROM dedup_choice").fetchall()
        self.assertEqual([(r["file_id"], r["action"]) for r in rows], [(blurred, "keep")])


class TestStorage(unittest.TestCase):
    """The table itself: the migration, and reading answers back."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = Path(self.tmp.name) / "test.db"

    def test_a_fresh_db_has_the_table_and_the_version(self):
        conn = connect(self.db)
        self.addCleanup(conn.close)
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(group_keeper)")}
        self.assertEqual(cols, {"group_key", "keeper_id", "source", "updated_at"})
        self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], 21)

    def test_a_v17_db_gains_the_table_and_keeps_its_rows(self):
        conn = connect(self.db)
        conn.execute(
            "INSERT INTO files (path, size, mtime, ext, media_type, indexed_at) "
            "VALUES ('/a.jpg', 1, 0.0, 'jpg', 'photo', 'x')")
        conn.execute("DROP TABLE group_keeper")
        # A real v17 DB predates F140's column too (v20) — leaving it in place would make
        # that migration add a column that already exists and raise.
        conn.execute("ALTER TABLE frame_quality DROP COLUMN junk_score")
        conn.execute("PRAGMA user_version = 17")
        conn.commit()
        conn.close()

        conn = connect(self.db)
        self.addCleanup(conn.close)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM files").fetchone()[0], 1)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM group_keeper").fetchone()[0], 0)
        self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], 21)

    def test_a_stored_recommendation_is_overwritten_not_duplicated(self):
        conn = connect(self.db)
        self.addCleanup(conn.close)
        conn.execute(
            """INSERT INTO files (path, size, mtime, ext, media_type, indexed_at)
               VALUES ('/a.jpg', 1, 0.0, 'jpg', 'photo', 'x')""")
        (file_id,) = conn.execute("SELECT id FROM files").fetchone()
        key = dedup.group_key([file_id])
        with conn:
            dedup.store_group_keeper(conn, key, file_id, "sharpness", "t1")
            dedup.store_group_keeper(conn, key, file_id, keeper_source(), "t2")
        stored = dedup.read_group_keepers(conn)
        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[key].source, keeper_source())
        self.assertEqual(dedup.read_group_keepers(conn, [key]), stored)
        self.assertEqual(dedup.read_group_keepers(conn, ["missing"]), {})


class TestConfigKeys(unittest.TestCase):
    """The three keys: defaults, and a typo that must not switch a model on."""

    def load(self, text: str):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "config.yaml"
        path.write_text(text, encoding="utf-8")
        return load_config(path)

    def test_the_defaults(self):
        cfg = self.load("sources: []\n")
        self.assertFalse(cfg.dedup.keeper_vlm)
        self.assertEqual(cfg.dedup.keeper_max_frames, 5)
        # 3, not 2: on a live collection 85% of groups are pairs, and looking at 73 of
        # them showed the two frames are indistinguishable — 1.44 s of VLM to answer a
        # question with no answer. The measurement is in config.py next to the value.
        self.assertEqual(cfg.dedup.keeper_min_group_size, 3)

    def test_the_values_are_read(self):
        cfg = self.load("dedup:\n  keeper_vlm: true\n  keeper_max_frames: 3\n"
                        "  keeper_min_group_size: 3\n")
        self.assertTrue(cfg.dedup.keeper_vlm)
        self.assertEqual(cfg.dedup.keeper_max_frames, 3)
        self.assertEqual(cfg.dedup.keeper_min_group_size, 3)

    def test_a_quoted_false_does_not_switch_the_model_on(self):
        cfg = self.load('dedup:\n  keeper_vlm: "false"\n')
        self.assertFalse(cfg.dedup.keeper_vlm)

    def test_garbage_numbers_fall_back_to_the_defaults(self):
        cfg = self.load("dedup:\n  keeper_max_frames: 0\n  keeper_min_group_size: nope\n")
        self.assertEqual(cfg.dedup.keeper_max_frames, 5)
        self.assertEqual(cfg.dedup.keeper_min_group_size, 3)

    def test_the_old_key_of_the_section_still_works(self):
        cfg = self.load("dedup:\n  canonical_strategy: largest\n")
        self.assertEqual(cfg.dedup.canonical_strategy, "largest")


if __name__ == "__main__":
    unittest.main()
