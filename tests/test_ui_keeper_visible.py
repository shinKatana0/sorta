"""F148: the stored keeper recommendation is visible in the Duplicates tab.

`group_keeper` has been filled since F132 and read by nobody: F132 was scoped to the
computation ("showing it is F126"), F126 was scoped to not touching the duplicates path
at all — the one path in the program that deletes files. Both were right, and the
showing fell between them.

What is checked here is mostly what the feature must NOT do. Preselecting the keeper is
a hint one click undoes; preselecting the rest for deletion would be a finished decision
about 37 files in a group of 38, taken on a recommendation about five of them. So the
central test below reads `dedup_choice` after the tab has been opened, not the markup:
the table is the decision, everything else is a suggestion.

F186 retired the model that could name a keeper; the reader did not move. A collection
that ran that question still carries its rows, and they are still shown as the model's —
which is why `MODEL_SOURCE` below is a stored string rather than a call into `junk`. What
a run writes today is the sharpness recommendation, and both captions are still here.
"""
from __future__ import annotations

import json
import unittest

from sorta import dedup, ui
from tests.test_ui_dupes import DupesTestBase

# The `group_keeper.source` a run wrote while the retired question was being asked
# (`vlm#<prompt fingerprint>`). Anything that is not `sharpness` is the model, by the one
# rule ui._dupes_payload applies — see `keeper_source` there.
MODEL_SOURCE = "vlm#abc12345"


class KeeperVisibleTestBase(DupesTestBase):
    def setUp(self):
        super().setUp()
        # The payload is cached per (db path, distance, db fingerprint) — a stale entry
        # from another test would hide the row this one stores.
        ui._dupes_cache_clear()

    def add_group(self, prefix: str, phash: str, sizes: list[int]) -> list[int]:
        """One near-duplicate group; frames differ only in resolution and file size.

        Returned best-first by the tab's own ranking (no sharpness anywhere, so it ranks
        by pixels then size) — which is what makes "the stored row named a DIFFERENT
        frame" easy to say below.
        """
        ids = []
        for i, size in enumerate(sizes):
            ids.append(self.add_dupe(f"{prefix}{i}.jpg", phash=phash,
                                     width=size, height=size, size=size))
        return ids

    def store_keeper(self, ids: list[int], keeper_id: int, source: str) -> None:
        with self.conn:
            dedup.store_group_keeper(self.conn, dedup.group_key(ids), keeper_id,
                                     source, "2026-08-03T00:00:00")
        ui._dupes_cache_clear()

    def groups(self) -> list[dict]:
        status, body, _ctype = self.get("/api/dupes")
        self.assertEqual(status, 200)
        return json.loads(body)

    def group_of(self, file_id: int) -> dict:
        for group in self.groups():
            if any(f["file_id"] == file_id for f in group["frames"]):
                return group
        self.fail(f"file {file_id} is in no near-duplicate group")

    def recommended_ids(self, group: dict) -> list[int]:
        return [f["file_id"] for f in group["frames"] if f["recommended"]]

    def choices(self) -> dict[int, str]:
        return {r["file_id"]: r["action"] for r in
                self.conn.execute("SELECT file_id, action FROM dedup_choice").fetchall()}

    def html(self, lang: str | None = None) -> str:
        _status, body, _ctype = self.get("/" if lang is None else f"/?lang={lang}")
        return body.decode("utf-8")


class TestTheRecommendationIsShown(KeeperVisibleTestBase):
    def test_the_caption_names_the_stored_frame_and_only_it(self):
        """The row wins over the local ranking — otherwise the model's answer is still
        invisible, which is the whole bug."""
        ids = self.add_group("a", "0" * 16, [900, 600, 300])
        self.store_keeper(ids, ids[2], MODEL_SOURCE)  # the SMALLEST frame
        self.start_server()
        group = self.group_of(ids[0])
        self.assertEqual(group["keeper_id"], ids[2])
        self.assertEqual(group["keeper_source"], "model")
        self.assertEqual(self.recommended_ids(group), [ids[2]])

    def test_the_keeper_is_preselected_while_nothing_was_decided(self):
        ids = self.add_group("a", "0" * 16, [900, 600, 300])
        self.store_keeper(ids, ids[1], MODEL_SOURCE)
        self.start_server()
        group = self.group_of(ids[0])
        self.assertEqual(self.recommended_ids(group), [ids[1]])
        for frame in group["frames"]:
            self.assertIsNone(frame["action"])
        # The flag is what checks the radio, and a decision outranks it — the same line
        # the tab has used since U3, now fed by the stored recommendation.
        self.assertIn('radio.checked = f.action === "keep" || (!f.action && f.recommended);',
                      self.html())

    def test_a_human_choice_overrides_the_recommendation_and_survives_a_reload(self):
        ids = self.add_group("a", "0" * 16, [900, 600, 300])
        self.store_keeper(ids, ids[0], MODEL_SOURCE)
        self.start_server()
        status, payload = self.post("/api/dupes/choice",
                                    {"group": ids, "keep_file_id": ids[2]})
        self.assertEqual(status, 200)
        self.assertTrue(payload.get("ok"))
        expected = {ids[0]: "to_delete", ids[1]: "to_delete", ids[2]: "keep"}
        for _reload in range(2):
            group = self.group_of(ids[0])
            self.assertEqual({f["file_id"]: f["action"] for f in group["frames"]},
                             expected)
            self.assertEqual(self.choices(), expected)
            # The advice does not change because it was declined — it is still shown,
            # and it is still not what the tab acts on.
            self.assertEqual(group["keeper_id"], ids[0])

    def test_the_source_is_distinguished(self):
        """Trust in advice depends on who gives it: two sources, two captions."""
        by_model = self.add_group("m", "0" * 16, [900, 600, 300])
        by_sharpness = self.add_group("s", "f" * 16, [900, 600, 300])
        self.store_keeper(by_model, by_model[1], MODEL_SOURCE)
        self.store_keeper(by_sharpness, by_sharpness[0], dedup.KEEPER_SOURCE_SHARPNESS)
        self.start_server()
        self.assertEqual(self.group_of(by_model[0])["keeper_source"], "model")
        self.assertEqual(self.group_of(by_sharpness[0])["keeper_source"], "sharpness")
        # The prompt fingerprint stays out of the interface: it is revision-of-the-
        # question bookkeeping, not something a person can act on.
        self.assertNotIn(MODEL_SOURCE, json.dumps(self.groups()))
        html = self.html()
        self.assertIn('g.keeper_source === "model"', html)
        self.assertIn("I18N.keeper_badge_model", html)
        self.assertIn("I18N.keeper_badge_sharpness", html)

    def test_the_caption_carries_what_the_recommendation_does_not_say(self):
        """There is exactly one per group, and a burst can hold two good moments —
        the hint is the only place that can say so without shouting it."""
        ids = self.add_group("a", "0" * 16, [900, 600, 300])
        self.store_keeper(ids, ids[1], MODEL_SOURCE)
        self.start_server()
        self.assertIn("badge.title = I18N.keeper_badge_hint;", self.html())


class TestNothingIsMarkedWithoutAHand(KeeperVisibleTestBase):
    """The test the feature exists to pass: a recommendation preselects the KEEPER and
    nothing else. Marking the rest would hand the user a finished deletion to confirm."""

    def test_opening_the_tab_marks_nothing_for_deletion(self):
        by_model = self.add_group("m", "0" * 16, [900, 600, 300])
        by_sharpness = self.add_group("s", "f" * 16, [900, 600, 300])
        self.store_keeper(by_model, by_model[2], MODEL_SOURCE)
        self.store_keeper(by_sharpness, by_sharpness[0], dedup.KEEPER_SOURCE_SHARPNESS)
        self.start_server()
        groups = self.groups()
        self.assertEqual(len(groups), 2)
        for group in groups:
            self.assertIsNotNone(group["keeper_source"])
            for frame in group["frames"]:
                self.assertIsNone(frame["action"])
        # The table is the decision — checked in the table, not in the markup.
        self.assertEqual(self.choices(), {})

    def test_the_review_workspace_marks_nothing_either(self):
        """The duplicates slice of the Review tab renders the same payload."""
        ids = self.add_group("a", "0" * 16, [900, 600, 300])
        self.store_keeper(ids, ids[2], MODEL_SOURCE)
        self.start_server()
        status, body, _ctype = self.get("/api/review?slice=dupes")
        self.assertEqual(status, 200)
        payload = json.loads(body)
        self.assertEqual(
            [c["count"] for c in payload["counts"] if c["slice"] == "dupes"], [1])
        # Undecided, because a recommendation is not a decision.
        self.assertEqual(
            [c["count"] for c in payload["pending"] if c["slice"] == "dupes"], [1])
        self.assertEqual(self.choices(), {})


class TestGroupsWithoutARow(KeeperVisibleTestBase):
    def test_a_pair_shows_no_recommendation(self):
        """Pairs are never asked about (`keeper_min_group_size: 3`, measured on 73 of
        them: the two frames are indistinguishable), so they get no row — and a caption
        reading "recommendation: sharpness" would send the user looking for a meaning
        that is not there. Stored through the real gate rather than by hand."""
        pair = self.add_group("p", "0" * 16, [900, 300])
        triple = self.add_group("t", "f" * 16, [900, 600, 300])
        min_size = self.cfg.dedup.keeper_min_group_size
        self.assertEqual(min_size, 3)
        with self.conn:
            for group in dedup.keeper_groups(self.conn, min_size=min_size):
                dedup.store_group_keeper(
                    self.conn, dedup.group_key([f.file_id for f in group]),
                    group[0].file_id, dedup.KEEPER_SOURCE_SHARPNESS, "2026-08-03")
        ui._dupes_cache_clear()
        self.start_server()
        pair_group = self.group_of(pair[0])
        self.assertIsNone(pair_group["keeper_source"])
        self.assertIsNone(pair_group["keeper_id"])
        # It still recommends by the tab's own ranking, exactly as it always has.
        self.assertEqual(self.recommended_ids(pair_group), [pair[0]])
        self.assertEqual(self.group_of(triple[0])["keeper_source"], "sharpness")

    def test_a_group_without_a_row_looks_as_it_did(self):
        ids = self.add_group("a", "0" * 16, [900, 600, 300])
        self.start_server()
        group = self.group_of(ids[0])
        self.assertIsNone(group["keeper_id"])
        self.assertIsNone(group["keeper_source"])
        self.assertEqual(group["recommended_by"], "resolution")
        self.assertEqual(self.recommended_ids(group), [ids[0]])
        self.assertEqual(
            set(group["frames"][0]),
            {"file_id", "name", "thumb_url", "width", "height", "size", "sharpness",
             "recommended", "action", "src_dir", "src_path"})

    def test_a_row_naming_a_frame_outside_the_group_is_ignored(self):
        """`group_key` is a hash of the membership, so this cannot normally happen —
        and if it ever does, the group falls back instead of losing its star."""
        ids = self.add_group("a", "0" * 16, [900, 600, 300])
        other = self.add_dupe("z.jpg", phash="f" * 16, width=10, height=10, size=10)
        self.store_keeper(ids, other, MODEL_SOURCE)
        self.start_server()
        group = self.group_of(ids[0])
        self.assertIsNone(group["keeper_source"])
        self.assertEqual(self.recommended_ids(group), [ids[0]])


class TestKeeperStrings(KeeperVisibleTestBase):
    def test_every_new_string_is_translated_three_ways(self):
        for key in ("keeper_badge_model", "keeper_badge_sharpness", "keeper_badge_hint"):
            with self.subTest(key=key):
                entry = ui._UI_STRINGS[key]
                self.assertEqual(set(entry), {"ru", "en", "ja"})
                for lang in ("ru", "en", "ja"):
                    self.assertTrue(entry[lang].strip())

    def test_the_captions_reach_the_page_in_each_language(self):
        self.start_server()
        for lang, expected in (("ru", "рекомендуем оставить · по модели"),
                               ("en", "recommended to keep · by the model"),
                               ("ja", "残すのがおすすめ · モデルの判断")):
            with self.subTest(lang=lang):
                self.assertIn(expected, self.html(lang))


if __name__ == "__main__":
    unittest.main()
