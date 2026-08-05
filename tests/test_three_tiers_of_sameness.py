"""F194: the duplicates screen holds three tiers of sameness, with three defaults.

One word covered three populations whose cost of a mistake differs by orders of
magnitude, and the screen applied the same apparatus to all of them — the wrong one, in
the wrong place. Counted on the live collection 2026-08-04:

    exact bytes        12 350 files over 7 631 originals — half the archive
    the same picture      299 groups, 652 frames
    similar frames        791 groups

The defaults follow the cost. Losing the wrong one of two byte-identical files loses
nothing, so that tier is a number. Losing the worse copy of one picture costs a copy, and
"keep the largest" is checkable, so that tier is a recommendation. Losing one of five
similar frames loses a photograph for good, and 111 groups labelled blind by the owner
say no signal we have beats a coin (sharpness 27%, arithmetic 28%, cascade 28%, the model
32%, random 30.4%) — so that tier recommends nothing at all.

The assertions below read the DATA rather than the markup wherever they can: the payload
is what the screen is built from, and `dedup_choice` is what a decision actually is.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from sorta import dedup, ui
from tests.test_ui_dupes import DupesTestBase

# The `group_keeper.source` a run wrote while the retired comparative question was being
# asked (F186). Anything that is not `sharpness` is the model's answer.
MODEL_SOURCE = "vlm#abc12345"

# Four pHashes within one bit of each other — a "similar" group, since the tier is read
# off the hashes and identical ones would make it the second tier instead.
NEAR = ["0" * 16, "0" * 15 + "1", "0" * 15 + "2", "0" * 15 + "4", "0" * 15 + "8"]
SAME = "f" * 16


class TiersTestBase(DupesTestBase):
    def setUp(self):
        super().setUp()
        ui._dupes_cache_clear()

    def tearDown(self):
        ui._dupes_cache_clear()
        super().tearDown()

    # --- fixtures ---------------------------------------------------------------

    def add_exact_copies(self, canonical: int, count: int) -> list[int]:
        """`count` byte-identical copies of one file — rows carrying `dup_of`.

        That is what `dedup.assign_duplicates` writes: one canonical file per blake3
        hash, the rest pointed at it. They are deliberately NOT near-duplicate group
        members — `near_duplicate_groups` excludes them — which is the whole first tier.
        """
        ids = []
        for i in range(count):
            fid = self.add_dupe(f"copy{canonical}_{i}.jpg", phash="a" * 16,
                                width=100, height=100, size=1000)
            self.conn.execute("UPDATE files SET dup_of = ? WHERE id = ?", (canonical, fid))
            ids.append(fid)
        self.conn.commit()
        ui._dupes_cache_clear()
        return ids

    def add_same_picture(self, prefix: str, sizes: list[int]) -> list[int]:
        """One picture in several files: one pHash, different resolutions and weights."""
        return [self.add_dupe(f"{prefix}{i}.jpg", phash=SAME, width=size, height=size,
                              size=size) for i, size in enumerate(sizes)]

    def add_similar(self, prefix: str, count: int, *, size: int = 500,
                    family: str = "0") -> list[int]:
        """A burst: frames that only look alike, each with its own pHash.

        `family` is the leading nibble, and a second burst needs its own — identical
        pHashes are unioned into ONE group by construction, so two bursts built from the
        same family would arrive as a single group rather than as two.
        """
        return [self.add_dupe(f"{prefix}{i}.jpg", phash=family * 15 + NEAR[i][-1],
                              width=size, height=size, size=size)
                for i in range(count)]

    def set_sharpness(self, file_id: int, value: float) -> None:
        self.conn.execute(
            "INSERT INTO frame_quality (file_id, sharpness, source, updated_at)"
            " VALUES (?, ?, 'classic', '2026-08-04T00:00:00')"
            " ON CONFLICT(file_id) DO UPDATE SET sharpness = excluded.sharpness",
            (file_id, value))
        self.conn.commit()
        ui._dupes_cache_clear()

    def store_keeper(self, ids: list[int], keeper_id: int, source: str) -> None:
        with self.conn:
            dedup.store_group_keeper(self.conn, dedup.group_key(ids), keeper_id,
                                     source, "2026-08-04T00:00:00")
        ui._dupes_cache_clear()

    # --- readers ----------------------------------------------------------------

    def payload(self) -> dict:
        status, body, _ctype = self.get("/api/dupes")
        self.assertEqual(status, 200)
        return json.loads(body)

    def group_of(self, file_id: int) -> dict:
        for group in self.payload()["groups"]:
            if any(f["file_id"] == file_id for f in group["frames"]):
                return group
        self.fail(f"file {file_id} is in no near-duplicate group")

    def order_of(self, group: dict) -> list[int]:
        return [f["file_id"] for f in group["frames"]]

    def recommended_ids(self, group: dict) -> list[int]:
        return [f["file_id"] for f in group["frames"] if f["recommended"]]

    def choices(self) -> dict[int, str]:
        return {r["file_id"]: r["action"] for r in
                self.conn.execute("SELECT file_id, action FROM dedup_choice").fetchall()}

    def html(self, lang: str | None = None) -> str:
        _status, body, _ctype = self.get("/" if lang is None else f"/?lang={lang}")
        return body.decode("utf-8")


class TestTheThreeTiersAreDistinguishable(TiersTestBase):
    """The main test: three tiers in one answer, each with its OWN default action."""

    def test_one_answer_names_all_three_and_each_defaults_differently(self):
        canonical = self.add_dupe("orig.jpg", phash="a" * 16, width=100, height=100,
                                  size=1000)
        self.add_exact_copies(canonical, 3)
        same = self.add_same_picture("s", [900, 300])
        similar = self.add_similar("b", 3)
        self.start_server()
        payload = self.payload()

        # Tier 1: a number, and not a single row on the screen.
        self.assertEqual(payload["exact"], {"copies": 3, "originals": 1})
        shown = {f["file_id"] for g in payload["groups"] for f in g["frames"]}
        self.assertEqual(shown, set(same) | set(similar))

        # Tier 2 and tier 3 are named as themselves, and their defaults differ.
        tiers = {g["tier"]: g for g in payload["groups"]}
        self.assertEqual(set(tiers), {"same_image", "similar"})
        self.assertEqual(self.recommended_ids(tiers["same_image"]), [same[0]])
        self.assertEqual(tiers["same_image"]["recommended_by"], "size")
        self.assertEqual(self.recommended_ids(tiers["similar"]), [])
        self.assertIsNone(tiers["similar"]["recommended_by"])

        # And none of the three has decided anything on its own.
        self.assertEqual(self.choices(), {})

    def test_the_tier_is_read_off_the_hashes(self):
        """Same pHash — one picture in two files; different ones — two pictures that
        merely resemble each other. Nothing stores the tier, so nothing can disagree."""
        same = self.add_same_picture("s", [900, 300])
        similar = self.add_similar("b", 2)
        self.start_server()
        self.assertEqual(self.group_of(same[0])["tier"], "same_image")
        self.assertEqual(self.group_of(similar[0])["tier"], "similar")


class TestNothingIsPreselectedAmongSimilarFrames(TiersTestBase):
    """The second-most important test: the interface used to highlight sharpness, which
    the measurement puts BELOW random. A highlighted frame reads as an answer."""

    def test_no_frame_of_a_similar_group_is_marked_in_the_payload(self):
        ids = self.add_similar("b", 5)
        for fid, value in zip(ids, [10.0, 900.0, 40.0, 500.0, 7.0]):
            self.set_sharpness(fid, value)
        self.start_server()
        group = self.group_of(ids[0])
        self.assertEqual(group["tier"], "similar")
        for frame in group["frames"]:
            self.assertFalse(frame["recommended"])
            self.assertIsNone(frame["action"])
        self.assertIsNone(group["recommended_by"])
        # Nothing in the answer names a best frame under any spelling.
        self.assertNotIn("keeper_id", group)
        self.assertNotIn("keeper_source", group)

    def test_the_control_starts_empty_and_can_hold_more_than_one(self):
        """`checked` follows a DECISION or the second tier's rule, and a similar group
        has neither — so every box of it starts empty. A checkbox rather than a radio is
        the other half: "these three of the five" has to be expressible at all."""
        ids = self.add_similar("b", 3)
        self.start_server()
        html = self.html()
        self.assertIn('keepBox.type = "checkbox";', html)
        self.assertIn('keepBox.checked = f.action === "keep" || (!f.action && f.recommended);',
                      html)
        self.assertNotIn('radio.name = "keep-"', html)
        for frame in self.group_of(ids[0])["frames"]:
            self.assertFalse(frame["recommended"])

    def test_the_order_is_offered_as_an_order_and_named_as_one(self):
        """Sharpness ranks the list — it is a fine order and a measured non-answer, and
        the caption says which of the two it is."""
        ids = self.add_similar("b", 3)
        for fid, value in zip(ids, [10.0, 900.0, 40.0]):
            self.set_sharpness(fid, value)
        self.start_server()
        group = self.group_of(ids[0])
        self.assertEqual(group["order"], "sharpness")
        self.assertEqual(self.order_of(group), [ids[1], ids[2], ids[0]])
        self.assertEqual(self.recommended_ids(group), [])

    def test_a_partly_measured_group_is_not_ordered_by_sharpness(self):
        """Half a comparison is not one: after F120 only personal photographs are
        measured, so a mixed group is ordinary and would otherwise be ranked by which
        frames happened to have a number."""
        ids = self.add_similar("b", 3)
        self.set_sharpness(ids[2], 5000.0)
        self.start_server()
        group = self.group_of(ids[0])
        self.assertEqual(group["order"], "size")

    def test_the_captions_say_that_nothing_can_choose(self):
        self.start_server()
        html = self.html("en")
        self.assertIn("None is preselected", html)
        self.assertIn("sorted by sharpness", html)
        # The retired verdicts are gone with the thing they named.
        self.assertNotIn("recommended to keep", html)
        self.assertNotIn("keeper_badge_model", html)


class TestKeepingSeveral(TiersTestBase):
    def test_three_of_five_are_kept_and_exactly_those(self):
        """A burst of five can hold three worth keeping: a portrait with the eyes open,
        another expression, a wide shot. "The best one" throws away two of them."""
        ids = self.add_similar("b", 5)
        self.start_server()
        keeps = [ids[0], ids[2], ids[4]]
        status, payload = self.post("/api/dupes/choices", {"groups": [
            {"group": ids, "keep_file_ids": keeps}]})
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"saved": 1})
        self.assertEqual(self.choices(), {
            ids[0]: "keep", ids[1]: "to_delete", ids[2]: "keep",
            ids[3]: "to_delete", ids[4]: "keep"})

    def test_keeping_nothing_cannot_be_expressed(self):
        """An empty keep list would mean "delete the whole group" — the one sentence
        this route must not be able to say. A group nobody chose in is simply not sent."""
        ids = self.add_similar("b", 3)
        self.start_server()
        status, payload = self.post("/api/dupes/choices", {"groups": [
            {"group": ids, "keep_file_ids": []}]})
        self.assertEqual(status, 400)
        self.assertIn("error", payload)
        self.assertEqual(self.choices(), {})

    def test_a_group_nobody_chose_in_keeps_every_frame(self):
        """The third tier's default, said the only way that cannot go wrong: no rows.

        Two bursts, and the decision in one of them says nothing about the other."""
        ids = self.add_similar("b", 3)
        other = self.add_similar("c", 2, family="e")
        self.start_server()
        self.assertEqual(len(self.payload()["groups"]), 2)
        status, _payload = self.post("/api/dupes/choices", {"groups": [
            {"group": other, "keep_file_ids": [other[0]]}]})
        self.assertEqual(status, 200)
        stored = self.choices()
        self.assertEqual(set(stored), set(other))
        for fid in ids:
            self.assertNotIn(fid, stored)

    def test_a_frame_named_twice_is_one_keeper(self):
        ids = self.add_similar("b", 3)
        self.start_server()
        status, _payload = self.post("/api/dupes/choices", {"groups": [
            {"group": ids, "keep_file_ids": [ids[0], ids[0], ids[1]]}]})
        self.assertEqual(status, 200)
        self.assertEqual(self.choices(),
                         {ids[0]: "keep", ids[1]: "keep", ids[2]: "to_delete"})

    def test_a_keeper_outside_the_group_is_refused_without_a_write(self):
        ids = self.add_similar("b", 3)
        stranger = self.add_same_picture("s", [900, 300])[0]
        self.start_server()
        status, payload = self.post("/api/dupes/choices", {"groups": [
            {"group": ids, "keep_file_ids": [ids[0], stranger]}]})
        self.assertEqual(status, 400)
        self.assertIn("error", payload)
        self.assertEqual(self.choices(), {})

    def test_the_single_keeper_body_still_works(self):
        """`keep_file_id` is the shape the three routes beside this one take, and the
        shape any older client sends. Widening must not break it."""
        ids = self.add_similar("b", 3)
        self.start_server()
        status, payload = self.post("/api/dupes/choices", {"groups": [
            {"group": ids, "keep_file_id": ids[1]}]})
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"saved": 1})
        self.assertEqual(self.choices(),
                         {ids[0]: "to_delete", ids[1]: "keep", ids[2]: "to_delete"})

    def test_the_client_sends_the_list(self):
        self.start_server()
        self.assertIn("keep_file_ids: keeps", self.html())


class TestTheSamePicture(TiersTestBase):
    def test_the_largest_file_is_suggested_and_leads_the_list(self):
        """Resolution and weight are facts, so the rule is applied where it holds —
        which is the tier F148 never reached while it was busy with the one below."""
        ids = self.add_same_picture("s", [900, 600, 300])
        self.start_server()
        group = self.group_of(ids[0])
        self.assertEqual(group["tier"], "same_image")
        self.assertEqual(group["recommended_by"], "size")
        self.assertEqual(self.order_of(group), [ids[0], ids[1], ids[2]])
        self.assertEqual(self.recommended_ids(group), [ids[0]])

    def test_sharpness_does_not_outrank_the_size_rule_here(self):
        """One picture at two scales: the smaller copy can score higher on a laplacian
        without being a better copy of anything. Size is the checkable statement."""
        ids = self.add_same_picture("s", [900, 300])
        self.set_sharpness(ids[0], 10.0)
        self.set_sharpness(ids[1], 5000.0)
        self.start_server()
        self.assertEqual(self.recommended_ids(self.group_of(ids[0])), [ids[0]])

    def test_a_person_can_pick_another_and_it_sticks(self):
        ids = self.add_same_picture("s", [900, 600, 300])
        self.start_server()
        status, _payload = self.post("/api/dupes/choices", {"groups": [
            {"group": ids, "keep_file_ids": [ids[2]]}]})
        self.assertEqual(status, 200)
        expected = {ids[0]: "to_delete", ids[1]: "to_delete", ids[2]: "keep"}
        self.assertEqual(self.choices(), expected)
        group = self.group_of(ids[0])
        self.assertEqual({f["file_id"]: f["action"] for f in group["frames"]}, expected)
        # The suggestion is still shown — it was declined, not withdrawn — and it is
        # still not what the screen acts on.
        self.assertEqual(self.recommended_ids(group), [ids[0]])

    def test_the_badge_names_the_rule(self):
        self.start_server()
        self.assertIn("I18N.dupe_largest_badge", self.html())


class TestExactCopiesAreCollapsedNotDeleted(TiersTestBase):
    def test_the_number_is_named_and_the_screen_stays_free(self):
        canonical = self.add_dupe("orig.jpg", phash="a" * 16, width=100, height=100,
                                  size=1000)
        copies = self.add_exact_copies(canonical, 12)
        self.start_server()
        payload = self.payload()
        self.assertEqual(payload["exact"], {"copies": 12, "originals": 1})
        self.assertEqual(payload["groups"], [])
        shown = {f["file_id"] for g in payload["groups"] for f in g["frames"]}
        self.assertEqual(shown & set(copies), set())

    def test_the_files_are_not_touched(self):
        canonical = self.add_dupe("orig.jpg", phash="a" * 16, width=100, height=100,
                                  size=1000)
        copies = self.add_exact_copies(canonical, 3)
        paths = [r["path"] for r in self.conn.execute(
            "SELECT path FROM files").fetchall()]
        self.start_server()
        self.payload()
        for path in paths:
            with self.subTest(path=path):
                self.assertTrue(Path(path).exists())
        rows = {r["id"] for r in self.conn.execute("SELECT id FROM files").fetchall()}
        self.assertEqual(rows, {canonical, *copies})
        self.assertEqual(self.choices(), {})

    def test_several_originals_are_counted_as_such(self):
        first = self.add_dupe("one.jpg", phash="a" * 16, width=100, height=100, size=1000)
        second = self.add_dupe("two.jpg", phash="b" * 16, width=100, height=100, size=1000)
        self.add_exact_copies(first, 2)
        self.add_exact_copies(second, 3)
        self.start_server()
        self.assertEqual(self.payload()["exact"], {"copies": 5, "originals": 2})

    def test_a_collection_without_copies_says_zero(self):
        self.add_similar("b", 2)
        self.start_server()
        self.assertEqual(self.payload()["exact"], {"copies": 0, "originals": 0})

    def test_the_sentence_says_that_nothing_was_deleted(self):
        self.start_server()
        for lang, expected in (("en", "Nothing was deleted"),
                               ("ru", "Ничего не удалено")):
            with self.subTest(lang=lang):
                self.assertIn(expected, self.html(lang))


class TestGroupKeeperIsAnOrderNow(TiersTestBase):
    def test_the_stored_row_leads_the_order_and_is_called_nothing(self):
        """`group_keeper` is filled by sharpness for free and is worth having as an
        ORDER. What it lost is the role of an answer."""
        ids = self.add_similar("b", 3)
        for fid, value in zip(ids, [900.0, 40.0, 10.0]):
            self.set_sharpness(fid, value)
        self.store_keeper(ids, ids[2], dedup.KEEPER_SOURCE_SHARPNESS)
        self.start_server()
        group = self.group_of(ids[0])
        self.assertEqual(self.order_of(group), [ids[2], ids[0], ids[1]])
        self.assertEqual(self.recommended_ids(group), [])
        self.assertIsNone(group["recommended_by"])
        self.assertEqual(group["order"], "sharpness")

    def test_the_retired_model_answer_is_not_smuggled_back_as_a_position(self):
        """F186 measured that question at a coin toss. Leading the list with it would be
        the same advice given by other means."""
        ids = self.add_similar("b", 3)
        for fid, value in zip(ids, [900.0, 40.0, 10.0]):
            self.set_sharpness(fid, value)
        self.store_keeper(ids, ids[2], MODEL_SOURCE)
        self.start_server()
        group = self.group_of(ids[0])
        self.assertEqual(self.order_of(group), [ids[0], ids[1], ids[2]])
        self.assertNotIn(MODEL_SOURCE, json.dumps(self.payload()))

    def test_the_table_is_still_written_by_the_stage(self):
        """The row keeps being filled — it is the free ranking, and it is still read."""
        self.add_similar("b", 3)
        groups = dedup.keeper_groups(self.conn, min_size=3)
        self.assertEqual(len(groups), 1)
        with self.conn:
            for group in groups:
                dedup.store_group_keeper(
                    self.conn, dedup.group_key([f.file_id for f in group]),
                    group[0].file_id, dedup.KEEPER_SOURCE_SHARPNESS, "2026-08-04")
        stored = dedup.read_group_keepers(self.conn)
        self.assertEqual(len(stored), 1)

    def test_a_row_naming_a_frame_outside_the_group_is_ignored(self):
        """`group_key` is a hash of the membership, so this cannot normally happen — and
        if it ever does, the group keeps its own order instead of losing it."""
        ids = self.add_similar("b", 3)
        for fid, value in zip(ids, [900.0, 40.0, 10.0]):
            self.set_sharpness(fid, value)
        stranger = self.add_same_picture("s", [900, 300])[0]
        self.store_keeper(ids, stranger, dedup.KEEPER_SOURCE_SHARPNESS)
        self.start_server()
        group = self.group_of(ids[0])
        self.assertEqual(self.order_of(group), [ids[0], ids[1], ids[2]])
        self.assertEqual(self.recommended_ids(group), [])

    def test_a_frame_carries_the_same_fields_it_always_did(self):
        """The card is what it was: the tiers changed what is ADVISED, not what a frame
        says about itself."""
        ids = self.add_similar("b", 2)
        self.start_server()
        self.assertEqual(
            set(self.group_of(ids[0])["frames"][0]),
            {"file_id", "name", "thumb_url", "width", "height", "size", "sharpness",
             "recommended", "action", "src_dir", "src_path"})

    def test_no_caption_in_the_product_calls_a_frame_the_best_one(self):
        self.start_server()
        html = self.html("ru")
        for gone in ("рекомендуем оставить", "★ рекомендовано"):
            with self.subTest(caption=gone):
                self.assertNotIn(gone, html)


class TestTheHumansOwnChoiceIsLeftAlone(TiersTestBase):
    def test_opening_the_screen_writes_nothing(self):
        canonical = self.add_dupe("orig.jpg", phash="a" * 16, width=100, height=100,
                                  size=1000)
        self.add_exact_copies(canonical, 4)
        self.add_same_picture("s", [900, 300])
        self.add_similar("b", 3)
        self.start_server()
        self.payload()
        self.payload()
        self.assertEqual(self.choices(), {})

    def test_an_existing_decision_is_not_overwritten_by_the_new_default(self):
        """The second tier suggests the largest file. A person who already kept the
        small one must find their own answer on the screen, not ours."""
        ids = self.add_same_picture("s", [900, 300])
        self.conn.execute(
            "INSERT INTO dedup_choice (file_id, action, updated_at) "
            "VALUES (?, 'keep', '2026-08-01T00:00:00')", (ids[1],))
        self.conn.execute(
            "INSERT INTO dedup_choice (file_id, action, updated_at) "
            "VALUES (?, 'to_delete', '2026-08-01T00:00:00')", (ids[0],))
        self.conn.commit()
        ui._dupes_cache_clear()
        self.start_server()
        for _reload in range(2):
            group = self.group_of(ids[0])
            self.assertEqual({f["file_id"]: f["action"] for f in group["frames"]},
                             {ids[0]: "to_delete", ids[1]: "keep"})
        rows = self.conn.execute(
            "SELECT updated_at FROM dedup_choice").fetchall()
        self.assertEqual({r["updated_at"] for r in rows}, {"2026-08-01T00:00:00"})

    def test_a_decision_in_a_similar_group_survives_untouched(self):
        ids = self.add_similar("b", 3)
        self.conn.execute(
            "INSERT INTO dedup_choice (file_id, action, updated_at) "
            "VALUES (?, 'keep', '2026-08-01T00:00:00')", (ids[2],))
        self.conn.commit()
        ui._dupes_cache_clear()
        self.start_server()
        group = self.group_of(ids[0])
        self.assertEqual({f["file_id"]: f["action"] for f in group["frames"]},
                         {ids[0]: None, ids[1]: None, ids[2]: "keep"})
        self.assertEqual(self.choices(), {ids[2]: "keep"})


class TestTierStrings(TiersTestBase):
    NEW_KEYS = ("dupe_exact_note", "dupe_same_image_note", "dupe_largest_badge",
                "dupe_similar_note", "dupe_order_sharpness", "dupe_order_size",
                "alert_choose_keeper")

    def test_every_caption_is_translated_three_ways(self):
        for key in self.NEW_KEYS:
            with self.subTest(key=key):
                entry = ui._UI_STRINGS[key]
                self.assertEqual(set(entry), {"ru", "en", "ja"})
                for lang in ("ru", "en", "ja"):
                    self.assertTrue(entry[lang].strip())

    def test_the_retired_captions_are_gone_from_the_catalog(self):
        for key in ("keeper_badge_model", "keeper_badge_sharpness", "keeper_badge_hint",
                    "recommended_badge"):
            with self.subTest(key=key):
                self.assertNotIn(key, ui._UI_STRINGS)

    def test_they_reach_the_page_in_each_language(self):
        self.start_server()
        for lang, expected in (("ru", "★ самый большой файл"),
                               ("en", "★ largest file"),
                               ("ja", "★ 最大のファイル")):
            with self.subTest(lang=lang):
                self.assertIn(expected, self.html(lang))


if __name__ == "__main__":
    unittest.main()
