"""F199: a group of duplicates says which tier it is in, and why it advises or does not.

The owner opened the duplicates screen on 2026-08-05 and saw two outwardly identical
pairs: one carrying "the largest file", the other carrying nothing, with the files
differing in size in both. The behaviour is right — one pHash across the group means one
picture stored twice, where "keep the largest" is a checkable statement about resolution
and weight; several pHashes mean different photographs, where 111 groups labelled blind
put every rule we have (27-32%) against picking at random (30.4%). The basis was simply
never said out loud, and a rule nobody names reads as chance — after which the suggestion
is distrusted in the tier where it holds.

So every group carries the two captions its tier is said with, and the third tier's
caption carries the number, in the form F171 set for the other slices. The assertions
below read the PAYLOAD wherever they can: the caption is a property of the answer, not of
the markup, which is what makes "every group has one" checkable by walking the groups.
"""
from __future__ import annotations

import unittest

from sorta import ui
from sorta.ui.review import _tier_captions
from tests.test_three_tiers_of_sameness import TiersTestBase


class TierCaptionTestBase(TiersTestBase):
    def add_picture_under(self, prefix: str, phash: str, sizes: list[int]) -> list[int]:
        """One picture in several files, under a pHash of its OWN.

        A second same-image group needs one: identical pHashes are unioned into a single
        group by construction, so two such bursts built from one hash would arrive as one.
        """
        return [self.add_dupe(f"{prefix}{i}.jpg", phash=phash, width=size, height=size,
                              size=size) for i, size in enumerate(sizes)]


class TestEveryGroupNamesItsTier(TierCaptionTestBase):
    """The main test: the tier reaches the screen, in words, on every group."""

    def test_one_phash_advises_and_several_phashes_deliberately_do_not(self):
        same = self.add_same_picture("s", [900, 300])
        similar = self.add_similar("b", 3)
        self.start_server()
        tiers = {g["tier"]: g for g in self.payload()["groups"]}
        self.assertEqual(set(tiers), {"same_image", "similar"})

        # One picture in two files: named as that, and the largest one is suggested.
        self.assertEqual(tiers["same_image"]["tier_caption"], "dupe_tier_same_image")
        self.assertEqual(tiers["same_image"]["tier_why"], "dupe_same_image_note")
        self.assertEqual(self.recommended_ids(tiers["same_image"]), [same[0]])
        self.assertEqual(tiers["same_image"]["recommended_by"], "size")

        # Different pictures that resemble each other: named as that, and nothing is
        # advised — the absence is the measurement's answer, and now it is a stated one.
        self.assertEqual(tiers["similar"]["tier_caption"], "dupe_tier_similar")
        self.assertEqual(tiers["similar"]["tier_why"], "dupe_similar_note")
        self.assertEqual(self.recommended_ids(tiers["similar"]), [])
        self.assertIsNone(tiers["similar"]["recommended_by"])
        for frame in tiers["similar"]["frames"]:
            self.assertFalse(frame["recommended"])
        self.assertEqual(set(similar),
                         {f["file_id"] for f in tiers["similar"]["frames"]})

    def test_every_group_of_a_mixed_collection_carries_a_caption(self):
        """By walking the groups, not by naming two of them: the defect was one pair
        explaining itself while the pair below it said nothing."""
        self.add_same_picture("s", [900, 300])
        self.add_picture_under("t", "c" * 16, [800, 400, 200])
        self.add_similar("b", 3)
        self.add_similar("c", 2, family="e")
        self.add_similar("d", 4, family="7")
        self.start_server()
        groups = self.payload()["groups"]
        self.assertEqual(len(groups), 5)
        for group in groups:
            with self.subTest(group=group["group"], tier=group["tier"]):
                self.assertIn(group["tier_caption"], ui._UI_STRINGS)
                self.assertIn(group["tier_why"], ui._UI_STRINGS)
                self.assertTrue(ui._UI_STRINGS[group["tier_caption"]]["en"].strip())

    def test_the_caption_follows_the_tier_and_nothing_else(self):
        """Same weight, different pictures — still the third tier. The owner's question
        was about size, and size is not what the tier is read from."""
        similar = self.add_similar("b", 3, size=500)
        self.start_server()
        group = self.group_of(similar[0])
        self.assertEqual(group["tier_caption"], "dupe_tier_similar")
        self.assertEqual(self.recommended_ids(group), [])

    def test_different_weights_in_one_picture_still_advise(self):
        """The mirror case: three weights, one pHash — the suggestion holds, because it
        is about which copy of ONE picture is the fuller one."""
        ids = self.add_same_picture("s", [900, 600, 300])
        self.start_server()
        group = self.group_of(ids[0])
        self.assertEqual(group["tier_caption"], "dupe_tier_same_image")
        self.assertEqual(self.recommended_ids(group), [ids[0]])

    def test_an_unknown_tier_still_gets_a_line_and_promises_nothing(self):
        """A group with no caption is the defect itself, so the table falls back rather
        than raising — and it falls back to the tier that advises nothing."""
        self.assertEqual(_tier_captions("something_new"),
                         _tier_captions("similar"))
        self.assertEqual(_tier_captions("something_new")["tier_caption"],
                         "dupe_tier_similar")


class TestTheThirdTierNamesItsNumber(TiersTestBase):
    """F171's form: a slice that says what it cannot do says what was measured."""

    MEASURED = ("111", "30.4")

    def test_the_caption_carries_the_measurement(self):
        note = ui._UI_STRINGS["dupe_similar_note"]
        for lang in ("ru", "en", "ja"):
            for number in self.MEASURED:
                with self.subTest(lang=lang, number=number):
                    self.assertIn(number, note[lang])

    def test_it_says_that_no_rule_beat_the_coin(self):
        self.start_server()
        self.assertIn("not one rule beat picking at random", self.html("en"))
        self.assertIn("ни одно правило не обыграло случайный выбор", self.html("ru"))

    def test_it_answers_the_question_about_the_file_size(self):
        """Different weights with no suggestion is exactly what was reported, so the
        caption says the weight is not what the tier follows from."""
        self.start_server()
        self.assertIn("not from the weight", self.html("en"))
        self.assertIn("а не размером", self.html("ru"))

    def test_the_second_tier_still_says_why_it_may_advise(self):
        self.start_server()
        self.assertIn("checkable facts", self.html("en"))
        self.assertIn("разрешение и вес проверяемы", self.html("ru"))


class TestTheFirstTierIsStillANumber(TiersTestBase):
    """F194's line, untouched: byte-identical copies are counted, never listed."""

    def test_exact_copies_are_a_pair_of_numbers_and_no_group(self):
        canonical = self.add_dupe("orig.jpg", phash="a" * 16, width=100, height=100,
                                  size=1000)
        copies = self.add_exact_copies(canonical, 9)
        self.add_similar("b", 2)
        self.start_server()
        payload = self.payload()
        self.assertEqual(payload["exact"], {"copies": 9, "originals": 1})
        shown = {f["file_id"] for g in payload["groups"] for f in g["frames"]}
        self.assertEqual(shown & set(copies), set())
        # And no group wears the first tier's name: it has no list to put a caption on.
        self.assertEqual({g["tier"] for g in payload["groups"]}, {"similar"})

    def test_its_own_line_is_unchanged(self):
        self.start_server()
        for lang, expected in (("en", "Nothing was deleted"),
                               ("ru", "Ничего не удалено")):
            with self.subTest(lang=lang):
                self.assertIn(expected, self.html(lang))


class TestTheCaptionsAreCatalogued(TiersTestBase):
    NEW_KEYS = ("dupe_tier_same_image", "dupe_tier_similar", "dupe_tier_why")

    def test_three_languages_each(self):
        for key in self.NEW_KEYS:
            with self.subTest(key=key):
                entry = ui._UI_STRINGS[key]
                self.assertEqual(set(entry), {"ru", "en", "ja"})
                for lang in ("ru", "en", "ja"):
                    self.assertTrue(entry[lang].strip())

    def test_they_reach_the_page_in_each_language(self):
        self.start_server()
        for lang, expected in (("ru", "Похожие кадры — выбор за вами"),
                               ("en", "Similar frames — the choice is yours"),
                               ("ja", "似ているコマ — 選ぶのはあなたです")):
            with self.subTest(lang=lang):
                self.assertIn(expected, self.html(lang))

    def test_every_key_the_payload_names_exists_in_the_catalog(self):
        """The server hands over caption KEYS, so a typo would be a group with a blank
        line instead of a crash. Every key any tier can name is checked here."""
        for tier in ("same_image", "similar"):
            with self.subTest(tier=tier):
                for key in _tier_captions(tier).values():
                    self.assertIn(key, ui._UI_STRINGS)


class TestTheScreenDrawsWhatTheAnswerSays(TiersTestBase):
    def test_the_line_comes_from_the_server_caption(self):
        self.start_server()
        html = self.html()
        self.assertIn("I18N[g.tier_caption]", html)
        self.assertIn("I18N[g.tier_why]", html)
        # The ternary over `tier` is what let an unrecognised tier wear the third one's
        # words; the caption is named by the answer now.
        self.assertNotIn('g.tier === "same_image"', html)

    def test_the_reasoning_folds_out_behind_the_line(self):
        """One line on the group (F199 requirement 5), the measured paragraph on request
        — the group header is not a place for four sentences."""
        self.start_server()
        html = self.html()
        self.assertIn('details.className = "dupe-tier-why"', html)
        self.assertIn('line.className = "dupe-tier-line"', html)
        self.assertIn("summary.textContent = I18N.dupe_tier_why", html)


if __name__ == "__main__":
    unittest.main()
