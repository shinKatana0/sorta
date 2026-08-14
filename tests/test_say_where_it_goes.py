"""F174: the action says WHERE the frame goes, instead of «return to the photos».

Two marks the slices offer read as ONE movement to the person making it — "this frame
does not belong in this slice" — and they are not the same movement at all:

* «not an animal» (`manual_pet`) edits a MEMBERSHIP. The frame has been lying in its city
  folder the whole time and stays there; nothing on disk moves, ever;
* «return to the photos» (`manual_overrides`) edits a VERDICT's route. Such a frame is
  NOT in the city layout, and returning it is a real transfer on the next `sort --apply`.

Neither said a word about it, and neither named the folder. The folder is not a guess —
`sorter.destinations` computes it from rows that are already in the database, with the
code that builds the plan. The test that matters most is the first one below: the caption
and the plan cannot be allowed to drift apart, so they are pinned to each other.
"""
from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from sorta import i18n, ui
from sorta.sorter import Destination, destinations, plan_and_sort

from tests import waiting
from tests.test_ui import UiServerTestBase
from tests.test_ui_animals import AnimalsTestBase
from tests.test_ui_junk_buckets import JunkViewTestBase


class DestinationTestBase(UiServerTestBase):
    """The layout answers of one small collection, asked both ways."""

    def plan_folders(self, mode: str = "city") -> dict[int, str]:
        """file_id -> the target DIRECTORY the plan builds, POSIX, relative to dest."""
        with redirect_stdout(io.StringIO()):
            report = plan_and_sort(self.cfg, self.conn, mode, self.root / "out",
                                   apply=False, write_reports=False)
        return {item.file_id: Path(item.target_rel).parent.as_posix()
                for item in report.plan}

    def destinations(self, *file_ids: int, assume: str | None = None,
                     mode: str = "city") -> dict[int, Destination]:
        return destinations(self.cfg, self.conn, mode, list(file_ids), assume)

    def mark_photo(self, file_id: int) -> None:
        self.conn.execute(
            """INSERT INTO manual_overrides (file_id, action, target, updated_at)
               VALUES (?, 'photo', NULL, '2026-08-04')""", (file_id,))
        self.conn.commit()

    def classify(self, file_id: int, verdict: str) -> None:
        self.conn.execute(
            """INSERT INTO media_class (file_id, verdict, source, score, updated_at, tier)
               VALUES (?, ?, 'vlm', NULL, '2026-08-04', 'vlm')""", (file_id, verdict))
        self.conn.commit()


class TestTheCaptionAndThePlanCannotDisagree(DestinationTestBase):
    """The one property the whole feature rests on: the folder in the caption is the
    folder the plan builds. Not a similar one, not one computed the same way — the same
    one, out of the same function."""

    def test_every_frame_gets_the_folder_the_plan_builds(self):
        placed, _p, _c = self.add_photo_file("a.jpg", country="ru", city="Moscow")
        country_only, _p2, _c2 = self.add_photo_file("b.jpg", country="th")
        nowhere, _p3, _c3 = self.add_photo_file("c.jpg")
        folders = self.plan_folders()
        told = self.destinations(placed, country_only, nowhere)
        self.assertEqual({fid: d.folder for fid, d in told.items()}, folders)

    def test_a_returned_frame_is_told_the_folder_the_written_mark_produces(self):
        """The junk view asks about a correction that has NOT been written yet. The
        answer has to match what the plan says once it IS written — that is the whole
        promise of the button."""
        file_id, _p, _c = self.add_photo_file("p.jpg", country="ru", city="Moscow")
        self.classify(file_id, "product")
        promised = self.destinations(file_id, assume="photo")[file_id]
        # the frame is still a product in the database — the plan carries it off
        self.assertEqual(self.plan_folders()[file_id],
                         i18n.folder("products", "en"))
        self.mark_photo(file_id)
        self.assertEqual(promised.folder, self.plan_folders()[file_id])
        self.assertEqual(promised.folder, "Russia/Moscow/2022")

    def test_without_an_assumed_action_the_answer_is_where_the_frame_lies_now(self):
        """What the animals slice asks. A product frame is answered with the service
        folder it is sitting in, because that is the truth about it right now."""
        file_id, _p, _c = self.add_photo_file("p.jpg", country="ru", city="Moscow")
        self.classify(file_id, "product")
        told = self.destinations(file_id)[file_id]
        self.assertEqual(told.folder, i18n.folder("products", "en"))
        self.assertEqual(told.folder, self.plan_folders()[file_id])

    def test_a_hand_placed_frame_keeps_the_folder_the_person_chose(self):
        """`reassign` outranks everything in the plan; a caption that ignored it would
        promise a city to a frame the person had already put somewhere else."""
        file_id, _p, _c = self.add_photo_file("a.jpg", country="ru", city="Moscow")
        self.conn.execute(
            """INSERT INTO manual_overrides (file_id, action, target, updated_at)
               VALUES (?, 'reassign', 'Отпуск/2019', '2026-08-04')""", (file_id,))
        self.conn.commit()
        told = self.destinations(file_id)[file_id]
        self.assertEqual(told.reason, "manual_reassign")
        self.assertEqual(told.folder, self.plan_folders()[file_id])

    def test_nothing_is_written_and_no_file_moves(self):
        file_id, path, content = self.add_photo_file("a.jpg", country="ru", city="Moscow")
        self.destinations(file_id, assume="photo")
        self.assertEqual(self.conn.execute(
            "SELECT COUNT(*) FROM manual_overrides").fetchone()[0], 0)
        self.assertEqual(self.conn.execute(
            "SELECT COUNT(*) FROM moves").fetchone()[0], 0)
        self.assertTrue(path.exists())
        self.assertEqual(path.read_bytes(), content)

    def test_an_unknown_id_is_absent_rather_than_invented(self):
        file_id, _p, _c = self.add_photo_file("a.jpg", country="ru", city="Moscow")
        told = self.destinations(file_id, 9999)
        self.assertEqual(set(told), {file_id})

    def test_no_ids_is_no_query(self):
        self.assertEqual(self.destinations(), {})

    def test_an_unknown_mode_is_refused(self):
        with self.assertRaises(ValueError):
            self.destinations(1, mode="по_настроению")


class TestTheFolderIsNamedHonestly(DestinationTestBase):
    def test_a_frame_without_geodata_goes_to_no_place_and_not_to_a_city(self):
        """The third of the collection the brief is about: 7 662 frames with no place at
        all. Naming a city there would be the one lie this feature exists to remove."""
        with_place, _p, _c = self.add_photo_file("a.jpg", country="ru", city="Moscow")
        nowhere, _p2, _c2 = self.add_photo_file("b.jpg")
        told = self.destinations(with_place, nowhere)
        self.assertEqual(told[nowhere].reason, "no_place")
        self.assertEqual(told[nowhere].folder, "_Unsorted/no_place")
        self.assertNotIn("Moscow", told[nowhere].folder)

    def test_a_country_without_a_city_goes_to_the_country_level(self):
        """F86: the one place signal there is must not be thrown away — and the caption
        must not round it up to a city either."""
        file_id, _p, _c = self.add_photo_file("a.jpg", country="th")
        told = self.destinations(file_id)[file_id]
        self.assertEqual(told.reason, "country_only")
        self.assertEqual(told.folder, "Thailand/2022")

    def test_the_folder_follows_the_language_of_the_layout(self):
        nowhere, _p, _c = self.add_photo_file("a.jpg")
        placed, _p2, _c2 = self.add_photo_file("b.jpg", country="ru", city="Moscow")
        for lang, no_place, city in (
            ("ru", "_Неразобрано/без_места", "Россия/Moscow/2022"),
            ("en", "_Unsorted/no_place", "Russia/Moscow/2022"),
            ("ja", "_未分類/場所不明", "ロシア/Moscow/2022"),
        ):
            with self.subTest(lang=lang):
                self.cfg.raw = {"language": lang}
                told = self.destinations(nowhere, placed)
                self.assertEqual(told[nowhere].folder, no_place)
                self.assertEqual(told[placed].folder, city)


class TestJunkCardsNameTheirDestination(JunkViewTestBase):
    """The bucket is an EXTRACTION from the canon: the frame is not in a city now, and
    returning it moves the file. So the card names the folder before anything is ticked."""

    def test_a_card_carries_the_folder_the_return_leads_to(self):
        file_id = self.add_classified("p.jpg", "product")
        self.start_server()
        item = self.junk()["items"][0]
        self.assertEqual(item["file_id"], file_id)
        self.assertEqual(item["dest"], "Russia/Moscow/2022")
        self.assertEqual(item["dest_reason"], "city")
        self.assertEqual(item["dest_group"], "city")

    def test_a_frame_with_no_place_is_told_so_instead_of_being_given_a_city(self):
        self.add_classified("p.jpg", "product", country=None, city=None)
        self.start_server()
        item = self.junk()["items"][0]
        self.assertEqual(item["dest"], "_Unsorted/no_place")
        self.assertEqual(item["dest_reason"], "no_place")
        self.assertEqual(item["dest_group"], "no_place")

    def test_the_folder_is_the_one_after_the_return_not_the_bucket_it_sits_in(self):
        """The bug this guards against is the obvious implementation: computing the
        destination of the frame AS IT IS, which answers «_Products» — the folder it is
        already in and the one place the button will never leave it in."""
        self.add_classified("p.jpg", "product")
        self.start_server()
        self.assertNotIn(i18n.folder("products", "en"), self.junk()["items"][0]["dest"])

    def test_a_document_names_its_destination_without_a_preview(self):
        """A sensitive class keeps the way back and loses only `thumb_url` (F133) — the
        caption is a folder name, not a rendering of the frame."""
        self.add_classified("d.jpg", "document")
        self.start_server()
        item = self.junk()["items"][0]
        self.assertNotIn("thumb_url", item)
        self.assertEqual(item["dest"], "Russia/Moscow/2022")

    def test_a_returned_frame_keeps_naming_where_it_is_headed(self):
        """The mark is written, the move is not: it happens on the next apply, and until
        then the card has to keep saying where the frame is going."""
        file_id = self.add_classified("p.jpg", "product")
        self.start_server()
        self.post("/api/overrides", {"file_ids": [file_id], "action": "photo"})
        item = self.junk()["items"][0]
        self.assertTrue(item["restored"])
        self.assertEqual(item["dest"], "Russia/Moscow/2022")

    def test_the_page_states_the_spread_and_not_one_folder_of_it(self):
        """What a bulk caption is made of: every card carries its OWN destination, so a
        selection of a dozen can be counted by group instead of being labelled with the
        first folder in it."""
        placed = self.add_classified("p.jpg", "product")
        nowhere = self.add_classified("q.jpg", "product", country=None, city=None)
        country = self.add_classified("r.jpg", "product", country="th", city=None)
        self.start_server()
        groups = {it["file_id"]: it["dest_group"] for it in self.junk()["items"]}
        self.assertEqual(groups, {placed: "city", nowhere: "no_place",
                                  country: "country"})


class TestAnimalCardsSayNothingMoves(AnimalsTestBase):
    """The mirror image: this slice is a VIEW over the canon. The frame is in its city
    folder and stays there — the fear the wording has to answer is «will this delete
    something»."""

    def post(self, path: str, data: object) -> tuple[int, dict]:
        answer = waiting.post_json(f"{self.base_url}{path}", data)
        return answer.status, answer.json()

    def test_the_card_names_the_folder_the_frame_already_lies_in(self):
        file_id, _p, _c = self.add_photo_file("cat.jpg", country="ru", city="Moscow")
        self.mark_animal(file_id)
        self.start_server()
        item = self.animals()["items"][0]
        self.assertEqual(item["dest"], "Russia/Moscow/2022")
        self.assertEqual(item["dest_reason"], "city")

    def test_marking_it_not_an_animal_moves_nothing_and_the_card_still_says_where(self):
        file_id, path, content = self.add_photo_file("cat.jpg", country="ru",
                                                     city="Moscow")
        self.mark_animal(file_id)
        self.start_server()
        status, payload = self.post("/api/animals/mark",
                                    {"file_ids": [file_id], "action": "not_animal"})
        self.assertEqual(status, 200)
        self.assertEqual(payload["items"][0]["dest"], "Russia/Moscow/2022")
        # the promise of the caption, checked rather than described
        self.assertTrue(path.exists())
        self.assertEqual(path.read_bytes(), content)
        self.assertEqual(self.conn.execute(
            "SELECT COUNT(*) FROM moves").fetchone()[0], 0)
        self.assertEqual(
            self.conn.execute("SELECT path FROM files WHERE id = ?",
                              (file_id,)).fetchone()[0], str(path.resolve()))

    def test_an_animal_without_a_place_is_told_the_service_folder_it_lies_in(self):
        file_id, _p, _c = self.add_photo_file("cat.jpg")
        self.mark_animal(file_id)
        self.start_server()
        item = self.animals()["items"][0]
        self.assertEqual(item["dest"], "_Unsorted/no_place")
        self.assertEqual(item["dest_reason"], "no_place")


class TestDestinationGrouping(unittest.TestCase):
    def test_the_reasons_a_bulk_caption_separates(self):
        cases = {"city": "city", "manual_reassign": "city", "country_only": "country",
                 "no_place": "no_place", "low_date": "undated", "downloaded": "undated"}
        for reason, group in cases.items():
            with self.subTest(reason=reason):
                self.assertEqual(
                    ui._destination_json(Destination(1, ("X",), reason))["dest_group"],
                    group)

    def test_an_ungrouped_reason_lands_in_other_rather_than_disappearing(self):
        """A group that silently lost frames would make the counts stop adding up to the
        selection — which is exactly what the caption is for."""
        told = ui._destination_json(Destination(1, ("_Documents",), "document"))
        self.assertEqual(told["dest_group"], "other")

    def test_a_missing_destination_adds_no_fields_at_all(self):
        self.assertEqual(ui._destination_json(None), {})

    def test_the_folder_is_posix_and_holds_no_file_name(self):
        told = Destination(1, ("Russia", "Moscow", "2022"), "city")
        self.assertEqual(told.folder, "Russia/Moscow/2022")


class TestOneNameForOneIntention(UiServerTestBase):
    """Requirement 4 of the brief: if it is one action for the person, it is one name in
    every slice, and the difference is a clarification UNDER it."""

    def setUp(self):
        super().setUp()
        self.html = ui._render_index_html("ru")

    def test_both_slices_offer_the_same_words(self):
        self.assertIn("I18N.slice_return_button", self.html)
        self.assertIn("{{slice_return_button}}", ui._INDEX_HTML_TEMPLATE)
        self.assertNotIn("junk_restore_button", ui._UI_STRINGS)
        self.assertNotIn("animals_mark_not_animal", ui._UI_STRINGS)

    def test_the_junk_card_and_the_animal_card_differ_only_below_the_button(self):
        self.assertIn("destLine(item, I18N.dest_goes_to)", self.html)
        self.assertIn("destLine(item, I18N.dest_stays_in)", self.html)

    def test_the_bulk_caption_counts_the_selection_instead_of_naming_one_folder(self):
        self.assertIn('id="junk-dest-summary"', self.html)
        self.assertIn("destSummary(junkSelectedItems())", self.html)
        self.assertIn("breakdown: destBreakdown(junkSelectedItems())", self.html)

    def test_the_page_derives_no_path_of_its_own(self):
        """The folder comes off the payload. A path built in JS would be a second copy
        of the layout rules and would start disagreeing with the plan."""
        self.assertIn("fmt(template, { folder: item.dest })", self.html)

    def test_nothing_is_applied_by_looking_at_a_caption(self):
        """The marks still pile up and land on `sort --apply` (requirement 5): the junk
        button writes the same override route it always did."""
        self.assertIn('postJson("/api/overrides", { file_ids: ids, action: action })',
                      self.html)
        self.assertIn('postJson("/api/animals/mark"', self.html)

    def test_no_external_resources(self):
        self.assertNotIn("http://", self.html)
        self.assertNotIn("https://", self.html)


class TestNewStringsAreTranslatedThreeWays(unittest.TestCase):
    KEYS = ("slice_return_button", "dest_goes_to", "dest_stays_in", "dest_unknown",
            "dest_why_no_place", "dest_why_country_only", "dest_why_low_date",
            "dest_why_downloaded", "dest_bulk_summary", "dest_bulk_item",
            "dest_group_city", "dest_group_country", "dest_group_no_place",
            "dest_group_undated", "dest_group_other", "junk_restore_confirm",
            "junk_restored_mark")

    def test_every_string_exists_in_all_three_languages(self):
        for key in self.KEYS:
            with self.subTest(key=key):
                entry = ui._UI_STRINGS[key]
                self.assertEqual(set(entry), {"ru", "en", "ja"})
                for lang, value in entry.items():
                    self.assertTrue(value.strip(), f"{key}/{lang} is empty")

    def test_the_placeholders_survive_translation(self):
        placeholders = {
            "dest_goes_to": ("{folder}",),
            "dest_stays_in": ("{folder}",),
            "dest_bulk_summary": ("{n}", "{breakdown}"),
            "dest_bulk_item": ("{n}", "{group}"),
            "junk_restore_confirm": ("{n}", "{breakdown}"),
        }
        for key, expected in placeholders.items():
            for lang in ("ru", "en", "ja"):
                with self.subTest(key=key, lang=lang):
                    for placeholder in expected:
                        self.assertIn(placeholder, ui._UI_STRINGS[key][lang])

    def test_a_group_label_exists_for_every_group_the_server_can_send(self):
        """The label is looked up as `dest_group_<group>` in JS — a group without a key
        would render as a raw English code next to translated ones."""
        for group in set(ui._DEST_GROUPS.values()) | {"other"}:
            with self.subTest(group=group):
                self.assertIn(f"dest_group_{group}", ui._UI_STRINGS)

    def test_the_button_is_rendered_in_the_page_language(self):
        for lang, expected in (("ru", "Вернуть в раскладку"),
                               ("en", "Return to the layout"),
                               ("ja", "振り分けに戻す")):
            with self.subTest(lang=lang):
                self.assertIn(expected, ui._render_index_html(lang))


class TestACaptionNeverBreaksThePage(DestinationTestBase):
    def test_a_failed_computation_costs_the_caption_and_not_the_grid(self):
        """Geo data can be missing and a layout can raise on a config this slice has no
        say over. A grid that 500s because a folder name could not be phrased is worse
        than a grid without the folder name."""
        file_id, _p, _c = self.add_photo_file("a.jpg", country="ru", city="Moscow")
        rows = self.conn.execute("SELECT id FROM files").fetchall()
        with mock.patch.object(ui.common, "destinations",
                               side_effect=ValueError("no geo base")):
            self.assertEqual(ui._destinations_for(self.cfg, self.conn, rows), {})
        self.assertEqual(
            set(ui._destinations_for(self.cfg, self.conn, rows)), {file_id})

    def test_no_rows_is_no_work(self):
        self.assertEqual(ui._destinations_for(self.cfg, self.conn, []), {})


if __name__ == "__main__":
    unittest.main()
