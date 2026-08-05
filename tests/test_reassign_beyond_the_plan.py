"""F203: a reassign target may be a folder the plan does not contain.

Before this, `POST /api/overrides` with `reassign` took a folder of the CURRENT plan —
the picker offered what the program had already proposed, and there was no way to say
«Россия/» (the country root, a branch the layout has had since F86) or to name a
directory that does not exist yet. The value was also stored unchecked and dropped hours
later, at apply time, into a log nobody reads.

So the two halves tested here are: the typed name reaches the plan and the disk, and a
name that is a PATH rather than a name comes back refused with a reason. The refusal is
checked by a request straight at the route — the field in the page is not the boundary,
the endpoint is.

The cleaning rule is deliberately ONE function (`sorter.manual_target_parts`, built on
`sorter._sanitize`), and one case here pins that: the folder the web app accepts and the
folder the layout writes cannot be spelled differently.
"""
from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from sorta import sorter, ui
from sorta.sorter import _sanitize, manual_target_parts, plan_and_sort

from tests.test_sorter_overrides import OverridesTestBase as SorterOverridesTestBase
from tests.test_ui_overrides import OverridesTestBase as UiOverridesTestBase


class RouteTestBase(UiOverridesTestBase):
    """The route half: the correction travels through HTTP, the layout runs after it.

    The same connection and the same files on disk, so a POST and the plan built from it
    are about one collection — which is the property the feature is: what the user typed
    is what the dry-run shows and what the apply writes.
    """

    def setUp(self):
        super().setUp()
        self.dest = self.root / "dest"

    def reassign(self, file_ids: list[int], target: str) -> tuple[int, dict]:
        return self.post("/api/overrides",
                         {"file_ids": file_ids, "action": "reassign", "target": target})

    def plan(self, apply: bool = False):
        with redirect_stdout(io.StringIO()):
            return plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=apply,
                                 write_reports=False)


class TestATargetThePlanLacksIsAccepted(RouteTestBase):
    def test_a_folder_outside_the_plan_is_stored_as_typed(self):
        fid, _p, _c = self.add_photo_file("a.jpg", country="ru", city="Moscow")
        self.start_server()
        status, payload = self.reassign([fid], "Поездки/Карелия 2019")
        self.assertEqual(status, 200)
        self.assertTrue(payload["ok"])
        self.assertEqual(self.rows(), {fid: ("reassign", "Поездки/Карелия 2019")})

    def test_the_new_folder_is_visible_in_the_dry_run(self):
        fid, _p, _c = self.add_photo_file("a.jpg", country="ru", city="Moscow")
        self.start_server()
        self.reassign([fid], "Поездки/Карелия 2019")
        item = self.plan().plan[0]
        self.assertEqual(item.target_rel, "Поездки/Карелия 2019/a.jpg")
        self.assertEqual(item.reason, "manual_reassign")

    def test_nothing_is_created_on_disk_before_the_apply(self):
        fid, _p, _c = self.add_photo_file("a.jpg", country="ru", city="Moscow")
        self.start_server()
        self.reassign([fid], "Поездки/Карелия 2019")
        self.plan()
        self.assertFalse(self.dest.exists())

    def test_the_apply_puts_the_file_into_the_folder_that_did_not_exist(self):
        fid, _p, _c = self.add_photo_file("a.jpg", country="ru", city="Moscow")
        self.start_server()
        self.reassign([fid], "Поездки/Карелия 2019")
        report = self.plan(apply=True)
        self.assertEqual(report.moved, 1)
        self.assertTrue((self.dest / "Поездки" / "Карелия 2019" / "a.jpg").is_file())

    def test_the_country_root_is_a_legal_target_through_the_route(self):
        # The one shape the picker could never offer: a country with no city and no year
        # under it. `country_only` exists in the layout — it just could not be ASKED for.
        fid, _p, _c = self.add_photo_file("a.jpg", country="ru", city="Moscow")
        self.start_server()
        status, _payload = self.reassign([fid], "Россия")
        self.assertEqual(status, 200)
        self.assertEqual(self.plan().plan[0].target_rel, "Россия/a.jpg")

    def test_the_plan_folders_stay_available_as_targets(self):
        # F203 adds an input, it does not take the suggestions away: naming a folder the
        # plan already has must keep working exactly as it did.
        moved, _p, _c = self.add_photo_file("a.jpg", country="ru", city="Moscow")
        self.add_photo_file("b.jpg", country="fr", city="Paris")
        self.start_server()
        _status, body, _ctype = self.get("/api/plan?mode=city")
        categories = [row["category"] for row in json.loads(body)["categories"]]
        existing = [c for c in categories if "Paris" in c or "Париж" in c][0]
        self.assertEqual(self.reassign([moved], existing)[0], 200)
        targets = {it.file_id: it.target_rel for it in self.plan().plan}
        self.assertEqual(targets[moved], f"{existing}/a.jpg")


class TestATargetThatLeavesTheRootIsRefused(RouteTestBase):
    """Requested straight at the endpoint: the page's field is not the boundary.

    Every one of these used to be answered 200 and written into `manual_overrides`. The
    sorter still refuses them when it reads the row (a database file is editable by other
    means), but the person who typed one heard nothing back until the layout ran.
    """

    CASES = {
        "..": sorter.TARGET_PARENT,
        "../evil": sorter.TARGET_PARENT,
        "../../evil": sorter.TARGET_PARENT,
        "./../x": sorter.TARGET_PARENT,
        "Россия/../../evil": sorter.TARGET_PARENT,
        "C:/windows": sorter.TARGET_NOT_RELATIVE,
        "/etc": sorter.TARGET_NOT_RELATIVE,
        "//server/share": sorter.TARGET_NOT_RELATIVE,
        "..\\..\\x": sorter.TARGET_NOT_RELATIVE,
        "Франция\\Париж": sorter.TARGET_NOT_RELATIVE,
        "   ": sorter.TARGET_EMPTY,
        "Кар\x01елия": sorter.TARGET_CONTROL,
        "Карелия\nРоссия": sorter.TARGET_CONTROL,
    }

    def setUp(self):
        super().setUp()
        self.fid, _p, _c = self.add_photo_file("a.jpg", country="ru", city="Moscow")
        self.start_server()

    def test_each_escape_is_refused_with_its_own_reason(self):
        for target, reason in self.CASES.items():
            with self.subTest(target=target):
                status, payload = self.reassign([self.fid], target)
                self.assertEqual(status, 400)
                self.assertEqual(payload["reason"], reason)

    def test_a_refused_target_writes_nothing(self):
        for target in self.CASES:
            with self.subTest(target=target):
                self.reassign([self.fid], target)
                self.assertEqual(self.rows(), {})

    def test_a_refused_target_does_not_replace_a_correction_already_made(self):
        self.reassign([self.fid], "Россия/Карелия")
        self.reassign([self.fid], "../../evil")
        self.assertEqual(self.rows(), {self.fid: ("reassign", "Россия/Карелия")})

    def test_the_file_keeps_its_automatic_place_after_a_refusal(self):
        self.reassign([self.fid], "../../evil")
        item = self.plan().plan[0]
        self.assertEqual(item.reason, "city")
        self.assertNotIn("evil", item.target_rel)

    def test_the_other_actions_are_unaffected_by_the_target_check(self):
        # `exclude`/`clear`/`photo` carry no target at all — a check that reached them
        # would refuse a body that is entirely correct.
        for action in ("exclude", "photo", "clear"):
            with self.subTest(action=action):
                status, _payload = self.post(
                    "/api/overrides", {"file_ids": [self.fid], "action": action})
                self.assertEqual(status, 200)

    def test_every_reason_has_a_sentence_in_all_three_languages(self):
        for reason in set(self.CASES.values()):
            with self.subTest(reason=reason):
                entry = ui._UI_STRINGS[f"override_target_bad_{reason}"]
                self.assertEqual(set(entry), {"ru", "en", "ja"})
                for lang, value in entry.items():
                    self.assertTrue(value.strip(), f"{reason}/{lang} is empty")


class TestOneCleaningRuleForBothPaths(SorterOverridesTestBase):
    """The web app's check and the layout clean a name with the SAME function.

    Two cleanings would drift: the page would promise a folder the layout does not
    write, and the correction would land somewhere the user never named.
    """

    NAMES = ["Па*ри?ж", " Франция / Париж ", "Россия/", "CON", "Карелия.",
             'Пу<те>ше"ствие']

    def test_the_accepted_name_is_the_sanitized_name_segment_by_segment(self):
        for raw in self.NAMES:
            with self.subTest(target=raw):
                parts, refusal = manual_target_parts(raw)
                self.assertIsNone(refusal)
                expected = [_sanitize(seg) for seg in raw.split("/")
                            if seg.strip() not in ("", ".")]
                self.assertEqual(parts, expected)

    def test_the_plan_lays_the_file_out_under_exactly_those_segments(self):
        for i, raw in enumerate(self.NAMES):
            with self.subTest(target=raw):
                fid = self.add_file(f"clean{i}/a.jpg", country="FR", city="Paris")
                self.override(fid, "reassign", raw)
                item = {it.file_id: it for it in self.plan().plan}[fid]
                parts, _refusal = manual_target_parts(raw)
                assert parts is not None
                self.assertEqual(item.target_rel.rsplit("/", 1)[0], "/".join(parts))

    def test_a_refused_name_is_refused_by_both_halves(self):
        for raw in ["../evil", "C:/windows", "Франция\\Париж", "  "]:
            with self.subTest(target=raw):
                parts, refusal = manual_target_parts(raw)
                self.assertIsNone(parts)
                self.assertIsNotNone(refusal)
                self.assertIsNone(sorter._manual_target_parts(raw, "a.jpg"))


class TestTheCountryRootIsALegalTarget(SorterOverridesTestBase):
    def test_the_file_lands_directly_under_the_country(self):
        fid = self.add_file("a.jpg", country="FR", city="Paris")
        self.override(fid, "reassign", "Россия")
        item = self.plan().plan[0]
        self.assertEqual(item.target_rel, "Россия/a.jpg")
        self.assertEqual(item.dst, self.dest / "Россия" / "a.jpg")

    def test_a_trailing_slash_is_the_same_country_root(self):
        # «Россия/» is how a person writes "the country folder itself" — an empty last
        # segment is not an empty target.
        fid = self.add_file("a.jpg", country="FR", city="Paris")
        self.override(fid, "reassign", "Россия/")
        self.assertEqual(self.plan().plan[0].target_rel, "Россия/a.jpg")

    def test_the_year_is_not_appended_to_a_hand_named_folder(self):
        # The automatic branch would put this file in France/Paris/2022 — the correction
        # is the whole target, not a prefix the layout keeps building on.
        fid = self.add_file("a.jpg", country="FR", city="Paris")
        self.override(fid, "reassign", "Россия")
        self.assertNotIn("2022", self.plan().plan[0].target_rel)


class TestNameConflictInANewFolder(SorterOverridesTestBase):
    def test_two_files_sent_to_the_same_new_folder_get_a_suffix(self):
        first = self.add_file("one/a.jpg", content=b"first")
        second = self.add_file("two/a.jpg", content=b"second")
        for fid in (first, second):
            self.override(fid, "reassign", "Поездки/Карелия")
        names = sorted(it.dst.name for it in self.plan().plan)
        self.assertEqual(names, ["a.jpg", "a_1.jpg"])

    def test_a_file_already_lying_in_the_folder_is_not_overwritten(self):
        occupied = self.dest / "Поездки" / "Карелия"
        occupied.mkdir(parents=True)
        (occupied / "a.jpg").write_bytes(b"already there")
        fid = self.add_file("one/a.jpg", content=b"mine")
        self.override(fid, "reassign", "Поездки/Карелия")
        with redirect_stdout(io.StringIO()):
            report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=True)
        self.assertEqual(report.moved, 1)
        self.assertEqual((occupied / "a.jpg").read_bytes(), b"already there")
        self.assertEqual((occupied / "a_1.jpg").read_bytes(), b"mine")


class TestTheFieldSaysItTakesANewFolder(UiOverridesTestBase):
    """The markup: a field that is typed into, with the plan's folders as suggestions."""

    def setUp(self):
        super().setUp()
        self.start_server()
        _status, body, _ctype = self.get("/")
        self.html = body.decode("utf-8")

    def test_the_target_is_an_input_with_the_plan_folders_as_a_datalist(self):
        self.assertIn('<input type="text" id="city-override-target"', self.html)
        self.assertIn('list="city-override-target-options"', self.html)
        self.assertIn('<datalist id="city-override-target-options">', self.html)
        # the suggestions are still filled from the aggregate that is already loaded
        self.assertIn("fillOverrideTargets(categories)", self.html)

    def test_the_page_says_a_folder_may_be_typed(self):
        for lang in ("ru", "en", "ja"):
            with self.subTest(lang=lang):
                self.assertTrue(ui._UI_STRINGS["override_target_hint"][lang].strip())
        # The page itself is served in the interface language — English here.
        hint = ui._UI_STRINGS["override_target_hint"]["en"]
        self.assertIn(hint.split(":")[0], self.html)

    def test_the_refusal_reasons_reach_the_page_as_sentences(self):
        self.assertIn("override_target_bad_", self.html)


class TestTheRuleIsNotDuplicated(unittest.TestCase):
    def test_the_web_app_asks_the_sorter_and_holds_no_second_cleaner(self):
        # The one guard against the failure this feature is most likely to grow: a
        # second `_sanitize` in ui/, written the day someone wants a slightly different
        # rule for the field.
        source = Path(ui.layout.__file__).read_text(encoding="utf-8")
        self.assertIn("manual_target_parts", source)
        self.assertNotIn("_FORBIDDEN_CHARS", source)


if __name__ == "__main__":
    unittest.main()
