"""F78: the undated bucket is split by whether the file was shot by a camera.

Measured on the live collection: 1057 of 1059 files in `_Unsorted/low_date/` carried
no camera trace whatsoever (no camera_make/camera_model, no GPS) and had numeric
messenger-cache names — the bucket is forwarded/downloaded pictures, not shots whose
date was lost. So a file with any camera trace stays in low_date and everything else
goes to `_Unsorted/downloaded/` with its own reason code.

Inherits the SorterTestBase fixtures from test_sorter.py; all FS operations — inside
its tmp dir.
"""
from __future__ import annotations

import io
import sqlite3
import unittest
from contextlib import redirect_stdout

from tests.test_sorter import SorterTestBase

from sorta import i18n
from sorta.sorter import _looks_like_a_camera_shot, plan_and_sort

LOW_DATE_DIR = "_Unsorted/low_date"
DOWNLOADED_DIR = "_Unsorted/downloaded"


def make_row(camera_make: str | None = None, camera_model: str | None = None,
             gps_lat: float | None = None) -> sqlite3.Row:
    """A real sqlite3.Row with just the columns the predicate reads."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(
            "SELECT ? AS camera_make, ? AS camera_model, ? AS gps_lat",
            (camera_make, camera_model, gps_lat)).fetchone()
    finally:
        conn.close()


class UndatedTestBase(SorterTestBase):
    def plan(self, mode: str = "city", **kwargs) -> object:
        with redirect_stdout(io.StringIO()):
            return plan_and_sort(self.cfg, self.conn, mode, self.dest, apply=False,
                                 **kwargs)

    def targets(self, report) -> dict[int, str]:
        return {it.file_id: it.target_rel for it in report.plan}

    def only(self, report):
        self.assertEqual(len(report.plan), 1)
        return report.plan[0]


class TestCameraShotPredicate(unittest.TestCase):
    """The predicate is pure — check it directly on every NULL/non-NULL combination."""

    def test_no_trace_at_all_is_not_a_camera_shot(self):
        self.assertFalse(_looks_like_a_camera_shot(make_row()))

    def test_any_single_trace_is_enough(self):
        self.assertTrue(_looks_like_a_camera_shot(make_row(camera_make="Apple")))
        self.assertTrue(_looks_like_a_camera_shot(make_row(camera_model="iPhone 12")))
        self.assertTrue(_looks_like_a_camera_shot(make_row(gps_lat=55.75)))

    def test_every_combination(self):
        makes = (None, "Apple")
        models = (None, "iPhone 12")
        lats = (None, 55.75)
        for make in makes:
            for model in models:
                for lat in lats:
                    expected = bool(make or model or lat is not None)
                    with self.subTest(make=make, model=model, lat=lat):
                        self.assertEqual(
                            _looks_like_a_camera_shot(make_row(make, model, lat)),
                            expected)

    def test_zero_latitude_still_counts(self):
        # 0.0 is a valid latitude — the predicate compares to None, not truthiness.
        self.assertTrue(_looks_like_a_camera_shot(make_row(gps_lat=0.0)))

    def test_empty_camera_strings_are_not_a_trace(self):
        # An empty EXIF string is the absence of a value, not a camera.
        self.assertFalse(_looks_like_a_camera_shot(make_row(camera_make="",
                                                            camera_model="")))


class TestUndatedSplit(UndatedTestBase):
    def test_camera_make_keeps_the_file_in_low_date(self):
        self.add_file("a.jpg", confidence="low", camera_make="Apple")
        item = self.only(self.plan())
        self.assertEqual(item.reason, "low_date")
        self.assertEqual(item.target_rel, f"{LOW_DATE_DIR}/a.jpg")

    def test_no_camera_trace_goes_to_downloaded(self):
        self.add_file("10083666931142353280.JPG", confidence="low")
        item = self.only(self.plan())
        self.assertEqual(item.reason, "downloaded")
        self.assertEqual(item.target_rel,
                         f"{DOWNLOADED_DIR}/10083666931142353280.JPG")

    def test_missing_taken_at_without_a_camera_also_goes_to_downloaded(self):
        self.add_file("a.jpg", taken_at=None, confidence=None)
        item = self.only(self.plan())
        self.assertEqual(item.reason, "downloaded")
        self.assertEqual(item.target_rel, f"{DOWNLOADED_DIR}/a.jpg")

    def test_camera_model_alone_is_enough(self):
        self.add_file("a.jpg", confidence="low", camera_model="iPhone 12")
        self.assertEqual(self.only(self.plan()).target_rel, f"{LOW_DATE_DIR}/a.jpg")

    def test_gps_alone_is_enough(self):
        self.add_file("a.jpg", confidence="low", gps_lat=55.75, gps_lon=37.61)
        self.assertEqual(self.only(self.plan()).target_rel, f"{LOW_DATE_DIR}/a.jpg")

    def test_split_in_every_mode(self):
        for mode in ("city", "person", "event"):
            with self.subTest(mode=mode):
                shot = self.add_file(f"{mode}_shot.jpg", confidence="low",
                                     camera_make="Apple")
                downloaded = self.add_file(f"{mode}_dl.jpg", confidence="low")
                targets = self.targets(self.plan(mode))
                self.assertEqual(targets[shot], f"{LOW_DATE_DIR}/{mode}_shot.jpg")
                self.assertEqual(targets[downloaded], f"{DOWNLOADED_DIR}/{mode}_dl.jpg")

    def test_folder_name_follows_the_config_language(self):
        self.add_file("a.jpg", confidence="low")
        for lang, expected in (("ru", "_Неразобрано/скачанное"),
                               ("ja", "_未分類/ダウンロード")):
            with self.subTest(lang=lang):
                self.cfg.raw = {"language": lang}
                item = self.only(self.plan())
                self.assertEqual(item.target_rel, f"{expected}/a.jpg")
                self.assertEqual(item.reason, "downloaded")  # the code is not localized

    def test_reason_reaches_the_csv(self):
        self.add_file("shot.jpg", confidence="low", camera_make="Apple")
        self.add_file("dl.jpg", confidence="low")
        report = self.plan()
        rows = {r["path"].rsplit("\\", 1)[-1].rsplit("/", 1)[-1]: r
                for r in self.read_csv(report.csv_path)}
        self.assertEqual(rows["shot.jpg"]["reason"], "low_date")
        self.assertEqual(rows["dl.jpg"]["reason"], "downloaded")
        self.assertEqual(rows["dl.jpg"]["target"], f"{DOWNLOADED_DIR}/dl.jpg")

    def test_live_collection_split(self):
        # The measured shape of the live bucket: 1057 files with no camera trace, 2
        # real shots. The split must go exactly along that line.
        downloaded = [self.add_file(f"dl/{i}.JPG", content=f"{i}".encode(),
                                    confidence="low") for i in range(1057)]
        shots = [self.add_file("shot1.jpg", confidence="low", camera_make="Apple"),
                 self.add_file("shot2.jpg", confidence="low", gps_lat=55.75,
                               gps_lon=37.61)]
        targets = self.targets(self.plan(write_reports=False))
        by_dir: dict[str, int] = {}
        for target in targets.values():
            by_dir[target.rsplit("/", 1)[0]] = by_dir.get(target.rsplit("/", 1)[0], 0) + 1
        self.assertEqual(by_dir, {DOWNLOADED_DIR: 1057, LOW_DATE_DIR: 2})
        self.assertTrue(all(targets[fid].startswith(DOWNLOADED_DIR)
                            for fid in downloaded))
        self.assertTrue(all(targets[fid].startswith(LOW_DATE_DIR) for fid in shots))


class TestDatedFilesUntouched(UndatedTestBase):
    def test_a_dated_file_ignores_every_camera_value(self):
        cases = {
            "none.jpg": {},
            "make.jpg": {"camera_make": "Apple"},
            "model.jpg": {"camera_model": "iPhone 12"},
            "gps.jpg": {"gps_lat": 55.75, "gps_lon": 37.61},
            "all.jpg": {"camera_make": "Apple", "camera_model": "iPhone 12",
                        "gps_lat": 55.75, "gps_lon": 37.61},
        }
        for name, extra in cases.items():
            with self.subTest(name=name):
                fid = self.add_file(name, country="FR", city="Paris", **extra)
                item = {it.file_id: it for it in self.plan().plan}[fid]
                self.assertEqual(item.reason, "city")
                self.assertEqual(item.target_rel, f"France/Paris/2022/{name}")

    def test_medium_confidence_is_not_low_and_stays_dated(self):
        self.add_file("a.jpg", confidence="medium", country="FR", city="Paris")
        item = self.only(self.plan())
        self.assertEqual(item.reason, "city")
        self.assertEqual(item.target_rel, "France/Paris/2022/a.jpg")

    def test_event_year_still_wins_over_the_split(self):
        # F5.1: a low-confidence file inside an event takes the event year — it never
        # reaches the undated branch, camera trace or not.
        fid = self.add_file("msg.jpg", confidence="low")
        self.add_event(fid, "Конференция", started_at="2021-01-01T09:00:00")
        item = self.only(self.plan("event"))
        self.assertEqual(item.reason, "event")
        self.assertEqual(item.target_rel, "2021/Конференция/msg.jpg")


class TestCheckOrderPreserved(UndatedTestBase):
    """Everything above the date check keeps winning — the split lives strictly inside
    the branch where the year could not be determined."""

    def test_document_wins(self):
        self.add_file("a.jpg", confidence="low", junk_verdict="document")
        item = self.only(self.plan())
        self.assertEqual(item.reason, "document")
        self.assertEqual(item.target_rel, "_Documents/a.jpg")

    def test_product_wins(self):
        self.add_file("a.jpg", confidence="low", junk_verdict="product")
        item = self.only(self.plan())
        self.assertEqual(item.reason, "product")
        self.assertEqual(item.target_rel, "_Products/a.jpg")

    def test_junk_wins(self):
        self.add_file("a.jpg", confidence="low", junk_verdict="screenshot")
        item = self.only(self.plan())
        self.assertEqual(item.reason, "junk")
        self.assertEqual(item.target_rel, "_Unsorted/junk/screenshot/a.jpg")

    def test_not_personal_wins(self):
        fid = self.add_file("a.jpg", confidence="low")
        self.conn.execute("UPDATE files SET not_personal = 1 WHERE id = ?", (fid,))
        self.conn.commit()
        item = self.only(self.plan())
        self.assertEqual(item.reason, "not_personal")
        self.assertEqual(item.target_rel, "_Unsorted/not_personal/a.jpg")

    def test_dedup_delete_wins(self):
        fid = self.add_file("a.jpg", confidence="low")
        self.conn.execute(
            "INSERT INTO dedup_choice (file_id, action, updated_at) "
            "VALUES (?, 'to_delete', '2026-07-26')", (fid,))
        self.conn.commit()
        item = self.only(self.plan())
        self.assertEqual(item.reason, "dedup_delete")
        self.assertEqual(item.target_rel, "_delete/a.jpg")

    def test_manual_reassign_wins(self):
        fid = self.add_file("a.jpg", confidence="low")
        self.conn.execute(
            """INSERT INTO manual_overrides (file_id, action, target, updated_at)
               VALUES (?, 'reassign', 'Франция/Париж/2014', '2026-07-26')""", (fid,))
        self.conn.commit()
        item = self.only(self.plan())
        self.assertEqual(item.reason, "manual_reassign")
        self.assertEqual(item.target_rel, "Франция/Париж/2014/a.jpg")


class TestNoLowConfidenceRegression(UndatedTestBase):
    """Data without an undated file is laid out exactly as before F78."""

    def build(self) -> dict[str, int]:
        return {
            "city": self.add_file("a.jpg", country="FR", city="Paris"),
            "document": self.add_file("b.jpg", junk_verdict="document"),
            "junk": self.add_file("c.jpg", junk_verdict="screenshot"),
            "person": self.add_file("d.jpg", country="FR", city="Paris",
                                    camera_make="Apple"),
            "noplace": self.add_file("e.jpg"),
        }

    def test_layout_matches_the_pre_f78_plan(self):
        ids = self.build()
        self.add_person(ids["person"], "Аня")
        expected = {
            "city": {
                ids["city"]: "France/Paris/2022/a.jpg",
                ids["document"]: "_Documents/b.jpg",
                ids["junk"]: "_Unsorted/junk/screenshot/c.jpg",
                ids["person"]: "France/Paris/2022/d.jpg",
                ids["noplace"]: "_Unsorted/no_place/e.jpg",
            },
            "person": {
                ids["city"]: "_Unsorted/no_faces/a.jpg",
                ids["document"]: "_Documents/b.jpg",
                ids["junk"]: "_Unsorted/junk/screenshot/c.jpg",
                ids["person"]: "Аня/2022/d.jpg",
                ids["noplace"]: "_Unsorted/no_faces/e.jpg",
            },
            "event": {
                ids["city"]: "2022/05/a.jpg",
                ids["document"]: "_Documents/b.jpg",
                ids["junk"]: "_Unsorted/junk/screenshot/c.jpg",
                ids["person"]: "2022/05/d.jpg",
                ids["noplace"]: "2022/05/e.jpg",
            },
        }
        for mode, targets in expected.items():
            with self.subTest(mode=mode):
                report = self.plan(mode)
                self.assertEqual(self.targets(report), targets)

    def test_no_downloaded_folder_or_reason_appears(self):
        self.build()
        for mode in ("city", "person", "event"):
            with self.subTest(mode=mode):
                report = self.plan(mode)
                self.assertNotIn("downloaded", [it.reason for it in report.plan])
                csv_text = report.csv_path.read_text(encoding="utf-8-sig")
                self.assertNotIn("downloaded", csv_text)


class TestFolderNameI18n(unittest.TestCase):
    def test_key_is_translated_into_all_three_languages(self):
        self.assertIn("downloaded", i18n.FOLDER_KEYS)
        values = {lang: i18n.folder("downloaded", lang) for lang in ("ru", "en", "ja")}
        for lang, value in values.items():
            self.assertTrue(value.strip(), f"downloaded/{lang} is empty")
        for lang in ("ru", "ja"):
            # folder() returns the key itself for an unknown key — that must not pass
            # for a translated language
            self.assertNotEqual(values[lang], "downloaded")
        self.assertEqual(len(set(values.values())), 3)

    def test_the_name_does_not_read_as_a_verdict(self):
        # The user may well want to browse this folder — it must not be named "junk".
        for lang in ("ru", "en", "ja"):
            name = i18n.folder("downloaded", lang)
            self.assertNotEqual(name, i18n.folder("junk", lang))
            self.assertNotEqual(name, i18n.folder("low_date", lang))


if __name__ == "__main__":
    unittest.main()
