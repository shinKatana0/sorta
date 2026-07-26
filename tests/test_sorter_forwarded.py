"""F83: a `meme` verdict on a file with no camera trace is routed as a `photo`.

Measured on the live collection: 3437 files carry no camera trace at all
(camera_make/camera_model/GPS all NULL) — forwarded and downloaded pictures. There
`photo` vs `meme` is a CLIP call on content alone, wrong in both directions, so two
files of one origin used to land in different folders. The rule collapses that one
undecidable pair and nothing else: `document` and `screenshot` keep their folders (a
scan has no camera EXIF either), and files WITH a camera trace are untouched.

Inherits the SorterTestBase fixtures from test_sorter.py; all FS operations — inside
its tmp dir.
"""
from __future__ import annotations

import io
import sqlite3
import unittest
from contextlib import redirect_stdout

from tests.test_sorter import SorterTestBase

from sorta.sorter import _is_indistinguishable_meme, plan_and_sort

DOWNLOADED_DIR = "_Unsorted/downloaded"
LOW_DATE_DIR = "_Unsorted/low_date"
MEME_DIR = "_Unsorted/junk/meme"
SCREENSHOT_DIR = "_Unsorted/junk/screenshot"
DOCUMENTS_DIR = "_Documents"


def make_row(junk_verdict: str | None = None, camera_make: str | None = None,
             camera_model: str | None = None,
             gps_lat: float | None = None) -> sqlite3.Row:
    """A real sqlite3.Row with just the columns the predicate reads."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(
            "SELECT ? AS junk_verdict, ? AS camera_make, ? AS camera_model, "
            "? AS gps_lat",
            (junk_verdict, camera_make, camera_model, gps_lat)).fetchone()
    finally:
        conn.close()


class ForwardedTestBase(SorterTestBase):
    def plan(self, mode: str = "city", **kwargs) -> object:
        with redirect_stdout(io.StringIO()):
            return plan_and_sort(self.cfg, self.conn, mode, self.dest, apply=False,
                                 **kwargs)

    def targets(self, report) -> dict[int, str]:
        return {it.file_id: it.target_rel for it in report.plan}

    def only(self, report):
        self.assertEqual(len(report.plan), 1)
        return report.plan[0]


class TestPredicate(unittest.TestCase):
    """The predicate is pure — check it directly, verdict by verdict."""

    def test_meme_without_any_trace(self):
        self.assertTrue(_is_indistinguishable_meme(make_row("meme")))

    def test_meme_with_any_single_trace_is_not_indistinguishable(self):
        self.assertFalse(_is_indistinguishable_meme(
            make_row("meme", camera_make="Apple")))
        self.assertFalse(_is_indistinguishable_meme(
            make_row("meme", camera_model="iPhone 12")))
        self.assertFalse(_is_indistinguishable_meme(make_row("meme", gps_lat=55.75)))

    def test_zero_latitude_is_still_a_trace(self):
        # 0.0 is a valid latitude — the reused F78 predicate compares to None.
        self.assertFalse(_is_indistinguishable_meme(make_row("meme", gps_lat=0.0)))

    def test_no_other_verdict_collapses(self):
        for verdict in ("document", "screenshot", "product", "photo", None):
            with self.subTest(verdict=verdict):
                self.assertFalse(_is_indistinguishable_meme(make_row(verdict)))


class TestMemeWithoutCameraTrace(ForwardedTestBase):
    def test_no_trace_and_no_reliable_date_goes_to_downloaded(self):
        self.add_file("10049075250049169544.JPG", confidence="low",
                      junk_verdict="meme", junk_source="clip")
        item = self.only(self.plan())
        self.assertEqual(item.reason, "downloaded")
        self.assertEqual(item.target_rel,
                         f"{DOWNLOADED_DIR}/10049075250049169544.JPG")

    def test_the_indistinguishable_pair_lands_together(self):
        # The live case from the brief: the same kind of file, two CLIP verdicts.
        meme = self.add_file("10049075250049169544.JPG", content=b"a",
                             confidence="low", junk_verdict="meme", junk_source="clip")
        photo = self.add_file("10632945825951394947.JPG", content=b"b",
                              confidence="low", junk_verdict="photo", junk_source="clip")
        targets = self.targets(self.plan())
        self.assertEqual(targets[meme].rsplit("/", 1)[0],
                         targets[photo].rsplit("/", 1)[0])
        self.assertTrue(targets[meme].startswith(DOWNLOADED_DIR))

    def test_missing_taken_at_also_goes_to_downloaded(self):
        self.add_file("a.jpg", taken_at=None, confidence=None, junk_verdict="meme")
        self.assertEqual(self.only(self.plan()).target_rel, f"{DOWNLOADED_DIR}/a.jpg")

    def test_collapses_in_every_mode(self):
        for mode in ("city", "person", "event"):
            with self.subTest(mode=mode):
                fid = self.add_file(f"{mode}.jpg", confidence="low",
                                    junk_verdict="meme")
                self.assertEqual(self.targets(self.plan(mode))[fid],
                                 f"{DOWNLOADED_DIR}/{mode}.jpg")

    def test_verdict_stays_in_the_db_and_in_the_csv(self):
        # The route changes, the verdict does not — reports and the UI still need it.
        fid = self.add_file("a.jpg", confidence="low", junk_verdict="meme",
                            junk_source="clip")
        report = self.plan()
        row = self.conn.execute(
            "SELECT verdict FROM media_class WHERE file_id = ?", (fid,)).fetchone()
        self.assertEqual(row["verdict"], "meme")
        csv_row = self.read_csv(report.csv_path)[0]
        self.assertEqual(csv_row["junk_verdict"], "meme")
        self.assertEqual(csv_row["target"], f"{DOWNLOADED_DIR}/a.jpg")


class TestCameraShotsUntouched(ForwardedTestBase):
    """With a camera trace `meme` is a meaningful verdict (0 false ones among 20743
    live camera shots) — nothing about those files changes."""

    def test_camera_make_keeps_the_meme_in_junk(self):
        self.add_file("a.jpg", confidence="low", junk_verdict="meme",
                      camera_make="Apple")
        item = self.only(self.plan())
        self.assertEqual(item.reason, "junk")
        self.assertEqual(item.target_rel, f"{MEME_DIR}/a.jpg")

    def test_any_single_trace_is_enough(self):
        cases = {
            "model.jpg": {"camera_model": "iPhone 12"},
            "gps.jpg": {"gps_lat": 55.75, "gps_lon": 37.61},
            "zero_gps.jpg": {"gps_lat": 0.0, "gps_lon": 0.0},
        }
        for name, extra in cases.items():
            with self.subTest(name=name):
                fid = self.add_file(name, confidence="low", junk_verdict="meme",
                                    **extra)
                self.assertEqual(self.targets(self.plan())[fid], f"{MEME_DIR}/{name}")

    def test_a_dated_camera_meme_also_stays_in_junk(self):
        self.add_file("a.jpg", junk_verdict="meme", camera_make="Apple",
                      country="FR", city="Paris")
        self.assertEqual(self.only(self.plan()).target_rel, f"{MEME_DIR}/a.jpg")


class TestOtherVerdictsUntouched(ForwardedTestBase):
    def test_document_without_a_camera_trace_stays_in_documents(self):
        # The key test: the rejected first version of this rule ("no camera -> down-
        # loaded") swept scanned documents out of the review folder. A scanner writes
        # no camera EXIF, so a scan looks exactly like a forwarded picture here.
        self.add_file("20201104_111852_IMG_0007.PNG", confidence="low",
                      junk_verdict="document", junk_source="clip")
        item = self.only(self.plan())
        self.assertEqual(item.reason, "document")
        self.assertEqual(item.target_rel, f"{DOCUMENTS_DIR}/20201104_111852_IMG_0007.PNG")

    def test_document_from_ocr_stays_too(self):
        self.add_file("a.jpg", confidence="low", junk_verdict="document",
                      junk_source="ocr")
        self.assertEqual(self.only(self.plan()).target_rel, f"{DOCUMENTS_DIR}/a.jpg")

    def test_screenshot_without_a_camera_trace_stays_in_junk(self):
        self.add_file("a.jpg", confidence="low", junk_verdict="screenshot",
                      junk_source="clip")
        item = self.only(self.plan())
        self.assertEqual(item.reason, "junk")
        self.assertEqual(item.target_rel, f"{SCREENSHOT_DIR}/a.jpg")

    def test_product_without_a_camera_trace_stays_in_products(self):
        self.add_file("a.jpg", confidence="low", junk_verdict="product")
        item = self.only(self.plan())
        self.assertEqual(item.reason, "product")
        self.assertEqual(item.target_rel, "_Products/a.jpg")


class TestDateStillDecides(ForwardedTestBase):
    """The rule changes the verdict branch only — the date branch below is F78's."""

    def test_a_reliable_date_lays_the_meme_out_by_city(self):
        self.add_file("a.jpg", junk_verdict="meme", country="FR", city="Paris")
        item = self.only(self.plan())
        self.assertEqual(item.reason, "city")
        self.assertEqual(item.target_rel, "France/Paris/2022/a.jpg")

    def test_a_reliable_date_lays_the_meme_out_by_date_in_event_mode(self):
        self.add_file("a.jpg", junk_verdict="meme")
        item = self.only(self.plan("event"))
        self.assertEqual(item.reason, "no_event")
        self.assertEqual(item.target_rel, "2022/05/a.jpg")

    def test_a_dated_meme_without_a_place_goes_to_no_place(self):
        self.add_file("a.jpg", junk_verdict="meme")
        item = self.only(self.plan())
        self.assertEqual(item.reason, "no_place")
        self.assertEqual(item.target_rel, "_Unsorted/no_place/a.jpg")

    def test_a_meme_inside_an_event_takes_the_event_year(self):
        fid = self.add_file("msg.jpg", confidence="low", junk_verdict="meme")
        self.add_event(fid, "Конференция", started_at="2021-01-01T09:00:00")
        item = self.only(self.plan("event"))
        self.assertEqual(item.reason, "event")
        self.assertEqual(item.target_rel, "2021/Конференция/msg.jpg")


class TestCheckOrderPreserved(ForwardedTestBase):
    """Everything that outranked the junk branch before still outranks it."""

    def test_dedup_delete_wins(self):
        fid = self.add_file("a.jpg", confidence="low", junk_verdict="meme")
        self.conn.execute(
            "INSERT INTO dedup_choice (file_id, action, updated_at) "
            "VALUES (?, 'to_delete', '2026-07-26')", (fid,))
        self.conn.commit()
        item = self.only(self.plan())
        self.assertEqual(item.reason, "dedup_delete")
        self.assertEqual(item.target_rel, "_delete/a.jpg")

    def test_not_personal_wins(self):
        fid = self.add_file("a.jpg", confidence="low", junk_verdict="meme")
        self.conn.execute("UPDATE files SET not_personal = 1 WHERE id = ?", (fid,))
        self.conn.commit()
        item = self.only(self.plan())
        self.assertEqual(item.reason, "not_personal")
        self.assertEqual(item.target_rel, "_Unsorted/not_personal/a.jpg")

    def test_manual_reassign_wins(self):
        fid = self.add_file("a.jpg", confidence="low", junk_verdict="meme")
        self.conn.execute(
            """INSERT INTO manual_overrides (file_id, action, target, updated_at)
               VALUES (?, 'reassign', 'Франция/Париж/2014', '2026-07-26')""", (fid,))
        self.conn.commit()
        item = self.only(self.plan())
        self.assertEqual(item.reason, "manual_reassign")
        self.assertEqual(item.target_rel, "Франция/Париж/2014/a.jpg")

    def test_manual_exclude_wins(self):
        fid = self.add_file("a.jpg", confidence="low", junk_verdict="meme")
        self.conn.execute(
            """INSERT INTO manual_overrides (file_id, action, target, updated_at)
               VALUES (?, 'exclude', NULL, '2026-07-26')""", (fid,))
        self.conn.commit()
        report = self.plan()
        self.assertEqual(report.plan, [])
        self.assertEqual(report.manual_excluded, 1)


class TestNoRegressionOnCameraShots(ForwardedTestBase):
    """A collection where every file carries a camera trace is laid out exactly as
    before F83 — in all three modes."""

    def build(self) -> dict[str, int]:
        shot = {"camera_make": "Apple", "camera_model": "iPhone 12"}
        return {
            "city": self.add_file("a.jpg", country="FR", city="Paris", **shot),
            "meme": self.add_file("b.jpg", junk_verdict="meme", **shot),
            "document": self.add_file("c.jpg", junk_verdict="document", **shot),
            "screenshot": self.add_file("d.jpg", junk_verdict="screenshot", **shot),
            "undated": self.add_file("e.jpg", confidence="low", **shot),
        }

    def test_layout_matches_the_pre_f83_plan(self):
        ids = self.build()
        self.add_person(ids["city"], "Аня")
        expected = {
            "city": {
                ids["city"]: "France/Paris/2022/a.jpg",
                ids["meme"]: f"{MEME_DIR}/b.jpg",
                ids["document"]: f"{DOCUMENTS_DIR}/c.jpg",
                ids["screenshot"]: f"{SCREENSHOT_DIR}/d.jpg",
                ids["undated"]: f"{LOW_DATE_DIR}/e.jpg",
            },
            "person": {
                ids["city"]: "Аня/2022/a.jpg",
                ids["meme"]: f"{MEME_DIR}/b.jpg",
                ids["document"]: f"{DOCUMENTS_DIR}/c.jpg",
                ids["screenshot"]: f"{SCREENSHOT_DIR}/d.jpg",
                ids["undated"]: f"{LOW_DATE_DIR}/e.jpg",
            },
            "event": {
                ids["city"]: "2022/05/a.jpg",
                ids["meme"]: f"{MEME_DIR}/b.jpg",
                ids["document"]: f"{DOCUMENTS_DIR}/c.jpg",
                ids["screenshot"]: f"{SCREENSHOT_DIR}/d.jpg",
                ids["undated"]: f"{LOW_DATE_DIR}/e.jpg",
            },
        }
        for mode, targets in expected.items():
            with self.subTest(mode=mode):
                self.assertEqual(self.targets(self.plan(mode)), targets)

    def test_reasons_are_unchanged(self):
        self.build()
        reasons = sorted(it.reason for it in self.plan().plan)
        self.assertEqual(reasons, ["city", "document", "junk", "junk", "low_date"])


class TestLiveCollectionShape(ForwardedTestBase):
    """Acceptance: the live counts of files with no camera trace at all."""

    def test_memes_collapse_and_documents_stay(self):
        memes = [self.add_file(f"meme/{i}.JPG", content=f"m{i}".encode(),
                               confidence="low", junk_verdict="meme",
                               junk_source="clip") for i in range(188)]
        documents = [self.add_file(f"doc/{i}.JPG", content=f"d{i}".encode(),
                                   confidence="low", junk_verdict="document",
                                   junk_source="clip") for i in range(131)]
        documents += [self.add_file(f"ocr/{i}.JPG", content=f"o{i}".encode(),
                                    confidence="low", junk_verdict="document",
                                    junk_source="ocr") for i in range(60)]
        screenshots = [self.add_file(f"ss/{i}.JPG", content=f"s{i}".encode(),
                                     confidence="low", junk_verdict="screenshot",
                                     junk_source="clip") for i in range(1340)]
        targets = self.targets(self.plan(write_reports=False))
        by_dir: dict[str, int] = {}
        for target in targets.values():
            folder = target.rsplit("/", 1)[0]
            by_dir[folder] = by_dir.get(folder, 0) + 1
        self.assertEqual(by_dir, {DOWNLOADED_DIR: 188, DOCUMENTS_DIR: 191,
                                  SCREENSHOT_DIR: 1340})
        self.assertTrue(all(targets[fid].startswith(DOWNLOADED_DIR) for fid in memes))
        self.assertTrue(all(targets[fid].startswith(DOCUMENTS_DIR)
                            for fid in documents))
        self.assertTrue(all(targets[fid].startswith(SCREENSHOT_DIR)
                            for fid in screenshots))


if __name__ == "__main__":
    unittest.main()
