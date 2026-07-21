"""Date heuristics — the most critical pure module."""
import unittest
from datetime import datetime

from sorta.dates import parse_exif_datetime, parse_filename_datetime, resolve_taken_at


class TestFilename(unittest.TestCase):
    CASES = {
        "IMG_20190705_123456.jpg": datetime(2019, 7, 5, 12, 34, 56),
        "PXL_20210101_101112000.jpg": datetime(2021, 1, 1, 10, 11, 12),
        "20190705_123456.jpg": datetime(2019, 7, 5, 12, 34, 56),
        "VID_20200315_180000.mp4": datetime(2020, 3, 15, 18, 0, 0),
        "Screenshot_20200101-101112.png": datetime(2020, 1, 1, 10, 11, 12),
        "IMG-20190705-WA0001.jpg": datetime(2019, 7, 5),
        "WhatsApp Image 2020-05-01 at 12.34.56.jpeg": datetime(2020, 5, 1, 12, 34, 56),
        "photo_2019-07-05_12-34-56.jpg": datetime(2019, 7, 5, 12, 34, 56),
        "2019-07-05 132.jpg": datetime(2019, 7, 5),
        "scan_20051231.tif": datetime(2005, 12, 31),
    }

    def test_known_patterns(self):
        for name, expected in self.CASES.items():
            with self.subTest(name=name):
                self.assertEqual(parse_filename_datetime(name), expected)

    def test_garbage_rejected(self):
        for name in ["DSC01234.jpg", "IMG_1234.jpg", "99999999_123456.jpg",
                     "18001231.jpg", "20991301.jpg", "photo.jpg", "12345678901234.jpg"]:
            with self.subTest(name=name):
                self.assertIsNone(parse_filename_datetime(name))

    def test_year_bounds_from_config(self):
        self.assertIsNone(parse_filename_datetime("IMG_20190705_123456.jpg", min_year=2020))


class TestExif(unittest.TestCase):
    def test_standard(self):
        self.assertEqual(parse_exif_datetime("2019:07:05 12:34:56"),
                         datetime(2019, 7, 5, 12, 34, 56))

    def test_garbage(self):
        for v in [None, "", "0000:00:00 00:00:00", "not a date", "1889:01:01 00:00:00"]:
            self.assertIsNone(parse_exif_datetime(v))


class TestCascade(unittest.TestCase):
    def test_exif_wins(self):
        ta = resolve_taken_at("2019:07:05 12:34:56", "IMG_20200101_000000.jpg", 0)
        self.assertEqual((ta.source, ta.confidence, ta.dt.year), ("exif", "high", 2019))

    def test_filename_fallback(self):
        ta = resolve_taken_at(None, "IMG_20200101_000000.jpg", 0)
        self.assertEqual((ta.source, ta.confidence), ("filename", "medium"))

    def test_mtime_last(self):
        ta = resolve_taken_at("garbage", "DSC01234.jpg", 1600000000.0)
        self.assertEqual((ta.source, ta.confidence), ("mtime", "low"))


if __name__ == "__main__":
    unittest.main()
