"""F65: the geo-data guard in diagnostics — pure stat() checks, no data is loaded."""
import logging
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from sorta.diagnostics import GeoDataHealth, geo_data_health, warn_if_geo_data_missing

LOGGER_NAME = "sorta.diagnostics"

# one places.tsv row — enough to make the file non-empty and realistic
PLACES_ROW = "100\t59.9\t30.3\tPPLA\tRU\t66\t\tSaint Petersburg\t1\n"


def health_of(data_dir: Path) -> GeoDataHealth:
    return geo_data_health(data_dir)


class TestGeoDataHealth(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.data_dir = Path(self.tmp.name) / "geo"
        self.data_dir.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def write_places(self, content: str = PLACES_ROW):
        (self.data_dir / "places.tsv").write_text(content, encoding="utf-8")

    def test_available_when_places_file_is_there(self):
        self.write_places()
        health = health_of(self.data_dir)
        self.assertTrue(health.available)
        self.assertIsNone(health.problem)
        self.assertGreater(health.size_bytes, 0)
        self.assertEqual(health.places_path, str(self.data_dir / "places.tsv"))

    def test_missing_file_is_a_problem_with_the_path(self):
        health = health_of(self.data_dir)
        self.assertFalse(health.available)
        self.assertEqual(health.problem, "file not found")
        self.assertIsNone(health.size_bytes)
        self.assertIn(str(self.data_dir), health.summary)
        self.assertIn("build_geodata.py", health.summary)

    def test_empty_file_is_a_problem_too(self):
        self.write_places("")
        health = health_of(self.data_dir)
        self.assertFalse(health.available)
        self.assertEqual(health.problem, "file is empty")

    def test_missing_directory_does_not_raise(self):
        health = health_of(self.data_dir / "does_not_exist")
        self.assertFalse(health.available)
        self.assertEqual(health.problem, "file not found")

    def test_summary_reports_the_size_when_healthy(self):
        self.write_places()
        self.assertIn(str(self.data_dir), health_of(self.data_dir).summary)
        self.assertIn("MB", health_of(self.data_dir).summary)

    def test_does_not_read_the_file(self):
        # the check must stay cheap on a 10 MB base: stat() only, never open()
        self.write_places()
        with mock.patch.object(Path, "open", side_effect=AssertionError("must not open")):
            self.assertTrue(health_of(self.data_dir).available)

    def test_default_uses_the_bundled_package_data(self):
        # no argument -> the resolver's own directory, which after F65 is inside
        # the package and therefore present in any installation
        health = geo_data_health()
        self.assertTrue(health.available)
        self.assertTrue(health.places_path.endswith("places.tsv"))


class TestWarnIfGeoDataMissing(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.data_dir = Path(self.tmp.name) / "geo"
        self.data_dir.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_warns_once_with_path_and_fix(self):
        health = health_of(self.data_dir)
        with self.assertLogs(LOGGER_NAME, level=logging.WARNING) as cm:
            warned = warn_if_geo_data_missing(health)
        self.assertTrue(warned)
        self.assertEqual(len(cm.records), 1)
        message = cm.records[0].getMessage()
        self.assertIn(str(self.data_dir / "places.tsv"), message)
        self.assertIn("build_geodata.py", message)

    def test_silent_when_data_is_there(self):
        (self.data_dir / "places.tsv").write_text("x\n", encoding="utf-8")
        with self.assertNoLogs(LOGGER_NAME, level=logging.WARNING):
            self.assertFalse(warn_if_geo_data_missing(health_of(self.data_dir)))

    def test_collects_health_itself_when_not_given(self):
        # the bundled data is in the tree -> nothing to report
        self.assertFalse(warn_if_geo_data_missing())

    def test_custom_logger_receives_the_warning(self):
        log = mock.Mock(spec=logging.Logger)
        self.assertTrue(warn_if_geo_data_missing(health_of(self.data_dir), log=log))
        log.warning.assert_called_once()


if __name__ == "__main__":
    unittest.main()
