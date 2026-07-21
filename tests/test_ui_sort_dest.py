"""Cities: the "Browse…" button next to the destination field + default path =
<source>_sorted (GET /api/sort/suggest-dest, helper _suggested_sort_dest)."""
from __future__ import annotations

import dataclasses
import json
import unittest
from pathlib import Path

from sorta.ui import _suggested_sort_dest
from tests.test_ui import UiServerTestBase


class TestSuggestDestHelper(UiServerTestBase):
    def test_from_config_source(self):
        # cfg.sources[0] -> <source>_sorted (POSIX)
        dest = _suggested_sort_dest(self.cfg, self.cfg.database)
        expected = (self.src_dir.parent / (self.src_dir.name + "_sorted")).as_posix()
        self.assertEqual(dest, expected)
        self.assertTrue(dest.endswith("_sorted"))

    def test_fallback_to_indexed_root_when_no_sources(self):
        # without cfg.sources — the common root of the indexed files from the DB
        self.add_photo_file("a.jpg")
        self.add_photo_file("b.jpg")
        cfg = dataclasses.replace(self.cfg, sources=[])
        dest = _suggested_sort_dest(cfg, self.cfg.database)
        self.assertTrue(dest.endswith("_sorted"))
        # root = self.src_dir (the common ancestor of a.jpg/b.jpg)
        self.assertEqual(dest, (self.src_dir.parent / (self.src_dir.name + "_sorted")).as_posix())

    def test_empty_when_nothing_known(self):
        cfg = dataclasses.replace(self.cfg, sources=[])
        # empty DB, no sources -> an empty string (the field is for manual entry)
        self.assertEqual(_suggested_sort_dest(cfg, Path(self.cfg.database)), "")


class TestSuggestDestEndpoint(UiServerTestBase):
    def test_endpoint_returns_dest(self):
        self.start_server()
        status, body, _c = self.get("/api/sort/suggest-dest")
        self.assertEqual(status, 200)
        payload = json.loads(body)
        self.assertIn("dest", payload)
        self.assertTrue(payload["dest"].endswith("_sorted"))


class TestSortBrowseButtonMarkup(UiServerTestBase):
    def test_city_tab_has_browse_and_suggest_wiring(self):
        self.start_server()
        _s, body, _c = self.get("/")
        html = body.decode("utf-8")
        self.assertIn('id="sort-browse-btn"', html)          # the "Browse…" button
        self.assertIn('"/api/sort/suggest-dest"', html)       # default prefill
        self.assertIn('"sort-browse-btn").addEventListener', html)


if __name__ == "__main__":
    unittest.main()
