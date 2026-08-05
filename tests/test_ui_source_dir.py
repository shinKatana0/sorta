"""The source folder must reach the UI — it is the best evidence about a frame.

41% of the validation collection sits in hand-named directories ("Тайланд 04.2025"),
i.e. a person's own labelling of place and date, and only the basename used to be
exposed. It is also what makes a wrong guess recognisable: a Colosseum match is
plainly wrong once you see the file lives under "карелия".
"""
from __future__ import annotations

import unittest
from pathlib import Path

from sorta import ui


class TestPlanItemCarriesSource(unittest.TestCase):
    def _item(self, src: str):
        from sorta.sorter import PlanItem

        return PlanItem(
            file_id=1, src=Path(src), dst=Path("D:/out/x.jpg"), in_place=False,
            target_rel="Италия/Рим", reason="city", taken_at="2010-01-01T00:00:00",
            taken_at_confidence="high", country="IT", city="Rome",
            place_confidence="visual", gps_lat=None, gps_lon=None, persons=[],
            event=None, junk_verdict="photo", junk_source="clip",
            db_hash=None, db_algo=None,
        )

    def test_source_dir_and_full_path_are_exposed(self):
        # A hand-named folder in Cyrillic, which is the case this exists for: the source
        # folder is what tells a person WHERE a frame came from when the place could not
        # be resolved. Synthetic — never a path out of anybody's real archive.
        payload = ui._plan_item_to_json(self._item(r"D:/Photos/отпуск/DSC00001.JPG"))
        self.assertEqual(payload["src_dir"], "отпуск")
        self.assertIn("отпуск", payload["src_path"])
        self.assertNotIn("DSC00001.JPG", payload["src_path"])  # folder, not the file

    def test_rendered_row_shows_the_source_folder(self):
        html = ui._render_index_html("ru")
        self.assertIn("item.src_dir", html)
        self.assertIn("item.src_path", html)

    def test_existing_fields_are_untouched(self):
        payload = ui._plan_item_to_json(self._item(r"D:/a/b/x.jpg"))
        for key in ("file_id", "name", "target_rel", "reason", "date", "geo",
                    "category", "thumb_url"):
            self.assertIn(key, payload)
        self.assertEqual(payload["name"], "x.jpg")


if __name__ == "__main__":
    unittest.main()
