"""F152: the face albums — `sorter.plan_album(kind='people'|'group'|'portrait')`.

The three slices are the whole feature, and `face_slice_ids_sql` is the one place they
are written down: this file checks that expression directly and then checks that the
album gathers exactly what it selects. Everything else (dry-run semantics, the
journal-before-the-operation invariant, `_resolve_dst` on a repeat gather) is inherited
from F34/F97 and pinned here for the new kinds, because inheritance that is not checked
is a plan, not a property.

The one failure mode worth the file on its own: `bbox = '[]'` is not a face. It is the
marker "this file was processed and had none", 24 195 of 24 196 live files carry one,
and a predicate that keeps it turns "with people" into "everything".
"""
from __future__ import annotations

import dataclasses
import io
import os
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from sorta import i18n
from sorta.sorter import face_slice_ids_sql, plan_album

from tests.test_sorter import SorterTestBase


class FaceAlbumTestBase(SorterTestBase):
    def add_frame(self, rel: str, *, width: int = 1000, height: int = 1000,
                  **kwargs) -> int:
        file_id = self.add_file(rel, **kwargs)
        self.conn.execute("UPDATE files SET width = ?, height = ? WHERE id = ?",
                          (width, height, file_id))
        self.conn.commit()
        return file_id

    def add_face(self, file_id: int, bbox: str = "[0.0,0.0,100.0,100.0]") -> None:
        self.conn.execute(
            "INSERT INTO faces (file_id, bbox, embedding) VALUES (?, ?, ?)",
            (file_id, bbox, b"embedding"))
        self.conn.commit()

    def add_face_marker(self, file_id: int) -> None:
        self.conn.execute(
            "INSERT INTO faces (file_id, bbox, embedding) VALUES (?, '[]', ?)",
            (file_id, b""))
        self.conn.commit()

    def add_square_face(self, file_id: int, side: float, *, at: float = 0.0) -> None:
        self.add_face(file_id, f"[{at},{at},{at + side},{at + side}]")

    def selected(self, slice_: str) -> list[int]:
        """The ids `face_slice_ids_sql` returns, asked directly."""
        sql, params = face_slice_ids_sql(self.cfg, slice_)
        return sorted(int(r[0]) for r in self.conn.execute(sql, params).fetchall())

    def gather(self, kind: str, **kwargs):
        """plan_album for one face kind, with the CLI chatter swallowed."""
        with redirect_stdout(io.StringIO()):
            return plan_album(self.cfg, self.conn, kind, "", self.dest, **kwargs)


class TestTheSliceExpression(FaceAlbumTestBase):
    def test_the_marker_row_is_not_a_face_in_any_of_the_three(self):
        marked = self.add_frame("marker.jpg")
        self.add_face_marker(marked)
        real = self.add_frame("real.jpg")
        self.add_square_face(real, 500)
        self.assertEqual(self.selected("people"), [real])
        self.assertEqual(self.selected("portrait"), [real])
        self.assertEqual(self.selected("group"), [])

    def test_each_file_appears_once_however_many_faces_it_holds(self):
        # The expression composes both as `IN (…)` and as something a caller counts, so
        # one row per file is a property and not an accident of the current call site.
        fid = self.add_frame("crowd.jpg")
        for _ in range(4):
            self.add_face(fid)
        self.assertEqual(self.selected("people"), [fid])
        self.assertEqual(self.selected("group"), [fid])

    def test_group_reads_its_threshold_from_the_config(self):
        pair = self.add_frame("pair.jpg")
        self.add_face(pair)
        self.add_face(pair)
        self.assertEqual(self.selected("group"), [])
        self.cfg.features = dataclasses.replace(self.cfg.features, group_photo_faces=2)
        self.assertEqual(self.selected("group"), [pair])

    def test_portrait_is_one_face_over_a_share_of_the_frame(self):
        big = self.add_frame("big.jpg")
        self.add_square_face(big, 400)              # 0.16 of a 1000x1000 frame
        small = self.add_frame("small.jpg")
        self.add_square_face(small, 100)            # 0.01
        two = self.add_frame("two.jpg")
        self.add_square_face(two, 500)
        self.add_square_face(two, 500, at=500)
        self.assertEqual(self.selected("portrait"), [big])

    def test_portrait_ignores_the_order_of_the_bbox_corners(self):
        # abs() rather than a subtraction that trusts the corner order: a negative area
        # would silently drop the frame instead of failing.
        fid = self.add_frame("flipped.jpg")
        self.add_face(fid, "[400.0,400.0,0.0,0.0]")
        self.assertEqual(self.selected("portrait"), [fid])

    def test_a_frame_without_dimensions_cannot_be_a_portrait(self):
        fid = self.add_file("nodims.jpg")           # width/height stay NULL
        self.add_square_face(fid, 900)
        self.assertEqual(self.selected("portrait"), [])
        self.assertEqual(self.selected("people"), [fid])

    def test_an_unknown_slice_is_refused(self):
        with self.assertRaises(ValueError):
            face_slice_ids_sql(self.cfg, "children")


class TestFaceAlbumSelection(FaceAlbumTestBase):
    def test_the_album_gathers_exactly_what_the_expression_selects(self):
        marked = self.add_frame("marker.jpg")
        self.add_face_marker(marked)
        real = self.add_frame("real.jpg")
        self.add_square_face(real, 500)
        report = self.gather("people", apply=False)
        self.assertEqual([it.file_id for it in report.plan], [real])

    def test_duplicates_and_unreadable_files_stay_out(self):
        canonical = self.add_frame("a.jpg")
        duplicate = self.add_frame("b.jpg")
        broken = self.add_frame("c.jpg")
        self.conn.execute("UPDATE files SET dup_of = ? WHERE id = ?",
                          (canonical, duplicate))
        self.conn.execute("UPDATE files SET error = 'cannot read' WHERE id = ?",
                          (broken,))
        self.conn.commit()
        for fid in (canonical, duplicate, broken):
            self.add_square_face(fid, 500)
        report = self.gather("portrait", apply=False)
        self.assertEqual([it.file_id for it in report.plan], [canonical])

    def test_the_selector_is_ignored_rather_than_used(self):
        fid = self.add_frame("a.jpg")
        self.add_face(fid)
        with redirect_stdout(io.StringIO()):
            empty = plan_album(self.cfg, self.conn, "people", "", self.dest)
            noise = plan_album(self.cfg, self.conn, "people", "whatever", self.dest)
        self.assertEqual([it.file_id for it in empty.plan],
                         [it.file_id for it in noise.plan])

    def test_where_still_narrows_the_slice(self):
        paris = self.add_frame("a.jpg", country="France", city="Paris")
        moscow = self.add_frame("b.jpg", country="Russia", city="Moskva")
        self.add_face(paris)
        self.add_face(moscow)
        report = self.gather("people", where=["city=Paris"], apply=False)
        self.assertEqual([it.file_id for it in report.plan], [paris])

    def test_default_album_names_come_from_the_folder_catalog(self):
        fid = self.add_frame("a.jpg")
        for i in range(3):
            self.add_square_face(fid, 500, at=i)
        for kind, key in (("people", "people"), ("group", "group_photos")):
            with self.subTest(kind=kind):
                report = self.gather(kind, apply=False)
                self.assertEqual(report.album_name, i18n.folder(key, "en"))

    def test_default_album_name_follows_the_configured_language(self):
        self.cfg.raw = {"language": "ru"}
        fid = self.add_frame("a.jpg")
        self.add_square_face(fid, 500)
        report = self.gather("portrait", apply=False)
        self.assertEqual(report.album_name, i18n.folder("portraits", "ru"))

    def test_empty_slice_does_not_crash_or_journal(self):
        report = self.gather("group", apply=True)
        self.assertEqual(report.plan, [])
        self.assertIsNone(report.batch_id)
        self.assertFalse(self.dest.exists())


class TestFaceAlbumApply(FaceAlbumTestBase):
    def test_dry_run_writes_nothing_to_the_db_or_the_filesystem(self):
        fid = self.add_frame("a.jpg")
        self.add_square_face(fid, 500)
        report = self.gather("portrait", mode="link", apply=False)
        self.assertEqual(len(report.plan), 1)
        self.assertFalse(self.dest.exists())
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM move_batches").fetchone()[0], 0)

    def test_apply_link_journals_with_the_album_mode_of_its_kind(self):
        fid = self.add_frame("sub/deep/a.jpg", content=b"faces")
        for i in range(3):
            self.add_square_face(fid, 50, at=i * 60)
        report = self.gather("group", mode="link", apply=True)
        self.assertEqual(report.transferred, 1)
        dst = self.dest / i18n.folder("group_photos", "en") / "a.jpg"
        self.assertTrue(dst.exists())
        self.assertGreaterEqual(os.stat(dst).st_nlink, 2)  # a hardlink, not a copy
        batch = self.conn.execute(
            "SELECT mode, operation FROM move_batches WHERE id = ?",
            (report.batch_id,)).fetchone()
        self.assertEqual(batch["mode"], "album_group")
        self.assertEqual(batch["operation"], "link")
        move = self.conn.execute(
            "SELECT file_id, dst, status FROM moves WHERE batch_id = ?",
            (report.batch_id,)).fetchone()
        self.assertEqual(move["file_id"], fid)
        self.assertEqual(move["status"], "done")
        self.assertEqual(Path(move["dst"]), dst)

    def test_apply_move_takes_the_file_out_of_the_canon(self):
        fid = self.add_frame("a.jpg", content=b"faces")
        self.add_square_face(fid, 500)
        report = self.gather("portrait", mode="move", apply=True)
        self.assertEqual(report.transferred, 1)
        dst = self.dest / i18n.folder("portraits", "en") / "a.jpg"
        self.assertTrue(dst.exists())
        self.assertFalse((self.src_dir / "a.jpg").exists())
        self.assertEqual(Path(self.path_of(fid)), dst)

    def test_apply_copy_leaves_the_canonical_original_in_place(self):
        fid = self.add_frame("a.jpg", content=b"faces")
        self.add_square_face(fid, 500)
        report = self.gather("people", mode="copy", apply=True)
        self.assertEqual(report.transferred, 1)
        self.assertEqual(self.path_of(fid), str((self.src_dir / "a.jpg").resolve()))

    def test_gathering_the_same_album_twice_makes_no_underscore_one_copies(self):
        fid = self.add_frame("a.jpg", content=b"faces")
        self.add_square_face(fid, 500)
        self.gather("people", mode="link", apply=True)
        second = self.gather("people", mode="link", apply=True)
        self.assertEqual(second.skipped_already_copied, 1)
        self.assertEqual(second.transferred, 0)
        album_dir = self.dest / i18n.folder("people", "en")
        self.assertEqual(sorted(p.name for p in album_dir.iterdir()), ["a.jpg"])


if __name__ == "__main__":
    unittest.main()
