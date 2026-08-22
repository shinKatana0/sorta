"""F241: the OS trash may refuse, and then nothing is deleted — not even half of it."""
from __future__ import annotations

import errno
import os
import unittest
from pathlib import Path
from unittest import mock

from send2trash.exceptions import TrashPermissionError

from sorta.ui import common
from sorta.ui.strings import _UI_STRINGS

from tests.test_ui_dupes import DupesTestBase


def _no_bin_anywhere():
    """A `send_to_trash` double for a volume with no trash: everything raises."""
    def double(path):
        raise TrashPermissionError(path)
    return double


def _refusing_only(*paths: str):
    """A `send_to_trash` double that raises for `paths` and takes everything else.

    The preflight probe is "everything else" here — that is the point: the volume
    answers yes and one individual file still will not go.
    """
    doomed = set(paths)

    def double(path):
        if path in doomed:
            raise PermissionError(errno.EACCES, "Permission denied", path)
    return double


class TrashRefusalBase(DupesTestBase):
    def add_file(self, rel: str) -> tuple[int, Path]:
        file_id = self.add_dupe(rel, phash="0" * 16, width=100, height=100, size=1000)
        row = self.conn.execute(
            "SELECT path FROM files WHERE id = ?", (file_id,)).fetchone()
        return file_id, Path(row["path"])

    def probe_paths(self, mock_trash) -> list[str]:
        """The other half of `trashed_paths`: the preflight probes and only those."""
        return [call.args[0] for call in mock_trash.call_args_list
                if Path(call.args[0]).name.startswith(common._TRASH_PROBE_PREFIX)]

    def file_ids(self) -> set[int]:
        return {r["id"] for r in self.conn.execute("SELECT id FROM files").fetchall()}


class TestAVolumeWithoutATrash(TrashRefusalBase):
    def test_nothing_is_touched_and_the_answer_says_why(self):
        first, first_path = self.add_file("a.jpg")
        second, second_path = self.add_file("b.jpg")
        self.start_server()

        with mock.patch("sorta.ui.common.send_to_trash",
                        side_effect=_no_bin_anywhere()) as trash, \
                mock.patch("sorta.imaging.preview_delete") as preview:
            status, payload = self.post("/api/photos/trash",
                                        {"file_ids": [first, second]})

        self.assertEqual(status, 200)
        self.assertEqual(payload["trashed"], [])
        self.assertEqual({r["file_id"] for r in payload["refused"]}, {first, second})
        self.assertEqual({r["reason"] for r in payload["refused"]},
                         {common.TRASH_REFUSED_NO_BIN})
        self.assertEqual(self.trashed_paths(trash), [])
        preview.assert_not_called()
        self.assertEqual(self.file_ids(), {first, second})
        self.assertTrue(first_path.exists())
        self.assertTrue(second_path.exists())

    def test_the_refusal_names_every_file_it_kept(self):
        first, _ = self.add_file("a.jpg")
        second, _ = self.add_file("b.jpg")
        self.start_server()

        with mock.patch("sorta.ui.common.send_to_trash", side_effect=_no_bin_anywhere()):
            _status, payload = self.post("/api/photos/trash",
                                         {"file_ids": [first, second]})

        self.assertEqual({r["name"] for r in payload["refused"]}, {"a.jpg", "b.jpg"})

    def test_a_refused_request_may_be_repeated(self):
        file_id, path = self.add_file("a.jpg")
        self.start_server()

        with mock.patch("sorta.ui.common.send_to_trash", side_effect=_no_bin_anywhere()):
            self.post("/api/photo/trash", {"file_id": file_id})
            _status, payload = self.post("/api/photo/trash", {"file_id": file_id})

        self.assertEqual([r["file_id"] for r in payload["refused"]], [file_id])
        self.assertEqual(self.file_ids(), {file_id})
        self.assertTrue(path.exists())

    def test_a_deletion_that_went_through_may_be_repeated(self):
        file_id, _ = self.add_file("a.jpg")
        self.start_server()

        with mock.patch("sorta.ui.common.send_to_trash"):
            self.post("/api/photo/trash", {"file_id": file_id})
            _status, payload = self.post("/api/photo/trash", {"file_id": file_id})

        self.assertEqual(payload, {"trashed": [], "refused": []})


class TestOneFileThatWillNotGo(TrashRefusalBase):
    def test_the_others_still_go_and_the_stuck_one_stays_whole(self):
        stuck, stuck_path = self.add_file("stuck.jpg")
        loose, loose_path = self.add_file("loose.jpg")
        self.start_server()

        with mock.patch("sorta.ui.common.send_to_trash",
                        side_effect=_refusing_only(str(stuck_path))), \
                mock.patch("sorta.imaging.preview_delete") as preview:
            status, payload = self.post("/api/photos/trash", {"file_ids": [stuck, loose]})

        self.assertEqual(status, 200)
        self.assertEqual([t["file_id"] for t in payload["trashed"]], [loose])
        self.assertEqual([r["file_id"] for r in payload["refused"]], [stuck])
        self.assertEqual(payload["refused"][0]["reason"], common.TRASH_REFUSED_PERMISSION)
        # The DELETE runs over what left, not over what was asked for.
        self.assertEqual(self.file_ids(), {stuck})
        self.assertTrue(stuck_path.exists())
        # And the frame that stayed keeps its preview.
        self.assertEqual([call.args[0] for call in preview.call_args_list],
                         [str(loose_path)])

    def test_the_stuck_file_keeps_its_dedup_choice_row(self):
        stuck, stuck_path = self.add_file("stuck.jpg")
        self.conn.execute(
            "INSERT INTO dedup_choice (file_id, action, updated_at) "
            "VALUES (?, 'to_delete', 'now')", (stuck,))
        self.conn.commit()
        self.start_server()

        with mock.patch("sorta.ui.common.send_to_trash",
                        side_effect=_refusing_only(str(stuck_path))):
            self.post("/api/photo/trash", {"file_id": stuck})

        self.assertIsNotNone(self.conn.execute(
            "SELECT file_id FROM dedup_choice WHERE file_id = ?", (stuck,)).fetchone())


class TestEveryRefusalIsInTheLog(TrashRefusalBase):
    def test_a_volume_without_a_trash_names_the_path_and_the_reason(self):
        file_id, path = self.add_file("a.jpg")
        self.start_server()

        with self.assertLogs("sorta.ui", level="WARNING") as logs, \
                mock.patch("sorta.ui.common.send_to_trash",
                           side_effect=_no_bin_anywhere()):
            self.post("/api/photo/trash", {"file_id": file_id})

        line = [m for m in logs.output if str(path) in m]
        self.assertEqual(len(line), 1)
        self.assertIn(common.TRASH_REFUSED_NO_BIN, line[0])

    def test_a_single_file_that_stuck_names_the_path_and_the_reason(self):
        file_id, path = self.add_file("a.jpg")
        self.start_server()

        with self.assertLogs("sorta.ui", level="WARNING") as logs, \
                mock.patch("sorta.ui.common.send_to_trash",
                           side_effect=_refusing_only(str(path))):
            self.post("/api/photo/trash", {"file_id": file_id})

        line = [m for m in logs.output if str(path) in m]
        self.assertEqual(len(line), 1)
        self.assertIn(common.TRASH_REFUSED_PERMISSION, line[0])


class TestTheProbeTidiesUpAfterItself(TrashRefusalBase):
    def probe_leftovers(self) -> list[Path]:
        return [p for p in self.src_dir.rglob("*")
                if p.name.startswith(common._TRASH_PROBE_PREFIX)]

    def test_nothing_is_left_behind_when_the_volume_accepts(self):
        file_id, _ = self.add_file("a.jpg")
        self.start_server()

        with mock.patch("sorta.ui.common.send_to_trash"):
            self.post("/api/photo/trash", {"file_id": file_id})

        self.assertEqual(self.probe_leftovers(), [])

    def test_nothing_is_left_behind_when_the_probe_raises(self):
        file_id, _ = self.add_file("a.jpg")
        self.start_server()

        with mock.patch("sorta.ui.common.send_to_trash", side_effect=_no_bin_anywhere()):
            self.post("/api/photo/trash", {"file_id": file_id})

        self.assertEqual(self.probe_leftovers(), [])

    def test_a_directory_that_cannot_hold_a_probe_is_not_a_verdict(self):
        self.assertIsNone(common._volume_accepts_trash(self.root / "no-such-folder"))


class TestOneProbePerVolume(TrashRefusalBase):
    def test_two_folders_of_one_volume_cost_one_probe(self):
        first, _ = self.add_file("one/a.jpg")
        second, _ = self.add_file("two/b.jpg")
        self.start_server()

        with mock.patch("sorta.ui.common.send_to_trash") as trash:
            self.post("/api/photos/trash", {"file_ids": [first, second]})

        self.assertEqual(len(self.probe_paths(trash)), 1)
        self.assertEqual(len(self.trashed_paths(trash)), 2)

    def test_two_volumes_cost_two_probes(self):
        """A machine with two volumes is not something a test may require, so the key
        function stands in for one — what is checked here is that the cache is keyed by
        it at all. Its own answers are TestVolumeKey's business."""
        first, _ = self.add_file("one/a.jpg")
        second, _ = self.add_file("two/b.jpg")
        self.start_server()

        with mock.patch("sorta.ui.common.send_to_trash") as trash, \
                mock.patch("sorta.ui.common._trash_volume_key",
                           side_effect=lambda path: str(Path(path).parent)):
            self.post("/api/photos/trash", {"file_ids": [first, second]})

        self.assertEqual(len(self.probe_paths(trash)), 2)


_TRASH_ROUTES = ("/api/photo/trash", "/api/photos/trash", "/api/dupes/trash")


class TestEveryRouteAnswersTheSame(TrashRefusalBase):
    """The three trash routes share `_trash_files`; this proves it rather than assuming."""

    def bodies(self, doomed: int, keeper: int) -> dict[str, dict]:
        return {
            "/api/photo/trash": {"file_id": doomed},
            "/api/photos/trash": {"file_ids": [doomed]},
            "/api/dupes/trash": {"group": [keeper, doomed], "keep_file_id": keeper},
        }

    def test_a_volume_without_a_trash_stops_all_three(self):
        self.start_server()
        for n, route in enumerate(_TRASH_ROUTES):
            with self.subTest(route=route):
                doomed, doomed_path = self.add_file(f"{n}/doomed.jpg")
                keeper, _ = self.add_file(f"{n}/keeper.jpg")
                with mock.patch("sorta.ui.common.send_to_trash",
                                side_effect=_no_bin_anywhere()):
                    status, payload = self.post(route, self.bodies(doomed, keeper)[route])
                self.assertEqual(status, 200)
                self.assertEqual(payload["trashed"], [])
                self.assertEqual([r["file_id"] for r in payload["refused"]], [doomed])
                self.assertEqual(payload["refused"][0]["reason"],
                                 common.TRASH_REFUSED_NO_BIN)
                self.assertIn(doomed, self.file_ids())
                self.assertTrue(doomed_path.exists())

    def test_all_three_name_both_halves_when_the_trash_takes_the_file(self):
        self.start_server()
        for n, route in enumerate(_TRASH_ROUTES):
            with self.subTest(route=route):
                doomed, _ = self.add_file(f"ok{n}/doomed.jpg")
                keeper, _ = self.add_file(f"ok{n}/keeper.jpg")
                with mock.patch("sorta.ui.common.send_to_trash"):
                    status, payload = self.post(route, self.bodies(doomed, keeper)[route])
                self.assertEqual(status, 200)
                self.assertEqual([t["file_id"] for t in payload["trashed"]], [doomed])
                self.assertEqual(payload["refused"], [])
                self.assertNotIn(doomed, self.file_ids())


class TestTheScreenSaysWhatStayed(TrashRefusalBase):
    def page(self) -> str:
        self.start_server()
        _status, body, _ctype = self.get("/")
        return body.decode("utf-8")

    def test_all_three_delete_flows_report_a_refusal(self):
        html = self.page()
        self.assertEqual(html.count("function reportTrashRefusal("), 1)
        for route in _TRASH_ROUTES:
            with self.subTest(route=route):
                after = html.split(f'postJson("{route}"', 1)[1][:400]
                self.assertIn("reportTrashRefusal(", after)

    def test_the_reason_the_page_branches_on_is_the_one_the_server_sends(self):
        self.assertIn(f'"{common.TRASH_REFUSED_NO_BIN}"', self.page())

    def test_no_english_sentence_is_spelled_in_the_markup(self):
        html = self.page()
        self.assertIn("I18N.trash_no_bin_on_volume", html)
        self.assertIn("I18N.trash_partly_refused", html)


class TestTheRefusalSentencesAreTranslated(unittest.TestCase):
    def test_both_are_in_all_three_languages(self):
        for key in ("trash_no_bin_on_volume", "trash_partly_refused"):
            with self.subTest(key=key):
                self.assertEqual(set(_UI_STRINGS[key]), {"ru", "en", "ja"})

    def test_the_partial_refusal_lists_the_files_in_every_language(self):
        for lang in ("ru", "en", "ja"):
            self.assertIn("{names}", _UI_STRINGS["trash_partly_refused"][lang])


class TestVolumeKey(unittest.TestCase):
    def test_two_folders_of_one_disk_share_a_key(self):
        here = Path(__file__).resolve()
        self.assertEqual(common._trash_volume_key(str(here.parent / "a.jpg")),
                         common._trash_volume_key(str(here.parent.parent / "b.jpg")))

    @unittest.skipUnless(os.name == "nt", "drive letters and UNC shares are a Windows shape")
    def test_drives_and_shares_are_separate_volumes(self):
        keys = {common._trash_volume_key(p) for p in
                (r"C:\photos\a.jpg", r"D:\photos\a.jpg", r"\\nas\photos\a.jpg")}
        self.assertEqual(len(keys), 3)
        self.assertEqual(common._trash_volume_key(r"c:\photos\a.jpg"),
                         common._trash_volume_key(r"C:\OTHER\b.jpg"))

    @unittest.skipIf(os.name == "nt", "a path without a drive letter")
    def test_a_path_answers_with_a_mount_point_it_lies_under(self):
        key = common._trash_volume_key(str(Path(__file__).resolve()))
        self.assertTrue(os.path.ismount(key))
        self.assertTrue(str(Path(__file__).resolve()).startswith(key))


class TestRefusalReason(unittest.TestCase):
    def test_a_trash_permission_error_means_the_volume_has_no_bin(self):
        self.assertEqual(common._refusal_reason(TrashPermissionError("a.jpg")),
                         common.TRASH_REFUSED_NO_BIN)

    def test_a_busy_file_is_in_use(self):
        self.assertEqual(common._refusal_reason(OSError(errno.EBUSY, "busy", "a.jpg")),
                         common.TRASH_REFUSED_IN_USE)

    @unittest.skipUnless(os.name == "nt", "ERROR_SHARING_VIOLATION is a Windows code")
    def test_a_sharing_violation_is_in_use(self):
        exc = OSError(errno.EACCES, "in use", "a.jpg", common._WIN_SHARING_VIOLATION)
        self.assertEqual(common._refusal_reason(exc), common.TRASH_REFUSED_IN_USE)

    def test_a_plain_permission_error_is_permission(self):
        exc = PermissionError(errno.EACCES, "Permission denied", "a.jpg")
        self.assertEqual(common._refusal_reason(exc), common.TRASH_REFUSED_PERMISSION)

    def test_anything_else_is_a_bare_failure(self):
        self.assertEqual(common._refusal_reason(OSError(errno.ENOENT, "gone", "a.jpg")),
                         common.TRASH_REFUSED_FAILED)


if __name__ == "__main__":
    unittest.main()
