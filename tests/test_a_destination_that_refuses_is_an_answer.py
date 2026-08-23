"""F249: a destination that will not take a folder is an answer, not a traceback.

Three things are held here, in the order the failure happens.

1. Before the first file moves, the destination is asked whether it accepts a directory.
   A refusal stops the run with nothing touched — no batch, no file, no half-laid-out
   collection.
2. A refusal on the hundredth file leaves the thread alive, the state closed and an
   answer that says how many files did move.
3. Those files are in the journal, so `undo` takes them back. The journal is committed
   before every operation, which is why this should hold — but nothing checked it on a
   failure, and that is what makes it worth a test rather than an assumption.
"""
from __future__ import annotations

import errno
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from sorta import sorter, ui
from sorta.config import Config
from sorta.sorter import DestinationRefused, check_dest_writable
from sorta.ui.strings import _UI_STRINGS

from tests.test_ui_sort import SortTestBase, _poll_until

_LANGS = ("ru", "en", "ja")
_APP_JS = Path(sorter.__file__).resolve().parent / "web" / "app" / "app.js"


def _refusing_transfer(after: int, exc: OSError):
    """A `_transfer` double that does the real thing `after` times, then raises `exc`.

    The refusal is a plain `OSError` and not a `TransferError`: that is the shape the
    engine does NOT catch (`_transfer` makes the target directory before it copies
    anything), and the shape that reached the owner's log as a bare traceback.
    """
    real = sorter._transfer
    calls = {"n": 0}

    def double(src, dst, src_hash=None, copy=False, link=False):
        calls["n"] += 1
        if calls["n"] > after:
            raise exc
        return real(src, dst, src_hash, copy=copy, link=link)

    return double, calls


class DestinationRefusalBase(SortTestBase):
    def moves(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT batch_id, status, src, dst FROM moves ORDER BY id").fetchall()
        return [dict(r) for r in rows]

    def batches(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT id, finished_at FROM move_batches ORDER BY id").fetchall()
        return [dict(r) for r in rows]

    def run_sort(self, dest: Path | str, mode: str = "move") -> dict:
        status, resp = self.post("/api/sort", {"dest": str(dest), "mode": mode})
        self.assertEqual(status, 200)
        self.assertTrue(resp.get("ok"))
        return _poll_until(self.sort_status, lambda d: d["finished"])


class TestTheProbeAnswersForTheDestination(unittest.TestCase):
    """`check_dest_writable` on its own: what it leaves behind and what it raises."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def test_a_destination_that_does_not_exist_yet_is_left_not_existing(self):
        dest = self.root / "photos" / "sorted"
        check_dest_writable(dest)
        self.assertFalse(dest.exists())
        self.assertFalse((self.root / "photos").exists())

    def test_an_existing_destination_keeps_exactly_what_it_had(self):
        dest = self.root / "sorted"
        dest.mkdir()
        (dest / "already.txt").write_text("x", encoding="utf-8")
        check_dest_writable(dest)
        self.assertEqual([p.name for p in dest.iterdir()], ["already.txt"])

    def test_a_destination_under_a_file_refuses_and_names_itself(self):
        blocker = self.root / "blocker"
        blocker.write_bytes(b"")
        dest = blocker / "sorted"
        with self.assertRaises(DestinationRefused) as caught:
            check_dest_writable(dest)
        self.assertEqual(caught.exception.params["dest"], str(dest))
        self.assertTrue(caught.exception.params["error"])
        self.assertIn(str(dest), str(caught.exception))

    def test_the_refusal_is_an_oserror_with_a_code(self):
        exc = DestinationRefused("no", "sort_dest_refused", dest="/b", error="denied")
        self.assertIsInstance(exc, OSError)
        self.assertEqual(exc.code, "sort_dest_refused")

    def test_a_permission_error_from_mkdir_is_the_refusal(self):
        """The live failure of the brief: WinError 5 on the destination's own folder."""
        dest = self.root / "sorted"
        denied = PermissionError(errno.EACCES, "Permission denied", str(dest))
        with mock.patch.object(Path, "mkdir", side_effect=denied):
            with self.assertRaises(DestinationRefused) as caught:
                check_dest_writable(dest)
        self.assertIn("Permission denied", str(caught.exception))


class TestNothingMovesWhenTheDestinationRefuses(DestinationRefusalBase):
    """Acceptance 1."""

    def test_not_one_file_is_touched_and_the_answer_says_which_folder(self):
        _fid, first, _c = self.add_photo_file("a.jpg", country="ru", city="Moscow")
        _fid2, second, _c2 = self.add_photo_file("b.jpg", country="ru", city="Moscow")
        blocker = self.root / "blocker"
        blocker.write_bytes(b"")
        self.start_server()

        final = self.run_sort(blocker / "dest")

        self.assertIsNone(final["result"])
        self.assertEqual(final["error_code"], "sort_dest_refused")
        self.assertEqual(final["error_params"]["dest"], str(blocker / "dest"))
        self.assertTrue(first.exists())
        self.assertTrue(second.exists())
        self.assertEqual(self.moves(), [])
        self.assertEqual(self.batches(), [])

    def test_the_server_is_alive_and_the_button_works_again(self):
        self.add_photo_file("a.jpg", country="ru", city="Moscow")
        blocker = self.root / "blocker"
        blocker.write_bytes(b"")
        self.start_server()

        self.run_sort(blocker / "dest")
        status, _body, _ctype = self.get("/")
        self.assertEqual(status, 200)

        final = self.run_sort(self.root / "dest")
        self.assertIsNone(final["error"])
        self.assertEqual(final["result"]["moved"], 1)

    def test_a_probe_that_went_through_leaves_the_destination_clean(self):
        """Acceptance 7, the half nobody would notice: the probe is not a folder the
        layout has to explain afterwards."""
        self.add_photo_file("a.jpg", country="ru", city="Moscow")
        self.start_server()
        dest = self.root / "dest"

        final = self.run_sort(dest, mode="copy")

        self.assertIsNone(final["error"])
        leftovers = [p for p in dest.rglob("*")
                     if p.name.startswith(sorter._DEST_PROBE_PREFIX)]
        self.assertEqual(leftovers, [])


class TestARefusalHalfwayIsStillAnAnswer(DestinationRefusalBase):
    """Acceptance 2 and 4 — the two the feature is actually for."""

    def stop_after_one(self, exc: OSError) -> dict:
        self.add_photo_file("a.jpg", country="ru", city="Moscow")
        self.add_photo_file("b.jpg", country="ru", city="Moscow")
        self.start_server()
        double, _calls = _refusing_transfer(1, exc)
        with mock.patch.object(sorter, "_transfer", double):
            return self.run_sort(self.root / "dest")

    def test_a_permission_error_says_the_path_the_count_and_the_reason(self):
        final = self.stop_after_one(
            PermissionError(errno.EACCES, "Permission denied", str(self.root / "dest")))

        self.assertFalse(final["running"])
        self.assertTrue(final["finished"])
        self.assertEqual(final["error_code"], "sort_stopped_by_filesystem")
        self.assertEqual(final["error_params"]["moved"], 1)
        self.assertEqual(final["error_params"]["path"], str(self.root / "dest"))
        self.assertIn("Permission denied", final["error_params"]["error"])

    def test_a_full_disk_is_an_answer_too(self):
        final = self.stop_after_one(
            OSError(errno.ENOSPC, "No space left on device", str(self.root / "dest")))

        self.assertEqual(final["error_code"], "sort_stopped_by_filesystem")
        self.assertIn("No space left on device", final["error_params"]["error"])
        self.assertEqual(final["error_params"]["moved"], 1)

    def test_the_server_survives_the_refusal(self):
        self.stop_after_one(PermissionError(errno.EACCES, "denied", "x"))
        status, _body, _ctype = self.get("/")
        self.assertEqual(status, 200)

    def test_the_count_is_read_off_the_journal_and_not_off_an_older_batch(self):
        """An earlier layout of the same collection must not be counted into this one's
        sentence — the number is what THIS run moved."""
        self.add_photo_file("a.jpg", country="ru", city="Moscow")
        self.add_photo_file("b.jpg", country="ru", city="Moscow")
        self.start_server()
        first = self.run_sort(self.root / "first", mode="copy")
        self.assertEqual(first["result"]["moved"], 2)

        double, _calls = _refusing_transfer(0, PermissionError(errno.EACCES, "no", "x"))
        with mock.patch.object(sorter, "_transfer", double):
            final = self.run_sort(self.root / "second", mode="copy")

        self.assertEqual(final["error_params"]["moved"], 0)


class TestWhatMovedCanStillComeBack(DestinationRefusalBase):
    """Acceptance 3: the journal after a refusal halfway is one `undo` can read."""

    def test_undo_returns_the_files_that_had_already_moved(self):
        _fid, first, _c = self.add_photo_file("a.jpg", country="ru", city="Moscow")
        _fid2, second, _c2 = self.add_photo_file("b.jpg", country="ru", city="Moscow")
        self.start_server()
        double, _calls = _refusing_transfer(
            1, PermissionError(errno.EACCES, "Permission denied", "x"))
        with mock.patch.object(sorter, "_transfer", double):
            final = self.run_sort(self.root / "dest", mode="move")

        self.assertEqual(final["error_params"]["moved"], 1)
        gone = [p for p in (first, second) if not p.exists()]
        self.assertEqual(len(gone), 1)

        stats = sorter.undo(self.conn)
        self.assertEqual(stats.undone, 1)
        self.assertTrue(first.exists())
        self.assertTrue(second.exists())
        self.assertEqual(
            {r["path"] for r in self.conn.execute("SELECT path FROM files")},
            {str(first), str(second)})

    def test_the_moves_tab_can_see_the_unfinished_batch(self):
        """What makes the roll back button reachable at all: the manifest the page reads
        after a failure has the batch and its rows."""
        self.add_photo_file("a.jpg", country="ru", city="Moscow")
        self.add_photo_file("b.jpg", country="ru", city="Moscow")
        self.start_server()
        double, _calls = _refusing_transfer(
            1, PermissionError(errno.EACCES, "Permission denied", "x"))
        with mock.patch.object(sorter, "_transfer", double):
            self.run_sort(self.root / "dest", mode="move")

        _status, body, _ctype = self.get("/api/moves")
        payload = json.loads(body)
        self.assertIsNotNone(payload["batch"])
        self.assertEqual(sum(1 for m in payload["moves"] if m["status"] == "done"), 1)


class TestTheOldAnswersDidNotMove(DestinationRefusalBase):
    """Acceptance 6 and 7."""

    def test_an_in_place_layout_with_two_sources_answers_as_it_always_did(self):
        other = self.root / "src2"
        other.mkdir()
        self.cfg = Config(sources=[self.src_dir, other],
                          database=self.cfg.database, raw={})
        self.add_photo_file("a.jpg", country="ru", city="Moscow")
        self.start_server()

        status, resp = self.post("/api/sort", {"dest": "", "mode": "move"})
        self.assertEqual(status, 200)
        self.assertTrue(resp.get("ok"))
        final = _poll_until(self.sort_status, lambda d: d["finished"])

        self.assertIn("in-place layout needs a single source", final["error"])
        self.assertIsNone(final["error_code"])
        self.assertEqual(final["error_params"], {})

    def test_a_layout_that_goes_through_reports_exactly_what_it_used_to(self):
        _fid, src, _c = self.add_photo_file("a.jpg", country="ru", city="Moscow")
        self.start_server()
        dest = self.root / "dest"

        final = self.run_sort(dest, mode="copy")

        self.assertIsNone(final["error"])
        self.assertIsNone(final["error_code"])
        self.assertEqual(final["error_params"], {})
        self.assertEqual(final["result"]["moved"], 1)
        self.assertEqual(final["result"]["failed"], 0)
        self.assertTrue(src.exists())
        self.assertEqual(len(list(dest.rglob("*.jpg"))), 1)

    def test_an_in_place_layout_with_one_source_still_runs(self):
        """The destination probed for an in-place run is the source folder itself — a
        preflight that got that wrong would refuse the whole in-place mode."""
        _fid, src, _c = self.add_photo_file("a.jpg", country="ru", city="Moscow")
        self.start_server()

        status, _resp = self.post("/api/sort", {"dest": "", "mode": "move"})
        self.assertEqual(status, 200)
        final = _poll_until(self.sort_status, lambda d: d["finished"])

        self.assertIsNone(final["error"])
        self.assertTrue(final["result"]["in_place"])
        self.assertFalse(src.exists())


class TestTheSentencesExistInThreeLanguages(unittest.TestCase):
    """Acceptance 5."""

    def test_both_codes_are_translated_and_keep_their_values(self):
        expected = {"fault_sort_dest_refused": ("{dest}", "{error}"),
                    "fault_sort_stopped_by_filesystem": ("{path}", "{error}", "{moved}")}
        for key, fields in expected.items():
            entry = _UI_STRINGS[key]
            self.assertEqual(set(entry), set(_LANGS))
            for lang in _LANGS:
                for field in fields:
                    with self.subTest(key=key, lang=lang, field=field):
                        self.assertIn(field, entry[lang])

    def test_no_english_sentence_is_spelled_into_the_page(self):
        source = _APP_JS.read_text(encoding="utf-8")
        self.assertIn('I18N["fault_" + data.error_code]', source)
        self.assertIn("sortErrorText(data)", source)
        self.assertIn("I18N.sort_undo_hint", source)


class TestTheStateCarriesTheRefusal(unittest.TestCase):
    def test_finish_and_the_snapshot_carry_the_code_and_the_values(self):
        state = ui.layout._SortState()
        state.try_start()
        state.finish("the destination does not accept a folder: /b: denied", None,
                     "sort_dest_refused", {"dest": "/b", "error": "denied"})
        snapshot = state.snapshot()
        self.assertEqual(snapshot["error_code"], "sort_dest_refused")
        self.assertEqual(snapshot["error_params"], {"dest": "/b", "error": "denied"})
        # It travels to the browser as JSON: a Path or an exception in `params` would be
        # a 500 at the moment of the failure it describes.
        self.assertEqual(json.loads(json.dumps(snapshot))["error_params"],
                         {"dest": "/b", "error": "denied"})

    def test_a_second_run_starts_without_the_previous_refusal(self):
        state = ui.layout._SortState()
        state.try_start()
        state.finish("boom", None, "sort_dest_refused", {"dest": "/b"})
        state.try_start()
        self.assertIsNone(state.snapshot()["error_code"])
        self.assertEqual(state.snapshot()["error_params"], {})


if __name__ == "__main__":
    unittest.main()
