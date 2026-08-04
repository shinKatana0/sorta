"""F177: the "is there a subject" question is gone — prompt, slice, album and answers.

The question was asked for the first and only time on 2026-08-03: 6,111 frames, 212 of
them called subjectless. Looked at by eye, those 212 are ordinary photographs — city
shots and studio work alike — so the signal separates nothing, exactly as `is_accidental`
turned out not to (F122, measured at 5% precision and retired).

Four properties, and the third is the one that cannot be skipped:

* the prompt no longer carries the question and the parser no longer looks for it, so a
  model that volunteers `subject` or `no_subject` anyway gets no column for it;
* the slice and the album are DELETED rather than hidden — a hidden slice comes back at
  the first edit of the file that hides it — and `no_subject` is now an unknown album
  kind, refused out loud;
* THE STORED ANSWERS ARE ERASED. Nothing else would ever reach them: the question is out
  of the prompt so the stage cannot overwrite them, `vlm.quality` is off so the stage does
  not run, and a stale fingerprint only means "recompute", never "this stored answer is
  wrong". Without the migration the slice would keep listing 212 frames of a question
  nobody asks;
* and the eyes answers SURVIVE that migration. They are the only quality answers a person
  has checked by eye, and they are one `UPDATE` away from being lost silently.
"""
from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from sorta import i18n, junk, sorter, ui
from sorta.db import SCHEMA_VERSION, connect
from sorta.junk import QualityFlags, parse_quality_answer

from tests.schema_history import roll_back_to
from tests.test_docs_guides import GUIDES, read
from tests.test_ui_review import ReviewTestBase


class TestThePromptAndTheParser(unittest.TestCase):
    """Brief test 1: the question is not asked, and the word is not read."""

    def test_the_prompt_does_not_mention_a_subject(self):
        self.assertNotIn("subject", junk._QUALITY_PROMPT.lower())

    def test_the_prompt_still_asks_about_the_eyes(self):
        """The half that stays — without this the case above passes on an empty prompt."""
        self.assertIn("eyes_open", junk._QUALITY_PROMPT)
        self.assertIn("eyes_closed", junk._QUALITY_PROMPT)

    def test_no_keyword_writes_the_retired_column(self):
        fields = {name for name, _keywords in junk._QUALITY_KEYWORDS}
        self.assertEqual(fields, {"eyes_open"})

    def test_the_word_is_ignored_wherever_it_appears(self):
        """Brief test 5 in the same breath: the eyes answer next to it still parses."""
        for answer in ("no_subject", "subject", "eyes_open subject",
                       "eyes_closed, no subject at all"):
            with self.subTest(answer=answer):
                self.assertIsNone(parse_quality_answer(answer).has_subject)
        self.assertEqual(parse_quality_answer("eyes_open subject"),
                         QualityFlags(eyes_open=True))
        self.assertEqual(parse_quality_answer("eyes_closed, no subject at all"),
                         QualityFlags(eyes_open=False))

    def test_an_answer_that_is_only_the_retired_word_is_not_an_answer(self):
        """It used to be one: `no_subject` alone parsed, was `known`, and was stored."""
        flags = parse_quality_answer("no_subject")
        self.assertFalse(flags.known)


class MigrationCase(unittest.TestCase):
    """A database as the version before this one left it — answers and all."""

    def old_database(self, rows: list[tuple[int | None, int | None]]) -> Path:
        """`rows` is (eyes_open, has_subject) per file, written under the old schema."""
        db = Path(self.tmp.name) / "old.db"
        conn = connect(db)
        for number, (eyes_open, has_subject) in enumerate(rows, start=1):
            conn.execute(
                "INSERT INTO files (id, path, size, mtime, ext, media_type, indexed_at) "
                "VALUES (?, ?, 1, 0.0, 'jpg', 'photo', '2026-08-01')",
                (number, f"/photos/{number}.jpg"))
            conn.execute(
                """INSERT INTO frame_quality (file_id, sharpness, eyes_open, has_subject,
                       source, updated_at)
                   VALUES (?, 120.0, ?, ?, 'vlm#abc12345', '2026-08-03')""",
                (number, eyes_open, has_subject))
        # The shape does not change with this version, so the fixture rolls the NUMBER
        # back and leaves the columns alone — which is what a live database looks like.
        roll_back_to(conn, SCHEMA_VERSION - 1)
        conn.commit()
        conn.close()
        return db

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)

    def scalar(self, conn: sqlite3.Connection, sql: str) -> int:
        return int(conn.execute(sql).fetchone()[0])


class TestTheMigrationErasesTheAnswers(MigrationCase):
    """Brief test 2: the main one — otherwise the slice keeps showing its 212 frames."""

    def test_no_answer_survives_it(self):
        db = self.old_database([(1, 1), (0, 0), (None, 1), (1, None)])
        conn = connect(db)
        left = self.scalar(
            conn, "SELECT COUNT(*) FROM frame_quality WHERE has_subject IS NOT NULL")
        (version,) = conn.execute("PRAGMA user_version").fetchone()
        conn.close()
        self.assertEqual(left, 0)
        self.assertEqual(version, SCHEMA_VERSION)

    def test_it_is_committed_and_not_only_visible_to_its_own_connection(self):
        """`_migrate` writes no COMMIT of its own — the `executescript` that applies the
        schema after it does. Without that, the erasure would vanish on close and the
        slice would come back on the next run."""
        db = self.old_database([(1, 1), (0, 1)])
        connect(db).close()                  # migrates, commits nothing by hand
        raw = sqlite3.connect(db)            # raw: no migration, just what is on disk
        left = int(raw.execute(
            "SELECT COUNT(*) FROM frame_quality WHERE has_subject IS NOT NULL"
        ).fetchone()[0])
        (version,) = raw.execute("PRAGMA user_version").fetchone()
        raw.close()
        self.assertEqual(left, 0)
        self.assertEqual(version, SCHEMA_VERSION)

    def test_the_fixture_really_carried_answers_before_it(self):
        """A rollback that wrote nothing would make the case above vacuous."""
        db = self.old_database([(1, 1), (0, 0)])
        raw = sqlite3.connect(db)  # raw: connect() would migrate it first
        stored = int(raw.execute(
            "SELECT COUNT(*) FROM frame_quality WHERE has_subject IS NOT NULL"
        ).fetchone()[0])
        raw.close()
        self.assertEqual(stored, 2)

    def test_it_runs_once_and_is_idempotent(self):
        db = self.old_database([(0, 0)])
        connect(db).close()
        conn = connect(db)   # already at the current version: nothing to migrate
        self.assertEqual(self.scalar(conn, "SELECT COUNT(*) FROM frame_quality"), 1)
        self.assertEqual(
            self.scalar(conn,
                        "SELECT COUNT(*) FROM frame_quality WHERE eyes_open IS NOT NULL"),
            1)
        conn.close()


class TestTheEyesAnswersSurviveIt(MigrationCase):
    """Brief test 2a: the paired case. One careless `UPDATE` loses them without a word.

    The proportions are the live run's, scaled down: answers about the eyes on most of
    the frames, closed eyes on a few of them, and a subject answer on every frame.
    """

    def rows(self) -> list[tuple[int | None, int | None]]:
        return ([(1, 1)] * 6 + [(0, 0)] * 3 + [(None, 1)] * 2)

    def test_every_eyes_answer_is_still_there_and_still_says_the_same(self):
        conn = connect(self.old_database(self.rows()))
        asked = self.scalar(
            conn, "SELECT COUNT(*) FROM frame_quality WHERE eyes_open IS NOT NULL")
        closed = self.scalar(
            conn, "SELECT COUNT(*) FROM frame_quality WHERE eyes_open = 0")
        conn.close()
        self.assertEqual(asked, 9)
        self.assertEqual(closed, 3)

    def test_the_rest_of_the_row_is_untouched(self):
        """The migration is one column wide: sharpness, the tier marker and the row
        itself all stay, so nothing is re-measured on the next run."""
        conn = connect(self.old_database([(1, 1)]))
        row = conn.execute("SELECT * FROM frame_quality WHERE file_id = 1").fetchone()
        conn.close()
        self.assertAlmostEqual(row["sharpness"], 120.0)
        self.assertEqual(row["source"], "vlm#abc12345")
        self.assertEqual(row["eyes_open"], 1)
        self.assertIsNone(row["has_subject"])

    def test_the_column_itself_stays(self):
        """Retired like `is_accidental`: NULL already means "not asked", and dropping a
        column in SQLite costs a table rebuild."""
        conn = connect(self.old_database([(1, 1)]))
        columns = {r["name"] for r in conn.execute("PRAGMA table_info(frame_quality)")}
        conn.close()
        self.assertIn("has_subject", columns)


class TestTheAlbumKindIsUnknown(unittest.TestCase):
    """Brief test 3: not a slice with no members — a name the program does not know."""

    def test_the_kind_is_gone_from_every_list_that_declares_it(self):
        for kinds in (sorter.QUALITY_ALBUM_KINDS, sorter.ALBUM_KINDS,
                      sorter.SELECTORLESS_ALBUM_KINDS):
            self.assertNotIn("no_subject", kinds)
        self.assertNotIn("no_subject", sorter.ALBUM_FOLDER_KEYS)
        self.assertNotIn("no_subject", i18n.FOLDER_KEYS)

    def test_the_slices_that_stay_are_still_declared(self):
        """F177 must not take the neighbours with it."""
        self.assertEqual(sorter.QUALITY_ALBUM_KINDS, ("blurred", "eyes_closed"))
        for kind in ("blurred", "eyes_closed"):
            self.assertIn(kind, sorter.ALBUM_FOLDER_KEYS)

    def test_planning_one_is_refused_by_name(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            conn = connect(Path(tmp) / "t.db")
            try:
                with self.assertRaises(ValueError) as caught:
                    sorter.plan_album(sorter.Config(), conn, "no_subject", "",
                                      Path(tmp) / "albums")
            finally:
                conn.close()
        message = str(caught.exception)
        self.assertIn("no_subject", message)          # which name was refused
        self.assertIn("eyes_closed", message)         # and what may be asked for instead

    def test_the_help_and_the_refusal_no_longer_offer_it(self):
        """The two places the terminal names the kinds — a list that still said
        `no_subject` would send a user straight into the error above."""
        for key in ("cli.help.album.kind", "cli.album.selector_required"):
            for lang in ("ru", "en", "ja"):
                with self.subTest(key=key, lang=lang):
                    text = i18n.cli_text(key, lang)
                    self.assertNotIn("no_subject", text)
                    self.assertIn("eyes_closed", text)


class TestTheSliceIsGoneFromTheInterface(ReviewTestBase):
    """Brief test 4: the workspace, its counters and the "Overview" row."""

    def test_the_switcher_declares_three_slices(self):
        self.assertEqual(ui._REVIEW_SLICES, ("dupes", "blurred", "eyes"))
        self.assertNotIn("subject", ui._REVIEW_SLICE_KIND)
        self.assertNotIn("subject", ui._REVIEW_SLICE_ORDER)

    def test_the_route_refuses_the_slice(self):
        self.start_server()
        status, body, _ctype = self.get("/api/review?slice=subject")
        self.assertEqual(status, 400)
        self.assertIn("error", json.loads(body))

    def test_the_counters_do_not_carry_it(self):
        self.add_reviewable("eyes.jpg", sharpness=500.0, eyes_open=0)
        self.start_server()
        counts = self.counts(self.review("?slice=blurred"))
        self.assertEqual(set(counts), {"dupes", "blurred", "eyes"})

    def test_the_overview_does_not_count_it(self):
        self.add_reviewable("eyes.jpg", sharpness=500.0, eyes_open=0)
        self.start_server()
        _status, body, _ctype = self.get("/api/overview")
        collection = json.loads(body)["collection"]
        self.assertNotIn("no_subject", collection)
        self.assertEqual(collection["eyes_closed"], 1)   # the neighbour still counts

    def test_the_album_route_refuses_the_kind(self):
        self.start_server()
        status, _body = self.post("/api/album",
                                  {"kind": "no_subject", "mode": "link", "apply": False})
        self.assertEqual(status, 400)

    def test_nothing_of_it_is_left_in_the_page(self):
        """Deleted, not hidden: no button, no counter, no string, no hint."""
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        for fragment in ('id="review-slice-subject"', 'id="review-count-subject"',
                         "review_slice_subject", "review_hint_subject",
                         "overview_no_subject", "no_subject"):
            with self.subTest(fragment=fragment):
                self.assertNotIn(fragment, html)
        for key in ("review_slice_subject", "review_hint_subject",
                    "overview_no_subject"):
            with self.subTest(key=key):
                self.assertNotIn(key, ui._UI_STRINGS)


class TestTheGuidesSayNothingAboutIt(unittest.TestCase):
    """Brief test 6: the documentation watchdog, in all three languages.

    The album kind, the folder name and the slice label, each in the language it is
    written in. A guide that still lists `sorta album no_subject` documents a command
    that now fails.
    """

    RETIRED = ("no_subject", "has_subject", "No subject", "Без сюжета", "без сюжета",
               "被写体なし")

    def test_no_guide_mentions_the_retired_question(self):
        for lang, path in GUIDES.items():
            text = read(path)
            for token in self.RETIRED:
                with self.subTest(lang=lang, token=token):
                    self.assertNotIn(token, text)

    def test_the_guides_still_describe_the_slices_that_stay(self):
        """The other half — a watchdog that passed on an emptied guide would be worse
        than none."""
        for lang, path in GUIDES.items():
            text = read(path)
            for token in ("sorta album blurred", "sorta album eyes_closed"):
                with self.subTest(lang=lang, token=token):
                    self.assertIn(token, text)


if __name__ == "__main__":
    unittest.main()
