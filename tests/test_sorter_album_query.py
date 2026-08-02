"""F129: the album from a query — `sorter.plan_album(kind='query')`.

The slice is what makes this kind different from the other three: it is not written down
anywhere in the database, it is the top of a CLIP ranking computed on the spot
(`search.search_text`), `features.search_limit` frames deep. Everything else — dry-run
semantics, the journal-before-the-operation invariant, `_resolve_dst` on a repeat gather —
is inherited from F34/F97 and pinned here for the new kind, because inheritance that is
not checked is a plan rather than a property.

No model is loaded: the encoder is injected, which is what the `encoder` parameter of
`plan_album` exists for.
"""
from __future__ import annotations

import io
import os
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

from sorta import cli, i18n, search
from sorta.config import FeaturesConfig
from sorta.junk import embedding_model, pack_embedding
from sorta.naming import naming_settings
from sorta.sorter import plan_album

from tests.test_search import encoder_for, unit
from tests.test_sorter import SorterTestBase


class QueryAlbumTestBase(SorterTestBase):
    def setUp(self):
        super().setUp()
        self.model = embedding_model(naming_settings(self.cfg))

    def add_photo(self, rel: str, vec: np.ndarray, *, content: bytes = b"data",
                  model: str | None = None, **kwargs) -> int:
        """An indexed file with the stored CLIP vector the junk stage would have left."""
        file_id = self.add_file(rel, content=content, **kwargs)
        self.conn.execute(
            """INSERT INTO clip_embeddings (file_id, model, dim, vec, updated_at)
               VALUES (?, ?, ?, ?, '2026-01-01')""",
            (file_id, model or self.model, int(vec.size), pack_embedding(vec)))
        self.conn.commit()
        return file_id

    def gather(self, query: str = "cake", **kwargs):
        """plan_album for the query kind, with the CLI chatter swallowed."""
        with redirect_stdout(io.StringIO()):
            return plan_album(self.cfg, self.conn, "query", query, self.dest,
                              encoder=encoder_for({}), **kwargs)

    def limit(self, n: int) -> None:
        self.cfg.features = FeaturesConfig(search_limit=n)


class TestQueryAlbumSelection(QueryAlbumTestBase):
    def test_the_slice_is_the_top_of_the_ranking(self):
        near = self.add_photo("near.jpg", unit(1.0, 0.1))
        middle = self.add_photo("middle.jpg", unit(1.0, 1.0))
        self.add_photo("far.jpg", unit(0.0, 1.0))
        self.limit(2)
        report = self.gather()
        self.assertEqual({it.file_id for it in report.plan}, {near, middle})

    def test_search_limit_bounds_the_sample(self):
        for i in range(5):
            self.add_photo(f"{i}.jpg", unit(1.0, 0.1 * i))
        self.limit(3)
        self.assertEqual(len(self.gather().plan), 3)

    def test_a_limit_larger_than_the_collection_is_not_an_error(self):
        ids = [self.add_photo(f"{i}.jpg", unit(1.0)) for i in range(3)]
        self.limit(5000)
        self.assertEqual({it.file_id for it in self.gather().plan}, set(ids))

    def test_duplicates_unreadable_files_and_video_stay_out(self):
        canonical = self.add_photo("a.jpg", unit(1.0), content=b"a")
        duplicate = self.add_photo("b.jpg", unit(1.0), content=b"b")
        broken = self.add_photo("c.jpg", unit(1.0), content=b"c")
        self.conn.execute("UPDATE files SET dup_of = ? WHERE id = ?",
                          (canonical, duplicate))
        self.conn.execute("UPDATE files SET error = 'cannot read' WHERE id = ?", (broken,))
        self.conn.commit()
        self.assertEqual([it.file_id for it in self.gather().plan], [canonical])

    def test_frames_of_another_model_never_reach_the_album(self):
        ours = self.add_photo("ours.jpg", unit(0.0, 1.0), content=b"ours")
        theirs = self.add_photo("theirs.jpg", unit(1.0), content=b"theirs",
                                model="other/model")
        plan = self.gather().plan
        self.assertEqual([it.file_id for it in plan], [ours])
        self.assertNotIn(theirs, [it.file_id for it in plan])

    def test_without_any_vector_the_album_refuses_with_a_reason(self):
        self.add_file("nothing.jpg")
        with self.assertRaises(search.EmbeddingsMissing) as ctx:
            self.gather()
        self.assertEqual(ctx.exception.reason, search.REASON_EMPTY)

    def test_where_still_narrows_the_slice(self):
        paris = self.add_photo("a.jpg", unit(1.0), content=b"a",
                               country="France", city="Paris")
        self.add_photo("b.jpg", unit(1.0), content=b"b", country="Russia", city="Moskva")
        report = self.gather(where=["city=Paris"])
        self.assertEqual([it.file_id for it in report.plan], [paris])

    def test_the_default_album_name_is_the_query(self):
        self.add_photo("a.jpg", unit(1.0))
        report = self.gather(query="cake")
        self.assertEqual(report.album_name, "cake")
        self.assertEqual(Path(report.dest).name, "cake")

    def test_an_explicit_name_still_wins(self):
        self.add_photo("a.jpg", unit(1.0))
        report = self.gather(query="cake", album_name="Birthdays")
        self.assertEqual(report.album_name, "Birthdays")


class TestQueryAlbumApply(QueryAlbumTestBase):
    def test_dry_run_writes_nothing_to_the_db_or_the_filesystem(self):
        self.add_photo("cake.jpg", unit(1.0))
        report = self.gather(mode="link", apply=False)
        self.assertEqual(len(report.plan), 1)
        self.assertFalse(self.dest.exists())
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM moves").fetchone()[0], 0)
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM move_batches").fetchone()[0], 0)

    def test_apply_link_journals_before_the_operation_with_the_album_query_mode(self):
        file_id = self.add_photo("sub/deep/cake.jpg", unit(1.0), content=b"cake")
        report = self.gather(query="cake", mode="link", apply=True)
        self.assertEqual(report.transferred, 1)
        dst = self.dest / "cake" / "cake.jpg"
        self.assertTrue(dst.exists())
        self.assertGreaterEqual(os.stat(dst).st_nlink, 2)  # a hardlink, not a copy
        batch = self.conn.execute(
            "SELECT mode, operation, dest_root FROM move_batches WHERE id = ?",
            (report.batch_id,)).fetchone()
        self.assertEqual(batch["mode"], "album_query")
        self.assertEqual(batch["operation"], "link")
        self.assertEqual(batch["dest_root"], str(self.dest.resolve()))
        move = self.conn.execute(
            "SELECT file_id, dst, status FROM moves WHERE batch_id = ?",
            (report.batch_id,)).fetchone()
        self.assertEqual(move["file_id"], file_id)
        self.assertEqual(move["status"], "done")
        self.assertEqual(Path(move["dst"]), dst)

    def test_gathering_the_same_album_twice_makes_no_underscore_one_copies(self):
        # F97, inherited: a file already sitting in the album folder byte-for-byte is left
        # alone instead of being re-materialized under a `_1` name.
        self.add_photo("cake.jpg", unit(1.0), content=b"cake")
        self.gather(query="cake", mode="link", apply=True)
        second = self.gather(query="cake", mode="link", apply=True)
        self.assertEqual(second.skipped_already_copied, 1)
        self.assertEqual(second.transferred, 0)
        album_dir = self.dest / "cake"
        self.assertEqual(sorted(p.name for p in album_dir.iterdir()), ["cake.jpg"])


class TestTheAlbumCommand(QueryAlbumTestBase):
    """`sorta album query "cake" --dest …` — the command, not just the planner."""

    def setUp(self):
        super().setUp()
        self.config_path = self.root / "config.yaml"
        self.config_path.write_text(
            f'database: "{self.cfg.database.as_posix()}"\nlanguage: en\n',
            encoding="utf-8")

    def album(self, query: str = "cake", **kwargs) -> None:
        # The command's own connection is handed the one this case already holds: on
        # Windows a second handle to the same file keeps the temporary directory alive
        # after the test (`_cmd_album` does not close what it opens), and the cleanup is
        # what fails, not the feature.
        with patch.object(search, "text_encoder", lambda s: encoder_for({})), \
                patch.object(cli, "connect", lambda _db: self.conn), \
                redirect_stdout(io.StringIO()):
            cli._cmd_album(str(self.config_path), "query", query, self.dest, **kwargs)

    def test_it_gathers_the_album_with_hardlinks(self):
        self.add_photo("cake.jpg", unit(1.0), content=b"cake")
        self.album(apply=True)
        dst = self.dest / "cake" / "cake.jpg"
        self.assertTrue(dst.exists())
        self.assertGreaterEqual(os.stat(dst).st_nlink, 2)

    def test_a_dry_run_writes_nothing(self):
        self.add_photo("cake.jpg", unit(1.0), content=b"cake")
        self.album()
        self.assertFalse(self.dest.exists())

    def test_without_embeddings_the_command_exits_with_the_reason(self):
        self.add_file("nothing.jpg")
        with self.assertRaises(SystemExit) as ctx:
            self.album(apply=True)
        self.assertEqual(str(ctx.exception),
                         i18n.cli_text("cli.search.no_embeddings", "en"))


if __name__ == "__main__":
    unittest.main()
