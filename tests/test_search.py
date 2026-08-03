"""F129: search by words — the engine, and the CLI that prints it.

F141 moved the table this reads from: the ranking comes out of `search_embeddings`, the
multilingual index, and not out of the classification vectors of `clip_embeddings`. That
is the one change here — every property below is F129's and holds unchanged, which is the
point of pinning them.

The properties under test are the ones the feature is worth nothing without:

* a query lands in the same space as the stored vectors, with a norm of 1;
* the ranking is deterministic and descending — a search that reshuffles between runs
  cannot be measured, and measuring it is a condition of the feature;
* a row of ANOTHER model never reaches the output. This is the main correctness case:
  mixing two embedding spaces produces a plausible ranking that nothing in the output
  marks as wrong;
* an empty table gives a REASON, not an empty list. "Nothing was found" and "nothing was
  ever computed" read identically in a list of zero lines, and only one of them is fixed
  by running `sorta junk`.

No model is loaded anywhere here: the encoder is a two-line fake, which is exactly why
`encode_query`/`search` take one instead of building it themselves.
"""
from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from sorta import cli, i18n, search
from sorta.config import Config, FeaturesConfig
from sorta.db import connect
from sorta.junk import pack_embedding, search_index_model

_DIM = 8


def unit(*values: float) -> np.ndarray:
    """A vector of the test dimension, padded with zeros and normalized."""
    vec = np.zeros(_DIM, dtype=np.float32)
    vec[:len(values)] = values
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm else vec


def encoder_for(vectors: dict[str, np.ndarray], scale: float = 3.0):
    """A fake CLIP text tower: a known direction per query, deliberately NOT normalized.

    The scale is the point — `encode_query` has to bring the vector back to a norm of 1
    on its own, because a dot product is only a cosine when both sides are unit vectors.
    An unknown query answers with the first axis, so a test that does not care about the
    direction does not have to declare one.
    """
    def encode(texts):
        return np.stack([vectors.get(t, unit(1.0)) * scale for t in texts])
    return encode


class SearchTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self.tmp.name)
        self.cfg = Config(database=self.root / "test.db", raw={"language": "en"})
        self.conn = connect(self.cfg.database)
        # F141: the model of the SEARCH side — `features.search_model`, not
        # `naming.clip.*`. The engine reads `search_embeddings` and nothing else, which is
        # what the fixture below writes.
        self.model = search_index_model(self.cfg)
        self._n = 0

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def add_photo(self, vec: np.ndarray | None = None, *, model: str | None = None,
                  dup_of: int | None = None, error: str | None = None,
                  media_type: str = "photo") -> int:
        """One indexed file, with its stored vector unless `vec` is None."""
        self._n += 1
        cur = self.conn.execute(
            """INSERT INTO files (path, size, mtime, ext, media_type, dup_of, error,
                   indexed_at)
               VALUES (?, 10, 0, '.jpg', ?, ?, ?, '2026-01-01')""",
            (str(self.root / f"{self._n}.jpg"), media_type, dup_of, error))
        file_id = int(cur.lastrowid or 0)
        if vec is not None:
            self.conn.execute(
                """INSERT INTO search_embeddings (file_id, model, dim, vec, updated_at)
                   VALUES (?, ?, ?, ?, '2026-01-01')""",
                (file_id, model or self.model, int(vec.size), pack_embedding(vec)))
        self.conn.commit()
        return file_id

    def search(self, query: str, limit: int = 10,
               vector: np.ndarray | None = None) -> list[tuple[int, float]]:
        """The engine, with the query pointing at the first axis unless told otherwise."""
        encoded = search.encode_query(
            query, encoder_for({query: unit(1.0) if vector is None else vector}))
        return search.search(self.conn, encoded, self.model, limit)


class TestEncodeQuery(SearchTestBase):
    def test_the_vector_has_the_model_width_and_a_norm_of_one(self):
        vec = search.encode_query("cake", encoder_for({"cake": unit(1.0, 1.0)}))
        self.assertEqual(vec.shape, (_DIM,))
        self.assertAlmostEqual(float(np.linalg.norm(vec)), 1.0, places=6)

    def test_the_direction_survives_the_normalization(self):
        raw = unit(0.0, 1.0)
        vec = search.encode_query("snow", encoder_for({"snow": raw}))
        np.testing.assert_allclose(vec, raw, atol=1e-6)

    def test_surrounding_whitespace_is_not_part_of_the_query(self):
        seen: list[list[str]] = []

        def encode(texts):
            seen.append(list(texts))
            return np.stack([unit(1.0) for _ in texts])

        search.encode_query("  cake \n", encode)
        self.assertEqual(seen, [["cake"]])

    def test_an_empty_query_is_refused_rather_than_encoded(self):
        with self.assertRaises(ValueError):
            search.encode_query("   ", encoder_for({}))

    def test_a_zero_vector_is_not_divided_by_zero(self):
        vec = search.encode_query("nothing", lambda texts: np.zeros((1, _DIM), np.float32))
        self.assertEqual(float(np.linalg.norm(vec)), 0.0)


class TestRanking(SearchTestBase):
    def test_scores_descend_and_the_nearest_frame_is_first(self):
        near = self.add_photo(unit(1.0, 0.1))
        middle = self.add_photo(unit(1.0, 1.0))
        far = self.add_photo(unit(0.0, 1.0))
        hits = self.search("cake")
        self.assertEqual([fid for fid, _s in hits], [near, middle, far])
        scores = [score for _fid, score in hits]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_the_score_is_the_cosine(self):
        self.add_photo(unit(1.0, 1.0))
        (_fid, score), = self.search("cake")
        self.assertAlmostEqual(score, float(np.dot(unit(1.0), unit(1.0, 1.0))), places=5)

    def test_the_same_query_returns_the_same_list_every_time(self):
        # Ties are the case that decides this: three identical vectors can come back in
        # any order unless something pins one, and file_id is what pins it.
        ids = [self.add_photo(unit(1.0, 1.0)) for _ in range(3)]
        for _attempt in range(3):
            self.assertEqual([fid for fid, _s in self.search("cake")], sorted(ids))

    def test_the_limit_cuts_the_sample_and_asking_for_more_is_harmless(self):
        ids = [self.add_photo(unit(1.0, 0.1 * i)) for i in range(5)]
        self.assertEqual(len(self.search("cake", limit=2)), 2)
        self.assertEqual(len(self.search("cake", limit=5000)), len(ids))
        self.assertEqual(self.search("cake", limit=0), [])


class TestOnlyOurOwnModelIsRanked(SearchTestBase):
    """The main correctness case: an incomparable vector must not rank, ever."""

    def test_a_row_of_another_model_is_not_in_the_output(self):
        ours = self.add_photo(unit(0.0, 1.0))              # far from the query
        theirs = self.add_photo(unit(1.0), model="other/model")  # right on it
        hits = self.search("cake")
        self.assertEqual([fid for fid, _s in hits], [ours])
        self.assertNotIn(theirs, [fid for fid, _s in hits])

    def test_a_table_holding_only_another_model_is_a_refusal_and_says_which(self):
        self.add_photo(unit(1.0), model="other/model")
        with self.assertRaises(search.EmbeddingsMissing) as ctx:
            self.search("cake")
        self.assertEqual(ctx.exception.reason, search.REASON_OTHER_MODEL)
        self.assertEqual(ctx.exception.stored, 0)
        self.assertEqual(ctx.exception.total, 1)
        self.assertEqual(ctx.exception.model, self.model)

    def test_a_row_of_the_wrong_width_is_dropped_instead_of_crashing_the_search(self):
        good = self.add_photo(unit(1.0))
        self.add_photo(np.ones(_DIM + 3, dtype=np.float32))
        self.assertEqual([fid for fid, _s in self.search("cake")], [good])


class TestTheEmptyTableIsAReason(SearchTestBase):
    def test_an_empty_table_raises_instead_of_returning_nothing(self):
        self.add_photo(None)  # an indexed file whose vector was never computed
        with self.assertRaises(search.EmbeddingsMissing) as ctx:
            self.search("cake")
        self.assertEqual(ctx.exception.reason, search.REASON_EMPTY)
        self.assertEqual((ctx.exception.total, ctx.exception.stored), (0, 0))

    def test_the_message_names_the_stage_that_fills_the_table(self):
        for lang in ("ru", "en", "ja"):
            with self.subTest(lang=lang):
                self.assertIn("junk", i18n.cli_text("cli.search.no_embeddings", lang))


class TestThePopulationIsCanonicalPhotographs(SearchTestBase):
    def test_duplicates_broken_files_and_video_stay_out(self):
        canonical = self.add_photo(unit(1.0))
        duplicate = self.add_photo(unit(1.0), dup_of=canonical)
        broken = self.add_photo(unit(1.0), error="cannot read")
        video = self.add_photo(unit(1.0), media_type="video")
        hits = [fid for fid, _s in self.search("cake")]
        self.assertEqual(hits, [canonical])
        for excluded in (duplicate, broken, video):
            self.assertNotIn(excluded, hits)


class TestSearchText(SearchTestBase):
    """The entry point the CLI, the album and the measurement share."""

    def test_the_limit_defaults_to_the_configured_sample_size(self):
        for _i in range(5):
            self.add_photo(unit(1.0))
        cfg = Config(database=self.cfg.database, features=FeaturesConfig(search_limit=2))
        hits = search.search_text(cfg, self.conn, "cake", encoder=encoder_for({}))
        self.assertEqual(len(hits), 2)

    def test_an_explicit_limit_wins_over_the_config(self):
        for _i in range(5):
            self.add_photo(unit(1.0))
        cfg = Config(database=self.cfg.database, features=FeaturesConfig(search_limit=2))
        hits = search.search_text(cfg, self.conn, "cake", limit=4,
                                  encoder=encoder_for({}))
        self.assertEqual(len(hits), 4)

    def test_file_paths_survives_more_ids_than_sqlite_binds_at_once(self):
        # `features.search_limit` is a user-set number, so the result list can be longer
        # than the parameter ceiling of a single statement.
        self.conn.executemany(
            "INSERT INTO files (path, size, mtime, ext, media_type, indexed_at) "
            "VALUES (?, 10, 0, '.jpg', 'photo', '2026-01-01')",
            [(str(self.root / f"bulk{i}.jpg"),) for i in range(1200)])
        self.conn.commit()
        ids = [int(r["id"]) for r in self.conn.execute("SELECT id FROM files")]
        self.assertEqual(len(search.file_paths(self.conn, ids)), len(ids))

    def test_file_paths_answers_for_the_ids_of_a_result(self):
        first = self.add_photo(unit(1.0))
        second = self.add_photo(unit(1.0))
        paths = search.file_paths(self.conn, [first, second])
        self.assertEqual(set(paths), {first, second})
        self.assertTrue(paths[first].endswith(".jpg"))
        self.assertEqual(search.file_paths(self.conn, []), {})


class TestTheSearchCommand(SearchTestBase):
    """`sorta search "cake"` — paths and scores, and the two refusals."""

    def setUp(self):
        super().setUp()
        self.config_path = self.root / "config.yaml"
        self.config_path.write_text(
            f'database: "{self.cfg.database.as_posix()}"\nlanguage: en\n',
            encoding="utf-8")

    def run_search(self, query: str = "cake", limit: int | None = None) -> str:
        buffer = io.StringIO()
        with patch.object(search, "text_encoder", lambda s: encoder_for({})), \
                contextlib.redirect_stdout(buffer):
            cli._cmd_search(str(self.config_path), query, limit=limit)
        return buffer.getvalue()

    def test_it_prints_a_path_and_a_score_per_frame(self):
        self.add_photo(unit(1.0))
        self.add_photo(unit(0.0, 1.0))
        printed = self.run_search()
        self.assertEqual(len(printed.strip().splitlines()), 3)  # two hits + the summary
        for file_id, _score in self.search("cake"):
            self.assertIn(search.file_paths(self.conn, [file_id])[file_id], printed)
        self.assertIn("1.000", printed)
        self.assertIn(i18n.cli_text("cli.search.done", "en", n=2, query="cake"), printed)

    def test_the_limit_flag_shortens_the_list(self):
        for _i in range(4):
            self.add_photo(unit(1.0))
        self.assertEqual(len(self.run_search(limit=2).strip().splitlines()), 3)

    def test_an_empty_table_exits_with_the_reason_instead_of_printing_nothing(self):
        self.add_photo(None)
        with self.assertRaises(SystemExit) as ctx:
            self.run_search()
        self.assertEqual(str(ctx.exception),
                         i18n.cli_text("cli.search.no_embeddings", "en"))

    def test_another_model_exits_with_a_message_naming_the_configured_one(self):
        self.add_photo(unit(1.0), model="other/model")
        with self.assertRaises(SystemExit) as ctx:
            self.run_search()
        self.assertEqual(str(ctx.exception), i18n.cli_text(
            "cli.search.other_model", "en", model=self.model, n=1))

    def test_an_empty_query_is_refused_before_the_model_is_touched(self):
        self.add_photo(unit(1.0))
        with self.assertRaises(SystemExit) as ctx:
            self.run_search(query="   ")
        self.assertEqual(str(ctx.exception), i18n.cli_text("cli.search.empty_query", "en"))


class TestTheCommandIsWiredIntoTheInterface(unittest.TestCase):
    def setUp(self):
        if cli.app is None:  # pragma: no cover — the argparse fallback
            self.skipTest("typer is not installed")

    def test_the_application_registers_a_search_command(self):
        app = cli.build_app("en")
        names = {info.name or info.callback.__name__ for info in app.registered_commands}
        self.assertIn("search", names)

    def test_the_help_of_the_command_exists_in_all_three_languages(self):
        for key in ("cli.help.search", "cli.help.search.query", "cli.help.search.limit"):
            texts = {lang: i18n.cli_text(key, lang) for lang in ("ru", "en", "ja")}
            with self.subTest(key=key):
                self.assertEqual(len(set(texts.values())), 3)
                for lang, text in texts.items():
                    self.assertNotIn("cli.", text, lang)


if __name__ == "__main__":
    unittest.main()
