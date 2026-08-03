"""F141: the search index — a second CLIP vector, from a model of its own.

The feature exists because of a measurement, so what is under test is every promise that
measurement was traded for. `ViT-L-14` answers English and does not answer Russian (22%
precision at top-5 against 98% for `xlm-roberta-base-ViT-B-32`, with four of eight
concepts returning nothing at all), and the thing that must NOT happen in response is a
swap of the pipeline's model: the landmark (F75), animal (F122) and cascade (F130)
thresholds are calibrated on its numbers. Hence a second vector, a second table and a
toggle that says what it costs.

The properties, in the order the brief lists them:

* with `features.search_index` off nothing is written and the rest of the run is
  unchanged, down to the number of classifier calls;
* with it on, every canonical photograph gets a vector, stored with the name of the model
  that produced it — F128's rule, not weakened;
* a row of another model is STALE: recomputed on the next run, and never used before it;
* search reads the SEARCH index and not the classification one — checked with vectors of
  two different widths in the two tables, so a search that read the wrong one cannot
  accidentally agree with a search that read the right one;
* an empty search index is a REASON (`EmbeddingsMissing`), not an empty result list;
* the F128 classification vectors are untouched and still written;
* nothing but a personal photograph reaches the search index, including a row left behind
  by an earlier run whose verdict has since changed.

No model is loaded anywhere: the image tower is a dict lookup and the text tower is two
lines, exactly as the rest of the junk and search suites do it.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from sorta import junk, search
from sorta.config import Config, FeaturesConfig, _features_from, _naming_from
from sorta.db import connect
from sorta.junk import (
    classify,
    embedding_model,
    read_search_embeddings,
    search_index_enabled,
    search_index_model,
    search_index_settings,
    unpack_embedding,
)
from tests.test_clip_embeddings import EmbeddingClassifier, deterministic_vector
from tests.test_junk import NO_OCR, FakeClassifier

_SCREENSHOT_IDX = 1  # the junk classes, in order: photo | screenshot | meme
# Deliberately NOT the width of the classification vectors below: a test that reads the
# wrong table has to fail on the shape, not merely on the numbers.
_SEARCH_DIM = 6
_CLIP_DIM = 8

DEFAULT_SEARCH_MODEL = "xlm-roberta-base-ViT-B-32/laion5b_s13b_b90k"


def search_vector(name: str) -> np.ndarray:
    """The search-side vector of a frame — its own width, so the tables cannot be mixed."""
    return deterministic_vector(f"search:{name}", _SEARCH_DIM)


class FakeSearchEncoder:
    """The search model's image tower: paths -> a vector each, None where it did not decode.

    The same shape `landmarks.CachingFeatureClassifier.encode` has, which is the whole
    reason `classify` takes the encoder as an argument. Every call is recorded: "a repeated
    run encodes nothing" is a promise about calls, not about rows.
    """

    def __init__(self, undecodable: set[str] | None = None, fails: bool = False) -> None:
        self.undecodable = set(undecodable or ())
        self.fails = fails
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, paths: list[str]) -> list[np.ndarray | None]:
        self.calls.append(tuple(paths))
        if self.fails:
            raise RuntimeError("the tower fell over")
        return [None if Path(p).name in self.undecodable else search_vector(Path(p).name)
                for p in paths]

    @property
    def encoded(self) -> list[str]:
        return [path for call in self.calls for path in call]


class SearchIndexCase(unittest.TestCase):
    """A DB, a config with the toggle ON, and the two tables side by side."""

    search_index = True

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = Config(sources=[Path(self.tmp.name)],
                          database=Path(self.tmp.name) / "test.db",
                          naming=_naming_from({}))
        self.cfg.features = FeaturesConfig(search_index=self.search_index)
        self.conn = connect(self.cfg.database)
        self.addCleanup(self.conn.close)

    def add_file(self, name: str, camera_make="Canon", camera_model="EOS") -> int:
        # A frame with camera EXIF is a "real photo" the fast tier will not call junk
        # (`junk._is_real_photo`), so anything meant to become a screenshot is added
        # without it — the same fixture rule the F128 suite follows.
        cur = self.conn.execute(
            """INSERT INTO files (path, size, mtime, ext, media_type, width, height,
                   camera_make, camera_model, indexed_at)
               VALUES (?, 1000, 0, 'jpg', 'photo', 4000, 3000, ?, ?, '2026-01-01')""",
            (f"/photos/{name}", camera_make, camera_model))
        self.conn.commit()
        return int(cur.lastrowid or 0)

    def run_classify(self, classifier=None, encoder=None, **kwargs):
        return classify(self.cfg, self.conn,
                        classifier=classifier if classifier is not None
                        else EmbeddingClassifier(),
                        text_detector=NO_OCR,
                        sharpness_detector=lambda _path, _faces: junk.Sharpness(100.0),
                        search_encoder=encoder, **kwargs)

    def indexed(self, fid: int):
        return self.conn.execute(
            "SELECT * FROM search_embeddings WHERE file_id = ?", (fid,)).fetchone()

    def index_rows(self) -> int:
        return int(self.conn.execute(
            "SELECT COUNT(*) FROM search_embeddings").fetchone()[0])


class TestTheToggleIsOff(SearchIndexCase):
    """Brief test 1: off is the default, and off means nothing happens."""

    search_index = False

    def test_the_default_config_does_not_index(self):
        self.assertFalse(FeaturesConfig().search_index)
        self.assertFalse(search_index_enabled(self.cfg))

    def test_nothing_is_written_and_nothing_is_encoded(self):
        fid = self.add_file("IMG_0001.jpg")
        encoder = FakeSearchEncoder()
        self.run_classify(encoder=encoder)
        self.assertIsNone(self.indexed(fid))
        self.assertEqual(encoder.calls, [])

    def test_the_rest_of_the_run_is_what_it_was(self):
        """The classification half is untouched: same verdicts, same vector, same calls."""
        fid = self.add_file("IMG_0001.jpg")
        clf = EmbeddingClassifier()
        stats = self.run_classify(clf, encoder=FakeSearchEncoder())
        verdict = self.conn.execute(
            "SELECT verdict FROM media_class WHERE file_id = ?", (fid,)).fetchone()
        self.assertEqual(verdict["verdict"], "photo")
        self.assertEqual(stats.embeddings_stored, 1)
        self.assertEqual(stats.search_vectors_stored, 0)
        self.assertEqual(clf.seen_paths, ["/photos/IMG_0001.jpg"] * 2)


class TestTheVectorIsStored(SearchIndexCase):
    """Brief test 2: with the toggle on, every photograph gets a vector of its own."""

    def test_every_canonical_photograph_is_indexed(self):
        ids = [self.add_file(f"IMG_000{i}.jpg") for i in range(1, 4)]
        stats = self.run_classify(encoder=FakeSearchEncoder())
        self.assertEqual(stats.search_vectors_stored, 3)
        self.assertEqual({int(r["file_id"]) for r in self.conn.execute(
            "SELECT file_id FROM search_embeddings")}, set(ids))

    def test_the_row_says_which_model_produced_it(self):
        """F128's rule, not weakened: a vector that does not name its model is rubbish."""
        fid = self.add_file("IMG_0001.jpg")
        self.run_classify(encoder=FakeSearchEncoder())
        self.assertEqual(self.indexed(fid)["model"], DEFAULT_SEARCH_MODEL)
        self.assertNotEqual(self.indexed(fid)["model"],
                            embedding_model(self.cfg.naming))

    def test_the_stored_vector_is_normalized_and_its_width_is_the_dim(self):
        fid = self.add_file("IMG_0001.jpg")
        self.run_classify(encoder=FakeSearchEncoder())
        row = self.indexed(fid)
        vec = unpack_embedding(row["vec"])
        self.assertEqual(row["dim"], _SEARCH_DIM)
        self.assertEqual(len(vec), row["dim"])
        self.assertAlmostEqual(float(np.linalg.norm(vec)), 1.0, places=3)

    def test_it_is_the_vector_of_that_frame_and_not_of_its_neighbour(self):
        first = self.add_file("IMG_0001.jpg")
        second = self.add_file("IMG_0002.jpg")
        self.run_classify(encoder=FakeSearchEncoder())
        stored = read_search_embeddings(self.conn, DEFAULT_SEARCH_MODEL)
        for fid, name in ((first, "IMG_0001.jpg"), (second, "IMG_0002.jpg")):
            expected = search_vector(name)
            expected = expected / np.linalg.norm(expected)
            self.assertGreater(float(stored[fid] @ expected), 0.999)

    def test_a_frame_that_does_not_encode_gets_no_row(self):
        # NULL does not happen in this table either: no vector means no row, and the frame
        # is selected again by the next run.
        fid = self.add_file("broken.jpg")
        encoder = FakeSearchEncoder(undecodable={"broken.jpg"})
        self.run_classify(encoder=encoder)
        self.assertIsNone(self.indexed(fid))
        self.run_classify(encoder=encoder)
        self.assertEqual(encoder.encoded, ["/photos/broken.jpg"] * 2)

    def test_the_pass_survives_a_tower_that_will_not_build(self):
        """The graceful fallback every optional half of this stage has."""
        fid = self.add_file("IMG_0001.jpg")

        def factory(_settings):
            raise RuntimeError("no weights on this machine")

        stats = self.run_classify(search_encoder_factory=factory)
        self.assertIsNone(self.indexed(fid))
        self.assertEqual(stats.by_verdict, {"photo": 1})   # the run itself is unharmed
        self.assertEqual(stats.embeddings_stored, 1)

    def test_a_batch_that_will_not_encode_costs_the_run_nothing(self):
        fid = self.add_file("IMG_0001.jpg")
        stats = self.run_classify(encoder=FakeSearchEncoder(fails=True))
        self.assertIsNone(self.indexed(fid))
        self.assertEqual(stats.search_vectors_stored, 0)
        self.assertEqual(stats.by_verdict, {"photo": 1})


class TestIncrementality(SearchIndexCase):
    """Brief test 3: the model is what makes a row stale — nothing else, and always."""

    def test_a_repeated_run_encodes_nothing(self):
        self.add_file("IMG_0001.jpg")
        encoder = FakeSearchEncoder()
        self.run_classify(encoder=encoder)
        self.assertEqual(len(encoder.encoded), 1)
        stats = self.run_classify(encoder=encoder)
        self.assertEqual(len(encoder.encoded), 1)      # not encoded a second time
        self.assertEqual(stats.search_vectors_stored, 0)

    def test_another_model_in_the_config_makes_the_row_stale(self):
        fid = self.add_file("IMG_0001.jpg")
        self.run_classify(encoder=FakeSearchEncoder())
        self.cfg.features = FeaturesConfig(
            search_index=True, search_model="other-tower/other-weights")
        encoder = FakeSearchEncoder()
        stats = self.run_classify(encoder=encoder)
        self.assertEqual(encoder.encoded, ["/photos/IMG_0001.jpg"])
        self.assertEqual(stats.search_vectors_stored, 1)
        self.assertEqual(self.indexed(fid)["model"], "other-tower/other-weights")
        self.assertEqual(self.index_rows(), 1)  # replaced, not added to

    def test_a_stale_row_is_never_read(self):
        """The other half of "stale": not used, not merely rewritten later.

        The read filters on the model, so a collection indexed by another tower ranks
        nothing at all rather than ranking plausibly in the wrong space.
        """
        fid = self.add_file("IMG_0001.jpg")
        self.run_classify(encoder=FakeSearchEncoder())
        self.assertEqual(set(read_search_embeddings(
            self.conn, DEFAULT_SEARCH_MODEL)), {fid})
        self.assertEqual(read_search_embeddings(self.conn, "other-tower/weights"), {})

    def test_a_new_photograph_is_indexed_when_the_rest_is_current(self):
        """The ordinary case of the toggle: a collection already classified, then a run."""
        self.add_file("IMG_0001.jpg")
        self.run_classify(encoder=FakeSearchEncoder())
        self.add_file("IMG_0002.jpg")
        encoder = FakeSearchEncoder()
        stats = self.run_classify(encoder=encoder)
        self.assertEqual(encoder.encoded, ["/photos/IMG_0002.jpg"])
        self.assertEqual(stats.search_vectors_stored, 1)

    def test_switching_the_toggle_on_indexes_an_already_classified_collection(self):
        """Nothing else in the stage has work, and the pass must still run.

        This is how the feature is actually switched on, and an early return that skipped
        it would leave it silently doing nothing until something unrelated went stale.
        """
        self.cfg.features = FeaturesConfig(search_index=False)
        fid = self.add_file("IMG_0001.jpg")
        self.run_classify()
        self.assertEqual(self.index_rows(), 0)

        self.cfg.features = FeaturesConfig(search_index=True)
        encoder = FakeSearchEncoder()
        stats = self.run_classify(encoder=encoder)
        self.assertEqual(stats.processed, 0)            # the junk half had nothing to do
        self.assertEqual(stats.search_vectors_stored, 1)
        self.assertIsNotNone(self.indexed(fid))


class TestClassificationVectorsAreUntouched(SearchIndexCase):
    """Brief test 6: F128 keeps working exactly as it did, next to this."""

    def test_both_tables_are_written_and_they_disagree_about_nothing(self):
        fid = self.add_file("IMG_0001.jpg")
        stats = self.run_classify(encoder=FakeSearchEncoder())
        classification = self.conn.execute(
            "SELECT * FROM clip_embeddings WHERE file_id = ?", (fid,)).fetchone()
        self.assertEqual(stats.embeddings_stored, 1)
        self.assertEqual(stats.search_vectors_stored, 1)
        self.assertEqual(classification["model"], embedding_model(self.cfg.naming))
        self.assertEqual(classification["dim"], _CLIP_DIM)
        self.assertEqual(self.indexed(fid)["dim"], _SEARCH_DIM)

    def test_the_classification_model_is_not_the_search_model(self):
        """The whole feature in one assertion: `naming.clip.*` is not touched."""
        self.assertEqual(embedding_model(self.cfg.naming), "ViT-L-14-quickgelu/openai")
        self.assertEqual(search_index_model(self.cfg), DEFAULT_SEARCH_MODEL)

    def test_switching_the_search_model_does_not_move_a_classification_threshold(self):
        """Brief boundary: no threshold of the pipeline is calibrated on the search model."""
        self.cfg.features = FeaturesConfig(
            search_index=True, search_model="other-tower/other-weights")
        self.assertEqual(self.cfg.naming.landmark_threshold, 0.85)
        self.assertEqual(self.cfg.features.pet_threshold,
                         FeaturesConfig().pet_threshold)
        self.assertEqual(self.cfg.features.pet_candidate_threshold,
                         FeaturesConfig().pet_candidate_threshold)


class TestOnlyPhotographs(SearchIndexCase):
    """Brief test 7: a screenshot is noise in a search over a family archive (F120)."""

    def test_a_screenshot_gets_no_row(self):
        photo = self.add_file("IMG_0001.jpg")
        shot = self.add_file("Screenshot_1.png", camera_make=None, camera_model=None)
        clf = EmbeddingClassifier({"Screenshot_1.png": (_SCREENSHOT_IDX, 0.99)})
        encoder = FakeSearchEncoder()
        self.run_classify(clf, encoder=encoder)
        self.assertIsNotNone(self.indexed(photo))
        self.assertIsNone(self.indexed(shot))
        self.assertNotIn("/photos/Screenshot_1.png", encoder.encoded)

    def test_a_row_left_by_an_earlier_run_is_removed_when_the_verdict_changes(self):
        """Incrementality would otherwise keep it PRECISELY because it is up to date."""
        shot = self.add_file("Screenshot_1.png", camera_make=None, camera_model=None)
        self.add_file("IMG_0001.jpg")
        # A row of the CURRENT model — nothing about it looks stale, which is the whole
        # difficulty: only the verdict says it does not belong here.
        self.conn.execute(
            "INSERT INTO search_embeddings (file_id, model, dim, vec, updated_at)"
            " VALUES (?, ?, ?, ?, 'old')",
            (shot, DEFAULT_SEARCH_MODEL, _SEARCH_DIM,
             junk.pack_embedding(search_vector("Screenshot_1.png"))))
        self.conn.commit()
        encoder = FakeSearchEncoder()
        self.run_classify(EmbeddingClassifier(
            {"Screenshot_1.png": (_SCREENSHOT_IDX, 0.99)}), encoder=encoder)
        self.assertIsNone(self.indexed(shot))
        self.assertNotIn("/photos/Screenshot_1.png", encoder.encoded)

    def test_a_duplicate_or_an_unreadable_file_is_not_indexed(self):
        keeper = self.add_file("IMG_0001.jpg")
        dupe = self.add_file("IMG_0002.jpg")
        broken = self.add_file("IMG_0003.jpg")
        self.conn.execute("UPDATE files SET dup_of = ? WHERE id = ?", (keeper, dupe))
        self.conn.execute("UPDATE files SET error = 'unreadable' WHERE id = ?", (broken,))
        self.conn.commit()
        self.run_classify(encoder=FakeSearchEncoder())
        self.assertEqual(self.index_rows(), 1)
        self.assertIsNotNone(self.indexed(keeper))

    def test_a_heuristics_only_run_indexes_nothing(self):
        # No CLIP at all is an explicit mode; a second CLIP pass inside it would be absurd.
        fid = self.add_file("IMG_0001.jpg")
        encoder = FakeSearchEncoder()
        self.run_classify(encoder=encoder, use_clip=False)
        self.assertIsNone(self.indexed(fid))
        self.assertEqual(encoder.calls, [])


class TestSearchReadsTheSearchIndex(SearchIndexCase):
    """Brief tests 4 and 5: which table the ranking comes out of, and what "empty" says.

    The two tables hold vectors of DIFFERENT WIDTHS here, which is the point: a search
    that read the classification table could not return the right ids by accident.
    """

    def encoder_for(self, vector: np.ndarray):
        return lambda texts: np.stack([vector for _ in texts])

    def store_classification_vector(self, fid: int, vec: np.ndarray) -> None:
        self.conn.execute(
            """INSERT INTO clip_embeddings (file_id, model, dim, vec, updated_at)
               VALUES (?, ?, ?, ?, '2026-01-01')
               ON CONFLICT(file_id) DO UPDATE SET vec = excluded.vec""",
            (fid, embedding_model(self.cfg.naming), int(vec.size),
             junk.pack_embedding(vec)))
        self.conn.commit()

    def test_the_ranking_comes_out_of_the_search_table(self):
        near = self.add_file("IMG_0001.jpg")
        far = self.add_file("IMG_0002.jpg")
        self.run_classify(encoder=FakeSearchEncoder())
        # The query points at the first frame's SEARCH vector; the classification table is
        # given the opposite ordering, so reading it would put `far` on top.
        query = search_vector("IMG_0001.jpg")
        self.store_classification_vector(near, -deterministic_vector("IMG_0001.jpg"))
        self.store_classification_vector(far, deterministic_vector("IMG_0002.jpg"))
        hits = search.search_text(self.cfg, self.conn, "cake",
                                  encoder=self.encoder_for(query))
        self.assertEqual([fid for fid, _score in hits], [near, far])

    def test_a_query_of_the_classification_width_ranks_nothing(self):
        """The width check is the last guard against the wrong text tower (768 vs 512)."""
        self.add_file("IMG_0001.jpg")
        self.run_classify(encoder=FakeSearchEncoder())
        with self.assertRaises(search.EmbeddingsMissing):
            search.search_text(self.cfg, self.conn, "cake",
                               encoder=self.encoder_for(np.ones(_CLIP_DIM,
                                                               dtype=np.float32)))

    def test_a_full_classification_table_and_no_search_index_is_a_reason(self):
        """Brief test 5, in the state that actually happens: the toggle was never on."""
        self.cfg.features = FeaturesConfig(search_index=False)
        self.add_file("IMG_0001.jpg")
        self.run_classify()
        self.assertEqual(int(self.conn.execute(
            "SELECT COUNT(*) FROM clip_embeddings").fetchone()[0]), 1)
        self.cfg.features = FeaturesConfig(search_index=True)
        with self.assertRaises(search.EmbeddingsMissing) as caught:
            search.search_text(self.cfg, self.conn, "cake",
                               encoder=self.encoder_for(search_vector("IMG_0001.jpg")))
        self.assertEqual(caught.exception.reason, search.REASON_EMPTY)
        self.assertEqual(caught.exception.total, 0)
        self.assertEqual(caught.exception.model, DEFAULT_SEARCH_MODEL)

    def test_an_index_of_another_model_says_so(self):
        self.add_file("IMG_0001.jpg")
        self.run_classify(encoder=FakeSearchEncoder())
        self.cfg.features = FeaturesConfig(
            search_index=True, search_model="other-tower/other-weights")
        with self.assertRaises(search.EmbeddingsMissing) as caught:
            search.search_text(self.cfg, self.conn, "cake",
                               encoder=self.encoder_for(search_vector("IMG_0001.jpg")))
        self.assertEqual(caught.exception.reason, search.REASON_OTHER_MODEL)
        self.assertEqual(caught.exception.total, 1)
        self.assertEqual(caught.exception.stored, 0)


class TestSettings(unittest.TestCase):
    """The config side: the name is a pair, and garbage does not reach open_clip."""

    def test_the_default_model_is_the_multilingual_one(self):
        self.assertEqual(FeaturesConfig().search_model, DEFAULT_SEARCH_MODEL)

    def test_a_name_without_weights_is_refused(self):
        """A bare architecture builds an UNTRAINED tower, which nothing downstream sees."""
        for bad in ("xlm-roberta-base-ViT-B-32", "/laion5b", "  ", 42, None, ["a/b"]):
            self.assertEqual(_features_from({"search_model": bad}).search_model,
                             DEFAULT_SEARCH_MODEL, msg=repr(bad))

    def test_a_well_formed_name_is_taken_as_written(self):
        parsed = _features_from({"search_model": " ViT-B-32/laion2b_s34b_b79k "})
        self.assertEqual(parsed.search_model, "ViT-B-32/laion2b_s34b_b79k")

    def test_the_toggle_reads_as_a_bool(self):
        self.assertTrue(_features_from({"search_index": True}).search_index)
        self.assertTrue(_features_from({"search_index": "yes"}).search_index)
        self.assertFalse(_features_from({}).search_index)

    def test_the_settings_of_the_search_tower_are_the_naming_ones_with_a_new_model(self):
        """Same batch size, same decode pool — the same previews, which is the brief's rule."""
        naming = _naming_from({})
        settings = search_index_settings(naming, DEFAULT_SEARCH_MODEL)
        self.assertEqual(settings.clip_model, "xlm-roberta-base-ViT-B-32")
        self.assertEqual(settings.clip_pretrained, "laion5b_s13b_b90k")
        self.assertEqual(settings.clip_batch_size, naming.clip_batch_size)
        self.assertEqual(settings.clip_decode_workers, naming.clip_decode_workers)
        self.assertEqual(naming.clip_model, "ViT-L-14-quickgelu")  # untouched

    def test_a_config_without_a_features_section_falls_back_to_the_defaults(self):
        cfg = Config()
        del cfg.features
        self.assertFalse(search_index_enabled(cfg))
        self.assertEqual(search_index_model(cfg), DEFAULT_SEARCH_MODEL)


class TestReadFilter(unittest.TestCase):
    """`read_search_embeddings` — the one function that reads the table, filter included."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.conn = connect(Path(self.tmp.name) / "test.db")
        self.addCleanup(self.conn.close)
        for i in range(1, 4):
            self.conn.execute(
                "INSERT INTO files (path, size, mtime, ext, media_type, indexed_at)"
                f" VALUES ('/{i}.jpg', 1, 0, 'jpg', 'photo', 'x')")
        for i, model in ((1, "ours"), (2, "ours"), (3, "theirs")):
            self.conn.execute(
                "INSERT INTO search_embeddings (file_id, model, dim, vec, updated_at)"
                " VALUES (?, ?, 4, ?, 'x')",
                (i, model, junk.pack_embedding(deterministic_vector(f"{i}.jpg", 4))))
        self.conn.commit()

    def test_rows_of_another_model_are_absent(self):
        self.assertEqual(set(read_search_embeddings(self.conn, "ours")), {1, 2})

    def test_the_file_filter_and_the_model_filter_both_apply(self):
        self.assertEqual(set(read_search_embeddings(self.conn, "ours", [2, 3])), {2})

    def test_more_ids_than_sqlite_binds_at_once(self):
        # The chunking `read_clip_embeddings` explains — a whole collection is the normal
        # argument here, and SQLite has a ceiling on bound parameters.
        self.assertEqual(set(read_search_embeddings(
            self.conn, "ours", list(range(1, 3000)))), {1, 2})

    def test_the_vector_comes_back_as_float32(self):
        vec = read_search_embeddings(self.conn, "ours")[1]
        self.assertEqual(vec.dtype, np.float32)
        self.assertEqual(len(vec), 4)


class TestTheStageIsUnchangedForEveryoneElse(SearchIndexCase):
    """A classifier that hands back no vectors is still fine — this pass has its own."""

    def test_a_plain_classifier_still_gets_a_search_index(self):
        fid = self.add_file("IMG_0001.jpg")
        stats = self.run_classify(FakeClassifier({}), encoder=FakeSearchEncoder())
        self.assertEqual(stats.embeddings_stored, 0)     # F128 off: no `features` method
        self.assertEqual(stats.search_vectors_stored, 1)  # F141 on: its own encoder
        self.assertIsNotNone(self.indexed(fid))
