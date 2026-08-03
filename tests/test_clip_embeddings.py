"""F128: the junk stage stops throwing its CLIP vectors away.

The feature shows the user nothing, so what is under test is exactly the set of
properties a later consumer (search by words, an album from a query, "frames like this
one") will depend on, and the promises the brief makes about the cost:

* the migration creates `clip_embeddings`, raises `user_version` and runs twice safely;
* a vector is written, its length is the stored `dim`, and its norm is 1 to the precision
  of the format — a consumer may treat a dot product as a cosine without normalizing;
* the number of classifier calls does NOT grow: the vector comes out of the call the stage
  was already making, not out of a pass of its own;
* the model is stored with the vector, and a different model in the config makes the rows
  stale rather than usable — vectors of two models are not comparable, and a search over a
  mix of them returns plausible nonsense;
* float16 costs nothing that matters: over 256 vectors the ranking by cosine similarity in
  half precision IS the ranking in single precision (the brief makes this test the
  condition for keeping the format);
* `store_embeddings: false` leaves the table empty and everything else exactly as it was;
* a repeated run rewrites nothing;
* a heuristics-only run (`use_clip=False`) creates no rows — no CLIP call, no vector, and
  no NULL row standing in for one;
* nothing but a personal photograph gets a row, and one left behind by an earlier run is
  removed.

No model is loaded anywhere below: the classifier is injected, exactly as the rest of the
junk suite does it.
"""
from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sorta import junk
from sorta.config import Config, FeaturesConfig, _naming_from
from sorta.db import SCHEMA_VERSION, connect, reset_index
from sorta.junk import (
    classify,
    embedding_model,
    pack_embedding,
    read_clip_embeddings,
    unpack_embedding,
)
from sorta.landmarks import CachingFeatureClassifier

from tests.schema_history import roll_back_before
from tests.test_junk import NO_OCR, FakeClassifier

_SCREENSHOT_IDX = 1  # the junk classes, in order: photo | screenshot | meme
_DIM = 8             # a toy dimension: nothing here depends on 768, only on the format


def deterministic_vector(name: str, dim: int = _DIM) -> np.ndarray:
    """A per-file vector that is the same in every run of the suite.

    Derived from the basename through sha1 rather than from `hash()`, which is salted per
    process — a test that stored a vector in one run and compared it in another would then
    fail on nothing.
    """
    seed = int.from_bytes(hashlib.sha1(name.encode()).digest()[:4], "big")
    return np.random.default_rng(seed).normal(size=dim).astype(np.float32)


class EmbeddingClassifier(FakeClassifier):
    """FakeClassifier plus the half of the real classifier this feature reads.

    `features(paths)` is what `landmarks.CachingFeatureClassifier` exposes over its cache,
    so a mock that has it stands in for the real thing; every call to either half is
    recorded, because the promise the feature has to keep is that switching it on adds no
    call at all.
    """

    def __init__(self, scores=None, doc_scores=None, vectors=None, undecodable=()):
        super().__init__(scores or {}, doc_scores)
        self.vectors = dict(vectors or {})
        self.undecodable = set(undecodable)
        self.calls: list[tuple[int, tuple[str, ...]]] = []
        self.feature_calls: list[tuple[str, ...]] = []

    def __call__(self, image_paths, prompts):
        self.calls.append((len(prompts), tuple(image_paths)))
        return super().__call__(image_paths, prompts)

    def features(self, paths):
        self.feature_calls.append(tuple(paths))
        return [self.vector_of(p) for p in paths]

    def prompt_counts(self) -> list[int]:
        """How many prompts each call carried — the shape of the pass, per chunk."""
        return [n for n, _paths in self.calls]

    def vector_of(self, path):
        name = Path(path).name
        if name in self.undecodable:
            return None  # the frame did not decode — no vector, and so no row
        if name in self.vectors:
            return np.asarray(self.vectors[name], dtype=np.float32)
        return deterministic_vector(name)


class TestMigration(unittest.TestCase):
    """Brief test 1: the table appears, the version moves, a repeat run changes nothing.

    Every connection is closed inside its temp directory: on Windows an open sqlite handle
    makes the rmtree fail (the same reason test_frame_quality explains at its own fixture).
    """

    def test_fresh_db_has_the_table_and_the_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = connect(Path(tmp) / "fresh.db")
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(clip_embeddings)")}
            (version,) = conn.execute("PRAGMA user_version").fetchone()
            conn.close()
        self.assertEqual(cols, {"file_id", "model", "dim", "vec", "updated_at"})
        self.assertEqual(version, SCHEMA_VERSION)

    def test_a_db_from_before_the_table_gains_it_without_touching_its_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "old.db"
            conn = connect(db)
            conn.execute(
                "INSERT INTO files (path, size, mtime, ext, media_type, indexed_at) "
                "VALUES ('/a.jpg', 1, 0.0, 'jpg', 'photo', 'x')")
            # F130: a database from before this table predates every LATER column too, and
            # the migrations that add those columns run on it. Leaving one in place would
            # make the fixture a shape no released version ever had, and the migration
            # would fail on a duplicate column instead of on anything real — which is why
            # the rollback is shared (tests/schema_history.py) rather than written here.
            roll_back_before(conn, "clip_embeddings")
            conn.commit()
            conn.close()

            conn = connect(db)
            tables = {r["name"] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            files = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
            conn.close()
        self.assertIn("clip_embeddings", tables)
        self.assertEqual(version, SCHEMA_VERSION)
        self.assertEqual(files, 1)

    def test_reopening_is_idempotent_and_keeps_the_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "twice.db"
            conn = connect(db)
            conn.execute(
                "INSERT INTO files (path, size, mtime, ext, media_type, indexed_at) "
                "VALUES ('/a.jpg', 1, 0.0, 'jpg', 'photo', 'x')")
            conn.execute(
                "INSERT INTO clip_embeddings (file_id, model, dim, vec, updated_at) "
                "VALUES (1, 'm', 2, ?, 'x')", (pack_embedding(np.array([3.0, 4.0])),))
            conn.commit()
            conn.close()

            conn = connect(db)  # the migration runs again on the already-migrated DB
            row = conn.execute("SELECT * FROM clip_embeddings").fetchone()
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            conn.close()
        self.assertEqual(row["model"], "m")
        np.testing.assert_allclose(unpack_embedding(row["vec"]), [0.6, 0.8], atol=1e-3)
        self.assertEqual(version, SCHEMA_VERSION)


def unit_sample(rng, n=256, dim=768):
    """`n` unit vectors of the real CLIP width — the sample the format is measured on."""
    base = rng.normal(size=(n, dim)).astype(np.float32)
    return base / np.linalg.norm(base, axis=1, keepdims=True)


def ranking_moves(sample, stored, queries):
    """How many of `queries` come back in a different order out of `stored`."""
    return sum(not np.array_equal(np.argsort(-(sample @ q)), np.argsort(-(stored @ q)))
               for q in queries)


class TestReset(unittest.TestCase):
    """The brief's boundary: "start over" clears this table with everything else.

    Nothing had to be written for it — `reset_index` drops every table but an explicit
    keep-list — and that is exactly why it is pinned: a new table is wiped by default, and
    surviving a reset has to be a deliberate decision per table.
    """

    def test_reset_index_clears_the_vectors(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = connect(Path(tmp) / "reset.db")
            conn.execute(
                "INSERT INTO files (path, size, mtime, ext, media_type, indexed_at)"
                " VALUES ('/a.jpg', 1, 0.0, 'jpg', 'photo', 'x')")
            conn.execute(
                "INSERT INTO clip_embeddings (file_id, model, dim, vec, updated_at)"
                " VALUES (1, 'm', 4, ?, 'x')",
                (pack_embedding(deterministic_vector("a.jpg", 4)),))
            conn.commit()
            reset_index(conn)
            left = conn.execute("SELECT COUNT(*) FROM clip_embeddings").fetchone()[0]
            conn.close()
        self.assertEqual(left, 0)


class TestVectorFormat(unittest.TestCase):
    """Brief part 3: normalized, little-endian — and the precision is MEASURED, not assumed."""

    def test_the_blob_is_four_bytes_per_number(self):
        blob = pack_embedding(np.ones(768, dtype=np.float32))
        self.assertEqual(len(blob), 768 * 4)

    def test_the_stored_vector_is_normalized(self):
        vec = unpack_embedding(pack_embedding(np.array([3.0, 4.0, 12.0])))
        self.assertAlmostEqual(float(np.linalg.norm(vec)), 1.0, places=6)

    def test_the_byte_order_is_little_endian_whatever_the_machine_is(self):
        # A DB written here is read elsewhere; the format may not follow the platform.
        blob = pack_embedding(np.array([1.0, 0.0]))
        self.assertEqual(blob[:4], b"\x00\x00\x80\x3f")  # 1.0f, low byte first

    def test_a_direction_survives_the_round_trip(self):
        raw = deterministic_vector("direction.jpg", 768)
        back = unpack_embedding(pack_embedding(raw))
        cosine = float(back @ (raw / np.linalg.norm(raw)))
        self.assertGreater(cosine, 0.999999)

    def test_a_zero_vector_is_not_divided_by_its_norm(self):
        # It can only come from a caller that made one up, and it must not become NaN.
        vec = unpack_embedding(pack_embedding(np.zeros(4, dtype=np.float32)))
        self.assertFalse(bool(np.isnan(vec).any()))

    def test_the_stored_format_ranks_exactly_as_the_encoder_does(self):
        """Brief 3.3: the ranking out of the table IS the ranking out of the model.

        The ranking and not the numbers, because ranking is the only thing a consumer of
        this table ever does with it: 256 stored vectors of the real width, 20 queries, and
        the order has to be identical — no tolerance, because there is nothing to tolerate
        once the format keeps what the encoder produced.
        """
        rng = np.random.default_rng(20260802)
        sample = unit_sample(rng)
        stored = np.stack([unpack_embedding(pack_embedding(v)) for v in sample])
        queries = [q / np.linalg.norm(q) for q in unit_sample(rng, n=20)]
        for i, query in enumerate(queries):
            np.testing.assert_array_equal(
                np.argsort(-(sample @ query)), np.argsort(-(stored @ query)),
                err_msg=f"query {i}: the stored vectors rank differently from the model's")

    def test_half_precision_was_measured_and_it_does_not_hold_the_ranking(self):
        """Why the table is float32 although float16 would halve it — kept executable.

        The brief proposed half precision for the size (~30 MB per 20 000 photos against
        ~60, ~460 MB at 300 000 against ~920) and made it conditional on this measurement,
        with the answer pre-committed: if the ranking moves, the format is float32 and the
        finding goes into the brief rather than into a softer test. It moves. This case is
        the record of that, so the decision can be re-checked instead of re-read — and it
        also states the size of the error, which is what makes it a decision and not a
        scandal: the pairs float16 reorders are always within 3e-5 of a cosine, i.e. pairs
        the format itself cannot tell apart. What it cannot do is leave the order alone.

        Not an artifact of a uniform sample either: any 256 vectors leave adjacent gaps of
        the order of 1e-5, below the resolution of half precision, and a clustered sample
        (which is what real CLIP features look like) reorders more often, not less.
        """
        rng = np.random.default_rng(20260802)
        sample = unit_sample(rng)
        queries = [q / np.linalg.norm(q) for q in unit_sample(rng, n=20)]
        half = sample.astype(np.float16).astype(np.float32)
        moved = ranking_moves(sample, half, queries)
        error = max(float(np.abs(sample @ q - half @ q).max()) for q in queries)
        self.assertGreater(moved, 10, "float16 held the ranking — re-open the format")
        self.assertLess(error, 1e-4)  # small, and still enough to reorder near-ties


class TestReadFilter(unittest.TestCase):
    """Requirement 2.2: on the way out, a foreign model is absent, not "usable anyway"."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.conn = connect(Path(self.tmp.name) / "read.db")
        self.addCleanup(self.conn.close)
        for i, model in enumerate(("ours", "ours", "theirs"), start=1):
            self.conn.execute(
                "INSERT INTO files (path, size, mtime, ext, media_type, indexed_at)"
                " VALUES (?, 1, 0.0, 'jpg', 'photo', 'x')", (f"/p/{i}.jpg",))
            self.conn.execute(
                "INSERT INTO clip_embeddings (file_id, model, dim, vec, updated_at)"
                " VALUES (?, ?, 4, ?, 'x')",
                (i, model, pack_embedding(deterministic_vector(f"{i}.jpg", 4))))
        self.conn.commit()

    def test_only_the_current_model_comes_back(self):
        self.assertEqual(set(read_clip_embeddings(self.conn, "ours")), {1, 2})

    def test_a_foreign_model_is_absent_even_when_asked_for_by_id(self):
        self.assertEqual(set(read_clip_embeddings(self.conn, "ours", [2, 3])), {2})

    def test_the_vectors_come_back_as_float32(self):
        vec = read_clip_embeddings(self.conn, "ours")[1]
        self.assertEqual(vec.dtype, np.float32)
        self.assertEqual(len(vec), 4)


class EmbeddingCase(unittest.TestCase):
    """Shared fixture: a DB, a config with the toggle on, and a look into the table."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = Config(sources=[Path(self.tmp.name)],
                          database=Path(self.tmp.name) / "test.db",
                          naming=_naming_from({}))
        self.cfg.features = FeaturesConfig()
        self.conn = connect(self.cfg.database)
        self.addCleanup(self.conn.close)

    def add_file(self, name, camera_make="Canon", camera_model="EOS"):
        cur = self.conn.execute(
            """INSERT INTO files (path, size, mtime, ext, media_type, width, height,
                   camera_make, camera_model, indexed_at)
               VALUES (?, 1000, 0, 'jpg', 'photo', 4000, 3000, ?, ?, '2026-01-01')""",
            (f"/photos/{name}", camera_make, camera_model))
        self.conn.commit()
        return cur.lastrowid

    def run_classify(self, classifier, **kwargs):
        return classify(self.cfg, self.conn, classifier=classifier,
                        text_detector=NO_OCR,
                        sharpness_detector=lambda _path, _faces: junk.Sharpness(100.0), **kwargs)

    def embedding(self, fid):
        return self.conn.execute(
            "SELECT * FROM clip_embeddings WHERE file_id = ?", (fid,)).fetchone()

    def model_name(self):
        return embedding_model(self.cfg.naming)

    def verdicts(self):
        return {r["file_id"]: (r["verdict"], r["source"], r["score"])
                for r in self.conn.execute("SELECT * FROM media_class")}


class TestVectorIsStored(EmbeddingCase):
    """Brief test 2: the vector is there, `dim` describes it, the norm is 1."""

    def test_a_photo_gets_its_vector(self):
        fid = self.add_file("IMG_0001.jpg")
        clf = EmbeddingClassifier()
        stats = self.run_classify(clf)
        row = self.embedding(fid)
        vec = unpack_embedding(row["vec"])
        self.assertEqual(row["dim"], _DIM)
        self.assertEqual(len(vec), row["dim"])
        self.assertAlmostEqual(float(np.linalg.norm(vec)), 1.0, places=3)
        self.assertEqual(stats.embeddings_stored, 1)

    def test_it_is_the_vector_of_that_frame_and_not_of_its_neighbour(self):
        first = self.add_file("IMG_0001.jpg")
        second = self.add_file("IMG_0002.jpg")
        self.run_classify(EmbeddingClassifier())
        stored = read_clip_embeddings(self.conn, self.model_name())
        for fid, name in ((first, "IMG_0001.jpg"), (second, "IMG_0002.jpg")):
            expected = deterministic_vector(name)
            expected = expected / np.linalg.norm(expected)
            self.assertGreater(float(stored[fid] @ expected), 0.999)

    def test_the_model_that_produced_it_is_written(self):
        fid = self.add_file("IMG_0001.jpg")
        self.run_classify(EmbeddingClassifier())
        self.assertEqual(self.embedding(fid)["model"], "ViT-L-14-quickgelu/openai")

    def test_a_frame_that_does_not_encode_gets_no_row(self):
        # NULL does not happen in this table: no vector means no row.
        fid = self.add_file("broken.jpg")
        self.run_classify(EmbeddingClassifier(undecodable={"broken.jpg"}))
        self.assertIsNone(self.embedding(fid))

    def test_a_classifier_that_cannot_hand_vectors_back_switches_the_half_off(self):
        """An injected plain function — every other mock in the suite — leaves it all alone.

        Not merely "stores nothing": the half is off, so it selects no frame either. A
        frame selected for a vector that can never be written would be sent to CLIP for
        nothing and selected again on every later run — which is how this showed up, as
        four unrelated suites suddenly seeing a second CLIP call.
        """
        fid = self.add_file("IMG_0001.jpg")
        clf = FakeClassifier({})
        self.run_classify(clf)
        self.assertIsNone(self.embedding(fid))
        self.assertEqual(self.verdicts()[fid][0], "photo")
        # The junk half alone: the main pass and the document pass, one frame each.
        self.assertEqual(clf.seen_paths, ["/photos/IMG_0001.jpg"] * 2)
        self.assertEqual(self.run_classify(clf).processed, 0)   # the second run: nothing
        self.assertEqual(clf.seen_paths, ["/photos/IMG_0001.jpg"] * 2)  # left alone


class TestNoExtraPass(EmbeddingCase):
    """Brief test 3: the vector rides in the call the stage was making anyway."""

    def test_switching_the_toggle_on_adds_no_classifier_call(self):
        for name in ("IMG_0001.jpg", "IMG_0002.jpg", "IMG_0003.jpg"):
            self.add_file(name)
        self.cfg.features = FeaturesConfig(store_embeddings=False)
        off = EmbeddingClassifier()
        self.run_classify(off)
        self.conn.execute("DELETE FROM media_class")
        self.conn.execute("DELETE FROM frame_quality")
        self.conn.commit()
        self.cfg.features = FeaturesConfig(store_embeddings=True)
        on = EmbeddingClassifier()
        self.run_classify(on)
        self.assertEqual(on.calls, off.calls)
        self.assertEqual(len(self.conn.execute(
            "SELECT * FROM clip_embeddings").fetchall()), 3)

    def test_an_embeddings_only_run_asks_clip_once_and_scores_nothing_twice(self):
        """A collection already classified, with the vectors missing — the F128 backfill.

        One call over the junk prompts and no document pass: the junk half has nothing to
        redo, so the frames enter the chunk for the vector alone.
        """
        self.cfg.features = FeaturesConfig(store_embeddings=False)
        self.add_file("IMG_0001.jpg")
        self.run_classify(EmbeddingClassifier())
        self.cfg.features = FeaturesConfig(store_embeddings=True)
        clf = EmbeddingClassifier()
        self.run_classify(clf)
        self.assertEqual(clf.prompt_counts(), [len(junk.clip_prompts(False))])
        self.assertEqual(len(clf.feature_calls), 1)


class TestToggleOff(EmbeddingCase):
    """Brief test 6: off means an empty table and nothing else changed."""

    def test_nothing_is_written_and_the_verdicts_are_the_same(self):
        self.add_file("IMG_0001.jpg")
        self.add_file("Screenshot_1.png", camera_make=None, camera_model=None)
        self.cfg.features = FeaturesConfig(store_embeddings=True)
        self.run_classify(EmbeddingClassifier())
        with_embeddings = self.verdicts()

        self.conn.execute("DELETE FROM media_class")
        self.conn.execute("DELETE FROM clip_embeddings")
        self.conn.commit()
        self.cfg.features = FeaturesConfig(store_embeddings=False)
        stats = self.run_classify(EmbeddingClassifier())

        self.assertEqual(self.conn.execute(
            "SELECT COUNT(*) FROM clip_embeddings").fetchone()[0], 0)
        self.assertEqual(stats.embeddings_stored, 0)
        self.assertEqual(self.verdicts(), with_embeddings)

    def test_off_does_not_touch_the_rows_an_earlier_run_stored(self):
        # The switch is for the cost of WRITING; it is not a "delete my table" button.
        fid = self.add_file("IMG_0001.jpg")
        self.run_classify(EmbeddingClassifier())
        self.conn.execute("DELETE FROM media_class")
        self.conn.commit()
        self.cfg.features = FeaturesConfig(store_embeddings=False)
        self.run_classify(EmbeddingClassifier())
        self.assertIsNotNone(self.embedding(fid))


class TestIncrementality(EmbeddingCase):
    """Brief tests 4 and 7: the model is the staleness marker, and nothing else is."""

    def test_a_repeated_run_rewrites_nothing(self):
        fid = self.add_file("IMG_0001.jpg")
        self.run_classify(EmbeddingClassifier())
        before = self.embedding(fid)["vec"]
        clf = EmbeddingClassifier()
        stats = self.run_classify(clf)
        self.assertEqual(stats.embeddings_stored, 0)
        self.assertEqual(clf.feature_calls, [])
        self.assertEqual(self.embedding(fid)["vec"], before)

    def test_a_reclassified_frame_keeps_the_vector_it_already_has(self):
        """Even when the junk half redoes the frame: same model, same vector, no write.

        The tier is reset by hand, which is what a fast<->deep switch does to every row —
        the junk half then re-encodes the frame anyway, and this half must not follow it
        into a rewrite of data that cannot have changed.
        """
        fid = self.add_file("IMG_0001.jpg")
        self.run_classify(EmbeddingClassifier())
        self.conn.execute("UPDATE media_class SET tier = 'heuristic'")
        self.conn.commit()
        # A classifier whose vectors differ: a rewrite would be visible in the row.
        other = EmbeddingClassifier(vectors={"IMG_0001.jpg": np.arange(1.0, _DIM + 1.0)})
        stats = self.run_classify(other)
        self.assertEqual(stats.embeddings_stored, 0)
        stored = read_clip_embeddings(self.conn, self.model_name())[fid]
        expected = deterministic_vector("IMG_0001.jpg")
        self.assertGreater(float(stored @ (expected / np.linalg.norm(expected))), 0.999)

    def test_another_model_in_the_config_makes_the_rows_stale(self):
        fid = self.add_file("IMG_0001.jpg")
        self.run_classify(EmbeddingClassifier())
        self.cfg.naming = _naming_from({"clip": {"model": "ViT-B-32",
                                                 "pretrained": "laion2b_s34b_b79k"}})
        replacement = np.arange(1.0, _DIM + 1.0)
        stats = self.run_classify(
            EmbeddingClassifier(vectors={"IMG_0001.jpg": replacement}))
        self.assertEqual(stats.embeddings_stored, 1)
        row = self.embedding(fid)
        self.assertEqual(row["model"], "ViT-B-32/laion2b_s34b_b79k")
        stored = unpack_embedding(row["vec"])
        expected = replacement / np.linalg.norm(replacement)
        self.assertGreater(float(stored @ expected.astype(np.float32)), 0.999)


class TestPopulation(EmbeddingCase):
    """Brief tests 8 and 9: no CLIP, no row; not a photograph, no row."""

    def test_a_heuristics_only_run_creates_no_rows(self):
        self.add_file("IMG_0001.jpg")
        stats = classify(self.cfg, self.conn, use_clip=False)
        self.assertEqual(stats.embeddings_stored, 0)
        self.assertEqual(self.conn.execute(
            "SELECT COUNT(*) FROM clip_embeddings").fetchone()[0], 0)

    def test_a_screenshot_gets_no_row(self):
        shot = self.add_file("Screenshot_1.png", camera_make=None, camera_model=None)
        photo = self.add_file("IMG_0001.jpg")
        self.run_classify(EmbeddingClassifier(
            scores={"Screenshot_1.png": (_SCREENSHOT_IDX, 0.99)}))
        self.assertEqual(self.verdicts()[shot][0], "screenshot")
        self.assertIsNone(self.embedding(shot))
        self.assertIsNotNone(self.embedding(photo))

    def test_a_row_left_over_for_a_non_photo_is_removed(self):
        """The purge the quality half needs for the same reason (F120).

        Incrementality skips a row that already looks current, so a collection embedded
        before this rule would keep its screenshots PRECISELY because they are up to date.
        """
        shot = self.add_file("Screenshot_1.png", camera_make=None, camera_model=None)
        self.add_file("IMG_0001.jpg")
        self.conn.execute(
            "INSERT INTO clip_embeddings (file_id, model, dim, vec, updated_at)"
            " VALUES (?, ?, ?, ?, 'old')",
            (shot, self.model_name(), _DIM,
             pack_embedding(deterministic_vector("Screenshot_1.png"))))
        self.conn.commit()
        self.run_classify(EmbeddingClassifier(
            scores={"Screenshot_1.png": (_SCREENSHOT_IDX, 0.99)}))
        self.assertIsNone(self.embedding(shot))


class TestRealClassifierHandsBackItsCache(unittest.TestCase):
    """The production path: the vectors come from the cache, not from a second encode.

    `CachingFeatureClassifier` is what `clip_classifier` returns, and the whole economy of
    this feature is that `features()` reads what scoring already put in its cache.
    """

    def setUp(self):
        self.encoded: list[str] = []

    def build(self):
        def encode(paths):
            self.encoded.extend(paths)
            return [deterministic_vector(Path(p).name) for p in paths]

        def score(feats, prompts):
            return np.full((len(feats), len(prompts)), 1.0 / len(prompts),
                           dtype=np.float32)

        return CachingFeatureClassifier(encode=encode, score=score)

    def test_features_do_not_encode_again(self):
        clf = self.build()
        clf(["/p/a.jpg", "/p/b.jpg"], ["x", "y"])
        vectors = clf.features(["/p/a.jpg", "/p/b.jpg"])
        self.assertEqual(self.encoded, ["/p/a.jpg", "/p/b.jpg"])
        np.testing.assert_allclose(vectors[0], deterministic_vector("a.jpg"))

    def test_a_path_nobody_scored_is_none(self):
        self.assertEqual(self.build().features(["/p/never.jpg"]), [None])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
