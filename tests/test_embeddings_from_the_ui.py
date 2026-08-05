"""F146: `clip_embeddings` fills on BOTH ways of starting a run, not only from the CLI.

The F128 tests all passed while the table stayed empty for every user of the web app,
because they called `junk.classify` directly with a real classifier and the hole was
between the layers: the wrappers each entry point hands to the stage
(`ui._LazyClassifierHolder`, `cli._LazySharedClassifier`) forwarded `__call__` and
nothing else, so the stage found no `features` on them, decided it could not store
vectors, and skipped that half without a word.

So the cases below start the stage the way the entry points start it — through
`_pipeline_steps()`, with the wrapper in the middle and the CLIP model replaced by the
factory the wrapper calls — and check the table afterwards. The general rule behind it:
a feature reachable by two paths is tested along both, or one of them eventually walks
away from the other in silence.

What is pinned:

* the wrapper hands the vectors of the real classifier back and stays lazy — asking it
  for features builds no model, and an unbuilt classifier has scored nothing to hand;
* a run started from the web app writes one vector per canonical photograph;
* a run started from the CLI writes the same vectors;
* `store_embeddings: false` writes nothing on either path, and says nothing either;
* a classifier that cannot hand vectors back leaves a warning with the reason, which is
  what was missing for the bug to live from F128 to a production run.

No model is loaded anywhere below: the factory is replaced, exactly as the rest of the
suite replaces it.
"""
from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from sorta import cli, junk, ui
from sorta.config import Config, FeaturesConfig, _naming_from
from sorta.db import connect
from sorta.junk import unpack_embedding
from sorta.landmarks import CachingFeatureClassifier
from tests.test_clip_embeddings import deterministic_vector
from tests.test_junk import NO_OCR, FakeClassifier

_LOGGER = "sorta.junk"


def caching_classifier(encoded: list[str]) -> CachingFeatureClassifier:
    """What `clip_classifier` returns in production, minus the model.

    A real `CachingFeatureClassifier` and not a mock with a dict of vectors: the object
    under test is the wrapper around it, and the promise is that `features()` reaches the
    cache the scoring call has just filled.
    """

    def encode(paths: list[str]) -> list[np.ndarray]:
        encoded.extend(paths)
        return [deterministic_vector(Path(p).name) for p in paths]

    def score(feats: np.ndarray, prompts: list[str]) -> np.ndarray:
        # Index 0 is "a photograph" in both prompt sets the stage uses (the junk classes
        # and the document pass), and a confident photograph is what keeps a frame in the
        # population that gets a vector at all.
        out = np.full((len(feats), len(prompts)), 0.01 / max(1, len(prompts) - 1),
                      dtype=np.float32)
        out[:, 0] = 0.99
        return out

    return CachingFeatureClassifier(encode=encode, score=score)


class TestWrappersProxyFeatures(unittest.TestCase):
    """Brief test 1: both wrappers hand the vectors back, and both stay lazy.

    The two classes are checked in one place because they are the same class twice — the
    UI one was written after the CLI one, and the fix has to hold for whichever of them a
    future entry point is copied from next.
    """

    wrappers = (("ui", ui._LazyClassifierHolder), ("cli", cli._LazySharedClassifier))

    def test_features_come_from_the_classifier_the_factory_built(self):
        for name, wrapper in self.wrappers:
            with self.subTest(entry_point=name):
                encoded: list[str] = []
                held = wrapper(lambda: caching_classifier(encoded))
                held(["/p/a.jpg", "/p/b.jpg"], ["x", "y", "z"])
                vectors = held.features(["/p/a.jpg", "/p/b.jpg"])
                self.assertEqual(encoded, ["/p/a.jpg", "/p/b.jpg"])  # no second encode
                np.testing.assert_allclose(vectors[0], deterministic_vector("a.jpg"))
                np.testing.assert_allclose(vectors[1], deterministic_vector("b.jpg"))

    def test_asking_for_features_builds_no_model(self):
        """Laziness, the property the wrapper exists for, in the direction the fix touched.

        A run whose landmarks and junk have nothing to do must not load CLIP, and a method
        added to the wrapper is a new way to break that. Nothing has been scored, so there
        is nothing to hand back — None per path, and no factory call to produce it.
        """
        for name, wrapper in self.wrappers:
            with self.subTest(entry_point=name):
                builds: list[int] = []

                def factory() -> CachingFeatureClassifier:
                    builds.append(1)
                    return caching_classifier([])

                held = wrapper(factory)
                self.assertEqual(held.features(["/p/never.jpg"]), [None])
                self.assertEqual(builds, [])

    def test_a_classifier_without_the_method_costs_nothing(self):
        """A factory whose product cannot hand features back is not an exception.

        It cannot happen with today's `clip_classifier`, and if it ever does the run must
        continue without vectors rather than stop — the stage says so in the log (see
        TestTheSkipIsNotSilent) instead of crashing a pipeline over an optional half.
        """
        for name, wrapper in self.wrappers:
            with self.subTest(entry_point=name):
                held = wrapper(lambda: (lambda paths, prompts: np.zeros(
                    (len(paths), len(prompts)), dtype=np.float32)))
                held(["/p/a.jpg"], ["x"])
                self.assertEqual(held.features(["/p/a.jpg"]), [None])


class EntryPointCase(unittest.TestCase):
    """A collection on disk, a DB, and the junk stage started the way a run starts it."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cfg = Config(sources=[self.root], database=self.root / "test.db",
                          naming=_naming_from({}))
        self.cfg.features = FeaturesConfig()
        self.conn = connect(self.cfg.database)
        self.addCleanup(self.conn.close)
        self.encoded: list[str] = []
        self.builds: list[int] = []

    def add_photo(self, name: str, dup_of: int | None = None,
                  camera_make: str | None = "Canon",
                  camera_model: str | None = "EOS") -> int:
        """A real (small) image on disk plus its row — the stage reads the file for sharpness."""
        path = self.root / name
        Image.new("RGB", (16, 16), (120, 90, 60)).save(path)
        cur = self.conn.execute(
            """INSERT INTO files (path, size, mtime, ext, media_type, width, height,
                   camera_make, camera_model, dup_of, indexed_at)
               VALUES (?, 1000, 0, ?, 'photo', 4000, 3000, ?, ?, ?, '2026-01-01')""",
            (str(path), path.suffix.lstrip("."), camera_make, camera_model, dup_of))
        self.conn.commit()
        return int(cur.lastrowid or 0)

    def run_stage(self, module) -> None:
        """Run junk through `module._pipeline_steps()` — wrapper included, model excluded.

        Everything between the entry point and the table is the real thing: the wrapper the
        module builds, the real `junk.classify`, the real `CachingFeatureClassifier`. Only
        the CLIP weights are replaced, at the one seam the wrapper's factory goes through.
        """

        def factory(_settings):
            self.builds.append(1)
            return caching_classifier(self.encoded)

        # Patch where the name is USED, not where it is re-exported. After F182 the web
        # app is a package: `sorta.ui` still exposes `clip_classifier`, but the call lives
        # in `sorta.ui.process`, which bound its own reference at import. Patching the
        # re-export leaves that reference alone — the run then goes through the REAL
        # weights, the vectors still land in the table, and only `builds` staying empty
        # says that the seam this test exists to hold was not held at all.
        target = getattr(module, "process", module)
        with mock.patch.object(target, "clip_classifier", factory):
            steps = dict(module._pipeline_steps())
            steps["junk"](self.cfg, self.conn, lambda done, total: None)

    def clear_the_run(self) -> None:
        """Everything the stage remembers about a previous run — so the next one redoes it."""
        for table in ("media_class", "frame_quality", "clip_embeddings"):
            self.conn.execute(f"DELETE FROM {table}")
        self.conn.commit()

    def stored(self) -> dict[int, bytes]:
        return {r["file_id"]: r["vec"]
                for r in self.conn.execute("SELECT file_id, vec FROM clip_embeddings")}


class TestBothEntryPointsStoreVectors(EntryPointCase):
    """Brief tests 2 and 3: the web app fills the table, and so does the CLI.

    The count is the number of CANONICAL photographs — a duplicate is not walked by the
    stage at all, and a screenshot is not something a search over personal photos should
    ever return.
    """

    def setUp(self):
        super().setUp()
        self.photos = [self.add_photo("IMG_0001.jpg"), self.add_photo("IMG_0002.jpg"),
                       self.add_photo("IMG_0003.jpg")]
        self.duplicate = self.add_photo("IMG_0003_copy.jpg", dup_of=self.photos[2])
        self.screenshot = self.add_photo("Screenshot_1.png", camera_make=None,
                                         camera_model=None)

    def test_a_run_from_the_web_app_fills_the_table(self):
        self.run_stage(ui)
        self.assertEqual(sorted(self.stored()), sorted(self.photos))
        self.assertEqual(self.builds, [1])  # the model was built once, by the wrapper

    def test_a_run_from_the_cli_fills_it_the_same_way(self):
        self.run_stage(cli)
        self.assertEqual(sorted(self.stored()), sorted(self.photos))

    def test_the_two_paths_store_the_same_vectors(self):
        """The rule the brief states in general: two ways in, one result out.

        Byte for byte, because the vectors come out of the same encoder in both runs — a
        difference here would mean one of the paths reached a different half of the stage.
        """
        self.run_stage(ui)
        from_the_ui = self.stored()
        self.clear_the_run()
        self.run_stage(cli)
        self.assertEqual(self.stored(), from_the_ui)
        self.assertTrue(from_the_ui)  # not "both stored nothing, equally"

    def test_the_vector_is_the_one_of_that_frame(self):
        self.run_stage(ui)
        vec = unpack_embedding(self.stored()[self.photos[0]])
        expected = deterministic_vector("IMG_0001.jpg")
        expected = expected / np.linalg.norm(expected)
        self.assertGreater(float(vec @ expected), 0.999)

    def test_the_frames_are_encoded_once_for_the_vectors_of_the_whole_run(self):
        """The F128 economy survives the wrapper: no pass of its own for the vectors."""
        self.run_stage(ui)
        self.assertEqual(sorted(self.encoded),
                         sorted(str(self.root / name) for name in
                                ("IMG_0001.jpg", "IMG_0002.jpg", "IMG_0003.jpg",
                                 "Screenshot_1.png")))


class TestToggleOff(EntryPointCase):
    """Brief test 4: off means an empty table on both paths — and no complaint about it."""

    def setUp(self):
        super().setUp()
        self.cfg.features = FeaturesConfig(store_embeddings=False)
        self.add_photo("IMG_0001.jpg")

    def test_neither_path_writes_a_vector(self):
        for name, module in (("ui", ui), ("cli", cli)):
            with self.subTest(entry_point=name):
                self.clear_the_run()
                self.run_stage(module)
                self.assertEqual(self.stored(), {})

    def test_the_run_stays_quiet(self):
        # The warning is about a table that was asked for and did not fill; a table nobody
        # asked for is not a problem, and a warning on every run would train the eye past it.
        with self.assertNoLogs(_LOGGER, level=logging.WARNING):
            self.run_stage(ui)


class TestTheSkipIsNotSilent(unittest.TestCase):
    """Brief test 5: a classifier that cannot produce vectors is reported, not ignored.

    This is what was missing for the bug to survive from F128 to the production run of
    2026-08-02: the half switched itself off correctly and said nothing, and an empty
    `clip_embeddings` reads exactly like a collection nobody has processed yet.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.cfg = Config(sources=[Path(self.tmp.name)],
                          database=Path(self.tmp.name) / "test.db",
                          naming=_naming_from({}))
        self.conn = connect(self.cfg.database)
        self.addCleanup(self.conn.close)
        self.conn.execute(
            """INSERT INTO files (path, size, mtime, ext, media_type, width, height,
                   camera_make, camera_model, indexed_at)
               VALUES ('/photos/IMG_0001.jpg', 1000, 0, 'jpg', 'photo', 4000, 3000,
                       'Canon', 'EOS', '2026-01-01')""")
        self.conn.commit()

    def classify(self, **kwargs):
        return junk.classify(self.cfg, self.conn, classifier=FakeClassifier({}),
                             text_detector=NO_OCR,
                             sharpness_detector=lambda _path, _faces: junk.Sharpness(100.0), **kwargs)

    def test_a_classifier_without_features_leaves_the_reason_in_the_log(self):
        self.cfg.features = FeaturesConfig(store_embeddings=True)
        with self.assertLogs(_LOGGER, level=logging.WARNING) as caught:
            self.classify()
        message = "\n".join(caught.output)
        self.assertIn("clip_embeddings", message)
        self.assertIn("store_embeddings", message)
        self.assertIn("FakeClassifier", message)  # which classifier, not just "some"

    def test_the_toggle_off_says_nothing(self):
        self.cfg.features = FeaturesConfig(store_embeddings=False)
        with self.assertNoLogs(_LOGGER, level=logging.WARNING):
            self.classify()

    def test_a_heuristics_only_run_says_nothing(self):
        # No CLIP was asked for, so no vector was ever expected — the same reason the
        # half itself does not select a single frame there.
        self.cfg.features = FeaturesConfig(store_embeddings=True)
        with self.assertNoLogs(_LOGGER, level=logging.WARNING):
            junk.classify(self.cfg, self.conn, use_clip=False)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
