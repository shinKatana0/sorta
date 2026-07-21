"""F6: landmark detection without GPS — CLIP mocked, threshold, the "unknown only" rule."""
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sorta.config import Config, _naming_from
from sorta.db import connect
from sorta.landmarks import (
    CachingFeatureClassifier,
    Landmark,
    detect_landmarks,
    load_landmarks,
)
from sorta.naming import NamingSettings

LANDMARKS_YAML = """\
landmarks:
  - prompt: "a photo of the Eiffel Tower in Paris"
    name: "Эйфелева башня"
    country: FR
    city: Paris
  - prompt: "a photo of the Colosseum amphitheatre in Rome"
    name: "Колизей"
    country: IT
    city: Rome
"""


class FakeClassifier:
    """Mock CLIP: probability by the file basename; the rest — zeros."""

    def __init__(self, scores):  # {basename: (landmark_index, prob)}
        self.scores = scores
        self.seen_paths = []

    def __call__(self, image_paths, prompts):
        self.seen_paths.extend(image_paths)
        out = np.zeros((len(image_paths), len(prompts)), dtype=np.float32)
        for i, p in enumerate(image_paths):
            idx, prob = self.scores.get(Path(p).name, (0, 0.0))
            out[i, idx] = prob
        return out


def _make_plain_and_caching(scores):
    """A pair of classifiers with identical scoring logic by the file basename:
    `plain` — the old scheme (each classify() decodes+encodes anew),
    `caching` — CachingFeatureClassifier (F19) over separate encode/score.
    Used by the equivalence test: the verdicts must match byte-for-byte.
    """
    def plain(image_paths, prompts):
        out = np.zeros((len(image_paths), len(prompts)), dtype=np.float32)
        for i, p in enumerate(image_paths):
            idx, prob = scores.get(Path(p).name, (0, 0.0))
            out[i, idx] = prob
        return out

    name_to_id: dict = {}
    id_to_name: dict = {}

    def encode(paths):
        result = []
        for p in paths:
            name = Path(p).name
            if name not in name_to_id:
                name_to_id[name] = len(name_to_id)
                id_to_name[name_to_id[name]] = name
            result.append(np.array([name_to_id[name]], dtype=np.float32))
        return result

    def score(feats, prompts):
        out = np.zeros((len(feats), len(prompts)), dtype=np.float32)
        for i, f in enumerate(feats):
            name = id_to_name[int(f[0])]
            idx, prob = scores.get(name, (0, 0.0))
            out[i, idx] = prob
        return out

    caching = CachingFeatureClassifier(encode=encode, score=score)
    return plain, caching


class TestCachingFeatureClassifier(unittest.TestCase):
    """F19: image features are cached by path — encode at most once over the object's
    lifetime, the scorer is called on each classify()."""

    def _counting(self, feature_dim=2):
        encode_calls = []
        score_calls = []

        def encode(paths):
            encode_calls.append(list(paths))
            return [np.full(feature_dim, len(p), dtype=np.float32) for p in paths]

        def score(feats, prompts):
            score_calls.append((len(feats), len(prompts)))
            return np.zeros((len(feats), len(prompts)), dtype=np.float32)

        return CachingFeatureClassifier(encode=encode, score=score), encode_calls, score_calls

    def test_same_paths_encoded_once_scored_each_call(self):
        clf, encode_calls, score_calls = self._counting()
        paths = ["/a.jpg", "/b.jpg"]
        clf(paths, ["p1"])
        clf(paths, ["p1", "p2"])  # a different prompt set, the same paths
        self.assertEqual(encode_calls, [paths])  # encode only on the first call
        self.assertEqual(len(score_calls), 2)  # the scorer — on each call

    def test_new_paths_only_missing_encoded(self):
        clf, encode_calls, _ = self._counting()
        clf(["/a.jpg", "/b.jpg"], ["p1"])
        clf(["/b.jpg", "/c.jpg"], ["p1"])
        self.assertEqual(encode_calls, [["/a.jpg", "/b.jpg"], ["/c.jpg"]])

    def test_broken_file_not_cached_retried_next_call(self):
        attempts = []

        def encode(paths):
            attempts.append(list(paths))
            return [None for _ in paths]  # a "corrupt" file — no features

        def score(feats, prompts):
            self.fail("score must not be called without valid features")

        clf = CachingFeatureClassifier(encode=encode, score=score)
        probs1 = clf(["/broken.jpg"], ["p1", "p2"])
        probs2 = clf(["/broken.jpg"], ["p1", "p2"])
        np.testing.assert_array_equal(probs1, np.zeros((1, 2), dtype=np.float32))
        np.testing.assert_array_equal(probs2, np.zeros((1, 2), dtype=np.float32))
        self.assertEqual(attempts, [["/broken.jpg"], ["/broken.jpg"]])  # retry every time

    def test_partial_broken_mixes_cached_and_retried(self):
        attempts = []

        def encode(paths):
            attempts.append(list(paths))
            return [None if p == "/broken.jpg" else np.array([1.0, 2.0]) for p in paths]

        def score(feats, prompts):
            return np.ones((len(feats), len(prompts)), dtype=np.float32)

        clf = CachingFeatureClassifier(encode=encode, score=score)
        clf(["/ok.jpg", "/broken.jpg"], ["p1"])
        clf(["/ok.jpg", "/broken.jpg"], ["p1"])
        # /ok.jpg is cached after the first call, /broken.jpg — anew every time
        self.assertEqual(attempts, [["/ok.jpg", "/broken.jpg"], ["/broken.jpg"]])


class TestCachingEquivalence(unittest.TestCase):
    """F19: the detector on the caching classifier gives the same places as on the
    plain one (no cache) — the optimization does not change the verdicts."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        yaml_path = Path(self.tmp.name) / "landmarks.yaml"
        yaml_path.write_text(LANDMARKS_YAML, encoding="utf-8")
        self.naming = {"landmarks_file": str(yaml_path), "landmark_threshold": 0.3}

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, classifier, db_name):
        cfg = Config(sources=[Path(self.tmp.name)],
                    database=Path(self.tmp.name) / db_name,
                    naming=_naming_from(self.naming))
        conn = connect(cfg.database)
        for name in ("eiffel.jpg", "colosseum.jpg", "blurry.jpg"):
            cur = conn.execute(
                """INSERT INTO files (path, size, mtime, ext, media_type, indexed_at)
                   VALUES (?, 1000, 0, 'jpg', 'photo', '2026-01-01')""",
                (f"/photos/{name}",))
            fid = cur.lastrowid
            conn.execute(
                """INSERT INTO places (file_id, country, region, city, confidence,
                       updated_at) VALUES (?, NULL, NULL, NULL, 'unknown', '2026-01-01')""",
                (fid,))
        conn.commit()
        detect_landmarks(cfg, conn, classifier=classifier)
        result = {
            r["path"]: (r["country"], r["city"], r["confidence"])
            for r in conn.execute(
                """SELECT f.path, p.country, p.city, p.confidence
                   FROM files f JOIN places p ON p.file_id = f.id ORDER BY f.path""")
        }
        conn.close()
        return result

    def test_caching_classifier_matches_plain(self):
        scores = {"eiffel.jpg": (0, 0.8), "colosseum.jpg": (1, 0.5), "blurry.jpg": (0, 0.2)}
        plain, caching = _make_plain_and_caching(scores)
        plain_result = self._run(plain, "plain.db")
        caching_result = self._run(caching, "caching.db")
        self.assertEqual(plain_result, caching_result)
        self.assertEqual(plain_result["/photos/eiffel.jpg"], ("FR", "Paris", "visual"))


class TestLoadLandmarks(unittest.TestCase):
    def test_load_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "landmarks.yaml"
            f.write_text(LANDMARKS_YAML, encoding="utf-8")
            lms = load_landmarks(f)
        self.assertEqual(len(lms), 2)
        self.assertEqual(lms[0], Landmark(
            prompt="a photo of the Eiffel Tower in Paris",
            name="Эйфелева башня", country="FR", city="Paris"))

    def test_missing_field_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "landmarks.yaml"
            f.write_text("landmarks:\n  - prompt: x\n    name: y\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_landmarks(f)

    def test_repo_default_list_is_valid(self):
        repo_yaml = Path(__file__).resolve().parents[1] / "data" / "landmarks.yaml"
        self.assertTrue(load_landmarks(repo_yaml))


class TestDetectLandmarks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        yaml_path = Path(self.tmp.name) / "landmarks.yaml"
        yaml_path.write_text(LANDMARKS_YAML, encoding="utf-8")
        # set the threshold explicitly — the tests check the matching logic, they do
        # not depend on the production default (raised to 0.85, see backlog #11)
        self.naming = {"landmarks_file": str(yaml_path), "landmark_threshold": 0.3}
        self.cfg = Config(sources=[Path(self.tmp.name)],
                          database=Path(self.tmp.name) / "test.db",
                          naming=_naming_from(self.naming))
        self.conn = connect(self.cfg.database)
        self._n = 0

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def add_file(self, name, confidence="unknown", media_type="photo",
                 dup_of=None, error=None, city=None):
        self._n += 1
        cur = self.conn.execute(
            """INSERT INTO files (path, size, mtime, ext, media_type, dup_of,
                   error, indexed_at)
               VALUES (?, 1000, 0, 'jpg', ?, ?, ?, '2026-01-01')""",
            (f"/photos/{name}", media_type, dup_of, error))
        fid = cur.lastrowid
        if error is None and dup_of is None:
            self.conn.execute(
                """INSERT INTO places (file_id, country, region, city, confidence,
                       updated_at) VALUES (?, NULL, NULL, ?, ?, '2026-01-01')""",
                (fid, city, confidence))
        self.conn.commit()
        return fid

    def place_of(self, fid):
        return self.conn.execute(
            "SELECT country, city, confidence FROM places WHERE file_id = ?",
            (fid,)).fetchone()

    def test_match_above_threshold_writes_visual(self):
        eiffel = self.add_file("eiffel.jpg")
        colosseum = self.add_file("colosseum.jpg")
        clf = FakeClassifier({"eiffel.jpg": (0, 0.8), "colosseum.jpg": (1, 0.5)})
        stats = detect_landmarks(self.cfg, self.conn, classifier=clf)
        self.assertEqual(stats.scanned, 2)
        self.assertEqual(stats.matched, 2)
        self.assertEqual(stats.by_landmark,
                         {"Эйфелева башня": 1, "Колизей": 1})
        self.assertEqual(tuple(self.place_of(eiffel)), ("FR", "Paris", "visual"))
        self.assertEqual(tuple(self.place_of(colosseum)), ("IT", "Rome", "visual"))

    def test_below_threshold_stays_unknown(self):
        fid = self.add_file("blurry.jpg")
        stats = detect_landmarks(self.cfg, self.conn,
                                 classifier=FakeClassifier({"blurry.jpg": (0, 0.2)}))
        self.assertEqual(stats.matched, 0)
        self.assertEqual(self.place_of(fid)["confidence"], "unknown")

    def test_only_unknown_rows_processed(self):
        gps = self.add_file("gps.jpg", confidence="exact_gps", city="Moskva")
        inferred = self.add_file("inferred.jpg", confidence="session_inferred",
                                 city="Moskva")
        unknown = self.add_file("eiffel.jpg")
        clf = FakeClassifier({n: (0, 0.99)
                              for n in ("gps.jpg", "inferred.jpg", "eiffel.jpg")})
        stats = detect_landmarks(self.cfg, self.conn, classifier=clf)
        self.assertEqual(stats.scanned, 1)  # only unknown went to CLIP
        self.assertEqual([Path(p).name for p in clf.seen_paths], ["eiffel.jpg"])
        self.assertEqual(self.place_of(gps)["confidence"], "exact_gps")
        self.assertEqual(self.place_of(gps)["city"], "Moskva")
        self.assertEqual(self.place_of(inferred)["confidence"], "session_inferred")
        self.assertEqual(self.place_of(unknown)["confidence"], "visual")

    def test_threshold_from_config(self):
        self.naming["landmark_threshold"] = 0.9
        self.cfg.naming = _naming_from(self.naming)
        fid = self.add_file("eiffel.jpg")
        detect_landmarks(self.cfg, self.conn,
                         classifier=FakeClassifier({"eiffel.jpg": (0, 0.8)}))
        self.assertEqual(self.place_of(fid)["confidence"], "unknown")

    def test_second_run_skips_visual_rows(self):
        self.add_file("eiffel.jpg")
        clf = FakeClassifier({"eiffel.jpg": (0, 0.8)})
        detect_landmarks(self.cfg, self.conn, classifier=clf)
        stats2 = detect_landmarks(self.cfg, self.conn, classifier=clf)
        self.assertEqual(stats2.scanned, 0)  # visual rows are not recomputed
        self.assertEqual(len(clf.seen_paths), 1)

    def test_skips_duplicates_errors_and_videos(self):
        canon = self.add_file("eiffel.jpg")
        self.add_file("dup.jpg", dup_of=canon)
        self.add_file("broken.jpg", error="boom")
        self.add_file("clip.mp4", media_type="video")
        stats = detect_landmarks(self.cfg, self.conn,
                                 classifier=FakeClassifier({}))
        self.assertEqual(stats.scanned, 1)

    def test_progress_first_call_has_full_total(self):
        # F52 (#37): a small stage (< clip_batch_size) — progress used not to be
        # called at all, GPS-less files showed "0 of 0".
        self.add_file("eiffel.jpg")
        calls = []
        detect_landmarks(
            self.cfg, self.conn, classifier=FakeClassifier({"eiffel.jpg": (0, 0.8)}),
            progress=lambda done, total: calls.append((done, total)))
        self.assertTrue(calls)
        self.assertEqual(calls[0], (0, 1))
        self.assertEqual(calls[-1], (1, 1))


class TestRealClipSmoke(unittest.TestCase):
    """The only smoke with a real model; enabled manually:
    SORTA_CLIP_SMOKE=1 (downloads the open_clip weights, slow)."""

    @unittest.skipUnless(os.environ.get("SORTA_CLIP_SMOKE"),
                         "set SORTA_CLIP_SMOKE=1 for a smoke with real CLIP")
    def test_real_model_classifies_generated_image(self):
        from PIL import Image

        from sorta.landmarks import clip_classifier
        clf = clip_classifier(NamingSettings(
            clip_model="ViT-B-32", clip_pretrained="laion2b_s34b_b79k"))
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "red.png"
            Image.new("RGB", (224, 224), (255, 0, 0)).save(p)
            probs = clf([str(p)], ["a solid red image", "a photo of a dog"])
        self.assertEqual(probs.shape, (1, 2))
        self.assertGreater(probs[0, 0], probs[0, 1])


if __name__ == "__main__":
    unittest.main()
