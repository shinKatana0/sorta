"""F113: the frame-quality cascade — classic -> CLIP, and the columns nobody asks about.

The properties under test are the ones the feature is about, not the plumbing:

* the migration creates `frame_quality`, raises `user_version` and runs twice safely;
* sharpness is written with every toggle off (it costs milliseconds and everybody wants it);
* `features.pets` off leaves the pet columns empty; on, they fill — and WITHOUT a second
  CLIP pass: the classifier makes exactly the calls it made before the feature existed,
  and the junk verdicts do not move under the pet prompts;
* the three answer columns (`eyes_open`, `has_subject`, `is_accidental`) stay NULL on
  every run — F186 retired the question that filled them, and NULL is what "not asked"
  means in a column that is still read;
* NULL and False stay distinguishable on the way back out;
* the second run measures nothing again (incrementality on `frame_quality.source`).

F186 retired the third tier of the cascade — the model asked about the frames of the
uncertain band. What went with it: the prompt and its parser, the scope that chose who was
asked, and the cases that drove a fake asker through `classify`. What stayed: the band
itself (`scripts/measure_frame_quality.py` prices the cascade over it) and the columns.

No model is loaded anywhere below: the classifier and the sharpness detector are injected,
exactly as the rest of the junk suite does it.
"""
from __future__ import annotations

import sqlite3
import tempfile
import unittest
import unittest.mock
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from sorta import junk
from sorta.config import Config, FeaturesConfig, VlmConfig, _naming_from
from sorta.db import SCHEMA_VERSION, connect
from sorta.junk import (
    classify,
    clip_prompts,
    laplacian_variance,
    pet_verdict,
    read_frame_quality,
    uncertain_band,
)
from tests.schema_history import roll_back_before
from tests.test_junk import NO_OCR, FakeClassifier

_PET_CLASSES = [cls for cls, _prompt in junk._PET_CLASSES]
_CAT_IDX = len(junk._CLIP_CLASSES) + _PET_CLASSES.index("cat")
_DOG_IDX = len(junk._CLIP_CLASSES) + _PET_CLASSES.index("dog")
_PHOTO_IDX = 0  # "a photograph", the junk group's first class


class QualityClassifier:
    """A CLIP mock that behaves like CLIP: a softmax over per-prompt LOGITS.

    Logits and not ready-made probabilities on purpose. The pet prompts join the same
    softmax as the junk classes, and the claim under test is that the junk classes keep
    the probabilities they had before — a claim only a mock built out of logits can
    answer, because a mock that assigns probabilities directly answers a question about
    the mock. `logits` is keyed by basename and by index in the FULL prompt row, and an
    index past the end of the current list is simply not there: the same table then drives
    a run with pets off and a run with pets on.

    Every call is logged, because the other promise of the feature is that switching pets
    on adds no call at all.
    """

    def __init__(self, logits=None, doc_scores=None):
        self.logits = logits or {}          # {basename: {prompt index: logit}}
        self.doc_scores = doc_scores or {}  # {basename: (index, probability)} — the doc pass
        self.calls: list[tuple[int, tuple[str, ...]]] = []  # (n_prompts, paths)

    def __call__(self, image_paths, prompts):
        self.calls.append((len(prompts), tuple(image_paths)))
        out = np.zeros((len(image_paths), len(prompts)), dtype=np.float32)
        for i, path in enumerate(image_paths):
            name = Path(path).name
            if len(prompts) == len(junk._DOCUMENT_CLASSES):
                idx, score = self.doc_scores.get(name, (0, 0.99))
                out[i, idx] = score
                remainder = max(0.0, (1.0 - score) / max(1, len(prompts) - 1))
                for j in range(len(prompts)):
                    if j != idx:
                        out[i, j] = remainder
                continue
            row = np.zeros(len(prompts), dtype=np.float64)
            for index, value in self.logits.get(name, {}).items():
                if index < len(prompts):
                    row[index] = value
            exps = np.exp(row - row.max())
            out[i] = exps / exps.sum()
        return out

    def prompt_counts(self) -> list[int]:
        return [n for n, _paths in self.calls]


# A logit large enough that its class takes the softmax past every threshold in play
# (e^4 / (e^4 + 2) = 0.96 over the three junk classes), and one that deliberately does not.
CONFIDENT = 4.0
UNSURE = 1.5


def flat_sharpness(value, face=None):
    """A sharpness detector that answers `value` for every frame (files are not on disk).

    F155: the detector now answers with both laplacians, so the helper takes the face one
    too — defaulted to None, which is what a frame with no face gets.
    """
    return lambda _path, _faces=junk.NO_FACES: junk.Sharpness(frame=value, face=face)


def sharpness_by_path(values, faces=None):
    """A detector reading a {path: frame sharpness} table; a path it does not know is None."""
    by_face = faces or {}
    return lambda path, _faces=junk.NO_FACES: junk.Sharpness(
        frame=values.get(path), face=by_face.get(path))


class FrameQualityCase(unittest.TestCase):
    """Shared fixture: an in-memory-ish DB, a config, and helpers over frame_quality."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = Config(sources=[Path(self.tmp.name)],
                          database=Path(self.tmp.name) / "test.db",
                          naming=_naming_from({}))
        self.conn = connect(self.cfg.database)
        self.addCleanup(self.conn.close)

    def features(self, **kwargs):
        self.cfg.features = FeaturesConfig(**kwargs)

    def vlm(self, **kwargs):
        self.cfg.vlm = VlmConfig(**kwargs)

    def deep_analysis_on(self):
        """F145: `vlm.enabled` — the master switch every VLM question of this stage needs.

        A subordinate key (`vlm.products`, `features.pets_verify`,
        `features.junk_rescue`) says WHAT to ask; this one says whether a model may be
        raised at all, so a case about any of them has to switch it on.

        It is the deep junk tier's own toggle too, and no case here is about that tier —
        hence the stubbed factory rather than a `vlm_classifier` argument at every
        classify() call: it answers `personal_photo`, so it moves no verdict, and above
        all it loads no weights on a machine that has the [vlm] extra installed. A case
        that injects a classifier of its own still wins, that argument is checked first.
        """
        self.cfg.naming = replace(self.cfg.naming, vlm_enabled=True)
        patch = unittest.mock.patch.object(
            junk, "qwen_vlm_classifier_factory",
            lambda _max_edge: (lambda _model: (lambda _path: "personal_photo")))
        patch.start()
        self.addCleanup(patch.stop)

    def add_file(self, name, camera_make="Canon", camera_model="EOS", phash=None,
                 has_face=False):
        cur = self.conn.execute(
            """INSERT INTO files (path, size, mtime, ext, media_type, width, height,
                   camera_make, camera_model, gps_lat, phash, indexed_at)
               VALUES (?, 1000, 0, 'jpg', 'photo', 4000, 3000, ?, ?, NULL, ?,
                       '2026-01-01')""",
            (f"/photos/{name}", camera_make, camera_model, phash))
        fid = cur.lastrowid
        if has_face:
            self.conn.execute(
                "INSERT INTO faces (file_id, bbox, embedding) VALUES (?, '[1,2,3,4]', ?)",
                (fid, b"\x00" * 4))
        self.conn.commit()
        return fid

    def quality(self, fid):
        return self.conn.execute(
            "SELECT * FROM frame_quality WHERE file_id = ?", (fid,)).fetchone()

    def media_class(self, fid):
        return self.conn.execute(
            "SELECT verdict, source, score FROM media_class WHERE file_id = ?",
            (fid,)).fetchone()


class TestMigration(unittest.TestCase):
    """Brief test 1: the table appears, the version moves, a repeat run changes nothing.

    Every connection is closed inside its temp directory: on Windows an open sqlite handle
    makes the rmtree fail (the same reason test_junk_tier explains at its own fixture).
    """

    def test_fresh_db_has_the_table_and_the_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = connect(Path(tmp) / "fresh.db")
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(frame_quality)")}
            (version,) = conn.execute("PRAGMA user_version").fetchone()
            conn.close()
        self.assertEqual(cols, {"file_id", "sharpness", "face_sharpness", "eye_openness",
                                "pet", "pet_score", "pet_vlm", "eyes_open", "has_subject",
                                "is_accidental", "junk_score", "source", "updated_at"})
        self.assertEqual(version, SCHEMA_VERSION)

    def test_a_db_from_before_the_table_gains_it_without_touching_its_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "old.db"
            conn = connect(db)
            conn.execute(
                "INSERT INTO files (path, size, mtime, ext, media_type, indexed_at) "
                "VALUES ('/a.jpg', 1, 0.0, 'jpg', 'photo', 'x')")
            roll_back_before(conn, "frame_quality")
            conn.commit()
            conn.close()

            conn = connect(db)
            tables = {r["name"] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            files = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
            conn.close()
        self.assertIn("frame_quality", tables)
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
                "INSERT INTO frame_quality (file_id, sharpness, source, updated_at) "
                "VALUES (1, 12.5, 'classic', 'x')")
            conn.commit()
            conn.close()

            conn = connect(db)  # the migration runs again on the already-migrated DB
            row = conn.execute("SELECT * FROM frame_quality").fetchone()
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            conn.close()
        self.assertAlmostEqual(row["sharpness"], 12.5)
        self.assertEqual(version, SCHEMA_VERSION)


class TestLaplacian(unittest.TestCase):
    """The classic tier itself: the number has to separate a blurred frame from a sharp one."""

    def test_flat_image_has_no_variance(self):
        self.assertEqual(laplacian_variance(Image.new("L", (32, 32), 128)), 0.0)

    def test_noise_is_sharper_than_a_gradient(self):
        rng = np.random.default_rng(7)
        noise = Image.fromarray(rng.integers(0, 255, (64, 64), dtype=np.uint8))
        gradient = Image.fromarray(
            np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (64, 1)))
        self.assertGreater(laplacian_variance(noise), laplacian_variance(gradient))

    def test_frame_without_an_interior_is_none_not_zero(self):
        # "nothing to measure" and "completely flat" are different statements.
        self.assertIsNone(laplacian_variance(Image.new("L", (2, 2), 10)))

    def test_detector_returns_none_for_a_missing_file(self):
        detector = junk.preview_sharpness_detector(256)
        measured = detector("/nowhere/at/all.jpg", junk.NO_FACES)
        self.assertIsNone(measured.frame)
        self.assertIsNone(measured.face)


class TestPetGroup(unittest.TestCase):
    """The pet group rides in the junk call — and must not disturb it (brief part 1)."""

    def test_prompts_are_unchanged_with_pets_off(self):
        self.assertEqual(clip_prompts(False),
                         [p for _c, p in junk._CLIP_CLASSES])

    def test_prompts_append_the_pet_group_when_on(self):
        prompts = clip_prompts(True)
        self.assertEqual(prompts[:len(junk._CLIP_CLASSES)], clip_prompts(False))
        self.assertEqual(len(prompts),
                         len(junk._CLIP_CLASSES) + len(junk._PET_CLASSES))

    def test_junk_probabilities_survive_the_longer_prompt_list(self):
        """The renormalization identity: a slice of a softmax IS the softmax of the slice.

        The junk threshold (0.85) was measured against three prompts; if appending the pet
        group moved these numbers, every junk verdict would shift with it.
        """
        logits = np.array([3.0, 1.0, 0.5, 2.0, -1.0, 0.2, 1.4, 0.9])
        full = np.exp(logits) / np.exp(logits).sum()
        alone = np.exp(logits[:3]) / np.exp(logits[:3]).sum()
        np.testing.assert_allclose(
            junk._group_probs(full, junk._JUNK_GROUP), alone, rtol=1e-6)

    def test_pet_score_ignores_the_junk_classes(self):
        row = np.zeros(len(clip_prompts(True)), dtype=np.float32)
        row[_PHOTO_IDX] = 0.6   # most of the mass is on "a photograph"
        row[_CAT_IDX] = 0.3
        row[-1] = 0.1
        pet, score = pet_verdict(row, 0.5)
        # within its own group the cat holds 0.3 / 0.4 = 0.75 — above the threshold
        self.assertEqual(pet, junk.PET_CLASS)
        self.assertAlmostEqual(score, 0.75, places=5)

    def test_below_the_threshold_there_is_no_class_but_there_is_a_score(self):
        row = np.zeros(len(clip_prompts(True)), dtype=np.float32)
        row[_DOG_IDX] = 0.2
        row[-1] = 0.8
        pet, score = pet_verdict(row, 0.5)
        self.assertIsNone(pet)
        self.assertAlmostEqual(score, 0.2, places=5)

    def test_a_row_without_a_pet_group_answers_nothing(self):
        pet, score = pet_verdict(np.array([0.9, 0.05, 0.05], dtype=np.float32), 0.5)
        self.assertIsNone(pet)
        self.assertEqual(score, 0.0)

    def test_a_zero_row_has_no_mass_to_renormalize(self):
        row = np.zeros(len(clip_prompts(True)), dtype=np.float32)  # undecodable frame
        pet, score = pet_verdict(row, 0.5)
        self.assertIsNone(pet)
        self.assertEqual(score, 0.0)


class TestUncertainBand(unittest.TestCase):
    """The band where the cheap tiers decided nothing (brief part 3, item 2).

    F186 retired its consumer — the band used to select the frames the quality model was
    asked about — and the function stayed for `scripts/measure_frame_quality.py`, which
    sweeps `features.sharpness_band_*` and `features.subject_score_min` over it. What it
    answers is unchanged, so the cases below are too.
    """

    def q(self, **kwargs):
        base = dict(pets=False, pet_threshold=0.6, sharpness_max_edge=512,
                    sharpness_band=(30.0, 300.0), subject_score_min=0.9)
        base.update(kwargs)
        return junk.QualitySettings(**base)

    def test_clearly_sharp_and_clearly_a_photo_is_not_asked_about(self):
        self.assertFalse(uncertain_band(900.0, 0.99, self.q()))

    def test_clearly_blurred_is_not_asked_about_either(self):
        # below the band the laplacian has already decided — that is not uncertainty.
        self.assertFalse(uncertain_band(5.0, 0.99, self.q()))

    def test_sharpness_inside_the_band_is_asked_about(self):
        self.assertTrue(uncertain_band(100.0, 0.99, self.q()))

    def test_a_low_clip_subject_score_is_asked_about(self):
        self.assertTrue(uncertain_band(900.0, 0.5, self.q()))

    def test_a_frame_without_sharpness_is_judged_on_the_subject_alone(self):
        self.assertTrue(uncertain_band(None, 0.5, self.q()))
        self.assertFalse(uncertain_band(None, 0.99, self.q()))

    def test_the_boundaries_come_from_the_config(self):
        wide = self.q(sharpness_band=(0.0, 10000.0))
        self.assertTrue(uncertain_band(900.0, 0.99, wide))


class TestSharpnessAlwaysWritten(FrameQualityCase):
    """Brief test 2: no toggle — the laplacian is written on a plain run."""

    def test_written_with_every_toggle_off(self):
        fid = self.add_file("IMG_0001.jpg")
        clf = QualityClassifier()
        stats = classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR,
                         sharpness_detector=flat_sharpness(123.5))
        row = self.quality(fid)
        self.assertAlmostEqual(row["sharpness"], 123.5)
        self.assertEqual(row["source"], junk.QUALITY_SOURCE_CLASSIC)
        self.assertEqual(stats.quality_rows, 1)

    def test_an_undecodable_frame_gets_null_not_zero(self):
        fid = self.add_file("broken.jpg")
        classify(self.cfg, self.conn, classifier=QualityClassifier(),
                 text_detector=NO_OCR, sharpness_detector=flat_sharpness(None))
        self.assertIsNone(self.quality(fid)["sharpness"])

    def test_a_sharpness_only_backfill_builds_no_clip_model(self):
        """The upgrade path: junk already classified, toggles off, only laplacians missing.

        Loading CLIP for that would be the entire cost of the run, for a question nobody
        asked — so no model is built, and none is called either (the stand-in classifier
        raises if it ever is).

        F128 is switched off here because with it on there IS a question for CLIP: the
        vectors of this collection are missing, and filling them is what that feature is.
        The property under test is unchanged — a run with nothing to ask builds nothing.
        """
        self.features(store_embeddings=False)
        fid = self.add_file("IMG_0001.jpg")
        classify(self.cfg, self.conn, classifier=QualityClassifier(),
                 text_detector=NO_OCR, sharpness_detector=flat_sharpness(1.0))
        self.conn.execute("DELETE FROM frame_quality")  # as if the table had just appeared
        self.conn.commit()

        def boom(_settings):
            raise AssertionError("no CLIP model may be built for a sharpness backfill")

        with unittest.mock.patch.object(junk, "clip_classifier", boom):
            classify(self.cfg, self.conn, text_detector=NO_OCR,
                     sharpness_detector=flat_sharpness(9.0))
        self.assertAlmostEqual(self.quality(fid)["sharpness"], 9.0)

    def test_the_default_detector_needs_no_model(self):
        # the pipeline default (preview + laplacian) on a path that does not exist:
        # a row is still written, with a NULL sharpness.
        fid = self.add_file("gone.jpg")
        classify(self.cfg, self.conn, classifier=QualityClassifier(),
                 text_detector=NO_OCR)
        self.assertIsNone(self.quality(fid)["sharpness"])


class TestPetsToggle(FrameQualityCase):
    """Brief test 3: off — empty; on — filled, and with no extra CLIP call."""

    def test_pets_off_leaves_the_columns_empty(self):
        fid = self.add_file("cat.jpg")
        clf = QualityClassifier(logits={"cat.jpg": {_CAT_IDX: CONFIDENT}})
        classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR,
                 sharpness_detector=flat_sharpness(500.0))
        row = self.quality(fid)
        self.assertIsNone(row["pet"])
        self.assertIsNone(row["pet_score"])
        self.assertEqual(row["source"], junk.QUALITY_SOURCE_CLASSIC)
        # and the call carried the old prompt list, unchanged
        self.assertEqual(clf.prompt_counts()[0], len(junk._CLIP_CLASSES))

    def test_pets_on_fills_the_columns(self):
        self.features(pets=True, pet_threshold=0.5)
        fid = self.add_file("cat.jpg")
        clf = QualityClassifier(logits={"cat.jpg": {_CAT_IDX: CONFIDENT}})
        stats = classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR,
                         sharpness_detector=flat_sharpness(500.0))
        row = self.quality(fid)
        self.assertEqual(row["pet"], junk.PET_CLASS)
        self.assertGreater(row["pet_score"], 0.5)
        self.assertEqual(junk.quality_tier(row["source"]), junk.QUALITY_SOURCE_CLIP)
        self.assertEqual(stats.pets_found, 1)

    def test_the_threshold_comes_from_the_config(self):
        self.features(pets=True, pet_threshold=0.95)
        fid = self.add_file("maybe_dog.jpg")
        clf = QualityClassifier(logits={"maybe_dog.jpg": {_DOG_IDX: UNSURE}})
        classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR,
                 sharpness_detector=flat_sharpness(500.0))
        row = self.quality(fid)
        self.assertIsNone(row["pet"])          # below the configured threshold
        self.assertIsNotNone(row["pet_score"])  # but the score is kept for re-measuring

    def test_pets_add_no_clip_call_and_no_second_pass(self):
        """The heart of the feature: the same number of classifier calls, on the same paths.

        A frame is encoded once either way — what changes is the length of the prompt list
        of the ONE call the stage already made.
        """
        self.add_file("a.jpg", camera_make=None, camera_model=None)
        self.add_file("b.jpg", camera_make=None, camera_model=None)

        without = QualityClassifier()
        classify(self.cfg, self.conn, classifier=without, text_detector=NO_OCR,
                 sharpness_detector=flat_sharpness(500.0))

        self.conn.execute("DELETE FROM media_class")
        self.conn.execute("DELETE FROM frame_quality")
        self.conn.commit()
        self.features(pets=True)
        with_pets = QualityClassifier()
        classify(self.cfg, self.conn, classifier=with_pets, text_detector=NO_OCR,
                 sharpness_detector=flat_sharpness(500.0))

        self.assertEqual(len(with_pets.calls), len(without.calls))
        self.assertEqual([paths for _n, paths in with_pets.calls],
                         [paths for _n, paths in without.calls])
        # the only difference is the length of the main prompt list
        self.assertEqual(with_pets.prompt_counts()[0],
                         without.prompt_counts()[0] + len(junk._PET_CLASSES))

    def test_junk_verdicts_do_not_move_when_pets_are_on(self):
        """The same frames and the same CLIP logits, pets off then on — same verdicts.

        This is what `_group_probs` is for. The mock hands out one set of logits to both
        runs; with pets on the softmax spreads over five more prompts, and the junk
        classes have to come back out of it exactly as they went in — otherwise
        `naming.junk_threshold` would be a different threshold in the two runs.
        """
        logits = {"shot.png": {1: CONFIDENT},      # screenshot, confidently
                  "meme.jpg": {2: CONFIDENT},      # meme, confidently
                  "cam.jpg": {_CAT_IDX: CONFIDENT}}  # nothing but a cat, in the pet group

        def run(pets):
            tmp = tempfile.TemporaryDirectory()
            self.addCleanup(tmp.cleanup)
            cfg = Config(sources=[Path(tmp.name)], database=Path(tmp.name) / "t.db",
                         naming=_naming_from({}), features=FeaturesConfig(pets=pets))
            conn = connect(cfg.database)
            self.addCleanup(conn.close)
            for name in logits:
                conn.execute(
                    """INSERT INTO files (path, size, mtime, ext, media_type, width,
                           height, camera_make, camera_model, gps_lat, indexed_at)
                       VALUES (?, 1000, 0, 'jpg', 'photo', 4000, 3000, NULL, NULL, NULL,
                               '2026-01-01')""",
                    (f"/photos/{name}",))
            conn.commit()
            classify(cfg, conn, classifier=QualityClassifier(logits=logits),
                     text_detector=NO_OCR, sharpness_detector=flat_sharpness(500.0))
            return {r["path"]: (r["verdict"], r["source"], r["score"])
                    for r in conn.execute(
                        """SELECT f.path, mc.verdict, mc.source, mc.score FROM files f
                           JOIN media_class mc ON mc.file_id = f.id ORDER BY f.path""")}

        off, on = run(False), run(True)
        self.assertEqual({p: v[:2] for p, v in off.items()},
                         {p: v[:2] for p, v in on.items()})
        self.assertEqual(off["/photos/shot.png"][0], "screenshot")  # not a vacuous pass
        self.assertEqual(off["/photos/meme.jpg"][0], "meme")
        for path, (_verdict, _source, score) in off.items():
            # equal to float32 rounding, which is all a renormalization can promise
            self.assertAlmostEqual(score, on[path][2], places=5)

    def test_switching_pets_on_refreshes_the_quality_rows_only(self):
        fid = self.add_file("cat.jpg")
        clf = QualityClassifier(logits={"cat.jpg": {_CAT_IDX: CONFIDENT}})
        classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR,
                 sharpness_detector=flat_sharpness(500.0))
        self.assertIsNone(self.quality(fid)["pet"])

        self.features(pets=True, pet_threshold=0.5)
        stats = classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR,
                         sharpness_detector=flat_sharpness(500.0))
        self.assertEqual(self.quality(fid)["pet"], junk.PET_CLASS)
        # junk classification did not have to be redone for it
        self.assertEqual(stats.processed, 0)


class TestQualityIncrementality(FrameQualityCase):
    """Brief test 6: a second run does not re-measure what the first one measured."""

    def setUp(self):
        super().setUp()
        self.deep_analysis_on()  # F145: a subordinate key alone raises nothing

    def test_the_second_run_writes_nothing_new(self):
        self.add_file("IMG_0001.jpg")
        clf = QualityClassifier()
        first = classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR,
                         sharpness_detector=flat_sharpness(500.0))
        self.assertEqual(first.quality_rows, 1)
        second = classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR,
                          sharpness_detector=flat_sharpness(500.0))
        self.assertEqual(second.quality_rows, 0)


class TestRetiredAnswerColumns(FrameQualityCase):
    """F186: the three answer columns stay, and stay NULL — nobody is asked about them.

    `eyes_open`, `has_subject` and `is_accidental` are what is left of a question that is
    no longer put to a model (F122, F177 and finally F167/F179 closed the three halves of
    it). Keeping the columns and keeping them empty is the honest record of that: NULL in
    a column that is still read means "not asked", where a dropped column would mean
    "never existed". The case below is the guard on that, and it is the reason the columns
    are not quietly reused for something else.
    """

    def setUp(self):
        super().setUp()
        self.deep_analysis_on()  # F145: the master switch is on and STILL nothing is asked

    def test_a_full_run_leaves_the_three_answer_columns_null(self):
        fid = self.add_file("IMG_0001.jpg")
        self.features(pets=True, pet_threshold=0.5)
        classify(self.cfg, self.conn, classifier=QualityClassifier(),
                 text_detector=NO_OCR, sharpness_detector=flat_sharpness(100.0))
        row = self.quality(fid)
        self.assertAlmostEqual(row["sharpness"], 100.0)  # the stage did run
        self.assertIsNone(row["eyes_open"])
        self.assertIsNone(row["has_subject"])
        self.assertIsNone(row["is_accidental"])

    def test_no_model_is_built_for_them(self):
        """The frames of the uncertain band used to be the population of a model call.

        A run over exactly such a frame (sharpness inside the band, CLIP unsure) must now
        raise nothing at all — the stand-in factory below fails the case if the deep tier
        is entered for a question that no longer exists.
        """
        self.features(sharpness_band_min=30.0, sharpness_band_max=300.0)
        fid = self.add_file("blurry.jpg")

        def factory(_model):
            raise AssertionError("no model may be built for the retired quality question")

        classify(self.cfg, self.conn, classifier=QualityClassifier(),
                 text_detector=NO_OCR, sharpness_detector=flat_sharpness(100.0),
                 pet_vlm_factory=factory, junk_rescue_vlm_factory=factory)
        self.assertIsNone(self.quality(fid)["eyes_open"])

    def test_false_and_null_are_distinguishable_on_the_way_out(self):
        """The read layer's own promise, and it outlives what used to write those rows.

        A collection measured before F186 still carries `eyes_open = 0` rows, and the one
        thing the reader must never do is turn that 0 into the same value as NULL: False
        is an answer, NULL is the absence of one. Nothing in the pipeline writes a 0 there
        any more, so the row is written by hand — which is also exactly the shape of the
        rows on disk this has to keep reading.
        """
        answered = self.add_file("closed.jpg")
        unasked = self.add_file("crisp.jpg")
        classify(self.cfg, self.conn, classifier=QualityClassifier(),
                 text_detector=NO_OCR, sharpness_detector=flat_sharpness(100.0))
        self.conn.execute(
            "UPDATE frame_quality SET eyes_open = 0 WHERE file_id = ?", (answered,))
        self.conn.commit()

        rows = read_frame_quality(self.conn)
        self.assertIs(rows[answered].eyes_open, False)
        self.assertIsNone(rows[answered].has_subject)
        self.assertIsNone(rows[answered].is_accidental)
        self.assertIsNone(rows[unasked].eyes_open)
        self.assertIsNone(rows[unasked].has_subject)
        # and the same distinction survives a filtered read
        only = read_frame_quality(self.conn, [unasked])
        self.assertEqual(set(only), {unasked})
        self.assertIsNone(only[unasked].is_accidental)
        self.assertEqual(read_frame_quality(self.conn, []), {})

    def test_reading_by_id_survives_a_collection_sized_request(self):
        # SQLite has a ceiling on bound parameters and a photo library reaches it: the
        # read chunks, and asking about 5 000 ids must not raise.
        fid = self.add_file("IMG_0001.jpg")
        classify(self.cfg, self.conn, classifier=QualityClassifier(),
                 text_detector=NO_OCR, sharpness_detector=flat_sharpness(42.0))
        rows = read_frame_quality(self.conn, list(range(1, 5001)))
        self.assertEqual(set(rows), {fid})
        self.assertAlmostEqual(rows[fid].sharpness, 42.0)

    def test_heuristics_only_runs_touch_nothing(self):
        self.features(pets=True)
        fid = self.add_file("Screenshot_1.png", camera_make=None, camera_model=None)

        def factory(_model):
            raise AssertionError("no model may be built on a heuristics-only run")

        classify(self.cfg, self.conn, use_clip=False, pet_vlm_factory=factory,
                 sharpness_detector=flat_sharpness(100.0))
        self.assertIsNone(self.quality(fid))


class TestQualitySettings(unittest.TestCase):
    """The settings object the pipeline and the measurement script share."""

    def test_read_from_the_config_sections(self):
        cfg = Config(features=FeaturesConfig(pets=True, pet_threshold=0.42,
                                             sharpness_band_min=1.0,
                                             sharpness_band_max=2.0),
                     vlm=VlmConfig(exclude_classes=("document",)))
        q = junk.quality_settings(cfg)
        self.assertTrue(q.pets)
        self.assertAlmostEqual(q.pet_threshold, 0.42)
        self.assertEqual(q.sharpness_band, (1.0, 2.0))
        self.assertEqual(q.exclude_classes, frozenset({"document"}))

    def test_the_source_marker_names_the_tier_that_ran(self):
        tier = junk.quality_tier
        self.assertEqual(tier(junk._quality_source(True, False, None)), "classic")
        self.assertEqual(tier(junk._quality_source(True, True, None)), "clip")
        # F186: the pet check is the asker that puts a row in the `vlm` tier now — the
        # frame-quality question that used to do it is retired.
        self.assertEqual(tier(junk._quality_source(True, True, lambda _p: "")), "vlm")
        # a heuristics-only run has no CLIP, so pets cannot have been computed
        self.assertEqual(tier(junk._quality_source(False, True, None)), "classic")

    def test_the_marker_carries_a_prompt_fingerprint(self):
        """F120: a prompt edit has to invalidate what the prompt produced.

        The marker used to name the tier alone, so rewriting the pet group left every
        stored label looking fresh — `vlm` still equals `vlm`. That is exactly what
        happened when F120 added five anti-classes, and the alternative to a fingerprint
        is asking a person to remember to empty a table by hand.
        """
        with_pets = junk._quality_source(True, True, None)
        self.assertRegex(with_pets, r"^clip#[0-9a-f]{8}$")
        with_model = junk._quality_source(True, True, lambda _p: "")
        self.assertRegex(with_model, r"^vlm#[0-9a-f]{8}$")
        # The check the model runs has a question of its own, so its tier has its own
        # fingerprint.
        self.assertNotEqual(with_pets.split("#")[1], with_model.split("#")[1])
        # Sharpness depends on no prompt: a sharpness-only collection must NOT be
        # invalidated every time somebody edits a CLIP prompt.
        self.assertEqual(junk._quality_source(True, False, None), "classic")

    def test_editing_a_prompt_changes_the_fingerprint(self):
        """The property the whole mechanism exists for, stated against the real list."""
        before = junk.quality_prompt_fingerprint(True)
        # `_PET_CLASSES` is what clip_prompts reads; `_PET_ANTI_CLASSES` is folded into
        # it at import, so patching that one would change nothing and prove nothing.
        extra = junk._PET_CLASSES + (("statue", "a photo of a statue of an animal"),)
        with unittest.mock.patch.object(junk, "_PET_CLASSES", extra):
            after = junk.quality_prompt_fingerprint(True)
        self.assertNotEqual(before, after)
        self.assertEqual(before, junk.quality_prompt_fingerprint(True))


class TestFrameQualityWithTheOldJunkMock(FrameQualityCase):
    """The rest of the junk suite keeps working: the F113 columns ride alongside it."""

    def test_a_screenshot_is_classified_and_deliberately_not_measured(self):
        """F120 changed this case, and the change is the point of F120.

        It used to assert that a screenshot gets a sharpness too. The first live run
        showed what that costs: screenshots average a laplacian of 2854 against a
        photograph's 1253 — hard edges and text are what the measure responds to — so a
        ranking over the whole collection put screenshots on top by construction, and
        45% of the sharpest frames were not photographs. "Are the eyes open", "is there
        a pet" and "how sharp is this" are questions about a personal photograph; asked
        of a screenshot they return a number that means nothing and outvotes the ones
        that do.
        """
        fid = self.add_file("Screenshot_1.png", camera_make=None, camera_model=None)
        classify(self.cfg, self.conn, classifier=FakeClassifier({}),
                 text_detector=NO_OCR, sharpness_detector=flat_sharpness(77.0))
        self.assertEqual(self.media_class(fid)["verdict"], "screenshot")
        self.assertIsNone(self.quality(fid))

    def test_a_photograph_is_still_measured(self):
        """The other half of the same rule — the population narrowed, it did not empty."""
        fid = self.add_file("IMG_0002.jpg")
        classify(self.cfg, self.conn, classifier=QualityClassifier(),
                 text_detector=NO_OCR, sharpness_detector=flat_sharpness(77.0))
        self.assertEqual(self.media_class(fid)["verdict"], "photo")
        self.assertAlmostEqual(self.quality(fid)["sharpness"], 77.0)

    def test_a_stale_row_of_a_reclassified_frame_is_removed(self):
        """A collection measured before F120 carries rows for screenshots and documents.
        They are not left to rot: the frame is walked, found not to be a photograph, and
        its row goes — otherwise today's contamination would outlive the fix."""
        fid = self.add_file("Screenshot_2.png", camera_make=None, camera_model=None)
        self.conn.execute(
            "INSERT INTO frame_quality (file_id, sharpness, source, updated_at)"
            " VALUES (?, 4242.0, 'classic', '2026-07-31T00:00:00')", (fid,))
        self.conn.commit()
        self.assertIsNotNone(self.quality(fid))
        classify(self.cfg, self.conn, classifier=FakeClassifier({}),
                 text_detector=NO_OCR, sharpness_detector=flat_sharpness(77.0))
        self.assertIsNone(self.quality(fid))

    def test_the_foreign_key_points_at_files(self):
        fid = self.add_file("IMG_0001.jpg")
        classify(self.cfg, self.conn, classifier=QualityClassifier(),
                 text_detector=NO_OCR, sharpness_detector=flat_sharpness(1.0))
        with self.assertRaises(sqlite3.IntegrityError):
            self.conn.execute(
                "INSERT INTO frame_quality (file_id, source, updated_at) "
                "VALUES (?, 'classic', 'x')", (fid + 999,))
            self.conn.commit()


if __name__ == "__main__":
    unittest.main()
