"""F113: the frame-quality cascade — classic -> CLIP -> VLM only for the uncertain.

The properties under test are the ones the feature is about, not the plumbing:

* the migration creates `frame_quality`, raises `user_version` and runs twice safely;
* sharpness is written with every toggle off (it costs milliseconds and everybody wants it);
* `features.pets` off leaves the pet columns empty; on, they fill — and WITHOUT a second
  CLIP pass: the classifier makes exactly the calls it made before the feature existed,
  and the junk verdicts do not move under the pet prompts;
* `vlm.quality` off leaves the three model columns NULL; on, only frames of the uncertain
  band are asked about and the rest stay NULL;
* the second run asks nothing again (incrementality on `frame_quality.source`);
* an answer that does not parse leaves NULL — never False — and NULL and False stay
  distinguishable on the way back out;
* a model that fails on a frame costs that frame's answers and nothing else: the classic
  and CLIP signals are still there afterwards.

No model is loaded anywhere below: the classifier, the sharpness detector and the VLM are
all injected, exactly as the rest of the junk suite does it.
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
from sorta.db import connect
from sorta.junk import (
    QualityFlags,
    classify,
    clip_prompts,
    laplacian_variance,
    parse_quality_answer,
    pet_verdict,
    read_frame_quality,
    uncertain_band,
)
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


def flat_sharpness(value):
    """A sharpness detector that answers `value` for every frame (files are not on disk)."""
    return lambda _path: value


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

        A subordinate key (`vlm.quality`, `features.pets_verify`, `dedup.keeper_vlm`,
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
        self.assertEqual(cols, {"file_id", "sharpness", "pet", "pet_score", "pet_vlm",
                                "eyes_open", "has_subject", "is_accidental",
                                "junk_score", "source", "updated_at"})
        self.assertEqual(version, 21)

    def test_v14_db_gains_the_table_without_touching_its_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "v14.db"
            conn = connect(db)
            conn.execute(
                "INSERT INTO files (path, size, mtime, ext, media_type, indexed_at) "
                "VALUES ('/a.jpg', 1, 0.0, 'jpg', 'photo', 'x')")
            conn.execute("DROP TABLE frame_quality")
            conn.execute("PRAGMA user_version = 14")
            conn.commit()
            conn.close()

            conn = connect(db)
            tables = {r["name"] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            files = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
            conn.close()
        self.assertIn("frame_quality", tables)
        self.assertEqual(version, 21)
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
        self.assertEqual(version, 21)


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
        self.assertIsNone(detector("/nowhere/at/all.jpg"))


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


class TestQualityAnswerParsing(unittest.TestCase):
    """Brief test 7: what the model says, read leniently — and NULL when it says nothing."""

    def test_the_expected_answer(self):
        # F122: `accidental`/`deliberate` is no longer asked, so the word is no longer
        # read — 5% precision on a labelled sample, against 10% in the frames the model
        # called deliberate. The keyword may still appear; it must be ignored.
        flags = parse_quality_answer("eyes_open subject deliberate")
        self.assertEqual(flags, QualityFlags(True, True, None))

    def test_spaces_punctuation_and_case_are_not_a_format(self):
        flags = parse_quality_answer("Eyes-Open, No Subject. Accidental!")
        self.assertEqual(flags, QualityFlags(True, False, None))

    def test_prose_around_the_keywords_still_parses(self):
        flags = parse_quality_answer(
            "The photo shows a person whose eyes_closed, and it has subject.")
        self.assertEqual(flags.eyes_open, False)
        self.assertEqual(flags.has_subject, True)
        self.assertIsNone(flags.is_accidental)

    def test_no_subject_is_not_read_as_subject(self):
        self.assertEqual(parse_quality_answer("no_subject").has_subject, False)

    def test_the_retired_accidental_keyword_is_ignored(self):
        """F122: the question is gone, so a model that still volunteers the word gets no
        column for it. NULL is the honest value — nobody asked."""
        self.assertIsNone(parse_quality_answer("not accidental").is_accidental)
        self.assertIsNone(parse_quality_answer("accidental").is_accidental)

    def test_an_unparsable_answer_is_all_none(self):
        for answer in ("", "I cannot help with that", "42", "да"):
            with self.subTest(answer=answer):
                flags = parse_quality_answer(answer)
                self.assertEqual(flags, QualityFlags())
                self.assertFalse(flags.known)

    def test_a_partial_answer_keeps_the_rest_none(self):
        flags = parse_quality_answer("subject")
        self.assertTrue(flags.known)
        self.assertIsNone(flags.eyes_open)
        self.assertIsNone(flags.is_accidental)


class TestUncertainBand(unittest.TestCase):
    """Which frames the model is asked about at all (brief part 3, item 2)."""

    def q(self, **kwargs):
        base = dict(pets=False, pet_threshold=0.6, sharpness_max_edge=512,
                    sharpness_band=(30.0, 300.0), subject_score_min=0.9,
                    vlm_quality=True, vlm_scope="groups")
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
    """Brief test 6: a second run neither re-measures nor re-asks."""

    def setUp(self):
        super().setUp()
        self.deep_analysis_on()  # F145: `vlm.quality` alone raises nothing

    def test_the_second_run_writes_nothing_new(self):
        self.add_file("IMG_0001.jpg")
        clf = QualityClassifier()
        first = classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR,
                         sharpness_detector=flat_sharpness(500.0))
        self.assertEqual(first.quality_rows, 1)
        second = classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR,
                          sharpness_detector=flat_sharpness(500.0))
        self.assertEqual(second.quality_rows, 0)

    def test_the_second_run_does_not_re_ask_the_model(self):
        self.vlm(quality=True, quality_scope="all")
        self.add_file("IMG_0002.jpg")
        asked: list[str] = []

        def ask(path):
            asked.append(path)
            return "eyes_open subject deliberate"

        clf = QualityClassifier()
        classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR,
                 sharpness_detector=flat_sharpness(100.0), quality_vlm=ask)
        self.assertEqual(len(asked), 1)
        classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR,
                 sharpness_detector=flat_sharpness(100.0), quality_vlm=ask)
        self.assertEqual(len(asked), 1)


class TestQualityVlm(FrameQualityCase):
    """Brief tests 4, 5, 8, 9: the model tier — its toggle, its population, its failures."""

    def setUp(self):
        super().setUp()
        self.deep_analysis_on()  # F145: `vlm.quality` alone raises nothing

    def test_disabled_leaves_the_model_columns_null(self):
        fid = self.add_file("IMG_0001.jpg")

        def factory(_model):
            raise AssertionError("the quality VLM must not be built when vlm.quality is off")

        classify(self.cfg, self.conn, classifier=QualityClassifier(),
                 text_detector=NO_OCR, sharpness_detector=flat_sharpness(100.0),
                 quality_vlm_factory=factory)
        row = self.quality(fid)
        self.assertIsNone(row["eyes_open"])
        self.assertIsNone(row["has_subject"])
        self.assertIsNone(row["is_accidental"])

    def test_only_the_uncertain_band_is_asked_about(self):
        self.vlm(quality=True, quality_scope="all")
        self.features(sharpness_band_min=30.0, sharpness_band_max=300.0)
        uncertain = self.add_file("blurry.jpg")
        certain = self.add_file("sharp.jpg")
        sharpness = {"/photos/blurry.jpg": 100.0, "/photos/sharp.jpg": 5000.0}
        # CLIP is confident about BOTH frames, so sharpness is the only thing that can put
        # one of them in the band — otherwise the test would pass for the wrong reason.
        clf = QualityClassifier(logits={"blurry.jpg": {_PHOTO_IDX: CONFIDENT},
                                        "sharp.jpg": {_PHOTO_IDX: CONFIDENT}})
        asked: list[str] = []

        def ask(path):
            asked.append(path)
            return "eyes_open subject deliberate"

        classify(self.cfg, self.conn, classifier=clf,
                 text_detector=NO_OCR, sharpness_detector=sharpness.get,
                 quality_vlm=ask)
        self.assertEqual(asked, ["/photos/blurry.jpg"])
        self.assertEqual(self.quality(uncertain)["has_subject"], 1)
        self.assertIsNone(self.quality(certain)["has_subject"])

    def test_the_scope_narrows_the_population_to_phash_groups(self):
        self.vlm(quality=True)  # quality_scope='groups' — the default
        grouped_a = self.add_file("g1.jpg", phash="ffffffffffffffff")
        grouped_b = self.add_file("g2.jpg", phash="ffffffffffffffff")
        lonely = self.add_file("solo.jpg", phash="0000000000000000")
        asked: list[str] = []

        def ask(path):
            asked.append(path)
            return "subject"

        classify(self.cfg, self.conn, classifier=QualityClassifier(),
                 text_detector=NO_OCR, sharpness_detector=flat_sharpness(100.0),
                 quality_vlm=ask)
        self.assertEqual(sorted(asked), ["/photos/g1.jpg", "/photos/g2.jpg"])
        self.assertEqual(self.quality(grouped_a)["has_subject"], 1)
        self.assertEqual(self.quality(grouped_b)["has_subject"], 1)
        self.assertIsNone(self.quality(lonely)["has_subject"])

    def test_the_events_scope_asks_about_event_frames(self):
        self.vlm(quality=True, quality_scope="events")
        in_event = self.add_file("e1.jpg")
        outside = self.add_file("e2.jpg")
        self.conn.execute(
            "INSERT INTO events (id, started_at, ended_at, name) "
            "VALUES (1, '2026-01-01', '2026-01-02', 'x')")
        self.conn.execute(
            "INSERT INTO event_files (event_id, file_id) VALUES (1, ?)", (in_event,))
        self.conn.commit()
        asked: list[str] = []

        classify(self.cfg, self.conn, classifier=QualityClassifier(),
                 text_detector=NO_OCR, sharpness_detector=flat_sharpness(100.0),
                 quality_vlm=lambda p: asked.append(p) or "subject")
        self.assertEqual(asked, ["/photos/e1.jpg"])
        self.assertIsNone(self.quality(outside)["has_subject"])

    def test_an_unparsable_answer_leaves_null_not_false(self):
        self.vlm(quality=True, quality_scope="all")
        fid = self.add_file("odd.jpg")
        classify(self.cfg, self.conn, classifier=QualityClassifier(),
                 text_detector=NO_OCR, sharpness_detector=flat_sharpness(100.0),
                 quality_vlm=lambda _p: "I'm not sure what this is")
        row = self.quality(fid)
        self.assertIsNone(row["eyes_open"])
        self.assertIsNone(row["has_subject"])
        self.assertIsNone(row["is_accidental"])

    def test_false_and_null_are_distinguishable_on_the_way_out(self):
        self.vlm(quality=True, quality_scope="all")
        answered = self.add_file("closed.jpg")
        unasked = self.add_file("crisp.jpg")
        sharpness = {"/photos/closed.jpg": 100.0, "/photos/crisp.jpg": 9000.0}
        clf = QualityClassifier(logits={"closed.jpg": {_PHOTO_IDX: CONFIDENT},
                                        "crisp.jpg": {_PHOTO_IDX: CONFIDENT}})
        classify(self.cfg, self.conn, classifier=clf,
                 text_detector=NO_OCR, sharpness_detector=sharpness.get,
                 quality_vlm=lambda _p: "eyes_closed no_subject accidental")
        rows = read_frame_quality(self.conn)
        self.assertIs(rows[answered].eyes_open, False)
        self.assertIs(rows[answered].has_subject, False)
        # F122: retired question, and the point of the case is that False and None stay
        # distinguishable — `is_accidental` now demonstrates the None half of it.
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

    def test_a_model_failure_on_one_frame_keeps_the_cheap_signals(self):
        self.vlm(quality=True, quality_scope="all")
        self.features(pets=True, pet_threshold=0.5)
        boom = self.add_file("boom.jpg")
        fine = self.add_file("fine.jpg")

        def ask(path):
            if path.endswith("boom.jpg"):
                raise RuntimeError("CUDA error: device-side assert triggered")
            return "eyes_open subject deliberate"

        clf = QualityClassifier(logits={"boom.jpg": {_CAT_IDX: CONFIDENT}})
        stats = classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR,
                         sharpness_detector=flat_sharpness(100.0), quality_vlm=ask)
        failed = self.quality(boom)
        self.assertAlmostEqual(failed["sharpness"], 100.0)   # the classic tier survived
        self.assertEqual(failed["pet"], junk.PET_CLASS)      # the CLIP tier survived
        self.assertIsNone(failed["eyes_open"])               # only the answers are missing
        self.assertEqual(self.quality(fine)["eyes_open"], 1)  # the neighbour is unaffected
        self.assertEqual(stats.quality_candidates, 2)
        self.assertEqual(stats.quality_answered, 1)

    def test_a_factory_that_raises_falls_back_to_the_cheap_tiers(self):
        self.vlm(quality=True, quality_scope="all")
        fid = self.add_file("IMG_0003.jpg")

        def broken_factory(_model):
            raise RuntimeError("transformers not installed")

        classify(self.cfg, self.conn, classifier=QualityClassifier(),
                 text_detector=NO_OCR, sharpness_detector=flat_sharpness(100.0),
                 quality_vlm_factory=broken_factory)
        row = self.quality(fid)
        self.assertAlmostEqual(row["sharpness"], 100.0)
        self.assertIsNone(row["eyes_open"])
        # the row is marked by the tier that actually ran, so a later run with a working
        # model picks it up instead of considering it done
        self.assertEqual(row["source"], junk.QUALITY_SOURCE_CLASSIC)

    def test_heuristics_only_runs_touch_nothing(self):
        self.features(pets=True)
        self.vlm(quality=True, quality_scope="all")
        fid = self.add_file("Screenshot_1.png", camera_make=None, camera_model=None)

        def factory(_model):
            raise AssertionError("no model may be built on a heuristics-only run")

        classify(self.cfg, self.conn, use_clip=False, quality_vlm_factory=factory,
                 sharpness_detector=flat_sharpness(100.0))
        self.assertIsNone(self.quality(fid))


class TestVlmQualityAsker(unittest.TestCase):
    """The real asker, over a fake runtime: the prompt goes out, the frame is decoded."""

    def test_the_frame_is_decoded_and_the_prompt_is_asked(self):
        seen: list[tuple[int, str, int]] = []

        def describe(frames, prompt, max_new_tokens):
            seen.append((len(frames), prompt, max_new_tokens))
            return "eyes_open subject deliberate"

        ask = junk.vlm_quality_asker(describe, max_edge=128)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.jpg"
            Image.new("RGB", (256, 192), (30, 60, 90)).save(path, "JPEG")
            answer = ask(str(path))
        self.assertEqual(parse_quality_answer(answer), QualityFlags(True, True, None))
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0][0], 1)
        self.assertIn("eyes_open", seen[0][1])

    def test_a_missing_file_is_an_empty_answer_not_a_crash(self):
        def describe(_frames, _prompt, _tokens):
            raise AssertionError("a vanished frame must never reach the model")

        self.assertEqual(junk.vlm_quality_asker(describe, 128)("/nowhere/x.jpg"), "")


class TestQualitySettings(unittest.TestCase):
    """The settings object the pipeline and the measurement script share."""

    def test_read_from_the_config_sections(self):
        cfg = Config(features=FeaturesConfig(pets=True, pet_threshold=0.42,
                                             sharpness_band_min=1.0,
                                             sharpness_band_max=2.0),
                     vlm=VlmConfig(quality=True, quality_scope="events"))
        q = junk.quality_settings(cfg)
        self.assertTrue(q.pets)
        self.assertAlmostEqual(q.pet_threshold, 0.42)
        self.assertEqual(q.sharpness_band, (1.0, 2.0))
        self.assertTrue(q.vlm_quality)
        self.assertEqual(q.vlm_scope, "events")

    def test_the_source_marker_names_the_tier_that_ran(self):
        tier = junk.quality_tier
        self.assertEqual(tier(junk._quality_source(True, False, None)), "classic")
        self.assertEqual(tier(junk._quality_source(True, True, None)), "clip")
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
        # The model is asked a question of its own, so its tier has its own fingerprint.
        self.assertNotEqual(with_pets.split("#")[1], with_model.split("#")[1])
        # Sharpness depends on no prompt: a sharpness-only collection must NOT be
        # invalidated every time somebody edits a CLIP prompt.
        self.assertEqual(junk._quality_source(True, False, None), "classic")

    def test_editing_a_prompt_changes_the_fingerprint(self):
        """The property the whole mechanism exists for, stated against the real list."""
        before = junk.quality_prompt_fingerprint(True, with_vlm=False)
        # `_PET_CLASSES` is what clip_prompts reads; `_PET_ANTI_CLASSES` is folded into
        # it at import, so patching that one would change nothing and prove nothing.
        extra = junk._PET_CLASSES + (("statue", "a photo of a statue of an animal"),)
        with unittest.mock.patch.object(junk, "_PET_CLASSES", extra):
            after = junk.quality_prompt_fingerprint(True, with_vlm=False)
        self.assertNotEqual(before, after)
        self.assertEqual(before, junk.quality_prompt_fingerprint(True, with_vlm=False))


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
