"""F68: incrementality runs on `media_class.tier`, not on `source`.

`source` answers "what decided the verdict" (heuristic | clip | ocr | vlm — user
facing, read by sorter.py), `tier` answers "which tier processed the row"
(heuristic | clip | vlm — the incrementality marker). Conflating them made two
whole classes of rows fail the "already processed" check and be reclassified on
every run:

(a) rows the OCR gate/rescue rewrote to source='ocr' inside the fast pass, while
    the active marker was 'clip';
(b) with the deep tier on, every row the VLM gate deliberately skipped (a clear
    personal photo keeps source='clip') — i.e. almost the whole collection.

The tests below pin both regressions, both directions of a tier switch, the
untouched semantics of `source`, and the v10 -> v11 migration backfill.
"""
import tempfile
import unittest
from pathlib import Path

from sorta.config import Config, _naming_from
from sorta.db import connect
from sorta.junk import classify
from tests.test_junk import NO_OCR, FakeClassifier, _RECEIPT_IDX


class TierTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = Config(sources=[Path(self.tmp.name)],
                          database=Path(self.tmp.name) / "test.db",
                          naming=_naming_from({}))
        self.conn = connect(self.cfg.database)

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def enable_vlm(self, enabled=True):
        object.__setattr__(self.cfg.naming, "vlm_enabled", enabled)

    def add_file(self, name, camera_make="Canon", camera_model="EOS", has_face=False):
        cur = self.conn.execute(
            """INSERT INTO files (path, size, mtime, ext, media_type, width, height,
                   camera_make, camera_model, gps_lat, indexed_at)
               VALUES (?, 1000, 0, 'jpg', 'photo', 4000, 3000, ?, ?, NULL, '2026-01-01')""",
            (f"/photos/{name}", camera_make, camera_model))
        fid = cur.lastrowid
        if has_face:
            self.conn.execute(
                "INSERT INTO faces (file_id, bbox, embedding) VALUES (?, '[1,2,3,4]', ?)",
                (fid, b"\x00" * 4))
        self.conn.commit()
        return fid

    def media_class(self, fid):
        return self.conn.execute(
            "SELECT verdict, source, score, tier FROM media_class WHERE file_id = ?",
            (fid,)).fetchone()

    def candidate_clf(self, name, main=(0, 0.99)):
        # doc_score 0.5: >= text_rescue_docscore_min (0.3) -> a VLM candidate, but
        # < document_threshold (0.9) -> the fast verdict stays 'photo'.
        return FakeClassifier({name: main}, doc_scores={name: (_RECEIPT_IDX, 0.5)})

    def assert_no_null_tier(self):
        (n,) = self.conn.execute(
            "SELECT COUNT(*) FROM media_class WHERE tier IS NULL").fetchone()
        self.assertEqual(n, 0)


class TestOcrRowsAreIncremental(TierTestBase):
    """Bug (a): the OCR gate/rescue rewrites source to 'ocr' inside the fast pass —
    under the old source-based marker those rows never matched 'clip' and were
    reclassified (CLIP + easyocr) on every single run. 597 rows on the production DB."""

    def test_ocr_fp_gate_row_not_reprocessed(self):
        # FP gate: CLIP is sure it is a document, but there is almost no text -> photo,
        # source='ocr'.
        fid = self.add_file("beach.jpg")
        clf = FakeClassifier({}, doc_scores={"beach.jpg": (_RECEIPT_IDX, 0.95)})
        classify(self.cfg, self.conn, classifier=clf, text_detector=lambda p, w, h: 0.01)
        row = self.media_class(fid)
        self.assertEqual((row["verdict"], row["source"]), ("photo", "ocr"))
        self.assertEqual(row["tier"], "clip")

        seen_before = len(clf.seen_paths)
        ocr_calls = []

        def counting_detector(path, _width, _height):
            ocr_calls.append(path)
            return 0.01

        stats2 = classify(self.cfg, self.conn, classifier=clf,
                          text_detector=counting_detector)
        self.assertEqual(stats2.processed, 0)
        self.assertEqual(stats2.skipped_incremental, 1)
        self.assertEqual(len(clf.seen_paths), seen_before)  # CLIP not re-run
        self.assertEqual(ocr_calls, [])                      # easyocr not re-run
        self.assertEqual(self.media_class(fid)["source"], "ocr")

    def test_ocr_fn_rescue_row_not_reprocessed(self):
        # FN rescue: dense text over the frame -> document, source='ocr'.
        fid = self.add_file("medform.jpg")
        clf = FakeClassifier({}, doc_scores={"medform.jpg": (_RECEIPT_IDX, 0.5)})
        classify(self.cfg, self.conn, classifier=clf, text_detector=lambda p, w, h: 0.5)
        row = self.media_class(fid)
        self.assertEqual((row["verdict"], row["source"], row["tier"]),
                         ("document", "ocr", "clip"))

        stats2 = classify(self.cfg, self.conn, classifier=clf,
                          text_detector=lambda p, w, h: 0.5)
        self.assertEqual(stats2.processed, 0)
        self.assertEqual(self.media_class(fid)["verdict"], "document")

    def test_mixed_clip_and_ocr_rows_all_skipped_on_second_run(self):
        plain = self.add_file("plain.jpg")     # stays source='clip'
        gated = self.add_file("beach2.jpg")    # rewritten to source='ocr'
        clf = FakeClassifier({}, doc_scores={"beach2.jpg": (_RECEIPT_IDX, 0.95)})
        classify(self.cfg, self.conn, classifier=clf, text_detector=lambda p, w, h: 0.01)
        self.assertEqual(self.media_class(plain)["source"], "clip")
        self.assertEqual(self.media_class(gated)["source"], "ocr")

        stats2 = classify(self.cfg, self.conn, classifier=clf,
                          text_detector=lambda p, w, h: 0.01)
        self.assertEqual(stats2.total, 2)
        self.assertEqual(stats2.processed, 0)
        self.assertEqual(stats2.skipped_incremental, 2)


class TestVlmRowsAreIncremental(TierTestBase):
    """Bug (b): with the deep tier on, the fast pass writes source='clip'/'ocr' to
    everything the VLM gate did not select — under the old marker ('vlm') those rows
    were reprocessed on every run, i.e. the whole collection. Semantically such a file
    IS handled by the vlm tier: the gate looked at it and decided not to call the model."""

    def test_non_candidate_row_not_reprocessed(self):
        self.enable_vlm()
        fid = self.add_file("beach.jpg", camera_make=None, camera_model=None)
        clf = FakeClassifier({"beach.jpg": (0, 0.99)})  # doc/prod low -> not a candidate

        def vlm(_path):
            raise AssertionError("the VLM must not be called for a clean photo")

        stats = classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR,
                         vlm_classifier=vlm)
        self.assertEqual(stats.processed, 1)
        row = self.media_class(fid)
        # source stays 'clip' (the fast pass decided), but the row was fully handled
        # by the vlm tier — that is exactly the distinction F68 introduces.
        self.assertEqual(row["source"], "clip")
        self.assertEqual(row["tier"], "vlm")

        stats2 = classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR,
                          vlm_classifier=vlm)
        self.assertEqual(stats2.processed, 0)
        self.assertEqual(stats2.skipped_incremental, 1)

    def test_whole_collection_skipped_on_second_deep_run(self):
        # a mix of everything the deep run leaves behind: candidate (source='vlm'),
        # non-candidate (source='clip'), OCR-rewritten (source='ocr'), screenshot
        # (source='clip'). The OCR row keeps its 'ocr' source only because the VLM
        # errored on it (#31) — otherwise every ocr row is a candidate too, since the
        # OCR gate and the VLM gate share the same doc_score >= 0.3 zone.
        self.enable_vlm()
        cand = self.add_file("scan.jpg", camera_make=None, camera_model=None)
        clean = self.add_file("beach.jpg", camera_make=None, camera_model=None)
        gated = self.add_file("street.jpg")
        shot = self.add_file("Screenshot_1.png", camera_make=None, camera_model=None)
        clf = FakeClassifier(
            {"scan.jpg": (0, 0.99), "beach.jpg": (0, 0.99), "street.jpg": (0, 0.99)},
            doc_scores={"scan.jpg": (_RECEIPT_IDX, 0.5),
                        "street.jpg": (_RECEIPT_IDX, 0.95)},
        )
        calls = []

        def vlm(path):
            calls.append(path)
            if path == "/photos/street.jpg":
                raise RuntimeError("CUDA error: device-side assert triggered")
            return "document"

        stats = classify(self.cfg, self.conn, classifier=clf,
                         text_detector=lambda p, w, h: 0.01, vlm_classifier=vlm)
        self.assertEqual(stats.processed, 4)
        self.assertEqual(
            [self.media_class(f)["source"] for f in (cand, clean, gated, shot)],
            ["vlm", "clip", "ocr", "clip"])
        self.assertEqual(
            [self.media_class(f)["tier"] for f in (cand, clean, gated, shot)],
            ["vlm"] * 4)
        self.assert_no_null_tier()

        vlm_calls_after_first = len(calls)
        stats2 = classify(self.cfg, self.conn, classifier=clf,
                          text_detector=lambda p, w, h: 0.01, vlm_classifier=vlm)
        self.assertEqual(stats2.total, 4)
        self.assertEqual(stats2.processed, 0)
        self.assertEqual(stats2.skipped_incremental, 4)
        self.assertEqual(len(calls), vlm_calls_after_first)


class TestTierSwitching(TierTestBase):
    """Any change of the active tier — upgrade or downgrade — reprocesses the rows
    of the other tier (the marker is compared for inequality, not ordered)."""

    def test_clip_to_vlm_reprocesses(self):
        fid = self.add_file("IMG_0400.jpg", camera_make=None, camera_model=None)
        clf_fast = FakeClassifier({"IMG_0400.jpg": (0, 0.99)})
        classify(self.cfg, self.conn, classifier=clf_fast, text_detector=NO_OCR)
        self.assertEqual(self.media_class(fid)["tier"], "clip")

        self.enable_vlm()
        clf = self.candidate_clf("IMG_0400.jpg")
        stats2 = classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR,
                          vlm_classifier=lambda path: "document")
        self.assertEqual(stats2.processed, 1)
        self.assertEqual(stats2.skipped_incremental, 0)
        row = self.media_class(fid)
        self.assertEqual((row["verdict"], row["source"], row["tier"]),
                         ("document", "vlm", "vlm"))

    def test_vlm_to_clip_reprocesses(self):
        self.enable_vlm()
        fid = self.add_file("IMG_0500.jpg", camera_make=None, camera_model=None)
        clf = self.candidate_clf("IMG_0500.jpg")
        classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR,
                 vlm_classifier=lambda path: "product")
        self.assertEqual(self.media_class(fid)["tier"], "vlm")

        self.enable_vlm(False)
        clf_fast = FakeClassifier({"IMG_0500.jpg": (0, 0.99)})
        stats2 = classify(self.cfg, self.conn, classifier=clf_fast, text_detector=NO_OCR)
        self.assertEqual(stats2.processed, 1)
        row = self.media_class(fid)
        self.assertEqual((row["source"], row["tier"]), ("clip", "clip"))

    def test_vlm_to_clip_reprocesses_non_candidates_too(self):
        # a downgrade must also pick up rows the VLM gate skipped: their source is
        # 'clip' already, only the tier tells them apart.
        self.enable_vlm()
        fid = self.add_file("beach.jpg", camera_make=None, camera_model=None)
        clf = FakeClassifier({"beach.jpg": (0, 0.99)})
        classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR,
                 vlm_classifier=lambda path: "document")
        self.assertEqual((self.media_class(fid)["source"], self.media_class(fid)["tier"]),
                         ("clip", "vlm"))

        self.enable_vlm(False)
        stats2 = classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR)
        self.assertEqual(stats2.processed, 1)
        self.assertEqual(self.media_class(fid)["tier"], "clip")

    def test_clip_to_heuristic_reprocesses(self):
        fid = self.add_file("IMG_0600.jpg", camera_make=None, camera_model=None)
        clf = FakeClassifier({"IMG_0600.jpg": (1, 0.9)})  # index 1 = screenshot
        classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR)
        self.assertEqual(self.media_class(fid)["tier"], "clip")

        stats2 = classify(self.cfg, self.conn, use_clip=False)
        self.assertEqual(stats2.processed, 1)
        row = self.media_class(fid)
        self.assertEqual((row["verdict"], row["source"], row["tier"]),
                         ("photo", "heuristic", "heuristic"))

    def test_heuristic_run_is_incremental_on_repeat(self):
        # use_clip=False used to have no marker at all (the branch returned early
        # before active_source was compared) — a heuristics-only run redid everything.
        self.add_file("Screenshot_1.png", camera_make=None, camera_model=None)
        self.add_file("IMG_0700.jpg")
        stats = classify(self.cfg, self.conn, use_clip=False)
        self.assertEqual(stats.processed, 2)
        self.assertEqual(stats.skipped_incremental, 0)

        stats2 = classify(self.cfg, self.conn, use_clip=False)
        self.assertEqual(stats2.processed, 0)
        self.assertEqual(stats2.skipped_incremental, 2)

    def test_heuristic_to_clip_reprocesses(self):
        fid = self.add_file("IMG_0800.jpg", camera_make=None, camera_model=None)
        classify(self.cfg, self.conn, use_clip=False)
        self.assertEqual(self.media_class(fid)["tier"], "heuristic")

        clf = FakeClassifier({"IMG_0800.jpg": (2, 0.9)})  # index 2 = meme
        stats2 = classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR)
        self.assertEqual(stats2.processed, 1)
        row = self.media_class(fid)
        self.assertEqual((row["verdict"], row["source"], row["tier"]),
                         ("meme", "clip", "clip"))


class TestSourceSemanticsUnchanged(TierTestBase):
    """`source` keeps its meaning and its values — sorter.py reads it
    (`mc.source AS junk_source`). F68 adds a column, it does not repurpose one."""

    def test_all_four_source_values_still_produced(self):
        # heuristic: a heuristics-only run
        heur = self.add_file("Screenshot_h.png", camera_make=None, camera_model=None)
        classify(self.cfg, self.conn, use_clip=False)
        self.assertEqual(self.media_class(heur)["source"], "heuristic")

        # clip: the fast pass decided
        clip = self.add_file("odd_name.jpg", camera_make=None, camera_model=None)
        # ocr: the FP gate rewrote the document verdict
        ocr = self.add_file("beach.jpg")
        clf = FakeClassifier({"odd_name.jpg": (1, 0.9)},
                             doc_scores={"beach.jpg": (_RECEIPT_IDX, 0.95)})
        classify(self.cfg, self.conn, classifier=clf,
                 text_detector=lambda p, w, h: 0.01)
        self.assertEqual(self.media_class(clip)["source"], "clip")
        self.assertEqual(self.media_class(ocr)["source"], "ocr")

        # vlm: the deep tier decided
        self.enable_vlm()
        vlm = self.add_file("scan.jpg", camera_make=None, camera_model=None)
        clf2 = self.candidate_clf("scan.jpg")
        classify(self.cfg, self.conn, classifier=clf2, text_detector=NO_OCR,
                 vlm_classifier=lambda path: "document")
        self.assertEqual(self.media_class(vlm)["source"], "vlm")
        self.assert_no_null_tier()

    def test_score_still_written_by_source_semantics(self):
        # the OCR branch keeps writing text_frac as the score (source='ocr') —
        # untouched by the tier change.
        fid = self.add_file("medform.jpg")
        clf = FakeClassifier({}, doc_scores={"medform.jpg": (_RECEIPT_IDX, 0.5)})
        classify(self.cfg, self.conn, classifier=clf, text_detector=lambda p, w, h: 0.5)
        row = self.media_class(fid)
        self.assertEqual(row["source"], "ocr")
        self.assertAlmostEqual(row["score"], 0.5, places=5)


class TestTierAlwaysWritten(TierTestBase):
    def test_no_null_tier_after_heuristic_run(self):
        self.add_file("Screenshot_1.png", camera_make=None, camera_model=None)
        self.add_file("IMG_0001.jpg")
        classify(self.cfg, self.conn, use_clip=False)
        self.assert_no_null_tier()

    def test_no_null_tier_after_fast_run(self):
        self.add_file("odd.jpg", camera_make=None, camera_model=None)
        self.add_file("portrait.jpg", has_face=True)
        self.add_file("beach.jpg")
        clf = FakeClassifier({"odd.jpg": (1, 0.9)},
                             doc_scores={"beach.jpg": (_RECEIPT_IDX, 0.95)})
        classify(self.cfg, self.conn, classifier=clf, text_detector=lambda p, w, h: 0.01)
        self.assert_no_null_tier()

    def test_no_null_tier_after_deep_run(self):
        self.enable_vlm()
        self.add_file("scan.jpg", camera_make=None, camera_model=None)
        self.add_file("beach.jpg", camera_make=None, camera_model=None)
        clf = self.candidate_clf("scan.jpg")
        classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR,
                 vlm_classifier=lambda path: "document")
        self.assert_no_null_tier()

    def test_tier_written_when_vlm_factory_fails(self):
        # graceful fallback to the fast tier: active_tier must be 'clip', not 'vlm' —
        # otherwise the next fast run would skip these rows as "done by vlm".
        self.enable_vlm()
        fid = self.add_file("IMG_0200.jpg", camera_make=None, camera_model=None)

        def broken_factory(model_name):
            raise RuntimeError("no CUDA / transformers not installed")

        clf = FakeClassifier({"IMG_0200.jpg": (0, 0.99)})
        classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR,
                 vlm_classifier_factory=broken_factory)
        self.assertEqual(self.media_class(fid)["tier"], "clip")


class TestSkippedIncrementalStat(TierTestBase):
    def test_counts_rows_already_done_by_the_active_tier(self):
        for i in range(3):
            self.add_file(f"IMG_{i}.jpg")
        clf = FakeClassifier({})
        stats = classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR)
        self.assertEqual((stats.total, stats.processed, stats.skipped_incremental),
                         (3, 3, 0))

        self.add_file("IMG_new.jpg")  # one fresh file joins the already-done three
        stats2 = classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR)
        self.assertEqual((stats2.total, stats2.processed, stats2.skipped_incremental),
                         (4, 1, 3))

        stats3 = classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR)
        self.assertEqual((stats3.total, stats3.processed, stats3.skipped_incremental),
                         (4, 0, 4))

    def test_zero_on_an_empty_index(self):
        stats = classify(self.cfg, self.conn, use_clip=False)
        self.assertEqual((stats.total, stats.processed, stats.skipped_incremental),
                         (0, 0, 0))

    def test_processed_plus_skipped_equals_total(self):
        self.add_file("a.jpg")
        classify(self.cfg, self.conn, use_clip=False)
        self.add_file("b.jpg")
        stats = classify(self.cfg, self.conn, use_clip=False)
        self.assertEqual(stats.processed + stats.skipped_incremental, stats.total)


class TestV10MigrationBackfill(unittest.TestCase):
    """v11 migration (already shipped by db/): an existing DB gets `tier` backfilled
    from `source` so the upgrade does NOT reclassify the whole collection. 'ocr' is a
    verdict of the fast (clip) tier, not a tier of its own — it must map to 'clip'."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "v10.db"
        self.cfg = Config(sources=[Path(self.tmp.name)], database=self.db,
                          naming=_naming_from({}))
        # Registered as a cleanup rather than in tearDown: cleanups run LIFO and
        # AFTER tearDown, so a conn.close() the tests register later must fire
        # before the directory goes away — on Windows an open sqlite handle makes
        # the rmtree fail with PermissionError.
        self.addCleanup(self.tmp.cleanup)

    def _build_v10_db(self, rows):
        """A DB as v10 left it: media_class without the `tier` column, user_version=10.

        Built from the current schema and then rolled back to the pre-v11 shape — that
        keeps `files` in sync with what classify() selects, while media_class is
        exactly what the migration will find.
        """
        conn = connect(self.db)
        ids = {}
        for name, _source in rows:
            cur = conn.execute(
                """INSERT INTO files (path, size, mtime, ext, media_type, width, height,
                       camera_make, camera_model, gps_lat, indexed_at)
                   VALUES (?, 1000, 0, 'jpg', 'photo', 4000, 3000, NULL, NULL, NULL,
                           '2026-01-01')""",
                (f"/photos/{name}",))
            ids[name] = cur.lastrowid
        conn.execute("DROP TABLE media_class")
        conn.execute(
            """CREATE TABLE media_class (
                   file_id INTEGER PRIMARY KEY REFERENCES files(id),
                   verdict TEXT NOT NULL, source TEXT NOT NULL, score REAL,
                   updated_at TEXT NOT NULL)""")
        for name, source in rows:
            conn.execute(
                """INSERT INTO media_class (file_id, verdict, source, score, updated_at)
                   VALUES (?, 'photo', ?, NULL, '2026-01-01')""",
                (ids[name], source))
        conn.execute("PRAGMA user_version = 10")
        conn.commit()
        conn.close()
        return ids

    def test_backfill_maps_source_to_tier(self):
        rows = [("a_clip.jpg", "clip"), ("b_ocr.jpg", "ocr"),
                ("c_vlm.jpg", "vlm"), ("d_heur.jpg", "heuristic")]
        ids = self._build_v10_db(rows)
        conn = connect(self.db)
        self.addCleanup(conn.close)
        (version,) = conn.execute("PRAGMA user_version").fetchone()
        self.assertGreaterEqual(version, 11)
        tiers = {
            name: conn.execute("SELECT tier FROM media_class WHERE file_id = ?",
                               (ids[name],)).fetchone()["tier"]
            for name, _source in rows
        }
        self.assertEqual(tiers, {"a_clip.jpg": "clip", "b_ocr.jpg": "clip",
                                 "c_vlm.jpg": "vlm", "d_heur.jpg": "heuristic"})

    def test_upgraded_clip_and_ocr_rows_are_not_reclassified(self):
        rows = [("a_clip.jpg", "clip"), ("b_ocr.jpg", "ocr")]
        self._build_v10_db(rows)
        conn = connect(self.db)
        self.addCleanup(conn.close)
        clf = FakeClassifier({})
        stats = classify(self.cfg, conn, classifier=clf, text_detector=NO_OCR)
        self.assertEqual(stats.total, 2)
        self.assertEqual(stats.processed, 0)
        self.assertEqual(stats.skipped_incremental, 2)
        self.assertEqual(clf.seen_paths, [])

    def test_upgraded_vlm_row_is_reclassified_by_a_clip_run(self):
        # a downgrade after the upgrade: the row is marked tier='vlm', the active tier
        # is 'clip' -> it must be redone (and only it).
        rows = [("a_clip.jpg", "clip"), ("c_vlm.jpg", "vlm")]
        self._build_v10_db(rows)
        conn = connect(self.db)
        self.addCleanup(conn.close)
        clf = FakeClassifier({})
        stats = classify(self.cfg, conn, classifier=clf, text_detector=NO_OCR)
        self.assertEqual(stats.processed, 1)
        self.assertEqual(stats.skipped_incremental, 1)
        self.assertEqual(set(clf.seen_paths), {"/photos/c_vlm.jpg"})

    def test_fresh_db_has_the_tier_column(self):
        conn = connect(self.db)
        self.addCleanup(conn.close)
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(media_class)")}
        self.assertIn("tier", cols)


class TestTierColumnIsIndependentOfSource(TierTestBase):
    def test_upsert_updates_tier_on_conflict(self):
        # the row already exists (written by an earlier tier) -> ON CONFLICT must
        # refresh `tier`, not only verdict/source/score.
        fid = self.add_file("IMG_0900.jpg", camera_make=None, camera_model=None)
        classify(self.cfg, self.conn, use_clip=False)
        self.assertEqual(self.media_class(fid)["tier"], "heuristic")
        clf = FakeClassifier({"IMG_0900.jpg": (0, 0.99)})
        classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR)
        self.assertEqual(self.media_class(fid)["tier"], "clip")

    def test_rows_without_media_class_are_always_todo(self):
        # a LEFT JOIN gives tier=NULL for a fresh file — NULL != any tier.
        fid = self.add_file("fresh.jpg")
        row = self.conn.execute(
            "SELECT tier FROM media_class WHERE file_id = ?", (fid,)).fetchone()
        self.assertIsNone(row)
        stats = classify(self.cfg, self.conn, use_clip=False)
        self.assertEqual(stats.processed, 1)


if __name__ == "__main__":
    unittest.main()
