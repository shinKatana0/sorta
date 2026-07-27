"""F90: the OCR gate as a measurable thing — the extracted decision + the sweep tool.

Two halves, and the seam between them is the point of the feature:

* `junk.clip_verdict` / `junk.ocr_gate_open` / `junk.apply_text_frac` — the fast-tier
  decision, lifted out of the classify() loop. The refactor must be invisible, so the
  central test replays a full classify() run through those functions and demands the
  same media_class rows;
* `scripts/measure_ocr_gate.py` — the table that prices the gate. Its sweep is pure
  arithmetic over per-frame aggregates, so it is tested without CLIP, without easyocr
  and without touching a photo.

The privacy rule of the tool (no paths in the output, no paths in the cache) is a test
here, not a comment: this is a table about documents, and a table about documents must
not become a list of where the documents are.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from sorta import junk
from sorta.config import Config, _naming_from
from sorta.db import connect
from sorta.junk import (
    GateSettings,
    apply_text_frac,
    classify,
    clip_verdict,
    gate_settings,
    ocr_gate_open,
)
from tests.test_junk import _RECEIPT_IDX, FakeClassifier

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "measure_ocr_gate.py"


def _load_script():
    """Import scripts/measure_ocr_gate.py — it is a script, not a package module."""
    spec = importlib.util.spec_from_file_location("measure_ocr_gate", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gate = _load_script()

G = GateSettings(junk_threshold=0.85, document_threshold=0.9, text_frac_min=0.08,
                 text_frac_document=0.15, text_rescue_docscore_min=0.3)


def frame(file_id: int, verdict: str = "photo", doc_score: float = 0.5,
          text_frac: float | None = None, has_faces: bool = False):
    return gate.Frame(file_id, has_faces, verdict, doc_score, text_frac)


class TestGateSettings(unittest.TestCase):
    """Every threshold of the gate in one object — the same one classify() uses."""

    def test_defaults_match_the_config_defaults(self):
        g = gate_settings(Config(database=Path("x.db")))
        self.assertEqual(g.junk_threshold, 0.85)
        self.assertEqual(g.document_threshold, 0.9)
        self.assertEqual(g.text_frac_min, junk._DEFAULT_TEXT_FRAC_MIN)
        self.assertEqual(g.text_frac_document, junk._DEFAULT_TEXT_FRAC_DOCUMENT)
        self.assertEqual(g.text_rescue_docscore_min,
                         junk._DEFAULT_TEXT_RESCUE_DOCSCORE_MIN)

    def test_config_values_win(self):
        cfg = Config(database=Path("x.db"), naming=_naming_from({
            "junk_threshold": 0.7, "document_threshold": 0.8, "text_frac_min": 0.05,
            "text_frac_document": 0.2, "text_rescue_docscore_min": 0.45}))
        g = gate_settings(cfg)
        self.assertEqual(
            (g.junk_threshold, g.document_threshold, g.text_frac_min,
             g.text_frac_document, g.text_rescue_docscore_min),
            (0.7, 0.8, 0.05, 0.2, 0.45))

    def test_missing_f38_fields_fall_back_to_the_module_defaults(self):
        # the getattr pattern junk.py has always used for the late-added fields
        class OldNaming:
            junk_threshold = 0.85
            document_threshold = 0.9

        g = gate_settings(unittest.mock.Mock(naming=OldNaming()))
        self.assertEqual(g.text_frac_min, junk._DEFAULT_TEXT_FRAC_MIN)
        self.assertEqual(g.text_rescue_docscore_min,
                         junk._DEFAULT_TEXT_RESCUE_DOCSCORE_MIN)


class TestClipVerdict(unittest.TestCase):
    """The branch ORDER is the contract (F13/F15/F22), not an implementation detail."""

    def test_screenshot_name_beats_everything(self):
        # F22: an explicit name overrides the document detection and the face veto
        self.assertEqual(
            clip_verdict("photo", 0.99, "screenshot", 0.99, True, G)[0], "screenshot")

    def test_document_clip_goes_before_the_camera_veto(self):
        # F15: a photographed document HAS camera EXIF — the veto must not save it
        verdict, score = clip_verdict("photo", 0.99, None, 0.95, True, G)
        self.assertEqual((verdict, score), ("document", 0.95))

    def test_document_threshold_is_inclusive(self):
        self.assertEqual(clip_verdict("photo", 0.5, None, 0.9, False, G)[0], "document")
        self.assertNotEqual(
            clip_verdict("photo", 0.5, None, 0.89, False, G)[0], "document")

    def test_real_photo_vetoes_the_junk_classes(self):
        self.assertEqual(clip_verdict("meme", 0.99, None, 0.1, True, G)[0], "photo")

    def test_junk_class_only_above_the_threshold(self):
        self.assertEqual(clip_verdict("meme", 0.86, None, 0.1, False, G)[0], "meme")
        self.assertEqual(clip_verdict("meme", 0.84, None, 0.1, False, G)[0], "photo")

    def test_no_doc_score_means_no_document(self):
        # frames with faces never get a document pass — None must not be compared
        self.assertEqual(clip_verdict("photo", 0.99, None, None, False, G)[0], "photo")


class TestOcrGateOpen(unittest.TestCase):
    """Which frames pay for OCR — the exact question F90 measures."""

    def test_faces_close_the_gate(self):
        self.assertFalse(ocr_gate_open(True, "document", 1.0, 0.3))
        self.assertFalse(ocr_gate_open(True, "photo", 1.0, 0.3))

    def test_document_is_gated_regardless_of_doc_score(self):
        # the FP gate is deliberately not limited by the threshold
        self.assertTrue(ocr_gate_open(False, "document", 0.0, 0.9))

    def test_photo_is_gated_by_the_threshold_inclusively(self):
        self.assertTrue(ocr_gate_open(False, "photo", 0.3, 0.3))
        self.assertFalse(ocr_gate_open(False, "photo", 0.29, 0.3))

    def test_screenshot_and_meme_never_reach_ocr(self):
        for verdict in ("screenshot", "meme", "product"):
            self.assertFalse(ocr_gate_open(False, verdict, 1.0, 0.0))


class TestApplyTextFrac(unittest.TestCase):
    """source == 'ocr' means, and only means, that OCR changed the verdict."""

    def test_fn_rescue(self):
        self.assertEqual(apply_text_frac("photo", 0.4, 0.15, G),
                         ("document", 0.15, "ocr"))

    def test_fp_gate(self):
        self.assertEqual(apply_text_frac("document", 0.95, 0.01, G),
                         ("photo", 0.01, "ocr"))

    def test_signal_that_confirms_the_verdict_leaves_the_clip_score(self):
        self.assertEqual(apply_text_frac("photo", 0.4, 0.14, G), ("photo", 0.4, "clip"))
        self.assertEqual(apply_text_frac("document", 0.95, 0.08, G),
                         ("document", 0.95, "clip"))

    def test_no_signal_changes_nothing(self):
        self.assertEqual(apply_text_frac("photo", 0.4, None, G), ("photo", 0.4, "clip"))

    def test_other_verdicts_are_untouched_by_ocr(self):
        for verdict in ("screenshot", "meme", "product"):
            self.assertEqual(apply_text_frac(verdict, 0.9, 0.99, G),
                             (verdict, 0.9, "clip"))


class TestClassifyUsesTheExtractedDecision(unittest.TestCase):
    """The anti-drift test: what the pipeline writes == what the helpers predict.

    The measurement tool prices the gate by calling these functions. If classify() ever
    stops going through them — an inlined branch, a threshold read from somewhere else
    — the table would describe a gate that is not running, and this test is what makes
    that impossible to do quietly.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = Config(sources=[Path(self.tmp.name)],
                          database=Path(self.tmp.name) / "test.db",
                          naming=_naming_from({"clip": {"batch_size": 4}}))
        self.conn = connect(self.cfg.database)
        self.addCleanup(self.conn.close)

    def _add(self, name: str, camera: bool = True, has_face: bool = False) -> int:
        cur = self.conn.execute(
            """INSERT INTO files (path, size, mtime, ext, media_type, width, height,
                   camera_make, camera_model, gps_lat, indexed_at)
               VALUES (?, 1000, 0, 'jpg', 'photo', 4000, 3000, ?, ?, NULL,
                       '2026-01-01')""",
            (f"/photos/{name}", "Canon" if camera else None,
             "EOS" if camera else None))
        fid = cur.lastrowid
        if has_face:
            self.conn.execute(
                "INSERT INTO faces (file_id, bbox, embedding) VALUES (?, '[1,2,3,4]', ?)",
                (fid, b"\x00" * 4))
        self.conn.commit()
        return fid

    def test_rows_match_the_helper_prediction_frame_by_frame(self):
        # a mixed collection: rescued, gated-but-unchanged, ungated, face-vetoed,
        # a confident CLIP document that OCR sends back (FP gate), and a screenshot
        # (name only: the heuristic stays silent on a file with camera EXIF)
        files = {  # name: (doc_score, text_frac, camera EXIF, a detected face)
            "rescue.jpg": (0.5, 0.4, True, False),
            "gated_quiet.jpg": (0.5, 0.01, True, False),
            "clear.jpg": (0.02, None, True, False),
            "face.jpg": (0.5, None, True, True),
            "beach.jpg": (0.95, 0.0, True, False),
            "Screenshot_1.jpg": (0.5, 0.9, False, False),
        }
        ids = {name: self._add(name, camera=spec[2], has_face=spec[3])
               for name, spec in files.items()}
        doc_scores = {name: (_RECEIPT_IDX, spec[0]) for name, spec in files.items()}
        fracs = {name: spec[1] for name, spec in files.items()}

        def detector(path, _width, _height):
            return fracs[Path(path).name]

        stats = classify(self.cfg, self.conn,
                         classifier=FakeClassifier({}, doc_scores=doc_scores),
                         text_detector=detector)
        self.assertEqual(stats.processed, len(files))

        g = gate_settings(self.cfg)
        rows = {r["file_id"]: (r["verdict"], r["source"], r["score"])
                for r in self.conn.execute(
                    "SELECT file_id, verdict, source, score FROM media_class")}
        for name, (doc_score, text_frac, camera, has_faces) in files.items():
            heuristic = junk.heuristic_verdict(
                f"/photos/{name}", 4000, 3000,
                "Canon" if camera else None, "EOS" if camera else None)
            # FakeClassifier gives the main pass "photo" with 0.99 by default; the
            # document pass is not run for frames with faces (doc_score None there)
            verdict, score = clip_verdict("photo", 0.99, heuristic,
                                          None if has_faces else doc_score,
                                          camera or has_faces, g)
            source = "clip"
            if ocr_gate_open(has_faces, verdict, doc_score, g.text_rescue_docscore_min):
                verdict, score, source = apply_text_frac(verdict, score, text_frac, g)
            got_verdict, got_source, got_score = rows[ids[name]]
            self.assertEqual((got_verdict, got_source), (verdict, source),
                             f"mismatch on {name}")
            self.assertAlmostEqual(got_score, score, places=5, msg=name)
        # and the fixture really exercises every branch it claims to
        verdicts = {name: rows[ids[name]][0] for name in files}
        self.assertEqual(verdicts["rescue.jpg"], "document")     # FN rescue
        self.assertEqual(verdicts["gated_quiet.jpg"], "photo")   # gated, unchanged
        self.assertEqual(verdicts["clear.jpg"], "photo")         # never gated
        self.assertEqual(verdicts["face.jpg"], "photo")          # face veto
        self.assertEqual(verdicts["beach.jpg"], "photo")         # FP gate
        self.assertEqual(verdicts["Screenshot_1.jpg"], "screenshot")


class TestSweep(unittest.TestCase):
    """The table: coverage, benefit and the time model per threshold."""

    def frames(self):
        return [
            frame(1, "photo", 0.55, 0.4),    # rescued from 0.5 down
            frame(2, "photo", 0.35, 0.4),    # rescued from 0.3 down
            frame(3, "photo", 0.35, 0.01),   # gated from 0.3 down, no benefit
            frame(4, "photo", 0.1, None),    # below the grid entirely
            frame(5, "document", 0.95, 0.0),  # FP fix at every threshold
            frame(6, "photo", 0.9, None),    # gated everywhere, OCR gave nothing
            frame(7, "photo", 0.5, 0.4, has_faces=True),  # face veto: never gated
        ]

    def test_counts_per_threshold(self):
        rows = gate.sweep(self.frames(), [0.3, 0.5], G, ocr_ms=100.0, startup_sec=0.0)
        low, high = rows
        self.assertEqual((low.gated, low.rescued, low.fp_fixed, low.no_signal),
                         (5, 2, 1, 1))
        self.assertEqual((high.gated, high.rescued, high.fp_fixed, high.no_signal),
                         (3, 1, 1, 1))
        self.assertEqual((low.changed, high.changed), (3, 2))

    def test_coverage_never_grows_with_the_threshold(self):
        rows = gate.sweep(self.frames(), [0.2, 0.3, 0.4, 0.5, 0.6], G,
                          ocr_ms=10.0, startup_sec=0.0)
        gated = [r.gated for r in rows]
        self.assertEqual(gated, sorted(gated, reverse=True))
        # the FP gate does not depend on the threshold — it is the rescue that is lost
        self.assertEqual({r.fp_fixed for r in rows}, {1})

    def test_time_is_gated_frames_plus_a_single_model_start(self):
        rows = gate.sweep(self.frames(), [0.3], G, ocr_ms=200.0, startup_sec=35.0)
        self.assertAlmostEqual(rows[0].seconds, 5 * 0.2 + 35.0)

    def test_a_gate_that_opens_for_nobody_pays_nothing(self):
        # F73 made the pool lazy: no candidate -> no easyocr Reader -> no 35 s
        rows = gate.sweep([frame(1, "photo", 0.1, None)], [0.5], G,
                          ocr_ms=200.0, startup_sec=35.0)
        self.assertEqual((rows[0].gated, rows[0].seconds), (0, 0.0))

    def test_empty_sample(self):
        rows = gate.sweep([], [0.3], G, ocr_ms=200.0, startup_sec=35.0)
        self.assertEqual((rows[0].gated, rows[0].changed, rows[0].seconds), (0, 0, 0.0))


class TestFormatTable(unittest.TestCase):
    """Aggregates only — and readable enough to decide a threshold from."""

    def rows(self):
        return gate.sweep([frame(1, "photo", 0.5, 0.4), frame(2, "photo", 0.1, None)],
                          [0.3, 0.6], G, ocr_ms=100.0, startup_sec=35.0)

    def test_a_row_per_threshold_with_shares_of_the_sample(self):
        text = gate.format_table(self.rows(), total=2, current=0.3)
        self.assertIn("0.30", text)
        self.assertIn("0.60", text)
        self.assertIn("50.0%", text)  # 1 of 2 frames gated at 0.30

    def test_the_configured_threshold_is_marked(self):
        text = gate.format_table(self.rows(), total=2, current=0.3)
        self.assertIn("0.30*", text)
        self.assertIn("naming.text_rescue_docscore_min", text)
        self.assertNotIn("0.60*", text)

    def test_no_paths_in_the_output(self):
        # the table is about documents; it must not say where they are
        text = gate.format_table(self.rows(), total=2, current=0.3)
        for leak in ("\\", ".jpg", "photos"):
            self.assertNotIn(leak, text)

    def test_price_column_survives_a_threshold_that_buys_nothing(self):
        text = gate.format_table(
            gate.sweep([frame(1, "photo", 0.5, 0.01)], [0.3], G, 100.0, 0.0),
            total=1, current=None)
        self.assertIn("—", text)


class TestProbeSummary(unittest.TestCase):
    """What the gate never sees — the number the table itself cannot produce."""

    def test_reports_documents_found_below_the_grid(self):
        frames = [frame(1, "photo", 0.05, 0.4),    # a document nobody would OCR
                  frame(2, "photo", 0.05, 0.0),
                  frame(3, "photo", 0.05, None),   # below the grid, not probed
                  frame(4, "photo", 0.5, 0.4)]     # gated: not part of the probe
        text = gate.probe_summary(frames, 0.2, G)
        self.assertIn("OCR на 2 кадрах из 3", text)
        self.assertIn("документов найдено 1", text)

    def test_silent_when_nothing_was_probed(self):
        self.assertEqual(gate.probe_summary([frame(1, "photo", 0.5, 0.4)], 0.2, G), "")


class TestCache(unittest.TestCase):
    """The cache is what makes a second grid free — and it stores no paths."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "gate.json"

    def test_round_trip(self):
        frames = [frame(1, "photo", 0.5, 0.4), frame(2, "document", 0.95, None,
                                                     has_faces=True)]
        gate.save_cache(self.path, frames, ocr_ms=271.0, floor=0.2)
        loaded, ocr_ms, floor = gate.load_cache(self.path)
        self.assertEqual(loaded, frames)
        self.assertEqual((ocr_ms, floor), (271.0, 0.2))

    def test_holds_file_ids_only(self):
        gate.save_cache(self.path, [frame(7, "document", 0.9, 0.5)], 100.0, 0.2)
        raw = self.path.read_text(encoding="utf-8")
        self.assertNotIn(".jpg", raw)
        self.assertNotIn("path", raw)
        self.assertEqual(json.loads(raw)["frames"][0][0], 7)

    def test_a_cache_of_another_version_is_refused(self):
        gate.save_cache(self.path, [frame(1)], 100.0, 0.2)
        data = json.loads(self.path.read_text(encoding="utf-8"))
        data["version"] = gate.CACHE_VERSION + 1
        self.path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaises(SystemExit):
            gate.load_cache(self.path)


class TestSampleRows(unittest.TestCase):
    """The sample is the same population junk.classify works on."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = Path(self.tmp.name) / "test.db"
        self.conn = connect(self.db)
        self.addCleanup(self.conn.close)

    def _add(self, name: str, media_type: str = "photo", dup_of=None, error=None,
             on_disk: bool = True, has_face: bool = False) -> int:
        path = Path(self.tmp.name) / name
        if on_disk:
            path.write_bytes(b"x")
        cur = self.conn.execute(
            """INSERT INTO files (path, size, mtime, ext, media_type, width, height,
                   dup_of, error, indexed_at)
               VALUES (?, 1, 0, 'jpg', ?, 100, 100, ?, ?, '2026-01-01')""",
            (str(path), media_type, dup_of, error))
        fid = cur.lastrowid
        if has_face:
            self.conn.execute(
                "INSERT INTO faces (file_id, bbox, embedding) VALUES (?, '[1,2,3,4]', ?)",
                (fid, b"\x00" * 4))
        self.conn.commit()
        return fid

    def test_only_canonical_existing_photos(self):
        keep = self._add("keep.jpg")
        original = self._add("orig.jpg")
        self._add("dup.jpg", dup_of=original)
        self._add("broken.jpg", error="boom")
        self._add("clip.mp4", media_type="video")
        self._add("gone.jpg", on_disk=False)
        got = {r["id"] for r in gate.sample_rows(str(self.db), 100, seed=1)}
        self.assertEqual(got, {keep, original})

    def test_has_faces_flag_is_carried(self):
        plain = self._add("plain.jpg")
        portrait = self._add("portrait.jpg", has_face=True)
        rows = {r["id"]: r["has_faces"]
                for r in gate.sample_rows(str(self.db), 100, seed=1)}
        self.assertFalse(rows[plain])
        self.assertTrue(rows[portrait])

    def test_sample_is_capped_and_deterministic(self):
        for i in range(10):
            self._add(f"f{i}.jpg")
        first = [r["id"] for r in gate.sample_rows(str(self.db), 4, seed=7)]
        self.assertEqual(len(first), 4)
        self.assertEqual(first, [r["id"] for r in gate.sample_rows(str(self.db), 4, 7)])


class TestMainReplay(unittest.TestCase):
    """End-to-end without models: a cached run prints the table and touches no photo."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg_path = Path(self.tmp.name) / "config.yaml"
        self.cfg_path.write_text(
            f"database: {Path(self.tmp.name).as_posix()}/sorta.db\n"
            "naming:\n  text_rescue_docscore_min: 0.3\n", encoding="utf-8")
        self.cache = Path(self.tmp.name) / "gate.json"
        gate.save_cache(self.cache, [frame(1, "photo", 0.5, 0.4),
                                     frame(2, "photo", 0.1, None)], 271.0, 0.2)

    def run_main(self, *extra: str) -> str:
        argv = ["measure_ocr_gate.py", "--config", str(self.cfg_path),
                "--cache", str(self.cache), *extra]
        out = io.StringIO()
        with unittest.mock.patch.object(sys, "argv", argv), \
                contextlib.redirect_stdout(out):
            gate.main()
        return out.getvalue()

    def test_replays_the_cache_instead_of_measuring(self):
        with unittest.mock.patch.object(gate, "measure") as measure:
            text = self.run_main()
        measure.assert_not_called()
        self.assertIn("ГЕЙТ OCR", text)
        self.assertIn("0.30*", text)  # the threshold from the config file is marked

    def test_ocr_ms_override_changes_only_the_time_column(self):
        slow = self.run_main("--ocr-ms", "1000", "--startup-sec", "0",
                             "--thresholds", "0.3")
        fast = self.run_main("--ocr-ms", "10", "--startup-sec", "0",
                             "--thresholds", "0.3")
        self.assertIn("1 с", slow)
        self.assertIn("0 с", fast)

    def test_an_empty_grid_is_refused(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            self.run_main("--thresholds")

    def test_a_cache_measured_above_the_grid_floor_is_flagged(self):
        gate.save_cache(self.cache, [frame(1, "photo", 0.5, 0.4)], 271.0, 0.4)
        text = self.run_main("--thresholds", "0.2", "0.4")
        self.assertIn("ВНИМАНИЕ", text)


if __name__ == "__main__":
    unittest.main()
