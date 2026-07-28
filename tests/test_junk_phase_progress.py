"""F100: the junk stage names the phase it is in — the bar stops freezing.

The stage reported no phases at all, and with the deep tier on that showed: the frame
counter ran through the fast pass and then stood at 100% for the whole VLM pass (on the
live run of 2026-07-28, 24 196 frames, that was forty minutes in which only the GPU
load told a working model from a hung process).

The point is an honest percent, not a caption: unlike HDBSCAN in clustering (F84), the
VLM pass IS measurable — the gate's candidate list is known before the loop starts — so
that phase reports `(done, total)` over the candidates, and the denominator switch is
readable exactly because the caption switches with it.

Everything else must be unchanged: verdicts, thresholds, a run without the deep tier,
and a callback that has no `phase` channel at all (the CLI path, quiet mode, tests).
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sorta.config import Config, _naming_from
from sorta.db import connect
from sorta.junk import (
    CLASSIFY_PHASE_CLIP,
    CLASSIFY_PHASE_OCR,
    CLASSIFY_PHASE_VLM,
    CLASSIFY_PHASE_WRITE,
    classify,
)
from tests.test_junk import NO_OCR, FakeClassifier, _RECEIPT_IDX


class _Recorder:
    """A stage callback with the phase channel (like ui._StageProgress/TaskProgress)."""

    def __init__(self) -> None:
        self.calls: list[tuple[str | None, int, int | None]] = []
        self.current: str | None = None
        self.phase_calls: list[str] = []

    def __call__(self, done: int, total: int | None = None) -> None:
        self.calls.append((self.current, done, total))

    def phase(self, name: str) -> None:
        self.current = name
        self.phase_calls.append(name)

    @property
    def phases(self) -> list[str]:
        """Phase names in the order they were first reported."""
        seen: list[str] = []
        for name in self.phase_calls:
            if name not in seen:
                seen.append(name)
        return seen

    def totals_of(self, phase: str) -> list[int | None]:
        return [total for name, _done, total in self.calls if name == phase]

    def dones_of(self, phase: str) -> list[int]:
        return [done for name, done, _total in self.calls if name == phase]


class JunkPhaseTestBase(unittest.TestCase):
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

    def set_batch_size(self, size):
        object.__setattr__(self.cfg.naming, "clip_batch_size", size)

    def add_file(self, name, camera_make=None, camera_model=None, has_face=False):
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

    def candidate_clf(self, names):
        """CLIP mock making every name a VLM candidate: doc_score 0.5 is above
        text_rescue_docscore_min (0.3) — a candidate — but below document_threshold
        (0.9), so the fast verdict stays 'photo' and nothing about it is under test."""
        return FakeClassifier({n: (0, 0.99) for n in names},
                              doc_scores={n: (_RECEIPT_IDX, 0.5) for n in names})

    def verdicts(self):
        return {r["file_id"]: r["verdict"] for r in self.conn.execute(
            "SELECT file_id, verdict FROM media_class")}


class TestFastTierPhases(JunkPhaseTestBase):
    """The fast pass: CLIP -> (OCR, only if the gate opened) -> the verdicts and writes."""

    def test_clip_and_write_phases_on_a_run_without_ocr(self):
        self.add_file("beach.jpg")
        rec = _Recorder()
        classify(self.cfg, self.conn, classifier=FakeClassifier({}),
                 text_detector=NO_OCR, progress=rec)
        self.assertEqual(rec.phases, [CLASSIFY_PHASE_CLIP, CLASSIFY_PHASE_WRITE])

    def test_ocr_phase_appears_when_the_gate_opens(self):
        self.add_file("scan.jpg")
        rec = _Recorder()
        classify(self.cfg, self.conn, classifier=self.candidate_clf(["scan.jpg"]),
                 text_detector=NO_OCR, progress=rec)
        self.assertEqual(
            rec.phases,
            [CLASSIFY_PHASE_CLIP, CLASSIFY_PHASE_OCR, CLASSIFY_PHASE_WRITE])

    def test_total_is_reported_from_the_very_first_call(self):
        # #37: a small/fast stage must still hand the denominator over immediately,
        # and now it arrives already labelled with a phase.
        for i in range(3):
            self.add_file(f"IMG_{i}.jpg")
        rec = _Recorder()
        classify(self.cfg, self.conn, classifier=FakeClassifier({}),
                 text_detector=NO_OCR, progress=rec)
        self.assertEqual(rec.calls[0], (CLASSIFY_PHASE_CLIP, 0, 3))

    def test_counter_does_not_restart_between_fast_phases(self):
        # The three fast phases share ONE counter of frames (they interleave per chunk,
        # F73) — a bar that jumped back to zero three times per chunk would be worse
        # than no phases at all.
        self.set_batch_size(2)
        for i in range(6):
            self.add_file(f"IMG_{i}.jpg")
        rec = _Recorder()
        classify(self.cfg, self.conn, classifier=FakeClassifier({}),
                 text_detector=NO_OCR, progress=rec)
        dones = [done for _name, done, _total in rec.calls]
        self.assertEqual(dones, sorted(dones))
        self.assertEqual(dones[-1], 6)
        self.assertEqual({t for _n, _d, t in rec.calls}, {6})

    def test_phase_is_not_re_reported_while_it_does_not_change(self):
        # The UI restarts the phase clock on every report; a repeated name would keep
        # resetting it for nothing.
        self.set_batch_size(1)
        for i in range(4):
            self.add_file(f"IMG_{i}.jpg")
        rec = _Recorder()
        classify(self.cfg, self.conn, classifier=FakeClassifier({}),
                 text_detector=NO_OCR, progress=rec)
        for prev, nxt in zip(rec.phase_calls, rec.phase_calls[1:]):
            self.assertNotEqual(prev, nxt)

    def test_every_chunk_reports_its_phases(self):
        # "No phase may stay silent for more than a few seconds": with several chunks
        # the fast phases come round again instead of being named once at the start.
        self.set_batch_size(1)
        for i in range(3):
            self.add_file(f"IMG_{i}.jpg")
        rec = _Recorder()
        classify(self.cfg, self.conn, classifier=FakeClassifier({}),
                 text_detector=NO_OCR, progress=rec)
        self.assertEqual(rec.phase_calls.count(CLASSIFY_PHASE_CLIP), 3)
        self.assertEqual(rec.phase_calls.count(CLASSIFY_PHASE_WRITE), 3)

    def test_heuristics_only_run_reports_the_write_phase(self):
        # use_clip=False classifies nothing — it only writes verdicts, and says so.
        self.add_file("Screenshot_1.png")
        self.add_file("IMG_1.jpg")
        rec = _Recorder()
        classify(self.cfg, self.conn, use_clip=False, progress=rec)
        self.assertEqual(rec.phases, [CLASSIFY_PHASE_WRITE])
        self.assertEqual(rec.totals_of(CLASSIFY_PHASE_WRITE), [2, 2])
        self.assertEqual(rec.dones_of(CLASSIFY_PHASE_WRITE)[-1], 2)


class TestDeepTierPhase(JunkPhaseTestBase):
    """The phase that used to leave the bar standing at 100%."""

    def test_vlm_phase_follows_the_fast_ones(self):
        self.add_file("scan.jpg")
        rec = _Recorder()
        self.enable_vlm()
        classify(self.cfg, self.conn, classifier=self.candidate_clf(["scan.jpg"]),
                 text_detector=NO_OCR, vlm_classifier=lambda path: "document",
                 progress=rec)
        self.assertEqual(
            rec.phases,
            [CLASSIFY_PHASE_CLIP, CLASSIFY_PHASE_OCR, CLASSIFY_PHASE_WRITE,
             CLASSIFY_PHASE_VLM])
        # the deep pass is the tail of the stage: nothing is reported after it
        self.assertEqual(rec.calls[-1][0], CLASSIFY_PHASE_VLM)

    def test_vlm_phase_counts_the_gate_candidates(self):
        names = [f"scan_{i}.jpg" for i in range(5)]
        for name in names:
            self.add_file(name)
        self.add_file("beach.jpg")  # not a candidate — outside the VLM denominator
        rec = _Recorder()
        self.enable_vlm()
        stats = classify(self.cfg, self.conn, classifier=self.candidate_clf(names),
                         text_detector=NO_OCR, vlm_classifier=lambda path: "document",
                         progress=rec)
        self.assertEqual(stats.vlm_candidates, 5)
        self.assertEqual(set(rec.totals_of(CLASSIFY_PHASE_VLM)), {5})
        # done grows one candidate at a time, up to the total — the whole point: at
        # ~1.2s a frame the user can tell "1 200 of 1 843" from "probably hung".
        self.assertEqual(rec.dones_of(CLASSIFY_PHASE_VLM), [0, 1, 2, 3, 4, 5])

    def test_a_vlm_error_on_one_frame_still_moves_the_counter(self):
        # #31: a model error leaves the fast verdict in place — but the frame IS done,
        # and a counter that skipped it would never reach its total.
        names = ["scan_0.jpg", "scan_1.jpg", "scan_2.jpg"]
        for name in names:
            self.add_file(name)
        rec = _Recorder()
        self.enable_vlm()

        def flaky(path):
            if path.endswith("scan_1.jpg"):
                raise RuntimeError("CUDA error: device-side assert triggered")
            return "document"

        classify(self.cfg, self.conn, classifier=self.candidate_clf(names),
                 text_detector=NO_OCR, vlm_classifier=flaky, progress=rec)
        self.assertEqual(rec.dones_of(CLASSIFY_PHASE_VLM), [0, 1, 2, 3])

    def test_denominator_switches_together_with_the_caption(self):
        # The switch from frames to candidates is honest only if the caption changes at
        # the same moment; otherwise the bar just slides backwards silently.
        names = [f"scan_{i}.jpg" for i in range(2)]
        for name in names:
            self.add_file(name)
        rec = _Recorder()
        self.enable_vlm()
        classify(self.cfg, self.conn, classifier=self.candidate_clf(names),
                 text_detector=NO_OCR, vlm_classifier=lambda path: "document",
                 progress=rec)
        fast_totals = {t for name, _d, t in rec.calls if name != CLASSIFY_PHASE_VLM}
        self.assertEqual(fast_totals, {2})
        first_vlm = rec.calls.index(
            next(c for c in rec.calls if c[0] == CLASSIFY_PHASE_VLM))
        self.assertEqual(rec.calls[first_vlm], (CLASSIFY_PHASE_VLM, 0, 2))

    def test_no_vlm_phase_when_the_gate_selects_nobody(self):
        # Zero candidates: the phase is never opened and nothing divides by zero.
        self.add_file("beach.jpg")
        rec = _Recorder()
        self.enable_vlm()

        def vlm(_path):
            raise AssertionError("the VLM must not be called for a clean photo")

        stats = classify(self.cfg, self.conn, classifier=FakeClassifier({}),
                         text_detector=NO_OCR, vlm_classifier=vlm, progress=rec)
        self.assertEqual(stats.vlm_candidates, 0)
        self.assertNotIn(CLASSIFY_PHASE_VLM, rec.phases)
        self.assertEqual(rec.calls[-1][1], 1)  # the fast counter still finished

    def test_no_vlm_phase_when_the_model_could_not_be_built(self):
        # Graceful fallback to the fast tier (F37-B): no deep pass happens, so no deep
        # phase may be announced either.
        self.add_file("scan.jpg")
        rec = _Recorder()
        self.enable_vlm()

        def broken_factory(model_name):
            raise RuntimeError("no CUDA / transformers not installed")

        classify(self.cfg, self.conn, classifier=self.candidate_clf(["scan.jpg"]),
                 text_detector=NO_OCR, vlm_classifier_factory=broken_factory,
                 progress=rec)
        self.assertNotIn(CLASSIFY_PHASE_VLM, rec.phases)


class TestDeepTierOffIsUnchanged(JunkPhaseTestBase):
    """Requirement 4: a run without the deep tier must not look or work differently."""

    def test_no_vlm_phase_and_one_denominator(self):
        for i in range(3):
            self.add_file(f"scan_{i}.jpg")
        rec = _Recorder()
        classify(self.cfg, self.conn,
                 classifier=self.candidate_clf([f"scan_{i}.jpg" for i in range(3)]),
                 text_detector=NO_OCR, progress=rec)
        self.assertNotIn(CLASSIFY_PHASE_VLM, rec.phases)
        self.assertEqual({t for _n, _d, t in rec.calls}, {3})
        self.assertEqual(rec.calls[-1][1], 3)

    def test_same_verdicts_with_and_without_a_callback(self):
        self.add_file("scan.jpg")
        self.add_file("beach.jpg")
        clf = self.candidate_clf(["scan.jpg"])
        classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR)
        without = self.verdicts()
        self.conn.execute("DELETE FROM media_class")
        self.conn.commit()
        classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR,
                 progress=_Recorder())
        self.assertEqual(self.verdicts(), without)

    def test_same_verdicts_with_and_without_the_phase_channel_on_a_deep_run(self):
        names = ["scan.jpg", "beach.jpg"]
        for name in names:
            self.add_file(name)
        self.enable_vlm()
        clf = self.candidate_clf(["scan.jpg"])
        classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR,
                 vlm_classifier=lambda path: "product",
                 progress=lambda done, total: None)
        plain = self.verdicts()
        self.conn.execute("DELETE FROM media_class")
        self.conn.commit()
        classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR,
                 vlm_classifier=lambda path: "product", progress=_Recorder())
        self.assertEqual(self.verdicts(), plain)


class TestCallbacksWithoutAPhaseChannel(JunkPhaseTestBase):
    """Requirement 5: `progress.phase` is not something every caller has."""

    def test_plain_function_gets_the_counter_and_no_error(self):
        names = ["scan.jpg", "beach.jpg"]
        for name in names:
            self.add_file(name)
        self.enable_vlm()
        seen: list[tuple[int, int | None]] = []
        stats = classify(self.cfg, self.conn, classifier=self.candidate_clf(["scan.jpg"]),
                         text_detector=NO_OCR, vlm_classifier=lambda path: "document",
                         progress=lambda done, total: seen.append((done, total)))
        self.assertEqual(stats.vlm_candidates, 1)
        self.assertIn((0, 2), seen)   # the fast pass, by frames
        self.assertIn((1, 1), seen)   # the deep pass, by candidates
        self.assertEqual(self.verdicts(), {1: "document", 2: "photo"})

    def test_no_callback_at_all_still_classifies(self):
        self.add_file("scan.jpg")
        self.enable_vlm()
        stats = classify(self.cfg, self.conn, classifier=self.candidate_clf(["scan.jpg"]),
                         text_detector=NO_OCR, vlm_classifier=lambda path: "document")
        self.assertEqual(stats.processed, 1)
        self.assertEqual(self.verdicts(), {1: "document"})

    def test_an_object_whose_phase_is_not_callable_is_not_used_as_one(self):
        # Duck typing, not hasattr-faith: a callback carrying a `phase` ATTRIBUTE that
        # happens not to be callable must be treated as a phase-less callback.
        self.add_file("beach.jpg")

        def cb(done, total=None):
            calls.append((done, total))

        calls: list[tuple[int, int | None]] = []
        cb.phase = "not a channel"  # type: ignore[attr-defined]
        classify(self.cfg, self.conn, classifier=FakeClassifier({}),
                 text_detector=NO_OCR, progress=cb)
        self.assertEqual(calls[-1], (1, 1))


class TestNothingElseMoved(JunkPhaseTestBase):
    """F100 is observability only — incrementality and the verdicts stay put."""

    def test_second_run_reports_nothing_because_there_is_nothing_to_do(self):
        self.add_file("scan.jpg")
        clf = self.candidate_clf(["scan.jpg"])
        classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR)
        rec = _Recorder()
        stats = classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR,
                         progress=rec)
        self.assertEqual(stats.processed, 0)
        self.assertEqual(rec.calls, [])
        self.assertEqual(rec.phases, [])

    def test_ocr_verdicts_are_still_applied_under_the_new_phases(self):
        fid = self.add_file("medform.jpg")
        rec = _Recorder()
        classify(self.cfg, self.conn, classifier=self.candidate_clf(["medform.jpg"]),
                 text_detector=lambda p, w, h: 0.5, progress=rec)
        row = self.conn.execute(
            "SELECT verdict, source FROM media_class WHERE file_id = ?",
            (fid,)).fetchone()
        self.assertEqual((row["verdict"], row["source"]), ("document", "ocr"))
        self.assertIn(CLASSIFY_PHASE_OCR, rec.phases)


class TestPhaseKeysAreStable(unittest.TestCase):
    """The keys are an interface: the served UI and (separately) the CLI label them."""

    def test_keys_are_identifiers_not_captions(self):
        keys = [CLASSIFY_PHASE_CLIP, CLASSIFY_PHASE_OCR, CLASSIFY_PHASE_VLM,
                CLASSIFY_PHASE_WRITE]
        self.assertEqual(len(set(keys)), 4)
        for key in keys:
            self.assertTrue(key.startswith("junk_"), key)
            self.assertTrue(key.isascii() and key.replace("_", "").isalnum(), key)


if __name__ == "__main__":
    unittest.main()
