"""F91: faces looks for a face on a preview and decodes the original only if it found one.

Two halves, both without a model:

- `_decode_preview_for_faces` on REAL files (Pillow only) — the gate frame itself: its
  size, its BGR channel order, its rotation, and every case where it must give up and
  let the old full-resolution path take the frame;
- `detect_faces` with both decode paths faked, so a "frame" says which file it is and
  which decode produced it. That is what pins the property the feature stands on: what
  is written into faces comes from the ORIGINAL, never from the preview, while the
  frames with no face never reach a full decode at all.
"""
from __future__ import annotations

import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import sorta.faces as faces_mod
from sorta import imaging
from sorta.faces import (
    EMBED_DIM,
    _decode_preview_for_faces,
    _GateDecoder,
    _split_for_gate,
    detect_faces,
)
from tests.test_faces import FacesTestCase
from tests.test_faces_parallel import path_index
from tests.test_imaging_preview import PreviewCacheTestCase, make_photo

INFER_WORKERS = 2
DECODE_WORKERS = 2
BBOX = [0.0, 0.0, 100.0, 100.0]
# A gated file's dimensions in the index: bigger than the preview, so the gate applies.
BIG = (4000, 3000)


# --- the gate frame --------------------------------------------------------

class DecodePreviewTest(PreviewCacheTestCase):
    """`_decode_preview_for_faces` on real JPEGs, through the real preview cache."""

    def photo(self, name: str = "big.jpg", size=(2400, 1600), **kwargs) -> Path:
        path = self.root / name
        make_photo(path, size=size, **kwargs)
        return path

    def test_frame_comes_back_at_the_preview_size(self):
        path = self.photo()
        img = _decode_preview_for_faces(str(path), None)
        assert img is not None
        # 2400x1600 -> 1536x1024: numpy is (height, width, channels)
        self.assertEqual(img.shape, (1024, 1536, 3))
        self.assertEqual(img.dtype, np.uint8)
        self.assertTrue(img.flags["C_CONTIGUOUS"], "insightface needs a contiguous buffer")

    def test_channels_are_bgr_like_the_full_resolution_path(self):
        path = self.photo()
        st = path.stat()
        rgb = imaging.decode_rgb_preview(
            str(path), st.st_mtime, st.st_size, max_edge=imaging.preview_max_edge())
        assert rgb is not None
        img = _decode_preview_for_faces(str(path), None)
        assert img is not None
        # the detector is fed BGR (_read_image_bgr does the same via cv2)
        np.testing.assert_array_equal(img[:, :, ::-1], np.asarray(rgb))

    def test_orientation_comes_from_the_index_not_from_the_file(self):
        # exif 6 = rotate 90° clockwise; the preview is stored unrotated, so the
        # rotation must be applied here exactly as _decode_for_faces applies it
        path = self.photo(orientation=6)
        upright = _decode_preview_for_faces(str(path), None)
        rotated = _decode_preview_for_faces(str(path), 6)
        assert upright is not None and rotated is not None
        self.assertEqual(rotated.shape, (1536, 1024, 3))
        np.testing.assert_array_equal(rotated, np.rot90(upright, 3))

    def test_a_picture_smaller_than_the_preview_gets_no_gate(self):
        # a downscale that saves nothing: the caller must take the old path
        self.assertIsNone(_decode_preview_for_faces(str(self.photo(size=(800, 600))), None))

    def test_missing_and_corrupt_files_get_no_gate(self):
        self.assertIsNone(_decode_preview_for_faces(str(self.root / "nope.jpg"), None))
        broken = self.root / "broken.jpg"
        broken.write_bytes(b"this is not a JPEG")
        self.assertIsNone(_decode_preview_for_faces(str(broken), None))

    def test_a_warm_cache_returns_the_same_pixels_as_a_cold_one(self):
        path = self.photo()
        cold = _decode_preview_for_faces(str(path), None)
        warm = _decode_preview_for_faces(str(path), None)
        assert cold is not None and warm is not None
        self.assertTrue(any(self.cache.rglob("*.jpg")), "the cold call must fill the cache")
        np.testing.assert_array_equal(cold, warm)


class DecodePreviewWithoutCacheTest(DecodePreviewTest):
    """The same, with the disk cache switched off — the win is the small decode, not it."""

    env = {imaging.ENV_PREVIEW_CACHE: "0"}

    def test_a_warm_cache_returns_the_same_pixels_as_a_cold_one(self):
        path = self.photo()
        first = _decode_preview_for_faces(str(path), None)
        second = _decode_preview_for_faces(str(path), None)
        assert first is not None and second is not None
        self.assertEqual(list(self.cache.rglob("*.jpg")), [],
                         "nothing may be written while the cache is off")
        np.testing.assert_array_equal(first, second)


class SplitForGateTest(unittest.TestCase):
    """Which rows the gate pass is worth running on (from the index, without a decode)."""

    def rows(self, *sizes: tuple[int | None, int | None]) -> list[dict]:
        return [{"id": i, "path": f"/photos/img_{i}.jpg", "orientation": None,
                 "width": w, "height": h} for i, (w, h) in enumerate(sizes, 1)]

    def ids(self, rows) -> list[int]:
        return [r["id"] for r in rows]

    def test_only_originals_bigger_than_the_preview_are_gated(self):
        edge = imaging.preview_max_edge()
        gated, direct = _split_for_gate(self.rows(
            (4000, 3000),        # 1: a camera frame — the case the feature is about
            (edge, edge),        # 2: exactly the preview size — nothing to save
            (edge + 1, 100),     # 3: one pixel over, on the long edge
            (100, edge + 1),     # 4: the long edge can be either of the two
            (800, 600),          # 5: a small picture
        ))
        self.assertEqual(self.ids(gated), [1, 3, 4])
        self.assertEqual(self.ids(direct), [2, 5])

    def test_unknown_dimensions_keep_the_old_path(self):
        gated, direct = _split_for_gate(self.rows(
            (None, None), (4000, None), (None, 3000), (0, 0), (4000, 3000)))
        self.assertEqual(self.ids(gated), [5])
        self.assertEqual(self.ids(direct), [1, 2, 3, 4])

    def test_no_rows(self):
        self.assertEqual(_split_for_gate([]), ([], []))


class GateDecoderTest(unittest.TestCase):
    """`_GateDecoder`: a preview if there is one, otherwise the old full decode — quietly."""

    def decoder_over(self, preview):
        full_calls: list[str] = []

        def full(path: str, orientation: int | None) -> np.ndarray:
            full_calls.append(path)
            return np.zeros((2, 2, 3), dtype=np.uint8)

        patches = mock.patch.multiple(
            faces_mod, _decode_preview_for_faces=preview, _decode_for_faces=full)
        return _GateDecoder(), full_calls, patches

    def test_a_preview_is_remembered_as_such(self):
        decoder, full_calls, patches = self.decoder_over(
            lambda path, orientation: np.ones((1, 1, 3), dtype=np.uint8))
        with patches:
            img = decoder("/photos/a.jpg", None)
        self.assertEqual(img.shape, (1, 1, 3))
        self.assertEqual(full_calls, [])
        self.assertTrue(decoder.previewed("/photos/a.jpg"))
        self.assertFalse(decoder.previewed("/photos/never-seen.jpg"))

    def test_no_preview_falls_back_to_the_full_decode(self):
        decoder, full_calls, patches = self.decoder_over(lambda path, orientation: None)
        with patches:
            img = decoder("/photos/b.jpg", None)
        self.assertEqual(img.shape, (2, 2, 3))
        self.assertEqual(full_calls, ["/photos/b.jpg"])
        self.assertFalse(decoder.previewed("/photos/b.jpg"))

    def test_a_raising_preview_does_not_fail_a_frame_the_old_path_can_read(self):
        def boom(path: str, orientation: int | None) -> np.ndarray:
            raise MemoryError("preview decode blew up")

        decoder, full_calls, patches = self.decoder_over(boom)
        with patches:
            img = decoder("/photos/c.jpg", None)
        self.assertEqual(img.shape, (2, 2, 3))
        self.assertEqual(full_calls, ["/photos/c.jpg"])
        self.assertFalse(decoder.previewed("/photos/c.jpg"))


# --- detect_faces with both decodes faked -----------------------------------

def frame(idx: int, *, preview: bool) -> np.ndarray:
    """A 1×1 "frame" that says which file it is and which decode produced it."""
    return np.array([[[idx, int(preview), 0]]], dtype=np.uint8)


def embedding(idx: int, *, preview: bool) -> np.ndarray:
    """Deliberately different per source: a preview embedding must never be stored."""
    return np.full(EMBED_DIM, float(idx + (100 if preview else 0)), dtype="<f4")


class FakeDecodes:
    """Stands in for both decode paths, recording what each one was asked for.

    `no_preview` are the files without a cheap gate frame (an undecodable source, a
    picture below the preview size); `broken` fail the full decode as a corrupt file
    would.
    """

    def __init__(self, no_preview: frozenset[int] = frozenset(),
                 broken: frozenset[int] = frozenset()) -> None:
        self.no_preview = no_preview
        self.broken = broken
        self._lock = threading.Lock()
        self.preview_calls: list[int] = []
        self.full_calls: list[int] = []

    def preview(self, path: str, orientation: int | None) -> np.ndarray | None:
        idx = path_index(path)
        with self._lock:
            self.preview_calls.append(idx)
        if idx in self.no_preview:
            return None
        return frame(idx, preview=True)

    def full(self, path: str, orientation: int | None) -> np.ndarray:
        idx = path_index(path)
        with self._lock:
            self.full_calls.append(idx)
        if idx in self.broken:
            raise ValueError("corrupt frame")
        return frame(idx, preview=False)


class FakeSessions:
    """An inference session that answers from what it sees — a preview or an original."""

    def __init__(self, faces_on: frozenset[int] = frozenset(),
                 fails_on: frozenset[int] = frozenset()) -> None:
        self.faces_on = faces_on
        self.fails_on = fails_on
        self._lock = threading.Lock()
        self.sessions_built = 0
        self.seen: list[tuple[int, bool]] = []  # (file index, was it a preview)
        self.infer_threads: set[int] = set()

    def __call__(self):
        with self._lock:
            self.sessions_built += 1

        def infer(img: np.ndarray) -> list[tuple[list[float], float, np.ndarray]]:
            idx, preview = int(img[0, 0, 0]), bool(img[0, 0, 1])
            with self._lock:
                self.seen.append((idx, preview))
                self.infer_threads.add(threading.get_ident())
            if idx in self.fails_on:
                raise RuntimeError(f"inference failed on {idx}")
            if idx not in self.faces_on:
                return []
            return [(list(BBOX), 0.95, embedding(idx, preview=preview))]

        return infer

    def previews_seen(self) -> list[int]:
        return sorted(idx for idx, preview in self.seen if preview)

    def originals_seen(self) -> list[int]:
        return sorted(idx for idx, preview in self.seen if not preview)


class GateTestCase(FacesTestCase):
    """A collection of gated files (big enough for a preview to be worth decoding)."""

    infer_workers = INFER_WORKERS

    def setUp(self):
        super().setUp()
        self.cfg.raw = {"faces": {"infer_workers": self.infer_workers,
                                  "decode_workers": DECODE_WORKERS}}

    def add_file(self, size: tuple[int, int] | None = BIG, **kwargs):
        """A file the gate applies to; size=None leaves the dimensions unknown."""
        file_id, path = super().add_file(**kwargs)
        if size is not None:
            with self.conn:
                self.conn.execute("UPDATE files SET width = ?, height = ? WHERE id = ?",
                                  (size[0], size[1], file_id))
        return file_id, path

    def detect(self, decodes: FakeDecodes, sessions: FakeSessions, **kwargs):
        with mock.patch.multiple(faces_mod,
                                 _decode_preview_for_faces=decodes.preview,
                                 _decode_for_faces=decodes.full):
            return detect_faces(self.cfg, self.conn, infer_factory=sessions, **kwargs)

    def faces_by_index(self) -> dict[int, list[tuple[str, tuple[float, ...]]]]:
        """{file index: [(bbox json, embedding)]} — keyed so runs can be compared."""
        out: dict[int, list[tuple[str, tuple[float, ...]]]] = {}
        for r in self.conn.execute(
            """SELECT fl.path AS path, fa.bbox AS bbox, fa.embedding AS embedding
               FROM faces fa JOIN files fl ON fl.id = fa.file_id ORDER BY fa.id"""
        ):
            out.setdefault(path_index(r["path"]), []).append(
                (r["bbox"], tuple(np.frombuffer(r["embedding"], dtype="<f4"))))
        return out


class DetectWithGateTest(GateTestCase):
    """The point of the feature: full decodes only where the gate found a face."""

    def test_faceless_frames_never_reach_a_full_decode(self):
        for _ in range(6):
            self.add_file()
        decodes, sessions = FakeDecodes(), FakeSessions(faces_on=frozenset({2, 5}))
        stats = self.detect(decodes, sessions)

        self.assertEqual(sorted(decodes.preview_calls), [1, 2, 3, 4, 5, 6])
        self.assertEqual(sorted(decodes.full_calls), [2, 5])  # 69% of a real run saved
        self.assertEqual(sessions.previews_seen(), [1, 2, 3, 4, 5, 6])
        self.assertEqual(sessions.originals_seen(), [2, 5])
        self.assertEqual((stats.files_total, stats.files_processed), (6, 6))
        self.assertEqual((stats.faces_found, stats.no_face_files, stats.errors), (2, 4, 0))

    def test_what_is_written_comes_from_the_original(self):
        # the acceptance criterion of the brief, in the only form a mock can pin it:
        # a preview-derived embedding must not survive into faces
        for _ in range(4):
            self.add_file()
        self.detect(FakeDecodes(), FakeSessions(faces_on=frozenset({1, 3})))
        rows = self.faces_by_index()
        for idx in (1, 3):
            self.assertEqual(rows[idx], [("[0.0, 0.0, 100.0, 100.0]",
                                          tuple(embedding(idx, preview=False)))])
        for idx in (2, 4):
            self.assertEqual(rows[idx], [("[]", ())])  # the "processed, no faces" marker

    def test_same_rows_as_a_run_without_the_gate(self):
        for _ in range(8):
            self.add_file()
        faces_on = frozenset({1, 2, 5, 8})
        self.detect(FakeDecodes(), FakeSessions(faces_on=faces_on))
        gated = self.faces_by_index()

        # the very same files with no dimensions in the index — every frame then takes
        # the untouched full-resolution path
        with self.conn:
            self.conn.execute("DELETE FROM faces")
            self.conn.execute("UPDATE files SET width = NULL, height = NULL")
        decodes = FakeDecodes()
        self.detect(decodes, FakeSessions(faces_on=faces_on))

        self.assertEqual(decodes.preview_calls, [], "no gate without known dimensions")
        self.assertEqual(sorted(decodes.full_calls), list(range(1, 9)))
        self.assertEqual(gated, self.faces_by_index())

    def test_the_original_has_the_last_word_when_the_gate_and_it_disagree(self):
        # the gate only asks "is there anything to crop": a frame it passes but the
        # detector then finds nothing on gets the ordinary "no faces" marker
        self.add_file()

        class GateSeesMore(FakeSessions):
            def __call__(self):
                infer = super().__call__()

                def only_on_previews(img):
                    return infer(img) if bool(img[0, 0, 1]) else []

                return only_on_previews

        stats = self.detect(FakeDecodes(), GateSeesMore(faces_on=frozenset({1})))
        self.assertEqual((stats.faces_found, stats.no_face_files), (0, 1))
        self.assertEqual(self.faces_by_index(), {1: [("[]", ())]})

    def test_a_mixed_collection_processes_every_frame_once(self):
        for _ in range(4):
            self.add_file()                 # gated
        for _ in range(3):
            self.add_file(size=None)        # dimensions unknown — the old path
        self.add_file(size=(800, 600))      # smaller than a preview — the old path
        decodes = FakeDecodes()
        stats = self.detect(decodes, FakeSessions(faces_on=frozenset({1, 6})))

        self.assertEqual(sorted(decodes.preview_calls), [1, 2, 3, 4])
        self.assertEqual(sorted(decodes.full_calls), [1, 5, 6, 7, 8])
        self.assertEqual((stats.files_processed, stats.faces_found), (8, 2))
        self.assertEqual(sorted(self.faces_by_index()), list(range(1, 9)))

    def test_progress_counts_every_frame_exactly_once(self):
        for _ in range(6):
            self.add_file()
        calls: list[tuple[int, int]] = []
        self.detect(FakeDecodes(), FakeSessions(faces_on=frozenset({4})),
                    progress=lambda done, total: calls.append((done, total)))
        self.assertEqual([done for done, _ in calls], list(range(1, 7)))
        self.assertTrue(all(total == 6 for _, total in calls))


class GateFallbackTest(GateTestCase):
    """A frame with no cheap preview goes the old way — once, and without a word."""

    def test_the_fallback_frame_is_decoded_and_inferred_exactly_once(self):
        for _ in range(4):
            self.add_file()
        decodes = FakeDecodes(no_preview=frozenset({2, 3}))
        sessions = FakeSessions(faces_on=frozenset({2, 4}))
        stats = self.detect(decodes, sessions)

        # 2 and 3 fell back inside the gate pass: one full decode, one inference, and
        # their hits are the answer — no second visit
        self.assertEqual(sorted(decodes.full_calls), [2, 3, 4])
        self.assertEqual(sessions.previews_seen(), [1, 4])
        self.assertEqual(sessions.originals_seen(), [2, 3, 4])
        self.assertEqual((stats.files_processed, stats.faces_found, stats.errors), (4, 2, 0))
        self.assertEqual(self.faces_by_index()[2],
                         [("[0.0, 0.0, 100.0, 100.0]", tuple(embedding(2, preview=False)))])

    def test_an_undecodable_frame_is_counted_and_left_for_the_next_run(self):
        for _ in range(3):
            self.add_file()
        decodes = FakeDecodes(no_preview=frozenset({2}), broken=frozenset({2}))
        stats = self.detect(decodes, FakeSessions(faces_on=frozenset({1})))
        self.assertEqual((stats.errors, stats.files_processed), (1, 2))
        self.assertNotIn(2, self.faces_by_index())  # no marker row — it will be retried

    def test_an_inference_error_on_the_gate_does_not_touch_the_other_frames(self):
        for _ in range(4):
            self.add_file()
        stats = self.detect(FakeDecodes(),
                            FakeSessions(faces_on=frozenset({1}), fails_on=frozenset({3})))
        self.assertEqual((stats.errors, stats.files_processed), (1, 3))
        self.assertEqual(sorted(self.faces_by_index()), [1, 2, 4])


class GateThreadingTest(GateTestCase):
    """Both passes keep the invariants of F87/F12.1: sessions off-thread, one writer."""

    def test_writes_stay_on_the_calling_thread_across_both_passes(self):
        for _ in range(6):
            self.add_file()
        write_threads: set[int] = set()
        real_write = faces_mod._write_hits

        def spy(conn, s, stats, r, hits, replace=False):
            write_threads.add(threading.get_ident())
            real_write(conn, s, stats, r, hits, replace)

        sessions = FakeSessions(faces_on=frozenset({1, 2, 3}))
        with mock.patch("sorta.faces._write_hits", spy):
            self.detect(FakeDecodes(), sessions)
        self.assertEqual(write_threads, {threading.get_ident()})
        self.assertNotIn(threading.get_ident(), sessions.infer_threads)
        self.assertEqual(len(self.faces_by_index()), 6)

    def test_both_passes_go_through_the_parallel_scheme(self):
        for _ in range(4):
            self.add_file()
        batches: list[int] = []
        real_parallel = faces_mod._detect_parallel

        def spy(rows, decode, factory, workers, decode_workers, on_result):
            batches.append(len(rows))
            real_parallel(rows, decode, factory, workers, decode_workers, on_result)

        with mock.patch("sorta.faces._detect_parallel", spy):
            self.detect(FakeDecodes(), FakeSessions(faces_on=frozenset({2})))
        self.assertEqual(batches, [4, 1])  # the gate over all four, then the one hit


class GateSingleWorkerTest(GateTestCase):
    """infer_workers=1 (the CPU profile): both passes share the one session."""

    infer_workers = 1

    def test_the_session_is_built_once_and_used_on_this_thread(self):
        for _ in range(5):
            self.add_file()
        sessions = FakeSessions(faces_on=frozenset({1, 4}))

        def never(*args, **kwargs):  # pragma: no cover — the assertion is "not called"
            raise AssertionError("infer_workers=1 must not go through _detect_parallel")

        with mock.patch("sorta.faces._detect_parallel", never):
            stats = self.detect(FakeDecodes(), sessions)
        # loading buffalo_l costs seconds — the gate must not pay it a second time
        self.assertEqual(sessions.sessions_built, 1)
        self.assertEqual(sessions.infer_threads, {threading.get_ident()})
        self.assertEqual((stats.files_processed, stats.faces_found), (5, 2))
        self.assertEqual(sessions.originals_seen(), [1, 4])

    def test_an_inference_error_is_counted_and_the_rest_are_written(self):
        for _ in range(3):
            self.add_file()
        stats = self.detect(FakeDecodes(),
                            FakeSessions(faces_on=frozenset({1}), fails_on=frozenset({2})))
        self.assertEqual((stats.errors, stats.files_processed), (1, 2))
        self.assertEqual(sorted(self.faces_by_index()), [1, 3])


class GateRescanTest(GateTestCase):
    """F89 rescan through the gate: old rows replaced, still one row per file."""

    def test_rescan_replaces_the_previous_faces(self):
        for _ in range(3):
            self.add_file()
        self.detect(FakeDecodes(), FakeSessions(faces_on=frozenset({1, 2, 3})))
        self.assertEqual(sorted(self.faces_by_index()), [1, 2, 3])

        decodes = FakeDecodes()
        stats = self.detect(decodes, FakeSessions(faces_on=frozenset({2})), rescan=True)
        self.assertEqual((stats.files_total, stats.files_processed), (3, 3))
        self.assertEqual(sorted(decodes.preview_calls), [1, 2, 3])
        self.assertEqual(decodes.full_calls, [2])
        rows = self.faces_by_index()
        self.assertEqual(rows[1], [("[]", ())])
        self.assertEqual(rows[2],
                         [("[0.0, 0.0, 100.0, 100.0]", tuple(embedding(2, preview=False)))])
        self.assertEqual(rows[3], [("[]", ())])

    def test_rescan_with_a_limit_still_gates(self):
        for _ in range(4):
            self.add_file()
        self.detect(FakeDecodes(), FakeSessions())
        decodes = FakeDecodes()
        stats = self.detect(decodes, FakeSessions(), rescan=True, limit=2)
        self.assertEqual(stats.files_total, 2)
        self.assertEqual(len(decodes.preview_calls), 2)
        self.assertEqual(decodes.full_calls, [])


class AnalyzerPathTest(GateTestCase):
    """The mock analyzer path (tests, smoke) stays serial and never sees the gate."""

    def test_analyzer_runs_on_the_original_only(self):
        for _ in range(3):
            self.add_file()
        seen: list[str] = []

        def analyzer(path: str, orientation: int | None):
            seen.append(path)
            return []

        with mock.patch.object(faces_mod, "_decode_preview_for_faces",
                               mock.Mock(side_effect=AssertionError("gated"))):
            stats = detect_faces(self.cfg, self.conn, analyzer=analyzer)
        self.assertEqual([path_index(p) for p in seen], [1, 2, 3])
        self.assertEqual((stats.files_processed, stats.no_face_files), (3, 3))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
