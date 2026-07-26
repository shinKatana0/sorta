"""F80: video_filmstrip — several frames of one clip, in the F67 preview cache.

Same rules as the F74 suite this builds on: every clip is generated on the spot with
PyAV, nothing touches the real collection or the network, and the cases that need a
real decoder are skipped where `av` is missing. The arithmetic (which seconds, which
key) and the seek/decode loop are covered against a fake av everywhere.

The one test that must never be deleted is the frame-0 compatibility one: if frame 0
of the strip ever stops being the frame F74 wrote, every tile already in the cache of
a 227 GB collection is silently invalidated.
"""
from __future__ import annotations

import importlib.util
import os
import unittest
import unittest.mock
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from sorta import imaging
from tests.test_imaging_preview import make_photo, stat_key
from tests.test_imaging_video import (
    BLACK,
    RED,
    FakeContainer,
    FakeFrame,
    VideoPreviewTestCase,
    fake_av,
    make_video,
)

HAVE_AV = importlib.util.find_spec("av") is not None

# 60 frames at 10 fps = 6 s: long enough that the six targets (~0.6 s apart) land on
# clearly different frames, short enough to encode in a fraction of a second.
CLIP_FRAMES = 60
CLIP_FPS = 10


def gradient_colors(count: int = CLIP_FRAMES) -> list[tuple[int, int, int]]:
    """Frame 0 black (a fade-in, as in the F74 fixture), then brightening steadily.

    A clip whose brightness only grows lets a test state "the frames are different"
    and "they are in ascending order of time" as one arithmetic check on the means.
    """
    return [BLACK] + [
        (30 + index * 3, 20 + index * 2, 10 + index) for index in range(1, count)
    ]


def make_gradient_video(path: Path, count: int = CLIP_FRAMES) -> None:
    make_video(path, colors=gradient_colors(count), fps=CLIP_FPS)


def means(frames: list[Image.Image]) -> list[float]:
    return [float(np.asarray(frame).mean()) for frame in frames]


def counting_av_open(opened: list[str]):
    """A replacement for av.open that records every path it is handed."""
    import av

    real_open = av.open

    def counting(path, *args, **kwargs):
        opened.append(str(path))
        return real_open(path, *args, **kwargs)

    return counting


class TestFrameCountSetting(unittest.TestCase):
    def test_default_and_env_override(self):
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(imaging.video_frames(), imaging.VIDEO_FRAMES)
        for value, expected in (("1", 1), ("12", 12), ("nope", imaging.VIDEO_FRAMES),
                                ("0", imaging.VIDEO_FRAMES), ("-3", imaging.VIDEO_FRAMES)):
            with unittest.mock.patch.dict(os.environ, {imaging.ENV_VIDEO_FRAMES: value}):
                self.assertEqual(imaging.video_frames(), expected)


class TestPreviewKeyFrames(unittest.TestCase):
    """The key of frame 0 may never move — it addresses every preview already stored."""

    def test_frame_zero_keeps_the_pre_f80_key(self):
        with_frame = imaging.preview_key("C:/clips/a.mp4", 1.5, 100, 0)
        self.assertEqual(with_frame, imaging.preview_key("C:/clips/a.mp4", 1.5, 100))

    def test_every_frame_gets_its_own_key(self):
        keys = [imaging.preview_key("C:/clips/a.mp4", 1.5, 100, i) for i in range(6)]
        self.assertEqual(len(set(keys)), 6)


class TestFilmstripTargets(unittest.TestCase):
    """Which seconds are grabbed — no decoder needed."""

    def targets(self, count: int, *, duration_ms: int = 60_000) -> list[float]:
        stream = SimpleNamespace(duration=duration_ms, time_base=Fraction(1, 1000))
        container = SimpleNamespace(duration=duration_ms * 1000)
        return imaging._filmstrip_targets(container, stream, count)

    def test_first_target_is_exactly_the_f74_one(self):
        stream = SimpleNamespace(duration=60_000, time_base=Fraction(1, 1000))
        container = SimpleNamespace(duration=60_000_000)
        self.assertEqual(
            self.targets(6)[0], imaging._target_seconds(container, stream))

    def test_default_count_spreads_over_the_briefed_fractions(self):
        # 60 s clip: the F74 second, then 20/40/60/80/95% of the duration.
        self.assertEqual(
            [round(t, 3) for t in self.targets(6)], [1.0, 12.0, 24.0, 36.0, 48.0, 57.0])

    def test_targets_ascend_and_stay_inside_the_clip(self):
        targets = self.targets(6)
        self.assertEqual(targets, sorted(targets))
        self.assertLess(targets[-1], 60.0)

    def test_one_frame_asks_for_the_f74_second_only(self):
        self.assertEqual(self.targets(1), [imaging.VIDEO_FRAME_SECONDS])

    def test_other_counts_still_end_on_the_tail_fraction(self):
        for count in (2, 3, 4, 9):
            targets = self.targets(count)
            self.assertEqual(len(targets), count)
            self.assertEqual(targets, sorted(targets))
            self.assertAlmostEqual(targets[-1], 60.0 * imaging.VIDEO_LAST_FRACTION)

    def test_unknown_duration_falls_back_to_a_single_target(self):
        stream = SimpleNamespace(duration=None, time_base=None)
        container = SimpleNamespace(duration=None)
        self.assertEqual(
            imaging._filmstrip_targets(container, stream, 6),
            [imaging.VIDEO_FRAME_SECONDS])


class TestGrabFilmstripAgainstFakeAv(unittest.TestCase):
    """The single-open seek loop itself, without a real decoder (CI has no av)."""

    def strip(self, frames, count=6) -> list[Image.Image]:
        container = FakeContainer(frames)
        self.container = container
        return imaging._grab_filmstrip(fake_av(container), "clip.mp4", count)

    def test_one_open_many_seeks_and_the_container_is_closed(self):
        frames = [FakeFrame(index / 10, RED) for index in range(10)]
        strip = self.strip(frames)
        self.assertEqual(len(strip), 6)
        self.assertEqual(len(self.container.seeks), 6)  # six targets, one container
        self.assertTrue(self.container.closed)

    def test_a_repeated_timestamp_is_not_returned_twice(self):
        # A 1 s fake clip with three frames: the first is before the skipped start, so
        # the six targets can only collapse onto the two behind it — not six copies.
        strip = self.strip([FakeFrame(0.0, BLACK), FakeFrame(0.15, RED),
                            FakeFrame(0.5, RED)])
        self.assertEqual(len(strip), 2)

    def test_a_stream_without_frames_gives_an_empty_strip(self):
        self.assertEqual(self.strip([]), [])

    def test_rotation_is_applied_to_every_frame(self):
        frames = [FakeFrame(index / 10, RED, rotation=90) for index in range(10)]
        for img in self.strip(frames):
            self.assertEqual(img.size, (8, 16))  # 16x8 landscape -> portrait

    def test_a_decode_that_blows_up_keeps_the_frames_already_taken(self):
        container = FakeContainer([FakeFrame(index / 10, RED) for index in range(10)])
        calls = {"n": 0}
        real_decode_at = imaging._decode_at

        def exploding(c, stream, target):
            calls["n"] += 1
            if calls["n"] > 2:
                raise RuntimeError("truncated stream")
            return real_decode_at(c, stream, target)

        with unittest.mock.patch.object(imaging, "_decode_at", exploding):
            strip = imaging._grab_filmstrip(fake_av(container), "clip.mp4", 6)
        self.assertEqual(len(strip), 2)
        self.assertTrue(container.closed)


class TestFilmstripWithoutAv(unittest.TestCase):
    def test_missing_package_gives_an_empty_strip_not_an_exception(self):
        with unittest.mock.patch.object(imaging, "_import_av", lambda: None):
            self.assertEqual(imaging._extract_filmstrip("clip.mp4", 6), [])


class TestFilmstripGate(unittest.TestCase):
    def test_the_whole_strip_is_taken_under_one_semaphore_slot(self):
        """Six frames per clip must not raise the number of clips decoded at once."""
        with unittest.mock.patch.object(imaging, "_video_gate") as gate, \
                unittest.mock.patch.object(imaging, "_import_av",
                                           lambda: SimpleNamespace()), \
                unittest.mock.patch.object(imaging, "_grab_filmstrip",
                                           lambda av, path, count: []):
            imaging._extract_filmstrip("clip.mp4", 6)
        self.assertEqual(gate.call_count, 1)


@unittest.skipUnless(HAVE_AV, "PyAV is not installed (it ships with the `vlm` extra)")
class TestFilmstrip(VideoPreviewTestCase):
    def test_six_frames_are_returned_and_they_are_all_different(self):
        src = self.root / "clip.mp4"
        make_gradient_video(src)
        strip = imaging.video_filmstrip(src, *stat_key(src), max_edge=128)

        self.assertEqual(len(strip), 6)
        values = means(strip)
        self.assertEqual(len(set(round(v, 1) for v in values)), 6, values)

    def test_frames_ascend_in_time_and_the_very_start_is_skipped(self):
        src = self.root / "clip.mp4"
        make_gradient_video(src)  # frame 0 is black, brightness grows from there
        strip = imaging.video_filmstrip(src, *stat_key(src), max_edge=128)
        values = means(strip)

        self.assertEqual(values, sorted(values), values)
        self.assertGreater(values[0], 20.0)  # the black frame 0 of the file, not taken

    def test_frame_zero_is_byte_for_byte_the_f74_single_frame(self):
        """The cache guard: F80 must not redraw a single tile the UI already has."""
        src = self.root / "clip.mp4"
        make_gradient_video(src)
        mtime, size = stat_key(src)

        single = imaging.decode_rgb_preview(src, mtime, size, max_edge=128)
        strip = imaging.video_filmstrip(src, mtime, size, max_edge=128)

        self.assertIsNotNone(single)
        self.assertEqual(strip[0].tobytes(), single.tobytes())
        self.assertEqual(strip[0].size, single.size)

    def test_frame_zero_matches_even_when_the_strip_is_built_first(self):
        """The same guard the other way round — a cold cache, the lightbox opened first."""
        src = self.root / "clip.mp4"
        make_gradient_video(src)
        mtime, size = stat_key(src)

        strip = imaging.video_filmstrip(src, mtime, size, max_edge=128)
        single = imaging.decode_rgb_preview(src, mtime, size, max_edge=128)

        self.assertEqual(strip[0].tobytes(), single.tobytes())

    def test_frame_zero_lands_on_the_pre_f80_cache_key(self):
        src = self.root / "clip.mp4"
        make_gradient_video(src)
        mtime, size = stat_key(src)
        imaging.video_filmstrip(src, mtime, size, max_edge=128)

        key = imaging.preview_key(src, mtime, size)
        self.assertTrue((self.cache / key[:2] / f"{key}.jpg").is_file())
        self.assertEqual(len(self.previews()), 6)  # one ordinary JPEG per frame

    def test_the_whole_strip_costs_exactly_one_container_open(self):
        import av

        src = self.root / "clip.mp4"
        make_gradient_video(src)
        opened: list[str] = []
        with unittest.mock.patch.object(av, "open", counting_av_open(opened)):
            strip = imaging.video_filmstrip(src, *stat_key(src), max_edge=128)

        self.assertEqual(len(strip), 6)
        self.assertEqual(opened, [str(src)])  # not six opens of a 4K file

    def test_a_second_call_reads_the_cache_without_opening_the_container(self):
        import av

        src = self.root / "clip.mp4"
        make_gradient_video(src)
        mtime, size = stat_key(src)
        opened: list[str] = []
        with unittest.mock.patch.object(av, "open", counting_av_open(opened)):
            cold = imaging.video_filmstrip(src, mtime, size, max_edge=128)
            warm = imaging.video_filmstrip(src, mtime, size, max_edge=128)

        self.assertEqual(len(opened), 1)  # the cold call only
        self.assertEqual([f.tobytes() for f in cold], [f.tobytes() for f in warm])

    def test_a_tile_already_in_the_cache_does_not_pass_for_a_built_strip(self):
        """F74 leaves frame 0 behind for every clip — that is not a one-frame strip."""
        src = self.root / "clip.mp4"
        make_gradient_video(src)
        mtime, size = stat_key(src)

        imaging.decode_rgb_preview(src, mtime, size, max_edge=128)  # the grid tile
        self.assertEqual(len(self.previews()), 1)
        strip = imaging.video_filmstrip(src, mtime, size, max_edge=128)
        self.assertEqual(len(strip), 6)

    def test_video_frame_serves_one_index_and_404_material_past_the_end(self):
        src = self.root / "clip.mp4"
        make_gradient_video(src)
        mtime, size = stat_key(src)

        self.assertIsNotNone(imaging.video_frame(src, mtime, size, 0, max_edge=128))
        self.assertIsNotNone(imaging.video_frame(src, mtime, size, 5, max_edge=128))
        self.assertIsNone(imaging.video_frame(src, mtime, size, 6, max_edge=128))
        self.assertIsNone(imaging.video_frame(src, mtime, size, 99, max_edge=128))
        self.assertIsNone(imaging.video_frame(src, mtime, size, -1, max_edge=128))

    def test_video_frame_reads_a_warm_frame_without_opening_the_container(self):
        import av

        src = self.root / "clip.mp4"
        make_gradient_video(src)
        mtime, size = stat_key(src)
        imaging.video_filmstrip(src, mtime, size, max_edge=128)

        opened: list[str] = []
        with unittest.mock.patch.object(av, "open", counting_av_open(opened)):
            frame = imaging.video_frame(src, mtime, size, 3, max_edge=128)
        self.assertIsNotNone(frame)
        self.assertEqual(opened, [])

    def test_a_clip_shorter_than_the_strip_returns_fewer_frames(self):
        src = self.root / "short.mp4"
        make_video(src, colors=[BLACK, RED, RED], fps=CLIP_FPS)  # 0.3 s, 3 frames
        strip = imaging.video_filmstrip(src, *stat_key(src), max_edge=128)

        self.assertGreaterEqual(len(strip), 1)
        self.assertLess(len(strip), 6)

    def test_a_corrupt_file_gives_an_empty_strip_without_raising(self):
        src = self.root / "broken.mp4"
        src.write_bytes(b"\x00\x01garbage that is definitely not a container" * 50)
        self.assertEqual(imaging.video_filmstrip(src, *stat_key(src), max_edge=128), [])
        self.assertEqual(self.previews(), [])

    def test_a_truncated_file_gives_an_empty_strip(self):
        src = self.root / "trunc.mp4"
        make_gradient_video(src)
        data = src.read_bytes()
        src.write_bytes(data[:len(data) // 4])
        self.assertEqual(imaging.video_filmstrip(src, *stat_key(src), max_edge=128), [])

    def test_a_missing_file_gives_an_empty_strip(self):
        self.assertEqual(
            imaging.video_filmstrip(self.root / "nope.mp4", 1.0, 10, max_edge=128), [])

    def test_a_file_without_a_video_stream_gives_an_empty_strip(self):
        src = self.root / "audio.mp4"
        src.write_bytes(b"still not a container")  # PyAV: no video stream to speak of
        self.assertEqual(imaging.video_filmstrip(src, *stat_key(src), max_edge=128), [])

    def test_rotation_is_applied_to_the_whole_strip(self):
        src = self.root / "portrait.mp4"
        make_video(src, colors=gradient_colors(), fps=CLIP_FPS, rotation=90)
        strip = imaging.video_filmstrip(src, *stat_key(src), max_edge=128)

        self.assertEqual(len(strip), 6)
        for frame in strip:  # 640x360 landscape source, stood up by the matrix
            self.assertLess(frame.size[0], frame.size[1])

    def test_a_cache_write_failure_still_returns_the_frames(self):
        src = self.root / "clip.mp4"
        make_gradient_video(src)

        def failing_save(*args, **kwargs):
            raise OSError("disk full")

        with unittest.mock.patch.object(Image.Image, "save", failing_save):
            strip = imaging.video_filmstrip(src, *stat_key(src), max_edge=128)
        self.assertEqual(len(strip), 6)
        self.assertEqual(max(strip[0].size), 128)
        self.assertEqual(self.previews(), [])

    def test_a_photo_has_no_filmstrip(self):
        src = self.root / "a.jpg"
        make_photo(src)
        mtime, size = stat_key(src)
        self.assertEqual(imaging.video_filmstrip(src, mtime, size, max_edge=96), [])
        # ...and video_frame keeps serving its single frame, as the lightbox expects
        self.assertIsNotNone(imaging.video_frame(src, mtime, size, 0, max_edge=96))
        self.assertIsNone(imaging.video_frame(src, mtime, size, 1, max_edge=96))


@unittest.skipUnless(HAVE_AV, "PyAV is not installed (it ships with the `vlm` extra)")
class TestFilmstripOfOne(VideoPreviewTestCase):
    """SORTA_VIDEO_FRAMES=1 — the documented way back to plain F74."""

    env = dict(VideoPreviewTestCase.env, SORTA_VIDEO_FRAMES="1")

    def test_one_configured_frame_is_exactly_the_f74_preview(self):
        src = self.root / "clip.mp4"
        make_gradient_video(src)
        mtime, size = stat_key(src)

        strip = imaging.video_filmstrip(src, mtime, size, max_edge=128)
        single = imaging.decode_rgb_preview(src, mtime, size, max_edge=128)

        self.assertEqual(len(strip), 1)
        self.assertEqual(strip[0].tobytes(), single.tobytes())
        self.assertEqual(len(self.previews()), 1)  # one cache entry, as before F80

    def test_the_lightbox_finds_no_second_frame(self):
        src = self.root / "clip.mp4"
        make_gradient_video(src)
        mtime, size = stat_key(src)
        self.assertIsNotNone(imaging.video_frame(src, mtime, size, 0, max_edge=128))
        self.assertIsNone(imaging.video_frame(src, mtime, size, 1, max_edge=128))


@unittest.skipUnless(HAVE_AV, "PyAV is not installed (it ships with the `vlm` extra)")
class TestFilmstripWithVideoPreviewsOff(VideoPreviewTestCase):
    env = dict(VideoPreviewTestCase.env, SORTA_VIDEO_PREVIEWS="0")

    def test_nothing_is_extracted_and_nothing_is_written(self):
        src = self.root / "clip.mp4"
        make_gradient_video(src)
        self.assertEqual(imaging.video_filmstrip(src, *stat_key(src), max_edge=128), [])
        self.assertEqual(self.previews(), [])


@unittest.skipUnless(HAVE_AV, "PyAV is not installed (it ships with the `vlm` extra)")
class TestFilmstripWithoutTheDiskCache(VideoPreviewTestCase):
    env = dict(VideoPreviewTestCase.env, SORTA_PREVIEW_CACHE="0")

    def test_frames_are_still_returned_but_nothing_is_stored(self):
        src = self.root / "clip.mp4"
        make_gradient_video(src)
        strip = imaging.video_filmstrip(src, *stat_key(src), max_edge=128)

        self.assertEqual(len(strip), 6)
        self.assertEqual(max(strip[0].size), 128)
        self.assertEqual(self.previews(), [])


if __name__ == "__main__":
    unittest.main()
