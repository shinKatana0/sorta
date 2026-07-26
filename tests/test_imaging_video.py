"""F74: video previews — one extracted frame in the same disk cache as photos.

Every clip used here is generated on the spot with PyAV (a few solid-colour frames,
mpeg4): the suite never touches the real collection and never goes to the network.

PyAV comes with the `vlm` extra only, so the tests that need a REAL clip are skipped
when `av` is missing (the CPU profile of CI). Everything that can be covered without a
decoder — the extension check, the env switches, the frame-choice/rotation arithmetic,
the semaphore, the missing-package fallback and the seek/decode loop against a fake av
— runs everywhere.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import threading
import time
import unittest
import unittest.mock
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from sorta import imaging
from tests.test_imaging_preview import PreviewCacheTestCase, make_photo, stat_key

HAVE_AV = importlib.util.find_spec("av") is not None
BLACK = (0, 0, 0)
RED = (200, 30, 30)

# Small on purpose: the whole point is a fast synthetic clip. Larger than the preview
# edge below, so the stored preview is really downscaled to it.
VIDEO_SIZE = (640, 360)
PREVIEW_EDGE = 256


def make_video(
    path: Path, colors=(BLACK, RED, RED, RED, RED, RED, RED, RED, RED, RED),
    size=VIDEO_SIZE, fps: int = 10, rotation: int | None = None,
) -> None:
    """A clip of solid-colour frames; `rotation` sets the container display matrix."""
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = size
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width, stream.height = width, height
        stream.pix_fmt = "yuv420p"
        if rotation is not None:
            stream.set_display_rotation(rotation)
        for color in colors:
            arr = np.zeros((height, width, 3), dtype=np.uint8)
            arr[:, :] = color
            for packet in stream.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


class FakeFrame:
    def __init__(self, when: float | None, color=RED, rotation: int = 0):
        self.time = when
        self.rotation = rotation
        self._color = color

    def to_image(self) -> Image.Image:
        return Image.new("RGB", (16, 8), self._color)


class FakeContainer:
    def __init__(self, frames, *, duration=1_000_000, seek_error=False):
        self.frames = frames
        self.duration = duration
        self.seek_error = seek_error
        self.seeks: list[int] = []
        self.decoded = 0
        self.closed = False
        self.stream = SimpleNamespace(
            duration=None, time_base=Fraction(1, 1000), metadata={}, thread_type=None)
        self.streams = SimpleNamespace(video=[self.stream])

    def __enter__(self) -> FakeContainer:
        return self

    def __exit__(self, *exc) -> bool:
        self.closed = True
        return False

    def seek(self, offset, **kwargs) -> None:
        if self.seek_error:
            raise RuntimeError("this container cannot seek")
        self.seeks.append(offset)

    def decode(self, stream):
        for frame in self.frames:
            self.decoded += 1
            yield frame


def fake_av(container: FakeContainer) -> SimpleNamespace:
    """The narrow slice of the av module that _grab_frame actually uses."""
    return SimpleNamespace(open=lambda path: container)


class TestIsVideoPath(unittest.TestCase):
    def test_recognizes_video_extensions_case_insensitively(self):
        for ext in ("mp4", "mov", "avi", "mkv", "mts", "m2ts", "3gp"):
            for name in (f"clip.{ext}", f"clip.{ext.upper()}", f"CLIP.{ext.capitalize()}"):
                self.assertTrue(imaging.is_video_path(name), name)
        self.assertTrue(imaging.is_video_path(Path(r"C:\photos\2020\IMG_0001.MOV")))

    def test_photos_are_not_video(self):
        for name in ("a.jpg", "a.jpeg", "a.heic", "a.png", "a.webp", "a.tif", "a.CR2", "a"):
            self.assertFalse(imaging.is_video_path(name), name)


class TestVideoSettings(unittest.TestCase):
    def test_enabled_by_default_and_off_switches(self):
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            self.assertTrue(imaging.video_previews_enabled())
        for value in ("0", "false", "NO", "off"):
            with unittest.mock.patch.dict(os.environ, {imaging.ENV_VIDEO_PREVIEWS: value}):
                self.assertFalse(imaging.video_previews_enabled())

    def test_workers_env_and_fallback(self):
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(imaging.video_workers(), imaging.VIDEO_WORKERS)
        for value, expected in (("2", 2), ("nope", imaging.VIDEO_WORKERS),
                                ("0", imaging.VIDEO_WORKERS)):
            with unittest.mock.patch.dict(os.environ, {imaging.ENV_VIDEO_WORKERS: value}):
                self.assertEqual(imaging.video_workers(), expected)


class TestFrameChoice(unittest.TestCase):
    """The arithmetic behind "which frame" and "which way up" — no decoder needed."""

    def test_short_clip_takes_a_tenth_of_its_duration(self):
        stream = SimpleNamespace(duration=2000, time_base=Fraction(1, 1000))
        self.assertAlmostEqual(_target(stream=stream), 0.2)

    def test_long_clip_takes_one_second(self):
        stream = SimpleNamespace(duration=60_000, time_base=Fraction(1, 1000))
        self.assertAlmostEqual(_target(stream=stream), imaging.VIDEO_FRAME_SECONDS)

    def test_falls_back_to_the_container_duration(self):
        stream = SimpleNamespace(duration=None, time_base=Fraction(1, 1000))
        self.assertAlmostEqual(_target(stream=stream, container_duration=2_000_000), 0.2)

    def test_unknown_duration_takes_one_second(self):
        stream = SimpleNamespace(duration=None, time_base=None)
        self.assertAlmostEqual(
            _target(stream=stream, container_duration=None), imaging.VIDEO_FRAME_SECONDS)
        stream = SimpleNamespace(duration=0, time_base=Fraction(1, 1000))
        self.assertAlmostEqual(_target(stream=stream), imaging.VIDEO_FRAME_SECONDS)

    def test_rotation_from_the_display_matrix_wins(self):
        frame = SimpleNamespace(rotation=-90)
        self.assertEqual(imaging._frame_rotation(frame, SimpleNamespace(metadata={})), -90)

    def test_rotation_falls_back_to_the_legacy_metadata_tag(self):
        frame = SimpleNamespace(rotation=0)
        stream = SimpleNamespace(metadata={"rotate": "90"})  # clockwise by convention
        self.assertEqual(imaging._frame_rotation(frame, stream), -90)

    def test_missing_or_broken_rotation_is_zero(self):
        self.assertEqual(
            imaging._frame_rotation(SimpleNamespace(), SimpleNamespace(metadata=None)), 0)
        self.assertEqual(
            imaging._frame_rotation(
                SimpleNamespace(rotation=None), SimpleNamespace(metadata={"rotate": "??"})), 0)

    def test_rotate_frame_keeps_an_unrotated_image_as_is(self):
        img = Image.new("RGB", (16, 8), RED)
        self.assertIs(imaging._rotate_frame(img, 0), img)
        self.assertIs(imaging._rotate_frame(img, 360), img)
        self.assertEqual(imaging._rotate_frame(img, 90).size, (8, 16))


def _target(*, stream, container_duration=1_000_000) -> float:
    return imaging._target_seconds(SimpleNamespace(duration=container_duration), stream)


class TestGrabFrameAgainstFakeAv(unittest.TestCase):
    """The seek/decode loop itself, without a real decoder (CI has no av)."""

    def test_decodes_forward_to_the_target_and_closes_the_container(self):
        frames = [FakeFrame(0.0, BLACK), FakeFrame(0.1, RED), FakeFrame(0.2, BLACK)]
        container = FakeContainer(frames)
        img = imaging._grab_frame(fake_av(container), "clip.mp4")
        self.assertIsNotNone(img)
        self.assertEqual(img.getpixel((0, 0)), RED)  # the frame at the target time
        self.assertTrue(container.closed)
        self.assertEqual(container.seeks, [100])  # 0.1 s in the stream time base

    def test_a_container_that_cannot_seek_falls_back_to_the_first_frame(self):
        container = FakeContainer([FakeFrame(0.0, RED)], seek_error=True)
        img = imaging._grab_frame(fake_av(container), "clip.mp4")
        self.assertIsNotNone(img)
        self.assertTrue(container.closed)

    def test_frames_without_timestamps_take_the_first_one(self):
        container = FakeContainer([FakeFrame(None, RED), FakeFrame(None, BLACK)])
        self.assertEqual(
            imaging._grab_frame(fake_av(container), "clip.mp4").getpixel((0, 0)), RED)

    def test_decoding_stops_at_the_frame_cap(self):
        """Timestamps that never reach the target must not drag in the whole clip."""
        frames = [FakeFrame(0.0, RED) for _ in range(imaging._VIDEO_MAX_DECODED_FRAMES + 50)]
        container = FakeContainer(frames)
        self.assertIsNotNone(imaging._grab_frame(fake_av(container), "clip.mp4"))
        self.assertEqual(container.decoded, imaging._VIDEO_MAX_DECODED_FRAMES)

    def test_an_empty_stream_gives_none(self):
        self.assertIsNone(imaging._grab_frame(fake_av(FakeContainer([])), "clip.mp4"))

    def test_rotation_is_applied_to_the_extracted_frame(self):
        container = FakeContainer([FakeFrame(0.1, RED, rotation=90)])
        img = imaging._grab_frame(fake_av(container), "clip.mp4")
        self.assertEqual(img.size, (8, 16))  # 16x8 landscape -> portrait


class TestMissingAv(unittest.TestCase):
    def test_without_the_package_the_call_returns_none_and_warns_once(self):
        with unittest.mock.patch.dict(sys.modules, {"av": None}), \
                unittest.mock.patch.object(imaging, "_av_warned", False), \
                self.assertLogs("sorta.imaging", level="WARNING") as logs:
            self.assertIsNone(imaging._extract_video_frame("clip.mp4"))
            self.assertIsNone(imaging._extract_video_frame("other.mp4"))
        self.assertEqual(len(logs.records), 1)  # one warning, not one per file

    def test_decode_rgb_preview_degrades_instead_of_raising(self):
        with unittest.mock.patch.dict(sys.modules, {"av": None}), \
                unittest.mock.patch.object(imaging, "_av_warned", True):
            self.assertIsNone(imaging.decode_rgb_preview("clip.mp4", 1.0, 10, max_edge=96))


class TestVideoSemaphore(unittest.TestCase):
    def test_never_more_extractions_at_once_than_configured(self):
        slots = 2
        running = 0
        peak = 0
        lock = threading.Lock()

        def slow_grab(av, path):
            nonlocal running, peak
            with lock:
                running += 1
                peak = max(peak, running)
            time.sleep(0.02)
            with lock:
                running -= 1
            return Image.new("RGB", (8, 8), RED)

        env = {imaging.ENV_VIDEO_WORKERS: str(slots)}
        with unittest.mock.patch.dict(os.environ, env), \
                unittest.mock.patch.object(imaging, "_import_av", lambda: SimpleNamespace()), \
                unittest.mock.patch.object(imaging, "_grab_frame", slow_grab):
            with ThreadPoolExecutor(max_workers=12) as pool:
                results = list(pool.map(
                    lambda i: imaging._extract_video_frame(f"clip{i}.mp4"), range(12)))

        self.assertEqual(len(results), 12)
        self.assertTrue(all(img is not None for img in results))
        self.assertLessEqual(peak, slots)
        self.assertGreater(peak, 1)  # the gate limits, it does not serialize

    def test_the_photo_path_is_never_gated(self):
        """decode_rgb_preview on a photo must not even build the video semaphore."""
        with unittest.mock.patch.object(imaging, "_video_gate") as gate:
            with unittest.mock.patch.dict(os.environ, {imaging.ENV_PREVIEW_CACHE: "0"}):
                imaging.decode_rgb_preview("missing.jpg", 1.0, 10, max_edge=96)
        gate.assert_not_called()


class VideoPreviewTestCase(PreviewCacheTestCase):
    """A tmp preview cache + a small preview edge, so a 640x360 clip is downscaled."""

    env = {
        imaging.ENV_VIDEO_PREVIEWS: "1",
        imaging.ENV_PREVIEW_MAX_EDGE: str(PREVIEW_EDGE),
    }


@unittest.skipUnless(HAVE_AV, "PyAV is not installed (it ships with the `vlm` extra)")
class TestVideoPreview(VideoPreviewTestCase):
    def test_creates_one_preview_bounded_by_preview_max_edge(self):
        src = self.root / "clip.mp4"
        make_video(src)
        img = imaging.decode_rgb_preview(src, *stat_key(src), max_edge=128)

        self.assertIsNotNone(img)
        self.assertEqual(max(img.size), 128)
        self.assertEqual(len(self.previews()), 1)
        with Image.open(self.previews()[0]) as stored:
            self.assertEqual(max(stored.size), imaging.preview_max_edge())
            self.assertEqual(stored.mode, "RGB")

    def test_the_preview_lands_on_the_same_key_as_a_photo_would(self):
        src = self.root / "clip.mp4"
        make_video(src)
        mtime, size = stat_key(src)
        imaging.decode_rgb_preview(src, mtime, size, max_edge=128)
        key = imaging.preview_key(src, mtime, size)
        self.assertTrue((self.cache / key[:2] / f"{key}.jpg").is_file())

    def test_second_call_is_served_from_the_cache_without_opening_the_container(self):
        import av

        src = self.root / "clip.mp4"
        make_video(src)
        mtime, size = stat_key(src)
        opened: list[str] = []
        real_open = av.open

        def counting_open(path, *args, **kwargs):
            opened.append(str(path))
            return real_open(path, *args, **kwargs)

        with unittest.mock.patch.object(av, "open", counting_open):
            first = imaging.decode_rgb_preview(src, mtime, size, max_edge=128)
            second = imaging.decode_rgb_preview(src, mtime, size, max_edge=128)

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(first.size, second.size)
        self.assertEqual(opened, [str(src)])  # exactly one open, for the cold call
        self.assertEqual(len(self.previews()), 1)

    def test_a_black_first_frame_does_not_become_the_preview(self):
        src = self.root / "fade-in.mp4"
        make_video(src)  # frame 0 black, the rest red
        img = imaging.decode_rgb_preview(src, *stat_key(src), max_edge=128)
        self.assertIsNotNone(img)
        self.assertGreater(float(np.asarray(img).mean()), 40.0)  # black would be ~0

    def test_container_rotation_is_applied(self):
        src = self.root / "portrait.mp4"
        make_video(src, rotation=90)
        img = imaging.decode_rgb_preview(src, *stat_key(src), max_edge=128)
        self.assertIsNotNone(img)
        # the source is 640x360 landscape — with the matrix applied it stands portrait
        self.assertLess(img.size[0], img.size[1])
        with Image.open(self.previews()[0]) as stored:
            self.assertLess(stored.size[0], stored.size[1])

    def test_a_clip_without_rotation_keeps_its_shape(self):
        src = self.root / "landscape.mp4"
        make_video(src)
        img = imaging.decode_rgb_preview(src, *stat_key(src), max_edge=128)
        self.assertGreater(img.size[0], img.size[1])

    def test_corrupt_file_returns_none_without_raising(self):
        src = self.root / "broken.mp4"
        src.write_bytes(b"\x00\x01garbage that is definitely not a container" * 50)
        self.assertIsNone(imaging.decode_rgb_preview(src, *stat_key(src), max_edge=128))
        self.assertEqual(self.previews(), [])

    def test_truncated_file_returns_none(self):
        src = self.root / "trunc.mp4"
        make_video(src)
        data = src.read_bytes()
        src.write_bytes(data[:len(data) // 4])
        self.assertIsNone(imaging.decode_rgb_preview(src, *stat_key(src), max_edge=128))
        self.assertEqual(self.previews(), [])

    def test_missing_file_returns_none(self):
        self.assertIsNone(
            imaging.decode_rgb_preview(self.root / "nope.mp4", 1.0, 10, max_edge=128))

    def test_file_without_a_video_stream_returns_none(self):
        import av

        src = self.root / "audio.mp4"
        try:
            with av.open(str(src), mode="w") as container:
                stream = container.add_stream("aac", rate=44100)
                stream.layout = "mono"
                samples = (np.sin(np.arange(44100) * 0.05) * 10000).astype("int16")
                frame = av.AudioFrame.from_ndarray(samples.reshape(1, -1),
                                                   format="s16", layout="mono")
                frame.sample_rate = 44100
                for packet in stream.encode(frame):
                    container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
        except Exception as exc:  # no aac encoder in this FFmpeg build
            self.skipTest(f"cannot generate an audio-only file: {exc}")
        self.assertIsNone(imaging.decode_rgb_preview(src, *stat_key(src), max_edge=128))
        self.assertEqual(self.previews(), [])

    def test_repeated_failures_do_not_leak_descriptors(self):
        """Broken clips are the norm on a real collection — the process must survive.

        Every extraction opens a container; without `with av.open(...)` a few hundred
        failures already exhaust the descriptors of the process.
        """
        src = self.root / "broken.mp4"
        src.write_bytes(b"not a container at all" * 100)
        mtime, size = stat_key(src)
        for _ in range(300):
            self.assertIsNone(imaging.decode_rgb_preview(src, mtime, size, max_edge=128))

    def test_a_cache_write_failure_still_returns_the_frame(self):
        src = self.root / "clip.mp4"
        make_video(src)

        def failing_save(*args, **kwargs):
            raise OSError("disk full")

        with unittest.mock.patch.object(Image.Image, "save", failing_save):
            img = imaging.decode_rgb_preview(src, *stat_key(src), max_edge=128)
        self.assertIsNotNone(img)
        self.assertEqual(max(img.size), 128)
        self.assertEqual(self.previews(), [])

    def test_grayscale_and_orientation_options_behave_as_for_photos(self):
        src = self.root / "clip.mp4"
        make_video(src)
        mtime, size = stat_key(src)
        cold = imaging.decode_rgb_preview(
            src, mtime, size, max_edge=96, grayscale=True, apply_orientation=True)
        warm = imaging.decode_rgb_preview(
            src, mtime, size, max_edge=96, grayscale=True, apply_orientation=True)
        for img in (cold, warm):
            self.assertIsNotNone(img)
            self.assertEqual(img.mode, "L")
            self.assertEqual(max(img.size), 96)
        self.assertEqual(cold.tobytes(), warm.tobytes())


@unittest.skipUnless(HAVE_AV, "PyAV is not installed (it ships with the `vlm` extra)")
class TestVideoPreviewsDisabled(VideoPreviewTestCase):
    env = {
        imaging.ENV_VIDEO_PREVIEWS: "0",
        imaging.ENV_PREVIEW_MAX_EDGE: str(PREVIEW_EDGE),
    }

    def test_returns_none_and_writes_nothing(self):
        src = self.root / "clip.mp4"
        make_video(src)
        self.assertIsNone(imaging.decode_rgb_preview(src, *stat_key(src), max_edge=128))
        self.assertEqual(self.previews(), [])
        self.assertFalse(self.cache.exists())


@unittest.skipUnless(HAVE_AV, "PyAV is not installed (it ships with the `vlm` extra)")
class TestVideoWithoutTheDiskCache(VideoPreviewTestCase):
    env = {
        imaging.ENV_VIDEO_PREVIEWS: "1",
        imaging.ENV_PREVIEW_CACHE: "0",
        imaging.ENV_PREVIEW_MAX_EDGE: str(PREVIEW_EDGE),
    }

    def test_frame_is_still_returned_but_nothing_is_stored(self):
        src = self.root / "clip.mp4"
        make_video(src)
        img = imaging.decode_rgb_preview(src, *stat_key(src), max_edge=128)
        self.assertIsNotNone(img)
        self.assertEqual(max(img.size), 128)
        self.assertEqual(self.previews(), [])
        self.assertFalse(self.cache.exists())


class TestPhotoPathUntouched(PreviewCacheTestCase):
    """F74 must be invisible to photos — the regression insurance of the feature."""

    def test_a_photo_never_reaches_the_video_extractor(self):
        src = self.root / "a.jpg"
        make_photo(src)

        def explode(path):
            raise AssertionError(f"the video extractor was called for a photo: {path}")

        with unittest.mock.patch.object(imaging, "_extract_video_frame", explode):
            img = imaging.decode_rgb_preview(src, *stat_key(src), max_edge=96)
        self.assertIsNotNone(img)
        self.assertEqual(len(self.previews()), 1)

    def test_decode_rgb_stays_image_only(self):
        """decode_rgb is used by the sorter thumbnails — F74 must not widen it."""
        src = self.root / "clip.mp4"
        src.write_bytes(b"whatever, decode_rgb does not decode video")
        self.assertIsNone(imaging.decode_rgb(src, 96))


if __name__ == "__main__":
    unittest.main()
