"""F88: the detector input size is pinned instead of left to insightface.

`app.prepare()` used to be called without `det_size`, which leaves insightface 1.0.1 in
its two-pass mode: on EVERY frame the network runs at 128x128 and then at 640x640, and
both passes cost the same ~78 ms (the price is the input-shape switch, not the
arithmetic). Measured on 100 real frames: 165.1 ms/frame with two passes against 16.5
with one, 56 faces against 57.

The tests below are about the one line that caused it — that `det_size` is passed at
all, that the config can move it, and that a garbage value in config.yaml falls back to
the default instead of killing an hour-long run from inside a worker thread. The model
is faked (a stub `insightface.app` module); what is verified is the call, not the GPU.
"""
from __future__ import annotations

import logging
import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from unittest import mock

import numpy as np

from sorta.config import Config, FacesConfig
from sorta.faces import (
    DET_SIZE_DEFAULT,
    EMBED_DIM,
    FacesSettings,
    _det_size,
    _insightface_infer,
    _settings,
)


def cfg_with(faces_raw: dict | None = None, **faces_fields) -> Config:
    """A Config whose faces section is exactly what a config.yaml would give."""
    return Config(
        sources=[Path("/photos")],
        faces=FacesConfig(**faces_fields),
        raw={"faces": faces_raw} if faces_raw is not None else {},
    )


class FakeFace:
    def __init__(self) -> None:
        self.bbox = np.array([10.0, 20.0, 110.0, 140.0], dtype=np.float32)
        self.det_score = np.float32(0.93)
        self.embedding = np.ones(EMBED_DIM, dtype=np.float32)


class FakeFaceAnalysis:
    """Records how the session was built and prepared; `get` returns one face."""

    built: list[dict] = []
    prepared: list[dict] = []

    def __init__(self, **kwargs) -> None:
        self.built.append(kwargs)

    def prepare(self, **kwargs) -> None:
        self.prepared.append(kwargs)

    def get(self, img: np.ndarray) -> list[FakeFace]:
        return [FakeFace()]


@contextmanager
def fake_insightface() -> Iterator[type[FakeFaceAnalysis]]:
    """Stand in for the insightface package (imported inside `_insightface_infer`)."""
    FakeFaceAnalysis.built = []
    FakeFaceAnalysis.prepared = []
    app_mod = types.ModuleType("insightface.app")
    app_mod.FaceAnalysis = FakeFaceAnalysis  # type: ignore[attr-defined]
    root_mod = types.ModuleType("insightface")
    root_mod.app = app_mod  # type: ignore[attr-defined]
    modules = {"insightface": root_mod, "insightface.app": app_mod}
    # _enable_cuda_dll_dirs walks site-packages and edits PATH — not this test's subject
    with mock.patch.dict(sys.modules, modules), \
            mock.patch("sorta.faces._enable_cuda_dll_dirs", lambda: None):
        yield FakeFaceAnalysis


class PrepareGetsDetSizeTest(unittest.TestCase):
    """The regression itself: `prepare` is called WITH `det_size`."""

    def prepare_kwargs(self, s: FacesSettings) -> dict:
        with fake_insightface() as analysis:
            _insightface_infer(s)
        self.assertEqual(len(analysis.prepared), 1, "one prepare per session")
        return analysis.prepared[0]

    def test_prepare_receives_an_explicit_det_size(self):
        kwargs = self.prepare_kwargs(FacesSettings())
        self.assertIn("det_size", kwargs, "without det_size the detector runs twice per frame")
        self.assertEqual(kwargs["det_size"], (DET_SIZE_DEFAULT, DET_SIZE_DEFAULT))

    def test_det_size_is_one_square_shape(self):
        # the point of the fix is a single, stable input shape — a pair of sizes
        # (what insightface picks by itself) is what cost 10x
        w, h = self.prepare_kwargs(FacesSettings(det_size=512))["det_size"]
        self.assertEqual((w, h), (512, 512))

    def test_the_settings_value_reaches_prepare(self):
        kwargs = self.prepare_kwargs(FacesSettings(det_size=320))
        self.assertEqual(kwargs["det_size"], (320, 320))

    def test_det_thresh_and_the_session_are_untouched(self):
        with fake_insightface() as analysis:
            _insightface_infer(FacesSettings(det_threshold=0.55))
        self.assertEqual(analysis.prepared[0]["det_thresh"], 0.55)
        self.assertEqual(analysis.prepared[0]["ctx_id"], 0)
        self.assertEqual(analysis.built[0]["name"], "buffalo_l")
        self.assertEqual(analysis.built[0]["allowed_modules"], ["detection", "recognition"])

    def test_infer_still_returns_bbox_score_embedding(self):
        with fake_insightface():
            infer = _insightface_infer(FacesSettings())
        hits = infer(np.zeros((4, 4, 3), dtype=np.uint8))
        (bbox, score, emb), = hits
        self.assertEqual(bbox, [10.0, 20.0, 110.0, 140.0])
        self.assertAlmostEqual(score, 0.93, places=5)
        self.assertEqual(np.asarray(emb).shape, (EMBED_DIM,))


class DetSizeFromConfigTest(unittest.TestCase):
    """`faces.det_size` is a raw-section key, like decode_workers/infer_workers."""

    def test_default_is_the_native_640(self):
        self.assertEqual(DET_SIZE_DEFAULT, 640)
        self.assertEqual(FacesSettings().det_size, 640)
        self.assertEqual(_det_size(cfg_with()), 640)

    def test_absent_key_in_a_present_faces_section(self):
        self.assertEqual(_det_size(cfg_with({"decode_workers": 8})), 640)

    def test_value_comes_from_the_config(self):
        self.assertEqual(_det_size(cfg_with({"det_size": 512})), 512)

    def test_a_string_from_yaml_is_accepted(self):
        self.assertEqual(_det_size(cfg_with({"det_size": "512"})), 512)

    def test_settings_carry_it_alongside_the_thresholds(self):
        s = _settings(cfg_with({"det_size": 800}, min_face_px=60, det_threshold=0.8))
        self.assertEqual((s.det_size, s.min_face_px, s.det_threshold), (800, 60, 0.8))

    def test_the_configured_value_ends_up_in_prepare(self):
        with fake_insightface() as analysis:
            _insightface_infer(_settings(cfg_with({"det_size": 512})))
        self.assertEqual(analysis.prepared[0]["det_size"], (512, 512))


class BadDetSizeFallsBackTest(unittest.TestCase):
    """A typo in config.yaml must not crash an hour into a run."""

    BAD = [0, -640, "abc", "", [640, 640], {}, 3.4j]

    def test_bad_values_fall_back_to_the_default_with_a_warning(self):
        for value in self.BAD:
            with self.subTest(value=value):
                with self.assertLogs(level=logging.WARNING) as logs:
                    got = _det_size(cfg_with({"det_size": value}))
                self.assertEqual(got, DET_SIZE_DEFAULT)
                self.assertIn("det_size", "".join(logs.output))

    def test_a_bad_value_still_produces_a_usable_session(self):
        with fake_insightface() as analysis, self.assertLogs(level=logging.WARNING):
            infer = _insightface_infer(_settings(cfg_with({"det_size": 0})))
        self.assertEqual(analysis.prepared[0]["det_size"], (640, 640))
        self.assertEqual(len(infer(np.zeros((4, 4, 3), dtype=np.uint8))), 1)

    def test_an_unset_key_does_not_warn(self):
        with self.assertNoLogs(level=logging.WARNING):
            self.assertEqual(_det_size(cfg_with({"decode_workers": 8})), 640)
            self.assertEqual(_det_size(cfg_with({"det_size": 640})), 640)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
