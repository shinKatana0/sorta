"""F102: `vlm.max_edge` has to reach the frame, not just be readable in the config.

A knob that loads and then quietly gets lost between the config and the decode is worse
than a constant: the constant at least tells the truth. So these tests follow the number
the whole way — cfg.vlm.max_edge -> the factory classify() builds -> the max_edge the
frame is actually decoded at — with the model faked away at the last step, since the
value is handed to imaging.decode_rgb_preview long before any weights are involved.

The second thing pinned here is the regression the brief asks for: a default config
must produce the same call it produced before this feature existed, at 896.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from sorta import imaging, junk
from sorta.config import DEFAULT_VLM_MAX_EDGE, Config, VlmConfig, _naming_from
from sorta.db import connect
from sorta.junk import classify, qwen_vlm_classifier_factory, vlm_classifier_from
from sorta.naming import SplitVlm
from tests.test_junk import NO_OCR, _RECEIPT_IDX, FakeClassifier

CANDIDATE_DOC_SCORE = 0.5  # inside the candidate zone — the fast tier hands it to the VLM


class RecordingDecode:
    """imaging.decode_rgb_preview replaced by a recorder of the size it was asked for."""

    def __init__(self, image: Image.Image | None = None):
        self.image = image if image is not None else Image.new("RGB", (8, 8))
        self.max_edges: list[int] = []

    def __call__(self, path, mtime, size, max_edge):
        self.max_edges.append(max_edge)
        return self.image


class DecodeCase(unittest.TestCase):
    """Every test here swaps the decode out — no model, no real image, one number.

    The file itself still has to exist: the classifier stats it first and answers
    `personal_photo` without asking anything when it does not (a vanished frame is not
    a question for the model).
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.frame = Path(self.tmp.name) / "frame.jpg"
        self.frame.write_bytes(b"not decoded here")
        self.decode = RecordingDecode()
        original = imaging.decode_rgb_preview
        imaging.decode_rgb_preview = self.decode  # type: ignore[assignment]
        self.addCleanup(setattr, imaging, "decode_rgb_preview", original)


class TestMaxEdgeReachesTheFrame(DecodeCase):
    """Test 6: the configured size is what the frame is decoded at."""

    def runtime(self):
        """A split runtime that answers `document` and remembers nothing else."""
        return SplitVlm(prepare=lambda frames, prompt: ("inputs", len(frames)),
                        generate=lambda prepared, max_new_tokens: "document")

    def test_the_serial_classifier_decodes_at_the_configured_size(self):
        def describe(frames, prompt, max_new_tokens):
            return "document"

        classifier = vlm_classifier_from(describe, max_edge=448)
        self.assertEqual(classifier(str(self.frame)), "document")
        self.assertEqual(self.decode.max_edges, [448])

    def test_the_split_classifier_decodes_at_the_configured_size(self):
        """The pipelined path prepares on other threads — the size must travel with it."""
        classifier = vlm_classifier_from(self.runtime(), max_edge=672)
        self.assertEqual(classifier(str(self.frame)), "document")
        self.assertEqual(self.decode.max_edges, [672])

    def test_the_default_is_the_896_that_shipped(self):
        vlm_classifier_from(self.runtime())(str(self.frame))
        self.assertEqual(self.decode.max_edges, [DEFAULT_VLM_MAX_EDGE])
        self.assertEqual(self.decode.max_edges, [896])

    def test_the_default_factory_carries_the_size_to_the_real_builder(self):
        """The factory interface stays (model_name) -> classifier; max_edge rides along."""
        seen: list[tuple[str, int]] = []
        original = junk.qwen_vlm_classifier
        junk.qwen_vlm_classifier = (  # type: ignore[assignment]
            lambda model_name, max_edge: seen.append((model_name, max_edge)))
        self.addCleanup(setattr, junk, "qwen_vlm_classifier", original)
        qwen_vlm_classifier_factory(448)("Qwen/some-model")
        self.assertEqual(seen, [("Qwen/some-model", 448)])


class TestClassifyPassesTheConfiguredRuntime(unittest.TestCase):
    """From cfg.vlm through classify() into the build of the real classifier."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.built: list[tuple[str, int]] = []
        original = junk.qwen_vlm_classifier
        junk.qwen_vlm_classifier = self._build  # type: ignore[assignment]
        self.addCleanup(setattr, junk, "qwen_vlm_classifier", original)

    def _build(self, model_name, max_edge):
        self.built.append((model_name, max_edge))
        return lambda path: "document"

    def cfg_with(self, vlm: VlmConfig) -> Config:
        return Config(sources=[Path(self.tmp.name)],
                      database=Path(self.tmp.name) / f"{len(self.built)}.db",
                      naming=_naming_from({}), vlm=vlm)

    def run_classify(self, vlm: VlmConfig):
        cfg = self.cfg_with(vlm)
        # The toggle lives on cfg.naming — that is the field --deep and the UI replace.
        object.__setattr__(cfg.naming, "vlm_enabled", vlm.enabled)
        conn = connect(cfg.database)
        self.addCleanup(conn.close)
        conn.execute(
            """INSERT INTO files (path, size, mtime, ext, media_type, width, height,
                   camera_make, camera_model, gps_lat, indexed_at)
               VALUES ('/photos/cand.jpg', 1000, 0, 'jpg', 'photo', 4000, 3000,
                       'Canon', 'EOS', NULL, '2026-01-01')""")
        conn.commit()
        clf = FakeClassifier({}, doc_scores={"cand.jpg": (_RECEIPT_IDX,
                                                          CANDIDATE_DOC_SCORE)})
        return classify(cfg, conn, classifier=clf, text_detector=NO_OCR)

    def test_the_configured_model_and_size_are_what_gets_built(self):
        self.run_classify(VlmConfig(enabled=True, model="Qwen/other", max_edge=448))
        self.assertEqual(self.built, [("Qwen/other", 448)])

    def test_a_default_config_builds_the_896_path_it_always_did(self):
        """Test 7 (regression): nothing about a config that says nothing has moved."""
        self.run_classify(VlmConfig(enabled=True))
        self.assertEqual(self.built, [(VlmConfig().model, 896)])

    def test_nothing_is_built_while_the_tier_is_off(self):
        self.run_classify(VlmConfig(max_edge=448))
        self.assertEqual(self.built, [])


if __name__ == "__main__":
    unittest.main()
