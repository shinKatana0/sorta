"""F187: the loader that gives F183 a candidate to run — and does not flatter it.

The measurement was merged with nothing to run it on: its only way in was
`AutoModelForImageToImage`, which accepts the Swin2SR family alone, where every published
weight enlarges. This loader adds one more way in — an ONNX file on disk — and everything
checked here is about the two ways it could quietly ruin the measurement it serves:

* THE CHECKS OF F183 MUST STILL BITE. A model that enlarges answers F169's question and a
  model that returns its input untouched is a null result wearing the costume of "did no
  harm". A loader that resized results to match its input would turn every x4 model into a
  passing 1:1 candidate, so the tests below run real graphs of both kinds through the
  loader and then through `probe_one_to_one`, which is the check that has to reject them;
* A SMALL PICTURE MUST NOT BE REFUSED FOR BEING SMALL. The exported graph will not run
  under 369 px a side, and the probe picture is 256x192 — without padding the candidate
  would be thrown out for the shape of the probe rather than for what it does.

No network and no weights: the graphs are built here with `onnx.helper` (a multiply, a
resize, an identity — a few hundred bytes each) and the padding is checked against a
session stub that records what it was shown. One class below closes the sockets outright
while a candidate is loaded and run, so "nothing goes to the network" is a fact about the
code rather than a promise about it.
"""
from __future__ import annotations

import importlib.util
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"

try:
    import onnx
    from onnx import TensorProto, helper
except ImportError:  # pragma: no cover — onnx arrives with insightface in both profiles
    onnx = None


def _load_script(name: str):
    """Import a file from scripts/ — they are scripts, not package modules."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


dc = _load_script("deblur_candidate")
md = _load_script("measure_deblur")


def noise(size=(300, 200), seed: int = 1) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8),
                           "RGB")


def _graph(path: Path, nodes, initializers=()) -> str:
    """Write a one-input, one-output ONNX graph with dynamic height and width.

    Dynamic on purpose: the candidate's own export is (batch, 3, height, width), and a
    fixture pinned to one size would test a loader nobody runs.
    """
    shape = [None, 3, None, None]
    graph = helper.make_graph(
        list(nodes), "candidate",
        [helper.make_tensor_value_info("lq", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)],
        list(initializers))
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return str(path)


def darkening(path: Path) -> str:
    """A 1:1 candidate that really moves the pixels — the shape a usable model has."""
    half = helper.make_tensor("half", TensorProto.FLOAT, [1], [0.5])
    return _graph(path, [helper.make_node("Mul", ["lq", "half"], ["output"])], [half])


def enlarging(path: Path) -> str:
    """A candidate that doubles the frame — F169's question, not this one."""
    scales = helper.make_tensor("scales", TensorProto.FLOAT, [4], [1.0, 1.0, 2.0, 2.0])
    return _graph(path, [helper.make_node("Resize", ["lq", "", "scales"], ["output"],
                                          mode="nearest")], [scales])


def untouched(path: Path) -> str:
    """A candidate that hands the frame back exactly as it got it."""
    return _graph(path, [helper.make_node("Identity", ["lq"], ["output"])])


class Session:
    """A stand-in for an onnxruntime session: it records what it was shown.

    Enough of the interface to be indistinguishable to the loader — a named input and a
    `run` that returns a list of NCHW arrays — and nothing else, because what the tests
    that use it are about is the arithmetic around the call, not the runtime.
    """

    def __init__(self, transform=None, name: str = "lq"):
        self.name = name
        self.shown: list[np.ndarray] = []
        self.transform = transform or (lambda array: array)

    def get_inputs(self):
        return [SimpleNamespace(name=self.name)]

    def run(self, _outputs, feed):
        array = feed[self.name]
        self.shown.append(np.array(array, copy=True))
        return [self.transform(array)]


@unittest.skipIf(onnx is None, "onnx is not installed")
class TestTheCandidateRunsAndTheMeasurementAcceptsIt(unittest.TestCase):
    """Brief item 1: `measure_deblur.py` needs ONE working 1:1 candidate, no more."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_the_loader_returns_a_function_that_gives_back_the_size_it_was_given(self):
        process = dc.load_onnx_restorer(darkening(self.root / "cand.onnx"),
                                        providers=("CPUExecutionProvider",))
        picture = noise((300, 200))

        result = process(picture)

        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, picture.size)
        self.assertEqual(result.mode, "RGB")

    def test_the_probe_of_f183_accepts_it_which_is_the_whole_point_of_the_feature(self):
        process = dc.load_onnx_restorer(darkening(self.root / "cand.onnx"),
                                        providers=("CPUExecutionProvider",))

        probe = md.probe_one_to_one(process)

        self.assertTrue(probe.usable, probe.reason)
        self.assertAlmostEqual(probe.scale, 1.0)
        self.assertGreater(probe.changed, md.PROBE_MIN_CHANGE)

    def test_a_candidate_that_enlarges_is_still_rejected_and_not_resized_into_shape(self):
        """The check that separates this feature's question from F169's. A loader that
        cropped or scaled the result to its input would disarm it silently — and so would
        one whose own padding distorted the multiplier the probe then reports."""
        process = dc.load_onnx_restorer(enlarging(self.root / "big.onnx"),
                                        providers=("CPUExecutionProvider",))

        self.assertEqual(process(noise((640, 480))).size, (1280, 960))   # nothing padded
        self.assertEqual(process(noise((300, 200))).size, (600, 400))    # padded and cropped
        probe = md.probe_one_to_one(process)
        self.assertFalse(probe.usable)
        self.assertAlmostEqual(probe.scale, 2.0)
        self.assertIn("один к одному", probe.reason)

    def test_a_candidate_that_returns_its_input_untouched_is_still_rejected(self):
        process = dc.load_onnx_restorer(untouched(self.root / "same.onnx"),
                                        providers=("CPUExecutionProvider",))

        probe = md.probe_one_to_one(process)

        self.assertFalse(probe.usable)
        self.assertEqual(probe.changed, 0.0)
        self.assertIn("без изменений", probe.reason)

    def test_a_frame_below_the_graph_s_minimum_is_padded_rather_than_refused(self):
        """The export will not run under 369 px a side, and the probe picture is 256x192.
        Refusing it would throw the candidate out for the size of the probe."""
        process = dc.load_onnx_restorer(darkening(self.root / "cand.onnx"),
                                        providers=("CPUExecutionProvider",))

        self.assertEqual(process(noise((64, 48))).size, (64, 48))


class TestTheModelIsShownAPictureItCanRun(unittest.TestCase):
    """What goes into the session, checked without a runtime: the padding, the scale of
    the numbers, and the crop that undoes the padding afterwards."""

    def test_a_small_picture_is_grown_to_the_minimum_side_on_both_axes(self):
        session = Session()
        process = dc.restorer_from_session(session)

        result = process(noise((64, 48)))

        shown = session.shown[0]
        self.assertEqual(shown.shape[:2], (1, 3))
        self.assertGreaterEqual(shown.shape[2], dc.MIN_SIDE)
        self.assertGreaterEqual(shown.shape[3], dc.MIN_SIDE)
        self.assertEqual(result.size, (64, 48))       # ...and the padding comes back off

    def test_a_frame_of_this_feature_s_population_is_shown_exactly_as_it_lies(self):
        """Everything above the 1024 px ceiling — which is the only population F183
        samples from — reaches the model untouched: no padding, no rescaling."""
        session = Session()

        dc.restorer_from_session(session)(noise((640, 480)))

        self.assertEqual(session.shown[0].shape, (1, 3, 480, 640))

    def test_the_pixels_arrive_as_the_export_expects_them_rgb_and_zero_to_one(self):
        session = Session()
        picture = Image.new("RGB", (400, 400), (255, 0, 0))

        dc.restorer_from_session(session)(picture)

        shown = session.shown[0][0]
        self.assertAlmostEqual(float(shown.max()), 1.0)
        self.assertAlmostEqual(float(shown.min()), 0.0)
        self.assertAlmostEqual(float(shown[0].mean()), 1.0)   # the red plane comes first

    def test_values_outside_the_range_are_clamped_and_never_wrapped_around(self):
        """A restoration model may answer with -0.2 or 1.4; eight-bit pixels may not, and
        a cast without a clamp turns an overshoot into a black speck."""
        process = dc.restorer_from_session(Session(lambda array: array * 4.0 - 1.0))

        result = np.asarray(process(Image.new("RGB", (400, 400), (128, 128, 128))))

        self.assertEqual(int(result.min()), 255)
        self.assertEqual(int(result.max()), 255)

    def test_the_padding_replicates_the_edge_and_leaves_a_big_frame_alone(self):
        small = np.zeros((4, 6, 3), dtype=np.float32)
        small[3, 5] = 1.0

        padded = dc.pad_to_minimum(small, minimum=8)

        self.assertEqual(padded.shape, (8, 8, 3))
        self.assertEqual(float(padded[7, 7, 0]), 1.0)   # the corner, replicated outward
        self.assertEqual(float(padded[0, 7, 0]), 0.0)
        big = np.zeros((10, 12, 3), dtype=np.float32)
        self.assertIs(dc.pad_to_minimum(big, minimum=8), big)


class TestARefusalIsASentenceAndNotAStackTrace(unittest.TestCase):
    """The caller prints whatever comes out of here as a row and moves to the next
    candidate (`measure_deblur.main`), so every failure has to be readable there."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_missing_weights_say_where_to_get_them(self):
        missing = self.root / "nafnet_deblur.onnx"

        with self.assertRaises(FileNotFoundError) as caught:
            dc.load_onnx_restorer(missing)

        message = str(caught.exception)
        self.assertIn(str(missing), message)
        self.assertIn(dc.WEIGHTS_URL, message)

    def test_a_file_that_is_not_a_graph_says_so_instead_of_raising_the_runtime_s_error(self):
        broken = self.root / "cand.onnx"
        broken.write_bytes(b"not a model")

        with self.assertRaises(RuntimeError) as caught:
            dc.load_onnx_restorer(broken, providers=("CPUExecutionProvider",))

        self.assertIn("ONNX", str(caught.exception))
        self.assertIn(str(broken), str(caught.exception))


class TestWhichLoaderTheNameOnTheCommandLineGoesTo(unittest.TestCase):
    """A dispatch on the extension, and deliberately nothing more: one candidate, not a
    table for models nobody has yet."""

    def test_a_path_to_weights_goes_to_the_loader_that_knows_the_format(self):
        self.assertIs(dc.loader_for("C:/AI/deblur/nafnet_deblur.onnx"),
                      dc.load_onnx_restorer)
        self.assertIs(dc.loader_for("weights.ONNX"), dc.load_onnx_restorer)

    def test_a_huggingface_name_still_goes_to_transformers_even_with_a_dot_in_it(self):
        self.assertIsNone(dc.loader_for("caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr"))
        self.assertIsNone(dc.loader_for("Qwen/Qwen2.5-VL-7B-Instruct"))

    def test_the_measurement_reaches_the_loader_and_its_refusal_reaches_the_caller(self):
        with self.assertRaises(FileNotFoundError):
            md.load_restorer(str(Path(tempfile.gettempdir()) / "no-such-weights.onnx"))


@unittest.skipIf(onnx is None, "onnx is not installed")
class TestNothingHereGoesToTheNetwork(unittest.TestCase):
    """The weights are a local file and the run must work offline — the ordinary state of
    this product. Checked by taking the sockets away rather than by reading the code."""

    def test_a_candidate_loads_and_runs_with_the_sockets_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            weights = darkening(Path(tmp) / "cand.onnx")
            def refuse(*_args, **_kwargs):
                raise AssertionError("the loader must not touch the network")

            with mock.patch.object(socket, "socket", refuse), \
                    mock.patch.object(socket, "create_connection", refuse):
                process = dc.load_onnx_restorer(weights,
                                                providers=("CPUExecutionProvider",))
                self.assertTrue(md.probe_one_to_one(process).usable)


if __name__ == "__main__":
    unittest.main()
