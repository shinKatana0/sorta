"""F214: which device the four ML stages ask for, and what CUDA machines still get.

The feature is one function replacing four copies of "cuda if available else cpu", and
the interesting part of it is not the new Apple rung — it is that Windows and Linux must
not be able to feel the change. So the CUDA case is checked BY EXHAUSTION over all four
call sites (`SITES` below), against constants spelled out literally here rather than
imported from the code under test: a test that asks `accel` what `accel` decided would
pass on a rewrite that quietly moved every machine to a different device.

Nothing here needs a GPU, a Mac, or the weights. Each site is driven through the module
it really lives in, with `torch` / `onnxruntime` / `open_clip` / `insightface` faked at
the import each stage does inside its own function, so what is measured is the argument
that reached the loader — the same thing the F88 det_size tests measure.

What these tests CANNOT say: whether MPS and CoreML produce the same verdicts as the
CPU. That needs the hardware (see the macOS job in check.yml and
`scripts/probe_accelerator.py`), and until somebody runs it the answer is unmeasured
rather than good.
"""
from __future__ import annotations

import contextlib
import logging
import re
import sys
import types
import unittest
from pathlib import Path
from typing import Any, Iterator
from unittest import mock

import numpy as np
from PIL import Image

from sorta import accel, faces, junk, landmarks, naming
from sorta.config import NamingConfig as NamingSettings
from sorta.faces import FacesSettings

# The two answers that shipped before this feature, written out by hand. Every CUDA
# assertion below compares against THESE, not against anything the new code computes.
HISTORICAL_DEVICE = "cuda"
HISTORICAL_DTYPE = "float16"
HISTORICAL_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]

_ROOT = Path(__file__).resolve().parent.parent


# --- the machines the sites are run on -------------------------------------------

class Machine:
    """What a hardware environment answers to the two availability questions."""

    def __init__(self, name: str, *, cuda: bool = False, mps: bool = False,
                 providers: tuple[str, ...] = ()) -> None:
        self.name = name
        self.cuda = cuda
        self.mps = mps
        self.providers = providers

    def __repr__(self) -> str:  # pragma: no cover — subTest labels only
        return self.name


NVIDIA = Machine("windows/linux with CUDA", cuda=True,
                 providers=("TensorrtExecutionProvider", "CUDAExecutionProvider",
                            "CPUExecutionProvider"))
APPLE = Machine("apple silicon", mps=True,
                providers=("CoreMLExecutionProvider", "CPUExecutionProvider"))
PLAIN = Machine("no accelerator at all", providers=("CPUExecutionProvider",))
# The CPU profile on a Windows/Linux box: onnxruntime without CUDA compiled in. It has
# to keep making the exact call it makes today, hence its own entry.
CPU_PROFILE = Machine("windows/linux, cpu profile",
                      providers=("AzureExecutionProvider", "CPUExecutionProvider"))


def fake_torch(machine: Machine) -> Any:
    """A torch module that answers about hardware and does the tensor moves we need."""
    torch = types.ModuleType("torch")
    torch.float16, torch.float32 = "float16", "float32"  # type: ignore[attr-defined]
    torch.cuda = types.SimpleNamespace(  # type: ignore[attr-defined]
        is_available=lambda: machine.cuda)
    torch.backends = types.SimpleNamespace(  # type: ignore[attr-defined]
        mps=types.SimpleNamespace(is_available=lambda: machine.mps))
    torch.no_grad = contextlib.nullcontext  # type: ignore[attr-defined]
    return torch


def fake_onnxruntime(machine: Machine) -> Any:
    ort = types.ModuleType("onnxruntime")
    ort.get_available_providers = lambda: list(machine.providers)  # type: ignore[attr-defined]
    return ort


@contextlib.contextmanager
def modules(**stubs: Any) -> Iterator[None]:
    """Install stub modules for the duration of a call, and put the real ones back."""
    with mock.patch.dict(sys.modules, stubs):
        yield


# --- the four call sites ----------------------------------------------------------

class RecordingFaceAnalysis:
    """insightface's session, recording the providers it was built with."""

    built: list[dict] = []

    def __init__(self, **kwargs: Any) -> None:
        self.built.append(kwargs)
        self.models = {"landmark_2d_106": object()}

    def prepare(self, **kwargs: Any) -> None:
        pass

    def get(self, img: Any) -> list[Any]:  # pragma: no cover — not what is measured
        return []


@contextlib.contextmanager
def fake_insightface() -> Iterator[type[RecordingFaceAnalysis]]:
    RecordingFaceAnalysis.built = []
    common = types.ModuleType("insightface.app.common")
    common.Face = lambda **kwargs: types.SimpleNamespace(**kwargs)  # type: ignore[attr-defined]
    app_mod = types.ModuleType("insightface.app")
    app_mod.FaceAnalysis = RecordingFaceAnalysis  # type: ignore[attr-defined]
    app_mod.common = common  # type: ignore[attr-defined]
    root = types.ModuleType("insightface")
    root.app = app_mod  # type: ignore[attr-defined]
    with modules(**{"insightface": root, "insightface.app": app_mod,
                    "insightface.app.common": common}), \
            mock.patch("sorta.faces._enable_cuda_dll_dirs", lambda: None):
        yield RecordingFaceAnalysis


@contextlib.contextmanager
def fake_transformers() -> Iterator[None]:
    class Qwen2_5_VLForConditionalGeneration:  # noqa: N801 — the transformers name
        @staticmethod
        def from_pretrained(model_name: str, **kwargs: Any) -> Any:
            return types.SimpleNamespace(eval=lambda: None, loaded=kwargs)

    module = types.ModuleType("transformers")
    module.Qwen2_5_VLForConditionalGeneration = (  # type: ignore[attr-defined]
        Qwen2_5_VLForConditionalGeneration)
    module.AutoProcessor = types.SimpleNamespace(  # type: ignore[attr-defined]
        from_pretrained=lambda name, **kw: object())
    with modules(transformers=module):
        yield


@contextlib.contextmanager
def fake_open_clip() -> Iterator[list[dict]]:
    """open_clip, recording the `device=` every model load was asked for."""
    loads: list[dict] = []

    def create_model_and_transforms(name: str, **kwargs: Any) -> tuple[Any, Any, Any]:
        loads.append({"model": name, **kwargs})
        model = types.SimpleNamespace(eval=lambda: None, to=lambda dev: None)
        preprocess = types.SimpleNamespace(
            transforms=[types.SimpleNamespace(size=224)])
        return model, None, preprocess

    module = types.ModuleType("open_clip")
    module.create_model_and_transforms = create_model_and_transforms  # type: ignore[attr-defined]
    module.get_tokenizer = lambda name: (lambda prompts: prompts)  # type: ignore[attr-defined]
    with modules(open_clip=module):
        yield loads


def naming_device(machine: Machine) -> Any:
    """Site 1 — the VLM loader in `naming`: the device string it reports."""
    with modules(torch=fake_torch(machine)), fake_transformers():
        _model, _processor, device = naming.load_qwen("Qwen/test")
    return device


def naming_dtype(machine: Machine) -> Any:
    with modules(torch=fake_torch(machine)), fake_transformers():
        model, _processor, _device = naming.load_qwen("Qwen/test")
    return model.loaded["torch_dtype"]


def landmarks_device(machine: Machine) -> Any:
    """Site 2 — the CLIP classifier in `landmarks`: the device open_clip was given."""
    with modules(torch=fake_torch(machine)), fake_open_clip() as loads:
        landmarks.clip_classifier(NamingSettings())
    return loads[0]["device"]


def faces_providers(machine: Machine) -> Any:
    """Site 3 — the detection session in `faces`: the providers it was built with."""
    with modules(onnxruntime=fake_onnxruntime(machine)), fake_insightface() as analysis:
        faces._insightface_infer(FacesSettings())
    return analysis.built[0]["providers"]


def junk_providers(machine: Machine) -> Any:
    """Site 4 — the eyelid landmarks in `junk`: the providers it was built with."""
    with modules(onnxruntime=fake_onnxruntime(machine)), fake_insightface() as analysis:
        junk.insightface_eye_landmarks()
    return analysis.built[0]["providers"]


TORCH_SITES = [("naming: the VLM loader", naming_device),
               ("landmarks: the CLIP classifier", landmarks_device)]
ORT_SITES = [("faces: the detection session", faces_providers),
             ("junk: the eyelid landmarks", junk_providers)]
SITES = TORCH_SITES + ORT_SITES


class CudaMachinesChooseExactlyWhatTheyChoseBeforeTest(unittest.TestCase):
    """Requirement 1, and the reason this feature is allowed to ship at all.

    Every site, not the one the brief happened to name — a fifth stage picking its own
    device is precisely the failure the single function exists to prevent, and a test
    that checks one site would not see it.
    """

    def test_every_torch_site_still_says_cuda(self):
        for name, site in TORCH_SITES:
            with self.subTest(site=name):
                self.assertEqual(site(NVIDIA), HISTORICAL_DEVICE)

    def test_every_onnxruntime_site_still_asks_for_the_same_two_providers(self):
        for name, site in ORT_SITES:
            with self.subTest(site=name):
                self.assertEqual(list(site(NVIDIA)), HISTORICAL_PROVIDERS)

    def test_the_dtype_on_cuda_is_still_half_precision(self):
        """float16 on CUDA, float32 everywhere else — the rule that already shipped."""
        self.assertEqual(naming_dtype(NVIDIA), HISTORICAL_DTYPE)
        self.assertEqual(naming_dtype(PLAIN), "float32")

    def test_a_cpu_only_windows_box_makes_the_same_call_it_makes_today(self):
        """onnxruntime without CUDA compiled in: the list is unchanged, not tidied.

        It would read better to ask such a machine for `[CPUExecutionProvider]` alone.
        It would also be a change to a platform this feature is not allowed to change,
        so the historical pair is what goes out, and onnxruntime keeps doing what it
        already does with a provider it does not have.
        """
        for name, site in ORT_SITES:
            with self.subTest(site=name):
                self.assertEqual(list(site(CPU_PROFILE)), HISTORICAL_PROVIDERS)
                self.assertEqual(list(site(PLAIN)), HISTORICAL_PROVIDERS)

    def test_mps_is_never_reached_while_cuda_answers_yes(self):
        """A machine with both would still be a CUDA machine — order, not preference."""
        both = Machine("cuda and mps at once", cuda=True, mps=True)
        self.assertEqual(accel.torch_device(fake_torch(both)), HISTORICAL_DEVICE)


class AppleSiliconGetsTheAcceleratorTest(unittest.TestCase):
    """Requirement 2: the rung that did not exist, at every site."""

    def test_every_torch_site_asks_for_mps(self):
        for name, site in TORCH_SITES:
            with self.subTest(site=name):
                self.assertEqual(site(APPLE), "mps")

    def test_every_onnxruntime_site_asks_for_coreml_first(self):
        for name, site in ORT_SITES:
            with self.subTest(site=name):
                self.assertEqual(list(site(APPLE)),
                                 ["CoreMLExecutionProvider", "CPUExecutionProvider"])

    def test_without_either_accelerator_every_torch_site_says_cpu(self):
        for name, site in TORCH_SITES:
            with self.subTest(site=name):
                self.assertEqual(site(PLAIN), "cpu")

    def test_an_absent_backend_is_not_an_error(self):
        """A torch too old to know about MPS, and one whose Metal build is broken."""
        old = types.ModuleType("torch")
        old.cuda = types.SimpleNamespace(is_available=lambda: False)  # type: ignore[attr-defined]
        old.backends = types.SimpleNamespace()  # type: ignore[attr-defined]
        self.assertEqual(accel.torch_device(old), "cpu")

        def explode() -> bool:
            raise RuntimeError("Torch not compiled with MPS enabled")

        broken = fake_torch(APPLE)
        broken.backends.mps.is_available = explode
        self.assertEqual(accel.torch_device(broken), "cpu")

    def test_an_absent_onnxruntime_leaves_the_historical_list(self):
        exploding = types.ModuleType("onnxruntime")

        def explode() -> list[str]:
            raise OSError("DLL load failed")

        exploding.get_available_providers = explode  # type: ignore[attr-defined]
        self.assertEqual(accel.available_providers(exploding), ())
        self.assertEqual(accel.onnx_providers(exploding), HISTORICAL_PROVIDERS)


class TheChoiceLivesInOnePlaceTest(unittest.TestCase):
    """Requirement 4: a second copy of the logic fails the gate rather than the run.

    Read as text on purpose. A behavioural test passes just as happily when a stage
    grows its own copy that happens to agree today — the copies this feature removed
    agreed too, right up to the moment a third device existed.
    """

    OWNED = ("naming.py", "landmarks.py", "faces.py", "junk.py")
    # `torch.cuda.is_available()` and a hand-written provider list: the two spellings
    # of the decision. Comments are stripped first, because these files EXPLAIN the
    # decision and the explanation must not read as a violation of it.
    COPIES = (re.compile(r"cuda\s*\.\s*is_available"),
              re.compile(r"[\"']CUDAExecutionProvider[\"']"),
              re.compile(r"[\"']CoreMLExecutionProvider[\"']"))

    def code_of(self, name: str) -> str:
        text = (_ROOT / "sorta" / name).read_text(encoding="utf-8")
        return "\n".join(line.split("#")[0] for line in text.splitlines())

    def test_no_stage_decides_for_itself(self):
        for name in self.OWNED:
            code = self.code_of(name)
            for pattern in self.COPIES:
                with self.subTest(module=name, pattern=pattern.pattern):
                    self.assertIsNone(
                        pattern.search(code),
                        f"sorta/{name} decides on a device itself — accel is the one place")

    def test_the_one_place_is_the_module_that_holds_it(self):
        code = self.code_of("accel.py")
        for pattern in self.COPIES:
            with self.subTest(pattern=pattern.pattern):
                self.assertIsNotNone(pattern.search(code))


class TheAcceleratorMayRefuseTest(unittest.TestCase):
    """Requirement 3: a refusal costs speed, not the stage.

    MPS is not a smaller CUDA — an operator it has no kernel for raises at the first
    call, halfway into a run that has already spent an hour. What must happen then is
    what happens when weights are missing: a line in the log and a slower path.
    """

    def fallback(self, device: str = "mps") -> tuple[accel.CpuFallback, list[str]]:
        moved: list[str] = []
        return accel.CpuFallback(device, moved.append, what="test"), moved

    def refusing_once(self, exc: BaseException):
        seen: list[str] = []

        def work(device: str) -> str:
            seen.append(device)
            if len(seen) == 1:
                raise exc
            return device

        return work, seen

    def test_a_refused_operator_finishes_on_the_cpu(self):
        fallback, moved = self.fallback()
        work, seen = self.refusing_once(
            NotImplementedError("The operator 'aten::_unique2' is not currently "
                                "implemented for the MPS device"))
        with self.assertLogs("sorta.accel", level=logging.WARNING) as logs:
            self.assertEqual(fallback.run(work), "cpu")
        self.assertEqual(seen, ["mps", "cpu"])
        self.assertEqual(moved, ["cpu"], "the weights move too, not just the string")
        self.assertIn("CPU", "".join(logs.output))

    def test_the_retreat_is_one_way(self):
        """After a refusal the whole rest of the run is on the CPU, not retried."""
        fallback, _moved = self.fallback()
        with self.assertLogs("sorta.accel", level=logging.WARNING):
            fallback.run(self.refusing_once(NotImplementedError("mps"))[0])
        self.assertEqual(fallback.device, "cpu")
        self.assertEqual(fallback.run(lambda device: device), "cpu")

    def test_a_second_failure_is_a_real_failure(self):
        fallback, _moved = self.fallback()

        def always(device: str) -> str:
            raise NotImplementedError("not currently implemented for the MPS device")

        with self.assertLogs("sorta.accel", level=logging.WARNING):
            with self.assertRaises(NotImplementedError):
                fallback.run(always)

    def test_an_ordinary_bug_is_not_swallowed(self):
        """A stage that is simply wrong has to look wrong, on any device."""
        fallback, moved = self.fallback()
        with self.assertRaises(KeyError):
            fallback.run(self.refusing_once(KeyError("input_ids"))[0])
        self.assertEqual(moved, [])

    def test_cuda_never_retreats(self):
        """Requirement 1 again: a CUDA machine fails exactly as loudly as before.

        A CUDA stage that quietly continued at a tenth of the speed would be a worse
        outcome than the crash it hid — and it would be a change to a platform this
        feature is not allowed to change.
        """
        fallback, moved = self.fallback(device="cuda")
        with self.assertRaises(NotImplementedError):
            fallback.run(self.refusing_once(NotImplementedError("out of memory"))[0])
        self.assertEqual(moved, [])

    def test_what_counts_as_the_accelerator_refusing(self):
        refusals = [NotImplementedError("aten::foo"),
                    RuntimeError("MPS backend out of memory"),
                    RuntimeError("Metal kernel not found for this dtype"),
                    RuntimeError("the operator is not currently implemented")]
        for exc in refusals:
            with self.subTest(exc=str(exc)):
                self.assertTrue(accel.is_accelerator_failure(exc))
        for exc in [KeyError("k"), ValueError("bad shape"), RuntimeError("disk full")]:
            with self.subTest(exc=str(exc)):
                self.assertFalse(accel.is_accelerator_failure(exc))


class TheVlmRetreatsWithItsWeightsTest(unittest.TestCase):
    """The same requirement, through the runtime that actually generates answers."""

    class Model:
        """Refuses the first generate the way MPS refuses an operator it lacks."""

        def __init__(self) -> None:
            self.moved: list[str] = []
            self.calls = 0

        def to(self, device: str) -> None:
            self.moved.append(device)

        def generate(self, **kwargs: Any) -> Any:
            self.calls += 1
            if self.calls == 1:
                raise NotImplementedError(
                    "The operator 'aten::isin.Tensor_Tensor_out' is not currently "
                    "implemented for the MPS device")
            return np.concatenate([kwargs["input_ids"], kwargs["pixel_values"]], axis=1)

    class Batch(dict):
        def __init__(self, data: dict[str, Any]) -> None:
            super().__init__(data)
            self.devices: list[str] = []

        def to(self, device: str) -> Any:
            self.devices.append(device)
            return self

    class Processor:
        def apply_chat_template(self, messages: Any, tokenize: bool,
                                add_generation_prompt: bool) -> str:
            return "prompt"

        def __call__(self, text: list[str], images: list[Image.Image],
                     return_tensors: str, **kwargs: Any) -> Any:
            return TheVlmRetreatsWithItsWeightsTest.Batch({
                "input_ids": np.zeros((len(text), 2), dtype=int),
                "pixel_values": np.array([[7]] * len(text), dtype=int)})

        def batch_decode(self, gen_ids: Any, skip_special_tokens: bool = True):
            return [f"answer-{int(row[-1])}" for row in gen_ids]

    def test_a_refused_generate_is_answered_on_the_cpu(self):
        model, processor = self.Model(), self.Processor()
        with modules(torch=fake_torch(APPLE)):
            vlm = naming.qwen_runtime(model, processor, "mps")
            prepared = vlm.prepare([Image.new("RGB", (4, 4))], "what is this")
            with self.assertLogs("sorta.accel", level=logging.WARNING):
                answer = vlm.generate(prepared, 8)
        self.assertEqual(answer, "answer-7", "the answer survives the retreat")
        self.assertEqual(model.moved, ["cpu"], "the weights follow the device string")
        self.assertEqual(prepared.devices, ["mps", "cpu"])

    def test_a_cuda_runtime_is_untouched_by_any_of_this(self):
        model, processor = self.Model(), self.Processor()
        model.calls = 1  # the second generate answers, as a working card would
        with modules(torch=fake_torch(NVIDIA)):
            vlm = naming.qwen_runtime(model, processor, "cuda")
            prepared = vlm.prepare([Image.new("RGB", (4, 4))], "what is this")
            with self.assertNoLogs("sorta.accel", level=logging.WARNING):
                answer = vlm.generate(prepared, 8)
        self.assertEqual(answer, "answer-7")
        self.assertEqual(prepared.devices, ["cuda"])
        self.assertEqual(model.moved, [])


class InferenceWorkersStayWhereTheyWereTest(unittest.TestCase):
    """The faces stage sizes its session pool by the same question, in the same place.

    Four parallel sessions were measured on CUDA (F12.1). A Mac gets one, like every
    other non-CUDA machine, because how many Metal sessions buy anything is a
    measurement nobody has made — and this feature does not optimise by guessing.
    """

    def workers(self, machine: Machine) -> int:
        from sorta.config import Config
        with modules(onnxruntime=fake_onnxruntime(machine)):
            return faces._infer_workers(Config(sources=[Path("/photos")], raw={}))

    def test_cuda_still_gets_four(self):
        self.assertEqual(self.workers(NVIDIA), 4)

    def test_every_other_machine_still_gets_one(self):
        for machine in (APPLE, PLAIN, CPU_PROFILE):
            with self.subTest(machine=machine.name):
                self.assertEqual(self.workers(machine), 1)


class WhatThisMachineOffersTest(unittest.TestCase):
    """`describe` — the line the macOS CI step prints, since nobody here owns a Mac."""

    def test_it_names_the_device_and_both_provider_lists(self):
        line = accel.describe(fake_torch(APPLE), fake_onnxruntime(APPLE))
        self.assertIn("torch device: mps", line)
        self.assertIn("mps: yes", line)
        self.assertIn("CoreMLExecutionProvider", line)

    def test_a_cuda_machine_describes_itself_as_one(self):
        line = accel.describe(fake_torch(NVIDIA), fake_onnxruntime(NVIDIA))
        self.assertIn("torch device: cuda", line)
        self.assertIn("cuda: yes", line)

    def test_a_missing_torch_is_reported_rather_than_raised(self):
        broken = types.ModuleType("torch")
        broken.cuda = types.SimpleNamespace()  # type: ignore[attr-defined]
        with self.assertLogs("sorta.accel", level=logging.WARNING):
            line = accel.describe(broken, fake_onnxruntime(PLAIN))
        self.assertIn("torch device: ?", line)


class VerdictsAreComparedNotAssumedTest(unittest.TestCase):
    """The comparison this feature owes and cannot pay in full without the hardware.

    A device is the same class of change as an attention kernel, and that one moved
    7-11 verdicts out of 300. The arithmetic of the comparison is testable here; the
    numbers it would produce on Metal are not.
    """

    def test_identical_scores_agree(self):
        agree, gap = accel.verdicts_agree([0.1, 0.9], [0.1, 0.9])
        self.assertTrue(agree)
        self.assertEqual(gap, 0.0)

    def test_a_drift_past_the_tolerance_is_a_disagreement(self):
        agree, gap = accel.verdicts_agree([0.1, 0.9], [0.1, 0.94])
        self.assertFalse(agree)
        self.assertAlmostEqual(gap, 0.04)

    def test_rounding_below_the_tolerance_is_not(self):
        agree, gap = accel.verdicts_agree([0.5], [0.5001])
        self.assertTrue(agree)
        self.assertLess(gap, 1e-3)

    def test_different_lengths_never_quietly_agree(self):
        agree, gap = accel.verdicts_agree([0.5], [0.5, 0.5])
        self.assertFalse(agree)
        self.assertEqual(gap, float("inf"))

    def test_nothing_scored_is_not_a_disagreement(self):
        self.assertEqual(accel.verdicts_agree([], []), (True, 0.0))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
