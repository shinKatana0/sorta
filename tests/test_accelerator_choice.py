"""F214/F220: which device the seven ML stages ask for, and what CUDA machines still get.

The feature is one function replacing seven copies of "cuda if available else cpu", and
the interesting part of it is not the new Apple rung — it is that Windows and Linux must
not be able to feel the change. So the CUDA case is checked BY EXHAUSTION over all seven
call sites (`SITES` below), against constants spelled out literally here rather than
imported from the code under test: a test that asks `accel` what `accel` decided would
pass on a rewrite that quietly moved every machine to a different device.

F220 added the last three — `detect`, `restore` and `search` kept their own line while
the four F214 converted moved on, which on a Mac meant three stages on the processor
beside four on the accelerator. The count in this file is not decoration: seven sites
walked here and seven modules walked by `TheChoiceLivesInOnePlaceTest` are what says the
decision is in one place rather than in one place and a few leftovers.

Nothing here needs a GPU, a Mac, or the weights. Each site is driven through the module
it really lives in, with `torch` / `onnxruntime` / `open_clip` / `insightface` /
`transformers` / `torchvision` faked at the import each stage does inside its own
function, so what is measured is the argument that reached the loader — the same thing
the F88 det_size tests measure.

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

from sorta import accel, detect, faces, junk, landmarks, naming, restore, search
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


class Placed:
    """A tensor, reduced to the one thing these tests measure: where it was put.

    `permute` returns itself and `to` returns a new one, the way torch does — so the
    device a frame ended up on is readable from the object the model was called with.
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = device

    def permute(self, *dims: int) -> "Placed":
        return self

    def to(self, device: str) -> "Placed":
        return Placed(device)


def fake_torch(machine: Machine) -> Any:
    """A torch module that answers about hardware and does the tensor moves we need."""
    torch = types.ModuleType("torch")
    torch.float16, torch.float32 = "float16", "float32"  # type: ignore[attr-defined]
    torch.cuda = types.SimpleNamespace(  # type: ignore[attr-defined]
        is_available=lambda: machine.cuda)
    torch.backends = types.SimpleNamespace(  # type: ignore[attr-defined]
        mps=types.SimpleNamespace(is_available=lambda: machine.mps))
    torch.no_grad = contextlib.nullcontext  # type: ignore[attr-defined]
    torch.from_numpy = lambda array: Placed()  # type: ignore[attr-defined]
    return torch


def refusal() -> NotImplementedError:
    """What MPS raises for an operator it has no Metal kernel for."""
    return NotImplementedError("The operator 'aten::_unique2' is not currently "
                               "implemented for the MPS device")


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


class RecordingDetector:
    """A torchvision detection model, recording where its weights and its frames went."""

    def __init__(self, refuse: int = 0) -> None:
        self.moved: list[str] = []
        self.frames: list[str] = []
        self._refuse = refuse

    def to(self, device: str) -> "RecordingDetector":
        self.moved.append(device)
        return self

    def eval(self) -> None:
        pass

    def __call__(self, tensors: list[Placed]) -> list[dict]:
        self.frames.append(tensors[0].device)
        if len(self.frames) <= self._refuse:
            raise refusal()
        return [{"labels": Column([17]), "scores": Column([0.9]),
                 "boxes": Column([[1.0, 2.0, 3.0, 4.0]])}]  # one cat


class Column:
    """A torch column of a prediction dict — `detect` only ever calls `.tolist()`."""

    def __init__(self, values: list) -> None:
        self._values = values

    def tolist(self) -> list:
        return self._values


DETECTOR_MODEL = "fasterrcnn_mobilenet_v3_large_fpn"


@contextlib.contextmanager
def detect_site(machine: Machine,
                refuse: int = 0) -> Iterator[tuple[RecordingDetector, Any]]:
    """Site 5 — the torchvision detector in `detect`, built inside the fake torch.

    The built detector is yielded rather than returned: its closure reads `torch` and
    `imaging` at CALL time, so a frame has to be run while the stubs are still installed.
    """
    model = RecordingDetector(refuse)
    detection = types.ModuleType("torchvision.models.detection")
    detection.__dict__[DETECTOR_MODEL] = lambda weights=None: model
    models = types.ModuleType("torchvision.models")
    models.detection = detection  # type: ignore[attr-defined]
    root = types.ModuleType("torchvision")
    root.models = models  # type: ignore[attr-defined]
    with modules(torch=fake_torch(machine), torchvision=root,
                 **{"torchvision.models": models,
                    "torchvision.models.detection": detection}):
        yield model, detect.torchvision_detector(DETECTOR_MODEL)


@contextlib.contextmanager
def a_frame() -> Iterator[str]:
    """A path that stats and decodes — the preview cache is not what is measured here."""
    with mock.patch("sorta.imaging.decode_rgb_preview",
                    lambda *args, **kwargs: Image.new("RGB", (4, 4))):
        yield str(Path(__file__))


class Reconstruction:
    """`.squeeze().float().cpu().clamp_(0, 1).numpy()` — the chain `restore` walks."""

    def __init__(self, array: np.ndarray) -> None:
        self._array = array

    def squeeze(self) -> "Reconstruction":
        return self

    def float(self) -> "Reconstruction":
        return self

    def cpu(self) -> "Reconstruction":
        return self

    def clamp_(self, low: float, high: float) -> "Reconstruction":
        return self

    def numpy(self) -> np.ndarray:
        return self._array


class RecordingUpscaler:
    """transformers' Swin2SR, recording where its weights went."""

    def __init__(self, refuse: int = 0) -> None:
        self.moved: list[str] = []
        self.calls = 0
        self._refuse = refuse

    def to(self, device: str) -> "RecordingUpscaler":
        self.moved.append(device)
        return self

    def eval(self) -> None:
        pass

    def __call__(self, **inputs: Any) -> Any:
        self.calls += 1
        if self.calls <= self._refuse:
            raise refusal()
        array = np.zeros((3, 4, 4), dtype=np.float32)
        return types.SimpleNamespace(
            reconstruction=types.SimpleNamespace(data=Reconstruction(array)))


class RecordingInputs(dict):
    """The processor's batch: a mapping (it is splatted) that records its `.to()`."""

    def __init__(self) -> None:
        super().__init__({"pixel_values": 0})
        self.devices: list[str] = []

    def to(self, device: str) -> "RecordingInputs":
        self.devices.append(device)
        return self


@contextlib.contextmanager
def restore_site(machine: Machine,
                 refuse: int = 0) -> Iterator[tuple[RecordingUpscaler,
                                                    RecordingInputs, Any]]:
    """Site 6 — the Swin2SR upscaler in `restore`, built inside the fake torch."""
    model, inputs = RecordingUpscaler(refuse), RecordingInputs()
    module = types.ModuleType("transformers")
    module.Swin2SRForImageSuperResolution = types.SimpleNamespace(  # type: ignore[attr-defined]
        from_pretrained=lambda name: model)
    module.AutoImageProcessor = types.SimpleNamespace(  # type: ignore[attr-defined]
        from_pretrained=lambda name: (lambda image, return_tensors: inputs))
    with modules(torch=fake_torch(machine), transformers=module):
        yield model, inputs, restore.load_swin2sr("caidas/swin2SR-test")


class Features:
    """A batch of text vectors: normalized in place, then handed over as numpy."""

    def norm(self, dim: int, keepdim: bool) -> float:
        return 1.0

    def __itruediv__(self, other: Any) -> "Features":
        return self

    def cpu(self) -> "Features":
        return self

    def numpy(self) -> np.ndarray:
        return np.zeros((1, 4), dtype=np.float32)


class Tokens:
    def __init__(self, device: str = "cpu") -> None:
        self.device = device

    def to(self, device: str) -> "Tokens":
        return Tokens(device)


class RecordingTextTower:
    """open_clip's model, as much of it as the search text encoder touches."""

    def __init__(self, refuse: int = 0) -> None:
        self.moved: list[str] = []
        self.encoded: list[str] = []
        self._refuse = refuse

    def to(self, device: str) -> "RecordingTextTower":
        self.moved.append(device)
        return self

    def eval(self) -> None:
        pass

    def encode_text(self, tokens: Tokens) -> Features:
        self.encoded.append(tokens.device)
        if len(self.encoded) <= self._refuse:
            raise refusal()
        return Features()


@contextlib.contextmanager
def search_site(machine: Machine,
                refuse: int = 0) -> Iterator[tuple[RecordingTextTower, Any]]:
    """Site 7 — the CLIP text tower in `search`, built inside the fake torch."""
    model = RecordingTextTower(refuse)
    module = types.ModuleType("open_clip")
    module.create_model_and_transforms = (  # type: ignore[attr-defined]
        lambda name, **kwargs: (model, None, None))
    module.get_tokenizer = lambda name: (lambda prompts: Tokens())  # type: ignore[attr-defined]
    with modules(torch=fake_torch(machine), open_clip=module):
        yield model, search.text_encoder(NamingSettings())


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


def detect_device(machine: Machine) -> Any:
    """Site 5 — the detector in `detect`: the device its weights were moved to."""
    with detect_site(machine) as (model, _detect):
        return model.moved[0]


def restore_device(machine: Machine) -> Any:
    """Site 6 — the upscaler in `restore`: the device its weights were moved to."""
    with restore_site(machine) as (model, _inputs, _upscale):
        return model.moved[0]


def search_device(machine: Machine) -> Any:
    """Site 7 — the text tower in `search`: the device open_clip was given."""
    with modules(torch=fake_torch(machine)), fake_open_clip() as loads:
        search.text_encoder(NamingSettings())
    return loads[0]["device"]


TORCH_SITES = [("naming: the VLM loader", naming_device),
               ("landmarks: the CLIP classifier", landmarks_device),
               # F220: the three that were still choosing for themselves.
               ("detect: the torchvision detector", detect_device),
               ("restore: the Swin2SR upscaler", restore_device),
               ("search: the CLIP text tower", search_device)]
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

    def test_all_seven_places_are_driven_here(self):
        """The count is the claim: seven stages asked, seven stages walked.

        A site that is added to `sorta` and not to this list is exactly the failure F220
        cleaned up after F214 — four converted, three forgotten, and nothing red.
        `TheChoiceLivesInOnePlaceTest` is the other half of the check and reads the
        source; this half says how many of them are actually run.
        """
        self.assertEqual(len(SITES), 7)

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

    def test_the_three_newest_sites_did_not_also_gain_a_dtype(self):
        """F220 moved the device choice and nothing else.

        The four stages F214 converted already had "float16 on CUDA, float32 otherwise";
        these three never did. Adding `accel.torch_dtype` beside the new device call
        would have been a change to the numbers a CUDA machine produces, dressed up as a
        tidy-up — half precision is a separate question and it comes with a measurement.
        """
        for name in ("detect.py", "restore.py", "search.py"):
            with self.subTest(module=name):
                code = (_ROOT / "sorta" / name).read_text(encoding="utf-8")
                self.assertNotIn("torch_dtype", code)
                self.assertNotIn("float16", code)

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

    F220 widened this from the four modules F214 converted to EVERY module under
    `sorta/`. The old list was a boundary drawn where one feature stopped: it said
    nothing about `detect`, `restore` and `search`, which kept their own line for three
    more features and would have kept it longer, and it would say nothing about the next
    stage that grows one. What is asserted now is the invariant itself — nobody but
    `accel` decides — so a fourth device (CoreML, ROCm, anything) is one edit again.
    """

    # The seven stages that ever had a device line of their own, named so that a rename
    # or a deletion cannot quietly shrink what the sweep below walks.
    STAGES = ("naming.py", "landmarks.py", "faces.py", "junk.py",
              "detect.py", "restore.py", "search.py")
    # The two modules allowed to name a device, and why each is allowed:
    #   accel.py        holds the decision — it IS the one place.
    #   diagnostics.py  ASKS about the hardware for a living. `sorta doctor` reports what
    #                   this machine has (torch's version, whether CUDA answers, which
    #                   providers onnxruntime offers), which is a question about the box
    #                   and not a choice about a stage. It also has to keep answering
    #                   when torch is broken enough that no stage could run at all, which
    #                   is why it reads the hardware directly rather than through `accel`.
    ALLOWED = ("accel.py", "diagnostics.py")
    # `torch.cuda.is_available()` and a hand-written provider list: the two spellings
    # of the decision. Comments are stripped first, because these files EXPLAIN the
    # decision and the explanation must not read as a violation of it.
    COPIES = (re.compile(r"cuda\s*\.\s*is_available"),
              re.compile(r"[\"']CUDAExecutionProvider[\"']"),
              re.compile(r"[\"']CoreMLExecutionProvider[\"']"))

    def code_of(self, path: Path) -> str:
        text = path.read_text(encoding="utf-8")
        return "\n".join(line.split("#")[0] for line in text.splitlines())

    def swept(self) -> list[Path]:
        """Every module of the package except the two that are allowed to ask."""
        return [path for path in sorted((_ROOT / "sorta").rglob("*.py"))
                if path.name not in self.ALLOWED]

    def test_no_module_decides_for_itself(self):
        for path in self.swept():
            code = self.code_of(path)
            where = path.relative_to(_ROOT).as_posix()
            for pattern in self.COPIES:
                with self.subTest(module=where, pattern=pattern.pattern):
                    self.assertIsNone(
                        pattern.search(code),
                        f"{where} decides on a device itself — accel is the one place")

    def test_all_seven_stages_are_inside_the_sweep(self):
        """The sweep is by directory, so this is what keeps it honest.

        `rglob` over a package finds nothing to complain about when a stage is renamed
        or moved, and a green gate would then mean "nothing was checked" rather than
        "nothing decides for itself".
        """
        swept = {path.name for path in self.swept()}
        for name in self.STAGES:
            with self.subTest(module=name):
                self.assertIn(name, swept)

    def test_every_stage_asks_the_one_place(self):
        """Not deciding is half of it; the other half is that the stage still asks."""
        asks = re.compile(r"accel\.(torch_device|onnx_providers|cuda_provider_available)")
        for name in self.STAGES:
            with self.subTest(module=name):
                self.assertRegex(self.code_of(_ROOT / "sorta" / name), asks)

    def test_the_one_place_is_the_module_that_holds_it(self):
        code = self.code_of(_ROOT / "sorta" / "accel.py")
        for pattern in self.COPIES:
            with self.subTest(pattern=pattern.pattern):
                self.assertIsNotNone(pattern.search(code))

    def test_the_excused_module_is_still_one_that_asks_rather_than_chooses(self):
        """`diagnostics` is excused BY NAME, so the excuse has to keep being true.

        An exception nobody rechecks is how a list of allowed files turns into a place
        to put things. If the day comes that `diagnostics` no longer reads the hardware
        itself, it stops needing to be here — and if it ever starts BUILDING a model on
        what it read, the excuse was wrong and this line should not have covered it.
        """
        code = self.code_of(_ROOT / "sorta" / "diagnostics.py")
        self.assertRegex(code, r"cuda\s*\.\s*is_available")
        self.assertNotIn(".to(device)", code)


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


class TheFrameGoesWhereTheWeightsWentTest(unittest.TestCase):
    """F220 requirement 4: one device, not two that happen to agree.

    `detect` read its device string twice — once to place the model and once to place
    every frame — and `restore` still does. Two reads of one variable are fine right up
    to the moment the variable can change under them, which is what a retreat to the CPU
    does: a model on the processor fed a tensor on the accelerator is a crash with a
    confusing message, halfway through a pass. So the detector places the frame on
    whatever device it is being run on rather than on the one the loader captured.
    """

    def test_the_detector_puts_its_frame_where_its_weights_are(self):
        for machine, expected in ((NVIDIA, "cuda"), (APPLE, "mps"), (PLAIN, "cpu")):
            with self.subTest(machine=machine.name):
                with detect_site(machine) as (model, detector), a_frame() as path:
                    found = detector(path)
                self.assertEqual(model.moved, [expected])
                self.assertEqual(model.frames, [expected])
                self.assertEqual([hit.label for hit in found], ["cat"])

    def test_the_upscaler_puts_its_pixels_where_its_weights_are(self):
        for machine, expected in ((NVIDIA, "cuda"), (APPLE, "mps"), (PLAIN, "cpu")):
            with self.subTest(machine=machine.name):
                with restore_site(machine) as (model, inputs, upscale):
                    upscale(Image.new("RGB", (4, 4)))
                self.assertEqual(model.moved, [expected])
                self.assertEqual(inputs.devices, [expected])


class WhichOfTheThreeRetreatsToTheCpuTest(unittest.TestCase):
    """F220: the fallback is a decision per site, not a wrapper applied everywhere.

    A swallowed exception where the exception mattered is a defect that surfaces a month
    later as "why is this suddenly slow", so each of the three was answered on its own:

        detect    wrapped. A cascade over thousands of candidates; a refusal at frame
                  900 throws away the 899 before it, and the frames left cost speed.
        search    wrapped. One phrase through a text tower is milliseconds on a
                  processor — the cheapest retreat there is, and without it a query
                  returns a traceback instead of results.
        restore   NOT wrapped. One frame per press of a button with a person waiting on
                  it, where the CPU path is minutes rather than seconds. A refusal here
                  is the answer (this model has no Metal kernel for this), the caller
                  already turns it into a reason a person can read, and nothing is lost:
                  the load is not cached and the next press retries.

    On CUDA none of this exists, which is the requirement F214 was built around.
    """

    def test_the_detector_finishes_the_pass_on_the_cpu(self):
        with detect_site(APPLE, refuse=1) as (model, detector), a_frame() as path:
            with self.assertLogs("sorta.accel", level=logging.WARNING):
                found = detector(path)
        self.assertEqual(model.frames, ["mps", "cpu"])
        self.assertEqual(model.moved, ["mps", "cpu"], "the weights follow the frame")
        self.assertEqual([hit.label for hit in found], ["cat"], "the frame still answers")

    def test_a_cuda_detector_fails_as_loudly_as_it_did_before(self):
        with detect_site(NVIDIA, refuse=1) as (model, detector), a_frame() as path:
            with self.assertRaises(NotImplementedError):
                detector(path)
        self.assertEqual(model.moved, ["cuda"], "nothing moved, nothing was caught")

    def test_the_text_tower_answers_the_query_on_the_cpu(self):
        with search_site(APPLE, refuse=1) as (model, encode):
            with self.assertLogs("sorta.accel", level=logging.WARNING):
                vectors = encode(["a cat on a roof"])
        self.assertEqual(model.encoded, ["mps", "cpu"])
        self.assertEqual(model.moved, ["cpu"])
        self.assertEqual(vectors.shape, (1, 4), "the query still has a vector")

    def test_a_cuda_text_tower_fails_as_loudly_as_it_did_before(self):
        with search_site(NVIDIA, refuse=1) as (model, encode):
            with self.assertRaises(NotImplementedError):
                encode(["a cat on a roof"])
        self.assertEqual(model.moved, [])

    def test_the_upscaler_says_no_rather_than_taking_four_minutes(self):
        """The one that is deliberately not wrapped — see the class docstring."""
        with restore_site(APPLE, refuse=1) as (model, _inputs, upscale):
            with self.assertNoLogs("sorta.accel", level=logging.WARNING):
                with self.assertRaises(NotImplementedError):
                    upscale(Image.new("RGB", (4, 4)))
        self.assertEqual(model.moved, ["mps"], "no retreat, so no second device")


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


class TheOnlyMacThisProjectHasIsARunnerTest(unittest.TestCase):
    """The macOS job in check.yml, checked as text because it cannot be checked by running.

    F214 was written without a push, so this job has never executed — that is stated in
    the workflow, in the changelog and in the report, and it is not something a test can
    fix. What a test CAN do is keep the job from being quietly dropped or reduced to an
    install that proves nothing: the three things it must do are asserted here, and a
    future edit that removes one fails the gate on the platforms that do run.
    """

    def workflow(self) -> dict:
        import yaml
        text = (_ROOT / ".github" / "workflows" / "check.yml").read_text(encoding="utf-8")
        return yaml.safe_load(text)

    def macos_job(self) -> dict:
        jobs = self.workflow()["jobs"]
        macos = [job for job in jobs.values()
                 if "macos" in str(job.get("runs-on", "")) or
                 "macos-latest" in str(job.get("strategy", ""))]
        self.assertTrue(macos, "no job runs on macOS — the Apple path has no hardware at all")
        return macos[0]

    def steps(self) -> str:
        return "\n".join(str(step.get("run", "")) for step in self.macos_job()["steps"])

    def test_the_install_is_the_cpu_profile(self):
        """`--extra gpu` is a CUDA profile: it does not install on a Mac and must not be
        asked for there."""
        run = self.steps()
        self.assertIn("--extra cpu", run)
        self.assertNotIn("--extra gpu", run)

    def test_the_machine_is_asked_what_it_offers(self):
        self.assertIn("probe_accelerator.py", self.steps())

    def test_the_suite_runs_there_too(self):
        self.assertIn("scripts/check.py", self.steps())

    def test_the_two_platforms_that_work_today_are_still_checked(self):
        matrix = self.workflow()["jobs"]["check"]["strategy"]["matrix"]["os"]
        self.assertIn("ubuntu-latest", matrix)
        self.assertIn("windows-latest", matrix)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
