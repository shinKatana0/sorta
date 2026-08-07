"""F214: the one place that asks the machine what it can compute on.

Until this module the question was answered four times, in four stages, by four copies
of the same two lines — `"cuda" if torch.cuda.is_available() else "cpu"` in `naming`
and `landmarks`, `["CUDAExecutionProvider", "CPUExecutionProvider"]` in `faces` and
`junk`. Four copies of a two-branch decision survive exactly as long as there are two
branches; the third one (Apple Silicon) is what this feature adds, and adding it in
four places is how the stages start disagreeing about which device a run is on.

The preference order, and what "available" means for each rung:

    torch          CUDA -> MPS -> CPU
    onnxruntime    CUDA -> CoreML -> CPU

**Absence is not an error.** Every rung is probed, and a machine without any of them
lands on the CPU with nothing logged — that is the ordinary case on the runner this
project is tested on.

**Windows and Linux must not feel this.** That is the load-bearing requirement, not the
speed: both platforms that work today walk this path. So the rules below are written to
be identical to the code they replace wherever CUDA is in the picture:

* `torch_device` asks `torch.cuda.is_available()` FIRST and returns the same string it
  always did; MPS is only ever reached when that answer is False, which on Windows and
  Linux it is only on machines that were already running on the CPU;
* `onnx_providers` returns the historical `["CUDAExecutionProvider",
  "CPUExecutionProvider"]` in every case except one — a runtime that offers CoreML and
  does not offer CUDA, i.e. a Mac. A CPU-only Windows box keeps asking for CUDA and
  keeps being handed the CPU by onnxruntime, exactly as it does today, because a list
  that is *probably* equivalent is not the same thing as the same list;
* `CpuFallback` re-raises everything on CUDA. The retreat below exists for MPS, whose
  failure mode CUDA does not have, and a stage that dies on a CUDA machine has to die
  the way it died before this module existed.

None of this has run on real Apple hardware — there is no Mac here, and the macOS
runner in `check.yml` reaches CI only on the first push. What that runner cannot answer
either is whether MPS and CoreML give the SAME VERDICTS as the CPU: F105's
`vision-sdpa` moved 7-11 verdicts out of 300 on a question that was formally identical,
and a different device is the same class of change. `scripts/probe_accelerator.py` is
the part of that comparison a runner can do; the collection-sized part is unmeasured.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Callable, Sequence, TypeVar

_log = logging.getLogger(__name__)

CPU = "cpu"
CUDA = "cuda"
MPS = "mps"

CUDA_PROVIDER = "CUDAExecutionProvider"
COREML_PROVIDER = "CoreMLExecutionProvider"
CPU_PROVIDER = "CPUExecutionProvider"

# The list every onnxruntime session in this project has asked for since the CUDA
# profile shipped, kept as a constant so that "unchanged on Windows and Linux" is one
# object rather than four spellings of one.
CUDA_PROVIDERS = (CUDA_PROVIDER, CPU_PROVIDER)
COREML_PROVIDERS = (COREML_PROVIDER, CPU_PROVIDER)

T = TypeVar("T")


def _mps_available(torch: Any) -> bool:
    """Does this torch have a working Metal backend?

    Wrapped whole: `torch.backends.mps` is missing on old builds, and `is_available()`
    is free to raise on a build that was compiled without Metal. Both mean the same
    thing here — no MPS — and neither is a reason to take a stage down.
    """
    try:
        backend = getattr(torch.backends, "mps", None)
        return bool(backend is not None and backend.is_available())
    except Exception:
        return False


def _import_torch() -> Any:
    import torch

    return torch


def torch_device(torch: Any | None = None) -> str:
    """The device a torch stage runs on: CUDA -> MPS -> CPU.

    `torch` is taken as an argument because every caller has already imported it
    locally (the module must import without torch installed, which is why those imports
    are inside the functions), and because it is what lets the choice be tested without
    a GPU, a Mac, or a monkeypatched `sys.modules`.

    The CUDA branch is deliberately the untouched original line, exceptions and all: a
    `torch.cuda.is_available()` that raises used to take the stage down, and this
    feature is not the place to change what a broken CUDA install does.
    """
    torch = _import_torch() if torch is None else torch
    if torch.cuda.is_available():
        return CUDA
    if _mps_available(torch):
        return MPS
    return CPU


def torch_dtype(torch: Any, device: str) -> Any:
    """`float16` on CUDA, `float32` everywhere else — the rule that already shipped.

    MPS deliberately gets `float32`. Half precision on Metal is a different question
    from which device to run on, it would move verdicts, and nobody here can measure by
    how much (see the module docstring).
    """
    return torch.float16 if device == CUDA else torch.float32


def available_providers(onnxruntime: Any | None = None) -> tuple[str, ...]:
    """The execution providers this onnxruntime build offers, or () if it cannot say.

    An absent or broken onnxruntime is CPU semantics, the same reading `faces` has
    given it since the parallel-sessions work.
    """
    try:
        runtime = onnxruntime
        if runtime is None:
            import onnxruntime as installed

            runtime = installed
        return tuple(runtime.get_available_providers())
    except Exception:
        return ()


def cuda_provider_available(onnxruntime: Any | None = None) -> bool:
    """Is onnxruntime built with CUDA here? (The GPU profile installs onnxruntime-gpu.)

    Used to size the parallel inference sessions of the faces stage, which is a
    different question from which providers to ask for — a CoreML machine keeps the
    single-session default until somebody measures Metal, per this feature's rule
    against optimising for Apple by guesswork.
    """
    return CUDA_PROVIDER in available_providers(onnxruntime)


def onnx_providers(onnxruntime: Any | None = None) -> list[str]:
    """The provider list for an onnxruntime session: CUDA -> CoreML -> CPU.

    Three cases, and two of them answer with the historical pair:

        CUDA offered            [CUDA, CPU]     — the machines that work today
        CoreML offered, no CUDA [CoreML, CPU]   — a Mac
        neither, or no answer   [CUDA, CPU]     — unchanged; onnxruntime hands such a
                                                  session the CPU itself, as it does
                                                  on every CPU-only box right now

    The last case is the one worth stating out loud: it would be tidier to return
    `[CPU]` there, and tidier is not the requirement. A CPU-only Windows machine must
    make the same call it made before this module, byte for byte.
    """
    available = available_providers(onnxruntime)
    if CUDA_PROVIDER in available:
        return list(CUDA_PROVIDERS)
    if COREML_PROVIDER in available:
        return list(COREML_PROVIDERS)
    return list(CUDA_PROVIDERS)


# An accelerator saying "not this operation" rather than "your code is wrong". MPS
# raises NotImplementedError for an operator it has no kernel for ("The operator
# 'aten::...' is not currently implemented for the MPS device") and a RuntimeError that
# names the backend for the rest — out of memory, an unsupported dtype, a placement it
# will not do. Anything else is a bug in the stage and has to keep looking like one.
_MPS_FAILURE = re.compile(r"\bmps\b|\bmetal\b|not currently implemented", re.IGNORECASE)


def is_accelerator_failure(exc: BaseException) -> bool:
    """Is this the accelerator refusing the work, rather than the work being wrong?"""
    if isinstance(exc, NotImplementedError):
        return True
    return bool(_MPS_FAILURE.search(str(exc)))


class CpuFallback:
    """A stage's torch device, and its one-way retreat to the CPU.

    MPS is not a smaller CUDA. An operator a model needs may simply have no Metal
    kernel, and that shows up at the first call rather than at load time — so a device
    that was chosen successfully can still refuse the work halfway into a stage. The
    answer is the one every ML stage in this project already gives to missing weights:
    a logged step down to a slower path that finishes the run, not a traceback through
    the middle of it.

    On CUDA nothing is caught. A CUDA machine that raises has to raise exactly as it
    did before F214 — a stage quietly finishing on the CPU at a tenth of the speed is a
    worse outcome than the failure it hid, and "Windows and Linux feel nothing" is this
    feature's first requirement.

    The retreat happens once. `move` is what puts the caller's model on the CPU (the
    device string alone does not move weights), `device` becomes "cpu" for every call
    after it, and a second failure is a real failure and propagates.
    """

    def __init__(self, device: str, move: Callable[[str], None] | None = None,
                 *, what: str = "the stage") -> None:
        self.device = device
        self._move = move
        self._what = what

    def run(self, work: Callable[[str], T]) -> T:
        """Run `work(device)`, retreating to the CPU once if the accelerator refuses."""
        try:
            return work(self.device)
        except Exception as exc:
            if self.device in (CPU, CUDA) or not is_accelerator_failure(exc):
                raise
            _log.warning(
                "%s: the %s device refused an operation (%s: %s) — continuing on the "
                "CPU for the rest of this run",
                self._what, self.device, type(exc).__name__, exc)
            self.device = CPU
            if self._move is not None:
                self._move(CPU)
            return work(CPU)


def describe(torch: Any | None = None, onnxruntime: Any | None = None) -> str:
    """One line naming what this machine offers — what the macOS CI step prints.

    Deliberately a string and not a dataclass: its only readers are a person looking at
    a runner's log and the probe script beside it. `sorta doctor` is the place where
    this belongs for a user, and it is not this feature's to change.
    """
    try:
        torch = _import_torch() if torch is None else torch
        device = torch_device(torch)
        cuda = bool(torch.cuda.is_available())
        mps = _mps_available(torch)
    except Exception as exc:
        device, cuda, mps = "?", False, False
        _log.warning("accel: torch unavailable (%s)", exc)
    providers = onnx_providers(onnxruntime)
    offered = available_providers(onnxruntime)
    return (f"torch device: {device} (cuda: {'yes' if cuda else 'no'}, "
            f"mps: {'yes' if mps else 'no'}); "
            f"onnxruntime providers requested: {', '.join(providers)}; "
            f"offered: {', '.join(offered) or '-'}")


def verdicts_agree(left: Sequence[float], right: Sequence[float],
                   tolerance: float = 1e-3) -> tuple[bool, float]:
    """(agree, largest gap) between two runs of the same scores on two devices.

    The measurement this feature owes and cannot pay in full: a device is the same
    class of change as an attention kernel, and that one moved verdicts. This is the
    part a runner with no photo collection can still do — the same frames scored twice,
    on the CPU and on the accelerator, compared as numbers rather than trusted.
    """
    if len(left) != len(right):
        return False, float("inf")
    gap = max((abs(a - b) for a, b in zip(left, right)), default=0.0)
    return gap <= tolerance, gap
