"""F63: GPU-health guard — do not stay silent when torch runs on the CPU.
F65: geo-data guard — the same for the bundled GeoNames files.
F76: the same guard for the opposite direction — a machine that HAS an NVIDIA GPU but
a CPU-only stack installed (see the nvidia-smi probe and the problem list below).

The `cpu`/`gpu` install profiles are mutually exclusive uv extras, but both ship the
SAME package name `torch` (indexes cu130 vs cpu). Any command with `--extra cpu` in a
GPU venv silently reinstalls torch as a CPU wheel, evicting `torch+cu130`. Meanwhile
`onnxruntime-gpu` is a DIFFERENT package name (not `onnxruntime`), so it survives and
`get_available_providers()` keeps reporting CUDA — face detection stays on the GPU and
masks the regression. CLIP (open-clip) and easyocr/CRAFT then quietly run on the CPU:
a large collection takes hours with the GPU idle, without a single signal.

The geo part is the same story with another mechanism: `sorta/data/geo/places.tsv`
did not travel into the wheel, the resolver found nothing and returned empty places
for 15 955 files with honest GPS — without a single message (F65). Here we only look
at the file: is it there, where did we look, how big is it — the base itself is never
loaded.

F76 closes the blind spot of the F63 definition: it compared the two libraries with
each other, so "both are CPU-only" counted as a legitimate CPU machine. On a machine
WITH a GPU that is the most expensive case of all (a run silently takes hours), and it
happened for real: `pip` unpacks `onnxruntime` (pulled in by insightface) and
`onnxruntime-gpu` into the same site-packages/onnxruntime/ directory, so whichever
lands last wins and faces end up on the CPU by coin flip. The physical presence of a
GPU is therefore probed separately, via `nvidia-smi` — deliberately not via torch,
whose import costs ~4.5 s and whose own installation is exactly what may be broken.

This module is a pure diagnostics layer: it touches no DB, imports torch/onnxruntime
(and geodata) lazily, runs `nvidia-smi` at most once per call and never raises at the
caller.
"""
from __future__ import annotations

import logging
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

_LOG = logging.getLogger(__name__)

# Providers that mean "onnxruntime is set up for the GPU" — i.e. a GPU is expected.
_CUDA_PROVIDERS = ("CUDAExecutionProvider", "TensorrtExecutionProvider")

_NOT_INSTALLED = "not installed"

# The two repair commands the warnings must name — a diagnosis without a fix just
# sends the user back to the docs.
_FIX_PROFILE = "uv sync --extra gpu --extra dev"
# F76: pip has to be forced here. `onnxruntime` and `onnxruntime-gpu` unpack into the
# same site-packages/onnxruntime/, so a plain reinstall (without --no-deps) drags the
# CPU-only `onnxruntime` back in as a dependency of insightface and overwrites the
# GPU binaries again.
_FIX_ORT = "python -m pip install --force-reinstall --no-deps onnxruntime-gpu"

_MISMATCH_PROBLEM = (
    "torch is a CPU-only build while onnxruntime offers CUDA — CLIP and OCR will run "
    "on the CPU and the GPU will sit idle, a large collection then takes hours. "
    f"Fix: {_FIX_PROFILE} (any command with --extra cpu silently replaces torch+cuXXX "
    "with the CPU wheel)."
)
_TORCH_IGNORES_GPU_PROBLEM = (
    "an NVIDIA GPU is present but torch is a CPU-only build — the [cpu] profile is "
    "installed, so CLIP and OCR will run on the CPU. Fix: reinstall with the [gpu] "
    f"profile: {_FIX_PROFILE}."
)
_ORT_IGNORES_GPU_PROBLEM = (
    "an NVIDIA GPU is present but onnxruntime exposes no CUDA provider — face and junk "
    "detection will fall back to CPUExecutionProvider (insightface pulls the CPU-only "
    "onnxruntime into the same site-packages/onnxruntime/ as onnxruntime-gpu, and "
    f"whichever unpacks last wins). Fix: {_FIX_ORT} — --no-deps is mandatory, without "
    "it pip pulls the CPU build back."
)

_PROBLEM_WARNING = "GPU setup problem: torch %s, onnxruntime providers: %s. %s"

# `nvidia-smi` ships with the driver, so it answers "is there a GPU in this machine"
# independently of which wheels were installed. Deliberately not torch: importing it
# costs ~4.5 s and a broken torch install is exactly what we are diagnosing.
_NVIDIA_SMI_CMD = ("nvidia-smi", "--query-gpu=name", "--format=csv,noheader")
# A machine with a half-installed driver can hang on this call — never wait longer.
_NVIDIA_SMI_TIMEOUT_S = 3.0

SmiRunner = Callable[[], "subprocess.CompletedProcess[str]"]


def _run_nvidia_smi() -> subprocess.CompletedProcess[str]:
    """Run the GPU query. Raises if the binary is missing or the call times out."""
    return subprocess.run(
        _NVIDIA_SMI_CMD,
        capture_output=True,
        text=True,
        timeout=_NVIDIA_SMI_TIMEOUT_S,
        check=False,
    )


def nvidia_gpu_present(run: SmiRunner | None = None) -> bool:
    """Does this machine physically have an NVIDIA GPU? Pure, cheap, torch-free.

    `run` is injectable so tests never touch the real binary. Anything that is not a
    successful call listing at least one GPU — no binary, non-zero exit code, timeout,
    empty output, any other exception — means "no GPU detected": the probe never
    raises at the caller and never blocks longer than _NVIDIA_SMI_TIMEOUT_S.
    """
    try:
        completed = (run or _run_nvidia_smi)()
        if completed.returncode != 0:
            return False
        return bool((completed.stdout or "").strip())
    except Exception:
        # FileNotFoundError — no driver at all; TimeoutExpired — a wedged one.
        return False


@dataclass
class GpuHealth:
    """The device state of the two independent stacks (torch, onnxruntime) plus the
    hardware itself.

    `ort_has_cuda`, `mismatch`, the F76 flags, `problems` and `summary` are derived —
    computed in __post_init__ instead of being properties, so that they survive
    dataclasses.asdict() (the UI banner and `sorta doctor` serialise this) and can
    never go out of sync with the inputs.

    `mismatch` keeps its F63 meaning (torch on the CPU while onnxruntime is on CUDA);
    the F76 cases sit next to it in their own flags, and `degraded` is the single
    "something is wrong" answer callers should branch on.
    """

    torch_version: str
    torch_cuda_available: bool
    torch_device_name: str | None
    ort_providers: tuple[str, ...]
    # Defaults to False: a caller that cannot probe the hardware gets exactly the
    # pre-F76 behaviour (only `mismatch` can fire) instead of a false alarm.
    gpu_present: bool = False
    ort_has_cuda: bool = field(init=False)
    mismatch: bool = field(init=False)
    torch_ignores_gpu: bool = field(init=False)
    ort_ignores_gpu: bool = field(init=False)
    degraded: bool = field(init=False)
    problems: tuple[str, ...] = field(init=False)
    summary: str = field(init=False)

    def __post_init__(self) -> None:
        self.ort_has_cuda = any(p in _CUDA_PROVIDERS for p in self.ort_providers)
        # F63: a GPU is expected (onnxruntime is on CUDA) but torch cannot see it.
        self.mismatch = self.ort_has_cuda and not self.torch_cuda_available
        # F76: the GPU is right there in the machine, yet a stack refuses to use it.
        # This is what "both are CPU-only" looked like before — silence.
        self.torch_ignores_gpu = self.gpu_present and not self.torch_cuda_available
        self.ort_ignores_gpu = self.gpu_present and not self.ort_has_cuda
        problems: list[str] = []
        if self.mismatch:
            problems.append(_MISMATCH_PROBLEM)
        elif self.torch_ignores_gpu:
            # Same conclusion as the mismatch above (torch is on the CPU), different
            # evidence — report one of them, not both.
            problems.append(_TORCH_IGNORES_GPU_PROBLEM)
        if self.ort_ignores_gpu:
            problems.append(_ORT_IGNORES_GPU_PROBLEM)
        self.problems = tuple(problems)
        self.degraded = bool(self.problems)
        self.summary = self._summary()

    def _summary(self) -> str:
        device = self.torch_device_name or "-"
        providers = ", ".join(self.ort_providers) or "-"
        lines = [
            f"torch: {self.torch_version} "
            f"(CUDA available: {_yes_no(self.torch_cuda_available)}, device: {device})",
            f"onnxruntime providers: {providers} (CUDA: {_yes_no(self.ort_has_cuda)})",
            f"NVIDIA GPU in the machine (nvidia-smi): {_yes_no(self.gpu_present)}",
            f"mismatch: {'YES' if self.mismatch else 'no'}",
        ]
        lines.extend(f"PROBLEM: {problem}" for problem in self.problems)
        return "\n".join(lines)


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _torch_state() -> tuple[str, bool, str | None]:
    """(version, cuda_available, device_name) — safe values if torch is absent/broken."""
    try:
        import torch
    except Exception:
        return _NOT_INSTALLED, False, None

    try:
        version = str(torch.__version__)
    except Exception:
        version = "unknown"
    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        # A broken CUDA runtime raises here instead of returning False.
        return version, False, None

    device_name: str | None = None
    if cuda_available:
        try:
            device_name = str(torch.cuda.get_device_name(0))
        except Exception:
            device_name = None
    return version, cuda_available, device_name


def _ort_providers() -> tuple[str, ...]:
    """The onnxruntime providers; an empty tuple if it is absent/broken."""
    try:
        import onnxruntime

        return tuple(str(p) for p in onnxruntime.get_available_providers())
    except Exception:
        return ()


def gpu_health(*, gpu_present: bool | None = None) -> GpuHealth:
    """Collect the device state of both stacks and of the hardware. Never raises.

    `gpu_present` overrides the nvidia-smi probe (tests, and callers that already know
    the answer); by default the probe runs — once, with a timeout.
    """
    version, cuda_available, device_name = _torch_state()
    return GpuHealth(
        torch_version=version,
        torch_cuda_available=cuda_available,
        torch_device_name=device_name,
        ort_providers=_ort_providers(),
        gpu_present=nvidia_gpu_present() if gpu_present is None else gpu_present,
    )


def warn_if_gpu_mismatch(
    health: GpuHealth | None = None, *, log: logging.Logger = _LOG
) -> bool:
    """Log exactly one warning if a GPU is present but a stack will not use it.

    Covers the F63 mismatch (torch on the CPU, onnxruntime on CUDA) and both F76 cases
    (a GPU in the machine while torch and/or onnxruntime are CPU-only). The warning
    always names the repair command. Returns True if a problem was reported. Call it
    once from an entry point (`sorta run` / `sorta ui` startup) — not inside a loop.
    """
    if health is None:
        health = gpu_health()
    if not health.degraded:
        return False
    log.warning(
        _PROBLEM_WARNING,
        health.torch_version,
        ", ".join(health.ort_providers) or "-",
        " ".join(health.problems),
    )
    return True


# --- F65: the bundled geo data ---------------------------------------------------

_PLACES_FILE = "places.tsv"

_GEO_FIX_HINT = (
    "Rebuild the bundled data with `python scripts/build_geodata.py` or reinstall "
    "sorta — until then every coordinate resolves to an empty place."
)

_GEO_MISSING_WARNING = "geo data unusable: %s (%s). " + _GEO_FIX_HINT


@dataclass
class GeoDataHealth:
    """Is the bundled `places.tsv` where the resolver looks for it?

    Paths are strings (like the other fields here) so that dataclasses.asdict() of
    this stays JSON-serialisable for `sorta doctor`/the UI banner. `size_bytes` is
    None when the file is absent; `available`/`places_path`/`summary` are derived in
    __post_init__ for the same reason as in GpuHealth — they cannot drift.
    """

    data_dir: str
    size_bytes: int | None
    available: bool = field(init=False)
    places_path: str = field(init=False)
    problem: str | None = field(init=False)
    summary: str = field(init=False)

    def __post_init__(self) -> None:
        self.places_path = str(Path(self.data_dir) / _PLACES_FILE)
        # A zero-byte file (an interrupted build/checkout) is just as unusable as none.
        if self.size_bytes is None:
            self.problem = "file not found"
        elif self.size_bytes == 0:
            self.problem = "file is empty"
        else:
            self.problem = None
        self.available = self.problem is None
        self.summary = self._summary()

    def _summary(self) -> str:
        if self.problem is not None:
            return f"geo data: {self.places_path} — {self.problem.upper()}. {_GEO_FIX_HINT}"
        size_mb = (self.size_bytes or 0) / (1024 * 1024)
        return f"geo data: {self.places_path} ({size_mb:.1f} MB)"


def _file_size(path: Path) -> int | None:
    """Size of the file in bytes; None — it is absent/unreadable. Reads no content."""
    try:
        return path.stat().st_size
    except OSError:
        return None


def geo_data_health(data_dir: str | Path | None = None) -> GeoDataHealth:
    """Check the bundled geo data at `data_dir` (default — where the resolver looks).

    Pure: only a stat() of places.tsv, no network, no DB, and the base is not loaded.
    The geodata import is lazy — it pulls numpy/scipy, which a diagnostics call on a
    broken install should not require.
    """
    if data_dir is None:
        from .geodata import GeoResolver

        data_dir = GeoResolver().data_dir
    directory = Path(data_dir)
    return GeoDataHealth(
        data_dir=str(directory), size_bytes=_file_size(directory / _PLACES_FILE),
    )


def warn_if_geo_data_missing(
    health: GeoDataHealth | None = None, *, log: logging.Logger = _LOG
) -> bool:
    """Log exactly one warning if the bundled geo data cannot be used.

    Returns True if the problem was reported. Like warn_if_gpu_mismatch: call it once
    from an entry point, not inside a loop.
    """
    if health is None:
        health = geo_data_health()
    if health.available:
        return False
    log.warning(_GEO_MISSING_WARNING, health.places_path, health.problem)
    return True
