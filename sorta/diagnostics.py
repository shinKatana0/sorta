"""F63/F65/F76/F230: say it out loud when the GPU or the geo data is not what it seems.

The `cpu`/`gpu` install profiles are mutually exclusive uv extras that ship the SAME
package name `torch` (indexes cu130 vs cpu), so any command with `--extra cpu` in a GPU
venv silently replaces `torch+cu130` with a CPU wheel. `onnxruntime-gpu` is a DIFFERENT
package name and survives, so `get_available_providers()` keeps reporting CUDA and faces
stay on the GPU — while CLIP and easyocr/CRAFT run on the CPU for hours with no signal.

F76 closed the blind spot of the F63 definition, which compared the two libraries with
each other and so read "both are CPU-only" as a legitimate CPU machine. It happened:
`onnxruntime` (pulled in by insightface) and `onnxruntime-gpu` unpack into the same
site-packages/onnxruntime/, so whichever lands last wins. Hence the separate `nvidia-smi`
probe — not torch, whose import costs ~4.5 s and whose install is what may be broken.

F65 is the same story by another mechanism: `sorta/data/geo/places.tsv` did not travel
into the wheel, and 15 955 files with honest GPS resolved to empty places without a
message. Only the file is looked at here — the base itself is never loaded.

F230: the repair commands are one per install kind (`uv sync --extra gpu --extra dev` is
nonsense on the copy the Windows installer made), the `nvidia-smi` query also brings back
the driver version, and the summary names the profile that actually WON.

Pure diagnostics: no DB, lazy imports, `nvidia-smi` at most once per call, never raises.
"""
from __future__ import annotations

import logging
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from . import install, launch

_LOG = logging.getLogger(__name__)

# Providers that mean "onnxruntime is set up for the GPU" — i.e. a GPU is expected.
_CUDA_PROVIDERS = ("CUDAExecutionProvider", "TensorrtExecutionProvider")

_NOT_INSTALLED = "not installed"

# The repair commands the warnings must name — a diagnosis without a fix sends people
# back to the docs. F230: one per install kind, chosen by `sorta.install`, because the
# single string that sat here was printed to the installed copy too, which has no project
# directory to `uv sync` in (met in a virtual machine on 2026-08-08).
_FIX_PROFILE: dict[str, str] = {
    install.KIND_CHECKOUT: "uv sync --extra gpu --extra dev",
    # The wizard installs the tier into the environment it finds itself in — what a tool
    # install and an installed copy both need. `--tiers gpu` keeps it one pasteable line.
    install.KIND_TOOL: "sorta-setup --tiers gpu",
    install.KIND_INSTALLED:
        "sorta-setup --tiers gpu (the Sorta setup item of the Start menu)",
}
# F76: pip has to be forced in a checkout — without `--no-deps` a reinstall drags the
# CPU-only `onnxruntime` back in as a dependency of insightface and overwrites the GPU
# binaries again. F230: the other two installs are not told to run `pip` at all. An
# installed copy is a `uv pip install --target` tree with no pip in it, and the wizard's
# acceleration tier already does the same repair with `--reinstall`.
_FIX_ORT: dict[str, str] = {
    install.KIND_CHECKOUT:
        "python -m pip install --force-reinstall --no-deps onnxruntime-gpu",
    install.KIND_TOOL: "sorta-setup --tiers gpu",
    install.KIND_INSTALLED:
        "sorta-setup --tiers gpu (the Sorta setup item of the Start menu)",
}


def fix_profile(kind: str | None = None) -> str:
    """How THIS install switches to the CUDA profile."""
    return _FIX_PROFILE[install.install_kind() if kind is None else kind]


def fix_ort(kind: str | None = None) -> str:
    """How THIS install repairs an onnxruntime that lost its CUDA provider."""
    return _FIX_ORT[install.install_kind() if kind is None else kind]


def _mismatch_problem(kind: str) -> str:
    return (
        "torch is a CPU-only build while onnxruntime offers CUDA — CLIP and OCR will run "
        "on the CPU and the GPU will sit idle, a large collection then takes hours. "
        f"Fix: {fix_profile(kind)} (any command with --extra cpu silently replaces "
        "torch+cuXXX with the CPU wheel)."
    )


def _torch_ignores_gpu_problem(kind: str) -> str:
    return (
        "an NVIDIA GPU is present but torch is a CPU-only build — the [cpu] profile is "
        "installed, so CLIP and OCR will run on the CPU. Fix: reinstall with the [gpu] "
        f"profile: {fix_profile(kind)}."
    )


def _ort_ignores_gpu_problem(kind: str) -> str:
    # The `--no-deps` warning belongs to the pip command and to nothing else: the other
    # two installs are told to run the wizard, which has no pip in the sentence.
    tail = (" — --no-deps is mandatory, without it pip pulls the CPU build back."
            if kind == install.KIND_CHECKOUT else ".")
    return (
        "an NVIDIA GPU is present but onnxruntime exposes no CUDA provider — face and "
        "junk detection will fall back to CPUExecutionProvider (insightface pulls the "
        "CPU-only onnxruntime into the same site-packages/onnxruntime/ as "
        f"onnxruntime-gpu, and whichever unpacks last wins). Fix: {fix_ort(kind)}{tail}"
    )


_PROBLEM_WARNING = "GPU setup problem: torch %s, onnxruntime providers: %s. %s"

# `nvidia-smi` ships with the driver, so it answers "is there a GPU here" whatever the
# wheels say. F230 added `driver_version` to the same query rather than a second call —
# the wizard has to know whether the driver is new enough for CUDA 13 before offering
# 2.5 GB of wheels, and one binary asked twice is how two answers start disagreeing.
_NVIDIA_SMI_CMD = ("nvidia-smi", "--query-gpu=name,driver_version",
                   "--format=csv,noheader")
# A machine with a half-installed driver can hang on this call — never wait longer.
_NVIDIA_SMI_TIMEOUT_S = 3.0

# The CUDA the `gpu` profile is built for (pyproject's pytorch-cu130 index) and the driver
# it needs. A CUDA major release has no forward compatibility: 13.0 requires r580 (580.65
# on Linux, 580.88 on Windows), and below that torch raises on import rather than running
# slowly. The floor is the MAJOR number on purpose — the minor differs between the two
# platforms, and refusing a card over a tenth of a version would be wrong more often.
CUDA_MAJOR = "13"
MIN_DRIVER_MAJOR = 580

# UNKNOWN is not OLD: `nvidia-smi` answered, so the card works and only its version could
# not be read — refusing the tier over that would leave an RTX machine on the CPU.
DRIVER_OK = "ok"
DRIVER_OLD = "old"
DRIVER_UNKNOWN = "unknown"

# F230: which profile an environment ended up on. `mixed` is the F76 state (one stack on
# CUDA, the other not), `unknown` has no torch at all; neither is a failure by itself.
PROFILE_GPU = "gpu"
PROFILE_CPU = "cpu"
PROFILE_MIXED = "mixed"
PROFILE_UNKNOWN = "unknown"
# What a CUDA wheel writes into its version: `2.13.0+cu130`.
_CUDA_BUILD_MARK = "+cu"

SmiRunner = Callable[[], "subprocess.CompletedProcess[str]"]


def _run_nvidia_smi() -> subprocess.CompletedProcess[str]:
    """Run the GPU query. Raises if the binary is missing or the call times out.

    F228: through `launch.run`, so a run started from the shortcut does not flash a
    console window at somebody who asked for a photo collection.
    """
    return launch.run(
        _NVIDIA_SMI_CMD,
        capture_output=True,
        text=True,
        timeout=_NVIDIA_SMI_TIMEOUT_S,
        check=False,
    )


@dataclass(frozen=True)
class NvidiaCard:
    """What `nvidia-smi` says about this machine: the card, and whether its driver fits.

    F230. `driver` is here because "the driver is too old" is a sentence with an action in
    it, unlike a traceback out of `import torch` two days later. `probed` is False on the
    default instance: "nothing was asked", which is not "there is no card".
    """

    name: str | None = None
    driver: str | None = None
    present: bool = False
    probed: bool = False

    @property
    def driver_major(self) -> int | None:
        """The leading number of `581.15`, or None when there is nothing to read."""
        head = (self.driver or "").strip().split(".")[0]
        try:
            return int(head)
        except ValueError:
            return None

    @property
    def driver_state(self) -> str:
        major = self.driver_major
        if major is None:
            return DRIVER_UNKNOWN
        return DRIVER_OK if major >= MIN_DRIVER_MAJOR else DRIVER_OLD

    @property
    def usable(self) -> bool:
        """Would the CUDA profile actually work here? The question the wizard asks."""
        return self.present and self.driver_state != DRIVER_OLD


def nvidia_card(run: SmiRunner | None = None) -> NvidiaCard:
    """The card of this machine, or a probed-and-absent answer. Never raises: anything but
    a successful call listing a GPU — no binary, non-zero exit, timeout, empty output, any
    exception — means "no GPU detected", within _NVIDIA_SMI_TIMEOUT_S."""
    absent = NvidiaCard(probed=True)
    try:
        completed = (run or _run_nvidia_smi)()
        if completed.returncode != 0:
            return absent
        first = (completed.stdout or "").strip().splitlines()
        if not first or not first[0].strip():
            return absent
        # `name, driver_version` as CSV; a runner faked with the name alone leaves the
        # driver unread rather than failing.
        parts = [part.strip() for part in first[0].split(",")]
        return NvidiaCard(name=parts[0] or None,
                          driver=parts[1] if len(parts) > 1 and parts[1] else None,
                          present=True, probed=True)
    except Exception:
        # FileNotFoundError — no driver at all; TimeoutExpired — a wedged one.
        return absent


def nvidia_gpu_present(run: SmiRunner | None = None) -> bool:
    """Does this machine physically have an NVIDIA GPU? Pure, cheap, torch-free — and the
    same probe `nvidia_card` reads (F230), so the start-up guard and the wizard cannot
    disagree about whether there is a card."""
    return nvidia_card(run).present


@dataclass
class GpuHealth:
    """The device state of the two stacks (torch, onnxruntime) and of the hardware.

    The derived fields are computed in __post_init__ rather than being properties, so they
    survive `dataclasses.asdict()` — the UI banner and `sorta doctor` serialise this.
    `mismatch` keeps its F63 meaning, the F76 cases have their own flags, and `degraded`
    is the one to branch on.
    """

    torch_version: str
    torch_cuda_available: bool
    torch_device_name: str | None
    ort_providers: tuple[str, ...]
    # False by default: a caller that cannot probe the hardware gets the pre-F76
    # behaviour (only `mismatch` can fire) rather than a false alarm.
    gpu_present: bool = False
    # F230: which install is going to READ the repair commands below. None asks
    # `sorta.install`; resolved to a string here so `asdict()` carries what was decided.
    install_kind: str | None = None
    ort_has_cuda: bool = field(init=False)
    mismatch: bool = field(init=False)
    torch_ignores_gpu: bool = field(init=False)
    ort_ignores_gpu: bool = field(init=False)
    degraded: bool = field(init=False)
    # F230: which of the two profiles this environment ended up on. They unpack into one
    # directory (F76), so after the wizard swaps them nothing else can say what won.
    profile: str = field(init=False)
    problems: tuple[str, ...] = field(init=False)
    summary: str = field(init=False)

    def __post_init__(self) -> None:
        if self.install_kind is None:
            self.install_kind = install.install_kind()
        self.ort_has_cuda = any(p in _CUDA_PROVIDERS for p in self.ort_providers)
        # F63: a GPU is expected (onnxruntime is on CUDA) but torch cannot see it.
        self.mismatch = self.ort_has_cuda and not self.torch_cuda_available
        # F76: the GPU is in the machine, yet a stack refuses to use it. Before this,
        # "both are CPU-only" looked legitimate and said nothing.
        self.torch_ignores_gpu = self.gpu_present and not self.torch_cuda_available
        self.ort_ignores_gpu = self.gpu_present and not self.ort_has_cuda
        self.profile = self._profile()
        problems: list[str] = []
        if self.mismatch:
            problems.append(_mismatch_problem(self.install_kind))
        elif self.torch_ignores_gpu:
            # Same conclusion as the mismatch above (torch is on the CPU), different
            # evidence — report one of them, not both.
            problems.append(_torch_ignores_gpu_problem(self.install_kind))
        if self.ort_ignores_gpu:
            problems.append(_ort_ignores_gpu_problem(self.install_kind))
        self.problems = tuple(problems)
        self.degraded = bool(self.problems)
        self.summary = self._summary()

    def _profile(self) -> str:
        """Which install profile won, read off the two packages that carry it. The torch
        BUILD and not `torch.cuda.is_available()`: a CUDA build with too old a driver
        still answers no, and the question is which wheels are installed."""
        if self.torch_version == _NOT_INSTALLED:
            return PROFILE_UNKNOWN
        torch_is_cuda = _CUDA_BUILD_MARK in self.torch_version
        if torch_is_cuda and self.ort_has_cuda:
            return PROFILE_GPU
        if not torch_is_cuda and not self.ort_has_cuda:
            return PROFILE_CPU
        return PROFILE_MIXED

    def _summary(self) -> str:
        device = self.torch_device_name or "-"
        providers = ", ".join(self.ort_providers) or "-"
        lines = [
            f"torch: {self.torch_version} "
            f"(CUDA available: {_yes_no(self.torch_cuda_available)}, device: {device})",
            f"onnxruntime providers: {providers} (CUDA: {_yes_no(self.ort_has_cuda)})",
            f"NVIDIA GPU in the machine (nvidia-smi): {_yes_no(self.gpu_present)}",
            # F230: the line the wizard sends a person to after it has REPLACED the
            # profile. The install kind is on it: it decides the repair commands below.
            f"install profile: {self.profile} (install: {self.install_kind})",
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


def gpu_health(*, gpu_present: bool | None = None,
               install_kind: str | None = None) -> GpuHealth:
    """Collect the device state of both stacks and of the hardware. Never raises.

    `gpu_present` overrides the nvidia-smi probe for a caller that already knows; by
    default it runs once, with a timeout. `install_kind` overrides F230's question.
    """
    version, cuda_available, device_name = _torch_state()
    return GpuHealth(
        torch_version=version,
        torch_cuda_available=cuda_available,
        torch_device_name=device_name,
        ort_providers=_ort_providers(),
        gpu_present=nvidia_gpu_present() if gpu_present is None else gpu_present,
        install_kind=install_kind,
    )


def warn_if_gpu_mismatch(
    health: GpuHealth | None = None, *, log: logging.Logger = _LOG
) -> bool:
    """Log exactly one warning if a GPU is present but a stack will not use it — the F63
    mismatch and both F76 cases, always with the repair command in it. Call it once from
    an entry point (`sorta run` / `sorta ui` startup), not inside a loop."""
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
    """Is the bundled `places.tsv` where the resolver looks for it? Paths are strings so
    `asdict()` stays JSON-serialisable for `sorta doctor` and the UI banner; the derived
    fields are computed in __post_init__ as in GpuHealth."""

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
    """Check the bundled geo data at `data_dir` (default — where the resolver looks). Only
    a stat() of places.tsv; the geodata import is lazy because it pulls numpy/scipy, which
    a diagnostics call on a broken install must not require."""
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
    """Log exactly one warning if the bundled geo data cannot be used. Like
    warn_if_gpu_mismatch: once from an entry point, not inside a loop."""
    if health is None:
        health = geo_data_health()
    if health.available:
        return False
    log.warning(_GEO_MISSING_WARNING, health.places_path, health.problem)
    return True


# --- F237: has this machine got the memory for a run? ----------------------------------

# What a run with the classification needs, in megabytes of FREE memory. Measured
# 2026-08-09, cpu profile: killed by the OOM killer on a 4 GB machine (Linux), finished on
# 6 GB (Windows) — the numbers the three guides state as the requirement. The floor is the
# size that DIED and not the size that passed: what is read here is free memory, always
# below the machine's total, so 6 GB would fire on the 6 GB machine a run is known to
# finish on — and a warning seen on every run stops being read (F233).
MEMORY_FLOOR_MB = 4000

_MEMINFO = Path("/proc/meminfo")
_MEM_AVAILABLE = "MemAvailable:"
_BYTES_PER_MB = 1024 * 1024


def _linux_available_mb() -> int | None:
    """`MemAvailable` of /proc/meminfo, in MB. None — no such line (kernels before 3.14).

    Not MemFree: on a machine that has been up for a day most of it is page cache, which
    a run may have back for the asking, and MemFree reads as an emergency on every one.
    """
    try:
        text = _MEMINFO.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith(_MEM_AVAILABLE):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1]) * 1024 // _BYTES_PER_MB  # the file counts in kB
            return None
    return None


def _windows_available_mb() -> int | None:
    """`ullAvailPhys` of GlobalMemoryStatusEx, in MB. None if the call fails.

    The structure is declared inside: `ctypes.windll` does not exist off Windows, and
    this module is imported on every platform.
    """
    try:
        import ctypes

        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return int(status.ullAvailPhys) // _BYTES_PER_MB
    except Exception:
        return None


def available_memory_mb() -> int | None:
    """Free memory in MB — or None on a platform this cannot ask without a new dependency.

    None is an answer and not a failure: macOS has no reading here (`vm_stat` counts
    pages of six kinds and the sum is not what MemAvailable means), and silence beats an
    invented number. Never raises.
    """
    if sys.platform.startswith("linux"):
        return _linux_available_mb()
    if sys.platform == "win32":
        return _windows_available_mb()
    return None


@dataclass(frozen=True)
class MemoryHealth:
    """Free memory against what a run with the classification needs. `available_mb` is
    None where the platform cannot be asked, and `low` is then False: a machine nobody
    could measure is left alone rather than warned at."""

    available_mb: int | None
    needed_mb: int = MEMORY_FLOOR_MB

    @property
    def low(self) -> bool:
        return self.available_mb is not None and self.available_mb < self.needed_mb


def memory_health(available_mb: int | None = None) -> MemoryHealth:
    """What this machine has free right now. `available_mb` overrides the probe for a
    caller that already asked. Never raises."""
    return MemoryHealth(
        available_memory_mb() if available_mb is None else available_mb)
