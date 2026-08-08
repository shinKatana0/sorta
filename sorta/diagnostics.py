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

F230 changes two things here and adds one. The repair commands are no longer a single
string printed to everybody — `uv sync --extra gpu --extra dev` is true in a checkout and
nonsense on the copy the Windows installer made, which has no project directory — so there
is one per install kind and `sorta.install` decides which. The `nvidia-smi` query now also
brings back the DRIVER VERSION, because the setup wizard has to say "your driver is older
than CUDA 13 needs" instead of installing 2.5 GB of wheels that will not import. And the
summary gained one line: which install profile actually WON, which after the wizard replaces
one profile with the other is a question nothing else in the product can answer (F76 —
`onnxruntime` and `onnxruntime-gpu` unpack into the same directory).

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

from . import install, launch

_LOG = logging.getLogger(__name__)

# Providers that mean "onnxruntime is set up for the GPU" — i.e. a GPU is expected.
_CUDA_PROVIDERS = ("CUDAExecutionProvider", "TensorrtExecutionProvider")

_NOT_INSTALLED = "not installed"

# The repair commands the warnings must name — a diagnosis without a fix just sends the
# user back to the docs.
#
# F230: and it has to be the fix for the install that is READING it. One string sat here
# and was printed to everybody, including the copy the Windows installer made, which has
# no project directory to `uv sync` in and no `dev` extra to ask for. The owner met it in
# a virtual machine on 2026-08-08. So there is one command per install kind, and
# `sorta.install` decides which one — the same single answer `doctor` and the wizard use.
_FIX_PROFILE: dict[str, str] = {
    install.KIND_CHECKOUT: "uv sync --extra gpu --extra dev",
    # The wizard installs the tier into the environment it finds itself in, which is what
    # a tool install and an installed copy both need; `--tiers gpu` is the non-interactive
    # form, so the sentence stays one pasteable command.
    install.KIND_TOOL: "sorta-setup --tiers gpu",
    install.KIND_INSTALLED:
        "sorta-setup --tiers gpu (the Sorta setup item of the Start menu)",
}
# F76: pip has to be forced in a checkout. `onnxruntime` and `onnxruntime-gpu` unpack
# into the same site-packages/onnxruntime/, so a plain reinstall (without --no-deps)
# drags the CPU-only `onnxruntime` back in as a dependency of insightface and overwrites
# the GPU binaries again.
#
# F230: the other two installs are not told to run `pip` at all. An installed copy is a
# `uv pip install --target` tree with no pip in it, and the wizard's acceleration tier
# already installs `onnxruntime-gpu` with `--reinstall` — which is the same repair,
# performed by the one thing on that machine that knows where the packages live.
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


def _mismatch_problem(kind: str | None) -> str:
    return (
        "torch is a CPU-only build while onnxruntime offers CUDA — CLIP and OCR will run "
        "on the CPU and the GPU will sit idle, a large collection then takes hours. "
        f"Fix: {fix_profile(kind)} (any command with --extra cpu silently replaces "
        "torch+cuXXX with the CPU wheel)."
    )


def _torch_ignores_gpu_problem(kind: str | None) -> str:
    return (
        "an NVIDIA GPU is present but torch is a CPU-only build — the [cpu] profile is "
        "installed, so CLIP and OCR will run on the CPU. Fix: reinstall with the [gpu] "
        f"profile: {fix_profile(kind)}."
    )


def _ort_ignores_gpu_problem(kind: str | None) -> str:
    return (
        "an NVIDIA GPU is present but onnxruntime exposes no CUDA provider — face and "
        "junk detection will fall back to CPUExecutionProvider (insightface pulls the "
        "CPU-only onnxruntime into the same site-packages/onnxruntime/ as "
        f"onnxruntime-gpu, and whichever unpacks last wins). Fix: {fix_ort(kind)}"
        + (" — --no-deps is mandatory, without it pip pulls the CPU build back."
           if (kind or install.install_kind()) == install.KIND_CHECKOUT else ".")
    )


_PROBLEM_WARNING = "GPU setup problem: torch %s, onnxruntime providers: %s. %s"

# `nvidia-smi` ships with the driver, so it answers "is there a GPU in this machine"
# independently of which wheels were installed. Deliberately not torch: importing it
# costs ~4.5 s and a broken torch install is exactly what we are diagnosing.
#
# F230 added `driver_version` to the same query rather than a second call: the wizard has
# to say whether the driver is new enough for CUDA 13 BEFORE offering 2.5 GB of wheels
# that would not import, and asking the same binary twice is how two answers about one
# machine start disagreeing.
_NVIDIA_SMI_CMD = ("nvidia-smi", "--query-gpu=name,driver_version",
                   "--format=csv,noheader")
# A machine with a half-installed driver can hang on this call — never wait longer.
_NVIDIA_SMI_TIMEOUT_S = 3.0

# The CUDA the `gpu` profile is built for (pyproject's pytorch-cu130 index), and the
# driver that CUDA needs. A CUDA major release has no forward compatibility on an older
# driver: 13.0 requires r580 (580.65 on Linux, 580.88 on Windows), so a machine with r5xx
# below that gets an import error out of torch rather than slow work. The floor is the
# major number on purpose — the minor differs between the two platforms, and a wizard
# that refused a card over a tenth of a version would be wrong more often than right.
CUDA_MAJOR = "13"
MIN_DRIVER_MAJOR = 580

# The three things that can be true of a driver, kept apart because they need different
# sentences. UNKNOWN is not OLD: `nvidia-smi` answered, so a driver is installed and the
# card works — only its version could not be read, and refusing the tier over that would
# leave an RTX machine on the CPU for a parsing failure.
DRIVER_OK = "ok"
DRIVER_OLD = "old"
DRIVER_UNKNOWN = "unknown"

# F230: which profile an environment ended up on, by the packages in it. `mixed` is the
# F76 state (one stack on CUDA, the other not) and `unknown` is an environment with no
# torch at all — neither is a failure by itself, and both are worth their own word.
PROFILE_GPU = "gpu"
PROFILE_CPU = "cpu"
PROFILE_MIXED = "mixed"
PROFILE_UNKNOWN = "unknown"
# What a CUDA wheel writes into its version: `2.13.0+cu130`.
_CUDA_BUILD_MARK = "+cu"

SmiRunner = Callable[[], "subprocess.CompletedProcess[str]"]


def _run_nvidia_smi() -> subprocess.CompletedProcess[str]:
    """Run the GPU query. Raises if the binary is missing or the call times out.

    F228: through `launch.run`, so the start-up probe of a run started from the shortcut
    does not flash a console window at somebody who asked for a photo collection.
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

    F230. `present` is the F76 question, unchanged. `driver` is the new half — the wizard
    may not offer 2.5 GB of CUDA 13 wheels to a machine whose driver cannot load them, and
    "the driver is too old" is a sentence with an action in it (update it), while a
    traceback out of `import torch` two days later is not.

    `probed` is False for the default instance, which is how a caller says "nothing was
    asked about the hardware" — different from "there is no card". The wizard behaves
    exactly as it did before F230 in that state instead of announcing an absence it never
    checked.
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
    """The card of this machine, or a probed-and-absent answer. Never raises.

    `run` is injectable so tests never touch the real binary. Anything that is not a
    successful call listing at least one GPU — no binary, non-zero exit code, timeout,
    empty output, any other exception — means "no GPU detected": the probe never
    raises at the caller and never blocks longer than _NVIDIA_SMI_TIMEOUT_S.
    """
    absent = NvidiaCard(probed=True)
    try:
        completed = (run or _run_nvidia_smi)()
        if completed.returncode != 0:
            return absent
        first = (completed.stdout or "").strip().splitlines()
        if not first or not first[0].strip():
            return absent
        # `name, driver_version` as CSV; a runner faked with the name alone (or a
        # `nvidia-smi` that does not know the field) simply leaves the driver unread.
        parts = [part.strip() for part in first[0].split(",")]
        return NvidiaCard(name=parts[0] or None,
                          driver=parts[1] if len(parts) > 1 and parts[1] else None,
                          present=True, probed=True)
    except Exception:
        # FileNotFoundError — no driver at all; TimeoutExpired — a wedged one.
        return absent


def nvidia_gpu_present(run: SmiRunner | None = None) -> bool:
    """Does this machine physically have an NVIDIA GPU? Pure, cheap, torch-free.

    One probe behind two questions (F230): `nvidia_card` reads the same line, so the
    start-up guard and the wizard can never disagree about whether there is a card.
    """
    return nvidia_card(run).present


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
    # F230: which install is going to READ the repair commands below. None asks
    # `sorta.install` — the answer is about this process and costs a couple of `is_file`
    # calls, and it is resolved to a string here so that `dataclasses.asdict()` (the UI
    # banner, `sorta doctor`, the run log) carries what was actually decided.
    install_kind: str | None = None
    ort_has_cuda: bool = field(init=False)
    mismatch: bool = field(init=False)
    torch_ignores_gpu: bool = field(init=False)
    ort_ignores_gpu: bool = field(init=False)
    degraded: bool = field(init=False)
    # F230: which of the two mutually exclusive profiles this environment actually ended
    # up on. `onnxruntime` and `onnxruntime-gpu` unpack into one directory (F76), so after
    # the wizard replaces the profile nobody but this line can say what won — which is
    # exactly what the wizard now sends people here to read.
    profile: str = field(init=False)
    problems: tuple[str, ...] = field(init=False)
    summary: str = field(init=False)

    def __post_init__(self) -> None:
        if self.install_kind is None:
            self.install_kind = install.install_kind()
        self.ort_has_cuda = any(p in _CUDA_PROVIDERS for p in self.ort_providers)
        # F63: a GPU is expected (onnxruntime is on CUDA) but torch cannot see it.
        self.mismatch = self.ort_has_cuda and not self.torch_cuda_available
        # F76: the GPU is right there in the machine, yet a stack refuses to use it.
        # This is what "both are CPU-only" looked like before — silence.
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
        """Which install profile won, read off the two packages that carry it.

        The torch BUILD and not `torch.cuda.is_available()`: a CUDA build on a machine
        whose driver is too old still answers no to that question, and the question here is
        which wheels are installed, not whether they are running. onnxruntime is the other
        half and can differ from it — that is the F76 trap, and `mixed` is the honest word
        for it rather than a coin flip between the two names.
        """
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
            # profile. It names the install kind too, because that is what decides which
            # repair command the problems below are worded with.
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

    `gpu_present` overrides the nvidia-smi probe (tests, and callers that already know
    the answer); by default the probe runs — once, with a timeout. `install_kind` is the
    same kind of override for F230's question about which install is reading this.
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
