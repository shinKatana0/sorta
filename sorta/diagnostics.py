"""F63: GPU-health guard — do not stay silent when torch runs on the CPU.

The `cpu`/`gpu` install profiles are mutually exclusive uv extras, but both ship the
SAME package name `torch` (indexes cu130 vs cpu). Any command with `--extra cpu` in a
GPU venv silently reinstalls torch as a CPU wheel, evicting `torch+cu130`. Meanwhile
`onnxruntime-gpu` is a DIFFERENT package name (not `onnxruntime`), so it survives and
`get_available_providers()` keeps reporting CUDA — face detection stays on the GPU and
masks the regression. CLIP (open-clip) and easyocr/CRAFT then quietly run on the CPU:
a large collection takes hours with the GPU idle, without a single signal.

This module is a pure diagnostics layer: it touches no DB, imports torch/onnxruntime
lazily (like faces with insightface) and never raises at the caller.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

_LOG = logging.getLogger(__name__)

# Providers that mean "onnxruntime is set up for the GPU" — i.e. a GPU is expected.
_CUDA_PROVIDERS = ("CUDAExecutionProvider", "TensorrtExecutionProvider")

_NOT_INSTALLED = "not installed"

_MISMATCH_WARNING = (
    "GPU mismatch: torch is a CPU-only build (%s) while onnxruntime offers CUDA "
    "(providers: %s). CLIP and OCR will run on the CPU and the GPU will sit idle — "
    "a large collection then takes hours. Fix: uv sync --extra gpu --extra dev "
    "(any command with --extra cpu silently replaces torch+cuXXX with the CPU wheel)."
)


@dataclass
class GpuHealth:
    """The device state of the two independent stacks: torch and onnxruntime.

    `ort_has_cuda`, `mismatch` and `summary` are derived — computed in __post_init__
    instead of being properties, so that they survive dataclasses.asdict() (the UI
    banner and `sorta doctor` serialise this) and can never go out of sync with the
    inputs.
    """

    torch_version: str
    torch_cuda_available: bool
    torch_device_name: str | None
    ort_providers: tuple[str, ...]
    ort_has_cuda: bool = field(init=False)
    mismatch: bool = field(init=False)
    summary: str = field(init=False)

    def __post_init__(self) -> None:
        self.ort_has_cuda = any(p in _CUDA_PROVIDERS for p in self.ort_providers)
        # A GPU is expected (onnxruntime is on CUDA) but torch cannot see it.
        # Pure CPU machines have neither — that is a legitimate setup, not a mismatch.
        self.mismatch = self.ort_has_cuda and not self.torch_cuda_available
        self.summary = self._summary()

    def _summary(self) -> str:
        device = self.torch_device_name or "-"
        providers = ", ".join(self.ort_providers) or "-"
        lines = [
            f"torch: {self.torch_version} "
            f"(CUDA available: {_yes_no(self.torch_cuda_available)}, device: {device})",
            f"onnxruntime providers: {providers} (CUDA: {_yes_no(self.ort_has_cuda)})",
        ]
        if self.mismatch:
            lines.append(
                "mismatch: YES — torch is a CPU-only build while onnxruntime runs on "
                "CUDA; CLIP and OCR fall back to the CPU. "
                "Fix: uv sync --extra gpu --extra dev"
            )
        else:
            lines.append("mismatch: no")
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


def gpu_health() -> GpuHealth:
    """Collect the device state of both stacks. Never raises at the caller."""
    version, cuda_available, device_name = _torch_state()
    return GpuHealth(
        torch_version=version,
        torch_cuda_available=cuda_available,
        torch_device_name=device_name,
        ort_providers=_ort_providers(),
    )


def warn_if_gpu_mismatch(
    health: GpuHealth | None = None, *, log: logging.Logger = _LOG
) -> bool:
    """Log exactly one warning if torch is CPU-only while a GPU is expected.

    Returns True if the mismatch was reported. Call it once from an entry point
    (`sorta run` / `sorta ui` startup) — not inside a loop.
    """
    if health is None:
        health = gpu_health()
    if not health.mismatch:
        return False
    log.warning(_MISMATCH_WARNING, health.torch_version, ", ".join(health.ort_providers))
    return True
