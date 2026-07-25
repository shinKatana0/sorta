"""F63: GPU-health guard — do not stay silent when torch runs on the CPU.
F65: geo-data guard — the same for the bundled GeoNames files.

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

This module is a pure diagnostics layer: it touches no DB, imports torch/onnxruntime
(and geodata) lazily and never raises at the caller.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

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
