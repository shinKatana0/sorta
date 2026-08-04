"""F187 — ONE candidate for the question F183 asks. Not a way of adding models.

`scripts/measure_deblur.py` was written, gated and merged with nothing to run it on. It
loads a candidate through `AutoModelForImageToImage`, the way the rest of this project
loads a restoration model, and that path accepts the Swin2SR family alone — where every
published weight is x2 or x4. Those answer F169's question rather than this one, and the
script's own probe rejects them (`probe_one_to_one`), which is exactly what it is for.

WHAT WAS TRIED, 2026-08-04, and what came of it:

    google/maxim-s3-deblurring-gopro       no preprocessor_config.json — transformers
                                           cannot build a processor for it at all
    cstr/nafnet-sidd-GGUF                  GGUF (llama.cpp)
    litert-community/NAFNet-*              TFLite
    mlx-community/NAFNet-*, Restormer-*    MLX — Apple Silicon
    shinuh/tf-restormer-*                  a bare TensorFlow checkpoint
    nyanko7/nafnet-models (.pth)           the authors' weights, but a .pth is only
                                           numbers: reading it needs the authors'
                                           research code (basicsr) in the dependencies
    swz30/Restormer (.pth, 100 MB)         the same, one file of weights per task

    opencv/deblurring_nafnet               ONNX, MIT, 1:1, dynamic size  <- this one

The brief was written believing there is no ONNX deblurring model on HuggingFace. There
is one: `deblurring_nafnet_2025may.onnx` in the OpenCV Zoo mirror — NAFNet trained for
deblurring, exported with height and width left dynamic, 92 MB, MIT (megvii-model). That
settles the choice the brief left open in favour of the option it preferred: onnxruntime
is what this project already runs for faces, so the candidate arrives without a research
repository in the dependencies and — the point of the whole feature — leaves without one
if the measurement's answer turns out to be "it does not help".

THE WEIGHTS ARE NOT COMMITTED and there is no config key for them. The path is an
argument, like every other model here; the download happens once:

    curl -L -o C:/AI/deblur/nafnet_deblur.onnx \\
      https://huggingface.co/opencv/deblurring_nafnet/resolve/main/deblurring_nafnet_2025may.onnx
    python scripts/measure_deblur.py --models C:/AI/deblur/nafnet_deblur.onnx

NOTHING IN THE PRODUCT DEPENDS ON THIS FILE. No config key, no button, no stage — the
feature exists to make one question askable, and if the blind pairs say "not closer to
what was there", this module is deleted together with the question and nothing has to be
untangled from anywhere.

WHAT THIS LOADER MUST NOT DO is flatter the candidate. `measure_deblur.probe_one_to_one`
rejects a model that enlarges (that is F169's question) and one that returns its input
untouched (a null result dressed as "did no harm"), and both checks are only worth
anything if what they measure is the model's own behaviour. So the size of the result is
the model's own — the loader removes the padding it added itself and never resizes a
result to the shape the caller was hoping for (see `restorer_from_session`).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))  # the repo root — for `sorta`

from sorta import restore  # noqa: E402

RestoreFn = restore.UpscaleFn

# Where the candidate comes from, printed in the refusal below so that a person who has
# no weights on disk is not left to search for them.
WEIGHTS_URL = ("https://huggingface.co/opencv/deblurring_nafnet/resolve/main/"
               "deblurring_nafnet_2025may.onnx")

# The exported graph refuses a frame whose side is under 369 px: its channel attention
# was traced with a fixed pooling window, and on a smaller feature map the Pad node
# behind it fails ("cannot use 'edge' mode to pad dimension..."). Measured by bisection
# on the file itself, on both axes independently — 369 exactly, in either direction.
#
# 384 is that number rounded up, and the padding it triggers costs this measurement
# nothing: the population F183 samples from is frames ABOVE the 1024 px ceiling, so the
# only picture small enough to be padded is the 256x192 noise of `probe_one_to_one`.
# Without it the probe would report "не запустилась" and the candidate would be thrown
# out for being small rather than for being the wrong instrument.
MIN_SIDE = 384

# The face detector's order (`faces.py`), for the same reason: CUDA when the machine has
# it, CPU when it does not, and never a hard failure over a missing provider.
PROVIDERS = ("CUDAExecutionProvider", "CPUExecutionProvider")


def pad_to_minimum(array: np.ndarray, minimum: int = MIN_SIDE) -> np.ndarray:
    """`array` (H, W, C) grown at the right and bottom edges to at least `minimum` a side.

    Edge replication rather than zeros or reflection: a black border invents a hard edge
    that a deblurring model would then try to sharpen, and the result of that leaks back
    into the frame through the model's global pooling. Bottom-right only, so the mapping
    back to the original pixels is the trivial `[:h, :w]` crop and there is no offset to
    get wrong.
    """
    height, width = array.shape[0], array.shape[1]
    pad_h, pad_w = max(0, minimum - height), max(0, minimum - width)
    if not pad_h and not pad_w:
        return array
    return np.pad(array, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")


def restorer_from_session(session: Any, minimum: int = MIN_SIDE) -> RestoreFn:
    """An onnxruntime session -> the `restore.UpscaleFn` the measurement runs its arms as.

    The convention is the exported graph's own (`opencv/deblurring_nafnet`, demo.py):
    one NCHW input of RGB floats scaled to 0..1, one output of the same shape. The clamp
    on the way out is `restore.load_swin2sr`'s — a restoration model is free to return
    values outside the range, and 8-bit pixels are not.

    THE RESULT KEEPS THE MODEL'S OWN MULTIPLIER. What is removed on the way out is the
    padding this loader added on the way in and nothing else: the crop is taken at the
    RATIO the model returned, so a graph that doubles its input comes back doubled and
    `probe_one_to_one` sees x2 — the check that tells this feature's question apart from
    F169's. A loader that resized the output to match its input would silently turn every
    x4 model into a passing 1:1 candidate.
    """
    input_name = session.get_inputs()[0].name

    def process(image: Image.Image) -> Image.Image:
        rgb = image.convert("RGB")
        array = np.asarray(rgb, dtype=np.float32) / 255.0
        padded = pad_to_minimum(array, minimum)
        batch = np.ascontiguousarray(padded.transpose(2, 0, 1)[None])
        output = session.run(None, {input_name: batch})[0]
        result = np.moveaxis(np.asarray(output)[0], source=0, destination=-1)
        keep_h = max(1, round(rgb.height * result.shape[0] / padded.shape[0]))
        keep_w = max(1, round(rgb.width * result.shape[1] / padded.shape[1]))
        result = result[:keep_h, :keep_w]
        return Image.fromarray((np.clip(result, 0.0, 1.0) * 255.0).round().astype(np.uint8))

    return process


def load_onnx_restorer(weights: str | Path,
                       providers: tuple[str, ...] = PROVIDERS) -> RestoreFn:
    """Open the candidate's ONNX file and return the function that runs it.

    Every way this can fail becomes a sentence a person can act on rather than a stack
    trace, because the caller prints it as a row and goes on to the next candidate
    (`measure_deblur.main`): weights that are not on disk, a runtime that is not
    installed, a file that is not a graph.
    """
    path = Path(weights)
    if not path.is_file():
        raise FileNotFoundError(
            f"весов нет: {path}. Скачать один раз (92 МБ, MIT): curl -L -o "
            f"{path} {WEIGHTS_URL}")
    try:
        import onnxruntime
    except ImportError as exc:  # pragma: no cover — onnxruntime is in both hardware extras
        raise ImportError(
            "onnxruntime не установлен: поставьте профиль железа "
            "(uv sync --extra gpu или --extra cpu)") from exc
    _enable_cuda_dll_dirs()
    options = onnxruntime.SessionOptions()
    # The export carries initialisers no node uses, and onnxruntime says so about each of
    # them — hundreds of warning lines in front of a table that is meant to be read.
    options.log_severity_level = 3
    try:
        session = onnxruntime.InferenceSession(str(path), options,
                                               providers=list(providers))
    except Exception as exc:  # noqa: BLE001 — a broken file is an answer, not a crash
        raise RuntimeError(f"файл не читается как ONNX-граф: {path} "
                           f"({type(exc).__name__}: {exc})") from exc
    return restorer_from_session(_session_that_actually_runs(
        onnxruntime, session, str(path), options, providers))


def _session_that_actually_runs(onnxruntime: Any, session: Any, path: str,
                                options: Any, providers: tuple[str, ...]) -> Any:
    """`session`, or a CPU-only one when the graph loads on the GPU but cannot RUN there.

    Found on the only candidate that exists (`deblurring_nafnet_2025may.onnx`, 2026-08-05):
    it is a QUANTIZED export, and the CUDA provider has no kernel for its
    `DequantizeLinear`. Nothing announces that at load time — the session is built, the
    provider list looks right, and the failure arrives on the first frame:

        RUNTIME_EXCEPTION: Non-zero status code returned while running DequantizeLinear
        node. Name:'intro.weight_quantized_node'. Unsupported quantization type.

    So the fall-back onnxruntime does for an unsupported OPERATOR does not happen for an
    unsupported KERNEL of a supported one, and the candidate would be thrown out as
    broken. It is not broken: on the CPU it runs and returns its input's size.

    One tiny frame decides it, before any measurement: cheaper than discovering it on
    frame one of sixteen, and it cannot flatter the result — this only ever moves a
    candidate from "does not run" to "runs slowly", never the other way.
    """
    if "CUDAExecutionProvider" not in providers:
        return session
    try:
        name = session.get_inputs()[0].name
        session.run(None, {name: np.zeros((1, 3, MIN_SIDE, MIN_SIDE), dtype=np.float32)})
        return session
    except Exception:  # noqa: BLE001 — the point is which provider, not which exception
        return onnxruntime.InferenceSession(path, options,
                                            providers=["CPUExecutionProvider"])


def _enable_cuda_dll_dirs() -> None:  # pragma: no cover — Windows-specific, needs CUDA
    """CUDA from pip wheels, found the way `faces.py` finds it — imported, not copied.

    A second version of that search would drift from the one the face detector uses, and
    the failure it produces is silent: onnxruntime falls back to the CPU and the price
    table below prints the cost of a card that was never used.
    """
    from sorta import faces

    faces._enable_cuda_dll_dirs()


_LOADERS: dict[str, Callable[[str], RestoreFn]] = {".onnx": load_onnx_restorer}


def loader_for(model_name: str) -> Callable[[str], RestoreFn] | None:
    """The loader for `model_name`, or None when it is a HuggingFace name for the
    transformers path. One entry, deliberately: this is a dispatch on a file extension,
    not a plugin table for models nobody has yet."""
    return _LOADERS.get(Path(model_name).suffix.lower())
